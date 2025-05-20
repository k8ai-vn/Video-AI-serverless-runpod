import os
import time
import json
import uuid
import random
import string
import logging
import datetime
import multiprocessing
from typing import Optional, List

import torch
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from diffusers.utils import export_to_video
from fastvideo.models.hunyuan_hf.modeling_hunyuan import HunyuanVideoTransformer3DModel
from transformers import BitsAndBytesConfig

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fastvideo.models.hunyuan_hf.pipeline_hunyuan import HunyuanVideoPipeline

# Define network storage paths
NETWORK_STORAGE_PATH = os.environ.get('NETWORK_STORAGE', '/workspace')
OUTPUT_PATH = os.path.join(NETWORK_STORAGE_PATH, 'outputs')
S3_BUCKET = 'ttv-storage'
S3_ACCESS_KEY = os.environ.get('S3_ACCESS_KEY')
print('S3_ACCESS_KEY', S3_ACCESS_KEY)
S3_SECRET_KEY = os.environ.get('S3_SECRET_KEY')
MODEL_PATH = os.environ.get('MODEL_PATH', '/workspace/data/FastHunyuan-diffusers')

# Create directories if they don't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Initialize S3 client
bucket_url = 'https://eu2.contabostorage.com/'
s3_client = boto3.client('s3',
                endpoint_url=bucket_url,
                aws_access_key_id=S3_ACCESS_KEY,
                aws_secret_access_key=S3_SECRET_KEY,
                config=Config(
                    request_checksum_calculation="when_required",
                    response_checksum_validation="when_required"
                )
            )

def upload_file(file_name, user_uuid, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    object_name = file_name.split('/')[-1]
    print('object_name', object_name)
    object_name = user_uuid + '/' + object_name
    print('object_name', object_name)
    response = s3_client.upload_file(file_name, bucket, object_name)
    
    # Create a presigned URL for the file
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': object_name},
        ExpiresIn=3600  # URL will be valid for 1 hour
    )   
    print(presigned_url)
    # delete file from local
    os.remove(file_name)
        
    return True

# Initialize the pipeline
pipeline = None
def initialize_pipeline(quantization="nf4"):
    global pipeline
    if pipeline is None:
        try:
            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            # Use bfloat16 for compatibility with FlashAttention
            weight_dtype = torch.bfloat16
            
            print(f"Loading model from: {MODEL_PATH}")
            
            # Check if model path exists
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model path {MODEL_PATH} does not exist")
            
            # Load transformer model with appropriate quantization
            if quantization == "nf4":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    llm_int8_skip_modules=["proj_out", "norm_out"]
                )
                transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                    MODEL_PATH,
                    subfolder="transformer/",
                    torch_dtype=weight_dtype,
                    quantization_config=quantization_config
                )
            elif quantization == "int8":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True, 
                    llm_int8_skip_modules=["proj_out", "norm_out"]
                )
                transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                    MODEL_PATH,
                    subfolder="transformer/",
                    torch_dtype=weight_dtype,
                    quantization_config=quantization_config
                )
            else:  # No quantization
                transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                    MODEL_PATH,
                    subfolder="transformer/",
                    torch_dtype=weight_dtype
                ).to(device)
            
            print("Max vram for read transformer:", round(torch.cuda.max_memory_allocated(device="cuda") / 1024**3, 3), "GiB")
            torch.cuda.reset_max_memory_allocated(device)
            
            # Initialize pipeline with the loaded transformer
            pipeline = HunyuanVideoPipeline.from_pretrained(
                MODEL_PATH, 
                transformer=transformer,
                torch_dtype=weight_dtype,
                local_files_only=True
            )
            torch.cuda.reset_max_memory_allocated(device)

            # Enable VAE tiling for better memory usage
            pipeline.enable_vae_tiling()
            
            # Set flow shift parameter
            pipeline.scheduler._shift = 17  # Default flow shift

            # Enable CPU offload
            pipeline.enable_model_cpu_offload()
            
            pipeline.enable_sequential_cpu_offload()
            print("Max vram for init pipeline:", round(torch.cuda.max_memory_allocated(device="cuda") / 1024**3, 3), "GiB")
            print("Pipeline initialized successfully")
        except Exception as e:
            print(f"Error initializing pipeline: {str(e)}")
            raise
    
    return pipeline

# Initialize FastAPI app
app = FastAPI(title="Hunyuan Video Generation API")

# Define request model
class VideoGenerationRequest(BaseModel):
    prompt: str = "A beautiful sunset over a calm ocean, with gentle waves."
    num_frames: int = 45
    num_inference_steps: int = 6
    guidance_scale: float = 1.0
    embedded_cfg_scale: float = 6.0
    width: int = 1280
    height: int = 720
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    user_uuid: str = "system_default"
    output_path: Optional[str] = None
    fps: int = 24
    flow_shift: int = 17
    quantization: str = "fp16"  # Options: fp16 and bf16, or None/empty string for no quantization
    video_length: Optional[float] = None  # New parameter for video length in seconds

# Set the start method for multiprocessing to 'spawn' to avoid CUDA re-initialization issues
multiprocessing.set_start_method('spawn', force=True)

# Define the video generation task function at module level (not nested)
def generate_video_task(prompt, output_path, video_path, video_file_name, user_uuid, 
                        height, width, num_frames, num_inference_steps, guidance_scale, 
                        negative_prompt, seed, flow_shift, fps, quantization, video_length=None):
    try:
        # Initialize the pipeline with specified quantization
        pipeline = initialize_pipeline(quantization)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if pipeline is None:
            raise ValueError("Failed to initialize pipeline")
        
        # Set up generator for reproducibility
        generator = torch.Generator("cpu").manual_seed(seed if seed is not None else torch.seed())
        torch.cuda.reset_max_memory_allocated(device)
        
        # Set flow shift parameter
        pipeline.scheduler._shift = flow_shift
        
        # Adjust num_frames based on video_length if provided
        if video_length is not None:
            num_frames = min(int(video_length * fps), 24)
            print(f"Adjusted num_frames to {num_frames} based on video_length of {video_length} seconds at {fps} fps")
        
        # Generate the video with torch.autocast for better performance
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Generate the video
            start_time = time.perf_counter()
            output = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).frames[0]
        
        # Check if output is None
        if output is None:
            raise ValueError("Pipeline returned None for output frames")
            
        # Export the video
        export_to_video(output, video_path, fps=fps)
        print(f"Video generated at: {video_path}")
        print("Time:", round(time.perf_counter() - start_time, 2), "seconds")
        print("Max vram for denoise:", round(torch.cuda.max_memory_allocated(device="cuda") / 1024**3, 3), "GiB")

        # Upload the video to S3
        upload_file(video_path, 
                    user_uuid, 
                    S3_BUCKET, 
                    video_file_name)
        print(f"Uploaded video to s3://{S3_BUCKET}/{user_uuid}/{video_file_name}")
    except Exception as e:
        print(f"Error in background task: {str(e)}")
        # Log the full traceback for better debugging
        import traceback
        traceback.print_exc()

@app.post("/generate-video")
async def generate_video(request: VideoGenerationRequest):
    try:
        print(f"Worker Start")
        
        # Initialize the pipeline if not already done
        pipeline = initialize_pipeline(request.quantization)

        # Define output path
        output_path = request.output_path or os.path.join(OUTPUT_PATH, 'my_videos/')
        os.makedirs(output_path, exist_ok=True)
        
        # Generate date time in DDMMYYYYHHMMSS format and random string
        current_time = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        video_file_name = f"{current_time}_{random_string}.mp4"
        video_path = os.path.join(output_path, video_file_name)
        
        # Create a task ID for tracking
        task_id = str(uuid.uuid4())
        
        # If video_length is provided, calculate the actual number of frames to be used
        actual_num_frames = request.num_frames
        if request.video_length is not None:
            actual_num_frames = int(request.video_length * request.fps)
            print(f"Using video_length parameter: {request.video_length} seconds at {request.fps} fps = {actual_num_frames} frames")
        
        # Start the background process
        process = multiprocessing.Process(
            target=generate_video_task,
            args=(
                request.prompt,
                output_path,
                video_path,
                video_file_name,
                request.user_uuid,
                request.height,
                request.width,
                actual_num_frames,  # Use the calculated number of frames
                request.num_inference_steps,
                request.guidance_scale,
                request.negative_prompt,
                request.seed,
                request.flow_shift,
                request.fps,
                request.quantization,
                request.video_length
            )
        )
        process.daemon = True  # Set as daemon so it doesn't block program exit
        process.start()
        
        # Return immediately with task ID
        s3_url = f"s3://{S3_BUCKET}/{request.user_uuid}/{video_file_name}"
        return {
            "output": {
                "status": "processing",
                "task_id": task_id,
                "prompt": request.prompt,
                "expected_video_path": video_path,
                "expected_s3_url": s3_url,
                "parameters": {
                    "num_frames": actual_num_frames,  # Return the actual number of frames
                    "width": request.width,
                    "height": request.height,
                    "num_inference_steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "embedded_cfg_scale": request.embedded_cfg_scale,
                    "flow_shift": request.flow_shift,
                    "seed": request.seed,
                    "fps": request.fps,
                    "quantization": request.quantization,
                    "video_length": request.video_length  # Include the video_length in the response
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Start the FastAPI server
if __name__ == '__main__':
    # Initialize the pipeline once at startup
    initialize_pipeline()
    # Start the FastAPI server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)