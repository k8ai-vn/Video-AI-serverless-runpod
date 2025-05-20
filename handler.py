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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define network storage paths
NETWORK_STORAGE_PATH = os.environ.get('NETWORK_STORAGE', '/workspace')
OUTPUT_PATH = os.path.join(NETWORK_STORAGE_PATH, 'outputs')
S3_BUCKET = 'ttv-storage'
S3_ACCESS_KEY = os.environ.get('S3_ACCESS_KEY')
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
    """Upload a file to an S3 bucket and clean up"""
    try:
        if object_name is None:
            object_name = os.path.basename(file_name)
        object_name = f"{user_uuid}/{object_name}"
        logging.info(f"Uploading file to s3://{bucket}/{object_name}")
        s3_client.upload_file(file_name, bucket, object_name)
        
        # Create a presigned URL for the file
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': object_name},
            ExpiresIn=3600
        )
        logging.info(f"Generated presigned URL: {presigned_url}")
        
        # Delete local file
        os.remove(file_name)
        logging.info(f"Deleted local file: {file_name}")
        return True
    except ClientError as e:
        logging.error(f"Failed to upload file to S3: {str(e)}")
        return False

# Initialize the pipeline
pipeline = None
def initialize_pipeline(quantization="nf4"):
    global pipeline
    if pipeline is None:
        try:
            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            weight_dtype = torch.bfloat16
            logging.info(f"Loading model from: {MODEL_PATH} on device: {device}")
            
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model path {MODEL_PATH} does not exist")
            
            # Load transformer model with quantization
            if quantization == "nf4":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,  # Enable double quantization
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
            else:
                transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                    MODEL_PATH,
                    subfolder="transformer/",
                    torch_dtype=weight_dtype
                ).to(device)
            
            logging.info(f"Max VRAM after loading transformer: {torch.cuda.max_memory_allocated(device='cuda') / 1024**3:.3f} GiB")
            torch.cuda.reset_max_memory_allocated(device)
            
            # Initialize pipeline
            pipeline = HunyuanVideoPipeline.from_pretrained(
                MODEL_PATH,
                transformer=transformer,
                torch_dtype=weight_dtype,
                local_files_only=True
            )
            
            # Enable VAE tiling with optimized tile size
            pipeline.scheduler._shift = 17
            
            # Enable sequential CPU offloading
            pipeline.enable_sequential_cpu_offload()
            
            logging.info(f"Max VRAM after pipeline init: {torch.cuda.max_memory_allocated(device='cuda') / 1024**3:.3f} GiB")
            torch.cuda.reset_max_memory_allocated(device)
            logging.info("Pipeline initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing pipeline: {str(e)}")
            raise
    return pipeline

# Initialize FastAPI app
app = FastAPI(title="Hunyuan Video Generation API")

# Define request model
class VideoGenerationRequest(BaseModel):
    prompt: str = "A beautiful sunset over a calm ocean, with gentle waves."
    num_frames: int = 24  # Reduced default frames
    num_inference_steps: int = 4  # Reduced inference steps
    guidance_scale: float = 1.0
    embedded_cfg_scale: float = 6.0
    width: int = 854  # Reduced resolution
    height: int = 480
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    user_uuid: str = "system_default"
    output_path: Optional[str] = None
    fps: int = 24
    flow_shift: int = 17
    quantization: str = "nf4"  # Default to nf4 quantization
    video_length: Optional[float] = None

# Set multiprocessing start method
multiprocessing.set_start_method('spawn', force=True)

# Video generation task
def generate_video_task(prompt, output_path, video_path, video_file_name, user_uuid,
                       height, width, num_frames, num_inference_steps, guidance_scale,
                       negative_prompt, seed, flow_shift, fps, quantization, video_length=None):
    try:
        pipeline = initialize_pipeline(quantization)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if pipeline is None:
            raise ValueError("Failed to initialize pipeline")
        
        generator = torch.Generator("cpu").manual_seed(seed if seed is not None else torch.seed())
        pipeline.scheduler._shift = flow_shift
        
        if video_length is not None:
            num_frames = min(int(video_length * fps), 24)  # Cap frames at 24
            logging.info(f"Adjusted num_frames to {num_frames} based on video_length {video_length}s at {fps} fps")
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
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
        
        if output is None:
            raise ValueError("Pipeline returned None for output frames")
        
        export_to_video(output, video_path, fps=fps)
        logging.info(f"Video generated at: {video_path}")
        logging.info(f"Generation time: {time.perf_counter() - start_time:.2f} seconds")
        logging.info(f"Max VRAM for denoise: {torch.cuda.max_memory_allocated(device='cuda') / 1024**3:.3f} GiB")
        
        # Upload to S3
        upload_file(video_path, user_uuid, S3_BUCKET, video_file_name)
        logging.info(f"Uploaded video to s3://{S3_BUCKET}/{user_uuid}/{video_file_name}")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    except Exception as e:
        logging.error(f"Error in background task: {str(e)}")
        import traceback
        traceback.print_exc()

@app.post("/generate-video")
async def generate_video(request: VideoGenerationRequest):
    try:
        logging.info("Worker Start")
        
        pipeline = initialize_pipeline(request.quantization)
        output_path = request.output_path or os.path.join(OUTPUT_PATH, 'my_videos/')
        os.makedirs(output_path, exist_ok=True)
        
        current_time = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        video_file_name = f"{current_time}_{random_string}.mp4"
        video_path = os.path.join(output_path, video_file_name)
        
        task_id = str(uuid.uuid4())
        
        actual_num_frames = request.num_frames
        if request.video_length is not None:
            actual_num_frames = min(int(request.video_length * request.fps), 24)
            logging.info(f"Using video_length: {request.video_length}s at {request.fps} fps = {actual_num_frames} frames")
        
        # Use Process Pool to limit concurrent processes
        with multiprocessing.Pool(processes=1) as pool:
            pool.apply_async(generate_video_task, args=(
                request.prompt,
                output_path,
                video_path,
                video_file_name,
                request.user_uuid,
                request.height,
                request.width,
                actual_num_frames,
                request.num_inference_steps,
                request.guidance_scale,
                request.negative_prompt,
                request.seed,
                request.flow_shift,
                request.fps,
                request.quantization,
                request.video_length
            ))
        
        s3_url = f"s3://{S3_BUCKET}/{request.user_uuid}/{video_file_name}"
        return {
            "output": {
                "status": "processing",
                "task_id": task_id,
                "prompt": request.prompt,
                "expected_video_path": video_path,
                "expected_s3_url": s3_url,
                "parameters": {
                    "num_frames": actual_num_frames,
                    "width": request.width,
                    "height": request.height,
                    "num_inference_steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "embedded_cfg_scale": request.embedded_cfg_scale,
                    "flow_shift": request.flow_shift,
                    "seed": request.seed,
                    "fps": request.fps,
                    "quantization": request.quantization,
                    "video_length": request.video_length
                }
            }
        }
    except Exception as e:
        logging.error(f"Error in generate_video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clear GPU memory after each request
        torch.cuda.empty_cache()

if __name__ == '__main__':
    initialize_pipeline()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)