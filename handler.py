import time  
import argparse
import json
import os
import datetime
import random
import string
import logging
import boto3
import torch
import uuid
import multiprocessing
from botocore.exceptions import ClientError
from botocore.config import Config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Import FastVideo components
from fastvideo import VideoGenerator, SamplingParam, PipelineConfig
from fastvideo.models.hunyuan_hf.modeling_hunyuan import HunyuanVideoTransformer3DModel
from fastvideo.models.hunyuan_hf.pipeline_hunyuan import HunyuanVideoPipeline
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, nccl_info
from diffusers import BitsAndBytesConfig
from diffusers.utils import export_to_video

# Define network storage paths
NETWORK_STORAGE_PATH = os.environ.get('NETWORK_STORAGE', '/workspace')
OUTPUT_PATH = os.path.join(NETWORK_STORAGE_PATH, 'outputs')
S3_BUCKET = 'ttv-storage'
S3_ACCESS_KEY = os.environ.get('S3_ACCESS_KEY')
print('S3_ACCESS_KEY', S3_ACCESS_KEY)
S3_SECRET_KEY = os.environ.get('S3_SECRET_KEY')
# MODEL_NAME = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
MODEL_NAME = 'data/FastHunyuan-diffusers'

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
    # try:
    # object_name = user_uuid + '/' + datetime.datetime.now().strftime('%d%m%Y%H%M%S') + '_' + file_name
    # object_name = last part of file_name
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
        
    # except ClientError as e:
    #     logging.error(e)
    #     return False
    return True

def initialize_distributed():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)
    initialize_sequence_parallel_state(world_size)

# Configure the pipeline
config = PipelineConfig.from_pretrained(MODEL_NAME)
config.num_gpus = 1 # how many GPUS to parallelize generation
# config.vae_config.vae_precision = "fp32"

# Initialize the generator at module level but only when needed
generator = None

def initialize_generator():
    global generator
    if generator is None:
        generator = VideoGenerator.from_pretrained(
            MODEL_NAME,
            pipeline_config=config
        )
    return generator

# Initialize FastAPI app
app = FastAPI(title="Video Generation API")

# Define request model
class VideoGenerationRequest(BaseModel):
    prompt: str = "A beautiful sunset over a calm ocean, with gentle waves."
    num_frames: int = 107
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 576
    seed: Optional[int] = None
    negative_prompt: str = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    user_uuid: str = "system_default"
    output_path: Optional[str] = None
    quantization: Optional[str] = None
    cpu_offload: bool = False
    model_path: Optional[str] = None
    flow_shift: int = 7
    fps: int = 24

@app.post("/generate-video")
async def generate_video(request: VideoGenerationRequest):
    try:
        print(f"Worker Start")
        
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
        
        # Start a background process for video generation
        def generate_video_task(prompt, request_params, video_path, video_file_name, user_uuid):
            try:
                # Check if using quantization
                if request_params.get("quantization"):
                    # Use Hunyuan pipeline with quantization
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model_id = request_params.get("model_path") or MODEL_NAME
                    
                    if request_params["quantization"] == "nf4":
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_quant_type="nf4",
                            llm_int8_skip_modules=["proj_out", "norm_out"]
                        )
                        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                            model_id,
                            subfolder="transformer/",
                            torch_dtype=torch.bfloat16,
                            quantization_config=quantization_config
                        )
                    elif request_params["quantization"] == "int8":
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True, 
                            llm_int8_skip_modules=["proj_out", "norm_out"]
                        )
                        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                            model_id,
                            subfolder="transformer/",
                            torch_dtype=torch.bfloat16,
                            quantization_config=quantization_config
                        )
                    else:
                        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                            model_id,
                            subfolder="transformer/",
                            torch_dtype=torch.bfloat16
                        ).to(device)
                    
                    if not request_params.get("cpu_offload"):
                        pipe = HunyuanVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
                        pipe.transformer = transformer
                    else:
                        pipe = HunyuanVideoPipeline.from_pretrained(
                            model_id, 
                            transformer=transformer, 
                            torch_dtype=torch.bfloat16
                        )
                    
                    pipe.scheduler._shift = request_params.get("flow_shift", 7)
                    pipe.vae.enable_tiling()
                    
                    if request_params.get("cpu_offload"):
                        pipe.enable_model_cpu_offload()
                    
                    # Generate the video
                    torch_generator = torch.Generator("cpu").manual_seed(request_params.get("seed") or 42)
                    
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = pipe(
                            prompt=prompt,
                            height=request_params.get("height", 576),
                            width=request_params.get("width", 1024),
                            num_frames=request_params.get("num_frames", 107),
                            num_inference_steps=request_params.get("num_inference_steps", 30),
                            generator=torch_generator,
                        ).frames[0]
                    
                    export_to_video(output, video_path, fps=request_params.get("fps", 24))
                    
                else:
                    # Use standard FastVideo generator
                    generator = initialize_generator()
                    
                    # Generation config
                    param = SamplingParam.from_pretrained(MODEL_NAME)
                    param.num_inference_steps = request_params.get("num_inference_steps", 30)
                    param.guidance_scale = request_params.get("guidance_scale", 7.5)
                    param.width = request_params.get("width", 1024)
                    param.height = request_params.get("height", 576)
                    param.negative_prompt = request_params.get("negative_prompt", "")
                    param.num_frames = request_params.get("num_frames", 107)
                    if request_params.get("seed") is not None:
                        param.seed = request_params.get("seed")
                    
                    # Generate the video
                    video_result = generator.generate_video(
                        prompt,
                        sampling_param=param,
                        output_path=os.path.dirname(video_path),
                        save_video=True
                    )
                    
                    # Check if video_result contains a path attribute or if it's a dictionary
                    if hasattr(video_result, 'path'):
                        video_path_exported = os.path.join(os.path.dirname(video_path), video_result.path)
                        os.rename(video_path_exported, video_path)
                    else:
                        mp4_files = [f for f in os.listdir(os.path.dirname(video_path)) if f.endswith('.mp4')]
                        if mp4_files:
                            newest_file = max(mp4_files, key=lambda f: os.path.getctime(os.path.join(os.path.dirname(video_path), f)))
                            os.rename(os.path.join(os.path.dirname(video_path), newest_file), video_path)
                
                print(f"Video path: {video_path}")
                
                # Upload the video to S3
                upload_file(video_path, 
                            user_uuid, 
                            S3_BUCKET, 
                            video_file_name)
                print(f"Uploaded video to s3://{S3_BUCKET}/{user_uuid}/{video_file_name}")
            except Exception as e:
                print(f"Error in background task: {str(e)}")
        
        # Prepare parameters for the task
        request_params = {
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "width": request.width,
            "height": request.height,
            "negative_prompt": request.negative_prompt,
            "num_frames": request.num_frames,
            "seed": request.seed,
            "quantization": request.quantization,
            "cpu_offload": request.cpu_offload,
            "model_path": request.model_path,
            "flow_shift": request.flow_shift,
            "fps": request.fps
        }
        
        # Start the background process
        process = multiprocessing.Process(
            target=generate_video_task,
            args=(
                request.prompt,
                request_params,
                video_path,
                video_file_name,
                request.user_uuid
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
                    "num_frames": request.num_frames,
                    "width": request.width,
                    "height": request.height,
                    "num_inference_steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "seed": request.seed,
                    "quantization": request.quantization,
                    "cpu_offload": request.cpu_offload
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Start the FastAPI server
if __name__ == '__main__':
    # Add multiprocessing support for Windows if needed
    multiprocessing.freeze_support()
    # Initialize the generator once at startup
    initialize_generator()
    # Start the FastAPI server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)