import os
import time
import json
import uuid
import random
import string
import logging
import datetime
import threading
from typing import Optional, List, Dict

import torch
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from diffusers.utils import export_to_video
from fastvideo.models.hunyuan_hf.modeling_hunyuan import HunyuanVideoTransformer3DModel
from transformers import BitsAndBytesConfig

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from fastvideo.models.hunyuan_hf.pipeline_hunyuan import HunyuanVideoPipeline

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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    logger.info(f'object_name: {object_name}')
    object_name = user_uuid + '/' + object_name
    logger.info(f'full object_name: {object_name}')
    
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
        
        # Create a presigned URL for the file
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': object_name},
            ExpiresIn=3600  # URL will be valid for 1 hour
        )   
        logger.info(f"Generated presigned URL: {presigned_url}")
        
        # delete file from local storage to save space
        os.remove(file_name)
        return presigned_url
    except ClientError as e:
        logger.error(f"Error uploading file to S3: {str(e)}")
        return None

# Global pipeline instance - only initialize once
PIPELINE = None
PIPELINE_LOCK = threading.Lock()

def get_pipeline(quantization="fp16"):
    """Get or initialize the pipeline with singleton pattern"""
    global PIPELINE
    
    with PIPELINE_LOCK:
        if PIPELINE is None:
            try:
                logger.info(f"Initializing pipeline with quantization: {quantization}")
                device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
                # Use bfloat16 for compatibility with FlashAttention
                weight_dtype = torch.bfloat16
                
                logger.info(f"Loading model from: {MODEL_PATH}")
                
                # Check if model path exists
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(f"Model path {MODEL_PATH} does not exist")
                
                # Load transformer model with appropriate quantization
                if quantization == "nf4":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4",
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
                else:  # fp16 or bf16
                    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                        MODEL_PATH,
                        subfolder="transformer/",
                        torch_dtype=weight_dtype
                    ).to(device)
                
                logger.info(f"Max VRAM for transformer load: {round(torch.cuda.max_memory_allocated(device='cuda') / 1024**3, 3)} GiB")
                torch.cuda.reset_max_memory_allocated(device)
                
                # Initialize pipeline with the loaded transformer
                PIPELINE = HunyuanVideoPipeline.from_pretrained(
                    MODEL_PATH, 
                    transformer=transformer,
                    torch_dtype=weight_dtype,
                    local_files_only=True
                )
                torch.cuda.reset_max_memory_allocated(device)

                # Enable VAE tiling for better memory usage
                PIPELINE.enable_vae_tiling()
                
                # Set flow shift parameter
                PIPELINE.scheduler._shift = 17  # Default flow shift

                # Enable CPU offload for memory efficiency
                PIPELINE.enable_model_cpu_offload()
                
                logger.info(f"Max VRAM for pipeline initialization: {round(torch.cuda.max_memory_allocated(device='cuda') / 1024**3, 3)} GiB")
                logger.info("Pipeline initialized successfully and will remain in memory")
            except Exception as e:
                logger.error(f"Error initializing pipeline: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                raise
    
    return PIPELINE

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
    quantization: str = "fp16"  # Options: fp16, bf16, nf4, int8
    video_length: Optional[float] = None  # Parameter for video length in seconds

# Store ongoing tasks
active_tasks: Dict[str, Dict] = {}
tasks_lock = threading.Lock()

# Background task for video generation
async def generate_video_task(
    task_id: str,
    prompt: str, 
    output_path: str,
    video_path: str,
    video_file_name: str,
    user_uuid: str, 
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float, 
    negative_prompt: Optional[str],
    seed: Optional[int],
    flow_shift: int,
    fps: int,
    quantization: str,
    video_length: Optional[float] = None
):
    try:
        # Update task status
        with tasks_lock:
            active_tasks[task_id]["status"] = "processing"
        
        # Get the already initialized pipeline
        pipeline = get_pipeline(quantization)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set up generator for reproducibility
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cpu").manual_seed(seed)
        torch.cuda.reset_max_memory_allocated(device)
        
        # Set flow shift parameter
        pipeline.scheduler._shift = flow_shift
        
        # Adjust num_frames based on video_length if provided
        if video_length is not None:
            num_frames = int(video_length * fps)
            logger.info(f"Adjusted num_frames to {num_frames} based on video_length of {video_length} seconds at {fps} fps")
        
        # Generate the video with torch.autocast for better performance
        logger.info(f"Starting video generation for task {task_id}")
        start_time = time.perf_counter()
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = pipeline(
                prompt=prompt,
                # negative_prompt=negative_prompt,
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
        generation_time = round(time.perf_counter() - start_time, 2)
        logger.info(f"Video generated at: {video_path}")
        logger.info(f"Generation time: {generation_time} seconds")
        logger.info(f"Max VRAM usage: {round(torch.cuda.max_memory_allocated(device='cuda') / 1024**3, 3)} GiB")

        # Upload the video to S3
        presigned_url = upload_file(video_path, user_uuid, S3_BUCKET, video_file_name)
        logger.info(f"Uploaded video to s3://{S3_BUCKET}/{user_uuid}/{video_file_name}")
        
        # Update task status to completed
        with tasks_lock:
            active_tasks[task_id]["status"] = "completed"
            active_tasks[task_id]["presigned_url"] = presigned_url
            active_tasks[task_id]["generation_time"] = generation_time
            active_tasks[task_id]["finished_at"] = datetime.datetime.now().isoformat()
            
    except Exception as e:
        logger.error(f"Error in background task: {str(e)}")
        # Log the full traceback for better debugging
        import traceback
        logger.error(traceback.format_exc())
        
        # Update task status to failed
        with tasks_lock:
            active_tasks[task_id]["status"] = "failed"
            active_tasks[task_id]["error"] = str(e)
            active_tasks[task_id]["finished_at"] = datetime.datetime.now().isoformat()

@app.post("/generate-video")
async def generate_video(request: VideoGenerationRequest, background_tasks: BackgroundTasks):
    try:
        logger.info("Received video generation request")
        
        # Make sure pipeline is initialized but don't reload
        _ = get_pipeline(request.quantization)

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
            logger.info(f"Using video_length parameter: {request.video_length} seconds at {request.fps} fps = {actual_num_frames} frames")
        
        # Store task information
        with tasks_lock:
            active_tasks[task_id] = {
                "status": "queued",
                "started_at": datetime.datetime.now().isoformat(),
                "prompt": request.prompt,
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
        
        # Add the task to background tasks
        background_tasks.add_task(
            generate_video_task,
            task_id,
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
        )
        
        # Return immediately with task ID
        s3_url = f"s3://{S3_BUCKET}/{request.user_uuid}/{video_file_name}"
        return {
            "output": {
                "status": "queued",
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
                    "seed": request.seed or "random",
                    "fps": request.fps,
                    "quantization": request.quantization,
                    "video_length": request.video_length
                }
            }
        }
    except Exception as e:
        logger.error(f"Error handling video generation request: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a task by task ID"""
    with tasks_lock:
        if task_id not in active_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"task": active_tasks[task_id]}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": PIPELINE is not None}

# Start the FastAPI server
if __name__ == '__main__':
    # Initialize the pipeline once at startup
    get_pipeline()
    # Start the FastAPI server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)