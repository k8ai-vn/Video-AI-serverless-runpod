import runpod
import time
import os
import boto3
import datetime
import random
import string
import logging
from botocore.exceptions import ClientError
from botocore.config import Config
import torch
import torch.multiprocessing as mp

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Check GPU availability
logging.info("CUDA available: %s", torch.cuda.is_available())
logging.info("Number of GPUs: %d", torch.cuda.device_count())
if torch.cuda.is_available():
    logging.info("GPU Name: %s", torch.cuda.get_device_name(0))

# Define network storage paths
NETWORK_STORAGE_PATH = os.environ.get('NETWORK_STORAGE', '/runpod-volume')
MODEL_CACHE_PATH = os.path.join(NETWORK_STORAGE_PATH, 'model_cache')
OUTPUT_PATH = os.path.join(NETWORK_STORAGE_PATH, 'outputs')
S3_BUCKET = 'ttv-storage'
S3_ACCESS_KEY = os.environ.get('S3_ACCESS_KEY')
S3_SECRET_KEY = os.environ.get('S3_SECRET_KEY')
MODEL_NAME = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'

# Create directories
os.makedirs(MODEL_CACHE_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Initialize S3 client
bucket_url = 'https://eu2.contabostorage.com/'
s3_client = boto3.client(
    's3',
    endpoint_url=bucket_url,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    config=Config(
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required"
    )
)

def upload_file(file_name, user_uuid, bucket, object_name=None):
    """Upload a file to an S3 bucket."""
    if object_name is None:
        object_name = os.path.basename(file_name)
    try:
        object_name = f"{user_uuid}/{datetime.datetime.now().strftime('%d%m%Y%H%M%S')}_{os.path.basename(file_name)}"
        s3_client.upload_file(file_name, bucket, object_name)
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': object_name},
            ExpiresIn=3600
        )
        logging.info("Uploaded to S3: %s", presigned_url)
        os.remove(file_name)
        return True, presigned_url
    except ClientError as e:
        logging.error("S3 upload failed: %s", e)
        return False, None

def worker(rank, world_size, event):
    """Worker function for distributed video generation."""
    try:
        # Set environment variables for distributed training
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["NCCL_DEBUG"] = "INFO"

        from fastvideo import VideoGenerator, SamplingParam, PipelineConfig

        # Initialize pipeline config
        config = PipelineConfig.from_pretrained(MODEL_NAME)
        # config.num_gpus = world_size
        config.vae_config.vae_precision = "fp32"

        # Create video generator
        logging.info("Rank %d: Initializing VideoGenerator", rank)
        generator = VideoGenerator.from_pretrained(MODEL_NAME, pipeline_config=config)

        # Extract input data
        input_data = event['input']
        prompt = input_data.get('prompt')
        num_inference_steps = input_data.get('num_inference_steps', 30)
        guidance_scale = input_data.get('guidance_scale', 7.5)
        width = input_data.get('width', 1024)
        height = input_data.get('height', 576)

        # Generation config
        param = SamplingParam.from_pretrained(MODEL_NAME)
        param.num_inference_steps = num_inference_steps
        param.guidance_scale = guidance_scale
        param.width = width
        param.height = height

        # Generate video
        output_path = input_data.get('output_path', os.path.join(OUTPUT_PATH, 'my_videos/'))
        os.makedirs(output_path, exist_ok=True)
        current_time = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        video_file_name = f"{current_time}_{random_string}.mp4"

        logging.info("Rank %d: Generating video for prompt: %s", rank, prompt)
        video = generator.generate_video(
            prompt,
            sampling_param=param,
            output_path=output_path,
            save_video=True
        )

        # Rename video file
        original_video_path = f"{output_path}{prompt}.mp4"
        new_video_path = f"{output_path}{video_file_name}"
        if os.path.exists(original_video_path):
            os.rename(original_video_path, new_video_path)
        else:
            logging.error("Video file %s not found", original_video_path)
            return None

        # Upload to S3
        success, presigned_url = upload_file(
            new_video_path,
            input_data.get('user_uuid', 'system_default'),
            S3_BUCKET,
            video_file_name
        )
        if not success:
            logging.error("Rank %d: Failed to upload video", rank)
            return None

        return {"prompt": prompt, "s3_url": presigned_url}
    except Exception as e:
        logging.error("Rank %d: Worker failed: %s", rank, str(e), exc_info=True)
        return None

def handler(event):
    """Serverless handler function."""
    try:
        world_size = 4
        logging.info("Starting %d worker processes", world_size)
        results = mp.spawn(worker, args=(world_size, event), nprocs=world_size, join=True)
        # Return result from rank 0 (main process)
        return worker(0, world_size, event) or {"error": "Video generation failed"}
    except Exception as e:
        logging.error("Handler failed: %s", str(e), exc_info=True)
        return {"error": str(e)}

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})