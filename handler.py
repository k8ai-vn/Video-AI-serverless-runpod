import runpod
import time  
from fastvideo import VideoGenerator, SamplingParam, PipelineConfig
import os
import boto3
import datetime
import random
import string
### UPLOAD FILE TO S3
import logging
from botocore.exceptions import ClientError
from botocore.config import Config
import torch
import uuid

# Define network storage paths
NETWORK_STORAGE_PATH = os.environ.get('NETWORK_STORAGE', '/workspace')
# MODEL_CACHE_PATH = os.path.join(NETWORK_STORAGE_PATH, 'model_cache')
OUTPUT_PATH = os.path.join(NETWORK_STORAGE_PATH, 'outputs')
S3_BUCKET = 'ttv-storage'
S3_ACCESS_KEY = os.environ.get('S3_ACCESS_KEY')
S3_SECRET_KEY = os.environ.get('S3_SECRET_KEY')
# MODEL_NAME = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
MODEL_NAME = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'

# Create directories if they don't exist
# os.makedirs(MODEL_CACHE_PATH, exist_ok=True)
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
    try:
        object_name = user_uuid + '/' + datetime.datetime.now().strftime('%d%m%Y%H%M%S') + '_' + file_name

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
        
    except ClientError as e:
        logging.error(e)
        return False
    return True

config = PipelineConfig.from_pretrained(MODEL_NAME)
# Can adjust any parameters
# Other arguments will be set to best defaults
config.num_gpus = 1 # how many GPUS to parallelize generation
config.vae_config.vae_precision = "fp32"

# Create a video generator with a pre-trained model
generator = VideoGenerator.from_pretrained(
    MODEL_NAME,
    pipeline_config=config
    # num_gpus=4,  # Adjust based on your hardware
)


def handler(event):
    """
    This function processes incoming requests to your Serverless endpoint.
    
    Args:
        event (dict): Contains the input data and request metadata
        
    Returns:
        Any: The result to be returned to the client
    """
    try:
        # Extract input data
        print(f"Worker Start")
        input_params = event.get("input", {})
        
        prompt = input_params.get("prompt", "A beautiful sunset over a calm ocean, with gentle waves.")
        num_inference_steps = input_params.get("num_inference_steps", 30)
        guidance_scale = input_params.get("guidance_scale", 7.5)
        width = input_params.get("width", 1024)
        height = input_params.get("height", 576)
        seed = input_params.get("seed")  # Optional, can be None
        
        # Override default configurations while keeping optimal defaults for other settings
        generator = VideoGenerator.from_pretrained(MODEL_NAME, pipeline_config=config)

        # Generation config
        param = SamplingParam.from_pretrained(MODEL_NAME)
        # Adjust specific sampling parameters
        # Other arguments will be set to best defaults
        param.num_inference_steps = num_inference_steps # higher quality
        param.guidance_scale = guidance_scale # stronger guidance
        param.width = width  # Higher resolution
        param.height = height
        if seed is not None:
            param.seed = seed
        
        # Define output path
        output_path = input_params.get('output_path', os.path.join(OUTPUT_PATH, 'my_videos/'))
        os.makedirs(output_path, exist_ok=True)
        
        # Generate date time in DDMMYYYYHHMMSS format and random string
        current_time = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        video_file_name = f"{current_time}_{random_string}.mp4"
        
        # Generate the video
        video = generator.generate_video(
            prompt,
            sampling_param=param,
            # return_frames=True,  # Also return frames from this call (defaults to False)
            output_path=output_path,  # Controls where videos are saved
            save_video=True
        )
        
        video_path_exported = os.path.join(output_path, video.path)
        os.rename(video_path_exported, f"{output_path}{video_file_name}")
        
        # Upload the video to S3
        s3_url = None
        if all(key in os.environ for key in ["S3_ACCESS_KEY", "S3_SECRET_KEY"]):
            try:
                upload_file(f"{output_path}{video_file_name}", 
                           input_params.get('user_uuid', 'system_default'), 
                           S3_BUCKET, 
                           video_file_name)
                s3_url = f"s3://{S3_BUCKET}/{input_params.get('user_uuid', 'system_default')}/{video_file_name}"
                print(f"Uploaded video to {s3_url}")
            except Exception as e:
                print(f"S3 upload error: {str(e)}")
        
        return {
            "output": {
                "prompt": prompt,
                "video_path": f"{output_path}{video_file_name}",
                "s3_url": s3_url,
                "parameters": {
                    "num_frames": param.num_frames,
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed
                }
            }
        }
    except Exception as e:
        return {"error": str(e)}

# Start the RunPod serverless handler
if __name__ == '__main__':
    runpod.serverless.start({"handler": handler})