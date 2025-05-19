from fastvideo import VideoGenerator
import os
import os
# import boto3
# import datetime
# import random
# import string
# ### UPLOAD FILE TO S3
# import logging
# from botocore.exceptions import ClientError
# from botocore.config import Config

# # Define network storage paths
NETWORK_STORAGE_PATH = os.environ.get('NETWORK_STORAGE', '/runpod-volume')
MODEL_CACHE_PATH = os.path.join(NETWORK_STORAGE_PATH, 'model_cache')
OUTPUT_PATH = os.path.join(NETWORK_STORAGE_PATH, 'outputs')
# S3_BUCKET = 'ttv-storage'
# S3_ACCESS_KEY = os.environ.get('S3_ACCESS_KEY')
# S3_SECRET_KEY = os.environ.get('S3_SECRET_KEY')
# # MODEL_NAME = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
MODEL_NAME = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'

def main():
    # Create a video generator with a pre-trained model
    generator = VideoGenerator.from_pretrained(
        MODEL_NAME,
        num_gpus=1,  # Adjust based on your hardware
    )

    # Define a prompt for your video
    prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest."

    # Generate the video
    video = generator.generate_video(
        prompt,
        return_frames=True,  # Also return frames from this call (defaults to False)
        output_path="my_videos/",  # Controls where videos are saved
        save_video=True
    )

if __name__ == '__main__':
    main()