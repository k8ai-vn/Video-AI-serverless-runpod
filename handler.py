import runpod
import os
import boto3
from botocore.config import Config
from fastvideo import VideoGenerator, SamplingParam, PipelineConfig
import torch
import uuid

def handler(event):
    try:
        # Get input parameters from the event
        input_params = event.get("input", {})
        prompt = input_params.get("prompt", "A beautiful sunset over a calm ocean, with gentle waves.")
        num_frames = input_params.get("num_frames", 45)
        width = input_params.get("width", 1024)
        height = input_params.get("height", 576)
        num_inference_steps = input_params.get("num_inference_steps", 6)
        guidance_scale = input_params.get("guidance_scale", 7.5)
        seed = input_params.get("seed")  # Optional, can be None
        
        # Setup model
        model_name = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        config = PipelineConfig.from_pretrained(model_name)
        config.vae_precision = "fp16"
        config.use_cpu_offload = True

        # Check available GPU count
        available_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {available_gpus}")
        
        if available_gpus == 0:
            print("No GPUs detected. Running on CPU only.")
            config.use_cpu_offload = True
        
        # Create the generator
        generator = VideoGenerator.from_pretrained(
            model_name,
            num_gpus=min(available_gpus, 4),
            pipeline_config=config
        )

        # Create and customize sampling parameters
        sampling_param = SamplingParam.from_pretrained(model_name)
        
        # Set parameters
        sampling_param.num_frames = num_frames
        sampling_param.width = width
        sampling_param.height = height
        sampling_param.num_inference_steps = num_inference_steps
        sampling_param.guidance_scale = guidance_scale
        if seed is not None:
            sampling_param.seed = seed

        # Create output directory
        output_dir = "/workspace/output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate video
        video_path = os.path.join(output_dir, "generated_video.mp4")
        video = generator.generate_video(
            prompt, 
            sampling_param=sampling_param, 
            output_path=output_dir,
            return_frames=False,
            save_video=True
        )
        
        # Generate a unique filename instead of using the prompt
        unique_filename = f"video_{uuid.uuid4()}.mp4"
        os.rename(video_path, f"/workspace/output/{unique_filename}")
        
        # Upload to S3 if credentials are provided
        s3_url = None
        if all(key in os.environ for key in ["S3_ACCESS_KEY", "S3_SECRET_KEY"]):
            try:
                bucket_url = 'https://eu2.contabostorage.com/'
                s3_client = boto3.client(
                    's3',
                    endpoint_url=bucket_url,
                    aws_access_key_id=os.environ["S3_ACCESS_KEY"],
                    aws_secret_access_key=os.environ["S3_SECRET_KEY"],
                    config=Config(
                        request_checksum_calculation="when_required",
                        response_checksum_validation="when_required"
                    )
                )
                
                # Use fixed bucket name
                bucket_name = 'ttv-storage'
                
                # Generate a unique filename instead of using the prompt
                unique_filename = f"video_{uuid.uuid4()}.mp4"
                s3_key = f"videos/{unique_filename}"
                
                s3_client.upload_file(f"/workspace/output/{unique_filename}", bucket_name, s3_key)
                s3_url = f"s3://{bucket_name}/{s3_key}"
                print(f"Uploaded video to {s3_url}")
            except Exception as e:
                print(f"S3 upload error: {str(e)}")
        
        return {
            "output": {
                "prompt": prompt,
                "video_path": video_path,
                "s3_url": s3_url,
                "parameters": {
                    "num_frames": num_frames,
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