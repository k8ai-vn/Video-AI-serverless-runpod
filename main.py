import os
import torch
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.utils import export_to_video

# Tối ưu hóa bộ nhớ ngay từ đầu
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# Tải mô hình chưng cất với float16 để giảm VRAM
pipe = LTXConditionPipeline.from_pretrained(
    "Lightricks/ltx-video-distilled",
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
    "Lightricks/ltxv-spatial-upscaler-0.9.7",
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True
)

# Chuyển mô hình sang GPU và bật tối ưu hóa
pipe.to("cuda")
pipe_upsample.to("cuda")
pipe.vae.enable_tiling()  # Giảm VRAM cho VAE
pipe.enable_sequential_cpu_offload()  # Chuyển tính toán sang CPU khi cần

# Cấu hình video
prompt = (
    "The video depicts a winding mountain road covered in snow, with a single vehicle traveling along it. "
    "The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by "
    "rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a "
    "winter drive through a mountainous region."
)
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
expected_height, expected_width = 704, 512
downscale_factor = 2 / 3
num_frames = 121

# Phần 1: Tạo video độ phân giải thấp
downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
latents = pipe(
    conditions=None,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=downscaled_width,
    height=downscaled_height,
    num_frames=num_frames,
    num_inference_steps=20,  # Giảm steps để tiết kiệm VRAM
    generator=torch.Generator().manual_seed(0),
    output_type="latent",
).frames

# Xóa bộ nhớ sau bước 1
torch.cuda.empty_cache()

# Phần 2: Upscale video bằng latent upsampler
upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
upscaled_latents = pipe_upsample(
    latents=latents,
    output_type="latent"
).frames

# Xóa bộ nhớ sau bước 2
torch.cuda.empty_cache()

# Phần 3: Denoise video upscale (giảm steps để tiết kiệm VRAM)
video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=upscaled_width,
    height=upscaled_height,
    num_frames=num_frames,
    denoise_strength=0.3,  # Giảm strength để giảm tính toán
    num_inference_steps=8,  # Giảm steps
    latents=upscaled_latents,
    decode_timestep=0.05,
    image_cond_noise_scale=0.025,
    generator=torch.Generator().manual_seed(0),
    output_type="pil",
).frames[0]

# Xóa bộ nhớ sau bước 3
torch.cuda.empty_cache()

# Phần 4: Downscale về độ phân giải mong muốn
video = [frame.resize((expected_width, expected_height)) for frame in video]

# Xuất video
export_to_video(video, "output.mp4", fps=24)