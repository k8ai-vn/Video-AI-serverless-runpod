import runpod
import os
import boto3
import datetime
import random
import string
import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from botocore.exceptions import ClientError
from botocore.config import Config
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import imageio
import numpy as np
import torchvision
from einops import rearrange

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
MODEL_PATH = os.path.join(MODEL_CACHE_PATH, 'hunyuan')
MASK_STRATEGY_FILE_PATH = os.path.join(MODEL_CACHE_PATH, 'mask_strategy.json')

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

def teacache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_attention_mask: torch.Tensor,
    mask_strategy=None,
    output_features=False,
    output_features_stride=8,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = False,
    guidance=None,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if guidance is None:
        guidance = torch.tensor([6016.0], device=hidden_states.device, dtype=torch.bfloat16)

    img = x = hidden_states
    text_mask = encoder_attention_mask
    t = timestep
    txt = encoder_hidden_states[:, 1:]
    text_states_2 = encoder_hidden_states[:, 0, :self.config.text_states_dim_2]
    _, _, ot, oh, ow = x.shape  # codespell:ignore
    tt, th, tw = (
        ot // self.patch_size[0],  # codespell:ignore
        oh // self.patch_size[1],  # codespell:ignore
        ow // self.patch_size[2],  # codespell:ignore
    )
    original_tt = nccl_info.sp_size * tt
    freqs_cos, freqs_sin = self.get_rotary_pos_embed((original_tt, th, tw))
    # Prepare modulation vectors.
    vec = self.time_in(t)

    # text modulation
    vec = vec + self.vector_in(text_states_2)

    # guidance modulation
    if self.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")

        # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
        vec = vec + self.guidance_in(guidance)

    # Embed image and text.
    img = self.img_in(img)
    if self.text_projection == "linear":
        txt = self.txt_in(txt)
    elif self.text_projection == "single_refiner":
        txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
    else:
        raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

    txt_seq_len = txt.shape[1]
    img_seq_len = img.shape[1]

    freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

    if self.enable_teacache:
        inp = img.clone()
        vec_ = vec.clone()
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.double_blocks[0].img_mod(vec_).chunk(6, dim=-1)
        normed_inp = self.double_blocks[0].img_norm1(inp)
        modulated_inp = modulate(normed_inp, shift=img_mod1_shift, scale=img_mod1_scale)
        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = [7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                ((modulated_inp - self.previous_modulated_input).abs().mean() /
                 self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0
    if self.enable_teacache:
        if not should_calc:
            img += self.previous_residual
        else:
            ori_img = img.clone()
            # --------------------- Pass through DiT blocks ------------------------
            for index, block in enumerate(self.double_blocks):
                double_block_args = [img, txt, vec, freqs_cis, text_mask, mask_strategy[index]]
                img, txt = block(*double_block_args)

            # Merge txt and img to pass through single stream blocks.
            x = torch.cat((img, txt), 1)
            if output_features:
                features_list = []
            if len(self.single_blocks) > 0:
                for index, block in enumerate(self.single_blocks):
                    single_block_args = [
                        x,
                        vec,
                        txt_seq_len,
                        (freqs_cos, freqs_sin),
                        text_mask,
                        mask_strategy[index + len(self.double_blocks)],
                    ]
                    x = block(*single_block_args)
                    if output_features and _ % output_features_stride == 0:
                        features_list.append(x[:, :img_seq_len, ...])

            img = x[:, :img_seq_len, ...]
            self.previous_residual = img - ori_img
    else:
        # --------------------- Pass through DiT blocks ------------------------
        for index, block in enumerate(self.double_blocks):
            double_block_args = [img, txt, vec, freqs_cis, text_mask, mask_strategy[index]]
            img, txt = block(*double_block_args)
        # Merge txt and img to pass through single stream blocks.
        x = torch.cat((img, txt), 1)
        if output_features:
            features_list = []
        if len(self.single_blocks) > 0:
            for index, block in enumerate(self.single_blocks):
                single_block_args = [
                    x,
                    vec,
                    txt_seq_len,
                    (freqs_cos, freqs_sin),
                    text_mask,
                    mask_strategy[index + len(self.double_blocks)],
                ]
                x = block(*single_block_args)
                if output_features and _ % output_features_stride == 0:
                    features_list.append(x[:, :img_seq_len, ...])

        img = x[:, :img_seq_len, ...]

    # ---------------------------- Final layer ------------------------------
    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

    img = self.unpatchify(img, tt, th, tw)
    assert not return_dict, "return_dict is not supported."
    if output_features:
        features_list = torch.stack(features_list, dim=0)
    else:
        features_list = None
    return (img, features_list)

def initialize_distributed():
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    logging.info("world_size: %d", world_size)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)
    from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, nccl_info
    initialize_sequence_parallel_state(world_size)

def worker(rank, world_size, event):
    """Worker function for distributed video generation."""
    try:
        # Set environment variables for distributed training
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["NCCL_DEBUG"] = "INFO"

        initialize_distributed()
        
        from fastvideo.models.hunyuan.inference import HunyuanVideoSampler
        from fastvideo.models.hunyuan.modules.modulate_layers import modulate
        from fastvideo.utils.parallel_states import nccl_info

        # Extract input data
        input_data = event['input']
        prompt = input_data.get('prompt', "A beautiful landscape")
        num_frames = input_data.get('num_frames', 16)
        height = input_data.get('height', 256)
        width = input_data.get('width', 256)
        num_inference_steps = input_data.get('num_inference_steps', 50)
        guidance_scale = input_data.get('guidance_scale', 1.0)
        embedded_cfg_scale = input_data.get('embedded_cfg_scale', 6.0)
        neg_prompt = input_data.get('neg_prompt', None)
        seed = input_data.get('seed', None)
        fps = input_data.get('fps', 24)
        rel_l1_thresh = input_data.get('rel_l1_thresh', 0.15)
        enable_teacache = input_data.get('enable_teacache', True)
        
        # Create output directory
        output_dir = os.path.join(OUTPUT_PATH, 'videos')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{datetime.datetime.now().strftime('%d%m%Y%H%M%S')}_{random.randint(1000, 9999)}.mp4")
        
        # Load mask strategy
        with open(MASK_STRATEGY_FILE_PATH, 'r') as f:
            mask_strategy = json.load(f)
        
        # Create args object to match the expected interface
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        args = Args(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            model_path=MODEL_PATH,
            output_path=output_path,
            fps=fps,
            sliding_block_size="8,6,10",
            denoise_type="flow",
            seed=seed,
            neg_prompt=neg_prompt,
            guidance_scale=guidance_scale,
            embedded_cfg_scale=embedded_cfg_scale,
            flow_shift=7,
            batch_size=1,
            num_videos=1,
            load_key="module",
            use_cpu_offload=False,
            dit_weight=os.path.join(MODEL_PATH, "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"),
            reproduce=False,
            disable_autocast=False,
            flow_reverse=False,
            flow_solver="euler",
            use_linear_quadratic_schedule=False,
            linear_schedule_end=25,
            model="HYVideo-T/2-cfgdistill",
            latent_channels=16,
            precision="bf16",
            rope_theta=256,
            vae="884-16c-hy",
            vae_precision="fp16",
            vae_tiling=True,
            vae_sp=False,
            text_encoder="llm",
            text_encoder_precision="fp16",
            text_states_dim=4096,
            text_len=256,
            tokenizer="llm",
            prompt_template="dit-llm-encode",
            prompt_template_video="dit-llm-encode-video",
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            text_encoder_2="clipL",
            text_encoder_precision_2="fp16",
            text_states_dim_2=768,
            tokenizer_2="clipL",
            text_len_2=77,
            skip_time_steps=10,
            mask_strategy_selected=[1, 2, 6],
            rel_l1_thresh=rel_l1_thresh,
            enable_teacache=enable_teacache,
            enable_torch_compile=False,
            mask_strategy_file_path=MASK_STRATEGY_FILE_PATH
        )
        
        logging.info("Loading HunyuanVideoSampler from %s", MODEL_PATH)
        hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(MODEL_PATH, args=args)
        
        # Apply teacache if enabled
        if enable_teacache:
            logging.info("Enabling TeaCache with threshold: %f", rel_l1_thresh)
            hunyuan_video_sampler.pipeline.transformer.__class__.enable_teacache = True
            hunyuan_video_sampler.pipeline.transformer.__class__.cnt = 0
            hunyuan_video_sampler.pipeline.transformer.__class__.num_steps = num_inference_steps
            hunyuan_video_sampler.pipeline.transformer.__class__.rel_l1_thresh = rel_l1_thresh
            hunyuan_video_sampler.pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
            hunyuan_video_sampler.pipeline.transformer.__class__.previous_modulated_input = None
            hunyuan_video_sampler.pipeline.transformer.__class__.previous_residual = None
            hunyuan_video_sampler.pipeline.transformer.__class__.forward = teacache_forward
        
        logging.info("Generating video for prompt: %s", prompt)
        outputs = hunyuan_video_sampler.predict(
            prompt=prompt,
            height=height,
            width=width,
            video_length=num_frames,
            seed=seed,
            negative_prompt=neg_prompt,
            infer_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_videos_per_prompt=1,
            flow_shift=7,
            batch_size=1,
            embedded_guidance_scale=embedded_cfg_scale,
            mask_strategy=mask_strategy,
        )
        
        videos = rearrange(outputs["samples"], "b c t h w -> t b c h w")
        output_frames = []
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=6)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            output_frames.append((x * 255).numpy().astype(np.uint8))
        
        imageio.mimsave(output_path, output_frames, fps=fps)
        logging.info("Video saved to %s", output_path)
        
        # Upload to S3
        success, presigned_url = upload_file(
            output_path,
            input_data.get('user_uuid', 'system_default'),
            S3_BUCKET,
            os.path.basename(output_path)
        )
        
        if not success:
            logging.error("Failed to upload video")
            return None
        
        return {"prompt": prompt, "s3_url": presigned_url}
    except Exception as e:
        logging.error("Worker failed: %s", str(e), exc_info=True)
        return None

def handler(event):
    """Serverless handler function."""
    try:
        world_size = torch.cuda.device_count()
        logging.info("Starting worker with %d GPUs", world_size)
        return worker(0, world_size, event) or {"error": "Video generation failed"}
    except Exception as e:
        logging.error("Handler failed: %s", str(e), exc_info=True)
        return {"error": str(e)}

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})