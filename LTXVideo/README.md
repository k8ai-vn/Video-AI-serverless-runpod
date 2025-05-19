<div align="center">

# Q8 LTX-Video

This is the repository for Q8 LTX-Video.

[Q8 Weights](https://huggingface.co/konakona/ltxvideo_q8) |
[Original repo](https://github.com/Lightricks/LTX-Video) |


</div>

## Table of Contents

- [Introduction](#introduction)
- [Model User Guide](#model-user-guide)
- [Acknowledgement](#acknowledgement)

# Introduction

LTX-VideoQ8 is 8bit adaptation of LTXVideo(https://github.com/Lightricks/LTX-Video) with no loss of accuracy and up to 3X speed up in NVIDIA ADA GPUs. Generate 720x480x121 videos in under a minute on RTX 4060 Laptop GPU with 8GB VRAM. Training code coming soon! (8GB VRAM is MORE than enough to full fine tune 2B transformer on ADA GPU with precalculated latents)

## Benchmarks
40 steps, RTX 4060 Laptop, CUDA 12.6, PyTorch 2.5.1

![Benchmarks](./docs/_static/output.png)

121x720x1280*: in diffusers the more steps it makes the slower it/sec gets, expected ~7min not 9mins according to it/sec of first 10 steps.

## Run locally

### Installation
The codebase was tested with Python 3.10.12, CUDA version 12.6, and supports PyTorch >= 2.5.1.


```bash
1) Install q8_kernels(https://github.com/KONAKONA666/q8_kernels)

2) git clone https://github.com/KONAKONA666/LTX-Video/tree/main
cd LTX-Video

python -m pip install -e .\[inference-script\]
```

Then, download the text encoder and vae from [Hugging Face](https://huggingface.co/Lightricks/LTX-Video) 
Download [Q8 version](https://huggingface.co/konakona/ltxvideo_q8)  or convert with q8_kernels.convert_weights 

```python
from huggingface_hub import snapshot_download

model_path = 'PATH'   # The local directory to save downloaded checkpoint
snapshot_download("konakona/ltxvideo_q8", local_dir=model_path, local_dir_use_symlinks=False, repo_type='model')
```

### Inference

follow the inference code in [inference.py](./inference.py):

#### For text-to-video generation:

```bash
python inference.py  --low_vram --transformer_type=q8_kernels --ckpt_dir  'PATH' --prompt "PROMPT" --height HEIGHT --width WIDTH --num_frames NUM_FRAMES --seed SEED
```

#### For image-to-video generation:

```bash
python inference.py --ckpt_dir 'PATH'  --low_vram --transformer_type=q8_kernels --prompt "PROMPT" --input_image_path IMAGE_PATH --height HEIGHT --width WIDTH --num_frames NUM_FRAMES --seed SEED
```

### Comparision
Left: 8bit, right 16bit

Find side to side comparisons in

https://github.com/KONAKONA666/LTX-Video/tree/main/docs/_static 

<!-- ![example1](./docs/_static/312661b4-974f-4db7-8e68-bc050debc782.gif)
![example2](./docs/_static/31632627-40ae-4dcf-aac9-99b70f908351.gif)
![example3](./docs/_static/62558328-6561-4486-9abe-4e13aa317577.gif)
![example4](./docs/_static/91d01bfa-e806-48b6-89b2-ed7a6733ac2f.gif)
![example5](./docs/_static/e37acb60-1f64-45b1-a8c1-4eff28af298a.gif)
![example5](./docs/_static/f989b225-8b82-4a2f-b119-91464803df95.gif)
 -->


# Model User Guide

## 📝 Prompt Engineering

When writing prompts, focus on detailed, chronological descriptions of actions and scenes. Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph. Start directly with the action, and keep descriptions literal and precise. Think like a cinematographer describing a shot list. Keep within 200 words. For best results, build your prompts using this structure:

* Start with main action in a single sentence
* Add specific details about movements and gestures
* Describe character/object appearances precisely
* Include background and environment details
* Specify camera angles and movements
* Describe lighting and colors
* Note any changes or sudden events
* See [examples](#introduction) for more inspiration.

## 🎮 Parameter Guide

* Resolution Preset: Higher resolutions for detailed scenes, lower for faster generation and simpler scenes. The model works on resolutions that are divisible by 32 and number of frames that are divisible by 8 + 1 (e.g. 257). In case the resolution or number of frames are not divisible by 32 or 8 + 1, the input will be padded with -1 and then cropped to the desired resolution and number of frames. The model works best on resolutions under 720 x 1280 and number of frames below 257
* Seed: Save seed values to recreate specific styles or compositions you like
* Guidance Scale: 3-3.5 are the recommended values
* Inference Steps: More steps (40+) for quality, fewer steps (20-30) for speed

## More to come...

# Acknowledgement

We are grateful for the following awesome projects when implementing LTX-Video:
* [DiT](https://github.com/facebookresearch/DiT) and [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha): vision transformers for image generation.
* [Lightricks](https://github.com/Lightricks/LTX-Video) for the model

[//]: # (## Citation)
