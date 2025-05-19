---
title: LTX Video Fast
emoji: 🎥
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
pinned: false
short_description: ultra-fast video model, LTX 0.9.7 13B distilled
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference



# Clone repository
git clone https://huggingface.co/spaces/Lightricks/ltx-video-distilled
cd ltx-video-distilled

# Create and activate Python environment
python -m venv env
source env/bin/activate

# Install dependencies and run
pip install -r requirements.txt
pip install gradio[mcp]

python app.py