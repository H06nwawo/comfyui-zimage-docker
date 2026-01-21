#!/bin/bash

set -e

echo "Downloading Z-Image-Turbo model files..."

pip3 install -q huggingface-hub

python3 - <<EOF
from huggingface_hub import hf_hub_download
import os
import shutil

token = os.environ.get('HUGGING_FACE_HUB_TOKEN', '')

print("Downloading diffusion model...")
hf_hub_download(
    repo_id="Comfy-Org/z_image_turbo",
    filename="split_files/diffusion_models/z_image_turbo_bf16.safetensors",
    local_dir="/tmp/models",
    token=token if token else None
)

print("Downloading text encoder...")
hf_hub_download(
    repo_id="Comfy-Org/z_image_turbo",
    filename="split_files/text_encoders/qwen_3_4b.safetensors",
    local_dir="/tmp/models",
    token=token if token else None
)

print("Downloading VAE...")
hf_hub_download(
    repo_id="Comfy-Org/z_image_turbo",
    filename="split_files/vae/ae.safetensors",
    local_dir="/tmp/models",
    token=token if token else None
)

# Move files to correct ComfyUI directories
os.makedirs("/app/models/diffusion_models", exist_ok=True)
os.makedirs("/app/models/clip", exist_ok=True)
os.makedirs("/app/models/vae", exist_ok=True)

shutil.move("/tmp/models/split_files/diffusion_models/z_image_turbo_bf16.safetensors", 
            "/app/models/diffusion_models/z_image_turbo_bf16.safetensors")

shutil.move("/tmp/models/split_files/text_encoders/qwen_3_4b.safetensors", 
            "/app/models/clip/qwen_3_4b.safetensors")

shutil.move("/tmp/models/split_files/vae/ae.safetensors", 
            "/app/models/vae/ae.safetensors")

print("All models ready!")
EOF

echo "Starting ComfyUI..."
python3 main.py --listen 0.0.0.0 --port 8188
