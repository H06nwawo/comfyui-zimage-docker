#!/bin/bash

set -e

echo "Downloading Z-Image-Turbo model files..."

pip3 install -q huggingface-hub

python3 - <<EOF
from huggingface_hub import hf_hub_download
import os

token = os.environ.get('HUGGING_FACE_HUB_TOKEN', '')

print("Downloading diffusion model...")
if not os.path.exists("/app/models/diffusion_models/z_image_turbo_bf16.safetensors"):
    hf_hub_download(
        repo_id="Comfy-Org/z_image_turbo",
        filename="split_files/diffusion_models/z_image_turbo_bf16.safetensors",
        local_dir="/app/models",
        token=token if token else None
    )
    print("Diffusion model downloaded")
else:
    print("Diffusion model already exists")

print("Downloading text encoder...")
if not os.path.exists("/app/models/text_encoders/qwen_3_4b.safetensors"):
    hf_hub_download(
        repo_id="Comfy-Org/z_image_turbo",
        filename="split_files/text_encoders/qwen_3_4b.safetensors",
        local_dir="/app/models",
        token=token if token else None
    )
    print("Text encoder downloaded")
else:
    print("Text encoder already exists")

print("Downloading VAE...")
if not os.path.exists("/app/models/vae/ae.safetensors"):
    hf_hub_download(
        repo_id="Comfy-Org/z_image_turbo",
        filename="split_files/vae/ae.safetensors",
        local_dir="/app/models",
        token=token if token else None
    )
    print("VAE downloaded")
else:
    print("VAE already exists")

print("All models ready!")
EOF

echo "Starting ComfyUI..."
python3 main.py --listen 0.0.0.0 --port 8188
