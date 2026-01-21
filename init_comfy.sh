#!/bin/bash

set -e

echo "Downloading Z-Image-Turbo model..."

if [ ! -f "/app/models/checkpoints/z_image_turbo.safetensors" ]; then
    pip3 install huggingface-hub
    python3 - <<EOF
from huggingface_hub import hf_hub_download
import os

token = os.environ.get('HUGGING_FACE_HUB_TOKEN', '')

hf_hub_download(
    repo_id="Tongyi-MAI/Z-Image-Turbo",
    filename="z_image_turbo.safetensors",
    local_dir="/app/models/checkpoints",
    token=token if token else None
)
EOF
    echo "Model downloaded successfully"
else
    echo "Model already exists, skipping download"
fi

echo "Starting ComfyUI..."
python3 main.py --listen 0.0.0.0 --port 8188
