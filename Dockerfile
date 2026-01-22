FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/comfyanonymous/ComfyUI.git . && \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models/diffusion_models && \
    mkdir -p /app/models/text_encoders && \
    mkdir -p /app/models/vae

COPY init_comfy.sh /app/init_comfy.sh
RUN chmod +x /app/init_comfy.sh

ARG CIVITAI_API_KEY
ENV CIVITAI_API_KEY=${CIVITAI_API_KEY}
ENV HUGGING_FACE_HUB_TOKEN=""

EXPOSE 8188

CMD ["/app/init_comfy.sh"]
