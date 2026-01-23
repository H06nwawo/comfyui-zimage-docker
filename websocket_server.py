#!/usr/bin/env python3
import asyncio
import websockets
import json
import aiohttp
import base64
from datetime import datetime
import os

COMFYUI_URL = "http://localhost:8188"

async def generate_image(websocket, prompt_text, width=1024, height=1024):
    """Genera imagen y envía actualizaciones de progreso por WebSocket"""
    try:
        # Enviar estado inicial
        await websocket.send(json.dumps({
            "type": "status",
            "message": "Iniciando generación..."
        }))

        # Workflow de ComfyUI
        workflow = {
            "1": {
                "inputs": {
                    "unet_name": "z_image_turbo_bf16.safetensors",
                    "weight_dtype": "default"
                },
                "class_type": "UNETLoader"
            },
            "2": {
                "inputs": {
                    "clip_name": "qwen_3_4b.safetensors",
                    "type": "lumina2"
                },
                "class_type": "CLIPLoader"
            },
            "3": {
                "inputs": {
                    "vae_name": "ae.safetensors"
                },
                "class_type": "VAELoader"
            },
            "4": {
                "inputs": {
                    "lora_name": "zimage_uncensored.safetensors",
                    "strength_model": 0.9,
                    "strength_clip": 0.9,
                    "model": ["1", 0],
                    "clip": ["2", 0]
                },
                "class_type": "LoraLoader"
            },
            "5": {
                "inputs": {
                    "text": prompt_text,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "6": {
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            "7": {
                "inputs": {
                    "seed": int(datetime.now().timestamp() * 1000),
                    "steps": 8,
                    "cfg": 0.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["5", 0],
                    "negative": ["5", 0],
                    "latent_image": ["6", 0]
                },
                "class_type": "KSampler"
            },
            "8": {
                "inputs": {
                    "samples": ["7", 0],
                    "vae": ["3", 0]
                },
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {
                    "filename_prefix": "zimage",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            }
        }

        # Enviar workflow a ComfyUI
        async with aiohttp.ClientSession() as session:
            await websocket.send(json.dumps({
                "type": "status",
                "message": "Enviando prompt a ComfyUI..."
            }))

            async with session.post(
                f"{COMFYUI_URL}/prompt",
                json={"prompt": workflow}
            ) as response:
                if response.status != 200:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Error al enviar prompt: {response.status}"
                    }))
                    return

                result = await response.json()
                prompt_id = result["prompt_id"]

            await websocket.send(json.dumps({
                "type": "status",
                "message": "Generando imagen...",
                "prompt_id": prompt_id
            }))

            # Polling interno (no visible al cliente) para obtener progreso
            for attempt in range(60):
                await asyncio.sleep(2)
                
                # Enviar progreso estimado
                progress_percent = min(95, (attempt + 1) * 100 / 30)
                await websocket.send(json.dumps({
                    "type": "progress",
                    "percent": int(progress_percent),
                    "step": attempt + 1,
                    "total_steps": 8
                }))

                # Verificar si está completo
                async with session.get(f"{COMFYUI_URL}/history/{prompt_id}") as hist_response:
                    if hist_response.status == 200:
                        history = await hist_response.json()
                        
                        if prompt_id in history and history[prompt_id].get("outputs"):
                            outputs = history[prompt_id]["outputs"]
                            
                            # Buscar la imagen generada
                            for output in outputs.values():
                                if output.get("images") and len(output["images"]) > 0:
                                    image_info = output["images"][0]
                                    filename = image_info["filename"]
                                    subfolder = image_info.get("subfolder", "")
                                    image_type = image_info.get("type", "output")
                                    
                                    # Descargar imagen
                                    image_url = f"{COMFYUI_URL}/view?filename={filename}&subfolder={subfolder}&type={image_type}"
                                    
                                    async with session.get(image_url) as img_response:
                                        if img_response.status == 200:
                                            image_bytes = await img_response.read()
                                            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                                            
                                            # Enviar imagen completa
                                            await websocket.send(json.dumps({
                                                "type": "complete",
                                                "image": image_base64,
                                                "prompt_id": prompt_id
                                            }))
                                            return

            # Timeout
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Timeout: la generación tardó demasiado"
            }))

    except Exception as e:
        await websocket.send(json.dumps({
            "type": "error",
            "message": f"Error: {str(e)}"
        }))


async def handle_client(websocket, path):
    """Maneja conexiones de clientes WebSocket"""
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if data.get("type") == "generate":
                    prompt = data.get("prompt", "")
                    width = data.get("width", 1024)
                    height = data.get("height", 1024)
                    
                    if not prompt:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "Prompt vacío"
                        }))
                        continue
                    
                    await generate_image(websocket, prompt, width, height)
                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Tipo de mensaje desconocido"
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "JSON inválido"
                }))
                
    except websockets.exceptions.ConnectionClosed:
        print("Cliente desconectado")


async def main():
    """Inicia servidor WebSocket"""
    print("Iniciando servidor WebSocket en ws://0.0.0.0:8765")
    async with websockets.serve(handle_client, "0.0.0.0", 8765):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
