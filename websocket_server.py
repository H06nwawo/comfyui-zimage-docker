#!/usr/bin/env python3
import asyncio
import websockets
import json
import aiohttp
import base64
from typing import Set, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComfyUIWebSocketServer:
    def __init__(self, comfyui_host: str = "127.0.0.1", comfyui_port: int = 8188):
        self.comfyui_base = f"http://{comfyui_host}:{comfyui_port}"
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.active_generations: Dict[str, str] = {}  # prompt_id -> client_id
        
    async def register_client(self, websocket: websockets.WebSocketServerProtocol):
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol):
        self.clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
        
    async def send_to_client(self, websocket: websockets.WebSocketServerProtocol, message: dict):
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            await self.unregister_client(websocket)
            
    async def broadcast(self, message: dict):
        if self.clients:
            await asyncio.gather(
                *[self.send_to_client(client, message) for client in self.clients],
                return_exceptions=True
            )
            
    async def generate_image(self, websocket: websockets.WebSocketServerProtocol, data: dict):
        prompt_text = data.get('prompt', '')
        width = data.get('width', 1024)
        height = data.get('height', 1024)
        client_id = data.get('client_id', 'unknown')
        
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
                    "seed": int(asyncio.get_event_loop().time() * 1000),
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
        
        try:
            await self.send_to_client(websocket, {
                'type': 'generation_started',
                'client_id': client_id,
                'prompt': prompt_text
            })
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.comfyui_base}/prompt",
                    json={'prompt': workflow},
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status != 200:
                        await self.send_to_client(websocket, {
                            'type': 'generation_error',
                            'client_id': client_id,
                            'error': f'ComfyUI returned status {response.status}'
                        })
                        return
                        
                    result = await response.json()
                    prompt_id = result.get('prompt_id')
                    
                    if not prompt_id:
                        await self.send_to_client(websocket, {
                            'type': 'generation_error',
                            'client_id': client_id,
                            'error': 'No prompt_id received'
                        })
                        return
                    
                    self.active_generations[prompt_id] = client_id
                    
                    await self.send_to_client(websocket, {
                        'type': 'generation_progress',
                        'client_id': client_id,
                        'prompt_id': prompt_id,
                        'status': 'processing'
                    })
                    
                    for attempt in range(60):
                        await asyncio.sleep(2)
                        
                        async with session.get(
                            f"{self.comfyui_base}/history/{prompt_id}"
                        ) as hist_response:
                            if hist_response.status == 200:
                                history = await hist_response.json()
                                
                                if prompt_id in history and history[prompt_id].get('outputs'):
                                    outputs = history[prompt_id]['outputs']
                                    
                                    for output in outputs.values():
                                        if output.get('images') and len(output['images']) > 0:
                                            image_info = output['images'][0]
                                            filename = image_info['filename']
                                            subfolder = image_info.get('subfolder', '')
                                            img_type = image_info.get('type', 'output')
                                            
                                            image_url = f"{self.comfyui_base}/view?filename={filename}&subfolder={subfolder}&type={img_type}"
                                            
                                            async with session.get(image_url) as img_response:
                                                if img_response.status == 200:
                                                    image_bytes = await img_response.read()
                                                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                                                    
                                                    await self.send_to_client(websocket, {
                                                        'type': 'generation_complete',
                                                        'client_id': client_id,
                                                        'prompt_id': prompt_id,
                                                        'image': image_base64
                                                    })
                                                    
                                                    if prompt_id in self.active_generations:
                                                        del self.active_generations[prompt_id]
                                                    
                                                    return
                    
                    await self.send_to_client(websocket, {
                        'type': 'generation_error',
                        'client_id': client_id,
                        'error': 'Generation timeout after 120 seconds'
                    })
                    
                    if prompt_id in self.active_generations:
                        del self.active_generations[prompt_id]
                        
        except asyncio.TimeoutError:
            await self.send_to_client(websocket, {
                'type': 'generation_error',
                'client_id': client_id,
                'error': 'Request timeout'
            })
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            await self.send_to_client(websocket, {
                'type': 'generation_error',
                'client_id': client_id,
                'error': str(e)
            })
            
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get('type')
                    
                    if msg_type == 'generate_image':
                        asyncio.create_task(self.generate_image(websocket, data))
                    elif msg_type == 'ping':
                        await self.send_to_client(websocket, {'type': 'pong'})
                        
                except json.JSONDecodeError:
                    await self.send_to_client(websocket, {
                        'type': 'error',
                        'error': 'Invalid JSON'
                    })
                except Exception as e:
                    logger.error(f"Error handling message: {str(e)}")
                    await self.send_to_client(websocket, {
                        'type': 'error',
                        'error': str(e)
                    })
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)
            
    async def start(self, host: str = "0.0.0.0", port: int = 8189):
        logger.info(f"Starting WebSocket server on {host}:{port}")
        async with websockets.serve(self.handle_client, host, port, ping_interval=30, ping_timeout=10):
            await asyncio.Future()

async def main():
    server = ComfyUIWebSocketServer()
    await server.start()

if __name__ == "__main__":
    asyncio.run(main())
