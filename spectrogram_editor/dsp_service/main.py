#!/usr/bin/env python3
"""
Real-time Spectrogram Editor - DSP Service
Main entry point for the GPU-accelerated DSP backend.
"""

import asyncio
import logging
import argparse
from pathlib import Path

import torch
import uvloop
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from nsgt_processor import NSGTProcessor
from nnaudio_gpu import NNAudioProcessor
from texture_bridge import TextureBridge
from brush_kernels import BrushKernels

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app for REST/WebSocket API
app = FastAPI(title="Spectrogram Editor DSP Service", version="1.0.0")

# CORS for Flutter web debugging
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global processors
nsgt_processor = None
nnaudio_processor = None
texture_bridge = None
brush_kernels = None


class SpectrogramEditorService:
    """Main service class coordinating NSGT, nnAudio, and GPU operations."""
    
    def __init__(self, device: str = "cuda", sample_rate: int = 44100):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        
        logger.info(f"Initializing DSP service on device: {self.device}")
        
        # Initialize processors
        self.nsgt_processor = NSGTProcessor(device=self.device, sr=sample_rate)
        self.nnaudio_processor = NNAudioProcessor(device=self.device, sr=sample_rate)
        self.texture_bridge = TextureBridge(device=self.device)
        self.brush_kernels = BrushKernels(device=self.device)
        
        # Current working tensors (magnitude, phase)
        self.magnitude_tensor = None
        self.phase_tensor = None
        self.original_tensor = None
        
        logger.info("DSP service initialized successfully")
    
    async def generate_spectrogram(self, audio_data: torch.Tensor, transform_type: str = "nsgt"):
        """Generate spectrogram using NSGT or nnAudio transforms."""
        try:
            if transform_type == "nsgt":
                magnitude, phase = await self.nsgt_processor.forward(audio_data)
            elif transform_type == "stft":
                magnitude, phase = await self.nnaudio_processor.stft(audio_data)
            elif transform_type == "mel":
                magnitude, phase = await self.nnaudio_processor.mel_spectrogram(audio_data)
            elif transform_type == "cqt":
                magnitude, phase = await self.nnaudio_processor.cqt(audio_data)
            else:
                raise ValueError(f"Unsupported transform type: {transform_type}")
            
            # Store working tensors
            self.magnitude_tensor = magnitude
            self.phase_tensor = phase
            self.original_tensor = magnitude.clone()
            
            # Convert to texture format
            texture_data = self.texture_bridge.tensor_to_texture(magnitude, phase)
            
            return {
                "success": True,
                "texture_id": texture_data["texture_id"],
                "dimensions": texture_data["dimensions"],
                "format": texture_data["format"]
            }
            
        except Exception as e:
            logger.error(f"Error generating spectrogram: {e}")
            return {"success": False, "error": str(e)}
    
    async def apply_brush(self, brush_type: str, roi: dict, params: dict):
        """Apply brush effect to spectrogram region."""
        try:
            if self.magnitude_tensor is None:
                raise ValueError("No spectrogram loaded")
            
            # Extract ROI bounds
            x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
            
            # Apply brush kernel
            if brush_type == "blur":
                result = await self.brush_kernels.gaussian_blur(
                    self.magnitude_tensor, x, y, w, h, 
                    radius=params.get("radius", 5.0),
                    strength=params.get("strength", 1.0)
                )
            elif brush_type == "warp":
                result = await self.brush_kernels.displacement_warp(
                    self.magnitude_tensor, x, y, w, h,
                    displacement=params.get("displacement", 1.0),
                    falloff=params.get("falloff", 0.5)
                )
            else:
                raise ValueError(f"Unsupported brush type: {brush_type}")
            
            # Update working tensor
            self.magnitude_tensor[y:y+h, x:x+w] = result
            
            # Update texture
            texture_data = self.texture_bridge.update_region(
                self.magnitude_tensor, self.phase_tensor, x, y, w, h
            )
            
            return {
                "success": True,
                "updated_region": {"x": x, "y": y, "width": w, "height": h},
                "texture_id": texture_data["texture_id"]
            }
            
        except Exception as e:
            logger.error(f"Error applying brush: {e}")
            return {"success": False, "error": str(e)}
    
    async def reset_to_original(self):
        """Reset spectrogram to original state."""
        if self.original_tensor is not None:
            self.magnitude_tensor = self.original_tensor.clone()
            texture_data = self.texture_bridge.tensor_to_texture(
                self.magnitude_tensor, self.phase_tensor
            )
            return {"success": True, "texture_id": texture_data["texture_id"]}
        return {"success": False, "error": "No original tensor available"}


# FastAPI endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    global nsgt_processor, nnaudio_processor, texture_bridge, brush_kernels
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
    
    # Initialize service
    service = SpectrogramEditorService()
    nsgt_processor = service.nsgt_processor
    nnaudio_processor = service.nnaudio_processor
    texture_bridge = service.texture_bridge
    brush_kernels = service.brush_kernels
    
    logger.info("DSP service started")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await websocket.accept()
    service = SpectrogramEditorService()
    
    try:
        while True:
            data = await websocket.receive_json()
            command = data.get("command")
            
            if command == "generate_spectrogram":
                # Handle spectrogram generation
                result = await service.generate_spectrogram(
                    torch.tensor(data["audio_data"]).to(service.device),
                    data.get("transform_type", "nsgt")
                )
                await websocket.send_json(result)
                
            elif command == "apply_brush":
                # Handle brush application
                result = await service.apply_brush(
                    data["brush_type"],
                    data["roi"],
                    data["params"]
                )
                await websocket.send_json(result)
                
            elif command == "reset":
                # Reset to original
                result = await service.reset_to_original()
                await websocket.send_json(result)
                
            else:
                await websocket.send_json({
                    "success": False, 
                    "error": f"Unknown command: {command}"
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"success": False, "error": str(e)})


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Spectrogram Editor DSP Service")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--device", default="cuda", help="PyTorch device (cuda/cpu)")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Use uvloop for better performance
    if hasattr(asyncio, 'set_event_loop_policy'):
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    
    # Run server
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower()
    )


if __name__ == "__main__":
    main()