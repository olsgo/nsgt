"""
Texture Bridge - Interface between GPU tensors and Flutter textures
Handles conversion and streaming of spectrogram data to Flutter UI.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
import uuid
import io
import base64


class TextureBridge:
    """Bridge between PyTorch tensors and Flutter texture system."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.active_textures = {}
        self.texture_cache = {}
        
        # Texture format settings
        self.texture_format = "RGBA8"
        self.colormap = "viridis"  # Default colormap
        
    def tensor_to_texture(self, magnitude: torch.Tensor, phase: torch.Tensor) -> Dict[str, Any]:
        """
        Convert magnitude/phase tensors to texture format.
        
        Args:
            magnitude: Magnitude tensor (batch, freq, time) or (freq, time)
            phase: Phase tensor (same shape as magnitude)
            
        Returns:
            Dictionary with texture information
        """
        # Ensure tensors are 2D
        if magnitude.dim() > 2:
            magnitude = magnitude.squeeze()
        if phase.dim() > 2:
            phase = phase.squeeze()
        
        # Normalize magnitude to [0, 1]
        mag_min, mag_max = magnitude.min(), magnitude.max()
        if mag_max > mag_min:
            magnitude_norm = (magnitude - mag_min) / (mag_max - mag_min)
        else:
            magnitude_norm = torch.zeros_like(magnitude)
        
        # Normalize phase to [0, 1] from [-π, π]
        phase_norm = (phase + torch.pi) / (2 * torch.pi)
        
        # Create RGBA texture
        height, width = magnitude_norm.shape
        rgba_texture = torch.zeros((height, width, 4), dtype=torch.float32, device=self.device)
        
        # Pack data into RGBA channels
        rgba_texture[..., 0] = magnitude_norm  # R: magnitude
        rgba_texture[..., 1] = phase_norm      # G: phase
        rgba_texture[..., 2] = 0.0             # B: reserved
        rgba_texture[..., 3] = 1.0             # A: alpha
        
        # Convert to 8-bit RGBA
        rgba_8bit = (rgba_texture * 255).clamp(0, 255).byte()
        
        # Generate texture ID
        texture_id = str(uuid.uuid4())
        
        # Store texture data
        texture_data = {
            "texture_id": texture_id,
            "dimensions": {"width": width, "height": height},
            "format": self.texture_format,
            "data": rgba_8bit,
            "magnitude_range": {"min": mag_min.item(), "max": mag_max.item()},
            "phase_range": {"min": -torch.pi.item(), "max": torch.pi.item()}
        }
        
        self.active_textures[texture_id] = texture_data
        return texture_data
    
    def update_region(self, magnitude: torch.Tensor, phase: torch.Tensor, 
                     x: int, y: int, width: int, height: int) -> Dict[str, Any]:
        """
        Update a region of the texture.
        
        Args:
            magnitude: Updated magnitude tensor
            phase: Updated phase tensor
            x, y: Top-left corner of region
            width, height: Region dimensions
            
        Returns:
            Updated texture information
        """
        # For now, regenerate the entire texture
        # TODO: Optimize for partial updates
        return self.tensor_to_texture(magnitude, phase)
    
    def apply_colormap(self, magnitude: torch.Tensor, colormap: str = "viridis") -> torch.Tensor:
        """
        Apply colormap to magnitude data.
        
        Args:
            magnitude: Normalized magnitude tensor [0, 1]
            colormap: Colormap name
            
        Returns:
            RGB tensor (height, width, 3)
        """
        # Simple viridis-like colormap implementation
        if colormap == "viridis":
            # Create a simple 3-color interpolation (purple -> blue -> green -> yellow)
            rgb = torch.zeros((*magnitude.shape, 3), dtype=torch.float32, device=self.device)
            
            # Purple to blue (0.0 to 0.25)
            mask1 = magnitude <= 0.25
            t1 = magnitude / 0.25
            rgb[mask1, 0] = 0.267 * (1 - t1[mask1])  # Purple to blue (R)
            rgb[mask1, 1] = 0.004 + 0.3 * t1[mask1]  # Purple to blue (G)
            rgb[mask1, 2] = 0.329 + 0.5 * t1[mask1]  # Purple to blue (B)
            
            # Blue to green (0.25 to 0.5)
            mask2 = (magnitude > 0.25) & (magnitude <= 0.5)
            t2 = (magnitude - 0.25) / 0.25
            rgb[mask2, 0] = 0.0                      # Blue to green (R)
            rgb[mask2, 1] = 0.3 + 0.4 * t2[mask2]   # Blue to green (G)
            rgb[mask2, 2] = 0.8 - 0.3 * t2[mask2]   # Blue to green (B)
            
            # Green to yellow (0.5 to 1.0)
            mask3 = magnitude > 0.5
            t3 = (magnitude - 0.5) / 0.5
            rgb[mask3, 0] = 0.9 * t3[mask3]         # Green to yellow (R)
            rgb[mask3, 1] = 0.7 + 0.3 * t3[mask3]   # Green to yellow (G)
            rgb[mask3, 2] = 0.5 * (1 - t3[mask3])   # Green to yellow (B)
            
        elif colormap == "hot":
            # Hot colormap (black -> red -> yellow -> white)
            rgb = torch.zeros((*magnitude.shape, 3), dtype=torch.float32, device=self.device)
            
            # Black to red
            mask1 = magnitude <= 0.33
            t1 = magnitude / 0.33
            rgb[mask1, 0] = t1[mask1]
            
            # Red to yellow
            mask2 = (magnitude > 0.33) & (magnitude <= 0.66)
            t2 = (magnitude - 0.33) / 0.33
            rgb[mask2, 0] = 1.0
            rgb[mask2, 1] = t2[mask2]
            
            # Yellow to white
            mask3 = magnitude > 0.66
            t3 = (magnitude - 0.66) / 0.34
            rgb[mask3, 0] = 1.0
            rgb[mask3, 1] = 1.0
            rgb[mask3, 2] = t3[mask3]
            
        else:
            # Fallback: grayscale
            rgb = magnitude.unsqueeze(-1).repeat(1, 1, 3)
        
        return rgb
    
    def tensor_to_image_data(self, magnitude: torch.Tensor, 
                           colormap: str = "viridis", 
                           log_scale: bool = True) -> Dict[str, Any]:
        """
        Convert tensor to image data for display.
        
        Args:
            magnitude: Magnitude tensor
            colormap: Colormap to apply
            log_scale: Whether to apply log scaling
            
        Returns:
            Image data dictionary
        """
        # Ensure 2D tensor
        if magnitude.dim() > 2:
            magnitude = magnitude.squeeze()
        
        # Apply log scaling if requested
        if log_scale:
            magnitude = torch.log1p(magnitude)  # log(1 + x) for numerical stability
        
        # Normalize to [0, 1]
        mag_min, mag_max = magnitude.min(), magnitude.max()
        if mag_max > mag_min:
            magnitude_norm = (magnitude - mag_min) / (mag_max - mag_min)
        else:
            magnitude_norm = torch.zeros_like(magnitude)
        
        # Apply colormap
        rgb_image = self.apply_colormap(magnitude_norm, colormap)
        
        # Convert to 8-bit
        rgb_8bit = (rgb_image * 255).clamp(0, 255).byte()
        
        # Convert to numpy for image encoding
        rgb_np = rgb_8bit.cpu().numpy()
        
        # Flip vertically for typical image display (frequency increases upward)
        rgb_np = np.flipud(rgb_np)
        
        return {
            "width": rgb_np.shape[1],
            "height": rgb_np.shape[0],
            "channels": 3,
            "data": rgb_np,
            "colormap": colormap,
            "log_scale": log_scale,
            "value_range": {"min": mag_min.item(), "max": mag_max.item()}
        }
    
    def encode_image_base64(self, image_data: Dict[str, Any], format: str = "PNG") -> str:
        """
        Encode image data as base64 string.
        
        Args:
            image_data: Image data from tensor_to_image_data
            format: Image format (PNG, JPEG)
            
        Returns:
            Base64 encoded image string
        """
        try:
            from PIL import Image
            
            # Create PIL image
            rgb_array = image_data["data"]
            pil_image = Image.fromarray(rgb_array, mode="RGB")
            
            # Encode to bytes
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format)
            
            # Encode as base64
            image_bytes = buffer.getvalue()
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            
            return f"data:image/{format.lower()};base64,{base64_string}"
            
        except ImportError:
            # Fallback: return raw data info
            return f"Raw image data: {image_data['width']}x{image_data['height']} RGB"
    
    def get_texture_info(self, texture_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a texture."""
        return self.active_textures.get(texture_id)
    
    def release_texture(self, texture_id: str) -> bool:
        """Release a texture from memory."""
        if texture_id in self.active_textures:
            del self.active_textures[texture_id]
            return True
        return False
    
    def list_active_textures(self) -> list:
        """List all active texture IDs."""
        return list(self.active_textures.keys())
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        total_pixels = 0
        texture_count = len(self.active_textures)
        
        for texture_data in self.active_textures.values():
            dims = texture_data["dimensions"]
            total_pixels += dims["width"] * dims["height"]
        
        # Estimate memory usage (4 bytes per pixel for RGBA)
        memory_mb = (total_pixels * 4) / (1024 * 1024)
        
        return {
            "texture_count": texture_count,
            "total_pixels": total_pixels,
            "estimated_memory_mb": memory_mb,
            "device": str(self.device)
        }