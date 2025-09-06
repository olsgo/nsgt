"""
Brush Kernels - GPU-accelerated brush effects for spectrogram editing
Implements blur, warp, and other brush effects using PyTorch operations.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class BrushKernels:
    """GPU-accelerated brush effect kernels."""
    
    def __init__(self, device: torch.device):
        self.device = device
        
        # Kernel cache for reuse
        self.gaussian_kernels = {}
        self.displacement_fields = {}
    
    def _create_gaussian_kernel(self, radius: float, sigma: Optional[float] = None) -> torch.Tensor:
        """
        Create a 2D Gaussian kernel.
        
        Args:
            radius: Kernel radius in pixels
            sigma: Standard deviation (defaults to radius/3)
            
        Returns:
            2D Gaussian kernel tensor
        """
        if sigma is None:
            sigma = radius / 3.0
        
        # Cache key
        cache_key = (radius, sigma)
        if cache_key in self.gaussian_kernels:
            return self.gaussian_kernels[cache_key]
        
        # Create kernel
        kernel_size = int(2 * radius + 1)
        center = kernel_size // 2
        
        # Create coordinate grids
        y, x = torch.meshgrid(
            torch.arange(kernel_size, dtype=torch.float32, device=self.device) - center,
            torch.arange(kernel_size, dtype=torch.float32, device=self.device) - center,
            indexing='ij'
        )
        
        # Gaussian formula
        gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum()  # Normalize
        
        # Add batch and channel dimensions for conv2d
        gaussian = gaussian.unsqueeze(0).unsqueeze(0)
        
        self.gaussian_kernels[cache_key] = gaussian
        return gaussian
    
    def _create_pressure_falloff(self, width: int, height: int, 
                               center_x: float, center_y: float, 
                               radius: float, falloff: float = 0.5) -> torch.Tensor:
        """
        Create pressure falloff mask for brush application.
        
        Args:
            width, height: Mask dimensions
            center_x, center_y: Brush center (normalized 0-1)
            radius: Brush radius in pixels
            falloff: Falloff curve steepness
            
        Returns:
            Pressure mask tensor
        """
        # Convert to pixel coordinates
        cx = center_x * width
        cy = center_y * height
        
        # Create coordinate grids
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=self.device),
            torch.arange(width, dtype=torch.float32, device=self.device),
            indexing='ij'
        )
        
        # Distance from center
        distance = torch.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Smooth falloff using smoothstep
        normalized_dist = torch.clamp(distance / radius, 0, 1)
        
        # Smoothstep function for smooth falloff
        t = 1 - normalized_dist
        pressure = t * t * (3 - 2 * t)  # Hermite interpolation
        
        # Apply falloff curve
        pressure = torch.pow(pressure, falloff)
        
        return pressure
    
    async def gaussian_blur(self, magnitude: torch.Tensor, 
                          x: int, y: int, width: int, height: int,
                          radius: float = 5.0, strength: float = 1.0) -> torch.Tensor:
        """
        Apply Gaussian blur to a region of the spectrogram.
        
        Args:
            magnitude: Input magnitude tensor (freq, time)
            x, y: Top-left corner of region
            width, height: Region dimensions
            radius: Blur radius
            strength: Effect strength (0-1)
            
        Returns:
            Blurred region tensor
        """
        # Extract region
        region = magnitude[y:y+height, x:x+width].clone()
        
        if region.numel() == 0:
            return region
        
        # Add batch and channel dimensions for convolution
        region_4d = region.unsqueeze(0).unsqueeze(0)
        
        # Create Gaussian kernel
        gaussian_kernel = self._create_gaussian_kernel(radius)
        
        # Apply padding to handle edges
        pad_size = int(radius)
        region_padded = F.pad(region_4d, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        
        # Apply Gaussian blur
        blurred = F.conv2d(region_padded, gaussian_kernel, padding=0)
        
        # Remove padding
        blurred = blurred[:, :, pad_size:-pad_size, pad_size:-pad_size]
        
        # Remove batch and channel dimensions
        blurred = blurred.squeeze(0).squeeze(0)
        
        # Blend with original based on strength
        result = (1 - strength) * region + strength * blurred
        
        return result
    
    async def displacement_warp(self, magnitude: torch.Tensor,
                              x: int, y: int, width: int, height: int,
                              displacement: float = 1.0, 
                              falloff: float = 0.5,
                              direction: Tuple[float, float] = (1.0, 0.0)) -> torch.Tensor:
        """
        Apply displacement warp effect to a region.
        
        Args:
            magnitude: Input magnitude tensor
            x, y: Top-left corner of region
            width, height: Region dimensions
            displacement: Maximum displacement in pixels
            falloff: Falloff curve steepness
            direction: Displacement direction (dx, dy)
            
        Returns:
            Warped region tensor
        """
        # Extract region
        region = magnitude[y:y+height, x:x+width].clone()
        
        if region.numel() == 0:
            return region
        
        # Create displacement field
        h, w = region.shape
        
        # Normalize direction vector
        dx, dy = direction
        norm = math.sqrt(dx*dx + dy*dy)
        if norm > 0:
            dx, dy = dx/norm, dy/norm
        
        # Create coordinate grids
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=self.device),
            torch.arange(w, dtype=torch.float32, device=self.device),
            indexing='ij'
        )
        
        # Create radial falloff from center
        center_x, center_y = w / 2, h / 2
        dist_x = grid_x - center_x
        dist_y = grid_y - center_y
        distance = torch.sqrt(dist_x**2 + dist_y**2)
        max_dist = min(w, h) / 2
        
        # Smooth falloff
        normalized_dist = torch.clamp(distance / max_dist, 0, 1)
        falloff_mask = 1 - normalized_dist**falloff
        
        # Apply displacement
        displacement_x = dx * displacement * falloff_mask
        displacement_y = dy * displacement * falloff_mask
        
        # Create sampling grid for grid_sample
        new_x = grid_x + displacement_x
        new_y = grid_y + displacement_y
        
        # Normalize coordinates to [-1, 1] for grid_sample
        new_x = 2 * new_x / (w - 1) - 1
        new_y = 2 * new_y / (h - 1) - 1
        
        # Stack coordinates
        grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)
        
        # Add batch and channel dimensions
        region_4d = region.unsqueeze(0).unsqueeze(0)
        
        # Apply displacement using grid_sample
        warped = F.grid_sample(
            region_4d, grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )
        
        # Remove batch and channel dimensions
        warped = warped.squeeze(0).squeeze(0)
        
        return warped
    
    async def spectral_filter(self, magnitude: torch.Tensor,
                            x: int, y: int, width: int, height: int,
                            filter_type: str = "lowpass",
                            cutoff: float = 0.5,
                            strength: float = 1.0) -> torch.Tensor:
        """
        Apply frequency-domain filtering.
        
        Args:
            magnitude: Input magnitude tensor
            x, y: Top-left corner of region
            width, height: Region dimensions
            filter_type: "lowpass", "highpass", "bandpass", "notch"
            cutoff: Filter cutoff frequency (0-1)
            strength: Effect strength
            
        Returns:
            Filtered region tensor
        """
        # Extract region
        region = magnitude[y:y+height, x:x+width].clone()
        
        if region.numel() == 0:
            return region
        
        h, w = region.shape
        
        # Create frequency coordinates
        freq_y = torch.fft.fftfreq(h, device=self.device).unsqueeze(1)
        freq_x = torch.fft.fftfreq(w, device=self.device).unsqueeze(0)
        
        # Radial frequency
        freq_mag = torch.sqrt(freq_x**2 + freq_y**2)
        
        # Create filter
        if filter_type == "lowpass":
            filter_mask = (freq_mag <= cutoff).float()
        elif filter_type == "highpass":
            filter_mask = (freq_mag >= cutoff).float()
        elif filter_type == "bandpass":
            # Assume cutoff is center, with fixed bandwidth
            bandwidth = 0.1
            filter_mask = ((freq_mag >= cutoff - bandwidth/2) & 
                          (freq_mag <= cutoff + bandwidth/2)).float()
        elif filter_type == "notch":
            # Notch filter (inverse bandpass)
            bandwidth = 0.05
            filter_mask = ~((freq_mag >= cutoff - bandwidth/2) & 
                           (freq_mag <= cutoff + bandwidth/2))
            filter_mask = filter_mask.float()
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        # Apply filter in frequency domain
        region_fft = torch.fft.fft2(region)
        filtered_fft = region_fft * filter_mask
        filtered = torch.fft.ifft2(filtered_fft).real
        
        # Blend with original
        result = (1 - strength) * region + strength * filtered
        
        return result
    
    async def magnitude_enhance(self, magnitude: torch.Tensor,
                              x: int, y: int, width: int, height: int,
                              enhancement: float = 1.5,
                              preserve_energy: bool = True) -> torch.Tensor:
        """
        Enhance magnitude values in a region.
        
        Args:
            magnitude: Input magnitude tensor
            x, y: Top-left corner of region
            width, height: Region dimensions
            enhancement: Enhancement factor
            preserve_energy: Whether to preserve total energy
            
        Returns:
            Enhanced region tensor
        """
        # Extract region
        region = magnitude[y:y+height, x:x+width].clone()
        
        if region.numel() == 0:
            return region
        
        if preserve_energy:
            original_energy = region.sum()
        
        # Apply enhancement (power law)
        enhanced = torch.pow(region, 1.0 / enhancement)
        
        if preserve_energy:
            # Scale to preserve energy
            new_energy = enhanced.sum()
            if new_energy > 0:
                enhanced = enhanced * (original_energy / new_energy)
        
        return enhanced
    
    async def adaptive_threshold(self, magnitude: torch.Tensor,
                               x: int, y: int, width: int, height: int,
                               threshold_percentile: float = 0.8,
                               strength: float = 1.0) -> torch.Tensor:
        """
        Apply adaptive thresholding to enhance prominent features.
        
        Args:
            magnitude: Input magnitude tensor
            x, y: Top-left corner of region
            width, height: Region dimensions
            threshold_percentile: Percentile for threshold calculation
            strength: Effect strength
            
        Returns:
            Thresholded region tensor
        """
        # Extract region
        region = magnitude[y:y+height, x:x+width].clone()
        
        if region.numel() == 0:
            return region
        
        # Calculate adaptive threshold
        threshold = torch.quantile(region, threshold_percentile)
        
        # Apply soft thresholding
        enhanced = torch.where(
            region > threshold,
            region,
            region * 0.3  # Attenuate below threshold
        )
        
        # Blend with original
        result = (1 - strength) * region + strength * enhanced
        
        return result