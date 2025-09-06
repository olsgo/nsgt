"""
NSGT Processor - Integration with Non-Stationary Gabor Transform
Provides GPU-accelerated NSGT operations for real-time spectrogram editing.
"""

import asyncio
import numpy as np
import torch
from typing import Tuple, Optional

# Import NSGT from the parent package
import sys
sys.path.append('../..')
from nsgt import NSGT_sliced, CQ_NSGT, LogScale, LinScale, MelScale, OctScale


class NSGTProcessor:
    """GPU-accelerated NSGT processor for real-time applications."""
    
    def __init__(self, device: torch.device, sr: int = 44100, 
                 fmin: float = 80, fmax: float = 8000, bins: int = 12):
        self.device = device
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.bins = bins
        
        # Initialize NSGT transform
        self.scale = LogScale(fmin, fmax, bins)
        self.nsgt = None
        self.nsgt_sliced = None
        
        # Cache for frequency bins and windows
        self.freq_bins = None
        self.windows = None
        
    async def initialize_transform(self, signal_length: int):
        """Initialize NSGT transform for given signal length."""
        try:
            # Initialize standard NSGT
            self.nsgt = CQ_NSGT(
                self.fmin, self.fmax, self.bins, 
                self.sr, signal_length, real=True
            )
            
            # Initialize sliced NSGT for streaming
            self.nsgt_sliced = NSGT_sliced(
                self.scale, self.sr, signal_length//4,  # slice length
                real=True, multichannel=True
            )
            
            # Cache frequency information
            self.freq_bins = self.nsgt.frqs
            
            return True
            
        except Exception as e:
            print(f"Error initializing NSGT: {e}")
            return False
    
    async def forward(self, audio_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward NSGT transform.
        
        Args:
            audio_tensor: Input audio tensor (channels, samples)
            
        Returns:
            Tuple of (magnitude, phase) tensors on GPU
        """
        # Convert to numpy for NSGT processing
        audio_np = audio_tensor.cpu().numpy()
        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]  # Add channel dimension
        
        # Initialize transform if needed
        if self.nsgt is None:
            await self.initialize_transform(audio_np.shape[-1])
        
        # Process each channel
        magnitude_list = []
        phase_list = []
        
        for ch in range(audio_np.shape[0]):
            # Forward transform
            coeffs = self.nsgt.forward(audio_np[ch])
            
            # Convert coefficients to magnitude/phase
            magnitude_ch = []
            phase_ch = []
            
            for coeff_band in coeffs:
                if coeff_band is not None:
                    mag = np.abs(coeff_band)
                    phase = np.angle(coeff_band)
                    magnitude_ch.append(mag)
                    phase_ch.append(phase)
            
            magnitude_list.append(np.array(magnitude_ch))
            phase_list.append(np.array(phase_ch))
        
        # Convert to tensors and move to GPU
        magnitude = torch.from_numpy(np.array(magnitude_list)).float().to(self.device)
        phase = torch.from_numpy(np.array(phase_list)).float().to(self.device)
        
        return magnitude, phase
    
    async def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        Inverse NSGT transform.
        
        Args:
            magnitude: Magnitude tensor
            phase: Phase tensor
            
        Returns:
            Reconstructed audio tensor
        """
        if self.nsgt is None:
            raise ValueError("NSGT not initialized")
        
        # Convert back to numpy
        mag_np = magnitude.cpu().numpy()
        phase_np = phase.cpu().numpy()
        
        # Reconstruct each channel
        audio_channels = []
        
        for ch in range(mag_np.shape[0]):
            # Reconstruct complex coefficients
            complex_coeffs = []
            for i in range(mag_np.shape[1]):
                complex_coeff = mag_np[ch, i] * np.exp(1j * phase_np[ch, i])
                complex_coeffs.append(complex_coeff)
            
            # Inverse transform
            audio_ch = self.nsgt.backward(complex_coeffs)
            audio_channels.append(audio_ch)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(np.array(audio_channels)).float().to(self.device)
        return audio_tensor
    
    async def forward_sliced(self, audio_stream):
        """
        Process audio stream with sliced NSGT for real-time applications.
        
        Args:
            audio_stream: Generator yielding audio chunks
            
        Yields:
            Spectrogram slices (magnitude, phase)
        """
        if self.nsgt_sliced is None:
            raise ValueError("Sliced NSGT not initialized")
        
        for audio_chunk in audio_stream:
            # Convert to numpy
            chunk_np = audio_chunk.cpu().numpy()
            if chunk_np.ndim == 1:
                chunk_np = chunk_np[np.newaxis, :]
            
            # Process with sliced NSGT
            for ch in range(chunk_np.shape[0]):
                coeffs = self.nsgt_sliced.forward(chunk_np[ch])
                
                # Convert to magnitude/phase
                magnitude_ch = []
                phase_ch = []
                
                for coeff_band in coeffs:
                    if coeff_band is not None:
                        mag = np.abs(coeff_band)
                        phase = np.angle(coeff_band)
                        magnitude_ch.append(mag)
                        phase_ch.append(phase)
                
                magnitude = torch.from_numpy(np.array(magnitude_ch)).float().to(self.device)
                phase = torch.from_numpy(np.array(phase_ch)).float().to(self.device)
                
                yield magnitude, phase
    
    def get_frequency_bins(self) -> Optional[np.ndarray]:
        """Get frequency bin centers."""
        return self.freq_bins
    
    def get_transform_info(self) -> dict:
        """Get information about the current transform."""
        return {
            "sample_rate": self.sr,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "bins_per_octave": self.bins,
            "frequency_bins": self.freq_bins.tolist() if self.freq_bins is not None else None,
            "num_bins": len(self.freq_bins) if self.freq_bins is not None else 0
        }
    
    async def precompute_lookup_table(self, target_height: int) -> torch.Tensor:
        """
        Precompute lookup table for mapping NSGT bins to uniform grid.
        
        Args:
            target_height: Target height for visualization grid
            
        Returns:
            Lookup tensor for bin mapping
        """
        if self.freq_bins is None:
            raise ValueError("NSGT not initialized")
        
        # Create uniform frequency grid for visualization
        freq_min = self.freq_bins[0]
        freq_max = self.freq_bins[-1]
        uniform_freqs = np.logspace(
            np.log10(freq_min), np.log10(freq_max), target_height
        )
        
        # Create mapping from NSGT bins to uniform grid
        lookup = np.zeros(len(self.freq_bins), dtype=np.int32)
        
        for i, freq in enumerate(self.freq_bins):
            # Find closest bin in uniform grid
            idx = np.argmin(np.abs(uniform_freqs - freq))
            lookup[i] = idx
        
        return torch.from_numpy(lookup).to(self.device)