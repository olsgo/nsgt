"""
nnAudio GPU Processor - GPU-accelerated spectrogram generation
Provides high-performance STFT, Mel, CQT, and VQT transforms using nnAudio.
"""

import asyncio
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any

try:
    import nnAudio
    from nnAudio import features
    NNAUDIO_AVAILABLE = True
except ImportError:
    print("Warning: nnAudio not available. Install with: pip install nnAudio")
    NNAUDIO_AVAILABLE = False


class NNAudioProcessor:
    """GPU-accelerated spectrogram processor using nnAudio."""
    
    def __init__(self, device: torch.device, sr: int = 44100):
        self.device = device
        self.sr = sr
        
        if not NNAUDIO_AVAILABLE:
            raise ImportError("nnAudio is required for GPU-accelerated processing")
        
        # Initialize transform modules
        self.stft_module = None
        self.mel_module = None
        self.cqt_module = None
        self.vqt_module = None
        
        # Default parameters
        self.default_params = {
            "stft": {
                "n_fft": 2048,
                "hop_length": 512,
                "window": "hann",
                "center": True,
                "pad_mode": "reflect"
            },
            "mel": {
                "n_fft": 2048,
                "hop_length": 512,
                "n_mels": 128,
                "fmin": 0,
                "fmax": None,
                "window": "hann",
                "center": True,
                "pad_mode": "reflect"
            },
            "cqt": {
                "hop_length": 512,
                "fmin": 32.70,  # C1
                "n_bins": 84,
                "bins_per_octave": 12,
                "filter_scale": 1,
                "norm": 1,
                "window": "hann",
                "center": True,
                "pad_mode": "reflect"
            },
            "vqt": {
                "hop_length": 512,
                "fmin": 32.70,
                "n_bins": 84,
                "gamma": 0,
                "bins_per_octave": 12,
                "filter_scale": 1,
                "norm": 1,
                "window": "hann",
                "center": True,
                "pad_mode": "reflect"
            }
        }
        
    def _initialize_stft(self, **kwargs) -> nn.Module:
        """Initialize STFT module."""
        params = {**self.default_params["stft"], **kwargs}
        return features.STFT(
            sr=self.sr,
            **params
        ).to(self.device)
    
    def _initialize_mel(self, **kwargs) -> nn.Module:
        """Initialize Mel spectrogram module."""
        params = {**self.default_params["mel"], **kwargs}
        if params["fmax"] is None:
            params["fmax"] = self.sr // 2
        return features.MelSpectrogram(
            sr=self.sr,
            **params
        ).to(self.device)
    
    def _initialize_cqt(self, **kwargs) -> nn.Module:
        """Initialize CQT module."""
        params = {**self.default_params["cqt"], **kwargs}
        return features.CQT(
            sr=self.sr,
            **params
        ).to(self.device)
    
    def _initialize_vqt(self, **kwargs) -> nn.Module:
        """Initialize VQT module."""
        params = {**self.default_params["vqt"], **kwargs}
        return features.VQT(
            sr=self.sr,
            **params
        ).to(self.device)
    
    async def stft(self, audio_tensor: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Short-Time Fourier Transform.
        
        Args:
            audio_tensor: Input audio tensor (batch, channels, samples) or (channels, samples)
            **kwargs: STFT parameters
            
        Returns:
            Tuple of (magnitude, phase) tensors
        """
        if self.stft_module is None or kwargs:
            self.stft_module = self._initialize_stft(**kwargs)
        
        # Ensure correct tensor dimensions
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
        
        # Compute STFT
        with torch.no_grad():
            stft_result = self.stft_module(audio_tensor)
            
            # nnAudio STFT returns (batch, freq, time, 2) where last dim is [real, imag]
            real_part = stft_result[..., 0]
            imag_part = stft_result[..., 1]
            
            # Compute magnitude and phase
            magnitude = torch.sqrt(real_part**2 + imag_part**2)
            phase = torch.atan2(imag_part, real_part)
            
        return magnitude, phase
    
    async def mel_spectrogram(self, audio_tensor: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Mel spectrogram.
        
        Args:
            audio_tensor: Input audio tensor
            **kwargs: Mel spectrogram parameters
            
        Returns:
            Tuple of (magnitude, phase) tensors
        """
        if self.mel_module is None or kwargs:
            self.mel_module = self._initialize_mel(**kwargs)
        
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        with torch.no_grad():
            mel_spec = self.mel_module(audio_tensor)
            
            # Mel spectrogram is magnitude-only, create dummy phase
            magnitude = mel_spec
            phase = torch.zeros_like(magnitude)
            
        return magnitude, phase
    
    async def cqt(self, audio_tensor: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Constant-Q Transform.
        
        Args:
            audio_tensor: Input audio tensor
            **kwargs: CQT parameters
            
        Returns:
            Tuple of (magnitude, phase) tensors
        """
        if self.cqt_module is None or kwargs:
            self.cqt_module = self._initialize_cqt(**kwargs)
        
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        with torch.no_grad():
            cqt_result = self.cqt_module(audio_tensor)
            
            # CQT returns complex values (batch, freq, time, 2)
            if cqt_result.dim() == 4 and cqt_result.shape[-1] == 2:
                real_part = cqt_result[..., 0]
                imag_part = cqt_result[..., 1]
                
                magnitude = torch.sqrt(real_part**2 + imag_part**2)
                phase = torch.atan2(imag_part, real_part)
            else:
                # If magnitude-only
                magnitude = torch.abs(cqt_result)
                phase = torch.zeros_like(magnitude)
            
        return magnitude, phase
    
    async def vqt(self, audio_tensor: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Variable-Q Transform.
        
        Args:
            audio_tensor: Input audio tensor
            **kwargs: VQT parameters
            
        Returns:
            Tuple of (magnitude, phase) tensors
        """
        if self.vqt_module is None or kwargs:
            self.vqt_module = self._initialize_vqt(**kwargs)
        
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        with torch.no_grad():
            vqt_result = self.vqt_module(audio_tensor)
            
            # VQT returns complex values
            if vqt_result.dim() == 4 and vqt_result.shape[-1] == 2:
                real_part = vqt_result[..., 0]
                imag_part = vqt_result[..., 1]
                
                magnitude = torch.sqrt(real_part**2 + imag_part**2)
                phase = torch.atan2(imag_part, real_part)
            else:
                magnitude = torch.abs(vqt_result)
                phase = torch.zeros_like(magnitude)
            
        return magnitude, phase
    
    async def inverse_stft(self, magnitude: torch.Tensor, phase: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Inverse STFT reconstruction.
        
        Args:
            magnitude: Magnitude tensor
            phase: Phase tensor
            **kwargs: iSTFT parameters
            
        Returns:
            Reconstructed audio tensor
        """
        # Reconstruct complex spectrogram
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)
        complex_spec = torch.stack([real_part, imag_part], dim=-1)
        
        # Use nnAudio iSTFT if available
        try:
            istft_module = features.iSTFT(
                sr=self.sr,
                n_fft=kwargs.get("n_fft", self.default_params["stft"]["n_fft"]),
                hop_length=kwargs.get("hop_length", self.default_params["stft"]["hop_length"]),
                window=kwargs.get("window", self.default_params["stft"]["window"]),
                center=kwargs.get("center", self.default_params["stft"]["center"])
            ).to(self.device)
            
            with torch.no_grad():
                audio_tensor = istft_module(complex_spec)
                
        except Exception:
            # Fallback to PyTorch's built-in iSTFT
            complex_tensor = torch.complex(real_part, imag_part)
            audio_tensor = torch.istft(
                complex_tensor,
                n_fft=kwargs.get("n_fft", self.default_params["stft"]["n_fft"]),
                hop_length=kwargs.get("hop_length", self.default_params["stft"]["hop_length"]),
                window=torch.hann_window(
                    kwargs.get("n_fft", self.default_params["stft"]["n_fft"])
                ).to(self.device),
                center=kwargs.get("center", self.default_params["stft"]["center"])
            )
        
        return audio_tensor
    
    def get_frequency_bins(self, transform_type: str, **kwargs) -> torch.Tensor:
        """Get frequency bins for a given transform type."""
        if transform_type == "stft":
            n_fft = kwargs.get("n_fft", self.default_params["stft"]["n_fft"])
            return torch.linspace(0, self.sr/2, n_fft//2 + 1).to(self.device)
            
        elif transform_type == "mel":
            n_mels = kwargs.get("n_mels", self.default_params["mel"]["n_mels"])
            fmin = kwargs.get("fmin", self.default_params["mel"]["fmin"])
            fmax = kwargs.get("fmax", self.sr // 2)
            
            # Mel scale frequency bins
            mel_min = 2595 * torch.log10(torch.tensor(1 + fmin / 700))
            mel_max = 2595 * torch.log10(torch.tensor(1 + fmax / 700))
            mel_bins = torch.linspace(mel_min, mel_max, n_mels)
            freq_bins = 700 * (10**(mel_bins / 2595) - 1)
            return freq_bins.to(self.device)
            
        elif transform_type == "cqt":
            fmin = kwargs.get("fmin", self.default_params["cqt"]["fmin"])
            n_bins = kwargs.get("n_bins", self.default_params["cqt"]["n_bins"])
            bins_per_octave = kwargs.get("bins_per_octave", self.default_params["cqt"]["bins_per_octave"])
            
            freq_bins = fmin * (2 ** (torch.arange(n_bins, dtype=torch.float32) / bins_per_octave))
            return freq_bins.to(self.device)
            
        else:
            raise ValueError(f"Unsupported transform type: {transform_type}")
    
    def get_transform_info(self, transform_type: str, **kwargs) -> Dict[str, Any]:
        """Get information about a transform configuration."""
        params = {**self.default_params[transform_type], **kwargs}
        freq_bins = self.get_frequency_bins(transform_type, **kwargs)
        
        return {
            "transform_type": transform_type,
            "sample_rate": self.sr,
            "parameters": params,
            "frequency_bins": freq_bins.cpu().tolist(),
            "num_bins": len(freq_bins),
            "device": str(self.device)
        }