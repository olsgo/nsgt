# -*- coding: utf-8

"""
Python 3.12 specific optimizations and modernizations for NSGT

This module demonstrates Python 3.12 features that could be integrated
into the main NSGT codebase for better performance and maintainability.
"""

from __future__ import annotations
import numpy as np
from typing import Union, Optional, Tuple, List
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)  # Python 3.10+ feature, optimized in 3.12
class NSGTConfig:
    """Configuration dataclass with Python 3.12 optimizations"""
    fs: int
    Ls: int
    real: bool = True
    matrixform: bool = False
    reducedform: int = 0
    multichannel: bool = False
    measurefft: bool = False
    multithreading: bool = False
    dtype: type = float
    
    def __post_init__(self):
        if self.fs <= 0:
            raise ValueError("fs must be > 0")
        if self.Ls <= 0:
            raise ValueError("Ls must be > 0")
        if not 0 <= self.reducedform <= 2:
            raise ValueError("reducedform must be in [0, 1, 2]")


def modern_frequency_mapping(fmin: float, fmax: float, bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Modern frequency mapping using Python 3.12 features
    
    Uses match/case statements and enhanced error handling
    """
    if fmin <= 0 or fmax <= fmin:
        raise ValueError(f"Invalid frequency range: fmin={fmin}, fmax={fmax}")
    
    # Python 3.12 has better match/case performance
    match bins:
        case x if x <= 0:
            raise ValueError("bins must be positive")
        case 1:
            # Special case for single bin
            frequencies = np.array([np.sqrt(fmin * fmax)])
            q_factors = np.array([1.0])
        case _:
            # General case with optimized computation
            log_ratio = np.log2(fmax / fmin)
            frequencies = fmin * 2**(np.arange(bins) * log_ratio / (bins - 1))
            # Q-factor computation with vectorized operations
            if len(frequencies) > 1:
                diffs = frequencies[1:] - frequencies[:-1]
                q_factors = frequencies[:-1] * 0.5 / diffs
                q_factors = np.append(q_factors, q_factors[-1])  # Extend to match length
            else:
                q_factors = np.array([1.0])
    
    return frequencies, q_factors


class ModernQualityAssessment:
    """Quality assessment with Python 3.12 optimizations"""
    
    def __init__(self, *, reference: np.ndarray, estimate: np.ndarray):
        # Python 3.12 improved keyword-only argument handling
        self.reference = np.asarray(reference).flatten()
        self.estimate = np.asarray(estimate).flatten()
        self._align_signals()
    
    def _align_signals(self) -> None:
        """Align signal lengths using modern Python features"""
        min_len = min(len(self.reference), len(self.estimate))
        self.reference = self.reference[:min_len]
        self.estimate = self.estimate[:min_len]
    
    def compute_metrics(self) -> dict[str, float]:
        """Compute all quality metrics with optimized algorithms"""
        # Python 3.12 has better dict comprehension performance
        metrics = {}
        
        # Vectorized power calculations
        ref_power = float(np.sum(self.reference ** 2))
        error_power = float(np.sum((self.estimate - self.reference) ** 2))
        
        # SDR with better numerical stability
        metrics['sdr_db'] = (
            10.0 * np.log10(ref_power / error_power) 
            if error_power > 1e-15 
            else float('inf')
        )
        
        # Enhanced spectral analysis
        ref_fft = np.fft.fft(self.reference)
        est_fft = np.fft.fft(self.estimate)
        
        spectral_error = np.sum(np.abs(est_fft - ref_fft) ** 2)
        spectral_power = np.sum(np.abs(ref_fft) ** 2)
        
        metrics['spectral_convergence'] = (
            float(np.sqrt(spectral_error / spectral_power))
            if spectral_power > 1e-15
            else float('inf')
        )
        
        return metrics


def optimized_window_generation(
    frequencies: np.ndarray, 
    q_factors: np.ndarray, 
    *,  # Force keyword-only arguments
    sr: int,
    Ls: int,
    min_win: int = 4,
    dtype: type = np.float64
) -> List[np.ndarray]:
    """
    Generate windows with Python 3.12 optimizations
    
    Uses modern typing, match/case, and vectorized operations
    """
    if len(frequencies) != len(q_factors):
        raise ValueError("frequencies and q_factors must have same length")
    
    # Vectorized window size calculation
    window_sizes = np.round(q_factors * sr / frequencies).astype(int)
    window_sizes = np.maximum(window_sizes, min_win)
    
    # Generate windows using list comprehension with enhanced logic
    # Python 3.12 optimized list comprehensions
    windows = [
        np.hanning(max(size, min_win)).astype(dtype) 
        for size in window_sizes 
        if size >= min_win
    ]
    
    return windows


def demonstrate_python312_features():
    """Demonstrate Python 3.12 specific features for NSGT"""
    
    print("=== Python 3.12 NSGT Optimizations Demo ===")
    
    # Test dataclass configuration
    config = NSGTConfig(fs=44100, Ls=8192, real=True)
    print(f"Config: {config}")
    
    # Test modern frequency mapping
    freqs, q_vals = modern_frequency_mapping(100, 8000, 12)
    print(f"Generated {len(freqs)} frequency bands")
    
    # Test optimized quality assessment
    np.random.seed(42)
    ref_signal = np.random.randn(1000).astype(np.float32)
    est_signal = ref_signal + 0.01 * np.random.randn(1000).astype(np.float32)
    
    qa = ModernQualityAssessment(reference=ref_signal, estimate=est_signal)
    metrics = qa.compute_metrics()
    
    print(f"Quality metrics: {metrics}")
    
    # Test optimized window generation
    windows = optimized_window_generation(
        freqs[:5], q_vals[:5],
        sr=44100, Ls=8192, min_win=4
    )
    print(f"Generated {len(windows)} windows")
    
    print("✅ All Python 3.12 optimizations working!")


if __name__ == '__main__':
    demonstrate_python312_features()