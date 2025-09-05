# -*- coding: utf-8

"""
Quality measurement framework for NSGT coefficients

Thomas Grill, 2025
http://grrrr.org/nsgt

Provides standardized quality metrics for evaluating NSGT transform quality:
- Signal-to-Distortion Ratio (SDR)
- Signal-to-Interference Ratio (SIR) 
- Signal-to-Artifacts Ratio (SAR)
- Spectral convergence measures
- Perfect reconstruction error analysis
"""

import numpy as np
from typing import Tuple, Optional, Union


def sdr(reference: np.ndarray, estimate: np.ndarray, 
        window: Optional[Union[str, np.ndarray]] = None) -> float:
    """
    Signal-to-Distortion Ratio in dB
    
    Args:
        reference: Original signal
        estimate: Reconstructed signal  
        window: Optional window function or name ('hann', 'hamming', etc.)
        
    Returns:
        SDR in dB (higher is better)
    """
    ref = np.asarray(reference).flatten()
    est = np.asarray(estimate).flatten()
    
    # Align lengths
    min_len = min(len(ref), len(est))
    ref = ref[:min_len]
    est = est[:min_len]
    
    # Apply windowing if specified
    if window is not None:
        if isinstance(window, str):
            if window == 'hann':
                w = np.hanning(min_len)
            elif window == 'hamming':
                w = np.hamming(min_len)
            else:
                raise ValueError(f"Unknown window type: {window}")
        else:
            w = np.asarray(window)
            if len(w) != min_len:
                raise ValueError("Window length must match signal length")
        ref = ref * w
        est = est * w
    
    # Compute SDR  
    signal_power = float(np.real(np.sum(ref ** 2)))
    noise_power = float(np.real(np.sum((est - ref) ** 2)))
    
    if noise_power == 0 or noise_power < 1e-15:
        return float('inf')
    if signal_power == 0 or signal_power < 1e-15:
        return float('-inf')
        
    return 10.0 * np.log10(signal_power / noise_power)


def sir(reference: np.ndarray, estimate: np.ndarray, 
        interference: Optional[np.ndarray] = None) -> float:
    """
    Signal-to-Interference Ratio in dB
    
    Args:
        reference: Original signal
        estimate: Reconstructed signal
        interference: Known interference component (optional)
        
    Returns:
        SIR in dB (higher is better)
    """
    ref = np.asarray(reference).flatten()
    est = np.asarray(estimate).flatten()
    
    min_len = min(len(ref), len(est))
    ref = ref[:min_len]
    est = est[:min_len]
    
    if interference is not None:
        interf = np.asarray(interference).flatten()[:min_len]
    else:
        # Estimate interference as low-frequency components of error
        error = est - ref
        # Simple high-pass filter approximation
        if len(error) > 2:
            interf = error - np.diff(error, prepend=error[0])
        else:
            interf = error
    
    signal_power = np.sum(ref ** 2)
    interf_power = np.sum(interf ** 2)
    
    if interf_power == 0:
        return float('inf')
    if signal_power == 0:
        return float('-inf')
        
    return 10.0 * np.log10(signal_power / interf_power)


def sar(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Signal-to-Artifacts Ratio in dB
    
    Args:
        reference: Original signal
        estimate: Reconstructed signal
        
    Returns:
        SAR in dB (higher is better) 
    """
    ref = np.asarray(reference).flatten()
    est = np.asarray(estimate).flatten()
    
    min_len = min(len(ref), len(est))
    ref = ref[:min_len]
    est = est[:min_len]
    
    # Artifacts are high-frequency components of error
    error = est - ref
    if len(error) > 2:
        artifacts = np.diff(error, prepend=error[0])
    else:
        artifacts = error
        
    signal_power = np.sum(ref ** 2)
    artifact_power = np.sum(artifacts ** 2)
    
    if artifact_power == 0:
        return float('inf')
    if signal_power == 0:
        return float('-inf')
        
    return 10.0 * np.log10(signal_power / artifact_power)


def spectral_convergence(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Spectral convergence measure (lower is better)
    
    Args:
        reference: Original signal
        estimate: Reconstructed signal
        
    Returns:
        Spectral convergence ratio (0 = perfect reconstruction)
    """
    ref = np.asarray(reference).flatten()
    est = np.asarray(estimate).flatten()
    
    min_len = min(len(ref), len(est))
    ref = ref[:min_len]
    est = est[:min_len]
    
    # FFT magnitudes
    ref_fft = np.fft.fft(ref)
    est_fft = np.fft.fft(est)
    
    ref_mag = np.abs(ref_fft)
    est_mag = np.abs(est_fft)
    
    numerator = np.sum((est_mag - ref_mag) ** 2)
    denominator = np.sum(ref_mag ** 2)
    
    if denominator == 0:
        return float('inf')
        
    return np.sqrt(numerator / denominator)


def perfect_reconstruction_error(reference: np.ndarray, estimate: np.ndarray,
                               metric: str = 'l2') -> float:
    """
    Perfect reconstruction error metrics
    
    Args:
        reference: Original signal  
        estimate: Reconstructed signal
        metric: Error metric ('l1', 'l2', 'linf', 'rel_l2')
        
    Returns:
        Error value according to specified metric
    """
    ref = np.asarray(reference).flatten()
    est = np.asarray(estimate).flatten()
    
    min_len = min(len(ref), len(est))
    ref = ref[:min_len]
    est = est[:min_len]
    
    error = est - ref
    
    if metric == 'l1':
        return np.sum(np.abs(error))
    elif metric == 'l2':
        return np.sqrt(np.sum(error ** 2))
    elif metric == 'linf':
        return np.max(np.abs(error))
    elif metric == 'rel_l2':
        ref_norm = np.sqrt(np.sum(ref ** 2))
        if ref_norm == 0:
            return float('inf')
        return np.sqrt(np.sum(error ** 2)) / ref_norm
    else:
        raise ValueError(f"Unknown metric: {metric}")


def comprehensive_quality_report(reference: np.ndarray, estimate: np.ndarray,
                               window: Optional[str] = None) -> dict:
    """
    Generate comprehensive quality assessment report
    
    Args:
        reference: Original signal
        estimate: Reconstructed signal
        window: Optional windowing for SDR computation
        
    Returns:
        Dictionary with all quality metrics
    """
    return {
        'sdr_db': sdr(reference, estimate, window),
        'sir_db': sir(reference, estimate),
        'sar_db': sar(reference, estimate),
        'spectral_convergence': spectral_convergence(reference, estimate),
        'l1_error': perfect_reconstruction_error(reference, estimate, 'l1'),
        'l2_error': perfect_reconstruction_error(reference, estimate, 'l2'),
        'linf_error': perfect_reconstruction_error(reference, estimate, 'linf'),
        'rel_l2_error': perfect_reconstruction_error(reference, estimate, 'rel_l2'),
        'snr_estimate_db': 10.0 * np.log10(np.sum(reference**2) / (np.sum((estimate-reference)**2) + 1e-12))
    }


def print_quality_report(report: dict) -> None:
    """Pretty print quality assessment report"""
    print("=" * 50)
    print("NSGT Quality Assessment Report")
    print("=" * 50)
    print(f"Signal-to-Distortion Ratio:  {report['sdr_db']:.2f} dB")
    print(f"Signal-to-Interference Ratio: {report['sir_db']:.2f} dB") 
    print(f"Signal-to-Artifacts Ratio:   {report['sar_db']:.2f} dB")
    print(f"Estimated SNR:               {report['snr_estimate_db']:.2f} dB")
    print("-" * 50)
    print(f"Spectral Convergence:        {report['spectral_convergence']:.2e}")
    print(f"L1 Error:                    {report['l1_error']:.2e}")  
    print(f"L2 Error:                    {report['l2_error']:.2e}")
    print(f"L∞ Error:                    {report['linf_error']:.2e}")
    print(f"Relative L2 Error:           {report['rel_l2_error']:.2e}")
    print("=" * 50)


# Convenience function for backward compatibility with existing tests
def quality_check(reference: np.ndarray, estimate: np.ndarray, 
                 tolerance: float = 1e-3, verbose: bool = False) -> Tuple[bool, dict]:
    """
    Quick quality check compatible with existing test framework
    
    Args:
        reference: Original signal
        estimate: Reconstructed signal  
        tolerance: L∞ error tolerance (default 1e-3 like existing tests)
        verbose: Print detailed report
        
    Returns:
        (passed, quality_report) tuple
    """
    report = comprehensive_quality_report(reference, estimate)
    passed = report['linf_error'] <= tolerance
    
    if verbose or not passed:
        print_quality_report(report)
        
    return passed, report