# -*- coding: utf-8

"""
Thomas Grill, 2011-2015
http://grrrr.org/nsgt

--
Original matlab code comments follow:

NSGFWIN.M
---------------------------------------------------------------
 [g,rfbas,M]=nsgfwin(fmin,bins,sr,Ls) creates a set of windows whose
 centers correspond to center frequencies to be
 used for the nonstationary Gabor transform with varying Q-factor. 
---------------------------------------------------------------

INPUT : fmin ...... Minimum frequency (in Hz)
        bins ...... Vector consisting of the number of bins per octave
        sr ........ Sampling rate (in Hz)
        Ls ........ Length of signal (in samples)

OUTPUT : g ......... Cell array of window functions.
         rfbas ..... Vector of positions of the center frequencies.
         M ......... Vector of lengths of the window functions.

AUTHOR(s) : Monika Dörfler, Gino Angelo Velasco, Nicki Holighaus, 2010

COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
http://nuhag.eu/
Permission is granted to modify and re-distribute this
code in any manner as long as this notice is preserved.
All standard disclaimers apply.

EXTERNALS : firwin
"""

import numpy as np
from .util import hannwin,_isseq

def nsgfwin(fmin, fmax, bins, sr, Ls, min_win=4):
    """
    Generate windows for Non-Stationary Gabor Transform
    
    Args:
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz) 
        bins: Number of bins per octave or array of bins
        sr: Sample rate (Hz)
        Ls: Signal length (samples)
        min_win: Minimum window size (samples)
        
    Returns:
        tuple: (windows, frequency_bases, window_lengths)
        
    Raises:
        ValueError: For invalid parameter combinations
    """
    # Parameter validation with clear error messages
    if fmin <= 0:
        raise ValueError(f"fmin must be positive, got {fmin}")
    if fmax <= fmin:
        raise ValueError(f"fmax ({fmax}) must be greater than fmin ({fmin})")
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}")
    if Ls <= 0:
        raise ValueError(f"Ls must be positive, got {Ls}")
    if min_win < 1:
        raise ValueError(f"min_win must be at least 1, got {min_win}")

    nf = sr/2
    
    if fmax > nf:
        fmax = nf
    
    b = int(np.ceil(np.log2(fmax/fmin))+1)

    if not _isseq(bins):
        bins = np.ones(b,dtype=int)*bins
    elif len(bins) < b:
        # TODO: test this branch!
        bins[bins <= 0] = 1
        bins = np.concatenate((bins, np.ones(b-len(bins), dtype=int)*np.min(bins)))
    
    fbas = []
    for kk,bkk in enumerate(bins):
        # Use logspace for more numerically stable computation
        start_exp = np.log2(fmin) + kk
        stop_exp = np.log2(fmin) + (kk+1)
        fbas.append(np.logspace(start_exp, stop_exp, bkk, endpoint=False, base=2))
    fbas = np.concatenate(fbas)

    if fbas[np.min(np.where(fbas>=fmax))] >= nf:
        fbas = fbas[:np.max(np.where(fbas<fmax))+1]
    else:
        # TODO: test this branch!
        fbas = fbas[:np.min(np.where(fbas>=fmax))+1]
    
    lbas = len(fbas)
    fbas = np.concatenate(((0.,), fbas, (nf,), sr-fbas[::-1]))
    fbas *= float(Ls)/sr
    
    # Vectorized computation for better performance
    M = np.empty(fbas.shape, dtype=int)
    M[0] = np.round(2.*fmin*Ls/sr)
    # Vectorized calculation for middle elements
    M[1:2*lbas+1] = np.round(fbas[2:2*lbas+2] - fbas[:2*lbas])
    M[-1] = np.round(Ls-fbas[-2])
    
    M = np.clip(M, min_win, np.inf).astype(int)
    g = [hannwin(m) for m in M]
    
    fbas[lbas] = (fbas[lbas-1]+fbas[lbas+1])/2
    fbas[lbas+2] = Ls-fbas[lbas]
    rfbas = np.round(fbas).astype(int)
    
    return g,rfbas,M
