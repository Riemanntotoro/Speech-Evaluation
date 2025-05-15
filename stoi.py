#!/usr/bin/env python3
"""
stoi.py

Compute STOI and ESTOI metrics using the pystoi library,
with handling for mismatched signal lengths.
"""
import soundfile as sf
from pystoi import stoi as _stoi

def _truncate_signals(ref, deg):
    """
    Trim both reference and degraded signals to the same length.
    """
    min_len = min(len(ref), len(deg))
    return ref[:min_len], deg[:min_len]

def stoi(ref_path: str, deg_path: str, fs: int = 16000) -> float:
    """
    Compute the Short-Time Objective Intelligibility (STOI) score.
    """
    ref, sr_ref = sf.read(ref_path, dtype='float32')
    deg, sr_deg = sf.read(deg_path, dtype='float32')
    if sr_ref != fs or sr_deg != fs:
        raise ValueError(f"Sampling rate mismatch: {fs} vs {sr_ref},{sr_deg}")
    ref, deg = _truncate_signals(ref, deg)
    return _stoi(ref, deg, fs, extended=False)

def estoi(ref_path: str, deg_path: str, fs: int = 16000) -> float:
    """
    Compute the Extended STOI (ESTOI) score.
    """
    ref, sr_ref = sf.read(ref_path, dtype='float32')
    deg, sr_deg = sf.read(deg_path, dtype='float32')
    if sr_ref != fs or sr_deg != fs:
        raise ValueError(f"Sampling rate mismatch: {fs} vs {sr_ref},{sr_deg}")
    ref, deg = _truncate_signals(ref, deg)
    return _stoi(ref, deg, fs, extended=True)
