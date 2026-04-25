"""Synthesize an 8-channel plane-wave WAV from a far-field source at a known DOA."""

from __future__ import annotations

import numpy as np


MIC_COUNT = 8
DEFAULT_RADIUS_M = 0.5
SPEED_OF_SOUND = 343.0


def mic_positions(radius_m: float = DEFAULT_RADIUS_M) -> np.ndarray:
    angles = np.deg2rad(np.arange(MIC_COUNT) * (360.0 / MIC_COUNT))
    return np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius_m


def synth_8ch(
    doa_deg: float,
    sample_rate: int = 48000,
    duration_sec: float = 3.0,
    radius_m: float = DEFAULT_RADIUS_M,
    speed_ms: float = SPEED_OF_SOUND,
    amplitude: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Synthesize a planar broadband (white-noise) wavefront arriving from `doa_deg`.

    Each mic receives a fractional-sample-shifted copy via FFT phase rotation, so the
    signal is broadband enough that GCC-PHAT yields an unambiguous TDOA peak.
    """
    n = int(sample_rate * duration_sec)
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n).astype(np.float32) * amplitude

    mics = mic_positions(radius_m)
    u = np.array([np.cos(np.deg2rad(doa_deg)), np.sin(np.deg2rad(doa_deg))])
    delays = -(mics @ u) / speed_ms  # arrival time at mic i relative to origin

    n_fft = int(2 ** np.ceil(np.log2(n)))
    spec = np.fft.rfft(base, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    omega = 2.0 * np.pi * freqs

    out = np.empty((MIC_COUNT, n), dtype=np.float32)
    for i, tau_i in enumerate(delays):
        shifted = np.fft.irfft(spec * np.exp(-1j * omega * tau_i), n=n_fft)
        out[i] = shifted[:n].astype(np.float32)
    return out
