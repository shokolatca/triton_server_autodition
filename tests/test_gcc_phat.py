import numpy as np
import pytest

from .conftest import load_model_module


localizer = load_model_module("localizer")


def _delayed_pair(n: int, delay: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (s_i, s_j) where s_j[k] = s_i[k - delay] (s_j delayed by `delay` samples)."""
    pad = abs(delay) + 16
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n + 2 * pad).astype(np.float32)
    s_i = base[pad: pad + n]
    s_j = base[pad - delay: pad - delay + n]
    return s_i, s_j


@pytest.mark.parametrize("sr", [44100, 48000, 96000])
@pytest.mark.parametrize("d", [-20, -5, 5, 20])
def test_gcc_phat_recovers_known_delay(sr: int, d: int) -> None:
    s_i, s_j = _delayed_pair(n=8192, delay=d, seed=hash((sr, d)) & 0xFFFFFFFF)
    tau = localizer.gcc_phat(s_i, s_j, sr)
    estimated = tau * sr
    assert abs(estimated - d) <= 1.0, f"sr={sr} d={d}: expected ~{d}, got {estimated:.3f}"


def test_gcc_phat_zero_delay() -> None:
    rng = np.random.default_rng(123)
    s = rng.standard_normal(4096).astype(np.float32)
    tau = localizer.gcc_phat(s, s, sr=48000)
    assert abs(tau * 48000) < 0.5
