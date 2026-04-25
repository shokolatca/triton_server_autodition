import numpy as np
import pytest

from .conftest import load_model_module
from .synth import synth_8ch


localizer = load_model_module("localizer")


@pytest.mark.parametrize("doa_true", [0.0, 30.0, 90.0, 135.0, 220.0, 315.0])
def test_localize_recovers_synthetic_doa(doa_true: float) -> None:
    sr = 48000
    audio = synth_8ch(doa_deg=doa_true, sample_rate=sr, duration_sec=1.0, seed=int(doa_true) + 1)
    precise, _, _ = localizer.localize(audio, sr)
    err = abs(((precise - doa_true) + 180.0) % 360.0 - 180.0)
    assert err <= 5.0, f"doa_true={doa_true}, got {precise} (err {err:.2f}°)"


def test_peak_amplitude_is_positive() -> None:
    sr = 48000
    audio = synth_8ch(doa_deg=45.0, sample_rate=sr, duration_sec=1.0, amplitude=0.7)
    amp = localizer.peak_amplitude(audio, sr)
    assert amp > 0.0
