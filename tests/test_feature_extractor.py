import numpy as np

from .conftest import load_model_module


feature_extractor = load_model_module("feature_extractor")


def test_compute_log_mel_shape_and_dtype() -> None:
    sr = 48000
    mono = np.random.default_rng(0).standard_normal(sr).astype(np.float32)
    feats = feature_extractor.compute_log_mel(
        mono, sr=sr, target_sr=16000, target_sec=10.0, n_mels=128, n_fft=1024, hop_length=512
    )
    assert feats.shape[0] == 1 and feats.shape[2] == 128
    assert feats.shape[1] > 0
    assert feats.dtype == np.float32


def test_compute_log_mel_pads_short_input() -> None:
    sr = 16000
    mono = np.zeros(sr // 2, dtype=np.float32)  # half a second
    feats = feature_extractor.compute_log_mel(
        mono, sr=sr, target_sr=16000, target_sec=10.0, n_mels=128, n_fft=1024, hop_length=512
    )
    expected_frames = 1 + (10 * 16000) // 512
    assert abs(feats.shape[1] - expected_frames) <= 1
