import numpy as np
import pytest

from .conftest import load_model_module


pytest.importorskip("transformers")
pytest.importorskip("torchaudio")

ast_feature_extractor = load_model_module("ast_feature_extractor")


def test_compute_ast_input_values_shape_and_dtype() -> None:
    sr = 48000
    mono = np.random.default_rng(1).standard_normal(sr).astype(np.float32)
    feats = ast_feature_extractor.compute_ast_input_values(
        mono,
        sr=sr,
        target_sr=16000,
        max_length=1024,
        num_mel_bins=128,
        mean=-4.2677393,
        std=4.5689974,
    )
    assert feats.shape == (1, 1024, 128)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()


def test_compute_ast_input_values_pads_short_input() -> None:
    sr = 16000
    mono = np.zeros(sr // 2, dtype=np.float32)
    feats = ast_feature_extractor.compute_ast_input_values(
        mono,
        sr=sr,
        target_sr=16000,
        max_length=1024,
        num_mel_bins=128,
        mean=-4.2677393,
        std=4.5689974,
    )
    assert feats.shape == (1, 1024, 128)
    assert np.isfinite(feats).all()
