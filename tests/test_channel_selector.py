import numpy as np
import pytest

from .conftest import load_model_module


channel_selector = load_model_module("channel_selector")


def _make_audio() -> np.ndarray:
    return np.tile(np.arange(8, dtype=np.float32)[:, None], (1, 64))


@pytest.mark.parametrize(
    "doa,expected",
    [
        (0.0, 0),
        (45.0, 1),
        (90.0, 2),
        (135.0, 3),
        (180.0, 4),
        (225.0, 5),
        (270.0, 6),
        (315.0, 7),
        # halfway between mics ties broken by argmin (lower index wins for default ties)
        (22.0, 0),
        (23.0, 1),
        (359.0, 0),
        (-1.0, 0),
    ],
)
def test_select_channel(doa: float, expected: int) -> None:
    audio = _make_audio()
    mono, idx = channel_selector.select_channel(audio, doa)
    assert idx == expected
    assert mono[0] == float(expected)


def test_select_channel_rejects_wrong_shape() -> None:
    audio = np.zeros((4, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        channel_selector.select_channel(audio, 0.0)
