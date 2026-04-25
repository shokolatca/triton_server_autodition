"""End-to-end smoke test against a running Triton instance.

Skipped automatically when Triton is unreachable.
Run after `docker compose up --build` and after `python scripts/export_dummy_classifier.py`.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from .synth import synth_8ch


pytest.importorskip("tritonclient.grpc")


def _try_connect(url: str):
    import tritonclient.grpc as grpcclient

    try:
        client = grpcclient.InferenceServerClient(url=url, verbose=False)
        if not client.is_server_live():
            return None
        if not client.is_model_ready("pipeline"):
            return None
        return client
    except Exception:
        return None


@pytest.fixture(scope="module")
def triton_client():
    url = os.environ.get("TRITON_URL", "localhost:8001")
    client = _try_connect(url)
    if client is None:
        pytest.skip(f"Triton not reachable at {url}")
    return client


@pytest.mark.parametrize("doa_true", [60.0, 200.0, 315.0])
def test_pipeline_end_to_end(triton_client, doa_true: float) -> None:
    import tritonclient.grpc as grpcclient

    sr = 48000
    audio = synth_8ch(doa_deg=doa_true, sample_rate=sr, duration_sec=3.0, seed=int(doa_true))

    audio_in = grpcclient.InferInput("audio", list(audio.shape), "FP32")
    audio_in.set_data_from_numpy(audio.astype(np.float32))
    sr_in = grpcclient.InferInput("sample_rate", [1], "INT32")
    sr_in.set_data_from_numpy(np.array([sr], dtype=np.int32))

    requested = ["class_id", "doa_deg", "selected_mic", "distance_m", "is_emv"]
    outputs = [grpcclient.InferRequestedOutput(name) for name in requested]
    response = triton_client.infer("pipeline", inputs=[audio_in, sr_in], outputs=outputs)

    doa = float(response.as_numpy("doa_deg")[0])
    selected_mic = int(response.as_numpy("selected_mic")[0])

    err = abs(((doa - doa_true) + 180.0) % 360.0 - 180.0)
    assert err <= 5.0, f"DOA error {err:.2f}° (true {doa_true}, got {doa})"

    expected_mic = int(round(doa_true / 45.0)) % 8
    assert selected_mic == expected_mic, f"selected_mic={selected_mic}, expected {expected_mic}"
