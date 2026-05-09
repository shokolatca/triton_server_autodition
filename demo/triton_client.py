"""Shared Triton gRPC call used by both the Gradio app and the CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import tritonclient.grpc as grpcclient


MIC_COUNT = 8
MODEL_NAME = "pipeline"
DEFAULT_CLASSIFIER_MODEL = "furletov_cnn"
CLASSIFIER_MODEL_CHOICES = {
    "furletov_cnn": "model-base",
    "furletov_ast": "model-medium",
    "us8k_cnn": "model-high",
    "us8k_ast": "model-ultra",
}
CLASSIFIER_LABELS = {
    "furletov_cnn": [
        "car_acceleration",
        "car_braking",
        "car_horn",
        "car_idling",
        "moto_acceleration",
        "moto_idling",
        "siren_1",
        "siren_4",
        "siren_5",
        "tram",
        "tram_acceleration",
        "tram_braking",
        "tram_ring",
        "truck_acceleration",
        "truck_braking",
        "truck_horn",
        "truck_idling",
    ],
    "furletov_ast": [
        "car_acceleration",
        "car_braking",
        "car_horn",
        "car_idling",
        "moto_acceleration",
        "moto_idling",
        "siren_1",
        "siren_4",
        "siren_5",
        "tram",
        "tram_acceleration",
        "tram_braking",
        "tram_ring",
        "truck_acceleration",
        "truck_braking",
        "truck_horn",
        "truck_idling",
    ],
    "us8k_cnn": [
        "air_conditioner",
        "car_horn",
        "children_playing",
        "dog_bark",
        "drilling",
        "engine_idling",
        "gun_shot",
        "jackhammer",
        "siren",
        "street_music",
    ],
    "us8k_ast": [
        "air_conditioner",
        "car_horn",
        "children_playing",
        "dog_bark",
        "drilling",
        "engine_idling",
        "gun_shot",
        "jackhammer",
        "siren",
        "street_music",
    ],
}


@dataclass
class PipelineResult:
    class_id: int
    class_name: str
    confidence: float
    probs: np.ndarray
    doa_deg: float
    selected_mic: int
    distance_m: float
    is_emv: bool


def load_8ch_wav(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True)
    if audio.shape[1] != MIC_COUNT:
        raise ValueError(f"expected {MIC_COUNT} channels, got {audio.shape[1]} in {path}")
    return audio.T.astype(np.float32, copy=False), int(sr)


def run_pipeline(
    audio_8ch: np.ndarray,
    sample_rate: int,
    url: str | None = None,
    classifier_model: str = DEFAULT_CLASSIFIER_MODEL,
) -> PipelineResult:
    url = url or os.environ.get("TRITON_URL", "localhost:8001")
    client = grpcclient.InferenceServerClient(url=url)

    audio_in = grpcclient.InferInput("audio", list(audio_8ch.shape), "FP32")
    audio_in.set_data_from_numpy(audio_8ch)
    sr_arr = np.array([sample_rate], dtype=np.int32)
    sr_in = grpcclient.InferInput("sample_rate", [1], "INT32")
    sr_in.set_data_from_numpy(sr_arr)
    classifier_in = grpcclient.InferInput("classifier_model", [1], "BYTES")
    classifier_in.set_data_from_numpy(np.array([classifier_model], dtype=object))

    requested = [
        "class_id", "class_name", "confidence", "probs",
        "doa_deg", "selected_mic", "distance_m", "is_emv",
    ]
    outputs = [grpcclient.InferRequestedOutput(name) for name in requested]
    response = client.infer(MODEL_NAME, inputs=[audio_in, sr_in, classifier_in], outputs=outputs)

    name_arr = response.as_numpy("class_name")[0]
    if isinstance(name_arr, bytes):
        class_name = name_arr.decode("utf-8")
    else:
        class_name = str(name_arr)

    return PipelineResult(
        class_id=int(response.as_numpy("class_id")[0]),
        class_name=class_name,
        confidence=float(response.as_numpy("confidence")[0]),
        probs=response.as_numpy("probs").astype(np.float32),
        doa_deg=float(response.as_numpy("doa_deg")[0]),
        selected_mic=int(response.as_numpy("selected_mic")[0]),
        distance_m=float(response.as_numpy("distance_m")[0]),
        is_emv=bool(response.as_numpy("is_emv")[0]),
    )
