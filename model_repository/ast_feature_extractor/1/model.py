"""AST input_values feature extractor based on Hugging Face transformers."""

import json

import numpy as np
from transformers import ASTFeatureExtractor

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    pb_utils = None

import librosa


def compute_ast_input_values(
    mono: np.ndarray,
    sr: int,
    target_sr: int,
    max_length: int,
    num_mel_bins: int,
    mean: float,
    std: float,
    feature_extractor: ASTFeatureExtractor | None = None,
) -> np.ndarray:
    if sr != target_sr:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)

    extractor = feature_extractor or ASTFeatureExtractor(
        sampling_rate=target_sr,
        num_mel_bins=num_mel_bins,
        max_length=max_length,
        mean=mean,
        std=std,
        do_normalize=True,
    )
    values = extractor(
        mono.astype(np.float32, copy=False),
        sampling_rate=target_sr,
        return_tensors="np",
    )["input_values"]
    return np.ascontiguousarray(values.astype(np.float32, copy=False))


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"])
        params = {k: v["string_value"] for k, v in config.get("parameters", {}).items()}
        self.target_sr = int(params.get("TARGET_SR", "16000"))
        self.max_length = int(params.get("MAX_LENGTH", "1024"))
        self.num_mel_bins = int(params.get("NUM_MEL_BINS", "128"))
        self.mean = float(params.get("MEAN", "-4.2677393"))
        self.std = float(params.get("STD", "4.5689974"))
        self.feature_extractor = ASTFeatureExtractor(
            sampling_rate=self.target_sr,
            num_mel_bins=self.num_mel_bins,
            max_length=self.max_length,
            mean=self.mean,
            std=self.std,
            do_normalize=True,
        )

    def execute(self, requests):
        responses = []
        for req in requests:
            mono = pb_utils.get_input_tensor_by_name(req, "mono_audio").as_numpy().astype(np.float32)
            sr = int(pb_utils.get_input_tensor_by_name(req, "sample_rate").as_numpy()[0])
            input_values = compute_ast_input_values(
                mono,
                sr,
                target_sr=self.target_sr,
                max_length=self.max_length,
                num_mel_bins=self.num_mel_bins,
                mean=self.mean,
                std=self.std,
                feature_extractor=self.feature_extractor,
            )
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor("input_values", input_values),
                ])
            )
        return responses
