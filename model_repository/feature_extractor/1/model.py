"""Mel-spectrogram feature extractor (librosa)."""

import json

import numpy as np

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    pb_utils = None

import librosa


def compute_log_mel(
    mono: np.ndarray,
    sr: int,
    target_sr: int,
    target_sec: float,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    if sr != target_sr:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
    n_target = int(target_sr * target_sec)
    mono = librosa.util.fix_length(mono, size=n_target)
    mel = librosa.feature.melspectrogram(
        y=mono.astype(np.float32, copy=False),
        sr=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    log_mel = librosa.power_to_db(mel).astype(np.float32)
    return np.ascontiguousarray(log_mel.T[None, :, :])


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"])
        params = {k: v["string_value"] for k, v in config.get("parameters", {}).items()}
        self.n_mels = int(params.get("N_MELS", "128"))
        self.n_fft = int(params.get("N_FFT", "1024"))
        self.hop_length = int(params.get("HOP_LENGTH", "512"))
        self.target_sr = int(params.get("TARGET_SR", "16000"))
        self.target_sec = float(params.get("TARGET_SEC", "10.0"))

    def execute(self, requests):
        responses = []
        for req in requests:
            mono = pb_utils.get_input_tensor_by_name(req, "mono_audio").as_numpy().astype(np.float32)
            sr = int(pb_utils.get_input_tensor_by_name(req, "sample_rate").as_numpy()[0])
            features = compute_log_mel(
                mono, sr,
                target_sr=self.target_sr,
                target_sec=self.target_sec,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor("features", features),
                ])
            )
        return responses
