"""Channel selector: pick the microphone closest to estimated DOA."""

import numpy as np

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    pb_utils = None


MIC_COUNT = 8
MIC_ANGLES_DEG = np.arange(MIC_COUNT) * (360.0 / MIC_COUNT)


def select_channel(audio: np.ndarray, doa_deg: float) -> tuple[np.ndarray, int]:
    if audio.shape[0] != MIC_COUNT:
        raise ValueError(f"expected {MIC_COUNT} channels, got {audio.shape[0]}")
    diff = np.abs(((MIC_ANGLES_DEG - doa_deg) + 180.0) % 360.0 - 180.0)
    idx = int(np.argmin(diff))
    return audio[idx].astype(np.float32, copy=False), idx


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for req in requests:
            audio = pb_utils.get_input_tensor_by_name(req, "audio").as_numpy()
            doa = float(pb_utils.get_input_tensor_by_name(req, "doa_deg").as_numpy()[0])
            mono, idx = select_channel(audio, doa)
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor("mono_audio", mono),
                    pb_utils.Tensor("selected_mic", np.array([idx], dtype=np.int32)),
                ])
            )
        return responses
