"""BLS orchestrator: localize -> select channel -> features -> classify -> distance (EmV)."""

import json
from pathlib import Path

import numpy as np

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    pb_utils = None


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def _find_labels(model_repository: str, model_name: str) -> Path:
    repo = Path(model_repository)
    candidates = [
        repo / "classifier" / "labels.json",
        repo.parent / "classifier" / "labels.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"labels.json not found near {model_repository} (model {model_name})")


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"])
        params = {k: v["string_value"] for k, v in config.get("parameters", {}).items()}
        self.A0 = float(params.get("A0", "1.0"))
        self.R0 = float(params.get("R0", "10.0"))

        labels_path = _find_labels(args["model_repository"], args["model_name"])
        labels = json.loads(labels_path.read_text())
        self.classes = labels["classes"]
        self.emv = set(labels.get("emv", []))
        self.clf_input = labels.get("input_name", "features")
        self.clf_output = labels.get("output_name", "logits")

    def execute(self, requests):
        responses = []
        for req in requests:
            audio_t = pb_utils.get_input_tensor_by_name(req, "audio")
            sr_t = pb_utils.get_input_tensor_by_name(req, "sample_rate")

            loc = self._infer("localizer", [audio_t, sr_t], ["doa_deg", "peak_amplitude"])
            doa = loc["doa_deg"]
            amp = float(loc["peak_amplitude"][0])

            doa_t = pb_utils.Tensor("doa_deg", doa.astype(np.float32))
            sel = self._infer("channel_selector", [audio_t, doa_t], ["mono_audio", "selected_mic"])

            mono_t = pb_utils.Tensor("mono_audio", sel["mono_audio"].astype(np.float32))
            feat = self._infer("feature_extractor", [mono_t, sr_t], ["features"])

            feat_t = pb_utils.Tensor(self.clf_input, feat["features"].astype(np.float32))
            clf = self._infer("classifier", [feat_t], [self.clf_output])
            logits = np.asarray(clf[self.clf_output]).reshape(-1)
            probs = _softmax(logits).astype(np.float32)
            cls_id = int(np.argmax(probs))
            cls_name = self.classes[cls_id]
            confidence = float(probs[cls_id])

            is_emv = cls_name in self.emv
            distance = (self.R0 * self.A0 / amp) if (is_emv and amp > 1e-9) else -1.0

            responses.append(
                pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor("class_id", np.array([cls_id], dtype=np.int32)),
                    pb_utils.Tensor("class_name", np.array([cls_name], dtype=object)),
                    pb_utils.Tensor("confidence", np.array([confidence], dtype=np.float32)),
                    pb_utils.Tensor("probs", probs),
                    pb_utils.Tensor("doa_deg", doa.astype(np.float32)),
                    pb_utils.Tensor("selected_mic", sel["selected_mic"].astype(np.int32)),
                    pb_utils.Tensor("distance_m", np.array([distance], dtype=np.float32)),
                    pb_utils.Tensor("is_emv", np.array([is_emv], dtype=bool)),
                ])
            )
        return responses

    def _infer(self, model_name: str, inputs, requested_outputs):
        req = pb_utils.InferenceRequest(
            model_name=model_name,
            requested_output_names=requested_outputs,
            inputs=inputs,
        )
        resp = req.exec()
        if resp.has_error():
            raise pb_utils.TritonModelException(resp.error().message())
        return {
            name: pb_utils.get_output_tensor_by_name(resp, name).as_numpy()
            for name in requested_outputs
        }
