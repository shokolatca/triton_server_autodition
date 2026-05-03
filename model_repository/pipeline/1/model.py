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


def _find_model_catalog(model_repository: str) -> Path | None:
    repo = Path(model_repository)
    candidates = [
        repo / "classifier" / "models.json",
        repo.parent / "classifier" / "models.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _decode_string_tensor(value: np.ndarray) -> str:
    item = value.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"])
        params = {k: v["string_value"] for k, v in config.get("parameters", {}).items()}
        self.A0 = float(params.get("A0", "1.0"))
        self.R0 = float(params.get("R0", "10.0"))

        catalog_path = _find_model_catalog(args["model_repository"])
        if catalog_path is None:
            labels_path = _find_labels(args["model_repository"], args["model_name"])
            labels = json.loads(labels_path.read_text())
            self.default_classifier = "default"
            self.classifiers = {
                "default": {
                    "triton_model": "classifier",
                    "classes": labels["classes"],
                    "emv": labels.get("emv", []),
                    "input_name": labels.get("input_name", "features"),
                    "output_name": labels.get("output_name", "logits"),
                    "feature_source": "mel_spectrogram",
                }
            }
        else:
            catalog = json.loads(catalog_path.read_text())
            self.default_classifier = catalog["default"]
            self.classifiers = catalog["models"]

    def execute(self, requests):
        responses = []
        for req in requests:
            audio_t = pb_utils.get_input_tensor_by_name(req, "audio")
            sr_t = pb_utils.get_input_tensor_by_name(req, "sample_rate")
            classifier_t = pb_utils.get_input_tensor_by_name(req, "classifier_model")
            classifier_id = (
                _decode_string_tensor(classifier_t.as_numpy())
                if classifier_t is not None
                else self.default_classifier
            )
            classifier = self.classifiers.get(classifier_id, self.classifiers[self.default_classifier])

            loc = self._infer("localizer", [audio_t, sr_t], ["doa_deg", "peak_amplitude"])
            doa = loc["doa_deg"]
            amp = float(loc["peak_amplitude"][0])

            doa_t = pb_utils.Tensor("doa_deg", doa.astype(np.float32))
            sel = self._infer("channel_selector", [audio_t, doa_t], ["mono_audio", "selected_mic"])

            mono_t = pb_utils.Tensor("mono_audio", sel["mono_audio"].astype(np.float32))
            feature_source = classifier.get("feature_source")
            if feature_source == "mel_spectrogram":
                feat = self._infer("feature_extractor", [mono_t, sr_t], ["features"])
                feat_name = "features"
            elif feature_source == "ast_input_values":
                feat = self._infer("ast_feature_extractor", [mono_t, sr_t], ["input_values"])
                feat_name = "input_values"
            else:
                raise pb_utils.TritonModelException(
                    f"Unsupported feature_source for classifier {classifier_id}: "
                    f"{feature_source}"
                )
            clf_input = classifier.get("input_name", "mel_spectrogram")
            clf_output = classifier.get("output_name", "class_logits")
            feat_t = pb_utils.Tensor(clf_input, feat[feat_name].astype(np.float32))
            clf = self._infer(classifier["triton_model"], [feat_t], [clf_output])
            logits = np.asarray(clf[clf_output]).reshape(-1)
            probs = _softmax(logits).astype(np.float32)
            cls_id = int(np.argmax(probs))
            classes = classifier["classes"]
            cls_name = classes[cls_id]
            confidence = float(probs[cls_id])

            is_emv = cls_name in set(classifier.get("emv", []))
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
