"""Generate a tiny dummy classifier ONNX so the pipeline can run before a real checkpoint is available.

Run from repo root:
    python scripts/export_dummy_classifier.py
"""

from pathlib import Path

import torch
import torch.nn as nn


N_CLASSES = 17
N_MELS = 128


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(16, N_CLASSES)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        features = mel_spectrogram.transpose(1, 2).unsqueeze(1)
        x = self.features(features).flatten(1)
        return self.head(x)


def main() -> None:
    out_path = Path(__file__).resolve().parent.parent / "model_repository" / "classifier" / "1" / "model.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = TinyClassifier().eval()
    dummy = torch.randn(1, 313, N_MELS)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["mel_spectrogram"],
        output_names=["class_logits"],
        dynamic_axes={"mel_spectrogram": {0: "batch", 1: "time"}, "class_logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
