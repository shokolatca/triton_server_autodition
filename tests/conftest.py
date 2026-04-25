"""Loaders so unit tests can import the Python-backend model modules without Triton installed."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_REPO = REPO_ROOT / "model_repository"


def load_model_module(model_name: str) -> ModuleType:
    path = MODEL_REPO / model_name / "1" / "model.py"
    spec = importlib.util.spec_from_file_location(f"sav_{model_name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
