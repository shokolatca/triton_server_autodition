"""CLI client: send an 8-channel WAV to the Triton pipeline and pretty-print the result."""

from __future__ import annotations

import argparse
import json

from .triton_client import load_8ch_wav, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="SAVOS pipeline CLI client")
    parser.add_argument("--wav", required=True, help="Path to 8-channel WAV")
    parser.add_argument("--url", default=None, help="Triton gRPC URL (default: env TRITON_URL or localhost:8001)")
    args = parser.parse_args()

    audio, sr = load_8ch_wav(args.wav)
    result = run_pipeline(audio, sr, url=args.url)

    payload = {
        "class_id": result.class_id,
        "class_name": result.class_name,
        "confidence": round(result.confidence, 4),
        "doa_deg": round(result.doa_deg, 2),
        "selected_mic": result.selected_mic,
        "distance_m": round(result.distance_m, 2),
        "is_emv": result.is_emv,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
