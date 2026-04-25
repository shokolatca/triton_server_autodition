"""Generate a synthetic 8-channel WAV (broadband, planar wavefront) for the demo.

Run from repo root:
    python scripts/generate_demo_asset.py --doa 75 --out demo/assets/example_8ch.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf

from tests.synth import synth_8ch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doa", type=float, default=75.0, help="Direction of arrival in degrees")
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("demo/assets/example_8ch.wav"),
        help="Output path (8-channel WAV)",
    )
    args = parser.parse_args()

    audio = synth_8ch(
        doa_deg=args.doa,
        sample_rate=args.sr,
        duration_sec=args.duration,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), audio.T, args.sr, subtype="PCM_24")
    print(f"wrote {args.out}  (DOA={args.doa}°, sr={args.sr}, {args.duration}s)")


if __name__ == "__main__":
    main()
