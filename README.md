# САВОС — Triton inference server

Inference repository for the **САВОС** project (acoustic perception for autonomous vehicles, [TZ.md](TZ.md)). An 8-channel WAV from a circular microphone array is fed into a Triton-served pipeline that:

1. **Localizes** the source (coarse GCC-PHAT on the «two-square» pairs from TZ §4.1.2 → SRP-PHAT refinement in a ±5° sector)
2. **Selects** the microphone closest to the estimated DOA
3. **Extracts** a log-Mel spectrogram (librosa)
4. **Classifies** the event into 17 traffic-sound classes via an ONNX CNN
5. For emergency-vehicle sirens, **estimates** the distance from the peak amplitude (TZ §4.1.3)

Everything runs in two containers brought up by `docker compose up`. A Gradio UI on port 7860 is the demo face for the commission.

## Classes (17)

Car acceleration · Car braking · Car horn · Car idling · Motorcycle acceleration · Motorcycle idling · Tram acceleration · Tram bell · Tram braking · Tram passing · Truck acceleration · Truck braking · Truck horn · Truck idling · Siren Ambulance · Siren Police · Siren Firefighters

## Quick start

```bash
# 1. Put the trained classifier ONNX into the model repo
cp /path/to/your/model.onnx model_repository/classifier/1/model.onnx
# (or, until you have a real one, generate a tiny dummy model:)
python scripts/export_dummy_classifier.py

# 2. Verify labels.json — its `classes` list must match the logits order of model.onnx,
#    and `input_name` / `output_name` must match the ONNX I/O names.
$EDITOR model_repository/classifier/labels.json

# 3. (Optional) Synthesize a demo 8-channel WAV
python scripts/generate_demo_asset.py --doa 75 --out demo/assets/example_8ch.wav

# 4. Bring everything up
docker compose up --build
```

Then open http://localhost:7860 and drop in an 8-channel WAV.

## Endpoints

| Service        | URL                                   |
|----------------|---------------------------------------|
| Gradio UI      | http://localhost:7860                 |
| Triton HTTP    | http://localhost:8000/v2              |
| Triton gRPC    | `localhost:8001`                      |
| Triton metrics | http://localhost:8002/metrics         |

## CLI smoke test

```bash
docker compose exec gradio python -m demo.client --wav demo/assets/example_8ch.wav
```

Outputs JSON with `class_name`, `confidence`, `doa_deg`, `selected_mic`, `distance_m`, `is_emv`.

## Repository layout

```
model_repository/
├── pipeline/            # BLS orchestrator (the one model the client calls)
├── localizer/           # Python: GCC-PHAT + SRP-PHAT
├── channel_selector/    # Python: nearest mic by DOA
├── feature_extractor/   # Python: log-Mel via librosa
└── classifier/          # ONNX Runtime backend
    ├── 1/model.onnx     # ← user-supplied checkpoint
    └── labels.json      # classes, EmV subset, ONNX I/O names
demo/
├── app.py               # Gradio UI
├── client.py            # CLI client
└── assets/              # demo WAV(s)
docker/                  # Dockerfile.triton, Dockerfile.gradio
docker-compose.yml
scripts/                 # one-off helpers (dummy ONNX, synth WAV)
tests/                   # pytest unit tests
```

## Microphone indexing convention

Channel `i` (0..7) in the WAV corresponds to the microphone at azimuth `i·45°` counter-clockwise from the vehicle forward axis (+X, 0°). Mic 0 is forward, mic 2 is left, mic 4 is rear, mic 6 is right.

## Conventions and assumptions

- ONNX classifier input: `FP32[B, 1, 64, T]` (batch, channel, n_mels, time). Time axis is dynamic.
- ONNX classifier output: `FP32[B, 17]` logits.
- Mel parameters (configurable in `model_repository/feature_extractor/config.pbtxt`): `target_sr=22050, target_sec=3.0, n_mels=64, n_fft=1024, hop_length=512`. Adjust to match the training-time settings of your checkpoint.
- ONNX I/O tensor names (configurable in `labels.json`): `features`, `logits`. If your export used different names (often `input` / `output`), edit `labels.json` and `model_repository/classifier/config.pbtxt` accordingly — no re-export needed.
- Distance calibration constants `A0`, `R0` are in `model_repository/pipeline/config.pbtxt` `parameters` and require empirical calibration.

## Development

```bash
pip install -r requirements.txt
pytest tests/                                         # unit tests, no Triton needed
python scripts/export_dummy_classifier.py             # build a tiny random ONNX
python scripts/generate_demo_asset.py --doa 75        # build a synthetic 8-ch WAV
docker compose up --build                             # full stack
```

## Troubleshooting

- **Classifier fails to load**: most often a tensor-name mismatch. Inspect with `python -c "import onnx, sys; m=onnx.load(sys.argv[1]); print([i.name for i in m.graph.input], [o.name for o in m.graph.output])" model_repository/classifier/1/model.onnx` and update `labels.json` + `classifier/config.pbtxt`.
- **Shape mismatch in classifier**: the dummy-export shape is `[B, 1, 64, T]`. If your real model expects `[B, 64, T]` (no channel dim), drop the second `1` in `feature_extractor/config.pbtxt` output and `classifier/config.pbtxt` input, and remove the extra `[None, ...]` in `feature_extractor/1/model.py`.
- **DOA looks wrong by ~180°**: check microphone numbering matches the +X-CCW convention above.
- **Gradio cannot reach Triton**: `docker compose logs triton` for backend load errors; the Gradio container expects `TRITON_URL=triton:8001` (set by compose).
