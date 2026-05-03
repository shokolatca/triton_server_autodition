"""Gradio demo: upload an 8-channel WAV, see class + DOA + distance."""

from __future__ import annotations

import os

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .triton_client import (
    CLASSIFIER_LABELS,
    CLASSIFIER_MODEL_CHOICES,
    DEFAULT_CLASSIFIER_MODEL,
    MIC_COUNT,
    PipelineResult,
    load_8ch_wav,
    run_pipeline,
)


def _polar_plot(doa_deg: float, selected_mic: int) -> plt.Figure:
    fig = plt.figure(figsize=(4.2, 4.2))
    ax = fig.add_subplot(111, projection="polar")
    # Keep the DOA math unchanged; only rotate the visual zero-angle to 12 o'clock.
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)

    mic_angles_rad = np.deg2rad(np.arange(MIC_COUNT) * (360.0 / MIC_COUNT))
    radii = np.ones(MIC_COUNT)
    ax.scatter(mic_angles_rad, radii, s=90, c="#888", zorder=3, label="microphones")
    ax.scatter(
        [mic_angles_rad[selected_mic]],
        [1.0],
        s=180,
        c="#e63946",
        zorder=4,
        label=f"selected mic #{selected_mic}",
    )
    for i in range(MIC_COUNT):
        ax.text(mic_angles_rad[i], 1.12, str(i), ha="center", va="center", fontsize=8)

    ax.annotate(
        "",
        xy=(np.deg2rad(doa_deg), 1.0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=2.5),
    )
    ax.set_rmax(1.22)
    ax.set_rticks([])
    ax.set_title(f"DOA = {doa_deg:.1f}°", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7, frameon=False, ncol=2)
    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(bottom=0.2)
    return fig


def _format_distance(result: PipelineResult) -> str:
    if not result.is_emv:
        return "— (not an emergency vehicle)"
    if result.distance_m <= 0:
        return "n/a (calibration not set)"
    return f"{result.distance_m:.1f} m"


def _format_label(result: PipelineResult, label_names: list[str]) -> dict[str, float]:
    top_idx = np.argsort(result.probs)[::-1][:5]
    return {label_names[i]: float(result.probs[i]) for i in top_idx}


def infer(wav_path: str | None, classifier_model: str):
    if not wav_path:
        return {}, None, "upload an 8-channel WAV"
    audio, sr = load_8ch_wav(wav_path)
    result = run_pipeline(
        audio,
        sr,
        url=os.environ.get("TRITON_URL"),
        classifier_model=classifier_model or DEFAULT_CLASSIFIER_MODEL,
    )
    labels = CLASSIFIER_LABELS.get(classifier_model, CLASSIFIER_LABELS[DEFAULT_CLASSIFIER_MODEL])
    return _format_label(result, labels), _polar_plot(result.doa_deg, result.selected_mic), _format_distance(result)


def build_ui() -> gr.Blocks:
    description = (
        "## САВОС — Acoustic perception for autonomous vehicles\n"
        "Upload an 8-channel WAV recorded by the circular microphone array. "
        "The pipeline localizes the source (DOA), picks the nearest microphone, "
        "extracts a Mel-spectrogram, and classifies the sound across 17 traffic classes. "
        "For emergency-vehicle sirens an amplitude-based distance is reported."
    )
    with gr.Blocks(title="САВОС") as demo:
        gr.Markdown(description)
        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(sources=["upload"], type="filepath", label="8-channel WAV")
                classifier_in = gr.Dropdown(
                    choices=[(label, model_id) for model_id, label in CLASSIFIER_MODEL_CHOICES.items()],
                    value=DEFAULT_CLASSIFIER_MODEL,
                    label="Classifier model",
                )
                run_btn = gr.Button("Run pipeline", variant="primary")
            with gr.Column(scale=2):
                label_out = gr.JSON(label="Predicted class probabilities")
                plot_out = gr.Plot(label="Microphone array — DOA")
                dist_out = gr.Textbox(label="Distance to source", interactive=False)
        run_btn.click(infer, inputs=[audio_in, classifier_in], outputs=[label_out, plot_out, dist_out])
    return demo


def main() -> None:
    build_ui().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
