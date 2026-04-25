"""Gradio demo: upload an 8-channel WAV, see class + DOA + distance."""

from __future__ import annotations

import os

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .triton_client import MIC_COUNT, PipelineResult, load_8ch_wav, run_pipeline


def _polar_plot(doa_deg: float, selected_mic: int) -> plt.Figure:
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    mic_angles_rad = np.deg2rad(np.arange(MIC_COUNT) * (360.0 / MIC_COUNT))
    radii = np.ones(MIC_COUNT)
    ax.scatter(mic_angles_rad, radii, s=120, c="#888", zorder=3, label="microphones")
    ax.scatter(
        [mic_angles_rad[selected_mic]],
        [1.0],
        s=240,
        c="#e63946",
        zorder=4,
        label=f"selected mic #{selected_mic}",
    )
    for i in range(MIC_COUNT):
        ax.text(mic_angles_rad[i], 1.18, str(i), ha="center", va="center", fontsize=9)

    ax.annotate(
        "",
        xy=(np.deg2rad(doa_deg), 1.0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=2.5),
    )
    ax.set_rmax(1.4)
    ax.set_rticks([])
    ax.set_title(f"DOA = {doa_deg:.1f}°", pad=18)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), fontsize=8, frameon=False)
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


LABELS = [
    "Car acceleration", "Car braking", "Car horn", "Car idling",
    "Motorcycle acceleration", "Motorcycle idling",
    "Tram acceleration", "Tram bell", "Tram braking", "Tram passing",
    "Truck acceleration", "Truck braking", "Truck horn", "Truck idling",
    "Siren Ambulance", "Siren Police", "Siren Firefighters",
]


def infer(wav_path: str | None):
    if not wav_path:
        return {}, None, "upload an 8-channel WAV"
    audio, sr = load_8ch_wav(wav_path)
    result = run_pipeline(audio, sr, url=os.environ.get("TRITON_URL"))
    return _format_label(result, LABELS), _polar_plot(result.doa_deg, result.selected_mic), _format_distance(result)


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
                run_btn = gr.Button("Run pipeline", variant="primary")
            with gr.Column(scale=2):
                label_out = gr.Label(num_top_classes=5, label="Predicted class")
                plot_out = gr.Plot(label="Microphone array — DOA")
                dist_out = gr.Textbox(label="Distance to source", interactive=False)
        run_btn.click(infer, inputs=[audio_in], outputs=[label_out, plot_out, dist_out])
    return demo


def main() -> None:
    build_ui().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
