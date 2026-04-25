"""DOA localization: coarse GCC-PHAT (two-square pairs) + sector SRP-PHAT refinement."""

import json

import numpy as np

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    pb_utils = None


MIC_COUNT = 8
TWO_SQUARE_PAIRS = [(0, 2), (2, 4), (4, 6), (6, 0), (1, 3), (3, 5), (5, 7), (7, 1)]
DEFAULT_RADIUS_M = 0.5
DEFAULT_SPEED_MS = 343.0


def mic_positions(radius_m: float) -> np.ndarray:
    """Return Nx2 array of microphone (x, y) positions for a circular array."""
    angles = np.deg2rad(np.arange(MIC_COUNT) * (360.0 / MIC_COUNT))
    return np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius_m


def gcc_phat(s_i: np.ndarray, s_j: np.ndarray, sr: int, max_tau: float | None = None) -> float:
    """Estimate delay of s_j relative to s_i in seconds. Positive => s_j arrives later."""
    n = int(2 ** np.ceil(np.log2(len(s_i) + len(s_j))))
    Xi = np.fft.rfft(s_i, n=n)
    Xj = np.fft.rfft(s_j, n=n)
    R = Xi * np.conj(Xj)
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n=n)
    cc = np.concatenate([cc[-(n // 2):], cc[: n // 2 + 1]])
    max_lag = n // 2
    if max_tau is not None:
        max_lag = min(max_lag, int(max_tau * sr))
    center = n // 2
    window = cc[center - max_lag: center + max_lag + 1]
    peak = int(np.argmax(np.abs(window)))
    lag_samples = peak - max_lag
    # GCC of (s_i, s_j) with R = Xi * conj(Xj) puts the peak at lag = -d when s_j is delayed by d.
    # Negate to return delay-of-j-relative-to-i.
    return -lag_samples / float(sr)


def coarse_doa_voting(
    audio: np.ndarray, sr: int, radius_m: float, speed_ms: float, step_deg: float = 1.0
) -> float:
    """Aggregate pair TDOAs by 1-degree voting; return best DOA in degrees."""
    mics = mic_positions(radius_m)
    measured = []
    for i, j in TWO_SQUARE_PAIRS:
        d_ij = float(np.linalg.norm(mics[i] - mics[j]))
        max_tau = d_ij / speed_ms
        tau = gcc_phat(audio[i], audio[j], sr, max_tau=max_tau)
        measured.append(((i, j), tau))
    candidates = np.arange(0.0, 360.0, step_deg)
    residuals = np.zeros_like(candidates)
    for k, theta in enumerate(candidates):
        u = np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))])
        total = 0.0
        for (i, j), tau in measured:
            tau_pred = float((mics[i] - mics[j]) @ u / speed_ms)
            total += (tau - tau_pred) ** 2
        residuals[k] = total
    return float(candidates[int(np.argmin(residuals))])


def srp_phat_sector(
    audio: np.ndarray,
    sr: int,
    center_deg: float,
    radius_m: float,
    speed_ms: float,
    sector_deg: float = 10.0,
    step_deg: float = 0.5,
) -> float:
    """Steered-response power with PHAT weighting in a ±sector_deg/2 window."""
    mics = mic_positions(radius_m)
    n = int(2 ** np.ceil(np.log2(audio.shape[1])))
    X = np.fft.rfft(audio, n=n, axis=1)
    X_phat = X / (np.abs(X) + 1e-12)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    omega = 2.0 * np.pi * freqs

    half = sector_deg / 2.0
    angles = np.arange(center_deg - half, center_deg + half + 1e-9, step_deg)
    best_power = -np.inf
    best_theta = float(center_deg)
    for theta_deg in angles:
        th = np.deg2rad(theta_deg)
        u = np.array([np.cos(th), np.sin(th)])
        p_dot_u = mics @ u  # [MIC_COUNT]
        phase = -omega[None, :] * (p_dot_u[:, None] / speed_ms)
        Y = np.sum(X_phat * np.exp(1j * phase), axis=0)
        power = float(np.sum(np.abs(Y) ** 2))
        if power > best_power:
            best_power = power
            best_theta = float(theta_deg)
    return best_theta % 360.0


def peak_amplitude(audio: np.ndarray, sr: int, window_sec: float = 3.0, peaks: int = 100) -> float:
    """Mean(|top peaks| + |bottom peaks|) on the channel with highest RMS, over the last window_sec."""
    n_window = min(audio.shape[1], int(window_sec * sr))
    window = audio[:, -n_window:]
    rms = np.sqrt(np.mean(window ** 2, axis=1) + 1e-12)
    best = window[int(np.argmax(rms))]
    if best.size < peaks:
        return float(np.mean(np.abs(best)))
    sorted_vals = np.sort(best)
    bottom = np.abs(sorted_vals[:peaks])
    top = np.abs(sorted_vals[-peaks:])
    return float(np.mean(np.concatenate([top, bottom])))


def localize(
    audio: np.ndarray, sr: int, radius_m: float = DEFAULT_RADIUS_M, speed_ms: float = DEFAULT_SPEED_MS
) -> tuple[float, float, float]:
    coarse = coarse_doa_voting(audio, sr, radius_m, speed_ms)
    precise = srp_phat_sector(audio, sr, coarse, radius_m, speed_ms)
    amp = peak_amplitude(audio, sr)
    return precise, coarse, amp


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"])
        params = {k: v["string_value"] for k, v in config.get("parameters", {}).items()}
        self.radius_m = float(params.get("ARRAY_RADIUS_M", DEFAULT_RADIUS_M))
        self.speed_ms = float(params.get("SPEED_OF_SOUND", DEFAULT_SPEED_MS))

    def execute(self, requests):
        responses = []
        for req in requests:
            audio = pb_utils.get_input_tensor_by_name(req, "audio").as_numpy().astype(np.float32)
            sr = int(pb_utils.get_input_tensor_by_name(req, "sample_rate").as_numpy()[0])
            precise, coarse, amp = localize(audio, sr, self.radius_m, self.speed_ms)
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor("doa_deg", np.array([precise], dtype=np.float32)),
                    pb_utils.Tensor("coarse_doa_deg", np.array([coarse], dtype=np.float32)),
                    pb_utils.Tensor("peak_amplitude", np.array([amp], dtype=np.float32)),
                ])
            )
        return responses
