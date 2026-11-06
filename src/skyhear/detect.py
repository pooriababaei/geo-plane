"""
skyhear.detect — Aircraft tone detection in audio spectrograms.

Computes STFT spectrograms with long windows for fine frequency resolution,
applies spectral whitening, and detects Doppler ridges from aircraft passes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal

logger = logging.getLogger(__name__)


@dataclass
class DetectedPass:
    """A single detected aircraft pass (Doppler ridge)."""

    # Time-frequency points along the ridge: (time_seconds, freq_hz)
    ridge_points: list[tuple[float, float]]
    # Approximate time of closest approach (steepest frequency descent)
    t_closest_approach: float
    # Duration of the pass in seconds
    duration: float
    # Signal-to-noise ratio estimate (optional)
    snr_db: float | None = None
    # Label for this pass
    label: str = ""

    @property
    def times(self) -> np.ndarray:
        return np.array([p[0] for p in self.ridge_points])

    @property
    def freqs(self) -> np.ndarray:
        return np.array([p[1] for p in self.ridge_points])


@dataclass
class DetectionResult:
    """Result of detecting aircraft passes in audio."""

    passes: list[DetectedPass] = field(default_factory=list)
    # Inter-pass intervals in seconds (strong fingerprint)
    pass_intervals: list[float] = field(default_factory=list)
    # Spectrogram data for visualization
    spectrogram: np.ndarray | None = None
    spec_times: np.ndarray | None = None
    spec_freqs: np.ndarray | None = None


def compute_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    window_seconds: float = 2.0,
    overlap_fraction: float = 0.9,
    freq_range: tuple[float, float] = (50.0, 4000.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a high-resolution STFT spectrogram optimized for detecting
    faint, slow-moving aircraft tones.

    Parameters
    ----------
    audio
        1D float32 audio samples.
    sample_rate
        Sample rate in Hz.
    window_seconds
        Window length in seconds. Longer = better frequency resolution.
        Spec recommends 1–4 s.
    overlap_fraction
        Fraction of window overlap (0.9 = 90% overlap for smooth time axis).
    freq_range
        (min_hz, max_hz) — frequency range to keep.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (Sxx, times, freqs) — power spectrogram, time axis, frequency axis.
    """
    nperseg = int(window_seconds * sample_rate)
    noverlap = int(nperseg * overlap_fraction)

    logger.info(
        "Computing STFT: window=%d samples (%.1f s), overlap=%.0f%%, nfft=%d",
        nperseg,
        window_seconds,
        overlap_fraction * 100,
        nperseg,
    )

    freqs, times, Zxx = signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,  # no zero-padding beyond window
        return_onesided=True,
    )

    # Convert to power spectrogram (magnitude squared)
    Sxx = np.abs(Zxx) ** 2

    # Crop to frequency range of interest
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs = freqs[freq_mask]
    Sxx = Sxx[freq_mask, :]

    logger.info(
        "Spectrogram: %d freq bins (%.1f–%.1f Hz), %d time frames (%.1f s)",
        len(freqs),
        freqs[0],
        freqs[-1],
        len(times),
        times[-1] - times[0],
    )

    return Sxx, times, freqs


def spectral_whitening(
    Sxx: np.ndarray,
    median_filter_size: int = 51,
) -> np.ndarray:
    """
    Apply spectral whitening to suppress broadband noise and reveal
    faint tonal ridges.

    For each frequency bin, divide by the running median across time,
    then take log. This normalizes out broadband variations and makes
    faint tones pop out.

    Parameters
    ----------
    Sxx
        Power spectrogram (freq x time).
    median_filter_size
        Size of the median filter along the time axis (in frames).
        Larger = smoother background estimate.

    Returns
    -------
    np.ndarray
        Whitened spectrogram (same shape).
    """
    # Compute running median along time axis for each frequency bin
    # This estimates the "background" level
    background = ndimage.median_filter(Sxx, size=(1, median_filter_size))

    # Avoid division by zero
    background = np.maximum(background, np.finfo(float).eps)

    # Whiten: ratio above background
    whitened = Sxx / background

    # Log-scale for better dynamic range
    whitened = np.log1p(whitened)

    return whitened


def find_ridges_manual_seeds(
    Sxx_whitened: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    seed_points: list[tuple[float, float]],
    track_window_hz: float = 20.0,
    min_ridge_length: int = 10,
) -> list[DetectedPass]:
    """
    Track ridges from manually provided seed points.

    From each seed point (t, f), track the peak in both directions
    along time, following the strongest peak within a frequency window.

    Parameters
    ----------
    Sxx_whitened
        Whitened spectrogram.
    times
        Time axis of the spectrogram.
    freqs
        Frequency axis of the spectrogram.
    seed_points
        List of (time_seconds, freq_hz) seed points — one per ridge.
    track_window_hz
        How far (in Hz) to search around the current peak when tracking.
    min_ridge_length
        Minimum number of points for a valid ridge.

    Returns
    -------
    list[DetectedPass]
        Detected passes, one per successfully tracked seed.
    """
    freq_res = freqs[1] - freqs[0]
    window_bins = max(1, int(track_window_hz / freq_res))

    passes = []

    for seed_idx, (seed_t, seed_f) in enumerate(seed_points):
        # Find nearest spectrogram indices
        t_idx = np.argmin(np.abs(times - seed_t))
        f_idx = np.argmin(np.abs(freqs - seed_f))

        # Track ridge in both directions from seed
        ridge_indices = _track_ridge_from_seed(
            Sxx_whitened, t_idx, f_idx, window_bins
        )

        if len(ridge_indices) < min_ridge_length:
            logger.warning(
                "Seed %d at (%.1f s, %.1f Hz): ridge too short (%d points), skipping",
                seed_idx,
                seed_t,
                seed_f,
                len(ridge_indices),
            )
            continue

        # Convert indices to physical coordinates
        ridge_points = [
            (float(times[ti]), float(freqs[fi])) for ti, fi in ridge_indices
        ]

        # Find closest approach: point of maximum frequency descent rate
        ridge_freqs = np.array([f for _, f in ridge_points])
        ridge_times = np.array([t for t, _ in ridge_points])
        df_dt = np.gradient(ridge_freqs, ridge_times)
        t_ca_idx = np.argmin(df_dt)  # steepest descent
        t_closest_approach = ridge_times[t_ca_idx]

        duration = ridge_times[-1] - ridge_times[0]

        detected = DetectedPass(
            ridge_points=ridge_points,
            t_closest_approach=t_closest_approach,
            duration=duration,
            label=f"pass_{seed_idx}",
        )
        passes.append(detected)
        logger.info(
            "Seed %d: tracked %d points over %.1f s, closest approach at %.1f s",
            seed_idx,
            len(ridge_points),
            duration,
            t_closest_approach,
        )

    # Compute inter-pass intervals
    if len(passes) >= 2:
        cas = sorted(p.t_closest_approach for p in passes)
        intervals = [cas[i + 1] - cas[i] for i in range(len(cas) - 1)]
    else:
        intervals = []

    return passes


def _track_ridge_from_seed(
    Sxx: np.ndarray,
    t_start: int,
    f_start: int,
    window_bins: int,
) -> list[tuple[int, int]]:
    """
    Track a ridge from a seed point in both time directions.

    At each step, find the peak within a window centered on the
    current frequency bin.
    """
    n_freqs, n_times = Sxx.shape
    ridge = [(t_start, f_start)]

    # Track forward
    current_f = f_start
    for t in range(t_start + 1, n_times):
        f_lo = max(0, current_f - window_bins)
        f_hi = min(n_freqs, current_f + window_bins + 1)
        segment = Sxx[f_lo:f_hi, t]

        # Stop if signal drops too low (below 2x median of the column)
        col_median = np.median(Sxx[:, t])
        if segment.max() < 2.0 * col_median:
            break

        peak_offset = np.argmax(segment)
        current_f = f_lo + peak_offset
        ridge.append((t, current_f))

    # Track backward
    current_f = f_start
    backward = []
    for t in range(t_start - 1, -1, -1):
        f_lo = max(0, current_f - window_bins)
        f_hi = min(n_freqs, current_f + window_bins + 1)
        segment = Sxx[f_lo:f_hi, t]

        col_median = np.median(Sxx[:, t])
        if segment.max() < 2.0 * col_median:
            break

        peak_offset = np.argmax(segment)
        current_f = f_lo + peak_offset
        backward.append((t, current_f))

    # Combine: backward (reversed) + seed + forward
    backward.reverse()
    ridge = backward + ridge

    return ridge


def auto_detect_ridges(
    Sxx_whitened: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    threshold_sigma: float = 2.5,
    min_ridge_duration_s: float = 8.0,
    track_window_hz: float = 20.0,
) -> list[DetectedPass]:
    """
    Automatically detect Doppler ridges in a whitened spectrogram.

    Uses dynamic programming-style peak tracking: find high-energy
    seed points, then track ridges from them.

    Parameters
    ----------
    Sxx_whitened
        Whitened spectrogram.
    times
        Time axis.
    freqs
        Frequency axis.
    threshold_sigma
        Detect peaks above this many standard deviations of the whitened
        spectrogram.
    min_ridge_duration_s
        Minimum ridge duration to keep (seconds).
    track_window_hz
        Tracking window width in Hz.

    Returns
    -------
    list[DetectedPass]
        Automatically detected passes.
    """
    # Find seed points: peaks well above the background
    mean_val = np.mean(Sxx_whitened)
    std_val = np.std(Sxx_whitened)
    threshold = mean_val + threshold_sigma * std_val

    # Find local maxima above threshold
    # Use a small structure for local max detection
    local_max = ndimage.maximum_filter(Sxx_whitened, size=(5, 5))
    peaks_mask = (Sxx_whitened == local_max) & (Sxx_whitened > threshold)

    peak_coords = np.argwhere(peaks_mask)  # (freq_idx, time_idx)
    if len(peak_coords) == 0:
        logger.warning("No peaks found above threshold (%.2f sigma)", threshold_sigma)
        return []

    # Sort by intensity (strongest first) for greedy tracking
    peak_intensities = Sxx_whitened[peak_coords[:, 0], peak_coords[:, 1]]
    order = np.argsort(-peak_intensities)
    peak_coords = peak_coords[order]

    logger.info("Found %d candidate seed peaks above %.1f sigma", len(peak_coords), threshold_sigma)

    # Greedy tracking: track from each seed, mark covered time frames
    freq_res = freqs[1] - freqs[0]
    window_bins = max(1, int(track_window_hz / freq_res))
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    min_ridge_frames = int(min_ridge_duration_s / dt)

    covered_times: set[int] = set()
    passes = []

    for f_idx, t_idx in peak_coords:
        if t_idx in covered_times:
            continue

        ridge = _track_ridge_from_seed(Sxx_whitened, t_idx, f_idx, window_bins)

        if len(ridge) < min_ridge_frames:
            continue

        # Mark time frames as covered
        for ti, _ in ridge:
            covered_times.add(ti)

        # Convert to physical coordinates
        ridge_points = [(float(times[ti]), float(freqs[fi])) for ti, fi in ridge]

        ridge_freqs = np.array([f for _, f in ridge_points])
        ridge_times = np.array([t for t, _ in ridge_points])
        df_dt = np.gradient(ridge_freqs, ridge_times)
        t_ca_idx = np.argmin(df_dt)
        t_closest_approach = ridge_times[t_ca_idx]

        detected = DetectedPass(
            ridge_points=ridge_points,
            t_closest_approach=t_closest_approach,
            duration=ridge_times[-1] - ridge_times[0],
            label=f"auto_pass_{len(passes)}",
        )
        passes.append(detected)

    # Sort by time of closest approach
    passes.sort(key=lambda p: p.t_closest_approach)

    logger.info("Auto-detected %d passes", len(passes))
    return passes


def plot_spectrogram(
    Sxx: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    passes: list[DetectedPass] | None = None,
    title: str = "Spectrogram",
    output_path: str | None = None,
    figsize: tuple[int, int] = (16, 8),
    cmap: str = "inferno",
    log_scale: bool = True,
) -> plt.Figure:
    """
    Plot a spectrogram with optional ridge overlays.

    Parameters
    ----------
    Sxx
        Power or whitened spectrogram (freq x time).
    times
        Time axis in seconds.
    freqs
        Frequency axis in Hz.
    passes
        Detected passes to overlay on the spectrogram.
    title
        Plot title.
    output_path
        If provided, save the figure to this path.
    figsize
        Figure size (width, height) in inches.
    cmap
        Colormap.
    log_scale
        If True, plot in dB scale.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    plot_data = 10 * np.log10(Sxx + np.finfo(float).eps) if log_scale else Sxx

    im = ax.pcolormesh(
        times,
        freqs,
        plot_data,
        cmap=cmap,
        shading="gouraud",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Power (dB)" if log_scale else "Power")

    # Overlay detected ridges
    if passes:
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(passes), 1)))
        for i, p in enumerate(passes):
            ax.plot(
                p.times,
                p.freqs,
                ".-",
                color=colors[i],
                linewidth=1.5,
                markersize=2,
                label=p.label or f"Pass {i}",
            )
            # Mark closest approach
            ax.axvline(
                p.t_closest_approach,
                color=colors[i],
                linestyle="--",
                alpha=0.5,
            )
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Spectrogram saved to %s", output_path)

    return fig
