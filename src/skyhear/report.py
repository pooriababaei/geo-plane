"""
skyhear.report — Generate result reports and visualizations.

Produces JSON results, overlay plots (observed vs predicted curves),
and map links for the top candidate locations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .detect import DetectedPass
from .model import predict_doppler_curve
from .search import CandidateLocation

logger = logging.getLogger(__name__)


def generate_results_json(
    candidates: list[CandidateLocation],
    output_path: str | Path,
) -> Path:
    """
    Write ranked candidates and flight matches to a JSON file.

    Parameters
    ----------
    candidates
        Ranked candidate locations (best first).
    output_path
        Path to write the JSON file.

    Returns
    -------
    Path
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "candidates": [
            {
                "rank": i + 1,
                "lat": c.lat,
                "lon": c.lon,
                "total_score": round(c.total_score, 4),
                "n_unexplained_passes": c.n_unexplained_passes,
                "negative_evidence_penalty": round(c.negative_evidence_penalty, 4),
                "flight_matches": [
                    {
                        "icao24": m.icao24,
                        "callsign": m.callsign,
                        "fitted_f0_hz": round(m.fitted_f0, 2),
                        "fitted_clock_offset_s": round(m.fitted_offset, 2),
                        "rms_residual_hz": round(m.rms_residual_hz, 4),
                    }
                    for m in c.flight_matches
                ],
                "google_maps_link": (
                    f"https://www.google.com/maps?q={c.lat:.6f},{c.lon:.6f}"
                ),
            }
            for i, c in enumerate(candidates)
        ],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results written to %s", output_path)
    return output_path


def plot_overlay(
    passes: list[DetectedPass],
    candidate: CandidateLocation,
    flight_tracks: dict[str, dict],
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (16, 6),
) -> plt.Figure:
    """
    Plot observed ridges overlaid with the best-fit predicted Doppler
    curves for the top candidate.

    Parameters
    ----------
    passes
        Detected passes.
    candidate
        The candidate location to visualize.
    flight_tracks
        Dict of icao24 → {timestamps, lat, lon, alt_m} for matched flights.
    output_path
        If provided, save the figure.
    figsize
        Figure size.

    Returns
    -------
    plt.Figure
        The overlay figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot observed ridges
    t_min_obs = float("inf")
    t_max_obs = -float("inf")
    for i, p in enumerate(passes):
        t_min_obs = min(t_min_obs, float(np.min(p.times)))
        t_max_obs = max(t_max_obs, float(np.max(p.times)))
        ax.plot(
            p.times,
            p.freqs,
            "o-",
            markersize=2,
            linewidth=1,
            label=f"Observed Pass {i+1} ({p.duration:.1f}s)",
            alpha=0.7,
        )

    # Plot predicted curves for matched flights
    for match in candidate.flight_matches:
        track = flight_tracks.get(match.icao24)
        if track is None:
            continue

        pred_times, pred_freqs = predict_doppler_curve(
            candidate.lat,
            candidate.lon,
            0.0,  # ground level
            np.array(track["lat"]),
            np.array(track["lon"]),
            np.array(track["alt_m"]),
            np.array(track["timestamps"]),
            f0=match.fitted_f0,
            clock_offset=match.fitted_offset,
        )

        # Align time axis if pred_times is absolute UTC and observed is relative
        if len(pred_times) > 0 and pred_times[0] > 1e6 and t_max_obs < 1e6:
            # find the base offset
            base_t = track["timestamps"][0]
            # align to observation window
            mask = (pred_times >= t_min_obs + base_t - 5.0) & (pred_times <= t_max_obs + base_t + 5.0)
            if np.any(mask):
                pred_times = pred_times[mask] - base_t
                pred_freqs = pred_freqs[mask]

        ax.plot(
            pred_times,
            pred_freqs,
            "--",
            linewidth=2,
            label=f"Predicted: {match.callsign or match.icao24} (f0={match.fitted_f0:.1f} Hz)",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(
        f"Doppler Overlay — Candidate ({candidate.lat:.5f}, {candidate.lon:.5f}), "
        f"Score: {candidate.total_score:.2f}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Overlay plot saved to %s", output_path)

    return fig


def print_summary(candidates: list[CandidateLocation], top_n: int = 5) -> str:
    """
    Generate a human-readable summary of top candidates.

    Returns the summary string.
    """
    lines = [
        "=" * 70,
        "  AIRCRAFT-NOISE GEOLOCATION RESULTS",
        "=" * 70,
        "",
    ]

    for i, c in enumerate(candidates[:top_n]):
        lines.append(f"  #{i + 1}  ({c.lat:.6f}, {c.lon:.6f})  score={c.total_score:.2f}")
        lines.append(
            f"       https://www.google.com/maps?q={c.lat:.6f},{c.lon:.6f}"
        )

        for m in c.flight_matches:
            lines.append(
                f"       ├─ {m.callsign or m.icao24}: "
                f"f0={m.fitted_f0:.1f} Hz, "
                f"offset={m.fitted_offset:+.1f} s, "
                f"residual={m.rms_residual_hz:.2f} Hz"
            )

        if c.n_unexplained_passes > 0:
            lines.append(f"       └─ {c.n_unexplained_passes} unexplained pass(es)")

        lines.append("")

    lines.append("=" * 70)
    summary = "\n".join(lines)
    print(summary)
    return summary
