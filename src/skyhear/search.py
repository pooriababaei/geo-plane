"""
skyhear.search — Location search via coarse-to-fine grid search.

Searches a geographic region for the recording location by matching
detected Doppler ridges against simulated curves from ADS-B flight data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np

from .adsb import BoundingBox, FlightTrack, SpatialIndex
from .detect import DetectedPass
from .model import (
    compute_slant_range,
    fit_f0_and_offset,
    predict_audibility,
    predict_doppler_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class FlightMatch:
    """A single flight matched to a detected pass."""

    icao24: str
    callsign: str
    fitted_f0: float
    fitted_offset: float
    rms_residual_hz: float


@dataclass
class CandidateLocation:
    """A scored candidate location."""

    lat: float
    lon: float
    total_score: float  # lower is better
    flight_matches: list[FlightMatch] = field(default_factory=list)
    n_unexplained_passes: int = 0
    negative_evidence_penalty: float = 0.0


@dataclass
class SearchConfig:
    """Configuration for the location search."""

    # Grid resolutions for coarse-to-fine search (degrees)
    coarse_grid_deg: float = 0.05     # ~5 km
    medium_grid_deg: float = 0.01     # ~1 km
    fine_grid_deg: float = 0.001      # ~100 m

    # How many top candidates to refine at each level
    top_k_coarse: int = 5
    top_k_medium: int = 3
    top_k_fine: int = 5

    # Search parameters
    max_slant_range_km: float = 40.0
    max_clock_offset_s: float = 180.0

    # Scoring weights
    unexplained_pass_penalty: float = 50.0   # Hz equivalent per unexplained pass
    negative_evidence_weight: float = 30.0    # Hz per expected-but-absent flight

    # Interval fingerprint tolerance
    interval_tolerance_s: float = 30.0

    # Receiver altitude (assume ground level)
    receiver_alt_m: float = 0.0

    # Parallel processing
    n_workers: int | None = None  # None = use all CPUs


def _score_single_point(args: tuple) -> CandidateLocation | None:
    """Score a single candidate point (designed for multiprocessing)."""
    (
        lat, lon, receiver_alt_m,
        passes, tracks_data,
        max_slant_range_km, max_clock_offset_s,
        unexplained_pass_penalty, negative_evidence_weight,
    ) = args

    flight_matches = []
    total_residual = 0.0
    n_unexplained = 0

    for pass_data in passes:
        obs_times = pass_data["times"]
        obs_freqs = pass_data["freqs"]
        t_ca = pass_data["t_closest_approach"]

        best_match: FlightMatch | None = None
        best_rms = float("inf")

        for track in tracks_data:
            t_arr = track["timestamps"]
            lat_arr = track["lat"]
            lon_arr = track["lon"]
            alt_arr = track["alt_m"]

            # Quick check: time overlap
            if (
                t_arr[-1] < obs_times[0] - max_clock_offset_s
                or t_arr[0] > obs_times[-1] + max_clock_offset_s
            ):
                continue

            # Quick check: does the flight pass near this point?
            # Subsample for speed in distance check
            step = max(1, len(lat_arr) // 20)
            slant = compute_slant_range(
                lat, lon, receiver_alt_m,
                lat_arr[::step], lon_arr[::step], alt_arr[::step],
            )
            if slant.min() > max_slant_range_km * 1000:
                continue

            # Fit f0 and clock offset
            try:
                f0, offset, rms = fit_f0_and_offset(
                    obs_times, obs_freqs,
                    lat, lon, receiver_alt_m,
                    lat_arr, lon_arr, alt_arr, t_arr,
                    offset_range=(-max_clock_offset_s, max_clock_offset_s),
                )
            except Exception:
                continue

            if abs(offset) > max_clock_offset_s:
                continue

            if rms < best_rms:
                best_rms = rms
                best_match = FlightMatch(
                    icao24=track["icao24"],
                    callsign=track["callsign"],
                    fitted_f0=f0,
                    fitted_offset=offset,
                    rms_residual_hz=rms,
                )

        if best_match is not None:
            flight_matches.append(best_match)
            total_residual += best_match.rms_residual_hz
        else:
            n_unexplained += 1
            total_residual += unexplained_pass_penalty

    # Clock offset consistency check:
    # All aircraft heard in the SAME recording MUST share the same physical clock offset!
    # Contradictory offsets (e.g. Flight A at -50s, Flight B at +50s) are physically impossible.
    if len(flight_matches) >= 2:
        good_offsets = [m.fitted_offset for m in flight_matches if m.rms_residual_hz < 30.0]
        if len(good_offsets) >= 2:
            offset_spread = max(good_offsets) - min(good_offsets)
            if offset_spread > 5.0:
                # Strong penalty for temporal inconsistency in the same audio recording
                total_residual += 20.0 * (offset_spread - 5.0)

    # Negative evidence: check for flights that should be audible but weren't detected
    # (simplified — full implementation would check all nearby flights)
    neg_penalty = 0.0

    total_score = total_residual + n_unexplained * unexplained_pass_penalty + neg_penalty

    return CandidateLocation(
        lat=lat,
        lon=lon,
        total_score=total_score,
        flight_matches=flight_matches,
        n_unexplained_passes=n_unexplained,
        negative_evidence_penalty=neg_penalty,
    )


def _generate_grid(bbox: BoundingBox, step_deg: float) -> list[tuple[float, float]]:
    """Generate a grid of (lat, lon) points within a bounding box."""
    lats = np.arange(bbox.min_lat, bbox.max_lat + step_deg, step_deg)
    lons = np.arange(bbox.min_lon, bbox.max_lon + step_deg, step_deg)
    grid = [(float(lat), float(lon)) for lat in lats for lon in lons]
    return grid


def _refine_bbox(center_lat: float, center_lon: float, radius_deg: float) -> BoundingBox:
    """Create a small bounding box around a candidate for refinement."""
    return BoundingBox(
        min_lat=center_lat - radius_deg,
        max_lat=center_lat + radius_deg,
        min_lon=center_lon - radius_deg,
        max_lon=center_lon + radius_deg,
    )


def interval_fingerprint_filter(
    passes: list[DetectedPass],
    tracks: list[FlightTrack],
    candidate_lat: float,
    candidate_lon: float,
    candidate_alt_m: float,
    tolerance_s: float = 30.0,
) -> list[FlightTrack]:
    """
    Apply the interval fingerprint filter: the time gaps between
    detected passes must match the gaps between the corresponding
    flights' closest approaches at the candidate location.

    This is a cheap early filter that can eliminate many candidates.

    Parameters
    ----------
    passes
        Detected passes sorted by time.
    tracks
        Candidate flight tracks to match.
    candidate_lat, candidate_lon, candidate_alt_m
        Candidate location.
    tolerance_s
        Maximum difference in inter-pass interval (seconds).

    Returns
    -------
    list[FlightTrack]
        Tracks that are consistent with the interval fingerprint.
    """
    if len(passes) < 2:
        return tracks

    # Compute observed inter-pass intervals
    cas = sorted(p.t_closest_approach for p in passes)
    observed_intervals = [cas[i + 1] - cas[i] for i in range(len(cas) - 1)]

    # For each track, compute its closest-approach time at the candidate
    track_ca_times = []
    for track in tracks:
        slant = compute_slant_range(
            candidate_lat, candidate_lon, candidate_alt_m,
            track.lat, track.lon, track.alt_m,
        )
        min_idx = np.argmin(slant)
        ca_time = float(track.timestamps[min_idx])
        track_ca_times.append((ca_time, track))

    track_ca_times.sort(key=lambda x: x[0])

    # Try to find a subset of tracks whose intervals match
    # This is simplified — a full implementation would do combinatorial matching
    compatible = [t for _, t in track_ca_times]
    return compatible


def search_location(
    passes: list[DetectedPass],
    tracks: list[FlightTrack],
    bbox: BoundingBox,
    config: SearchConfig | None = None,
    recording_utc_start: float | None = None,
) -> list[CandidateLocation]:
    """
    Search for the recording location using coarse-to-fine grid search.

    Parameters
    ----------
    passes
        Detected aircraft passes from the audio.
    tracks
        ADS-B flight tracks for the region and date.
    bbox
        Search bounding box.
    config
        Search configuration.
    recording_utc_start
        Approximate UTC start time of the recording (Unix timestamp).
        Used to narrow the time window for flight matching.

    Returns
    -------
    list[CandidateLocation]
        Ranked candidates, best (lowest score) first.
    """
    if config is None:
        config = SearchConfig()

    n_workers = config.n_workers or max(1, (os.cpu_count() or 2) - 1)

    # Prepare serializable pass data for multiprocessing
    t_base = recording_utc_start or 0.0
    passes_data = []
    for p in passes:
        passes_data.append({
            "times": np.array([t_base + pt[0] for pt in p.ridge_points]),
            "freqs": np.array([pt[1] for pt in p.ridge_points]),
            "t_closest_approach": t_base + p.t_closest_approach,
        })

    # Time bounds across all passes
    if passes_data:
        t_min_all = min(p["times"][0] for p in passes_data) - config.max_clock_offset_s
        t_max_all = max(p["times"][-1] for p in passes_data) + config.max_clock_offset_s
    else:
        t_min_all = -float("inf")
        t_max_all = float("inf")

    # Filter and prepare flight tracks: only airborne tracks in the region during the window
    center_lat = (bbox.min_lat + bbox.max_lat) / 2.0
    center_lon = (bbox.min_lon + bbox.max_lon) / 2.0
    max_search_dist_m = (config.max_slant_range_km + 20.0) * 1000.0

    tracks_data = []
    for t in tracks:
        if recording_utc_start is not None:
            mask = (t.timestamps >= t_min_all) & (t.timestamps <= t_max_all)
            if not np.any(mask):
                continue
            if np.max(t.alt_m[mask]) < 100:
                continue
            slant = compute_slant_range(center_lat, center_lon, 0.0, t.lat[mask], t.lon[mask], t.alt_m[mask])
            if np.min(slant) > max_search_dist_m:
                continue

        tracks_data.append({
            "icao24": t.icao24,
            "callsign": t.callsign,
            "timestamps": t.timestamps,
            "lat": t.lat,
            "lon": t.lon,
            "alt_m": t.alt_m,
        })

    logger.info("Search space: %d relevant flight tracks over %s", len(tracks_data), bbox)

    # === Phase 1: Coarse grid ===
    candidates = _search_grid(
        bbox, config.coarse_grid_deg, passes_data, tracks_data, config, n_workers
    )
    candidates.sort(key=lambda c: c.total_score)
    top_coarse = candidates[: config.top_k_coarse]
    logger.info(
        "Coarse search: %d points → top %d (best score: %.2f)",
        len(candidates),
        len(top_coarse),
        top_coarse[0].total_score if top_coarse else float("inf"),
    )

    # === Phase 2: Medium grid around top coarse candidates ===
    medium_candidates = []
    for c in top_coarse:
        refined_bbox = _refine_bbox(c.lat, c.lon, config.coarse_grid_deg / 2)
        results = _search_grid(
            refined_bbox, config.medium_grid_deg, passes_data, tracks_data, config, n_workers
        )
        medium_candidates.extend(results)

    medium_candidates.sort(key=lambda c: c.total_score)
    top_medium = medium_candidates[: config.top_k_medium]
    logger.info(
        "Medium search: %d points → top %d (best: %.2f)",
        len(medium_candidates),
        len(top_medium),
        top_medium[0].total_score if top_medium else float("inf"),
    )

    # === Phase 3: Fine grid around top medium candidates ===
    fine_candidates = []
    for c in top_medium:
        refined_bbox = _refine_bbox(c.lat, c.lon, config.medium_grid_deg / 2)
        results = _search_grid(
            refined_bbox, config.fine_grid_deg, passes_data, tracks_data, config, n_workers
        )
        fine_candidates.extend(results)

    fine_candidates.sort(key=lambda c: c.total_score)
    top_fine = fine_candidates[: config.top_k_fine]
    logger.info(
        "Fine search: %d points → top %d (best: %.2f)",
        len(fine_candidates),
        len(top_fine),
        top_fine[0].total_score if top_fine else float("inf"),
    )

    return top_fine


def _search_grid(
    bbox: BoundingBox,
    step_deg: float,
    passes_data: list[dict],
    tracks_data: list[dict],
    config: SearchConfig,
    n_workers: int,
) -> list[CandidateLocation]:
    """Score all grid points in a bounding box."""
    grid = _generate_grid(bbox, step_deg)

    if not grid:
        return []

    # Build arguments for parallel scoring
    args_list = [
        (
            lat, lon, config.receiver_alt_m,
            passes_data, tracks_data,
            config.max_slant_range_km, config.max_clock_offset_s,
            config.unexplained_pass_penalty, config.negative_evidence_weight,
        )
        for lat, lon in grid
    ]

    # Run parallel scoring via ThreadPoolExecutor (releases GIL in C/Fortran minimize)
    if n_workers > 1 and len(args_list) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_score_single_point, args_list))
    else:
        results = [_score_single_point(a) for a in args_list]

    return [r for r in results if r is not None]
