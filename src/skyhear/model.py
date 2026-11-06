"""
skyhear.model — Forward Doppler simulator.

Given a receiver position and a flight track, simulates the observed
frequency curve accounting for Doppler shift, retarded time, and
Earth's curvature (ECEF coordinates).
"""

from __future__ import annotations

import logging

import numpy as np
from pyproj import Transformer

logger = logging.getLogger(__name__)

# Speed of sound at sea level, 15°C (m/s)
# Can be adjusted for temperature: c ≈ 331.3 + 0.606 * T_celsius
SPEED_OF_SOUND = 343.0

# WGS84 → ECEF transformer (cached)
_transformer_to_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
_transformer_from_ecef = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)


def lla_to_ecef(
    lat: float | np.ndarray,
    lon: float | np.ndarray,
    alt_m: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert geodetic (lat, lon, altitude) to ECEF (x, y, z) coordinates.

    Parameters
    ----------
    lat, lon
        Latitude and longitude in degrees.
    alt_m
        Altitude above WGS84 ellipsoid in meters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (x, y, z) in meters.
    """
    x, y, z = _transformer_to_ecef.transform(lon, lat, alt_m)
    return np.asarray(x), np.asarray(y), np.asarray(z)


def compute_slant_range(
    receiver_lat: float,
    receiver_lon: float,
    receiver_alt_m: float,
    aircraft_lat: np.ndarray,
    aircraft_lon: np.ndarray,
    aircraft_alt_m: np.ndarray,
) -> np.ndarray:
    """
    Compute slant range (3D distance) between a ground receiver and
    aircraft positions, using ECEF coordinates for Earth curvature.

    Returns
    -------
    np.ndarray
        Slant range in meters, same length as aircraft arrays.
    """
    rx, ry, rz = lla_to_ecef(receiver_lat, receiver_lon, receiver_alt_m)
    ax, ay, az = lla_to_ecef(aircraft_lat, aircraft_lon, aircraft_alt_m)

    dx = ax - rx
    dy = ay - ry
    dz = az - rz

    return np.sqrt(dx**2 + dy**2 + dz**2)


def compute_range_rate(
    receiver_lat: float,
    receiver_lon: float,
    receiver_alt_m: float,
    aircraft_lat: np.ndarray,
    aircraft_lon: np.ndarray,
    aircraft_alt_m: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute slant range and its time derivative (range rate) numerically.

    The range rate r'(t) determines the Doppler shift:
      f_obs = f0 * c / (c + r'(t_e))

    Parameters
    ----------
    receiver_lat, receiver_lon, receiver_alt_m
        Receiver position.
    aircraft_lat, aircraft_lon, aircraft_alt_m
        Aircraft trajectory positions (arrays).
    timestamps
        Corresponding Unix timestamps (uniform 1 s spacing assumed).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (slant_range, range_rate) — both in meters and m/s.
    """
    slant_range = compute_slant_range(
        receiver_lat, receiver_lon, receiver_alt_m,
        aircraft_lat, aircraft_lon, aircraft_alt_m,
    )

    # Numerical derivative: dr/dt
    # Use central differences for interior points, forward/backward at edges
    range_rate = np.gradient(slant_range, timestamps)

    return slant_range, range_rate


def retarded_time_correction(
    timestamps: np.ndarray,
    slant_range: np.ndarray,
    speed_of_sound: float = SPEED_OF_SOUND,
    iterations: int = 2,
) -> np.ndarray:
    """
    Compute retarded (emission) times iteratively.

    Sound arriving at time t was emitted at t_e ≈ t − r/c.
    Because the aircraft moves during propagation, we iterate:
      t_e^(0) = t - r(t)/c
      t_e^(n+1) = t - r(t_e^(n))/c

    Parameters
    ----------
    timestamps
        Observation times (uniform grid).
    slant_range
        Slant range at each timestamp.
    speed_of_sound
        Speed of sound in m/s.
    iterations
        Number of iterations (2 is usually sufficient).

    Returns
    -------
    np.ndarray
        Emission times corresponding to each observation time.
    """
    # Initial guess: t_e = t - r(t)/c
    t_emission = timestamps - slant_range / speed_of_sound

    for _ in range(iterations):
        # Interpolate slant range at the current emission time estimate
        r_at_emission = np.interp(t_emission, timestamps, slant_range)
        # Refine emission time
        t_emission = timestamps - r_at_emission / speed_of_sound

    return t_emission


def predict_doppler_curve(
    receiver_lat: float,
    receiver_lon: float,
    receiver_alt_m: float,
    aircraft_lat: np.ndarray,
    aircraft_lon: np.ndarray,
    aircraft_alt_m: np.ndarray,
    timestamps: np.ndarray,
    f0: float,
    clock_offset: float = 0.0,
    speed_of_sound: float = SPEED_OF_SOUND,
    temperature_celsius: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict the Doppler frequency curve for a flight as heard from a
    ground receiver.

    This is the core forward model:
      f_obs(t) = f0 * c / (c + r'(t_e))

    Parameters
    ----------
    receiver_lat, receiver_lon, receiver_alt_m
        Receiver (ground) position.
    aircraft_lat, aircraft_lon, aircraft_alt_m
        Aircraft trajectory (arrays of same length as timestamps).
    timestamps
        Unix timestamps of the aircraft trajectory (1 s spacing).
    f0
        Emitted tone frequency in Hz (engine fundamental).
    clock_offset
        Offset in seconds: recording_time = UTC_time + clock_offset.
    speed_of_sound
        Speed of sound in m/s.
    temperature_celsius
        If provided, overrides speed_of_sound with temperature-corrected value.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (observation_times, predicted_freqs) — times are in recording time
        (UTC + clock_offset), freqs in Hz.
    """
    if temperature_celsius is not None:
        speed_of_sound = 331.3 + 0.606 * temperature_celsius

    # Compute slant range and range rate
    slant_range, range_rate = compute_range_rate(
        receiver_lat, receiver_lon, receiver_alt_m,
        aircraft_lat, aircraft_lon, aircraft_alt_m,
        timestamps,
    )

    # Acoustic propagation arrival time: sound emitted at t arrives at t + r/c
    t_arrival = timestamps + slant_range / speed_of_sound

    # Doppler formula: f_obs = f0 * c / (c + r'(t_e))
    # Sound emitted with range rate r'(t_e) is heard at t_arrival
    f_observed = f0 * speed_of_sound / (speed_of_sound + range_rate)

    # Convert observation times to recording time
    observation_times = t_arrival + clock_offset

    return observation_times, f_observed


def predict_audibility(
    slant_range: np.ndarray,
    aircraft_alt_m: np.ndarray,
    max_audible_range_m: float = 30000.0,
    min_altitude_m: float = 500.0,
) -> np.ndarray:
    """
    Predict crude audibility envelope: whether the aircraft tone would
    be detectable at a receiver position.

    Simple model: audible if slant range < max_audible_range
    and altitude > min_altitude (too low = too fast ground roll noise).

    Parameters
    ----------
    slant_range
        Slant range in meters.
    aircraft_alt_m
        Aircraft altitude in meters.
    max_audible_range_m
        Maximum distance for audibility.
    min_altitude_m
        Minimum altitude for in-flight tone emission.

    Returns
    -------
    np.ndarray
        Boolean array: True where the aircraft tone should be audible.
    """
    return (slant_range < max_audible_range_m) & (aircraft_alt_m > min_altitude_m)


def fit_f0_and_offset(
    observed_times: np.ndarray,
    observed_freqs: np.ndarray,
    receiver_lat: float,
    receiver_lon: float,
    receiver_alt_m: float,
    aircraft_lat: np.ndarray,
    aircraft_lon: np.ndarray,
    aircraft_alt_m: np.ndarray,
    flight_timestamps: np.ndarray,
    f0_range: tuple[float, float] = (50.0, 5000.0),
    offset_range: tuple[float, float] = (-300.0, 300.0),
    speed_of_sound: float = SPEED_OF_SOUND,
) -> tuple[float, float, float]:
    """
    Fit the emitted frequency f0 and clock offset to minimize the
    residual between observed and predicted Doppler curves.

    Uses scipy.optimize.minimize (bounded) on the RMS residual.

    Parameters
    ----------
    observed_times
        Observed ridge times (recording time).
    observed_freqs
        Observed ridge frequencies.
    receiver_lat, receiver_lon, receiver_alt_m
        Candidate receiver position.
    aircraft_lat, aircraft_lon, aircraft_alt_m, flight_timestamps
        Flight track data.
    f0_range
        Bounds for f0 search.
    offset_range
        Bounds for clock offset search (seconds).
    speed_of_sound
        Speed of sound in m/s.

    Returns
    -------
    tuple[float, float, float]
        (best_f0, best_offset, rms_residual_hz)
    """
    # Precompute slant range, range rate, and arrival times (independent of f0/offset)
    slant_range, range_rate = compute_range_rate(
        receiver_lat, receiver_lon, receiver_alt_m,
        aircraft_lat, aircraft_lon, aircraft_alt_m,
        flight_timestamps,
    )
    t_arrival = flight_timestamps + slant_range / speed_of_sound

    # Grid search offset with exact closed-form optimal f0, then polish with Brent
    n_grid = max(21, int(round((offset_range[1] - offset_range[0]) / 2.0)))
    offsets = np.linspace(offset_range[0], offset_range[1], n_grid)

    best_rms = float("inf")
    best_offset = 0.0
    best_f0 = float(np.median(observed_freqs))

    for off in offsets:
        obs_utc = observed_times - off
        rr = np.interp(obs_utc, t_arrival, range_rate)
        psi = speed_of_sound / (speed_of_sound + rr)
        denom = float(np.dot(psi, psi))
        if denom > 1e-12:
            f0_cand = float(np.dot(observed_freqs, psi) / denom)
            f0_cand = float(np.clip(f0_cand, f0_range[0], f0_range[1]))
        else:
            f0_cand = best_f0
        f_pred = f0_cand * psi
        rms = float(np.sqrt(np.mean((observed_freqs - f_pred) ** 2)))
        if rms < best_rms:
            best_rms = rms
            best_offset = float(off)
            best_f0 = f0_cand

    # Polish offset in neighborhood using minimize_scalar (Brent)
    bracket_min = max(offset_range[0], best_offset - 2.0)
    bracket_max = min(offset_range[1], best_offset + 2.0)
    if bracket_min < best_offset < bracket_max:
        from scipy.optimize import minimize_scalar

        def objective(off: float) -> float:
            obs_u = observed_times - off
            rr = np.interp(obs_u, t_arrival, range_rate)
            psi = speed_of_sound / (speed_of_sound + rr)
            denom = float(np.dot(psi, psi))
            f0_c = float(np.dot(observed_freqs, psi) / denom) if denom > 1e-12 else best_f0
            f0_c = float(np.clip(f0_c, f0_range[0], f0_range[1]))
            f_p = f0_c * psi
            return float(np.sqrt(np.mean((observed_freqs - f_p) ** 2)))

        try:
            res = minimize_scalar(objective, bracket=(bracket_min, best_offset, bracket_max), method="brent")
            if res.fun < best_rms:
                best_offset = float(res.x)
                best_rms = float(res.fun)
                obs_u = observed_times - best_offset
                rr = np.interp(obs_u, t_arrival, range_rate)
                psi = speed_of_sound / (speed_of_sound + rr)
                denom = float(np.dot(psi, psi))
                best_f0 = float(np.dot(observed_freqs, psi) / denom) if denom > 1e-12 else best_f0
                best_f0 = float(np.clip(best_f0, f0_range[0], f0_range[1]))
        except Exception:
            pass

    return float(best_f0), float(best_offset), float(best_rms)
