"""
Tests for the forward Doppler model.

Validates the physics engine with a synthetic scenario: generate a
Doppler tone for a known receiver location and a simulated flight
track, verify the predicted curve matches the analytical expectation.
"""

from __future__ import annotations

import numpy as np
import pytest

from skyhear.model import (
    SPEED_OF_SOUND,
    compute_range_rate,
    compute_slant_range,
    fit_f0_and_offset,
    lla_to_ecef,
    predict_audibility,
    predict_doppler_curve,
    retarded_time_correction,
)


class TestECEFConversion:
    """Test ECEF coordinate conversions."""

    def test_origin_equator_prime_meridian(self):
        """Point on the equator at prime meridian, sea level."""
        x, y, z = lla_to_ecef(0.0, 0.0, 0.0)
        # Should be roughly at Earth's equatorial radius
        assert abs(float(x) - 6378137.0) < 1.0  # WGS84 semi-major axis
        assert abs(float(y)) < 1.0
        assert abs(float(z)) < 1.0

    def test_north_pole(self):
        """North pole should have z ≈ semi-minor axis."""
        x, y, z = lla_to_ecef(90.0, 0.0, 0.0)
        assert abs(float(x)) < 1.0
        assert abs(float(y)) < 1.0
        assert abs(float(z) - 6356752.314) < 1.0  # WGS84 semi-minor axis

    def test_altitude(self):
        """Adding altitude should increase distance from center."""
        x1, y1, z1 = lla_to_ecef(45.0, 90.0, 0.0)
        x2, y2, z2 = lla_to_ecef(45.0, 90.0, 10000.0)
        r1 = np.sqrt(float(x1) ** 2 + float(y1) ** 2 + float(z1) ** 2)
        r2 = np.sqrt(float(x2) ** 2 + float(y2) ** 2 + float(z2) ** 2)
        assert r2 - r1 == pytest.approx(10000.0, abs=1.0)


class TestSlantRange:
    """Test slant range computation."""

    def test_directly_overhead(self):
        """Aircraft directly overhead — slant range should equal altitude."""
        r = compute_slant_range(
            40.0, -105.0, 0.0,
            np.array([40.0]),
            np.array([-105.0]),
            np.array([10000.0]),
        )
        assert r[0] == pytest.approx(10000.0, rel=0.01)

    def test_slant_range_increases_with_offset(self):
        """Moving aircraft sideways should increase slant range."""
        r1 = compute_slant_range(
            40.0, -105.0, 0.0,
            np.array([40.0]), np.array([-105.0]), np.array([10000.0]),
        )
        r2 = compute_slant_range(
            40.0, -105.0, 0.0,
            np.array([40.01]), np.array([-105.0]), np.array([10000.0]),
        )
        assert float(r2[0]) > float(r1[0])


class TestRangeRate:
    """Test range rate (dr/dt) computation."""

    def test_approaching_aircraft(self):
        """An approaching aircraft should have negative range rate."""
        # Simulate a flight passing overhead at constant altitude
        n = 601
        t = np.arange(n, dtype=float)  # 0 to 600 seconds

        receiver_lat, receiver_lon = 40.0, -105.0

        # Aircraft flying east, starting west of receiver
        lat = np.full(n, 40.0)
        # Move from west to east: about 0.1 degrees over 600s
        lon = np.linspace(-105.1, -104.9, n)
        alt = np.full(n, 10000.0)

        slant, rate = compute_range_rate(
            receiver_lat, receiver_lon, 0.0,
            lat, lon, alt, t,
        )

        # Range rate should be negative in first half (approaching)
        # and positive in second half (receding)
        assert rate[50] < 0  # approaching
        assert rate[550] > 0  # receding

        # Minimum slant range should be near the middle
        min_idx = np.argmin(slant)
        assert abs(min_idx - 300) < 50


class TestDopplerCurve:
    """Test the forward Doppler curve prediction."""

    def test_symmetric_flyby(self):
        """A symmetric flyby should produce a symmetric Doppler curve."""
        n = 601
        t = np.arange(n, dtype=float)

        # Aircraft flying straight east over the receiver
        lat = np.full(n, 40.0)
        lon = np.linspace(-105.05, -104.95, n)
        alt = np.full(n, 10000.0)

        obs_times, obs_freqs = predict_doppler_curve(
            40.0, -105.0, 0.0,
            lat, lon, alt, t,
            f0=1000.0,
        )

        # Frequency should start above f0 (approaching) and end below (receding)
        assert obs_freqs[10] > 1000.0
        assert obs_freqs[-10] < 1000.0

        # Curve should be roughly symmetric around the midpoint
        mid = len(obs_freqs) // 2
        freq_range_first_half = obs_freqs[10] - obs_freqs[mid]
        freq_range_second_half = obs_freqs[mid] - obs_freqs[-10]
        assert freq_range_first_half == pytest.approx(freq_range_second_half, rel=0.1)

    def test_doppler_shift_magnitude(self):
        """Doppler shift should be consistent with aircraft speed."""
        n = 601
        t = np.arange(n, dtype=float)

        lat = np.full(n, 40.0)
        lon = np.linspace(-105.05, -104.95, n)
        alt = np.full(n, 10000.0)

        f0 = 1000.0
        obs_times, obs_freqs = predict_doppler_curve(
            40.0, -105.0, 0.0,
            lat, lon, alt, t,
            f0=f0,
        )

        # Max Doppler shift should be f0 * v_radial / c
        # For a high flyby, the max radial velocity is < aircraft speed
        max_shift = max(abs(obs_freqs.max() - f0), abs(obs_freqs.min() - f0))
        # Should be reasonable (< 100 Hz for a 250 m/s aircraft at 1000 Hz)
        assert 0 < max_shift < 100

    def test_clock_offset(self):
        """Clock offset should shift times but not frequencies."""
        n = 301
        t = np.arange(n, dtype=float)
        lat = np.full(n, 40.0)
        lon = np.linspace(-105.02, -104.98, n)
        alt = np.full(n, 10000.0)

        t1, f1 = predict_doppler_curve(40.0, -105.0, 0.0, lat, lon, alt, t, f0=1000.0, clock_offset=0.0)
        t2, f2 = predict_doppler_curve(40.0, -105.0, 0.0, lat, lon, alt, t, f0=1000.0, clock_offset=10.0)

        # Frequencies should be identical
        np.testing.assert_array_almost_equal(f1, f2, decimal=6)

        # Times should differ by the offset
        np.testing.assert_array_almost_equal(t2 - t1, 10.0, decimal=6)


class TestRetardedTime:
    """Test retarded time correction."""

    def test_correction_reduces_with_proximity(self):
        """Retarded time correction should be smaller when aircraft is closer."""
        t = np.arange(0, 100, 1.0)
        range_far = np.full_like(t, 30000.0)  # 30 km
        range_close = np.full_like(t, 3000.0)  # 3 km

        te_far = retarded_time_correction(t, range_far)
        te_close = retarded_time_correction(t, range_close)

        delay_far = np.mean(t - te_far)
        delay_close = np.mean(t - te_close)

        assert delay_far > delay_close
        assert delay_far == pytest.approx(30000.0 / SPEED_OF_SOUND, rel=0.1)
        assert delay_close == pytest.approx(3000.0 / SPEED_OF_SOUND, rel=0.1)


class TestAudibility:
    """Test audibility prediction."""

    def test_close_audible(self):
        """Close, high aircraft should be audible."""
        result = predict_audibility(
            np.array([5000.0]),
            np.array([5000.0]),
        )
        assert result[0] is True or result[0] == True  # noqa

    def test_far_inaudible(self):
        """Very far aircraft should be inaudible."""
        result = predict_audibility(
            np.array([100000.0]),
            np.array([10000.0]),
        )
        assert result[0] is False or result[0] == False  # noqa


class TestFitF0AndOffset:
    """Test fitting f0 and clock offset to a synthetic curve."""

    def test_recover_known_parameters(self):
        """Generate a synthetic curve and recover the parameters."""
        n = 301
        t = np.arange(n, dtype=float)

        lat = np.full(n, 40.0)
        lon = np.linspace(-105.03, -104.97, n)
        alt = np.full(n, 10000.0)

        true_f0 = 850.0
        true_offset = 5.0

        # Generate "observed" curve with known parameters
        obs_times, obs_freqs = predict_doppler_curve(
            40.0, -105.0, 0.0,
            lat, lon, alt, t,
            f0=true_f0,
            clock_offset=true_offset,
        )

        # Subsample to simulate real observations
        idx = np.arange(10, len(obs_times) - 10, 5)
        obs_t_sub = obs_times[idx]
        obs_f_sub = obs_freqs[idx]

        # Fit
        fitted_f0, fitted_offset, rms = fit_f0_and_offset(
            obs_t_sub, obs_f_sub,
            40.0, -105.0, 0.0,
            lat, lon, alt, t,
        )

        # Should recover parameters within tolerance
        assert fitted_f0 == pytest.approx(true_f0, rel=0.01)
        assert fitted_offset == pytest.approx(true_offset, abs=1.0)
        assert rms < 1.0  # < 1 Hz residual
