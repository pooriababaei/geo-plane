"""
Tests for the detection module.
"""

from __future__ import annotations

import numpy as np
import pytest

from skyhear.detect import (
    DetectedPass,
    compute_spectrogram,
    spectral_whitening,
    find_ridges_manual_seeds,
    auto_detect_ridges,
)


def _make_sine_sweep(
    sr: int = 44100,
    duration: float = 10.0,
    f_start: float = 500.0,
    f_end: float = 400.0,
) -> np.ndarray:
    """Create a synthetic linear frequency sweep (chirp) to simulate a Doppler tone."""
    t = np.arange(0, duration, 1.0 / sr)
    # Linear chirp
    freq = np.linspace(f_start, f_end, len(t))
    phase = 2 * np.pi * np.cumsum(freq) / sr
    signal = 0.1 * np.sin(phase)
    # Add noise
    noise = 0.01 * np.random.randn(len(t))
    return (signal + noise).astype(np.float32)


class TestSpectrogram:
    """Test spectrogram computation."""

    def test_basic_spectrogram(self):
        """Should produce a valid spectrogram from a test signal."""
        sr = 44100
        audio = _make_sine_sweep(sr=sr, duration=5.0)
        Sxx, times, freqs = compute_spectrogram(audio, sr, window_seconds=1.0)

        assert Sxx.shape[0] == len(freqs)
        assert Sxx.shape[1] == len(times)
        assert freqs[0] >= 50.0
        assert freqs[-1] <= 4000.0
        assert Sxx.min() >= 0  # power is non-negative

    def test_frequency_range(self):
        """Spectrogram should respect the frequency range parameter."""
        sr = 44100
        audio = _make_sine_sweep(sr=sr, duration=3.0)
        Sxx, times, freqs = compute_spectrogram(
            audio, sr, freq_range=(200.0, 800.0)
        )
        assert freqs[0] >= 200.0
        assert freqs[-1] <= 800.0


class TestWhitening:
    """Test spectral whitening."""

    def test_whitening_shape(self):
        """Whitened spectrogram should have the same shape."""
        Sxx = np.random.rand(100, 200) + 0.01
        whitened = spectral_whitening(Sxx)
        assert whitened.shape == Sxx.shape

    def test_whitening_enhances_tones(self):
        """Whitening should enhance a tone shorter than the median window."""
        # The median filter computes background over a window of length W.
        # If a tone is shorter than W/2, the median filter ignores it
        # (reports the true background), so Sxx / background >> 1.
        n_freq, n_time = 200, 500
        np.random.seed(42)
        Sxx = np.random.rand(n_freq, n_time) * 0.01  # noise floor (median ≈ 0.005)

        # Add a short burst tone (10 frames) — much shorter than default filter (51)
        tone_start = 245
        tone_end = 255
        Sxx[100, tone_start:tone_end] += 0.05

        whitened = spectral_whitening(Sxx, median_filter_size=51)

        # Tone should be clearly elevated above background
        tone_level = whitened[100, 250]
        # Compare to the noise floor at that same time frame
        other_freqs = np.delete(whitened[:, 250], 100)
        noise_level = np.median(other_freqs)
        assert tone_level > noise_level * 1.5


class TestRidgeTracking:
    """Test ridge tracking from seed points."""

    def test_manual_seed_tracking(self):
        """Should track a synthetic ridge from a seed point."""
        sr = 44100
        audio = _make_sine_sweep(sr=sr, duration=10.0, f_start=500.0, f_end=400.0)
        Sxx, times, freqs = compute_spectrogram(audio, sr, window_seconds=1.0)
        whitened = spectral_whitening(Sxx)

        # Seed at the middle of the sweep (~450 Hz)
        seed_t = 5.0
        seed_f = 450.0

        passes = find_ridges_manual_seeds(
            whitened, times, freqs,
            seed_points=[(seed_t, seed_f)],
            track_window_hz=30.0,
            min_ridge_length=5,
        )

        # Should detect at least one pass
        assert len(passes) >= 1
        assert len(passes[0].ridge_points) >= 5


class TestDetectedPass:
    """Test DetectedPass dataclass."""

    def test_times_and_freqs(self):
        """Should correctly extract time and frequency arrays."""
        points = [(1.0, 500.0), (2.0, 495.0), (3.0, 490.0), (4.0, 480.0)]
        p = DetectedPass(
            ridge_points=points,
            t_closest_approach=3.0,
            duration=3.0,
        )
        np.testing.assert_array_equal(p.times, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(p.freqs, [500.0, 495.0, 490.0, 480.0])
