"""
Tests for the ingest module.
"""

from __future__ import annotations

import numpy as np
import pytest

from skyhear.ingest import _sha256, load_audio


class TestSHA256:
    """Test file hashing."""

    def test_hash_consistency(self, tmp_path):
        """Same content should produce the same hash."""
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        h1 = _sha256(f)
        h2 = _sha256(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex length


class TestLoadAudio:
    """Test audio loading."""

    def test_load_wav(self, tmp_path):
        """Should load a WAV file into a numpy array."""
        import soundfile as sf

        # Create a test WAV with values in [-1, 1] to avoid clipping
        wav_path = tmp_path / "test.wav"
        sr = 22050
        audio = (0.5 * np.random.randn(sr * 2)).astype(np.float32)  # 2 seconds
        audio = np.clip(audio, -0.99, 0.99)  # stay within valid range
        sf.write(str(wav_path), audio, sr)

        loaded, loaded_sr = load_audio(wav_path)
        assert loaded_sr == sr
        assert loaded.ndim == 1
        assert len(loaded) == len(audio)
        np.testing.assert_array_almost_equal(loaded, audio, decimal=4)
