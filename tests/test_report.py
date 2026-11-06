"""
Tests for the report module.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from skyhear.detect import DetectedPass
from skyhear.report import (
    generate_results_json,
    plot_overlay,
    print_summary,
)
from skyhear.search import CandidateLocation, FlightMatch


@pytest.fixture
def sample_candidates() -> list[CandidateLocation]:
    """Sample candidates for testing reports."""
    return [
        CandidateLocation(
            lat=39.8561,
            lon=-104.6737,
            total_score=1.23,
            flight_matches=[
                FlightMatch(
                    icao24="a00001",
                    callsign="UAL123",
                    fitted_f0=1200.0,
                    fitted_offset=2.5,
                    rms_residual_hz=1.23,
                )
            ],
            n_unexplained_passes=0,
            negative_evidence_penalty=0.0,
        ),
        CandidateLocation(
            lat=40.0,
            lon=-105.0,
            total_score=15.67,
            flight_matches=[],
            n_unexplained_passes=1,
            negative_evidence_penalty=5.0,
        ),
    ]


class TestGenerateResultsJson:
    """Test JSON result output."""

    def test_json_structure(self, tmp_path, sample_candidates):
        out_file = tmp_path / "results.json"
        generate_results_json(sample_candidates, out_file)

        assert out_file.exists()
        with open(out_file) as f:
            data = json.load(f)

        assert "candidates" in data
        assert len(data["candidates"]) == 2

        top = data["candidates"][0]
        assert top["rank"] == 1
        assert top["lat"] == pytest.approx(39.8561)
        assert top["lon"] == pytest.approx(-104.6737)
        assert top["total_score"] == pytest.approx(1.23)
        assert "google_maps_link" in top
        assert "39.856100,-104.673700" in top["google_maps_link"]

        # Check flight match
        assert len(top["flight_matches"]) == 1
        fm = top["flight_matches"][0]
        assert fm["icao24"] == "a00001"
        assert fm["callsign"] == "UAL123"
        assert fm["fitted_f0_hz"] == 1200.0


class TestPrintSummary:
    """Test text summary generation."""

    def test_summary_content(self, sample_candidates):
        summary = print_summary(sample_candidates, top_n=5)

        assert "AIRCRAFT-NOISE GEOLOCATION RESULTS" in summary
        assert "#1" in summary
        assert "39.856100" in summary
        assert "UAL123" in summary
        assert "Confidence: " in summary
        assert "#2" in summary
        assert "1 unexplained pass(es)" in summary


class TestPlotOverlay:
    """Test Doppler overlay plot."""

    def test_plot_generation(self, tmp_path, sample_candidates):
        # Create a mock pass
        detected_pass = DetectedPass(
            ridge_points=[(0.0, 1205.0), (10.0, 1200.0), (20.0, 1195.0)],
            t_closest_approach=10.0,
            duration=20.0,
            label="pass_0",
        )

        # Mock flight track data
        flight_tracks = {
            "a00001": {
                "timestamps": np.linspace(-10, 30, 41),
                "lat": np.linspace(39.8, 39.9, 41),
                "lon": np.linspace(-104.8, -104.5, 41),
                "alt_m": np.full(41, 10000.0),
            }
        }

        out_img = tmp_path / "overlay.png"
        fig = plot_overlay(
            passes=[detected_pass],
            candidate=sample_candidates[0],
            flight_tracks=flight_tracks,
            output_path=out_img,
        )

        assert out_img.exists()
        assert out_img.stat().st_size > 0
