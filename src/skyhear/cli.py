"""
skyhear.cli — Command-line interface.

Usage:
    geo-locate --input clip.mp4 --date 2026-09-08 --region bbox:24,-125,50,-66 --top 10 --out results/
    geo-locate --input clip.wav --date 2026-09-08 --region center:40.0,-105.0,radius_km:300
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

logger = logging.getLogger("skyhear")


def _load_env_file(path: Path | None = None) -> None:
    """Load key-value pairs from a .env file into os.environ if not already set."""
    if path is None:
        path = Path.cwd() / ".env"
    if not path.is_file():
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


_load_env_file()


KNOWN_METROS: dict[str, tuple[float, float, float]] = {
    "vancouver": (49.25, -123.12, 45.0),
    "seattle": (47.60, -122.33, 45.0),
    "toronto": (43.65, -79.38, 45.0),
    "calgary": (51.04, -114.07, 45.0),
    "montreal": (45.50, -73.57, 45.0),
    "edmonton": (53.54, -113.49, 45.0),
    "victoria": (48.42, -123.36, 35.0),
    "portland": (45.52, -122.68, 45.0),
    "los angeles": (34.05, -118.25, 60.0),
    "la": (34.05, -118.25, 60.0),
    "san francisco": (37.77, -122.42, 50.0),
    "sf": (37.77, -122.42, 50.0),
    "bay area": (37.77, -122.42, 50.0),
    "new york": (40.71, -74.01, 50.0),
    "nyc": (40.71, -74.01, 50.0),
    "chicago": (41.88, -87.63, 50.0),
    "london": (51.51, -0.13, 50.0),
}


def _detect_local_metro() -> tuple[float, float, float, str]:
    """Auto-detect user's approximate metro area via IP or fallback."""
    import urllib.request
    import json
    try:
        req = urllib.request.Request("http://ip-api.com/json/", headers={"User-Agent": "skyhear/0.1"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            d = json.loads(resp.read().decode())
            lat = float(d.get("lat", 49.25))
            lon = float(d.get("lon", -123.12))
            city = d.get("city", "Vancouver")
            return lat, lon, 45.0, city
    except Exception:
        return 49.25, -123.12, 45.0, "Vancouver"


def _geocode_name(name: str) -> tuple[float, float, float] | None:
    """Geocode a city or place name into (lat, lon, radius_km)."""
    norm = name.lower().strip()
    if norm in KNOWN_METROS:
        return KNOWN_METROS[norm]
    import urllib.request
    import urllib.parse
    import json
    try:
        q = urllib.parse.quote(name)
        url = f"https://nominatim.openstreetmap.org/search?q={q}&format=json&limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "skyhear-acoustic/0.1"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            res = json.loads(resp.read().decode())
            if res and len(res) > 0:
                lat = float(res[0]["lat"])
                lon = float(res[0]["lon"])
                return lat, lon, 45.0
    except Exception:
        pass
    return None


def _parse_region(region_str: str, is_fresh: bool = False):
    """
    Parse a region string into a BoundingBox.

    Formats:
        - Plain English city name: "Vancouver", "Seattle", "Toronto"
        - Continent / Auto: "North America", "auto", "local"
        - Explicit bbox: "bbox:min_lat,min_lon,max_lat,max_lon"
        - Explicit center: "center:lat,lon,radius_km:300"
    """
    from .adsb import BoundingBox

    norm = region_str.strip().lower()

    if norm.startswith("bbox:"):
        parts = norm[5:].split(",")
        if len(parts) != 4:
            raise click.BadParameter(
                "bbox format: bbox:min_lat,min_lon,max_lat,max_lon"
            )
        min_lat, min_lon, max_lat, max_lon = map(float, parts)
        return BoundingBox(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)

    elif norm.startswith("center:"):
        rest = norm[7:]
        parts = rest.split(",")
        if len(parts) < 2:
            raise click.BadParameter(
                "center format: center:lat,lon,radius_km:300"
            )
        lat = float(parts[0])
        lon_and_radius = parts[1:]
        radius_km = 100.0
        lon_str = lon_and_radius[0]
        for part in lon_and_radius:
            if part.startswith("radius_km:"):
                radius_km = float(part.split(":")[1])
            else:
                lon_str = part
        lon = float(lon_str)
        return BoundingBox.from_center_radius(lat, lon, radius_km)

    # Broad regions / auto / local
    if norm in ("north america", "na", "auto", "local", "here", "default"):
        lat, lon, r_km, name = _detect_local_metro()
        click.echo(f"🌐 Resolved region '{region_str}' to local airspace: {name} ({lat:.4f}, {lon:.4f}, radius {r_km:.0f}km)")
        return BoundingBox.from_center_radius(lat, lon, r_km)

    # Named city or area lookup
    coords = _geocode_name(region_str)
    if coords is not None:
        lat, lon, r_km = coords
        click.echo(f"📍 Resolved region '{region_str}' to ({lat:.4f}, {lon:.4f}, radius {r_km:.0f}km)")
        return BoundingBox.from_center_radius(lat, lon, r_km)

    raise click.BadParameter(
        f"Could not resolve region '{region_str}'. "
        "Use a city name ('Vancouver', 'Seattle'), 'North America', bbox:min_lat,min_lon,max_lat,max_lon, "
        "or center:lat,lon,radius_km:N"
    )


@click.command()
@click.argument("audio_file", required=False, type=click.Path(exists=True))
@click.option(
    "--input", "-i",
    "input_opt",
    default=None,
    type=click.Path(exists=True),
    help="Input media file (mp4, mov, wav, ogg, mp3, etc.)",
)
@click.option(
    "--date", "-d",
    default=None,
    help="Recording date in YYYY-MM-DD format (auto-inferred with --fresh)",
)
@click.option(
    "--fresh",
    is_flag=True,
    default=False,
    help="Use live/recent ADS-B data for clips recorded in the past hour",
)
@click.option(
    "--region", "-r",
    default=None,
    help="Search region: city name (e.g. 'Vancouver'), 'North America', bbox:..., or center:lat,lon,radius_km:N",
)
@click.option(
    "--top", "-n",
    default=3,
    show_default=True,
    help="Number of top candidates to return",
)
@click.option(
    "--out", "-o",
    "output_dir",
    default="results",
    show_default=True,
    type=click.Path(),
    help="Output directory for results",
)
@click.option(
    "--start-time",
    default=None,
    help="Approximate recording start time in UTC (HH:MM:SS). Helps narrow search.",
)
@click.option(
    "--auto-detect / --manual",
    default=True,
    show_default=True,
    help="Auto-detect ridges (default) or use manual seed picking",
)
@click.option(
    "--min-duration",
    default=8.0,
    show_default=True,
    help="Minimum ridge duration in seconds for Doppler pass detection",
)
@click.option(
    "--adsb-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to local directory with readsb / adsb.lol / tar1090 JSON trace files",
)
@click.option(
    "--opensky-user",
    default=None,
    envvar="OPENSKY_USER",
    help="OpenSky Network username (or set OPENSKY_USER env var)",
)
@click.option(
    "--opensky-pass",
    default=None,
    envvar="OPENSKY_PASS",
    help="OpenSky Network password (or set OPENSKY_PASS env var)",
)
@click.option(
    "-v", "--verbose",
    count=True,
    help="Increase verbosity (-v info, -vv debug)",
)
def main(
    audio_file: str | None,
    input_opt: str | None,
    date: str | None,
    fresh: bool,
    region: str | None,
    top: int,
    output_dir: str,
    start_time: str | None,
    auto_detect: bool,
    min_duration: float,
    adsb_dir: str | None,
    opensky_user: str | None,
    opensky_pass: str | None,
    verbose: int,
):
    """
    Geo-locate a recording by matching aircraft Doppler signatures
    against ADS-B flight data.
    """
    # Configure logging
    log_level = logging.WARNING
    if verbose >= 2:
        log_level = logging.DEBUG
    elif verbose >= 1:
        log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from datetime import datetime, timezone
    from .adsb import BoundingBox, fetch_opensky_tracks, build_flight_tracks
    from .detect import (
        auto_detect_ridges,
        compute_spectrogram,
        plot_spectrogram,
        spectral_whitening,
    )
    import matplotlib.pyplot as plt
    from .ingest import extract_audio, load_audio
    from .report import generate_results_json, print_summary
    from .search import SearchConfig, search_location

    input_file = audio_file or input_opt
    if not input_file:
        raise click.UsageError(
            "Missing input media file. Provide it as an argument or via --input.\n"
            "Example: geo-locate clip.ogg --region Vancouver --fresh"
        )

    if fresh:
        if not date:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            click.echo(f"⚡ Fresh mode: auto-selected today's date ({date})")
    elif not date:
        raise click.UsageError(
            "Missing required option '--date' / '-d' (e.g. --date 2026-06-07).\n"
            "If the recording was made in the past hour, simply add '--fresh'."
        )

    if not region:
        if fresh:
            region = "auto"
        else:
            raise click.UsageError(
                "Missing required option '--region' / '-r' (e.g. --region 'Vancouver')."
            )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bbox = _parse_region(region, is_fresh=fresh)

    # === Step 1: Ingest ===
    ingest_result = extract_audio(input_file, output_dir=output_path / "audio")
    click.echo(f"🎧 Media: {Path(input_file).name} ({ingest_result.duration_seconds:.1f}s)")

    # === Step 2: Detect aircraft tones ===
    audio, sr = load_audio(ingest_result.wav_path)
    Sxx, times, freqs = compute_spectrogram(audio, sr)
    Sxx_w = spectral_whitening(Sxx)

    # Save spectrogram
    spec_path = output_path / "spectrogram.png"

    if auto_detect:
        passes = auto_detect_ridges(Sxx_w, times, freqs, min_ridge_duration_s=min_duration)
    else:
        fig = plot_spectrogram(Sxx_w, times, freqs, title="Whitened Spectrogram")
        fig.savefig(str(spec_path), dpi=150)
        click.echo(f"   Spectrogram saved to {spec_path}")
        sys.exit(0)

    if not passes:
        click.echo("⚠️  No aircraft tones detected.")
        sys.exit(1)

    click.echo(f"🔍 Acoustic Analysis: Detected {len(passes)} Doppler tone(s)")
    fig = plot_spectrogram(Sxx_w, times, freqs, passes=passes, title="Detected Passes")
    fig.savefig(str(spec_path), dpi=150)
    plt.close(fig)

    # === Step 3: Determine recording time ===
    recording_utc_start = None
    if start_time:
        try:
            dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            recording_utc_start = dt.timestamp()
        except ValueError:
            dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            recording_utc_start = dt.timestamp()
        click.echo(f"⏱️  Time: {start_time} UTC")
    elif fresh:
        file_mtime = os.path.getmtime(input_file)
        now_ts = datetime.now(timezone.utc).timestamp()
        if now_ts - 7200 <= file_mtime <= now_ts + 60:
            recording_utc_start = file_mtime - ingest_result.duration_seconds
            rec_dt = datetime.fromtimestamp(recording_utc_start, tz=timezone.utc)
            click.echo(f"⏱️  Time: ~{rec_dt.strftime('%H:%M:%S UTC')} (from file)")
        else:
            recording_utc_start = now_ts - ingest_result.duration_seconds
            rec_dt = datetime.fromtimestamp(recording_utc_start, tz=timezone.utc)
            click.echo(f"⏱️  Time: ~{rec_dt.strftime('%H:%M:%S UTC')} (live)")
    else:
        try:
            import av
            container = av.open(str(input_file))
            ct = container.metadata.get("creation_time")
            if ct:
                dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                recording_utc_start = dt.timestamp()
                # Differentiate video vs audio container recording metadata
                is_video = any(s.type == "video" for s in container.streams)
                source_label = "video recording metadata" if is_video else "audio recording metadata"
                click.echo(f"⏱️  Time: {dt.strftime('%H:%M:%S UTC')} ({source_label})")
        except Exception:
            pass

    # === Step 4: Fetch ADS-B data ===
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    is_today = (date == today_utc)

    if adsb_dir is not None:
        from .adsb import load_readsb_traces_dir
        tracks = load_readsb_traces_dir(adsb_dir, bbox=bbox)
    elif fresh or is_today:
        from .adsb import fetch_live_tracks
        tracks = fetch_live_tracks(
            bbox=bbox,
            target_time=recording_utc_start,
            username=opensky_user,
            password=opensky_pass,
        )
    else:
        # 1. Try local cache or GitHub daily archive (adsblol/globe_history)
        from .adsb import download_github_traces
        click.echo(f"✈️  Retrieving historical flight paths for {date}...")
        tracks = download_github_traces(date, bbox=bbox, target_time=recording_utc_start)

        # 2. Fall back to OpenSky Network if GitHub archive not yet published
        if not tracks:
            try:
                df = fetch_opensky_tracks(
                    date, bbox,
                    username=opensky_user,
                    password=opensky_pass,
                )
                if not df.empty:
                    tracks = build_flight_tracks(df)
            except Exception:
                pass

        # 3. Final fallback to live radar feeds
        if not tracks:
            from .adsb import fetch_live_tracks
            tracks = fetch_live_tracks(
                bbox=bbox,
                target_time=recording_utc_start,
                username=opensky_user,
                password=opensky_pass,
            )

    if not tracks:
        click.echo("⚠️  No matching flight tracks found.")
        sys.exit(1)

    click.echo(f"✈️  Radar: Loaded {len(tracks)} flight tracks")

    # === Step 5: Search ===
    click.echo("🎯 Geolocating...")
    config = SearchConfig()
    candidates = search_location(
        passes, tracks, bbox, config, recording_utc_start=recording_utc_start
    )

    if not candidates:
        click.echo("⚠️  No candidate locations found.")
        sys.exit(1)

    # === Step 6: Report ===
    click.echo(f"\n📍 Top {min(top, len(candidates))} candidates:")
    summary = print_summary(candidates, top_n=top)

    # Save results
    results_path = generate_results_json(candidates[:top], output_path / "results.json")

    # Save summary
    with open(output_path / "summary.txt", "w") as f:
        f.write(summary)

    # Save overlay plot for top candidate
    try:
        from .report import plot_overlay
        tracks_dict = {
            t.icao24: {
                "timestamps": t.timestamps,
                "lat": t.lat,
                "lon": t.lon,
                "alt_m": t.alt_m,
            }
            for t in tracks
        }
        overlay_fig = plot_overlay(
            passes=passes,
            candidate=candidates[0],
            flight_tracks=tracks_dict,
            output_path=output_path / "overlay.png",
        )
        plt.close(overlay_fig)
        click.echo(f"   Doppler overlay saved to {output_path / 'overlay.png'}")
    except Exception as e:
        logger.warning("Failed to generate overlay plot: %s", e)

    click.echo(f"\n✅ Results saved to {output_path}/")


if __name__ == "__main__":
    main()
