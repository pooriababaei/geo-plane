"""
skyhear.ingest — Audio extraction and preprocessing.

Extracts audio from media files (video/audio) into mono WAV format
using ffmpeg, preserving the original sample rate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of ingesting a media file."""

    wav_path: Path
    sample_rate: int
    duration_seconds: float
    num_samples: int
    source_file: str
    source_hash: str  # SHA-256 of the source file

    def to_metadata_dict(self) -> dict:
        d = asdict(self)
        d["wav_path"] = str(d["wav_path"])
        return d


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _check_ffmpeg() -> str | None:
    """Return the path to ffmpeg, or None if not found."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.debug("ffmpeg found: %s", result.stdout.split("\n")[0])
        return "ffmpeg"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _extract_audio_av(input_path: Path, wav_path: Path) -> tuple[int, float, int]:
    """Extract audio using PyAV (bundled libav) when ffmpeg CLI is not installed."""
    import av
    import numpy as np

    container = av.open(str(input_path))
    audio_stream = next((s for s in container.streams if s.type == "audio"), None)
    if not audio_stream:
        raise ValueError(f"No audio stream found in {input_path}")

    sr = audio_stream.codec_context.sample_rate or 44100
    resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)

    chunks = []
    for frame in container.decode(audio_stream):
        for resampled_frame in resampler.resample(frame):
            chunks.append(resampled_frame.to_ndarray()[0])

    for resampled_frame in resampler.resample(None):
        chunks.append(resampled_frame.to_ndarray()[0])

    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    sf.write(str(wav_path), audio, sr, subtype="FLOAT")
    duration = len(audio) / sr
    return sr, duration, len(audio)


def extract_audio(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    force: bool = False,
) -> IngestResult:
    """
    Extract audio from a media file to mono WAV.

    Parameters
    ----------
    input_path
        Path to the input media file (.mp4, .mov, .wav, .mp3, etc.).
    output_dir
        Directory to write the output WAV and metadata JSON.
        Defaults to a directory next to the input file.
    force
        If True, overwrite existing output files.

    Returns
    -------
    IngestResult
        Contains paths and metadata about the extracted audio.
    """
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ffmpeg = _check_ffmpeg()

    # Determine output paths
    if output_dir is None:
        output_dir = input_path.parent / "ingest_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_path = output_dir / f"{input_path.stem}.wav"
    meta_path = output_dir / f"{input_path.stem}_meta.json"

    if wav_path.exists() and not force:
        logger.info("Output WAV already exists, loading metadata: %s", wav_path)
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            return IngestResult(
                wav_path=wav_path,
                sample_rate=meta["sample_rate"],
                duration_seconds=meta["duration_seconds"],
                num_samples=meta["num_samples"],
                source_file=meta["source_file"],
                source_hash=meta["source_hash"],
            )

    # Compute source hash
    logger.info("Computing hash of source file...")
    source_hash = _sha256(input_path)

    if ffmpeg is not None:
        # Probe original sample rate using ffprobe
        sample_rate = _probe_sample_rate(input_path)
        logger.info("Detected sample rate: %d Hz", sample_rate)

        # Extract audio with ffmpeg: mono WAV, preserve sample rate
        logger.info("Extracting audio with ffmpeg: %s → %s", input_path.name, wav_path.name)
        cmd = [
            ffmpeg,
            "-y" if force else "-n",
            "-i",
            str(input_path),
            "-vn",              # no video
            "-ac",
            "1",                # mono
            "-ar",
            str(sample_rate),   # preserve sample rate
            "-c:a",
            "pcm_f32le",        # 32-bit float WAV
            str(wav_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
        info = sf.info(str(wav_path))
        sample_rate = info.samplerate
        duration_s = info.duration
        num_frames = info.frames
    else:
        logger.info("Extracting audio with PyAV: %s → %s", input_path.name, wav_path.name)
        sample_rate, duration_s, num_frames = _extract_audio_av(input_path, wav_path)

    ingest_result = IngestResult(
        wav_path=wav_path,
        sample_rate=sample_rate,
        duration_seconds=duration_s,
        num_samples=num_frames,
        source_file=str(input_path),
        source_hash=source_hash,
    )

    # Save metadata JSON
    with open(meta_path, "w") as f:
        json.dump(ingest_result.to_metadata_dict(), f, indent=2)
    logger.info("Metadata saved to %s", meta_path)

    return ingest_result


def _probe_sample_rate(path: Path) -> int:
    """Use ffprobe to detect the audio sample rate of a file."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        logger.warning("Could not probe sample rate, defaulting to 44100 Hz")
        return 44100


def load_audio(wav_path: str | Path) -> tuple["np.ndarray", int]:
    """
    Load a WAV file into a numpy array.

    Returns
    -------
    tuple[np.ndarray, int]
        (samples, sample_rate) — samples is a 1D float32 array.
    """
    import numpy as np

    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        # Force mono by averaging channels
        data = data.mean(axis=1)
    return data, sr
