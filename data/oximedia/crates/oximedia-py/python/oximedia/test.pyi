"""
Type stubs for ``oximedia.test`` — synthetic test-media generators.

Source: ``crates/oximedia-py/src/test_media.rs``
"""

from __future__ import annotations

from typing import List

from . import AudioFrame, VideoFrame

def synthetic_video_frame(
    width: int = 1920,
    height: int = 1080,
    pts: int = 0,
    pixel_format: str = "yuv420p",
) -> VideoFrame:
    """Generate a single synthetic video frame with a deterministic pattern."""
    ...

def synthetic_audio_frame(
    sample_rate: int = 48000,
    channels: int = 2,
    duration_ms: float = 20.0,
    frequency_hz: float = 440.0,
    pts: int = 0,
) -> AudioFrame:
    """Generate a synthetic audio frame containing a sine tone."""
    ...

def generate_video_sequence(
    count: int,
    width: int = 1920,
    height: int = 1080,
    pixel_format: str = "yuv420p",
) -> List[VideoFrame]:
    """Generate a sequence of ``count`` synthetic video frames."""
    ...

def generate_audio_sequence(
    count: int,
    sample_rate: int = 48000,
    channels: int = 2,
    duration_ms: float = 20.0,
) -> List[AudioFrame]:
    """Generate a sequence of ``count`` synthetic audio frames."""
    ...

def solid_color_frame(
    width: int,
    height: int,
    y: int = 128,
    cb: int = 128,
    cr: int = 128,
    pts: int = 0,
) -> VideoFrame:
    """Generate a solid-colour YUV420p frame from individual component values."""
    ...

def checkerboard_frame(
    width: int,
    height: int,
    square_size: int = 64,
    pts: int = 0,
) -> VideoFrame:
    """Generate a black-and-white checkerboard YUV420p frame."""
    ...

def silence_frame(
    sample_rate: int = 48000,
    channels: int = 2,
    duration_ms: float = 20.0,
    pts: int = 0,
) -> AudioFrame:
    """Generate an all-zero audio frame."""
    ...
