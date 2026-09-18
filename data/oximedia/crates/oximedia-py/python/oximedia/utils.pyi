"""
Type stubs for ``oximedia.utils`` — common media helper functions.

Source: ``crates/oximedia-py/src/utils_py.rs``
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

def duration_to_timecode(seconds: float, fps: float = 25.0) -> str:
    """Convert a duration in seconds to ``HH:MM:SS:FF`` SMPTE timecode.

    Raises ``ValueError`` if ``seconds`` < 0 or ``fps`` <= 0.
    """
    ...

def timecode_to_duration(timecode: str, fps: float = 25.0) -> float:
    """Parse ``HH:MM:SS:FF`` timecode and return the duration in seconds."""
    ...

def fps_to_rational(fps: float) -> Tuple[int, int]:
    """Convert a floating-point FPS to ``(numerator, denominator)``.

    Common rates (29.97, 23.976, 59.94, …) are recognised exactly.
    """
    ...

def format_duration(seconds: float, precision: int = 3) -> str:
    """Format a duration as a human-readable string ("``1h 2m 3.456s``")."""
    ...

def estimate_bitrate(file_size_bytes: int, duration_seconds: float) -> float:
    """Estimate the average bitrate in kbps."""
    ...

def calculate_frame_size(
    width: int,
    height: int,
    pixel_format: str = "yuv420p",
) -> int:
    """Return the size, in bytes, of an uncompressed frame."""
    ...

def media_info_summary(
    path: str,
    duration_seconds: float,
    video_codec: Optional[str],
    audio_codec: Optional[str],
    fps: float = 25.0,
) -> Dict[str, Any]:
    """Build a human-readable summary dict for a media file."""
    ...
