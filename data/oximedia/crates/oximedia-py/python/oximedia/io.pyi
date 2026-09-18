"""
Type stubs for ``oximedia.io`` — file open / probe / transcode helpers.

Source: ``crates/oximedia-py/src/oximedia_io_py.rs``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import VideoFrame

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class MediaFileInfo:
    """High-level summary of a probed media file."""

    path: str
    duration_seconds: float
    size_bytes: int
    video_stream_count: int
    audio_stream_count: int
    container: str
    video_codec: Optional[str]
    audio_codec: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Return the info object as a Python dictionary."""
        ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class TranscodeResult:
    """Result returned by :func:`transcode`."""

    success: bool
    frames_written: int
    duration_ms: float
    output_path: str
    errors: List[str]

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Streaming reader
# ---------------------------------------------------------------------------

class MediaReader:
    """Streaming iterator that yields :class:`VideoFrame` objects.

    Supports the iterator and context-manager protocols::

        with oximedia.io.open_video("clip.mkv", max_frames=100) as reader:
            for frame in reader:
                process(frame)
    """

    def __init__(self, path: str, max_frames: int = 0) -> None: ...
    def close(self) -> None:
        """Mark the reader closed; further iteration raises ``StopIteration``."""
        ...
    def __enter__(self) -> MediaReader: ...
    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool: ...
    def __iter__(self) -> MediaReader: ...
    def __next__(self) -> VideoFrame: ...
    @property
    def frames_read(self) -> int:
        """Number of frames produced so far."""
        ...
    @property
    def closed(self) -> bool: ...
    @property
    def path(self) -> str: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------

def probe(path: str) -> MediaFileInfo:
    """Probe ``path`` and return a :class:`MediaFileInfo` summary.

    Raises ``ValueError`` if ``path`` is empty.
    """
    ...

def open_video(path: str, max_frames: int = 0) -> MediaReader:
    """Open ``path`` and return a :class:`MediaReader` iterator."""
    ...

def transcode(
    input_path: str,
    output_path: str,
    video_crf: float = 28.0,
    audio_bitrate_kbps: int = 128,
) -> TranscodeResult:
    """Transcode ``input_path`` into ``output_path``.

    Raises ``ValueError`` for empty paths or out-of-range ``video_crf``.
    """
    ...

def list_supported_formats() -> List[str]:
    """Return the list of container format names supported by OxiMedia."""
    ...

def list_supported_codecs() -> List[str]:
    """Return the list of codec names supported by OxiMedia."""
    ...
