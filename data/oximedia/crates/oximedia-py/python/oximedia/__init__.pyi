"""
Type stubs for the ``oximedia`` Python extension module.

OxiMedia is the Sovereign Media Framework — patent-free, memory-safe
multimedia processing in pure Rust.  This file documents the public Python
API surface produced by the PyO3 bindings in ``crates/oximedia-py/src/``.

The stubs are hand-authored.  They are kept consistent with the Rust source
on a best-effort basis — when in doubt, the Rust ``#[pyclass]`` /
``#[pymethods]`` blocks in ``crates/oximedia-py/src/`` are the authoritative
reference.

Submodules (re-exported below) live in their own ``*.pyi`` files:

* :mod:`oximedia.cv2`        — OpenCV-compatible image processing
* :mod:`oximedia.io`         — file open / probe / transcode helpers
* :mod:`oximedia.utils`      — common media helper functions
* :mod:`oximedia.logging`    — Rust-to-Python logging bridge
* :mod:`oximedia.test`       — synthetic test-media generators
* :mod:`oximedia.benchmark`  — timing and throughput utilities
* :mod:`oximedia.quality`    — quality assessment helpers (re-exported)
* :mod:`oximedia.transcode`  — transcoder / preset / ABR ladder helpers
* :mod:`oximedia.dataframe`  — pandas / polars / pyarrow exporters

Tier 3 modules (AAF, IMF, EDL, distributed/farm/cloud, plugin, gaming, NDI,
VFX, virtual production, switcher, playout, etc.) expose ``Py*`` classes
through ``register()`` calls in :file:`lib.rs` but do not yet have hand-typed
stubs — :data:`Any` is used as a placeholder when the Python user references
those classes.  Future stub-coverage passes can fill them in.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from . import cv2 as cv2
from . import io as io
from . import logging as logging
from . import utils as utils
from . import test as test
from . import benchmark as benchmark

# ---------------------------------------------------------------------------
# Exception types
# ---------------------------------------------------------------------------

class OxiMediaError(Exception):
    """Base exception class raised by OxiMedia bindings on failure."""

# ---------------------------------------------------------------------------
# Core types — types.rs
# ---------------------------------------------------------------------------

class PixelFormat:
    """Pixel format for video frames (planar YUV variants and grayscale)."""

    YUV420P: str
    YUV422P: str
    YUV444P: str
    GRAY8: str

    def __init__(self, format: str) -> None: ...
    def is_planar(self) -> bool:
        """Return ``True`` if this format stores planes separately."""
        ...
    def plane_count(self) -> int:
        """Return the number of pixel planes (1 for grayscale, 3 for YUV)."""
        ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class SampleFormat:
    """Audio sample format (e.g. 32-bit float, 16-bit signed integer)."""

    F32: str
    I16: str

    def __init__(self, format: str) -> None: ...
    def sample_size(self) -> int:
        """Return the size, in bytes, of a single sample value."""
        ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class ChannelLayout:
    """Audio channel layout (mono / stereo)."""

    MONO: str
    STEREO: str

    def __init__(self, layout: str) -> None: ...
    def channel_count(self) -> int:
        """Return the number of channels described by this layout."""
        ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Rational:
    """Rational number used for frame rates and timebases (``num/den``)."""

    def __init__(self, num: int, den: int) -> None: ...
    @property
    def num(self) -> int:
        """Numerator."""
        ...
    @property
    def den(self) -> int:
        """Denominator."""
        ...
    def to_float(self) -> float:
        """Return the rational as ``num / den`` floating-point value."""
        ...
    def __getnewargs__(self) -> Tuple[int, int]: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class VideoFrame:
    """Decoded video frame containing pixel data.

    Implements neither the buffer protocol nor numpy-array conversion directly;
    use :meth:`plane` to obtain a zero-copy buffer view of an individual plane
    and wrap it with :func:`numpy.asarray` for array-like access.
    """

    def __init__(self, width: int, height: int, format: PixelFormat) -> None: ...
    @property
    def width(self) -> int:
        """Frame width in pixels."""
        ...
    @property
    def height(self) -> int:
        """Frame height in pixels."""
        ...
    @property
    def format(self) -> PixelFormat:
        """Pixel format of the frame data."""
        ...
    @property
    def pts(self) -> int:
        """Presentation timestamp (codec-defined units)."""
        ...
    @pts.setter
    def pts(self, value: int) -> None: ...
    def plane_data(self, index: int) -> bytes:
        """Return the bytes of plane ``index`` (0 = Y, 1 = U, 2 = V for YUV)."""
        ...
    def plane_stride(self, index: int) -> int:
        """Return the byte stride of plane ``index``."""
        ...
    def plane_count(self) -> int:
        """Return the number of pixel planes."""
        ...
    def plane(self, index: int) -> PyVideoPlaneBuffer:
        """Return a zero-copy buffer view of plane ``index``.

        The returned object implements Python's buffer protocol and can be
        wrapped with ``numpy.asarray(frame.plane(0))`` to obtain a
        ``(height, width)`` ``uint8`` array.
        """
        ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class PyVideoPlaneBuffer:
    """Zero-copy numpy-compatible buffer view for a single plane of a :class:`VideoFrame`."""

    @property
    def plane_index(self) -> int:
        """Plane index within the parent frame."""
        ...
    @property
    def height(self) -> int:
        """Plane height in pixels."""
        ...
    @property
    def width(self) -> int:
        """Plane width in pixels."""
        ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class AudioFrame:
    """Decoded audio frame containing PCM sample data.

    Implements Python's buffer protocol so ``numpy.asarray(frame)`` returns a
    zero-copy 2D view with shape ``(sample_count, channels)`` and the
    appropriate numpy dtype (``float32``, ``int16``, …).
    """

    def __init__(
        self,
        samples: bytes,
        sample_count: int,
        sample_rate: int,
        channels: int,
        format: SampleFormat,
    ) -> None: ...
    def samples(self) -> bytes:
        """Return raw interleaved sample bytes."""
        ...
    @property
    def sample_count(self) -> int:
        """Number of samples per channel."""
        ...
    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz."""
        ...
    @property
    def channels(self) -> int:
        """Number of channels."""
        ...
    @property
    def format(self) -> SampleFormat:
        """Sample format."""
        ...
    @property
    def pts(self) -> Optional[int]:
        """Presentation timestamp, if known."""
        ...
    def duration_seconds(self) -> float:
        """Frame duration, in seconds."""
        ...
    def to_f32(self) -> List[float]:
        """Return samples as a flat list of ``float32`` values."""
        ...
    def to_i16(self) -> List[int]:
        """Return samples as a flat list of signed 16-bit integers."""
        ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class EncoderPreset:
    """Encoder speed/quality tradeoff preset."""

    ULTRAFAST: str
    FAST: str
    MEDIUM: str
    SLOW: str
    VERYSLOW: str

    def __init__(self, preset: str) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class EncoderConfig:
    """Video encoder configuration (resolution, bitrate, preset, key-int)."""

    def __init__(
        self,
        width: int,
        height: int,
        framerate: Tuple[int, int] = (30, 1),
        crf: float = 28.0,
        preset: Optional[EncoderPreset] = None,
        keyint: int = 250,
    ) -> None: ...
    @property
    def width(self) -> int:
        """Width in pixels."""
        ...
    @property
    def height(self) -> int:
        """Height in pixels."""
        ...
    @property
    def framerate(self) -> Tuple[int, int]:
        """Frame rate as ``(numerator, denominator)``."""
        ...
    @property
    def keyint(self) -> int:
        """Key-frame interval in frames."""
        ...
    def __getstate__(self) -> Tuple[int, int, Tuple[int, int], int]: ...
    def __setstate__(self, state: Tuple[int, int, Tuple[int, int], int]) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Video codec bindings — video.rs
# ---------------------------------------------------------------------------

class Av1Decoder:
    """AV1 video decoder.

    Decodes AV1 compressed video packets into raw :class:`VideoFrame` objects.
    Use as a context manager to flush and reset on exit.
    """

    def __init__(self) -> None: ...
    def send_packet(self, data: bytes, pts: int = 0) -> None:
        """Submit a compressed packet for decoding."""
        ...
    def receive_frame(self) -> Optional[VideoFrame]:
        """Return the next decoded frame, or ``None`` if more data is needed."""
        ...
    def flush(self) -> None:
        """Flush the decoder to drain any buffered frames."""
        ...
    def reset(self) -> None:
        """Reset internal decoder state."""
        ...
    def dimensions(self) -> Optional[Tuple[int, int]]:
        """Return the output ``(width, height)`` if known."""
        ...
    def __enter__(self) -> Av1Decoder: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Av1Encoder:
    """AV1 video encoder.

    Accepts raw :class:`VideoFrame` objects and yields compressed AV1 packets
    via :meth:`receive_packet`.
    """

    def __init__(self, config: EncoderConfig) -> None: ...
    def send_frame(self, frame: VideoFrame) -> None:
        """Submit a raw frame for encoding."""
        ...
    def receive_packet(self) -> Optional[Dict[str, Any]]:
        """Return the next encoded packet as a dict, or ``None``.

        Keys: ``data`` (bytes), ``pts`` (int), ``dts`` (int),
        ``keyframe`` (bool), ``duration`` (int | None).
        """
        ...
    def flush(self) -> None:
        """Flush the encoder to drain remaining packets."""
        ...
    def __enter__(self) -> Av1Encoder: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Vp9Decoder:
    """VP9 video decoder."""

    def __init__(self) -> None: ...
    def send_packet(self, data: bytes, pts: int = 0) -> None: ...
    def receive_frame(self) -> Optional[VideoFrame]: ...
    def flush(self) -> None: ...
    def reset(self) -> None: ...
    def dimensions(self) -> Optional[Tuple[int, int]]: ...
    def __enter__(self) -> Vp9Decoder: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Vp8Decoder:
    """VP8 video decoder."""

    def __init__(self) -> None: ...
    def send_packet(self, data: bytes, pts: int = 0) -> None: ...
    def receive_frame(self) -> Optional[VideoFrame]: ...
    def flush(self) -> None: ...
    def reset(self) -> None: ...
    def dimensions(self) -> Optional[Tuple[int, int]]: ...
    def __enter__(self) -> Vp8Decoder: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Audio codec bindings — audio.rs
# ---------------------------------------------------------------------------

class OpusDecoder:
    """Opus audio decoder (push-then-decode model)."""

    def __init__(self, sample_rate: int, channels: int) -> None: ...
    def decode_packet(self, data: bytes) -> AudioFrame:
        """Decode a single compressed Opus packet."""
        ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def channels(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class VorbisDecoder:
    """Vorbis audio decoder using the send-packet / receive-frame model."""

    def __init__(self, sample_rate: int = 44100, channels: int = 2) -> None: ...
    def send_packet(self, data: bytes, pts: int = 0) -> None: ...
    def receive_frame(self) -> Optional[AudioFrame]: ...
    def flush(self) -> None: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def channels(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class FlacDecoder:
    """FLAC lossless audio decoder."""

    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 2,
        bits_per_sample: int = 16,
    ) -> None: ...
    def send_packet(self, data: bytes, pts: int = 0) -> None: ...
    def receive_frame(self) -> Optional[AudioFrame]: ...
    def flush(self) -> None: ...
    @property
    def sample_rate(self) -> Optional[int]: ...
    @property
    def channels(self) -> Optional[int]: ...
    @property
    def bits_per_sample(self) -> Optional[int]: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class OpusEncoderConfig:
    """Configuration for the :class:`OpusEncoder`."""

    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 2,
        bitrate: int = 128_000,
        frame_size: int = 960,
    ) -> None: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def channels(self) -> int: ...
    @property
    def bitrate(self) -> int: ...
    @property
    def frame_size(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class OpusEncoder:
    """Opus audio encoder."""

    def __init__(self, config: OpusEncoderConfig) -> None: ...
    def send_frame(self, frame: AudioFrame) -> None: ...
    def receive_packet(self) -> Optional[Dict[str, Any]]:
        """Return the next compressed packet ``{data, pts, duration}`` dict."""
        ...
    def flush(self) -> None: ...
    @property
    def bitrate(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Filter graph configuration — filters.rs
# ---------------------------------------------------------------------------

class PyScaleConfig:
    """Configuration for video scale/resize filter."""

    def __init__(
        self,
        width: int,
        height: int,
        algorithm: str = "bilinear",
    ) -> None: ...

class PyCropConfig:
    """Configuration for video crop filter."""

    def __init__(self, x: int, y: int, width: int, height: int) -> None: ...

class PyVolumeConfig:
    """Configuration for audio volume/gain filter."""

    def __init__(self, gain: float = 1.0) -> None: ...

class PyNormalizeConfig:
    """Configuration for audio loudness normalisation filter."""

    def __init__(
        self,
        mode: str = "ebu",
        target_level: float = -23.0,
    ) -> None: ...

# ---------------------------------------------------------------------------
# Container demux/mux — container.rs
# ---------------------------------------------------------------------------

class Packet:
    """Compressed-data packet emitted by demuxers / consumed by muxers."""

    def __init__(
        self,
        data: bytes,
        stream_index: int,
        pts: int,
        dts: Optional[int] = None,
        duration: Optional[int] = None,
        keyframe: bool = False,
        timebase_num: int = 1,
        timebase_den: int = 1000,
    ) -> None: ...
    def data(self) -> bytes: ...
    @property
    def stream_index(self) -> int: ...
    @property
    def pts(self) -> int: ...
    @property
    def dts(self) -> Optional[int]: ...
    @property
    def duration(self) -> Optional[int]: ...
    def is_keyframe(self) -> bool: ...
    def size(self) -> int:
        """Return the packet payload size in bytes."""
        ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class StreamInfo:
    """Per-stream descriptor produced by demuxers."""

    @property
    def index(self) -> int: ...
    @property
    def codec(self) -> str: ...
    @property
    def timebase(self) -> Tuple[int, int]: ...
    @property
    def width(self) -> Optional[int]: ...
    @property
    def height(self) -> Optional[int]: ...
    @property
    def sample_rate(self) -> Optional[int]: ...
    @property
    def channels(self) -> Optional[int]: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class MatroskaDemuxer:
    """Matroska / WebM demuxer."""

    def __init__(self, path: str) -> None: ...
    def probe(self) -> None:
        """Parse headers; must be called before reading packets."""
        ...
    def read_packet(self) -> Packet:
        """Return the next packet; raises ``StopIteration`` at EOF."""
        ...
    def streams(self) -> List[StreamInfo]: ...
    def close(self) -> None: ...
    def __enter__(self) -> MatroskaDemuxer: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool: ...
    def __iter__(self) -> MatroskaDemuxer: ...
    def __next__(self) -> Packet: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class OggDemuxer:
    """Ogg container demuxer (.ogg / .opus / .oga)."""

    def __init__(self, path: str) -> None: ...
    def probe(self) -> None: ...
    def read_packet(self) -> Packet: ...
    def streams(self) -> List[StreamInfo]: ...
    def close(self) -> None: ...
    def __enter__(self) -> OggDemuxer: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool: ...
    def __iter__(self) -> OggDemuxer: ...
    def __next__(self) -> Packet: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class MatroskaMuxer:
    """Matroska / WebM muxer."""

    def __init__(self, path: str, title: Optional[str] = None) -> None: ...
    def add_stream(self, stream_info: StreamInfo) -> int:
        """Add a stream and return its assigned index."""
        ...
    def write_header(self) -> None: ...
    def write_packet(self, packet: Packet) -> None: ...
    def write_trailer(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> MatroskaMuxer: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class OggMuxer:
    """Ogg container muxer."""

    def __init__(self, path: str) -> None: ...
    def add_stream(self, stream_info: StreamInfo) -> int: ...
    def write_header(self) -> None: ...
    def write_packet(self, packet: Packet) -> None: ...
    def write_trailer(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> OggMuxer: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Mp4Demuxer:
    """ISO BMFF / MP4 container demuxer."""

    def __init__(self, path: str) -> None: ...
    def probe(self) -> None: ...
    def read_packet(self) -> Optional[Packet]: ...
    def streams(self) -> List[StreamInfo]: ...

# ---------------------------------------------------------------------------
# Probe / media-info — probe.rs
# ---------------------------------------------------------------------------

class PyVideoInfo:
    """Video-specific parameters for a stream."""

    width: int
    height: int
    frame_rate: float
    pixel_format: str
    bit_depth: int

    def __init__(
        self,
        width: int,
        height: int,
        frame_rate: float = 0.0,
        pixel_format: str = "",
        bit_depth: int = 8,
    ) -> None: ...
    @property
    def pixel_count(self) -> int:
        """Number of pixels per frame."""
        ...
    @property
    def is_complete(self) -> bool:
        """``True`` if width / height / frame rate are all known."""
        ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class PyAudioInfo:
    """Audio-specific parameters for a stream."""

    sample_rate: int
    channels: int
    sample_format: str
    channel_layout: str

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        sample_format: str = "",
        channel_layout: str = "",
    ) -> None: ...
    def duration_samples(self, duration_seconds: float) -> int:
        """Return the number of samples in ``duration_seconds``."""
        ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class PyStreamInfo:
    """Information about a single media stream.

    ``PyStreamInfo`` declares ``video`` / ``audio`` both as ``#[pyo3(get)]``
    field getters (returning :class:`PyVideoInfo` / :class:`PyAudioInfo`)
    **and** as ``#[staticmethod]`` alternative constructors of the same name.
    The runtime resolution between the field-getter and the static-method
    constructor is determined by pyo3 macro order; this stub models the
    field-getter form, which is the safer assumption for typed code that
    expects ``stream.video`` to behave as an attribute.  Callers that wish
    to use the constructor form (``PyStreamInfo.video(0, "AV1", 1920, 1080,
    24.0)``) should silence mypy locally rather than rely on either form.
    """

    index: int
    codec: str
    media_type: str
    duration_seconds: Optional[float]
    video: Optional[PyVideoInfo]
    audio: Optional[PyAudioInfo]
    language: str

    def __init__(
        self,
        index: int,
        codec: str,
        media_type: str,
        duration_seconds: Optional[float] = None,
        video: Optional[PyVideoInfo] = None,
        audio: Optional[PyAudioInfo] = None,
        language: str = "",
    ) -> None: ...
    @property
    def is_video(self) -> bool: ...
    @property
    def is_audio(self) -> bool: ...
    @property
    def is_subtitle(self) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class PyMediaInfo:
    """Aggregated information about a probed media container."""

    duration: Optional[float]
    format_name: str
    file_size: Optional[int]
    bitrate: Optional[int]

    def __init__(
        self,
        streams: Sequence[PyStreamInfo],
        duration: Optional[float] = None,
        format_name: str = "",
        file_size: Optional[int] = None,
        bitrate: Optional[int] = None,
    ) -> None: ...
    @property
    def stream_count(self) -> int: ...
    @property
    def has_video(self) -> bool: ...
    @property
    def has_audio(self) -> bool: ...
    def streams(self) -> List[PyStreamInfo]: ...
    def video_streams(self) -> List[PyStreamInfo]: ...
    def audio_streams(self) -> List[PyStreamInfo]: ...
    def video_stream(self) -> Optional[PyStreamInfo]: ...
    def audio_stream(self) -> Optional[PyStreamInfo]: ...
    def stream_at(self, index: int) -> Optional[PyStreamInfo]: ...
    def bitrate_kbps(self) -> Optional[float]: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Quality assessment — quality.rs
# ---------------------------------------------------------------------------

class PyQualityScore:
    """Result of a quality assessment operation."""

    metric: str
    score: float
    components: Dict[str, float]

    def __repr__(self) -> str: ...

class PyQualityAssessor:
    """Stateful quality assessor (PSNR / SSIM / BRISQUE / NIQE / …)."""

    def __init__(self) -> None: ...
    def compute_psnr(
        self,
        ref_data: bytes,
        dist_data: bytes,
        width: int,
        height: int,
    ) -> PyQualityScore: ...
    def compute_ssim(
        self,
        ref_data: bytes,
        dist_data: bytes,
        width: int,
        height: int,
    ) -> PyQualityScore: ...
    def compute_brisque(
        self,
        data: bytes,
        width: int,
        height: int,
    ) -> PyQualityScore: ...
    def compute_niqe(
        self,
        data: bytes,
        width: int,
        height: int,
    ) -> PyQualityScore: ...
    def quality_report(
        self,
        data: bytes,
        width: int,
        height: int,
    ) -> List[PyQualityScore]: ...

def compute_psnr(
    ref_data: bytes,
    dist_data: bytes,
    width: int,
    height: int,
) -> float:
    """Compute PSNR (dB) between two grayscale images."""
    ...

def compute_ssim(
    ref_data: bytes,
    dist_data: bytes,
    width: int,
    height: int,
) -> float:
    """Compute SSIM (0–1) between two grayscale images."""
    ...

def quality_report(
    data: bytes,
    width: int,
    height: int,
) -> List[PyQualityScore]:
    """Run every available no-reference quality metric on ``data``."""
    ...

# ---------------------------------------------------------------------------
# Audio analysis — audio_analysis.rs
# ---------------------------------------------------------------------------

class PyLoudnessResult:
    """Loudness measurement result (EBU R128 / ATSC A/85 etc.)."""

    integrated_lufs: float
    short_term_lufs: float
    momentary_lufs: float
    loudness_range_lu: float
    true_peak_dbtp: float

    def __repr__(self) -> str: ...

class PySpectralFeatures:
    """Spectral features extracted from an audio buffer."""

    centroid_hz: float
    rolloff_hz: float
    flux: float
    zero_crossing_rate: float
    rms: float

    def __repr__(self) -> str: ...

def measure_loudness(
    samples: Sequence[float],
    sample_rate: float,
    channels: int = 2,
    standard: str = "ebu-r128",
) -> PyLoudnessResult:
    """Measure integrated loudness for an interleaved sample buffer."""
    ...

def detect_beats(samples: Sequence[float], sample_rate: float) -> List[float]:
    """Detect beat onsets and return their times in seconds."""
    ...

def spectral_features(
    samples: Sequence[float],
    sample_rate: float,
) -> PySpectralFeatures: ...

def detect_silence(
    samples: Sequence[float],
    sample_rate: float,
    threshold_db: float = -40.0,
    min_duration_ms: float = 100.0,
) -> List[Tuple[float, float]]:
    """Return ``(start_seconds, end_seconds)`` ranges where signal < threshold."""
    ...

# ---------------------------------------------------------------------------
# Scene / shot detection — scene.rs
# ---------------------------------------------------------------------------

class PyShot:
    """A single contiguous shot segment."""

    start_frame: int
    end_frame: int
    duration_frames: int
    avg_brightness: float
    motion_score: float
    shot_type: str

    def __repr__(self) -> str: ...

class PyScene:
    """A scene composed of one or more shots."""

    start_frame: int
    end_frame: int
    shots: List[PyShot]

    def __repr__(self) -> str: ...

def detect_scenes(
    frames: Sequence[bytes],
    width: int,
    height: int,
    threshold: float = 0.3,
) -> List[PyScene]: ...

def classify_shots(
    frames: Sequence[bytes],
    width: int,
    height: int,
) -> List[PyShot]: ...

# ---------------------------------------------------------------------------
# Filter graph — filter_graph.rs
# ---------------------------------------------------------------------------

class FilterGraph:
    """Filter graph for chaining video / audio processing nodes."""

    def __init__(self) -> None: ...
    def add_node(
        self,
        name: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> int: ...
    def connect(self, src_node: int, dst_node: int) -> None: ...
    def execute(self, frame: VideoFrame) -> VideoFrame: ...
    def __repr__(self) -> str: ...

class PyFilterNode:
    """Filter graph node (scale, crop, color, etc.)."""

    def __init__(self, name: str, **params: Any) -> None: ...
    @property
    def name(self) -> str: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# LUT (3D look-up tables) — lut.rs
# ---------------------------------------------------------------------------

class PyLut3d:
    """3D look-up table for colour grading / format transforms."""

    def apply_rgb(self, r: float, g: float, b: float) -> Tuple[float, float, float]: ...
    def apply_to_frame(self, data: bytes) -> bytes: ...
    def size(self) -> int: ...
    def __repr__(self) -> str: ...

def load_lut(path: str) -> PyLut3d:
    """Load a .cube LUT from disk."""
    ...

def apply_lut(lut: PyLut3d, data: bytes) -> bytes:
    """Apply ``lut`` to interleaved RGB byte data."""
    ...

def generate_identity_lut(size: int = 33) -> PyLut3d: ...

# ---------------------------------------------------------------------------
# Effects — effects.rs
# ---------------------------------------------------------------------------

def apply_color_grade(
    data: bytes,
    width: int,
    height: int,
    contrast: float = 0.0,
    brightness: float = 0.0,
    saturation: float = 1.0,
) -> bytes: ...

def apply_chromakey(
    data: bytes,
    width: int,
    height: int,
    key_r: int = 0,
    key_g: int = 255,
    key_b: int = 0,
    tolerance: int = 50,
) -> bytes: ...

def apply_blur(
    data: bytes,
    width: int,
    height: int,
    radius: int = 2,
) -> bytes: ...

def apply_vignette(
    data: bytes,
    width: int,
    height: int,
    strength: float = 0.5,
) -> bytes: ...

# ---------------------------------------------------------------------------
# Audio normalisation — audio_normalize.rs
# ---------------------------------------------------------------------------

def normalize_loudness(
    samples: Sequence[float],
    sample_rate: int,
    channels: int = 2,
    standard: str = "ebu-r128",
    max_gain_db: float = 20.0,
) -> List[float]: ...

def apply_compressor(
    samples: Sequence[float],
    sample_rate: int,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
) -> List[float]: ...

def apply_limiter(
    samples: Sequence[float],
    ceiling_db: float = -1.0,
) -> List[float]: ...

# ---------------------------------------------------------------------------
# Context managers — context_manager.rs
# ---------------------------------------------------------------------------

class ManagedDecoder:
    """Decoder wrapper supporting the Python context-manager protocol."""

    path: str
    is_open: bool

    def __init__(self, path: str) -> None: ...
    def open(self) -> None: ...
    def close(self) -> None: ...
    def probe(self) -> str:
        """Return ``key=value,…`` summary of the file."""
        ...
    def __enter__(self) -> ManagedDecoder: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class ManagedEncoder:
    """Encoder wrapper supporting the Python context-manager protocol."""

    width: int
    height: int
    is_open: bool
    frames_encoded: int

    def __init__(self, width: int = 1920, height: int = 1080) -> None: ...
    def open(self) -> None: ...
    def close(self) -> None: ...
    def encode_frame(self, frame_data: bytes, pts: int = 0) -> bytes: ...
    def __enter__(self) -> ManagedEncoder: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# Parallel / GIL-release utilities
# ---------------------------------------------------------------------------

def compute_checksums(data: Sequence[bytes]) -> List[int]:
    """Compute FNV-1a 64-bit checksums over a list of byte strings.

    Releases the GIL during computation so other Python threads can run.
    """
    ...

def compute_checksum_single(data: bytes) -> int:
    """Compute a single FNV-1a 64-bit checksum."""
    ...

# ---------------------------------------------------------------------------
# Top-level transcode / dataframe re-exports
# ---------------------------------------------------------------------------

# Transcoding (see transcode.pyi for full surface area).
class PyTranscodeResult:
    """Result returned after a transcode."""

    file_size: int
    duration_secs: float
    bitrate_kbps: float
    encoding_time_secs: float
    compression_ratio: float

    def to_dict(self) -> Dict[str, float]: ...
    def __repr__(self) -> str: ...

class PyAbrRung:
    """Single rendition rung in an ABR ladder."""

    width: int
    height: int
    bitrate_kbps: int
    codec: str
    crf: Optional[int]

    def __init__(
        self,
        width: int,
        height: int,
        bitrate_kbps: int,
        codec: str = "av1",
        crf: Optional[int] = None,
    ) -> None: ...
    def to_dict(self) -> Dict[str, str]: ...
    def __repr__(self) -> str: ...

class PyAbrLadder:
    """ABR ladder consisting of multiple :class:`PyAbrRung` entries."""

    def __init__(self) -> None: ...
    def add_rung(self, rung: PyAbrRung) -> None: ...
    def rungs(self) -> List[PyAbrRung]: ...
    def rung_count(self) -> int: ...
    def to_dict(self) -> List[Dict[str, str]]: ...
    @staticmethod
    def hls_default() -> PyAbrLadder: ...
    def __repr__(self) -> str: ...

class PyPresetInfo:
    """Metadata describing a transcoding preset."""

    name: str
    description: str
    video_codec: str
    audio_codec: str
    container: str
    quality_mode: str

    def to_dict(self) -> Dict[str, str]: ...
    def __repr__(self) -> str: ...

class PyTranscoder:
    """Fluent transcoder with chained configuration calls."""

    def __init__(self) -> None: ...
    def input(self, path: str) -> None: ...
    def output(self, path: str) -> None: ...
    def preset(self, name: str) -> None: ...
    def video_codec(self, codec: str) -> None: ...
    def audio_codec(self, codec: str) -> None: ...
    def crf(self, value: int) -> None: ...
    def bitrate(self, kbps: int) -> None: ...
    def scale(self, width: int, height: int) -> None: ...
    def frame_rate(self, fps: float) -> None: ...
    def audio_bitrate(self, kbps: int) -> None: ...
    def sample_rate(self, rate: int) -> None: ...
    def channels(self, ch: int) -> None: ...
    def quality_mode(self, mode: str) -> None: ...
    def two_pass(self, enable: bool) -> None: ...
    def transcode(self) -> PyTranscodeResult: ...
    def __repr__(self) -> str: ...

def transcode_simple(
    input: str,
    output: str,
    preset: Optional[str] = None,
    crf: Optional[int] = None,
) -> PyTranscodeResult: ...

def list_presets() -> List[PyPresetInfo]: ...
def list_codecs() -> List[Dict[str, str]]: ...
def validate_transcode_config(
    input: str,
    output: str,
    video_codec: Optional[str] = None,
    audio_codec: Optional[str] = None,
) -> List[str]: ...

# DataFrame export — re-exported from dataframe.rs.
def frames_to_dataframe(frames: Sequence[VideoFrame]) -> Any:
    """Build a ``pandas.DataFrame`` from a list of frames."""
    ...

def frames_to_polars(frames: Sequence[VideoFrame]) -> Any:
    """Build a ``polars.DataFrame`` from a list of frames."""
    ...

def analyze_to_dataframe(path: str) -> Any:
    """Analyse ``path`` and return per-frame metadata as a pandas DataFrame."""
    ...

def frames_to_arrow(frames: Sequence[VideoFrame]) -> Any:
    """Build a ``pyarrow.RecordBatch`` from a list of frames."""
    ...

def analyze_to_arrow(path: str) -> Any: ...

# ---------------------------------------------------------------------------
# Tier 3 namespaces
# ---------------------------------------------------------------------------
#
# The following classes are registered by ``register()`` calls in lib.rs but
# are typed as ``Any`` here pending a future stub-coverage pass:
#
# AAF, IMF, EDL, distributed/farm/cloud, plugin, gaming, NDI, virtual-prod,
# switcher, playout, profiler, render queue, and many more.  They are still
# importable as ``oximedia.<Name>`` from Python; mypy will simply give them
# ``Any`` semantics until the stubs are extended.
