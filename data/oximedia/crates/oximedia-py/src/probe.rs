//! Python bindings for media probing / container inspection.
//!
//! This module provides:
//!
//! - [`PyMediaInfo`] — a summary of streams found in a media container.
//! - [`PyStreamInfo`] — per-stream information (codec, dimensions, sample-rate, …).
//! - [`PyVideoInfo`] — video-specific parameters extracted from a stream.
//! - [`PyAudioInfo`] — audio-specific parameters extracted from a stream.
//!
//! # Python Example
//!
//! ```python
//! import oximedia
//!
//! streams = [
//!     oximedia.PyStreamInfo.video(0, "AV1", 1920, 1080, 30.0),
//!     oximedia.PyStreamInfo.audio(1, "Opus", 48000, 2),
//! ]
//! info = oximedia.PyMediaInfo(streams, duration=62.5, format_name="Matroska/WebM")
//!
//! print(info)                    # MediaInfo: 2 stream(s), 62.50s, Matroska/WebM
//! print(info.video_streams())    # [<PyStreamInfo …>]
//! print(info.has_video)          # True
//! print(info.has_audio)          # True
//! print(info.stream_count)       # 2
//! ```

use pyo3::prelude::*;
use pyo3::types::PyList;

// ─────────────────────────────────────────────────────────────
//  PyVideoInfo
// ─────────────────────────────────────────────────────────────

/// Video-specific parameters for a stream.
///
/// # Python Example
///
/// ```python
/// vi = oximedia.PyVideoInfo(1920, 1080, 30.0, pixel_format="yuv420p")
/// print(vi.width, vi.height, vi.frame_rate)
/// ```
#[pyclass(name = "PyVideoInfo")]
#[derive(Clone)]
pub struct PyVideoInfo {
    /// Frame width in pixels.
    #[pyo3(get)]
    pub width: u32,
    /// Frame height in pixels.
    #[pyo3(get)]
    pub height: u32,
    /// Nominal frame rate in frames per second (0 if unknown).
    #[pyo3(get)]
    pub frame_rate: f64,
    /// Pixel format string (e.g. `"yuv420p"`), empty if unknown.
    #[pyo3(get)]
    pub pixel_format: String,
    /// Bit depth per component (0 if unknown).
    #[pyo3(get)]
    pub bit_depth: u8,
}

#[pymethods]
impl PyVideoInfo {
    /// Construct video parameters.
    ///
    /// # Arguments
    ///
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    /// * `frame_rate` - Frames per second (0.0 = unknown)
    /// * `pixel_format` - Pixel format string (default `""`)
    /// * `bit_depth` - Bit depth (default `8`)
    #[new]
    #[pyo3(signature = (width, height, frame_rate=0.0, pixel_format="", bit_depth=8))]
    fn new(width: u32, height: u32, frame_rate: f64, pixel_format: &str, bit_depth: u8) -> Self {
        Self {
            width,
            height,
            frame_rate,
            pixel_format: pixel_format.to_string(),
            bit_depth,
        }
    }

    /// Return the total number of pixels per frame.
    #[getter]
    fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Return `True` if dimensions and frame rate are all known (non-zero).
    #[getter]
    fn is_complete(&self) -> bool {
        self.width > 0 && self.height > 0 && self.frame_rate > 0.0
    }

    fn __str__(&self) -> String {
        format!(
            "PyVideoInfo({}x{}, {:.3} fps, {})",
            self.width, self.height, self.frame_rate, self.pixel_format
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "PyVideoInfo(width={}, height={}, frame_rate={}, pixel_format={:?}, bit_depth={})",
            self.width, self.height, self.frame_rate, self.pixel_format, self.bit_depth
        )
    }
}

// ─────────────────────────────────────────────────────────────
//  PyAudioInfo
// ─────────────────────────────────────────────────────────────

/// Audio-specific parameters for a stream.
///
/// # Python Example
///
/// ```python
/// ai = oximedia.PyAudioInfo(48000, 2, sample_format="f32")
/// print(ai.sample_rate, ai.channels, ai.duration_samples(62.5))
/// ```
#[pyclass(name = "PyAudioInfo")]
#[derive(Clone)]
pub struct PyAudioInfo {
    /// Sample rate in Hz.
    #[pyo3(get)]
    pub sample_rate: u32,
    /// Number of channels.
    #[pyo3(get)]
    pub channels: u8,
    /// Sample format string (e.g. `"f32"`, `"i16"`), empty if unknown.
    #[pyo3(get)]
    pub sample_format: String,
    /// Nominal channel layout string (e.g. `"stereo"`, `"5.1"`), empty if unknown.
    #[pyo3(get)]
    pub channel_layout: String,
}

#[pymethods]
impl PyAudioInfo {
    /// Construct audio parameters.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of channels
    /// * `sample_format` - Format string (default `""`)
    /// * `channel_layout` - Layout string (default `""`)
    #[new]
    #[pyo3(signature = (sample_rate, channels, sample_format="", channel_layout=""))]
    fn new(sample_rate: u32, channels: u8, sample_format: &str, channel_layout: &str) -> Self {
        Self {
            sample_rate,
            channels,
            sample_format: sample_format.to_string(),
            channel_layout: channel_layout.to_string(),
        }
    }

    /// Compute how many samples correspond to `duration_seconds`.
    fn duration_samples(&self, duration_seconds: f64) -> u64 {
        (duration_seconds * f64::from(self.sample_rate)) as u64
    }

    fn __str__(&self) -> String {
        format!(
            "PyAudioInfo({}Hz, {} ch, {})",
            self.sample_rate, self.channels, self.sample_format
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "PyAudioInfo(sample_rate={}, channels={}, sample_format={:?})",
            self.sample_rate, self.channels, self.sample_format
        )
    }
}

// ─────────────────────────────────────────────────────────────
//  PyStreamInfo
// ─────────────────────────────────────────────────────────────

/// Information about a single media stream.
///
/// # Python Example
///
/// ```python
/// vs = oximedia.PyStreamInfo.video(0, "AV1", 3840, 2160, 60.0)
/// as_ = oximedia.PyStreamInfo.audio(1, "Opus", 48000, 2)
/// print(vs.is_video, as_.is_audio)  # True, True
/// ```
#[pyclass(name = "PyStreamInfo")]
#[derive(Clone)]
pub struct PyStreamInfo {
    /// Zero-based stream index within the container.
    #[pyo3(get)]
    pub index: usize,
    /// Codec name (e.g. `"AV1"`, `"Opus"`).
    #[pyo3(get)]
    pub codec: String,
    /// Media type: `"video"`, `"audio"`, `"subtitle"`, or `"data"`.
    #[pyo3(get)]
    pub media_type: String,
    /// Duration of the stream in seconds, `None` if unknown.
    #[pyo3(get)]
    pub duration_seconds: Option<f64>,
    /// Video parameters, `None` for non-video streams.
    #[pyo3(get)]
    pub video: Option<PyVideoInfo>,
    /// Audio parameters, `None` for non-audio streams.
    #[pyo3(get)]
    pub audio: Option<PyAudioInfo>,
    /// Human-readable language tag (e.g. `"eng"`), empty if unknown.
    #[pyo3(get)]
    pub language: String,
}

#[pymethods]
impl PyStreamInfo {
    /// Create a generic stream.
    ///
    /// Prefer the specialised constructors [`video`][PyStreamInfo::video] and
    /// [`audio`][PyStreamInfo::audio] where possible.
    #[new]
    #[pyo3(signature = (index, codec, media_type, duration_seconds=None, video=None, audio=None, language=""))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        index: usize,
        codec: &str,
        media_type: &str,
        duration_seconds: Option<f64>,
        video: Option<PyVideoInfo>,
        audio: Option<PyAudioInfo>,
        language: &str,
    ) -> Self {
        Self {
            index,
            codec: codec.to_string(),
            media_type: media_type.to_string(),
            duration_seconds,
            video,
            audio,
            language: language.to_string(),
        }
    }

    /// Convenience constructor for a video stream.
    ///
    /// # Arguments
    ///
    /// * `index` - Stream index
    /// * `codec` - Codec name string
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    /// * `frame_rate` - Frames per second (0.0 = unknown)
    /// * `duration_seconds` - Duration (default `None`)
    #[staticmethod]
    #[pyo3(signature = (index, codec, width, height, frame_rate=0.0, duration_seconds=None))]
    fn video(
        index: usize,
        codec: &str,
        width: u32,
        height: u32,
        frame_rate: f64,
        duration_seconds: Option<f64>,
    ) -> Self {
        let vi = PyVideoInfo::new(width, height, frame_rate, "", 8);
        Self {
            index,
            codec: codec.to_string(),
            media_type: "video".to_string(),
            duration_seconds,
            video: Some(vi),
            audio: None,
            language: String::new(),
        }
    }

    /// Convenience constructor for an audio stream.
    ///
    /// # Arguments
    ///
    /// * `index` - Stream index
    /// * `codec` - Codec name string
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of channels
    /// * `duration_seconds` - Duration (default `None`)
    #[staticmethod]
    #[pyo3(signature = (index, codec, sample_rate, channels, duration_seconds=None))]
    fn audio(
        index: usize,
        codec: &str,
        sample_rate: u32,
        channels: u8,
        duration_seconds: Option<f64>,
    ) -> Self {
        let ai = PyAudioInfo::new(sample_rate, channels, "", "");
        Self {
            index,
            codec: codec.to_string(),
            media_type: "audio".to_string(),
            duration_seconds,
            video: None,
            audio: Some(ai),
            language: String::new(),
        }
    }

    /// True if this is a video stream.
    #[getter]
    fn is_video(&self) -> bool {
        self.media_type == "video"
    }

    /// True if this is an audio stream.
    #[getter]
    fn is_audio(&self) -> bool {
        self.media_type == "audio"
    }

    /// True if this is a subtitle stream.
    #[getter]
    fn is_subtitle(&self) -> bool {
        self.media_type == "subtitle"
    }

    fn __str__(&self) -> String {
        match self.media_type.as_str() {
            "video" => {
                let vi = self.video.as_ref();
                format!(
                    "PyStreamInfo(#{}, {}, video {}x{} {:.2}fps)",
                    self.index,
                    self.codec,
                    vi.map_or(0, |v| v.width),
                    vi.map_or(0, |v| v.height),
                    vi.map_or(0.0, |v| v.frame_rate),
                )
            }
            "audio" => {
                let ai = self.audio.as_ref();
                format!(
                    "PyStreamInfo(#{}, {}, audio {}Hz {}ch)",
                    self.index,
                    self.codec,
                    ai.map_or(0, |a| a.sample_rate),
                    ai.map_or(0, |a| u32::from(a.channels)),
                )
            }
            _ => format!(
                "PyStreamInfo(#{}, {}, {})",
                self.index, self.codec, self.media_type
            ),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyStreamInfo(index={}, codec={:?}, media_type={:?}, duration={:?})",
            self.index, self.codec, self.media_type, self.duration_seconds
        )
    }
}

// ─────────────────────────────────────────────────────────────
//  PyMediaInfo
// ─────────────────────────────────────────────────────────────

/// Aggregated information about a probed media container.
///
/// # Python Example
///
/// ```python
/// import oximedia
///
/// streams = [
///     oximedia.PyStreamInfo.video(0, "AV1", 1920, 1080, 23.976),
///     oximedia.PyStreamInfo.audio(1, "Opus", 48000, 2),
/// ]
/// info = oximedia.PyMediaInfo(streams, duration=120.0, format_name="Matroska/WebM")
///
/// print(info.has_video)        # True
/// print(info.has_audio)        # True
/// print(info.stream_count)     # 2
/// print(info.video_stream())   # first video stream or None
/// ```
#[pyclass(name = "PyMediaInfo")]
#[derive(Clone)]
pub struct PyMediaInfo {
    /// All streams found in the container.
    streams: Vec<PyStreamInfo>,
    /// Total container duration in seconds, `None` if unknown.
    #[pyo3(get)]
    pub duration: Option<f64>,
    /// Human-readable container/format name (e.g. `"Matroska/WebM"`).
    #[pyo3(get)]
    pub format_name: String,
    /// File size in bytes, `None` if not applicable.
    #[pyo3(get)]
    pub file_size: Option<u64>,
    /// Overall bitrate in bits per second, `None` if unknown.
    #[pyo3(get)]
    pub bitrate: Option<u64>,
}

#[pymethods]
impl PyMediaInfo {
    /// Construct a `PyMediaInfo` from a list of streams.
    ///
    /// # Arguments
    ///
    /// * `streams` - List of [`PyStreamInfo`] objects
    /// * `duration` - Total duration in seconds (default `None`)
    /// * `format_name` - Format/container name string (default `""`)
    /// * `file_size` - File size in bytes (default `None`)
    /// * `bitrate` - Overall bitrate in bps (default `None`)
    #[new]
    #[pyo3(signature = (streams, duration=None, format_name="", file_size=None, bitrate=None))]
    fn new(
        streams: Vec<PyStreamInfo>,
        duration: Option<f64>,
        format_name: &str,
        file_size: Option<u64>,
        bitrate: Option<u64>,
    ) -> Self {
        Self {
            streams,
            duration,
            format_name: format_name.to_string(),
            file_size,
            bitrate,
        }
    }

    /// Return the total number of streams.
    #[getter]
    fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Return `True` if the container contains at least one video stream.
    #[getter]
    fn has_video(&self) -> bool {
        self.streams.iter().any(|s| s.is_video())
    }

    /// Return `True` if the container contains at least one audio stream.
    #[getter]
    fn has_audio(&self) -> bool {
        self.streams.iter().any(|s| s.is_audio())
    }

    /// Return all streams as a Python list.
    fn streams<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Py<PyStreamInfo>> = self
            .streams
            .iter()
            .map(|s| Py::new(py, s.clone()))
            .collect::<PyResult<_>>()?;
        let pylist = PyList::empty(py);
        for item in items {
            pylist.append(item)?;
        }
        Ok(pylist)
    }

    /// Return only the video streams.
    fn video_streams<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let pylist = PyList::empty(py);
        for s in self.streams.iter().filter(|s| s.is_video()) {
            pylist.append(Py::new(py, s.clone())?)?;
        }
        Ok(pylist)
    }

    /// Return only the audio streams.
    fn audio_streams<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let pylist = PyList::empty(py);
        for s in self.streams.iter().filter(|s| s.is_audio()) {
            pylist.append(Py::new(py, s.clone())?)?;
        }
        Ok(pylist)
    }

    /// Return the first video stream, or `None`.
    fn video_stream(&self) -> Option<PyStreamInfo> {
        self.streams.iter().find(|s| s.is_video()).cloned()
    }

    /// Return the first audio stream, or `None`.
    fn audio_stream(&self) -> Option<PyStreamInfo> {
        self.streams.iter().find(|s| s.is_audio()).cloned()
    }

    /// Return the stream at a given index, or `None`.
    fn stream_at(&self, index: usize) -> Option<PyStreamInfo> {
        self.streams.get(index).cloned()
    }

    /// Return the overall bitrate in kilobits per second, or `None`.
    fn bitrate_kbps(&self) -> Option<f64> {
        self.bitrate.map(|b| b as f64 / 1000.0)
    }

    fn __str__(&self) -> String {
        let dur = self
            .duration
            .map_or_else(|| "?".to_string(), |d| format!("{d:.2}s"));
        format!(
            "PyMediaInfo: {} stream(s), {}, {}",
            self.streams.len(),
            dur,
            self.format_name
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "PyMediaInfo(streams={}, duration={:?}, format_name={:?})",
            self.streams.len(),
            self.duration,
            self.format_name
        )
    }
}

// ─────────────────────────────────────────────────────────────
//  Unit tests (compiled for native targets)
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_video_stream() -> PyStreamInfo {
        PyStreamInfo::video(0, "AV1", 1920, 1080, 24.0, Some(120.0))
    }

    fn make_audio_stream() -> PyStreamInfo {
        PyStreamInfo::audio(1, "Opus", 48000, 2, Some(120.0))
    }

    fn make_media_info() -> PyMediaInfo {
        PyMediaInfo::new(
            vec![make_video_stream(), make_audio_stream()],
            Some(120.0),
            "Matroska/WebM",
            Some(52_428_800),
            Some(3_500_000),
        )
    }

    // ── PyVideoInfo ───────────────────────────────────────────

    #[test]
    fn test_video_info_new() {
        let vi = PyVideoInfo::new(3840, 2160, 60.0, "yuv420p", 10);
        assert_eq!(vi.width, 3840);
        assert_eq!(vi.height, 2160);
        assert!((vi.frame_rate - 60.0).abs() < f64::EPSILON);
        assert_eq!(vi.pixel_format, "yuv420p");
        assert_eq!(vi.bit_depth, 10);
    }

    #[test]
    fn test_video_info_pixel_count() {
        let vi = PyVideoInfo::new(1920, 1080, 30.0, "", 8);
        assert_eq!(vi.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_video_info_is_complete() {
        let complete = PyVideoInfo::new(1920, 1080, 25.0, "", 8);
        assert!(complete.is_complete());
        let incomplete = PyVideoInfo::new(1920, 1080, 0.0, "", 8);
        assert!(!incomplete.is_complete());
    }

    // ── PyAudioInfo ───────────────────────────────────────────

    #[test]
    fn test_audio_info_new() {
        let ai = PyAudioInfo::new(44100, 6, "f32", "5.1");
        assert_eq!(ai.sample_rate, 44100);
        assert_eq!(ai.channels, 6);
        assert_eq!(ai.sample_format, "f32");
        assert_eq!(ai.channel_layout, "5.1");
    }

    #[test]
    fn test_audio_info_duration_samples() {
        let ai = PyAudioInfo::new(48000, 2, "", "");
        assert_eq!(ai.duration_samples(1.0), 48000);
        assert_eq!(ai.duration_samples(0.5), 24000);
    }

    // ── PyStreamInfo ──────────────────────────────────────────

    #[test]
    fn test_stream_info_video() {
        let s = make_video_stream();
        assert!(s.is_video());
        assert!(!s.is_audio());
        assert!(!s.is_subtitle());
        assert_eq!(s.index, 0);
        assert_eq!(s.codec, "AV1");
        assert!(s.video.is_some());
        assert!(s.audio.is_none());
    }

    #[test]
    fn test_stream_info_audio() {
        let s = make_audio_stream();
        assert!(s.is_audio());
        assert!(!s.is_video());
        assert_eq!(s.index, 1);
        assert_eq!(s.codec, "Opus");
        assert!(s.audio.is_some());
        assert!(s.video.is_none());
    }

    #[test]
    fn test_stream_info_generic_new() {
        let s = PyStreamInfo::new(2, "FLAC", "audio", Some(10.0), None, None, "eng");
        assert_eq!(s.index, 2);
        assert_eq!(s.language, "eng");
        assert_eq!(s.duration_seconds, Some(10.0));
    }

    #[test]
    fn test_stream_info_str() {
        let s = make_video_stream();
        let repr = s.__str__();
        assert!(repr.contains("AV1"), "Expected AV1 in: {repr}");
        assert!(repr.contains("1920"), "Expected 1920 in: {repr}");
    }

    // ── PyMediaInfo ───────────────────────────────────────────

    #[test]
    fn test_media_info_basic() {
        let info = make_media_info();
        assert_eq!(info.stream_count(), 2);
        assert!(info.has_video());
        assert!(info.has_audio());
        assert_eq!(info.duration, Some(120.0));
        assert_eq!(info.format_name, "Matroska/WebM");
    }

    #[test]
    fn test_media_info_video_stream() {
        let info = make_media_info();
        let vs = info.video_stream().expect("should have video stream");
        assert!(vs.is_video());
        assert_eq!(vs.codec, "AV1");
    }

    #[test]
    fn test_media_info_audio_stream() {
        let info = make_media_info();
        let as_ = info.audio_stream().expect("should have audio stream");
        assert!(as_.is_audio());
        assert_eq!(as_.codec, "Opus");
    }

    #[test]
    fn test_media_info_stream_at() {
        let info = make_media_info();
        assert!(info.stream_at(0).is_some());
        assert!(info.stream_at(1).is_some());
        assert!(info.stream_at(99).is_none());
    }

    #[test]
    fn test_media_info_bitrate_kbps() {
        let info = make_media_info();
        let kbps = info.bitrate_kbps().expect("should have bitrate");
        assert!((kbps - 3500.0).abs() < 0.01);
    }

    #[test]
    fn test_media_info_no_video() {
        let info = PyMediaInfo::new(vec![make_audio_stream()], Some(60.0), "Ogg", None, None);
        assert!(!info.has_video());
        assert!(info.has_audio());
        assert!(info.video_stream().is_none());
    }

    #[test]
    fn test_media_info_str() {
        let info = make_media_info();
        let s = info.__str__();
        assert!(s.contains("2 stream(s)"), "Got: {s}");
        assert!(s.contains("Matroska/WebM"), "Got: {s}");
    }

    #[test]
    fn test_media_info_repr() {
        let info = make_media_info();
        let r = info.__repr__();
        assert!(r.contains("PyMediaInfo"), "Got: {r}");
    }
}
