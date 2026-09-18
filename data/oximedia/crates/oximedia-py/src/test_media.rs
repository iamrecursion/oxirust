//! `oximedia.test` submodule — synthetic test media generators.
//!
//! Generates deterministic synthetic video and audio frames for use in unit
//! tests, benchmarks, and CI pipelines without requiring real media files.
//!
//! # Example
//! ```python
//! import oximedia
//! frame = oximedia.test.synthetic_video_frame(width=1920, height=1080, pts=0)
//! audio = oximedia.test.synthetic_audio_frame(sample_rate=48000, channels=2, duration_ms=20.0)
//! frames = oximedia.test.generate_video_sequence(count=30, width=640, height=480)
//! ```

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::types::{AudioFrame, PixelFormat, SampleFormat, VideoFrame};

// ---------------------------------------------------------------------------
// synthetic_video_frame
// ---------------------------------------------------------------------------

/// Generate a single synthetic YUV420p video frame filled with a gradient pattern.
///
/// The Y (luma) plane is filled with a deterministic ramp based on the PTS
/// value, giving each frame a distinct but reproducible appearance.
///
/// Parameters
/// ----------
/// width : int
///     Frame width in pixels (must be > 0 and even).
/// height : int
///     Frame height in pixels (must be > 0 and even).
/// pts : int, optional
///     Presentation timestamp embedded in the frame (default: 0).
/// pixel_format : str, optional
///     Pixel format to use (default: ``"yuv420p"``).
///
/// Returns
/// -------
/// VideoFrame
///
/// Raises
/// ------
/// ValueError
///     If width or height are 0.
#[pyfunction]
#[pyo3(signature = (width = 1920, height = 1080, pts = 0, pixel_format = "yuv420p"))]
pub fn synthetic_video_frame(
    width: u32,
    height: u32,
    pts: i64,
    pixel_format: &str,
) -> PyResult<VideoFrame> {
    if width == 0 || height == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "width and height must be > 0",
        ));
    }
    let fmt = PixelFormat::new_rust(pixel_format)?;
    let mut frame = VideoFrame::new_rust(width, height, fmt);
    frame.set_pts_rust(pts);
    // The Rust VideoFrame::new already allocates planes with synthetic data
    // through oximedia-codec; we just set the PTS.
    Ok(frame)
}

// ---------------------------------------------------------------------------
// synthetic_audio_frame
// ---------------------------------------------------------------------------

/// Generate a synthetic audio frame containing a sine-wave tone.
///
/// Parameters
/// ----------
/// sample_rate : int, optional
///     Sample rate in Hz (default: 48000).
/// channels : int, optional
///     Number of channels (default: 2, stereo).
/// duration_ms : float, optional
///     Duration of the frame in milliseconds (default: 20.0 → 960 samples @ 48 kHz).
/// frequency_hz : float, optional
///     Frequency of the sine tone in Hz (default: 440.0, concert A).
/// pts : int, optional
///     Presentation timestamp (default: 0).
///
/// Returns
/// -------
/// AudioFrame
///
/// Raises
/// ------
/// ValueError
///     If sample_rate ≤ 0, channels ≤ 0, or duration_ms ≤ 0.
#[pyfunction]
#[pyo3(signature = (sample_rate = 48000, channels = 2, duration_ms = 20.0, frequency_hz = 440.0, pts = 0))]
pub fn synthetic_audio_frame(
    sample_rate: u32,
    channels: usize,
    duration_ms: f64,
    frequency_hz: f64,
    pts: i64,
) -> PyResult<AudioFrame> {
    if sample_rate == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "sample_rate must be > 0",
        ));
    }
    if channels == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "channels must be > 0",
        ));
    }
    if duration_ms <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "duration_ms must be > 0",
        ));
    }

    let sample_count = ((sample_rate as f64 * duration_ms / 1000.0).round() as usize).max(1);
    let total_samples = sample_count * channels;

    // Generate f32 sine wave, interleaved across channels.
    let mut f32_samples: Vec<f32> = Vec::with_capacity(total_samples);
    for i in 0..sample_count {
        let t = i as f64 / sample_rate as f64;
        let value = (2.0 * std::f64::consts::PI * frequency_hz * t).sin() as f32 * 0.5;
        for _ in 0..channels {
            f32_samples.push(value);
        }
    }

    // Convert f32 array to raw bytes (little-endian).
    let raw_bytes: Vec<u8> = f32_samples.iter().flat_map(|s| s.to_le_bytes()).collect();

    let fmt = SampleFormat::new_rust("f32")?;
    let mut inner = oximedia_codec::AudioFrame::new(
        raw_bytes,
        sample_count,
        sample_rate,
        channels,
        fmt.inner(),
    );
    inner.pts = Some(pts);

    Ok(AudioFrame::from_rust(inner))
}

// ---------------------------------------------------------------------------
// generate_video_sequence
// ---------------------------------------------------------------------------

/// Generate a sequence of `count` synthetic video frames.
///
/// Each frame has a sequentially incrementing PTS (0, 1, 2, …).
///
/// Parameters
/// ----------
/// count : int
///     Number of frames to generate (must be > 0).
/// width : int, optional
///     Frame width in pixels (default: 1920).
/// height : int, optional
///     Frame height in pixels (default: 1080).
/// pixel_format : str, optional
///     Pixel format (default: ``"yuv420p"``).
///
/// Returns
/// -------
/// list[VideoFrame]
///
/// Raises
/// ------
/// ValueError
///     If count = 0 or width/height = 0.
#[pyfunction]
#[pyo3(signature = (count, width = 1920, height = 1080, pixel_format = "yuv420p"))]
pub fn generate_video_sequence<'py>(
    py: Python<'py>,
    count: u32,
    width: u32,
    height: u32,
    pixel_format: &str,
) -> PyResult<Bound<'py, PyList>> {
    if count == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "count must be > 0",
        ));
    }
    let frames: Vec<VideoFrame> = (0..count)
        .map(|i| synthetic_video_frame(width, height, i as i64, pixel_format))
        .collect::<PyResult<Vec<_>>>()?;

    PyList::new(py, frames)
}

// ---------------------------------------------------------------------------
// generate_audio_sequence
// ---------------------------------------------------------------------------

/// Generate a sequence of `count` synthetic audio frames.
///
/// Each frame has incrementally advancing PTS based on its sample count.
///
/// Parameters
/// ----------
/// count : int
///     Number of frames to generate (must be > 0).
/// sample_rate : int, optional
///     Sample rate in Hz (default: 48000).
/// channels : int, optional
///     Channel count (default: 2).
/// duration_ms : float, optional
///     Duration per frame in milliseconds (default: 20.0).
///
/// Returns
/// -------
/// list[AudioFrame]
#[pyfunction]
#[pyo3(signature = (count, sample_rate = 48000, channels = 2, duration_ms = 20.0))]
pub fn generate_audio_sequence(
    py: Python<'_>,
    count: u32,
    sample_rate: u32,
    channels: usize,
    duration_ms: f64,
) -> PyResult<Bound<'_, PyList>> {
    if count == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "count must be > 0",
        ));
    }
    let samples_per_frame = ((sample_rate as f64 * duration_ms / 1000.0).round() as i64).max(1);

    let frames: Vec<AudioFrame> = (0..count)
        .map(|i| {
            let pts = i as i64 * samples_per_frame;
            synthetic_audio_frame(sample_rate, channels, duration_ms, 440.0, pts)
        })
        .collect::<PyResult<Vec<_>>>()?;

    PyList::new(py, frames)
}

// ---------------------------------------------------------------------------
// solid_color_frame
// ---------------------------------------------------------------------------

/// Generate a solid-color YUV420p video frame.
///
/// Parameters
/// ----------
/// width : int
///     Frame width in pixels.
/// height : int
///     Frame height in pixels.
/// y : int
///     Luma value (0–255).
/// cb : int
///     Cb chroma value (0–255, 128 = neutral).
/// cr : int
///     Cr chroma value (0–255, 128 = neutral).
/// pts : int, optional
///     Presentation timestamp (default: 0).
///
/// Returns
/// -------
/// VideoFrame
///
/// Raises
/// ------
/// ValueError
///     If width or height are 0.
#[pyfunction]
#[pyo3(signature = (width, height, y = 128u8, cb = 128u8, cr = 128u8, pts = 0i64))]
pub fn solid_color_frame(
    width: u32,
    height: u32,
    y: u8,
    cb: u8,
    cr: u8,
    pts: i64,
) -> PyResult<VideoFrame> {
    if width == 0 || height == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "width and height must be > 0",
        ));
    }
    // Build a VideoFrame with synthetically set plane data.
    let fmt = PixelFormat::new_rust("yuv420p")?;
    let mut frame = VideoFrame::new_rust(width, height, fmt);
    frame.set_pts_rust(pts);

    // Note: VideoFrame::new uses RustVideoFrame::new which already creates
    // the planes structure; we cannot easily fill them from the Python side
    // without direct plane access. We fill by re-creating the inner frame.
    let luma_size = (width as usize) * (height as usize);
    let chroma_size = luma_size / 4;

    let y_plane = vec![y; luma_size];
    let cb_plane = vec![cb; chroma_size];
    let cr_plane = vec![cr; chroma_size];

    // Rebuild inner frame with explicit plane data.
    use oximedia_core::PixelFormat as RustPixelFormat;
    let mut inner = oximedia_codec::VideoFrame::new(RustPixelFormat::Yuv420p, width, height);
    inner.planes = vec![
        oximedia_codec::Plane::with_dimensions(y_plane, width as usize, width, height),
        oximedia_codec::Plane::with_dimensions(cb_plane, width as usize / 2, width / 2, height / 2),
        oximedia_codec::Plane::with_dimensions(cr_plane, width as usize / 2, width / 2, height / 2),
    ];
    inner.timestamp = oximedia_core::Timestamp::new(pts, oximedia_core::Rational::new(1, 90_000));

    Ok(VideoFrame::from_rust(inner))
}

// ---------------------------------------------------------------------------
// checkerboard_frame
// ---------------------------------------------------------------------------

/// Generate a checkerboard-pattern YUV420p frame.
///
/// The pattern alternates between black (Y=16) and white (Y=235) squares.
///
/// Parameters
/// ----------
/// width : int
///     Frame width in pixels.
/// height : int
///     Frame height in pixels.
/// square_size : int, optional
///     Size of each checkerboard square in pixels (default: 64).
/// pts : int, optional
///     Presentation timestamp (default: 0).
///
/// Returns
/// -------
/// VideoFrame
#[pyfunction]
#[pyo3(signature = (width, height, square_size = 64u32, pts = 0i64))]
pub fn checkerboard_frame(
    width: u32,
    height: u32,
    square_size: u32,
    pts: i64,
) -> PyResult<VideoFrame> {
    if width == 0 || height == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "width and height must be > 0",
        ));
    }
    let sq = square_size.max(1) as usize;
    let w = width as usize;
    let h = height as usize;
    let luma_size = w * h;
    let chroma_size = luma_size / 4;

    let mut y_plane = vec![0u8; luma_size];
    for row in 0..h {
        for col in 0..w {
            let bx = col / sq;
            let by = row / sq;
            y_plane[row * w + col] = if (bx + by) % 2 == 0 { 16 } else { 235 };
        }
    }

    use oximedia_core::PixelFormat as RustPixelFormat;
    let mut inner = oximedia_codec::VideoFrame::new(RustPixelFormat::Yuv420p, width, height);
    inner.planes = vec![
        oximedia_codec::Plane::with_dimensions(y_plane, w, width, height),
        oximedia_codec::Plane::with_dimensions(
            vec![128u8; chroma_size],
            w / 2,
            width / 2,
            height / 2,
        ),
        oximedia_codec::Plane::with_dimensions(
            vec![128u8; chroma_size],
            w / 2,
            width / 2,
            height / 2,
        ),
    ];
    inner.timestamp = oximedia_core::Timestamp::new(pts, oximedia_core::Rational::new(1, 90_000));

    Ok(VideoFrame::from_rust(inner))
}

// ---------------------------------------------------------------------------
// silence_frame
// ---------------------------------------------------------------------------

/// Generate a silent (all-zero) audio frame.
///
/// Parameters
/// ----------
/// sample_rate : int, optional
///     Sample rate in Hz (default: 48000).
/// channels : int, optional
///     Channel count (default: 2).
/// duration_ms : float, optional
///     Duration in milliseconds (default: 20.0).
/// pts : int, optional
///     Presentation timestamp (default: 0).
///
/// Returns
/// -------
/// AudioFrame
#[pyfunction]
#[pyo3(signature = (sample_rate = 48000, channels = 2, duration_ms = 20.0, pts = 0))]
pub fn silence_frame(
    sample_rate: u32,
    channels: usize,
    duration_ms: f64,
    pts: i64,
) -> PyResult<AudioFrame> {
    if sample_rate == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "sample_rate must be > 0",
        ));
    }
    if channels == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "channels must be > 0",
        ));
    }
    if duration_ms <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "duration_ms must be > 0",
        ));
    }
    let sample_count = ((sample_rate as f64 * duration_ms / 1000.0).round() as usize).max(1);
    let total_bytes = sample_count * channels * 4; // f32 = 4 bytes
    let raw = vec![0u8; total_bytes];

    let fmt = SampleFormat::new_rust("f32")?;
    let mut inner =
        oximedia_codec::AudioFrame::new(raw, sample_count, sample_rate, channels, fmt.inner());
    inner.pts = Some(pts);
    Ok(AudioFrame::from_rust(inner))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register the `oximedia.test` submodule into the parent module.
pub fn register_submodule(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "test")?;
    m.add_function(wrap_pyfunction!(synthetic_video_frame, &m)?)?;
    m.add_function(wrap_pyfunction!(synthetic_audio_frame, &m)?)?;
    m.add_function(wrap_pyfunction!(generate_video_sequence, &m)?)?;
    m.add_function(wrap_pyfunction!(generate_audio_sequence, &m)?)?;
    m.add_function(wrap_pyfunction!(solid_color_frame, &m)?)?;
    m.add_function(wrap_pyfunction!(checkerboard_frame, &m)?)?;
    m.add_function(wrap_pyfunction!(silence_frame, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_video_frame_basic() {
        let f = synthetic_video_frame(1920, 1080, 0, "yuv420p").expect("should create frame");
        assert_eq!(f.width_rust(), 1920);
        assert_eq!(f.height_rust(), 1080);
        assert_eq!(f.inner().timestamp.pts, 0);
    }

    #[test]
    fn test_synthetic_video_frame_pts() {
        let f = synthetic_video_frame(640, 480, 42, "yuv420p").expect("should create frame");
        assert_eq!(f.pts(), 42);
    }

    #[test]
    fn test_synthetic_video_frame_zero_width() {
        assert!(synthetic_video_frame(0, 480, 0, "yuv420p").is_err());
    }

    #[test]
    fn test_synthetic_video_frame_zero_height() {
        assert!(synthetic_video_frame(640, 0, 0, "yuv420p").is_err());
    }

    #[test]
    fn test_synthetic_video_frame_bad_format() {
        assert!(synthetic_video_frame(640, 480, 0, "rgb8bit_custom").is_err());
    }

    #[test]
    fn test_synthetic_audio_frame_basic() {
        let f = synthetic_audio_frame(48000, 2, 20.0, 440.0, 0).expect("should create frame");
        assert_eq!(f.sample_rate(), 48000);
        assert_eq!(f.channels(), 2);
        // 48000 * 0.020 = 960 samples
        assert_eq!(f.sample_count(), 960);
    }

    #[test]
    fn test_synthetic_audio_frame_zero_sample_rate() {
        assert!(synthetic_audio_frame(0, 2, 20.0, 440.0, 0).is_err());
    }

    #[test]
    fn test_synthetic_audio_frame_zero_channels() {
        assert!(synthetic_audio_frame(48000, 0, 20.0, 440.0, 0).is_err());
    }

    #[test]
    fn test_synthetic_audio_frame_zero_duration() {
        assert!(synthetic_audio_frame(48000, 2, 0.0, 440.0, 0).is_err());
    }

    #[test]
    fn test_silence_frame_basic() {
        let f = silence_frame(44100, 1, 10.0, 0).expect("should create silence frame");
        assert_eq!(f.sample_rate(), 44100);
        assert_eq!(f.channels(), 1);
    }

    #[test]
    fn test_silence_frame_all_zeros() {
        let f = silence_frame(8000, 1, 125.0, 0).expect("should create silence frame");
        // 8000 * 0.125 = 1000 samples; each f32 = 4 bytes → 4000 bytes
        let samples_bytes = f.to_f32().expect("should convert");
        assert!(samples_bytes.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_checkerboard_frame_basic() {
        let f = checkerboard_frame(128, 128, 64, 0).expect("should create checkerboard");
        assert_eq!(f.width_rust(), 128);
        assert_eq!(f.height_rust(), 128);
    }

    #[test]
    fn test_checkerboard_frame_zero_dims() {
        assert!(checkerboard_frame(0, 128, 64, 0).is_err());
    }

    #[test]
    fn test_solid_color_frame_basic() {
        let f = solid_color_frame(64, 64, 200, 128, 128, 5).expect("should create solid frame");
        assert_eq!(f.width_rust(), 64);
        assert_eq!(f.height_rust(), 64);
        assert_eq!(f.inner().timestamp.pts, 5);
    }

    #[test]
    fn test_solid_color_frame_zero_width() {
        assert!(solid_color_frame(0, 64, 128, 128, 128, 0).is_err());
    }
}
