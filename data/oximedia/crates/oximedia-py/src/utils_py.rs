//! `oximedia.utils` submodule — common media helper functions.
//!
//! Provides pure-Rust helper functions exposed to Python for common
//! media utility operations: timecode arithmetic, rational frame-rate
//! conversion, duration formatting, bitrate estimation, and more.
//!
//! # Example
//! ```python
//! import oximedia
//! tc = oximedia.utils.duration_to_timecode(3723.5, fps=25.0)
//! # "01:02:03:12"
//! r = oximedia.utils.fps_to_rational(29.97)
//! # (30000, 1001)
//! ```

use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// duration_to_timecode
// ---------------------------------------------------------------------------

/// Convert a duration in seconds to a SMPTE timecode string.
///
/// Produces a string of the form ``HH:MM:SS:FF`` (hours, minutes, seconds,
/// frames).  Drop-frame notation is not supported in this helper; for full
/// drop-frame support use the :class:`oximedia.timecode.Timecode` class.
///
/// Parameters
/// ----------
/// seconds : float
///     Duration in seconds (must be ≥ 0).
/// fps : float, optional
///     Frame rate (default: 25.0).
///
/// Returns
/// -------
/// str
///     Timecode in ``HH:MM:SS:FF`` format.
///
/// Raises
/// ------
/// ValueError
///     If `seconds` < 0 or `fps` ≤ 0.
#[pyfunction]
#[pyo3(signature = (seconds, fps = 25.0))]
pub fn duration_to_timecode(seconds: f64, fps: f64) -> PyResult<String> {
    if seconds < 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "seconds must be >= 0",
        ));
    }
    if fps <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "fps must be > 0",
        ));
    }
    let total_frames = (seconds * fps).floor() as u64;
    let fps_int = fps.round() as u64;
    let fps_safe = fps_int.max(1);

    let frames = total_frames % fps_safe;
    let total_secs = total_frames / fps_safe;
    let secs = total_secs % 60;
    let total_mins = total_secs / 60;
    let mins = total_mins % 60;
    let hours = total_mins / 60;

    Ok(format!("{hours:02}:{mins:02}:{secs:02}:{frames:02}"))
}

// ---------------------------------------------------------------------------
// timecode_to_duration
// ---------------------------------------------------------------------------

/// Parse a SMPTE timecode string into a duration in seconds.
///
/// Accepts ``HH:MM:SS:FF`` format.
///
/// Parameters
/// ----------
/// timecode : str
///     Timecode string (e.g. ``"01:02:03:12"``).
/// fps : float, optional
///     Frame rate used to interpret the frame field (default: 25.0).
///
/// Returns
/// -------
/// float
///     Duration in seconds.
///
/// Raises
/// ------
/// ValueError
///     If the timecode string is malformed or fps ≤ 0.
#[pyfunction]
#[pyo3(signature = (timecode, fps = 25.0))]
pub fn timecode_to_duration(timecode: &str, fps: f64) -> PyResult<f64> {
    if fps <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "fps must be > 0",
        ));
    }
    let parts: Vec<&str> = timecode.split(':').collect();
    if parts.len() != 4 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Expected HH:MM:SS:FF, got: {:?}",
            timecode
        )));
    }
    let parse = |s: &str, name: &str| -> PyResult<u64> {
        s.trim().parse::<u64>().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Cannot parse {name} field '{s}' as integer"
            ))
        })
    };
    let h = parse(parts[0], "hours")?;
    let m = parse(parts[1], "minutes")?;
    let s = parse(parts[2], "seconds")?;
    let f = parse(parts[3], "frames")?;

    if m >= 60 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "minutes must be < 60",
        ));
    }
    if s >= 60 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "seconds must be < 60",
        ));
    }

    let total_secs = (h * 3600 + m * 60 + s) as f64;
    let frame_secs = f as f64 / fps;
    Ok(total_secs + frame_secs)
}

// ---------------------------------------------------------------------------
// fps_to_rational
// ---------------------------------------------------------------------------

/// Convert a floating-point frame rate to a rational (numerator, denominator) tuple.
///
/// Common frame rates are recognised and returned as exact rationals.
///
/// Parameters
/// ----------
/// fps : float
///     Frame rate value (e.g. 29.97, 23.976, 25.0).
///
/// Returns
/// -------
/// tuple[int, int]
///     ``(numerator, denominator)`` in lowest terms.
///
/// Raises
/// ------
/// ValueError
///     If fps ≤ 0.
#[pyfunction]
pub fn fps_to_rational(fps: f64) -> PyResult<(u32, u32)> {
    if fps <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "fps must be > 0",
        ));
    }
    // Well-known rates.
    let (num, den) = match (fps * 1000.0).round() as u64 {
        23976 => (24000, 1001),
        24000 => (24, 1),
        25000 => (25, 1),
        29970 => (30000, 1001),
        30000 => (30, 1),
        47952 => (48000, 1001),
        48000 => (48, 1),
        50000 => (50, 1),
        59940 => (60000, 1001),
        60000 => (60, 1),
        120000 => (120, 1),
        _ => {
            // Generic: multiply by 1000 to preserve 3 decimal places, then reduce.
            let num_big = (fps * 1000.0).round() as u64;
            let den_big: u64 = 1000;
            let g = gcd(num_big, den_big);
            ((num_big / g) as u32, (den_big / g) as u32)
        }
    };
    Ok((num, den))
}

/// Greatest common divisor (Euclidean).
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1)
}

// ---------------------------------------------------------------------------
// format_duration
// ---------------------------------------------------------------------------

/// Format a duration in seconds as a human-readable string.
///
/// Parameters
/// ----------
/// seconds : float
///     Duration in seconds (must be ≥ 0).
/// precision : int, optional
///     Decimal places for sub-second component (default: 3 → milliseconds).
///
/// Returns
/// -------
/// str
///     E.g. ``"1h 2m 3.456s"`` or ``"45.000s"``.
///
/// Raises
/// ------
/// ValueError
///     If seconds < 0 or precision > 9.
#[pyfunction]
#[pyo3(signature = (seconds, precision = 3))]
pub fn format_duration(seconds: f64, precision: u8) -> PyResult<String> {
    if seconds < 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "seconds must be >= 0",
        ));
    }
    if precision > 9 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "precision must be <= 9",
        ));
    }
    let h = (seconds / 3600.0) as u64;
    let m = ((seconds % 3600.0) / 60.0) as u64;
    let s = seconds % 60.0;

    let sec_str = format!("{s:.prec$}", prec = precision as usize);

    if h > 0 {
        Ok(format!("{h}h {m}m {sec_str}s"))
    } else if m > 0 {
        Ok(format!("{m}m {sec_str}s"))
    } else {
        Ok(format!("{sec_str}s"))
    }
}

// ---------------------------------------------------------------------------
// estimate_bitrate
// ---------------------------------------------------------------------------

/// Estimate the average bitrate of a media stream.
///
/// Parameters
/// ----------
/// file_size_bytes : int
///     Total file size in bytes.
/// duration_seconds : float
///     Stream duration in seconds.
///
/// Returns
/// -------
/// float
///     Average bitrate in kilobits per second (kbps).
///
/// Raises
/// ------
/// ValueError
///     If duration_seconds ≤ 0 or file_size_bytes < 0.
#[pyfunction]
pub fn estimate_bitrate(file_size_bytes: u64, duration_seconds: f64) -> PyResult<f64> {
    if duration_seconds <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "duration_seconds must be > 0",
        ));
    }
    let bits = file_size_bytes as f64 * 8.0;
    Ok(bits / duration_seconds / 1000.0)
}

// ---------------------------------------------------------------------------
// calculate_frame_size
// ---------------------------------------------------------------------------

/// Calculate the uncompressed size of a single video frame.
///
/// Parameters
/// ----------
/// width : int
///     Frame width in pixels.
/// height : int
///     Frame height in pixels.
/// pixel_format : str, optional
///     Pixel format name (default: ``"yuv420p"``).
///
/// Returns
/// -------
/// int
///     Frame size in bytes.
///
/// Raises
/// ------
/// ValueError
///     If width or height are 0, or if the pixel format is not recognised.
#[pyfunction]
#[pyo3(signature = (width, height, pixel_format = "yuv420p"))]
pub fn calculate_frame_size(width: u32, height: u32, pixel_format: &str) -> PyResult<u64> {
    if width == 0 || height == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "width and height must be > 0",
        ));
    }
    let pixels = u64::from(width) * u64::from(height);
    let bytes = match pixel_format {
        "yuv420p" => pixels * 3 / 2,
        "yuv422p" => pixels * 2,
        "yuv444p" => pixels * 3,
        "gray8" => pixels,
        "rgb24" | "bgr24" => pixels * 3,
        "rgba" | "bgra" => pixels * 4,
        "yuv420p10le" | "yuv420p10be" => pixels * 3 / 2 * 2,
        other => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown pixel format: '{other}'"
            )))
        }
    };
    Ok(bytes)
}

// ---------------------------------------------------------------------------
// media_info_summary
// ---------------------------------------------------------------------------

/// Build a human-readable summary dict for a media file from basic metadata.
///
/// Parameters
/// ----------
/// path : str
///     File path.
/// duration_seconds : float
///     Duration in seconds.
/// video_codec : str | None
///     Video codec name.
/// audio_codec : str | None
///     Audio codec name.
/// fps : float, optional
///     Frame rate (default: 25.0).
///
/// Returns
/// -------
/// dict[str, object]
///     Summary dictionary suitable for display.
#[pyfunction]
#[pyo3(signature = (path, duration_seconds, video_codec, audio_codec, fps = 25.0))]
pub fn media_info_summary<'py>(
    py: Python<'py>,
    path: &str,
    duration_seconds: f64,
    video_codec: Option<&str>,
    audio_codec: Option<&str>,
    fps: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("path", path)?;
    d.set_item("duration_seconds", duration_seconds)?;
    let tc = if duration_seconds >= 0.0 && fps > 0.0 {
        duration_to_timecode(duration_seconds, fps).unwrap_or_else(|_| "??:??:??:??".to_string())
    } else {
        "??:??:??:??".to_string()
    };
    d.set_item("timecode_in", "00:00:00:00")?;
    d.set_item("timecode_out", tc)?;
    d.set_item("video_codec", video_codec)?;
    d.set_item("audio_codec", audio_codec)?;
    d.set_item("fps", fps)?;
    Ok(d)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register the `oximedia.utils` submodule into the parent module.
pub fn register_submodule(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "utils")?;
    m.add_function(wrap_pyfunction!(duration_to_timecode, &m)?)?;
    m.add_function(wrap_pyfunction!(timecode_to_duration, &m)?)?;
    m.add_function(wrap_pyfunction!(fps_to_rational, &m)?)?;
    m.add_function(wrap_pyfunction!(format_duration, &m)?)?;
    m.add_function(wrap_pyfunction!(estimate_bitrate, &m)?)?;
    m.add_function(wrap_pyfunction!(calculate_frame_size, &m)?)?;
    m.add_function(wrap_pyfunction!(media_info_summary, &m)?)?;
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
    fn test_duration_to_timecode_zero() {
        assert_eq!(
            duration_to_timecode(0.0, 25.0).expect("zero duration TC"),
            "00:00:00:00"
        );
    }

    #[test]
    fn test_duration_to_timecode_one_hour() {
        // 3600 seconds at 25 fps → 01:00:00:00
        assert_eq!(
            duration_to_timecode(3600.0, 25.0).expect("one-hour TC"),
            "01:00:00:00"
        );
    }

    #[test]
    fn test_duration_to_timecode_mixed() {
        // 3723.5 s at 25 fps: 1h 2m 3s + 0.5*25=12f → 01:02:03:12
        let tc = duration_to_timecode(3723.5, 25.0).expect("mixed duration TC");
        assert_eq!(tc, "01:02:03:12");
    }

    #[test]
    fn test_duration_to_timecode_negative() {
        assert!(duration_to_timecode(-1.0, 25.0).is_err());
    }

    #[test]
    fn test_duration_to_timecode_zero_fps() {
        assert!(duration_to_timecode(10.0, 0.0).is_err());
    }

    #[test]
    fn test_timecode_to_duration_round_trip() {
        let secs = 3723.5;
        let tc = duration_to_timecode(secs, 25.0).expect("roundtrip TC encode");
        let back = timecode_to_duration(&tc, 25.0).expect("roundtrip TC decode");
        assert!((back - secs).abs() < 0.1);
    }

    #[test]
    fn test_timecode_to_duration_bad_format() {
        assert!(timecode_to_duration("01:02:03", 25.0).is_err());
    }

    #[test]
    fn test_fps_to_rational_known() {
        assert_eq!(fps_to_rational(25.0).expect("25 fps"), (25, 1));
        assert_eq!(fps_to_rational(29.97).expect("29.97 fps"), (30000, 1001));
        assert_eq!(fps_to_rational(23.976).expect("23.976 fps"), (24000, 1001));
        assert_eq!(fps_to_rational(24.0).expect("24 fps"), (24, 1));
        assert_eq!(fps_to_rational(30.0).expect("30 fps"), (30, 1));
        assert_eq!(fps_to_rational(60.0).expect("60 fps"), (60, 1));
    }

    #[test]
    fn test_fps_to_rational_zero() {
        assert!(fps_to_rational(0.0).is_err());
    }

    #[test]
    fn test_fps_to_rational_generic() {
        let (n, d) = fps_to_rational(10.0).expect("10 fps generic");
        assert_eq!(n as f64 / d as f64, 10.0);
    }

    #[test]
    fn test_format_duration_hours() {
        let s = format_duration(3723.456, 3).expect("format hours duration");
        assert!(s.contains("1h"));
        assert!(s.contains("2m"));
    }

    #[test]
    fn test_format_duration_minutes() {
        let s = format_duration(125.5, 2).expect("format minutes duration");
        assert!(s.contains("2m"));
    }

    #[test]
    fn test_format_duration_seconds_only() {
        let s = format_duration(45.123, 3).expect("format seconds duration");
        assert_eq!(s, "45.123s");
    }

    #[test]
    fn test_format_duration_negative() {
        assert!(format_duration(-1.0, 3).is_err());
    }

    #[test]
    fn test_format_duration_bad_precision() {
        assert!(format_duration(10.0, 10).is_err());
    }

    #[test]
    fn test_estimate_bitrate() {
        // 1 MB over 8 seconds = 1_000_000 * 8 bits / 8s / 1000 = 1000 kbps
        let kbps = estimate_bitrate(1_000_000, 8.0).expect("estimate bitrate for valid input");
        assert!((kbps - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_estimate_bitrate_zero_duration() {
        assert!(estimate_bitrate(1024, 0.0).is_err());
    }

    #[test]
    fn test_calculate_frame_size_yuv420p() {
        // 1920×1080 yuv420p = 1920*1080*3/2 = 3,110,400
        let s = calculate_frame_size(1920, 1080, "yuv420p").expect("frame size for yuv420p");
        assert_eq!(s, 3_110_400);
    }

    #[test]
    fn test_calculate_frame_size_rgb24() {
        let s = calculate_frame_size(4, 4, "rgb24").expect("frame size for rgb24");
        assert_eq!(s, 4 * 4 * 3);
    }

    #[test]
    fn test_calculate_frame_size_zero_dims() {
        assert!(calculate_frame_size(0, 1080, "yuv420p").is_err());
    }

    #[test]
    fn test_calculate_frame_size_unknown_format() {
        assert!(calculate_frame_size(1920, 1080, "xyzcustom").is_err());
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(7, 3), 1);
        assert_eq!(gcd(0, 5), 5);
    }
}
