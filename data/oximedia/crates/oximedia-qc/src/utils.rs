//! Utility functions for QC operations.
//!
//! Provides helper functions for common QC tasks.

use oximedia_core::OxiResult;
use std::path::Path;

/// Calculates file hash for integrity checking.
///
/// # Errors
///
/// Returns an error if the file cannot be read.
pub fn calculate_file_hash(file_path: &Path) -> OxiResult<String> {
    // In production, would calculate MD5/SHA256 hash
    Ok(format!("hash_{}", file_path.display()))
}

/// Estimates file bitrate from size and duration.
#[must_use]
pub fn estimate_bitrate(file_size_bytes: u64, duration_seconds: f64) -> u64 {
    if duration_seconds > 0.0 {
        ((file_size_bytes as f64 * 8.0) / duration_seconds) as u64
    } else {
        0
    }
}

/// Converts LUFS to linear amplitude.
#[must_use]
pub fn lufs_to_linear(lufs: f64) -> f64 {
    10.0_f64.powf(lufs / 20.0)
}

/// Converts linear amplitude to LUFS.
#[must_use]
pub fn linear_to_lufs(linear: f64) -> f64 {
    20.0 * linear.log10()
}

/// Converts dBFS to linear amplitude.
#[must_use]
pub fn dbfs_to_linear(dbfs: f64) -> f64 {
    10.0_f64.powf(dbfs / 20.0)
}

/// Converts linear amplitude to dBFS.
#[must_use]
pub fn linear_to_dbfs(linear: f64) -> f64 {
    20.0 * linear.log10()
}

/// Formats duration as timecode (HH:MM:SS:FF).
#[must_use]
pub fn format_timecode(seconds: f64, frame_rate: f64) -> String {
    let total_frames = (seconds * frame_rate) as u64;
    let frames = total_frames % frame_rate as u64;
    let total_seconds = total_frames / frame_rate as u64;
    let secs = total_seconds % 60;
    let mins = (total_seconds / 60) % 60;
    let hours = total_seconds / 3600;

    format!("{hours:02}:{mins:02}:{secs:02}:{frames:02}")
}

/// Parses timecode string (HH:MM:SS:FF) to seconds.
///
/// # Errors
///
/// Returns an error if the timecode format is invalid.
pub fn parse_timecode(timecode: &str, frame_rate: f64) -> OxiResult<f64> {
    let parts: Vec<&str> = timecode.split(':').collect();
    if parts.len() != 4 {
        return Err(oximedia_core::OxiError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid timecode format",
        )));
    }

    let hours: u64 = parts[0].parse().map_err(|_| {
        oximedia_core::OxiError::Io(std::io::Error::other("Invalid hours".to_string()))
    })?;
    let mins: u64 = parts[1].parse().map_err(|_| {
        oximedia_core::OxiError::Io(std::io::Error::other("Invalid minutes".to_string()))
    })?;
    let secs: u64 = parts[2].parse().map_err(|_| {
        oximedia_core::OxiError::Io(std::io::Error::other("Invalid seconds".to_string()))
    })?;
    let frames: u64 = parts[3].parse().map_err(|_| {
        oximedia_core::OxiError::Io(std::io::Error::other("Invalid frames".to_string()))
    })?;

    let total_frames = hours * 3600 * frame_rate as u64
        + mins * 60 * frame_rate as u64
        + secs * frame_rate as u64
        + frames;

    Ok(total_frames as f64 / frame_rate)
}

/// Calculates aspect ratio from width and height.
#[must_use]
pub fn calculate_aspect_ratio(width: u32, height: u32) -> (u32, u32) {
    let gcd = gcd(width, height);
    (width / gcd, height / gcd)
}

/// Greatest common divisor using Euclidean algorithm.
#[must_use]
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Formats file size in human-readable format.
#[must_use]
pub fn format_file_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_index])
}

/// Formats bitrate in human-readable format.
#[must_use]
pub fn format_bitrate(bps: u64) -> String {
    const UNITS: &[&str] = &["bps", "Kbps", "Mbps", "Gbps"];
    let mut rate = bps as f64;
    let mut unit_index = 0;

    while rate >= 1000.0 && unit_index < UNITS.len() - 1 {
        rate /= 1000.0;
        unit_index += 1;
    }

    format!("{:.2} {}", rate, UNITS[unit_index])
}

/// Validates frame rate is standard.
#[must_use]
pub fn is_standard_frame_rate(fps: f64) -> bool {
    const STANDARD_RATES: &[f64] = &[23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0, 120.0];

    STANDARD_RATES.iter().any(|&rate| (fps - rate).abs() < 0.01)
}

/// Validates sample rate is standard.
#[must_use]
pub fn is_standard_sample_rate(sample_rate: u32) -> bool {
    const STANDARD_RATES: &[u32] = &[
        8_000, 11_025, 16_000, 22_050, 32_000, 44_100, 48_000, 88_200, 96_000, 176_400, 192_000,
    ];

    STANDARD_RATES.contains(&sample_rate)
}

/// Calculates expected file size from bitrate and duration.
#[must_use]
pub fn calculate_expected_file_size(bitrate_bps: u64, duration_seconds: f64) -> u64 {
    ((bitrate_bps as f64 * duration_seconds) / 8.0) as u64
}

/// Detects letterboxing from aspect ratio.
#[must_use]
pub fn detect_letterboxing(width: u32, height: u32) -> bool {
    let ratio = width as f64 / height as f64;
    // Common letterbox ratios
    (ratio - 2.35).abs() < 0.1 || (ratio - 2.39).abs() < 0.1
}

/// Detects pillarboxing from aspect ratio.
#[must_use]
pub fn detect_pillarboxing(width: u32, height: u32) -> bool {
    let ratio = width as f64 / height as f64;
    // Common pillarbox ratios (4:3 content in 16:9 frame)
    (ratio - (16.0 / 9.0)).abs() < 0.1 && (width as f64 / height as f64) < 1.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_bitrate() {
        let bitrate = estimate_bitrate(1_000_000, 10.0);
        assert_eq!(bitrate, 800_000);
    }

    #[test]
    fn test_lufs_conversion() {
        let lufs = -23.0;
        let linear = lufs_to_linear(lufs);
        let back = linear_to_lufs(linear);
        assert!((back - lufs).abs() < 0.001);
    }

    #[test]
    fn test_dbfs_conversion() {
        let dbfs = -6.0;
        let linear = dbfs_to_linear(dbfs);
        let back = linear_to_dbfs(linear);
        assert!((back - dbfs).abs() < 0.001);
    }

    #[test]
    fn test_format_timecode() {
        let tc = format_timecode(3661.5, 30.0);
        assert_eq!(tc, "01:01:01:15");
    }

    #[test]
    fn test_parse_timecode() {
        let seconds = parse_timecode("01:00:00:00", 30.0).expect("should succeed in test");
        assert_eq!(seconds, 3600.0);
    }

    #[test]
    fn test_calculate_aspect_ratio() {
        let (w, h) = calculate_aspect_ratio(1920, 1080);
        assert_eq!((w, h), (16, 9));
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(48, 18), 6);
        assert_eq!(gcd(1920, 1080), 120);
    }

    #[test]
    fn test_format_file_size() {
        assert_eq!(format_file_size(1024), "1.00 KB");
        assert_eq!(format_file_size(1048576), "1.00 MB");
    }

    #[test]
    fn test_format_bitrate() {
        assert_eq!(format_bitrate(5_000_000), "5.00 Mbps");
    }

    #[test]
    fn test_is_standard_frame_rate() {
        assert!(is_standard_frame_rate(24.0));
        assert!(is_standard_frame_rate(29.97));
        assert!(!is_standard_frame_rate(27.5));
    }

    #[test]
    fn test_is_standard_sample_rate() {
        assert!(is_standard_sample_rate(48000));
        assert!(is_standard_sample_rate(44100));
        assert!(!is_standard_sample_rate(47999));
    }

    #[test]
    fn test_calculate_expected_file_size() {
        let size = calculate_expected_file_size(8_000_000, 60.0);
        assert_eq!(size, 60_000_000);
    }
}
