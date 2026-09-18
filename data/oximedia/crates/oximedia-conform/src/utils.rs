//! Utility functions for conforming operations.

use crate::error::{ConformError, ConformResult};
use crate::types::{FrameRate, Timecode};
use std::path::{Path, PathBuf};

/// Convert seconds to timecode.
#[must_use]
pub fn seconds_to_timecode(seconds: f64, fps: FrameRate) -> Timecode {
    let total_frames = (seconds * fps.as_f64()) as u64;
    Timecode::from_frames(total_frames, fps)
}

/// Convert timecode to seconds.
#[must_use]
pub fn timecode_to_seconds(tc: Timecode, fps: FrameRate) -> f64 {
    tc.to_frames(fps) as f64 / fps.as_f64()
}

/// Format duration in seconds as human-readable string.
#[must_use]
pub fn format_duration(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{seconds:.1}s")
    } else if seconds < 3600.0 {
        let minutes = (seconds / 60.0) as u32;
        let secs = seconds % 60.0;
        format!("{minutes}m {secs:.0}s")
    } else {
        let hours = (seconds / 3600.0) as u32;
        let minutes = ((seconds % 3600.0) / 60.0) as u32;
        format!("{hours}h {minutes}m")
    }
}

/// Format file size as human-readable string.
#[must_use]
pub fn format_file_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes < KB {
        format!("{bytes} B")
    } else if bytes < MB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else if bytes < GB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes < TB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    }
}

/// Sanitize filename for safe use.
#[must_use]
pub fn sanitize_filename(filename: &str) -> String {
    filename
        .replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_")
        .trim()
        .to_string()
}

/// Get file extension.
#[must_use]
pub fn get_extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
}

/// Change file extension.
#[must_use]
pub fn change_extension(path: &Path, new_ext: &str) -> PathBuf {
    let mut result = path.to_path_buf();
    result.set_extension(new_ext);
    result
}

/// Get stem (filename without extension).
#[must_use]
pub fn get_stem(path: &Path) -> Option<String> {
    path.file_stem().and_then(|s| s.to_str()).map(String::from)
}

/// Join paths safely.
///
/// # Errors
///
/// Returns an error if the path cannot be constructed.
pub fn safe_join<P: AsRef<Path>, Q: AsRef<Path>>(base: P, path: Q) -> ConformResult<PathBuf> {
    let base = base.as_ref();
    let path = path.as_ref();

    // Check if path is absolute
    if path.is_absolute() {
        return Err(ConformError::InvalidConfig(
            "Cannot join with absolute path".to_string(),
        ));
    }

    // Check for path traversal
    let joined = base.join(path);
    if !joined.starts_with(base) {
        return Err(ConformError::InvalidConfig(
            "Path traversal detected".to_string(),
        ));
    }

    Ok(joined)
}

/// Ensure directory exists.
///
/// # Errors
///
/// Returns an error if the directory cannot be created.
pub fn ensure_dir_exists<P: AsRef<Path>>(path: P) -> ConformResult<()> {
    if !path.as_ref().exists() {
        std::fs::create_dir_all(path)?;
    }
    Ok(())
}

/// Get relative path from base to target.
pub fn get_relative_path<P: AsRef<Path>, Q: AsRef<Path>>(base: P, target: Q) -> Option<PathBuf> {
    pathdiff::diff_paths(target, base)
}

/// Calculate frame count from duration and frame rate.
#[must_use]
pub fn duration_to_frames(duration_seconds: f64, fps: FrameRate) -> u64 {
    (duration_seconds * fps.as_f64()).round() as u64
}

/// Calculate duration from frame count and frame rate.
#[must_use]
pub fn frames_to_duration(frames: u64, fps: FrameRate) -> f64 {
    frames as f64 / fps.as_f64()
}

/// Round to nearest frame boundary.
#[must_use]
pub fn round_to_frame(seconds: f64, fps: FrameRate) -> f64 {
    let frames = (seconds * fps.as_f64()).round();
    frames / fps.as_f64()
}

/// Check if a path is safe (no traversal, no absolute).
#[must_use]
pub fn is_safe_path(path: &Path) -> bool {
    !path.is_absolute()
        && !path
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
}

/// Truncate string to maximum length with ellipsis.
#[must_use]
pub fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len < 3 {
        s[..max_len].to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Generate unique filename by appending number if file exists.
#[must_use]
pub fn generate_unique_filename(base_path: &Path) -> PathBuf {
    if !base_path.exists() {
        return base_path.to_path_buf();
    }

    let stem = get_stem(base_path).unwrap_or_else(|| "file".to_string());
    let ext = get_extension(base_path);
    let parent = base_path.parent().unwrap_or_else(|| Path::new("."));

    for i in 1..10000 {
        let new_name = if let Some(ref e) = ext {
            format!("{stem}_{i}.{e}")
        } else {
            format!("{stem}_{i}")
        };

        let new_path = parent.join(new_name);
        if !new_path.exists() {
            return new_path;
        }
    }

    // Fallback: use UUID
    let uuid = uuid::Uuid::new_v4();
    let new_name = if let Some(ref e) = ext {
        format!("{stem}_{uuid}.{e}")
    } else {
        format!("{stem}_{uuid}")
    };
    parent.join(new_name)
}

/// Calculate edit distance between two strings (Levenshtein).
#[must_use]
pub fn edit_distance(a: &str, b: &str) -> usize {
    strsim::levenshtein(a, b)
}

/// Calculate similarity score between two strings (0.0 - 1.0).
#[must_use]
pub fn similarity_score(a: &str, b: &str) -> f64 {
    strsim::jaro_winkler(a, b)
}

/// Parse timecode from string.
///
/// # Errors
///
/// Returns an error if the timecode format is invalid.
pub fn parse_timecode(s: &str) -> ConformResult<Timecode> {
    Timecode::parse(s).map_err(ConformError::InvalidTimecode)
}

/// Format timecode with drop-frame indicator.
#[must_use]
pub fn format_timecode_df(tc: Timecode, drop_frame: bool) -> String {
    let separator = if drop_frame { ';' } else { ':' };
    format!(
        "{:02}:{:02}:{:02}{}{:02}",
        tc.hours, tc.minutes, tc.seconds, separator, tc.frames
    )
}

/// Calculate checksum for a byte slice using xxHash.
#[must_use]
pub fn quick_checksum(data: &[u8]) -> u64 {
    use xxhash_rust::xxh3::xxh3_64;
    xxh3_64(data)
}

/// Validate frame rate value.
pub fn validate_frame_rate(fps: f64) -> ConformResult<()> {
    if fps <= 0.0 || fps > 1000.0 {
        return Err(ConformError::InvalidFrameRate(format!(
            "Invalid frame rate: {fps}"
        )));
    }
    Ok(())
}

/// Round frame rate to standard value.
#[must_use]
pub fn round_to_standard_fps(fps: f64) -> FrameRate {
    const STANDARD_RATES: &[(f64, FrameRate)] = &[
        (23.976, FrameRate::Fps23976),
        (24.0, FrameRate::Fps24),
        (25.0, FrameRate::Fps25),
        (29.97, FrameRate::Fps2997NDF),
        (30.0, FrameRate::Fps30),
        (50.0, FrameRate::Fps50),
        (59.94, FrameRate::Fps5994),
        (60.0, FrameRate::Fps60),
    ];

    let mut closest = FrameRate::Fps25;
    let mut min_diff = f64::MAX;

    for &(rate, fps_enum) in STANDARD_RATES {
        let diff = (fps - rate).abs();
        if diff < min_diff {
            min_diff = diff;
            closest = fps_enum;
        }
    }

    if min_diff > 0.5 {
        FrameRate::Custom(fps)
    } else {
        closest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seconds_to_timecode() {
        let tc = seconds_to_timecode(3661.0, FrameRate::Fps25); // 1 hour, 1 minute, 1 second
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 1);
        assert_eq!(tc.seconds, 1);
    }

    #[test]
    fn test_timecode_to_seconds() {
        let tc = Timecode::new(1, 0, 0, 0);
        let secs = timecode_to_seconds(tc, FrameRate::Fps25);
        assert!((secs - 3600.0).abs() < 0.1);
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30.0), "30.0s");
        assert_eq!(format_duration(90.0), "1m 30s");
        assert_eq!(format_duration(3660.0), "1h 1m");
    }

    #[test]
    fn test_format_file_size() {
        assert_eq!(format_file_size(512), "512 B");
        assert_eq!(format_file_size(1024), "1.00 KB");
        assert_eq!(format_file_size(1024 * 1024), "1.00 MB");
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("file:name?.txt"), "file_name_.txt");
        assert_eq!(sanitize_filename("test/path\\file"), "test_path_file");
    }

    #[test]
    fn test_get_extension() {
        assert_eq!(
            get_extension(Path::new("file.txt")),
            Some("txt".to_string())
        );
        assert_eq!(get_extension(Path::new("file")), None);
    }

    #[test]
    fn test_change_extension() {
        let path = Path::new("file.txt");
        let new_path = change_extension(path, "md");
        assert_eq!(new_path, PathBuf::from("file.md"));
    }

    #[test]
    fn test_is_safe_path() {
        assert!(is_safe_path(Path::new("subdir/file.txt")));
        assert!(!is_safe_path(Path::new("/absolute/path")));
        assert!(!is_safe_path(Path::new("../parent/path")));
    }

    #[test]
    fn test_truncate_string() {
        assert_eq!(truncate_string("hello world", 5), "he...");
        assert_eq!(truncate_string("hello", 10), "hello");
    }

    #[test]
    fn test_edit_distance() {
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("test", "test"), 0);
    }

    #[test]
    fn test_similarity_score() {
        let score = similarity_score("hello", "hallo");
        assert!(score > 0.8);
        let score = similarity_score("abc", "xyz");
        assert!(score < 0.5);
    }

    #[test]
    fn test_round_to_standard_fps() {
        assert_eq!(round_to_standard_fps(24.0), FrameRate::Fps24);
        assert_eq!(round_to_standard_fps(29.97), FrameRate::Fps2997NDF);
        assert_eq!(round_to_standard_fps(25.1), FrameRate::Fps25);
        match round_to_standard_fps(120.0) {
            FrameRate::Custom(fps) => assert!((fps - 120.0).abs() < 0.01),
            _ => panic!("Should be custom"),
        }
    }

    #[test]
    fn test_duration_to_frames() {
        let frames = duration_to_frames(1.0, FrameRate::Fps25);
        assert_eq!(frames, 25);
    }

    #[test]
    fn test_frames_to_duration() {
        let duration = frames_to_duration(25, FrameRate::Fps25);
        assert!((duration - 1.0).abs() < f64::EPSILON);
    }
}
