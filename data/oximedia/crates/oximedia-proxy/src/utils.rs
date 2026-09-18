//! Utility functions for proxy workflows.

use crate::Result;
use std::path::{Path, PathBuf};

/// File utilities for proxy operations.
pub struct FileUtils;

impl FileUtils {
    /// Calculate the SHA-256 hash of a file for verification.
    ///
    /// Reads the file in 8 KiB chunks so that arbitrarily large files can be
    /// hashed without loading them entirely into memory. The result is the
    /// lowercase hex encoding of the 32-byte digest.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or read.
    pub fn calculate_hash(path: &Path) -> Result<String> {
        use sha2::{Digest, Sha256};
        use std::io::{BufReader, Read};

        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];

        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }

        let digest = hasher.finalize();
        Ok(StringUtils::bytes_to_hex(digest.as_slice()))
    }

    /// Get file size in human-readable format.
    #[must_use]
    pub fn format_file_size(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB", "PB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        format!("{:.2} {}", size, UNITS[unit_index])
    }

    /// Get file modification time as Unix timestamp.
    ///
    /// # Errors
    ///
    /// Returns an error if the file metadata cannot be read.
    pub fn get_modification_time(path: &Path) -> Result<i64> {
        let metadata = std::fs::metadata(path)?;
        let modified = metadata.modified()?;
        let duration = modified
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| {
                crate::ProxyError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e))
            })?;

        Ok(duration.as_secs() as i64)
    }

    /// Copy file with progress tracking.
    pub fn copy_with_progress<F>(
        source: &Path,
        destination: &Path,
        mut progress_callback: F,
    ) -> Result<u64>
    where
        F: FnMut(u64, u64),
    {
        let total_size = std::fs::metadata(source)?.len();
        let mut copied = 0u64;

        // Placeholder: would implement chunked copying with progress
        progress_callback(copied, total_size);

        std::fs::copy(source, destination)?;
        copied = total_size;
        progress_callback(copied, total_size);

        Ok(copied)
    }

    /// Create directory if it doesn't exist.
    pub fn ensure_directory(path: &Path) -> Result<()> {
        if !path.exists() {
            std::fs::create_dir_all(path)?;
        }
        Ok(())
    }

    /// Get available disk space for a path.
    ///
    /// # Errors
    ///
    /// Returns an error if disk space cannot be determined.
    pub fn get_available_space(_path: &Path) -> Result<u64> {
        // Placeholder: would use platform-specific APIs
        Ok(0)
    }

    /// Check if file is a video file based on extension.
    #[must_use]
    pub fn is_video_file(path: &Path) -> bool {
        const VIDEO_EXTENSIONS: &[&str] = &[
            "mp4", "mov", "avi", "mkv", "wmv", "flv", "webm", "m4v", "mpg", "mpeg", "3gp", "3g2",
            "mxf", "ts", "m2ts", "mts",
        ];

        if let Some(ext) = path.extension() {
            if let Some(ext_str) = ext.to_str() {
                return VIDEO_EXTENSIONS.contains(&ext_str.to_lowercase().as_str());
            }
        }

        false
    }

    /// Get codec name from file extension.
    #[must_use]
    pub fn guess_codec_from_extension(path: &Path) -> Option<String> {
        if let Some(ext) = path.extension() {
            match ext.to_str()?.to_lowercase().as_str() {
                "mp4" | "m4v" => Some("h264".to_string()),
                "webm" => Some("vp9".to_string()),
                "mkv" => Some("h264".to_string()),
                "mov" => Some("prores".to_string()),
                "avi" => Some("h264".to_string()),
                _ => None,
            }
        } else {
            None
        }
    }
}

/// Path utilities for proxy workflows.
pub struct PathUtils;

impl PathUtils {
    /// Normalize path separators for cross-platform compatibility.
    #[must_use]
    pub fn normalize_path(path: &Path) -> PathBuf {
        // Convert to canonical form if possible
        if let Ok(canonical) = path.canonicalize() {
            canonical
        } else {
            path.to_path_buf()
        }
    }

    /// Get relative path from base to target.
    ///
    /// # Errors
    ///
    /// Returns an error if the paths don't share a common base.
    pub fn get_relative_path(base: &Path, target: &Path) -> Result<PathBuf> {
        target
            .strip_prefix(base)
            .map(|p| p.to_path_buf())
            .map_err(|e| crate::ProxyError::InvalidInput(e.to_string()))
    }

    /// Make path absolute.
    #[must_use]
    pub fn make_absolute(path: &Path) -> PathBuf {
        if path.is_absolute() {
            path.to_path_buf()
        } else if let Ok(current_dir) = std::env::current_dir() {
            current_dir.join(path)
        } else {
            path.to_path_buf()
        }
    }

    /// Get filename without extension.
    #[must_use]
    pub fn get_stem(path: &Path) -> String {
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string()
    }

    /// Change file extension.
    #[must_use]
    pub fn change_extension(path: &Path, new_ext: &str) -> PathBuf {
        let mut new_path = path.to_path_buf();
        new_path.set_extension(new_ext);
        new_path
    }

    /// Generate unique filename by appending number.
    pub fn make_unique_filename(path: &Path) -> PathBuf {
        if !path.exists() {
            return path.to_path_buf();
        }

        let stem = Self::get_stem(path);
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let parent = path.parent().unwrap_or_else(|| Path::new("."));

        for i in 1..1000 {
            let new_filename = if ext.is_empty() {
                format!("{}_{}", stem, i)
            } else {
                format!("{}_{}.{}", stem, i, ext)
            };

            let new_path = parent.join(new_filename);
            if !new_path.exists() {
                return new_path;
            }
        }

        path.to_path_buf()
    }

    /// Check if two paths point to the same file.
    #[must_use]
    pub fn are_same_file(path1: &Path, path2: &Path) -> bool {
        if let (Ok(canon1), Ok(canon2)) = (path1.canonicalize(), path2.canonicalize()) {
            canon1 == canon2
        } else {
            path1 == path2
        }
    }
}

/// Time utilities for proxy workflows.
pub struct TimeUtils;

impl TimeUtils {
    /// Format duration in seconds to human-readable string.
    #[must_use]
    pub fn format_duration(seconds: f64) -> String {
        let hours = (seconds / 3600.0).floor() as u64;
        let minutes = ((seconds % 3600.0) / 60.0).floor() as u64;
        let secs = (seconds % 60.0).floor() as u64;
        let ms = ((seconds % 1.0) * 1000.0).floor() as u64;

        if hours > 0 {
            format!("{}:{:02}:{:02}.{:03}", hours, minutes, secs, ms)
        } else if minutes > 0 {
            format!("{}:{:02}.{:03}", minutes, secs, ms)
        } else {
            format!("{}.{:03}s", secs, ms)
        }
    }

    /// Parse duration string to seconds.
    ///
    /// # Errors
    ///
    /// Returns an error if the format is invalid.
    pub fn parse_duration(duration_str: &str) -> Result<f64> {
        // Simple parser for formats like "01:23:45" or "1h 23m 45s"
        let parts: Vec<&str> = duration_str.split(':').collect();

        match parts.len() {
            3 => {
                // HH:MM:SS format
                let hours: f64 = parts[0].parse().map_err(|e| {
                    crate::ProxyError::InvalidInput(format!("Invalid hours: {}", e))
                })?;
                let minutes: f64 = parts[1].parse().map_err(|e| {
                    crate::ProxyError::InvalidInput(format!("Invalid minutes: {}", e))
                })?;
                let seconds: f64 = parts[2].parse().map_err(|e| {
                    crate::ProxyError::InvalidInput(format!("Invalid seconds: {}", e))
                })?;

                Ok(hours * 3600.0 + minutes * 60.0 + seconds)
            }
            2 => {
                // MM:SS format
                let minutes: f64 = parts[0].parse().map_err(|e| {
                    crate::ProxyError::InvalidInput(format!("Invalid minutes: {}", e))
                })?;
                let seconds: f64 = parts[1].parse().map_err(|e| {
                    crate::ProxyError::InvalidInput(format!("Invalid seconds: {}", e))
                })?;

                Ok(minutes * 60.0 + seconds)
            }
            1 => {
                // Just seconds
                parts[0].parse().map_err(|e| {
                    crate::ProxyError::InvalidInput(format!("Invalid duration: {}", e))
                })
            }
            _ => Err(crate::ProxyError::InvalidInput(
                "Invalid duration format".to_string(),
            )),
        }
    }

    /// Get current Unix timestamp.
    #[must_use]
    pub fn current_timestamp() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("infallible: system clock is always after UNIX_EPOCH")
            .as_secs() as i64
    }

    /// Format timestamp to ISO 8601 string.
    #[must_use]
    pub fn format_timestamp(_timestamp: i64) -> String {
        // Placeholder: would format to ISO 8601
        "2024-01-01T00:00:00Z".to_string()
    }
}

/// String utilities for proxy workflows.
pub struct StringUtils;

impl StringUtils {
    /// Sanitize filename by removing invalid characters.
    #[must_use]
    pub fn sanitize_filename(filename: &str) -> String {
        const INVALID_CHARS: &[char] = &['/', '\\', ':', '*', '?', '"', '<', '>', '|'];

        filename
            .chars()
            .map(|c| if INVALID_CHARS.contains(&c) { '_' } else { c })
            .collect()
    }

    /// Generate a unique identifier.
    #[must_use]
    pub fn generate_id() -> String {
        use std::time::SystemTime;

        let now = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("infallible: system clock is always after UNIX_EPOCH");

        format!("{:x}", now.as_nanos())
    }

    /// Truncate string to maximum length.
    #[must_use]
    pub fn truncate(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else {
            format!("{}...", &s[..max_len.saturating_sub(3)])
        }
    }

    /// Convert bytes to hex string.
    #[must_use]
    pub fn bytes_to_hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Parse hex string to bytes.
    pub fn hex_to_bytes(hex: &str) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        let mut chars = hex.chars();

        while let (Some(high), Some(low)) = (chars.next(), chars.next()) {
            let byte = u8::from_str_radix(&format!("{}{}", high, low), 16)
                .map_err(|e| crate::ProxyError::InvalidInput(e.to_string()))?;
            bytes.push(byte);
        }

        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_file_size() {
        assert_eq!(FileUtils::format_file_size(500), "500.00 B");
        assert_eq!(FileUtils::format_file_size(1024), "1.00 KB");
        assert_eq!(FileUtils::format_file_size(1_048_576), "1.00 MB");
        assert_eq!(FileUtils::format_file_size(1_073_741_824), "1.00 GB");
    }

    #[test]
    fn test_is_video_file() {
        assert!(FileUtils::is_video_file(Path::new("test.mp4")));
        assert!(FileUtils::is_video_file(Path::new("test.mov")));
        assert!(FileUtils::is_video_file(Path::new("test.mkv")));
        assert!(!FileUtils::is_video_file(Path::new("test.txt")));
    }

    #[test]
    fn test_guess_codec() {
        assert_eq!(
            FileUtils::guess_codec_from_extension(Path::new("test.mp4")),
            Some("h264".to_string())
        );
        assert_eq!(
            FileUtils::guess_codec_from_extension(Path::new("test.webm")),
            Some("vp9".to_string())
        );
    }

    #[test]
    fn test_get_stem() {
        assert_eq!(PathUtils::get_stem(Path::new("test.mp4")), "test");
        assert_eq!(PathUtils::get_stem(Path::new("/path/to/file.ext")), "file");
    }

    #[test]
    fn test_change_extension() {
        let path = Path::new("test.mp4");
        let new_path = PathUtils::change_extension(path, "mov");
        assert_eq!(new_path, PathBuf::from("test.mov"));
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(TimeUtils::format_duration(30.5), "30.500s");
        assert_eq!(TimeUtils::format_duration(90.0), "1:30.000");
        assert_eq!(TimeUtils::format_duration(3665.0), "1:01:05.000");
    }

    #[test]
    fn test_parse_duration() {
        assert_eq!(
            TimeUtils::parse_duration("30").expect("should succeed in test"),
            30.0
        );
        assert_eq!(
            TimeUtils::parse_duration("1:30").expect("should succeed in test"),
            90.0
        );
        assert_eq!(
            TimeUtils::parse_duration("1:01:05").expect("should succeed in test"),
            3665.0
        );
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(
            StringUtils::sanitize_filename("test/file:name.mp4"),
            "test_file_name.mp4"
        );
    }

    #[test]
    fn test_truncate() {
        assert_eq!(StringUtils::truncate("hello", 10), "hello");
        assert_eq!(StringUtils::truncate("hello world", 8), "hello...");
    }

    #[test]
    fn test_bytes_to_hex() {
        assert_eq!(StringUtils::bytes_to_hex(&[0, 1, 255]), "0001ff");
    }

    #[test]
    fn test_hex_to_bytes() {
        let bytes = StringUtils::hex_to_bytes("0001ff").expect("should succeed in test");
        assert_eq!(bytes, vec![0, 1, 255]);
    }

    #[test]
    fn test_current_timestamp() {
        let ts = TimeUtils::current_timestamp();
        assert!(ts > 0);
    }

    #[test]
    fn test_calculate_hash_known_digest() {
        use std::io::Write;

        // SHA-256("abc") per FIPS 180-4 test vector.
        let expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

        let mut file_path = std::env::temp_dir();
        file_path.push(format!(
            "oximedia_proxy_calculate_hash_{}.bin",
            StringUtils::generate_id()
        ));

        {
            let mut file = std::fs::File::create(&file_path).expect("create temp file");
            file.write_all(b"abc").expect("write temp file");
        }

        let hash = FileUtils::calculate_hash(&file_path).expect("hash temp file");
        let _ = std::fs::remove_file(&file_path);

        assert_eq!(hash, expected);
    }

    #[test]
    fn test_calculate_hash_empty_file() {
        // SHA-256 of zero-length input.
        let expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

        let mut file_path = std::env::temp_dir();
        file_path.push(format!(
            "oximedia_proxy_calculate_hash_empty_{}.bin",
            StringUtils::generate_id()
        ));

        {
            let _file = std::fs::File::create(&file_path).expect("create temp file");
        }

        let hash = FileUtils::calculate_hash(&file_path).expect("hash empty file");
        let _ = std::fs::remove_file(&file_path);

        assert_eq!(hash, expected);
    }

    #[test]
    fn test_calculate_hash_missing_file() {
        let mut file_path = std::env::temp_dir();
        file_path.push(format!(
            "oximedia_proxy_calculate_hash_missing_{}.bin",
            StringUtils::generate_id()
        ));

        assert!(FileUtils::calculate_hash(&file_path).is_err());
    }
}
