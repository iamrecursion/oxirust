//! Constants for conform operations.

/// Default pre-roll frames.
pub const DEFAULT_PRE_ROLL_FRAMES: u64 = 10;

/// Default post-roll frames.
pub const DEFAULT_POST_ROLL_FRAMES: u64 = 10;

/// Default match threshold.
pub const DEFAULT_MATCH_THRESHOLD: f64 = 0.8;

/// High quality match threshold.
pub const HIGH_QUALITY_THRESHOLD: f64 = 0.9;

/// Low quality match threshold.
pub const LOW_QUALITY_THRESHOLD: f64 = 0.7;

/// Default fuzzy matching max distance.
pub const DEFAULT_FUZZY_MAX_DISTANCE: usize = 3;

/// Default duration tolerance in seconds.
pub const DEFAULT_DURATION_TOLERANCE: f64 = 0.5;

/// Maximum parallel jobs (0 = auto-detect).
pub const MAX_PARALLEL_JOBS: usize = 0;

/// Default cache TTL in seconds.
pub const DEFAULT_CACHE_TTL_SECS: u64 = 3600;

/// Maximum LRU cache capacity.
pub const MAX_LRU_CACHE_CAPACITY: usize = 1000;

/// Progress update interval in milliseconds.
pub const PROGRESS_UPDATE_INTERVAL_MS: u64 = 100;

/// Default batch size for parallel processing.
pub const DEFAULT_BATCH_SIZE: usize = 10;

/// Common video file extensions.
pub const VIDEO_EXTENSIONS: &[&str] = &[
    "mov", "mp4", "m4v", "avi", "mkv", "mxf", "dpx", "r3d", "arri", "braw",
];

/// Common audio file extensions.
pub const AUDIO_EXTENSIONS: &[&str] = &["wav", "aif", "aiff", "mp3", "aac", "flac"];

/// Common image sequence extensions.
pub const IMAGE_EXTENSIONS: &[&str] = &["dpx", "tiff", "tif", "png", "jpg", "jpeg", "exr"];

/// Standard frame rates.
pub const STANDARD_FRAME_RATES: &[(f64, &str)] = &[
    (23.976, "23.976"),
    (24.0, "24"),
    (25.0, "25"),
    (29.97, "29.97"),
    (30.0, "30"),
    (50.0, "50"),
    (59.94, "59.94"),
    (60.0, "60"),
];

/// Common video resolutions.
pub const COMMON_RESOLUTIONS: &[(u32, u32, &str)] = &[
    (1920, 1080, "1080p"),
    (1280, 720, "720p"),
    (3840, 2160, "4K UHD"),
    (4096, 2160, "4K DCI"),
    (7680, 4320, "8K"),
    (2048, 1080, "2K"),
];

/// EDL comment markers.
pub const EDL_COMMENT_MARKERS: &[&str] = &["*", "//", "#"];

/// Maximum timecode hours.
pub const MAX_TIMECODE_HOURS: u8 = 23;

/// Maximum timecode minutes.
pub const MAX_TIMECODE_MINUTES: u8 = 59;

/// Maximum timecode seconds.
pub const MAX_TIMECODE_SECONDS: u8 = 59;

/// Minimum match score for consideration.
pub const MIN_MATCH_SCORE: f64 = 0.5;

/// Default segment duration for splitting timelines (in seconds).
pub const DEFAULT_SEGMENT_DURATION_SECS: f64 = 600.0; // 10 minutes

/// Buffer size for file I/O operations.
pub const FILE_IO_BUFFER_SIZE: usize = 8192;

/// Maximum filename length.
pub const MAX_FILENAME_LENGTH: usize = 255;

/// Maximum path length.
pub const MAX_PATH_LENGTH: usize = 4096;

/// Default report title.
pub const DEFAULT_REPORT_TITLE: &str = "Conform Report";

/// Version string format.
pub const VERSION_FORMAT: &str = "v{major}.{minor}.{patch}";

/// Database schema version.
pub const DATABASE_SCHEMA_VERSION: i32 = 1;

/// Default session timeout in seconds.
pub const DEFAULT_SESSION_TIMEOUT_SECS: u64 = 3600;

/// Maximum retry attempts for failed operations.
pub const MAX_RETRY_ATTEMPTS: usize = 3;

/// Retry delay in milliseconds.
pub const RETRY_DELAY_MS: u64 = 1000;

/// Default log level.
pub const DEFAULT_LOG_LEVEL: &str = "info";

/// Application name.
pub const APP_NAME: &str = "OxiMedia Conform";

/// Configuration file name.
pub const CONFIG_FILE_NAME: &str = "conform.json";

/// Database file name.
pub const DATABASE_FILE_NAME: &str = "conform.db";

/// Cache directory name.
pub const CACHE_DIR_NAME: &str = ".conform_cache";

/// Temporary directory name.
pub const TEMP_DIR_NAME: &str = ".conform_temp";

/// Report directory name.
pub const REPORT_DIR_NAME: &str = "reports";

/// Maximum ambiguous match candidates to show.
pub const MAX_AMBIGUOUS_CANDIDATES: usize = 10;

/// Default HTML report CSS class prefix.
pub const HTML_CLASS_PREFIX: &str = "conform-";

/// CSV delimiter.
pub const CSV_DELIMITER: char = ',';

/// TSV delimiter.
pub const TSV_DELIMITER: char = '\t';

/// Newline character for reports.
pub const NEWLINE: &str = "\n";

/// Tab character for formatting.
pub const TAB: &str = "\t";

/// Default number of progress bar segments.
pub const PROGRESS_BAR_SEGMENTS: usize = 40;

/// ETA calculation window (number of samples).
pub const ETA_WINDOW_SIZE: usize = 10;

/// Bytes per kilobyte.
pub const BYTES_PER_KB: u64 = 1024;

/// Bytes per megabyte.
pub const BYTES_PER_MB: u64 = BYTES_PER_KB * 1024;

/// Bytes per gigabyte.
pub const BYTES_PER_GB: u64 = BYTES_PER_MB * 1024;

/// Bytes per terabyte.
pub const BYTES_PER_TB: u64 = BYTES_PER_GB * 1024;

/// Seconds per minute.
pub const SECS_PER_MINUTE: u64 = 60;

/// Seconds per hour.
pub const SECS_PER_HOUR: u64 = SECS_PER_MINUTE * 60;

/// Seconds per day.
pub const SECS_PER_DAY: u64 = SECS_PER_HOUR * 24;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_PRE_ROLL_FRAMES, 10);
        assert_eq!(DEFAULT_POST_ROLL_FRAMES, 10);
        assert!((DEFAULT_MATCH_THRESHOLD - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_file_extensions() {
        assert!(VIDEO_EXTENSIONS.contains(&"mov"));
        assert!(AUDIO_EXTENSIONS.contains(&"wav"));
        assert!(IMAGE_EXTENSIONS.contains(&"dpx"));
    }

    #[test]
    fn test_byte_constants() {
        assert_eq!(BYTES_PER_KB, 1024);
        assert_eq!(BYTES_PER_MB, 1024 * 1024);
        assert_eq!(BYTES_PER_GB, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_time_constants() {
        assert_eq!(SECS_PER_MINUTE, 60);
        assert_eq!(SECS_PER_HOUR, 3600);
        assert_eq!(SECS_PER_DAY, 86400);
    }
}
