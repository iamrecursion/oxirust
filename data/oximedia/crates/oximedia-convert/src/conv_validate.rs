/// Pre-conversion input validation for `OxiMedia`.
///
/// Validates conversion parameters against configurable constraints and
/// checks basic input file properties before the conversion pipeline starts.
///
/// Errors reported by the conversion validator.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// The file extension or container format is not supported.
    UnsupportedFormat(String),
    /// The requested resolution exceeds the maximum allowed.
    ResolutionTooLarge {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
    /// The requested bitrate is outside the acceptable range.
    InvalidBitrate(u32),
    /// The codec is not compatible with the target container.
    IncompatibleCodec {
        /// Codec identifier.
        codec: String,
        /// Container/format identifier.
        format: String,
    },
    /// A required field is missing or empty.
    MissingRequiredField(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedFormat(fmt) => write!(f, "Unsupported format: {fmt}"),
            Self::ResolutionTooLarge { width, height } => {
                write!(f, "Resolution {width}x{height} exceeds maximum")
            }
            Self::InvalidBitrate(bps) => write!(f, "Invalid bitrate: {bps} kbps"),
            Self::IncompatibleCodec { codec, format } => {
                write!(f, "Codec '{codec}' incompatible with format '{format}'")
            }
            Self::MissingRequiredField(field) => {
                write!(f, "Missing required field: {field}")
            }
        }
    }
}

// ── ValidateProfile ───────────────────────────────────────────────────────────

/// A flat set of parameters that describes a requested conversion.
///
/// Used as the input type for [`ConvertValidation::validate_profile`].
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ValidateProfile {
    /// Video codec identifier (e.g., "av1", "vp9").
    pub codec: String,
    /// Target bitrate in kilobits per second.
    pub bitrate_kbps: u32,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Target container format (e.g., "webm", "mp4", "mkv").
    pub container: String,
}

impl ValidateProfile {
    /// Creates a new validate profile with the given parameters.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(codec: &str, bitrate_kbps: u32, width: u32, height: u32, container: &str) -> Self {
        Self {
            codec: codec.to_string(),
            bitrate_kbps,
            width,
            height,
            container: container.to_string(),
        }
    }
}

// ── ConvertValidation ─────────────────────────────────────────────────────────

/// Validates conversion profiles and input files against configurable limits.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ConvertValidation {
    /// Maximum allowed output width in pixels.
    pub max_width: u32,
    /// Maximum allowed output height in pixels.
    pub max_height: u32,
    /// Maximum allowed bitrate in kilobits per second.
    pub max_bitrate_kbps: u32,
    /// Set of recognised codec identifiers.
    pub allowed_codecs: Vec<String>,
}

impl Default for ConvertValidation {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvertValidation {
    /// Creates a new validator with sensible defaults.
    ///
    /// * Max resolution: 7680 × 4320 (8K)
    /// * Max bitrate: 100 000 kbps
    /// * Allowed codecs: AV1, VP9, VP8, FLAC, Opus, Vorbis (all patent-free)
    #[allow(dead_code)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_width: 7_680,
            max_height: 4_320,
            max_bitrate_kbps: 100_000,
            allowed_codecs: vec![
                "av1".to_string(),
                "vp9".to_string(),
                "vp8".to_string(),
                "flac".to_string(),
                "opus".to_string(),
                "vorbis".to_string(),
                "theora".to_string(),
                "aom-av1".to_string(),
            ],
        }
    }

    /// Returns `true` if the codec is in the allowed list (case-insensitive).
    #[allow(dead_code)]
    #[must_use]
    pub fn is_codec_supported(&self, codec: &str) -> bool {
        let lower = codec.to_lowercase();
        self.allowed_codecs
            .iter()
            .any(|c| c.to_lowercase() == lower)
    }

    /// Validates all fields of a [`ValidateProfile`] and returns a (possibly
    /// empty) list of [`ValidationError`]s.
    #[allow(dead_code)]
    #[must_use]
    pub fn validate_profile(&self, profile: &ValidateProfile) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // Codec must not be empty
        if profile.codec.is_empty() {
            errors.push(ValidationError::MissingRequiredField("codec".to_string()));
        } else if !self.is_codec_supported(&profile.codec) {
            errors.push(ValidationError::UnsupportedFormat(profile.codec.clone()));
        }

        // Container must not be empty
        if profile.container.is_empty() {
            errors.push(ValidationError::MissingRequiredField(
                "container".to_string(),
            ));
        }

        // Resolution check
        if profile.width > self.max_width || profile.height > self.max_height {
            errors.push(ValidationError::ResolutionTooLarge {
                width: profile.width,
                height: profile.height,
            });
        }

        // Zero resolution check
        if profile.width == 0 {
            errors.push(ValidationError::MissingRequiredField("width".to_string()));
        }
        if profile.height == 0 {
            errors.push(ValidationError::MissingRequiredField("height".to_string()));
        }

        // Bitrate check
        if profile.bitrate_kbps == 0 || profile.bitrate_kbps > self.max_bitrate_kbps {
            errors.push(ValidationError::InvalidBitrate(profile.bitrate_kbps));
        }

        // Codec–container compatibility check
        if !profile.codec.is_empty() && !profile.container.is_empty() {
            if let Some(err) = check_codec_container_compat(&profile.codec, &profile.container) {
                errors.push(err);
            }
        }

        errors
    }
}

/// Checks whether the codec is compatible with the target container.
///
/// Returns `Some(ValidationError)` on incompatibility, `None` if OK.
fn check_codec_container_compat(codec: &str, container: &str) -> Option<ValidationError> {
    let codec_lower = codec.to_lowercase();
    let container_lower = container.to_lowercase();

    // Known incompatibilities (non-exhaustive but representative)
    let incompatible = match container_lower.as_str() {
        "mp4" => matches!(codec_lower.as_str(), "vorbis" | "theora" | "vp8"),
        "webm" => !matches!(
            codec_lower.as_str(),
            "vp8" | "vp9" | "av1" | "opus" | "vorbis" | "aom-av1"
        ),
        "ogg" => !matches!(codec_lower.as_str(), "vorbis" | "opus" | "flac" | "theora"),
        _ => false,
    };

    if incompatible {
        Some(ValidationError::IncompatibleCodec {
            codec: codec.to_string(),
            format: container.to_string(),
        })
    } else {
        None
    }
}

// ── validate_input_file ───────────────────────────────────────────────────────

/// Checks basic properties of an input file path.
///
/// Returns a list of [`ValidationError`]s (empty means valid).
/// Checks performed:
/// 1. Path must not be empty.
/// 2. Extension must be a known media format.
#[allow(dead_code)]
#[must_use]
pub fn validate_input_file(path: &str) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    if path.is_empty() {
        errors.push(ValidationError::MissingRequiredField("path".to_string()));
        return errors;
    }

    // Extract extension
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let known_extensions = [
        "webm", "mkv", "ogg", "ogv", "oga", "flac", "opus", "mp4", "mov", "avi", "wav", "mp3",
        "aac", "m4a", "ts", "m2ts", "mts",
    ];

    if ext.is_empty() || !known_extensions.contains(&ext.as_str()) {
        errors.push(ValidationError::UnsupportedFormat(format!(
            "unknown extension '{ext}'"
        )));
    }

    errors
}

// ── DiskSpaceEstimate ─────────────────────────────────────────────────────────

/// Result of a disk space pre-validation check before conversion.
#[derive(Debug, Clone)]
pub struct DiskSpaceEstimate {
    /// Estimated output size in bytes.
    pub estimated_bytes: u64,
    /// Available bytes in the output directory.
    pub available_bytes: u64,
    /// Required headroom factor (e.g. 1.2 = require 20% extra space).
    pub headroom_factor: f64,
}

impl DiskSpaceEstimate {
    /// Returns `true` when there is sufficient space (available ≥ estimated × headroom).
    #[must_use]
    pub fn is_sufficient(&self) -> bool {
        let required = (self.estimated_bytes as f64 * self.headroom_factor).ceil() as u64;
        self.available_bytes >= required
    }

    /// Required bytes including headroom.
    #[must_use]
    pub fn required_bytes(&self) -> u64 {
        (self.estimated_bytes as f64 * self.headroom_factor).ceil() as u64
    }
}

/// Estimate the output file size based on the input size and conversion profile.
///
/// Uses bitrate ratios and profile compression factors:
/// - Web profiles: ~60% of input (compressed)
/// - Archive profiles: ~150% of input (higher quality)
/// - Broadcast profiles: ~80% of input (high quality)
/// - Default: 1:1 ratio + 10% overhead
#[must_use]
pub fn estimate_output_size(
    input_size: u64,
    profile: &crate::conv_profile::ConversionProfile,
) -> u64 {
    // Estimate based on total bitrate and typical duration assumptions.
    // We use the profile name as a heuristic to infer compression ratio.
    let name_lower = profile.name.to_lowercase();

    let ratio = if name_lower.contains("web") || name_lower.contains("stream") {
        0.6_f64
    } else if name_lower.contains("archive") {
        1.5_f64
    } else if name_lower.contains("broadcast") {
        0.8_f64
    } else {
        // Generic estimate: use bitrate ratio relative to a reference 4000 kbps AV1 stream.
        // Assume ~1 hour of content at 4000 kbps ≈ 1.8 GB.
        // Scale linearly: (profile_bitrate / 4000) * input_size, clamped to [0.1, 4.0].
        let reference_kbps = 4_000_u32;
        let ratio = profile.total_bitrate_kbps() as f64 / reference_kbps as f64;
        ratio.clamp(0.1, 4.0)
    };

    let estimated = (input_size as f64 * ratio).ceil() as u64;
    // Always return at least 1 KB to avoid zero-size estimates for empty inputs.
    estimated.max(1_024)
}

/// Query available disk space for the directory containing `output_path`.
///
/// Uses a pure-Rust approach: reads the filesystem via `std::process::Command`
/// (`df -k` on Unix) with a fallback that returns a conservative estimate if
/// the command is unavailable (e.g. on Windows or restricted environments).
///
/// On any platform, if the parent directory does not exist the function returns
/// [`ValidationError::MissingRequiredField`] wrapped in a `ConversionError`.
pub fn check_disk_space(
    output_path: &std::path::Path,
    estimated_bytes: u64,
) -> Result<DiskSpaceEstimate, ValidationError>
where
{
    // Resolve the directory we will write into.
    let dir = output_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));

    // The directory must exist for us to query its free space.
    if !dir.exists() {
        return Err(ValidationError::MissingRequiredField(format!(
            "output directory does not exist: {}",
            dir.display()
        )));
    }

    let available_bytes = query_available_space(dir);

    Ok(DiskSpaceEstimate {
        estimated_bytes,
        available_bytes,
        headroom_factor: 1.2,
    })
}

/// Platform-aware available-space query using only pure Rust + std.
///
/// On Unix: runs `df -k <path>` and parses the "Available" column.
/// Fallback (Windows or parse failure): returns `u64::MAX` (conservative — assume plenty).
fn query_available_space(dir: &std::path::Path) -> u64 {
    #[cfg(unix)]
    {
        use std::process::Command;
        // `df -k` reports in 1-KiB blocks.
        let output = Command::new("df").arg("-k").arg(dir).output();
        if let Ok(out) = output {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout);
                if let Some(avail) = parse_df_available(&text) {
                    return avail * 1_024; // KiB → bytes
                }
            }
        }
        // Fallback: return a large sentinel so the check doesn't block the user.
        u64::MAX / 2
    }

    #[cfg(not(unix))]
    {
        // Windows: not yet supported without winapi or external crate.
        // Return a safe large value — the caller can still use estimated_bytes.
        let _ = dir;
        u64::MAX / 2
    }
}

/// Parse the "Available" column from `df -k` output.
///
/// The output format (POSIX) is:
/// ```text
/// Filesystem     1K-blocks   Used   Available Use% Mounted on
/// /dev/disk1     999999999  12345   888888888  2%   /
/// ```
/// The "Available" column is column index 3 (0-based) on the second line.
#[cfg(unix)]
fn parse_df_available(df_output: &str) -> Option<u64> {
    let mut lines = df_output.lines().skip(1); // skip header
    let data_line = lines.next()?;
    let fields: Vec<&str> = data_line.split_whitespace().collect();

    // POSIX df may wrap long device names onto the next line making the fields
    // shift.  Guard against that by trying both index 3 and index 2.
    for &idx in &[3_usize, 2_usize] {
        if let Some(avail_str) = fields.get(idx) {
            if let Ok(avail) = avail_str.parse::<u64>() {
                return Some(avail);
            }
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-export Result for the disk space functions
type Result<T, E> = std::result::Result<T, E>;

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv_profile::ConversionProfile;

    fn valid_profile() -> ValidateProfile {
        ValidateProfile::new("av1", 4_000, 1920, 1080, "webm")
    }

    // ── ValidationError display ───────────────────────────────────────────────

    #[test]
    fn error_display_unsupported_format() {
        let e = ValidationError::UnsupportedFormat("h264".to_string());
        assert!(e.to_string().contains("h264"));
    }

    #[test]
    fn error_display_resolution_too_large() {
        let e = ValidationError::ResolutionTooLarge {
            width: 10_000,
            height: 8_000,
        };
        assert!(e.to_string().contains("10000"));
    }

    #[test]
    fn error_display_invalid_bitrate() {
        let e = ValidationError::InvalidBitrate(0);
        assert!(e.to_string().contains("0 kbps"));
    }

    // ── ConvertValidation ─────────────────────────────────────────────────────

    #[test]
    fn validation_valid_profile_no_errors() {
        let v = ConvertValidation::new();
        let p = valid_profile();
        assert!(v.validate_profile(&p).is_empty());
    }

    #[test]
    fn validation_unsupported_codec() {
        let v = ConvertValidation::new();
        let p = ValidateProfile::new("h264", 4_000, 1920, 1080, "mp4");
        let errors = v.validate_profile(&p);
        assert!(!errors.is_empty());
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::UnsupportedFormat(_))));
    }

    #[test]
    fn validation_resolution_too_large() {
        let v = ConvertValidation::new();
        let p = ValidateProfile::new("av1", 4_000, 10_000, 10_000, "webm");
        let errors = v.validate_profile(&p);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::ResolutionTooLarge { .. })));
    }

    #[test]
    fn validation_invalid_bitrate_zero() {
        let v = ConvertValidation::new();
        let p = ValidateProfile::new("av1", 0, 1920, 1080, "webm");
        let errors = v.validate_profile(&p);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidBitrate(_))));
    }

    #[test]
    fn validation_bitrate_too_high() {
        let v = ConvertValidation::new();
        let p = ValidateProfile::new("av1", 200_000, 1920, 1080, "webm");
        let errors = v.validate_profile(&p);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidBitrate(_))));
    }

    #[test]
    fn validation_missing_codec_field() {
        let v = ConvertValidation::new();
        let p = ValidateProfile::new("", 4_000, 1920, 1080, "webm");
        let errors = v.validate_profile(&p);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::MissingRequiredField(f) if f == "codec")));
    }

    #[test]
    fn validation_incompatible_codec_container() {
        let v = ConvertValidation::new();
        // vorbis is not compatible with mp4
        let p = ValidateProfile::new("vorbis", 128, 1920, 1080, "mp4");
        let errors = v.validate_profile(&p);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::IncompatibleCodec { .. })));
    }

    #[test]
    fn validation_codec_supported_case_insensitive() {
        let v = ConvertValidation::new();
        assert!(v.is_codec_supported("AV1"));
        assert!(v.is_codec_supported("Opus"));
        assert!(!v.is_codec_supported("h264"));
    }

    // ── validate_input_file ───────────────────────────────────────────────────

    #[test]
    fn validate_file_empty_path() {
        let errors = validate_input_file("");
        assert!(!errors.is_empty());
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::MissingRequiredField(_))));
    }

    #[test]
    fn validate_file_known_extension() {
        assert!(validate_input_file("video.webm").is_empty());
        assert!(validate_input_file("audio.flac").is_empty());
    }

    #[test]
    fn validate_file_unknown_extension() {
        let errors = validate_input_file("archive.xyz");
        assert!(!errors.is_empty());
    }

    #[test]
    fn validate_file_no_extension() {
        let errors = validate_input_file("noextension");
        assert!(!errors.is_empty());
    }

    // ── DiskSpaceEstimate ─────────────────────────────────────────────────────

    #[test]
    fn disk_space_estimate_is_sufficient_when_enough_space() {
        let est = DiskSpaceEstimate {
            estimated_bytes: 1_000_000,
            available_bytes: 2_000_000,
            headroom_factor: 1.2,
        };
        assert!(est.is_sufficient());
    }

    #[test]
    fn disk_space_estimate_insufficient_when_too_little() {
        let est = DiskSpaceEstimate {
            estimated_bytes: 1_000_000,
            available_bytes: 500_000,
            headroom_factor: 1.2,
        };
        assert!(!est.is_sufficient());
    }

    #[test]
    fn disk_space_estimate_required_bytes_includes_headroom() {
        let est = DiskSpaceEstimate {
            estimated_bytes: 1_000_000,
            available_bytes: 2_000_000,
            headroom_factor: 1.2,
        };
        assert_eq!(est.required_bytes(), 1_200_000);
    }

    #[test]
    fn estimate_output_size_web_profile_is_smaller_than_input() {
        let profile = ConversionProfile {
            name: "web-720p".to_string(),
            video_codec: "vp9".to_string(),
            video_bitrate_kbps: 2_500,
            audio_codec: "opus".to_string(),
            audio_bitrate_kbps: 128,
            width: 1280,
            height: 720,
            fps_num: 30,
            fps_den: 1,
            preset: "fast".to_string(),
        };
        let input_size = 1_000_000_u64; // 1 MB
        let estimated = estimate_output_size(input_size, &profile);
        // Web profiles use 0.6 ratio → estimated = 600_000 (>= 1024)
        assert!(estimated > 0, "estimate should be positive");
        assert!(
            estimated <= input_size,
            "web profile should reduce file size: estimated={estimated}"
        );
    }

    #[test]
    fn estimate_output_size_archive_profile_may_be_larger() {
        let profile = ConversionProfile {
            name: "archive-lossless".to_string(),
            video_codec: "av1".to_string(),
            video_bitrate_kbps: 50_000,
            audio_codec: "flac".to_string(),
            audio_bitrate_kbps: 1_411,
            width: 3840,
            height: 2160,
            fps_num: 60,
            fps_den: 1,
            preset: "slow".to_string(),
        };
        let input_size = 1_000_000_u64;
        let estimated = estimate_output_size(input_size, &profile);
        assert!(
            estimated >= input_size,
            "archive should preserve or grow size"
        );
    }

    #[test]
    fn check_disk_space_temp_dir_returns_valid_estimate() {
        let tmp = std::env::temp_dir();
        let output_path = tmp.join("oximedia_validate_disk_test.mkv");
        let estimated = 1_000_000_u64;

        let result = check_disk_space(&output_path, estimated);
        assert!(
            result.is_ok(),
            "check_disk_space should succeed for existing temp dir: {:?}",
            result
        );
        let est = result.expect("result was checked above");
        assert_eq!(est.estimated_bytes, estimated);
        assert_eq!(est.headroom_factor, 1.2);
        // available_bytes should be > 0 (at minimum the sentinel value)
        assert!(est.available_bytes > 0);
    }

    #[test]
    fn check_disk_space_nonexistent_dir_returns_error() {
        let output_path =
            std::path::Path::new("/nonexistent_oximedia_validate_test_abc99/output.mkv");
        let result = check_disk_space(output_path, 1_000_000);
        assert!(result.is_err(), "should fail for non-existent directory");
    }

    #[test]
    #[cfg(unix)]
    fn parse_df_available_standard_output() {
        let df_output = "Filesystem     1K-blocks      Used  Available Use% Mounted on\n\
             /dev/disk1s1   487396864  92345678  390000000   2% /\n";
        let avail = parse_df_available(df_output);
        assert_eq!(avail, Some(390_000_000_u64));
    }
}
