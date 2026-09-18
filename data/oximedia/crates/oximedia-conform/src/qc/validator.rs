//! Validation for conform sessions and matches.
//!
//! This module contains:
//! - [`Validator`]: general conforming validation (score, handles, file existence)
//! - [`CodecValidator`]: codec-specific rules for AV1 video and Opus audio streams

use crate::config::ConformConfig;
use crate::error::ConformResult;
use crate::types::{ClipMatch, ClipReference};
use serde::{Deserialize, Serialize};

/// Validation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Validation errors.
    pub errors: Vec<ValidationError>,
    /// Validation warnings.
    pub warnings: Vec<ValidationWarning>,
    /// Is validation successful.
    pub is_valid: bool,
}

/// Validation error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Clip ID.
    pub clip_id: String,
    /// Error message.
    pub message: String,
    /// Error severity.
    pub severity: ErrorSeverity,
}

/// Validation warning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Clip ID.
    pub clip_id: String,
    /// Warning message.
    pub message: String,
}

/// Error severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Critical error.
    Critical,
    /// Error.
    Error,
    /// Warning.
    Warning,
}

/// Validator for conform sessions.
pub struct Validator {
    config: ConformConfig,
}

impl Validator {
    /// Create a new validator.
    #[must_use]
    pub fn new(config: ConformConfig) -> Self {
        Self { config }
    }

    /// Validate a match.
    #[must_use]
    pub fn validate_match(&self, clip_match: &ClipMatch) -> ValidationReport {
        let mut errors = Vec::new();
        let warnings = Vec::new();

        // Check match score
        if clip_match.score < self.config.match_threshold {
            errors.push(ValidationError {
                clip_id: clip_match.clip.id.clone(),
                message: format!(
                    "Match score {:.2} below threshold {:.2}",
                    clip_match.score, self.config.match_threshold
                ),
                severity: ErrorSeverity::Error,
            });
        }

        // Check handles
        if !self.config.allow_missing_handles {
            if let Err(e) = self.check_handles(&clip_match.clip, &clip_match.media) {
                errors.push(ValidationError {
                    clip_id: clip_match.clip.id.clone(),
                    message: e.to_string(),
                    severity: ErrorSeverity::Warning,
                });
            }
        }

        // Check file existence
        if !clip_match.media.path.exists() {
            errors.push(ValidationError {
                clip_id: clip_match.clip.id.clone(),
                message: format!("Media file not found: {}", clip_match.media.path.display()),
                severity: ErrorSeverity::Critical,
            });
        }

        ValidationReport {
            is_valid: errors.is_empty(),
            errors,
            warnings,
        }
    }

    /// Check if source has sufficient handles.
    fn check_handles(
        &self,
        _clip: &ClipReference,
        _media: &crate::types::MediaFile,
    ) -> ConformResult<()> {
        // Placeholder: would check if media has sufficient pre/post roll
        Ok(())
    }

    /// Validate all matches.
    #[must_use]
    pub fn validate_all(&self, matches: &[ClipMatch]) -> ValidationReport {
        let mut all_errors = Vec::new();
        let mut all_warnings = Vec::new();

        for clip_match in matches {
            let report = self.validate_match(clip_match);
            all_errors.extend(report.errors);
            all_warnings.extend(report.warnings);
        }

        ValidationReport {
            is_valid: all_errors.is_empty(),
            errors: all_errors,
            warnings: all_warnings,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AV1 codec-specific validation
// ─────────────────────────────────────────────────────────────────────────────

/// AV1 sequence profile as defined in the AV1 specification (§6.4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Av1Profile {
    /// Profile 0 — 4:2:0 chroma subsampling, bit depths 8 and 10.
    Main,
    /// Profile 1 — 4:4:4 chroma subsampling, bit depths 8 and 10.
    High,
    /// Profile 2 — 4:2:0 / 4:2:2 / 4:4:4 chroma, bit depths 8, 10, and 12.
    Professional,
}

/// AV1 level as defined in Annex A of the AV1 specification.
///
/// The numeric suffix encodes `tier × 10 + level_index` where tier 0 = Main,
/// tier 1 = High.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Av1Level {
    /// Level 2.0 — max luma sample rate 1 474 560 s⁻¹.
    Level2_0,
    /// Level 2.1 — max luma sample rate 2 228 224 s⁻¹.
    Level2_1,
    /// Level 3.0 — max luma sample rate 4 587 520 s⁻¹.
    Level3_0,
    /// Level 3.1 — max luma sample rate 8 355 840 s⁻¹.
    Level3_1,
    /// Level 4.0 — max luma sample rate 19 975 680 s⁻¹.
    Level4_0,
    /// Level 4.1 — max luma sample rate 31 950 720 s⁻¹.
    Level4_1,
    /// Level 5.0 — max luma sample rate 70 778 880 s⁻¹.
    Level5_0,
    /// Level 5.1 — max luma sample rate 141 557 760 s⁻¹.
    Level5_1,
    /// Level 5.2 — max luma sample rate 283 115 520 s⁻¹.
    Level5_2,
    /// Level 5.3 — max luma sample rate 566 231 040 s⁻¹.
    Level5_3,
    /// Level 6.0 — max luma sample rate 1 069 547 520 s⁻¹.
    Level6_0,
    /// Level 6.1 — max luma sample rate 2 139 095 040 s⁻¹.
    Level6_1,
    /// Level 6.2 — max luma sample rate 4 278 190 080 s⁻¹.
    Level6_2,
    /// Level 6.3 — max luma sample rate 4 278 190 080 s⁻¹ (high tier).
    Level6_3,
}

/// Maximum luma sample rate (samples per second) per AV1 level (Annex A, Table A.2).
///
/// Represented as u64 to avoid f64 precision issues.
impl Av1Level {
    /// Maximum luma picture size (width × height) in samples.
    #[must_use]
    pub const fn max_picture_size(self) -> u64 {
        match self {
            Self::Level2_0 => 147_456,
            Self::Level2_1 => 278_784,
            Self::Level3_0 => 665_856,
            Self::Level3_1 => 1_065_024,
            Self::Level4_0 => 2_359_296,
            Self::Level4_1 => 2_359_296,
            Self::Level5_0 => 8_912_896,
            Self::Level5_1 => 8_912_896,
            Self::Level5_2 => 8_912_896,
            Self::Level5_3 => 8_912_896,
            Self::Level6_0 => 35_651_584,
            Self::Level6_1 => 35_651_584,
            Self::Level6_2 => 35_651_584,
            Self::Level6_3 => 35_651_584,
        }
    }

    /// Maximum luma sample rate (samples per second).
    #[must_use]
    pub const fn max_luma_sample_rate(self) -> u64 {
        match self {
            Self::Level2_0 => 4_423_680,
            Self::Level2_1 => 8_363_008,
            Self::Level3_0 => 19_975_680,
            Self::Level3_1 => 37_838_848,
            Self::Level4_0 => 70_778_880,
            Self::Level4_1 => 141_557_760,
            Self::Level5_0 => 267_386_880,
            Self::Level5_1 => 534_773_760,
            Self::Level5_2 => 1_069_547_520,
            Self::Level5_3 => 1_069_547_520,
            Self::Level6_0 => 1_069_547_520,
            Self::Level6_1 => 2_139_095_040,
            Self::Level6_2 => 4_278_190_080,
            Self::Level6_3 => 4_278_190_080,
        }
    }

    /// Maximum decoder bitrate (bits per second) for the Main tier.
    #[must_use]
    pub const fn max_bitrate_main_tier_bps(self) -> u64 {
        match self {
            Self::Level2_0 => 1_500_000,
            Self::Level2_1 => 3_000_000,
            Self::Level3_0 => 6_000_000,
            Self::Level3_1 => 10_000_000,
            Self::Level4_0 => 12_000_000,
            Self::Level4_1 => 20_000_000,
            Self::Level5_0 => 30_000_000,
            Self::Level5_1 => 40_000_000,
            Self::Level5_2 => 60_000_000,
            Self::Level5_3 => 60_000_000,
            Self::Level6_0 => 60_000_000,
            Self::Level6_1 => 100_000_000,
            Self::Level6_2 => 160_000_000,
            Self::Level6_3 => 160_000_000,
        }
    }
}

/// Parameters describing an AV1 video stream to be validated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Av1StreamParams {
    /// Sequence profile.
    pub profile: Av1Profile,
    /// Claimed level.
    pub level: Av1Level,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame rate (frames per second).
    pub frame_rate: f64,
    /// Actual stream bitrate in bits per second.
    pub bitrate_bps: u64,
    /// Bit depth (8, 10, or 12).
    pub bit_depth: u8,
    /// Chroma subsampling in CSS notation ("4:2:0", "4:2:2", "4:4:4").
    pub chroma_subsampling: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Opus codec-specific validation
// ─────────────────────────────────────────────────────────────────────────────

/// Opus audio application mode (RFC 6716 §2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpusApplication {
    /// VoIP — speech optimised, uses SILK/CELT hybrids.
    Voip,
    /// Audio — music/general audio, full CELT bandwidth.
    Audio,
    /// Restricted low-delay — CELT-only, lowest latency.
    RestrictedLowdelay,
}

/// Parameters describing an Opus audio stream to be validated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpusStreamParams {
    /// Number of audio channels (1–8 per RFC 7845).
    pub channels: u8,
    /// Stream bitrate in bits per second.
    pub bitrate_bps: u32,
    /// Sample rate of the input signal in Hz.
    /// Opus internally operates at 8/12/16/24/48 kHz.
    pub input_sample_rate: u32,
    /// Application mode.
    pub application: OpusApplication,
    /// Frame duration in milliseconds (2.5, 5, 10, 20, 40, or 60 ms).
    pub frame_duration_ms: f32,
    /// Whether variable bitrate (VBR) is enabled.
    pub vbr: bool,
    /// Complexity setting (0–10).
    pub complexity: u8,
}

// ─────────────────────────────────────────────────────────────────────────────
// CodecValidationIssue
// ─────────────────────────────────────────────────────────────────────────────

/// A single codec-validation issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecValidationIssue {
    /// Human-readable description.
    pub message: String,
    /// Whether this issue is fatal (true) or advisory (false).
    pub fatal: bool,
}

/// Result of codec-specific validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecValidationReport {
    /// Issues found.  An empty list means validation passed.
    pub issues: Vec<CodecValidationIssue>,
    /// `true` if there are no fatal issues.
    pub passed: bool,
}

impl CodecValidationReport {
    fn new(issues: Vec<CodecValidationIssue>) -> Self {
        let passed = issues.iter().all(|i| !i.fatal);
        Self { issues, passed }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CodecValidator
// ─────────────────────────────────────────────────────────────────────────────

/// Codec-specific validator implementing AV1 level constraints (Annex A of
/// the AV1 specification) and Opus bitrate / channel constraints (RFC 6716 /
/// RFC 7845).
///
/// # AV1 Checks
///
/// - Picture size ≤ level `max_picture_size`
/// - Luma sample rate (width × height × fps) ≤ level `max_luma_sample_rate`
/// - Bitrate ≤ level `max_bitrate_main_tier_bps`
/// - Profile / chroma-subsampling compatibility
/// - Bit depth constraints per profile
///
/// # Opus Checks
///
/// - Channel count in [1, 8]
/// - Bitrate in [6 000, 510 000] bps per channel (overall range 6–510 kbps)
/// - Input sample rate must be one of the supported Opus internal rates
///   (8/12/16/24/48 kHz) or an arbitrary PCM rate that Opus accepts
/// - Frame duration must be a supported value
/// - Complexity in [0, 10]
/// - VoIP application with ≥ 2 channels raises an advisory warning
pub struct CodecValidator;

impl CodecValidator {
    /// Create a new codec validator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Validate an AV1 video stream against its claimed level constraints.
    #[must_use]
    pub fn validate_av1(&self, params: &Av1StreamParams) -> CodecValidationReport {
        let mut issues = Vec::new();

        // ── Profile / chroma compatibility ────────────────────────────────
        match params.profile {
            Av1Profile::Main => {
                if params.chroma_subsampling != "4:2:0" {
                    issues.push(CodecValidationIssue {
                        message: format!(
                            "AV1 Main profile supports only 4:2:0 chroma; found '{}'",
                            params.chroma_subsampling
                        ),
                        fatal: true,
                    });
                }
                if !matches!(params.bit_depth, 8 | 10) {
                    issues.push(CodecValidationIssue {
                        message: format!(
                            "AV1 Main profile supports bit depths 8 and 10; found {}",
                            params.bit_depth
                        ),
                        fatal: true,
                    });
                }
            }
            Av1Profile::High => {
                if params.chroma_subsampling != "4:4:4" {
                    issues.push(CodecValidationIssue {
                        message: format!(
                            "AV1 High profile supports only 4:4:4 chroma; found '{}'",
                            params.chroma_subsampling
                        ),
                        fatal: true,
                    });
                }
                if !matches!(params.bit_depth, 8 | 10) {
                    issues.push(CodecValidationIssue {
                        message: format!(
                            "AV1 High profile supports bit depths 8 and 10; found {}",
                            params.bit_depth
                        ),
                        fatal: true,
                    });
                }
            }
            Av1Profile::Professional => {
                if !matches!(params.bit_depth, 8 | 10 | 12) {
                    issues.push(CodecValidationIssue {
                        message: format!(
                            "AV1 Professional profile supports bit depths 8, 10, and 12; found {}",
                            params.bit_depth
                        ),
                        fatal: true,
                    });
                }
            }
        }

        // ── Picture size ──────────────────────────────────────────────────
        let picture_samples = u64::from(params.width) * u64::from(params.height);
        let max_pic = params.level.max_picture_size();
        if picture_samples > max_pic {
            issues.push(CodecValidationIssue {
                message: format!(
                    "AV1 {level:?}: picture size {w}×{h} = {ps} samples exceeds level maximum {max_pic}",
                    level = params.level,
                    w = params.width,
                    h = params.height,
                    ps = picture_samples,
                ),
                fatal: true,
            });
        }

        // ── Luma sample rate ──────────────────────────────────────────────
        if params.frame_rate > 0.0 {
            let lsr = (picture_samples as f64 * params.frame_rate) as u64;
            let max_lsr = params.level.max_luma_sample_rate();
            if lsr > max_lsr {
                issues.push(CodecValidationIssue {
                    message: format!(
                        "AV1 {level:?}: luma sample rate {lsr} s⁻¹ exceeds level maximum {max_lsr}",
                        level = params.level,
                    ),
                    fatal: true,
                });
            }
        } else {
            issues.push(CodecValidationIssue {
                message: "AV1: frame_rate must be > 0".to_string(),
                fatal: true,
            });
        }

        // ── Bitrate ───────────────────────────────────────────────────────
        let max_br = params.level.max_bitrate_main_tier_bps();
        if params.bitrate_bps > max_br {
            issues.push(CodecValidationIssue {
                message: format!(
                    "AV1 {level:?}: bitrate {actual} bps exceeds Main-tier level maximum {max_br} bps",
                    level = params.level,
                    actual = params.bitrate_bps,
                ),
                fatal: false, // advisory: High-tier may allow more
            });
        }

        CodecValidationReport::new(issues)
    }

    /// Validate an Opus audio stream against RFC 6716 / RFC 7845 constraints.
    #[must_use]
    pub fn validate_opus(&self, params: &OpusStreamParams) -> CodecValidationReport {
        let mut issues = Vec::new();

        // ── Channel count ─────────────────────────────────────────────────
        if params.channels == 0 {
            issues.push(CodecValidationIssue {
                message: "Opus: channel count must be at least 1".to_string(),
                fatal: true,
            });
        } else if params.channels > 8 {
            issues.push(CodecValidationIssue {
                message: format!(
                    "Opus: channel count {} exceeds RFC 7845 maximum of 8",
                    params.channels
                ),
                fatal: true,
            });
        }

        // ── Bitrate range ─────────────────────────────────────────────────
        // RFC 6716: supported range is 6–510 kbps total.  Per-channel minimum
        // is 6 kbps; overall maximum depends on application mode:
        //   VoIP: 32 kbps per channel is practical upper bound (advisory).
        //   Audio: up to 256 kbps per channel is reasonable.
        // Hard limits: total 6 000 – 510 000 bps.
        const OPUS_MIN_BPS: u32 = 6_000;
        const OPUS_MAX_BPS: u32 = 510_000;

        if params.bitrate_bps < OPUS_MIN_BPS {
            issues.push(CodecValidationIssue {
                message: format!(
                    "Opus: bitrate {} bps is below minimum {} bps",
                    params.bitrate_bps, OPUS_MIN_BPS
                ),
                fatal: true,
            });
        }
        if params.bitrate_bps > OPUS_MAX_BPS {
            issues.push(CodecValidationIssue {
                message: format!(
                    "Opus: bitrate {} bps exceeds maximum {} bps",
                    params.bitrate_bps, OPUS_MAX_BPS
                ),
                fatal: true,
            });
        }

        // Per-channel advisory checks.
        if params.channels > 0 {
            let per_channel_bps = params.bitrate_bps / u32::from(params.channels);
            match params.application {
                OpusApplication::Voip if per_channel_bps > 64_000 => {
                    issues.push(CodecValidationIssue {
                        message: format!(
                            "Opus VoIP: per-channel bitrate {} bps is unusually high (> 64 kbps); consider Audio application mode",
                            per_channel_bps
                        ),
                        fatal: false,
                    });
                }
                OpusApplication::Audio if per_channel_bps < 24_000 => {
                    issues.push(CodecValidationIssue {
                        message: format!(
                            "Opus Audio: per-channel bitrate {} bps may be too low for high-quality music reproduction (< 24 kbps)",
                            per_channel_bps
                        ),
                        fatal: false,
                    });
                }
                _ => {}
            }
        }

        // ── Sample rate ───────────────────────────────────────────────────
        // Opus internally resamples to 8/12/16/24/48 kHz.  Any input PCM rate
        // is technically accepted, but rates outside these values result in
        // resampling that may introduce artefacts.  We warn on non-standard
        // rates while accepting all values ≥ 8 000 Hz.
        const OPUS_INTERNAL_RATES: [u32; 5] = [8_000, 12_000, 16_000, 24_000, 48_000];
        if params.input_sample_rate < 8_000 {
            issues.push(CodecValidationIssue {
                message: format!(
                    "Opus: input sample rate {} Hz is below the minimum supported 8 000 Hz",
                    params.input_sample_rate
                ),
                fatal: true,
            });
        } else if !OPUS_INTERNAL_RATES.contains(&params.input_sample_rate) {
            issues.push(CodecValidationIssue {
                message: format!(
                    "Opus: input sample rate {} Hz is not a native Opus rate (8/12/16/24/48 kHz); resampling will occur",
                    params.input_sample_rate
                ),
                fatal: false,
            });
        }

        // ── Frame duration ────────────────────────────────────────────────
        // RFC 6716 §2.1.4 supported frame sizes: 2.5, 5, 10, 20, 40, 60 ms.
        const VALID_FRAME_DURATIONS: [f32; 6] = [2.5, 5.0, 10.0, 20.0, 40.0, 60.0];
        let is_valid_duration = VALID_FRAME_DURATIONS
            .iter()
            .any(|&d| (params.frame_duration_ms - d).abs() < 0.01);
        if !is_valid_duration {
            issues.push(CodecValidationIssue {
                message: format!(
                    "Opus: frame duration {} ms is not a supported value (2.5/5/10/20/40/60 ms)",
                    params.frame_duration_ms
                ),
                fatal: true,
            });
        }

        // ── Complexity ────────────────────────────────────────────────────
        if params.complexity > 10 {
            issues.push(CodecValidationIssue {
                message: format!("Opus: complexity {} is outside [0, 10]", params.complexity),
                fatal: true,
            });
        }

        // ── Application / channel advisory ────────────────────────────────
        if params.application == OpusApplication::Voip && params.channels >= 2 {
            issues.push(CodecValidationIssue {
                message: format!(
                    "Opus VoIP application with {} channels: VoIP is optimised for mono/stereo speech; consider Audio application for multi-channel",
                    params.channels
                ),
                fatal: false,
            });
        }

        CodecValidationReport::new(issues)
    }
}

impl Default for CodecValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FrameRate, MatchMethod, MediaFile, Timecode, TrackType};
    use std::path::PathBuf;

    fn create_test_clip() -> ClipReference {
        ClipReference {
            id: "test".to_string(),
            source_file: Some("test.mov".to_string()),
            source_in: Timecode::new(1, 0, 0, 0),
            source_out: Timecode::new(1, 0, 10, 0),
            record_in: Timecode::new(1, 0, 0, 0),
            record_out: Timecode::new(1, 0, 10, 0),
            track: TrackType::Video,
            fps: FrameRate::Fps25,
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_validator_creation() {
        let config = ConformConfig::default();
        let _validator = Validator::new(config);
    }

    #[test]
    fn test_validate_match_score() {
        let config = ConformConfig::default();
        let validator = Validator::new(config);

        let clip = create_test_clip();
        let media = MediaFile::new(PathBuf::from("/nonexistent/test.mov"));

        let clip_match = ClipMatch {
            clip,
            media,
            score: 0.5,
            method: MatchMethod::ExactFilename,
            details: String::new(),
        };

        let report = validator.validate_match(&clip_match);
        assert!(!report.is_valid);
        assert!(!report.errors.is_empty());
    }

    // ── CodecValidator helpers ────────────────────────────────────────────

    fn valid_av1_params() -> Av1StreamParams {
        Av1StreamParams {
            profile: Av1Profile::Main,
            level: Av1Level::Level4_0,
            width: 1920,
            height: 1080,
            frame_rate: 24.0,
            bitrate_bps: 8_000_000,
            bit_depth: 8,
            chroma_subsampling: "4:2:0".to_string(),
        }
    }

    fn valid_opus_params() -> OpusStreamParams {
        OpusStreamParams {
            channels: 2,
            bitrate_bps: 128_000,
            input_sample_rate: 48_000,
            application: OpusApplication::Audio,
            frame_duration_ms: 20.0,
            vbr: true,
            complexity: 5,
        }
    }

    // ── AV1 tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_av1_valid_stream_passes() {
        let cv = CodecValidator::new();
        let params = valid_av1_params();
        let report = cv.validate_av1(&params);
        assert!(
            report.passed,
            "valid AV1 stream should pass; issues: {:?}",
            report.issues
        );
        assert!(report.issues.is_empty());
    }

    #[test]
    fn test_av1_main_wrong_chroma_fails() {
        let cv = CodecValidator::new();
        let mut params = valid_av1_params();
        params.chroma_subsampling = "4:4:4".to_string();
        let report = cv.validate_av1(&params);
        assert!(!report.passed, "Main profile with 4:4:4 should fail");
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.fatal && i.message.contains("4:2:0")),
            "should mention 4:2:0 constraint"
        );
    }

    #[test]
    fn test_av1_professional_allows_12bit() {
        let cv = CodecValidator::new();
        let mut params = valid_av1_params();
        params.profile = Av1Profile::Professional;
        params.bit_depth = 12;
        params.chroma_subsampling = "4:2:0".to_string();
        // Picture size and LSR are within Level4_0 limits for 1920×1080@24.
        let report = cv.validate_av1(&params);
        // Professional profile allows 12-bit — no fatal issues from profile check.
        let profile_fatal = report
            .issues
            .iter()
            .any(|i| i.fatal && i.message.contains("bit depth"));
        assert!(!profile_fatal, "Professional profile should allow 12-bit");
    }

    #[test]
    fn test_av1_picture_size_exceeds_level() {
        let cv = CodecValidator::new();
        let mut params = valid_av1_params();
        // Level 2.0 max picture size = 147 456; 1920×1080 = 2 073 600 > limit.
        params.level = Av1Level::Level2_0;
        let report = cv.validate_av1(&params);
        let has_size_issue = report
            .issues
            .iter()
            .any(|i| i.fatal && i.message.contains("picture size"));
        assert!(has_size_issue, "should flag picture size violation");
    }

    #[test]
    fn test_av1_zero_frame_rate_fails() {
        let cv = CodecValidator::new();
        let mut params = valid_av1_params();
        params.frame_rate = 0.0;
        let report = cv.validate_av1(&params);
        let has_fps_issue = report
            .issues
            .iter()
            .any(|i| i.fatal && i.message.contains("frame_rate"));
        assert!(has_fps_issue, "zero frame rate should be flagged");
    }

    #[test]
    fn test_av1_bitrate_over_limit_advisory_not_fatal() {
        let cv = CodecValidator::new();
        let mut params = valid_av1_params();
        // Level 2.0 max = 1.5 Mbps; set 50 Mbps.
        params.level = Av1Level::Level2_0;
        params.width = 320;
        params.height = 240;
        params.frame_rate = 30.0;
        params.bitrate_bps = 50_000_000;
        let report = cv.validate_av1(&params);
        let bitrate_issue = report.issues.iter().find(|i| i.message.contains("bitrate"));
        // Bitrate issue exists but is advisory (fatal = false).
        if let Some(issue) = bitrate_issue {
            assert!(!issue.fatal, "bitrate overage should be advisory only");
        }
    }

    #[test]
    fn test_av1_level_constants_monotone() {
        // Verify that higher levels have >= picture size and LSR.
        let levels = [
            Av1Level::Level2_0,
            Av1Level::Level3_0,
            Av1Level::Level4_0,
            Av1Level::Level5_0,
            Av1Level::Level6_0,
        ];
        for w in levels.windows(2) {
            let lo = w[0];
            let hi = w[1];
            assert!(
                hi.max_picture_size() >= lo.max_picture_size(),
                "{hi:?} max_picture_size should be >= {lo:?}"
            );
            assert!(
                hi.max_luma_sample_rate() >= lo.max_luma_sample_rate(),
                "{hi:?} max_luma_sample_rate should be >= {lo:?}"
            );
        }
    }

    // ── Opus tests ────────────────────────────────────────────────────────

    #[test]
    fn test_opus_valid_stream_passes() {
        let cv = CodecValidator::new();
        let params = valid_opus_params();
        let report = cv.validate_opus(&params);
        assert!(
            report.passed,
            "valid Opus stream should pass; issues: {:?}",
            report.issues
        );
    }

    #[test]
    fn test_opus_zero_channels_fails() {
        let cv = CodecValidator::new();
        let mut params = valid_opus_params();
        params.channels = 0;
        let report = cv.validate_opus(&params);
        assert!(!report.passed);
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.fatal && i.message.contains("channel count")),
            "should flag zero channels"
        );
    }

    #[test]
    fn test_opus_too_many_channels_fails() {
        let cv = CodecValidator::new();
        let mut params = valid_opus_params();
        params.channels = 9;
        let report = cv.validate_opus(&params);
        assert!(!report.passed);
        assert!(report
            .issues
            .iter()
            .any(|i| i.fatal && i.message.contains("maximum of 8")),);
    }

    #[test]
    fn test_opus_bitrate_below_minimum_fails() {
        let cv = CodecValidator::new();
        let mut params = valid_opus_params();
        params.bitrate_bps = 1_000; // below 6 kbps
        let report = cv.validate_opus(&params);
        assert!(!report.passed);
        assert!(report
            .issues
            .iter()
            .any(|i| i.fatal && i.message.contains("below minimum")),);
    }

    #[test]
    fn test_opus_bitrate_above_maximum_fails() {
        let cv = CodecValidator::new();
        let mut params = valid_opus_params();
        params.bitrate_bps = 600_000; // above 510 kbps
        let report = cv.validate_opus(&params);
        assert!(!report.passed);
        assert!(report
            .issues
            .iter()
            .any(|i| i.fatal && i.message.contains("exceeds maximum")),);
    }

    #[test]
    fn test_opus_invalid_sample_rate_fatal() {
        let cv = CodecValidator::new();
        let mut params = valid_opus_params();
        params.input_sample_rate = 4_000; // below 8 kHz
        let report = cv.validate_opus(&params);
        assert!(!report.passed);
        assert!(report
            .issues
            .iter()
            .any(|i| i.fatal && i.message.contains("below the minimum")),);
    }

    #[test]
    fn test_opus_non_native_sample_rate_advisory() {
        let cv = CodecValidator::new();
        let mut params = valid_opus_params();
        params.input_sample_rate = 44_100; // not a native Opus rate
        let report = cv.validate_opus(&params);
        // Should pass (no fatal), but have an advisory.
        assert!(report.passed, "44100 Hz advisory should not be fatal");
        assert!(
            report
                .issues
                .iter()
                .any(|i| !i.fatal && i.message.contains("resampling")),
            "should warn about resampling"
        );
    }

    #[test]
    fn test_opus_invalid_frame_duration_fails() {
        let cv = CodecValidator::new();
        let mut params = valid_opus_params();
        params.frame_duration_ms = 15.0; // not a valid Opus frame size
        let report = cv.validate_opus(&params);
        assert!(!report.passed);
        assert!(report
            .issues
            .iter()
            .any(|i| i.fatal && i.message.contains("frame duration")),);
    }

    #[test]
    fn test_opus_complexity_out_of_range_fails() {
        let cv = CodecValidator::new();
        let mut params = valid_opus_params();
        params.complexity = 11;
        let report = cv.validate_opus(&params);
        assert!(!report.passed);
        assert!(report
            .issues
            .iter()
            .any(|i| i.fatal && i.message.contains("complexity")),);
    }

    #[test]
    fn test_opus_voip_multichannel_advisory() {
        let cv = CodecValidator::new();
        let mut params = valid_opus_params();
        params.application = OpusApplication::Voip;
        params.channels = 4;
        params.bitrate_bps = 128_000;
        let report = cv.validate_opus(&params);
        // Advisory (non-fatal), so still passes.
        assert!(report.passed, "advisory should not cause failure");
        assert!(
            report
                .issues
                .iter()
                .any(|i| !i.fatal && i.message.contains("VoIP")),
            "should have VoIP multi-channel advisory"
        );
    }

    #[test]
    fn test_opus_voip_high_per_channel_bitrate_advisory() {
        let cv = CodecValidator::new();
        let params = OpusStreamParams {
            channels: 1,
            bitrate_bps: 128_000, // 128 kbps on single-channel VoIP
            input_sample_rate: 48_000,
            application: OpusApplication::Voip,
            frame_duration_ms: 20.0,
            vbr: false,
            complexity: 5,
        };
        let report = cv.validate_opus(&params);
        assert!(report.passed, "high VoIP bitrate is advisory, not fatal");
        assert!(report
            .issues
            .iter()
            .any(|i| !i.fatal && i.message.contains("unusually high")),);
    }

    #[test]
    fn test_opus_audio_low_per_channel_bitrate_advisory() {
        let cv = CodecValidator::new();
        let params = OpusStreamParams {
            channels: 2,
            bitrate_bps: 32_000, // 16 kbps per channel — low for music
            input_sample_rate: 48_000,
            application: OpusApplication::Audio,
            frame_duration_ms: 20.0,
            vbr: true,
            complexity: 5,
        };
        let report = cv.validate_opus(&params);
        assert!(
            report.passed,
            "low audio bitrate advisory should not be fatal"
        );
        assert!(report
            .issues
            .iter()
            .any(|i| !i.fatal && i.message.contains("too low")),);
    }
}
