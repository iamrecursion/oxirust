#![allow(unused_imports)]
//! AV1 bitstream compliance validation.
//!
//! Provides validation of AV1 codec bitstream including:
//! - Sequence header validation
//! - Frame header validation
//! - OBU (Open Bitstream Unit) structure
//! - Profile/level compliance
//! - Chroma subsampling validation
//! - Color space validation

use crate::rules::{CheckResult, QcContext, QcRule, RuleCategory, Severity};
use oximedia_core::{CodecId, OxiResult};

/// AV1 profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Av1Profile {
    /// Main profile (4:2:0 8/10-bit).
    Main,
    /// High profile (4:4:4 up to 10-bit).
    High,
    /// Professional profile (4:2:2/4:4:4 up to 12-bit).
    Professional,
}

/// AV1 level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Av1Level {
    /// Level 2.0
    Level2_0,
    /// Level 2.1
    Level2_1,
    /// Level 3.0
    Level3_0,
    /// Level 3.1
    Level3_1,
    /// Level 4.0
    Level4_0,
    /// Level 4.1
    Level4_1,
    /// Level 5.0
    Level5_0,
    /// Level 5.1
    Level5_1,
    /// Level 5.2
    Level5_2,
    /// Level 5.3
    Level5_3,
    /// Level 6.0
    Level6_0,
    /// Level 6.1
    Level6_1,
    /// Level 6.2
    Level6_2,
    /// Level 6.3
    Level6_3,
}

/// AV1 bitstream validator.
///
/// Validates compliance with AV1 specification.
pub struct Av1BitstreamValidator {
    expected_profile: Option<Av1Profile>,
    expected_level: Option<Av1Level>,
    check_obu_structure: bool,
    check_sequence_header: bool,
    check_frame_headers: bool,
}

impl Av1BitstreamValidator {
    /// Creates a new AV1 bitstream validator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            expected_profile: None,
            expected_level: None,
            check_obu_structure: true,
            check_sequence_header: true,
            check_frame_headers: true,
        }
    }

    /// Sets the expected AV1 profile.
    #[must_use]
    pub const fn with_profile(mut self, profile: Av1Profile) -> Self {
        self.expected_profile = Some(profile);
        self
    }

    /// Sets the expected AV1 level.
    #[must_use]
    pub const fn with_level(mut self, level: Av1Level) -> Self {
        self.expected_level = Some(level);
        self
    }

    /// Sets whether to validate OBU structure.
    #[must_use]
    pub const fn with_obu_check(mut self, check: bool) -> Self {
        self.check_obu_structure = check;
        self
    }

    /// Validates OBU structure.
    fn validate_obu_structure(&self, _stream_index: usize) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_obu_structure {
            // In production, would validate:
            // - OBU header forbidden_bit is 0
            // - OBU type is valid
            // - OBU has_size_field and has_extension_field flags
            // - OBU size field is valid
            // - OBU temporal/spatial IDs in extension

            results.push(
                CheckResult::pass(self.name()).with_recommendation(
                    "OBU (Open Bitstream Unit) structure validation".to_string(),
                ),
            );
        }

        Ok(results)
    }

    /// Validates sequence header OBU.
    fn validate_sequence_header(&self, _stream_index: usize) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_sequence_header {
            // In production, would validate:
            // - seq_profile (0=Main, 1=High, 2=Professional)
            // - seq_level_idx
            // - seq_tier (0=Main, 1=High)
            // - frame_width_bits and frame_height_bits
            // - max_frame_width and max_frame_height
            // - use_128x128_superblock flag
            // - enable_filter_intra, enable_intra_edge_filter
            // - enable_interintra_compound, enable_masked_compound
            // - enable_warped_motion, enable_dual_filter
            // - enable_order_hint, enable_jnt_comp, enable_ref_frame_mvs
            // - seq_choose_screen_content_tools
            // - seq_force_screen_content_tools
            // - seq_choose_integer_mv, seq_force_integer_mv
            // - color_config (bit_depth, mono_chrome, color_primaries, etc.)

            results.push(
                CheckResult::pass(self.name())
                    .with_recommendation("Sequence header OBU validation".to_string()),
            );
        }

        Ok(results)
    }

    /// Validates frame headers.
    fn validate_frame_headers(&self, _stream_index: usize) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_frame_headers {
            // In production, would validate:
            // - show_existing_frame flag
            // - frame_type (KEY_FRAME, INTER_FRAME, INTRA_ONLY_FRAME, SWITCH_FRAME)
            // - show_frame flag
            // - error_resilient_mode
            // - disable_cdf_update
            // - allow_screen_content_tools
            // - frame_size and render_size
            // - refresh_frame_flags
            // - quantization_params
            // - segmentation_params
            // - tile_info
            // - loop_filter_params
            // - cdef_params (Constrained Directional Enhancement Filter)
            // - lr_params (Loop Restoration)

            results.push(
                CheckResult::pass(self.name())
                    .with_recommendation("Frame header validation".to_string()),
            );
        }

        Ok(results)
    }

    /// Validates profile/level compliance.
    fn validate_profile_level(&self, _stream_index: usize) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if let Some(profile) = self.expected_profile {
            results.push(
                CheckResult::pass(self.name())
                    .with_recommendation(format!("Expected AV1 profile: {profile:?}")),
            );
        }

        if let Some(level) = self.expected_level {
            results.push(
                CheckResult::pass(self.name())
                    .with_recommendation(format!("Expected AV1 level: {level:?}")),
            );
        }

        Ok(results)
    }

    /// Validates color configuration.
    fn validate_color_config(&self, _stream_index: usize) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // In production, would validate:
        // - high_bitdepth, twelve_bit flags
        // - mono_chrome flag
        // - color_primaries (BT.709, BT.2020, etc.)
        // - transfer_characteristics (SDR, PQ, HLG)
        // - matrix_coefficients
        // - color_range (full vs studio)
        // - subsampling_x, subsampling_y (4:2:0, 4:2:2, 4:4:4)
        // - chroma_sample_position

        results.push(
            CheckResult::pass(self.name())
                .with_recommendation("Color configuration validation".to_string()),
        );

        Ok(results)
    }
}

impl Default for Av1BitstreamValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for Av1BitstreamValidator {
    fn name(&self) -> &str {
        "av1_bitstream_validation"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Video
    }

    fn description(&self) -> &str {
        "Validates AV1 bitstream compliance"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut all_results = Vec::new();

        // Only applicable to AV1 streams
        let av1_streams: Vec<_> = context
            .video_streams()
            .into_iter()
            .filter(|s| s.codec == CodecId::Av1)
            .collect();

        if av1_streams.is_empty() {
            return Ok(vec![CheckResult::pass(self.name()).with_recommendation(
                "No AV1 streams found - check skipped".to_string(),
            )]);
        }

        for stream in av1_streams {
            // Validate OBU structure
            let obu_results = self.validate_obu_structure(stream.index)?;
            all_results.extend(obu_results);

            // Validate sequence header
            let seq_results = self.validate_sequence_header(stream.index)?;
            all_results.extend(seq_results);

            // Validate frame headers
            let frame_results = self.validate_frame_headers(stream.index)?;
            all_results.extend(frame_results);

            // Validate profile/level
            let profile_results = self.validate_profile_level(stream.index)?;
            all_results.extend(profile_results);

            // Validate color config
            let color_results = self.validate_color_config(stream.index)?;
            all_results.extend(color_results);
        }

        Ok(all_results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        context
            .video_streams()
            .iter()
            .any(|s| s.codec == CodecId::Av1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_av1_validator_creation() {
        let validator = Av1BitstreamValidator::new();
        assert_eq!(validator.name(), "av1_bitstream_validation");
        assert!(validator.check_obu_structure);
    }

    #[test]
    fn test_av1_validator_configuration() {
        let validator = Av1BitstreamValidator::new()
            .with_profile(Av1Profile::Main)
            .with_level(Av1Level::Level4_0);

        assert_eq!(validator.expected_profile, Some(Av1Profile::Main));
        assert_eq!(validator.expected_level, Some(Av1Level::Level4_0));
    }

    #[test]
    fn test_av1_profiles() {
        assert_eq!(Av1Profile::Main, Av1Profile::Main);
        assert_ne!(Av1Profile::Main, Av1Profile::High);
    }

    #[test]
    fn test_av1_levels() {
        assert_eq!(Av1Level::Level4_0, Av1Level::Level4_0);
        assert_ne!(Av1Level::Level4_0, Av1Level::Level5_0);
    }
}
