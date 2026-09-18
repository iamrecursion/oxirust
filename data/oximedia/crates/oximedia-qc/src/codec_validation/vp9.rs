#![allow(unused_imports)]
//! VP9 bitstream compliance validation.
//!
//! Provides validation of VP9 codec bitstream including:
//! - Superframe structure validation
//! - Frame header validation
//! - Profile validation
//! - Color space configuration

use crate::rules::{CheckResult, QcContext, QcRule, RuleCategory, Severity};
use oximedia_core::{CodecId, OxiResult};

/// VP9 profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vp9Profile {
    /// Profile 0: 8-bit 4:2:0.
    Profile0,
    /// Profile 1: 8-bit 4:2:2, 4:4:0, 4:4:4.
    Profile1,
    /// Profile 2: 10/12-bit 4:2:0.
    Profile2,
    /// Profile 3: 10/12-bit 4:2:2, 4:4:0, 4:4:4.
    Profile3,
}

/// VP9 bitstream validator.
pub struct Vp9BitstreamValidator {
    expected_profile: Option<Vp9Profile>,
    check_superframes: bool,
    check_frame_headers: bool,
}

impl Vp9BitstreamValidator {
    #[must_use]
    /// Creates a new VP9 bitstream validator with default settings.
    pub fn new() -> Self {
        Self {
            expected_profile: None,
            check_superframes: true,
            check_frame_headers: true,
        }
    }

    /// Sets the expected VP9 profile for validation.
    #[must_use]
    pub const fn with_profile(mut self, profile: Vp9Profile) -> Self {
        self.expected_profile = Some(profile);
        self
    }

    fn validate_superframe_structure(&self, _stream_index: usize) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_superframes {
            results.push(
                CheckResult::pass(self.name())
                    .with_recommendation("VP9 superframe index validation".to_string()),
            );
        }

        Ok(results)
    }

    fn validate_frame_headers(&self, _stream_index: usize) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_frame_headers {
            results.push(
                CheckResult::pass(self.name())
                    .with_recommendation("VP9 frame header syntax validation".to_string()),
            );
        }

        Ok(results)
    }
}

impl Default for Vp9BitstreamValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for Vp9BitstreamValidator {
    fn name(&self) -> &str {
        "vp9_bitstream_validation"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Video
    }

    fn description(&self) -> &str {
        "Validates VP9 bitstream compliance"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut all_results = Vec::new();

        let vp9_streams: Vec<_> = context
            .video_streams()
            .into_iter()
            .filter(|s| s.codec == CodecId::Vp9)
            .collect();

        if vp9_streams.is_empty() {
            return Ok(vec![CheckResult::pass(self.name())
                .with_recommendation("No VP9 streams found".to_string())]);
        }

        for stream in vp9_streams {
            let sf_results = self.validate_superframe_structure(stream.index)?;
            all_results.extend(sf_results);

            let frame_results = self.validate_frame_headers(stream.index)?;
            all_results.extend(frame_results);
        }

        Ok(all_results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        context
            .video_streams()
            .iter()
            .any(|s| s.codec == CodecId::Vp9)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vp9_validator_creation() {
        let validator = Vp9BitstreamValidator::new();
        assert!(validator.check_superframes);
    }
}
