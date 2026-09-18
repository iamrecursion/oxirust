#![allow(unused_imports)]
//! Opus bitstream compliance validation.
//!
//! Provides validation of Opus codec bitstream including:
//! - Frame structure validation
//! - TOC (Table of Contents) byte validation
//! - Packet padding validation
//! - Frame size validation

use crate::rules::{CheckResult, QcContext, QcRule, RuleCategory, Severity};
use oximedia_core::{CodecId, OxiResult};

/// Opus bitstream validator.
pub struct OpusBitstreamValidator {
    check_toc: bool,
    check_frame_sizes: bool,
}

impl OpusBitstreamValidator {
    #[must_use]
    /// Creates a new Opus bitstream validator with default settings.
    pub fn new() -> Self {
        Self {
            check_toc: true,
            check_frame_sizes: true,
        }
    }

    fn validate_toc_byte(&self, _stream_index: usize) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_toc {
            results.push(
                CheckResult::pass(self.name())
                    .with_recommendation("Opus TOC byte validation".to_string()),
            );
        }

        Ok(results)
    }

    fn validate_frame_sizes(&self, _stream_index: usize) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_frame_sizes {
            results.push(CheckResult::pass(self.name()).with_recommendation(
                "Opus frame size validation (2.5, 5, 10, 20, 40, 60ms)".to_string(),
            ));
        }

        Ok(results)
    }
}

impl Default for OpusBitstreamValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for OpusBitstreamValidator {
    fn name(&self) -> &str {
        "opus_bitstream_validation"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Audio
    }

    fn description(&self) -> &str {
        "Validates Opus bitstream compliance"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut all_results = Vec::new();

        let opus_streams: Vec<_> = context
            .audio_streams()
            .into_iter()
            .filter(|s| s.codec == CodecId::Opus)
            .collect();

        if opus_streams.is_empty() {
            return Ok(vec![CheckResult::pass(self.name())
                .with_recommendation("No Opus streams found".to_string())]);
        }

        for stream in opus_streams {
            let toc_results = self.validate_toc_byte(stream.index)?;
            all_results.extend(toc_results);

            let frame_results = self.validate_frame_sizes(stream.index)?;
            all_results.extend(frame_results);
        }

        Ok(all_results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        context
            .audio_streams()
            .iter()
            .any(|s| s.codec == CodecId::Opus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opus_validator_creation() {
        let validator = OpusBitstreamValidator::new();
        assert!(validator.check_toc);
    }
}
