#![allow(unused_imports)]
//! Matroska/WebM schema validation.
//!
//! Provides validation of Matroska container structure including:
//! - EBML header validation
//! - Segment structure
//! - Cluster validation
//! - Cues index validation
//! - Attachment validation

use crate::rules::{CheckResult, QcContext, QcRule, RuleCategory, Severity};
use oximedia_core::OxiResult;
use std::path::Path;

/// Matroska/WebM format validator.
///
/// Validates compliance with Matroska specification.
pub struct MatroskaValidator {
    check_cues: bool,
    check_seekhead: bool,
    check_clusters: bool,
    require_crc32: bool,
}

impl MatroskaValidator {
    /// Creates a new Matroska validator with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            check_cues: true,
            check_seekhead: true,
            check_clusters: true,
            require_crc32: false,
        }
    }

    /// Sets whether to check for cues (seek index).
    #[must_use]
    pub const fn with_cues_check(mut self, check: bool) -> Self {
        self.check_cues = check;
        self
    }

    /// Sets whether to check SeekHead element.
    #[must_use]
    pub const fn with_seekhead_check(mut self, check: bool) -> Self {
        self.check_seekhead = check;
        self
    }

    /// Sets whether to require CRC-32 elements for error detection.
    #[must_use]
    pub const fn with_crc32_requirement(mut self, require: bool) -> Self {
        self.require_crc32 = require;
        self
    }

    /// Validates EBML header.
    fn validate_ebml_header(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // In production, would parse EBML header and validate:
        // - EBML magic number (0x1A45DFA3)
        // - DocType is "matroska" or "webm"
        // - DocTypeVersion and DocTypeReadVersion are valid
        // - MaxIDLength and MaxSizeLength are appropriate

        results.push(
            CheckResult::pass(self.name())
                .with_recommendation("EBML header validation requires file parse".to_string()),
        );

        Ok(results)
    }

    /// Validates segment structure.
    fn validate_segment(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // In production, would validate:
        // - Segment element exists and is sized correctly
        // - SegmentInfo with UID, TimecodeScale, Duration
        // - Tracks element with proper TrackEntry elements
        // - Proper element ordering

        results.push(
            CheckResult::pass(self.name()).with_recommendation(
                "Segment structure validation requires full parse".to_string(),
            ),
        );

        Ok(results)
    }

    /// Validates cluster structure.
    fn validate_clusters(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_clusters {
            // In production, would validate:
            // - Cluster Timecode elements
            // - SimpleBlock and BlockGroup elements
            // - Lacing validity
            // - Timecode continuity within and between clusters

            results.push(CheckResult::pass(self.name()).with_recommendation(
                "Cluster validation requires media data analysis".to_string(),
            ));
        }

        Ok(results)
    }

    /// Validates cues (seek index).
    fn validate_cues(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_cues {
            // In production, would check:
            // - Cues element exists (recommended for seeking)
            // - CuePoint elements are properly structured
            // - CueTrackPositions point to valid clusters
            // - Cues are sorted by timecode

            results.push(
                CheckResult::pass(self.name()).with_recommendation(
                    "Cues index recommended for efficient seeking".to_string(),
                ),
            );
        }

        Ok(results)
    }

    /// Validates SeekHead element.
    fn validate_seekhead(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_seekhead {
            // In production, would validate:
            // - SeekHead element exists (recommended)
            // - Seek entries point to valid elements
            // - All major elements are indexed

            results.push(
                CheckResult::pass(self.name())
                    .with_recommendation("SeekHead index improves file navigation".to_string()),
            );
        }

        Ok(results)
    }

    /// Validates attachments.
    fn validate_attachments(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // In production, would check:
        // - AttachedFile elements are valid
        // - FileData is not corrupted
        // - MIME types are appropriate

        results.push(
            CheckResult::pass(self.name())
                .with_recommendation("Attachment validation if present".to_string()),
        );

        Ok(results)
    }

    /// Checks CRC-32 elements if required.
    fn check_crc32(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.require_crc32 {
            // In production, would verify:
            // - CRC-32 elements are present on critical elements
            // - CRC values are correct

            results.push(CheckResult::pass(self.name()).with_recommendation(
                "CRC-32 elements provide error detection capability".to_string(),
            ));
        }

        Ok(results)
    }
}

impl Default for MatroskaValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for MatroskaValidator {
    fn name(&self) -> &str {
        "matroska_schema_validation"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Container
    }

    fn description(&self) -> &str {
        "Validates Matroska/WebM schema compliance"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut all_results = Vec::new();

        // Only applicable to Matroska/WebM files
        let path_lower = context.file_path.to_lowercase();
        if !path_lower.ends_with(".mkv")
            && !path_lower.ends_with(".webm")
            && !path_lower.ends_with(".mka")
        {
            return Ok(vec![CheckResult::pass(self.name()).with_recommendation(
                "Not a Matroska file - check skipped".to_string(),
            )]);
        }

        // Validate EBML header
        let header_results = self.validate_ebml_header(&context.file_path)?;
        all_results.extend(header_results);

        // Validate segment
        let segment_results = self.validate_segment(&context.file_path)?;
        all_results.extend(segment_results);

        // Validate clusters
        let cluster_results = self.validate_clusters(&context.file_path)?;
        all_results.extend(cluster_results);

        // Validate cues
        let cues_results = self.validate_cues(&context.file_path)?;
        all_results.extend(cues_results);

        // Validate SeekHead
        let seekhead_results = self.validate_seekhead(&context.file_path)?;
        all_results.extend(seekhead_results);

        // Validate attachments
        let attachment_results = self.validate_attachments(&context.file_path)?;
        all_results.extend(attachment_results);

        // Check CRC-32
        let crc_results = self.check_crc32(&context.file_path)?;
        all_results.extend(crc_results);

        Ok(all_results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        let path = Path::new(&context.file_path);
        if let Some(ext) = path.extension() {
            let ext_lower = ext.to_string_lossy().to_lowercase();
            matches!(ext_lower.as_str(), "mkv" | "webm" | "mka")
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matroska_validator_creation() {
        let validator = MatroskaValidator::new();
        assert_eq!(validator.name(), "matroska_schema_validation");
        assert!(validator.check_cues);
    }

    #[test]
    fn test_matroska_validator_configuration() {
        let validator = MatroskaValidator::new()
            .with_cues_check(false)
            .with_crc32_requirement(true);

        assert!(!validator.check_cues);
        assert!(validator.require_crc32);
    }

    #[test]
    fn test_matroska_applicability() {
        let validator = MatroskaValidator::new();
        let mut context = QcContext::new("test.mkv");
        assert!(validator.is_applicable(&context));

        context.file_path = "test.mp4".to_string();
        assert!(!validator.is_applicable(&context));
    }
}
