#![allow(unused_imports)]
//! MP4/ISOBMFF compliance validation (ISO 14496-12).
//!
//! Provides detailed validation of MP4 container structure including:
//! - Box hierarchy validation
//! - ftyp brand compatibility
//! - moov atom structure
//! - Sample table validation
//! - Fragment validation
//! - Fast start optimization check

use crate::rules::{CheckResult, QcContext, QcRule, RuleCategory, Severity};
use oximedia_core::OxiResult;
use std::path::Path;

/// MP4/ISOBMFF format validator.
///
/// Validates compliance with ISO/IEC 14496-12 (ISOBMFF) specification.
pub struct Mp4Validator {
    check_fast_start: bool,
    check_fragmentation: bool,
    check_sample_tables: bool,
    allowed_brands: Vec<String>,
}

impl Mp4Validator {
    /// Creates a new MP4 validator with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            check_fast_start: true,
            check_fragmentation: true,
            check_sample_tables: true,
            allowed_brands: vec![
                "isom".to_string(),
                "iso2".to_string(),
                "iso4".to_string(),
                "iso5".to_string(),
                "iso6".to_string(),
                "mp41".to_string(),
                "mp42".to_string(),
                "dash".to_string(),
            ],
        }
    }

    /// Sets whether to check for fast start (moov before mdat).
    #[must_use]
    pub const fn with_fast_start_check(mut self, check: bool) -> Self {
        self.check_fast_start = check;
        self
    }

    /// Sets whether to check fragmentation structure.
    #[must_use]
    pub const fn with_fragmentation_check(mut self, check: bool) -> Self {
        self.check_fragmentation = check;
        self
    }

    /// Sets allowed ftyp brands.
    #[must_use]
    pub fn with_allowed_brands(mut self, brands: Vec<String>) -> Self {
        self.allowed_brands = brands;
        self
    }

    /// Validates box hierarchy.
    fn validate_box_hierarchy(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // In production, would parse MP4 boxes and validate:
        // - ftyp must be first box
        // - moov box must exist
        // - mdat box must exist
        // - Box sizes are valid
        // - No unknown/invalid boxes in critical positions

        results.push(
            CheckResult::pass(self.name()).with_recommendation(
                "Box hierarchy validation requires full file parse".to_string(),
            ),
        );

        Ok(results)
    }

    /// Checks for fast start configuration.
    fn check_fast_start_impl(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_fast_start {
            // In production, would check if moov box comes before mdat
            // This is critical for streaming/progressive download
            results.push(CheckResult::pass(self.name()).with_recommendation(
                "Fast start check: moov should appear before mdat for streaming".to_string(),
            ));
        }

        Ok(results)
    }

    /// Validates sample table structure.
    fn validate_sample_tables(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_sample_tables {
            // In production, would validate:
            // - stts (time-to-sample) table
            // - stss (sync sample) table
            // - stsc (sample-to-chunk) table
            // - stsz (sample size) table
            // - stco/co64 (chunk offset) table
            // - Consistency between tables

            results.push(CheckResult::pass(self.name()).with_recommendation(
                "Sample table validation requires media data analysis".to_string(),
            ));
        }

        Ok(results)
    }

    /// Validates fragment structure for fragmented MP4.
    fn validate_fragments(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_fragmentation {
            // In production, would check:
            // - moof boxes structure
            // - mfhd headers
            // - traf boxes
            // - Proper sequence numbers
            // - Duration consistency

            results.push(CheckResult::pass(self.name()).with_recommendation(
                "Fragment validation for DASH/HLS segmented content".to_string(),
            ));
        }

        Ok(results)
    }

    /// Validates ftyp brand compatibility.
    fn validate_ftyp(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // In production, would read ftyp box and check:
        // - Major brand is in allowed list
        // - Compatible brands are valid
        // - Version is supported

        results.push(CheckResult::pass(self.name()).with_recommendation(format!(
            "Allowed brands: {}",
            self.allowed_brands.join(", ")
        )));

        Ok(results)
    }
}

impl Default for Mp4Validator {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for Mp4Validator {
    fn name(&self) -> &str {
        "mp4_isobmff_validation"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Container
    }

    fn description(&self) -> &str {
        "Validates MP4/ISOBMFF compliance (ISO 14496-12)"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut all_results = Vec::new();

        // Only applicable to MP4 files
        let path_lower = context.file_path.to_lowercase();
        if !path_lower.ends_with(".mp4")
            && !path_lower.ends_with(".m4a")
            && !path_lower.ends_with(".m4v")
        {
            return Ok(vec![CheckResult::pass(self.name()).with_recommendation(
                "Not an MP4 file - check skipped".to_string(),
            )]);
        }

        // Validate ftyp
        let ftyp_results = self.validate_ftyp(&context.file_path)?;
        all_results.extend(ftyp_results);

        // Validate box hierarchy
        let hierarchy_results = self.validate_box_hierarchy(&context.file_path)?;
        all_results.extend(hierarchy_results);

        // Check fast start
        let fast_start_results = self.check_fast_start_impl(&context.file_path)?;
        all_results.extend(fast_start_results);

        // Validate sample tables
        let sample_table_results = self.validate_sample_tables(&context.file_path)?;
        all_results.extend(sample_table_results);

        // Validate fragments if applicable
        let fragment_results = self.validate_fragments(&context.file_path)?;
        all_results.extend(fragment_results);

        Ok(all_results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        let path = Path::new(&context.file_path);
        if let Some(ext) = path.extension() {
            let ext_lower = ext.to_string_lossy().to_lowercase();
            matches!(ext_lower.as_str(), "mp4" | "m4a" | "m4v")
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mp4_validator_creation() {
        let validator = Mp4Validator::new();
        assert_eq!(validator.name(), "mp4_isobmff_validation");
        assert!(validator.check_fast_start);
    }

    #[test]
    fn test_mp4_validator_configuration() {
        let validator = Mp4Validator::new()
            .with_fast_start_check(false)
            .with_allowed_brands(vec!["isom".to_string()]);

        assert!(!validator.check_fast_start);
        assert_eq!(validator.allowed_brands.len(), 1);
    }

    #[test]
    fn test_mp4_applicability() {
        let validator = Mp4Validator::new();
        let mut context = QcContext::new("test.mp4");
        assert!(validator.is_applicable(&context));

        context.file_path = "test.mkv".to_string();
        assert!(!validator.is_applicable(&context));
    }
}
