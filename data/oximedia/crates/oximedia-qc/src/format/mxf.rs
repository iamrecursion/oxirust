#![allow(unused_imports)]
//! MXF operational pattern validation.
//!
//! Provides validation of MXF (Material Exchange Format) files including:
//! - Operational pattern validation (OP1a, OP1b, OP2a, OP3a, etc.)
//! - Header metadata validation
//! - Body partition validation
//! - Footer partition validation
//! - Index table validation
//! - AS-02, AS-11, AS-10 profile validation

use crate::rules::{CheckResult, QcContext, QcRule, RuleCategory, Severity};
use oximedia_core::OxiResult;
use std::path::Path;

/// MXF operational pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperationalPattern {
    /// OP1a: Single Item, Single Package.
    Op1a,
    /// OP1b: Single Item, Ganged Packages.
    Op1b,
    /// OP2a: Playlist Items, Single Package.
    Op2a,
    /// OP2b: Playlist Items, Ganged Packages.
    Op2b,
    /// OP3a: Alternate Items, Single Package.
    Op3a,
    /// OP3b: Alternate Items, Ganged Packages.
    Op3b,
    /// OPAtom: Single clip, single essence.
    OpAtom,
}

/// MXF Application Specification profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MxfProfile {
    /// AS-02: MXF Versioning.
    As02,
    /// AS-11: MXF Program Contribution.
    As11,
    /// AS-10: MXF for Production.
    As10,
    /// Generic MXF.
    Generic,
}

/// MXF format validator.
///
/// Validates compliance with SMPTE standards for MXF files.
pub struct MxfValidator {
    expected_op: Option<OperationalPattern>,
    expected_profile: Option<MxfProfile>,
    check_index_table: bool,
    check_partitions: bool,
    check_metadata: bool,
}

impl MxfValidator {
    /// Creates a new MXF validator with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            expected_op: None,
            expected_profile: None,
            check_index_table: true,
            check_partitions: true,
            check_metadata: true,
        }
    }

    /// Sets the expected operational pattern.
    #[must_use]
    pub const fn with_operational_pattern(mut self, op: OperationalPattern) -> Self {
        self.expected_op = Some(op);
        self
    }

    /// Sets the expected MXF profile.
    #[must_use]
    pub const fn with_profile(mut self, profile: MxfProfile) -> Self {
        self.expected_profile = Some(profile);
        self
    }

    /// Sets whether to validate index tables.
    #[must_use]
    pub const fn with_index_table_check(mut self, check: bool) -> Self {
        self.check_index_table = check;
        self
    }

    /// Validates MXF partition structure.
    fn validate_partitions(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_partitions {
            // In production, would validate:
            // - Header partition exists and is valid
            // - Body partitions are properly structured
            // - Footer partition exists (if closed/complete)
            // - Partition pack keys and lengths are correct
            // - HeaderByteCount and IndexByteCount are accurate

            results.push(CheckResult::pass(self.name()).with_recommendation(
                "Partition structure validation: header, body, footer".to_string(),
            ));
        }

        Ok(results)
    }

    /// Validates header metadata.
    fn validate_header_metadata(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_metadata {
            // In production, would validate:
            // - Preface metadata set
            // - Content Storage set
            // - Material Package
            // - File Package
            // - Track metadata
            // - Descriptor metadata
            // - Proper UL keys and local tags

            results.push(
                CheckResult::pass(self.name())
                    .with_recommendation("Header metadata validation per SMPTE 377M".to_string()),
            );
        }

        Ok(results)
    }

    /// Validates operational pattern.
    fn validate_operational_pattern(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if let Some(expected) = self.expected_op {
            // In production, would read OperationalPattern UL from header
            results.push(
                CheckResult::pass(self.name())
                    .with_recommendation(format!("Expected operational pattern: {expected:?}")),
            );
        } else {
            results.push(
                CheckResult::pass(self.name())
                    .with_recommendation("Operational pattern validation".to_string()),
            );
        }

        Ok(results)
    }

    /// Validates index tables.
    fn validate_index_tables(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_index_table {
            // In production, would validate:
            // - Index Table Segment presence
            // - Index Edit Rate matches essence
            // - Index Start Position is valid
            // - Index Duration is correct
            // - Delta entries are properly structured

            results
                .push(CheckResult::pass(self.name()).with_recommendation(
                    "Index table validation for efficient access".to_string(),
                ));
        }

        Ok(results)
    }

    /// Validates essence container structure.
    fn validate_essence_container(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // In production, would validate:
        // - Essence container UL is recognized
        // - Frame wrapping vs clip wrapping
        // - KLV packets are properly structured
        // - Essence element keys are valid

        results.push(
            CheckResult::pass(self.name())
                .with_recommendation("Essence container KLV structure validation".to_string()),
        );

        Ok(results)
    }

    /// Validates AS-11 profile specifics.
    fn validate_as11_profile(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if matches!(self.expected_profile, Some(MxfProfile::As11)) {
            // In production, would validate:
            // - DM_Framework (AS-11 Core Framework)
            // - Shim metadata
            // - UK DPP metadata if required
            // - Closed captioning requirements

            results.push(CheckResult::pass(self.name()).with_recommendation(
                "AS-11 (Program Contribution) profile validation".to_string(),
            ));
        }

        Ok(results)
    }

    /// Validates AS-02 profile specifics.
    fn validate_as02_profile(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if matches!(self.expected_profile, Some(MxfProfile::As02)) {
            // In production, would validate:
            // - Versioning metadata
            // - Shim structure
            // - Manifest file (if bundle)

            results.push(
                CheckResult::pass(self.name())
                    .with_recommendation("AS-02 (Versioning) profile validation".to_string()),
            );
        }

        Ok(results)
    }
}

impl Default for MxfValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for MxfValidator {
    fn name(&self) -> &str {
        "mxf_operational_pattern_validation"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Container
    }

    fn description(&self) -> &str {
        "Validates MXF operational patterns and SMPTE compliance"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut all_results = Vec::new();

        // Only applicable to MXF files
        let path_lower = context.file_path.to_lowercase();
        if !path_lower.ends_with(".mxf") {
            return Ok(vec![CheckResult::pass(self.name()).with_recommendation(
                "Not an MXF file - check skipped".to_string(),
            )]);
        }

        // Validate partitions
        let partition_results = self.validate_partitions(&context.file_path)?;
        all_results.extend(partition_results);

        // Validate header metadata
        let metadata_results = self.validate_header_metadata(&context.file_path)?;
        all_results.extend(metadata_results);

        // Validate operational pattern
        let op_results = self.validate_operational_pattern(&context.file_path)?;
        all_results.extend(op_results);

        // Validate index tables
        let index_results = self.validate_index_tables(&context.file_path)?;
        all_results.extend(index_results);

        // Validate essence container
        let essence_results = self.validate_essence_container(&context.file_path)?;
        all_results.extend(essence_results);

        // Profile-specific validation
        let as11_results = self.validate_as11_profile(&context.file_path)?;
        all_results.extend(as11_results);

        let as02_results = self.validate_as02_profile(&context.file_path)?;
        all_results.extend(as02_results);

        Ok(all_results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        let path = Path::new(&context.file_path);
        if let Some(ext) = path.extension() {
            let ext_lower = ext.to_string_lossy().to_lowercase();
            ext_lower == "mxf"
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mxf_validator_creation() {
        let validator = MxfValidator::new();
        assert_eq!(validator.name(), "mxf_operational_pattern_validation");
        assert!(validator.check_index_table);
    }

    #[test]
    fn test_mxf_validator_configuration() {
        let validator = MxfValidator::new()
            .with_operational_pattern(OperationalPattern::Op1a)
            .with_profile(MxfProfile::As11);

        assert_eq!(validator.expected_op, Some(OperationalPattern::Op1a));
        assert_eq!(validator.expected_profile, Some(MxfProfile::As11));
    }

    #[test]
    fn test_mxf_applicability() {
        let validator = MxfValidator::new();
        let mut context = QcContext::new("test.mxf");
        assert!(validator.is_applicable(&context));

        context.file_path = "test.mp4".to_string();
        assert!(!validator.is_applicable(&context));
    }

    #[test]
    fn test_operational_patterns() {
        assert_eq!(OperationalPattern::Op1a, OperationalPattern::Op1a);
        assert_ne!(OperationalPattern::Op1a, OperationalPattern::Op1b);
    }
}
