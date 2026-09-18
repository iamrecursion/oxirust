// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Compatibility validation for file concatenation.

use crate::{ConversionError, Result};
use std::path::Path;

/// Validator for checking if files can be concatenated.
#[derive(Debug, Clone)]
pub struct CompatibilityValidator {
    strict_mode: bool,
}

impl CompatibilityValidator {
    /// Create a new compatibility validator.
    #[must_use]
    pub fn new() -> Self {
        Self { strict_mode: true }
    }

    /// Set strict mode (requires exact codec/format match).
    #[must_use]
    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }

    /// Validate that files can be concatenated.
    pub fn validate<P: AsRef<Path>>(&self, files: &[P]) -> Result<ValidationReport> {
        if files.len() < 2 {
            return Err(ConversionError::ValidationFailed(
                "Need at least 2 files to concatenate".to_string(),
            ));
        }

        let issues: Vec<ValidationIssue> = Vec::new();
        let warnings: Vec<String> = Vec::new();

        // Placeholder for actual validation
        // In a real implementation, this would:
        // 1. Check codec compatibility
        // 2. Check resolution consistency
        // 3. Check frame rate consistency
        // 4. Check audio parameters

        if self.strict_mode && !issues.is_empty() {
            return Err(ConversionError::ValidationFailed(format!(
                "Validation failed with {} issues",
                issues.len()
            )));
        }

        let requires_reencode = !issues.is_empty();
        Ok(ValidationReport {
            compatible: issues.is_empty(),
            issues,
            warnings,
            requires_reencode,
        })
    }

    /// Quick check if files are compatible.
    pub fn are_compatible<P: AsRef<Path>>(&self, files: &[P]) -> bool {
        self.validate(files).map(|r| r.compatible).unwrap_or(false)
    }

    /// Check if files need re-encoding to concatenate.
    pub fn requires_reencode<P: AsRef<Path>>(&self, files: &[P]) -> bool {
        self.validate(files)
            .map(|r| r.requires_reencode)
            .unwrap_or(true)
    }
}

impl Default for CompatibilityValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Report from compatibility validation.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Whether files are compatible
    pub compatible: bool,
    /// Compatibility issues found
    pub issues: Vec<ValidationIssue>,
    /// Warnings (non-blocking)
    pub warnings: Vec<String>,
    /// Whether re-encoding is required
    pub requires_reencode: bool,
}

impl ValidationReport {
    /// Check if validation passed without issues.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.compatible && self.issues.is_empty()
    }

    /// Get a summary of the validation.
    #[must_use]
    pub fn summary(&self) -> String {
        if self.is_ok() {
            "Files are compatible for concatenation".to_string()
        } else {
            format!(
                "Found {} issues, {} warnings. Re-encoding {}",
                self.issues.len(),
                self.warnings.len(),
                if self.requires_reencode {
                    "required"
                } else {
                    "not required"
                }
            )
        }
    }
}

/// A validation issue.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Issue type
    pub kind: IssueKind,
    /// Issue description
    pub description: String,
    /// File index where issue was found
    pub file_index: usize,
}

/// Type of validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueKind {
    /// Codec mismatch
    CodecMismatch,
    /// Resolution mismatch
    ResolutionMismatch,
    /// Frame rate mismatch
    FrameRateMismatch,
    /// Audio parameters mismatch
    AudioMismatch,
    /// Format mismatch
    FormatMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = CompatibilityValidator::new();
        assert!(validator.strict_mode);
    }

    #[test]
    fn test_validator_settings() {
        let validator = CompatibilityValidator::new().with_strict_mode(false);

        assert!(!validator.strict_mode);
    }

    #[test]
    fn test_validation_report() {
        let report = ValidationReport {
            compatible: true,
            issues: Vec::new(),
            warnings: Vec::new(),
            requires_reencode: false,
        };

        assert!(report.is_ok());
        assert!(report.summary().contains("compatible"));

        let report = ValidationReport {
            compatible: false,
            issues: vec![ValidationIssue {
                kind: IssueKind::CodecMismatch,
                description: "Different codecs".to_string(),
                file_index: 1,
            }],
            warnings: Vec::new(),
            requires_reencode: true,
        };

        assert!(!report.is_ok());
        assert!(report.summary().contains("1 issues"));
    }
}
