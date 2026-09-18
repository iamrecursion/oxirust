//! Validation report generation.

/// Validation report.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Total number of links checked.
    pub total_links: usize,

    /// Number of valid links.
    pub valid_links: usize,

    /// List of errors found.
    pub errors: Vec<String>,

    /// List of warnings.
    pub warnings: Vec<String>,
}

impl ValidationReport {
    /// Check if validation passed (no errors).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get the number of errors.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Get the number of warnings.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }

    /// Generate a summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Validation: {} valid / {} total - {} errors, {} warnings",
            self.valid_links,
            self.total_links,
            self.error_count(),
            self.warning_count()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_report() {
        let report = ValidationReport {
            total_links: 10,
            valid_links: 8,
            errors: vec!["Error 1".to_string()],
            warnings: vec!["Warning 1".to_string()],
        };

        assert!(!report.is_valid());
        assert_eq!(report.error_count(), 1);
        assert_eq!(report.warning_count(), 1);
    }

    #[test]
    fn test_valid_report() {
        let report = ValidationReport {
            total_links: 5,
            valid_links: 5,
            errors: Vec::new(),
            warnings: Vec::new(),
        };

        assert!(report.is_valid());
        assert_eq!(report.error_count(), 0);
    }
}
