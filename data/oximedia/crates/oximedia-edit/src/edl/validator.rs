//! EDL validation.

use super::{Edl, EdlResult};

/// EDL validation report.
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    /// Validation errors.
    pub errors: Vec<String>,
    /// Validation warnings.
    pub warnings: Vec<String>,
}

impl ValidationReport {
    /// Returns `true` if there are no errors.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

/// EDL validator.
#[derive(Debug, Default)]
pub struct EdlValidator;

impl EdlValidator {
    /// Create a new validator.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Validate an EDL and return a report.
    pub fn validate(&self, edl: &Edl) -> EdlResult<ValidationReport> {
        let mut report = ValidationReport::default();

        if edl.title.is_empty() {
            report.warnings.push("EDL has no title".to_string());
        }

        if edl.events.is_empty() {
            report.warnings.push("EDL has no events".to_string());
        }

        // Check for duplicate event numbers
        let mut seen = std::collections::HashSet::new();
        for event in &edl.events {
            if !seen.insert(event.number) {
                report
                    .errors
                    .push(format!("Duplicate event number: {}", event.number));
            }
        }

        Ok(report)
    }
}
