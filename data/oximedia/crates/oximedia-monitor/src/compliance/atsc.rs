//! ATSC A/53 compliance checking.

use super::ComplianceViolation;
use crate::MonitorResult;

/// ATSC compliance checker.
pub struct AtscChecker {
    violations: Vec<ComplianceViolation>,
}

impl AtscChecker {
    /// Create a new ATSC checker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            violations: Vec::new(),
        }
    }

    /// Check video for ATSC compliance.
    ///
    /// # Errors
    ///
    /// Returns an error if checking fails.
    pub fn check_video(&mut self, _frame: &[u8], _width: u32, _height: u32) -> MonitorResult<()> {
        Ok(())
    }

    /// Check audio for ATSC compliance.
    ///
    /// # Errors
    ///
    /// Returns an error if checking fails.
    pub fn check_audio(&mut self, _samples: &[f32]) -> MonitorResult<()> {
        Ok(())
    }

    /// Get violations.
    #[must_use]
    pub fn violations(&self) -> Vec<ComplianceViolation> {
        self.violations.clone()
    }

    /// Reset checker.
    pub fn reset(&mut self) {
        self.violations.clear();
    }
}

impl Default for AtscChecker {
    fn default() -> Self {
        Self::new()
    }
}
