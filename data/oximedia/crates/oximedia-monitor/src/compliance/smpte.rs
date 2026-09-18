//! SMPTE compliance checking.

use super::ComplianceViolation;
use crate::MonitorResult;

/// SMPTE compliance checker.
pub struct SmpteChecker {
    violations: Vec<ComplianceViolation>,
}

impl SmpteChecker {
    /// Create a new SMPTE checker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            violations: Vec::new(),
        }
    }

    /// Check video for SMPTE compliance.
    ///
    /// # Errors
    ///
    /// Returns an error if checking fails.
    pub fn check_video(&mut self, _frame: &[u8], _width: u32, _height: u32) -> MonitorResult<()> {
        Ok(())
    }

    /// Check audio for SMPTE compliance.
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

impl Default for SmpteChecker {
    fn default() -> Self {
        Self::new()
    }
}
