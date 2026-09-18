//! Broadcast compliance checking.

pub mod atsc;
pub mod ebu;
pub mod report;
pub mod smpte;

use crate::MonitorResult;
use serde::{Deserialize, Serialize};

pub use atsc::AtscChecker;
pub use ebu::EbuChecker;
pub use report::ComplianceReportGenerator;
pub use smpte::SmpteChecker;

/// Compliance standard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceStandard {
    /// EBU R128 (European Broadcasting Union).
    EbuR128,

    /// SMPTE standards.
    Smpte,

    /// ATSC (US broadcast).
    Atsc,
}

/// Compliance violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    /// Violation type.
    pub violation_type: String,

    /// Severity.
    pub severity: ViolationSeverity,

    /// Description.
    pub description: String,

    /// Frame number or timestamp.
    pub timestamp: u64,
}

/// Violation severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Information only.
    Info,

    /// Warning.
    Warning,

    /// Error.
    Error,

    /// Critical error.
    Critical,
}

/// Compliance report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Standard being checked.
    pub standard: Option<ComplianceStandard>,

    /// Is compliant?
    pub is_compliant: bool,

    /// Violations found.
    pub violations: Vec<ComplianceViolation>,

    /// Audio compliant?
    pub audio_compliant: bool,

    /// Video compliant?
    pub video_compliant: bool,
}

/// Compliance checker.
pub struct ComplianceChecker {
    standard: ComplianceStandard,
    ebu_checker: EbuChecker,
    smpte_checker: SmpteChecker,
    atsc_checker: AtscChecker,
    violations: Vec<ComplianceViolation>,
}

impl ComplianceChecker {
    /// Create a new compliance checker.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn new(standard: ComplianceStandard) -> MonitorResult<Self> {
        Ok(Self {
            standard,
            ebu_checker: EbuChecker::new(),
            smpte_checker: SmpteChecker::new(),
            atsc_checker: AtscChecker::new(),
            violations: Vec::new(),
        })
    }

    /// Check video frame for compliance.
    ///
    /// # Errors
    ///
    /// Returns an error if checking fails.
    pub fn check_video_frame(
        &mut self,
        frame: &[u8],
        width: u32,
        height: u32,
    ) -> MonitorResult<()> {
        match self.standard {
            ComplianceStandard::EbuR128 => {
                self.ebu_checker.check_video(frame, width, height)?;
            }
            ComplianceStandard::Smpte => {
                self.smpte_checker.check_video(frame, width, height)?;
            }
            ComplianceStandard::Atsc => {
                self.atsc_checker.check_video(frame, width, height)?;
            }
        }
        Ok(())
    }

    /// Check audio samples for compliance.
    ///
    /// # Errors
    ///
    /// Returns an error if checking fails.
    pub fn check_audio_samples(&mut self, samples: &[f32]) -> MonitorResult<()> {
        match self.standard {
            ComplianceStandard::EbuR128 => {
                self.ebu_checker.check_audio(samples)?;
            }
            ComplianceStandard::Smpte => {
                self.smpte_checker.check_audio(samples)?;
            }
            ComplianceStandard::Atsc => {
                self.atsc_checker.check_audio(samples)?;
            }
        }
        Ok(())
    }

    /// Generate compliance report.
    pub fn generate_report(&mut self) -> ComplianceReport {
        let violations = match self.standard {
            ComplianceStandard::EbuR128 => self.ebu_checker.violations(),
            ComplianceStandard::Smpte => self.smpte_checker.violations(),
            ComplianceStandard::Atsc => self.atsc_checker.violations(),
        };

        ComplianceReport {
            standard: Some(self.standard),
            is_compliant: violations.is_empty(),
            violations,
            audio_compliant: true,
            video_compliant: true,
        }
    }

    /// Reset checker.
    pub fn reset(&mut self) {
        self.ebu_checker.reset();
        self.smpte_checker.reset();
        self.atsc_checker.reset();
        self.violations.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compliance_checker() {
        let result = ComplianceChecker::new(ComplianceStandard::EbuR128);
        assert!(result.is_ok());
    }
}
