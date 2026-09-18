//! Continuity checking and reporting.

pub mod check;
pub mod continuity_errors;
pub mod report;

pub use check::{ContinuityChecker, ContinuityIssue, IssueType, Severity};
pub use continuity_errors::{
    AxisViolation, ContinuityError, ContinuityErrorType, ContinuityReport as ContinuityErrorReport,
    CoverageType as ErrorCoverageType, JumpCutDetector, ShotData,
};
pub use report::{ContinuityReport, ContinuityReporter};
