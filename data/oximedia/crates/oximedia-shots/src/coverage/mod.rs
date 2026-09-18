//! Coverage analysis.

pub mod analyze;
pub mod coverage_map;
pub mod master;
pub mod single;
pub mod twoshot;

pub use analyze::{CoverageAnalyzer, CoverageReport};
pub use coverage_map::{
    CoverageAnalyzer as CoverageMapAnalyzer, CoverageMap, CoverageReport as CoverageMapReport,
    CoverageType as CoverageMapType, EyelineChecker, EyelineIssue, IssueSeverity, ShotAngle,
};
pub use master::MasterDetector;
pub use single::SingleDetector;
pub use twoshot::TwoShotDetector;
