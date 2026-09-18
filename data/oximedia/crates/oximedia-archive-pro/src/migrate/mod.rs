//! Format migration planning and execution

pub mod execute;
pub mod planner;
pub mod validate;

pub use execute::{MigrationExecutor, MigrationResult};
pub use planner::{MigrationPlan, MigrationPlanner, MigrationStrategy};
pub use validate::{MigrationValidator, ValidationResult};

use serde::{Deserialize, Serialize};

/// Migration priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum MigrationPriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical - immediate action required
    Critical,
}

impl MigrationPriority {
    /// Returns the priority name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::Critical => "Critical",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        assert!(MigrationPriority::Low < MigrationPriority::Medium);
        assert!(MigrationPriority::High < MigrationPriority::Critical);
    }
}
