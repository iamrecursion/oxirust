//! Change request priority levels.

use serde::{Deserialize, Serialize};

/// Priority level for change requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ChangePriority {
    /// Low priority - cosmetic changes.
    Low,
    /// Normal priority - standard improvements.
    Normal,
    /// High priority - important changes.
    High,
    /// Critical priority - must be fixed.
    Critical,
}

impl ChangePriority {
    /// Get all priority levels.
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![Self::Low, Self::Normal, Self::High, Self::Critical]
    }

    /// Check if priority is high or critical.
    #[must_use]
    pub fn is_urgent(self) -> bool {
        matches!(self, Self::High | Self::Critical)
    }

    /// Get priority name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Normal => "Normal",
            Self::High => "High",
            Self::Critical => "Critical",
        }
    }
}

impl std::fmt::Display for ChangePriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        assert!(ChangePriority::Low < ChangePriority::Normal);
        assert!(ChangePriority::Normal < ChangePriority::High);
        assert!(ChangePriority::High < ChangePriority::Critical);
    }

    #[test]
    fn test_priority_is_urgent() {
        assert!(!ChangePriority::Low.is_urgent());
        assert!(!ChangePriority::Normal.is_urgent());
        assert!(ChangePriority::High.is_urgent());
        assert!(ChangePriority::Critical.is_urgent());
    }

    #[test]
    fn test_priority_name() {
        assert_eq!(ChangePriority::Low.name(), "Low");
        assert_eq!(ChangePriority::Normal.name(), "Normal");
        assert_eq!(ChangePriority::High.name(), "High");
        assert_eq!(ChangePriority::Critical.name(), "Critical");
    }

    #[test]
    fn test_priority_all() {
        let all = ChangePriority::all();
        assert_eq!(all.len(), 4);
    }
}
