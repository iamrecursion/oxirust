//! Alert severity levels.

use serde::{Deserialize, Serialize};

/// Alert severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational notification.
    Info,
    /// Warning - attention needed.
    Warning,
    /// Critical - immediate action required.
    Critical,
}

impl AlertSeverity {
    /// Get the numeric priority (higher = more severe).
    #[must_use]
    pub const fn priority(&self) -> u8 {
        match self {
            Self::Info => 1,
            Self::Warning => 2,
            Self::Critical => 3,
        }
    }

    /// Get the emoji representation.
    #[must_use]
    pub const fn emoji(&self) -> &'static str {
        match self {
            Self::Info => "ℹ️",
            Self::Warning => "⚠️",
            Self::Critical => "🚨",
        }
    }

    /// Get the color code (hex).
    #[must_use]
    pub const fn color(&self) -> &'static str {
        match self {
            Self::Info => "#0099FF",     // Blue
            Self::Warning => "#FFAA00",  // Orange
            Self::Critical => "#FF0000", // Red
        }
    }

    /// Check if this severity requires immediate action.
    #[must_use]
    pub const fn is_urgent(&self) -> bool {
        matches!(self, Self::Critical)
    }
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_ordering() {
        assert!(AlertSeverity::Critical > AlertSeverity::Warning);
        assert!(AlertSeverity::Warning > AlertSeverity::Info);
    }

    #[test]
    fn test_severity_priority() {
        assert_eq!(AlertSeverity::Info.priority(), 1);
        assert_eq!(AlertSeverity::Warning.priority(), 2);
        assert_eq!(AlertSeverity::Critical.priority(), 3);
    }

    #[test]
    fn test_severity_is_urgent() {
        assert!(!AlertSeverity::Info.is_urgent());
        assert!(!AlertSeverity::Warning.is_urgent());
        assert!(AlertSeverity::Critical.is_urgent());
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(AlertSeverity::Info.to_string(), "INFO");
        assert_eq!(AlertSeverity::Warning.to_string(), "WARNING");
        assert_eq!(AlertSeverity::Critical.to_string(), "CRITICAL");
    }
}
