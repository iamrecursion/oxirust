//! Alert history tracking.

use crate::alert::Alert;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Alert history record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRecord {
    /// Alert.
    pub alert: Alert,
    /// Acknowledged timestamp.
    pub acknowledged_at: Option<DateTime<Utc>>,
    /// Resolved timestamp.
    pub resolved_at: Option<DateTime<Utc>>,
}

/// Alert history.
pub struct AlertHistory {
    records: parking_lot::RwLock<Vec<AlertRecord>>,
}

impl AlertHistory {
    /// Create a new alert history.
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: parking_lot::RwLock::new(Vec::new()),
        }
    }

    /// Add an alert to history.
    pub fn add(&self, alert: Alert) {
        self.records.write().push(AlertRecord {
            alert,
            acknowledged_at: None,
            resolved_at: None,
        });
    }

    /// Get all records.
    #[must_use]
    pub fn records(&self) -> Vec<AlertRecord> {
        self.records.read().clone()
    }
}

impl Default for AlertHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alert::AlertSeverity;

    #[test]
    fn test_alert_history() {
        let history = AlertHistory::new();

        let alert = Alert::new("test", AlertSeverity::Warning, "Test", "test.metric", 100.0);

        history.add(alert);

        assert_eq!(history.records().len(), 1);
    }
}
