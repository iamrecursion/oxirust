//! Alert animations and notifications.

use std::time::Duration;

/// Alert manager for stream alerts.
pub struct AlertManager {
    alerts: Vec<Alert>,
}

/// Alert.
#[derive(Debug, Clone)]
pub struct Alert {
    /// Alert type
    pub alert_type: AlertType,
    /// Message
    pub message: String,
    /// Duration
    pub duration: Duration,
}

/// Alert type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertType {
    /// Follower alert
    Follower,
    /// Subscriber alert
    Subscriber,
    /// Donation alert
    Donation,
    /// Host alert
    Host,
    /// Raid alert
    Raid,
    /// Custom alert
    Custom,
}

impl AlertManager {
    /// Create a new alert manager.
    #[must_use]
    pub fn new() -> Self {
        Self { alerts: Vec::new() }
    }

    /// Queue an alert.
    pub fn queue_alert(&mut self, alert: Alert) {
        self.alerts.push(alert);
    }

    /// Get next alert.
    pub fn next_alert(&mut self) -> Option<Alert> {
        if self.alerts.is_empty() {
            None
        } else {
            Some(self.alerts.remove(0))
        }
    }

    /// Clear all alerts.
    pub fn clear(&mut self) {
        self.alerts.clear();
    }

    /// Get alert count.
    #[must_use]
    pub fn alert_count(&self) -> usize {
        self.alerts.len()
    }
}

impl Default for AlertManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_manager_creation() {
        let manager = AlertManager::new();
        assert_eq!(manager.alert_count(), 0);
    }

    #[test]
    fn test_queue_alert() {
        let mut manager = AlertManager::new();
        let alert = Alert {
            alert_type: AlertType::Follower,
            message: "New follower!".to_string(),
            duration: Duration::from_secs(3),
        };
        manager.queue_alert(alert);
        assert_eq!(manager.alert_count(), 1);
    }
}
