//! System notifications (in-app).

use crate::notify::NotificationType;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// System notification for in-app display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemNotification {
    /// Notification ID.
    pub id: String,
    /// Notification type.
    pub notification_type: NotificationType,
    /// Title.
    pub title: String,
    /// Message.
    pub message: String,
    /// Severity level.
    pub severity: NotificationSeverity,
    /// Action button (if any).
    pub action: Option<NotificationAction>,
    /// Expiration timestamp.
    pub expires_at: Option<DateTime<Utc>>,
    /// Created timestamp.
    pub created_at: DateTime<Utc>,
}

/// Notification severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotificationSeverity {
    /// Informational.
    Info,
    /// Success.
    Success,
    /// Warning.
    Warning,
    /// Error.
    Error,
}

/// Notification action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationAction {
    /// Action label.
    pub label: String,
    /// Action URL or handler.
    pub action: String,
}

impl SystemNotification {
    /// Create a new system notification.
    #[must_use]
    pub fn new(notification_type: NotificationType, title: String, message: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            notification_type,
            title,
            message,
            severity: NotificationSeverity::Info,
            action: None,
            expires_at: None,
            created_at: Utc::now(),
        }
    }

    /// Set severity.
    #[must_use]
    pub fn with_severity(mut self, severity: NotificationSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Set action.
    #[must_use]
    pub fn with_action(mut self, label: String, action: String) -> Self {
        self.action = Some(NotificationAction { label, action });
        self
    }

    /// Set expiration.
    #[must_use]
    pub fn with_expiration(mut self, expires_at: DateTime<Utc>) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Check if notification is expired.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Utc::now() > expires_at
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_system_notification_creation() {
        let notification = SystemNotification::new(
            NotificationType::CommentAdded,
            "New Comment".to_string(),
            "A comment was added".to_string(),
        );

        assert_eq!(notification.severity, NotificationSeverity::Info);
        assert!(notification.action.is_none());
    }

    #[test]
    fn test_system_notification_with_severity() {
        let notification = SystemNotification::new(
            NotificationType::ApprovalRejected,
            "Rejected".to_string(),
            "Your submission was rejected".to_string(),
        )
        .with_severity(NotificationSeverity::Error);

        assert_eq!(notification.severity, NotificationSeverity::Error);
    }

    #[test]
    fn test_system_notification_with_action() {
        let notification = SystemNotification::new(
            NotificationType::TaskAssigned,
            "Task Assigned".to_string(),
            "You have a new task".to_string(),
        )
        .with_action("View Task".to_string(), "/task/123".to_string());

        assert!(notification.action.is_some());
    }

    #[test]
    fn test_system_notification_is_expired() {
        let past = Utc::now() - Duration::hours(1);
        let notification = SystemNotification::new(
            NotificationType::SessionClosed,
            "Session Closed".to_string(),
            "The review session was closed".to_string(),
        )
        .with_expiration(past);

        assert!(notification.is_expired());

        let future = Utc::now() + Duration::hours(1);
        let notification = SystemNotification::new(
            NotificationType::SessionClosed,
            "Session Closed".to_string(),
            "The review session was closed".to_string(),
        )
        .with_expiration(future);

        assert!(!notification.is_expired());
    }
}
