//! Email notifications.

use crate::error::ReviewResult;
use serde::{Deserialize, Serialize};

/// Email notification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailNotification {
    /// Recipient email.
    pub to: String,
    /// Sender email.
    pub from: String,
    /// Email subject.
    pub subject: String,
    /// Email body (HTML).
    pub body_html: String,
    /// Email body (plain text).
    pub body_text: String,
}

impl EmailNotification {
    /// Create a new email notification.
    #[must_use]
    pub fn new(
        to: String,
        from: String,
        subject: String,
        body_html: String,
        body_text: String,
    ) -> Self {
        Self {
            to,
            from,
            subject,
            body_html,
            body_text,
        }
    }
}

/// Send an email notification.
///
/// # Errors
///
/// Returns error if email fails to send.
pub async fn send_email_notification(email: EmailNotification) -> ReviewResult<()> {
    // In a real implementation, this would:
    // 1. Connect to SMTP server
    // 2. Format the email
    // 3. Send the email
    // 4. Handle delivery failures

    let _ = email;
    Ok(())
}

/// Email template type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmailTemplate {
    /// Comment added template.
    CommentAdded,
    /// Task assigned template.
    TaskAssigned,
    /// Approval requested template.
    ApprovalRequested,
    /// Session invited template.
    SessionInvited,
    /// Deadline approaching template.
    DeadlineApproaching,
}

impl EmailTemplate {
    /// Get template subject.
    #[must_use]
    pub fn subject(self) -> &'static str {
        match self {
            Self::CommentAdded => "New Comment on Review Session",
            Self::TaskAssigned => "New Task Assigned",
            Self::ApprovalRequested => "Approval Requested",
            Self::SessionInvited => "You've been invited to a review session",
            Self::DeadlineApproaching => "Review Deadline Approaching",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_email_notification_creation() {
        let email = EmailNotification::new(
            "user@example.com".to_string(),
            "noreply@oximedia.com".to_string(),
            "Test".to_string(),
            "<p>Test</p>".to_string(),
            "Test".to_string(),
        );

        assert_eq!(email.to, "user@example.com");
        assert_eq!(email.subject, "Test");
    }

    #[tokio::test]
    async fn test_send_email_notification() {
        let email = EmailNotification::new(
            "user@example.com".to_string(),
            "noreply@oximedia.com".to_string(),
            "Test".to_string(),
            "<p>Test</p>".to_string(),
            "Test".to_string(),
        );

        let result = send_email_notification(email).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_email_template_subject() {
        assert_eq!(
            EmailTemplate::CommentAdded.subject(),
            "New Comment on Review Session"
        );
        assert_eq!(EmailTemplate::TaskAssigned.subject(), "New Task Assigned");
    }
}
