//! Webhook notifications.

use crate::error::ReviewResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Webhook notification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookNotification {
    /// Webhook URL.
    pub url: String,
    /// HTTP method (POST, PUT, etc.).
    pub method: String,
    /// Request headers.
    pub headers: HashMap<String, String>,
    /// Request payload.
    pub payload: serde_json::Value,
}

impl WebhookNotification {
    /// Create a new webhook notification.
    #[must_use]
    pub fn new(url: String, payload: serde_json::Value) -> Self {
        Self {
            url,
            method: "POST".to_string(),
            headers: HashMap::new(),
            payload,
        }
    }

    /// Add a header.
    #[must_use]
    pub fn with_header(mut self, key: String, value: String) -> Self {
        self.headers.insert(key, value);
        self
    }

    /// Set HTTP method.
    #[must_use]
    pub fn with_method(mut self, method: String) -> Self {
        self.method = method;
        self
    }
}

/// Send a webhook notification.
///
/// # Errors
///
/// Returns error if webhook fails to send.
pub async fn send_webhook(webhook: WebhookNotification) -> ReviewResult<()> {
    // In a real implementation, this would:
    // 1. Make HTTP request to webhook URL
    // 2. Include headers and payload
    // 3. Handle response and errors
    // 4. Implement retry logic

    let _ = webhook;
    Ok(())
}

/// Webhook delivery status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookDelivery {
    /// Delivery ID.
    pub id: String,
    /// Webhook URL.
    pub url: String,
    /// HTTP status code.
    pub status_code: u16,
    /// Response body.
    pub response: String,
    /// Delivery attempts.
    pub attempts: u32,
    /// Success status.
    pub success: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webhook_notification_creation() {
        let payload = serde_json::json!({
            "event": "comment_added",
            "data": {
                "comment_id": "123"
            }
        });

        let webhook = WebhookNotification::new("https://example.com/webhook".to_string(), payload);

        assert_eq!(webhook.url, "https://example.com/webhook");
        assert_eq!(webhook.method, "POST");
    }

    #[test]
    fn test_webhook_notification_with_header() {
        let payload = serde_json::json!({});
        let webhook = WebhookNotification::new("https://example.com".to_string(), payload)
            .with_header("Authorization".to_string(), "Bearer token".to_string());

        assert!(webhook.headers.contains_key("Authorization"));
    }

    #[test]
    fn test_webhook_notification_with_method() {
        let payload = serde_json::json!({});
        let webhook = WebhookNotification::new("https://example.com".to_string(), payload)
            .with_method("PUT".to_string());

        assert_eq!(webhook.method, "PUT");
    }

    #[tokio::test]
    async fn test_send_webhook() {
        let payload = serde_json::json!({
            "test": "data"
        });

        let webhook = WebhookNotification::new("https://example.com/webhook".to_string(), payload);

        let result = send_webhook(webhook).await;
        assert!(result.is_ok());
    }
}
