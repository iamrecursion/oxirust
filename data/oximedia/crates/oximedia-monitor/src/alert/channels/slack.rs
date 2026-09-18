//! Slack alert channel.

use crate::alert::channels::AlertChannel;
use crate::alert::severity::AlertSeverity;
use crate::alert::Alert;
use crate::error::{MonitorError, MonitorResult};
use async_trait::async_trait;
use serde_json::{json, Value};

/// Slack alert channel.
pub struct SlackChannel {
    webhook_url: String,
    client: reqwest::Client,
}

impl SlackChannel {
    /// Create a new Slack channel.
    #[must_use]
    pub fn new(webhook_url: String) -> Self {
        Self {
            webhook_url,
            client: reqwest::Client::new(),
        }
    }

    /// Format the alert as a Slack message payload.
    fn format_payload(alert: &Alert) -> Value {
        let color = match alert.severity {
            AlertSeverity::Critical => "#FF0000",
            AlertSeverity::Warning => "#FFAA00",
            AlertSeverity::Info => "#0099FF",
        };

        let severity_icon = match alert.severity {
            AlertSeverity::Critical => ":rotating_light:",
            AlertSeverity::Warning => ":warning:",
            AlertSeverity::Info => ":information_source:",
        };

        let timestamp_str = alert.timestamp.format("%Y-%m-%d %H:%M:%S UTC").to_string();

        let mut fields = vec![
            json!({
                "title": "Severity",
                "value": alert.severity.to_string(),
                "short": true
            }),
            json!({
                "title": "Metric",
                "value": alert.metric_name,
                "short": true
            }),
            json!({
                "title": "Value",
                "value": format!("{:.4}", alert.metric_value),
                "short": true
            }),
            json!({
                "title": "Time",
                "value": timestamp_str,
                "short": true
            }),
        ];

        if let Some(threshold) = alert.threshold {
            fields.push(json!({
                "title": "Threshold",
                "value": format!("{:.4}", threshold),
                "short": true
            }));
        }

        if !alert.labels.is_empty() {
            let labels_str: Vec<String> = alert
                .labels
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            fields.push(json!({
                "title": "Labels",
                "value": labels_str.join(", "),
                "short": false
            }));
        }

        json!({
            "text": format!("{} *[{}] {}*", severity_icon, alert.severity, alert.rule_name),
            "attachments": [
                {
                    "color": color,
                    "title": alert.rule_name,
                    "text": alert.message,
                    "fields": fields,
                    "footer": format!("Alert ID: {}", alert.id),
                    "ts": alert.timestamp.timestamp()
                }
            ]
        })
    }
}

#[async_trait]
impl AlertChannel for SlackChannel {
    async fn send(&self, alert: &Alert) -> MonitorResult<()> {
        let payload = Self::format_payload(alert);

        let response = self
            .client
            .post(&self.webhook_url)
            .json(&payload)
            .send()
            .await
            .map_err(MonitorError::Http)?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(MonitorError::Channel(format!(
                "Slack webhook returned HTTP {status}: {body}"
            )));
        }

        Ok(())
    }
}
