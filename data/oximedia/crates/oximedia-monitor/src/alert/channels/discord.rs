//! Discord alert channel.

use crate::alert::channels::AlertChannel;
use crate::alert::severity::AlertSeverity;
use crate::alert::Alert;
use crate::error::{MonitorError, MonitorResult};
use async_trait::async_trait;
use serde_json::{json, Value};

/// Discord color codes for embed borders.
const COLOR_CRITICAL: u32 = 0xFF0000; // Red
const COLOR_WARNING: u32 = 0xFF8C00; // Dark Orange
const COLOR_INFO: u32 = 0x0099FF; // Blue

/// Discord alert channel.
pub struct DiscordChannel {
    webhook_url: String,
    client: reqwest::Client,
}

impl DiscordChannel {
    /// Create a new Discord channel.
    #[must_use]
    pub fn new(webhook_url: String) -> Self {
        Self {
            webhook_url,
            client: reqwest::Client::new(),
        }
    }

    /// Map severity to a Discord embed color integer.
    #[must_use]
    fn severity_color(severity: &AlertSeverity) -> u32 {
        match severity {
            AlertSeverity::Critical => COLOR_CRITICAL,
            AlertSeverity::Warning => COLOR_WARNING,
            AlertSeverity::Info => COLOR_INFO,
        }
    }

    /// Format the alert as a Discord webhook payload.
    fn format_payload(alert: &Alert) -> Value {
        let color = Self::severity_color(&alert.severity);
        let timestamp_iso = alert.timestamp.to_rfc3339();

        let mut fields = vec![
            json!({ "name": "Severity", "value": alert.severity.to_string(), "inline": true }),
            json!({ "name": "Metric", "value": &alert.metric_name, "inline": true }),
            json!({ "name": "Value", "value": format!("{:.4}", alert.metric_value), "inline": true }),
        ];

        if let Some(threshold) = alert.threshold {
            fields.push(json!({
                "name": "Threshold",
                "value": format!("{:.4}", threshold),
                "inline": true
            }));
        }

        fields.push(json!({
            "name": "State",
            "value": format!("{:?}", alert.state),
            "inline": true
        }));

        if !alert.labels.is_empty() {
            let labels_str: Vec<String> = alert
                .labels
                .iter()
                .map(|(k, v)| format!("`{k}={v}`"))
                .collect();
            fields.push(json!({
                "name": "Labels",
                "value": labels_str.join(" "),
                "inline": false
            }));
        }

        let severity_prefix = match alert.severity {
            AlertSeverity::Critical => "🚨 CRITICAL",
            AlertSeverity::Warning => "⚠️ WARNING",
            AlertSeverity::Info => "ℹ️ INFO",
        };

        json!({
            "embeds": [
                {
                    "title": format!("{}: {}", severity_prefix, alert.rule_name),
                    "description": alert.message,
                    "color": color,
                    "fields": fields,
                    "footer": {
                        "text": format!("Alert ID: {}", alert.id)
                    },
                    "timestamp": timestamp_iso
                }
            ]
        })
    }
}

#[async_trait]
impl AlertChannel for DiscordChannel {
    async fn send(&self, alert: &Alert) -> MonitorResult<()> {
        let payload = Self::format_payload(alert);

        let response = self
            .client
            .post(&self.webhook_url)
            .json(&payload)
            .send()
            .await
            .map_err(MonitorError::Http)?;

        // Discord returns 204 No Content on success.
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(MonitorError::Channel(format!(
                "Discord webhook returned HTTP {status}: {body}"
            )));
        }

        Ok(())
    }
}
