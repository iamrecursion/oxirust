//! Generic webhook alert channel.

use crate::alert::channels::AlertChannel;
use crate::alert::Alert;
use crate::error::{MonitorError, MonitorResult};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Generic webhook alert channel.
///
/// POSTs a JSON body containing full alert details to the configured URL.
/// Supports configurable custom headers.
pub struct WebhookChannel {
    url: String,
    headers: HashMap<String, String>,
    client: reqwest::Client,
}

impl WebhookChannel {
    /// Create a new webhook channel with no custom headers.
    #[must_use]
    pub fn new(url: String) -> Self {
        Self {
            url,
            headers: HashMap::new(),
            client: reqwest::Client::new(),
        }
    }

    /// Create a new webhook channel with custom headers.
    #[must_use]
    pub fn with_headers(url: String, headers: HashMap<String, String>) -> Self {
        Self {
            url,
            headers,
            client: reqwest::Client::new(),
        }
    }

    /// Add a custom header.
    pub fn add_header(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.headers.insert(key.into(), value.into());
    }

    /// Build the JSON payload for the webhook.
    fn build_payload(alert: &Alert) -> Value {
        let labels: Vec<Value> = alert
            .labels
            .iter()
            .map(|(k, v)| json!({ "key": k, "value": v }))
            .collect();

        json!({
            "id": alert.id,
            "rule_name": alert.rule_name,
            "severity": alert.severity.to_string(),
            "message": alert.message,
            "metric": {
                "name": alert.metric_name,
                "value": alert.metric_value,
                "threshold": alert.threshold
            },
            "timestamp": alert.timestamp.to_rfc3339(),
            "state": format!("{:?}", alert.state),
            "labels": labels
        })
    }
}

#[async_trait]
impl AlertChannel for WebhookChannel {
    async fn send(&self, alert: &Alert) -> MonitorResult<()> {
        let payload = Self::build_payload(alert);

        let mut request = self
            .client
            .post(&self.url)
            .json(&payload)
            .header("User-Agent", "oximedia-monitor/1.0");

        for (key, value) in &self.headers {
            request = request.header(key.as_str(), value.as_str());
        }

        let response = request.send().await.map_err(MonitorError::Http)?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(MonitorError::Channel(format!(
                "Webhook POST to {} returned HTTP {}: {}",
                self.url, status, body
            )));
        }

        Ok(())
    }
}
