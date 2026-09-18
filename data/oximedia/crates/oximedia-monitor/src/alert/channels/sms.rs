//! SMS alert channel (via Twilio-compatible REST API).

use crate::alert::channels::AlertChannel;
use crate::alert::Alert;
use crate::config::SmsConfig;
use crate::error::{MonitorError, MonitorResult};
use async_trait::async_trait;

/// Twilio Messages API endpoint base URL.
const TWILIO_API_BASE: &str = "https://api.twilio.com/2010-04-01";

/// SMS alert channel.
pub struct SmsChannel {
    config: SmsConfig,
    client: reqwest::Client,
}

impl SmsChannel {
    /// Create a new SMS channel.
    #[must_use]
    pub fn new(config: SmsConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Build a concise SMS body for the given alert.
    fn format_body(alert: &Alert) -> String {
        let mut msg = format!(
            "[{}] {}: {}",
            alert.severity, alert.rule_name, alert.message
        );

        msg.push_str(&format!(
            " | {}={:.4}",
            alert.metric_name, alert.metric_value
        ));

        if let Some(threshold) = alert.threshold {
            msg.push_str(&format!(" (threshold {threshold:.4})"));
        }

        msg.push_str(&format!(
            " @ {}",
            alert.timestamp.format("%Y-%m-%d %H:%M UTC")
        ));

        // SMS messages are limited to 160 chars for a single SMS segment.
        // Truncate gracefully if the message is too long.
        if msg.len() > 1600 {
            msg.truncate(1597);
            msg.push_str("...");
        }

        msg
    }

    /// Send a single SMS to one recipient.
    async fn send_to(&self, to: &str, body: &str) -> MonitorResult<()> {
        let url = format!(
            "{}/Accounts/{}/Messages.json",
            TWILIO_API_BASE, self.config.account_sid
        );

        let params = [
            ("From", self.config.from_number.as_str()),
            ("To", to),
            ("Body", body),
        ];

        let response = self
            .client
            .post(&url)
            .basic_auth(&self.config.account_sid, Some(&self.config.auth_token))
            .form(&params)
            .send()
            .await
            .map_err(MonitorError::Http)?;

        if !response.status().is_success() {
            let status = response.status();
            let resp_body = response.text().await.unwrap_or_default();
            return Err(MonitorError::Channel(format!(
                "Twilio API returned HTTP {status} for recipient {to}: {resp_body}"
            )));
        }

        Ok(())
    }
}

#[async_trait]
impl AlertChannel for SmsChannel {
    async fn send(&self, alert: &Alert) -> MonitorResult<()> {
        let body = Self::format_body(alert);

        let mut last_err: Option<MonitorError> = None;

        for to_number in &self.config.to_numbers {
            if let Err(e) = self.send_to(to_number, &body).await {
                // Log the error but continue sending to remaining recipients.
                tracing::error!("Failed to send SMS to {}: {}", to_number, e);
                last_err = Some(e);
            }
        }

        // Propagate the last error if all sends failed.
        if let Some(err) = last_err {
            if self.config.to_numbers.len() == 1 {
                return Err(err);
            }
        }

        Ok(())
    }
}
