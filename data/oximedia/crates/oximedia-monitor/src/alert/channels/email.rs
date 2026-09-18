//! Email alert channel.

use crate::alert::channels::AlertChannel;
use crate::alert::severity::AlertSeverity;
use crate::alert::Alert;
use crate::config::EmailConfig;
use crate::error::{MonitorError, MonitorResult};
use async_trait::async_trait;
use lettre::message::{header::ContentType, Mailbox, MultiPart, SinglePart};
use lettre::transport::smtp::authentication::Credentials;
use lettre::{AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor};

/// Email alert channel.
pub struct EmailChannel {
    config: EmailConfig,
}

impl EmailChannel {
    /// Create a new email channel.
    #[must_use]
    pub fn new(config: EmailConfig) -> Self {
        Self { config }
    }

    /// Build the email subject.
    fn build_subject(alert: &Alert) -> String {
        format!(
            "[{}] {} - {}",
            alert.severity, alert.rule_name, alert.metric_name
        )
    }

    /// Build the plain-text email body.
    fn build_text_body(alert: &Alert) -> String {
        let mut body = format!(
            "Alert: {}\nSeverity: {}\nMessage: {}\n\nDetails\n-------\n",
            alert.rule_name, alert.severity, alert.message
        );

        body.push_str(&format!(
            "Metric:    {}\nValue:     {:.4}\n",
            alert.metric_name, alert.metric_value
        ));

        if let Some(threshold) = alert.threshold {
            body.push_str(&format!("Threshold: {threshold:.4}\n"));
        }

        body.push_str(&format!(
            "Time:      {}\nAlert ID:  {}\nState:     {:?}\n",
            alert.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            alert.id,
            alert.state
        ));

        if !alert.labels.is_empty() {
            body.push_str("\nLabels\n------\n");
            for (k, v) in &alert.labels {
                body.push_str(&format!("{k}: {v}\n"));
            }
        }

        body
    }

    /// Build the HTML email body.
    fn build_html_body(alert: &Alert) -> String {
        let (bg_color, border_color) = match alert.severity {
            AlertSeverity::Critical => ("#fff0f0", "#ff0000"),
            AlertSeverity::Warning => ("#fffbf0", "#ffaa00"),
            AlertSeverity::Info => ("#f0f8ff", "#0099ff"),
        };

        let mut rows = format!(
            "<tr><td><b>Metric</b></td><td>{}</td></tr>\
             <tr><td><b>Value</b></td><td>{:.4}</td></tr>",
            alert.metric_name, alert.metric_value
        );

        if let Some(threshold) = alert.threshold {
            rows.push_str(&format!(
                "<tr><td><b>Threshold</b></td><td>{threshold:.4}</td></tr>"
            ));
        }

        rows.push_str(&format!(
            "<tr><td><b>Time</b></td><td>{}</td></tr>\
             <tr><td><b>Alert ID</b></td><td>{}</td></tr>\
             <tr><td><b>State</b></td><td>{:?}</td></tr>",
            alert.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            alert.id,
            alert.state
        ));

        if !alert.labels.is_empty() {
            let label_rows: String = alert
                .labels
                .iter()
                .map(|(k, v)| format!("<tr><td><b>{k}</b></td><td>{v}</td></tr>"))
                .collect();
            rows.push_str(&format!(
                "<tr><td colspan='2'><b>Labels</b></td></tr>{label_rows}"
            ));
        }

        format!(
            r#"<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
  <div style="background:{bg};border-left:4px solid {border};padding:16px;margin-bottom:16px;">
    <h2 style="margin:0 0 8px 0;">[{severity}] {rule}</h2>
    <p style="margin:0;font-size:14px;">{message}</p>
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:14px;">
    {rows}
  </table>
</body>
</html>"#,
            bg = bg_color,
            border = border_color,
            severity = alert.severity,
            rule = alert.rule_name,
            message = alert.message,
            rows = rows
        )
    }
}

#[async_trait]
impl AlertChannel for EmailChannel {
    async fn send(&self, alert: &Alert) -> MonitorResult<()> {
        let subject = Self::build_subject(alert);
        let text_body = Self::build_text_body(alert);
        let html_body = Self::build_html_body(alert);

        let from: Mailbox = self
            .config
            .from_address
            .parse()
            .map_err(|e| MonitorError::Email(format!("Invalid from address: {e}")))?;

        // Build the message with all recipients.
        let mut message_builder = Message::builder().from(from).subject(subject);

        for to_addr in &self.config.to_addresses {
            let mailbox: Mailbox = to_addr
                .parse()
                .map_err(|e| MonitorError::Email(format!("Invalid to address '{to_addr}': {e}")))?;
            message_builder = message_builder.to(mailbox);
        }

        let email = message_builder
            .multipart(
                MultiPart::alternative()
                    .singlepart(
                        SinglePart::builder()
                            .header(ContentType::TEXT_PLAIN)
                            .body(text_body),
                    )
                    .singlepart(
                        SinglePart::builder()
                            .header(ContentType::TEXT_HTML)
                            .body(html_body),
                    ),
            )
            .map_err(|e| MonitorError::Email(format!("Failed to build email: {e}")))?;

        let creds = Credentials::new(
            self.config.smtp_username.clone(),
            self.config.smtp_password.clone(),
        );

        let transport = if self.config.use_tls {
            AsyncSmtpTransport::<Tokio1Executor>::relay(&self.config.smtp_server)
                .map_err(|e| MonitorError::Email(format!("SMTP relay error: {e}")))?
                .port(self.config.smtp_port)
                .credentials(creds)
                .build()
        } else {
            AsyncSmtpTransport::<Tokio1Executor>::builder_dangerous(&self.config.smtp_server)
                .port(self.config.smtp_port)
                .credentials(creds)
                .build()
        };

        transport
            .send(email)
            .await
            .map_err(|e| MonitorError::Email(format!("Failed to send email: {e}")))?;

        Ok(())
    }
}
