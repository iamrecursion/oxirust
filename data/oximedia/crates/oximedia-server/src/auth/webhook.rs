//! Webhook notifications for stream events.

use crate::error::{ServerError, ServerResult};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{error, info};

/// Webhook event type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WebhookEventType {
    /// Stream started publishing.
    StreamStarted,
    /// Stream stopped publishing.
    StreamStopped,
    /// Viewer connected.
    ViewerConnected,
    /// Viewer disconnected.
    ViewerDisconnected,
    /// Recording started.
    RecordingStarted,
    /// Recording stopped.
    RecordingStopped,
    /// Transcoding started.
    TranscodingStarted,
    /// Transcoding completed.
    TranscodingCompleted,
    /// Stream error.
    StreamError,
}

/// Webhook event payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookEvent {
    /// Event type.
    pub event_type: WebhookEventType,

    /// Timestamp.
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Stream key.
    pub stream_key: Option<String>,

    /// Application name.
    pub app_name: Option<String>,

    /// Additional data.
    #[serde(flatten)]
    pub data: serde_json::Value,
}

impl WebhookEvent {
    /// Creates a new webhook event.
    #[must_use]
    pub fn new(event_type: WebhookEventType) -> Self {
        Self {
            event_type,
            timestamp: chrono::Utc::now(),
            stream_key: None,
            app_name: None,
            data: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// Sets the stream key.
    #[must_use]
    pub fn with_stream_key(mut self, stream_key: impl Into<String>) -> Self {
        self.stream_key = Some(stream_key.into());
        self
    }

    /// Sets the app name.
    #[must_use]
    pub fn with_app_name(mut self, app_name: impl Into<String>) -> Self {
        self.app_name = Some(app_name.into());
        self
    }

    /// Sets additional data.
    #[must_use]
    pub fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = data;
        self
    }
}

/// Webhook configuration.
#[derive(Debug, Clone)]
pub struct WebhookConfig {
    /// Webhook URLs.
    pub urls: Vec<String>,

    /// Timeout for webhook requests.
    pub timeout: std::time::Duration,

    /// Retry attempts.
    pub retry_attempts: u32,

    /// Enable webhook authentication.
    pub auth_enabled: bool,

    /// Webhook secret for HMAC signing.
    pub secret: Option<String>,
}

impl Default for WebhookConfig {
    fn default() -> Self {
        Self {
            urls: Vec::new(),
            timeout: std::time::Duration::from_secs(10),
            retry_attempts: 3,
            auth_enabled: false,
            secret: None,
        }
    }
}

/// Webhook notifier.
#[allow(dead_code)]
pub struct WebhookNotifier {
    /// Configuration.
    config: WebhookConfig,

    /// HTTP client.
    client: reqwest::Client,

    /// Event queue.
    event_tx: mpsc::UnboundedSender<WebhookEvent>,
}

impl WebhookNotifier {
    /// Creates a new webhook notifier.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying `reqwest::Client` cannot be
    /// constructed (e.g. if the platform's TLS backend fails to
    /// initialize).
    pub fn new(config: WebhookConfig) -> ServerResult<Self> {
        // Ensure the process-wide Pure-Rust `rustls-rustcrypto` provider is
        // installed before building the TLS-capable `reqwest::Client` below
        // (idempotent; safe even if another entry point already installed
        // one).
        oximedia_net::install_default_crypto_provider();

        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| ServerError::Internal(format!("Failed to create HTTP client: {e}")))?;

        let (event_tx, mut event_rx) = mpsc::unbounded_channel();

        // Spawn worker task
        let urls = config.urls.clone();
        let worker_client = client.clone();
        let secret = config.secret.clone();

        tokio::spawn(async move {
            while let Some(event) = event_rx.recv().await {
                for url in &urls {
                    if let Err(e) = Self::send_webhook(&worker_client, url, &event, &secret).await {
                        error!("Failed to send webhook to {}: {}", url, e);
                    } else {
                        info!("Sent webhook to {}: {:?}", url, event.event_type);
                    }
                }
            }
        });

        Ok(Self {
            config,
            client,
            event_tx,
        })
    }

    /// Sends a webhook event.
    async fn send_webhook(
        client: &reqwest::Client,
        url: &str,
        event: &WebhookEvent,
        secret: &Option<String>,
    ) -> ServerResult<()> {
        let body = serde_json::to_string(event)
            .map_err(|e| ServerError::Internal(format!("Failed to serialize event: {e}")))?;

        let mut request = client.post(url).header("Content-Type", "application/json");

        // Add HMAC signature if secret is configured
        if let Some(secret_key) = secret {
            let signature = Self::compute_signature(&body, secret_key);
            request = request.header("X-Webhook-Signature", signature);
        }

        let response = request
            .body(body)
            .send()
            .await
            .map_err(|e| ServerError::Internal(format!("Failed to send webhook: {e}")))?;

        if !response.status().is_success() {
            return Err(ServerError::Internal(format!(
                "Webhook request failed: {}",
                response.status()
            )));
        }

        Ok(())
    }

    /// Computes HMAC signature for webhook payload.
    fn compute_signature(payload: &str, secret: &str) -> String {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        hasher.update(secret.as_bytes());
        hasher.update(payload.as_bytes());
        let result = hasher.finalize();

        hex::encode(result)
    }

    /// Notifies about an event.
    pub fn notify(&self, event: WebhookEvent) {
        let _ = self.event_tx.send(event);
    }

    /// Notifies about stream start.
    pub fn notify_stream_started(&self, app_name: &str, stream_key: &str) {
        let event = WebhookEvent::new(WebhookEventType::StreamStarted)
            .with_app_name(app_name)
            .with_stream_key(stream_key);

        self.notify(event);
    }

    /// Notifies about stream stop.
    pub fn notify_stream_stopped(&self, app_name: &str, stream_key: &str) {
        let event = WebhookEvent::new(WebhookEventType::StreamStopped)
            .with_app_name(app_name)
            .with_stream_key(stream_key);

        self.notify(event);
    }

    /// Notifies about viewer connection.
    pub fn notify_viewer_connected(&self, app_name: &str, stream_key: &str, viewer_id: &str) {
        let event = WebhookEvent::new(WebhookEventType::ViewerConnected)
            .with_app_name(app_name)
            .with_stream_key(stream_key)
            .with_data(serde_json::json!({ "viewer_id": viewer_id }));

        self.notify(event);
    }

    /// Notifies about viewer disconnection.
    pub fn notify_viewer_disconnected(&self, app_name: &str, stream_key: &str, viewer_id: &str) {
        let event = WebhookEvent::new(WebhookEventType::ViewerDisconnected)
            .with_app_name(app_name)
            .with_stream_key(stream_key)
            .with_data(serde_json::json!({ "viewer_id": viewer_id }));

        self.notify(event);
    }

    /// Notifies about recording start.
    pub fn notify_recording_started(&self, stream_key: &str, recording_id: &str) {
        let event = WebhookEvent::new(WebhookEventType::RecordingStarted)
            .with_stream_key(stream_key)
            .with_data(serde_json::json!({ "recording_id": recording_id }));

        self.notify(event);
    }

    /// Notifies about recording stop.
    pub fn notify_recording_stopped(&self, stream_key: &str, recording_id: &str) {
        let event = WebhookEvent::new(WebhookEventType::RecordingStopped)
            .with_stream_key(stream_key)
            .with_data(serde_json::json!({ "recording_id": recording_id }));

        self.notify(event);
    }
}
