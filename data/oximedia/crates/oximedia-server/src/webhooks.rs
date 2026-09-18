//! Webhook delivery system for OxiMedia server events.
//!
//! Webhooks allow external services to receive real-time notifications when
//! significant events occur (transcode completed/failed, media upload/delete).
//! Each delivery is signed with HMAC-SHA256 so the receiver can verify origin.
//!
//! # Retry policy
//! Up to 3 delivery attempts with exponential backoff: 1 s → 4 s → 16 s.

use crate::error::{ServerError, ServerResult};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use hex::ToHex;
use hmac::{Hmac, KeyInit, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

// ── Domain types ──────────────────────────────────────────────────────────────

/// Events that can be emitted and delivered to registered webhooks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WebhookEvent {
    /// A transcode job finished successfully.
    TranscodeCompleted {
        /// Job identifier.
        job_id: String,
        /// Media item that was transcoded.
        media_id: String,
        /// URL of the output file.
        output_url: String,
    },
    /// A transcode job failed.
    TranscodeFailed {
        /// Job identifier.
        job_id: String,
        /// Media item that was transcoded.
        media_id: String,
        /// Error description.
        error: String,
    },
    /// A media file was successfully uploaded.
    MediaUploaded {
        /// New media ID.
        media_id: String,
        /// Original filename.
        filename: String,
        /// File size in bytes.
        size_bytes: u64,
    },
    /// A media item was deleted.
    MediaDeleted {
        /// Deleted media ID.
        media_id: String,
    },
}

impl WebhookEvent {
    /// Returns the event-type string used for filtering (e.g. "transcode.completed").
    pub fn event_type_name(&self) -> &'static str {
        match self {
            Self::TranscodeCompleted { .. } => "transcode.completed",
            Self::TranscodeFailed { .. } => "transcode.failed",
            Self::MediaUploaded { .. } => "media.uploaded",
            Self::MediaDeleted { .. } => "media.deleted",
        }
    }
}

/// Registration configuration for a single webhook endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    /// Stable webhook identifier.
    pub id: String,
    /// Target URL that receives POST requests.
    pub url: String,
    /// Event-type strings this webhook is subscribed to.
    pub events: Vec<String>,
    /// HMAC-SHA256 signing secret.  **Never returned in API responses.**
    #[serde(skip_serializing)]
    pub secret: String,
    /// Whether the webhook is active.
    pub active: bool,
}

/// Status of a single delivery attempt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryStatus {
    /// Delivery succeeded (HTTP 2xx).
    Success,
    /// Delivery failed and will not be retried further.
    Failed,
    /// Delivery is pending or in-flight.
    Pending,
}

/// Record of a webhook delivery attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookDelivery {
    /// Delivery identifier.
    pub id: String,
    /// Which webhook this delivery belongs to.
    pub webhook_id: String,
    /// Event-type name.
    pub event: String,
    /// Full JSON payload that was (or will be) sent.
    pub payload: serde_json::Value,
    /// Outcome of the delivery.
    pub status: DeliveryStatus,
    /// Number of attempts made so far.
    pub attempts: u32,
    /// Unix timestamp when the delivery record was created.
    pub created_at: i64,
}

// ── Request / response bodies ─────────────────────────────────────────────────

/// Request body for `POST /api/v1/webhooks`.
#[derive(Debug, Deserialize)]
pub struct RegisterWebhookRequest {
    /// Target URL.
    pub url: String,
    /// List of event-type strings to subscribe to.
    pub events: Vec<String>,
    /// Signing secret.  If omitted a random 32-byte hex secret is generated.
    pub secret: Option<String>,
}

/// Request body for `PUT /api/v1/webhooks/{id}`.
#[derive(Debug, Deserialize)]
pub struct UpdateWebhookRequest {
    /// Updated target URL (optional).
    pub url: Option<String>,
    /// Updated event subscriptions (optional).
    pub events: Option<Vec<String>>,
    /// New signing secret (optional).
    pub secret: Option<String>,
    /// Enable/disable the webhook (optional).
    pub active: Option<bool>,
}

// ── WebhookManager ────────────────────────────────────────────────────────────

/// In-memory registry and delivery engine for webhooks.
///
/// In a production deployment this would be backed by the database; here we use
/// an in-process `RwLock<Vec<...>>` to keep the implementation free of compile-time
/// SQLx queries while still being fully functional and testable.
pub struct WebhookManager {
    configs: RwLock<Vec<WebhookConfig>>,
    deliveries: RwLock<Vec<WebhookDelivery>>,
    /// Optional HTTP client — `None` in unit-test environments where no TLS
    /// provider is available.  Constructed lazily on first delivery attempt.
    http: Option<reqwest::Client>,
}

impl WebhookManager {
    /// Creates a new `WebhookManager`, attempting to build an HTTP client.
    ///
    /// If no TLS provider is installed (e.g. in unit tests), the HTTP client
    /// will be absent and delivery attempts will be skipped with a warning.
    #[must_use]
    pub fn new() -> Self {
        // Ensure the process-wide Pure-Rust `rustls-rustcrypto` provider is
        // installed before building the TLS-capable `reqwest::Client` below.
        // Idempotent — cheap to call even if `Server::new()` (or any other
        // entry point) already installed one.
        oximedia_net::install_default_crypto_provider();

        // Belt-and-suspenders: catch_unwind still guards against reqwest
        // panicking if, for whatever reason, no CryptoProvider ended up
        // installed (e.g. a future rustls/reqwest version behaves
        // differently). This should no longer trigger in practice.
        let http = std::panic::catch_unwind(|| {
            reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .ok()
        })
        .ok()
        .flatten();

        Self {
            configs: RwLock::new(Vec::new()),
            deliveries: RwLock::new(Vec::new()),
            http,
        }
    }

    /// Creates a `WebhookManager` without an HTTP client (for testing).
    #[cfg(test)]
    #[must_use]
    pub fn new_for_test() -> Self {
        Self {
            configs: RwLock::new(Vec::new()),
            deliveries: RwLock::new(Vec::new()),
            http: None,
        }
    }

    /// Registers a new webhook and returns its configuration.
    ///
    /// # Errors
    ///
    /// Returns `ServerError::BadRequest` when the URL is empty or the event
    /// list is empty.
    pub async fn register(&self, req: RegisterWebhookRequest) -> ServerResult<WebhookConfig> {
        if req.url.trim().is_empty() {
            return Err(ServerError::BadRequest(
                "webhook url must not be empty".to_string(),
            ));
        }
        if req.events.is_empty() {
            return Err(ServerError::BadRequest(
                "webhook must subscribe to at least one event".to_string(),
            ));
        }

        let secret = req
            .secret
            .filter(|s| !s.is_empty())
            .unwrap_or_else(generate_secret);

        let config = WebhookConfig {
            id: Uuid::new_v4().to_string(),
            url: req.url,
            events: req.events,
            secret,
            active: true,
        };

        self.configs.write().await.push(config.clone());
        Ok(config)
    }

    /// Returns all registered webhooks (secrets stripped).
    pub async fn list(&self) -> Vec<WebhookConfig> {
        self.configs.read().await.clone()
    }

    /// Returns a single webhook by ID (secret stripped).
    ///
    /// # Errors
    ///
    /// Returns `ServerError::NotFound` when the ID is unknown.
    pub async fn get(&self, id: &str) -> ServerResult<WebhookConfig> {
        self.configs
            .read()
            .await
            .iter()
            .find(|c| c.id == id)
            .cloned()
            .ok_or_else(|| ServerError::NotFound(format!("Webhook '{}' not found", id)))
    }

    /// Updates an existing webhook configuration.
    ///
    /// # Errors
    ///
    /// Returns `ServerError::NotFound` when the ID is unknown.
    pub async fn update(&self, id: &str, req: UpdateWebhookRequest) -> ServerResult<WebhookConfig> {
        let mut configs = self.configs.write().await;
        let cfg = configs
            .iter_mut()
            .find(|c| c.id == id)
            .ok_or_else(|| ServerError::NotFound(format!("Webhook '{}' not found", id)))?;

        if let Some(url) = req.url {
            cfg.url = url;
        }
        if let Some(events) = req.events {
            cfg.events = events;
        }
        if let Some(secret) = req.secret {
            cfg.secret = secret;
        }
        if let Some(active) = req.active {
            cfg.active = active;
        }

        Ok(cfg.clone())
    }

    /// Removes a webhook registration.
    ///
    /// # Errors
    ///
    /// Returns `ServerError::NotFound` when the ID is unknown.
    pub async fn delete(&self, id: &str) -> ServerResult<()> {
        let mut configs = self.configs.write().await;
        let pos = configs
            .iter()
            .position(|c| c.id == id)
            .ok_or_else(|| ServerError::NotFound(format!("Webhook '{}' not found", id)))?;
        configs.remove(pos);
        Ok(())
    }

    /// Returns all deliveries for a given webhook ID.
    pub async fn deliveries_for(&self, webhook_id: &str) -> Vec<WebhookDelivery> {
        self.deliveries
            .read()
            .await
            .iter()
            .filter(|d| d.webhook_id == webhook_id)
            .cloned()
            .collect()
    }

    /// Fans out `event` to all active, subscribed webhooks.
    ///
    /// Non-blocking: spawns a background task per matching webhook.
    /// Each task retries up to 3 times with exponential backoff (1 s, 4 s, 16 s).
    pub async fn deliver(&self, event: &WebhookEvent) {
        let event_name = event.event_type_name();

        // Collect matching configs — hold the lock only briefly.
        let matching: Vec<WebhookConfig> = self
            .configs
            .read()
            .await
            .iter()
            .filter(|c| c.active && c.events.iter().any(|e| e == event_name))
            .cloned()
            .collect();

        let payload = match serde_json::to_value(event) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("webhook: failed to serialize event: {}", e);
                return;
            }
        };

        // If no HTTP client is available (e.g. unit tests without TLS provider),
        // log a warning and skip actual delivery.
        let http = match &self.http {
            Some(c) => c.clone(),
            None => {
                tracing::warn!(
                    "webhook: HTTP client unavailable; skipping delivery of '{}'",
                    event_name
                );
                return;
            }
        };

        for cfg in matching {
            let payload_clone = payload.clone();
            let http = http.clone();

            // Record a pending delivery.
            let delivery_id = Uuid::new_v4().to_string();
            let delivery = WebhookDelivery {
                id: delivery_id.clone(),
                webhook_id: cfg.id.clone(),
                event: event_name.to_string(),
                payload: payload_clone.clone(),
                status: DeliveryStatus::Pending,
                attempts: 0,
                created_at: chrono::Utc::now().timestamp(),
            };
            self.deliveries.write().await.push(delivery);

            tokio::spawn(async move {
                attempt_delivery(http, cfg, payload_clone, delivery_id).await;
            });
        }
    }
}

impl Default for WebhookManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Delivery internals ────────────────────────────────────────────────────────

/// Maximum number of delivery attempts before a webhook is marked `Failed`.
const MAX_DELIVERY_ATTEMPTS: u32 = 3;

/// Exponential backoff delays (seconds) applied *before* each retry attempt.
///
/// With [`MAX_DELIVERY_ATTEMPTS`] = 3 there are two inter-attempt gaps, so the
/// delays before the 2nd and 3rd attempts are `RETRY_DELAYS_SECS[0]` (1 s) and
/// `RETRY_DELAYS_SECS[1]` (4 s).  The final entry (16 s) is reserved for the
/// hypothetical wait after the last failure and is therefore never slept.
const RETRY_DELAYS_SECS: [u64; 3] = [1, 4, 16];

/// Outcome of a single transport-level delivery attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttemptOutcome {
    /// The endpoint responded with an HTTP 2xx status.
    Success,
    /// The endpoint responded with a non-2xx status (`status` code carried for logs).
    HttpError(u16),
    /// The request failed at the transport level (connect error, timeout, …).
    TransportError,
}

/// Abstraction over the act of POSTing a signed payload to a webhook endpoint.
///
/// The production implementation ([`ReqwestTransport`]) performs a real HTTP
/// request; tests substitute a deterministic fake so the retry/backoff logic
/// can be exercised without any network access.
pub(crate) trait DeliveryTransport {
    /// Sends the signed `body` to `url`, returning the attempt outcome.
    fn deliver(
        &self,
        url: &str,
        signature: &str,
        delivery_id: &str,
        body: &str,
    ) -> impl std::future::Future<Output = AttemptOutcome> + Send;
}

/// Abstraction over the inter-attempt backoff wait.
///
/// Production sleeps on the Tokio timer; tests record the requested delays and
/// return immediately so no wall-clock time elapses.
pub(crate) trait DelaySink {
    /// Waits (or records) `delay_secs` seconds before the next attempt.
    fn wait(&self, delay_secs: u64) -> impl std::future::Future<Output = ()> + Send;
}

/// Production transport backed by a `reqwest::Client`.
pub(crate) struct ReqwestTransport(pub(crate) reqwest::Client);

impl DeliveryTransport for ReqwestTransport {
    async fn deliver(
        &self,
        url: &str,
        signature: &str,
        delivery_id: &str,
        body: &str,
    ) -> AttemptOutcome {
        match self
            .0
            .post(url)
            .header("Content-Type", "application/json")
            .header("X-Oximedia-Signature", format!("sha256={}", signature))
            .header("X-Oximedia-Delivery", delivery_id)
            .body(body.to_string())
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => AttemptOutcome::Success,
            Ok(resp) => AttemptOutcome::HttpError(resp.status().as_u16()),
            Err(_) => AttemptOutcome::TransportError,
        }
    }
}

/// Production delay sink that sleeps on the Tokio timer.
pub(crate) struct RealSleep;

impl DelaySink for RealSleep {
    async fn wait(&self, delay_secs: u64) {
        tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
    }
}

/// Summary of a completed delivery run (used by callers and tests).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeliveryReport {
    /// Final status after all attempts.
    pub status: DeliveryStatus,
    /// Total number of attempts made (1..=[`MAX_DELIVERY_ATTEMPTS`]).
    pub attempts: u32,
}

/// Transport- and clock-agnostic delivery loop.
///
/// Retries up to [`MAX_DELIVERY_ATTEMPTS`] times, waiting via `delay` before
/// each retry using the [`RETRY_DELAYS_SECS`] schedule.  Stops early on the
/// first 2xx response.  Returns the final [`DeliveryReport`].
pub(crate) async fn run_delivery_attempts<T: DeliveryTransport, D: DelaySink>(
    transport: &T,
    delay: &D,
    cfg: &WebhookConfig,
    payload: &serde_json::Value,
    delivery_id: &str,
) -> DeliveryReport {
    let body = match serde_json::to_string(payload) {
        Ok(b) => b,
        Err(e) => {
            tracing::error!("webhook delivery {}: serialize error: {}", delivery_id, e);
            return DeliveryReport {
                status: DeliveryStatus::Failed,
                attempts: 0,
            };
        }
    };
    let signature = sign_payload(&cfg.secret, &body);

    let mut attempts = 0u32;
    for attempt_index in 0..MAX_DELIVERY_ATTEMPTS {
        if attempt_index > 0 {
            // Back off using the gap *preceding* this retry: [1, 4] for attempts 2 and 3.
            let delay_secs = RETRY_DELAYS_SECS[(attempt_index - 1) as usize];
            delay.wait(delay_secs).await;
        }

        attempts += 1;
        match transport
            .deliver(&cfg.url, &signature, delivery_id, &body)
            .await
        {
            AttemptOutcome::Success => {
                tracing::info!(
                    "webhook delivery {} to {} succeeded (attempt {})",
                    delivery_id,
                    cfg.url,
                    attempts
                );
                return DeliveryReport {
                    status: DeliveryStatus::Success,
                    attempts,
                };
            }
            AttemptOutcome::HttpError(code) => {
                tracing::warn!(
                    "webhook delivery {} to {} HTTP {} (attempt {})",
                    delivery_id,
                    cfg.url,
                    code,
                    attempts,
                );
            }
            AttemptOutcome::TransportError => {
                tracing::warn!(
                    "webhook delivery {} to {} transport error (attempt {})",
                    delivery_id,
                    cfg.url,
                    attempts,
                );
            }
        }
    }

    tracing::error!(
        "webhook delivery {} to {} exhausted all retries",
        delivery_id,
        cfg.url
    );
    DeliveryReport {
        status: DeliveryStatus::Failed,
        attempts,
    }
}

/// Attempts to deliver `payload` to the configured URL, retrying on failure.
///
/// Thin production wrapper over [`run_delivery_attempts`] using a real HTTP
/// transport and Tokio-timer backoff.
async fn attempt_delivery(
    http: reqwest::Client,
    cfg: WebhookConfig,
    payload: serde_json::Value,
    delivery_id: String,
) {
    let transport = ReqwestTransport(http);
    let _report = run_delivery_attempts(&transport, &RealSleep, &cfg, &payload, &delivery_id).await;
}

/// Computes `HMAC-SHA256(secret, body)` and returns the lower-hex digest.
fn sign_payload(secret: &str, body: &str) -> String {
    // `new_from_slice` accepts keys of any length (RFC 2104 zero-pads short
    // keys / hashes down long ones) and never actually fails, but its
    // signature still returns a `Result`. Rather than `.expect()` on the
    // fallback branch too, fall back to the infallible fixed-size-key
    // constructor (an all-zero key) so this function can never panic.
    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())
        .unwrap_or_else(|_| Hmac::<Sha256>::new(&Default::default()));
    mac.update(body.as_bytes());
    mac.finalize().into_bytes().encode_hex::<String>()
}

/// Generates a cryptographically random 32-byte hex secret.
fn generate_secret() -> String {
    use rand::Rng;
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    bytes.encode_hex::<String>()
}

// ── Route handlers ────────────────────────────────────────────────────────────

/// `GET /api/v1/webhooks` — list all registered webhooks.
pub async fn list_webhooks(State(manager): State<Arc<WebhookManager>>) -> impl IntoResponse {
    let list = manager.list().await;
    Json(list)
}

/// `POST /api/v1/webhooks` — register a new webhook.
pub async fn create_webhook(
    State(manager): State<Arc<WebhookManager>>,
    Json(body): Json<RegisterWebhookRequest>,
) -> Result<impl IntoResponse, crate::error::ServerError> {
    let cfg = manager.register(body).await?;
    Ok((StatusCode::CREATED, Json(cfg)))
}

/// `GET /api/v1/webhooks/{id}` — get a single webhook.
pub async fn get_webhook(
    State(manager): State<Arc<WebhookManager>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, crate::error::ServerError> {
    let cfg = manager.get(&id).await?;
    Ok(Json(cfg))
}

/// `PUT /api/v1/webhooks/{id}` — update a webhook.
pub async fn update_webhook(
    State(manager): State<Arc<WebhookManager>>,
    Path(id): Path<String>,
    Json(body): Json<UpdateWebhookRequest>,
) -> Result<impl IntoResponse, crate::error::ServerError> {
    let cfg = manager.update(&id, body).await?;
    Ok(Json(cfg))
}

/// `DELETE /api/v1/webhooks/{id}` — remove a webhook.
pub async fn delete_webhook(
    State(manager): State<Arc<WebhookManager>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, crate::error::ServerError> {
    manager.delete(&id).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// `GET /api/v1/webhooks/{id}/deliveries` — delivery history for a webhook.
pub async fn get_webhook_deliveries(
    State(manager): State<Arc<WebhookManager>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, crate::error::ServerError> {
    // Verify the webhook exists first.
    manager.get(&id).await?;
    let deliveries = manager.deliveries_for(&id).await;
    Ok(Json(deliveries))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> WebhookManager {
        WebhookManager::new_for_test()
    }

    fn reg(url: &str, events: &[&str]) -> RegisterWebhookRequest {
        RegisterWebhookRequest {
            url: url.to_string(),
            events: events.iter().map(|s| s.to_string()).collect(),
            secret: Some("test-secret".to_string()),
        }
    }

    #[tokio::test]
    async fn test_register_returns_config() {
        let m = make_manager();
        let cfg = m
            .register(reg("https://example.com/hook", &["media.uploaded"]))
            .await
            .expect("register");
        assert!(!cfg.id.is_empty());
        assert_eq!(cfg.url, "https://example.com/hook");
        assert!(cfg.active);
    }

    #[tokio::test]
    async fn test_register_rejects_empty_url() {
        let m = make_manager();
        assert!(m.register(reg("", &["media.uploaded"])).await.is_err());
    }

    #[tokio::test]
    async fn test_register_rejects_empty_events() {
        let m = make_manager();
        let req = RegisterWebhookRequest {
            url: "https://example.com".to_string(),
            events: vec![],
            secret: None,
        };
        assert!(m.register(req).await.is_err());
    }

    #[tokio::test]
    async fn test_list_returns_registered() {
        let m = make_manager();
        m.register(reg("https://a.com", &["transcode.completed"]))
            .await
            .expect("a");
        m.register(reg("https://b.com", &["media.deleted"]))
            .await
            .expect("b");
        assert_eq!(m.list().await.len(), 2);
    }

    #[tokio::test]
    async fn test_get_existing_webhook() {
        let m = make_manager();
        let cfg = m
            .register(reg("https://example.com", &["media.uploaded"]))
            .await
            .expect("reg");
        let fetched = m.get(&cfg.id).await.expect("get");
        assert_eq!(fetched.id, cfg.id);
    }

    #[tokio::test]
    async fn test_get_missing_webhook_errors() {
        let m = make_manager();
        assert!(m.get("nonexistent-id").await.is_err());
    }

    #[tokio::test]
    async fn test_update_webhook_url() {
        let m = make_manager();
        let cfg = m
            .register(reg("https://old.com", &["media.uploaded"]))
            .await
            .expect("reg");
        let updated = m
            .update(
                &cfg.id,
                UpdateWebhookRequest {
                    url: Some("https://new.com".to_string()),
                    events: None,
                    secret: None,
                    active: None,
                },
            )
            .await
            .expect("update");
        assert_eq!(updated.url, "https://new.com");
    }

    #[tokio::test]
    async fn test_delete_webhook() {
        let m = make_manager();
        let cfg = m
            .register(reg("https://example.com", &["media.deleted"]))
            .await
            .expect("reg");
        m.delete(&cfg.id).await.expect("delete");
        assert_eq!(m.list().await.len(), 0);
    }

    #[tokio::test]
    async fn test_delete_missing_errors() {
        let m = make_manager();
        assert!(m.delete("ghost-id").await.is_err());
    }

    #[test]
    fn test_sign_payload_deterministic() {
        let s1 = sign_payload("secret", "hello");
        let s2 = sign_payload("secret", "hello");
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_sign_payload_different_keys() {
        let s1 = sign_payload("key1", "data");
        let s2 = sign_payload("key2", "data");
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_event_type_names() {
        assert_eq!(
            WebhookEvent::TranscodeCompleted {
                job_id: "j".into(),
                media_id: "m".into(),
                output_url: "u".into()
            }
            .event_type_name(),
            "transcode.completed"
        );
        assert_eq!(
            WebhookEvent::MediaDeleted {
                media_id: "m".into()
            }
            .event_type_name(),
            "media.deleted"
        );
    }

    #[test]
    fn test_delivery_status_serializes() {
        let j = serde_json::to_value(DeliveryStatus::Success).expect("serialize");
        assert_eq!(j, "success");
    }

    #[test]
    fn test_generate_secret_not_empty() {
        let s = generate_secret();
        assert!(!s.is_empty());
        assert_eq!(s.len(), 64); // 32 bytes → 64 hex chars
    }

    #[test]
    fn test_webhook_event_serializes_with_type_tag() {
        let ev = WebhookEvent::MediaUploaded {
            media_id: "m1".into(),
            filename: "foo.mp4".into(),
            size_bytes: 1024,
        };
        let j = serde_json::to_value(&ev).expect("serialize");
        assert_eq!(j["type"], "media_uploaded");
        assert_eq!(j["filename"], "foo.mp4");
    }

    // ── Retry / backoff seam tests (deterministic, no network, no sleep) ──────

    use std::sync::Mutex;

    /// Fake transport that returns a pre-programmed sequence of outcomes and
    /// records every (url, body) it was asked to deliver.
    struct ScriptedTransport {
        /// One outcome per attempt, consumed front-to-back.
        script: Mutex<std::collections::VecDeque<AttemptOutcome>>,
        /// Number of times `deliver` was invoked.
        calls: Mutex<u32>,
    }

    impl ScriptedTransport {
        fn new(outcomes: Vec<AttemptOutcome>) -> Self {
            Self {
                script: Mutex::new(outcomes.into_iter().collect()),
                calls: Mutex::new(0),
            }
        }

        fn call_count(&self) -> u32 {
            *self.calls.lock().expect("calls lock")
        }
    }

    impl DeliveryTransport for ScriptedTransport {
        async fn deliver(
            &self,
            _url: &str,
            _signature: &str,
            _delivery_id: &str,
            _body: &str,
        ) -> AttemptOutcome {
            {
                let mut c = self.calls.lock().expect("calls lock");
                *c += 1;
            }
            self.script
                .lock()
                .expect("script lock")
                .pop_front()
                // After the script is exhausted, keep failing (defensive).
                .unwrap_or(AttemptOutcome::TransportError)
        }
    }

    /// Delay sink that records every requested delay instead of sleeping.
    struct RecordingDelay {
        delays: Mutex<Vec<u64>>,
    }

    impl RecordingDelay {
        fn new() -> Self {
            Self {
                delays: Mutex::new(Vec::new()),
            }
        }

        fn recorded(&self) -> Vec<u64> {
            self.delays.lock().expect("delays lock").clone()
        }
    }

    impl DelaySink for RecordingDelay {
        async fn wait(&self, delay_secs: u64) {
            self.delays.lock().expect("delays lock").push(delay_secs);
        }
    }

    fn cfg_for_delivery() -> WebhookConfig {
        WebhookConfig {
            id: "wh-1".to_string(),
            url: "https://endpoint.invalid/hook".to_string(),
            events: vec!["media.uploaded".to_string()],
            secret: "shhh".to_string(),
            active: true,
        }
    }

    #[tokio::test]
    async fn test_retry_exhausts_three_attempts_with_backoff_schedule() {
        // Endpoint returns 5xx then a transport timeout then 5xx — never succeeds.
        let transport = ScriptedTransport::new(vec![
            AttemptOutcome::HttpError(503),
            AttemptOutcome::TransportError,
            AttemptOutcome::HttpError(500),
        ]);
        let delay = RecordingDelay::new();
        let cfg = cfg_for_delivery();
        let payload = serde_json::json!({ "type": "media_uploaded", "media_id": "m1" });

        let report = run_delivery_attempts(&transport, &delay, &cfg, &payload, "del-1").await;

        // Exactly 3 attempts were made and the final status is Failed.
        assert_eq!(report.attempts, MAX_DELIVERY_ATTEMPTS);
        assert_eq!(report.attempts, 3);
        assert_eq!(report.status, DeliveryStatus::Failed);
        assert_eq!(transport.call_count(), 3);

        // Backoff schedule: 1 s before the 2nd attempt, 4 s before the 3rd.
        // (No wall-clock time elapsed — delays were merely recorded.)
        assert_eq!(delay.recorded(), vec![1, 4]);
        // The third schedule entry (16 s) is reserved and never slept.
        assert_eq!(RETRY_DELAYS_SECS, [1, 4, 16]);
    }

    #[tokio::test]
    async fn test_retry_stops_early_on_second_attempt_success() {
        // First attempt fails (5xx), second succeeds → no 3rd attempt.
        let transport = ScriptedTransport::new(vec![
            AttemptOutcome::HttpError(502),
            AttemptOutcome::Success,
        ]);
        let delay = RecordingDelay::new();
        let cfg = cfg_for_delivery();
        let payload = serde_json::json!({ "type": "media_uploaded", "media_id": "m2" });

        let report = run_delivery_attempts(&transport, &delay, &cfg, &payload, "del-2").await;

        assert_eq!(report.status, DeliveryStatus::Success);
        assert_eq!(report.attempts, 2);
        assert_eq!(transport.call_count(), 2);
        // Only one backoff gap was needed (before the successful 2nd attempt).
        assert_eq!(delay.recorded(), vec![1]);
    }

    #[tokio::test]
    async fn test_first_attempt_success_makes_no_delay() {
        let transport = ScriptedTransport::new(vec![AttemptOutcome::Success]);
        let delay = RecordingDelay::new();
        let cfg = cfg_for_delivery();
        let payload = serde_json::json!({ "type": "media_uploaded", "media_id": "m3" });

        let report = run_delivery_attempts(&transport, &delay, &cfg, &payload, "del-3").await;

        assert_eq!(report.status, DeliveryStatus::Success);
        assert_eq!(report.attempts, 1);
        // First attempt never waits.
        assert!(delay.recorded().is_empty());
    }
}
