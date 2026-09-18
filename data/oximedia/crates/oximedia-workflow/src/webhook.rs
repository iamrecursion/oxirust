//! Outbound webhook notifier for workflow lifecycle events.
//!
//! This module provides [`WebhookNotifier`] which builds JSON payloads and
//! HMAC-SHA256 signatures for outbound HTTP POST webhooks fired when workflow
//! events occur (started, completed, failed, step completed/failed).
//!
//! This is distinct from [`crate::triggers::WebhookTrigger`] which handles
//! *inbound* webhooks that start a workflow.  The notifier sends *outbound*
//! notifications to external systems.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_workflow::webhook::{WebhookConfig, WebhookEvent, WebhookNotifier, WorkflowContext};
//!
//! let config = WebhookConfig {
//!     url: "https://example.com/hooks/workflow".to_string(),
//!     secret: Some("my-secret".to_string()),
//!     events: vec![WebhookEvent::WorkflowCompleted, WebhookEvent::WorkflowFailed],
//!     max_retries: 3,
//!     timeout_ms: 5_000,
//! };
//!
//! let notifier = WebhookNotifier::new(config);
//!
//! let ctx = WorkflowContext {
//!     workflow_id: "wf-001".to_string(),
//!     workflow_name: "transcode-pipeline".to_string(),
//!     state: "completed".to_string(),
//!     variables: std::collections::HashMap::new(),
//! };
//!
//! let payload = notifier.build_payload(&WebhookEvent::WorkflowCompleted, &ctx);
//! let signature = notifier.compute_signature(&payload);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Events that can trigger an outbound webhook notification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebhookEvent {
    /// Fired when a workflow transitions to the *running* state.
    WorkflowStarted,
    /// Fired when a workflow reaches a terminal *completed* state.
    WorkflowCompleted,
    /// Fired when a workflow reaches a terminal *failed* state.
    WorkflowFailed,
    /// Fired when a named workflow step (task) succeeds.
    StepCompleted {
        /// Name of the step that completed.
        step_name: String,
    },
    /// Fired when a named workflow step (task) fails.
    StepFailed {
        /// Name of the step that failed.
        step_name: String,
    },
}

impl WebhookEvent {
    /// A stable dot-separated event type string suitable for JSON payloads.
    #[must_use]
    pub fn event_type(&self) -> &str {
        match self {
            Self::WorkflowStarted => "workflow.started",
            Self::WorkflowCompleted => "workflow.completed",
            Self::WorkflowFailed => "workflow.failed",
            Self::StepCompleted { .. } => "step.completed",
            Self::StepFailed { .. } => "step.failed",
        }
    }

    /// Optional step name for step-level events; `None` for workflow-level events.
    #[must_use]
    pub fn step_name(&self) -> Option<&str> {
        match self {
            Self::StepCompleted { step_name } | Self::StepFailed { step_name } => {
                Some(step_name.as_str())
            }
            _ => None,
        }
    }
}

/// Configuration for an outbound webhook endpoint.
#[derive(Debug, Clone)]
pub struct WebhookConfig {
    /// Destination URL for HTTP POST notifications.
    pub url: String,
    /// Optional HMAC-SHA256 signing secret.  When set, the notifier will
    /// add an `X-Hub-Signature-256` header to each notification.
    pub secret: Option<String>,
    /// Set of events that should trigger a notification.
    /// An empty `events` list means *no* events are sent.
    pub events: Vec<WebhookEvent>,
    /// Maximum number of delivery retries on failure (caller-managed).
    pub max_retries: u32,
    /// Per-attempt timeout in milliseconds (caller-managed).
    pub timeout_ms: u64,
}

impl Default for WebhookConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            secret: None,
            events: Vec::new(),
            max_retries: 3,
            timeout_ms: 5_000,
        }
    }
}

/// Contextual information about a workflow included in every notification payload.
#[derive(Debug, Clone, Default)]
pub struct WorkflowContext {
    /// Unique workflow instance identifier.
    pub workflow_id: String,
    /// Human-readable workflow name.
    pub workflow_name: String,
    /// Current workflow state string (e.g. `"running"`, `"completed"`, `"failed"`).
    pub state: String,
    /// Arbitrary key-value variables from the workflow execution context.
    pub variables: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// WebhookNotifier
// ---------------------------------------------------------------------------

/// Builds outbound webhook payloads and signatures.
///
/// The notifier is stateless with respect to HTTP transport — it only builds
/// the JSON body and HMAC-SHA256 signature.  Callers are responsible for
/// actually sending the HTTP POST (e.g. using `reqwest` or `hyper`).
#[derive(Debug, Clone)]
pub struct WebhookNotifier {
    config: WebhookConfig,
}

impl WebhookNotifier {
    /// Create a new notifier with the given configuration.
    #[must_use]
    pub fn new(config: WebhookConfig) -> Self {
        Self { config }
    }

    /// Returns a reference to the underlying configuration.
    #[must_use]
    pub fn config(&self) -> &WebhookConfig {
        &self.config
    }

    /// Build the JSON payload string for the given `event` and `context`.
    ///
    /// The payload is a JSON object with the following fields:
    /// - `event_type`: dot-separated event name (see [`WebhookEvent::event_type`])
    /// - `workflow_id`: from `context.workflow_id`
    /// - `workflow_name`: from `context.workflow_name`
    /// - `state`: from `context.state`
    /// - `timestamp_ms`: Unix epoch in milliseconds
    /// - `variables`: from `context.variables`
    /// - `step_name` *(optional)*: only present for step-level events
    #[must_use]
    pub fn build_payload(&self, event: &WebhookEvent, context: &WorkflowContext) -> String {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();

        let mut payload = serde_json::json!({
            "event_type": event.event_type(),
            "workflow_id": context.workflow_id,
            "workflow_name": context.workflow_name,
            "state": context.state,
            "timestamp_ms": timestamp_ms,
            "variables": context.variables,
        });

        if let Some(step) = event.step_name() {
            if let Some(obj) = payload.as_object_mut() {
                obj.insert(
                    "step_name".to_string(),
                    serde_json::Value::String(step.to_string()),
                );
            }
        }

        payload.to_string()
    }

    /// Compute HMAC-SHA256 of `payload` using the configured secret.
    ///
    /// Returns `None` when no secret is configured, or `Some(hex_string)` (64
    /// lowercase hex characters) when a secret is set.
    #[must_use]
    pub fn compute_signature(&self, payload: &str) -> Option<String> {
        self.config
            .secret
            .as_ref()
            .map(|secret| hmac_sha256(secret.as_bytes(), payload.as_bytes()))
    }

    /// Returns `true` when the notifier is configured to send a notification for
    /// the given event.
    ///
    /// Matching is done by event type string so that, for example, any
    /// `StepCompleted { .. }` event matches a `StepCompleted { step_name: _ }` entry.
    #[must_use]
    pub fn should_notify(&self, event: &WebhookEvent) -> bool {
        self.config
            .events
            .iter()
            .any(|e| e.event_type() == event.event_type())
    }

    /// Build an HTTP headers map for a notification.
    ///
    /// Always includes:
    /// - `Content-Type: application/json`
    ///
    /// When a secret is configured, also includes:
    /// - `X-Hub-Signature-256: <hex-digest>`
    #[must_use]
    pub fn build_headers(&self, payload: &str) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());

        if let Some(sig) = self.compute_signature(payload) {
            headers.insert("X-Hub-Signature-256".to_string(), sig);
        }

        headers
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust HMAC-SHA256 (private — mirrors triggers.rs implementation)
// ---------------------------------------------------------------------------

/// Compute HMAC-SHA256 of `message` keyed with `key`, returned as lowercase hex.
fn hmac_sha256(key: &[u8], message: &[u8]) -> String {
    const BLOCK: usize = 64;

    let mut k = [0u8; BLOCK];
    if key.len() > BLOCK {
        let h = sha256(key);
        k[..32].copy_from_slice(&h);
    } else {
        k[..key.len()].copy_from_slice(key);
    }

    let mut i_key_pad = [0u8; BLOCK];
    let mut o_key_pad = [0u8; BLOCK];
    for i in 0..BLOCK {
        i_key_pad[i] = k[i] ^ 0x36;
        o_key_pad[i] = k[i] ^ 0x5c;
    }

    let mut inner_input = Vec::with_capacity(BLOCK + message.len());
    inner_input.extend_from_slice(&i_key_pad);
    inner_input.extend_from_slice(message);
    let inner_hash = sha256(&inner_input);

    let mut outer_input = Vec::with_capacity(BLOCK + 32);
    outer_input.extend_from_slice(&o_key_pad);
    outer_input.extend_from_slice(&inner_hash);
    let outer_hash = sha256(&outer_input);

    outer_hash
        .iter()
        .fold(String::with_capacity(64), |mut s, b| {
            s.push_str(&format!("{b:02x}"));
            s
        })
}

/// Minimal pure-Rust SHA-256 (NIST FIPS 180-4).
#[allow(clippy::many_single_char_names)]
fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];

    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for (i, word_bytes) in chunk.chunks(4).enumerate().take(16) {
            w[i] = u32::from_be_bytes([word_bytes[0], word_bytes[1], word_bytes[2], word_bytes[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut digest = [0u8; 32];
    for (i, &word) in h.iter().enumerate() {
        digest[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_notifier(events: Vec<WebhookEvent>, secret: Option<&str>) -> WebhookNotifier {
        WebhookNotifier::new(WebhookConfig {
            url: "https://example.com/hooks/workflow".to_string(),
            secret: secret.map(str::to_string),
            events,
            max_retries: 3,
            timeout_ms: 5_000,
        })
    }

    fn make_context() -> WorkflowContext {
        WorkflowContext {
            workflow_id: "wf-001".to_string(),
            workflow_name: "transcode-pipeline".to_string(),
            state: "completed".to_string(),
            variables: HashMap::new(),
        }
    }

    #[test]
    fn test_webhook_event_types() {
        assert_eq!(
            WebhookEvent::WorkflowStarted.event_type(),
            "workflow.started"
        );
        assert_eq!(
            WebhookEvent::WorkflowCompleted.event_type(),
            "workflow.completed"
        );
        assert_eq!(WebhookEvent::WorkflowFailed.event_type(), "workflow.failed");
        assert_eq!(
            WebhookEvent::StepCompleted {
                step_name: "encode".to_string()
            }
            .event_type(),
            "step.completed"
        );
        assert_eq!(
            WebhookEvent::StepFailed {
                step_name: "qc-check".to_string()
            }
            .event_type(),
            "step.failed"
        );
    }

    #[test]
    fn test_webhook_event_step_name() {
        let ev = WebhookEvent::StepCompleted {
            step_name: "encode".to_string(),
        };
        assert_eq!(ev.step_name(), Some("encode"));
        assert!(WebhookEvent::WorkflowStarted.step_name().is_none());
    }

    #[test]
    fn test_build_payload_workflow_started() {
        let notifier = make_notifier(vec![WebhookEvent::WorkflowStarted], None);
        let ctx = make_context();
        let payload = notifier.build_payload(&WebhookEvent::WorkflowStarted, &ctx);

        let parsed: serde_json::Value =
            serde_json::from_str(&payload).expect("payload must be valid JSON");

        assert_eq!(parsed["event_type"], "workflow.started");
        assert_eq!(parsed["workflow_id"], "wf-001");
        assert_eq!(parsed["workflow_name"], "transcode-pipeline");
        assert_eq!(parsed["state"], "completed");
        assert!(parsed["timestamp_ms"].is_number());
        assert!(parsed["variables"].is_object());
        // step_name should NOT be present for workflow-level events
        assert!(parsed.get("step_name").is_none());
    }

    #[test]
    fn test_build_payload_step_completed_includes_step_name() {
        let notifier = make_notifier(vec![], None);
        let ctx = make_context();
        let ev = WebhookEvent::StepCompleted {
            step_name: "encode".to_string(),
        };
        let payload = notifier.build_payload(&ev, &ctx);

        let parsed: serde_json::Value =
            serde_json::from_str(&payload).expect("payload must be valid JSON");
        assert_eq!(parsed["step_name"], "encode");
        assert_eq!(parsed["event_type"], "step.completed");
    }

    #[test]
    fn test_compute_signature_no_secret_returns_none() {
        let notifier = make_notifier(vec![], None);
        let sig = notifier.compute_signature("payload");
        assert!(sig.is_none(), "no secret should produce None signature");
    }

    #[test]
    fn test_compute_signature_with_secret_returns_64_hex_chars() {
        let notifier = make_notifier(vec![], Some("s3cr3t"));
        let sig = notifier.compute_signature("some payload");
        let sig_str = sig.expect("should have signature");
        assert_eq!(sig_str.len(), 64, "HMAC-SHA256 hex should be 64 chars");
        assert!(
            sig_str.chars().all(|c| c.is_ascii_hexdigit()),
            "signature should be hex digits"
        );
    }

    #[test]
    fn test_compute_signature_deterministic() {
        let notifier = make_notifier(vec![], Some("key"));
        let sig1 = notifier.compute_signature("hello");
        let sig2 = notifier.compute_signature("hello");
        assert_eq!(sig1, sig2, "same input should produce same signature");
    }

    #[test]
    fn test_compute_signature_different_payloads_differ() {
        let notifier = make_notifier(vec![], Some("key"));
        let sig1 = notifier.compute_signature("hello");
        let sig2 = notifier.compute_signature("world");
        assert_ne!(
            sig1, sig2,
            "different payloads should produce different signatures"
        );
    }

    #[test]
    fn test_should_notify_matching_event() {
        let notifier = make_notifier(
            vec![
                WebhookEvent::WorkflowCompleted,
                WebhookEvent::WorkflowFailed,
            ],
            None,
        );
        assert!(notifier.should_notify(&WebhookEvent::WorkflowCompleted));
        assert!(notifier.should_notify(&WebhookEvent::WorkflowFailed));
    }

    #[test]
    fn test_should_notify_non_matching_event() {
        let notifier = make_notifier(vec![WebhookEvent::WorkflowCompleted], None);
        assert!(!notifier.should_notify(&WebhookEvent::WorkflowStarted));
        assert!(!notifier.should_notify(&WebhookEvent::WorkflowFailed));
    }

    #[test]
    fn test_should_notify_step_event_matches_by_type() {
        // Any StepCompleted event should match if StepCompleted is in the list
        let notifier = make_notifier(
            vec![WebhookEvent::StepCompleted {
                step_name: "*".to_string(),
            }],
            None,
        );
        assert!(notifier.should_notify(&WebhookEvent::StepCompleted {
            step_name: "encode".to_string()
        }));
        assert!(notifier.should_notify(&WebhookEvent::StepCompleted {
            step_name: "qc".to_string()
        }));
    }

    #[test]
    fn test_should_notify_empty_events_returns_false() {
        let notifier = make_notifier(vec![], None);
        assert!(!notifier.should_notify(&WebhookEvent::WorkflowStarted));
    }

    #[test]
    fn test_build_headers_without_secret() {
        let notifier = make_notifier(vec![], None);
        let headers = notifier.build_headers("payload");
        assert_eq!(
            headers.get("Content-Type").map(String::as_str),
            Some("application/json")
        );
        assert!(!headers.contains_key("X-Hub-Signature-256"));
    }

    #[test]
    fn test_build_headers_with_secret() {
        let notifier = make_notifier(vec![], Some("secret"));
        let headers = notifier.build_headers("test payload");
        assert_eq!(
            headers.get("Content-Type").map(String::as_str),
            Some("application/json")
        );
        let sig = headers
            .get("X-Hub-Signature-256")
            .expect("signature header should be present");
        assert_eq!(sig.len(), 64);
    }

    #[test]
    fn test_build_headers_signature_matches_compute_signature() {
        let notifier = make_notifier(vec![], Some("my-key"));
        let payload = "test-payload";
        let headers = notifier.build_headers(payload);
        let expected = notifier
            .compute_signature(payload)
            .expect("should have sig");
        let actual = headers
            .get("X-Hub-Signature-256")
            .expect("should have header");
        assert_eq!(*actual, expected);
    }

    #[test]
    fn test_build_payload_with_variables() {
        let notifier = make_notifier(vec![], None);
        let mut vars = HashMap::new();
        vars.insert(
            "output_path".to_string(),
            serde_json::json!("/out/clip.mp4"),
        );
        vars.insert("duration_secs".to_string(), serde_json::json!(120));
        let ctx = WorkflowContext {
            workflow_id: "wf-42".to_string(),
            workflow_name: "ingest".to_string(),
            state: "running".to_string(),
            variables: vars,
        };
        let payload = notifier.build_payload(&WebhookEvent::WorkflowStarted, &ctx);
        let parsed: serde_json::Value = serde_json::from_str(&payload).expect("valid JSON");
        assert_eq!(parsed["variables"]["output_path"], "/out/clip.mp4");
        assert_eq!(parsed["variables"]["duration_secs"], 120);
    }

    #[test]
    fn test_sha256_known_empty_value() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let hash = sha256(b"");
        let hex: String = hash.iter().fold(String::new(), |mut s, b| {
            s.push_str(&format!("{b:02x}"));
            s
        });
        assert_eq!(
            hex,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_hmac_sha256_is_64_hex_chars() {
        let mac = hmac_sha256(b"key", b"message");
        assert_eq!(mac.len(), 64);
    }
}
