//! Workflow trigger system.
//!
//! Provides flexible trigger types for initiating workflow execution,
//! including schedule-based, file arrival, API, event-based, and webhook triggers.
//!
//! # Webhook Triggers
//!
//! The `Webhook` trigger variant allows external systems to start a workflow by
//! sending an HTTP POST request to a registered endpoint. The
//! [`WebhookTrigger`] struct describes the expected path, optional HMAC-SHA256
//! secret validation, and field filters that must be present in the JSON body.
//!
//! The [`WebhookRouter`] provides in-process matching: given a raw HTTP path and
//! JSON body, it returns the workflow IDs whose webhook triggers are satisfied.
//! Actual HTTP server integration is left to the caller (axum, hyper, etc.).

#![allow(dead_code)]

use std::collections::HashMap;

/// Trigger type for workflow execution.
#[derive(Debug, Clone)]
pub enum TriggerType {
    /// Cron-style schedule trigger.
    Schedule(ScheduleTrigger),
    /// File arrival trigger.
    FileArrival(FileArrivalTrigger),
    /// API call trigger.
    ApiCall,
    /// Event-based trigger.
    EventBased(EventTrigger),
    /// Manual start trigger.
    ManualStart,
    /// Dependency-based trigger.
    Dependency,
    /// HTTP POST webhook trigger.
    Webhook(WebhookTrigger),
}

/// Schedule trigger using cron expressions.
#[derive(Debug, Clone)]
pub struct ScheduleTrigger {
    /// Cron expression (e.g. "0 9 * * 1-5" for weekdays at 9am).
    pub cron_expr: String,
    /// Timezone identifier (e.g. "UTC", "`America/New_York`").
    pub timezone: String,
    /// Maximum number of runs (None = unlimited).
    pub max_runs: Option<u32>,
}

impl ScheduleTrigger {
    /// Create a new schedule trigger.
    #[must_use]
    pub fn new(cron_expr: impl Into<String>, timezone: impl Into<String>) -> Self {
        Self {
            cron_expr: cron_expr.into(),
            timezone: timezone.into(),
            max_runs: None,
        }
    }

    /// Set maximum runs.
    #[must_use]
    pub fn with_max_runs(mut self, max_runs: u32) -> Self {
        self.max_runs = Some(max_runs);
        self
    }

    /// Calculate next fire time in milliseconds.
    ///
    /// Simplified implementation: parses HH:MM from cron expression
    /// (fields: second minute hour day month weekday).
    /// Returns the next fire time in ms from `now_ms`.
    #[must_use]
    pub fn next_fire_ms(&self, now_ms: u64) -> u64 {
        // Parse HH:MM from cron: expect format "S M H ..."
        // We extract field index 2 (hour) and 1 (minute).
        let parts: Vec<&str> = self.cron_expr.split_whitespace().collect();
        if parts.len() < 3 {
            // Default: fire in 1 hour
            return now_ms + 3_600_000;
        }

        let minute: u64 = parts[1].parse().unwrap_or(0);
        let hour: u64 = parts[2].parse().unwrap_or(0);

        // Current time components from ms
        let now_secs = now_ms / 1000;
        let seconds_in_day = now_secs % 86400;
        let current_hour = seconds_in_day / 3600;
        let current_minute = (seconds_in_day % 3600) / 60;
        let day_start_ms = now_ms - (seconds_in_day * 1000);

        let target_ms = day_start_ms + hour * 3_600_000 + minute * 60_000;

        if target_ms > now_ms {
            target_ms
        } else if hour == current_hour && minute == current_minute {
            // Same minute - fire in next minute
            now_ms + 60_000
        } else {
            // Tomorrow same time
            target_ms + 86_400_000
        }
    }
}

/// File arrival trigger configuration.
#[derive(Debug, Clone)]
pub struct FileArrivalTrigger {
    /// Directory path to watch.
    pub watch_path: String,
    /// File pattern to match (glob-style, supports `*` wildcard).
    pub pattern: String,
    /// Minimum file size in bytes.
    pub min_size_bytes: u64,
    /// Wait for file to be stable for this many seconds.
    pub stable_for_secs: u32,
}

impl FileArrivalTrigger {
    /// Create a new file arrival trigger.
    #[must_use]
    pub fn new(
        watch_path: impl Into<String>,
        pattern: impl Into<String>,
        min_size_bytes: u64,
        stable_for_secs: u32,
    ) -> Self {
        Self {
            watch_path: watch_path.into(),
            pattern: pattern.into(),
            min_size_bytes,
            stable_for_secs,
        }
    }

    /// Check if a file path and size matches this trigger's criteria.
    ///
    /// Supports glob-style pattern with `*` wildcard matching any sequence of characters.
    #[must_use]
    pub fn matches(&self, path: &str, size: u64) -> bool {
        if size < self.min_size_bytes {
            return false;
        }

        // Extract filename from path
        let filename = path.rsplit('/').next().unwrap_or(path);

        glob_match(&self.pattern, filename)
    }
}

/// Glob-style pattern matching with `*` wildcard.
fn glob_match(pattern: &str, text: &str) -> bool {
    let pattern_bytes = pattern.as_bytes();
    let text_bytes = text.as_bytes();
    glob_match_inner(pattern_bytes, text_bytes)
}

fn glob_match_inner(pattern: &[u8], text: &[u8]) -> bool {
    match (pattern.first(), text.first()) {
        (None, None) => true,
        (Some(&b'*'), _) => {
            // Try matching * with 0 characters, then 1, 2, ... characters
            glob_match_inner(&pattern[1..], text)
                || (!text.is_empty() && glob_match_inner(pattern, &text[1..]))
        }
        (None, Some(_)) | (Some(_), None) => false,
        (Some(&p), Some(&t)) => p == t && glob_match_inner(&pattern[1..], &text[1..]),
    }
}

/// Event-based trigger.
#[derive(Debug, Clone)]
pub struct EventTrigger {
    /// Type of event to watch for.
    pub event_type: String,
    /// Key-value filter conditions that must all match.
    pub filter: HashMap<String, String>,
}

impl EventTrigger {
    /// Create a new event trigger.
    #[must_use]
    pub fn new(event_type: impl Into<String>) -> Self {
        Self {
            event_type: event_type.into(),
            filter: HashMap::new(),
        }
    }

    /// Add a filter condition.
    #[must_use]
    pub fn with_filter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.filter.insert(key.into(), value.into());
        self
    }

    /// Check if an event matches this trigger's conditions.
    #[must_use]
    pub fn matches(&self, event: &HashMap<String, String>) -> bool {
        for (key, expected_value) in &self.filter {
            match event.get(key) {
                Some(actual_value) if actual_value == expected_value => {}
                _ => return false,
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Webhook trigger
// ---------------------------------------------------------------------------

/// Configuration for an HTTP POST webhook trigger.
///
/// When an HTTP POST arrives at `path`, the JSON body is decoded and checked:
/// 1. If `secret` is set, the `X-Hub-Signature-256` header must contain a valid
///    HMAC-SHA256 signature of the raw body (pure-Rust implementation).
/// 2. All entries in `required_fields` must be present in the JSON object with
///    the expected string values.
///
/// Real HTTP integration is the caller's responsibility; see [`WebhookRouter`].
#[derive(Debug, Clone)]
pub struct WebhookTrigger {
    /// URL path that activates this trigger (e.g. `"/webhooks/ingest-ready"`).
    pub path: String,
    /// Optional HMAC-SHA256 secret for signature verification.
    pub secret: Option<String>,
    /// JSON body fields that must match to fire the trigger.
    pub required_fields: HashMap<String, String>,
    /// Optional description.
    pub description: String,
}

impl WebhookTrigger {
    /// Create a new webhook trigger for the given path.
    #[must_use]
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            secret: None,
            required_fields: HashMap::new(),
            description: String::new(),
        }
    }

    /// Set an HMAC-SHA256 secret for signature verification.
    #[must_use]
    pub fn with_secret(mut self, secret: impl Into<String>) -> Self {
        self.secret = Some(secret.into());
        self
    }

    /// Require a JSON body field to have a specific value.
    #[must_use]
    pub fn with_required_field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.required_fields.insert(key.into(), value.into());
        self
    }

    /// Set a human-readable description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Verify the HMAC-SHA256 signature of a raw payload.
    ///
    /// `signature` is the hex-encoded digest (e.g. the value of the
    /// `X-Hub-Signature-256` header without any `sha256=` prefix).
    ///
    /// Returns `true` when the signature is valid *or* no secret is configured.
    #[must_use]
    pub fn verify_signature(&self, payload: &[u8], signature: &str) -> bool {
        let Some(ref secret) = self.secret else {
            return true; // no secret configured → always valid
        };
        let expected = hmac_sha256(secret.as_bytes(), payload);
        // Compare in constant time to prevent timing attacks.
        constant_time_eq(&expected, signature.trim())
    }

    /// Check whether a parsed JSON body satisfies the required-fields filter.
    #[must_use]
    pub fn body_matches(&self, body: &serde_json::Value) -> bool {
        for (key, expected) in &self.required_fields {
            match body.get(key).and_then(|v| v.as_str()) {
                Some(actual) if actual == expected.as_str() => {}
                _ => return false,
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// HMAC-SHA256 pure-Rust implementation (no external crypto crate)
// ---------------------------------------------------------------------------

/// Compute HMAC-SHA256 of `message` keyed with `key`.
///
/// Returns the hex-encoded digest.
fn hmac_sha256(key: &[u8], message: &[u8]) -> String {
    // SHA-256 block size = 64 bytes
    const BLOCK: usize = 64;

    // Prepare K: truncate or hash key if longer than block size.
    let mut k = [0u8; BLOCK];
    if key.len() > BLOCK {
        let h = sha256(key);
        k[..32].copy_from_slice(&h);
    } else {
        k[..key.len()].copy_from_slice(key);
    }

    // i_pad and o_pad XOR
    let mut i_key_pad = [0u8; BLOCK];
    let mut o_key_pad = [0u8; BLOCK];
    for i in 0..BLOCK {
        i_key_pad[i] = k[i] ^ 0x36;
        o_key_pad[i] = k[i] ^ 0x5c;
    }

    // inner = SHA256(i_key_pad || message)
    let mut inner_input = Vec::with_capacity(BLOCK + message.len());
    inner_input.extend_from_slice(&i_key_pad);
    inner_input.extend_from_slice(message);
    let inner_hash = sha256(&inner_input);

    // outer = SHA256(o_key_pad || inner)
    let mut outer_input = Vec::with_capacity(BLOCK + 32);
    outer_input.extend_from_slice(&o_key_pad);
    outer_input.extend_from_slice(&inner_hash);
    let outer_hash = sha256(&outer_input);

    // Hex-encode
    outer_hash
        .iter()
        .fold(String::with_capacity(64), |mut s, b| {
            s.push_str(&format!("{b:02x}"));
            s
        })
}

/// Minimal SHA-256 implementation (NIST FIPS 180-4).
#[allow(clippy::many_single_char_names)]
fn sha256(data: &[u8]) -> [u8; 32] {
    // Initial hash values (first 32 bits of fractional parts of sqrt of first 8 primes)
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // Round constants (first 32 bits of fractional parts of cbrt of first 64 primes)
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    // Pre-processing: adding padding bits
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit (64-byte) chunk
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

/// Constant-time byte equality to prevent timing attacks.
fn constant_time_eq(a: &str, b: &str) -> bool {
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    if ab.len() != bb.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in ab.iter().zip(bb.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ---------------------------------------------------------------------------
// WebhookRouter
// ---------------------------------------------------------------------------

/// Routes incoming HTTP POST webhooks to the matching registered workflow IDs.
///
/// This is an in-process routing layer. Callers are expected to extract the
/// HTTP path, raw body bytes, and optional signature header, then call
/// [`WebhookRouter::route`] to obtain the list of matching workflow IDs.
#[derive(Debug, Default)]
pub struct WebhookRouter {
    /// Map from workflow ID to its list of webhook triggers.
    routes: HashMap<String, Vec<WebhookTrigger>>,
}

impl WebhookRouter {
    /// Create an empty router.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a webhook trigger for a workflow.
    pub fn register(&mut self, workflow_id: impl Into<String>, trigger: WebhookTrigger) {
        self.routes
            .entry(workflow_id.into())
            .or_default()
            .push(trigger);
    }

    /// Remove all webhook triggers for a workflow.
    pub fn remove(&mut self, workflow_id: &str) {
        self.routes.remove(workflow_id);
    }

    /// Evaluate an incoming HTTP POST and return workflow IDs that should be triggered.
    ///
    /// # Parameters
    /// - `path`: The URL path of the request (e.g. `"/webhooks/ingest-ready"`).
    /// - `body_bytes`: Raw request body bytes (used for HMAC verification).
    /// - `body_json`: Parsed JSON body (used for field matching).
    /// - `signature`: Optional value of the `X-Hub-Signature-256` header.
    ///
    /// A workflow is triggered when **at least one** of its webhook triggers
    /// matches the path, passes signature verification, and satisfies all
    /// required body fields.
    #[must_use]
    pub fn route(
        &self,
        path: &str,
        body_bytes: &[u8],
        body_json: &serde_json::Value,
        signature: Option<&str>,
    ) -> Vec<String> {
        let mut triggered = Vec::new();

        for (workflow_id, triggers) in &self.routes {
            for trigger in triggers {
                if trigger.path != path {
                    continue;
                }
                // Verify signature if a secret is configured
                let sig = signature.unwrap_or("");
                if !trigger.verify_signature(body_bytes, sig) {
                    continue;
                }
                if trigger.body_matches(body_json) {
                    triggered.push(workflow_id.clone());
                    break; // Only trigger once per workflow
                }
            }
        }

        triggered
    }

    /// List all registered workflow IDs.
    #[must_use]
    pub fn workflow_ids(&self) -> Vec<&str> {
        self.routes.keys().map(String::as_str).collect()
    }

    /// Count total registered triggers across all workflows.
    #[must_use]
    pub fn trigger_count(&self) -> usize {
        self.routes.values().map(Vec::len).sum()
    }
}

/// Condition that combines multiple triggers.
#[derive(Debug, Clone)]
pub enum TriggerCondition {
    /// All triggers must fire.
    All(Vec<TriggerType>),
    /// Any trigger can fire.
    Any(Vec<TriggerType>),
    /// No triggers should fire (negation).
    None(Vec<TriggerType>),
}

/// Engine that manages triggers and evaluates them.
#[derive(Debug, Default)]
pub struct TriggerEngine {
    /// Map from workflow ID to its list of triggers.
    triggers: HashMap<String, Vec<TriggerType>>,
}

impl TriggerEngine {
    /// Create a new trigger engine.
    #[must_use]
    pub fn new() -> Self {
        Self {
            triggers: HashMap::new(),
        }
    }

    /// Add a trigger for a workflow.
    pub fn add_trigger(&mut self, workflow_id: &str, trigger: TriggerType) {
        self.triggers
            .entry(workflow_id.to_string())
            .or_default()
            .push(trigger);
    }

    /// Evaluate file arrival event and return workflow IDs that should be triggered.
    #[must_use]
    pub fn evaluate_file_arrival(&self, path: &str, size: u64) -> Vec<String> {
        let mut triggered = Vec::new();

        for (workflow_id, triggers) in &self.triggers {
            for trigger in triggers {
                if let TriggerType::FileArrival(file_trigger) = trigger {
                    if file_trigger.matches(path, size) {
                        triggered.push(workflow_id.clone());
                        break; // Only trigger once per workflow
                    }
                }
            }
        }

        triggered
    }

    /// Evaluate an event and return workflow IDs that should be triggered.
    #[must_use]
    pub fn evaluate_event(&self, event: &HashMap<String, String>) -> Vec<String> {
        let mut triggered = Vec::new();

        for (workflow_id, triggers) in &self.triggers {
            for trigger in triggers {
                if let TriggerType::EventBased(event_trigger) = trigger {
                    if event_trigger.matches(event) {
                        triggered.push(workflow_id.clone());
                        break;
                    }
                }
            }
        }

        triggered
    }

    /// Get triggers for a workflow.
    #[must_use]
    pub fn get_triggers(&self, workflow_id: &str) -> &[TriggerType] {
        self.triggers.get(workflow_id).map_or(&[], Vec::as_slice)
    }

    /// Remove all triggers for a workflow.
    pub fn remove_workflow(&mut self, workflow_id: &str) {
        self.triggers.remove(workflow_id);
    }

    /// List all registered workflow IDs.
    #[must_use]
    pub fn workflow_ids(&self) -> Vec<&str> {
        self.triggers.keys().map(String::as_str).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_trigger_next_fire_future_today() {
        // Set now to 8:00 AM (in ms), trigger at 9:00 AM
        let now_ms = 8 * 3_600_000_u64; // 8h in ms from day start
        let trigger = ScheduleTrigger::new("0 0 9 * * *", "UTC");
        let next = trigger.next_fire_ms(now_ms);
        // Should be 9:00 AM = 9 * 3_600_000
        assert_eq!(next, 9 * 3_600_000);
    }

    #[test]
    fn test_schedule_trigger_next_fire_tomorrow() {
        // Set now to 10:00 AM, trigger at 9:00 AM → tomorrow
        let now_ms = 10 * 3_600_000_u64;
        let trigger = ScheduleTrigger::new("0 0 9 * * *", "UTC");
        let next = trigger.next_fire_ms(now_ms);
        assert!(next > now_ms);
    }

    #[test]
    fn test_schedule_trigger_with_max_runs() {
        let trigger = ScheduleTrigger::new("0 0 9 * * *", "UTC").with_max_runs(5);
        assert_eq!(trigger.max_runs, Some(5));
    }

    #[test]
    fn test_file_arrival_trigger_matches_pattern() {
        let trigger = FileArrivalTrigger::new("/watch", "*.mp4", 1000, 5);
        assert!(trigger.matches("/watch/video.mp4", 5000));
        assert!(!trigger.matches("/watch/video.mp4", 500)); // too small
        assert!(!trigger.matches("/watch/video.mov", 5000)); // wrong ext
    }

    #[test]
    fn test_file_arrival_trigger_wildcard_pattern() {
        let trigger = FileArrivalTrigger::new("/ingest", "mxf_*_v2.mxf", 0, 0);
        assert!(trigger.matches("/ingest/mxf_cam1_v2.mxf", 0));
        assert!(!trigger.matches("/ingest/mxf_cam1_v1.mxf", 0));
    }

    #[test]
    fn test_file_arrival_trigger_no_wildcard() {
        let trigger = FileArrivalTrigger::new("/dir", "exact.mp4", 0, 0);
        assert!(trigger.matches("/dir/exact.mp4", 0));
        assert!(!trigger.matches("/dir/other.mp4", 0));
    }

    #[test]
    fn test_event_trigger_matches() {
        let trigger = EventTrigger::new("media.ready")
            .with_filter("format", "mp4")
            .with_filter("resolution", "4k");

        let mut event = HashMap::new();
        event.insert("format".to_string(), "mp4".to_string());
        event.insert("resolution".to_string(), "4k".to_string());
        event.insert("extra_field".to_string(), "ignored".to_string());

        assert!(trigger.matches(&event));
    }

    #[test]
    fn test_event_trigger_no_match() {
        let trigger = EventTrigger::new("media.ready").with_filter("format", "mp4");

        let mut event = HashMap::new();
        event.insert("format".to_string(), "mov".to_string());

        assert!(!trigger.matches(&event));
    }

    #[test]
    fn test_event_trigger_empty_filter() {
        let trigger = EventTrigger::new("any.event");
        let event = HashMap::new();
        assert!(trigger.matches(&event));
    }

    #[test]
    fn test_trigger_engine_add_and_evaluate_file() {
        let mut engine = TriggerEngine::new();
        engine.add_trigger(
            "workflow-1",
            TriggerType::FileArrival(FileArrivalTrigger::new("/ingest", "*.mxf", 1000, 5)),
        );
        engine.add_trigger(
            "workflow-2",
            TriggerType::FileArrival(FileArrivalTrigger::new("/ingest", "*.mp4", 1000, 5)),
        );

        let triggered = engine.evaluate_file_arrival("/ingest/clip.mxf", 50_000);
        assert_eq!(triggered.len(), 1);
        assert_eq!(triggered[0], "workflow-1");
    }

    #[test]
    fn test_trigger_engine_multiple_workflows() {
        let mut engine = TriggerEngine::new();
        engine.add_trigger(
            "wf-a",
            TriggerType::FileArrival(FileArrivalTrigger::new("/watch", "*.mp4", 0, 0)),
        );
        engine.add_trigger(
            "wf-b",
            TriggerType::FileArrival(FileArrivalTrigger::new("/watch", "*.mp4", 0, 0)),
        );

        let triggered = engine.evaluate_file_arrival("/watch/test.mp4", 1);
        assert_eq!(triggered.len(), 2);
    }

    #[test]
    fn test_trigger_engine_no_match() {
        let mut engine = TriggerEngine::new();
        engine.add_trigger(
            "wf-1",
            TriggerType::FileArrival(FileArrivalTrigger::new("/watch", "*.mxf", 0, 0)),
        );

        let triggered = engine.evaluate_file_arrival("/watch/test.mp4", 1);
        assert!(triggered.is_empty());
    }

    #[test]
    fn test_trigger_engine_remove_workflow() {
        let mut engine = TriggerEngine::new();
        engine.add_trigger(
            "wf-1",
            TriggerType::FileArrival(FileArrivalTrigger::new("/watch", "*.mp4", 0, 0)),
        );

        engine.remove_workflow("wf-1");

        let triggered = engine.evaluate_file_arrival("/watch/test.mp4", 1);
        assert!(triggered.is_empty());
    }

    #[test]
    fn test_trigger_condition_variants() {
        let triggers = vec![TriggerType::ManualStart, TriggerType::ApiCall];
        let _all = TriggerCondition::All(triggers.clone());
        let _any = TriggerCondition::Any(triggers.clone());
        let _none = TriggerCondition::None(triggers);
        // Just verify construction works
    }

    #[test]
    fn test_glob_match_star_extension() {
        assert!(glob_match("*.mp4", "video.mp4"));
        assert!(glob_match("*.mp4", ".mp4"));
        assert!(!glob_match("*.mp4", "video.mov"));
    }

    #[test]
    fn test_glob_match_multiple_stars() {
        assert!(glob_match("*_*_v2.*", "clip_cam1_v2.mxf"));
        assert!(!glob_match("*_*_v2.*", "clip_cam1_v1.mxf"));
    }

    // -----------------------------------------------------------------------
    // WebhookTrigger
    // -----------------------------------------------------------------------

    #[test]
    fn test_webhook_trigger_basic_match() {
        let trigger =
            WebhookTrigger::new("/hooks/ingest").with_required_field("event", "ingest.ready");

        let body = serde_json::json!({"event": "ingest.ready", "asset": "clip-001"});
        assert!(trigger.body_matches(&body));
    }

    #[test]
    fn test_webhook_trigger_field_mismatch() {
        let trigger =
            WebhookTrigger::new("/hooks/ingest").with_required_field("event", "ingest.ready");

        let body = serde_json::json!({"event": "transcode.done"});
        assert!(!trigger.body_matches(&body));
    }

    #[test]
    fn test_webhook_trigger_missing_field() {
        let trigger =
            WebhookTrigger::new("/hooks/ingest").with_required_field("event", "ingest.ready");

        let body = serde_json::json!({"other": "value"});
        assert!(!trigger.body_matches(&body));
    }

    #[test]
    fn test_webhook_trigger_no_secret_always_valid() {
        let trigger = WebhookTrigger::new("/hooks/test");
        assert!(trigger.verify_signature(b"payload", ""));
        assert!(trigger.verify_signature(b"payload", "any_sig"));
    }

    #[test]
    fn test_webhook_trigger_with_secret_valid() {
        let trigger = WebhookTrigger::new("/hooks/test").with_secret("mysecret");
        let payload = b"hello world";
        // Compute expected HMAC
        let expected = hmac_sha256(b"mysecret", payload);
        assert!(trigger.verify_signature(payload, &expected));
    }

    #[test]
    fn test_webhook_trigger_with_secret_invalid() {
        let trigger = WebhookTrigger::new("/hooks/test").with_secret("mysecret");
        assert!(!trigger.verify_signature(b"payload", "wrong_signature"));
    }

    #[test]
    fn test_webhook_trigger_builder_chain() {
        let trigger = WebhookTrigger::new("/hooks/media")
            .with_secret("s3cr3t")
            .with_required_field("type", "video")
            .with_description("Video ingest webhook");

        assert_eq!(trigger.path, "/hooks/media");
        assert!(trigger.secret.is_some());
        assert_eq!(trigger.required_fields.len(), 1);
        assert_eq!(trigger.description, "Video ingest webhook");
    }

    // -----------------------------------------------------------------------
    // WebhookRouter
    // -----------------------------------------------------------------------

    #[test]
    fn test_webhook_router_basic_route() {
        let mut router = WebhookRouter::new();
        let trigger = WebhookTrigger::new("/hooks/ingest").with_required_field("event", "ready");
        router.register("wf-ingest", trigger);

        let body = serde_json::json!({"event": "ready"});
        let body_bytes = serde_json::to_vec(&body).expect("serialize");
        let triggered = router.route("/hooks/ingest", &body_bytes, &body, None);
        assert_eq!(triggered.len(), 1);
        assert_eq!(triggered[0], "wf-ingest");
    }

    #[test]
    fn test_webhook_router_path_no_match() {
        let mut router = WebhookRouter::new();
        router.register("wf-1", WebhookTrigger::new("/hooks/ingest"));

        let body = serde_json::json!({});
        let body_bytes = serde_json::to_vec(&body).expect("serialize");
        let triggered = router.route("/hooks/other", &body_bytes, &body, None);
        assert!(triggered.is_empty());
    }

    #[test]
    fn test_webhook_router_multiple_workflows() {
        let mut router = WebhookRouter::new();
        router.register("wf-a", WebhookTrigger::new("/hooks/event"));
        router.register("wf-b", WebhookTrigger::new("/hooks/event"));

        let body = serde_json::json!({});
        let body_bytes = serde_json::to_vec(&body).expect("serialize");
        let mut triggered = router.route("/hooks/event", &body_bytes, &body, None);
        triggered.sort();
        assert_eq!(triggered.len(), 2);
    }

    #[test]
    fn test_webhook_router_remove() {
        let mut router = WebhookRouter::new();
        router.register("wf-1", WebhookTrigger::new("/hooks/test"));
        router.remove("wf-1");

        let body = serde_json::json!({});
        let body_bytes = serde_json::to_vec(&body).expect("serialize");
        let triggered = router.route("/hooks/test", &body_bytes, &body, None);
        assert!(triggered.is_empty());
    }

    #[test]
    fn test_webhook_router_trigger_count() {
        let mut router = WebhookRouter::new();
        router.register("wf-1", WebhookTrigger::new("/hooks/a"));
        router.register("wf-1", WebhookTrigger::new("/hooks/b"));
        router.register("wf-2", WebhookTrigger::new("/hooks/c"));
        assert_eq!(router.trigger_count(), 3);
    }

    #[test]
    fn test_sha256_known_value() {
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
    fn test_hmac_sha256_non_empty() {
        // Just verify it produces a 64-char hex string
        let mac = hmac_sha256(b"key", b"message");
        assert_eq!(mac.len(), 64);
        assert!(mac.chars().all(|c| c.is_ascii_hexdigit()));
    }
}

// =============================================================================
// WebhookPayload — inbound HTTP payload received by WebhookTriggerServer
// =============================================================================

/// An inbound HTTP POST payload received by the webhook trigger server.
#[derive(Debug, Clone)]
pub struct WebhookPayload {
    /// HTTP headers as lowercase-key map.
    pub headers: HashMap<String, String>,
    /// Raw request body bytes.
    pub body: Vec<u8>,
    /// Unix timestamp (milliseconds) at which the request was received.
    pub timestamp: u64,
    /// URL path of the request (e.g. `"/webhooks/ingest-ready"`).
    pub path: String,
}

impl WebhookPayload {
    /// Create a new `WebhookPayload`.
    #[must_use]
    pub fn new(
        path: impl Into<String>,
        headers: HashMap<String, String>,
        body: Vec<u8>,
        timestamp: u64,
    ) -> Self {
        Self {
            headers,
            body,
            timestamp,
            path: path.into(),
        }
    }

    /// Look up a header value using a **case-insensitive** key.
    ///
    /// Returns `None` if the header is absent.
    #[must_use]
    pub fn header(&self, key: &str) -> Option<&str> {
        let key_lower = key.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| k.to_ascii_lowercase() == key_lower)
            .map(|(_, v)| v.as_str())
    }

    /// Return the body as a UTF-8 `&str`, or `None` if the body is not valid UTF-8.
    #[must_use]
    pub fn body_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.body).ok()
    }

    /// Parse the body as JSON.
    ///
    /// # Errors
    ///
    /// Returns a `serde_json::Error` if the body is not valid JSON.
    pub fn body_json(&self) -> Result<serde_json::Value, serde_json::Error>
    where
        Self: Sized,
    {
        serde_json::from_slice(&self.body)
    }
}

// =============================================================================
// WebhookTriggerServer — pure-Rust HTTP/1.1 server for inbound webhooks
// =============================================================================

/// A minimal pure-Rust HTTP/1.1 server that listens for inbound HTTP POST
/// requests and dispatches them to registered [`WebhookTrigger`] handlers via
/// a [`WebhookRouter`].
///
/// Uses [`std::net::TcpListener`] — no async runtime is required.
/// Each accepted connection is handled synchronously (blocking I/O).
///
/// # Security note
///
/// This is designed for low-volume internal webhook reception.  Production
/// deployments should sit behind a TLS-terminating reverse proxy.
#[cfg(not(target_arch = "wasm32"))]
pub struct WebhookTriggerServer {
    /// Address to bind to (e.g. `"127.0.0.1:0"` for an ephemeral port).
    bind_addr: String,
    /// Shared, thread-safe router.
    router: std::sync::Arc<std::sync::RwLock<WebhookRouter>>,
    /// Atomic flag used to signal the accept loop to stop.
    running: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(not(target_arch = "wasm32"))]
impl WebhookTriggerServer {
    /// Create a new server bound to `bind_addr`.
    ///
    /// The server does not start listening until [`start`] is called.
    ///
    /// [`start`]: WebhookTriggerServer::start
    #[must_use]
    pub fn new(bind_addr: impl Into<String>) -> Self {
        Self {
            bind_addr: bind_addr.into(),
            router: std::sync::Arc::new(std::sync::RwLock::new(WebhookRouter::new())),
            running: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Register a webhook trigger for a workflow.
    pub fn register_trigger(&self, workflow_id: impl Into<String>, trigger: WebhookTrigger) {
        if let Ok(mut r) = self.router.write() {
            r.register(workflow_id, trigger);
        }
    }

    /// Return `true` if the accept loop is currently active.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Start the server, binding to the configured address.
    ///
    /// Spawns a background thread that accepts connections.  The
    /// [`std::thread::JoinHandle`] is returned so the caller can optionally
    /// join the thread.
    ///
    /// # Errors
    ///
    /// Returns `std::io::Error` if the TCP socket cannot be bound.
    pub fn start(&self) -> Result<std::thread::JoinHandle<()>, std::io::Error> {
        use std::net::TcpListener;
        use std::sync::atomic::Ordering;

        let listener = TcpListener::bind(&self.bind_addr)?;
        // Set a short read timeout so the accept loop can check the `running`
        // flag periodically even when no connections arrive.
        listener.set_nonblocking(true)?;

        let router = std::sync::Arc::clone(&self.router);
        let running = std::sync::Arc::clone(&self.running);
        running.store(true, Ordering::SeqCst);

        let handle = std::thread::spawn(move || {
            while running.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _addr)) => {
                        let router_ref = std::sync::Arc::clone(&router);
                        // Handle each connection synchronously in-place.
                        // For a high-throughput server, spawn a thread per
                        // connection; here we keep it simple.
                        Self::handle_connection(stream, router_ref);
                    }
                    Err(ref e)
                        if e.kind() == std::io::ErrorKind::WouldBlock
                            || e.kind() == std::io::ErrorKind::TimedOut =>
                    {
                        // Non-blocking: no connection ready, spin.
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    Err(_) => {
                        // Other errors (e.g. socket closed) → exit loop.
                        break;
                    }
                }
            }
            running.store(false, Ordering::SeqCst);
        });

        Ok(handle)
    }

    /// Stop the accept loop.  The background thread will exit on its next
    /// iteration.
    pub fn stop(&self) {
        self.running
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Handle a single HTTP/1.1 connection: parse the request, route it, and
    /// send a response.
    ///
    /// Returns the list of workflow IDs that were triggered (for testing).
    pub fn handle_connection(
        mut stream: std::net::TcpStream,
        router: std::sync::Arc<std::sync::RwLock<WebhookRouter>>,
    ) -> Vec<String> {
        use std::io::{BufRead, BufReader, Read, Write};

        // Accepted sockets inherit the non-blocking flag from the listener.
        // Switch to blocking so that BufReader reads complete without WouldBlock.
        let _ = stream.set_nonblocking(false);

        // 1. Read the request line + headers.
        let mut reader = BufReader::new(&mut stream);

        let mut request_line = String::new();
        if reader.read_line(&mut request_line).is_err() {
            let _ = stream.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n");
            return Vec::new();
        }

        // Parse METHOD PATH HTTP/1.1
        let parts: Vec<&str> = request_line.trim().splitn(3, ' ').collect();
        if parts.len() < 2 {
            let _ = stream.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n");
            return Vec::new();
        }
        let path = parts[1].to_string();

        // 2. Read headers until blank line.
        let mut headers: HashMap<String, String> = HashMap::new();
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let trimmed = line.trim_end_matches(['\r', '\n']);
                    if trimmed.is_empty() {
                        break; // end of headers
                    }
                    if let Some(colon_pos) = trimmed.find(':') {
                        let key = trimmed[..colon_pos].trim().to_ascii_lowercase();
                        let value = trimmed[colon_pos + 1..].trim().to_string();
                        headers.insert(key, value);
                    }
                }
            }
        }

        // 3. Read body (Content-Length bytes).
        let content_length: usize = headers
            .get("content-length")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        let mut body = vec![0u8; content_length];
        if content_length > 0 {
            if reader.read_exact(&mut body).is_err() {
                let _ = stream.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n");
                return Vec::new();
            }
        }

        // 4. Extract optional HMAC signature.
        let signature = headers.get("x-hub-signature-256").map(String::as_str);

        // 5. Parse body as JSON; fall back to empty object on parse failure.
        let body_json: serde_json::Value =
            serde_json::from_slice(&body).unwrap_or_else(|_| serde_json::json!({}));

        // 6. Route to matching workflows.
        let triggered = router
            .read()
            .map(|r| r.route(&path, &body, &body_json, signature))
            .unwrap_or_default();

        // 7. Send response.
        let status = if triggered.is_empty() {
            "404 Not Found"
        } else {
            "200 OK"
        };
        let resp = format!("HTTP/1.1 {status}\r\nContent-Length: 0\r\n\r\n");
        let _ = stream.write_all(resp.as_bytes());

        triggered
    }

    /// Return the actual bound address (useful when binding to port 0).
    ///
    /// # Errors
    ///
    /// Returns `std::io::Error` if the address cannot be determined.
    pub fn local_addr(&self) -> Result<std::net::SocketAddr, std::io::Error> {
        // We can only know the port after binding, so callers should use the
        // handle returned by `start()`.  This helper re-binds momentarily which
        // is not ideal in production, but is sufficient for test introspection.
        std::net::TcpListener::bind(&self.bind_addr).and_then(|l| l.local_addr())
    }
}

// =============================================================================
// WebhookTriggerServer tests
// =============================================================================

#[cfg(test)]
mod server_tests {
    use super::*;

    // ── WebhookPayload ────────────────────────────────────────────────────────

    #[test]
    fn test_webhook_payload_construction() {
        let mut hdrs = HashMap::new();
        hdrs.insert("content-type".to_string(), "application/json".to_string());
        let body = b"{\"event\":\"push\"}".to_vec();
        let payload = WebhookPayload::new("/hooks/push", hdrs, body.clone(), 1_700_000_000_000);

        assert_eq!(payload.path, "/hooks/push");
        assert_eq!(payload.body, body);
        assert_eq!(payload.timestamp, 1_700_000_000_000);
    }

    #[test]
    fn test_webhook_payload_header_case_insensitive() {
        let mut hdrs = HashMap::new();
        hdrs.insert("X-Hub-Signature-256".to_string(), "abc123".to_string());
        let payload = WebhookPayload::new("/", hdrs, Vec::new(), 0);

        assert_eq!(payload.header("x-hub-signature-256"), Some("abc123"));
        assert_eq!(payload.header("X-HUB-SIGNATURE-256"), Some("abc123"));
        assert_eq!(payload.header("X-Hub-Signature-256"), Some("abc123"));
        assert!(payload.header("missing-header").is_none());
    }

    #[test]
    fn test_webhook_payload_body_str_valid_utf8() {
        let body = b"hello world".to_vec();
        let payload = WebhookPayload::new("/", HashMap::new(), body, 0);
        assert_eq!(payload.body_str(), Some("hello world"));
    }

    #[test]
    fn test_webhook_payload_body_str_invalid_utf8() {
        let body = vec![0xFF, 0xFE]; // invalid UTF-8
        let payload = WebhookPayload::new("/", HashMap::new(), body, 0);
        assert!(payload.body_str().is_none());
    }

    #[test]
    fn test_webhook_payload_body_json_valid() {
        let body = b"{\"key\":42}".to_vec();
        let payload = WebhookPayload::new("/", HashMap::new(), body, 0);
        let json = payload.body_json().expect("valid JSON");
        assert_eq!(json["key"], 42);
    }

    #[test]
    fn test_webhook_payload_body_json_invalid() {
        let body = b"not json".to_vec();
        let payload = WebhookPayload::new("/", HashMap::new(), body, 0);
        assert!(payload.body_json().is_err());
    }

    #[test]
    fn test_webhook_payload_empty_body() {
        let payload = WebhookPayload::new("/", HashMap::new(), Vec::new(), 0);
        assert_eq!(payload.body_str(), Some(""));
        assert!(payload.body_json().is_err()); // empty is not valid JSON
    }

    // ── WebhookTriggerServer — construction & state ───────────────────────────

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_server_new_not_running() {
        let server = WebhookTriggerServer::new("127.0.0.1:0");
        assert!(!server.is_running());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_server_register_trigger() {
        let server = WebhookTriggerServer::new("127.0.0.1:0");
        server.register_trigger("wf-1", WebhookTrigger::new("/hooks/test"));
        // Verify the trigger is stored in the router
        let router_guard = server.router.read().expect("read lock");
        assert_eq!(router_guard.trigger_count(), 1);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_server_register_multiple_triggers() {
        let server = WebhookTriggerServer::new("127.0.0.1:0");
        server.register_trigger("wf-1", WebhookTrigger::new("/hooks/a"));
        server.register_trigger("wf-1", WebhookTrigger::new("/hooks/b"));
        server.register_trigger("wf-2", WebhookTrigger::new("/hooks/c"));
        let router_guard = server.router.read().expect("read lock");
        assert_eq!(router_guard.trigger_count(), 3);
    }

    // ── Integration: start server, send POST, verify triggered ───────────────

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_server_start_stop() {
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("local addr").port();
        drop(listener);

        let server = WebhookTriggerServer::new(format!("127.0.0.1:{port}"));
        server.register_trigger("wf-webhook", WebhookTrigger::new("/hooks/deploy"));

        let handle = server.start().expect("start server");
        assert!(server.is_running());

        server.stop();
        // Give the thread a moment to exit.
        std::thread::sleep(std::time::Duration::from_millis(50));
        let _ = handle.join();
        assert!(!server.is_running());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_server_receives_post_triggers_workflow() {
        use std::io::Write;
        use std::net::{TcpListener, TcpStream};

        // Bind to a free port first to learn the port number.
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("local addr").port();
        drop(listener);

        let server = WebhookTriggerServer::new(format!("127.0.0.1:{port}"));
        server.register_trigger("my-workflow", WebhookTrigger::new("/hooks/push"));

        let _handle = server.start().expect("start server");
        // Brief sleep to let the accept loop spin up.
        std::thread::sleep(std::time::Duration::from_millis(30));

        // Send a minimal HTTP POST.
        let body = b"{}";
        let request = format!(
            "POST /hooks/push HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );

        let mut conn = TcpStream::connect(format!("127.0.0.1:{port}")).expect("connect");
        conn.write_all(request.as_bytes()).expect("write request");
        conn.write_all(body).expect("write body");

        // Read the response.
        let mut resp_buf = [0u8; 256];
        let n = {
            use std::io::Read;
            conn.read(&mut resp_buf).unwrap_or(0)
        };
        let resp_str = std::str::from_utf8(&resp_buf[..n]).unwrap_or("");
        assert!(
            resp_str.starts_with("HTTP/1.1 200"),
            "expected 200, got: {resp_str}"
        );

        server.stop();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_server_wrong_path_returns_404() {
        use std::io::Write;
        use std::net::{TcpListener, TcpStream};

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("local addr").port();
        drop(listener);

        let server = WebhookTriggerServer::new(format!("127.0.0.1:{port}"));
        server.register_trigger("wf-x", WebhookTrigger::new("/hooks/correct"));

        let _handle = server.start().expect("start");
        std::thread::sleep(std::time::Duration::from_millis(30));

        let body = b"{}";
        let request = format!(
            "POST /hooks/wrong HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let mut conn = TcpStream::connect(format!("127.0.0.1:{port}")).expect("connect");
        conn.write_all(request.as_bytes()).expect("write");
        conn.write_all(body).expect("write body");

        let mut buf = [0u8; 256];
        let n = {
            use std::io::Read;
            conn.read(&mut buf).unwrap_or(0)
        };
        let resp = std::str::from_utf8(&buf[..n]).unwrap_or("");
        assert!(
            resp.starts_with("HTTP/1.1 404"),
            "expected 404, got: {resp}"
        );

        server.stop();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_server_signature_verification_accepted() {
        use std::io::Write;
        use std::net::{TcpListener, TcpStream};

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("local addr").port();
        drop(listener);

        let secret = "s3cr3t";
        let server = WebhookTriggerServer::new(format!("127.0.0.1:{port}"));
        server.register_trigger(
            "secure-wf",
            WebhookTrigger::new("/hooks/secure").with_secret(secret),
        );

        let _handle = server.start().expect("start");
        std::thread::sleep(std::time::Duration::from_millis(30));

        let body = b"{}";
        // Compute correct HMAC-SHA256 signature via the existing pure-Rust implementation.
        // We replicate the public interface via WebhookTrigger::verify_signature logic.
        let sig = {
            // Use a temporary trigger to call compute_signature path.
            let _dummy_trigger = WebhookTrigger::new("/").with_secret(secret);
            // We manually compute HMAC-SHA256 using the private fn by a round-trip:
            // verify_signature returns true when sig matches → we need the correct value.
            // Since we can't call private fn directly, use the outbound WebhookNotifier.
            use crate::webhook::{WebhookConfig, WebhookNotifier};
            let notifier = WebhookNotifier::new(WebhookConfig {
                url: String::new(),
                secret: Some(secret.to_string()),
                events: Vec::new(),
                max_retries: 0,
                timeout_ms: 0,
            });
            notifier
                .compute_signature(std::str::from_utf8(body).expect("utf8"))
                .expect("sig")
        };

        let request = format!(
            "POST /hooks/secure HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nX-Hub-Signature-256: {}\r\n\r\n",
            body.len(),
            sig
        );
        let mut conn = TcpStream::connect(format!("127.0.0.1:{port}")).expect("connect");
        conn.write_all(request.as_bytes()).expect("write");
        conn.write_all(body).expect("write body");

        let mut buf = [0u8; 256];
        let n = {
            use std::io::Read;
            conn.read(&mut buf).unwrap_or(0)
        };
        let resp = std::str::from_utf8(&buf[..n]).unwrap_or("");
        assert!(
            resp.starts_with("HTTP/1.1 200"),
            "expected 200 with valid sig, got: {resp}"
        );

        server.stop();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_server_bad_signature_returns_404() {
        use std::io::Write;
        use std::net::{TcpListener, TcpStream};

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("local addr").port();
        drop(listener);

        let server = WebhookTriggerServer::new(format!("127.0.0.1:{port}"));
        server.register_trigger(
            "secure-wf",
            WebhookTrigger::new("/hooks/secure").with_secret("correct-secret"),
        );

        let _handle = server.start().expect("start");
        std::thread::sleep(std::time::Duration::from_millis(30));

        let body = b"{}";
        let bad_sig = "a".repeat(64); // wrong 64-char hex string
        let request = format!(
            "POST /hooks/secure HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nX-Hub-Signature-256: {bad_sig}\r\n\r\n",
            body.len()
        );
        let mut conn = TcpStream::connect(format!("127.0.0.1:{port}")).expect("connect");
        conn.write_all(request.as_bytes()).expect("write");
        conn.write_all(body).expect("write body");

        let mut buf = [0u8; 256];
        let n = {
            use std::io::Read;
            conn.read(&mut buf).unwrap_or(0)
        };
        let resp = std::str::from_utf8(&buf[..n]).unwrap_or("");
        assert!(
            resp.starts_with("HTTP/1.1 404"),
            "bad sig should get 404, got: {resp}"
        );

        server.stop();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_server_body_field_matching() {
        use std::io::Write;
        use std::net::{TcpListener, TcpStream};

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("local addr").port();
        drop(listener);

        let server = WebhookTriggerServer::new(format!("127.0.0.1:{port}"));
        server.register_trigger(
            "wf-prod",
            WebhookTrigger::new("/hooks/deploy").with_required_field("environment", "production"),
        );

        let _handle = server.start().expect("start");
        std::thread::sleep(std::time::Duration::from_millis(30));

        // Matching body — send headers + body as a single write to avoid
        // a race where the server closes the connection before the second write.
        let body = br#"{"environment":"production"}"#;
        let mut full_request = format!(
            "POST /hooks/deploy HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        ).into_bytes();
        full_request.extend_from_slice(body);

        let mut conn = TcpStream::connect(format!("127.0.0.1:{port}")).expect("connect");
        conn.write_all(&full_request).expect("write full request");
        // Shut down the write side so the server knows we are done sending.
        conn.shutdown(std::net::Shutdown::Write).ok();

        let mut buf = [0u8; 256];
        let n = {
            use std::io::Read;
            conn.read(&mut buf).unwrap_or(0)
        };
        let resp = std::str::from_utf8(&buf[..n]).unwrap_or("");
        assert!(
            resp.starts_with("HTTP/1.1 200"),
            "matching body should trigger: {resp}"
        );

        server.stop();
    }
}
