//! Extended WHIP/WHEP functionality: OAuth2 token validation, session metrics, event hooks.
//!
//! This module provides production-grade extensions to the base WHIP/WHEP implementation:
//!
//! - [`BearerTokenValidator`] — HMAC-SHA256 signed JWT-style bearer token validation
//! - [`TokenClaims`] — Decoded claims from a bearer token (subject, expiry, scopes)
//! - [`SessionMetrics`] — Per-session statistics (bytes, packets, RTT, jitter)
//! - [`MetricsRegistry`] — Aggregated metrics across all WHIP/WHEP sessions
//! - [`EventHook`] — Trait for receiving session lifecycle events
//! - [`EventDispatcher`] — Fan-out event delivery to multiple registered hooks
//! - [`SessionEvent`] — Enum of lifecycle events (created, activated, terminated, error)
//! - [`AuditLog`] — Append-only audit record of session lifecycle events
//! - [`WhipWhepExtendedEndpoint`] — Full-featured endpoint combining all extensions

use crate::error::{NetError, NetResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ─── Token Validation ────────────────────────────────────────────────────────

/// Scope permissions encoded in a bearer token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenScope {
    /// May publish (WHIP ingest).
    Publish,
    /// May subscribe (WHEP playback).
    Subscribe,
    /// Full admin access.
    Admin,
    /// Custom application-defined scope.
    Custom(String),
}

impl TokenScope {
    /// Parses a scope string into a `TokenScope`.
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        match s {
            "publish" => Self::Publish,
            "subscribe" => Self::Subscribe,
            "admin" => Self::Admin,
            other => Self::Custom(other.to_owned()),
        }
    }

    /// Returns the string representation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Publish => "publish",
            Self::Subscribe => "subscribe",
            Self::Admin => "admin",
            Self::Custom(s) => s.as_str(),
        }
    }
}

/// Claims decoded from a bearer token.
#[derive(Debug, Clone)]
pub struct TokenClaims {
    /// Subject (stream key or user ID).
    pub subject: String,
    /// Issuer identifier.
    pub issuer: String,
    /// Expiry as Unix timestamp (seconds).
    pub expires_at: u64,
    /// Issued-at as Unix timestamp (seconds).
    pub issued_at: u64,
    /// Granted scopes.
    pub scopes: Vec<TokenScope>,
    /// JWT ID (unique per token).
    pub jwt_id: String,
    /// Custom metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

impl TokenClaims {
    /// Returns true if the token has not expired.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.expires_at > now
    }

    /// Returns true if the token has the given scope.
    #[must_use]
    pub fn has_scope(&self, scope: &TokenScope) -> bool {
        self.scopes.contains(scope) || self.scopes.contains(&TokenScope::Admin)
    }

    /// Returns the remaining validity duration.
    #[must_use]
    pub fn remaining_ttl(&self) -> Option<Duration> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if self.expires_at > now {
            Some(Duration::from_secs(self.expires_at - now))
        } else {
            None
        }
    }
}

/// HMAC-SHA256-based bearer token validator.
///
/// Tokens are encoded as `<header_b64>.<payload_b64>.<sig_hex>` where:
/// - `header_b64` = base64url of `{"alg":"HS256","typ":"JWT"}`
/// - `payload_b64` = base64url of JSON claims
/// - `sig_hex` = hex of HMAC-SHA256(secret, `<header_b64>.<payload_b64>`)
///
/// This is a simplified but production-sufficient validator that avoids
/// external JWT crate dependencies while remaining compatible with token
/// generators that follow the same scheme.
#[derive(Debug, Clone)]
pub struct BearerTokenValidator {
    /// HMAC signing secret (raw bytes).
    secret: Vec<u8>,
    /// Allowed issuers (empty = accept any).
    allowed_issuers: Vec<String>,
    /// Clock skew tolerance in seconds.
    clock_skew_secs: u64,
}

impl BearerTokenValidator {
    /// Creates a new validator with the given HMAC secret.
    #[must_use]
    pub fn new(secret: impl AsRef<[u8]>) -> Self {
        Self {
            secret: secret.as_ref().to_vec(),
            allowed_issuers: Vec::new(),
            clock_skew_secs: 30,
        }
    }

    /// Adds an allowed issuer.  If no issuers are added, all issuers are accepted.
    #[must_use]
    pub fn with_issuer(mut self, issuer: impl Into<String>) -> Self {
        self.allowed_issuers.push(issuer.into());
        self
    }

    /// Sets clock skew tolerance (default 30 s).
    #[must_use]
    pub fn with_clock_skew(mut self, secs: u64) -> Self {
        self.clock_skew_secs = secs;
        self
    }

    /// Validates a `Bearer <token>` Authorization header value.
    ///
    /// Strips the `Bearer ` prefix, then validates the token.
    pub fn validate_header(&self, auth_header: &str) -> NetResult<TokenClaims> {
        let token = auth_header.strip_prefix("Bearer ").ok_or_else(|| {
            NetError::authentication("Authorization header must start with 'Bearer '")
        })?;
        self.validate_token(token)
    }

    /// Validates a raw token string.
    pub fn validate_token(&self, token: &str) -> NetResult<TokenClaims> {
        let parts: Vec<&str> = token.splitn(3, '.').collect();
        if parts.len() != 3 {
            return Err(NetError::authentication("Token must have exactly 3 parts"));
        }

        let header_b64 = parts[0];
        let payload_b64 = parts[1];
        let provided_sig = parts[2];

        // Verify signature
        let message = format!("{header_b64}.{payload_b64}");
        let expected_sig = hmac_sha256_hex(&self.secret, message.as_bytes());
        if !constant_time_eq(provided_sig, &expected_sig) {
            return Err(NetError::authentication("Token signature invalid"));
        }

        // Decode payload
        let payload_bytes = base64url_decode(payload_b64)
            .map_err(|_| NetError::authentication("Token payload not valid base64url"))?;
        let payload_str = std::str::from_utf8(&payload_bytes)
            .map_err(|_| NetError::authentication("Token payload not valid UTF-8"))?;

        let claims = parse_claims(payload_str)?;

        // Check expiry
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if claims.expires_at + self.clock_skew_secs < now {
            return Err(NetError::authentication("Token has expired"));
        }

        // Check issuer
        if !self.allowed_issuers.is_empty() && !self.allowed_issuers.contains(&claims.issuer) {
            return Err(NetError::authentication(format!(
                "Issuer '{}' not in allowed list",
                claims.issuer
            )));
        }

        Ok(claims)
    }

    /// Generates a signed token with the given claims (for testing/server-side issuance).
    #[must_use]
    pub fn generate_token(&self, claims: &TokenClaims) -> String {
        let header = base64url_encode(b"{\"alg\":\"HS256\",\"typ\":\"JWT\"}");
        let payload = build_payload_json(claims);
        let payload_b64 = base64url_encode(payload.as_bytes());
        let message = format!("{header}.{payload_b64}");
        let sig = hmac_sha256_hex(&self.secret, message.as_bytes());
        format!("{message}.{sig}")
    }
}

// ─── Session Metrics ─────────────────────────────────────────────────────────

/// Per-session statistics for a WHIP or WHEP connection.
#[derive(Debug, Clone, Default)]
pub struct SessionMetrics {
    /// Total bytes received from the peer.
    pub bytes_received: u64,
    /// Total bytes sent to the peer.
    pub bytes_sent: u64,
    /// Total RTP packets received.
    pub packets_received: u64,
    /// Total RTP packets sent.
    pub packets_sent: u64,
    /// Packets reported lost (from RTCP RR).
    pub packets_lost: u64,
    /// Cumulative packet loss fraction (0.0–1.0).
    pub loss_fraction: f64,
    /// Round-trip time in milliseconds (from RTCP XR or SR/RR).
    pub rtt_ms: f64,
    /// Jitter in milliseconds.
    pub jitter_ms: f64,
    /// Audio codec in use (e.g. "opus").
    pub audio_codec: Option<String>,
    /// Video codec in use (e.g. "av1", "vp9", "h264").
    pub video_codec: Option<String>,
    /// Current receive bitrate in bits/s.
    pub receive_bitrate_bps: u64,
    /// Current send bitrate in bits/s.
    pub send_bitrate_bps: u64,
    /// Last time metrics were updated.
    pub last_updated: Option<Instant>,
}

impl SessionMetrics {
    /// Creates a new zero-initialised metrics object.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates byte/packet counters and recomputes derived metrics.
    pub fn update(
        &mut self,
        bytes_recv: u64,
        bytes_sent: u64,
        pkts_recv: u64,
        pkts_sent: u64,
        pkts_lost: u64,
        rtt_ms: f64,
        jitter_ms: f64,
    ) {
        let elapsed_secs = self
            .last_updated
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(1.0)
            .max(0.001);

        let delta_recv = bytes_recv.saturating_sub(self.bytes_received);
        let delta_sent = bytes_sent.saturating_sub(self.bytes_sent);
        self.receive_bitrate_bps = ((delta_recv as f64 * 8.0) / elapsed_secs) as u64;
        self.send_bitrate_bps = ((delta_sent as f64 * 8.0) / elapsed_secs) as u64;

        self.bytes_received = bytes_recv;
        self.bytes_sent = bytes_sent;
        self.packets_received = pkts_recv;
        self.packets_sent = pkts_sent;
        self.packets_lost = pkts_lost;
        self.rtt_ms = rtt_ms;
        self.jitter_ms = jitter_ms;

        let total = pkts_recv + pkts_lost;
        self.loss_fraction = if total > 0 {
            pkts_lost as f64 / total as f64
        } else {
            0.0
        };

        self.last_updated = Some(Instant::now());
    }

    /// Returns a JSON-compatible map representation.
    #[must_use]
    pub fn to_map(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("bytes_received".to_owned(), self.bytes_received.to_string());
        m.insert("bytes_sent".to_owned(), self.bytes_sent.to_string());
        m.insert(
            "packets_received".to_owned(),
            self.packets_received.to_string(),
        );
        m.insert("packets_sent".to_owned(), self.packets_sent.to_string());
        m.insert("packets_lost".to_owned(), self.packets_lost.to_string());
        m.insert(
            "loss_fraction".to_owned(),
            format!("{:.4}", self.loss_fraction),
        );
        m.insert("rtt_ms".to_owned(), format!("{:.2}", self.rtt_ms));
        m.insert("jitter_ms".to_owned(), format!("{:.2}", self.jitter_ms));
        m.insert(
            "receive_bitrate_bps".to_owned(),
            self.receive_bitrate_bps.to_string(),
        );
        m.insert(
            "send_bitrate_bps".to_owned(),
            self.send_bitrate_bps.to_string(),
        );
        if let Some(ref codec) = self.audio_codec {
            m.insert("audio_codec".to_owned(), codec.clone());
        }
        if let Some(ref codec) = self.video_codec {
            m.insert("video_codec".to_owned(), codec.clone());
        }
        m
    }
}

/// Aggregated metrics registry across all sessions.
#[derive(Debug, Default)]
pub struct MetricsRegistry {
    /// Per-session metrics keyed by session ID.
    sessions: HashMap<String, SessionMetrics>,
}

impl MetricsRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or updates metrics for a session.
    pub fn upsert(&mut self, session_id: impl Into<String>, metrics: SessionMetrics) {
        self.sessions.insert(session_id.into(), metrics);
    }

    /// Removes metrics for a terminated session.
    pub fn remove(&mut self, session_id: &str) {
        self.sessions.remove(session_id);
    }

    /// Returns metrics for a single session.
    #[must_use]
    pub fn get(&self, session_id: &str) -> Option<&SessionMetrics> {
        self.sessions.get(session_id)
    }

    /// Returns aggregate totals across all sessions.
    #[must_use]
    pub fn aggregate(&self) -> SessionMetrics {
        let mut total = SessionMetrics::new();
        for m in self.sessions.values() {
            total.bytes_received += m.bytes_received;
            total.bytes_sent += m.bytes_sent;
            total.packets_received += m.packets_received;
            total.packets_sent += m.packets_sent;
            total.packets_lost += m.packets_lost;
            total.receive_bitrate_bps += m.receive_bitrate_bps;
            total.send_bitrate_bps += m.send_bitrate_bps;
        }
        let n = self.sessions.len();
        if n > 0 {
            let sum_rtt: f64 = self.sessions.values().map(|m| m.rtt_ms).sum();
            let sum_jitter: f64 = self.sessions.values().map(|m| m.jitter_ms).sum();
            total.rtt_ms = sum_rtt / n as f64;
            total.jitter_ms = sum_jitter / n as f64;
            let total_pkts = total.packets_received + total.packets_lost;
            total.loss_fraction = if total_pkts > 0 {
                total.packets_lost as f64 / total_pkts as f64
            } else {
                0.0
            };
        }
        total
    }

    /// Returns the number of tracked sessions.
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Returns a Prometheus-compatible text exposition of current metrics.
    #[must_use]
    pub fn prometheus_exposition(&self) -> String {
        let agg = self.aggregate();
        let mut out = String::new();
        out.push_str("# HELP whip_whep_sessions_total Number of active sessions\n");
        out.push_str("# TYPE whip_whep_sessions_total gauge\n");
        out.push_str(&format!(
            "whip_whep_sessions_total {}\n",
            self.sessions.len()
        ));
        out.push_str("# HELP whip_whep_bytes_received_total Bytes received across all sessions\n");
        out.push_str("# TYPE whip_whep_bytes_received_total counter\n");
        out.push_str(&format!(
            "whip_whep_bytes_received_total {}\n",
            agg.bytes_received
        ));
        out.push_str("# HELP whip_whep_bytes_sent_total Bytes sent across all sessions\n");
        out.push_str("# TYPE whip_whep_bytes_sent_total counter\n");
        out.push_str(&format!("whip_whep_bytes_sent_total {}\n", agg.bytes_sent));
        out.push_str("# HELP whip_whep_rtt_ms_avg Average RTT across sessions (ms)\n");
        out.push_str("# TYPE whip_whep_rtt_ms_avg gauge\n");
        out.push_str(&format!("whip_whep_rtt_ms_avg {:.2}\n", agg.rtt_ms));
        out.push_str("# HELP whip_whep_loss_fraction_avg Average packet loss fraction\n");
        out.push_str("# TYPE whip_whep_loss_fraction_avg gauge\n");
        out.push_str(&format!(
            "whip_whep_loss_fraction_avg {:.4}\n",
            agg.loss_fraction
        ));
        out
    }
}

// ─── Event System ─────────────────────────────────────────────────────────────

/// Lifecycle event emitted for WHIP/WHEP sessions.
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// A new session was created (before SDP exchange).
    Created {
        session_id: String,
        protocol: SessionProtocol,
        remote_addr: Option<String>,
    },
    /// SDP negotiation completed; session is now active.
    Activated {
        session_id: String,
        protocol: SessionProtocol,
        stream_key: Option<String>,
    },
    /// A trickle ICE candidate was received.
    CandidateReceived {
        session_id: String,
        candidate: String,
    },
    /// Session was cleanly terminated (DELETE received).
    Terminated {
        session_id: String,
        protocol: SessionProtocol,
        duration: Duration,
        final_metrics: Option<Box<SessionMetrics>>,
    },
    /// Authentication failure.
    AuthFailed {
        remote_addr: Option<String>,
        reason: String,
    },
    /// Protocol or media error occurred.
    Error {
        session_id: Option<String>,
        code: String,
        message: String,
    },
}

/// Identifies whether a session is WHIP (ingest) or WHEP (playback).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionProtocol {
    /// WebRTC HTTP Ingestion Protocol.
    Whip,
    /// WebRTC HTTP Egress Protocol.
    Whep,
}

impl SessionProtocol {
    /// Returns the protocol name string.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Whip => "WHIP",
            Self::Whep => "WHEP",
        }
    }
}

/// Trait implemented by event hook subscribers.
pub trait EventHook: Send + Sync {
    /// Called when a session event occurs.
    fn on_event(&self, event: &SessionEvent);
}

/// Fan-out event dispatcher that delivers events to multiple registered hooks.
#[derive(Default)]
pub struct EventDispatcher {
    hooks: Vec<Arc<dyn EventHook>>,
}

impl EventDispatcher {
    /// Creates an empty dispatcher.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new event hook.
    pub fn register(&mut self, hook: Arc<dyn EventHook>) {
        self.hooks.push(hook);
    }

    /// Dispatches an event to all registered hooks.
    pub fn emit(&self, event: &SessionEvent) {
        for hook in &self.hooks {
            hook.on_event(event);
        }
    }

    /// Returns the number of registered hooks.
    #[must_use]
    pub fn hook_count(&self) -> usize {
        self.hooks.len()
    }
}

impl std::fmt::Debug for EventDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventDispatcher")
            .field("hook_count", &self.hooks.len())
            .finish()
    }
}

// ─── Audit Log ────────────────────────────────────────────────────────────────

/// A single record in the audit log.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// When this entry was recorded.
    pub timestamp: Instant,
    /// Session ID (if applicable).
    pub session_id: Option<String>,
    /// Category of event.
    pub category: AuditCategory,
    /// Human-readable description.
    pub description: String,
}

/// Category of audit log entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditCategory {
    /// Session lifecycle (create, activate, terminate).
    Session,
    /// Authentication attempt.
    Auth,
    /// Media or protocol error.
    Error,
    /// Administrative action.
    Admin,
}

impl AuditCategory {
    /// Returns the category name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Session => "session",
            Self::Auth => "auth",
            Self::Error => "error",
            Self::Admin => "admin",
        }
    }
}

/// Append-only audit log.
///
/// Records a bounded history of session lifecycle events for compliance
/// and debugging purposes.  Oldest entries are dropped when capacity
/// is exceeded.
#[derive(Debug)]
pub struct AuditLog {
    entries: Vec<AuditEntry>,
    max_entries: usize,
}

impl AuditLog {
    /// Creates a new audit log with the given maximum entry count.
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries: max_entries.max(1),
        }
    }

    /// Appends an entry, evicting the oldest if at capacity.
    pub fn append(
        &mut self,
        session_id: Option<String>,
        category: AuditCategory,
        description: impl Into<String>,
    ) {
        if self.entries.len() >= self.max_entries {
            self.entries.remove(0);
        }
        self.entries.push(AuditEntry {
            timestamp: Instant::now(),
            session_id,
            category,
            description: description.into(),
        });
    }

    /// Returns all entries matching a given category.
    #[must_use]
    pub fn entries_by_category(&self, category: AuditCategory) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.category == category)
            .collect()
    }

    /// Returns all entries for a specific session.
    #[must_use]
    pub fn entries_for_session(&self, session_id: &str) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.session_id.as_deref() == Some(session_id))
            .collect()
    }

    /// Returns the total number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the log is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the most recent `n` entries.
    #[must_use]
    pub fn tail(&self, n: usize) -> &[AuditEntry] {
        let start = self.entries.len().saturating_sub(n);
        &self.entries[start..]
    }
}

/// Newtype wrapper that implements `EventHook` and forwards events to the inner `AuditLog`.
pub struct AuditLogHook(pub Arc<Mutex<AuditLog>>);

impl EventHook for AuditLogHook {
    fn on_event(&self, event: &SessionEvent) {
        let Ok(mut log) = self.0.lock() else { return };
        match event {
            SessionEvent::Created {
                session_id,
                protocol,
                ..
            } => {
                log.append(
                    Some(session_id.clone()),
                    AuditCategory::Session,
                    format!("{} session created", protocol.name()),
                );
            }
            SessionEvent::Activated {
                session_id,
                protocol,
                ..
            } => {
                log.append(
                    Some(session_id.clone()),
                    AuditCategory::Session,
                    format!("{} session activated", protocol.name()),
                );
            }
            SessionEvent::Terminated {
                session_id,
                duration,
                ..
            } => {
                log.append(
                    Some(session_id.clone()),
                    AuditCategory::Session,
                    format!("Session terminated after {:.1}s", duration.as_secs_f64()),
                );
            }
            SessionEvent::AuthFailed { reason, .. } => {
                log.append(None, AuditCategory::Auth, format!("Auth failed: {reason}"));
            }
            SessionEvent::Error {
                session_id,
                message,
                ..
            } => {
                log.append(
                    session_id.clone(),
                    AuditCategory::Error,
                    format!("Error: {message}"),
                );
            }
            SessionEvent::CandidateReceived { session_id, .. } => {
                log.append(
                    Some(session_id.clone()),
                    AuditCategory::Session,
                    "ICE candidate received".to_owned(),
                );
            }
        }
    }
}

// ─── Extended Endpoint ────────────────────────────────────────────────────────

/// Configuration for the extended WHIP/WHEP endpoint.
#[derive(Debug, Clone)]
pub struct ExtendedEndpointConfig {
    /// HMAC secret for bearer token signing/validation.  Empty = no auth.
    pub token_secret: Vec<u8>,
    /// Allowed token issuers (empty = accept any).
    pub allowed_issuers: Vec<String>,
    /// Maximum audit log entries.
    pub audit_log_capacity: usize,
    /// Maximum simultaneous sessions.
    pub max_sessions: usize,
    /// Session idle timeout.
    pub session_timeout: Duration,
    /// Whether Prometheus metrics are enabled.
    pub enable_prometheus: bool,
}

impl Default for ExtendedEndpointConfig {
    fn default() -> Self {
        Self {
            token_secret: Vec::new(),
            allowed_issuers: Vec::new(),
            audit_log_capacity: 10_000,
            max_sessions: 1_000,
            session_timeout: Duration::from_secs(300),
            enable_prometheus: true,
        }
    }
}

/// Session creation result returned to the HTTP layer.
#[derive(Debug, Clone)]
pub struct SessionCreated {
    /// Unique session ID.
    pub session_id: String,
    /// Resource path (e.g. `/whip/resource/<id>`).
    pub resource_path: String,
    /// SDP answer.
    pub sdp_answer: String,
    /// HTTP Link headers carrying ICE server URLs.
    pub ice_link_headers: Vec<String>,
    /// ETag for conditional requests.
    pub etag: String,
}

/// Lightweight internal session record.
#[derive(Debug)]
struct SessionRecord {
    protocol: SessionProtocol,
    stream_key: Option<String>,
    auth_subject: Option<String>,
    created_at: Instant,
    terminated: bool,
}

/// Full-featured WHIP/WHEP endpoint integrating OAuth2 validation,
/// per-session metrics, event hooks, and an audit log.
#[derive(Debug)]
pub struct WhipWhepExtendedEndpoint {
    config: ExtendedEndpointConfig,
    validator: Option<BearerTokenValidator>,
    /// WHIP sessions.
    sessions: HashMap<String, SessionRecord>,
    /// Metrics registry.
    metrics: MetricsRegistry,
    /// Event dispatcher.
    dispatcher: EventDispatcher,
    /// Audit log (also registered as a hook).
    audit_log: Arc<Mutex<AuditLog>>,
    /// Session counter.
    counter: u64,
}

impl WhipWhepExtendedEndpoint {
    /// Creates a new extended endpoint with the given configuration.
    #[must_use]
    pub fn new(config: ExtendedEndpointConfig) -> Self {
        let validator = if config.token_secret.is_empty() {
            None
        } else {
            let mut v = BearerTokenValidator::new(&config.token_secret);
            for issuer in &config.allowed_issuers {
                v = v.with_issuer(issuer.clone());
            }
            Some(v)
        };

        let audit = Arc::new(Mutex::new(AuditLog::new(config.audit_log_capacity)));
        let mut dispatcher = EventDispatcher::new();
        dispatcher.register(Arc::new(AuditLogHook(Arc::clone(&audit))) as Arc<dyn EventHook>);

        Self {
            config,
            validator,
            sessions: HashMap::new(),
            metrics: MetricsRegistry::new(),
            dispatcher,
            audit_log: audit,
            counter: 0,
        }
    }

    /// Registers an additional event hook.
    pub fn add_hook(&mut self, hook: Arc<dyn EventHook>) {
        self.dispatcher.register(hook);
    }

    /// Handles a WHIP POST request.
    ///
    /// `auth_header` should be the `Authorization` header value, or `None` if absent.
    /// `remote_addr` is the client IP string for audit purposes.
    pub fn handle_whip_post(
        &mut self,
        offer_sdp: &str,
        auth_header: Option<&str>,
        remote_addr: Option<&str>,
        ice_link_headers: Vec<String>,
    ) -> NetResult<SessionCreated> {
        // Auth check
        let auth_subject = self.check_auth(auth_header, &TokenScope::Publish, remote_addr)?;

        if self.sessions.len() >= self.config.max_sessions {
            return Err(NetError::connection("Maximum sessions reached"));
        }

        let session_id = self.next_id();
        let etag = format!("W/\"{}\"", djb2_hash(&session_id));
        let resource_path = format!("/whip/resource/{session_id}");

        // Generate SDP answer
        let sdp_answer = generate_sdp_answer_recvonly(offer_sdp);

        let record = SessionRecord {
            protocol: SessionProtocol::Whip,
            stream_key: None,
            auth_subject: auth_subject.clone(),
            created_at: Instant::now(),
            terminated: false,
        };
        self.sessions.insert(session_id.clone(), record);
        self.metrics.upsert(&session_id, SessionMetrics::new());

        self.dispatcher.emit(&SessionEvent::Created {
            session_id: session_id.clone(),
            protocol: SessionProtocol::Whip,
            remote_addr: remote_addr.map(|s| s.to_owned()),
        });

        Ok(SessionCreated {
            session_id,
            resource_path,
            sdp_answer,
            ice_link_headers,
            etag,
        })
    }

    /// Handles a WHEP POST request.
    pub fn handle_whep_post(
        &mut self,
        offer_sdp: &str,
        auth_header: Option<&str>,
        stream_key: Option<&str>,
        remote_addr: Option<&str>,
        ice_link_headers: Vec<String>,
    ) -> NetResult<SessionCreated> {
        let auth_subject = self.check_auth(auth_header, &TokenScope::Subscribe, remote_addr)?;

        if self.sessions.len() >= self.config.max_sessions {
            return Err(NetError::connection("Maximum sessions reached"));
        }

        let session_id = self.next_id();
        let etag = format!("W/\"{}\"", djb2_hash(&session_id));
        let resource_path = format!("/whep/resource/{session_id}");

        let sdp_answer = generate_sdp_answer_sendonly(offer_sdp);

        let record = SessionRecord {
            protocol: SessionProtocol::Whep,
            stream_key: stream_key.map(|s| s.to_owned()),
            auth_subject: auth_subject.clone(),
            created_at: Instant::now(),
            terminated: false,
        };
        self.sessions.insert(session_id.clone(), record);
        self.metrics.upsert(&session_id, SessionMetrics::new());

        self.dispatcher.emit(&SessionEvent::Created {
            session_id: session_id.clone(),
            protocol: SessionProtocol::Whep,
            remote_addr: remote_addr.map(|s| s.to_owned()),
        });

        Ok(SessionCreated {
            session_id,
            resource_path,
            sdp_answer,
            ice_link_headers,
            etag,
        })
    }

    /// Marks a session as activated (call after ICE completion).
    pub fn activate_session(&mut self, session_id: &str) -> NetResult<()> {
        let record = self
            .sessions
            .get(session_id)
            .ok_or_else(|| NetError::not_found(format!("Session not found: {session_id}")))?;

        let event = SessionEvent::Activated {
            session_id: session_id.to_owned(),
            protocol: record.protocol,
            stream_key: record.stream_key.clone(),
        };
        self.dispatcher.emit(&event);
        Ok(())
    }

    /// Handles a PATCH request with trickle ICE candidates.
    pub fn handle_patch(&mut self, session_id: &str, sdp_fragment: &str) -> NetResult<()> {
        if !self.sessions.contains_key(session_id) {
            return Err(NetError::not_found(format!(
                "Session not found: {session_id}"
            )));
        }

        for line in sdp_fragment.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("a=candidate:") {
                self.dispatcher.emit(&SessionEvent::CandidateReceived {
                    session_id: session_id.to_owned(),
                    candidate: trimmed.to_owned(),
                });
            }
        }

        Ok(())
    }

    /// Handles a DELETE request, terminating the session.
    pub fn handle_delete(&mut self, session_id: &str) -> NetResult<()> {
        let record = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| NetError::not_found(format!("Session not found: {session_id}")))?;

        if record.terminated {
            return Err(NetError::invalid_state("Session already terminated"));
        }

        record.terminated = true;
        let duration = record.created_at.elapsed();
        let protocol = record.protocol;
        let final_metrics = self.metrics.get(session_id).cloned();

        self.dispatcher.emit(&SessionEvent::Terminated {
            session_id: session_id.to_owned(),
            protocol,
            duration,
            final_metrics: final_metrics.map(Box::new),
        });

        self.metrics.remove(session_id);
        Ok(())
    }

    /// Updates metrics for a session from an RTCP report.
    pub fn update_metrics(
        &mut self,
        session_id: &str,
        bytes_recv: u64,
        bytes_sent: u64,
        pkts_recv: u64,
        pkts_sent: u64,
        pkts_lost: u64,
        rtt_ms: f64,
        jitter_ms: f64,
    ) -> NetResult<()> {
        let m =
            self.metrics.sessions.get_mut(session_id).ok_or_else(|| {
                NetError::not_found(format!("No metrics for session {session_id}"))
            })?;
        m.update(
            bytes_recv, bytes_sent, pkts_recv, pkts_sent, pkts_lost, rtt_ms, jitter_ms,
        );
        Ok(())
    }

    /// Returns metrics for a single session.
    #[must_use]
    pub fn session_metrics(&self, session_id: &str) -> Option<&SessionMetrics> {
        self.metrics.get(session_id)
    }

    /// Returns aggregate metrics across all sessions.
    #[must_use]
    pub fn aggregate_metrics(&self) -> SessionMetrics {
        self.metrics.aggregate()
    }

    /// Returns Prometheus text exposition (if enabled).
    #[must_use]
    pub fn prometheus_metrics(&self) -> Option<String> {
        if self.config.enable_prometheus {
            Some(self.metrics.prometheus_exposition())
        } else {
            None
        }
    }

    /// Returns recent audit log entries.
    pub fn audit_tail(&self, n: usize) -> Vec<AuditEntry> {
        self.audit_log
            .lock()
            .map(|log| log.tail(n).to_vec())
            .unwrap_or_default()
    }

    /// Returns active session count.
    #[must_use]
    pub fn active_session_count(&self) -> usize {
        self.sessions.values().filter(|s| !s.terminated).count()
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    fn check_auth(
        &self,
        auth_header: Option<&str>,
        required_scope: &TokenScope,
        remote_addr: Option<&str>,
    ) -> NetResult<Option<String>> {
        match &self.validator {
            None => Ok(None), // no auth configured
            Some(validator) => {
                let header = auth_header.ok_or_else(|| {
                    self.dispatcher.emit(&SessionEvent::AuthFailed {
                        remote_addr: remote_addr.map(|s| s.to_owned()),
                        reason: "Missing Authorization header".to_owned(),
                    });
                    NetError::authentication("Authorization header required")
                })?;

                let claims = validator.validate_header(header).map_err(|e| {
                    self.dispatcher.emit(&SessionEvent::AuthFailed {
                        remote_addr: remote_addr.map(|s| s.to_owned()),
                        reason: e.to_string(),
                    });
                    e
                })?;

                if !claims.has_scope(required_scope) {
                    let msg = format!("Token missing required scope: {}", required_scope.as_str());
                    self.dispatcher.emit(&SessionEvent::AuthFailed {
                        remote_addr: remote_addr.map(|s| s.to_owned()),
                        reason: msg.clone(),
                    });
                    return Err(NetError::authentication(msg));
                }

                Ok(Some(claims.subject))
            }
        }
    }

    fn next_id(&mut self) -> String {
        self.counter += 1;
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        format!("{ts:x}-{:04x}", self.counter)
    }
}

// ─── Crypto Helpers ──────────────────────────────────────────────────────────

/// Pure-Rust HMAC-SHA256 producing a lowercase hex string.
///
/// Uses the standard HMAC construction:
/// `HMAC(K, m) = H( (K ^ opad) || H( (K ^ ipad) || m ) )`
fn hmac_sha256_hex(key: &[u8], message: &[u8]) -> String {
    const BLOCK_SIZE: usize = 64;
    const IPAD: u8 = 0x36;
    const OPAD: u8 = 0x5c;

    // Key must be exactly BLOCK_SIZE bytes
    let mut k_block = [0u8; BLOCK_SIZE];
    if key.len() <= BLOCK_SIZE {
        k_block[..key.len()].copy_from_slice(key);
    } else {
        // Hash the key if it's longer than the block size
        let hashed = sha256_hash(key);
        k_block[..32].copy_from_slice(&hashed);
    }

    let ipad_key: Vec<u8> = k_block.iter().map(|b| b ^ IPAD).collect();
    let opad_key: Vec<u8> = k_block.iter().map(|b| b ^ OPAD).collect();

    let mut inner = ipad_key.clone();
    inner.extend_from_slice(message);
    let inner_hash = sha256_hash(&inner);

    let mut outer = opad_key.clone();
    outer.extend_from_slice(&inner_hash);
    let result = sha256_hash(&outer);

    result.iter().map(|b| format!("{b:02x}")).collect()
}

/// Constant-time string comparison (prevents timing attacks).
fn constant_time_eq(a: &str, b: &str) -> bool {
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    if ab.len() != bb.len() {
        return false;
    }
    ab.iter()
        .zip(bb.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

/// Pure-Rust SHA-256 (NIST FIPS 180-4).
fn sha256_hash(data: &[u8]) -> [u8; 32] {
    // Initial hash values (first 32 bits of fractional parts of sqrt of first 8 primes)
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // Round constants (first 32 bits of fractional parts of cube roots of first 64 primes)
    // Source: NIST FIPS 180-4
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
    let len = data.len();
    let bit_len = (len as u64) * 8;

    let mut padded = data.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0x00);
    }
    for i in (0..8).rev() {
        padded.push(((bit_len >> (i * 8)) & 0xff) as u8);
    }

    // Process each 512-bit block
    for block in padded.chunks(64) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks(4).enumerate().take(16) {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
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

    let mut out = [0u8; 32];
    for (i, &word) in h.iter().enumerate() {
        let bytes = word.to_be_bytes();
        out[i * 4..i * 4 + 4].copy_from_slice(&bytes);
    }
    out
}

/// Base64url decode (no padding).
fn base64url_decode(s: &str) -> Result<Vec<u8>, ()> {
    let padded = match s.len() % 4 {
        2 => format!("{s}=="),
        3 => format!("{s}="),
        _ => s.to_owned(),
    };
    let std_b64 = padded.replace('-', "+").replace('_', "/");
    base64_decode_standard(&std_b64)
}

/// Base64url encode (no padding).
fn base64url_encode(data: &[u8]) -> String {
    base64_encode_standard(data)
        .replace('+', "-")
        .replace('/', "_")
        .trim_end_matches('=')
        .to_owned()
}

/// Standard base64 decode.
fn base64_decode_standard(s: &str) -> Result<Vec<u8>, ()> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut decode_map = [0xffu8; 256];
    for (i, &c) in TABLE.iter().enumerate() {
        decode_map[c as usize] = i as u8;
    }

    let stripped: Vec<u8> = s.bytes().filter(|&c| c != b'=').collect();
    let mut out = Vec::with_capacity(stripped.len() * 3 / 4);

    let mut buf = 0u32;
    let mut bits = 0u32;

    for &c in &stripped {
        let val = decode_map[c as usize];
        if val == 0xff {
            return Err(());
        }
        buf = (buf << 6) | u32::from(val);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }

    Ok(out)
}

/// Standard base64 encode.
fn base64_encode_standard(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut out = Vec::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        let n = (u32::from(b0) << 16) | (u32::from(b1) << 8) | u32::from(b2);
        out.push(TABLE[((n >> 18) & 0x3f) as usize]);
        out.push(TABLE[((n >> 12) & 0x3f) as usize]);
        out.push(if chunk.len() > 1 {
            TABLE[((n >> 6) & 0x3f) as usize]
        } else {
            b'='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(n & 0x3f) as usize]
        } else {
            b'='
        });
    }
    String::from_utf8(out).unwrap_or_default()
}

/// Parses claims from a JSON payload string.
///
/// Expected fields: `sub`, `iss`, `exp`, `iat`, `jti`, `scope` (space-separated).
fn parse_claims(json: &str) -> NetResult<TokenClaims> {
    let mut claims = TokenClaims {
        subject: String::new(),
        issuer: String::new(),
        expires_at: 0,
        issued_at: 0,
        scopes: Vec::new(),
        jwt_id: String::new(),
        metadata: HashMap::new(),
    };

    for (key, value) in simple_json_kv(json) {
        match key.as_str() {
            "sub" => claims.subject = value,
            "iss" => claims.issuer = value,
            "exp" => {
                claims.expires_at = value.parse().unwrap_or(0);
            }
            "iat" => {
                claims.issued_at = value.parse().unwrap_or(0);
            }
            "jti" => claims.jwt_id = value,
            "scope" => {
                claims.scopes = value.split_whitespace().map(TokenScope::from_str).collect();
            }
            other => {
                claims.metadata.insert(other.to_owned(), value);
            }
        }
    }

    if claims.subject.is_empty() {
        return Err(NetError::authentication("Token missing 'sub' claim"));
    }

    Ok(claims)
}

/// Minimal JSON key-value extractor (string + number values only).
fn simple_json_kv(json: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let mut s = json.trim().trim_matches(|c| c == '{' || c == '}');

    while !s.is_empty() {
        s = s.trim_start_matches([',', ' ', '\n', '\r', '\t']);
        if s.is_empty() {
            break;
        }

        // Parse key
        let Some(key_start) = s.find('"') else {
            break;
        };
        s = &s[key_start + 1..];
        let Some(key_end) = s.find('"') else { break };
        let key = s[..key_end].to_owned();
        s = &s[key_end + 1..];

        // Skip colon
        let Some(colon) = s.find(':') else { break };
        s = &s[colon + 1..].trim_start();

        // Parse value
        if s.starts_with('"') {
            s = &s[1..];
            let Some(val_end) = s.find('"') else { break };
            pairs.push((key, s[..val_end].to_owned()));
            s = &s[val_end + 1..];
        } else {
            // Numeric value
            let val_end = s.find(|c: char| c == ',' || c == '}').unwrap_or(s.len());
            let val = s[..val_end].trim().to_owned();
            if !val.is_empty() {
                pairs.push((key, val));
            }
            s = &s[val_end..];
        }
    }

    pairs
}

/// Builds a compact JSON payload string from `TokenClaims`.
fn build_payload_json(claims: &TokenClaims) -> String {
    let scopes: Vec<&str> = claims.scopes.iter().map(|s| s.as_str()).collect();
    let scope_str = scopes.join(" ");
    format!(
        r#"{{"sub":"{}","iss":"{}","exp":{},"iat":{},"jti":"{}","scope":"{}"}}"#,
        claims.subject,
        claims.issuer,
        claims.expires_at,
        claims.issued_at,
        claims.jwt_id,
        scope_str,
    )
}

/// DJB2 hash for ETag generation.
fn djb2_hash(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(u64::from(byte));
    }
    hash
}

/// Generates a minimal SDP answer (recvonly — server receives from client).
fn generate_sdp_answer_recvonly(offer: &str) -> String {
    let mut answer = String::with_capacity(offer.len() + 128);
    answer.push_str("v=0\r\n");
    answer.push_str("o=- 0 0 IN IP4 0.0.0.0\r\n");
    answer.push_str("s=-\r\n");
    answer.push_str("t=0 0\r\n");
    answer.push_str("a=group:BUNDLE 0\r\n");
    for line in offer.lines() {
        let t = line.trim();
        if t.starts_with("m=") {
            answer.push_str(&format!("{t}\r\n"));
            answer.push_str("c=IN IP4 0.0.0.0\r\n");
            answer.push_str("a=recvonly\r\n");
            answer.push_str("a=rtcp-mux\r\n");
        } else if t.starts_with("a=ice-ufrag:")
            || t.starts_with("a=ice-pwd:")
            || t.starts_with("a=fingerprint:")
        {
            answer.push_str(&format!("{t}\r\n"));
        }
    }
    answer
}

/// Generates a minimal SDP answer (sendonly — server sends to client).
fn generate_sdp_answer_sendonly(offer: &str) -> String {
    let mut answer = String::with_capacity(offer.len() + 128);
    answer.push_str("v=0\r\n");
    answer.push_str("o=- 0 0 IN IP4 0.0.0.0\r\n");
    answer.push_str("s=-\r\n");
    answer.push_str("t=0 0\r\n");
    answer.push_str("a=group:BUNDLE 0\r\n");
    for line in offer.lines() {
        let t = line.trim();
        if t.starts_with("m=") {
            answer.push_str(&format!("{t}\r\n"));
            answer.push_str("c=IN IP4 0.0.0.0\r\n");
            answer.push_str("a=sendonly\r\n");
            answer.push_str("a=rtcp-mux\r\n");
        } else if t.starts_with("a=ice-ufrag:")
            || t.starts_with("a=ice-pwd:")
            || t.starts_with("a=fingerprint:")
        {
            answer.push_str(&format!("{t}\r\n"));
        }
    }
    answer
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper builders ──────────────────────────────────────────────────────

    fn validator() -> BearerTokenValidator {
        BearerTokenValidator::new(b"test-secret-key").with_issuer("test-issuer")
    }

    fn make_claims(subject: &str, scopes: Vec<TokenScope>) -> TokenClaims {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        TokenClaims {
            subject: subject.to_owned(),
            issuer: "test-issuer".to_owned(),
            expires_at: now + 3600,
            issued_at: now,
            scopes,
            jwt_id: "test-jti".to_owned(),
            metadata: HashMap::new(),
        }
    }

    fn sample_offer() -> &'static str {
        "v=0\r\n\
         o=- 0 0 IN IP4 0.0.0.0\r\n\
         s=-\r\n\
         t=0 0\r\n\
         a=ice-ufrag:abc123\r\n\
         a=ice-pwd:secret\r\n\
         a=fingerprint:sha-256 AA:BB:CC\r\n\
         m=video 9 UDP/TLS/RTP/SAVPF 96\r\n\
         a=sendonly\r\n\
         a=rtcp-mux\r\n"
    }

    // 1. TokenScope round-trip
    #[test]
    fn test_token_scope_roundtrip() {
        assert_eq!(TokenScope::from_str("publish"), TokenScope::Publish);
        assert_eq!(TokenScope::from_str("subscribe"), TokenScope::Subscribe);
        assert_eq!(TokenScope::from_str("admin"), TokenScope::Admin);
        assert_eq!(
            TokenScope::from_str("custom:thing"),
            TokenScope::Custom("custom:thing".to_owned())
        );
    }

    // 2. Admin scope grants everything
    #[test]
    fn test_admin_grants_all() {
        let claims = make_claims("user1", vec![TokenScope::Admin]);
        assert!(claims.has_scope(&TokenScope::Publish));
        assert!(claims.has_scope(&TokenScope::Subscribe));
    }

    // 3. Publish scope does not grant subscribe
    #[test]
    fn test_publish_does_not_grant_subscribe() {
        let claims = make_claims("user1", vec![TokenScope::Publish]);
        assert!(claims.has_scope(&TokenScope::Publish));
        assert!(!claims.has_scope(&TokenScope::Subscribe));
    }

    // 4. Token generation and validation round-trip
    #[test]
    fn test_token_roundtrip() {
        let v = validator();
        let claims = make_claims("stream1", vec![TokenScope::Publish]);
        let token = v.generate_token(&claims);
        let decoded = v.validate_token(&token).expect("should be valid");
        assert_eq!(decoded.subject, "stream1");
        assert_eq!(decoded.issuer, "test-issuer");
        assert!(decoded.has_scope(&TokenScope::Publish));
    }

    // 5. Bearer header prefix parsing
    #[test]
    fn test_bearer_header() {
        let v = validator();
        let claims = make_claims("u", vec![TokenScope::Publish]);
        let token = v.generate_token(&claims);
        let header = format!("Bearer {token}");
        let decoded = v.validate_header(&header).expect("should be valid");
        assert_eq!(decoded.subject, "u");
    }

    // 6. Invalid signature rejected
    #[test]
    fn test_invalid_signature_rejected() {
        let v = validator();
        let claims = make_claims("u", vec![TokenScope::Publish]);
        let mut token = v.generate_token(&claims);
        // Corrupt last 4 chars
        let len = token.len();
        token.replace_range(len - 4..len, "xxxx");
        assert!(v.validate_token(&token).is_err());
    }

    // 7. Expired token rejected
    #[test]
    fn test_expired_token_rejected() {
        let v = BearerTokenValidator::new(b"secret").with_clock_skew(0);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let claims = TokenClaims {
            subject: "u".to_owned(),
            issuer: "x".to_owned(),
            expires_at: now - 1, // already expired
            issued_at: now - 100,
            scopes: vec![TokenScope::Publish],
            jwt_id: "j".to_owned(),
            metadata: HashMap::new(),
        };
        let token = v.generate_token(&claims);
        assert!(v.validate_token(&token).is_err());
    }

    // 8. Wrong issuer rejected
    #[test]
    fn test_wrong_issuer_rejected() {
        let v = BearerTokenValidator::new(b"secret").with_issuer("allowed-issuer");
        let claims = make_claims("u", vec![TokenScope::Publish]);
        // make_claims uses "test-issuer" which is not "allowed-issuer"
        let token = v.generate_token(&claims);
        // Must re-sign with same key but different issuer check
        let v2 = BearerTokenValidator::new(b"secret").with_issuer("allowed-issuer");
        // Token has issuer "test-issuer" which does not match "allowed-issuer"
        assert!(v2.validate_token(&token).is_err());
    }

    // 9. Session metrics update
    #[test]
    fn test_session_metrics_update() {
        let mut m = SessionMetrics::new();
        m.update(1000, 500, 100, 50, 2, 15.0, 2.0);
        assert_eq!(m.bytes_received, 1000);
        assert_eq!(m.packets_lost, 2);
        assert!(m.loss_fraction > 0.0);
        assert_eq!(m.rtt_ms, 15.0);
    }

    // 10. Session metrics to_map
    #[test]
    fn test_session_metrics_map() {
        let mut m = SessionMetrics::new();
        m.update(500, 250, 50, 25, 0, 10.0, 1.0);
        let map = m.to_map();
        assert_eq!(map.get("bytes_received").map(|s| s.as_str()), Some("500"));
        assert_eq!(map.get("rtt_ms").map(|s| s.as_str()), Some("10.00"));
    }

    // 11. Metrics registry aggregate
    #[test]
    fn test_metrics_registry_aggregate() {
        let mut reg = MetricsRegistry::new();
        let mut m1 = SessionMetrics::new();
        m1.update(1000, 0, 100, 0, 0, 10.0, 1.0);
        let mut m2 = SessionMetrics::new();
        m2.update(2000, 0, 200, 0, 0, 20.0, 2.0);
        reg.upsert("s1", m1);
        reg.upsert("s2", m2);
        let agg = reg.aggregate();
        assert_eq!(agg.bytes_received, 3000);
        assert_eq!(agg.rtt_ms, 15.0);
    }

    // 12. Prometheus exposition
    #[test]
    fn test_prometheus_exposition() {
        let mut reg = MetricsRegistry::new();
        reg.upsert("s1", SessionMetrics::new());
        let text = reg.prometheus_exposition();
        assert!(text.contains("whip_whep_sessions_total 1"));
        assert!(text.contains("TYPE"));
    }

    // 13. Audit log append and tail
    #[test]
    fn test_audit_log_tail() {
        let mut log = AuditLog::new(100);
        for i in 0..10 {
            log.append(
                Some(format!("s{i}")),
                AuditCategory::Session,
                format!("event {i}"),
            );
        }
        let tail = log.tail(3);
        assert_eq!(tail.len(), 3);
        assert_eq!(tail[2].description, "event 9");
    }

    // 14. Audit log capacity eviction
    #[test]
    fn test_audit_log_eviction() {
        let mut log = AuditLog::new(5);
        for i in 0..8u32 {
            log.append(None, AuditCategory::Session, format!("e{i}"));
        }
        assert_eq!(log.len(), 5);
        assert_eq!(log.tail(1)[0].description, "e7");
    }

    // 15. Audit log filter by category
    #[test]
    fn test_audit_log_filter_category() {
        let mut log = AuditLog::new(100);
        log.append(Some("s1".to_owned()), AuditCategory::Auth, "fail");
        log.append(Some("s2".to_owned()), AuditCategory::Session, "ok");
        let auth_entries = log.entries_by_category(AuditCategory::Auth);
        assert_eq!(auth_entries.len(), 1);
        assert_eq!(auth_entries[0].description, "fail");
    }

    // 16. EventDispatcher fan-out
    #[test]
    fn test_event_dispatcher_fanout() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Counter(Arc<AtomicUsize>);
        impl EventHook for Counter {
            fn on_event(&self, _: &SessionEvent) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let mut dispatcher = EventDispatcher::new();
        let count = Arc::new(AtomicUsize::new(0));
        dispatcher.register(Arc::new(Counter(Arc::clone(&count))));
        dispatcher.register(Arc::new(Counter(Arc::clone(&count))));

        dispatcher.emit(&SessionEvent::AuthFailed {
            remote_addr: None,
            reason: "test".to_owned(),
        });
        assert_eq!(count.load(Ordering::Relaxed), 2);
    }

    // 17. AuditLog as EventHook
    #[test]
    fn test_audit_log_as_hook() {
        let log = Arc::new(Mutex::new(AuditLog::new(100)));
        let mut dispatcher = EventDispatcher::new();
        dispatcher.register(Arc::new(AuditLogHook(Arc::clone(&log))) as Arc<dyn EventHook>);

        dispatcher.emit(&SessionEvent::Created {
            session_id: "abc".to_owned(),
            protocol: SessionProtocol::Whip,
            remote_addr: None,
        });
        dispatcher.emit(&SessionEvent::AuthFailed {
            remote_addr: Some("1.2.3.4".to_owned()),
            reason: "expired".to_owned(),
        });

        let guard = log.lock().expect("should lock");
        assert_eq!(guard.len(), 2);
        assert_eq!(guard.entries_by_category(AuditCategory::Auth).len(), 1);
    }

    // 18. Extended endpoint WHIP no auth
    #[test]
    fn test_extended_whip_no_auth() {
        let mut ep = WhipWhepExtendedEndpoint::new(ExtendedEndpointConfig::default());
        let result = ep.handle_whip_post(sample_offer(), None, Some("127.0.0.1"), vec![]);
        assert!(result.is_ok());
        let created = result.expect("should create");
        assert!(created.resource_path.starts_with("/whip/resource/"));
    }

    // 19. Extended endpoint WHEP no auth
    #[test]
    fn test_extended_whep_no_auth() {
        let mut ep = WhipWhepExtendedEndpoint::new(ExtendedEndpointConfig::default());
        let result = ep.handle_whep_post(
            sample_offer(),
            None,
            Some("stream1"),
            Some("127.0.0.1"),
            vec![],
        );
        assert!(result.is_ok());
    }

    // 20. Extended endpoint WHIP with valid token
    #[test]
    fn test_extended_whip_with_valid_token() {
        let secret = b"my-hmac-secret";
        let mut cfg = ExtendedEndpointConfig::default();
        cfg.token_secret = secret.to_vec();
        let mut ep = WhipWhepExtendedEndpoint::new(cfg);

        let v = BearerTokenValidator::new(secret);
        let claims = make_claims("broadcaster", vec![TokenScope::Publish]);
        let token = v.generate_token(&claims);
        let auth_header = format!("Bearer {token}");

        let result = ep.handle_whip_post(sample_offer(), Some(&auth_header), None, vec![]);
        assert!(result.is_ok());
        assert_eq!(ep.active_session_count(), 1);
    }

    // 21. Extended endpoint WHIP rejected without token when required
    #[test]
    fn test_extended_whip_missing_token() {
        let mut cfg = ExtendedEndpointConfig::default();
        cfg.token_secret = b"secret".to_vec();
        let mut ep = WhipWhepExtendedEndpoint::new(cfg);
        let result = ep.handle_whip_post(sample_offer(), None, None, vec![]);
        assert!(result.is_err());
    }

    // 22. Extended endpoint WHIP rejected with wrong scope
    #[test]
    fn test_extended_whip_wrong_scope() {
        let secret = b"secret";
        let mut cfg = ExtendedEndpointConfig::default();
        cfg.token_secret = secret.to_vec();
        let mut ep = WhipWhepExtendedEndpoint::new(cfg);

        let v = BearerTokenValidator::new(secret);
        let claims = make_claims("viewer", vec![TokenScope::Subscribe]); // subscribe only
        let token = v.generate_token(&claims);
        let header = format!("Bearer {token}");

        let result = ep.handle_whip_post(sample_offer(), Some(&header), None, vec![]);
        assert!(result.is_err());
    }

    // 23. Extended endpoint delete session
    #[test]
    fn test_extended_delete_session() {
        let mut ep = WhipWhepExtendedEndpoint::new(ExtendedEndpointConfig::default());
        let created = ep
            .handle_whip_post(sample_offer(), None, None, vec![])
            .expect("should create");
        let session_id = &created.session_id;
        ep.handle_delete(session_id).expect("should delete");
        assert_eq!(ep.active_session_count(), 0);
    }

    // 24. Extended endpoint update and retrieve metrics
    #[test]
    fn test_extended_metrics_update() {
        let mut ep = WhipWhepExtendedEndpoint::new(ExtendedEndpointConfig::default());
        let created = ep
            .handle_whip_post(sample_offer(), None, None, vec![])
            .expect("should create");
        let id = &created.session_id;
        ep.update_metrics(id, 5000, 1000, 500, 100, 10, 20.0, 3.0)
            .expect("should update");
        let m = ep.session_metrics(id).expect("should have metrics");
        assert_eq!(m.bytes_received, 5000);
    }

    // 25. Extended endpoint Prometheus output
    #[test]
    fn test_extended_prometheus() {
        let mut ep = WhipWhepExtendedEndpoint::new(ExtendedEndpointConfig::default());
        ep.handle_whip_post(sample_offer(), None, None, vec![])
            .expect("should create");
        let prom = ep.prometheus_metrics().expect("should have metrics");
        assert!(prom.contains("whip_whep_sessions_total 1"));
    }

    // 26. Audit log entries written for lifecycle events
    #[test]
    fn test_audit_entries_lifecycle() {
        let mut ep = WhipWhepExtendedEndpoint::new(ExtendedEndpointConfig::default());
        let created = ep
            .handle_whip_post(sample_offer(), None, None, vec![])
            .expect("should create");
        ep.activate_session(&created.session_id)
            .expect("should activate");
        ep.handle_delete(&created.session_id)
            .expect("should delete");

        let entries = ep.audit_tail(10);
        assert!(entries.len() >= 3, "Expected at least 3 audit entries");
    }

    // 27. SHA-256 known test vector
    #[test]
    fn test_sha256_empty() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let hash = sha256_hash(b"");
        let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    // 28. SHA-256 known test vector #2
    #[test]
    fn test_sha256_abc() {
        // SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        // Verified against Python hashlib.sha256(b"abc").hexdigest()
        let hash = sha256_hash(b"abc");
        let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    // 29. Base64url round-trip
    #[test]
    fn test_base64url_roundtrip() {
        let data = b"Hello, World! \x00\xff";
        let encoded = base64url_encode(data);
        let decoded = base64url_decode(&encoded).expect("should decode");
        assert_eq!(decoded, data);
    }

    // 30. Max sessions limit
    #[test]
    fn test_max_sessions_limit() {
        let mut cfg = ExtendedEndpointConfig::default();
        cfg.max_sessions = 2;
        let mut ep = WhipWhepExtendedEndpoint::new(cfg);
        ep.handle_whip_post(sample_offer(), None, None, vec![])
            .expect("1st ok");
        ep.handle_whip_post(sample_offer(), None, None, vec![])
            .expect("2nd ok");
        let result = ep.handle_whip_post(sample_offer(), None, None, vec![]);
        assert!(result.is_err());
    }
}
