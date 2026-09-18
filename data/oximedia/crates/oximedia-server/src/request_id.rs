//! Request ID propagation for end-to-end request tracing.
//!
//! Generates and propagates unique request identifiers through all handler
//! layers. If an incoming request carries an `X-Request-ID` header the value
//! is reused; otherwise a new UUID v4 is generated.  The ID is always set on
//! the response via `X-Request-ID` and made available to handlers through an
//! axum extension.
//!
//! # Usage
//!
//! ```rust,ignore
//! use oximedia_server::request_id::{RequestIdLayer, RequestId};
//!
//! // Add the layer to your router:
//! let app = Router::new()
//!     .route("/api/data", get(handler))
//!     .layer(RequestIdLayer::new());
//!
//! // Extract inside a handler:
//! async fn handler(request_id: RequestId) -> String {
//!     format!("Request ID: {}", request_id.as_str())
//! }
//! ```

#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

// ── Request ID value type ────────────────────────────────────────────────────

/// A validated, non-empty request identifier string.
///
/// Created from an incoming `X-Request-ID` header or freshly generated.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RequestId(String);

impl RequestId {
    /// Maximum permitted length for an incoming request ID header value.
    /// Prevents clients from injecting arbitrarily large IDs.
    pub const MAX_LENGTH: usize = 128;

    /// The canonical HTTP header name for request ID propagation.
    pub const HEADER_NAME: &'static str = "X-Request-ID";

    /// Generates a fresh request ID from a UUID v4.
    pub fn generate() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Attempts to parse a request ID from a raw header value.
    ///
    /// Returns `None` if the value is empty, too long, or contains
    /// non-printable / non-ASCII characters.
    pub fn from_header(value: &str) -> Option<Self> {
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed.len() > Self::MAX_LENGTH {
            return None;
        }
        // Only allow printable ASCII (0x20..0x7E)
        if trimmed.bytes().all(|b| (0x20..=0x7E).contains(&b)) {
            Some(Self(trimmed.to_string()))
        } else {
            None
        }
    }

    /// Returns the request ID as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the wrapper and returns the inner `String`.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for RequestId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

// ── Request ID Layer ─────────────────────────────────────────────────────────

/// Configuration for the request ID layer.
#[derive(Debug, Clone)]
pub struct RequestIdConfig {
    /// Whether to accept an incoming `X-Request-ID` header.
    /// When `false`, always generates a fresh ID.
    pub trust_incoming: bool,
    /// Whether to echo the request ID in the response header.
    pub echo_in_response: bool,
    /// Optional prefix appended to generated IDs for service identification.
    pub prefix: Option<String>,
}

impl Default for RequestIdConfig {
    fn default() -> Self {
        Self {
            trust_incoming: true,
            echo_in_response: true,
            prefix: None,
        }
    }
}

/// Middleware layer that injects a [`RequestId`] into each request.
#[derive(Debug, Clone)]
pub struct RequestIdLayer {
    config: RequestIdConfig,
}

impl RequestIdLayer {
    /// Creates a new layer with default configuration.
    pub fn new() -> Self {
        Self {
            config: RequestIdConfig::default(),
        }
    }

    /// Creates a layer with a custom configuration.
    pub fn with_config(config: RequestIdConfig) -> Self {
        Self { config }
    }

    /// Resolves the request ID for a given set of headers.
    ///
    /// If `trust_incoming` is `true` and the headers contain a valid
    /// `X-Request-ID`, that value is used; otherwise a new one is generated.
    pub fn resolve(&self, headers: &HashMap<String, String>) -> RequestId {
        if self.config.trust_incoming {
            if let Some(value) = headers.get(RequestId::HEADER_NAME) {
                if let Some(id) = RequestId::from_header(value) {
                    return id;
                }
            }
            // Also check lowercase variant
            if let Some(value) = headers.get("x-request-id") {
                if let Some(id) = RequestId::from_header(value) {
                    return id;
                }
            }
        }
        self.generate_id()
    }

    /// Generates a new request ID, optionally with a prefix.
    fn generate_id(&self) -> RequestId {
        match &self.config.prefix {
            Some(prefix) => RequestId(format!("{}-{}", prefix, Uuid::new_v4())),
            None => RequestId::generate(),
        }
    }

    /// Returns the response headers that should be added.
    pub fn response_headers(&self, id: &RequestId) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        if self.config.echo_in_response {
            headers.insert(RequestId::HEADER_NAME.to_string(), id.to_string());
        }
        headers
    }
}

impl Default for RequestIdLayer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Request context with tracing ─────────────────────────────────────────────

/// A traced request context that carries the request ID alongside tracing
/// metadata for structured logging and downstream propagation.
#[derive(Debug, Clone)]
pub struct TracedRequest {
    /// The request ID for this request.
    pub request_id: RequestId,
    /// The originating service (if provided via `X-Forwarded-Service`).
    pub source_service: Option<String>,
    /// The parent request ID (for multi-hop tracing chains).
    pub parent_request_id: Option<RequestId>,
    /// Arbitrary trace baggage carried through the chain.
    pub baggage: HashMap<String, String>,
}

impl TracedRequest {
    /// Creates a new traced request with the given ID.
    pub fn new(request_id: RequestId) -> Self {
        Self {
            request_id,
            source_service: None,
            parent_request_id: None,
            baggage: HashMap::new(),
        }
    }

    /// Resolves from incoming headers, extracting parent ID and service info.
    pub fn from_headers(headers: &HashMap<String, String>, layer: &RequestIdLayer) -> Self {
        let request_id = layer.resolve(headers);
        let parent_request_id = headers
            .get("X-Parent-Request-ID")
            .or_else(|| headers.get("x-parent-request-id"))
            .and_then(|v| RequestId::from_header(v));
        let source_service = headers
            .get("X-Forwarded-Service")
            .or_else(|| headers.get("x-forwarded-service"))
            .cloned();

        Self {
            request_id,
            source_service,
            parent_request_id,
            baggage: HashMap::new(),
        }
    }

    /// Adds a baggage key-value pair.
    #[must_use]
    pub fn with_baggage(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.baggage.insert(key.into(), value.into());
        self
    }

    /// Produces the headers that should be propagated to downstream services.
    pub fn propagation_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        headers.insert(
            RequestId::HEADER_NAME.to_string(),
            self.request_id.to_string(),
        );
        headers.insert(
            "X-Parent-Request-ID".to_string(),
            self.request_id.to_string(),
        );
        if let Some(service) = &self.source_service {
            headers.insert("X-Forwarded-Service".to_string(), service.clone());
        }
        headers
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RequestId ────────────────────────────────────────────────────────────

    #[test]
    fn test_generate_produces_valid_uuid() {
        let id = RequestId::generate();
        // UUID v4 is 36 chars: 8-4-4-4-12
        assert_eq!(id.as_str().len(), 36);
        assert!(id.as_str().contains('-'));
    }

    #[test]
    fn test_generate_unique() {
        let a = RequestId::generate();
        let b = RequestId::generate();
        assert_ne!(a, b);
    }

    #[test]
    fn test_from_header_valid() {
        let id = RequestId::from_header("abc-123-def");
        assert!(id.is_some());
        assert_eq!(id.as_ref().map(|i| i.as_str()), Some("abc-123-def"));
    }

    #[test]
    fn test_from_header_trims_whitespace() {
        let id = RequestId::from_header("  req-42  ");
        assert!(id.is_some());
        assert_eq!(id.as_ref().map(|i| i.as_str()), Some("req-42"));
    }

    #[test]
    fn test_from_header_rejects_empty() {
        assert!(RequestId::from_header("").is_none());
        assert!(RequestId::from_header("   ").is_none());
    }

    #[test]
    fn test_from_header_rejects_too_long() {
        let long = "x".repeat(RequestId::MAX_LENGTH + 1);
        assert!(RequestId::from_header(&long).is_none());
    }

    #[test]
    fn test_from_header_rejects_non_ascii() {
        assert!(RequestId::from_header("req-\x01-bad").is_none());
        assert!(RequestId::from_header("req-\u{0080}-bad").is_none());
    }

    #[test]
    fn test_from_header_accepts_max_length() {
        let max = "a".repeat(RequestId::MAX_LENGTH);
        assert!(RequestId::from_header(&max).is_some());
    }

    #[test]
    fn test_display() {
        let id = RequestId::from_header("my-request").expect("valid");
        assert_eq!(format!("{}", id), "my-request");
    }

    #[test]
    fn test_into_inner() {
        let id = RequestId::from_header("inner-test").expect("valid");
        let s: String = id.into_inner();
        assert_eq!(s, "inner-test");
    }

    // ── RequestIdLayer ──────────────────────────────────────────────────────

    #[test]
    fn test_layer_generates_when_no_header() {
        let layer = RequestIdLayer::new();
        let headers = HashMap::new();
        let id = layer.resolve(&headers);
        assert_eq!(id.as_str().len(), 36); // UUID format
    }

    #[test]
    fn test_layer_trusts_incoming_header() {
        let layer = RequestIdLayer::new();
        let mut headers = HashMap::new();
        headers.insert("X-Request-ID".to_string(), "client-req-99".to_string());
        let id = layer.resolve(&headers);
        assert_eq!(id.as_str(), "client-req-99");
    }

    #[test]
    fn test_layer_trusts_lowercase_header() {
        let layer = RequestIdLayer::new();
        let mut headers = HashMap::new();
        headers.insert("x-request-id".to_string(), "lower-id".to_string());
        let id = layer.resolve(&headers);
        assert_eq!(id.as_str(), "lower-id");
    }

    #[test]
    fn test_layer_ignores_incoming_when_untrusted() {
        let config = RequestIdConfig {
            trust_incoming: false,
            ..Default::default()
        };
        let layer = RequestIdLayer::with_config(config);
        let mut headers = HashMap::new();
        headers.insert("X-Request-ID".to_string(), "should-ignore".to_string());
        let id = layer.resolve(&headers);
        assert_ne!(id.as_str(), "should-ignore");
    }

    #[test]
    fn test_layer_generates_with_prefix() {
        let config = RequestIdConfig {
            prefix: Some("svc-media".to_string()),
            ..Default::default()
        };
        let layer = RequestIdLayer::with_config(config);
        let headers = HashMap::new();
        let id = layer.resolve(&headers);
        assert!(id.as_str().starts_with("svc-media-"));
    }

    #[test]
    fn test_layer_response_headers_echo() {
        let layer = RequestIdLayer::new();
        let id = RequestId::from_header("resp-echo").expect("valid");
        let headers = layer.response_headers(&id);
        assert_eq!(
            headers.get("X-Request-ID").map(String::as_str),
            Some("resp-echo")
        );
    }

    #[test]
    fn test_layer_response_headers_no_echo() {
        let config = RequestIdConfig {
            echo_in_response: false,
            ..Default::default()
        };
        let layer = RequestIdLayer::with_config(config);
        let id = RequestId::generate();
        let headers = layer.response_headers(&id);
        assert!(headers.is_empty());
    }

    #[test]
    fn test_layer_falls_back_on_invalid_header() {
        let layer = RequestIdLayer::new();
        let mut headers = HashMap::new();
        headers.insert("X-Request-ID".to_string(), String::new());
        let id = layer.resolve(&headers);
        // Should generate new since empty header is invalid
        assert_eq!(id.as_str().len(), 36);
    }

    // ── TracedRequest ───────────────────────────────────────────────────────

    #[test]
    fn test_traced_request_new() {
        let id = RequestId::generate();
        let traced = TracedRequest::new(id.clone());
        assert_eq!(traced.request_id, id);
        assert!(traced.source_service.is_none());
        assert!(traced.parent_request_id.is_none());
    }

    #[test]
    fn test_traced_request_from_headers_with_parent() {
        let layer = RequestIdLayer::new();
        let mut headers = HashMap::new();
        headers.insert("X-Request-ID".to_string(), "child-req".to_string());
        headers.insert("X-Parent-Request-ID".to_string(), "parent-req".to_string());
        headers.insert("X-Forwarded-Service".to_string(), "api-gateway".to_string());

        let traced = TracedRequest::from_headers(&headers, &layer);
        assert_eq!(traced.request_id.as_str(), "child-req");
        assert_eq!(
            traced.parent_request_id.as_ref().map(|p| p.as_str()),
            Some("parent-req")
        );
        assert_eq!(traced.source_service.as_deref(), Some("api-gateway"));
    }

    #[test]
    fn test_traced_request_propagation_headers() {
        let id = RequestId::from_header("prop-req").expect("valid");
        let traced = TracedRequest::new(id).with_baggage("tenant", "acme");
        let headers = traced.propagation_headers();
        assert_eq!(
            headers.get("X-Request-ID").map(String::as_str),
            Some("prop-req")
        );
        assert_eq!(
            headers.get("X-Parent-Request-ID").map(String::as_str),
            Some("prop-req")
        );
    }

    #[test]
    fn test_traced_request_baggage() {
        let id = RequestId::generate();
        let traced = TracedRequest::new(id)
            .with_baggage("tenant", "acme")
            .with_baggage("region", "eu-west-1");
        assert_eq!(
            traced.baggage.get("tenant").map(String::as_str),
            Some("acme")
        );
        assert_eq!(
            traced.baggage.get("region").map(String::as_str),
            Some("eu-west-1")
        );
    }

    #[test]
    fn test_traced_request_propagation_includes_service() {
        let id = RequestId::from_header("svc-req").expect("valid");
        let mut traced = TracedRequest::new(id);
        traced.source_service = Some("transcoder".to_string());
        let headers = traced.propagation_headers();
        assert_eq!(
            headers.get("X-Forwarded-Service").map(String::as_str),
            Some("transcoder")
        );
    }

    // ── Edge cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_request_id_as_ref() {
        let id = RequestId::from_header("ref-test").expect("valid");
        let s: &str = id.as_ref();
        assert_eq!(s, "ref-test");
    }

    #[test]
    fn test_request_id_clone_eq() {
        let a = RequestId::from_header("clone-me").expect("valid");
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn test_request_id_hash_consistent() {
        let a = RequestId::from_header("hash-me").expect("valid");
        let b = RequestId::from_header("hash-me").expect("valid");
        let mut map = HashMap::new();
        map.insert(a, 1);
        assert_eq!(map.get(&b), Some(&1));
    }
}
