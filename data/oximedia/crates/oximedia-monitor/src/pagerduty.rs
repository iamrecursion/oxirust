//! PagerDuty Events API v2 integration — pure Rust, no external HTTP client.
//!
//! Sends trigger and resolve events to PagerDuty using a hand-rolled
//! HTTP/1.1 client built on [`std::net::TcpStream`].  No `reqwest`, `hyper`,
//! or any other HTTP crate is used; the only runtime requirement is a standard
//! TCP connection to the configured endpoint host.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_monitor::pagerduty::{PagerDutyClient, PagerDutyEvent, PagerDutySeverity};
//!
//! let client = PagerDutyClient::new("my-routing-key")
//!     .with_endpoint("http://localhost:8080/v2/enqueue");
//!
//! let event = PagerDutyEvent::new("High CPU usage", PagerDutySeverity::Critical)
//!     .with_component("cpu-monitor")
//!     .with_detail("host", "encoder-01");
//!
//! // In real code: let dedup_key = client.trigger(&event)?;
//! ```

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use crate::error::{MonitorError, MonitorResult};

// ---------------------------------------------------------------------------
// PagerDutySeverity
// ---------------------------------------------------------------------------

/// PagerDuty event severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagerDutySeverity {
    /// Highest urgency — page immediately.
    Critical,
    /// Service-affecting error.
    Error,
    /// Potential problem — warning threshold crossed.
    Warning,
    /// Informational event.
    Info,
}

impl PagerDutySeverity {
    /// The PagerDuty API string value for this severity.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Critical => "critical",
            Self::Error => "error",
            Self::Warning => "warning",
            Self::Info => "info",
        }
    }
}

// ---------------------------------------------------------------------------
// PagerDutyEvent
// ---------------------------------------------------------------------------

/// A PagerDuty Events API v2 event payload.
#[derive(Debug, Clone)]
pub struct PagerDutyEvent {
    /// Short human-readable summary shown on the incident.
    pub summary: String,
    /// Severity of the event.
    pub severity: PagerDutySeverity,
    /// Optional component that is alerting (e.g. `"encoding-pipeline"`).
    pub component: Option<String>,
    /// Optional logical group (e.g. `"media-cluster"`).
    pub group: Option<String>,
    /// Arbitrary key-value details surfaced in the PagerDuty UI.
    pub details: BTreeMap<String, String>,
    /// Optional deduplication key; events with the same key update a single
    /// incident rather than opening a new one.
    pub dedup_key: Option<String>,
}

impl PagerDutyEvent {
    /// Create a minimal event with the given `summary` and `severity`.
    #[must_use]
    pub fn new(summary: impl Into<String>, severity: PagerDutySeverity) -> Self {
        Self {
            summary: summary.into(),
            severity,
            component: None,
            group: None,
            details: BTreeMap::new(),
            dedup_key: None,
        }
    }

    /// Set the component field.
    #[must_use]
    pub fn with_component(mut self, c: impl Into<String>) -> Self {
        self.component = Some(c.into());
        self
    }

    /// Set the group field.
    #[must_use]
    pub fn with_group(mut self, g: impl Into<String>) -> Self {
        self.group = Some(g.into());
        self
    }

    /// Add a key-value detail entry.
    #[must_use]
    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.details.insert(key.into(), value.into());
        self
    }

    /// Set the deduplication key.
    #[must_use]
    pub fn with_dedup_key(mut self, key: impl Into<String>) -> Self {
        self.dedup_key = Some(key.into());
        self
    }

    /// Serialize this event as a PagerDuty Events API v2 **trigger** JSON body.
    #[must_use]
    pub fn to_trigger_json(&self, routing_key: &str) -> String {
        let mut detail_fields: Vec<String> =
            self.details.iter().map(|(k, v)| json_field(k, v)).collect();
        // Ensure we always emit a details object (may be empty).
        let details_obj = json_object(&detail_fields);

        let mut payload_fields = vec![
            json_field("summary", &self.summary),
            json_field("severity", self.severity.as_str()),
        ];
        if let Some(ref comp) = self.component {
            payload_fields.push(json_field("component", comp));
        }
        if let Some(ref grp) = self.group {
            payload_fields.push(json_field("group", grp));
        }
        detail_fields.clear();
        payload_fields.push(format!("{}: {details_obj}", json_str("custom_details")));
        let payload_obj = json_object(&payload_fields);

        let mut top = vec![
            json_field("routing_key", routing_key),
            json_field("event_action", "trigger"),
            format!("{}: {payload_obj}", json_str("payload")),
        ];
        if let Some(ref dk) = self.dedup_key {
            top.push(json_field("dedup_key", dk));
        }

        json_object(&top)
    }

    /// Serialize a **resolve** action JSON body.
    #[must_use]
    pub fn to_resolve_json(dedup_key: &str, routing_key: &str) -> String {
        json_object(&[
            json_field("routing_key", routing_key),
            json_field("event_action", "resolve"),
            json_field("dedup_key", dedup_key),
        ])
    }
}

// ---------------------------------------------------------------------------
// PagerDutyClient
// ---------------------------------------------------------------------------

/// PagerDuty Events API v2 client.
///
/// Sends HTTP/1.1 POST requests over [`TcpStream`].  For HTTPS endpoints
/// the plain-TCP implementation will fail at the TLS handshake; point the
/// client at an HTTP endpoint (e.g. a local mock) for testing, or terminate
/// TLS externally (reverse proxy / stunnel).
#[derive(Debug, Clone)]
pub struct PagerDutyClient {
    /// Integration (routing) key from the PagerDuty service configuration.
    pub routing_key: String,
    /// Full HTTP endpoint URL, e.g. `https://events.pagerduty.com/v2/enqueue`.
    pub endpoint: String,
    /// TCP connect + read timeout in milliseconds.
    pub timeout_ms: u64,
}

impl PagerDutyClient {
    /// Create a client with the default PagerDuty endpoint.
    #[must_use]
    pub fn new(routing_key: impl Into<String>) -> Self {
        Self {
            routing_key: routing_key.into(),
            endpoint: "https://events.pagerduty.com/v2/enqueue".to_string(),
            timeout_ms: 10_000,
        }
    }

    /// Override the API endpoint (useful for mock servers in tests).
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Set the TCP timeout in milliseconds.
    #[must_use]
    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Trigger a PagerDuty incident.
    ///
    /// Returns the `dedup_key` from the PagerDuty response, which can later
    /// be passed to [`PagerDutyClient::resolve`].
    ///
    /// # Errors
    ///
    /// Returns an error if the TCP connection fails, the HTTP response is
    /// non-2xx, or URL parsing fails.
    pub fn trigger(&self, event: &PagerDutyEvent) -> MonitorResult<String> {
        let body = event.to_trigger_json(&self.routing_key);
        let parsed = parse_url(&self.endpoint)?;
        let (status, response_body) = send_http_post(
            &parsed.host,
            parsed.port,
            &parsed.path,
            &body,
            self.timeout_ms,
        )?;

        if !(200..300).contains(&status) {
            return Err(MonitorError::Other(format!(
                "PagerDuty trigger failed: HTTP {status} — {response_body}"
            )));
        }

        // Extract dedup_key from response JSON.
        let dedup_key = extract_json_string_field(&response_body, "dedup_key")
            .or_else(|| event.dedup_key.clone())
            .unwrap_or_else(|| format!("pd-{}", event.summary.len()));

        Ok(dedup_key)
    }

    /// Resolve a PagerDuty incident by its `dedup_key`.
    ///
    /// # Errors
    ///
    /// Returns an error if the TCP connection fails or the server returns a
    /// non-2xx status.
    pub fn resolve(&self, dedup_key: &str) -> MonitorResult<()> {
        let body = PagerDutyEvent::to_resolve_json(dedup_key, &self.routing_key);
        let parsed = parse_url(&self.endpoint)?;
        let (status, response_body) = send_http_post(
            &parsed.host,
            parsed.port,
            &parsed.path,
            &body,
            self.timeout_ms,
        )?;

        if !(200..300).contains(&status) {
            return Err(MonitorError::Other(format!(
                "PagerDuty resolve failed: HTTP {status} — {response_body}"
            )));
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Private: URL parsing
// ---------------------------------------------------------------------------

struct ParsedUrl {
    #[allow(dead_code)]
    scheme: String,
    host: String,
    port: u16,
    path: String,
}

fn parse_url(url: &str) -> MonitorResult<ParsedUrl> {
    // Strip scheme.
    let (scheme, rest) = if let Some(r) = url.strip_prefix("https://") {
        ("https".to_string(), r)
    } else if let Some(r) = url.strip_prefix("http://") {
        ("http".to_string(), r)
    } else {
        return Err(MonitorError::Other(format!(
            "unsupported URL scheme in '{url}' (expected http:// or https://)"
        )));
    };

    let default_port: u16 = if scheme == "https" { 443 } else { 80 };

    // Split host[:port] from /path.
    let (authority, path) = if let Some(slash_pos) = rest.find('/') {
        (&rest[..slash_pos], rest[slash_pos..].to_string())
    } else {
        (rest, "/".to_string())
    };

    // Split host from optional :port.
    let (host, port) = if let Some(colon_pos) = authority.rfind(':') {
        let host = &authority[..colon_pos];
        let port_str = &authority[colon_pos + 1..];
        let port = port_str.parse::<u16>().map_err(|_| {
            MonitorError::Other(format!("invalid port '{port_str}' in URL '{url}'"))
        })?;
        (host.to_string(), port)
    } else {
        (authority.to_string(), default_port)
    };

    Ok(ParsedUrl {
        scheme,
        host,
        port,
        path,
    })
}

// ---------------------------------------------------------------------------
// Private: raw HTTP/1.1 POST over TcpStream
// ---------------------------------------------------------------------------

/// Send an HTTP/1.1 POST request and return `(status_code, response_body)`.
fn send_http_post(
    host: &str,
    port: u16,
    path: &str,
    body: &str,
    timeout_ms: u64,
) -> MonitorResult<(u16, String)> {
    let timeout = Duration::from_millis(timeout_ms);
    let addr = format!("{host}:{port}");

    let mut stream = TcpStream::connect_timeout(
        &addr
            .parse()
            .map_err(|_| MonitorError::Other(format!("invalid address: '{addr}'")))?,
        timeout,
    )
    .map_err(MonitorError::Io)?;

    stream
        .set_read_timeout(Some(timeout))
        .map_err(MonitorError::Io)?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(MonitorError::Io)?;

    let request = format!(
        "POST {path} HTTP/1.1\r\n\
         Host: {host}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {len}\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        len = body.len(),
    );

    stream
        .write_all(request.as_bytes())
        .map_err(MonitorError::Io)?;
    stream.flush().map_err(MonitorError::Io)?;

    let mut raw = String::new();
    stream.read_to_string(&mut raw).map_err(MonitorError::Io)?;

    parse_http_response(&raw)
}

/// Parse an HTTP/1.1 response and return `(status_code, body)`.
fn parse_http_response(raw: &str) -> MonitorResult<(u16, String)> {
    // Status line: HTTP/1.1 <code> <reason>
    let status_line = raw.lines().next().unwrap_or("");
    let parts: Vec<&str> = status_line.splitn(3, ' ').collect();
    if parts.len() < 2 {
        return Err(MonitorError::Other(format!(
            "malformed HTTP status line: '{status_line}'"
        )));
    }
    let status: u16 = parts[1]
        .parse()
        .map_err(|_| MonitorError::Other(format!("invalid HTTP status code: '{}'", parts[1])))?;

    // Body follows the blank line separating headers from body.
    let body = if let Some(pos) = raw.find("\r\n\r\n") {
        raw[pos + 4..].to_string()
    } else if let Some(pos) = raw.find("\n\n") {
        raw[pos + 2..].to_string()
    } else {
        String::new()
    };

    Ok((status, body))
}

// ---------------------------------------------------------------------------
// Private: JSON helpers
// ---------------------------------------------------------------------------

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn json_str(s: &str) -> String {
    format!("\"{}\"", json_escape(s))
}

fn json_field(key: &str, value: &str) -> String {
    format!("{}: {}", json_str(key), json_str(value))
}

fn json_object(fields: &[String]) -> String {
    format!("{{{}}}", fields.join(","))
}

/// Extract a JSON string-field value from a flat JSON object.
///
/// Only handles simple `"field":"value"` patterns — sufficient for the
/// structured PagerDuty API responses.
fn extract_json_string_field(json: &str, field: &str) -> Option<String> {
    let needle = format!("\"{}\"", field);
    let pos = json.find(&needle)?;
    let after_key = &json[pos + needle.len()..];
    // Expect a colon then optional whitespace then opening quote.
    let colon_pos = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();
    if !after_colon.starts_with('"') {
        return None;
    }
    let inner = &after_colon[1..];
    let end = inner.find('"')?;
    Some(inner[..end].to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as IoWrite;
    use std::net::TcpListener;

    // ── PagerDutySeverity ──────────────────────────────────────────────────

    #[test]
    fn test_severity_as_str_critical() {
        assert_eq!(PagerDutySeverity::Critical.as_str(), "critical");
    }

    #[test]
    fn test_severity_as_str_error() {
        assert_eq!(PagerDutySeverity::Error.as_str(), "error");
    }

    #[test]
    fn test_severity_as_str_warning() {
        assert_eq!(PagerDutySeverity::Warning.as_str(), "warning");
    }

    #[test]
    fn test_severity_as_str_info() {
        assert_eq!(PagerDutySeverity::Info.as_str(), "info");
    }

    // ── PagerDutyEvent builder ─────────────────────────────────────────────

    #[test]
    fn test_event_builder_minimal() {
        let ev = PagerDutyEvent::new("Disk full", PagerDutySeverity::Critical);
        assert_eq!(ev.summary, "Disk full");
        assert_eq!(ev.severity, PagerDutySeverity::Critical);
        assert!(ev.component.is_none());
        assert!(ev.group.is_none());
        assert!(ev.details.is_empty());
        assert!(ev.dedup_key.is_none());
    }

    #[test]
    fn test_event_builder_with_all_fields() {
        let ev = PagerDutyEvent::new("CPU high", PagerDutySeverity::Warning)
            .with_component("cpu")
            .with_group("media-cluster")
            .with_detail("host", "enc-01")
            .with_dedup_key("cpu-alert-enc01");
        assert_eq!(ev.component.as_deref(), Some("cpu"));
        assert_eq!(ev.group.as_deref(), Some("media-cluster"));
        assert_eq!(ev.details.get("host").map(String::as_str), Some("enc-01"));
        assert_eq!(ev.dedup_key.as_deref(), Some("cpu-alert-enc01"));
    }

    // ── to_trigger_json ────────────────────────────────────────────────────

    #[test]
    fn test_trigger_json_contains_routing_key() {
        let ev = PagerDutyEvent::new("Test", PagerDutySeverity::Info);
        let json = ev.to_trigger_json("routing-key-abc");
        assert!(json.contains("routing-key-abc"), "json: {json}");
    }

    #[test]
    fn test_trigger_json_contains_summary() {
        let ev = PagerDutyEvent::new("High bitrate deviation", PagerDutySeverity::Warning);
        let json = ev.to_trigger_json("key");
        assert!(json.contains("High bitrate deviation"), "json: {json}");
    }

    #[test]
    fn test_trigger_json_contains_severity() {
        let ev = PagerDutyEvent::new("Alert", PagerDutySeverity::Error);
        let json = ev.to_trigger_json("key");
        assert!(json.contains("\"error\""), "json: {json}");
    }

    #[test]
    fn test_trigger_json_action_is_trigger() {
        let ev = PagerDutyEvent::new("Alert", PagerDutySeverity::Critical);
        let json = ev.to_trigger_json("key");
        assert!(json.contains("\"trigger\""), "json: {json}");
    }

    #[test]
    fn test_trigger_json_contains_dedup_key() {
        let ev =
            PagerDutyEvent::new("Alert", PagerDutySeverity::Critical).with_dedup_key("incident-42");
        let json = ev.to_trigger_json("key");
        assert!(json.contains("incident-42"), "json: {json}");
    }

    #[test]
    fn test_trigger_json_contains_component() {
        let ev =
            PagerDutyEvent::new("Alert", PagerDutySeverity::Warning).with_component("av1-encoder");
        let json = ev.to_trigger_json("key");
        assert!(json.contains("av1-encoder"), "json: {json}");
    }

    #[test]
    fn test_trigger_json_contains_detail() {
        let ev = PagerDutyEvent::new("Alert", PagerDutySeverity::Warning)
            .with_detail("datacenter", "eu-central-1");
        let json = ev.to_trigger_json("key");
        assert!(json.contains("eu-central-1"), "json: {json}");
    }

    // ── to_resolve_json ────────────────────────────────────────────────────

    #[test]
    fn test_resolve_json_action_is_resolve() {
        let json = PagerDutyEvent::to_resolve_json("dk-123", "routing-key");
        assert!(json.contains("\"resolve\""), "json: {json}");
    }

    #[test]
    fn test_resolve_json_contains_dedup_key() {
        let json = PagerDutyEvent::to_resolve_json("my-dedup-key", "rk");
        assert!(json.contains("my-dedup-key"), "json: {json}");
    }

    // ── PagerDutyClient ────────────────────────────────────────────────────

    #[test]
    fn test_client_default_endpoint() {
        let client = PagerDutyClient::new("key");
        assert!(
            client.endpoint.contains("pagerduty.com"),
            "endpoint: {}",
            client.endpoint
        );
    }

    // ── parse_url ──────────────────────────────────────────────────────────

    #[test]
    fn test_parse_url_http() {
        let p = parse_url("http://localhost/v2/enqueue").expect("valid");
        assert_eq!(p.scheme, "http");
        assert_eq!(p.host, "localhost");
        assert_eq!(p.port, 80);
        assert_eq!(p.path, "/v2/enqueue");
    }

    #[test]
    fn test_parse_url_with_port() {
        let p = parse_url("http://127.0.0.1:9090/path").expect("valid");
        assert_eq!(p.host, "127.0.0.1");
        assert_eq!(p.port, 9090);
        assert_eq!(p.path, "/path");
    }

    #[test]
    fn test_parse_url_https_default_port() {
        let p = parse_url("https://events.pagerduty.com/v2/enqueue").expect("valid");
        assert_eq!(p.scheme, "https");
        assert_eq!(p.host, "events.pagerduty.com");
        assert_eq!(p.port, 443);
    }

    // ── extract_json_string_field ──────────────────────────────────────────

    #[test]
    fn test_extract_json_string_field_found() {
        let json = r#"{"status":"success","dedup_key":"abc-123"}"#;
        let val = extract_json_string_field(json, "dedup_key");
        assert_eq!(val.as_deref(), Some("abc-123"));
    }

    #[test]
    fn test_extract_json_string_field_not_found() {
        let json = r#"{"status":"success"}"#;
        assert!(extract_json_string_field(json, "dedup_key").is_none());
    }

    // ── JSON helpers ───────────────────────────────────────────────────────

    #[test]
    fn test_json_escaping() {
        let s = json_str("say \"hello\"\nworld");
        assert!(s.contains("\\\""), "escaped quote: {s}");
        assert!(s.contains("\\n"), "escaped newline: {s}");
    }

    // ── Mock server: trigger ───────────────────────────────────────────────

    fn spawn_mock_server(response: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let port = listener.local_addr().expect("local_addr").port();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                // Drain the request.
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                // Send response.
                let _ = stream.write_all(response.as_bytes());
            }
        });
        port
    }

    #[test]
    fn test_client_trigger_via_mock_server() {
        let response = "HTTP/1.1 202 Accepted\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"status\":\"success\",\"message\":\"Event processed\",\"dedup_key\":\"mock-dedup-42\"}";
        let port = spawn_mock_server(response);

        let client = PagerDutyClient::new("test-routing-key")
            .with_endpoint(format!("http://127.0.0.1:{port}/v2/enqueue"))
            .with_timeout_ms(3000);

        let ev = PagerDutyEvent::new("CPU alert", PagerDutySeverity::Critical);
        let dedup_key = client.trigger(&ev).expect("trigger should succeed");
        assert_eq!(dedup_key, "mock-dedup-42");
    }

    #[test]
    fn test_client_resolve_via_mock_server() {
        let response = "HTTP/1.1 202 Accepted\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"status\":\"success\",\"message\":\"Event processed\"}";
        let port = spawn_mock_server(response);

        let client = PagerDutyClient::new("test-routing-key")
            .with_endpoint(format!("http://127.0.0.1:{port}/v2/enqueue"))
            .with_timeout_ms(3000);

        client
            .resolve("some-dedup-key")
            .expect("resolve should succeed");
    }

    #[test]
    fn test_client_trigger_server_error() {
        let response =
            "HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\n\r\nserver error";
        let port = spawn_mock_server(response);

        let client = PagerDutyClient::new("key")
            .with_endpoint(format!("http://127.0.0.1:{port}/v2/enqueue"))
            .with_timeout_ms(3000);

        let ev = PagerDutyEvent::new("Alert", PagerDutySeverity::Warning);
        assert!(client.trigger(&ev).is_err(), "expected Err on 500 response");
    }
}
