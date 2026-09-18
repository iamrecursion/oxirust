//! OpsGenie Alerts API integration — pure Rust, no external HTTP client.
//!
//! Sends create, close, and acknowledge requests to the OpsGenie Alerts API
//! using a hand-rolled HTTP/1.1 client built on [`std::net::TcpStream`].
//! No `reqwest`, `hyper`, or any other HTTP crate is used.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_monitor::opsgenie::{OpsGenieClient, OpsGenieAlert, OpsGeniePriority};
//!
//! let client = OpsGenieClient::new("my-api-key")
//!     .with_endpoint("http://localhost:8080/v2/alerts");
//!
//! let alert = OpsGenieAlert::new("High encoder queue depth")
//!     .with_priority(OpsGeniePriority::P2)
//!     .with_alias("enc-queue-high")
//!     .with_tag("service:media");
//!
//! // In real code: let request_id = client.create_alert(&alert)?;
//! ```

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use crate::error::{MonitorError, MonitorResult};

// ---------------------------------------------------------------------------
// OpsGeniePriority
// ---------------------------------------------------------------------------

/// OpsGenie alert priority level (P1 = highest, P5 = lowest).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpsGeniePriority {
    /// Critical — immediate response required.
    P1,
    /// High — urgent attention needed.
    P2,
    /// Moderate — default priority.
    P3,
    /// Low — informational.
    P4,
    /// Minimal — no escalation.
    P5,
}

impl OpsGeniePriority {
    /// The OpsGenie API string for this priority.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::P1 => "P1",
            Self::P2 => "P2",
            Self::P3 => "P3",
            Self::P4 => "P4",
            Self::P5 => "P5",
        }
    }
}

impl Default for OpsGeniePriority {
    fn default() -> Self {
        Self::P3
    }
}

// ---------------------------------------------------------------------------
// OpsGenieAlert
// ---------------------------------------------------------------------------

/// An OpsGenie Create Alert request payload.
#[derive(Debug, Clone)]
pub struct OpsGenieAlert {
    /// Short message shown in OpsGenie (required).
    pub message: String,
    /// Unique client-defined identifier used for close/acknowledge.
    pub alias: Option<String>,
    /// Longer description shown in the alert detail view.
    pub description: Option<String>,
    /// Alert priority.
    pub priority: OpsGeniePriority,
    /// Tags to attach to the alert.
    pub tags: Vec<String>,
    /// Arbitrary key-value details.
    pub details: BTreeMap<String, String>,
    /// Responder team names.
    pub responders: Vec<String>,
}

impl OpsGenieAlert {
    /// Create a minimal alert with the given `message`.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            alias: None,
            description: None,
            priority: OpsGeniePriority::default(),
            tags: Vec::new(),
            details: BTreeMap::new(),
            responders: Vec::new(),
        }
    }

    /// Set the alias (deduplication key).
    #[must_use]
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.alias = Some(alias.into());
        self
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the priority.
    #[must_use]
    pub fn with_priority(mut self, p: OpsGeniePriority) -> Self {
        self.priority = p;
        self
    }

    /// Add a tag.
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add a key-value detail entry.
    #[must_use]
    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.details.insert(key.into(), value.into());
        self
    }

    /// Add a responder team name.
    #[must_use]
    pub fn with_responder(mut self, responder: impl Into<String>) -> Self {
        self.responders.push(responder.into());
        self
    }

    /// Serialize this alert as an OpsGenie Create Alert API JSON body.
    #[must_use]
    pub fn to_create_json(&self) -> String {
        let mut fields = vec![
            json_field("message", &self.message),
            json_field("priority", self.priority.as_str()),
        ];

        if let Some(ref alias) = self.alias {
            fields.push(json_field("alias", alias));
        }
        if let Some(ref desc) = self.description {
            fields.push(json_field("description", desc));
        }

        // tags array
        let tags_arr: Vec<String> = self.tags.iter().map(|t| json_str(t)).collect();
        fields.push(format!("{}: [{}]", json_str("tags"), tags_arr.join(",")));

        // details object
        let detail_fields: Vec<String> =
            self.details.iter().map(|(k, v)| json_field(k, v)).collect();
        fields.push(format!(
            "{}: {{{}}}",
            json_str("details"),
            detail_fields.join(",")
        ));

        // responders array (each as {"name":"...","type":"team"})
        let resp_arr: Vec<String> = self
            .responders
            .iter()
            .map(|r| {
                format!(
                    "{{{}}}",
                    vec![json_field("name", r), json_field("type", "team")].join(",")
                )
            })
            .collect();
        fields.push(format!(
            "{}: [{}]",
            json_str("responders"),
            resp_arr.join(",")
        ));

        json_object(&fields)
    }

    /// Serialize a close/acknowledge action JSON body with an optional `note`.
    #[must_use]
    pub fn close_json(note: &str) -> String {
        json_object(&[json_field("note", note)])
    }
}

// ---------------------------------------------------------------------------
// OpsGenieClient
// ---------------------------------------------------------------------------

/// OpsGenie Alerts API client.
///
/// Uses [`TcpStream`] for plain-HTTP connections.  For production HTTPS
/// endpoints, terminate TLS externally (reverse proxy / stunnel) or point
/// at a local test server.
#[derive(Debug, Clone)]
pub struct OpsGenieClient {
    /// OpsGenie API key (sent as `Authorization: GenieKey <api_key>`).
    pub api_key: String,
    /// OpsGenie Alerts API endpoint URL.
    pub endpoint: String,
    /// TCP connect + read timeout in milliseconds.
    pub timeout_ms: u64,
}

impl OpsGenieClient {
    /// Create a client targeting the EU OpsGenie Alerts API endpoint.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            endpoint: "https://api.eu.opsgenie.com/v2/alerts".to_string(),
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

    /// Create an OpsGenie alert.
    ///
    /// Returns the `requestId` from the OpsGenie response on success.
    ///
    /// # Errors
    ///
    /// Returns an error if the TCP connection fails, URL parsing fails, or
    /// the server returns a non-2xx status.
    pub fn create_alert(&self, alert: &OpsGenieAlert) -> MonitorResult<String> {
        let body = alert.to_create_json();
        let parsed = parse_url(&self.endpoint)?;
        let auth = format!("GenieKey {}", self.api_key);
        let (status, response_body) = send_http_post(
            &parsed.host,
            parsed.port,
            &parsed.path,
            &auth,
            &body,
            self.timeout_ms,
        )?;

        if !(200..300).contains(&status) {
            return Err(MonitorError::Other(format!(
                "OpsGenie create_alert failed: HTTP {status} — {response_body}"
            )));
        }

        let request_id = extract_json_string_field(&response_body, "requestId")
            .unwrap_or_else(|| "ok".to_string());
        Ok(request_id)
    }

    /// Close an alert identified by its `alias`.
    ///
    /// Posts to `{endpoint}/{alias}/close`.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection fails or the server returns
    /// non-2xx.
    pub fn close_alert(&self, alias: &str, note: &str) -> MonitorResult<()> {
        let body = OpsGenieAlert::close_json(note);
        let url = format!("{}/{}/close", self.endpoint, url_encode(alias));
        let parsed = parse_url(&url)?;
        let auth = format!("GenieKey {}", self.api_key);
        let (status, response_body) = send_http_post(
            &parsed.host,
            parsed.port,
            &parsed.path,
            &auth,
            &body,
            self.timeout_ms,
        )?;

        if !(200..300).contains(&status) {
            return Err(MonitorError::Other(format!(
                "OpsGenie close_alert failed: HTTP {status} — {response_body}"
            )));
        }
        Ok(())
    }

    /// Acknowledge an alert identified by its `alias`.
    ///
    /// Posts to `{endpoint}/{alias}/acknowledge`.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection fails or the server returns
    /// non-2xx.
    pub fn acknowledge_alert(&self, alias: &str, note: &str) -> MonitorResult<()> {
        let body = OpsGenieAlert::close_json(note);
        let url = format!("{}/{}/acknowledge", self.endpoint, url_encode(alias));
        let parsed = parse_url(&url)?;
        let auth = format!("GenieKey {}", self.api_key);
        let (status, response_body) = send_http_post(
            &parsed.host,
            parsed.port,
            &parsed.path,
            &auth,
            &body,
            self.timeout_ms,
        )?;

        if !(200..300).contains(&status) {
            return Err(MonitorError::Other(format!(
                "OpsGenie acknowledge_alert failed: HTTP {status} — {response_body}"
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Private: minimal percent-encoding for alias in URL path
// ---------------------------------------------------------------------------

fn url_encode(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            ' ' => "%20".to_string(),
            other => {
                let mut buf = [0u8; 4];
                let encoded = other.encode_utf8(&mut buf);
                encoded
                    .as_bytes()
                    .iter()
                    .map(|b| format!("%{b:02X}"))
                    .collect()
            }
        })
        .collect()
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

    let (authority, path) = if let Some(slash_pos) = rest.find('/') {
        (&rest[..slash_pos], rest[slash_pos..].to_string())
    } else {
        (rest, "/".to_string())
    };

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

fn send_http_post(
    host: &str,
    port: u16,
    path: &str,
    auth_value: &str,
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
         Authorization: {auth_value}\r\n\
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

fn parse_http_response(raw: &str) -> MonitorResult<(u16, String)> {
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

fn extract_json_string_field(json: &str, field: &str) -> Option<String> {
    let needle = format!("\"{}\"", field);
    let pos = json.find(&needle)?;
    let after_key = &json[pos + needle.len()..];
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

    // ── OpsGeniePriority ───────────────────────────────────────────────────

    #[test]
    fn test_priority_as_str_all_variants() {
        assert_eq!(OpsGeniePriority::P1.as_str(), "P1");
        assert_eq!(OpsGeniePriority::P2.as_str(), "P2");
        assert_eq!(OpsGeniePriority::P3.as_str(), "P3");
        assert_eq!(OpsGeniePriority::P4.as_str(), "P4");
        assert_eq!(OpsGeniePriority::P5.as_str(), "P5");
    }

    #[test]
    fn test_priority_default_is_p3() {
        assert_eq!(OpsGeniePriority::default(), OpsGeniePriority::P3);
    }

    // ── OpsGenieAlert builder ──────────────────────────────────────────────

    #[test]
    fn test_alert_builder_minimal() {
        let alert = OpsGenieAlert::new("Test alert");
        assert_eq!(alert.message, "Test alert");
        assert!(alert.alias.is_none());
        assert!(alert.description.is_none());
        assert_eq!(alert.priority, OpsGeniePriority::P3);
        assert!(alert.tags.is_empty());
        assert!(alert.details.is_empty());
        assert!(alert.responders.is_empty());
    }

    #[test]
    fn test_alert_builder_all_fields() {
        let alert = OpsGenieAlert::new("Full alert")
            .with_alias("full-alert-alias")
            .with_description("Detailed description")
            .with_priority(OpsGeniePriority::P1)
            .with_tag("service:media")
            .with_detail("host", "enc-01")
            .with_responder("media-sre");
        assert_eq!(alert.alias.as_deref(), Some("full-alert-alias"));
        assert_eq!(alert.description.as_deref(), Some("Detailed description"));
        assert_eq!(alert.priority, OpsGeniePriority::P1);
        assert_eq!(alert.tags, vec!["service:media"]);
        assert_eq!(
            alert.details.get("host").map(String::as_str),
            Some("enc-01")
        );
        assert_eq!(alert.responders, vec!["media-sre"]);
    }

    // ── to_create_json ─────────────────────────────────────────────────────

    #[test]
    fn test_alert_to_create_json_contains_message() {
        let alert = OpsGenieAlert::new("Disk near capacity");
        let json = alert.to_create_json();
        assert!(json.contains("Disk near capacity"), "json: {json}");
    }

    #[test]
    fn test_alert_to_create_json_contains_priority() {
        let alert = OpsGenieAlert::new("Alert").with_priority(OpsGeniePriority::P1);
        let json = alert.to_create_json();
        assert!(json.contains("\"P1\""), "json: {json}");
    }

    #[test]
    fn test_alert_to_create_json_contains_alias() {
        let alert = OpsGenieAlert::new("Alert").with_alias("my-alias-key");
        let json = alert.to_create_json();
        assert!(json.contains("my-alias-key"), "json: {json}");
    }

    #[test]
    fn test_alert_to_create_json_contains_description() {
        let alert = OpsGenieAlert::new("Alert")
            .with_description("Buffer underrun detected on channel HD-42");
        let json = alert.to_create_json();
        assert!(json.contains("Buffer underrun detected"), "json: {json}");
    }

    #[test]
    fn test_alert_to_create_json_contains_tags() {
        let alert = OpsGenieAlert::new("Alert")
            .with_tag("service:live-stream")
            .with_tag("env:production");
        let json = alert.to_create_json();
        assert!(json.contains("service:live-stream"), "json: {json}");
        assert!(json.contains("env:production"), "json: {json}");
    }

    #[test]
    fn test_alert_to_create_json_contains_details() {
        let alert = OpsGenieAlert::new("Alert").with_detail("datacenter", "eu-central-1");
        let json = alert.to_create_json();
        assert!(json.contains("eu-central-1"), "json: {json}");
    }

    #[test]
    fn test_alert_to_create_json_contains_responders() {
        let alert = OpsGenieAlert::new("Alert").with_responder("media-ops");
        let json = alert.to_create_json();
        assert!(json.contains("media-ops"), "json: {json}");
        assert!(json.contains("\"team\""), "json: {json}");
    }

    // ── close_json ─────────────────────────────────────────────────────────

    #[test]
    fn test_alert_close_json_contains_note() {
        let json = OpsGenieAlert::close_json("Resolved by automation");
        assert!(json.contains("Resolved by automation"), "json: {json}");
    }

    // ── OpsGenieClient ─────────────────────────────────────────────────────

    #[test]
    fn test_client_default_endpoint_contains_opsgenie() {
        let client = OpsGenieClient::new("key");
        assert!(
            client.endpoint.contains("opsgenie.com"),
            "endpoint: {}",
            client.endpoint
        );
    }

    // ── parse_url ──────────────────────────────────────────────────────────

    #[test]
    fn test_parse_url_http_localhost() {
        let p = parse_url("http://localhost/v2/alerts").expect("valid");
        assert_eq!(p.host, "localhost");
        assert_eq!(p.port, 80);
        assert_eq!(p.path, "/v2/alerts");
    }

    #[test]
    fn test_parse_url_custom_port() {
        let p = parse_url("http://127.0.0.1:9191/v2/alerts").expect("valid");
        assert_eq!(p.host, "127.0.0.1");
        assert_eq!(p.port, 9191);
    }

    // ── JSON helpers ───────────────────────────────────────────────────────

    #[test]
    fn test_json_string_escaping() {
        let s = json_str("value with \"quotes\" and\nnewline");
        assert!(s.contains("\\\""), "json: {s}");
        assert!(s.contains("\\n"), "json: {s}");
    }

    // ── Mock server helpers ────────────────────────────────────────────────

    fn spawn_mock_server_once(response: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let port = listener.local_addr().expect("local_addr").port();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 8192];
                let _ = stream.read(&mut buf);
                let _ = stream.write_all(response.as_bytes());
            }
        });
        port
    }

    // ── Mock server integration tests ──────────────────────────────────────

    #[test]
    fn test_client_create_alert_via_mock_server() {
        let response = "HTTP/1.1 202 Accepted\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"result\":\"Request will be processed\",\"took\":0.1,\"requestId\":\"mock-req-99\"}";
        let port = spawn_mock_server_once(response);

        let client = OpsGenieClient::new("test-api-key")
            .with_endpoint(format!("http://127.0.0.1:{port}/v2/alerts"))
            .with_timeout_ms(3000);

        let alert = OpsGenieAlert::new("Test alert").with_priority(OpsGeniePriority::P2);
        let request_id = client.create_alert(&alert).expect("create should succeed");
        assert_eq!(request_id, "mock-req-99");
    }

    #[test]
    fn test_client_close_alert_via_mock_server() {
        let response = "HTTP/1.1 202 Accepted\r\nConnection: close\r\n\r\n{\"result\":\"Request will be processed\"}";
        let port = spawn_mock_server_once(response);

        let client = OpsGenieClient::new("test-api-key")
            .with_endpoint(format!("http://127.0.0.1:{port}/v2/alerts"))
            .with_timeout_ms(3000);

        client
            .close_alert("my-alias", "Auto-resolved")
            .expect("close should succeed");
    }

    #[test]
    fn test_client_acknowledge_alert_via_mock_server() {
        let response = "HTTP/1.1 202 Accepted\r\nConnection: close\r\n\r\n{\"result\":\"Request will be processed\"}";
        let port = spawn_mock_server_once(response);

        let client = OpsGenieClient::new("test-api-key")
            .with_endpoint(format!("http://127.0.0.1:{port}/v2/alerts"))
            .with_timeout_ms(3000);

        client
            .acknowledge_alert("my-alias", "Acknowledged")
            .expect("acknowledge should succeed");
    }

    #[test]
    fn test_client_server_error_returns_err() {
        let response =
            "HTTP/1.1 422 Unprocessable Entity\r\nConnection: close\r\n\r\n{\"message\":\"invalid\"}";
        let port = spawn_mock_server_once(response);

        let client = OpsGenieClient::new("key")
            .with_endpoint(format!("http://127.0.0.1:{port}/v2/alerts"))
            .with_timeout_ms(3000);

        let alert = OpsGenieAlert::new("Bad alert");
        assert!(
            client.create_alert(&alert).is_err(),
            "expected Err on 422 response"
        );
    }
}
