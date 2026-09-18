//! RTSP 1.0 wire-format messages (RFC 2326).
//!
//! RTSP looks superficially like HTTP/1.1 but is a distinct protocol:
//! - request lines are `METHOD rtsp://uri RTSP/1.0`
//! - responses are `RTSP/1.0 STATUS REASON`
//! - headers are CRLF-terminated key: value pairs, header block terminated by blank CRLF
//! - an optional message body of length `Content-Length` follows
//! - a `CSeq` header is mandatory on every request/response and increments per request

use std::collections::BTreeMap;
use std::fmt;

use crate::error::NetError;

/// RTSP protocol version handled by this implementation.
pub const RTSP_VERSION: &str = "RTSP/1.0";

/// Upper bound on an RTSP message body (`Content-Length`).
///
/// RTSP bodies are normally small (SDP descriptions, parameter lists). This
/// cap rejects a hostile peer that advertises an enormous `Content-Length`
/// (e.g. `18446744073709551615`) purely to force an integer overflow or an
/// unbounded allocation in the parser. 16 MiB is far larger than any
/// legitimate RTSP payload.
pub const MAX_RTSP_BODY_LEN: usize = 16 * 1024 * 1024;

/// RTSP request method (RFC 2326 §10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Method {
    /// Discover server capabilities.
    Options,
    /// Retrieve a media description (typically SDP).
    Describe,
    /// Upload a media description.
    Announce,
    /// Establish a transport for a single stream.
    Setup,
    /// Start delivery of a previously set-up stream.
    Play,
    /// Pause delivery without releasing resources.
    Pause,
    /// Release the session and all transports.
    Teardown,
    /// Retrieve parameters by name.
    GetParameter,
    /// Set parameters.
    SetParameter,
    /// Reposition a recording stream.
    Redirect,
    /// Begin recording.
    Record,
}

impl Method {
    /// Wire form of the method name.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::Method;
    /// assert_eq!(Method::Describe.as_str(), "DESCRIBE");
    /// assert_eq!(Method::GetParameter.as_str(), "GET_PARAMETER");
    /// ```
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Options => "OPTIONS",
            Self::Describe => "DESCRIBE",
            Self::Announce => "ANNOUNCE",
            Self::Setup => "SETUP",
            Self::Play => "PLAY",
            Self::Pause => "PAUSE",
            Self::Teardown => "TEARDOWN",
            Self::GetParameter => "GET_PARAMETER",
            Self::SetParameter => "SET_PARAMETER",
            Self::Redirect => "REDIRECT",
            Self::Record => "RECORD",
        }
    }

    /// Parse a method token; case-sensitive per RFC 2326.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::Protocol`] if `s` is not one of the RFC 2326
    /// method names.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::Method;
    /// assert_eq!(Method::parse("SETUP").unwrap(), Method::Setup);
    /// assert!(Method::parse("frobnicate").is_err());
    /// ```
    pub fn parse(s: &str) -> Result<Self, NetError> {
        Ok(match s {
            "OPTIONS" => Self::Options,
            "DESCRIBE" => Self::Describe,
            "ANNOUNCE" => Self::Announce,
            "SETUP" => Self::Setup,
            "PLAY" => Self::Play,
            "PAUSE" => Self::Pause,
            "TEARDOWN" => Self::Teardown,
            "GET_PARAMETER" => Self::GetParameter,
            "SET_PARAMETER" => Self::SetParameter,
            "REDIRECT" => Self::Redirect,
            "RECORD" => Self::Record,
            other => {
                return Err(NetError::Protocol(format!("unknown RTSP method: {other}")));
            }
        })
    }
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Case-insensitive ASCII-lowercased header name used as the storage key.
///
/// RTSP header names are case-insensitive on the wire (RFC 2326 §12);
/// internally we always store the lowercase form so lookups don't need a
/// case-insensitive map.
#[derive(Debug, Clone, Default)]
pub struct Headers {
    inner: BTreeMap<String, String>,
}

impl Headers {
    /// Empty header set.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::message::Headers;
    /// let h = Headers::new();
    /// assert!(h.is_empty());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (overwrite) a header. The name is canonicalized to lowercase
    /// for internal storage; lookups via [`get`](Self::get) are
    /// case-insensitive.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::message::Headers;
    /// let mut h = Headers::new();
    /// h.insert("CSeq", "7");
    /// assert_eq!(h.get("cseq"), Some("7"));
    /// ```
    pub fn insert(&mut self, name: &str, value: impl Into<String>) {
        self.inner.insert(name.to_ascii_lowercase(), value.into());
    }

    /// Get a header value, case-insensitively.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::message::Headers;
    /// let mut h = Headers::new();
    /// h.insert("Session", "ABCD1234;timeout=30");
    /// assert_eq!(h.get("session"), Some("ABCD1234;timeout=30"));
    /// assert!(h.get("nonexistent").is_none());
    /// ```
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&str> {
        self.inner
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }

    /// Number of headers set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True if no headers are set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterate over `(lowercase-name, value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.inner.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

/// An RTSP request constructed by the client.
#[derive(Debug, Clone)]
pub struct Request {
    /// Method to invoke.
    pub method: Method,
    /// Target URI (full `rtsp://host[:port]/path` or `*` for OPTIONS).
    pub uri: String,
    /// Header block.
    pub headers: Headers,
    /// Optional message body (used by ANNOUNCE, SET_PARAMETER, etc.).
    pub body: Vec<u8>,
}

impl Request {
    /// Construct a bare request with the mandatory CSeq header.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::{Method, Request};
    /// let req = Request::new(Method::Options, "rtsp://cam/stream", 1);
    /// assert_eq!(req.uri, "rtsp://cam/stream");
    /// assert_eq!(req.headers.get("cseq"), Some("1"));
    /// ```
    #[must_use]
    pub fn new(method: Method, uri: impl Into<String>, cseq: u32) -> Self {
        let mut headers = Headers::new();
        headers.insert("CSeq", cseq.to_string());
        Self {
            method,
            uri: uri.into(),
            headers,
            body: Vec::new(),
        }
    }

    /// Builder helper: set a header.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::{Method, Request};
    /// let req = Request::new(Method::Describe, "rtsp://x/", 1)
    ///     .with_header("Accept", "application/sdp");
    /// assert_eq!(req.headers.get("Accept"), Some("application/sdp"));
    /// ```
    #[must_use]
    pub fn with_header(mut self, name: &str, value: impl Into<String>) -> Self {
        self.headers.insert(name, value);
        self
    }

    /// Builder helper: attach a message body and the matching `Content-Length`.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::{Method, Request};
    /// let req = Request::new(Method::Announce, "rtsp://x/", 1)
    ///     .with_body(b"v=0\r\n".to_vec());
    /// assert_eq!(req.headers.get("Content-Length"), Some("5"));
    /// ```
    #[must_use]
    pub fn with_body(mut self, body: Vec<u8>) -> Self {
        self.headers
            .insert("Content-Length", body.len().to_string());
        self.body = body;
        self
    }

    /// Serialize to the RTSP wire format.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::{Method, Request};
    /// let wire = Request::new(Method::Options, "rtsp://cam", 1).encode();
    /// let text = std::str::from_utf8(&wire).unwrap();
    /// assert!(text.starts_with("OPTIONS rtsp://cam RTSP/1.0\r\n"));
    /// assert!(text.ends_with("\r\n\r\n"));
    /// ```
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(256 + self.body.len());
        out.extend_from_slice(self.method.as_str().as_bytes());
        out.push(b' ');
        out.extend_from_slice(self.uri.as_bytes());
        out.push(b' ');
        out.extend_from_slice(RTSP_VERSION.as_bytes());
        out.extend_from_slice(b"\r\n");
        for (name, value) in self.headers.iter() {
            // Render with the original "Title-Case" convention. We re-title
            // because we lowercased on insert; this matches what most RTSP
            // servers send and what wireshark expects in captures.
            let mut chars = name.chars();
            let mut titled = String::with_capacity(name.len());
            let mut upper_next = true;
            while let Some(c) = chars.next() {
                if upper_next {
                    titled.extend(c.to_uppercase());
                    upper_next = false;
                } else {
                    titled.push(c);
                }
                if c == '-' {
                    upper_next = true;
                }
            }
            out.extend_from_slice(titled.as_bytes());
            out.extend_from_slice(b": ");
            out.extend_from_slice(value.as_bytes());
            out.extend_from_slice(b"\r\n");
        }
        out.extend_from_slice(b"\r\n");
        out.extend_from_slice(&self.body);
        out
    }
}

/// An RTSP response received from the server.
#[derive(Debug, Clone)]
pub struct Response {
    /// 3-digit status code (RFC 2326 §11).
    pub status: u16,
    /// Human-readable reason phrase.
    pub reason: String,
    /// Header block (lowercased keys).
    pub headers: Headers,
    /// Body bytes (length matches `Content-Length`).
    pub body: Vec<u8>,
}

impl Response {
    /// True for any 2xx status.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::{message::{Headers, Response}, Method};
    /// let resp = Response {
    ///     status: 200,
    ///     reason: "OK".into(),
    ///     headers: Headers::new(),
    ///     body: vec![],
    /// };
    /// assert!(resp.is_success());
    /// ```
    #[must_use]
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// True for 401 Unauthorized.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::message::{Headers, Response};
    /// let resp = Response {
    ///     status: 401,
    ///     reason: "Unauthorized".into(),
    ///     headers: Headers::new(),
    ///     body: vec![],
    /// };
    /// assert!(resp.is_unauthorized());
    /// ```
    #[must_use]
    pub fn is_unauthorized(&self) -> bool {
        self.status == 401
    }

    /// Convert non-2xx status into a [`NetError::Http`].
    pub fn into_http_error(self) -> NetError {
        NetError::Http {
            status: self.status,
            message: self.reason,
        }
    }
}

/// Outcome of attempting to parse an RTSP response out of a byte buffer.
///
/// RTSP messages can be split across reads — `NeedMore` tells the caller to
/// read more bytes and retry. `Parsed { consumed, response }` reports how
/// many bytes were consumed so the caller can drain its buffer.
#[derive(Debug)]
pub enum ParseStatus {
    /// More bytes needed before a complete message is available.
    NeedMore,
    /// A complete response was parsed, consuming this many bytes.
    Parsed {
        /// Number of bytes consumed from the input.
        consumed: usize,
        /// The parsed response.
        response: Response,
    },
}

/// Outcome of attempting to parse an RTSP request out of a byte buffer.
///
/// Mirror of [`ParseStatus`] but for server-side request parsing.
#[derive(Debug)]
pub enum RequestParseStatus {
    /// More bytes needed before a complete request is available.
    NeedMore,
    /// A complete request was parsed, consuming this many bytes.
    Complete {
        /// Number of bytes consumed from the input.
        consumed: usize,
        /// The parsed request.
        request: Request,
    },
}

/// Parse error returned when a message is structurally invalid.
pub type ParseError = NetError;

/// Try to parse a single RTSP response from `buf`.
///
/// Returns `NeedMore` when the header block or body is not yet complete.
/// Returns a `Protocol` error on a malformed status line.
///
/// # Errors
///
/// Returns [`NetError::Protocol`] if `buf` contains an invalid status
/// line (e.g. wrong protocol version) or malformed headers, and
/// [`NetError::Parse`] on a bad `Content-Length` value.
///
/// # Example
///
/// ```
/// use oximedia_net::rtsp::message::{try_parse_response, ParseStatus};
///
/// let wire = b"RTSP/1.0 200 OK\r\nCSeq: 3\r\n\r\n";
/// match try_parse_response(wire).unwrap() {
///     ParseStatus::Parsed { consumed, response } => {
///         assert_eq!(consumed, wire.len());
///         assert_eq!(response.status, 200);
///     }
///     ParseStatus::NeedMore => panic!("complete buffer should parse"),
/// }
/// ```
pub fn try_parse_response(buf: &[u8]) -> Result<ParseStatus, NetError> {
    // Find end of header block (CRLF CRLF).
    let header_end = match find_double_crlf(buf) {
        Some(pos) => pos,
        None => return Ok(ParseStatus::NeedMore),
    };

    let header_bytes = &buf[..header_end];
    let text = std::str::from_utf8(header_bytes)
        .map_err(|e| NetError::Protocol(format!("invalid UTF-8 in headers: {e}")))?;

    let mut lines = text.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| NetError::Protocol("empty RTSP response".into()))?;
    let (status, reason) = parse_status_line(status_line)?;

    let mut headers = Headers::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| NetError::Protocol(format!("malformed header line: {line:?}")))?;
        headers.insert(name.trim(), value.trim().to_string());
    }

    let body_start = header_end + 4; // skip the trailing CRLFCRLF
    let content_length: usize = headers
        .get("Content-Length")
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| NetError::Protocol(format!("bad Content-Length: {e}")))
        })
        .transpose()?
        .unwrap_or(0);

    // Reject an absurd Content-Length before any pointer arithmetic: a peer
    // sending `Content-Length: 18446744073709551615` would otherwise make
    // `body_start + content_length` wrap and yield a start>end slice range.
    if content_length > MAX_RTSP_BODY_LEN {
        return Err(NetError::Protocol(format!(
            "RTSP Content-Length {content_length} exceeds maximum {MAX_RTSP_BODY_LEN}"
        )));
    }
    // Checked add so a length near usize::MAX cannot overflow the end offset.
    let body_end = body_start
        .checked_add(content_length)
        .ok_or_else(|| NetError::Protocol("RTSP response body length overflow".into()))?;

    if buf.len() < body_end {
        return Ok(ParseStatus::NeedMore);
    }

    let body = buf[body_start..body_end].to_vec();
    Ok(ParseStatus::Parsed {
        consumed: body_end,
        response: Response {
            status,
            reason: reason.to_string(),
            headers,
            body,
        },
    })
}

/// Try to parse a single RTSP request from `buf`.
///
/// Returns `NeedMore` when the header block or body is not yet complete.
/// Returns a `Protocol` error on a malformed request line.
///
/// # Errors
///
/// Returns [`NetError::Protocol`] if `buf` contains an invalid request line
/// (e.g. wrong protocol version) or malformed headers, and [`NetError::Parse`]
/// on a bad `Content-Length` value.
///
/// # Example
///
/// ```
/// use oximedia_net::rtsp::message::{try_parse_request, RequestParseStatus};
///
/// let wire = b"OPTIONS rtsp://cam/ RTSP/1.0\r\nCSeq: 1\r\n\r\n";
/// match try_parse_request(wire).unwrap() {
///     RequestParseStatus::Complete { consumed, request } => {
///         assert_eq!(consumed, wire.len());
///         use oximedia_net::rtsp::Method;
///         assert_eq!(request.method, Method::Options);
///     }
///     RequestParseStatus::NeedMore => panic!("complete buffer should parse"),
/// }
/// ```
pub fn try_parse_request(buf: &[u8]) -> Result<RequestParseStatus, ParseError> {
    let header_end = match find_double_crlf(buf) {
        Some(pos) => pos,
        None => return Ok(RequestParseStatus::NeedMore),
    };

    let header_bytes = &buf[..header_end];
    let text = std::str::from_utf8(header_bytes)
        .map_err(|e| NetError::Protocol(format!("invalid UTF-8 in request headers: {e}")))?;

    let mut lines = text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| NetError::Protocol("empty RTSP request".into()))?;

    let (method, uri) = parse_request_line(request_line)?;

    let mut headers = Headers::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| NetError::Protocol(format!("malformed header line: {line:?}")))?;
        headers.insert(name.trim(), value.trim().to_string());
    }

    let body_start = header_end + 4; // skip the trailing CRLFCRLF
    let content_length: usize = headers
        .get("Content-Length")
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| NetError::Protocol(format!("bad Content-Length: {e}")))
        })
        .transpose()?
        .unwrap_or(0);

    // Reject an absurd Content-Length before any pointer arithmetic: a peer
    // sending `Content-Length: 18446744073709551615` would otherwise make
    // `body_start + content_length` wrap and yield a start>end slice range.
    if content_length > MAX_RTSP_BODY_LEN {
        return Err(NetError::Protocol(format!(
            "RTSP Content-Length {content_length} exceeds maximum {MAX_RTSP_BODY_LEN}"
        )));
    }
    // Checked add so a length near usize::MAX cannot overflow the end offset.
    let body_end = body_start
        .checked_add(content_length)
        .ok_or_else(|| NetError::Protocol("RTSP request body length overflow".into()))?;

    if buf.len() < body_end {
        return Ok(RequestParseStatus::NeedMore);
    }

    let body = buf[body_start..body_end].to_vec();
    Ok(RequestParseStatus::Complete {
        consumed: body_end,
        request: Request {
            method,
            uri,
            headers,
            body,
        },
    })
}

/// Parse a request line: `METHOD SP Request-URI SP RTSP/1.0`.
fn parse_request_line(line: &str) -> Result<(Method, String), NetError> {
    let mut parts = line.splitn(3, ' ');
    let method_str = parts
        .next()
        .ok_or_else(|| NetError::Protocol("missing method in request line".into()))?;
    let uri = parts
        .next()
        .ok_or_else(|| NetError::Protocol("missing URI in request line".into()))?
        .to_string();
    let version = parts
        .next()
        .ok_or_else(|| NetError::Protocol("missing RTSP version in request line".into()))?;
    if !version.starts_with("RTSP/") {
        return Err(NetError::Protocol(format!(
            "expected RTSP version, got {version:?}"
        )));
    }
    let method = Method::parse(method_str)?;
    Ok((method, uri))
}

impl Response {
    /// Encode this response to bytes ready for sending over a TCP connection.
    ///
    /// Format: `"RTSP/1.0 {status} {reason}\r\n"` + headers + `"\r\n"` + body.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_net::rtsp::message::{Headers, Response};
    /// let mut headers = Headers::new();
    /// headers.insert("CSeq", "1");
    /// let resp = Response {
    ///     status: 200,
    ///     reason: "OK".into(),
    ///     headers,
    ///     body: vec![],
    /// };
    /// let wire = resp.encode();
    /// let text = std::str::from_utf8(&wire).unwrap();
    /// assert!(text.starts_with("RTSP/1.0 200 OK\r\n"));
    /// assert!(text.contains("Cseq: 1\r\n"));
    /// assert!(text.ends_with("\r\n\r\n"));
    /// ```
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let reason = status_reason(self.status);
        let mut out = Vec::with_capacity(256 + self.body.len());
        out.extend_from_slice(RTSP_VERSION.as_bytes());
        out.push(b' ');
        out.extend_from_slice(self.status.to_string().as_bytes());
        out.push(b' ');
        out.extend_from_slice(reason.as_bytes());
        out.extend_from_slice(b"\r\n");

        for (name, value) in self.headers.iter() {
            // Title-case the header name to match wire convention
            let mut titled = String::with_capacity(name.len());
            let mut upper_next = true;
            for c in name.chars() {
                if upper_next {
                    titled.extend(c.to_uppercase());
                    upper_next = false;
                } else {
                    titled.push(c);
                }
                if c == '-' {
                    upper_next = true;
                }
            }
            out.extend_from_slice(titled.as_bytes());
            out.extend_from_slice(b": ");
            out.extend_from_slice(value.as_bytes());
            out.extend_from_slice(b"\r\n");
        }
        out.extend_from_slice(b"\r\n");
        out.extend_from_slice(&self.body);
        out
    }

    /// Build a minimal response with status, reason, and the echoed CSeq.
    #[must_use]
    pub fn build(status: u16, cseq: u32) -> Self {
        let mut headers = Headers::new();
        headers.insert("CSeq", cseq.to_string());
        Self {
            status,
            reason: status_reason(status).to_string(),
            headers,
            body: Vec::new(),
        }
    }
}

/// Standard reason phrase for well-known RTSP status codes (RFC 2326 §11.1).
fn status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        250 => "Low on Storage Space",
        300 => "Multiple Choices",
        301 => "Moved Permanently",
        302 => "Moved Temporarily",
        303 => "See Other",
        304 => "Not Modified",
        305 => "Use Proxy",
        400 => "Bad Request",
        401 => "Unauthorized",
        402 => "Payment Required",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        406 => "Not Acceptable",
        407 => "Proxy Authentication Required",
        408 => "Request Timeout",
        410 => "Gone",
        411 => "Length Required",
        412 => "Precondition Failed",
        413 => "Request Entity Too Large",
        414 => "Request-URI Too Long",
        415 => "Unsupported Media Type",
        451 => "Parameter Not Understood",
        452 => "Conference Not Found",
        453 => "Not Enough Bandwidth",
        454 => "Session Not Found",
        455 => "Method Not Valid in This State",
        456 => "Header Field Not Valid for Resource",
        457 => "Invalid Range",
        458 => "Parameter Is Read-Only",
        459 => "Aggregate Operation Not Allowed",
        460 => "Only Aggregate Operation Allowed",
        461 => "Unsupported Transport",
        462 => "Destination Unreachable",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        505 => "RTSP Version Not Supported",
        551 => "Option Not Supported",
        _ => "Unknown",
    }
}

fn find_double_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn parse_status_line(line: &str) -> Result<(u16, &str), NetError> {
    // Expected: "RTSP/1.0 200 OK"
    let mut parts = line.splitn(3, ' ');
    let version = parts
        .next()
        .ok_or_else(|| NetError::Protocol("missing RTSP version".into()))?;
    if !version.starts_with("RTSP/") {
        return Err(NetError::Protocol(format!(
            "expected RTSP version, got {version:?}"
        )));
    }
    let status = parts
        .next()
        .ok_or_else(|| NetError::Protocol("missing status code".into()))?
        .parse::<u16>()
        .map_err(|e| NetError::Protocol(format!("bad status code: {e}")))?;
    let reason = parts.next().unwrap_or("");
    Ok((status, reason))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn method_roundtrip() {
        for m in [
            Method::Options,
            Method::Describe,
            Method::Setup,
            Method::Play,
            Method::Pause,
            Method::Teardown,
        ] {
            assert_eq!(Method::parse(m.as_str()).unwrap(), m);
        }
    }

    #[test]
    fn unknown_method_errors() {
        assert!(Method::parse("FROBNICATE").is_err());
    }

    #[test]
    fn encode_options_request() {
        let req = Request::new(Method::Options, "rtsp://camera.local/stream", 1);
        let wire = req.encode();
        let s = std::str::from_utf8(&wire).unwrap();
        assert!(s.starts_with("OPTIONS rtsp://camera.local/stream RTSP/1.0\r\n"));
        assert!(s.contains("Cseq: 1\r\n"));
        assert!(s.ends_with("\r\n\r\n"));
    }

    #[test]
    fn encode_request_with_body_sets_content_length() {
        let body = b"v=0\r\n".to_vec();
        let req = Request::new(Method::Announce, "rtsp://x/y", 2)
            .with_header("Content-Type", "application/sdp")
            .with_body(body.clone());
        let s = String::from_utf8(req.encode()).unwrap();
        assert!(s.contains("Content-Length: 5\r\n"));
        assert!(s.contains("Content-Type: application/sdp\r\n"));
        assert!(s.ends_with("v=0\r\n"));
    }

    #[test]
    fn header_lookup_is_case_insensitive() {
        let mut h = Headers::new();
        h.insert("CSeq", "1");
        assert_eq!(h.get("cseq"), Some("1"));
        assert_eq!(h.get("CSEQ"), Some("1"));
        assert_eq!(h.get("CSeq"), Some("1"));
    }

    #[test]
    fn parse_simple_response() {
        let bytes = b"RTSP/1.0 200 OK\r\nCSeq: 3\r\nServer: TestCam/1.0\r\n\r\n";
        match try_parse_response(bytes).unwrap() {
            ParseStatus::Parsed { consumed, response } => {
                assert_eq!(consumed, bytes.len());
                assert_eq!(response.status, 200);
                assert_eq!(response.reason, "OK");
                assert_eq!(response.headers.get("CSeq"), Some("3"));
                assert_eq!(response.headers.get("Server"), Some("TestCam/1.0"));
                assert!(response.body.is_empty());
            }
            ParseStatus::NeedMore => panic!("expected complete parse"),
        }
    }

    #[test]
    fn parse_response_with_body() {
        let body = "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\n";
        let raw = format!(
            "RTSP/1.0 200 OK\r\nCSeq: 4\r\nContent-Length: {}\r\nContent-Type: application/sdp\r\n\r\n{}",
            body.len(),
            body
        );
        match try_parse_response(raw.as_bytes()).unwrap() {
            ParseStatus::Parsed { consumed, response } => {
                assert_eq!(consumed, raw.len());
                assert_eq!(response.status, 200);
                assert_eq!(response.body, body.as_bytes());
            }
            ParseStatus::NeedMore => panic!("expected complete parse"),
        }
    }

    #[test]
    fn partial_header_block_needs_more() {
        let bytes = b"RTSP/1.0 200 OK\r\nCSeq: 5\r\n";
        assert!(matches!(
            try_parse_response(bytes).unwrap(),
            ParseStatus::NeedMore
        ));
    }

    #[test]
    fn partial_body_needs_more() {
        let raw = b"RTSP/1.0 200 OK\r\nCSeq: 6\r\nContent-Length: 10\r\n\r\nshort";
        assert!(matches!(
            try_parse_response(raw).unwrap(),
            ParseStatus::NeedMore
        ));
    }

    #[test]
    fn parse_unauthorized() {
        let bytes = b"RTSP/1.0 401 Unauthorized\r\nCSeq: 1\r\nWWW-Authenticate: Digest realm=\"x\", nonce=\"y\"\r\n\r\n";
        let ParseStatus::Parsed { response, .. } = try_parse_response(bytes).unwrap() else {
            panic!("expected complete parse");
        };
        assert!(response.is_unauthorized());
        assert!(response.headers.get("www-authenticate").is_some());
    }

    #[test]
    fn malformed_status_errors() {
        let bytes = b"HTTP/1.1 200 OK\r\nCSeq: 1\r\n\r\n";
        assert!(try_parse_response(bytes).is_err());
    }

    #[test]
    fn parse_request_options() {
        let wire = b"OPTIONS rtsp://cam.local/stream RTSP/1.0\r\nCSeq: 1\r\n\r\n";
        match try_parse_request(wire).unwrap() {
            RequestParseStatus::Complete { consumed, request } => {
                assert_eq!(consumed, wire.len());
                assert_eq!(request.method, Method::Options);
                assert_eq!(request.uri, "rtsp://cam.local/stream");
                assert_eq!(request.headers.get("cseq"), Some("1"));
            }
            RequestParseStatus::NeedMore => panic!("expected complete parse"),
        }
    }

    #[test]
    fn parse_request_need_more() {
        let partial = b"OPTIONS rtsp://cam/ RTSP/1.0\r\nCSeq: 1\r\n";
        assert!(matches!(
            try_parse_request(partial).unwrap(),
            RequestParseStatus::NeedMore
        ));
    }

    #[test]
    fn parse_request_with_body() {
        let body = "v=0\r\n";
        let raw = format!(
            "ANNOUNCE rtsp://x/y RTSP/1.0\r\nCSeq: 3\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        match try_parse_request(raw.as_bytes()).unwrap() {
            RequestParseStatus::Complete { consumed, request } => {
                assert_eq!(consumed, raw.len());
                assert_eq!(request.method, Method::Announce);
                assert_eq!(request.body, body.as_bytes());
            }
            RequestParseStatus::NeedMore => panic!("expected complete parse"),
        }
    }

    #[test]
    fn response_encode_roundtrip() {
        let resp = Response::build(200, 7);
        let wire = resp.encode();
        let text = std::str::from_utf8(&wire).unwrap();
        assert!(text.starts_with("RTSP/1.0 200 OK\r\n"));
        assert!(text.contains("Cseq: 7\r\n"));
        assert!(text.ends_with("\r\n\r\n"));
    }

    #[test]
    fn response_encode_with_body() {
        let mut headers = Headers::new();
        headers.insert("CSeq", "5");
        headers.insert("Content-Type", "application/sdp");
        let body = b"v=0\r\n".to_vec();
        headers.insert("Content-Length", body.len().to_string());
        let resp = Response {
            status: 200,
            reason: "OK".into(),
            headers,
            body: body.clone(),
        };
        let wire = resp.encode();
        let text = std::str::from_utf8(&wire).unwrap();
        assert!(text.starts_with("RTSP/1.0 200 OK\r\n"));
        assert!(text.ends_with("v=0\r\n"));
    }

    #[test]
    fn status_reason_known_codes() {
        assert_eq!(status_reason(200), "OK");
        assert_eq!(status_reason(400), "Bad Request");
        assert_eq!(status_reason(401), "Unauthorized");
        assert_eq!(status_reason(404), "Not Found");
        assert_eq!(status_reason(454), "Session Not Found");
        assert_eq!(status_reason(455), "Method Not Valid in This State");
        assert_eq!(status_reason(461), "Unsupported Transport");
        assert_eq!(status_reason(501), "Not Implemented");
        assert_eq!(status_reason(503), "Service Unavailable");
    }

    #[test]
    fn oversized_content_length_response_is_rejected_not_panic() {
        // `Content-Length` == u64::MAX would make `body_start + content_length`
        // wrap; the parser must reject it cleanly rather than panic.
        let wire = b"RTSP/1.0 200 OK\r\nCSeq: 1\r\nContent-Length: 18446744073709551615\r\n\r\n";
        let result = try_parse_response(wire);
        assert!(result.is_err(), "oversized Content-Length must be an error");
    }

    #[test]
    fn oversized_content_length_request_is_rejected_not_panic() {
        // Same wrap attack on the request path (server/connection.rs feed).
        let wire =
            b"ANNOUNCE rtsp://cam/ RTSP/1.0\r\nCSeq: 1\r\nContent-Length: 18446744073709551615\r\n\r\n";
        let result = try_parse_request(wire);
        assert!(result.is_err(), "oversized Content-Length must be an error");
    }

    #[test]
    fn content_length_over_cap_is_rejected() {
        // A value that parses fine but exceeds MAX_RTSP_BODY_LEN (16 MiB).
        let over = MAX_RTSP_BODY_LEN + 1;
        let wire = format!("RTSP/1.0 200 OK\r\nCSeq: 1\r\nContent-Length: {over}\r\n\r\n");
        assert!(try_parse_response(wire.as_bytes()).is_err());
    }
}
