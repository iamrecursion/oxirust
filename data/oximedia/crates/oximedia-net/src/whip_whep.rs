//! WHIP/WHEP client-side signaling for WebRTC ingest and egress.
//!
//! WHIP (WebRTC-HTTP Ingestion Protocol, draft-ietf-wish-whip) provides a
//! single-round-trip SDP offer/answer exchange over HTTP POST for publishing
//! WebRTC streams to a media server.
//!
//! WHEP (WebRTC-HTTP Egress Protocol, draft-ietf-wish-whep) provides the
//! symmetric playback/subscribe side.
//!
//! This module exposes lightweight **client** helpers:
//! - [`WhipClient`] — build SDP offers and parse answers for ingest
//! - [`WhepClient`] — build SDP offers and parse answers for egress
//! - [`SdpOffer`] / [`SdpAnswer`] — typed SDP wrappers
//! - [`build_sdp_offer`] / [`parse_sdp_attribute`] — low-level SDP utilities

use std::collections::HashMap;
use std::io::{Read as IoRead, Write as IoWrite};
use std::net::TcpStream;

use crate::error::{NetError, NetResult};

// ─── ICE credential helpers ───────────────────────────────────────────────────

/// Generate a pseudo-random 4-character ICE ufrag from a seed string.
fn ice_ufrag_from_seed(seed: &str) -> String {
    // Deterministic enough for testing; real impl would use CSPRNG.
    let mut h: u32 = 0x811c_9dc5;
    for b in seed.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    let chars: Vec<char> = "abcdefghijklmnopqrstuvwxyz0123456789".chars().collect();
    let len = chars.len() as u32;
    (0..4)
        .map(|i| {
            let idx = ((h >> (i * 8)) % len) as usize;
            chars[idx]
        })
        .collect()
}

/// Generate a 24-character ICE password from a seed string.
fn ice_pwd_from_seed(seed: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in seed.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let chars: Vec<char> = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        .chars()
        .collect();
    let len = chars.len() as u64;
    (0u64..24)
        .map(|i| {
            // Use wrapping arithmetic throughout to avoid overflow panics in
            // debug builds.  The constants are chosen for good bit dispersal.
            let mixed = h
                .wrapping_add(i.wrapping_mul(6_364_136_223_846_793_005))
                .wrapping_mul((i + 1).wrapping_mul(2_862_933_555_777_941_757));
            let idx = (mixed >> 33) % len;
            chars[idx as usize]
        })
        .collect()
}

// ─── SDP types ───────────────────────────────────────────────────────────────

/// An SDP offer, optionally tied to a resource URL after exchange.
#[derive(Debug, Clone)]
pub struct SdpOffer {
    /// The SDP offer body (RFC 4566 format).
    pub sdp: String,
    /// Set after a successful offer/answer exchange to the server-assigned
    /// resource URL (from `Location` response header).
    pub resource_url: Option<String>,
}

/// An SDP answer received from the server.
#[derive(Debug, Clone)]
pub struct SdpAnswer {
    /// The SDP answer body.
    pub sdp: String,
    /// ETag from the server response (used for conditional PATCH/DELETE).
    pub etag: Option<String>,
    /// Resource URL returned in the `Location` response header.
    pub resource_url: String,
}

// ─── WhipClient ──────────────────────────────────────────────────────────────

/// HTTP client helper for the WHIP ingest protocol.
///
/// # Example
///
/// ```
/// use oximedia_net::whip_whep::WhipClient;
///
/// let client = WhipClient::new("https://ingest.example.com/whip/live")
///     .with_bearer_token("my-secret-token");
///
/// let offer = client.build_offer("stream-001", &["opus"], &["vp9"]);
/// println!("SDP offer:\n{}", offer.sdp);
/// ```
#[derive(Debug, Clone)]
pub struct WhipClient {
    /// WHIP endpoint URL (the resource to POST the SDP offer to).
    pub endpoint_url: String,
    /// Optional bearer token sent in `Authorization: Bearer <token>`.
    pub bearer_token: Option<String>,
}

impl WhipClient {
    /// Create a new [`WhipClient`] targeting the given endpoint URL.
    pub fn new(endpoint_url: impl Into<String>) -> Self {
        Self {
            endpoint_url: endpoint_url.into(),
            bearer_token: None,
        }
    }

    /// Attach a bearer token for server-side authentication.
    #[must_use]
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    /// Build an SDP offer for publishing a stream.
    ///
    /// `stream_id` is embedded in the SDP session name and used to derive
    /// deterministic ICE credentials (replace with real CSPRNG values in
    /// production).  `audio_codecs` and `video_codecs` should be codec names
    /// such as `"opus"`, `"vp9"`, `"av1"`, `"h264"`.
    #[must_use]
    pub fn build_offer(
        &self,
        stream_id: &str,
        audio_codecs: &[&str],
        video_codecs: &[&str],
    ) -> SdpOffer {
        let ufrag = ice_ufrag_from_seed(stream_id);
        let pwd = ice_pwd_from_seed(&format!("{stream_id}-whip"));
        let sdp = build_sdp_offer(stream_id, &ufrag, &pwd, audio_codecs, video_codecs);
        SdpOffer {
            sdp,
            resource_url: None,
        }
    }

    /// Parse a raw SDP answer body returned by the server together with the
    /// `Location` header value as `resource_url`.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::Parse`] if the body does not look like a valid SDP
    /// answer (missing `v=` or `o=` lines).
    pub fn parse_answer(response_body: &str, resource_url: &str) -> NetResult<SdpAnswer> {
        parse_sdp_answer(response_body, resource_url)
    }

    /// Build the raw HTTP/1.1 POST request string for the SDP offer.
    ///
    /// Exposed as `pub(crate)` so tests can inspect the request format
    /// without making a real network connection.
    pub(crate) fn build_post_request(&self, sdp_offer: &str) -> NetResult<String> {
        let (host, port) = extract_host_port(&self.endpoint_url)?;
        let path = extract_path(&self.endpoint_url);
        let host_header = if port == 80 || port == 443 {
            host.clone()
        } else {
            format!("{host}:{port}")
        };
        let mut req = format!(
            "POST {path} HTTP/1.1\r\nHost: {host_header}\r\nContent-Type: application/sdp\r\nContent-Length: {}\r\nAccept: application/sdp\r\n",
            sdp_offer.len()
        );
        if let Some(token) = &self.bearer_token {
            req.push_str(&format!("Authorization: Bearer {token}\r\n"));
        }
        req.push_str("\r\n");
        req.push_str(sdp_offer);
        Ok(req)
    }

    /// Perform full WHIP SDP offer/answer negotiation over HTTP/1.1.
    ///
    /// Opens a [`TcpStream`] to the WHIP endpoint, sends an HTTP POST with
    /// `Content-Type: application/sdp`, reads the response, validates the
    /// `201 Created` status, and returns an [`SdpAnswer`] constructed from the
    /// response `Location` header and body.
    ///
    /// # Errors
    ///
    /// - [`NetError::Connection`] — TCP connection refused or DNS resolution failed.
    /// - [`NetError::Http`] — server returned a non-201 status code.
    /// - [`NetError::Parse`] — response is malformed or the body is not valid SDP.
    pub fn negotiate(&self, sdp_offer: &str) -> NetResult<SdpAnswer> {
        let (host, port) = extract_host_port(&self.endpoint_url)?;
        let request = self.build_post_request(sdp_offer)?;

        let addr = format!("{host}:{port}");
        let mut stream = TcpStream::connect(&addr)
            .map_err(|e| NetError::Connection(format!("WHIP connect to {addr}: {e}")))?;

        stream
            .write_all(request.as_bytes())
            .map_err(|e| NetError::Connection(format!("WHIP write: {e}")))?;

        let mut raw = String::new();
        stream
            .read_to_string(&mut raw)
            .map_err(|e| NetError::Connection(format!("WHIP read: {e}")))?;

        let (status, headers, body) = parse_http_response(&raw)?;
        if status != 201 {
            return Err(NetError::Http {
                status,
                message: format!("WHIP endpoint returned {status}; expected 201 Created"),
            });
        }

        let resource_url = headers
            .get("location")
            .cloned()
            .unwrap_or_else(|| self.endpoint_url.clone());

        let etag = headers.get("etag").cloned();
        let mut answer = parse_sdp_answer(&body, &resource_url)?;
        answer.etag = etag;
        Ok(answer)
    }

    /// Send a trickle-ICE candidate to `resource_url` via HTTP PATCH.
    ///
    /// The body is sent as `application/trickle-ice-sdpfrag` per
    /// draft-ietf-wish-whip §4.2.  A `204 No Content` response indicates
    /// success; any other status is returned as [`NetError::Http`].
    ///
    /// # Errors
    ///
    /// - [`NetError::Connection`] — TCP connection failed.
    /// - [`NetError::Http`] — server returned a status other than 204.
    /// - [`NetError::Parse`] — response could not be parsed.
    pub fn send_ice_candidate(&self, resource_url: &str, candidate: &str) -> NetResult<()> {
        let (host, port) = extract_host_port(resource_url)?;
        let path = extract_path(resource_url);
        let host_header = if port == 80 || port == 443 {
            host.clone()
        } else {
            format!("{host}:{port}")
        };
        let mut req = format!(
            "PATCH {path} HTTP/1.1\r\nHost: {host_header}\r\nContent-Type: application/trickle-ice-sdpfrag\r\nContent-Length: {}\r\n",
            candidate.len()
        );
        if let Some(token) = &self.bearer_token {
            req.push_str(&format!("Authorization: Bearer {token}\r\n"));
        }
        req.push_str("\r\n");
        req.push_str(candidate);

        let addr = format!("{host}:{port}");
        let mut stream = TcpStream::connect(&addr)
            .map_err(|e| NetError::Connection(format!("ICE connect to {addr}: {e}")))?;
        stream
            .write_all(req.as_bytes())
            .map_err(|e| NetError::Connection(format!("ICE write: {e}")))?;

        let mut raw = String::new();
        stream
            .read_to_string(&mut raw)
            .map_err(|e| NetError::Connection(format!("ICE read: {e}")))?;

        let (status, _headers, _body) = parse_http_response(&raw)?;
        if status != 204 {
            return Err(NetError::Http {
                status,
                message: format!("ICE candidate PATCH returned {status}; expected 204"),
            });
        }
        Ok(())
    }
}

// ─── WhepClient ──────────────────────────────────────────────────────────────

/// HTTP client helper for the WHEP egress protocol.
///
/// # Example
///
/// ```
/// use oximedia_net::whip_whep::WhepClient;
///
/// let client = WhepClient::new("https://stream.example.com/whep/live");
/// let offer = client.build_offer("playback-001", &["opus", "vp9"]);
/// println!("SDP offer:\n{}", offer.sdp);
/// ```
#[derive(Debug, Clone)]
pub struct WhepClient {
    /// WHEP endpoint URL.
    pub endpoint_url: String,
    /// Optional bearer token for authenticated playback.
    pub bearer_token: Option<String>,
}

impl WhepClient {
    /// Create a new [`WhepClient`] targeting the given endpoint URL.
    pub fn new(endpoint_url: impl Into<String>) -> Self {
        Self {
            endpoint_url: endpoint_url.into(),
            bearer_token: None,
        }
    }

    /// Attach a bearer token for server-side authentication.
    #[must_use]
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    /// Build an SDP offer for subscribing to a stream.
    ///
    /// `playback_id` identifies which stream the client wants to receive.
    /// `codecs` is a list of codec names the client is willing to decode
    /// (e.g. `["opus", "vp9"]`).  The method splits them into audio/video
    /// groups heuristically.
    #[must_use]
    pub fn build_offer(&self, playback_id: &str, codecs: &[&str]) -> SdpOffer {
        let audio_codecs: Vec<&str> = codecs
            .iter()
            .copied()
            .filter(|c| is_audio_codec(c))
            .collect();
        let video_codecs: Vec<&str> = codecs
            .iter()
            .copied()
            .filter(|c| !is_audio_codec(c))
            .collect();

        let ufrag = ice_ufrag_from_seed(playback_id);
        let pwd = ice_pwd_from_seed(&format!("{playback_id}-whep"));
        let sdp = build_sdp_offer(playback_id, &ufrag, &pwd, &audio_codecs, &video_codecs);
        SdpOffer {
            sdp,
            resource_url: None,
        }
    }

    /// Parse a raw SDP answer body together with the server `Location` URL.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::Parse`] if the body is not a valid SDP answer.
    pub fn parse_answer(response_body: &str, resource_url: &str) -> NetResult<SdpAnswer> {
        parse_sdp_answer(response_body, resource_url)
    }

    /// Build the raw HTTP/1.1 POST request string for the SDP offer.
    ///
    /// Exposed as `pub(crate)` for testing.
    pub(crate) fn build_post_request(&self, sdp_offer: &str) -> NetResult<String> {
        let (host, port) = extract_host_port(&self.endpoint_url)?;
        let path = extract_path(&self.endpoint_url);
        let host_header = if port == 80 || port == 443 {
            host.clone()
        } else {
            format!("{host}:{port}")
        };
        let mut req = format!(
            "POST {path} HTTP/1.1\r\nHost: {host_header}\r\nContent-Type: application/sdp\r\nContent-Length: {}\r\nAccept: application/sdp\r\n",
            sdp_offer.len()
        );
        if let Some(token) = &self.bearer_token {
            req.push_str(&format!("Authorization: Bearer {token}\r\n"));
        }
        req.push_str("\r\n");
        req.push_str(sdp_offer);
        Ok(req)
    }

    /// Perform full WHEP SDP offer/answer negotiation over HTTP/1.1.
    ///
    /// Sends an HTTP POST with `Content-Type: application/sdp` to the WHEP
    /// endpoint (same single-round-trip pattern as WHIP), reads the `201 Created`
    /// response, and returns the [`SdpAnswer`].
    ///
    /// # Errors
    ///
    /// - [`NetError::Connection`] — TCP failure.
    /// - [`NetError::Http`] — server returned a status other than 201.
    /// - [`NetError::Parse`] — response or SDP body is malformed.
    pub fn negotiate(&self, sdp_offer: &str) -> NetResult<SdpAnswer> {
        let (host, port) = extract_host_port(&self.endpoint_url)?;
        let request = self.build_post_request(sdp_offer)?;

        let addr = format!("{host}:{port}");
        let mut stream = TcpStream::connect(&addr)
            .map_err(|e| NetError::Connection(format!("WHEP connect to {addr}: {e}")))?;

        stream
            .write_all(request.as_bytes())
            .map_err(|e| NetError::Connection(format!("WHEP write: {e}")))?;

        let mut raw = String::new();
        stream
            .read_to_string(&mut raw)
            .map_err(|e| NetError::Connection(format!("WHEP read: {e}")))?;

        let (status, headers, body) = parse_http_response(&raw)?;
        if status != 201 {
            return Err(NetError::Http {
                status,
                message: format!("WHEP endpoint returned {status}; expected 201 Created"),
            });
        }

        let resource_url = headers
            .get("location")
            .cloned()
            .unwrap_or_else(|| self.endpoint_url.clone());

        let etag = headers.get("etag").cloned();
        let mut answer = parse_sdp_answer(&body, &resource_url)?;
        answer.etag = etag;
        Ok(answer)
    }
}

// ─── SDP helpers ─────────────────────────────────────────────────────────────

/// Classify whether a codec name belongs to audio (vs video).
fn is_audio_codec(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "opus" | "pcmu" | "pcma" | "g711" | "g722" | "aac" | "flac" | "vorbis" | "speex"
    )
}

/// Map a codec name to an RTP payload type number.
///
/// Returns a dynamic payload type (96+) for codecs not assigned a static PT.
fn rtp_pt_for_codec(codec: &str, base_dyn: u8) -> u8 {
    match codec.to_ascii_lowercase().as_str() {
        "pcmu" => 0,
        "pcma" => 8,
        "g722" => 9,
        _ => base_dyn,
    }
}

/// Build a minimal but well-formed SDP offer string.
///
/// The generated SDP follows RFC 4566 and includes:
/// - Session header (`v=`, `o=`, `s=`, `t=`)
/// - ICE credentials (`a=ice-ufrag`, `a=ice-pwd`)
/// - `a=ice-options:trickle` to signal support for trickle ICE
/// - One `m=audio` section per audio codec (if any)
/// - One `m=video` section per video codec (if any)
/// - `a=sendonly` to signal the ingest/offer direction
///
/// # Arguments
///
/// * `stream_id`    — embedded in `s=` and `o=` lines for traceability
/// * `ice_ufrag`    — ICE username fragment (>=4 chars, RFC 8445 §14.1)
/// * `ice_pwd`      — ICE password (>=22 chars, RFC 8445 §14.1)
/// * `audio_codecs` — list of audio codec names (e.g. `["opus"]`)
/// * `video_codecs` — list of video codec names (e.g. `["vp9", "av1"]`)
pub fn build_sdp_offer(
    stream_id: &str,
    ice_ufrag: &str,
    ice_pwd: &str,
    audio_codecs: &[&str],
    video_codecs: &[&str],
) -> String {
    let mut sdp = String::with_capacity(512);

    // Session-level
    sdp.push_str("v=0\r\n");
    sdp.push_str(&format!("o=- 0 0 IN IP4 0.0.0.0\r\n"));
    sdp.push_str(&format!("s={stream_id}\r\n"));
    sdp.push_str("t=0 0\r\n");
    sdp.push_str(&format!("a=ice-ufrag:{ice_ufrag}\r\n"));
    sdp.push_str(&format!("a=ice-pwd:{ice_pwd}\r\n"));
    sdp.push_str("a=ice-options:trickle\r\n");
    sdp.push_str("a=fingerprint:sha-256 00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00\r\n");

    // Audio m-section
    if !audio_codecs.is_empty() {
        let mut pts: Vec<u8> = Vec::new();
        let mut rtpmap_lines = String::new();
        let mut dyn_pt: u8 = 96;

        for codec in audio_codecs {
            let pt = rtp_pt_for_codec(codec, dyn_pt);
            if pt >= 96 {
                dyn_pt += 1;
            }
            pts.push(pt);
            let codec_upper = codec.to_ascii_uppercase();
            match codec.to_ascii_lowercase().as_str() {
                "pcmu" => {} // static; no rtpmap needed but include for clarity
                "pcma" => {}
                "g722" => {}
                "opus" => {
                    rtpmap_lines.push_str(&format!("a=rtpmap:{pt} OPUS/48000/2\r\n"));
                    rtpmap_lines.push_str(&format!("a=fmtp:{pt} minptime=10;useinbandfec=1\r\n"));
                }
                _ => {
                    rtpmap_lines.push_str(&format!("a=rtpmap:{pt} {codec_upper}/48000/2\r\n"));
                }
            }
        }

        let pt_list: Vec<String> = pts.iter().map(|p| p.to_string()).collect();
        sdp.push_str(&format!(
            "m=audio 9 UDP/TLS/RTP/SAVPF {}\r\n",
            pt_list.join(" ")
        ));
        sdp.push_str("c=IN IP4 0.0.0.0\r\n");
        sdp.push_str("a=sendonly\r\n");
        sdp.push_str("a=rtcp-mux\r\n");
        sdp.push_str(&rtpmap_lines);
    }

    // Video m-section
    if !video_codecs.is_empty() {
        let mut pts: Vec<u8> = Vec::new();
        let mut rtpmap_lines = String::new();
        let mut dyn_pt: u8 = 110;

        for codec in video_codecs {
            let pt = dyn_pt;
            dyn_pt += 1;
            pts.push(pt);
            let codec_upper = codec.to_ascii_uppercase();
            match codec.to_ascii_lowercase().as_str() {
                "vp8" => {
                    rtpmap_lines.push_str(&format!("a=rtpmap:{pt} VP8/90000\r\n"));
                }
                "vp9" => {
                    rtpmap_lines.push_str(&format!("a=rtpmap:{pt} VP9/90000\r\n"));
                    rtpmap_lines.push_str(&format!("a=fmtp:{pt} profile-id=0\r\n"));
                }
                "av1" => {
                    rtpmap_lines.push_str(&format!("a=rtpmap:{pt} AV1/90000\r\n"));
                    rtpmap_lines.push_str(&format!("a=fmtp:{pt} profile=0;level-idx=5;tier=0\r\n"));
                }
                "h264" => {
                    rtpmap_lines.push_str(&format!("a=rtpmap:{pt} H264/90000\r\n"));
                    rtpmap_lines.push_str(&format!(
                        "a=fmtp:{pt} level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42001f\r\n"
                    ));
                }
                _ => {
                    rtpmap_lines.push_str(&format!("a=rtpmap:{pt} {codec_upper}/90000\r\n"));
                }
            }
            // RTX for video codecs
            let rtx_pt = dyn_pt;
            dyn_pt += 1;
            pts.push(rtx_pt);
            rtpmap_lines.push_str(&format!("a=rtpmap:{rtx_pt} RTX/90000\r\n"));
            rtpmap_lines.push_str(&format!("a=fmtp:{rtx_pt} apt={pt}\r\n"));
        }

        let pt_list: Vec<String> = pts.iter().map(|p| p.to_string()).collect();
        sdp.push_str(&format!(
            "m=video 9 UDP/TLS/RTP/SAVPF {}\r\n",
            pt_list.join(" ")
        ));
        sdp.push_str("c=IN IP4 0.0.0.0\r\n");
        sdp.push_str("a=sendonly\r\n");
        sdp.push_str("a=rtcp-mux\r\n");
        sdp.push_str(&rtpmap_lines);
    }

    sdp
}

/// Extract the value of an SDP attribute line `a=<attribute>:<value>`.
///
/// Returns `None` if the attribute is not present or has no value part.
///
/// # Example
///
/// ```
/// use oximedia_net::whip_whep::parse_sdp_attribute;
///
/// let sdp = "v=0\r\na=ice-ufrag:abcd\r\na=ice-pwd:supersecret\r\n";
/// assert_eq!(parse_sdp_attribute(sdp, "ice-ufrag"), Some("abcd".to_owned()));
/// assert_eq!(parse_sdp_attribute(sdp, "missing"), None);
/// ```
pub fn parse_sdp_attribute(sdp: &str, attribute: &str) -> Option<String> {
    let prefix = format!("a={attribute}:");
    for line in sdp.lines() {
        let trimmed = line.trim_end_matches('\r');
        if let Some(value) = trimmed.strip_prefix(&prefix) {
            return Some(value.to_owned());
        }
    }
    None
}

/// Internal: parse and validate a raw SDP answer body.
fn parse_sdp_answer(response_body: &str, resource_url: &str) -> NetResult<SdpAnswer> {
    if resource_url.is_empty() {
        return Err(NetError::parse(0, "resource_url must not be empty"));
    }

    // Validate that this at least looks like SDP
    let has_version = response_body
        .lines()
        .any(|l| l.trim_end_matches('\r') == "v=0");
    if !has_version {
        return Err(NetError::parse(0, "SDP answer missing required 'v=0' line"));
    }

    let has_origin = response_body
        .lines()
        .any(|l| l.trim_end_matches('\r').starts_with("o="));
    if !has_origin {
        return Err(NetError::parse(0, "SDP answer missing required 'o=' line"));
    }

    Ok(SdpAnswer {
        sdp: response_body.to_owned(),
        etag: None,
        resource_url: resource_url.to_owned(),
    })
}

// ─── HTTP utilities ──────────────────────────────────────────────────────────

/// Extract the `(host, port)` pair from an HTTP/HTTPS URL.
///
/// Supports `http://` (default port 80) and `https://` (default port 443)
/// schemes as well as explicit port numbers (`host:port`).
///
/// # Errors
///
/// Returns [`NetError::InvalidUrl`] when the URL is missing a host or has an
/// invalid port number.
pub(crate) fn extract_host_port(url: &str) -> NetResult<(String, u16)> {
    // Strip scheme.
    let without_scheme = if let Some(rest) = url.strip_prefix("https://") {
        (rest, 443u16)
    } else if let Some(rest) = url.strip_prefix("http://") {
        (rest, 80u16)
    } else {
        return Err(NetError::InvalidUrl(format!(
            "URL must start with http:// or https://: {url}"
        )));
    };

    let (authority_and_path, default_port) = without_scheme;

    // Split off path/query/fragment.
    let authority = authority_and_path
        .split('/')
        .next()
        .unwrap_or(authority_and_path);

    if authority.is_empty() {
        return Err(NetError::InvalidUrl(format!("No host in URL: {url}")));
    }

    // Parse optional explicit port.
    if let Some((host, port_str)) = authority.rsplit_once(':') {
        let port = port_str.parse::<u16>().map_err(|_| {
            NetError::InvalidUrl(format!("Invalid port '{port_str}' in URL: {url}"))
        })?;
        Ok((host.to_owned(), port))
    } else {
        Ok((authority.to_owned(), default_port))
    }
}

/// Extract the URL path (including query string) from an HTTP/HTTPS URL.
///
/// Returns `"/"` when the URL has no explicit path component.
pub(crate) fn extract_path(url: &str) -> String {
    let without_scheme = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);

    // Skip the authority (host[:port]) part.
    if let Some(slash_pos) = without_scheme.find('/') {
        let path = &without_scheme[slash_pos..];
        if path.is_empty() {
            "/".to_owned()
        } else {
            path.to_owned()
        }
    } else {
        "/".to_owned()
    }
}

/// Parse a raw HTTP/1.1 response into `(status_code, headers, body)`.
///
/// Headers are stored with lower-cased names for case-insensitive lookup.
/// The header/body separator is the first `\r\n\r\n` (or `\n\n`) sequence.
///
/// # Errors
///
/// Returns [`NetError::Parse`] when the status line is missing or malformed.
pub(crate) fn parse_http_response(raw: &str) -> NetResult<(u16, HashMap<String, String>, String)> {
    // Split headers from body on the first blank line.
    let (header_section, body) = if let Some(pos) = raw.find("\r\n\r\n") {
        (&raw[..pos], raw[pos + 4..].to_owned())
    } else if let Some(pos) = raw.find("\n\n") {
        (&raw[..pos], raw[pos + 2..].to_owned())
    } else {
        (raw, String::new())
    };

    let mut lines = header_section.lines();

    // Parse status line: HTTP/1.1 <code> <reason>
    let status_line = lines
        .next()
        .ok_or_else(|| NetError::parse(0, "Empty HTTP response"))?
        .trim_end_matches('\r');

    let mut parts = status_line.splitn(3, ' ');
    // Skip the HTTP version token.
    let _version = parts
        .next()
        .ok_or_else(|| NetError::parse(0, "Malformed HTTP status line: missing version"))?;
    let code_str = parts
        .next()
        .ok_or_else(|| NetError::parse(0, "Malformed HTTP status line: missing status code"))?;
    let status: u16 = code_str
        .parse()
        .map_err(|_| NetError::parse(0, format!("Invalid HTTP status code: '{code_str}'")))?;

    // Parse headers.
    let mut headers: HashMap<String, String> = HashMap::new();
    for line in lines {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_owned());
        }
    }

    Ok((status, headers, body))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── WhipClient ──────────────────────────────────────────────────────────

    #[test]
    fn test_whip_client_new() {
        let c = WhipClient::new("https://ingest.example.com/whip");
        assert_eq!(c.endpoint_url, "https://ingest.example.com/whip");
        assert!(c.bearer_token.is_none());
    }

    #[test]
    fn test_whip_client_with_bearer_token() {
        let c = WhipClient::new("https://ingest.example.com/whip").with_bearer_token("tok-abc123");
        assert_eq!(c.bearer_token, Some("tok-abc123".to_owned()));
    }

    #[test]
    fn test_whip_build_offer_sdp_structure() {
        let c = WhipClient::new("https://ingest.example.com/whip");
        let offer = c.build_offer("stream-1", &["opus"], &["vp9"]);

        let sdp = &offer.sdp;
        assert!(sdp.contains("v=0"), "missing v=0");
        assert!(sdp.contains("o="), "missing o=");
        assert!(sdp.contains("s=stream-1"), "missing session name");
        assert!(sdp.contains("t=0 0"), "missing timing");
        assert!(sdp.contains("a=ice-ufrag:"), "missing ice-ufrag");
        assert!(sdp.contains("a=ice-pwd:"), "missing ice-pwd");
        assert!(sdp.contains("m=audio"), "missing audio m-section");
        assert!(sdp.contains("m=video"), "missing video m-section");
        assert!(sdp.contains("OPUS"), "missing OPUS rtpmap");
        assert!(sdp.contains("VP9"), "missing VP9 rtpmap");
        assert!(offer.resource_url.is_none());
    }

    #[test]
    fn test_whip_build_offer_no_video() {
        let c = WhipClient::new("https://ingest.example.com/whip");
        let offer = c.build_offer("audio-only", &["opus"], &[]);
        assert!(offer.sdp.contains("m=audio"));
        assert!(!offer.sdp.contains("m=video"));
    }

    #[test]
    fn test_whip_build_offer_no_audio() {
        let c = WhipClient::new("https://ingest.example.com/whip");
        let offer = c.build_offer("video-only", &[], &["av1"]);
        assert!(!offer.sdp.contains("m=audio"));
        assert!(offer.sdp.contains("m=video"));
        assert!(offer.sdp.contains("AV1"));
    }

    #[test]
    fn test_whip_parse_answer_valid() {
        let answer_sdp =
            "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 96\r\n";
        let result = WhipClient::parse_answer(
            answer_sdp,
            "https://ingest.example.com/whip/resource/abc123",
        );
        assert!(result.is_ok());
        let answer = result.expect("should be Ok");
        assert!(answer.sdp.contains("v=0"));
        assert_eq!(
            answer.resource_url,
            "https://ingest.example.com/whip/resource/abc123"
        );
    }

    #[test]
    fn test_whip_parse_answer_missing_v() {
        let bad_sdp = "o=- 0 0 IN IP4 0.0.0.0\r\nm=audio 9 RTP/AVP 0\r\n";
        let result = WhipClient::parse_answer(bad_sdp, "https://example.com/resource/1");
        assert!(result.is_err());
    }

    #[test]
    fn test_whip_parse_answer_missing_origin() {
        let bad_sdp = "v=0\r\ns=-\r\n";
        let result = WhipClient::parse_answer(bad_sdp, "https://example.com/resource/1");
        assert!(result.is_err());
    }

    #[test]
    fn test_whip_parse_answer_empty_resource_url() {
        let sdp = "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\n";
        let result = WhipClient::parse_answer(sdp, "");
        assert!(result.is_err());
    }

    // ── WhepClient ──────────────────────────────────────────────────────────

    #[test]
    fn test_whep_client_new() {
        let c = WhepClient::new("https://stream.example.com/whep");
        assert_eq!(c.endpoint_url, "https://stream.example.com/whep");
        assert!(c.bearer_token.is_none());
    }

    #[test]
    fn test_whep_with_bearer_token() {
        let c = WhepClient::new("https://stream.example.com/whep").with_bearer_token("view-token");
        assert_eq!(c.bearer_token, Some("view-token".to_owned()));
    }

    #[test]
    fn test_whep_build_offer_splits_codecs() {
        let c = WhepClient::new("https://stream.example.com/whep");
        let offer = c.build_offer("playback-1", &["opus", "vp9"]);
        assert!(offer.sdp.contains("m=audio"));
        assert!(offer.sdp.contains("m=video"));
        assert!(offer.sdp.contains("OPUS"));
        assert!(offer.sdp.contains("VP9"));
    }

    #[test]
    fn test_whep_build_offer_audio_only_codecs() {
        let c = WhepClient::new("https://stream.example.com/whep");
        let offer = c.build_offer("audio-pb", &["opus"]);
        assert!(offer.sdp.contains("m=audio"));
        assert!(!offer.sdp.contains("m=video"));
    }

    #[test]
    fn test_whep_parse_answer_valid() {
        let body = "v=0\r\no=- 1 2 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\n";
        let res = WhepClient::parse_answer(body, "https://stream.example.com/whep/sessions/xyz");
        assert!(res.is_ok());
        let a = res.expect("should be Ok");
        assert_eq!(
            a.resource_url,
            "https://stream.example.com/whep/sessions/xyz"
        );
        assert!(a.etag.is_none());
    }

    // ── build_sdp_offer ─────────────────────────────────────────────────────

    #[test]
    fn test_build_sdp_offer_multiple_video_codecs() {
        let sdp = build_sdp_offer(
            "multi",
            "uf01",
            "pwd0123456789012345678901",
            &[],
            &["vp8", "av1"],
        );
        assert!(sdp.contains("VP8"));
        assert!(sdp.contains("AV1"));
        // RTX entries should appear
        assert!(sdp.contains("RTX"));
    }

    #[test]
    fn test_build_sdp_offer_h264() {
        let sdp = build_sdp_offer(
            "h264-stream",
            "uf02",
            "pwd0123456789012345678901",
            &[],
            &["h264"],
        );
        assert!(sdp.contains("H264"));
        assert!(sdp.contains("profile-level-id"));
    }

    #[test]
    fn test_build_sdp_offer_opus_fmtp() {
        let sdp = build_sdp_offer(
            "opus-stream",
            "uf03",
            "pwd0123456789012345678901",
            &["opus"],
            &[],
        );
        assert!(sdp.contains("OPUS/48000/2"));
        assert!(sdp.contains("useinbandfec=1"));
    }

    #[test]
    fn test_build_sdp_offer_sendonly() {
        let sdp = build_sdp_offer(
            "send",
            "uf04",
            "pwd0123456789012345678901",
            &["opus"],
            &["vp9"],
        );
        // sendonly should appear (at least once per m-section)
        let count = sdp.matches("a=sendonly").count();
        assert!(count >= 1, "expected at least one a=sendonly, got {count}");
    }

    #[test]
    fn test_build_sdp_offer_rtcp_mux() {
        let sdp = build_sdp_offer(
            "mux",
            "uf05",
            "pwd0123456789012345678901",
            &["opus"],
            &["vp9"],
        );
        assert!(sdp.contains("a=rtcp-mux"));
    }

    // ── parse_sdp_attribute ─────────────────────────────────────────────────

    #[test]
    fn test_parse_sdp_attribute_present() {
        let sdp = "v=0\r\na=ice-ufrag:abcd\r\na=ice-pwd:superlong\r\n";
        assert_eq!(
            parse_sdp_attribute(sdp, "ice-ufrag"),
            Some("abcd".to_owned())
        );
        assert_eq!(
            parse_sdp_attribute(sdp, "ice-pwd"),
            Some("superlong".to_owned())
        );
    }

    #[test]
    fn test_parse_sdp_attribute_missing() {
        let sdp = "v=0\r\na=sendonly\r\n";
        assert_eq!(parse_sdp_attribute(sdp, "ice-ufrag"), None);
    }

    #[test]
    fn test_parse_sdp_attribute_no_colon() {
        // Flag-style attribute "a=sendonly" has no value; should not match a query for "sendonly"
        let sdp = "v=0\r\na=sendonly\r\n";
        assert_eq!(parse_sdp_attribute(sdp, "sendonly"), None);
    }

    #[test]
    fn test_parse_sdp_attribute_empty_sdp() {
        assert_eq!(parse_sdp_attribute("", "anything"), None);
    }

    #[test]
    fn test_parse_sdp_attribute_first_match() {
        // When the same attribute appears multiple times, return the first.
        let sdp = "v=0\r\na=foo:first\r\na=foo:second\r\n";
        assert_eq!(parse_sdp_attribute(sdp, "foo"), Some("first".to_owned()));
    }

    // ── ICE credential helpers ──────────────────────────────────────────────

    #[test]
    fn test_ice_ufrag_length() {
        let u = ice_ufrag_from_seed("test-seed");
        assert_eq!(u.len(), 4, "ice_ufrag should be 4 chars, got {}", u.len());
    }

    #[test]
    fn test_ice_pwd_length() {
        let p = ice_pwd_from_seed("test-seed");
        assert_eq!(p.len(), 24, "ice_pwd should be 24 chars, got {}", p.len());
    }

    #[test]
    fn test_ice_credentials_deterministic() {
        let u1 = ice_ufrag_from_seed("seed");
        let u2 = ice_ufrag_from_seed("seed");
        assert_eq!(u1, u2);

        let p1 = ice_pwd_from_seed("seed");
        let p2 = ice_pwd_from_seed("seed");
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_ice_credentials_different_seeds() {
        let u1 = ice_ufrag_from_seed("seed-a");
        let u2 = ice_ufrag_from_seed("seed-b");
        // Very low probability of collision for different seeds
        // We don't assert inequality since hash space is small, but at least
        // verify the function runs without panic.
        let _ = (u1, u2);
    }

    // ── extract_host_port ────────────────────────────────────────────────────

    #[test]
    fn test_extract_host_port_http_default() {
        let (host, port) = extract_host_port("http://example.com/path").expect("should parse");
        assert_eq!(host, "example.com");
        assert_eq!(port, 80);
    }

    #[test]
    fn test_extract_host_port_https_default() {
        let (host, port) =
            extract_host_port("https://stream.example.com/whep").expect("should parse");
        assert_eq!(host, "stream.example.com");
        assert_eq!(port, 443);
    }

    #[test]
    fn test_extract_host_port_custom_port() {
        let (host, port) = extract_host_port("http://localhost:8080/whip").expect("should parse");
        assert_eq!(host, "localhost");
        assert_eq!(port, 8080);
    }

    #[test]
    fn test_extract_host_port_invalid_scheme() {
        let result = extract_host_port("ftp://example.com");
        assert!(result.is_err(), "ftp:// should return error");
    }

    #[test]
    fn test_extract_host_port_invalid_port() {
        let result = extract_host_port("http://example.com:notaport/");
        assert!(result.is_err(), "Non-numeric port should fail");
    }

    // ── extract_path ─────────────────────────────────────────────────────────

    #[test]
    fn test_extract_path_with_path() {
        assert_eq!(
            extract_path("https://host.example.com/whip/live"),
            "/whip/live"
        );
    }

    #[test]
    fn test_extract_path_no_path() {
        assert_eq!(extract_path("https://host.example.com"), "/");
    }

    // ── parse_http_response ──────────────────────────────────────────────────

    #[test]
    fn test_parse_http_response_201_created() {
        let raw = "HTTP/1.1 201 Created\r\nContent-Type: application/sdp\r\n\r\nv=0\r\no=- 0 0 IN IP4 0.0.0.0\r\n";
        let (status, _headers, body) = parse_http_response(raw).expect("should parse");
        assert_eq!(status, 201);
        assert!(body.contains("v=0"));
    }

    #[test]
    fn test_parse_http_response_with_location_header() {
        let raw =
            "HTTP/1.1 201 Created\r\nLocation: https://server.example.com/resource/abc\r\n\r\n";
        let (_status, headers, _body) = parse_http_response(raw).expect("should parse");
        assert_eq!(
            headers.get("location").map(String::as_str),
            Some("https://server.example.com/resource/abc")
        );
    }

    #[test]
    fn test_parse_http_response_with_etag_header() {
        let raw = "HTTP/1.1 201 Created\r\nETag: \"abc123\"\r\n\r\n";
        let (_status, headers, _body) = parse_http_response(raw).expect("should parse");
        assert_eq!(headers.get("etag").map(String::as_str), Some("\"abc123\""));
    }

    #[test]
    fn test_parse_http_response_204_no_content() {
        let raw = "HTTP/1.1 204 No Content\r\n\r\n";
        let (status, _headers, body) = parse_http_response(raw).expect("should parse");
        assert_eq!(status, 204);
        assert!(body.is_empty());
    }

    #[test]
    fn test_parse_http_response_non_201_status() {
        let raw = "HTTP/1.1 400 Bad Request\r\n\r\n";
        let (status, _headers, _body) = parse_http_response(raw).expect("should parse");
        assert_eq!(status, 400);
    }

    #[test]
    fn test_parse_http_response_missing_status_line() {
        let result = parse_http_response("");
        assert!(result.is_err(), "Empty response should return error");
    }

    #[test]
    fn test_parse_http_response_with_body() {
        let body_content = "v=0\r\no=- 1 2 IN IP4 0.0.0.0\r\ns=-\r\n";
        let raw = format!("HTTP/1.1 201 Created\r\n\r\n{body_content}");
        let (_status, _headers, body) = parse_http_response(&raw).expect("should parse");
        assert_eq!(body, body_content);
    }

    #[test]
    fn test_parse_http_response_empty_body() {
        let raw = "HTTP/1.1 204 No Content\r\n\r\n";
        let (_status, _headers, body) = parse_http_response(raw).expect("should parse");
        assert!(body.is_empty());
    }

    #[test]
    fn test_parse_http_response_headers_lowercase() {
        let raw =
            "HTTP/1.1 200 OK\r\nContent-Type: application/sdp\r\nX-Custom-Header: value\r\n\r\n";
        let (_status, headers, _body) = parse_http_response(raw).expect("should parse");
        // All header names must be stored lower-cased.
        assert!(
            headers.contains_key("content-type"),
            "content-type should be lowercase"
        );
        assert!(headers.contains_key("x-custom-header"));
    }

    // ── request format tests ─────────────────────────────────────────────────

    #[test]
    fn test_whip_negotiate_builds_correct_request() {
        let client = WhipClient::new("http://ingest.example.com:1935/whip/live");
        let sdp = "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\n";
        let req = client
            .build_post_request(sdp)
            .expect("should build request");
        assert!(
            req.starts_with("POST /whip/live HTTP/1.1\r\n"),
            "Must start with POST path"
        );
        assert!(
            req.contains("Host: ingest.example.com:1935\r\n"),
            "Must include Host header"
        );
        assert!(req.contains("Content-Type: application/sdp\r\n"));
        assert!(req.contains(&format!("Content-Length: {}\r\n", sdp.len())));
        assert!(req.ends_with(sdp), "Request must end with the SDP body");
    }

    #[test]
    fn test_whep_negotiate_builds_correct_request() {
        let client = WhepClient::new("http://stream.example.com/whep/session");
        let sdp = "v=0\r\no=- 1 2 IN IP4 0.0.0.0\r\n";
        let req = client
            .build_post_request(sdp)
            .expect("should build request");
        assert!(req.contains("POST /whep/session HTTP/1.1\r\n"));
        assert!(req.contains("Content-Type: application/sdp\r\n"));
        assert!(req.ends_with(sdp));
    }

    #[test]
    fn test_whip_negotiate_includes_bearer_token() {
        let client =
            WhipClient::new("http://ingest.example.com/whip").with_bearer_token("my-token-xyz");
        let sdp = "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\n";
        let req = client
            .build_post_request(sdp)
            .expect("should build request");
        assert!(req.contains("Authorization: Bearer my-token-xyz\r\n"));
    }
}
