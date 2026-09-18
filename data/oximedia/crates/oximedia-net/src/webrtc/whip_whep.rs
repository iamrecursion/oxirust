//! WHIP/WHEP (WebRTC-HTTP Ingestion/Egress Protocol) implementation.
//!
//! WHIP (RFC draft-ietf-wish-whip) defines a simple HTTP-based signaling
//! protocol for WebRTC ingest (publishing) to media servers.
//!
//! WHEP (RFC draft-ietf-wish-whep) defines the counterpart for playback
//! (subscribing) from media servers.
//!
//! Both protocols use a single HTTP POST to exchange SDP offer/answer,
//! with DELETE to tear down sessions.

use crate::error::{NetError, NetResult};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

// ─── Common Types ─────────────────────────────────────────────────────────────

/// ICE server configuration for WHIP/WHEP.
#[derive(Debug, Clone)]
pub struct IceServerConfig {
    /// STUN/TURN server URLs.
    pub urls: Vec<String>,
    /// Username for TURN.
    pub username: Option<String>,
    /// Credential for TURN.
    pub credential: Option<String>,
    /// Credential type ("password" or "oauth").
    pub credential_type: Option<String>,
}

impl IceServerConfig {
    /// Creates a STUN-only server config.
    #[must_use]
    pub fn stun(url: impl Into<String>) -> Self {
        Self {
            urls: vec![url.into()],
            username: None,
            credential: None,
            credential_type: None,
        }
    }

    /// Creates a TURN server config with credentials.
    #[must_use]
    pub fn turn(
        url: impl Into<String>,
        username: impl Into<String>,
        credential: impl Into<String>,
    ) -> Self {
        Self {
            urls: vec![url.into()],
            username: Some(username.into()),
            credential: Some(credential.into()),
            credential_type: Some("password".to_owned()),
        }
    }

    /// Returns the Link header value for 201 responses.
    #[must_use]
    pub fn to_link_header(&self) -> String {
        let mut parts = Vec::new();
        for url in &self.urls {
            let mut link = format!("<{url}>; rel=\"ice-server\"");
            if let Some(ref user) = self.username {
                link.push_str(&format!("; username=\"{user}\""));
            }
            if let Some(ref cred) = self.credential {
                link.push_str(&format!("; credential=\"{cred}\""));
            }
            if let Some(ref ct) = self.credential_type {
                link.push_str(&format!("; credential-type=\"{ct}\""));
            }
            parts.push(link);
        }
        parts.join(", ")
    }
}

// ─── WHIP ─────────────────────────────────────────────────────────────────────

/// WHIP session state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhipState {
    /// Waiting for initial offer.
    WaitingOffer,
    /// Offer received, answer generated.
    Negotiating,
    /// Session established.
    Active,
    /// Session terminated.
    Terminated,
}

impl WhipState {
    /// Returns a human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::WaitingOffer => "waiting_offer",
            Self::Negotiating => "negotiating",
            Self::Active => "active",
            Self::Terminated => "terminated",
        }
    }
}

/// WHIP session representing a single ingest connection.
#[derive(Debug, Clone)]
pub struct WhipSession {
    /// Unique session ID (used as resource URL path).
    pub session_id: String,
    /// Current session state.
    pub state: WhipState,
    /// SDP offer from the client.
    pub offer_sdp: Option<String>,
    /// SDP answer from the server.
    pub answer_sdp: Option<String>,
    /// Bearer token for authentication (if any).
    pub auth_token: Option<String>,
    /// ICE candidates trickled via PATCH.
    pub trickle_candidates: Vec<String>,
    /// Session creation time.
    pub created_at: Instant,
    /// ETag for conditional requests.
    pub etag: String,
    /// Extensions (Link headers, custom metadata).
    pub extensions: HashMap<String, String>,
}

impl WhipSession {
    /// Creates a new WHIP session with the given ID.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        let id = session_id.into();
        let etag = format!("W/\"{}\"", simple_hash(&id));
        Self {
            session_id: id,
            state: WhipState::WaitingOffer,
            offer_sdp: None,
            answer_sdp: None,
            auth_token: None,
            trickle_candidates: Vec::new(),
            created_at: Instant::now(),
            etag,
            extensions: HashMap::new(),
        }
    }

    /// Processes a WHIP POST request (SDP offer).
    ///
    /// Returns the SDP answer to send back in the 201 response.
    pub fn process_offer(&mut self, offer_sdp: &str) -> NetResult<String> {
        if self.state != WhipState::WaitingOffer && self.state != WhipState::Negotiating {
            return Err(NetError::invalid_state(format!(
                "Cannot process offer in state: {}",
                self.state.name()
            )));
        }

        self.offer_sdp = Some(offer_sdp.to_owned());

        // Generate a minimal SDP answer based on the offer
        let answer = generate_sdp_answer(offer_sdp);
        self.answer_sdp = Some(answer.clone());
        self.state = WhipState::Negotiating;

        Ok(answer)
    }

    /// Processes trickle ICE candidates via PATCH.
    ///
    /// Accepts SDP fragment with `a=candidate:` lines.
    pub fn add_trickle_candidates(&mut self, sdp_fragment: &str) -> NetResult<()> {
        if self.state == WhipState::Terminated {
            return Err(NetError::invalid_state("Session terminated"));
        }

        for line in sdp_fragment.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("a=candidate:") || trimmed.starts_with("a=end-of-candidates") {
                self.trickle_candidates.push(trimmed.to_owned());
            }
        }

        // If we received an end-of-candidates, mark session as active
        if sdp_fragment.contains("end-of-candidates") {
            self.state = WhipState::Active;
        }

        Ok(())
    }

    /// Terminates the session (DELETE request).
    pub fn terminate(&mut self) {
        self.state = WhipState::Terminated;
    }

    /// Returns whether the session is active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.state == WhipState::Active
    }

    /// Returns the session duration.
    #[must_use]
    pub fn duration(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Returns the resource URL path for this session.
    #[must_use]
    pub fn resource_path(&self) -> String {
        format!("/whip/resource/{}", self.session_id)
    }
}

// ─── WHEP ─────────────────────────────────────────────────────────────────────

/// WHEP session state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhepState {
    /// Waiting for client offer.
    WaitingOffer,
    /// Negotiating (offer received, answer sent).
    Negotiating,
    /// Playback active.
    Active,
    /// Session terminated.
    Terminated,
}

impl WhepState {
    /// Returns a human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::WaitingOffer => "waiting_offer",
            Self::Negotiating => "negotiating",
            Self::Active => "active",
            Self::Terminated => "terminated",
        }
    }
}

/// WHEP session representing a single playback connection.
#[derive(Debug, Clone)]
pub struct WhepSession {
    /// Unique session ID.
    pub session_id: String,
    /// Current session state.
    pub state: WhepState,
    /// SDP offer from the client.
    pub offer_sdp: Option<String>,
    /// SDP answer from the server.
    pub answer_sdp: Option<String>,
    /// Bearer token for authentication.
    pub auth_token: Option<String>,
    /// ICE candidates trickled via PATCH.
    pub trickle_candidates: Vec<String>,
    /// Session creation time.
    pub created_at: Instant,
    /// ETag for conditional requests.
    pub etag: String,
    /// Stream key this session is subscribed to.
    pub stream_key: Option<String>,
    /// Layer selection for simulcast/SVC.
    pub selected_layer: Option<LayerSelection>,
}

impl WhepSession {
    /// Creates a new WHEP session.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        let id = session_id.into();
        let etag = format!("W/\"{}\"", simple_hash(&id));
        Self {
            session_id: id,
            state: WhepState::WaitingOffer,
            offer_sdp: None,
            answer_sdp: None,
            auth_token: None,
            trickle_candidates: Vec::new(),
            created_at: Instant::now(),
            etag,
            stream_key: None,
            selected_layer: None,
        }
    }

    /// Processes a WHEP POST request (SDP offer).
    ///
    /// Returns the SDP answer with receive-only media directions.
    pub fn process_offer(&mut self, offer_sdp: &str) -> NetResult<String> {
        if self.state != WhepState::WaitingOffer && self.state != WhepState::Negotiating {
            return Err(NetError::invalid_state(format!(
                "Cannot process offer in state: {}",
                self.state.name()
            )));
        }

        self.offer_sdp = Some(offer_sdp.to_owned());

        // Generate answer with sendonly (server sends to client)
        let answer = generate_whep_answer(offer_sdp);
        self.answer_sdp = Some(answer.clone());
        self.state = WhepState::Negotiating;

        Ok(answer)
    }

    /// Processes trickle ICE candidates via PATCH.
    pub fn add_trickle_candidates(&mut self, sdp_fragment: &str) -> NetResult<()> {
        if self.state == WhepState::Terminated {
            return Err(NetError::invalid_state("Session terminated"));
        }

        for line in sdp_fragment.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("a=candidate:") || trimmed.starts_with("a=end-of-candidates") {
                self.trickle_candidates.push(trimmed.to_owned());
            }
        }

        if sdp_fragment.contains("end-of-candidates") {
            self.state = WhepState::Active;
        }

        Ok(())
    }

    /// Selects a simulcast/SVC layer.
    pub fn select_layer(&mut self, layer: LayerSelection) -> NetResult<()> {
        if self.state == WhepState::Terminated {
            return Err(NetError::invalid_state("Session terminated"));
        }
        self.selected_layer = Some(layer);
        Ok(())
    }

    /// Terminates the session.
    pub fn terminate(&mut self) {
        self.state = WhepState::Terminated;
    }

    /// Returns whether the session is active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.state == WhepState::Active
    }

    /// Returns the resource URL path.
    #[must_use]
    pub fn resource_path(&self) -> String {
        format!("/whep/resource/{}", self.session_id)
    }
}

/// Layer selection for simulcast/SVC in WHEP.
#[derive(Debug, Clone)]
pub struct LayerSelection {
    /// Encoding ID (for simulcast).
    pub encoding_id: Option<String>,
    /// Spatial layer (for SVC).
    pub spatial_layer: Option<u8>,
    /// Temporal layer (for SVC).
    pub temporal_layer: Option<u8>,
    /// Maximum width.
    pub max_width: Option<u32>,
    /// Maximum height.
    pub max_height: Option<u32>,
    /// Maximum bitrate.
    pub max_bitrate: Option<u64>,
    /// Maximum framerate.
    pub max_framerate: Option<f64>,
}

impl LayerSelection {
    /// Creates a new layer selection for a specific encoding.
    #[must_use]
    pub fn encoding(id: impl Into<String>) -> Self {
        Self {
            encoding_id: Some(id.into()),
            spatial_layer: None,
            temporal_layer: None,
            max_width: None,
            max_height: None,
            max_bitrate: None,
            max_framerate: None,
        }
    }

    /// Creates a spatial/temporal layer selection.
    #[must_use]
    pub fn svc(spatial: u8, temporal: u8) -> Self {
        Self {
            encoding_id: None,
            spatial_layer: Some(spatial),
            temporal_layer: Some(temporal),
            max_width: None,
            max_height: None,
            max_bitrate: None,
            max_framerate: None,
        }
    }

    /// Sets maximum resolution.
    #[must_use]
    pub fn with_max_resolution(mut self, width: u32, height: u32) -> Self {
        self.max_width = Some(width);
        self.max_height = Some(height);
        self
    }
}

// ─── WHIP/WHEP Endpoint ──────────────────────────────────────────────────────

/// Configuration for a WHIP/WHEP endpoint.
#[derive(Debug, Clone)]
pub struct EndpointConfig {
    /// Base URL for the endpoint (e.g., `https://server.example.com`).
    pub base_url: String,
    /// WHIP endpoint path (e.g., "/whip").
    pub whip_path: String,
    /// WHEP endpoint path (e.g., "/whep").
    pub whep_path: String,
    /// ICE servers to advertise.
    pub ice_servers: Vec<IceServerConfig>,
    /// Require bearer token authentication.
    pub require_auth: bool,
    /// Maximum simultaneous sessions.
    pub max_sessions: usize,
    /// Session timeout (idle sessions are cleaned up).
    pub session_timeout: Duration,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8080".to_owned(),
            whip_path: "/whip".to_owned(),
            whep_path: "/whep".to_owned(),
            ice_servers: vec![IceServerConfig::stun("stun:stun.l.google.com:19302")],
            require_auth: false,
            max_sessions: 100,
            session_timeout: Duration::from_secs(300),
        }
    }
}

/// WHIP/WHEP endpoint manager.
///
/// Manages WHIP ingest sessions and WHEP playback sessions,
/// handling the HTTP-based signaling lifecycle.
#[derive(Debug)]
pub struct WhipWhepEndpoint {
    /// Configuration.
    config: EndpointConfig,
    /// Active WHIP sessions.
    whip_sessions: HashMap<String, WhipSession>,
    /// Active WHEP sessions.
    whep_sessions: HashMap<String, WhepSession>,
    /// Session counter for ID generation.
    session_counter: u64,
}

impl WhipWhepEndpoint {
    /// Creates a new endpoint with the given configuration.
    #[must_use]
    pub fn new(config: EndpointConfig) -> Self {
        Self {
            config,
            whip_sessions: HashMap::new(),
            whep_sessions: HashMap::new(),
            session_counter: 0,
        }
    }

    /// Creates a new WHIP session (called on POST to WHIP endpoint).
    pub fn create_whip_session(
        &mut self,
        offer_sdp: &str,
        auth_token: Option<&str>,
    ) -> NetResult<(String, String)> {
        if self.whip_sessions.len() >= self.config.max_sessions {
            return Err(NetError::connection("Maximum sessions reached"));
        }

        if self.config.require_auth && auth_token.is_none() {
            return Err(NetError::authentication("Bearer token required"));
        }

        let session_id = self.generate_session_id();
        let mut session = WhipSession::new(&session_id);
        session.auth_token = auth_token.map(|s| s.to_owned());

        let answer = session.process_offer(offer_sdp)?;
        let resource_path = session.resource_path();

        self.whip_sessions.insert(session_id, session);

        Ok((resource_path, answer))
    }

    /// Creates a new WHEP session (called on POST to WHEP endpoint).
    pub fn create_whep_session(
        &mut self,
        offer_sdp: &str,
        stream_key: Option<&str>,
        auth_token: Option<&str>,
    ) -> NetResult<(String, String)> {
        if self.whep_sessions.len() >= self.config.max_sessions {
            return Err(NetError::connection("Maximum sessions reached"));
        }

        if self.config.require_auth && auth_token.is_none() {
            return Err(NetError::authentication("Bearer token required"));
        }

        let session_id = self.generate_session_id();
        let mut session = WhepSession::new(&session_id);
        session.auth_token = auth_token.map(|s| s.to_owned());
        session.stream_key = stream_key.map(|s| s.to_owned());

        let answer = session.process_offer(offer_sdp)?;
        let resource_path = session.resource_path();

        self.whep_sessions.insert(session_id, session);

        Ok((resource_path, answer))
    }

    /// Handles a PATCH request to trickle ICE candidates for WHIP.
    pub fn trickle_whip(&mut self, session_id: &str, sdp_fragment: &str) -> NetResult<()> {
        let session = self
            .whip_sessions
            .get_mut(session_id)
            .ok_or_else(|| NetError::not_found(format!("WHIP session not found: {session_id}")))?;
        session.add_trickle_candidates(sdp_fragment)
    }

    /// Handles a PATCH request to trickle ICE candidates for WHEP.
    pub fn trickle_whep(&mut self, session_id: &str, sdp_fragment: &str) -> NetResult<()> {
        let session = self
            .whep_sessions
            .get_mut(session_id)
            .ok_or_else(|| NetError::not_found(format!("WHEP session not found: {session_id}")))?;
        session.add_trickle_candidates(sdp_fragment)
    }

    /// Handles a DELETE request for a WHIP session.
    pub fn delete_whip_session(&mut self, session_id: &str) -> NetResult<()> {
        let session = self
            .whip_sessions
            .get_mut(session_id)
            .ok_or_else(|| NetError::not_found(format!("WHIP session not found: {session_id}")))?;
        session.terminate();
        Ok(())
    }

    /// Handles a DELETE request for a WHEP session.
    pub fn delete_whep_session(&mut self, session_id: &str) -> NetResult<()> {
        let session = self
            .whep_sessions
            .get_mut(session_id)
            .ok_or_else(|| NetError::not_found(format!("WHEP session not found: {session_id}")))?;
        session.terminate();
        Ok(())
    }

    /// Returns the number of active WHIP sessions.
    #[must_use]
    pub fn active_whip_count(&self) -> usize {
        self.whip_sessions
            .values()
            .filter(|s| s.state != WhipState::Terminated)
            .count()
    }

    /// Returns the number of active WHEP sessions.
    #[must_use]
    pub fn active_whep_count(&self) -> usize {
        self.whep_sessions
            .values()
            .filter(|s| s.state != WhepState::Terminated)
            .count()
    }

    /// Cleans up terminated and timed-out sessions.
    pub fn cleanup(&mut self) {
        let timeout = self.config.session_timeout;
        self.whip_sessions
            .retain(|_, s| s.state != WhipState::Terminated && s.created_at.elapsed() < timeout);
        self.whep_sessions
            .retain(|_, s| s.state != WhepState::Terminated && s.created_at.elapsed() < timeout);
    }

    /// Returns ICE server Link headers for 201 responses.
    #[must_use]
    pub fn ice_server_headers(&self) -> Vec<String> {
        self.config
            .ice_servers
            .iter()
            .map(|s| s.to_link_header())
            .collect()
    }

    /// Returns a WHIP session by ID.
    #[must_use]
    pub fn get_whip_session(&self, id: &str) -> Option<&WhipSession> {
        self.whip_sessions.get(id)
    }

    /// Returns a WHEP session by ID.
    #[must_use]
    pub fn get_whep_session(&self, id: &str) -> Option<&WhepSession> {
        self.whep_sessions.get(id)
    }

    fn generate_session_id(&mut self) -> String {
        self.session_counter += 1;
        let ts = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        format!("{ts:x}-{:04x}", self.session_counter)
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Generates a minimal SDP answer from an offer (WHIP: recvonly).
fn generate_sdp_answer(offer: &str) -> String {
    let mut answer = String::with_capacity(offer.len());
    answer.push_str("v=0\r\n");
    answer.push_str("o=- 0 0 IN IP4 0.0.0.0\r\n");
    answer.push_str("s=-\r\n");
    answer.push_str("t=0 0\r\n");
    answer.push_str("a=group:BUNDLE 0\r\n");

    // Extract media lines from offer and mirror them with recvonly
    for line in offer.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("m=") {
            answer.push_str(&format!("{trimmed}\r\n"));
            answer.push_str("c=IN IP4 0.0.0.0\r\n");
            answer.push_str("a=recvonly\r\n");
            answer.push_str("a=rtcp-mux\r\n");
        } else if trimmed.starts_with("a=ice-ufrag:") || trimmed.starts_with("a=ice-pwd:") {
            answer.push_str(&format!("{trimmed}\r\n"));
        } else if trimmed.starts_with("a=fingerprint:") {
            answer.push_str(&format!("{trimmed}\r\n"));
        }
    }

    answer
}

/// Generates a minimal SDP answer for WHEP (sendonly from server perspective).
fn generate_whep_answer(offer: &str) -> String {
    let mut answer = String::with_capacity(offer.len());
    answer.push_str("v=0\r\n");
    answer.push_str("o=- 0 0 IN IP4 0.0.0.0\r\n");
    answer.push_str("s=-\r\n");
    answer.push_str("t=0 0\r\n");
    answer.push_str("a=group:BUNDLE 0\r\n");

    for line in offer.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("m=") {
            answer.push_str(&format!("{trimmed}\r\n"));
            answer.push_str("c=IN IP4 0.0.0.0\r\n");
            answer.push_str("a=sendonly\r\n");
            answer.push_str("a=rtcp-mux\r\n");
        } else if trimmed.starts_with("a=ice-ufrag:") || trimmed.starts_with("a=ice-pwd:") {
            answer.push_str(&format!("{trimmed}\r\n"));
        } else if trimmed.starts_with("a=fingerprint:") {
            answer.push_str(&format!("{trimmed}\r\n"));
        }
    }

    answer
}

/// Simple string hash for ETag generation.
fn simple_hash(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(u64::from(byte));
    }
    hash
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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

    fn sample_candidates() -> &'static str {
        "a=candidate:1 1 udp 2130706431 192.168.1.1 50000 typ host\r\n\
         a=end-of-candidates\r\n"
    }

    // 1. ICE server STUN config
    #[test]
    fn test_ice_server_stun() {
        let cfg = IceServerConfig::stun("stun:stun.example.com:3478");
        assert_eq!(cfg.urls.len(), 1);
        assert!(cfg.username.is_none());
    }

    // 2. ICE server TURN config
    #[test]
    fn test_ice_server_turn() {
        let cfg = IceServerConfig::turn("turn:turn.example.com", "user", "pass");
        assert!(cfg.username.is_some());
        assert!(cfg.credential.is_some());
    }

    // 3. ICE server Link header
    #[test]
    fn test_ice_server_link_header() {
        let cfg = IceServerConfig::stun("stun:stun.example.com:3478");
        let header = cfg.to_link_header();
        assert!(header.contains("ice-server"));
        assert!(header.contains("stun:stun.example.com"));
    }

    // 4. TURN Link header with credentials
    #[test]
    fn test_turn_link_header() {
        let cfg = IceServerConfig::turn("turn:t.example.com", "user", "pass");
        let header = cfg.to_link_header();
        assert!(header.contains("username=\"user\""));
        assert!(header.contains("credential=\"pass\""));
    }

    // 5. WHIP state names
    #[test]
    fn test_whip_state_names() {
        assert_eq!(WhipState::WaitingOffer.name(), "waiting_offer");
        assert_eq!(WhipState::Active.name(), "active");
        assert_eq!(WhipState::Terminated.name(), "terminated");
    }

    // 6. WHIP session creation
    #[test]
    fn test_whip_session_new() {
        let session = WhipSession::new("test-session");
        assert_eq!(session.state, WhipState::WaitingOffer);
        assert_eq!(session.session_id, "test-session");
        assert!(!session.is_active());
    }

    // 7. WHIP session process offer
    #[test]
    fn test_whip_process_offer() {
        let mut session = WhipSession::new("test");
        let result = session.process_offer(sample_offer());
        assert!(result.is_ok());
        assert_eq!(session.state, WhipState::Negotiating);
        assert!(session.offer_sdp.is_some());
        assert!(session.answer_sdp.is_some());
    }

    // 8. WHIP answer contains recvonly
    #[test]
    fn test_whip_answer_recvonly() {
        let mut session = WhipSession::new("test");
        let answer = session.process_offer(sample_offer()).expect("should work");
        assert!(answer.contains("recvonly"));
        assert!(answer.contains("rtcp-mux"));
    }

    // 9. WHIP trickle ICE candidates
    #[test]
    fn test_whip_trickle_candidates() {
        let mut session = WhipSession::new("test");
        session.process_offer(sample_offer()).expect("should work");
        session
            .add_trickle_candidates(sample_candidates())
            .expect("should work");
        assert_eq!(session.trickle_candidates.len(), 2);
        assert_eq!(session.state, WhipState::Active);
    }

    // 10. WHIP session terminate
    #[test]
    fn test_whip_terminate() {
        let mut session = WhipSession::new("test");
        session.terminate();
        assert_eq!(session.state, WhipState::Terminated);
    }

    // 11. WHIP resource path
    #[test]
    fn test_whip_resource_path() {
        let session = WhipSession::new("abc123");
        assert_eq!(session.resource_path(), "/whip/resource/abc123");
    }

    // 12. WHEP session creation
    #[test]
    fn test_whep_session_new() {
        let session = WhepSession::new("test-whep");
        assert_eq!(session.state, WhepState::WaitingOffer);
        assert!(!session.is_active());
    }

    // 13. WHEP process offer
    #[test]
    fn test_whep_process_offer() {
        let mut session = WhepSession::new("test");
        let result = session.process_offer(sample_offer());
        assert!(result.is_ok());
        assert_eq!(session.state, WhepState::Negotiating);
    }

    // 14. WHEP answer contains sendonly
    #[test]
    fn test_whep_answer_sendonly() {
        let mut session = WhepSession::new("test");
        let answer = session.process_offer(sample_offer()).expect("should work");
        assert!(answer.contains("sendonly"));
    }

    // 15. WHEP layer selection
    #[test]
    fn test_whep_layer_selection() {
        let mut session = WhepSession::new("test");
        session.process_offer(sample_offer()).expect("should work");
        let layer = LayerSelection::encoding("mid").with_max_resolution(1920, 1080);
        session.select_layer(layer).expect("should work");
        assert!(session.selected_layer.is_some());
    }

    // 16. WHEP SVC layer selection
    #[test]
    fn test_whep_svc_layer() {
        let layer = LayerSelection::svc(2, 1);
        assert_eq!(layer.spatial_layer, Some(2));
        assert_eq!(layer.temporal_layer, Some(1));
    }

    // 17. WHEP resource path
    #[test]
    fn test_whep_resource_path() {
        let session = WhepSession::new("xyz789");
        assert_eq!(session.resource_path(), "/whep/resource/xyz789");
    }

    // 18. Endpoint config default
    #[test]
    fn test_endpoint_config_default() {
        let cfg = EndpointConfig::default();
        assert_eq!(cfg.whip_path, "/whip");
        assert_eq!(cfg.whep_path, "/whep");
        assert!(!cfg.require_auth);
        assert_eq!(cfg.max_sessions, 100);
    }

    // 19. Endpoint create WHIP session
    #[test]
    fn test_endpoint_create_whip() {
        let mut endpoint = WhipWhepEndpoint::new(EndpointConfig::default());
        let (path, answer) = endpoint
            .create_whip_session(sample_offer(), None)
            .expect("should work");
        assert!(path.starts_with("/whip/resource/"));
        assert!(!answer.is_empty());
        assert_eq!(endpoint.active_whip_count(), 1);
    }

    // 20. Endpoint create WHEP session
    #[test]
    fn test_endpoint_create_whep() {
        let mut endpoint = WhipWhepEndpoint::new(EndpointConfig::default());
        let (path, _answer) = endpoint
            .create_whep_session(sample_offer(), Some("stream1"), None)
            .expect("should work");
        assert!(path.starts_with("/whep/resource/"));
        assert_eq!(endpoint.active_whep_count(), 1);
    }

    // 21. Endpoint auth required
    #[test]
    fn test_endpoint_auth_required() {
        let mut cfg = EndpointConfig::default();
        cfg.require_auth = true;
        let mut endpoint = WhipWhepEndpoint::new(cfg);
        let result = endpoint.create_whip_session(sample_offer(), None);
        assert!(result.is_err());
    }

    // 22. Endpoint auth with token
    #[test]
    fn test_endpoint_auth_with_token() {
        let mut cfg = EndpointConfig::default();
        cfg.require_auth = true;
        let mut endpoint = WhipWhepEndpoint::new(cfg);
        let result = endpoint.create_whip_session(sample_offer(), Some("my-token"));
        assert!(result.is_ok());
    }

    // 23. Endpoint max sessions
    #[test]
    fn test_endpoint_max_sessions() {
        let mut cfg = EndpointConfig::default();
        cfg.max_sessions = 1;
        let mut endpoint = WhipWhepEndpoint::new(cfg);
        endpoint
            .create_whip_session(sample_offer(), None)
            .expect("should work");
        let result = endpoint.create_whip_session(sample_offer(), None);
        assert!(result.is_err());
    }

    // 24. Endpoint trickle WHIP
    #[test]
    fn test_endpoint_trickle_whip() {
        let mut endpoint = WhipWhepEndpoint::new(EndpointConfig::default());
        let (path, _) = endpoint
            .create_whip_session(sample_offer(), None)
            .expect("should work");
        let session_id = path.trim_start_matches("/whip/resource/").to_owned();
        endpoint
            .trickle_whip(&session_id, sample_candidates())
            .expect("should work");
    }

    // 25. Endpoint delete session
    #[test]
    fn test_endpoint_delete_whip() {
        let mut endpoint = WhipWhepEndpoint::new(EndpointConfig::default());
        let (path, _) = endpoint
            .create_whip_session(sample_offer(), None)
            .expect("should work");
        let session_id = path.trim_start_matches("/whip/resource/").to_owned();
        endpoint
            .delete_whip_session(&session_id)
            .expect("should work");
        assert_eq!(endpoint.active_whip_count(), 0);
    }

    // 26. Endpoint cleanup
    #[test]
    fn test_endpoint_cleanup() {
        let mut endpoint = WhipWhepEndpoint::new(EndpointConfig::default());
        endpoint
            .create_whip_session(sample_offer(), None)
            .expect("should work");
        // Cleanup should not remove active sessions
        endpoint.cleanup();
        assert_eq!(endpoint.active_whip_count(), 1);
    }

    // 27. Endpoint ICE server headers
    #[test]
    fn test_endpoint_ice_headers() {
        let endpoint = WhipWhepEndpoint::new(EndpointConfig::default());
        let headers = endpoint.ice_server_headers();
        assert!(!headers.is_empty());
        assert!(headers[0].contains("ice-server"));
    }

    // 28. Endpoint session lookup
    #[test]
    fn test_endpoint_session_lookup() {
        let mut endpoint = WhipWhepEndpoint::new(EndpointConfig::default());
        let (path, _) = endpoint
            .create_whip_session(sample_offer(), None)
            .expect("should work");
        let session_id = path.trim_start_matches("/whip/resource/");
        assert!(endpoint.get_whip_session(session_id).is_some());
        assert!(endpoint.get_whep_session("nonexistent").is_none());
    }

    // 29. WHIP cannot process offer after termination
    #[test]
    fn test_whip_offer_after_terminate() {
        let mut session = WhipSession::new("test");
        session.terminate();
        let result = session.process_offer(sample_offer());
        assert!(result.is_err());
    }

    // 30. WHEP cannot trickle after termination
    #[test]
    fn test_whep_trickle_after_terminate() {
        let mut session = WhepSession::new("test");
        session.terminate();
        let result = session.add_trickle_candidates(sample_candidates());
        assert!(result.is_err());
    }

    // 31. WHIP session ETag
    #[test]
    fn test_whip_session_etag() {
        let session = WhipSession::new("test");
        assert!(session.etag.starts_with("W/\""));
    }

    // 32. WHEP stream key
    #[test]
    fn test_whep_stream_key() {
        let mut session = WhepSession::new("test");
        session.stream_key = Some("live/stream1".to_owned());
        assert_eq!(session.stream_key.as_deref(), Some("live/stream1"));
    }

    // 33. Endpoint not found errors
    #[test]
    fn test_endpoint_not_found() {
        let mut endpoint = WhipWhepEndpoint::new(EndpointConfig::default());
        assert!(endpoint.trickle_whip("nonexistent", "").is_err());
        assert!(endpoint.delete_whep_session("nonexistent").is_err());
    }

    // 34. Simple hash consistency
    #[test]
    fn test_simple_hash() {
        let h1 = simple_hash("test");
        let h2 = simple_hash("test");
        assert_eq!(h1, h2);
        assert_ne!(simple_hash("a"), simple_hash("b"));
    }
}
