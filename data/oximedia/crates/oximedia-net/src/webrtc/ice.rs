//! ICE (Interactive Connectivity Establishment) candidate handling.
//!
//! This module provides types for working with ICE candidates used
//! for NAT traversal in WebRTC.

#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::similar_names)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::if_not_else)]
#![allow(clippy::format_push_string)]
#![allow(clippy::single_match_else)]
#![allow(clippy::redundant_slicing)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::format_collect)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::unused_async)]
#![allow(clippy::identity_op)]
use crate::error::{NetError, NetResult};
use std::fmt;
use std::net::IpAddr;

/// ICE candidate type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateType {
    /// Host candidate - local address.
    Host,
    /// Server reflexive - address from STUN.
    ServerReflexive,
    /// Peer reflexive - discovered during connectivity checks.
    PeerReflexive,
    /// Relay candidate - address from TURN server.
    Relay,
}

impl CandidateType {
    /// Parses from string.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "host" => Some(Self::Host),
            "srflx" => Some(Self::ServerReflexive),
            "prflx" => Some(Self::PeerReflexive),
            "relay" => Some(Self::Relay),
            _ => None,
        }
    }

    /// Returns string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::ServerReflexive => "srflx",
            Self::PeerReflexive => "prflx",
            Self::Relay => "relay",
        }
    }

    /// Returns candidate priority preference.
    #[must_use]
    pub const fn preference(&self) -> u32 {
        match self {
            Self::Host => 126,
            Self::PeerReflexive => 110,
            Self::ServerReflexive => 100,
            Self::Relay => 0,
        }
    }
}

impl fmt::Display for CandidateType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Transport protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransportProtocol {
    /// UDP transport.
    #[default]
    Udp,
    /// TCP transport.
    Tcp,
}

impl TransportProtocol {
    /// Parses from string.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "udp" => Some(Self::Udp),
            "tcp" => Some(Self::Tcp),
            _ => None,
        }
    }

    /// Returns string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Udp => "udp",
            Self::Tcp => "tcp",
        }
    }
}

impl fmt::Display for TransportProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// TCP candidate type (for TCP candidates).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcpType {
    /// Active - initiates connection.
    Active,
    /// Passive - waits for connection.
    Passive,
    /// Simultaneous open.
    So,
}

impl TcpType {
    /// Parses from string.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "active" => Some(Self::Active),
            "passive" => Some(Self::Passive),
            "so" => Some(Self::So),
            _ => None,
        }
    }

    /// Returns string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Passive => "passive",
            Self::So => "so",
        }
    }
}

/// ICE candidate.
#[derive(Debug, Clone)]
pub struct IceCandidate {
    /// Foundation string.
    pub foundation: String,
    /// Component ID (1 = RTP, 2 = RTCP).
    pub component: u32,
    /// Transport protocol.
    pub protocol: TransportProtocol,
    /// Priority.
    pub priority: u32,
    /// IP address.
    pub address: String,
    /// Port number.
    pub port: u16,
    /// Candidate type.
    pub candidate_type: CandidateType,
    /// Related address (for reflexive/relay).
    pub related_address: Option<String>,
    /// Related port.
    pub related_port: Option<u16>,
    /// TCP type (for TCP candidates).
    pub tcp_type: Option<TcpType>,
    /// Extension attributes.
    pub extensions: Vec<(String, String)>,
}

impl IceCandidate {
    /// Creates a new host candidate.
    #[must_use]
    pub fn host(foundation: impl Into<String>, address: impl Into<String>, port: u16) -> Self {
        let addr = address.into();
        Self {
            foundation: foundation.into(),
            component: 1,
            protocol: TransportProtocol::Udp,
            priority: calculate_priority(CandidateType::Host, 1, 1),
            address: addr,
            port,
            candidate_type: CandidateType::Host,
            related_address: None,
            related_port: None,
            tcp_type: None,
            extensions: Vec::new(),
        }
    }

    /// Creates a server reflexive candidate.
    #[must_use]
    pub fn server_reflexive(
        foundation: impl Into<String>,
        address: impl Into<String>,
        port: u16,
        related_address: impl Into<String>,
        related_port: u16,
    ) -> Self {
        Self {
            foundation: foundation.into(),
            component: 1,
            protocol: TransportProtocol::Udp,
            priority: calculate_priority(CandidateType::ServerReflexive, 1, 1),
            address: address.into(),
            port,
            candidate_type: CandidateType::ServerReflexive,
            related_address: Some(related_address.into()),
            related_port: Some(related_port),
            tcp_type: None,
            extensions: Vec::new(),
        }
    }

    /// Creates a relay candidate.
    #[must_use]
    pub fn relay(
        foundation: impl Into<String>,
        address: impl Into<String>,
        port: u16,
        related_address: impl Into<String>,
        related_port: u16,
    ) -> Self {
        Self {
            foundation: foundation.into(),
            component: 1,
            protocol: TransportProtocol::Udp,
            priority: calculate_priority(CandidateType::Relay, 1, 1),
            address: address.into(),
            port,
            candidate_type: CandidateType::Relay,
            related_address: Some(related_address.into()),
            related_port: Some(related_port),
            tcp_type: None,
            extensions: Vec::new(),
        }
    }

    /// Sets the component ID.
    #[must_use]
    pub const fn with_component(mut self, component: u32) -> Self {
        self.component = component;
        self
    }

    /// Sets the protocol.
    #[must_use]
    pub const fn with_protocol(mut self, protocol: TransportProtocol) -> Self {
        self.protocol = protocol;
        self
    }

    /// Sets the priority.
    #[must_use]
    pub const fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Sets the TCP type.
    #[must_use]
    pub fn with_tcp_type(mut self, tcp_type: TcpType) -> Self {
        self.tcp_type = Some(tcp_type);
        self
    }

    /// Returns true if this is a host candidate.
    #[must_use]
    pub const fn is_host(&self) -> bool {
        matches!(self.candidate_type, CandidateType::Host)
    }

    /// Returns true if this is a relay candidate.
    #[must_use]
    pub const fn is_relay(&self) -> bool {
        matches!(self.candidate_type, CandidateType::Relay)
    }

    /// Formats as SDP candidate attribute.
    #[must_use]
    pub fn to_sdp(&self) -> String {
        let mut parts = vec![
            format!("candidate:{}", self.foundation),
            self.component.to_string(),
            self.protocol.to_string(),
            self.priority.to_string(),
            self.address.clone(),
            self.port.to_string(),
            "typ".to_string(),
            self.candidate_type.to_string(),
        ];

        if let (Some(ref addr), Some(port)) = (&self.related_address, self.related_port) {
            parts.push("raddr".to_string());
            parts.push(addr.clone());
            parts.push("rport".to_string());
            parts.push(port.to_string());
        }

        if let Some(ref tcp_type) = self.tcp_type {
            parts.push("tcptype".to_string());
            parts.push(tcp_type.as_str().to_string());
        }

        for (key, value) in &self.extensions {
            parts.push(key.clone());
            parts.push(value.clone());
        }

        parts.join(" ")
    }

    /// Parses from SDP candidate attribute value.
    ///
    /// # Errors
    ///
    /// Returns an error if the candidate is malformed.
    pub fn parse(s: &str) -> NetResult<Self> {
        let s = s.strip_prefix("candidate:").unwrap_or(s);
        let parts: Vec<&str> = s.split_whitespace().collect();

        if parts.len() < 8 {
            return Err(NetError::parse(0, "Candidate too short"));
        }

        let foundation = parts[0].to_string();
        let component: u32 = parts[1]
            .parse()
            .map_err(|_| NetError::parse(0, "Invalid component"))?;
        let protocol = TransportProtocol::parse(parts[2])
            .ok_or_else(|| NetError::parse(0, "Invalid protocol"))?;
        let priority: u32 = parts[3]
            .parse()
            .map_err(|_| NetError::parse(0, "Invalid priority"))?;
        let address = parts[4].to_string();
        let port: u16 = parts[5]
            .parse()
            .map_err(|_| NetError::parse(0, "Invalid port"))?;

        // parts[6] should be "typ"
        let candidate_type = CandidateType::parse(parts[7])
            .ok_or_else(|| NetError::parse(0, "Invalid candidate type"))?;

        let mut candidate = Self {
            foundation,
            component,
            protocol,
            priority,
            address,
            port,
            candidate_type,
            related_address: None,
            related_port: None,
            tcp_type: None,
            extensions: Vec::new(),
        };

        // Parse optional attributes
        let mut i = 8;
        while i + 1 < parts.len() {
            match parts[i] {
                "raddr" => {
                    candidate.related_address = Some(parts[i + 1].to_string());
                    i += 2;
                }
                "rport" => {
                    candidate.related_port = parts[i + 1].parse().ok();
                    i += 2;
                }
                "tcptype" => {
                    candidate.tcp_type = TcpType::parse(parts[i + 1]);
                    i += 2;
                }
                key => {
                    candidate
                        .extensions
                        .push((key.to_string(), parts[i + 1].to_string()));
                    i += 2;
                }
            }
        }

        Ok(candidate)
    }
}

/// ICE server configuration (STUN/TURN).
#[derive(Debug, Clone)]
pub struct IceServer {
    /// Server URLs.
    pub urls: Vec<String>,
    /// Username (for TURN).
    pub username: Option<String>,
    /// Credential (for TURN).
    pub credential: Option<String>,
}

impl IceServer {
    /// Creates a STUN server configuration.
    #[must_use]
    pub fn stun(url: impl Into<String>) -> Self {
        Self {
            urls: vec![url.into()],
            username: None,
            credential: None,
        }
    }

    /// Creates a TURN server configuration.
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
        }
    }

    /// Adds a URL.
    #[must_use]
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.urls.push(url.into());
        self
    }

    /// Returns true if this is a TURN server.
    #[must_use]
    pub fn is_turn(&self) -> bool {
        self.urls.iter().any(|u| u.starts_with("turn:"))
    }
}

/// Calculates ICE candidate priority.
///
/// Priority = (2^24) * type_preference + (2^8) * local_preference + (2^0) * (256 - component_id)
#[must_use]
pub fn calculate_priority(
    candidate_type: CandidateType,
    local_preference: u32,
    component_id: u32,
) -> u32 {
    let type_pref = candidate_type.preference();
    let local_pref = local_preference.min(65535);
    let component = (256 - component_id.min(256)).min(255);

    (type_pref << 24) | (local_pref << 8) | component
}

/// Computes foundation string for candidate.
///
/// Foundation is based on: type, base IP, server (for srflx/relay), protocol
#[must_use]
#[allow(dead_code)]
pub fn compute_foundation(
    candidate_type: CandidateType,
    base_address: &str,
    protocol: TransportProtocol,
    server: Option<&str>,
) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    candidate_type.as_str().hash(&mut hasher);
    base_address.hash(&mut hasher);
    protocol.as_str().hash(&mut hasher);
    if let Some(s) = server {
        s.hash(&mut hasher);
    }

    format!("{:x}", hasher.finish() & 0xFFFF_FFFF)
}

/// Validates an IP address string.
#[must_use]
#[allow(dead_code)]
pub fn is_valid_ip(s: &str) -> bool {
    s.parse::<IpAddr>().is_ok()
}

// =========================================================
// Extended ICE types (IceAgent, IcePair, IcePairState, etc.)
// =========================================================

/// ICE transport type for the extended IceCandidate2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IceTransport {
    /// UDP transport.
    Udp,
    /// TCP transport.
    Tcp,
    /// TLS transport.
    Tls,
}

/// ICE candidate type (extended, with type_preference).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IceCandidateType {
    /// Host candidate.
    Host,
    /// Server reflexive candidate.
    ServerReflexive,
    /// Peer reflexive candidate.
    PeerReflexive,
    /// Relay candidate.
    Relay,
}

impl IceCandidateType {
    /// Returns the type preference value used in priority calculation.
    #[must_use]
    pub const fn type_preference(self) -> u32 {
        match self {
            Self::Host => 126,
            Self::PeerReflexive => 110,
            Self::ServerReflexive => 100,
            Self::Relay => 0,
        }
    }
}

/// Extended ICE candidate with transport field.
#[derive(Debug, Clone)]
pub struct IceCandidate2 {
    /// Foundation string.
    pub foundation: String,
    /// Component ID.
    pub component: u32,
    /// Transport type.
    pub transport: IceTransport,
    /// Priority.
    pub priority: u32,
    /// IP address string.
    pub addr: String,
    /// Port.
    pub port: u16,
    /// Candidate type.
    pub candidate_type: IceCandidateType,
}

/// ICE pair state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcePairState {
    /// Waiting to be checked.
    Waiting,
    /// Check in progress.
    InProgress,
    /// Check succeeded.
    Succeeded,
    /// Check failed.
    Failed,
    /// Frozen (lower priority, not yet started).
    Frozen,
}

/// A pair of local and remote ICE candidates.
#[derive(Debug, Clone)]
pub struct IcePair {
    /// Local candidate.
    pub local: IceCandidate2,
    /// Remote candidate.
    pub remote: IceCandidate2,
    /// Current state.
    pub state: IcePairState,
}

impl IcePair {
    /// Creates a new pair in Waiting state.
    #[must_use]
    pub fn new(local: IceCandidate2, remote: IceCandidate2) -> Self {
        Self {
            local,
            remote,
            state: IcePairState::Waiting,
        }
    }

    /// Returns true if this pair has been nominated (Succeeded state).
    #[must_use]
    pub fn is_nominated(&self) -> bool {
        self.state == IcePairState::Succeeded
    }
}

/// ICE checklist – manages candidate pairs for connectivity checks.
#[derive(Debug, Default)]
pub struct IceChecklist {
    /// All candidate pairs.
    pub pairs: Vec<IcePair>,
}

impl IceChecklist {
    /// Creates a new empty checklist.
    #[must_use]
    pub fn new() -> Self {
        Self { pairs: Vec::new() }
    }

    /// Forms pairs by combining all local and remote candidates.
    pub fn form_pairs(
        &mut self,
        local_candidates: &[IceCandidate2],
        remote_candidates: &[IceCandidate2],
    ) {
        self.pairs.clear();
        for local in local_candidates {
            for remote in remote_candidates {
                self.pairs.push(IcePair::new(local.clone(), remote.clone()));
            }
        }
    }

    /// Returns a mutable reference to the next pair that should be checked.
    ///
    /// Prefers `Waiting` pairs before `InProgress` ones.
    #[must_use]
    pub fn next_pair_to_check(&mut self) -> Option<&mut IcePair> {
        // First try a Waiting pair.
        if let Some(idx) = self
            .pairs
            .iter()
            .position(|p| p.state == IcePairState::Waiting)
        {
            return Some(&mut self.pairs[idx]);
        }
        // Fall back to an InProgress pair.
        if let Some(idx) = self
            .pairs
            .iter()
            .position(|p| p.state == IcePairState::InProgress)
        {
            return Some(&mut self.pairs[idx]);
        }
        None
    }

    /// Returns the first nominated (Succeeded) pair, if any.
    #[must_use]
    pub fn nominated_pair(&self) -> Option<&IcePair> {
        self.pairs
            .iter()
            .find(|p| p.state == IcePairState::Succeeded)
    }
}

/// ICE credentials (ufrag + password).
#[derive(Debug, Clone)]
pub struct IceCredentials {
    /// Username fragment.
    pub ufrag: String,
    /// Password.
    pub pwd: String,
}

impl IceCredentials {
    /// Generates deterministic credentials based on a simple timestamp hash.
    ///
    /// For real applications use a cryptographically random source.
    #[must_use]
    pub fn generate() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);

        // Simple deterministic string from timestamp.
        let ufrag = format!("{:08x}", (ts & 0xFFFF_FFFF) as u32);
        let pwd = format!(
            "{:016x}{:016x}",
            ts as u64,
            ts.wrapping_mul(6364136223846793005)
        );
        Self { ufrag, pwd }
    }
}

/// Simple ICE agent that gathers host candidates.
#[derive(Debug, Default)]
pub struct IceAgentSimple {
    /// Gathered local candidates.
    pub local_candidates: Vec<IceCandidate2>,
    /// Received remote candidates.
    pub remote_candidates: Vec<IceCandidate2>,
}

impl IceAgentSimple {
    /// Creates a new agent.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Gathers a host candidate for the given IP address and a fixed port.
    ///
    /// Returns all gathered candidates (currently just the single host candidate).
    #[must_use]
    pub fn gather_candidates(ip: &str) -> Vec<IceCandidate2> {
        let candidate = IceCandidate2 {
            foundation: format!("host-{ip}"),
            component: 1,
            transport: IceTransport::Udp,
            priority: Self::compute_priority(IceCandidateType::Host.type_preference(), 65535, 1),
            addr: ip.to_string(),
            port: 0,
            candidate_type: IceCandidateType::Host,
        };
        vec![candidate]
    }

    /// Computes ICE candidate priority.
    ///
    /// Formula: `(2^24 * type_pref) + (2^8 * local_pref) + (256 - component)`
    #[must_use]
    pub fn compute_priority(type_pref: u32, local_pref: u32, component: u32) -> u32 {
        let component_val = if component < 256 { 256 - component } else { 0 };
        (2u32.pow(24))
            .wrapping_mul(type_pref)
            .wrapping_add((2u32.pow(8)).wrapping_mul(local_pref))
            .wrapping_add(component_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_type() {
        assert_eq!(CandidateType::parse("host"), Some(CandidateType::Host));
        assert_eq!(
            CandidateType::parse("srflx"),
            Some(CandidateType::ServerReflexive)
        );
        assert_eq!(CandidateType::Host.as_str(), "host");
        assert_eq!(CandidateType::Relay.as_str(), "relay");
    }

    #[test]
    fn test_transport_protocol() {
        assert_eq!(
            TransportProtocol::parse("udp"),
            Some(TransportProtocol::Udp)
        );
        assert_eq!(TransportProtocol::Tcp.as_str(), "tcp");
    }

    #[test]
    fn test_host_candidate() {
        let candidate = IceCandidate::host("1", "192.168.1.100", 54321);

        assert!(candidate.is_host());
        assert!(!candidate.is_relay());
        assert_eq!(candidate.address, "192.168.1.100");
        assert_eq!(candidate.port, 54321);
    }

    #[test]
    fn test_candidate_to_sdp() {
        let candidate = IceCandidate::host("abc123", "192.168.1.100", 54321).with_component(1);

        let sdp = candidate.to_sdp();
        assert!(sdp.contains("candidate:abc123"));
        assert!(sdp.contains("192.168.1.100"));
        assert!(sdp.contains("54321"));
        assert!(sdp.contains("typ host"));
    }

    #[test]
    fn test_srflx_candidate_sdp() {
        let candidate =
            IceCandidate::server_reflexive("abc", "203.0.113.5", 12345, "192.168.1.100", 54321);

        let sdp = candidate.to_sdp();
        assert!(sdp.contains("typ srflx"));
        assert!(sdp.contains("raddr 192.168.1.100"));
        assert!(sdp.contains("rport 54321"));
    }

    #[test]
    fn test_parse_candidate() {
        let sdp = "candidate:abc123 1 udp 2130706431 192.168.1.100 54321 typ host";
        let candidate = IceCandidate::parse(sdp).expect("should succeed in test");

        assert_eq!(candidate.foundation, "abc123");
        assert_eq!(candidate.component, 1);
        assert_eq!(candidate.protocol, TransportProtocol::Udp);
        assert_eq!(candidate.address, "192.168.1.100");
        assert_eq!(candidate.port, 54321);
        assert_eq!(candidate.candidate_type, CandidateType::Host);
    }

    #[test]
    fn test_parse_srflx_candidate() {
        let sdp =
            "candidate:xyz 1 udp 100 203.0.113.5 12345 typ srflx raddr 192.168.1.100 rport 54321";
        let candidate = IceCandidate::parse(sdp).expect("should succeed in test");

        assert_eq!(candidate.candidate_type, CandidateType::ServerReflexive);
        assert_eq!(candidate.related_address, Some("192.168.1.100".to_string()));
        assert_eq!(candidate.related_port, Some(54321));
    }

    #[test]
    fn test_ice_server_stun() {
        let server = IceServer::stun("stun:stun.example.com:3478");
        assert!(!server.is_turn());
        assert!(server.username.is_none());
    }

    #[test]
    fn test_ice_server_turn() {
        let server = IceServer::turn("turn:turn.example.com:3478", "user", "pass");
        assert!(server.is_turn());
        assert_eq!(server.username, Some("user".to_string()));
    }

    #[test]
    fn test_calculate_priority() {
        let host_priority = calculate_priority(CandidateType::Host, 65535, 1);
        let relay_priority = calculate_priority(CandidateType::Relay, 65535, 1);

        // Host should have higher priority
        assert!(host_priority > relay_priority);
    }

    #[test]
    fn test_compute_foundation() {
        let f1 = compute_foundation(
            CandidateType::Host,
            "192.168.1.1",
            TransportProtocol::Udp,
            None,
        );
        let f2 = compute_foundation(
            CandidateType::Host,
            "192.168.1.1",
            TransportProtocol::Udp,
            None,
        );
        let f3 = compute_foundation(
            CandidateType::Host,
            "192.168.1.2",
            TransportProtocol::Udp,
            None,
        );

        assert_eq!(f1, f2);
        assert_ne!(f1, f3);
    }

    #[test]
    fn test_is_valid_ip() {
        assert!(is_valid_ip("192.168.1.1"));
        assert!(is_valid_ip("::1"));
        assert!(!is_valid_ip("not.an.ip"));
    }

    // ---- Tests for extended ICE types ----

    #[test]
    fn test_ice_candidate_type_preference() {
        assert!(
            IceCandidateType::Host.type_preference() > IceCandidateType::Relay.type_preference()
        );
        assert!(
            IceCandidateType::PeerReflexive.type_preference()
                > IceCandidateType::ServerReflexive.type_preference()
        );
    }

    #[test]
    fn test_ice_transport_variants() {
        let t = IceTransport::Udp;
        assert_eq!(t, IceTransport::Udp);
        assert_ne!(t, IceTransport::Tcp);
    }

    #[test]
    fn test_ice_pair_new() {
        let local = IceCandidate2 {
            foundation: "l1".into(),
            component: 1,
            transport: IceTransport::Udp,
            priority: 1000,
            addr: "192.168.1.1".into(),
            port: 5000,
            candidate_type: IceCandidateType::Host,
        };
        let remote = IceCandidate2 {
            foundation: "r1".into(),
            component: 1,
            transport: IceTransport::Udp,
            priority: 900,
            addr: "192.168.1.2".into(),
            port: 5001,
            candidate_type: IceCandidateType::Host,
        };
        let pair = IcePair::new(local, remote);
        assert_eq!(pair.state, IcePairState::Waiting);
        assert!(!pair.is_nominated());
    }

    #[test]
    fn test_ice_pair_nominated() {
        let local = IceCandidate2 {
            foundation: "l1".into(),
            component: 1,
            transport: IceTransport::Udp,
            priority: 1000,
            addr: "10.0.0.1".into(),
            port: 4000,
            candidate_type: IceCandidateType::Host,
        };
        let remote = IceCandidate2 {
            foundation: "r1".into(),
            component: 1,
            transport: IceTransport::Udp,
            priority: 900,
            addr: "10.0.0.2".into(),
            port: 4001,
            candidate_type: IceCandidateType::Host,
        };
        let mut pair = IcePair::new(local, remote);
        assert!(!pair.is_nominated());
        pair.state = IcePairState::Succeeded;
        assert!(pair.is_nominated());
    }

    #[test]
    fn test_ice_checklist_form_pairs() {
        let local = vec![IceCandidate2 {
            foundation: "l1".into(),
            component: 1,
            transport: IceTransport::Udp,
            priority: 1000,
            addr: "10.0.0.1".into(),
            port: 4000,
            candidate_type: IceCandidateType::Host,
        }];
        let remote = vec![
            IceCandidate2 {
                foundation: "r1".into(),
                component: 1,
                transport: IceTransport::Udp,
                priority: 900,
                addr: "10.0.0.2".into(),
                port: 4001,
                candidate_type: IceCandidateType::Host,
            },
            IceCandidate2 {
                foundation: "r2".into(),
                component: 1,
                transport: IceTransport::Udp,
                priority: 800,
                addr: "10.0.0.3".into(),
                port: 4002,
                candidate_type: IceCandidateType::ServerReflexive,
            },
        ];
        let mut checklist = IceChecklist::new();
        checklist.form_pairs(&local, &remote);
        assert_eq!(checklist.pairs.len(), 2);
    }

    #[test]
    fn test_ice_checklist_next_pair() {
        let make_cand = |addr: &str, port: u16| IceCandidate2 {
            foundation: "f".into(),
            component: 1,
            transport: IceTransport::Udp,
            priority: 1000,
            addr: addr.into(),
            port,
            candidate_type: IceCandidateType::Host,
        };
        let local = vec![make_cand("10.0.0.1", 4000)];
        let remote = vec![make_cand("10.0.0.2", 4001)];
        let mut checklist = IceChecklist::new();
        checklist.form_pairs(&local, &remote);

        // Should return the Waiting pair.
        let next = checklist.next_pair_to_check();
        assert!(next.is_some());

        // Mark as Failed; no more pairs to check.
        checklist.pairs[0].state = IcePairState::Failed;
        assert!(checklist.next_pair_to_check().is_none());
    }

    #[test]
    fn test_ice_checklist_nominated_pair() {
        let make_cand = |addr: &str, port: u16| IceCandidate2 {
            foundation: "f".into(),
            component: 1,
            transport: IceTransport::Udp,
            priority: 1000,
            addr: addr.into(),
            port,
            candidate_type: IceCandidateType::Host,
        };
        let local = vec![make_cand("10.0.0.1", 4000)];
        let remote = vec![make_cand("10.0.0.2", 4001)];
        let mut checklist = IceChecklist::new();
        checklist.form_pairs(&local, &remote);

        assert!(checklist.nominated_pair().is_none());
        checklist.pairs[0].state = IcePairState::Succeeded;
        assert!(checklist.nominated_pair().is_some());
    }

    #[test]
    fn test_ice_credentials_generate() {
        let creds1 = IceCredentials::generate();
        assert!(!creds1.ufrag.is_empty());
        assert!(!creds1.pwd.is_empty());
        // ufrag should be 8 hex chars
        assert_eq!(creds1.ufrag.len(), 8);
    }

    #[test]
    fn test_ice_agent_simple_gather() {
        let candidates = IceAgentSimple::gather_candidates("192.168.1.100");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].addr, "192.168.1.100");
        assert_eq!(candidates[0].candidate_type, IceCandidateType::Host);
    }

    #[test]
    fn test_compute_priority() {
        let p1 = IceAgentSimple::compute_priority(126, 65535, 1);
        let p2 = IceAgentSimple::compute_priority(0, 65535, 1);
        assert!(p1 > p2);

        // Component: lower component ID => higher priority.
        let p3 = IceAgentSimple::compute_priority(126, 65535, 1);
        let p4 = IceAgentSimple::compute_priority(126, 65535, 2);
        assert!(p3 > p4);
    }

    #[test]
    fn test_ice_candidate_type2_variants() {
        assert_eq!(IceCandidateType::Host.type_preference(), 126);
        assert_eq!(IceCandidateType::Relay.type_preference(), 0);
        assert_eq!(IceCandidateType::ServerReflexive.type_preference(), 100);
        assert_eq!(IceCandidateType::PeerReflexive.type_preference(), 110);
    }
}
