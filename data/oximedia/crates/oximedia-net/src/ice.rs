//! ICE (Interactive Connectivity Establishment) for WebRTC NAT traversal.
//!
//! Implements RFC 8445 (Interactive Connectivity Establishment), which defines
//! the process of gathering network candidates and verifying connectivity
//! between peers so that WebRTC sessions can be established even when both
//! endpoints are behind NAT.
//!
//! Key concepts:
//! - **Candidate** — a transport address (host, server-reflexive, or relay) on
//!   which a peer can receive media.
//! - **Candidate pair** — one local and one remote candidate that may form a
//!   working communication path.
//! - **Connectivity check** — a STUN Binding Request/Response used to verify
//!   a candidate pair.
//! - **Nomination** — the process of selecting the best working pair.
//!
//! This module provides:
//! - Candidate types and priorities (RFC 8445 §5.1.2)
//! - Candidate pair management and state machines
//! - Connectivity check scheduling (RFC 8445 §6.1.4)
//! - ICE agent (controlling/controlled roles)
//! - TURN relay integration

#![allow(dead_code)]

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

// ─── Candidate Type ───────────────────────────────────────────────────────────

/// ICE candidate type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CandidateType {
    /// Directly reachable address on the local interface.
    Host,
    /// Address obtained via STUN (the NAT's public-facing address).
    ServerReflexive,
    /// Address provided by a TURN relay server.
    Relay,
    /// Peer-reflexive candidate discovered during connectivity checks.
    PeerReflexive,
}

impl CandidateType {
    /// Returns the type preference for priority calculation (RFC 8445 §5.1.2.1).
    #[must_use]
    pub const fn type_preference(&self) -> u32 {
        match self {
            Self::Host => 126,
            Self::PeerReflexive => 110,
            Self::ServerReflexive => 100,
            Self::Relay => 0,
        }
    }

    /// Returns the candidate type name as used in SDP.
    #[must_use]
    pub const fn sdp_name(&self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::ServerReflexive => "srflx",
            Self::Relay => "relay",
            Self::PeerReflexive => "prflx",
        }
    }
}

// ─── Transport Protocol ───────────────────────────────────────────────────────

/// Transport protocol for an ICE candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransportProtocol {
    /// User Datagram Protocol.
    Udp,
    /// Transmission Control Protocol.
    Tcp,
}

impl TransportProtocol {
    /// Returns the protocol name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Udp => "UDP",
            Self::Tcp => "TCP",
        }
    }
}

// ─── Candidate ────────────────────────────────────────────────────────────────

/// An ICE candidate (RFC 8445 §5.1).
#[derive(Debug, Clone)]
pub struct IceCandidate {
    /// Foundation: a string grouping candidates with the same base address.
    pub foundation: String,
    /// Component identifier (1 = RTP, 2 = RTCP).
    pub component: u8,
    /// Transport protocol.
    pub protocol: TransportProtocol,
    /// Candidate priority (higher = preferred).
    pub priority: u32,
    /// Transport address of this candidate.
    pub address: SocketAddr,
    /// Candidate type.
    pub candidate_type: CandidateType,
    /// Related address (e.g. host address for server-reflexive candidates).
    pub related_address: Option<SocketAddr>,
    /// Generation number.
    pub generation: u32,
}

impl IceCandidate {
    /// Creates a new host candidate with automatically computed priority.
    #[must_use]
    pub fn host(address: SocketAddr, component: u8) -> Self {
        let priority = Self::compute_priority(CandidateType::Host, 65535, component);
        Self {
            foundation: format!("host-{}-{}", address.ip(), address.port()),
            component,
            protocol: TransportProtocol::Udp,
            priority,
            address,
            candidate_type: CandidateType::Host,
            related_address: None,
            generation: 0,
        }
    }

    /// Creates a server-reflexive candidate.
    #[must_use]
    pub fn server_reflexive(srflx_addr: SocketAddr, base_addr: SocketAddr, component: u8) -> Self {
        let priority = Self::compute_priority(CandidateType::ServerReflexive, 65535, component);
        Self {
            foundation: format!("srflx-{}-{}", srflx_addr.ip(), base_addr.port()),
            component,
            protocol: TransportProtocol::Udp,
            priority,
            address: srflx_addr,
            candidate_type: CandidateType::ServerReflexive,
            related_address: Some(base_addr),
            generation: 0,
        }
    }

    /// Creates a relay candidate.
    #[must_use]
    pub fn relay(relay_addr: SocketAddr, base_addr: SocketAddr, component: u8) -> Self {
        let priority = Self::compute_priority(CandidateType::Relay, 65535, component);
        Self {
            foundation: format!("relay-{}-{}", relay_addr.ip(), relay_addr.port()),
            component,
            protocol: TransportProtocol::Udp,
            priority,
            address: relay_addr,
            candidate_type: CandidateType::Relay,
            related_address: Some(base_addr),
            generation: 0,
        }
    }

    /// Computes the candidate priority (RFC 8445 §5.1.2.1).
    #[must_use]
    pub fn compute_priority(candidate_type: CandidateType, local_pref: u32, component: u8) -> u32 {
        let type_pref = candidate_type.type_preference();
        (2u32.pow(24)) * type_pref
            + (2u32.pow(8)) * local_pref
            + (256u32.saturating_sub(u32::from(component)))
    }

    /// Returns the SDP candidate line for this candidate.
    #[must_use]
    pub fn to_sdp(&self) -> String {
        let mut sdp = format!(
            "candidate:{} {} {} {} {} {} typ {}",
            self.foundation,
            self.component,
            self.protocol.name().to_lowercase(),
            self.priority,
            self.address.ip(),
            self.address.port(),
            self.candidate_type.sdp_name(),
        );
        if let Some(rel) = self.related_address {
            sdp.push_str(&format!(" raddr {} rport {}", rel.ip(), rel.port()));
        }
        sdp.push_str(&format!(" generation {}", self.generation));
        sdp
    }
}

// ─── Candidate Pair State ─────────────────────────────────────────────────────

/// State of an ICE candidate pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairState {
    /// Waiting to be checked.
    Waiting,
    /// Connectivity check in progress.
    InProgress,
    /// Connectivity check succeeded.
    Succeeded,
    /// Connectivity check failed.
    Failed,
    /// Pair was frozen (lower priority than an unfinished one).
    Frozen,
}

// ─── Candidate Pair ───────────────────────────────────────────────────────────

/// An ICE candidate pair (one local + one remote candidate).
#[derive(Debug)]
pub struct CandidatePair {
    /// Local candidate.
    pub local: IceCandidate,
    /// Remote candidate.
    pub remote: IceCandidate,
    /// Pair priority (RFC 8445 §6.1.2.3).
    pub priority: u64,
    /// Current state.
    pub state: PairState,
    /// Whether this pair has been nominated.
    pub nominated: bool,
    /// Number of connectivity check attempts.
    pub check_count: u32,
    /// Last check time.
    pub last_check: Option<Instant>,
    /// Round-trip time from the last successful check.
    pub rtt: Option<Duration>,
}

impl CandidatePair {
    /// Creates a new candidate pair.
    #[must_use]
    pub fn new(local: IceCandidate, remote: IceCandidate, controlling: bool) -> Self {
        // Priority formula from RFC 8445 §6.1.2.3
        let g = if controlling {
            u64::from(local.priority)
        } else {
            u64::from(remote.priority)
        };
        let d = if controlling {
            u64::from(remote.priority)
        } else {
            u64::from(local.priority)
        };
        let priority = 2u64.pow(32) * g.min(d) + 2 * g.max(d) + u64::from(g > d);

        Self {
            local,
            remote,
            priority,
            state: PairState::Frozen,
            nominated: false,
            check_count: 0,
            last_check: None,
            rtt: None,
        }
    }

    /// Returns whether this pair can be checked (Waiting state).
    #[must_use]
    pub const fn can_check(&self) -> bool {
        matches!(self.state, PairState::Waiting)
    }

    /// Returns a unique pair key for use in a hash map.
    #[must_use]
    pub fn key(&self) -> String {
        format!("{}-{}", self.local.address, self.remote.address)
    }
}

// ─── ICE Agent Role ───────────────────────────────────────────────────────────

/// ICE agent role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IceRole {
    /// The controlling agent nominates the final candidate pair.
    Controlling,
    /// The controlled agent responds to nomination.
    Controlled,
}

impl IceRole {
    /// Returns the role name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Controlling => "controlling",
            Self::Controlled => "controlled",
        }
    }
}

// ─── ICE Agent State ──────────────────────────────────────────────────────────

/// ICE agent lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IceState {
    /// Not yet started.
    New,
    /// Gathering local candidates.
    Gathering,
    /// Checking candidate pairs.
    Checking,
    /// Found at least one working pair.
    Connected,
    /// ICE has completed (nominated pair selected).
    Completed,
    /// All candidate pairs failed.
    Failed,
    /// Agent was closed.
    Closed,
}

// ─── ICE Configuration ────────────────────────────────────────────────────────

/// Configuration for the ICE agent.
#[derive(Debug, Clone)]
pub struct IceConfig {
    /// Local ICE role.
    pub role: IceRole,
    /// STUN server addresses.
    pub stun_servers: Vec<SocketAddr>,
    /// TURN server address (optional).
    pub turn_server: Option<SocketAddr>,
    /// TURN credentials.
    pub turn_username: Option<String>,
    /// TURN password.
    pub turn_password: Option<String>,
    /// Connectivity check interval.
    pub check_interval: Duration,
    /// Maximum connectivity checks before giving up a pair.
    pub max_checks: u32,
    /// Candidate gathering timeout.
    pub gather_timeout: Duration,
    /// Use Trickle ICE (send candidates as they are discovered).
    pub trickle: bool,
}

impl Default for IceConfig {
    fn default() -> Self {
        Self {
            role: IceRole::Controlling,
            stun_servers: Vec::new(),
            turn_server: None,
            turn_username: None,
            turn_password: None,
            check_interval: Duration::from_millis(20),
            max_checks: 6,
            gather_timeout: Duration::from_secs(5),
            trickle: true,
        }
    }
}

impl IceConfig {
    /// Creates a new configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a STUN server.
    #[must_use]
    pub fn with_stun(mut self, addr: SocketAddr) -> Self {
        self.stun_servers.push(addr);
        self
    }

    /// Sets the TURN server.
    #[must_use]
    pub fn with_turn(mut self, addr: SocketAddr, username: &str, password: &str) -> Self {
        self.turn_server = Some(addr);
        self.turn_username = Some(username.to_owned());
        self.turn_password = Some(password.to_owned());
        self
    }

    /// Sets the role.
    #[must_use]
    pub const fn with_role(mut self, role: IceRole) -> Self {
        self.role = role;
        self
    }
}

// ─── Connectivity Check ───────────────────────────────────────────────────────

/// Result of a simulated connectivity check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckResult {
    /// Check succeeded.
    Success,
    /// Check timed out.
    Timeout,
    /// Received an ICMP unreachable error.
    Unreachable,
}

// ─── ICE Agent ────────────────────────────────────────────────────────────────

/// ICE agent managing candidate gathering, pairing, and connectivity checks.
pub struct IceAgent {
    /// Configuration.
    config: IceConfig,
    /// Current state.
    state: IceState,
    /// Local candidates gathered.
    local_candidates: Vec<IceCandidate>,
    /// Remote candidates received from the peer (via SDP / Trickle).
    remote_candidates: Vec<IceCandidate>,
    /// Candidate pairs sorted by priority (descending).
    pairs: Vec<CandidatePair>,
    /// Nominated pair key.
    nominated_pair_key: Option<String>,
    /// ICE ufrag (username fragment) for STUN credential.
    local_ufrag: String,
    /// ICE password for STUN credential.
    local_pwd: String,
    /// Tie-breaker value for role conflict resolution.
    tie_breaker: u64,
}

impl IceAgent {
    /// Creates a new ICE agent.
    #[must_use]
    pub fn new(config: IceConfig) -> Self {
        // Simple pseudo-random tie-breaker from system time nanos.
        let tie_breaker = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64)
            .unwrap_or(12345);

        Self {
            config,
            state: IceState::New,
            local_candidates: Vec::new(),
            remote_candidates: Vec::new(),
            pairs: Vec::new(),
            nominated_pair_key: None,
            local_ufrag: format!("ufrag{tie_breaker:08x}"),
            local_pwd: format!("pwd{tie_breaker:016x}"),
            tie_breaker,
        }
    }

    /// Returns the current ICE state.
    #[must_use]
    pub const fn state(&self) -> IceState {
        self.state
    }

    /// Returns the local ICE role.
    #[must_use]
    pub const fn role(&self) -> IceRole {
        self.config.role
    }

    /// Returns the local ufrag.
    #[must_use]
    pub fn local_ufrag(&self) -> &str {
        &self.local_ufrag
    }

    /// Returns the local ICE password.
    #[must_use]
    pub fn local_pwd(&self) -> &str {
        &self.local_pwd
    }

    /// Returns a reference to the local candidates.
    #[must_use]
    pub fn local_candidates(&self) -> &[IceCandidate] {
        &self.local_candidates
    }

    /// Returns a reference to the remote candidates.
    #[must_use]
    pub fn remote_candidates(&self) -> &[IceCandidate] {
        &self.remote_candidates
    }

    /// Returns a reference to all candidate pairs.
    #[must_use]
    pub fn candidate_pairs(&self) -> &[CandidatePair] {
        &self.pairs
    }

    /// Returns the number of candidate pairs in the given state.
    #[must_use]
    pub fn pairs_in_state(&self, state: PairState) -> usize {
        self.pairs.iter().filter(|p| p.state == state).count()
    }

    /// Starts gathering local candidates.
    ///
    /// In a real implementation this would bind sockets, send STUN Binding
    /// Requests to the configured STUN servers, and allocate TURN channels.
    /// Here we model the gathering phase structurally.
    pub fn start_gathering(&mut self) {
        self.state = IceState::Gathering;
    }

    /// Adds a locally discovered host candidate.
    pub fn add_local_candidate(&mut self, candidate: IceCandidate) {
        self.local_candidates.push(candidate);
        // Pair the new local candidate against all existing remote candidates.
        self.pair_new_local(self.local_candidates.len() - 1);
    }

    /// Adds a remote candidate received from the peer (Trickle ICE).
    pub fn add_remote_candidate(&mut self, candidate: IceCandidate) {
        self.remote_candidates.push(candidate);
        let remote_idx = self.remote_candidates.len() - 1;
        self.pair_new_remote(remote_idx);
    }

    /// Begins connectivity checks.  Transitions to `Checking`.
    ///
    /// Unfreezes the highest-priority pair per foundation.
    pub fn start_checks(&mut self) {
        self.state = IceState::Checking;

        // Unfreeze one pair per foundation (RFC 8445 §6.1.2.6).
        let mut seen_foundations: HashMap<String, bool> = HashMap::new();
        for pair in &mut self.pairs {
            let foundation = pair.local.foundation.clone();
            if let std::collections::hash_map::Entry::Vacant(e) = seen_foundations.entry(foundation)
            {
                pair.state = PairState::Waiting;
                e.insert(true);
            }
        }
    }

    /// Simulates performing a connectivity check on the next waiting pair.
    ///
    /// Returns the pair key of the checked pair, or `None` if no pair was waiting.
    pub fn perform_next_check(&mut self, result: CheckResult) -> Option<String> {
        let idx = self
            .pairs
            .iter()
            .position(|p| p.state == PairState::Waiting)?;

        self.pairs[idx].state = PairState::InProgress;
        self.pairs[idx].last_check = Some(Instant::now());
        self.pairs[idx].check_count += 1;

        let key = self.pairs[idx].key();

        match result {
            CheckResult::Success => {
                self.pairs[idx].state = PairState::Succeeded;
                self.pairs[idx].rtt = Some(Duration::from_millis(5));
                if self.state == IceState::Checking {
                    self.state = IceState::Connected;
                }
            }
            CheckResult::Timeout | CheckResult::Unreachable => {
                self.pairs[idx].state = PairState::Failed;
                // Re-check all failed → if none waiting and none in-progress, declare failed.
                if self.pairs.iter().all(|p| p.state == PairState::Failed) {
                    self.state = IceState::Failed;
                }
            }
        }

        Some(key)
    }

    /// Nominates the best succeeded pair (controlling agent only).
    ///
    /// Returns the nominated pair key or `None` if no succeeded pair exists.
    pub fn nominate_best_pair(&mut self) -> Option<String> {
        if self.config.role != IceRole::Controlling {
            return None;
        }

        // Find the highest-priority succeeded pair.
        let best_idx = self
            .pairs
            .iter()
            .enumerate()
            .filter(|(_, p)| p.state == PairState::Succeeded)
            .max_by_key(|(_, p)| p.priority)
            .map(|(i, _)| i)?;

        self.pairs[best_idx].nominated = true;
        let key = self.pairs[best_idx].key();
        self.nominated_pair_key = Some(key.clone());
        self.state = IceState::Completed;
        Some(key)
    }

    /// Returns the nominated pair, if any.
    #[must_use]
    pub fn nominated_pair(&self) -> Option<&CandidatePair> {
        let key = self.nominated_pair_key.as_ref()?;
        self.pairs.iter().find(|p| &p.key() == key)
    }

    /// Closes the agent.
    pub fn close(&mut self) {
        self.state = IceState::Closed;
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn pair_new_local(&mut self, local_idx: usize) {
        let controlling = self.config.role == IceRole::Controlling;
        let local = self.local_candidates[local_idx].clone();
        let remote_count = self.remote_candidates.len();
        for ri in 0..remote_count {
            let remote = self.remote_candidates[ri].clone();
            if local.component == remote.component {
                let pair = CandidatePair::new(local.clone(), remote, controlling);
                self.insert_pair(pair);
            }
        }
    }

    fn pair_new_remote(&mut self, remote_idx: usize) {
        let controlling = self.config.role == IceRole::Controlling;
        let remote = self.remote_candidates[remote_idx].clone();
        let local_count = self.local_candidates.len();
        for li in 0..local_count {
            let local = self.local_candidates[li].clone();
            if local.component == remote.component {
                let pair = CandidatePair::new(local, remote.clone(), controlling);
                self.insert_pair(pair);
            }
        }
    }

    fn insert_pair(&mut self, pair: CandidatePair) {
        // Keep pairs sorted by descending priority.
        let pos = self
            .pairs
            .iter()
            .position(|p| p.priority < pair.priority)
            .unwrap_or(self.pairs.len());
        self.pairs.insert(pos, pair);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn local_addr() -> SocketAddr {
        "127.0.0.1:5000".parse().expect("valid addr")
    }

    fn remote_addr() -> SocketAddr {
        "192.168.1.100:5000".parse().expect("valid addr")
    }

    // 1. CandidateType type preferences
    #[test]
    fn test_candidate_type_preference() {
        assert_eq!(CandidateType::Host.type_preference(), 126);
        assert!(CandidateType::Host.type_preference() > CandidateType::Relay.type_preference());
    }

    // 2. CandidateType SDP names
    #[test]
    fn test_candidate_type_sdp_names() {
        assert_eq!(CandidateType::Host.sdp_name(), "host");
        assert_eq!(CandidateType::ServerReflexive.sdp_name(), "srflx");
        assert_eq!(CandidateType::Relay.sdp_name(), "relay");
    }

    // 3. Host candidate construction
    #[test]
    fn test_host_candidate_construction() {
        let c = IceCandidate::host(local_addr(), 1);
        assert_eq!(c.candidate_type, CandidateType::Host);
        assert_eq!(c.component, 1);
        assert_eq!(c.address, local_addr());
        assert!(c.priority > 0);
    }

    // 4. Server-reflexive candidate
    #[test]
    fn test_server_reflexive_candidate() {
        let srflx: SocketAddr = "203.0.113.5:5000".parse().expect("valid addr");
        let c = IceCandidate::server_reflexive(srflx, local_addr(), 1);
        assert_eq!(c.candidate_type, CandidateType::ServerReflexive);
        assert_eq!(c.related_address, Some(local_addr()));
    }

    // 5. Relay candidate
    #[test]
    fn test_relay_candidate() {
        let relay: SocketAddr = "198.51.100.10:3478".parse().expect("valid addr");
        let c = IceCandidate::relay(relay, local_addr(), 1);
        assert_eq!(c.candidate_type, CandidateType::Relay);
    }

    // 6. Priority ordering: Host > ServerReflexive > Relay
    #[test]
    fn test_candidate_priority_ordering() {
        let host = IceCandidate::host(local_addr(), 1);
        let srflx = IceCandidate::server_reflexive(remote_addr(), local_addr(), 1);
        let relay = IceCandidate::relay(remote_addr(), local_addr(), 1);
        assert!(host.priority > srflx.priority);
        assert!(srflx.priority > relay.priority);
    }

    // 7. SDP candidate line contains required fields
    #[test]
    fn test_candidate_to_sdp() {
        let c = IceCandidate::host(local_addr(), 1);
        let sdp = c.to_sdp();
        assert!(sdp.starts_with("candidate:"));
        assert!(sdp.contains("host"));
        assert!(sdp.contains("udp"));
    }

    // 8. IceConfig defaults
    #[test]
    fn test_ice_config_defaults() {
        let cfg = IceConfig::default();
        assert_eq!(cfg.role, IceRole::Controlling);
        assert!(cfg.trickle);
        assert!(cfg.stun_servers.is_empty());
    }

    // 9. IceConfig builder
    #[test]
    fn test_ice_config_builder() {
        let stun: SocketAddr = "8.8.8.8:3478".parse().expect("valid addr");
        let cfg = IceConfig::new()
            .with_stun(stun)
            .with_role(IceRole::Controlled);
        assert_eq!(cfg.role, IceRole::Controlled);
        assert_eq!(cfg.stun_servers.len(), 1);
    }

    // 10. IceAgent initial state
    #[test]
    fn test_ice_agent_initial_state() {
        let agent = IceAgent::new(IceConfig::default());
        assert_eq!(agent.state(), IceState::New);
        assert!(agent.local_candidates().is_empty());
        assert!(agent.remote_candidates().is_empty());
    }

    // 11. IceAgent start_gathering
    #[test]
    fn test_ice_agent_start_gathering() {
        let mut agent = IceAgent::new(IceConfig::default());
        agent.start_gathering();
        assert_eq!(agent.state(), IceState::Gathering);
    }

    // 12. Adding candidates creates pairs
    #[test]
    fn test_ice_agent_pair_creation() {
        let mut agent = IceAgent::new(IceConfig::default());
        agent.add_local_candidate(IceCandidate::host(local_addr(), 1));
        agent.add_remote_candidate(IceCandidate::host(remote_addr(), 1));
        assert_eq!(agent.candidate_pairs().len(), 1);
    }

    // 13. start_checks unfreezes pairs
    #[test]
    fn test_ice_agent_start_checks() {
        let mut agent = IceAgent::new(IceConfig::default());
        agent.add_local_candidate(IceCandidate::host(local_addr(), 1));
        agent.add_remote_candidate(IceCandidate::host(remote_addr(), 1));
        agent.start_checks();
        assert_eq!(agent.pairs_in_state(PairState::Waiting), 1);
    }

    // 14. Successful connectivity check transitions state
    #[test]
    fn test_ice_agent_check_success() {
        let mut agent = IceAgent::new(IceConfig::default());
        agent.add_local_candidate(IceCandidate::host(local_addr(), 1));
        agent.add_remote_candidate(IceCandidate::host(remote_addr(), 1));
        agent.start_checks();
        agent.perform_next_check(CheckResult::Success);
        assert_eq!(agent.state(), IceState::Connected);
    }

    // 15. Failed connectivity check marks pair as failed
    #[test]
    fn test_ice_agent_check_failure() {
        let mut agent = IceAgent::new(IceConfig::default());
        agent.add_local_candidate(IceCandidate::host(local_addr(), 1));
        agent.add_remote_candidate(IceCandidate::host(remote_addr(), 1));
        agent.start_checks();
        agent.perform_next_check(CheckResult::Timeout);
        assert_eq!(agent.pairs_in_state(PairState::Failed), 1);
        assert_eq!(agent.state(), IceState::Failed);
    }

    // 16. Nomination by controlling agent
    #[test]
    fn test_ice_agent_nomination() {
        let mut agent = IceAgent::new(IceConfig::new().with_role(IceRole::Controlling));
        agent.add_local_candidate(IceCandidate::host(local_addr(), 1));
        agent.add_remote_candidate(IceCandidate::host(remote_addr(), 1));
        agent.start_checks();
        agent.perform_next_check(CheckResult::Success);
        let key = agent.nominate_best_pair();
        assert!(key.is_some());
        assert_eq!(agent.state(), IceState::Completed);
        assert!(agent.nominated_pair().is_some());
    }

    // 17. Controlled agent cannot nominate
    #[test]
    fn test_ice_agent_controlled_cannot_nominate() {
        let mut agent = IceAgent::new(IceConfig::new().with_role(IceRole::Controlled));
        agent.add_local_candidate(IceCandidate::host(local_addr(), 1));
        agent.add_remote_candidate(IceCandidate::host(remote_addr(), 1));
        agent.start_checks();
        agent.perform_next_check(CheckResult::Success);
        assert!(agent.nominate_best_pair().is_none());
    }

    // 18. Close agent
    #[test]
    fn test_ice_agent_close() {
        let mut agent = IceAgent::new(IceConfig::default());
        agent.close();
        assert_eq!(agent.state(), IceState::Closed);
    }

    // 19. Local ufrag and password are non-empty
    #[test]
    fn test_ice_agent_credentials() {
        let agent = IceAgent::new(IceConfig::default());
        assert!(!agent.local_ufrag().is_empty());
        assert!(!agent.local_pwd().is_empty());
    }

    // 20. IceRole names
    #[test]
    fn test_ice_role_names() {
        assert_eq!(IceRole::Controlling.name(), "controlling");
        assert_eq!(IceRole::Controlled.name(), "controlled");
    }

    // 21. Different component candidates don't pair
    #[test]
    fn test_ice_different_components_dont_pair() {
        let mut agent = IceAgent::new(IceConfig::default());
        agent.add_local_candidate(IceCandidate::host(local_addr(), 1)); // RTP
        agent.add_remote_candidate(IceCandidate::host(remote_addr(), 2)); // RTCP
        assert_eq!(agent.candidate_pairs().len(), 0);
    }
}
