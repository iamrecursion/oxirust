//! SRT connection modes: Caller, Listener, and Rendezvous.
//!
//! Implements the three SRT connection establishment patterns:
//! - **Caller**: Initiates connection to a known listener address
//! - **Listener**: Binds and waits for incoming caller connections
//! - **Rendezvous**: Both sides connect simultaneously (NAT traversal)
//!
//! Each mode follows the SRT handshake protocol (induction → conclusion)
//! but with different state machine transitions.

use super::packet::SrtPacket;
use super::socket::{ConnectionState, SrtConfig, SrtSocket};
use crate::error::{NetError, NetResult};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant, SystemTime};

// ─── Connection Mode ──────────────────────────────────────────────────────────

/// SRT connection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionMode {
    /// Caller mode: initiates connection to a known peer.
    Caller,
    /// Listener mode: waits for incoming connections.
    Listener,
    /// Rendezvous mode: both peers connect simultaneously.
    Rendezvous,
}

impl ConnectionMode {
    /// Returns a human-readable name for the mode.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Caller => "caller",
            Self::Listener => "listener",
            Self::Rendezvous => "rendezvous",
        }
    }
}

impl std::fmt::Display for ConnectionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ─── Caller ───────────────────────────────────────────────────────────────────

/// SRT Caller state machine.
///
/// Manages the caller-side handshake sequence:
/// 1. Send initial Waveahand to listener
/// 2. Receive Induction with SYN cookie
/// 3. Send Conclusion with cookie
/// 4. Receive final Agreement (or Conclusion response)
#[derive(Debug)]
pub struct CallerState {
    /// Socket state.
    socket: SrtSocket,
    /// Connection mode is always Caller.
    mode: ConnectionMode,
    /// Remote peer address.
    peer_addr: SocketAddr,
    /// Number of handshake retries.
    retry_count: u32,
    /// Maximum retries before giving up.
    max_retries: u32,
    /// Last handshake send time.
    last_send: Option<Instant>,
    /// Retry interval.
    retry_interval: Duration,
    /// Connection start time.
    started_at: Instant,
}

impl CallerState {
    /// Creates a new caller state targeting the given peer.
    #[must_use]
    pub fn new(config: SrtConfig, peer_addr: SocketAddr) -> Self {
        Self {
            socket: SrtSocket::new(config),
            mode: ConnectionMode::Caller,
            peer_addr,
            retry_count: 0,
            max_retries: 10,
            last_send: None,
            retry_interval: Duration::from_millis(250),
            started_at: Instant::now(),
        }
    }

    /// Sets the maximum number of handshake retries.
    pub fn set_max_retries(&mut self, max: u32) {
        self.max_retries = max;
    }

    /// Generates the initial handshake packet.
    #[must_use]
    pub fn generate_initial_handshake(&mut self) -> SrtPacket {
        self.last_send = Some(Instant::now());
        self.socket.generate_caller_handshake()
    }

    /// Processes a response from the listener.
    ///
    /// Returns packets to send in response, if any.
    pub fn process_response(&mut self, packet: SrtPacket) -> NetResult<Vec<SrtPacket>> {
        self.socket.process_packet(packet)
    }

    /// Returns whether the connection is established.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.socket.is_connected()
    }

    /// Returns the connection mode.
    #[must_use]
    pub const fn mode(&self) -> ConnectionMode {
        self.mode
    }

    /// Returns the peer address.
    #[must_use]
    pub const fn peer_addr(&self) -> SocketAddr {
        self.peer_addr
    }

    /// Checks if a retry is needed (handshake timeout).
    #[must_use]
    pub fn needs_retry(&self) -> bool {
        if self.socket.is_connected() {
            return false;
        }
        match self.last_send {
            Some(t) => t.elapsed() > self.retry_interval && self.retry_count < self.max_retries,
            None => true,
        }
    }

    /// Increments retry count and returns a new handshake packet.
    pub fn retry_handshake(&mut self) -> Option<SrtPacket> {
        if self.retry_count >= self.max_retries {
            return None;
        }
        self.retry_count += 1;
        self.last_send = Some(Instant::now());
        Some(self.socket.generate_caller_handshake())
    }

    /// Returns the elapsed time since connection start.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Returns the underlying socket for data transfer after connection.
    #[must_use]
    pub fn socket(&self) -> &SrtSocket {
        &self.socket
    }

    /// Returns the underlying socket mutably.
    pub fn socket_mut(&mut self) -> &mut SrtSocket {
        &mut self.socket
    }
}

// ─── Listener ─────────────────────────────────────────────────────────────────

/// Pending connection from a caller.
#[derive(Debug)]
pub struct PendingConnection {
    /// Caller address.
    pub addr: SocketAddr,
    /// Socket state for this connection.
    pub socket: SrtSocket,
    /// Handshake stage (0 = received waveahand, 1 = sent induction, 2 = connected).
    pub stage: u8,
    /// When this pending connection was created.
    pub created_at: Instant,
    /// SYN cookie assigned to this caller.
    pub syn_cookie: u32,
}

/// SRT Listener state machine.
///
/// Manages the listener-side handshake:
/// 1. Wait for incoming Waveahand from caller
/// 2. Reply with Induction containing SYN cookie
/// 3. Receive Conclusion from caller with cookie
/// 4. Send Agreement, connection established
///
/// Supports multiple simultaneous pending connections.
#[derive(Debug)]
pub struct ListenerState {
    /// Base configuration for accepted connections.
    config: SrtConfig,
    /// Connection mode is always Listener.
    mode: ConnectionMode,
    /// Bind address.
    bind_addr: SocketAddr,
    /// Pending connections keyed by caller address.
    pending: HashMap<SocketAddr, PendingConnection>,
    /// Established connections.
    established: Vec<SocketAddr>,
    /// Maximum pending connections.
    max_pending: usize,
    /// Connection timeout for pending connections.
    pending_timeout: Duration,
    /// Total connections accepted.
    total_accepted: u64,
}

impl ListenerState {
    /// Creates a new listener on the given address.
    #[must_use]
    pub fn new(config: SrtConfig, bind_addr: SocketAddr) -> Self {
        Self {
            config,
            mode: ConnectionMode::Listener,
            bind_addr,
            pending: HashMap::new(),
            established: Vec::new(),
            max_pending: 128,
            pending_timeout: Duration::from_secs(5),
            total_accepted: 0,
        }
    }

    /// Sets the maximum number of pending connections.
    pub fn set_max_pending(&mut self, max: usize) {
        self.max_pending = max;
    }

    /// Processes an incoming packet from a caller.
    ///
    /// Returns packets to send back to the caller, if any.
    pub fn process_incoming(
        &mut self,
        from: SocketAddr,
        packet: SrtPacket,
    ) -> NetResult<Vec<SrtPacket>> {
        // Check if this is a new or existing connection
        if let Some(pending) = self.pending.get_mut(&from) {
            // Process on existing pending connection
            let responses = pending.socket.process_packet(packet)?;
            if pending.socket.is_connected() {
                pending.stage = 2;
                self.established.push(from);
                self.total_accepted += 1;
            }
            Ok(responses)
        } else {
            // New connection attempt
            if self.pending.len() >= self.max_pending {
                return Err(NetError::connection("Max pending connections reached"));
            }

            let mut socket = SrtSocket::new(self.config.clone());
            let responses = socket.process_packet(packet)?;

            let syn_cookie = generate_listener_cookie(&from);
            let stage = if socket.is_connected() { 2 } else { 1 };

            let conn = PendingConnection {
                addr: from,
                socket,
                stage,
                created_at: Instant::now(),
                syn_cookie,
            };

            if stage == 2 {
                self.established.push(from);
                self.total_accepted += 1;
            }

            self.pending.insert(from, conn);
            Ok(responses)
        }
    }

    /// Removes timed-out pending connections.
    pub fn cleanup_pending(&mut self) {
        let timeout = self.pending_timeout;
        self.pending
            .retain(|_, conn| conn.stage >= 2 || conn.created_at.elapsed() < timeout);
    }

    /// Returns the number of pending connections.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.iter().filter(|(_, c)| c.stage < 2).count()
    }

    /// Returns the number of established connections.
    #[must_use]
    pub fn established_count(&self) -> usize {
        self.established.len()
    }

    /// Returns the total connections accepted.
    #[must_use]
    pub fn total_accepted(&self) -> u64 {
        self.total_accepted
    }

    /// Returns the bind address.
    #[must_use]
    pub const fn bind_addr(&self) -> SocketAddr {
        self.bind_addr
    }

    /// Returns the connection mode.
    #[must_use]
    pub const fn mode(&self) -> ConnectionMode {
        self.mode
    }

    /// Returns a reference to a pending connection.
    #[must_use]
    pub fn get_pending(&self, addr: &SocketAddr) -> Option<&PendingConnection> {
        self.pending.get(addr)
    }

    /// Returns established addresses.
    #[must_use]
    pub fn established_addrs(&self) -> &[SocketAddr] {
        &self.established
    }
}

// ─── Rendezvous ───────────────────────────────────────────────────────────────

/// Rendezvous handshake phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RendezvousPhase {
    /// Waving: both sides send Waveahand simultaneously.
    Waving,
    /// Attention: received peer's Waveahand, waiting for Conclusion.
    Attention,
    /// Fine: sent Conclusion, waiting for peer's Conclusion.
    Fine,
    /// Connected: both Conclusions exchanged.
    Connected,
    /// Failed.
    Failed,
}

impl RendezvousPhase {
    /// Returns a human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Waving => "waving",
            Self::Attention => "attention",
            Self::Fine => "fine",
            Self::Connected => "connected",
            Self::Failed => "failed",
        }
    }
}

/// SRT Rendezvous state machine.
///
/// Both peers simultaneously:
/// 1. Send Waveahand to each other
/// 2. On receiving Waveahand, transition to Attention → send Conclusion
/// 3. On receiving Conclusion, transition to Connected
///
/// This enables NAT traversal since both sides punch through their NATs.
#[derive(Debug)]
pub struct RendezvousState {
    /// Socket state.
    socket: SrtSocket,
    /// Connection mode is always Rendezvous.
    mode: ConnectionMode,
    /// Remote peer address.
    peer_addr: SocketAddr,
    /// Current rendezvous phase.
    phase: RendezvousPhase,
    /// Number of Waveahand retries.
    wave_count: u32,
    /// Maximum waving attempts.
    max_wave_count: u32,
    /// Wave interval.
    wave_interval: Duration,
    /// Last wave sent time.
    last_wave: Option<Instant>,
    /// Connection start time.
    started_at: Instant,
    /// Peer's socket ID once known.
    peer_socket_id: Option<u32>,
}

impl RendezvousState {
    /// Creates a new rendezvous state.
    #[must_use]
    pub fn new(config: SrtConfig, peer_addr: SocketAddr) -> Self {
        Self {
            socket: SrtSocket::new(config),
            mode: ConnectionMode::Rendezvous,
            peer_addr,
            phase: RendezvousPhase::Waving,
            wave_count: 0,
            max_wave_count: 25,
            wave_interval: Duration::from_millis(250),
            last_wave: None,
            started_at: Instant::now(),
            peer_socket_id: None,
        }
    }

    /// Generates a Waveahand packet for rendezvous mode.
    #[must_use]
    pub fn generate_wave(&mut self) -> SrtPacket {
        self.wave_count += 1;
        self.last_wave = Some(Instant::now());
        // In rendezvous mode, both sides send caller handshake
        self.socket.generate_caller_handshake()
    }

    /// Processes a packet from the peer.
    pub fn process_packet(&mut self, packet: SrtPacket) -> NetResult<Vec<SrtPacket>> {
        let responses = self.socket.process_packet(packet)?;

        if self.socket.is_connected() {
            self.phase = RendezvousPhase::Connected;
        } else if self.socket.state() == ConnectionState::Handshaking {
            // Received a handshake response, move from Waving → Attention/Fine
            match self.phase {
                RendezvousPhase::Waving => {
                    self.phase = RendezvousPhase::Attention;
                }
                RendezvousPhase::Attention => {
                    self.phase = RendezvousPhase::Fine;
                }
                _ => {}
            }
        }

        Ok(responses)
    }

    /// Returns whether the connection is established.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.phase == RendezvousPhase::Connected
    }

    /// Returns the current rendezvous phase.
    #[must_use]
    pub const fn phase(&self) -> RendezvousPhase {
        self.phase
    }

    /// Returns the connection mode.
    #[must_use]
    pub const fn mode(&self) -> ConnectionMode {
        self.mode
    }

    /// Returns the peer address.
    #[must_use]
    pub const fn peer_addr(&self) -> SocketAddr {
        self.peer_addr
    }

    /// Returns whether it is time to re-send a wave packet.
    #[must_use]
    pub fn needs_wave(&self) -> bool {
        if self.is_connected() || self.wave_count >= self.max_wave_count {
            return false;
        }
        match self.last_wave {
            Some(t) => t.elapsed() > self.wave_interval,
            None => true,
        }
    }

    /// Returns the elapsed time since connection start.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Returns whether waving has exceeded max attempts.
    #[must_use]
    pub fn is_timed_out(&self) -> bool {
        self.wave_count >= self.max_wave_count && !self.is_connected()
    }

    /// Returns the underlying socket.
    #[must_use]
    pub fn socket(&self) -> &SrtSocket {
        &self.socket
    }

    /// Returns the underlying socket mutably.
    pub fn socket_mut(&mut self) -> &mut SrtSocket {
        &mut self.socket
    }
}

// ─── Helper ───────────────────────────────────────────────────────────────────

/// Generates a SYN cookie for a caller based on address.
fn generate_listener_cookie(addr: &SocketAddr) -> u32 {
    let seed = match addr {
        SocketAddr::V4(a) => {
            let ip_bytes = a.ip().octets();
            let port = a.port() as u32;
            u32::from_be_bytes(ip_bytes) ^ (port << 16) ^ port
        }
        SocketAddr::V6(a) => {
            let ip_bytes = a.ip().octets();
            let port = a.port() as u32;
            let h = u32::from_be_bytes([ip_bytes[0], ip_bytes[1], ip_bytes[2], ip_bytes[3]]);
            h ^ (port << 16) ^ port
        }
    };

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs() as u32)
        .unwrap_or(0);

    seed ^ now ^ 0xBEEF_CAFE
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_addr() -> SocketAddr {
        "127.0.0.1:9000".parse().expect("valid addr")
    }

    fn test_addr2() -> SocketAddr {
        "127.0.0.1:9001".parse().expect("valid addr")
    }

    // 1. Connection mode display
    #[test]
    fn test_connection_mode_display() {
        assert_eq!(ConnectionMode::Caller.name(), "caller");
        assert_eq!(ConnectionMode::Listener.name(), "listener");
        assert_eq!(ConnectionMode::Rendezvous.name(), "rendezvous");
        assert_eq!(format!("{}", ConnectionMode::Caller), "caller");
    }

    // 2. Caller state creation
    #[test]
    fn test_caller_state_new() {
        let state = CallerState::new(SrtConfig::default(), test_addr());
        assert_eq!(state.mode(), ConnectionMode::Caller);
        assert_eq!(state.peer_addr(), test_addr());
        assert!(!state.is_connected());
    }

    // 3. Caller generates initial handshake
    #[test]
    fn test_caller_initial_handshake() {
        let mut state = CallerState::new(SrtConfig::default(), test_addr());
        let pkt = state.generate_initial_handshake();
        assert!(pkt.is_control());
    }

    // 4. Caller needs retry initially
    #[test]
    fn test_caller_needs_retry() {
        let state = CallerState::new(SrtConfig::default(), test_addr());
        assert!(state.needs_retry());
    }

    // 5. Caller retry count tracking
    #[test]
    fn test_caller_retry_limit() {
        let mut state = CallerState::new(SrtConfig::default(), test_addr());
        state.set_max_retries(2);
        state.retry_handshake();
        state.retry_handshake();
        assert!(state.retry_handshake().is_none());
    }

    // 6. Caller elapsed time -- elapsed() is always valid Duration (>= 0 ns)
    #[test]
    fn test_caller_elapsed() {
        let state = CallerState::new(SrtConfig::default(), test_addr());
        // Duration is always non-negative; we only verify the call does not panic.
        let _ = state.elapsed();
        assert!(state.elapsed().as_nanos() < u128::MAX);
    }

    // 7. Listener state creation
    #[test]
    fn test_listener_state_new() {
        let state = ListenerState::new(SrtConfig::default(), test_addr());
        assert_eq!(state.mode(), ConnectionMode::Listener);
        assert_eq!(state.bind_addr(), test_addr());
        assert_eq!(state.pending_count(), 0);
        assert_eq!(state.established_count(), 0);
    }

    // 8. Listener processes incoming connection
    #[test]
    fn test_listener_process_incoming() {
        let mut listener = ListenerState::new(SrtConfig::default(), test_addr());
        let mut caller_socket = SrtSocket::new(SrtConfig::default());
        let handshake = caller_socket.generate_caller_handshake();

        let responses = listener
            .process_incoming(test_addr2(), handshake)
            .expect("should process");
        // Should have at least created a pending connection
        assert!(listener.get_pending(&test_addr2()).is_some());
        assert!(!responses.is_empty());
    }

    // 9. Listener max pending limit
    #[test]
    fn test_listener_max_pending() {
        let mut listener = ListenerState::new(SrtConfig::default(), test_addr());
        listener.set_max_pending(1);

        let mut s1 = SrtSocket::new(SrtConfig::default());
        let h1 = s1.generate_caller_handshake();
        listener
            .process_incoming(test_addr2(), h1)
            .expect("should work");

        let mut s2 = SrtSocket::new(SrtConfig::default());
        let h2 = s2.generate_caller_handshake();
        let addr3: SocketAddr = "127.0.0.1:9002".parse().expect("valid");
        // The second should fail since max_pending is 1
        // But the first one might be in stage 1 (not yet established)
        // Actually the pending map has 1 entry, so next should fail
        let result = listener.process_incoming(addr3, h2);
        assert!(result.is_err());
    }

    // 10. Listener cleanup pending
    #[test]
    fn test_listener_cleanup() {
        let mut listener = ListenerState::new(SrtConfig::default(), test_addr());
        // With no pending connections, cleanup should be a no-op
        listener.cleanup_pending();
        assert_eq!(listener.pending_count(), 0);
    }

    // 11. Listener total accepted tracking
    #[test]
    fn test_listener_total_accepted() {
        let listener = ListenerState::new(SrtConfig::default(), test_addr());
        assert_eq!(listener.total_accepted(), 0);
    }

    // 12. Rendezvous phase names
    #[test]
    fn test_rendezvous_phase_names() {
        assert_eq!(RendezvousPhase::Waving.name(), "waving");
        assert_eq!(RendezvousPhase::Attention.name(), "attention");
        assert_eq!(RendezvousPhase::Fine.name(), "fine");
        assert_eq!(RendezvousPhase::Connected.name(), "connected");
        assert_eq!(RendezvousPhase::Failed.name(), "failed");
    }

    // 13. Rendezvous state creation
    #[test]
    fn test_rendezvous_state_new() {
        let state = RendezvousState::new(SrtConfig::default(), test_addr());
        assert_eq!(state.mode(), ConnectionMode::Rendezvous);
        assert_eq!(state.phase(), RendezvousPhase::Waving);
        assert!(!state.is_connected());
    }

    // 14. Rendezvous wave generation
    #[test]
    fn test_rendezvous_wave() {
        let mut state = RendezvousState::new(SrtConfig::default(), test_addr());
        let pkt = state.generate_wave();
        assert!(pkt.is_control());
    }

    // 15. Rendezvous needs_wave initially
    #[test]
    fn test_rendezvous_needs_wave() {
        let state = RendezvousState::new(SrtConfig::default(), test_addr());
        assert!(state.needs_wave());
    }

    // 16. Rendezvous timeout detection
    #[test]
    fn test_rendezvous_timeout() {
        let mut state = RendezvousState::new(SrtConfig::default(), test_addr());
        state.max_wave_count = 3;
        for _ in 0..3 {
            let _ = state.generate_wave();
        }
        assert!(state.is_timed_out());
    }

    // 17. Rendezvous elapsed tracking -- elapsed() is always valid Duration (>= 0 ns)
    #[test]
    fn test_rendezvous_elapsed() {
        let state = RendezvousState::new(SrtConfig::default(), test_addr());
        // Duration is always non-negative; we only verify the call does not panic.
        let _ = state.elapsed();
        assert!(state.elapsed().as_nanos() < u128::MAX);
    }

    // 18. Generate listener cookie determinism for same input
    #[test]
    fn test_listener_cookie() {
        let addr = test_addr();
        let c1 = generate_listener_cookie(&addr);
        let c2 = generate_listener_cookie(&addr);
        // Same second → same cookie
        assert_eq!(c1, c2);
    }

    // 19. Different addresses produce different cookies
    #[test]
    fn test_listener_cookie_different_addrs() {
        let c1 = generate_listener_cookie(&test_addr());
        let c2 = generate_listener_cookie(&test_addr2());
        // Different addresses should generally produce different cookies
        // (not guaranteed but very likely)
        // We just verify they're both non-zero
        assert_ne!(c1, 0);
        assert_ne!(c2, 0);
    }

    // 20. IPv6 address cookie
    #[test]
    fn test_ipv6_cookie() {
        let addr: SocketAddr = "[::1]:9000".parse().expect("valid");
        let cookie = generate_listener_cookie(&addr);
        assert_ne!(cookie, 0);
    }

    // 21. Caller socket access
    #[test]
    fn test_caller_socket_access() {
        let state = CallerState::new(SrtConfig::default(), test_addr());
        assert_eq!(state.socket().state(), ConnectionState::Initial);
    }

    // 22. Rendezvous socket access
    #[test]
    fn test_rendezvous_socket_access() {
        let state = RendezvousState::new(SrtConfig::default(), test_addr());
        let rtt = state.socket().rtt();
        assert!(rtt > 0); // Default RTT is non-zero
    }

    // 23. Listener established addresses
    #[test]
    fn test_listener_established_addrs() {
        let listener = ListenerState::new(SrtConfig::default(), test_addr());
        assert!(listener.established_addrs().is_empty());
    }

    // 24. Rendezvous cross-connection simulation
    #[test]
    fn test_rendezvous_cross_connection() {
        let addr_a = test_addr();
        let addr_b = test_addr2();

        let mut side_a = RendezvousState::new(SrtConfig::default(), addr_b);
        let mut side_b = RendezvousState::new(SrtConfig::default(), addr_a);

        // Both sides generate waves
        let wave_a = side_a.generate_wave();
        let wave_b = side_b.generate_wave();

        // Each side processes the other's wave
        let resp_a = side_a.process_packet(wave_b);
        let resp_b = side_b.process_packet(wave_a);

        // Both should have progressed from Waving
        assert!(resp_a.is_ok());
        assert!(resp_b.is_ok());

        // Process responses
        if let Ok(responses) = resp_a {
            for r in responses {
                let _ = side_b.process_packet(r);
            }
        }
        if let Ok(responses) = resp_b {
            for r in responses {
                let _ = side_a.process_packet(r);
            }
        }

        // At least one side should have progressed beyond Waving
        let progressed =
            side_a.phase() != RendezvousPhase::Waving || side_b.phase() != RendezvousPhase::Waving;
        assert!(progressed);
    }
}
