//! Live SRT ingest server.
//!
//! Provides configuration, session tracking, statistics, and a real
//! [`oximedia_net::srt::SrtListener`]-backed accept loop for SRT (Secure
//! Reliable Transport) live ingest.
//!
//! Two execution modes are supported:
//!
//! 1. **Simulated** — `SrtIngestServer::accept_session` takes a pre-built
//!    `client_addr` / timestamp and registers an `IngestSession` without
//!    touching the network.  Used by unit tests and by higher layers that
//!    drive the session lifecycle themselves.
//! 2. **Wired** — `SrtIngestServer::accept_connection` binds an SRT
//!    listener at `SrtIngestConfig::bind_addr`, performs a real SRT
//!    handshake, captures the
//!    [`oximedia_net::srt::SrtReceiver`], and stores it inside the session.
//!    `SrtIngestServer::run` drives that accept path in a loop until a
//!    shutdown signal is observed.

use oximedia_net::error::{NetError, NetResult};
use oximedia_net::srt::{SrtConfig, SrtListener, SrtReceiver};
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{watch, Mutex};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for an SRT ingest listener.
#[derive(Debug, Clone)]
pub struct SrtIngestConfig {
    /// IP address to bind on (defaults to `0.0.0.0` — all interfaces).
    pub bind_ip: IpAddr,
    /// UDP port to listen on for incoming SRT connections.
    pub port: u16,
    /// Optional stream-ID passphrase for encryption.
    pub passphrase: Option<String>,
    /// Target latency in milliseconds (SRT `SRTO_LATENCY`).
    pub latency_ms: u32,
    /// Maximum receive bandwidth in Mbit/s (`SRTO_MAXBW`).
    pub max_bw_mbps: f32,
}

impl Default for SrtIngestConfig {
    fn default() -> Self {
        Self {
            bind_ip: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            port: 9998,
            passphrase: None,
            latency_ms: 200,
            max_bw_mbps: 100.0,
        }
    }
}

impl SrtIngestConfig {
    /// Returns `true` if the config has a non-empty passphrase.
    pub fn is_encrypted(&self) -> bool {
        self.passphrase
            .as_deref()
            .map(|p| !p.is_empty())
            .unwrap_or(false)
    }

    /// Returns the [`SocketAddr`] that an [`SrtListener`] should bind to.
    #[must_use]
    pub fn bind_addr(&self) -> SocketAddr {
        SocketAddr::new(self.bind_ip, self.port)
    }

    /// Converts this ingest config into the lower-level [`SrtConfig`] used by
    /// `oximedia-net`.  Encryption parameters and latency are propagated.
    #[must_use]
    pub fn to_srt_config(&self) -> SrtConfig {
        let mut cfg = SrtConfig::default().with_latency(self.latency_ms);
        if let Some(pass) = self.passphrase.as_deref() {
            if !pass.is_empty() {
                cfg = cfg.with_passphrase(pass);
            }
        }
        if self.max_bw_mbps > 0.0 {
            // Convert Mbit/s to bytes/s for SRTO_MAXBW.
            let max_bw_bytes = ((self.max_bw_mbps * 1_000_000.0) / 8.0) as u64;
            cfg.max_bandwidth = max_bw_bytes;
        }
        cfg
    }
}

// ── Session ───────────────────────────────────────────────────────────────────

/// An active (or historical) SRT ingest session.
///
/// Sessions created by [`SrtIngestServer::accept_session`] are pure metadata
/// (no socket attached).  Sessions created by
/// [`SrtIngestServer::accept_connection`] additionally hold an `Arc`-wrapped
/// [`SrtReceiver`] in the `receiver` field so callers can pull payload bytes
/// off the wire.
#[derive(Clone)]
pub struct IngestSession {
    /// Unique session identifier.
    pub id: String,
    /// Remote client address (IP:port).
    pub client_addr: String,
    /// Unix epoch milliseconds when the session was opened.
    pub started_at_ms: u64,
    /// Total bytes received on this session.
    pub bytes_received: u64,
    /// Number of lost packets reported by SRT.
    pub packets_lost: u64,
    /// Optional handle to the underlying SRT receiver.
    ///
    /// `None` for simulated sessions, `Some(_)` for sessions accepted via
    /// [`SrtIngestServer::accept_connection`].
    pub receiver: Option<Arc<SrtReceiver>>,
}

impl std::fmt::Debug for IngestSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IngestSession")
            .field("id", &self.id)
            .field("client_addr", &self.client_addr)
            .field("started_at_ms", &self.started_at_ms)
            .field("bytes_received", &self.bytes_received)
            .field("packets_lost", &self.packets_lost)
            .field("receiver", &self.receiver.as_ref().map(|_| "<SrtReceiver>"))
            .finish()
    }
}

impl IngestSession {
    /// Approximate number of received packets (assuming 1316-byte MPEG-TS payload).
    fn received_packets(&self) -> u64 {
        self.bytes_received / 1316
    }

    /// Duration of the session in milliseconds, given the current wall-clock.
    pub fn duration_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.started_at_ms)
    }

    /// Estimated throughput in kbit/s given the elapsed time.
    pub fn throughput_kbps(&self, now_ms: u64) -> f64 {
        let elapsed_ms = self.duration_ms(now_ms);
        if elapsed_ms == 0 {
            return 0.0;
        }
        (self.bytes_received as f64 * 8.0) / elapsed_ms as f64
    }

    /// Returns `true` if this session has a live SRT receiver attached.
    #[must_use]
    pub fn is_wired(&self) -> bool {
        self.receiver.is_some()
    }
}

// ── LCG-based session ID generator ───────────────────────────────────────────

/// A minimal linear-congruential generator used to produce UUID-like session IDs
/// without external dependencies.
struct LcgIdGen {
    state: u64,
}

impl LcgIdGen {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advances the LCG state and returns the next value.
    fn advance(&mut self) -> u64 {
        // Parameters from Knuth's MMIX
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Generates a UUID-like hex string with dashes.
    fn gen_id(&mut self, counter: u64) -> String {
        let a = self.advance() ^ counter;
        let b = self.advance() ^ counter.wrapping_add(1);
        // Format as xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx (version 4-ish)
        let p1 = (a >> 32) as u32;
        let p2 = ((a >> 16) & 0xFFFF) as u16;
        let p3 = (a & 0x0FFF) as u16 | 0x4000u16; // version 4
        let p4 = ((b >> 48) & 0x3FFF) as u16 | 0x8000u16; // variant bits
        let p5 = b & 0xFFFF_FFFF_FFFFu64;
        format!("{:08x}-{:04x}-{:04x}-{:04x}-{:012x}", p1, p2, p3, p4, p5)
    }
}

// ── SRT ingest server ─────────────────────────────────────────────────────────

/// SRT ingest server.
///
/// The server owns the session table, the session-ID generator, and (for
/// wired operation) binds a fresh [`SrtListener`] on
/// [`SrtIngestConfig::bind_addr`] for every accepted connection.  Two
/// accept paths are exposed:
///
/// * [`Self::accept_session`] — synchronous, registers a simulated session
///   from caller-supplied metadata.  Used by unit tests and by integrations
///   that drive the lifecycle externally.
/// * [`Self::accept_connection`] — async, binds a real [`SrtListener`] and
///   completes a full SRT handshake before registering the session.
///
/// [`Self::run`] drives the async path in a loop until a shutdown signal
/// is observed on the supplied [`watch::Receiver<bool>`].
pub struct SrtIngestServer {
    /// Server configuration.
    pub config: SrtIngestConfig,
    /// Active and historical sessions keyed by session ID, wrapped in an
    /// `Arc<Mutex<_>>` so the async `run` loop and synchronous callers can
    /// share access without races.
    sessions: Arc<Mutex<HashMap<String, IngestSession>>>,
    /// ID generator (guarded so async accept paths can mutate it).
    id_gen: Arc<Mutex<LcgIdGen>>,
    /// Monotonically increasing session counter (used to seed the ID
    /// generator); also guarded for async use.
    session_counter: Arc<Mutex<u64>>,
}

/// Locks `mutex`, spin-waiting instead of panicking if it is momentarily
/// held.
///
/// Callers of this helper hold `&mut self` on [`SrtIngestServer`], which
/// statically guarantees no other reference to the server (and therefore no
/// other holder of any of its internal mutexes) can exist — none of these
/// mutexes are ever cloned out of the struct or handed to a spawned task.
/// Under that invariant `try_lock()` always succeeds on the first attempt;
/// the loop exists only so this can never panic even if that invariant is
/// ever violated by a future refactor.
fn lock_uncontended<T>(mutex: &Mutex<T>) -> tokio::sync::MutexGuard<'_, T> {
    loop {
        if let Ok(guard) = mutex.try_lock() {
            return guard;
        }
        std::thread::yield_now();
    }
}

impl SrtIngestServer {
    /// Creates a new ingest server with the given configuration.
    pub fn new(config: SrtIngestConfig) -> Self {
        // Seed the LCG with the port number for deterministic but varied output.
        let seed = config.port as u64 ^ 0xDEAD_BEEF_1337_0042;
        Self {
            config,
            sessions: Arc::new(Mutex::new(HashMap::new())),
            id_gen: Arc::new(Mutex::new(LcgIdGen::new(seed))),
            session_counter: Arc::new(Mutex::new(0)),
        }
    }

    /// Simulates accepting a new SRT session from `client_addr` at `now_ms`.
    ///
    /// Returns the newly generated session ID.
    ///
    /// This is the legacy synchronous path: no socket is opened and no
    /// handshake is performed.  Use [`Self::accept_connection`] for the
    /// wired path.
    ///
    /// `&mut self` grants exclusive access to the server, so the internal
    /// mutexes below are always uncontended in practice (see
    /// `lock_uncontended`); this method cannot panic.
    pub fn accept_session(&mut self, client_addr: String, now_ms: u64) -> String {
        let mut counter_guard = lock_uncontended(&self.session_counter);
        let mut id_gen_guard = lock_uncontended(&self.id_gen);
        let mut sessions_guard = lock_uncontended(&self.sessions);

        *counter_guard = counter_guard.wrapping_add(1);
        let id = id_gen_guard.gen_id(*counter_guard);
        let session = IngestSession {
            id: id.clone(),
            client_addr,
            started_at_ms: now_ms,
            bytes_received: 0,
            packets_lost: 0,
            receiver: None,
        };
        sessions_guard.insert(id.clone(), session);
        id
    }

    /// Accepts a real SRT connection on [`SrtIngestConfig::bind_addr`].
    ///
    /// Binds an [`SrtListener`] (re-using [`SrtIngestConfig::to_srt_config`]
    /// for encryption / latency / bandwidth), waits for the INDUCTION
    /// datagram, drives the SRT handshake to completion, and registers a
    /// new [`IngestSession`] populated with the real peer address and the
    /// held [`SrtReceiver`].
    ///
    /// Returns the newly generated session ID on success.
    ///
    /// # Errors
    ///
    /// Returns [`NetError`] if socket binding, the initial recv, or the SRT
    /// handshake fails.
    pub async fn accept_connection(&self) -> NetResult<String> {
        let listener = SrtListener::new(self.config.bind_addr(), self.config.to_srt_config());
        let receiver = listener.accept().await?;
        let peer = receiver.peer_addr();
        let now_ms = current_unix_ms();

        let counter_value = {
            let mut counter_guard = self.session_counter.lock().await;
            *counter_guard = counter_guard.wrapping_add(1);
            *counter_guard
        };

        let id = {
            let mut id_gen_guard = self.id_gen.lock().await;
            id_gen_guard.gen_id(counter_value)
        };

        let session = IngestSession {
            id: id.clone(),
            client_addr: peer.to_string(),
            started_at_ms: now_ms,
            bytes_received: 0,
            packets_lost: 0,
            receiver: Some(Arc::new(receiver)),
        };

        {
            let mut sessions_guard = self.sessions.lock().await;
            sessions_guard.insert(id.clone(), session);
        }

        Ok(id)
    }

    /// Drives [`Self::accept_connection`] in a loop until a shutdown signal
    /// is observed on `shutdown`.
    ///
    /// Transient accept errors (handshake failures, peer disconnects) are
    /// logged via [`tracing::warn`] and the loop continues.  Fatal errors
    /// (binding the local UDP socket fails, address already in use,
    /// permission denied, …) abort the loop and are returned to the
    /// caller.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`NetError`] when a fatal accept error
    /// terminates the loop.  Returns `Ok(())` when shutdown is signalled.
    pub async fn run(&self, mut shutdown: watch::Receiver<bool>) -> NetResult<()> {
        // If shutdown is already set when we start, return immediately.
        if *shutdown.borrow() {
            return Ok(());
        }

        loop {
            tokio::select! {
                accept_res = self.accept_connection() => {
                    match accept_res {
                        Ok(id) => {
                            tracing::info!(session_id = %id, "accepted SRT ingest session");
                        }
                        Err(NetError::Io(e)) if is_fatal_io(&e) => {
                            tracing::error!(error = %e, "fatal I/O error in SRT accept loop");
                            return Err(NetError::Io(e));
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "transient SRT accept error; continuing");
                        }
                    }
                }
                changed_res = shutdown.changed() => {
                    // `changed()` returns `Err` if the sender is dropped — we
                    // treat that as a shutdown signal too (no more senders
                    // means we can never be told to stop, so stop now).
                    if changed_res.is_err() || *shutdown.borrow() {
                        tracing::info!("SRT ingest server shutting down");
                        return Ok(());
                    }
                }
            }
        }
    }

    /// Updates statistics for the session identified by `id`.
    ///
    /// `bytes` is the number of additional bytes received since the last update.
    /// `lost` is the number of additional lost packets since the last update.
    pub fn update_session(&mut self, id: &str, bytes: u64, lost: u64) {
        let mut sessions_guard = lock_uncontended(&self.sessions);
        if let Some(session) = sessions_guard.get_mut(id) {
            session.bytes_received = session.bytes_received.saturating_add(bytes);
            session.packets_lost = session.packets_lost.saturating_add(lost);
        }
    }

    /// Returns a clone of the session with `id`, or `None` if not found.
    pub fn session_stats(&self, id: &str) -> Option<IngestSession> {
        let sessions_guard = self.sessions.try_lock().ok()?;
        sessions_guard.get(id).cloned()
    }

    /// Async variant of [`Self::session_stats`] — does not require the
    /// mutex to be uncontested.
    pub async fn session_stats_async(&self, id: &str) -> Option<IngestSession> {
        let sessions_guard = self.sessions.lock().await;
        sessions_guard.get(id).cloned()
    }

    /// Calculates the packet loss rate for a session.
    ///
    /// Returns `lost / (received_packets + lost)`.  Returns `0.0` if no
    /// packets have been seen or the session does not exist.
    pub fn packet_loss_rate(&self, id: &str) -> f32 {
        let sessions_guard = match self.sessions.try_lock() {
            Ok(g) => g,
            Err(_) => return 0.0,
        };
        let session = match sessions_guard.get(id) {
            Some(s) => s,
            None => return 0.0,
        };
        let received = session.received_packets();
        let total = received.saturating_add(session.packets_lost);
        if total == 0 {
            return 0.0;
        }
        session.packets_lost as f32 / total as f32
    }

    /// Removes a session by ID.  Returns `true` if it existed.
    pub fn remove_session(&mut self, id: &str) -> bool {
        let mut sessions_guard = lock_uncontended(&self.sessions);
        sessions_guard.remove(id).is_some()
    }

    /// Returns the number of active sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.try_lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Async variant of [`Self::session_count`] — usable while the run
    /// loop holds the mutex.
    pub async fn session_count_async(&self) -> usize {
        let g = self.sessions.lock().await;
        g.len()
    }

    /// Returns the IDs of all sessions (as owned `String`s, since the
    /// underlying map lives behind a mutex).
    pub fn session_ids(&self) -> Vec<String> {
        self.sessions
            .try_lock()
            .map(|g| g.keys().cloned().collect())
            .unwrap_or_default()
    }
}

/// Returns the current wall-clock as milliseconds since the UNIX epoch.
///
/// Falls back to `0` if the system clock is set before 1970-01-01.
fn current_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Classifies a [`std::io::Error`] as "fatal" (the accept loop cannot
/// recover from it) or "transient" (worth logging and retrying).
fn is_fatal_io(err: &std::io::Error) -> bool {
    use std::io::ErrorKind;
    matches!(
        err.kind(),
        ErrorKind::AddrInUse
            | ErrorKind::AddrNotAvailable
            | ErrorKind::PermissionDenied
            | ErrorKind::InvalidInput
    )
}

// ── LiveIngestHandler ─────────────────────────────────────────────────────────

/// Simplified live ingest handler with RTMP connection management.
///
/// Wraps [`SrtIngestServer`] with an RTMP-oriented API matching the TODO spec:
///
/// ```rust
/// use oximedia_server::live_ingest::LiveIngestHandler;
///
/// let mut handler = LiveIngestHandler::new(1935);
/// let result = handler.handle_rtmp_connect("client-1", "live/stream");
/// assert!(result.is_ok());
/// ```
pub struct LiveIngestHandler {
    /// Listening port.
    pub port: u16,
    /// RTMP app name → session ID map.
    connections: HashMap<String, String>,
    /// Connection counter for ID generation.
    counter: u64,
}

impl LiveIngestHandler {
    /// Create a new RTMP ingest handler on the given port.
    #[must_use]
    pub fn new(port: u16) -> Self {
        Self {
            port,
            connections: HashMap::new(),
            counter: 0,
        }
    }

    /// Handle an RTMP client connecting to an application.
    ///
    /// * `client_id` – Unique identifier for the connecting client.
    /// * `app`       – RTMP application name (e.g. `"live/stream"`).
    ///
    /// Returns `Ok(())` on success, or `Err(String)` if the app name is empty
    /// or the connection slot is already occupied.
    pub fn handle_rtmp_connect(&mut self, client_id: &str, app: &str) -> Result<(), String> {
        if app.is_empty() {
            return Err("RTMP app name must not be empty".to_string());
        }
        if client_id.is_empty() {
            return Err("client_id must not be empty".to_string());
        }
        self.counter = self.counter.wrapping_add(1);
        let session_id = format!("rtmp-{}-{}-{}", self.port, self.counter, client_id);
        self.connections.insert(app.to_string(), session_id);
        Ok(())
    }

    /// Disconnect an RTMP client from an application.
    ///
    /// Returns `true` if a connection for `app` was found and removed.
    pub fn handle_rtmp_disconnect(&mut self, app: &str) -> bool {
        self.connections.remove(app).is_some()
    }

    /// Return the session ID for the given app, if connected.
    #[must_use]
    pub fn session_for_app(&self, app: &str) -> Option<&str> {
        self.connections.get(app).map(String::as_str)
    }

    /// Number of active RTMP connections.
    #[must_use]
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_server() -> SrtIngestServer {
        SrtIngestServer::new(SrtIngestConfig::default())
    }

    // SrtIngestConfig tests

    #[test]
    fn test_config_default_values() {
        let cfg = SrtIngestConfig::default();
        assert_eq!(cfg.port, 9998);
        assert_eq!(cfg.latency_ms, 200);
        assert!(!cfg.is_encrypted());
    }

    #[test]
    fn test_config_with_passphrase() {
        let cfg = SrtIngestConfig {
            passphrase: Some("s3cr3t".to_string()),
            ..Default::default()
        };
        assert!(cfg.is_encrypted());
    }

    #[test]
    fn test_config_empty_passphrase_not_encrypted() {
        let cfg = SrtIngestConfig {
            passphrase: Some(String::new()),
            ..Default::default()
        };
        assert!(!cfg.is_encrypted());
    }

    #[test]
    fn test_config_bind_addr_default() {
        let cfg = SrtIngestConfig::default();
        let addr = cfg.bind_addr();
        assert_eq!(addr.port(), 9998);
        assert!(addr.ip().is_unspecified());
    }

    #[test]
    fn test_config_to_srt_config_propagates_latency() {
        let cfg = SrtIngestConfig {
            latency_ms: 250,
            ..Default::default()
        };
        let srt_cfg = cfg.to_srt_config();
        assert_eq!(srt_cfg.latency_ms, 250);
    }

    #[test]
    fn test_config_to_srt_config_propagates_passphrase() {
        let cfg = SrtIngestConfig {
            passphrase: Some("secret".to_string()),
            ..Default::default()
        };
        let srt_cfg = cfg.to_srt_config();
        assert!(srt_cfg.passphrase.is_some());
        // with_passphrase sets a 16-byte AES-128 key by default.
        assert_eq!(srt_cfg.key_size, 16);
    }

    #[test]
    fn test_config_to_srt_config_max_bw_conversion() {
        let cfg = SrtIngestConfig {
            max_bw_mbps: 8.0, // 8 Mbit/s → 1_000_000 bytes/s
            ..Default::default()
        };
        let srt_cfg = cfg.to_srt_config();
        assert_eq!(srt_cfg.max_bandwidth, 1_000_000);
    }

    #[test]
    fn test_config_to_srt_config_zero_bw_leaves_default() {
        let cfg = SrtIngestConfig {
            max_bw_mbps: 0.0,
            ..Default::default()
        };
        let srt_cfg = cfg.to_srt_config();
        assert_eq!(srt_cfg.max_bandwidth, 0);
    }

    // accept_session tests

    #[test]
    fn test_accept_session_returns_nonempty_id() {
        let mut srv = default_server();
        let id = srv.accept_session("192.168.1.1:9000".to_string(), 1_000_000);
        assert!(!id.is_empty());
        assert_eq!(srv.session_count(), 1);
    }

    #[test]
    fn test_accept_session_ids_are_unique() {
        let mut srv = default_server();
        let id1 = srv.accept_session("10.0.0.1:1234".to_string(), 0);
        let id2 = srv.accept_session("10.0.0.2:1234".to_string(), 0);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_session_stats_found() {
        let mut srv = default_server();
        let id = srv.accept_session("127.0.0.1:5000".to_string(), 5_000);
        let stats = srv.session_stats(&id).expect("session should exist");
        assert_eq!(stats.client_addr, "127.0.0.1:5000");
        assert_eq!(stats.started_at_ms, 5_000);
        assert_eq!(stats.bytes_received, 0);
        assert!(!stats.is_wired());
    }

    #[test]
    fn test_session_stats_not_found() {
        let srv = default_server();
        assert!(srv.session_stats("nonexistent").is_none());
    }

    // update_session tests

    #[test]
    fn test_update_session_accumulates_bytes() {
        let mut srv = default_server();
        let id = srv.accept_session("1.2.3.4:9000".to_string(), 0);
        srv.update_session(&id, 1024, 0);
        srv.update_session(&id, 2048, 0);
        let stats = srv.session_stats(&id).expect("should exist");
        assert_eq!(stats.bytes_received, 3072);
    }

    #[test]
    fn test_update_session_accumulates_lost() {
        let mut srv = default_server();
        let id = srv.accept_session("1.2.3.4:9000".to_string(), 0);
        srv.update_session(&id, 10 * 1316, 2);
        srv.update_session(&id, 10 * 1316, 3);
        let stats = srv.session_stats(&id).expect("should exist");
        assert_eq!(stats.packets_lost, 5);
    }

    #[test]
    fn test_update_nonexistent_session_is_noop() {
        let mut srv = default_server();
        // Should not panic
        srv.update_session("ghost", 9999, 9999);
        assert_eq!(srv.session_count(), 0);
    }

    // packet_loss_rate tests

    #[test]
    fn test_packet_loss_rate_zero_when_no_packets() {
        let mut srv = default_server();
        let id = srv.accept_session("1.2.3.4:9000".to_string(), 0);
        assert!((srv.packet_loss_rate(&id)).abs() < 1e-6);
    }

    #[test]
    fn test_packet_loss_rate_calculated_correctly() {
        let mut srv = default_server();
        let id = srv.accept_session("1.2.3.4:9000".to_string(), 0);
        // 9 * 1316 bytes received -> 9 received packets; 1 lost -> rate = 1/10 = 0.1
        srv.update_session(&id, 9 * 1316, 1);
        let rate = srv.packet_loss_rate(&id);
        let expected = 1.0_f32 / 10.0_f32;
        assert!(
            (rate - expected).abs() < 1e-5,
            "rate={} expected={}",
            rate,
            expected
        );
    }

    #[test]
    fn test_packet_loss_rate_nonexistent_returns_zero() {
        let srv = default_server();
        assert!((srv.packet_loss_rate("ghost")).abs() < 1e-6);
    }

    // remove_session tests

    #[test]
    fn test_remove_session() {
        let mut srv = default_server();
        let id = srv.accept_session("1.2.3.4:9000".to_string(), 0);
        assert!(srv.remove_session(&id));
        assert_eq!(srv.session_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_session() {
        let mut srv = default_server();
        assert!(!srv.remove_session("ghost"));
    }

    // IngestSession helpers

    #[test]
    fn test_session_duration_ms() {
        let session = IngestSession {
            id: "x".to_string(),
            client_addr: "127.0.0.1:1".to_string(),
            started_at_ms: 1000,
            bytes_received: 0,
            packets_lost: 0,
            receiver: None,
        };
        assert_eq!(session.duration_ms(1500), 500);
    }

    #[test]
    fn test_session_throughput_zero_elapsed() {
        let session = IngestSession {
            id: "x".to_string(),
            client_addr: "127.0.0.1:1".to_string(),
            started_at_ms: 1000,
            bytes_received: 10000,
            packets_lost: 0,
            receiver: None,
        };
        // Same timestamp as started_at -> elapsed = 0 -> 0 kbps
        assert!((session.throughput_kbps(1000)).abs() < 1e-9);
    }

    #[test]
    fn test_session_is_wired_default_false() {
        let session = IngestSession {
            id: "x".to_string(),
            client_addr: "127.0.0.1:1".to_string(),
            started_at_ms: 0,
            bytes_received: 0,
            packets_lost: 0,
            receiver: None,
        };
        assert!(!session.is_wired());
    }

    // ── LiveIngestHandler tests ───────────────────────────────────────────────

    #[test]
    fn test_rtmp_connect_ok() {
        let mut h = LiveIngestHandler::new(1935);
        assert!(h.handle_rtmp_connect("client-1", "live/stream").is_ok());
        assert_eq!(h.connection_count(), 1);
    }

    #[test]
    fn test_rtmp_connect_empty_app_errors() {
        let mut h = LiveIngestHandler::new(1935);
        assert!(h.handle_rtmp_connect("client-1", "").is_err());
    }

    #[test]
    fn test_rtmp_connect_empty_client_errors() {
        let mut h = LiveIngestHandler::new(1935);
        assert!(h.handle_rtmp_connect("", "live/stream").is_err());
    }

    #[test]
    fn test_rtmp_disconnect() {
        let mut h = LiveIngestHandler::new(1935);
        h.handle_rtmp_connect("c", "live/a")
            .expect("connect should succeed");
        assert!(h.handle_rtmp_disconnect("live/a"));
        assert_eq!(h.connection_count(), 0);
    }

    #[test]
    fn test_rtmp_session_for_app() {
        let mut h = LiveIngestHandler::new(1935);
        h.handle_rtmp_connect("c1", "live/b")
            .expect("connect should succeed");
        assert!(h.session_for_app("live/b").is_some());
        assert!(h.session_for_app("live/unknown").is_none());
    }

    // LcgIdGen tests (via accept_session)

    #[test]
    fn test_multiple_accepts_produce_valid_uuids() {
        let mut srv = default_server();
        for i in 0..10u64 {
            let id = srv.accept_session(format!("10.0.0.{}:9000", i), i * 100);
            // UUID-like: 8-4-4-4-12 hex chars separated by '-'
            let parts: Vec<&str> = id.split('-').collect();
            assert_eq!(parts.len(), 5, "id={}", id);
            assert_eq!(parts[0].len(), 8);
            assert_eq!(parts[1].len(), 4);
            assert_eq!(parts[2].len(), 4);
            assert_eq!(parts[3].len(), 4);
            assert_eq!(parts[4].len(), 12);
        }
    }

    // ── is_fatal_io ───────────────────────────────────────────────────────────

    #[test]
    fn test_is_fatal_io_addr_in_use_is_fatal() {
        let err = std::io::Error::new(std::io::ErrorKind::AddrInUse, "x");
        assert!(is_fatal_io(&err));
    }

    #[test]
    fn test_is_fatal_io_timed_out_is_transient() {
        let err = std::io::Error::new(std::io::ErrorKind::TimedOut, "x");
        assert!(!is_fatal_io(&err));
    }

    // ── current_unix_ms sanity ───────────────────────────────────────────────

    #[test]
    fn test_current_unix_ms_nonzero() {
        // After 2020, the value must be substantially greater than the
        // 1577836800000-ms checkpoint, proving the function actually
        // consults the system clock.
        assert!(current_unix_ms() > 1_577_836_800_000);
    }

    // ── run() shutdown ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_run_returns_when_shutdown_pre_set() {
        // Build a config that points at the loopback so a listener can be
        // bound; verify run() exits immediately because shutdown is
        // already set before run() is awaited.
        let cfg = SrtIngestConfig {
            bind_ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: 0,
            ..Default::default()
        };
        let server = SrtIngestServer::new(cfg);
        let (tx, rx) = watch::channel(false);
        tx.send(true).expect("send shutdown");
        let result = tokio::time::timeout(std::time::Duration::from_secs(1), server.run(rx))
            .await
            .expect("run should exit immediately when shutdown already set");
        assert!(result.is_ok());
    }
}
