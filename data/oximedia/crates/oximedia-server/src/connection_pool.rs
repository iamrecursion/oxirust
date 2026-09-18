//! Connection pool tracking for the OxiMedia server.
//!
//! Provides an in-process registry that monitors active HTTP/WebSocket
//! connections, enforces concurrency limits, tracks per-IP connection
//! counts, and exposes aggregate statistics.

#![allow(dead_code)]
#![allow(missing_docs)]

/// State of a tracked connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnState {
    /// Connection handshake in progress.
    Handshaking,
    /// Connection is fully established and active.
    Active,
    /// Connection is draining (graceful close in progress).
    Draining,
    /// Connection has been closed.
    Closed,
}

impl ConnState {
    /// Returns `true` when the connection is still occupying a slot.
    pub fn is_open(self) -> bool {
        matches!(
            self,
            ConnState::Handshaking | ConnState::Active | ConnState::Draining
        )
    }
}

/// Metadata for a single tracked connection.
#[derive(Debug, Clone)]
pub struct ConnEntry {
    /// Unique connection identifier.
    pub id: String,
    /// Remote IP address.
    pub remote_ip: String,
    /// Protocol label (e.g. "HTTP/1.1", "WebSocket").
    pub protocol: String,
    /// Current state.
    pub state: ConnState,
    /// Timestamp (ms since epoch) when the connection was accepted.
    pub accepted_at_ms: u64,
    /// Number of requests served on this connection.
    pub requests_served: u64,
}

impl ConnEntry {
    /// Creates a new entry in the `Handshaking` state.
    pub fn new(
        id: impl Into<String>,
        remote_ip: impl Into<String>,
        protocol: impl Into<String>,
        accepted_at_ms: u64,
    ) -> Self {
        Self {
            id: id.into(),
            remote_ip: remote_ip.into(),
            protocol: protocol.into(),
            state: ConnState::Handshaking,
            accepted_at_ms,
            requests_served: 0,
        }
    }

    /// Returns how long the connection has been open (ms).
    pub fn age_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.accepted_at_ms)
    }

    /// Returns `true` if the connection has been open longer than `ttl_ms`.
    pub fn is_stale(&self, now_ms: u64, ttl_ms: u64) -> bool {
        self.age_ms(now_ms) > ttl_ms
    }
}

/// Aggregate statistics exported by the pool.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total connections currently tracked (all states).
    pub total: usize,
    /// Connections in the `Active` state.
    pub active: usize,
    /// Connections in the `Handshaking` state.
    pub handshaking: usize,
    /// Connections in the `Draining` state.
    pub draining: usize,
    /// Connections in the `Closed` state.
    pub closed: usize,
    /// Peak concurrency seen since the pool was created.
    pub peak_active: usize,
}

/// An in-process registry tracking all server connections.
#[derive(Debug, Default)]
pub struct ConnectionPool {
    /// All connections ever registered (including closed).
    connections: Vec<ConnEntry>,
    /// Maximum number of simultaneous open connections allowed.
    max_connections: usize,
    /// Cumulative count of all connections accepted.
    total_accepted: u64,
    /// Recorded peak of simultaneously open connections.
    peak_open: usize,
}

impl ConnectionPool {
    /// Creates a new pool with the given concurrency limit.
    pub fn new(max_connections: usize) -> Self {
        Self {
            connections: Vec::new(),
            max_connections,
            total_accepted: 0,
            peak_open: 0,
        }
    }

    /// Attempts to register a new connection.
    ///
    /// Returns `false` when the pool is already at its limit and the
    /// connection should be rejected.
    pub fn register(&mut self, entry: ConnEntry) -> bool {
        let open = self.open_count();
        if open >= self.max_connections {
            return false;
        }
        self.connections.push(entry);
        self.total_accepted += 1;
        let new_open = self.open_count();
        if new_open > self.peak_open {
            self.peak_open = new_open;
        }
        true
    }

    /// Transitions a connection to `Active` state.
    ///
    /// Returns `true` when the connection was found.
    pub fn mark_active(&mut self, id: &str) -> bool {
        if let Some(e) = self.connections.iter_mut().find(|e| e.id == id) {
            e.state = ConnState::Active;
            true
        } else {
            false
        }
    }

    /// Transitions a connection to `Draining` state.
    pub fn mark_draining(&mut self, id: &str) -> bool {
        if let Some(e) = self.connections.iter_mut().find(|e| e.id == id) {
            e.state = ConnState::Draining;
            true
        } else {
            false
        }
    }

    /// Marks a connection as `Closed`.
    pub fn close(&mut self, id: &str) -> bool {
        if let Some(e) = self.connections.iter_mut().find(|e| e.id == id) {
            e.state = ConnState::Closed;
            true
        } else {
            false
        }
    }

    /// Records a request served on the connection identified by `id`.
    pub fn record_request(&mut self, id: &str) {
        if let Some(e) = self.connections.iter_mut().find(|e| e.id == id) {
            e.requests_served += 1;
        }
    }

    /// Returns the number of currently open connections.
    pub fn open_count(&self) -> usize {
        self.connections
            .iter()
            .filter(|e| e.state.is_open())
            .count()
    }

    /// Returns the number of open connections from a specific IP.
    pub fn connections_from_ip(&self, ip: &str) -> usize {
        self.connections
            .iter()
            .filter(|e| e.state.is_open() && e.remote_ip == ip)
            .count()
    }

    /// Returns all connections whose age exceeds `ttl_ms`.
    pub fn stale_connections(&self, now_ms: u64, ttl_ms: u64) -> Vec<&ConnEntry> {
        self.connections
            .iter()
            .filter(|e| e.state.is_open() && e.is_stale(now_ms, ttl_ms))
            .collect()
    }

    /// Removes all `Closed` connections from internal storage.
    pub fn prune_closed(&mut self) {
        self.connections.retain(|e| e.state != ConnState::Closed);
    }

    /// Returns aggregate statistics for the pool.
    pub fn stats(&self) -> PoolStats {
        let mut stats = PoolStats {
            total: self.connections.len(),
            peak_active: self.peak_open,
            ..PoolStats::default()
        };
        for e in &self.connections {
            match e.state {
                ConnState::Active => stats.active += 1,
                ConnState::Handshaking => stats.handshaking += 1,
                ConnState::Draining => stats.draining += 1,
                ConnState::Closed => stats.closed += 1,
            }
        }
        stats
    }

    /// Total connections accepted since the pool was created.
    pub fn total_accepted(&self) -> u64 {
        self.total_accepted
    }

    /// Returns the peak number of simultaneously open connections.
    pub fn peak_open(&self) -> usize {
        self.peak_open
    }

    /// Checks whether the pool can accept another connection.
    pub fn has_capacity(&self) -> bool {
        self.open_count() < self.max_connections
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(id: &str, ip: &str) -> ConnEntry {
        ConnEntry::new(id, ip, "HTTP/1.1", 1_000)
    }

    // --- ConnState ---

    #[test]
    fn test_conn_state_is_open() {
        assert!(ConnState::Handshaking.is_open());
        assert!(ConnState::Active.is_open());
        assert!(ConnState::Draining.is_open());
        assert!(!ConnState::Closed.is_open());
    }

    // --- ConnEntry ---

    #[test]
    fn test_conn_entry_age() {
        let e = make_entry("c1", "10.0.0.1");
        assert_eq!(e.age_ms(6_000), 5_000);
    }

    #[test]
    fn test_conn_entry_is_stale_true() {
        let e = make_entry("c1", "10.0.0.1");
        assert!(e.is_stale(10_001, 9_000));
    }

    #[test]
    fn test_conn_entry_is_stale_false() {
        let e = make_entry("c1", "10.0.0.1");
        assert!(!e.is_stale(5_000, 9_000));
    }

    #[test]
    fn test_conn_entry_initial_state() {
        let e = make_entry("c1", "10.0.0.1");
        assert_eq!(e.state, ConnState::Handshaking);
        assert_eq!(e.requests_served, 0);
    }

    // --- ConnectionPool registration ---

    #[test]
    fn test_register_success() {
        let mut pool = ConnectionPool::new(10);
        assert!(pool.register(make_entry("c1", "1.1.1.1")));
        assert_eq!(pool.open_count(), 1);
    }

    #[test]
    fn test_register_at_capacity_rejected() {
        let mut pool = ConnectionPool::new(2);
        pool.register(make_entry("c1", "1.1.1.1"));
        pool.register(make_entry("c2", "1.1.1.2"));
        let accepted = pool.register(make_entry("c3", "1.1.1.3"));
        assert!(!accepted);
        assert_eq!(pool.open_count(), 2);
    }

    #[test]
    fn test_has_capacity_true_and_false() {
        let mut pool = ConnectionPool::new(1);
        assert!(pool.has_capacity());
        pool.register(make_entry("c1", "1.1.1.1"));
        assert!(!pool.has_capacity());
    }

    // --- State transitions ---

    #[test]
    fn test_mark_active() {
        let mut pool = ConnectionPool::new(5);
        pool.register(make_entry("c1", "1.1.1.1"));
        assert!(pool.mark_active("c1"));
        let s = pool
            .connections
            .iter()
            .find(|e| e.id == "c1")
            .expect("should succeed in test");
        assert_eq!(s.state, ConnState::Active);
    }

    #[test]
    fn test_mark_draining() {
        let mut pool = ConnectionPool::new(5);
        pool.register(make_entry("c1", "1.1.1.1"));
        pool.mark_active("c1");
        assert!(pool.mark_draining("c1"));
        let s = pool
            .connections
            .iter()
            .find(|e| e.id == "c1")
            .expect("should succeed in test");
        assert_eq!(s.state, ConnState::Draining);
    }

    #[test]
    fn test_close_reduces_open_count() {
        let mut pool = ConnectionPool::new(5);
        pool.register(make_entry("c1", "1.1.1.1"));
        pool.mark_active("c1");
        assert_eq!(pool.open_count(), 1);
        pool.close("c1");
        assert_eq!(pool.open_count(), 0);
    }

    #[test]
    fn test_mark_active_missing_id() {
        let mut pool = ConnectionPool::new(5);
        assert!(!pool.mark_active("ghost"));
    }

    // --- Per-IP counts ---

    #[test]
    fn test_connections_from_ip() {
        let mut pool = ConnectionPool::new(10);
        pool.register(make_entry("c1", "192.168.1.1"));
        pool.register(make_entry("c2", "192.168.1.1"));
        pool.register(make_entry("c3", "10.0.0.1"));
        assert_eq!(pool.connections_from_ip("192.168.1.1"), 2);
        assert_eq!(pool.connections_from_ip("10.0.0.1"), 1);
    }

    // --- Request recording ---

    #[test]
    fn test_record_request() {
        let mut pool = ConnectionPool::new(5);
        pool.register(make_entry("c1", "1.1.1.1"));
        pool.record_request("c1");
        pool.record_request("c1");
        let e = pool
            .connections
            .iter()
            .find(|e| e.id == "c1")
            .expect("should succeed in test");
        assert_eq!(e.requests_served, 2);
    }

    // --- Stale connections ---

    #[test]
    fn test_stale_connections_detected() {
        let mut pool = ConnectionPool::new(5);
        pool.register(make_entry("c1", "1.1.1.1")); // accepted at 1000ms
        pool.mark_active("c1");
        // now=10_000, ttl=5_000 → age=9_000 > 5_000 → stale
        let stale = pool.stale_connections(10_000, 5_000);
        assert_eq!(stale.len(), 1);
    }

    #[test]
    fn test_no_stale_connections_fresh() {
        let mut pool = ConnectionPool::new(5);
        pool.register(make_entry("c1", "1.1.1.1"));
        pool.mark_active("c1");
        // now=1_100, ttl=5_000 → age=100 < 5_000 → not stale
        let stale = pool.stale_connections(1_100, 5_000);
        assert!(stale.is_empty());
    }

    // --- Prune and stats ---

    #[test]
    fn test_prune_closed() {
        let mut pool = ConnectionPool::new(5);
        pool.register(make_entry("c1", "1.1.1.1"));
        pool.register(make_entry("c2", "1.1.1.2"));
        pool.close("c1");
        pool.prune_closed();
        assert_eq!(pool.connections.len(), 1);
        assert_eq!(pool.connections[0].id, "c2");
    }

    #[test]
    fn test_stats_aggregation() {
        let mut pool = ConnectionPool::new(10);
        pool.register(make_entry("c1", "1.1.1.1"));
        pool.mark_active("c1");
        pool.register(make_entry("c2", "1.1.1.2"));
        pool.mark_active("c2");
        pool.mark_draining("c2");
        pool.register(make_entry("c3", "1.1.1.3")); // stays handshaking
        let s = pool.stats();
        assert_eq!(s.active, 1);
        assert_eq!(s.draining, 1);
        assert_eq!(s.handshaking, 1);
        assert_eq!(s.closed, 0);
        assert_eq!(s.total, 3);
    }

    #[test]
    fn test_total_accepted_counter() {
        let mut pool = ConnectionPool::new(10);
        pool.register(make_entry("c1", "1.1.1.1"));
        pool.register(make_entry("c2", "1.1.1.2"));
        pool.close("c1");
        pool.prune_closed();
        assert_eq!(pool.total_accepted(), 2);
    }

    #[test]
    fn test_peak_open_tracked() {
        let mut pool = ConnectionPool::new(10);
        pool.register(make_entry("c1", "1.1.1.1"));
        pool.register(make_entry("c2", "1.1.1.2"));
        pool.register(make_entry("c3", "1.1.1.3"));
        pool.close("c1");
        pool.close("c2");
        pool.close("c3");
        assert_eq!(pool.peak_open(), 3);
    }
}
