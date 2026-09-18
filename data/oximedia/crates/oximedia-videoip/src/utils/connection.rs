//! Connection management and health monitoring.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

/// Connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Connection is being established.
    Connecting,
    /// Connection is active and healthy.
    Active,
    /// Connection is experiencing issues.
    Degraded,
    /// Connection has timed out.
    Timeout,
    /// Connection is closed.
    Closed,
}

/// Connection information and statistics.
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    /// Remote address.
    pub addr: SocketAddr,
    /// Connection state.
    pub state: ConnectionState,
    /// Time when connection was established.
    pub established_at: Instant,
    /// Time of last received packet.
    pub last_received: Instant,
    /// Time of last sent packet.
    pub last_sent: Instant,
    /// Total packets received.
    pub packets_received: u64,
    /// Total packets sent.
    pub packets_sent: u64,
    /// Total bytes received.
    pub bytes_received: u64,
    /// Total bytes sent.
    pub bytes_sent: u64,
    /// Packet loss rate (0.0-1.0).
    pub loss_rate: f64,
    /// Round-trip time in microseconds.
    pub rtt_us: u64,
}

impl ConnectionInfo {
    /// Creates a new connection info.
    #[must_use]
    pub fn new(addr: SocketAddr) -> Self {
        let now = Instant::now();
        Self {
            addr,
            state: ConnectionState::Connecting,
            established_at: now,
            last_received: now,
            last_sent: now,
            packets_received: 0,
            packets_sent: 0,
            bytes_received: 0,
            bytes_sent: 0,
            loss_rate: 0.0,
            rtt_us: 0,
        }
    }

    /// Updates the connection state based on activity.
    pub fn update_state(&mut self, timeout: Duration) {
        let now = Instant::now();
        let idle_time = now.duration_since(self.last_received);

        self.state = if idle_time > timeout {
            ConnectionState::Timeout
        } else if self.loss_rate > 0.1 {
            ConnectionState::Degraded
        } else if self.packets_received > 0 {
            ConnectionState::Active
        } else {
            ConnectionState::Connecting
        };
    }

    /// Records a packet received.
    pub fn record_received(&mut self, bytes: usize) {
        self.last_received = Instant::now();
        self.packets_received += 1;
        self.bytes_received += bytes as u64;
    }

    /// Records a packet sent.
    pub fn record_sent(&mut self, bytes: usize) {
        self.last_sent = Instant::now();
        self.packets_sent += 1;
        self.bytes_sent += bytes as u64;
    }

    /// Returns the connection uptime.
    #[must_use]
    pub fn uptime(&self) -> Duration {
        Instant::now().duration_since(self.established_at)
    }

    /// Returns true if the connection is healthy.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.state == ConnectionState::Active && self.loss_rate < 0.05
    }
}

/// Connection manager for tracking multiple connections.
pub struct ConnectionManager {
    /// Active connections.
    connections: HashMap<SocketAddr, ConnectionInfo>,
    /// Connection timeout.
    timeout: Duration,
    /// Maximum connections to track.
    max_connections: usize,
}

impl ConnectionManager {
    /// Creates a new connection manager.
    #[must_use]
    pub fn new(timeout: Duration, max_connections: usize) -> Self {
        Self {
            connections: HashMap::new(),
            timeout,
            max_connections,
        }
    }

    /// Adds or updates a connection.
    pub fn add_connection(&mut self, addr: SocketAddr) -> &mut ConnectionInfo {
        if !self.connections.contains_key(&addr) && self.connections.len() >= self.max_connections {
            // Remove oldest connection
            if let Some((oldest_addr, _)) = self
                .connections
                .iter()
                .min_by_key(|(_, info)| info.last_received)
            {
                let oldest_addr = *oldest_addr;
                self.connections.remove(&oldest_addr);
            }
        }

        self.connections
            .entry(addr)
            .or_insert_with(|| ConnectionInfo::new(addr))
    }

    /// Gets connection information.
    #[must_use]
    pub fn get_connection(&self, addr: &SocketAddr) -> Option<&ConnectionInfo> {
        self.connections.get(addr)
    }

    /// Gets mutable connection information.
    pub fn get_connection_mut(&mut self, addr: &SocketAddr) -> Option<&mut ConnectionInfo> {
        self.connections.get_mut(addr)
    }

    /// Removes a connection.
    pub fn remove_connection(&mut self, addr: &SocketAddr) {
        self.connections.remove(addr);
    }

    /// Updates all connection states.
    pub fn update_states(&mut self) {
        for conn in self.connections.values_mut() {
            conn.update_state(self.timeout);
        }
    }

    /// Removes timed-out connections.
    pub fn cleanup_timeouts(&mut self) {
        self.connections
            .retain(|_, conn| conn.state != ConnectionState::Timeout);
    }

    /// Returns the number of active connections.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.connections
            .values()
            .filter(|conn| conn.state == ConnectionState::Active)
            .count()
    }

    /// Returns all connection addresses.
    #[must_use]
    pub fn connection_addrs(&self) -> Vec<SocketAddr> {
        self.connections.keys().copied().collect()
    }

    /// Returns connections in a specific state.
    #[must_use]
    pub fn connections_by_state(&self, state: ConnectionState) -> Vec<&ConnectionInfo> {
        self.connections
            .values()
            .filter(|conn| conn.state == state)
            .collect()
    }

    /// Returns statistics for all connections.
    #[must_use]
    pub fn total_stats(&self) -> ConnectionStats {
        let mut stats = ConnectionStats::default();

        for conn in self.connections.values() {
            stats.total_packets_received += conn.packets_received;
            stats.total_packets_sent += conn.packets_sent;
            stats.total_bytes_received += conn.bytes_received;
            stats.total_bytes_sent += conn.bytes_sent;
        }

        stats.active_connections = self.active_count();
        stats.total_connections = self.connections.len();

        stats
    }
}

/// Aggregate statistics for all connections.
#[derive(Debug, Clone, Default)]
pub struct ConnectionStats {
    /// Number of active connections.
    pub active_connections: usize,
    /// Total number of connections.
    pub total_connections: usize,
    /// Total packets received across all connections.
    pub total_packets_received: u64,
    /// Total packets sent across all connections.
    pub total_packets_sent: u64,
    /// Total bytes received across all connections.
    pub total_bytes_received: u64,
    /// Total bytes sent across all connections.
    pub total_bytes_sent: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_info_creation() {
        let addr = "127.0.0.1:5000".parse().expect("should succeed in test");
        let info = ConnectionInfo::new(addr);

        assert_eq!(info.addr, addr);
        assert_eq!(info.state, ConnectionState::Connecting);
        assert_eq!(info.packets_received, 0);
    }

    #[test]
    fn test_connection_record() {
        let addr = "127.0.0.1:5000".parse().expect("should succeed in test");
        let mut info = ConnectionInfo::new(addr);

        info.record_received(100);
        info.record_sent(200);

        assert_eq!(info.packets_received, 1);
        assert_eq!(info.packets_sent, 1);
        assert_eq!(info.bytes_received, 100);
        assert_eq!(info.bytes_sent, 200);
    }

    #[test]
    fn test_connection_state_update() {
        let addr = "127.0.0.1:5000".parse().expect("should succeed in test");
        let mut info = ConnectionInfo::new(addr);

        info.record_received(100);
        info.update_state(Duration::from_secs(1));

        assert_eq!(info.state, ConnectionState::Active);

        // Simulate high packet loss
        info.loss_rate = 0.2;
        info.update_state(Duration::from_secs(1));

        assert_eq!(info.state, ConnectionState::Degraded);
    }

    #[test]
    fn test_connection_manager() {
        let mut manager = ConnectionManager::new(Duration::from_secs(5), 10);

        let addr1: SocketAddr = "127.0.0.1:5000".parse().expect("should succeed in test");
        let addr2: SocketAddr = "127.0.0.1:5001".parse().expect("should succeed in test");

        manager.add_connection(addr1);
        manager.add_connection(addr2);

        assert_eq!(manager.connections.len(), 2);
        assert!(manager.get_connection(&addr1).is_some());
        assert!(manager.get_connection(&addr2).is_some());
    }

    #[test]
    fn test_connection_manager_max_connections() {
        let mut manager = ConnectionManager::new(Duration::from_secs(5), 2);

        let addr1: SocketAddr = "127.0.0.1:5000".parse().expect("should succeed in test");
        let addr2: SocketAddr = "127.0.0.1:5001".parse().expect("should succeed in test");
        let addr3: SocketAddr = "127.0.0.1:5002".parse().expect("should succeed in test");

        manager.add_connection(addr1);
        manager.add_connection(addr2);
        manager.add_connection(addr3); // Should remove oldest

        assert_eq!(manager.connections.len(), 2);
    }

    #[test]
    fn test_connection_cleanup() {
        let mut manager = ConnectionManager::new(Duration::from_millis(100), 10);

        let addr: SocketAddr = "127.0.0.1:5000".parse().expect("should succeed in test");
        manager.add_connection(addr);

        std::thread::sleep(Duration::from_millis(200));

        manager.update_states();
        manager.cleanup_timeouts();

        assert_eq!(manager.connections.len(), 0);
    }

    #[test]
    fn test_connection_stats() {
        let mut manager = ConnectionManager::new(Duration::from_secs(5), 10);

        let addr1: SocketAddr = "127.0.0.1:5000".parse().expect("should succeed in test");
        let addr2: SocketAddr = "127.0.0.1:5001".parse().expect("should succeed in test");

        let conn1 = manager.add_connection(addr1);
        conn1.record_received(100);
        conn1.record_sent(200);

        let conn2 = manager.add_connection(addr2);
        conn2.record_received(150);
        conn2.record_sent(250);

        let stats = manager.total_stats();

        assert_eq!(stats.total_connections, 2);
        assert_eq!(stats.total_packets_received, 2);
        assert_eq!(stats.total_packets_sent, 2);
        assert_eq!(stats.total_bytes_received, 250);
        assert_eq!(stats.total_bytes_sent, 450);
    }
}
