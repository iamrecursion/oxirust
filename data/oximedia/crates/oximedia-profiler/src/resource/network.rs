//! Network socket tracking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Instant;

/// Network statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    /// Total bytes sent.
    pub bytes_sent: u64,

    /// Total bytes received.
    pub bytes_received: u64,

    /// Number of active connections.
    pub active_connections: usize,

    /// Average bandwidth (bytes/sec).
    pub avg_bandwidth: f64,
}

/// Network socket information.
#[derive(Debug, Clone)]
pub struct SocketInfo {
    /// Remote address.
    pub addr: SocketAddr,

    /// When connected.
    pub connected_at: Instant,

    /// Bytes sent.
    pub bytes_sent: u64,

    /// Bytes received.
    pub bytes_received: u64,
}

/// Network tracker.
#[derive(Debug)]
pub struct NetworkTracker {
    sockets: HashMap<u64, SocketInfo>,
    next_id: u64,
    total_bytes_sent: u64,
    total_bytes_received: u64,
}

impl NetworkTracker {
    /// Create a new network tracker.
    pub fn new() -> Self {
        Self {
            sockets: HashMap::new(),
            next_id: 0,
            total_bytes_sent: 0,
            total_bytes_received: 0,
        }
    }

    /// Track a new connection.
    pub fn track_connect(&mut self, addr: SocketAddr) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let socket = SocketInfo {
            addr,
            connected_at: Instant::now(),
            bytes_sent: 0,
            bytes_received: 0,
        };

        self.sockets.insert(id, socket);
        id
    }

    /// Track connection close.
    pub fn track_disconnect(&mut self, id: u64) -> bool {
        self.sockets.remove(&id).is_some()
    }

    /// Track bytes sent.
    pub fn track_send(&mut self, id: u64, bytes: u64) {
        if let Some(socket) = self.sockets.get_mut(&id) {
            socket.bytes_sent += bytes;
            self.total_bytes_sent += bytes;
        }
    }

    /// Track bytes received.
    pub fn track_receive(&mut self, id: u64, bytes: u64) {
        if let Some(socket) = self.sockets.get_mut(&id) {
            socket.bytes_received += bytes;
            self.total_bytes_received += bytes;
        }
    }

    /// Get network statistics.
    pub fn stats(&self) -> NetworkStats {
        let avg_bandwidth = 0.0; // Would calculate from time deltas

        NetworkStats {
            bytes_sent: self.total_bytes_sent,
            bytes_received: self.total_bytes_received,
            active_connections: self.sockets.len(),
            avg_bandwidth,
        }
    }

    /// Get active connection count.
    pub fn connection_count(&self) -> usize {
        self.sockets.len()
    }
}

impl Default for NetworkTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_tracker() {
        let mut tracker = NetworkTracker::new();
        let addr = "127.0.0.1:8080".parse().expect("should succeed in test");
        let id = tracker.track_connect(addr);

        assert_eq!(tracker.connection_count(), 1);

        tracker.track_disconnect(id);
        assert_eq!(tracker.connection_count(), 0);
    }

    #[test]
    fn test_network_stats() {
        let mut tracker = NetworkTracker::new();
        let addr = "127.0.0.1:8080".parse().expect("should succeed in test");
        let id = tracker.track_connect(addr);

        tracker.track_send(id, 1000);
        tracker.track_receive(id, 500);

        let stats = tracker.stats();
        assert_eq!(stats.bytes_sent, 1000);
        assert_eq!(stats.bytes_received, 500);
    }
}
