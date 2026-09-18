//! Connection management for routing matrix.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of audio connection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConnectionType {
    /// Analog audio connection
    Analog,
    /// AES3 digital audio
    Aes3,
    /// MADI multi-channel digital
    Madi,
    /// Dante audio-over-IP
    Dante,
    /// Embedded audio in SDI
    EmbeddedSdi,
}

/// Quality priority for routing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RoutingPriority {
    /// Low priority (can be interrupted)
    Low,
    /// Normal priority
    #[default]
    Normal,
    /// High priority (protected)
    High,
    /// Critical priority (never interrupted)
    Critical,
}

/// Represents a single audio connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    /// Unique identifier for this connection
    pub id: ConnectionId,
    /// Source input index
    pub source: usize,
    /// Destination output index
    pub destination: usize,
    /// Connection type
    pub connection_type: ConnectionType,
    /// Priority of this connection
    pub priority: RoutingPriority,
    /// Gain applied to this connection (in dB)
    pub gain_db: f32,
    /// Whether this connection is active
    pub active: bool,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

/// Unique identifier for a connection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConnectionId(u64);

impl ConnectionId {
    /// Create a new connection ID
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the inner ID value
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl Connection {
    /// Create a new connection
    #[must_use]
    pub fn new(
        id: ConnectionId,
        source: usize,
        destination: usize,
        connection_type: ConnectionType,
    ) -> Self {
        Self {
            id,
            source,
            destination,
            connection_type,
            priority: RoutingPriority::Normal,
            gain_db: 0.0,
            active: true,
            metadata: HashMap::new(),
        }
    }

    /// Set the priority of this connection
    #[must_use]
    pub fn with_priority(mut self, priority: RoutingPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set the gain for this connection
    #[must_use]
    pub fn with_gain(mut self, gain_db: f32) -> Self {
        self.gain_db = gain_db;
        self
    }

    /// Add metadata to this connection
    #[must_use]
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Activate or deactivate this connection
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    /// Check if this connection is active
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }
}

/// Manager for all connections in the routing system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionManager {
    /// All connections indexed by ID
    connections: HashMap<ConnectionId, Connection>,
    /// Next connection ID to assign
    next_id: u64,
    /// Index of connections by source
    source_index: HashMap<usize, Vec<ConnectionId>>,
    /// Index of connections by destination
    destination_index: HashMap<usize, Vec<ConnectionId>>,
}

impl Default for ConnectionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ConnectionManager {
    /// Create a new connection manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            connections: HashMap::new(),
            next_id: 1,
            source_index: HashMap::new(),
            destination_index: HashMap::new(),
        }
    }

    /// Add a new connection
    pub fn add_connection(
        &mut self,
        source: usize,
        destination: usize,
        connection_type: ConnectionType,
    ) -> ConnectionId {
        let id = ConnectionId::new(self.next_id);
        self.next_id += 1;

        let connection = Connection::new(id, source, destination, connection_type);

        self.source_index.entry(source).or_default().push(id);
        self.destination_index
            .entry(destination)
            .or_default()
            .push(id);

        self.connections.insert(id, connection);
        id
    }

    /// Remove a connection
    pub fn remove_connection(&mut self, id: ConnectionId) -> Option<Connection> {
        if let Some(connection) = self.connections.remove(&id) {
            // Remove from indices
            if let Some(sources) = self.source_index.get_mut(&connection.source) {
                sources.retain(|&conn_id| conn_id != id);
            }
            if let Some(destinations) = self.destination_index.get_mut(&connection.destination) {
                destinations.retain(|&conn_id| conn_id != id);
            }
            Some(connection)
        } else {
            None
        }
    }

    /// Get a connection by ID
    #[must_use]
    pub fn get_connection(&self, id: ConnectionId) -> Option<&Connection> {
        self.connections.get(&id)
    }

    /// Get a mutable reference to a connection
    pub fn get_connection_mut(&mut self, id: ConnectionId) -> Option<&mut Connection> {
        self.connections.get_mut(&id)
    }

    /// Get all connections from a source
    #[must_use]
    pub fn get_connections_from_source(&self, source: usize) -> Vec<&Connection> {
        self.source_index
            .get(&source)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.connections.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all connections to a destination
    #[must_use]
    pub fn get_connections_to_destination(&self, destination: usize) -> Vec<&Connection> {
        self.destination_index
            .get(&destination)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.connections.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all active connections
    #[must_use]
    pub fn get_active_connections(&self) -> Vec<&Connection> {
        self.connections
            .values()
            .filter(|conn| conn.active)
            .collect()
    }

    /// Get all connections of a specific type
    #[must_use]
    pub fn get_connections_by_type(&self, connection_type: ConnectionType) -> Vec<&Connection> {
        self.connections
            .values()
            .filter(|conn| conn.connection_type == connection_type)
            .collect()
    }

    /// Get total number of connections
    #[must_use]
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Clear all connections
    pub fn clear(&mut self) {
        self.connections.clear();
        self.source_index.clear();
        self.destination_index.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_creation() {
        let conn = Connection::new(ConnectionId::new(1), 0, 1, ConnectionType::Analog);
        assert_eq!(conn.source, 0);
        assert_eq!(conn.destination, 1);
        assert!(conn.is_active());
    }

    #[test]
    fn test_connection_builder() {
        let conn = Connection::new(ConnectionId::new(1), 0, 1, ConnectionType::Analog)
            .with_priority(RoutingPriority::High)
            .with_gain(-6.0)
            .with_metadata("label".to_string(), "Main Mix".to_string());

        assert_eq!(conn.priority, RoutingPriority::High);
        assert!((conn.gain_db - (-6.0)).abs() < f32::EPSILON);
        assert_eq!(conn.metadata.get("label"), Some(&"Main Mix".to_string()));
    }

    #[test]
    fn test_connection_manager() {
        let mut manager = ConnectionManager::new();

        let id1 = manager.add_connection(0, 1, ConnectionType::Analog);
        let _id2 = manager.add_connection(0, 2, ConnectionType::Aes3);
        let _id3 = manager.add_connection(1, 1, ConnectionType::Analog);

        assert_eq!(manager.connection_count(), 3);

        // Test source queries
        let from_source_0 = manager.get_connections_from_source(0);
        assert_eq!(from_source_0.len(), 2);

        // Test destination queries
        let to_dest_1 = manager.get_connections_to_destination(1);
        assert_eq!(to_dest_1.len(), 2);

        // Test removal
        manager.remove_connection(id1);
        assert_eq!(manager.connection_count(), 2);
        assert!(manager.get_connection(id1).is_none());

        // Verify indices updated
        let from_source_0_after = manager.get_connections_from_source(0);
        assert_eq!(from_source_0_after.len(), 1);
    }

    #[test]
    fn test_connection_type_filter() {
        let mut manager = ConnectionManager::new();

        manager.add_connection(0, 1, ConnectionType::Analog);
        manager.add_connection(0, 2, ConnectionType::Aes3);
        manager.add_connection(1, 1, ConnectionType::Analog);

        let analog_conns = manager.get_connections_by_type(ConnectionType::Analog);
        assert_eq!(analog_conns.len(), 2);

        let aes3_conns = manager.get_connections_by_type(ConnectionType::Aes3);
        assert_eq!(aes3_conns.len(), 1);
    }

    #[test]
    fn test_active_connections() {
        let mut manager = ConnectionManager::new();

        let id1 = manager.add_connection(0, 1, ConnectionType::Analog);
        let id2 = manager.add_connection(0, 2, ConnectionType::Analog);

        if let Some(conn) = manager.get_connection_mut(id1) {
            conn.set_active(false);
        }

        let active = manager.get_active_connections();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, id2);
    }
}
