//! NMOS IS-04 / IS-05 concepts for node registry, device model, and connection management.
//!
//! NMOS (Networked Media Open Specifications) defines a set of open APIs for
//! professional media over IP, covering discovery (IS-04) and connection management (IS-05).

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::fmt;

/// Universally unique identifier (simplified UUID v4 representation).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NmosId(String);

impl NmosId {
    /// Create an ID from a static string slice (for testing).
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        Self(s.to_string())
    }

    /// Return the inner string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for NmosId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// NMOS resource version timestamp (tai:nanoseconds format: "secs:nanos").
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NmosVersion {
    /// Seconds component.
    pub secs: u64,
    /// Nanoseconds component.
    pub nanos: u32,
}

impl NmosVersion {
    /// Create a new version.
    #[must_use]
    pub fn new(secs: u64, nanos: u32) -> Self {
        Self { secs, nanos }
    }

    /// Format as "secs:nanos".
    #[must_use]
    pub fn to_string_repr(&self) -> String {
        format!("{}:{}", self.secs, self.nanos)
    }

    /// Increment by 1 nanosecond.
    #[must_use]
    pub fn bump(&self) -> Self {
        if self.nanos == 999_999_999 {
            Self::new(self.secs + 1, 0)
        } else {
            Self::new(self.secs, self.nanos + 1)
        }
    }
}

/// IS-04 node resource.
#[derive(Debug, Clone)]
pub struct NmosNode {
    /// Node unique ID.
    pub id: NmosId,
    /// Human-readable label.
    pub label: String,
    /// Node description.
    pub description: String,
    /// Resource version.
    pub version: NmosVersion,
    /// Node hostname.
    pub hostname: String,
    /// API endpoints (scheme, host, port).
    pub api_endpoints: Vec<(String, String, u16)>,
    /// Attached device IDs.
    pub devices: Vec<NmosId>,
}

impl NmosNode {
    /// Create a new node resource.
    #[must_use]
    pub fn new(id: NmosId, label: impl Into<String>, hostname: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            description: String::new(),
            version: NmosVersion::new(0, 1),
            hostname: hostname.into(),
            api_endpoints: Vec::new(),
            devices: Vec::new(),
        }
    }

    /// Add an API endpoint.
    pub fn add_endpoint(&mut self, scheme: &str, host: &str, port: u16) {
        self.api_endpoints
            .push((scheme.to_string(), host.to_string(), port));
    }
}

/// IS-04 device type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceType {
    /// Generic device.
    Generic,
    /// Pipeline device.
    Pipeline,
}

/// IS-04 device resource.
#[derive(Debug, Clone)]
pub struct NmosDevice {
    /// Device unique ID.
    pub id: NmosId,
    /// Parent node ID.
    pub node_id: NmosId,
    /// Human-readable label.
    pub label: String,
    /// Device type.
    pub device_type: DeviceType,
    /// Resource version.
    pub version: NmosVersion,
    /// Associated sender IDs.
    pub senders: Vec<NmosId>,
    /// Associated receiver IDs.
    pub receivers: Vec<NmosId>,
}

impl NmosDevice {
    /// Create a new device resource.
    #[must_use]
    pub fn new(id: NmosId, node_id: NmosId, label: impl Into<String>) -> Self {
        Self {
            id,
            node_id,
            label: label.into(),
            device_type: DeviceType::Generic,
            version: NmosVersion::new(0, 1),
            senders: Vec::new(),
            receivers: Vec::new(),
        }
    }
}

/// IS-05 transport parameters for RTP.
#[derive(Debug, Clone)]
pub struct RtpTransportParams {
    /// Destination IP address.
    pub destination_ip: String,
    /// Destination port.
    pub destination_port: u16,
    /// Source IP address.
    pub source_ip: Option<String>,
    /// RTP enabled flag.
    pub rtp_enabled: bool,
}

impl RtpTransportParams {
    /// Create new RTP transport parameters.
    #[must_use]
    pub fn new(destination_ip: impl Into<String>, destination_port: u16) -> Self {
        Self {
            destination_ip: destination_ip.into(),
            destination_port,
            source_ip: None,
            rtp_enabled: true,
        }
    }
}

/// IS-05 connection state.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ConnectionState {
    /// No active connection.
    #[default]
    Inactive,
    /// Connection is being activated.
    Activating,
    /// Connection is active.
    Active,
    /// Connection activation failed.
    Failed(String),
}

/// IS-05 sender connection resource.
#[derive(Debug, Clone)]
pub struct SenderConnection {
    /// Sender ID.
    pub sender_id: NmosId,
    /// Transport parameters.
    pub transport_params: RtpTransportParams,
    /// Active transport parameters (committed).
    pub active_params: Option<RtpTransportParams>,
    /// Connection state.
    pub state: ConnectionState,
}

impl SenderConnection {
    /// Create a new sender connection (inactive).
    #[must_use]
    pub fn new(sender_id: NmosId, transport_params: RtpTransportParams) -> Self {
        Self {
            sender_id,
            transport_params,
            active_params: None,
            state: ConnectionState::Inactive,
        }
    }

    /// Stage and activate the connection.
    pub fn activate(&mut self) {
        self.active_params = Some(self.transport_params.clone());
        self.state = ConnectionState::Active;
    }

    /// Deactivate the connection.
    pub fn deactivate(&mut self) {
        self.active_params = None;
        self.state = ConnectionState::Inactive;
    }
}

/// IS-05 receiver connection resource.
#[derive(Debug, Clone)]
pub struct ReceiverConnection {
    /// Receiver ID.
    pub receiver_id: NmosId,
    /// Sender ID to connect to (if any).
    pub sender_id: Option<NmosId>,
    /// Transport parameters.
    pub transport_params: Option<RtpTransportParams>,
    /// Connection state.
    pub state: ConnectionState,
}

impl ReceiverConnection {
    /// Create a new receiver connection (inactive).
    #[must_use]
    pub fn new(receiver_id: NmosId) -> Self {
        Self {
            receiver_id,
            sender_id: None,
            transport_params: None,
            state: ConnectionState::Inactive,
        }
    }

    /// Connect receiver to a sender with given transport parameters.
    pub fn connect(&mut self, sender_id: NmosId, params: RtpTransportParams) {
        self.sender_id = Some(sender_id);
        self.transport_params = Some(params);
        self.state = ConnectionState::Active;
    }

    /// Disconnect receiver.
    pub fn disconnect(&mut self) {
        self.sender_id = None;
        self.transport_params = None;
        self.state = ConnectionState::Inactive;
    }
}

/// NMOS node registry — tracks nodes, devices, senders, receivers.
#[derive(Debug, Default)]
pub struct NmosRegistry {
    /// Registered nodes.
    pub nodes: HashMap<NmosId, NmosNode>,
    /// Registered devices.
    pub devices: HashMap<NmosId, NmosDevice>,
    /// Sender connections.
    pub sender_connections: HashMap<NmosId, SenderConnection>,
    /// Receiver connections.
    pub receiver_connections: HashMap<NmosId, ReceiverConnection>,
}

impl NmosRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a node.
    pub fn register_node(&mut self, node: NmosNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Register a device.
    pub fn register_device(&mut self, device: NmosDevice) {
        self.devices.insert(device.id.clone(), device);
    }

    /// Register a sender connection.
    pub fn register_sender(&mut self, conn: SenderConnection) {
        self.sender_connections.insert(conn.sender_id.clone(), conn);
    }

    /// Register a receiver connection.
    pub fn register_receiver(&mut self, conn: ReceiverConnection) {
        self.receiver_connections
            .insert(conn.receiver_id.clone(), conn);
    }

    /// Remove a node and its associated devices.
    pub fn unregister_node(&mut self, node_id: &NmosId) -> bool {
        if self.nodes.remove(node_id).is_some() {
            self.devices.retain(|_, d| &d.node_id != node_id);
            true
        } else {
            false
        }
    }

    /// Count total registered resources.
    #[must_use]
    pub fn total_resources(&self) -> usize {
        self.nodes.len() + self.devices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: &str) -> NmosId {
        NmosId::from_str(s)
    }

    #[test]
    fn test_nmos_id_display() {
        let nid = id("abc-123");
        assert_eq!(format!("{nid}"), "abc-123");
    }

    #[test]
    fn test_nmos_version_ordering() {
        let v1 = NmosVersion::new(1, 0);
        let v2 = NmosVersion::new(1, 1);
        let v3 = NmosVersion::new(2, 0);
        assert!(v1 < v2);
        assert!(v2 < v3);
    }

    #[test]
    fn test_nmos_version_string_repr() {
        let v = NmosVersion::new(12345, 678);
        assert_eq!(v.to_string_repr(), "12345:678");
    }

    #[test]
    fn test_nmos_version_bump_nanos() {
        let v = NmosVersion::new(1, 100);
        let v2 = v.bump();
        assert_eq!(v2, NmosVersion::new(1, 101));
    }

    #[test]
    fn test_nmos_version_bump_overflow() {
        let v = NmosVersion::new(1, 999_999_999);
        let v2 = v.bump();
        assert_eq!(v2, NmosVersion::new(2, 0));
    }

    #[test]
    fn test_nmos_node_creation() {
        let node = NmosNode::new(id("node-1"), "Camera A", "camera-a.local");
        assert_eq!(node.label, "Camera A");
        assert_eq!(node.hostname, "camera-a.local");
        assert!(node.devices.is_empty());
    }

    #[test]
    fn test_nmos_node_add_endpoint() {
        let mut node = NmosNode::new(id("node-2"), "Test", "test.local");
        node.add_endpoint("http", "192.168.1.10", 80);
        assert_eq!(node.api_endpoints.len(), 1);
        assert_eq!(node.api_endpoints[0].2, 80);
    }

    #[test]
    fn test_rtp_transport_params() {
        let p = RtpTransportParams::new("239.0.0.1", 5004);
        assert_eq!(p.destination_ip, "239.0.0.1");
        assert_eq!(p.destination_port, 5004);
        assert!(p.rtp_enabled);
    }

    #[test]
    fn test_sender_connection_activate() {
        let params = RtpTransportParams::new("239.0.0.2", 5004);
        let mut conn = SenderConnection::new(id("sender-1"), params);
        assert_eq!(conn.state, ConnectionState::Inactive);
        conn.activate();
        assert_eq!(conn.state, ConnectionState::Active);
        assert!(conn.active_params.is_some());
    }

    #[test]
    fn test_sender_connection_deactivate() {
        let params = RtpTransportParams::new("239.0.0.3", 5004);
        let mut conn = SenderConnection::new(id("sender-2"), params);
        conn.activate();
        conn.deactivate();
        assert_eq!(conn.state, ConnectionState::Inactive);
        assert!(conn.active_params.is_none());
    }

    #[test]
    fn test_receiver_connection_connect_disconnect() {
        let mut conn = ReceiverConnection::new(id("recv-1"));
        let params = RtpTransportParams::new("239.0.0.4", 5004);
        conn.connect(id("sender-99"), params);
        assert_eq!(conn.state, ConnectionState::Active);
        assert!(conn.sender_id.is_some());
        conn.disconnect();
        assert_eq!(conn.state, ConnectionState::Inactive);
        assert!(conn.sender_id.is_none());
    }

    #[test]
    fn test_registry_register_and_count() {
        let mut reg = NmosRegistry::new();
        let node = NmosNode::new(id("n1"), "Node 1", "n1.local");
        let device = NmosDevice::new(id("d1"), id("n1"), "Device 1");
        reg.register_node(node);
        reg.register_device(device);
        assert_eq!(reg.total_resources(), 2);
    }

    #[test]
    fn test_registry_unregister_node_removes_devices() {
        let mut reg = NmosRegistry::new();
        let node = NmosNode::new(id("n2"), "Node 2", "n2.local");
        let device = NmosDevice::new(id("d2"), id("n2"), "Device 2");
        reg.register_node(node);
        reg.register_device(device);
        let removed = reg.unregister_node(&id("n2"));
        assert!(removed);
        assert_eq!(reg.total_resources(), 0);
    }

    #[test]
    fn test_registry_unregister_nonexistent() {
        let mut reg = NmosRegistry::new();
        let removed = reg.unregister_node(&id("ghost"));
        assert!(!removed);
    }

    #[test]
    fn test_connection_state_default() {
        let s = ConnectionState::default();
        assert_eq!(s, ConnectionState::Inactive);
    }
}
