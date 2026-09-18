//! Dante audio-over-IP metadata (respects Audinate IP, metadata only).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Dante flow metadata (not actual implementation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DanteFlowMetadata {
    /// Flow name
    pub name: String,
    /// Source device name
    pub source_device: String,
    /// Source channel
    pub source_channel: u16,
    /// Destination device name
    pub destination_device: String,
    /// Destination channel
    pub destination_channel: u16,
    /// Sample rate
    pub sample_rate: u32,
    /// Bit depth
    pub bit_depth: u8,
    /// Latency in microseconds
    pub latency_us: u32,
    /// Flow status
    pub status: FlowStatus,
}

/// Dante flow status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowStatus {
    /// Flow is active
    Active,
    /// Flow is inactive
    Inactive,
    /// Flow has errors
    Error,
}

/// Dante device metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DanteDeviceMetadata {
    /// Device name
    pub name: String,
    /// Device model
    pub model: String,
    /// Number of transmit channels
    pub tx_channels: u16,
    /// Number of receive channels
    pub rx_channels: u16,
    /// Sample rates supported
    pub sample_rates: Vec<u32>,
    /// Network interface
    pub interface: String,
    /// Device status
    pub status: DeviceStatus,
}

/// Dante device status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceStatus {
    /// Device is online
    Online,
    /// Device is offline
    Offline,
    /// Device has errors
    Error,
}

/// Dante routing configuration (metadata only)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DanteRouting {
    /// Device metadata
    pub devices: HashMap<String, DanteDeviceMetadata>,
    /// Flow metadata
    pub flows: Vec<DanteFlowMetadata>,
    /// Network configuration
    pub network_config: NetworkConfig,
}

/// Network configuration metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Primary network interface
    pub primary_interface: String,
    /// Secondary network interface (redundancy)
    pub secondary_interface: Option<String>,
    /// `QoS` priority
    pub qos_priority: u8,
    /// Multicast mode enabled
    pub multicast: bool,
}

impl DanteRouting {
    /// Create a new Dante routing configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            devices: HashMap::new(),
            flows: Vec::new(),
            network_config: NetworkConfig {
                primary_interface: String::from("eth0"),
                secondary_interface: None,
                qos_priority: 4,
                multicast: false,
            },
        }
    }

    /// Add a device
    pub fn add_device(&mut self, device: DanteDeviceMetadata) {
        self.devices.insert(device.name.clone(), device);
    }

    /// Add a flow
    pub fn add_flow(&mut self, flow: DanteFlowMetadata) {
        self.flows.push(flow);
    }

    /// Get device by name
    #[must_use]
    pub fn get_device(&self, name: &str) -> Option<&DanteDeviceMetadata> {
        self.devices.get(name)
    }

    /// Get all active flows
    #[must_use]
    pub fn active_flows(&self) -> Vec<&DanteFlowMetadata> {
        self.flows
            .iter()
            .filter(|f| f.status == FlowStatus::Active)
            .collect()
    }

    /// Get flow count
    #[must_use]
    pub fn flow_count(&self) -> usize {
        self.flows.len()
    }

    /// Get device count
    #[must_use]
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }
}

impl Default for DanteRouting {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dante_routing_creation() {
        let routing = DanteRouting::new();
        assert_eq!(routing.device_count(), 0);
        assert_eq!(routing.flow_count(), 0);
    }

    #[test]
    fn test_add_device() {
        let mut routing = DanteRouting::new();

        let device = DanteDeviceMetadata {
            name: "Device1".to_string(),
            model: "Model X".to_string(),
            tx_channels: 64,
            rx_channels: 64,
            sample_rates: vec![48000, 96000],
            interface: "eth0".to_string(),
            status: DeviceStatus::Online,
        };

        routing.add_device(device);
        assert_eq!(routing.device_count(), 1);
        assert!(routing.get_device("Device1").is_some());
    }

    #[test]
    fn test_add_flow() {
        let mut routing = DanteRouting::new();

        let flow = DanteFlowMetadata {
            name: "Flow1".to_string(),
            source_device: "Dev1".to_string(),
            source_channel: 0,
            destination_device: "Dev2".to_string(),
            destination_channel: 0,
            sample_rate: 48000,
            bit_depth: 24,
            latency_us: 1000,
            status: FlowStatus::Active,
        };

        routing.add_flow(flow);
        assert_eq!(routing.flow_count(), 1);
    }

    #[test]
    fn test_active_flows() {
        let mut routing = DanteRouting::new();

        let active_flow = DanteFlowMetadata {
            name: "Active".to_string(),
            source_device: "Dev1".to_string(),
            source_channel: 0,
            destination_device: "Dev2".to_string(),
            destination_channel: 0,
            sample_rate: 48000,
            bit_depth: 24,
            latency_us: 1000,
            status: FlowStatus::Active,
        };

        let inactive_flow = DanteFlowMetadata {
            name: "Inactive".to_string(),
            source_device: "Dev1".to_string(),
            source_channel: 1,
            destination_device: "Dev2".to_string(),
            destination_channel: 1,
            sample_rate: 48000,
            bit_depth: 24,
            latency_us: 1000,
            status: FlowStatus::Inactive,
        };

        routing.add_flow(active_flow);
        routing.add_flow(inactive_flow);

        let active = routing.active_flows();
        assert_eq!(active.len(), 1);
    }
}
