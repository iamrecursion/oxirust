//! Dante audio-over-IP device discovery (metadata simulation).
//!
//! This module simulates Dante mDNS-based discovery using the
//! `_netaudio-cmc._udp` (Control & Monitoring Channel) and
//! `_netaudio-arc._udp` (Audio Routing Channel) service types.
//!
//! No Audinate-proprietary protocol details are reproduced; this is
//! purely a structural metadata simulation for integration testing.

#![allow(dead_code)]

use std::collections::HashMap;

/// A Dante-capable device discovered on the network.
#[derive(Debug, Clone)]
pub struct DanteDevice {
    /// Device name (DNS-SD service instance name)
    pub name: String,
    /// IPv4 address octets
    pub ip: [u8; 4],
    /// UDP port for control channel
    pub port: u16,
    /// Number of audio channels
    pub channel_count: u32,
    /// Sample rate in Hz (e.g. 48000)
    pub sample_rate: u32,
}

impl DanteDevice {
    /// Create a new `DanteDevice`.
    pub fn new(
        name: impl Into<String>,
        ip: [u8; 4],
        port: u16,
        channel_count: u32,
        sample_rate: u32,
    ) -> Self {
        Self {
            name: name.into(),
            ip,
            port,
            channel_count,
            sample_rate,
        }
    }

    /// Return the IP address as a dotted-decimal string.
    pub fn ip_string(&self) -> String {
        format!(
            "{}.{}.{}.{}",
            self.ip[0], self.ip[1], self.ip[2], self.ip[3]
        )
    }
}

/// Simulated mDNS record types used by Dante.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DanteMdnsRecord {
    /// Control and Monitoring Channel (`_netaudio-cmc._udp`)
    ControlMonitoringChannel,
    /// Audio Routing Channel (`_netaudio-arc._udp`)
    AudioRoutingChannel,
}

impl DanteMdnsRecord {
    /// Return the DNS-SD service type string.
    #[must_use]
    pub fn service_type(&self) -> &str {
        match self {
            DanteMdnsRecord::ControlMonitoringChannel => "_netaudio-cmc._udp",
            DanteMdnsRecord::AudioRoutingChannel => "_netaudio-arc._udp",
        }
    }
}

/// A single audio channel on a Dante device (transmitter or receiver).
#[derive(Debug, Clone)]
pub struct DanteChannel {
    /// Owning device name
    pub device: String,
    /// Zero-based channel index
    pub channel_idx: u32,
    /// Human-readable channel label
    pub label: String,
    /// `true` if this is a transmitter channel, `false` for a receiver channel
    pub is_transmitter: bool,
}

impl DanteChannel {
    /// Create a new `DanteChannel`.
    pub fn new(
        device: impl Into<String>,
        channel_idx: u32,
        label: impl Into<String>,
        is_transmitter: bool,
    ) -> Self {
        Self {
            device: device.into(),
            channel_idx,
            label: label.into(),
            is_transmitter,
        }
    }
}

/// A Dante subscription: a receiver channel listening to a transmitter channel.
#[derive(Debug, Clone)]
pub struct DanteSubscription {
    /// Transmitter device name
    pub tx_device: String,
    /// Transmitter channel index
    pub tx_channel: u32,
    /// Receiver device name
    pub rx_device: String,
    /// Receiver channel index
    pub rx_channel: u32,
    /// Target latency in microseconds
    pub latency_us: u32,
}

impl DanteSubscription {
    /// Create a new `DanteSubscription`.
    pub fn new(
        tx_device: impl Into<String>,
        tx_channel: u32,
        rx_device: impl Into<String>,
        rx_channel: u32,
        latency_us: u32,
    ) -> Self {
        Self {
            tx_device: tx_device.into(),
            tx_channel,
            rx_device: rx_device.into(),
            rx_channel,
            latency_us,
        }
    }
}

/// Calculate the recommended Dante latency for a given hop count.
///
/// Base latency is 1000 µs plus 250 µs per additional network hop.
#[must_use]
pub fn calculate_latency(hop_count: u32) -> u32 {
    1000 + 250 * hop_count
}

/// Simulated Dante device discovery registry.
#[derive(Debug, Default)]
pub struct DanteDiscovery {
    devices: HashMap<String, DanteDevice>,
}

impl DanteDiscovery {
    /// Create an empty discovery registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add (or replace) a discovered device.
    pub fn add_device(&mut self, device: DanteDevice) {
        self.devices.insert(device.name.clone(), device);
    }

    /// Remove a device by name; returns the removed device if it existed.
    pub fn remove_device(&mut self, name: &str) -> Option<DanteDevice> {
        self.devices.remove(name)
    }

    /// Find a device by name.
    pub fn find_device(&self, name: &str) -> Option<&DanteDevice> {
        self.devices.get(name)
    }

    /// Return the number of currently discovered devices.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// List all discovered device names.
    pub fn device_names(&self) -> Vec<&str> {
        self.devices.keys().map(String::as_str).collect()
    }
}

/// Dante audio router: manages subscriptions between channels.
#[derive(Debug, Default)]
pub struct DanteRouter {
    subscriptions: Vec<DanteSubscription>,
}

impl DanteRouter {
    /// Create a new router.
    pub fn new() -> Self {
        Self::default()
    }

    /// Subscribe a receiver channel to a transmitter channel.
    ///
    /// If the subscription already exists it is kept unchanged.
    pub fn subscribe(&mut self, subscription: DanteSubscription) {
        let already_exists = self.subscriptions.iter().any(|s| {
            s.rx_device == subscription.rx_device && s.rx_channel == subscription.rx_channel
        });
        if !already_exists {
            self.subscriptions.push(subscription);
        }
    }

    /// Unsubscribe a receiver channel.
    pub fn unsubscribe(&mut self, rx_device: &str, rx_channel: u32) {
        self.subscriptions
            .retain(|s| !(s.rx_device == rx_device && s.rx_channel == rx_channel));
    }

    /// List all current subscriptions.
    pub fn list_subscriptions(&self) -> Vec<&DanteSubscription> {
        self.subscriptions.iter().collect()
    }

    /// Return the number of active subscriptions.
    pub fn subscription_count(&self) -> usize {
        self.subscriptions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dante_device_creation() {
        let dev = DanteDevice::new("Studio-A", [192, 168, 1, 10], 4440, 32, 48000);
        assert_eq!(dev.name, "Studio-A");
        assert_eq!(dev.ip, [192, 168, 1, 10]);
        assert_eq!(dev.channel_count, 32);
        assert_eq!(dev.sample_rate, 48000);
    }

    #[test]
    fn test_dante_device_ip_string() {
        let dev = DanteDevice::new("Dev", [10, 0, 0, 1], 4440, 8, 44100);
        assert_eq!(dev.ip_string(), "10.0.0.1");
    }

    #[test]
    fn test_mdns_record_service_types() {
        assert_eq!(
            DanteMdnsRecord::ControlMonitoringChannel.service_type(),
            "_netaudio-cmc._udp"
        );
        assert_eq!(
            DanteMdnsRecord::AudioRoutingChannel.service_type(),
            "_netaudio-arc._udp"
        );
    }

    #[test]
    fn test_dante_channel() {
        let ch = DanteChannel::new("Console", 0, "Main L", true);
        assert_eq!(ch.device, "Console");
        assert!(ch.is_transmitter);
    }

    #[test]
    fn test_calculate_latency_zero_hops() {
        assert_eq!(calculate_latency(0), 1000);
    }

    #[test]
    fn test_calculate_latency_with_hops() {
        assert_eq!(calculate_latency(1), 1250);
        assert_eq!(calculate_latency(4), 2000);
    }

    #[test]
    fn test_discovery_add_find_device() {
        let mut disc = DanteDiscovery::new();
        let dev = DanteDevice::new("Mixer-1", [10, 1, 1, 1], 4440, 64, 48000);
        disc.add_device(dev);
        assert_eq!(disc.device_count(), 1);
        assert!(disc.find_device("Mixer-1").is_some());
        assert!(disc.find_device("Unknown").is_none());
    }

    #[test]
    fn test_discovery_remove_device() {
        let mut disc = DanteDiscovery::new();
        disc.add_device(DanteDevice::new("Dev-A", [10, 0, 0, 1], 4440, 8, 48000));
        let removed = disc.remove_device("Dev-A");
        assert!(removed.is_some());
        assert_eq!(disc.device_count(), 0);
    }

    #[test]
    fn test_router_subscribe() {
        let mut router = DanteRouter::new();
        let sub = DanteSubscription::new("Console", 0, "Recorder", 0, 1000);
        router.subscribe(sub);
        assert_eq!(router.subscription_count(), 1);
    }

    #[test]
    fn test_router_subscribe_duplicate_ignored() {
        let mut router = DanteRouter::new();
        router.subscribe(DanteSubscription::new("Tx", 0, "Rx", 0, 1000));
        // Same rx_device + rx_channel -> duplicate, should be ignored
        router.subscribe(DanteSubscription::new("Tx2", 1, "Rx", 0, 2000));
        assert_eq!(router.subscription_count(), 1);
    }

    #[test]
    fn test_router_unsubscribe() {
        let mut router = DanteRouter::new();
        router.subscribe(DanteSubscription::new("Tx", 0, "Rx", 0, 1000));
        router.unsubscribe("Rx", 0);
        assert_eq!(router.subscription_count(), 0);
    }

    #[test]
    fn test_router_list_subscriptions() {
        let mut router = DanteRouter::new();
        router.subscribe(DanteSubscription::new("Tx", 0, "Rx-A", 0, 1000));
        router.subscribe(DanteSubscription::new("Tx", 1, "Rx-B", 0, 1250));
        let list = router.list_subscriptions();
        assert_eq!(list.len(), 2);
    }
}
