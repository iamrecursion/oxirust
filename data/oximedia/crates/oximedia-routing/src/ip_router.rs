//! IP media routing for SMPTE ST 2110 workflows.
//!
//! Manages multicast IP flows for video, audio, and ancillary data,
//! including subscription and unsubscription of receiver endpoints.

#![allow(dead_code)]

/// The type of a SMPTE ST 2110 IP flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpFlowType {
    /// ST 2110-20 uncompressed video.
    Video2110_20,
    /// ST 2110-30 audio (AES67-compatible).
    Audio2110_30,
    /// ST 2110-40 ancillary data.
    Ancillary2110_40,
}

impl IpFlowType {
    /// Returns the IANA-registered or conventional default UDP port for this flow type.
    #[must_use]
    pub fn default_port(&self) -> u16 {
        match self {
            Self::Video2110_20 => 20000,
            Self::Audio2110_30 => 30000,
            Self::Ancillary2110_40 => 40000,
        }
    }
}

/// An IP media flow conforming to SMPTE ST 2110.
#[derive(Debug, Clone)]
pub struct IpFlow {
    /// Unique flow identifier.
    pub id: u64,
    /// Source IP address (unicast).
    pub source_ip: String,
    /// Multicast group address (e.g. `239.0.0.1`).
    pub multicast_group: String,
    /// UDP destination port.
    pub port: u16,
    /// Flow type (video, audio, or ancillary).
    pub flow_type: IpFlowType,
    /// Approximate bandwidth in Mbit/s.
    pub bandwidth_mbps: f32,
}

impl IpFlow {
    /// Returns `true` when `multicast_group` is an IPv4 multicast address
    /// (224.0.0.0 – 239.255.255.255).
    #[must_use]
    pub fn is_multicast(&self) -> bool {
        // IPv4 multicast range: first octet 224–239.
        self.multicast_group
            .split('.')
            .next()
            .and_then(|s| s.parse::<u8>().ok())
            .is_some_and(|first_octet| (224..=239).contains(&first_octet))
    }
}

/// A routing table for IP flows with subscriber management.
#[derive(Debug, Clone, Default)]
pub struct IpRoutingTable {
    /// All registered IP flows.
    pub flows: Vec<IpFlow>,
    /// Active subscriptions: (`flow_id`, `subscriber_ip`) pairs.
    pub subscriptions: Vec<(u64, String)>,
}

impl IpRoutingTable {
    /// Creates an empty `IpRoutingTable`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an IP flow to the table.
    ///
    /// If a flow with the same ID already exists it is replaced.
    pub fn add_flow(&mut self, flow: IpFlow) {
        if let Some(existing) = self.flows.iter_mut().find(|f| f.id == flow.id) {
            *existing = flow;
        } else {
            self.flows.push(flow);
        }
    }

    /// Subscribes `subscriber` to flow `flow_id`.
    ///
    /// Returns `false` if the flow does not exist or the subscription already exists.
    pub fn subscribe(&mut self, flow_id: u64, subscriber: &str) -> bool {
        if !self.flows.iter().any(|f| f.id == flow_id) {
            return false;
        }
        let key = (flow_id, subscriber.to_string());
        if self.subscriptions.contains(&key) {
            return false;
        }
        self.subscriptions.push(key);
        true
    }

    /// Removes the subscription of `subscriber` from flow `flow_id`.
    ///
    /// Returns `false` if no such subscription existed.
    pub fn unsubscribe(&mut self, flow_id: u64, subscriber: &str) -> bool {
        let before = self.subscriptions.len();
        self.subscriptions
            .retain(|(fid, sub)| !(*fid == flow_id && sub == subscriber));
        self.subscriptions.len() < before
    }

    /// Returns all subscriber IPs currently subscribed to `flow_id`.
    #[must_use]
    pub fn subscribers_of(&self, flow_id: u64) -> Vec<&str> {
        self.subscriptions
            .iter()
            .filter_map(|(fid, sub)| {
                if *fid == flow_id {
                    Some(sub.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns all flows that `subscriber` is subscribed to.
    #[must_use]
    pub fn flows_for_subscriber(&self, subscriber: &str) -> Vec<&IpFlow> {
        let subscribed_ids: Vec<u64> = self
            .subscriptions
            .iter()
            .filter_map(|(fid, sub)| if sub == subscriber { Some(*fid) } else { None })
            .collect();
        self.flows
            .iter()
            .filter(|f| subscribed_ids.contains(&f.id))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_flow(id: u64, flow_type: IpFlowType) -> IpFlow {
        IpFlow {
            id,
            source_ip: "10.0.0.1".to_string(),
            multicast_group: format!("239.0.0.{id}"),
            port: flow_type.default_port(),
            flow_type,
            bandwidth_mbps: 1000.0,
        }
    }

    #[test]
    fn test_ip_flow_type_default_ports() {
        assert_eq!(IpFlowType::Video2110_20.default_port(), 20000);
        assert_eq!(IpFlowType::Audio2110_30.default_port(), 30000);
        assert_eq!(IpFlowType::Ancillary2110_40.default_port(), 40000);
    }

    #[test]
    fn test_is_multicast_true() {
        let flow = make_flow(1, IpFlowType::Video2110_20);
        assert!(flow.is_multicast());
    }

    #[test]
    fn test_is_multicast_false_unicast() {
        let flow = IpFlow {
            id: 2,
            source_ip: "10.0.0.2".to_string(),
            multicast_group: "10.0.0.50".to_string(),
            port: 20000,
            flow_type: IpFlowType::Video2110_20,
            bandwidth_mbps: 100.0,
        };
        assert!(!flow.is_multicast());
    }

    #[test]
    fn test_add_flow() {
        let mut table = IpRoutingTable::new();
        table.add_flow(make_flow(1, IpFlowType::Video2110_20));
        assert_eq!(table.flows.len(), 1);
    }

    #[test]
    fn test_add_flow_replaces_existing() {
        let mut table = IpRoutingTable::new();
        table.add_flow(make_flow(1, IpFlowType::Video2110_20));
        table.add_flow(make_flow(1, IpFlowType::Audio2110_30));
        assert_eq!(table.flows.len(), 1);
        assert_eq!(table.flows[0].flow_type, IpFlowType::Audio2110_30);
    }

    #[test]
    fn test_subscribe_success() {
        let mut table = IpRoutingTable::new();
        table.add_flow(make_flow(1, IpFlowType::Video2110_20));
        assert!(table.subscribe(1, "192.168.1.100"));
    }

    #[test]
    fn test_subscribe_nonexistent_flow() {
        let mut table = IpRoutingTable::new();
        assert!(!table.subscribe(99, "192.168.1.100"));
    }

    #[test]
    fn test_subscribe_duplicate_returns_false() {
        let mut table = IpRoutingTable::new();
        table.add_flow(make_flow(1, IpFlowType::Video2110_20));
        table.subscribe(1, "192.168.1.100");
        assert!(!table.subscribe(1, "192.168.1.100"));
    }

    #[test]
    fn test_unsubscribe_success() {
        let mut table = IpRoutingTable::new();
        table.add_flow(make_flow(1, IpFlowType::Video2110_20));
        table.subscribe(1, "192.168.1.100");
        assert!(table.unsubscribe(1, "192.168.1.100"));
        assert!(table.subscribers_of(1).is_empty());
    }

    #[test]
    fn test_unsubscribe_nonexistent_returns_false() {
        let mut table = IpRoutingTable::new();
        assert!(!table.unsubscribe(1, "192.168.1.100"));
    }

    #[test]
    fn test_subscribers_of() {
        let mut table = IpRoutingTable::new();
        table.add_flow(make_flow(1, IpFlowType::Video2110_20));
        table.subscribe(1, "10.0.0.10");
        table.subscribe(1, "10.0.0.11");
        let subs = table.subscribers_of(1);
        assert_eq!(subs.len(), 2);
        assert!(subs.contains(&"10.0.0.10"));
        assert!(subs.contains(&"10.0.0.11"));
    }

    #[test]
    fn test_flows_for_subscriber() {
        let mut table = IpRoutingTable::new();
        table.add_flow(make_flow(1, IpFlowType::Video2110_20));
        table.add_flow(make_flow(2, IpFlowType::Audio2110_30));
        table.subscribe(1, "10.0.0.5");
        table.subscribe(2, "10.0.0.5");
        let flows = table.flows_for_subscriber("10.0.0.5");
        assert_eq!(flows.len(), 2);
    }

    #[test]
    fn test_flows_for_subscriber_empty_when_no_subs() {
        let mut table = IpRoutingTable::new();
        table.add_flow(make_flow(1, IpFlowType::Video2110_20));
        assert!(table.flows_for_subscriber("10.0.0.5").is_empty());
    }
}
