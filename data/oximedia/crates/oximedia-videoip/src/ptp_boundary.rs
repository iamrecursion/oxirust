//! PTP (IEEE 1588) boundary clock implementation.
//!
//! Supports grandmaster selection, announce messages, and clock class management
//! for synchronization in SMPTE ST 2110 environments.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// PTP clock class values per IEEE 1588-2019.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum ClockClass {
    /// Primary reference clock (grandmaster).
    Primary = 6,
    /// Holdover with in-spec accuracy.
    HoldoverInSpec = 7,
    /// Holdover out-of-spec.
    HoldoverOutOfSpec = 52,
    /// Degraded: no synchronization source.
    Degraded = 193,
    /// Default / unknown.
    #[default]
    Default = 248,
    /// Slave-only clock.
    SlaveOnly = 255,
}

/// PTP clock accuracy enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum ClockAccuracy {
    /// Within 25 nanoseconds.
    Within25ns = 0x20,
    /// Within 100 nanoseconds.
    Within100ns = 0x21,
    /// Within 250 nanoseconds.
    Within250ns = 0x22,
    /// Within 1 microsecond.
    Within1us = 0x23,
    /// Within 25 microseconds.
    Within25us = 0x25,
    /// Within 100 microseconds.
    Within100us = 0x26,
    /// Unknown accuracy.
    #[default]
    Unknown = 0xFE,
}

/// Unique PTP clock identity (8 bytes / EUI-64).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ClockIdentity(pub [u8; 8]);

impl ClockIdentity {
    /// Create a new clock identity from raw bytes.
    #[must_use]
    pub fn new(bytes: [u8; 8]) -> Self {
        Self(bytes)
    }

    /// Create a dummy identity for testing.
    #[must_use]
    pub fn from_u64(v: u64) -> Self {
        Self(v.to_be_bytes())
    }
}

/// PTP port identity (clock identity + port number).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PortIdentity {
    /// Clock identity.
    pub clock_identity: ClockIdentity,
    /// Port number.
    pub port_number: u16,
}

impl PortIdentity {
    /// Create a new port identity.
    #[must_use]
    pub fn new(clock_identity: ClockIdentity, port_number: u16) -> Self {
        Self {
            clock_identity,
            port_number,
        }
    }
}

/// PTP announce message payload.
#[derive(Debug, Clone)]
pub struct AnnounceMessage {
    /// Source port identity.
    pub source_port_identity: PortIdentity,
    /// Steps removed from the grandmaster.
    pub steps_removed: u16,
    /// Grandmaster clock class.
    pub grandmaster_clock_class: ClockClass,
    /// Grandmaster clock accuracy.
    pub grandmaster_clock_accuracy: ClockAccuracy,
    /// Grandmaster clock variance (AVAR).
    pub grandmaster_offset_scaled_log_variance: u16,
    /// Grandmaster priority 1.
    pub grandmaster_priority1: u8,
    /// Grandmaster priority 2.
    pub grandmaster_priority2: u8,
    /// Grandmaster identity.
    pub grandmaster_identity: ClockIdentity,
    /// Sequence ID.
    pub sequence_id: u16,
    /// Time of reception.
    pub received_at: Instant,
}

impl AnnounceMessage {
    /// Create a new announce message.
    #[must_use]
    pub fn new(
        source_port_identity: PortIdentity,
        grandmaster_identity: ClockIdentity,
        clock_class: ClockClass,
        clock_accuracy: ClockAccuracy,
        priority1: u8,
        priority2: u8,
        steps_removed: u16,
        sequence_id: u16,
    ) -> Self {
        Self {
            source_port_identity,
            steps_removed,
            grandmaster_clock_class: clock_class,
            grandmaster_clock_accuracy: clock_accuracy,
            grandmaster_offset_scaled_log_variance: 0x4E5D,
            grandmaster_priority1: priority1,
            grandmaster_priority2: priority2,
            grandmaster_identity,
            sequence_id,
            received_at: Instant::now(),
        }
    }

    /// Compute the data set comparison vector (clock class, accuracy, variance, priority1, identity, priority2).
    #[must_use]
    pub fn comparison_vector(&self) -> (u8, u8, u16, u8, [u8; 8], u8) {
        (
            self.grandmaster_clock_class as u8,
            self.grandmaster_clock_accuracy as u8,
            self.grandmaster_offset_scaled_log_variance,
            self.grandmaster_priority1,
            self.grandmaster_identity.0,
            self.grandmaster_priority2,
        )
    }
}

/// State of a PTP port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PortState {
    /// Initializing.
    #[default]
    Initializing,
    /// Faulty.
    Faulty,
    /// Disabled.
    Disabled,
    /// Listening for announce messages.
    Listening,
    /// Pre-master (won BMCA, not yet master).
    PreMaster,
    /// Master.
    Master,
    /// Passive (another master is better).
    Passive,
    /// Uncalibrated slave.
    Uncalibrated,
    /// Slave.
    Slave,
}

/// Best master clock algorithm (BMCA) result.
#[derive(Debug, Clone)]
pub struct BmcaResult {
    /// Winning announce message (if any).
    pub best_announce: Option<AnnounceMessage>,
    /// Is this node the grandmaster?
    pub is_grandmaster: bool,
}

/// PTP boundary clock — manages multiple ports and runs BMCA.
#[derive(Debug)]
pub struct PtpBoundaryClock {
    /// Local clock identity.
    pub clock_identity: ClockIdentity,
    /// Local priority 1.
    pub priority1: u8,
    /// Local priority 2.
    pub priority2: u8,
    /// Local clock class.
    pub clock_class: ClockClass,
    /// Local clock accuracy.
    pub clock_accuracy: ClockAccuracy,
    /// Announce receipt timeout (number of announce intervals before timeout).
    pub announce_receipt_timeout: u8,
    /// Port states keyed by port number.
    pub port_states: HashMap<u16, PortState>,
    /// Pending announce messages per foreign master.
    foreign_masters: HashMap<PortIdentity, Vec<AnnounceMessage>>,
    /// Current grandmaster.
    pub current_grandmaster: Option<ClockIdentity>,
}

impl PtpBoundaryClock {
    /// Create a new boundary clock instance.
    #[must_use]
    pub fn new(
        clock_identity: ClockIdentity,
        priority1: u8,
        priority2: u8,
        clock_class: ClockClass,
        clock_accuracy: ClockAccuracy,
    ) -> Self {
        Self {
            clock_identity,
            priority1,
            priority2,
            clock_class,
            clock_accuracy,
            announce_receipt_timeout: 3,
            port_states: HashMap::new(),
            foreign_masters: HashMap::new(),
            current_grandmaster: None,
        }
    }

    /// Add a port to the boundary clock.
    pub fn add_port(&mut self, port_number: u16) {
        self.port_states.insert(port_number, PortState::Listening);
    }

    /// Record an incoming announce message on a port.
    pub fn receive_announce(&mut self, msg: AnnounceMessage) {
        let key = msg.source_port_identity;
        self.foreign_masters.entry(key).or_default().push(msg);
        // Keep only the last 4 announce messages per foreign master.
        if let Some(v) = self.foreign_masters.get_mut(&key) {
            if v.len() > 4 {
                v.remove(0);
            }
        }
    }

    /// Run the Best Master Clock Algorithm (BMCA) and update port states.
    pub fn run_bmca(&mut self) -> BmcaResult {
        let timeout = Duration::from_secs(4);
        let now = Instant::now();

        // Filter out stale foreign masters.
        self.foreign_masters.retain(|_, msgs| {
            msgs.last()
                .is_some_and(|m| now.duration_since(m.received_at) < timeout)
        });

        // Find the best announce among all foreign masters.
        let best = self
            .foreign_masters
            .values()
            .filter_map(|msgs| msgs.last())
            .min_by_key(|m| m.comparison_vector());

        if let Some(b) = best {
            let gm_id = b.grandmaster_identity;
            let cv_foreign = b.comparison_vector();

            // Build local data set comparison vector.
            let local_cv = (
                self.clock_class as u8,
                self.clock_accuracy as u8,
                0x4E5Du16,
                self.priority1,
                self.clock_identity.0,
                self.priority2,
            );

            if cv_foreign < local_cv {
                // A foreign master is better.
                self.current_grandmaster = Some(gm_id);
                // Slave the port that received it; master the rest.
                let slave_port = b.source_port_identity.port_number;
                for (port, state) in &mut self.port_states {
                    *state = if *port == slave_port {
                        PortState::Slave
                    } else {
                        PortState::Master
                    };
                }
                BmcaResult {
                    best_announce: Some(b.clone()),
                    is_grandmaster: false,
                }
            } else {
                self.become_grandmaster();
                BmcaResult {
                    best_announce: None,
                    is_grandmaster: true,
                }
            }
        } else {
            self.become_grandmaster();
            BmcaResult {
                best_announce: None,
                is_grandmaster: true,
            }
        }
    }

    /// Transition all ports to master (this node is grandmaster).
    fn become_grandmaster(&mut self) {
        self.current_grandmaster = Some(self.clock_identity);
        for state in self.port_states.values_mut() {
            *state = PortState::Master;
        }
    }

    /// Expire announce receipt for a port (no announce received in time).
    pub fn expire_announce_receipt(&mut self, port_number: u16) {
        if let Some(state) = self.port_states.get_mut(&port_number) {
            if *state == PortState::Slave {
                *state = PortState::Listening;
            }
        }
    }

    /// Return the number of active foreign masters.
    #[must_use]
    pub fn foreign_master_count(&self) -> usize {
        self.foreign_masters.len()
    }
}

/// Announce message interval (log2 seconds).
#[derive(Debug, Clone, Copy)]
pub struct AnnounceInterval(pub i8);

impl AnnounceInterval {
    /// Convert to a Duration.
    #[must_use]
    pub fn to_duration(self) -> Duration {
        let secs = 2.0_f64.powi(i32::from(self.0));
        Duration::from_secs_f64(secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity(v: u64) -> ClockIdentity {
        ClockIdentity::from_u64(v)
    }

    fn make_port(clock: u64, port: u16) -> PortIdentity {
        PortIdentity::new(make_identity(clock), port)
    }

    fn make_announce(
        source_clock: u64,
        gm_clock: u64,
        class: ClockClass,
        priority1: u8,
    ) -> AnnounceMessage {
        AnnounceMessage::new(
            make_port(source_clock, 1),
            make_identity(gm_clock),
            class,
            ClockAccuracy::Within100ns,
            priority1,
            128,
            0,
            1,
        )
    }

    #[test]
    fn test_clock_identity_roundtrip() {
        let id = ClockIdentity::from_u64(0xDEAD_BEEF_1234_5678);
        assert_eq!(id.0, 0xDEAD_BEEF_1234_5678u64.to_be_bytes());
    }

    #[test]
    fn test_clock_class_ordering() {
        assert!(ClockClass::Primary < ClockClass::HoldoverInSpec);
        assert!(ClockClass::HoldoverInSpec < ClockClass::Degraded);
        assert!(ClockClass::Degraded < ClockClass::Default);
    }

    #[test]
    fn test_announce_message_comparison_vector() {
        let msg = make_announce(1, 2, ClockClass::Primary, 128);
        let cv = msg.comparison_vector();
        assert_eq!(cv.0, ClockClass::Primary as u8);
        assert_eq!(cv.3, 128);
    }

    #[test]
    fn test_boundary_clock_add_port() {
        let mut bc = PtpBoundaryClock::new(
            make_identity(0xAA),
            128,
            128,
            ClockClass::Default,
            ClockAccuracy::Unknown,
        );
        bc.add_port(1);
        bc.add_port(2);
        assert_eq!(bc.port_states.len(), 2);
        assert_eq!(bc.port_states[&1], PortState::Listening);
    }

    #[test]
    fn test_bmca_no_foreign_masters_becomes_grandmaster() {
        let mut bc = PtpBoundaryClock::new(
            make_identity(0xBB),
            128,
            128,
            ClockClass::Default,
            ClockAccuracy::Unknown,
        );
        bc.add_port(1);
        let result = bc.run_bmca();
        assert!(result.is_grandmaster);
        assert!(result.best_announce.is_none());
        assert_eq!(bc.port_states[&1], PortState::Master);
    }

    #[test]
    fn test_bmca_better_foreign_master_makes_slave() {
        let mut bc = PtpBoundaryClock::new(
            make_identity(0xCC),
            128,
            128,
            ClockClass::Default,
            ClockAccuracy::Unknown,
        );
        bc.add_port(1);
        bc.add_port(2);

        // Foreign master with Primary clock class is much better.
        let msg = make_announce(0xDD, 0xDD, ClockClass::Primary, 64);
        bc.receive_announce(msg);

        let result = bc.run_bmca();
        assert!(!result.is_grandmaster);
        assert!(result.best_announce.is_some());
        assert_eq!(bc.port_states[&1], PortState::Slave);
        assert_eq!(bc.port_states[&2], PortState::Master);
    }

    #[test]
    fn test_receive_announce_caps_at_four() {
        let mut bc = PtpBoundaryClock::new(
            make_identity(0xEE),
            128,
            128,
            ClockClass::Default,
            ClockAccuracy::Unknown,
        );
        for i in 0..6u16 {
            let msg = make_announce(0xFF, 0xFF, ClockClass::Primary, 128);
            let mut m = msg;
            m.sequence_id = i;
            bc.receive_announce(m);
        }
        // Only 4 are kept per foreign master.
        let count: usize = bc.foreign_masters.values().map(|v| v.len()).sum();
        assert_eq!(count, 4);
    }

    #[test]
    fn test_foreign_master_count() {
        let mut bc = PtpBoundaryClock::new(
            make_identity(0x11),
            128,
            128,
            ClockClass::Default,
            ClockAccuracy::Unknown,
        );
        bc.receive_announce(make_announce(0x21, 0x21, ClockClass::Primary, 128));
        bc.receive_announce(make_announce(0x22, 0x22, ClockClass::Primary, 128));
        assert_eq!(bc.foreign_master_count(), 2);
    }

    #[test]
    fn test_expire_announce_sets_listening() {
        let mut bc = PtpBoundaryClock::new(
            make_identity(0x33),
            128,
            128,
            ClockClass::Default,
            ClockAccuracy::Unknown,
        );
        bc.add_port(5);
        *bc.port_states.get_mut(&5).expect("should succeed in test") = PortState::Slave;
        bc.expire_announce_receipt(5);
        assert_eq!(bc.port_states[&5], PortState::Listening);
    }

    #[test]
    fn test_expire_announce_non_slave_unchanged() {
        let mut bc = PtpBoundaryClock::new(
            make_identity(0x44),
            128,
            128,
            ClockClass::Default,
            ClockAccuracy::Unknown,
        );
        bc.add_port(3);
        *bc.port_states.get_mut(&3).expect("should succeed in test") = PortState::Master;
        bc.expire_announce_receipt(3);
        assert_eq!(bc.port_states[&3], PortState::Master);
    }

    #[test]
    fn test_announce_interval_duration_positive() {
        let interval = AnnounceInterval(1);
        let d = interval.to_duration();
        assert!((d.as_secs_f64() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_announce_interval_duration_negative() {
        let interval = AnnounceInterval(-1);
        let d = interval.to_duration();
        assert!((d.as_secs_f64() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_port_state_default() {
        let s = PortState::default();
        assert_eq!(s, PortState::Initializing);
    }

    #[test]
    fn test_clock_class_default() {
        let c = ClockClass::default();
        assert_eq!(c, ClockClass::Default);
    }

    #[test]
    fn test_clock_accuracy_default() {
        let a = ClockAccuracy::default();
        assert_eq!(a, ClockAccuracy::Unknown);
    }
}
