//! PTP slave clock implementation with BMCA score and two-step clock formula.

/// 64-bit PTP clock identity, typically derived from a MAC address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PtpClockIdentity {
    /// Raw 8-byte clock identity.
    pub bytes: [u8; 8],
}

impl PtpClockIdentity {
    /// Construct a clock identity from a 48-bit MAC address using the EUI-64 extension.
    ///
    /// The 0xFF-0xFE bytes are inserted in positions 3 and 4.
    #[must_use]
    pub fn from_mac(mac: [u8; 6]) -> Self {
        let bytes = [mac[0], mac[1], mac[2], 0xFF, 0xFE, mac[3], mac[4], mac[5]];
        Self { bytes }
    }

    /// Format the identity as a lower-case hexadecimal string with no separators.
    #[must_use]
    pub fn to_hex(&self) -> String {
        self.bytes
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join("")
    }
}

/// State of a PTP port as defined in IEEE 1588-2019.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtpPortState {
    /// Port is initialising.
    Initializing,
    /// Port has detected a fault.
    Faulty,
    /// Port has been administratively disabled.
    Disabled,
    /// Port is listening for Announce messages.
    Listening,
    /// Port is qualifying itself as a potential master.
    PreMaster,
    /// Port is acting as a master clock.
    Master,
    /// Port is passive (not selected by BMCA).
    Passive,
    /// Port is calibrating to its master.
    Uncalibrated,
    /// Port is locked to a master clock.
    Slave,
}

impl PtpPortState {
    /// Returns `true` if the port is in a synchronized state (Slave).
    #[must_use]
    pub fn is_synchronized(&self) -> bool {
        matches!(self, Self::Slave)
    }
}

/// Signed time offset between a slave and its master.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtpOffset {
    /// Sub-second offset in nanoseconds.
    pub nanoseconds: i64,
    /// Whole-second component of the offset.
    pub seconds: i64,
}

impl PtpOffset {
    /// Total offset expressed purely in nanoseconds.
    #[must_use]
    pub fn total_ns(&self) -> i64 {
        self.seconds
            .saturating_mul(1_000_000_000)
            .saturating_add(self.nanoseconds)
    }

    /// Returns `true` if the absolute total offset is within `max_ns` nanoseconds.
    #[must_use]
    pub fn is_within(&self, max_ns: i64) -> bool {
        self.total_ns().abs() <= max_ns
    }
}

/// BMCA (Best Master Clock Algorithm) score for clock comparison.
///
/// Lower values are better (a clock with a lower score wins the election).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtpBmcaScore {
    /// Priority 1 (configured by the operator).
    pub priority1: u8,
    /// Clock class (lower is better: 6 = GPS-locked, 135 = free-running).
    pub clock_class: u8,
    /// Clock accuracy enumeration.
    pub clock_accuracy: u8,
    /// Offset scaled log variance (lower = more stable).
    pub offset_scaled_log_variance: u16,
    /// Priority 2 (secondary preference).
    pub priority2: u8,
}

impl PtpBmcaScore {
    /// Returns `true` if `self` beats (is a better clock than) `other`.
    ///
    /// Comparison follows the simplified BMCA: priority1, then clock_class,
    /// then clock_accuracy, then offset_scaled_log_variance, then priority2.
    #[must_use]
    pub fn beats(&self, other: &Self) -> bool {
        if self.priority1 != other.priority1 {
            return self.priority1 < other.priority1;
        }
        if self.clock_class != other.clock_class {
            return self.clock_class < other.clock_class;
        }
        if self.clock_accuracy != other.clock_accuracy {
            return self.clock_accuracy < other.clock_accuracy;
        }
        if self.offset_scaled_log_variance != other.offset_scaled_log_variance {
            return self.offset_scaled_log_variance < other.offset_scaled_log_variance;
        }
        self.priority2 < other.priority2
    }
}

/// PTP slave state machine that tracks offset and delay to a master.
pub struct PtpSlave {
    /// Current clock offset from master in nanoseconds.
    pub offset_ns: i64,
    /// Current path delay estimate in nanoseconds.
    pub delay_ns: u64,
    /// Current port state.
    pub state: PtpPortState,
}

impl PtpSlave {
    /// Create a new `PtpSlave` in the `Listening` state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            offset_ns: 0,
            delay_ns: 0,
            state: PtpPortState::Listening,
        }
    }

    /// Update the offset and delay using the two-step clock formula.
    ///
    /// | Symbol | Meaning |
    /// |--------|---------|
    /// | t1     | Sync message departure (master TX), in nanoseconds |
    /// | t2     | Sync message arrival (slave RX), in nanoseconds |
    /// | t3     | Delay_Req departure (slave TX), in nanoseconds |
    /// | t4     | Delay_Req arrival (master RX), in nanoseconds |
    ///
    /// - `mean_path_delay = ((t2 - t1) + (t4 - t3)) / 2`
    /// - `offset_from_master = (t2 - t1) - mean_path_delay`
    pub fn update_offset(&mut self, t1: u64, t2: u64, t3: u64, t4: u64) {
        let forward_delay = (t2 as i64).wrapping_sub(t1 as i64);
        let reverse_delay = (t4 as i64).wrapping_sub(t3 as i64);
        let mean_path_delay = (forward_delay + reverse_delay) / 2;

        self.offset_ns = forward_delay - mean_path_delay;
        self.delay_ns = mean_path_delay.unsigned_abs();
        self.state = PtpPortState::Slave;
    }
}

impl Default for PtpSlave {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_identity_from_mac() {
        let mac = [0x00, 0x1A, 0x2B, 0x3C, 0x4D, 0x5E];
        let id = PtpClockIdentity::from_mac(mac);
        assert_eq!(id.bytes[0], 0x00);
        assert_eq!(id.bytes[1], 0x1A);
        assert_eq!(id.bytes[2], 0x2B);
        assert_eq!(id.bytes[3], 0xFF);
        assert_eq!(id.bytes[4], 0xFE);
        assert_eq!(id.bytes[5], 0x3C);
        assert_eq!(id.bytes[6], 0x4D);
        assert_eq!(id.bytes[7], 0x5E);
    }

    #[test]
    fn test_clock_identity_to_hex_length() {
        let mac = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
        let id = PtpClockIdentity::from_mac(mac);
        let hex = id.to_hex();
        // 8 bytes × 2 hex digits each = 16 chars
        assert_eq!(hex.len(), 16);
    }

    #[test]
    fn test_clock_identity_to_hex_content() {
        let mac = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55];
        let id = PtpClockIdentity::from_mac(mac);
        let hex = id.to_hex();
        assert!(hex.starts_with("001122"));
        assert!(hex.contains("fffe"));
        assert!(hex.ends_with("334455"));
    }

    #[test]
    fn test_port_state_is_synchronized_slave() {
        assert!(PtpPortState::Slave.is_synchronized());
    }

    #[test]
    fn test_port_state_is_synchronized_other() {
        assert!(!PtpPortState::Master.is_synchronized());
        assert!(!PtpPortState::Listening.is_synchronized());
        assert!(!PtpPortState::Uncalibrated.is_synchronized());
        assert!(!PtpPortState::Faulty.is_synchronized());
    }

    #[test]
    fn test_ptp_offset_total_ns_positive() {
        let offset = PtpOffset {
            nanoseconds: 500_000_000,
            seconds: 1,
        };
        assert_eq!(offset.total_ns(), 1_500_000_000);
    }

    #[test]
    fn test_ptp_offset_total_ns_negative() {
        let offset = PtpOffset {
            nanoseconds: -500_000_000,
            seconds: -1,
        };
        assert_eq!(offset.total_ns(), -1_500_000_000);
    }

    #[test]
    fn test_ptp_offset_is_within() {
        let offset = PtpOffset {
            nanoseconds: 100,
            seconds: 0,
        };
        assert!(offset.is_within(200));
        assert!(!offset.is_within(50));
    }

    #[test]
    fn test_bmca_score_beats_priority1() {
        let better = PtpBmcaScore {
            priority1: 1,
            clock_class: 6,
            clock_accuracy: 0x20,
            offset_scaled_log_variance: 0x4E5D,
            priority2: 128,
        };
        let worse = PtpBmcaScore {
            priority1: 2,
            clock_class: 6,
            clock_accuracy: 0x20,
            offset_scaled_log_variance: 0x4E5D,
            priority2: 128,
        };
        assert!(better.beats(&worse));
        assert!(!worse.beats(&better));
    }

    #[test]
    fn test_bmca_score_beats_clock_class() {
        let a = PtpBmcaScore {
            priority1: 128,
            clock_class: 6,
            clock_accuracy: 0x20,
            offset_scaled_log_variance: 0,
            priority2: 128,
        };
        let b = PtpBmcaScore {
            priority1: 128,
            clock_class: 135,
            clock_accuracy: 0x20,
            offset_scaled_log_variance: 0,
            priority2: 128,
        };
        assert!(a.beats(&b));
    }

    #[test]
    fn test_ptp_slave_initial_state() {
        let slave = PtpSlave::new();
        assert_eq!(slave.state, PtpPortState::Listening);
        assert_eq!(slave.offset_ns, 0);
    }

    #[test]
    fn test_ptp_slave_update_offset_symmetric() {
        // Symmetric path: t1=100ns, t2=200ns (100ns delay), t3=210ns, t4=310ns (100ns delay)
        // mean_path_delay = ((200-100) + (310-210)) / 2 = 100
        // offset = (200-100) - 100 = 0
        let mut slave = PtpSlave::new();
        slave.update_offset(100, 200, 210, 310);
        assert_eq!(slave.offset_ns, 0);
        assert_eq!(slave.delay_ns, 100);
        assert_eq!(slave.state, PtpPortState::Slave);
    }

    #[test]
    fn test_ptp_slave_update_offset_nonzero() {
        // t1=1000, t2=1050, t3=1060, t4=1100
        // forward = 50, reverse = 40, mean = 45
        // offset = 50 - 45 = 5
        let mut slave = PtpSlave::new();
        slave.update_offset(1000, 1050, 1060, 1100);
        assert_eq!(slave.offset_ns, 5);
        assert_eq!(slave.delay_ns, 45);
    }

    #[test]
    fn test_ptp_slave_default() {
        let slave = PtpSlave::default();
        assert_eq!(slave.state, PtpPortState::Listening);
    }
}
