//! gPTP (IEEE 802.1AS) Generalized Precision Time Protocol implementation.
//!
//! This module provides IEEE 802.1AS support for audio/video bridging (AVB)
//! networks, including the Best Master Clock Algorithm (BMCA) and peer delay
//! mechanism.

/// gPTP domain configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct GptpDomain {
    /// Domain number (0-127)
    pub domain_number: u8,
    /// Priority 1 (lower is better)
    pub priority1: u8,
    /// Priority 2 (tiebreaker for BMCA)
    pub priority2: u8,
    /// Clock class (lower is better)
    pub clock_class: u8,
}

impl GptpDomain {
    /// Create a new gPTP domain with default priorities.
    #[must_use]
    pub fn new(domain_number: u8) -> Self {
        Self {
            domain_number,
            priority1: 128,
            priority2: 128,
            clock_class: 135,
        }
    }

    /// Create a gPTP domain with custom priorities.
    #[must_use]
    pub fn with_priorities(
        domain_number: u8,
        priority1: u8,
        priority2: u8,
        clock_class: u8,
    ) -> Self {
        Self {
            domain_number,
            priority1,
            priority2,
            clock_class,
        }
    }
}

/// gPTP port state machine states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum GptpPortState {
    /// Port is initializing
    Initializing,
    /// Port is listening for Announce messages
    Listening,
    /// Port is a pre-master (candidate)
    PreMaster,
    /// Port is the grandmaster clock source
    Master,
    /// Port is in passive state (not selected)
    Passive,
    /// Port is calibrating its clock
    Uncalibrated,
    /// Port is synchronized to a master
    Slave,
}

impl GptpPortState {
    /// Returns true if this port state is a sync source.
    #[must_use]
    pub fn is_sync_source(&self) -> bool {
        matches!(self, GptpPortState::Master)
    }

    /// Returns true if this port is receiving sync from a master.
    #[must_use]
    pub fn is_synchronized(&self) -> bool {
        matches!(self, GptpPortState::Slave)
    }
}

/// gPTP timestamp representation.
///
/// The `seconds` field uses u64 but only the lower 48 bits are valid per
/// IEEE 802.1AS (u48 semantics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub struct GptpTimestamp {
    /// Seconds since epoch (only lower 48 bits are used per IEEE 802.1AS)
    pub seconds: u64,
    /// Nanoseconds (0 to 999_999_999)
    pub nanoseconds: u32,
}

impl GptpTimestamp {
    /// Create a new gPTP timestamp.
    #[must_use]
    pub fn new(seconds: u64, nanoseconds: u32) -> Self {
        Self {
            seconds: seconds & 0x0000_FFFF_FFFF_FFFF, // Mask to 48 bits
            nanoseconds,
        }
    }

    /// Convert to total nanoseconds.
    #[must_use]
    pub fn to_ns(&self) -> u64 {
        self.seconds
            .saturating_mul(1_000_000_000)
            .saturating_add(u64::from(self.nanoseconds))
    }

    /// Create a gPTP timestamp from total nanoseconds.
    #[must_use]
    pub fn from_ns(ns: u64) -> Self {
        let seconds = ns / 1_000_000_000;
        let nanoseconds = (ns % 1_000_000_000) as u32;
        Self::new(seconds, nanoseconds)
    }
}

/// gPTP Sync message.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct GptpSyncMessage {
    /// Sequence ID for matching Follow_Up
    pub sequence_id: u16,
    /// Port identity (clock identity 8 bytes + port number 2 bytes)
    pub port_id: [u8; 10],
    /// Origin timestamp from grandmaster
    pub origin_timestamp: GptpTimestamp,
    /// Correction field in nanoseconds (scaled ns, stored as i64)
    pub correction_ns: i64,
}

impl GptpSyncMessage {
    /// Create a new Sync message.
    #[must_use]
    pub fn new(
        sequence_id: u16,
        port_id: [u8; 10],
        origin_timestamp: GptpTimestamp,
        correction_ns: i64,
    ) -> Self {
        Self {
            sequence_id,
            port_id,
            origin_timestamp,
            correction_ns,
        }
    }
}

/// gPTP Follow_Up message.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct GptpFollowUpMessage {
    /// Precise origin timestamp (two-step clock)
    pub precise_origin_timestamp: GptpTimestamp,
    /// Cumulative correction in nanoseconds
    pub cumulative_correction_ns: i64,
}

impl GptpFollowUpMessage {
    /// Create a new Follow_Up message.
    #[must_use]
    pub fn new(precise_origin_timestamp: GptpTimestamp, cumulative_correction_ns: i64) -> Self {
        Self {
            precise_origin_timestamp,
            cumulative_correction_ns,
        }
    }
}

/// gPTP peer delay measurement.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct GptpPathDelay {
    /// Mean path delay in nanoseconds (peer-to-peer mechanism)
    pub mean_path_delay_ns: i64,
    /// Neighbor rate ratio (frequency offset relative to neighbor)
    pub neighbor_rate_ratio: f64,
}

impl GptpPathDelay {
    /// Create a new path delay measurement.
    #[must_use]
    pub fn new(mean_path_delay_ns: i64, neighbor_rate_ratio: f64) -> Self {
        Self {
            mean_path_delay_ns,
            neighbor_rate_ratio,
        }
    }

    /// Check if the path delay is valid (non-negative and rate ratio reasonable).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.mean_path_delay_ns >= 0
            && self.neighbor_rate_ratio > 0.9
            && self.neighbor_rate_ratio < 1.1
    }
}

/// gPTP clock maintaining synchronization state.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GptpClock {
    /// Current offset from master in nanoseconds
    pub offset_from_master_ns: i64,
    /// Rate ratio relative to grandmaster
    pub rate_ratio: f64,
    /// Path delay to master
    path_delay: GptpPathDelay,
    /// Port state
    port_state: GptpPortState,
}

impl GptpClock {
    /// Create a new gPTP clock.
    #[must_use]
    pub fn new() -> Self {
        Self {
            offset_from_master_ns: 0,
            rate_ratio: 1.0,
            path_delay: GptpPathDelay::new(0, 1.0),
            port_state: GptpPortState::Initializing,
        }
    }

    /// Update synchronization from a Sync message and its reception time.
    ///
    /// Returns the computed offset from master in nanoseconds.
    pub fn update_from_sync(&mut self, msg: &GptpSyncMessage, recv_time_ns: u64) -> i64 {
        // offset = T2 - T1 - path_delay - correction
        let t1_ns = msg.origin_timestamp.to_ns();
        let path_delay_ns = self.path_delay.mean_path_delay_ns.max(0) as u64;
        let correction_ns = msg.correction_ns.max(0) as u64;

        let recv_adjusted = recv_time_ns.saturating_sub(path_delay_ns + correction_ns);
        self.offset_from_master_ns = recv_adjusted as i64 - t1_ns as i64;
        self.offset_from_master_ns
    }

    /// Update synchronization from a Follow_Up message.
    ///
    /// Returns the computed offset from master in nanoseconds using the
    /// precise timestamp.
    pub fn update_from_follow_up(&mut self, msg: &GptpFollowUpMessage, recv_time_ns: u64) -> i64 {
        let t1_ns = msg.precise_origin_timestamp.to_ns();
        let path_delay_ns = self.path_delay.mean_path_delay_ns.max(0) as u64;
        let correction_ns = msg.cumulative_correction_ns.max(0) as u64;

        let recv_adjusted = recv_time_ns.saturating_sub(path_delay_ns + correction_ns);
        self.offset_from_master_ns = recv_adjusted as i64 - t1_ns as i64;
        self.offset_from_master_ns
    }

    /// Set the current path delay measurement.
    pub fn set_path_delay(&mut self, path_delay: GptpPathDelay) {
        self.path_delay = path_delay;
    }

    /// Set the port state.
    pub fn set_port_state(&mut self, state: GptpPortState) {
        self.port_state = state;
    }

    /// Get the current port state.
    #[must_use]
    pub fn port_state(&self) -> GptpPortState {
        self.port_state
    }
}

impl Default for GptpClock {
    fn default() -> Self {
        Self::new()
    }
}

/// Best Master Clock Algorithm (BMCA) for selecting the grandmaster.
///
/// Compares two gPTP domain configurations according to IEEE 802.1AS
/// priority ordering: priority1, clock_class, priority2.
#[allow(dead_code)]
pub struct BestMasterClockAlgorithm;

impl BestMasterClockAlgorithm {
    /// Compare two domain configurations.
    ///
    /// Returns `Ordering::Less` if `a` is a better (preferred) master than `b`.
    #[must_use]
    pub fn compare(a: &GptpDomain, b: &GptpDomain) -> std::cmp::Ordering {
        // IEEE 802.1AS BMCA: lower priority1 wins
        if a.priority1 != b.priority1 {
            return a.priority1.cmp(&b.priority1);
        }
        // Then lower clock_class wins
        if a.clock_class != b.clock_class {
            return a.clock_class.cmp(&b.clock_class);
        }
        // Then lower priority2 wins
        if a.priority2 != b.priority2 {
            return a.priority2.cmp(&b.priority2);
        }
        // Tie (equal identity or domain number)
        a.domain_number.cmp(&b.domain_number)
    }

    /// Select the best master from a list of domain configurations.
    #[must_use]
    pub fn select_best<'a>(domains: &'a [GptpDomain]) -> Option<&'a GptpDomain> {
        domains.iter().min_by(|a, b| Self::compare(a, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gptp_domain_new() {
        let domain = GptpDomain::new(0);
        assert_eq!(domain.domain_number, 0);
        assert_eq!(domain.priority1, 128);
        assert_eq!(domain.priority2, 128);
        assert_eq!(domain.clock_class, 135);
    }

    #[test]
    fn test_gptp_domain_with_priorities() {
        let domain = GptpDomain::with_priorities(1, 50, 100, 6);
        assert_eq!(domain.domain_number, 1);
        assert_eq!(domain.priority1, 50);
        assert_eq!(domain.priority2, 100);
        assert_eq!(domain.clock_class, 6);
    }

    #[test]
    fn test_port_state_is_sync_source() {
        assert!(GptpPortState::Master.is_sync_source());
        assert!(!GptpPortState::Slave.is_sync_source());
        assert!(!GptpPortState::Passive.is_sync_source());
        assert!(!GptpPortState::Listening.is_sync_source());
        assert!(!GptpPortState::Initializing.is_sync_source());
    }

    #[test]
    fn test_port_state_is_synchronized() {
        assert!(GptpPortState::Slave.is_synchronized());
        assert!(!GptpPortState::Master.is_synchronized());
        assert!(!GptpPortState::Passive.is_synchronized());
    }

    #[test]
    fn test_gptp_timestamp_to_ns() {
        let ts = GptpTimestamp::new(1, 500_000_000);
        assert_eq!(ts.to_ns(), 1_500_000_000);
    }

    #[test]
    fn test_gptp_timestamp_from_ns() {
        let ts = GptpTimestamp::from_ns(1_500_000_000);
        assert_eq!(ts.seconds, 1);
        assert_eq!(ts.nanoseconds, 500_000_000);
    }

    #[test]
    fn test_gptp_timestamp_roundtrip() {
        let original_ns = 123_456_789_012u64;
        let ts = GptpTimestamp::from_ns(original_ns);
        assert_eq!(ts.to_ns(), original_ns);
    }

    #[test]
    fn test_gptp_timestamp_48bit_mask() {
        // Seconds exceeding 48 bits should be masked
        let ts = GptpTimestamp::new(u64::MAX, 0);
        assert!(ts.seconds <= 0x0000_FFFF_FFFF_FFFF);
    }

    #[test]
    fn test_gptp_clock_update_from_sync() {
        let mut clock = GptpClock::new();
        let origin_ts = GptpTimestamp::from_ns(1_000_000_000);
        let msg = GptpSyncMessage::new(1, [0u8; 10], origin_ts, 0);

        // recv_time = 1_000_001_000 ns, no path delay, no correction
        // offset = 1_000_001_000 - 1_000_000_000 = 1000 ns
        let offset = clock.update_from_sync(&msg, 1_000_001_000);
        assert_eq!(offset, 1000);
        assert_eq!(clock.offset_from_master_ns, 1000);
    }

    #[test]
    fn test_gptp_clock_update_with_path_delay() {
        let mut clock = GptpClock::new();
        clock.set_path_delay(GptpPathDelay::new(500, 1.0));

        let origin_ts = GptpTimestamp::from_ns(1_000_000_000);
        let msg = GptpSyncMessage::new(1, [0u8; 10], origin_ts, 0);

        // recv_time = 1_000_001_000, path_delay = 500
        // offset = (1_000_001_000 - 500) - 1_000_000_000 = 500 ns
        let offset = clock.update_from_sync(&msg, 1_000_001_000);
        assert_eq!(offset, 500);
    }

    #[test]
    fn test_gptp_path_delay_validity() {
        assert!(GptpPathDelay::new(1000, 1.0).is_valid());
        assert!(!GptpPathDelay::new(-1, 1.0).is_valid());
        assert!(!GptpPathDelay::new(1000, 0.5).is_valid());
        assert!(!GptpPathDelay::new(1000, 1.5).is_valid());
    }

    #[test]
    fn test_bmca_compare_priority1() {
        let a = GptpDomain::with_priorities(0, 50, 128, 135);
        let b = GptpDomain::with_priorities(0, 100, 128, 135);
        // a has lower priority1, so a < b (a is better master)
        assert_eq!(
            BestMasterClockAlgorithm::compare(&a, &b),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn test_bmca_compare_clock_class() {
        let a = GptpDomain::with_priorities(0, 128, 128, 6);
        let b = GptpDomain::with_priorities(0, 128, 128, 135);
        assert_eq!(
            BestMasterClockAlgorithm::compare(&a, &b),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn test_bmca_compare_priority2() {
        let a = GptpDomain::with_priorities(0, 128, 50, 135);
        let b = GptpDomain::with_priorities(0, 128, 100, 135);
        assert_eq!(
            BestMasterClockAlgorithm::compare(&a, &b),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn test_bmca_select_best() {
        let domains = vec![
            GptpDomain::with_priorities(0, 100, 128, 135),
            GptpDomain::with_priorities(0, 50, 128, 135),
            GptpDomain::with_priorities(0, 200, 128, 135),
        ];
        let best = BestMasterClockAlgorithm::select_best(&domains).expect("should succeed in test");
        assert_eq!(best.priority1, 50);
    }

    #[test]
    fn test_bmca_select_best_empty() {
        let domains: Vec<GptpDomain> = Vec::new();
        assert!(BestMasterClockAlgorithm::select_best(&domains).is_none());
    }

    #[test]
    fn test_gptp_clock_default() {
        let clock = GptpClock::default();
        assert_eq!(clock.offset_from_master_ns, 0);
        assert!((clock.rate_ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gptp_follow_up() {
        let ts = GptpTimestamp::from_ns(2_000_000_000);
        let msg = GptpFollowUpMessage::new(ts, 100);
        assert_eq!(msg.precise_origin_timestamp.to_ns(), 2_000_000_000);
        assert_eq!(msg.cumulative_correction_ns, 100);
    }
}
