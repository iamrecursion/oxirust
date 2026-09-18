//! PTP Transparent Clock implementation.
//!
//! Transparent clocks measure and correct for residence time and path delay.

use super::message::Header;
use super::{ClockIdentity, PtpTimestamp};
use crate::error::{TimeSyncError, TimeSyncResult};
use std::collections::HashMap;

/// Transparent clock type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransparentClockType {
    /// End-to-end transparent clock
    E2E,
    /// Peer-to-peer transparent clock
    P2P,
}

/// Transparent clock implementation.
pub struct TransparentClock {
    /// Clock identity
    clock_identity: ClockIdentity,
    /// Clock type
    clock_type: TransparentClockType,
    /// Ingress timestamps (keyed by sequence ID)
    ingress_timestamps: HashMap<u16, PtpTimestamp>,
    /// Maximum entries in timestamp cache
    max_cache_entries: usize,
}

impl TransparentClock {
    /// Create a new transparent clock.
    #[must_use]
    pub fn new(clock_identity: ClockIdentity, clock_type: TransparentClockType) -> Self {
        Self {
            clock_identity,
            clock_type,
            ingress_timestamps: HashMap::new(),
            max_cache_entries: 1000,
        }
    }

    /// Record ingress timestamp for a message.
    pub fn record_ingress(&mut self, sequence_id: u16, timestamp: PtpTimestamp) {
        // Clean up old entries if cache is too large
        if self.ingress_timestamps.len() >= self.max_cache_entries {
            self.ingress_timestamps.clear();
        }

        self.ingress_timestamps.insert(sequence_id, timestamp);
    }

    /// Update correction field for a message on egress.
    pub fn update_correction(
        &mut self,
        header: &mut Header,
        egress_timestamp: PtpTimestamp,
    ) -> TimeSyncResult<()> {
        if let Some(ingress_timestamp) = self.ingress_timestamps.get(&header.sequence_id) {
            // Calculate residence time
            let residence_time = egress_timestamp.diff(ingress_timestamp);

            // Update correction field (in units of nanoseconds * 2^16)
            let correction_increment = (i128::from(residence_time) << 16) as i64;
            header.correction_field = header
                .correction_field
                .checked_add(correction_increment)
                .ok_or(TimeSyncError::Overflow)?;

            // Remove from cache
            self.ingress_timestamps.remove(&header.sequence_id);
        }

        Ok(())
    }

    /// Get clock type.
    #[must_use]
    pub fn clock_type(&self) -> TransparentClockType {
        self.clock_type
    }

    /// Get clock identity.
    #[must_use]
    pub fn clock_identity(&self) -> ClockIdentity {
        self.clock_identity
    }
}

/// Peer delay measurement for P2P transparent clocks.
#[derive(Debug, Clone)]
pub struct PeerDelay {
    /// Peer port identity
    pub peer_port: u16,
    /// Measured delay (nanoseconds)
    pub delay: i64,
    /// Timestamp of measurement
    pub measured_at: PtpTimestamp,
}

impl PeerDelay {
    /// Create a new peer delay measurement.
    #[must_use]
    pub fn new(peer_port: u16, delay: i64, measured_at: PtpTimestamp) -> Self {
        Self {
            peer_port,
            delay,
            measured_at,
        }
    }

    /// Check if measurement is stale (older than threshold).
    #[must_use]
    pub fn is_stale(&self, now: &PtpTimestamp, threshold_ns: i64) -> bool {
        let age = now.diff(&self.measured_at);
        age > threshold_ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transparent_clock_creation() {
        let clock_id = ClockIdentity::random();
        let tc = TransparentClock::new(clock_id, TransparentClockType::E2E);

        assert_eq!(tc.clock_identity(), clock_id);
        assert_eq!(tc.clock_type(), TransparentClockType::E2E);
    }

    #[test]
    fn test_residence_time_calculation() {
        let clock_id = ClockIdentity::random();
        let mut tc = TransparentClock::new(clock_id, TransparentClockType::E2E);

        let ingress = PtpTimestamp::new(1000, 0).expect("should succeed in test");
        let egress = PtpTimestamp::new(1000, 500_000).expect("should succeed in test");

        tc.record_ingress(1, ingress);

        let mut header = Header {
            message_type: super::super::message::MessageType::Sync,
            version: 2,
            message_length: 44,
            domain: super::super::Domain::DEFAULT,
            flags: super::super::message::Flags::default(),
            correction_field: 0,
            source_port_identity: super::super::PortIdentity::new(clock_id, 1),
            sequence_id: 1,
            control: 0,
            log_message_interval: 0,
        };

        tc.update_correction(&mut header, egress)
            .expect("should succeed in test");

        // Residence time is 500,000 ns = 500 microseconds
        // Correction field is in units of nanoseconds * 2^16
        let expected_correction = (500_000_i128 << 16) as i64;
        assert_eq!(header.correction_field, expected_correction);
    }

    #[test]
    fn test_peer_delay_staleness() {
        let now = PtpTimestamp::new(1000, 0).expect("should succeed in test");
        let measured_at = PtpTimestamp::new(999, 0).expect("should succeed in test");

        let delay = PeerDelay::new(1, 1000, measured_at);

        assert!(delay.is_stale(&now, 500_000_000)); // 0.5 second threshold
        assert!(!delay.is_stale(&now, 2_000_000_000)); // 2 second threshold
    }
}
