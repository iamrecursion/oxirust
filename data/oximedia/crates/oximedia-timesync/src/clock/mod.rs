//! Clock discipline and synchronization algorithms.

pub mod discipline;
pub mod drift;
pub mod holdover;
pub mod offset;
pub mod selection;

use std::time::Duration;

/// Clock synchronization state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncState {
    /// Unsynchronized
    Unsync,
    /// Synchronizing
    Syncing,
    /// Synchronized
    Synced,
    /// Holdover mode
    Holdover,
}

/// Clock source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClockSource {
    /// PTP
    Ptp,
    /// NTP
    Ntp,
    /// GPS
    Gps,
    /// Timecode (LTC/MTC)
    Timecode,
    /// System clock
    System,
    /// Free-running
    FreeRunning,
}

/// Clock statistics.
#[derive(Debug, Clone)]
pub struct ClockStats {
    /// Current offset from reference (nanoseconds)
    pub offset_ns: i64,
    /// Estimated frequency offset (ppb - parts per billion)
    pub freq_offset_ppb: f64,
    /// Jitter (nanoseconds)
    pub jitter_ns: u64,
    /// Time in current state
    pub state_time: Duration,
    /// Synchronization state
    pub state: SyncState,
    /// Clock source
    pub source: ClockSource,
}

impl Default for ClockStats {
    fn default() -> Self {
        Self {
            offset_ns: 0,
            freq_offset_ppb: 0.0,
            jitter_ns: 0,
            state_time: Duration::ZERO,
            state: SyncState::Unsync,
            source: ClockSource::System,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_state() {
        let state = SyncState::Synced;
        assert_eq!(state, SyncState::Synced);
    }

    #[test]
    fn test_clock_source() {
        let source = ClockSource::Ptp;
        assert_eq!(source, ClockSource::Ptp);
    }

    #[test]
    fn test_clock_stats_default() {
        let stats = ClockStats::default();
        assert_eq!(stats.offset_ns, 0);
        assert_eq!(stats.state, SyncState::Unsync);
    }
}
