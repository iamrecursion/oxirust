//! PTP dataset definitions.
//!
//! Implements the datasets required by IEEE 1588-2019.

use super::message::ClockQuality;
use super::{ClockIdentity, PortIdentity};

/// Default dataset - describes the clock's own properties.
#[derive(Debug, Clone)]
pub struct DefaultDataSet {
    /// Whether this is a two-step clock
    pub two_step_flag: bool,
    /// Clock identity
    pub clock_identity: ClockIdentity,
    /// Number of PTP ports
    pub number_ports: u16,
    /// Clock quality
    pub clock_quality: ClockQuality,
    /// Priority 1
    pub priority1: u8,
    /// Priority 2
    pub priority2: u8,
    /// Domain number
    pub domain_number: u8,
    /// Slave only flag
    pub slave_only: bool,
}

impl DefaultDataSet {
    /// Create a new default dataset with default values.
    #[must_use]
    pub fn new(clock_identity: ClockIdentity) -> Self {
        Self {
            two_step_flag: true, // Most implementations use two-step
            clock_identity,
            number_ports: 1,
            clock_quality: ClockQuality {
                clock_class: 248,                   // Default clock class
                clock_accuracy: 0xFE,               // Unknown accuracy
                offset_scaled_log_variance: 0xFFFF, // Unknown variance
            },
            priority1: 128, // Default priority
            priority2: 128, // Default priority
            domain_number: 0,
            slave_only: false,
        }
    }

    /// Set this clock as a grandmaster-capable clock.
    pub fn set_grandmaster_capable(&mut self, clock_class: u8, accuracy: u8) {
        self.clock_quality.clock_class = clock_class;
        self.clock_quality.clock_accuracy = accuracy;
        self.clock_quality.offset_scaled_log_variance = 0x4000; // ~1 microsecond
        self.priority1 = 128;
        self.priority2 = 128;
    }

    /// Set this clock as slave-only.
    pub fn set_slave_only(&mut self) {
        self.slave_only = true;
        self.clock_quality.clock_class = 255; // Slave-only
    }
}

/// Current dataset - describes the clock's current state.
#[derive(Debug, Clone, Default)]
pub struct CurrentDataSet {
    /// Steps removed from grandmaster
    pub steps_removed: u16,
    /// Offset from master (nanoseconds)
    pub offset_from_master: i64,
    /// Mean path delay (nanoseconds)
    pub mean_path_delay: i64,
}

/// Parent dataset - describes the current master.
#[derive(Debug, Clone)]
pub struct ParentDataSet {
    /// Parent port identity
    pub parent_port_identity: PortIdentity,
    /// Parent stats (sync messages received, etc.)
    pub parent_stats: bool,
    /// Observed parent offset scaled log variance
    pub observed_parent_offset_scaled_log_variance: u16,
    /// Observed parent clock phase change rate
    pub observed_parent_clock_phase_change_rate: i32,
    /// Grandmaster identity
    pub grandmaster_identity: ClockIdentity,
    /// Grandmaster clock quality
    pub grandmaster_clock_quality: ClockQuality,
    /// Grandmaster priority 1
    pub grandmaster_priority1: u8,
    /// Grandmaster priority 2
    pub grandmaster_priority2: u8,
}

impl ParentDataSet {
    /// Create a parent dataset from local clock (when we are master).
    #[must_use]
    pub fn from_local(local: &DefaultDataSet) -> Self {
        Self {
            parent_port_identity: PortIdentity::new(local.clock_identity, 0),
            parent_stats: false,
            observed_parent_offset_scaled_log_variance: 0xFFFF,
            observed_parent_clock_phase_change_rate: 0,
            grandmaster_identity: local.clock_identity,
            grandmaster_clock_quality: local.clock_quality,
            grandmaster_priority1: local.priority1,
            grandmaster_priority2: local.priority2,
        }
    }
}

/// Time properties dataset - describes UTC and time properties.
#[derive(Debug, Clone)]
pub struct TimePropertiesDataSet {
    /// Current UTC offset (seconds)
    pub current_utc_offset: i16,
    /// Current UTC offset valid flag
    pub current_utc_offset_valid: bool,
    /// Leap 59 flag
    pub leap59: bool,
    /// Leap 61 flag
    pub leap61: bool,
    /// Time traceable flag
    pub time_traceable: bool,
    /// Frequency traceable flag
    pub frequency_traceable: bool,
    /// PTP timescale flag
    pub ptp_timescale: bool,
    /// Time source
    pub time_source: TimeSource,
}

impl Default for TimePropertiesDataSet {
    fn default() -> Self {
        Self {
            current_utc_offset: 37, // As of 2017
            current_utc_offset_valid: true,
            leap59: false,
            leap61: false,
            time_traceable: false,
            frequency_traceable: false,
            ptp_timescale: true,
            time_source: TimeSource::InternalOscillator,
        }
    }
}

/// Time source enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TimeSource {
    /// Atomic clock
    AtomicClock = 0x10,
    /// GPS
    Gps = 0x20,
    /// Terrestrial radio
    TerrestrialRadio = 0x30,
    /// PTP
    Ptp = 0x40,
    /// NTP
    Ntp = 0x50,
    /// Hand set
    HandSet = 0x60,
    /// Other
    Other = 0x90,
    /// Internal oscillator
    InternalOscillator = 0xA0,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_dataset_creation() {
        let clock_id = ClockIdentity([1, 2, 3, 4, 5, 6, 7, 8]);
        let dataset = DefaultDataSet::new(clock_id);

        assert_eq!(dataset.clock_identity, clock_id);
        assert_eq!(dataset.priority1, 128);
        assert_eq!(dataset.priority2, 128);
        assert!(!dataset.slave_only);
    }

    #[test]
    fn test_grandmaster_capable() {
        let clock_id = ClockIdentity([1, 2, 3, 4, 5, 6, 7, 8]);
        let mut dataset = DefaultDataSet::new(clock_id);

        dataset.set_grandmaster_capable(6, 0x20);
        assert_eq!(dataset.clock_quality.clock_class, 6);
        assert_eq!(dataset.clock_quality.clock_accuracy, 0x20);
    }

    #[test]
    fn test_slave_only() {
        let clock_id = ClockIdentity([1, 2, 3, 4, 5, 6, 7, 8]);
        let mut dataset = DefaultDataSet::new(clock_id);

        dataset.set_slave_only();
        assert!(dataset.slave_only);
        assert_eq!(dataset.clock_quality.clock_class, 255);
    }

    #[test]
    fn test_parent_from_local() {
        let clock_id = ClockIdentity([1, 2, 3, 4, 5, 6, 7, 8]);
        let local = DefaultDataSet::new(clock_id);
        let parent = ParentDataSet::from_local(&local);

        assert_eq!(parent.grandmaster_identity, clock_id);
        assert_eq!(parent.grandmaster_priority1, 128);
    }
}
