//! Genlock synchronization for virtual production.
//!
//! Provides sync signal generation, lock detection, and phase alignment
//! for synchronizing LED walls, cameras, and rendering engines.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::time::Duration;

/// Type of genlock sync signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncSignalType {
    /// Black burst (PAL/NTSC analog composite sync).
    BlackBurst,
    /// Tri-level sync (HD reference).
    TriLevel,
    /// SDI embedded sync.
    SdiEmbedded,
    /// Word clock (audio-derived sync).
    WordClock,
    /// GPS-derived PPS (pulse-per-second).
    Gps,
}

impl SyncSignalType {
    /// Returns a human-readable name for the signal type.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::BlackBurst => "Black Burst (analog composite)",
            Self::TriLevel => "Tri-Level Sync (HD reference)",
            Self::SdiEmbedded => "SDI Embedded Sync",
            Self::WordClock => "Word Clock",
            Self::Gps => "GPS PPS",
        }
    }
}

/// Current lock state of a genlock signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockState {
    /// No signal detected.
    NoSignal,
    /// Signal detected but not yet locked.
    Acquiring,
    /// Locked to the reference signal.
    Locked,
    /// Previously locked; signal lost.
    Lost,
}

impl LockState {
    /// Returns `true` if the genlock is currently locked.
    #[must_use]
    pub fn is_locked(&self) -> bool {
        matches!(self, Self::Locked)
    }
}

/// Phase alignment information relative to the reference signal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseAlignment {
    /// Phase offset in degrees [−180.0, 180.0].
    pub offset_degrees: f32,
    /// Frequency error in parts-per-million.
    pub frequency_error_ppm: f32,
    /// Whether the alignment is within acceptable tolerances.
    pub aligned: bool,
}

impl PhaseAlignment {
    /// Create a phase alignment measurement.
    #[must_use]
    pub fn new(offset_degrees: f32, frequency_error_ppm: f32, tolerance_degrees: f32) -> Self {
        let aligned =
            offset_degrees.abs() <= tolerance_degrees && frequency_error_ppm.abs() <= 10.0;
        Self {
            offset_degrees,
            frequency_error_ppm,
            aligned,
        }
    }

    /// Phase offset expressed as a fraction of a full cycle [−0.5, 0.5].
    #[must_use]
    pub fn phase_fraction(&self) -> f32 {
        self.offset_degrees / 360.0
    }

    /// Phase offset expressed in microseconds for a given frame rate.
    #[must_use]
    pub fn offset_microseconds(&self, fps: f32) -> f32 {
        if fps <= 0.0 {
            return 0.0;
        }
        let frame_period_us = 1_000_000.0 / fps;
        (self.offset_degrees / 360.0) * frame_period_us
    }
}

/// Configuration for a genlock generator.
#[derive(Debug, Clone)]
pub struct GenlockGeneratorConfig {
    /// Type of sync signal to generate.
    pub signal_type: SyncSignalType,
    /// Output frame rate numerator.
    pub fps_num: u32,
    /// Output frame rate denominator.
    pub fps_den: u32,
    /// Phase offset to apply to the generated signal in degrees.
    pub phase_offset_deg: f32,
    /// Whether to enable frame-accurate jitter measurement.
    pub measure_jitter: bool,
}

impl GenlockGeneratorConfig {
    /// Create a default HD configuration (29.97 fps tri-level).
    #[must_use]
    pub fn hd_default() -> Self {
        Self {
            signal_type: SyncSignalType::TriLevel,
            fps_num: 30000,
            fps_den: 1001,
            phase_offset_deg: 0.0,
            measure_jitter: true,
        }
    }

    /// Create a cinema configuration (24 fps tri-level).
    #[must_use]
    pub fn cinema() -> Self {
        Self {
            signal_type: SyncSignalType::TriLevel,
            fps_num: 24,
            fps_den: 1,
            phase_offset_deg: 0.0,
            measure_jitter: true,
        }
    }

    /// Effective frame rate as f64.
    #[must_use]
    pub fn fps(&self) -> f64 {
        f64::from(self.fps_num) / f64::from(self.fps_den)
    }

    /// Frame period as a `Duration`.
    #[must_use]
    pub fn frame_period(&self) -> Duration {
        let nanos = (1_000_000_000.0 / self.fps()) as u64;
        Duration::from_nanos(nanos)
    }
}

impl Default for GenlockGeneratorConfig {
    fn default() -> Self {
        Self::hd_default()
    }
}

/// Genlock generator that emits sync pulses at a configured frame rate.
#[allow(dead_code)]
pub struct GenlockGenerator {
    config: GenlockGeneratorConfig,
    /// Total number of sync pulses emitted.
    pulse_count: u64,
    /// Accumulated jitter in nanoseconds (if measurement enabled).
    jitter_ns_accumulated: f64,
}

impl GenlockGenerator {
    /// Create a new genlock generator.
    #[must_use]
    pub fn new(config: GenlockGeneratorConfig) -> Self {
        Self {
            config,
            pulse_count: 0,
            jitter_ns_accumulated: 0.0,
        }
    }

    /// Simulate emitting a single sync pulse, recording optional jitter.
    pub fn emit_pulse(&mut self, jitter_ns: f64) {
        self.pulse_count += 1;
        if self.config.measure_jitter {
            self.jitter_ns_accumulated += jitter_ns.abs();
        }
    }

    /// Number of pulses emitted.
    #[must_use]
    pub fn pulse_count(&self) -> u64 {
        self.pulse_count
    }

    /// Average jitter in nanoseconds (0 if not measuring).
    #[must_use]
    pub fn average_jitter_ns(&self) -> f64 {
        if !self.config.measure_jitter || self.pulse_count == 0 {
            return 0.0;
        }
        self.jitter_ns_accumulated / self.pulse_count as f64
    }

    /// Configuration reference.
    #[must_use]
    pub fn config(&self) -> &GenlockGeneratorConfig {
        &self.config
    }
}

/// Detects lock state by monitoring incoming sync pulses.
#[allow(dead_code)]
pub struct LockDetector {
    /// Expected frame period.
    expected_period: Duration,
    /// Maximum allowed deviation to stay locked.
    tolerance_ns: u64,
    /// Current lock state.
    state: LockState,
    /// Consecutive in-tolerance pulses required to declare lock.
    lock_threshold: u32,
    /// Current run of in-tolerance pulses.
    consecutive_in_tolerance: u32,
}

impl LockDetector {
    /// Create a lock detector for a given expected period.
    #[must_use]
    pub fn new(expected_period: Duration, tolerance_ns: u64, lock_threshold: u32) -> Self {
        Self {
            expected_period,
            tolerance_ns,
            state: LockState::NoSignal,
            lock_threshold,
            consecutive_in_tolerance: 0,
        }
    }

    /// Process an incoming pulse with the given measured period.
    pub fn process_pulse(&mut self, measured_period: Duration) {
        let expected_ns = self.expected_period.as_nanos() as i64;
        let measured_ns = measured_period.as_nanos() as i64;
        let deviation = (measured_ns - expected_ns).unsigned_abs();

        if deviation <= self.tolerance_ns {
            self.consecutive_in_tolerance += 1;
            if self.consecutive_in_tolerance >= self.lock_threshold {
                self.state = LockState::Locked;
            } else {
                self.state = LockState::Acquiring;
            }
        } else {
            self.consecutive_in_tolerance = 0;
            self.state = match self.state {
                LockState::Locked => LockState::Lost,
                _ => LockState::Acquiring,
            };
        }
    }

    /// Current lock state.
    #[must_use]
    pub fn state(&self) -> LockState {
        self.state
    }

    /// Whether currently locked.
    #[must_use]
    pub fn is_locked(&self) -> bool {
        self.state.is_locked()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_signal_description() {
        assert_eq!(
            SyncSignalType::TriLevel.description(),
            "Tri-Level Sync (HD reference)"
        );
        assert_eq!(
            SyncSignalType::BlackBurst.description(),
            "Black Burst (analog composite)"
        );
    }

    #[test]
    fn test_lock_state_is_locked() {
        assert!(LockState::Locked.is_locked());
        assert!(!LockState::NoSignal.is_locked());
        assert!(!LockState::Acquiring.is_locked());
        assert!(!LockState::Lost.is_locked());
    }

    #[test]
    fn test_phase_alignment_aligned_within_tolerance() {
        let pa = PhaseAlignment::new(1.0, 0.5, 5.0);
        assert!(pa.aligned);
    }

    #[test]
    fn test_phase_alignment_not_aligned_offset_exceeds_tolerance() {
        let pa = PhaseAlignment::new(10.0, 0.0, 5.0);
        assert!(!pa.aligned);
    }

    #[test]
    fn test_phase_alignment_not_aligned_frequency_error() {
        let pa = PhaseAlignment::new(0.0, 15.0, 5.0);
        assert!(!pa.aligned);
    }

    #[test]
    fn test_phase_alignment_phase_fraction() {
        let pa = PhaseAlignment::new(90.0, 0.0, 180.0);
        assert!((pa.phase_fraction() - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_phase_alignment_offset_microseconds_24fps() {
        // 90 degrees at 24fps: frame period = 1_000_000/24 ≈ 41666.7 us; quarter = ~10416.7 us
        let pa = PhaseAlignment::new(90.0, 0.0, 180.0);
        let us = pa.offset_microseconds(24.0);
        let expected = (1_000_000.0_f32 / 24.0) * 0.25;
        assert!((us - expected).abs() < 1.0);
    }

    #[test]
    fn test_phase_alignment_offset_microseconds_zero_fps() {
        let pa = PhaseAlignment::new(90.0, 0.0, 180.0);
        assert_eq!(pa.offset_microseconds(0.0), 0.0);
    }

    #[test]
    fn test_genlock_generator_config_fps() {
        let config = GenlockGeneratorConfig::hd_default();
        // 30000 / 1001 ≈ 29.97
        assert!((config.fps() - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_genlock_generator_config_cinema_fps() {
        let config = GenlockGeneratorConfig::cinema();
        assert!((config.fps() - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_genlock_generator_config_frame_period() {
        let config = GenlockGeneratorConfig::cinema();
        let period = config.frame_period();
        // 24fps → ~41.667ms
        assert!((period.as_secs_f64() - 1.0 / 24.0).abs() < 0.001);
    }

    #[test]
    fn test_genlock_generator_emit_pulse() {
        let mut gen = GenlockGenerator::new(GenlockGeneratorConfig::default());
        gen.emit_pulse(100.0);
        gen.emit_pulse(200.0);
        assert_eq!(gen.pulse_count(), 2);
        // Average jitter should be (100+200)/2 = 150
        assert!((gen.average_jitter_ns() - 150.0).abs() < 1e-5);
    }

    #[test]
    fn test_genlock_generator_no_jitter_measurement() {
        let mut config = GenlockGeneratorConfig::default();
        config.measure_jitter = false;
        let mut gen = GenlockGenerator::new(config);
        gen.emit_pulse(500.0);
        assert_eq!(gen.average_jitter_ns(), 0.0);
    }

    #[test]
    fn test_lock_detector_acquires_then_locks() {
        let period = Duration::from_nanos(41_666_667); // ~24fps
        let mut detector = LockDetector::new(period, 10_000, 3);
        assert_eq!(detector.state(), LockState::NoSignal);

        detector.process_pulse(period);
        assert_eq!(detector.state(), LockState::Acquiring);

        detector.process_pulse(period);
        assert_eq!(detector.state(), LockState::Acquiring);

        detector.process_pulse(period);
        assert_eq!(detector.state(), LockState::Locked);
        assert!(detector.is_locked());
    }

    #[test]
    fn test_lock_detector_loses_lock_on_deviation() {
        let period = Duration::from_nanos(41_666_667);
        let mut detector = LockDetector::new(period, 10_000, 1);
        detector.process_pulse(period);
        assert!(detector.is_locked());

        // Send a pulse with large deviation
        detector.process_pulse(Duration::from_nanos(period.as_nanos() as u64 + 100_000));
        assert_eq!(detector.state(), LockState::Lost);
    }
}
