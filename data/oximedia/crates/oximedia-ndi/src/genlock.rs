//! Genlock (generator lock) synchronization for NDI streams.
//!
//! Provides reference-clock locking so that multiple NDI senders can
//! produce frames aligned to a common timing reference.  Supports
//! software-based genlock using NTP-style offset calculation or a local
//! master-clock model.

#![allow(dead_code)]

use std::fmt;
use std::time::{Duration, Instant};

/// Genlock reference source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenlockSource {
    /// Free-running (no genlock).
    FreeRun,
    /// Lock to an internal software reference.
    Internal,
    /// Lock to an external NDI source by name hash.
    ExternalNdi,
    /// Lock to an NTP-derived clock.
    Ntp,
}

impl fmt::Display for GenlockSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FreeRun => write!(f, "FreeRun"),
            Self::Internal => write!(f, "Internal"),
            Self::ExternalNdi => write!(f, "ExternalNDI"),
            Self::Ntp => write!(f, "NTP"),
        }
    }
}

/// Phase relationship between the local clock and the reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseOffset {
    /// Signed offset in microseconds (positive = local is ahead).
    pub offset_us: i64,
    /// Jitter (standard deviation) in microseconds.
    pub jitter_us: f64,
    /// Number of measurement samples used.
    pub sample_count: u32,
}

impl PhaseOffset {
    /// Create a zero-offset measurement.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            offset_us: 0,
            jitter_us: 0.0,
            sample_count: 0,
        }
    }

    /// Whether the offset is within acceptable tolerance.
    #[must_use]
    pub fn is_locked(&self, tolerance_us: i64) -> bool {
        self.offset_us.abs() <= tolerance_us
    }

    /// Magnitude of offset in microseconds.
    #[must_use]
    pub fn magnitude_us(&self) -> u64 {
        self.offset_us.unsigned_abs()
    }
}

/// Genlock configuration parameters.
#[derive(Debug, Clone)]
pub struct GenlockConfig {
    /// Reference source.
    pub source: GenlockSource,
    /// Target frame rate numerator.
    pub fps_num: u32,
    /// Target frame rate denominator.
    pub fps_den: u32,
    /// Phase tolerance in microseconds before declaring lock loss.
    pub tolerance_us: i64,
    /// Low-pass filter coefficient for offset smoothing (0.0 .. 1.0).
    pub filter_alpha: f64,
    /// Maximum consecutive lock-loss measurements before alarm.
    pub max_unlock_count: u32,
}

impl Default for GenlockConfig {
    fn default() -> Self {
        Self {
            source: GenlockSource::Internal,
            fps_num: 30,
            fps_den: 1,
            tolerance_us: 500,
            filter_alpha: 0.1,
            max_unlock_count: 10,
        }
    }
}

impl GenlockConfig {
    /// Frame interval in microseconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn frame_interval_us(&self) -> f64 {
        if self.fps_num == 0 {
            return 0.0;
        }
        (self.fps_den as f64 / self.fps_num as f64) * 1_000_000.0
    }
}

/// Genlock state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenlockState {
    /// Searching for reference.
    Acquiring,
    /// Phase locked to reference.
    Locked,
    /// Lost lock — attempting to re-acquire.
    Unlocked,
}

/// The genlock tracker that maintains phase alignment.
#[derive(Debug)]
pub struct GenlockTracker {
    /// Current configuration.
    config: GenlockConfig,
    /// Current state.
    state: GenlockState,
    /// Smoothed offset in microseconds.
    smoothed_offset_us: f64,
    /// Running jitter estimate.
    jitter_us: f64,
    /// Number of consecutive out-of-tolerance measurements.
    unlock_streak: u32,
    /// Total measurements taken.
    total_samples: u64,
    /// Timestamp of the last measurement.
    last_measure: Option<Instant>,
}

impl GenlockTracker {
    /// Create a new tracker with the given configuration.
    #[must_use]
    pub fn new(config: GenlockConfig) -> Self {
        Self {
            config,
            state: GenlockState::Acquiring,
            smoothed_offset_us: 0.0,
            jitter_us: 0.0,
            unlock_streak: 0,
            total_samples: 0,
            last_measure: None,
        }
    }

    /// Current genlock state.
    #[must_use]
    pub fn state(&self) -> GenlockState {
        self.state
    }

    /// Whether genlock is currently locked.
    #[must_use]
    pub fn is_locked(&self) -> bool {
        self.state == GenlockState::Locked
    }

    /// Current smoothed offset in microseconds.
    #[must_use]
    pub fn smoothed_offset_us(&self) -> f64 {
        self.smoothed_offset_us
    }

    /// Current jitter estimate in microseconds.
    #[must_use]
    pub fn jitter_us(&self) -> f64 {
        self.jitter_us
    }

    /// Total measurement count.
    #[must_use]
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Feed a new phase offset measurement.
    ///
    /// Updates the smoothed offset and jitter estimate, and transitions
    /// the state machine accordingly.
    #[allow(clippy::cast_precision_loss)]
    pub fn feed(&mut self, raw_offset_us: i64) {
        let alpha = self.config.filter_alpha;
        let raw = raw_offset_us as f64;

        // Exponential moving average for offset
        self.smoothed_offset_us = alpha * raw + (1.0 - alpha) * self.smoothed_offset_us;

        // Jitter estimate: exponential moving average of |error|
        let err = (raw - self.smoothed_offset_us).abs();
        self.jitter_us = alpha * err + (1.0 - alpha) * self.jitter_us;

        self.total_samples += 1;
        self.last_measure = Some(Instant::now());

        // State transitions
        let within_tolerance =
            (self.smoothed_offset_us.round() as i64).abs() <= self.config.tolerance_us;
        if within_tolerance {
            self.unlock_streak = 0;
            if self.state != GenlockState::Locked {
                self.state = GenlockState::Locked;
            }
        } else {
            self.unlock_streak += 1;
            if self.unlock_streak >= self.config.max_unlock_count {
                self.state = GenlockState::Unlocked;
            }
        }
    }

    /// Get the current phase offset snapshot.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn phase_offset(&self) -> PhaseOffset {
        PhaseOffset {
            offset_us: self.smoothed_offset_us.round() as i64,
            jitter_us: self.jitter_us,
            sample_count: self.total_samples.min(u64::from(u32::MAX)) as u32,
        }
    }

    /// Reset the tracker to the acquiring state.
    pub fn reset(&mut self) {
        self.state = GenlockState::Acquiring;
        self.smoothed_offset_us = 0.0;
        self.jitter_us = 0.0;
        self.unlock_streak = 0;
        self.total_samples = 0;
        self.last_measure = None;
    }

    /// Duration since the last measurement, if any.
    #[must_use]
    pub fn time_since_last_measure(&self) -> Option<Duration> {
        self.last_measure.map(|t| t.elapsed())
    }
}

/// Report summarising genlock status for logging / monitoring.
#[derive(Debug, Clone)]
pub struct GenlockReport {
    /// Current state.
    pub state: GenlockState,
    /// Current smoothed offset (us).
    pub offset_us: f64,
    /// Current jitter (us).
    pub jitter_us: f64,
    /// Total samples processed.
    pub total_samples: u64,
    /// Reference source type.
    pub source: GenlockSource,
}

impl GenlockTracker {
    /// Generate a snapshot report.
    #[must_use]
    pub fn report(&self) -> GenlockReport {
        GenlockReport {
            state: self.state,
            offset_us: self.smoothed_offset_us,
            jitter_us: self.jitter_us,
            total_samples: self.total_samples,
            source: self.config.source,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genlock_source_display() {
        assert_eq!(GenlockSource::FreeRun.to_string(), "FreeRun");
        assert_eq!(GenlockSource::Ntp.to_string(), "NTP");
    }

    #[test]
    fn test_phase_offset_zero() {
        let po = PhaseOffset::zero();
        assert_eq!(po.offset_us, 0);
        assert!(po.is_locked(100));
    }

    #[test]
    fn test_phase_offset_locked() {
        let po = PhaseOffset {
            offset_us: 50,
            jitter_us: 10.0,
            sample_count: 5,
        };
        assert!(po.is_locked(100));
        assert!(!po.is_locked(30));
    }

    #[test]
    fn test_phase_offset_magnitude() {
        let po = PhaseOffset {
            offset_us: -123,
            jitter_us: 0.0,
            sample_count: 1,
        };
        assert_eq!(po.magnitude_us(), 123);
    }

    #[test]
    fn test_default_config() {
        let cfg = GenlockConfig::default();
        assert_eq!(cfg.source, GenlockSource::Internal);
        assert_eq!(cfg.fps_num, 30);
        assert_eq!(cfg.tolerance_us, 500);
    }

    #[test]
    fn test_frame_interval() {
        let cfg = GenlockConfig {
            fps_num: 60,
            fps_den: 1,
            ..GenlockConfig::default()
        };
        let interval = cfg.frame_interval_us();
        assert!((interval - 16666.666).abs() < 1.0);
    }

    #[test]
    fn test_frame_interval_zero_fps() {
        let cfg = GenlockConfig {
            fps_num: 0,
            fps_den: 1,
            ..GenlockConfig::default()
        };
        assert!((cfg.frame_interval_us()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tracker_initial_state() {
        let tracker = GenlockTracker::new(GenlockConfig::default());
        assert_eq!(tracker.state(), GenlockState::Acquiring);
        assert!(!tracker.is_locked());
        assert_eq!(tracker.total_samples(), 0);
    }

    #[test]
    fn test_tracker_locks_on_small_offset() {
        let mut tracker = GenlockTracker::new(GenlockConfig::default());
        for _ in 0..20 {
            tracker.feed(10);
        }
        assert!(tracker.is_locked());
    }

    #[test]
    fn test_tracker_unlocks_on_large_offset() {
        let cfg = GenlockConfig {
            max_unlock_count: 3,
            tolerance_us: 100,
            filter_alpha: 1.0, // no smoothing so raw offset passes through
            ..GenlockConfig::default()
        };
        let mut tracker = GenlockTracker::new(cfg);
        // First lock it
        for _ in 0..5 {
            tracker.feed(0);
        }
        assert!(tracker.is_locked());
        // Now feed large offsets
        for _ in 0..5 {
            tracker.feed(50_000);
        }
        assert_eq!(tracker.state(), GenlockState::Unlocked);
    }

    #[test]
    fn test_tracker_reset() {
        let mut tracker = GenlockTracker::new(GenlockConfig::default());
        tracker.feed(100);
        tracker.feed(200);
        tracker.reset();
        assert_eq!(tracker.state(), GenlockState::Acquiring);
        assert_eq!(tracker.total_samples(), 0);
    }

    #[test]
    fn test_report() {
        let mut tracker = GenlockTracker::new(GenlockConfig::default());
        tracker.feed(42);
        let report = tracker.report();
        assert_eq!(report.source, GenlockSource::Internal);
        assert_eq!(report.total_samples, 1);
    }

    #[test]
    fn test_jitter_estimate() {
        let cfg = GenlockConfig {
            filter_alpha: 0.5,
            ..GenlockConfig::default()
        };
        let mut tracker = GenlockTracker::new(cfg);
        // Alternating offsets should produce non-zero jitter
        for i in 0..20 {
            let offset = if i % 2 == 0 { 100 } else { -100 };
            tracker.feed(offset);
        }
        assert!(tracker.jitter_us() > 0.0);
    }
}
