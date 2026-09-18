#![allow(dead_code)]

//! Clock recovery for media streams.
//!
//! Recovers a stable clock reference from incoming media packet timestamps
//! using PLL (phase-locked loop) techniques and adaptive filtering.

use std::collections::VecDeque;
use std::fmt;

/// State of the clock recovery PLL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PllState {
    /// Acquiring lock — initial phase.
    Acquiring,
    /// Locked — tracking the source clock.
    Locked,
    /// Holdover — maintaining last known frequency.
    Holdover,
    /// Free-running — no reference available.
    FreeRun,
}

impl fmt::Display for PllState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Acquiring => write!(f, "Acquiring"),
            Self::Locked => write!(f, "Locked"),
            Self::Holdover => write!(f, "Holdover"),
            Self::FreeRun => write!(f, "FreeRun"),
        }
    }
}

/// Configuration for the clock recovery PLL.
#[derive(Debug, Clone)]
pub struct PllConfig {
    /// Proportional gain (controls response speed).
    pub kp: f64,
    /// Integral gain (controls steady-state error).
    pub ki: f64,
    /// Lock threshold in nanoseconds — if phase error stays below this, PLL is locked.
    pub lock_threshold_ns: f64,
    /// Number of consecutive good samples needed to declare lock.
    pub lock_count: usize,
    /// Number of consecutive missing samples before entering holdover.
    pub holdover_count: usize,
    /// Maximum phase error history.
    pub history_size: usize,
    /// Nominal rate in ticks per second (e.g. 90000 for MPEG-TS).
    pub nominal_rate: f64,
}

impl Default for PllConfig {
    fn default() -> Self {
        Self {
            kp: 0.01,
            ki: 0.001,
            lock_threshold_ns: 1_000.0, // 1 us
            lock_count: 20,
            holdover_count: 10,
            history_size: 256,
            nominal_rate: 90_000.0,
        }
    }
}

/// Clock recovery PLL.
#[derive(Debug, Clone)]
pub struct ClockRecovery {
    /// Configuration.
    config: PllConfig,
    /// Current PLL state.
    state: PllState,
    /// Current phase error in nanoseconds.
    phase_error_ns: f64,
    /// Integral accumulator.
    integral: f64,
    /// Frequency offset estimate (ppb).
    freq_offset_ppb: f64,
    /// Count of consecutive good samples.
    good_count: usize,
    /// Count of consecutive missing samples.
    missing_count: usize,
    /// Phase error history.
    history: VecDeque<f64>,
    /// Total samples processed.
    total_samples: u64,
    /// Last reference timestamp (ticks).
    last_ref_ts: Option<u64>,
    /// Last local timestamp (nanoseconds).
    last_local_ns: Option<u64>,
}

impl ClockRecovery {
    /// Create a new clock recovery instance with default config.
    pub fn new() -> Self {
        Self::with_config(PllConfig::default())
    }

    /// Create a clock recovery instance with a custom config.
    pub fn with_config(config: PllConfig) -> Self {
        Self {
            history: VecDeque::with_capacity(config.history_size),
            config,
            state: PllState::FreeRun,
            phase_error_ns: 0.0,
            integral: 0.0,
            freq_offset_ppb: 0.0,
            good_count: 0,
            missing_count: 0,
            total_samples: 0,
            last_ref_ts: None,
            last_local_ns: None,
        }
    }

    /// Current PLL state.
    pub fn state(&self) -> PllState {
        self.state
    }

    /// Current phase error in nanoseconds.
    pub fn phase_error_ns(&self) -> f64 {
        self.phase_error_ns
    }

    /// Current frequency offset estimate in parts-per-billion.
    pub fn freq_offset_ppb(&self) -> f64 {
        self.freq_offset_ppb
    }

    /// Total samples processed.
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Feed a reference timestamp (in source clock ticks) and the corresponding
    /// local time (nanoseconds). Returns the corrected local time.
    #[allow(clippy::cast_precision_loss)]
    pub fn feed(&mut self, ref_ts: u64, local_ns: u64) -> u64 {
        self.total_samples += 1;
        self.missing_count = 0;

        if self.state == PllState::FreeRun {
            self.state = PllState::Acquiring;
        }

        let corrected = if let (Some(prev_ref), Some(prev_local)) =
            (self.last_ref_ts, self.last_local_ns)
        {
            let ref_delta = ref_ts.wrapping_sub(prev_ref) as f64;
            let expected_delta_ns = ref_delta / self.config.nominal_rate * 1_000_000_000.0;
            let actual_delta_ns = local_ns.wrapping_sub(prev_local) as f64;
            self.phase_error_ns = actual_delta_ns - expected_delta_ns;

            // PI control
            self.integral += self.phase_error_ns * self.config.ki;
            let correction = self.phase_error_ns * self.config.kp + self.integral;

            // Update frequency offset estimate
            if expected_delta_ns.abs() > f64::EPSILON {
                self.freq_offset_ppb = (self.phase_error_ns / expected_delta_ns) * 1_000_000_000.0;
            }

            // Record history
            if self.history.len() >= self.config.history_size {
                self.history.pop_front();
            }
            self.history.push_back(self.phase_error_ns);

            // Update lock state
            if self.phase_error_ns.abs() < self.config.lock_threshold_ns {
                self.good_count += 1;
                if self.good_count >= self.config.lock_count && self.state != PllState::Locked {
                    self.state = PllState::Locked;
                }
            } else {
                self.good_count = 0;
                if self.state == PllState::Locked {
                    self.state = PllState::Acquiring;
                }
            }

            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let adj = correction as i64;
            if adj >= 0 {
                local_ns.wrapping_sub(adj as u64)
            } else {
                local_ns.wrapping_add(adj.unsigned_abs())
            }
        } else {
            local_ns
        };

        self.last_ref_ts = Some(ref_ts);
        self.last_local_ns = Some(local_ns);
        corrected
    }

    /// Signal a missing reference sample (source dropout).
    pub fn signal_missing(&mut self) {
        self.missing_count += 1;
        self.good_count = 0;
        if self.missing_count >= self.config.holdover_count {
            match self.state {
                PllState::Locked | PllState::Acquiring => self.state = PllState::Holdover,
                _ => {}
            }
        }
    }

    /// Mean phase error over the history window.
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_phase_error_ns(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.history.iter().sum();
        sum / self.history.len() as f64
    }

    /// RMS phase error over the history window.
    #[allow(clippy::cast_precision_loss)]
    pub fn rms_phase_error_ns(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = self.history.iter().map(|e| e * e).sum();
        (sum_sq / self.history.len() as f64).sqrt()
    }

    /// Peak phase error in the history window.
    pub fn peak_phase_error_ns(&self) -> f64 {
        self.history.iter().map(|e| e.abs()).fold(0.0_f64, f64::max)
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.state = PllState::FreeRun;
        self.phase_error_ns = 0.0;
        self.integral = 0.0;
        self.freq_offset_ppb = 0.0;
        self.good_count = 0;
        self.missing_count = 0;
        self.history.clear();
        self.total_samples = 0;
        self.last_ref_ts = None;
        self.last_local_ns = None;
    }

    /// Whether the PLL is currently locked.
    pub fn is_locked(&self) -> bool {
        self.state == PllState::Locked
    }

    /// Number of history entries.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

impl Default for ClockRecovery {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_state() {
        let cr = ClockRecovery::new();
        assert_eq!(cr.state(), PllState::FreeRun);
        assert_eq!(cr.total_samples(), 0);
    }

    #[test]
    fn test_first_feed_acquires() {
        let mut cr = ClockRecovery::new();
        cr.feed(0, 0);
        assert_eq!(cr.state(), PllState::Acquiring);
        assert_eq!(cr.total_samples(), 1);
    }

    #[test]
    fn test_feed_updates_samples() {
        let mut cr = ClockRecovery::new();
        cr.feed(0, 0);
        cr.feed(90_000, 1_000_000_000);
        assert_eq!(cr.total_samples(), 2);
    }

    #[test]
    fn test_perfect_clock_locks() {
        let mut cr = ClockRecovery::with_config(PllConfig {
            lock_count: 5,
            lock_threshold_ns: 10.0,
            nominal_rate: 90_000.0,
            ..PllConfig::default()
        });
        // Feed perfect samples: exactly 1 second apart at 90kHz
        for i in 0..30 {
            let ref_ts = i * 90_000;
            let local_ns = i * 1_000_000_000;
            cr.feed(ref_ts, local_ns);
        }
        assert_eq!(cr.state(), PllState::Locked);
    }

    #[test]
    fn test_signal_missing_holdover() {
        let mut cr = ClockRecovery::with_config(PllConfig {
            holdover_count: 3,
            ..PllConfig::default()
        });
        cr.feed(0, 0);
        cr.feed(90_000, 1_000_000_000);
        // Simulate acquiring state
        assert_eq!(cr.state(), PllState::Acquiring);
        for _ in 0..5 {
            cr.signal_missing();
        }
        assert_eq!(cr.state(), PllState::Holdover);
    }

    #[test]
    fn test_mean_phase_error_empty() {
        let cr = ClockRecovery::new();
        assert!((cr.mean_phase_error_ns() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rms_phase_error_empty() {
        let cr = ClockRecovery::new();
        assert!((cr.rms_phase_error_ns() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_peak_phase_error_empty() {
        let cr = ClockRecovery::new();
        assert!((cr.peak_phase_error_ns() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reset() {
        let mut cr = ClockRecovery::new();
        cr.feed(0, 0);
        cr.feed(90_000, 1_000_000_000);
        cr.reset();
        assert_eq!(cr.state(), PllState::FreeRun);
        assert_eq!(cr.total_samples(), 0);
        assert_eq!(cr.history_len(), 0);
    }

    #[test]
    fn test_is_locked() {
        let cr = ClockRecovery::new();
        assert!(!cr.is_locked());
    }

    #[test]
    fn test_pll_state_display() {
        assert_eq!(PllState::Acquiring.to_string(), "Acquiring");
        assert_eq!(PllState::Locked.to_string(), "Locked");
        assert_eq!(PllState::Holdover.to_string(), "Holdover");
        assert_eq!(PllState::FreeRun.to_string(), "FreeRun");
    }

    #[test]
    fn test_default_config() {
        let cfg = PllConfig::default();
        assert!((cfg.nominal_rate - 90_000.0).abs() < f64::EPSILON);
        assert_eq!(cfg.lock_count, 20);
    }

    #[test]
    fn test_freq_offset_initial() {
        let cr = ClockRecovery::new();
        assert!((cr.freq_offset_ppb() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_history_grows() {
        let mut cr = ClockRecovery::with_config(PllConfig {
            history_size: 100,
            ..PllConfig::default()
        });
        cr.feed(0, 0);
        for i in 1..=10_u64 {
            cr.feed(i * 90_000, i * 1_000_000_000);
        }
        assert_eq!(cr.history_len(), 10);
    }

    #[test]
    fn test_history_bounded() {
        let mut cr = ClockRecovery::with_config(PllConfig {
            history_size: 5,
            ..PllConfig::default()
        });
        cr.feed(0, 0);
        for i in 1..=20_u64 {
            cr.feed(i * 90_000, i * 1_000_000_000);
        }
        assert!(cr.history_len() <= 5);
    }

    #[test]
    fn test_missing_resets_good_count() {
        let mut cr = ClockRecovery::new();
        cr.feed(0, 0);
        cr.feed(90_000, 1_000_000_000);
        cr.signal_missing();
        // good_count should be 0, so even if next samples are good,
        // it has to re-accumulate
        assert!(!cr.is_locked());
    }
}
