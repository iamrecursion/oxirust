#![allow(dead_code)]
//! Phase-locked loop (PLL) for clock synchronization.
//!
//! Implements a digital PLL that tracks a reference clock signal and generates
//! a stable, phase-aligned output. Suitable for genlock and media clock recovery.

/// Operating state of the PLL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PllState {
    /// PLL is not locked; acquiring phase.
    Acquiring,
    /// PLL is locked to the reference.
    Locked,
    /// PLL has lost lock and is re-acquiring.
    Relocking,
    /// PLL is in holdover mode (reference lost).
    Holdover,
}

/// Configuration for the phase-locked loop.
#[derive(Debug, Clone, Copy)]
pub struct PllConfig {
    /// Proportional gain for the loop filter.
    pub proportional_gain: f64,
    /// Integral gain for the loop filter.
    pub integral_gain: f64,
    /// Phase error threshold for lock detection (nanoseconds).
    pub lock_threshold_ns: f64,
    /// Number of consecutive in-threshold samples needed to declare lock.
    pub lock_count: u32,
    /// Phase error threshold to declare loss of lock (nanoseconds).
    pub unlock_threshold_ns: f64,
    /// Maximum frequency adjustment in parts per billion.
    pub max_freq_adjust_ppb: f64,
}

impl Default for PllConfig {
    fn default() -> Self {
        Self {
            proportional_gain: 0.1,
            integral_gain: 0.01,
            lock_threshold_ns: 100.0,
            lock_count: 10,
            unlock_threshold_ns: 1000.0,
            max_freq_adjust_ppb: 100_000.0, // 100 ppm
        }
    }
}

impl PllConfig {
    /// Creates a new configuration with the given gains.
    #[must_use]
    pub fn new(proportional_gain: f64, integral_gain: f64) -> Self {
        Self {
            proportional_gain,
            integral_gain,
            ..Default::default()
        }
    }

    /// Sets the lock threshold in nanoseconds.
    #[must_use]
    pub fn with_lock_threshold_ns(mut self, ns: f64) -> Self {
        self.lock_threshold_ns = ns;
        self
    }

    /// Sets the unlock threshold in nanoseconds.
    #[must_use]
    pub fn with_unlock_threshold_ns(mut self, ns: f64) -> Self {
        self.unlock_threshold_ns = ns;
        self
    }

    /// Sets the maximum frequency adjustment in ppb.
    #[must_use]
    pub fn with_max_freq_ppb(mut self, ppb: f64) -> Self {
        self.max_freq_adjust_ppb = ppb;
        self
    }
}

/// Output of a PLL update step.
#[derive(Debug, Clone, Copy)]
pub struct PllOutput {
    /// Frequency adjustment in parts per billion.
    pub freq_adjust_ppb: f64,
    /// Phase error in nanoseconds.
    pub phase_error_ns: f64,
    /// Current PLL state.
    pub state: PllState,
    /// Whether a lock transition occurred this step.
    pub lock_changed: bool,
}

/// Statistics about PLL operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct PllStats {
    /// Total number of updates.
    pub total_updates: u64,
    /// Number of lock acquisitions.
    pub lock_acquisitions: u64,
    /// Number of lock losses.
    pub lock_losses: u64,
    /// Peak phase error observed in nanoseconds.
    pub peak_phase_error_ns: f64,
    /// Mean phase error in nanoseconds (exponential average).
    pub mean_phase_error_ns: f64,
    /// Current frequency offset in ppb.
    pub current_freq_ppb: f64,
}

/// Digital phase-locked loop for clock synchronization.
#[derive(Debug)]
pub struct PhaseLockLoop {
    /// Configuration.
    config: PllConfig,
    /// Current state.
    state: PllState,
    /// Integral accumulator for the PI controller.
    integral: f64,
    /// Current frequency adjustment in ppb.
    freq_adjust_ppb: f64,
    /// Count of consecutive in-lock samples.
    consecutive_locked: u32,
    /// Previous phase error.
    prev_phase_error_ns: f64,
    /// Statistics.
    stats: PllStats,
}

impl PhaseLockLoop {
    /// Creates a new PLL with the given configuration.
    #[must_use]
    pub fn new(config: PllConfig) -> Self {
        Self {
            config,
            state: PllState::Acquiring,
            integral: 0.0,
            freq_adjust_ppb: 0.0,
            consecutive_locked: 0,
            prev_phase_error_ns: 0.0,
            stats: PllStats::default(),
        }
    }

    /// Creates a PLL with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(PllConfig::default())
    }

    /// Updates the PLL with a new phase error measurement.
    ///
    /// Returns the computed frequency adjustment and current state.
    pub fn update(&mut self, phase_error_ns: f64) -> PllOutput {
        self.stats.total_updates += 1;

        // Track peak error
        let abs_error = phase_error_ns.abs();
        if abs_error > self.stats.peak_phase_error_ns {
            self.stats.peak_phase_error_ns = abs_error;
        }

        // Exponential moving average of error
        self.stats.mean_phase_error_ns = self.stats.mean_phase_error_ns * 0.95 + abs_error * 0.05;

        let old_state = self.state;

        // PI loop filter
        let p_term = self.config.proportional_gain * phase_error_ns;
        self.integral += self.config.integral_gain * phase_error_ns;

        // Clamp integral to prevent windup
        let max_integral = self.config.max_freq_adjust_ppb / self.config.integral_gain.max(1e-12);
        self.integral = self.integral.clamp(-max_integral, max_integral);

        self.freq_adjust_ppb = (p_term + self.integral).clamp(
            -self.config.max_freq_adjust_ppb,
            self.config.max_freq_adjust_ppb,
        );

        self.stats.current_freq_ppb = self.freq_adjust_ppb;

        // Lock state machine
        self.update_lock_state(abs_error);

        let lock_changed = self.state != old_state;
        if lock_changed {
            match self.state {
                PllState::Locked => self.stats.lock_acquisitions += 1,
                PllState::Relocking => self.stats.lock_losses += 1,
                _ => {}
            }
        }

        self.prev_phase_error_ns = phase_error_ns;

        PllOutput {
            freq_adjust_ppb: self.freq_adjust_ppb,
            phase_error_ns,
            state: self.state,
            lock_changed,
        }
    }

    /// Updates the lock state machine.
    fn update_lock_state(&mut self, abs_error: f64) {
        match self.state {
            PllState::Acquiring | PllState::Relocking => {
                if abs_error <= self.config.lock_threshold_ns {
                    self.consecutive_locked += 1;
                    if self.consecutive_locked >= self.config.lock_count {
                        self.state = PllState::Locked;
                    }
                } else {
                    self.consecutive_locked = 0;
                }
            }
            PllState::Locked => {
                if abs_error > self.config.unlock_threshold_ns {
                    self.state = PllState::Relocking;
                    self.consecutive_locked = 0;
                }
            }
            PllState::Holdover => {
                // Stay in holdover until external action
            }
        }
    }

    /// Enters holdover mode (reference lost).
    pub fn enter_holdover(&mut self) {
        self.state = PllState::Holdover;
        self.consecutive_locked = 0;
    }

    /// Exits holdover mode and begins re-acquiring.
    pub fn exit_holdover(&mut self) {
        if self.state == PllState::Holdover {
            self.state = PllState::Relocking;
        }
    }

    /// Returns the current PLL state.
    #[must_use]
    pub fn state(&self) -> PllState {
        self.state
    }

    /// Returns whether the PLL is currently locked.
    #[must_use]
    pub fn is_locked(&self) -> bool {
        self.state == PllState::Locked
    }

    /// Returns the current frequency adjustment in ppb.
    #[must_use]
    pub fn freq_adjust_ppb(&self) -> f64 {
        self.freq_adjust_ppb
    }

    /// Returns the current statistics.
    #[must_use]
    pub fn stats(&self) -> &PllStats {
        &self.stats
    }

    /// Resets the PLL to initial state.
    pub fn reset(&mut self) {
        self.state = PllState::Acquiring;
        self.integral = 0.0;
        self.freq_adjust_ppb = 0.0;
        self.consecutive_locked = 0;
        self.prev_phase_error_ns = 0.0;
        self.stats = PllStats::default();
    }

    /// Returns the current integral accumulator value.
    #[must_use]
    pub fn integral_value(&self) -> f64 {
        self.integral
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &PllConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pll_initial_state() {
        let pll = PhaseLockLoop::with_defaults();
        assert_eq!(pll.state(), PllState::Acquiring);
        assert!(!pll.is_locked());
        assert!((pll.freq_adjust_ppb() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pll_config_default() {
        let cfg = PllConfig::default();
        assert!((cfg.proportional_gain - 0.1).abs() < f64::EPSILON);
        assert!((cfg.integral_gain - 0.01).abs() < f64::EPSILON);
        assert!((cfg.lock_threshold_ns - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pll_config_builder() {
        let cfg = PllConfig::new(0.2, 0.02)
            .with_lock_threshold_ns(50.0)
            .with_unlock_threshold_ns(500.0)
            .with_max_freq_ppb(50_000.0);
        assert!((cfg.proportional_gain - 0.2).abs() < f64::EPSILON);
        assert!((cfg.lock_threshold_ns - 50.0).abs() < f64::EPSILON);
        assert!((cfg.unlock_threshold_ns - 500.0).abs() < f64::EPSILON);
        assert!((cfg.max_freq_adjust_ppb - 50_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pll_update_produces_output() {
        let mut pll = PhaseLockLoop::with_defaults();
        let output = pll.update(50.0);
        assert!((output.phase_error_ns - 50.0).abs() < f64::EPSILON);
        assert_eq!(output.state, PllState::Acquiring);
    }

    #[test]
    fn test_pll_lock_acquisition() {
        let config = PllConfig {
            lock_count: 3,
            lock_threshold_ns: 100.0,
            ..PllConfig::default()
        };
        let mut pll = PhaseLockLoop::new(config);

        // Feed small errors to reach lock
        for _ in 0..3 {
            let out = pll.update(10.0);
            if out.state == PllState::Locked {
                break;
            }
        }
        assert!(pll.is_locked());
        assert_eq!(pll.stats().lock_acquisitions, 1);
    }

    #[test]
    fn test_pll_lock_loss() {
        let config = PllConfig {
            lock_count: 2,
            lock_threshold_ns: 100.0,
            unlock_threshold_ns: 500.0,
            ..PllConfig::default()
        };
        let mut pll = PhaseLockLoop::new(config);

        // Acquire lock
        pll.update(10.0);
        pll.update(10.0);
        assert!(pll.is_locked());

        // Lose lock with big error
        let out = pll.update(1000.0);
        assert_eq!(out.state, PllState::Relocking);
        assert!(out.lock_changed);
        assert_eq!(pll.stats().lock_losses, 1);
    }

    #[test]
    fn test_pll_frequency_adjustment() {
        let mut pll = PhaseLockLoop::with_defaults();
        // Positive phase error -> positive frequency adjustment
        let out = pll.update(1000.0);
        assert!(out.freq_adjust_ppb > 0.0);

        // Negative phase error
        let mut pll2 = PhaseLockLoop::with_defaults();
        let out2 = pll2.update(-1000.0);
        assert!(out2.freq_adjust_ppb < 0.0);
    }

    #[test]
    fn test_pll_freq_clamping() {
        let config = PllConfig {
            proportional_gain: 10.0,
            max_freq_adjust_ppb: 1000.0,
            ..PllConfig::default()
        };
        let mut pll = PhaseLockLoop::new(config);
        let out = pll.update(1_000_000.0);
        assert!(out.freq_adjust_ppb <= 1000.0);
        assert!(out.freq_adjust_ppb >= -1000.0);
    }

    #[test]
    fn test_pll_holdover() {
        let mut pll = PhaseLockLoop::with_defaults();
        pll.enter_holdover();
        assert_eq!(pll.state(), PllState::Holdover);

        // Update in holdover should stay in holdover
        pll.update(10.0);
        assert_eq!(pll.state(), PllState::Holdover);

        pll.exit_holdover();
        assert_eq!(pll.state(), PllState::Relocking);
    }

    #[test]
    fn test_pll_reset() {
        let mut pll = PhaseLockLoop::with_defaults();
        pll.update(100.0);
        pll.update(200.0);
        assert!(pll.stats().total_updates > 0);

        pll.reset();
        assert_eq!(pll.state(), PllState::Acquiring);
        assert_eq!(pll.stats().total_updates, 0);
        assert!((pll.integral_value() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pll_convergence() {
        let mut pll = PhaseLockLoop::new(PllConfig {
            proportional_gain: 0.3,
            integral_gain: 0.05,
            lock_count: 5,
            lock_threshold_ns: 50.0,
            ..PllConfig::default()
        });

        // Simulate decreasing phase error (convergence)
        let mut error = 500.0;
        for _ in 0..100 {
            let out = pll.update(error);
            // The PLL drives the frequency to correct the error
            error -= out.freq_adjust_ppb * 0.001;
            error *= 0.95; // natural damping
        }
        // After convergence, error should be small
        assert!(error.abs() < 50.0);
    }

    #[test]
    fn test_pll_stats_tracking() {
        let mut pll = PhaseLockLoop::with_defaults();
        pll.update(100.0);
        pll.update(200.0);
        pll.update(50.0);

        assert_eq!(pll.stats().total_updates, 3);
        assert!((pll.stats().peak_phase_error_ns - 200.0).abs() < f64::EPSILON);
        assert!(pll.stats().mean_phase_error_ns > 0.0);
    }
}
