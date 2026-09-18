#![allow(dead_code)]
//! Clock frequency and phase steering algorithms.
//!
//! This module implements algorithms for smoothly adjusting a local clock's
//! frequency and phase to converge on a reference time source. It supports
//! both hard stepping (for large offsets) and soft slewing (for fine tuning).

/// Mode of clock adjustment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SteeringMode {
    /// Hard step: immediately adjust the clock by the full offset.
    Step,
    /// Soft slew: gradually adjust the clock frequency to converge.
    Slew,
    /// Disabled: no adjustments are made.
    Disabled,
}

impl SteeringMode {
    /// Returns a human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Step => "Step",
            Self::Slew => "Slew",
            Self::Disabled => "Disabled",
        }
    }
}

/// Result of a steering computation.
#[derive(Debug, Clone, Copy)]
pub struct SteeringAction {
    /// Frequency adjustment in parts per billion (ppb).
    pub freq_adj_ppb: f64,
    /// Phase step in nanoseconds (0 if slewing).
    pub phase_step_ns: f64,
    /// The steering mode that was used.
    pub mode: SteeringMode,
}

impl SteeringAction {
    /// Returns `true` if no adjustment is needed.
    #[must_use]
    pub fn is_noop(&self) -> bool {
        self.freq_adj_ppb.abs() < f64::EPSILON && self.phase_step_ns.abs() < f64::EPSILON
    }
}

/// Configuration for the clock steering algorithm.
#[derive(Debug, Clone, Copy)]
pub struct SteeringConfig {
    /// Threshold in nanoseconds above which a hard step is used.
    pub step_threshold_ns: f64,
    /// Maximum slew rate in ppb.
    pub max_slew_ppb: f64,
    /// Proportional gain for the PI controller.
    pub kp: f64,
    /// Integral gain for the PI controller.
    pub ki: f64,
    /// Maximum integral accumulator value (anti-windup).
    pub max_integral: f64,
}

impl Default for SteeringConfig {
    fn default() -> Self {
        Self {
            step_threshold_ns: 1_000_000.0, // 1 ms
            max_slew_ppb: 500.0,
            kp: 0.7,
            ki: 0.05,
            max_integral: 200.0,
        }
    }
}

/// PI (proportional-integral) controller for clock steering.
#[derive(Debug, Clone)]
pub struct ClockSteering {
    /// Configuration parameters.
    config: SteeringConfig,
    /// Integral accumulator.
    integral: f64,
    /// Previous offset for tracking convergence.
    prev_offset_ns: f64,
    /// Number of updates performed.
    update_count: u64,
    /// Total absolute offset accumulated (for statistics).
    total_abs_offset: f64,
    /// Current steering mode.
    mode: SteeringMode,
    /// Whether the controller is locked (converged).
    locked: bool,
    /// Lock threshold in nanoseconds.
    lock_threshold_ns: f64,
    /// Number of consecutive samples within lock threshold.
    lock_count: u32,
    /// Samples needed to declare lock.
    lock_required: u32,
}

impl ClockSteering {
    /// Creates a new clock steering controller with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(SteeringConfig::default())
    }

    /// Creates a new clock steering controller with the given configuration.
    #[must_use]
    pub fn with_config(config: SteeringConfig) -> Self {
        Self {
            config,
            integral: 0.0,
            prev_offset_ns: 0.0,
            update_count: 0,
            total_abs_offset: 0.0,
            mode: SteeringMode::Slew,
            locked: false,
            lock_threshold_ns: 1000.0, // 1 us
            lock_count: 0,
            lock_required: 10,
        }
    }

    /// Sets the lock threshold in nanoseconds.
    #[must_use]
    pub fn with_lock_threshold(mut self, threshold_ns: f64) -> Self {
        self.lock_threshold_ns = threshold_ns;
        self
    }

    /// Sets the number of consecutive good samples required for lock.
    #[must_use]
    pub const fn with_lock_required(mut self, count: u32) -> Self {
        self.lock_required = count;
        self
    }

    /// Computes the steering action for the given measured offset.
    ///
    /// `offset_ns` is the measured offset from the reference in nanoseconds.
    /// Positive means local clock is ahead; negative means behind.
    pub fn update(&mut self, offset_ns: f64) -> SteeringAction {
        self.update_count += 1;
        self.total_abs_offset += offset_ns.abs();

        // Check if offset is too large for slewing
        if offset_ns.abs() > self.config.step_threshold_ns {
            self.integral = 0.0;
            self.prev_offset_ns = 0.0;
            self.locked = false;
            self.lock_count = 0;
            self.mode = SteeringMode::Step;
            return SteeringAction {
                freq_adj_ppb: 0.0,
                phase_step_ns: -offset_ns,
                mode: SteeringMode::Step,
            };
        }

        // PI controller
        self.integral += offset_ns * self.config.ki;

        // Anti-windup
        if self.integral > self.config.max_integral {
            self.integral = self.config.max_integral;
        } else if self.integral < -self.config.max_integral {
            self.integral = -self.config.max_integral;
        }

        let proportional = offset_ns * self.config.kp;
        let mut freq_adj = -(proportional + self.integral);

        // Clamp to max slew rate
        if freq_adj > self.config.max_slew_ppb {
            freq_adj = self.config.max_slew_ppb;
        } else if freq_adj < -self.config.max_slew_ppb {
            freq_adj = -self.config.max_slew_ppb;
        }

        self.prev_offset_ns = offset_ns;
        self.mode = SteeringMode::Slew;

        // Update lock state
        if offset_ns.abs() < self.lock_threshold_ns {
            self.lock_count += 1;
            if self.lock_count >= self.lock_required {
                self.locked = true;
            }
        } else {
            self.lock_count = 0;
            self.locked = false;
        }

        SteeringAction {
            freq_adj_ppb: freq_adj,
            phase_step_ns: 0.0,
            mode: SteeringMode::Slew,
        }
    }

    /// Returns `true` if the controller is in locked (converged) state.
    #[must_use]
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Returns the current steering mode.
    #[must_use]
    pub fn current_mode(&self) -> SteeringMode {
        self.mode
    }

    /// Returns the number of updates performed.
    #[must_use]
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Returns the mean absolute offset over all updates.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_absolute_offset(&self) -> f64 {
        if self.update_count == 0 {
            return 0.0;
        }
        self.total_abs_offset / self.update_count as f64
    }

    /// Returns the current integral accumulator value.
    #[must_use]
    pub fn integral_value(&self) -> f64 {
        self.integral
    }

    /// Resets the controller state.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_offset_ns = 0.0;
        self.update_count = 0;
        self.total_abs_offset = 0.0;
        self.locked = false;
        self.lock_count = 0;
        self.mode = SteeringMode::Slew;
    }

    /// Disables the controller; subsequent updates return no-op actions.
    pub fn disable(&mut self) {
        self.mode = SteeringMode::Disabled;
    }
}

impl Default for ClockSteering {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_steering_mode_label() {
        assert_eq!(SteeringMode::Step.label(), "Step");
        assert_eq!(SteeringMode::Slew.label(), "Slew");
        assert_eq!(SteeringMode::Disabled.label(), "Disabled");
    }

    #[test]
    fn test_default_config() {
        let cfg = SteeringConfig::default();
        assert!((cfg.step_threshold_ns - 1_000_000.0).abs() < f64::EPSILON);
        assert!((cfg.max_slew_ppb - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_new_controller() {
        let ctrl = ClockSteering::new();
        assert_eq!(ctrl.update_count(), 0);
        assert!(!ctrl.is_locked());
        assert!((ctrl.mean_absolute_offset() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_small_offset_slews() {
        let mut ctrl = ClockSteering::new();
        let action = ctrl.update(500.0); // 500 ns offset
        assert_eq!(action.mode, SteeringMode::Slew);
        assert!((action.phase_step_ns - 0.0).abs() < f64::EPSILON);
        // Frequency should be adjusted (negative because local is ahead)
        assert!(action.freq_adj_ppb < 0.0);
    }

    #[test]
    fn test_large_offset_steps() {
        let mut ctrl = ClockSteering::new();
        let action = ctrl.update(2_000_000.0); // 2 ms, above threshold
        assert_eq!(action.mode, SteeringMode::Step);
        assert!((action.phase_step_ns - (-2_000_000.0)).abs() < f64::EPSILON);
        assert!((action.freq_adj_ppb - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_negative_offset_steps() {
        let mut ctrl = ClockSteering::new();
        let action = ctrl.update(-5_000_000.0);
        assert_eq!(action.mode, SteeringMode::Step);
        assert!((action.phase_step_ns - 5_000_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_convergence_to_lock() {
        let mut ctrl = ClockSteering::new()
            .with_lock_required(3)
            .with_lock_threshold(2000.0);
        for _ in 0..3 {
            ctrl.update(100.0); // well within lock threshold
        }
        assert!(ctrl.is_locked());
    }

    #[test]
    fn test_lock_lost_on_large_offset() {
        let mut ctrl = ClockSteering::new()
            .with_lock_required(3)
            .with_lock_threshold(2000.0);
        for _ in 0..3 {
            ctrl.update(100.0);
        }
        assert!(ctrl.is_locked());
        ctrl.update(5000.0); // exceeds lock threshold
        assert!(!ctrl.is_locked());
    }

    #[test]
    fn test_anti_windup() {
        let mut ctrl = ClockSteering::with_config(SteeringConfig {
            max_integral: 50.0,
            ki: 1.0,
            ..SteeringConfig::default()
        });
        for _ in 0..100 {
            ctrl.update(1000.0);
        }
        assert!(ctrl.integral_value() <= 50.0);
    }

    #[test]
    fn test_slew_rate_clamped() {
        let mut ctrl = ClockSteering::with_config(SteeringConfig {
            max_slew_ppb: 100.0,
            kp: 10.0,
            ..SteeringConfig::default()
        });
        let action = ctrl.update(500_000.0);
        assert!(action.freq_adj_ppb.abs() <= 100.0 + f64::EPSILON);
    }

    #[test]
    fn test_mean_absolute_offset() {
        let mut ctrl = ClockSteering::new();
        ctrl.update(100.0);
        ctrl.update(200.0);
        ctrl.update(300.0);
        let mean = ctrl.mean_absolute_offset();
        assert!((mean - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reset() {
        let mut ctrl = ClockSteering::new();
        ctrl.update(500.0);
        ctrl.update(600.0);
        ctrl.reset();
        assert_eq!(ctrl.update_count(), 0);
        assert!(!ctrl.is_locked());
        assert!((ctrl.integral_value() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_noop_action() {
        let action = SteeringAction {
            freq_adj_ppb: 0.0,
            phase_step_ns: 0.0,
            mode: SteeringMode::Slew,
        };
        assert!(action.is_noop());

        let action2 = SteeringAction {
            freq_adj_ppb: 1.0,
            phase_step_ns: 0.0,
            mode: SteeringMode::Slew,
        };
        assert!(!action2.is_noop());
    }
}
