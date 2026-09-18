//! Learning rate schedulers for TensorLogic training.
//!
//! Provides classic and adaptive scheduling strategies:
//! - Step decay
//! - Cosine annealing (with optional warm restarts)
//! - Linear warmup
//! - Cyclical learning rates
//! - One-cycle policy

use thiserror::Error;

/// Error types for scheduler operations.
#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error("Invalid config: {0}")]
    InvalidConfig(String),
    #[error("Scheduler exhausted after {0} steps")]
    Exhausted(usize),
}

/// Trait for learning rate schedulers.
pub trait LrSchedulerV2: Send {
    /// Advance one step and return the new learning rate.
    fn step(&mut self) -> f64;
    /// Return the current learning rate without advancing.
    fn current_lr(&self) -> f64;
    /// Reset the scheduler to its initial state.
    fn reset(&mut self);
    /// Total number of steps taken.
    fn steps_taken(&self) -> usize;
    /// Whether the scheduler has completed a cycle (if applicable).
    fn completed_cycle(&self) -> bool {
        false
    }
}

// ------- StepDecayScheduler -------

/// Multiplies the learning rate by `gamma` every `step_size` steps.
///
/// lr_t = base_lr * gamma^floor(t / step_size)
pub struct StepDecayScheduler {
    base_lr: f64,
    gamma: f64,
    step_size: usize,
    current_step: usize,
}

impl StepDecayScheduler {
    /// Create a new step decay scheduler.
    ///
    /// # Errors
    /// Returns [`SchedulerError::InvalidConfig`] if any parameter is invalid.
    pub fn new(base_lr: f64, gamma: f64, step_size: usize) -> Result<Self, SchedulerError> {
        if base_lr <= 0.0 {
            return Err(SchedulerError::InvalidConfig(
                "base_lr must be positive".into(),
            ));
        }
        if !(0.0..=1.0).contains(&gamma) {
            return Err(SchedulerError::InvalidConfig(
                "gamma must be in (0, 1]".into(),
            ));
        }
        if step_size == 0 {
            return Err(SchedulerError::InvalidConfig(
                "step_size must be > 0".into(),
            ));
        }
        Ok(StepDecayScheduler {
            base_lr,
            gamma,
            step_size,
            current_step: 0,
        })
    }
}

impl LrSchedulerV2 for StepDecayScheduler {
    fn step(&mut self) -> f64 {
        self.current_step += 1;
        self.current_lr()
    }

    fn current_lr(&self) -> f64 {
        let exponent = self.current_step / self.step_size;
        self.base_lr * self.gamma.powi(exponent as i32)
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }

    fn steps_taken(&self) -> usize {
        self.current_step
    }
}

// ------- CosineAnnealingScheduler -------

/// Cosine annealing with optional warm restarts (SGDR).
///
/// lr_t = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * t_cur / t_max))
///
/// If `restart_period` is Some(T), restarts every T steps (warm restarts).
pub struct CosineAnnealingScheduler {
    max_lr: f64,
    min_lr: f64,
    t_max: usize,
    restart_period: Option<usize>,
    current_step: usize,
    cycle_count: usize,
}

impl CosineAnnealingScheduler {
    /// Create a new cosine annealing scheduler.
    ///
    /// # Errors
    /// Returns [`SchedulerError::InvalidConfig`] if any parameter is invalid.
    pub fn new(max_lr: f64, min_lr: f64, t_max: usize) -> Result<Self, SchedulerError> {
        if max_lr < min_lr {
            return Err(SchedulerError::InvalidConfig(
                "max_lr must be >= min_lr".into(),
            ));
        }
        if t_max == 0 {
            return Err(SchedulerError::InvalidConfig("t_max must be > 0".into()));
        }
        Ok(CosineAnnealingScheduler {
            max_lr,
            min_lr,
            t_max,
            restart_period: None,
            current_step: 0,
            cycle_count: 0,
        })
    }

    /// Enable warm restarts every `period` steps.
    pub fn with_warm_restarts(mut self, period: usize) -> Self {
        self.restart_period = Some(period);
        self
    }
}

impl LrSchedulerV2 for CosineAnnealingScheduler {
    fn step(&mut self) -> f64 {
        self.current_step += 1;
        if let Some(period) = self.restart_period {
            if period > 0 && self.current_step.is_multiple_of(period) {
                self.current_step = 0;
                self.cycle_count += 1;
            }
        }
        self.current_lr()
    }

    fn current_lr(&self) -> f64 {
        let t_cur = self.current_step.min(self.t_max) as f64;
        let t_max = self.t_max as f64;
        let cos_val = (std::f64::consts::PI * t_cur / t_max).cos();
        self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + cos_val)
    }

    fn reset(&mut self) {
        self.current_step = 0;
        self.cycle_count = 0;
    }

    fn steps_taken(&self) -> usize {
        self.current_step
    }

    fn completed_cycle(&self) -> bool {
        self.cycle_count > 0
    }
}

// ------- WarmupScheduler -------

/// Linear warmup followed by another scheduler.
///
/// During warmup: lr = warmup_start_lr + (warmup_end_lr - warmup_start_lr) * (step / warmup_steps)
/// After warmup: delegates to the inner scheduler.
pub struct WarmupScheduler {
    warmup_steps: usize,
    warmup_start_lr: f64,
    warmup_end_lr: f64,
    inner: Box<dyn LrSchedulerV2>,
    current_step: usize,
    inner_started: bool,
}

impl WarmupScheduler {
    /// Create a new warmup scheduler wrapping an inner scheduler.
    ///
    /// # Errors
    /// Returns [`SchedulerError::InvalidConfig`] if `warmup_steps` is zero.
    pub fn new(
        warmup_steps: usize,
        warmup_start_lr: f64,
        warmup_end_lr: f64,
        inner: Box<dyn LrSchedulerV2>,
    ) -> Result<Self, SchedulerError> {
        if warmup_steps == 0 {
            return Err(SchedulerError::InvalidConfig(
                "warmup_steps must be > 0".into(),
            ));
        }
        Ok(WarmupScheduler {
            warmup_steps,
            warmup_start_lr,
            warmup_end_lr,
            inner,
            current_step: 0,
            inner_started: false,
        })
    }
}

impl LrSchedulerV2 for WarmupScheduler {
    fn step(&mut self) -> f64 {
        self.current_step += 1;
        if self.current_step >= self.warmup_steps {
            self.inner_started = true;
            self.inner.step()
        } else {
            self.current_lr()
        }
    }

    fn current_lr(&self) -> f64 {
        if self.inner_started || self.current_step >= self.warmup_steps {
            self.inner.current_lr()
        } else {
            let frac = self.current_step as f64 / self.warmup_steps as f64;
            self.warmup_start_lr + frac * (self.warmup_end_lr - self.warmup_start_lr)
        }
    }

    fn reset(&mut self) {
        self.current_step = 0;
        self.inner_started = false;
        self.inner.reset();
    }

    fn steps_taken(&self) -> usize {
        self.current_step
    }
}

// ------- CyclicalScheduler -------

/// Cyclical learning rates (CLR) — oscillates between min_lr and max_lr.
///
/// Uses triangular policy: linear up then linear down, period = 2 * step_size.
pub struct CyclicalScheduler {
    min_lr: f64,
    max_lr: f64,
    step_size: usize,
    current_step: usize,
}

impl CyclicalScheduler {
    /// Create a new cyclical learning rate scheduler.
    ///
    /// # Errors
    /// Returns [`SchedulerError::InvalidConfig`] if any parameter is invalid.
    pub fn new(min_lr: f64, max_lr: f64, step_size: usize) -> Result<Self, SchedulerError> {
        if max_lr <= min_lr {
            return Err(SchedulerError::InvalidConfig(
                "max_lr must be > min_lr".into(),
            ));
        }
        if step_size == 0 {
            return Err(SchedulerError::InvalidConfig(
                "step_size must be > 0".into(),
            ));
        }
        Ok(CyclicalScheduler {
            min_lr,
            max_lr,
            step_size,
            current_step: 0,
        })
    }
}

impl LrSchedulerV2 for CyclicalScheduler {
    fn step(&mut self) -> f64 {
        self.current_step += 1;
        self.current_lr()
    }

    fn current_lr(&self) -> f64 {
        let cycle = self.current_step / (2 * self.step_size);
        let x = (self.current_step as f64 / self.step_size as f64) - 2.0 * cycle as f64 - 1.0;
        let frac = (1.0 - x.abs()).max(0.0);
        self.min_lr + (self.max_lr - self.min_lr) * frac
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }

    fn steps_taken(&self) -> usize {
        self.current_step
    }
}

// ------- OneCycleLrScheduler -------

/// One-cycle learning rate policy.
///
/// Phase 1 (pct_start of total_steps): linear ramp from base_lr to max_lr
/// Phase 2 (remaining): cosine decay from max_lr to min_lr
pub struct OneCycleLrScheduler {
    base_lr: f64,
    max_lr: f64,
    min_lr: f64,
    total_steps: usize,
    pct_start: f64,
    current_step: usize,
}

impl OneCycleLrScheduler {
    /// Create a new one-cycle learning rate scheduler.
    ///
    /// # Errors
    /// Returns [`SchedulerError::InvalidConfig`] if any parameter is invalid.
    pub fn new(
        base_lr: f64,
        max_lr: f64,
        min_lr: f64,
        total_steps: usize,
        pct_start: f64,
    ) -> Result<Self, SchedulerError> {
        if max_lr <= base_lr {
            return Err(SchedulerError::InvalidConfig(
                "max_lr must be > base_lr".into(),
            ));
        }
        if !(0.0..=1.0).contains(&pct_start) {
            return Err(SchedulerError::InvalidConfig(
                "pct_start must be in [0, 1]".into(),
            ));
        }
        if total_steps == 0 {
            return Err(SchedulerError::InvalidConfig(
                "total_steps must be > 0".into(),
            ));
        }
        Ok(OneCycleLrScheduler {
            base_lr,
            max_lr,
            min_lr,
            total_steps,
            pct_start,
            current_step: 0,
        })
    }
}

impl LrSchedulerV2 for OneCycleLrScheduler {
    fn step(&mut self) -> f64 {
        self.current_step = (self.current_step + 1).min(self.total_steps);
        self.current_lr()
    }

    fn current_lr(&self) -> f64 {
        let warmup_steps = (self.total_steps as f64 * self.pct_start) as usize;
        if self.current_step <= warmup_steps {
            let frac = if warmup_steps == 0 {
                1.0
            } else {
                self.current_step as f64 / warmup_steps as f64
            };
            self.base_lr + frac * (self.max_lr - self.base_lr)
        } else {
            let decay_steps = self.total_steps - warmup_steps;
            let t = self.current_step - warmup_steps;
            let frac = if decay_steps == 0 {
                1.0
            } else {
                t as f64 / decay_steps as f64
            };
            let cos_val = (std::f64::consts::PI * frac).cos();
            self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + cos_val)
        }
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }

    fn steps_taken(&self) -> usize {
        self.current_step
    }
}

// ------- SchedulerConfig -------

/// Builder for creating scheduler configurations.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// The type of scheduler to construct.
    pub scheduler_type: SchedulerType,
    /// Base learning rate.
    pub base_lr: f64,
    /// Optional maximum learning rate (used by cyclical and one-cycle schedulers).
    pub max_lr: Option<f64>,
    /// Optional minimum learning rate floor.
    pub min_lr: Option<f64>,
    /// Optional total number of training steps.
    pub total_steps: Option<usize>,
    /// Optional step size for decay / cyclical half-period.
    pub step_size: Option<usize>,
    /// Optional decay factor (used by step decay).
    pub gamma: Option<f64>,
    /// Optional number of warmup steps.
    pub warmup_steps: Option<usize>,
    /// Optional fraction of total steps used for warmup in one-cycle.
    pub pct_start: Option<f64>,
}

/// Enum identifying the scheduler algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerType {
    /// Step decay: multiply LR by gamma every step_size steps.
    StepDecay,
    /// Cosine annealing without restarts.
    CosineAnnealing,
    /// Cosine annealing with warm restarts (SGDR).
    CosineAnnealingWarmRestarts,
    /// Linear warmup followed by an inner scheduler.
    Warmup,
    /// Cyclical (triangular) learning rates.
    Cyclical,
    /// One-cycle learning rate policy.
    OneCycle,
}

impl SchedulerConfig {
    /// Create a step-decay scheduler configuration.
    pub fn step_decay(base_lr: f64, gamma: f64, step_size: usize) -> Self {
        SchedulerConfig {
            scheduler_type: SchedulerType::StepDecay,
            base_lr,
            max_lr: None,
            min_lr: None,
            total_steps: None,
            step_size: Some(step_size),
            gamma: Some(gamma),
            warmup_steps: None,
            pct_start: None,
        }
    }

    /// Create a cosine annealing scheduler configuration.
    pub fn cosine(base_lr: f64, min_lr: f64, t_max: usize) -> Self {
        SchedulerConfig {
            scheduler_type: SchedulerType::CosineAnnealing,
            base_lr,
            max_lr: None,
            min_lr: Some(min_lr),
            total_steps: Some(t_max),
            step_size: None,
            gamma: None,
            warmup_steps: None,
            pct_start: None,
        }
    }

    /// Create a one-cycle scheduler configuration.
    pub fn one_cycle(base_lr: f64, max_lr: f64, total_steps: usize) -> Self {
        SchedulerConfig {
            scheduler_type: SchedulerType::OneCycle,
            base_lr,
            max_lr: Some(max_lr),
            min_lr: Some(base_lr * 0.01),
            total_steps: Some(total_steps),
            step_size: None,
            gamma: None,
            warmup_steps: None,
            pct_start: Some(0.3),
        }
    }
}

// ------- Tests -------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ---- StepDecayScheduler ----

    #[test]
    fn test_step_decay_initial_lr() {
        let s = StepDecayScheduler::new(0.1, 0.5, 10).expect("valid config");
        assert_abs_diff_eq!(s.current_lr(), 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_step_decay_after_step_size() {
        let mut s = StepDecayScheduler::new(0.1, 0.5, 5).expect("valid config");
        for _ in 0..5 {
            s.step();
        }
        // floor(5/5) = 1 → 0.1 * 0.5^1 = 0.05
        assert_abs_diff_eq!(s.current_lr(), 0.05, epsilon = 1e-10);
    }

    #[test]
    fn test_step_decay_multiple_decays() {
        let mut s = StepDecayScheduler::new(0.1, 0.5, 4).expect("valid config");
        for _ in 0..12 {
            s.step();
        }
        // floor(12/4) = 3 → 0.1 * 0.5^3 = 0.0125
        assert_abs_diff_eq!(s.current_lr(), 0.0125, epsilon = 1e-10);
    }

    #[test]
    fn test_step_decay_invalid_gamma() {
        let result = StepDecayScheduler::new(0.1, 1.5, 10);
        assert!(result.is_err(), "gamma > 1.0 should return Err");
    }

    #[test]
    fn test_step_decay_reset() {
        let mut s = StepDecayScheduler::new(0.1, 0.5, 5).expect("valid config");
        for _ in 0..10 {
            s.step();
        }
        let after_steps = s.current_lr();
        assert!(after_steps < 0.1, "LR should have decayed");
        s.reset();
        assert_abs_diff_eq!(s.current_lr(), 0.1, epsilon = 1e-10);
        assert_eq!(s.steps_taken(), 0);
    }

    // ---- CosineAnnealingScheduler ----

    #[test]
    fn test_cosine_initial_is_max() {
        let s = CosineAnnealingScheduler::new(0.1, 0.001, 100).expect("valid config");
        // At step 0, cos(0) = 1 → lr = min + 0.5*(max-min)*(1+1) = max
        assert_abs_diff_eq!(s.current_lr(), 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_at_tmax() {
        let mut s = CosineAnnealingScheduler::new(0.1, 0.001, 100).expect("valid config");
        for _ in 0..100 {
            s.step();
        }
        // At t_max, cos(pi) = -1 → lr = min + 0.5*(max-min)*0 = min
        assert_abs_diff_eq!(s.current_lr(), 0.001, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_monotone_decrease() {
        let mut s = CosineAnnealingScheduler::new(0.1, 0.001, 50).expect("valid config");
        let mut prev = s.current_lr();
        for _ in 0..50 {
            let lr = s.step();
            assert!(
                lr <= prev + 1e-12,
                "LR should not increase: prev={prev}, lr={lr}"
            );
            prev = lr;
        }
    }

    #[test]
    fn test_cosine_warm_restarts_resets() {
        let period = 10;
        let mut s = CosineAnnealingScheduler::new(0.1, 0.001, 100)
            .expect("valid config")
            .with_warm_restarts(period);

        // Step up to just before the restart
        for _ in 0..(period - 1) {
            s.step();
        }
        let lr_before_restart = s.current_lr();

        // This step triggers the restart (current_step == period)
        let lr_after_restart = s.step();

        // After restart, current_step resets to 0 → LR should be near max
        assert!(
            lr_after_restart > lr_before_restart,
            "LR should increase after warm restart: before={lr_before_restart}, after={lr_after_restart}"
        );
        assert!(s.completed_cycle());
    }

    #[test]
    fn test_cosine_invalid_config() {
        let result = CosineAnnealingScheduler::new(0.001, 0.1, 100);
        assert!(result.is_err(), "max_lr < min_lr should return Err");
    }

    // ---- WarmupScheduler ----

    #[test]
    fn test_warmup_starts_low() {
        let inner = Box::new(CosineAnnealingScheduler::new(0.1, 0.001, 100).expect("valid inner"));
        let mut s = WarmupScheduler::new(10, 1e-6, 0.1, inner).expect("valid warmup config");
        // step 1 → frac = 1/10 → lr ≈ 1e-6 + 0.1*(0.1 - 1e-6)
        let lr = s.step();
        assert!(
            lr < 0.1,
            "First warmup LR should be much less than warmup_end_lr"
        );
        assert!(lr > 0.0, "First warmup LR should be positive");
    }

    #[test]
    fn test_warmup_ends_high() {
        let inner = Box::new(CosineAnnealingScheduler::new(0.1, 0.001, 100).expect("valid inner"));
        let mut s = WarmupScheduler::new(5, 0.0, 0.1, inner).expect("valid warmup config");
        // After warmup_steps steps, delegates to inner
        for _ in 0..5 {
            s.step();
        }
        // Now inner has been stepped once (step at current_step == warmup_steps)
        // The inner scheduler should return a value >= min_lr
        let lr = s.current_lr();
        assert!(
            lr >= 0.001,
            "After warmup, LR should be from inner scheduler (>= min_lr)"
        );
    }

    #[test]
    fn test_warmup_invalid_zero_steps() {
        let inner = Box::new(CosineAnnealingScheduler::new(0.1, 0.001, 100).expect("valid inner"));
        let result = WarmupScheduler::new(0, 0.0, 0.1, inner);
        assert!(result.is_err(), "warmup_steps=0 should return Err");
    }

    // ---- CyclicalScheduler ----

    #[test]
    fn test_cyclical_min_at_start() {
        let s = CyclicalScheduler::new(0.001, 0.1, 5).expect("valid config");
        // At step 0: cycle=0, x = 0/5 - 0 - 1 = -1, frac = max(0, 1-1) = 0 → min_lr
        assert_abs_diff_eq!(s.current_lr(), 0.001, epsilon = 1e-10);
    }

    #[test]
    fn test_cyclical_max_at_half_period() {
        let mut s = CyclicalScheduler::new(0.001, 0.1, 5).expect("valid config");
        // At step step_size=5: cycle=0, x = 5/5 - 0 - 1 = 0, frac=1 → max_lr
        for _ in 0..5 {
            s.step();
        }
        assert_abs_diff_eq!(s.current_lr(), 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_cyclical_min_at_full_period() {
        let step_size = 5;
        let mut s = CyclicalScheduler::new(0.001, 0.1, step_size).expect("valid config");
        // At step 2*step_size=10: cycle=1, x = 10/5 - 2*1 - 1 = -1, frac=0 → min_lr
        for _ in 0..(2 * step_size) {
            s.step();
        }
        assert_abs_diff_eq!(s.current_lr(), 0.001, epsilon = 1e-10);
    }

    // ---- OneCycleLrScheduler ----

    #[test]
    fn test_one_cycle_starts_at_base() {
        let s = OneCycleLrScheduler::new(0.001, 0.1, 0.0001, 100, 0.3).expect("valid config");
        // At step 0 (no steps taken): frac=0 → base_lr
        assert_abs_diff_eq!(s.current_lr(), 0.001, epsilon = 1e-10);
    }

    #[test]
    fn test_one_cycle_peaks_at_warmup_end() {
        let total_steps = 100;
        let pct_start = 0.3;
        let base_lr = 0.001;
        let max_lr = 0.1;
        let mut s = OneCycleLrScheduler::new(base_lr, max_lr, 0.0001, total_steps, pct_start)
            .expect("valid config");
        let warmup_steps = (total_steps as f64 * pct_start) as usize; // 30
        for _ in 0..warmup_steps {
            s.step();
        }
        // At exactly warmup_steps: frac=1.0 → base_lr + 1.0*(max_lr-base_lr) = max_lr
        assert_abs_diff_eq!(s.current_lr(), max_lr, epsilon = 1e-9);
    }

    #[test]
    fn test_one_cycle_ends_at_min() {
        let total_steps = 100;
        let min_lr = 0.0001;
        let mut s =
            OneCycleLrScheduler::new(0.001, 0.1, min_lr, total_steps, 0.3).expect("valid config");
        for _ in 0..total_steps {
            s.step();
        }
        // At total_steps: frac=1.0 in decay phase, cos(pi)=-1 → min_lr + 0 = min_lr
        assert_abs_diff_eq!(s.current_lr(), min_lr, epsilon = 1e-9);
    }

    // ---- SchedulerConfig builders ----

    #[test]
    fn test_scheduler_config_builders() {
        let step_cfg = SchedulerConfig::step_decay(0.1, 0.5, 10);
        assert_eq!(step_cfg.scheduler_type, SchedulerType::StepDecay);
        assert_abs_diff_eq!(step_cfg.base_lr, 0.1, epsilon = 1e-10);
        assert_eq!(step_cfg.gamma, Some(0.5));
        assert_eq!(step_cfg.step_size, Some(10));

        let cosine_cfg = SchedulerConfig::cosine(0.1, 0.001, 100);
        assert_eq!(cosine_cfg.scheduler_type, SchedulerType::CosineAnnealing);
        assert_abs_diff_eq!(cosine_cfg.base_lr, 0.1, epsilon = 1e-10);
        assert_eq!(cosine_cfg.min_lr, Some(0.001));
        assert_eq!(cosine_cfg.total_steps, Some(100));

        let oc_cfg = SchedulerConfig::one_cycle(0.001, 0.1, 500);
        assert_eq!(oc_cfg.scheduler_type, SchedulerType::OneCycle);
        assert_abs_diff_eq!(oc_cfg.base_lr, 0.001, epsilon = 1e-10);
        assert_eq!(oc_cfg.max_lr, Some(0.1));
        assert_eq!(oc_cfg.total_steps, Some(500));
        assert_eq!(oc_cfg.pct_start, Some(0.3));
        // min_lr should be base_lr * 0.01
        assert_abs_diff_eq!(oc_cfg.min_lr.unwrap(), 0.001 * 0.01, epsilon = 1e-15);
    }
}
