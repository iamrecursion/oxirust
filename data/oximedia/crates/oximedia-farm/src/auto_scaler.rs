#![allow(dead_code)]
//! Auto-scaling engine for the encoding farm worker pool.
//!
//! [`AutoScaler`] evaluates the current queue depth and recommends whether the
//! farm should scale up (add workers) or scale down (remove workers) based on
//! configurable thresholds.  It enforces a cooldown period between successive
//! scaling events so the pool does not thrash.
//!
//! The caller is responsible for:
//! 1. Calling [`AutoScaler::evaluate`] periodically with the current queue depth
//!    and a monotonic Unix timestamp.
//! 2. Deciding whether to act on the returned [`ScaleDecision`].
//! 3. Calling [`AutoScaler::apply_decision`] to record the decision so that the
//!    cooldown and worker count are updated correctly.
//!
//! # Example
//!
//! ```
//! use oximedia_farm::auto_scaler::{AutoScaleConfig, AutoScaler, ScaleDecision};
//!
//! let config = AutoScaleConfig {
//!     min_workers: 2,
//!     max_workers: 16,
//!     scale_up_threshold: 3.0,
//!     scale_down_threshold: 0.5,
//!     cooldown_secs: 60,
//!     scale_up_step: 2,
//!     scale_down_step: 1,
//! };
//! let mut scaler = AutoScaler::new(config, 4);
//!
//! // Queue is deep → scale up recommended.
//! let decision = scaler.evaluate(20, 0);
//! assert_eq!(decision, ScaleDecision::ScaleUp(2));
//! scaler.apply_decision(&decision, 0);
//! assert_eq!(scaler.current_workers(), 6);
//! ```

// ── ScaleDecision ─────────────────────────────────────────────────────────────

/// The scaling recommendation produced by [`AutoScaler::evaluate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScaleDecision {
    /// No action needed; the current worker count is appropriate.
    NoChange,
    /// Add the specified number of workers.
    ScaleUp(usize),
    /// Remove the specified number of workers.
    ScaleDown(usize),
}

// ── AutoScaleConfig ───────────────────────────────────────────────────────────

/// Configuration parameters that govern when and by how much scaling occurs.
#[derive(Debug, Clone)]
pub struct AutoScaleConfig {
    /// Minimum number of workers that must be maintained at all times.
    pub min_workers: usize,
    /// Maximum number of workers the pool may grow to.
    pub max_workers: usize,
    /// Queue depth **per worker** above which scale-up is triggered.
    ///
    /// E.g. `3.0` means: if there are 4 workers and 13+ jobs queued the scaler
    /// will recommend adding workers (13 / 4 = 3.25 > 3.0).
    pub scale_up_threshold: f32,
    /// Queue depth **per worker** below which scale-down is triggered.
    ///
    /// E.g. `0.5` means: scale down when the average worker has fewer than
    /// half a job waiting.
    pub scale_down_threshold: f32,
    /// Minimum seconds that must elapse between two successive scaling events.
    pub cooldown_secs: u64,
    /// Number of workers to add in a single scale-up event.
    pub scale_up_step: usize,
    /// Number of workers to remove in a single scale-down event.
    pub scale_down_step: usize,
}

impl Default for AutoScaleConfig {
    fn default() -> Self {
        Self {
            min_workers: 1,
            max_workers: 32,
            scale_up_threshold: 3.0,
            scale_down_threshold: 0.5,
            cooldown_secs: 60,
            scale_up_step: 2,
            scale_down_step: 2,
        }
    }
}

// ── AutoScaler ────────────────────────────────────────────────────────────────

/// Stateful auto-scaler that recommends worker pool size changes.
pub struct AutoScaler {
    config: AutoScaleConfig,
    /// Current number of workers in the pool.
    current_workers: usize,
    /// Unix timestamp (seconds) of the last applied scaling event.
    /// `None` means no scaling event has ever occurred, which bypasses the cooldown
    /// so the very first evaluation is never blocked.
    last_scale_time: Option<u64>,
}

impl AutoScaler {
    /// Create a new auto-scaler with the given configuration and initial worker count.
    ///
    /// `initial_workers` is clamped to `[config.min_workers, config.max_workers]`.
    #[must_use]
    pub fn new(config: AutoScaleConfig, initial_workers: usize) -> Self {
        let clamped = initial_workers
            .max(config.min_workers)
            .min(config.max_workers);
        Self {
            config,
            current_workers: clamped,
            last_scale_time: None,
        }
    }

    /// Evaluate whether scaling is needed right now.
    ///
    /// # Parameters
    /// - `queue_depth`: total number of jobs currently waiting in the queue.
    /// - `now_secs`: current Unix timestamp in seconds (monotonically increasing).
    ///
    /// # Returns
    /// A [`ScaleDecision`] describing the recommended action.  The scaler does
    /// **not** automatically apply the decision; call [`Self::apply_decision`]
    /// to commit it.
    #[must_use]
    pub fn evaluate(&self, queue_depth: usize, now_secs: u64) -> ScaleDecision {
        // Respect the cooldown window.  `None` means no prior scaling event, so
        // the very first evaluation is never blocked.
        if let Some(last) = self.last_scale_time {
            if now_secs.saturating_sub(last) < self.config.cooldown_secs {
                return ScaleDecision::NoChange;
            }
        }

        // Edge case: no workers running → treat as infinite depth per worker.
        if self.current_workers == 0 {
            let add = self
                .config
                .scale_up_step
                .min(self.config.max_workers.saturating_sub(self.current_workers));
            return if add > 0 {
                ScaleDecision::ScaleUp(add)
            } else {
                ScaleDecision::NoChange
            };
        }

        #[allow(clippy::cast_precision_loss)]
        let depth_per_worker = queue_depth as f32 / self.current_workers as f32;

        if depth_per_worker > self.config.scale_up_threshold
            && self.current_workers < self.config.max_workers
        {
            let headroom = self.config.max_workers - self.current_workers;
            let add = self.config.scale_up_step.min(headroom);
            return ScaleDecision::ScaleUp(add);
        }

        if depth_per_worker < self.config.scale_down_threshold
            && self.current_workers > self.config.min_workers
        {
            let slack = self.current_workers - self.config.min_workers;
            let remove = self.config.scale_down_step.min(slack);
            return ScaleDecision::ScaleDown(remove);
        }

        ScaleDecision::NoChange
    }

    /// Apply a scaling decision, updating the worker count and cooldown timer.
    ///
    /// `NoChange` does **not** reset the cooldown clock; only actual scaling
    /// events (`ScaleUp`/`ScaleDown`) record `now_secs` as the last scale time.
    pub fn apply_decision(&mut self, decision: &ScaleDecision, now_secs: u64) {
        match decision {
            ScaleDecision::NoChange => {} // no state change
            ScaleDecision::ScaleUp(n) => {
                self.current_workers = (self.current_workers + n).min(self.config.max_workers);
                self.last_scale_time = Some(now_secs);
            }
            ScaleDecision::ScaleDown(n) => {
                self.current_workers = self
                    .current_workers
                    .saturating_sub(*n)
                    .max(self.config.min_workers);
                self.last_scale_time = Some(now_secs);
            }
        }
    }

    /// Return the current worker count.
    #[must_use]
    pub fn current_workers(&self) -> usize {
        self.current_workers
    }

    /// Return a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &AutoScaleConfig {
        &self.config
    }

    /// Unix timestamp (seconds) of the last applied scaling event.
    /// Returns `None` if no scaling event has ever been applied.
    #[must_use]
    pub fn last_scale_time(&self) -> Option<u64> {
        self.last_scale_time
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> AutoScaleConfig {
        AutoScaleConfig {
            min_workers: 2,
            max_workers: 16,
            scale_up_threshold: 3.0,
            scale_down_threshold: 0.5,
            cooldown_secs: 60,
            scale_up_step: 2,
            scale_down_step: 1,
        }
    }

    // ── basic decisions ───────────────────────────────────────────────────────

    #[test]
    fn test_no_change_when_within_thresholds() {
        let scaler = AutoScaler::new(default_config(), 4);
        // depth_per_worker = 8 / 4 = 2.0 — between 0.5 and 3.0
        assert_eq!(scaler.evaluate(8, 1000), ScaleDecision::NoChange);
    }

    #[test]
    fn test_scale_up_when_queue_deep() {
        let scaler = AutoScaler::new(default_config(), 4);
        // depth_per_worker = 16 / 4 = 4.0 > 3.0 → ScaleUp(2)
        assert_eq!(scaler.evaluate(16, 1000), ScaleDecision::ScaleUp(2));
    }

    #[test]
    fn test_scale_down_when_queue_shallow() {
        let scaler = AutoScaler::new(default_config(), 4);
        // depth_per_worker = 1 / 4 = 0.25 < 0.5 → ScaleDown(1)
        assert_eq!(scaler.evaluate(1, 1000), ScaleDecision::ScaleDown(1));
    }

    // ── cooldown ─────────────────────────────────────────────────────────────

    #[test]
    fn test_cooldown_prevents_rapid_scaling() {
        let mut scaler = AutoScaler::new(default_config(), 4);
        // First scale-up at t=100.
        let d1 = scaler.evaluate(20, 100);
        assert_eq!(d1, ScaleDecision::ScaleUp(2));
        scaler.apply_decision(&d1, 100);

        // 30 seconds later — still within 60s cooldown.
        assert_eq!(scaler.evaluate(20, 130), ScaleDecision::NoChange);

        // 60 seconds after the last scale event — cooldown elapsed.
        assert_eq!(scaler.evaluate(20, 160), ScaleDecision::ScaleUp(2));
    }

    // ── boundary clamping ─────────────────────────────────────────────────────

    #[test]
    fn test_scale_up_clamped_at_max_workers() {
        let config = AutoScaleConfig {
            min_workers: 1,
            max_workers: 5,
            scale_up_threshold: 1.0,
            scale_up_step: 10, // would exceed max
            ..Default::default()
        };
        let scaler = AutoScaler::new(config, 4); // already at max-1
                                                 // Recommend adding min(10, 5-4)=1 workers.
        assert_eq!(scaler.evaluate(100, 1000), ScaleDecision::ScaleUp(1));
    }

    #[test]
    fn test_scale_down_clamped_at_min_workers() {
        let config = AutoScaleConfig {
            min_workers: 3,
            max_workers: 16,
            scale_down_threshold: 0.5,
            scale_down_step: 10, // would go below min
            ..Default::default()
        };
        let scaler = AutoScaler::new(config, 4); // one above min
                                                 // Recommend removing min(10, 4-3)=1 workers.
        assert_eq!(scaler.evaluate(0, 1000), ScaleDecision::ScaleDown(1));
    }

    // ── zero workers edge case ────────────────────────────────────────────────

    #[test]
    fn test_scale_up_when_zero_workers() {
        let config = AutoScaleConfig {
            min_workers: 0, // allow zero
            max_workers: 8,
            scale_up_step: 3,
            ..Default::default()
        };
        let scaler = AutoScaler::new(config, 0);
        assert_eq!(scaler.evaluate(5, 1000), ScaleDecision::ScaleUp(3));
    }

    // ── apply_decision ────────────────────────────────────────────────────────

    #[test]
    fn test_apply_decision_scale_up() {
        let mut scaler = AutoScaler::new(default_config(), 4);
        scaler.apply_decision(&ScaleDecision::ScaleUp(2), 500);
        assert_eq!(scaler.current_workers(), 6);
        assert_eq!(scaler.last_scale_time(), Some(500));
    }

    #[test]
    fn test_apply_decision_scale_down() {
        let mut scaler = AutoScaler::new(default_config(), 6);
        scaler.apply_decision(&ScaleDecision::ScaleDown(1), 600);
        assert_eq!(scaler.current_workers(), 5);
        assert_eq!(scaler.last_scale_time(), Some(600));
    }

    #[test]
    fn test_apply_decision_no_change() {
        let mut scaler = AutoScaler::new(default_config(), 4);
        scaler.apply_decision(&ScaleDecision::NoChange, 700);
        assert_eq!(scaler.current_workers(), 4);
        assert_eq!(scaler.last_scale_time(), None); // not updated
    }

    // ── apply respects bounds ─────────────────────────────────────────────────

    #[test]
    fn test_apply_scale_up_does_not_exceed_max() {
        let mut scaler = AutoScaler::new(default_config(), 15); // max=16
        scaler.apply_decision(&ScaleDecision::ScaleUp(10), 100);
        assert_eq!(scaler.current_workers(), 16); // clamped at max
    }

    #[test]
    fn test_apply_scale_down_does_not_go_below_min() {
        let mut scaler = AutoScaler::new(default_config(), 3); // min=2
        scaler.apply_decision(&ScaleDecision::ScaleDown(10), 100);
        assert_eq!(scaler.current_workers(), 2); // clamped at min
    }

    // ── initial worker clamping ───────────────────────────────────────────────

    #[test]
    fn test_initial_workers_clamped_to_bounds() {
        let config = AutoScaleConfig {
            min_workers: 4,
            max_workers: 8,
            ..Default::default()
        };
        let scaler_low = AutoScaler::new(config.clone(), 1);
        assert_eq!(scaler_low.current_workers(), 4); // clamped up to min

        let scaler_high = AutoScaler::new(config, 100);
        assert_eq!(scaler_high.current_workers(), 8); // clamped down to max
    }
}
