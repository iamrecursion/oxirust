//! Convergence monitoring and adaptive damping for iterative inference.
//!
//! Tracks message residuals during belief propagation iterations,
//! detects convergence/divergence, and provides adaptive damping schedules.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors related to convergence monitoring configuration and execution.
#[derive(Debug, Error)]
pub enum ConvergenceError {
    /// Tolerance must be a positive value.
    #[error("Invalid tolerance: {0} (must be positive)")]
    InvalidTolerance(f64),
    /// Damping factor must be within the range [0, 1].
    #[error("Invalid damping factor: {0} (must be in [0, 1])")]
    InvalidDamping(f64),
    /// The algorithm did not converge within the allowed iterations.
    #[error("Max iterations reached: {0}")]
    MaxIterationsReached(usize),
}

/// Configuration for convergence monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceConfig {
    /// Convergence tolerance (residual below this means converged).
    pub tolerance: f64,
    /// Maximum iterations before declaring non-convergence.
    pub max_iterations: usize,
    /// Initial damping factor (0 = no damping, 1 = full damping).
    pub damping_factor: f64,
    /// Number of consecutive converged iterations before declaring convergence.
    pub patience: usize,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        ConvergenceConfig {
            tolerance: 1e-6,
            max_iterations: 100,
            damping_factor: 0.5,
            patience: 3,
        }
    }
}

impl ConvergenceConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the convergence tolerance.
    pub fn with_tolerance(mut self, t: f64) -> Self {
        self.tolerance = t;
        self
    }

    /// Set the maximum number of iterations.
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set the initial damping factor.
    pub fn with_damping(mut self, d: f64) -> Self {
        self.damping_factor = d;
        self
    }

    /// Set the patience (consecutive converged iterations required).
    pub fn with_patience(mut self, p: usize) -> Self {
        self.patience = p;
        self
    }

    /// Validate the configuration parameters.
    pub fn validate(&self) -> Result<(), ConvergenceError> {
        if self.tolerance <= 0.0 {
            return Err(ConvergenceError::InvalidTolerance(self.tolerance));
        }
        if !(0.0..=1.0).contains(&self.damping_factor) {
            return Err(ConvergenceError::InvalidDamping(self.damping_factor));
        }
        Ok(())
    }
}

/// Damping schedule types for controlling how damping evolves over iterations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DampingSchedule {
    /// Fixed damping throughout all iterations.
    Fixed(f64),
    /// Linear interpolation from `start` to `end` over `total_steps`.
    Linear {
        /// Starting damping value.
        start: f64,
        /// Ending damping value.
        end: f64,
        /// Total steps over which to interpolate.
        total_steps: usize,
    },
    /// Exponential decay: `initial * decay^iteration`.
    Exponential {
        /// Initial damping value.
        initial: f64,
        /// Decay rate per iteration.
        decay: f64,
    },
    /// Increase damping when residual grows, decrease when stable.
    Adaptive {
        /// Minimum damping floor.
        base: f64,
        /// Rate at which damping increases on divergence.
        increase_rate: f64,
        /// Rate at which damping decreases on convergence.
        decrease_rate: f64,
    },
}

impl DampingSchedule {
    /// Get damping factor for a given iteration and optional residual information.
    ///
    /// # Arguments
    /// * `iteration` - Current iteration number
    /// * `prev_residual` - Residual from the previous iteration (if available)
    /// * `curr_residual` - Residual from the current iteration (if available)
    /// * `current_damping` - The current damping factor (used by adaptive schedule)
    pub fn get_damping(
        &self,
        iteration: usize,
        prev_residual: Option<f64>,
        curr_residual: Option<f64>,
        current_damping: f64,
    ) -> f64 {
        match self {
            DampingSchedule::Fixed(d) => *d,
            DampingSchedule::Linear {
                start,
                end,
                total_steps,
            } => {
                if *total_steps == 0 {
                    return *start;
                }
                let frac = (iteration as f64 / *total_steps as f64).min(1.0);
                start + frac * (end - start)
            }
            DampingSchedule::Exponential { initial, decay } => {
                initial * decay.powi(iteration as i32)
            }
            DampingSchedule::Adaptive {
                base,
                increase_rate,
                decrease_rate,
            } => match (prev_residual, curr_residual) {
                (Some(prev), Some(curr)) if curr > prev => {
                    // Diverging: increase damping
                    (current_damping + increase_rate).min(0.99)
                }
                (Some(_prev), Some(_curr)) => {
                    // Converging: decrease damping toward base
                    (current_damping - decrease_rate).max(*base)
                }
                _ => current_damping,
            },
        }
    }
}

/// Current state of convergence tracking.
#[derive(Debug, Clone)]
pub struct ConvergenceState {
    /// Current iteration count.
    pub iteration: usize,
    /// Whether the algorithm has converged.
    pub converged: bool,
    /// Whether the algorithm has diverged.
    pub diverged: bool,
    /// History of residual values per iteration.
    pub residual_history: Vec<f64>,
    /// History of damping values per iteration.
    pub damping_history: Vec<f64>,
    /// Number of consecutive iterations below tolerance.
    pub consecutive_converged: usize,
}

impl ConvergenceState {
    /// Create a fresh convergence state.
    pub fn new() -> Self {
        ConvergenceState {
            iteration: 0,
            converged: false,
            diverged: false,
            residual_history: Vec::new(),
            damping_history: Vec::new(),
            consecutive_converged: 0,
        }
    }

    /// Return the most recent residual value, if any.
    pub fn latest_residual(&self) -> Option<f64> {
        self.residual_history.last().copied()
    }

    /// Compute the convergence rate as the ratio of the last two residuals.
    ///
    /// Returns `None` if fewer than two residuals have been recorded.
    pub fn convergence_rate(&self) -> Option<f64> {
        if self.residual_history.len() < 2 {
            return None;
        }
        let n = self.residual_history.len();
        let r0 = self.residual_history[n - 2];
        let r1 = self.residual_history[n - 1];
        if r0 > 1e-15 {
            Some(r1 / r0)
        } else {
            Some(0.0)
        }
    }
}

impl Default for ConvergenceState {
    fn default() -> Self {
        Self::new()
    }
}

/// Monitors convergence of iterative algorithms such as belief propagation.
///
/// Tracks residuals, manages damping schedules, and detects convergence or divergence.
pub struct ConvergenceMonitor {
    config: ConvergenceConfig,
    schedule: DampingSchedule,
    state: ConvergenceState,
    current_damping: f64,
}

impl ConvergenceMonitor {
    /// Create a new convergence monitor with the given configuration and schedule.
    pub fn new(
        config: ConvergenceConfig,
        schedule: DampingSchedule,
    ) -> Result<Self, ConvergenceError> {
        config.validate()?;
        let initial_damping = config.damping_factor;
        Ok(ConvergenceMonitor {
            config,
            schedule,
            state: ConvergenceState::new(),
            current_damping: initial_damping,
        })
    }

    /// Create a monitor with default configuration and fixed damping.
    pub fn with_default_config() -> Self {
        let config = ConvergenceConfig::default();
        let damping = config.damping_factor;
        let schedule = DampingSchedule::Fixed(damping);
        ConvergenceMonitor {
            config,
            schedule,
            state: ConvergenceState::new(),
            current_damping: damping,
        }
    }

    /// Record a new iteration with its residual.
    ///
    /// Returns `true` if the algorithm should continue iterating,
    /// `false` if converged, diverged, or max iterations reached.
    pub fn record_iteration(&mut self, residual: f64) -> bool {
        let prev_residual = self.state.latest_residual();
        self.state.iteration += 1;
        self.state.residual_history.push(residual);

        // Update damping according to schedule
        self.current_damping = self.schedule.get_damping(
            self.state.iteration,
            prev_residual,
            Some(residual),
            self.current_damping,
        );
        self.state.damping_history.push(self.current_damping);

        // Check convergence: residual below tolerance
        if residual < self.config.tolerance {
            self.state.consecutive_converged += 1;
            if self.state.consecutive_converged >= self.config.patience {
                self.state.converged = true;
                return false;
            }
        } else {
            self.state.consecutive_converged = 0;
        }

        // Check divergence: residual growing for 5+ consecutive iterations
        if self.state.residual_history.len() >= 5 {
            let recent = &self.state.residual_history[self.state.residual_history.len() - 5..];
            let diverging = recent.windows(2).all(|w| w[1] > w[0]);
            if diverging {
                self.state.diverged = true;
                return false;
            }
        }

        // Check max iterations
        if self.state.iteration >= self.config.max_iterations {
            return false;
        }

        true
    }

    /// Get the current damping factor.
    pub fn current_damping(&self) -> f64 {
        self.current_damping
    }

    /// Get a reference to the current convergence state.
    pub fn state(&self) -> &ConvergenceState {
        &self.state
    }

    /// Check if the algorithm has converged.
    pub fn is_converged(&self) -> bool {
        self.state.converged
    }

    /// Check if the algorithm has diverged.
    pub fn is_diverged(&self) -> bool {
        self.state.diverged
    }

    /// Get the current iteration count.
    pub fn iteration(&self) -> usize {
        self.state.iteration
    }

    /// Reset the monitor to its initial state.
    pub fn reset(&mut self) {
        self.state = ConvergenceState::new();
        self.current_damping = self.config.damping_factor;
    }

    /// Get summary statistics from the inference run.
    pub fn stats(&self) -> InferenceStats {
        InferenceStats {
            total_iterations: self.state.iteration,
            final_residual: self.state.latest_residual().unwrap_or(f64::NAN),
            converged: self.state.converged,
            diverged: self.state.diverged,
            convergence_rate: self.state.convergence_rate(),
            final_damping: self.current_damping,
        }
    }
}

/// Summary statistics from an inference run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceStats {
    /// Total number of iterations executed.
    pub total_iterations: usize,
    /// The residual value at the final iteration.
    pub final_residual: f64,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Whether the algorithm diverged.
    pub diverged: bool,
    /// Convergence rate (ratio of last two residuals), if available.
    pub convergence_rate: Option<f64>,
    /// The damping factor at the final iteration.
    pub final_damping: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ConvergenceConfig::default();
        assert!((config.tolerance - 1e-6).abs() < 1e-15);
        assert_eq!(config.max_iterations, 100);
        assert!((config.damping_factor - 0.5).abs() < 1e-15);
        assert_eq!(config.patience, 3);
    }

    #[test]
    fn test_config_validate_good() {
        let config = ConvergenceConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_bad_tolerance() {
        let config = ConvergenceConfig::new().with_tolerance(0.0);
        let err = config.validate().unwrap_err();
        assert!(matches!(err, ConvergenceError::InvalidTolerance(_)));
    }

    #[test]
    fn test_config_validate_bad_damping() {
        let config = ConvergenceConfig::new().with_damping(2.0);
        let err = config.validate().unwrap_err();
        assert!(matches!(err, ConvergenceError::InvalidDamping(_)));
    }

    #[test]
    fn test_config_builder() {
        let config = ConvergenceConfig::new()
            .with_tolerance(1e-4)
            .with_max_iterations(50)
            .with_damping(0.3)
            .with_patience(5);
        assert!((config.tolerance - 1e-4).abs() < 1e-15);
        assert_eq!(config.max_iterations, 50);
        assert!((config.damping_factor - 0.3).abs() < 1e-15);
        assert_eq!(config.patience, 5);
    }

    #[test]
    fn test_damping_fixed() {
        let schedule = DampingSchedule::Fixed(0.7);
        assert!((schedule.get_damping(0, None, None, 0.5) - 0.7).abs() < 1e-15);
        assert!((schedule.get_damping(10, Some(0.1), Some(0.05), 0.5) - 0.7).abs() < 1e-15);
        assert!((schedule.get_damping(100, None, None, 0.9) - 0.7).abs() < 1e-15);
    }

    #[test]
    fn test_damping_linear() {
        let schedule = DampingSchedule::Linear {
            start: 0.8,
            end: 0.2,
            total_steps: 10,
        };
        // At step 0
        assert!((schedule.get_damping(0, None, None, 0.0) - 0.8).abs() < 1e-15);
        // At step 5 (midpoint)
        assert!((schedule.get_damping(5, None, None, 0.0) - 0.5).abs() < 1e-15);
        // At step 10
        assert!((schedule.get_damping(10, None, None, 0.0) - 0.2).abs() < 1e-15);
        // Beyond total_steps, clamps at end
        assert!((schedule.get_damping(20, None, None, 0.0) - 0.2).abs() < 1e-15);
    }

    #[test]
    fn test_damping_exponential() {
        let schedule = DampingSchedule::Exponential {
            initial: 1.0,
            decay: 0.5,
        };
        // Step 0: 1.0 * 0.5^0 = 1.0
        assert!((schedule.get_damping(0, None, None, 0.0) - 1.0).abs() < 1e-15);
        // Step 1: 1.0 * 0.5^1 = 0.5
        assert!((schedule.get_damping(1, None, None, 0.0) - 0.5).abs() < 1e-15);
        // Step 2: 1.0 * 0.5^2 = 0.25
        assert!((schedule.get_damping(2, None, None, 0.0) - 0.25).abs() < 1e-15);
    }

    #[test]
    fn test_damping_adaptive_increases_on_diverge() {
        let schedule = DampingSchedule::Adaptive {
            base: 0.1,
            increase_rate: 0.1,
            decrease_rate: 0.05,
        };
        // Residual grew from 0.5 to 0.8 => damping should increase
        let result = schedule.get_damping(1, Some(0.5), Some(0.8), 0.4);
        assert!((result - 0.5).abs() < 1e-15); // 0.4 + 0.1 = 0.5
    }

    #[test]
    fn test_damping_adaptive_decreases_on_converge() {
        let schedule = DampingSchedule::Adaptive {
            base: 0.1,
            increase_rate: 0.1,
            decrease_rate: 0.05,
        };
        // Residual dropped from 0.8 to 0.5 => damping should decrease
        let result = schedule.get_damping(1, Some(0.8), Some(0.5), 0.4);
        assert!((result - 0.35).abs() < 1e-15); // 0.4 - 0.05 = 0.35
    }

    #[test]
    fn test_monitor_converges() {
        let config = ConvergenceConfig::new()
            .with_tolerance(1e-3)
            .with_patience(2);
        let monitor_result = ConvergenceMonitor::new(config, DampingSchedule::Fixed(0.5));
        assert!(monitor_result.is_ok());
        let mut monitor = monitor_result.expect("valid config");

        // Feed residuals that decrease and eventually stay below tolerance
        assert!(monitor.record_iteration(1.0));
        assert!(monitor.record_iteration(0.1));
        assert!(monitor.record_iteration(0.0009)); // below tol (1e-3), consecutive=1
                                                   // Second consecutive below tolerance => converged (patience=2)
        assert!(!monitor.record_iteration(0.0005));

        assert!(monitor.is_converged());
        assert!(!monitor.is_diverged());
    }

    #[test]
    fn test_monitor_patience() {
        let config = ConvergenceConfig::new()
            .with_tolerance(1e-3)
            .with_patience(3);
        let mut monitor =
            ConvergenceMonitor::new(config, DampingSchedule::Fixed(0.5)).expect("valid config");

        // Two below tolerance, then one above resets counter
        assert!(monitor.record_iteration(0.0001)); // consecutive=1
        assert!(monitor.record_iteration(0.0002)); // consecutive=2
        assert!(monitor.record_iteration(0.01)); // above tol, consecutive=0
        assert!(monitor.record_iteration(0.0001)); // consecutive=1
        assert!(monitor.record_iteration(0.0002)); // consecutive=2
        assert!(!monitor.record_iteration(0.0003)); // consecutive=3 => converged

        assert!(monitor.is_converged());
    }

    #[test]
    fn test_monitor_max_iterations() {
        let config = ConvergenceConfig::new()
            .with_tolerance(1e-10)
            .with_max_iterations(5);
        let mut monitor =
            ConvergenceMonitor::new(config, DampingSchedule::Fixed(0.5)).expect("valid config");

        // Feed residuals that never converge
        for i in 0..4 {
            let residual = 1.0 / (i as f64 + 1.0);
            assert!(monitor.record_iteration(residual), "iteration {i}");
        }
        // 5th iteration should return false (max reached)
        assert!(!monitor.record_iteration(0.1));
        assert!(!monitor.is_converged());
        assert_eq!(monitor.iteration(), 5);
    }

    #[test]
    fn test_monitor_diverge_detection() {
        let config = ConvergenceConfig::new()
            .with_tolerance(1e-10)
            .with_max_iterations(100);
        let mut monitor =
            ConvergenceMonitor::new(config, DampingSchedule::Fixed(0.5)).expect("valid config");

        // Feed strictly increasing residuals
        assert!(monitor.record_iteration(1.0));
        assert!(monitor.record_iteration(2.0));
        assert!(monitor.record_iteration(3.0));
        assert!(monitor.record_iteration(4.0));
        // 5th growing residual triggers divergence
        assert!(!monitor.record_iteration(5.0));

        assert!(monitor.is_diverged());
        assert!(!monitor.is_converged());
    }

    #[test]
    fn test_monitor_reset() {
        let config = ConvergenceConfig::new()
            .with_tolerance(1e-3)
            .with_patience(1);
        let mut monitor =
            ConvergenceMonitor::new(config, DampingSchedule::Fixed(0.5)).expect("valid config");

        // Converge
        assert!(!monitor.record_iteration(0.0001));
        assert!(monitor.is_converged());
        assert_eq!(monitor.iteration(), 1);

        // Reset
        monitor.reset();
        assert!(!monitor.is_converged());
        assert!(!monitor.is_diverged());
        assert_eq!(monitor.iteration(), 0);
        assert!(monitor.state().residual_history.is_empty());
    }

    #[test]
    fn test_monitor_stats() {
        let config = ConvergenceConfig::new()
            .with_tolerance(1e-3)
            .with_patience(2);
        let mut monitor =
            ConvergenceMonitor::new(config, DampingSchedule::Fixed(0.3)).expect("valid config");

        monitor.record_iteration(0.5);
        monitor.record_iteration(0.0001);
        monitor.record_iteration(0.00005);

        let stats = monitor.stats();
        assert_eq!(stats.total_iterations, 3);
        assert!((stats.final_residual - 0.00005).abs() < 1e-15);
        assert!(stats.converged);
        assert!(!stats.diverged);
        assert!((stats.final_damping - 0.3).abs() < 1e-15);
        assert!(stats.convergence_rate.is_some());
    }

    #[test]
    fn test_convergence_rate() {
        let mut state = ConvergenceState::new();
        // No residuals
        assert!(state.convergence_rate().is_none());

        // One residual
        state.residual_history.push(1.0);
        assert!(state.convergence_rate().is_none());

        // Two residuals: rate = 0.5 / 1.0 = 0.5
        state.residual_history.push(0.5);
        let rate = state.convergence_rate().expect("should have rate");
        assert!((rate - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_state_default() {
        let state = ConvergenceState::default();
        assert_eq!(state.iteration, 0);
        assert!(!state.converged);
        assert!(!state.diverged);
        assert!(state.residual_history.is_empty());
        assert!(state.damping_history.is_empty());
        assert_eq!(state.consecutive_converged, 0);
    }
}
