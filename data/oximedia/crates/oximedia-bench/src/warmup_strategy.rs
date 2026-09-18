#![allow(dead_code)]
//! Configurable warmup strategies for benchmarks.
//!
//! Warmup is essential for reliable benchmarking because the first few
//! iterations often trigger JIT compilation, cache population, or OS
//! scheduling effects. This module provides several warmup strategies and
//! helpers for determining when the system has reached a steady state.

use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Kind of warmup strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmupKind {
    /// Run a fixed number of warmup iterations.
    FixedIterations,
    /// Run warmup for a fixed wall-clock duration.
    FixedDuration,
    /// Run until the coefficient of variation drops below a threshold.
    AdaptiveCv,
    /// Skip warmup entirely.
    None,
}

/// Configuration for the warmup phase.
#[derive(Debug, Clone)]
pub struct WarmupConfig {
    /// Which strategy to use.
    pub kind: WarmupKind,
    /// Number of iterations (used by `FixedIterations`).
    pub iterations: usize,
    /// Duration (used by `FixedDuration`).
    pub duration: Duration,
    /// Coefficient-of-variation threshold (used by `AdaptiveCv`).
    pub cv_threshold: f64,
    /// Maximum iterations for adaptive strategies.
    pub max_iterations: usize,
    /// Minimum iterations for adaptive strategies.
    pub min_iterations: usize,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            kind: WarmupKind::FixedIterations,
            iterations: 3,
            duration: Duration::from_secs(2),
            cv_threshold: 0.05,
            max_iterations: 50,
            min_iterations: 2,
        }
    }
}

impl WarmupConfig {
    /// Create a new config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the warmup kind.
    pub fn with_kind(mut self, kind: WarmupKind) -> Self {
        self.kind = kind;
        self
    }

    /// Set the number of fixed iterations.
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.iterations = n;
        self
    }

    /// Set the fixed duration.
    pub fn with_duration(mut self, d: Duration) -> Self {
        self.duration = d;
        self
    }

    /// Set the CV threshold for adaptive warmup.
    pub fn with_cv_threshold(mut self, cv: f64) -> Self {
        self.cv_threshold = cv;
        self
    }

    /// Set the maximum iterations for adaptive warmup.
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }
}

/// Outcome of a warmup phase.
#[derive(Debug, Clone)]
pub struct WarmupResult {
    /// Total wall-clock time spent warming up.
    pub elapsed: Duration,
    /// Number of iterations executed.
    pub iterations_run: usize,
    /// Whether steady-state was reached (for adaptive strategies).
    pub steady_state_reached: bool,
    /// Per-iteration timings (nanoseconds).
    pub iter_nanos: Vec<u64>,
}

impl WarmupResult {
    /// Mean iteration time in nanoseconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_nanos(&self) -> f64 {
        if self.iter_nanos.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.iter_nanos.iter().sum();
        sum as f64 / self.iter_nanos.len() as f64
    }

    /// Coefficient of variation of the iteration times.
    #[allow(clippy::cast_precision_loss)]
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.iter_nanos.len() < 2 {
            return f64::INFINITY;
        }
        let mean = self.mean_nanos();
        if mean == 0.0 {
            return f64::INFINITY;
        }
        let variance: f64 = self
            .iter_nanos
            .iter()
            .map(|&n| {
                let diff = n as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / (self.iter_nanos.len() as f64 - 1.0);
        variance.sqrt() / mean
    }
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

/// Executes warmup according to the configured strategy.
#[derive(Debug, Clone)]
pub struct WarmupRunner {
    /// Configuration.
    config: WarmupConfig,
}

impl WarmupRunner {
    /// Create a new runner.
    pub fn new(config: WarmupConfig) -> Self {
        Self { config }
    }

    /// Run warmup, calling `work` for each iteration.
    /// Returns the warmup result.
    pub fn run<F: FnMut()>(&self, mut work: F) -> WarmupResult {
        match self.config.kind {
            WarmupKind::None => WarmupResult {
                elapsed: Duration::ZERO,
                iterations_run: 0,
                steady_state_reached: true,
                iter_nanos: Vec::new(),
            },
            WarmupKind::FixedIterations => self.run_fixed_iterations(&mut work),
            WarmupKind::FixedDuration => self.run_fixed_duration(&mut work),
            WarmupKind::AdaptiveCv => self.run_adaptive_cv(&mut work),
        }
    }

    fn run_fixed_iterations<F: FnMut()>(&self, work: &mut F) -> WarmupResult {
        let total_start = Instant::now();
        let mut timings = Vec::with_capacity(self.config.iterations);

        for _ in 0..self.config.iterations {
            let start = Instant::now();
            work();
            timings.push(start.elapsed().as_nanos() as u64);
        }

        WarmupResult {
            elapsed: total_start.elapsed(),
            iterations_run: self.config.iterations,
            steady_state_reached: true,
            iter_nanos: timings,
        }
    }

    fn run_fixed_duration<F: FnMut()>(&self, work: &mut F) -> WarmupResult {
        let total_start = Instant::now();
        let mut timings = Vec::new();
        let mut count = 0;

        while total_start.elapsed() < self.config.duration {
            let start = Instant::now();
            work();
            timings.push(start.elapsed().as_nanos() as u64);
            count += 1;
            if count >= self.config.max_iterations {
                break;
            }
        }

        WarmupResult {
            elapsed: total_start.elapsed(),
            iterations_run: count,
            steady_state_reached: true,
            iter_nanos: timings,
        }
    }

    fn run_adaptive_cv<F: FnMut()>(&self, work: &mut F) -> WarmupResult {
        let total_start = Instant::now();
        let mut timings: Vec<u64> = Vec::new();
        let mut steady = false;

        for i in 0..self.config.max_iterations {
            let start = Instant::now();
            work();
            timings.push(start.elapsed().as_nanos() as u64);

            if i + 1 >= self.config.min_iterations {
                let cv = cv_of(&timings);
                if cv <= self.config.cv_threshold {
                    steady = true;
                    break;
                }
            }
        }

        WarmupResult {
            elapsed: total_start.elapsed(),
            iterations_run: timings.len(),
            steady_state_reached: steady,
            iter_nanos: timings,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Coefficient of variation for a slice of u64.
#[allow(clippy::cast_precision_loss)]
fn cv_of(values: &[u64]) -> f64 {
    if values.len() < 2 {
        return f64::INFINITY;
    }
    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<u64>() as f64 / n;
    if mean == 0.0 {
        return f64::INFINITY;
    }
    let var: f64 = values
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    var.sqrt() / mean
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = WarmupConfig::default();
        assert_eq!(cfg.kind, WarmupKind::FixedIterations);
        assert_eq!(cfg.iterations, 3);
    }

    #[test]
    fn test_config_builder() {
        let cfg = WarmupConfig::new()
            .with_kind(WarmupKind::AdaptiveCv)
            .with_iterations(10)
            .with_cv_threshold(0.03)
            .with_max_iterations(100);
        assert_eq!(cfg.kind, WarmupKind::AdaptiveCv);
        assert_eq!(cfg.iterations, 10);
        assert!((cfg.cv_threshold - 0.03).abs() < f64::EPSILON);
        assert_eq!(cfg.max_iterations, 100);
    }

    #[test]
    fn test_warmup_none() {
        let runner = WarmupRunner::new(WarmupConfig::new().with_kind(WarmupKind::None));
        let result = runner.run(|| {});
        assert_eq!(result.iterations_run, 0);
        assert!(result.steady_state_reached);
    }

    #[test]
    fn test_warmup_fixed_iterations() {
        let runner = WarmupRunner::new(WarmupConfig::new().with_iterations(5));
        let mut count = 0u32;
        let result = runner.run(|| {
            count += 1;
        });
        assert_eq!(result.iterations_run, 5);
        assert_eq!(count, 5);
        assert_eq!(result.iter_nanos.len(), 5);
    }

    #[test]
    fn test_warmup_fixed_duration() {
        let cfg = WarmupConfig::new()
            .with_kind(WarmupKind::FixedDuration)
            .with_duration(Duration::from_millis(50))
            .with_max_iterations(1000);
        let runner = WarmupRunner::new(cfg);
        let mut counter = 0u64;
        let result = runner.run(|| {
            counter += 1;
            std::hint::black_box(counter);
        });
        assert!(result.iterations_run > 0);
        assert!(result.steady_state_reached);
    }

    #[test]
    fn test_warmup_adaptive_cv() {
        let cfg = WarmupConfig::new()
            .with_kind(WarmupKind::AdaptiveCv)
            .with_cv_threshold(0.5)
            .with_max_iterations(20);
        let runner = WarmupRunner::new(cfg);
        let result = runner.run(|| {
            std::hint::black_box(1 + 1);
        });
        assert!(result.iterations_run >= 2);
    }

    #[test]
    fn test_warmup_result_mean() {
        let r = WarmupResult {
            elapsed: Duration::ZERO,
            iterations_run: 3,
            steady_state_reached: true,
            iter_nanos: vec![100, 200, 300],
        };
        assert!((r.mean_nanos() - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_warmup_result_cv() {
        let r = WarmupResult {
            elapsed: Duration::ZERO,
            iterations_run: 3,
            steady_state_reached: true,
            iter_nanos: vec![100, 100, 100],
        };
        assert!((r.coefficient_of_variation() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_warmup_result_cv_varying() {
        let r = WarmupResult {
            elapsed: Duration::ZERO,
            iterations_run: 4,
            steady_state_reached: true,
            iter_nanos: vec![100, 200, 100, 200],
        };
        let cv = r.coefficient_of_variation();
        assert!(cv > 0.0);
        assert!(cv < 1.0);
    }

    #[test]
    fn test_warmup_result_empty_mean() {
        let r = WarmupResult {
            elapsed: Duration::ZERO,
            iterations_run: 0,
            steady_state_reached: true,
            iter_nanos: vec![],
        };
        assert!((r.mean_nanos() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_warmup_result_single_cv() {
        let r = WarmupResult {
            elapsed: Duration::ZERO,
            iterations_run: 1,
            steady_state_reached: true,
            iter_nanos: vec![42],
        };
        assert!(r.coefficient_of_variation().is_infinite());
    }

    #[test]
    fn test_cv_of_constant() {
        let vals = vec![50u64, 50, 50, 50];
        assert!((cv_of(&vals) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cv_of_empty() {
        assert!(cv_of(&[]).is_infinite());
    }
}
