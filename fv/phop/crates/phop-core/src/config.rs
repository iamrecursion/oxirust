//! Discovery configuration.

use serde::{Deserialize, Serialize};

/// Temperature annealing schedule for the Gumbel-Softmax topology relaxation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TempSchedule {
    /// Linear interpolation from `tau_start` to `tau_end`.
    Linear,
    /// Cosine interpolation from `tau_start` to `tau_end` (default).
    #[default]
    Cosine,
}

/// Compute backend for the expensive numeric inner loops (constant fitting).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Backend {
    /// CPU (always available; exact `f64`).
    #[default]
    Cpu,
    /// NVIDIA CUDA GPU (requires the `gpu-cuda` feature and a device at runtime; falls back to
    /// CPU otherwise). Single-precision coarse fit, refined to `f64` by the CPU LM polish.
    Cuda,
    /// Apple Metal GPU (requires the `gpu-metal` feature and a Metal device at runtime; falls back
    /// to CPU otherwise, macOS only). Single-precision coarse forward; exact `f64` stays on the CPU.
    Metal,
}

/// Configuration for a [`crate::Discoverer`] run.
///
/// Construct with [`Config::default`] and adjust via the builder methods, e.g.
/// ```
/// use phop_core::Config;
/// let cfg = Config::default().population(256).max_depth(10).max_epochs(2_000);
/// assert_eq!(cfg.population, 256);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Number of candidate trees evaluated jointly.
    pub population: usize,
    /// Maximum depth of candidate EML trees.
    pub max_depth: usize,
    /// Maximum number of optimization epochs.
    pub max_epochs: usize,
    /// Adam learning rate.
    pub learning_rate: f64,
    /// Weight on the complexity penalty in the multi-objective loss.
    pub lambda_complexity: f64,
    /// Weight on the sparsity penalty (pressure toward the constant `1`).
    pub lambda_sparsity: f64,
    /// Weight on the parsimony (depth) penalty.
    pub lambda_parsimony: f64,
    /// Initial Gumbel-Softmax temperature.
    pub tau_start: f64,
    /// Final Gumbel-Softmax temperature.
    pub tau_end: f64,
    /// Temperature annealing schedule.
    pub temp_schedule: TempSchedule,
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Number of solutions to keep on the Pareto front.
    pub top_k: usize,
    /// Compute backend for constant fitting (CPU by default; CUDA when built and available).
    pub backend: Backend,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            population: 256,
            max_depth: 10,
            max_epochs: 2_000,
            learning_rate: 0.05,
            lambda_complexity: 1e-3,
            lambda_sparsity: 1e-3,
            lambda_parsimony: 1e-3,
            tau_start: 2.0,
            tau_end: 0.1,
            temp_schedule: TempSchedule::Cosine,
            seed: 0,
            top_k: 5,
            backend: Backend::Cpu,
        }
    }
}

impl Config {
    /// Set the population size.
    #[must_use]
    pub fn population(mut self, p: usize) -> Self {
        self.population = p;
        self
    }

    /// Set the maximum tree depth.
    #[must_use]
    pub fn max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }

    /// Set the maximum number of epochs.
    #[must_use]
    pub fn max_epochs(mut self, n: usize) -> Self {
        self.max_epochs = n;
        self
    }

    /// Set the Adam learning rate.
    #[must_use]
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the RNG seed.
    #[must_use]
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Set the number of Pareto solutions to keep.
    #[must_use]
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set the compute backend for constant fitting.
    #[must_use]
    pub fn backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Temperature at training progress `t in [0, 1]` under the configured schedule.
    #[must_use]
    pub fn temperature(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self.temp_schedule {
            TempSchedule::Linear => self.tau_start + (self.tau_end - self.tau_start) * t,
            TempSchedule::Cosine => {
                let c = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
                self.tau_end + (self.tau_start - self.tau_end) * c
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_defaults_to_cpu_and_builder_sets_cuda() {
        assert_eq!(Config::default().backend, Backend::Cpu);
        assert_eq!(
            Config::default().backend(Backend::Cuda).backend,
            Backend::Cuda
        );
    }
}
