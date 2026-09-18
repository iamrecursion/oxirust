//! Configuration for low-rank tensor approximation.

/// Configuration for low-rank approximation of tensor operations.
#[derive(Debug, Clone)]
pub struct LowRankConfig {
    /// Target rank for SVD truncation
    pub rank: usize,
    /// Maximum power iteration steps for randomized SVD
    pub max_iterations: usize,
    /// Convergence tolerance (Frobenius norm ratio)
    pub tolerance: f64,
    /// Whether to normalize singular vectors
    pub normalize: bool,
    /// Frobenius error threshold to trigger approximation (ratio to original norm)
    pub error_threshold: f64,
    /// Minimum matrix dimension to consider for approximation
    pub min_dimension: usize,
}

impl LowRankConfig {
    /// Create a new configuration with the given target rank.
    pub fn new(rank: usize) -> Self {
        LowRankConfig {
            rank,
            ..Default::default()
        }
    }

    /// Set the convergence tolerance.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set the Frobenius error threshold for approximation candidacy.
    pub fn with_error_threshold(mut self, threshold: f64) -> Self {
        self.error_threshold = threshold;
        self
    }

    /// Set the maximum power iteration steps.
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set the minimum matrix dimension to consider.
    pub fn with_min_dimension(mut self, min_dim: usize) -> Self {
        self.min_dimension = min_dim;
        self
    }
}

impl Default for LowRankConfig {
    fn default() -> Self {
        LowRankConfig {
            rank: 4,
            max_iterations: 100,
            tolerance: 1e-6,
            normalize: true,
            error_threshold: 0.1,
            min_dimension: 32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_rank_config_default() {
        let cfg = LowRankConfig::default();
        assert_eq!(cfg.rank, 4);
        assert_eq!(cfg.max_iterations, 100);
        assert!((cfg.tolerance - 1e-6).abs() < 1e-15);
        assert!(cfg.normalize);
        assert!((cfg.error_threshold - 0.1).abs() < 1e-15);
        assert_eq!(cfg.min_dimension, 32);
    }

    #[test]
    fn test_low_rank_config_builder() {
        let cfg = LowRankConfig::new(8)
            .with_tolerance(1e-4)
            .with_error_threshold(0.05)
            .with_max_iterations(50)
            .with_min_dimension(16);
        assert_eq!(cfg.rank, 8);
        assert!((cfg.tolerance - 1e-4).abs() < 1e-15);
        assert!((cfg.error_threshold - 0.05).abs() < 1e-15);
        assert_eq!(cfg.max_iterations, 50);
        assert_eq!(cfg.min_dimension, 16);
    }
}
