//! Sparse training utilities

/// Configuration for sparse training
#[derive(Debug, Clone)]
pub struct SparseTrainingConfig {
    /// Whether to enable sparse training
    pub enabled: bool,
    /// Sparsity ratio
    pub sparsity_ratio: f32,
}

impl Default for SparseTrainingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sparsity_ratio: 0.0,
        }
    }
}
