//! Gradient checkpointing utilities

/// Configuration for gradient checkpointing
#[derive(Debug, Clone)]
pub struct GradientCheckpointingConfig {
    /// Whether to enable gradient checkpointing
    pub enabled: bool,
    /// Checkpointing frequency
    pub checkpoint_frequency: usize,
}

impl Default for GradientCheckpointingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            checkpoint_frequency: 1,
        }
    }
}
