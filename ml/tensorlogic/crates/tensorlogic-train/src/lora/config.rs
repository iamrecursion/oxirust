//! LoRA configuration.

/// Configuration for a LoRA adapter layer.
///
/// LoRA (Low-Rank Adaptation, Hu et al., 2021) decomposes weight updates as
/// `dW = B @ A` where `B in R^{d x r}`, `A in R^{r x k}`, and `r << min(d, k)`.
/// The effective scaling factor applied is `alpha / rank`.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Low-rank dimension r.
    pub rank: usize,
    /// Scaling factor: effective scaling = alpha / rank.
    pub alpha: f64,
    /// Dropout probability for training regularisation (0.0 = no dropout).
    pub dropout: f64,
    /// Which weight matrices to adapt (matched by name).
    pub target_modules: Vec<String>,
    /// RNG seed for reproducible A-matrix initialisation.
    pub seed: u64,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: Vec::new(),
            seed: 42,
        }
    }
}
