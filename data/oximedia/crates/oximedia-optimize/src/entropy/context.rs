//! Context modeling optimization.

/// Entropy statistics.
#[derive(Debug, Clone, Copy)]
pub struct EntropyStats {
    /// Total bits used.
    pub total_bits: u64,
    /// Number of symbols encoded.
    pub num_symbols: u64,
    /// Average bits per symbol.
    pub avg_bits_per_symbol: f64,
    /// Compression ratio.
    pub compression_ratio: f64,
}

impl Default for EntropyStats {
    fn default() -> Self {
        Self {
            total_bits: 0,
            num_symbols: 0,
            avg_bits_per_symbol: 0.0,
            compression_ratio: 1.0,
        }
    }
}

impl EntropyStats {
    /// Creates stats from counts.
    #[must_use]
    pub fn new(total_bits: u64, num_symbols: u64, uncompressed_bits: u64) -> Self {
        let avg_bits_per_symbol = if num_symbols > 0 {
            total_bits as f64 / num_symbols as f64
        } else {
            0.0
        };

        let compression_ratio = if total_bits > 0 {
            uncompressed_bits as f64 / total_bits as f64
        } else {
            1.0
        };

        Self {
            total_bits,
            num_symbols,
            avg_bits_per_symbol,
            compression_ratio,
        }
    }
}

/// Context model for entropy coding.
#[derive(Debug, Clone)]
pub struct ContextModel {
    /// Probability states for each context.
    states: Vec<u8>,
    /// Number of contexts.
    num_contexts: usize,
}

impl ContextModel {
    /// Creates a new context model.
    #[must_use]
    pub fn new(num_contexts: usize) -> Self {
        // Initialize with neutral probability (state 63 = 0.5 probability)
        Self {
            states: vec![63; num_contexts],
            num_contexts,
        }
    }

    /// Updates context state based on symbol.
    pub fn update(&mut self, context_idx: usize, symbol: bool) {
        if context_idx >= self.num_contexts {
            return;
        }

        let state = &mut self.states[context_idx];
        if symbol {
            // Move towards 1
            *state = state.saturating_add(1).min(126);
        } else {
            // Move towards 0
            *state = state.saturating_sub(1);
        }
    }

    /// Gets probability for a context.
    #[must_use]
    pub fn get_probability(&self, context_idx: usize) -> f64 {
        if context_idx >= self.num_contexts {
            return 0.5;
        }

        // Convert state to probability
        f64::from(self.states[context_idx]) / 126.0
    }

    /// Estimates bit cost for a symbol.
    #[must_use]
    pub fn estimate_bit_cost(&self, context_idx: usize, symbol: bool) -> f64 {
        let prob = self.get_probability(context_idx);
        let symbol_prob = if symbol { prob } else { 1.0 - prob };

        if symbol_prob > 0.0 {
            -symbol_prob.log2()
        } else {
            16.0 // Maximum cost for impossible symbol
        }
    }
}

/// Context optimizer for entropy coding.
pub struct ContextOptimizer {
    models: Vec<ContextModel>,
    enable_adaptive: bool,
}

impl Default for ContextOptimizer {
    fn default() -> Self {
        Self::new(256, true)
    }
}

impl ContextOptimizer {
    /// Creates a new context optimizer.
    #[must_use]
    pub fn new(num_contexts: usize, enable_adaptive: bool) -> Self {
        Self {
            models: vec![ContextModel::new(num_contexts)],
            enable_adaptive,
        }
    }

    /// Selects optimal context for a symbol.
    #[must_use]
    pub fn select_context(&self, neighbors: &[bool], position: usize) -> usize {
        // Simple context selection based on neighbors
        let mut context = 0usize;

        for (i, &neighbor) in neighbors.iter().enumerate().take(4) {
            if neighbor {
                context |= 1 << i;
            }
        }

        // Add position-based context
        context += (position % 8) * 16;

        context.min(255)
    }

    /// Encodes a symbol and updates context.
    #[allow(dead_code)]
    pub fn encode_symbol(&mut self, symbol: bool, context_idx: usize, model_idx: usize) -> f64 {
        if model_idx >= self.models.len() {
            return 0.0;
        }

        let cost = self.models[model_idx].estimate_bit_cost(context_idx, symbol);

        if self.enable_adaptive {
            self.models[model_idx].update(context_idx, symbol);
        }

        cost
    }

    /// Calculates entropy for a sequence of symbols.
    #[must_use]
    pub fn calculate_entropy(&self, symbols: &[bool]) -> f64 {
        if symbols.is_empty() {
            return 0.0;
        }

        // Calculate empirical probabilities
        let ones = symbols.iter().filter(|&&s| s).count();
        let p1 = ones as f64 / symbols.len() as f64;
        let p0 = 1.0 - p1;

        let mut entropy = 0.0;
        if p0 > 0.0 {
            entropy -= p0 * p0.log2();
        }
        if p1 > 0.0 {
            entropy -= p1 * p1.log2();
        }

        entropy
    }

    /// Gets statistics for encoding.
    #[must_use]
    pub fn get_stats(&self, total_bits: u64, num_symbols: u64) -> EntropyStats {
        let uncompressed_bits = num_symbols; // 1 bit per symbol if uncompressed
        EntropyStats::new(total_bits, num_symbols, uncompressed_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_model_creation() {
        let model = ContextModel::new(128);
        assert_eq!(model.num_contexts, 128);
        assert_eq!(model.states.len(), 128);
    }

    #[test]
    fn test_context_model_probability() {
        let model = ContextModel::new(1);
        let prob = model.get_probability(0);
        assert!((prob - 0.5).abs() < 0.01); // Initial state should be ~0.5
    }

    #[test]
    fn test_context_model_update() {
        let mut model = ContextModel::new(1);
        let initial_prob = model.get_probability(0);

        model.update(0, true);
        let updated_prob = model.get_probability(0);
        assert!(updated_prob > initial_prob); // Probability should increase
    }

    #[test]
    fn test_bit_cost_estimation() {
        let model = ContextModel::new(1);
        let cost_true = model.estimate_bit_cost(0, true);
        let cost_false = model.estimate_bit_cost(0, false);
        assert!(cost_true > 0.0);
        assert!(cost_false > 0.0);
        assert!((cost_true - cost_false).abs() < 0.01); // Should be similar for neutral state
    }

    #[test]
    fn test_context_optimizer_creation() {
        let optimizer = ContextOptimizer::default();
        assert!(optimizer.enable_adaptive);
        assert_eq!(optimizer.models.len(), 1);
    }

    #[test]
    fn test_context_selection() {
        let optimizer = ContextOptimizer::default();
        let neighbors = vec![true, false, true, false];
        let context = optimizer.select_context(&neighbors, 0);
        assert!(context < 256);
    }

    #[test]
    fn test_entropy_calculation() {
        let optimizer = ContextOptimizer::default();

        // All same symbols -> entropy = 0
        let uniform = vec![true; 100];
        let entropy_uniform = optimizer.calculate_entropy(&uniform);
        assert_eq!(entropy_uniform, 0.0);

        // 50/50 split -> entropy = 1
        let mut balanced = vec![true; 50];
        balanced.extend(vec![false; 50]);
        let entropy_balanced = optimizer.calculate_entropy(&balanced);
        assert!((entropy_balanced - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_entropy_stats() {
        let stats = EntropyStats::new(100, 200, 200);
        assert_eq!(stats.total_bits, 100);
        assert_eq!(stats.num_symbols, 200);
        assert_eq!(stats.avg_bits_per_symbol, 0.5);
        assert_eq!(stats.compression_ratio, 2.0);
    }

    #[test]
    fn test_get_stats() {
        let optimizer = ContextOptimizer::default();
        let stats = optimizer.get_stats(100, 200);
        assert_eq!(stats.total_bits, 100);
        assert_eq!(stats.num_symbols, 200);
    }
}
