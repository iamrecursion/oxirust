//! Bidirectional prediction optimization.

use super::MotionVector;

/// Bidirectional prediction optimizer.
pub struct BidirectionalOptimizer {
    enable_weighted_prediction: bool,
}

impl Default for BidirectionalOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl BidirectionalOptimizer {
    /// Creates a new bidirectional optimizer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            enable_weighted_prediction: true,
        }
    }

    /// Optimizes bidirectional prediction.
    #[allow(dead_code)]
    #[must_use]
    pub fn optimize(
        &self,
        src: &[u8],
        ref_fwd: &[u8],
        ref_bwd: &[u8],
        mv_fwd: MotionVector,
        mv_bwd: MotionVector,
    ) -> BiPredResult {
        let cost_fwd = self.calculate_uni_pred_cost(src, ref_fwd);
        let cost_bwd = self.calculate_uni_pred_cost(src, ref_bwd);
        let cost_bi = self.calculate_bi_pred_cost(src, ref_fwd, ref_bwd);

        let (mode, cost) = if cost_bi < cost_fwd && cost_bi < cost_bwd {
            (BiPredMode::Bidirectional, cost_bi)
        } else if cost_fwd < cost_bwd {
            (BiPredMode::ForwardOnly, cost_fwd)
        } else {
            (BiPredMode::BackwardOnly, cost_bwd)
        };

        BiPredResult {
            mode,
            mv_fwd,
            mv_bwd,
            cost,
            weight_fwd: 0.5,
            weight_bwd: 0.5,
        }
    }

    fn calculate_uni_pred_cost(&self, src: &[u8], reference: &[u8]) -> f64 {
        // Simplified cost (would use actual SAD/SATD)
        src.iter()
            .zip(reference)
            .map(|(&s, &r)| f64::from(s.abs_diff(r)))
            .sum()
    }

    fn calculate_bi_pred_cost(&self, src: &[u8], ref_fwd: &[u8], ref_bwd: &[u8]) -> f64 {
        // Average the two references and compare with source
        src.iter()
            .zip(ref_fwd.iter().zip(ref_bwd))
            .map(|(&s, (&f, &b))| {
                let avg = (u16::from(f) + u16::from(b)).div_ceil(2) as u8;
                f64::from(s.abs_diff(avg))
            })
            .sum()
    }

    /// Calculates optimal weights for weighted prediction.
    #[allow(dead_code)]
    #[must_use]
    pub fn calculate_weights(&self, ref_fwd: &[u8], ref_bwd: &[u8], _src: &[u8]) -> (f64, f64) {
        if !self.enable_weighted_prediction {
            return (0.5, 0.5);
        }

        // Simple variance-based weighting
        let var_fwd = self.calculate_variance(ref_fwd);
        let var_bwd = self.calculate_variance(ref_bwd);
        let total_var = var_fwd + var_bwd;

        if total_var > 0.0 {
            let weight_bwd = var_fwd / total_var;
            let weight_fwd = 1.0 - weight_bwd;
            (weight_fwd, weight_bwd)
        } else {
            (0.5, 0.5)
        }
    }

    fn calculate_variance(&self, pixels: &[u8]) -> f64 {
        if pixels.is_empty() {
            return 0.0;
        }

        let mean = pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / pixels.len() as f64;
        pixels
            .iter()
            .map(|&p| {
                let diff = f64::from(p) - mean;
                diff * diff
            })
            .sum::<f64>()
            / pixels.len() as f64
    }
}

/// Bidirectional prediction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiPredMode {
    /// Use forward reference only.
    ForwardOnly,
    /// Use backward reference only.
    BackwardOnly,
    /// Use both references.
    Bidirectional,
}

/// Bidirectional prediction result.
#[derive(Debug, Clone, Copy)]
pub struct BiPredResult {
    /// Selected prediction mode.
    pub mode: BiPredMode,
    /// Forward motion vector.
    pub mv_fwd: MotionVector,
    /// Backward motion vector.
    pub mv_bwd: MotionVector,
    /// Total cost.
    pub cost: f64,
    /// Forward reference weight.
    pub weight_fwd: f64,
    /// Backward reference weight.
    pub weight_bwd: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bidirectional_optimizer_creation() {
        let optimizer = BidirectionalOptimizer::new();
        assert!(optimizer.enable_weighted_prediction);
    }

    #[test]
    fn test_variance_calculation() {
        let optimizer = BidirectionalOptimizer::new();
        let flat = vec![128u8; 64];
        let var_flat = optimizer.calculate_variance(&flat);
        assert_eq!(var_flat, 0.0);

        let varied: Vec<u8> = (0..64).map(|i| i as u8).collect();
        let var_varied = optimizer.calculate_variance(&varied);
        assert!(var_varied > 0.0);
    }

    #[test]
    fn test_weight_calculation() {
        let optimizer = BidirectionalOptimizer::new();
        let ref_fwd = vec![100u8; 64];
        let ref_bwd = vec![150u8; 64];
        let src = vec![125u8; 64];

        let (w_fwd, w_bwd) = optimizer.calculate_weights(&ref_fwd, &ref_bwd, &src);
        assert!((w_fwd + w_bwd - 1.0).abs() < 0.001);
        assert!(w_fwd >= 0.0 && w_fwd <= 1.0);
        assert!(w_bwd >= 0.0 && w_bwd <= 1.0);
    }
}
