//! Cost estimation functions for RDO.

/// Cost metrics for distortion measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostMetric {
    /// Sum of Absolute Differences.
    Sad,
    /// Sum of Absolute Transformed Differences (Hadamard).
    Satd,
    /// Sum of Squared Errors.
    Sse,
    /// Structural Similarity Index.
    Ssim,
}

/// Cost estimate combining distortion and rate.
#[derive(Debug, Clone, Copy)]
pub struct CostEstimate {
    /// Distortion value.
    pub distortion: f64,
    /// Rate in bits.
    pub rate: f64,
    /// Combined RD cost.
    pub cost: f64,
    /// Metric used for distortion.
    pub metric: CostMetric,
}

impl CostEstimate {
    /// Creates a new cost estimate.
    #[must_use]
    pub fn new(distortion: f64, rate: f64, lambda: f64, metric: CostMetric) -> Self {
        let cost = distortion + lambda * rate;
        Self {
            distortion,
            rate,
            cost,
            metric,
        }
    }

    /// Checks if this cost is better than another.
    #[must_use]
    pub fn is_better_than(&self, other: &Self) -> bool {
        self.cost < other.cost
    }
}

/// Bit cost estimation for mode signaling.
pub struct BitCost {
    /// Cost in bits for signaling each mode.
    mode_costs: Vec<f64>,
}

impl BitCost {
    /// Creates a new bit cost estimator.
    #[must_use]
    pub fn new(num_modes: usize) -> Self {
        // Uniform distribution initially
        let cost = (num_modes as f64).log2();
        Self {
            mode_costs: vec![cost; num_modes],
        }
    }

    /// Updates costs based on probability distribution.
    pub fn update_from_probabilities(&mut self, probabilities: &[f64]) {
        assert_eq!(probabilities.len(), self.mode_costs.len());
        for (cost, &prob) in self.mode_costs.iter_mut().zip(probabilities) {
            *cost = if prob > 0.0 {
                -prob.log2()
            } else {
                1000.0 // Very high cost for impossible mode
            };
        }
    }

    /// Gets the bit cost for a specific mode.
    #[must_use]
    pub fn get_cost(&self, mode: usize) -> f64 {
        self.mode_costs.get(mode).copied().unwrap_or(1000.0)
    }
}

/// Calculates SAD (Sum of Absolute Differences).
#[must_use]
pub fn calculate_sad(src: &[u8], dst: &[u8]) -> u32 {
    assert_eq!(src.len(), dst.len());
    src.iter()
        .zip(dst)
        .map(|(&s, &d)| u32::from(s.abs_diff(d)))
        .sum()
}

/// Calculates SSE (Sum of Squared Errors).
#[must_use]
pub fn calculate_sse(src: &[u8], dst: &[u8]) -> u64 {
    assert_eq!(src.len(), dst.len());
    src.iter()
        .zip(dst)
        .map(|(&s, &d)| {
            let diff = i32::from(s) - i32::from(d);
            (diff * diff) as u64
        })
        .sum()
}

/// Calculates SATD (Sum of Absolute Transformed Differences) using Hadamard transform.
///
/// For 4x4 blocks only.
#[must_use]
pub fn calculate_satd_4x4(src: &[u8], dst: &[u8]) -> u32 {
    assert_eq!(src.len(), 16);
    assert_eq!(dst.len(), 16);

    // Calculate differences
    let mut diff = [0i16; 16];
    for ((&s, &d), diff_val) in src.iter().zip(dst).zip(&mut diff) {
        *diff_val = i16::from(s) - i16::from(d);
    }

    // Apply 4x4 Hadamard transform
    hadamard_4x4(&mut diff);

    // Sum absolute values
    diff.iter().map(|&x| u32::from(x.unsigned_abs())).sum()
}

/// Applies 4x4 Hadamard transform in-place.
fn hadamard_4x4(block: &mut [i16; 16]) {
    // Horizontal transform
    for row in 0..4 {
        let i = row * 4;
        let a0 = block[i] + block[i + 2];
        let a1 = block[i + 1] + block[i + 3];
        let a2 = block[i] - block[i + 2];
        let a3 = block[i + 1] - block[i + 3];

        block[i] = a0 + a1;
        block[i + 1] = a2 + a3;
        block[i + 2] = a0 - a1;
        block[i + 3] = a2 - a3;
    }

    // Vertical transform
    for col in 0..4 {
        let a0 = block[col] + block[col + 8];
        let a1 = block[col + 4] + block[col + 12];
        let a2 = block[col] - block[col + 8];
        let a3 = block[col + 4] - block[col + 12];

        block[col] = a0 + a1;
        block[col + 4] = a2 + a3;
        block[col + 8] = a0 - a1;
        block[col + 12] = a2 - a3;
    }
}

/// Calculates PSNR (Peak Signal-to-Noise Ratio) from SSE.
#[must_use]
pub fn calculate_psnr(sse: f64, num_pixels: usize) -> f64 {
    if sse == 0.0 {
        return f64::INFINITY;
    }
    let mse = sse / num_pixels as f64;
    let max_val = 255.0;
    20.0 * (max_val / mse.sqrt()).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sad_identical() {
        let block = vec![100u8; 16];
        assert_eq!(calculate_sad(&block, &block), 0);
    }

    #[test]
    fn test_sad_different() {
        let src = vec![100u8; 16];
        let dst = vec![110u8; 16];
        assert_eq!(calculate_sad(&src, &dst), 160); // 16 * 10
    }

    #[test]
    fn test_sse_identical() {
        let block = vec![100u8; 16];
        assert_eq!(calculate_sse(&block, &block), 0);
    }

    #[test]
    fn test_sse_different() {
        let src = vec![100u8; 16];
        let dst = vec![110u8; 16];
        assert_eq!(calculate_sse(&src, &dst), 1600); // 16 * 10^2
    }

    #[test]
    fn test_satd_identical() {
        let block = vec![100u8; 16];
        assert_eq!(calculate_satd_4x4(&block, &block), 0);
    }

    #[test]
    fn test_psnr_zero_error() {
        let psnr = calculate_psnr(0.0, 100);
        assert!(psnr.is_infinite());
    }

    #[test]
    fn test_psnr_some_error() {
        let psnr = calculate_psnr(100.0, 100);
        assert!(psnr > 0.0 && psnr < 100.0);
    }

    #[test]
    fn test_cost_estimate() {
        let cost = CostEstimate::new(100.0, 50.0, 2.0, CostMetric::Sad);
        assert_eq!(cost.cost, 200.0); // 100 + 2*50
        assert_eq!(cost.distortion, 100.0);
        assert_eq!(cost.rate, 50.0);
    }

    #[test]
    fn test_bit_cost() {
        let mut bit_cost = BitCost::new(4);
        assert!((bit_cost.get_cost(0) - 2.0).abs() < 0.001); // log2(4) = 2

        bit_cost.update_from_probabilities(&[0.5, 0.25, 0.125, 0.125]);
        assert!((bit_cost.get_cost(0) - 1.0).abs() < 0.001); // -log2(0.5) = 1
        assert!((bit_cost.get_cost(1) - 2.0).abs() < 0.001); // -log2(0.25) = 2
    }
}
