//! Variance-based adaptive quantization.

/// Variance map for a frame.
#[derive(Debug, Clone)]
pub struct VarianceMap {
    /// Variance values for each block.
    pub variances: Vec<f64>,
    /// Block width.
    pub block_width: usize,
    /// Block height.
    pub block_height: usize,
}

impl VarianceMap {
    /// Creates a new variance map.
    #[must_use]
    pub fn new(block_width: usize, block_height: usize) -> Self {
        Self {
            variances: Vec::new(),
            block_width,
            block_height,
        }
    }
}

/// Variance-based AQ.
pub struct VarianceAq {
    bias_strength: f64,
}

impl Default for VarianceAq {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl VarianceAq {
    /// Creates a new variance AQ.
    #[must_use]
    pub fn new(bias_strength: f64) -> Self {
        Self { bias_strength }
    }

    /// Calculates variance for a block.
    #[must_use]
    pub fn calculate_variance(&self, pixels: &[u8]) -> f64 {
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

    /// Converts variance to QP offset.
    #[must_use]
    pub fn variance_to_qp_offset(&self, variance: f64, strength: f64) -> i8 {
        // Low variance (flat areas) -> positive offset (higher QP, more compression)
        // High variance (textured) -> negative offset (lower QP, preserve detail)

        let normalized_var = (variance / 1000.0).min(1.0);
        let offset = (1.0 - normalized_var) * self.bias_strength * strength * 6.0;

        offset.clamp(-10.0, 10.0) as i8
    }

    /// Builds variance map for a frame.
    #[allow(dead_code)]
    #[must_use]
    pub fn build_variance_map(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
        block_size: usize,
    ) -> VarianceMap {
        let blocks_x = width / block_size;
        let blocks_y = height / block_size;
        let mut variances = Vec::with_capacity(blocks_x * blocks_y);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let mut block = Vec::with_capacity(block_size * block_size);

                for y in 0..block_size {
                    for x in 0..block_size {
                        let px = bx * block_size + x;
                        let py = by * block_size + y;
                        if px < width && py < height {
                            block.push(pixels[py * width + px]);
                        }
                    }
                }

                let variance = self.calculate_variance(&block);
                variances.push(variance);
            }
        }

        VarianceMap {
            variances,
            block_width: blocks_x,
            block_height: blocks_y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variance_aq_creation() {
        let aq = VarianceAq::default();
        assert_eq!(aq.bias_strength, 1.0);
    }

    #[test]
    fn test_variance_calculation_flat() {
        let aq = VarianceAq::default();
        let flat = vec![128u8; 64];
        let variance = aq.calculate_variance(&flat);
        assert_eq!(variance, 0.0);
    }

    #[test]
    fn test_variance_calculation_varied() {
        let aq = VarianceAq::default();
        let varied: Vec<u8> = (0..64).map(|i| i as u8).collect();
        let variance = aq.calculate_variance(&varied);
        assert!(variance > 0.0);
    }

    #[test]
    fn test_variance_to_qp_offset_low_variance() {
        let aq = VarianceAq::default();
        let offset = aq.variance_to_qp_offset(10.0, 1.0);
        assert!(offset > 0); // Low variance -> positive offset
    }

    #[test]
    fn test_variance_to_qp_offset_high_variance() {
        let aq = VarianceAq::default();
        let offset = aq.variance_to_qp_offset(2000.0, 1.0);
        assert!(offset <= 0); // High variance -> negative or zero offset
    }

    #[test]
    fn test_variance_map_creation() {
        let map = VarianceMap::new(10, 8);
        assert_eq!(map.block_width, 10);
        assert_eq!(map.block_height, 8);
        assert!(map.variances.is_empty());
    }

    #[test]
    fn test_build_variance_map() {
        let aq = VarianceAq::default();
        let pixels = vec![128u8; 256]; // 16x16 image
        let map = aq.build_variance_map(&pixels, 16, 16, 8);
        assert_eq!(map.block_width, 2);
        assert_eq!(map.block_height, 2);
        assert_eq!(map.variances.len(), 4);
    }
}
