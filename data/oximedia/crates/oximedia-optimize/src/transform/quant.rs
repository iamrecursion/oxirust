//! Adaptive quantization optimization.

/// Adaptive QP calculator.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveQp {
    /// Base QP.
    pub base_qp: u8,
    /// QP offset.
    pub offset: i8,
    /// Effective QP.
    pub effective_qp: u8,
}

impl AdaptiveQp {
    /// Creates a new adaptive QP.
    #[must_use]
    pub fn new(base_qp: u8, offset: i8) -> Self {
        let effective_qp = (i16::from(base_qp) + i16::from(offset)).clamp(0, 63) as u8;

        Self {
            base_qp,
            offset,
            effective_qp,
        }
    }
}

/// Quantization optimizer.
pub struct QuantizationOptimizer {
    enable_trellis: bool,
    deadzone_offset: f64,
}

impl Default for QuantizationOptimizer {
    fn default() -> Self {
        Self::new(false, 0.0)
    }
}

impl QuantizationOptimizer {
    /// Creates a new quantization optimizer.
    #[must_use]
    pub fn new(enable_trellis: bool, deadzone_offset: f64) -> Self {
        Self {
            enable_trellis,
            deadzone_offset,
        }
    }

    /// Quantizes coefficients with optional trellis optimization.
    #[allow(dead_code)]
    #[must_use]
    pub fn quantize(&self, coeffs: &[i16], qp: u8) -> Vec<i16> {
        if self.enable_trellis {
            self.trellis_quantize(coeffs, qp)
        } else {
            self.simple_quantize(coeffs, qp)
        }
    }

    fn simple_quantize(&self, coeffs: &[i16], qp: u8) -> Vec<i16> {
        let scale = self.qp_to_scale(qp);
        let deadzone = (f64::from(scale) * self.deadzone_offset) as i16;

        coeffs
            .iter()
            .map(|&c| if c.abs() < deadzone { 0 } else { c / scale })
            .collect()
    }

    fn trellis_quantize(&self, coeffs: &[i16], qp: u8) -> Vec<i16> {
        // Simplified trellis quantization
        // Real implementation would use dynamic programming
        let scale = self.qp_to_scale(qp);
        let mut quantized = vec![0i16; coeffs.len()];

        for (i, &c) in coeffs.iter().enumerate() {
            let q = c / scale;

            // Try q, q-1, q+1 and pick best
            let candidates = [q.saturating_sub(1), q, q.saturating_add(1)];
            let mut best_q = q;
            let mut best_cost = f64::MAX;

            for &candidate in &candidates {
                let reconstructed = candidate * scale;
                let distortion = (c - reconstructed).abs();
                let rate = if candidate == 0 { 0 } else { candidate.abs() };
                let cost = f64::from(distortion) + f64::from(rate);

                if cost < best_cost {
                    best_cost = cost;
                    best_q = candidate;
                }
            }

            quantized[i] = best_q;
        }

        quantized
    }

    fn qp_to_scale(&self, qp: u8) -> i16 {
        // Simplified QP to scale conversion
        // Real implementation would use proper QP tables
        (1 << (qp / 6)).max(1)
    }

    /// Calculates optimal deadzone for a block.
    #[allow(dead_code)]
    #[must_use]
    pub fn calculate_deadzone(&self, variance: f64) -> f64 {
        // Larger deadzone for low variance (flat areas)
        if variance < 100.0 {
            1.5
        } else if variance < 500.0 {
            1.0
        } else {
            0.5
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_qp() {
        let qp = AdaptiveQp::new(26, 3);
        assert_eq!(qp.base_qp, 26);
        assert_eq!(qp.offset, 3);
        assert_eq!(qp.effective_qp, 29);
    }

    #[test]
    fn test_adaptive_qp_clamping() {
        let qp_high = AdaptiveQp::new(60, 10);
        assert_eq!(qp_high.effective_qp, 63); // Clamped

        let qp_low = AdaptiveQp::new(5, -10);
        assert_eq!(qp_low.effective_qp, 0); // Clamped
    }

    #[test]
    fn test_quantization_optimizer_creation() {
        let optimizer = QuantizationOptimizer::default();
        assert!(!optimizer.enable_trellis);
        assert_eq!(optimizer.deadzone_offset, 0.0);
    }

    #[test]
    fn test_simple_quantize() {
        let optimizer = QuantizationOptimizer::default();
        let coeffs = vec![100, 50, 25, 10, 5];
        let quantized = optimizer.simple_quantize(&coeffs, 12);
        assert!(quantized.iter().all(|&q| q <= 100));
    }

    #[test]
    fn test_deadzone_calculation() {
        let optimizer = QuantizationOptimizer::default();
        let dz_low = optimizer.calculate_deadzone(50.0);
        let dz_high = optimizer.calculate_deadzone(1000.0);
        assert!(dz_low > dz_high); // Larger deadzone for low variance
    }
}
