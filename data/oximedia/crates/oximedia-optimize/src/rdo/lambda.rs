//! Lambda calculation for rate-distortion optimization.

use crate::OptimizationLevel;

/// Lambda calculator for RDO.
pub struct LambdaCalculator {
    multiplier: f64,
    level: OptimizationLevel,
}

impl LambdaCalculator {
    /// Creates a new lambda calculator.
    #[must_use]
    pub fn new(multiplier: f64, level: OptimizationLevel) -> Self {
        Self { multiplier, level }
    }

    /// Calculates lambda for a given QP.
    ///
    /// Uses the formula: lambda = multiplier * 0.85 * 2^((QP-12)/3)
    #[must_use]
    pub fn calculate(&self, qp: u8) -> f64 {
        let base_lambda = 0.85 * 2_f64.powf(f64::from(qp - 12) / 3.0);
        let level_multiplier = self.level_multiplier();
        self.multiplier * level_multiplier * base_lambda
    }

    /// Calculates lambda for motion estimation.
    ///
    /// Motion lambda is typically sqrt(RDO lambda).
    #[must_use]
    pub fn calculate_motion_lambda(&self, qp: u8) -> f64 {
        self.calculate(qp).sqrt()
    }

    fn level_multiplier(&self) -> f64 {
        match self.level {
            OptimizationLevel::Fast => 0.8,
            OptimizationLevel::Medium => 1.0,
            OptimizationLevel::Slow => 1.1,
            OptimizationLevel::Placebo => 1.2,
        }
    }
}

/// Parameters for lambda calculation.
#[derive(Debug, Clone, Copy)]
pub struct LambdaParams {
    /// Base QP value.
    pub qp: u8,
    /// Frame type multiplier (I/P/B).
    pub frame_type_multiplier: f64,
    /// Temporal layer multiplier.
    pub temporal_multiplier: f64,
}

impl Default for LambdaParams {
    fn default() -> Self {
        Self {
            qp: 26,
            frame_type_multiplier: 1.0,
            temporal_multiplier: 1.0,
        }
    }
}

impl LambdaParams {
    /// Creates parameters for an I-frame.
    #[must_use]
    pub fn for_i_frame(qp: u8) -> Self {
        Self {
            qp,
            frame_type_multiplier: 0.57,
            temporal_multiplier: 1.0,
        }
    }

    /// Creates parameters for a P-frame.
    #[must_use]
    pub fn for_p_frame(qp: u8) -> Self {
        Self {
            qp,
            frame_type_multiplier: 0.68,
            temporal_multiplier: 1.0,
        }
    }

    /// Creates parameters for a B-frame.
    #[must_use]
    pub fn for_b_frame(qp: u8, temporal_layer: u8) -> Self {
        let temporal_multiplier = 1.0 + 0.05 * f64::from(temporal_layer);
        Self {
            qp,
            frame_type_multiplier: 0.68,
            temporal_multiplier,
        }
    }

    /// Calculates the effective lambda.
    #[must_use]
    pub fn effective_lambda(&self, calculator: &LambdaCalculator) -> f64 {
        calculator.calculate(self.qp) * self.frame_type_multiplier * self.temporal_multiplier
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lambda_calculation() {
        let calc = LambdaCalculator::new(1.0, OptimizationLevel::Medium);
        let lambda = calc.calculate(26);
        assert!(lambda > 0.0);
        assert!(lambda < 100.0); // Reasonable range
    }

    #[test]
    fn test_lambda_increases_with_qp() {
        let calc = LambdaCalculator::new(1.0, OptimizationLevel::Medium);
        let lambda_low = calc.calculate(20);
        let lambda_high = calc.calculate(30);
        assert!(lambda_high > lambda_low);
    }

    #[test]
    fn test_motion_lambda() {
        let calc = LambdaCalculator::new(1.0, OptimizationLevel::Medium);
        let lambda = calc.calculate(26);
        let motion_lambda = calc.calculate_motion_lambda(26);
        assert!((motion_lambda - lambda.sqrt()).abs() < 0.001);
    }

    #[test]
    fn test_frame_type_params() {
        let i_params = LambdaParams::for_i_frame(26);
        let p_params = LambdaParams::for_p_frame(26);
        let b_params = LambdaParams::for_b_frame(26, 0);

        assert!(i_params.frame_type_multiplier < p_params.frame_type_multiplier);
        assert_eq!(b_params.temporal_multiplier, 1.0);
    }
}
