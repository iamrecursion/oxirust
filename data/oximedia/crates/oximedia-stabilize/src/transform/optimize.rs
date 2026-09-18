//! Transform optimization to minimize cropping and maximize output quality.

use crate::error::{StabilizeError, StabilizeResult};
use crate::transform::calculate::StabilizationTransform;

/// Transform optimizer that minimizes cropping while maintaining stabilization quality.
pub struct TransformOptimizer {
    target_crop_ratio: f64,
    optimization_iterations: usize,
}

impl TransformOptimizer {
    /// Create a new transform optimizer.
    #[must_use]
    pub fn new(target_crop_ratio: f64) -> Self {
        Self {
            target_crop_ratio: target_crop_ratio.clamp(0.0, 1.0),
            optimization_iterations: 10,
        }
    }

    /// Optimize transforms to minimize cropping.
    ///
    /// # Errors
    ///
    /// Returns an error if the transforms vector is empty.
    pub fn optimize(
        &self,
        transforms: &[StabilizationTransform],
        width: usize,
        height: usize,
    ) -> StabilizeResult<Vec<StabilizationTransform>> {
        if transforms.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        let mut optimized = transforms.to_vec();

        for _ in 0..self.optimization_iterations {
            optimized = self.optimize_iteration(&optimized, width, height)?;
        }

        Ok(optimized)
    }

    /// Single optimization iteration.
    fn optimize_iteration(
        &self,
        transforms: &[StabilizationTransform],
        width: usize,
        height: usize,
    ) -> StabilizeResult<Vec<StabilizationTransform>> {
        // Calculate maximum allowed motion based on crop ratio
        let max_dx = width as f64 * (1.0 - self.target_crop_ratio) / 2.0;
        let max_dy = height as f64 * (1.0 - self.target_crop_ratio) / 2.0;

        let optimized: Vec<_> = transforms
            .iter()
            .map(|t| {
                let mut opt = *t;

                // Limit translation to avoid excessive cropping
                opt.dx = opt.dx.clamp(-max_dx, max_dx);
                opt.dy = opt.dy.clamp(-max_dy, max_dy);

                // Limit rotation to avoid corner cropping
                let max_angle = 0.1; // ~5.7 degrees
                opt.angle = opt.angle.clamp(-max_angle, max_angle);

                opt
            })
            .collect();

        Ok(optimized)
    }

    /// Calculate required crop for transforms.
    #[must_use]
    pub fn calculate_crop_bounds(
        &self,
        transforms: &[StabilizationTransform],
        width: usize,
        height: usize,
    ) -> CropBounds {
        let mut max_dx = 0.0_f64;
        let mut max_dy = 0.0_f64;

        for transform in transforms {
            max_dx = max_dx.max(transform.dx.abs());
            max_dy = max_dy.max(transform.dy.abs());
        }

        let crop_x = (max_dx / width as f64).min(0.5);
        let crop_y = (max_dy / height as f64).min(0.5);

        CropBounds {
            left: crop_x,
            right: crop_x,
            top: crop_y,
            bottom: crop_y,
            total_crop_ratio: 1.0 - (1.0 - 2.0 * crop_x) * (1.0 - 2.0 * crop_y),
        }
    }
}

/// Crop bounds for stabilized video.
#[derive(Debug, Clone, Copy)]
pub struct CropBounds {
    /// Left crop ratio
    pub left: f64,
    /// Right crop ratio
    pub right: f64,
    /// Top crop ratio
    pub top: f64,
    /// Bottom crop ratio
    pub bottom: f64,
    /// Total area crop ratio
    pub total_crop_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let optimizer = TransformOptimizer::new(0.9);
        assert!((optimizer.target_crop_ratio - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_crop_bounds() {
        let optimizer = TransformOptimizer::new(0.9);
        let transforms = vec![
            StabilizationTransform::new(10.0, 5.0, 0.0, 1.0, 0),
            StabilizationTransform::new(15.0, 8.0, 0.0, 1.0, 1),
        ];

        let bounds = optimizer.calculate_crop_bounds(&transforms, 100, 100);
        assert!(bounds.total_crop_ratio >= 0.0);
        assert!(bounds.total_crop_ratio <= 1.0);
    }
}
