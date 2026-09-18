//! Transform interpolation for temporal smoothness.

use crate::error::{StabilizeError, StabilizeResult};
use crate::transform::calculate::StabilizationTransform;

/// Transform interpolator for smooth transitions.
pub struct TransformInterpolator;

impl TransformInterpolator {
    /// Create a new transform interpolator.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Interpolate between two transforms.
    #[must_use]
    pub fn interpolate(
        &self,
        t1: &StabilizationTransform,
        t2: &StabilizationTransform,
        alpha: f64,
    ) -> StabilizationTransform {
        let alpha = alpha.clamp(0.0, 1.0);
        let beta = 1.0 - alpha;

        StabilizationTransform {
            dx: beta * t1.dx + alpha * t2.dx,
            dy: beta * t1.dy + alpha * t2.dy,
            angle: beta * t1.angle + alpha * t2.angle,
            scale: beta * t1.scale + alpha * t2.scale,
            frame_index: t1.frame_index,
            confidence: beta * t1.confidence + alpha * t2.confidence,
        }
    }

    /// Upsample transforms for frame rate conversion.
    ///
    /// # Errors
    ///
    /// Returns an error if transforms is empty or factor is less than 1.
    pub fn upsample(
        &self,
        transforms: &[StabilizationTransform],
        factor: usize,
    ) -> StabilizeResult<Vec<StabilizationTransform>> {
        if transforms.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        if factor < 1 {
            return Err(StabilizeError::invalid_parameter(
                "factor",
                factor.to_string(),
            ));
        }

        let mut upsampled = Vec::with_capacity(transforms.len() * factor);

        for i in 0..transforms.len() - 1 {
            for j in 0..factor {
                let alpha = j as f64 / factor as f64;
                let interpolated = self.interpolate(&transforms[i], &transforms[i + 1], alpha);
                upsampled.push(interpolated);
            }
        }

        // Add last transform
        upsampled.push(transforms[transforms.len() - 1]);

        Ok(upsampled)
    }
}

impl Default for TransformInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolator_creation() {
        let _interpolator = TransformInterpolator::new();
    }

    #[test]
    fn test_interpolation() {
        let interpolator = TransformInterpolator::new();
        let t1 = StabilizationTransform::new(0.0, 0.0, 0.0, 1.0, 0);
        let t2 = StabilizationTransform::new(10.0, 10.0, 0.0, 1.0, 1);

        let mid = interpolator.interpolate(&t1, &t2, 0.5);
        assert!((mid.dx - 5.0).abs() < 1e-10);
        assert!((mid.dy - 5.0).abs() < 1e-10);
    }
}
