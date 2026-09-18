//! Zoom optimization to minimize black borders.

use crate::error::{StabilizeError, StabilizeResult};
use crate::transform::calculate::StabilizationTransform;

/// Zoom optimizer.
#[derive(Debug)]
pub struct ZoomOptimizer {
    target_crop: f64,
}

impl ZoomOptimizer {
    /// Create a new zoom optimizer.
    #[must_use]
    pub fn new(target_crop: f64) -> Self {
        Self {
            target_crop: target_crop.clamp(0.0, 1.0),
        }
    }

    /// Optimize zoom for transforms.
    ///
    /// # Errors
    ///
    /// Returns an error if transforms are empty.
    pub fn optimize(
        &self,
        transforms: &[StabilizationTransform],
        _width: usize,
        _height: usize,
    ) -> StabilizeResult<Vec<StabilizationTransform>> {
        if transforms.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        // Calculate optimal zoom
        let max_motion = transforms.iter().map(|t| t.magnitude()).fold(0.0, f64::max);

        let zoom_factor = 1.0 + max_motion * 0.001;

        let optimized: Vec<_> = transforms
            .iter()
            .map(|t| {
                let mut opt = *t;
                opt.scale *= zoom_factor;
                opt
            })
            .collect();

        Ok(optimized)
    }
}

/// Advanced zoom calculation algorithms.
pub mod advanced {
    use crate::transform::calculate::StabilizationTransform;

    /// Automatic zoom level calculator.
    pub struct AutoZoomCalculator {
        min_zoom: f64,
        max_zoom: f64,
        target_fill: f64,
    }

    impl AutoZoomCalculator {
        /// Create a new auto zoom calculator.
        #[must_use]
        pub fn new() -> Self {
            Self {
                min_zoom: 1.0,
                max_zoom: 2.0,
                target_fill: 0.95,
            }
        }

        /// Calculate optimal zoom for frame sequence.
        #[must_use]
        pub fn calculate_zoom(
            &self,
            transforms: &[StabilizationTransform],
            _width: usize,
            _height: usize,
        ) -> Vec<f64> {
            transforms
                .iter()
                .map(|t| {
                    let motion_magnitude = t.magnitude();
                    let base_zoom = 1.0 + motion_magnitude / 100.0;
                    base_zoom.clamp(self.min_zoom, self.max_zoom)
                })
                .collect()
        }

        /// Calculate zoom to maintain minimum fill ratio.
        #[must_use]
        pub fn calculate_fill_zoom(
            &self,
            transforms: &[StabilizationTransform],
            width: usize,
            height: usize,
        ) -> Vec<f64> {
            let max_translation = transforms
                .iter()
                .map(|t| t.dx.abs().max(t.dy.abs()))
                .fold(0.0, f64::max);

            let required_zoom = 1.0 / (1.0 - max_translation / width.min(height) as f64);
            vec![required_zoom.clamp(self.min_zoom, self.max_zoom); transforms.len()]
        }

        /// Smooth zoom transitions.
        #[must_use]
        pub fn smooth_zoom(&self, zoom_levels: &[f64], window_size: usize) -> Vec<f64> {
            let mut smoothed = Vec::with_capacity(zoom_levels.len());
            let half = window_size / 2;

            for i in 0..zoom_levels.len() {
                let start = i.saturating_sub(half);
                let end = (i + half + 1).min(zoom_levels.len());
                let sum: f64 = zoom_levels[start..end].iter().sum();
                let avg = sum / (end - start) as f64;
                smoothed.push(avg);
            }

            smoothed
        }
    }

    impl Default for AutoZoomCalculator {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_auto_zoom() {
            let calculator = AutoZoomCalculator::new();
            let transforms = vec![StabilizationTransform::identity(0); 10];
            let zoom = calculator.calculate_zoom(&transforms, 1920, 1080);
            assert_eq!(zoom.len(), 10);
        }

        #[test]
        fn test_smooth_zoom() {
            let calculator = AutoZoomCalculator::new();
            let zoom = vec![1.0, 1.5, 1.0, 1.5, 1.0];
            let smoothed = calculator.smooth_zoom(&zoom, 3);
            assert_eq!(smoothed.len(), 5);
        }
    }
}
