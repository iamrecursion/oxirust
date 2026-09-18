//! Video stabilization using registration results.
//!
//! Applies motion smoothing to registration transforms for stable video output.

use super::TransformMatrix;

/// Motion smoothing filter for stabilization.
#[derive(Debug, Clone)]
pub struct MotionSmoother {
    /// Smoothing window radius (frames).
    window_radius: usize,
    /// Transform history.
    transforms: Vec<TransformMatrix>,
}

impl MotionSmoother {
    /// Create a new motion smoother.
    #[must_use]
    pub fn new(window_radius: usize) -> Self {
        Self {
            window_radius: window_radius.max(1),
            transforms: Vec::new(),
        }
    }

    /// Add a transform to the history.
    pub fn add_transform(&mut self, transform: TransformMatrix) {
        self.transforms.push(transform);
    }

    /// Get the smoothed transform for a given frame index.
    #[must_use]
    pub fn get_smoothed(&self, frame_idx: usize) -> TransformMatrix {
        if self.transforms.is_empty() {
            return TransformMatrix::identity();
        }

        let start = frame_idx.saturating_sub(self.window_radius);
        let end = (frame_idx + self.window_radius + 1).min(self.transforms.len());

        if start >= end {
            return self.transforms[frame_idx.min(self.transforms.len() - 1)].clone();
        }

        // Average the translations within the window
        let mut avg_tx = 0.0;
        let mut avg_ty = 0.0;
        let count = (end - start) as f64;

        for t in &self.transforms[start..end] {
            let (tx, ty) = t.get_translation();
            avg_tx += tx;
            avg_ty += ty;
        }

        TransformMatrix::translation(avg_tx / count, avg_ty / count)
    }

    /// Get the number of stored transforms.
    #[must_use]
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }

    /// Clear history.
    pub fn clear(&mut self) {
        self.transforms.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_smoother_new() {
        let smoother = MotionSmoother::new(5);
        assert!(smoother.is_empty());
        assert_eq!(smoother.len(), 0);
    }

    #[test]
    fn test_motion_smoother_add_and_get() {
        let mut smoother = MotionSmoother::new(2);
        smoother.add_transform(TransformMatrix::translation(10.0, 0.0));
        smoother.add_transform(TransformMatrix::translation(20.0, 0.0));
        smoother.add_transform(TransformMatrix::translation(30.0, 0.0));

        let smoothed = smoother.get_smoothed(1);
        let (tx, _ty) = smoothed.get_translation();
        assert!(tx > 5.0 && tx < 35.0);
    }

    #[test]
    fn test_motion_smoother_empty() {
        let smoother = MotionSmoother::new(2);
        let result = smoother.get_smoothed(0);
        let (tx, ty) = result.get_translation();
        assert!((tx).abs() < 1e-6);
        assert!((ty).abs() < 1e-6);
    }

    #[test]
    fn test_motion_smoother_clear() {
        let mut smoother = MotionSmoother::new(2);
        smoother.add_transform(TransformMatrix::translation(10.0, 0.0));
        assert_eq!(smoother.len(), 1);
        smoother.clear();
        assert!(smoother.is_empty());
    }
}
