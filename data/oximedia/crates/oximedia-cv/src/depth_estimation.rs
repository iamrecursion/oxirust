#![allow(dead_code)]
//! Stereo depth estimation from disparity maps.
//!
//! Provides disparity map access, depth conversion, and reliability filtering.

/// A disparity map produced by a stereo matching algorithm.
///
/// Disparity values represent the horizontal pixel offset between matching
/// points in the left and right rectified stereo images.
#[derive(Debug, Clone)]
pub struct DisparityMap {
    /// Disparity values in row-major order (pixels)
    data: Vec<f32>,
    /// Image width
    width: usize,
    /// Image height
    height: usize,
    /// Minimum valid disparity (invalid pixels are stored as this value or lower)
    min_valid: f32,
}

impl DisparityMap {
    /// Create a disparity map from a flat buffer.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != width * height`.
    pub fn new(data: Vec<f32>, width: usize, height: usize) -> Self {
        assert_eq!(
            data.len(),
            width * height,
            "data length must equal width × height"
        );
        Self {
            data,
            width,
            height,
            min_valid: 1.0,
        }
    }

    /// Create a disparity map filled with a constant disparity value.
    pub fn constant(width: usize, height: usize, value: f32) -> Self {
        Self {
            data: vec![value; width * height],
            width,
            height,
            min_valid: 1.0,
        }
    }

    /// Get the disparity at pixel `(x, y)`, or `None` if out of bounds.
    pub fn at(&self, x: usize, y: usize) -> Option<f32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(self.data[y * self.width + x])
    }

    /// Set the minimum valid disparity threshold.
    #[must_use]
    pub fn set_min_valid(mut self, min: f32) -> Self {
        self.min_valid = min;
        self
    }

    /// Returns `true` if the disparity at `(x, y)` is considered valid.
    pub fn is_valid_at(&self, x: usize, y: usize) -> bool {
        self.at(x, y).is_some_and(|d| d >= self.min_valid)
    }

    /// Compute the average disparity over all valid pixels.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_disparity(&self) -> f32 {
        let valid: Vec<f32> = self
            .data
            .iter()
            .copied()
            .filter(|&d| d >= self.min_valid)
            .collect();
        if valid.is_empty() {
            return 0.0;
        }
        valid.iter().sum::<f32>() / valid.len() as f32
    }

    /// Compute the maximum disparity over all valid pixels.
    pub fn max_disparity(&self) -> f32 {
        self.data
            .iter()
            .copied()
            .filter(|&d| d >= self.min_valid)
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Count valid (non-occluded) pixels.
    pub fn valid_pixel_count(&self) -> usize {
        self.data.iter().filter(|&&d| d >= self.min_valid).count()
    }

    /// Image dimensions.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Total pixel count.
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }
}

/// A depth estimate for a single pixel.
#[derive(Debug, Clone, Copy)]
pub struct DepthEstimate {
    /// Estimated depth in world units (e.g., metres)
    pub depth_m: f32,
    /// Confidence in the estimate (0.0–1.0)
    pub confidence: f32,
    /// Disparity value that produced this estimate
    pub disparity: f32,
}

impl DepthEstimate {
    /// Create a new depth estimate.
    pub fn new(depth_m: f32, confidence: f32, disparity: f32) -> Self {
        Self {
            depth_m,
            confidence,
            disparity,
        }
    }

    /// Returns `true` if the confidence exceeds the given threshold.
    pub fn is_reliable(&self) -> bool {
        self.confidence >= 0.5
    }

    /// Returns `true` if the confidence exceeds a custom threshold.
    pub fn is_reliable_at(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }

    /// Convert depth to centimetres.
    pub fn depth_cm(&self) -> f32 {
        self.depth_m * 100.0
    }
}

/// Camera stereo parameters for depth conversion.
#[derive(Debug, Clone)]
pub struct StereoParams {
    /// Baseline distance between the two cameras in metres
    pub baseline_m: f32,
    /// Horizontal focal length in pixels
    pub focal_px: f32,
}

impl StereoParams {
    /// Create stereo parameters.
    pub fn new(baseline_m: f32, focal_px: f32) -> Self {
        Self {
            baseline_m,
            focal_px,
        }
    }

    /// Convert a disparity value to depth in metres.
    ///
    /// depth = (baseline × focal) / disparity
    pub fn disparity_to_depth(&self, disparity: f32) -> f32 {
        if disparity < 1e-6 {
            return f32::INFINITY;
        }
        (self.baseline_m * self.focal_px) / disparity
    }
}

/// Stereo depth estimator that converts a `DisparityMap` into per-pixel depth estimates.
#[derive(Debug)]
pub struct DepthEstimator {
    params: StereoParams,
    /// Maximum reliable depth in metres (estimates beyond this get low confidence)
    max_depth_m: f32,
}

impl DepthEstimator {
    /// Create a depth estimator with the given stereo parameters.
    pub fn new(params: StereoParams) -> Self {
        Self {
            params,
            max_depth_m: 50.0,
        }
    }

    /// Set the maximum reliable depth.
    #[must_use]
    pub fn with_max_depth(mut self, max_m: f32) -> Self {
        self.max_depth_m = max_m;
        self
    }

    /// Estimate depth for every pixel in the disparity map.
    ///
    /// Returns a flat vector of `DepthEstimate` values in row-major order.
    pub fn estimate_from_stereo(&self, map: &DisparityMap) -> Vec<DepthEstimate> {
        map.data
            .iter()
            .map(|&disp| self.estimate_single(disp, map.min_valid))
            .collect()
    }

    /// Estimate depth for a specific pixel location.
    pub fn estimate_at(&self, map: &DisparityMap, x: usize, y: usize) -> Option<DepthEstimate> {
        let disp = map.at(x, y)?;
        Some(self.estimate_single(disp, map.min_valid))
    }

    /// Compute average depth over the valid region.
    #[allow(clippy::cast_precision_loss)]
    pub fn average_depth(&self, map: &DisparityMap) -> f32 {
        let estimates = self.estimate_from_stereo(map);
        let reliable: Vec<f32> = estimates
            .iter()
            .filter(|e| e.is_reliable() && e.depth_m.is_finite())
            .map(|e| e.depth_m)
            .collect();
        if reliable.is_empty() {
            return 0.0;
        }
        reliable.iter().sum::<f32>() / reliable.len() as f32
    }

    fn estimate_single(&self, disp: f32, min_valid: f32) -> DepthEstimate {
        if disp < min_valid {
            return DepthEstimate::new(f32::INFINITY, 0.0, disp);
        }
        let depth = self.params.disparity_to_depth(disp);
        let confidence = if depth.is_infinite() || depth > self.max_depth_m {
            0.1
        } else {
            // Confidence falls off linearly with depth
            1.0 - (depth / self.max_depth_m).min(1.0)
        };
        DepthEstimate::new(depth, confidence, disp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_map_4x4(value: f32) -> DisparityMap {
        DisparityMap::constant(4, 4, value)
    }

    #[test]
    fn test_disparity_map_at_valid() {
        let m = make_map_4x4(5.0);
        assert_eq!(m.at(0, 0), Some(5.0));
        assert_eq!(m.at(3, 3), Some(5.0));
    }

    #[test]
    fn test_disparity_map_at_out_of_bounds() {
        let m = make_map_4x4(5.0);
        assert!(m.at(4, 0).is_none());
        assert!(m.at(0, 4).is_none());
    }

    #[test]
    fn test_disparity_map_avg() {
        let m = make_map_4x4(10.0);
        assert!((m.avg_disparity() - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_disparity_map_avg_with_invalids() {
        let mut data = vec![0.0f32; 4]; // first 3 invalid (< min_valid=1)
        data[3] = 8.0;
        let m = DisparityMap::new(data, 4, 1);
        assert!((m.avg_disparity() - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_disparity_map_dimensions() {
        let m = DisparityMap::constant(8, 6, 1.0);
        assert_eq!(m.dimensions(), (8, 6));
        assert_eq!(m.pixel_count(), 48);
    }

    #[test]
    fn test_disparity_map_valid_pixel_count() {
        let data = vec![0.0, 2.0, 3.0, 0.0, 5.0];
        let m = DisparityMap::new(data, 5, 1);
        assert_eq!(m.valid_pixel_count(), 3);
    }

    #[test]
    fn test_disparity_map_max_disparity() {
        let data = vec![2.0f32, 5.0, 3.0, 10.0, 1.0];
        let m = DisparityMap::new(data, 5, 1);
        assert!((m.max_disparity() - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_depth_estimate_is_reliable() {
        let reliable = DepthEstimate::new(5.0, 0.8, 10.0);
        let unreliable = DepthEstimate::new(5.0, 0.3, 10.0);
        assert!(reliable.is_reliable());
        assert!(!unreliable.is_reliable());
    }

    #[test]
    fn test_depth_estimate_is_reliable_custom() {
        let e = DepthEstimate::new(5.0, 0.7, 10.0);
        assert!(e.is_reliable_at(0.5));
        assert!(!e.is_reliable_at(0.9));
    }

    #[test]
    fn test_depth_estimate_depth_cm() {
        let e = DepthEstimate::new(2.5, 0.9, 10.0);
        assert!((e.depth_cm() - 250.0).abs() < 1e-4);
    }

    #[test]
    fn test_stereo_params_disparity_to_depth() {
        let p = StereoParams::new(0.1, 600.0); // baseline 10 cm, focal 600 px
        let depth = p.disparity_to_depth(10.0);
        // Expected: 0.1 * 600 / 10 = 6 m
        assert!((depth - 6.0).abs() < 1e-4);
    }

    #[test]
    fn test_stereo_params_zero_disparity_gives_infinity() {
        let p = StereoParams::new(0.1, 600.0);
        assert!(p.disparity_to_depth(0.0).is_infinite());
    }

    #[test]
    fn test_estimator_estimate_from_stereo_length() {
        let p = StereoParams::new(0.1, 600.0);
        let est = DepthEstimator::new(p);
        let map = DisparityMap::constant(4, 4, 10.0);
        let estimates = est.estimate_from_stereo(&map);
        assert_eq!(estimates.len(), 16);
    }

    #[test]
    fn test_estimator_all_reliable_close_range() {
        let p = StereoParams::new(0.1, 600.0);
        // depth = 0.1 * 600 / 30 = 2 m  → well within max 50 m
        let est = DepthEstimator::new(p);
        let map = DisparityMap::constant(2, 2, 30.0);
        let estimates = est.estimate_from_stereo(&map);
        assert!(estimates.iter().all(|e| e.is_reliable()));
    }

    #[test]
    fn test_estimator_average_depth() {
        let p = StereoParams::new(0.1, 600.0);
        let est = DepthEstimator::new(p);
        // All pixels at disparity 30 → depth = 2 m
        let map = DisparityMap::constant(3, 3, 30.0);
        let avg = est.average_depth(&map);
        assert!((avg - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_estimator_invalid_disparity_low_confidence() {
        let p = StereoParams::new(0.1, 600.0);
        let est = DepthEstimator::new(p);
        let map = DisparityMap::constant(1, 1, 0.0); // all invalid
        let estimates = est.estimate_from_stereo(&map);
        assert_eq!(estimates[0].confidence, 0.0);
    }
}
