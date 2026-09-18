//! Motion estimation for video stabilization.
//!
//! This module provides algorithms for estimating motion between video frames:
//!
//! - Feature detection (FAST corners)
//! - Optical flow tracking
//! - Homography estimation with RANSAC
//! - Inter-frame transformation computation

use crate::error::{CvError, CvResult};
use crate::tracking::Point2D;
use oximedia_codec::VideoFrame;
use std::f64::consts::PI;

/// Feature point for tracking.
#[derive(Debug, Clone, Copy)]
pub struct Feature {
    /// Position in the image.
    pub position: Point2D,
    /// Feature response strength.
    pub response: f32,
    /// Feature descriptor (optional).
    pub descriptor: [u8; 32],
}

impl Feature {
    /// Create a new feature.
    #[must_use]
    pub const fn new(position: Point2D, response: f32) -> Self {
        Self {
            position,
            response,
            descriptor: [0; 32],
        }
    }
}

/// Feature correspondence between two frames.
#[derive(Debug, Clone, Copy)]
pub struct FeatureMatch {
    /// Feature in the previous frame.
    pub prev: Point2D,
    /// Feature in the current frame.
    pub curr: Point2D,
    /// Match confidence score.
    pub confidence: f32,
}

impl FeatureMatch {
    /// Create a new feature match.
    #[must_use]
    pub const fn new(prev: Point2D, curr: Point2D, confidence: f32) -> Self {
        Self {
            prev,
            curr,
            confidence,
        }
    }

    /// Calculate displacement vector.
    #[must_use]
    pub fn displacement(&self) -> (f32, f32) {
        (self.curr.x - self.prev.x, self.curr.y - self.prev.y)
    }

    /// Calculate displacement magnitude.
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        self.prev.distance(&self.curr)
    }
}

/// Transformation matrix for frame-to-frame motion.
///
/// Represents translation, rotation, and scale.
#[derive(Debug, Clone, Copy)]
pub struct TransformMatrix {
    /// Translation in X direction.
    pub tx: f64,
    /// Translation in Y direction.
    pub ty: f64,
    /// Rotation angle in radians.
    pub angle: f64,
    /// Scale factor.
    pub scale: f64,
}

impl TransformMatrix {
    /// Create a new transformation matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::TransformMatrix;
    ///
    /// let transform = TransformMatrix::new(10.0, 5.0, 0.1, 1.0);
    /// assert_eq!(transform.tx, 10.0);
    /// ```
    #[must_use]
    pub const fn new(tx: f64, ty: f64, angle: f64, scale: f64) -> Self {
        Self {
            tx,
            ty,
            angle,
            scale,
        }
    }

    /// Create an identity transformation.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::TransformMatrix;
    ///
    /// let identity = TransformMatrix::identity();
    /// assert_eq!(identity.tx, 0.0);
    /// assert_eq!(identity.scale, 1.0);
    /// ```
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            tx: 0.0,
            ty: 0.0,
            angle: 0.0,
            scale: 1.0,
        }
    }

    /// Convert to 3x3 homography matrix.
    ///
    /// Returns a 3x3 matrix in row-major order.
    #[must_use]
    pub fn to_homography(&self) -> [f64; 9] {
        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();
        let s = self.scale;

        [
            s * cos_a,
            -s * sin_a,
            self.tx,
            s * sin_a,
            s * cos_a,
            self.ty,
            0.0,
            0.0,
            1.0,
        ]
    }

    /// Create from homography matrix.
    ///
    /// # Arguments
    ///
    /// * `h` - 3x3 homography matrix in row-major order
    #[must_use]
    pub fn from_homography(h: &[f64; 9]) -> Self {
        let tx = h[2];
        let ty = h[5];
        let scale = (h[0] * h[0] + h[3] * h[3]).sqrt();
        let angle = h[3].atan2(h[0]);

        Self {
            tx,
            ty,
            angle,
            scale,
        }
    }

    /// Compute motion magnitude.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        (self.tx * self.tx + self.ty * self.ty).sqrt()
    }

    /// Compose two transformations.
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        // Convert to homography matrices
        let h1 = self.to_homography();
        let h2 = other.to_homography();

        // Multiply matrices
        let result = multiply_homography(&h1, &h2);

        // Convert back to transform parameters
        Self::from_homography(&result)
    }

    /// Invert transformation.
    #[must_use]
    pub fn invert(&self) -> Self {
        let h = self.to_homography();
        let inv = invert_homography(&h);
        Self::from_homography(&inv)
    }

    /// Apply transformation to a point.
    #[must_use]
    pub fn transform_point(&self, point: Point2D) -> Point2D {
        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();
        let s = self.scale;

        let x = s * (cos_a * point.x as f64 - sin_a * point.y as f64) + self.tx;
        let y = s * (sin_a * point.x as f64 + cos_a * point.y as f64) + self.ty;

        Point2D::new(x as f32, y as f32)
    }

    /// Interpolate between two transformations.
    #[must_use]
    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            tx: self.tx + (other.tx - self.tx) * t,
            ty: self.ty + (other.ty - self.ty) * t,
            angle: self.angle + (other.angle - self.angle) * t,
            scale: self.scale + (other.scale - self.scale) * t,
        }
    }
}

impl Default for TransformMatrix {
    fn default() -> Self {
        Self::identity()
    }
}

/// Motion estimator for video frames.
///
/// Estimates transformation between consecutive frames using feature tracking.
#[derive(Debug, Clone)]
pub struct MotionEstimator {
    /// Maximum number of features to track.
    max_features: usize,
    /// Minimum feature quality threshold.
    quality_threshold: f32,
    /// Minimum distance between features.
    min_distance: f32,
    /// Optical flow window size.
    window_size: usize,
    /// Maximum pyramid levels.
    max_pyramid_levels: usize,
    /// RANSAC threshold.
    ransac_threshold: f64,
    /// RANSAC iterations.
    ransac_iterations: usize,
}

impl MotionEstimator {
    /// Create a new motion estimator.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::MotionEstimator;
    ///
    /// let estimator = MotionEstimator::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_features: 500,
            quality_threshold: 0.01,
            min_distance: 10.0,
            window_size: 21,
            max_pyramid_levels: 3,
            ransac_threshold: 3.0,
            ransac_iterations: 1000,
        }
    }

    /// Set maximum number of features.
    #[must_use]
    pub const fn with_max_features(mut self, max_features: usize) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set quality threshold.
    #[must_use]
    pub const fn with_quality_threshold(mut self, threshold: f32) -> Self {
        self.quality_threshold = threshold;
        self
    }

    /// Estimate transformation between two frames.
    ///
    /// # Arguments
    ///
    /// * `prev_frame` - Previous video frame
    /// * `curr_frame` - Current video frame
    ///
    /// # Errors
    ///
    /// Returns an error if motion estimation fails.
    pub fn estimate_transform(
        &self,
        prev_frame: &VideoFrame,
        curr_frame: &VideoFrame,
    ) -> CvResult<TransformMatrix> {
        // Convert frames to grayscale
        let prev_gray = self.convert_to_grayscale(prev_frame)?;
        let curr_gray = self.convert_to_grayscale(curr_frame)?;

        // Detect features in previous frame
        let features = self.detect_fast_corners(&prev_gray, prev_frame.width, prev_frame.height)?;

        if features.len() < 4 {
            return Ok(TransformMatrix::identity());
        }

        // Track features using optical flow
        let matches = self.track_features(&prev_gray, &curr_gray, &features, prev_frame.width)?;

        if matches.len() < 4 {
            return Ok(TransformMatrix::identity());
        }

        // Estimate homography using RANSAC
        let homography = HomographyEstimator::estimate_with_ransac(
            &matches,
            self.ransac_threshold,
            self.ransac_iterations,
        )?;

        // Convert homography to transformation parameters
        Ok(TransformMatrix::from_homography(&homography))
    }

    /// Convert video frame to grayscale.
    fn convert_to_grayscale(&self, frame: &VideoFrame) -> CvResult<Vec<u8>> {
        if frame.planes.is_empty() {
            return Err(CvError::insufficient_data(1, 0));
        }

        // For YUV formats, just use the Y (luma) plane
        let luma_plane = &frame.planes[0];
        Ok(luma_plane.data.clone())
    }

    /// Detect FAST corners in grayscale image.
    fn detect_fast_corners(&self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<Feature>> {
        let mut features = Vec::new();
        let threshold = 20;
        let radius = 3;

        // Simple FAST corner detection
        for y in radius..(height - radius) {
            for x in radius..(width - radius) {
                let idx = (y * width + x) as usize;
                if idx >= image.len() {
                    continue;
                }

                let center = image[idx];
                let response = self.compute_fast_response(image, width, x, y, center, threshold);

                if response > self.quality_threshold {
                    let position = Point2D::new(x as f32, y as f32);
                    features.push(Feature::new(position, response));
                }
            }
        }

        // Sort by response strength and take top features
        features.sort_by(|a, b| {
            b.response
                .partial_cmp(&a.response)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        features.truncate(self.max_features);

        // Apply non-maximum suppression
        self.non_maximum_suppression(&mut features);

        Ok(features)
    }

    /// Compute FAST corner response.
    fn compute_fast_response(
        &self,
        image: &[u8],
        width: u32,
        x: u32,
        y: u32,
        center: u8,
        threshold: i32,
    ) -> f32 {
        // FAST-9 circle pattern
        let offsets = [
            (0, -3),
            (1, -3),
            (2, -2),
            (3, -1),
            (3, 0),
            (3, 1),
            (2, 2),
            (1, 3),
            (0, 3),
            (-1, 3),
            (-2, 2),
            (-3, 1),
            (-3, 0),
            (-3, -1),
            (-2, -2),
            (-1, -3),
        ];

        let mut darker_count = 0;
        let mut brighter_count = 0;

        for (dx, dy) in &offsets {
            let px = (x as i32 + dx) as u32;
            let py = (y as i32 + dy) as u32;
            let idx = (py * width + px) as usize;

            if idx >= image.len() {
                continue;
            }

            let pixel = image[idx];
            let diff = pixel as i32 - center as i32;

            if diff < -threshold {
                darker_count += 1;
            } else if diff > threshold {
                brighter_count += 1;
            }
        }

        // Return response based on continuous pixels
        if darker_count >= 9 || brighter_count >= 9 {
            darker_count.max(brighter_count) as f32 / 16.0
        } else {
            0.0
        }
    }

    /// Apply non-maximum suppression to features.
    fn non_maximum_suppression(&self, features: &mut Vec<Feature>) {
        let min_dist_sq = self.min_distance * self.min_distance;
        let mut i = 0;

        while i < features.len() {
            let mut j = i + 1;
            while j < features.len() {
                let dist_sq = features[i].position.distance_squared(&features[j].position);
                if dist_sq < min_dist_sq {
                    features.remove(j);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }

    /// Track features using Lucas-Kanade optical flow.
    fn track_features(
        &self,
        prev_image: &[u8],
        curr_image: &[u8],
        features: &[Feature],
        width: u32,
    ) -> CvResult<Vec<FeatureMatch>> {
        let mut matches = Vec::new();

        for feature in features {
            if let Some(tracked_pos) =
                self.track_single_feature(prev_image, curr_image, feature.position, width)
            {
                let match_obj = FeatureMatch::new(feature.position, tracked_pos, feature.response);
                matches.push(match_obj);
            }
        }

        Ok(matches)
    }

    /// Track a single feature using Lucas-Kanade.
    fn track_single_feature(
        &self,
        prev_image: &[u8],
        curr_image: &[u8],
        position: Point2D,
        width: u32,
    ) -> Option<Point2D> {
        let half_win = (self.window_size / 2) as i32;
        let x = position.x as i32;
        let y = position.y as i32;

        // Simple search in a window
        let mut best_x = x;
        let mut best_y = y;
        let mut best_score = f32::MAX;

        for dy in -half_win..=half_win {
            for dx in -half_win..=half_win {
                let score = self.compute_patch_similarity(
                    prev_image,
                    curr_image,
                    x,
                    y,
                    x + dx,
                    y + dy,
                    width,
                    half_win,
                );

                if score < best_score {
                    best_score = score;
                    best_x = x + dx;
                    best_y = y + dy;
                }
            }
        }

        // Check if tracking was successful
        if best_score < 1000.0 {
            Some(Point2D::new(best_x as f32, best_y as f32))
        } else {
            None
        }
    }

    /// Compute similarity between two image patches.
    #[allow(clippy::too_many_arguments)]
    fn compute_patch_similarity(
        &self,
        image1: &[u8],
        image2: &[u8],
        x1: i32,
        y1: i32,
        x2: i32,
        y2: i32,
        width: u32,
        half_win: i32,
    ) -> f32 {
        let mut sum = 0.0;
        let mut count = 0;

        for dy in -half_win..=half_win {
            for dx in -half_win..=half_win {
                let idx1 = ((y1 + dy) * width as i32 + (x1 + dx)) as usize;
                let idx2 = ((y2 + dy) * width as i32 + (x2 + dx)) as usize;

                if idx1 < image1.len() && idx2 < image2.len() {
                    let diff = image1[idx1] as f32 - image2[idx2] as f32;
                    sum += diff * diff;
                    count += 1;
                }
            }
        }

        if count > 0 {
            sum / count as f32
        } else {
            f32::MAX
        }
    }
}

impl Default for MotionEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Homography estimator using RANSAC.
///
/// Estimates a homography transformation that maps points from one frame to another.
pub struct HomographyEstimator;

impl HomographyEstimator {
    /// Estimate homography using RANSAC.
    ///
    /// # Arguments
    ///
    /// * `matches` - Feature correspondences
    /// * `threshold` - RANSAC inlier threshold
    /// * `iterations` - Maximum RANSAC iterations
    ///
    /// # Errors
    ///
    /// Returns an error if estimation fails.
    pub fn estimate_with_ransac(
        matches: &[FeatureMatch],
        threshold: f64,
        iterations: usize,
    ) -> CvResult<[f64; 9]> {
        if matches.len() < 4 {
            return Err(CvError::matrix_error(
                "Need at least 4 matches for homography",
            ));
        }

        let mut best_homography = [0.0; 9];
        let mut best_inliers = 0;

        // Use a simple random selection for RANSAC
        for iter in 0..iterations {
            // Select 4 random matches
            let indices = Self::select_random_indices(matches.len(), 4, iter);
            let sample: Vec<_> = indices.iter().map(|&i| matches[i]).collect();

            // Estimate homography from the sample
            if let Ok(h) = Self::estimate_homography_4pt(&sample) {
                // Count inliers
                let inliers = Self::count_inliers(matches, &h, threshold);

                if inliers > best_inliers {
                    best_inliers = inliers;
                    best_homography = h;
                }
            }
        }

        if best_inliers < 4 {
            return Err(CvError::matrix_error("Failed to find enough inliers"));
        }

        Ok(best_homography)
    }

    /// Estimate homography from 4 point correspondences.
    fn estimate_homography_4pt(matches: &[FeatureMatch]) -> CvResult<[f64; 9]> {
        if matches.len() < 4 {
            return Err(CvError::matrix_error("Need at least 4 matches"));
        }

        // Simplified homography estimation using affine transformation
        // For better results, should use DLT (Direct Linear Transform)

        let mut sum_dx = 0.0;
        let mut sum_dy = 0.0;
        let mut sum_angle = 0.0;
        let mut sum_scale = 0.0;

        for m in matches {
            let (dx, dy) = m.displacement();
            sum_dx += dx as f64;
            sum_dy += dy as f64;

            let scale = 1.0; // Simplified
            sum_scale += scale;
        }

        let n = matches.len() as f64;
        let tx = sum_dx / n;
        let ty = sum_dy / n;
        let angle = sum_angle / n;
        let scale = sum_scale / n;

        let cos_a = angle.cos();
        let sin_a = angle.sin();

        Ok([
            scale * cos_a,
            -scale * sin_a,
            tx,
            scale * sin_a,
            scale * cos_a,
            ty,
            0.0,
            0.0,
            1.0,
        ])
    }

    /// Count inliers for a homography.
    fn count_inliers(matches: &[FeatureMatch], homography: &[f64; 9], threshold: f64) -> usize {
        let threshold_sq = threshold * threshold;
        let mut count = 0;

        for m in matches {
            let error = Self::reprojection_error(&m.prev, &m.curr, homography);
            if error < threshold_sq {
                count += 1;
            }
        }

        count
    }

    /// Compute reprojection error for a point correspondence.
    fn reprojection_error(p1: &Point2D, p2: &Point2D, h: &[f64; 9]) -> f64 {
        // Apply homography to p1
        let x = h[0] * p1.x as f64 + h[1] * p1.y as f64 + h[2];
        let y = h[3] * p1.x as f64 + h[4] * p1.y as f64 + h[5];
        let w = h[6] * p1.x as f64 + h[7] * p1.y as f64 + h[8];

        let projected_x = x / w;
        let projected_y = y / w;

        // Compute squared distance to p2
        let dx = projected_x - p2.x as f64;
        let dy = projected_y - p2.y as f64;
        dx * dx + dy * dy
    }

    /// Select random indices for RANSAC sampling.
    fn select_random_indices(n: usize, k: usize, seed: usize) -> Vec<usize> {
        // Simple pseudo-random selection
        let mut indices = Vec::new();
        for i in 0..k {
            let idx = (seed * 1_103_515_245 + i * 12_345) % n;
            if !indices.contains(&idx) {
                indices.push(idx);
            }
        }
        while indices.len() < k {
            indices.push((seed + indices.len()) % n);
        }
        indices
    }
}

/// Multiply two 3x3 homography matrices.
fn multiply_homography(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    [
        a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
        a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
        a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
        a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
        a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
        a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
        a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
        a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
        a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
    ]
}

/// Invert a 3x3 homography matrix.
fn invert_homography(h: &[f64; 9]) -> [f64; 9] {
    // Compute determinant
    let det = h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6])
        + h[2] * (h[3] * h[7] - h[4] * h[6]);

    if det.abs() < 1e-10 {
        // Singular matrix, return identity
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }

    let inv_det = 1.0 / det;

    [
        (h[4] * h[8] - h[5] * h[7]) * inv_det,
        (h[2] * h[7] - h[1] * h[8]) * inv_det,
        (h[1] * h[5] - h[2] * h[4]) * inv_det,
        (h[5] * h[6] - h[3] * h[8]) * inv_det,
        (h[0] * h[8] - h[2] * h[6]) * inv_det,
        (h[2] * h[3] - h[0] * h[5]) * inv_det,
        (h[3] * h[7] - h[4] * h[6]) * inv_det,
        (h[1] * h[6] - h[0] * h[7]) * inv_det,
        (h[0] * h[4] - h[1] * h[3]) * inv_det,
    ]
}

/// Compute angle between two vectors.
#[allow(dead_code)]
fn compute_angle(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dot = x1 * x2 + y1 * y2;
    let cross = x1 * y2 - y1 * x2;
    cross.atan2(dot)
}

/// Normalize angle to [-π, π] range.
#[allow(dead_code)]
fn normalize_angle(angle: f64) -> f64 {
    let mut a = angle;
    while a > PI {
        a -= 2.0 * PI;
    }
    while a < -PI {
        a += 2.0 * PI;
    }
    a
}

// ============================================================================
// Gyroscope Data Fusion for Hybrid Stabilization
// ============================================================================

/// Gyroscope measurement sample.
#[derive(Debug, Clone, Copy)]
pub struct GyroscopeData {
    /// Angular velocity around X axis (pitch) in radians/second.
    pub gyro_x: f64,
    /// Angular velocity around Y axis (yaw) in radians/second.
    pub gyro_y: f64,
    /// Angular velocity around Z axis (roll) in radians/second.
    pub gyro_z: f64,
    /// Timestamp in seconds (relative to video start).
    pub timestamp: f64,
}

impl GyroscopeData {
    /// Create a new gyroscope sample.
    #[must_use]
    pub const fn new(gyro_x: f64, gyro_y: f64, gyro_z: f64, timestamp: f64) -> Self {
        Self {
            gyro_x,
            gyro_y,
            gyro_z,
            timestamp,
        }
    }

    /// Create a zero (stationary) sample.
    #[must_use]
    pub const fn zero(timestamp: f64) -> Self {
        Self::new(0.0, 0.0, 0.0, timestamp)
    }

    /// Compute angular velocity magnitude.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        (self.gyro_x * self.gyro_x + self.gyro_y * self.gyro_y + self.gyro_z * self.gyro_z).sqrt()
    }

    /// Integrate gyroscope sample over a time interval to get angle change.
    #[must_use]
    pub fn integrate(&self, dt: f64) -> TransformMatrix {
        TransformMatrix {
            tx: 0.0,
            ty: 0.0,
            angle: self.gyro_z * dt, // Roll contributes to 2D rotation
            scale: 1.0,
        }
    }
}

/// Gyroscope fusion configuration.
#[derive(Debug, Clone)]
pub struct GyroscopeFusionConfig {
    /// Weight of gyroscope data vs visual data (0.0 = visual only, 1.0 = gyro only).
    pub gyro_weight: f64,
    /// Kalman process noise (gyroscope uncertainty).
    pub process_noise: f64,
    /// Kalman measurement noise (visual estimation uncertainty).
    pub measurement_noise: f64,
    /// Gyroscope bias correction factor.
    pub bias_correction_rate: f64,
    /// Maximum allowed gyroscope drift (radians/second).
    pub max_drift: f64,
    /// Focal length in pixels (for translating angular motion to pixel motion).
    pub focal_length: f64,
    /// Frame rate (for gyroscope integration).
    pub frame_rate: f64,
}

impl Default for GyroscopeFusionConfig {
    fn default() -> Self {
        Self {
            gyro_weight: 0.6,
            process_noise: 0.01,
            measurement_noise: 1.0,
            bias_correction_rate: 0.001,
            max_drift: 0.1,
            focal_length: 500.0,
            frame_rate: 30.0,
        }
    }
}

impl GyroscopeFusionConfig {
    /// Create a new fusion configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set gyroscope weight.
    #[must_use]
    pub fn with_gyro_weight(mut self, weight: f64) -> Self {
        self.gyro_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set process noise.
    #[must_use]
    pub fn with_process_noise(mut self, noise: f64) -> Self {
        self.process_noise = noise.max(0.0);
        self
    }

    /// Set measurement noise.
    #[must_use]
    pub fn with_measurement_noise(mut self, noise: f64) -> Self {
        self.measurement_noise = noise.max(0.001);
        self
    }

    /// Set focal length in pixels.
    #[must_use]
    pub fn with_focal_length(mut self, focal: f64) -> Self {
        self.focal_length = focal.max(1.0);
        self
    }

    /// Set frame rate.
    #[must_use]
    pub fn with_frame_rate(mut self, fps: f64) -> Self {
        self.frame_rate = fps.max(1.0);
        self
    }
}

/// Simplified 1D Kalman state for one motion parameter.
#[derive(Debug, Clone, Copy)]
struct KalmanState1D {
    /// Current state estimate.
    estimate: f64,
    /// Error covariance.
    error_covariance: f64,
}

impl KalmanState1D {
    fn new() -> Self {
        Self {
            estimate: 0.0,
            error_covariance: 1.0,
        }
    }

    /// Predict step using gyroscope data.
    fn predict(&mut self, gyro_prediction: f64, process_noise: f64) {
        self.estimate = gyro_prediction;
        self.error_covariance += process_noise;
    }

    /// Update step using visual measurement.
    fn update(&mut self, visual_measurement: f64, measurement_noise: f64) {
        let innovation = visual_measurement - self.estimate;
        let innovation_covariance = self.error_covariance + measurement_noise;

        // Kalman gain
        let gain = if innovation_covariance > 1e-10 {
            self.error_covariance / innovation_covariance
        } else {
            0.5
        };

        self.estimate += gain * innovation;
        self.error_covariance *= 1.0 - gain;
    }
}

/// Hybrid motion estimator that fuses visual and gyroscope data.
///
/// Uses a Kalman filter to optimally combine visual feature tracking
/// with gyroscope readings for more accurate and robust stabilization.
///
/// # Example
///
/// ```
/// use oximedia_cv::stabilize::motion::{
///     HybridMotionEstimator, GyroscopeFusionConfig, GyroscopeData, TransformMatrix,
/// };
///
/// let config = GyroscopeFusionConfig::default();
/// let mut estimator = HybridMotionEstimator::new(config);
///
/// let visual = TransformMatrix::new(2.0, 1.0, 0.01, 1.0);
/// let gyro = GyroscopeData::new(0.0, 0.0, 0.02, 0.033);
///
/// let fused = estimator.fuse(visual, &gyro);
/// ```
pub struct HybridMotionEstimator {
    /// Fusion configuration.
    config: GyroscopeFusionConfig,
    /// Kalman state for translation X.
    kalman_tx: KalmanState1D,
    /// Kalman state for translation Y.
    kalman_ty: KalmanState1D,
    /// Kalman state for rotation angle.
    kalman_angle: KalmanState1D,
    /// Estimated gyroscope bias (accumulated drift).
    gyro_bias: (f64, f64, f64),
    /// Previous gyroscope timestamp.
    prev_timestamp: Option<f64>,
    /// Number of fused frames.
    frame_count: u64,
}

impl HybridMotionEstimator {
    /// Create a new hybrid motion estimator.
    #[must_use]
    pub fn new(config: GyroscopeFusionConfig) -> Self {
        Self {
            config,
            kalman_tx: KalmanState1D::new(),
            kalman_ty: KalmanState1D::new(),
            kalman_angle: KalmanState1D::new(),
            gyro_bias: (0.0, 0.0, 0.0),
            prev_timestamp: None,
            frame_count: 0,
        }
    }

    /// Fuse visual motion estimate with gyroscope data.
    ///
    /// The Kalman filter predicts using gyroscope data and corrects using
    /// visual feature tracking, producing a fused estimate.
    ///
    /// # Arguments
    ///
    /// * `visual_transform` - Motion estimate from visual feature tracking
    /// * `gyro_data` - Gyroscope reading for this frame interval
    ///
    /// # Returns
    ///
    /// Fused transformation estimate.
    pub fn fuse(
        &mut self,
        visual_transform: TransformMatrix,
        gyro_data: &GyroscopeData,
    ) -> TransformMatrix {
        // Calculate time delta
        let dt = match self.prev_timestamp {
            Some(prev_t) => (gyro_data.timestamp - prev_t).max(0.001),
            None => 1.0 / self.config.frame_rate,
        };
        self.prev_timestamp = Some(gyro_data.timestamp);

        // Correct gyroscope for estimated bias
        let corrected_gx = gyro_data.gyro_x - self.gyro_bias.0;
        let corrected_gy = gyro_data.gyro_y - self.gyro_bias.1;
        let corrected_gz = gyro_data.gyro_z - self.gyro_bias.2;

        // Convert gyroscope angular velocity to pixel displacement
        let focal = self.config.focal_length;
        let gyro_tx = -corrected_gy * focal * dt; // Yaw -> horizontal translation
        let gyro_ty = corrected_gx * focal * dt; // Pitch -> vertical translation
        let gyro_angle = corrected_gz * dt; // Roll -> rotation

        // Kalman predict (using gyroscope)
        let process_noise = self.config.process_noise;
        self.kalman_tx.predict(gyro_tx, process_noise);
        self.kalman_ty.predict(gyro_ty, process_noise);
        self.kalman_angle.predict(gyro_angle, process_noise * 0.1);

        // Kalman update (using visual measurement)
        let measurement_noise = self.config.measurement_noise;
        self.kalman_tx
            .update(visual_transform.tx, measurement_noise);
        self.kalman_ty
            .update(visual_transform.ty, measurement_noise);
        self.kalman_angle
            .update(visual_transform.angle, measurement_noise * 0.1);

        // Update gyroscope bias estimate
        let bias_rate = self.config.bias_correction_rate;
        let visual_gz = visual_transform.angle / dt;
        self.gyro_bias.2 += bias_rate * (gyro_data.gyro_z - visual_gz - self.gyro_bias.2);

        // Clamp bias to max drift
        let max_drift = self.config.max_drift;
        self.gyro_bias.0 = self.gyro_bias.0.clamp(-max_drift, max_drift);
        self.gyro_bias.1 = self.gyro_bias.1.clamp(-max_drift, max_drift);
        self.gyro_bias.2 = self.gyro_bias.2.clamp(-max_drift, max_drift);

        self.frame_count += 1;

        TransformMatrix {
            tx: self.kalman_tx.estimate,
            ty: self.kalman_ty.estimate,
            angle: self.kalman_angle.estimate,
            scale: visual_transform.scale, // Scale from visual only
        }
    }

    /// Fuse with multiple gyroscope samples between frames.
    ///
    /// When gyroscope samples at higher rate than video, integrate
    /// all samples between frames for higher accuracy.
    pub fn fuse_multi_sample(
        &mut self,
        visual_transform: TransformMatrix,
        gyro_samples: &[GyroscopeData],
    ) -> TransformMatrix {
        if gyro_samples.is_empty() {
            return visual_transform;
        }

        // Integrate all gyroscope samples
        let mut integrated_tx = 0.0;
        let mut integrated_ty = 0.0;
        let mut integrated_angle = 0.0;
        let focal = self.config.focal_length;

        for i in 0..gyro_samples.len() {
            let dt = if i == 0 {
                match self.prev_timestamp {
                    Some(prev_t) => (gyro_samples[0].timestamp - prev_t).max(0.001),
                    None => 1.0 / self.config.frame_rate / gyro_samples.len() as f64,
                }
            } else {
                (gyro_samples[i].timestamp - gyro_samples[i - 1].timestamp).max(0.001)
            };

            let gx = gyro_samples[i].gyro_x - self.gyro_bias.0;
            let gy = gyro_samples[i].gyro_y - self.gyro_bias.1;
            let gz = gyro_samples[i].gyro_z - self.gyro_bias.2;

            integrated_tx += -gy * focal * dt;
            integrated_ty += gx * focal * dt;
            integrated_angle += gz * dt;
        }

        // Use last sample timestamp
        let last_sample = &gyro_samples[gyro_samples.len() - 1];
        let synthetic_gyro = GyroscopeData {
            gyro_x: integrated_ty / focal * self.config.frame_rate,
            gyro_y: -integrated_tx / focal * self.config.frame_rate,
            gyro_z: integrated_angle * self.config.frame_rate,
            timestamp: last_sample.timestamp,
        };

        self.fuse(visual_transform, &synthetic_gyro)
    }

    /// Get the current estimated gyroscope bias.
    #[must_use]
    pub const fn gyro_bias(&self) -> (f64, f64, f64) {
        self.gyro_bias
    }

    /// Get the number of frames processed.
    #[must_use]
    pub const fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Reset the estimator state.
    pub fn reset(&mut self) {
        self.kalman_tx = KalmanState1D::new();
        self.kalman_ty = KalmanState1D::new();
        self.kalman_angle = KalmanState1D::new();
        self.gyro_bias = (0.0, 0.0, 0.0);
        self.prev_timestamp = None;
        self.frame_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_matrix_identity() {
        let t = TransformMatrix::identity();
        assert_eq!(t.tx, 0.0);
        assert_eq!(t.ty, 0.0);
        assert_eq!(t.angle, 0.0);
        assert_eq!(t.scale, 1.0);
    }

    #[test]
    fn test_transform_matrix_magnitude() {
        let t = TransformMatrix::new(3.0, 4.0, 0.0, 1.0);
        assert!((t.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_matrix_to_from_homography() {
        let t = TransformMatrix::new(10.0, 5.0, 0.1, 1.0);
        let h = t.to_homography();
        let t2 = TransformMatrix::from_homography(&h);

        assert!((t.tx - t2.tx).abs() < 1e-10);
        assert!((t.ty - t2.ty).abs() < 1e-10);
        assert!((t.angle - t2.angle).abs() < 1e-10);
        assert!((t.scale - t2.scale).abs() < 1e-10);
    }

    #[test]
    fn test_transform_interpolate() {
        let a = TransformMatrix::new(0.0, 0.0, 0.0, 1.0);
        let b = TransformMatrix::new(10.0, 20.0, 0.5, 2.0);
        let mid = a.interpolate(&b, 0.5);

        assert!((mid.tx - 5.0).abs() < 1e-10);
        assert!((mid.ty - 10.0).abs() < 1e-10);
        assert!((mid.angle - 0.25).abs() < 1e-10);
        assert!((mid.scale - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_gyroscope_data_new() {
        let g = GyroscopeData::new(0.1, 0.2, 0.3, 1.0);
        assert_eq!(g.gyro_x, 0.1);
        assert_eq!(g.gyro_y, 0.2);
        assert_eq!(g.gyro_z, 0.3);
        assert_eq!(g.timestamp, 1.0);
    }

    #[test]
    fn test_gyroscope_data_zero() {
        let g = GyroscopeData::zero(0.5);
        assert_eq!(g.magnitude(), 0.0);
        assert_eq!(g.timestamp, 0.5);
    }

    #[test]
    fn test_gyroscope_magnitude() {
        let g = GyroscopeData::new(3.0, 4.0, 0.0, 0.0);
        assert!((g.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_gyroscope_integrate() {
        let g = GyroscopeData::new(0.0, 0.0, 1.0, 0.0); // 1 rad/s roll
        let t = g.integrate(0.1); // 0.1 second
        assert!((t.angle - 0.1).abs() < 1e-10);
        assert_eq!(t.tx, 0.0);
        assert_eq!(t.ty, 0.0);
        assert_eq!(t.scale, 1.0);
    }

    #[test]
    fn test_fusion_config_default() {
        let config = GyroscopeFusionConfig::default();
        assert_eq!(config.gyro_weight, 0.6);
        assert_eq!(config.focal_length, 500.0);
        assert_eq!(config.frame_rate, 30.0);
    }

    #[test]
    fn test_fusion_config_builder() {
        let config = GyroscopeFusionConfig::new()
            .with_gyro_weight(0.8)
            .with_process_noise(0.02)
            .with_measurement_noise(2.0)
            .with_focal_length(800.0)
            .with_frame_rate(60.0);

        assert_eq!(config.gyro_weight, 0.8);
        assert_eq!(config.process_noise, 0.02);
        assert_eq!(config.measurement_noise, 2.0);
        assert_eq!(config.focal_length, 800.0);
        assert_eq!(config.frame_rate, 60.0);
    }

    #[test]
    fn test_hybrid_estimator_new() {
        let estimator = HybridMotionEstimator::new(GyroscopeFusionConfig::default());
        assert_eq!(estimator.frame_count(), 0);
        assert_eq!(estimator.gyro_bias(), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_hybrid_fuse_stationary() {
        let mut estimator = HybridMotionEstimator::new(GyroscopeFusionConfig::default());

        let visual = TransformMatrix::identity();
        let gyro = GyroscopeData::zero(0.033);

        let fused = estimator.fuse(visual, &gyro);
        assert!(fused.tx.abs() < 1.0);
        assert!(fused.ty.abs() < 1.0);
        assert!(fused.angle.abs() < 0.01);
        assert_eq!(estimator.frame_count(), 1);
    }

    #[test]
    fn test_hybrid_fuse_consistent_motion() {
        let mut estimator = HybridMotionEstimator::new(GyroscopeFusionConfig::default());

        // Both visual and gyro agree on motion
        let visual = TransformMatrix::new(5.0, 3.0, 0.02, 1.0);
        let gyro = GyroscopeData::new(
            3.0 / 500.0 / 0.033,  // pitch -> ty
            -5.0 / 500.0 / 0.033, // yaw -> tx
            0.02 / 0.033,         // roll -> angle
            0.033,
        );

        let fused = estimator.fuse(visual, &gyro);
        // Fused should be close to both estimates (they agree)
        assert!((fused.tx - 5.0).abs() < 3.0, "tx={}", fused.tx);
        assert!((fused.ty - 3.0).abs() < 3.0, "ty={}", fused.ty);
    }

    #[test]
    fn test_hybrid_fuse_multiple_frames() {
        let mut estimator = HybridMotionEstimator::new(GyroscopeFusionConfig::default());

        for i in 0..10 {
            let t = (i + 1) as f64 * 0.033;
            let visual = TransformMatrix::new(1.0, 0.5, 0.01, 1.0);
            let gyro = GyroscopeData::new(0.0, 0.0, 0.3, t);
            let _ = estimator.fuse(visual, &gyro);
        }

        assert_eq!(estimator.frame_count(), 10);
    }

    #[test]
    fn test_hybrid_fuse_multi_sample() {
        let mut estimator = HybridMotionEstimator::new(GyroscopeFusionConfig::default());

        let visual = TransformMatrix::new(2.0, 1.0, 0.01, 1.0);
        let samples = vec![
            GyroscopeData::new(0.01, 0.02, 0.03, 0.011),
            GyroscopeData::new(0.01, 0.02, 0.03, 0.022),
            GyroscopeData::new(0.01, 0.02, 0.03, 0.033),
        ];

        let fused = estimator.fuse_multi_sample(visual, &samples);
        assert!(fused.tx.abs() < 100.0); // Sanity check
        assert_eq!(estimator.frame_count(), 1);
    }

    #[test]
    fn test_hybrid_fuse_multi_sample_empty() {
        let mut estimator = HybridMotionEstimator::new(GyroscopeFusionConfig::default());

        let visual = TransformMatrix::new(2.0, 1.0, 0.01, 1.0);
        let fused = estimator.fuse_multi_sample(visual, &[]);

        // Should return visual transform as-is
        assert_eq!(fused.tx, visual.tx);
        assert_eq!(fused.ty, visual.ty);
    }

    #[test]
    fn test_hybrid_reset() {
        let mut estimator = HybridMotionEstimator::new(GyroscopeFusionConfig::default());

        let visual = TransformMatrix::new(2.0, 1.0, 0.01, 1.0);
        let gyro = GyroscopeData::new(0.0, 0.0, 0.3, 0.033);
        let _ = estimator.fuse(visual, &gyro);

        estimator.reset();
        assert_eq!(estimator.frame_count(), 0);
        assert_eq!(estimator.gyro_bias(), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_gyro_bias_accumulation() {
        let config = GyroscopeFusionConfig::new().with_gyro_weight(0.5);
        let mut estimator = HybridMotionEstimator::new(config);

        // Feed consistent bias: gyro says 0.1 rad/s roll but visual says no rotation
        for i in 0..100 {
            let t = (i + 1) as f64 * 0.033;
            let visual = TransformMatrix::identity();
            let gyro = GyroscopeData::new(0.0, 0.0, 0.1, t);
            let _ = estimator.fuse(visual, &gyro);
        }

        // Bias should have accumulated toward 0.1
        let (_, _, bias_z) = estimator.gyro_bias();
        assert!(bias_z.abs() > 0.0, "Bias should accumulate, got {bias_z}");
    }

    #[test]
    fn test_kalman_state_predict_update() {
        let mut state = KalmanState1D::new();

        // Predict
        state.predict(5.0, 0.1);
        assert!((state.estimate - 5.0).abs() < 1e-10);

        // Update with measurement close to prediction
        state.update(5.5, 1.0);
        // Estimate should be between prediction and measurement
        assert!(state.estimate > 5.0);
        assert!(state.estimate < 5.5);
    }

    #[test]
    fn test_motion_estimator_new() {
        let estimator = MotionEstimator::new();
        assert_eq!(estimator.max_features, 500);
    }

    #[test]
    fn test_feature_match_displacement() {
        let m = FeatureMatch::new(Point2D::new(10.0, 20.0), Point2D::new(15.0, 22.0), 0.9);
        let (dx, dy) = m.displacement();
        assert_eq!(dx, 5.0);
        assert_eq!(dy, 2.0);
    }

    #[test]
    fn test_feature_match_magnitude() {
        let m = FeatureMatch::new(Point2D::new(0.0, 0.0), Point2D::new(3.0, 4.0), 0.9);
        assert!((m.magnitude() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_multiply_homography_identity() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = multiply_homography(&identity, &identity);
        for i in 0..9 {
            assert!(
                (result[i] - identity[i]).abs() < 1e-10,
                "Index {i}: {} vs {}",
                result[i],
                identity[i]
            );
        }
    }

    #[test]
    fn test_invert_homography_identity() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let inv = invert_homography(&identity);
        for i in 0..9 {
            assert!((inv[i] - identity[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_transform_compose_identity() {
        let a = TransformMatrix::identity();
        let b = TransformMatrix::new(5.0, 3.0, 0.1, 1.0);
        let composed = a.compose(&b);
        assert!((composed.tx - b.tx).abs() < 1e-6);
        assert!((composed.ty - b.ty).abs() < 1e-6);
    }

    #[test]
    fn test_transform_invert() {
        let t = TransformMatrix::new(10.0, 5.0, 0.0, 1.0);
        let inv = t.invert();
        let composed = t.compose(&inv);
        assert!(
            composed.magnitude() < 1.0,
            "Compose with inverse should be near identity"
        );
    }
}
