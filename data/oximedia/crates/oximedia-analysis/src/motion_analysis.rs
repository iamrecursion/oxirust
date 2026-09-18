//! Motion analysis: optical flow estimation, motion vectors, and camera motion classification.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

/// A 2D motion vector.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MotionVector {
    /// Horizontal displacement (pixels).
    pub dx: f32,
    /// Vertical displacement (pixels).
    pub dy: f32,
}

impl MotionVector {
    /// Create a new motion vector.
    #[must_use]
    pub fn new(dx: f32, dy: f32) -> Self {
        Self { dx, dy }
    }

    /// Compute the magnitude of the motion vector.
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Compute the angle in radians (atan2).
    #[must_use]
    pub fn angle(&self) -> f32 {
        self.dy.atan2(self.dx)
    }

    /// Return a zero vector.
    #[must_use]
    pub fn zero() -> Self {
        Self { dx: 0.0, dy: 0.0 }
    }
}

/// Camera motion type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CameraMotion {
    /// Camera is static.
    #[default]
    Static,
    /// Horizontal pan.
    Pan,
    /// Vertical tilt.
    Tilt,
    /// Zoom in.
    ZoomIn,
    /// Zoom out.
    ZoomOut,
    /// Camera roll / rotation.
    Roll,
    /// Combination of movements.
    Complex,
}

/// A block-based motion field for a single frame transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionField {
    /// Width of the frame in pixels.
    pub frame_width: usize,
    /// Height of the frame in pixels.
    pub frame_height: usize,
    /// Block size used for estimation.
    pub block_size: usize,
    /// Motion vectors, row-major order.
    pub vectors: Vec<MotionVector>,
}

impl MotionField {
    /// Create a zero-filled motion field.
    #[must_use]
    pub fn new(frame_width: usize, frame_height: usize, block_size: usize) -> Self {
        let cols = frame_width.div_ceil(block_size);
        let rows = frame_height.div_ceil(block_size);
        let count = cols * rows;
        Self {
            frame_width,
            frame_height,
            block_size,
            vectors: vec![MotionVector::zero(); count],
        }
    }

    /// Number of blocks horizontally.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.frame_width.div_ceil(self.block_size)
    }

    /// Number of blocks vertically.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.frame_height.div_ceil(self.block_size)
    }

    /// Mean motion vector across all blocks.
    #[must_use]
    pub fn mean_vector(&self) -> MotionVector {
        if self.vectors.is_empty() {
            return MotionVector::zero();
        }
        let n = self.vectors.len() as f32;
        let dx = self.vectors.iter().map(|v| v.dx).sum::<f32>() / n;
        let dy = self.vectors.iter().map(|v| v.dy).sum::<f32>() / n;
        MotionVector::new(dx, dy)
    }

    /// Mean motion magnitude across all blocks.
    #[must_use]
    pub fn mean_magnitude(&self) -> f32 {
        if self.vectors.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.vectors.iter().map(MotionVector::magnitude).sum();
        sum / self.vectors.len() as f32
    }

    /// Variance of motion magnitudes.
    #[must_use]
    pub fn magnitude_variance(&self) -> f32 {
        if self.vectors.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_magnitude();
        let variance: f32 = self
            .vectors
            .iter()
            .map(|v| {
                let d = v.magnitude() - mean;
                d * d
            })
            .sum::<f32>()
            / self.vectors.len() as f32;
        variance
    }
}

/// Optical flow estimator using a simplified block-matching approach.
///
/// Uses Sum of Absolute Differences (SAD) on luma (Y) plane.
#[derive(Debug)]
pub struct BlockMatchingEstimator {
    /// Block size in pixels.
    pub block_size: usize,
    /// Search radius in pixels.
    pub search_radius: usize,
    /// Previous frame luma data.
    prev_frame: Option<Vec<u8>>,
    /// Previous frame dimensions.
    prev_dims: Option<(usize, usize)>,
}

impl BlockMatchingEstimator {
    /// Create a new estimator.
    #[must_use]
    pub fn new(block_size: usize, search_radius: usize) -> Self {
        Self {
            block_size,
            search_radius,
            prev_frame: None,
            prev_dims: None,
        }
    }

    /// Estimate motion field between the previous and current frames.
    ///
    /// Returns None if no previous frame is available.
    pub fn estimate(&mut self, y_plane: &[u8], width: usize, height: usize) -> Option<MotionField> {
        let result =
            if let (Some(prev), Some((pw, ph))) = (self.prev_frame.as_deref(), self.prev_dims) {
                if pw == width && ph == height {
                    Some(self.block_match(prev, y_plane, width, height))
                } else {
                    None
                }
            } else {
                None
            };

        self.prev_frame = Some(y_plane.to_vec());
        self.prev_dims = Some((width, height));
        result
    }

    /// Inner block matching implementation.
    fn block_match(&self, prev: &[u8], curr: &[u8], width: usize, height: usize) -> MotionField {
        let bs = self.block_size;
        let sr = self.search_radius;
        let cols = width.div_ceil(bs);
        let rows = height.div_ceil(bs);
        let mut vectors = Vec::with_capacity(cols * rows);

        for row in 0..rows {
            for col in 0..cols {
                let bx = col * bs;
                let by = row * bs;

                let bw = bs.min(width.saturating_sub(bx));
                let bh = bs.min(height.saturating_sub(by));

                let mut best_sad = u64::MAX;
                let mut best_dx = 0i32;
                let mut best_dy = 0i32;

                let dy_start = -(sr as i32);
                let dy_end = sr as i32;
                let dx_start = -(sr as i32);
                let dx_end = sr as i32;

                let mut dy = dy_start;
                while dy <= dy_end {
                    let mut dx = dx_start;
                    while dx <= dx_end {
                        let rx = bx as i32 + dx;
                        let ry = by as i32 + dy;
                        if rx < 0 || ry < 0 {
                            dx += 1;
                            continue;
                        }
                        let rx = rx as usize;
                        let ry = ry as usize;
                        if rx + bw > width || ry + bh > height {
                            dx += 1;
                            continue;
                        }

                        let mut sad: u64 = 0;
                        for py in 0..bh {
                            for px in 0..bw {
                                let ref_val = i32::from(prev[(by + py) * width + bx + px]);
                                let cand_val = i32::from(curr[(ry + py) * width + rx + px]);
                                sad += u64::from((ref_val - cand_val).unsigned_abs());
                            }
                        }

                        if sad < best_sad {
                            best_sad = sad;
                            best_dx = dx;
                            best_dy = dy;
                        }
                        dx += 1;
                    }
                    dy += 1;
                }

                vectors.push(MotionVector::new(best_dx as f32, best_dy as f32));
            }
        }

        MotionField {
            frame_width: width,
            frame_height: height,
            block_size: bs,
            vectors,
        }
    }
}

/// Classify camera motion from a motion field.
///
/// Uses the dominant direction of the mean global vector and variance.
#[must_use]
pub fn classify_camera_motion(field: &MotionField) -> CameraMotion {
    let mean = field.mean_vector();
    let mag = mean.magnitude();
    let variance = field.magnitude_variance();

    // Static: very little motion.
    if mag < 0.5 {
        return CameraMotion::Static;
    }

    // High variance suggests complex or zoom.
    if variance > mag * mag * 2.0 {
        return CameraMotion::Complex;
    }

    let angle = mean.angle();
    let pi = std::f32::consts::PI;

    // Pan: horizontal dominant.
    if angle.abs() < pi / 4.0 || angle.abs() > 3.0 * pi / 4.0 {
        return CameraMotion::Pan;
    }

    // Tilt: vertical dominant.
    CameraMotion::Tilt
}

/// Aggregate motion statistics over multiple frames.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MotionFieldStats {
    /// Number of frames processed.
    pub frame_count: usize,
    /// Mean motion magnitude across all frames.
    pub mean_magnitude: f32,
    /// Maximum motion magnitude seen.
    pub max_magnitude: f32,
    /// Dominant camera motion type.
    pub dominant_motion: CameraMotion,
}

impl MotionFieldStats {
    /// Update statistics with a new motion field.
    pub fn update(&mut self, field: &MotionField) {
        let mag = field.mean_magnitude();
        let total_mag = self.mean_magnitude * self.frame_count as f32 + mag;
        self.frame_count += 1;
        self.mean_magnitude = total_mag / self.frame_count as f32;
        if mag > self.max_magnitude {
            self.max_magnitude = mag;
        }
        let motion = classify_camera_motion(field);
        if motion != CameraMotion::Static {
            self.dominant_motion = motion;
        }
    }
}

// ---------------------------------------------------------------------------
// Full Camera Motion Classifier
// ---------------------------------------------------------------------------

/// Detailed camera motion classification result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraMotionClassification {
    /// Primary motion type.
    pub motion_type: CameraMotion,
    /// Confidence in the classification (0.0-1.0).
    pub confidence: f32,
    /// Magnitude of the detected motion (pixels/frame).
    pub magnitude: f32,
    /// Direction angle in degrees (0 = right, 90 = down, etc.).
    pub direction_degrees: f32,
    /// Whether motion appears "handheld" (small, noisy jitter).
    pub is_handheld: bool,
}

/// Configuration for camera motion classification.
#[derive(Debug, Clone)]
pub struct CameraMotionConfig {
    /// Motion magnitude below which the camera is considered static.
    pub static_threshold: f32,
    /// Maximum magnitude to be considered handheld jitter.
    pub handheld_max_magnitude: f32,
    /// Minimum variance-to-mean ratio to flag handheld.
    pub handheld_variance_ratio: f32,
    /// Zoom detection: ratio of magnitude variance to mean magnitude
    /// above which a radial pattern is considered zoom.
    pub zoom_variance_threshold: f32,
}

impl Default for CameraMotionConfig {
    fn default() -> Self {
        Self {
            static_threshold: 0.5,
            handheld_max_magnitude: 2.0,
            handheld_variance_ratio: 0.5,
            zoom_variance_threshold: 1.5,
        }
    }
}

/// Classify camera motion from a motion field with full detail.
///
/// This implements a multi-criterion classifier:
/// 1. **Static**: very low mean magnitude.
/// 2. **Handheld**: small noisy jitter (low magnitude, high variance ratio).
/// 3. **Zoom**: radial divergence/convergence pattern from centre.
/// 4. **Pan/Tilt**: dominant horizontal or vertical translation.
/// 5. **Roll**: rotational pattern around centre.
/// 6. **Complex**: no single dominant pattern.
#[must_use]
pub fn classify_camera_motion_detailed(
    field: &MotionField,
    config: &CameraMotionConfig,
) -> CameraMotionClassification {
    let mean = field.mean_vector();
    let mag = mean.magnitude();
    let variance = field.magnitude_variance();
    let mean_mag = field.mean_magnitude();

    // 1. Static
    if mean_mag < config.static_threshold {
        return CameraMotionClassification {
            motion_type: CameraMotion::Static,
            confidence: 1.0 - (mean_mag / config.static_threshold).min(1.0),
            magnitude: mean_mag,
            direction_degrees: 0.0,
            is_handheld: false,
        };
    }

    let angle_deg = mean.angle().to_degrees();
    // Normalise to 0..360
    let direction_degrees = if angle_deg < 0.0 {
        angle_deg + 360.0
    } else {
        angle_deg
    };

    // 2. Handheld jitter
    let variance_ratio = if mean_mag > f32::EPSILON {
        variance / (mean_mag * mean_mag)
    } else {
        0.0
    };
    if mean_mag < config.handheld_max_magnitude && variance_ratio > config.handheld_variance_ratio {
        return CameraMotionClassification {
            motion_type: CameraMotion::Static,
            confidence: 0.6,
            magnitude: mean_mag,
            direction_degrees,
            is_handheld: true,
        };
    }

    // 3. Zoom detection — check for radial divergence/convergence
    let zoom_score = compute_zoom_score(field);
    if zoom_score.abs() > config.zoom_variance_threshold {
        let motion_type = if zoom_score > 0.0 {
            CameraMotion::ZoomIn
        } else {
            CameraMotion::ZoomOut
        };
        return CameraMotionClassification {
            motion_type,
            confidence: (zoom_score.abs() / 5.0).min(1.0),
            magnitude: mean_mag,
            direction_degrees,
            is_handheld: false,
        };
    }

    // 4. Roll detection — check for rotational pattern.
    // Only consider roll when the mean translation vector is small
    // relative to the mean magnitude (i.e. motion is not dominated by
    // a coherent translation).
    let translation_ratio = if mean_mag > f32::EPSILON {
        mag / mean_mag
    } else {
        0.0
    };
    let roll_score = compute_roll_score(field);
    if roll_score > 1.0 && translation_ratio < 0.5 {
        return CameraMotionClassification {
            motion_type: CameraMotion::Roll,
            confidence: (roll_score / 5.0).min(1.0),
            magnitude: mean_mag,
            direction_degrees,
            is_handheld: false,
        };
    }

    // 5. Pan/Tilt — dominant translation direction
    let pi = std::f32::consts::PI;
    let abs_angle = mean.angle().abs();

    // High variance relative to translation magnitude => complex
    if variance > mag * mag * 2.0 {
        return CameraMotionClassification {
            motion_type: CameraMotion::Complex,
            confidence: 0.5,
            magnitude: mean_mag,
            direction_degrees,
            is_handheld: false,
        };
    }

    // Pan: horizontal dominant
    if abs_angle < pi / 4.0 || abs_angle > 3.0 * pi / 4.0 {
        let confidence = (mag / mean_mag).min(1.0);
        return CameraMotionClassification {
            motion_type: CameraMotion::Pan,
            confidence,
            magnitude: mean_mag,
            direction_degrees,
            is_handheld: false,
        };
    }

    // Tilt: vertical dominant
    let confidence = (mag / mean_mag).min(1.0);
    CameraMotionClassification {
        motion_type: CameraMotion::Tilt,
        confidence,
        magnitude: mean_mag,
        direction_degrees,
        is_handheld: false,
    }
}

/// Compute a "zoom score": positive = zoom in, negative = zoom out.
///
/// Measures radial divergence from the frame centre. Vectors pointing
/// outward from the centre indicate zoom-in; inward indicates zoom-out.
fn compute_zoom_score(field: &MotionField) -> f32 {
    if field.vectors.is_empty() {
        return 0.0;
    }
    let cols = field.cols();
    let rows = field.rows();
    if cols == 0 || rows == 0 {
        return 0.0;
    }

    let cx = (cols as f32 - 1.0) / 2.0;
    let cy = (rows as f32 - 1.0) / 2.0;

    let mut radial_sum = 0.0f32;
    let mut count = 0u32;

    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            if idx >= field.vectors.len() {
                continue;
            }
            let v = &field.vectors[idx];
            let rx = c as f32 - cx;
            let ry = r as f32 - cy;
            let r_mag = (rx * rx + ry * ry).sqrt();
            if r_mag < 0.5 {
                continue; // skip centre block
            }
            // Dot product of (motion vector) and (radial direction) gives
            // radial component: positive = outward = zoom in
            let radial = (v.dx * rx + v.dy * ry) / r_mag;
            radial_sum += radial;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }
    radial_sum / count as f32
}

/// Compute a "roll score": higher means more rotational pattern.
///
/// Measures tangential (perpendicular to radial) component of motion vectors.
fn compute_roll_score(field: &MotionField) -> f32 {
    if field.vectors.is_empty() {
        return 0.0;
    }
    let cols = field.cols();
    let rows = field.rows();
    if cols == 0 || rows == 0 {
        return 0.0;
    }

    let cx = (cols as f32 - 1.0) / 2.0;
    let cy = (rows as f32 - 1.0) / 2.0;

    let mut tangential_sum = 0.0f32;
    let mut count = 0u32;

    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            if idx >= field.vectors.len() {
                continue;
            }
            let v = &field.vectors[idx];
            let rx = c as f32 - cx;
            let ry = r as f32 - cy;
            let r_mag = (rx * rx + ry * ry).sqrt();
            if r_mag < 0.5 {
                continue;
            }
            // Tangential direction is perpendicular to radial: (-ry, rx) / r_mag
            let tang = (-v.dx * ry + v.dy * rx) / r_mag;
            tangential_sum += tang.abs();
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }
    tangential_sum / count as f32
}

/// Stateful camera motion tracker that accumulates per-frame classifications.
#[derive(Debug)]
pub struct CameraMotionTracker {
    config: CameraMotionConfig,
    estimator: BlockMatchingEstimator,
    classifications: Vec<CameraMotionClassification>,
}

impl CameraMotionTracker {
    /// Create a new tracker.
    #[must_use]
    pub fn new(config: CameraMotionConfig, block_size: usize, search_radius: usize) -> Self {
        Self {
            config,
            estimator: BlockMatchingEstimator::new(block_size, search_radius),
            classifications: Vec::new(),
        }
    }

    /// Feed a Y-plane frame. Returns the classification if a motion field
    /// was produced (i.e., not the first frame).
    pub fn push_frame(
        &mut self,
        y_plane: &[u8],
        width: usize,
        height: usize,
    ) -> Option<CameraMotionClassification> {
        let field = self.estimator.estimate(y_plane, width, height)?;
        let cls = classify_camera_motion_detailed(&field, &self.config);
        self.classifications.push(cls.clone());
        Some(cls)
    }

    /// Return all classifications so far.
    #[must_use]
    pub fn classifications(&self) -> &[CameraMotionClassification] {
        &self.classifications
    }

    /// Compute dominant camera motion across all frames.
    #[must_use]
    pub fn dominant_motion(&self) -> CameraMotion {
        if self.classifications.is_empty() {
            return CameraMotion::Static;
        }

        let mut counts = [0usize; 7];
        for cls in &self.classifications {
            let idx = match cls.motion_type {
                CameraMotion::Static => 0,
                CameraMotion::Pan => 1,
                CameraMotion::Tilt => 2,
                CameraMotion::ZoomIn => 3,
                CameraMotion::ZoomOut => 4,
                CameraMotion::Roll => 5,
                CameraMotion::Complex => 6,
            };
            counts[idx] += 1;
        }

        let motions = [
            CameraMotion::Static,
            CameraMotion::Pan,
            CameraMotion::Tilt,
            CameraMotion::ZoomIn,
            CameraMotion::ZoomOut,
            CameraMotion::Roll,
            CameraMotion::Complex,
        ];

        let mut best_idx = 0;
        for (i, &c) in counts.iter().enumerate() {
            if c > counts[best_idx] {
                best_idx = i;
            }
        }
        motions[best_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_vector_magnitude() {
        let v = MotionVector::new(3.0, 4.0);
        assert!((v.magnitude() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_motion_vector_zero_magnitude() {
        assert!((MotionVector::zero().magnitude()).abs() < 1e-5);
    }

    #[test]
    fn test_motion_vector_angle() {
        let v = MotionVector::new(1.0, 0.0);
        assert!((v.angle()).abs() < 1e-5);
    }

    #[test]
    fn test_motion_field_creation() {
        let field = MotionField::new(16, 16, 8);
        assert_eq!(field.cols(), 2);
        assert_eq!(field.rows(), 2);
        assert_eq!(field.vectors.len(), 4);
    }

    #[test]
    fn test_motion_field_non_divisible_dims() {
        let field = MotionField::new(17, 17, 8);
        assert_eq!(field.cols(), 3);
        assert_eq!(field.rows(), 3);
    }

    #[test]
    fn test_motion_field_mean_zero() {
        let field = MotionField::new(16, 16, 8);
        let mean = field.mean_vector();
        assert!((mean.dx).abs() < 1e-5);
        assert!((mean.dy).abs() < 1e-5);
    }

    #[test]
    fn test_motion_field_mean_nonzero() {
        let mut field = MotionField::new(16, 16, 8);
        for v in &mut field.vectors {
            v.dx = 2.0;
            v.dy = 3.0;
        }
        let mean = field.mean_vector();
        assert!((mean.dx - 2.0).abs() < 1e-5);
        assert!((mean.dy - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_motion_field_mean_magnitude() {
        let mut field = MotionField::new(16, 16, 8);
        for v in &mut field.vectors {
            v.dx = 3.0;
            v.dy = 4.0;
        }
        assert!((field.mean_magnitude() - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_classify_static() {
        let field = MotionField::new(16, 16, 8); // all zeros
        assert_eq!(classify_camera_motion(&field), CameraMotion::Static);
    }

    #[test]
    fn test_classify_pan() {
        let mut field = MotionField::new(16, 16, 8);
        for v in &mut field.vectors {
            v.dx = 5.0;
            v.dy = 0.5;
        }
        let motion = classify_camera_motion(&field);
        assert_eq!(motion, CameraMotion::Pan);
    }

    #[test]
    fn test_classify_tilt() {
        let mut field = MotionField::new(16, 16, 8);
        for v in &mut field.vectors {
            v.dx = 0.2;
            v.dy = 5.0;
        }
        let motion = classify_camera_motion(&field);
        assert_eq!(motion, CameraMotion::Tilt);
    }

    #[test]
    fn test_block_matching_no_prev_returns_none() {
        let mut est = BlockMatchingEstimator::new(8, 4);
        let frame = vec![0u8; 64];
        let result = est.estimate(&frame, 8, 8);
        assert!(result.is_none());
    }

    #[test]
    fn test_block_matching_second_frame_returns_field() {
        let mut est = BlockMatchingEstimator::new(8, 2);
        let frame1 = vec![128u8; 64];
        let frame2 = vec![130u8; 64];
        let _ = est.estimate(&frame1, 8, 8);
        let result = est.estimate(&frame2, 8, 8);
        assert!(result.is_some());
    }

    #[test]
    fn test_motion_stats_update() {
        let mut stats = MotionFieldStats::default();
        let field = MotionField::new(16, 16, 8);
        stats.update(&field);
        assert_eq!(stats.frame_count, 1);
    }

    #[test]
    fn test_camera_motion_default() {
        assert_eq!(CameraMotion::default(), CameraMotion::Static);
    }

    // -----------------------------------------------------------------------
    // Camera motion classification tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_classify_detailed_static() {
        let field = MotionField::new(32, 32, 8);
        let config = CameraMotionConfig::default();
        let cls = classify_camera_motion_detailed(&field, &config);
        assert_eq!(cls.motion_type, CameraMotion::Static);
        assert!(cls.confidence > 0.5);
        assert!(!cls.is_handheld);
    }

    #[test]
    fn test_classify_detailed_pan() {
        let mut field = MotionField::new(32, 32, 8);
        for v in &mut field.vectors {
            v.dx = 5.0;
            v.dy = 0.3;
        }
        let config = CameraMotionConfig::default();
        let cls = classify_camera_motion_detailed(&field, &config);
        assert_eq!(cls.motion_type, CameraMotion::Pan);
        assert!(cls.magnitude > 4.0);
    }

    #[test]
    fn test_classify_detailed_tilt() {
        let mut field = MotionField::new(32, 32, 8);
        for v in &mut field.vectors {
            v.dx = 0.2;
            v.dy = 5.0;
        }
        let config = CameraMotionConfig::default();
        let cls = classify_camera_motion_detailed(&field, &config);
        assert_eq!(cls.motion_type, CameraMotion::Tilt);
    }

    #[test]
    fn test_classify_detailed_zoom_in() {
        // Radial outward vectors from centre
        let mut field = MotionField::new(32, 32, 8);
        let cols = field.cols();
        let rows = field.rows();
        let cx = (cols as f32 - 1.0) / 2.0;
        let cy = (rows as f32 - 1.0) / 2.0;
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let rx = c as f32 - cx;
                let ry = r as f32 - cy;
                let mag = (rx * rx + ry * ry).sqrt().max(0.01);
                field.vectors[idx] = MotionVector::new(rx / mag * 3.0, ry / mag * 3.0);
            }
        }
        let config = CameraMotionConfig::default();
        let cls = classify_camera_motion_detailed(&field, &config);
        assert_eq!(cls.motion_type, CameraMotion::ZoomIn);
    }

    #[test]
    fn test_classify_detailed_zoom_out() {
        // Radial inward vectors toward centre
        let mut field = MotionField::new(32, 32, 8);
        let cols = field.cols();
        let rows = field.rows();
        let cx = (cols as f32 - 1.0) / 2.0;
        let cy = (rows as f32 - 1.0) / 2.0;
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let rx = c as f32 - cx;
                let ry = r as f32 - cy;
                let mag = (rx * rx + ry * ry).sqrt().max(0.01);
                field.vectors[idx] = MotionVector::new(-rx / mag * 3.0, -ry / mag * 3.0);
            }
        }
        let config = CameraMotionConfig::default();
        let cls = classify_camera_motion_detailed(&field, &config);
        assert_eq!(cls.motion_type, CameraMotion::ZoomOut);
    }

    #[test]
    fn test_camera_motion_tracker_empty() {
        let tracker = CameraMotionTracker::new(CameraMotionConfig::default(), 8, 4);
        assert_eq!(tracker.dominant_motion(), CameraMotion::Static);
        assert!(tracker.classifications().is_empty());
    }

    #[test]
    fn test_camera_motion_tracker_first_frame_none() {
        let mut tracker = CameraMotionTracker::new(CameraMotionConfig::default(), 8, 2);
        let frame = vec![128u8; 32 * 32];
        let result = tracker.push_frame(&frame, 32, 32);
        assert!(result.is_none());
    }

    #[test]
    fn test_camera_motion_tracker_two_frames() {
        let mut tracker = CameraMotionTracker::new(CameraMotionConfig::default(), 8, 2);
        let frame1 = vec![128u8; 32 * 32];
        let frame2 = vec![130u8; 32 * 32];
        let _ = tracker.push_frame(&frame1, 32, 32);
        let result = tracker.push_frame(&frame2, 32, 32);
        assert!(result.is_some());
        assert_eq!(tracker.classifications().len(), 1);
    }

    #[test]
    fn test_zoom_score_zero_field() {
        let field = MotionField::new(32, 32, 8);
        let score = compute_zoom_score(&field);
        assert!(score.abs() < 0.01);
    }

    #[test]
    fn test_roll_score_zero_field() {
        let field = MotionField::new(32, 32, 8);
        let score = compute_roll_score(&field);
        assert!(score.abs() < 0.01);
    }

    #[test]
    fn test_classification_direction_degrees_range() {
        let mut field = MotionField::new(16, 16, 8);
        for v in &mut field.vectors {
            v.dx = -3.0;
            v.dy = 4.0;
        }
        let config = CameraMotionConfig::default();
        let cls = classify_camera_motion_detailed(&field, &config);
        assert!(cls.direction_degrees >= 0.0 && cls.direction_degrees < 360.0);
    }
}
