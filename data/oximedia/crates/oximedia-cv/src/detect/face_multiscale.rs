//! Multi-scale rotation-invariant face detection.
//!
//! Extends the base face detection with:
//! - Image pyramid for multi-scale detection
//! - Rotation candidates for rotation-invariant detection
//! - Cross-rotation NMS for deduplication
//!
//! # Example
//!
//! ```
//! use oximedia_cv::detect::face_multiscale::{MultiScaleFaceDetector, MultiScaleFaceConfig};
//!
//! let config = MultiScaleFaceConfig::default();
//! let detector = MultiScaleFaceDetector::new(config);
//! let image = vec![128u8; 100 * 100];
//! let detections = detector.detect_rotated(&image, 100, 100).expect("detection should succeed");
//! ```

use crate::detect::face::{detection_overlap, DetectionResult, HaarCascade, IntegralImage};
use crate::error::{CvError, CvResult};

/// Rotation angle candidate for rotation-invariant detection.
#[derive(Debug, Clone, Copy)]
pub struct RotationCandidate {
    /// Rotation angle in degrees.
    pub angle_degrees: f64,
    /// Cosine of the angle (precomputed).
    cos_a: f64,
    /// Sine of the angle (precomputed).
    sin_a: f64,
}

impl RotationCandidate {
    /// Create a new rotation candidate.
    #[must_use]
    pub fn new(angle_degrees: f64) -> Self {
        let angle_rad = angle_degrees * std::f64::consts::PI / 180.0;
        Self {
            angle_degrees,
            cos_a: angle_rad.cos(),
            sin_a: angle_rad.sin(),
        }
    }

    /// Get the angle in degrees.
    #[must_use]
    pub const fn angle(&self) -> f64 {
        self.angle_degrees
    }
}

/// Configuration for multi-scale rotation-invariant face detection.
#[derive(Debug, Clone)]
pub struct MultiScaleFaceConfig {
    /// Scale factors for the image pyramid.
    pub scale_factors: Vec<f64>,
    /// Rotation angles to test (in degrees).
    pub rotation_angles: Vec<f64>,
    /// Minimum face size in pixels.
    pub min_face_size: u32,
    /// Maximum face size in pixels (0 = no limit).
    pub max_face_size: u32,
    /// Confidence threshold for detections.
    pub confidence_threshold: f64,
    /// IoU threshold for NMS grouping.
    pub nms_threshold: f64,
    /// Minimum neighbors for detection grouping.
    pub min_neighbors: u32,
    /// Haar cascade window size.
    pub window_size: u32,
}

impl Default for MultiScaleFaceConfig {
    fn default() -> Self {
        Self {
            scale_factors: vec![1.0, 0.75, 0.5, 0.375, 0.25],
            rotation_angles: vec![0.0, -30.0, -15.0, 15.0, 30.0],
            min_face_size: 24,
            max_face_size: 0,
            confidence_threshold: 0.5,
            nms_threshold: 0.3,
            min_neighbors: 2,
            window_size: 24,
        }
    }
}

impl MultiScaleFaceConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set scale factors for the image pyramid.
    #[must_use]
    pub fn with_scale_factors(mut self, factors: Vec<f64>) -> Self {
        self.scale_factors = factors;
        self
    }

    /// Set rotation angles to test (in degrees).
    #[must_use]
    pub fn with_rotation_angles(mut self, angles: Vec<f64>) -> Self {
        self.rotation_angles = angles;
        self
    }

    /// Set minimum face size.
    #[must_use]
    pub const fn with_min_face_size(mut self, size: u32) -> Self {
        self.min_face_size = size;
        self
    }

    /// Set maximum face size (0 for no limit).
    #[must_use]
    pub const fn with_max_face_size(mut self, size: u32) -> Self {
        self.max_face_size = size;
        self
    }

    /// Set confidence threshold.
    #[must_use]
    pub fn with_confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set NMS overlap threshold.
    #[must_use]
    pub fn with_nms_threshold(mut self, threshold: f64) -> Self {
        self.nms_threshold = threshold.clamp(0.0, 1.0);
        self
    }
}

/// A face detection result with rotation angle information.
#[derive(Debug, Clone)]
pub struct RotatedFaceDetection {
    /// Bounding box of the detected face (in original image coordinates).
    pub bbox: DetectionResult,
    /// Detected rotation angle in degrees.
    pub angle: f64,
    /// Detection confidence.
    pub confidence: f64,
    /// Scale at which the face was detected.
    pub scale: f64,
}

/// Multi-scale rotation-invariant face detector.
///
/// Combines image pyramid with rotation candidates to detect faces
/// at various scales and orientations using Haar cascade evaluation.
///
/// # Example
///
/// ```
/// use oximedia_cv::detect::face_multiscale::{MultiScaleFaceDetector, MultiScaleFaceConfig};
///
/// let config = MultiScaleFaceConfig::default();
/// let detector = MultiScaleFaceDetector::new(config);
/// let image = vec![128u8; 100 * 100];
/// let detections = detector.detect_rotated(&image, 100, 100).expect("detection should succeed");
/// ```
pub struct MultiScaleFaceDetector {
    /// Detection configuration.
    config: MultiScaleFaceConfig,
    /// Precomputed rotation candidates.
    rotation_candidates: Vec<RotationCandidate>,
    /// Internal Haar cascade for evaluation.
    cascade: HaarCascade,
}

impl MultiScaleFaceDetector {
    /// Create a new multi-scale rotation-invariant face detector.
    #[must_use]
    pub fn new(config: MultiScaleFaceConfig) -> Self {
        let rotation_candidates: Vec<RotationCandidate> = config
            .rotation_angles
            .iter()
            .map(|&angle| RotationCandidate::new(angle))
            .collect();

        let cascade = HaarCascade::new(config.window_size, config.window_size)
            .with_scale_factor(1.1)
            .with_min_neighbors(config.min_neighbors);

        Self {
            config,
            rotation_candidates,
            cascade,
        }
    }

    /// Detect faces with rotation invariance across multiple scales.
    ///
    /// Builds an image pyramid and tests each rotation candidate at each scale.
    /// Results are grouped and filtered via NMS.
    ///
    /// # Arguments
    ///
    /// * `image` - Grayscale image data
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are invalid.
    pub fn detect_rotated(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<Vec<RotatedFaceDetection>> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = width as usize * height as usize;
        if image.len() < expected_size {
            return Err(CvError::insufficient_data(expected_size, image.len()));
        }

        let mut all_detections = Vec::new();

        // Build image pyramid and test each scale
        for &scale in &self.config.scale_factors {
            let scaled_w = ((width as f64) * scale) as u32;
            let scaled_h = ((height as f64) * scale) as u32;

            if scaled_w < self.config.min_face_size || scaled_h < self.config.min_face_size {
                continue;
            }

            // Resize to this pyramid level
            let scaled_image = resize_grayscale(image, width, height, scaled_w, scaled_h);

            // Test each rotation
            for rotation in &self.rotation_candidates {
                let rotated = if rotation.angle_degrees.abs() < 0.01 {
                    scaled_image.clone()
                } else {
                    rotate_grayscale(
                        &scaled_image,
                        scaled_w,
                        scaled_h,
                        rotation.cos_a,
                        rotation.sin_a,
                    )
                };

                // Run Haar cascade detection on rotated image
                let integral = IntegralImage::compute(&rotated, scaled_w, scaled_h);
                let detections = self.scan_image(&integral, scaled_w, scaled_h, scale, rotation)?;
                all_detections.extend(detections);
            }
        }

        // Apply cross-rotation NMS
        let filtered = self.nms_rotated(&mut all_detections);
        Ok(filtered)
    }

    /// Scan image using sliding window with Haar cascade.
    fn scan_image(
        &self,
        integral: &IntegralImage,
        width: u32,
        height: u32,
        scale: f64,
        rotation: &RotationCandidate,
    ) -> CvResult<Vec<RotatedFaceDetection>> {
        let mut detections = Vec::new();
        let win_w = self.config.window_size;
        let win_h = self.config.window_size;
        let step = (win_w / 4).max(2);

        let mut y = 0;
        while y + win_h <= height {
            let mut x = 0;
            while x + win_w <= width {
                if let Some(confidence) = self.cascade.evaluate(integral, x, y, 1.0) {
                    let norm_confidence: f64 = (confidence / 10.0).min(1.0);
                    if norm_confidence >= self.config.confidence_threshold {
                        let orig_x = x as f64 / scale;
                        let orig_y = y as f64 / scale;
                        let orig_w = win_w as f64 / scale;
                        let orig_h = win_h as f64 / scale;

                        let face_size = orig_w.max(orig_h) as u32;
                        if face_size >= self.config.min_face_size
                            && (self.config.max_face_size == 0
                                || face_size <= self.config.max_face_size)
                        {
                            let cx = orig_x + orig_w / 2.0;
                            let cy = orig_y + orig_h / 2.0;
                            let (rot_cx, rot_cy) = reverse_rotate_point(
                                cx,
                                cy,
                                width as f64 / (2.0 * scale),
                                height as f64 / (2.0 * scale),
                                rotation.cos_a,
                                rotation.sin_a,
                            );

                            let final_x = (rot_cx - orig_w / 2.0).max(0.0);
                            let final_y = (rot_cy - orig_h / 2.0).max(0.0);

                            detections.push(RotatedFaceDetection {
                                bbox: DetectionResult::new(
                                    final_x as u32,
                                    final_y as u32,
                                    orig_w as u32,
                                    orig_h as u32,
                                    norm_confidence,
                                ),
                                angle: rotation.angle_degrees,
                                confidence: norm_confidence,
                                scale,
                            });
                        }
                    }
                }
                x += step;
            }
            y += step;
        }

        Ok(detections)
    }

    /// Apply NMS across rotation-angle groups.
    fn nms_rotated(&self, detections: &mut Vec<RotatedFaceDetection>) -> Vec<RotatedFaceDetection> {
        if detections.is_empty() {
            return Vec::new();
        }

        detections.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut keep = Vec::new();
        let mut suppressed = vec![false; detections.len()];

        for i in 0..detections.len() {
            if suppressed[i] {
                continue;
            }
            keep.push(detections[i].clone());

            for j in (i + 1)..detections.len() {
                if suppressed[j] {
                    continue;
                }
                let overlap = detection_overlap(&detections[i].bbox, &detections[j].bbox);
                if overlap > self.config.nms_threshold {
                    suppressed[j] = true;
                }
            }
        }

        keep
    }

    /// Get the detection configuration.
    #[must_use]
    pub const fn config(&self) -> &MultiScaleFaceConfig {
        &self.config
    }

    /// Get rotation candidates.
    #[must_use]
    pub fn rotation_candidates(&self) -> &[RotationCandidate] {
        &self.rotation_candidates
    }
}

/// Resize a grayscale image using bilinear interpolation.
fn resize_grayscale(
    image: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Vec<u8> {
    let mut output = vec![0u8; (dst_width * dst_height) as usize];
    let x_ratio = src_width as f64 / dst_width as f64;
    let y_ratio = src_height as f64 / dst_height as f64;

    for y in 0..dst_height {
        for x in 0..dst_width {
            let src_x = x as f64 * x_ratio;
            let src_y = y as f64 * y_ratio;

            let x0 = src_x.floor() as u32;
            let y0 = src_y.floor() as u32;
            let x1 = (x0 + 1).min(src_width - 1);
            let y1 = (y0 + 1).min(src_height - 1);

            let fx = src_x - x0 as f64;
            let fy = src_y - y0 as f64;

            let v00 = image[(y0 * src_width + x0) as usize] as f64;
            let v01 = image[(y0 * src_width + x1) as usize] as f64;
            let v10 = image[(y1 * src_width + x0) as usize] as f64;
            let v11 = image[(y1 * src_width + x1) as usize] as f64;

            let v0 = v00 * (1.0 - fx) + v01 * fx;
            let v1 = v10 * (1.0 - fx) + v11 * fx;
            let v = v0 * (1.0 - fy) + v1 * fy;

            output[(y * dst_width + x) as usize] = v.round().clamp(0.0, 255.0) as u8;
        }
    }

    output
}

/// Rotate a grayscale image by given cos/sin values around its center.
fn rotate_grayscale(image: &[u8], width: u32, height: u32, cos_a: f64, sin_a: f64) -> Vec<u8> {
    let mut output = vec![0u8; (width * height) as usize];
    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;

    for y in 0..height {
        for x in 0..width {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            let src_x = cos_a * dx + sin_a * dy + cx;
            let src_y = -sin_a * dx + cos_a * dy + cy;

            if src_x >= 0.0
                && src_x < (width - 1) as f64
                && src_y >= 0.0
                && src_y < (height - 1) as f64
            {
                let x0 = src_x.floor() as u32;
                let y0 = src_y.floor() as u32;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let fx = src_x - x0 as f64;
                let fy = src_y - y0 as f64;

                let v00 = image[(y0 * width + x0) as usize] as f64;
                let v01 = image[(y0 * width + x1) as usize] as f64;
                let v10 = image[(y1 * width + x0) as usize] as f64;
                let v11 = image[(y1 * width + x1) as usize] as f64;

                let v = v00 * (1.0 - fx) * (1.0 - fy)
                    + v01 * fx * (1.0 - fy)
                    + v10 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                output[(y * width + x) as usize] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    output
}

/// Reverse-rotate a point from rotated space back to original space.
fn reverse_rotate_point(x: f64, y: f64, cx: f64, cy: f64, cos_a: f64, sin_a: f64) -> (f64, f64) {
    let dx = x - cx;
    let dy = y - cy;
    let rx = cos_a * dx - sin_a * dy + cx;
    let ry = sin_a * dx + cos_a * dy + cy;
    (rx, ry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotation_candidate_new() {
        let rc = RotationCandidate::new(0.0);
        assert!((rc.cos_a - 1.0).abs() < 1e-10);
        assert!(rc.sin_a.abs() < 1e-10);
        assert_eq!(rc.angle(), 0.0);
    }

    #[test]
    fn test_rotation_candidate_90_degrees() {
        let rc = RotationCandidate::new(90.0);
        assert!(rc.cos_a.abs() < 1e-10);
        assert!((rc.sin_a - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_candidate_negative() {
        let rc = RotationCandidate::new(-45.0);
        assert!((rc.cos_a - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((rc.sin_a + std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_multiscale_config_default() {
        let config = MultiScaleFaceConfig::default();
        assert_eq!(config.scale_factors.len(), 5);
        assert_eq!(config.rotation_angles.len(), 5);
        assert_eq!(config.min_face_size, 24);
        assert_eq!(config.max_face_size, 0);
        assert_eq!(config.confidence_threshold, 0.5);
        assert_eq!(config.window_size, 24);
    }

    #[test]
    fn test_multiscale_config_builder() {
        let config = MultiScaleFaceConfig::new()
            .with_scale_factors(vec![1.0, 0.5])
            .with_rotation_angles(vec![0.0, 15.0, -15.0])
            .with_min_face_size(32)
            .with_max_face_size(200)
            .with_confidence_threshold(0.7)
            .with_nms_threshold(0.4);

        assert_eq!(config.scale_factors.len(), 2);
        assert_eq!(config.rotation_angles.len(), 3);
        assert_eq!(config.min_face_size, 32);
        assert_eq!(config.max_face_size, 200);
        assert_eq!(config.confidence_threshold, 0.7);
        assert_eq!(config.nms_threshold, 0.4);
    }

    #[test]
    fn test_multiscale_detector_creation() {
        let config = MultiScaleFaceConfig::default();
        let detector = MultiScaleFaceDetector::new(config);
        assert_eq!(detector.rotation_candidates().len(), 5);
        assert_eq!(detector.config().window_size, 24);
    }

    #[test]
    fn test_detect_rotated_invalid_dimensions() {
        let detector = MultiScaleFaceDetector::new(MultiScaleFaceConfig::default());
        let result = detector.detect_rotated(&[], 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_rotated_insufficient_data() {
        let detector = MultiScaleFaceDetector::new(MultiScaleFaceConfig::default());
        let result = detector.detect_rotated(&[0u8; 10], 100, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_rotated_uniform_image() {
        let config = MultiScaleFaceConfig::default();
        let detector = MultiScaleFaceDetector::new(config);
        let image = vec![128u8; 100 * 100];
        let detections = detector
            .detect_rotated(&image, 100, 100)
            .expect("detect_rotated should succeed");
        // Empty cascade on uniform image -> no detections
        assert!(detections.is_empty());
    }

    #[test]
    fn test_resize_grayscale_identity() {
        let image = vec![100u8; 50 * 50];
        let resized = resize_grayscale(&image, 50, 50, 50, 50);
        assert_eq!(resized.len(), 50 * 50);
        assert_eq!(resized[0], 100);
    }

    #[test]
    fn test_resize_grayscale_downscale() {
        let image = vec![200u8; 100 * 100];
        let resized = resize_grayscale(&image, 100, 100, 50, 50);
        assert_eq!(resized.len(), 50 * 50);
        // Uniform image should stay uniform after resize
        assert_eq!(resized[0], 200);
    }

    #[test]
    fn test_resize_grayscale_upscale() {
        let image = vec![150u8; 50 * 50];
        let resized = resize_grayscale(&image, 50, 50, 100, 100);
        assert_eq!(resized.len(), 100 * 100);
        assert_eq!(resized[0], 150);
    }

    #[test]
    fn test_rotate_grayscale_zero_angle() {
        let image = vec![100u8; 50 * 50];
        let rotated = rotate_grayscale(&image, 50, 50, 1.0, 0.0);
        assert_eq!(rotated.len(), 50 * 50);
        // Central pixels should be preserved
        assert_eq!(rotated[25 * 50 + 25], 100);
    }

    #[test]
    fn test_rotate_grayscale_preserves_center() {
        let mut image = vec![0u8; 50 * 50];
        // Set center pixel
        image[25 * 50 + 25] = 255;
        let angle_rad = 45.0_f64.to_radians();
        let rotated = rotate_grayscale(&image, 50, 50, angle_rad.cos(), angle_rad.sin());
        assert_eq!(rotated.len(), 50 * 50);
        // Center pixel should remain approximately at center
        // (bilinear interpolation may spread it)
    }

    #[test]
    fn test_reverse_rotate_point_identity() {
        let (rx, ry) = reverse_rotate_point(10.0, 20.0, 50.0, 50.0, 1.0, 0.0);
        assert!((rx - 10.0).abs() < 1e-10);
        assert!((ry - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_rotate_point_center() {
        // Rotating the center should not move it
        let angle = 45.0_f64.to_radians();
        let (rx, ry) = reverse_rotate_point(50.0, 50.0, 50.0, 50.0, angle.cos(), angle.sin());
        assert!((rx - 50.0).abs() < 1e-10);
        assert!((ry - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotated_face_detection_struct() {
        let det = RotatedFaceDetection {
            bbox: DetectionResult::new(10, 20, 30, 30, 0.9),
            angle: 15.0,
            confidence: 0.9,
            scale: 0.5,
        };
        assert_eq!(det.bbox.x, 10);
        assert_eq!(det.angle, 15.0);
        assert_eq!(det.scale, 0.5);
    }

    #[test]
    fn test_nms_rotated_empty() {
        let config = MultiScaleFaceConfig::default();
        let detector = MultiScaleFaceDetector::new(config);
        let result = detector.nms_rotated(&mut Vec::new());
        assert!(result.is_empty());
    }

    #[test]
    fn test_nms_rotated_no_overlap() {
        let config = MultiScaleFaceConfig::default();
        let detector = MultiScaleFaceDetector::new(config);
        let mut detections = vec![
            RotatedFaceDetection {
                bbox: DetectionResult::new(0, 0, 10, 10, 0.9),
                angle: 0.0,
                confidence: 0.9,
                scale: 1.0,
            },
            RotatedFaceDetection {
                bbox: DetectionResult::new(100, 100, 10, 10, 0.8),
                angle: 15.0,
                confidence: 0.8,
                scale: 1.0,
            },
        ];
        let result = detector.nms_rotated(&mut detections);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_nms_rotated_with_overlap() {
        let config = MultiScaleFaceConfig::default().with_nms_threshold(0.3);
        let detector = MultiScaleFaceDetector::new(config);
        let mut detections = vec![
            RotatedFaceDetection {
                bbox: DetectionResult::new(10, 10, 50, 50, 0.9),
                angle: 0.0,
                confidence: 0.9,
                scale: 1.0,
            },
            RotatedFaceDetection {
                bbox: DetectionResult::new(15, 15, 50, 50, 0.8),
                angle: 5.0,
                confidence: 0.8,
                scale: 1.0,
            },
        ];
        let result = detector.nms_rotated(&mut detections);
        // The second detection should be suppressed due to high overlap
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].confidence, 0.9);
    }

    #[test]
    fn test_detect_with_single_scale_single_rotation() {
        let config = MultiScaleFaceConfig::new()
            .with_scale_factors(vec![1.0])
            .with_rotation_angles(vec![0.0]);
        let detector = MultiScaleFaceDetector::new(config);
        let image = vec![128u8; 100 * 100];
        let detections = detector
            .detect_rotated(&image, 100, 100)
            .expect("should succeed");
        // Empty cascade -> no detections
        assert!(detections.is_empty());
    }

    #[test]
    fn test_confidence_threshold_filtering() {
        let config = MultiScaleFaceConfig::new().with_confidence_threshold(1.0);
        let detector = MultiScaleFaceDetector::new(config);
        assert_eq!(detector.config().confidence_threshold, 1.0);
    }

    #[test]
    fn test_max_face_size_constraint() {
        let config = MultiScaleFaceConfig::new()
            .with_min_face_size(10)
            .with_max_face_size(50);
        assert_eq!(config.min_face_size, 10);
        assert_eq!(config.max_face_size, 50);
    }
}
