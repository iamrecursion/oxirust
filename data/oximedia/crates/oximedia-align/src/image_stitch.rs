//! Image stitching pipeline.
//!
//! Implements a full pipeline for stitching multiple overlapping images:
//!
//! 1. Detect features in each image (ORB / pyramid ORB for scale invariance)
//! 2. Match features between adjacent image pairs
//! 3. Estimate homography via RANSAC
//! 4. Blend overlapping regions with linear alpha blending
//!
//! The pipeline is designed to work entirely in pure Rust with no unsafe code.

#![allow(clippy::cast_precision_loss)]

use crate::features::{BinaryDescriptor, FeatureMatcher, Keypoint, OrbDetector};
use crate::spatial::{HomographyEstimator, RansacConfig};
use crate::{AlignError, AlignResult};

/// Configuration for the image stitching pipeline.
#[derive(Debug, Clone)]
pub struct StitchConfig {
    /// Maximum number of features per image.
    pub max_features: usize,
    /// Maximum Hamming distance for a valid feature match.
    pub max_match_distance: u32,
    /// Lowe's ratio test threshold.
    pub ratio_threshold: f32,
    /// RANSAC inlier threshold (pixels).
    pub ransac_threshold: f64,
    /// Minimum number of RANSAC inliers.
    pub min_inliers: usize,
    /// Blending overlap width as a fraction of image width [0, 0.5].
    pub blend_overlap: f64,
}

impl Default for StitchConfig {
    fn default() -> Self {
        Self {
            max_features: 1000,
            max_match_distance: 64,
            ratio_threshold: 0.75,
            ransac_threshold: 3.0,
            min_inliers: 10,
            blend_overlap: 0.1,
        }
    }
}

/// A single image with pre-computed features.
#[derive(Debug, Clone)]
pub struct ImageWithFeatures {
    /// Raw pixel data (grayscale, row-major).
    pub pixels: Vec<u8>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Detected keypoints.
    pub keypoints: Vec<Keypoint>,
    /// Corresponding BRIEF descriptors.
    pub descriptors: Vec<BinaryDescriptor>,
}

impl ImageWithFeatures {
    /// Extract features from a grayscale image.
    ///
    /// # Errors
    ///
    /// Returns [`AlignError`] if the pixel buffer size does not match the declared dimensions.
    pub fn from_gray(
        pixels: Vec<u8>,
        width: usize,
        height: usize,
        max_features: usize,
    ) -> AlignResult<Self> {
        if pixels.len() != width * height {
            return Err(AlignError::InvalidConfig(format!(
                "Expected {} pixels, got {}",
                width * height,
                pixels.len()
            )));
        }

        let orb = OrbDetector::new(max_features);
        let (keypoints, descriptors) = orb.detect_and_compute(&pixels, width, height)?;

        Ok(Self {
            pixels,
            width,
            height,
            keypoints,
            descriptors,
        })
    }
}

/// Pairwise homography between two images.
#[derive(Debug, Clone)]
pub struct PairHomography {
    /// Index of the source image.
    pub src_idx: usize,
    /// Index of the destination image.
    pub dst_idx: usize,
    /// Homography matrix (row-major 3×3) mapping src → dst.
    pub h: [f64; 9],
    /// Number of RANSAC inliers.
    pub num_inliers: usize,
    /// Confidence (inlier ratio).
    pub confidence: f64,
}

/// Result of the stitching pipeline.
#[derive(Debug, Clone)]
pub struct StitchedImage {
    /// Stitched pixel data (grayscale, row-major).
    pub pixels: Vec<u8>,
    /// Width of the stitched image.
    pub width: usize,
    /// Height of the stitched image.
    pub height: usize,
    /// Per-pair homographies used during stitching.
    pub homographies: Vec<PairHomography>,
}

/// Full image stitching pipeline.
pub struct ImageStitcher {
    /// Pipeline configuration.
    pub config: StitchConfig,
}

impl ImageStitcher {
    /// Create a new image stitcher with the given configuration.
    #[must_use]
    pub fn new(config: StitchConfig) -> Self {
        Self { config }
    }

    /// Stitch a sequence of overlapping images.
    ///
    /// Images should be provided in order (left to right for horizontal panoramas).
    ///
    /// # Errors
    ///
    /// Returns [`AlignError`] if fewer than two images are provided or if any
    /// stage of the pipeline fails.
    pub fn stitch(&self, images: &mut [ImageWithFeatures]) -> AlignResult<StitchedImage> {
        if images.len() < 2 {
            return Err(AlignError::InsufficientData(
                "Need at least 2 images to stitch".to_string(),
            ));
        }

        let mut homographies = Vec::new();

        // Step 1: estimate pairwise homographies between adjacent images
        for i in 0..images.len() - 1 {
            let (src, dst) = (&images[i], &images[i + 1]);
            let h = self.estimate_pairwise_homography(src, dst, i, i + 1)?;
            homographies.push(h);
        }

        // Step 2: compose homographies into a chain (accumulate from left)
        // Reference frame is the first image (identity homography).
        let cumulative = Self::chain_homographies(&homographies);

        // Step 3: compute output canvas size and warp each image
        let (canvas_w, canvas_h, offsets) = self.compute_canvas(images, &cumulative);

        // Step 4: blend images onto the canvas
        let output =
            self.blend_images_onto_canvas(images, &cumulative, &offsets, canvas_w, canvas_h);

        Ok(StitchedImage {
            pixels: output,
            width: canvas_w,
            height: canvas_h,
            homographies,
        })
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    fn estimate_pairwise_homography(
        &self,
        src: &ImageWithFeatures,
        dst: &ImageWithFeatures,
        src_idx: usize,
        dst_idx: usize,
    ) -> AlignResult<PairHomography> {
        let matcher =
            FeatureMatcher::new(self.config.max_match_distance, self.config.ratio_threshold);
        let matches = matcher.match_features(
            &src.keypoints,
            &src.descriptors,
            &dst.keypoints,
            &dst.descriptors,
        );

        if matches.len() < self.config.min_inliers {
            return Err(AlignError::InsufficientData(format!(
                "Too few matches between image {} and {}: {} < {}",
                src_idx,
                dst_idx,
                matches.len(),
                self.config.min_inliers
            )));
        }

        let ransac_config = RansacConfig {
            threshold: self.config.ransac_threshold,
            max_iterations: 1000,
            min_inliers: self.config.min_inliers,
        };

        let estimator = HomographyEstimator::new(ransac_config);
        let (h_homo, inlier_mask) = estimator.estimate(&matches)?;

        // Convert nalgebra Matrix3 → row-major [f64; 9]
        let m = &h_homo.matrix;
        let h_mat: [f64; 9] = [
            m[(0, 0)],
            m[(0, 1)],
            m[(0, 2)],
            m[(1, 0)],
            m[(1, 1)],
            m[(1, 2)],
            m[(2, 0)],
            m[(2, 1)],
            m[(2, 2)],
        ];

        let num_inliers = inlier_mask.iter().filter(|&&b| b).count();
        let confidence = num_inliers as f64 / matches.len() as f64;

        Ok(PairHomography {
            src_idx,
            dst_idx,
            h: h_mat,
            num_inliers,
            confidence,
        })
    }

    /// Compose a chain of pairwise homographies into per-image transforms
    /// relative to the first image (image 0 has identity).
    fn chain_homographies(pairs: &[PairHomography]) -> Vec<[f64; 9]> {
        let mut cumulative = Vec::with_capacity(pairs.len() + 1);
        // Image 0: identity
        cumulative.push(Self::identity_h());

        let mut current = Self::identity_h();
        for ph in pairs {
            current = Self::compose_h(&current, &ph.h);
            cumulative.push(current);
        }

        cumulative
    }

    /// Compute bounding box of the warped canvas.
    /// Returns (canvas_w, canvas_h, per-image (x_offset, y_offset)).
    fn compute_canvas(
        &self,
        images: &[ImageWithFeatures],
        homographies: &[[f64; 9]],
    ) -> (usize, usize, Vec<(f64, f64)>) {
        let mut min_x = 0.0_f64;
        let mut min_y = 0.0_f64;
        let mut max_x = 0.0_f64;
        let mut max_y = 0.0_f64;

        for (img, h) in images.iter().zip(homographies.iter()) {
            let corners = [
                (0.0f64, 0.0f64),
                (img.width as f64, 0.0),
                (img.width as f64, img.height as f64),
                (0.0, img.height as f64),
            ];
            for (cx, cy) in &corners {
                let (px, py) = Self::project_h(h, *cx, *cy);
                min_x = min_x.min(px);
                min_y = min_y.min(py);
                max_x = max_x.max(px);
                max_y = max_y.max(py);
            }
        }

        let canvas_w = (max_x - min_x).ceil() as usize + 1;
        let canvas_h = (max_y - min_y).ceil() as usize + 1;

        // Compute per-image offset = (-min_x, -min_y)
        let offsets = vec![(-min_x, -min_y); images.len()];

        (canvas_w.max(1), canvas_h.max(1), offsets)
    }

    /// Warp and blend images onto the canvas using backward mapping.
    fn blend_images_onto_canvas(
        &self,
        images: &[ImageWithFeatures],
        homographies: &[[f64; 9]],
        offsets: &[(f64, f64)],
        canvas_w: usize,
        canvas_h: usize,
    ) -> Vec<u8> {
        let n = canvas_w * canvas_h;
        let mut canvas = vec![0u8; n];
        let mut weights = vec![0.0f64; n];

        for (img_idx, (img, h)) in images.iter().zip(homographies.iter()).enumerate() {
            let (ox, oy) = offsets[img_idx];

            // Compute inverse homography for backward mapping
            let inv_h = match Self::invert_h(h) {
                Some(inv) => inv,
                None => continue,
            };

            // Compute projected bounding box for this image
            let corners = [
                (0.0f64, 0.0f64),
                (img.width as f64, 0.0),
                (img.width as f64, img.height as f64),
                (0.0, img.height as f64),
            ];
            let (mut min_cx, mut min_cy) = (f64::MAX, f64::MAX);
            let (mut max_cx, mut max_cy) = (f64::MIN, f64::MIN);
            for (cx, cy) in &corners {
                let (px, py) = Self::project_h(h, *cx, *cy);
                let px = px + ox;
                let py = py + oy;
                min_cx = min_cx.min(px);
                min_cy = min_cy.min(py);
                max_cx = max_cx.max(px);
                max_cy = max_cy.max(py);
            }

            let x0 = (min_cx.floor() as isize).max(0) as usize;
            let y0 = (min_cy.floor() as isize).max(0) as usize;
            let x1 = (max_cx.ceil() as usize + 1).min(canvas_w);
            let y1 = (max_cy.ceil() as usize + 1).min(canvas_h);

            for cy in y0..y1 {
                for cx in x0..x1 {
                    // Back-project canvas pixel to source image
                    let wx = cx as f64 - ox;
                    let wy = cy as f64 - oy;
                    let (sx, sy) = Self::project_h(&inv_h, wx, wy);

                    // Check if within source image bounds
                    if sx < 0.0
                        || sy < 0.0
                        || sx >= img.width as f64 - 1.0
                        || sy >= img.height as f64 - 1.0
                    {
                        continue;
                    }

                    // Bilinear interpolation
                    let pixel = Self::bilinear_sample(&img.pixels, img.width, img.height, sx, sy);

                    // Compute blending weight: distance from image border
                    let weight = Self::blend_weight(
                        sx,
                        sy,
                        img.width,
                        img.height,
                        self.config.blend_overlap,
                    );

                    let out_idx = cy * canvas_w + cx;
                    canvas[out_idx] = ((f64::from(canvas[out_idx]) * weights[out_idx]
                        + f64::from(pixel) * weight)
                        / (weights[out_idx] + weight + 1e-15))
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    weights[out_idx] += weight;
                }
            }
        }

        canvas
    }

    /// Compute a blend weight based on distance to the image border.
    fn blend_weight(x: f64, y: f64, w: usize, h: usize, overlap: f64) -> f64 {
        let margin_x = w as f64 * overlap;
        let margin_y = h as f64 * overlap;

        let wx = (x / margin_x)
            .min((w as f64 - x) / margin_x)
            .min(1.0)
            .max(0.0);
        let wy = (y / margin_y)
            .min((h as f64 - y) / margin_y)
            .min(1.0)
            .max(0.0);

        (wx * wy).max(1e-6)
    }

    /// Bilinear sample from a grayscale image.
    fn bilinear_sample(pixels: &[u8], width: usize, height: usize, x: f64, y: f64) -> u8 {
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);

        let fx = x - x0 as f64;
        let fy = y - y0 as f64;

        let i00 = f64::from(pixels[y0 * width + x0]);
        let i10 = f64::from(pixels[y0 * width + x1]);
        let i01 = f64::from(pixels[y1 * width + x0]);
        let i11 = f64::from(pixels[y1 * width + x1]);

        let val = i00 * (1.0 - fx) * (1.0 - fy)
            + i10 * fx * (1.0 - fy)
            + i01 * (1.0 - fx) * fy
            + i11 * fx * fy;

        val.round().clamp(0.0, 255.0) as u8
    }

    /// Compose two homographies: result = h2 ∘ h1 (first apply h1, then h2).
    fn compose_h(h1: &[f64; 9], h2: &[f64; 9]) -> [f64; 9] {
        let mut result = [0.0_f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut val = 0.0;
                for k in 0..3 {
                    val += h2[i * 3 + k] * h1[k * 3 + j];
                }
                result[i * 3 + j] = val;
            }
        }
        result
    }

    /// Project a point through a homography matrix.
    fn project_h(h: &[f64; 9], x: f64, y: f64) -> (f64, f64) {
        let w = h[6] * x + h[7] * y + h[8];
        if w.abs() < 1e-14 {
            return (x, y);
        }
        (
            (h[0] * x + h[1] * y + h[2]) / w,
            (h[3] * x + h[4] * y + h[5]) / w,
        )
    }

    /// Identity homography.
    fn identity_h() -> [f64; 9] {
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    }

    /// Invert a 3×3 homography matrix (returns None if singular).
    fn invert_h(h: &[f64; 9]) -> Option<[f64; 9]> {
        let det = h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6])
            + h[2] * (h[3] * h[7] - h[4] * h[6]);

        if det.abs() < 1e-14 {
            return None;
        }

        let inv_det = 1.0 / det;

        Some([
            (h[4] * h[8] - h[5] * h[7]) * inv_det,
            (h[2] * h[7] - h[1] * h[8]) * inv_det,
            (h[1] * h[5] - h[2] * h[4]) * inv_det,
            (h[5] * h[6] - h[3] * h[8]) * inv_det,
            (h[0] * h[8] - h[2] * h[6]) * inv_det,
            (h[2] * h[3] - h[0] * h[5]) * inv_det,
            (h[3] * h[7] - h[4] * h[6]) * inv_det,
            (h[1] * h[6] - h[0] * h[7]) * inv_det,
            (h[0] * h[4] - h[1] * h[3]) * inv_det,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_image(w: usize, h: usize, offset_x: u8) -> ImageWithFeatures {
        let mut pixels = vec![0u8; w * h];
        // Add a checkerboard-like pattern for feature detection
        for y in 0..h {
            for x in 0..w {
                let val = if (x / 8 + y / 8) % 2 == 0 {
                    (200_u16 + u16::from(offset_x)).min(255) as u8
                } else {
                    50
                };
                pixels[y * w + x] = val;
            }
        }
        // Add some corner-like features
        let add_corner = |pixels: &mut Vec<u8>, cx: usize, cy: usize| {
            for dy in 0..5usize {
                for dx in 0..5usize {
                    if cx + dx < w && cy + dy < h {
                        pixels[(cy + dy) * w + (cx + dx)] = 255;
                    }
                }
            }
        };
        for i in [10, 30, 50, 70, 90] {
            for j in [10, 30, 50, 70, 90] {
                if i < w && j < h {
                    add_corner(&mut pixels, i, j);
                }
            }
        }
        ImageWithFeatures {
            pixels,
            width: w,
            height: h,
            keypoints: Vec::new(),
            descriptors: Vec::new(),
        }
    }

    #[test]
    fn test_stitch_config_default() {
        let c = StitchConfig::default();
        assert_eq!(c.max_features, 1000);
        assert!(c.blend_overlap > 0.0);
    }

    #[test]
    fn test_image_with_features_size_mismatch() {
        // 10 bytes != 5*5=25 → should return an error.
        let result = ImageWithFeatures::from_gray(vec![0u8; 10], 5, 5, 100);
        assert!(
            result.is_err(),
            "5*5=25 but only 10 bytes → mismatch should fail"
        );
    }

    #[test]
    fn test_image_with_features_exact_size() {
        let result = ImageWithFeatures::from_gray(vec![128u8; 64 * 64], 64, 64, 100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_identity_homography_projection() {
        let h = ImageStitcher::identity_h();
        let (px, py) = ImageStitcher::project_h(&h, 10.0, 20.0);
        assert!((px - 10.0).abs() < 1e-10);
        assert!((py - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_invert_identity() {
        let h = ImageStitcher::identity_h();
        let inv = ImageStitcher::invert_h(&h).expect("identity should be invertible");
        // Inverse of identity should be identity
        for (a, b) in inv.iter().zip(h.iter()) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
    }

    #[test]
    fn test_compose_identity() {
        let h = ImageStitcher::identity_h();
        let composed = ImageStitcher::compose_h(&h, &h);
        for (a, b) in composed.iter().zip(h.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_invert_singular_returns_none() {
        let singular = [0.0_f64; 9];
        assert!(ImageStitcher::invert_h(&singular).is_none());
    }

    #[test]
    fn test_bilinear_sample_corner() {
        let pixels = vec![0u8, 100, 200, 50];
        let val = ImageStitcher::bilinear_sample(&pixels, 2, 2, 0.5, 0.5);
        let expected = (0.0 + 100.0 + 200.0 + 50.0) / 4.0;
        assert!((f64::from(val) - expected).abs() < 2.0);
    }

    #[test]
    fn test_chain_homographies_single() {
        let pairs = vec![PairHomography {
            src_idx: 0,
            dst_idx: 1,
            h: ImageStitcher::identity_h(),
            num_inliers: 10,
            confidence: 1.0,
        }];
        let chain = ImageStitcher::chain_homographies(&pairs);
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn test_stitcher_requires_two_images() {
        let stitcher = ImageStitcher::new(StitchConfig::default());
        let result = stitcher.stitch(&mut []);
        assert!(result.is_err());
        let mut single = vec![
            ImageWithFeatures::from_gray(vec![128u8; 32 * 32], 32, 32, 100)
                .expect("should succeed"),
        ];
        let result2 = stitcher.stitch(&mut single);
        assert!(result2.is_err());
    }

    #[test]
    fn test_blend_weight_center() {
        let w = blend_weight_helper(50.0, 50.0, 100, 100, 0.1);
        // Center should have weight 1.0
        assert!(w > 0.9, "center weight should be near 1: {w}");
    }

    fn blend_weight_helper(x: f64, y: f64, w: usize, h: usize, overlap: f64) -> f64 {
        ImageStitcher::blend_weight(x, y, w, h, overlap)
    }

    #[test]
    fn test_make_test_image_dimensions() {
        let img = make_test_image(64, 32, 0);
        assert_eq!(img.width, 64);
        assert_eq!(img.height, 32);
        assert_eq!(img.pixels.len(), 64 * 32);
    }
}
