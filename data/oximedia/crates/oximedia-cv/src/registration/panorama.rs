//! Panorama stitching using feature matching and homography estimation.
//!
//! This module provides panorama construction from overlapping images
//! by detecting features, matching them, computing homographies via RANSAC,
//! and blending the warped images together.

use super::feature_based::{
    ransac_homography, FeatureDetector, FeatureDetectorType, FeatureMatcher,
};
use super::{TransformMatrix, TransformationType};
use crate::error::{CvError, CvResult};

/// Configuration for panorama stitching.
#[derive(Debug, Clone)]
pub struct PanoramaConfig {
    /// Maximum features per image.
    pub max_features: usize,
    /// RANSAC reprojection threshold (pixels).
    pub ransac_threshold: f64,
    /// RANSAC max iterations.
    pub ransac_iterations: usize,
    /// RANSAC confidence level.
    pub ransac_confidence: f64,
    /// Blending band width (pixels) for linear feathering.
    pub blend_width: usize,
    /// Minimum inlier ratio to accept a match.
    pub min_inlier_ratio: f64,
}

impl Default for PanoramaConfig {
    fn default() -> Self {
        Self {
            max_features: 500,
            ransac_threshold: 3.0,
            ransac_iterations: 2000,
            ransac_confidence: 0.99,
            blend_width: 32,
            min_inlier_ratio: 0.25,
        }
    }
}

/// Result of pairwise image matching for panorama.
#[derive(Debug, Clone)]
pub struct PairwiseMatch {
    /// Index of the first image.
    pub idx_a: usize,
    /// Index of the second image.
    pub idx_b: usize,
    /// Homography from image B into image A's coordinate frame.
    pub homography: TransformMatrix,
    /// Number of inlier matches.
    pub inlier_count: usize,
    /// Inlier ratio.
    pub inlier_ratio: f64,
    /// Confidence score.
    pub confidence: f64,
}

/// Panorama stitcher for combining multiple overlapping grayscale images.
#[derive(Debug)]
pub struct PanoramaStitcher {
    config: PanoramaConfig,
}

impl PanoramaStitcher {
    /// Create a new panorama stitcher with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: PanoramaConfig::default(),
        }
    }

    /// Create a new panorama stitcher with custom configuration.
    #[must_use]
    pub fn with_config(config: PanoramaConfig) -> Self {
        Self { config }
    }

    /// Compute pairwise homography between two grayscale images.
    ///
    /// # Errors
    ///
    /// Returns an error if feature detection or homography estimation fails.
    pub fn match_pair(
        &self,
        image_a: &[u8],
        width_a: u32,
        height_a: u32,
        image_b: &[u8],
        width_b: u32,
        height_b: u32,
    ) -> CvResult<PairwiseMatch> {
        // Detect features in both images
        let detector = FeatureDetector::new(FeatureDetectorType::Harris)
            .with_max_features(self.config.max_features);

        let (kps_a, descs_a) = detector.detect_and_compute(image_a, width_a, height_a)?;
        let (kps_b, descs_b) = detector.detect_and_compute(image_b, width_b, height_b)?;

        if kps_a.len() < 4 || kps_b.len() < 4 {
            return Err(CvError::matrix_error(
                "insufficient features for panorama matching",
            ));
        }

        // Match features
        let matcher = FeatureMatcher::new();
        let matches = matcher.match_descriptors(&descs_a, &descs_b);

        if matches.len() < 4 {
            return Err(CvError::matrix_error(
                "insufficient matches for homography estimation",
            ));
        }

        // Extract matched point coordinates
        let src_points: Vec<(f64, f64)> = matches
            .iter()
            .map(|m| {
                (
                    f64::from(kps_b[m.train_idx].x),
                    f64::from(kps_b[m.train_idx].y),
                )
            })
            .collect();
        let dst_points: Vec<(f64, f64)> = matches
            .iter()
            .map(|m| {
                (
                    f64::from(kps_a[m.query_idx].x),
                    f64::from(kps_a[m.query_idx].y),
                )
            })
            .collect();

        // RANSAC homography
        let (homography, inliers) = ransac_homography(
            &src_points,
            &dst_points,
            self.config.ransac_threshold,
            self.config.ransac_iterations,
            self.config.ransac_confidence,
        )?;

        let inlier_count = inliers.iter().filter(|&&b| b).count();
        let inlier_ratio = inlier_count as f64 / matches.len() as f64;

        if inlier_ratio < self.config.min_inlier_ratio {
            return Err(CvError::matrix_error(format!(
                "inlier ratio {:.2} below threshold {:.2}",
                inlier_ratio, self.config.min_inlier_ratio
            )));
        }

        // Confidence based on inlier count and ratio
        let confidence = (inlier_ratio * (inlier_count as f64 / 20.0).min(1.0)).min(1.0);

        Ok(PairwiseMatch {
            idx_a: 0,
            idx_b: 1,
            homography,
            inlier_count,
            inlier_ratio,
            confidence,
        })
    }

    /// Stitch two grayscale images into a panorama.
    ///
    /// The result is a grayscale image in a newly allocated buffer.
    /// Returns `(stitched_data, output_width, output_height)`.
    ///
    /// # Errors
    ///
    /// Returns an error if matching or warping fails.
    pub fn stitch_pair(
        &self,
        image_a: &[u8],
        width_a: u32,
        height_a: u32,
        image_b: &[u8],
        width_b: u32,
        height_b: u32,
    ) -> CvResult<(Vec<u8>, u32, u32)> {
        let pair_match = self.match_pair(image_a, width_a, height_a, image_b, width_b, height_b)?;

        // Compute bounding box of warped image B in image A's coordinate frame
        let corners_b = [
            (0.0, 0.0),
            (width_b as f64, 0.0),
            (width_b as f64, height_b as f64),
            (0.0, height_b as f64),
        ];

        let mut min_x = 0.0f64;
        let mut min_y = 0.0f64;
        let mut max_x = width_a as f64;
        let mut max_y = height_a as f64;

        for &(cx, cy) in &corners_b {
            let (tx, ty) = pair_match.homography.transform_point(cx, cy);
            min_x = min_x.min(tx);
            min_y = min_y.min(ty);
            max_x = max_x.max(tx);
            max_y = max_y.max(ty);
        }

        // Clamp to reasonable size (prevent degenerate homographies)
        let max_dim = ((width_a + width_b) * 2) as f64;
        min_x = min_x.max(-max_dim);
        min_y = min_y.max(-max_dim);
        max_x = max_x.min(max_dim);
        max_y = max_y.min(max_dim);

        let out_w = (max_x - min_x).ceil() as u32;
        let out_h = (max_y - min_y).ceil() as u32;

        if out_w == 0 || out_h == 0 || out_w > 16384 || out_h > 16384 {
            return Err(CvError::matrix_error("degenerate panorama dimensions"));
        }

        let offset_x = -min_x;
        let offset_y = -min_y;

        let mut output = vec![0u8; (out_w * out_h) as usize];
        let mut weight_map = vec![0.0f32; (out_w * out_h) as usize];

        // Place image A at offset
        place_image(
            &mut output,
            &mut weight_map,
            out_w,
            out_h,
            image_a,
            width_a,
            height_a,
            offset_x,
            offset_y,
        );

        // Warp and place image B using the homography
        let inv_h = pair_match.homography.inverse()?;

        warp_and_blend(
            &mut output,
            &mut weight_map,
            out_w,
            out_h,
            image_b,
            width_b,
            height_b,
            &inv_h,
            offset_x,
            offset_y,
            self.config.blend_width,
        );

        Ok((output, out_w, out_h))
    }

    /// Estimate cumulative homographies for a sequence of images.
    ///
    /// Given N images, computes homographies that map each image into the
    /// coordinate frame of the reference (middle) image.
    ///
    /// # Errors
    ///
    /// Returns an error if any pairwise matching fails.
    pub fn compute_chain_homographies(
        &self,
        images: &[(&[u8], u32, u32)],
    ) -> CvResult<Vec<TransformMatrix>> {
        let n = images.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n == 1 {
            return Ok(vec![TransformMatrix::identity()]);
        }

        let ref_idx = n / 2;
        let mut homographies = vec![TransformMatrix::identity(); n];

        // Forward chain: ref_idx -> ref_idx+1 -> ...
        for i in ref_idx..n.saturating_sub(1) {
            let (img_a, w_a, h_a) = images[i];
            let (img_b, w_b, h_b) = images[i + 1];

            let pair = self.match_pair(img_a, w_a, h_a, img_b, w_b, h_b)?;
            // pair.homography maps B into A
            // cumulative: image[i+1] into ref = H[i] * H_pair
            homographies[i + 1] = homographies[i].compose(&pair.homography.inverse()?);
        }

        // Backward chain: ref_idx -> ref_idx-1 -> ...
        for i in (1..=ref_idx).rev() {
            let (img_a, w_a, h_a) = images[i];
            let (img_b, w_b, h_b) = images[i - 1];

            let pair = self.match_pair(img_a, w_a, h_a, img_b, w_b, h_b)?;
            homographies[i - 1] = homographies[i].compose(&pair.homography.inverse()?);
        }

        Ok(homographies)
    }
}

impl Default for PanoramaStitcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Place a source image into the output canvas at the given offset.
fn place_image(
    output: &mut [u8],
    weight_map: &mut [f32],
    out_w: u32,
    out_h: u32,
    src: &[u8],
    src_w: u32,
    src_h: u32,
    offset_x: f64,
    offset_y: f64,
) {
    let ox = offset_x.round() as i32;
    let oy = offset_y.round() as i32;

    for sy in 0..src_h as i32 {
        for sx in 0..src_w as i32 {
            let dx = sx + ox;
            let dy = sy + oy;
            if dx >= 0 && dx < out_w as i32 && dy >= 0 && dy < out_h as i32 {
                let si = sy as usize * src_w as usize + sx as usize;
                let di = dy as usize * out_w as usize + dx as usize;
                output[di] = src[si];
                weight_map[di] = 1.0;
            }
        }
    }
}

/// Warp source image using inverse homography and blend into the output canvas.
fn warp_and_blend(
    output: &mut [u8],
    weight_map: &mut [f32],
    out_w: u32,
    out_h: u32,
    src: &[u8],
    src_w: u32,
    src_h: u32,
    inv_homography: &TransformMatrix,
    offset_x: f64,
    offset_y: f64,
    blend_width: usize,
) {
    let sw = src_w as f64;
    let sh = src_h as f64;

    for dy in 0..out_h {
        for dx in 0..out_w {
            // Map output pixel back to source image B coordinates
            let px = dx as f64 - offset_x;
            let py = dy as f64 - offset_y;
            let (sx, sy) = inv_homography.transform_point(px, py);

            if sx >= 0.0 && sx < sw - 1.0 && sy >= 0.0 && sy < sh - 1.0 {
                // Bilinear interpolation in source
                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = (x0 + 1).min(src_w as usize - 1);
                let y1 = (y0 + 1).min(src_h as usize - 1);

                let fx = sx - sx.floor();
                let fy = sy - sy.floor();

                let v00 = src[y0 * src_w as usize + x0] as f64;
                let v10 = src[y0 * src_w as usize + x1] as f64;
                let v01 = src[y1 * src_w as usize + x0] as f64;
                let v11 = src[y1 * src_w as usize + x1] as f64;

                let val = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                let di = dy as usize * out_w as usize + dx as usize;
                let existing_weight = weight_map[di];

                if existing_weight < f32::EPSILON {
                    // No existing pixel - just place
                    output[di] = val.round().clamp(0.0, 255.0) as u8;
                    weight_map[di] = 1.0;
                } else {
                    // Blend in overlap region using distance-based feathering
                    let edge_dist = edge_distance(sx, sy, sw, sh);
                    let blend_w = if blend_width > 0 {
                        (edge_dist / blend_width as f64).clamp(0.0, 1.0) as f32
                    } else {
                        0.5
                    };

                    let total_weight = existing_weight + blend_w;
                    if total_weight > f32::EPSILON {
                        let blended = (output[di] as f32 * existing_weight + val as f32 * blend_w)
                            / total_weight;
                        output[di] = blended.round().clamp(0.0, 255.0) as u8;
                        weight_map[di] = total_weight.min(1.0);
                    }
                }
            }
        }
    }
}

/// Compute distance from a point to the nearest edge of an image.
fn edge_distance(x: f64, y: f64, w: f64, h: f64) -> f64 {
    let dx = x.min(w - 1.0 - x);
    let dy = y.min(h - 1.0 - y);
    dx.min(dy).max(0.0)
}

/// Compute cylindrical projection coordinates.
///
/// Maps a pixel (x, y) from an image of size (w, h) with focal length f
/// to cylindrical coordinates (theta, h_cyl).
#[must_use]
pub fn cylindrical_project(x: f64, y: f64, w: f64, h: f64, focal: f64) -> (f64, f64) {
    let cx = w / 2.0;
    let cy = h / 2.0;
    let theta = ((x - cx) / focal).atan();
    let h_cyl = (y - cy) / ((x - cx).powi(2) + focal * focal).sqrt();
    (theta, h_cyl)
}

/// Inverse cylindrical projection: from cylindrical coords back to image coords.
#[must_use]
pub fn cylindrical_unproject(theta: f64, h_cyl: f64, w: f64, h: f64, focal: f64) -> (f64, f64) {
    let cx = w / 2.0;
    let cy = h / 2.0;
    let x = focal * theta.tan() + cx;
    let y = h_cyl * (focal / theta.cos()) + cy;
    (x, y)
}

/// Warp a grayscale image to cylindrical projection.
///
/// Returns `(warped_data, out_width, out_height)`.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn warp_cylindrical(
    image: &[u8],
    width: u32,
    height: u32,
    focal_length: f64,
) -> CvResult<(Vec<u8>, u32, u32)> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }
    let size = (width * height) as usize;
    if image.len() < size {
        return Err(CvError::insufficient_data(size, image.len()));
    }

    let w = width as f64;
    let h = height as f64;
    let mut output = vec![0u8; size];

    for y in 0..height {
        for x in 0..width {
            let (theta, h_cyl) = cylindrical_project(x as f64, y as f64, w, h, focal_length);
            let (src_x, src_y) = cylindrical_unproject(theta, h_cyl, w, h, focal_length);

            let sx = src_x.round() as i32;
            let sy = src_y.round() as i32;

            if sx >= 0 && sx < width as i32 && sy >= 0 && sy < height as i32 {
                output[y as usize * width as usize + x as usize] =
                    image[sy as usize * width as usize + sx as usize];
            }
        }
    }

    Ok((output, width, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_image(width: u32, height: u32, offset_x: i32) -> Vec<u8> {
        let mut img = vec![0u8; (width * height) as usize];
        // Create a distinctive pattern
        for y in 10..height.saturating_sub(10) {
            for x in 10..width.saturating_sub(10) {
                let px = x as i32 + offset_x;
                let val = ((px.abs() * 37 + y as i32 * 59) % 256) as u8;
                img[y as usize * width as usize + x as usize] = val;
            }
        }
        img
    }

    #[test]
    fn test_panorama_config_default() {
        let config = PanoramaConfig::default();
        assert_eq!(config.max_features, 500);
        assert!((config.ransac_threshold - 3.0).abs() < f64::EPSILON);
        assert_eq!(config.blend_width, 32);
    }

    #[test]
    fn test_panorama_stitcher_creation() {
        let stitcher = PanoramaStitcher::new();
        assert_eq!(stitcher.config.max_features, 500);
    }

    #[test]
    fn test_panorama_stitcher_with_config() {
        let config = PanoramaConfig {
            max_features: 200,
            ..PanoramaConfig::default()
        };
        let stitcher = PanoramaStitcher::with_config(config);
        assert_eq!(stitcher.config.max_features, 200);
    }

    #[test]
    fn test_edge_distance() {
        assert!((edge_distance(5.0, 5.0, 100.0, 100.0) - 5.0).abs() < f64::EPSILON);
        assert!((edge_distance(50.0, 50.0, 100.0, 100.0) - 49.0).abs() < f64::EPSILON);
        assert!((edge_distance(0.0, 0.0, 100.0, 100.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cylindrical_project_center() {
        let (theta, h) = cylindrical_project(50.0, 50.0, 100.0, 100.0, 100.0);
        assert!(theta.abs() < 1e-10, "center pixel should have theta~0");
        assert!(h.abs() < 1e-10, "center pixel should have h~0");
    }

    #[test]
    fn test_cylindrical_roundtrip() {
        let w = 100.0;
        let h = 100.0;
        let f = 200.0;
        let (theta, hc) = cylindrical_project(30.0, 40.0, w, h, f);
        let (rx, ry) = cylindrical_unproject(theta, hc, w, h, f);
        assert!((rx - 30.0).abs() < 1e-6, "x roundtrip failed: {rx}");
        assert!((ry - 40.0).abs() < 1e-6, "y roundtrip failed: {ry}");
    }

    #[test]
    fn test_warp_cylindrical() {
        let img = vec![128u8; 100 * 100];
        let (result, w, h) = warp_cylindrical(&img, 100, 100, 200.0).expect("warp should succeed");
        assert_eq!(w, 100);
        assert_eq!(h, 100);
        assert_eq!(result.len(), 10000);
        // Center pixels should be preserved for a uniform image
        assert_eq!(result[50 * 100 + 50], 128);
    }

    #[test]
    fn test_warp_cylindrical_invalid() {
        assert!(warp_cylindrical(&[], 0, 0, 100.0).is_err());
    }

    #[test]
    fn test_place_image() {
        let mut output = vec![0u8; 200 * 100];
        let mut weight_map = vec![0.0f32; 200 * 100];
        let src = vec![128u8; 100 * 100];

        place_image(
            &mut output,
            &mut weight_map,
            200,
            100,
            &src,
            100,
            100,
            0.0,
            0.0,
        );

        // First 100 columns should be 128
        assert_eq!(output[0], 128);
        assert_eq!(output[50 * 200 + 50], 128);
        // After column 100 should still be 0
        assert_eq!(output[50 * 200 + 150], 0);
    }

    #[test]
    fn test_pairwise_match_insufficient_features() {
        let stitcher = PanoramaStitcher::new();
        // Blank images have no features
        let img_a = vec![0u8; 100 * 100];
        let img_b = vec![0u8; 100 * 100];
        let result = stitcher.match_pair(&img_a, 100, 100, &img_b, 100, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_chain_homographies_single() {
        let stitcher = PanoramaStitcher::new();
        let img = vec![128u8; 100 * 100];
        let images: Vec<(&[u8], u32, u32)> = vec![(&img, 100, 100)];
        let result = stitcher
            .compute_chain_homographies(&images)
            .expect("single image should succeed");
        assert_eq!(result.len(), 1);
        // Should be identity
        let (x, y) = result[0].transform_point(10.0, 20.0);
        assert!((x - 10.0).abs() < 1e-6);
        assert!((y - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_chain_homographies_empty() {
        let stitcher = PanoramaStitcher::new();
        let result = stitcher
            .compute_chain_homographies(&[])
            .expect("empty should succeed");
        assert!(result.is_empty());
    }
}
