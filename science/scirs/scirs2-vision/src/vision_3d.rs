//! 3D Vision module - Stereo vision, depth estimation, and structure from motion
//!
//! This module provides capabilities for 3D reconstruction, stereo matching,
//! depth estimation, and structure from motion.

use crate::error::{Result, VisionError};
use image::{DynamicImage, GrayImage, Rgb, RgbImage};
use scirs2_core::ndarray::{Array2, Array3};
use std::collections::HashMap;

/// Stereo matching algorithm
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StereoAlgorithm {
    /// Semi-Global Block Matching
    SGBM,
    /// Block Matching
    BM,
    /// Graph Cut stereo
    GraphCut,
}

/// Stereo matching parameters for SGBM
#[derive(Debug, Clone)]
pub struct SGBMParams {
    /// Minimum disparity
    pub min_disparity: i32,
    /// Number of disparities (must be divisible by 16)
    pub num_disparities: i32,
    /// Block size (odd number >= 1)
    pub block_size: usize,
    /// P1 parameter for smoothness
    pub p1: i32,
    /// P2 parameter for smoothness
    pub p2: i32,
    /// Maximum allowed difference
    pub disp_12_max_diff: i32,
    /// Pre-filter cap
    pub pre_filter_cap: i32,
    /// Uniqueness ratio (0-100)
    pub uniqueness_ratio: i32,
    /// Speckle window size
    pub speckle_window_size: usize,
    /// Speckle range
    pub speckle_range: i32,
}

impl Default for SGBMParams {
    fn default() -> Self {
        Self {
            min_disparity: 0,
            num_disparities: 64,
            block_size: 5,
            p1: 8 * 3 * 5 * 5,  // 8 * number of channels * block_size^2
            p2: 32 * 3 * 5 * 5, // 32 * number of channels * block_size^2
            disp_12_max_diff: 1,
            pre_filter_cap: 63,
            uniqueness_ratio: 10,
            speckle_window_size: 100,
            speckle_range: 32,
        }
    }
}

/// Block Matching parameters
#[derive(Debug, Clone)]
pub struct BMParams {
    /// Minimum disparity
    pub min_disparity: i32,
    /// Number of disparities
    pub num_disparities: i32,
    /// Block size
    pub block_size: usize,
}

impl Default for BMParams {
    fn default() -> Self {
        Self {
            min_disparity: 0,
            num_disparities: 64,
            block_size: 15,
        }
    }
}

/// Disparity map result
#[derive(Debug, Clone)]
pub struct DisparityMap {
    /// Disparity values (height × width)
    pub disparity: Array2<f32>,
    /// Confidence map
    pub confidence: Array2<f32>,
}

impl DisparityMap {
    /// Convert disparity to depth map
    ///
    /// # Arguments
    ///
    /// * `focal_length` - Camera focal length in pixels
    /// * `baseline` - Stereo baseline in meters
    ///
    /// # Returns
    ///
    /// * Depth map in meters
    pub fn to_depth_map(&self, focal_length: f32, baseline: f32) -> Array2<f32> {
        let (height, width) = self.disparity.dim();
        let mut depth = Array2::zeros((height, width));

        for y in 0..height {
            for x in 0..width {
                let disp = self.disparity[[y, x]];
                if disp > 0.0 {
                    depth[[y, x]] = (focal_length * baseline) / disp;
                }
            }
        }

        depth
    }

    /// Visualize disparity map as color image
    pub fn visualize(&self) -> Result<RgbImage> {
        let (height, width) = self.disparity.dim();
        let mut img = RgbImage::new(width as u32, height as u32);

        // Find min and max disparity for normalization
        let mut min_disp = f32::INFINITY;
        let mut max_disp = f32::NEG_INFINITY;

        for y in 0..height {
            for x in 0..width {
                let disp = self.disparity[[y, x]];
                if disp > 0.0 {
                    min_disp = min_disp.min(disp);
                    max_disp = max_disp.max(disp);
                }
            }
        }

        let range = max_disp - min_disp;
        if range == 0.0 {
            return Ok(img);
        }

        for y in 0..height {
            for x in 0..width {
                let disp = self.disparity[[y, x]];
                if disp > 0.0 {
                    let normalized = ((disp - min_disp) / range * 255.0) as u8;
                    img.put_pixel(
                        x as u32,
                        y as u32,
                        Rgb([normalized, normalized, normalized]),
                    );
                } else {
                    img.put_pixel(x as u32, y as u32, Rgb([0, 0, 0]));
                }
            }
        }

        Ok(img)
    }
}

/// Stereo matching using Semi-Global Block Matching (SGBM)
///
/// # Arguments
///
/// * `left` - Left stereo image
/// * `right` - Right stereo image
/// * `params` - SGBM parameters
///
/// # Returns
///
/// * Disparity map
pub fn stereo_sgbm(
    left: &DynamicImage,
    right: &DynamicImage,
    params: &SGBMParams,
) -> Result<DisparityMap> {
    let left_gray = left.to_luma8();
    let right_gray = right.to_luma8();

    if left_gray.dimensions() != right_gray.dimensions() {
        return Err(VisionError::InvalidParameter(
            "Left and right images must have same dimensions".to_string(),
        ));
    }

    let (width, height) = left_gray.dimensions();
    let mut disparity = Array2::zeros((height as usize, width as usize));
    let mut confidence = Array2::zeros((height as usize, width as usize));

    // Semi-global matching algorithm
    for y in (params.block_size / 2)..(height as usize - params.block_size / 2) {
        for x in (params.block_size / 2)..(width as usize - params.block_size / 2) {
            let mut best_disp = 0;
            let mut best_cost = f32::INFINITY;

            // Search for best disparity
            for d in params.min_disparity..(params.min_disparity + params.num_disparities) {
                let match_x = x as i32 - d;
                if match_x < (params.block_size / 2) as i32 {
                    continue;
                }
                if match_x >= (width as i32 - params.block_size as i32 / 2) {
                    continue;
                }

                // Compute matching cost using SAD (Sum of Absolute Differences)
                let mut sad = 0.0f32;
                let radius = params.block_size / 2;

                for dy in 0..params.block_size {
                    for dx in 0..params.block_size {
                        let ly = y + dy - radius;
                        let lx = x + dx - radius;
                        let rx = match_x as usize + dx - radius;

                        let left_val = left_gray.get_pixel(lx as u32, ly as u32)[0] as f32;
                        let right_val = right_gray.get_pixel(rx as u32, ly as u32)[0] as f32;

                        sad += (left_val - right_val).abs();
                    }
                }

                // Apply smoothness penalty
                let smoothness_cost = if d > 0 {
                    let prev_cost = if x > 0 { disparity[[y, x - 1]] } else { 0.0 };
                    let diff = (d as f32 - prev_cost).abs();
                    if diff == 1.0 {
                        params.p1 as f32
                    } else if diff > 1.0 {
                        params.p2 as f32
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let total_cost = sad + smoothness_cost;

                if total_cost < best_cost {
                    best_cost = total_cost;
                    best_disp = d;
                }
            }

            disparity[[y, x]] = best_disp as f32;
            confidence[[y, x]] = 1.0 / (1.0 + best_cost);
        }
    }

    // Post-processing: remove speckles
    if params.speckle_window_size > 0 {
        remove_speckles(
            &mut disparity,
            params.speckle_window_size,
            params.speckle_range,
        )?;
    }

    Ok(DisparityMap {
        disparity,
        confidence,
    })
}

/// Stereo matching using Block Matching (BM)
///
/// # Arguments
///
/// * `left` - Left stereo image
/// * `right` - Right stereo image
/// * `params` - BM parameters
///
/// # Returns
///
/// * Disparity map
pub fn stereo_bm(
    left: &DynamicImage,
    right: &DynamicImage,
    params: &BMParams,
) -> Result<DisparityMap> {
    let left_gray = left.to_luma8();
    let right_gray = right.to_luma8();

    if left_gray.dimensions() != right_gray.dimensions() {
        return Err(VisionError::InvalidParameter(
            "Left and right images must have same dimensions".to_string(),
        ));
    }

    let (width, height) = left_gray.dimensions();
    let mut disparity = Array2::zeros((height as usize, width as usize));
    let mut confidence = Array2::zeros((height as usize, width as usize));

    let radius = params.block_size / 2;

    for y in radius..(height as usize - radius) {
        for x in radius..(width as usize - radius) {
            let mut best_disp = 0;
            let mut best_cost = f32::INFINITY;

            // Search for best disparity
            for d in params.min_disparity..(params.min_disparity + params.num_disparities) {
                let match_x = x as i32 - d;
                if match_x < radius as i32 || match_x >= (width as i32 - radius as i32) {
                    continue;
                }

                // Compute SAD
                let mut sad = 0.0f32;

                for dy in 0..params.block_size {
                    for dx in 0..params.block_size {
                        let ly = y + dy - radius;
                        let lx = x + dx - radius;
                        let rx = match_x as usize + dx - radius;

                        let left_val = left_gray.get_pixel(lx as u32, ly as u32)[0] as f32;
                        let right_val = right_gray.get_pixel(rx as u32, ly as u32)[0] as f32;

                        sad += (left_val - right_val).abs();
                    }
                }

                if sad < best_cost {
                    best_cost = sad;
                    best_disp = d;
                }
            }

            disparity[[y, x]] = best_disp as f32;
            confidence[[y, x]] =
                1.0 / (1.0 + best_cost / (params.block_size * params.block_size) as f32);
        }
    }

    Ok(DisparityMap {
        disparity,
        confidence,
    })
}

/// Remove speckles from disparity map
fn remove_speckles(disparity: &mut Array2<f32>, window_size: usize, max_diff: i32) -> Result<()> {
    let (height, width) = disparity.dim();
    let mut labels = Array2::zeros((height, width));
    let mut label_id = 1;

    // Connected component labeling
    for y in 0..height {
        for x in 0..width {
            if disparity[[y, x]] > 0.0 && labels[[y, x]] == 0 {
                // Flood fill
                let mut stack = vec![(y, x)];
                let mut region_size = 0;
                let base_disp = disparity[[y, x]];

                while let Some((cy, cx)) = stack.pop() {
                    if labels[[cy, cx]] > 0 {
                        continue;
                    }

                    let disp = disparity[[cy, cx]];
                    if (disp - base_disp).abs() > max_diff as f32 {
                        continue;
                    }

                    labels[[cy, cx]] = label_id;
                    region_size += 1;

                    // Add neighbors
                    for (dy, dx) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
                        let ny = cy as i32 + dy;
                        let nx = cx as i32 + dx;

                        if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                            let ny = ny as usize;
                            let nx = nx as usize;
                            if disparity[[ny, nx]] > 0.0 && labels[[ny, nx]] == 0 {
                                stack.push((ny, nx));
                            }
                        }
                    }
                }

                // Remove small regions
                if region_size < window_size {
                    for y in 0..height {
                        for x in 0..width {
                            if labels[[y, x]] == label_id {
                                disparity[[y, x]] = 0.0;
                            }
                        }
                    }
                }

                label_id += 1;
            }
        }
    }

    Ok(())
}

/// 3D point cloud
#[derive(Debug, Clone)]
pub struct PointCloud {
    /// 3D points (N × 3: x, y, z)
    pub points: Array2<f32>,
    /// RGB colors (N × 3: r, g, b)
    pub colors: Option<Array2<u8>>,
}

/// Triangulate 3D points from stereo disparity
///
/// # Arguments
///
/// * `disparity` - Disparity map
/// * `focal_length` - Camera focal length in pixels
/// * `baseline` - Stereo baseline in meters
/// * `cx` - Principal point x-coordinate
/// * `cy` - Principal point y-coordinate
///
/// # Returns
///
/// * 3D point cloud
pub fn triangulate_points(
    disparity: &DisparityMap,
    focal_length: f32,
    baseline: f32,
    cx: f32,
    cy: f32,
) -> Result<PointCloud> {
    let (height, width) = disparity.disparity.dim();
    let mut points_vec = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let d = disparity.disparity[[y, x]];
            if d > 0.0 {
                // Compute 3D coordinates
                let z = (focal_length * baseline) / d;
                let x3d = ((x as f32 - cx) * z) / focal_length;
                let y3d = ((y as f32 - cy) * z) / focal_length;

                points_vec.push([x3d, y3d, z]);
            }
        }
    }

    let num_points = points_vec.len();
    let mut points = Array2::zeros((num_points, 3));

    for (i, p) in points_vec.iter().enumerate() {
        points[[i, 0]] = p[0];
        points[[i, 1]] = p[1];
        points[[i, 2]] = p[2];
    }

    Ok(PointCloud {
        points,
        colors: None,
    })
}

/// Structure from Motion (SfM) result
#[derive(Debug, Clone)]
pub struct SfMResult {
    /// 3D point cloud
    pub point_cloud: PointCloud,
    /// Camera poses (N × 12: 3×4 projection matrices)
    pub camera_poses: Vec<Array2<f32>>,
    /// Feature correspondences
    pub correspondences: HashMap<usize, Vec<(usize, f32, f32)>>,
}

/// Perform structure from motion reconstruction
///
/// # Arguments
///
/// * `images` - Sequence of images
/// * `focal_length` - Camera focal length
///
/// # Returns
///
/// * SfM reconstruction result
pub fn structure_from_motion(images: &[DynamicImage], focal_length: f32) -> Result<SfMResult> {
    if images.len() < 2 {
        return Err(VisionError::InvalidParameter(
            "Need at least 2 images for SfM".to_string(),
        ));
    }

    // Placeholder implementation
    // In a real implementation, this would:
    // 1. Extract features from all images
    // 2. Match features across images
    // 3. Estimate camera poses
    // 4. Triangulate 3D points
    // 5. Bundle adjustment

    let point_cloud = PointCloud {
        points: Array2::zeros((0, 3)),
        colors: None,
    };

    let camera_poses = Vec::new();
    let correspondences = HashMap::new();

    Ok(SfMResult {
        point_cloud,
        camera_poses,
        correspondences,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_sgbm_params_default() {
        let params = SGBMParams::default();
        assert_eq!(params.block_size, 5);
        assert_eq!(params.num_disparities, 64);
    }

    #[test]
    fn test_bm_params_default() {
        let params = BMParams::default();
        assert_eq!(params.block_size, 15);
    }

    #[test]
    fn test_stereo_bm() {
        let left = DynamicImage::ImageRgb8(RgbImage::new(64, 64));
        let right = DynamicImage::ImageRgb8(RgbImage::new(64, 64));

        let result = stereo_bm(&left, &right, &BMParams::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_stereo_sgbm() {
        let left = DynamicImage::ImageRgb8(RgbImage::new(64, 64));
        let right = DynamicImage::ImageRgb8(RgbImage::new(64, 64));

        let result = stereo_sgbm(&left, &right, &SGBMParams::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_disparity_to_depth() {
        let disparity = Array2::from_elem((10, 10), 1.0);
        let confidence = Array2::from_elem((10, 10), 1.0);
        let disp_map = DisparityMap {
            disparity,
            confidence,
        };

        let depth = disp_map.to_depth_map(500.0, 0.1);
        assert_eq!(depth.dim(), (10, 10));
    }

    #[test]
    fn test_disparity_visualize() {
        let disparity = Array2::from_elem((10, 10), 1.0);
        let confidence = Array2::from_elem((10, 10), 1.0);
        let disp_map = DisparityMap {
            disparity,
            confidence,
        };

        let vis = disp_map.visualize();
        assert!(vis.is_ok());
    }

    #[test]
    fn test_triangulate_points() {
        let disparity = Array2::from_elem((10, 10), 1.0);
        let confidence = Array2::from_elem((10, 10), 1.0);
        let disp_map = DisparityMap {
            disparity,
            confidence,
        };

        let cloud = triangulate_points(&disp_map, 500.0, 0.1, 320.0, 240.0);
        assert!(cloud.is_ok());
    }

    #[test]
    fn test_sfm_insufficient_images() {
        let images = vec![DynamicImage::ImageRgb8(RgbImage::new(64, 64))];
        let result = structure_from_motion(&images, 500.0);
        assert!(result.is_err());
    }
}
