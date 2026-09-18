//! Depth map utilities: storage, normalization, inversion, point-cloud
//! back-projection, statistics, and multi-map fusion.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

/// A 2-D depth map stored as a flat row-major `f32` buffer.
#[derive(Debug, Clone)]
pub struct DepthMap {
    /// Raw depth values in row-major order.
    pub data: Vec<f32>,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Minimum expected depth (used as hint; may differ from actual data min).
    pub min_depth: f32,
    /// Maximum expected depth (used as hint; may differ from actual data max).
    pub max_depth: f32,
}

impl DepthMap {
    /// Create a new `DepthMap` filled with `fill_value`.
    #[must_use]
    pub fn new(
        width: usize,
        height: usize,
        fill_value: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> Self {
        Self {
            data: vec![fill_value; width * height],
            width,
            height,
            min_depth,
            max_depth,
        }
    }

    /// Return the depth value at column `x`, row `y`.
    ///
    /// # Panics
    ///
    /// Panics when `x >= self.width` or `y >= self.height`.
    #[must_use]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        assert!(
            x < self.width && y < self.height,
            "depth map index ({x},{y}) out of bounds"
        );
        self.data[y * self.width + x]
    }

    /// Return a new `DepthMap` with all values linearly remapped to `[0, 1]`.
    ///
    /// If all values are identical the result is all-zero.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let (min, max) = data_min_max(&self.data);
        let range = max - min;
        let normalized: Vec<f32> = if range < 1e-9 {
            vec![0.0; self.data.len()]
        } else {
            self.data.iter().map(|&v| (v - min) / range).collect()
        };
        Self {
            data: normalized,
            width: self.width,
            height: self.height,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    /// Return a new `DepthMap` where every value `v` becomes `max - v + min`.
    #[must_use]
    pub fn invert(&self) -> Self {
        let (min, max) = data_min_max(&self.data);
        let inverted: Vec<f32> = self.data.iter().map(|&v| max + min - v).collect();
        Self {
            data: inverted,
            width: self.width,
            height: self.height,
            min_depth: self.min_depth,
            max_depth: self.max_depth,
        }
    }
}

fn data_min_max(data: &[f32]) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in data {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    if min == f32::INFINITY {
        (0.0, 0.0)
    } else {
        (min, max)
    }
}

/// Back-projects a depth map to a 3-D point cloud using pinhole camera intrinsics.
pub struct DepthToPointCloud;

impl DepthToPointCloud {
    /// Convert each pixel of `depth` into a 3-D point `(X, Y, Z)` using:
    ///
    /// ```text
    /// X = (u - cx) * depth / fx
    /// Y = (v - cy) * depth / fy
    /// Z = depth
    /// ```
    ///
    /// Pixels with a depth of exactly `0.0` are skipped and produce no point.
    #[must_use]
    pub fn convert(depth: &DepthMap, fx: f32, fy: f32, cx: f32, cy: f32) -> Vec<(f32, f32, f32)> {
        let mut points = Vec::with_capacity(depth.width * depth.height);
        for y in 0..depth.height {
            for x in 0..depth.width {
                let z = depth.data[y * depth.width + x];
                if z == 0.0 {
                    continue;
                }
                let px = (x as f32 - cx) * z / fx;
                let py = (y as f32 - cy) * z / fy;
                points.push((px, py, z));
            }
        }
        points
    }
}

/// Aggregate statistics computed from a `DepthMap`.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthStats {
    /// Mean depth of all (finite) pixels.
    pub mean: f32,
    /// Standard deviation of all (finite) pixels.
    pub std_dev: f32,
    /// Minimum depth value.
    pub min: f32,
    /// Maximum depth value.
    pub max: f32,
    /// Number of pixels (all pixels count; zero-depth pixels are included).
    pub valid_pixels: usize,
}

impl DepthStats {
    /// Compute statistics for `depth`.
    ///
    /// `valid_pixels` counts all pixels in the map. Returns `None` if the map
    /// is empty.
    #[must_use]
    pub fn compute(depth: &DepthMap) -> Option<Self> {
        let n = depth.data.len();
        if n == 0 {
            return None;
        }
        let (min, max) = data_min_max(&depth.data);
        let mean = depth.data.iter().sum::<f32>() / n as f32;
        let variance = depth
            .data
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f32>()
            / n as f32;
        let std_dev = variance.sqrt();
        Some(Self {
            mean,
            std_dev,
            min,
            max,
            valid_pixels: n,
        })
    }
}

/// Fusion utilities for combining multiple depth maps.
pub struct DepthFusion;

impl DepthFusion {
    /// Element-wise average of all maps in `maps`.
    ///
    /// All maps must have the same dimensions. Returns `None` if `maps` is
    /// empty or the dimensions are inconsistent.
    #[must_use]
    pub fn average(maps: &[DepthMap]) -> Option<DepthMap> {
        if maps.is_empty() {
            return None;
        }
        let w = maps[0].width;
        let h = maps[0].height;
        if maps.iter().any(|m| m.width != w || m.height != h) {
            return None;
        }
        let len = w * h;
        let mut result = vec![0.0f32; len];
        for m in maps {
            for (r, &v) in result.iter_mut().zip(m.data.iter()) {
                *r += v;
            }
        }
        let count = maps.len() as f32;
        for v in &mut result {
            *v /= count;
        }
        let (min_d, max_d) = data_min_max(&result);
        Some(DepthMap {
            data: result,
            width: w,
            height: h,
            min_depth: min_d,
            max_depth: max_d,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_map(w: usize, h: usize, fill: f32) -> DepthMap {
        DepthMap::new(w, h, fill, 0.0, 10.0)
    }

    // ---- DepthMap ----

    #[test]
    fn test_depth_map_at() {
        let mut dm = make_map(3, 2, 1.0);
        dm.data[1 * 3 + 2] = 7.5;
        assert!((dm.at(2, 1) - 7.5).abs() < 1e-6);
    }

    #[test]
    fn test_depth_map_at_zero_zero() {
        let dm = make_map(4, 4, 3.0);
        assert!((dm.at(0, 0) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_range_is_zero_to_one() {
        let mut dm = make_map(2, 1, 0.0);
        dm.data[0] = 0.0;
        dm.data[1] = 10.0;
        let n = dm.normalize();
        assert!((n.data[0] - 0.0).abs() < 1e-6);
        assert!((n.data[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_uniform_all_zero() {
        let dm = make_map(4, 4, 5.0);
        let n = dm.normalize();
        assert!(n.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_normalize_midpoint() {
        let mut dm = make_map(3, 1, 0.0);
        dm.data = vec![0.0, 5.0, 10.0];
        let n = dm.normalize();
        assert!((n.data[1] - 0.5).abs() < 1e-6, "mid = {}", n.data[1]);
    }

    #[test]
    fn test_invert_swaps_min_max() {
        let mut dm = make_map(2, 1, 0.0);
        dm.data = vec![0.0, 10.0];
        let inv = dm.invert();
        // 0 becomes 10, 10 becomes 0
        assert!((inv.data[0] - 10.0).abs() < 1e-6);
        assert!((inv.data[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_invert_uniform() {
        let dm = make_map(3, 3, 4.0);
        let inv = dm.invert();
        // All same value: min == max == 4.0, so every inverted pixel == 4+4-4 == 4.
        assert!(inv.data.iter().all(|&v| (v - 4.0).abs() < 1e-6));
    }

    // ---- DepthToPointCloud ----

    #[test]
    fn test_point_cloud_length() {
        let dm = make_map(4, 4, 1.0);
        let pts = DepthToPointCloud::convert(&dm, 500.0, 500.0, 2.0, 2.0);
        assert_eq!(pts.len(), 16);
    }

    #[test]
    fn test_point_cloud_zero_depth_skipped() {
        let mut dm = make_map(2, 1, 1.0);
        dm.data[0] = 0.0; // skip
        let pts = DepthToPointCloud::convert(&dm, 500.0, 500.0, 0.0, 0.0);
        assert_eq!(pts.len(), 1);
    }

    #[test]
    fn test_point_cloud_z_equals_depth() {
        let mut dm = make_map(1, 1, 0.0);
        dm.data[0] = 3.5;
        let pts = DepthToPointCloud::convert(&dm, 100.0, 100.0, 0.0, 0.0);
        assert_eq!(pts.len(), 1);
        assert!((pts[0].2 - 3.5).abs() < 1e-6);
    }

    // ---- DepthStats ----

    #[test]
    fn test_depth_stats_compute_basic() {
        let mut dm = make_map(2, 1, 0.0);
        dm.data = vec![2.0, 4.0];
        let s = DepthStats::compute(&dm).expect("should succeed in test");
        assert!((s.mean - 3.0).abs() < 1e-6);
        assert!((s.min - 2.0).abs() < 1e-6);
        assert!((s.max - 4.0).abs() < 1e-6);
        assert_eq!(s.valid_pixels, 2);
    }

    #[test]
    fn test_depth_stats_empty_returns_none() {
        let dm = DepthMap {
            data: vec![],
            width: 0,
            height: 0,
            min_depth: 0.0,
            max_depth: 0.0,
        };
        assert!(DepthStats::compute(&dm).is_none());
    }

    #[test]
    fn test_depth_stats_std_dev_uniform() {
        let dm = make_map(4, 4, 5.0);
        let s = DepthStats::compute(&dm).expect("should succeed in test");
        assert!(s.std_dev.abs() < 1e-6);
    }

    // ---- DepthFusion ----

    #[test]
    fn test_fusion_average_basic() {
        let mut a = make_map(2, 1, 0.0);
        a.data = vec![0.0, 4.0];
        let mut b = make_map(2, 1, 0.0);
        b.data = vec![2.0, 8.0];
        let avg = DepthFusion::average(&[a, b]).expect("should succeed in test");
        assert!((avg.data[0] - 1.0).abs() < 1e-6);
        assert!((avg.data[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_fusion_average_empty_returns_none() {
        assert!(DepthFusion::average(&[]).is_none());
    }

    #[test]
    fn test_fusion_average_mismatched_dims_returns_none() {
        let a = make_map(2, 2, 1.0);
        let b = make_map(3, 2, 1.0);
        assert!(DepthFusion::average(&[a, b]).is_none());
    }

    #[test]
    fn test_fusion_average_single_map() {
        let mut dm = make_map(2, 1, 0.0);
        dm.data = vec![3.0, 7.0];
        let avg = DepthFusion::average(&[dm]).expect("should succeed in test");
        assert!((avg.data[0] - 3.0).abs() < 1e-6);
        assert!((avg.data[1] - 7.0).abs() < 1e-6);
    }
}
