//! Crowd density and flow analysis.
//!
//! This module estimates crowd density and flow from video data:
//! - **Density Estimation** - Motion-vector based density mapping
//! - **Flow Analysis** - Directional flow vectors
//! - **Region Classification** - Empty / Sparse / Moderate / Dense / `VeryDense`

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────
// CrowdDensity
// ─────────────────────────────────────────────────────────────

/// Qualitative crowd density level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrowdDensity {
    /// No people detected
    Empty,
    /// 0–0.5 persons per m²
    Sparse,
    /// 0.5–2 persons per m²
    Moderate,
    /// 2–4 persons per m²
    Dense,
    /// > 4 persons per m²
    VeryDense,
}

impl CrowdDensity {
    /// Returns the representative persons-per-m² value for this density level.
    #[must_use]
    pub fn people_per_m2(&self) -> f32 {
        match self {
            Self::Empty => 0.0,
            Self::Sparse => 0.25,
            Self::Moderate => 1.0,
            Self::Dense => 3.0,
            Self::VeryDense => 6.0,
        }
    }

    /// Convert a raw density value (0–1) to a `CrowdDensity` level.
    #[must_use]
    pub fn from_raw(value: f32) -> Self {
        if value < 0.05 {
            Self::Empty
        } else if value < 0.25 {
            Self::Sparse
        } else if value < 0.55 {
            Self::Moderate
        } else if value < 0.80 {
            Self::Dense
        } else {
            Self::VeryDense
        }
    }

    /// Returns the level name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Empty => "Empty",
            Self::Sparse => "Sparse",
            Self::Moderate => "Moderate",
            Self::Dense => "Dense",
            Self::VeryDense => "VeryDense",
        }
    }
}

// ─────────────────────────────────────────────────────────────
// CrowdRegion
// ─────────────────────────────────────────────────────────────

/// A spatial region with an associated crowd density estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrowdRegion {
    /// Bounding box `(x, y, width, height)` in pixels
    pub bbox: (u32, u32, u32, u32),
    /// Qualitative density level
    pub density: CrowdDensity,
    /// Estimated person count in this region
    pub estimated_count: u32,
}

impl CrowdRegion {
    /// Returns the area of this region in pixels.
    #[must_use]
    pub fn area(&self) -> u32 {
        self.bbox.2 * self.bbox.3
    }
}

// ─────────────────────────────────────────────────────────────
// DensityMap
// ─────────────────────────────────────────────────────────────

/// A spatial density map.
///
/// Values are in \[0, 1\] where 1 represents the maximum possible density.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityMap {
    /// Map width (number of blocks)
    pub width: u32,
    /// Map height (number of blocks)
    pub height: u32,
    /// Density values (row-major, one value per block)
    pub values: Vec<f32>,
}

impl DensityMap {
    /// Create a new density map filled with zeros.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            values: vec![0.0; (width * height) as usize],
        }
    }

    /// Returns the sum of all density values.
    #[must_use]
    pub fn total_density(&self) -> f32 {
        self.values.iter().sum()
    }

    /// Returns the block coordinates and value of the highest-density block.
    ///
    /// `block_size` is used only for context; this method works on the block
    /// grid directly.
    #[must_use]
    pub fn peak_region(&self, _block_size: u32) -> (u32, u32, f32) {
        if self.values.is_empty() {
            return (0, 0, 0.0);
        }

        let (idx, &val) = self
            .values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        let w = self.width as usize;
        let bx = (idx % w) as u32;
        let by = (idx / w) as u32;
        (bx, by, val)
    }

    /// Returns the density value at block coordinates `(bx, by)`.
    #[must_use]
    pub fn get(&self, bx: u32, by: u32) -> f32 {
        let idx = by * self.width + bx;
        self.values.get(idx as usize).copied().unwrap_or(0.0)
    }

    /// Returns the average density value.
    #[must_use]
    pub fn average_density(&self) -> f32 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.total_density() / self.values.len() as f32
    }
}

// ─────────────────────────────────────────────────────────────
// CrowdAnalyzer
// ─────────────────────────────────────────────────────────────

/// Crowd density estimator.
pub struct CrowdAnalyzer;

impl CrowdAnalyzer {
    /// Estimate crowd density from a set of motion vectors.
    ///
    /// Motion vectors cover the full image `width × height` pixels.
    /// We partition the image into blocks and map motion magnitude to density.
    ///
    /// * `motion_vectors` – `(vx, vy)` per-pixel or sampled motion vectors
    /// * `width` – image width in pixels
    /// * `height` – image height in pixels
    #[must_use]
    pub fn estimate_from_motion(
        motion_vectors: &[(f32, f32)],
        width: u32,
        height: u32,
    ) -> DensityMap {
        const BLOCK_SIZE: u32 = 16;
        let bw = width.div_ceil(BLOCK_SIZE);
        let bh = height.div_ceil(BLOCK_SIZE);

        let mut map = DensityMap::new(bw, bh);

        if motion_vectors.is_empty() || width == 0 || height == 0 {
            return map;
        }

        // Estimate mapping from vector count to grid
        let vecs_per_row = motion_vectors.len() / height.max(1) as usize;
        let vecs_per_row = vecs_per_row.max(1);

        let mut block_mag_sum = vec![0.0f32; (bw * bh) as usize];
        let mut block_count = vec![0u32; (bw * bh) as usize];

        for (idx, &(vx, vy)) in motion_vectors.iter().enumerate() {
            let px = (idx % vecs_per_row) as u32;
            let py = (idx / vecs_per_row) as u32;

            // Clamp to image bounds
            if px >= width || py >= height {
                continue;
            }

            let bx = px / BLOCK_SIZE;
            let by = py / BLOCK_SIZE;
            if bx >= bw || by >= bh {
                continue;
            }

            let mag = (vx * vx + vy * vy).sqrt();
            let bidx = (by * bw + bx) as usize;
            block_mag_sum[bidx] += mag;
            block_count[bidx] += 1;
        }

        // Compute max for normalization
        let max_avg = block_mag_sum
            .iter()
            .zip(block_count.iter())
            .filter(|(_, &c)| c > 0)
            .map(|(&s, &c)| s / c as f32)
            .fold(0.0f32, f32::max);

        for (i, val) in map.values.iter_mut().enumerate() {
            if block_count[i] > 0 && max_avg > 0.0 {
                *val = (block_mag_sum[i] / block_count[i] as f32 / max_avg).min(1.0);
            }
        }

        map
    }

    /// Segment the density map into crowd regions.
    #[must_use]
    pub fn segment_regions(map: &DensityMap, block_pixel_size: u32) -> Vec<CrowdRegion> {
        let mut regions = Vec::new();
        for by in 0..map.height {
            for bx in 0..map.width {
                let raw = map.get(bx, by);
                let density = CrowdDensity::from_raw(raw);
                if density != CrowdDensity::Empty {
                    let px = bx * block_pixel_size;
                    let py = by * block_pixel_size;
                    let count = (raw * 4.0 + 0.5) as u32; // rough estimate
                    regions.push(CrowdRegion {
                        bbox: (px, py, block_pixel_size, block_pixel_size),
                        density,
                        estimated_count: count,
                    });
                }
            }
        }
        regions
    }
}

// ─────────────────────────────────────────────────────────────
// CrowdFlowVector
// ─────────────────────────────────────────────────────────────

/// A directional crowd flow vector covering a spatial region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrowdFlowVector {
    /// Flow direction in degrees (0 = right, 90 = down, counter-clockwise)
    pub direction_deg: f32,
    /// Flow speed (magnitude, arbitrary units)
    pub speed: f32,
    /// Spatial region `(x, y, width, height)`
    pub region: (u32, u32, u32, u32),
}

impl CrowdFlowVector {
    /// Returns true if flow is significant (speed > 0.1).
    #[must_use]
    pub fn is_flowing(&self) -> bool {
        self.speed > 0.1
    }
}

// ─────────────────────────────────────────────────────────────
// CrowdFlowAnalyzer
// ─────────────────────────────────────────────────────────────

/// Compute crowd flow vectors from consecutive density maps.
pub struct CrowdFlowAnalyzer;

impl CrowdFlowAnalyzer {
    /// Compute per-block flow vectors from two density maps.
    ///
    /// Flow direction and speed are derived from the density gradient.
    #[must_use]
    pub fn compute_flow(
        prev_density: &DensityMap,
        curr_density: &DensityMap,
    ) -> Vec<CrowdFlowVector> {
        if prev_density.width != curr_density.width
            || prev_density.height != curr_density.height
            || prev_density.values.is_empty()
        {
            return Vec::new();
        }

        let bw = prev_density.width as usize;
        let bh = prev_density.height as usize;
        let mut flows = Vec::new();

        for by in 0..bh {
            for bx in 0..bw {
                let idx = by * bw + bx;
                let delta = curr_density.values[idx] - prev_density.values[idx];

                // Compute spatial gradient in current map
                let left = if bx > 0 {
                    curr_density.values[by * bw + (bx - 1)]
                } else {
                    curr_density.values[idx]
                };
                let right = if bx + 1 < bw {
                    curr_density.values[by * bw + (bx + 1)]
                } else {
                    curr_density.values[idx]
                };
                let top = if by > 0 {
                    curr_density.values[(by - 1) * bw + bx]
                } else {
                    curr_density.values[idx]
                };
                let bottom = if by + 1 < bh {
                    curr_density.values[(by + 1) * bw + bx]
                } else {
                    curr_density.values[idx]
                };

                let gx = right - left;
                let gy = bottom - top;

                // Speed is derived from temporal density change magnitude
                let speed = delta.abs() + (gx * gx + gy * gy).sqrt() * 0.5;

                // Direction in degrees (atan2)
                let direction_deg = if gx.abs() > 1e-6 || gy.abs() > 1e-6 {
                    gy.atan2(gx).to_degrees()
                } else {
                    0.0
                };

                flows.push(CrowdFlowVector {
                    direction_deg,
                    speed,
                    region: (bx as u32, by as u32, 1, 1),
                });
            }
        }

        flows
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CrowdDensity ──────────────────────────────────────────

    #[test]
    fn test_crowd_density_people_per_m2() {
        assert!((CrowdDensity::Empty.people_per_m2() - 0.0).abs() < f32::EPSILON);
        assert!(CrowdDensity::VeryDense.people_per_m2() > CrowdDensity::Dense.people_per_m2());
    }

    #[test]
    fn test_crowd_density_from_raw_empty() {
        assert_eq!(CrowdDensity::from_raw(0.0), CrowdDensity::Empty);
        assert_eq!(CrowdDensity::from_raw(0.04), CrowdDensity::Empty);
    }

    #[test]
    fn test_crowd_density_from_raw_sparse() {
        assert_eq!(CrowdDensity::from_raw(0.1), CrowdDensity::Sparse);
    }

    #[test]
    fn test_crowd_density_from_raw_moderate() {
        assert_eq!(CrowdDensity::from_raw(0.4), CrowdDensity::Moderate);
    }

    #[test]
    fn test_crowd_density_from_raw_very_dense() {
        assert_eq!(CrowdDensity::from_raw(0.9), CrowdDensity::VeryDense);
    }

    #[test]
    fn test_crowd_density_names() {
        assert_eq!(CrowdDensity::Moderate.name(), "Moderate");
        assert_eq!(CrowdDensity::VeryDense.name(), "VeryDense");
    }

    // ── CrowdRegion ───────────────────────────────────────────

    #[test]
    fn test_crowd_region_area() {
        let r = CrowdRegion {
            bbox: (0, 0, 32, 32),
            density: CrowdDensity::Moderate,
            estimated_count: 2,
        };
        assert_eq!(r.area(), 1024);
    }

    // ── DensityMap ────────────────────────────────────────────

    #[test]
    fn test_density_map_new() {
        let m = DensityMap::new(4, 4);
        assert_eq!(m.values.len(), 16);
        assert!(m.total_density().abs() < f32::EPSILON);
    }

    #[test]
    fn test_density_map_total() {
        let m = DensityMap {
            width: 2,
            height: 2,
            values: vec![0.25, 0.50, 0.10, 0.15],
        };
        let total = m.total_density();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_density_map_peak_region() {
        let m = DensityMap {
            width: 2,
            height: 2,
            values: vec![0.1, 0.9, 0.3, 0.2],
        };
        let (bx, by, val) = m.peak_region(16);
        assert_eq!(bx, 1);
        assert_eq!(by, 0);
        assert!((val - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_density_map_average() {
        let m = DensityMap {
            width: 2,
            height: 2,
            values: vec![0.0, 0.5, 0.5, 1.0],
        };
        let avg = m.average_density();
        assert!((avg - 0.5).abs() < 1e-5);
    }

    // ── CrowdAnalyzer ─────────────────────────────────────────

    #[test]
    fn test_estimate_from_motion_empty() {
        let map = CrowdAnalyzer::estimate_from_motion(&[], 100, 100);
        assert_eq!(map.total_density(), 0.0);
    }

    #[test]
    fn test_estimate_from_motion_uniform() {
        let mvs: Vec<(f32, f32)> = vec![(1.0, 0.0); 100 * 100];
        let map = CrowdAnalyzer::estimate_from_motion(&mvs, 100, 100);
        // All blocks should have the same density
        let total = map.total_density();
        assert!(total > 0.0);
    }

    #[test]
    fn test_estimate_from_motion_zero_vectors() {
        let mvs: Vec<(f32, f32)> = vec![(0.0, 0.0); 100 * 100];
        let map = CrowdAnalyzer::estimate_from_motion(&mvs, 100, 100);
        assert_eq!(map.total_density(), 0.0);
    }

    // ── CrowdFlowVector ───────────────────────────────────────

    #[test]
    fn test_is_flowing_true() {
        let v = CrowdFlowVector {
            direction_deg: 45.0,
            speed: 0.5,
            region: (0, 0, 1, 1),
        };
        assert!(v.is_flowing());
    }

    #[test]
    fn test_is_flowing_false() {
        let v = CrowdFlowVector {
            direction_deg: 0.0,
            speed: 0.05,
            region: (0, 0, 1, 1),
        };
        assert!(!v.is_flowing());
    }

    // ── CrowdFlowAnalyzer ─────────────────────────────────────

    #[test]
    fn test_compute_flow_equal_maps() {
        let map = DensityMap {
            width: 2,
            height: 2,
            values: vec![0.5, 0.5, 0.5, 0.5],
        };
        let flows = CrowdFlowAnalyzer::compute_flow(&map, &map);
        // Equal maps → delta = 0 everywhere
        assert_eq!(flows.len(), 4);
        for f in &flows {
            assert!(f.speed < 0.5); // should be low
        }
    }

    #[test]
    fn test_compute_flow_mismatched_size() {
        let m1 = DensityMap::new(2, 2);
        let m2 = DensityMap::new(3, 3);
        let flows = CrowdFlowAnalyzer::compute_flow(&m1, &m2);
        assert!(flows.is_empty());
    }

    #[test]
    fn test_compute_flow_increasing_density() {
        let prev = DensityMap {
            width: 2,
            height: 2,
            values: vec![0.1, 0.1, 0.1, 0.1],
        };
        let curr = DensityMap {
            width: 2,
            height: 2,
            values: vec![0.5, 0.9, 0.2, 0.3],
        };
        let flows = CrowdFlowAnalyzer::compute_flow(&prev, &curr);
        assert_eq!(flows.len(), 4);
        // At least some flows should be non-zero
        let any_flowing = flows.iter().any(|f| f.is_flowing());
        assert!(any_flowing);
    }
}
