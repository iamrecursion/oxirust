#![allow(dead_code)]
//! Parallax compensation for video stabilization.
//!
//! When a camera translates (rather than purely rotating), objects at different
//! depths exhibit different amounts of apparent motion (parallax). Applying a
//! single global stabilization transform to such footage causes artifacts at
//! depth discontinuities. This module provides tools to detect, estimate, and
//! compensate for parallax effects during stabilization.
//!
//! # Features
//!
//! - **Depth layer segmentation**: Classify pixels into near/far layers
//! - **Per-layer motion estimation**: Independent motion for each depth layer
//! - **Blended compensation**: Smooth transitions between depth layers
//! - **Parallax magnitude estimation**: Quantify parallax severity

/// Depth layer classification for parallax analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthLayer {
    /// Near foreground objects.
    Near,
    /// Mid-range objects.
    Mid,
    /// Far background objects.
    Far,
}

/// Configuration for parallax compensation.
#[derive(Debug, Clone)]
pub struct ParallaxConfig {
    /// Number of depth layers to segment.
    pub num_layers: usize,
    /// Block size for motion estimation (pixels).
    pub block_size: usize,
    /// Search range for block matching (pixels).
    pub search_range: usize,
    /// Blending margin width between layers (pixels).
    pub blend_margin: usize,
    /// Minimum parallax magnitude to trigger compensation.
    pub min_parallax_threshold: f64,
    /// Temporal smoothing factor for layer motion.
    pub temporal_smoothing: f64,
}

impl Default for ParallaxConfig {
    fn default() -> Self {
        Self {
            num_layers: 3,
            block_size: 16,
            search_range: 32,
            blend_margin: 8,
            min_parallax_threshold: 2.0,
            temporal_smoothing: 0.7,
        }
    }
}

impl ParallaxConfig {
    /// Create a new parallax configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of depth layers.
    #[must_use]
    pub const fn with_num_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    /// Set the block size for motion estimation.
    #[must_use]
    pub const fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }

    /// Set the search range.
    #[must_use]
    pub const fn with_search_range(mut self, range: usize) -> Self {
        self.search_range = range;
        self
    }

    /// Set the blending margin between layers.
    #[must_use]
    pub const fn with_blend_margin(mut self, margin: usize) -> Self {
        self.blend_margin = margin;
        self
    }
}

/// A 2D motion vector.
#[derive(Debug, Clone, Copy)]
pub struct MotionVec {
    /// Horizontal displacement.
    pub dx: f64,
    /// Vertical displacement.
    pub dy: f64,
}

impl MotionVec {
    /// Create a new motion vector.
    #[must_use]
    pub const fn new(dx: f64, dy: f64) -> Self {
        Self { dx, dy }
    }

    /// Zero motion vector.
    #[must_use]
    pub const fn zero() -> Self {
        Self { dx: 0.0, dy: 0.0 }
    }

    /// Compute the magnitude of the motion vector.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Compute the difference between two motion vectors.
    #[must_use]
    pub fn diff(&self, other: &Self) -> Self {
        Self {
            dx: self.dx - other.dx,
            dy: self.dy - other.dy,
        }
    }
}

/// Per-layer motion estimate for a single frame pair.
#[derive(Debug, Clone)]
pub struct LayerMotion {
    /// Which depth layer this motion corresponds to.
    pub layer: DepthLayer,
    /// Estimated global motion for this layer.
    pub motion: MotionVec,
    /// Confidence of the estimate (0.0..=1.0).
    pub confidence: f64,
    /// Number of blocks used for estimation.
    pub block_count: usize,
}

/// Depth map representation (single-channel, row-major).
#[derive(Debug, Clone)]
pub struct DepthMap {
    /// Width of the depth map.
    pub width: usize,
    /// Height of the depth map.
    pub height: usize,
    /// Depth values (0.0 = near, 1.0 = far).
    pub data: Vec<f64>,
}

impl DepthMap {
    /// Create a new depth map with given dimensions and uniform depth.
    #[must_use]
    pub fn uniform(width: usize, height: usize, depth: f64) -> Self {
        Self {
            width,
            height,
            data: vec![depth; width * height],
        }
    }

    /// Get the depth at a specific pixel coordinate.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> Option<f64> {
        if x < self.width && y < self.height {
            Some(self.data[y * self.width + x])
        } else {
            None
        }
    }

    /// Set the depth at a specific pixel coordinate.
    pub fn set(&mut self, x: usize, y: usize, val: f64) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = val;
        }
    }

    /// Classify a pixel into a depth layer based on thresholds.
    #[must_use]
    pub fn classify(&self, x: usize, y: usize, num_layers: usize) -> Option<usize> {
        self.get(x, y).map(|d| {
            let layer = (d * num_layers as f64).floor() as usize;
            layer.min(num_layers.saturating_sub(1))
        })
    }
}

/// Parallax compensator for multi-layer stabilization.
#[derive(Debug)]
pub struct ParallaxCompensator {
    /// Configuration.
    config: ParallaxConfig,
    /// Previous frame data for inter-frame analysis.
    prev_frame: Option<Vec<u8>>,
    /// Previous frame dimensions (width, height).
    prev_dims: (usize, usize),
    /// Accumulated per-layer motion estimates.
    layer_motions: Vec<Vec<MotionVec>>,
}

impl ParallaxCompensator {
    /// Create a new parallax compensator.
    #[must_use]
    pub fn new(config: ParallaxConfig) -> Self {
        let num_layers = config.num_layers;
        Self {
            config,
            prev_frame: None,
            prev_dims: (0, 0),
            layer_motions: vec![Vec::new(); num_layers],
        }
    }

    /// Create a compensator with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(ParallaxConfig::default())
    }

    /// Estimate parallax magnitude between two frames.
    ///
    /// Returns the maximum motion difference between depth layers.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn estimate_parallax(layer_motions: &[LayerMotion]) -> f64 {
        if layer_motions.len() < 2 {
            return 0.0;
        }
        let mut max_diff = 0.0_f64;
        for i in 0..layer_motions.len() {
            for j in (i + 1)..layer_motions.len() {
                let diff = layer_motions[i].motion.diff(&layer_motions[j].motion);
                max_diff = max_diff.max(diff.magnitude());
            }
        }
        max_diff
    }

    /// Perform block matching for a single block.
    ///
    /// Returns the best motion vector and the matching cost.
    #[allow(clippy::cast_precision_loss)]
    pub fn block_match(
        prev: &[u8],
        curr: &[u8],
        width: usize,
        height: usize,
        bx: usize,
        by: usize,
        block_size: usize,
        search_range: usize,
    ) -> (MotionVec, f64) {
        let mut best_mv = MotionVec::zero();
        let mut best_cost = f64::MAX;
        let sr = search_range as isize;

        for dy in -sr..=sr {
            for dx in -sr..=sr {
                let cost = Self::block_sad(prev, curr, width, height, bx, by, block_size, dx, dy);
                if cost < best_cost {
                    best_cost = cost;
                    best_mv = MotionVec::new(dx as f64, dy as f64);
                }
            }
        }
        (best_mv, best_cost)
    }

    /// Compute Sum of Absolute Differences for a block at given offset.
    #[allow(clippy::cast_precision_loss)]
    fn block_sad(
        prev: &[u8],
        curr: &[u8],
        width: usize,
        height: usize,
        bx: usize,
        by: usize,
        block_size: usize,
        dx: isize,
        dy: isize,
    ) -> f64 {
        let mut sad = 0.0_f64;
        let mut count = 0u32;
        for row in 0..block_size {
            for col in 0..block_size {
                let cy = by + row;
                let cx = bx + col;
                let py = (by as isize + row as isize + dy) as usize;
                let px = (bx as isize + col as isize + dx) as usize;
                if cy < height && cx < width && py < height && px < width {
                    let cv = f64::from(curr[cy * width + cx]);
                    let pv = f64::from(prev[py * width + px]);
                    sad += (cv - pv).abs();
                    count += 1;
                }
            }
        }
        if count == 0 {
            f64::MAX
        } else {
            sad / f64::from(count)
        }
    }

    /// Estimate per-layer motion from a depth map and two frames.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_layer_motions(
        &self,
        prev: &[u8],
        curr: &[u8],
        depth_map: &DepthMap,
        width: usize,
        height: usize,
    ) -> Vec<LayerMotion> {
        let bs = self.config.block_size.max(1);
        let sr = self.config.search_range;
        let n_layers = self.config.num_layers;

        let mut layer_dx: Vec<Vec<f64>> = vec![Vec::new(); n_layers];
        let mut layer_dy: Vec<Vec<f64>> = vec![Vec::new(); n_layers];

        let cols = width / bs;
        let rows = height / bs;

        for by_idx in 0..rows {
            for bx_idx in 0..cols {
                let bx = bx_idx * bs;
                let by = by_idx * bs;
                let center_x = bx + bs / 2;
                let center_y = by + bs / 2;
                let layer_idx = depth_map
                    .classify(center_x, center_y, n_layers)
                    .unwrap_or(0);
                let (mv, _cost) = Self::block_match(prev, curr, width, height, bx, by, bs, sr);
                layer_dx[layer_idx].push(mv.dx);
                layer_dy[layer_idx].push(mv.dy);
            }
        }

        let layers = [DepthLayer::Near, DepthLayer::Mid, DepthLayer::Far];
        (0..n_layers)
            .map(|i| {
                let count = layer_dx[i].len();
                if count == 0 {
                    LayerMotion {
                        layer: layers[i.min(2)],
                        motion: MotionVec::zero(),
                        confidence: 0.0,
                        block_count: 0,
                    }
                } else {
                    let avg_dx: f64 = layer_dx[i].iter().sum::<f64>() / count as f64;
                    let avg_dy: f64 = layer_dy[i].iter().sum::<f64>() / count as f64;
                    LayerMotion {
                        layer: layers[i.min(2)],
                        motion: MotionVec::new(avg_dx, avg_dy),
                        confidence: (count as f64 / (cols * rows) as f64).min(1.0),
                        block_count: count,
                    }
                }
            })
            .collect()
    }

    /// Compute a blended compensation vector for a given pixel based on depth.
    #[must_use]
    pub fn blend_compensation(
        layer_motions: &[LayerMotion],
        depth: f64,
        num_layers: usize,
    ) -> MotionVec {
        if layer_motions.is_empty() || num_layers == 0 {
            return MotionVec::zero();
        }
        let float_layer = depth * (num_layers as f64 - 1.0);
        let lower = float_layer.floor() as usize;
        let upper = (lower + 1).min(num_layers - 1).min(layer_motions.len() - 1);
        let lower = lower.min(layer_motions.len() - 1);
        let frac = float_layer - float_layer.floor();
        let dx =
            layer_motions[lower].motion.dx * (1.0 - frac) + layer_motions[upper].motion.dx * frac;
        let dy =
            layer_motions[lower].motion.dy * (1.0 - frac) + layer_motions[upper].motion.dy * frac;
        MotionVec::new(dx, dy)
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub const fn config(&self) -> &ParallaxConfig {
        &self.config
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.prev_frame = None;
        self.prev_dims = (0, 0);
        for motions in &mut self.layer_motions {
            motions.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = ParallaxConfig::default();
        assert_eq!(cfg.num_layers, 3);
        assert_eq!(cfg.block_size, 16);
        assert_eq!(cfg.search_range, 32);
    }

    #[test]
    fn test_config_builder() {
        let cfg = ParallaxConfig::new()
            .with_num_layers(5)
            .with_block_size(8)
            .with_search_range(16)
            .with_blend_margin(4);
        assert_eq!(cfg.num_layers, 5);
        assert_eq!(cfg.block_size, 8);
        assert_eq!(cfg.search_range, 16);
        assert_eq!(cfg.blend_margin, 4);
    }

    #[test]
    fn test_motion_vec_magnitude() {
        let mv = MotionVec::new(3.0, 4.0);
        assert!((mv.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_motion_vec_diff() {
        let a = MotionVec::new(5.0, 3.0);
        let b = MotionVec::new(2.0, 1.0);
        let d = a.diff(&b);
        assert!((d.dx - 3.0).abs() < 1e-10);
        assert!((d.dy - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_depth_map_uniform() {
        let dm = DepthMap::uniform(10, 10, 0.5);
        assert_eq!(dm.data.len(), 100);
        assert!((dm.get(5, 5).expect("should succeed in test") - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_depth_map_classify() {
        let mut dm = DepthMap::uniform(10, 10, 0.0);
        dm.set(5, 5, 0.8);
        let layer = dm.classify(5, 5, 3);
        assert_eq!(layer, Some(2)); // 0.8 * 3 = 2.4 → floor → 2
    }

    #[test]
    fn test_depth_map_out_of_bounds() {
        let dm = DepthMap::uniform(10, 10, 0.5);
        assert!(dm.get(20, 20).is_none());
        assert!(dm.classify(20, 20, 3).is_none());
    }

    #[test]
    fn test_estimate_parallax_no_layers() {
        let parallax = ParallaxCompensator::estimate_parallax(&[]);
        assert!((parallax - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_parallax_two_layers() {
        let layers = vec![
            LayerMotion {
                layer: DepthLayer::Near,
                motion: MotionVec::new(10.0, 0.0),
                confidence: 0.9,
                block_count: 50,
            },
            LayerMotion {
                layer: DepthLayer::Far,
                motion: MotionVec::new(2.0, 0.0),
                confidence: 0.8,
                block_count: 50,
            },
        ];
        let parallax = ParallaxCompensator::estimate_parallax(&layers);
        assert!((parallax - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_block_match_identical() {
        // Use a textured frame (non-constant) so the best match is uniquely at (0,0)
        let mut frame = vec![0u8; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                frame[y * 64 + x] = ((x * 7 + y * 13) % 256) as u8;
            }
        }
        let (mv, cost) = ParallaxCompensator::block_match(&frame, &frame, 64, 64, 16, 16, 8, 4);
        assert!((mv.dx).abs() < 1e-10);
        assert!((mv.dy).abs() < 1e-10);
        assert!(cost < 1e-10);
    }

    #[test]
    fn test_blend_compensation() {
        let layers = vec![
            LayerMotion {
                layer: DepthLayer::Near,
                motion: MotionVec::new(10.0, 0.0),
                confidence: 1.0,
                block_count: 100,
            },
            LayerMotion {
                layer: DepthLayer::Mid,
                motion: MotionVec::new(5.0, 0.0),
                confidence: 1.0,
                block_count: 100,
            },
            LayerMotion {
                layer: DepthLayer::Far,
                motion: MotionVec::new(0.0, 0.0),
                confidence: 1.0,
                block_count: 100,
            },
        ];
        let mv = ParallaxCompensator::blend_compensation(&layers, 0.5, 3);
        assert!((mv.dx - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_compensator_reset() {
        let mut comp = ParallaxCompensator::with_defaults();
        comp.prev_frame = Some(vec![0u8; 100]);
        comp.reset();
        assert!(comp.prev_frame.is_none());
    }
}
