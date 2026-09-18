#![allow(dead_code)]
//! Region of Interest (ROI) encoding optimization.
//!
//! This module provides tools for defining regions of interest within video frames
//! and adjusting encoding parameters (QP offsets, bitrate allocation) to prioritize
//! visual quality in those regions. Common use cases include face-aware encoding,
//! text region preservation, and broadcast graphics protection.
//!
//! The ROI encoder integrates with the main [`Optimizer`](crate::Optimizer) pipeline
//! via [`RoiOptimizeResult`], which provides per-CTU QP adjustments that the
//! optimizer applies on top of its base AQ decisions.

/// Coordinate type for ROI regions.
#[allow(clippy::cast_precision_loss)]
type Coord = i32;

/// A rectangular region of interest within a frame.
#[derive(Debug, Clone, PartialEq)]
pub struct RoiRegion {
    /// Left edge in pixels.
    pub x: Coord,
    /// Top edge in pixels.
    pub y: Coord,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Priority weight (0.0 = ignore, 1.0 = normal, >1.0 = boosted).
    pub priority: f64,
    /// Optional label for the region.
    pub label: String,
}

impl RoiRegion {
    /// Creates a new ROI region with default priority.
    pub fn new(x: Coord, y: Coord, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            priority: 1.0,
            label: String::new(),
        }
    }

    /// Creates a new ROI region with a given priority weight.
    pub fn with_priority(x: Coord, y: Coord, width: u32, height: u32, priority: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
            priority,
            label: String::new(),
        }
    }

    /// Sets the label for this region.
    pub fn set_label(&mut self, label: &str) {
        self.label = label.to_string();
    }

    /// Returns the area of the region in pixels.
    #[allow(clippy::cast_precision_loss)]
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Returns the right edge coordinate.
    pub fn right(&self) -> Coord {
        self.x + self.width as Coord
    }

    /// Returns the bottom edge coordinate.
    pub fn bottom(&self) -> Coord {
        self.y + self.height as Coord
    }

    /// Checks whether a pixel coordinate falls inside this region.
    pub fn contains(&self, px: Coord, py: Coord) -> bool {
        px >= self.x && px < self.right() && py >= self.y && py < self.bottom()
    }

    /// Checks whether two regions overlap.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.x < other.right()
            && self.right() > other.x
            && self.y < other.bottom()
            && self.bottom() > other.y
    }

    /// Returns the intersection area with another region, or 0 if they don't overlap.
    pub fn intersection_area(&self, other: &Self) -> u64 {
        let left = self.x.max(other.x);
        let top = self.y.max(other.y);
        let right = self.right().min(other.right());
        let bottom = self.bottom().min(other.bottom());
        if right > left && bottom > top {
            (right - left) as u64 * (bottom - top) as u64
        } else {
            0
        }
    }
}

/// QP adjustment mode for ROI regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QpAdjustMode {
    /// Apply an absolute QP offset (delta).
    AbsoluteOffset,
    /// Scale the base QP by a factor derived from priority.
    PriorityScale,
    /// Use a fixed QP value for the region regardless of base QP.
    FixedQp,
}

/// Configuration for the ROI encoder optimizer.
#[derive(Debug, Clone)]
pub struct RoiEncoderConfig {
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
    /// CTU (Coding Tree Unit) size for block-level QP mapping.
    pub ctu_size: u32,
    /// Base QP for the frame.
    pub base_qp: u8,
    /// Maximum negative QP offset allowed for boosted regions.
    pub max_qp_reduction: u8,
    /// Maximum positive QP offset allowed for background regions.
    pub max_qp_increase: u8,
    /// QP adjustment mode.
    pub adjust_mode: QpAdjustMode,
}

impl Default for RoiEncoderConfig {
    fn default() -> Self {
        Self {
            frame_width: 1920,
            frame_height: 1080,
            ctu_size: 64,
            base_qp: 28,
            max_qp_reduction: 10,
            max_qp_increase: 6,
            adjust_mode: QpAdjustMode::PriorityScale,
        }
    }
}

/// A QP delta map for a single frame, organized by CTU blocks.
#[derive(Debug, Clone)]
pub struct QpDeltaMap {
    /// Number of CTU columns.
    pub cols: usize,
    /// Number of CTU rows.
    pub rows: usize,
    /// QP deltas, stored row-major.
    pub deltas: Vec<i8>,
}

impl QpDeltaMap {
    /// Creates a new zero-filled QP delta map.
    pub fn new(cols: usize, rows: usize) -> Self {
        Self {
            cols,
            rows,
            deltas: vec![0; cols * rows],
        }
    }

    /// Gets the delta for a specific CTU.
    pub fn get(&self, col: usize, row: usize) -> i8 {
        if col < self.cols && row < self.rows {
            self.deltas[row * self.cols + col]
        } else {
            0
        }
    }

    /// Sets the delta for a specific CTU.
    pub fn set(&mut self, col: usize, row: usize, delta: i8) {
        if col < self.cols && row < self.rows {
            self.deltas[row * self.cols + col] = delta;
        }
    }

    /// Returns the average delta across the map.
    #[allow(clippy::cast_precision_loss)]
    pub fn average_delta(&self) -> f64 {
        if self.deltas.is_empty() {
            return 0.0;
        }
        let sum: i64 = self.deltas.iter().map(|&d| i64::from(d)).sum();
        sum as f64 / self.deltas.len() as f64
    }

    /// Returns the number of CTUs with non-zero deltas.
    pub fn active_ctu_count(&self) -> usize {
        self.deltas.iter().filter(|&&d| d != 0).count()
    }

    /// Merges another QP delta map, adding deltas element-wise with clamping.
    pub fn merge_additive(&mut self, other: &Self, max_magnitude: i8) {
        if self.cols != other.cols || self.rows != other.rows {
            return;
        }
        for i in 0..self.deltas.len() {
            let sum = i16::from(self.deltas[i]) + i16::from(other.deltas[i]);
            self.deltas[i] = sum.clamp(i16::from(-max_magnitude), i16::from(max_magnitude)) as i8;
        }
    }
}

/// The ROI encoder optimizer generates per-CTU QP delta maps from ROI regions.
#[derive(Debug)]
pub struct RoiEncoder {
    /// Encoder configuration.
    config: RoiEncoderConfig,
    /// Current set of ROI regions.
    regions: Vec<RoiRegion>,
}

impl RoiEncoder {
    /// Creates a new ROI encoder with the given configuration.
    pub fn new(config: RoiEncoderConfig) -> Self {
        Self {
            config,
            regions: Vec::new(),
        }
    }

    /// Adds a region of interest.
    pub fn add_region(&mut self, region: RoiRegion) {
        self.regions.push(region);
    }

    /// Clears all ROI regions.
    pub fn clear_regions(&mut self) {
        self.regions.clear();
    }

    /// Returns the number of configured regions.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Returns the current regions.
    pub fn regions(&self) -> &[RoiRegion] {
        &self.regions
    }

    /// Returns the encoder configuration.
    pub fn config(&self) -> &RoiEncoderConfig {
        &self.config
    }

    /// Generates a QP delta map for the current set of regions.
    #[allow(clippy::cast_precision_loss)]
    pub fn generate_qp_map(&self) -> QpDeltaMap {
        let cols =
            ((self.config.frame_width + self.config.ctu_size - 1) / self.config.ctu_size) as usize;
        let rows =
            ((self.config.frame_height + self.config.ctu_size - 1) / self.config.ctu_size) as usize;
        let mut map = QpDeltaMap::new(cols, rows);

        if self.regions.is_empty() {
            return map;
        }

        for row in 0..rows {
            for col in 0..cols {
                let ctu_x = (col as u32 * self.config.ctu_size) as Coord;
                let ctu_y = (row as u32 * self.config.ctu_size) as Coord;
                let ctu_region =
                    RoiRegion::new(ctu_x, ctu_y, self.config.ctu_size, self.config.ctu_size);

                let mut max_priority: f64 = 0.0;
                for region in &self.regions {
                    if region.overlaps(&ctu_region) {
                        let overlap = region.intersection_area(&ctu_region);
                        let ctu_area = ctu_region.area();
                        let coverage = if ctu_area > 0 {
                            overlap as f64 / ctu_area as f64
                        } else {
                            0.0
                        };
                        let effective = region.priority * coverage;
                        if effective > max_priority {
                            max_priority = effective;
                        }
                    }
                }

                let delta = self.compute_delta(max_priority);
                map.set(col, row, delta);
            }
        }

        map
    }

    /// Generates an [`RoiOptimizeResult`] for pipeline integration with the main Optimizer.
    ///
    /// This produces a result that includes both the QP delta map and per-block
    /// quality weights that the Optimizer uses to adjust AQ decisions.
    #[allow(clippy::cast_precision_loss)]
    pub fn optimize_frame(&self) -> RoiOptimizeResult {
        let map = self.generate_qp_map();
        let analysis = analyze_qp_map(&map);

        // Compute per-CTU quality weights (higher priority = lower weight = more bits)
        let quality_weights: Vec<f64> = map
            .deltas
            .iter()
            .map(|&d| {
                // Convert delta to weight: negative delta (boost) -> higher weight
                let w = 1.0 - f64::from(d) / f64::from(self.config.max_qp_reduction as i8);
                w.clamp(0.5, 2.0)
            })
            .collect();

        let bitrate_impact = self.estimate_bitrate_impact();

        RoiOptimizeResult {
            qp_map: map,
            quality_weights,
            analysis,
            estimated_bitrate_change: bitrate_impact,
            has_active_regions: !self.regions.is_empty(),
        }
    }

    /// Computes a QP delta from a priority value.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    fn compute_delta(&self, priority: f64) -> i8 {
        if priority <= 0.0 {
            return 0;
        }
        match self.config.adjust_mode {
            QpAdjustMode::AbsoluteOffset => {
                let offset = -(priority * f64::from(self.config.max_qp_reduction));
                offset.round().max(-(i8::MAX as f64)).min(0.0) as i8
            }
            QpAdjustMode::PriorityScale => {
                if priority > 1.0 {
                    let reduction =
                        (priority - 1.0).min(1.0) * f64::from(self.config.max_qp_reduction);
                    -(reduction
                        .round()
                        .min(f64::from(self.config.max_qp_reduction)) as i8)
                } else if priority < 1.0 {
                    let increase = (1.0 - priority) * f64::from(self.config.max_qp_increase);
                    increase.round().min(f64::from(self.config.max_qp_increase)) as i8
                } else {
                    0
                }
            }
            QpAdjustMode::FixedQp => {
                let target_qp = (f64::from(self.config.base_qp) / priority).round() as i16;
                let delta = target_qp - i16::from(self.config.base_qp);
                delta
                    .max(-i16::from(self.config.max_qp_reduction))
                    .min(i16::from(self.config.max_qp_increase)) as i8
            }
        }
    }

    /// Estimates the bitrate savings/cost relative to uniform encoding.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_bitrate_impact(&self) -> f64 {
        let map = self.generate_qp_map();
        let avg_delta = map.average_delta();
        // Rough model: each QP unit ~= 12% bitrate change
        let factor = 2.0_f64.powf(-avg_delta / 6.0);
        factor - 1.0
    }
}

/// Result of ROI optimization for integration with the main Optimizer pipeline.
#[derive(Debug, Clone)]
pub struct RoiOptimizeResult {
    /// Per-CTU QP delta map.
    pub qp_map: QpDeltaMap,
    /// Per-CTU quality weights (1.0 = normal, >1.0 = boosted region).
    pub quality_weights: Vec<f64>,
    /// Summary analysis of the QP map.
    pub analysis: RoiAnalysisSummary,
    /// Estimated bitrate change factor (0.0 = no change, positive = more bits).
    pub estimated_bitrate_change: f64,
    /// Whether any active ROI regions exist.
    pub has_active_regions: bool,
}

impl RoiOptimizeResult {
    /// Creates an empty result with no active regions.
    pub fn empty(cols: usize, rows: usize) -> Self {
        Self {
            qp_map: QpDeltaMap::new(cols, rows),
            quality_weights: vec![1.0; cols * rows],
            analysis: RoiAnalysisSummary {
                total_ctus: cols * rows,
                roi_ctus: 0,
                avg_delta: 0.0,
                min_delta: 0,
                max_delta: 0,
            },
            estimated_bitrate_change: 0.0,
            has_active_regions: false,
        }
    }

    /// Returns the QP delta for a specific CTU position.
    pub fn qp_delta_at(&self, col: usize, row: usize) -> i8 {
        self.qp_map.get(col, row)
    }

    /// Returns the quality weight for a specific CTU position.
    #[allow(clippy::cast_precision_loss)]
    pub fn quality_weight_at(&self, col: usize, row: usize) -> f64 {
        if col < self.qp_map.cols && row < self.qp_map.rows {
            let idx = row * self.qp_map.cols + col;
            if idx < self.quality_weights.len() {
                return self.quality_weights[idx];
            }
        }
        1.0
    }
}

/// Summary statistics for ROI encoding analysis.
#[derive(Debug, Clone)]
pub struct RoiAnalysisSummary {
    /// Total number of CTUs in the frame.
    pub total_ctus: usize,
    /// Number of CTUs touched by at least one ROI.
    pub roi_ctus: usize,
    /// Average QP delta across the frame.
    pub avg_delta: f64,
    /// Minimum delta (most boosted).
    pub min_delta: i8,
    /// Maximum delta (most reduced quality).
    pub max_delta: i8,
}

/// Analyzes a QP delta map and returns summary statistics.
pub fn analyze_qp_map(map: &QpDeltaMap) -> RoiAnalysisSummary {
    let total_ctus = map.deltas.len();
    let roi_ctus = map.active_ctu_count();
    let avg_delta = map.average_delta();
    let min_delta = map.deltas.iter().copied().min().unwrap_or(0);
    let max_delta = map.deltas.iter().copied().max().unwrap_or(0);
    RoiAnalysisSummary {
        total_ctus,
        roi_ctus,
        avg_delta,
        min_delta,
        max_delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roi_region_new() {
        let r = RoiRegion::new(10, 20, 100, 200);
        assert_eq!(r.x, 10);
        assert_eq!(r.y, 20);
        assert_eq!(r.width, 100);
        assert_eq!(r.height, 200);
        assert!((r.priority - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_roi_region_with_priority() {
        let r = RoiRegion::with_priority(0, 0, 50, 50, 2.5);
        assert!((r.priority - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_roi_region_area() {
        let r = RoiRegion::new(0, 0, 100, 200);
        assert_eq!(r.area(), 20_000);
    }

    #[test]
    fn test_roi_region_contains() {
        let r = RoiRegion::new(10, 10, 50, 50);
        assert!(r.contains(10, 10));
        assert!(r.contains(30, 30));
        assert!(!r.contains(60, 60));
        assert!(!r.contains(9, 10));
    }

    #[test]
    fn test_roi_region_overlaps() {
        let a = RoiRegion::new(0, 0, 100, 100);
        let b = RoiRegion::new(50, 50, 100, 100);
        assert!(a.overlaps(&b));

        let c = RoiRegion::new(200, 200, 10, 10);
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_roi_region_intersection_area() {
        let a = RoiRegion::new(0, 0, 100, 100);
        let b = RoiRegion::new(50, 50, 100, 100);
        assert_eq!(a.intersection_area(&b), 50 * 50);

        let c = RoiRegion::new(200, 200, 10, 10);
        assert_eq!(a.intersection_area(&c), 0);
    }

    #[test]
    fn test_roi_region_label() {
        let mut r = RoiRegion::new(0, 0, 10, 10);
        r.set_label("face");
        assert_eq!(r.label, "face");
    }

    #[test]
    fn test_qp_delta_map_new() {
        let map = QpDeltaMap::new(3, 2);
        assert_eq!(map.cols, 3);
        assert_eq!(map.rows, 2);
        assert_eq!(map.deltas.len(), 6);
    }

    #[test]
    fn test_qp_delta_map_get_set() {
        let mut map = QpDeltaMap::new(4, 4);
        map.set(1, 2, -5);
        assert_eq!(map.get(1, 2), -5);
        assert_eq!(map.get(0, 0), 0);
        // Out of bounds returns 0
        assert_eq!(map.get(10, 10), 0);
    }

    #[test]
    fn test_qp_delta_map_average() {
        let mut map = QpDeltaMap::new(2, 2);
        map.set(0, 0, -4);
        map.set(1, 0, -4);
        map.set(0, 1, 0);
        map.set(1, 1, 0);
        assert!((map.average_delta() - (-2.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_qp_delta_map_active_count() {
        let mut map = QpDeltaMap::new(3, 3);
        map.set(0, 0, -2);
        map.set(2, 2, 3);
        assert_eq!(map.active_ctu_count(), 2);
    }

    #[test]
    fn test_qp_delta_map_merge_additive() {
        let mut map1 = QpDeltaMap::new(2, 2);
        map1.set(0, 0, -3);
        map1.set(1, 1, 2);

        let mut map2 = QpDeltaMap::new(2, 2);
        map2.set(0, 0, -4);
        map2.set(1, 1, 3);

        map1.merge_additive(&map2, 6);
        assert_eq!(map1.get(0, 0), -6); // clamped to -6
        assert_eq!(map1.get(1, 1), 5);
    }

    #[test]
    fn test_roi_encoder_empty_regions() {
        let config = RoiEncoderConfig {
            frame_width: 128,
            frame_height: 128,
            ctu_size: 64,
            ..Default::default()
        };
        let enc = RoiEncoder::new(config);
        let map = enc.generate_qp_map();
        assert_eq!(map.cols, 2);
        assert_eq!(map.rows, 2);
        assert_eq!(map.active_ctu_count(), 0);
    }

    #[test]
    fn test_roi_encoder_with_region() {
        let config = RoiEncoderConfig {
            frame_width: 256,
            frame_height: 256,
            ctu_size: 64,
            adjust_mode: QpAdjustMode::AbsoluteOffset,
            max_qp_reduction: 6,
            ..Default::default()
        };
        let mut enc = RoiEncoder::new(config);
        enc.add_region(RoiRegion::with_priority(0, 0, 64, 64, 1.0));
        let map = enc.generate_qp_map();
        // The CTU at (0,0) should have a negative delta
        assert!(map.get(0, 0) < 0);
    }

    #[test]
    fn test_analyze_qp_map() {
        let mut map = QpDeltaMap::new(4, 4);
        map.set(0, 0, -3);
        map.set(3, 3, 2);
        let summary = analyze_qp_map(&map);
        assert_eq!(summary.total_ctus, 16);
        assert_eq!(summary.roi_ctus, 2);
        assert_eq!(summary.min_delta, -3);
        assert_eq!(summary.max_delta, 2);
    }

    #[test]
    fn test_roi_encoder_bitrate_impact_no_regions() {
        let config = RoiEncoderConfig {
            frame_width: 128,
            frame_height: 128,
            ctu_size: 64,
            ..Default::default()
        };
        let enc = RoiEncoder::new(config);
        let impact = enc.estimate_bitrate_impact();
        assert!(impact.abs() < f64::EPSILON);
    }

    #[test]
    fn test_roi_encoder_region_count() {
        let config = RoiEncoderConfig::default();
        let mut enc = RoiEncoder::new(config);
        assert_eq!(enc.region_count(), 0);
        enc.add_region(RoiRegion::new(0, 0, 100, 100));
        enc.add_region(RoiRegion::new(200, 200, 50, 50));
        assert_eq!(enc.region_count(), 2);
        enc.clear_regions();
        assert_eq!(enc.region_count(), 0);
    }

    // --- New tests for ROI pipeline integration ---

    #[test]
    fn test_roi_optimize_frame_no_regions() {
        let config = RoiEncoderConfig {
            frame_width: 128,
            frame_height: 128,
            ctu_size: 64,
            ..Default::default()
        };
        let enc = RoiEncoder::new(config);
        let result = enc.optimize_frame();
        assert!(!result.has_active_regions);
        assert_eq!(result.analysis.roi_ctus, 0);
        assert!((result.estimated_bitrate_change - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_roi_optimize_frame_with_regions() {
        let config = RoiEncoderConfig {
            frame_width: 256,
            frame_height: 256,
            ctu_size: 64,
            adjust_mode: QpAdjustMode::AbsoluteOffset,
            max_qp_reduction: 8,
            ..Default::default()
        };
        let mut enc = RoiEncoder::new(config);
        enc.add_region(RoiRegion::with_priority(0, 0, 128, 128, 1.5));
        let result = enc.optimize_frame();
        assert!(result.has_active_regions);
        assert!(result.analysis.roi_ctus > 0);
        // Boosted region should have quality weight > 1.0
        let w = result.quality_weight_at(0, 0);
        assert!(w > 1.0, "Quality weight in ROI should be > 1.0: {}", w);
    }

    #[test]
    fn test_roi_optimize_result_empty() {
        let result = RoiOptimizeResult::empty(4, 4);
        assert!(!result.has_active_regions);
        assert_eq!(result.quality_weights.len(), 16);
        assert!((result.quality_weight_at(0, 0) - 1.0).abs() < f64::EPSILON);
        assert_eq!(result.qp_delta_at(0, 0), 0);
    }

    #[test]
    fn test_roi_optimize_result_accessors() {
        let config = RoiEncoderConfig {
            frame_width: 128,
            frame_height: 128,
            ctu_size: 64,
            adjust_mode: QpAdjustMode::AbsoluteOffset,
            max_qp_reduction: 6,
            ..Default::default()
        };
        let mut enc = RoiEncoder::new(config);
        enc.add_region(RoiRegion::with_priority(0, 0, 64, 64, 1.0));
        let result = enc.optimize_frame();
        // CTU (0,0) should have negative delta
        assert!(result.qp_delta_at(0, 0) < 0);
        // Out of bounds should return defaults
        assert_eq!(result.qp_delta_at(100, 100), 0);
        assert!((result.quality_weight_at(100, 100) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_roi_high_priority_gets_more_bits() {
        let config = RoiEncoderConfig {
            frame_width: 256,
            frame_height: 256,
            ctu_size: 64,
            adjust_mode: QpAdjustMode::AbsoluteOffset,
            max_qp_reduction: 10,
            ..Default::default()
        };
        let mut enc = RoiEncoder::new(config);
        // High priority face region
        enc.add_region(RoiRegion::with_priority(0, 0, 64, 64, 2.0));
        let result = enc.optimize_frame();
        let delta_roi = result.qp_delta_at(0, 0);
        let delta_bg = result.qp_delta_at(3, 3);
        // ROI should have more negative delta (lower QP = more bits)
        assert!(
            delta_roi < delta_bg,
            "ROI delta ({}) should be less than background ({})",
            delta_roi,
            delta_bg
        );
    }

    #[test]
    fn test_roi_encoder_regions_accessor() {
        let config = RoiEncoderConfig::default();
        let mut enc = RoiEncoder::new(config);
        enc.add_region(RoiRegion::new(0, 0, 100, 100));
        enc.add_region(RoiRegion::with_priority(200, 200, 50, 50, 2.0));
        assert_eq!(enc.regions().len(), 2);
        assert_eq!(enc.regions()[0].width, 100);
        assert!((enc.regions()[1].priority - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_merge_additive_different_sizes_noop() {
        let mut map1 = QpDeltaMap::new(2, 2);
        map1.set(0, 0, -3);
        let map2 = QpDeltaMap::new(3, 3);
        map1.merge_additive(&map2, 10);
        // Should not change since sizes differ
        assert_eq!(map1.get(0, 0), -3);
    }
}
