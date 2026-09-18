//! Error correction for corrupted media frames.
//!
//! This module provides strategies for detecting and repairing various types
//! of video frame corruption, including bit errors, dropouts, and block corruption.

/// Types of corruption that can affect a frame region.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CorruptionType {
    /// Individual bit errors within a byte.
    BitError,
    /// Repeated bytes causing a stuttering artifact.
    ByteStutter,
    /// Entire DCT block is corrupted.
    BlockCorruption,
    /// Signal dropout causing missing data.
    Dropout,
    /// Frame is frozen (identical to previous frame).
    Freeze,
    /// Only part of a frame was received/decoded.
    PartialFrame,
}

/// A region within a frame that is corrupted.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct CorruptionRegion {
    /// Frame index (0-based) containing this region.
    pub frame_idx: u64,
    /// X coordinate of the top-left corner of the corrupted region.
    pub x: u32,
    /// Y coordinate of the top-left corner of the corrupted region.
    pub y: u32,
    /// Width of the corrupted region in pixels.
    pub width: u32,
    /// Height of the corrupted region in pixels.
    pub height: u32,
    /// Type of corruption detected.
    pub corruption_type: CorruptionType,
    /// Severity score (0.0 = minor, 1.0 = completely corrupted).
    pub severity: f32,
}

impl CorruptionRegion {
    /// Create a new corruption region.
    #[must_use]
    pub const fn new(
        frame_idx: u64,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        corruption_type: CorruptionType,
        severity: f32,
    ) -> Self {
        Self {
            frame_idx,
            x,
            y,
            width,
            height,
            corruption_type,
            severity,
        }
    }

    /// Total number of pixels in this region.
    #[must_use]
    pub const fn pixel_count(&self) -> u32 {
        self.width * self.height
    }
}

/// Strategy to use when repairing a corrupted region.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RepairStrategy {
    /// Blend between previous and next frames.
    Interpolate,
    /// Copy pixels from the nearest valid neighbor frame.
    CopyFromNeighbor,
    /// Fill the region using a median filter of surrounding pixels.
    InpaintMedian,
    /// Fill the region using a bilateral filter of surrounding pixels.
    InpaintBilateral,
    /// Drop the corrupted frame entirely.
    SkipFrame,
}

/// Selects the best repair strategy for a given corruption region.
pub struct ErrorCorrector;

impl ErrorCorrector {
    /// Select the most appropriate repair strategy for a corrupted region.
    ///
    /// Strategy selection logic:
    /// - Freeze / PartialFrame → Interpolate (temporal information available)
    /// - Dropout with low severity → CopyFromNeighbor
    /// - BlockCorruption / BitError with low severity → InpaintMedian
    /// - ByteStutter → InpaintBilateral (bilateral is better for texture)
    /// - High severity (> 0.8) → SkipFrame
    #[must_use]
    pub fn select_strategy(region: &CorruptionRegion) -> RepairStrategy {
        if region.severity > 0.8 {
            return RepairStrategy::SkipFrame;
        }
        match region.corruption_type {
            CorruptionType::Freeze | CorruptionType::PartialFrame => RepairStrategy::Interpolate,
            CorruptionType::Dropout => RepairStrategy::CopyFromNeighbor,
            CorruptionType::BlockCorruption | CorruptionType::BitError => {
                RepairStrategy::InpaintMedian
            }
            CorruptionType::ByteStutter => RepairStrategy::InpaintBilateral,
        }
    }
}

/// Interpolates a corrupted region by blending previous and next frames.
pub struct FrameInterpolator;

impl FrameInterpolator {
    /// Interpolate a corrupted region using a 50% blend of the previous and next frames.
    ///
    /// `prev_frame` and `next_frame` must be flat row-major arrays of `f32` pixel values.
    /// `width` is the width of the full frame in pixels.
    ///
    /// Returns the blended pixel values for the region (row-major, same dimensions as region).
    #[must_use]
    pub fn interpolate_region(
        prev_frame: &[f32],
        next_frame: &[f32],
        region: &CorruptionRegion,
        width: u32,
    ) -> Vec<f32> {
        let mut result = Vec::with_capacity((region.width * region.height) as usize);
        for row in 0..region.height {
            for col in 0..region.width {
                let frame_x = region.x + col;
                let frame_y = region.y + row;
                let idx = (frame_y * width + frame_x) as usize;
                let prev_val = prev_frame.get(idx).copied().unwrap_or(0.0);
                let next_val = next_frame.get(idx).copied().unwrap_or(0.0);
                result.push((prev_val + next_val) * 0.5);
            }
        }
        result
    }
}

/// Inpaints a corrupted region using the median of border pixels.
pub struct MedianInpainter;

impl MedianInpainter {
    /// Inpaint a corrupted region by filling it with the median value from a 5-pixel
    /// border around the region.
    ///
    /// `frame` is modified in-place (row-major, `width` × `height` pixels).
    pub fn inpaint(frame: &mut Vec<f32>, region: &CorruptionRegion, width: u32, height: u32) {
        let border_radius = 5u32;
        let mut border_vals: Vec<f32> = Vec::new();

        // Collect pixels from the border ring around the region
        let x_start = region.x.saturating_sub(border_radius);
        let y_start = region.y.saturating_sub(border_radius);
        let x_end = (region.x + region.width + border_radius).min(width);
        let y_end = (region.y + region.height + border_radius).min(height);

        for row in y_start..y_end {
            for col in x_start..x_end {
                // Only take pixels outside the corrupted region
                let in_region = col >= region.x
                    && col < region.x + region.width
                    && row >= region.y
                    && row < region.y + region.height;
                if !in_region {
                    let idx = (row * width + col) as usize;
                    if let Some(&val) = frame.get(idx) {
                        border_vals.push(val);
                    }
                }
            }
        }

        // Compute median
        let fill_val = if border_vals.is_empty() {
            0.0
        } else {
            border_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = border_vals.len() / 2;
            if border_vals.len() % 2 == 0 {
                (border_vals[mid - 1] + border_vals[mid]) * 0.5
            } else {
                border_vals[mid]
            }
        };

        // Fill the region
        for row in region.y..(region.y + region.height).min(height) {
            for col in region.x..(region.x + region.width).min(width) {
                let idx = (row * width + col) as usize;
                if idx < frame.len() {
                    frame[idx] = fill_val;
                }
            }
        }
    }
}

/// A record of a single repair operation.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct RepairRecord {
    /// Frame index that was repaired.
    pub frame_idx: u64,
    /// Strategy used for repair.
    pub strategy: RepairStrategy,
    /// Severity before repair.
    pub severity_before: f32,
    /// Estimated severity after repair.
    pub severity_after: f32,
}

impl RepairRecord {
    /// Create a new repair record.
    #[must_use]
    pub const fn new(
        frame_idx: u64,
        strategy: RepairStrategy,
        severity_before: f32,
        severity_after: f32,
    ) -> Self {
        Self {
            frame_idx,
            strategy,
            severity_before,
            severity_after,
        }
    }
}

/// Log of all repair operations performed.
#[derive(Debug, Clone, Default)]
pub struct RepairLog {
    /// All recorded repair operations.
    pub repairs: Vec<RepairRecord>,
}

impl RepairLog {
    /// Create a new empty repair log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a repair record to the log.
    pub fn add_repair(&mut self, frame: u64, strategy: RepairStrategy, region: CorruptionRegion) {
        let severity_after = (region.severity * 0.1).clamp(0.0, 1.0);
        self.repairs.push(RepairRecord::new(
            frame,
            strategy,
            region.severity,
            severity_after,
        ));
    }

    /// Generate a human-readable summary of all repairs.
    #[must_use]
    pub fn summary(&self) -> String {
        if self.repairs.is_empty() {
            return "No repairs performed.".to_string();
        }
        let total = self.repairs.len();
        let avg_severity_before: f32 =
            self.repairs.iter().map(|r| r.severity_before).sum::<f32>() / total as f32;
        let avg_severity_after: f32 =
            self.repairs.iter().map(|r| r.severity_after).sum::<f32>() / total as f32;
        format!(
            "Repairs: {total} | Avg severity before: {avg_severity_before:.3} | Avg severity after: {avg_severity_after:.3}"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_region(ct: CorruptionType, severity: f32) -> CorruptionRegion {
        CorruptionRegion::new(0, 10, 10, 20, 20, ct, severity)
    }

    #[test]
    fn test_corruption_region_pixel_count() {
        let r = make_region(CorruptionType::BitError, 0.3);
        assert_eq!(r.pixel_count(), 400);
    }

    #[test]
    fn test_strategy_skip_high_severity() {
        let r = make_region(CorruptionType::BitError, 0.9);
        assert_eq!(
            ErrorCorrector::select_strategy(&r),
            RepairStrategy::SkipFrame
        );
    }

    #[test]
    fn test_strategy_freeze_interpolate() {
        let r = make_region(CorruptionType::Freeze, 0.5);
        assert_eq!(
            ErrorCorrector::select_strategy(&r),
            RepairStrategy::Interpolate
        );
    }

    #[test]
    fn test_strategy_partial_frame_interpolate() {
        let r = make_region(CorruptionType::PartialFrame, 0.4);
        assert_eq!(
            ErrorCorrector::select_strategy(&r),
            RepairStrategy::Interpolate
        );
    }

    #[test]
    fn test_strategy_dropout_copy_neighbor() {
        let r = make_region(CorruptionType::Dropout, 0.3);
        assert_eq!(
            ErrorCorrector::select_strategy(&r),
            RepairStrategy::CopyFromNeighbor
        );
    }

    #[test]
    fn test_strategy_block_corruption_inpaint_median() {
        let r = make_region(CorruptionType::BlockCorruption, 0.4);
        assert_eq!(
            ErrorCorrector::select_strategy(&r),
            RepairStrategy::InpaintMedian
        );
    }

    #[test]
    fn test_strategy_byte_stutter_inpaint_bilateral() {
        let r = make_region(CorruptionType::ByteStutter, 0.3);
        assert_eq!(
            ErrorCorrector::select_strategy(&r),
            RepairStrategy::InpaintBilateral
        );
    }

    #[test]
    fn test_frame_interpolator_basic() {
        let prev: Vec<f32> = vec![0.0; 100];
        let next: Vec<f32> = vec![100.0; 100];
        let region = CorruptionRegion::new(0, 0, 0, 2, 2, CorruptionType::Freeze, 0.5);
        let result = FrameInterpolator::interpolate_region(&prev, &next, &region, 10);
        assert_eq!(result.len(), 4);
        for val in &result {
            assert!((val - 50.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_frame_interpolator_out_of_bounds() {
        let prev: Vec<f32> = vec![1.0; 4];
        let next: Vec<f32> = vec![3.0; 4];
        // Region larger than frame - should still return values (with fallback 0.0)
        let region = CorruptionRegion::new(0, 0, 0, 10, 10, CorruptionType::Dropout, 0.2);
        let result = FrameInterpolator::interpolate_region(&prev, &next, &region, 4);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_median_inpainter_basic() {
        let width = 10u32;
        let height = 10u32;
        let mut frame: Vec<f32> = (0..100).map(|_| 50.0).collect();
        // Corrupt a 2x2 region
        let region = CorruptionRegion::new(0, 4, 4, 2, 2, CorruptionType::BitError, 0.5);
        MedianInpainter::inpaint(&mut frame, &region, width, height);
        // Region should now be filled with median of surrounding border pixels (50.0)
        let idx = (4 * width + 4) as usize;
        assert!((frame[idx] - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_repair_log_add_and_summary() {
        let mut log = RepairLog::new();
        let region = make_region(CorruptionType::Freeze, 0.6);
        log.add_repair(0, RepairStrategy::Interpolate, region);
        let summary = log.summary();
        assert!(summary.contains("Repairs: 1"));
    }

    #[test]
    fn test_repair_log_empty_summary() {
        let log = RepairLog::new();
        assert_eq!(log.summary(), "No repairs performed.");
    }

    #[test]
    fn test_repair_log_multiple_repairs() {
        let mut log = RepairLog::new();
        for i in 0..5 {
            let region = CorruptionRegion::new(i, 0, 0, 4, 4, CorruptionType::BitError, 0.5);
            log.add_repair(i, RepairStrategy::InpaintMedian, region);
        }
        assert_eq!(log.repairs.len(), 5);
        let summary = log.summary();
        assert!(summary.contains("Repairs: 5"));
    }

    #[test]
    fn test_repair_record_new() {
        let rec = RepairRecord::new(7, RepairStrategy::SkipFrame, 0.9, 0.0);
        assert_eq!(rec.frame_idx, 7);
        assert_eq!(rec.strategy, RepairStrategy::SkipFrame);
        assert!((rec.severity_before - 0.9).abs() < f32::EPSILON);
    }
}
