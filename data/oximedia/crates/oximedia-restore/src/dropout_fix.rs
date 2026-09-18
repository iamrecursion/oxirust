//! Detection and repair of video dropout artefacts.

#![allow(dead_code)]

/// The spatial extent of a video dropout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropoutType {
    /// A single horizontal line is lost or corrupted.
    SingleLine,
    /// Several consecutive horizontal lines are affected.
    MultiLine,
    /// A rectangular block region is corrupted.
    Block,
}

impl DropoutType {
    /// Return a severity score for the dropout type: higher is worse.
    ///
    /// - `SingleLine` → 1
    /// - `MultiLine`  → 2
    /// - `Block`      → 3
    #[must_use]
    pub fn severity(&self) -> u8 {
        match self {
            Self::SingleLine => 1,
            Self::MultiLine => 2,
            Self::Block => 3,
        }
    }

    /// Return a short description.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::SingleLine => "single-line dropout",
            Self::MultiLine => "multi-line dropout",
            Self::Block => "block dropout",
        }
    }
}

/// A single dropout region within a video frame.
#[derive(Debug, Clone)]
pub struct DropoutRegion {
    /// Row at which the dropout starts.
    pub row: u32,
    /// Number of rows affected.
    pub row_count: u32,
    /// Column at which the dropout starts (0 for full-width line dropouts).
    pub col: u32,
    /// Number of columns affected.
    pub col_count: u32,
    /// Classification of this dropout.
    pub dropout_type: DropoutType,
    /// Confidence score in the range `[0.0, 1.0]`.
    pub confidence: f32,
}

impl DropoutRegion {
    /// Create a new dropout region.
    #[must_use]
    pub fn new(
        row: u32,
        row_count: u32,
        col: u32,
        col_count: u32,
        dropout_type: DropoutType,
    ) -> Self {
        Self {
            row,
            row_count,
            col,
            col_count,
            dropout_type,
            confidence: 1.0,
        }
    }

    /// Return the pixel area of this dropout region.
    #[must_use]
    pub fn area(&self) -> u64 {
        u64::from(self.row_count) * u64::from(self.col_count)
    }
}

/// A video frame with associated dropout information.
#[derive(Debug, Clone)]
pub struct DropoutFrame {
    /// Frame index in the video stream.
    pub frame_index: u64,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Detected dropout regions in this frame.
    pub dropouts: Vec<DropoutRegion>,
}

impl DropoutFrame {
    /// Create a new dropout frame descriptor.
    #[must_use]
    pub fn new(frame_index: u64, width: u32, height: u32) -> Self {
        Self {
            frame_index,
            width,
            height,
            dropouts: Vec::new(),
        }
    }

    /// Return `true` if any dropout regions have been recorded.
    #[must_use]
    pub fn has_dropouts(&self) -> bool {
        !self.dropouts.is_empty()
    }

    /// Return the number of dropout regions.
    #[must_use]
    pub fn dropout_count(&self) -> usize {
        self.dropouts.len()
    }

    /// Return the total pixel area affected by dropouts.
    #[must_use]
    pub fn total_dropout_area(&self) -> u64 {
        self.dropouts.iter().map(|d| d.area()).sum()
    }

    /// Return the fraction of the frame affected by dropouts (`[0.0, 1.0]`).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn dropout_fraction(&self) -> f32 {
        let frame_area = u64::from(self.width) * u64::from(self.height);
        if frame_area == 0 {
            return 0.0;
        }
        (self.total_dropout_area() as f32 / frame_area as f32).clamp(0.0, 1.0)
    }

    /// Return the maximum severity across all dropout regions, or `0` if there are none.
    #[must_use]
    pub fn max_severity(&self) -> u8 {
        self.dropouts
            .iter()
            .map(|d| d.dropout_type.severity())
            .max()
            .unwrap_or(0)
    }
}

/// Strategy used when repairing a dropout region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairStrategy {
    /// Copy the pixel row/block from the previous frame.
    PreviousFrame,
    /// Interpolate between the frame above and below the dropout.
    TemporalInterpolation,
    /// Use the neighbouring pixels in the same frame (spatial median).
    SpatialMedian,
    /// Replace with a solid colour (last resort).
    SolidFill,
}

/// Repairs dropout artefacts in video frames.
pub struct DropoutFixer {
    strategy: RepairStrategy,
    /// Maximum dropout area (in pixels) this fixer will attempt to repair.
    pub max_repair_area: u64,
}

impl DropoutFixer {
    /// Create a new fixer with the given strategy.
    #[must_use]
    pub fn new(strategy: RepairStrategy) -> Self {
        Self {
            strategy,
            max_repair_area: 10_000,
        }
    }

    /// Return the active repair strategy.
    #[must_use]
    pub fn strategy(&self) -> RepairStrategy {
        self.strategy
    }

    /// Repair all dropouts in `frame`, operating on a flat row-major `pixels` buffer.
    ///
    /// `pixels` must have `frame.width * frame.height` elements.
    /// Returns the repaired pixel buffer or an error string if the buffer size is wrong.
    pub fn repair(&self, frame: &DropoutFrame, pixels: &[f32]) -> Result<Vec<f32>, String> {
        let expected = (frame.width as usize) * (frame.height as usize);
        if pixels.len() != expected {
            return Err(format!(
                "pixel buffer length {} != expected {expected}",
                pixels.len()
            ));
        }
        let mut out = pixels.to_vec();
        for dropout in &frame.dropouts {
            if dropout.area() > self.max_repair_area {
                continue; // too large to repair safely
            }
            self.apply_repair(&mut out, frame.width, frame.height, dropout);
        }
        Ok(out)
    }

    fn apply_repair(
        &self,
        pixels: &mut Vec<f32>,
        width: u32,
        _height: u32,
        dropout: &DropoutRegion,
    ) {
        match self.strategy {
            RepairStrategy::SolidFill => {
                for r in dropout.row..dropout.row + dropout.row_count {
                    for c in dropout.col..dropout.col + dropout.col_count {
                        let idx = (r as usize) * (width as usize) + (c as usize);
                        if idx < pixels.len() {
                            pixels[idx] = 0.0;
                        }
                    }
                }
            }
            RepairStrategy::SpatialMedian => {
                // Replace each dropout pixel with the mean of the row above and below
                for r in dropout.row..dropout.row + dropout.row_count {
                    for c in dropout.col..dropout.col + dropout.col_count {
                        let idx = (r as usize) * (width as usize) + (c as usize);
                        if idx < pixels.len() {
                            let above = if r > 0 {
                                pixels[idx - width as usize]
                            } else {
                                0.0
                            };
                            let below_idx = idx + width as usize;
                            let below = if below_idx < pixels.len() {
                                pixels[below_idx]
                            } else {
                                0.0
                            };
                            pixels[idx] = (above + below) * 0.5;
                        }
                    }
                }
            }
            // For PreviousFrame and TemporalInterpolation we can't do anything without the
            // reference frame, so fall back to solid-fill for the in-place operation.
            _ => {
                for r in dropout.row..dropout.row + dropout.row_count {
                    for c in dropout.col..dropout.col + dropout.col_count {
                        let idx = (r as usize) * (width as usize) + (c as usize);
                        if idx < pixels.len() {
                            pixels[idx] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

/// Accumulated statistics about dropouts across a set of frames.
#[derive(Debug, Clone, Default)]
pub struct DropoutStats {
    /// Total number of frames analysed.
    pub total_frames: u64,
    /// Number of frames that contained at least one dropout.
    pub frames_with_dropouts: u64,
    /// Total dropout regions detected.
    pub total_dropout_regions: u64,
    /// Total pixel area affected by dropouts.
    pub total_dropout_pixels: u64,
    /// Total pixel area across all frames.
    pub total_pixels: u64,
}

impl DropoutStats {
    /// Create zeroed statistics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Incorporate the results from a single `DropoutFrame` into the statistics.
    pub fn record_frame(&mut self, frame: &DropoutFrame) {
        self.total_frames += 1;
        self.total_pixels += u64::from(frame.width) * u64::from(frame.height);
        if frame.has_dropouts() {
            self.frames_with_dropouts += 1;
            self.total_dropout_regions += frame.dropout_count() as u64;
            self.total_dropout_pixels += frame.total_dropout_area();
        }
    }

    /// Fraction of frames that contained dropouts (`[0.0, 1.0]`).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn frame_dropout_rate(&self) -> f32 {
        if self.total_frames == 0 {
            return 0.0;
        }
        self.frames_with_dropouts as f32 / self.total_frames as f32
    }

    /// Fraction of total pixels affected by dropouts (`[0.0, 1.0]`).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn pct_affected(&self) -> f32 {
        if self.total_pixels == 0 {
            return 0.0;
        }
        (self.total_dropout_pixels as f32 / self.total_pixels as f32).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- DropoutType ---

    #[test]
    fn test_dropout_type_severity_order() {
        assert!(DropoutType::SingleLine.severity() < DropoutType::MultiLine.severity());
        assert!(DropoutType::MultiLine.severity() < DropoutType::Block.severity());
    }

    #[test]
    fn test_dropout_type_description() {
        assert_eq!(DropoutType::Block.description(), "block dropout");
        assert_eq!(DropoutType::SingleLine.description(), "single-line dropout");
    }

    // --- DropoutRegion ---

    #[test]
    fn test_dropout_region_area() {
        let r = DropoutRegion::new(10, 3, 20, 100, DropoutType::MultiLine);
        assert_eq!(r.area(), 300);
    }

    // --- DropoutFrame ---

    #[test]
    fn test_dropout_frame_no_dropouts() {
        let f = DropoutFrame::new(0, 1920, 1080);
        assert!(!f.has_dropouts());
        assert_eq!(f.dropout_count(), 0);
    }

    #[test]
    fn test_dropout_frame_has_dropouts() {
        let mut f = DropoutFrame::new(1, 1920, 1080);
        f.dropouts
            .push(DropoutRegion::new(100, 1, 0, 1920, DropoutType::SingleLine));
        assert!(f.has_dropouts());
    }

    #[test]
    fn test_dropout_frame_total_area() {
        let mut f = DropoutFrame::new(0, 100, 100);
        f.dropouts
            .push(DropoutRegion::new(0, 5, 0, 10, DropoutType::Block));
        assert_eq!(f.total_dropout_area(), 50);
    }

    #[test]
    fn test_dropout_frame_fraction() {
        let mut f = DropoutFrame::new(0, 100, 100); // 10000 px
        f.dropouts
            .push(DropoutRegion::new(0, 10, 0, 100, DropoutType::MultiLine)); // 1000 px
        let frac = f.dropout_fraction();
        assert!((frac - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_dropout_frame_max_severity() {
        let mut f = DropoutFrame::new(0, 640, 480);
        f.dropouts
            .push(DropoutRegion::new(0, 1, 0, 640, DropoutType::SingleLine));
        f.dropouts
            .push(DropoutRegion::new(10, 5, 0, 640, DropoutType::MultiLine));
        assert_eq!(f.max_severity(), 2);
    }

    #[test]
    fn test_dropout_frame_zero_area_fraction() {
        let f = DropoutFrame::new(0, 0, 0);
        assert_eq!(f.dropout_fraction(), 0.0);
    }

    // --- DropoutFixer ---

    #[test]
    fn test_fixer_repair_no_dropouts() {
        let frame = DropoutFrame::new(0, 4, 4);
        let pixels = vec![0.5_f32; 16];
        let fixer = DropoutFixer::new(RepairStrategy::SolidFill);
        let out = fixer
            .repair(&frame, &pixels)
            .expect("should succeed in test");
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_fixer_repair_wrong_size_returns_err() {
        let frame = DropoutFrame::new(0, 4, 4);
        let pixels = vec![0.5_f32; 10]; // wrong size
        let fixer = DropoutFixer::new(RepairStrategy::SolidFill);
        assert!(fixer.repair(&frame, &pixels).is_err());
    }

    #[test]
    fn test_fixer_repair_solid_fill_zeroes_region() {
        let mut frame = DropoutFrame::new(0, 4, 4);
        frame
            .dropouts
            .push(DropoutRegion::new(1, 1, 0, 4, DropoutType::SingleLine));
        let pixels = vec![0.8_f32; 16];
        let fixer = DropoutFixer::new(RepairStrategy::SolidFill);
        let out = fixer
            .repair(&frame, &pixels)
            .expect("should succeed in test");
        // Row 1 should be zeroed
        for col in 0..4 {
            assert_eq!(out[4 + col], 0.0);
        }
    }

    #[test]
    fn test_fixer_strategy_accessor() {
        let fixer = DropoutFixer::new(RepairStrategy::TemporalInterpolation);
        assert_eq!(fixer.strategy(), RepairStrategy::TemporalInterpolation);
    }

    // --- DropoutStats ---

    #[test]
    fn test_stats_empty() {
        let s = DropoutStats::new();
        assert_eq!(s.frame_dropout_rate(), 0.0);
        assert_eq!(s.pct_affected(), 0.0);
    }

    #[test]
    fn test_stats_record_clean_frame() {
        let mut s = DropoutStats::new();
        let f = DropoutFrame::new(0, 100, 100);
        s.record_frame(&f);
        assert_eq!(s.total_frames, 1);
        assert_eq!(s.frames_with_dropouts, 0);
    }

    #[test]
    fn test_stats_pct_affected() {
        let mut s = DropoutStats::new();
        let mut f = DropoutFrame::new(0, 100, 100); // 10000 px
        f.dropouts
            .push(DropoutRegion::new(0, 10, 0, 100, DropoutType::MultiLine)); // 1000 px
        s.record_frame(&f);
        assert!((s.pct_affected() - 0.1).abs() < 1e-4);
    }
}
