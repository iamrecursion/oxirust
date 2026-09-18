#![allow(dead_code)]
//! Post-stabilization quality reporting.
//!
//! Produces a structured report of how well stabilization performed,
//! including per-frame shake detection and aggregate reduction metrics.

/// Quality classification of the stabilization result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilizationQuality {
    /// Excellent — shake almost fully removed
    Excellent,
    /// Good — noticeable improvement with minor residual shake
    Good,
    /// Fair — moderate improvement; some shake remains
    Fair,
    /// Poor — minimal improvement; significant shake remains
    Poor,
    /// Unprocessed — no stabilization was applied
    Unprocessed,
}

impl StabilizationQuality {
    /// Numeric score: 1.0 = excellent, 0.0 = poor/unprocessed.
    pub fn score(self) -> f64 {
        match self {
            Self::Excellent => 1.0,
            Self::Good => 0.75,
            Self::Fair => 0.50,
            Self::Poor => 0.25,
            Self::Unprocessed => 0.0,
        }
    }

    /// Classify a reduction percentage (0–100) into a quality level.
    pub fn from_reduction(pct: f64) -> Self {
        if pct >= 90.0 {
            Self::Excellent
        } else if pct >= 70.0 {
            Self::Good
        } else if pct >= 40.0 {
            Self::Fair
        } else if pct > 0.0 {
            Self::Poor
        } else {
            Self::Unprocessed
        }
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Excellent => "Excellent",
            Self::Good => "Good",
            Self::Fair => "Fair",
            Self::Poor => "Poor",
            Self::Unprocessed => "Unprocessed",
        }
    }

    /// Returns `true` if the quality is at least `Good`.
    pub fn is_acceptable(self) -> bool {
        matches!(self, Self::Excellent | Self::Good)
    }
}

/// Per-frame stabilization statistics.
#[derive(Debug, Clone)]
pub struct FrameStat {
    /// Frame index
    pub index: usize,
    /// Motion magnitude before stabilization (pixels)
    pub motion_before: f64,
    /// Motion magnitude after stabilization (pixels)
    pub motion_after: f64,
    /// Whether this frame was considered shaky before stabilization
    pub was_shaky: bool,
    /// Crop percentage applied to this frame (0–100)
    pub crop_pct: f64,
    /// Effective crop area in pixels (width × height of the stable output region).
    /// `None` when frame dimensions are not available.
    pub crop_area_px: Option<u64>,
    /// Per-frame quality classification.
    pub quality: StabilizationQuality,
}

impl FrameStat {
    /// Create a new frame stat record with minimal fields.
    pub fn new(
        index: usize,
        motion_before: f64,
        motion_after: f64,
        was_shaky: bool,
        crop_pct: f64,
    ) -> Self {
        let reduction = if motion_before < 1e-9 {
            100.0
        } else {
            ((motion_before - motion_after) / motion_before * 100.0).clamp(0.0, 100.0)
        };
        Self {
            index,
            motion_before,
            motion_after,
            was_shaky,
            crop_pct,
            crop_area_px: None,
            quality: StabilizationQuality::from_reduction(reduction),
        }
    }

    /// Create a frame stat with all fields including crop area and quality.
    #[must_use]
    pub fn with_details(
        index: usize,
        motion_before: f64,
        motion_after: f64,
        was_shaky: bool,
        crop_pct: f64,
        frame_width: u32,
        frame_height: u32,
    ) -> Self {
        // Effective crop area shrinks proportionally with crop percentage.
        // crop_pct is the fraction of width/height removed on each side (0–50).
        let crop_fraction = (crop_pct / 100.0).clamp(0.0, 0.5);
        let eff_w = (frame_width as f64 * (1.0 - 2.0 * crop_fraction))
            .max(1.0)
            .round() as u64;
        let eff_h = (frame_height as f64 * (1.0 - 2.0 * crop_fraction))
            .max(1.0)
            .round() as u64;
        let crop_area = eff_w * eff_h;

        let reduction = if motion_before < 1e-9 {
            100.0
        } else {
            ((motion_before - motion_after) / motion_before * 100.0).clamp(0.0, 100.0)
        };

        Self {
            index,
            motion_before,
            motion_after,
            was_shaky,
            crop_pct,
            crop_area_px: Some(crop_area),
            quality: StabilizationQuality::from_reduction(reduction),
        }
    }

    /// Per-frame reduction percentage.
    pub fn reduction_pct(&self) -> f64 {
        if self.motion_before < 1e-9 {
            return 100.0;
        }
        let delta = self.motion_before - self.motion_after;
        (delta / self.motion_before * 100.0).clamp(0.0, 100.0)
    }
}

/// Complete stabilization report for a processed video.
#[derive(Debug, Clone)]
pub struct StabilizeReport {
    /// Per-frame statistics
    pub frame_stats: Vec<FrameStat>,
    /// Shake threshold used during analysis (pixels)
    pub shake_threshold: f64,
    /// Total processing time in seconds
    pub processing_time_s: f64,
    /// Configuration notes (e.g., smoothing window used)
    pub notes: Vec<String>,
}

impl StabilizeReport {
    /// Create a new report.
    pub fn new(frame_stats: Vec<FrameStat>, shake_threshold: f64) -> Self {
        Self {
            frame_stats,
            shake_threshold,
            processing_time_s: 0.0,
            notes: Vec::new(),
        }
    }

    /// Attach processing time to the report.
    pub fn with_processing_time(mut self, secs: f64) -> Self {
        self.processing_time_s = secs;
        self
    }

    /// Add a note to the report.
    pub fn add_note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }

    /// Number of frames in the report.
    pub fn frame_count(&self) -> usize {
        self.frame_stats.len()
    }

    /// Count of frames that were classified as shaky before stabilization.
    pub fn shaky_frames(&self) -> usize {
        self.frame_stats.iter().filter(|f| f.was_shaky).count()
    }

    /// Fraction of shaky frames before stabilization (0.0–1.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn shaky_fraction(&self) -> f64 {
        let n = self.frame_stats.len();
        if n == 0 {
            return 0.0;
        }
        self.shaky_frames() as f64 / n as f64
    }

    /// Average motion magnitude before stabilization.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_motion_before(&self) -> f64 {
        let n = self.frame_stats.len();
        if n == 0 {
            return 0.0;
        }
        self.frame_stats
            .iter()
            .map(|f| f.motion_before)
            .sum::<f64>()
            / n as f64
    }

    /// Average motion magnitude after stabilization.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_motion_after(&self) -> f64 {
        let n = self.frame_stats.len();
        if n == 0 {
            return 0.0;
        }
        self.frame_stats.iter().map(|f| f.motion_after).sum::<f64>() / n as f64
    }

    /// Overall motion reduction percentage (0–100).
    pub fn reduction_pct(&self) -> f64 {
        let before = self.avg_motion_before();
        if before < 1e-9 {
            return 100.0;
        }
        let after = self.avg_motion_after();
        ((before - after) / before * 100.0).clamp(0.0, 100.0)
    }

    /// Average crop percentage applied across all frames.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_crop_pct(&self) -> f64 {
        let n = self.frame_stats.len();
        if n == 0 {
            return 0.0;
        }
        self.frame_stats.iter().map(|f| f.crop_pct).sum::<f64>() / n as f64
    }

    /// Maximum crop percentage across all frames.
    pub fn max_crop_pct(&self) -> f64 {
        self.frame_stats
            .iter()
            .map(|f| f.crop_pct)
            .fold(0.0_f64, f64::max)
    }

    /// Overall quality classification based on reduction percentage.
    pub fn quality(&self) -> StabilizationQuality {
        StabilizationQuality::from_reduction(self.reduction_pct())
    }
}

/// Generates a `StabilizeReport` from raw before/after motion arrays.
#[derive(Debug, Default)]
pub struct StabilizeAnalyzer {
    /// Motion magnitude threshold for classifying a frame as shaky
    pub shake_threshold: f64,
}

impl StabilizeAnalyzer {
    /// Create an analyzer with a default shake threshold of 5 pixels.
    pub fn new() -> Self {
        Self {
            shake_threshold: 5.0,
        }
    }

    /// Create an analyzer with a custom shake threshold.
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            shake_threshold: threshold,
        }
    }

    /// Analyse motion vectors and produce a full `StabilizeReport`.
    ///
    /// `motion_before` and `motion_after` must have the same length.
    pub fn analyze(
        &self,
        motion_before: &[f64],
        motion_after: &[f64],
        crop_pcts: &[f64],
    ) -> StabilizeReport {
        let n = motion_before.len().min(motion_after.len());
        let crop_len = crop_pcts.len();

        let frame_stats: Vec<FrameStat> = (0..n)
            .map(|i| {
                let before = motion_before[i];
                let after = motion_after[i];
                let crop = if i < crop_len { crop_pcts[i] } else { 0.0 };
                FrameStat::new(i, before, after, before > self.shake_threshold, crop)
            })
            .collect();

        StabilizeReport::new(frame_stats, self.shake_threshold)
    }

    /// Analyse using equal crop percentage for every frame.
    pub fn analyze_uniform_crop(
        &self,
        motion_before: &[f64],
        motion_after: &[f64],
        crop_pct: f64,
    ) -> StabilizeReport {
        let crops = vec![crop_pct; motion_before.len()];
        self.analyze(motion_before, motion_after, &crops)
    }

    /// Analyse motion vectors and produce a full `StabilizeReport` with per-frame
    /// crop area in pixels and per-frame quality classification.
    ///
    /// `motion_before` and `motion_after` must have the same length.
    /// `frame_width` / `frame_height` are the dimensions of the original frames.
    pub fn analyze_with_dimensions(
        &self,
        motion_before: &[f64],
        motion_after: &[f64],
        crop_pcts: &[f64],
        frame_width: u32,
        frame_height: u32,
    ) -> StabilizeReport {
        let n = motion_before.len().min(motion_after.len());
        let crop_len = crop_pcts.len();

        let frame_stats: Vec<FrameStat> = (0..n)
            .map(|i| {
                let before = motion_before[i];
                let after = motion_after[i];
                let crop = if i < crop_len { crop_pcts[i] } else { 0.0 };
                FrameStat::with_details(
                    i,
                    before,
                    after,
                    before > self.shake_threshold,
                    crop,
                    frame_width,
                    frame_height,
                )
            })
            .collect();

        StabilizeReport::new(frame_stats, self.shake_threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_report(n: usize, before: f64, after: f64) -> StabilizeReport {
        let analyzer = StabilizeAnalyzer::new();
        let b = vec![before; n];
        let a = vec![after; n];
        analyzer.analyze_uniform_crop(&b, &a, 5.0)
    }

    #[test]
    fn test_quality_score() {
        assert!((StabilizationQuality::Excellent.score() - 1.0).abs() < 1e-10);
        assert!((StabilizationQuality::Good.score() - 0.75).abs() < 1e-10);
        assert!((StabilizationQuality::Unprocessed.score()).abs() < 1e-10);
    }

    #[test]
    fn test_quality_from_reduction_excellent() {
        assert_eq!(
            StabilizationQuality::from_reduction(95.0),
            StabilizationQuality::Excellent
        );
    }

    #[test]
    fn test_quality_from_reduction_good() {
        assert_eq!(
            StabilizationQuality::from_reduction(75.0),
            StabilizationQuality::Good
        );
    }

    #[test]
    fn test_quality_from_reduction_fair() {
        assert_eq!(
            StabilizationQuality::from_reduction(50.0),
            StabilizationQuality::Fair
        );
    }

    #[test]
    fn test_quality_from_reduction_poor() {
        assert_eq!(
            StabilizationQuality::from_reduction(10.0),
            StabilizationQuality::Poor
        );
    }

    #[test]
    fn test_quality_from_reduction_unprocessed() {
        assert_eq!(
            StabilizationQuality::from_reduction(0.0),
            StabilizationQuality::Unprocessed
        );
    }

    #[test]
    fn test_quality_is_acceptable() {
        assert!(StabilizationQuality::Excellent.is_acceptable());
        assert!(StabilizationQuality::Good.is_acceptable());
        assert!(!StabilizationQuality::Fair.is_acceptable());
        assert!(!StabilizationQuality::Poor.is_acceptable());
    }

    #[test]
    fn test_frame_stat_reduction_pct() {
        let f = FrameStat::new(0, 20.0, 4.0, true, 5.0);
        assert!((f.reduction_pct() - 80.0).abs() < 1e-9);
    }

    #[test]
    fn test_frame_stat_reduction_pct_zero_before() {
        let f = FrameStat::new(0, 0.0, 0.0, false, 0.0);
        assert!((f.reduction_pct() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_report_shaky_frames() {
        let report = make_report(10, 20.0, 2.0); // threshold=5, before=20 → all shaky
        assert_eq!(report.shaky_frames(), 10);
    }

    #[test]
    fn test_report_shaky_fraction() {
        let report = make_report(10, 20.0, 2.0);
        assert!((report.shaky_fraction() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_report_reduction_pct() {
        let report = make_report(5, 10.0, 1.0); // 90% reduction
        assert!((report.reduction_pct() - 90.0).abs() < 1e-9);
    }

    #[test]
    fn test_report_quality_excellent() {
        let report = make_report(5, 10.0, 0.5); // 95% reduction
        assert_eq!(report.quality(), StabilizationQuality::Excellent);
    }

    #[test]
    fn test_report_avg_crop() {
        let report = make_report(4, 10.0, 1.0); // uniform crop = 5.0
        assert!((report.avg_crop_pct() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_report_max_crop() {
        let analyzer = StabilizeAnalyzer::new();
        let b = vec![10.0, 10.0, 10.0];
        let a = vec![1.0, 1.0, 1.0];
        let crops = vec![2.0, 7.0, 4.0];
        let report = analyzer.analyze(&b, &a, &crops);
        assert!((report.max_crop_pct() - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_report_empty() {
        let report = make_report(0, 0.0, 0.0);
        assert_eq!(report.frame_count(), 0);
        assert_eq!(report.shaky_frames(), 0);
        assert!((report.shaky_fraction()).abs() < 1e-10);
    }

    #[test]
    fn test_analyzer_custom_threshold() {
        let analyzer = StabilizeAnalyzer::with_threshold(2.0);
        let b = vec![3.0, 1.0, 3.0]; // first and last exceed threshold
        let a = vec![0.5, 0.5, 0.5];
        let report = analyzer.analyze_uniform_crop(&b, &a, 0.0);
        assert_eq!(report.shaky_frames(), 2);
    }

    #[test]
    fn test_frame_stat_with_details_crop_area() {
        let stat = FrameStat::with_details(0, 10.0, 1.0, true, 5.0, 100, 80);
        // 5% crop on each side: effective w = 90, h = 72, area = 6480
        let area = stat.crop_area_px.expect("crop area should be set");
        assert!(area < 100 * 80, "crop area should be less than full frame");
        assert!(area > 0);
    }

    #[test]
    fn test_frame_stat_with_details_zero_crop() {
        let stat = FrameStat::with_details(0, 10.0, 1.0, true, 0.0, 100, 80);
        let area = stat.crop_area_px.expect("should have area");
        assert_eq!(area, 100 * 80, "zero crop → full frame area");
    }

    #[test]
    fn test_frame_stat_with_details_quality_excellent() {
        let stat = FrameStat::with_details(0, 100.0, 0.5, true, 0.0, 100, 80);
        // 99.5% reduction → Excellent
        assert_eq!(stat.quality, StabilizationQuality::Excellent);
    }

    #[test]
    fn test_frame_stat_quality_new_method() {
        let stat = FrameStat::new(0, 20.0, 4.0, true, 5.0);
        // 80% reduction → Good
        assert_eq!(stat.quality, StabilizationQuality::Good);
    }

    #[test]
    fn test_analyze_with_dimensions_produces_crop_area() {
        let analyzer = StabilizeAnalyzer::new();
        let b = vec![10.0; 5];
        let a = vec![1.0; 5];
        let crops = vec![10.0; 5]; // 10% crop per side
        let report = analyzer.analyze_with_dimensions(&b, &a, &crops, 1920, 1080);
        assert_eq!(report.frame_count(), 5);
        for stat in &report.frame_stats {
            let area = stat.crop_area_px.expect("should have area");
            // effective w = 1920 * 0.8 = 1536, h = 1080 * 0.8 = 864
            assert_eq!(area, 1536 * 864);
        }
    }

    #[test]
    fn test_analyze_with_dimensions_quality_classification() {
        let analyzer = StabilizeAnalyzer::new();
        let b = vec![50.0; 4];
        let a = vec![0.5; 4]; // 99% reduction
        let crops = vec![0.0; 4];
        let report = analyzer.analyze_with_dimensions(&b, &a, &crops, 1280, 720);
        for stat in &report.frame_stats {
            assert_eq!(stat.quality, StabilizationQuality::Excellent);
        }
    }
}
