#![allow(dead_code)]
//! Frame-level RPU analysis and statistics for Dolby Vision.
//!
//! Provides tools for analyzing the metadata characteristics of individual
//! frames and sequences of RPU data. Useful for QC, validation, debugging,
//! and adaptive processing decisions.

use std::collections::HashMap;

/// Frame-level luminance statistics extracted from RPU metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameLuminanceStats {
    /// Minimum luminance of the frame in nits.
    pub min_nits: f64,
    /// Maximum luminance of the frame in nits.
    pub max_nits: f64,
    /// Average luminance of the frame in nits.
    pub avg_nits: f64,
    /// MaxFALL (Maximum Frame Average Light Level) in nits.
    pub max_fall: f64,
}

impl FrameLuminanceStats {
    /// Create new luminance stats.
    #[must_use]
    pub fn new(min_nits: f64, max_nits: f64, avg_nits: f64) -> Self {
        Self {
            min_nits,
            max_nits,
            avg_nits,
            max_fall: avg_nits,
        }
    }

    /// The dynamic range of this frame in stops.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn dynamic_range_stops(&self) -> f64 {
        if self.min_nits <= 0.0 || self.max_nits <= 0.0 {
            return 0.0;
        }
        (self.max_nits / self.min_nits).log2()
    }

    /// Whether this frame is HDR (peak > 100 nits).
    #[must_use]
    pub fn is_hdr(&self) -> bool {
        self.max_nits > 100.0
    }

    /// The contrast ratio of this frame.
    #[must_use]
    pub fn contrast_ratio(&self) -> f64 {
        if self.min_nits <= 0.0 {
            return 0.0;
        }
        self.max_nits / self.min_nits
    }
}

/// Trim pass statistics for a single frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrimPassStats {
    /// Target display peak luminance in nits.
    pub target_peak_nits: f64,
    /// Slope adjustment factor.
    pub slope: f64,
    /// Offset adjustment factor.
    pub offset: f64,
    /// Power/gamma adjustment factor.
    pub power: f64,
    /// Chroma weight adjustment.
    pub chroma_weight: f64,
    /// Saturation gain adjustment.
    pub saturation_gain: f64,
}

impl TrimPassStats {
    /// Create default trim pass stats.
    #[must_use]
    pub fn new(target_peak_nits: f64) -> Self {
        Self {
            target_peak_nits,
            slope: 1.0,
            offset: 0.0,
            power: 1.0,
            chroma_weight: 1.0,
            saturation_gain: 1.0,
        }
    }

    /// Whether the trim pass has any active adjustments (non-identity).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        (self.slope - 1.0).abs() < 1e-6
            && self.offset.abs() < 1e-6
            && (self.power - 1.0).abs() < 1e-6
            && (self.chroma_weight - 1.0).abs() < 1e-6
            && (self.saturation_gain - 1.0).abs() < 1e-6
    }
}

/// Analysis data for a single RPU frame.
#[derive(Debug, Clone)]
pub struct FrameAnalysis {
    /// Frame number (0-indexed).
    pub frame_number: u64,
    /// Presentation timestamp in 90kHz ticks (if available).
    pub pts: Option<i64>,
    /// Luminance statistics.
    pub luminance: FrameLuminanceStats,
    /// Per-target-display trim pass stats.
    pub trim_passes: Vec<TrimPassStats>,
    /// Whether the frame has active area metadata (letterbox/pillarbox).
    pub has_active_area: bool,
    /// Active area: left offset in pixels.
    pub active_area_left: u16,
    /// Active area: right offset in pixels.
    pub active_area_right: u16,
    /// Active area: top offset in pixels.
    pub active_area_top: u16,
    /// Active area: bottom offset in pixels.
    pub active_area_bottom: u16,
    /// Whether this is a scene cut frame.
    pub is_scene_cut: bool,
}

impl FrameAnalysis {
    /// Create a new frame analysis for the given frame number.
    #[must_use]
    pub fn new(frame_number: u64, luminance: FrameLuminanceStats) -> Self {
        Self {
            frame_number,
            pts: None,
            luminance,
            trim_passes: Vec::new(),
            has_active_area: false,
            active_area_left: 0,
            active_area_right: 0,
            active_area_top: 0,
            active_area_bottom: 0,
            is_scene_cut: false,
        }
    }

    /// Add a trim pass to this frame analysis.
    pub fn add_trim_pass(&mut self, pass: TrimPassStats) {
        self.trim_passes.push(pass);
    }

    /// Number of trim passes for this frame.
    #[must_use]
    pub fn trim_pass_count(&self) -> usize {
        self.trim_passes.len()
    }

    /// Whether all trim passes are identity (no adjustments).
    #[must_use]
    pub fn all_trims_identity(&self) -> bool {
        self.trim_passes.iter().all(|t| t.is_identity())
    }
}

/// Aggregated statistics over a sequence of frames.
#[derive(Debug, Clone)]
pub struct SequenceStats {
    /// Total number of frames analyzed.
    pub frame_count: u64,
    /// Minimum luminance across all frames.
    pub global_min_nits: f64,
    /// Maximum luminance across all frames (MaxCLL).
    pub global_max_nits: f64,
    /// Average of per-frame average luminance (MaxFALL approximation).
    pub global_avg_nits: f64,
    /// Number of scene cuts detected.
    pub scene_cut_count: u64,
    /// Number of frames with active area metadata.
    pub active_area_frame_count: u64,
    /// Histogram of peak luminance buckets.
    luminance_histogram: HashMap<u32, u64>,
    /// Sum of per-frame averages (for running calculation).
    avg_accumulator: f64,
}

impl SequenceStats {
    /// Create new empty sequence stats.
    #[must_use]
    pub fn new() -> Self {
        Self {
            frame_count: 0,
            global_min_nits: f64::MAX,
            global_max_nits: 0.0,
            global_avg_nits: 0.0,
            scene_cut_count: 0,
            active_area_frame_count: 0,
            luminance_histogram: HashMap::new(),
            avg_accumulator: 0.0,
        }
    }

    /// Update stats with a new frame analysis.
    #[allow(clippy::cast_precision_loss)]
    pub fn update(&mut self, frame: &FrameAnalysis) {
        self.frame_count += 1;

        if frame.luminance.min_nits < self.global_min_nits {
            self.global_min_nits = frame.luminance.min_nits;
        }
        if frame.luminance.max_nits > self.global_max_nits {
            self.global_max_nits = frame.luminance.max_nits;
        }

        self.avg_accumulator += frame.luminance.avg_nits;
        self.global_avg_nits = self.avg_accumulator / self.frame_count as f64;

        if frame.is_scene_cut {
            self.scene_cut_count += 1;
        }
        if frame.has_active_area {
            self.active_area_frame_count += 1;
        }

        // Bucket peak luminance into 100-nit buckets
        let bucket = (frame.luminance.max_nits / 100.0) as u32;
        *self.luminance_histogram.entry(bucket).or_insert(0) += 1;
    }

    /// Get the MaxCLL (Maximum Content Light Level) value.
    #[must_use]
    pub fn max_cll(&self) -> f64 {
        self.global_max_nits
    }

    /// Get the MaxFALL (Maximum Frame Average Light Level) value.
    #[must_use]
    pub fn max_fall(&self) -> f64 {
        self.global_avg_nits
    }

    /// Global dynamic range in stops.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn dynamic_range_stops(&self) -> f64 {
        if self.global_min_nits <= 0.0 || self.global_max_nits <= 0.0 {
            return 0.0;
        }
        (self.global_max_nits / self.global_min_nits).log2()
    }

    /// Get the luminance histogram (bucket_index -> frame_count).
    /// Each bucket represents a 100-nit range.
    #[must_use]
    pub fn luminance_histogram(&self) -> &HashMap<u32, u64> {
        &self.luminance_histogram
    }

    /// Percentage of frames that are scene cuts.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn scene_cut_percentage(&self) -> f64 {
        if self.frame_count == 0 {
            return 0.0;
        }
        (self.scene_cut_count as f64 / self.frame_count as f64) * 100.0
    }

    /// Estimated average scene length in frames.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_scene_length(&self) -> f64 {
        if self.scene_cut_count == 0 {
            return self.frame_count as f64;
        }
        self.frame_count as f64 / (self.scene_cut_count as f64 + 1.0)
    }
}

impl Default for SequenceStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Compares two frame analyses to detect metadata changes.
#[derive(Debug, Clone)]
pub struct FrameDiff {
    /// Whether luminance metadata changed.
    pub luminance_changed: bool,
    /// Whether trim pass count changed.
    pub trim_count_changed: bool,
    /// Absolute change in max luminance.
    pub max_nits_delta: f64,
    /// Absolute change in average luminance.
    pub avg_nits_delta: f64,
    /// Whether active area changed.
    pub active_area_changed: bool,
}

impl FrameDiff {
    /// Compute the difference between two frame analyses.
    #[must_use]
    pub fn compute(a: &FrameAnalysis, b: &FrameAnalysis) -> Self {
        let max_nits_delta = (b.luminance.max_nits - a.luminance.max_nits).abs();
        let avg_nits_delta = (b.luminance.avg_nits - a.luminance.avg_nits).abs();

        let luminance_changed = max_nits_delta > 0.01 || avg_nits_delta > 0.01;
        let trim_count_changed = a.trim_passes.len() != b.trim_passes.len();
        let active_area_changed = a.has_active_area != b.has_active_area
            || a.active_area_left != b.active_area_left
            || a.active_area_right != b.active_area_right
            || a.active_area_top != b.active_area_top
            || a.active_area_bottom != b.active_area_bottom;

        Self {
            luminance_changed,
            trim_count_changed,
            max_nits_delta,
            avg_nits_delta,
            active_area_changed,
        }
    }

    /// Whether any metadata field changed.
    #[must_use]
    pub fn has_changes(&self) -> bool {
        self.luminance_changed || self.trim_count_changed || self.active_area_changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_luminance_stats_creation() {
        let stats = FrameLuminanceStats::new(0.01, 1000.0, 200.0);
        assert!((stats.min_nits - 0.01).abs() < f64::EPSILON);
        assert!((stats.max_nits - 1000.0).abs() < f64::EPSILON);
        assert!((stats.avg_nits - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frame_luminance_dynamic_range() {
        let stats = FrameLuminanceStats::new(0.01, 10240.0, 100.0);
        let stops = stats.dynamic_range_stops();
        // 10240/0.01 = 1024000, log2 ~= 20
        assert!(stops > 19.0, "Got {stops}");
    }

    #[test]
    fn test_frame_luminance_dynamic_range_zero() {
        let stats = FrameLuminanceStats::new(0.0, 1000.0, 100.0);
        assert!((stats.dynamic_range_stops() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frame_luminance_is_hdr() {
        assert!(FrameLuminanceStats::new(0.01, 500.0, 100.0).is_hdr());
        assert!(!FrameLuminanceStats::new(0.01, 80.0, 40.0).is_hdr());
    }

    #[test]
    fn test_frame_luminance_contrast_ratio() {
        let stats = FrameLuminanceStats::new(0.01, 1000.0, 100.0);
        assert!((stats.contrast_ratio() - 100_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trim_pass_identity() {
        let pass = TrimPassStats::new(600.0);
        assert!(pass.is_identity());

        let mut adjusted = TrimPassStats::new(600.0);
        adjusted.slope = 1.5;
        assert!(!adjusted.is_identity());
    }

    #[test]
    fn test_frame_analysis_creation() {
        let lum = FrameLuminanceStats::new(0.01, 1000.0, 200.0);
        let frame = FrameAnalysis::new(0, lum);
        assert_eq!(frame.frame_number, 0);
        assert_eq!(frame.trim_pass_count(), 0);
        assert!(frame.all_trims_identity());
    }

    #[test]
    fn test_frame_analysis_trim_passes() {
        let lum = FrameLuminanceStats::new(0.01, 1000.0, 200.0);
        let mut frame = FrameAnalysis::new(5, lum);
        frame.add_trim_pass(TrimPassStats::new(600.0));
        frame.add_trim_pass(TrimPassStats::new(100.0));
        assert_eq!(frame.trim_pass_count(), 2);
        assert!(frame.all_trims_identity());

        let mut custom = TrimPassStats::new(300.0);
        custom.slope = 0.8;
        frame.add_trim_pass(custom);
        assert!(!frame.all_trims_identity());
    }

    #[test]
    fn test_sequence_stats_empty() {
        let stats = SequenceStats::new();
        assert_eq!(stats.frame_count, 0);
        assert_eq!(stats.scene_cut_count, 0);
        assert!((stats.scene_cut_percentage() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sequence_stats_update() {
        let mut stats = SequenceStats::new();
        let lum1 = FrameLuminanceStats::new(0.01, 500.0, 100.0);
        let frame1 = FrameAnalysis::new(0, lum1);
        stats.update(&frame1);

        assert_eq!(stats.frame_count, 1);
        assert!((stats.global_max_nits - 500.0).abs() < f64::EPSILON);
        assert!((stats.global_min_nits - 0.01).abs() < f64::EPSILON);

        let lum2 = FrameLuminanceStats::new(0.005, 1000.0, 300.0);
        let mut frame2 = FrameAnalysis::new(1, lum2);
        frame2.is_scene_cut = true;
        stats.update(&frame2);

        assert_eq!(stats.frame_count, 2);
        assert!((stats.global_max_nits - 1000.0).abs() < f64::EPSILON);
        assert!((stats.global_min_nits - 0.005).abs() < f64::EPSILON);
        assert_eq!(stats.scene_cut_count, 1);
    }

    #[test]
    fn test_sequence_stats_max_cll_fall() {
        let mut stats = SequenceStats::new();
        let frames = [
            FrameLuminanceStats::new(0.01, 500.0, 100.0),
            FrameLuminanceStats::new(0.01, 800.0, 200.0),
            FrameLuminanceStats::new(0.01, 300.0, 50.0),
        ];
        for (i, lum) in frames.iter().enumerate() {
            let frame = FrameAnalysis::new(i as u64, *lum);
            stats.update(&frame);
        }
        assert!((stats.max_cll() - 800.0).abs() < f64::EPSILON);
        // MaxFALL should be approximately the average of averages
        let expected_fall = (100.0 + 200.0 + 50.0) / 3.0;
        assert!((stats.max_fall() - expected_fall).abs() < 0.01);
    }

    #[test]
    fn test_sequence_average_scene_length() {
        let mut stats = SequenceStats::new();
        for i in 0..100u64 {
            let lum = FrameLuminanceStats::new(0.01, 500.0, 100.0);
            let mut frame = FrameAnalysis::new(i, lum);
            // Scene cut every 25 frames
            if i > 0 && i % 25 == 0 {
                frame.is_scene_cut = true;
            }
            stats.update(&frame);
        }
        // 100 frames, 3 scene cuts => 4 scenes => avg ~25
        let avg = stats.average_scene_length();
        assert!(avg > 20.0 && avg < 30.0, "Got {avg}");
    }

    #[test]
    fn test_frame_diff_no_changes() {
        let lum = FrameLuminanceStats::new(0.01, 500.0, 100.0);
        let a = FrameAnalysis::new(0, lum);
        let b = FrameAnalysis::new(1, lum);
        let diff = FrameDiff::compute(&a, &b);
        assert!(!diff.has_changes());
        assert!(!diff.luminance_changed);
    }

    #[test]
    fn test_frame_diff_luminance_changed() {
        let a = FrameAnalysis::new(0, FrameLuminanceStats::new(0.01, 500.0, 100.0));
        let b = FrameAnalysis::new(1, FrameLuminanceStats::new(0.01, 800.0, 200.0));
        let diff = FrameDiff::compute(&a, &b);
        assert!(diff.has_changes());
        assert!(diff.luminance_changed);
        assert!((diff.max_nits_delta - 300.0).abs() < f64::EPSILON);
        assert!((diff.avg_nits_delta - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frame_diff_active_area_changed() {
        let lum = FrameLuminanceStats::new(0.01, 500.0, 100.0);
        let a = FrameAnalysis::new(0, lum);
        let mut b = FrameAnalysis::new(1, lum);
        b.has_active_area = true;
        b.active_area_top = 100;
        b.active_area_bottom = 100;
        let diff = FrameDiff::compute(&a, &b);
        assert!(diff.has_changes());
        assert!(diff.active_area_changed);
    }
}
