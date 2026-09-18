//! RPU statistics: min/max/avg luminance reporting per scene.
//!
//! Aggregates per-frame Level 1 metadata across a sequence of frames and scenes,
//! providing structured luminance statistics for mastering QC workflows.

use crate::Level1Metadata;

// ---------------------------------------------------------------------------
// Per-frame statistics
// ---------------------------------------------------------------------------

/// Luminance statistics for a single RPU frame.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameLuminanceStats {
    /// Frame index within the sequence (0-based).
    pub frame_index: u64,
    /// Minimum PQ code value (0–4095).
    pub min_pq: u16,
    /// Maximum PQ code value (0–4095).
    pub max_pq: u16,
    /// Average PQ code value (0–4095).
    pub avg_pq: u16,
    /// Minimum luminance in nits (approximate).
    pub min_nits: f32,
    /// Maximum luminance in nits (approximate).
    pub max_nits: f32,
    /// Average luminance in nits (approximate).
    pub avg_nits: f32,
}

impl FrameLuminanceStats {
    /// Build from a [`Level1Metadata`] instance.
    #[must_use]
    pub fn from_level1(frame_index: u64, l1: &Level1Metadata) -> Self {
        Self {
            frame_index,
            min_pq: l1.min_pq,
            max_pq: l1.max_pq,
            avg_pq: l1.avg_pq,
            min_nits: pq_to_nits_approx(l1.min_pq),
            max_nits: pq_to_nits_approx(l1.max_pq),
            avg_nits: pq_to_nits_approx(l1.avg_pq),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-scene statistics
// ---------------------------------------------------------------------------

/// Aggregated luminance statistics for a single scene (group of frames).
#[derive(Debug, Clone)]
pub struct SceneLuminanceStats {
    /// Scene index within the sequence (0-based).
    pub scene_index: u32,
    /// Index of the first frame in this scene.
    pub start_frame: u64,
    /// Index of the last frame in this scene (inclusive).
    pub end_frame: u64,
    /// Minimum PQ across all frames in the scene.
    pub min_pq: u16,
    /// Maximum PQ across all frames in the scene.
    pub max_pq: u16,
    /// Arithmetic mean average PQ across all frames.
    pub avg_pq: f32,
    /// Minimum luminance in nits.
    pub min_nits: f32,
    /// Maximum luminance in nits.
    pub max_nits: f32,
    /// Average luminance in nits.
    pub avg_nits: f32,
    /// Number of frames in the scene.
    pub frame_count: u32,
}

impl SceneLuminanceStats {
    /// Compute scene statistics from a slice of per-frame stats.
    ///
    /// Returns `None` if `frames` is empty.
    #[must_use]
    pub fn compute(scene_index: u32, frames: &[FrameLuminanceStats]) -> Option<Self> {
        if frames.is_empty() {
            return None;
        }

        let start_frame = frames.first().map_or(0, |f| f.frame_index);
        let end_frame = frames.last().map_or(0, |f| f.frame_index);

        let min_pq = frames.iter().map(|f| f.min_pq).min().unwrap_or(0);
        let max_pq = frames.iter().map(|f| f.max_pq).max().unwrap_or(0);
        let avg_pq = frames.iter().map(|f| f64::from(f.avg_pq)).sum::<f64>() / frames.len() as f64;

        let min_nits = pq_to_nits_approx(min_pq);
        let max_nits = pq_to_nits_approx(max_pq);
        Some(Self {
            scene_index,
            start_frame,
            end_frame,
            min_pq,
            max_pq,
            avg_pq: avg_pq as f32,
            min_nits,
            max_nits,
            avg_nits: pq_to_nits_approx((avg_pq as u16).min(4095)),
            frame_count: frames.len() as u32,
        })
    }
}

// ---------------------------------------------------------------------------
// Sequence-level statistics
// ---------------------------------------------------------------------------

/// Full sequence luminance report covering all scenes and global aggregates.
#[derive(Debug, Clone)]
pub struct SequenceLuminanceReport {
    /// Per-scene statistics.
    pub scenes: Vec<SceneLuminanceStats>,
    /// Global minimum PQ across the entire sequence.
    pub global_min_pq: u16,
    /// Global maximum PQ across the entire sequence.
    pub global_max_pq: u16,
    /// Global average PQ across the entire sequence.
    pub global_avg_pq: f32,
    /// Total number of frames in the sequence.
    pub total_frames: u64,
    /// Total number of scenes.
    pub total_scenes: u32,
    /// Global minimum luminance in nits.
    pub global_min_nits: f32,
    /// Global maximum luminance in nits.
    pub global_max_nits: f32,
}

impl SequenceLuminanceReport {
    /// Build a sequence report from per-scene statistics.
    ///
    /// Returns an empty-but-valid report when `scenes` is empty.
    #[must_use]
    pub fn from_scenes(scenes: Vec<SceneLuminanceStats>) -> Self {
        if scenes.is_empty() {
            return Self {
                scenes,
                global_min_pq: 0,
                global_max_pq: 0,
                global_avg_pq: 0.0,
                total_frames: 0,
                total_scenes: 0,
                global_min_nits: 0.0,
                global_max_nits: 0.0,
            };
        }

        let global_min_pq = scenes.iter().map(|s| s.min_pq).min().unwrap_or(0);
        let global_max_pq = scenes.iter().map(|s| s.max_pq).max().unwrap_or(0);
        let total_frames: u64 = scenes.iter().map(|s| u64::from(s.frame_count)).sum();

        let weighted_avg: f64 = scenes
            .iter()
            .map(|s| f64::from(s.avg_pq) * f64::from(s.frame_count))
            .sum::<f64>();
        let global_avg_pq = if total_frames > 0 {
            (weighted_avg / total_frames as f64) as f32
        } else {
            0.0
        };

        let total_scenes = scenes.len() as u32;
        let global_min_nits = pq_to_nits_approx(global_min_pq);
        let global_max_nits = pq_to_nits_approx(global_max_pq);

        Self {
            scenes,
            global_min_pq,
            global_max_pq,
            global_avg_pq,
            total_frames,
            total_scenes,
            global_min_nits,
            global_max_nits,
        }
    }

    /// Returns a human-readable summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Sequence: {} frames, {} scenes | PQ min={} max={} avg={:.0} | \
             Nits min={:.3} max={:.1} avg={:.1}",
            self.total_frames,
            self.total_scenes,
            self.global_min_pq,
            self.global_max_pq,
            self.global_avg_pq,
            self.global_min_nits,
            self.global_max_nits,
            pq_to_nits_approx((self.global_avg_pq as u16).min(4095)),
        )
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Incremental statistics accumulator for a stream of RPU frames.
///
/// Call [`RpuStatsBuilder::push_frame`] for each frame and
/// [`RpuStatsBuilder::finish`] to get the final report.
#[derive(Debug, Default)]
pub struct RpuStatsBuilder {
    /// Accumulated per-frame statistics.
    frame_stats: Vec<FrameLuminanceStats>,
    /// Scene boundaries (frame indices at which a new scene starts).
    scene_boundaries: Vec<u64>,
}

impl RpuStatsBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a frame's Level 1 metadata to the accumulator.
    ///
    /// If `is_scene_start` is `true`, the frame is recorded as the beginning
    /// of a new scene.
    pub fn push_frame(&mut self, frame_index: u64, l1: &Level1Metadata, is_scene_start: bool) {
        let stats = FrameLuminanceStats::from_level1(frame_index, l1);
        if is_scene_start || self.frame_stats.is_empty() {
            self.scene_boundaries.push(frame_index);
        }
        self.frame_stats.push(stats);
    }

    /// Consume the builder and return the full sequence report.
    #[must_use]
    pub fn finish(self) -> SequenceLuminanceReport {
        if self.frame_stats.is_empty() {
            return SequenceLuminanceReport::from_scenes(Vec::new());
        }

        // Partition frames into scenes
        let boundaries = &self.scene_boundaries;
        let mut scenes: Vec<SceneLuminanceStats> = Vec::new();

        for (scene_idx, &scene_start) in boundaries.iter().enumerate() {
            let scene_end = boundaries.get(scene_idx + 1).copied().unwrap_or(u64::MAX);

            let scene_frames: Vec<FrameLuminanceStats> = self
                .frame_stats
                .iter()
                .filter(|f| f.frame_index >= scene_start && f.frame_index < scene_end)
                .cloned()
                .collect();

            if let Some(stats) = SceneLuminanceStats::compute(scene_idx as u32, &scene_frames) {
                scenes.push(stats);
            }
        }

        SequenceLuminanceReport::from_scenes(scenes)
    }

    /// Number of frames accumulated so far.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frame_stats.len()
    }

    /// Number of scene boundaries recorded so far.
    #[must_use]
    pub fn scene_count(&self) -> usize {
        self.scene_boundaries.len()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a PQ code (0–4095) to an approximate nits value.
///
/// Uses a simplified ST.2084 inverse EOTF.
#[must_use]
#[inline]
pub fn pq_to_nits_approx(pq: u16) -> f32 {
    if pq == 0 {
        return 0.0;
    }
    const M1_INV: f64 = 1.0 / 0.159_301_758_113_479_8;
    const M2_INV: f64 = 1.0 / 78.843_750;
    const C1: f64 = 0.835_937_5;
    const C2: f64 = 18.851_562_5;
    const C3: f64 = 18.6875;

    let v = (f64::from(pq) / 4095.0).powf(M2_INV);
    let y = ((v - C1).max(0.0) / (C2 - C3 * v)).powf(M1_INV);
    (y * 10_000.0).clamp(0.0, 10_001.0) as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn l1(min: u16, max: u16, avg: u16) -> Level1Metadata {
        Level1Metadata {
            min_pq: min,
            max_pq: max,
            avg_pq: avg,
        }
    }

    #[test]
    fn test_frame_stats_from_level1() {
        let meta = l1(100, 3000, 1500);
        let stats = FrameLuminanceStats::from_level1(0, &meta);
        assert_eq!(stats.min_pq, 100);
        assert_eq!(stats.max_pq, 3000);
        assert_eq!(stats.avg_pq, 1500);
        assert!(stats.min_nits >= 0.0);
        assert!(stats.max_nits > stats.min_nits);
    }

    #[test]
    fn test_scene_stats_single_frame() {
        let frames = vec![FrameLuminanceStats::from_level1(0, &l1(50, 2000, 1000))];
        let scene = SceneLuminanceStats::compute(0, &frames).expect("should compute");
        assert_eq!(scene.frame_count, 1);
        assert_eq!(scene.min_pq, 50);
        assert_eq!(scene.max_pq, 2000);
    }

    #[test]
    fn test_scene_stats_empty() {
        let result = SceneLuminanceStats::compute(0, &[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_scene_stats_multiple_frames() {
        let frames = vec![
            FrameLuminanceStats::from_level1(0, &l1(100, 2000, 1000)),
            FrameLuminanceStats::from_level1(1, &l1(50, 3000, 1500)),
            FrameLuminanceStats::from_level1(2, &l1(200, 2500, 1200)),
        ];
        let scene = SceneLuminanceStats::compute(0, &frames).expect("should compute");
        assert_eq!(scene.min_pq, 50);
        assert_eq!(scene.max_pq, 3000);
        assert_eq!(scene.frame_count, 3);
        assert!((scene.avg_pq - (1000.0 + 1500.0 + 1200.0) / 3.0).abs() < 1.0);
    }

    #[test]
    fn test_builder_empty_finish() {
        let builder = RpuStatsBuilder::new();
        let report = builder.finish();
        assert_eq!(report.total_frames, 0);
        assert_eq!(report.total_scenes, 0);
    }

    #[test]
    fn test_builder_single_scene() {
        let mut builder = RpuStatsBuilder::new();
        for i in 0..5u64 {
            builder.push_frame(i, &l1(100, 2000 + i as u16 * 100, 1000), i == 0);
        }
        assert_eq!(builder.frame_count(), 5);
        assert_eq!(builder.scene_count(), 1);

        let report = builder.finish();
        assert_eq!(report.total_frames, 5);
        assert_eq!(report.total_scenes, 1);
    }

    #[test]
    fn test_builder_multiple_scenes() {
        let mut builder = RpuStatsBuilder::new();
        // Scene 0: frames 0–2
        for i in 0..3u64 {
            builder.push_frame(i, &l1(100, 2000, 1000), i == 0);
        }
        // Scene 1: frames 3–5
        for i in 3..6u64 {
            builder.push_frame(i, &l1(50, 3000, 1500), i == 3);
        }
        let report = builder.finish();
        assert_eq!(report.total_scenes, 2);
        assert_eq!(report.total_frames, 6);
        assert_eq!(report.global_min_pq, 50);
        assert_eq!(report.global_max_pq, 3000);
    }

    #[test]
    fn test_sequence_report_summary_non_empty() {
        let mut builder = RpuStatsBuilder::new();
        builder.push_frame(0, &l1(100, 2000, 1000), true);
        let report = builder.finish();
        let summary = report.summary();
        assert!(summary.contains("frame"));
        assert!(summary.contains("scene"));
    }

    #[test]
    fn test_pq_to_nits_zero() {
        assert_eq!(pq_to_nits_approx(0), 0.0);
    }

    #[test]
    fn test_pq_to_nits_max() {
        let nits = pq_to_nits_approx(4095);
        assert!(nits > 9000.0, "nits={nits}");
    }

    #[test]
    fn test_pq_to_nits_100nit_range() {
        // PQ code ≈ 2081 maps to ~100 nits
        let nits = pq_to_nits_approx(2081);
        assert!(nits > 80.0 && nits < 130.0, "nits={nits}");
    }

    #[test]
    fn test_sequence_report_empty() {
        let report = SequenceLuminanceReport::from_scenes(vec![]);
        assert_eq!(report.global_min_pq, 0);
        assert_eq!(report.global_max_pq, 0);
        assert_eq!(report.total_frames, 0);
    }
}
