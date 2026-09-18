//! Video signal measurement: luma/chroma statistics and broadcast safety analysis.
//!
//! Provides per-frame luma and chroma statistics, a combined `VideoMeasure`
//! for evaluating broadcast safety, and a `VideoMeasureHistory` for
//! tracking the worst-case frame across a clip.

#![allow(dead_code)]

/// Luma (Y) statistics for a single video frame.
#[derive(Debug, Clone)]
pub struct LumaStats {
    /// Minimum luma value in the frame (0–1023 for 10-bit, 0–255 for 8-bit).
    pub min: f32,
    /// Maximum luma value in the frame.
    pub max: f32,
    /// Mean luma value.
    pub mean: f32,
    /// Bit depth of the signal (8 or 10).
    pub bit_depth: u8,
}

impl LumaStats {
    /// Create new luma statistics.
    pub fn new(min: f32, max: f32, mean: f32, bit_depth: u8) -> Self {
        Self {
            min,
            max,
            mean,
            bit_depth,
        }
    }

    /// Maximum legal luma value for the given bit depth.
    #[allow(clippy::cast_precision_loss)]
    fn legal_max(&self) -> f32 {
        match self.bit_depth {
            10 => 940.0,
            _ => 235.0, // 8-bit
        }
    }

    /// Minimum legal luma value for the given bit depth.
    #[allow(clippy::cast_precision_loss)]
    fn legal_min(&self) -> f32 {
        match self.bit_depth {
            10 => 64.0,
            _ => 16.0, // 8-bit
        }
    }

    /// Returns `true` if the maximum luma exceeds the legal broadcast ceiling.
    pub fn is_clipping(&self) -> bool {
        self.max > self.legal_max()
    }

    /// Returns `true` if luma is within legal broadcast range.
    pub fn is_legal(&self) -> bool {
        self.min >= self.legal_min() && self.max <= self.legal_max()
    }

    /// How far above legal max the peak extends (0.0 if not clipping).
    pub fn clip_headroom(&self) -> f32 {
        (self.max - self.legal_max()).max(0.0)
    }
}

/// Chroma (Cb/Cr) statistics for a single video frame.
#[derive(Debug, Clone)]
pub struct ChromaStats {
    /// Maximum chroma excursion (distance from neutral, 0–512 for 10-bit).
    pub max_excursion: f32,
    /// Saturation percentage (0.0–100.0+).
    pub saturation_pct: f32,
    /// Bit depth of the signal (8 or 10).
    pub bit_depth: u8,
}

impl ChromaStats {
    /// Create new chroma statistics.
    pub fn new(max_excursion: f32, saturation_pct: f32, bit_depth: u8) -> Self {
        Self {
            max_excursion,
            saturation_pct,
            bit_depth,
        }
    }

    /// Maximum legal chroma excursion for the bit depth.
    fn legal_excursion_max(&self) -> f32 {
        match self.bit_depth {
            10 => 448.0,
            _ => 112.0,
        }
    }

    /// Returns `true` if chroma is within legal broadcast limits.
    pub fn is_legal(&self) -> bool {
        self.max_excursion <= self.legal_excursion_max() && self.saturation_pct <= 100.0
    }

    /// Returns `true` if chroma is above legal limits.
    pub fn is_over_saturated(&self) -> bool {
        self.saturation_pct > 100.0
    }
}

/// Combined video measurement for a single frame.
#[derive(Debug, Clone)]
pub struct VideoMeasure {
    /// Frame index (0-based).
    pub frame_index: u64,
    /// Presentation timestamp in seconds.
    pub pts_secs: f64,
    /// Luma statistics.
    pub luma: LumaStats,
    /// Chroma statistics.
    pub chroma: ChromaStats,
}

impl VideoMeasure {
    /// Create a new video measure.
    pub fn new(frame_index: u64, pts_secs: f64, luma: LumaStats, chroma: ChromaStats) -> Self {
        Self {
            frame_index,
            pts_secs,
            luma,
            chroma,
        }
    }

    /// Returns `true` if both luma and chroma are within broadcast-legal limits.
    pub fn is_broadcast_safe(&self) -> bool {
        self.luma.is_legal() && self.chroma.is_legal()
    }

    /// Returns `true` if luma is clipping above legal max.
    pub fn has_luma_clipping(&self) -> bool {
        self.luma.is_clipping()
    }

    /// Returns `true` if chroma is over-saturated.
    pub fn has_chroma_excess(&self) -> bool {
        self.chroma.is_over_saturated()
    }

    /// A combined "badness" score (higher = worse signal quality).
    #[allow(clippy::cast_precision_loss)]
    pub fn badness_score(&self) -> f32 {
        let luma_clip = self.luma.clip_headroom();
        let chroma_sat = (self.chroma.saturation_pct - 100.0).max(0.0);
        luma_clip + chroma_sat
    }
}

/// History of video measurements across multiple frames.
#[derive(Debug, Default)]
pub struct VideoMeasureHistory {
    measures: Vec<VideoMeasure>,
}

impl VideoMeasureHistory {
    /// Create an empty history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a frame measurement.
    pub fn add(&mut self, measure: VideoMeasure) {
        self.measures.push(measure);
    }

    /// Number of frames measured.
    pub fn frame_count(&self) -> usize {
        self.measures.len()
    }

    /// Return the frame with the highest badness score (worst signal quality).
    /// Returns `None` if history is empty.
    pub fn worst_frame(&self) -> Option<&VideoMeasure> {
        self.measures.iter().max_by(|a, b| {
            a.badness_score()
                .partial_cmp(&b.badness_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Number of frames that are not broadcast-safe.
    pub fn unsafe_frame_count(&self) -> usize {
        self.measures
            .iter()
            .filter(|m| !m.is_broadcast_safe())
            .count()
    }

    /// Percentage of frames that are broadcast-safe (0.0–100.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn safe_percentage(&self) -> f32 {
        if self.measures.is_empty() {
            return 100.0;
        }
        let safe = self
            .measures
            .iter()
            .filter(|m| m.is_broadcast_safe())
            .count();
        (safe as f32 / self.measures.len() as f32) * 100.0
    }

    /// Returns `true` if all measured frames are broadcast-safe.
    pub fn all_safe(&self) -> bool {
        self.measures.iter().all(VideoMeasure::is_broadcast_safe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn luma_8(min: f32, max: f32) -> LumaStats {
        LumaStats::new(min, max, (min + max) / 2.0, 8)
    }

    fn chroma_8(excursion: f32, sat_pct: f32) -> ChromaStats {
        ChromaStats::new(excursion, sat_pct, 8)
    }

    fn measure(frame: u64, luma: LumaStats, chroma: ChromaStats) -> VideoMeasure {
        VideoMeasure::new(frame, frame as f64 / 25.0, luma, chroma)
    }

    #[test]
    fn test_luma_stats_is_clipping_true() {
        let l = luma_8(16.0, 240.0); // 240 > 235
        assert!(l.is_clipping());
    }

    #[test]
    fn test_luma_stats_is_clipping_false() {
        let l = luma_8(16.0, 230.0);
        assert!(!l.is_clipping());
    }

    #[test]
    fn test_luma_stats_is_legal_true() {
        let l = luma_8(16.0, 235.0);
        assert!(l.is_legal());
    }

    #[test]
    fn test_luma_stats_is_legal_false_below_min() {
        let l = luma_8(0.0, 200.0); // 0 < 16
        assert!(!l.is_legal());
    }

    #[test]
    fn test_luma_stats_clip_headroom_zero_when_no_clip() {
        let l = luma_8(16.0, 200.0);
        assert_eq!(l.clip_headroom(), 0.0);
    }

    #[test]
    fn test_luma_stats_clip_headroom_positive() {
        let l = luma_8(16.0, 245.0); // 245 - 235 = 10
        assert!((l.clip_headroom() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_luma_10bit_legal_max() {
        let l = LumaStats::new(64.0, 940.0, 502.0, 10);
        assert!(l.is_legal());
        assert!(!l.is_clipping());
    }

    #[test]
    fn test_chroma_stats_is_legal_true() {
        let c = chroma_8(100.0, 95.0);
        assert!(c.is_legal());
    }

    #[test]
    fn test_chroma_stats_is_legal_false_oversaturated() {
        let c = chroma_8(100.0, 110.0);
        assert!(!c.is_legal());
    }

    #[test]
    fn test_chroma_stats_is_over_saturated() {
        let c = chroma_8(80.0, 105.0);
        assert!(c.is_over_saturated());
    }

    #[test]
    fn test_video_measure_is_broadcast_safe_true() {
        let m = measure(0, luma_8(16.0, 230.0), chroma_8(80.0, 90.0));
        assert!(m.is_broadcast_safe());
    }

    #[test]
    fn test_video_measure_is_broadcast_safe_false_luma() {
        let m = measure(0, luma_8(16.0, 240.0), chroma_8(80.0, 90.0));
        assert!(!m.is_broadcast_safe());
    }

    #[test]
    fn test_video_measure_has_luma_clipping() {
        let m = measure(0, luma_8(16.0, 250.0), chroma_8(80.0, 90.0));
        assert!(m.has_luma_clipping());
    }

    #[test]
    fn test_video_measure_badness_score_zero_when_safe() {
        let m = measure(0, luma_8(16.0, 220.0), chroma_8(80.0, 90.0));
        assert_eq!(m.badness_score(), 0.0);
    }

    #[test]
    fn test_history_worst_frame() {
        let mut h = VideoMeasureHistory::new();
        h.add(measure(0, luma_8(16.0, 220.0), chroma_8(80.0, 90.0))); // safe
        h.add(measure(1, luma_8(16.0, 250.0), chroma_8(80.0, 90.0))); // clipping
        let worst = h.worst_frame().expect("should succeed in test");
        assert_eq!(worst.frame_index, 1);
    }

    #[test]
    fn test_history_worst_frame_empty() {
        let h = VideoMeasureHistory::new();
        assert!(h.worst_frame().is_none());
    }

    #[test]
    fn test_history_unsafe_frame_count() {
        let mut h = VideoMeasureHistory::new();
        h.add(measure(0, luma_8(16.0, 230.0), chroma_8(80.0, 90.0)));
        h.add(measure(1, luma_8(16.0, 250.0), chroma_8(80.0, 90.0)));
        assert_eq!(h.unsafe_frame_count(), 1);
    }

    #[test]
    fn test_history_safe_percentage_all_safe() {
        let mut h = VideoMeasureHistory::new();
        h.add(measure(0, luma_8(16.0, 220.0), chroma_8(80.0, 90.0)));
        assert!((h.safe_percentage() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_history_all_safe_empty() {
        let h = VideoMeasureHistory::new();
        assert!(h.all_safe());
    }
}
