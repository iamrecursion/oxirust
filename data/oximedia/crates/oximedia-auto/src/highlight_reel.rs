//! Automatic highlight reel generation via greedy score-based clip selection.

#![allow(dead_code)]

/// Score metadata for a single candidate clip.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipScore {
    /// Unique clip identifier.
    pub clip_id: u64,
    /// First frame of the clip (inclusive).
    pub start_frame: u64,
    /// Last frame of the clip (inclusive).
    pub end_frame: u64,
    /// Composite highlight score (higher is better).
    pub score: f32,
    /// Human-readable reason this clip was scored.
    pub reason: String,
}

impl ClipScore {
    /// Number of frames in this clip.
    ///
    /// Returns 0 when `end_frame < start_frame`.
    pub fn duration_frames(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }

    /// Returns `true` when the clip score meets or exceeds `threshold`.
    pub fn is_highlight(&self, threshold: f32) -> bool {
        self.score >= threshold
    }
}

/// Configuration for highlight reel generation.
#[derive(Debug, Clone, PartialEq)]
pub struct HighlightConfig {
    /// Minimum clip duration in frames to consider for the reel.
    pub min_clip_duration_frames: u32,
    /// Maximum total reel duration in frames.
    pub max_reel_duration_frames: u32,
    /// Score threshold — clips below this score are ignored.
    pub score_threshold: f32,
}

impl HighlightConfig {
    /// Default configuration suitable for most content.
    pub fn default_config() -> Self {
        Self {
            min_clip_duration_frames: 24,       // ~1 second at 24 fps
            max_reel_duration_frames: 24 * 120, // 2 minutes at 24 fps
            score_threshold: 0.5,
        }
    }
}

impl Default for HighlightConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Selects the best clips from a scored candidate list to form a highlight reel.
pub struct HighlightSelector {
    /// Active configuration.
    pub config: HighlightConfig,
}

impl HighlightSelector {
    /// Create a new selector with the given configuration.
    pub fn new(config: HighlightConfig) -> Self {
        Self { config }
    }

    /// Select highlights using a greedy descending-score approach.
    ///
    /// 1. Filters clips shorter than `min_clip_duration_frames` and below `score_threshold`.
    /// 2. Sorts remaining clips by score descending.
    /// 3. Adds clips greedily until `max_reel_duration_frames` would be exceeded.
    /// 4. Returns the selected clips sorted by `start_frame`.
    pub fn select_highlights(&self, clips: &[ClipScore]) -> Vec<ClipScore> {
        let min_dur = u64::from(self.config.min_clip_duration_frames);
        let max_total = u64::from(self.config.max_reel_duration_frames);

        // Filter
        let mut eligible: Vec<&ClipScore> = clips
            .iter()
            .filter(|c| c.duration_frames() >= min_dur && c.score >= self.config.score_threshold)
            .collect();

        // Sort by score descending (ties broken by clip_id for determinism)
        eligible.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.clip_id.cmp(&b.clip_id))
        });

        let mut selected: Vec<ClipScore> = Vec::new();
        let mut total: u64 = 0;

        for clip in eligible {
            let dur = clip.duration_frames();
            if total + dur <= max_total {
                selected.push(clip.clone());
                total += dur;
            }
        }

        // Sort selected by start_frame
        selected.sort_by_key(|c| c.start_frame);
        selected
    }

    /// Total frame count of the provided clip list.
    pub fn total_duration(clips: &[ClipScore]) -> u64 {
        clips.iter().map(ClipScore::duration_frames).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clip(id: u64, start: u64, end: u64, score: f32) -> ClipScore {
        ClipScore {
            clip_id: id,
            start_frame: start,
            end_frame: end,
            score,
            reason: format!("test clip {id}"),
        }
    }

    // ---- ClipScore tests ----

    #[test]
    fn test_duration_frames_normal() {
        let c = make_clip(1, 10, 50, 0.8);
        assert_eq!(c.duration_frames(), 40);
    }

    #[test]
    fn test_duration_frames_zero_when_inverted() {
        let c = make_clip(1, 50, 10, 0.5);
        assert_eq!(c.duration_frames(), 0);
    }

    #[test]
    fn test_is_highlight_above_threshold() {
        let c = make_clip(1, 0, 100, 0.9);
        assert!(c.is_highlight(0.5));
    }

    #[test]
    fn test_is_highlight_below_threshold() {
        let c = make_clip(1, 0, 100, 0.3);
        assert!(!c.is_highlight(0.5));
    }

    #[test]
    fn test_is_highlight_at_exact_threshold() {
        let c = make_clip(1, 0, 100, 0.5);
        assert!(c.is_highlight(0.5));
    }

    // ---- HighlightConfig tests ----

    #[test]
    fn test_default_config_values() {
        let cfg = HighlightConfig::default_config();
        assert_eq!(cfg.min_clip_duration_frames, 24);
        assert!((cfg.score_threshold - 0.5).abs() < 1e-6);
        assert!(cfg.max_reel_duration_frames > 0);
    }

    // ---- HighlightSelector tests ----

    #[test]
    fn test_select_empty_input() {
        let sel = HighlightSelector::new(HighlightConfig::default_config());
        let result = sel.select_highlights(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_select_filters_below_threshold() {
        let sel = HighlightSelector::new(HighlightConfig::default_config());
        let clips = vec![make_clip(1, 0, 100, 0.1), make_clip(2, 200, 300, 0.2)];
        let result = sel.select_highlights(&clips);
        assert!(result.is_empty());
    }

    #[test]
    fn test_select_filters_short_clips() {
        let mut cfg = HighlightConfig::default_config();
        cfg.min_clip_duration_frames = 30;
        let sel = HighlightSelector::new(cfg);
        // This clip is only 10 frames
        let clips = vec![make_clip(1, 0, 10, 0.9)];
        let result = sel.select_highlights(&clips);
        assert!(result.is_empty());
    }

    #[test]
    fn test_select_greedy_by_score() {
        let cfg = HighlightConfig {
            min_clip_duration_frames: 1,
            max_reel_duration_frames: 150,
            score_threshold: 0.0,
        };
        let sel = HighlightSelector::new(cfg);
        // Three clips summing to 150 frames total; all should fit
        let clips = vec![
            make_clip(1, 0, 50, 0.9),
            make_clip(2, 60, 110, 0.7),
            make_clip(3, 120, 170, 0.5),
        ];
        let result = sel.select_highlights(&clips);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_select_respects_max_duration() {
        let cfg = HighlightConfig {
            min_clip_duration_frames: 1,
            max_reel_duration_frames: 50,
            score_threshold: 0.0,
        };
        let sel = HighlightSelector::new(cfg);
        let clips = vec![
            make_clip(1, 0, 30, 0.9),   // 30 frames, best score
            make_clip(2, 40, 70, 0.8),  // 30 frames → would exceed 50 total
            make_clip(3, 80, 100, 0.7), // 20 frames, fits
        ];
        let result = sel.select_highlights(&clips);
        // clip 1 (30f, best) + clip 3 (20f) = 50 frames
        assert_eq!(result.len(), 2);
        assert!(result.iter().any(|c| c.clip_id == 1));
        assert!(result.iter().any(|c| c.clip_id == 3));
    }

    #[test]
    fn test_select_output_sorted_by_start_frame() {
        let cfg = HighlightConfig {
            min_clip_duration_frames: 1,
            max_reel_duration_frames: 10_000,
            score_threshold: 0.0,
        };
        let sel = HighlightSelector::new(cfg);
        let clips = vec![
            make_clip(3, 200, 250, 0.6),
            make_clip(1, 0, 50, 0.9),
            make_clip(2, 100, 150, 0.8),
        ];
        let result = sel.select_highlights(&clips);
        assert_eq!(result[0].start_frame, 0);
        assert_eq!(result[1].start_frame, 100);
        assert_eq!(result[2].start_frame, 200);
    }

    #[test]
    fn test_total_duration_empty() {
        assert_eq!(HighlightSelector::total_duration(&[]), 0);
    }

    #[test]
    fn test_total_duration_multiple_clips() {
        let clips = vec![make_clip(1, 0, 50, 1.0), make_clip(2, 100, 130, 1.0)];
        assert_eq!(HighlightSelector::total_duration(&clips), 80);
    }
}
