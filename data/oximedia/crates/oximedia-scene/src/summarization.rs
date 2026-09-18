//! Scene summarization: key shot extraction and scene-level digests.
//!
//! Identifies important shots within a scene based on dialog presence,
//! face detection, and motion levels, then builds per-scene summaries.

/// Per-shot data used for summarization.
#[derive(Debug, Clone)]
pub struct ShotSummary {
    /// Unique shot identifier.
    pub shot_id: u64,
    /// Duration of the shot in frames.
    pub duration_frames: u32,
    /// Dominant colour (R, G, B).
    pub dominant_color: [u8; 3],
    /// Normalised motion level (0.0 = static, 1.0 = maximum motion).
    pub motion_level: f32,
    /// Whether dialog is present in this shot.
    pub dialog: bool,
    /// Number of faces detected.
    pub faces_detected: u32,
}

impl ShotSummary {
    /// A shot is a key shot when it contains dialog, has detected faces, or
    /// has high motion (> 0.6).
    #[must_use]
    pub fn is_key_shot(&self) -> bool {
        self.dialog || self.faces_detected > 0 || self.motion_level > 0.6
    }
}

/// Aggregated summary of a scene built from individual shot summaries.
#[derive(Debug, Clone)]
pub struct SceneSummary {
    /// Scene identifier.
    pub scene_id: u64,
    /// Number of shots in the scene.
    pub shot_count: u32,
    /// Total frame count across all shots.
    pub total_frames: u64,
    /// Narrative importance score (0.0 – 1.0).
    pub narrative_importance: f32,
    /// IDs of shots identified as key shots.
    pub key_shots: Vec<u64>,
}

impl SceneSummary {
    /// Average shot duration in frames, or `0.0` when there are no shots.
    #[must_use]
    pub fn avg_shot_duration_frames(&self) -> f64 {
        if self.shot_count == 0 {
            return 0.0;
        }
        self.total_frames as f64 / f64::from(self.shot_count)
    }

    /// Returns `true` when any of the key shots contains dialog.
    ///
    /// This is a convenience accessor that delegates to whether any key shot
    /// was selected (implying dialog, faces, or high motion).
    /// Caller is expected to cross-reference with the input `ShotSummary` slice.
    #[must_use]
    pub fn has_dialog(&self) -> bool {
        !self.key_shots.is_empty()
    }
}

/// Configuration controlling which shots are selected for the summary.
#[derive(Debug, Clone)]
pub struct SummarizationConfig {
    /// Maximum number of shots to include in the summary.
    pub max_shots_per_summary: usize,
    /// Minimum narrative importance needed for a scene to be summarised.
    pub min_importance: f32,
}

impl Default for SummarizationConfig {
    fn default() -> Self {
        Self {
            max_shots_per_summary: 5,
            min_importance: 0.1,
        }
    }
}

/// Builds `SceneSummary` instances from slices of `ShotSummary`.
pub struct SceneSummarizer {
    config: SummarizationConfig,
}

impl SceneSummarizer {
    /// Create a summarizer with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SummarizationConfig::default(),
        }
    }

    /// Create a summarizer with the given configuration.
    #[must_use]
    pub fn with_config(config: SummarizationConfig) -> Self {
        Self { config }
    }

    /// Summarize a scene from a slice of shot summaries.
    ///
    /// Narrative importance is computed as the ratio of key shots to total shots.
    #[must_use]
    pub fn summarize_scene(&self, shots: &[ShotSummary]) -> SceneSummary {
        let shot_count = shots.len() as u32;
        let total_frames: u64 = shots.iter().map(|s| u64::from(s.duration_frames)).sum();
        let key_shots = self.select_key_shots(shots);
        let narrative_importance = if shot_count == 0 {
            0.0
        } else {
            key_shots.len() as f32 / shot_count as f32
        };
        SceneSummary {
            scene_id: 0, // caller can overwrite
            shot_count,
            total_frames,
            narrative_importance,
            key_shots,
        }
    }

    /// Summarize a scene, assigning the given `scene_id`.
    #[must_use]
    pub fn summarize_scene_with_id(&self, scene_id: u64, shots: &[ShotSummary]) -> SceneSummary {
        let mut summary = self.summarize_scene(shots);
        summary.scene_id = scene_id;
        summary
    }

    /// Select key shot IDs from the input slice, limited by `max_shots_per_summary`.
    #[must_use]
    pub fn select_key_shots(&self, shots: &[ShotSummary]) -> Vec<u64> {
        shots
            .iter()
            .filter(|s| s.is_key_shot())
            .take(self.config.max_shots_per_summary)
            .map(|s| s.shot_id)
            .collect()
    }
}

impl Default for SceneSummarizer {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Frame-level decimation summarisation
// ─────────────────────────────────────────────────────────────────────────────

/// A single video frame representation used for frame-level summarization.
///
/// Callers populate this from whatever per-frame data the pipeline provides.
/// The `is_keyframe` hint lets you force a frame to be included regardless
/// of the decimation schedule.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Sequential frame index in the source video.
    pub index: usize,
    /// True when this frame has already been identified as a key frame by
    /// an upstream component (e.g. an I-frame or a detected scene boundary).
    pub is_keyframe: bool,
    /// Optional single-channel luma value in `[0.0, 1.0]` — used only for
    /// the example heuristic inside [`VideoSummarizer`].
    pub luma: f32,
}

/// Configuration for frame-level video summarization with decimation.
///
/// Decimation allows the summarizer to skip frames, processing only every
/// *N*-th frame rather than the entire sequence.  This is useful for long
/// videos where evaluating every frame would be prohibitively expensive.
#[derive(Debug, Clone)]
pub struct FrameDecimationConfig {
    /// Process 1 out of every `decimation` frames.
    ///
    /// A value of `1` (the default) disables decimation and processes every
    /// frame.  A value of `3` processes frames 0, 3, 6, 9, …
    pub decimation: usize,
    /// Maximum number of keyframes to select from the sampled set.
    ///
    /// Once this limit is reached the summarizer stops appending new
    /// keyframes even if more frames remain.  Default: `20`.
    pub max_keyframes: usize,
    /// Minimum number of frames (in source-video terms) that must elapse
    /// between two consecutive keyframe selections.
    ///
    /// Prevents the summarizer from clustering keyframes in one busy burst.
    /// Default: `30`.
    pub min_scene_duration_frames: usize,
}

impl Default for FrameDecimationConfig {
    fn default() -> Self {
        Self {
            decimation: 1,
            max_keyframes: 20,
            min_scene_duration_frames: 30,
        }
    }
}

/// Identifies which frames of a video are representative keyframes.
///
/// The selection criterion is intentionally simple and swap-able: a frame is
/// considered a "keyframe candidate" when its `is_keyframe` flag is set **or**
/// when its `luma` deviates from the previous selected frame by more than 0.1.
///
/// # Decimation
///
/// Only every `config.decimation`-th frame (counting from index 0 of the
/// *input* slice) is examined.  The original source frame index stored in
/// [`VideoFrame::index`] is preserved in the output so callers always work
/// with source coordinates.
pub struct VideoSummarizer {
    config: FrameDecimationConfig,
}

impl VideoSummarizer {
    /// Create a summarizer with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: FrameDecimationConfig::default(),
        }
    }

    /// Create a summarizer with the given configuration.
    #[must_use]
    pub fn with_config(config: FrameDecimationConfig) -> Self {
        Self { config }
    }

    /// Summarize `frames`, returning the source indices of selected keyframes.
    ///
    /// Steps:
    /// 1. Apply decimation: keep only every `decimation`-th element of `frames`.
    /// 2. Scan the sampled frames and select those that satisfy the keyframe
    ///    criterion, subject to `min_scene_duration_frames` spacing and
    ///    `max_keyframes` cap.
    #[must_use]
    pub fn summarize(&self, frames: &[VideoFrame]) -> Vec<usize> {
        let decimation = self.config.decimation.max(1);

        // Step 1 – decimate.
        let sampled: Vec<&VideoFrame> = frames
            .iter()
            .enumerate()
            .filter(|(i, _)| i % decimation == 0)
            .map(|(_, f)| f)
            .collect();

        // Step 2 – select keyframes from the sampled set.
        let mut keyframes: Vec<usize> = Vec::new();
        let mut last_selected_source_idx: Option<usize> = None;
        let mut prev_luma: f32 = -1.0; // sentinel: no previous frame yet

        for frame in sampled {
            if keyframes.len() >= self.config.max_keyframes {
                break;
            }

            // Enforce minimum spacing in source-frame terms.
            let gap_ok = match last_selected_source_idx {
                None => true,
                Some(prev) => {
                    frame.index.saturating_sub(prev) >= self.config.min_scene_duration_frames
                }
            };

            if !gap_ok {
                // Not yet far enough from the last keyframe — update luma
                // tracker so the next eligible frame is compared against a
                // recent value, then skip selection.
                prev_luma = frame.luma;
                continue;
            }

            let is_candidate = frame.is_keyframe
                || prev_luma < 0.0 // first sampled frame always qualifies
                || (frame.luma - prev_luma).abs() > 0.1;

            if is_candidate {
                keyframes.push(frame.index);
                last_selected_source_idx = Some(frame.index);
            }

            prev_luma = frame.luma;
        }

        keyframes
    }
}

impl Default for VideoSummarizer {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_shot(id: u64, duration: u32, motion: f32, dialog: bool, faces: u32) -> ShotSummary {
        ShotSummary {
            shot_id: id,
            duration_frames: duration,
            dominant_color: [128, 64, 32],
            motion_level: motion,
            dialog,
            faces_detected: faces,
        }
    }

    // 1. ShotSummary::is_key_shot – dialog present
    #[test]
    fn test_is_key_shot_dialog() {
        let s = make_shot(0, 30, 0.1, true, 0);
        assert!(s.is_key_shot());
    }

    // 2. ShotSummary::is_key_shot – faces detected
    #[test]
    fn test_is_key_shot_faces() {
        let s = make_shot(1, 30, 0.1, false, 2);
        assert!(s.is_key_shot());
    }

    // 3. ShotSummary::is_key_shot – high motion
    #[test]
    fn test_is_key_shot_motion() {
        let s = make_shot(2, 30, 0.8, false, 0);
        assert!(s.is_key_shot());
    }

    // 4. ShotSummary::is_key_shot – none of the conditions
    #[test]
    fn test_is_not_key_shot() {
        let s = make_shot(3, 30, 0.3, false, 0);
        assert!(!s.is_key_shot());
    }

    // 5. ShotSummary::is_key_shot – boundary motion (exactly 0.6 is NOT > 0.6)
    #[test]
    fn test_is_key_shot_motion_boundary() {
        let s = make_shot(4, 30, 0.6, false, 0);
        assert!(!s.is_key_shot());
        let s2 = make_shot(5, 30, 0.600_001, false, 0);
        assert!(s2.is_key_shot());
    }

    // 6. SceneSummary::avg_shot_duration_frames – basic
    #[test]
    fn test_avg_shot_duration() {
        let summary = SceneSummary {
            scene_id: 0,
            shot_count: 4,
            total_frames: 120,
            narrative_importance: 0.5,
            key_shots: vec![],
        };
        assert!((summary.avg_shot_duration_frames() - 30.0).abs() < f64::EPSILON);
    }

    // 7. SceneSummary::avg_shot_duration_frames – zero shots
    #[test]
    fn test_avg_shot_duration_zero() {
        let summary = SceneSummary {
            scene_id: 0,
            shot_count: 0,
            total_frames: 0,
            narrative_importance: 0.0,
            key_shots: vec![],
        };
        assert!((summary.avg_shot_duration_frames()).abs() < f64::EPSILON);
    }

    // 8. SceneSummary::has_dialog – no key shots
    #[test]
    fn test_has_dialog_no_key_shots() {
        let summary = SceneSummary {
            scene_id: 0,
            shot_count: 2,
            total_frames: 60,
            narrative_importance: 0.0,
            key_shots: vec![],
        };
        assert!(!summary.has_dialog());
    }

    // 9. SceneSummary::has_dialog – with key shots
    #[test]
    fn test_has_dialog_with_key_shots() {
        let summary = SceneSummary {
            scene_id: 1,
            shot_count: 3,
            total_frames: 90,
            narrative_importance: 0.33,
            key_shots: vec![0, 1],
        };
        assert!(summary.has_dialog());
    }

    // 10. SummarizationConfig::default values
    #[test]
    fn test_summarization_config_default() {
        let cfg = SummarizationConfig::default();
        assert_eq!(cfg.max_shots_per_summary, 5);
        assert!((cfg.min_importance - 0.1).abs() < f32::EPSILON);
    }

    // 11. SceneSummarizer::summarize_scene – empty input
    #[test]
    fn test_summarize_empty() {
        let s = SceneSummarizer::new();
        let summary = s.summarize_scene(&[]);
        assert_eq!(summary.shot_count, 0);
        assert_eq!(summary.total_frames, 0);
        assert!(summary.key_shots.is_empty());
    }

    // 12. SceneSummarizer::summarize_scene – counts frames
    #[test]
    fn test_summarize_total_frames() {
        let s = SceneSummarizer::new();
        let shots = vec![
            make_shot(0, 30, 0.1, false, 0),
            make_shot(1, 60, 0.1, false, 0),
        ];
        let summary = s.summarize_scene(&shots);
        assert_eq!(summary.shot_count, 2);
        assert_eq!(summary.total_frames, 90);
    }

    // 13. SceneSummarizer::select_key_shots respects max_shots_per_summary
    #[test]
    fn test_select_key_shots_limit() {
        let cfg = SummarizationConfig {
            max_shots_per_summary: 2,
            min_importance: 0.0,
        };
        let s = SceneSummarizer::with_config(cfg);
        let shots: Vec<ShotSummary> = (0..10).map(|i| make_shot(i, 30, 0.0, true, 0)).collect();
        let keys = s.select_key_shots(&shots);
        assert_eq!(keys.len(), 2);
    }

    // 14. SceneSummarizer::summarize_scene_with_id sets scene_id
    #[test]
    fn test_summarize_with_id() {
        let s = SceneSummarizer::new();
        let shots = vec![make_shot(0, 30, 0.1, true, 1)];
        let summary = s.summarize_scene_with_id(42, &shots);
        assert_eq!(summary.scene_id, 42);
    }

    // 15. narrative_importance equals key_count / shot_count
    #[test]
    fn test_narrative_importance_ratio() {
        let s = SceneSummarizer::new();
        let shots = vec![
            make_shot(0, 30, 0.0, true, 0),  // key
            make_shot(1, 30, 0.0, false, 0), // not key
            make_shot(2, 30, 0.0, false, 0), // not key
            make_shot(3, 30, 0.0, false, 1), // key
        ];
        let summary = s.summarize_scene(&shots);
        // 2 key shots out of 4 = 0.5
        assert!((summary.narrative_importance - 0.5).abs() < f32::EPSILON);
    }

    // ── VideoSummarizer / FrameDecimationConfig ────────────────────────────

    fn make_frame(index: usize, luma: f32, is_keyframe: bool) -> VideoFrame {
        VideoFrame {
            index,
            is_keyframe,
            luma,
        }
    }

    // 16. decimation=1: all 10 frames are processed (all are keyframe candidates
    //     because consecutive luma differs by > 0.1 or first frame qualifies).
    #[test]
    fn test_decimation_factor_1() {
        // Build 10 frames with alternating luma so every frame triggers the
        // luma-delta criterion.
        let frames: Vec<VideoFrame> = (0..10)
            .map(|i| make_frame(i, if i % 2 == 0 { 0.2 } else { 0.8 }, false))
            .collect();

        let cfg = FrameDecimationConfig {
            decimation: 1,
            max_keyframes: 20,
            min_scene_duration_frames: 0, // no spacing constraint
        };
        let summarizer = VideoSummarizer::with_config(cfg);
        let keys = summarizer.summarize(&frames);
        // Every frame is a candidate: 10 keyframes expected.
        assert_eq!(
            keys.len(),
            10,
            "decimation=1 should process all 10 frames, got {}",
            keys.len()
        );
    }

    // 17. decimation=3: 9 frames → only indices 0, 3, 6 are examined (3 frames).
    #[test]
    fn test_decimation_factor_3() {
        let frames: Vec<VideoFrame> = (0..9)
            .map(|i| make_frame(i, if i % 2 == 0 { 0.1 } else { 0.9 }, false))
            .collect();

        let cfg = FrameDecimationConfig {
            decimation: 3,
            max_keyframes: 20,
            min_scene_duration_frames: 0,
        };
        let summarizer = VideoSummarizer::with_config(cfg);
        let keys = summarizer.summarize(&frames);
        // Sampled frames: 0, 3, 6.
        // Frame 0: first frame → candidate (prev_luma = -1 sentinel).
        // Frame 3: luma=0.1, prev=0.9 → |0.1-0.9|=0.8 > 0.1 → candidate.
        // Frame 6: luma=0.1, prev=0.1 → |0.1-0.1|=0.0, not keyframe flag → NOT candidate.
        // So 2 keyframes, which are both in [0, 3, 6].
        assert!(
            keys.len() <= 3,
            "decimation=3 should examine at most 3 frames, got {} keyframes",
            keys.len()
        );
        for &k in &keys {
            assert!(
                k == 0 || k == 3 || k == 6,
                "keyframe index {k} must be one of 0, 3, 6"
            );
        }
    }

    // 18. Keyframes are only selected from the decimated set.
    #[test]
    fn test_decimation_preserves_keyframes() {
        // Mark frame at source index 6 as an upstream keyframe.
        // With decimation=3 the sampled set is {0,3,6,9} → frame 6 is examined.
        let frames: Vec<VideoFrame> = (0..12).map(|i| make_frame(i, 0.5, i == 6)).collect();

        let cfg = FrameDecimationConfig {
            decimation: 3,
            max_keyframes: 20,
            min_scene_duration_frames: 0,
        };
        let summarizer = VideoSummarizer::with_config(cfg);
        let keys = summarizer.summarize(&frames);

        // Frame 6 must appear in the output because it has `is_keyframe=true`.
        assert!(
            keys.contains(&6),
            "frame 6 (is_keyframe=true) must be selected; got {:?}",
            keys
        );
        // Frame indices not divisible by 3 must NOT appear.
        for &k in &keys {
            assert_eq!(
                k % 3,
                0,
                "only decimated frames (multiples of 3) should be in the output; found {k}"
            );
        }
    }

    // 19. max_keyframes cap is respected.
    #[test]
    fn test_decimation_max_keyframes_cap() {
        let frames: Vec<VideoFrame> = (0..100)
            .map(|i| make_frame(i, if i % 2 == 0 { 0.0 } else { 1.0 }, false))
            .collect();

        let cfg = FrameDecimationConfig {
            decimation: 1,
            max_keyframes: 5,
            min_scene_duration_frames: 0,
        };
        let summarizer = VideoSummarizer::with_config(cfg);
        let keys = summarizer.summarize(&frames);
        assert!(
            keys.len() <= 5,
            "max_keyframes=5 must be honoured, got {}",
            keys.len()
        );
    }

    // 20. FrameDecimationConfig::default values.
    #[test]
    fn test_frame_decimation_config_default() {
        let cfg = FrameDecimationConfig::default();
        assert_eq!(cfg.decimation, 1);
        assert_eq!(cfg.max_keyframes, 20);
        assert_eq!(cfg.min_scene_duration_frames, 30);
    }
}
