//! Automated highlight clip generation from playout.
//!
//! Monitors playout and captures highlight clips based on configurable
//! triggers such as sports scores, keyword detection, bookmarks, or manual
//! operator marking.

/// What caused a highlight clip to be created.
#[derive(Debug, Clone, PartialEq)]
pub enum HighlightTrigger {
    /// Operator manually marked the highlight
    Manual,
    /// Triggered by a sports score event
    SportsScore,
    /// Triggered by keyword detection in commentary or captions
    KeywordDetect,
    /// Triggered by wall-clock time schedule
    TimeBased,
    /// Operator set a bookmark during playout
    BookmarkSet,
}

impl HighlightTrigger {
    /// Returns true if this trigger is generated automatically (without
    /// operator action).
    pub fn is_automatic(&self) -> bool {
        matches!(
            self,
            Self::SportsScore | Self::KeywordDetect | Self::TimeBased
        )
    }
}

/// A highlight clip defined by in/out timecodes.
#[derive(Debug, Clone)]
pub struct HighlightClip {
    /// Unique clip identifier
    pub id: u64,
    /// In-point timecode (frame number)
    pub in_tc: u64,
    /// Out-point timecode (frame number)
    pub out_tc: u64,
    /// What triggered this highlight
    pub trigger: HighlightTrigger,
    /// Human-readable label
    pub label: String,
}

impl HighlightClip {
    /// Return the duration of this clip in frames.
    ///
    /// Returns 0 if out_tc <= in_tc.
    pub fn duration_frames(&self) -> u64 {
        self.out_tc.saturating_sub(self.in_tc)
    }

    /// Returns true if this clip is shorter than `min_duration` frames.
    pub fn is_short(&self, min_duration: u64) -> bool {
        self.duration_frames() < min_duration
    }
}

/// Engine that records and queries highlight clips.
#[derive(Debug, Clone)]
pub struct HighlightEngine {
    /// All captured highlight clips in chronological order
    pub clips: Vec<HighlightClip>,
    /// Counter used to assign unique IDs
    pub next_id: u64,
    /// Clips shorter than this (in frames) are rejected
    pub min_duration_frames: u64,
}

impl HighlightEngine {
    /// Create a new engine that rejects clips shorter than `min_duration` frames.
    pub fn new(min_duration: u64) -> Self {
        Self {
            clips: Vec::new(),
            next_id: 1,
            min_duration_frames: min_duration,
        }
    }

    /// Attempt to add a new highlight clip.
    ///
    /// Returns `Some(id)` if the clip was accepted (duration >= minimum), or
    /// `None` if the clip was too short or the out-point is before the
    /// in-point.
    pub fn add_highlight(
        &mut self,
        in_tc: u64,
        out_tc: u64,
        trigger: HighlightTrigger,
        label: impl Into<String>,
    ) -> Option<u64> {
        let duration = out_tc.saturating_sub(in_tc);
        if duration < self.min_duration_frames {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.clips.push(HighlightClip {
            id,
            in_tc,
            out_tc,
            trigger,
            label: label.into(),
        });
        Some(id)
    }

    /// Return references to the most recent `count` clips (most-recent last).
    ///
    /// If `count` exceeds the total number of clips, all clips are returned.
    pub fn recent_clips(&self, count: usize) -> Vec<&HighlightClip> {
        let start = self.clips.len().saturating_sub(count);
        self.clips[start..].iter().collect()
    }

    /// Return references to all clips with the given trigger type.
    pub fn clips_by_trigger(&self, t: &HighlightTrigger) -> Vec<&HighlightClip> {
        self.clips.iter().filter(|c| &c.trigger == t).collect()
    }
}

// ---------------------------------------------------------------------------
// Scene-analysis-based Highlight Extraction
// ---------------------------------------------------------------------------

/// Configuration for [`HighlightExtractor`].
#[derive(Debug, Clone)]
pub struct HighlightConfig {
    /// Minimum clip duration in seconds. Segments shorter than this are
    /// padded or discarded. Default: 3.0 s.
    pub min_duration_secs: f32,
    /// Maximum clip duration in seconds. Segments are capped at this length.
    /// Default: 15.0 s.
    pub max_duration_secs: f32,
    /// Maximum number of highlight segments to return. Default: 5.
    pub top_n: usize,
    /// Weight applied to motion scores when computing composite interest.
    /// Currently reserved for future multi-signal fusion; stored for API
    /// completeness. Default: 0.5.
    pub motion_weight: f32,
}

impl Default for HighlightConfig {
    fn default() -> Self {
        Self {
            min_duration_secs: 3.0,
            max_duration_secs: 15.0,
            top_n: 5,
            motion_weight: 0.5,
        }
    }
}

/// A highlight segment identified by [`HighlightExtractor::extract`].
#[derive(Debug, Clone)]
pub struct HighlightSegment {
    /// Index of the first frame in the segment (inclusive).
    pub start_frame: usize,
    /// Index of the last frame in the segment (inclusive).
    pub end_frame: usize,
    /// Average frame-score across the segment (`0.0`–`1.0`).
    pub score: f32,
}

/// Extracts highlight segments from a per-frame interest score sequence.
///
/// The algorithm:
/// 1. Smooth frame scores with a sliding window of width
///    `ceil(min_duration_secs * fps)` to produce a "region energy" signal.
///    This makes broad high-scoring regions easier to detect than isolated
///    one-frame spikes.
/// 2. Find local maxima in the smoothed signal that exceed the threshold
///    `0.5` — these mark the centres of candidate highlight regions.  A
///    fallback to the **raw** signal is used when no smoothed peak is found
///    above the threshold (e.g. for very sharp Gaussian-shaped score bursts
///    that are diluted by the large smoothing window).
/// 3. Expand each peak centre into a segment spanning
///    `[peak − half_max, peak + half_max]` (where `half_max = max_duration / 2`),
///    then ensure the segment is at least `min_duration` frames long.
/// 4. Score each segment as the mean of the **raw** (un-smoothed) frame
///    scores inside the segment window.
/// 5. Deduplicate overlapping segments, retaining the highest-scored one
///    within each overlapping group.
/// 6. Return the top-N segments sorted by score (highest first).
#[derive(Debug)]
pub struct HighlightExtractor {
    config: HighlightConfig,
}

impl HighlightExtractor {
    /// Create a new extractor with the given configuration.
    pub fn new(config: HighlightConfig) -> Self {
        Self { config }
    }

    /// Score and extract highlight segments from `frame_scores`.
    ///
    /// - `frame_scores`: per-frame interest score in `[0.0, 1.0]`.
    /// - `frame_rate`: frames per second (used to convert duration constraints
    ///   from seconds to frame counts).
    ///
    /// Returns up to `config.top_n` segments sorted by score descending.
    /// Returns an empty `Vec` when `frame_scores` is empty or `frame_rate ≤ 0`.
    pub fn extract(&self, frame_scores: &[f32], frame_rate: f32) -> Vec<HighlightSegment> {
        if frame_scores.is_empty() || frame_rate <= 0.0 {
            return Vec::new();
        }

        let n = frame_scores.len();
        let fps = frame_rate.max(1.0);

        // Segment length constraints in frames.
        let min_frames = ((self.config.min_duration_secs * fps).ceil() as usize).max(1);
        let max_frames = ((self.config.max_duration_secs * fps).ceil() as usize).max(min_frames);

        // --- Step 1: sliding-window smoothing ---
        // Window = min_duration so that sustained activity raises the average
        // above the threshold.  For brief spikes, we fall back to raw scores.
        let smooth_window = min_frames.min(n);
        let smoothed = Self::sliding_average_static(frame_scores, smooth_window);

        // --- Step 2: find local maxima above threshold 0.5 ---
        const THRESHOLD: f32 = 0.5;
        let mut peaks = Self::find_local_maxima_static(&smoothed, THRESHOLD);

        // Fallback: if no peak survived smoothing (e.g. sharp Gaussian input),
        // search the raw scores directly.  This handles bursts whose energy
        // is diluted by the min_duration smoothing window but whose raw peak
        // clearly exceeds the threshold.
        if peaks.is_empty() {
            peaks = Self::find_local_maxima_static(frame_scores, THRESHOLD);
        }

        if peaks.is_empty() {
            return Vec::new();
        }

        // --- Steps 3–4: extend each peak into a segment, score it ---
        let half_max = max_frames / 2;
        let mut segments: Vec<HighlightSegment> = peaks
            .into_iter()
            .map(|peak| {
                let start = peak.saturating_sub(half_max);
                let end = (peak + half_max).min(n.saturating_sub(1));

                // Ensure segment spans at least min_frames.
                let (start, end) = Self::ensure_min_length_static(start, end, min_frames, n);

                let score = Self::mean_score_static(frame_scores, start, end);
                HighlightSegment {
                    start_frame: start,
                    end_frame: end,
                    score,
                }
            })
            .collect();

        // --- Step 5: deduplicate overlapping segments (keep highest score) ---
        segments.sort_by_key(|s| s.start_frame);
        let segments = Self::deduplicate_static(segments);

        // --- Step 6: top-N by score descending ---
        let mut segments = segments;
        segments.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        segments.truncate(self.config.top_n);
        segments
    }

    /// Sliding-window average (centred) using prefix sums — O(n).
    fn sliding_average_static(scores: &[f32], window: usize) -> Vec<f32> {
        let w = window.max(1);
        let n = scores.len();
        let mut out = Vec::with_capacity(n);

        // Build prefix sum array of length n+1.
        let mut prefix = vec![0.0f64; n + 1];
        for (i, &s) in scores.iter().enumerate() {
            prefix[i + 1] = prefix[i] + s as f64;
        }

        let half = w / 2;
        for i in 0..n {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(n);
            let count = (hi - lo) as f64;
            let sum = prefix[hi] - prefix[lo];
            out.push((sum / count) as f32);
        }
        out
    }

    /// Find indices of strict local maxima in `scores` that exceed `threshold`.
    ///
    /// A frame `i` is a local maximum if:
    /// - `scores[i] >= threshold`
    /// - `scores[i] >= scores[i-1]` (or `i == 0`)
    /// - `scores[i] >= scores[i+1]` (or `i == n-1`)
    fn find_local_maxima_static(scores: &[f32], threshold: f32) -> Vec<usize> {
        let n = scores.len();
        let mut peaks = Vec::new();
        for i in 0..n {
            if scores[i] < threshold {
                continue;
            }
            let left_ok = i == 0 || scores[i] >= scores[i - 1];
            let right_ok = i + 1 >= n || scores[i] >= scores[i + 1];
            if left_ok && right_ok {
                peaks.push(i);
            }
        }
        peaks
    }

    /// Grow `[start, end]` to span at least `min_frames` frames, clamped to
    /// `[0, n)`. Tries to extend the end first, then the start if needed.
    fn ensure_min_length_static(
        start: usize,
        end: usize,
        min_frames: usize,
        n: usize,
    ) -> (usize, usize) {
        let current_len = end.saturating_sub(start) + 1;
        if current_len >= min_frames || n == 0 {
            return (start, end);
        }
        let needed = min_frames - current_len;
        let new_end = (end + needed).min(n.saturating_sub(1));
        let extended_len = new_end.saturating_sub(start) + 1;
        if extended_len >= min_frames {
            return (start, new_end);
        }
        // Still short — extend the start backward.
        let still_needed = min_frames - extended_len;
        let new_start = start.saturating_sub(still_needed);
        (new_start, new_end)
    }

    /// Mean of `frame_scores[start..=end]`.
    fn mean_score_static(scores: &[f32], start: usize, end: usize) -> f32 {
        let end_clamped = end.min(scores.len().saturating_sub(1));
        if start > end_clamped || scores.is_empty() {
            return 0.0;
        }
        let slice = &scores[start..=end_clamped];
        slice.iter().sum::<f32>() / slice.len() as f32
    }

    /// Deduplicate overlapping segments by keeping only the highest-scored
    /// segment within each contiguous overlapping group.
    ///
    /// Input must be sorted ascending by `start_frame`.
    fn deduplicate_static(segments: Vec<HighlightSegment>) -> Vec<HighlightSegment> {
        let mut out: Vec<HighlightSegment> = Vec::new();
        for seg in segments {
            if let Some(last) = out.last_mut() {
                if seg.start_frame <= last.end_frame {
                    // Overlapping — keep the higher-scored one.
                    if seg.score > last.score {
                        *last = seg;
                    }
                    continue;
                }
            }
            out.push(seg);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- HighlightTrigger tests ---

    #[test]
    fn test_sports_score_is_automatic() {
        assert!(HighlightTrigger::SportsScore.is_automatic());
    }

    #[test]
    fn test_keyword_detect_is_automatic() {
        assert!(HighlightTrigger::KeywordDetect.is_automatic());
    }

    #[test]
    fn test_time_based_is_automatic() {
        assert!(HighlightTrigger::TimeBased.is_automatic());
    }

    #[test]
    fn test_manual_is_not_automatic() {
        assert!(!HighlightTrigger::Manual.is_automatic());
    }

    #[test]
    fn test_bookmark_is_not_automatic() {
        assert!(!HighlightTrigger::BookmarkSet.is_automatic());
    }

    // --- HighlightClip tests ---

    #[test]
    fn test_clip_duration_frames() {
        let clip = HighlightClip {
            id: 1,
            in_tc: 100,
            out_tc: 150,
            trigger: HighlightTrigger::Manual,
            label: "Goal".into(),
        };
        assert_eq!(clip.duration_frames(), 50);
    }

    #[test]
    fn test_clip_duration_frames_zero_when_out_before_in() {
        let clip = HighlightClip {
            id: 2,
            in_tc: 200,
            out_tc: 100,
            trigger: HighlightTrigger::Manual,
            label: "Invalid".into(),
        };
        assert_eq!(clip.duration_frames(), 0);
    }

    #[test]
    fn test_clip_is_short_true() {
        let clip = HighlightClip {
            id: 3,
            in_tc: 0,
            out_tc: 10,
            trigger: HighlightTrigger::Manual,
            label: "Short".into(),
        };
        assert!(clip.is_short(25));
    }

    #[test]
    fn test_clip_is_short_false() {
        let clip = HighlightClip {
            id: 4,
            in_tc: 0,
            out_tc: 50,
            trigger: HighlightTrigger::Manual,
            label: "Long".into(),
        };
        assert!(!clip.is_short(25));
    }

    // --- HighlightEngine tests ---

    #[test]
    fn test_engine_add_highlight_returns_id() {
        let mut eng = HighlightEngine::new(25);
        let id = eng.add_highlight(0, 50, HighlightTrigger::Manual, "Test");
        assert_eq!(id, Some(1));
    }

    #[test]
    fn test_engine_add_highlight_too_short_returns_none() {
        let mut eng = HighlightEngine::new(25);
        let id = eng.add_highlight(0, 10, HighlightTrigger::Manual, "Too short");
        assert!(id.is_none());
    }

    #[test]
    fn test_engine_ids_increment() {
        let mut eng = HighlightEngine::new(10);
        let id1 = eng.add_highlight(0, 50, HighlightTrigger::Manual, "A");
        let id2 = eng.add_highlight(100, 200, HighlightTrigger::BookmarkSet, "B");
        assert_eq!(id1, Some(1));
        assert_eq!(id2, Some(2));
    }

    #[test]
    fn test_engine_recent_clips_count() {
        let mut eng = HighlightEngine::new(10);
        for i in 0..5u64 {
            eng.add_highlight(i * 100, i * 100 + 50, HighlightTrigger::Manual, "x");
        }
        let recent = eng.recent_clips(3);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn test_engine_recent_clips_exceeds_total() {
        let mut eng = HighlightEngine::new(10);
        eng.add_highlight(0, 50, HighlightTrigger::Manual, "only");
        let recent = eng.recent_clips(10);
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_engine_clips_by_trigger() {
        let mut eng = HighlightEngine::new(10);
        eng.add_highlight(0, 50, HighlightTrigger::Manual, "A");
        eng.add_highlight(100, 200, HighlightTrigger::SportsScore, "B");
        eng.add_highlight(300, 400, HighlightTrigger::SportsScore, "C");
        let sports = eng.clips_by_trigger(&HighlightTrigger::SportsScore);
        assert_eq!(sports.len(), 2);
    }

    #[test]
    fn test_engine_clips_by_trigger_empty() {
        let eng = HighlightEngine::new(10);
        let found = eng.clips_by_trigger(&HighlightTrigger::KeywordDetect);
        assert!(found.is_empty());
    }

    // --- HighlightExtractor tests ---

    /// Helper: build a score array with a single Gaussian-like peak at `center`.
    fn scores_with_peak(len: usize, center: usize, peak: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let dist = (i as f32 - center as f32).abs();
                (peak * (-dist / 5.0).exp()).clamp(0.0, 1.0)
            })
            .collect()
    }

    #[test]
    fn test_highlight_single_peak() {
        // 300-frame input at 25 fps with one peak in the middle.
        let fps = 25.0f32;
        let scores = scores_with_peak(300, 150, 0.9);
        let cfg = HighlightConfig {
            min_duration_secs: 3.0,
            max_duration_secs: 10.0,
            top_n: 5,
            motion_weight: 0.5,
        };
        let ext = HighlightExtractor::new(cfg);
        let highlights = ext.extract(&scores, fps);
        assert!(
            !highlights.is_empty(),
            "expected at least one highlight segment"
        );
        let h = &highlights[0];
        // The segment must cover the peak frame.
        assert!(
            h.start_frame <= 150 && h.end_frame >= 150,
            "peak frame 150 not covered: [{}, {}]",
            h.start_frame,
            h.end_frame
        );
    }

    #[test]
    fn test_highlight_top_n() {
        // 10 well-separated peaks; top_n=3 → exactly 3 results.
        let fps = 25.0f32;
        let len = 1000usize;
        let mut scores = vec![0.0f32; len];
        // Place 10 peaks at 50-frame intervals starting at 50.
        for k in 0..10usize {
            let center = 50 + k * 90;
            if center < len {
                // Each peak has a slightly different height for deterministic top-3.
                let height = 0.55 + k as f32 * 0.04;
                scores[center] = height.min(1.0);
                if center > 0 {
                    scores[center - 1] = (height - 0.1).max(0.0);
                }
                if center + 1 < len {
                    scores[center + 1] = (height - 0.1).max(0.0);
                }
            }
        }
        let cfg = HighlightConfig {
            min_duration_secs: 1.0,
            max_duration_secs: 3.0,
            top_n: 3,
            motion_weight: 0.5,
        };
        let ext = HighlightExtractor::new(cfg);
        let highlights = ext.extract(&scores, fps);
        assert_eq!(
            highlights.len(),
            3,
            "expected exactly 3 highlights (top_n=3)"
        );
        // Results must be sorted by score descending.
        for w in highlights.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "highlights not sorted by score: {} < {}",
                w[0].score,
                w[1].score
            );
        }
    }

    #[test]
    fn test_highlight_min_duration() {
        // A single-frame spike must be padded to at least min_duration_secs.
        let fps = 25.0f32;
        let mut scores = vec![0.0f32; 200];
        scores[100] = 0.9; // single high frame
        let min_duration_secs = 3.0f32;
        let cfg = HighlightConfig {
            min_duration_secs,
            max_duration_secs: 15.0,
            top_n: 1,
            motion_weight: 0.5,
        };
        let ext = HighlightExtractor::new(cfg);
        let highlights = ext.extract(&scores, fps);
        assert!(
            !highlights.is_empty(),
            "expected a highlight from the spike"
        );
        let h = &highlights[0];
        let duration_frames = h.end_frame - h.start_frame + 1;
        let min_frames = (min_duration_secs * fps).ceil() as usize;
        assert!(
            duration_frames >= min_frames,
            "segment too short: {} frames < min {} frames",
            duration_frames,
            min_frames
        );
    }

    #[test]
    fn test_highlight_empty_input() {
        let ext = HighlightExtractor::new(HighlightConfig::default());
        let highlights = ext.extract(&[], 25.0);
        assert!(
            highlights.is_empty(),
            "expected empty output for empty input"
        );
    }

    #[test]
    fn test_highlight_all_zeros() {
        // No frame exceeds threshold → no highlights.
        let ext = HighlightExtractor::new(HighlightConfig::default());
        let scores = vec![0.1f32; 100];
        let highlights = ext.extract(&scores, 25.0);
        assert!(
            highlights.is_empty(),
            "expected no highlights when all scores < 0.5"
        );
    }
}
