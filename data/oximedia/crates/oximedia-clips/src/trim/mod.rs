//! Smart clip trimming - automatic in/out point detection.

/// Reason for a suggested trim point.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimReason {
    /// Silence detected in audio.
    SilenceDetected,
    /// Black frame detected in video.
    BlackFrame,
    /// Scene change detected.
    SceneChange,
    /// Nearest keyframe.
    Keyframe,
    /// Manually specified.
    Manual,
}

/// A single trim point with metadata.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct TrimPoint {
    /// Frame number.
    pub frame: u64,
    /// Reason this trim point was suggested.
    pub reason: TrimReason,
    /// Confidence in this suggestion (0.0..=1.0).
    pub confidence: f32,
}

impl TrimPoint {
    /// Create a new trim point.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(frame: u64, reason: TrimReason, confidence: f32) -> Self {
        Self {
            frame,
            reason,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

/// A suggested in/out trim range.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct TrimSuggestion {
    /// Suggested in point.
    pub in_point: TrimPoint,
    /// Suggested out point.
    pub out_point: TrimPoint,
    /// Duration in frames between in and out.
    pub duration_frames: u64,
}

impl TrimSuggestion {
    /// Create a new trim suggestion, computing duration automatically.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(in_point: TrimPoint, out_point: TrimPoint) -> Self {
        let duration_frames = out_point.frame.saturating_sub(in_point.frame);
        Self {
            in_point,
            out_point,
            duration_frames,
        }
    }
}

/// Detects silence regions in audio.
#[allow(dead_code)]
pub struct SilenceTrimmer;

impl SilenceTrimmer {
    /// Find silence regions in a sequence of RMS values.
    ///
    /// Returns a list of `(start_frame, end_frame)` pairs for each silence region.
    ///
    /// * `rms_values` - Per-frame RMS amplitude values.
    /// * `threshold_db` - Silence threshold in dBFS (e.g. -40.0).
    /// * `min_duration_frames` - Minimum consecutive silent frames to count as a region.
    #[allow(dead_code)]
    #[must_use]
    pub fn find_silence_regions(
        rms_values: &[f32],
        threshold_db: f32,
        min_duration_frames: u32,
    ) -> Vec<(u64, u64)> {
        let mut regions = Vec::new();
        let mut in_silence = false;
        let mut silence_start: u64 = 0;

        for (i, &rms) in rms_values.iter().enumerate() {
            let db = 20.0 * (rms.max(1e-10_f32)).log10();
            let frame = i as u64;

            if db < threshold_db {
                if !in_silence {
                    in_silence = true;
                    silence_start = frame;
                }
            } else if in_silence {
                in_silence = false;
                let duration = frame - silence_start;
                if duration >= u64::from(min_duration_frames) {
                    regions.push((silence_start, frame - 1));
                }
            }
        }

        // Handle trailing silence
        if in_silence {
            let end_frame = rms_values.len() as u64 - 1;
            let duration = end_frame - silence_start + 1;
            if duration >= u64::from(min_duration_frames) {
                regions.push((silence_start, end_frame));
            }
        }

        regions
    }
}

/// Detects black frames in video.
#[allow(dead_code)]
pub struct BlackFrameDetector;

impl BlackFrameDetector {
    /// Find black frames by checking per-frame luma means.
    ///
    /// * `luma_means` - Average luma value per frame (0.0..=255.0 or 0.0..=1.0).
    /// * `threshold` - Luma values below this are considered black.
    #[allow(dead_code)]
    #[must_use]
    pub fn find_black_frames(luma_means: &[f32], threshold: f32) -> Vec<u64> {
        luma_means
            .iter()
            .enumerate()
            .filter_map(|(i, &luma)| {
                if luma < threshold {
                    Some(i as u64)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Suggests smart trim points for a clip.
#[allow(dead_code)]
pub struct SmartTrimmer;

impl SmartTrimmer {
    /// Suggest trim in/out points for a clip.
    ///
    /// * `duration_frames` - Total number of frames in the clip.
    /// * `fps` - Frames per second.
    /// * `silence_frames` - Frame indices where silence starts (in/out edges).
    /// * `scene_changes` - Frame indices of detected scene changes.
    #[allow(dead_code)]
    #[must_use]
    pub fn suggest_trims(
        duration_frames: u64,
        _fps: f64,
        silence_frames: &[u64],
        scene_changes: &[u64],
    ) -> Vec<TrimSuggestion> {
        if duration_frames == 0 {
            return Vec::new();
        }

        let mut suggestions = Vec::new();

        // Use scene changes to create suggestions
        if !scene_changes.is_empty() {
            let mut boundaries: Vec<u64> = std::iter::once(0)
                .chain(scene_changes.iter().copied())
                .chain(std::iter::once(duration_frames))
                .collect();
            boundaries.dedup();

            for window in boundaries.windows(2) {
                let start = window[0];
                let end = window[1].saturating_sub(1);
                if end > start {
                    // Refine in/out if there's silence near boundaries
                    let in_frame = Self::nearest_silence_boundary(start, silence_frames, 5, true)
                        .unwrap_or(start);
                    let out_frame = Self::nearest_silence_boundary(end, silence_frames, 5, false)
                        .unwrap_or(end);

                    let in_reason = if in_frame != start {
                        TrimReason::SilenceDetected
                    } else {
                        TrimReason::SceneChange
                    };
                    let out_reason = if out_frame != end {
                        TrimReason::SilenceDetected
                    } else {
                        TrimReason::SceneChange
                    };

                    let in_pt = TrimPoint::new(in_frame, in_reason, 0.85);
                    let out_pt = TrimPoint::new(out_frame, out_reason, 0.85);
                    suggestions.push(TrimSuggestion::new(in_pt, out_pt));
                }
            }
        } else {
            // No scene changes; use silence to trim head/tail
            let in_frame = silence_frames
                .iter()
                .find(|&&f| f < duration_frames / 4)
                .copied()
                .unwrap_or(0);
            let out_frame = silence_frames
                .iter()
                .rev()
                .find(|&&f| f > duration_frames * 3 / 4)
                .copied()
                .unwrap_or(duration_frames.saturating_sub(1));

            let in_reason = if in_frame > 0 {
                TrimReason::SilenceDetected
            } else {
                TrimReason::Keyframe
            };
            let out_reason = if out_frame < duration_frames.saturating_sub(1) {
                TrimReason::SilenceDetected
            } else {
                TrimReason::Keyframe
            };

            let in_pt = TrimPoint::new(in_frame, in_reason, 0.7);
            let out_pt = TrimPoint::new(out_frame, out_reason, 0.7);
            suggestions.push(TrimSuggestion::new(in_pt, out_pt));
        }

        suggestions
    }

    /// Find the nearest silence frame within `window` frames of `target`.
    /// If `prefer_after` is true, prefer a frame >= target; otherwise <= target.
    fn nearest_silence_boundary(
        target: u64,
        silence_frames: &[u64],
        window: u64,
        prefer_after: bool,
    ) -> Option<u64> {
        silence_frames
            .iter()
            .copied()
            .filter(|&f| {
                if prefer_after {
                    f >= target && f <= target + window
                } else {
                    f <= target && f + window >= target
                }
            })
            .min_by_key(|&f| {
                if prefer_after {
                    f.wrapping_sub(target)
                } else {
                    target.wrapping_sub(f)
                }
            })
    }
}

/// Batch trim operations across multiple clips.
#[allow(dead_code)]
pub struct TrimBatch {
    /// List of `(clip_id, TrimSuggestion)` pairs.
    pub clips: Vec<(u64, TrimSuggestion)>,
}

impl TrimBatch {
    /// Create a new empty batch.
    #[allow(dead_code)]
    #[must_use]
    pub fn new() -> Self {
        Self { clips: Vec::new() }
    }

    /// Add a trim suggestion for a clip.
    #[allow(dead_code)]
    pub fn add(&mut self, clip_id: u64, suggestion: TrimSuggestion) {
        self.clips.push((clip_id, suggestion));
    }

    /// Get the number of clips in this batch.
    #[allow(dead_code)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.clips.len()
    }

    /// Returns true if the batch is empty.
    #[allow(dead_code)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }

    /// Get total trimmed duration across all clips (in frames).
    #[allow(dead_code)]
    #[must_use]
    pub fn total_duration_frames(&self) -> u64 {
        self.clips.iter().map(|(_, s)| s.duration_frames).sum()
    }
}

impl Default for TrimBatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_point_confidence_clamped() {
        let pt = TrimPoint::new(10, TrimReason::Manual, 1.5);
        assert_eq!(pt.confidence, 1.0);

        let pt2 = TrimPoint::new(5, TrimReason::BlackFrame, -0.5);
        assert_eq!(pt2.confidence, 0.0);
    }

    #[test]
    fn test_trim_suggestion_duration() {
        let in_pt = TrimPoint::new(10, TrimReason::Keyframe, 0.9);
        let out_pt = TrimPoint::new(50, TrimReason::Keyframe, 0.9);
        let sug = TrimSuggestion::new(in_pt, out_pt);
        assert_eq!(sug.duration_frames, 40);
    }

    #[test]
    fn test_trim_suggestion_same_frame() {
        let in_pt = TrimPoint::new(20, TrimReason::Manual, 1.0);
        let out_pt = TrimPoint::new(20, TrimReason::Manual, 1.0);
        let sug = TrimSuggestion::new(in_pt, out_pt);
        assert_eq!(sug.duration_frames, 0);
    }

    #[test]
    fn test_silence_trimmer_basic() {
        // All frames below -60 dBFS (rms ~ 0.001)
        let rms_values = vec![0.001_f32; 10];
        let regions = SilenceTrimmer::find_silence_regions(&rms_values, -40.0, 3);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0], (0, 9));
    }

    #[test]
    fn test_silence_trimmer_no_silence() {
        let rms_values = vec![0.5_f32; 20]; // loud
        let regions = SilenceTrimmer::find_silence_regions(&rms_values, -40.0, 3);
        assert!(regions.is_empty());
    }

    #[test]
    fn test_silence_trimmer_mixed() {
        // Frames 0-4 silence, 5-14 loud, 15-19 silence
        let mut rms_values = vec![0.001_f32; 5];
        rms_values.extend(vec![0.5_f32; 10]);
        rms_values.extend(vec![0.001_f32; 5]);
        let regions = SilenceTrimmer::find_silence_regions(&rms_values, -40.0, 3);
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0], (0, 4));
        assert_eq!(regions[1], (15, 19));
    }

    #[test]
    fn test_silence_trimmer_min_duration_filter() {
        // Only 2 silent frames - should be filtered (min=3)
        let mut rms_values = vec![0.5_f32; 5];
        rms_values.extend(vec![0.001_f32; 2]);
        rms_values.extend(vec![0.5_f32; 5]);
        let regions = SilenceTrimmer::find_silence_regions(&rms_values, -40.0, 3);
        assert!(regions.is_empty());
    }

    #[test]
    fn test_black_frame_detector_basic() {
        let luma = vec![0.01, 0.02, 0.5, 0.8, 0.03, 0.9, 0.01];
        let black = BlackFrameDetector::find_black_frames(&luma, 0.05);
        assert_eq!(black, vec![0, 1, 4, 6]);
    }

    #[test]
    fn test_black_frame_detector_none() {
        let luma = vec![0.5_f32; 10];
        let black = BlackFrameDetector::find_black_frames(&luma, 0.05);
        assert!(black.is_empty());
    }

    #[test]
    fn test_black_frame_detector_all() {
        let luma = vec![0.01_f32; 5];
        let black = BlackFrameDetector::find_black_frames(&luma, 0.05);
        assert_eq!(black.len(), 5);
    }

    #[test]
    fn test_smart_trimmer_empty() {
        let suggestions = SmartTrimmer::suggest_trims(0, 25.0, &[], &[]);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_smart_trimmer_with_scene_changes() {
        let suggestions = SmartTrimmer::suggest_trims(100, 25.0, &[], &[50]);
        assert_eq!(suggestions.len(), 2);
        // First segment: frames 0..49
        assert_eq!(suggestions[0].in_point.frame, 0);
        // Second segment: frames 50..99
        assert_eq!(suggestions[1].in_point.frame, 50);
    }

    #[test]
    fn test_smart_trimmer_no_scene_changes() {
        let suggestions = SmartTrimmer::suggest_trims(100, 25.0, &[], &[]);
        assert_eq!(suggestions.len(), 1);
    }

    #[test]
    fn test_trim_batch() {
        let mut batch = TrimBatch::new();
        assert!(batch.is_empty());

        let in_pt = TrimPoint::new(0, TrimReason::Keyframe, 1.0);
        let out_pt = TrimPoint::new(100, TrimReason::Keyframe, 1.0);
        batch.add(1, TrimSuggestion::new(in_pt, out_pt));

        let in_pt2 = TrimPoint::new(0, TrimReason::Keyframe, 1.0);
        let out_pt2 = TrimPoint::new(50, TrimReason::Keyframe, 1.0);
        batch.add(2, TrimSuggestion::new(in_pt2, out_pt2));

        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
        assert_eq!(batch.total_duration_frames(), 150);
    }

    #[test]
    fn test_trim_reasons() {
        assert_eq!(
            TrimPoint::new(0, TrimReason::SilenceDetected, 0.8).reason,
            TrimReason::SilenceDetected
        );
        assert_eq!(
            TrimPoint::new(0, TrimReason::SceneChange, 0.8).reason,
            TrimReason::SceneChange
        );
    }
}
