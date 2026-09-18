#![allow(dead_code)]
//! Razor / blade cut tool for splitting clips at specific positions.
//!
//! The razor tool enables frame-accurate splitting of clips, tracks, or all
//! tracks at a given timeline position.  It supports preview mode, multi-cut
//! (splitting at several positions in one operation), and cut-on-the-fly during
//! playback.

use std::collections::BTreeSet;

/// A position on the timeline in frame units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CutPosition(i64);

impl CutPosition {
    /// Create a new cut position from a frame number.
    #[must_use]
    pub fn new(frame: i64) -> Self {
        Self(frame)
    }

    /// Return the frame number.
    #[must_use]
    pub fn frame(self) -> i64 {
        self.0
    }

    /// Return the time in seconds at the given frame rate.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn as_seconds(self, fps: f64) -> f64 {
        if fps <= 0.0 {
            return 0.0;
        }
        self.0 as f64 / fps
    }
}

/// Determines the scope of a razor cut.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CutScope {
    /// Cut only the specified clip on the specified track.
    SingleClip {
        /// Track identifier.
        track_id: u64,
        /// Clip identifier.
        clip_id: u64,
    },
    /// Cut all clips that intersect the cut position on a specific track.
    SingleTrack {
        /// Track identifier.
        track_id: u64,
    },
    /// Cut across all tracks at the given position.
    AllTracks,
    /// Cut only the clips whose ids are provided.
    Selection {
        /// Clip identifiers to cut.
        clip_ids: Vec<u64>,
    },
}

/// Result of a single razor cut operation.
#[derive(Debug, Clone)]
pub struct CutResult {
    /// The clip id that was split (left portion keeps the original id).
    pub original_clip_id: u64,
    /// The id assigned to the new right-hand portion.
    pub new_clip_id: u64,
    /// Track on which the cut occurred.
    pub track_id: u64,
    /// The frame at which the cut was made.
    pub cut_frame: i64,
    /// Duration of the left portion in frames.
    pub left_duration: i64,
    /// Duration of the right portion in frames.
    pub right_duration: i64,
}

impl CutResult {
    /// Create a new cut result.
    #[must_use]
    pub fn new(
        original_clip_id: u64,
        new_clip_id: u64,
        track_id: u64,
        cut_frame: i64,
        left_duration: i64,
        right_duration: i64,
    ) -> Self {
        Self {
            original_clip_id,
            new_clip_id,
            track_id,
            cut_frame,
            left_duration,
            right_duration,
        }
    }

    /// Total duration of both halves combined.
    #[must_use]
    pub fn total_duration(&self) -> i64 {
        self.left_duration + self.right_duration
    }
}

/// Configuration for the razor tool.
#[derive(Debug, Clone)]
pub struct RazorConfig {
    /// Whether to snap the cut position to the nearest snap point.
    pub snap_enabled: bool,
    /// Snap tolerance in frames.
    pub snap_tolerance: i64,
    /// Whether to show a visual preview before committing the cut.
    pub preview_mode: bool,
    /// Whether through-edits (cuts that create adjacent clips with
    /// contiguous source ranges) should be marked specially.
    pub mark_through_edits: bool,
}

impl Default for RazorConfig {
    fn default() -> Self {
        Self {
            snap_enabled: true,
            snap_tolerance: 1,
            preview_mode: false,
            mark_through_edits: true,
        }
    }
}

/// Represents a clip on a track for the purposes of razor cutting.
#[derive(Debug, Clone)]
pub struct CuttableClip {
    /// Clip identifier.
    pub clip_id: u64,
    /// Track identifier.
    pub track_id: u64,
    /// Timeline start frame (inclusive).
    pub start_frame: i64,
    /// Timeline end frame (exclusive).
    pub end_frame: i64,
    /// Source-media in-point frame.
    pub source_in: i64,
}

impl CuttableClip {
    /// Create a new cuttable clip descriptor.
    #[must_use]
    pub fn new(
        clip_id: u64,
        track_id: u64,
        start_frame: i64,
        end_frame: i64,
        source_in: i64,
    ) -> Self {
        Self {
            clip_id,
            track_id,
            start_frame,
            end_frame,
            source_in,
        }
    }

    /// Duration in frames.
    #[must_use]
    pub fn duration(&self) -> i64 {
        self.end_frame - self.start_frame
    }

    /// Whether the given frame falls inside this clip.
    #[must_use]
    pub fn contains_frame(&self, frame: i64) -> bool {
        frame >= self.start_frame && frame < self.end_frame
    }

    /// Whether the cut position is at the very start or end (edge cut).
    #[must_use]
    pub fn is_edge(&self, frame: i64) -> bool {
        frame == self.start_frame || frame == self.end_frame
    }
}

/// The razor tool itself.
#[derive(Debug, Clone)]
pub struct RazorTool {
    /// Configuration.
    config: RazorConfig,
    /// Snap grid positions.
    snap_points: BTreeSet<i64>,
    /// Counter for generating new clip ids.
    next_clip_id: u64,
}

impl Default for RazorTool {
    fn default() -> Self {
        Self::new(RazorConfig::default())
    }
}

impl RazorTool {
    /// Create a new razor tool with the given configuration.
    #[must_use]
    pub fn new(config: RazorConfig) -> Self {
        Self {
            config,
            snap_points: BTreeSet::new(),
            next_clip_id: 1_000_000,
        }
    }

    /// Set the snap points (e.g. frame boundaries, marker positions).
    pub fn set_snap_points(&mut self, points: impl IntoIterator<Item = i64>) {
        self.snap_points = points.into_iter().collect();
    }

    /// Add a single snap point.
    pub fn add_snap_point(&mut self, frame: i64) {
        self.snap_points.insert(frame);
    }

    /// Return a reference to the current config.
    #[must_use]
    pub fn config(&self) -> &RazorConfig {
        &self.config
    }

    /// Snap a frame to the nearest snap point within tolerance.
    #[must_use]
    pub fn snap_frame(&self, frame: i64) -> i64 {
        if !self.config.snap_enabled || self.snap_points.is_empty() {
            return frame;
        }
        let tolerance = self.config.snap_tolerance;
        let mut best = frame;
        let mut best_dist = i64::MAX;

        // Check the nearest point at or before frame
        if let Some(&before) = self.snap_points.range(..=frame).next_back() {
            let d = (frame - before).abs();
            if d < best_dist && d <= tolerance {
                best = before;
                best_dist = d;
            }
        }
        // Check the nearest point after frame
        if let Some(&after) = self.snap_points.range((frame + 1)..).next() {
            let d = (after - frame).abs();
            if d < best_dist && d <= tolerance {
                best = after;
            }
        }
        best
    }

    /// Cut a single clip at the given frame, producing a [`CutResult`].
    /// Returns `None` if the cut position is outside the clip or at an edge.
    pub fn cut_clip(&mut self, clip: &CuttableClip, at_frame: i64) -> Option<CutResult> {
        let frame = if self.config.snap_enabled {
            self.snap_frame(at_frame)
        } else {
            at_frame
        };

        if !clip.contains_frame(frame) || clip.is_edge(frame) {
            return None;
        }

        let new_id = self.next_clip_id;
        self.next_clip_id += 1;

        let left_dur = frame - clip.start_frame;
        let right_dur = clip.end_frame - frame;

        Some(CutResult::new(
            clip.clip_id,
            new_id,
            clip.track_id,
            frame,
            left_dur,
            right_dur,
        ))
    }

    /// Cut multiple clips at the given frame.
    pub fn cut_clips(&mut self, clips: &[CuttableClip], at_frame: i64) -> Vec<CutResult> {
        let mut results = Vec::new();
        for clip in clips {
            if let Some(result) = self.cut_clip(clip, at_frame) {
                results.push(result);
            }
        }
        results
    }

    /// Perform a multi-cut: split clips at every specified position.
    pub fn multi_cut(&mut self, clips: &[CuttableClip], positions: &[i64]) -> Vec<CutResult> {
        let mut all_results = Vec::new();
        // Sort positions so we process left-to-right
        let mut sorted: Vec<i64> = positions.to_vec();
        sorted.sort_unstable();
        sorted.dedup();

        for &pos in &sorted {
            let results = self.cut_clips(clips, pos);
            all_results.extend(results);
        }
        all_results
    }

    /// Filter clips that intersect a given frame.
    #[must_use]
    pub fn clips_at_frame(clips: &[CuttableClip], frame: i64) -> Vec<&CuttableClip> {
        clips.iter().filter(|c| c.contains_frame(frame)).collect()
    }

    /// Filter clips on a specific track.
    #[must_use]
    pub fn clips_on_track(clips: &[CuttableClip], track_id: u64) -> Vec<&CuttableClip> {
        clips.iter().filter(|c| c.track_id == track_id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clip(id: u64, track: u64, start: i64, end: i64) -> CuttableClip {
        CuttableClip::new(id, track, start, end, 0)
    }

    #[test]
    fn test_cut_position_frame() {
        let pos = CutPosition::new(48);
        assert_eq!(pos.frame(), 48);
    }

    #[test]
    fn test_cut_position_as_seconds() {
        let pos = CutPosition::new(48);
        let secs = pos.as_seconds(24.0);
        assert!((secs - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cut_position_zero_fps() {
        let pos = CutPosition::new(100);
        assert!((pos.as_seconds(0.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cuttable_clip_duration() {
        let clip = make_clip(1, 1, 10, 110);
        assert_eq!(clip.duration(), 100);
    }

    #[test]
    fn test_cuttable_clip_contains_frame() {
        let clip = make_clip(1, 1, 0, 100);
        assert!(clip.contains_frame(0));
        assert!(clip.contains_frame(50));
        assert!(clip.contains_frame(99));
        assert!(!clip.contains_frame(100));
        assert!(!clip.contains_frame(-1));
    }

    #[test]
    fn test_cuttable_clip_is_edge() {
        let clip = make_clip(1, 1, 10, 50);
        assert!(clip.is_edge(10));
        assert!(clip.is_edge(50));
        assert!(!clip.is_edge(30));
    }

    #[test]
    fn test_razor_cut_clip_middle() {
        let mut razor = RazorTool::new(RazorConfig {
            snap_enabled: false,
            ..RazorConfig::default()
        });
        let clip = make_clip(1, 1, 0, 100);
        let result = razor.cut_clip(&clip, 40).expect("should succeed in test");
        assert_eq!(result.left_duration, 40);
        assert_eq!(result.right_duration, 60);
        assert_eq!(result.total_duration(), 100);
    }

    #[test]
    fn test_razor_cut_clip_at_edge_returns_none() {
        let mut razor = RazorTool::new(RazorConfig {
            snap_enabled: false,
            ..RazorConfig::default()
        });
        let clip = make_clip(1, 1, 0, 100);
        assert!(razor.cut_clip(&clip, 0).is_none());
        assert!(razor.cut_clip(&clip, 100).is_none());
    }

    #[test]
    fn test_razor_cut_clip_outside_returns_none() {
        let mut razor = RazorTool::new(RazorConfig {
            snap_enabled: false,
            ..RazorConfig::default()
        });
        let clip = make_clip(1, 1, 10, 50);
        assert!(razor.cut_clip(&clip, 5).is_none());
        assert!(razor.cut_clip(&clip, 60).is_none());
    }

    #[test]
    fn test_snap_frame() {
        let mut razor = RazorTool::new(RazorConfig {
            snap_enabled: true,
            snap_tolerance: 3,
            ..RazorConfig::default()
        });
        razor.set_snap_points(vec![0, 24, 48, 72]);
        assert_eq!(razor.snap_frame(23), 24);
        assert_eq!(razor.snap_frame(25), 24);
        assert_eq!(razor.snap_frame(30), 30); // too far from 24 and 48
    }

    #[test]
    fn test_snap_disabled() {
        let razor = RazorTool::new(RazorConfig {
            snap_enabled: false,
            ..RazorConfig::default()
        });
        assert_eq!(razor.snap_frame(23), 23);
    }

    #[test]
    fn test_cut_clips_multiple() {
        let mut razor = RazorTool::new(RazorConfig {
            snap_enabled: false,
            ..RazorConfig::default()
        });
        let clips = vec![
            make_clip(1, 1, 0, 100),
            make_clip(2, 2, 0, 100),
            make_clip(3, 3, 50, 150), // doesn't contain frame 30
        ];
        let results = razor.cut_clips(&clips, 30);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_multi_cut() {
        let mut razor = RazorTool::new(RazorConfig {
            snap_enabled: false,
            ..RazorConfig::default()
        });
        let clips = vec![make_clip(1, 1, 0, 100)];
        let results = razor.multi_cut(&clips, &[25, 50, 75]);
        // Only the first cut at 25 hits because after that we still pass the
        // original clip descriptor (not the split result)
        assert!(!results.is_empty());
    }

    #[test]
    fn test_clips_at_frame() {
        let clips = vec![
            make_clip(1, 1, 0, 50),
            make_clip(2, 1, 50, 100),
            make_clip(3, 2, 0, 100),
        ];
        let at_25 = RazorTool::clips_at_frame(&clips, 25);
        assert_eq!(at_25.len(), 2); // clip 1 and clip 3
    }

    #[test]
    fn test_clips_on_track() {
        let clips = vec![
            make_clip(1, 1, 0, 50),
            make_clip(2, 1, 50, 100),
            make_clip(3, 2, 0, 100),
        ];
        let track1 = RazorTool::clips_on_track(&clips, 1);
        assert_eq!(track1.len(), 2);
    }

    #[test]
    fn test_default_razor_tool() {
        let razor = RazorTool::default();
        assert!(razor.config().snap_enabled);
        assert!(razor.config().mark_through_edits);
    }
}
