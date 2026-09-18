#![allow(dead_code)]
//! Blade/razor tool for cutting clips at specific frame positions.

/// Controls which tracks the blade tool affects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BladeMode {
    /// Cuts only the clip under the playhead on the active track.
    Single,
    /// Cuts all clips across every track at the playhead position.
    AllTracks,
    /// Cuts a clip and all clips linked to it (audio/video pairs).
    Linked,
}

impl BladeMode {
    /// Returns `true` if this mode cuts clips on every track.
    #[must_use]
    pub fn cuts_all(&self) -> bool {
        matches!(self, BladeMode::AllTracks)
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            BladeMode::Single => "Single",
            BladeMode::AllTracks => "All Tracks",
            BladeMode::Linked => "Linked",
        }
    }
}

/// Describes where a single blade cut will occur.
#[derive(Debug, Clone)]
pub struct BladeCut {
    /// Track index to cut on.
    pub track_index: usize,
    /// Clip id to cut.
    pub clip_id: u64,
    /// Frame position within the timeline at which to cut.
    pub cut_frame: i64,
}

impl BladeCut {
    /// Creates a new blade cut descriptor.
    #[must_use]
    pub fn new(track_index: usize, clip_id: u64, cut_frame: i64) -> Self {
        Self {
            track_index,
            clip_id,
            cut_frame,
        }
    }

    /// Returns a cut snapped to the nearest frame boundary (integer).
    /// In practice frames are already integers; this enforces the invariant.
    #[must_use]
    pub fn at_frame(mut self, frame: i64) -> Self {
        self.cut_frame = frame;
        self
    }
}

/// The outcome produced after applying the blade tool.
#[derive(Debug, Clone)]
pub struct BladeResult {
    /// All cuts that were applied during this operation.
    pub cuts: Vec<BladeCut>,
    /// Number of new clip segments created (= `cuts.len()` for single cuts each split = 1 new).
    pub new_segments: usize,
}

impl BladeResult {
    /// Creates a blade result from a list of applied cuts.
    #[must_use]
    pub fn new(cuts: Vec<BladeCut>) -> Self {
        let new_segments = cuts.len();
        Self { cuts, new_segments }
    }

    /// Number of cuts applied.
    #[must_use]
    pub fn cuts_applied(&self) -> usize {
        self.cuts.len()
    }
}

/// The blade/razor tool implementation.
#[derive(Debug, Clone)]
pub struct BladeTool {
    /// Current operating mode.
    pub mode: BladeMode,
    /// Snap threshold in frames; 0 disables snapping.
    pub snap_threshold: u32,
}

impl BladeTool {
    /// Creates a blade tool with the given mode and snap threshold.
    #[must_use]
    pub fn new(mode: BladeMode, snap_threshold: u32) -> Self {
        Self {
            mode,
            snap_threshold,
        }
    }

    /// Performs a cut on the given set of (`track_index`, `clip_id`, `clip_start`, `clip_end`) tuples
    /// at `frame`, returning a `BladeResult`.
    ///
    /// Only clips whose range `[clip_start, clip_end)` contains `frame` receive a cut.
    #[must_use]
    pub fn cut(&self, clips: &[(usize, u64, i64, i64)], frame: i64) -> BladeResult {
        let cuts: Vec<BladeCut> = clips
            .iter()
            .filter(|(_, _, start, end)| frame > *start && frame < *end)
            .map(|(track, id, _, _)| BladeCut::new(*track, *id, frame))
            .collect();
        BladeResult::new(cuts)
    }

    /// Returns the cuts that *would* be applied without actually applying them.
    #[must_use]
    pub fn preview_cut(&self, clips: &[(usize, u64, i64, i64)], frame: i64) -> Vec<BladeCut> {
        self.cut(clips, frame).cuts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_not_all() {
        assert!(!BladeMode::Single.cuts_all());
    }

    #[test]
    fn test_all_tracks_cuts_all() {
        assert!(BladeMode::AllTracks.cuts_all());
    }

    #[test]
    fn test_linked_not_all() {
        assert!(!BladeMode::Linked.cuts_all());
    }

    #[test]
    fn test_blade_mode_labels() {
        assert_eq!(BladeMode::Single.label(), "Single");
        assert_eq!(BladeMode::AllTracks.label(), "All Tracks");
        assert_eq!(BladeMode::Linked.label(), "Linked");
    }

    #[test]
    fn test_blade_cut_at_frame() {
        let cut = BladeCut::new(0, 1, 50).at_frame(75);
        assert_eq!(cut.cut_frame, 75);
    }

    #[test]
    fn test_blade_result_cuts_applied() {
        let cuts = vec![BladeCut::new(0, 1, 50), BladeCut::new(1, 2, 50)];
        let result = BladeResult::new(cuts);
        assert_eq!(result.cuts_applied(), 2);
        assert_eq!(result.new_segments, 2);
    }

    #[test]
    fn test_blade_tool_cut_single_clip() {
        let tool = BladeTool::new(BladeMode::Single, 2);
        let clips = vec![(0usize, 1u64, 0i64, 100i64)];
        let result = tool.cut(&clips, 50);
        assert_eq!(result.cuts_applied(), 1);
        assert_eq!(result.cuts[0].cut_frame, 50);
    }

    #[test]
    fn test_blade_tool_no_cut_outside_range() {
        let tool = BladeTool::new(BladeMode::Single, 0);
        let clips = vec![(0usize, 1u64, 0i64, 100i64)];
        // Frame 150 is outside the clip
        let result = tool.cut(&clips, 150);
        assert_eq!(result.cuts_applied(), 0);
    }

    #[test]
    fn test_blade_tool_no_cut_at_boundary() {
        let tool = BladeTool::new(BladeMode::Single, 0);
        let clips = vec![(0usize, 1u64, 0i64, 100i64)];
        // Cutting exactly at start (0) should NOT cut (frame must be strictly inside)
        let result = tool.cut(&clips, 0);
        assert_eq!(result.cuts_applied(), 0);
    }

    #[test]
    fn test_blade_tool_all_tracks_cut_multiple() {
        let tool = BladeTool::new(BladeMode::AllTracks, 0);
        let clips = vec![
            (0usize, 10u64, 0i64, 200i64),
            (1usize, 20u64, 0i64, 200i64),
            (2usize, 30u64, 50i64, 150i64),
        ];
        let result = tool.cut(&clips, 100);
        assert_eq!(result.cuts_applied(), 3);
    }

    #[test]
    fn test_preview_cut_matches_cut() {
        let tool = BladeTool::new(BladeMode::Single, 0);
        let clips = vec![(0usize, 5u64, 10i64, 90i64)];
        let preview = tool.preview_cut(&clips, 40);
        let actual = tool.cut(&clips, 40).cuts;
        assert_eq!(preview.len(), actual.len());
    }

    #[test]
    fn test_blade_result_empty() {
        let result = BladeResult::new(vec![]);
        assert_eq!(result.cuts_applied(), 0);
    }

    #[test]
    fn test_blade_tool_default_fields() {
        let tool = BladeTool::new(BladeMode::Linked, 5);
        assert_eq!(tool.snap_threshold, 5);
        assert_eq!(tool.mode, BladeMode::Linked);
    }
}
