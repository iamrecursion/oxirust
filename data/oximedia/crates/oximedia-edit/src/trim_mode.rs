//! Trim mode and dual-roller trim operations.
//!
//! Provides types for single-roller and dual-roller (ripple) trimming of clips
//! on a timeline.

#![allow(dead_code)]

/// Which end of a clip is being trimmed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimSide {
    /// Trim the in-point (left/start edge).
    In,
    /// Trim the out-point (right/end edge).
    Out,
    /// Trim both edges simultaneously (e.g., slip within a dual-roller trim).
    Both,
}

impl TrimSide {
    /// Returns `true` when trimming the in-point.
    #[must_use]
    pub fn is_in_point(self) -> bool {
        matches!(self, TrimSide::In | TrimSide::Both)
    }

    /// Returns `true` when trimming the out-point.
    #[must_use]
    pub fn is_out_point(self) -> bool {
        matches!(self, TrimSide::Out | TrimSide::Both)
    }
}

/// A single-roller trim handle attached to one clip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrimRoller {
    /// Clip being trimmed.
    pub clip_id: u64,
    /// Which edge is being trimmed.
    pub side: TrimSide,
    /// Current frame position of the trim point.
    pub current_frame: i64,
}

impl TrimRoller {
    /// Create a new trim roller.
    #[must_use]
    pub fn new(clip_id: u64, side: TrimSide, frame: i64) -> Self {
        Self {
            clip_id,
            side,
            current_frame: frame,
        }
    }

    /// Return a new roller offset by `delta` frames.
    #[must_use]
    pub fn offset_by(&self, delta: i64) -> Self {
        Self {
            clip_id: self.clip_id,
            side: self.side,
            current_frame: self.current_frame + delta,
        }
    }
}

/// A dual-roller trim simultaneously adjusts the out-point of one clip and
/// the in-point of the adjacent clip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DualRollerTrim {
    /// The outgoing (earlier) clip.
    pub outgoing_clip: u64,
    /// The incoming (later) clip.
    pub incoming_clip: u64,
    /// Current transition frame (shared edit point).
    pub frame: i64,
}

impl DualRollerTrim {
    /// Create a new dual-roller trim.
    #[must_use]
    pub fn new(outgoing_clip: u64, incoming_clip: u64, frame: i64) -> Self {
        Self {
            outgoing_clip,
            incoming_clip,
            frame,
        }
    }

    /// Return a new dual-roller trim shifted by `delta` frames.
    #[must_use]
    pub fn shift(&self, delta: i64) -> Self {
        Self {
            outgoing_clip: self.outgoing_clip,
            incoming_clip: self.incoming_clip,
            frame: self.frame + delta,
        }
    }

    /// Return the current shared transition frame.
    #[must_use]
    pub fn get_transition_point(&self) -> i64 {
        self.frame
    }
}

/// Constraints applied to trimming operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrimConstraints {
    /// Minimum allowed clip duration in frames.
    pub min_clip_duration_frames: u32,
    /// Snap tolerance in frames (trim snaps to nearby markers/cuts within this distance).
    pub snap_tolerance_frames: u32,
}

impl Default for TrimConstraints {
    fn default() -> Self {
        Self {
            min_clip_duration_frames: 1,
            snap_tolerance_frames: 5,
        }
    }
}

impl TrimConstraints {
    /// Returns `true` when `new_duration` satisfies the minimum duration constraint.
    #[must_use]
    pub fn validate_trim_result(&self, new_duration: i64) -> bool {
        new_duration >= i64::from(self.min_clip_duration_frames)
    }
}

/// Which edge of a clip is being trimmed (`InPoint` or `OutPoint`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimEdge {
    /// The start / in-point of the clip.
    InPoint,
    /// The end / out-point of the clip.
    OutPoint,
}

impl TrimEdge {
    /// Returns the opposite edge.
    #[must_use]
    pub fn opposite(self) -> Self {
        match self {
            TrimEdge::InPoint => TrimEdge::OutPoint,
            TrimEdge::OutPoint => TrimEdge::InPoint,
        }
    }
}

/// The trim mode governing how adjacent clips are affected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimMode {
    /// Ripple: shift all downstream clips by the trim delta.
    Ripple,
    /// Roll: simultaneously trim the outgoing and incoming clip (dual-roller).
    Roll,
    /// Slip: shift the source in/out without moving the clip on the timeline.
    Slip,
    /// Slide: move the clip, trimming adjacent clips to fill the gap.
    Slide,
}

impl TrimMode {
    /// Human-readable description.
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            TrimMode::Ripple => "Ripple: downstream clips shift by the trim amount",
            TrimMode::Roll => "Roll: adjacent clips absorb the change at the edit point",
            TrimMode::Slip => "Slip: source content shifts without moving the clip",
            TrimMode::Slide => "Slide: clip moves; neighbours are trimmed to compensate",
        }
    }

    /// Returns `true` when this mode directly modifies neighbouring clips.
    #[must_use]
    pub fn affects_neighbors(&self) -> bool {
        matches!(self, TrimMode::Roll | TrimMode::Slide)
    }
}

/// A single trim operation applied to a clip edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrimOperation {
    /// Clip being trimmed.
    pub clip_id: u64,
    /// Which edge is being trimmed.
    pub edge: TrimEdge,
    /// Trim mode.
    pub mode: TrimMode,
    /// Number of frames to move the edge (positive = later, negative = earlier).
    pub delta_frames: i64,
}

impl TrimOperation {
    /// Create a new trim operation.
    #[must_use]
    pub fn new(clip_id: u64, edge: TrimEdge, mode: TrimMode, delta_frames: i64) -> Self {
        Self {
            clip_id,
            edge,
            mode,
            delta_frames,
        }
    }

    /// Returns `true` when the operation extends the clip (makes it longer).
    #[must_use]
    pub fn is_extend(&self) -> bool {
        match self.edge {
            // Moving the in-point earlier extends the clip.
            TrimEdge::InPoint => self.delta_frames < 0,
            // Moving the out-point later extends the clip.
            TrimEdge::OutPoint => self.delta_frames > 0,
        }
    }

    /// Returns `true` when the operation shrinks the clip (makes it shorter).
    #[must_use]
    pub fn is_shrink(&self) -> bool {
        match self.edge {
            TrimEdge::InPoint => self.delta_frames > 0,
            TrimEdge::OutPoint => self.delta_frames < 0,
        }
    }
}

/// Clamp-only constraints for trim operations.
///
/// Note: `min_duration_frames` and `max_duration_frames` are separate from
/// `TrimConstraints` (which retains the existing snap-tolerance fields).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrimDurationConstraints {
    /// Minimum clip duration in frames.
    pub min_duration_frames: u32,
    /// Maximum clip duration in frames.
    pub max_duration_frames: u32,
    /// When `true`, snap to keyframe boundaries.
    pub snap_to_keyframe: bool,
}

impl Default for TrimDurationConstraints {
    fn default() -> Self {
        Self {
            min_duration_frames: 1,
            max_duration_frames: u32::MAX,
            snap_to_keyframe: false,
        }
    }
}

impl TrimDurationConstraints {
    /// Clamp `delta` so that `current_duration + delta` stays within
    /// `[min_duration_frames, max_duration_frames]`.
    #[must_use]
    pub fn clamp_delta(&self, current_duration: u64, delta: i64) -> i64 {
        let min = i64::from(self.min_duration_frames);
        let max = i64::from(self.max_duration_frames);
        let raw = current_duration as i64 + delta;
        let clamped = raw.clamp(min, max);
        clamped - current_duration as i64
    }
}

/// An undo/redo history of trim operations.
#[derive(Debug, Clone)]
pub struct TrimHistory {
    /// Committed operations (oldest first).
    operations: Vec<TrimOperation>,
    /// Undone operations (most-recently-undone first).
    redo_stack: Vec<TrimOperation>,
    /// Maximum number of undoable operations to keep.
    pub max_undos: usize,
}

impl TrimHistory {
    /// Create a new history with the given undo limit.
    #[must_use]
    pub fn new(max_undos: usize) -> Self {
        Self {
            operations: Vec::new(),
            redo_stack: Vec::new(),
            max_undos,
        }
    }

    /// Push a new operation, clearing the redo stack and enforcing the undo cap.
    pub fn push(&mut self, op: TrimOperation) {
        self.redo_stack.clear();
        self.operations.push(op);
        if self.operations.len() > self.max_undos {
            self.operations.remove(0);
        }
    }

    /// Undo the most recent operation; returns it if available.
    pub fn undo(&mut self) -> Option<TrimOperation> {
        let op = self.operations.pop()?;
        self.redo_stack.push(op.clone());
        Some(op)
    }

    /// Redo the most recently undone operation; returns it if available.
    pub fn redo(&mut self) -> Option<TrimOperation> {
        let op = self.redo_stack.pop()?;
        self.operations.push(op.clone());
        Some(op)
    }

    /// Returns `true` when there is at least one operation to undo.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        !self.operations.is_empty()
    }

    /// Returns `true` when there is at least one operation to redo.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- TrimSide tests -----

    #[test]
    fn test_trim_side_in_is_in_point() {
        assert!(TrimSide::In.is_in_point());
        assert!(!TrimSide::In.is_out_point());
    }

    #[test]
    fn test_trim_side_out_is_out_point() {
        assert!(TrimSide::Out.is_out_point());
        assert!(!TrimSide::Out.is_in_point());
    }

    #[test]
    fn test_trim_side_both() {
        assert!(TrimSide::Both.is_in_point());
        assert!(TrimSide::Both.is_out_point());
    }

    // ----- TrimRoller tests -----

    #[test]
    fn test_trim_roller_new() {
        let roller = TrimRoller::new(5, TrimSide::In, 100);
        assert_eq!(roller.clip_id, 5);
        assert_eq!(roller.side, TrimSide::In);
        assert_eq!(roller.current_frame, 100);
    }

    #[test]
    fn test_trim_roller_offset_positive() {
        let roller = TrimRoller::new(5, TrimSide::Out, 200);
        let shifted = roller.offset_by(30);
        assert_eq!(shifted.current_frame, 230);
        assert_eq!(shifted.clip_id, 5);
    }

    #[test]
    fn test_trim_roller_offset_negative() {
        let roller = TrimRoller::new(5, TrimSide::Out, 200);
        let shifted = roller.offset_by(-50);
        assert_eq!(shifted.current_frame, 150);
    }

    #[test]
    fn test_trim_roller_offset_zero() {
        let roller = TrimRoller::new(5, TrimSide::In, 200);
        let shifted = roller.offset_by(0);
        assert_eq!(shifted.current_frame, 200);
    }

    // ----- DualRollerTrim tests -----

    #[test]
    fn test_dual_roller_new() {
        let trim = DualRollerTrim::new(1, 2, 300);
        assert_eq!(trim.outgoing_clip, 1);
        assert_eq!(trim.incoming_clip, 2);
        assert_eq!(trim.frame, 300);
    }

    #[test]
    fn test_dual_roller_shift_forward() {
        let trim = DualRollerTrim::new(1, 2, 300);
        let shifted = trim.shift(10);
        assert_eq!(shifted.get_transition_point(), 310);
    }

    #[test]
    fn test_dual_roller_shift_backward() {
        let trim = DualRollerTrim::new(1, 2, 300);
        let shifted = trim.shift(-20);
        assert_eq!(shifted.get_transition_point(), 280);
    }

    #[test]
    fn test_dual_roller_get_transition_point() {
        let trim = DualRollerTrim::new(3, 4, 500);
        assert_eq!(trim.get_transition_point(), 500);
    }

    // ----- TrimConstraints tests -----

    #[test]
    fn test_trim_constraints_default() {
        let c = TrimConstraints::default();
        assert_eq!(c.min_clip_duration_frames, 1);
        assert_eq!(c.snap_tolerance_frames, 5);
    }

    #[test]
    fn test_trim_constraints_validate_valid() {
        let c = TrimConstraints::default();
        assert!(c.validate_trim_result(10));
    }

    #[test]
    fn test_trim_constraints_validate_minimum() {
        let c = TrimConstraints::default();
        assert!(c.validate_trim_result(1));
    }

    #[test]
    fn test_trim_constraints_validate_zero_duration() {
        let c = TrimConstraints::default();
        assert!(!c.validate_trim_result(0));
    }

    #[test]
    fn test_trim_constraints_validate_negative_duration() {
        let c = TrimConstraints::default();
        assert!(!c.validate_trim_result(-1));
    }

    // ----- TrimEdge tests -----

    #[test]
    fn test_trim_edge_opposite_in_point() {
        assert_eq!(TrimEdge::InPoint.opposite(), TrimEdge::OutPoint);
    }

    #[test]
    fn test_trim_edge_opposite_out_point() {
        assert_eq!(TrimEdge::OutPoint.opposite(), TrimEdge::InPoint);
    }

    // ----- TrimMode tests -----

    #[test]
    fn test_trim_mode_description_not_empty() {
        for mode in [
            TrimMode::Ripple,
            TrimMode::Roll,
            TrimMode::Slip,
            TrimMode::Slide,
        ] {
            assert!(!mode.description().is_empty());
        }
    }

    #[test]
    fn test_trim_mode_affects_neighbors() {
        assert!(!TrimMode::Ripple.affects_neighbors());
        assert!(TrimMode::Roll.affects_neighbors());
        assert!(!TrimMode::Slip.affects_neighbors());
        assert!(TrimMode::Slide.affects_neighbors());
    }

    // ----- TrimOperation tests -----

    #[test]
    fn test_trim_operation_is_extend_out_point_positive() {
        let op = TrimOperation::new(1, TrimEdge::OutPoint, TrimMode::Ripple, 10);
        assert!(op.is_extend());
        assert!(!op.is_shrink());
    }

    #[test]
    fn test_trim_operation_is_shrink_out_point_negative() {
        let op = TrimOperation::new(1, TrimEdge::OutPoint, TrimMode::Ripple, -10);
        assert!(op.is_shrink());
        assert!(!op.is_extend());
    }

    #[test]
    fn test_trim_operation_is_extend_in_point_negative() {
        let op = TrimOperation::new(1, TrimEdge::InPoint, TrimMode::Roll, -5);
        assert!(op.is_extend());
        assert!(!op.is_shrink());
    }

    #[test]
    fn test_trim_operation_is_shrink_in_point_positive() {
        let op = TrimOperation::new(1, TrimEdge::InPoint, TrimMode::Roll, 5);
        assert!(op.is_shrink());
        assert!(!op.is_extend());
    }

    #[test]
    fn test_trim_operation_zero_delta_neither() {
        let op = TrimOperation::new(1, TrimEdge::OutPoint, TrimMode::Slip, 0);
        assert!(!op.is_extend());
        assert!(!op.is_shrink());
    }

    // ----- TrimDurationConstraints tests -----

    #[test]
    fn test_trim_duration_constraints_clamp_within_bounds() {
        let c = TrimDurationConstraints {
            min_duration_frames: 5,
            max_duration_frames: 100,
            snap_to_keyframe: false,
        };
        // current 50, delta +10 -> 60, within [5,100]
        assert_eq!(c.clamp_delta(50, 10), 10);
    }

    #[test]
    fn test_trim_duration_constraints_clamp_to_min() {
        let c = TrimDurationConstraints {
            min_duration_frames: 10,
            max_duration_frames: 200,
            snap_to_keyframe: false,
        };
        // current 15, delta -20 -> would be -5, clamp to 10 -> delta = -5
        assert_eq!(c.clamp_delta(15, -20), -5);
    }

    #[test]
    fn test_trim_duration_constraints_clamp_to_max() {
        let c = TrimDurationConstraints {
            min_duration_frames: 1,
            max_duration_frames: 30,
            snap_to_keyframe: false,
        };
        // current 25, delta +20 -> would be 45, clamp to 30 -> delta = 5
        assert_eq!(c.clamp_delta(25, 20), 5);
    }

    // ----- TrimHistory tests -----

    #[test]
    fn test_trim_history_can_undo_after_push() {
        let mut h = TrimHistory::new(10);
        assert!(!h.can_undo());
        h.push(TrimOperation::new(
            1,
            TrimEdge::OutPoint,
            TrimMode::Ripple,
            5,
        ));
        assert!(h.can_undo());
    }

    #[test]
    fn test_trim_history_undo_returns_op() {
        let mut h = TrimHistory::new(10);
        let op = TrimOperation::new(7, TrimEdge::InPoint, TrimMode::Roll, -3);
        h.push(op.clone());
        assert_eq!(h.undo(), Some(op));
    }

    #[test]
    fn test_trim_history_redo_after_undo() {
        let mut h = TrimHistory::new(10);
        let op = TrimOperation::new(2, TrimEdge::OutPoint, TrimMode::Slide, 8);
        h.push(op.clone());
        h.undo();
        assert!(h.can_redo());
        assert_eq!(h.redo(), Some(op));
    }

    #[test]
    fn test_trim_history_push_clears_redo() {
        let mut h = TrimHistory::new(10);
        h.push(TrimOperation::new(
            1,
            TrimEdge::OutPoint,
            TrimMode::Ripple,
            1,
        ));
        h.undo();
        assert!(h.can_redo());
        h.push(TrimOperation::new(
            2,
            TrimEdge::OutPoint,
            TrimMode::Ripple,
            2,
        ));
        assert!(!h.can_redo());
    }

    #[test]
    fn test_trim_history_max_undos_enforced() {
        let mut h = TrimHistory::new(3);
        for i in 0..5u64 {
            h.push(TrimOperation::new(
                i,
                TrimEdge::OutPoint,
                TrimMode::Ripple,
                1,
            ));
        }
        // Only 3 operations retained.
        assert_eq!(h.operations.len(), 3);
    }
}
