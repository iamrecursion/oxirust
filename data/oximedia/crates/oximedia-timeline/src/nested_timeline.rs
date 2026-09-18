//! Nested / compound clip support for the timeline.
//!
//! This module models clips that reference other timelines (nested sequences),
//! allowing hierarchical composition of timelines.

#![allow(dead_code)]

/// Nesting depth of a clip in the timeline hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NestDepth {
    /// Top-level timeline clip.
    Root,
    /// One level of nesting.
    One,
    /// Two levels of nesting.
    Two,
    /// Three levels of nesting (maximum).
    Three,
}

impl NestDepth {
    /// Returns the numeric nesting level (0 = Root, 1 = One, …).
    #[must_use]
    pub fn level(self) -> u8 {
        match self {
            Self::Root => 0,
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
        }
    }

    /// Returns `true` if this depth can be nested one level deeper.
    ///
    /// Returns `false` for `Three` (maximum depth).
    #[must_use]
    pub fn can_nest_deeper(self) -> bool {
        !matches!(self, Self::Three)
    }
}

/// A clip that references (nests) another timeline.
#[derive(Debug, Clone)]
pub struct NestedClip {
    /// Unique identifier for this clip.
    pub id: u64,
    /// Human-readable name.
    pub name: String,
    /// ID of the nested timeline that this clip references.
    pub timeline_id: u64,
    /// First frame of the clip on the parent timeline (inclusive).
    pub in_point: u64,
    /// Last frame of the clip on the parent timeline (exclusive).
    pub out_point: u64,
    /// Nesting depth of this clip.
    pub depth: NestDepth,
}

impl NestedClip {
    /// Creates a new nested clip.
    #[must_use]
    pub fn new(
        id: u64,
        name: impl Into<String>,
        timeline_id: u64,
        in_point: u64,
        out_point: u64,
        depth: NestDepth,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            timeline_id,
            in_point,
            out_point,
            depth,
        }
    }

    /// Returns the number of frames spanned by this clip.
    ///
    /// Returns `0` if `out_point <= in_point`.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.out_point.saturating_sub(self.in_point)
    }

    /// Returns `true` if this clip is at the root level.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.depth == NestDepth::Root
    }
}

/// A nested timeline: a root clip together with its direct child clips.
#[derive(Debug, Clone)]
pub struct NestedTimeline {
    /// The root (outermost) clip that owns this timeline.
    pub root_clip: NestedClip,
    /// Direct children of the root clip, in insertion order.
    pub children: Vec<NestedClip>,
}

impl NestedTimeline {
    /// Creates a new nested timeline with the given root clip and no children.
    #[must_use]
    pub fn new(root_clip: NestedClip) -> Self {
        Self {
            root_clip,
            children: Vec::new(),
        }
    }

    /// Adds a child clip to this timeline.
    pub fn add_child(&mut self, clip: NestedClip) {
        self.children.push(clip);
    }

    /// Returns IDs of all clips (root first, then children) whose depth level
    /// is ≤ `depth_limit`, in insertion order.
    #[must_use]
    pub fn flatten(&self, depth_limit: u8) -> Vec<u64> {
        let mut ids = Vec::new();
        if self.root_clip.depth.level() <= depth_limit {
            ids.push(self.root_clip.id);
        }
        for child in &self.children {
            if child.depth.level() <= depth_limit {
                ids.push(child.id);
            }
        }
        ids
    }

    /// Returns the number of direct children (not counting the root).
    #[must_use]
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Returns the maximum nesting level found across the root and all children.
    #[must_use]
    pub fn max_depth(&self) -> u8 {
        let root_level = self.root_clip.depth.level();
        self.children
            .iter()
            .map(|c| c.depth.level())
            .fold(root_level, std::cmp::Ord::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- NestDepth tests ---

    #[test]
    fn test_nest_depth_levels() {
        assert_eq!(NestDepth::Root.level(), 0);
        assert_eq!(NestDepth::One.level(), 1);
        assert_eq!(NestDepth::Two.level(), 2);
        assert_eq!(NestDepth::Three.level(), 3);
    }

    #[test]
    fn test_can_nest_deeper_root() {
        assert!(NestDepth::Root.can_nest_deeper());
    }

    #[test]
    fn test_can_nest_deeper_one() {
        assert!(NestDepth::One.can_nest_deeper());
    }

    #[test]
    fn test_can_nest_deeper_two() {
        assert!(NestDepth::Two.can_nest_deeper());
    }

    #[test]
    fn test_can_nest_deeper_three_false() {
        assert!(!NestDepth::Three.can_nest_deeper());
    }

    // --- NestedClip tests ---

    #[test]
    fn test_nested_clip_duration() {
        let clip = NestedClip::new(1, "A", 10, 10, 60, NestDepth::Root);
        assert_eq!(clip.duration_frames(), 50);
    }

    #[test]
    fn test_nested_clip_duration_inverted() {
        let clip = NestedClip::new(1, "A", 10, 100, 50, NestDepth::Root);
        assert_eq!(clip.duration_frames(), 0);
    }

    #[test]
    fn test_nested_clip_is_root_true() {
        let clip = NestedClip::new(1, "A", 10, 0, 100, NestDepth::Root);
        assert!(clip.is_root());
    }

    #[test]
    fn test_nested_clip_is_root_false() {
        let clip = NestedClip::new(1, "A", 10, 0, 100, NestDepth::One);
        assert!(!clip.is_root());
    }

    // --- NestedTimeline tests ---

    fn make_root() -> NestedClip {
        NestedClip::new(100, "Root", 99, 0, 240, NestDepth::Root)
    }

    #[test]
    fn test_child_count_empty() {
        let nt = NestedTimeline::new(make_root());
        assert_eq!(nt.child_count(), 0);
    }

    #[test]
    fn test_add_child_increases_count() {
        let mut nt = NestedTimeline::new(make_root());
        nt.add_child(NestedClip::new(1, "Child1", 1, 0, 100, NestDepth::One));
        assert_eq!(nt.child_count(), 1);
    }

    #[test]
    fn test_flatten_no_limit() {
        let mut nt = NestedTimeline::new(make_root());
        nt.add_child(NestedClip::new(1, "C1", 1, 0, 100, NestDepth::One));
        nt.add_child(NestedClip::new(2, "C2", 2, 100, 200, NestDepth::Two));
        let ids = nt.flatten(3);
        assert_eq!(ids, vec![100, 1, 2]);
    }

    #[test]
    fn test_flatten_with_depth_limit() {
        let mut nt = NestedTimeline::new(make_root());
        nt.add_child(NestedClip::new(1, "C1", 1, 0, 100, NestDepth::One));
        nt.add_child(NestedClip::new(2, "C2", 2, 100, 200, NestDepth::Two));
        // Limit = 1: root (0) and C1 (1) pass; C2 (2) is excluded
        let ids = nt.flatten(1);
        assert_eq!(ids, vec![100, 1]);
    }

    #[test]
    fn test_flatten_root_only() {
        let nt = NestedTimeline::new(make_root());
        let ids = nt.flatten(0);
        assert_eq!(ids, vec![100]);
    }

    #[test]
    fn test_max_depth_no_children() {
        let nt = NestedTimeline::new(make_root());
        assert_eq!(nt.max_depth(), 0);
    }

    #[test]
    fn test_max_depth_with_children() {
        let mut nt = NestedTimeline::new(make_root());
        nt.add_child(NestedClip::new(1, "C1", 1, 0, 100, NestDepth::One));
        nt.add_child(NestedClip::new(2, "C2", 2, 100, 200, NestDepth::Three));
        assert_eq!(nt.max_depth(), 3);
    }

    #[test]
    fn test_flatten_returns_insertion_order() {
        let mut nt = NestedTimeline::new(make_root());
        nt.add_child(NestedClip::new(10, "X", 5, 0, 50, NestDepth::One));
        nt.add_child(NestedClip::new(20, "Y", 5, 50, 100, NestDepth::One));
        let ids = nt.flatten(1);
        assert_eq!(ids[0], 100); // root first
        assert_eq!(ids[1], 10);
        assert_eq!(ids[2], 20);
    }
}
