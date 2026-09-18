//! Per-clip color correction layer for the timeline.
//!
//! This module models color correction nodes scoped to clips, frame ranges,
//! or the entire timeline, and provides a layer for managing them.

#![allow(dead_code)]

/// Scope of a color correction node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectionScope {
    /// Applied only to a specific clip.
    Clip,
    /// Applied over a frame range (not necessarily tied to a clip).
    Range,
    /// Applied globally across the whole timeline.
    Global,
}

impl CorrectionScope {
    /// Returns `true` for `Clip` and `Range` scopes (local, not global).
    #[must_use]
    pub fn is_local(self) -> bool {
        matches!(self, Self::Clip | Self::Range)
    }
}

/// A single color correction node attached to a region on the timeline.
#[derive(Debug, Clone)]
pub struct ColorCorrectionNode {
    /// ID of the clip this node is associated with (0 for global/range nodes).
    pub clip_id: u64,
    /// Scope of this color correction.
    pub scope: CorrectionScope,
    /// First frame of the correction range (inclusive).
    pub in_point: u64,
    /// Last frame of the correction range (exclusive).
    pub out_point: u64,
    /// Reference ID to the CDL/LUT grade stored elsewhere.
    pub grade_id: String,
}

impl ColorCorrectionNode {
    /// Creates a new color correction node.
    #[must_use]
    pub fn new(
        clip_id: u64,
        scope: CorrectionScope,
        in_point: u64,
        out_point: u64,
        grade_id: impl Into<String>,
    ) -> Self {
        Self {
            clip_id,
            scope,
            in_point,
            out_point,
            grade_id: grade_id.into(),
        }
    }

    /// Returns the number of frames spanned by this node.
    ///
    /// Returns `0` if `out_point <= in_point`.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.out_point.saturating_sub(self.in_point)
    }

    /// Returns `true` if `frame` falls within `[in_point, out_point)`.
    #[must_use]
    pub fn contains_frame(&self, frame: u64) -> bool {
        frame >= self.in_point && frame < self.out_point
    }
}

/// An ordered collection of `ColorCorrectionNode`s forming a correction layer.
#[derive(Debug, Clone, Default)]
pub struct ColorCorrectionLayer {
    /// All correction nodes in insertion order.
    pub nodes: Vec<ColorCorrectionNode>,
}

impl ColorCorrectionLayer {
    /// Creates a new empty layer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a node to the layer.
    pub fn add(&mut self, node: ColorCorrectionNode) {
        self.nodes.push(node);
    }

    /// Returns references to all nodes that cover `frame`.
    #[must_use]
    pub fn find_for_frame(&self, frame: u64) -> Vec<&ColorCorrectionNode> {
        self.nodes
            .iter()
            .filter(|n| n.contains_frame(frame))
            .collect()
    }

    /// Removes all nodes associated with `clip_id`.
    pub fn remove_by_clip(&mut self, clip_id: u64) {
        self.nodes.retain(|n| n.clip_id != clip_id);
    }

    /// Returns the total number of nodes in the layer.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- CorrectionScope tests ---

    #[test]
    fn test_scope_clip_is_local() {
        assert!(CorrectionScope::Clip.is_local());
    }

    #[test]
    fn test_scope_range_is_local() {
        assert!(CorrectionScope::Range.is_local());
    }

    #[test]
    fn test_scope_global_is_not_local() {
        assert!(!CorrectionScope::Global.is_local());
    }

    // --- ColorCorrectionNode tests ---

    #[test]
    fn test_node_duration_frames() {
        let node = ColorCorrectionNode::new(1, CorrectionScope::Clip, 10, 50, "grade_a");
        assert_eq!(node.duration_frames(), 40);
    }

    #[test]
    fn test_node_duration_zero_when_inverted() {
        let node = ColorCorrectionNode::new(1, CorrectionScope::Clip, 50, 10, "grade_a");
        assert_eq!(node.duration_frames(), 0);
    }

    #[test]
    fn test_node_contains_frame_inside() {
        let node = ColorCorrectionNode::new(1, CorrectionScope::Clip, 10, 50, "grade_a");
        assert!(node.contains_frame(25));
    }

    #[test]
    fn test_node_contains_frame_at_in_point() {
        let node = ColorCorrectionNode::new(1, CorrectionScope::Clip, 10, 50, "grade_a");
        assert!(node.contains_frame(10));
    }

    #[test]
    fn test_node_contains_frame_at_out_point_excluded() {
        let node = ColorCorrectionNode::new(1, CorrectionScope::Clip, 10, 50, "grade_a");
        assert!(!node.contains_frame(50));
    }

    #[test]
    fn test_node_contains_frame_before() {
        let node = ColorCorrectionNode::new(1, CorrectionScope::Clip, 10, 50, "grade_a");
        assert!(!node.contains_frame(5));
    }

    #[test]
    fn test_node_contains_frame_after() {
        let node = ColorCorrectionNode::new(1, CorrectionScope::Clip, 10, 50, "grade_a");
        assert!(!node.contains_frame(100));
    }

    // --- ColorCorrectionLayer tests ---

    #[test]
    fn test_layer_empty() {
        let layer = ColorCorrectionLayer::new();
        assert_eq!(layer.node_count(), 0);
    }

    #[test]
    fn test_layer_add_increases_count() {
        let mut layer = ColorCorrectionLayer::new();
        layer.add(ColorCorrectionNode::new(
            1,
            CorrectionScope::Clip,
            0,
            100,
            "g1",
        ));
        assert_eq!(layer.node_count(), 1);
    }

    #[test]
    fn test_layer_find_for_frame_matches() {
        let mut layer = ColorCorrectionLayer::new();
        layer.add(ColorCorrectionNode::new(
            1,
            CorrectionScope::Clip,
            0,
            100,
            "g1",
        ));
        layer.add(ColorCorrectionNode::new(
            2,
            CorrectionScope::Clip,
            200,
            300,
            "g2",
        ));
        let matches = layer.find_for_frame(50);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].clip_id, 1);
    }

    #[test]
    fn test_layer_find_for_frame_no_match() {
        let mut layer = ColorCorrectionLayer::new();
        layer.add(ColorCorrectionNode::new(
            1,
            CorrectionScope::Clip,
            0,
            100,
            "g1",
        ));
        assert!(layer.find_for_frame(150).is_empty());
    }

    #[test]
    fn test_layer_find_for_frame_multiple_matches() {
        let mut layer = ColorCorrectionLayer::new();
        layer.add(ColorCorrectionNode::new(
            1,
            CorrectionScope::Clip,
            0,
            100,
            "g1",
        ));
        layer.add(ColorCorrectionNode::new(
            0,
            CorrectionScope::Global,
            0,
            1000,
            "g_global",
        ));
        let matches = layer.find_for_frame(50);
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_layer_remove_by_clip() {
        let mut layer = ColorCorrectionLayer::new();
        layer.add(ColorCorrectionNode::new(
            1,
            CorrectionScope::Clip,
            0,
            100,
            "g1",
        ));
        layer.add(ColorCorrectionNode::new(
            2,
            CorrectionScope::Clip,
            100,
            200,
            "g2",
        ));
        layer.remove_by_clip(1);
        assert_eq!(layer.node_count(), 1);
        assert_eq!(layer.nodes[0].clip_id, 2);
    }
}
