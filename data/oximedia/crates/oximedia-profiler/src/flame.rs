//! Flame graph generation via call stack sampling and time accumulation.
//!
//! This module provides utilities for building flame graphs from sampled
//! call stacks, merging stack trees, and serialising results for rendering.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::Duration;

/// A single frame in a call stack.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StackFrame {
    /// Function or symbol name.
    pub name: String,
    /// Source file, if available.
    pub file: Option<String>,
    /// Source line, if available.
    pub line: Option<u32>,
}

impl StackFrame {
    /// Create a new stack frame with just a name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            file: None,
            line: None,
        }
    }

    /// Create a stack frame with file/line information.
    pub fn with_location(name: impl Into<String>, file: impl Into<String>, line: u32) -> Self {
        Self {
            name: name.into(),
            file: Some(file.into()),
            line: Some(line),
        }
    }
}

/// A sampled call stack (ordered from outermost to innermost frame).
#[derive(Debug, Clone)]
pub struct CallStack {
    /// Frames from root to leaf.
    pub frames: Vec<StackFrame>,
    /// Approximate duration represented by this sample.
    pub sample_duration: Duration,
}

impl CallStack {
    /// Create a new call stack.
    pub fn new(frames: Vec<StackFrame>, sample_duration: Duration) -> Self {
        Self {
            frames,
            sample_duration,
        }
    }

    /// Depth of the stack (number of frames).
    pub fn depth(&self) -> usize {
        self.frames.len()
    }
}

/// A node in the flame graph tree.
#[derive(Debug, Clone)]
pub struct FlameNode {
    /// Symbol name for this node.
    pub name: String,
    /// Cumulative self time (time spent in this exact frame, not children).
    pub self_time: Duration,
    /// Total time including all descendants.
    pub total_time: Duration,
    /// Number of samples hitting this node.
    pub sample_count: u64,
    /// Child nodes keyed by name.
    pub children: HashMap<String, FlameNode>,
}

impl FlameNode {
    /// Create a new empty node.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            self_time: Duration::ZERO,
            total_time: Duration::ZERO,
            sample_count: 0,
            children: HashMap::new(),
        }
    }

    /// Self time as a fraction of the provided total duration.
    #[allow(clippy::cast_precision_loss)]
    pub fn self_fraction(&self, total: Duration) -> f64 {
        if total.is_zero() {
            0.0
        } else {
            self.self_time.as_secs_f64() / total.as_secs_f64()
        }
    }

    /// Total time as a fraction of the provided total duration.
    #[allow(clippy::cast_precision_loss)]
    pub fn total_fraction(&self, total: Duration) -> f64 {
        if total.is_zero() {
            0.0
        } else {
            self.total_time.as_secs_f64() / total.as_secs_f64()
        }
    }

    /// Recursive count of all descendant nodes (including self).
    pub fn node_count(&self) -> usize {
        1 + self
            .children
            .values()
            .map(|c| c.node_count())
            .sum::<usize>()
    }

    /// Insert a stack path (slice of frame names) into this node's children tree.
    /// `self` is the *parent* node; stats for `self` are already updated by the caller.
    fn insert_path(&mut self, path: &[String], sample_duration: Duration) {
        if path.is_empty() {
            self.self_time += sample_duration;
            return;
        }
        let child = self
            .children
            .entry(path[0].clone())
            .or_insert_with(|| FlameNode::new(path[0].clone()));
        child.total_time += sample_duration;
        child.sample_count += 1;
        child.insert_path(&path[1..], sample_duration);
    }
}

/// Builder that accumulates sampled call stacks into a flame graph tree.
#[derive(Debug, Default)]
pub struct FlameGraphBuilder {
    /// Root-level nodes indexed by name.
    root_children: HashMap<String, FlameNode>,
    total_samples: u64,
    total_duration: Duration,
}

impl FlameGraphBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a sampled call stack to the tree.
    pub fn add_stack(&mut self, stack: &CallStack) {
        if stack.frames.is_empty() {
            return;
        }
        self.total_samples += 1;
        self.total_duration += stack.sample_duration;

        let names: Vec<String> = stack.frames.iter().map(|f| f.name.clone()).collect();
        let root_name = &names[0];
        let root = self
            .root_children
            .entry(root_name.clone())
            .or_insert_with(|| FlameNode::new(root_name.clone()));

        root.total_time += stack.sample_duration;
        root.sample_count += 1;

        if names.len() == 1 {
            root.self_time += stack.sample_duration;
        } else {
            root.insert_path(&names[1..], stack.sample_duration);
        }
    }

    /// Merge another builder's data into this one.
    pub fn merge(&mut self, other: FlameGraphBuilder) {
        self.total_samples += other.total_samples;
        self.total_duration += other.total_duration;
        merge_children(&mut self.root_children, other.root_children);
    }

    /// Total wall-clock duration represented by all samples.
    pub fn total_duration(&self) -> Duration {
        self.total_duration
    }

    /// Total number of samples recorded.
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Collect root-level nodes sorted by total time descending.
    pub fn top_nodes(&self) -> Vec<&FlameNode> {
        let mut nodes: Vec<&FlameNode> = self.root_children.values().collect();
        nodes.sort_by(|a, b| b.total_time.cmp(&a.total_time));
        nodes
    }

    /// Serialise to the "folded" text format used by flamegraph.pl and similar tools.
    ///
    /// Each line is `frame1;frame2;...;leafN count`.
    pub fn to_folded(&self) -> String {
        let mut lines = Vec::new();
        for node in self.root_children.values() {
            collect_folded(node, &mut Vec::new(), &mut lines);
        }
        lines.sort();
        lines.join("\n")
    }
}

fn merge_children(dst: &mut HashMap<String, FlameNode>, src: HashMap<String, FlameNode>) {
    for (name, src_node) in src {
        let dst_node = dst
            .entry(name.clone())
            .or_insert_with(|| FlameNode::new(name));
        dst_node.sample_count += src_node.sample_count;
        dst_node.self_time += src_node.self_time;
        dst_node.total_time += src_node.total_time;
        merge_children(&mut dst_node.children, src_node.children);
    }
}

fn collect_folded(node: &FlameNode, path: &mut Vec<String>, out: &mut Vec<String>) {
    path.push(node.name.clone());
    if node.children.is_empty() {
        out.push(format!("{} {}", path.join(";"), node.sample_count));
    } else {
        for child in node.children.values() {
            collect_folded(child, path, out);
        }
    }
    path.pop();
}

/// A complete, built flame graph ready for rendering.
#[derive(Debug)]
pub struct FlameGraph {
    /// Root-level nodes sorted by total time descending.
    pub roots: Vec<FlameNode>,
    /// Total sample count across all stacks.
    pub total_samples: u64,
    /// Total duration represented.
    pub total_duration: Duration,
}

impl FlameGraph {
    /// Build a `FlameGraph` from a builder.
    pub fn from_builder(builder: FlameGraphBuilder) -> Self {
        let mut roots: Vec<FlameNode> = builder.root_children.into_values().collect();
        roots.sort_by(|a, b| b.total_time.cmp(&a.total_time));
        Self {
            roots,
            total_samples: builder.total_samples,
            total_duration: builder.total_duration,
        }
    }

    /// Total node count across the entire tree.
    pub fn total_nodes(&self) -> usize {
        self.roots.iter().map(|n| n.node_count()).sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_stack(names: &[&str], millis: u64) -> CallStack {
        let frames = names.iter().map(|n| StackFrame::new(*n)).collect();
        CallStack::new(frames, Duration::from_millis(millis))
    }

    #[test]
    fn test_stack_frame_new() {
        let f = StackFrame::new("main");
        assert_eq!(f.name, "main");
        assert!(f.file.is_none());
        assert!(f.line.is_none());
    }

    #[test]
    fn test_stack_frame_with_location() {
        let f = StackFrame::with_location("foo", "foo.rs", 42);
        assert_eq!(f.name, "foo");
        assert_eq!(f.file.as_deref(), Some("foo.rs"));
        assert_eq!(f.line, Some(42));
    }

    #[test]
    fn test_call_stack_depth() {
        let s = make_stack(&["a", "b", "c"], 10);
        assert_eq!(s.depth(), 3);
    }

    #[test]
    fn test_builder_empty() {
        let b = FlameGraphBuilder::new();
        assert_eq!(b.total_samples(), 0);
        assert_eq!(b.total_duration(), Duration::ZERO);
    }

    #[test]
    fn test_builder_single_stack() {
        let mut b = FlameGraphBuilder::new();
        b.add_stack(&make_stack(&["main", "render"], 10));
        assert_eq!(b.total_samples(), 1);
        assert_eq!(b.total_duration(), Duration::from_millis(10));
    }

    #[test]
    fn test_builder_multiple_stacks_same_root() {
        let mut b = FlameGraphBuilder::new();
        b.add_stack(&make_stack(&["main", "a"], 5));
        b.add_stack(&make_stack(&["main", "b"], 5));
        let top = b.top_nodes();
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].name, "main");
        assert_eq!(top[0].sample_count, 2);
    }

    #[test]
    fn test_builder_empty_stack_ignored() {
        let mut b = FlameGraphBuilder::new();
        let empty = CallStack::new(vec![], Duration::from_millis(10));
        b.add_stack(&empty);
        assert_eq!(b.total_samples(), 0);
    }

    #[test]
    fn test_flame_node_self_fraction() {
        let mut node = FlameNode::new("x");
        node.self_time = Duration::from_millis(50);
        let frac = node.self_fraction(Duration::from_millis(100));
        assert!((frac - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_flame_node_total_fraction_zero_total() {
        let node = FlameNode::new("x");
        assert_eq!(node.total_fraction(Duration::ZERO), 0.0);
    }

    #[test]
    fn test_flame_node_node_count() {
        let mut parent = FlameNode::new("p");
        parent.children.insert("c".to_string(), FlameNode::new("c"));
        assert_eq!(parent.node_count(), 2);
    }

    #[test]
    fn test_merge_builders() {
        let mut b1 = FlameGraphBuilder::new();
        b1.add_stack(&make_stack(&["main"], 10));
        let mut b2 = FlameGraphBuilder::new();
        b2.add_stack(&make_stack(&["main"], 20));
        b1.merge(b2);
        assert_eq!(b1.total_samples(), 2);
        assert_eq!(b1.total_duration(), Duration::from_millis(30));
    }

    #[test]
    fn test_to_folded_format() {
        let mut b = FlameGraphBuilder::new();
        b.add_stack(&make_stack(&["main", "foo"], 10));
        let folded = b.to_folded();
        assert!(folded.contains("main;foo"));
        assert!(folded.contains('1'));
    }

    #[test]
    fn test_flame_graph_from_builder() {
        let mut b = FlameGraphBuilder::new();
        b.add_stack(&make_stack(&["a"], 5));
        b.add_stack(&make_stack(&["b"], 15));
        let fg = FlameGraph::from_builder(b);
        // roots sorted by total_time descending
        assert_eq!(fg.roots[0].name, "b");
        assert_eq!(fg.total_samples, 2);
    }

    #[test]
    fn test_flame_graph_total_nodes() {
        let mut b = FlameGraphBuilder::new();
        b.add_stack(&make_stack(&["r", "c1"], 5));
        b.add_stack(&make_stack(&["r", "c2"], 5));
        let fg = FlameGraph::from_builder(b);
        // r -> c1, r -> c2 = 3 nodes
        assert_eq!(fg.total_nodes(), 3);
    }

    #[test]
    fn test_top_nodes_ordering() {
        let mut b = FlameGraphBuilder::new();
        b.add_stack(&make_stack(&["slow"], 100));
        b.add_stack(&make_stack(&["fast"], 1));
        let top = b.top_nodes();
        assert_eq!(top[0].name, "slow");
    }
}
