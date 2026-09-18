//! Flame graph generation from CPU profiling samples.
//!
//! # Configurable capacity
//!
//! [`FlameGraphGenerator`] supports a [`with_capacity`](FlameGraphGenerator::with_capacity)
//! constructor that pre-allocates the sample buffer and imposes a `depth_cap`
//! on stack frames.  Frames beyond `depth_cap` are silently dropped.
//!
//! # Iterative tree insertion
//!
//! Stack frame traversal is **iterative** using an explicit index-path
//! `Vec<usize>` rather than recursion, so arbitrarily deep call stacks cannot
//! overflow the system stack.  At each level the algorithm resolves the next
//! child by index, avoiding any mutable aliasing.

use crate::cpu::sample::{Sample, StackFrame};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Default limits
// ---------------------------------------------------------------------------

/// Default sample-buffer pre-allocation (used by [`FlameGraphGenerator::new`]).
const DEFAULT_SAMPLES_CAP: usize = 4096;

/// Default maximum frame depth (used by [`FlameGraphGenerator::new`]).
///
/// Frames beyond this depth are silently dropped during tree insertion.
const DEFAULT_DEPTH_CAP: usize = 512;

// ---------------------------------------------------------------------------
// FlameNode
// ---------------------------------------------------------------------------

/// A single node in the flame graph call tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameNode {
    /// Function name.
    pub name: String,

    /// Sample count.
    pub value: u64,

    /// Child nodes.
    pub children: Vec<FlameNode>,
}

// ---------------------------------------------------------------------------
// FlameGraphData
// ---------------------------------------------------------------------------

/// Output of a flame graph generation pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameGraphData {
    /// Root node.
    pub root: FlameNode,

    /// Total samples.
    pub total_samples: u64,
}

// ---------------------------------------------------------------------------
// FlameGraphGenerator
// ---------------------------------------------------------------------------

/// Accumulates CPU profiling samples and generates a [`FlameGraphData`] tree.
///
/// # Iterative insertion
///
/// [`add_samples`](Self::add_samples) internally calls an iterative (non-
/// recursive) tree-building algorithm bounded by `depth_cap`.  Stacks deeper
/// than `depth_cap` are truncated at that depth rather than triggering a stack
/// overflow.
#[derive(Debug)]
pub struct FlameGraphGenerator {
    samples: Vec<Sample>,
    /// Maximum number of frames to walk per sample.
    depth_cap: usize,
}

impl FlameGraphGenerator {
    /// Create a new flame graph generator with default capacity and depth cap.
    pub fn new() -> Self {
        Self {
            samples: Vec::with_capacity(DEFAULT_SAMPLES_CAP),
            depth_cap: DEFAULT_DEPTH_CAP,
        }
    }

    /// Create a new flame graph generator with explicit capacity and depth cap.
    ///
    /// - `samples_cap`: pre-allocates the internal sample buffer.
    /// - `depth_cap`: frames beyond this depth are silently dropped.
    pub fn with_capacity(samples_cap: usize, depth_cap: usize) -> Self {
        Self {
            samples: Vec::with_capacity(samples_cap),
            depth_cap,
        }
    }

    /// Add samples to the generator.
    pub fn add_samples(&mut self, samples: &[Sample]) {
        self.samples.extend_from_slice(samples);
    }

    /// Generate flame graph data from all accumulated samples.
    pub fn generate(&self) -> FlameGraphData {
        let mut root = FlameNode {
            name: "root".to_string(),
            value: 0,
            children: Vec::new(),
        };

        for sample in &self.samples {
            add_sample_iterative(&mut root, &sample.stack, self.depth_cap);
        }

        FlameGraphData {
            root,
            total_samples: self.samples.len() as u64,
        }
    }

    /// Get total sample count.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Clear all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Return the configured depth cap.
    pub fn depth_cap(&self) -> usize {
        self.depth_cap
    }
}

// ---------------------------------------------------------------------------
// Iterative tree insertion (safe, index-based)
// ---------------------------------------------------------------------------

/// Insert one sample into the tree using an explicit index-path stack.
///
/// The algorithm builds a `Vec<usize>` path (one index per frame) by making
/// two passes:
///
/// 1. **Path-building pass** — walk the tree from `root`, finding or creating
///    child nodes and recording their indices.  All node increments happen in
///    this pass.
/// 2. **Result** — the last index in the path corresponds to the leaf node,
///    whose `value` is already incremented.
///
/// This is fully safe (no raw pointers, no recursion) and bounded by
/// `depth_cap`.
fn add_sample_iterative(root: &mut FlameNode, stack: &[StackFrame], depth_cap: usize) {
    if stack.is_empty() {
        return;
    }

    root.value += 1;

    let depth = stack.len().min(depth_cap);

    // We keep a mutable reference to the *current* node.  We advance it by
    // finding/creating the next child and then re-borrowing the child.
    //
    // Rust's borrow checker allows this because at each step we pass the
    // borrow into `ensure_child` which returns a mutable reference to one of
    // the node's children, relinquishing the parent borrow.
    let mut current: &mut FlameNode = root;

    for frame in &stack[..depth] {
        let name = frame.function.as_str();
        current = ensure_child(current, name);
    }

    // The last `current` is the leaf; increment its sample count.
    current.value += 1;
}

/// Find or create a child with the given name, returning a mutable reference
/// to it.
///
/// The returned reference borrows from `node.children`, so `node` is
/// effectively consumed for the lifetime of the returned reference.
fn ensure_child<'a>(node: &'a mut FlameNode, name: &str) -> &'a mut FlameNode {
    // Check whether a child with this name already exists.
    let pos = node.children.iter().position(|c| c.name == name);
    let idx = match pos {
        Some(i) => {
            node.children[i].value += 1;
            i
        }
        None => {
            let i = node.children.len();
            node.children.push(FlameNode {
                name: name.to_owned(),
                value: 1,
                children: Vec::new(),
            });
            i
        }
    };
    &mut node.children[idx]
}

impl Default for FlameGraphGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flame_graph_generator() {
        let mut generator = FlameGraphGenerator::new();

        let mut sample = Sample::new(1, 50.0);
        sample.add_frame(StackFrame::new("func1".to_string(), 0x1000));
        sample.add_frame(StackFrame::new("func2".to_string(), 0x2000));

        generator.add_samples(&[sample]);
        assert_eq!(generator.sample_count(), 1);

        let data = generator.generate();
        assert_eq!(data.total_samples, 1);
        assert_eq!(data.root.value, 1);
    }

    #[test]
    fn test_flame_graph_merging() {
        let mut generator = FlameGraphGenerator::new();

        // Two samples with same stack prefix
        let mut sample1 = Sample::new(1, 50.0);
        sample1.add_frame(StackFrame::new("func1".to_string(), 0x1000));
        sample1.add_frame(StackFrame::new("func2".to_string(), 0x2000));

        let mut sample2 = Sample::new(1, 50.0);
        sample2.add_frame(StackFrame::new("func1".to_string(), 0x1000));
        sample2.add_frame(StackFrame::new("func3".to_string(), 0x3000));

        generator.add_samples(&[sample1, sample2]);

        let data = generator.generate();
        assert_eq!(data.root.children.len(), 1);
        assert_eq!(data.root.children[0].name, "func1");
        assert_eq!(data.root.children[0].children.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Sub-item 32 new tests
    // -----------------------------------------------------------------------

    /// Generate a flamegraph with a 10 000-deep stack, assert no stack overflow
    /// + result has ≤ depth_cap frames below root.
    #[test]
    fn test_flamegraph_capacity_smoke() {
        const DEPTH_CAP: usize = 64;
        let mut generator = FlameGraphGenerator::with_capacity(8, DEPTH_CAP);

        // Build a very deep single-chain sample.
        let mut sample = Sample::new(1, 0.0);
        for i in 0..10_000 {
            sample.add_frame(StackFrame::new(format!("frame_{i}"), i as u64));
        }

        generator.add_samples(&[sample]);

        // Must not panic or stack-overflow.
        let data = generator.generate();
        assert_eq!(data.total_samples, 1);

        // Measure the actual depth by walking the first-child chain.
        let mut depth = 0usize;
        let mut node = &data.root;
        while !node.children.is_empty() {
            depth += 1;
            node = &node.children[0];
        }

        assert!(
            depth <= DEPTH_CAP,
            "depth {depth} exceeds depth_cap {DEPTH_CAP}"
        );
    }

    /// `FlameGraphGenerator::with_capacity(1000, 50)` + add samples,
    /// generate, no panic.
    #[test]
    fn test_flamegraph_with_capacity_constructor() {
        let mut generator = FlameGraphGenerator::with_capacity(1000, 50);
        assert_eq!(generator.depth_cap(), 50);

        let mut sample = Sample::new(1, 10.0);
        for i in 0..30 {
            sample.add_frame(StackFrame::new(format!("f{i}"), i as u64));
        }
        generator.add_samples(&[sample]);

        let data = generator.generate();
        assert_eq!(data.total_samples, 1);
        assert_eq!(data.root.value, 1);
    }
}
