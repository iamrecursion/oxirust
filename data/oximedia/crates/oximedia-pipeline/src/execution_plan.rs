//! Execution planning and pipeline optimization.
//!
//! Converts a [`PipelineGraph`] into an ordered [`ExecutionPlan`] composed of
//! [`ExecutionStage`]s, each annotated with a [`ResourceEstimate`].  The
//! [`PipelineOptimizer`] can then merge or split stages to improve throughput.

use std::collections::{HashMap, HashSet};

use crate::graph::PipelineGraph;
use crate::node::{FilterConfig, NodeId, NodeSpec, NodeType};
use crate::PipelineError;

// == ResourceEstimate =========================================================

/// Estimated resource requirements for a single execution stage or an entire
/// plan.
///
/// `memory_bytes` is the estimated heap footprint in bytes.  `cpu_weight` is a
/// raw cost value derived from [`FilterConfig::cost_estimate`]; the planner
/// does **not** normalise it to `[0.0, 1.0]`.  `is_gpu_candidate` flags
/// workloads that would benefit from GPU acceleration (heavy pixel transforms).
#[derive(Debug, Clone)]
pub struct ResourceEstimate {
    /// Estimated memory usage in bytes.
    pub memory_bytes: u64,
    /// Relative CPU cost (raw weight).
    pub cpu_weight: f64,
    /// Whether the workload could benefit from GPU acceleration.
    pub is_gpu_candidate: bool,
}

impl ResourceEstimate {
    /// Create a new `ResourceEstimate`.
    pub fn new(memory_bytes: u64, cpu_weight: f64, is_gpu_candidate: bool) -> Self {
        Self {
            memory_bytes,
            cpu_weight,
            is_gpu_candidate,
        }
    }

    /// Combine two estimates additively (memory and CPU are summed; GPU flag
    /// is OR-ed).
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            memory_bytes: self.memory_bytes.saturating_add(other.memory_bytes),
            cpu_weight: self.cpu_weight + other.cpu_weight,
            is_gpu_candidate: self.is_gpu_candidate || other.is_gpu_candidate,
        }
    }
}

impl Default for ResourceEstimate {
    fn default() -> Self {
        Self {
            memory_bytes: 0,
            cpu_weight: 0.0,
            is_gpu_candidate: false,
        }
    }
}

// == ExecutionStage ===========================================================

/// A group of pipeline nodes that are scheduled to execute together.
///
/// Nodes within a stage share the same *depth level* in the DAG, meaning no
/// dependency edges exist between them and they can potentially be processed in
/// parallel.
#[derive(Debug, Clone)]
pub struct ExecutionStage {
    /// Sequential stage identifier (0-based).
    pub stage_id: usize,
    /// Nodes assigned to this stage.
    pub nodes: Vec<NodeId>,
    /// Aggregated resource estimate for every node in the stage.
    pub resource_estimate: ResourceEstimate,
    /// Stage IDs that must complete before this stage can start.
    pub dependencies: Vec<usize>,
    /// Whether the nodes in this stage can be executed in parallel.
    ///
    /// `true` when the stage contains multiple independent nodes (i.e. nodes
    /// that share no edges among themselves within this stage).
    pub parallel: bool,
    /// Groups of nodes within this stage that are on independent branches
    /// and can be scheduled concurrently.
    ///
    /// When `parallel` is `true`, each inner `Vec` is an independent branch.
    /// When `parallel` is `false`, there is a single group containing all nodes.
    pub parallel_groups: Vec<Vec<NodeId>>,
}

// == ExecutionPlan ============================================================

/// A fully-resolved execution plan consisting of ordered stages.
///
/// Constructed by [`ExecutionPlanner::plan`] and optionally refined by
/// [`PipelineOptimizer::optimize`].
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Stages in execution order (stage 0 runs first).
    pub stages: Vec<ExecutionStage>,
    /// Aggregated estimate across all stages.
    pub total_estimate: ResourceEstimate,
}

impl ExecutionPlan {
    /// Build an `ExecutionPlan` from a list of stages.
    ///
    /// The total resource estimate is computed automatically as the sum of all
    /// per-stage estimates.
    pub fn new(stages: Vec<ExecutionStage>) -> Self {
        let total_estimate = stages.iter().fold(ResourceEstimate::default(), |acc, s| {
            acc.merge(&s.resource_estimate)
        });
        Self {
            stages,
            total_estimate,
        }
    }

    /// Number of stages in the plan.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Flatten all stages into a single ordered `Vec<NodeId>`.
    pub fn node_order(&self) -> Vec<NodeId> {
        self.stages
            .iter()
            .flat_map(|s| s.nodes.iter().copied())
            .collect()
    }

    /// Returns `true` when the plan contains no stages.
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Returns the number of stages that contain parallelisable nodes.
    pub fn parallel_stage_count(&self) -> usize {
        self.stages.iter().filter(|s| s.parallel).count()
    }

    /// Returns the maximum degree of parallelism across all stages
    /// (i.e. the stage with the most parallel groups).
    pub fn max_parallelism(&self) -> usize {
        self.stages
            .iter()
            .map(|s| s.parallel_groups.len())
            .max()
            .unwrap_or_default()
    }
}

// == ExecutionPlanner =========================================================

/// Converts a [`PipelineGraph`] into an [`ExecutionPlan`].
///
/// Nodes are grouped into stages by their *depth level*: the longest path from
/// any source to a node determines its depth, and all nodes at the same depth
/// are placed in the same stage.
pub struct ExecutionPlanner;

impl ExecutionPlanner {
    /// Build an execution plan from a validated pipeline graph.
    ///
    /// Returns `PipelineError::CycleDetected` if the graph contains a cycle.
    pub fn plan(graph: &PipelineGraph) -> Result<ExecutionPlan, PipelineError> {
        let sorted = graph.topological_sort()?;

        if sorted.is_empty() {
            return Ok(ExecutionPlan::new(Vec::new()));
        }

        // Assign depth levels (longest path from sources)
        let mut levels: HashMap<NodeId, usize> = HashMap::new();

        for &node_id in &sorted {
            let preds = graph.predecessors(&node_id);
            let level = if preds.is_empty() {
                0
            } else {
                preds
                    .iter()
                    .filter_map(|p| levels.get(p))
                    .map(|l| l + 1)
                    .max()
                    .unwrap_or_default()
            };
            levels.insert(node_id, level);
        }

        let max_level: usize = levels.values().copied().max().unwrap_or_default();

        // Group nodes by level
        let mut level_groups: Vec<Vec<NodeId>> = vec![Vec::new(); max_level + 1];

        for &node_id in &sorted {
            if let Some(&level) = levels.get(&node_id) {
                if let Some(group) = level_groups.get_mut(level) {
                    group.push(node_id);
                }
            }
        }

        // Build stages
        let mut stages = Vec::with_capacity(level_groups.len());

        for (stage_id, nodes) in level_groups.into_iter().enumerate() {
            let resource_estimate = nodes
                .iter()
                .map(|id| estimate_node_resources(graph.nodes.get(id)))
                .fold(ResourceEstimate::default(), |acc, r| acc.merge(&r));

            // Dependencies: deduplicated predecessor stage-ids.
            let mut deps: Vec<usize> = Vec::new();
            for &nid in &nodes {
                for pred in graph.predecessors(&nid) {
                    if let Some(&pred_level) = levels.get(&pred) {
                        if pred_level < stage_id && !deps.contains(&pred_level) {
                            deps.push(pred_level);
                        }
                    }
                }
            }
            deps.sort_unstable();

            // Detect parallelism: any stage with multiple nodes can execute
            // them concurrently (no intra-stage edges exist in a DAG grouped
            // by depth level). `parallel_groups` further partitions nodes by
            // shared transitive ancestors — nodes on completely independent
            // branches (disjoint ancestor sets) form separate groups.
            let parallel_groups = compute_parallel_groups(&nodes, graph);
            let parallel = nodes.len() > 1;

            stages.push(ExecutionStage {
                stage_id,
                nodes,
                resource_estimate,
                dependencies: deps,
                parallel,
                parallel_groups,
            });
        }

        Ok(ExecutionPlan::new(stages))
    }
}

// == PipelineOptimizer ========================================================

/// Optimizes an [`ExecutionPlan`] by merging adjacent stages.
///
/// The default strategy merges consecutive stages whose combined CPU weight
/// falls below `Self::MERGE_THRESHOLD`, reducing scheduling overhead for
/// lightweight operations.
pub struct PipelineOptimizer;

impl PipelineOptimizer {
    /// Maximum combined CPU weight below which two adjacent stages are merged.
    const MERGE_THRESHOLD: f64 = 10.0;

    /// Optimize an execution plan.
    ///
    /// Adjacent stages whose combined `cpu_weight` is below
    /// `Self::MERGE_THRESHOLD` are collapsed into a single stage.  The
    /// resulting plan has renumbered stage IDs and recomputed dependency
    /// indices.
    pub fn optimize(plan: ExecutionPlan) -> ExecutionPlan {
        if plan.stages.len() <= 1 {
            return plan;
        }

        // Group consecutive stages that can be merged
        let mut groups: Vec<Vec<ExecutionStage>> = Vec::new();

        for stage in plan.stages {
            let should_merge = groups.last().map_or(false, |group| {
                let group_weight: f64 = group.iter().map(|s| s.resource_estimate.cpu_weight).sum();
                group_weight + stage.resource_estimate.cpu_weight < Self::MERGE_THRESHOLD
            });

            if should_merge {
                if let Some(group) = groups.last_mut() {
                    group.push(stage);
                }
            } else {
                groups.push(vec![stage]);
            }
        }

        // Build old_stage_id -> new_stage_id mapping
        let mut old_to_new: HashMap<usize, usize> = HashMap::new();
        for (new_id, group) in groups.iter().enumerate() {
            for stage in group {
                old_to_new.insert(stage.stage_id, new_id);
            }
        }

        // Collapse each group into a single stage
        let stages: Vec<ExecutionStage> = groups
            .into_iter()
            .enumerate()
            .map(|(new_id, group)| {
                let nodes: Vec<NodeId> =
                    group.iter().flat_map(|s| s.nodes.iter().copied()).collect();

                let resource_estimate = group.iter().fold(ResourceEstimate::default(), |acc, s| {
                    acc.merge(&s.resource_estimate)
                });

                // Remap dependencies, remove self-references and duplicates.
                let mut deps: Vec<usize> = group
                    .iter()
                    .flat_map(|s| s.dependencies.iter())
                    .filter_map(|&old_dep| old_to_new.get(&old_dep).copied())
                    .filter(|&d| d != new_id)
                    .collect();
                deps.sort_unstable();
                deps.dedup();

                let parallel = nodes.len() > 1;
                let parallel_groups = if parallel {
                    nodes.iter().map(|n| vec![*n]).collect()
                } else {
                    vec![nodes.clone()]
                };

                ExecutionStage {
                    stage_id: new_id,
                    nodes,
                    resource_estimate,
                    dependencies: deps,
                    parallel,
                    parallel_groups,
                }
            })
            .collect();

        ExecutionPlan::new(stages)
    }
}

// == Internal helpers =========================================================

/// Collect the full transitive ancestor set for a node by walking up
/// all predecessor edges in the graph (BFS).
fn transitive_ancestors(node: &NodeId, graph: &PipelineGraph) -> HashSet<NodeId> {
    let mut visited = HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    for p in graph.predecessors(node) {
        queue.push_back(p);
    }
    while let Some(current) = queue.pop_front() {
        if visited.insert(current) {
            for p in graph.predecessors(&current) {
                if !visited.contains(&p) {
                    queue.push_back(p);
                }
            }
        }
    }
    visited
}

/// Partition a set of same-level nodes into independent parallel groups.
///
/// Nodes that share a common ancestor (transitively, not just direct
/// predecessors) are placed in the same group. Nodes on completely
/// independent branches of the DAG end up in separate groups, allowing
/// them to be scheduled concurrently.
///
/// The implementation uses a union-find structure: for each pair of nodes
/// in the stage, if their transitive ancestor sets overlap, they belong
/// to the same group.
fn compute_parallel_groups(nodes: &[NodeId], graph: &PipelineGraph) -> Vec<Vec<NodeId>> {
    if nodes.len() <= 1 {
        return vec![nodes.to_vec()];
    }

    // Build per-node transitive ancestor set
    let ancestors: HashMap<NodeId, HashSet<NodeId>> = nodes
        .iter()
        .map(|&nid| {
            let ancs = transitive_ancestors(&nid, graph);
            (nid, ancs)
        })
        .collect();

    // Union-Find via parent map
    let mut parent: HashMap<NodeId, NodeId> = nodes.iter().map(|&n| (n, n)).collect();

    fn find(parent: &mut HashMap<NodeId, NodeId>, x: NodeId) -> NodeId {
        let mut root = x;
        while let Some(&p) = parent.get(&root) {
            if p == root {
                break;
            }
            root = p;
        }
        // Path compression
        let mut current = x;
        while current != root {
            if let Some(&next) = parent.get(&current) {
                parent.insert(current, root);
                current = next;
            } else {
                break;
            }
        }
        root
    }

    fn union(parent: &mut HashMap<NodeId, NodeId>, a: NodeId, b: NodeId) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent.insert(rb, ra);
        }
    }

    // Merge nodes that share at least one common transitive ancestor
    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            let a = nodes[i];
            let b = nodes[j];
            let ancs_a = ancestors.get(&a);
            let ancs_b = ancestors.get(&b);
            if let (Some(aa), Some(ab)) = (ancs_a, ancs_b) {
                if !aa.is_disjoint(ab) {
                    union(&mut parent, a, b);
                }
            }
        }
    }

    // Group by root
    let mut groups_map: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for &nid in nodes {
        let root = find(&mut parent, nid);
        groups_map.entry(root).or_default().push(nid);
    }

    groups_map.into_values().collect()
}

/// Estimate resource requirements for a single pipeline node.
fn estimate_node_resources(spec: Option<&NodeSpec>) -> ResourceEstimate {
    let spec = match spec {
        Some(s) => s,
        None => return ResourceEstimate::default(),
    };

    match &spec.node_type {
        NodeType::Source(_) => ResourceEstimate::new(4 * 1024 * 1024, 2.0, false),
        NodeType::Sink(_) => ResourceEstimate::new(2 * 1024 * 1024, 1.0, false),
        NodeType::Filter(config) => {
            let cost = config.cost_estimate() as f64;
            let memory_bytes = match config {
                FilterConfig::Scale { width, height } => {
                    // Rough: 4 bytes/pixel x 2 buffers (input + output).
                    let pixels = (*width as u64) * (*height as u64);
                    pixels * 8
                }
                FilterConfig::Overlay => 16 * 1024 * 1024,
                FilterConfig::Concat { count } => (*count as u64).max(1) * 4 * 1024 * 1024,
                _ => 2 * 1024 * 1024,
            };
            let is_gpu = matches!(
                config,
                FilterConfig::Scale { .. } | FilterConfig::Overlay | FilterConfig::Pad { .. }
            );
            ResourceEstimate::new(memory_bytes, cost, is_gpu)
        }
        NodeType::Split => ResourceEstimate::new(4 * 1024 * 1024, 1.0, false),
        NodeType::Merge => ResourceEstimate::new(4 * 1024 * 1024, 2.0, false),
        NodeType::Null => ResourceEstimate::new(1024 * 1024, 0.5, false),
        NodeType::Conditional(_) => ResourceEstimate::new(512 * 1024, 0.5, false),
    }
}

// == PipelineCheckpoint =======================================================

/// The processing state of a single node within a checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeState {
    /// Node has not yet started processing.
    Pending,
    /// Node is currently processing (partial progress).
    InProgress {
        /// An opaque progress indicator (e.g. frame count, byte offset).
        progress: u64,
    },
    /// Node has completed processing.
    Completed,
    /// Node encountered an error and was skipped or failed.
    Failed {
        /// Description of the failure.
        reason: String,
    },
}

/// A snapshot of pipeline execution state at a particular point in time.
///
/// Checkpoints allow saving and restoring pipeline progress, enabling
/// resume-from-failure, pipeline migration, and debugging.
#[derive(Debug, Clone)]
pub struct PipelineCheckpoint {
    /// The graph this checkpoint applies to (frozen copy).
    graph: PipelineGraph,
    /// Per-node processing state at checkpoint time.
    node_states: HashMap<NodeId, NodeState>,
    /// The stage index that was active when the checkpoint was taken.
    active_stage: usize,
    /// Optional label for this checkpoint.
    label: String,
}

impl PipelineCheckpoint {
    /// Create a new checkpoint from a graph, marking all nodes as `Pending`.
    pub fn new(graph: &PipelineGraph, label: impl Into<String>) -> Self {
        let node_states = graph
            .nodes
            .keys()
            .map(|&id| (id, NodeState::Pending))
            .collect();
        Self {
            graph: graph.clone(),
            node_states,
            active_stage: 0,
            label: label.into(),
        }
    }

    /// Create a checkpoint at a specific stage, with completed nodes marked.
    pub fn at_stage(
        graph: &PipelineGraph,
        plan: &ExecutionPlan,
        completed_through_stage: usize,
        label: impl Into<String>,
    ) -> Self {
        let mut node_states: HashMap<NodeId, NodeState> = graph
            .nodes
            .keys()
            .map(|&id| (id, NodeState::Pending))
            .collect();

        // Mark nodes in completed stages
        for stage in &plan.stages {
            if stage.stage_id <= completed_through_stage {
                for &nid in &stage.nodes {
                    node_states.insert(nid, NodeState::Completed);
                }
            }
        }

        let active_stage = (completed_through_stage + 1).min(plan.stage_count());

        Self {
            graph: graph.clone(),
            node_states,
            active_stage,
            label: label.into(),
        }
    }

    /// Update the state of a specific node.
    pub fn set_node_state(
        &mut self,
        node_id: &NodeId,
        state: NodeState,
    ) -> Result<(), PipelineError> {
        if !self.node_states.contains_key(node_id) {
            return Err(PipelineError::NodeNotFound(node_id.to_string()));
        }
        self.node_states.insert(*node_id, state);
        Ok(())
    }

    /// Get the state of a specific node.
    pub fn node_state(&self, node_id: &NodeId) -> Result<&NodeState, PipelineError> {
        self.node_states
            .get(node_id)
            .ok_or_else(|| PipelineError::NodeNotFound(node_id.to_string()))
    }

    /// Return the active stage index.
    pub fn active_stage(&self) -> usize {
        self.active_stage
    }

    /// Set the active stage index.
    pub fn set_active_stage(&mut self, stage: usize) {
        self.active_stage = stage;
    }

    /// Return the checkpoint label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Return a reference to the frozen graph.
    pub fn graph(&self) -> &PipelineGraph {
        &self.graph
    }

    /// Return all node IDs that are in the `Completed` state.
    pub fn completed_nodes(&self) -> Vec<NodeId> {
        self.node_states
            .iter()
            .filter(|(_, s)| matches!(s, NodeState::Completed))
            .map(|(&id, _)| id)
            .collect()
    }

    /// Return all node IDs that are still `Pending`.
    pub fn pending_nodes(&self) -> Vec<NodeId> {
        self.node_states
            .iter()
            .filter(|(_, s)| matches!(s, NodeState::Pending))
            .map(|(&id, _)| id)
            .collect()
    }

    /// Return all node IDs that have `Failed`.
    pub fn failed_nodes(&self) -> Vec<NodeId> {
        self.node_states
            .iter()
            .filter(|(_, s)| matches!(s, NodeState::Failed { .. }))
            .map(|(&id, _)| id)
            .collect()
    }

    /// Create a resumption plan: an `ExecutionPlan` that skips already-completed
    /// stages and starts from the active stage.
    pub fn resume_plan(&self) -> Result<ExecutionPlan, PipelineError> {
        let full_plan = ExecutionPlanner::plan(&self.graph)?;

        let remaining_stages: Vec<ExecutionStage> = full_plan
            .stages
            .into_iter()
            .filter(|s| s.stage_id >= self.active_stage)
            .enumerate()
            .map(|(new_id, mut stage)| {
                let old_id = stage.stage_id;
                stage.stage_id = new_id;
                // Remap dependencies to exclude completed stages
                stage.dependencies = stage
                    .dependencies
                    .iter()
                    .filter(|&&d| d >= self.active_stage)
                    .map(|&d| d - self.active_stage)
                    .filter(|&d| d < old_id - self.active_stage)
                    .collect();
                stage
            })
            .collect();

        Ok(ExecutionPlan::new(remaining_stages))
    }
}

// == Tests ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::*;

    // -- Helpers --------------------------------------------------------------

    fn video_spec() -> StreamSpec {
        StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25)
    }

    fn video_spec_sized(w: u32, h: u32) -> StreamSpec {
        StreamSpec::video(FrameFormat::Yuv420p, w, h, 25)
    }

    /// Helper to construct an `ExecutionStage` with sensible defaults for the
    /// `parallel` and `parallel_groups` fields.
    fn make_stage(
        stage_id: usize,
        nodes: Vec<NodeId>,
        resource_estimate: ResourceEstimate,
        dependencies: Vec<usize>,
    ) -> ExecutionStage {
        let parallel = nodes.len() > 1;
        let parallel_groups = if parallel {
            nodes.iter().map(|&n| vec![n]).collect()
        } else {
            vec![nodes.clone()]
        };
        ExecutionStage {
            stage_id,
            nodes,
            resource_estimate,
            dependencies,
            parallel,
            parallel_groups,
        }
    }

    fn make_linear_graph() -> (PipelineGraph, NodeId, NodeId, NodeId) {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), video_spec());
        let filt = NodeSpec::filter(
            "scale",
            FilterConfig::Scale {
                width: 1280,
                height: 720,
            },
            video_spec(),
            video_spec_sized(1280, 720),
        );
        let sink = NodeSpec::sink("sink", SinkConfig::Null, video_spec());
        let src_id = g.add_node(src);
        let filt_id = g.add_node(filt);
        let sink_id = g.add_node(sink);
        let _ = g.connect(src_id, "default", filt_id, "default");
        let _ = g.connect(filt_id, "default", sink_id, "default");
        (g, src_id, filt_id, sink_id)
    }

    fn make_branch_merge_graph() -> PipelineGraph {
        //  src -- split --+-- hflip --+-- merge -- sink
        //                 +-- vflip --+
        let mut g = PipelineGraph::new();
        let vs = video_spec();

        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), vs.clone());

        let split = NodeSpec::new(
            "split",
            NodeType::Split,
            vec![("default".to_string(), vs.clone())],
            vec![
                ("out0".to_string(), vs.clone()),
                ("out1".to_string(), vs.clone()),
            ],
        );

        let filter_a = NodeSpec::filter("hflip", FilterConfig::Hflip, vs.clone(), vs.clone());
        let filter_b = NodeSpec::filter("vflip", FilterConfig::Vflip, vs.clone(), vs.clone());

        let merge = NodeSpec::new(
            "merge",
            NodeType::Merge,
            vec![
                ("in0".to_string(), vs.clone()),
                ("in1".to_string(), vs.clone()),
            ],
            vec![("default".to_string(), vs.clone())],
        );
        let sink = NodeSpec::sink("sink", SinkConfig::Null, vs);

        let src_id = g.add_node(src);
        let split_id = g.add_node(split);
        let fa_id = g.add_node(filter_a);
        let fb_id = g.add_node(filter_b);
        let merge_id = g.add_node(merge);
        let sink_id = g.add_node(sink);

        let _ = g.connect(src_id, "default", split_id, "default");
        let _ = g.connect(split_id, "out0", fa_id, "default");
        let _ = g.connect(split_id, "out1", fb_id, "default");
        let _ = g.connect(fa_id, "default", merge_id, "in0");
        let _ = g.connect(fb_id, "default", merge_id, "in1");
        let _ = g.connect(merge_id, "default", sink_id, "default");

        g
    }

    // -- Tests ----------------------------------------------------------------

    #[test]
    fn plan_empty_graph() {
        let g = PipelineGraph::new();
        let plan = ExecutionPlanner::plan(&g).expect("plan should succeed");
        assert_eq!(plan.stage_count(), 0);
        assert!(plan.is_empty());
        assert!(plan.node_order().is_empty());
    }

    #[test]
    fn plan_linear_pipeline() {
        let (g, src_id, filt_id, sink_id) = make_linear_graph();
        let plan = ExecutionPlanner::plan(&g).expect("plan should succeed");

        // 3 nodes at depths 0, 1, 2 -> 3 stages.
        assert_eq!(plan.stage_count(), 3);
        assert!(!plan.is_empty());

        let order = plan.node_order();
        assert_eq!(order.len(), 3);

        // Stage 0 has no deps, stage 1 depends on 0, stage 2 depends on 1.
        assert!(plan.stages[0].dependencies.is_empty());
        assert!(plan.stages[1].dependencies.contains(&0));
        assert!(plan.stages[2].dependencies.contains(&1));

        // All three nodes must appear in the flattened order.
        assert!(order.contains(&src_id));
        assert!(order.contains(&filt_id));
        assert!(order.contains(&sink_id));
    }

    #[test]
    fn plan_branch_merge() {
        let g = make_branch_merge_graph();
        let plan = ExecutionPlanner::plan(&g).expect("plan should succeed");

        // 6 nodes: src(0), split(1), hflip+vflip(2), merge(3), sink(4)
        assert_eq!(plan.node_order().len(), 6);

        // The two filters share the same depth -> same stage.
        let parallel_stage = plan.stages.iter().find(|s| s.nodes.len() == 2);
        assert!(
            parallel_stage.is_some(),
            "expected one stage with two parallel nodes"
        );
    }

    #[test]
    fn plan_single_node() {
        let mut g = PipelineGraph::new();
        let vs = video_spec();
        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), vs);
        let _ = g.add_node(src);

        let plan = ExecutionPlanner::plan(&g).expect("plan should succeed");
        assert_eq!(plan.stage_count(), 1);
        assert_eq!(plan.node_order().len(), 1);
        assert!(plan.stages[0].dependencies.is_empty());
    }

    #[test]
    fn resource_estimate_default() {
        let est = ResourceEstimate::default();
        assert_eq!(est.memory_bytes, 0);
        assert!((est.cpu_weight - 0.0).abs() < f64::EPSILON);
        assert!(!est.is_gpu_candidate);
    }

    #[test]
    fn resource_estimate_merge() {
        let a = ResourceEstimate::new(100, 2.0, false);
        let b = ResourceEstimate::new(200, 3.0, true);
        let c = a.merge(&b);
        assert_eq!(c.memory_bytes, 300);
        assert!((c.cpu_weight - 5.0).abs() < f64::EPSILON);
        assert!(c.is_gpu_candidate);
    }

    #[test]
    fn resource_estimate_per_stage() {
        let (g, ..) = make_linear_graph();
        let plan = ExecutionPlanner::plan(&g).expect("plan should succeed");

        // The scale filter stage should flag GPU candidacy.
        let scale_stage = &plan.stages[1];
        assert!(scale_stage.resource_estimate.is_gpu_candidate);
        assert!(scale_stage.resource_estimate.cpu_weight > 0.0);
        assert!(scale_stage.resource_estimate.memory_bytes > 0);
    }

    #[test]
    fn total_estimate_sums_stages() {
        let (g, ..) = make_linear_graph();
        let plan = ExecutionPlanner::plan(&g).expect("plan should succeed");

        let sum_mem: u64 = plan
            .stages
            .iter()
            .map(|s| s.resource_estimate.memory_bytes)
            .sum();
        assert_eq!(plan.total_estimate.memory_bytes, sum_mem);

        let sum_cpu: f64 = plan
            .stages
            .iter()
            .map(|s| s.resource_estimate.cpu_weight)
            .sum();
        assert!((plan.total_estimate.cpu_weight - sum_cpu).abs() < f64::EPSILON);
    }

    #[test]
    fn optimizer_passthrough_single_stage() {
        let stages = vec![make_stage(
            0,
            vec![NodeId::new()],
            ResourceEstimate::new(1024, 5.0, false),
            vec![],
        )];
        let plan = ExecutionPlan::new(stages);
        let optimized = PipelineOptimizer::optimize(plan.clone());
        assert_eq!(optimized.stage_count(), plan.stage_count());
    }

    #[test]
    fn optimizer_merges_lightweight_stages() {
        // Three lightweight stages (cpu_weight < threshold combined).
        let stages = vec![
            make_stage(
                0,
                vec![NodeId::new()],
                ResourceEstimate::new(1024, 1.0, false),
                vec![],
            ),
            make_stage(
                1,
                vec![NodeId::new()],
                ResourceEstimate::new(1024, 2.0, false),
                vec![0],
            ),
            make_stage(
                2,
                vec![NodeId::new()],
                ResourceEstimate::new(1024, 3.0, false),
                vec![1],
            ),
        ];
        let plan = ExecutionPlan::new(stages);
        let optimized = PipelineOptimizer::optimize(plan);

        // All three fit under the 10.0 threshold (1+2+3=6) -> merged into 1.
        assert_eq!(optimized.stage_count(), 1);
        assert_eq!(optimized.node_order().len(), 3);
        assert!((optimized.total_estimate.cpu_weight - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn optimizer_preserves_heavy_stages() {
        let stages = vec![
            make_stage(
                0,
                vec![NodeId::new()],
                ResourceEstimate::new(1024, 1.0, false),
                vec![],
            ),
            make_stage(
                1,
                vec![NodeId::new()],
                ResourceEstimate::new(4096, 15.0, true),
                vec![0],
            ),
            make_stage(
                2,
                vec![NodeId::new()],
                ResourceEstimate::new(1024, 2.0, false),
                vec![1],
            ),
        ];
        let plan = ExecutionPlan::new(stages);
        let optimized = PipelineOptimizer::optimize(plan);

        // Stage 1 is heavy (15.0) -> cannot merge with stage 0.
        assert!(optimized.stage_count() >= 2);
    }

    #[test]
    fn optimizer_renumbers_stage_ids() {
        let stages = vec![
            make_stage(
                0,
                vec![NodeId::new()],
                ResourceEstimate::new(1024, 1.0, false),
                vec![],
            ),
            make_stage(
                1,
                vec![NodeId::new()],
                ResourceEstimate::new(1024, 2.0, false),
                vec![0],
            ),
            make_stage(
                2,
                vec![NodeId::new()],
                ResourceEstimate::new(4096, 20.0, true),
                vec![1],
            ),
        ];
        let plan = ExecutionPlan::new(stages);
        let optimized = PipelineOptimizer::optimize(plan);

        // Stages 0+1 merge (1+2<10), stage 2 stays -> 2 stages.
        assert_eq!(optimized.stage_count(), 2);
        assert_eq!(optimized.stages[0].stage_id, 0);
        assert_eq!(optimized.stages[1].stage_id, 1);

        // Stage 1 (old stage 2) should depend on stage 0 (merged 0+1).
        assert!(optimized.stages[1].dependencies.contains(&0));
    }

    #[test]
    fn optimizer_on_real_plan() {
        let g = make_branch_merge_graph();
        let plan = ExecutionPlanner::plan(&g).expect("plan should succeed");
        let before = plan.stage_count();
        let optimized = PipelineOptimizer::optimize(plan);

        // Never produce more stages than before.
        assert!(optimized.stage_count() <= before);
        // All nodes still present.
        assert_eq!(optimized.node_order().len(), 6);
    }

    #[test]
    fn execution_plan_new_computes_total() {
        let stages = vec![
            make_stage(
                0,
                vec![NodeId::new()],
                ResourceEstimate::new(100, 1.0, false),
                vec![],
            ),
            make_stage(
                1,
                vec![NodeId::new()],
                ResourceEstimate::new(200, 3.0, true),
                vec![0],
            ),
        ];
        let plan = ExecutionPlan::new(stages);
        assert_eq!(plan.total_estimate.memory_bytes, 300);
        assert!((plan.total_estimate.cpu_weight - 4.0).abs() < f64::EPSILON);
        assert!(plan.total_estimate.is_gpu_candidate);
    }

    // ── Parallel execution tests ─────────────────────────────────────────

    #[test]
    fn parallel_stage_detected_in_branch_merge() {
        let g = make_branch_merge_graph();
        let plan = ExecutionPlanner::plan(&g).expect("plan should succeed");

        // hflip and vflip share the same depth level and come from the same
        // split node -> they share a transitive ancestor -> one group (not
        // independent branches). The `parallel` flag should still be true
        // because there are 2 nodes in the stage.
        let two_node_stage = plan
            .stages
            .iter()
            .find(|s| s.nodes.len() == 2)
            .expect("should have a stage with 2 nodes");
        assert!(two_node_stage.parallel);
    }

    #[test]
    fn independent_sources_are_separate_groups() {
        // Two completely independent pipelines: src1->sink1, src2->sink2
        let mut g = PipelineGraph::new();
        let vs = video_spec();
        let src1 = NodeSpec::source("src1", SourceConfig::File("a.mp4".into()), vs.clone());
        let sink1 = NodeSpec::sink("sink1", SinkConfig::Null, vs.clone());
        let src2 = NodeSpec::source("src2", SourceConfig::File("b.mp4".into()), vs.clone());
        let sink2 = NodeSpec::sink("sink2", SinkConfig::Null, vs);
        let s1 = g.add_node(src1);
        let sk1 = g.add_node(sink1);
        let s2 = g.add_node(src2);
        let sk2 = g.add_node(sink2);
        let _ = g.connect(s1, "default", sk1, "default");
        let _ = g.connect(s2, "default", sk2, "default");

        let plan = ExecutionPlanner::plan(&g).expect("plan should succeed");

        // Sources are at depth 0 -> same stage, but independent branches
        let source_stage = &plan.stages[0];
        assert_eq!(source_stage.nodes.len(), 2);
        assert!(source_stage.parallel);
        // They share no ancestors -> separate groups
        assert_eq!(source_stage.parallel_groups.len(), 2);
    }

    #[test]
    fn parallel_stage_count_and_max_parallelism() {
        let g = make_branch_merge_graph();
        let plan = ExecutionPlanner::plan(&g).expect("plan should succeed");
        // At least one parallel stage (hflip+vflip)
        assert!(plan.parallel_stage_count() >= 1);
        assert!(plan.max_parallelism() >= 1);
    }

    #[test]
    fn transitive_ancestor_detection() {
        // src -> filt_a -> filt_c  (filt_c and filt_d at same depth)
        // src -> filt_b -> filt_d
        // filt_c and filt_d share a transitive ancestor (src) even though
        // their direct predecessors (filt_a, filt_b) are different.
        let mut g = PipelineGraph::new();
        let vs = video_spec();
        let src = NodeSpec::source("src", SourceConfig::File("a.mp4".into()), vs.clone());
        let fa = NodeSpec::filter("fa", FilterConfig::Hflip, vs.clone(), vs.clone());
        let fb = NodeSpec::filter("fb", FilterConfig::Vflip, vs.clone(), vs.clone());
        let fc = NodeSpec::filter("fc", FilterConfig::Hflip, vs.clone(), vs.clone());
        let fd = NodeSpec::filter("fd", FilterConfig::Vflip, vs.clone(), vs.clone());
        let sink_c = NodeSpec::sink("sink_c", SinkConfig::Null, vs.clone());
        let sink_d = NodeSpec::sink("sink_d", SinkConfig::Null, vs);

        let src_id = g.add_node(src);
        // Need a split node to fan out
        let split = NodeSpec::new(
            "split",
            NodeType::Split,
            vec![("default".to_string(), video_spec())],
            vec![
                ("out0".to_string(), video_spec()),
                ("out1".to_string(), video_spec()),
            ],
        );
        let split_id = g.add_node(split);
        let fa_id = g.add_node(fa);
        let fb_id = g.add_node(fb);
        let fc_id = g.add_node(fc);
        let fd_id = g.add_node(fd);
        let sc_id = g.add_node(sink_c);
        let sd_id = g.add_node(sink_d);

        let _ = g.connect(src_id, "default", split_id, "default");
        let _ = g.connect(split_id, "out0", fa_id, "default");
        let _ = g.connect(split_id, "out1", fb_id, "default");
        let _ = g.connect(fa_id, "default", fc_id, "default");
        let _ = g.connect(fb_id, "default", fd_id, "default");
        let _ = g.connect(fc_id, "default", sc_id, "default");
        let _ = g.connect(fd_id, "default", sd_id, "default");

        let plan = ExecutionPlanner::plan(&g).expect("plan should succeed");

        // fc and fd are at the same depth (3). Their direct predecessors
        // (fa, fb) are different, but they share src transitively.
        // With transitive detection, they should be in the SAME group.
        let depth3_stage = plan
            .stages
            .iter()
            .find(|s| s.nodes.len() == 2 && s.nodes.contains(&fc_id) && s.nodes.contains(&fd_id));
        assert!(depth3_stage.is_some(), "fc and fd should be in same stage");
        let stage = depth3_stage.expect("checked above");
        // They share transitive ancestor split_id -> same group
        assert_eq!(
            stage.parallel_groups.len(),
            1,
            "fc and fd share ancestors -> one group"
        );
    }

    // ── Checkpoint tests ─────────────────────────────────────────────────

    #[test]
    fn checkpoint_new_all_pending() {
        let (g, src_id, filt_id, sink_id) = make_linear_graph();
        let cp = PipelineCheckpoint::new(&g, "initial");
        assert_eq!(cp.label(), "initial");
        assert_eq!(cp.active_stage(), 0);
        assert_eq!(cp.pending_nodes().len(), 3);
        assert!(cp.completed_nodes().is_empty());
        assert!(cp.failed_nodes().is_empty());
        assert!(cp.node_state(&src_id).is_ok());
        assert!(cp.node_state(&filt_id).is_ok());
        assert!(cp.node_state(&sink_id).is_ok());
    }

    #[test]
    fn checkpoint_at_stage() {
        let (g, ..) = make_linear_graph();
        let plan = ExecutionPlanner::plan(&g).expect("plan");
        let cp = PipelineCheckpoint::at_stage(&g, &plan, 1, "after_filter");
        assert_eq!(cp.active_stage(), 2);
        // Stages 0 and 1 completed -> 2 nodes completed, 1 pending
        assert_eq!(cp.completed_nodes().len(), 2);
        assert_eq!(cp.pending_nodes().len(), 1);
    }

    #[test]
    fn checkpoint_set_node_state() {
        let (g, src_id, ..) = make_linear_graph();
        let mut cp = PipelineCheckpoint::new(&g, "test");
        assert!(cp
            .set_node_state(&src_id, NodeState::InProgress { progress: 42 })
            .is_ok());
        let state = cp.node_state(&src_id).expect("exists");
        assert_eq!(*state, NodeState::InProgress { progress: 42 });
    }

    #[test]
    fn checkpoint_set_failed() {
        let (g, _, filt_id, _) = make_linear_graph();
        let mut cp = PipelineCheckpoint::new(&g, "test");
        assert!(cp
            .set_node_state(
                &filt_id,
                NodeState::Failed {
                    reason: "oom".into()
                }
            )
            .is_ok());
        assert_eq!(cp.failed_nodes().len(), 1);
    }

    #[test]
    fn checkpoint_resume_plan() {
        let (g, ..) = make_linear_graph();
        let plan = ExecutionPlanner::plan(&g).expect("plan");
        let cp = PipelineCheckpoint::at_stage(&g, &plan, 0, "after_src");
        let resume = cp.resume_plan().expect("resume plan");
        // Original had 3 stages (0,1,2). After completing 0, resume from 1.
        assert!(resume.stage_count() <= 2);
    }

    #[test]
    fn checkpoint_nonexistent_node() {
        let (g, ..) = make_linear_graph();
        let mut cp = PipelineCheckpoint::new(&g, "test");
        let fake = NodeId::new();
        assert!(cp.set_node_state(&fake, NodeState::Completed).is_err());
        assert!(cp.node_state(&fake).is_err());
    }
}
