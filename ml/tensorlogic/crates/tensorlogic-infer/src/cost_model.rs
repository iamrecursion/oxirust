//! FLOP and memory cost model for `EinsumGraph`.
//!
//! Provides best-effort estimates of computational cost (FLOPs) and memory
//! footprint for every node in an [`EinsumGraph`], as well as utilities to
//! rank nodes by cost, detect bottlenecks, and produce a cost-aware execution
//! schedule.
//!
//! ## Usage
//!
//! ```rust
//! use tensorlogic_infer::cost_model::{CostModel, CostModelConfig};
//! use tensorlogic_ir::{EinsumGraph, EinsumNode};
//!
//! let mut graph = EinsumGraph::new();
//! let a = graph.add_tensor("A");
//! let b = graph.add_tensor("B");
//! let c = graph.add_tensor("C");
//! graph.add_node(EinsumNode::einsum("ij,jk->ik", vec![a, b], vec![c])).unwrap();
//!
//! let model = CostModel::with_default();
//! let summary = model.estimate_graph(&graph);
//! assert_eq!(summary.num_nodes, 1);
//! ```

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt::Write as FmtWrite;

use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

// ─────────────────────────────────────────────────────────────────────────────
// FlopEstimate
// ─────────────────────────────────────────────────────────────────────────────

/// FLOP estimate for a single node or the entire graph.
///
/// `total_flops = 2 * multiply_adds + activations + comparisons`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlopEstimate {
    /// Number of fused multiply-add operations.
    pub multiply_adds: u64,
    /// Activation function evaluations (exp, tanh, sigmoid, …).
    pub activations: u64,
    /// Comparison operations (max, min, argmax, …).
    pub comparisons: u64,
    /// Pre-computed total: `2 * multiply_adds + activations + comparisons`.
    pub total_flops: u64,
}

impl FlopEstimate {
    /// Create a zero estimate.
    pub fn zero() -> Self {
        FlopEstimate {
            multiply_adds: 0,
            activations: 0,
            comparisons: 0,
            total_flops: 0,
        }
    }

    /// Create an estimate from raw counts; `total_flops` is derived.
    pub fn new(multiply_adds: u64, activations: u64, comparisons: u64) -> Self {
        let total_flops = 2 * multiply_adds + activations + comparisons;
        FlopEstimate {
            multiply_adds,
            activations,
            comparisons,
            total_flops,
        }
    }

    /// Add two estimates together.
    pub fn add(&self, other: &FlopEstimate) -> FlopEstimate {
        FlopEstimate::new(
            self.multiply_adds.saturating_add(other.multiply_adds),
            self.activations.saturating_add(other.activations),
            self.comparisons.saturating_add(other.comparisons),
        )
    }

    /// Scale all counts by `factor`.
    pub fn scale(&self, factor: u64) -> FlopEstimate {
        FlopEstimate::new(
            self.multiply_adds.saturating_mul(factor),
            self.activations.saturating_mul(factor),
            self.comparisons.saturating_mul(factor),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoryCostEstimate
// ─────────────────────────────────────────────────────────────────────────────

/// Memory estimate for a single node.
///
/// Named `MemoryCostEstimate` to avoid collision with
/// [`crate::memory::MemoryEstimate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryCostEstimate {
    /// Bytes needed to hold all input tensors.
    pub input_bytes: u64,
    /// Bytes needed for the output tensor.
    pub output_bytes: u64,
    /// Temporary workspace bytes required during execution.
    pub workspace_bytes: u64,
    /// Peak bytes: `input_bytes + output_bytes + workspace_bytes`.
    pub peak_bytes: u64,
}

impl MemoryCostEstimate {
    /// Create a zero estimate.
    pub fn zero() -> Self {
        MemoryCostEstimate {
            input_bytes: 0,
            output_bytes: 0,
            workspace_bytes: 0,
            peak_bytes: 0,
        }
    }

    /// Create from components; `peak_bytes` is derived.
    pub fn new(input_bytes: u64, output_bytes: u64, workspace_bytes: u64) -> Self {
        let peak_bytes = input_bytes
            .saturating_add(output_bytes)
            .saturating_add(workspace_bytes);
        MemoryCostEstimate {
            input_bytes,
            output_bytes,
            workspace_bytes,
            peak_bytes,
        }
    }

    /// Sum of all byte components.
    pub fn total_bytes(&self) -> u64 {
        self.input_bytes
            .saturating_add(self.output_bytes)
            .saturating_add(self.workspace_bytes)
    }

    /// Add two estimates together.
    pub fn add(&self, other: &MemoryCostEstimate) -> MemoryCostEstimate {
        MemoryCostEstimate::new(
            self.input_bytes.saturating_add(other.input_bytes),
            self.output_bytes.saturating_add(other.output_bytes),
            self.workspace_bytes.saturating_add(other.workspace_bytes),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NodeCostEstimate
// ─────────────────────────────────────────────────────────────────────────────

/// Cost estimate for a single node in the graph.
///
/// Named `NodeCostEstimate` to avoid collision with
/// [`crate::scheduling::NodeCost`].
#[derive(Debug, Clone)]
pub struct NodeCostEstimate {
    /// Index of the node in the graph's `nodes` slice.
    pub node_id: usize,
    /// Human-readable operation name (e.g. `"Einsum(ij,jk->ik)"`).
    pub op_name: String,
    /// Estimated output shape (best-effort; may be a placeholder).
    pub output_shape: Vec<usize>,
    /// FLOP estimate for this node.
    pub flops: FlopEstimate,
    /// Memory estimate for this node.
    pub memory: MemoryCostEstimate,
    /// `true` if this node's `total_flops > graph_avg_flops * 3`.
    pub is_bottleneck: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// GraphCostSummary
// ─────────────────────────────────────────────────────────────────────────────

/// Full cost summary for an [`EinsumGraph`].
#[derive(Debug, Clone)]
pub struct GraphCostSummary {
    /// Per-node cost estimates, in node-index order.
    pub node_costs: Vec<NodeCostEstimate>,
    /// Sum of FLOPs across all nodes.
    pub total_flops: FlopEstimate,
    /// Sum of memory estimates across all nodes.
    pub total_memory: MemoryCostEstimate,
    /// Maximum `peak_bytes` across all nodes.
    pub peak_memory_bytes: u64,
    /// Node indices flagged as bottlenecks.
    pub bottleneck_nodes: Vec<usize>,
    /// Total number of nodes estimated.
    pub num_nodes: usize,
    /// Estimated wall-clock time in nanoseconds; `None` if throughput unknown.
    pub estimated_time_ns: Option<u64>,
}

impl GraphCostSummary {
    /// Format a human-readable table: `node_id | op | shape | flops | mem`.
    pub fn format_table(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "{:<8} | {:<30} | {:<20} | {:<12} | {:<12}",
            "node_id", "op", "shape", "flops", "mem_bytes"
        );
        let _ = writeln!(out, "{}", "-".repeat(90));
        for nc in &self.node_costs {
            let shape_str = format!("{:?}", nc.output_shape);
            let _ = writeln!(
                out,
                "{:<8} | {:<30} | {:<20} | {:<12} | {:<12}",
                nc.node_id,
                truncate_str(&nc.op_name, 30),
                truncate_str(&shape_str, 20),
                nc.flops.total_flops,
                nc.memory.total_bytes(),
            );
        }
        let _ = writeln!(out, "{}", "-".repeat(90));
        let _ = writeln!(
            out,
            "TOTAL{:>3} | {:>30} | {:>20} | {:<12} | {:<12}",
            "",
            "",
            "",
            self.total_flops.total_flops,
            self.total_memory.total_bytes(),
        );
        out
    }

    /// Return the `k` nodes with the highest `total_flops`, sorted descending.
    pub fn top_k_by_flops(&self, k: usize) -> Vec<&NodeCostEstimate> {
        let mut refs: Vec<&NodeCostEstimate> = self.node_costs.iter().collect();
        refs.sort_by_key(|b| std::cmp::Reverse(b.flops.total_flops));
        refs.truncate(k);
        refs
    }

    /// Format a breakdown of memory usage per node.
    pub fn memory_breakdown(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "{:<8} | {:<30} | {:<12} | {:<12} | {:<12} | {:<12}",
            "node_id", "op", "input_B", "output_B", "workspace_B", "peak_B"
        );
        let _ = writeln!(out, "{}", "-".repeat(90));
        for nc in &self.node_costs {
            let _ = writeln!(
                out,
                "{:<8} | {:<30} | {:<12} | {:<12} | {:<12} | {:<12}",
                nc.node_id,
                truncate_str(&nc.op_name, 30),
                nc.memory.input_bytes,
                nc.memory.output_bytes,
                nc.memory.workspace_bytes,
                nc.memory.peak_bytes,
            );
        }
        let _ = writeln!(out, "{}", "-".repeat(90));
        let _ = writeln!(out, "Peak graph memory: {} bytes", self.peak_memory_bytes);
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CostModelConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for cost estimation.
#[derive(Debug, Clone)]
pub struct CostModelConfig {
    /// Bytes per element: 8 for f64, 4 for f32, 2 for bf16/f16.
    pub element_size_bytes: u8,
    /// If `Some(t)`, use `t` GFLOP/s to compute an estimated wall-clock time.
    pub throughput_gflops: Option<f64>,
    /// Shape hints for named tensors: `(tensor_name, shape)`.
    pub assume_shapes: Vec<(String, Vec<usize>)>,
}

impl Default for CostModelConfig {
    fn default() -> Self {
        CostModelConfig {
            element_size_bytes: 8,
            throughput_gflops: None,
            assume_shapes: vec![],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CostModel
// ─────────────────────────────────────────────────────────────────────────────

/// The main cost model; use [`CostModel::estimate_graph`] to get a full
/// [`GraphCostSummary`].
pub struct CostModel {
    config: CostModelConfig,
}

impl CostModel {
    /// Create with the supplied config.
    pub fn new(config: CostModelConfig) -> Self {
        CostModel { config }
    }

    /// Create with default config (f64 elements, no throughput hint).
    pub fn with_default() -> Self {
        CostModel::new(CostModelConfig::default())
    }

    // ── Public helpers ────────────────────────────────────────────────────────

    /// Estimate costs for the entire graph, returning a [`GraphCostSummary`].
    pub fn estimate_graph(&self, graph: &EinsumGraph) -> GraphCostSummary {
        // Build a map of tensor name → shape hint from config.
        let shape_hints: HashMap<&str, &[usize]> = self
            .config
            .assume_shapes
            .iter()
            .map(|(name, shape)| (name.as_str(), shape.as_slice()))
            .collect();

        // First pass: infer shapes for each node.
        // We propagate shapes through the DAG in topological order.
        let topo = kahn_topological_sort(graph);
        let mut tensor_shapes: HashMap<usize, Vec<usize>> = HashMap::new();

        // Seed tensor shapes from hints (match by tensor name).
        for (idx, name) in graph.tensors.iter().enumerate() {
            if let Some(sh) = shape_hints.get(name.as_str()) {
                tensor_shapes.insert(idx, sh.to_vec());
            }
        }

        // Second pass: estimate cost per node in topological order.
        let mut node_costs_map: BTreeMap<usize, NodeCostEstimate> = BTreeMap::new();

        for &node_idx in &topo {
            let node = match graph.nodes.get(node_idx) {
                Some(n) => n,
                None => continue,
            };

            // Gather input shapes (use [1,1] placeholder when unknown).
            let input_shapes: Vec<Vec<usize>> = node
                .inputs
                .iter()
                .map(|&t_idx| {
                    tensor_shapes
                        .get(&t_idx)
                        .cloned()
                        .unwrap_or_else(|| vec![1, 1])
                })
                .collect();

            let nc = self.estimate_node_internal(node_idx, node, &input_shapes);

            // Propagate output shapes.
            for &out_idx in &node.outputs {
                tensor_shapes.insert(out_idx, nc.output_shape.clone());
            }

            node_costs_map.insert(node_idx, nc);
        }

        // Collect in node-index order.
        let node_costs: Vec<NodeCostEstimate> = node_costs_map.into_values().collect();

        // Aggregate totals.
        let mut total_flops = FlopEstimate::zero();
        let mut total_memory = MemoryCostEstimate::zero();
        let mut peak_memory_bytes: u64 = 0;

        for nc in &node_costs {
            total_flops = total_flops.add(&nc.flops);
            total_memory = total_memory.add(&nc.memory);
            if nc.memory.peak_bytes > peak_memory_bytes {
                peak_memory_bytes = nc.memory.peak_bytes;
            }
        }

        // Compute average FLOPs for bottleneck detection.
        let avg_flops = if node_costs.is_empty() {
            0u64
        } else {
            total_flops.total_flops / node_costs.len() as u64
        };
        let bottleneck_threshold = avg_flops.saturating_mul(3);

        // Re-annotate bottlenecks and collect their IDs.
        let mut final_costs: Vec<NodeCostEstimate> = node_costs;
        let mut bottleneck_nodes: Vec<usize> = Vec::new();
        for nc in &mut final_costs {
            if nc.flops.total_flops > bottleneck_threshold {
                nc.is_bottleneck = true;
                bottleneck_nodes.push(nc.node_id);
            }
        }

        // Estimated time.
        let estimated_time_ns = self.config.throughput_gflops.map(|gflops| {
            let total_gflops = total_flops.total_flops as f64 / 1e9;
            let seconds = total_gflops / gflops.max(1e-12);
            (seconds * 1e9) as u64
        });

        GraphCostSummary {
            num_nodes: final_costs.len(),
            node_costs: final_costs,
            total_flops,
            total_memory,
            peak_memory_bytes,
            bottleneck_nodes,
            estimated_time_ns,
        }
    }

    /// Estimate cost for a single node given known input shapes.
    pub fn estimate_node(
        &self,
        node: &EinsumNode,
        input_shapes: &[Vec<usize>],
    ) -> NodeCostEstimate {
        self.estimate_node_internal(0, node, input_shapes)
    }

    /// Estimate FLOPs for an einsum contraction.
    ///
    /// Strategy: multiply all unique index dimension sizes together.  That
    /// product is the number of multiply-add operations.
    pub fn estimate_einsum_flops(equation: &str, input_shapes: &[Vec<usize>]) -> FlopEstimate {
        // Parse the equation: "ab,bc->ac" style.
        // We build a map from index character → known dimension size.
        let parts: Vec<&str> = equation.splitn(2, "->").collect();
        let lhs = parts.first().copied().unwrap_or("");

        let input_specs: Vec<&str> = lhs.split(',').collect();
        let mut index_sizes: HashMap<char, usize> = HashMap::new();

        for (spec, shape) in input_specs.iter().zip(input_shapes.iter()) {
            for (ch, &dim) in spec.chars().zip(shape.iter()) {
                // Use the maximum seen size for a given index (conservative).
                let entry = index_sizes.entry(ch).or_insert(0);
                if dim > *entry {
                    *entry = dim;
                }
            }
        }

        // Product of all index sizes = number of multiply-adds.
        let multiply_adds: u64 = index_sizes
            .values()
            .map(|&s| s as u64)
            .fold(1u64, u64::saturating_mul);

        // If we found no indices at all, treat it as a trivial scalar op.
        let multiply_adds = if index_sizes.is_empty() {
            1
        } else {
            multiply_adds
        };

        FlopEstimate::new(multiply_adds, 0, 0)
    }

    /// Estimate FLOPs for an [`OpType`] given input/output shapes.
    fn estimate_op_flops(
        &self,
        op: &OpType,
        input_shapes: &[Vec<usize>],
        output_shape: &[usize],
    ) -> FlopEstimate {
        match op {
            OpType::Einsum { spec } => Self::estimate_einsum_flops(spec, input_shapes),
            OpType::ElemUnary { op } => {
                // Output size = number of activations.
                let n: u64 = output_shape.iter().map(|&d| d as u64).product();
                let n = n.max(1);
                match op.as_str() {
                    "relu" | "neg" | "abs" | "sign" | "floor" | "ceil" | "round" => {
                        // Simple ops: 1 op each, categorised as comparison/comparison-like.
                        FlopEstimate::new(0, 0, n)
                    }
                    "exp" | "log" | "sqrt" | "rsqrt" | "sigmoid" | "tanh" | "gelu" | "silu"
                    | "sin" | "cos" | "tan" | "erf" => {
                        // Transcendental: count as activations.
                        FlopEstimate::new(0, n, 0)
                    }
                    _ => {
                        // Unknown unary: assume one multiply-add per element.
                        FlopEstimate::new(n, 0, 0)
                    }
                }
            }
            OpType::ElemBinary { op } => {
                let n: u64 = output_shape.iter().map(|&d| d as u64).product();
                let n = n.max(1);
                match op.as_str() {
                    "add" | "sub" | "mul" | "div" => FlopEstimate::new(n, 0, 0),
                    "max" | "min" | "gt" | "lt" | "ge" | "le" | "eq" | "ne" => {
                        FlopEstimate::new(0, 0, n)
                    }
                    _ => FlopEstimate::new(n, 0, 0),
                }
            }
            OpType::Reduce { op, axes } => {
                // For a reduction: multiply_adds = input_elements (one add per input elem).
                let input_shape = input_shapes
                    .first()
                    .map(|s| s.as_slice())
                    .unwrap_or(&[1, 1]);
                let input_elements: u64 = input_shape.iter().map(|&d| d as u64).product();
                let input_elements = input_elements.max(1);

                // Number of axes reduced over (to estimate reduction depth).
                let n_axes = axes.len().max(1);
                match op.as_str() {
                    "sum" | "mean" => FlopEstimate::new(input_elements, 0, 0),
                    "max" | "min" | "argmax" | "argmin" => {
                        FlopEstimate::new(0, 0, input_elements * n_axes as u64)
                    }
                    "prod" => FlopEstimate::new(input_elements, 0, 0),
                    _ => FlopEstimate::new(input_elements, 0, 0),
                }
            }
        }
    }

    /// Infer the output shape for a node given input shapes.
    ///
    /// For einsum, parses the equation to determine output dimension sizes.
    /// For other ops, attempts shape propagation heuristics.  Falls back to a
    /// non-empty placeholder `[1]` when inference is not possible.
    pub fn infer_output_shape(node: &EinsumNode, input_shapes: &[Vec<usize>]) -> Vec<usize> {
        match &node.op {
            OpType::Einsum { spec } => infer_einsum_output_shape(spec, input_shapes),
            OpType::ElemUnary { .. } => {
                // Output has the same shape as the input.
                input_shapes.first().cloned().unwrap_or_else(|| vec![1])
            }
            OpType::ElemBinary { .. } => {
                // Output: broadcast shape (simplified: max of each dimension).
                broadcast_shapes(input_shapes)
            }
            OpType::Reduce { axes, .. } => {
                let input = input_shapes.first().map(|s| s.as_slice()).unwrap_or(&[1]);
                reduce_output_shape(input, axes)
            }
        }
    }

    /// Sort nodes by descending FLOP cost.
    pub fn rank_by_flops(summary: &GraphCostSummary) -> Vec<&NodeCostEstimate> {
        let mut refs: Vec<&NodeCostEstimate> = summary.node_costs.iter().collect();
        refs.sort_by_key(|b| std::cmp::Reverse(b.flops.total_flops));
        refs
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn estimate_node_internal(
        &self,
        node_idx: usize,
        node: &EinsumNode,
        input_shapes: &[Vec<usize>],
    ) -> NodeCostEstimate {
        let output_shape = Self::infer_output_shape(node, input_shapes);
        let flops = self.estimate_op_flops(&node.op, input_shapes, &output_shape);
        let memory = self.estimate_memory(input_shapes, &output_shape);
        let op_name = node.operation_description();

        NodeCostEstimate {
            node_id: node_idx,
            op_name,
            output_shape,
            flops,
            memory,
            is_bottleneck: false, // set in the graph-level pass
        }
    }

    fn estimate_memory(
        &self,
        input_shapes: &[Vec<usize>],
        output_shape: &[usize],
    ) -> MemoryCostEstimate {
        let elem = self.config.element_size_bytes as u64;

        let input_bytes: u64 = input_shapes
            .iter()
            .map(|sh| {
                sh.iter()
                    .map(|&d| d as u64)
                    .product::<u64>()
                    .saturating_mul(elem)
            })
            .fold(0u64, u64::saturating_add);

        let output_bytes: u64 = output_shape
            .iter()
            .map(|&d| d as u64)
            .product::<u64>()
            .saturating_mul(elem);

        // Workspace: heuristic – 50% of the larger of input or output.
        let workspace_bytes = input_bytes.max(output_bytes) / 2;

        MemoryCostEstimate::new(input_bytes, output_bytes, workspace_bytes)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CostAwareSchedule
// ─────────────────────────────────────────────────────────────────────────────

/// A topologically valid execution order that puts expensive operations early
/// (useful for parallelism analysis).
#[derive(Debug, Clone)]
pub struct CostAwareSchedule {
    /// Node IDs in execution order.
    pub order: Vec<usize>,
    /// Total FLOPs along the critical path.
    pub critical_path_flops: u64,
    /// Parallelism score ∈ \[0, 1\]; higher means more operations can run in
    /// parallel relative to total work.
    pub parallelism_score: f64,
}

impl CostAwareSchedule {
    /// Compute a topologically-valid order that places expensive ops early.
    ///
    /// Uses Kahn's BFS algorithm internally, with a max-FLOP tie-breaking rule
    /// so that heavy operations bubble to the front within each ready frontier.
    pub fn from_graph(graph: &EinsumGraph, summary: &GraphCostSummary) -> Self {
        // Build a cost lookup: node_id → total_flops.
        let flop_map: HashMap<usize, u64> = summary
            .node_costs
            .iter()
            .map(|nc| (nc.node_id, nc.flops.total_flops))
            .collect();

        // Build adjacency and in-degree from DAG edges.
        let n = graph.nodes.len();
        let mut in_degree = vec![0usize; n];
        // tensor_produced_by: tensor_idx → node_idx
        let mut produced_by: HashMap<usize, usize> = HashMap::new();
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            for &out_t in &node.outputs {
                produced_by.insert(out_t, node_idx);
            }
        }

        // Build in-degree: node X depends on node Y if Y produces a tensor
        // that X consumes.
        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            for &in_t in &node.inputs {
                if let Some(&pred_node) = produced_by.get(&in_t) {
                    if pred_node != node_idx {
                        in_degree[node_idx] += 1;
                        predecessors[node_idx].push(pred_node);
                    }
                }
            }
        }

        // Deduplicate and recompute in_degree from unique predecessors.
        for (node_idx, preds) in predecessors.iter_mut().enumerate() {
            preds.sort_unstable();
            preds.dedup();
            in_degree[node_idx] = preds.len();
        }

        // Build successor list.
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (node_idx, preds) in predecessors.iter().enumerate() {
            for &pred in preds {
                successors[pred].push(node_idx);
            }
        }

        // Kahn's BFS with FLOP-descending priority within the ready frontier.
        let mut ready: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        ready.sort_by(|&a, &b| {
            flop_map
                .get(&b)
                .unwrap_or(&0)
                .cmp(flop_map.get(&a).unwrap_or(&0))
        });

        let mut order: Vec<usize> = Vec::with_capacity(n);
        let mut remaining_in_degree = in_degree;

        while !ready.is_empty() {
            // Sort by descending FLOPs.
            ready.sort_by(|&a, &b| {
                flop_map
                    .get(&b)
                    .unwrap_or(&0)
                    .cmp(flop_map.get(&a).unwrap_or(&0))
            });
            let node_idx = ready.remove(0);
            order.push(node_idx);

            for &succ in &successors[node_idx] {
                remaining_in_degree[succ] = remaining_in_degree[succ].saturating_sub(1);
                if remaining_in_degree[succ] == 0 {
                    ready.push(succ);
                }
            }
        }

        // Append any nodes not reached (e.g. isolated nodes or cycles).
        for i in 0..n {
            if !order.contains(&i) {
                order.push(i);
            }
        }

        // Critical path: longest FLOP-weighted path.
        let critical_path_flops = compute_critical_path_flops(graph, &flop_map);

        // Parallelism score: ratio of critical-path flops to total flops.
        let total_flops = summary.total_flops.total_flops;
        let parallelism_score = if total_flops == 0 {
            1.0
        } else {
            let serial_fraction = critical_path_flops as f64 / total_flops as f64;
            (1.0 - serial_fraction).clamp(0.0, 1.0)
        };

        CostAwareSchedule {
            order,
            critical_path_flops,
            parallelism_score,
        }
    }

    /// Format the schedule as a human-readable table.
    pub fn format_schedule(&self, summary: &GraphCostSummary) -> String {
        let cost_map: HashMap<usize, &NodeCostEstimate> = summary
            .node_costs
            .iter()
            .map(|nc| (nc.node_id, nc))
            .collect();

        let mut out = String::new();
        let _ = writeln!(
            out,
            "{:<6} | {:<8} | {:<30} | {:<14} | bottleneck",
            "step", "node_id", "op", "flops"
        );
        let _ = writeln!(out, "{}", "-".repeat(70));
        for (step, &nid) in self.order.iter().enumerate() {
            let (op_name, flops, is_bn) = cost_map
                .get(&nid)
                .map(|nc| (nc.op_name.as_str(), nc.flops.total_flops, nc.is_bottleneck))
                .unwrap_or(("?", 0, false));
            let _ = writeln!(
                out,
                "{:<6} | {:<8} | {:<30} | {:<14} | {}",
                step,
                nid,
                truncate_str(op_name, 30),
                flops,
                if is_bn { "YES" } else { "no" },
            );
        }
        let _ = writeln!(out, "{}", "-".repeat(70));
        let _ = writeln!(out, "Critical-path FLOPs: {}", self.critical_path_flops);
        let _ = writeln!(out, "Parallelism score  : {:.4}", self.parallelism_score);
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Kahn's algorithm for topological sort.  Returns node indices in a valid
/// execution order (nodes with no incoming edges first).
fn kahn_topological_sort(graph: &EinsumGraph) -> Vec<usize> {
    let n = graph.nodes.len();
    if n == 0 {
        return vec![];
    }

    // tensor_produced_by: tensor_idx → node_idx.
    let mut produced_by: HashMap<usize, usize> = HashMap::new();
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &out_t in &node.outputs {
            produced_by.insert(out_t, node_idx);
        }
    }

    // in_degree[i] = number of nodes that must complete before node i.
    let mut in_degree = vec![0usize; n];
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (node_idx, node) in graph.nodes.iter().enumerate() {
        let mut unique_preds: Vec<usize> = node
            .inputs
            .iter()
            .filter_map(|&t| produced_by.get(&t).copied())
            .filter(|&pred| pred != node_idx)
            .collect();
        unique_preds.sort_unstable();
        unique_preds.dedup();
        in_degree[node_idx] = unique_preds.len();
        for pred in unique_preds {
            successors[pred].push(node_idx);
        }
    }

    let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut order = Vec::with_capacity(n);

    while let Some(idx) = queue.pop_front() {
        order.push(idx);
        for &succ in &successors[idx] {
            in_degree[succ] = in_degree[succ].saturating_sub(1);
            if in_degree[succ] == 0 {
                queue.push_back(succ);
            }
        }
    }

    // Append remaining (handles cycles gracefully).
    for i in 0..n {
        if !order.contains(&i) {
            order.push(i);
        }
    }

    order
}

/// Compute the longest FLOP-weighted path through the graph (critical path).
fn compute_critical_path_flops(graph: &EinsumGraph, flop_map: &HashMap<usize, u64>) -> u64 {
    let n = graph.nodes.len();
    if n == 0 {
        return 0;
    }

    let topo = kahn_topological_sort(graph);

    let mut produced_by: HashMap<usize, usize> = HashMap::new();
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        for &out_t in &node.outputs {
            produced_by.insert(out_t, node_idx);
        }
    }

    // dp[i] = max cumulative FLOPs to reach and finish node i.
    let mut dp = vec![0u64; n];

    for &node_idx in &topo {
        let node = match graph.nodes.get(node_idx) {
            Some(n) => n,
            None => continue,
        };
        let self_flops = *flop_map.get(&node_idx).unwrap_or(&0);

        let max_pred: u64 = node
            .inputs
            .iter()
            .filter_map(|&t| produced_by.get(&t))
            .filter(|&&pred| pred != node_idx)
            .map(|&pred| *dp.get(pred).unwrap_or(&0))
            .max()
            .unwrap_or(0);

        dp[node_idx] = max_pred.saturating_add(self_flops);
    }

    *dp.iter().max().unwrap_or(&0)
}

/// Infer the output shape for an einsum equation given input shapes.
fn infer_einsum_output_shape(spec: &str, input_shapes: &[Vec<usize>]) -> Vec<usize> {
    let parts: Vec<&str> = spec.splitn(2, "->").collect();
    let lhs = parts.first().copied().unwrap_or("");
    let rhs = parts.get(1).copied().unwrap_or("");

    let input_specs: Vec<&str> = lhs.split(',').collect();

    // Build index → size map.
    let mut index_sizes: HashMap<char, usize> = HashMap::new();
    for (spec_part, shape) in input_specs.iter().zip(input_shapes.iter()) {
        for (ch, &dim) in spec_part.chars().zip(shape.iter()) {
            let entry = index_sizes.entry(ch).or_insert(0);
            if dim > *entry {
                *entry = dim;
            }
        }
    }

    if rhs.is_empty() {
        // Scalar output.
        return vec![1];
    }

    let output_shape: Vec<usize> = rhs
        .chars()
        .map(|ch| *index_sizes.get(&ch).unwrap_or(&1))
        .collect();

    if output_shape.is_empty() {
        vec![1]
    } else {
        output_shape
    }
}

/// Simple element-wise broadcast: take the maximum size for each position.
fn broadcast_shapes(shapes: &[Vec<usize>]) -> Vec<usize> {
    if shapes.is_empty() {
        return vec![1];
    }
    let max_rank = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut result = vec![1usize; max_rank];
    for shape in shapes {
        let offset = max_rank - shape.len();
        for (i, &d) in shape.iter().enumerate() {
            let pos = offset + i;
            if d > result[pos] {
                result[pos] = d;
            }
        }
    }
    result
}

/// Compute the output shape after reducing `axes` from `input_shape`.
fn reduce_output_shape(input_shape: &[usize], axes: &[usize]) -> Vec<usize> {
    input_shape
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| if axes.contains(&i) { None } else { Some(d) })
        .collect::<Vec<_>>()
        .into_iter()
        .chain(std::iter::once(1)) // ensure non-empty
        .take(input_shape.len().max(1))
        .collect()
}

/// Truncate `s` to `max_len` characters, appending `…` if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_owned()
    } else {
        format!("{}…", &s[..max_len.saturating_sub(1)])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{EinsumGraph, EinsumNode};

    // ── FlopEstimate ──────────────────────────────────────────────────────────

    #[test]
    fn test_flop_estimate_zero() {
        let f = FlopEstimate::zero();
        assert_eq!(f.multiply_adds, 0);
        assert_eq!(f.activations, 0);
        assert_eq!(f.comparisons, 0);
        assert_eq!(f.total_flops, 0);
    }

    #[test]
    fn test_flop_estimate_add() {
        let a = FlopEstimate::new(10, 2, 3);
        let b = FlopEstimate::new(5, 1, 1);
        let c = a.add(&b);
        assert_eq!(c.multiply_adds, 15);
        assert_eq!(c.activations, 3);
        assert_eq!(c.comparisons, 4);
    }

    #[test]
    fn test_flop_estimate_total_flops() {
        // total_flops = 2 * multiply_adds + activations + comparisons
        let f = FlopEstimate::new(10, 3, 5);
        assert_eq!(f.total_flops, 2 * 10 + 3 + 5);
    }

    // ── MemoryCostEstimate ────────────────────────────────────────────────────

    #[test]
    fn test_memory_estimate_zero() {
        let m = MemoryCostEstimate::zero();
        assert_eq!(m.input_bytes, 0);
        assert_eq!(m.output_bytes, 0);
        assert_eq!(m.workspace_bytes, 0);
        assert_eq!(m.peak_bytes, 0);
    }

    #[test]
    fn test_memory_estimate_total() {
        let m = MemoryCostEstimate::new(100, 200, 50);
        assert!(m.total_bytes() > 0);
        assert_eq!(m.total_bytes(), 350);
        assert_eq!(m.peak_bytes, 350);
    }

    // ── CostModel construction ────────────────────────────────────────────────

    #[test]
    fn test_cost_model_with_default() {
        let model = CostModel::with_default();
        assert_eq!(model.config.element_size_bytes, 8);
        assert!(model.config.throughput_gflops.is_none());
    }

    // ── estimate_einsum_flops ─────────────────────────────────────────────────

    #[test]
    fn test_estimate_einsum_flops_simple() {
        // "ij,jk->ik" with shapes [2,3] and [3,4]
        // Indices: i=2, j=3, k=4  →  multiply_adds = 2*3*4 = 24
        let flops = CostModel::estimate_einsum_flops("ij,jk->ik", &[vec![2, 3], vec![3, 4]]);
        assert_eq!(flops.multiply_adds, 24);
        assert_eq!(flops.total_flops, 48); // 2 * 24
    }

    // ── infer_output_shape ────────────────────────────────────────────────────

    #[test]
    fn test_infer_output_shape_placeholder() {
        let node = EinsumNode::elem_unary("relu", 0, 1);
        let shape = CostModel::infer_output_shape(&node, &[vec![3, 4]]);
        assert!(!shape.is_empty());
    }

    // ── GraphCostSummary formatting ───────────────────────────────────────────

    fn make_single_node_graph() -> EinsumGraph {
        let mut g = EinsumGraph::new();
        let a = g.add_tensor("A");
        let b = g.add_tensor("B");
        let c = g.add_tensor("C");
        g.add_node(EinsumNode::einsum("ij,jk->ik", vec![a, b], vec![c]))
            .expect("add_node");
        g
    }

    #[test]
    fn test_graph_cost_summary_format_table() {
        let g = make_single_node_graph();
        let model = CostModel::with_default();
        let summary = model.estimate_graph(&g);
        let table = summary.format_table();
        assert!(!table.is_empty());
        // Should contain a header with "node_id".
        assert!(table.contains("node_id"));
    }

    #[test]
    fn test_graph_cost_summary_memory_breakdown() {
        let g = make_single_node_graph();
        let model = CostModel::with_default();
        let summary = model.estimate_graph(&g);
        let bd = summary.memory_breakdown();
        assert!(!bd.is_empty());
        assert!(bd.contains("node_id"));
    }

    // ── top_k_by_flops / rank_by_flops ───────────────────────────────────────

    #[test]
    fn test_top_k_by_flops() {
        let mut g = EinsumGraph::new();
        let a = g.add_tensor("A");
        let b = g.add_tensor("B");
        let c = g.add_tensor("C");
        let d = g.add_tensor("D");
        let e = g.add_tensor("E");
        // Node 0: big matmul
        g.add_node(EinsumNode::einsum("ij,jk->ik", vec![a, b], vec![c]))
            .expect("n0");
        // Node 1: small unary
        g.add_node(EinsumNode::elem_unary("relu", c, d))
            .expect("n1");
        // Node 2: medium binary
        g.add_node(EinsumNode::elem_binary("add", c, d, e))
            .expect("n2");

        let config = CostModelConfig {
            assume_shapes: vec![("A".into(), vec![4, 8]), ("B".into(), vec![8, 16])],
            ..Default::default()
        };
        let model = CostModel::new(config);
        let summary = model.estimate_graph(&g);

        let top1 = summary.top_k_by_flops(1);
        assert_eq!(top1.len(), 1);
        // top1 should have the most FLOPs.
        let max_flops = summary
            .node_costs
            .iter()
            .map(|nc| nc.flops.total_flops)
            .max()
            .unwrap_or(0);
        assert_eq!(top1[0].flops.total_flops, max_flops);
    }

    #[test]
    fn test_rank_by_flops_sorted() {
        let mut g = EinsumGraph::new();
        let a = g.add_tensor("A");
        let b = g.add_tensor("B");
        let c = g.add_tensor("C");
        let d = g.add_tensor("D");
        g.add_node(EinsumNode::einsum("ij,jk->ik", vec![a, b], vec![c]))
            .expect("n0");
        g.add_node(EinsumNode::elem_unary("relu", c, d))
            .expect("n1");

        let model = CostModel::with_default();
        let summary = model.estimate_graph(&g);
        let ranked = CostModel::rank_by_flops(&summary);
        for w in ranked.windows(2) {
            assert!(w[0].flops.total_flops >= w[1].flops.total_flops);
        }
    }

    // ── empty / single / multi node graphs ───────────────────────────────────

    #[test]
    fn test_cost_model_estimate_graph_empty() {
        let g = EinsumGraph::new();
        let model = CostModel::with_default();
        let summary = model.estimate_graph(&g);
        assert_eq!(summary.num_nodes, 0);
        assert_eq!(summary.total_flops.total_flops, 0);
    }

    #[test]
    fn test_cost_model_estimate_graph_single_node() {
        let g = make_single_node_graph();
        let model = CostModel::with_default();
        let summary = model.estimate_graph(&g);
        assert_eq!(summary.num_nodes, 1);
        assert_eq!(summary.node_costs.len(), 1);
    }

    #[test]
    fn test_cost_model_estimate_graph_multi_node() {
        let mut g = EinsumGraph::new();
        let a = g.add_tensor("A");
        let b = g.add_tensor("B");
        let c = g.add_tensor("C");
        let d = g.add_tensor("D");
        let e = g.add_tensor("E");
        g.add_node(EinsumNode::einsum("ij,jk->ik", vec![a, b], vec![c]))
            .expect("n0");
        g.add_node(EinsumNode::elem_unary("relu", c, d))
            .expect("n1");
        g.add_node(EinsumNode::reduce("sum", vec![1], d, e))
            .expect("n2");
        let model = CostModel::with_default();
        let summary = model.estimate_graph(&g);
        assert_eq!(summary.num_nodes, 3);
    }

    // ── CostAwareSchedule ─────────────────────────────────────────────────────

    fn make_chain_graph() -> EinsumGraph {
        // A → B → C (chain of unary ops)
        let mut g = EinsumGraph::new();
        let a = g.add_tensor("A");
        let b = g.add_tensor("B");
        let c = g.add_tensor("C");
        g.add_node(EinsumNode::elem_unary("relu", a, b))
            .expect("n0");
        g.add_node(EinsumNode::elem_unary("exp", b, c)).expect("n1");
        g
    }

    #[test]
    fn test_cost_aware_schedule_topological_order() {
        let g = make_chain_graph();
        let model = CostModel::with_default();
        let summary = model.estimate_graph(&g);
        let sched = CostAwareSchedule::from_graph(&g, &summary);

        // Both nodes must appear exactly once.
        assert_eq!(sched.order.len(), 2);
        // Node 0 produces tensor B; node 1 consumes it – so 0 must come before 1.
        let pos0 = sched.order.iter().position(|&x| x == 0).unwrap_or(100);
        let pos1 = sched.order.iter().position(|&x| x == 1).unwrap_or(100);
        assert!(pos0 < pos1, "node 0 must precede node 1 in schedule");
    }

    #[test]
    fn test_cost_aware_schedule_format_schedule() {
        let g = make_chain_graph();
        let model = CostModel::with_default();
        let summary = model.estimate_graph(&g);
        let sched = CostAwareSchedule::from_graph(&g, &summary);
        let txt = sched.format_schedule(&summary);
        assert!(!txt.is_empty());
        assert!(txt.contains("step"));
    }

    // ── Bottleneck detection ──────────────────────────────────────────────────

    #[test]
    fn test_bottleneck_detection() {
        // Create a graph where one node's FLOPs > 3 * average.
        // Strategy: big matmul (i=100,j=100,k=100 → 2_000_000 flops) alongside
        // tiny scalar unary ops on "S" (shape [1]).
        // FLOPs: matmul=2_000_000, relu_s≈1, exp_s≈1.
        // avg = (2_000_000+1+1)/3 ≈ 666_667; threshold = 3*666_667 = 2_000_001.
        // Matmul (2_000_000) is just under. Use 200×200 to be safe:
        //   i=200,j=200,k=200 → ma=8_000_000, flops=16_000_000
        //   avg=(16_000_000+1+1)/3≈5_333_334; threshold≈16_000_002.
        // Hmm still borderline. Use 100×100×100 matmul + 2 scalar ops (1 flop each).
        //   matmul flops = 2*1_000_000 = 2_000_000
        //   scalar total = 2
        //   avg = 2_000_002/3 = 667_000; threshold = 3*667_000 = 2_001_000
        //   matmul < threshold. Instead force tiny ops by using "S" shape [1]:
        //   relu on [1]: comparisons=1, flops=1; exp on [1]: activations=1, flops=1
        //   avg = (2_000_000+1+1)/3 ≈ 666_667; threshold = 2_000_001; matmul=2_000_000 → NOT flagged
        //
        // Solution: use a larger matmul (500×500×500 → ma=125_000_000, flops=250_000_000)
        //   so avg=(250_000_000+1+1)/3≈83_333_334; threshold=250_000_002; flops > threshold? no.
        //
        // The bottleneck condition is strictly >, so matmul must be > 3 * average.
        // With 3 nodes: avg = (M+a+b)/3 where a,b are small.
        // M > 3*(M+a+b)/3  =>  M > M+a+b  =>  0 > a+b  -- IMPOSSIBLE with 3 equal-weight nodes.
        //
        // With 4 nodes and a,b,c << M:
        //   avg = (M+a+b+c)/4; M > 3*(M+a+b+c)/4  =>  4M > 3M+3(a+b+c)  =>  M > 3(a+b+c).
        // So with a=b=c=1, M > 9.  Use M=10_000.
        let mut g = EinsumGraph::new();
        let a = g.add_tensor("A"); // 100×100
        let b = g.add_tensor("B"); // 100×100
        let s = g.add_tensor("S"); // scalar [1]
        let c = g.add_tensor("C"); // matmul output
        let d = g.add_tensor("D");
        let e = g.add_tensor("E");
        let f = g.add_tensor("F");

        // Node 0: big matmul
        g.add_node(EinsumNode::einsum("ij,jk->ik", vec![a, b], vec![c]))
            .expect("matmul");
        // Node 1,2,3: tiny scalar unary ops
        g.add_node(EinsumNode::elem_unary("relu", s, d))
            .expect("relu");
        g.add_node(EinsumNode::elem_unary("exp", s, e))
            .expect("exp");
        g.add_node(EinsumNode::elem_unary("neg", s, f))
            .expect("neg");

        let config = CostModelConfig {
            assume_shapes: vec![
                ("A".into(), vec![100, 100]),
                ("B".into(), vec![100, 100]),
                ("S".into(), vec![1]),
            ],
            ..Default::default()
        };
        let model = CostModel::new(config);
        let summary = model.estimate_graph(&g);

        // matmul flops = 2 * (100*100*100) = 2_000_000
        // relu/exp/neg on [1]: each has 1 flop
        // avg = (2_000_000 + 1 + 1 + 1) / 4 = 500_000 (approx)
        // threshold = 3 * 500_000 = 1_500_000
        // matmul (2_000_000) > threshold (1_500_000) → bottleneck
        assert!(
            summary.bottleneck_nodes.contains(&0),
            "matmul node must be a bottleneck; bottlenecks: {:?}, node_costs: {:?}",
            summary.bottleneck_nodes,
            summary
                .node_costs
                .iter()
                .map(|nc| (nc.node_id, nc.flops.total_flops))
                .collect::<Vec<_>>()
        );
    }

    // ── Config ────────────────────────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = CostModelConfig::default();
        assert_eq!(cfg.element_size_bytes, 8);
        assert!(cfg.throughput_gflops.is_none());
    }

    // ── Throughput / time estimate ────────────────────────────────────────────

    #[test]
    fn test_throughput_time_estimate() {
        let g = make_single_node_graph();
        let config = CostModelConfig {
            throughput_gflops: Some(10.0), // 10 GFLOP/s
            assume_shapes: vec![("A".into(), vec![4, 4]), ("B".into(), vec![4, 4])],
            ..Default::default()
        };
        let model = CostModel::new(config);
        let summary = model.estimate_graph(&g);
        assert!(
            summary.estimated_time_ns.is_some(),
            "estimated_time_ns must be Some when throughput is set"
        );
    }
}
