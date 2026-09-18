//! Evaluation plan for lazy execution of an EinsumGraph.
//!
//! An `EvaluationPlan` captures the topological execution order derived from
//! `DependencyAnalysis`, together with per-node memory estimates and a
//! schedule for releasing intermediate tensors as early as possible.

use std::collections::HashMap;
use tensorlogic_ir::EinsumGraph;

use crate::dependency_analyzer::DependencyAnalysis;

/// Memory estimate for a single graph node.
#[derive(Debug, Clone)]
pub struct NodeMemoryEstimate {
    /// Index of the node in `EinsumGraph::nodes`.
    pub node_index: usize,
    /// Estimated size of the output tensor in bytes.
    pub estimated_bytes: usize,
    /// Shape of the output tensor (if known from `shape_hints`).
    pub output_shape: Option<Vec<usize>>,
}

/// A complete plan for evaluating a graph in a memory-optimal order.
///
/// The plan groups nodes into *levels* (derived from topological analysis) so
/// that all nodes in a single level can be executed without waiting for each
/// other.  It also records which intermediate tensors may be freed after each
/// level, and an estimate of peak simultaneous memory usage.
#[derive(Debug, Clone)]
pub struct EvaluationPlan {
    /// Nodes grouped by topological level (level 0 has no dependencies).
    pub levels: Vec<Vec<usize>>,
    /// Maximum estimated bytes live at the same time.
    pub peak_memory_bytes: usize,
    /// `freeable_after[i]` contains node indices whose output can be freed
    /// once level `i` has finished executing.
    pub freeable_after: Vec<Vec<usize>>,
    /// Per-node memory estimates.
    pub node_estimates: Vec<NodeMemoryEstimate>,
    /// Rough total floating-point operation count (currently a linear
    /// heuristic; replace with cost-model data if available).
    pub estimated_total_flops: u64,
}

impl EvaluationPlan {
    /// Build an evaluation plan for `graph`.
    ///
    /// `shape_hints` may supply `node_index → output_shape` information so
    /// that the plan can estimate memory more accurately.  Pass `None` to get
    /// a plan with zero-byte estimates for all nodes.
    pub fn build(graph: &EinsumGraph, shape_hints: Option<&HashMap<usize, Vec<usize>>>) -> Self {
        if graph.nodes.is_empty() {
            return Self {
                levels: Vec::new(),
                peak_memory_bytes: 0,
                freeable_after: Vec::new(),
                node_estimates: Vec::new(),
                estimated_total_flops: 0,
            };
        }

        let analysis = DependencyAnalysis::analyze(graph);

        // Build per-node memory estimates.
        let node_estimates: Vec<NodeMemoryEstimate> = (0..graph.nodes.len())
            .map(|node_idx| {
                let output_shape = shape_hints.and_then(|h| h.get(&node_idx)).cloned();
                let estimated_bytes = output_shape
                    .as_ref()
                    .map(|s| s.iter().product::<usize>() * 8)
                    .unwrap_or(0);
                NodeMemoryEstimate {
                    node_index: node_idx,
                    estimated_bytes,
                    output_shape,
                }
            })
            .collect();

        // For each level, determine which nodes produced their last consumer in
        // *this* level and can therefore be freed after it.
        let num_levels = analysis.num_levels;
        let mut freeable_after: Vec<Vec<usize>> = vec![Vec::new(); num_levels.max(1)];

        for op in &analysis.operations {
            if op.dependents.is_empty() {
                // No downstream consumers — free after the node's own level.
                let level = op.execution_level.min(num_levels.saturating_sub(1));
                freeable_after[level].push(op.node_index);
            } else {
                // Free after the last level that depends on this node.
                let last_level = op
                    .dependents
                    .iter()
                    .filter_map(|&dep_idx| {
                        analysis.operations.get(dep_idx).map(|d| d.execution_level)
                    })
                    .max()
                    .unwrap_or(op.execution_level);
                let level = last_level.min(num_levels.saturating_sub(1));
                freeable_after[level].push(op.node_index);
            }
        }

        // Estimate peak memory: simulate the live-set across levels.
        let mut live_bytes: usize = 0;
        let mut peak_memory_bytes: usize = 0;
        let mut live_set: HashMap<usize, usize> = HashMap::new(); // node_idx → bytes

        for (level_idx, level_nodes) in analysis.execution_levels.iter().enumerate() {
            // Materialise outputs for this level.
            for &node_idx in level_nodes {
                let bytes = node_estimates
                    .get(node_idx)
                    .map(|e| e.estimated_bytes)
                    .unwrap_or(0);
                live_set.insert(node_idx, bytes);
                live_bytes = live_bytes.saturating_add(bytes);
            }
            if live_bytes > peak_memory_bytes {
                peak_memory_bytes = live_bytes;
            }
            // Free nodes that are no longer needed.
            for &freed_node in &freeable_after[level_idx] {
                if let Some(bytes) = live_set.remove(&freed_node) {
                    live_bytes = live_bytes.saturating_sub(bytes);
                }
            }
        }

        // Simple flop estimate: proportional to total output elements × level count.
        let total_elements: usize = node_estimates
            .iter()
            .map(|e| {
                e.output_shape
                    .as_ref()
                    .map(|s| s.iter().product::<usize>())
                    .unwrap_or(1)
            })
            .sum();
        let estimated_total_flops = (total_elements as u64).saturating_mul(num_levels as u64 + 1);

        Self {
            levels: analysis.execution_levels,
            peak_memory_bytes,
            freeable_after,
            node_estimates,
            estimated_total_flops,
        }
    }

    /// Returns `true` when the peak estimated memory fits within `available_bytes`.
    pub fn can_fit_in_memory(&self, available_bytes: usize) -> bool {
        self.peak_memory_bytes <= available_bytes
    }

    /// Human-readable summary of the plan.
    pub fn summary(&self) -> String {
        format!(
            "EvaluationPlan {{ levels: {}, nodes: {}, peak_memory: {} bytes, flops: {} }}",
            self.levels_count(),
            self.total_nodes(),
            self.peak_memory_bytes,
            self.estimated_total_flops,
        )
    }

    /// Number of topological levels in the plan.
    pub fn levels_count(&self) -> usize {
        self.levels.len()
    }

    /// Total number of nodes across all levels.
    pub fn total_nodes(&self) -> usize {
        self.levels.iter().map(|l| l.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

    fn single_node_graph() -> EinsumGraph {
        let mut g = EinsumGraph::new();
        let a = g.add_tensor("a");
        let b = g.add_tensor("b");
        g.add_input(a).unwrap();
        let node = EinsumNode {
            inputs: vec![a],
            outputs: vec![b],
            op: OpType::ElemUnary {
                op: "relu".to_string(),
            },
            metadata: None,
        };
        g.add_node(node).unwrap();
        g.add_output(b).unwrap();
        g
    }

    fn three_node_chain_graph() -> EinsumGraph {
        // t0 -> node0 -> t1 -> node1 -> t2 -> node2 -> t3
        let mut g = EinsumGraph::new();
        let t0 = g.add_tensor("t0");
        let t1 = g.add_tensor("t1");
        let t2 = g.add_tensor("t2");
        let t3 = g.add_tensor("t3");
        g.add_input(t0).unwrap();
        g.add_node(EinsumNode {
            inputs: vec![t0],
            outputs: vec![t1],
            op: OpType::ElemUnary {
                op: "relu".to_string(),
            },
            metadata: None,
        })
        .unwrap();
        g.add_node(EinsumNode {
            inputs: vec![t1],
            outputs: vec![t2],
            op: OpType::ElemUnary {
                op: "relu".to_string(),
            },
            metadata: None,
        })
        .unwrap();
        g.add_node(EinsumNode {
            inputs: vec![t2],
            outputs: vec![t3],
            op: OpType::ElemUnary {
                op: "relu".to_string(),
            },
            metadata: None,
        })
        .unwrap();
        g.add_output(t3).unwrap();
        g
    }

    #[test]
    fn test_evaluation_plan_empty_graph() {
        let g = EinsumGraph::new();
        let plan = EvaluationPlan::build(&g, None);
        assert_eq!(plan.levels_count(), 0);
        assert_eq!(plan.total_nodes(), 0);
        assert_eq!(plan.peak_memory_bytes, 0);
    }

    #[test]
    fn test_evaluation_plan_single_node() {
        let g = single_node_graph();
        let plan = EvaluationPlan::build(&g, None);
        assert_eq!(plan.total_nodes(), 1);
        assert!(plan.levels_count() >= 1);
    }

    #[test]
    fn test_evaluation_plan_linear_chain() {
        let g = three_node_chain_graph();
        let plan = EvaluationPlan::build(&g, None);
        // A strict dependency chain must have exactly 3 levels
        assert_eq!(plan.levels_count(), 3);
        assert_eq!(plan.total_nodes(), 3);
    }

    #[test]
    fn test_evaluation_plan_can_fit_large_memory() {
        let g = single_node_graph();
        let plan = EvaluationPlan::build(&g, None);
        assert!(plan.can_fit_in_memory(usize::MAX));
    }

    #[test]
    fn test_evaluation_plan_cannot_fit_zero() {
        let g = three_node_chain_graph();
        let mut hints = HashMap::new();
        hints.insert(0_usize, vec![1024_usize]);
        let plan = EvaluationPlan::build(&g, Some(&hints));
        // peak_memory_bytes > 0 due to hints, so 0 bytes is not enough
        assert!(!plan.can_fit_in_memory(0));
    }

    #[test]
    fn test_evaluation_plan_summary_nonempty() {
        let g = single_node_graph();
        let plan = EvaluationPlan::build(&g, None);
        let s = plan.summary();
        assert!(!s.is_empty());
        assert!(s.contains("EvaluationPlan"));
    }
}
