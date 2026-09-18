//! Lazy wrapper around an `EinsumGraph` that tracks per-node computation state.

use tensorlogic_ir::EinsumGraph;

use crate::Scirs2Tensor;

use super::plan::EvaluationPlan;
use super::tensor::LazyTensor;

/// A lazily-evaluated view of an [`EinsumGraph`].
///
/// Each node in the graph is represented by a [`LazyTensor`] that starts in
/// the *pending* state.  As the execution backend computes nodes it can call
/// [`LazyTensor::set`] to store the result; subsequent reads of the same node
/// return the cached value without re-computation.
///
/// The graph reference is borrowed immutably so that `LazyEinsumGraph` can be
/// used alongside the original graph without copying it.
pub struct LazyEinsumGraph<'g> {
    /// The underlying EinsumGraph (borrowed).
    pub graph: &'g EinsumGraph,
    /// One lazy tensor per *node* in the graph (indexed by node index).
    pub tensors: Vec<LazyTensor<Scirs2Tensor>>,
    /// Cached evaluation plan (computed on demand by [`build_plan`]).
    plan: Option<EvaluationPlan>,
}

impl<'g> LazyEinsumGraph<'g> {
    /// Create a new `LazyEinsumGraph` with all nodes in the *pending* state.
    pub fn new(graph: &'g EinsumGraph) -> Self {
        let tensors = (0..graph.nodes.len())
            .map(|_| LazyTensor::pending(None))
            .collect();
        Self {
            graph,
            tensors,
            plan: None,
        }
    }

    /// Returns `true` when every node in the graph has been computed.
    pub fn is_fully_evaluated(&self) -> bool {
        self.tensors.iter().all(|t| t.is_computed())
    }

    /// Return the output tensor of the graph, if it has been computed.
    ///
    /// Uses the first entry of `EinsumGraph::outputs` as the output node.
    /// Note that `outputs` contains *tensor* indices while `tensors` is
    /// indexed by *node* index; for a typical compiled graph the output
    /// tensor index maps to the last node, so we look for the node whose
    /// output index matches.
    pub fn get_output(&self) -> Option<Scirs2Tensor> {
        let output_tensor_idx = *self.graph.outputs.first()?;

        // Find the node whose output list contains `output_tensor_idx`.
        let node_idx = self
            .graph
            .nodes
            .iter()
            .enumerate()
            .find(|(_, node)| node.outputs.contains(&output_tensor_idx))
            .map(|(idx, _)| idx)?;

        self.tensors.get(node_idx)?.get()
    }

    /// Clear all computed tensors back to the *pending* state and discard the
    /// cached evaluation plan.  Useful when input tensors change.
    pub fn invalidate(&mut self) {
        for t in &self.tensors {
            t.take();
        }
        self.plan = None;
    }

    /// Sum of `memory_estimate_bytes` across all node tensors.
    pub fn total_memory_estimate(&self) -> usize {
        self.tensors.iter().map(|t| t.memory_estimate_bytes()).sum()
    }

    /// Build (or return the cached) [`EvaluationPlan`] for this graph.
    pub fn build_plan(&mut self) -> &EvaluationPlan {
        if self.plan.is_none() {
            self.plan = Some(EvaluationPlan::build(self.graph, None));
        }
        self.plan.as_ref().expect("plan was just inserted")
    }

    /// Number of nodes in the underlying graph.
    pub fn node_count(&self) -> usize {
        self.graph.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

    fn empty_graph() -> EinsumGraph {
        EinsumGraph::new()
    }

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

    fn two_node_graph() -> EinsumGraph {
        let mut g = EinsumGraph::new();
        let t0 = g.add_tensor("t0");
        let t1 = g.add_tensor("t1");
        let t2 = g.add_tensor("t2");
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
        g.add_output(t2).unwrap();
        g
    }

    #[test]
    fn test_lazy_graph_new_not_evaluated() {
        let g = single_node_graph();
        let lg = LazyEinsumGraph::new(&g);
        assert!(!lg.is_fully_evaluated());
        assert!(lg.get_output().is_none());
    }

    #[test]
    fn test_lazy_graph_invalidate() {
        let g = single_node_graph();
        let mut lg = LazyEinsumGraph::new(&g);
        // Manually mark the single node as computed.
        use scirs2_core::ndarray::ArrayD;
        let dummy: Scirs2Tensor = ArrayD::zeros(scirs2_core::ndarray::IxDyn(&[1]));
        lg.tensors[0].set(dummy);
        assert!(lg.is_fully_evaluated());
        lg.invalidate();
        assert!(!lg.is_fully_evaluated());
    }

    #[test]
    fn test_lazy_graph_node_count() {
        let g = two_node_graph();
        let lg = LazyEinsumGraph::new(&g);
        assert_eq!(lg.node_count(), 2);
    }

    #[test]
    fn test_lazy_graph_total_memory_estimate() {
        let g = empty_graph();
        let lg = LazyEinsumGraph::new(&g);
        // No nodes → zero memory
        assert_eq!(lg.total_memory_estimate(), 0);
    }
}
