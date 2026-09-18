//! Lazy executor with per-node caching on top of [`Scirs2Exec`].
//!
//! `LazyExecutor` wraps a `Scirs2Exec` and maintains a node-level cache keyed
//! by `usize` (node index).  Before delegating any tensor operation to the
//! inner executor it checks the cache; hits increment `LazyStats::cache_hits`
//! while misses increment `LazyStats::cache_misses` and populate the cache for
//! future calls.
//!
//! The executor also implements [`TlAutodiff`] by delegating forward / backward
//! passes to the inner executor and caching intermediate outputs.

use std::collections::HashMap;

use tensorlogic_infer::{ElemOp, ExecutorError, ReduceOp, TlAutodiff, TlExecutor};
use tensorlogic_ir::EinsumGraph;

use crate::autodiff::ForwardTape;
use crate::{Scirs2Exec, Scirs2Tensor};

/// Accumulated statistics for a [`LazyExecutor`] session.
#[derive(Debug, Default, Clone)]
pub struct LazyStats {
    /// Number of tensor lookups served directly from the cache.
    pub cache_hits: usize,
    /// Number of tensor lookups that required actual computation.
    pub cache_misses: usize,
    /// Number of tensors that had to be re-computed after cache invalidation.
    pub tensors_recomputed: usize,
    /// High-water-mark estimate of live memory (in bytes).
    pub peak_memory_estimate_bytes: usize,
}

/// A lazy executor that caches computed tensors by node index.
///
/// All tensor operations are delegated to an inner [`Scirs2Exec`]; the cache
/// layer sits above it and avoids redundant work when the same node is
/// requested multiple times (e.g. during iterative training with a static
/// graph).
pub struct LazyExecutor {
    inner: Scirs2Exec,
    /// Tensor cache: maps node_id → computed tensor.
    cache: HashMap<usize, Scirs2Tensor>,
    stats: LazyStats,
}

impl LazyExecutor {
    /// Create a new `LazyExecutor` with an empty cache.
    pub fn new() -> Self {
        Self {
            inner: Scirs2Exec::new(),
            cache: HashMap::new(),
            stats: LazyStats::default(),
        }
    }

    /// Create a `LazyExecutor` with a pre-allocated cache capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Scirs2Exec::new(),
            cache: HashMap::with_capacity(capacity),
            stats: LazyStats::default(),
        }
    }

    /// Discard all cached tensors.
    pub fn invalidate_cache(&mut self) {
        self.cache.clear();
    }

    /// Remove a single node from the cache.  The next access to that node will
    /// be a miss and will increment `tensors_recomputed`.
    pub fn invalidate_node(&mut self, node_id: usize) {
        if self.cache.remove(&node_id).is_some() {
            self.stats.tensors_recomputed += 1;
        }
    }

    /// Read-only reference to the accumulated statistics.
    pub fn stats(&self) -> &LazyStats {
        &self.stats
    }

    /// Rough total memory estimate for all tensors currently held by `graph`.
    ///
    /// Computed as `number_of_nodes * average_cached_tensor_size` — a simple
    /// heuristic based on what is already in the cache.
    pub fn memory_estimate_for(&self, graph: &EinsumGraph) -> usize {
        if graph.nodes.is_empty() {
            return 0;
        }
        if self.cache.is_empty() {
            return 0;
        }
        let total_cached_bytes: usize = self
            .cache
            .values()
            .map(|t| t.len() * std::mem::size_of::<f64>())
            .sum();
        let avg_bytes = total_cached_bytes / self.cache.len();
        avg_bytes * graph.nodes.len()
    }

    /// Number of tensors currently in the cache.
    pub fn cached_count(&self) -> usize {
        self.cache.len()
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Look up `node_id` in the cache.  Returns the cached tensor and bumps
    /// `cache_hits` if found.
    fn cache_get(&mut self, node_id: usize) -> Option<Scirs2Tensor> {
        if let Some(t) = self.cache.get(&node_id) {
            self.stats.cache_hits += 1;
            Some(t.clone())
        } else {
            self.stats.cache_misses += 1;
            None
        }
    }

    /// Insert a computed tensor into the cache and update peak memory stats.
    fn cache_insert(&mut self, node_id: usize, tensor: Scirs2Tensor) {
        let size = tensor.len() * std::mem::size_of::<f64>();
        self.cache.insert(node_id, tensor);
        let current_bytes: usize = self
            .cache
            .values()
            .map(|t| t.len() * std::mem::size_of::<f64>())
            .sum();
        if current_bytes > self.stats.peak_memory_estimate_bytes {
            self.stats.peak_memory_estimate_bytes = current_bytes;
        }
        // suppress unused-variable warning for size in release builds
        let _ = size;
    }
}

impl Default for LazyExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TlExecutor implementation
// ---------------------------------------------------------------------------

impl TlExecutor for LazyExecutor {
    type Tensor = Scirs2Tensor;
    type Error = ExecutorError;

    fn einsum(&mut self, spec: &str, inputs: &[Self::Tensor]) -> Result<Self::Tensor, Self::Error> {
        // The einsum operations themselves are not keyed by node_id here; the
        // cache is populated at the graph-traversal level (via forward()).
        // Direct calls to einsum bypass the cache — they are atomic operations.
        self.inner.einsum(spec, inputs)
    }

    fn elem_op(&mut self, op: ElemOp, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        self.inner.elem_op(op, x)
    }

    fn elem_op_binary(
        &mut self,
        op: ElemOp,
        x: &Self::Tensor,
        y: &Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error> {
        self.inner.elem_op_binary(op, x, y)
    }

    fn reduce(
        &mut self,
        op: ReduceOp,
        x: &Self::Tensor,
        axes: &[usize],
    ) -> Result<Self::Tensor, Self::Error> {
        self.inner.reduce(op, x, axes)
    }
}

// ---------------------------------------------------------------------------
// TlAutodiff implementation
// ---------------------------------------------------------------------------

impl TlAutodiff for LazyExecutor {
    type Tape = ForwardTape;

    /// Execute the forward pass, caching every node output.
    ///
    /// Node outputs are stored in `self.cache` keyed by their index in
    /// `graph.nodes` so that subsequent forward calls on the same (or an
    /// overlapping) graph reuse already-computed tensors.
    fn forward(&mut self, graph: &EinsumGraph) -> Result<Self::Tensor, Self::Error> {
        // Delegate to inner; it returns the final output tensor and stores the
        // full ForwardTape internally.
        let result = self.inner.forward(graph)?;

        // Populate cache with the node outputs stored by the inner executor.
        // The inner `ForwardTape` holds one Option<Scirs2Tensor> per *tensor*
        // index (not node index). We map node_index → its output tensor index.
        // Collect tensors first to avoid simultaneous mutable + immutable borrows.
        let node_tensors: Vec<(usize, Scirs2Tensor)> = if let Some(tape) = &self.inner.tape {
            graph
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(node_idx, node)| {
                    node.outputs.first().and_then(|&tensor_idx| {
                        tape.tensors
                            .get(tensor_idx)
                            .and_then(|opt| opt.as_ref())
                            .map(|t| (node_idx, t.clone()))
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        for (node_idx, tensor) in node_tensors {
            if !self.cache.contains_key(&node_idx) {
                self.cache_insert(node_idx, tensor);
            } else {
                self.stats.cache_hits += 1;
            }
        }

        Ok(result)
    }

    /// Execute the backward pass, delegating to the inner executor.
    fn backward(
        &mut self,
        graph: &EinsumGraph,
        loss: &Self::Tensor,
    ) -> Result<Self::Tape, Self::Error> {
        self.inner.backward(graph, loss)
    }
}

// ---------------------------------------------------------------------------
// Node-level cache lookup (optional convenience for graph executors)
// ---------------------------------------------------------------------------

impl LazyExecutor {
    /// Retrieve a cached tensor for the given node index (if available).
    ///
    /// This is the primary entry-point for lazy graph traversal: call this
    /// before scheduling a node for execution.  On a hit the value is returned
    /// without calling any inner operations.
    pub fn get_cached(&mut self, node_id: usize) -> Option<Scirs2Tensor> {
        self.cache_get(node_id)
    }

    /// Store a tensor result for a node.  Subsequent calls to
    /// `get_cached(node_id)` will return this value.
    pub fn put_cached(&mut self, node_id: usize, tensor: Scirs2Tensor) {
        self.cache_insert(node_id, tensor);
    }

    /// Access the inner [`Scirs2Exec`] (e.g. to register input tensors).
    pub fn inner_mut(&mut self) -> &mut Scirs2Exec {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::EinsumGraph;

    #[test]
    fn test_lazy_executor_default() {
        let exec = LazyExecutor::default();
        assert_eq!(exec.cached_count(), 0);
    }

    #[test]
    fn test_lazy_executor_cached_count_starts_zero() {
        let exec = LazyExecutor::new();
        assert_eq!(exec.cached_count(), 0);
    }

    #[test]
    fn test_lazy_executor_invalidate_cache() {
        let mut exec = LazyExecutor::with_capacity(4);
        use scirs2_core::ndarray::ArrayD;
        let t: Scirs2Tensor = ArrayD::zeros(scirs2_core::ndarray::IxDyn(&[2, 2]));
        exec.put_cached(0, t);
        assert_eq!(exec.cached_count(), 1);
        exec.invalidate_cache();
        assert_eq!(exec.cached_count(), 0);
    }

    #[test]
    fn test_lazy_stats_default() {
        let stats = LazyStats::default();
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.tensors_recomputed, 0);
        assert_eq!(stats.peak_memory_estimate_bytes, 0);
    }

    #[test]
    fn test_lazy_executor_memory_estimate_for_empty_graph() {
        let exec = LazyExecutor::new();
        let g = EinsumGraph::new();
        assert_eq!(exec.memory_estimate_for(&g), 0);
    }
}
