//! Declarative model graph builder API.
//!
//! Provides two complementary APIs for composing neural networks:
//!
//! 1. **`Sequential`** — an ordered list of named boxed layers.  Each layer
//!    receives the output of the previous one.  This is the simplest and most
//!    common architecture pattern.
//!
//! 2. **`ModelGraph`** — a directed acyclic graph (DAG) of named nodes.  Each
//!    node may depend on one or more upstream nodes, enabling skip connections,
//!    multi-branch architectures, and arbitrary residual patterns.
//!
//! Both builders validate shapes lazily (at `forward` time) rather than
//! eagerly at construction, keeping the API straightforward while remaining
//! compatible with the existing `NeuralError` error system.
//!
//! ## Example — Sequential
//!
//! ```rust
//! use oximedia_neural::graph::{Sequential, LayerFn};
//! use oximedia_neural::tensor::Tensor;
//! use std::sync::Arc;
//!
//! let mut seq = Sequential::new();
//! seq.add("relu", Arc::new(|t: &Tensor| {
//!     let out: Vec<f32> = t.data().iter().map(|&x| x.max(0.0)).collect();
//!     Tensor::from_data(out, t.shape().to_vec())
//! }));
//!
//! let input = Tensor::from_data(vec![-1.0, 2.0, -3.0], vec![3]).unwrap();
//! let output = seq.forward(&input).unwrap();
//! assert_eq!(output.data(), &[0.0, 2.0, 0.0]);
//! ```
//!
//! ## Example — ModelGraph with skip connection
//!
//! ```rust
//! use oximedia_neural::graph::ModelGraph;
//! use oximedia_neural::tensor::Tensor;
//! use std::sync::Arc;
//! use std::collections::HashMap;
//!
//! let mut graph = ModelGraph::new();
//! graph.add_input("x");
//! graph.add_node("identity", vec!["x"], Arc::new(|inputs: &[&Tensor]| {
//!     Ok(inputs[0].clone())
//! }));
//! graph.add_node("out", vec!["x", "identity"], Arc::new(|inputs: &[&Tensor]| {
//!     // Simple element-wise add (skip connection)
//!     let a = inputs[0].data();
//!     let b = inputs[1].data();
//!     let c: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
//!     Tensor::from_data(c, inputs[0].shape().to_vec())
//! }));
//!
//! let input = Tensor::from_data(vec![1.0, 2.0], vec![2]).unwrap();
//! let out = graph.forward_single("x", &input, "out").unwrap();
//! assert_eq!(out.data(), &[2.0, 4.0]);
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::NeuralError;
use crate::tensor::Tensor;

// ──────────────────────────────────────────────────────────────────────────────
// Type aliases
// ──────────────────────────────────────────────────────────────────────────────

/// A single-input, single-output layer function.
///
/// Takes a reference to the input tensor and returns a new tensor.
pub type LayerFn = Arc<dyn Fn(&Tensor) -> Result<Tensor, NeuralError> + Send + Sync>;

/// A multi-input, single-output graph node function.
///
/// Takes a slice of input tensor references and returns a new tensor.
pub type GraphNodeFn = Arc<dyn Fn(&[&Tensor]) -> Result<Tensor, NeuralError> + Send + Sync>;

// ──────────────────────────────────────────────────────────────────────────────
// Sequential
// ──────────────────────────────────────────────────────────────────────────────

/// An ordered stack of named layers.
///
/// Layers are applied sequentially: the output of layer `i` is the input
/// to layer `i+1`.
pub struct Sequential {
    layers: Vec<(String, LayerFn)>,
}

impl Sequential {
    /// Creates an empty `Sequential` model.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Appends a named layer to the end of the sequence.
    pub fn add(&mut self, name: impl Into<String>, layer: LayerFn) {
        self.layers.push((name.into(), layer));
    }

    /// Returns the number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns `true` if no layers have been added.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Returns the layer names in order.
    pub fn layer_names(&self) -> Vec<&str> {
        self.layers.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Runs the full forward pass through all layers.
    ///
    /// Returns the output of the final layer.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        if self.layers.is_empty() {
            return Ok(input.clone());
        }
        let mut current = self.layers[0].1(input)?;
        for (_, layer_fn) in &self.layers[1..] {
            current = layer_fn(&current)?;
        }
        Ok(current)
    }

    /// Runs a partial forward pass, stopping after layer with the given name.
    ///
    /// Returns the output of the named layer, or an error if the name is not
    /// found.
    pub fn forward_until(&self, input: &Tensor, stop_name: &str) -> Result<Tensor, NeuralError> {
        // First verify the stop_name exists.
        if !self.layers.iter().any(|(n, _)| n == stop_name) {
            return Err(NeuralError::InvalidShape(format!(
                "Sequential::forward_until: layer '{stop_name}' not found"
            )));
        }
        let mut current: Option<Tensor> = None;
        for (name, layer_fn) in &self.layers {
            let x = match &current {
                Some(t) => t,
                None => input,
            };
            let out = layer_fn(x)?;
            let is_stop = name == stop_name;
            current = Some(out);
            if is_stop {
                break;
            }
        }
        current.ok_or_else(|| {
            NeuralError::InvalidShape(format!(
                "Sequential::forward_until: layer '{stop_name}' not found"
            ))
        })
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GraphNode — internal DAG node
// ──────────────────────────────────────────────────────────────────────────────

/// Describes a single node in a `ModelGraph`.
struct GraphNode {
    /// Names of the input nodes this node depends on (in order).
    inputs: Vec<String>,
    /// The computation to perform.
    func: GraphNodeFn,
}

// ──────────────────────────────────────────────────────────────────────────────
// ModelGraph
// ──────────────────────────────────────────────────────────────────────────────

/// A directed acyclic computation graph of named nodes.
///
/// Input nodes are declared with [`add_input`](ModelGraph::add_input) and have
/// no computation function — they receive values from the caller.
///
/// Intermediate and output nodes are declared with
/// [`add_node`](ModelGraph::add_node).  Each node lists the upstream nodes it
/// reads from.
///
/// Execution proceeds in topological order (insertion order, which must be
/// topologically sorted by the caller).
pub struct ModelGraph {
    /// Ordered node names (insertion order, must be topological).
    node_order: Vec<String>,
    /// Input node names (no computation, values injected externally).
    input_names: Vec<String>,
    /// Computation nodes.
    nodes: HashMap<String, GraphNode>,
}

impl ModelGraph {
    /// Creates an empty `ModelGraph`.
    pub fn new() -> Self {
        Self {
            node_order: Vec::new(),
            input_names: Vec::new(),
            nodes: HashMap::new(),
        }
    }

    /// Registers a named input node.
    ///
    /// Input nodes hold tensors injected by the caller at forward time.
    pub fn add_input(&mut self, name: impl Into<String>) {
        let n = name.into();
        if !self.node_order.contains(&n) {
            self.node_order.push(n.clone());
        }
        if !self.input_names.contains(&n) {
            self.input_names.push(n);
        }
    }

    /// Adds a computation node that depends on the listed upstream nodes.
    ///
    /// `input_names` are the names of the upstream nodes whose outputs are
    /// passed as the `inputs` slice to `func`, in the order given.
    ///
    /// **Caller responsibility**: nodes must be added in topological order
    /// (all dependencies of a node must be added before it).
    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        input_names: Vec<impl Into<String>>,
        func: GraphNodeFn,
    ) {
        let n = name.into();
        let dep_names: Vec<String> = input_names.into_iter().map(|s| s.into()).collect();
        if !self.node_order.contains(&n) {
            self.node_order.push(n.clone());
        }
        self.nodes.insert(
            n,
            GraphNode {
                inputs: dep_names,
                func,
            },
        );
    }

    /// Runs a forward pass, providing a single named input tensor, and returns
    /// the activation of the named output node.
    ///
    /// * `input_name` — the name of the input node that receives `input`.
    /// * `input` — the tensor to inject at `input_name`.
    /// * `output_name` — the name of the node whose value to return.
    pub fn forward_single(
        &self,
        input_name: &str,
        input: &Tensor,
        output_name: &str,
    ) -> Result<Tensor, NeuralError> {
        let mut inputs = HashMap::new();
        inputs.insert(input_name.to_string(), input.clone());
        self.forward(&inputs, output_name)
    }

    /// Runs a forward pass with multiple named input tensors.
    ///
    /// * `inputs` — map of input-node-name → tensor.
    /// * `output_name` — the name of the node whose value to return.
    pub fn forward(
        &self,
        inputs: &HashMap<String, Tensor>,
        output_name: &str,
    ) -> Result<Tensor, NeuralError> {
        // Validate that all declared input nodes are supplied.
        for inp_name in &self.input_names {
            if !inputs.contains_key(inp_name.as_str()) {
                return Err(NeuralError::EmptyInput(format!(
                    "ModelGraph::forward: input node '{inp_name}' not provided"
                )));
            }
        }

        // Propagate in topological (insertion) order.
        let mut cache: HashMap<String, Tensor> = inputs.clone();

        for node_name in &self.node_order {
            if cache.contains_key(node_name.as_str()) {
                // Already present (was an injected input).
                continue;
            }
            let node = self.nodes.get(node_name.as_str()).ok_or_else(|| {
                NeuralError::InvalidShape(format!(
                    "ModelGraph::forward: node '{node_name}' has no computation"
                ))
            })?;

            // Gather input tensors.
            let dep_tensors: Result<Vec<&Tensor>, NeuralError> = node
                .inputs
                .iter()
                .map(|dep| {
                    cache.get(dep.as_str()).ok_or_else(|| {
                        NeuralError::InvalidShape(format!(
                            "ModelGraph::forward: dependency '{dep}' of '{node_name}' not available"
                        ))
                    })
                })
                .collect();
            let dep_tensors = dep_tensors?;

            let out = (node.func)(dep_tensors.as_slice())?;
            cache.insert(node_name.clone(), out);

            if node_name == output_name {
                break;
            }
        }

        cache.remove(output_name).ok_or_else(|| {
            NeuralError::InvalidShape(format!(
                "ModelGraph::forward: output node '{output_name}' not found or not reached"
            ))
        })
    }

    /// Returns the names of all nodes in insertion order.
    pub fn node_names(&self) -> Vec<&str> {
        self.node_order.iter().map(|s| s.as_str()).collect()
    }

    /// Returns `true` if the graph contains no nodes.
    pub fn is_empty(&self) -> bool {
        self.node_order.is_empty()
    }
}

impl Default for ModelGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SequentialBuilder — fluent API
// ──────────────────────────────────────────────────────────────────────────────

/// Fluent builder for `Sequential` models.
///
/// ```rust
/// use oximedia_neural::graph::{SequentialBuilder, LayerFn};
/// use oximedia_neural::tensor::Tensor;
/// use std::sync::Arc;
///
/// let model = SequentialBuilder::new()
///     .layer("relu", Arc::new(|t: &Tensor| {
///         let d: Vec<f32> = t.data().iter().map(|&x| x.max(0.0)).collect();
///         Tensor::from_data(d, t.shape().to_vec())
///     }))
///     .build();
///
/// let x = Tensor::from_data(vec![-1.0, 2.0], vec![2]).unwrap();
/// let y = model.forward(&x).unwrap();
/// assert_eq!(y.data(), &[0.0, 2.0]);
/// ```
pub struct SequentialBuilder {
    seq: Sequential,
}

impl SequentialBuilder {
    /// Creates a new `SequentialBuilder`.
    pub fn new() -> Self {
        Self {
            seq: Sequential::new(),
        }
    }

    /// Adds a named layer.
    pub fn layer(mut self, name: impl Into<String>, f: LayerFn) -> Self {
        self.seq.add(name, f);
        self
    }

    /// Consumes the builder and returns the `Sequential` model.
    pub fn build(self) -> Sequential {
        self.seq
    }
}

impl Default for SequentialBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::NeuralError;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    fn relu_fn() -> LayerFn {
        Arc::new(|t: &Tensor| {
            let d: Vec<f32> = t.data().iter().map(|&x| x.max(0.0)).collect();
            Tensor::from_data(d, t.shape().to_vec())
        })
    }

    fn scale_fn(factor: f32) -> LayerFn {
        Arc::new(move |t: &Tensor| {
            let d: Vec<f32> = t.data().iter().map(|&x| x * factor).collect();
            Tensor::from_data(d, t.shape().to_vec())
        })
    }

    // ── Sequential ───────────────────────────────────────────────────────────

    #[test]
    fn test_sequential_empty_passes_through() {
        let seq = Sequential::new();
        let x = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3]).expect("tensor from_data");
        let y = seq.forward(&x).expect("forward pass");
        assert_eq!(y.data(), x.data());
    }

    #[test]
    fn test_sequential_single_relu() {
        let mut seq = Sequential::new();
        seq.add("relu", relu_fn());
        let x = Tensor::from_data(vec![-1.0, 2.0, -3.0], vec![3]).expect("tensor from_data");
        let y = seq.forward(&x).expect("forward pass");
        assert!(close(y.data()[0], 0.0));
        assert!(close(y.data()[1], 2.0));
        assert!(close(y.data()[2], 0.0));
    }

    #[test]
    fn test_sequential_two_layers() {
        let mut seq = Sequential::new();
        seq.add("scale2", scale_fn(2.0));
        seq.add("scale3", scale_fn(3.0));
        let x = Tensor::from_data(vec![1.0, 2.0], vec![2]).expect("tensor from_data");
        let y = seq.forward(&x).expect("forward pass");
        // 1.0 * 2 * 3 = 6.0, 2.0 * 2 * 3 = 12.0
        assert!(close(y.data()[0], 6.0));
        assert!(close(y.data()[1], 12.0));
    }

    #[test]
    fn test_sequential_layer_names() {
        let mut seq = Sequential::new();
        seq.add("a", relu_fn());
        seq.add("b", relu_fn());
        seq.add("c", relu_fn());
        assert_eq!(seq.layer_names(), &["a", "b", "c"]);
    }

    #[test]
    fn test_sequential_len() {
        let mut seq = Sequential::new();
        assert_eq!(seq.len(), 0);
        assert!(seq.is_empty());
        seq.add("l1", relu_fn());
        assert_eq!(seq.len(), 1);
        assert!(!seq.is_empty());
    }

    #[test]
    fn test_sequential_forward_until() {
        let mut seq = Sequential::new();
        seq.add("double", scale_fn(2.0));
        seq.add("triple", scale_fn(3.0));
        seq.add("quad", scale_fn(4.0));
        let x = Tensor::from_data(vec![1.0], vec![1]).expect("tensor from_data");
        // Stop after "double": 1.0 * 2 = 2.0
        let y = seq.forward_until(&x, "double").expect("forward_until");
        assert!(close(y.data()[0], 2.0));
    }

    #[test]
    fn test_sequential_forward_until_not_found() {
        let mut seq = Sequential::new();
        seq.add("a", relu_fn());
        let x = Tensor::ones(vec![2]).expect("tensor ones");
        assert!(seq.forward_until(&x, "missing").is_err());
    }

    #[test]
    fn test_sequential_error_propagates() {
        let mut seq = Sequential::new();
        seq.add(
            "bad",
            Arc::new(|_t: &Tensor| {
                Err(NeuralError::EmptyInput(
                    "intentional test error".to_string(),
                ))
            }),
        );
        let x = Tensor::ones(vec![2]).expect("tensor ones");
        assert!(seq.forward(&x).is_err());
    }

    // ── SequentialBuilder ─────────────────────────────────────────────────────

    #[test]
    fn test_sequential_builder_fluent() {
        let model = SequentialBuilder::new()
            .layer("scale2", scale_fn(2.0))
            .layer("relu", relu_fn())
            .build();
        let x = Tensor::from_data(vec![-1.0, 3.0], vec![2]).expect("tensor from_data");
        let y = model.forward(&x).expect("forward pass");
        // -1.0 * 2 = -2.0 → relu → 0.0; 3.0 * 2 = 6.0 → relu → 6.0
        assert!(close(y.data()[0], 0.0));
        assert!(close(y.data()[1], 6.0));
    }

    // ── ModelGraph ────────────────────────────────────────────────────────────

    #[test]
    fn test_graph_single_input_output() {
        let mut graph = ModelGraph::new();
        graph.add_input("x");
        graph.add_node(
            "out",
            vec!["x"],
            Arc::new(|inputs: &[&Tensor]| {
                let d: Vec<f32> = inputs[0].data().iter().map(|&v| v * 2.0).collect();
                Tensor::from_data(d, inputs[0].shape().to_vec())
            }),
        );
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3]).expect("tensor from_data");
        let out = graph
            .forward_single("x", &input, "out")
            .expect("forward_single");
        assert!(close(out.data()[0], 2.0));
        assert!(close(out.data()[1], 4.0));
        assert!(close(out.data()[2], 6.0));
    }

    #[test]
    fn test_graph_skip_connection() {
        let mut graph = ModelGraph::new();
        graph.add_input("x");
        // Branch: identity clone of x
        graph.add_node(
            "id",
            vec!["x"],
            Arc::new(|inputs: &[&Tensor]| Ok(inputs[0].clone())),
        );
        // Add x + id(x) = 2*x
        graph.add_node(
            "out",
            vec!["x", "id"],
            Arc::new(|inputs: &[&Tensor]| {
                let a = inputs[0].data();
                let b = inputs[1].data();
                let c: Vec<f32> = a.iter().zip(b).map(|(x, y)| x + y).collect();
                Tensor::from_data(c, inputs[0].shape().to_vec())
            }),
        );
        let input = Tensor::from_data(vec![1.0, 2.0], vec![2]).expect("tensor from_data");
        let out = graph
            .forward_single("x", &input, "out")
            .expect("forward_single");
        assert!(close(out.data()[0], 2.0));
        assert!(close(out.data()[1], 4.0));
    }

    #[test]
    fn test_graph_multiple_inputs() {
        let mut graph = ModelGraph::new();
        graph.add_input("a");
        graph.add_input("b");
        graph.add_node(
            "sum",
            vec!["a", "b"],
            Arc::new(|inputs: &[&Tensor]| {
                let da = inputs[0].data();
                let db = inputs[1].data();
                let c: Vec<f32> = da.iter().zip(db).map(|(x, y)| x + y).collect();
                Tensor::from_data(c, inputs[0].shape().to_vec())
            }),
        );
        let ta = Tensor::from_data(vec![1.0, 2.0], vec![2]).expect("tensor from_data");
        let tb = Tensor::from_data(vec![10.0, 20.0], vec![2]).expect("tensor from_data");
        let mut map = HashMap::new();
        map.insert("a".to_string(), ta);
        map.insert("b".to_string(), tb);
        let out = graph.forward(&map, "sum").expect("forward pass");
        assert!(close(out.data()[0], 11.0));
        assert!(close(out.data()[1], 22.0));
    }

    #[test]
    fn test_graph_missing_input_error() {
        let mut graph = ModelGraph::new();
        graph.add_input("x");
        graph.add_node(
            "out",
            vec!["x"],
            Arc::new(|inputs: &[&Tensor]| Ok(inputs[0].clone())),
        );
        // Don't provide "x"
        let map: HashMap<String, Tensor> = HashMap::new();
        assert!(graph.forward(&map, "out").is_err());
    }

    #[test]
    fn test_graph_output_not_found_error() {
        let mut graph = ModelGraph::new();
        graph.add_input("x");
        let input = Tensor::ones(vec![2]).expect("tensor ones");
        assert!(graph.forward_single("x", &input, "nonexistent").is_err());
    }

    #[test]
    fn test_graph_node_names() {
        let mut graph = ModelGraph::new();
        graph.add_input("x");
        graph.add_node(
            "a",
            vec!["x"],
            Arc::new(|inp: &[&Tensor]| Ok(inp[0].clone())),
        );
        let names = graph.node_names();
        assert!(names.contains(&"x"));
        assert!(names.contains(&"a"));
    }

    #[test]
    fn test_graph_chained_nodes() {
        let mut graph = ModelGraph::new();
        graph.add_input("x");
        // x → double → triple → result
        graph.add_node(
            "double",
            vec!["x"],
            Arc::new(|inp: &[&Tensor]| {
                let d: Vec<f32> = inp[0].data().iter().map(|&v| v * 2.0).collect();
                Tensor::from_data(d, inp[0].shape().to_vec())
            }),
        );
        graph.add_node(
            "triple",
            vec!["double"],
            Arc::new(|inp: &[&Tensor]| {
                let d: Vec<f32> = inp[0].data().iter().map(|&v| v * 3.0).collect();
                Tensor::from_data(d, inp[0].shape().to_vec())
            }),
        );
        let input = Tensor::from_data(vec![1.0], vec![1]).expect("tensor from_data");
        let out = graph
            .forward_single("x", &input, "triple")
            .expect("forward_single");
        // 1.0 * 2 * 3 = 6.0
        assert!(close(out.data()[0], 6.0));
    }
}
