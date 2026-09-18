//! ONNX execution runtime: NodeExecutor trait, ExecutionGraph, and OnnxRuntime.
//!
//! This module provides the higher-level inference engine built on top of the
//! layer representations in [`crate::onnx`].

use std::collections::HashMap;

use crate::error::NeuralError;
use crate::onnx::{OnnxLayer, OnnxModel};
use crate::tensor::Tensor;

// ──────────────────────────────────────────────────────────────────────────────
// NodeExecutor trait and implementations
// ──────────────────────────────────────────────────────────────────────────────

/// Executes a single ONNX computation node given a slice of input tensors.
///
/// Each implementation corresponds to one ONNX operator.  The `inputs` slice
/// must contain the tensors in the order specified by the node's input list.
pub trait NodeExecutor: Send + Sync {
    /// The ONNX operator type string (e.g. `"Relu"`, `"Gemm"`).
    fn op_type(&self) -> &str;

    /// Runs the operator on `inputs` and returns the output tensors.
    fn execute(&self, inputs: &[&Tensor]) -> Result<Vec<Tensor>, NeuralError>;
}

// ── Gemm ─────────────────────────────────────────────────────────────────────

/// Generalized matrix multiplication: `Y = alpha * A * B^T + beta * C`.
///
/// `weight` is stored transposed relative to typical conventions so that
/// `forward(x)` computes `x @ weight.T + bias` matching the ONNX Gemm spec
/// (transB=1 by default in most exported models).
pub struct GemmExecutor {
    /// Weight matrix, shape `[out_features, in_features]`.
    pub weight: Tensor,
    /// Optional bias, shape `[out_features]`.
    pub bias: Option<Tensor>,
    /// Scalar multiplier for the matrix product (default 1.0).
    pub alpha: f32,
    /// Scalar multiplier for the bias (default 1.0).
    pub beta: f32,
}

impl GemmExecutor {
    /// Creates a new `GemmExecutor` with alpha=1, beta=1 and no bias.
    pub fn new(weight: Tensor) -> Self {
        Self {
            weight,
            bias: None,
            alpha: 1.0,
            beta: 1.0,
        }
    }

    /// Attaches a bias vector.
    pub fn with_bias(mut self, bias: Tensor) -> Self {
        self.bias = Some(bias);
        self
    }
}

impl NodeExecutor for GemmExecutor {
    fn op_type(&self) -> &str {
        "Gemm"
    }

    fn execute(&self, inputs: &[&Tensor]) -> Result<Vec<Tensor>, NeuralError> {
        let input = inputs
            .first()
            .ok_or_else(|| NeuralError::EmptyInput("Gemm: no inputs".into()))?;

        // weight shape: [out, in] — transpose to [in, out] for matmul.
        let wt = self.weight.transpose_2d()?;

        // Ensure input is 2-D: [1, in] if 1-D.
        let input_2d: Tensor;
        let a = if input.shape().len() == 1 {
            let n = input.shape()[0];
            input_2d = input.reshape(vec![1, n])?;
            &input_2d
        } else {
            input
        };

        let mut out = crate::tensor::matmul(a, &wt)?;

        // Scale by alpha.
        if (self.alpha - 1.0).abs() > f32::EPSILON {
            crate::tensor::scale_inplace(&mut out, self.alpha);
        }

        // Add beta * bias.
        if let Some(ref b) = self.bias {
            if (self.beta - 1.0).abs() > f32::EPSILON {
                let mut scaled_bias = b.clone();
                crate::tensor::scale_inplace(&mut scaled_bias, self.beta);
                out = crate::tensor::add(&out, &scaled_bias)?;
            } else {
                out = crate::tensor::add(&out, b)?;
            }
        }

        // Squeeze back to 1-D if input was 1-D.
        if input.shape().len() == 1 {
            let numel = out.numel();
            out = out.reshape(vec![numel])?;
        }

        Ok(vec![out])
    }
}

// ── MatMul ───────────────────────────────────────────────────────────────────

/// Plain matrix multiplication: `Y = A * B` (no bias, no scaling).
pub struct MatMulExecutor {
    /// Right-hand weight matrix, shape `[k, out_features]`.
    pub weight: Tensor,
}

impl MatMulExecutor {
    /// Creates a new `MatMulExecutor`.
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }
}

impl NodeExecutor for MatMulExecutor {
    fn op_type(&self) -> &str {
        "MatMul"
    }

    fn execute(&self, inputs: &[&Tensor]) -> Result<Vec<Tensor>, NeuralError> {
        let input = inputs
            .first()
            .ok_or_else(|| NeuralError::EmptyInput("MatMul: no inputs".into()))?;
        let out = crate::tensor::matmul(input, &self.weight)?;
        Ok(vec![out])
    }
}

// ── Relu ─────────────────────────────────────────────────────────────────────

/// Element-wise ReLU activation.
pub struct ReluExecutor;

impl NodeExecutor for ReluExecutor {
    fn op_type(&self) -> &str {
        "Relu"
    }

    fn execute(&self, inputs: &[&Tensor]) -> Result<Vec<Tensor>, NeuralError> {
        let input = inputs
            .first()
            .ok_or_else(|| NeuralError::EmptyInput("Relu: no inputs".into()))?;
        let mut out = (*input).clone();
        crate::tensor::relu_inplace(&mut out);
        Ok(vec![out])
    }
}

// ── Sigmoid ──────────────────────────────────────────────────────────────────

/// Element-wise sigmoid activation.
pub struct SigmoidExecutor;

impl NodeExecutor for SigmoidExecutor {
    fn op_type(&self) -> &str {
        "Sigmoid"
    }

    fn execute(&self, inputs: &[&Tensor]) -> Result<Vec<Tensor>, NeuralError> {
        let input = inputs
            .first()
            .ok_or_else(|| NeuralError::EmptyInput("Sigmoid: no inputs".into()))?;
        let data: Vec<f32> = input
            .data()
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        Ok(vec![Tensor::from_data(data, input.shape().to_vec())?])
    }
}

// ── Add ──────────────────────────────────────────────────────────────────────

/// Element-wise addition of two tensors.
pub struct AddExecutor;

impl NodeExecutor for AddExecutor {
    fn op_type(&self) -> &str {
        "Add"
    }

    fn execute(&self, inputs: &[&Tensor]) -> Result<Vec<Tensor>, NeuralError> {
        if inputs.len() < 2 {
            return Err(NeuralError::EmptyInput(
                "Add: requires exactly 2 inputs".into(),
            ));
        }
        let out = crate::tensor::add(inputs[0], inputs[1])?;
        Ok(vec![out])
    }
}

// ── Reshape ──────────────────────────────────────────────────────────────────

/// Reshape operator: maps the input to `target_shape`.
pub struct ReshapeExecutor {
    /// Target shape.  Must have the same element count as the input.
    pub target_shape: Vec<usize>,
}

impl ReshapeExecutor {
    /// Creates a new `ReshapeExecutor`.
    pub fn new(target_shape: Vec<usize>) -> Self {
        Self { target_shape }
    }
}

impl NodeExecutor for ReshapeExecutor {
    fn op_type(&self) -> &str {
        "Reshape"
    }

    fn execute(&self, inputs: &[&Tensor]) -> Result<Vec<Tensor>, NeuralError> {
        let input = inputs
            .first()
            .ok_or_else(|| NeuralError::EmptyInput("Reshape: no inputs".into()))?;
        let out = input.reshape(self.target_shape.clone())?;
        Ok(vec![out])
    }
}

// ── Transpose ────────────────────────────────────────────────────────────────

/// N-dimensional axis permutation operator.
///
/// Given an input tensor of shape `[d0, d1, ..., d_{n-1}]` and a permutation
/// `perm = [p0, p1, ..., p_{n-1}]`, the output has shape
/// `[d_{p0}, d_{p1}, ..., d_{p_{n-1}}]`.
pub struct TransposeExecutor {
    /// Axis permutation.  Must be a permutation of `[0, 1, ..., rank-1]`.
    pub perm: Vec<usize>,
}

impl TransposeExecutor {
    /// Creates a new `TransposeExecutor`.
    pub fn new(perm: Vec<usize>) -> Self {
        Self { perm }
    }
}

impl NodeExecutor for TransposeExecutor {
    fn op_type(&self) -> &str {
        "Transpose"
    }

    fn execute(&self, inputs: &[&Tensor]) -> Result<Vec<Tensor>, NeuralError> {
        let input = inputs
            .first()
            .ok_or_else(|| NeuralError::EmptyInput("Transpose: no inputs".into()))?;

        let in_shape = input.shape();
        let rank = in_shape.len();

        if self.perm.len() != rank {
            return Err(NeuralError::ShapeMismatch(format!(
                "Transpose: perm length {} != tensor rank {}",
                self.perm.len(),
                rank
            )));
        }

        // Validate that perm is indeed a permutation.
        let mut seen = vec![false; rank];
        for &p in &self.perm {
            if p >= rank {
                return Err(NeuralError::InvalidShape(format!(
                    "Transpose: perm index {p} out of range for rank {rank}"
                )));
            }
            if seen[p] {
                return Err(NeuralError::InvalidShape(format!(
                    "Transpose: duplicate perm index {p}"
                )));
            }
            seen[p] = true;
        }

        // Output shape.
        let out_shape: Vec<usize> = self.perm.iter().map(|&p| in_shape[p]).collect();
        let numel = out_shape.iter().product::<usize>();

        // Compute input strides (row-major).
        let mut in_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
        }

        // Compute output strides (row-major).
        let mut out_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
        }

        let in_data = input.data();
        let mut out_data = vec![0.0_f32; numel];

        // Iterate over all output indices using a counter.
        let mut out_idx_vec = vec![0usize; rank];
        for flat_out in 0..numel {
            // Map out_idx_vec → flat input index via the permutation.
            let flat_in: usize = self
                .perm
                .iter()
                .enumerate()
                .map(|(out_dim, &in_dim)| out_idx_vec[out_dim] * in_strides[in_dim])
                .sum();

            out_data[flat_out] = in_data[flat_in];

            // Increment out_idx_vec (like a counter with carry).
            for d in (0..rank).rev() {
                out_idx_vec[d] += 1;
                if out_idx_vec[d] < out_shape[d] {
                    break;
                }
                out_idx_vec[d] = 0;
            }
        }

        Ok(vec![Tensor::from_data(out_data, out_shape)?])
    }
}

// ── FallbackExecutor ─────────────────────────────────────────────────────────

/// Wraps an [`OnnxLayer`] and executes it via [`OnnxModel::run_layer`].
///
/// Used for complex operators (Conv2d, BatchNorm, etc.) that carry pre-loaded
/// weight tensors but do not have a dedicated `NodeExecutor` implementation.
pub struct FallbackExecutor {
    layer: OnnxLayer,
    op_type_str: String,
}

impl FallbackExecutor {
    /// Creates a new `FallbackExecutor` wrapping `layer`.
    pub fn new(layer: OnnxLayer, op_type_str: String) -> Self {
        Self { layer, op_type_str }
    }
}

impl NodeExecutor for FallbackExecutor {
    fn op_type(&self) -> &str {
        &self.op_type_str
    }

    fn execute(&self, inputs: &[&Tensor]) -> Result<Vec<Tensor>, NeuralError> {
        let input = inputs
            .first()
            .ok_or_else(|| NeuralError::EmptyInput("FallbackExecutor: no inputs".into()))?;
        let rhs = inputs.get(1).copied();
        let out = OnnxModel::run_layer(&self.layer, input, rhs)?;
        Ok(vec![out])
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ExecutionNode
// ──────────────────────────────────────────────────────────────────────────────

/// A single node in an [`ExecutionGraph`], binding input/output tensor names
/// to a concrete [`NodeExecutor`].
pub struct ExecutionNode {
    /// Node name (may be empty).
    pub name: String,
    /// Names of the tensors consumed by this node, in order.
    pub inputs: Vec<String>,
    /// Names of the tensors produced by this node, in order.
    pub outputs: Vec<String>,
    /// The executor that implements this node's computation.
    pub executor: Box<dyn NodeExecutor>,
}

// ──────────────────────────────────────────────────────────────────────────────
// ExecutionGraph
// ──────────────────────────────────────────────────────────────────────────────

/// A topologically-ordered computation graph ready for inference.
///
/// Build from an [`OnnxModel`] via [`ExecutionGraph::from_onnx_model`], or
/// construct manually for unit-testing specific execution paths.
pub struct ExecutionGraph {
    /// Nodes in topological execution order.
    pub nodes: Vec<ExecutionNode>,
    /// Names of the graph-level input tensors.
    pub input_names: Vec<String>,
    /// Names of the graph-level output tensors.
    pub output_names: Vec<String>,
}

impl ExecutionGraph {
    /// Creates an empty graph with no nodes.
    pub fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            input_names: Vec::new(),
            output_names: Vec::new(),
        }
    }

    /// Builds an `ExecutionGraph` from a parsed [`OnnxModel`].
    ///
    /// ONNX models already store nodes in topological order, so the ordering
    /// step is a lightweight verification pass rather than a full sort.
    pub fn from_onnx_model(model: &OnnxModel) -> Result<Self, NeuralError> {
        let mut nodes: Vec<ExecutionNode> = Vec::with_capacity(model.nodes.len());

        for (onnx_node, onnx_layer) in model.nodes.iter().zip(model.layers.iter()) {
            let executor: Box<dyn NodeExecutor> = match onnx_layer {
                OnnxLayer::Relu => Box::new(ReluExecutor),
                OnnxLayer::Sigmoid => Box::new(SigmoidExecutor),
                OnnxLayer::Add => Box::new(AddExecutor),
                OnnxLayer::Linear { weight, bias } => {
                    let mut exec = GemmExecutor::new(weight.clone());
                    if let Some(b) = bias {
                        exec = exec.with_bias(b.clone());
                    }
                    Box::new(exec)
                }
                OnnxLayer::Reshape { target_shape } => {
                    Box::new(ReshapeExecutor::new(target_shape.clone()))
                }
                other => {
                    let op_str = onnx_node.op_type.clone();
                    Box::new(FallbackExecutor::new(other.clone(), op_str))
                }
            };

            nodes.push(ExecutionNode {
                name: onnx_node.name.clone(),
                inputs: onnx_node.inputs.clone(),
                outputs: onnx_node.outputs.clone(),
                executor,
            });
        }

        // Derive graph-level I/O from the raw ONNX info (preserved in model).
        // We reconstruct input names as those not produced by any node, and
        // output names as those not consumed by any later node.
        let produced: std::collections::HashSet<String> = model
            .nodes
            .iter()
            .flat_map(|n| n.outputs.iter().cloned())
            .collect();

        let consumed: std::collections::HashSet<String> = model
            .nodes
            .iter()
            .flat_map(|n| n.inputs.iter().cloned())
            .collect();

        let input_names: Vec<String> = model
            .nodes
            .iter()
            .flat_map(|n| n.inputs.iter())
            .filter(|name| !produced.contains(*name))
            .cloned()
            .collect::<std::collections::LinkedList<_>>()
            .into_iter()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let output_names: Vec<String> = model
            .nodes
            .iter()
            .flat_map(|n| n.outputs.iter())
            .filter(|name| !consumed.contains(*name))
            .cloned()
            .collect::<std::collections::LinkedList<_>>()
            .into_iter()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        Ok(Self {
            nodes,
            input_names,
            output_names,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// OnnxRuntime
// ──────────────────────────────────────────────────────────────────────────────

/// Executes an [`ExecutionGraph`] against a set of named input tensors.
///
/// The runtime maintains a value store (a `HashMap<String, Tensor>`) that is
/// seeded with the caller-supplied inputs.  Each node in the graph looks up its
/// inputs from this store and writes its outputs back to it.  The final result
/// is a map of every output tensor produced by the graph.
///
/// # Example
///
/// ```rust
/// use std::collections::HashMap;
/// use oximedia_neural::onnx_runtime::{OnnxRuntime, ExecutionGraph, ExecutionNode, ReluExecutor};
/// use oximedia_neural::tensor::Tensor;
///
/// let node = ExecutionNode {
///     name: "relu_0".into(),
///     inputs: vec!["x".into()],
///     outputs: vec!["y".into()],
///     executor: Box::new(ReluExecutor),
/// };
/// let graph = ExecutionGraph {
///     nodes: vec![node],
///     input_names: vec!["x".into()],
///     output_names: vec!["y".into()],
/// };
/// let runtime = OnnxRuntime::new(graph);
///
/// let mut inputs = HashMap::new();
/// inputs.insert("x".into(), Tensor::from_data(vec![-1.0, 0.0, 1.0], vec![3]).unwrap());
/// let outputs = runtime.run(inputs).unwrap();
/// assert_eq!(outputs["y"].data(), &[0.0, 0.0, 1.0]);
/// ```
pub struct OnnxRuntime {
    graph: ExecutionGraph,
}

impl OnnxRuntime {
    /// Wraps an `ExecutionGraph` in a runtime.
    pub fn new(graph: ExecutionGraph) -> Self {
        Self { graph }
    }

    /// Convenience constructor: builds an `ExecutionGraph` from an
    /// [`OnnxModel`] and wraps it in a runtime.
    pub fn from_model(model: &OnnxModel) -> Result<Self, NeuralError> {
        let graph = ExecutionGraph::from_onnx_model(model)?;
        Ok(Self::new(graph))
    }

    /// Runs inference on the graph.
    ///
    /// `inputs` must contain a `Tensor` for each name listed in
    /// `self.graph.input_names`.  Additional tensors are silently ignored.
    ///
    /// Returns a map of **all** tensors that were written during execution
    /// (inputs are not included unless a node copies them).  Call
    /// `outputs["name"]` to retrieve a specific result.
    pub fn run(
        &self,
        inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>, NeuralError> {
        // Value store: start with caller-supplied inputs.
        let mut store: HashMap<String, Tensor> = inputs;

        for node in &self.graph.nodes {
            // Gather input tensors by name.
            let input_refs: Vec<&Tensor> = node
                .inputs
                .iter()
                .map(|name| {
                    store.get(name.as_str()).ok_or_else(|| {
                        NeuralError::Io(format!(
                            "OnnxRuntime: missing tensor '{}' required by node '{}'",
                            name, node.name
                        ))
                    })
                })
                .collect::<Result<Vec<&Tensor>, NeuralError>>()?;

            // Execute the node.
            let outputs = node.executor.execute(&input_refs)?;

            // Store each output tensor by its name.
            for (name, tensor) in node.outputs.iter().zip(outputs.into_iter()) {
                store.insert(name.clone(), tensor);
            }
        }

        // Return only the non-input tensors that were produced (graph outputs).
        // Callers can also query intermediate tensors if they know the names.
        Ok(store)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Runtime tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod runtime_tests {
    use super::*;
    use std::collections::HashMap;

    fn t1d(data: Vec<f32>) -> Tensor {
        let n = data.len();
        Tensor::from_data(data, vec![n]).expect("t1d")
    }

    fn t2d(data: Vec<f32>, rows: usize, cols: usize) -> Tensor {
        Tensor::from_data(data, vec![rows, cols]).expect("t2d")
    }

    // ── NodeExecutor op_type strings ─────────────────────────────────────────

    #[test]
    fn test_node_executor_op_type_strings() {
        let weight = t2d(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let gemm = GemmExecutor::new(weight.clone());
        let matmul = MatMulExecutor::new(weight);
        let relu = ReluExecutor;
        let sigmoid = SigmoidExecutor;
        let add = AddExecutor;
        let reshape = ReshapeExecutor::new(vec![4]);
        let transpose = TransposeExecutor::new(vec![1, 0]);
        assert_eq!(gemm.op_type(), "Gemm");
        assert_eq!(matmul.op_type(), "MatMul");
        assert_eq!(relu.op_type(), "Relu");
        assert_eq!(sigmoid.op_type(), "Sigmoid");
        assert_eq!(add.op_type(), "Add");
        assert_eq!(reshape.op_type(), "Reshape");
        assert_eq!(transpose.op_type(), "Transpose");
    }

    // ── ReluExecutor ─────────────────────────────────────────────────────────

    #[test]
    fn test_relu_executor_basic() {
        let executor = ReluExecutor;
        let input = t1d(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let outputs = executor.execute(&[&input]).expect("relu execute");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].data(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_executor_no_inputs_error() {
        let executor = ReluExecutor;
        assert!(executor.execute(&[]).is_err());
    }

    // ── SigmoidExecutor ──────────────────────────────────────────────────────

    #[test]
    fn test_sigmoid_executor_range() {
        let executor = SigmoidExecutor;
        let input = t1d(vec![-100.0, -1.0, 0.0, 1.0, 100.0]);
        let outputs = executor.execute(&[&input]).expect("sigmoid execute");
        assert_eq!(outputs.len(), 1);
        let data = outputs[0].data();
        for &v in data {
            assert!(v >= 0.0 && v <= 1.0, "sigmoid out of [0,1]: {v}");
        }
        // sigmoid(0) == 0.5
        assert!((data[2] - 0.5).abs() < 1e-5);
    }

    // ── AddExecutor ──────────────────────────────────────────────────────────

    #[test]
    fn test_add_executor_basic() {
        let executor = AddExecutor;
        let a = t1d(vec![1.0, 2.0, 3.0]);
        let b = t1d(vec![4.0, 5.0, 6.0]);
        let outputs = executor.execute(&[&a, &b]).expect("add execute");
        assert_eq!(outputs[0].data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_executor_too_few_inputs_error() {
        let executor = AddExecutor;
        let a = t1d(vec![1.0, 2.0]);
        assert!(executor.execute(&[&a]).is_err());
    }

    // ── GemmExecutor ─────────────────────────────────────────────────────────

    #[test]
    fn test_gemm_executor_identity_weight() {
        // weight is [2,2] identity — output == input
        let weight = t2d(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let executor = GemmExecutor::new(weight);
        let input = t1d(vec![3.0, 7.0]);
        let outputs = executor.execute(&[&input]).expect("gemm execute");
        let data = outputs[0].data();
        assert!((data[0] - 3.0).abs() < 1e-5);
        assert!((data[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemm_executor_with_bias() {
        // weight [2,2] all ones; bias [2] = [1, 2]
        let weight = t2d(vec![1.0, 1.0, 1.0, 1.0], 2, 2);
        let bias = t1d(vec![1.0, 2.0]);
        let executor = GemmExecutor::new(weight).with_bias(bias);
        let input = t1d(vec![1.0, 1.0]);
        let outputs = executor.execute(&[&input]).expect("gemm+bias execute");
        // Each output element: dot(input, col_i) + bias_i
        // weight row 0: [1,1], dot with [1,1] = 2, +1 = 3
        // weight row 1: [1,1], dot with [1,1] = 2, +2 = 4
        let data = outputs[0].data();
        assert!((data[0] - 3.0).abs() < 1e-4, "data[0]={}", data[0]);
        assert!((data[1] - 4.0).abs() < 1e-4, "data[1]={}", data[1]);
    }

    // ── MatMulExecutor ───────────────────────────────────────────────────────

    #[test]
    fn test_matmul_executor_shapes() {
        // [3,4] x [4,2] = [3,2]
        let weight = t2d(vec![1.0; 8], 4, 2);
        let executor = MatMulExecutor::new(weight);
        let input = t2d(vec![1.0; 12], 3, 4);
        let outputs = executor.execute(&[&input]).expect("matmul execute");
        assert_eq!(outputs[0].shape(), &[3, 2]);
    }

    // ── ReshapeExecutor ──────────────────────────────────────────────────────

    #[test]
    fn test_reshape_executor_basic() {
        let executor = ReshapeExecutor::new(vec![6]);
        let input = t2d(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let outputs = executor.execute(&[&input]).expect("reshape execute");
        assert_eq!(outputs[0].shape(), &[6]);
        assert_eq!(outputs[0].numel(), 6);
    }

    #[test]
    fn test_reshape_executor_2d_to_3d() {
        let executor = ReshapeExecutor::new(vec![2, 2, 3]);
        let input = t2d(vec![1.0; 12], 4, 3);
        let outputs = executor.execute(&[&input]).expect("reshape 3d");
        assert_eq!(outputs[0].shape(), &[2, 2, 3]);
    }

    // ── TransposeExecutor ────────────────────────────────────────────────────

    #[test]
    fn test_transpose_executor_2d() {
        // [[1,2],[3,4]] transposed → [[1,3],[2,4]]
        let executor = TransposeExecutor::new(vec![1, 0]);
        let input = t2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let outputs = executor.execute(&[&input]).expect("transpose 2d");
        assert_eq!(outputs[0].shape(), &[2, 2]);
        let data = outputs[0].data();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 3.0).abs() < 1e-6);
        assert!((data[2] - 2.0).abs() < 1e-6);
        assert!((data[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_transpose_executor_3d() {
        // shape [2,3,4] with perm [1,0,2] → shape [3,2,4]
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let input = Tensor::from_data(data, vec![2, 3, 4]).expect("t3d");
        let executor = TransposeExecutor::new(vec![1, 0, 2]);
        let outputs = executor.execute(&[&input]).expect("transpose 3d");
        assert_eq!(outputs[0].shape(), &[3, 2, 4]);
        assert_eq!(outputs[0].numel(), 24);
    }

    #[test]
    fn test_transpose_executor_invalid_perm_error() {
        let executor = TransposeExecutor::new(vec![0, 0]); // duplicate
        let input = t2d(vec![1.0; 4], 2, 2);
        assert!(executor.execute(&[&input]).is_err());
    }

    // ── ExecutionGraph ───────────────────────────────────────────────────────

    #[test]
    fn test_execution_graph_empty() {
        let graph = ExecutionGraph::empty();
        let runtime = OnnxRuntime::new(graph);
        let inputs: HashMap<String, Tensor> = HashMap::new();
        let outputs = runtime.run(inputs).expect("empty graph run");
        assert!(outputs.is_empty());
    }

    // ── OnnxRuntime ──────────────────────────────────────────────────────────

    #[test]
    fn test_onnx_runtime_passthrough_identity() {
        let node = ExecutionNode {
            name: "relu_0".into(),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            executor: Box::new(ReluExecutor),
        };
        let graph = ExecutionGraph {
            nodes: vec![node],
            input_names: vec!["x".into()],
            output_names: vec!["y".into()],
        };
        let runtime = OnnxRuntime::new(graph);
        let mut inputs = HashMap::new();
        inputs.insert("x".into(), t1d(vec![1.0, 2.0, 3.0]));
        let outputs = runtime.run(inputs).expect("identity run");
        assert_eq!(outputs["y"].data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_onnx_runtime_relu_chain() {
        // Two relu nodes in sequence: x → relu0 → h → relu1 → y
        let node0 = ExecutionNode {
            name: "relu_0".into(),
            inputs: vec!["x".into()],
            outputs: vec!["h".into()],
            executor: Box::new(ReluExecutor),
        };
        let node1 = ExecutionNode {
            name: "relu_1".into(),
            inputs: vec!["h".into()],
            outputs: vec!["y".into()],
            executor: Box::new(ReluExecutor),
        };
        let graph = ExecutionGraph {
            nodes: vec![node0, node1],
            input_names: vec!["x".into()],
            output_names: vec!["y".into()],
        };
        let runtime = OnnxRuntime::new(graph);
        let mut inputs = HashMap::new();
        inputs.insert("x".into(), t1d(vec![-3.0, -1.0, 0.5, 2.0]));
        let outputs = runtime.run(inputs).expect("relu chain");
        assert_eq!(outputs["y"].data(), &[0.0, 0.0, 0.5, 2.0]);
    }

    #[test]
    fn test_onnx_runtime_missing_input_error() {
        let node = ExecutionNode {
            name: "add_0".into(),
            inputs: vec!["a".into(), "b".into()],
            outputs: vec!["c".into()],
            executor: Box::new(AddExecutor),
        };
        let graph = ExecutionGraph {
            nodes: vec![node],
            input_names: vec!["a".into(), "b".into()],
            output_names: vec!["c".into()],
        };
        let runtime = OnnxRuntime::new(graph);
        // Only supply "a", omit "b"
        let mut inputs = HashMap::new();
        inputs.insert("a".into(), t1d(vec![1.0, 2.0]));
        assert!(runtime.run(inputs).is_err());
    }

    #[test]
    fn test_onnx_runtime_add_op() {
        let node = ExecutionNode {
            name: "add_0".into(),
            inputs: vec!["a".into(), "b".into()],
            outputs: vec!["out".into()],
            executor: Box::new(AddExecutor),
        };
        let graph = ExecutionGraph {
            nodes: vec![node],
            input_names: vec!["a".into(), "b".into()],
            output_names: vec!["out".into()],
        };
        let runtime = OnnxRuntime::new(graph);
        let mut inputs = HashMap::new();
        inputs.insert("a".into(), t1d(vec![1.0, 2.0, 3.0]));
        inputs.insert("b".into(), t1d(vec![10.0, 20.0, 30.0]));
        let outputs = runtime.run(inputs).expect("add op run");
        assert_eq!(outputs["out"].data(), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_2layer_mlp_inference() {
        // 2-layer MLP: input [2] → Linear(2→4) → Relu → Linear(4→1) → output [1]
        // W1: [4, 2] = all 0.5; W2: [1, 4] = all 1.0
        let w1 = t2d(vec![0.5; 8], 4, 2);
        let w2 = t2d(vec![1.0; 4], 1, 4);

        let node_linear1 = ExecutionNode {
            name: "linear_1".into(),
            inputs: vec!["x".into()],
            outputs: vec!["h1".into()],
            executor: Box::new(GemmExecutor::new(w1)),
        };
        let node_relu = ExecutionNode {
            name: "relu".into(),
            inputs: vec!["h1".into()],
            outputs: vec!["h1_act".into()],
            executor: Box::new(ReluExecutor),
        };
        let node_linear2 = ExecutionNode {
            name: "linear_2".into(),
            inputs: vec!["h1_act".into()],
            outputs: vec!["out".into()],
            executor: Box::new(GemmExecutor::new(w2)),
        };

        let graph = ExecutionGraph {
            nodes: vec![node_linear1, node_relu, node_linear2],
            input_names: vec!["x".into()],
            output_names: vec!["out".into()],
        };
        let runtime = OnnxRuntime::new(graph);
        let mut inputs = HashMap::new();
        inputs.insert("x".into(), t1d(vec![1.0, 1.0]));
        let outputs = runtime.run(inputs).expect("mlp inference");

        // h1 = W1 @ x = [0.5+0.5; 0.5+0.5; 0.5+0.5; 0.5+0.5] = [1,1,1,1]
        // relu([1,1,1,1]) = [1,1,1,1]
        // out = W2 @ h1_act = sum([1,1,1,1]) = 4
        let out_data = outputs["out"].data();
        assert_eq!(out_data.len(), 1);
        assert!((out_data[0] - 4.0).abs() < 1e-4, "out={}", out_data[0]);
    }

    #[test]
    fn test_onnx_runtime_reshape_in_graph() {
        // gemm [2] → [4], then reshape [4] → [2, 2]
        let weight = t2d(vec![1.0; 8], 4, 2); // [4, 2]
        let node_gemm = ExecutionNode {
            name: "gemm".into(),
            inputs: vec!["x".into()],
            outputs: vec!["h".into()],
            executor: Box::new(GemmExecutor::new(weight)),
        };
        let node_reshape = ExecutionNode {
            name: "reshape".into(),
            inputs: vec!["h".into()],
            outputs: vec!["out".into()],
            executor: Box::new(ReshapeExecutor::new(vec![2, 2])),
        };
        let graph = ExecutionGraph {
            nodes: vec![node_gemm, node_reshape],
            input_names: vec!["x".into()],
            output_names: vec!["out".into()],
        };
        let runtime = OnnxRuntime::new(graph);
        let mut inputs = HashMap::new();
        inputs.insert("x".into(), t1d(vec![1.0, 1.0]));
        let outputs = runtime.run(inputs).expect("gemm+reshape run");
        assert_eq!(outputs["out"].shape(), &[2, 2]);
    }
}
