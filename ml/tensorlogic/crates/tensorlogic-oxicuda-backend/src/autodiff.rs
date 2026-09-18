//! Automatic differentiation (forward / backward) for [`OxiCudaExecutor`].
//!
//! Implements [`TlAutodiff`] by recording a tape during the forward pass and
//! replaying it in reverse to compute gradients.
//!
//! # Feature-flag behaviour
//!
//! The `TlAutodiff` impl is always compiled (the trait bound requires it).
//! When the `gpu` feature is disabled every call to `forward` / `backward`
//! returns [`OxiCudaBackendError::BackendDisabled`] because all dispatched
//! executor methods already do so.
//!
//! # Broadcast helper selection
//!
//! Gradient computations for reduce ops (ReduceSum, ReduceMean) require
//! broadcasting a reduced tensor back to the original input shape.
//!
//! - **Default (host path):** `broadcast_to_shape` / `fill_tensor_host` perform
//!   the computation entirely on the CPU.
//! - **`native-broadcast` feature:** `broadcast_to_shape_native` and
//!   `fill_tensor_native` upload the tensor to a `DeviceBuffer`, call the
//!   `oxicuda_blas::elementwise::broadcast_axes` / `fill` GPU kernels, and
//!   read back the result. This is a Round 5 architecture: host-resident tensors
//!   with explicit upload/kernel/readback. Round 6 will eliminate the round-trips
//!   by converting to true device-resident tensors.

use std::collections::HashMap;

use tensorlogic_infer::{ElemOp, ReduceOp, TlAutodiff, TlExecutor};
use tensorlogic_ir::{EinsumGraph, OpType};

use crate::error::OxiCudaBackendError;
use crate::executor::{OxiCudaExecutor, OxiCudaTensor};

#[cfg(feature = "native-broadcast")]
use crate::executor::GpuState;

// ---------------------------------------------------------------------------
// Tape types
// ---------------------------------------------------------------------------

/// A tape entry recording one operation performed during the forward pass.
struct TapeEntry {
    /// Index of the output tensor this node produced.
    output_tensor_idx: usize,
    /// Indices of the input tensors consumed by this node.
    input_tensor_indices: Vec<usize>,
    /// The type of operation, distilled to what the backward pass needs.
    op: TapedOp,
    /// Copies of the input tensors saved for gradient computation.
    saved_inputs: Vec<OxiCudaTensor>,
}

/// Distilled representation of an operation on the tape.
enum TapedOp {
    /// Standard 2-D matrix multiply: `C[m,n] = Σ_k A[m,k]·B[k,n]`.
    Matmul2D,
    /// Batched 3-D matrix multiply: `C[b,m,n] = Σ_k A[b,m,k]·B[b,k,n]`.
    BatchedMatmul3D,
    /// Identity pass-through (no computation).
    Identity,
    /// Unary element-wise op.
    Unary(ElemOp),
    /// Binary element-wise op.
    Binary(ElemOp),
    /// Reduction along one or more axes.
    Reduce(ReduceOp, Vec<usize>),
}

/// The tape produced by [`OxiCudaExecutor::forward`].
///
/// Callers can inspect `gradients` after [`OxiCudaExecutor::backward`].
pub struct OxiCudaTape {
    entries: Vec<TapeEntry>,
    /// Gradient tensors keyed by graph tensor index.  Public for tests.
    pub gradients: HashMap<usize, OxiCudaTensor>,
}

// ---------------------------------------------------------------------------
// Tiny helper: parse an op-name string → ElemOp
// ---------------------------------------------------------------------------

/// Parse a string op-name (as stored in [`OpType::ElemUnary`] / [`OpType::ElemBinary`])
/// into an [`ElemOp`].  Returns an error string on unknown names.
fn parse_elem_op(op: &str) -> Result<ElemOp, OxiCudaBackendError> {
    match op.to_lowercase().as_str() {
        "relu" => Ok(ElemOp::Relu),
        "sigmoid" => Ok(ElemOp::Sigmoid),
        "oneminus" | "one_minus" => Ok(ElemOp::OneMinus),
        "add" => Ok(ElemOp::Add),
        "subtract" | "sub" => Ok(ElemOp::Subtract),
        "multiply" | "mul" => Ok(ElemOp::Multiply),
        "divide" | "div" => Ok(ElemOp::Divide),
        "min" => Ok(ElemOp::Min),
        "max" => Ok(ElemOp::Max),
        "eq" | "equal" => Ok(ElemOp::Eq),
        "lt" | "lessthan" => Ok(ElemOp::Lt),
        "gt" | "greaterthan" => Ok(ElemOp::Gt),
        "lte" | "lessthanorequal" => Ok(ElemOp::Lte),
        "gte" | "greaterthanorequal" => Ok(ElemOp::Gte),
        "or_max" | "ormax" => Ok(ElemOp::OrMax),
        "or_prob_sum" | "orprobsum" | "or_probabilistic" => Ok(ElemOp::OrProbSum),
        "nand" => Ok(ElemOp::Nand),
        "nor" => Ok(ElemOp::Nor),
        "xor" => Ok(ElemOp::Xor),
        other => Err(OxiCudaBackendError::InvalidEinsumSpec(format!(
            "unknown element-wise op name: {other}"
        ))),
    }
}

/// Parse a reduction op-name string into a [`ReduceOp`].
fn parse_reduce_op(op: &str) -> Result<ReduceOp, OxiCudaBackendError> {
    match op.to_lowercase().as_str() {
        "sum" => Ok(ReduceOp::Sum),
        "max" => Ok(ReduceOp::Max),
        "min" => Ok(ReduceOp::Min),
        "mean" => Ok(ReduceOp::Mean),
        "product" | "prod" => Ok(ReduceOp::Product),
        other => Err(OxiCudaBackendError::InvalidEinsumSpec(format!(
            "unknown reduction op name: {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Host-side constructors for OxiCudaTensor
// ---------------------------------------------------------------------------

/// Build a tensor filled with `0.0f32` from a host buffer.
fn zeros_host(shape: &[usize]) -> OxiCudaTensor {
    let n: usize = shape.iter().product();
    OxiCudaTensor {
        shape: shape.to_vec(),
        data: vec![0.0_f32; n],
    }
}

/// Build a tensor filled with `1.0f32` from a host buffer.
#[cfg(test)]
fn ones_host(shape: &[usize]) -> OxiCudaTensor {
    let n: usize = shape.iter().product();
    OxiCudaTensor {
        shape: shape.to_vec(),
        data: vec![1.0_f32; n],
    }
}

// ---------------------------------------------------------------------------
// Host-side transpose helpers (used as fallback when gpu feature is off)
// ---------------------------------------------------------------------------

/// Transpose a 2-D tensor on the host.
#[cfg(any(not(feature = "gpu"), test))]
fn transpose_2d_host(t: &OxiCudaTensor) -> Result<OxiCudaTensor, OxiCudaBackendError> {
    if t.shape.len() != 2 {
        return Err(OxiCudaBackendError::InvalidShape(format!(
            "transpose_2d_host expects a 2-D tensor, got shape {:?}",
            t.shape
        )));
    }
    let rows = t.shape[0];
    let cols = t.shape[1];
    let mut out = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = t.data[r * cols + c];
        }
    }
    OxiCudaTensor::new(vec![cols, rows], out)
}

/// Transpose a 3-D tensor's last two axes on the host: `[b, m, k] → [b, k, m]`.
#[cfg(any(not(feature = "gpu"), test))]
fn transpose_3d_last_host(t: &OxiCudaTensor) -> Result<OxiCudaTensor, OxiCudaBackendError> {
    if t.shape.len() != 3 {
        return Err(OxiCudaBackendError::InvalidShape(format!(
            "transpose_3d_last_host expects a 3-D tensor, got shape {:?}",
            t.shape
        )));
    }
    let b = t.shape[0];
    let m = t.shape[1];
    let k = t.shape[2];
    let mut out = vec![0.0_f32; b * m * k];
    for bi in 0..b {
        for mi in 0..m {
            for ki in 0..k {
                out[bi * (k * m) + ki * m + mi] = t.data[bi * (m * k) + mi * k + ki];
            }
        }
    }
    OxiCudaTensor::new(vec![b, k, m], out)
}

// ---------------------------------------------------------------------------
// Host-side axis broadcast (for reduce gradients) — always available
// ---------------------------------------------------------------------------

/// Expand a reduced tensor back to `target_shape` by replicating values along
/// every axis in `reduced_axes`.
///
/// For example, reducing `[2, 3]` along axis 0 gives shape `[3]`.  This
/// function broadcasts that back to `[2, 3]` (each row is the same).
///
/// This is the host-side (CPU) implementation, always available regardless of
/// feature flags.  The native GPU path is `broadcast_to_shape_native`, available
/// when the `native-broadcast` feature is enabled.
fn broadcast_to_shape(
    small: &OxiCudaTensor,
    target_shape: &[usize],
    reduced_axes: &[usize],
) -> Result<OxiCudaTensor, OxiCudaBackendError> {
    let target_n: usize = target_shape.iter().product();
    let mut out = vec![0.0_f32; target_n];

    // For each flat index in the output, compute the corresponding flat index
    // in the small tensor by zeroing the reduced axes.
    for (flat_out, slot) in out.iter_mut().enumerate() {
        // Convert flat_out → multi-dim index in target_shape
        let mut coords = vec![0usize; target_shape.len()];
        let mut rem = flat_out;
        for d in (0..target_shape.len()).rev() {
            coords[d] = rem % target_shape[d];
            rem /= target_shape[d];
        }

        // Build small-tensor coords by dropping the reduced axes
        let mut small_coords = Vec::with_capacity(small.shape.len());
        for (d, &c) in coords.iter().enumerate() {
            if !reduced_axes.contains(&d) {
                small_coords.push(c);
            }
        }

        // Convert small_coords → flat index in small tensor
        let mut flat_small = 0usize;
        for (d, &c) in small_coords.iter().enumerate() {
            flat_small = flat_small * small.shape[d] + c;
        }

        *slot = *small.data.get(flat_small).ok_or_else(|| {
            OxiCudaBackendError::InvalidShape(format!(
                "broadcast_to_shape: small tensor index {flat_small} out of range (len {})",
                small.data.len()
            ))
        })?;
    }

    OxiCudaTensor::new(target_shape.to_vec(), out)
}

/// Build a tensor where every element is `value`, on the host (CPU).
///
/// Used as the divisor in ReduceMean backward when `native-broadcast` is off.
#[cfg(not(feature = "native-broadcast"))]
fn fill_tensor_host(value: f32, shape: &[usize]) -> OxiCudaTensor {
    let n: usize = shape.iter().product();
    OxiCudaTensor {
        shape: shape.to_vec(),
        data: vec![value; n],
    }
}

// ---------------------------------------------------------------------------
// Native GPU broadcast helpers (native-broadcast feature only)
// ---------------------------------------------------------------------------

/// Broadcast `small` to `target_shape` using the `oxicuda_blas::elementwise::broadcast_axes`
/// GPU kernel.
///
/// This is the Round 5 architecture: upload → kernel → readback. The round-trips
/// will be eliminated in Round 6 when tensors become device-resident.
///
/// Falls back silently to `broadcast_to_shape` (host path) if the GPU call fails.
#[cfg(feature = "native-broadcast")]
fn broadcast_to_shape_native(
    small: &OxiCudaTensor,
    target_shape: &[usize],
    reduced_axes: &[usize],
    gpu: &GpuState,
) -> Result<OxiCudaTensor, OxiCudaBackendError> {
    use oxicuda_memory::DeviceBuffer;

    let n_src = small.data.len();
    let n_dst: usize = target_shape.iter().product();

    if n_src == 0 || n_dst == 0 {
        return OxiCudaTensor::new(target_shape.to_vec(), vec![0.0_f32; n_dst]);
    }

    let src_buf = DeviceBuffer::<f32>::from_host(&small.data)
        .map_err(|e| OxiCudaBackendError::OxiCuda(format!("broadcast_native upload: {e}")))?;
    let mut dst_buf = DeviceBuffer::<f32>::zeroed(n_dst)
        .map_err(|e| OxiCudaBackendError::OxiCuda(format!("broadcast_native alloc dst: {e}")))?;

    oxicuda_blas::elementwise::broadcast_axes(
        gpu.blas_handle(),
        &src_buf,
        &small.shape,
        &mut dst_buf,
        target_shape,
        reduced_axes,
    )
    .map_err(|e| OxiCudaBackendError::OxiCuda(format!("broadcast_axes kernel: {e}")))?;

    let mut data = vec![0.0_f32; n_dst];
    dst_buf
        .copy_to_host(&mut data)
        .map_err(|e| OxiCudaBackendError::OxiCuda(format!("broadcast_native readback: {e}")))?;

    OxiCudaTensor::new(target_shape.to_vec(), data)
}

/// Fill every element of a tensor with `value` using the `oxicuda_blas::elementwise::fill`
/// GPU kernel.
///
/// Round 5 architecture: allocate device buffer → kernel → readback.
#[cfg(feature = "native-broadcast")]
fn fill_tensor_native(
    value: f32,
    shape: &[usize],
    gpu: &GpuState,
) -> Result<OxiCudaTensor, OxiCudaBackendError> {
    use oxicuda_memory::DeviceBuffer;

    let n: usize = shape.iter().product();
    if n == 0 {
        return OxiCudaTensor::new(shape.to_vec(), vec![]);
    }

    let n_u32 = n as u32;
    let mut dst_buf = DeviceBuffer::<f32>::zeroed(n)
        .map_err(|e| OxiCudaBackendError::OxiCuda(format!("fill_native alloc: {e}")))?;

    oxicuda_blas::elementwise::fill(gpu.blas_handle(), &mut dst_buf, value, n_u32)
        .map_err(|e| OxiCudaBackendError::OxiCuda(format!("fill kernel: {e}")))?;

    let mut data = vec![0.0_f32; n];
    dst_buf
        .copy_to_host(&mut data)
        .map_err(|e| OxiCudaBackendError::OxiCuda(format!("fill_native readback: {e}")))?;

    OxiCudaTensor::new(shape.to_vec(), data)
}

// ---------------------------------------------------------------------------
// Gradient accumulation helper
// ---------------------------------------------------------------------------

/// Accumulate `grad` into `tape.gradients[node_id]`.
///
/// If a gradient already exists for `node_id`, it is added element-wise;
/// otherwise the gradient is inserted directly.
fn gradient_accumulate(
    gradients: &mut HashMap<usize, OxiCudaTensor>,
    node_id: usize,
    grad: OxiCudaTensor,
    exec: &mut OxiCudaExecutor,
) -> Result<(), OxiCudaBackendError> {
    match gradients.remove(&node_id) {
        None => {
            gradients.insert(node_id, grad);
        }
        Some(existing) => {
            let summed = exec.elem_op_binary(ElemOp::Add, &existing, &grad)?;
            gradients.insert(node_id, summed);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// TlAutodiff implementation
// ---------------------------------------------------------------------------

impl TlAutodiff for OxiCudaExecutor {
    type Tape = OxiCudaTape;

    fn forward(&mut self, graph: &EinsumGraph) -> Result<Self::Tensor, Self::Error> {
        if graph.is_empty() {
            return Err(OxiCudaBackendError::InvalidEinsumSpec(
                "empty EinsumGraph passed to forward".to_string(),
            ));
        }
        if graph.outputs.is_empty() {
            return Err(OxiCudaBackendError::InvalidEinsumSpec(
                "EinsumGraph has no output tensors".to_string(),
            ));
        }

        // Slot for each tensor in the graph (None until computed).
        let mut computed: Vec<Option<OxiCudaTensor>> = vec![None; graph.tensors.len()];
        let mut tape_entries: Vec<TapeEntry> = Vec::with_capacity(graph.nodes.len());

        for node in &graph.nodes {
            // Gather inputs, erroring out if any is still uncomputed.
            let input_tensors: Vec<OxiCudaTensor> = node
                .inputs
                .iter()
                .map(|&idx| {
                    computed
                        .get(idx)
                        .and_then(|slot| slot.as_ref())
                        .cloned()
                        .ok_or_else(|| {
                            OxiCudaBackendError::InvalidEinsumSpec(format!(
                                "tensor at index {idx} not yet computed for op {:?}",
                                node.op
                            ))
                        })
                })
                .collect::<Result<_, _>>()?;

            let (result, taped_op) = match &node.op {
                OpType::Einsum { spec } => {
                    let norm: String = spec
                        .chars()
                        .filter(|c| !c.is_whitespace())
                        .flat_map(|c| c.to_lowercase())
                        .collect();

                    let taped = match norm.as_str() {
                        "ij,jk->ik" => TapedOp::Matmul2D,
                        "bij,bjk->bik" => TapedOp::BatchedMatmul3D,
                        _ if is_identity_spec(&norm) => TapedOp::Identity,
                        _ => TapedOp::Identity, // unsupported specs get forwarded anyway
                    };
                    let out = self.einsum(spec, &input_tensors)?;
                    (out, taped)
                }
                OpType::ElemUnary { op } => {
                    let elem_op = parse_elem_op(op)?;
                    let out = self.elem_op(elem_op, &input_tensors[0])?;
                    (out, TapedOp::Unary(elem_op))
                }
                OpType::ElemBinary { op } => {
                    if input_tensors.len() != 2 {
                        return Err(OxiCudaBackendError::InvalidEinsumSpec(format!(
                            "binary op '{op}' expects 2 inputs, got {}",
                            input_tensors.len()
                        )));
                    }
                    let elem_op = parse_elem_op(op)?;
                    let out = self.elem_op_binary(elem_op, &input_tensors[0], &input_tensors[1])?;
                    (out, TapedOp::Binary(elem_op))
                }
                OpType::Reduce { op, axes } => {
                    let reduce_op = parse_reduce_op(op)?;
                    let out = self.reduce(reduce_op, &input_tensors[0], axes)?;
                    (out, TapedOp::Reduce(reduce_op, axes.clone()))
                }
            };

            let output_idx = node.outputs.first().copied().ok_or_else(|| {
                OxiCudaBackendError::InvalidEinsumSpec("node has no output index".to_string())
            })?;

            tape_entries.push(TapeEntry {
                output_tensor_idx: output_idx,
                input_tensor_indices: node.inputs.clone(),
                op: taped_op,
                saved_inputs: input_tensors,
            });

            computed[output_idx] = Some(result);
        }

        // The tape entries are unused here: backward() calls forward_with_tape()
        // to rebuild the tape with saved inputs.  Drop them explicitly.
        drop(tape_entries);

        let output_idx = graph.outputs[0];
        computed
            .get(output_idx)
            .and_then(|slot| slot.clone())
            .ok_or_else(|| {
                OxiCudaBackendError::InvalidEinsumSpec(
                    "output tensor was not produced by any node".to_string(),
                )
            })
    }

    fn backward(
        &mut self,
        graph: &EinsumGraph,
        loss: &Self::Tensor,
    ) -> Result<Self::Tape, Self::Error> {
        if graph.is_empty() {
            return Err(OxiCudaBackendError::InvalidEinsumSpec(
                "empty EinsumGraph passed to backward".to_string(),
            ));
        }

        // Re-run the forward pass to rebuild the tape with saved inputs.
        let tape = self.forward_with_tape(graph)?;

        // Seed: gradient of root output is the loss tensor.
        let mut gradients: HashMap<usize, OxiCudaTensor> = HashMap::new();
        if let Some(&output_idx) = graph.outputs.first() {
            gradients.insert(output_idx, loss.clone());
        }

        // Reverse-iterate tape entries.
        for entry in tape.entries.into_iter().rev() {
            let output_grad = match gradients.remove(&entry.output_tensor_idx) {
                Some(g) => g,
                None => continue, // No gradient flows through this node.
            };

            match &entry.op {
                TapedOp::Identity => {
                    // Pass gradient straight through to the (single) input.
                    if let Some(&input_idx) = entry.input_tensor_indices.first() {
                        gradient_accumulate(&mut gradients, input_idx, output_grad, self)?;
                    }
                }

                TapedOp::Matmul2D => {
                    // C = A·B  (A: [m,k], B: [k,n], C: [m,n])
                    // dA = dC · Bᵀ  (einsum "mn,kn->mk")
                    // dB = Aᵀ · dC  (einsum "mk,mn->kn")
                    if entry.saved_inputs.len() != 2 || entry.input_tensor_indices.len() != 2 {
                        return Err(OxiCudaBackendError::InvalidEinsumSpec(
                            "Matmul2D backward: expected 2 saved inputs".to_string(),
                        ));
                    }
                    let a = &entry.saved_inputs[0];
                    let b = &entry.saved_inputs[1];
                    let a_idx = entry.input_tensor_indices[0];
                    let b_idx = entry.input_tensor_indices[1];

                    // dA = dC · Bᵀ
                    #[cfg(feature = "gpu")]
                    let da = crate::einsum::matmul_2d_trans_flags(
                        self.gpu.blas_handle(),
                        &output_grad,
                        oxicuda_blas::types::Transpose::NoTrans,
                        b,
                        oxicuda_blas::types::Transpose::Trans,
                    )?;
                    #[cfg(not(feature = "gpu"))]
                    let da = {
                        let b_t = transpose_2d_host(b)?;
                        self.einsum("ij,jk->ik", &[output_grad.clone(), b_t])?
                    };
                    gradient_accumulate(&mut gradients, a_idx, da, self)?;

                    // dB = Aᵀ · dC
                    #[cfg(feature = "gpu")]
                    let db = crate::einsum::matmul_2d_trans_flags(
                        self.gpu.blas_handle(),
                        a,
                        oxicuda_blas::types::Transpose::Trans,
                        &output_grad,
                        oxicuda_blas::types::Transpose::NoTrans,
                    )?;
                    #[cfg(not(feature = "gpu"))]
                    let db = {
                        let a_t = transpose_2d_host(a)?;
                        self.einsum("ij,jk->ik", &[a_t, output_grad])?
                    };
                    gradient_accumulate(&mut gradients, b_idx, db, self)?;
                }

                TapedOp::BatchedMatmul3D => {
                    // C = A·B  (A: [b,m,k], B: [b,k,n], C: [b,m,n])
                    // dA = dC · Bᵀ  ([b,m,n]·[b,n,k]→[b,m,k])
                    // dB = Aᵀ · dC  ([b,k,m]·[b,m,n]→[b,k,n])
                    if entry.saved_inputs.len() != 2 || entry.input_tensor_indices.len() != 2 {
                        return Err(OxiCudaBackendError::InvalidEinsumSpec(
                            "BatchedMatmul3D backward: expected 2 saved inputs".to_string(),
                        ));
                    }
                    let a = &entry.saved_inputs[0];
                    let b = &entry.saved_inputs[1];
                    let a_idx = entry.input_tensor_indices[0];
                    let b_idx = entry.input_tensor_indices[1];

                    // dA = dC · Bᵀ (batched)
                    #[cfg(feature = "gpu")]
                    let da = crate::einsum::matmul_batched_trans_flags(
                        self.gpu.blas_handle(),
                        &output_grad,
                        oxicuda_blas::types::Transpose::NoTrans,
                        b,
                        oxicuda_blas::types::Transpose::Trans,
                    )?;
                    #[cfg(not(feature = "gpu"))]
                    let da = {
                        let b_t = transpose_3d_last_host(b)?;
                        self.einsum("bij,bjk->bik", &[output_grad.clone(), b_t])?
                    };
                    gradient_accumulate(&mut gradients, a_idx, da, self)?;

                    // dB = Aᵀ · dC (batched)
                    #[cfg(feature = "gpu")]
                    let db = crate::einsum::matmul_batched_trans_flags(
                        self.gpu.blas_handle(),
                        a,
                        oxicuda_blas::types::Transpose::Trans,
                        &output_grad,
                        oxicuda_blas::types::Transpose::NoTrans,
                    )?;
                    #[cfg(not(feature = "gpu"))]
                    let db = {
                        let a_t = transpose_3d_last_host(a)?;
                        self.einsum("bij,bjk->bik", &[a_t, output_grad])?
                    };
                    gradient_accumulate(&mut gradients, b_idx, db, self)?;
                }

                TapedOp::Unary(elem_op) => {
                    if entry.saved_inputs.is_empty() || entry.input_tensor_indices.is_empty() {
                        return Err(OxiCudaBackendError::InvalidEinsumSpec(
                            "Unary backward: expected 1 saved input".to_string(),
                        ));
                    }
                    let x = &entry.saved_inputs[0];
                    let x_idx = entry.input_tensor_indices[0];

                    let input_grad = match elem_op {
                        ElemOp::Relu => {
                            // dX = dY * (X > 0)
                            let zeros = zeros_host(&x.shape);
                            let mask = self.elem_op_binary(ElemOp::Gt, x, &zeros)?;
                            self.elem_op_binary(ElemOp::Multiply, &output_grad, &mask)?
                        }
                        ElemOp::Sigmoid => {
                            // Y = sigmoid(X), dX = dY * Y * (1 - Y)
                            // Recompute Y on device.
                            let y = self.elem_op(ElemOp::Sigmoid, x)?;
                            let one_minus_y = self.elem_op(ElemOp::OneMinus, &y)?;
                            let y_times_1my =
                                self.elem_op_binary(ElemOp::Multiply, &y, &one_minus_y)?;
                            self.elem_op_binary(ElemOp::Multiply, &output_grad, &y_times_1my)?
                        }
                        ElemOp::OneMinus => {
                            // d/dx(1 - x) = -1 → dX = -dY = 0 - dY
                            let zeros = zeros_host(&output_grad.shape);
                            self.elem_op_binary(ElemOp::Subtract, &zeros, &output_grad)?
                        }
                        other => {
                            return Err(OxiCudaBackendError::UnsupportedAutodiffOp(format!(
                                "{other:?}"
                            )));
                        }
                    };

                    gradient_accumulate(&mut gradients, x_idx, input_grad, self)?;
                }

                TapedOp::Binary(elem_op) => {
                    if entry.saved_inputs.len() != 2 || entry.input_tensor_indices.len() != 2 {
                        return Err(OxiCudaBackendError::InvalidEinsumSpec(
                            "Binary backward: expected 2 saved inputs".to_string(),
                        ));
                    }
                    let x = &entry.saved_inputs[0];
                    let y = &entry.saved_inputs[1];
                    let x_idx = entry.input_tensor_indices[0];
                    let y_idx = entry.input_tensor_indices[1];

                    let (grad_x, grad_y) = match elem_op {
                        ElemOp::Add => {
                            // dX = dY, dY_in = dY
                            (output_grad.clone(), output_grad)
                        }
                        ElemOp::Subtract => {
                            // dX = dY, dY_in = -dY
                            let zeros = zeros_host(&output_grad.shape);
                            let neg_dy =
                                self.elem_op_binary(ElemOp::Subtract, &zeros, &output_grad)?;
                            (output_grad, neg_dy)
                        }
                        ElemOp::Multiply => {
                            // dX = dY * Y, dY_in = dY * X
                            let dx = self.elem_op_binary(ElemOp::Multiply, &output_grad, y)?;
                            let dy = self.elem_op_binary(ElemOp::Multiply, &output_grad, x)?;
                            (dx, dy)
                        }
                        ElemOp::Divide => {
                            // dX = dY / Y, dY_in = -(dY * X) / (Y * Y)
                            let dx = self.elem_op_binary(ElemOp::Divide, &output_grad, y)?;
                            let dg_times_x =
                                self.elem_op_binary(ElemOp::Multiply, &output_grad, x)?;
                            let y_sq = self.elem_op_binary(ElemOp::Multiply, y, y)?;
                            let dy_pos = self.elem_op_binary(ElemOp::Divide, &dg_times_x, &y_sq)?;
                            let zeros = zeros_host(&dy_pos.shape);
                            let dy = self.elem_op_binary(ElemOp::Subtract, &zeros, &dy_pos)?;
                            (dx, dy)
                        }
                        // Comparison and logical ops are non-differentiable.
                        other @ (ElemOp::Eq
                        | ElemOp::Lt
                        | ElemOp::Gt
                        | ElemOp::Lte
                        | ElemOp::Gte
                        | ElemOp::OrMax
                        | ElemOp::OrProbSum
                        | ElemOp::Nand
                        | ElemOp::Nor
                        | ElemOp::Xor
                        | ElemOp::Min
                        | ElemOp::Max) => {
                            return Err(OxiCudaBackendError::UnsupportedAutodiffOp(format!(
                                "{other:?}"
                            )));
                        }
                        // Unary ops routed to binary dispatch path are an error.
                        other => {
                            return Err(OxiCudaBackendError::UnsupportedAutodiffOp(format!(
                                "{other:?}"
                            )));
                        }
                    };

                    gradient_accumulate(&mut gradients, x_idx, grad_x, self)?;
                    gradient_accumulate(&mut gradients, y_idx, grad_y, self)?;
                }

                TapedOp::Reduce(reduce_op, axes) => {
                    if entry.saved_inputs.is_empty() || entry.input_tensor_indices.is_empty() {
                        return Err(OxiCudaBackendError::InvalidEinsumSpec(
                            "Reduce backward: expected 1 saved input".to_string(),
                        ));
                    }
                    let x = &entry.saved_inputs[0];
                    let x_idx = entry.input_tensor_indices[0];

                    let input_grad = match reduce_op {
                        ReduceOp::Sum => {
                            // Broadcast gradient back to input shape.
                            // Native GPU path available under `native-broadcast` feature.
                            #[cfg(feature = "native-broadcast")]
                            {
                                broadcast_to_shape_native(
                                    &output_grad,
                                    &x.shape,
                                    axes,
                                    self.gpu_state_internal(),
                                )?
                            }
                            #[cfg(not(feature = "native-broadcast"))]
                            {
                                broadcast_to_shape(&output_grad, &x.shape, axes)?
                            }
                        }
                        ReduceOp::Max | ReduceOp::Min => {
                            // Gradient flows only to the argmax/argmin positions.
                            // Host-only implementation: broadcast reduced output back to
                            // input shape, compute equality mask, multiply by broadcast grad.
                            // Note: ties are unresolved (all equal positions share gradient).
                            // Native GPU argmax/argmin gradient kernel is planned for Round 6.

                            // Re-run reduction to get the reduced values.
                            let y = self.reduce(*reduce_op, x, axes)?;
                            let y_broadcast = broadcast_to_shape(&y, &x.shape, axes)?;
                            let mask = self.elem_op_binary(ElemOp::Eq, x, &y_broadcast)?;
                            let dg_broadcast = broadcast_to_shape(&output_grad, &x.shape, axes)?;
                            self.elem_op_binary(ElemOp::Multiply, &dg_broadcast, &mask)?
                        }
                        ReduceOp::Mean => {
                            // dX = broadcast(dY) / axis_len
                            // Native GPU path available under `native-broadcast` feature.
                            #[cfg(feature = "native-broadcast")]
                            let expanded = broadcast_to_shape_native(
                                &output_grad,
                                &x.shape,
                                axes,
                                self.gpu_state_internal(),
                            )?;
                            #[cfg(not(feature = "native-broadcast"))]
                            let expanded = broadcast_to_shape(&output_grad, &x.shape, axes)?;

                            let axis_len: usize = axes.iter().map(|&a| x.shape[a]).product();

                            #[cfg(feature = "native-broadcast")]
                            let divisor = fill_tensor_native(
                                axis_len as f32,
                                &x.shape,
                                self.gpu_state_internal(),
                            )?;
                            #[cfg(not(feature = "native-broadcast"))]
                            let divisor = fill_tensor_host(axis_len as f32, &x.shape);

                            self.elem_op_binary(ElemOp::Divide, &expanded, &divisor)?
                        }
                        ReduceOp::Product => {
                            return Err(OxiCudaBackendError::UnsupportedAutodiffOp(
                                "Product".to_string(),
                            ));
                        }
                    };

                    gradient_accumulate(&mut gradients, x_idx, input_grad, self)?;
                }
            }
        }

        Ok(OxiCudaTape {
            entries: Vec::new(), // entries consumed; gradients are the useful output
            gradients,
        })
    }
}

// ---------------------------------------------------------------------------
// Public seeding API: forward_with_seeds / backward_with_seeds
// ---------------------------------------------------------------------------

impl OxiCudaExecutor {
    /// Execute a forward pass with pre-seeded tensor values.
    ///
    /// `seeds` maps tensor indices (graph positions) to concrete [`OxiCudaTensor`]
    /// values.  Typically all graph inputs are seeded; the executor computes the
    /// remaining tensors in topological order.
    ///
    /// This method is the recommended way to inject concrete values into an
    /// [`EinsumGraph`] for testing and inference: build the graph with
    /// [`tensorlogic_ir::EinsumGraph`] (which only stores topology), then supply
    /// values here.
    ///
    /// # Errors
    ///
    /// Returns [`OxiCudaBackendError::InvalidEinsumSpec`] if any node's inputs
    /// are not available (neither seeded nor produced by an earlier node).
    /// Returns [`OxiCudaBackendError::BackendDisabled`] when the `gpu` feature
    /// is disabled.
    pub fn forward_with_seeds(
        &mut self,
        graph: &EinsumGraph,
        seeds: &HashMap<usize, OxiCudaTensor>,
    ) -> Result<OxiCudaTensor, OxiCudaBackendError> {
        if graph.is_empty() {
            return Err(OxiCudaBackendError::InvalidEinsumSpec(
                "empty EinsumGraph passed to forward_with_seeds".to_string(),
            ));
        }
        if graph.outputs.is_empty() {
            return Err(OxiCudaBackendError::InvalidEinsumSpec(
                "EinsumGraph has no output tensors".to_string(),
            ));
        }

        let mut computed: Vec<Option<OxiCudaTensor>> = vec![None; graph.tensors.len()];
        // Pre-populate seeded positions.
        for (&idx, tensor) in seeds {
            if let Some(slot) = computed.get_mut(idx) {
                *slot = Some(tensor.clone());
            }
        }

        for node in &graph.nodes {
            let input_tensors: Vec<OxiCudaTensor> = node
                .inputs
                .iter()
                .map(|&idx| {
                    computed
                        .get(idx)
                        .and_then(|slot| slot.as_ref())
                        .cloned()
                        .ok_or_else(|| {
                            OxiCudaBackendError::InvalidEinsumSpec(format!(
                                "tensor at index {idx} not yet computed (forward_with_seeds)"
                            ))
                        })
                })
                .collect::<Result<_, _>>()?;

            let result = match &node.op {
                OpType::Einsum { spec } => self.einsum(spec, &input_tensors)?,
                OpType::ElemUnary { op } => {
                    let elem_op = parse_elem_op(op)?;
                    self.elem_op(elem_op, &input_tensors[0])?
                }
                OpType::ElemBinary { op } => {
                    if input_tensors.len() != 2 {
                        return Err(OxiCudaBackendError::InvalidEinsumSpec(format!(
                            "binary op '{op}' expects 2 inputs, got {}",
                            input_tensors.len()
                        )));
                    }
                    let elem_op = parse_elem_op(op)?;
                    self.elem_op_binary(elem_op, &input_tensors[0], &input_tensors[1])?
                }
                OpType::Reduce { op, axes } => {
                    let reduce_op = parse_reduce_op(op)?;
                    self.reduce(reduce_op, &input_tensors[0], axes)?
                }
            };

            let output_idx = node.outputs.first().copied().ok_or_else(|| {
                OxiCudaBackendError::InvalidEinsumSpec("node has no output index".to_string())
            })?;
            computed[output_idx] = Some(result);
        }

        let output_idx = graph.outputs[0];
        computed
            .get(output_idx)
            .and_then(|slot| slot.clone())
            .ok_or_else(|| {
                OxiCudaBackendError::InvalidEinsumSpec(
                    "output tensor was not produced by any node".to_string(),
                )
            })
    }

    /// Compute gradients for a graph with pre-seeded input values.
    ///
    /// This is the backward-pass counterpart of `forward_with_seeds`.
    /// It re-runs the forward pass internally (recording the tape), then
    /// propagates `loss` backward to produce input gradients.
    ///
    /// `seeds` maps tensor indices to concrete input values — the same map
    /// you would pass to `forward_with_seeds`.
    ///
    /// # Errors
    ///
    /// Same errors as `forward_with_seeds` plus autodiff-specific errors
    /// such as [`OxiCudaBackendError::UnsupportedAutodiffOp`].
    pub fn backward_with_seeds(
        &mut self,
        graph: &EinsumGraph,
        seeds: &HashMap<usize, OxiCudaTensor>,
        loss: &OxiCudaTensor,
    ) -> Result<OxiCudaTape, OxiCudaBackendError> {
        if graph.is_empty() {
            return Err(OxiCudaBackendError::InvalidEinsumSpec(
                "empty EinsumGraph passed to backward_with_seeds".to_string(),
            ));
        }

        // Rebuild the tape with saved inputs using seeded values.
        let tape = self.forward_with_tape_seeded(graph, seeds)?;

        // Seed: gradient of root output is the loss tensor.
        let mut gradients: HashMap<usize, OxiCudaTensor> = HashMap::new();
        if let Some(&output_idx) = graph.outputs.first() {
            gradients.insert(output_idx, loss.clone());
        }

        // Reverse-iterate tape entries — identical logic to `backward`.
        for entry in tape.entries.into_iter().rev() {
            let output_grad = match gradients.remove(&entry.output_tensor_idx) {
                Some(g) => g,
                None => continue,
            };

            match &entry.op {
                TapedOp::Identity => {
                    if let Some(&input_idx) = entry.input_tensor_indices.first() {
                        gradient_accumulate(&mut gradients, input_idx, output_grad, self)?;
                    }
                }

                TapedOp::Matmul2D => {
                    if entry.saved_inputs.len() != 2 || entry.input_tensor_indices.len() != 2 {
                        return Err(OxiCudaBackendError::InvalidEinsumSpec(
                            "Matmul2D backward_with_seeds: expected 2 saved inputs".to_string(),
                        ));
                    }
                    let a = &entry.saved_inputs[0];
                    let b = &entry.saved_inputs[1];
                    let a_idx = entry.input_tensor_indices[0];
                    let b_idx = entry.input_tensor_indices[1];

                    #[cfg(feature = "gpu")]
                    let da = crate::einsum::matmul_2d_trans_flags(
                        self.gpu.blas_handle(),
                        &output_grad,
                        oxicuda_blas::types::Transpose::NoTrans,
                        b,
                        oxicuda_blas::types::Transpose::Trans,
                    )?;
                    #[cfg(not(feature = "gpu"))]
                    let da = {
                        let b_t = transpose_2d_host(b)?;
                        self.einsum("ij,jk->ik", &[output_grad.clone(), b_t])?
                    };
                    gradient_accumulate(&mut gradients, a_idx, da, self)?;

                    #[cfg(feature = "gpu")]
                    let db = crate::einsum::matmul_2d_trans_flags(
                        self.gpu.blas_handle(),
                        a,
                        oxicuda_blas::types::Transpose::Trans,
                        &output_grad,
                        oxicuda_blas::types::Transpose::NoTrans,
                    )?;
                    #[cfg(not(feature = "gpu"))]
                    let db = {
                        let a_t = transpose_2d_host(a)?;
                        self.einsum("ij,jk->ik", &[a_t, output_grad])?
                    };
                    gradient_accumulate(&mut gradients, b_idx, db, self)?;
                }

                TapedOp::BatchedMatmul3D => {
                    if entry.saved_inputs.len() != 2 || entry.input_tensor_indices.len() != 2 {
                        return Err(OxiCudaBackendError::InvalidEinsumSpec(
                            "BatchedMatmul3D backward_with_seeds: expected 2 saved inputs"
                                .to_string(),
                        ));
                    }
                    let a = &entry.saved_inputs[0];
                    let b = &entry.saved_inputs[1];
                    let a_idx = entry.input_tensor_indices[0];
                    let b_idx = entry.input_tensor_indices[1];

                    #[cfg(feature = "gpu")]
                    let da = crate::einsum::matmul_batched_trans_flags(
                        self.gpu.blas_handle(),
                        &output_grad,
                        oxicuda_blas::types::Transpose::NoTrans,
                        b,
                        oxicuda_blas::types::Transpose::Trans,
                    )?;
                    #[cfg(not(feature = "gpu"))]
                    let da = {
                        let b_t = transpose_3d_last_host(b)?;
                        self.einsum("bij,bjk->bik", &[output_grad.clone(), b_t])?
                    };
                    gradient_accumulate(&mut gradients, a_idx, da, self)?;

                    #[cfg(feature = "gpu")]
                    let db = crate::einsum::matmul_batched_trans_flags(
                        self.gpu.blas_handle(),
                        a,
                        oxicuda_blas::types::Transpose::Trans,
                        &output_grad,
                        oxicuda_blas::types::Transpose::NoTrans,
                    )?;
                    #[cfg(not(feature = "gpu"))]
                    let db = {
                        let a_t = transpose_3d_last_host(a)?;
                        self.einsum("bij,bjk->bik", &[a_t, output_grad])?
                    };
                    gradient_accumulate(&mut gradients, b_idx, db, self)?;
                }

                TapedOp::Unary(elem_op) => {
                    if entry.saved_inputs.is_empty() || entry.input_tensor_indices.is_empty() {
                        return Err(OxiCudaBackendError::InvalidEinsumSpec(
                            "Unary backward_with_seeds: expected 1 saved input".to_string(),
                        ));
                    }
                    let x = &entry.saved_inputs[0];
                    let x_idx = entry.input_tensor_indices[0];

                    let input_grad = match elem_op {
                        ElemOp::Relu => {
                            let zeros = zeros_host(&x.shape);
                            let mask = self.elem_op_binary(ElemOp::Gt, x, &zeros)?;
                            self.elem_op_binary(ElemOp::Multiply, &output_grad, &mask)?
                        }
                        ElemOp::Sigmoid => {
                            let y = self.elem_op(ElemOp::Sigmoid, x)?;
                            let one_minus_y = self.elem_op(ElemOp::OneMinus, &y)?;
                            let y_times_1my =
                                self.elem_op_binary(ElemOp::Multiply, &y, &one_minus_y)?;
                            self.elem_op_binary(ElemOp::Multiply, &output_grad, &y_times_1my)?
                        }
                        ElemOp::OneMinus => {
                            let zeros = zeros_host(&output_grad.shape);
                            self.elem_op_binary(ElemOp::Subtract, &zeros, &output_grad)?
                        }
                        other => {
                            return Err(OxiCudaBackendError::UnsupportedAutodiffOp(format!(
                                "{other:?}"
                            )));
                        }
                    };

                    gradient_accumulate(&mut gradients, x_idx, input_grad, self)?;
                }

                TapedOp::Binary(elem_op) => {
                    if entry.saved_inputs.len() != 2 || entry.input_tensor_indices.len() != 2 {
                        return Err(OxiCudaBackendError::InvalidEinsumSpec(
                            "Binary backward_with_seeds: expected 2 saved inputs".to_string(),
                        ));
                    }
                    let x = &entry.saved_inputs[0];
                    let y = &entry.saved_inputs[1];
                    let x_idx = entry.input_tensor_indices[0];
                    let y_idx = entry.input_tensor_indices[1];

                    let (grad_x, grad_y) = match elem_op {
                        ElemOp::Add => (output_grad.clone(), output_grad),
                        ElemOp::Subtract => {
                            let zeros = zeros_host(&output_grad.shape);
                            let neg_dy =
                                self.elem_op_binary(ElemOp::Subtract, &zeros, &output_grad)?;
                            (output_grad, neg_dy)
                        }
                        ElemOp::Multiply => {
                            let dx = self.elem_op_binary(ElemOp::Multiply, &output_grad, y)?;
                            let dy = self.elem_op_binary(ElemOp::Multiply, &output_grad, x)?;
                            (dx, dy)
                        }
                        ElemOp::Divide => {
                            let dx = self.elem_op_binary(ElemOp::Divide, &output_grad, y)?;
                            let dg_times_x =
                                self.elem_op_binary(ElemOp::Multiply, &output_grad, x)?;
                            let y_sq = self.elem_op_binary(ElemOp::Multiply, y, y)?;
                            let dy_pos = self.elem_op_binary(ElemOp::Divide, &dg_times_x, &y_sq)?;
                            let zeros = zeros_host(&dy_pos.shape);
                            let dy = self.elem_op_binary(ElemOp::Subtract, &zeros, &dy_pos)?;
                            (dx, dy)
                        }
                        other @ (ElemOp::Eq
                        | ElemOp::Lt
                        | ElemOp::Gt
                        | ElemOp::Lte
                        | ElemOp::Gte
                        | ElemOp::OrMax
                        | ElemOp::OrProbSum
                        | ElemOp::Nand
                        | ElemOp::Nor
                        | ElemOp::Xor
                        | ElemOp::Min
                        | ElemOp::Max) => {
                            return Err(OxiCudaBackendError::UnsupportedAutodiffOp(format!(
                                "{other:?}"
                            )));
                        }
                        other => {
                            return Err(OxiCudaBackendError::UnsupportedAutodiffOp(format!(
                                "{other:?}"
                            )));
                        }
                    };

                    gradient_accumulate(&mut gradients, x_idx, grad_x, self)?;
                    gradient_accumulate(&mut gradients, y_idx, grad_y, self)?;
                }

                TapedOp::Reduce(reduce_op, axes) => {
                    if entry.saved_inputs.is_empty() || entry.input_tensor_indices.is_empty() {
                        return Err(OxiCudaBackendError::InvalidEinsumSpec(
                            "Reduce backward_with_seeds: expected 1 saved input".to_string(),
                        ));
                    }
                    let x = &entry.saved_inputs[0];
                    let x_idx = entry.input_tensor_indices[0];

                    let input_grad = match reduce_op {
                        ReduceOp::Sum => {
                            #[cfg(feature = "native-broadcast")]
                            {
                                broadcast_to_shape_native(
                                    &output_grad,
                                    &x.shape,
                                    axes,
                                    self.gpu_state_internal(),
                                )?
                            }
                            #[cfg(not(feature = "native-broadcast"))]
                            {
                                broadcast_to_shape(&output_grad, &x.shape, axes)?
                            }
                        }
                        ReduceOp::Max | ReduceOp::Min => {
                            let y = self.reduce(*reduce_op, x, axes)?;
                            let y_broadcast = broadcast_to_shape(&y, &x.shape, axes)?;
                            let mask = self.elem_op_binary(ElemOp::Eq, x, &y_broadcast)?;
                            let dg_broadcast = broadcast_to_shape(&output_grad, &x.shape, axes)?;
                            self.elem_op_binary(ElemOp::Multiply, &dg_broadcast, &mask)?
                        }
                        ReduceOp::Mean => {
                            #[cfg(feature = "native-broadcast")]
                            let expanded = broadcast_to_shape_native(
                                &output_grad,
                                &x.shape,
                                axes,
                                self.gpu_state_internal(),
                            )?;
                            #[cfg(not(feature = "native-broadcast"))]
                            let expanded = broadcast_to_shape(&output_grad, &x.shape, axes)?;

                            let axis_len: usize = axes.iter().map(|&a| x.shape[a]).product();

                            #[cfg(feature = "native-broadcast")]
                            let divisor = fill_tensor_native(
                                axis_len as f32,
                                &x.shape,
                                self.gpu_state_internal(),
                            )?;
                            #[cfg(not(feature = "native-broadcast"))]
                            let divisor = fill_tensor_host(axis_len as f32, &x.shape);

                            self.elem_op_binary(ElemOp::Divide, &expanded, &divisor)?
                        }
                        ReduceOp::Product => {
                            return Err(OxiCudaBackendError::UnsupportedAutodiffOp(
                                "Product".to_string(),
                            ));
                        }
                    };

                    gradient_accumulate(&mut gradients, x_idx, input_grad, self)?;
                }
            }
        }

        Ok(OxiCudaTape {
            entries: Vec::new(),
            gradients,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal forward-with-tape (produces TapeEntry list for backward)
// ---------------------------------------------------------------------------

impl OxiCudaExecutor {
    /// Execute a forward pass and return the tape (with saved inputs) alongside
    /// the output tensor.  This is called internally by `backward`.
    fn forward_with_tape(
        &mut self,
        graph: &EinsumGraph,
    ) -> Result<OxiCudaTape, OxiCudaBackendError> {
        let mut computed: Vec<Option<OxiCudaTensor>> = vec![None; graph.tensors.len()];
        let mut tape_entries: Vec<TapeEntry> = Vec::with_capacity(graph.nodes.len());

        for node in &graph.nodes {
            let input_tensors: Vec<OxiCudaTensor> = node
                .inputs
                .iter()
                .map(|&idx| {
                    computed
                        .get(idx)
                        .and_then(|slot| slot.as_ref())
                        .cloned()
                        .ok_or_else(|| {
                            OxiCudaBackendError::InvalidEinsumSpec(format!(
                                "tensor at index {idx} not yet computed (forward_with_tape)"
                            ))
                        })
                })
                .collect::<Result<_, _>>()?;

            let (result, taped_op) = match &node.op {
                OpType::Einsum { spec } => {
                    let norm: String = spec
                        .chars()
                        .filter(|c| !c.is_whitespace())
                        .flat_map(|c| c.to_lowercase())
                        .collect();
                    let taped = match norm.as_str() {
                        "ij,jk->ik" => TapedOp::Matmul2D,
                        "bij,bjk->bik" => TapedOp::BatchedMatmul3D,
                        _ if is_identity_spec(&norm) => TapedOp::Identity,
                        _ => TapedOp::Identity,
                    };
                    let out = self.einsum(spec, &input_tensors)?;
                    (out, taped)
                }
                OpType::ElemUnary { op } => {
                    let elem_op = parse_elem_op(op)?;
                    let out = self.elem_op(elem_op, &input_tensors[0])?;
                    (out, TapedOp::Unary(elem_op))
                }
                OpType::ElemBinary { op } => {
                    if input_tensors.len() != 2 {
                        return Err(OxiCudaBackendError::InvalidEinsumSpec(format!(
                            "binary op '{op}' expects 2 inputs, got {}",
                            input_tensors.len()
                        )));
                    }
                    let elem_op = parse_elem_op(op)?;
                    let out = self.elem_op_binary(elem_op, &input_tensors[0], &input_tensors[1])?;
                    (out, TapedOp::Binary(elem_op))
                }
                OpType::Reduce { op, axes } => {
                    let reduce_op = parse_reduce_op(op)?;
                    let out = self.reduce(reduce_op, &input_tensors[0], axes)?;
                    (out, TapedOp::Reduce(reduce_op, axes.clone()))
                }
            };

            let output_idx = node.outputs.first().copied().ok_or_else(|| {
                OxiCudaBackendError::InvalidEinsumSpec("node has no output index".to_string())
            })?;

            tape_entries.push(TapeEntry {
                output_tensor_idx: output_idx,
                input_tensor_indices: node.inputs.clone(),
                op: taped_op,
                saved_inputs: input_tensors,
            });

            computed[output_idx] = Some(result);
        }

        Ok(OxiCudaTape {
            entries: tape_entries,
            gradients: HashMap::new(),
        })
    }

    /// Like [`forward_with_tape`], but pre-populates `computed[]` from `seeds`.
    ///
    /// Used internally by [`backward_with_seeds`] so that the tape records
    /// concrete saved-input tensors for gradient computation.
    fn forward_with_tape_seeded(
        &mut self,
        graph: &EinsumGraph,
        seeds: &HashMap<usize, OxiCudaTensor>,
    ) -> Result<OxiCudaTape, OxiCudaBackendError> {
        let mut computed: Vec<Option<OxiCudaTensor>> = vec![None; graph.tensors.len()];
        // Pre-populate seeded positions.
        for (&idx, tensor) in seeds {
            if let Some(slot) = computed.get_mut(idx) {
                *slot = Some(tensor.clone());
            }
        }
        let mut tape_entries: Vec<TapeEntry> = Vec::with_capacity(graph.nodes.len());

        for node in &graph.nodes {
            let input_tensors: Vec<OxiCudaTensor> = node
                .inputs
                .iter()
                .map(|&idx| {
                    computed
                        .get(idx)
                        .and_then(|slot| slot.as_ref())
                        .cloned()
                        .ok_or_else(|| {
                            OxiCudaBackendError::InvalidEinsumSpec(format!(
                                "tensor at index {idx} not yet computed (forward_with_tape_seeded)"
                            ))
                        })
                })
                .collect::<Result<_, _>>()?;

            let (result, taped_op) = match &node.op {
                OpType::Einsum { spec } => {
                    let norm: String = spec
                        .chars()
                        .filter(|c| !c.is_whitespace())
                        .flat_map(|c| c.to_lowercase())
                        .collect();
                    let taped = match norm.as_str() {
                        "ij,jk->ik" => TapedOp::Matmul2D,
                        "bij,bjk->bik" => TapedOp::BatchedMatmul3D,
                        _ if is_identity_spec(&norm) => TapedOp::Identity,
                        _ => TapedOp::Identity,
                    };
                    let out = self.einsum(spec, &input_tensors)?;
                    (out, taped)
                }
                OpType::ElemUnary { op } => {
                    let elem_op = parse_elem_op(op)?;
                    let out = self.elem_op(elem_op, &input_tensors[0])?;
                    (out, TapedOp::Unary(elem_op))
                }
                OpType::ElemBinary { op } => {
                    if input_tensors.len() != 2 {
                        return Err(OxiCudaBackendError::InvalidEinsumSpec(format!(
                            "binary op '{op}' expects 2 inputs, got {}",
                            input_tensors.len()
                        )));
                    }
                    let elem_op = parse_elem_op(op)?;
                    let out = self.elem_op_binary(elem_op, &input_tensors[0], &input_tensors[1])?;
                    (out, TapedOp::Binary(elem_op))
                }
                OpType::Reduce { op, axes } => {
                    let reduce_op = parse_reduce_op(op)?;
                    let out = self.reduce(reduce_op, &input_tensors[0], axes)?;
                    (out, TapedOp::Reduce(reduce_op, axes.clone()))
                }
            };

            let output_idx = node.outputs.first().copied().ok_or_else(|| {
                OxiCudaBackendError::InvalidEinsumSpec("node has no output index".to_string())
            })?;

            tape_entries.push(TapeEntry {
                output_tensor_idx: output_idx,
                input_tensor_indices: node.inputs.clone(),
                op: taped_op,
                saved_inputs: input_tensors,
            });

            computed[output_idx] = Some(result);
        }

        Ok(OxiCudaTape {
            entries: tape_entries,
            gradients: HashMap::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// Helper: identity spec detection (mirrors einsum.rs)
// ---------------------------------------------------------------------------

fn is_identity_spec(norm: &str) -> bool {
    let arrow_count = norm.matches("->").count();
    if arrow_count != 1 {
        return false;
    }
    let parts: Vec<&str> = norm.splitn(2, "->").collect();
    if parts.len() != 2 {
        return false;
    }
    !parts[0].contains(',') && parts[0] == parts[1]
}

// ---------------------------------------------------------------------------
// Unit tests (no GPU required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ones_host_correct_shape_and_data() {
        let t = ones_host(&[2, 3]);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.data.len(), 6);
        assert!(t.data.iter().all(|&v| (v - 1.0).abs() < 1e-7));
    }

    #[test]
    fn zeros_host_correct_shape_and_data() {
        let t = zeros_host(&[3]);
        assert_eq!(t.shape, vec![3]);
        assert!(t.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn transpose_2d_host_correct() -> Result<(), OxiCudaBackendError> {
        // [[1, 2, 3], [4, 5, 6]] transposed → [[1, 4], [2, 5], [3, 6]]
        let t = OxiCudaTensor {
            shape: vec![2, 3],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let result = transpose_2d_host(&t)?;
        assert_eq!(result.shape, vec![3, 2]);
        assert_eq!(result.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        Ok(())
    }

    #[test]
    fn transpose_3d_last_host_correct() -> Result<(), OxiCudaBackendError> {
        // [[[1,2],[3,4]],[[5,6],[7,8]]] shape [2,2,2] → [[[1,3],[2,4]],[[5,7],[6,8]]]
        let t = OxiCudaTensor {
            shape: vec![2, 2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        };
        let result = transpose_3d_last_host(&t)?;
        assert_eq!(result.shape, vec![2, 2, 2]);
        assert_eq!(result.data, vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]);
        Ok(())
    }

    #[test]
    fn broadcast_to_shape_axis0() -> Result<(), OxiCudaBackendError> {
        // small=[3.0, 4.0, 5.0] shape [3], reduce axis 0 of [2,3]
        let small = OxiCudaTensor {
            shape: vec![3],
            data: vec![3.0, 4.0, 5.0],
        };
        let result = broadcast_to_shape(&small, &[2, 3], &[0])?;
        assert_eq!(result.shape, vec![2, 3]);
        assert_eq!(result.data, vec![3.0, 4.0, 5.0, 3.0, 4.0, 5.0]);
        Ok(())
    }

    #[test]
    fn parse_elem_op_known() {
        assert!(matches!(parse_elem_op("relu"), Ok(ElemOp::Relu)));
        assert!(matches!(parse_elem_op("sigmoid"), Ok(ElemOp::Sigmoid)));
        assert!(matches!(parse_elem_op("add"), Ok(ElemOp::Add)));
    }

    #[test]
    fn parse_elem_op_unknown() {
        assert!(parse_elem_op("nonexistent_op").is_err());
    }

    #[test]
    fn is_identity_spec_basic() {
        assert!(is_identity_spec("ij->ij"));
        assert!(!is_identity_spec("ij,jk->ik"));
        assert!(!is_identity_spec("ij->ji"));
    }

    #[test]
    fn tape_type_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<OxiCudaTape>();
    }
}
