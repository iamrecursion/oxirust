//! Core n-dimensional tensor type and operations.
//!
//! All tensors use **row-major (C-contiguous)** storage: the last index varies fastest.
//! The flat index for indices `[i0, i1, …, i_{n-1}]` in a tensor with shape
//! `[s0, s1, …, s_{n-1}]` is:
//!
//! ```text
//! flat = i0*(s1*s2*…*s_{n-1}) + i1*(s2*…*s_{n-1}) + … + i_{n-1}
//! ```

use crate::error::NeuralError;

// ──────────────────────────────────────────────────────────────────────────────
// Tensor
// ──────────────────────────────────────────────────────────────────────────────

/// An n-dimensional, f32, row-major tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Flat data buffer.
    pub(crate) data: Vec<f32>,
    /// Shape of each dimension.
    pub(crate) shape: Vec<usize>,
}

impl Tensor {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Creates a zero-filled tensor with the given shape.
    ///
    /// Returns an error if any dimension is zero.
    pub fn new(shape: Vec<usize>) -> Result<Self, NeuralError> {
        if shape.iter().any(|&d| d == 0) {
            return Err(NeuralError::InvalidShape(
                "all dimensions must be > 0".to_string(),
            ));
        }
        let numel = shape.iter().product();
        Ok(Self {
            data: vec![0.0_f32; numel],
            shape,
        })
    }

    /// Creates a zero-filled tensor — convenience alias for `new`.
    pub fn zeros(shape: Vec<usize>) -> Result<Self, NeuralError> {
        Self::new(shape)
    }

    /// Creates a one-filled tensor with the given shape.
    pub fn ones(shape: Vec<usize>) -> Result<Self, NeuralError> {
        if shape.iter().any(|&d| d == 0) {
            return Err(NeuralError::InvalidShape(
                "all dimensions must be > 0".to_string(),
            ));
        }
        let numel = shape.iter().product();
        Ok(Self {
            data: vec![1.0_f32; numel],
            shape,
        })
    }

    /// Creates a tensor from existing data and a shape.
    ///
    /// Returns an error if the data length does not match the product of the
    /// shape dimensions, or if any dimension is zero.
    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, NeuralError> {
        if shape.iter().any(|&d| d == 0) {
            return Err(NeuralError::InvalidShape(
                "all dimensions must be > 0".to_string(),
            ));
        }
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(NeuralError::InvalidShape(format!(
                "data length {} does not match shape product {}",
                data.len(),
                expected
            )));
        }
        Ok(Self { data, shape })
    }

    // ── Metadata ──────────────────────────────────────────────────────────────

    /// Returns the number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Returns a reference to the shape slice.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns a reference to the flat data buffer.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Returns a mutable reference to the flat data buffer.
    pub fn data_mut(&mut self) -> &mut Vec<f32> {
        &mut self.data
    }

    // ── Indexing ──────────────────────────────────────────────────────────────

    /// Computes the row-major flat index for the given multi-dimensional indices.
    ///
    /// Does **not** perform bounds checking; use `get` / `set` for safe access.
    pub fn flat_index(&self, indices: &[usize]) -> usize {
        let mut idx = 0usize;
        let mut stride = 1usize;
        // Iterate from the last dimension to the first.
        for dim in (0..self.shape.len()).rev() {
            idx += indices[dim] * stride;
            stride *= self.shape[dim];
        }
        idx
    }

    /// Returns the element at the given multi-dimensional indices.
    pub fn get(&self, indices: &[usize]) -> Result<f32, NeuralError> {
        self.validate_indices(indices)?;
        Ok(self.data[self.flat_index(indices)])
    }

    /// Sets the element at the given multi-dimensional indices.
    pub fn set(&mut self, indices: &[usize], value: f32) -> Result<(), NeuralError> {
        self.validate_indices(indices)?;
        let flat = self.flat_index(indices);
        self.data[flat] = value;
        Ok(())
    }

    /// Validates that the given indices are compatible with this tensor's shape.
    fn validate_indices(&self, indices: &[usize]) -> Result<(), NeuralError> {
        if indices.len() != self.shape.len() {
            return Err(NeuralError::IndexOutOfBounds(format!(
                "rank {} tensor requires {} indices, got {}",
                self.shape.len(),
                self.shape.len(),
                indices.len()
            )));
        }
        for (dim, (&idx, &size)) in indices.iter().zip(self.shape.iter()).enumerate() {
            if idx >= size {
                return Err(NeuralError::IndexOutOfBounds(format!(
                    "index {} out of bounds for dimension {} of size {}",
                    idx, dim, size
                )));
            }
        }
        Ok(())
    }

    // ── Shape manipulation ────────────────────────────────────────────────────

    /// Reshapes the tensor, returning a new tensor with the same data.
    ///
    /// The total number of elements must be preserved.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, NeuralError> {
        if new_shape.iter().any(|&d| d == 0) {
            return Err(NeuralError::InvalidShape(
                "all dimensions must be > 0".to_string(),
            ));
        }
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(NeuralError::ShapeMismatch(format!(
                "cannot reshape tensor of {} elements into shape {:?} ({} elements)",
                self.numel(),
                new_shape,
                new_numel,
            )));
        }
        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
        })
    }

    /// Extracts a slice along `dim` from index `start` (inclusive) to `end` (exclusive).
    ///
    /// All other dimensions are preserved.
    pub fn slice(&self, dim: usize, start: usize, end: usize) -> Result<Self, NeuralError> {
        if dim >= self.shape.len() {
            return Err(NeuralError::IndexOutOfBounds(format!(
                "dimension {} out of range for rank-{} tensor",
                dim,
                self.shape.len()
            )));
        }
        if start >= end {
            return Err(NeuralError::InvalidShape(format!(
                "slice start {} must be less than end {}",
                start, end
            )));
        }
        if end > self.shape[dim] {
            return Err(NeuralError::IndexOutOfBounds(format!(
                "slice end {} exceeds dimension {} of size {}",
                end, dim, self.shape[dim]
            )));
        }

        let mut new_shape = self.shape.clone();
        new_shape[dim] = end - start;
        let new_numel: usize = new_shape.iter().product();
        let mut new_data = Vec::with_capacity(new_numel);

        // Strides for the original tensor.
        let total_dims = self.shape.len();
        let mut strides = vec![1usize; total_dims];
        for i in (0..total_dims - 1).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }

        // Iterate over every output element position.
        let mut out_indices = vec![0usize; total_dims];
        for _ in 0..new_numel {
            // Map output indices to input indices (offset slice dimension).
            let mut in_indices = out_indices.clone();
            in_indices[dim] += start;

            let flat: usize = in_indices
                .iter()
                .zip(strides.iter())
                .map(|(&i, &s)| i * s)
                .sum();
            new_data.push(self.data[flat]);

            // Increment out_indices in row-major order.
            for d in (0..total_dims).rev() {
                out_indices[d] += 1;
                if out_indices[d] < new_shape[d] {
                    break;
                }
                out_indices[d] = 0;
            }
        }

        Ok(Self {
            data: new_data,
            shape: new_shape,
        })
    }

    /// Unsqueezes (adds) a dimension of size 1 at the given position.
    ///
    /// `dim` can range from `0` to `self.ndim()` inclusive.
    pub fn unsqueeze(&self, dim: usize) -> Result<Self, NeuralError> {
        if dim > self.shape.len() {
            return Err(NeuralError::IndexOutOfBounds(format!(
                "unsqueeze: dim {} out of range for rank-{} tensor (max {})",
                dim,
                self.shape.len(),
                self.shape.len()
            )));
        }
        let mut new_shape = self.shape.clone();
        new_shape.insert(dim, 1);
        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
        })
    }

    /// Squeezes (removes) all dimensions of size 1, or a specific one.
    pub fn squeeze(&self, dim: Option<usize>) -> Result<Self, NeuralError> {
        let new_shape: Vec<usize> = match dim {
            Some(d) => {
                if d >= self.shape.len() {
                    return Err(NeuralError::IndexOutOfBounds(format!(
                        "squeeze: dim {} out of range for rank-{} tensor",
                        d,
                        self.shape.len()
                    )));
                }
                if self.shape[d] != 1 {
                    self.shape.clone()
                } else {
                    self.shape
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| i != d)
                        .map(|(_, &s)| s)
                        .collect()
                }
            }
            None => self.shape.iter().copied().filter(|&s| s != 1).collect(),
        };
        // If all dims were 1 and we squeezed everything, keep a scalar shape [1].
        let final_shape = if new_shape.is_empty() {
            vec![1]
        } else {
            new_shape
        };
        Ok(Self {
            data: self.data.clone(),
            shape: final_shape,
        })
    }

    /// Transposes a 2-D tensor (swaps rows and columns).
    pub fn transpose_2d(&self) -> Result<Self, NeuralError> {
        if self.shape.len() != 2 {
            return Err(NeuralError::InvalidShape(format!(
                "transpose_2d requires a 2-D tensor, got rank {}",
                self.shape.len()
            )));
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut new_data = vec![0.0_f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                new_data[c * rows + r] = self.data[r * cols + c];
            }
        }
        Ok(Self {
            data: new_data,
            shape: vec![cols, rows],
        })
    }

    // ── Batch dimension support ───────────────────────────────────────────────

    /// Returns the batch size if this tensor has 4 or more dimensions.
    ///
    /// Returns `None` for tensors with fewer than 4 dimensions.
    /// For a 4-D tensor `[N, C, H, W]`, returns `Some(N)`.
    pub fn batch_size(&self) -> Option<usize> {
        if self.shape.len() >= 4 {
            Some(self.shape[0])
        } else {
            None
        }
    }

    /// Stacks a `Vec<Tensor>` along a new leading batch dimension.
    ///
    /// All tensors in `items` must have identical shapes.
    /// Returns a tensor with shape `[N, d0, d1, ...]` where `N = items.len()`.
    ///
    /// # Errors
    ///
    /// - `EmptyInput` if `items` is empty.
    /// - `ShapeMismatch` if any tensor has a shape different from `items[0]`.
    pub fn from_batch(items: Vec<Tensor>) -> Result<Self, NeuralError> {
        if items.is_empty() {
            return Err(NeuralError::EmptyInput(
                "Tensor::from_batch: items must not be empty".to_string(),
            ));
        }
        let item_shape = items[0].shape.clone();
        let item_numel = items[0].numel();

        for (i, t) in items.iter().enumerate().skip(1) {
            if t.shape != item_shape {
                return Err(NeuralError::ShapeMismatch(format!(
                    "Tensor::from_batch: tensor[{}] shape {:?} != tensor[0] shape {:?}",
                    i, t.shape, item_shape
                )));
            }
        }

        let batch_size = items.len();
        let mut data = Vec::with_capacity(batch_size * item_numel);
        for t in &items {
            data.extend_from_slice(t.data());
        }

        let mut new_shape = Vec::with_capacity(1 + item_shape.len());
        new_shape.push(batch_size);
        new_shape.extend_from_slice(&item_shape);

        Ok(Self {
            data,
            shape: new_shape,
        })
    }

    /// Splits a tensor along the leading batch dimension (dim 0).
    ///
    /// Requires `ndim >= 2`. Returns a `Vec` of tensors each with shape
    /// equal to `self.shape[1..]`.
    ///
    /// # Errors
    ///
    /// - `InvalidShape` if the tensor has fewer than 2 dimensions.
    pub fn unbatch(self) -> Result<Vec<Tensor>, NeuralError> {
        if self.shape.len() < 2 {
            return Err(NeuralError::InvalidShape(format!(
                "Tensor::unbatch: requires at least 2 dimensions, got rank {}",
                self.shape.len()
            )));
        }
        let batch_size = self.shape[0];
        let item_shape: Vec<usize> = self.shape[1..].to_vec();
        let item_numel: usize = item_shape.iter().product();

        let mut result = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let start = b * item_numel;
            let end = start + item_numel;
            let item_data = self.data[start..end].to_vec();
            result.push(Tensor::from_data(item_data, item_shape.clone())?);
        }
        Ok(result)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Free functions operating on tensors
// ──────────────────────────────────────────────────────────────────────────────

/// Matrix multiplication for 2-D tensors.
///
/// `a` must have shape `[n, k]` and `b` must have shape `[k, m]`.
/// Returns a tensor of shape `[n, m]`.
///
/// On x86_64 platforms with AVX2 or SSE4.1, the inner dot-product loop is
/// accelerated with SIMD intrinsics.  A scalar fallback is used on all other
/// platforms.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuralError> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(NeuralError::InvalidShape(
            "matmul requires 2-D tensors".to_string(),
        ));
    }
    let (n, k) = (a.shape[0], a.shape[1]);
    let (bk, m) = (b.shape[0], b.shape[1]);
    if k != bk {
        return Err(NeuralError::ShapeMismatch(format!(
            "matmul: inner dimensions do not match: {} vs {}",
            k, bk
        )));
    }
    let mut out_data = vec![0.0_f32; n * m];
    matmul_dispatch(&a.data, &b.data, &mut out_data, n, k, m);
    Tensor::from_data(out_data, vec![n, m])
}

// ── SIMD-accelerated matmul via scirs2-core ──────────────────────────────────

/// Dispatches matmul to scirs2-core's blocked GEMM implementation, which
/// provides cache-optimised, platform-specific SIMD kernels (AVX-512/AVX2/SSE
/// on x86_64, NEON on aarch64) with automatic fallback to scalar code.
#[allow(unsafe_code)]
fn matmul_dispatch(a: &[f32], b: &[f32], out: &mut [f32], n: usize, k: usize, m: usize) {
    use scirs2_core::simd::gemm::{blocked_gemm_f32, MatMulConfig};

    let config = MatMulConfig::for_f32();
    // SAFETY: pointers are valid for the given dimensions and leading
    // dimensions equal the column count (row-major contiguous layout).
    unsafe {
        blocked_gemm_f32(
            n,
            k,
            m,
            1.0, // alpha
            a.as_ptr(),
            k, // lda
            b.as_ptr(),
            m,   // ldb
            0.0, // beta
            out.as_mut_ptr(),
            m, // ldc
            &config,
        );
    }
}

/// Element-wise addition of two tensors.
///
/// Shapes must be identical (no broadcasting of different rank) — except that a
/// 1-D tensor of size equal to the last dimension of `a` is broadcast across all
/// leading dimensions (useful for bias addition).
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuralError> {
    // Exact same shape.
    if a.shape == b.shape {
        let data: Vec<f32> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(x, y)| x + y)
            .collect();
        return Tensor::from_data(data, a.shape.clone());
    }
    // Broadcast: b is 1-D and its size equals the last dimension of a.
    if b.shape.len() == 1 && !a.shape.is_empty() && b.shape[0] == *a.shape.last().unwrap_or(&0) {
        let last = b.shape[0];
        let data: Vec<f32> = a
            .data
            .iter()
            .enumerate()
            .map(|(i, &x)| x + b.data[i % last])
            .collect();
        return Tensor::from_data(data, a.shape.clone());
    }
    Err(NeuralError::ShapeMismatch(format!(
        "add: incompatible shapes {:?} and {:?}",
        a.shape, b.shape
    )))
}

/// Element-wise multiplication of two tensors.
///
/// Shapes must be identical.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuralError> {
    if a.shape != b.shape {
        return Err(NeuralError::ShapeMismatch(format!(
            "mul: shapes {:?} and {:?} differ",
            a.shape, b.shape
        )));
    }
    let data: Vec<f32> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x * y)
        .collect();
    Tensor::from_data(data, a.shape.clone())
}

/// Reduces a tensor by summing along the given dimension.
///
/// The output tensor has the same rank but the reduced dimension has size 1.
pub fn sum_along(t: &Tensor, dim: usize) -> Result<Tensor, NeuralError> {
    if dim >= t.shape.len() {
        return Err(NeuralError::IndexOutOfBounds(format!(
            "sum_along: dimension {} out of range for rank-{} tensor",
            dim,
            t.shape.len()
        )));
    }
    let mut out_shape = t.shape.clone();
    out_shape[dim] = 1;
    let out_numel: usize = out_shape.iter().product();
    let mut out_data = vec![0.0_f32; out_numel];

    // Build strides for the input tensor.
    let total_dims = t.shape.len();
    let mut strides = vec![1usize; total_dims];
    for i in (0..total_dims - 1).rev() {
        strides[i] = strides[i + 1] * t.shape[i + 1];
    }

    // Build strides for the output tensor.
    let mut out_strides = vec![1usize; total_dims];
    for i in (0..total_dims - 1).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    let mut in_indices = vec![0usize; total_dims];
    for in_flat in 0..t.numel() {
        // Compute output flat index (collapse `dim` to 0).
        let mut out_flat = 0usize;
        for d in 0..total_dims {
            let out_idx = if d == dim { 0 } else { in_indices[d] };
            out_flat += out_idx * out_strides[d];
        }
        out_data[out_flat] += t.data[in_flat];

        // Advance in_indices.
        for d in (0..total_dims).rev() {
            in_indices[d] += 1;
            if in_indices[d] < t.shape[d] {
                break;
            }
            in_indices[d] = 0;
        }
    }

    Tensor::from_data(out_data, out_shape)
}

// ──────────────────────────────────────────────────────────────────────────────
// In-place operations
// ──────────────────────────────────────────────────────────────────────────────

/// Applies ReLU in-place: `x = max(0, x)` for every element.
pub fn relu_inplace(t: &mut Tensor) {
    for v in t.data.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Element-wise in-place addition: `a[i] += b[i]`.
///
/// Supports the same broadcasting rules as [`add`]:
/// - Identical shapes.
/// - `b` is 1-D and matches the last dimension of `a`.
/// - Full NumPy-style broadcasting (see [`broadcast_add`]).
pub fn add_inplace(a: &mut Tensor, b: &Tensor) -> Result<(), NeuralError> {
    if a.shape == b.shape {
        for (av, bv) in a.data.iter_mut().zip(b.data.iter()) {
            *av += *bv;
        }
        return Ok(());
    }
    // 1-D bias broadcast.
    if b.shape.len() == 1 && !a.shape.is_empty() {
        let last = a.shape.last().copied().unwrap_or(0);
        if b.shape[0] == last {
            for (i, av) in a.data.iter_mut().enumerate() {
                *av += b.data[i % last];
            }
            return Ok(());
        }
    }
    // General broadcast.
    if let Some(strides_b) = compute_broadcast_strides(&a.shape, &b.shape) {
        let ndim = a.shape.len();
        let numel = a.data.len();
        let strides_a = compute_strides(&a.shape);
        let mut indices = vec![0usize; ndim];
        for flat_a in 0..numel {
            let mut flat_b = 0usize;
            for d in 0..ndim {
                flat_b += indices[d] * strides_b[d];
            }
            a.data[flat_a] += b.data[flat_b];
            // Advance indices.
            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < a.shape[d] {
                    break;
                }
                indices[d] = 0;
            }
            let _ = &strides_a; // suppress unused warning
        }
        return Ok(());
    }
    Err(NeuralError::ShapeMismatch(format!(
        "add_inplace: incompatible shapes {:?} and {:?}",
        a.shape, b.shape
    )))
}

/// Element-wise in-place multiplication: `a[i] *= b[i]`.
///
/// Shapes must be identical.
pub fn mul_inplace(a: &mut Tensor, b: &Tensor) -> Result<(), NeuralError> {
    if a.shape != b.shape {
        return Err(NeuralError::ShapeMismatch(format!(
            "mul_inplace: shapes {:?} and {:?} differ",
            a.shape, b.shape
        )));
    }
    for (av, bv) in a.data.iter_mut().zip(b.data.iter()) {
        *av *= *bv;
    }
    Ok(())
}

/// Scales every element in-place: `t[i] *= scalar`.
pub fn scale_inplace(t: &mut Tensor, scalar: f32) {
    for v in t.data.iter_mut() {
        *v *= scalar;
    }
}

/// Clips every element in-place to `[lo, hi]`.
pub fn clamp_inplace(t: &mut Tensor, lo: f32, hi: f32) {
    for v in t.data.iter_mut() {
        if *v < lo {
            *v = lo;
        } else if *v > hi {
            *v = hi;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Broadcasting helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Computes the contiguous strides for a given shape (row-major).
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Returns broadcast strides for `b_shape` into `a_shape`, or `None` if incompatible.
///
/// NumPy-style broadcasting: `b_shape` is right-aligned with `a_shape`.
/// For each dimension, `b_shape[d]` must equal `a_shape[d]` or be 1.
/// If `b_shape[d] == 1`, the stride for that dimension is set to 0 (repeat).
fn compute_broadcast_strides(a_shape: &[usize], b_shape: &[usize]) -> Option<Vec<usize>> {
    let a_ndim = a_shape.len();
    let b_ndim = b_shape.len();
    if b_ndim > a_ndim {
        return None;
    }
    let offset = a_ndim - b_ndim;
    let b_strides = compute_strides(b_shape);

    let mut out_strides = vec![0usize; a_ndim];
    for d in 0..a_ndim {
        if d < offset {
            // b is implicitly 1 in this dimension → stride 0.
            out_strides[d] = 0;
        } else {
            let bd = d - offset;
            if b_shape[bd] == a_shape[d] {
                out_strides[d] = b_strides[bd];
            } else if b_shape[bd] == 1 {
                out_strides[d] = 0;
            } else {
                return None;
            }
        }
    }
    Some(out_strides)
}

/// Element-wise addition with full NumPy-style broadcasting.
///
/// The output shape is the broadcast-compatible shape of `a` and `b`.
/// For simplicity the output always takes `a`'s shape when `a.ndim() >= b.ndim()`.
pub fn broadcast_add(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuralError> {
    // If shapes match, fast path.
    if a.shape == b.shape {
        return add(a, b);
    }
    // Try a as the larger tensor.
    if a.shape.len() >= b.shape.len() {
        if let Some(b_strides) = compute_broadcast_strides(&a.shape, &b.shape) {
            let ndim = a.shape.len();
            let numel = a.data.len();
            let mut out_data = Vec::with_capacity(numel);
            let mut indices = vec![0usize; ndim];
            for flat_a in 0..numel {
                let mut flat_b = 0usize;
                for d in 0..ndim {
                    flat_b += indices[d] * b_strides[d];
                }
                out_data.push(a.data[flat_a] + b.data[flat_b]);
                for d in (0..ndim).rev() {
                    indices[d] += 1;
                    if indices[d] < a.shape[d] {
                        break;
                    }
                    indices[d] = 0;
                }
            }
            return Tensor::from_data(out_data, a.shape.clone());
        }
    }
    // Try b as the larger tensor.
    if b.shape.len() > a.shape.len() {
        if let Some(a_strides) = compute_broadcast_strides(&b.shape, &a.shape) {
            let ndim = b.shape.len();
            let numel = b.data.len();
            let mut out_data = Vec::with_capacity(numel);
            let mut indices = vec![0usize; ndim];
            for flat_b in 0..numel {
                let mut flat_a = 0usize;
                for d in 0..ndim {
                    flat_a += indices[d] * a_strides[d];
                }
                out_data.push(a.data[flat_a] + b.data[flat_b]);
                for d in (0..ndim).rev() {
                    indices[d] += 1;
                    if indices[d] < b.shape[d] {
                        break;
                    }
                    indices[d] = 0;
                }
            }
            return Tensor::from_data(out_data, b.shape.clone());
        }
    }
    Err(NeuralError::ShapeMismatch(format!(
        "broadcast_add: incompatible shapes {:?} and {:?}",
        a.shape, b.shape
    )))
}

/// Element-wise multiplication with full NumPy-style broadcasting.
pub fn broadcast_mul(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuralError> {
    if a.shape == b.shape {
        return mul(a, b);
    }
    if a.shape.len() >= b.shape.len() {
        if let Some(b_strides) = compute_broadcast_strides(&a.shape, &b.shape) {
            let ndim = a.shape.len();
            let numel = a.data.len();
            let mut out_data = Vec::with_capacity(numel);
            let mut indices = vec![0usize; ndim];
            for flat_a in 0..numel {
                let mut flat_b = 0usize;
                for d in 0..ndim {
                    flat_b += indices[d] * b_strides[d];
                }
                out_data.push(a.data[flat_a] * b.data[flat_b]);
                for d in (0..ndim).rev() {
                    indices[d] += 1;
                    if indices[d] < a.shape[d] {
                        break;
                    }
                    indices[d] = 0;
                }
            }
            return Tensor::from_data(out_data, a.shape.clone());
        }
    }
    if b.shape.len() > a.shape.len() {
        if let Some(a_strides) = compute_broadcast_strides(&b.shape, &a.shape) {
            let ndim = b.shape.len();
            let numel = b.data.len();
            let mut out_data = Vec::with_capacity(numel);
            let mut indices = vec![0usize; ndim];
            for flat_b in 0..numel {
                let mut flat_a = 0usize;
                for d in 0..ndim {
                    flat_a += indices[d] * a_strides[d];
                }
                out_data.push(a.data[flat_a] * b.data[flat_b]);
                for d in (0..ndim).rev() {
                    indices[d] += 1;
                    if indices[d] < b.shape[d] {
                        break;
                    }
                    indices[d] = 0;
                }
            }
            return Tensor::from_data(out_data, b.shape.clone());
        }
    }
    Err(NeuralError::ShapeMismatch(format!(
        "broadcast_mul: incompatible shapes {:?} and {:?}",
        a.shape, b.shape
    )))
}

/// Batched matrix multiplication for 3-D tensors `[B, N, K] @ [B, K, M] → [B, N, M]`.
///
/// Also supports broadcasting: `[B, N, K] @ [K, M]` broadcasts the 2-D matrix
/// across the batch dimension.
pub fn batched_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuralError> {
    // Case 1: both 3-D
    if a.ndim() == 3 && b.ndim() == 3 {
        let (ba, na, ka) = (a.shape[0], a.shape[1], a.shape[2]);
        let (bb, kb, mb) = (b.shape[0], b.shape[1], b.shape[2]);
        if ba != bb {
            return Err(NeuralError::ShapeMismatch(format!(
                "batched_matmul: batch sizes differ: {} vs {}",
                ba, bb
            )));
        }
        if ka != kb {
            return Err(NeuralError::ShapeMismatch(format!(
                "batched_matmul: inner dimensions do not match: {} vs {}",
                ka, kb
            )));
        }
        let batch = ba;
        let (n, k, m) = (na, ka, mb);
        let mut out_data = vec![0.0_f32; batch * n * m];
        for bi in 0..batch {
            let a_off = bi * n * k;
            let b_off = bi * k * m;
            let o_off = bi * n * m;
            matmul_dispatch(
                &a.data[a_off..a_off + n * k],
                &b.data[b_off..b_off + k * m],
                &mut out_data[o_off..o_off + n * m],
                n,
                k,
                m,
            );
        }
        return Tensor::from_data(out_data, vec![batch, n, m]);
    }
    // Case 2: a is 3-D, b is 2-D → broadcast b
    if a.ndim() == 3 && b.ndim() == 2 {
        let (batch, n, ka) = (a.shape[0], a.shape[1], a.shape[2]);
        let (kb, m) = (b.shape[0], b.shape[1]);
        if ka != kb {
            return Err(NeuralError::ShapeMismatch(format!(
                "batched_matmul: inner dimensions do not match: {} vs {}",
                ka, kb
            )));
        }
        let k = ka;
        let mut out_data = vec![0.0_f32; batch * n * m];
        for bi in 0..batch {
            let a_off = bi * n * k;
            let o_off = bi * n * m;
            matmul_dispatch(
                &a.data[a_off..a_off + n * k],
                &b.data,
                &mut out_data[o_off..o_off + n * m],
                n,
                k,
                m,
            );
        }
        return Tensor::from_data(out_data, vec![batch, n, m]);
    }
    // Fallback to regular 2-D matmul.
    if a.ndim() == 2 && b.ndim() == 2 {
        return matmul(a, b);
    }
    Err(NeuralError::InvalidShape(format!(
        "batched_matmul: unsupported ranks {} and {}",
        a.ndim(),
        b.ndim()
    )))
}

// ──────────────────────────────────────────────────────────────────────────────
// matmul_simd — explicit SIMD matrix-matrix multiply (AVX2 / scalar fallback)
// ──────────────────────────────────────────────────────────────────────────────

/// SIMD-accelerated matrix-matrix multiplication.
///
/// Computes `C = A × B` where
/// - `A` has shape `[m × k]` (row-major, length `m * k`)
/// - `B` has shape `[k × n]` (row-major, length `k * n`)
/// - `C` has shape `[m × n]` (row-major, length `m * n`)
///
/// On x86_64 systems with the AVX2 feature enabled at compile-time the hot
/// inner loop uses 256-bit FMA intrinsics loading 8 floats at a time.  On all
/// other targets (or when the `avx2` feature is absent) the function falls
/// back to an optimised scalar loop that the compiler can still auto-vectorise.
///
/// The function is useful when you already hold raw `Vec<f32>` slices (e.g.
/// the output of [`crate::im2col::im2col`]) and want to avoid the overhead of
/// wrapping them in [`Tensor`] objects for a single GEMM call.
#[allow(unsafe_code)]
pub fn matmul_simd(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);

    let mut c = vec![0.0_f32; m * n];
    matmul_simd_into(a, b, &mut c, m, k, n);
    c
}

/// Writes `C = A × B` into the pre-allocated output slice `c`.
///
/// `c` must have length `m * n` and will be **overwritten** (not accumulated).
#[allow(unsafe_code)]
pub fn matmul_simd_into(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    // Zero the output first so both code paths start from the same state.
    for v in c.iter_mut() {
        *v = 0.0;
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: we check for AVX2 at compile-time via `target_feature`.
        // All pointer arithmetic stays within the pre-validated slice bounds.
        unsafe { matmul_avx2(a, b, c, m, k, n) }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        matmul_scalar(a, b, c, m, k, n);
    }
}

// ── AVX2 path ─────────────────────────────────────────────────────────────────

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(unsafe_code)]
unsafe fn matmul_avx2(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    use std::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_broadcast_ss, _mm256_loadu_ps, _mm256_mul_ps,
        _mm256_setzero_ps, _mm256_storeu_ps,
    };

    // Tiled outer-product accumulation:
    // For each row `i` of A and each column-block of B (8-wide), accumulate.
    let n8 = n / 8 * 8; // largest multiple of 8 ≤ n

    for i in 0..m {
        let a_row = i * k;
        let c_row = i * n;

        // Process 8-wide column blocks of C.
        for j8 in (0..n8).step_by(8) {
            // Accumulator register (8 floats).
            let mut acc: __m256 = _mm256_setzero_ps();

            for p in 0..k {
                // Broadcast A[i,p] to all 8 lanes.
                let a_val = _mm256_broadcast_ss(&a[a_row + p]);
                // Load 8 consecutive elements from B row `p`, columns j8..j8+8.
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(p * n + j8));
                // acc += A[i,p] * B[p, j8..j8+8]
                acc = _mm256_add_ps(acc, _mm256_mul_ps(a_val, b_vec));
            }

            // Store 8 results into C[i, j8..j8+8].
            _mm256_storeu_ps(c.as_mut_ptr().add(c_row + j8), acc);
        }

        // Scalar tail for remaining columns.
        for j in n8..n {
            let mut sum = 0.0_f32;
            for p in 0..k {
                sum += a[a_row + p] * b[p * n + j];
            }
            c[c_row + j] = sum;
        }
    }
}

// ── Scalar fallback ───────────────────────────────────────────────────────────

/// Scalar GEMM — used when AVX2 is not available.
///
/// Uses a cache-friendly i-p-j loop order so the compiler's auto-vectoriser
/// has the best chance of emitting SIMD code.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
fn matmul_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    matmul_scalar_impl(a, b, c, m, k, n);
}

/// Unconditional scalar implementation (always compiled, used by tests too).
fn matmul_scalar_impl(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // i-p-j loop order: A[i,p] is broadcast across the entire row of B[p, *],
    // which maximises write reuse in the inner loop and is LLVM-friendly.
    for i in 0..m {
        let a_row = i * k;
        let c_row = i * n;
        for p in 0..k {
            let a_val = a[a_row + p];
            let b_row = p * n;
            for j in 0..n {
                c[c_row + j] += a_val * b[b_row + j];
            }
        }
    }
}

// ── Runtime-dispatch variant (always available) ───────────────────────────────

/// SIMD matrix multiply with **runtime** AVX2 detection.
///
/// Unlike [`matmul_simd`] — which selects the code path at compile-time via
/// `target_feature` — this variant queries CPUID at runtime using
/// [`std::arch::is_x86_feature_detected!`] so it works even when the binary
/// is not compiled with `-C target-feature=+avx2`.  On non-x86 targets it
/// always uses the scalar path.
///
/// # Performance
/// The runtime check adds a single atomic load (~1 ns) per call; for large
/// matrices this is negligible.  Use [`matmul_simd`] instead when you can
/// guarantee the target CPU at compile time.
#[allow(unsafe_code)]
pub fn matmul_simd_runtime(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);

    let mut c = vec![0.0_f32; m * n];

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: we just confirmed AVX2 availability at runtime.
            unsafe { matmul_avx2_dynamic(a, b, &mut c, m, k, n) };
            return c;
        }
    }

    matmul_scalar_impl(a, b, &mut c, m, k, n);
    c
}

/// AVX2 GEMM callable regardless of compile-time `target_feature` setting.
///
/// Marked with `#[target_feature(enable = "avx2")]` so the compiler emits
/// AVX2 instructions even without a global `-C target-feature=+avx2` flag.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn matmul_avx2_dynamic(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    use std::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_broadcast_ss, _mm256_loadu_ps, _mm256_mul_ps,
        _mm256_setzero_ps, _mm256_storeu_ps,
    };

    let n8 = n / 8 * 8;

    for i in 0..m {
        let a_row = i * k;
        let c_row = i * n;

        for j8 in (0..n8).step_by(8) {
            let mut acc: __m256 = _mm256_setzero_ps();
            for p in 0..k {
                let a_val = _mm256_broadcast_ss(&a[a_row + p]);
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(p * n + j8));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(a_val, b_vec));
            }
            _mm256_storeu_ps(c.as_mut_ptr().add(c_row + j8), acc);
        }

        for j in n8..n {
            let mut sum = 0.0_f32;
            for p in 0..k {
                sum += a[a_row + p] * b[p * n + j];
            }
            c[c_row + j] = sum;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn test_zeros_shape() {
        let t = Tensor::zeros(vec![2, 3]).expect("tensor zeros");
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        assert!(t.data().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let t = Tensor::ones(vec![3, 2]).expect("tensor ones");
        assert!(t.data().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_from_data_ok() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor from_data");
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_from_data_length_mismatch() {
        let result = Tensor::from_data(vec![1.0, 2.0], vec![2, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_dim_error() {
        assert!(Tensor::new(vec![0, 3]).is_err());
        assert!(Tensor::zeros(vec![2, 0]).is_err());
        assert!(Tensor::ones(vec![0]).is_err());
    }

    // ── Metadata ──────────────────────────────────────────────────────────────

    #[test]
    fn test_ndim() {
        let t = Tensor::zeros(vec![2, 3, 4]).expect("tensor zeros");
        assert_eq!(t.ndim(), 3);
    }

    #[test]
    fn test_numel() {
        let t = Tensor::zeros(vec![4, 5]).expect("tensor zeros");
        assert_eq!(t.numel(), 20);
    }

    // ── Indexing ──────────────────────────────────────────────────────────────

    #[test]
    fn test_get_set_2d() {
        let mut t = Tensor::zeros(vec![3, 3]).expect("tensor zeros");
        t.set(&[1, 2], 7.0).expect("set");
        assert!(close(t.get(&[1, 2]).expect("get"), 7.0));
    }

    #[test]
    fn test_get_out_of_bounds() {
        let t = Tensor::zeros(vec![2, 2]).expect("tensor zeros");
        assert!(t.get(&[2, 0]).is_err());
    }

    #[test]
    fn test_flat_index_row_major() {
        // shape [2,3]: element [1,2] → 1*3 + 2 = 5
        let t = Tensor::zeros(vec![2, 3]).expect("tensor zeros");
        assert_eq!(t.flat_index(&[1, 2]), 5);
    }

    #[test]
    fn test_flat_index_3d() {
        // shape [2,3,4]: element [1,2,3] → 1*12 + 2*4 + 3 = 23
        let t = Tensor::zeros(vec![2, 3, 4]).expect("tensor zeros");
        assert_eq!(t.flat_index(&[1, 2, 3]), 23);
    }

    #[test]
    fn test_set_get_3d() {
        let mut t = Tensor::zeros(vec![2, 3, 4]).expect("tensor zeros");
        t.set(&[1, 2, 3], 99.0).expect("set");
        assert!(close(t.get(&[1, 2, 3]).expect("get"), 99.0));
    }

    // ── Reshape ───────────────────────────────────────────────────────────────

    #[test]
    fn test_reshape_ok() {
        let t = Tensor::ones(vec![2, 6]).expect("tensor ones");
        let r = t.reshape(vec![3, 4]).expect("reshape");
        assert_eq!(r.shape(), &[3, 4]);
        assert_eq!(r.numel(), 12);
    }

    #[test]
    fn test_reshape_numel_mismatch() {
        let t = Tensor::ones(vec![2, 3]).expect("tensor ones");
        assert!(t.reshape(vec![2, 4]).is_err());
    }

    // ── Slice ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_slice_2d_rows() {
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let t = Tensor::from_data(data, vec![3, 2]).expect("tensor from_data");
        // Rows 1..3 → [[2,3],[4,5]]
        let s = t.slice(0, 1, 3).expect("slice");
        assert_eq!(s.shape(), &[2, 2]);
        assert!(close(s.get(&[0, 0]).expect("get"), 2.0));
        assert!(close(s.get(&[1, 1]).expect("get"), 5.0));
    }

    #[test]
    fn test_slice_bad_range() {
        let t = Tensor::ones(vec![4, 4]).expect("tensor ones");
        assert!(t.slice(0, 3, 2).is_err());
        assert!(t.slice(0, 0, 5).is_err());
    }

    // ── Transpose ─────────────────────────────────────────────────────────────

    #[test]
    fn test_transpose_2d() {
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let t = Tensor::from_data(data, vec![2, 3]).expect("tensor from_data");
        let tr = t.transpose_2d().expect("transpose");
        assert_eq!(tr.shape(), &[3, 2]);
        assert!(close(tr.get(&[0, 0]).expect("get"), 0.0));
        assert!(close(tr.get(&[1, 0]).expect("get"), 1.0));
        assert!(close(tr.get(&[2, 0]).expect("get"), 2.0));
        assert!(close(tr.get(&[0, 1]).expect("get"), 3.0));
    }

    #[test]
    fn test_transpose_non_2d_error() {
        let t = Tensor::zeros(vec![2, 3, 4]).expect("tensor zeros");
        assert!(t.transpose_2d().is_err());
    }

    // ── Matmul ────────────────────────────────────────────────────────────────

    #[test]
    fn test_matmul_basic() {
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor from_data");
        let b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).expect("tensor from_data");
        let c = matmul(&a, &b).expect("matmul");
        assert!(close(c.get(&[0, 0]).expect("get"), 19.0));
        assert!(close(c.get(&[0, 1]).expect("get"), 22.0));
        assert!(close(c.get(&[1, 0]).expect("get"), 43.0));
        assert!(close(c.get(&[1, 1]).expect("get"), 50.0));
    }

    #[test]
    fn test_matmul_rect() {
        // [1,3] @ [3,2] → [1,2]
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![1, 3]).expect("tensor from_data");
        let b = Tensor::from_data(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0], vec![3, 2])
            .expect("tensor from_data");
        let c = matmul(&a, &b).expect("matmul");
        assert_eq!(c.shape(), &[1, 2]);
        // [1*1+2*0+3*1, 1*0+2*1+3*0] = [4, 2]
        assert!(close(c.get(&[0, 0]).expect("get"), 4.0));
        assert!(close(c.get(&[0, 1]).expect("get"), 2.0));
    }

    #[test]
    fn test_matmul_dim_mismatch() {
        let a = Tensor::zeros(vec![2, 3]).expect("tensor zeros");
        let b = Tensor::zeros(vec![4, 2]).expect("tensor zeros");
        assert!(matmul(&a, &b).is_err());
    }

    // ── Add ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_add_same_shape() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3]).expect("tensor from_data");
        let b = Tensor::from_data(vec![4.0, 5.0, 6.0], vec![3]).expect("tensor from_data");
        let c = add(&a, &b).expect("add");
        assert!(close(c.data()[0], 5.0));
        assert!(close(c.data()[1], 7.0));
        assert!(close(c.data()[2], 9.0));
    }

    #[test]
    fn test_add_broadcast_bias() {
        // [2,3] + [3] → broadcast last dim
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
            .expect("tensor from_data");
        let b = Tensor::from_data(vec![10.0, 20.0, 30.0], vec![3]).expect("tensor from_data");
        let c = add(&a, &b).expect("add");
        assert!(close(c.get(&[0, 0]).expect("get"), 11.0));
        assert!(close(c.get(&[1, 2]).expect("get"), 36.0));
    }

    #[test]
    fn test_add_incompatible() {
        let a = Tensor::zeros(vec![2, 3]).expect("tensor zeros");
        let b = Tensor::zeros(vec![2, 4]).expect("tensor zeros");
        assert!(add(&a, &b).is_err());
    }

    // ── Mul ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_mul_elementwise() {
        let a = Tensor::from_data(vec![2.0, 3.0, 4.0], vec![3]).expect("tensor from_data");
        let b = Tensor::from_data(vec![5.0, 6.0, 7.0], vec![3]).expect("tensor from_data");
        let c = mul(&a, &b).expect("mul");
        assert!(close(c.data()[0], 10.0));
        assert!(close(c.data()[1], 18.0));
        assert!(close(c.data()[2], 28.0));
    }

    #[test]
    fn test_mul_shape_mismatch() {
        let a = Tensor::zeros(vec![3]).expect("tensor zeros");
        let b = Tensor::zeros(vec![4]).expect("tensor zeros");
        assert!(mul(&a, &b).is_err());
    }

    // ── Sum along ─────────────────────────────────────────────────────────────

    #[test]
    fn test_sum_along_dim0() {
        // [[1,2],[3,4]] summed along dim 0 → [[4,6]]
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor from_data");
        let s = sum_along(&t, 0).expect("sum_along");
        assert_eq!(s.shape(), &[1, 2]);
        assert!(close(s.data()[0], 4.0));
        assert!(close(s.data()[1], 6.0));
    }

    #[test]
    fn test_sum_along_dim1() {
        // [[1,2],[3,4]] summed along dim 1 → [[3],[7]]
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor from_data");
        let s = sum_along(&t, 1).expect("sum_along");
        assert_eq!(s.shape(), &[2, 1]);
        assert!(close(s.data()[0], 3.0));
        assert!(close(s.data()[1], 7.0));
    }

    #[test]
    fn test_sum_along_out_of_range() {
        let t = Tensor::zeros(vec![2, 3]).expect("tensor zeros");
        assert!(sum_along(&t, 2).is_err());
    }

    // ── SIMD vs scalar matmul verification ──────────────────────────────────

    #[test]
    fn test_matmul_simd_matches_scalar_small() {
        // 3x4 @ 4x5
        let a_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.5 + 1.0).collect();
        let b_data: Vec<f32> = (0..20).map(|i| (i as f32) * 0.3 - 2.0).collect();
        let a = Tensor::from_data(a_data.clone(), vec![3, 4]).expect("tensor from_data");
        let b = Tensor::from_data(b_data.clone(), vec![4, 5]).expect("tensor from_data");

        // SIMD path (dispatched)
        let simd_result = matmul(&a, &b).expect("matmul");

        // Manual scalar computation for verification
        let mut expected = vec![0.0_f32; 15];
        for row in 0..3 {
            for col in 0..5 {
                let mut acc = 0.0_f32;
                for inner in 0..4 {
                    acc += a_data[row * 4 + inner] * b_data[inner * 5 + col];
                }
                expected[row * 5 + col] = acc;
            }
        }

        for (i, (&got, &want)) in simd_result.data().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "mismatch at index {i}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn test_matmul_simd_matches_scalar_large() {
        // Test with dimensions that exercise SIMD chunking (k=17 is not divisible by 8 or 4)
        let n = 5;
        let k = 17;
        let m = 7;
        let a_data: Vec<f32> = (0..(n * k))
            .map(|i| ((i * 7 + 3) % 100) as f32 * 0.01)
            .collect();
        let b_data: Vec<f32> = (0..(k * m))
            .map(|i| ((i * 11 + 5) % 100) as f32 * 0.01)
            .collect();

        let a = Tensor::from_data(a_data.clone(), vec![n, k]).expect("tensor from_data");
        let b = Tensor::from_data(b_data.clone(), vec![k, m]).expect("tensor from_data");

        let simd_result = matmul(&a, &b).expect("matmul");

        // Scalar reference
        let mut expected = vec![0.0_f32; n * m];
        for row in 0..n {
            for col in 0..m {
                let mut acc = 0.0_f32;
                for inner in 0..k {
                    acc += a_data[row * k + inner] * b_data[inner * m + col];
                }
                expected[row * m + col] = acc;
            }
        }

        for (i, (&got, &want)) in simd_result.data().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-2,
                "mismatch at index {i}: got {got}, want {want}"
            );
        }
    }

    // ── Unsqueeze / Squeeze ──────────────────────────────────────────────────

    #[test]
    fn test_unsqueeze_dim0() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3]).expect("tensor from_data");
        let u = t.unsqueeze(0).expect("unsqueeze");
        assert_eq!(u.shape(), &[1, 3]);
    }

    #[test]
    fn test_unsqueeze_dim_last() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3]).expect("tensor from_data");
        let u = t.unsqueeze(1).expect("unsqueeze");
        assert_eq!(u.shape(), &[3, 1]);
    }

    #[test]
    fn test_unsqueeze_out_of_range() {
        let t = Tensor::zeros(vec![2, 3]).expect("tensor zeros");
        assert!(t.unsqueeze(3).is_err());
    }

    #[test]
    fn test_squeeze_all() {
        let t = Tensor::from_data(vec![42.0], vec![1, 1, 1]).expect("tensor from_data");
        let s = t.squeeze(None).expect("squeeze");
        assert_eq!(s.shape(), &[1]); // scalar guard
    }

    #[test]
    fn test_squeeze_specific_dim() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![1, 3]).expect("tensor from_data");
        let s = t.squeeze(Some(0)).expect("squeeze");
        assert_eq!(s.shape(), &[3]);
    }

    // ── In-place operations ──────────────────────────────────────────────────

    #[test]
    fn test_relu_inplace() {
        let mut t =
            Tensor::from_data(vec![-2.0, 0.0, 3.0, -1.0], vec![4]).expect("tensor from_data");
        relu_inplace(&mut t);
        assert!(close(t.data()[0], 0.0));
        assert!(close(t.data()[1], 0.0));
        assert!(close(t.data()[2], 3.0));
        assert!(close(t.data()[3], 0.0));
    }

    #[test]
    fn test_add_inplace_same_shape() {
        let mut a = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3]).expect("tensor from_data");
        let b = Tensor::from_data(vec![10.0, 20.0, 30.0], vec![3]).expect("tensor from_data");
        add_inplace(&mut a, &b).expect("add_inplace");
        assert!(close(a.data()[0], 11.0));
        assert!(close(a.data()[1], 22.0));
        assert!(close(a.data()[2], 33.0));
    }

    #[test]
    fn test_add_inplace_broadcast_bias() {
        let mut a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
            .expect("tensor from_data");
        let b = Tensor::from_data(vec![10.0, 20.0, 30.0], vec![3]).expect("tensor from_data");
        add_inplace(&mut a, &b).expect("add_inplace");
        assert!(close(a.data()[0], 11.0));
        assert!(close(a.data()[5], 36.0));
    }

    #[test]
    fn test_mul_inplace() {
        let mut a = Tensor::from_data(vec![2.0, 3.0], vec![2]).expect("tensor from_data");
        let b = Tensor::from_data(vec![4.0, 5.0], vec![2]).expect("tensor from_data");
        mul_inplace(&mut a, &b).expect("mul_inplace");
        assert!(close(a.data()[0], 8.0));
        assert!(close(a.data()[1], 15.0));
    }

    #[test]
    fn test_scale_inplace() {
        let mut t = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3]).expect("tensor from_data");
        scale_inplace(&mut t, 2.0);
        assert!(close(t.data()[0], 2.0));
        assert!(close(t.data()[2], 6.0));
    }

    #[test]
    fn test_clamp_inplace() {
        let mut t = Tensor::from_data(vec![-5.0, 0.5, 10.0], vec![3]).expect("tensor from_data");
        clamp_inplace(&mut t, 0.0, 1.0);
        assert!(close(t.data()[0], 0.0));
        assert!(close(t.data()[1], 0.5));
        assert!(close(t.data()[2], 1.0));
    }

    // ── Broadcasting ─────────────────────────────────────────────────────────

    #[test]
    fn test_broadcast_add_nchw_bias() {
        // [N,C,H,W] + [1,C,1,1] — typical batch norm bias addition
        let a = Tensor::ones(vec![2, 3, 4, 4]).expect("tensor ones"); // all 1s
        let b =
            Tensor::from_data(vec![10.0, 20.0, 30.0], vec![1, 3, 1, 1]).expect("tensor from_data");
        let c = broadcast_add(&a, &b).expect("add");
        assert_eq!(c.shape(), &[2, 3, 4, 4]);
        // Channel 0 should all be 11.0, channel 1 → 21.0, channel 2 → 31.0
        assert!(close(c.data()[0], 11.0)); // N=0, C=0
        assert!(close(c.data()[16], 21.0)); // N=0, C=1, first spatial
        assert!(close(c.data()[32], 31.0)); // N=0, C=2
    }

    #[test]
    fn test_broadcast_add_scalar() {
        // [2,3] + [1] — scalar broadcast
        let a = Tensor::ones(vec![2, 3]).expect("tensor ones");
        let b = Tensor::from_data(vec![5.0], vec![1]).expect("tensor from_data");
        let c = broadcast_add(&a, &b).expect("add");
        assert_eq!(c.shape(), &[2, 3]);
        assert!(c.data().iter().all(|&v| close(v, 6.0)));
    }

    #[test]
    fn test_broadcast_add_incompatible() {
        let a = Tensor::zeros(vec![2, 3]).expect("tensor zeros");
        let b = Tensor::zeros(vec![2, 4]).expect("tensor zeros");
        assert!(broadcast_add(&a, &b).is_err());
    }

    #[test]
    fn test_broadcast_mul_channel_scale() {
        // [2,3,2,2] * [1,3,1,1] — per-channel scaling
        let a = Tensor::ones(vec![2, 3, 2, 2]).expect("tensor ones");
        let b = Tensor::from_data(vec![2.0, 3.0, 4.0], vec![1, 3, 1, 1]).expect("tensor from_data");
        let c = broadcast_mul(&a, &b).expect("broadcast_mul");
        assert_eq!(c.shape(), &[2, 3, 2, 2]);
        assert!(close(c.data()[0], 2.0)); // C=0
        assert!(close(c.data()[4], 3.0)); // C=1
        assert!(close(c.data()[8], 4.0)); // C=2
    }

    // ── Batched matmul ───────────────────────────────────────────────────────

    #[test]
    fn test_batched_matmul_basic() {
        // [2, 2, 3] @ [2, 3, 2] → [2, 2, 2]
        let a = Tensor::ones(vec![2, 2, 3]).expect("tensor ones");
        let b = Tensor::ones(vec![2, 3, 2]).expect("tensor ones");
        let c = batched_matmul(&a, &b).expect("matmul");
        assert_eq!(c.shape(), &[2, 2, 2]);
        // Each element should be 3.0 (sum of 3 ones)
        assert!(c.data().iter().all(|&v| close(v, 3.0)));
    }

    #[test]
    fn test_batched_matmul_broadcast_2d() {
        // [2, 2, 3] @ [3, 2] → [2, 2, 2]
        let a = Tensor::ones(vec![2, 2, 3]).expect("tensor ones");
        let b = Tensor::ones(vec![3, 2]).expect("tensor ones");
        let c = batched_matmul(&a, &b).expect("matmul");
        assert_eq!(c.shape(), &[2, 2, 2]);
        assert!(c.data().iter().all(|&v| close(v, 3.0)));
    }

    #[test]
    fn test_batched_matmul_batch_mismatch() {
        let a = Tensor::ones(vec![2, 2, 3]).expect("tensor ones");
        let b = Tensor::ones(vec![3, 3, 2]).expect("tensor ones");
        assert!(batched_matmul(&a, &b).is_err());
    }

    // ── 4D tensor basic ops ──────────────────────────────────────────────────

    #[test]
    fn test_4d_tensor_construction() {
        let t = Tensor::zeros(vec![2, 3, 4, 4]).expect("tensor zeros");
        assert_eq!(t.ndim(), 4);
        assert_eq!(t.numel(), 2 * 3 * 4 * 4);
    }

    #[test]
    fn test_4d_get_set() {
        let mut t = Tensor::zeros(vec![2, 3, 4, 4]).expect("tensor zeros");
        t.set(&[1, 2, 3, 3], 42.0).expect("set");
        assert!(close(t.get(&[1, 2, 3, 3]).expect("get"), 42.0));
    }

    #[test]
    fn test_add_inplace_general_broadcast() {
        // [2,3,2] + [1,3,1] via general broadcast
        let mut a = Tensor::ones(vec![2, 3, 2]).expect("tensor ones");
        let b = Tensor::from_data(vec![10.0, 20.0, 30.0], vec![1, 3, 1]).expect("tensor from_data");
        add_inplace(&mut a, &b).expect("add_inplace");
        assert!(close(a.data()[0], 11.0)); // [0,0,0]
        assert!(close(a.data()[2], 21.0)); // [0,1,0]
        assert!(close(a.data()[4], 31.0)); // [0,2,0]
    }

    // ── matmul_simd ──────────────────────────────────────────────────────────

    /// Reference scalar GEMM used to validate the optimised paths.
    fn ref_gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    c[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
        c
    }

    #[test]
    fn test_matmul_simd_identity() {
        // A = I_3, B = some matrix → C == B
        let a: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let c = matmul_simd(&a, &b, 3, 3, 3);
        for (i, (got, exp)) in c.iter().zip(b.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "identity mismatch at {i}: got={got}, exp={exp}"
            );
        }
    }

    #[test]
    fn test_matmul_simd_rectangular() {
        // 2×3 @ 3×4 → 2×4
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b: Vec<f32> = vec![
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ];
        let expected = ref_gemm(&a, &b, 2, 3, 4);
        let got = matmul_simd(&a, &b, 2, 3, 4);
        assert_eq!(got.len(), 8);
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "rect mismatch at {i}: got={g}, exp={e}"
            );
        }
    }

    #[test]
    fn test_matmul_simd_large_n_multiple_of_8() {
        // 4×16 @ 16×8 → 4×8  (n=8 exercises the AVX2 full-width path)
        let m = 4;
        let k = 16;
        let n = 8;
        let a: Vec<f32> = (0..m * k).map(|x| x as f32 * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|x| x as f32 * 0.01).collect();
        let expected = ref_gemm(&a, &b, m, k, n);
        let got = matmul_simd(&a, &b, m, k, n);
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-3,
                "large-n mismatch at {i}: got={g}, exp={e}"
            );
        }
    }

    #[test]
    fn test_matmul_simd_non_multiple_of_8() {
        // n=13 exercises the scalar tail in both paths.
        let m = 3;
        let k = 5;
        let n = 13;
        let a: Vec<f32> = (0..m * k).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|x| x as f32 * 0.1).collect();
        let expected = ref_gemm(&a, &b, m, k, n);
        let got = matmul_simd(&a, &b, m, k, n);
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-2,
                "tail mismatch at {i}: got={g}, exp={e}"
            );
        }
    }

    #[test]
    fn test_matmul_simd_agrees_with_tensor_matmul() {
        // matmul_simd result must match the scirs2-core blocked GEMM path.
        let m = 5;
        let k = 7;
        let n = 6;
        let a_data: Vec<f32> = (0..m * k).map(|x| (x as f32 + 1.0) * 0.5).collect();
        let b_data: Vec<f32> = (0..k * n).map(|x| (x as f32 + 1.0) * 0.3).collect();

        let ta = Tensor::from_data(a_data.clone(), vec![m, k]).expect("tensor from_data");
        let tb = Tensor::from_data(b_data.clone(), vec![k, n]).expect("tensor from_data");
        let tensor_result = matmul(&ta, &tb).expect("matmul");

        let simd_result = matmul_simd(&a_data, &b_data, m, k, n);

        for (i, (s, t)) in simd_result
            .iter()
            .zip(tensor_result.data().iter())
            .enumerate()
        {
            assert!(
                (s - t).abs() < 1e-3,
                "simd vs tensor_matmul mismatch at {i}: simd={s}, tensor={t}"
            );
        }
    }

    #[test]
    fn test_matmul_simd_runtime_agrees() {
        // matmul_simd_runtime must match matmul_simd for any input.
        let m = 4;
        let k = 8;
        let n = 9;
        let a: Vec<f32> = (0..m * k).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|x| x as f32 * 0.1).collect();
        let compile_time = matmul_simd(&a, &b, m, k, n);
        let runtime = matmul_simd_runtime(&a, &b, m, k, n);
        for (i, (c, r)) in compile_time.iter().zip(runtime.iter()).enumerate() {
            assert!(
                (c - r).abs() < 1e-4,
                "runtime mismatch at {i}: compile={c}, runtime={r}"
            );
        }
    }
}
