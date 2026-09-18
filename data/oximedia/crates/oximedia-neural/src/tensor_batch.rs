//! Extended tensor operations: views, concatenation, splitting, and
//! broadcasting-aware binary operations.
//!
//! These are separated from the core `tensor` module to keep individual
//! source files under the 2 000-line limit.

use crate::error::NeuralError;
use crate::tensor::Tensor;

// ──────────────────────────────────────────────────────────────────────────────
// TensorView — read-only view into a Tensor without copying
// ──────────────────────────────────────────────────────────────────────────────

/// A non-owning, read-only view into a [`Tensor`]'s data with a potentially
/// different logical shape (reshape-as-view) or a contiguous sub-region
/// (slice-as-view).
///
/// The view borrows the underlying `&[f32]` data and stores an offset plus
/// a logical shape, so no allocation is needed.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    /// Reference to the underlying flat data.
    data: &'a [f32],
    /// Logical shape of this view.
    shape: &'a [usize],
    /// Offset into the data slice.
    offset: usize,
    /// Number of elements visible through this view.
    numel: usize,
}

/// Owned-shape variant used when the view's shape differs from the source.
#[derive(Debug, Clone)]
pub struct TensorViewOwned<'a> {
    /// Reference to the underlying flat data.
    data: &'a [f32],
    /// Owned logical shape (may differ from the source tensor).
    shape: Vec<usize>,
    /// Offset into the data slice.
    offset: usize,
    /// Number of elements visible through this view.
    numel: usize,
}

impl TensorView<'_> {
    /// Returns the logical shape of this view.
    pub fn shape(&self) -> &[usize] {
        self.shape
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the number of visible elements.
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Returns the visible data as a contiguous slice.
    pub fn data(&self) -> &[f32] {
        &self.data[self.offset..self.offset + self.numel]
    }

    /// Materialises this view into a new owned `Tensor`.
    pub fn to_tensor(&self) -> Result<Tensor, NeuralError> {
        Tensor::from_data(self.data().to_vec(), self.shape.to_vec())
    }
}

impl TensorViewOwned<'_> {
    /// Returns the logical shape of this view.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the number of visible elements.
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Returns the visible data as a contiguous slice.
    pub fn data(&self) -> &[f32] {
        &self.data[self.offset..self.offset + self.numel]
    }

    /// Materialises this view into a new owned `Tensor`.
    pub fn to_tensor(&self) -> Result<Tensor, NeuralError> {
        Tensor::from_data(self.data().to_vec(), self.shape.clone())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// View creation on Tensor
// ──────────────────────────────────────────────────────────────────────────────

impl Tensor {
    /// Returns a read-only view that borrows the tensor's shape and data.
    ///
    /// This is zero-copy and zero-allocation.
    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            data: &self.data,
            shape: &self.shape,
            offset: 0,
            numel: self.data.len(),
        }
    }

    /// Returns a view that reinterprets the data with `new_shape` without
    /// copying.
    ///
    /// The total number of elements must be preserved.
    pub fn reshape_view(&self, new_shape: Vec<usize>) -> Result<TensorViewOwned<'_>, NeuralError> {
        if new_shape.iter().any(|&d| d == 0) {
            return Err(NeuralError::InvalidShape(
                "reshape_view: all dimensions must be > 0".to_string(),
            ));
        }
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(NeuralError::ShapeMismatch(format!(
                "reshape_view: cannot view {} elements as shape {:?} ({} elements)",
                self.numel(),
                new_shape,
                new_numel,
            )));
        }
        Ok(TensorViewOwned {
            data: &self.data,
            shape: new_shape,
            offset: 0,
            numel: new_numel,
        })
    }

    /// Returns a view of a contiguous slice along dimension 0.
    ///
    /// For a tensor with shape `[D0, D1, ..., Dn]`, `slice_view(start, end)`
    /// returns a view of shape `[end - start, D1, ..., Dn]` without copying.
    pub fn slice_view(&self, start: usize, end: usize) -> Result<TensorViewOwned<'_>, NeuralError> {
        if self.shape.is_empty() {
            return Err(NeuralError::InvalidShape(
                "slice_view: tensor has no dimensions".to_string(),
            ));
        }
        if start >= end {
            return Err(NeuralError::InvalidShape(format!(
                "slice_view: start {} must be < end {}",
                start, end
            )));
        }
        if end > self.shape[0] {
            return Err(NeuralError::IndexOutOfBounds(format!(
                "slice_view: end {} exceeds dim 0 size {}",
                end, self.shape[0]
            )));
        }
        let inner_numel: usize = if self.shape.len() > 1 {
            self.shape[1..].iter().product()
        } else {
            1
        };
        let offset = start * inner_numel;
        let count = (end - start) * inner_numel;
        let mut new_shape = self.shape.clone();
        new_shape[0] = end - start;
        Ok(TensorViewOwned {
            data: &self.data,
            shape: new_shape,
            offset,
            numel: count,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Concatenation along arbitrary axis
// ──────────────────────────────────────────────────────────────────────────────

/// Concatenates a list of tensors along the given axis.
///
/// All tensors must have the same number of dimensions and identical sizes
/// in every dimension except `axis`.
///
/// # Errors
///
/// - `EmptyInput` if `tensors` is empty.
/// - `InvalidShape` / `ShapeMismatch` on rank or dimension mismatches.
pub fn concat(tensors: &[&Tensor], axis: usize) -> Result<Tensor, NeuralError> {
    if tensors.is_empty() {
        return Err(NeuralError::EmptyInput(
            "concat: tensors list must not be empty".to_string(),
        ));
    }
    let ndim = tensors[0].ndim();
    if axis >= ndim {
        return Err(NeuralError::IndexOutOfBounds(format!(
            "concat: axis {} out of range for rank-{} tensors",
            axis, ndim
        )));
    }
    // Validate all shapes are compatible.
    for (i, t) in tensors.iter().enumerate().skip(1) {
        if t.ndim() != ndim {
            return Err(NeuralError::ShapeMismatch(format!(
                "concat: tensor[{}] rank {} != tensor[0] rank {}",
                i,
                t.ndim(),
                ndim
            )));
        }
        for d in 0..ndim {
            if d != axis && t.shape()[d] != tensors[0].shape()[d] {
                return Err(NeuralError::ShapeMismatch(format!(
                    "concat: tensor[{}] dim {} size {} != tensor[0] dim {} size {}",
                    i,
                    d,
                    t.shape()[d],
                    d,
                    tensors[0].shape()[d]
                )));
            }
        }
    }

    // Compute output shape.
    let mut out_shape = tensors[0].shape().to_vec();
    let total_along_axis: usize = tensors.iter().map(|t| t.shape()[axis]).sum();
    out_shape[axis] = total_along_axis;
    let out_numel: usize = out_shape.iter().product();

    // Compute strides.
    let outer: usize = out_shape[..axis].iter().product();
    let inner: usize = if axis + 1 < ndim {
        out_shape[axis + 1..].iter().product()
    } else {
        1
    };

    let mut out_data = vec![0.0_f32; out_numel];

    // For each outer index, copy slabs from each tensor.
    let mut axis_offset = 0usize;
    for t in tensors {
        let t_axis_size = t.shape()[axis];
        let t_inner: usize = if axis + 1 < ndim {
            t.shape()[axis + 1..].iter().product()
        } else {
            1
        };
        for o in 0..outer {
            let out_base = o * (total_along_axis * inner) + axis_offset * inner;
            let in_base = o * (t_axis_size * t_inner);
            let slab_len = t_axis_size * t_inner;
            out_data[out_base..out_base + slab_len]
                .copy_from_slice(&t.data()[in_base..in_base + slab_len]);
        }
        axis_offset += t_axis_size;
    }

    Tensor::from_data(out_data, out_shape)
}

/// Splits a tensor along the given axis into `num_splits` equal parts.
///
/// The size of `axis` must be divisible by `num_splits`.
///
/// # Errors
///
/// - `InvalidShape` if `axis` is out of range.
/// - `ShapeMismatch` if the axis size is not divisible by `num_splits`.
pub fn split(tensor: &Tensor, axis: usize, num_splits: usize) -> Result<Vec<Tensor>, NeuralError> {
    let ndim = tensor.ndim();
    if axis >= ndim {
        return Err(NeuralError::IndexOutOfBounds(format!(
            "split: axis {} out of range for rank-{} tensor",
            axis, ndim
        )));
    }
    if num_splits == 0 {
        return Err(NeuralError::InvalidShape(
            "split: num_splits must be > 0".to_string(),
        ));
    }
    let axis_size = tensor.shape()[axis];
    if axis_size % num_splits != 0 {
        return Err(NeuralError::ShapeMismatch(format!(
            "split: axis {} size {} is not divisible by num_splits {}",
            axis, axis_size, num_splits
        )));
    }
    let chunk_size = axis_size / num_splits;

    let outer: usize = tensor.shape()[..axis].iter().product();
    let inner: usize = if axis + 1 < ndim {
        tensor.shape()[axis + 1..].iter().product()
    } else {
        1
    };

    let mut result = Vec::with_capacity(num_splits);
    for s in 0..num_splits {
        let mut chunk_shape = tensor.shape().to_vec();
        chunk_shape[axis] = chunk_size;
        let chunk_numel: usize = chunk_shape.iter().product();
        let mut chunk_data = Vec::with_capacity(chunk_numel);

        for o in 0..outer {
            let in_base = o * (axis_size * inner) + s * chunk_size * inner;
            chunk_data.extend_from_slice(&tensor.data()[in_base..in_base + chunk_size * inner]);
        }

        result.push(Tensor::from_data(chunk_data, chunk_shape)?);
    }

    Ok(result)
}

/// Splits a tensor along the given axis at the specified split sizes.
///
/// The sum of `sizes` must equal `tensor.shape()[axis]`.
pub fn split_sizes(
    tensor: &Tensor,
    axis: usize,
    sizes: &[usize],
) -> Result<Vec<Tensor>, NeuralError> {
    let ndim = tensor.ndim();
    if axis >= ndim {
        return Err(NeuralError::IndexOutOfBounds(format!(
            "split_sizes: axis {} out of range for rank-{} tensor",
            axis, ndim
        )));
    }
    let axis_size = tensor.shape()[axis];
    let total: usize = sizes.iter().sum();
    if total != axis_size {
        return Err(NeuralError::ShapeMismatch(format!(
            "split_sizes: sum of sizes {} != axis {} size {}",
            total, axis, axis_size
        )));
    }

    let outer: usize = tensor.shape()[..axis].iter().product();
    let inner: usize = if axis + 1 < ndim {
        tensor.shape()[axis + 1..].iter().product()
    } else {
        1
    };

    let mut result = Vec::with_capacity(sizes.len());
    let mut offset = 0usize;
    for &sz in sizes {
        let mut chunk_shape = tensor.shape().to_vec();
        chunk_shape[axis] = sz;
        let chunk_numel: usize = chunk_shape.iter().product();
        let mut chunk_data = Vec::with_capacity(chunk_numel);

        for o in 0..outer {
            let in_base = o * (axis_size * inner) + offset * inner;
            chunk_data.extend_from_slice(&tensor.data()[in_base..in_base + sz * inner]);
        }

        result.push(Tensor::from_data(chunk_data, chunk_shape)?);
        offset += sz;
    }

    Ok(result)
}

// ──────────────────────────────────────────────────────────────────────────────
// Broadcasting-aware binary subtraction
// ──────────────────────────────────────────────────────────────────────────────

/// Element-wise subtraction with NumPy-style broadcasting: `a - b`.
pub fn broadcast_sub(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuralError> {
    use crate::tensor::{broadcast_add, scale_inplace};
    // Negate b, then add.
    let mut neg_b = b.clone();
    scale_inplace(&mut neg_b, -1.0);
    broadcast_add(a, &neg_b)
}

// ──────────────────────────────────────────────────────────────────────────────
// Batch operations
// ──────────────────────────────────────────────────────────────────────────────

/// Applies a function element-wise to each sample in a batch tensor.
///
/// The tensor shape is `[B, ...]`. The function receives each sample as a
/// flat `&mut [f32]` slice and is called `B` times.
pub fn batch_apply<F>(tensor: &mut Tensor, f: F) -> Result<(), NeuralError>
where
    F: Fn(&mut [f32]),
{
    if tensor.ndim() < 2 {
        return Err(NeuralError::InvalidShape(
            "batch_apply: tensor must have at least 2 dimensions".to_string(),
        ));
    }
    let batch_size = tensor.shape()[0];
    let sample_numel: usize = tensor.shape()[1..].iter().product();
    let data = tensor.data_mut();
    for b in 0..batch_size {
        let start = b * sample_numel;
        f(&mut data[start..start + sample_numel]);
    }
    Ok(())
}

/// Reduces along the batch dimension (dim 0) by averaging.
///
/// Input shape `[B, ...]`, output shape `[...]`.
pub fn batch_mean(tensor: &Tensor) -> Result<Tensor, NeuralError> {
    if tensor.ndim() < 2 {
        return Err(NeuralError::InvalidShape(
            "batch_mean: tensor must have at least 2 dimensions".to_string(),
        ));
    }
    let batch_size = tensor.shape()[0];
    let sample_shape = tensor.shape()[1..].to_vec();
    let sample_numel: usize = sample_shape.iter().product();
    let mut out_data = vec![0.0_f32; sample_numel];

    for b in 0..batch_size {
        let start = b * sample_numel;
        for i in 0..sample_numel {
            out_data[i] += tensor.data()[start + i];
        }
    }

    let inv_b = 1.0 / batch_size as f32;
    for v in out_data.iter_mut() {
        *v *= inv_b;
    }

    Tensor::from_data(out_data, sample_shape)
}

/// Reduces along the batch dimension (dim 0) by taking element-wise max.
///
/// Input shape `[B, ...]`, output shape `[...]`.
pub fn batch_max(tensor: &Tensor) -> Result<Tensor, NeuralError> {
    if tensor.ndim() < 2 {
        return Err(NeuralError::InvalidShape(
            "batch_max: tensor must have at least 2 dimensions".to_string(),
        ));
    }
    let batch_size = tensor.shape()[0];
    let sample_shape = tensor.shape()[1..].to_vec();
    let sample_numel: usize = sample_shape.iter().product();
    let mut out_data = vec![f32::NEG_INFINITY; sample_numel];

    for b in 0..batch_size {
        let start = b * sample_numel;
        for i in 0..sample_numel {
            let v = tensor.data()[start + i];
            if v > out_data[i] {
                out_data[i] = v;
            }
        }
    }

    Tensor::from_data(out_data, sample_shape)
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

    // ── TensorView ──────────────────────────────────────────────────────────

    #[test]
    fn test_view_basic() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).ok();
        let t = t.expect("should succeed");
        let v = t.view();
        assert_eq!(v.shape(), &[2, 2]);
        assert_eq!(v.numel(), 4);
        assert_eq!(v.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_view_to_tensor() {
        let t = Tensor::ones(vec![3, 4]).expect("ok");
        let v = t.view();
        let t2 = v.to_tensor().expect("ok");
        assert_eq!(t2.shape(), &[3, 4]);
        assert_eq!(t2.data(), t.data());
    }

    #[test]
    fn test_reshape_view_basic() {
        let t = Tensor::from_data((0..12).map(|i| i as f32).collect(), vec![3, 4]).expect("ok");
        let v = t.reshape_view(vec![4, 3]).expect("ok");
        assert_eq!(v.shape(), &[4, 3]);
        assert_eq!(v.numel(), 12);
        assert_eq!(v.data(), t.data());
    }

    #[test]
    fn test_reshape_view_mismatch() {
        let t = Tensor::ones(vec![3, 4]).expect("ok");
        assert!(t.reshape_view(vec![5, 3]).is_err());
    }

    #[test]
    fn test_reshape_view_zero_dim() {
        let t = Tensor::ones(vec![3, 4]).expect("ok");
        assert!(t.reshape_view(vec![0, 12]).is_err());
    }

    #[test]
    fn test_slice_view_basic() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).expect("ok");
        let v = t.slice_view(1, 3).expect("ok");
        assert_eq!(v.shape(), &[2, 2]);
        assert_eq!(v.data(), &[3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_slice_view_single_row() {
        let t =
            Tensor::from_data(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![3, 2]).expect("ok");
        let v = t.slice_view(0, 1).expect("ok");
        assert_eq!(v.shape(), &[1, 2]);
        assert_eq!(v.data(), &[10.0, 20.0]);
    }

    #[test]
    fn test_slice_view_out_of_bounds() {
        let t = Tensor::ones(vec![3, 2]).expect("ok");
        assert!(t.slice_view(0, 4).is_err());
    }

    #[test]
    fn test_slice_view_start_ge_end() {
        let t = Tensor::ones(vec![3, 2]).expect("ok");
        assert!(t.slice_view(2, 1).is_err());
    }

    // ── concat ──────────────────────────────────────────────────────────────

    #[test]
    fn test_concat_axis0() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("ok");
        let b = Tensor::from_data(vec![5.0, 6.0], vec![1, 2]).expect("ok");
        let c = concat(&[&a, &b], 0).expect("ok");
        assert_eq!(c.shape(), &[3, 2]);
        assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_concat_axis1() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("ok");
        let b = Tensor::from_data(vec![5.0, 6.0], vec![2, 1]).expect("ok");
        let c = concat(&[&a, &b], 1).expect("ok");
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.data(), &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_concat_3d_axis1() {
        let a = Tensor::from_data((0..12).map(|i| i as f32).collect(), vec![2, 2, 3]).expect("ok");
        let b = Tensor::from_data((12..18).map(|i| i as f32).collect(), vec![2, 1, 3]).expect("ok");
        let c = concat(&[&a, &b], 1).expect("ok");
        assert_eq!(c.shape(), &[2, 3, 3]);
    }

    #[test]
    fn test_concat_empty_error() {
        let result: Result<Tensor, _> = concat(&[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_concat_rank_mismatch() {
        let a = Tensor::ones(vec![2, 3]).expect("ok");
        let b = Tensor::ones(vec![2, 3, 1]).expect("ok");
        assert!(concat(&[&a, &b], 0).is_err());
    }

    #[test]
    fn test_concat_shape_mismatch_non_axis_dim() {
        let a = Tensor::ones(vec![2, 3]).expect("ok");
        let b = Tensor::ones(vec![2, 4]).expect("ok");
        assert!(concat(&[&a, &b], 0).is_err());
    }

    // ── split ───────────────────────────────────────────────────────────────

    #[test]
    fn test_split_axis0() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]).expect("ok");
        let parts = split(&t, 0, 3).expect("ok");
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].data(), &[1.0, 2.0]);
        assert_eq!(parts[1].data(), &[3.0, 4.0]);
        assert_eq!(parts[2].data(), &[5.0, 6.0]);
    }

    #[test]
    fn test_split_2d_axis1() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4])
            .expect("ok");
        let parts = split(&t, 1, 2).expect("ok");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].shape(), &[2, 2]);
        assert_eq!(parts[0].data(), &[1.0, 2.0, 5.0, 6.0]);
        assert_eq!(parts[1].data(), &[3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn test_split_not_divisible_error() {
        let t = Tensor::ones(vec![5]).expect("ok");
        assert!(split(&t, 0, 3).is_err());
    }

    #[test]
    fn test_split_zero_splits_error() {
        let t = Tensor::ones(vec![6]).expect("ok");
        assert!(split(&t, 0, 0).is_err());
    }

    // ── split_sizes ─────────────────────────────────────────────────────────

    #[test]
    fn test_split_sizes_basic() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).expect("ok");
        let parts = split_sizes(&t, 0, &[2, 3]).expect("ok");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].data(), &[1.0, 2.0]);
        assert_eq!(parts[1].data(), &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_split_sizes_sum_mismatch() {
        let t = Tensor::ones(vec![5]).expect("ok");
        assert!(split_sizes(&t, 0, &[2, 2]).is_err());
    }

    // ── broadcast_sub ───────────────────────────────────────────────────────

    #[test]
    fn test_broadcast_sub_same_shape() {
        let a = Tensor::from_data(vec![5.0, 3.0, 1.0], vec![3]).expect("ok");
        let b = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3]).expect("ok");
        let c = broadcast_sub(&a, &b).expect("ok");
        assert_eq!(c.data(), &[4.0, 1.0, -2.0]);
    }

    #[test]
    fn test_broadcast_sub_broadcast() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("ok");
        let b = Tensor::from_data(vec![1.0, 0.0], vec![1, 2]).expect("ok");
        let c = broadcast_sub(&a, &b).expect("ok");
        assert_eq!(c.shape(), &[2, 2]);
        assert!(close(c.data()[0], 0.0));
        assert!(close(c.data()[1], 2.0));
        assert!(close(c.data()[2], 2.0));
        assert!(close(c.data()[3], 4.0));
    }

    // ── batch_apply ─────────────────────────────────────────────────────────

    #[test]
    fn test_batch_apply_relu() {
        let mut t = Tensor::from_data(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]).expect("ok");
        batch_apply(&mut t, |s| {
            for v in s.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        })
        .expect("ok");
        assert_eq!(t.data(), &[0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_batch_apply_1d_error() {
        let mut t = Tensor::ones(vec![5]).expect("ok");
        assert!(batch_apply(&mut t, |_| {}).is_err());
    }

    // ── batch_mean ──────────────────────────────────────────────────────────

    #[test]
    fn test_batch_mean_basic() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("ok");
        let m = batch_mean(&t).expect("ok");
        assert_eq!(m.shape(), &[2]);
        assert!(close(m.data()[0], 2.0));
        assert!(close(m.data()[1], 3.0));
    }

    #[test]
    fn test_batch_mean_3d() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2])
            .expect("ok");
        let m = batch_mean(&t).expect("ok");
        assert_eq!(m.shape(), &[2, 2]);
        // mean of [1,2,3,4] and [5,6,7,8] element-wise
        assert!(close(m.data()[0], 3.0));
        assert!(close(m.data()[1], 4.0));
        assert!(close(m.data()[2], 5.0));
        assert!(close(m.data()[3], 6.0));
    }

    // ── batch_max ───────────────────────────────────────────────────────────

    #[test]
    fn test_batch_max_basic() {
        let t = Tensor::from_data(vec![1.0, 4.0, 3.0, 2.0], vec![2, 2]).expect("ok");
        let m = batch_max(&t).expect("ok");
        assert_eq!(m.shape(), &[2]);
        assert!(close(m.data()[0], 3.0));
        assert!(close(m.data()[1], 4.0));
    }

    // ── concat + split roundtrip ────────────────────────────────────────────

    #[test]
    fn test_concat_split_roundtrip() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("ok");
        let b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).expect("ok");
        let c = concat(&[&a, &b], 0).expect("ok");
        assert_eq!(c.shape(), &[4, 2]);
        let parts = split(&c, 0, 2).expect("ok");
        assert_eq!(parts[0].data(), a.data());
        assert_eq!(parts[1].data(), b.data());
    }

    #[test]
    fn test_concat_many_tensors() {
        let t1 = Tensor::ones(vec![2, 3]).expect("ok");
        let t2 = Tensor::zeros(vec![2, 3]).expect("ok");
        let t3 = Tensor::ones(vec![2, 3]).expect("ok");
        let c = concat(&[&t1, &t2, &t3], 0).expect("ok");
        assert_eq!(c.shape(), &[6, 3]);
    }

    #[test]
    fn test_split_3d_axis0() {
        let t = Tensor::from_data((0..24).map(|i| i as f32).collect(), vec![4, 2, 3]).expect("ok");
        let parts = split(&t, 0, 2).expect("ok");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].shape(), &[2, 2, 3]);
        assert_eq!(parts[1].shape(), &[2, 2, 3]);
    }
}
