//! Gather/Scatter operations for tensor indexing and selection.
//!
//! This module provides fundamental gather and scatter operations used in
//! machine learning and tensor computation, including:
//!
//! - **gather**: Select slices from a tensor along an axis by index.
//! - **gather_nd**: Element-wise gather using an index tensor of the same rank.
//! - **scatter_add**: Accumulate values into a target tensor at given indices.
//! - **scatter_max / scatter_min**: Scatter with max/min reduction semantics.
//! - **top_k**: Retrieve top-k (or bottom-k) values and their indices along an axis.
//! - **masked_select**: Extract elements where a boolean mask is true.
//! - **masked_fill**: Replace masked positions with a fill value.
//! - **IndexStats**: Statistics about an index array (coverage, duplicates, etc.).

use scirs2_core::ndarray::{Array, Array1, ArrayD, Axis, Dimension, IxDyn};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error type for gather/scatter operations.
#[derive(Debug, Clone)]
pub enum GatherScatterError {
    /// An index value (after negative normalization) was out of bounds.
    OutOfBoundsIndex {
        index: i64,
        axis_len: usize,
        axis: usize,
    },
    /// Two shapes that were required to match did not.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// The requested axis exceeds the tensor's number of dimensions.
    AxisOutOfRange { axis: usize, ndim: usize },
    /// The input tensor (or index collection) was empty when it must not be.
    EmptyInput,
    /// `gather_nd` requires `indices` and `input` to have the same rank.
    IndexRankMismatch {
        input_ndim: usize,
        index_ndim: usize,
    },
    /// Requested `k` exceeds the axis dimension length.
    KTooLarge { k: usize, axis_len: usize },
}

impl std::fmt::Display for GatherScatterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GatherScatterError::OutOfBoundsIndex {
                index,
                axis_len,
                axis,
            } => write!(
                f,
                "index {index} is out of bounds for axis {axis} with length {axis_len}"
            ),
            GatherScatterError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected:?}, got {got:?}")
            }
            GatherScatterError::AxisOutOfRange { axis, ndim } => {
                write!(
                    f,
                    "axis {axis} is out of range for tensor with {ndim} dimensions"
                )
            }
            GatherScatterError::EmptyInput => write!(f, "empty input"),
            GatherScatterError::IndexRankMismatch {
                input_ndim,
                index_ndim,
            } => write!(
                f,
                "index rank mismatch: input has {input_ndim} dims, indices have {index_ndim} dims"
            ),
            GatherScatterError::KTooLarge { k, axis_len } => {
                write!(f, "k={k} exceeds axis length {axis_len}")
            }
        }
    }
}

impl std::error::Error for GatherScatterError {}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Normalize a potentially-negative index into a non-negative position.
///
/// Negative indices wrap from the end Python-style (−1 → last).
/// Returns an error if the normalised value is still out of `[0, axis_len)`.
#[inline]
fn normalize_index(raw: i64, axis_len: usize, axis: usize) -> Result<usize, GatherScatterError> {
    let len = axis_len as i64;
    let normalized = if raw < 0 { len + raw } else { raw };
    if normalized < 0 || normalized >= len {
        return Err(GatherScatterError::OutOfBoundsIndex {
            index: raw,
            axis_len,
            axis,
        });
    }
    Ok(normalized as usize)
}

/// Validate that `axis < ndim`.
#[inline]
fn check_axis(axis: usize, ndim: usize) -> Result<(), GatherScatterError> {
    if axis >= ndim {
        return Err(GatherScatterError::AxisOutOfRange { axis, ndim });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// gather
// ---------------------------------------------------------------------------

/// Gather slices from `input` along `axis` using scalar indices.
///
/// Negative indices are supported (Python-style: −1 = last element).
///
/// # Shape behaviour
/// - `input` shape `[d0, d1, …, dA, …, dN]` and `axis = A`
/// - `output` shape `[d0, d1, …, k, …, dN]` where `k = indices.len()`
///
/// # Errors
/// - [`GatherScatterError::AxisOutOfRange`] if `axis >= input.ndim()`
/// - [`GatherScatterError::EmptyInput`] if `indices` is empty
/// - [`GatherScatterError::OutOfBoundsIndex`] if any index is out of range
pub fn gather(
    input: &ArrayD<f64>,
    indices: &[i64],
    axis: usize,
) -> Result<ArrayD<f64>, GatherScatterError> {
    let ndim = input.ndim();
    check_axis(axis, ndim)?;

    if indices.is_empty() {
        return Err(GatherScatterError::EmptyInput);
    }

    let axis_len = input.shape()[axis];

    // Collect one view per index, then stack them.
    let mut slices: Vec<ArrayD<f64>> = Vec::with_capacity(indices.len());
    for &raw in indices {
        let idx = normalize_index(raw, axis_len, axis)?;
        let view = input.index_axis(Axis(axis), idx);
        slices.push(view.to_owned().into_dyn());
    }

    // Stack the collected slices back along `axis`.
    let views: Vec<_> = slices.iter().map(|s| s.view()).collect();
    let stacked = scirs2_core::ndarray::stack(Axis(axis), &views).map_err(|_| {
        GatherScatterError::ShapeMismatch {
            expected: input.shape().to_vec(),
            got: vec![],
        }
    })?;
    Ok(stacked)
}

// ---------------------------------------------------------------------------
// gather_nd
// ---------------------------------------------------------------------------

/// Gather with an index tensor of the same rank as `input`.
///
/// For every position `(i0, i1, …, iN)` in the output:
/// ```text
/// output[i0,..,iN] = input[i0,..,indices[i0,..,iN],..,iN]
/// ```
/// where `indices` replaces the `axis` dimension.
///
/// # Requirements
/// `input.ndim() == indices.ndim()`
///
/// # Errors
/// - [`GatherScatterError::AxisOutOfRange`]
/// - [`GatherScatterError::IndexRankMismatch`]
/// - [`GatherScatterError::ShapeMismatch`] if non-axis dimensions differ
/// - [`GatherScatterError::OutOfBoundsIndex`] if any index is out of range
pub fn gather_nd(
    input: &ArrayD<f64>,
    indices: &ArrayD<i64>,
    axis: usize,
) -> Result<ArrayD<f64>, GatherScatterError> {
    let ndim = input.ndim();
    check_axis(axis, ndim)?;

    if indices.ndim() != ndim {
        return Err(GatherScatterError::IndexRankMismatch {
            input_ndim: ndim,
            index_ndim: indices.ndim(),
        });
    }

    // Non-axis dimensions must match between input and indices.
    for d in 0..ndim {
        if d != axis && input.shape()[d] != indices.shape()[d] {
            return Err(GatherScatterError::ShapeMismatch {
                expected: input.shape().to_vec(),
                got: indices.shape().to_vec(),
            });
        }
    }

    let axis_len = input.shape()[axis];
    let output_shape = indices.shape().to_vec();
    let total = output_shape.iter().product::<usize>();

    let mut out_flat: Vec<f64> = Vec::with_capacity(total);

    // Iterate over every position in the indices tensor.
    for (multi_idx, &raw_idx) in indices.indexed_iter() {
        let gather_pos = normalize_index(raw_idx, axis_len, axis)?;

        // Build the full multi-index into `input`.
        let mut input_idx: Vec<usize> = multi_idx.slice().to_vec();
        input_idx[axis] = gather_pos;

        let val =
            input
                .get(IxDyn(&input_idx))
                .copied()
                .ok_or(GatherScatterError::OutOfBoundsIndex {
                    index: raw_idx,
                    axis_len,
                    axis,
                })?;
        out_flat.push(val);
    }

    Array::from_shape_vec(IxDyn(&output_shape), out_flat).map_err(|_| {
        GatherScatterError::ShapeMismatch {
            expected: output_shape,
            got: vec![],
        }
    })
}

// ---------------------------------------------------------------------------
// Internal scatter kernel
// ---------------------------------------------------------------------------

/// Generic scatter kernel — applies `combine` to accumulate values.
fn scatter_generic<F>(
    input: &ArrayD<f64>,
    indices: &[i64],
    axis: usize,
    output_size: usize,
    init_value: f64,
    combine: F,
) -> Result<ArrayD<f64>, GatherScatterError>
where
    F: Fn(f64, f64) -> f64,
{
    let ndim = input.ndim();
    check_axis(axis, ndim)?;

    let in_axis_len = input.shape()[axis];
    if indices.len() != in_axis_len {
        return Err(GatherScatterError::ShapeMismatch {
            expected: vec![in_axis_len],
            got: vec![indices.len()],
        });
    }

    // Build output shape: replace axis dimension with output_size.
    let mut out_shape = input.shape().to_vec();
    out_shape[axis] = output_size;
    let out_total = out_shape.iter().product::<usize>();

    let mut out_data: Vec<f64> = vec![init_value; out_total];

    // For each slice along the input axis…
    for (i, &raw) in indices.iter().enumerate() {
        let dst = normalize_index(raw, output_size, axis)?;

        // Iterate over all positions in the slice (all axes except `axis`).
        let input_slice = input.index_axis(Axis(axis), i);
        let output_slice_offset = compute_axis_offset(&out_shape, axis, dst);

        for (flat_in, &val) in input_slice.iter().enumerate() {
            let flat_out = output_slice_offset + slice_flat_index(&out_shape, axis, flat_in);
            out_data[flat_out] = combine(out_data[flat_out], val);
        }
    }

    Array::from_shape_vec(IxDyn(&out_shape), out_data).map_err(|_| {
        GatherScatterError::ShapeMismatch {
            expected: out_shape,
            got: vec![],
        }
    })
}

/// Compute the flat offset of position `pos` along `axis` in a tensor of `shape`.
///
/// Multiplies `pos` by the stride for that axis.
fn compute_axis_offset(shape: &[usize], axis: usize, pos: usize) -> usize {
    // Stride for axis = product of all dims beyond axis.
    let stride: usize = shape[axis + 1..].iter().product();
    pos * stride
}

/// Given a flat index within a slice (i.e., ignoring the `axis` dimension),
/// convert it back to a flat index in the full tensor at `axis = 0` offset.
///
/// The inner dimensions (beyond `axis`) cycle the fastest; the outer ones cycle slowly.
fn slice_flat_index(shape: &[usize], axis: usize, flat_in: usize) -> usize {
    // The slice has shape = shape[0..axis] ++ shape[axis+1..].
    let inner_size: usize = shape[axis + 1..].iter().product();
    let outer_idx = flat_in / inner_size; // index into the outer (pre-axis) part
    let inner_idx = flat_in % inner_size; // index into the inner (post-axis) part

    // Outer stride = shape[axis] * inner_size (full row through the axis dimension).
    let outer_stride = shape[axis] * inner_size;
    outer_idx * outer_stride + inner_idx
}

// ---------------------------------------------------------------------------
// scatter_add
// ---------------------------------------------------------------------------

/// Scatter-add: accumulate input slices into output positions along `axis`.
///
/// For each `(i, idx)` in `enumerate(indices)`:
/// `output[…, idx, …] += input[…, i, …]`
///
/// `output_size` defines the length of `axis` in the output.
/// `init_value` is the initial fill (0.0 for sum identity).
///
/// # Errors
/// - [`GatherScatterError::AxisOutOfRange`]
/// - [`GatherScatterError::ShapeMismatch`] if `indices.len() != input.shape()[axis]`
/// - [`GatherScatterError::OutOfBoundsIndex`] if any index ∉ `[0, output_size)`
pub fn scatter_add(
    input: &ArrayD<f64>,
    indices: &[i64],
    axis: usize,
    output_size: usize,
    init_value: f64,
) -> Result<ArrayD<f64>, GatherScatterError> {
    scatter_generic(input, indices, axis, output_size, init_value, |a, b| a + b)
}

// ---------------------------------------------------------------------------
// scatter_max
// ---------------------------------------------------------------------------

/// Scatter-max: scatter with max reduction semantics.
///
/// Positions that receive no input retain `init_value`.
/// A common choice is `f64::NEG_INFINITY` for a true maximum.
///
/// # Errors
/// Same as [`scatter_add`].
pub fn scatter_max(
    input: &ArrayD<f64>,
    indices: &[i64],
    axis: usize,
    output_size: usize,
    init_value: f64,
) -> Result<ArrayD<f64>, GatherScatterError> {
    scatter_generic(input, indices, axis, output_size, init_value, f64::max)
}

// ---------------------------------------------------------------------------
// scatter_min
// ---------------------------------------------------------------------------

/// Scatter-min: scatter with min reduction semantics.
///
/// Positions that receive no input retain `init_value`.
/// A common choice is `f64::INFINITY` for a true minimum.
///
/// # Errors
/// Same as [`scatter_add`].
pub fn scatter_min(
    input: &ArrayD<f64>,
    indices: &[i64],
    axis: usize,
    output_size: usize,
    init_value: f64,
) -> Result<ArrayD<f64>, GatherScatterError> {
    scatter_generic(input, indices, axis, output_size, init_value, f64::min)
}

// ---------------------------------------------------------------------------
// top_k
// ---------------------------------------------------------------------------

/// Return the top-`k` values and their original indices along `axis`.
///
/// The output shape has `axis` replaced by `k`.
/// When `largest = true` the greatest values are returned (sorted descending);
/// when `false` the smallest are returned (sorted ascending).
///
/// # Errors
/// - [`GatherScatterError::AxisOutOfRange`]
/// - [`GatherScatterError::KTooLarge`] if `k > axis_len`
/// - [`GatherScatterError::EmptyInput`] if input is empty
pub fn top_k(
    input: &ArrayD<f64>,
    k: usize,
    axis: usize,
    largest: bool,
) -> Result<(ArrayD<f64>, ArrayD<i64>), GatherScatterError> {
    let ndim = input.ndim();
    check_axis(axis, ndim)?;

    let axis_len = input.shape()[axis];
    if k > axis_len {
        return Err(GatherScatterError::KTooLarge { k, axis_len });
    }
    if input.is_empty() {
        return Err(GatherScatterError::EmptyInput);
    }

    // Output shape: axis dimension replaced by k.
    let mut out_shape = input.shape().to_vec();
    out_shape[axis] = k;
    let out_total = out_shape.iter().product::<usize>();

    let mut val_flat: Vec<f64> = Vec::with_capacity(out_total);
    let mut idx_flat: Vec<i64> = Vec::with_capacity(out_total);

    // Iterate over all outer slices (everything except the axis dimension).
    let outer_size: usize = input.shape()[..axis].iter().product();
    let inner_size: usize = input.shape()[axis + 1..].iter().product();

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Collect (value, original_index) pairs for this 1-D slice.
            let mut pairs: Vec<(f64, usize)> = (0..axis_len)
                .map(|a| {
                    // Build multi-index.
                    let mut midx = vec![0usize; ndim];
                    let outer_stride_per_dim =
                        compute_outer_multi_index(input.shape(), axis, outer);
                    let inner_multi = compute_inner_multi_index(input.shape(), axis, inner);
                    midx[..axis].copy_from_slice(&outer_stride_per_dim[..axis]);
                    midx[axis] = a;
                    for d in (axis + 1)..ndim {
                        midx[d] = inner_multi[d - axis - 1];
                    }
                    let val = input.get(IxDyn(&midx)).copied().unwrap_or(f64::NAN);
                    (val, a)
                })
                .collect();

            // Sort: descending for largest, ascending for smallest.
            pairs.sort_by(|(a, _), (b, _)| {
                if largest {
                    b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                }
            });

            for (v, i) in pairs.iter().take(k) {
                val_flat.push(*v);
                idx_flat.push(*i as i64);
            }
        }
    }

    // The flat layout must interleave: outer × k × inner.
    // Current layout is outer × inner × k — we need to transpose.
    let val_array = reorder_topk_output(&val_flat, outer_size, k, inner_size, &out_shape)?;
    let idx_array = reorder_topk_output_i64(&idx_flat, outer_size, k, inner_size, &out_shape)?;

    Ok((val_array, idx_array))
}

/// Convert flat layout `(outer, inner, k)` → `(outer, k, inner)` for `f64`.
fn reorder_topk_output(
    data: &[f64],
    outer: usize,
    k: usize,
    inner: usize,
    shape: &[usize],
) -> Result<ArrayD<f64>, GatherScatterError> {
    let total = outer * k * inner;
    let mut out: Vec<f64> = vec![0.0; total];
    for o in 0..outer {
        for ki in 0..k {
            for i in 0..inner {
                let src = o * inner * k + i * k + ki;
                let dst = o * k * inner + ki * inner + i;
                out[dst] = data[src];
            }
        }
    }
    Array::from_shape_vec(IxDyn(shape), out).map_err(|_| GatherScatterError::ShapeMismatch {
        expected: shape.to_vec(),
        got: vec![],
    })
}

/// Same as `reorder_topk_output` but for `i64`.
fn reorder_topk_output_i64(
    data: &[i64],
    outer: usize,
    k: usize,
    inner: usize,
    shape: &[usize],
) -> Result<ArrayD<i64>, GatherScatterError> {
    let total = outer * k * inner;
    let mut out: Vec<i64> = vec![0; total];
    for o in 0..outer {
        for ki in 0..k {
            for i in 0..inner {
                let src = o * inner * k + i * k + ki;
                let dst = o * k * inner + ki * inner + i;
                out[dst] = data[src];
            }
        }
    }
    Array::from_shape_vec(IxDyn(shape), out).map_err(|_| GatherScatterError::ShapeMismatch {
        expected: shape.to_vec(),
        got: vec![],
    })
}

/// Decompose a flat outer index into per-dimension indices for dimensions `0..axis`.
fn compute_outer_multi_index(shape: &[usize], axis: usize, flat: usize) -> Vec<usize> {
    let mut result = vec![0usize; axis];
    let mut remaining = flat;
    for d in (0..axis).rev() {
        result[d] = remaining % shape[d];
        remaining /= shape[d];
    }
    result
}

/// Decompose a flat inner index into per-dimension indices for dimensions `axis+1..ndim`.
fn compute_inner_multi_index(shape: &[usize], axis: usize, flat: usize) -> Vec<usize> {
    let ndim = shape.len();
    let inner_dims = &shape[axis + 1..ndim];
    let mut result = vec![0usize; inner_dims.len()];
    let mut remaining = flat;
    for d in (0..inner_dims.len()).rev() {
        result[d] = remaining % inner_dims[d];
        remaining /= inner_dims[d];
    }
    result
}

// ---------------------------------------------------------------------------
// masked_select
// ---------------------------------------------------------------------------

/// Return all elements of `input` where the corresponding `mask` element is `true`.
///
/// The result is always a 1-D array.  `input` and `mask` must have the same shape.
///
/// # Errors
/// - [`GatherScatterError::ShapeMismatch`] if shapes differ
pub fn masked_select(
    input: &ArrayD<f64>,
    mask: &ArrayD<bool>,
) -> Result<Array1<f64>, GatherScatterError> {
    if input.shape() != mask.shape() {
        return Err(GatherScatterError::ShapeMismatch {
            expected: input.shape().to_vec(),
            got: mask.shape().to_vec(),
        });
    }

    let selected: Vec<f64> = input
        .iter()
        .zip(mask.iter())
        .filter_map(|(&v, &m)| if m { Some(v) } else { None })
        .collect();

    Ok(Array1::from(selected))
}

// ---------------------------------------------------------------------------
// masked_fill
// ---------------------------------------------------------------------------

/// Return a copy of `input` with masked positions replaced by `fill_value`.
///
/// `input` and `mask` must have the same shape.
///
/// # Errors
/// - [`GatherScatterError::ShapeMismatch`] if shapes differ
pub fn masked_fill(
    input: &ArrayD<f64>,
    mask: &ArrayD<bool>,
    fill_value: f64,
) -> Result<ArrayD<f64>, GatherScatterError> {
    if input.shape() != mask.shape() {
        return Err(GatherScatterError::ShapeMismatch {
            expected: input.shape().to_vec(),
            got: mask.shape().to_vec(),
        });
    }

    let data: Vec<f64> = input
        .iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m { fill_value } else { v })
        .collect();

    Array::from_shape_vec(IxDyn(input.shape()), data).map_err(|_| {
        GatherScatterError::ShapeMismatch {
            expected: input.shape().to_vec(),
            got: vec![],
        }
    })
}

// ---------------------------------------------------------------------------
// IndexStats
// ---------------------------------------------------------------------------

/// Statistics about an array of indices.
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Minimum index value (raw, may be negative before normalisation).
    pub min_index: i64,
    /// Maximum index value (raw, may be negative before normalisation).
    pub max_index: i64,
    /// Number of distinct index values.
    pub unique_indices: usize,
    /// Total number of index values (including duplicates).
    pub total_indices: usize,
    /// `true` if any index appears more than once.
    pub has_duplicates: bool,
    /// `true` if any index is negative.
    pub has_negatives: bool,
    /// `unique_indices / output_size` when `output_size` is provided, else `NaN`.
    pub coverage: f64,
}

impl IndexStats {
    /// Compute statistics for the given index slice.
    ///
    /// `output_size` is used to compute `coverage`; pass `None` if not applicable.
    pub fn compute(indices: &[i64], output_size: Option<usize>) -> Self {
        if indices.is_empty() {
            return IndexStats {
                min_index: 0,
                max_index: 0,
                unique_indices: 0,
                total_indices: 0,
                has_duplicates: false,
                has_negatives: false,
                coverage: f64::NAN,
            };
        }

        let mut min_index = indices[0];
        let mut max_index = indices[0];
        let mut has_negatives = false;
        let mut unique_set = HashSet::new();

        for &idx in indices {
            if idx < min_index {
                min_index = idx;
            }
            if idx > max_index {
                max_index = idx;
            }
            if idx < 0 {
                has_negatives = true;
            }
            unique_set.insert(idx);
        }

        let unique_indices = unique_set.len();
        let total_indices = indices.len();
        let has_duplicates = unique_indices < total_indices;

        let coverage = match output_size {
            Some(sz) if sz > 0 => unique_indices as f64 / sz as f64,
            _ => f64::NAN,
        };

        IndexStats {
            min_index,
            max_index,
            unique_indices,
            total_indices,
            has_duplicates,
            has_negatives,
            coverage,
        }
    }

    /// Return `true` if the indices form a permutation of `0..size` (each value
    /// appears exactly once, no value is missing).
    pub fn is_permutation(&self, size: usize) -> bool {
        if self.total_indices != size || self.has_duplicates || self.has_negatives {
            return false;
        }
        if size == 0 {
            return true;
        }
        self.min_index == 0 && self.max_index == (size as i64 - 1)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array, IxDyn};

    // Helper: construct a 2-D ArrayD<f64> from a Vec<Vec<f64>>.
    fn make_2d(rows: Vec<Vec<f64>>) -> ArrayD<f64> {
        let nrows = rows.len();
        let ncols = rows[0].len();
        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        Array::from_shape_vec(IxDyn(&[nrows, ncols]), flat).expect("make_2d: shape mismatch")
    }

    // Helper: construct a 1-D ArrayD<f64>.
    fn make_1d(data: Vec<f64>) -> ArrayD<f64> {
        let n = data.len();
        Array::from_shape_vec(IxDyn(&[n]), data).expect("make_1d")
    }

    // -----------------------------------------------------------------------
    // gather
    // -----------------------------------------------------------------------

    #[test]
    fn test_gather_axis0_basic() {
        // shape [4, 3], gather rows 1 and 3
        let input = make_2d(vec![
            vec![0.0, 1.0, 2.0],
            vec![3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0],
        ]);
        let result = gather(&input, &[1, 3], 0).expect("gather axis0");
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[[0, 0]], 3.0);
        assert_eq!(result[[0, 2]], 5.0);
        assert_eq!(result[[1, 0]], 9.0);
        assert_eq!(result[[1, 2]], 11.0);
    }

    #[test]
    fn test_gather_axis1_basic() {
        // shape [3, 5], gather columns 0, 2, 4
        let input = make_2d(vec![
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0, 9.0],
            vec![10.0, 11.0, 12.0, 13.0, 14.0],
        ]);
        let result = gather(&input, &[0, 2, 4], 1).expect("gather axis1");
        assert_eq!(result.shape(), &[3, 3]);
        assert_eq!(result[[0, 1]], 2.0);
        assert_eq!(result[[2, 2]], 14.0);
    }

    #[test]
    fn test_gather_negative_index() {
        // −1 should select the last row (index 3 in a 4-row tensor).
        let input = make_2d(vec![
            vec![0.0, 1.0],
            vec![2.0, 3.0],
            vec![4.0, 5.0],
            vec![6.0, 7.0],
        ]);
        let result = gather(&input, &[-1], 0).expect("gather negative");
        assert_eq!(result.shape(), &[1, 2]);
        assert_eq!(result[[0, 0]], 6.0);
        assert_eq!(result[[0, 1]], 7.0);
    }

    #[test]
    fn test_gather_out_of_bounds() {
        let input = make_2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let err = gather(&input, &[5], 0).unwrap_err();
        matches!(err, GatherScatterError::OutOfBoundsIndex { .. });
    }

    // -----------------------------------------------------------------------
    // gather_nd
    // -----------------------------------------------------------------------

    #[test]
    fn test_gather_nd_basic() {
        // input shape [3, 4]; indices shape [3, 4] with axis=1.
        // indices[i, j] selects the column for each row.
        let input = make_2d(vec![
            vec![10.0, 20.0, 30.0, 40.0],
            vec![50.0, 60.0, 70.0, 80.0],
            vec![90.0, 100.0, 110.0, 120.0],
        ]);
        // Each row picks a single column index (broadcast via index tensor).
        let idx_data: Vec<i64> = vec![3, 2, 1, 0, 0, 1, 2, 3, 2, 2, 2, 2];
        let indices = Array::from_shape_vec(IxDyn(&[3, 4]), idx_data).unwrap();
        let result = gather_nd(&input, &indices, 1).expect("gather_nd");
        assert_eq!(result.shape(), &[3, 4]);
        // input[0][3]=40, input[0][2]=30, …
        assert_eq!(result[[0, 0]], 40.0);
        assert_eq!(result[[0, 1]], 30.0);
        assert_eq!(result[[2, 0]], 110.0);
    }

    // -----------------------------------------------------------------------
    // scatter_add
    // -----------------------------------------------------------------------

    #[test]
    fn test_scatter_add_basic() {
        // input shape [3, 5], indices [0, 2, 1], axis=0, output_size=4
        let input = make_2d(vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0, 9.0, 10.0],
            vec![11.0, 12.0, 13.0, 14.0, 15.0],
        ]);
        let result = scatter_add(&input, &[0, 2, 1], 0, 4, 0.0).expect("scatter_add basic");
        assert_eq!(result.shape(), &[4, 5]);
        // row 0 ← input row 0
        assert_eq!(result[[0, 0]], 1.0);
        // row 1 ← input row 2
        assert_eq!(result[[1, 0]], 11.0);
        // row 2 ← input row 1
        assert_eq!(result[[2, 0]], 6.0);
        // row 3 ← nothing → 0.0
        assert_eq!(result[[3, 0]], 0.0);
    }

    #[test]
    fn test_scatter_add_duplicate_indices() {
        // Two input rows map to the same output row → values should sum.
        let input = make_2d(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        let result = scatter_add(&input, &[0, 0, 1], 0, 2, 0.0).expect("scatter_add dup");
        assert_eq!(result.shape(), &[2, 2]);
        // row 0 = row0 + row1 = [4, 6]
        assert_eq!(result[[0, 0]], 4.0);
        assert_eq!(result[[0, 1]], 6.0);
        // row 1 = row2 = [5, 6]
        assert_eq!(result[[1, 0]], 5.0);
    }

    #[test]
    fn test_scatter_add_shape() {
        let input = make_1d(vec![1.0, 2.0, 3.0]);
        let result = scatter_add(&input, &[0, 2, 4], 0, 6, 0.0).expect("scatter_add shape");
        assert_eq!(result.shape(), &[6]);
    }

    // -----------------------------------------------------------------------
    // scatter_max
    // -----------------------------------------------------------------------

    #[test]
    fn test_scatter_max_basic() {
        let input = make_2d(vec![vec![5.0, 1.0], vec![3.0, 9.0], vec![7.0, 2.0]]);
        // indices [0, 0, 1], output_size=2
        let result = scatter_max(&input, &[0, 0, 1], 0, 2, f64::NEG_INFINITY).expect("scatter_max");
        // row0 = max([5,1],[3,9]) = [5,9]
        assert_eq!(result[[0, 0]], 5.0);
        assert_eq!(result[[0, 1]], 9.0);
        // row1 = [7,2]
        assert_eq!(result[[1, 0]], 7.0);
        assert_eq!(result[[1, 1]], 2.0);
    }

    // -----------------------------------------------------------------------
    // scatter_min
    // -----------------------------------------------------------------------

    #[test]
    fn test_scatter_min_basic() {
        let input = make_2d(vec![vec![5.0, 1.0], vec![3.0, 9.0], vec![7.0, 2.0]]);
        let result = scatter_min(&input, &[0, 0, 1], 0, 2, f64::INFINITY).expect("scatter_min");
        // row0 = min([5,1],[3,9]) = [3,1]
        assert_eq!(result[[0, 0]], 3.0);
        assert_eq!(result[[0, 1]], 1.0);
        // row1 = [7,2]
        assert_eq!(result[[1, 0]], 7.0);
    }

    // -----------------------------------------------------------------------
    // top_k
    // -----------------------------------------------------------------------

    #[test]
    fn test_top_k_largest() {
        // 1-D: [3,1,4,1,5,9] → top-2 = [9,5]
        let input = make_1d(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
        let (vals, idxs) = top_k(&input, 2, 0, true).expect("top_k largest");
        assert_eq!(vals.shape(), &[2]);
        assert_eq!(vals[[0]], 9.0);
        assert_eq!(vals[[1]], 5.0);
        assert_eq!(idxs[[0]], 5); // original index of 9
        assert_eq!(idxs[[1]], 4); // original index of 5
    }

    #[test]
    fn test_top_k_smallest() {
        // 1-D: [3,1,4,1,5,9] → bottom-2 = [1,1]
        let input = make_1d(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
        let (vals, _idxs) = top_k(&input, 2, 0, false).expect("top_k smallest");
        assert_eq!(vals.shape(), &[2]);
        assert_eq!(vals[[0]], 1.0);
        assert_eq!(vals[[1]], 1.0);
    }

    #[test]
    fn test_top_k_k_larger_than_dim() {
        let input = make_1d(vec![1.0, 2.0]);
        let err = top_k(&input, 5, 0, true).unwrap_err();
        matches!(err, GatherScatterError::KTooLarge { .. });
    }

    // -----------------------------------------------------------------------
    // masked_select
    // -----------------------------------------------------------------------

    #[test]
    fn test_masked_select_basic() {
        let input = make_2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let mask_data = vec![true, false, true, false, true, false];
        let mask = Array::from_shape_vec(IxDyn(&[2, 3]), mask_data).expect("mask shape");
        let result = masked_select(&input, &mask).expect("masked_select");
        assert_eq!(result.len(), 3);
        // Elements at positions (0,0)=1, (0,2)=3, (1,1)=5
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 3.0);
        assert_eq!(result[2], 5.0);
    }

    #[test]
    fn test_masked_select_all_false() {
        let input = make_1d(vec![1.0, 2.0, 3.0]);
        let mask = Array::from_shape_vec(IxDyn(&[3]), vec![false, false, false]).expect("mask");
        let result = masked_select(&input, &mask).expect("masked_select all_false");
        assert_eq!(result.len(), 0);
    }

    // -----------------------------------------------------------------------
    // masked_fill
    // -----------------------------------------------------------------------

    #[test]
    fn test_masked_fill_basic() {
        let input = make_1d(vec![1.0, 2.0, 3.0, 4.0]);
        let mask =
            Array::from_shape_vec(IxDyn(&[4]), vec![false, true, false, true]).expect("mask");
        let result = masked_fill(&input, &mask, -99.0).expect("masked_fill");
        assert_eq!(result.shape(), &[4]);
        assert_eq!(result[[0]], 1.0);
        assert_eq!(result[[1]], -99.0);
        assert_eq!(result[[2]], 3.0);
        assert_eq!(result[[3]], -99.0);
    }

    #[test]
    fn test_masked_fill_shape_mismatch() {
        let input = make_1d(vec![1.0, 2.0, 3.0]);
        let mask = Array::from_shape_vec(IxDyn(&[2]), vec![true, false]).expect("mask");
        let err = masked_fill(&input, &mask, 0.0).unwrap_err();
        matches!(err, GatherScatterError::ShapeMismatch { .. });
    }

    // -----------------------------------------------------------------------
    // IndexStats
    // -----------------------------------------------------------------------

    #[test]
    fn test_index_stats_basic() {
        let indices: Vec<i64> = vec![0, 2, 4, 2, -1];
        let stats = IndexStats::compute(&indices, Some(5));
        assert_eq!(stats.min_index, -1);
        assert_eq!(stats.max_index, 4);
        assert_eq!(stats.total_indices, 5);
        assert_eq!(stats.unique_indices, 4); // {-1,0,2,4}
        assert!(stats.has_duplicates);
        assert!(stats.has_negatives);
        // coverage = 4/5
        assert!((stats.coverage - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_index_stats_is_permutation_true() {
        let indices: Vec<i64> = vec![0, 1, 2, 3];
        let stats = IndexStats::compute(&indices, Some(4));
        assert!(stats.is_permutation(4));
    }

    #[test]
    fn test_index_stats_is_permutation_false() {
        // [0, 0, 1] — duplicate, not a permutation.
        let indices: Vec<i64> = vec![0, 0, 1];
        let stats = IndexStats::compute(&indices, Some(3));
        assert!(!stats.is_permutation(3));
    }
}
