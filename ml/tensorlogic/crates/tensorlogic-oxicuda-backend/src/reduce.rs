//! Per-axis reduction for the OxiCUDA backend.
//!
//! # Implementation strategy
//!
//! This module implements per-axis reduction via the native GPU kernel
//! `oxicuda_blas::reduction::reduce_axis`. The tensor is viewed as
//! `[outer, axis_len, inner]` (contiguous, row-major), and the kernel
//! reduces along `axis_len` in a single GPU launch — no CPU transpose
//! or per-row loop required.
//!
//! ## Dimension decomposition
//!
//! Given a tensor with `shape` and a reduction `axis`:
//! - `outer = product(shape[0..axis])` — number of "outer" slices.
//! - `axis_len = shape[axis]` — the dimension being reduced.
//! - `inner = product(shape[axis+1..])` — number of "inner" elements per slice.
//!
//! The output shape is `shape` with `axis` removed: `shape[..axis] ++ shape[axis+1..]`.
//!
//! ## All reduction ops are supported
//!
//! `Sum`, `Max`, `Min`, `Mean`, and `Product` are all dispatched through
//! `oxicuda_blas::reduction::reduce_axis` which provides native GPU kernels
//! for every `ReductionOp` variant.
//!
//! ## Multi-axis reduction
//!
//! When `axes` contains more than one axis the reductions are applied sequentially,
//! largest axis first, re-indexing dimensions after each pass. When `axes` is empty
//! the input tensor is returned unchanged (identity).

#[cfg(feature = "gpu")]
use tensorlogic_infer::ReduceOp;

#[cfg(feature = "gpu")]
use crate::error::OxiCudaBackendError;
#[cfg(feature = "gpu")]
use crate::executor::OxiCudaTensor;

#[cfg(feature = "gpu")]
use oxicuda_blas::handle::BlasHandle;
#[cfg(feature = "gpu")]
use oxicuda_blas::reduction;
#[cfg(feature = "gpu")]
use oxicuda_memory::DeviceBuffer;

// ---------------------------------------------------------------------------
// GPU per-axis reduction
// ---------------------------------------------------------------------------

/// Reduce `tensor` along one axis using the given `op`.
///
/// Returns a tensor with `axis` removed from the shape.
///
/// Uses `oxicuda_blas::reduction::reduce_axis` — a native GPU per-axis
/// kernel. No CPU transpose or per-row loop.
#[cfg(feature = "gpu")]
fn reduce_one_axis(
    handle: &BlasHandle,
    op: ReduceOp,
    tensor: &OxiCudaTensor,
    axis: usize,
) -> Result<OxiCudaTensor, OxiCudaBackendError> {
    let ndim = tensor.shape.len();
    if ndim == 0 {
        return Err(OxiCudaBackendError::InvalidShape(
            "reduce on a scalar (0-dim) tensor is not supported".into(),
        ));
    }
    if axis >= ndim {
        return Err(OxiCudaBackendError::InvalidShape(format!(
            "axis {axis} is out of range for tensor with {ndim} dimensions"
        )));
    }

    let axis_len = tensor.shape[axis];
    if axis_len == 0 {
        return Err(OxiCudaBackendError::InvalidShape(format!(
            "cannot reduce along axis {axis} with length 0"
        )));
    }

    // Decompose shape into (outer, axis_len, inner).
    let outer: usize = tensor.shape[..axis].iter().product();
    let inner: usize = tensor.shape[axis + 1..].iter().product();

    let outer_u32 = u32::try_from(outer).map_err(|_| {
        OxiCudaBackendError::DimensionOverflow(format!("outer={outer} exceeds u32::MAX"))
    })?;
    let axis_len_u32 = u32::try_from(axis_len).map_err(|_| {
        OxiCudaBackendError::DimensionOverflow(format!("axis_len={axis_len} exceeds u32::MAX"))
    })?;
    let inner_u32 = u32::try_from(inner).map_err(|_| {
        OxiCudaBackendError::DimensionOverflow(format!("inner={inner} exceeds u32::MAX"))
    })?;

    let blas_op = match op {
        ReduceOp::Sum => reduction::ReductionOp::Sum,
        ReduceOp::Max => reduction::ReductionOp::Max,
        ReduceOp::Min => reduction::ReductionOp::Min,
        ReduceOp::Mean => reduction::ReductionOp::Mean,
        ReduceOp::Product => reduction::ReductionOp::Product,
    };

    let d_input = DeviceBuffer::<f32>::from_host(&tensor.data)?;

    // Allocate at least 1 element to avoid zero-length device buffer.
    let out_elem_count = outer.saturating_mul(inner);
    let alloc_count = out_elem_count.max(1);
    let mut d_output = DeviceBuffer::<f32>::zeroed(alloc_count)?;

    reduction::reduce_axis(
        handle,
        blas_op,
        outer_u32,
        axis_len_u32,
        inner_u32,
        &d_input,
        &mut d_output,
    )?;

    handle.stream().synchronize()?;

    // Copy exactly outer*inner elements back; guard against empty output.
    let mut host_out = vec![0.0_f32; out_elem_count];
    if out_elem_count > 0 {
        d_output.copy_to_host(&mut host_out)?;
    }

    let out_shape: Vec<usize> = tensor.shape[..axis]
        .iter()
        .chain(tensor.shape[axis + 1..].iter())
        .copied()
        .collect();

    OxiCudaTensor::new(out_shape, host_out)
}

/// Reduce a tensor along one or more axes.
///
/// - If `axes` is empty, returns a clone of the input (identity).
/// - If `axes` contains duplicates or out-of-range values, returns an error.
/// - Axes are processed from largest to smallest so that earlier axis indices
///   remain valid after each reduction shrinks the shape.
#[cfg(feature = "gpu")]
pub fn dispatch_reduce(
    handle: &BlasHandle,
    op: ReduceOp,
    x: &OxiCudaTensor,
    axes: &[usize],
) -> Result<OxiCudaTensor, OxiCudaBackendError> {
    if axes.is_empty() {
        return Ok(x.clone());
    }

    // Validate and sort axes; remove duplicates.
    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort_unstable();
    sorted_axes.dedup();

    // Validate all axes are in range.
    let ndim = x.shape.len();
    for &a in &sorted_axes {
        if a >= ndim {
            return Err(OxiCudaBackendError::InvalidShape(format!(
                "axis {a} is out of range for tensor with {ndim} dimensions"
            )));
        }
    }

    let mut current = x.clone();

    // Apply axes in reverse (largest first) so earlier indices stay valid
    // after each reduction shrinks the shape.
    let mut axes_desc = sorted_axes;
    axes_desc.reverse();

    for axis in axes_desc {
        current = reduce_one_axis(handle, op, &current, axis)?;
    }

    Ok(current)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::error::OxiCudaBackendError;

    /// Helper: compute (outer, inner, out_shape) for a given shape and axis —
    /// mirrors the decomposition used in `reduce_one_axis`.
    fn decompose(shape: &[usize], axis: usize) -> (usize, usize, Vec<usize>) {
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();
        let out_shape: Vec<usize> = shape[..axis]
            .iter()
            .chain(shape[axis + 1..].iter())
            .copied()
            .collect();
        (outer, inner, out_shape)
    }

    #[test]
    fn decompose_axis_shape_2d_axis0() {
        // shape [3, 4], axis 0 → outer=1, axis_len=3, inner=4, out_shape=[4]
        let (outer, inner, out_shape) = decompose(&[3, 4], 0);
        assert_eq!(outer, 1);
        assert_eq!(inner, 4);
        assert_eq!(out_shape, vec![4]);
    }

    #[test]
    fn decompose_axis_shape_2d_axis1() {
        // shape [3, 4], axis 1 → outer=3, inner=1, out_shape=[3]
        let (outer, inner, out_shape) = decompose(&[3, 4], 1);
        assert_eq!(outer, 3);
        assert_eq!(inner, 1);
        assert_eq!(out_shape, vec![3]);
    }

    #[test]
    fn decompose_axis_shape_3d_middle() {
        // shape [2, 3, 4], axis 1 → outer=2, axis_len=3, inner=4, out_shape=[2,4]
        let (outer, inner, out_shape) = decompose(&[2, 3, 4], 1);
        assert_eq!(outer, 2);
        assert_eq!(inner, 4);
        assert_eq!(out_shape, vec![2, 4]);
    }

    #[test]
    fn decompose_axis_shape_3d_first() {
        // shape [2, 3, 4], axis 0 → outer=1, axis_len=2, inner=12, out_shape=[3,4]
        let (outer, inner, out_shape) = decompose(&[2, 3, 4], 0);
        assert_eq!(outer, 1);
        assert_eq!(inner, 12);
        assert_eq!(out_shape, vec![3, 4]);
    }

    #[test]
    fn decompose_axis_shape_3d_last() {
        // shape [2, 3, 4], axis 2 → outer=6, axis_len=4, inner=1, out_shape=[2,3]
        let (outer, inner, out_shape) = decompose(&[2, 3, 4], 2);
        assert_eq!(outer, 6);
        assert_eq!(inner, 1);
        assert_eq!(out_shape, vec![2, 3]);
    }

    #[test]
    fn decompose_axis_1d() {
        // shape [5], axis 0 → outer=1, inner=1, out_shape=[]
        let (outer, inner, out_shape) = decompose(&[5], 0);
        assert_eq!(outer, 1);
        assert_eq!(inner, 1);
        assert_eq!(out_shape, Vec::<usize>::new());
    }

    #[test]
    fn unsupported_spec_display() {
        let e = OxiCudaBackendError::UnsupportedSpec("weird->spec".to_string());
        assert!(e.to_string().contains("weird->spec"));
    }
}
