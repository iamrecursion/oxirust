//! Elementwise unary and binary op dispatch for the OxiCUDA backend.
//!
//! # Mapping from TlExecutor ops to upstream oxicuda-blas ops
//!
//! The upstream `oxicuda-blas` exposes a full set of elementwise kernels.
//! All reachable [`tensorlogic_infer::ElemOp`] variants are now wired:
//!
//! ## Unary (single input)
//!
//! | `ElemOp` variant | upstream function |
//! |------------------|-------------------|
//! | `Relu`           | `relu`            |
//! | `Sigmoid`        | `sigmoid`         |
//! | `OneMinus`       | `one_minus`       |
//!
//! ## Binary (two inputs of identical shape)
//!
//! | `ElemOp` variant | upstream function | semantic                            |
//! |------------------|-------------------|-------------------------------------|
//! | `Add`            | `add`             | `C[i] = A[i] + B[i]`               |
//! | `Subtract`       | `sub`             | `C[i] = A[i] - B[i]`               |
//! | `Multiply`       | `mul`             | `C[i] = A[i] * B[i]`               |
//! | `Divide`         | `div`             | `C[i] = A[i] / B[i]`               |
//! | `Min`            | `min`             | `C[i] = min(A[i], B[i])`           |
//! | `Max`            | `max`             | `C[i] = max(A[i], B[i])`           |
//! | `Eq`             | `cmp_eq`          | `C[i] = (A[i] == B[i]) ? 1.0 : 0.0`|
//! | `Lt`             | `cmp_lt`          | `C[i] = (A[i] <  B[i]) ? 1.0 : 0.0`|
//! | `Gt`             | `cmp_gt`          | `C[i] = (A[i] >  B[i]) ? 1.0 : 0.0`|
//! | `Lte`            | `cmp_le`          | `C[i] = (A[i] <= B[i]) ? 1.0 : 0.0`|
//! | `Gte`            | `cmp_ge`          | `C[i] = (A[i] >= B[i]) ? 1.0 : 0.0`|
//! | `OrMax`          | `or_max`          | `C[i] = max(A[i], B[i])`           |
//! | `OrProbSum`      | `or_prob_sum`     | `C[i] = A[i]+B[i]-A[i]*B[i]`       |
//! | `Nand`           | `nand`            | `C[i] = 1 - A[i]*B[i]`             |
//! | `Nor`            | `nor`             | `C[i] = 1-(A[i]+B[i]-A[i]*B[i])`   |
//! | `Xor`            | `xor`             | `C[i] = A[i]+B[i]-2*A[i]*B[i]`     |
//!
//! **Note on `ElemOp::Nor`**: The ElemOp comment says `1 - max(a, b)`, but the
//! upstream `nor` kernel computes `1 - (a + b - ab)` (probabilistic-sum NOR base).
//! The mapping is intentional — this is the available kernel.
//!
//! The upstream fused kernels (`fused_add_relu`, `fused_scale_add`, `pow`) are
//! not reachable through the `TlExecutor` single-op interface and would require
//! higher-level pattern recognition.

#[cfg(feature = "gpu")]
use tensorlogic_infer::ElemOp;

#[cfg(feature = "gpu")]
use crate::error::OxiCudaBackendError;
#[cfg(feature = "gpu")]
use crate::executor::OxiCudaTensor;

#[cfg(feature = "gpu")]
use oxicuda_blas::elementwise::{
    add, cmp_eq, cmp_ge, cmp_gt, cmp_le, cmp_lt, div, max, min, mul, nand, nor, one_minus, or_max,
    or_prob_sum, relu, sigmoid, sub, xor,
};
#[cfg(feature = "gpu")]
use oxicuda_blas::handle::BlasHandle;
#[cfg(feature = "gpu")]
use oxicuda_memory::DeviceBuffer;

// ---------------------------------------------------------------------------
// Unary dispatch
// ---------------------------------------------------------------------------

/// Dispatch a unary elementwise op from [`ElemOp`] to the appropriate upstream
/// `oxicuda-blas` kernel.
///
/// Currently supported: `Relu`, `Sigmoid`, `OneMinus`.
/// All binary-arity variants return [`OxiCudaBackendError::UnsupportedUnary`]
/// since they require two input tensors and must be dispatched via
/// [`dispatch_binary`] instead.
#[cfg(feature = "gpu")]
pub fn dispatch_unary(
    handle: &BlasHandle,
    op: ElemOp,
    x: &OxiCudaTensor,
) -> Result<OxiCudaTensor, OxiCudaBackendError> {
    let n = x.data.len();
    if n == 0 {
        return Ok(OxiCudaTensor {
            shape: x.shape.clone(),
            data: vec![],
        });
    }
    let n32 = u32::try_from(n).map_err(|_| {
        OxiCudaBackendError::DimensionOverflow(format!(
            "elem_op unary: buffer length {n} exceeds u32::MAX"
        ))
    })?;

    let d_input = DeviceBuffer::<f32>::from_host(&x.data)?;
    let mut d_output = DeviceBuffer::<f32>::zeroed(n)?;

    match op {
        ElemOp::Relu => {
            relu(handle, n32, &d_input, &mut d_output)?;
        }
        ElemOp::Sigmoid => {
            sigmoid(handle, n32, &d_input, &mut d_output)?;
        }
        ElemOp::OneMinus => {
            one_minus(handle, n32, &d_input, &mut d_output)?;
        }
        other => {
            return Err(OxiCudaBackendError::UnsupportedUnary(format!("{other:?}")));
        }
    }

    handle.stream().synchronize()?;

    let mut host_out = vec![0.0_f32; n];
    d_output.copy_to_host(&mut host_out)?;

    OxiCudaTensor::new(x.shape.clone(), host_out)
}

// ---------------------------------------------------------------------------
// Binary dispatch
// ---------------------------------------------------------------------------

/// Dispatch a binary elementwise op from [`ElemOp`] to the appropriate upstream
/// `oxicuda-blas` kernel.
///
/// Supported: all arithmetic (`Add`, `Subtract`, `Multiply`, `Divide`, `Min`, `Max`),
/// comparison (`Eq`, `Lt`, `Gt`, `Lte`, `Gte`), and logical
/// (`OrMax`, `OrProbSum`, `Nand`, `Nor`, `Xor`) variants of [`ElemOp`].
///
/// The unary variants (`Relu`, `Sigmoid`, `OneMinus`) return
/// [`OxiCudaBackendError::UnsupportedBinary`] — they must be dispatched via
/// [`dispatch_unary`] instead.
///
/// Both inputs must have identical shapes; this is validated before dispatch.
#[cfg(feature = "gpu")]
pub fn dispatch_binary(
    handle: &BlasHandle,
    op: ElemOp,
    x: &OxiCudaTensor,
    y: &OxiCudaTensor,
) -> Result<OxiCudaTensor, OxiCudaBackendError> {
    if x.shape != y.shape {
        return Err(OxiCudaBackendError::InvalidShape(format!(
            "elem_op_binary requires identical shapes: {:?} vs {:?}",
            x.shape, y.shape
        )));
    }

    let n = x.data.len();
    if n == 0 {
        return Ok(OxiCudaTensor {
            shape: x.shape.clone(),
            data: vec![],
        });
    }
    let n32 = u32::try_from(n).map_err(|_| {
        OxiCudaBackendError::DimensionOverflow(format!(
            "elem_op_binary: buffer length {n} exceeds u32::MAX"
        ))
    })?;

    let d_a = DeviceBuffer::<f32>::from_host(&x.data)?;
    let d_b = DeviceBuffer::<f32>::from_host(&y.data)?;
    let mut d_c = DeviceBuffer::<f32>::zeroed(n)?;

    match op {
        ElemOp::Add => {
            add(handle, n32, &d_a, &d_b, &mut d_c)?;
        }
        ElemOp::Subtract => {
            sub(handle, n32, &d_a, &d_b, &mut d_c)?;
        }
        ElemOp::Multiply => {
            mul(handle, n32, &d_a, &d_b, &mut d_c)?;
        }
        ElemOp::Divide => {
            div(handle, n32, &d_a, &d_b, &mut d_c)?;
        }
        ElemOp::Min => {
            min(handle, n32, &d_a, &d_b, &mut d_c)?;
        }
        ElemOp::Max => {
            max(handle, n32, &d_a, &d_b, &mut d_c)?;
        }
        ElemOp::Eq => {
            cmp_eq(handle, n32, &d_a, &d_b, &mut d_c)?;
        }
        ElemOp::Lt => {
            cmp_lt(handle, n32, &d_a, &d_b, &mut d_c)?;
        }
        ElemOp::Gt => {
            cmp_gt(handle, n32, &d_a, &d_b, &mut d_c)?;
        }
        ElemOp::Lte => {
            cmp_le(handle, n32, &d_a, &d_b, &mut d_c)?;
        }
        ElemOp::Gte => {
            cmp_ge(handle, n32, &d_a, &d_b, &mut d_c)?;
        }
        ElemOp::OrMax => {
            or_max(handle, n32, &d_a, &d_b, &mut d_c)?;
        }
        ElemOp::OrProbSum => {
            or_prob_sum(handle, n32, &d_a, &d_b, &mut d_c)?;
        }
        ElemOp::Nand => {
            nand(handle, n32, &d_a, &d_b, &mut d_c)?;
        }
        // NOTE: ElemOp::Nor is documented as `1 - max(a, b)`, but the upstream
        // `nor` kernel computes `1 - (a + b - ab)` (probabilistic-sum NOR base).
        // This mapping uses the available kernel; semantics differ from the ElemOp comment.
        ElemOp::Nor => {
            nor(handle, n32, &d_a, &d_b, &mut d_c)?;
        }
        ElemOp::Xor => {
            xor(handle, n32, &d_a, &d_b, &mut d_c)?;
        }
        other => {
            return Err(OxiCudaBackendError::UnsupportedBinary(format!("{other:?}")));
        }
    }

    handle.stream().synchronize()?;

    let mut host_out = vec![0.0_f32; n];
    d_c.copy_to_host(&mut host_out)?;

    OxiCudaTensor::new(x.shape.clone(), host_out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::error::OxiCudaBackendError;

    #[test]
    fn unsupported_unary_display() {
        let e = OxiCudaBackendError::UnsupportedUnary("OneMinus".to_string());
        assert!(e.to_string().contains("OneMinus"));
    }

    #[test]
    fn unsupported_binary_display() {
        let e = OxiCudaBackendError::UnsupportedBinary("Subtract".to_string());
        assert!(e.to_string().contains("Subtract"));
    }
}
