//! [`OxiCudaExecutor`] -- GPU tensor executor built on the OxiCUDA stack.
//!
//! For MVP the type is always compiled so downstream code can name it
//! regardless of feature flags. When the `gpu` feature is off every
//! [`tensorlogic_infer::TlExecutor`] method returns
//! [`OxiCudaBackendError::BackendDisabled`]. When `gpu` is on, the executor
//! dispatches:
//! - `einsum` → `crate::einsum::dispatch_einsum` (supports matmul 2D, batched
//!   3D, and identity specs; everything else returns `UnsupportedSpec`).
//! - `elem_op` → `crate::elem_ops::dispatch_unary` (Relu, Sigmoid supported).
//! - `elem_op_binary` → `crate::elem_ops::dispatch_binary` (Add, Multiply).
//! - `reduce` → `crate::reduce::dispatch_reduce` (Sum, Max, Min, Mean per
//!   axis; Product returns Unsupported).

use tensorlogic_infer::{ElemOp, ReduceOp, TlExecutor};

use crate::error::OxiCudaBackendError;

#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use oxicuda_blas::handle::BlasHandle;
#[cfg(feature = "gpu")]
use oxicuda_driver::{Context, Device, Stream};

/// A host-resident tensor handle used by the MVP scaffold.
///
/// The real implementation will eventually carry a device pointer and a
/// shape; for now the shape and a flat `f32` buffer are enough to exercise
/// the public API and unblock downstream wiring. Keeping the type
/// feature-independent means the crate exports the same public surface
/// with or without `--features gpu`.
#[derive(Clone, Debug, PartialEq)]
pub struct OxiCudaTensor {
    /// Row-major shape of the tensor (e.g. `[m, k]` for a 2D matrix).
    pub shape: Vec<usize>,
    /// Flat row-major `f32` buffer. Length must equal the product of `shape`.
    pub data: Vec<f32>,
}

impl OxiCudaTensor {
    /// Construct a new tensor, validating that the buffer length matches the shape.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, OxiCudaBackendError> {
        let expected: usize = shape.iter().copied().product();
        if expected != data.len() {
            return Err(OxiCudaBackendError::InvalidShape(format!(
                "shape {:?} implies {expected} elements, got buffer of length {}",
                shape,
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }
}

// ---- GpuState (gpu feature only) ----

/// Opaque handle to the live GPU context and BLAS state.
///
/// Obtained via [`OxiCudaExecutor::gpu_state`].  Passed to GPU sub-features
/// such as `fft::forward_c2c_1d`; do not construct directly.
#[cfg(feature = "gpu")]
pub struct GpuState {
    context: Arc<Context>,
    blas_handle: BlasHandle,
}

#[cfg(feature = "gpu")]
impl std::fmt::Debug for GpuState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuState")
            .field("context", &self.context)
            .field("blas_handle", &"<oxicuda BlasHandle>")
            .finish()
    }
}

#[cfg(feature = "gpu")]
impl GpuState {
    fn new() -> Result<Self, OxiCudaBackendError> {
        oxicuda_driver::init()?;
        let dev = Device::get(0)?;
        let context = Arc::new(Context::new(&dev)?);
        let stream = Stream::new(&context)?;
        let blas_handle = BlasHandle::with_stream(&context, stream)?;
        Ok(Self {
            context,
            blas_handle,
        })
    }

    /// Returns a reference to the CUDA context owned by this state.
    ///
    /// Only available when the `fft` feature is enabled (its only caller).
    #[cfg(feature = "fft")]
    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Returns a reference to the BLAS handle owned by this state.
    ///
    /// Available for all `gpu`-feature callers within this crate.
    /// Used by `autodiff.rs` for gradient GEMM calls and by the
    /// `native-broadcast` feature for fill/broadcast_axes kernels.
    pub(crate) fn blas_handle(&self) -> &BlasHandle {
        &self.blas_handle
    }
}

/// OxiCUDA-backed tensor executor.
///
/// Construct via [`OxiCudaExecutor::new`]. When the crate is built without
/// the `gpu` feature every method returns
/// [`OxiCudaBackendError::BackendDisabled`].
#[derive(Debug)]
pub struct OxiCudaExecutor {
    #[cfg(feature = "gpu")]
    pub(crate) gpu: GpuState,
}

impl OxiCudaExecutor {
    /// Create a new executor.
    ///
    /// With the `gpu` feature enabled this initializes the CUDA driver,
    /// selects device 0, creates a context, stream, and BLAS handle.
    ///
    /// When the `gpu` feature is *not* enabled this returns
    /// [`OxiCudaBackendError::BackendDisabled`].
    #[cfg(feature = "gpu")]
    pub fn new() -> Result<Self, OxiCudaBackendError> {
        let gpu = GpuState::new()?;
        Ok(Self { gpu })
    }

    /// Stub constructor when the `gpu` feature is disabled.
    #[cfg(not(feature = "gpu"))]
    pub fn new() -> Result<Self, OxiCudaBackendError> {
        Err(OxiCudaBackendError::BackendDisabled)
    }

    /// Returns a reference to the internal [`GpuState`] for FFT operations.
    ///
    /// Returns [`OxiCudaBackendError::BackendDisabled`] when the executor
    /// was created without the `gpu` feature enabled.
    #[cfg(feature = "gpu")]
    pub fn gpu_state(&self) -> Result<&GpuState, OxiCudaBackendError> {
        Ok(&self.gpu)
    }

    /// Returns a crate-internal reference to the live [`GpuState`].
    ///
    /// Used by the `native-broadcast` autodiff helpers to access the BLAS
    /// handle directly from within the crate (e.g. `autodiff.rs`).  Not
    /// part of the public API.
    #[cfg(feature = "native-broadcast")]
    pub(crate) fn gpu_state_internal(&self) -> &GpuState {
        &self.gpu
    }
}

// ---- TlExecutor impl -- without `gpu` feature (pure-Rust stub) ----

#[cfg(not(feature = "gpu"))]
impl TlExecutor for OxiCudaExecutor {
    type Tensor = OxiCudaTensor;
    type Error = OxiCudaBackendError;

    fn einsum(
        &mut self,
        _spec: &str,
        _inputs: &[Self::Tensor],
    ) -> Result<Self::Tensor, Self::Error> {
        Err(OxiCudaBackendError::BackendDisabled)
    }

    fn elem_op(&mut self, _op: ElemOp, _x: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        Err(OxiCudaBackendError::BackendDisabled)
    }

    fn elem_op_binary(
        &mut self,
        _op: ElemOp,
        _x: &Self::Tensor,
        _y: &Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error> {
        Err(OxiCudaBackendError::BackendDisabled)
    }

    fn reduce(
        &mut self,
        _op: ReduceOp,
        _x: &Self::Tensor,
        _axes: &[usize],
    ) -> Result<Self::Tensor, Self::Error> {
        Err(OxiCudaBackendError::BackendDisabled)
    }
}

// ---- TlExecutor impl -- with `gpu` feature (OxiCUDA-backed) ----

#[cfg(feature = "gpu")]
impl TlExecutor for OxiCudaExecutor {
    type Tensor = OxiCudaTensor;
    type Error = OxiCudaBackendError;

    fn einsum(&mut self, spec: &str, inputs: &[Self::Tensor]) -> Result<Self::Tensor, Self::Error> {
        crate::einsum::dispatch_einsum(&self.gpu.blas_handle, spec, inputs)
    }

    fn elem_op(&mut self, op: ElemOp, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        crate::elem_ops::dispatch_unary(&self.gpu.blas_handle, op, x)
    }

    fn elem_op_binary(
        &mut self,
        op: ElemOp,
        x: &Self::Tensor,
        y: &Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error> {
        crate::elem_ops::dispatch_binary(&self.gpu.blas_handle, op, x, y)
    }

    fn reduce(
        &mut self,
        op: ReduceOp,
        x: &Self::Tensor,
        axes: &[usize],
    ) -> Result<Self::Tensor, Self::Error> {
        crate::reduce::dispatch_reduce(&self.gpu.blas_handle, op, x, axes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_new_validates_shape() {
        let ok = OxiCudaTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        assert!(ok.is_ok());

        let bad = OxiCudaTensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0]);
        assert!(matches!(bad, Err(OxiCudaBackendError::InvalidShape(_))));
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn disabled_backend_rejects_new() {
        match OxiCudaExecutor::new() {
            Err(OxiCudaBackendError::BackendDisabled) => {}
            Err(other) => panic!("expected BackendDisabled, got {other:?}"),
            Ok(_) => panic!("expected BackendDisabled, got Ok"),
        }
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn disabled_error_message_contains_hint() {
        let e = OxiCudaBackendError::BackendDisabled;
        let msg = e.to_string();
        assert!(msg.contains("--features gpu"), "got: {msg}");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn matmul_rejects_bad_input_count() {
        let mut exec = match OxiCudaExecutor::new() {
            Ok(e) => e,
            Err(err) => {
                eprintln!("skipping (no GPU): {err}");
                return;
            }
        };
        let a = match OxiCudaTensor::new(vec![2, 3], vec![0.0; 6]) {
            Ok(t) => t,
            Err(err) => panic!("tensor build failed: {err}"),
        };
        let result = exec.einsum("ij,jk->ik", &[a]);
        assert!(matches!(
            result,
            Err(OxiCudaBackendError::InvalidEinsumSpec(_))
        ));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn matmul_rejects_inner_dim_mismatch() {
        let mut exec = match OxiCudaExecutor::new() {
            Ok(e) => e,
            Err(err) => {
                eprintln!("skipping (no GPU): {err}");
                return;
            }
        };
        let a = match OxiCudaTensor::new(vec![2, 3], vec![0.0; 6]) {
            Ok(t) => t,
            Err(err) => panic!("tensor build failed: {err}"),
        };
        let b = match OxiCudaTensor::new(vec![4, 5], vec![0.0; 20]) {
            Ok(t) => t,
            Err(err) => panic!("tensor build failed: {err}"),
        };
        let result = exec.einsum("ij,jk->ik", &[a, b]);
        assert!(matches!(result, Err(OxiCudaBackendError::InvalidShape(_))));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn unknown_spec_is_unsupported() {
        let mut exec = match OxiCudaExecutor::new() {
            Ok(e) => e,
            Err(err) => {
                eprintln!("skipping (no GPU): {err}");
                return;
            }
        };
        let a = match OxiCudaTensor::new(vec![2, 2], vec![0.0; 4]) {
            Ok(t) => t,
            Err(err) => panic!("tensor build failed: {err}"),
        };
        let result = exec.einsum("ii->", &[a]);
        assert!(matches!(
            result,
            Err(OxiCudaBackendError::UnsupportedSpec(_))
        ));
    }
}
