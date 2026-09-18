//! Error type for the OxiCUDA backend.

use thiserror::Error;

/// All errors returned by [`crate::OxiCudaExecutor`].
#[derive(Debug, Error)]
pub enum OxiCudaBackendError {
    /// The crate was built without the `gpu` feature, so the backend cannot run.
    #[error("OxiCUDA GPU backend not enabled — compile with --features gpu")]
    BackendDisabled,

    /// The requested operation is outside the MVP surface of this crate.
    #[error("unsupported operation for OxiCUDA MVP: {0}")]
    Unsupported(String),

    /// The einsum spec string was not recognised by the MVP parser.
    #[error("invalid einsum spec: {0}")]
    InvalidEinsumSpec(String),

    /// A tensor shape did not match what the operation expected.
    #[error("invalid tensor shape: {0}")]
    InvalidShape(String),

    /// An einsum spec string was not recognised or is not supported by this backend.
    #[error("unsupported einsum spec: {0}")]
    UnsupportedSpec(String),

    /// A unary elementwise operation is not supported by this backend.
    #[error("unsupported unary op: {0}")]
    UnsupportedUnary(String),

    /// A binary elementwise operation is not supported by this backend.
    #[error("unsupported binary op: {0}")]
    UnsupportedBinary(String),

    /// A runtime error was propagated from OxiCUDA itself.
    #[cfg(feature = "gpu")]
    #[error("OxiCUDA runtime error: {0}")]
    OxiCuda(String),

    /// A dimension exceeded `u32::MAX`, which the BLAS API requires.
    #[cfg(feature = "gpu")]
    #[error("dimension overflow: {0}")]
    DimensionOverflow(String),

    /// The FFT sub-feature is disabled; compile with `--features fft` (which implies `gpu`).
    #[error("OxiCUDA FFT not enabled — compile with --features fft")]
    FftDisabled,

    /// An OxiCUDA FFT operation failed.
    #[cfg(all(feature = "gpu", feature = "fft"))]
    #[error("OxiCUDA FFT error: {0}")]
    Fft(String),

    /// Attempted autodiff backward through an op with no gradient implementation.
    #[error("no autodiff gradient for op: {0}")]
    UnsupportedAutodiffOp(String),
}

#[cfg(feature = "gpu")]
impl From<oxicuda_driver::CudaError> for OxiCudaBackendError {
    fn from(err: oxicuda_driver::CudaError) -> Self {
        Self::OxiCuda(err.to_string())
    }
}

#[cfg(feature = "gpu")]
impl From<oxicuda_blas::BlasError> for OxiCudaBackendError {
    fn from(err: oxicuda_blas::BlasError) -> Self {
        Self::OxiCuda(err.to_string())
    }
}

#[cfg(all(feature = "gpu", feature = "fft"))]
impl From<oxicuda_fft::FftError> for OxiCudaBackendError {
    fn from(err: oxicuda_fft::FftError) -> Self {
        Self::Fft(err.to_string())
    }
}
