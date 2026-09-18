//! OxiCUDA GPU backend for TensorLogic.
//!
//! Pure-Rust CUDA execution backend built on the OxiCUDA ecosystem
//! (`oxicuda-backend`, `oxicuda-blas`). Requires only the NVIDIA driver at
//! runtime — no CUDA SDK, no `nvcc`, no C/C++ toolchain.
//!
//! Disabled by default. Enable with `--features gpu`.
//! Run `cargo bench --features gpu` for GPU vs CPU matmul performance comparison.
//!
//! # Example
//!
//! ```
//! use tensorlogic_oxicuda_backend::{OxiCudaBackendError, OxiCudaExecutor};
//!
//! // With default features (no `gpu`), construction returns BackendDisabled.
//! // With `gpu` on a machine without an NVIDIA GPU, returns OxiCuda runtime error.
//! match OxiCudaExecutor::new() {
//!     Ok(_exec) => {
//!         // gpu feature is enabled and GPU is available.
//!     }
//!     Err(OxiCudaBackendError::BackendDisabled) => {
//!         // pure-Rust default path; expected when `gpu` is off.
//!     }
//!     Err(_other) => {
//!         // GPU feature enabled but no GPU available (e.g. macOS, CI).
//!     }
//! }
//! ```

#![warn(missing_docs)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

pub mod autodiff;
pub mod einsum;
pub mod elem_ops;
pub mod error;
pub mod executor;
pub mod fft;
pub mod reduce;

pub use autodiff::OxiCudaTape;
pub use error::OxiCudaBackendError;
pub use executor::{OxiCudaExecutor, OxiCudaTensor};

#[cfg(feature = "gpu")]
pub use executor::GpuState;

#[cfg(all(feature = "gpu", feature = "fft"))]
pub use fft::{forward_c2c_1d, inverse_c2c_1d};
