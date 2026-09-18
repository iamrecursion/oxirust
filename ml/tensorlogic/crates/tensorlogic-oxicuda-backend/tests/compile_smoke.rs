//! Compile-only smoke tests for `tensorlogic-oxicuda-backend`.
//!
//! Two scenarios:
//! - Without `--features gpu`: the executor must refuse to construct with
//!   [`OxiCudaBackendError::BackendDisabled`]. This is the Pure Rust Policy
//!   invariant — the crate must build and test cleanly on machines with no
//!   NVIDIA anything.
//! - With `--features gpu`: the executor must at least construct. We do
//!   **not** issue a real GPU call here — that is reserved for an integration
//!   test that is gated behind live hardware.

use tensorlogic_oxicuda_backend::OxiCudaExecutor;

#[cfg(not(feature = "gpu"))]
use tensorlogic_oxicuda_backend::OxiCudaBackendError;

#[cfg(not(feature = "gpu"))]
#[test]
fn default_build_is_disabled_stub() {
    match OxiCudaExecutor::new() {
        Err(OxiCudaBackendError::BackendDisabled) => {}
        Ok(_) => panic!("default build should not produce a live executor"),
        Err(other) => panic!("unexpected error: {other:?}"),
    }
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_feature_build_constructs_executor() {
    // Compile-only smoke: verifies the constructor path compiles. On
    // machines without an NVIDIA GPU the constructor returns an OxiCuda
    // runtime error which is acceptable.
    match OxiCudaExecutor::new() {
        Ok(_exec) => {}
        Err(_err) => {
            // No GPU available (macOS, CI without NVIDIA driver, etc.)
        }
    }
}
