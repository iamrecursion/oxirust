//! GPU infrastructure framework.
//!
//! This module provides a pure-Rust GPU abstraction layer with:
//! - Device enumeration and capability inspection (`device`)
//! - Memory buffer management and pool tracking (`memory`)
//! - Kernel configuration and launch results (`kernel`)
//! - Backend trait and stub implementation (`executor`)
//!
//! Enable with the `gpu` feature flag.

pub mod device;
pub mod executor;
pub mod kernel;
pub mod memory;

pub use device::{GpuDevice, GpuRequirement};
pub use executor::{create_gpu_backend, CudaStub, GpuBackend, GpuError};
pub use kernel::{KernelConfig, KernelLaunchResult};
pub use memory::{GpuBuffer, GpuMemoryPool};
