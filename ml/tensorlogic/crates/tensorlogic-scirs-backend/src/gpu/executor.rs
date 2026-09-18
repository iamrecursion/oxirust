//! GPU backend trait, error types, and stub implementation.

use std::any::Any;

use thiserror::Error;

use crate::device::DeviceType;
use crate::gpu::device::GpuDevice;
use crate::gpu::kernel::{KernelConfig, KernelLaunchResult};
use crate::gpu::memory::GpuBuffer;

/// Errors that can occur during GPU operations.
#[derive(Debug, Error)]
pub enum GpuError {
    /// GPU is not available (no CUDA, stub mode, etc.).
    #[error("GPU not available: {reason}")]
    NotAvailable { reason: String },

    /// Requested device index does not exist.
    #[error("GPU device {index} not found")]
    DeviceNotFound { index: usize },

    /// Allocation would exceed available device memory.
    #[error("GPU out of memory: requested {requested} bytes, available {available}")]
    OutOfMemory { requested: usize, available: usize },

    /// Kernel failed to launch or execute.
    #[error("Kernel launch failed: {0}")]
    KernelLaunchFailed(String),

    /// Host-device or device-host data transfer failed.
    #[error("Data transfer failed: {0}")]
    TransferFailed(String),

    /// Device synchronization barrier failed.
    #[error("GPU synchronization failed: {0}")]
    SyncFailed(String),
}

/// Core trait that every GPU backend must implement.
///
/// The memory-management methods use raw bytes to remain dyn-compatible.
/// Typed wrappers (`allocate_typed`, `transfer_typed`) are provided as
/// free functions / extension methods outside the trait.
///
/// Implementors must be `Send` so that GPU operations can be dispatched
/// from worker threads.
pub trait GpuBackend: Send {
    /// Allocate `byte_count` bytes on the device, returning an opaque handle.
    fn allocate_raw(&mut self, byte_count: usize) -> Result<Box<dyn Any + Send>, GpuError>;

    /// Deallocate a buffer previously returned by `allocate_raw` or `transfer_to_device_raw`.
    fn deallocate_raw(&mut self, handle: Box<dyn Any + Send>) -> Result<(), GpuError>;

    /// Copy a host byte slice to device memory, returning an opaque handle.
    fn transfer_to_device_raw(&mut self, bytes: &[u8]) -> Result<Box<dyn Any + Send>, GpuError>;

    /// Copy `byte_count` bytes from a device handle back to a host `Vec<u8>`.
    fn transfer_to_host_raw(
        &self,
        handle: &dyn Any,
        byte_count: usize,
    ) -> Result<Vec<u8>, GpuError>;

    /// Launch a GPU kernel described by `config`.
    fn launch_kernel(&mut self, config: &KernelConfig) -> Result<KernelLaunchResult, GpuError>;

    /// Block until all previously issued operations on this backend have completed.
    fn synchronize(&mut self) -> Result<(), GpuError>;

    /// Return a reference to the device information for this backend.
    fn device_info(&self) -> &GpuDevice;

    /// Return `true` if this backend has a usable GPU runtime.
    fn is_available() -> bool
    where
        Self: Sized;
}

/// Typed helper: allocate a `GpuBuffer<T>` using `backend.allocate_raw`.
pub fn allocate<T: Send + 'static>(
    backend: &mut dyn GpuBackend,
    count: usize,
) -> Result<GpuBuffer<T>, GpuError> {
    let byte_count = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(GpuError::OutOfMemory {
            requested: usize::MAX,
            available: 0,
        })?;
    // We call allocate_raw only to propagate the error; the GpuBuffer is our
    // logical handle — the raw handle is immediately discarded in stub mode.
    backend.allocate_raw(byte_count)?;
    Ok(GpuBuffer::new(backend.device_info().index, byte_count))
}

/// Typed helper: copy a host slice to device memory.
pub fn transfer_to_device<T: Copy + Send + 'static>(
    backend: &mut dyn GpuBackend,
    host: &[T],
) -> Result<GpuBuffer<T>, GpuError> {
    // SAFETY: T: Copy ensures no destructors; byte_len is the exact byte size of the slice.
    let byte_len = std::mem::size_of_val(host);
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, byte_len) };
    backend.transfer_to_device_raw(bytes)?;
    Ok(GpuBuffer::new(backend.device_info().index, bytes.len()))
}

/// Typed helper: copy `len` elements from a device buffer back to a `Vec<T>`.
///
/// The `buf` parameter is used only for its metadata (device index, size).
/// The actual device handle must be passed separately as a `&dyn Any`.
pub fn transfer_to_host<T: Copy + Send + 'static>(
    backend: &dyn GpuBackend,
    _buf: &GpuBuffer<T>,
    len: usize,
    handle: &dyn Any,
) -> Result<Vec<T>, GpuError> {
    let byte_count = len
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(GpuError::OutOfMemory {
            requested: usize::MAX,
            available: 0,
        })?;
    // In stub mode transfer_to_host_raw will return NotAvailable.
    let raw_bytes = backend.transfer_to_host_raw(handle, byte_count)?;
    // Safety: raw_bytes came from device memory that was originally populated
    // by copying valid T values.  This path only executes when a real backend
    // returns data, so the reinterpretation is sound.
    let elem_size = std::mem::size_of::<T>();
    if elem_size == 0 {
        return Ok(vec![]);
    }
    let mut result: Vec<T> = Vec::with_capacity(raw_bytes.len() / elem_size);
    for chunk in raw_bytes.chunks_exact(elem_size) {
        let mut arr = vec![0u8; elem_size];
        arr.copy_from_slice(chunk);
        // SAFETY: We verified chunk length matches T's size.
        let value = unsafe { std::ptr::read(arr.as_ptr() as *const T) };
        result.push(value);
    }
    Ok(result)
}

/// A stub CUDA backend for use when no CUDA runtime is installed.
///
/// All operations return [`GpuError::NotAvailable`] except `device_info()`
/// and `is_available()`.
pub struct CudaStub {
    pseudo_device: GpuDevice,
}

impl CudaStub {
    /// Create a new stub backend for the given device index.
    pub fn new(device_index: usize) -> Self {
        let mut device = Self::pseudo_device();
        device.index = device_index;
        Self {
            pseudo_device: device,
        }
    }

    /// Build a placeholder [`GpuDevice`] describing the stub.
    pub fn pseudo_device() -> GpuDevice {
        GpuDevice {
            index: 0,
            device_type: DeviceType::Cuda,
            name: "CUDA Stub (not available)".to_string(),
            total_memory_bytes: 0,
            free_memory_bytes: 0,
            compute_capability: None,
            supports_fp16: false,
            supports_bf16: false,
        }
    }
}

impl GpuBackend for CudaStub {
    fn allocate_raw(&mut self, _byte_count: usize) -> Result<Box<dyn Any + Send>, GpuError> {
        Err(GpuError::NotAvailable {
            reason: "CUDA not available in stub mode".to_string(),
        })
    }

    fn deallocate_raw(&mut self, _handle: Box<dyn Any + Send>) -> Result<(), GpuError> {
        Err(GpuError::NotAvailable {
            reason: "CUDA not available in stub mode".to_string(),
        })
    }

    fn transfer_to_device_raw(&mut self, _bytes: &[u8]) -> Result<Box<dyn Any + Send>, GpuError> {
        Err(GpuError::NotAvailable {
            reason: "CUDA not available in stub mode".to_string(),
        })
    }

    fn transfer_to_host_raw(
        &self,
        _handle: &dyn Any,
        _byte_count: usize,
    ) -> Result<Vec<u8>, GpuError> {
        Err(GpuError::NotAvailable {
            reason: "CUDA not available in stub mode".to_string(),
        })
    }

    fn launch_kernel(&mut self, _config: &KernelConfig) -> Result<KernelLaunchResult, GpuError> {
        Err(GpuError::NotAvailable {
            reason: "CUDA not available in stub mode".to_string(),
        })
    }

    fn synchronize(&mut self) -> Result<(), GpuError> {
        Err(GpuError::NotAvailable {
            reason: "CUDA not available in stub mode".to_string(),
        })
    }

    fn device_info(&self) -> &GpuDevice {
        &self.pseudo_device
    }

    fn is_available() -> bool {
        false
    }
}

/// Create the best available GPU backend for the given device index.
///
/// Currently always returns a [`CudaStub`] wrapped in a `Box<dyn GpuBackend>`
/// because no real CUDA runtime is linked. When CUDA becomes available this
/// function should probe the runtime and return a real implementation.
pub fn create_gpu_backend(device_index: usize) -> Result<Box<dyn GpuBackend>, GpuError> {
    // In stub mode we always return a stub backend rather than an error so
    // callers can still introspect the device descriptor.
    Ok(Box::new(CudaStub::new(device_index)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceType;
    use crate::gpu::kernel::KernelConfig;

    #[test]
    fn test_cuda_stub_creation() {
        let stub = CudaStub::new(0);
        let info = stub.device_info();
        assert_eq!(info.index, 0);
        assert_eq!(info.device_type, DeviceType::Cuda);
        assert!(!info.name.is_empty());
    }

    #[test]
    fn test_cuda_stub_is_not_available() {
        assert!(!CudaStub::is_available());
    }

    #[test]
    fn test_allocate_returns_error() {
        let mut stub = CudaStub::new(0);
        let result = stub.allocate_raw(1024);
        assert!(result.is_err());
        match result {
            Err(GpuError::NotAvailable { .. }) => {}
            other => panic!("Expected NotAvailable, got {:?}", other),
        }
    }

    #[test]
    fn test_transfer_to_device_returns_error() {
        let mut stub = CudaStub::new(0);
        let data = vec![1.0_f32, 2.0, 3.0];
        let result = transfer_to_device(&mut stub, &data);
        assert!(result.is_err());
        match result {
            Err(GpuError::NotAvailable { .. }) => {}
            other => panic!("Expected NotAvailable, got {:?}", other),
        }
    }

    #[test]
    fn test_synchronize_returns_error() {
        let mut stub = CudaStub::new(0);
        let result = stub.synchronize();
        assert!(result.is_err());
        match result {
            Err(GpuError::NotAvailable { .. }) => {}
            other => panic!("Expected NotAvailable, got {:?}", other),
        }
    }

    #[test]
    fn test_error_messages_contain_reason() {
        let err = GpuError::NotAvailable {
            reason: "no CUDA driver found".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("no CUDA driver found"));

        let oom = GpuError::OutOfMemory {
            requested: 1024,
            available: 512,
        };
        let oom_msg = oom.to_string();
        assert!(oom_msg.contains("1024"));
        assert!(oom_msg.contains("512"));
    }

    #[test]
    fn test_create_gpu_backend_fails_gracefully() {
        // create_gpu_backend returns Ok(stub) — callers should then call is_available
        // or attempt an operation and handle the resulting NotAvailable error.
        let backend_result = create_gpu_backend(0);
        assert!(backend_result.is_ok());

        let mut backend = match backend_result {
            Ok(b) => b,
            Err(e) => panic!("create_gpu_backend should not fail: {e}"),
        };

        // Confirm operations on the returned backend give a well-formed error.
        let config = KernelConfig::new("noop");
        let launch_result = backend.launch_kernel(&config);
        assert!(launch_result.is_err());
    }
}
