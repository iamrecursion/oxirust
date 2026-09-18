//! Error types for hardware acceleration.

use thiserror::Error;

/// Result type alias for acceleration operations.
pub type AccelResult<T> = Result<T, AccelError>;

/// Errors that can occur during hardware acceleration operations.
#[derive(Debug, Error)]
pub enum AccelError {
    /// Vulkan initialization failed.
    #[error("Vulkan initialization failed: {0}")]
    VulkanInit(String),

    /// No suitable GPU device found.
    #[error("No suitable GPU device found")]
    NoDevice,

    /// Device selection failed.
    #[error("Device selection failed: {0}")]
    DeviceSelection(String),

    /// Buffer allocation failed.
    #[error("Buffer allocation failed: {0}")]
    BufferAllocation(String),

    /// Buffer upload failed.
    #[error("Buffer upload to GPU failed: {0}")]
    BufferUpload(String),

    /// Buffer download failed.
    #[error("Buffer download from GPU failed: {0}")]
    BufferDownload(String),

    /// Shader compilation failed.
    #[error("Shader compilation failed: {0}")]
    ShaderCompilation(String),

    /// Compute pipeline creation failed.
    #[error("Compute pipeline creation failed: {0}")]
    PipelineCreation(String),

    /// Command buffer creation failed.
    #[error("Command buffer creation failed: {0}")]
    CommandBuffer(String),

    /// Compute dispatch failed.
    #[error("Compute dispatch failed: {0}")]
    Dispatch(String),

    /// Synchronization error.
    #[error("GPU synchronization error: {0}")]
    Synchronization(String),

    /// Invalid input dimensions.
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// Invalid pixel format.
    #[error("Invalid pixel format: {0}")]
    InvalidFormat(String),

    /// Unsupported operation.
    #[error("Unsupported operation: {0}")]
    Unsupported(String),

    /// Buffer size mismatch.
    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch {
        /// Expected buffer size.
        expected: usize,
        /// Actual buffer size.
        actual: usize,
    },

    /// Memory mapping failed.
    #[error("Memory mapping failed: {0}")]
    MemoryMap(String),

    /// Out of memory.
    #[error("Out of GPU memory")]
    OutOfMemory,

    /// Generic Vulkan error.
    #[error("Vulkan error: {0}")]
    Vulkan(String),

    /// Core library error.
    #[error("Core error: {0}")]
    Core(#[from] oximedia_core::OxiError),
}

#[cfg(feature = "vulkan-backend")]
impl From<vulkano::Validated<vulkano::VulkanError>> for AccelError {
    fn from(err: vulkano::Validated<vulkano::VulkanError>) -> Self {
        AccelError::Vulkan(format!("{err:?}"))
    }
}

#[cfg(feature = "vulkan-backend")]
impl From<vulkano::VulkanError> for AccelError {
    fn from(err: vulkano::VulkanError) -> Self {
        AccelError::Vulkan(format!("{err:?}"))
    }
}

#[cfg(feature = "vulkan-backend")]
impl From<vulkano::sync::HostAccessError> for AccelError {
    fn from(err: vulkano::sync::HostAccessError) -> Self {
        AccelError::Synchronization(format!("{err:?}"))
    }
}
