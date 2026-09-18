//! GPU buffer management for efficient data transfer.

use crate::error::{AccelError, AccelResult};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer,
};
use vulkano::device::DeviceOwned;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::sync::GpuFuture;
use vulkano::DeviceSize;

use vulkano::device::Queue;

/// Buffer manager for GPU memory operations.
pub struct BufferManager {
    allocator: Arc<StandardMemoryAllocator>,
    command_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
}

impl BufferManager {
    /// Creates a new buffer manager.
    #[must_use]
    pub fn new(allocator: Arc<StandardMemoryAllocator>, queue: Arc<Queue>) -> Self {
        let command_allocator = Arc::new(StandardCommandBufferAllocator::new(
            allocator.device().clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        Self {
            allocator,
            command_allocator,
            queue,
        }
    }

    /// Creates a staging buffer (CPU-visible) for data upload.
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails.
    pub fn create_staging_buffer(&self, size: DeviceSize) -> AccelResult<Subbuffer<[u8]>> {
        Buffer::new_slice::<u8>(
            self.allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            size,
        )
        .map_err(|e| {
            AccelError::BufferAllocation(format!("Staging buffer allocation failed: {e:?}"))
        })
    }

    /// Creates a device-local buffer (GPU-only) for compute operations.
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails.
    pub fn create_device_buffer(
        &self,
        size: DeviceSize,
        usage: BufferUsage,
    ) -> AccelResult<Subbuffer<[u8]>> {
        Buffer::new_slice::<u8>(
            self.allocator.clone(),
            BufferCreateInfo {
                usage: usage | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            size,
        )
        .map_err(|e| {
            AccelError::BufferAllocation(format!("Device buffer allocation failed: {e:?}"))
        })
    }

    /// Creates a read-back buffer (GPU to CPU) for downloading results.
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails.
    pub fn create_readback_buffer(&self, size: DeviceSize) -> AccelResult<Subbuffer<[u8]>> {
        Buffer::new_slice::<u8>(
            self.allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            size,
        )
        .map_err(|e| {
            AccelError::BufferAllocation(format!("Readback buffer allocation failed: {e:?}"))
        })
    }

    /// Uploads data to a GPU buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Data size doesn't match buffer size
    /// - Memory mapping fails
    /// - Upload operation fails
    pub fn upload_data(&self, data: &[u8], dst_buffer: &Subbuffer<[u8]>) -> AccelResult<()> {
        if data.len() as DeviceSize != dst_buffer.size() {
            return Err(AccelError::BufferSizeMismatch {
                #[allow(clippy::cast_possible_truncation)]
                expected: { dst_buffer.size() as usize },
                actual: data.len(),
            });
        }

        let staging = self.create_staging_buffer(data.len() as DeviceSize)?;

        {
            let mut write = staging.write().map_err(|e| {
                AccelError::MemoryMap(format!("Failed to map staging buffer: {e:?}"))
            })?;
            write.copy_from_slice(data);
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_allocator.clone(),
            self.allocator.device().active_queue_family_indices()[0],
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| {
            AccelError::CommandBuffer(format!("Failed to create command buffer: {e:?}"))
        })?;

        builder
            .copy_buffer(CopyBufferInfo::buffers(staging.clone(), dst_buffer.clone()))
            .map_err(|e| {
                AccelError::BufferUpload(format!("Failed to record copy command: {e:?}"))
            })?;

        let command_buffer = builder.build().map_err(|e| {
            AccelError::CommandBuffer(format!("Failed to build command buffer: {e:?}"))
        })?;

        vulkano::sync::now(self.allocator.device().clone())
            .then_execute(self.queue.clone(), command_buffer)
            .map_err(|e| AccelError::BufferUpload(format!("Failed to execute upload: {e:?}")))?
            .then_signal_fence_and_flush()
            .map_err(|e| AccelError::BufferUpload(format!("Failed to flush upload: {e:?}")))?
            .wait(None)
            .map_err(|e| AccelError::Synchronization(format!("Upload sync failed: {e:?}")))?;

        Ok(())
    }

    /// Downloads data from a GPU buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Buffer size exceeds `usize::MAX`
    /// - Memory mapping fails
    /// - Download operation fails
    pub fn download_data(&self, src_buffer: &Subbuffer<[u8]>) -> AccelResult<Vec<u8>> {
        let size = src_buffer.size();
        let readback = self.create_readback_buffer(size)?;

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_allocator.clone(),
            self.allocator.device().active_queue_family_indices()[0],
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| {
            AccelError::CommandBuffer(format!("Failed to create command buffer: {e:?}"))
        })?;

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                src_buffer.clone(),
                readback.clone(),
            ))
            .map_err(|e| {
                AccelError::BufferDownload(format!("Failed to record copy command: {e:?}"))
            })?;

        let command_buffer = builder.build().map_err(|e| {
            AccelError::CommandBuffer(format!("Failed to build command buffer: {e:?}"))
        })?;

        vulkano::sync::now(self.allocator.device().clone())
            .then_execute(self.queue.clone(), command_buffer)
            .map_err(|e| AccelError::BufferDownload(format!("Failed to execute download: {e:?}")))?
            .then_signal_fence_and_flush()
            .map_err(|e| AccelError::BufferDownload(format!("Failed to flush download: {e:?}")))?
            .wait(None)
            .map_err(|e| AccelError::Synchronization(format!("Download sync failed: {e:?}")))?;

        let read = readback
            .read()
            .map_err(|e| AccelError::MemoryMap(format!("Failed to map readback buffer: {e:?}")))?;

        Ok(read.to_vec())
    }

    /// Creates a command buffer builder for compute operations.
    ///
    /// # Errors
    ///
    /// Returns an error if command buffer creation fails.
    pub fn create_command_buffer(
        &self,
    ) -> AccelResult<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>> {
        AutoCommandBufferBuilder::primary(
            self.command_allocator.clone(),
            self.allocator.device().active_queue_family_indices()[0],
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| AccelError::CommandBuffer(format!("Failed to create command buffer: {e:?}")))
    }
}
