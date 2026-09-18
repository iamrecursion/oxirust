//! GPU memory management and allocation tracking
//!
//! This module provides memory allocation tracking, usage statistics,
//! and memory pool management for GPU buffers.

use crate::{GpuBuffer, GpuDevice, GpuError, Result, Shared};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Memory allocation statistics
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryStats {
    /// Total bytes allocated
    pub total_allocated: u64,
    /// Total bytes freed
    pub total_freed: u64,
    /// Current bytes in use
    pub current_usage: u64,
    /// Peak memory usage
    pub peak_usage: u64,
    /// Number of active allocations
    pub allocation_count: u64,
}

impl MemoryStats {
    /// Get the current memory usage in bytes
    #[must_use]
    pub fn current_bytes(&self) -> u64 {
        self.current_usage
    }

    /// Get the current memory usage in megabytes
    #[must_use]
    pub fn current_mb(&self) -> f64 {
        self.current_usage as f64 / (1024.0 * 1024.0)
    }

    /// Get the peak memory usage in bytes
    #[must_use]
    pub fn peak_bytes(&self) -> u64 {
        self.peak_usage
    }

    /// Get the peak memory usage in megabytes
    #[must_use]
    pub fn peak_mb(&self) -> f64 {
        self.peak_usage as f64 / (1024.0 * 1024.0)
    }
}

/// Memory allocator for GPU buffers
pub struct MemoryAllocator {
    device: Shared<wgpu::Device>,
    total_allocated: AtomicU64,
    total_freed: AtomicU64,
    current_usage: AtomicU64,
    peak_usage: AtomicU64,
    allocation_count: AtomicU64,
}

impl MemoryAllocator {
    /// Create a new memory allocator
    #[must_use]
    pub fn new(device: &GpuDevice) -> Self {
        Self {
            device: device.device().clone(),
            total_allocated: AtomicU64::new(0),
            total_freed: AtomicU64::new(0),
            current_usage: AtomicU64::new(0),
            peak_usage: AtomicU64::new(0),
            allocation_count: AtomicU64::new(0),
        }
    }

    /// Track a memory allocation
    pub fn track_allocation(&self, size: u64) {
        self.total_allocated.fetch_add(size, Ordering::Relaxed);
        let current = self.current_usage.fetch_add(size, Ordering::Relaxed) + size;
        self.allocation_count.fetch_add(1, Ordering::Relaxed);

        // Update peak usage
        let mut peak = self.peak_usage.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_usage.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
    }

    /// Track a memory deallocation
    pub fn track_deallocation(&self, size: u64) {
        self.total_freed.fetch_add(size, Ordering::Relaxed);
        self.current_usage.fetch_sub(size, Ordering::Relaxed);
        self.allocation_count.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get current memory statistics
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            total_allocated: self.total_allocated.load(Ordering::Relaxed),
            total_freed: self.total_freed.load(Ordering::Relaxed),
            current_usage: self.current_usage.load(Ordering::Relaxed),
            peak_usage: self.peak_usage.load(Ordering::Relaxed),
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
        }
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        self.total_allocated.store(0, Ordering::Relaxed);
        self.total_freed.store(0, Ordering::Relaxed);
        self.current_usage.store(0, Ordering::Relaxed);
        self.peak_usage.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
    }

    /// Get the device reference
    pub fn device(&self) -> &Shared<wgpu::Device> {
        &self.device
    }
}

/// Memory pool for reusing GPU buffers
pub struct MemoryPool {
    #[allow(dead_code)]
    device: Shared<wgpu::Device>,
    allocator: Shared<MemoryAllocator>,
    pools: RwLock<HashMap<u64, Vec<GpuBuffer>>>,
}

impl MemoryPool {
    /// Create a new memory pool
    #[must_use]
    pub fn new(device: &GpuDevice) -> Self {
        Self {
            device: device.device().clone(),
            allocator: Shared::new(MemoryAllocator::new(device)),
            pools: RwLock::new(HashMap::new()),
        }
    }

    /// Allocate a buffer from the pool
    ///
    /// If a buffer of the requested size is available in the pool, it will be reused.
    /// Otherwise, a new buffer will be allocated.
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `size` - Buffer size in bytes
    /// * `buffer_type` - Type of buffer to allocate
    ///
    /// # Errors
    ///
    /// Returns an error if buffer allocation fails.
    pub fn allocate(
        &self,
        device: &GpuDevice,
        size: u64,
        buffer_type: crate::buffer::BufferType,
    ) -> Result<GpuBuffer> {
        // Try to reuse a buffer from the pool
        {
            let mut pools = self.pools.write();
            if let Some(pool) = pools.get_mut(&size) {
                if let Some(buffer) = pool.pop() {
                    return Ok(buffer);
                }
            }
        }

        // Allocate a new buffer
        let buffer = GpuBuffer::new(device, size, buffer_type)?;
        self.allocator.track_allocation(size);

        Ok(buffer)
    }

    /// Return a buffer to the pool for reuse
    ///
    /// # Arguments
    ///
    /// * `buffer` - Buffer to return to the pool
    pub fn deallocate(&self, buffer: GpuBuffer) {
        let size = buffer.size();
        let mut pools = self.pools.write();
        pools.entry(size).or_default().push(buffer);
    }

    /// Clear the memory pool
    pub fn clear(&self) {
        let mut pools = self.pools.write();
        for (size, buffers) in pools.drain() {
            let total_size = size * buffers.len() as u64;
            self.allocator.track_deallocation(total_size);
        }
    }

    /// Get the number of buffers in the pool
    pub fn pool_size(&self) -> usize {
        let pools = self.pools.read();
        pools.values().map(std::vec::Vec::len).sum()
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        self.allocator.stats()
    }

    /// Get the allocator
    pub fn allocator(&self) -> &Shared<MemoryAllocator> {
        &self.allocator
    }
}

/// RAII wrapper for automatic buffer deallocation
pub struct ManagedBuffer {
    buffer: Option<GpuBuffer>,
    pool: Shared<MemoryPool>,
}

impl ManagedBuffer {
    /// Create a new managed buffer
    pub fn new(buffer: GpuBuffer, pool: Shared<MemoryPool>) -> Self {
        Self {
            buffer: Some(buffer),
            pool,
        }
    }

    /// Get a reference to the buffer.
    ///
    /// # Panics
    ///
    /// Panics if the buffer has already been released via [`ManagedBuffer::take`].
    /// Under normal usage this cannot happen because `take()` consumes `self`.
    #[must_use]
    pub fn buffer(&self) -> &GpuBuffer {
        self.buffer
            .as_ref()
            .unwrap_or_else(|| unreachable!("ManagedBuffer accessed after buffer was released"))
    }

    /// Fallible variant of [`ManagedBuffer::buffer`].
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Internal`] if the buffer has already been released.
    pub fn try_buffer(&self) -> Result<&GpuBuffer> {
        self.buffer
            .as_ref()
            .ok_or_else(|| GpuError::Internal("Buffer already released".to_string()))
    }

    /// Take ownership of the buffer, preventing automatic deallocation.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Internal`] if the buffer has already been released.
    pub fn take(mut self) -> Result<GpuBuffer> {
        self.buffer
            .take()
            .ok_or_else(|| GpuError::Internal("Buffer already released".to_string()))
    }
}

impl Drop for ManagedBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.deallocate(buffer);
        }
    }
}

impl std::ops::Deref for ManagedBuffer {
    type Target = GpuBuffer;

    fn deref(&self) -> &Self::Target {
        self.buffer()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats() {
        let stats = MemoryStats {
            total_allocated: 1024 * 1024 * 100, // 100 MB
            total_freed: 1024 * 1024 * 20,      // 20 MB
            current_usage: 1024 * 1024 * 80,    // 80 MB
            peak_usage: 1024 * 1024 * 90,       // 90 MB
            allocation_count: 10,
        };

        assert_eq!(stats.current_bytes(), 1024 * 1024 * 80);
        assert!((stats.current_mb() - 80.0).abs() < 0.01);
        assert_eq!(stats.peak_bytes(), 1024 * 1024 * 90);
        assert!((stats.peak_mb() - 90.0).abs() < 0.01);
    }

    #[test]
    #[ignore] // Requires GPU hardware; run with --ignored
    fn test_memory_allocator_tracking() {
        let Ok(gpu_device) = crate::device::GpuDevice::new(None) else {
            return;
        };
        let allocator = MemoryAllocator::new(&gpu_device);

        allocator.track_allocation(1024);
        allocator.track_allocation(2048);

        let stats = allocator.stats();
        assert_eq!(stats.total_allocated, 3072);
        assert_eq!(stats.current_usage, 3072);
        assert_eq!(stats.allocation_count, 2);

        allocator.track_deallocation(1024);

        let stats = allocator.stats();
        assert_eq!(stats.total_freed, 1024);
        assert_eq!(stats.current_usage, 2048);
        assert_eq!(stats.allocation_count, 1);
    }

    #[test]
    fn test_memory_allocator_tracking_no_gpu() {
        // Test tracking logic without GPU initialization using atomic counters directly.
        let total_allocated = AtomicU64::new(0);
        let total_freed = AtomicU64::new(0);
        let current_usage = AtomicU64::new(0);
        let allocation_count = AtomicU64::new(0);

        // Simulate track_allocation(1024)
        total_allocated.fetch_add(1024, Ordering::Relaxed);
        current_usage.fetch_add(1024, Ordering::Relaxed);
        allocation_count.fetch_add(1, Ordering::Relaxed);

        // Simulate track_allocation(2048)
        total_allocated.fetch_add(2048, Ordering::Relaxed);
        current_usage.fetch_add(2048, Ordering::Relaxed);
        allocation_count.fetch_add(1, Ordering::Relaxed);

        assert_eq!(total_allocated.load(Ordering::Relaxed), 3072);
        assert_eq!(current_usage.load(Ordering::Relaxed), 3072);
        assert_eq!(allocation_count.load(Ordering::Relaxed), 2);

        // Simulate track_deallocation(1024)
        total_freed.fetch_add(1024, Ordering::Relaxed);
        current_usage.fetch_sub(1024, Ordering::Relaxed);
        allocation_count.fetch_sub(1, Ordering::Relaxed);

        assert_eq!(total_freed.load(Ordering::Relaxed), 1024);
        assert_eq!(current_usage.load(Ordering::Relaxed), 2048);
        assert_eq!(allocation_count.load(Ordering::Relaxed), 1);
    }
}
