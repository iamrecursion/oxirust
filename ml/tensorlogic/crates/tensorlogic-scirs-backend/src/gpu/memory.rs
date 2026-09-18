//! GPU memory management: buffers and memory pools.

use std::marker::PhantomData;
use std::mem::size_of;

use super::executor::GpuError;

/// A handle to a GPU buffer containing elements of type `T`.
///
/// This is a logical handle — in stub mode no actual GPU memory is allocated.
#[derive(Debug)]
pub struct GpuBuffer<T> {
    device_index: usize,
    size: usize, // size in bytes
    _marker: PhantomData<T>,
}

impl<T> GpuBuffer<T> {
    /// Create a new buffer handle (internal use).
    pub(crate) fn new(device_index: usize, size: usize) -> Self {
        Self {
            device_index,
            size,
            _marker: PhantomData,
        }
    }

    /// Returns the buffer size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the device index this buffer belongs to.
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Returns the number of elements of type `T` that fit in this buffer.
    pub fn element_count(&self) -> usize {
        let elem_size = size_of::<T>();
        self.size.checked_div(elem_size).unwrap_or(0)
    }
}

/// A GPU memory pool that tracks allocations and peak usage for a single device.
#[derive(Debug)]
pub struct GpuMemoryPool {
    pub device_index: usize,
    total_allocated_bytes: usize,
    peak_allocated_bytes: usize,
    allocation_count: usize,
}

impl Default for GpuMemoryPool {
    fn default() -> Self {
        Self::new(0)
    }
}

impl GpuMemoryPool {
    /// Create a new memory pool for the given device index.
    pub fn new(device_index: usize) -> Self {
        Self {
            device_index,
            total_allocated_bytes: 0,
            peak_allocated_bytes: 0,
            allocation_count: 0,
        }
    }

    /// Attempt to allocate a buffer of `count` elements of type `T`.
    ///
    /// In stub mode (no CUDA runtime), this always returns [`GpuError::NotAvailable`].
    pub fn allocate<T>(&mut self, _count: usize) -> Result<GpuBuffer<T>, GpuError> {
        Err(GpuError::NotAvailable {
            reason: "CUDA not available in stub mode".to_string(),
        })
    }

    /// Release a previously allocated buffer.
    ///
    /// In stub mode this is a no-op that succeeds, keeping accounting consistent.
    pub fn deallocate<T>(&mut self, _buffer: GpuBuffer<T>) -> Result<(), GpuError> {
        // Stub: nothing to actually free, but we should not error here.
        Ok(())
    }

    /// Returns the total bytes currently considered allocated by this pool.
    pub fn allocated_bytes(&self) -> usize {
        self.total_allocated_bytes
    }

    /// Returns the peak allocation in bytes seen since the last reset.
    pub fn peak_bytes(&self) -> usize {
        self.peak_allocated_bytes
    }

    /// Returns the number of successful allocations performed so far.
    pub fn allocation_count(&self) -> usize {
        self.allocation_count
    }

    /// Reset the peak allocation counter to the current allocation level.
    pub fn reset_peak(&mut self) {
        self.peak_allocated_bytes = self.total_allocated_bytes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_memory_pool_new() {
        let pool = GpuMemoryPool::new(2);
        assert_eq!(pool.device_index, 2);
        assert_eq!(pool.allocated_bytes(), 0);
        assert_eq!(pool.peak_bytes(), 0);
        assert_eq!(pool.allocation_count(), 0);
    }

    #[test]
    fn test_allocate_returns_not_available() {
        let mut pool = GpuMemoryPool::new(0);
        let result: Result<GpuBuffer<f32>, GpuError> = pool.allocate(1024);
        assert!(result.is_err());
        match result {
            Err(GpuError::NotAvailable { reason }) => {
                assert!(!reason.is_empty());
            }
            other => panic!("Expected NotAvailable, got {:?}", other),
        }
    }

    #[test]
    fn test_deallocate_no_panic() {
        let mut pool = GpuMemoryPool::new(0);
        // Manually construct a buffer to test deallocation (simulating a "leaked" handle).
        let buf: GpuBuffer<f64> = GpuBuffer::new(0, 256);
        let result = pool.deallocate(buf);
        assert!(result.is_ok());
    }

    #[test]
    fn test_peak_tracking_initial() {
        let pool = GpuMemoryPool::new(0);
        // Initially both allocated and peak are 0.
        assert_eq!(pool.allocated_bytes(), 0);
        assert_eq!(pool.peak_bytes(), 0);
    }

    #[test]
    fn test_allocation_count() {
        let pool = GpuMemoryPool::new(0);
        // No successful allocations in stub mode.
        assert_eq!(pool.allocation_count(), 0);
    }

    #[test]
    fn test_reset_peak() {
        let mut pool = GpuMemoryPool::new(0);
        // Manually set internal state to simulate prior allocations.
        pool.peak_allocated_bytes = 4096;
        pool.total_allocated_bytes = 2048;
        pool.reset_peak();
        assert_eq!(pool.peak_bytes(), 2048);
    }
}
