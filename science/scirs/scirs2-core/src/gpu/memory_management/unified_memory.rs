//! Unified memory allocator for CPU+GPU shared pages.
//!
//! On systems without real GPU hardware, falls back to CPU-only allocation.
//! On systems with GPU support, tracks synchronisation state between host
//! (CPU) and device (GPU) memory views of the same logical buffer.
//!
//! # Design
//!
//! - `UnifiedBuffer<T>` wraps a `Vec<T>` as the authoritative CPU-side backing
//!   store.  A `SyncState` enum records which side holds the canonical data.
//! - `UnifiedAllocator` tracks total allocated bytes and enforces a soft
//!   capacity limit, returning `GpuError::OutOfMemory` when exceeded.
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::gpu::memory_management::unified_memory::{UnifiedAllocator, SyncState};
//!
//! let alloc = UnifiedAllocator::new_cpu_only(16 * 1024 * 1024);
//! let mut buf = alloc.allocate::<f32>(256).expect("allocation failed");
//! buf.host_ptr_mut().iter_mut().enumerate().for_each(|(i, x)| *x = i as f32);
//! buf.mark_host_modified();
//! assert_eq!(buf.host_ptr()[3], 3.0_f32);
//! assert_eq!(buf.sync_state(), SyncState::HostCurrent);
//! ```

use crate::gpu::GpuError;
use std::sync::atomic::{AtomicUsize, Ordering};

// ─────────────────────────────────────────────────────────────────────────────
// SyncState
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks which memory side holds the most recent version of a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncState {
    /// The CPU (host) side holds the latest data.
    HostCurrent,
    /// The GPU (device) side holds the latest data.
    DeviceCurrent,
    /// Both host and device are in sync.
    Synchronized,
}

// ─────────────────────────────────────────────────────────────────────────────
// UnifiedBuffer<T>
// ─────────────────────────────────────────────────────────────────────────────

/// A memory region accessible from both CPU and GPU.
///
/// On systems without real GPU support (or when created via
/// [`UnifiedAllocator::new_cpu_only`]), all GPU-sync operations are
/// no-ops and the data lives solely in the CPU `Vec<T>`.
pub struct UnifiedBuffer<T> {
    /// Backing CPU-side storage.
    data: Vec<T>,
    /// Number of elements (also `data.len()`).
    size: usize,
    /// `Some(id)` when bound to a GPU device, `None` for CPU-only.
    device_id: Option<usize>,
    /// Which side holds the current canonical data.
    sync_state: SyncState,
}

impl<T: Clone + Default> UnifiedBuffer<T> {
    // ------------------------------------------------------------------
    // Internal constructor
    // ------------------------------------------------------------------

    fn new(size: usize, device_id: Option<usize>) -> Self {
        UnifiedBuffer {
            data: vec![T::default(); size],
            size,
            device_id,
            sync_state: SyncState::Synchronized,
        }
    }

    // ------------------------------------------------------------------
    // Host-side access
    // ------------------------------------------------------------------

    /// Returns a shared slice of the CPU-side data.
    pub fn host_ptr(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable slice of the CPU-side data.
    ///
    /// Does not automatically update the sync state; call
    /// [`mark_host_modified`](Self::mark_host_modified) if you modify
    /// elements and want subsequent [`sync_to_device`](Self::sync_to_device)
    /// calls to act on the change.
    pub fn host_ptr_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    // ------------------------------------------------------------------
    // Sync state management
    // ------------------------------------------------------------------

    /// Returns the current synchronisation state.
    pub fn sync_state(&self) -> SyncState {
        self.sync_state
    }

    /// Marks the host as holding newer data than the device.
    pub fn mark_host_modified(&mut self) {
        self.sync_state = SyncState::HostCurrent;
    }

    /// Marks the device as holding newer data than the host.
    pub fn mark_device_modified(&mut self) {
        self.sync_state = SyncState::DeviceCurrent;
    }

    /// Synchronise device data back to the host.
    ///
    /// In CPU-only mode (no `device_id`) this is a no-op.  With real GPU
    /// support a DMA transfer from device to host would be issued here.
    pub fn sync_to_host(&mut self) -> Result<(), GpuError> {
        if self.device_id.is_none() {
            // CPU-only: nothing to do; host is always current.
            self.sync_state = SyncState::Synchronized;
            return Ok(());
        }
        // GPU mode: in a real implementation a device→host copy would occur
        // here.  For now we record the state change.
        self.sync_state = SyncState::Synchronized;
        Ok(())
    }

    /// Synchronise host data to the device.
    ///
    /// In CPU-only mode this is a no-op.  With real GPU support a DMA
    /// transfer from host to device would be issued here.
    pub fn sync_to_device(&mut self) -> Result<(), GpuError> {
        if self.device_id.is_none() {
            // CPU-only: nothing to do.
            self.sync_state = SyncState::Synchronized;
            return Ok(());
        }
        // GPU mode: host→device copy would occur here.
        self.sync_state = SyncState::Synchronized;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Capacity
    // ------------------------------------------------------------------

    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` if the buffer contains no elements.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Returns the device this buffer is associated with, if any.
    pub fn device_id(&self) -> Option<usize> {
        self.device_id
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// UnifiedAllocator
// ─────────────────────────────────────────────────────────────────────────────

/// Allocator that issues `UnifiedBuffer` instances and tracks total usage.
///
/// The allocator enforces a soft capacity limit; attempting to allocate beyond
/// `max_bytes` returns `GpuError::OutOfMemory`.
pub struct UnifiedAllocator {
    /// Optional GPU device ID.  `None` disables device-side functionality.
    device_id: Option<usize>,
    /// Running total of allocated bytes, maintained atomically.
    allocated_bytes: AtomicUsize,
    /// Maximum bytes this allocator will hand out.
    max_bytes: usize,
}

impl UnifiedAllocator {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Creates an allocator bound to a GPU device.
    ///
    /// # Arguments
    ///
    /// * `device_id` — GPU device index (0-based).
    /// * `max_bytes` — soft upper bound on total allocation.
    pub fn new(device_id: usize, max_bytes: usize) -> Self {
        UnifiedAllocator {
            device_id: Some(device_id),
            allocated_bytes: AtomicUsize::new(0),
            max_bytes,
        }
    }

    /// Creates a CPU-only allocator (no GPU involvement).
    ///
    /// All `sync_to_host` / `sync_to_device` calls on buffers produced by
    /// this allocator are no-ops.
    pub fn new_cpu_only(max_bytes: usize) -> Self {
        UnifiedAllocator {
            device_id: None,
            allocated_bytes: AtomicUsize::new(0),
            max_bytes,
        }
    }

    // ------------------------------------------------------------------
    // Allocation
    // ------------------------------------------------------------------

    /// Allocates a new `UnifiedBuffer<T>` with `size` elements.
    ///
    /// Returns `Err(GpuError::OutOfMemory)` if the requested byte count would
    /// push total usage over `max_bytes`.
    pub fn allocate<T: Clone + Default>(&self, size: usize) -> Result<UnifiedBuffer<T>, GpuError> {
        let byte_count = size
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| GpuError::OutOfMemory("allocation size overflow".to_string()))?;

        // Compare-and-swap loop to enforce the limit without a Mutex.
        loop {
            let current = self.allocated_bytes.load(Ordering::Acquire);
            let new_total = current.checked_add(byte_count).ok_or_else(|| {
                GpuError::OutOfMemory("total allocation counter overflow".to_string())
            })?;

            if new_total > self.max_bytes {
                return Err(GpuError::OutOfMemory(format!(
                    "requested {} bytes but only {} bytes remain (capacity {})",
                    byte_count,
                    self.max_bytes.saturating_sub(current),
                    self.max_bytes
                )));
            }

            // Attempt to commit the reservation.
            if self
                .allocated_bytes
                .compare_exchange(current, new_total, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
            // Another thread raced; retry.
        }

        Ok(UnifiedBuffer::new(size, self.device_id))
    }

    // ------------------------------------------------------------------
    // Introspection
    // ------------------------------------------------------------------

    /// Returns the number of bytes currently allocated through this allocator.
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes.load(Ordering::Acquire)
    }

    /// Returns the maximum number of bytes this allocator will hand out.
    pub fn capacity_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Returns the device ID if this allocator is GPU-bound.
    pub fn device_id(&self) -> Option<usize> {
        self.device_id
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_allocator_basic() {
        let alloc = UnifiedAllocator::new_cpu_only(1024 * 1024);
        let mut buf = alloc
            .allocate::<f32>(64)
            .expect("allocation should succeed");

        // Write values through host_ptr_mut.
        for (i, elem) in buf.host_ptr_mut().iter_mut().enumerate() {
            *elem = i as f32;
        }

        // Read back through host_ptr.
        let data = buf.host_ptr();
        assert_eq!(data.len(), 64);
        for (i, &val) in data.iter().enumerate() {
            assert!(
                (val - i as f32).abs() < f32::EPSILON,
                "element {i} expected {}, got {val}",
                i as f32
            );
        }

        assert_eq!(buf.len(), 64);
        assert!(!buf.is_empty());
        assert_eq!(alloc.allocated_bytes(), 64 * std::mem::size_of::<f32>());
    }

    #[test]
    fn test_unified_allocator_sync_state() {
        let alloc = UnifiedAllocator::new_cpu_only(1024 * 1024);
        let mut buf = alloc
            .allocate::<u64>(32)
            .expect("allocation should succeed");

        // Fresh buffer should be Synchronized (no actual device side).
        assert_eq!(buf.sync_state(), SyncState::Synchronized);

        // Mark host as modified.
        buf.mark_host_modified();
        assert_eq!(buf.sync_state(), SyncState::HostCurrent);

        // Sync to device (no-op in CPU mode, but state should clear).
        buf.sync_to_device()
            .expect("sync_to_device should not fail");
        assert_eq!(buf.sync_state(), SyncState::Synchronized);

        // Mark device as modified.
        buf.mark_device_modified();
        assert_eq!(buf.sync_state(), SyncState::DeviceCurrent);

        // Sync back to host.
        buf.sync_to_host().expect("sync_to_host should not fail");
        assert_eq!(buf.sync_state(), SyncState::Synchronized);
    }

    #[test]
    fn test_unified_allocator_overflow() {
        // Only 100 bytes of capacity.
        let alloc = UnifiedAllocator::new_cpu_only(100);

        // 10 f32 values = 40 bytes — succeeds.
        let _buf1 = alloc
            .allocate::<f32>(10)
            .expect("first allocation should succeed");

        // Another 10 f32 = 40 bytes — still within 80 total, succeeds.
        let _buf2 = alloc
            .allocate::<f32>(10)
            .expect("second allocation should succeed");

        // A third 10 f32 = 40 bytes — would push total to 120, must fail.
        let result = alloc.allocate::<f32>(10);
        assert!(
            result.is_err(),
            "allocation beyond capacity should return Err"
        );

        match result {
            Err(GpuError::OutOfMemory(_)) => {} // correct
            Err(other) => panic!("expected OutOfMemory, got {:?}", other),
            Ok(_) => panic!("expected Err but got Ok"),
        }
    }

    #[test]
    fn test_unified_allocator_empty_buffer() {
        let alloc = UnifiedAllocator::new_cpu_only(1024);
        let buf = alloc
            .allocate::<i32>(0)
            .expect("zero-size allocation should succeed");
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_unified_allocator_gpu_device_id() {
        let alloc = UnifiedAllocator::new(0, 1024 * 1024);
        assert_eq!(alloc.device_id(), Some(0));

        let buf = alloc
            .allocate::<u8>(128)
            .expect("gpu allocation should succeed");
        assert_eq!(buf.device_id(), Some(0));
    }
}
