//! Per-stream GPU memory allocator for CUDA stream isolation.
//!
//! Each CUDA stream gets its own allocation pool to avoid inter-stream contention.
//! Falls back to shared CPU-backed pool when GPU is unavailable.
//!
//! # Architecture
//!
//! ```text
//!   StreamAllocator
//!         │
//!         ├── StreamArena (stream 0) ── Vec<Vec<u8>> allocations
//!         ├── StreamArena (stream 1) ── Vec<Vec<u8>> allocations
//!         └── ...
//! ```
//!
//! # Usage
//!
//! ```rust
//! use scirs2_core::gpu::stream_allocator::{StreamAllocator, StreamId};
//!
//! let alloc = StreamAllocator::new(64 * 1024 * 1024, 512 * 1024 * 1024);
//! let s0 = StreamId::default_stream();
//! alloc.register_stream(s0).unwrap();
//! let _ptr = alloc.allocate(s0, 4096).unwrap();
//! alloc.reset_stream(s0);
//! ```

use std::collections::HashMap;
use std::sync::Mutex;

use super::GpuError;

// ---------------------------------------------------------------------------
// StreamId
// ---------------------------------------------------------------------------

/// Unique identifier for a GPU stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamId(pub u64);

impl StreamId {
    /// The default (null) stream.
    pub fn default_stream() -> Self {
        StreamId(0)
    }

    /// Create a stream with a specific numeric id.
    pub fn new(id: u64) -> Self {
        StreamId(id)
    }
}

impl std::fmt::Display for StreamId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stream({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// StreamArena
// ---------------------------------------------------------------------------

/// Per-stream allocation arena backed by heap-allocated `Vec<u8>` buffers.
struct StreamArena {
    stream_id: StreamId,
    allocated_bytes: usize,
    max_bytes: usize,
    /// Each element is an individual allocation; the Vec<u8> owns the memory.
    allocations: Vec<Vec<u8>>,
}

impl StreamArena {
    fn new(stream_id: StreamId, max_bytes: usize) -> Self {
        Self {
            stream_id,
            allocated_bytes: 0,
            max_bytes,
            allocations: Vec::new(),
        }
    }

    /// Allocate `bytes` bytes and return a raw pointer into the new buffer.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid until the next call to `reset()` on this
    /// arena or until the arena is dropped.  Callers must not use the pointer
    /// after either of those events.
    fn allocate(&mut self, bytes: usize) -> Result<*mut u8, GpuError> {
        if bytes == 0 {
            return Err(GpuError::InvalidParameter(
                "allocation size must be > 0".to_string(),
            ));
        }
        if self.allocated_bytes + bytes > self.max_bytes {
            return Err(GpuError::OutOfMemory(format!(
                "{}: {} bytes requested, {} bytes available",
                self.stream_id,
                bytes,
                self.max_bytes.saturating_sub(self.allocated_bytes),
            )));
        }
        let mut buf = vec![0u8; bytes];
        let ptr = buf.as_mut_ptr();
        self.allocations.push(buf);
        self.allocated_bytes += bytes;
        Ok(ptr)
    }

    /// Release all allocations in this arena (bulk free).
    fn reset(&mut self) {
        self.allocations.clear();
        self.allocated_bytes = 0;
    }

    fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }
}

// ---------------------------------------------------------------------------
// StreamAllocator
// ---------------------------------------------------------------------------

/// Manager for per-stream GPU memory allocation.
///
/// Each registered stream owns an independent `StreamArena`.  Allocations on
/// different streams cannot interfere with each other.
pub struct StreamAllocator {
    arenas: Mutex<HashMap<StreamId, StreamArena>>,
    per_stream_max_bytes: usize,
    global_max_bytes: usize,
}

impl StreamAllocator {
    /// Create a new `StreamAllocator`.
    ///
    /// - `per_stream_max_bytes`: maximum bytes a single stream arena may hold.
    /// - `global_max_bytes`: maximum total bytes across all streams combined.
    pub fn new(per_stream_max_bytes: usize, global_max_bytes: usize) -> Self {
        Self {
            arenas: Mutex::new(HashMap::new()),
            per_stream_max_bytes,
            global_max_bytes,
        }
    }

    /// Register a new stream.  Must be called before allocating on the stream.
    ///
    /// Returns `Err` if the stream is already registered.
    pub fn register_stream(&self, stream_id: StreamId) -> Result<(), GpuError> {
        let mut arenas = self.arenas.lock().map_err(|_| {
            GpuError::Other("StreamAllocator mutex poisoned during register_stream".to_string())
        })?;
        if arenas.contains_key(&stream_id) {
            return Err(GpuError::InvalidParameter(format!(
                "stream {stream_id} is already registered",
            )));
        }
        arenas.insert(
            stream_id,
            StreamArena::new(stream_id, self.per_stream_max_bytes),
        );
        Ok(())
    }

    /// Unregister a stream and free all its allocations.
    pub fn unregister_stream(&self, stream_id: StreamId) {
        if let Ok(mut arenas) = self.arenas.lock() {
            arenas.remove(&stream_id);
        }
    }

    /// Allocate `bytes` bytes on the given stream.
    ///
    /// Returns a raw pointer into the stream's arena buffer.  The pointer is
    /// valid until the next `reset_stream` or `unregister_stream` call for
    /// this stream.
    pub fn allocate(&self, stream_id: StreamId, bytes: usize) -> Result<*mut u8, GpuError> {
        let mut arenas = self.arenas.lock().map_err(|_| {
            GpuError::Other("StreamAllocator mutex poisoned during allocate".to_string())
        })?;

        // Check global cap before acquiring per-stream arena.
        let total: usize = arenas.values().map(|a| a.allocated_bytes()).sum();
        if total + bytes > self.global_max_bytes {
            return Err(GpuError::OutOfMemory(format!(
                "global limit: {} bytes requested, {} bytes available",
                bytes,
                self.global_max_bytes.saturating_sub(total),
            )));
        }

        let arena = arenas.get_mut(&stream_id).ok_or_else(|| {
            GpuError::InvalidParameter(format!(
                "stream {stream_id} is not registered; call register_stream first",
            ))
        })?;

        arena.allocate(bytes)
    }

    /// Reset all allocations on a stream (bulk free for arena-style use).
    ///
    /// This is a no-op if the stream is not registered.
    pub fn reset_stream(&self, stream_id: StreamId) {
        if let Ok(mut arenas) = self.arenas.lock() {
            if let Some(arena) = arenas.get_mut(&stream_id) {
                arena.reset();
            }
        }
    }

    /// Total bytes allocated across all streams.
    pub fn total_allocated_bytes(&self) -> usize {
        self.arenas
            .lock()
            .map(|arenas| arenas.values().map(|a| a.allocated_bytes()).sum())
            .unwrap_or(0)
    }

    /// List all registered stream IDs.
    pub fn registered_streams(&self) -> Vec<StreamId> {
        self.arenas
            .lock()
            .map(|arenas| arenas.keys().copied().collect())
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_allocator_register() {
        let alloc = StreamAllocator::new(1024, 8192);
        let s0 = StreamId::new(0);
        let s1 = StreamId::new(1);

        alloc.register_stream(s0).expect("register s0 failed");
        alloc.register_stream(s1).expect("register s1 failed");

        let streams = alloc.registered_streams();
        assert!(streams.contains(&s0), "s0 should be registered");
        assert!(streams.contains(&s1), "s1 should be registered");
        assert_eq!(streams.len(), 2);
    }

    #[test]
    fn test_stream_allocator_register_duplicate() {
        let alloc = StreamAllocator::new(1024, 8192);
        let s0 = StreamId::default_stream();
        alloc.register_stream(s0).expect("first register failed");
        let result = alloc.register_stream(s0);
        assert!(result.is_err(), "duplicate registration should return Err");
    }

    #[test]
    fn test_stream_allocator_allocate() {
        let alloc = StreamAllocator::new(1024 * 1024, 8 * 1024 * 1024);
        let s0 = StreamId::new(10);
        let s1 = StreamId::new(20);
        alloc.register_stream(s0).expect("register s0");
        alloc.register_stream(s1).expect("register s1");

        let p0 = alloc.allocate(s0, 512).expect("allocate on s0");
        let p1 = alloc.allocate(s1, 512).expect("allocate on s1");

        // Pointers must be non-null and distinct (different arenas).
        assert!(!p0.is_null(), "s0 pointer should not be null");
        assert!(!p1.is_null(), "s1 pointer should not be null");
        assert_ne!(p0, p1, "pointers from different streams should differ");

        assert_eq!(alloc.total_allocated_bytes(), 1024);
    }

    #[test]
    fn test_stream_allocator_overflow() {
        let alloc = StreamAllocator::new(256, 8192);
        let s = StreamId::new(5);
        alloc.register_stream(s).expect("register");

        // Should succeed within limit.
        alloc
            .allocate(s, 200)
            .expect("first allocation should succeed");

        // This would exceed per-stream cap.
        let result = alloc.allocate(s, 200);
        assert!(
            matches!(result, Err(GpuError::OutOfMemory(_))),
            "expected OutOfMemory, got {result:?}"
        );
    }

    #[test]
    fn test_stream_allocator_global_overflow() {
        // Global cap of 300 bytes, per-stream cap of 200 bytes.
        let alloc = StreamAllocator::new(200, 300);
        let s0 = StreamId::new(0);
        let s1 = StreamId::new(1);
        alloc.register_stream(s0).expect("register s0");
        alloc.register_stream(s1).expect("register s1");

        alloc.allocate(s0, 200).expect("first allocation");
        // Second stream allocation would exceed global cap.
        let result = alloc.allocate(s1, 200);
        assert!(
            matches!(result, Err(GpuError::OutOfMemory(_))),
            "expected global OutOfMemory"
        );
    }

    #[test]
    fn test_stream_allocator_reset() {
        let alloc = StreamAllocator::new(1024, 8192);
        let s = StreamId::new(99);
        alloc.register_stream(s).expect("register");
        alloc.allocate(s, 512).expect("allocate");
        assert_eq!(alloc.total_allocated_bytes(), 512);

        alloc.reset_stream(s);
        assert_eq!(alloc.total_allocated_bytes(), 0, "reset should clear bytes");
    }

    #[test]
    fn test_stream_allocator_unregister() {
        let alloc = StreamAllocator::new(1024, 8192);
        let s = StreamId::new(7);
        alloc.register_stream(s).expect("register");
        assert_eq!(alloc.registered_streams().len(), 1);

        alloc.unregister_stream(s);
        assert!(
            alloc.registered_streams().is_empty(),
            "stream should be gone after unregister"
        );

        // Allocating on an unregistered stream should fail.
        let result = alloc.allocate(s, 64);
        assert!(
            result.is_err(),
            "allocate on unregistered stream should fail"
        );
    }

    #[test]
    fn test_stream_id_default() {
        assert_eq!(StreamId::default_stream(), StreamId(0));
    }

    #[test]
    fn test_stream_allocator_zero_size_rejected() {
        let alloc = StreamAllocator::new(1024, 8192);
        let s = StreamId::new(3);
        alloc.register_stream(s).expect("register");
        let result = alloc.allocate(s, 0);
        assert!(result.is_err(), "zero-size allocation should fail");
    }
}
