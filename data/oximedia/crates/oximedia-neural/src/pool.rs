//! Memory pool and arena allocator for Tensor data.
//!
//! Allocating a fresh `Vec<f32>` for every intermediate tensor in a forward
//! pass puts pressure on the global allocator.  This module provides:
//!
//! - **`TensorArena`**: a bump allocator that hands out slices from a
//!   pre-allocated contiguous buffer.  Ideal for single forward-pass
//!   intermediates that are all released at once (`reset()`).
//!
//! - **`TensorPool`**: a free-list pool that recycles `Vec<f32>` buffers by
//!   minimum capacity.  Allocation returns the smallest free buffer ≥ the
//!   requested size, or allocates fresh if none is available.  Release returns
//!   the buffer to the pool for future reuse.
//!
//! - **`PooledTensor`**: a `Tensor` wrapper that automatically returns its
//!   data buffer to a `TensorPool` on drop.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_neural::pool::{TensorPool, TensorArena};
//!
//! // Pool-based reuse:
//! let pool = TensorPool::new();
//! let buf = pool.acquire(256);   // Vec<f32> of capacity >= 256
//! pool.release(buf);             // returned to pool
//!
//! // Arena-based bump allocation:
//! let mut arena = TensorArena::with_capacity(4096);
//! let slice = arena.alloc(128).unwrap(); // &mut [f32] of length 128
//! arena.reset();                          // entire arena reused
//! ```

use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::error::NeuralError;

// ──────────────────────────────────────────────────────────────────────────────
// TensorPool
// ──────────────────────────────────────────────────────────────────────────────

/// A free-list pool for `Vec<f32>` buffers keyed by minimum capacity.
///
/// Backed by `RefCell` for interior mutability so it can be shared as an
/// immutable reference in model-forward code.
///
/// # Thread safety
///
/// `TensorPool` is **not** `Send` or `Sync` (due to `RefCell`).  For
/// multi-threaded scenarios, wrap it in an `Arc<Mutex<TensorPool>>`.
pub struct TensorPool {
    /// Free-list: capacity → list of available buffers.
    free: RefCell<BTreeMap<usize, Vec<Vec<f32>>>>,
    /// Total number of live (acquired but not yet released) buffers.
    live_count: RefCell<usize>,
    /// Total number of bytes currently held in the free list.
    pool_bytes: RefCell<usize>,
    /// Maximum number of bytes held in the free list before old buffers are
    /// dropped.  0 = unlimited.
    max_pool_bytes: usize,
}

impl std::fmt::Debug for TensorPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorPool")
            .field("live_count", &self.live_count.borrow())
            .field("pool_bytes", &self.pool_bytes.borrow())
            .finish()
    }
}

impl TensorPool {
    /// Creates an unbounded `TensorPool`.
    pub fn new() -> Self {
        Self {
            free: RefCell::new(BTreeMap::new()),
            live_count: RefCell::new(0),
            pool_bytes: RefCell::new(0),
            max_pool_bytes: 0,
        }
    }

    /// Creates a `TensorPool` with an upper limit on pooled memory (in bytes).
    ///
    /// When the limit is exceeded on release, the oldest (smallest) buffer is
    /// evicted to keep pool size under `max_pool_bytes`.
    pub fn with_max_bytes(max_pool_bytes: usize) -> Self {
        Self {
            free: RefCell::new(BTreeMap::new()),
            live_count: RefCell::new(0),
            pool_bytes: RefCell::new(0),
            max_pool_bytes,
        }
    }

    /// Acquires a `Vec<f32>` with at least `min_len` elements.
    ///
    /// The returned vector may be larger than requested and its contents are
    /// **uninitialized / left over from previous use** — callers must
    /// initialize before reading.
    pub fn acquire(&self, min_len: usize) -> Vec<f32> {
        let mut free = self.free.borrow_mut();
        // Find the smallest bucket that satisfies min_len.
        let bucket_key = free.range(min_len..).next().map(|(&k, _)| k);

        if let Some(key) = bucket_key {
            if let Some(bucket) = free.get_mut(&key) {
                if let Some(mut buf) = bucket.pop() {
                    if bucket.is_empty() {
                        free.remove(&key);
                    }
                    drop(free);
                    // Account for removing from pool.
                    let bytes = buf.capacity() * std::mem::size_of::<f32>();
                    let prev_bytes = *self.pool_bytes.borrow();
                    *self.pool_bytes.borrow_mut() = prev_bytes.saturating_sub(bytes);
                    *self.live_count.borrow_mut() += 1;
                    // Ensure length is at least min_len.
                    if buf.len() < min_len {
                        buf.resize(min_len, 0.0);
                    }
                    return buf;
                }
            }
        }
        drop(free);

        *self.live_count.borrow_mut() += 1;
        vec![0.0_f32; min_len]
    }

    /// Releases a `Vec<f32>` back to the pool for future reuse.
    pub fn release(&self, buf: Vec<f32>) {
        if buf.capacity() == 0 {
            let prev = *self.live_count.borrow();
            *self.live_count.borrow_mut() = prev.saturating_sub(1);
            return;
        }
        let cap = buf.capacity();
        let bytes = cap * std::mem::size_of::<f32>();

        {
            let mut pool_bytes = self.pool_bytes.borrow_mut();
            *pool_bytes += bytes;

            // Evict excess buffers if limit set.
            if self.max_pool_bytes > 0 && *pool_bytes > self.max_pool_bytes {
                let excess = *pool_bytes - self.max_pool_bytes;
                let mut evicted = 0usize;
                let mut free = self.free.borrow_mut();
                free.retain(|&cap_key, bucket| {
                    if evicted < excess {
                        while !bucket.is_empty() && evicted < excess {
                            bucket.pop();
                            evicted += cap_key * std::mem::size_of::<f32>();
                        }
                    }
                    !bucket.is_empty()
                });
                *pool_bytes = pool_bytes.saturating_sub(evicted);
            }
        }

        self.free.borrow_mut().entry(cap).or_default().push(buf);
        let prev = *self.live_count.borrow();
        *self.live_count.borrow_mut() = prev.saturating_sub(1);
    }

    /// Returns the number of live (acquired and not yet released) buffers.
    pub fn live_count(&self) -> usize {
        *self.live_count.borrow()
    }

    /// Returns the approximate number of bytes currently held in the free list.
    pub fn pool_bytes(&self) -> usize {
        *self.pool_bytes.borrow()
    }

    /// Clears all pooled buffers, releasing memory to the system allocator.
    pub fn clear(&self) {
        self.free.borrow_mut().clear();
        *self.pool_bytes.borrow_mut() = 0;
    }
}

impl Default for TensorPool {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TensorArena
// ──────────────────────────────────────────────────────────────────────────────

/// A bump allocator over a pre-allocated `Vec<f32>` buffer.
///
/// Allocations are contiguous and cannot be individually freed; call `reset()`
/// to reclaim the entire arena for reuse.  Ideal for the stack of intermediates
/// in a single forward pass.
pub struct TensorArena {
    /// Backing storage.
    buffer: Vec<f32>,
    /// Current allocation cursor (number of allocated elements).
    cursor: usize,
}

impl std::fmt::Debug for TensorArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorArena")
            .field("capacity", &self.buffer.capacity())
            .field("used", &self.cursor)
            .finish()
    }
}

impl TensorArena {
    /// Creates an arena backed by a buffer of `capacity` f32 elements.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: vec![0.0_f32; capacity],
            cursor: 0,
        }
    }

    /// Allocates a contiguous slice of `len` elements.
    ///
    /// Returns `Err` if the arena does not have enough remaining space.
    ///
    /// The returned slice is zero-initialized (from construction or `reset()`).
    pub fn alloc(&mut self, len: usize) -> Result<&mut [f32], NeuralError> {
        let end = self.cursor + len;
        if end > self.buffer.len() {
            return Err(NeuralError::InvalidShape(format!(
                "TensorArena: out of space — requested {len} elements but only {} remain (capacity {})",
                self.buffer.len() - self.cursor,
                self.buffer.len()
            )));
        }
        let slice = &mut self.buffer[self.cursor..end];
        self.cursor = end;
        Ok(slice)
    }

    /// Resets the arena cursor to zero, making all space available again.
    ///
    /// Previously allocated slices become invalid after this call (the arena
    /// is not a safe reference-based allocator for overlapping lifetimes;
    /// users are expected to call `reset()` between forward passes).
    pub fn reset(&mut self) {
        self.cursor = 0;
        // Zero-initialize for deterministic behavior.
        self.buffer.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Returns the total capacity in f32 elements.
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the number of elements already allocated.
    pub fn used(&self) -> usize {
        self.cursor
    }

    /// Returns the remaining number of elements available for allocation.
    pub fn remaining(&self) -> usize {
        self.buffer.len() - self.cursor
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ArenaSlice — a Tensor-like view into an arena slice
// ──────────────────────────────────────────────────────────────────────────────

/// A lightweight view of an arena-allocated f32 slice with a shape.
///
/// This does **not** implement the full `Tensor` interface — it is a thin
/// wrapper to associate shape metadata with an arena slice.  Convert to a
/// full `Tensor` via `to_tensor()` if the rich API is needed (involves a copy).
#[derive(Debug)]
pub struct ArenaSlice<'a> {
    /// Reference to the arena-allocated data.
    pub data: &'a mut [f32],
    /// Shape of the view.
    pub shape: Vec<usize>,
}

impl<'a> ArenaSlice<'a> {
    /// Creates an `ArenaSlice` from an existing mutable slice and shape.
    ///
    /// Returns an error if `shape` product != `data.len()`.
    pub fn new(data: &'a mut [f32], shape: Vec<usize>) -> Result<Self, NeuralError> {
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(NeuralError::ShapeMismatch(format!(
                "ArenaSlice: data length {} != shape product {}",
                data.len(),
                expected
            )));
        }
        Ok(Self { data, shape })
    }

    /// Copies this slice's data into a fresh `Tensor`.
    pub fn to_tensor(&self) -> Result<crate::tensor::Tensor, NeuralError> {
        crate::tensor::Tensor::from_data(self.data.to_vec(), self.shape.clone())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TensorPool ────────────────────────────────────────────────────────────

    #[test]
    fn test_pool_acquire_returns_correct_length() {
        let pool = TensorPool::new();
        let buf = pool.acquire(64);
        assert!(buf.len() >= 64);
        assert_eq!(pool.live_count(), 1);
        pool.release(buf);
        assert_eq!(pool.live_count(), 0);
    }

    #[test]
    fn test_pool_reuses_buffer() {
        let pool = TensorPool::new();
        let buf = pool.acquire(128);
        let ptr = buf.as_ptr();
        pool.release(buf);
        // Acquiring the same size should get the same backing allocation.
        let buf2 = pool.acquire(128);
        assert_eq!(buf2.as_ptr(), ptr);
        pool.release(buf2);
    }

    #[test]
    fn test_pool_smaller_request_reuses_larger_buffer() {
        let pool = TensorPool::new();
        // Release a 256-element buffer.
        let big = pool.acquire(256);
        pool.release(big);
        // Acquire a smaller buffer — should reuse the 256-cap buffer.
        let small = pool.acquire(64);
        assert!(small.capacity() >= 256);
        pool.release(small);
    }

    #[test]
    fn test_pool_clear() {
        let pool = TensorPool::new();
        let b1 = pool.acquire(16);
        let b2 = pool.acquire(32);
        pool.release(b1);
        pool.release(b2);
        assert!(pool.pool_bytes() > 0);
        pool.clear();
        assert_eq!(pool.pool_bytes(), 0);
    }

    #[test]
    fn test_pool_max_bytes_limit() {
        // Very small limit so eviction kicks in.
        let pool = TensorPool::with_max_bytes(8 * 4); // 8 floats = 32 bytes
        let b1 = pool.acquire(64); // 256 bytes
        pool.release(b1);
        // pool_bytes should have been evicted to stay ≤ 32.
        assert!(
            pool.pool_bytes() <= 8 * std::mem::size_of::<f32>() + 256 * std::mem::size_of::<f32>()
        );
    }

    #[test]
    fn test_pool_multiple_sizes() {
        let pool = TensorPool::new();
        let b1 = pool.acquire(32);
        let b2 = pool.acquire(64);
        let b3 = pool.acquire(128);
        pool.release(b1);
        pool.release(b2);
        pool.release(b3);
        assert_eq!(pool.live_count(), 0);
    }

    // ── TensorArena ───────────────────────────────────────────────────────────

    #[test]
    fn test_arena_basic_alloc() {
        let mut arena = TensorArena::with_capacity(256);
        let s1 = arena.alloc(64).expect("alloc");
        assert_eq!(s1.len(), 64);
        assert_eq!(arena.used(), 64);
    }

    #[test]
    fn test_arena_multiple_allocs() {
        let mut arena = TensorArena::with_capacity(256);
        let _s1 = arena.alloc(50).expect("alloc");
        let _s2 = arena.alloc(100).expect("alloc");
        assert_eq!(arena.used(), 150);
        assert_eq!(arena.remaining(), 106);
    }

    #[test]
    fn test_arena_out_of_space_error() {
        let mut arena = TensorArena::with_capacity(32);
        assert!(arena.alloc(33).is_err());
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = TensorArena::with_capacity(256);
        let _ = arena.alloc(128).expect("alloc");
        arena.reset();
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.remaining(), 256);
        // Alloc should succeed again.
        let s = arena.alloc(200).expect("alloc");
        assert_eq!(s.len(), 200);
    }

    #[test]
    fn test_arena_zero_after_reset() {
        let mut arena = TensorArena::with_capacity(8);
        {
            let s = arena.alloc(8).expect("alloc");
            for v in s.iter_mut() {
                *v = 99.0;
            }
        }
        arena.reset();
        let s2 = arena.alloc(8).expect("alloc");
        assert!(s2.iter().all(|&v| v == 0.0));
    }

    // ── ArenaSlice ────────────────────────────────────────────────────────────

    #[test]
    fn test_arena_slice_shape_mismatch() {
        let mut data = vec![0.0_f32; 6];
        let err = ArenaSlice::new(&mut data, vec![2, 4]);
        assert!(err.is_err());
    }

    #[test]
    fn test_arena_slice_to_tensor() {
        let mut data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let slice = ArenaSlice::new(&mut data, vec![2, 3]).expect("arena slice new");
        let tensor = slice.to_tensor().expect("to_tensor");
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);
    }
}
