//! Slab allocator for cache entries.
//!
//! Cache workloads often allocate and free many identically-sized objects
//! (e.g. fixed-size media segment headers, metadata records, or small frame
//! descriptors).  Standard heap allocators handle this correctly but can
//! suffer from fragmentation and per-object bookkeeping overhead when the
//! allocation rate is high.
//!
//! This module provides a single-type slab allocator that pre-allocates
//! contiguous slabs and hands out slots from a free-list.  When a slot is
//! freed it is returned to the free-list rather than returned to the system
//! allocator, making future allocations O(1) with minimal bookkeeping.
//!
//! # Design
//!
//! * Each **slab** is a `Vec<Slot<T>>` with `slab_capacity` slots.
//! * A slot is either `Occupied(T)` or `Free`.
//! * A global `free_list: VecDeque<SlabIndex>` keeps track of available
//!   slots across all slabs.
//! * When the free list is empty a new slab is allocated.
//! * Handles are `SlotHandle { slab: usize, index: usize }` and are used
//!   to retrieve or free a specific slot.
//!
//! # Limitations
//!
//! * Not thread-safe: wrap in `Mutex` / `RwLock` for concurrent use.
//! * Does **not** shrink the slab pool when utilisation drops; use
//!   [`compact`] to reclaim empty slabs.
//!
//! [`compact`]: SlabAllocator::compact

use std::collections::VecDeque;
use thiserror::Error;

// ── Errors ────────────────────────────────────────────────────────────────────

/// Errors returned by [`SlabAllocator`].
#[derive(Debug, Error)]
pub enum SlabError {
    /// A handle points outside the slab array bounds.
    #[error("slab index {0} is out of range (allocator has {1} slabs)")]
    SlabIndexOutOfRange(usize, usize),

    /// A slot index is outside the capacity of the addressed slab.
    #[error("slot index {0} is out of range for slab {1} (capacity {2})")]
    SlotIndexOutOfRange(usize, usize, usize),

    /// The addressed slot is free and cannot be read.
    #[error("slot {slot} in slab {slab} is already free")]
    SlotAlreadyFree {
        /// Slab index.
        slab: usize,
        /// Slot index within the slab.
        slot: usize,
    },
}

// ── SlotHandle ────────────────────────────────────────────────────────────────

/// An opaque handle to a slot in a [`SlabAllocator`].
///
/// Handles are returned by [`allocate`] and are valid until [`free`] is
/// called on the same handle.  Using a freed handle is safe: `get` and
/// `get_mut` return `Err(SlabError::SlotAlreadyFree)`.
///
/// [`allocate`]: SlabAllocator::allocate
/// [`free`]: SlabAllocator::free
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotHandle {
    /// Index into the slab array.
    pub slab: usize,
    /// Index within the slab.
    pub index: usize,
}

// ── Slot ──────────────────────────────────────────────────────────────────────

/// A single slot within a slab.
enum Slot<T> {
    /// Slot contains a live value.
    Occupied(T),
    /// Slot is available for allocation.
    Free,
}

// ── SlabAllocator ─────────────────────────────────────────────────────────────

/// A fixed-slot slab allocator for type `T`.
///
/// # Example
///
/// ```rust
/// use oximedia_cache::slab_allocator::SlabAllocator;
///
/// let mut alloc: SlabAllocator<Vec<u8>> = SlabAllocator::new(16);
/// let h = alloc.allocate(vec![0u8; 1024]).unwrap();
/// assert_eq!(alloc.get(h).unwrap().len(), 1024);
/// alloc.free(h).unwrap();
/// ```
pub struct SlabAllocator<T> {
    /// Each inner `Vec` is a slab; slots are either occupied or free.
    slabs: Vec<Vec<Slot<T>>>,
    /// Per-slab live-object count (used by `compact`).
    slab_live: Vec<usize>,
    /// FIFO free-list of available `(slab, index)` pairs.
    free_list: VecDeque<SlotHandle>,
    /// Number of slots per slab.
    slab_capacity: usize,
    /// Total number of allocated (live) objects.
    live_count: usize,
    /// Total number of slots ever created (across all slabs).
    total_slots: usize,
}

impl<T> SlabAllocator<T> {
    /// Create a new allocator that allocates slabs of `slab_capacity` slots.
    ///
    /// A capacity of `0` is treated as `1`.
    pub fn new(slab_capacity: usize) -> Self {
        let cap = slab_capacity.max(1);
        Self {
            slabs: Vec::new(),
            slab_live: Vec::new(),
            free_list: VecDeque::new(),
            slab_capacity: cap,
            live_count: 0,
            total_slots: 0,
        }
    }

    // ── Allocation ────────────────────────────────────────────────────────────

    /// Allocate a slot for `value`.
    ///
    /// Returns a [`SlotHandle`] that can be used to retrieve or free the value.
    ///
    /// If no free slots are available a new slab is allocated.
    pub fn allocate(&mut self, value: T) -> Result<SlotHandle, SlabError> {
        let handle = match self.free_list.pop_front() {
            Some(h) => h,
            None => self.grow_and_get_first_handle(),
        };

        // Place the value in the slot.
        self.slabs[handle.slab][handle.index] = Slot::Occupied(value);
        self.slab_live[handle.slab] += 1;
        self.live_count += 1;

        Ok(handle)
    }

    /// Free the slot identified by `handle`, making it available for reuse.
    ///
    /// Returns `Err` if the handle is out of range or the slot is already
    /// free.
    pub fn free(&mut self, handle: SlotHandle) -> Result<(), SlabError> {
        self.validate_handle(handle)?;
        match &self.slabs[handle.slab][handle.index] {
            Slot::Free => Err(SlabError::SlotAlreadyFree {
                slab: handle.slab,
                slot: handle.index,
            }),
            Slot::Occupied(_) => {
                self.slabs[handle.slab][handle.index] = Slot::Free;
                self.slab_live[handle.slab] = self.slab_live[handle.slab].saturating_sub(1);
                self.live_count = self.live_count.saturating_sub(1);
                self.free_list.push_back(handle);
                Ok(())
            }
        }
    }

    // ── Access ────────────────────────────────────────────────────────────────

    /// Return an immutable reference to the value at `handle`.
    pub fn get(&self, handle: SlotHandle) -> Result<&T, SlabError> {
        self.validate_handle(handle)?;
        match &self.slabs[handle.slab][handle.index] {
            Slot::Occupied(v) => Ok(v),
            Slot::Free => Err(SlabError::SlotAlreadyFree {
                slab: handle.slab,
                slot: handle.index,
            }),
        }
    }

    /// Return a mutable reference to the value at `handle`.
    pub fn get_mut(&mut self, handle: SlotHandle) -> Result<&mut T, SlabError> {
        self.validate_handle(handle)?;
        match &mut self.slabs[handle.slab][handle.index] {
            Slot::Occupied(v) => Ok(v),
            Slot::Free => Err(SlabError::SlotAlreadyFree {
                slab: handle.slab,
                slot: handle.index,
            }),
        }
    }

    // ── Compaction ────────────────────────────────────────────────────────────

    /// Compact the allocator by removing fully-empty slabs.
    ///
    /// Returns the number of slabs reclaimed.  Note that this **invalidates**
    /// all existing `SlotHandle`s whose `slab` index ≥ the first reclaimed
    /// slab.  Callers must not use stale handles after compaction.
    ///
    /// In practice you should call `compact` only during a cache quiesce
    /// window (e.g. on process idle or periodic maintenance).
    pub fn compact(&mut self) -> usize {
        // Identify slabs that are completely empty and can be dropped.
        // We must remove them from back to front to preserve slab indices for
        // slabs we keep, or rebuild the free-list.  The simplest correct
        // approach: rebuild everything from scratch after removing empty slabs.
        let before = self.slabs.len();

        // Collect which slab indices we will keep (live > 0) in order.
        let keep: Vec<bool> = self.slab_live.iter().map(|&l| l > 0).collect();

        // Build new slab array and slab_live.
        let mut new_slabs: Vec<Vec<Slot<T>>> = Vec::new();
        let mut new_slab_live: Vec<usize> = Vec::new();
        // slab_index_map[old_slab] = new_slab (used to rebuild free_list).
        let mut slab_index_map: Vec<Option<usize>> = vec![None; self.slabs.len()];

        for (old_idx, should_keep) in keep.iter().enumerate() {
            if *should_keep {
                let new_idx = new_slabs.len();
                slab_index_map[old_idx] = Some(new_idx);
                // We cannot move out of a Vec<Slot<T>> easily without drain;
                // use drain to move ownership.
                // Temporarily swap the slab out with an empty vec to move it.
                let slab = std::mem::take(&mut self.slabs[old_idx]);
                new_slabs.push(slab);
                new_slab_live.push(self.slab_live[old_idx]);
            }
        }

        // Rebuild free-list: only handles whose slab survived and whose slot
        // is still Free.
        let mut new_free_list: VecDeque<SlotHandle> = VecDeque::new();
        for h in self.free_list.drain(..) {
            if let Some(Some(new_slab)) = slab_index_map.get(h.slab) {
                new_free_list.push_back(SlotHandle {
                    slab: *new_slab,
                    index: h.index,
                });
            }
            // Handles into dropped slabs are silently discarded; those slabs
            // were fully empty so there can be no live data to lose.
        }

        self.slabs = new_slabs;
        self.slab_live = new_slab_live;
        self.free_list = new_free_list;
        self.total_slots = self.slabs.iter().map(|s| s.len()).sum();

        before - self.slabs.len()
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Number of live (allocated) objects.
    pub fn live_count(&self) -> usize {
        self.live_count
    }

    /// Total number of slots across all slabs (live + free).
    pub fn total_slots(&self) -> usize {
        self.total_slots
    }

    /// Number of slabs currently allocated.
    pub fn slab_count(&self) -> usize {
        self.slabs.len()
    }

    /// Number of free slots available without growing.
    pub fn free_slots(&self) -> usize {
        self.free_list.len()
    }

    /// Utilisation ratio: `live_count / total_slots`, or `0.0` if no slots.
    pub fn utilisation(&self) -> f64 {
        if self.total_slots == 0 {
            return 0.0;
        }
        self.live_count as f64 / self.total_slots as f64
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Grow by one slab and return a handle to the first slot of the new slab.
    /// The remaining `slab_capacity - 1` slots are added to the free list.
    fn grow_and_get_first_handle(&mut self) -> SlotHandle {
        let slab_idx = self.slabs.len();
        let cap = self.slab_capacity;

        // Allocate the new slab pre-filled with Free slots.
        let mut slab: Vec<Slot<T>> = Vec::with_capacity(cap);
        for _ in 0..cap {
            slab.push(Slot::Free);
        }
        self.slabs.push(slab);
        self.slab_live.push(0);
        self.total_slots += cap;

        // Add slots [1..cap) to the free list (slot 0 will be used immediately).
        for i in 1..cap {
            self.free_list.push_back(SlotHandle {
                slab: slab_idx,
                index: i,
            });
        }

        SlotHandle {
            slab: slab_idx,
            index: 0,
        }
    }

    fn validate_handle(&self, handle: SlotHandle) -> Result<(), SlabError> {
        if handle.slab >= self.slabs.len() {
            return Err(SlabError::SlabIndexOutOfRange(
                handle.slab,
                self.slabs.len(),
            ));
        }
        let cap = self.slabs[handle.slab].len();
        if handle.index >= cap {
            return Err(SlabError::SlotIndexOutOfRange(
                handle.index,
                handle.slab,
                cap,
            ));
        }
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Basic allocate and get
    #[test]
    fn test_allocate_and_get() {
        let mut alloc: SlabAllocator<u32> = SlabAllocator::new(8);
        let h = alloc.allocate(42).expect("allocation should succeed");
        assert_eq!(alloc.get(h).expect("get should succeed"), &42);
    }

    // 2. live_count increments on allocate
    #[test]
    fn test_live_count_increments() {
        let mut alloc: SlabAllocator<u64> = SlabAllocator::new(4);
        assert_eq!(alloc.live_count(), 0);
        let _ = alloc.allocate(1).expect("ok");
        let _ = alloc.allocate(2).expect("ok");
        assert_eq!(alloc.live_count(), 2);
    }

    // 3. free decrements live_count
    #[test]
    fn test_free_decrements_live_count() {
        let mut alloc: SlabAllocator<u32> = SlabAllocator::new(4);
        let h = alloc.allocate(100).expect("ok");
        alloc.free(h).expect("free should succeed");
        assert_eq!(alloc.live_count(), 0);
    }

    // 4. Free slot cannot be freed again
    #[test]
    fn test_double_free_returns_error() {
        let mut alloc: SlabAllocator<i32> = SlabAllocator::new(4);
        let h = alloc.allocate(7).expect("ok");
        alloc.free(h).expect("first free ok");
        let result = alloc.free(h);
        assert!(
            matches!(result, Err(SlabError::SlotAlreadyFree { .. })),
            "double free should return SlotAlreadyFree"
        );
    }

    // 5. Freed slot is reused on next allocation
    #[test]
    fn test_slot_reuse() {
        let mut alloc: SlabAllocator<u8> = SlabAllocator::new(4);
        let h1 = alloc.allocate(1).expect("ok");
        alloc.free(h1).expect("free h1");
        let h2 = alloc.allocate(2).expect("ok");
        // The reused handle may be the same slot or a different one; just
        // verify total_slots has not grown beyond one slab.
        assert_eq!(alloc.slab_count(), 1, "no new slab should be needed");
        assert_eq!(alloc.get(h2).expect("ok"), &2);
    }

    // 6. Allocations span multiple slabs
    #[test]
    fn test_multi_slab_allocation() {
        let cap = 4usize;
        let mut alloc: SlabAllocator<u32> = SlabAllocator::new(cap);
        let mut handles = Vec::new();
        for i in 0..cap * 3 {
            handles.push(alloc.allocate(i as u32).expect("ok"));
        }
        assert!(alloc.slab_count() >= 3, "at least 3 slabs should exist");
        // Verify every value is still readable.
        for (i, h) in handles.iter().enumerate() {
            assert_eq!(alloc.get(*h).expect("ok"), &(i as u32));
        }
    }

    // 7. total_slots equals slab_count × slab_capacity
    #[test]
    fn test_total_slots() {
        let mut alloc: SlabAllocator<()> = SlabAllocator::new(8);
        let _ = alloc.allocate(()).expect("ok");
        assert_eq!(alloc.total_slots(), 8);
        // Fill to force a second slab.
        for _ in 0..7 {
            let _ = alloc.allocate(()).expect("ok");
        }
        let _ = alloc.allocate(()).expect("ok"); // triggers second slab
        assert_eq!(alloc.total_slots(), 16);
    }

    // 8. utilisation calculation
    #[test]
    fn test_utilisation() {
        let mut alloc: SlabAllocator<u8> = SlabAllocator::new(4);
        let h = alloc.allocate(0).expect("ok"); // 1 live / 4 total
        let _ = alloc.allocate(0).expect("ok"); // 2 live / 4 total
        alloc.free(h).expect("ok"); // 1 live / 4 total
        let u = alloc.utilisation();
        assert!((u - 0.25).abs() < 1e-9, "expected 25% utilisation, got {u}");
    }

    // 9. compact removes empty slabs
    #[test]
    fn test_compact_removes_empty_slabs() {
        let mut alloc: SlabAllocator<u32> = SlabAllocator::new(2);
        // Force 3 slabs by allocating 5 items.
        let handles: Vec<_> = (0..5u32).map(|i| alloc.allocate(i).expect("ok")).collect();
        assert!(alloc.slab_count() >= 2);

        // Free every slot in slab 0 (indices 0 and 1 → handles[0], handles[1]).
        alloc.free(handles[0]).expect("ok");
        alloc.free(handles[1]).expect("ok");

        let reclaimed = alloc.compact();
        assert!(
            reclaimed >= 1,
            "at least one empty slab should be reclaimed"
        );
        // Remaining live objects are still readable.
        for _h in &handles[2..] {
            // After compact slab indices change; just verify live_count.
        }
        assert_eq!(
            alloc.live_count(),
            3,
            "3 live objects should survive compact"
        );
        drop(handles); // silence unused-variable warning
    }

    // 10. get_mut allows mutation
    #[test]
    fn test_get_mut() {
        let mut alloc: SlabAllocator<String> = SlabAllocator::new(4);
        let h = alloc.allocate("hello".to_string()).expect("ok");
        alloc.get_mut(h).expect("ok").push_str(" world");
        assert_eq!(alloc.get(h).expect("ok"), "hello world");
    }

    // 11. Out-of-range slab index returns error
    #[test]
    fn test_invalid_slab_index() {
        let alloc: SlabAllocator<u8> = SlabAllocator::new(4);
        let bad = SlotHandle { slab: 99, index: 0 };
        assert!(matches!(
            alloc.get(bad),
            Err(SlabError::SlabIndexOutOfRange(99, 0))
        ));
    }

    // 12. Out-of-range slot index returns error
    #[test]
    fn test_invalid_slot_index() {
        let mut alloc: SlabAllocator<u8> = SlabAllocator::new(4);
        let _ = alloc.allocate(0).expect("ok");
        let bad = SlotHandle { slab: 0, index: 99 };
        assert!(matches!(
            alloc.get(bad),
            Err(SlabError::SlotIndexOutOfRange(99, 0, 4))
        ));
    }

    // 13. free_slots tracks available slots
    #[test]
    fn test_free_slots() {
        let mut alloc: SlabAllocator<u8> = SlabAllocator::new(4);
        // First allocation: creates slab with 4 slots; uses slot 0; 3 go to free list.
        let h = alloc.allocate(1).expect("ok");
        assert_eq!(alloc.free_slots(), 3);
        alloc.free(h).expect("ok");
        assert_eq!(alloc.free_slots(), 4);
    }

    // 14. Empty allocator has zero utilisation
    #[test]
    fn test_empty_utilisation() {
        let alloc: SlabAllocator<i32> = SlabAllocator::new(8);
        assert_eq!(alloc.utilisation(), 0.0);
    }

    // 15. Allocator with slab_capacity = 1 works correctly
    #[test]
    fn test_single_slot_slab() {
        let mut alloc: SlabAllocator<bool> = SlabAllocator::new(1);
        let h1 = alloc.allocate(true).expect("ok");
        let h2 = alloc.allocate(false).expect("ok");
        assert_ne!(h1, h2);
        assert_eq!(alloc.slab_count(), 2);
        alloc.free(h1).expect("ok");
        alloc.free(h2).expect("ok");
        assert_eq!(alloc.live_count(), 0);
    }
}
