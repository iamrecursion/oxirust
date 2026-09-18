//! Memory allocation type tracking and peak-bytes analysis.
//!
//! # Lock-free design
//!
//! The hot recording path (`record`) is entirely lock-free:
//!
//! - The three scalar counters (`seq_counter`, `current_bytes`, `peak_bytes`)
//!   use `AtomicU64` for wait-free updates.
//! - `AllocRecord`s are pushed into a [`crossbeam_deque::Injector`], a
//!   thread-safe, lock-free multi-producer queue.
//!
//! Query methods (`total_bytes`, `records_by_type`, etc.) drain the injector
//! into a local `Vec` (snapshot semantics).  Because the injector is drained
//! rather than cloned, multiple concurrent readers will each see a disjoint
//! subset of records — this is intentional for single-consumer reporting
//! scenarios.  Use `records()` when you need a complete snapshot.
//!
//! # `AllocRecord` tag
//!
//! `tag` is now `&'static str` to keep the hot path allocation-free.

use crossbeam_deque::{Injector, Steal};
use std::sync::atomic::{AtomicU64, Ordering};

/// Category of a memory allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationType {
    /// Stack-allocated memory (e.g., local variables, function frames).
    Stack,
    /// Heap-allocated memory (e.g., `Box`, `Vec`, `malloc`).
    Heap,
    /// Memory-mapped region (e.g., `mmap`, file-backed pages).
    Mmap,
}

impl AllocationType {
    /// Returns `true` if this allocation lives on the heap.
    pub fn is_heap(&self) -> bool {
        matches!(self, AllocationType::Heap)
    }

    /// Human-readable label for the allocation type.
    pub fn label(&self) -> &'static str {
        match self {
            AllocationType::Stack => "stack",
            AllocationType::Heap => "heap",
            AllocationType::Mmap => "mmap",
        }
    }
}

/// A single allocation record.
///
/// `tag` is `&'static str` so `record()` requires no heap allocation.
#[derive(Debug, Clone, Copy)]
pub struct AllocRecord {
    /// Allocation type.
    pub alloc_type: AllocationType,
    /// Size in bytes.
    pub size_bytes: usize,
    /// Call-site tag (e.g., `"module::function"`).
    pub tag: &'static str,
    /// Monotonic sequence number assigned at record time.
    pub seq: u64,
}

/// Threshold above which an allocation is considered "large".
const LARGE_ALLOC_THRESHOLD: usize = 1 << 20; // 1 MiB

impl AllocRecord {
    /// Create a new allocation record.
    pub fn new(alloc_type: AllocationType, size_bytes: usize, tag: &'static str, seq: u64) -> Self {
        Self {
            alloc_type,
            size_bytes,
            tag,
            seq,
        }
    }

    /// Returns `true` if the allocation is ≥ 1 MiB.
    pub fn is_large(&self) -> bool {
        self.size_bytes >= LARGE_ALLOC_THRESHOLD
    }
}

// ---------------------------------------------------------------------------
// AllocationTracker
// ---------------------------------------------------------------------------

/// Tracker that accumulates allocation records and computes aggregate stats.
///
/// ## Concurrency
///
/// `record()` and `free()` are `&self` — they can be called from multiple
/// threads simultaneously with **no locking**:
///
/// - Scalar counters use `AtomicU64` (wait-free).
/// - Records are pushed into a [`crossbeam_deque::Injector`] (lock-free).
///
/// Query methods drain the injector into a local `Vec` — they see all records
/// pushed before the drain starts.
pub struct AllocationTracker {
    records: Injector<AllocRecord>,
    /// Monotonically increasing sequence counter.
    seq_counter: AtomicU64,
    /// Live byte total (incremented on record, decremented on free).
    current_bytes: AtomicU64,
    /// Peak simultaneous live bytes seen since the last reset.
    peak_bytes: AtomicU64,
}

impl std::fmt::Debug for AllocationTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AllocationTracker")
            .field("current_bytes", &self.current_bytes.load(Ordering::Relaxed))
            .field("peak_bytes", &self.peak_bytes.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl AllocationTracker {
    /// Create a new, empty tracker.
    pub fn new() -> Self {
        Self {
            records: Injector::new(),
            seq_counter: AtomicU64::new(0),
            current_bytes: AtomicU64::new(0),
            peak_bytes: AtomicU64::new(0),
        }
    }

    /// Record a new allocation — lock-free, allocation-free.
    ///
    /// `tag` must be a `&'static str` to avoid heap allocation on the hot
    /// path.
    pub fn record(&self, alloc_type: AllocationType, size_bytes: usize, tag: &'static str) {
        // Assign a unique sequence number.
        let seq = self.seq_counter.fetch_add(1, Ordering::Relaxed);

        // Add size to live byte total.
        let new_current = self
            .current_bytes
            .fetch_add(size_bytes as u64, Ordering::Relaxed)
            + size_bytes as u64;

        // CAS-max loop to update peak_bytes.
        let mut old_peak = self.peak_bytes.load(Ordering::Relaxed);
        loop {
            let new_peak = old_peak.max(new_current);
            match self.peak_bytes.compare_exchange_weak(
                old_peak,
                new_peak,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => old_peak = actual,
            }
        }

        // Push record into the lock-free injector.
        let record = AllocRecord::new(alloc_type, size_bytes, tag, seq);
        self.records.push(record);
    }

    /// Simulate a deallocation (reduces current live byte count).
    ///
    /// Does not remove the record — it remains in the injector history.
    pub fn free(&self, size_bytes: usize) {
        let _ = self
            .current_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(size_bytes as u64))
            });
    }

    /// Drains all pending records from the injector into a local `Vec`.
    ///
    /// This is a snapshot: all records pushed before this call are consumed.
    fn drain_records(&self) -> Vec<AllocRecord> {
        let mut out = Vec::new();
        loop {
            match self.records.steal() {
                Steal::Success(r) => out.push(r),
                Steal::Empty => break,
                Steal::Retry => continue,
            }
        }
        out
    }

    /// Sum of all recorded allocation sizes in bytes.
    ///
    /// Drains the injector — records are consumed.
    pub fn total_bytes(&self) -> usize {
        self.drain_records().iter().map(|r| r.size_bytes).sum()
    }

    /// Peak simultaneous live bytes seen during this session.
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes.load(Ordering::Relaxed) as usize
    }

    /// Current live byte total (after any `free` calls).
    pub fn current_bytes(&self) -> usize {
        self.current_bytes.load(Ordering::Relaxed) as usize
    }

    /// Number of recorded allocations (drains the injector).
    pub fn record_count(&self) -> usize {
        self.drain_records().len()
    }

    /// All records — drains the injector into a `Vec`.
    pub fn records(&self) -> Vec<AllocRecord> {
        self.drain_records()
    }

    /// Records filtered by allocation type (drains the injector).
    pub fn records_by_type(&self, alloc_type: AllocationType) -> Vec<AllocRecord> {
        self.drain_records()
            .into_iter()
            .filter(|r| r.alloc_type == alloc_type)
            .collect()
    }

    /// Records that exceed the large-allocation threshold (drains the
    /// injector).
    pub fn large_allocations(&self) -> Vec<AllocRecord> {
        self.drain_records()
            .into_iter()
            .filter(|r| r.is_large())
            .collect()
    }

    /// Bytes allocated broken down by type: (stack, heap, mmap).
    ///
    /// Drains the injector.
    pub fn bytes_by_type(&self) -> (usize, usize, usize) {
        let records = self.drain_records();
        let mut stack = 0usize;
        let mut heap = 0usize;
        let mut mmap = 0usize;
        for r in &records {
            match r.alloc_type {
                AllocationType::Stack => stack += r.size_bytes,
                AllocationType::Heap => heap += r.size_bytes,
                AllocationType::Mmap => mmap += r.size_bytes,
            }
        }
        (stack, heap, mmap)
    }

    /// Clear all records and reset counters.
    ///
    /// Requires `&mut self` for exclusive ownership so resets do not race
    /// with concurrent record/free operations.
    pub fn reset(&mut self) {
        // Drain and discard any pending records.
        loop {
            match self.records.steal() {
                Steal::Empty => break,
                Steal::Success(_) | Steal::Retry => continue,
            }
        }
        self.seq_counter.store(0, Ordering::Relaxed);
        self.current_bytes.store(0, Ordering::Relaxed);
        self.peak_bytes.store(0, Ordering::Relaxed);
    }
}

impl Default for AllocationTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_allocation_type_is_heap() {
        assert!(AllocationType::Heap.is_heap());
        assert!(!AllocationType::Stack.is_heap());
        assert!(!AllocationType::Mmap.is_heap());
    }

    #[test]
    fn test_allocation_type_labels() {
        assert_eq!(AllocationType::Stack.label(), "stack");
        assert_eq!(AllocationType::Heap.label(), "heap");
        assert_eq!(AllocationType::Mmap.label(), "mmap");
    }

    #[test]
    fn test_alloc_record_is_large_false() {
        let r = AllocRecord::new(AllocationType::Heap, 512, "fn_a", 0);
        assert!(!r.is_large());
    }

    #[test]
    fn test_alloc_record_is_large_true() {
        let r = AllocRecord::new(AllocationType::Heap, 2 * 1024 * 1024, "fn_b", 1);
        assert!(r.is_large());
    }

    #[test]
    fn test_alloc_record_is_large_boundary() {
        let r = AllocRecord::new(AllocationType::Heap, LARGE_ALLOC_THRESHOLD, "fn_c", 2);
        assert!(r.is_large());
    }

    #[test]
    fn test_tracker_total_bytes() {
        let t = AllocationTracker::new();
        t.record(AllocationType::Heap, 100, "a");
        t.record(AllocationType::Stack, 50, "b");
        assert_eq!(t.total_bytes(), 150);
    }

    #[test]
    fn test_tracker_peak_bytes() {
        let t = AllocationTracker::new();
        t.record(AllocationType::Heap, 1000, "alloc1");
        t.record(AllocationType::Heap, 500, "alloc2");
        t.free(1000);
        t.record(AllocationType::Heap, 100, "alloc3");
        // Peak was 1500 before the free.
        assert_eq!(t.peak_bytes(), 1500);
    }

    #[test]
    fn test_tracker_current_bytes_after_free() {
        let t = AllocationTracker::new();
        t.record(AllocationType::Heap, 200, "x");
        t.free(100);
        assert_eq!(t.current_bytes(), 100);
    }

    #[test]
    fn test_tracker_free_saturates_at_zero() {
        let t = AllocationTracker::new();
        t.record(AllocationType::Heap, 10, "y");
        t.free(9999);
        assert_eq!(t.current_bytes(), 0);
    }

    #[test]
    fn test_tracker_record_count() {
        let t = AllocationTracker::new();
        for _ in 0..5 {
            t.record(AllocationType::Heap, 10, "item");
        }
        assert_eq!(t.record_count(), 5);
    }

    #[test]
    fn test_tracker_records_by_type() {
        let t = AllocationTracker::new();
        t.record(AllocationType::Heap, 100, "h1");
        t.record(AllocationType::Stack, 50, "s1");
        t.record(AllocationType::Heap, 200, "h2");
        let heap_records = t.records_by_type(AllocationType::Heap);
        assert_eq!(heap_records.len(), 2);
    }

    #[test]
    fn test_tracker_large_allocations() {
        let t = AllocationTracker::new();
        t.record(AllocationType::Heap, 512, "small");
        t.record(AllocationType::Heap, 2 * 1024 * 1024, "big");
        let large = t.large_allocations();
        assert_eq!(large.len(), 1);
        assert_eq!(large[0].tag, "big");
    }

    #[test]
    fn test_tracker_bytes_by_type() {
        let t = AllocationTracker::new();
        t.record(AllocationType::Stack, 100, "s");
        t.record(AllocationType::Heap, 200, "h");
        t.record(AllocationType::Mmap, 400, "m");
        let (stack, heap, mmap) = t.bytes_by_type();
        assert_eq!(stack, 100);
        assert_eq!(heap, 200);
        assert_eq!(mmap, 400);
    }

    #[test]
    fn test_tracker_reset() {
        let mut t = AllocationTracker::new();
        t.record(AllocationType::Heap, 1000, "x");
        t.reset();
        assert_eq!(t.record_count(), 0);
        assert_eq!(t.total_bytes(), 0);
        assert_eq!(t.peak_bytes(), 0);
    }

    #[test]
    fn test_seq_monotonic() {
        let t = AllocationTracker::new();
        t.record(AllocationType::Heap, 1, "a");
        t.record(AllocationType::Heap, 2, "b");
        let mut records = t.records();
        records.sort_by_key(|r| r.seq);
        let seqs: Vec<u64> = records.iter().map(|r| r.seq).collect();
        assert_eq!(seqs, vec![0, 1]);
    }

    // -----------------------------------------------------------------------
    // Atomic counter concurrency tests
    // -----------------------------------------------------------------------

    /// Spawn 8 threads, each recording 1_000 allocations of 100 B.
    /// After joining, `current_bytes` must equal 800_000.
    #[test]
    fn test_atomic_counter_concurrent() {
        const N_THREADS: usize = 8;
        const RECORDS_PER_THREAD: usize = 1_000;
        const ALLOC_SIZE: usize = 100;
        const EXPECTED_TOTAL: usize = N_THREADS * RECORDS_PER_THREAD * ALLOC_SIZE;

        let tracker = Arc::new(AllocationTracker::new());

        let handles: Vec<_> = (0..N_THREADS)
            .map(|_tid| {
                let t = Arc::clone(&tracker);
                std::thread::spawn(move || {
                    for _ in 0..RECORDS_PER_THREAD {
                        t.record(AllocationType::Heap, ALLOC_SIZE, "thread_worker");
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        assert_eq!(
            tracker.current_bytes(),
            EXPECTED_TOTAL,
            "current_bytes after all records must equal {}",
            EXPECTED_TOTAL
        );

        // Free all allocations.
        for _ in 0..N_THREADS * RECORDS_PER_THREAD {
            tracker.free(ALLOC_SIZE);
        }

        assert_eq!(tracker.current_bytes(), 0);
        assert!(tracker.peak_bytes() >= EXPECTED_TOTAL);
    }

    /// Single-threaded: allocate 100 B + 200 B, free 100 B, free 200 B.
    #[test]
    fn test_atomic_peak_bytes_correctness() {
        let t = AllocationTracker::new();

        t.record(AllocationType::Heap, 100, "a"); // current=100, peak=100
        t.record(AllocationType::Heap, 200, "b"); // current=300, peak=300
        t.free(100); //                              current=200, peak=300
        t.free(200); //                              current=0,   peak=300

        assert_eq!(t.current_bytes(), 0);
        assert_eq!(t.peak_bytes(), 300);
    }

    // -----------------------------------------------------------------------
    // Sub-item 31 new test
    // -----------------------------------------------------------------------

    /// 8 threads each call `record()` 1 000 times; after join, drain + count
    /// all records = 8 000 (no loss).
    #[test]
    fn test_allocation_tracker_concurrent_record() {
        const N_THREADS: usize = 8;
        const RECORDS_PER_THREAD: usize = 1_000;

        let tracker = Arc::new(AllocationTracker::new());

        let handles: Vec<_> = (0..N_THREADS)
            .map(|_| {
                let t = Arc::clone(&tracker);
                std::thread::spawn(move || {
                    for _ in 0..RECORDS_PER_THREAD {
                        t.record(AllocationType::Heap, 64, "concurrent_test");
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        let all_records = tracker.records();
        assert_eq!(
            all_records.len(),
            N_THREADS * RECORDS_PER_THREAD,
            "expected {} records, got {}",
            N_THREADS * RECORDS_PER_THREAD,
            all_records.len()
        );
    }
}
