//! Memory usage profiling.
//!
//! Track individual allocations, compute peak and current usage,
//! and generate a summary report including potential leak candidates.

/// A recorded memory allocation.
#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    /// Unique identifier assigned by the profiler.
    pub id: u64,
    /// Size of the allocation in bytes.
    pub size_bytes: usize,
    /// Source location or label describing where the allocation originated.
    pub location: String,
    /// Millisecond timestamp when the allocation was recorded.
    pub timestamp_ms: u64,
}

impl MemoryAllocation {
    /// Create a new allocation record.
    pub fn new(id: u64, size_bytes: usize, location: &str, timestamp_ms: u64) -> Self {
        Self {
            id,
            size_bytes,
            location: location.to_string(),
            timestamp_ms,
        }
    }

    /// Return how many milliseconds ago this allocation was made relative to `now`.
    pub fn age_ms(&self, now: u64) -> u64 {
        now.saturating_sub(self.timestamp_ms)
    }
}

/// Tracks memory allocations and frees during a profiling session.
#[derive(Debug, Default)]
pub struct MemoryProfiler {
    allocations: Vec<MemoryAllocation>,
    /// Peak recorded usage in bytes.
    pub peak_bytes: usize,
    /// Current live usage in bytes.
    pub current_bytes: usize,
    next_id: u64,
}

impl MemoryProfiler {
    /// Create a new, empty memory profiler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an allocation of `size` bytes at `location` and return the allocation id.
    pub fn record_alloc(&mut self, size: usize, location: &str) -> u64 {
        self.record_alloc_at(size, location, 0)
    }

    /// Record an allocation with an explicit timestamp and return the allocation id.
    pub fn record_alloc_at(&mut self, size: usize, location: &str, timestamp_ms: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.allocations
            .push(MemoryAllocation::new(id, size, location, timestamp_ms));
        self.current_bytes = self.current_bytes.saturating_add(size);
        if self.current_bytes > self.peak_bytes {
            self.peak_bytes = self.current_bytes;
        }
        id
    }

    /// Record a free for the allocation with `id`.
    ///
    /// Returns `true` if the allocation was found and removed.
    pub fn record_free(&mut self, id: u64) -> bool {
        if let Some(pos) = self.allocations.iter().position(|a| a.id == id) {
            let size = self.allocations[pos].size_bytes;
            self.allocations.remove(pos);
            self.current_bytes = self.current_bytes.saturating_sub(size);
            true
        } else {
            false
        }
    }

    /// Return the peak memory usage observed.
    pub fn peak_usage(&self) -> usize {
        self.peak_bytes
    }

    /// Return the current live memory usage.
    pub fn current_usage(&self) -> usize {
        self.current_bytes
    }

    /// Return references to all live allocations whose size exceeds `threshold`.
    pub fn large_allocations(&self, threshold: usize) -> Vec<&MemoryAllocation> {
        self.allocations
            .iter()
            .filter(|a| a.size_bytes > threshold)
            .collect()
    }

    /// Return the number of live allocations.
    pub fn allocation_count(&self) -> u64 {
        self.allocations.len() as u64
    }

    /// Return references to all live allocations.
    pub fn allocations(&self) -> &[MemoryAllocation] {
        &self.allocations
    }
}

/// A summarised memory report generated from a [`MemoryProfiler`].
#[derive(Debug, Clone)]
pub struct MemoryReport {
    /// Peak observed memory usage in bytes.
    pub peak_bytes: usize,
    /// Average allocation size in bytes.
    pub avg_bytes: f64,
    /// Total number of allocations recorded (including freed ones).
    pub allocation_count: u64,
    /// Number of allocations that are still live and older than a given threshold.
    pub leak_candidates: usize,
}

impl MemoryReport {
    /// Build a report from the current state of `profiler`.
    ///
    /// Allocations older than `age_threshold_ms` relative to `now_ms` that are
    /// still live are counted as leak candidates.
    pub fn from_profiler(
        profiler: &MemoryProfiler,
        now_ms: u64,
        age_threshold_ms: u64,
    ) -> MemoryReport {
        let allocations = profiler.allocations();
        let avg_bytes = if allocations.is_empty() {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let total = allocations.iter().map(|a| a.size_bytes).sum::<usize>() as f64;
            #[allow(clippy::cast_precision_loss)]
            let count = allocations.len() as f64;
            total / count
        };
        let leak_candidates = allocations
            .iter()
            .filter(|a| a.age_ms(now_ms) > age_threshold_ms)
            .count();

        MemoryReport {
            peak_bytes: profiler.peak_usage(),
            avg_bytes,
            allocation_count: profiler.allocation_count(),
            leak_candidates,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_age_ms_normal() {
        let alloc = MemoryAllocation::new(0, 100, "test", 400);
        assert_eq!(alloc.age_ms(1000), 600);
    }

    #[test]
    fn test_alloc_age_ms_saturating() {
        let alloc = MemoryAllocation::new(0, 100, "test", 1000);
        assert_eq!(alloc.age_ms(500), 0);
    }

    #[test]
    fn test_record_alloc_increments_id() {
        let mut p = MemoryProfiler::new();
        let id0 = p.record_alloc(256, "heap");
        let id1 = p.record_alloc(512, "heap");
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
    }

    #[test]
    fn test_current_usage_after_alloc() {
        let mut p = MemoryProfiler::new();
        p.record_alloc(1024, "buf");
        p.record_alloc(2048, "buf");
        assert_eq!(p.current_usage(), 3072);
    }

    #[test]
    fn test_peak_usage_tracks_maximum() {
        let mut p = MemoryProfiler::new();
        let id = p.record_alloc(4096, "large");
        assert_eq!(p.peak_usage(), 4096);
        p.record_free(id);
        // Peak should not drop
        assert_eq!(p.peak_usage(), 4096);
        assert_eq!(p.current_usage(), 0);
    }

    #[test]
    fn test_record_free_known_id() {
        let mut p = MemoryProfiler::new();
        let id = p.record_alloc(1000, "x");
        assert!(p.record_free(id));
        assert_eq!(p.current_usage(), 0);
        assert_eq!(p.allocation_count(), 0);
    }

    #[test]
    fn test_record_free_unknown_id() {
        let mut p = MemoryProfiler::new();
        assert!(!p.record_free(999));
    }

    #[test]
    fn test_large_allocations_threshold() {
        let mut p = MemoryProfiler::new();
        p.record_alloc(100, "small");
        p.record_alloc(2000, "large");
        p.record_alloc(3000, "larger");
        let large = p.large_allocations(1000);
        assert_eq!(large.len(), 2);
    }

    #[test]
    fn test_large_allocations_none_above_threshold() {
        let mut p = MemoryProfiler::new();
        p.record_alloc(100, "a");
        p.record_alloc(200, "b");
        let large = p.large_allocations(500);
        assert!(large.is_empty());
    }

    #[test]
    fn test_memory_report_avg_bytes() {
        let mut p = MemoryProfiler::new();
        p.record_alloc_at(1000, "a", 0);
        p.record_alloc_at(3000, "b", 0);
        let report = MemoryReport::from_profiler(&p, 1000, 500);
        assert!((report.avg_bytes - 2000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_memory_report_leak_candidates() {
        let mut p = MemoryProfiler::new();
        p.record_alloc_at(500, "old", 100);
        p.record_alloc_at(500, "recent", 900);
        // now = 1000, threshold = 400 => age > 400 qualifies as leak
        let report = MemoryReport::from_profiler(&p, 1000, 400);
        assert_eq!(report.leak_candidates, 1);
    }

    #[test]
    fn test_memory_report_empty_profiler() {
        let p = MemoryProfiler::new();
        let report = MemoryReport::from_profiler(&p, 1000, 500);
        assert_eq!(report.peak_bytes, 0);
        assert_eq!(report.avg_bytes, 0.0);
        assert_eq!(report.allocation_count, 0);
        assert_eq!(report.leak_candidates, 0);
    }
}
