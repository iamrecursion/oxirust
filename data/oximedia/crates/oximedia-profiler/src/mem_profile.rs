//! Memory profiling: allocation tracking, peak usage, and fragmentation metrics.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Category of a memory allocation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AllocCategory {
    /// Video frame buffers.
    VideoFrame,
    /// Audio sample buffers.
    AudioBuffer,
    /// Compressed bitstream data.
    BitstreamData,
    /// Metadata and index structures.
    Metadata,
    /// General heap allocation.
    General,
    /// User-defined category.
    Custom(String),
}

impl std::fmt::Display for AllocCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VideoFrame => write!(f, "VideoFrame"),
            Self::AudioBuffer => write!(f, "AudioBuffer"),
            Self::BitstreamData => write!(f, "BitstreamData"),
            Self::Metadata => write!(f, "Metadata"),
            Self::General => write!(f, "General"),
            Self::Custom(s) => write!(f, "Custom({})", s),
        }
    }
}

/// Record of a single allocation event.
#[derive(Debug, Clone)]
pub struct AllocRecord {
    /// Unique allocation identifier.
    pub id: u64,
    /// Size in bytes.
    pub size: usize,
    /// Category of this allocation.
    pub category: AllocCategory,
    /// When the allocation occurred (relative to profiler start).
    pub timestamp: Duration,
    /// Callsite tag (e.g. function name).
    pub callsite: String,
    /// Whether this allocation has been freed.
    pub freed: bool,
    /// When the allocation was freed (if freed).
    pub free_timestamp: Option<Duration>,
}

impl AllocRecord {
    /// Lifetime of this allocation, if it has been freed.
    pub fn lifetime(&self) -> Option<Duration> {
        self.free_timestamp
            .map(|ft| ft.saturating_sub(self.timestamp))
    }

    /// Whether this allocation is still live.
    pub fn is_live(&self) -> bool {
        !self.freed
    }
}

/// Snapshot of memory state at a point in time.
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Timestamp of snapshot.
    pub timestamp: Duration,
    /// Total bytes currently live.
    pub live_bytes: usize,
    /// Number of live allocations.
    pub live_count: u64,
    /// Cumulative bytes allocated so far.
    pub total_allocated: usize,
    /// Cumulative bytes freed so far.
    pub total_freed: usize,
}

impl MemorySnapshot {
    /// Net memory pressure: live_bytes / peak_bytes ratio if peak is available.
    #[allow(clippy::cast_precision_loss)]
    pub fn fragmentation_estimate(&self, peak_bytes: usize) -> f64 {
        if peak_bytes == 0 {
            return 0.0;
        }
        // Simple fragmentation heuristic: unused / peak
        let unused = peak_bytes.saturating_sub(self.live_bytes);
        unused as f64 / peak_bytes as f64
    }
}

/// Tracks memory allocations and computes profiling metrics.
#[derive(Debug)]
pub struct MemoryProfiler {
    start_time: Instant,
    next_id: u64,
    records: HashMap<u64, AllocRecord>,
    snapshots: Vec<MemorySnapshot>,
    live_bytes: usize,
    live_count: u64,
    peak_bytes: usize,
    total_allocated: usize,
    total_freed: usize,
}

impl MemoryProfiler {
    /// Create a new memory profiler.
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            next_id: 1,
            records: HashMap::new(),
            snapshots: Vec::new(),
            live_bytes: 0,
            live_count: 0,
            peak_bytes: 0,
            total_allocated: 0,
            total_freed: 0,
        }
    }

    fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Record an allocation and return its ID.
    pub fn alloc(
        &mut self,
        size: usize,
        category: AllocCategory,
        callsite: impl Into<String>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let record = AllocRecord {
            id,
            size,
            category,
            timestamp: self.elapsed(),
            callsite: callsite.into(),
            freed: false,
            free_timestamp: None,
        };
        self.records.insert(id, record);
        self.live_bytes += size;
        self.live_count += 1;
        self.total_allocated += size;
        if self.live_bytes > self.peak_bytes {
            self.peak_bytes = self.live_bytes;
        }
        id
    }

    /// Record a free for the given allocation id. Returns false if id unknown.
    pub fn free(&mut self, id: u64) -> bool {
        let elapsed = self.elapsed();
        if let Some(rec) = self.records.get_mut(&id) {
            if rec.freed {
                return false;
            }
            rec.freed = true;
            rec.free_timestamp = Some(elapsed);
            self.live_bytes = self.live_bytes.saturating_sub(rec.size);
            self.live_count = self.live_count.saturating_sub(1);
            self.total_freed += rec.size;
            true
        } else {
            false
        }
    }

    /// Take a snapshot of current memory state.
    pub fn snapshot(&mut self) {
        let snap = MemorySnapshot {
            timestamp: self.elapsed(),
            live_bytes: self.live_bytes,
            live_count: self.live_count,
            total_allocated: self.total_allocated,
            total_freed: self.total_freed,
        };
        self.snapshots.push(snap);
    }

    /// Current live memory in bytes.
    pub fn live_bytes(&self) -> usize {
        self.live_bytes
    }

    /// Peak memory usage in bytes.
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }

    /// Number of live allocations.
    pub fn live_count(&self) -> u64 {
        self.live_count
    }

    /// Total bytes ever allocated.
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Allocations that were never freed (potential leaks).
    pub fn live_allocations(&self) -> Vec<&AllocRecord> {
        self.records.values().filter(|r| r.is_live()).collect()
    }

    /// Per-category breakdown of live bytes.
    pub fn category_breakdown(&self) -> HashMap<String, usize> {
        let mut map: HashMap<String, usize> = HashMap::new();
        for rec in self.records.values().filter(|r| r.is_live()) {
            *map.entry(rec.category.to_string()).or_default() += rec.size;
        }
        map
    }

    /// Fragmentation estimate: proportion of peak that is no longer live.
    #[allow(clippy::cast_precision_loss)]
    pub fn fragmentation(&self) -> f64 {
        if self.peak_bytes == 0 {
            return 0.0;
        }
        let unused = self.peak_bytes.saturating_sub(self.live_bytes);
        unused as f64 / self.peak_bytes as f64
    }

    /// Average lifetime of freed allocations.
    pub fn avg_lifetime(&self) -> Duration {
        let freed: Vec<Duration> = self.records.values().filter_map(|r| r.lifetime()).collect();
        if freed.is_empty() {
            return Duration::ZERO;
        }
        let total_ns: u128 = freed.iter().map(|d| d.as_nanos()).sum();
        Duration::from_nanos((total_ns / freed.len() as u128) as u64)
    }

    /// All snapshots taken so far.
    pub fn snapshots(&self) -> &[MemorySnapshot] {
        &self.snapshots
    }
}

impl Default for MemoryProfiler {
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

    #[test]
    fn test_new_profiler_zero_state() {
        let p = MemoryProfiler::new();
        assert_eq!(p.live_bytes(), 0);
        assert_eq!(p.peak_bytes(), 0);
        assert_eq!(p.live_count(), 0);
        assert_eq!(p.total_allocated(), 0);
    }

    #[test]
    fn test_alloc_increases_live() {
        let mut p = MemoryProfiler::new();
        p.alloc(1024, AllocCategory::VideoFrame, "test");
        assert_eq!(p.live_bytes(), 1024);
        assert_eq!(p.live_count(), 1);
        assert_eq!(p.total_allocated(), 1024);
    }

    #[test]
    fn test_free_decreases_live() {
        let mut p = MemoryProfiler::new();
        let id = p.alloc(512, AllocCategory::AudioBuffer, "buf");
        assert!(p.free(id));
        assert_eq!(p.live_bytes(), 0);
        assert_eq!(p.live_count(), 0);
    }

    #[test]
    fn test_double_free_returns_false() {
        let mut p = MemoryProfiler::new();
        let id = p.alloc(100, AllocCategory::General, "g");
        p.free(id);
        assert!(!p.free(id));
    }

    #[test]
    fn test_free_unknown_id_returns_false() {
        let mut p = MemoryProfiler::new();
        assert!(!p.free(999));
    }

    #[test]
    fn test_peak_tracks_maximum() {
        let mut p = MemoryProfiler::new();
        let id1 = p.alloc(1000, AllocCategory::General, "a");
        p.alloc(2000, AllocCategory::General, "b");
        assert_eq!(p.peak_bytes(), 3000);
        p.free(id1);
        // peak should NOT decrease
        assert_eq!(p.peak_bytes(), 3000);
    }

    #[test]
    fn test_live_allocations_count() {
        let mut p = MemoryProfiler::new();
        let id1 = p.alloc(100, AllocCategory::General, "a");
        p.alloc(200, AllocCategory::General, "b");
        p.free(id1);
        assert_eq!(p.live_allocations().len(), 1);
    }

    #[test]
    fn test_snapshot_captured() {
        let mut p = MemoryProfiler::new();
        p.alloc(500, AllocCategory::Metadata, "m");
        p.snapshot();
        assert_eq!(p.snapshots().len(), 1);
        assert_eq!(p.snapshots()[0].live_bytes, 500);
    }

    #[test]
    fn test_category_breakdown() {
        let mut p = MemoryProfiler::new();
        p.alloc(300, AllocCategory::VideoFrame, "v");
        p.alloc(100, AllocCategory::AudioBuffer, "a");
        let bd = p.category_breakdown();
        assert_eq!(bd.get("VideoFrame"), Some(&300));
        assert_eq!(bd.get("AudioBuffer"), Some(&100));
    }

    #[test]
    fn test_fragmentation_zero_when_all_live() {
        let mut p = MemoryProfiler::new();
        p.alloc(1000, AllocCategory::General, "x");
        assert_eq!(p.fragmentation(), 0.0);
    }

    #[test]
    fn test_fragmentation_nonzero_after_free() {
        let mut p = MemoryProfiler::new();
        let id = p.alloc(1000, AllocCategory::General, "x");
        p.free(id);
        assert!(p.fragmentation() > 0.0);
    }

    #[test]
    fn test_avg_lifetime_no_frees() {
        let p = MemoryProfiler::new();
        assert_eq!(p.avg_lifetime(), Duration::ZERO);
    }

    #[test]
    fn test_alloc_record_is_live() {
        let mut p = MemoryProfiler::new();
        let id = p.alloc(64, AllocCategory::BitstreamData, "bs");
        let rec = p.records.get(&id).expect("should succeed in test");
        assert!(rec.is_live());
        assert!(rec.lifetime().is_none());
    }

    #[test]
    fn test_memory_snapshot_fragmentation_estimate() {
        let snap = MemorySnapshot {
            timestamp: Duration::ZERO,
            live_bytes: 500,
            live_count: 1,
            total_allocated: 1000,
            total_freed: 500,
        };
        let frag = snap.fragmentation_estimate(1000);
        assert!((frag - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_alloc_category_display() {
        assert_eq!(AllocCategory::VideoFrame.to_string(), "VideoFrame");
        assert_eq!(
            AllocCategory::Custom("x".to_string()).to_string(),
            "Custom(x)"
        );
    }
}
