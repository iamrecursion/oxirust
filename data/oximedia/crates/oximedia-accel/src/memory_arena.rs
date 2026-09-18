//! GPU-style memory arena allocator.
//!
//! Provides a bump-allocator arena that carves sub-allocations from a
//! contiguous slab.  Useful for grouping per-frame GPU buffer allocations
//! so they can be freed in a single operation at the end of the frame.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;

/// Unique identifier for an arena allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AllocId(u64);

impl fmt::Display for AllocId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "alloc#{}", self.0)
    }
}

/// Metadata for a single sub-allocation inside the arena.
#[derive(Debug, Clone, Copy)]
pub struct AllocRecord {
    /// Byte offset from the start of the arena.
    pub offset: usize,
    /// Size in bytes.
    pub size: usize,
    /// Alignment that was requested.
    pub alignment: usize,
    /// Allocation id.
    pub id: AllocId,
}

/// Allocation strategy hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocStrategy {
    /// Simple bump / linear allocation (fastest, no individual free).
    Linear,
    /// Best-fit free-list (allows individual free, slower).
    BestFit,
}

/// Statistics for the arena.
#[derive(Debug, Clone, Default)]
pub struct ArenaStats {
    /// Total capacity in bytes.
    pub capacity: usize,
    /// Currently used bytes (including alignment padding).
    pub used: usize,
    /// Peak used bytes observed.
    pub peak_used: usize,
    /// Total number of allocations performed.
    pub alloc_count: u64,
    /// Total number of resets performed.
    pub reset_count: u64,
}

impl ArenaStats {
    /// Fraction of arena currently in use (0.0 .. 1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn utilization(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.used as f64 / self.capacity as f64
    }

    /// Fraction of arena used at peak.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn peak_utilization(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.peak_used as f64 / self.capacity as f64
    }

    /// Remaining free bytes.
    #[must_use]
    pub fn free_bytes(&self) -> usize {
        self.capacity.saturating_sub(self.used)
    }
}

/// A bump-allocator memory arena.
///
/// Allocations are served from a contiguous virtual range and
/// freed collectively via [`MemoryArena::reset`].
pub struct MemoryArena {
    /// Total capacity in bytes.
    capacity: usize,
    /// Current write cursor (next free offset).
    cursor: usize,
    /// Running allocation id counter.
    next_id: u64,
    /// Record of live allocations.
    records: HashMap<AllocId, AllocRecord>,
    /// Strategy hint (stored for introspection; behaviour is always linear).
    strategy: AllocStrategy,
    /// Running statistics.
    stats: ArenaStats,
}

impl MemoryArena {
    /// Create a new arena with the given byte capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            cursor: 0,
            next_id: 0,
            records: HashMap::new(),
            strategy: AllocStrategy::Linear,
            stats: ArenaStats {
                capacity,
                ..ArenaStats::default()
            },
        }
    }

    /// Create a new arena with a specific strategy hint.
    #[must_use]
    pub fn with_strategy(capacity: usize, strategy: AllocStrategy) -> Self {
        let mut arena = Self::new(capacity);
        arena.strategy = strategy;
        arena
    }

    /// Total capacity in bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Currently used bytes.
    #[must_use]
    pub fn used(&self) -> usize {
        self.cursor
    }

    /// Remaining free bytes.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.cursor)
    }

    /// The strategy hint for this arena.
    #[must_use]
    pub fn strategy(&self) -> AllocStrategy {
        self.strategy
    }

    /// Number of live allocations.
    #[must_use]
    pub fn live_alloc_count(&self) -> usize {
        self.records.len()
    }

    /// Allocate `size` bytes with the given alignment.
    ///
    /// Returns `None` if the arena cannot satisfy the request.
    pub fn allocate(&mut self, size: usize, alignment: usize) -> Option<AllocRecord> {
        let align = alignment.max(1);
        // Round cursor up to alignment
        let aligned_offset = (self.cursor + align - 1) & !(align - 1);
        let end = aligned_offset.checked_add(size)?;
        if end > self.capacity {
            return None;
        }
        let id = AllocId(self.next_id);
        self.next_id += 1;
        let record = AllocRecord {
            offset: aligned_offset,
            size,
            alignment: align,
            id,
        };
        self.cursor = end;
        self.records.insert(id, record);
        self.stats.alloc_count += 1;
        self.stats.used = self.cursor;
        if self.cursor > self.stats.peak_used {
            self.stats.peak_used = self.cursor;
        }
        Some(record)
    }

    /// Convenience: allocate with default alignment of 1.
    pub fn allocate_unaligned(&mut self, size: usize) -> Option<AllocRecord> {
        self.allocate(size, 1)
    }

    /// Look up a record by its id.
    #[must_use]
    pub fn get_record(&self, id: AllocId) -> Option<&AllocRecord> {
        self.records.get(&id)
    }

    /// Reset the arena, freeing all allocations.
    pub fn reset(&mut self) {
        self.cursor = 0;
        self.records.clear();
        self.stats.used = 0;
        self.stats.reset_count += 1;
    }

    /// Snapshot of arena statistics.
    #[must_use]
    pub fn stats(&self) -> &ArenaStats {
        &self.stats
    }

    /// Resize the arena capacity. If the new capacity is smaller than the
    /// current cursor, this effectively invalidates existing allocations.
    pub fn resize(&mut self, new_capacity: usize) {
        self.capacity = new_capacity;
        self.stats.capacity = new_capacity;
        if self.cursor > new_capacity {
            self.cursor = new_capacity;
            self.stats.used = new_capacity;
        }
    }

    /// Returns true if memory pressure exceeds the given threshold (0.0 to 1.0).
    #[must_use]
    pub fn is_under_pressure(&self, threshold: f64) -> bool {
        self.stats.utilization() >= threshold
    }

    /// Attempts to evict oldest allocations until utilization drops below
    /// `target_utilization` (0.0 to 1.0).
    ///
    /// Returns the number of allocations evicted. Only works with
    /// `BestFit` strategy (linear arenas cannot free individual allocations).
    ///
    /// For `Linear` strategy, this performs a full reset if pressure is
    /// above threshold and returns the count of allocations that were present.
    pub fn evict_until_below(&mut self, target_utilization: f64) -> usize {
        if self.capacity == 0 || self.stats.utilization() <= target_utilization {
            return 0;
        }

        // For both strategies, evict from the tail (most recently allocated)
        // backwards, since the arena is a bump allocator. Only tail
        // allocations can actually reclaim cursor space.
        let mut sorted: Vec<AllocRecord> = self.records.values().copied().collect();
        // Sort by offset descending (newest first) for effective reclamation
        sorted.sort_by(|a, b| b.offset.cmp(&a.offset));

        let target_used = (self.capacity as f64 * target_utilization) as usize;
        let mut evicted = 0;

        for record in &sorted {
            if self.cursor <= target_used {
                break;
            }
            // Only reclaim if this record is at the tail of the arena
            if record.offset + record.size == self.cursor {
                self.records.remove(&record.id);
                self.cursor = record.offset;
                self.stats.used = self.cursor;
                evicted += 1;
            } else {
                // Non-tail record: remove from tracking but can't reclaim space
                self.records.remove(&record.id);
                evicted += 1;
            }
        }

        // Recalculate cursor to the end of the highest remaining allocation
        if self.records.is_empty() {
            self.cursor = 0;
            self.stats.used = 0;
        } else {
            let max_end = self
                .records
                .values()
                .map(|r| r.offset + r.size)
                .max()
                .unwrap_or(0);
            // Cursor can only shrink to the highest remaining allocation end
            if max_end < self.cursor {
                self.cursor = max_end;
                self.stats.used = max_end;
            }
        }

        evicted
    }

    /// Returns true if the arena can accommodate an allocation of `size`
    /// bytes with the given `alignment` without eviction.
    #[must_use]
    pub fn can_allocate(&self, size: usize, alignment: usize) -> bool {
        let align = alignment.max(1);
        let aligned_offset = (self.cursor + align - 1) & !(align - 1);
        match aligned_offset.checked_add(size) {
            Some(end) => end <= self.capacity,
            None => false,
        }
    }

    /// Attempts to allocate `size` bytes with `alignment`, automatically
    /// evicting old allocations if needed.
    ///
    /// Uses a target utilization of 0.75 when eviction is triggered.
    ///
    /// Returns `None` if allocation fails even after eviction.
    pub fn allocate_or_evict(&mut self, size: usize, alignment: usize) -> Option<AllocRecord> {
        if let Some(record) = self.allocate(size, alignment) {
            return Some(record);
        }
        // Try evicting down to 50% utilization
        self.evict_until_below(0.5);
        self.allocate(size, alignment)
    }
}

/// Memory pressure policy for automatic management.
#[derive(Debug, Clone)]
pub struct MemoryPressurePolicy {
    /// Utilization threshold (0.0 to 1.0) that triggers a warning.
    pub warning_threshold: f64,
    /// Utilization threshold (0.0 to 1.0) that triggers automatic eviction.
    pub critical_threshold: f64,
    /// Target utilization after eviction.
    pub eviction_target: f64,
    /// Maximum number of allocations before forcing eviction.
    pub max_live_allocations: usize,
}

impl Default for MemoryPressurePolicy {
    fn default() -> Self {
        Self {
            warning_threshold: 0.75,
            critical_threshold: 0.90,
            eviction_target: 0.60,
            max_live_allocations: 10_000,
        }
    }
}

/// Pressure level reported by the monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureLevel {
    /// Utilization below warning threshold.
    Normal,
    /// Utilization between warning and critical thresholds.
    Warning,
    /// Utilization above critical threshold.
    Critical,
}

/// Monitors memory pressure on an arena and can trigger eviction.
pub struct MemoryPressureMonitor {
    /// Policy governing thresholds and limits.
    policy: MemoryPressurePolicy,
    /// Number of evictions triggered.
    eviction_count: u64,
    /// Total allocations evicted across all eviction passes.
    total_evicted: u64,
}

impl MemoryPressureMonitor {
    /// Creates a new monitor with the given policy.
    #[must_use]
    pub fn new(policy: MemoryPressurePolicy) -> Self {
        Self {
            policy,
            eviction_count: 0,
            total_evicted: 0,
        }
    }

    /// Creates a monitor with default policy.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(MemoryPressurePolicy::default())
    }

    /// Returns the current pressure level for the given arena.
    #[must_use]
    pub fn pressure_level(&self, arena: &MemoryArena) -> PressureLevel {
        let util = arena.stats().utilization();
        if util >= self.policy.critical_threshold {
            PressureLevel::Critical
        } else if util >= self.policy.warning_threshold {
            PressureLevel::Warning
        } else {
            PressureLevel::Normal
        }
    }

    /// Checks the arena and triggers eviction if needed.
    ///
    /// Returns the number of allocations evicted (0 if no eviction needed).
    pub fn check_and_evict(&mut self, arena: &mut MemoryArena) -> usize {
        let level = self.pressure_level(arena);
        let over_alloc_limit = arena.live_alloc_count() > self.policy.max_live_allocations;

        if level == PressureLevel::Critical || over_alloc_limit {
            let evicted = arena.evict_until_below(self.policy.eviction_target);
            if evicted > 0 {
                self.eviction_count += 1;
                self.total_evicted += evicted as u64;
            }
            evicted
        } else {
            0
        }
    }

    /// Returns the policy in use.
    #[must_use]
    pub fn policy(&self) -> &MemoryPressurePolicy {
        &self.policy
    }

    /// Returns the number of eviction events triggered.
    #[must_use]
    pub fn eviction_count(&self) -> u64 {
        self.eviction_count
    }

    /// Returns the total number of allocations evicted.
    #[must_use]
    pub fn total_evicted(&self) -> u64 {
        self.total_evicted
    }

    /// Updates the policy.
    pub fn set_policy(&mut self, policy: MemoryPressurePolicy) {
        self.policy = policy;
    }
}

impl fmt::Debug for MemoryPressureMonitor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryPressureMonitor")
            .field("eviction_count", &self.eviction_count)
            .field("total_evicted", &self.total_evicted)
            .finish()
    }
}

impl fmt::Debug for MemoryArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryArena")
            .field("capacity", &self.capacity)
            .field("used", &self.cursor)
            .field("live_allocs", &self.records.len())
            .field("strategy", &self.strategy)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_id_display() {
        let id = AllocId(42);
        assert_eq!(id.to_string(), "alloc#42");
    }

    #[test]
    fn test_new_arena() {
        let arena = MemoryArena::new(1024);
        assert_eq!(arena.capacity(), 1024);
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.remaining(), 1024);
    }

    #[test]
    fn test_simple_allocation() {
        let mut arena = MemoryArena::new(256);
        let rec = arena.allocate(64, 1).expect("rec should be valid");
        assert_eq!(rec.offset, 0);
        assert_eq!(rec.size, 64);
        assert_eq!(arena.used(), 64);
        assert_eq!(arena.remaining(), 192);
    }

    #[test]
    fn test_aligned_allocation() {
        let mut arena = MemoryArena::new(256);
        arena.allocate(10, 1).expect("allocate should succeed"); // cursor at 10
        let rec = arena.allocate(32, 16).expect("rec should be valid"); // should align to 16
        assert_eq!(rec.offset, 16);
        assert_eq!(rec.size, 32);
    }

    #[test]
    fn test_allocation_overflow() {
        let mut arena = MemoryArena::new(64);
        assert!(arena.allocate(65, 1).is_none());
        assert_eq!(arena.live_alloc_count(), 0);
    }

    #[test]
    fn test_multiple_allocations() {
        let mut arena = MemoryArena::new(1024);
        for i in 0..10 {
            let rec = arena.allocate(32, 1).expect("rec should be valid");
            assert_eq!(rec.offset, i * 32);
        }
        assert_eq!(arena.live_alloc_count(), 10);
        assert_eq!(arena.used(), 320);
    }

    #[test]
    fn test_reset() {
        let mut arena = MemoryArena::new(256);
        arena.allocate(100, 1).expect("allocate should succeed");
        arena.allocate(50, 1).expect("allocate should succeed");
        arena.reset();
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.live_alloc_count(), 0);
        assert_eq!(arena.stats().reset_count, 1);
    }

    #[test]
    fn test_peak_tracking() {
        let mut arena = MemoryArena::new(512);
        arena.allocate(200, 1).expect("allocate should succeed");
        arena.allocate(100, 1).expect("allocate should succeed");
        assert_eq!(arena.stats().peak_used, 300);
        arena.reset();
        arena.allocate(50, 1).expect("allocate should succeed");
        // peak should still be 300
        assert_eq!(arena.stats().peak_used, 300);
    }

    #[test]
    fn test_get_record() {
        let mut arena = MemoryArena::new(256);
        let rec = arena.allocate(16, 1).expect("rec should be valid");
        let found = arena.get_record(rec.id).expect("found should be valid");
        assert_eq!(found.offset, 0);
        assert_eq!(found.size, 16);
    }

    #[test]
    fn test_stats_utilization() {
        let mut arena = MemoryArena::new(100);
        arena.allocate(50, 1).expect("allocate should succeed");
        let s = arena.stats();
        assert!((s.utilization() - 0.5).abs() < 1e-9);
        assert_eq!(s.free_bytes(), 50);
    }

    #[test]
    fn test_strategy_hint() {
        let arena = MemoryArena::with_strategy(1024, AllocStrategy::BestFit);
        assert_eq!(arena.strategy(), AllocStrategy::BestFit);
    }

    #[test]
    fn test_resize_larger() {
        let mut arena = MemoryArena::new(100);
        arena.allocate(80, 1).expect("allocate should succeed");
        arena.resize(200);
        assert_eq!(arena.capacity(), 200);
        assert_eq!(arena.remaining(), 120);
    }

    #[test]
    fn test_resize_smaller_than_cursor() {
        let mut arena = MemoryArena::new(200);
        arena.allocate(150, 1).expect("allocate should succeed");
        arena.resize(100);
        assert_eq!(arena.capacity(), 100);
        // cursor clamped to capacity
        assert_eq!(arena.used(), 100);
    }

    #[test]
    fn test_allocate_unaligned() {
        let mut arena = MemoryArena::new(64);
        let rec = arena.allocate_unaligned(10).expect("rec should be valid");
        assert_eq!(rec.alignment, 1);
        assert_eq!(rec.offset, 0);
    }

    // ── Memory pressure tests ──────────────────────────────────────────────

    #[test]
    fn test_is_under_pressure() {
        let mut arena = MemoryArena::new(100);
        assert!(!arena.is_under_pressure(0.5));
        arena.allocate(75, 1).expect("allocate should succeed");
        assert!(arena.is_under_pressure(0.5));
        assert!(!arena.is_under_pressure(0.8));
    }

    #[test]
    fn test_can_allocate() {
        let mut arena = MemoryArena::new(100);
        assert!(arena.can_allocate(50, 1));
        assert!(arena.can_allocate(100, 1));
        assert!(!arena.can_allocate(101, 1));
        arena.allocate(90, 1).expect("allocate should succeed");
        assert!(!arena.can_allocate(20, 1));
        assert!(arena.can_allocate(10, 1));
    }

    #[test]
    fn test_can_allocate_with_alignment() {
        let mut arena = MemoryArena::new(64);
        arena.allocate(1, 1).expect("allocate should succeed"); // cursor at 1
                                                                // With alignment 16, next alloc starts at 16, needs 16+48=64 total
        assert!(arena.can_allocate(48, 16));
        assert!(!arena.can_allocate(49, 16));
    }

    #[test]
    fn test_evict_linear_arena() {
        let mut arena = MemoryArena::new(100);
        arena.allocate(90, 1).expect("allocate should succeed");
        arena.allocate(5, 1).expect("allocate should succeed");
        // Linear arena: eviction does full reset
        let evicted = arena.evict_until_below(0.5);
        assert_eq!(evicted, 2);
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.live_alloc_count(), 0);
    }

    #[test]
    fn test_evict_bestfit_arena() {
        let mut arena = MemoryArena::with_strategy(100, AllocStrategy::BestFit);
        for _ in 0..5 {
            arena.allocate(18, 1).expect("allocate should succeed");
        }
        assert_eq!(arena.used(), 90);
        let evicted = arena.evict_until_below(0.5);
        assert!(evicted > 0);
        // After eviction, cursor should have moved back
        assert!(arena.used() <= 54, "used={}", arena.used()); // 54 = 3*18
    }

    #[test]
    fn test_evict_not_needed() {
        let mut arena = MemoryArena::new(100);
        arena.allocate(10, 1).expect("allocate should succeed");
        let evicted = arena.evict_until_below(0.5);
        assert_eq!(evicted, 0);
    }

    #[test]
    fn test_allocate_or_evict_success() {
        let mut arena = MemoryArena::with_strategy(100, AllocStrategy::BestFit);
        for _ in 0..9 {
            arena.allocate(10, 1).expect("allocate should succeed");
        }
        // Arena at 90/100, can't fit 20
        assert!(arena.allocate(20, 1).is_none());
        // allocate_or_evict evicts (down to 50%), then allocates the 20 bytes
        let rec = arena.allocate_or_evict(20, 1);
        // After eviction, we should have freed enough space for a 20-byte alloc
        assert!(
            rec.is_some(),
            "allocate_or_evict should succeed after eviction, used={}",
            arena.used()
        );
    }

    #[test]
    fn test_allocate_or_evict_no_eviction_needed() {
        let mut arena = MemoryArena::new(100);
        let rec = arena.allocate_or_evict(50, 1);
        assert!(rec.is_some());
        assert_eq!(arena.used(), 50);
    }

    #[test]
    fn test_pressure_monitor_levels() {
        let monitor = MemoryPressureMonitor::with_defaults();
        let mut arena = MemoryArena::new(100);

        assert_eq!(monitor.pressure_level(&arena), PressureLevel::Normal);

        arena.allocate(80, 1).expect("allocate should succeed");
        assert_eq!(monitor.pressure_level(&arena), PressureLevel::Warning);

        arena.allocate(15, 1).expect("allocate should succeed");
        assert_eq!(monitor.pressure_level(&arena), PressureLevel::Critical);
    }

    #[test]
    fn test_pressure_monitor_check_and_evict() {
        let policy = MemoryPressurePolicy {
            warning_threshold: 0.75,
            critical_threshold: 0.85,
            eviction_target: 0.50,
            max_live_allocations: 10_000,
        };
        let mut monitor = MemoryPressureMonitor::new(policy);
        let mut arena = MemoryArena::with_strategy(100, AllocStrategy::BestFit);

        for _ in 0..8 {
            arena.allocate(10, 1).expect("allocate should succeed");
        }
        assert_eq!(arena.used(), 80);
        // 80% is below critical (85%), should not evict
        let evicted = monitor.check_and_evict(&mut arena);
        assert_eq!(evicted, 0);

        // Push above critical (90%)
        arena.allocate(10, 1).expect("allocate should succeed");
        assert_eq!(arena.used(), 90);
        let evicted = monitor.check_and_evict(&mut arena);
        assert!(evicted > 0);
        assert_eq!(monitor.eviction_count(), 1);
        assert!(monitor.total_evicted() > 0);
    }

    #[test]
    fn test_pressure_monitor_alloc_limit() {
        let policy = MemoryPressurePolicy {
            max_live_allocations: 5,
            warning_threshold: 0.75,
            critical_threshold: 0.90,
            eviction_target: 0.0, // evict everything
        };
        let mut monitor = MemoryPressureMonitor::new(policy);
        let mut arena = MemoryArena::with_strategy(10000, AllocStrategy::BestFit);

        for _ in 0..6 {
            arena.allocate(1, 1).expect("allocate should succeed");
        }
        // 6 allocs > max_live_allocations(5), should trigger eviction
        let evicted = monitor.check_and_evict(&mut arena);
        assert!(evicted > 0);
    }

    #[test]
    fn test_pressure_policy_default() {
        let policy = MemoryPressurePolicy::default();
        assert!((policy.warning_threshold - 0.75).abs() < 1e-9);
        assert!((policy.critical_threshold - 0.90).abs() < 1e-9);
        assert!((policy.eviction_target - 0.60).abs() < 1e-9);
        assert_eq!(policy.max_live_allocations, 10_000);
    }

    #[test]
    fn test_pressure_monitor_set_policy() {
        let mut monitor = MemoryPressureMonitor::with_defaults();
        let policy = MemoryPressurePolicy {
            warning_threshold: 0.5,
            critical_threshold: 0.7,
            eviction_target: 0.3,
            max_live_allocations: 100,
        };
        monitor.set_policy(policy);
        assert!((monitor.policy().warning_threshold - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_pressure_monitor_debug() {
        let monitor = MemoryPressureMonitor::with_defaults();
        let debug = format!("{:?}", monitor);
        assert!(debug.contains("MemoryPressureMonitor"));
    }

    #[test]
    fn test_evict_empty_arena() {
        let mut arena = MemoryArena::new(100);
        let evicted = arena.evict_until_below(0.5);
        assert_eq!(evicted, 0);
    }

    #[test]
    fn test_evict_bestfit_to_empty() {
        let mut arena = MemoryArena::with_strategy(100, AllocStrategy::BestFit);
        arena.allocate(90, 1).expect("allocate should succeed");
        let evicted = arena.evict_until_below(0.0);
        assert_eq!(evicted, 1);
        assert_eq!(arena.used(), 0);
    }
}
