//! Memory allocation tracking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Information about a memory allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationInfo {
    /// Size in bytes.
    pub size: usize,

    /// Timestamp of allocation.
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,

    /// Location (function/file).
    pub location: Option<String>,

    /// Stack trace.
    pub stack_trace: Vec<String>,

    /// Whether this allocation has been freed.
    pub freed: bool,
}

impl AllocationInfo {
    /// Create a new allocation info.
    pub fn new(size: usize) -> Self {
        Self {
            size,
            timestamp: Instant::now(),
            location: None,
            stack_trace: Vec::new(),
            freed: false,
        }
    }

    /// Set the location.
    pub fn with_location(mut self, location: String) -> Self {
        self.location = Some(location);
        self
    }

    /// Add a stack frame.
    pub fn add_stack_frame(&mut self, frame: String) {
        self.stack_trace.push(frame);
    }

    /// Mark as freed.
    pub fn mark_freed(&mut self) {
        self.freed = true;
    }

    /// Get the age of this allocation.
    pub fn age(&self) -> std::time::Duration {
        self.timestamp.elapsed()
    }
}

/// Memory allocation tracker.
#[derive(Debug)]
pub struct MemoryTracker {
    allocations: HashMap<u64, AllocationInfo>,
    next_id: u64,
    total_allocated: usize,
    total_freed: usize,
    peak_memory: usize,
    current_memory: usize,
    allocation_count: u64,
    free_count: u64,
    running: bool,
}

impl MemoryTracker {
    /// Create a new memory tracker.
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            next_id: 0,
            total_allocated: 0,
            total_freed: 0,
            peak_memory: 0,
            current_memory: 0,
            allocation_count: 0,
            free_count: 0,
            running: false,
        }
    }

    /// Start tracking.
    pub fn start(&mut self) {
        self.running = true;
        self.allocations.clear();
        self.total_allocated = 0;
        self.total_freed = 0;
        self.peak_memory = 0;
        self.current_memory = 0;
        self.allocation_count = 0;
        self.free_count = 0;
    }

    /// Stop tracking.
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Track an allocation.
    pub fn track_allocation(&mut self, size: usize, location: Option<String>) -> u64 {
        if !self.running {
            return 0;
        }

        let id = self.next_id;
        self.next_id += 1;

        let mut info = AllocationInfo::new(size);
        if let Some(loc) = location {
            info = info.with_location(loc);
        }

        self.allocations.insert(id, info);
        self.total_allocated += size;
        self.current_memory += size;
        self.allocation_count += 1;

        if self.current_memory > self.peak_memory {
            self.peak_memory = self.current_memory;
        }

        id
    }

    /// Track a deallocation.
    pub fn track_free(&mut self, id: u64) {
        if !self.running {
            return;
        }

        if let Some(info) = self.allocations.get_mut(&id) {
            if !info.freed {
                info.mark_freed();
                self.total_freed += info.size;
                self.current_memory = self.current_memory.saturating_sub(info.size);
                self.free_count += 1;
            }
        }
    }

    /// Get current memory usage.
    pub fn current_memory(&self) -> usize {
        self.current_memory
    }

    /// Get peak memory usage.
    pub fn peak_memory(&self) -> usize {
        self.peak_memory
    }

    /// Get total allocated bytes.
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Get total freed bytes.
    pub fn total_freed(&self) -> usize {
        self.total_freed
    }

    /// Get allocation count.
    pub fn allocation_count(&self) -> u64 {
        self.allocation_count
    }

    /// Get free count.
    pub fn free_count(&self) -> u64 {
        self.free_count
    }

    /// Get all allocations.
    pub fn allocations(&self) -> &HashMap<u64, AllocationInfo> {
        &self.allocations
    }

    /// Get active allocations (not freed).
    pub fn active_allocations(&self) -> Vec<(&u64, &AllocationInfo)> {
        self.allocations
            .iter()
            .filter(|(_, info)| !info.freed)
            .collect()
    }

    /// Generate a summary report.
    pub fn summary(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!(
            "  Current Memory: {} bytes\n",
            self.current_memory
        ));
        report.push_str(&format!("  Peak Memory: {} bytes\n", self.peak_memory));
        report.push_str(&format!(
            "  Total Allocated: {} bytes\n",
            self.total_allocated
        ));
        report.push_str(&format!("  Total Freed: {} bytes\n", self.total_freed));
        report.push_str(&format!("  Allocations: {}\n", self.allocation_count));
        report.push_str(&format!("  Frees: {}\n", self.free_count));

        let active = self.active_allocations();
        report.push_str(&format!("  Active Allocations: {}\n", active.len()));

        report
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_info() {
        let info = AllocationInfo::new(1024);
        assert_eq!(info.size, 1024);
        assert!(!info.freed);

        let info = info.with_location("test.rs:42".to_string());
        assert_eq!(info.location.as_deref(), Some("test.rs:42"));
    }

    #[test]
    fn test_memory_tracker_basic() {
        let mut tracker = MemoryTracker::new();
        tracker.start();

        let id1 = tracker.track_allocation(1024, None);
        assert_eq!(tracker.current_memory(), 1024);
        assert_eq!(tracker.allocation_count(), 1);

        let id2 = tracker.track_allocation(2048, None);
        assert_eq!(tracker.current_memory(), 3072);
        assert_eq!(tracker.allocation_count(), 2);

        tracker.track_free(id1);
        assert_eq!(tracker.current_memory(), 2048);
        assert_eq!(tracker.free_count(), 1);

        tracker.track_free(id2);
        assert_eq!(tracker.current_memory(), 0);
        assert_eq!(tracker.free_count(), 2);

        tracker.stop();
    }

    #[test]
    fn test_memory_tracker_peak() {
        let mut tracker = MemoryTracker::new();
        tracker.start();

        tracker.track_allocation(1000, None);
        tracker.track_allocation(2000, None);
        assert_eq!(tracker.peak_memory(), 3000);

        let id = tracker.track_allocation(3000, None);
        assert_eq!(tracker.peak_memory(), 6000);

        tracker.track_free(id);
        assert_eq!(tracker.current_memory(), 3000);
        assert_eq!(tracker.peak_memory(), 6000); // Peak remains

        tracker.stop();
    }

    #[test]
    fn test_active_allocations() {
        let mut tracker = MemoryTracker::new();
        tracker.start();

        let id1 = tracker.track_allocation(1024, None);
        let _id2 = tracker.track_allocation(2048, None);
        tracker.track_free(id1);

        let active = tracker.active_allocations();
        assert_eq!(active.len(), 1);

        tracker.stop();
    }

    #[test]
    fn test_memory_tracker_summary() {
        let mut tracker = MemoryTracker::new();
        tracker.start();
        tracker.track_allocation(1024, None);
        tracker.stop();

        let summary = tracker.summary();
        assert!(summary.contains("Current Memory"));
        assert!(summary.contains("Peak Memory"));
        assert!(summary.contains("Allocations"));
    }
}
