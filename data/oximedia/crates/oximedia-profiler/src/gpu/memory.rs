//! GPU memory tracking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// GPU memory statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryStats {
    /// Total GPU memory in bytes.
    pub total_memory: u64,

    /// Used GPU memory in bytes.
    pub used_memory: u64,

    /// Free GPU memory in bytes.
    pub free_memory: u64,

    /// Number of allocations.
    pub allocation_count: u64,

    /// Largest allocation size.
    pub largest_allocation: u64,

    /// Average allocation size.
    pub avg_allocation_size: f64,

    /// Memory usage percentage (0.0-100.0).
    pub usage_percentage: f64,
}

/// GPU memory allocation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuMemoryType {
    /// Buffer memory.
    Buffer,

    /// Texture memory.
    Texture,

    /// Render target memory.
    RenderTarget,

    /// Uniform buffer.
    Uniform,

    /// Staging buffer.
    Staging,
}

/// GPU memory allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAllocation {
    /// Allocation ID.
    pub id: u64,

    /// Size in bytes.
    pub size: u64,

    /// Memory type.
    pub memory_type: GpuMemoryType,

    /// Name/label.
    pub name: Option<String>,
}

/// GPU memory tracker.
#[derive(Debug)]
pub struct GpuMemoryTracker {
    allocations: HashMap<u64, GpuAllocation>,
    next_id: u64,
    total_memory: u64,
    used_memory: u64,
    allocations_by_type: HashMap<GpuMemoryType, Vec<u64>>,
}

impl GpuMemoryTracker {
    /// Create a new GPU memory tracker.
    pub fn new(total_memory: u64) -> Self {
        Self {
            allocations: HashMap::new(),
            next_id: 0,
            total_memory,
            used_memory: 0,
            allocations_by_type: HashMap::new(),
        }
    }

    /// Allocate GPU memory.
    pub fn allocate(
        &mut self,
        size: u64,
        memory_type: GpuMemoryType,
        name: Option<String>,
    ) -> Option<u64> {
        if self.used_memory + size > self.total_memory {
            return None;
        }

        let id = self.next_id;
        self.next_id += 1;

        let allocation = GpuAllocation {
            id,
            size,
            memory_type,
            name,
        };

        self.allocations.insert(id, allocation);
        self.used_memory += size;

        self.allocations_by_type
            .entry(memory_type)
            .or_default()
            .push(id);

        Some(id)
    }

    /// Free GPU memory.
    pub fn free(&mut self, id: u64) -> bool {
        if let Some(allocation) = self.allocations.remove(&id) {
            self.used_memory = self.used_memory.saturating_sub(allocation.size);

            if let Some(ids) = self.allocations_by_type.get_mut(&allocation.memory_type) {
                ids.retain(|&x| x != id);
            }

            true
        } else {
            false
        }
    }

    /// Get memory statistics.
    pub fn stats(&self) -> GpuMemoryStats {
        let allocation_count = self.allocations.len() as u64;
        let free_memory = self.total_memory.saturating_sub(self.used_memory);

        let largest_allocation = self.allocations.values().map(|a| a.size).max().unwrap_or(0);

        let avg_allocation_size = if allocation_count > 0 {
            self.used_memory as f64 / allocation_count as f64
        } else {
            0.0
        };

        let usage_percentage = if self.total_memory > 0 {
            (self.used_memory as f64 / self.total_memory as f64) * 100.0
        } else {
            0.0
        };

        GpuMemoryStats {
            total_memory: self.total_memory,
            used_memory: self.used_memory,
            free_memory,
            allocation_count,
            largest_allocation,
            avg_allocation_size,
            usage_percentage,
        }
    }

    /// Get memory usage by type.
    pub fn usage_by_type(&self, memory_type: GpuMemoryType) -> u64 {
        self.allocations_by_type
            .get(&memory_type)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.allocations.get(id))
                    .map(|a| a.size)
                    .sum()
            })
            .unwrap_or(0)
    }

    /// Get total memory.
    pub fn total_memory(&self) -> u64 {
        self.total_memory
    }

    /// Get used memory.
    pub fn used_memory(&self) -> u64 {
        self.used_memory
    }

    /// Get allocation count.
    pub fn allocation_count(&self) -> usize {
        self.allocations.len()
    }

    /// Generate a summary report.
    pub fn summary(&self) -> String {
        let stats = self.stats();
        let mut report = String::new();

        report.push_str(&format!(
            "  Total Memory: {} bytes ({:.2} MB)\n",
            stats.total_memory,
            stats.total_memory as f64 / 1_000_000.0
        ));
        report.push_str(&format!(
            "  Used Memory: {} bytes ({:.2} MB)\n",
            stats.used_memory,
            stats.used_memory as f64 / 1_000_000.0
        ));
        report.push_str(&format!(
            "  Free Memory: {} bytes ({:.2} MB)\n",
            stats.free_memory,
            stats.free_memory as f64 / 1_000_000.0
        ));
        report.push_str(&format!("  Usage: {:.2}%\n", stats.usage_percentage));
        report.push_str(&format!("  Allocations: {}\n", stats.allocation_count));

        report.push_str("\n  Memory by Type:\n");
        for memory_type in [
            GpuMemoryType::Buffer,
            GpuMemoryType::Texture,
            GpuMemoryType::RenderTarget,
            GpuMemoryType::Uniform,
            GpuMemoryType::Staging,
        ] {
            let usage = self.usage_by_type(memory_type);
            if usage > 0 {
                report.push_str(&format!(
                    "    {:?}: {} bytes ({:.2} MB)\n",
                    memory_type,
                    usage,
                    usage as f64 / 1_000_000.0
                ));
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_memory_tracker() {
        let tracker = GpuMemoryTracker::new(1_000_000_000);
        assert_eq!(tracker.total_memory(), 1_000_000_000);
        assert_eq!(tracker.used_memory(), 0);
    }

    #[test]
    fn test_allocate_free() {
        let mut tracker = GpuMemoryTracker::new(1_000_000_000);

        let id1 = tracker.allocate(1000, GpuMemoryType::Buffer, None);
        assert!(id1.is_some());
        assert_eq!(tracker.used_memory(), 1000);

        let id2 = tracker.allocate(2000, GpuMemoryType::Texture, None);
        assert!(id2.is_some());
        assert_eq!(tracker.used_memory(), 3000);

        assert!(tracker.free(id1.expect("should succeed in test")));
        assert_eq!(tracker.used_memory(), 2000);

        assert!(tracker.free(id2.expect("should succeed in test")));
        assert_eq!(tracker.used_memory(), 0);
    }

    #[test]
    fn test_allocation_limit() {
        let mut tracker = GpuMemoryTracker::new(1000);

        let id1 = tracker.allocate(600, GpuMemoryType::Buffer, None);
        assert!(id1.is_some());

        let id2 = tracker.allocate(500, GpuMemoryType::Buffer, None);
        assert!(id2.is_none()); // Exceeds limit

        let id3 = tracker.allocate(400, GpuMemoryType::Buffer, None);
        assert!(id3.is_some());
    }

    #[test]
    fn test_usage_by_type() {
        let mut tracker = GpuMemoryTracker::new(1_000_000_000);

        tracker.allocate(1000, GpuMemoryType::Buffer, None);
        tracker.allocate(2000, GpuMemoryType::Buffer, None);
        tracker.allocate(3000, GpuMemoryType::Texture, None);

        assert_eq!(tracker.usage_by_type(GpuMemoryType::Buffer), 3000);
        assert_eq!(tracker.usage_by_type(GpuMemoryType::Texture), 3000);
        assert_eq!(tracker.usage_by_type(GpuMemoryType::RenderTarget), 0);
    }

    #[test]
    fn test_gpu_memory_stats() {
        let mut tracker = GpuMemoryTracker::new(1_000_000);

        tracker.allocate(100_000, GpuMemoryType::Buffer, None);
        tracker.allocate(200_000, GpuMemoryType::Texture, None);

        let stats = tracker.stats();
        assert_eq!(stats.total_memory, 1_000_000);
        assert_eq!(stats.used_memory, 300_000);
        assert_eq!(stats.free_memory, 700_000);
        assert_eq!(stats.allocation_count, 2);
        assert_eq!(stats.largest_allocation, 200_000);
    }
}
