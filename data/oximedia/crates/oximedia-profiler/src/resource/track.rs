//! Resource tracking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Resource statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStats {
    /// Number of file handles.
    pub file_handles: usize,

    /// Number of network sockets.
    pub network_sockets: usize,

    /// Number of threads.
    pub thread_count: usize,

    /// Total memory allocated.
    pub total_memory: usize,
}

/// Resource type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// File handle.
    File,

    /// Network socket.
    Socket,

    /// Thread.
    Thread,

    /// Memory allocation.
    Memory,

    /// Other resource.
    Other,
}

/// Resource tracker.
#[derive(Debug)]
pub struct ResourceTracker {
    resources: HashMap<u64, (ResourceType, String)>,
    next_id: u64,
    counts: HashMap<ResourceType, usize>,
}

impl ResourceTracker {
    /// Create a new resource tracker.
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
            next_id: 0,
            counts: HashMap::new(),
        }
    }

    /// Track a resource.
    pub fn track(&mut self, resource_type: ResourceType, name: String) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        self.resources.insert(id, (resource_type, name));
        *self.counts.entry(resource_type).or_insert(0) += 1;

        id
    }

    /// Untrack a resource.
    pub fn untrack(&mut self, id: u64) -> bool {
        if let Some((resource_type, _)) = self.resources.remove(&id) {
            if let Some(count) = self.counts.get_mut(&resource_type) {
                *count = count.saturating_sub(1);
            }
            true
        } else {
            false
        }
    }

    /// Get resource count by type.
    pub fn count(&self, resource_type: ResourceType) -> usize {
        self.counts.get(&resource_type).copied().unwrap_or(0)
    }

    /// Get total resource count.
    pub fn total_count(&self) -> usize {
        self.resources.len()
    }

    /// Get statistics.
    pub fn stats(&self) -> ResourceStats {
        ResourceStats {
            file_handles: self.count(ResourceType::File),
            network_sockets: self.count(ResourceType::Socket),
            thread_count: self.count(ResourceType::Thread),
            total_memory: 0, // Would be tracked separately
        }
    }

    /// Generate a summary.
    pub fn summary(&self) -> String {
        let stats = self.stats();
        format!(
            "Files: {}, Sockets: {}, Threads: {}, Total: {}",
            stats.file_handles,
            stats.network_sockets,
            stats.thread_count,
            self.total_count()
        )
    }
}

impl Default for ResourceTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_tracker() {
        let mut tracker = ResourceTracker::new();
        assert_eq!(tracker.total_count(), 0);

        let id = tracker.track(ResourceType::File, "test.txt".to_string());
        assert_eq!(tracker.count(ResourceType::File), 1);

        tracker.untrack(id);
        assert_eq!(tracker.count(ResourceType::File), 0);
    }

    #[test]
    fn test_resource_stats() {
        let mut tracker = ResourceTracker::new();
        tracker.track(ResourceType::File, "file1".to_string());
        tracker.track(ResourceType::Socket, "socket1".to_string());

        let stats = tracker.stats();
        assert_eq!(stats.file_handles, 1);
        assert_eq!(stats.network_sockets, 1);
    }
}
