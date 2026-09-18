//! Distributed array operations
//!
//! This module provides distributed array functionality for scaling operations
//! across multiple nodes in a cluster environment.

use crate::error::CoreResult;

/// Distributed array structure for parallel processing across nodes
#[derive(Debug)]
pub struct DistributedArray<T> {
    local_chunk: Vec<T>,
    total_size: usize,
    chunk_start: usize,
    chunk_end: usize,
    nodeid: String,
}

impl<T> DistributedArray<T>
where
    T: Clone + Send + Sync,
{
    /// Create a new distributed array chunk
    pub fn new(local_chunk: Vec<T>, total_size: usize, chunk_start: usize, nodeid: String) -> Self {
        let chunk_end = chunk_start + local_chunk.len();
        Self {
            local_chunk,
            total_size,
            chunk_start,
            chunk_end,
            nodeid,
        }
    }

    /// Get the local chunk of the distributed array
    pub fn local_data(&self) -> &[T] {
        &self.local_chunk
    }

    /// Get the size of the local chunk
    pub fn local_size(&self) -> usize {
        self.local_chunk.len()
    }

    /// Get the total size across all nodes
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Get the global start index of this chunk
    pub fn chunk_start(&self) -> usize {
        self.chunk_start
    }

    /// Get the global end index of this chunk
    pub fn chunk_end(&self) -> usize {
        self.chunk_end
    }

    /// Get the node ID hosting this chunk
    pub fn nodeid(&self) -> &str {
        &self.nodeid
    }
}

/// Manager for distributed array operations
#[derive(Debug)]
pub struct DistributedArrayManager {
    nodeid: String,
    clustersize: usize,
}

impl DistributedArrayManager {
    /// Create a new distributed array manager
    pub fn new(nodeid: String, clustersize: usize) -> Self {
        Self {
            nodeid,
            clustersize,
        }
    }

    /// Distribute an array across cluster nodes
    pub fn distribute_array<T>(&self, data: Vec<T>) -> CoreResult<DistributedArray<T>>
    where
        T: Clone + Send + Sync,
    {
        let total_size = data.len();
        let _chunk_size = total_size.div_ceil(self.clustersize);

        // For this simple implementation, just create a local chunk
        // In a real implementation, this would distribute across actual nodes
        let chunk_start = 0;
        let local_chunk = data;

        Ok(DistributedArray::new(
            local_chunk,
            total_size,
            chunk_start,
            self.nodeid.clone(),
        ))
    }

    /// Gather results from distributed computation
    pub fn gather_results<T>(&self, localresult: Vec<T>) -> CoreResult<Vec<T>>
    where
        T: Clone + Send + Sync,
    {
        // In a real implementation, this would gather from all nodes
        Ok(localresult)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_array_creation() {
        let data = vec![1, 2, 3, 4, 5];
        let nodeid = "node1".to_string();
        let array = DistributedArray::new(data.clone(), 10, 0, nodeid.clone());

        assert_eq!(array.local_data(), &data);
        assert_eq!(array.local_size(), 5);
        assert_eq!(array.total_size(), 10);
        assert_eq!(array.chunk_start(), 0);
        assert_eq!(array.chunk_end(), 5);
        assert_eq!(array.nodeid(), "node1");
    }

    #[test]
    fn test_distributed_array_manager() {
        let manager = DistributedArrayManager::new("node1".to_string(), 3);
        let data = vec![1, 2, 3, 4, 5, 6];

        let distributed = manager.distribute_array(data).expect("Operation failed");
        assert_eq!(distributed.total_size(), 6);
    }
}
