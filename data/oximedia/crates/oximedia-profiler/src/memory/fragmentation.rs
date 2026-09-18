//! Memory fragmentation analysis.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Memory fragmentation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentationReport {
    /// Total memory size.
    pub total_memory: usize,

    /// Used memory.
    pub used_memory: usize,

    /// Free memory.
    pub free_memory: usize,

    /// Number of free blocks.
    pub free_blocks: usize,

    /// Largest free block size.
    pub largest_free_block: usize,

    /// Average free block size.
    pub avg_free_block_size: f64,

    /// Fragmentation ratio (0.0-1.0).
    pub fragmentation_ratio: f64,

    /// Severity (0.0-1.0).
    pub severity: f64,
}

impl FragmentationReport {
    /// Check if fragmentation is critical.
    pub fn is_critical(&self) -> bool {
        self.severity > 0.7
    }

    /// Check if fragmentation is significant.
    pub fn is_significant(&self) -> bool {
        self.severity > 0.4
    }

    /// Get a description of the fragmentation.
    pub fn description(&self) -> String {
        let criticality = if self.is_critical() {
            "CRITICAL"
        } else if self.is_significant() {
            "SIGNIFICANT"
        } else {
            "LOW"
        };

        format!(
            "[{}] Fragmentation: {:.2}% (ratio: {:.2}, {} free blocks, largest: {} bytes)",
            criticality,
            self.severity * 100.0,
            self.fragmentation_ratio,
            self.free_blocks,
            self.largest_free_block
        )
    }
}

/// Memory block.
#[derive(Debug, Clone)]
struct MemoryBlock {
    #[allow(dead_code)]
    address: usize,
    size: usize,
    used: bool,
}

/// Memory fragmentation analyzer.
#[derive(Debug)]
pub struct FragmentationAnalyzer {
    blocks: BTreeMap<usize, MemoryBlock>,
    total_memory: usize,
}

impl FragmentationAnalyzer {
    /// Create a new fragmentation analyzer.
    pub fn new(total_memory: usize) -> Self {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            0,
            MemoryBlock {
                address: 0,
                size: total_memory,
                used: false,
            },
        );

        Self {
            blocks,
            total_memory,
        }
    }

    /// Allocate memory.
    pub fn allocate(&mut self, size: usize) -> Option<usize> {
        // Find first free block large enough
        let address = self
            .blocks
            .iter()
            .find(|(_, block)| !block.used && block.size >= size)
            .map(|(addr, _)| *addr)?;

        if let Some(block) = self.blocks.get_mut(&address) {
            if block.size > size {
                // Split the block
                let new_address = address + size;
                let new_size = block.size - size;
                block.size = size;
                block.used = true;

                self.blocks.insert(
                    new_address,
                    MemoryBlock {
                        address: new_address,
                        size: new_size,
                        used: false,
                    },
                );
            } else {
                block.used = true;
            }
        }

        Some(address)
    }

    /// Free memory.
    pub fn free(&mut self, address: usize) -> bool {
        if let Some(block) = self.blocks.get_mut(&address) {
            if block.used {
                block.used = false;
                self.coalesce(address);
                return true;
            }
        }
        false
    }

    /// Coalesce adjacent free blocks.
    fn coalesce(&mut self, address: usize) {
        // Try to merge with next block
        if let Some(block) = self.blocks.get(&address).cloned() {
            if !block.used {
                let next_address = address + block.size;
                if let Some(next_block) = self.blocks.get(&next_address).cloned() {
                    if !next_block.used {
                        self.blocks.remove(&next_address);
                        if let Some(current) = self.blocks.get_mut(&address) {
                            current.size += next_block.size;
                        }
                    }
                }
            }
        }

        // Try to merge with previous block
        if let Some((&prev_address, prev_block)) = self.blocks.range(..address).next_back() {
            if !prev_block.used && prev_address + prev_block.size == address {
                let current_size = self.blocks.get(&address).map(|b| b.size).unwrap_or(0);
                self.blocks.remove(&address);
                if let Some(prev) = self.blocks.get_mut(&prev_address) {
                    prev.size += current_size;
                }
            }
        }
    }

    /// Analyze fragmentation.
    pub fn analyze(&self) -> FragmentationReport {
        let mut used_memory = 0;
        let mut free_memory = 0;
        let mut free_blocks = 0;
        let mut largest_free_block = 0;
        let mut total_free_size = 0;

        for block in self.blocks.values() {
            if block.used {
                used_memory += block.size;
            } else {
                free_memory += block.size;
                free_blocks += 1;
                total_free_size += block.size;
                if block.size > largest_free_block {
                    largest_free_block = block.size;
                }
            }
        }

        let avg_free_block_size = if free_blocks > 0 {
            total_free_size as f64 / free_blocks as f64
        } else {
            0.0
        };

        let fragmentation_ratio = if free_memory > 0 {
            1.0 - (largest_free_block as f64 / free_memory as f64)
        } else {
            0.0
        };

        let severity = if free_blocks <= 1 {
            0.0
        } else {
            fragmentation_ratio * (free_blocks as f64 / 10.0).min(1.0)
        };

        FragmentationReport {
            total_memory: self.total_memory,
            used_memory,
            free_memory,
            free_blocks,
            largest_free_block,
            avg_free_block_size,
            fragmentation_ratio,
            severity,
        }
    }

    /// Get total memory.
    pub fn total_memory(&self) -> usize {
        self.total_memory
    }

    /// Get number of blocks.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fragmentation_analyzer() {
        let analyzer = FragmentationAnalyzer::new(1000);
        assert_eq!(analyzer.total_memory(), 1000);
        assert_eq!(analyzer.block_count(), 1);
    }

    #[test]
    fn test_allocate_free() {
        let mut analyzer = FragmentationAnalyzer::new(1000);

        let addr1 = analyzer.allocate(100);
        assert!(addr1.is_some());

        let addr2 = analyzer.allocate(200);
        assert!(addr2.is_some());

        assert!(analyzer.free(addr1.expect("should succeed in test")));
        assert!(analyzer.free(addr2.expect("should succeed in test")));
    }

    #[test]
    fn test_fragmentation_report() {
        let mut analyzer = FragmentationAnalyzer::new(1000);

        analyzer.allocate(100);
        analyzer.allocate(200);
        analyzer.allocate(100);

        let report = analyzer.analyze();
        assert_eq!(report.total_memory, 1000);
        assert_eq!(report.used_memory, 400);
        assert_eq!(report.free_memory, 600);
    }

    #[test]
    fn test_coalesce() {
        let mut analyzer = FragmentationAnalyzer::new(1000);

        let addr1 = analyzer.allocate(100).expect("should succeed in test");
        let addr2 = analyzer.allocate(100).expect("should succeed in test");
        let _addr3 = analyzer.allocate(100).expect("should succeed in test");

        analyzer.free(addr1);
        analyzer.free(addr2);

        // After coalescing, should have fewer blocks
        let report = analyzer.analyze();
        assert!(report.largest_free_block >= 200);
    }

    #[test]
    fn test_fragmentation_severity() {
        let mut analyzer = FragmentationAnalyzer::new(1000);

        // Create fragmentation
        let addrs: Vec<_> = (0..5).map(|_| analyzer.allocate(100)).collect();

        // Free every other allocation
        for (i, addr) in addrs.iter().enumerate() {
            if i % 2 == 0 {
                if let Some(a) = addr {
                    analyzer.free(*a);
                }
            }
        }

        let report = analyzer.analyze();
        assert!(report.fragmentation_ratio > 0.0);
    }
}
