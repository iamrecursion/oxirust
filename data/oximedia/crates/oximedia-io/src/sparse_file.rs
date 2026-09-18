#![allow(dead_code)]
//! Sparse file handling with hole detection and efficient I/O.
//!
//! Provides abstractions for working with sparse files, detecting data
//! regions and holes, and performing efficient reads/writes that skip
//! over zero-filled regions.

use std::fmt;

/// Represents a region within a sparse file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseRegion {
    /// A region containing actual data.
    Data {
        /// Offset from start of file.
        offset: u64,
        /// Length of the data region.
        length: u64,
    },
    /// A hole (zero-filled region).
    Hole {
        /// Offset from start of file.
        offset: u64,
        /// Length of the hole.
        length: u64,
    },
}

impl SparseRegion {
    /// Get the offset of this region.
    pub fn offset(&self) -> u64 {
        match self {
            Self::Data { offset, .. } | Self::Hole { offset, .. } => *offset,
        }
    }

    /// Get the length of this region.
    pub fn length(&self) -> u64 {
        match self {
            Self::Data { length, .. } | Self::Hole { length, .. } => *length,
        }
    }

    /// Get the end offset (exclusive).
    pub fn end(&self) -> u64 {
        self.offset() + self.length()
    }

    /// Check if this is a data region.
    pub fn is_data(&self) -> bool {
        matches!(self, Self::Data { .. })
    }

    /// Check if this is a hole.
    pub fn is_hole(&self) -> bool {
        matches!(self, Self::Hole { .. })
    }

    /// Check if a given offset falls within this region.
    pub fn contains(&self, offset: u64) -> bool {
        offset >= self.offset() && offset < self.end()
    }
}

impl fmt::Display for SparseRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Data { offset, length } => {
                write!(
                    f,
                    "Data[{offset:#x}..{:#x}, {length} bytes]",
                    offset + length
                )
            }
            Self::Hole { offset, length } => {
                write!(
                    f,
                    "Hole[{offset:#x}..{:#x}, {length} bytes]",
                    offset + length
                )
            }
        }
    }
}

/// A map of sparse regions for a file.
#[derive(Debug, Clone)]
pub struct SparseMap {
    /// Total file size.
    file_size: u64,
    /// Ordered list of regions.
    regions: Vec<SparseRegion>,
}

impl SparseMap {
    /// Create a sparse map from a list of regions.
    pub fn new(file_size: u64, regions: Vec<SparseRegion>) -> Self {
        Self { file_size, regions }
    }

    /// Create a fully-dense map (no holes).
    pub fn dense(file_size: u64) -> Self {
        let regions = if file_size > 0 {
            vec![SparseRegion::Data {
                offset: 0,
                length: file_size,
            }]
        } else {
            Vec::new()
        };
        Self { file_size, regions }
    }

    /// Create a fully-sparse map (all holes).
    pub fn all_holes(file_size: u64) -> Self {
        let regions = if file_size > 0 {
            vec![SparseRegion::Hole {
                offset: 0,
                length: file_size,
            }]
        } else {
            Vec::new()
        };
        Self { file_size, regions }
    }

    /// Get the total file size.
    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Get the regions.
    pub fn regions(&self) -> &[SparseRegion] {
        &self.regions
    }

    /// Get the number of regions.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Get total data bytes (non-hole).
    pub fn data_bytes(&self) -> u64 {
        self.regions
            .iter()
            .filter(|r| r.is_data())
            .map(|r| r.length())
            .sum()
    }

    /// Get total hole bytes.
    pub fn hole_bytes(&self) -> u64 {
        self.regions
            .iter()
            .filter(|r| r.is_hole())
            .map(|r| r.length())
            .sum()
    }

    /// Get the sparseness ratio (0.0 = dense, 1.0 = all holes).
    #[allow(clippy::cast_precision_loss)]
    pub fn sparseness(&self) -> f64 {
        if self.file_size == 0 {
            return 0.0;
        }
        self.hole_bytes() as f64 / self.file_size as f64
    }

    /// Find the region containing a given offset.
    pub fn region_at(&self, offset: u64) -> Option<&SparseRegion> {
        self.regions.iter().find(|r| r.contains(offset))
    }

    /// Check if a given offset is in a data region.
    pub fn is_data_at(&self, offset: u64) -> bool {
        self.region_at(offset).map_or(false, |r| r.is_data())
    }

    /// Check if a given offset is in a hole.
    pub fn is_hole_at(&self, offset: u64) -> bool {
        self.region_at(offset).map_or(false, |r| r.is_hole())
    }

    /// Get only data regions.
    pub fn data_regions(&self) -> Vec<&SparseRegion> {
        self.regions.iter().filter(|r| r.is_data()).collect()
    }

    /// Get only hole regions.
    pub fn hole_regions(&self) -> Vec<&SparseRegion> {
        self.regions.iter().filter(|r| r.is_hole()).collect()
    }
}

/// Detect sparse regions in a byte buffer by scanning for zero-filled blocks.
#[allow(clippy::cast_possible_truncation)]
pub fn detect_sparse_regions(data: &[u8], block_size: usize) -> SparseMap {
    let file_size = data.len() as u64;
    if data.is_empty() {
        return SparseMap::new(file_size, Vec::new());
    }

    let block_size = block_size.max(1);
    let mut regions = Vec::new();
    let mut offset = 0usize;

    while offset < data.len() {
        let end = (offset + block_size).min(data.len());
        let block = &data[offset..end];
        let is_zero = block.iter().all(|&b| b == 0);

        let region_type_is_hole = is_zero;

        // Try to merge with previous region of the same type.
        if let Some(last) = regions.last_mut() {
            let can_merge = match last {
                SparseRegion::Hole { .. } => region_type_is_hole,
                SparseRegion::Data { .. } => !region_type_is_hole,
            };
            if can_merge {
                match last {
                    SparseRegion::Data { length, .. } | SparseRegion::Hole { length, .. } => {
                        *length += (end - offset) as u64;
                    }
                }
                offset = end;
                continue;
            }
        }

        let region = if region_type_is_hole {
            SparseRegion::Hole {
                offset: offset as u64,
                length: (end - offset) as u64,
            }
        } else {
            SparseRegion::Data {
                offset: offset as u64,
                length: (end - offset) as u64,
            }
        };
        regions.push(region);
        offset = end;
    }

    SparseMap::new(file_size, regions)
}

/// Statistics about sparseness.
#[derive(Debug, Clone)]
pub struct SparseStats {
    /// Total file size.
    pub file_size: u64,
    /// Number of data regions.
    pub data_region_count: usize,
    /// Number of hole regions.
    pub hole_region_count: usize,
    /// Total data bytes.
    pub data_bytes: u64,
    /// Total hole bytes.
    pub hole_bytes: u64,
    /// Sparseness ratio.
    pub sparseness: f64,
}

impl SparseStats {
    /// Compute statistics from a sparse map.
    pub fn from_map(map: &SparseMap) -> Self {
        Self {
            file_size: map.file_size(),
            data_region_count: map.data_regions().len(),
            hole_region_count: map.hole_regions().len(),
            data_bytes: map.data_bytes(),
            hole_bytes: map.hole_bytes(),
            sparseness: map.sparseness(),
        }
    }

    /// Estimate disk savings from sparse storage.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimated_savings_percent(&self) -> f64 {
        self.sparseness * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_region_data() {
        let r = SparseRegion::Data {
            offset: 100,
            length: 200,
        };
        assert!(r.is_data());
        assert!(!r.is_hole());
        assert_eq!(r.offset(), 100);
        assert_eq!(r.length(), 200);
        assert_eq!(r.end(), 300);
    }

    #[test]
    fn test_sparse_region_hole() {
        let r = SparseRegion::Hole {
            offset: 0,
            length: 50,
        };
        assert!(r.is_hole());
        assert!(!r.is_data());
    }

    #[test]
    fn test_sparse_region_contains() {
        let r = SparseRegion::Data {
            offset: 10,
            length: 20,
        };
        assert!(!r.contains(9));
        assert!(r.contains(10));
        assert!(r.contains(29));
        assert!(!r.contains(30));
    }

    #[test]
    fn test_sparse_region_display() {
        let r = SparseRegion::Data {
            offset: 0,
            length: 100,
        };
        let s = format!("{r}");
        assert!(s.contains("Data"));
        assert!(s.contains("100 bytes"));
    }

    #[test]
    fn test_sparse_map_dense() {
        let map = SparseMap::dense(1000);
        assert_eq!(map.file_size(), 1000);
        assert_eq!(map.data_bytes(), 1000);
        assert_eq!(map.hole_bytes(), 0);
        assert!((map.sparseness() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sparse_map_all_holes() {
        let map = SparseMap::all_holes(500);
        assert_eq!(map.data_bytes(), 0);
        assert_eq!(map.hole_bytes(), 500);
        assert!((map.sparseness() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sparse_map_empty() {
        let map = SparseMap::dense(0);
        assert_eq!(map.region_count(), 0);
        assert!((map.sparseness() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sparse_map_region_at() {
        let map = SparseMap::new(
            200,
            vec![
                SparseRegion::Data {
                    offset: 0,
                    length: 100,
                },
                SparseRegion::Hole {
                    offset: 100,
                    length: 100,
                },
            ],
        );
        assert!(map.is_data_at(50));
        assert!(map.is_hole_at(150));
        assert!(!map.is_data_at(200));
    }

    #[test]
    fn test_detect_all_zeros() {
        let data = vec![0u8; 256];
        let map = detect_sparse_regions(&data, 64);
        assert_eq!(map.data_bytes(), 0);
        assert_eq!(map.hole_bytes(), 256);
        assert_eq!(map.region_count(), 1);
    }

    #[test]
    fn test_detect_all_data() {
        let data = vec![0xFFu8; 256];
        let map = detect_sparse_regions(&data, 64);
        assert_eq!(map.data_bytes(), 256);
        assert_eq!(map.hole_bytes(), 0);
        assert_eq!(map.region_count(), 1);
    }

    #[test]
    fn test_detect_mixed() {
        // 64 bytes data, 64 bytes hole, 64 bytes data
        let mut data = vec![0xABu8; 64];
        data.extend_from_slice(&[0u8; 64]);
        data.extend_from_slice(&[0xCDu8; 64]);
        let map = detect_sparse_regions(&data, 64);
        assert_eq!(map.region_count(), 3);
        assert_eq!(map.data_bytes(), 128);
        assert_eq!(map.hole_bytes(), 64);
    }

    #[test]
    fn test_detect_empty() {
        let map = detect_sparse_regions(&[], 64);
        assert_eq!(map.region_count(), 0);
        assert_eq!(map.file_size(), 0);
    }

    #[test]
    fn test_sparse_stats() {
        let map = SparseMap::new(
            1000,
            vec![
                SparseRegion::Data {
                    offset: 0,
                    length: 600,
                },
                SparseRegion::Hole {
                    offset: 600,
                    length: 400,
                },
            ],
        );
        let stats = SparseStats::from_map(&map);
        assert_eq!(stats.data_region_count, 1);
        assert_eq!(stats.hole_region_count, 1);
        assert_eq!(stats.data_bytes, 600);
        assert_eq!(stats.hole_bytes, 400);
        assert!((stats.sparseness - 0.4).abs() < f64::EPSILON);
        assert!((stats.estimated_savings_percent() - 40.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_data_and_hole_regions_filter() {
        let map = SparseMap::new(
            300,
            vec![
                SparseRegion::Data {
                    offset: 0,
                    length: 100,
                },
                SparseRegion::Hole {
                    offset: 100,
                    length: 100,
                },
                SparseRegion::Data {
                    offset: 200,
                    length: 100,
                },
            ],
        );
        assert_eq!(map.data_regions().len(), 2);
        assert_eq!(map.hole_regions().len(), 1);
    }
}
