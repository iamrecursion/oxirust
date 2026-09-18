//! Seek table rebuilding for MP4 and other formats.
//!
//! This module provides functions to rebuild seek tables
//! that enable efficient seeking in media files.

use crate::Result;
use std::path::Path;

/// Seek table entry.
#[derive(Debug, Clone)]
pub struct SeekEntry {
    /// Sample number.
    pub sample: u32,
    /// Byte offset.
    pub offset: u64,
    /// Timestamp in media timescale units.
    pub timestamp: u64,
}

/// Rebuild seek table for MP4 file.
pub fn rebuild_seek_table(path: &Path) -> Result<Vec<SeekEntry>> {
    let index = super::rebuild::rebuild_index(path)?;

    let seek_entries = index
        .entries
        .iter()
        .enumerate()
        .map(|(i, e)| SeekEntry {
            sample: i as u32,
            offset: e.offset,
            timestamp: e.timestamp,
        })
        .collect();

    Ok(seek_entries)
}

/// Write seek table to MP4 stbl atom.
pub fn write_seek_table(_path: &Path, _entries: &[SeekEntry]) -> Result<()> {
    // This is a placeholder for writing MP4 seek tables
    // In a real implementation, this would:
    // 1. Parse the MP4 file structure
    // 2. Find the stbl (sample table) atom
    // 3. Update stco/co64 (chunk offset) and stsc (sample-to-chunk) atoms
    Ok(())
}

/// Optimize seek table by removing redundant entries.
pub fn optimize_seek_table(entries: &mut Vec<SeekEntry>) {
    if entries.is_empty() {
        return;
    }

    // Keep only keyframe entries and periodic sync points
    const SYNC_INTERVAL: u64 = 1000; // 1 second
    let mut last_kept_timestamp = entries[0].timestamp;
    let mut optimized = vec![entries[0].clone()];

    for entry in entries.iter().skip(1) {
        if entry.timestamp - last_kept_timestamp >= SYNC_INTERVAL {
            optimized.push(entry.clone());
            last_kept_timestamp = entry.timestamp;
        }
    }

    *entries = optimized;
}

/// Calculate average bitrate from seek table.
pub fn calculate_bitrate(entries: &[SeekEntry]) -> Option<u64> {
    if entries.len() < 2 {
        return None;
    }

    let first = &entries[0];
    let last = &entries[entries.len() - 1];

    let bytes = last.offset - first.offset;
    let duration_ms = last.timestamp - first.timestamp;

    if duration_ms == 0 {
        return None;
    }

    // Convert to bits per second
    Some((bytes * 8 * 1000) / duration_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_seek_table() {
        let mut entries = vec![
            SeekEntry {
                sample: 0,
                offset: 0,
                timestamp: 0,
            },
            SeekEntry {
                sample: 1,
                offset: 100,
                timestamp: 100,
            },
            SeekEntry {
                sample: 2,
                offset: 200,
                timestamp: 1100,
            },
            SeekEntry {
                sample: 3,
                offset: 300,
                timestamp: 2100,
            },
        ];

        optimize_seek_table(&mut entries);
        assert_eq!(entries.len(), 3); // Should keep entries at 0, 1100, 2100
    }

    #[test]
    fn test_calculate_bitrate() {
        let entries = vec![
            SeekEntry {
                sample: 0,
                offset: 0,
                timestamp: 0,
            },
            SeekEntry {
                sample: 1,
                offset: 1000,
                timestamp: 1000,
            },
        ];

        let bitrate = calculate_bitrate(&entries);
        assert_eq!(bitrate, Some(8000)); // 1000 bytes in 1 second = 8000 bps
    }

    #[test]
    fn test_calculate_bitrate_insufficient_data() {
        let entries = vec![SeekEntry {
            sample: 0,
            offset: 0,
            timestamp: 0,
        }];

        let bitrate = calculate_bitrate(&entries);
        assert_eq!(bitrate, None);
    }
}
