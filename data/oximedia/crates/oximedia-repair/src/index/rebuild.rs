//! Index rebuilding functionality.
//!
//! This module provides functions to rebuild missing or corrupted indices
//! in media container files.

use crate::Result;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Index entry representing a seekable point in the media.
#[derive(Debug, Clone)]
pub struct IndexEntry {
    /// Timestamp in milliseconds.
    pub timestamp: u64,
    /// Byte offset in file.
    pub offset: u64,
    /// Is this a keyframe/sync point.
    pub is_keyframe: bool,
    /// Stream/track ID.
    pub stream_id: u32,
}

/// Rebuilt index structure.
#[derive(Debug)]
pub struct Index {
    /// Index entries.
    pub entries: Vec<IndexEntry>,
    /// Total duration in milliseconds.
    pub duration: u64,
}

/// Rebuild index for a media file.
pub fn rebuild_index(path: &Path) -> Result<Index> {
    let mut file = File::open(path)?;
    let mut entries = Vec::new();

    // Scan file for keyframes and sync points
    let keyframes = scan_for_keyframes(&mut file)?;

    // Convert keyframes to index entries
    for (offset, timestamp) in keyframes {
        entries.push(IndexEntry {
            timestamp,
            offset,
            is_keyframe: true,
            stream_id: 0,
        });
    }

    // Sort by timestamp
    entries.sort_by_key(|e| e.timestamp);

    // Calculate duration
    let duration = entries.last().map(|e| e.timestamp).unwrap_or(0);

    Ok(Index { entries, duration })
}

/// Scan file for keyframes.
fn scan_for_keyframes(file: &mut File) -> Result<Vec<(u64, u64)>> {
    let mut keyframes = Vec::new();
    let file_size = file.metadata()?.len();

    const CHUNK_SIZE: usize = 1024 * 1024;
    let mut buffer = vec![0u8; CHUNK_SIZE];
    let mut offset = 0u64;

    while offset < file_size {
        file.seek(SeekFrom::Start(offset))?;
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }

        let chunk = &buffer[..bytes_read];

        // Look for NAL unit start codes (H.264/H.265)
        let mut i = 0;
        while i + 5 <= chunk.len() {
            if chunk[i..i + 3] == [0x00, 0x00, 0x01] {
                let nal_type = chunk[i + 3] & 0x1F;
                // Type 5 is IDR (keyframe) in H.264
                if nal_type == 5 {
                    let position = offset + i as u64;
                    let timestamp = estimate_timestamp(position, file_size);
                    keyframes.push((position, timestamp));
                }
                i += 4;
            } else {
                i += 1;
            }
        }

        offset += bytes_read as u64;
    }

    Ok(keyframes)
}

/// Estimate timestamp based on file position.
fn estimate_timestamp(offset: u64, total_size: u64) -> u64 {
    // Simple linear estimation
    // Assumes constant bitrate and 1-hour video
    const ASSUMED_DURATION_MS: u64 = 3600000; // 1 hour
    (offset * ASSUMED_DURATION_MS) / total_size
}

/// Write index to file.
pub fn write_index(index: &Index, output: &Path) -> Result<()> {
    use std::io::Write;

    let mut file = File::create(output)?;

    // Write header
    file.write_all(b"OXIX")?; // OxiMedia Index
    file.write_all(&(index.entries.len() as u32).to_le_bytes())?;
    file.write_all(&index.duration.to_le_bytes())?;

    // Write entries
    for entry in &index.entries {
        file.write_all(&entry.timestamp.to_le_bytes())?;
        file.write_all(&entry.offset.to_le_bytes())?;
        file.write_all(&[if entry.is_keyframe { 1 } else { 0 }])?;
        file.write_all(&entry.stream_id.to_le_bytes())?;
    }

    Ok(())
}

/// Read index from file.
pub fn read_index(path: &Path) -> Result<Index> {
    let mut file = File::open(path)?;

    // Read header
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != b"OXIX" {
        return Err(crate::RepairError::RepairFailed(
            "Invalid index file".to_string(),
        ));
    }

    let mut count_bytes = [0u8; 4];
    file.read_exact(&mut count_bytes)?;
    let count = u32::from_le_bytes(count_bytes) as usize;

    let mut duration_bytes = [0u8; 8];
    file.read_exact(&mut duration_bytes)?;
    let duration = u64::from_le_bytes(duration_bytes);

    // Read entries
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let mut timestamp_bytes = [0u8; 8];
        file.read_exact(&mut timestamp_bytes)?;
        let timestamp = u64::from_le_bytes(timestamp_bytes);

        let mut offset_bytes = [0u8; 8];
        file.read_exact(&mut offset_bytes)?;
        let offset = u64::from_le_bytes(offset_bytes);

        let mut keyframe_byte = [0u8; 1];
        file.read_exact(&mut keyframe_byte)?;
        let is_keyframe = keyframe_byte[0] != 0;

        let mut stream_id_bytes = [0u8; 4];
        file.read_exact(&mut stream_id_bytes)?;
        let stream_id = u32::from_le_bytes(stream_id_bytes);

        entries.push(IndexEntry {
            timestamp,
            offset,
            is_keyframe,
            stream_id,
        });
    }

    Ok(Index { entries, duration })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_timestamp() {
        let ts = estimate_timestamp(1000, 10000);
        assert_eq!(ts, 360000); // 10% through 1-hour video
    }

    #[test]
    fn test_estimate_timestamp_half() {
        let ts = estimate_timestamp(5000, 10000);
        assert_eq!(ts, 1800000); // 50% through 1-hour video
    }

    #[test]
    fn test_index_entry_creation() {
        let entry = IndexEntry {
            timestamp: 1000,
            offset: 500,
            is_keyframe: true,
            stream_id: 0,
        };

        assert_eq!(entry.timestamp, 1000);
        assert_eq!(entry.offset, 500);
        assert!(entry.is_keyframe);
    }
}
