//! Matroska cluster writing.

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::demux::matroska::ebml::element_id;

// ============================================================================
// Cluster Writer
// ============================================================================

/// Writer for Matroska clusters.
///
/// A cluster is a container for blocks within a Matroska file.
/// Each cluster has a timestamp and contains blocks with relative timecodes.
#[derive(Debug)]
pub struct ClusterWriter {
    /// Cluster timestamp in timecode scale units.
    pub timecode: i64,

    /// Position of the cluster in the segment (for cues).
    pub position: u64,

    /// Maximum cluster duration.
    max_duration: i64,

    /// Maximum cluster size.
    max_size: usize,

    /// Current cluster size in bytes.
    current_size: usize,

    /// Last block timestamp in this cluster.
    last_timecode: i64,

    /// Number of blocks in this cluster.
    block_count: usize,
}

impl ClusterWriter {
    /// Creates a new cluster writer.
    ///
    /// # Arguments
    ///
    /// * `timecode` - Cluster timestamp in timecode scale units
    /// * `position` - Position of cluster in segment
    /// * `max_duration` - Maximum cluster duration in timecode units
    /// * `max_size` - Maximum cluster size in bytes
    #[must_use]
    pub const fn new(timecode: i64, position: u64, max_duration: i64, max_size: usize) -> Self {
        Self {
            timecode,
            position,
            max_duration,
            max_size,
            current_size: 0,
            last_timecode: timecode,
            block_count: 0,
        }
    }

    /// Returns whether a new cluster should be started.
    ///
    /// A new cluster should be started if:
    /// - The duration would exceed the maximum
    /// - The size would exceed the maximum
    #[must_use]
    pub fn should_start_new(&self, new_timecode: i64, block_size: usize) -> bool {
        // Check duration limit
        let duration = new_timecode - self.timecode;
        if duration > self.max_duration {
            return true;
        }

        // Check size limit
        let estimated_block_size = block_size + 10; // Header overhead
        if self.current_size + estimated_block_size > self.max_size {
            return true;
        }

        false
    }

    /// Adds a block to the cluster tracking.
    pub fn add_block(&mut self, timecode: i64, size: usize) {
        self.current_size += size + 10; // Approximate header overhead
        self.last_timecode = timecode;
        self.block_count += 1;
    }

    /// Returns the cluster duration so far.
    #[must_use]
    pub fn duration(&self) -> i64 {
        self.last_timecode - self.timecode
    }

    /// Returns the number of blocks in this cluster.
    #[must_use]
    pub const fn block_count(&self) -> usize {
        self.block_count
    }

    /// Returns the current cluster size in bytes.
    #[must_use]
    pub const fn current_size(&self) -> usize {
        self.current_size
    }
}

/// Builds a `SimpleBlock` for Matroska.
///
/// # Arguments
///
/// * `track_num` - Track number (1-based)
/// * `timecode` - Relative timecode (signed 16-bit)
/// * `data` - Block data
/// * `keyframe` - Whether this is a keyframe
/// * `invisible` - Whether the block is invisible
/// * `discardable` - Whether the block is discardable
///
/// # Returns
///
/// Encoded `SimpleBlock` bytes.
#[must_use]
#[allow(dead_code)]
pub fn build_simple_block(
    track_num: u64,
    timecode: i16,
    data: &[u8],
    keyframe: bool,
    invisible: bool,
    discardable: bool,
) -> Vec<u8> {
    let mut block = Vec::with_capacity(data.len() + 10);

    // Track number as VINT
    block.extend(encode_track_vint(track_num));

    // Timecode (relative to cluster, signed 16-bit big-endian)
    block.extend(&timecode.to_be_bytes());

    // Flags
    let mut flags: u8 = 0;
    if keyframe {
        flags |= 0x80;
    }
    if invisible {
        flags |= 0x08;
    }
    if discardable {
        flags |= 0x01;
    }
    block.push(flags);

    // Data
    block.extend_from_slice(data);

    block
}

/// Builds a `BlockGroup` for Matroska.
///
/// Block groups are used when additional metadata is needed:
/// - Block duration
/// - Reference blocks (for B-frames)
/// - Additions (side data)
///
/// # Arguments
///
/// * `track_num` - Track number (1-based)
/// * `timecode` - Relative timecode (signed 16-bit)
/// * `data` - Block data
/// * `duration` - Block duration in timecode units (optional)
/// * `reference_block` - Reference to previous block timecode (optional)
///
/// # Returns
///
/// Encoded `BlockGroup` bytes.
#[must_use]
#[allow(dead_code)]
pub fn build_block_group(
    track_num: u64,
    timecode: i16,
    data: &[u8],
    duration: Option<i64>,
    reference_block: Option<i16>,
) -> Vec<u8> {
    let mut content = Vec::new();

    // Build Block element
    let mut block = Vec::with_capacity(data.len() + 10);
    block.extend(encode_track_vint(track_num));
    block.extend(&timecode.to_be_bytes());
    block.push(0x00); // No flags for Block (only SimpleBlock has keyframe flag)
    block.extend_from_slice(data);

    // Write Block element
    content.extend(encode_element_id(element_id::BLOCK));
    content.extend(encode_vint_size(block.len() as u64));
    content.extend(block);

    // BlockDuration (optional)
    if let Some(dur) = duration {
        content.extend(encode_element_id(element_id::BLOCK_DURATION));
        let dur_bytes = encode_uint(dur as u64);
        content.extend(encode_vint_size(dur_bytes.len() as u64));
        content.extend(dur_bytes);
    }

    // ReferenceBlock (optional) - signed integer relative to block timecode
    if let Some(ref_block) = reference_block {
        content.extend(encode_element_id(element_id::REFERENCE_BLOCK));
        let ref_bytes = encode_sint(i64::from(ref_block));
        content.extend(encode_vint_size(ref_bytes.len() as u64));
        content.extend(ref_bytes);
    }

    // Wrap in BlockGroup element
    let mut result = Vec::new();
    result.extend(encode_element_id(element_id::BLOCK_GROUP));
    result.extend(encode_vint_size(content.len() as u64));
    result.extend(content);

    result
}

// ============================================================================
// Encoding Helpers
// ============================================================================

/// Encodes a track number as VINT.
fn encode_track_vint(track_num: u64) -> Vec<u8> {
    if track_num < 0x80 {
        vec![0x80 | track_num as u8]
    } else if track_num < 0x4000 {
        vec![0x40 | (track_num >> 8) as u8, track_num as u8]
    } else if track_num < 0x20_0000 {
        vec![
            0x20 | (track_num >> 16) as u8,
            (track_num >> 8) as u8,
            track_num as u8,
        ]
    } else {
        vec![
            0x10 | (track_num >> 24) as u8,
            (track_num >> 16) as u8,
            (track_num >> 8) as u8,
            track_num as u8,
        ]
    }
}

/// Encodes an element ID to bytes.
fn encode_element_id(id: u32) -> Vec<u8> {
    if id <= 0x7F {
        vec![id as u8]
    } else if id <= 0x3FFF {
        vec![(id >> 8) as u8, id as u8]
    } else if id <= 0x1F_FFFF {
        vec![(id >> 16) as u8, (id >> 8) as u8, id as u8]
    } else {
        vec![
            (id >> 24) as u8,
            (id >> 16) as u8,
            (id >> 8) as u8,
            id as u8,
        ]
    }
}

/// Encodes a VINT size.
fn encode_vint_size(size: u64) -> Vec<u8> {
    if size < 0x7F {
        vec![0x80 | size as u8]
    } else if size < 0x3FFF {
        vec![0x40 | (size >> 8) as u8, size as u8]
    } else if size < 0x1F_FFFF {
        vec![0x20 | (size >> 16) as u8, (size >> 8) as u8, size as u8]
    } else if size < 0x0FFF_FFFF {
        vec![
            0x10 | (size >> 24) as u8,
            (size >> 16) as u8,
            (size >> 8) as u8,
            size as u8,
        ]
    } else {
        vec![
            0x01,
            (size >> 48) as u8,
            (size >> 40) as u8,
            (size >> 32) as u8,
            (size >> 24) as u8,
            (size >> 16) as u8,
            (size >> 8) as u8,
            size as u8,
        ]
    }
}

/// Encodes an unsigned integer with minimal bytes.
fn encode_uint(value: u64) -> Vec<u8> {
    if value == 0 {
        vec![0]
    } else if value <= 0xFF {
        vec![value as u8]
    } else if value <= 0xFFFF {
        vec![(value >> 8) as u8, value as u8]
    } else if value <= 0xFF_FFFF {
        vec![(value >> 16) as u8, (value >> 8) as u8, value as u8]
    } else if value <= 0xFFFF_FFFF {
        vec![
            (value >> 24) as u8,
            (value >> 16) as u8,
            (value >> 8) as u8,
            value as u8,
        ]
    } else {
        vec![
            (value >> 56) as u8,
            (value >> 48) as u8,
            (value >> 40) as u8,
            (value >> 32) as u8,
            (value >> 24) as u8,
            (value >> 16) as u8,
            (value >> 8) as u8,
            value as u8,
        ]
    }
}

/// Encodes a signed integer with minimal bytes.
fn encode_sint(value: i64) -> Vec<u8> {
    if (-0x40..0x40).contains(&value) {
        vec![value as u8]
    } else if (-0x2000..0x2000).contains(&value) {
        vec![(value >> 8) as u8, value as u8]
    } else if (-0x10_0000..0x10_0000).contains(&value) {
        vec![(value >> 16) as u8, (value >> 8) as u8, value as u8]
    } else if (-0x0800_0000..0x0800_0000).contains(&value) {
        vec![
            (value >> 24) as u8,
            (value >> 16) as u8,
            (value >> 8) as u8,
            value as u8,
        ]
    } else {
        vec![
            (value >> 56) as u8,
            (value >> 48) as u8,
            (value >> 40) as u8,
            (value >> 32) as u8,
            (value >> 24) as u8,
            (value >> 16) as u8,
            (value >> 8) as u8,
            value as u8,
        ]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_writer_new() {
        let writer = ClusterWriter::new(1000, 0, 5000, 5 * 1024 * 1024);
        assert_eq!(writer.timecode, 1000);
        assert_eq!(writer.position, 0);
        assert_eq!(writer.block_count(), 0);
    }

    #[test]
    fn test_cluster_writer_add_block() {
        let mut writer = ClusterWriter::new(0, 0, 5000, 5 * 1024 * 1024);
        writer.add_block(100, 1000);
        assert_eq!(writer.block_count(), 1);
        assert_eq!(writer.last_timecode, 100);
    }

    #[test]
    fn test_cluster_writer_duration() {
        let mut writer = ClusterWriter::new(0, 0, 5000, 5 * 1024 * 1024);
        writer.add_block(1000, 100);
        assert_eq!(writer.duration(), 1000);
    }

    #[test]
    fn test_should_start_new_duration() {
        let writer = ClusterWriter::new(0, 0, 5000, 5 * 1024 * 1024);
        assert!(!writer.should_start_new(4999, 100));
        assert!(writer.should_start_new(5001, 100));
    }

    #[test]
    fn test_should_start_new_size() {
        let writer = ClusterWriter::new(0, 0, 5000, 1000);
        assert!(!writer.should_start_new(100, 500));
        assert!(writer.should_start_new(100, 1000));
    }

    #[test]
    fn test_build_simple_block() {
        let block = build_simple_block(1, 0, &[1, 2, 3, 4], true, false, false);
        assert!(!block.is_empty());
        // Track number (VINT: 0x81), timecode (2 bytes), flags (1 byte), data (4 bytes)
        assert!(block.len() >= 8);
        assert_eq!(block[0], 0x81); // Track 1
        assert_eq!(block[3], 0x80); // Keyframe flag
    }

    #[test]
    fn test_build_simple_block_flags() {
        let keyframe = build_simple_block(1, 0, &[], true, false, false);
        assert_eq!(keyframe[3], 0x80);

        let invisible = build_simple_block(1, 0, &[], false, true, false);
        assert_eq!(invisible[3], 0x08);

        let discardable = build_simple_block(1, 0, &[], false, false, true);
        assert_eq!(discardable[3], 0x01);

        let all_flags = build_simple_block(1, 0, &[], true, true, true);
        assert_eq!(all_flags[3], 0x89);
    }

    #[test]
    fn test_build_block_group() {
        let group = build_block_group(1, 100, &[1, 2, 3], Some(500), None);
        assert!(!group.is_empty());
        // BlockGroup is a multi-byte ID: check that it contains the ID
        // The actual bytes depend on encoding
        assert!(group.len() > 10); // Should have header, block, and optional fields
    }

    #[test]
    fn test_encode_track_vint() {
        assert_eq!(encode_track_vint(1), vec![0x81]);
        assert_eq!(encode_track_vint(127), vec![0xFF]);
        assert_eq!(encode_track_vint(128), vec![0x40, 0x80]);
    }

    #[test]
    fn test_encode_uint() {
        assert_eq!(encode_uint(0), vec![0]);
        assert_eq!(encode_uint(255), vec![255]);
        assert_eq!(encode_uint(256), vec![1, 0]);
    }

    #[test]
    fn test_encode_sint() {
        assert_eq!(encode_sint(0), vec![0]);
        assert_eq!(encode_sint(-1), vec![0xFF]);
        assert_eq!(encode_sint(63), vec![63]);
        assert_eq!(encode_sint(-64), vec![0xC0]);
    }
}
