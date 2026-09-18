//! Matroska cue (seek index) writing.

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

use crate::demux::matroska::ebml::element_id;

// ============================================================================
// Cue Point
// ============================================================================

/// A single cue point for seeking.
#[derive(Debug, Clone)]
pub struct CuePoint {
    /// Timestamp in timecode scale units.
    pub time: u64,

    /// Track number.
    pub track: u64,

    /// Position of the cluster in the segment.
    pub cluster_position: u64,

    /// Relative position within the cluster (optional).
    pub relative_position: Option<u64>,

    /// Block number within the cluster (optional).
    pub block_number: Option<u64>,
}

impl CuePoint {
    /// Creates a new cue point.
    #[must_use]
    pub const fn new(time: u64, track: u64, cluster_position: u64) -> Self {
        Self {
            time,
            track,
            cluster_position,
            relative_position: None,
            block_number: None,
        }
    }

    /// Sets the relative position within the cluster.
    #[must_use]
    pub const fn with_relative_position(mut self, pos: u64) -> Self {
        self.relative_position = Some(pos);
        self
    }

    /// Sets the block number within the cluster.
    #[must_use]
    pub const fn with_block_number(mut self, num: u64) -> Self {
        self.block_number = Some(num);
        self
    }
}

// ============================================================================
// Cue Writer
// ============================================================================

/// Writer for Matroska cues (seek index).
///
/// Cues provide random access points for seeking in a Matroska file.
/// They map timestamps to cluster positions.
#[derive(Debug, Default)]
pub struct CueWriter {
    /// Collected cue points.
    pub cue_points: Vec<CuePoint>,

    /// Minimum interval between cue points (in timecode units).
    min_interval: u64,

    /// Last cue point timestamp per track.
    last_cue_time: Vec<u64>,
}

impl CueWriter {
    /// Creates a new cue writer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cue_points: Vec::new(),
            min_interval: 1000, // Default: 1 second at default timecode scale
            last_cue_time: Vec::new(),
        }
    }

    /// Sets the minimum interval between cue points.
    #[must_use]
    pub const fn with_min_interval(mut self, interval: u64) -> Self {
        self.min_interval = interval;
        self
    }

    /// Adds a cue point.
    ///
    /// Returns true if the cue point was added, false if it was skipped
    /// due to being too close to a previous cue point.
    pub fn add_cue_point(&mut self, time: u64, track: u64, cluster_position: u64) -> bool {
        // Ensure we have tracking for this track
        let track_idx = track as usize;
        while self.last_cue_time.len() <= track_idx {
            self.last_cue_time.push(0);
        }

        // Check if we should add a cue point (minimum interval)
        let last_time = self.last_cue_time[track_idx];
        if time > 0 && time < last_time + self.min_interval {
            return false;
        }

        self.cue_points
            .push(CuePoint::new(time, track, cluster_position));
        self.last_cue_time[track_idx] = time;
        true
    }

    /// Adds a detailed cue point with all optional fields.
    pub fn add_detailed_cue_point(
        &mut self,
        time: u64,
        track: u64,
        cluster_position: u64,
        relative_position: Option<u64>,
        block_number: Option<u64>,
    ) {
        let mut cue = CuePoint::new(time, track, cluster_position);
        if let Some(rel_pos) = relative_position {
            cue = cue.with_relative_position(rel_pos);
        }
        if let Some(block_num) = block_number {
            cue = cue.with_block_number(block_num);
        }
        self.cue_points.push(cue);
    }

    /// Builds the complete Cues element content.
    ///
    /// Returns the encoded bytes for the Cues element content
    /// (without the Cues element header).
    #[must_use]
    pub fn build(&self) -> Vec<u8> {
        let mut content = Vec::new();

        for cue in &self.cue_points {
            let cue_point = Self::build_cue_point(cue);
            content.extend(cue_point);
        }

        content
    }

    /// Builds a single `CuePoint` element.
    fn build_cue_point(cue: &CuePoint) -> Vec<u8> {
        let mut content = Vec::new();

        // CueTime
        content.extend(encode_element_id(element_id::CUE_TIME));
        let time_bytes = encode_uint(cue.time);
        content.extend(encode_vint_size(time_bytes.len() as u64));
        content.extend(time_bytes);

        // CueTrackPositions
        let track_positions = Self::build_cue_track_positions(cue);
        content.extend(encode_element_id(element_id::CUE_TRACK_POSITIONS));
        content.extend(encode_vint_size(track_positions.len() as u64));
        content.extend(track_positions);

        // Wrap in CuePoint element
        let mut result = Vec::new();
        result.extend(encode_element_id(element_id::CUE_POINT));
        result.extend(encode_vint_size(content.len() as u64));
        result.extend(content);

        result
    }

    /// Builds a `CueTrackPositions` element.
    fn build_cue_track_positions(cue: &CuePoint) -> Vec<u8> {
        let mut content = Vec::new();

        // CueTrack
        content.extend(encode_element_id(element_id::CUE_TRACK));
        let track_bytes = encode_uint(cue.track);
        content.extend(encode_vint_size(track_bytes.len() as u64));
        content.extend(track_bytes);

        // CueClusterPosition
        content.extend(encode_element_id(element_id::CUE_CLUSTER_POSITION));
        let pos_bytes = encode_uint(cue.cluster_position);
        content.extend(encode_vint_size(pos_bytes.len() as u64));
        content.extend(pos_bytes);

        // CueRelativePosition (optional)
        if let Some(rel_pos) = cue.relative_position {
            content.extend(encode_element_id(element_id::CUE_RELATIVE_POSITION));
            let rel_bytes = encode_uint(rel_pos);
            content.extend(encode_vint_size(rel_bytes.len() as u64));
            content.extend(rel_bytes);
        }

        // CueBlockNumber (optional)
        if let Some(block_num) = cue.block_number {
            content.extend(encode_element_id(element_id::CUE_BLOCK_NUMBER));
            let block_bytes = encode_uint(block_num);
            content.extend(encode_vint_size(block_bytes.len() as u64));
            content.extend(block_bytes);
        }

        content
    }

    /// Returns the number of cue points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cue_points.len()
    }

    /// Returns true if there are no cue points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cue_points.is_empty()
    }

    /// Clears all cue points.
    pub fn clear(&mut self) {
        self.cue_points.clear();
        self.last_cue_time.clear();
    }
}

// ============================================================================
// Encoding Helpers
// ============================================================================

/// Encodes an element ID to bytes.
///
/// EBML element IDs already include their class marker in the value,
/// so we just output the bytes that make up the ID.
fn encode_element_id(id: u32) -> Vec<u8> {
    if id <= 0xFF {
        vec![id as u8]
    } else if id <= 0xFFFF {
        vec![(id >> 8) as u8, id as u8]
    } else if id <= 0xFF_FFFF {
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
    if size < 0x80 {
        vec![0x80 | size as u8]
    } else if size < 0x4000 {
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cue_point_new() {
        let cue = CuePoint::new(1000, 1, 500);
        assert_eq!(cue.time, 1000);
        assert_eq!(cue.track, 1);
        assert_eq!(cue.cluster_position, 500);
        assert!(cue.relative_position.is_none());
        assert!(cue.block_number.is_none());
    }

    #[test]
    fn test_cue_point_with_optional_fields() {
        let cue = CuePoint::new(1000, 1, 500)
            .with_relative_position(100)
            .with_block_number(5);

        assert_eq!(cue.relative_position, Some(100));
        assert_eq!(cue.block_number, Some(5));
    }

    #[test]
    fn test_cue_writer_new() {
        let writer = CueWriter::new();
        assert!(writer.is_empty());
        assert_eq!(writer.len(), 0);
    }

    #[test]
    fn test_cue_writer_add_cue_point() {
        let mut writer = CueWriter::new().with_min_interval(0);

        assert!(writer.add_cue_point(0, 1, 0));
        assert!(writer.add_cue_point(1000, 1, 1000));
        assert_eq!(writer.len(), 2);
    }

    #[test]
    fn test_cue_writer_min_interval() {
        let mut writer = CueWriter::new().with_min_interval(1000);

        assert!(writer.add_cue_point(0, 1, 0));
        // Should be skipped (too close)
        assert!(!writer.add_cue_point(500, 1, 500));
        // Should be added (>= min_interval)
        assert!(writer.add_cue_point(1000, 1, 1000));

        assert_eq!(writer.len(), 2);
    }

    #[test]
    fn test_cue_writer_add_detailed_cue_point() {
        let mut writer = CueWriter::new();

        writer.add_detailed_cue_point(1000, 1, 500, Some(100), Some(5));

        assert_eq!(writer.len(), 1);
        let cue = &writer.cue_points[0];
        assert_eq!(cue.relative_position, Some(100));
        assert_eq!(cue.block_number, Some(5));
    }

    #[test]
    fn test_cue_writer_build() {
        let mut writer = CueWriter::new().with_min_interval(0);

        writer.add_cue_point(0, 1, 0);
        writer.add_cue_point(5000, 1, 10000);

        let content = writer.build();
        assert!(!content.is_empty());

        // Check that it starts with CuePoint element ID (0xBB)
        assert_eq!(content[0], 0xBB);
    }

    #[test]
    fn test_cue_writer_clear() {
        let mut writer = CueWriter::new().with_min_interval(0);

        writer.add_cue_point(0, 1, 0);
        writer.add_cue_point(1000, 1, 1000);

        writer.clear();
        assert!(writer.is_empty());
    }

    #[test]
    fn test_encode_element_id() {
        // 1-byte ID
        assert_eq!(encode_element_id(0xBB), vec![0xBB]);
        // 2-byte ID
        assert_eq!(encode_element_id(0x53AB), vec![0x53, 0xAB]);
    }

    #[test]
    fn test_encode_uint() {
        assert_eq!(encode_uint(0), vec![0]);
        assert_eq!(encode_uint(127), vec![127]);
        assert_eq!(encode_uint(255), vec![255]);
        assert_eq!(encode_uint(256), vec![1, 0]);
        assert_eq!(encode_uint(1000), vec![0x03, 0xE8]);
    }
}
