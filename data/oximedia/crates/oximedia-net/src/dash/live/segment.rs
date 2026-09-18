//! Segment generation and management for DASH live streaming.
//!
//! This module handles the creation of DASH segments from live input,
//! including initialization segment generation, media segment packaging,
//! and segment alignment across multiple representations.

#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]

use bytes::{Bytes, BytesMut};
use oximedia_container::Packet;
use std::collections::HashMap;
use std::time::Duration;

/// Segment generator for creating DASH segments from live input.
///
/// This structure manages the state and buffers needed to generate
/// initialization and media segments for DASH live streaming.
#[derive(Debug)]
pub struct LiveSegmentGenerator {
    /// Representation ID.
    representation_id: String,
    /// Timescale (units per second).
    timescale: u32,
    /// Target segment duration.
    target_duration: Duration,
    /// Current segment buffer.
    current_buffer: BytesMut,
    /// Current segment start time.
    current_start_time: u64,
    /// Current segment duration accumulator.
    current_duration: u64,
    /// Next segment number.
    next_segment_number: u64,
    /// Initialization segment data.
    init_segment: Option<Bytes>,
    /// Codec information.
    codec: CodecInfo,
    /// Segment alignment mode.
    alignment: SegmentAlignment,
    /// Pending packets awaiting segment boundary.
    pending_packets: Vec<Packet>,
}

/// Codec information for segment generation.
#[derive(Debug, Clone)]
pub struct CodecInfo {
    /// Codec string (e.g., "avc1.4d401f").
    pub codec: String,
    /// MIME type.
    pub mime_type: String,
    /// Codec-specific initialization data.
    pub init_data: Option<Bytes>,
    /// Is video codec.
    pub is_video: bool,
}

/// Segment alignment mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentAlignment {
    /// No alignment required.
    None,
    /// Align segments across representations by time.
    Time,
    /// Align segments by presentation time and ensure SAP alignment.
    SapAligned,
}

/// Generated segment data.
#[derive(Debug, Clone)]
pub struct GeneratedSegment {
    /// Segment number.
    pub number: u64,
    /// Segment data.
    pub data: Bytes,
    /// Start time in timescale units.
    pub start_time: u64,
    /// Duration in timescale units.
    pub duration: u64,
    /// Contains keyframe.
    pub has_keyframe: bool,
    /// Size in bytes.
    pub size: usize,
}

impl LiveSegmentGenerator {
    /// Creates a new live segment generator.
    ///
    /// # Arguments
    ///
    /// * `representation_id` - Unique representation identifier
    /// * `timescale` - Timescale in units per second
    /// * `target_duration` - Target segment duration
    /// * `codec` - Codec information
    #[must_use]
    pub fn new(
        representation_id: impl Into<String>,
        timescale: u32,
        target_duration: Duration,
        codec: CodecInfo,
    ) -> Self {
        Self {
            representation_id: representation_id.into(),
            timescale,
            target_duration,
            current_buffer: BytesMut::new(),
            current_start_time: 0,
            current_duration: 0,
            next_segment_number: 1,
            init_segment: None,
            codec,
            alignment: SegmentAlignment::Time,
            pending_packets: Vec::new(),
        }
    }

    /// Sets the segment alignment mode.
    pub fn set_alignment(&mut self, alignment: SegmentAlignment) {
        self.alignment = alignment;
    }

    /// Generates the initialization segment.
    ///
    /// # Arguments
    ///
    /// * `extra_data` - Codec extra data (e.g., SPS/PPS for H.264)
    ///
    /// # Returns
    ///
    /// The initialization segment data
    pub fn generate_init_segment(&mut self, extra_data: Option<&[u8]>) -> Bytes {
        // In a real implementation, this would create an MP4 initialization segment
        // with ftyp, moov, and track metadata boxes
        let mut init = BytesMut::new();

        // Mock ftyp box
        init.extend_from_slice(b"\x00\x00\x00\x18");
        init.extend_from_slice(b"ftyp");
        init.extend_from_slice(b"iso5");
        init.extend_from_slice(b"\x00\x00\x00\x00");
        init.extend_from_slice(b"iso5dash");

        // Mock moov box with track information
        let moov_data = self.create_moov_box(extra_data);
        init.extend_from_slice(&moov_data);

        let bytes = init.freeze();
        self.init_segment = Some(bytes.clone());
        bytes
    }

    /// Returns the cached initialization segment.
    #[must_use]
    pub fn init_segment(&self) -> Option<&Bytes> {
        self.init_segment.as_ref()
    }

    /// Adds a packet to the current segment buffer.
    ///
    /// # Arguments
    ///
    /// * `packet` - The packet to add
    ///
    /// # Returns
    ///
    /// `Some(segment)` if a segment was completed, `None` otherwise
    pub fn add_packet(&mut self, packet: Packet) -> Option<GeneratedSegment> {
        // Rescale PTS to our timescale
        let pts_rescaled = packet
            .timestamp
            .rescale(oximedia_core::Rational::new(1, self.timescale as i64));
        let pts = pts_rescaled.pts as u64;

        // Check if we should finalize the current segment
        if self.should_finalize_segment(&packet, pts) {
            let segment = self.finalize_current_segment();

            // Start new segment
            self.current_start_time = pts;
            self.current_duration = 0;

            // Add packet to new segment
            self.current_buffer.extend_from_slice(&packet.data);
            self.current_duration += self.estimate_packet_duration(&packet);
            self.pending_packets.push(packet);

            return Some(segment);
        }

        // Add to current segment
        self.current_buffer.extend_from_slice(&packet.data);
        self.current_duration += self.estimate_packet_duration(&packet);
        self.pending_packets.push(packet);

        None
    }

    /// Forces finalization of the current segment.
    ///
    /// # Returns
    ///
    /// The completed segment, or `None` if no data is buffered
    pub fn finalize_segment(&mut self) -> Option<GeneratedSegment> {
        if self.current_buffer.is_empty() {
            return None;
        }

        Some(self.finalize_current_segment())
    }

    /// Returns the current segment number.
    #[must_use]
    pub const fn current_segment_number(&self) -> u64 {
        self.next_segment_number
    }

    /// Returns the representation ID.
    #[must_use]
    pub fn representation_id(&self) -> &str {
        &self.representation_id
    }

    /// Returns the timescale.
    #[must_use]
    pub const fn timescale(&self) -> u32 {
        self.timescale
    }

    /// Returns the target segment duration.
    #[must_use]
    pub const fn target_duration(&self) -> Duration {
        self.target_duration
    }

    /// Returns the codec information.
    #[must_use]
    pub fn codec(&self) -> &CodecInfo {
        &self.codec
    }

    /// Checks if the current segment should be finalized.
    fn should_finalize_segment(&self, packet: &Packet, pts: u64) -> bool {
        // Empty buffer means we're starting fresh
        if self.current_buffer.is_empty() {
            return false;
        }

        let target_duration_units = self.duration_to_units(self.target_duration);

        // Check duration threshold
        if self.current_duration >= target_duration_units {
            // For video, wait for a keyframe if SAP-aligned
            if self.codec.is_video && self.alignment == SegmentAlignment::SapAligned {
                return packet
                    .flags
                    .contains(oximedia_container::PacketFlags::KEYFRAME);
            }
            return true;
        }

        // Check for forced boundaries (discontinuities, etc.)
        if pts < self.current_start_time {
            // PTS went backwards, force boundary
            return true;
        }

        false
    }

    /// Finalizes the current segment.
    fn finalize_current_segment(&mut self) -> GeneratedSegment {
        let number = self.next_segment_number;
        self.next_segment_number += 1;

        // Create segment data with moof and mdat boxes
        let segment_data = self.create_segment_boxes();

        let has_keyframe = self
            .pending_packets
            .iter()
            .any(|p| p.flags.contains(oximedia_container::PacketFlags::KEYFRAME));

        let segment = GeneratedSegment {
            number,
            data: segment_data.clone(),
            start_time: self.current_start_time,
            duration: self.current_duration,
            has_keyframe,
            size: segment_data.len(),
        };

        // Reset for next segment
        self.current_buffer.clear();
        self.pending_packets.clear();
        self.current_duration = 0;

        segment
    }

    /// Creates MP4 segment boxes (moof + mdat).
    fn create_segment_boxes(&self) -> Bytes {
        let mut segment = BytesMut::new();

        // Mock moof box
        let moof_data = self.create_moof_box();
        segment.extend_from_slice(&moof_data);

        // mdat box with media data
        let mdat_size = 8 + self.current_buffer.len();
        segment.extend_from_slice(&(mdat_size as u32).to_be_bytes());
        segment.extend_from_slice(b"mdat");
        segment.extend_from_slice(&self.current_buffer);

        segment.freeze()
    }

    /// Creates a mock moof (movie fragment) box.
    fn create_moof_box(&self) -> Vec<u8> {
        let mut moof = Vec::new();

        // Simplified moof structure
        let moof_header = vec![
            0x00, 0x00, 0x00, 0x28, // size
            b'm', b'o', b'o', b'f', // mfhd box
            0x00, 0x00, 0x00, 0x10, b'm', b'f', b'h', b'd', 0x00, 0x00, 0x00, 0x00,
        ];
        moof.extend_from_slice(&moof_header);

        // Add sequence number
        moof.extend_from_slice(&(self.next_segment_number as u32).to_be_bytes());

        // In a real implementation, this would include traf boxes with timing info
        moof
    }

    /// Creates a mock moov box for initialization.
    fn create_moov_box(&self, _extra_data: Option<&[u8]>) -> Vec<u8> {
        let mut moov = Vec::new();

        // Simplified moov structure
        let moov_header = vec![
            0x00, 0x00, 0x00, 0x40, // size (placeholder)
            b'm', b'o', b'o', b'v', // mvhd box
            0x00, 0x00, 0x00, 0x20, b'm', b'v', b'h', b'd', 0x00, 0x00, 0x00,
            0x00, // version + flags
        ];
        moov.extend_from_slice(&moov_header);

        // Timescale
        moov.extend_from_slice(&self.timescale.to_be_bytes());

        // Duration (unknown for live)
        moov.extend_from_slice(&[0, 0, 0, 0]);

        // In a real implementation, this would include trak boxes
        moov
    }

    /// Estimates the duration of a packet.
    fn estimate_packet_duration(&self, _packet: &Packet) -> u64 {
        // In a real implementation, this would calculate based on:
        // - Frame rate for video
        // - Sample count for audio
        // For now, use a simple estimate
        self.duration_to_units(self.target_duration) / 60 // Assume 60 packets per segment
    }

    /// Converts a duration to timescale units.
    fn duration_to_units(&self, duration: Duration) -> u64 {
        (duration.as_secs_f64() * f64::from(self.timescale)) as u64
    }
}

/// Manages multiple segment generators for different representations.
#[derive(Debug)]
pub struct MultiRepresentationGenerator {
    /// Generators indexed by representation ID.
    generators: HashMap<String, LiveSegmentGenerator>,
    /// Segment alignment coordinator.
    alignment: AlignmentCoordinator,
}

impl MultiRepresentationGenerator {
    /// Creates a new multi-representation generator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            generators: HashMap::new(),
            alignment: AlignmentCoordinator::new(),
        }
    }

    /// Adds a representation.
    pub fn add_representation(&mut self, generator: LiveSegmentGenerator) {
        let id = generator.representation_id().to_string();
        self.generators.insert(id, generator);
    }

    /// Adds a packet to the appropriate representation.
    ///
    /// # Arguments
    ///
    /// * `representation_id` - The representation to add to
    /// * `packet` - The packet to add
    ///
    /// # Returns
    ///
    /// Generated segment if one was completed
    pub fn add_packet(
        &mut self,
        representation_id: &str,
        packet: Packet,
    ) -> Option<GeneratedSegment> {
        let generator = self.generators.get_mut(representation_id)?;
        let segment = generator.add_packet(packet)?;

        // Update alignment coordinator
        self.alignment
            .register_segment(representation_id, segment.number, segment.start_time);

        Some(segment)
    }

    /// Forces segment finalization for all representations.
    pub fn finalize_all(&mut self) -> Vec<(String, GeneratedSegment)> {
        let mut segments = Vec::new();

        for (id, generator) in &mut self.generators {
            if let Some(segment) = generator.finalize_segment() {
                segments.push((id.clone(), segment));
            }
        }

        segments
    }

    /// Returns the generator for a representation.
    #[must_use]
    pub fn generator(&self, representation_id: &str) -> Option<&LiveSegmentGenerator> {
        self.generators.get(representation_id)
    }

    /// Returns a mutable reference to a generator.
    pub fn generator_mut(&mut self, representation_id: &str) -> Option<&mut LiveSegmentGenerator> {
        self.generators.get_mut(representation_id)
    }

    /// Returns all representation IDs.
    pub fn representation_ids(&self) -> Vec<&str> {
        self.generators.keys().map(String::as_str).collect()
    }

    /// Checks if representations are aligned at a given segment number.
    #[must_use]
    pub fn are_aligned(&self, segment_number: u64) -> bool {
        self.alignment.are_aligned(segment_number)
    }
}

impl Default for MultiRepresentationGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Coordinates segment alignment across representations.
#[derive(Debug)]
struct AlignmentCoordinator {
    /// Segment timing info by representation.
    segments: HashMap<String, Vec<SegmentTiming>>,
}

#[derive(Debug, Clone, Copy)]
struct SegmentTiming {
    number: u64,
    start_time: u64,
}

impl AlignmentCoordinator {
    fn new() -> Self {
        Self {
            segments: HashMap::new(),
        }
    }

    fn register_segment(&mut self, representation_id: &str, number: u64, start_time: u64) {
        let timing = SegmentTiming { number, start_time };
        self.segments
            .entry(representation_id.to_string())
            .or_default()
            .push(timing);
    }

    fn are_aligned(&self, segment_number: u64) -> bool {
        if self.segments.len() <= 1 {
            return true;
        }

        let mut start_times = Vec::new();
        for timings in self.segments.values() {
            if let Some(timing) = timings.iter().find(|t| t.number == segment_number) {
                start_times.push(timing.start_time);
            } else {
                return false; // Not all representations have this segment
            }
        }

        // Check if all start times are the same (or within a small threshold)
        if start_times.is_empty() {
            return false;
        }

        let first = start_times[0];
        start_times.iter().all(|&t| t == first)
    }
}

impl GeneratedSegment {
    /// Returns the segment duration in seconds.
    #[must_use]
    pub fn duration_secs(&self, timescale: u32) -> f64 {
        self.duration as f64 / timescale as f64
    }

    /// Returns the start time in seconds.
    #[must_use]
    pub fn start_time_secs(&self, timescale: u32) -> f64 {
        self.start_time as f64 / timescale as f64
    }
}

impl CodecInfo {
    /// Creates codec info for H.264/AVC.
    #[must_use]
    pub fn h264(profile: u8, level: u8) -> Self {
        let codec = format!("avc1.{profile:02x}{level:02x}");
        Self {
            codec,
            mime_type: "video/mp4".to_string(),
            init_data: None,
            is_video: true,
        }
    }

    /// Creates codec info for H.265/HEVC.
    #[must_use]
    pub fn h265() -> Self {
        Self {
            codec: "hvc1.1.6.L93.B0".to_string(),
            mime_type: "video/mp4".to_string(),
            init_data: None,
            is_video: true,
        }
    }

    /// Creates codec info for AAC audio.
    #[must_use]
    pub fn aac() -> Self {
        Self {
            codec: "mp4a.40.2".to_string(),
            mime_type: "audio/mp4".to_string(),
            init_data: None,
            is_video: false,
        }
    }

    /// Creates codec info for Opus audio.
    #[must_use]
    pub fn opus() -> Self {
        Self {
            codec: "opus".to_string(),
            mime_type: "audio/webm".to_string(),
            init_data: None,
            is_video: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_container::PacketFlags;
    use oximedia_core::{Rational, Timestamp};

    fn create_test_packet(pts: u64, is_keyframe: bool) -> Packet {
        let flags = if is_keyframe {
            PacketFlags::KEYFRAME
        } else {
            PacketFlags::empty()
        };

        Packet::new(
            0,
            Bytes::from(vec![0u8; 1024]),
            Timestamp::new(pts as i64, Rational::new(1, 90000)),
            flags,
        )
    }

    #[test]
    fn test_segment_generator_creation() {
        let codec = CodecInfo::h264(0x4d, 0x40);
        let gen = LiveSegmentGenerator::new("720p", 90000, Duration::from_secs(2), codec);

        assert_eq!(gen.representation_id(), "720p");
        assert_eq!(gen.timescale(), 90000);
        assert_eq!(gen.current_segment_number(), 1);
    }

    #[test]
    fn test_init_segment_generation() {
        let codec = CodecInfo::h264(0x4d, 0x40);
        let mut gen = LiveSegmentGenerator::new("720p", 90000, Duration::from_secs(2), codec);

        let init = gen.generate_init_segment(None);
        assert!(!init.is_empty());
        assert!(gen.init_segment().is_some());
    }

    #[test]
    fn test_add_packet() {
        let codec = CodecInfo::h264(0x4d, 0x40);
        let mut gen = LiveSegmentGenerator::new("720p", 90000, Duration::from_secs(2), codec);

        let packet = create_test_packet(0, true);
        let result = gen.add_packet(packet);

        // First packet shouldn't complete a segment
        assert!(result.is_none());
    }

    #[test]
    fn test_codec_info() {
        let h264 = CodecInfo::h264(0x4d, 0x40);
        assert_eq!(h264.codec, "avc1.4d40");
        assert!(h264.is_video);

        let aac = CodecInfo::aac();
        assert_eq!(aac.codec, "mp4a.40.2");
        assert!(!aac.is_video);
    }

    #[test]
    fn test_multi_representation_generator() {
        let mut multi = MultiRepresentationGenerator::new();

        let codec = CodecInfo::h264(0x4d, 0x40);
        let gen = LiveSegmentGenerator::new("720p", 90000, Duration::from_secs(2), codec);
        multi.add_representation(gen);

        assert!(multi.generator("720p").is_some());
        assert!(multi.generator("1080p").is_none());
    }
}
