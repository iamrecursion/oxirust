//! Segment generation for live streaming.
//!
//! This module handles the generation of media segments for HLS and DASH,
//! including segmentation, packaging, and caching.

use super::{MediaPacket, MediaType};
use crate::error::NetResult;
use bytes::{Bytes, BytesMut};
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

/// Segment configuration.
#[derive(Debug, Clone)]
pub struct SegmentConfig {
    /// Target segment duration.
    pub duration: Duration,

    /// Keyframe interval (for forcing segmentation).
    pub keyframe_interval: u64,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            duration: Duration::from_secs(2),
            keyframe_interval: 60,
        }
    }
}

/// Media segment.
#[derive(Debug, Clone)]
pub struct MediaSegment {
    /// Segment ID.
    pub id: Uuid,

    /// Segment sequence number.
    pub sequence: u64,

    /// Segment start timestamp (milliseconds).
    pub start_timestamp: u64,

    /// Segment duration (milliseconds).
    pub duration: u64,

    /// Segment data.
    pub data: Bytes,

    /// Is initialization segment.
    pub is_init: bool,

    /// Contains keyframe.
    pub has_keyframe: bool,

    /// Variant ID.
    pub variant_id: Option<String>,

    /// Media type.
    pub media_type: MediaType,

    /// Byte range offset (for packed segments).
    pub byte_offset: Option<u64>,

    /// Byte range length.
    pub byte_length: Option<u64>,
}

impl MediaSegment {
    /// Creates a new media segment.
    #[must_use]
    pub fn new(
        sequence: u64,
        start_timestamp: u64,
        duration: u64,
        data: Bytes,
        media_type: MediaType,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            sequence,
            start_timestamp,
            duration,
            data,
            is_init: false,
            has_keyframe: false,
            variant_id: None,
            media_type,
            byte_offset: None,
            byte_length: None,
        }
    }

    /// Creates an initialization segment.
    #[must_use]
    pub fn init_segment(data: Bytes, media_type: MediaType) -> Self {
        Self {
            id: Uuid::new_v4(),
            sequence: 0,
            start_timestamp: 0,
            duration: 0,
            data,
            is_init: true,
            has_keyframe: false,
            variant_id: None,
            media_type,
            byte_offset: None,
            byte_length: None,
        }
    }

    /// Returns the segment filename for HLS.
    #[must_use]
    pub fn hls_filename(&self) -> String {
        if self.is_init {
            format!("init_{}.mp4", self.id)
        } else {
            format!("seg_{}_{}.m4s", self.sequence, self.id)
        }
    }

    /// Returns the segment filename for DASH.
    #[must_use]
    pub fn dash_filename(&self) -> String {
        let media_str = match self.media_type {
            MediaType::Video => "video",
            MediaType::Audio => "audio",
            MediaType::Metadata => "metadata",
        };

        if self.is_init {
            format!("init_{media_str}.mp4")
        } else {
            format!("{media_str}_{}_{}.m4s", self.sequence, self.id)
        }
    }

    /// Returns the duration in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        self.duration as f64 / 1000.0
    }
}

/// Segmentation state for a stream.
struct SegmentationState {
    /// Current sequence number.
    sequence: u64,

    /// Packets accumulated for current segment.
    current_packets: Vec<MediaPacket>,

    /// Current segment start timestamp.
    current_start_ts: Option<u64>,

    /// Last keyframe timestamp.
    last_keyframe_ts: u64,

    /// Completed segments.
    segments: VecDeque<Arc<MediaSegment>>,

    /// Maximum segments to keep.
    max_segments: usize,
}

impl SegmentationState {
    fn new() -> Self {
        Self {
            sequence: 0,
            current_packets: Vec::new(),
            current_start_ts: None,
            last_keyframe_ts: 0,
            segments: VecDeque::new(),
            max_segments: 100,
        }
    }

    fn add_packet(&mut self, packet: MediaPacket) {
        if self.current_start_ts.is_none() {
            self.current_start_ts = Some(packet.timestamp);
        }
        self.current_packets.push(packet);
    }

    fn should_finalize(&self, packet: &MediaPacket, target_duration_ms: u64) -> bool {
        if let Some(start_ts) = self.current_start_ts {
            let elapsed = packet.timestamp.saturating_sub(start_ts);
            if packet.keyframe && elapsed >= target_duration_ms {
                return true;
            }
        }
        false
    }

    fn finalize_segment(&mut self, media_type: MediaType) -> Option<Arc<MediaSegment>> {
        if self.current_packets.is_empty() {
            return None;
        }

        let start_ts = self.current_start_ts?;
        let last_ts = self.current_packets.last()?.timestamp;
        let duration = last_ts.saturating_sub(start_ts);

        // Package packets into segment data
        let data = Self::package_packets(&self.current_packets);

        let _has_keyframe = self.current_packets.iter().any(|p| p.keyframe);

        let segment = Arc::new(MediaSegment::new(
            self.sequence,
            start_ts,
            duration,
            data,
            media_type,
        ));

        self.sequence += 1;
        self.current_packets.clear();
        self.current_start_ts = None;

        // Store segment
        self.segments.push_back(segment.clone());
        if self.segments.len() > self.max_segments {
            self.segments.pop_front();
        }

        Some(segment)
    }

    fn package_packets(packets: &[MediaPacket]) -> Bytes {
        // Simple packaging - in production, would use proper MP4/fMP4 packaging
        let mut buf = BytesMut::new();
        for packet in packets {
            buf.extend_from_slice(&packet.data);
        }
        buf.freeze()
    }

    fn get_segments(&self, count: usize) -> Vec<Arc<MediaSegment>> {
        self.segments
            .iter()
            .rev()
            .take(count)
            .rev()
            .cloned()
            .collect()
    }

    fn get_segment_by_sequence(&self, sequence: u64) -> Option<Arc<MediaSegment>> {
        self.segments
            .iter()
            .find(|s| s.sequence == sequence)
            .cloned()
    }
}

/// Segment generator.
pub struct SegmentGenerator {
    /// Configuration.
    config: SegmentConfig,

    /// Video segmentation state.
    video_state: RwLock<SegmentationState>,

    /// Audio segmentation state.
    audio_state: RwLock<SegmentationState>,

    /// Initialization segment for video.
    video_init: RwLock<Option<Arc<MediaSegment>>>,

    /// Initialization segment for audio.
    audio_init: RwLock<Option<Arc<MediaSegment>>>,
}

impl SegmentGenerator {
    /// Creates a new segment generator.
    #[must_use]
    pub fn new(config: SegmentConfig) -> Self {
        Self {
            config,
            video_state: RwLock::new(SegmentationState::new()),
            audio_state: RwLock::new(SegmentationState::new()),
            video_init: RwLock::new(None),
            audio_init: RwLock::new(None),
        }
    }

    /// Adds a media packet.
    pub fn add_packet(&self, packet: &MediaPacket) -> NetResult<()> {
        let target_duration_ms = self.config.duration.as_millis() as u64;

        match packet.media_type {
            MediaType::Video => {
                let mut state = self.video_state.write();

                if state.should_finalize(packet, target_duration_ms) {
                    if let Some(_segment) = state.finalize_segment(MediaType::Video) {
                        // Segment finalized - could emit event here
                    }
                }

                state.add_packet(packet.clone());
            }
            MediaType::Audio => {
                let mut state = self.audio_state.write();

                if state.should_finalize(packet, target_duration_ms) {
                    if let Some(_segment) = state.finalize_segment(MediaType::Audio) {
                        // Segment finalized
                    }
                }

                state.add_packet(packet.clone());
            }
            MediaType::Metadata => {
                // Handle metadata packets
            }
        }

        Ok(())
    }

    /// Gets recent video segments.
    #[must_use]
    pub fn get_video_segments(&self, count: usize) -> Vec<Arc<MediaSegment>> {
        let state = self.video_state.read();
        state.get_segments(count)
    }

    /// Gets recent audio segments.
    #[must_use]
    pub fn get_audio_segments(&self, count: usize) -> Vec<Arc<MediaSegment>> {
        let state = self.audio_state.read();
        state.get_segments(count)
    }

    /// Gets video segment by sequence.
    #[must_use]
    pub fn get_video_segment(&self, sequence: u64) -> Option<Arc<MediaSegment>> {
        let state = self.video_state.read();
        state.get_segment_by_sequence(sequence)
    }

    /// Gets audio segment by sequence.
    #[must_use]
    pub fn get_audio_segment(&self, sequence: u64) -> Option<Arc<MediaSegment>> {
        let state = self.audio_state.read();
        state.get_segment_by_sequence(sequence)
    }

    /// Sets video initialization segment.
    pub fn set_video_init(&self, data: Bytes) {
        let segment = Arc::new(MediaSegment::init_segment(data, MediaType::Video));
        *self.video_init.write() = Some(segment);
    }

    /// Sets audio initialization segment.
    pub fn set_audio_init(&self, data: Bytes) {
        let segment = Arc::new(MediaSegment::init_segment(data, MediaType::Audio));
        *self.audio_init.write() = Some(segment);
    }

    /// Gets video initialization segment.
    #[must_use]
    pub fn get_video_init(&self) -> Option<Arc<MediaSegment>> {
        self.video_init.read().clone()
    }

    /// Gets audio initialization segment.
    #[must_use]
    pub fn get_audio_init(&self) -> Option<Arc<MediaSegment>> {
        self.audio_init.read().clone()
    }

    /// Forces finalization of current segments.
    pub fn flush(&self) {
        {
            let mut state = self.video_state.write();
            state.finalize_segment(MediaType::Video);
        }
        {
            let mut state = self.audio_state.write();
            state.finalize_segment(MediaType::Audio);
        }
    }

    /// Resets the segment generator.
    pub fn reset(&self) {
        *self.video_state.write() = SegmentationState::new();
        *self.audio_state.write() = SegmentationState::new();
        *self.video_init.write() = None;
        *self.audio_init.write() = None;
    }

    /// Returns current video sequence number.
    #[must_use]
    pub fn video_sequence(&self) -> u64 {
        self.video_state.read().sequence
    }

    /// Returns current audio sequence number.
    #[must_use]
    pub fn audio_sequence(&self) -> u64 {
        self.audio_state.read().sequence
    }
}

/// Segment cache for efficient serving.
pub struct SegmentCache {
    /// Cached segments.
    cache: RwLock<VecDeque<Arc<MediaSegment>>>,

    /// Maximum cache size.
    max_size: usize,
}

impl SegmentCache {
    /// Creates a new segment cache.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: RwLock::new(VecDeque::with_capacity(max_size)),
            max_size,
        }
    }

    /// Adds a segment to the cache.
    pub fn add(&self, segment: Arc<MediaSegment>) {
        let mut cache = self.cache.write();
        cache.push_back(segment);
        if cache.len() > self.max_size {
            cache.pop_front();
        }
    }

    /// Gets a segment by sequence number.
    #[must_use]
    pub fn get(&self, sequence: u64) -> Option<Arc<MediaSegment>> {
        let cache = self.cache.read();
        cache.iter().find(|s| s.sequence == sequence).cloned()
    }

    /// Gets the most recent segments.
    #[must_use]
    pub fn get_recent(&self, count: usize) -> Vec<Arc<MediaSegment>> {
        let cache = self.cache.read();
        cache.iter().rev().take(count).rev().cloned().collect()
    }

    /// Clears the cache.
    pub fn clear(&self) {
        self.cache.write().clear();
    }

    /// Returns cache size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.cache.read().len()
    }
}
