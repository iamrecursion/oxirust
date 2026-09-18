//! DVR (Digital Video Recording) buffer for time-shifting.
//!
//! This module implements a time-based buffer that allows viewers to:
//! - Seek to earlier points in the live stream
//! - Pause and resume live playback
//! - Replay recent content

use super::{MediaPacket, MediaType};
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

/// DVR configuration.
#[derive(Debug, Clone)]
pub struct DvrConfig {
    /// DVR window duration (how far back viewers can seek).
    pub window_duration: Duration,

    /// Segment duration for DVR indexing.
    pub segment_duration: Duration,
}

impl Default for DvrConfig {
    fn default() -> Self {
        Self {
            window_duration: Duration::from_secs(3600), // 1 hour
            segment_duration: Duration::from_secs(2),
        }
    }
}

/// DVR segment containing packets.
#[derive(Debug, Clone)]
pub struct DvrSegment {
    /// Segment start timestamp (milliseconds).
    pub start_timestamp: u64,

    /// Segment end timestamp (milliseconds).
    pub end_timestamp: u64,

    /// Video packets.
    pub video_packets: Vec<MediaPacket>,

    /// Audio packets.
    pub audio_packets: Vec<MediaPacket>,

    /// Metadata packets.
    pub metadata_packets: Vec<MediaPacket>,
}

impl DvrSegment {
    /// Creates a new DVR segment.
    #[must_use]
    pub fn new(start_timestamp: u64) -> Self {
        Self {
            start_timestamp,
            end_timestamp: start_timestamp,
            video_packets: Vec::new(),
            audio_packets: Vec::new(),
            metadata_packets: Vec::new(),
        }
    }

    /// Adds a packet to the segment.
    pub fn add_packet(&mut self, packet: MediaPacket) {
        self.end_timestamp = self.end_timestamp.max(packet.timestamp);

        match packet.media_type {
            MediaType::Video => self.video_packets.push(packet),
            MediaType::Audio => self.audio_packets.push(packet),
            MediaType::Metadata => self.metadata_packets.push(packet),
        }
    }

    /// Returns the segment duration in milliseconds.
    #[must_use]
    pub fn duration(&self) -> u64 {
        self.end_timestamp.saturating_sub(self.start_timestamp)
    }

    /// Returns total packet count.
    #[must_use]
    pub fn packet_count(&self) -> usize {
        self.video_packets.len() + self.audio_packets.len() + self.metadata_packets.len()
    }

    /// Returns total size in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.video_packets
            .iter()
            .map(|p| p.data.len())
            .sum::<usize>()
            + self
                .audio_packets
                .iter()
                .map(|p| p.data.len())
                .sum::<usize>()
            + self
                .metadata_packets
                .iter()
                .map(|p| p.data.len())
                .sum::<usize>()
    }

    /// Checks if the segment contains a specific timestamp.
    #[must_use]
    pub fn contains_timestamp(&self, timestamp: u64) -> bool {
        timestamp >= self.start_timestamp && timestamp <= self.end_timestamp
    }
}

/// DVR buffer for time-shifting.
pub struct DvrBuffer {
    /// Configuration.
    config: DvrConfig,

    /// Buffered segments.
    segments: VecDeque<Arc<RwLock<DvrSegment>>>,

    /// Current segment being written.
    current_segment: Option<Arc<RwLock<DvrSegment>>>,

    /// Current segment start timestamp.
    current_segment_start: u64,

    /// Total packets buffered.
    total_packets: usize,

    /// Total bytes buffered.
    total_bytes: usize,
}

impl DvrBuffer {
    /// Creates a new DVR buffer.
    #[must_use]
    pub fn new(config: DvrConfig) -> Self {
        Self {
            config,
            segments: VecDeque::new(),
            current_segment: None,
            current_segment_start: 0,
            total_packets: 0,
            total_bytes: 0,
        }
    }

    /// Adds a media packet to the buffer.
    pub fn add_packet(&mut self, packet: MediaPacket) {
        let segment_duration_ms = self.config.segment_duration.as_millis() as u64;

        // Check if we need a new segment
        let need_new_segment = if let Some(_current) = &self.current_segment {
            let elapsed = packet.timestamp.saturating_sub(self.current_segment_start);
            elapsed >= segment_duration_ms
        } else {
            true
        };

        if need_new_segment {
            self.finalize_current_segment();
            self.current_segment_start = packet.timestamp;
            let new_segment = Arc::new(RwLock::new(DvrSegment::new(packet.timestamp)));
            self.current_segment = Some(new_segment);
        }

        // Add packet to current segment
        if let Some(segment) = &self.current_segment {
            let mut seg = segment.write();
            self.total_bytes += packet.data.len();
            self.total_packets += 1;
            seg.add_packet(packet);
        }

        // Trim old segments
        self.trim_old_segments();
    }

    /// Finalizes the current segment and adds it to the buffer.
    fn finalize_current_segment(&mut self) {
        if let Some(segment) = self.current_segment.take() {
            self.segments.push_back(segment);
        }
    }

    /// Trims segments outside the DVR window.
    fn trim_old_segments(&mut self) {
        if self.segments.is_empty() {
            return;
        }

        let window_ms = self.config.window_duration.as_millis() as u64;

        // Get the latest timestamp
        let latest_timestamp = if let Some(current) = &self.current_segment {
            let seg = current.read();
            seg.end_timestamp
        } else if let Some(last) = self.segments.back() {
            let seg = last.read();
            seg.end_timestamp
        } else {
            return;
        };

        // Remove segments older than the window
        let cutoff_timestamp = latest_timestamp.saturating_sub(window_ms);

        while let Some(first) = self.segments.front() {
            let seg = first.read();
            if seg.end_timestamp < cutoff_timestamp {
                self.total_packets -= seg.packet_count();
                self.total_bytes -= seg.size_bytes();
                drop(seg);
                self.segments.pop_front();
            } else {
                break;
            }
        }
    }

    /// Gets packets in a time range.
    #[must_use]
    pub fn get_packets_in_range(
        &self,
        start_timestamp: u64,
        end_timestamp: u64,
    ) -> Vec<MediaPacket> {
        let mut packets = Vec::new();

        for segment in &self.segments {
            let seg = segment.read();

            if seg.end_timestamp < start_timestamp {
                continue;
            }

            if seg.start_timestamp > end_timestamp {
                break;
            }

            // Add video packets in range
            for packet in &seg.video_packets {
                if packet.timestamp >= start_timestamp && packet.timestamp <= end_timestamp {
                    packets.push(packet.clone());
                }
            }

            // Add audio packets in range
            for packet in &seg.audio_packets {
                if packet.timestamp >= start_timestamp && packet.timestamp <= end_timestamp {
                    packets.push(packet.clone());
                }
            }
        }

        // Sort by timestamp
        packets.sort_by_key(|p| p.timestamp);
        packets
    }

    /// Gets all packets of a specific media type.
    #[must_use]
    pub fn get_packets_by_type(&self, media_type: MediaType) -> Vec<MediaPacket> {
        let mut packets = Vec::new();

        for segment in &self.segments {
            let seg = segment.read();

            let segment_packets = match media_type {
                MediaType::Video => &seg.video_packets,
                MediaType::Audio => &seg.audio_packets,
                MediaType::Metadata => &seg.metadata_packets,
            };

            packets.extend(segment_packets.iter().cloned());
        }

        packets
    }

    /// Gets the earliest available timestamp.
    #[must_use]
    pub fn earliest_timestamp(&self) -> Option<u64> {
        self.segments.front().map(|seg| seg.read().start_timestamp)
    }

    /// Gets the latest available timestamp.
    #[must_use]
    pub fn latest_timestamp(&self) -> Option<u64> {
        if let Some(current) = &self.current_segment {
            return Some(current.read().end_timestamp);
        }

        self.segments.back().map(|seg| seg.read().end_timestamp)
    }

    /// Gets the total buffered duration.
    #[must_use]
    pub fn buffered_duration(&self) -> Duration {
        if let (Some(earliest), Some(latest)) = (self.earliest_timestamp(), self.latest_timestamp())
        {
            let duration_ms = latest.saturating_sub(earliest);
            Duration::from_millis(duration_ms)
        } else {
            Duration::ZERO
        }
    }

    /// Returns segment count.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Returns total packet count.
    #[must_use]
    pub fn packet_count(&self) -> usize {
        self.total_packets
    }

    /// Returns total bytes buffered.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Clears the buffer.
    pub fn clear(&mut self) {
        self.segments.clear();
        self.current_segment = None;
        self.total_packets = 0;
        self.total_bytes = 0;
    }

    /// Gets DVR statistics.
    #[must_use]
    pub fn stats(&self) -> DvrStats {
        DvrStats {
            segment_count: self.segment_count(),
            packet_count: self.packet_count(),
            total_bytes: self.total_bytes(),
            buffered_duration: self.buffered_duration(),
            earliest_timestamp: self.earliest_timestamp(),
            latest_timestamp: self.latest_timestamp(),
        }
    }
}

/// DVR buffer statistics.
#[derive(Debug, Clone)]
pub struct DvrStats {
    /// Number of segments.
    pub segment_count: usize,

    /// Total packets.
    pub packet_count: usize,

    /// Total bytes.
    pub total_bytes: usize,

    /// Buffered duration.
    pub buffered_duration: Duration,

    /// Earliest timestamp.
    pub earliest_timestamp: Option<u64>,

    /// Latest timestamp.
    pub latest_timestamp: Option<u64>,
}

/// DVR playback session.
pub struct DvrPlayback {
    /// DVR buffer reference.
    buffer: Arc<RwLock<DvrBuffer>>,

    /// Current playback position (timestamp).
    position: u64,

    /// Playback speed (1.0 = normal).
    speed: f64,

    /// Is paused.
    paused: bool,
}

impl DvrPlayback {
    /// Creates a new DVR playback session.
    #[must_use]
    pub fn new(buffer: Arc<RwLock<DvrBuffer>>, start_position: u64) -> Self {
        Self {
            buffer,
            position: start_position,
            speed: 1.0,
            paused: false,
        }
    }

    /// Seeks to a specific timestamp.
    pub fn seek(&mut self, timestamp: u64) -> bool {
        let buf = self.buffer.read();

        if let (Some(earliest), Some(latest)) = (buf.earliest_timestamp(), buf.latest_timestamp()) {
            if timestamp >= earliest && timestamp <= latest {
                self.position = timestamp;
                return true;
            }
        }

        false
    }

    /// Seeks to live edge.
    pub fn seek_to_live(&mut self) {
        let buf = self.buffer.read();
        if let Some(latest) = buf.latest_timestamp() {
            self.position = latest;
        }
    }

    /// Pauses playback.
    pub fn pause(&mut self) {
        self.paused = true;
    }

    /// Resumes playback.
    pub fn resume(&mut self) {
        self.paused = false;
    }

    /// Sets playback speed.
    pub fn set_speed(&mut self, speed: f64) {
        self.speed = speed.max(0.25).min(4.0);
    }

    /// Gets next packet for playback.
    #[must_use]
    pub fn next_packet(&mut self) -> Option<MediaPacket> {
        if self.paused {
            return None;
        }

        let buf = self.buffer.read();
        let packets = buf.get_packets_in_range(self.position, self.position + 1000);

        if let Some(packet) = packets.first() {
            self.position = packet.timestamp + packet.duration;
            Some(packet.clone())
        } else {
            None
        }
    }

    /// Checks if playback is at live edge.
    #[must_use]
    pub fn is_at_live(&self) -> bool {
        let buf = self.buffer.read();
        if let Some(latest) = buf.latest_timestamp() {
            latest.saturating_sub(self.position) < 5000 // Within 5 seconds
        } else {
            false
        }
    }

    /// Gets current position.
    #[must_use]
    pub const fn position(&self) -> u64 {
        self.position
    }

    /// Gets playback speed.
    #[must_use]
    pub const fn speed(&self) -> f64 {
        self.speed
    }

    /// Checks if paused.
    #[must_use]
    pub const fn is_paused(&self) -> bool {
        self.paused
    }
}
