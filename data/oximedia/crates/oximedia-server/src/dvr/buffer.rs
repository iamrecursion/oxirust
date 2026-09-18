//! DVR ring buffer for time-shifting.

use oximedia_net::rtmp::MediaPacket;
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::time::Duration;

/// DVR configuration.
#[derive(Debug, Clone)]
pub struct DvrConfig {
    /// Window duration (how far back viewers can seek).
    pub window_duration: Duration,

    /// Segment duration.
    pub segment_duration: Duration,

    /// Maximum buffer size in bytes.
    pub max_buffer_size: usize,
}

impl Default for DvrConfig {
    fn default() -> Self {
        Self {
            window_duration: Duration::from_secs(3600), // 1 hour
            segment_duration: Duration::from_secs(2),
            max_buffer_size: 500 * 1024 * 1024, // 500 MB
        }
    }
}

/// Buffered segment.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BufferedSegment {
    /// Segment number.
    number: u64,

    /// Start timestamp.
    start_timestamp: u64,

    /// End timestamp.
    end_timestamp: u64,

    /// Packets in this segment.
    packets: Vec<MediaPacket>,

    /// Segment size in bytes.
    size: usize,
}

/// DVR ring buffer.
pub struct DvrBuffer {
    /// Configuration.
    config: DvrConfig,

    /// Buffered segments.
    segments: RwLock<VecDeque<BufferedSegment>>,

    /// Current segment being built.
    current_segment: RwLock<Option<BufferedSegment>>,

    /// Next segment number.
    next_segment_number: RwLock<u64>,

    /// Total buffer size.
    total_size: RwLock<usize>,
}

impl DvrBuffer {
    /// Creates a new DVR buffer.
    #[must_use]
    pub fn new(config: DvrConfig) -> Self {
        Self {
            config,
            segments: RwLock::new(VecDeque::new()),
            current_segment: RwLock::new(None),
            next_segment_number: RwLock::new(0),
            total_size: RwLock::new(0),
        }
    }

    /// Adds a packet to the buffer.
    pub fn add_packet(&self, packet: MediaPacket) {
        let mut current = self.current_segment.write();

        // Initialize current segment if needed
        if current.is_none() {
            let number = *self.next_segment_number.read();
            *current = Some(BufferedSegment {
                number,
                start_timestamp: packet.timestamp as u64,
                end_timestamp: packet.timestamp as u64,
                packets: Vec::new(),
                size: 0,
            });
        }

        if let Some(segment) = current.as_mut() {
            segment.end_timestamp = packet.timestamp as u64;
            segment.size += packet.data.len();
            segment.packets.push(packet);

            // Check if we should finalize this segment
            let duration_ms = segment.end_timestamp - segment.start_timestamp;
            if duration_ms >= self.config.segment_duration.as_millis() as u64 {
                // Finalize segment. `current` is provably `Some` here (we are
                // inside the `if let Some(segment) = current.as_mut()` arm
                // above), but we still match instead of `.expect()`-ing so
                // this path can never panic even under future refactors.
                if let Some(finished_segment) = current.take() {
                    self.finalize_segment(finished_segment);
                }
            }
        }
    }

    /// Finalizes a segment and adds it to the buffer.
    fn finalize_segment(&self, segment: BufferedSegment) {
        let segment_size = segment.size;

        let mut segments = self.segments.write();
        segments.push_back(segment);

        // Update total size
        *self.total_size.write() += segment_size;

        // Remove old segments if needed
        self.remove_old_segments();

        // Increment segment number
        *self.next_segment_number.write() += 1;
    }

    /// Removes old segments based on window duration and max buffer size.
    fn remove_old_segments(&self) {
        let mut segments = self.segments.write();
        let mut total_size = self.total_size.write();

        if segments.is_empty() {
            return;
        }

        // Get the latest timestamp
        let latest_ts = segments.back().map(|s| s.end_timestamp).unwrap_or(0);

        // Remove segments outside the window
        while let Some(first) = segments.front() {
            let age_ms = latest_ts.saturating_sub(first.start_timestamp);

            if age_ms > self.config.window_duration.as_millis() as u64
                || *total_size > self.config.max_buffer_size
            {
                if let Some(removed) = segments.pop_front() {
                    *total_size = total_size.saturating_sub(removed.size);
                }
            } else {
                break;
            }
        }
    }

    /// Gets segments within a time range.
    #[must_use]
    pub fn get_segments(&self, start_time: u64, end_time: u64) -> Vec<MediaPacket> {
        let segments = self.segments.read();
        let mut packets = Vec::new();

        for segment in segments.iter() {
            // Check if segment overlaps with requested range
            if segment.start_timestamp <= end_time && segment.end_timestamp >= start_time {
                for packet in &segment.packets {
                    if (packet.timestamp as u64) >= start_time
                        && (packet.timestamp as u64) <= end_time
                    {
                        packets.push(packet.clone());
                    }
                }
            }
        }

        packets
    }

    /// Gets all segments.
    #[must_use]
    pub fn get_all_segments(&self) -> Vec<MediaPacket> {
        let segments = self.segments.read();
        let mut packets = Vec::new();

        for segment in segments.iter() {
            packets.extend_from_slice(&segment.packets);
        }

        packets
    }

    /// Gets the number of segments in the buffer.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        let segments = self.segments.read();
        segments.len()
    }

    /// Gets the total buffer size.
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        *self.total_size.read()
    }

    /// Gets the time range of buffered data.
    #[must_use]
    pub fn time_range(&self) -> Option<(u64, u64)> {
        let segments = self.segments.read();

        if let (Some(first), Some(last)) = (segments.front(), segments.back()) {
            Some((first.start_timestamp, last.end_timestamp))
        } else {
            None
        }
    }

    /// Clears the buffer.
    pub fn clear(&self) {
        let mut segments = self.segments.write();
        segments.clear();
        *self.total_size.write() = 0;
        *self.current_segment.write() = None;
        *self.next_segment_number.write() = 0;
    }
}
