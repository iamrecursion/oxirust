//! Multi-track coordination and management.
//!
//! Provides sophisticated track management for complex multi-track scenarios.

#![forbid(unsafe_code)]

use oximedia_core::{CodecId, OxiError, OxiResult};
use std::collections::{HashMap, VecDeque};

use crate::{Packet, StreamInfo};

/// Track synchronization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    /// Synchronize all tracks (wait for all).
    All,
    /// Synchronize video tracks only.
    Video,
    /// No synchronization (best effort).
    None,
}

/// Configuration for track manager.
#[derive(Debug, Clone)]
pub struct TrackManagerConfig {
    /// Synchronization mode.
    pub sync_mode: SyncMode,
    /// Maximum buffer size per track in packets.
    pub max_buffer_size: usize,
    /// Enable automatic interleaving.
    pub auto_interleave: bool,
    /// Target interleave duration in milliseconds.
    pub interleave_duration_ms: u64,
}

impl Default for TrackManagerConfig {
    fn default() -> Self {
        Self {
            sync_mode: SyncMode::All,
            max_buffer_size: 100,
            auto_interleave: true,
            interleave_duration_ms: 500,
        }
    }
}

impl TrackManagerConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sync_mode: SyncMode::All,
            max_buffer_size: 100,
            auto_interleave: true,
            interleave_duration_ms: 500,
        }
    }

    /// Sets the synchronization mode.
    #[must_use]
    pub const fn with_sync_mode(mut self, mode: SyncMode) -> Self {
        self.sync_mode = mode;
        self
    }

    /// Sets the maximum buffer size.
    #[must_use]
    pub const fn with_max_buffer_size(mut self, size: usize) -> Self {
        self.max_buffer_size = size;
        self
    }

    /// Enables automatic interleaving.
    #[must_use]
    pub const fn with_auto_interleave(mut self, enabled: bool) -> Self {
        self.auto_interleave = enabled;
        self
    }

    /// Sets the interleave duration.
    #[must_use]
    pub const fn with_interleave_duration(mut self, duration_ms: u64) -> Self {
        self.interleave_duration_ms = duration_ms;
        self
    }
}

/// Information about a managed track.
#[derive(Debug, Clone)]
pub struct TrackInfo {
    /// Stream information.
    pub stream_info: StreamInfo,
    /// Whether this track is enabled.
    pub enabled: bool,
    /// Track priority (higher = more important).
    pub priority: i32,
    /// Language code (e.g., "eng", "jpn").
    pub language: Option<String>,
    /// Track label/name.
    pub label: Option<String>,
}

impl TrackInfo {
    /// Creates a new track info.
    #[must_use]
    pub const fn new(stream_info: StreamInfo) -> Self {
        Self {
            stream_info,
            enabled: true,
            priority: 0,
            language: None,
            label: None,
        }
    }

    /// Sets the enabled state.
    #[must_use]
    pub const fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Sets the priority.
    #[must_use]
    pub const fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Sets the language.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Sets the label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Statistics for a track.
#[derive(Debug, Clone, Copy, Default)]
pub struct TrackStats {
    /// Number of packets received.
    pub packets_received: u64,
    /// Number of packets dropped.
    pub packets_dropped: u64,
    /// Total bytes received.
    pub bytes_received: u64,
    /// Last timestamp.
    pub last_timestamp: Option<i64>,
}

impl TrackStats {
    /// Creates new statistics.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            packets_received: 0,
            packets_dropped: 0,
            bytes_received: 0,
            last_timestamp: None,
        }
    }

    /// Updates statistics with a new packet.
    pub fn update(&mut self, packet: &Packet) {
        self.packets_received += 1;
        self.bytes_received += packet.size() as u64;
        self.last_timestamp = Some(packet.pts());
    }

    /// Increments the dropped packet count.
    pub fn record_drop(&mut self) {
        self.packets_dropped += 1;
    }
}

/// Manager for coordinating multiple tracks.
pub struct TrackManager {
    config: TrackManagerConfig,
    tracks: Vec<TrackInfo>,
    buffers: HashMap<usize, VecDeque<Packet>>,
    stats: HashMap<usize, TrackStats>,
}

impl TrackManager {
    /// Creates a new track manager with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(TrackManagerConfig::default())
    }

    /// Creates a new track manager with custom configuration.
    #[must_use]
    pub fn with_config(config: TrackManagerConfig) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            buffers: HashMap::new(),
            stats: HashMap::new(),
        }
    }

    /// Adds a track to the manager.
    pub fn add_track(&mut self, info: TrackInfo) -> usize {
        let index = self.tracks.len();
        self.tracks.push(info);
        self.buffers.insert(index, VecDeque::new());
        self.stats.insert(index, TrackStats::new());
        index
    }

    /// Returns information about all tracks.
    #[must_use]
    pub fn tracks(&self) -> &[TrackInfo] {
        &self.tracks
    }

    /// Returns statistics for a track.
    #[must_use]
    pub fn track_stats(&self, index: usize) -> Option<&TrackStats> {
        self.stats.get(&index)
    }

    /// Adds a packet to a track's buffer.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the track index is invalid or the buffer is full.
    pub fn add_packet(&mut self, packet: Packet) -> OxiResult<()> {
        let index = packet.stream_index;

        if index >= self.tracks.len() {
            return Err(OxiError::InvalidData(format!(
                "Invalid track index: {index}"
            )));
        }

        if !self.tracks[index].enabled {
            return Ok(());
        }

        let buffer = self
            .buffers
            .get_mut(&index)
            .ok_or_else(|| OxiError::InvalidData("Track not found".into()))?;

        if buffer.len() >= self.config.max_buffer_size {
            // Drop oldest packet
            buffer.pop_front();
            if let Some(stats) = self.stats.get_mut(&index) {
                stats.record_drop();
            }
        }

        if let Some(stats) = self.stats.get_mut(&index) {
            stats.update(&packet);
        }

        buffer.push_back(packet);

        Ok(())
    }

    /// Gets the next packet to process based on synchronization mode.
    pub fn get_next_packet(&mut self) -> Option<Packet> {
        if !self.config.auto_interleave {
            // Just return first available packet
            return self.get_first_packet();
        }

        match self.config.sync_mode {
            SyncMode::All => self.get_synchronized_packet(),
            SyncMode::Video => self.get_video_synchronized_packet(),
            SyncMode::None => self.get_first_packet(),
        }
    }

    /// Gets the first available packet from any track.
    fn get_first_packet(&mut self) -> Option<Packet> {
        for index in 0..self.tracks.len() {
            if let Some(buffer) = self.buffers.get_mut(&index) {
                if let Some(packet) = buffer.pop_front() {
                    return Some(packet);
                }
            }
        }
        None
    }

    /// Gets a packet synchronized across all tracks.
    fn get_synchronized_packet(&mut self) -> Option<Packet> {
        // Find track with earliest timestamp
        let mut earliest_index = None;
        let mut earliest_pts = i64::MAX;

        for (index, buffer) in &self.buffers {
            if self.tracks[*index].enabled {
                if let Some(packet) = buffer.front() {
                    if packet.pts() < earliest_pts {
                        earliest_pts = packet.pts();
                        earliest_index = Some(*index);
                    }
                }
            }
        }

        if let Some(index) = earliest_index {
            self.buffers.get_mut(&index)?.pop_front()
        } else {
            None
        }
    }

    /// Gets a packet synchronized with video tracks.
    fn get_video_synchronized_packet(&mut self) -> Option<Packet> {
        // Find video track with earliest timestamp
        let mut earliest_index = None;
        let mut earliest_pts = i64::MAX;

        for (index, buffer) in &self.buffers {
            if self.tracks[*index].enabled {
                let is_video = matches!(
                    self.tracks[*index].stream_info.codec,
                    CodecId::Av1 | CodecId::Vp8 | CodecId::Vp9
                );

                if is_video {
                    if let Some(packet) = buffer.front() {
                        if packet.pts() < earliest_pts {
                            earliest_pts = packet.pts();
                            earliest_index = Some(*index);
                        }
                    }
                }
            }
        }

        if let Some(index) = earliest_index {
            self.buffers.get_mut(&index)?.pop_front()
        } else {
            // No video packets, get any packet
            self.get_first_packet()
        }
    }

    /// Returns the number of buffered packets for a track.
    #[must_use]
    pub fn buffered_count(&self, index: usize) -> usize {
        self.buffers.get(&index).map_or(0, VecDeque::len)
    }

    /// Returns the total number of buffered packets across all tracks.
    #[must_use]
    pub fn total_buffered(&self) -> usize {
        self.buffers.values().map(VecDeque::len).sum()
    }

    /// Clears all buffers.
    pub fn clear_buffers(&mut self) {
        for buffer in self.buffers.values_mut() {
            buffer.clear();
        }
    }

    /// Enables or disables a track.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the track index is out of bounds.
    pub fn set_track_enabled(&mut self, index: usize, enabled: bool) -> OxiResult<()> {
        self.tracks
            .get_mut(index)
            .ok_or_else(|| OxiError::InvalidData("Track not found".into()))?
            .enabled = enabled;
        Ok(())
    }
}

impl Default for TrackManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper for calculating optimal interleaving.
pub struct InterleavingCalculator {
    #[allow(dead_code)]
    target_duration_ms: u64,
}

impl InterleavingCalculator {
    /// Creates a new interleaving calculator.
    #[must_use]
    pub const fn new(target_duration_ms: u64) -> Self {
        Self { target_duration_ms }
    }

    /// Calculates the interleaving order for packets.
    ///
    /// Returns a list of stream indices in the order they should be written.
    #[must_use]
    pub fn calculate_order(&self, packets: &[&Packet]) -> Vec<usize> {
        let mut packets_with_index: Vec<(usize, &Packet)> =
            packets.iter().enumerate().map(|(i, p)| (i, *p)).collect();

        // Sort by timestamp
        packets_with_index.sort_by_key(|(_, p)| p.pts());

        packets_with_index.into_iter().map(|(i, _)| i).collect()
    }

    /// Checks if packets need reordering for optimal interleaving.
    #[must_use]
    pub fn needs_reordering(&self, packets: &[&Packet]) -> bool {
        if packets.len() < 2 {
            return false;
        }

        // Check if packets are already in timestamp order
        for i in 1..packets.len() {
            if packets[i].pts() < packets[i - 1].pts() {
                return true;
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use oximedia_core::{Rational, Timestamp};

    fn create_test_stream(index: usize, codec: CodecId) -> StreamInfo {
        let mut stream = StreamInfo::new(index, codec, Rational::new(1, 48000));
        stream.codec_params = crate::stream::CodecParams::audio(48000, 2);
        stream
    }

    fn create_test_packet(stream_index: usize, pts: i64) -> Packet {
        Packet::new(
            stream_index,
            Bytes::new(),
            Timestamp::new(pts, Rational::new(1, 1000)),
            crate::PacketFlags::empty(),
        )
    }

    #[test]
    fn test_track_manager_config() {
        let config = TrackManagerConfig::new()
            .with_sync_mode(SyncMode::Video)
            .with_max_buffer_size(50)
            .with_auto_interleave(false)
            .with_interleave_duration(1000);

        assert_eq!(config.sync_mode, SyncMode::Video);
        assert_eq!(config.max_buffer_size, 50);
        assert!(!config.auto_interleave);
        assert_eq!(config.interleave_duration_ms, 1000);
    }

    #[test]
    fn test_track_info() {
        let stream = create_test_stream(0, CodecId::Opus);
        let track = TrackInfo::new(stream)
            .with_enabled(true)
            .with_priority(10)
            .with_language("eng")
            .with_label("English");

        assert!(track.enabled);
        assert_eq!(track.priority, 10);
        assert_eq!(track.language, Some("eng".into()));
        assert_eq!(track.label, Some("English".into()));
    }

    #[test]
    fn test_track_stats() {
        let mut stats = TrackStats::new();
        assert_eq!(stats.packets_received, 0);

        let packet = create_test_packet(0, 1000);
        stats.update(&packet);

        assert_eq!(stats.packets_received, 1);
        assert_eq!(stats.last_timestamp, Some(1000));

        stats.record_drop();
        assert_eq!(stats.packets_dropped, 1);
    }

    #[test]
    fn test_track_manager() {
        let mut manager = TrackManager::new();

        let stream = create_test_stream(0, CodecId::Opus);
        let track = TrackInfo::new(stream);
        let index = manager.add_track(track);

        assert_eq!(index, 0);
        assert_eq!(manager.tracks().len(), 1);

        let packet = create_test_packet(0, 1000);
        assert!(manager.add_packet(packet).is_ok());

        assert_eq!(manager.buffered_count(0), 1);
        assert_eq!(manager.total_buffered(), 1);
    }

    #[test]
    fn test_interleaving_calculator() {
        let calc = InterleavingCalculator::new(500);

        let packets = vec![
            create_test_packet(0, 1000),
            create_test_packet(1, 500),
            create_test_packet(0, 1500),
        ];

        let packet_refs: Vec<&Packet> = packets.iter().collect();

        assert!(calc.needs_reordering(&packet_refs));

        let order = calc.calculate_order(&packet_refs);
        assert_eq!(order, vec![1, 0, 2]); // Sorted by timestamp
    }
}
