//! Chunked transfer encoding for Low-Latency DASH (LL-DASH).
//!
//! This module implements chunked transfer of DASH segments, enabling
//! ultra-low latency streaming by transmitting segments in smaller chunks
//! as they are generated, rather than waiting for complete segments.

#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]

use bytes::{Bytes, BytesMut};
use std::collections::VecDeque;
use std::time::{Duration, SystemTime};

/// Default chunk size for low-latency streaming (in bytes).
const DEFAULT_CHUNK_SIZE: usize = 16384; // 16KB

/// Minimum chunk size.
const MIN_CHUNK_SIZE: usize = 4096; // 4KB

/// Chunked segment transfer manager.
///
/// This structure manages the chunking of segments for low-latency delivery,
/// allowing clients to start playback before the entire segment is available.
#[derive(Debug)]
pub struct ChunkedTransfer {
    /// Target chunk size in bytes.
    chunk_size: usize,
    /// Chunk buffer for the current segment.
    current_chunks: VecDeque<Chunk>,
    /// Current segment number.
    current_segment_number: u64,
    /// Buffer for accumulating data before chunking.
    accumulator: BytesMut,
    /// Chunk sequence number within segment.
    chunk_sequence: u32,
    /// Enable chunked transfer.
    enabled: bool,
}

/// A chunk of segment data.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Segment number this chunk belongs to.
    pub segment_number: u64,
    /// Chunk sequence number within segment.
    pub sequence: u32,
    /// Chunk data.
    pub data: Bytes,
    /// Is this the last chunk of the segment.
    pub is_last: bool,
    /// Timestamp when chunk was created.
    pub created_at: SystemTime,
    /// Byte offset within segment.
    pub byte_offset: usize,
}

/// Chunked transfer configuration.
#[derive(Debug, Clone)]
pub struct ChunkedConfig {
    /// Chunk size in bytes.
    pub chunk_size: usize,
    /// Enable chunked transfer.
    pub enabled: bool,
    /// Maximum chunks to buffer per segment.
    pub max_chunks_per_segment: usize,
}

impl Default for ChunkedConfig {
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
            enabled: true,
            max_chunks_per_segment: 64,
        }
    }
}

impl ChunkedTransfer {
    /// Creates a new chunked transfer manager.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(ChunkedConfig::default())
    }

    /// Creates a chunked transfer manager with custom configuration.
    #[must_use]
    pub fn with_config(config: ChunkedConfig) -> Self {
        let chunk_size = config.chunk_size.max(MIN_CHUNK_SIZE);

        Self {
            chunk_size,
            current_chunks: VecDeque::new(),
            current_segment_number: 0,
            accumulator: BytesMut::new(),
            chunk_sequence: 0,
            enabled: config.enabled,
        }
    }

    /// Starts a new segment.
    ///
    /// # Arguments
    ///
    /// * `segment_number` - The segment number
    pub fn start_segment(&mut self, segment_number: u64) {
        self.current_segment_number = segment_number;
        self.chunk_sequence = 0;
        self.current_chunks.clear();
        self.accumulator.clear();
    }

    /// Adds data to the current segment.
    ///
    /// This will chunk the data if it exceeds the chunk size.
    ///
    /// # Arguments
    ///
    /// * `data` - Data to add
    ///
    /// # Returns
    ///
    /// Vector of completed chunks
    pub fn add_data(&mut self, data: &[u8]) -> Vec<Chunk> {
        if !self.enabled {
            return Vec::new();
        }

        self.accumulator.extend_from_slice(data);

        let mut chunks = Vec::new();

        while self.accumulator.len() >= self.chunk_size {
            let chunk_data = self.accumulator.split_to(self.chunk_size).freeze();
            let chunk = self.create_chunk(chunk_data, false);
            chunks.push(chunk.clone());
            self.current_chunks.push_back(chunk);
        }

        chunks
    }

    /// Finalizes the current segment.
    ///
    /// This creates a final chunk with any remaining data.
    ///
    /// # Returns
    ///
    /// The final chunk, if any data remains
    pub fn finalize_segment(&mut self) -> Option<Chunk> {
        if !self.enabled || self.accumulator.is_empty() {
            return None;
        }

        let chunk_data = self.accumulator.split().freeze();
        let chunk = self.create_chunk(chunk_data, true);
        self.current_chunks.push_back(chunk.clone());

        Some(chunk)
    }

    /// Returns all chunks for the current segment.
    pub fn chunks(&self) -> Vec<&Chunk> {
        self.current_chunks.iter().collect()
    }

    /// Returns a specific chunk by sequence number.
    #[must_use]
    pub fn get_chunk(&self, sequence: u32) -> Option<&Chunk> {
        self.current_chunks.iter().find(|c| c.sequence == sequence)
    }

    /// Returns the number of chunks in the current segment.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.current_chunks.len()
    }

    /// Checks if chunked transfer is enabled.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Sets whether chunked transfer is enabled.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns the chunk size.
    #[must_use]
    pub const fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Sets the chunk size.
    pub fn set_chunk_size(&mut self, size: usize) {
        self.chunk_size = size.max(MIN_CHUNK_SIZE);
    }

    /// Clears all chunks.
    pub fn clear(&mut self) {
        self.current_chunks.clear();
        self.accumulator.clear();
        self.chunk_sequence = 0;
    }

    /// Creates a chunk from data.
    fn create_chunk(&mut self, data: Bytes, is_last: bool) -> Chunk {
        let byte_offset = (self.chunk_sequence as usize) * self.chunk_size;
        let chunk = Chunk {
            segment_number: self.current_segment_number,
            sequence: self.chunk_sequence,
            data,
            is_last,
            created_at: SystemTime::now(),
            byte_offset,
        };

        self.chunk_sequence += 1;
        chunk
    }
}

impl Default for ChunkedTransfer {
    fn default() -> Self {
        Self::new()
    }
}

/// Producer reference time for LL-DASH.
///
/// This structure represents the mapping between presentation time
/// and wall-clock time, enabling clients to synchronize playback.
#[derive(Debug, Clone)]
pub struct ProducerReferenceTime {
    /// Presentation time in timescale units.
    pub presentation_time: u64,
    /// Wall clock time.
    pub wall_clock_time: SystemTime,
    /// Timescale.
    pub timescale: u32,
}

impl ProducerReferenceTime {
    /// Creates a new producer reference time.
    #[must_use]
    pub fn new(presentation_time: u64, wall_clock_time: SystemTime, timescale: u32) -> Self {
        Self {
            presentation_time,
            wall_clock_time,
            timescale,
        }
    }

    /// Creates a reference time for the current moment.
    #[must_use]
    pub fn now(presentation_time: u64, timescale: u32) -> Self {
        Self::new(presentation_time, SystemTime::now(), timescale)
    }

    /// Returns the presentation time in seconds.
    #[must_use]
    pub fn presentation_time_secs(&self) -> f64 {
        self.presentation_time as f64 / self.timescale as f64
    }

    /// Formats the wall clock time as ISO 8601.
    #[must_use]
    pub fn wall_clock_iso8601(&self) -> String {
        super::timeline::TimelineManager::format_system_time(self.wall_clock_time)
    }
}

/// Chunked segment metadata for LL-DASH manifests.
#[derive(Debug, Clone)]
pub struct ChunkedSegmentMetadata {
    /// Segment number.
    pub segment_number: u64,
    /// Total number of chunks.
    pub chunk_count: u32,
    /// Chunk size in bytes.
    pub chunk_size: usize,
    /// Total segment size.
    pub total_size: usize,
    /// Chunk availability times.
    pub chunk_times: Vec<SystemTime>,
}

impl ChunkedSegmentMetadata {
    /// Creates new metadata.
    #[must_use]
    pub fn new(segment_number: u64, chunk_size: usize) -> Self {
        Self {
            segment_number,
            chunk_count: 0,
            chunk_size,
            total_size: 0,
            chunk_times: Vec::new(),
        }
    }

    /// Adds a chunk to the metadata.
    pub fn add_chunk(&mut self, chunk: &Chunk) {
        self.chunk_count += 1;
        self.total_size += chunk.data.len();
        self.chunk_times.push(chunk.created_at);
    }

    /// Returns the average chunk creation interval.
    #[must_use]
    pub fn average_chunk_interval(&self) -> Option<Duration> {
        if self.chunk_times.len() < 2 {
            return None;
        }

        let mut total_duration = Duration::ZERO;
        for i in 1..self.chunk_times.len() {
            if let Ok(diff) = self.chunk_times[i].duration_since(self.chunk_times[i - 1]) {
                total_duration += diff;
            }
        }

        let count = (self.chunk_times.len() - 1) as u32;
        Some(total_duration / count)
    }
}

/// Chunk delivery coordinator for multiple representations.
#[derive(Debug)]
pub struct ChunkCoordinator {
    /// Chunked transfers by representation ID.
    transfers: std::collections::HashMap<String, ChunkedTransfer>,
    /// Producer reference times.
    reference_times: Vec<ProducerReferenceTime>,
    /// Suggested presentation delay.
    suggested_presentation_delay: Duration,
}

impl ChunkCoordinator {
    /// Creates a new chunk coordinator.
    #[must_use]
    pub fn new(suggested_presentation_delay: Duration) -> Self {
        Self {
            transfers: std::collections::HashMap::new(),
            reference_times: Vec::new(),
            suggested_presentation_delay,
        }
    }

    /// Adds a representation.
    pub fn add_representation(&mut self, representation_id: String, config: ChunkedConfig) {
        let transfer = ChunkedTransfer::with_config(config);
        self.transfers.insert(representation_id, transfer);
    }

    /// Starts a segment for a representation.
    pub fn start_segment(&mut self, representation_id: &str, segment_number: u64) {
        if let Some(transfer) = self.transfers.get_mut(representation_id) {
            transfer.start_segment(segment_number);
        }
    }

    /// Adds data to a representation.
    pub fn add_data(&mut self, representation_id: &str, data: &[u8]) -> Vec<Chunk> {
        if let Some(transfer) = self.transfers.get_mut(representation_id) {
            transfer.add_data(data)
        } else {
            Vec::new()
        }
    }

    /// Finalizes a segment.
    pub fn finalize_segment(&mut self, representation_id: &str) -> Option<Chunk> {
        if let Some(transfer) = self.transfers.get_mut(representation_id) {
            transfer.finalize_segment()
        } else {
            None
        }
    }

    /// Adds a producer reference time point.
    pub fn add_reference_time(&mut self, reference: ProducerReferenceTime) {
        self.reference_times.push(reference);

        // Keep only recent reference times (last 10)
        if self.reference_times.len() > 10 {
            self.reference_times
                .drain(0..self.reference_times.len() - 10);
        }
    }

    /// Returns the latest producer reference time.
    #[must_use]
    pub fn latest_reference_time(&self) -> Option<&ProducerReferenceTime> {
        self.reference_times.last()
    }

    /// Returns the suggested presentation delay.
    #[must_use]
    pub const fn suggested_presentation_delay(&self) -> Duration {
        self.suggested_presentation_delay
    }

    /// Sets the suggested presentation delay.
    pub fn set_suggested_presentation_delay(&mut self, delay: Duration) {
        self.suggested_presentation_delay = delay;
    }

    /// Returns a reference to a chunked transfer.
    #[must_use]
    pub fn transfer(&self, representation_id: &str) -> Option<&ChunkedTransfer> {
        self.transfers.get(representation_id)
    }
}

impl Chunk {
    /// Returns the chunk size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns the byte range for this chunk.
    #[must_use]
    pub fn byte_range(&self) -> (usize, usize) {
        let start = self.byte_offset;
        let end = start + self.data.len() - 1;
        (start, end)
    }

    /// Formats the byte range as an HTTP Range header value.
    #[must_use]
    pub fn range_header(&self) -> String {
        let (start, end) = self.byte_range();
        format!("bytes={start}-{end}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunked_transfer_creation() {
        let transfer = ChunkedTransfer::new();
        assert!(transfer.is_enabled());
        assert_eq!(transfer.chunk_size(), DEFAULT_CHUNK_SIZE);
        assert_eq!(transfer.chunk_count(), 0);
    }

    #[test]
    fn test_chunked_transfer_with_config() {
        let config = ChunkedConfig {
            chunk_size: 8192,
            enabled: true,
            max_chunks_per_segment: 32,
        };

        let transfer = ChunkedTransfer::with_config(config);
        assert_eq!(transfer.chunk_size(), 8192);
    }

    #[test]
    fn test_add_data_creates_chunks() {
        let mut transfer = ChunkedTransfer::new();
        transfer.start_segment(1);

        // Add data larger than chunk size
        let data = vec![0u8; DEFAULT_CHUNK_SIZE * 2 + 1024];
        let chunks = transfer.add_data(&data);

        // Should create 2 chunks, with remainder in accumulator
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].sequence, 0);
        assert_eq!(chunks[1].sequence, 1);
    }

    #[test]
    fn test_finalize_segment() {
        let mut transfer = ChunkedTransfer::new();
        transfer.start_segment(1);

        let data = vec![0u8; 1024];
        transfer.add_data(&data);

        let final_chunk = transfer.finalize_segment();
        assert!(final_chunk.is_some());

        let chunk = final_chunk.expect("should succeed in test");
        assert!(chunk.is_last);
        assert_eq!(chunk.size(), 1024);
    }

    #[test]
    fn test_chunk_byte_range() {
        let chunk = Chunk {
            segment_number: 1,
            sequence: 0,
            data: Bytes::from(vec![0u8; 100]),
            is_last: false,
            created_at: SystemTime::now(),
            byte_offset: 0,
        };

        let (start, end) = chunk.byte_range();
        assert_eq!(start, 0);
        assert_eq!(end, 99);

        let range = chunk.range_header();
        assert_eq!(range, "bytes=0-99");
    }

    #[test]
    fn test_producer_reference_time() {
        let prt = ProducerReferenceTime::now(90000, 90000);
        assert_eq!(prt.presentation_time_secs(), 1.0);

        let iso = prt.wall_clock_iso8601();
        assert!(!iso.is_empty());
    }

    #[test]
    fn test_chunk_coordinator() {
        let mut coordinator = ChunkCoordinator::new(Duration::from_secs(2));

        coordinator.add_representation("720p".to_string(), ChunkedConfig::default());
        coordinator.start_segment("720p", 1);

        let data = vec![0u8; DEFAULT_CHUNK_SIZE + 100];
        let chunks = coordinator.add_data("720p", &data);

        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_chunked_segment_metadata() {
        let mut metadata = ChunkedSegmentMetadata::new(1, 16384);

        let chunk = Chunk {
            segment_number: 1,
            sequence: 0,
            data: Bytes::from(vec![0u8; 16384]),
            is_last: false,
            created_at: SystemTime::now(),
            byte_offset: 0,
        };

        metadata.add_chunk(&chunk);
        assert_eq!(metadata.chunk_count, 1);
        assert_eq!(metadata.total_size, 16384);
    }

    #[test]
    fn test_clear() {
        let mut transfer = ChunkedTransfer::new();
        transfer.start_segment(1);
        transfer.add_data(&vec![0u8; 1024]);

        transfer.clear();
        assert_eq!(transfer.chunk_count(), 0);
    }
}
