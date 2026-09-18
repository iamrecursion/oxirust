//! Streaming muxer for live output.
//!
//! Provides progressive muxing without pre-buffering,
//! optimized for live streaming scenarios. Includes CMAF
//! chunked transfer encoding for low-latency delivery.

#![forbid(unsafe_code)]

#[cfg(not(target_arch = "wasm32"))]
use async_trait::async_trait;
#[cfg(not(target_arch = "wasm32"))]
use oximedia_core::OxiResult;
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::mpsc;
#[cfg(not(target_arch = "wasm32"))]
use tokio::time::Instant;

#[cfg(not(target_arch = "wasm32"))]
use crate::{Muxer, MuxerConfig, Packet, StreamInfo};

/// Configuration for streaming muxer.
#[derive(Clone, Debug)]
pub struct StreamingMuxerConfig {
    /// Target latency in milliseconds.
    pub target_latency_ms: u64,
    /// Enable low-latency mode (no buffering).
    pub low_latency: bool,
    /// Fragment duration in milliseconds (for fragmented formats).
    pub fragment_duration_ms: Option<u64>,
    /// Enable real-time mode (enforce timing).
    pub realtime: bool,
}

impl Default for StreamingMuxerConfig {
    fn default() -> Self {
        Self {
            target_latency_ms: 1000,
            low_latency: false,
            fragment_duration_ms: None,
            realtime: false,
        }
    }
}

impl StreamingMuxerConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            target_latency_ms: 1000,
            low_latency: false,
            fragment_duration_ms: None,
            realtime: false,
        }
    }

    /// Enables low-latency mode.
    #[must_use]
    pub const fn with_low_latency(mut self, enabled: bool) -> Self {
        self.low_latency = enabled;
        self
    }

    /// Sets the target latency.
    #[must_use]
    pub const fn with_target_latency(mut self, latency_ms: u64) -> Self {
        self.target_latency_ms = latency_ms;
        self
    }

    /// Sets the fragment duration.
    #[must_use]
    pub const fn with_fragment_duration(mut self, duration_ms: u64) -> Self {
        self.fragment_duration_ms = Some(duration_ms);
        self
    }

    /// Enables real-time mode.
    #[must_use]
    pub const fn with_realtime(mut self, enabled: bool) -> Self {
        self.realtime = enabled;
        self
    }
}

/// Wrapper that adds streaming capabilities to any muxer.
#[cfg(not(target_arch = "wasm32"))]
pub struct StreamingMuxer<M: Muxer> {
    inner: M,
    #[allow(dead_code)]
    streaming_config: StreamingMuxerConfig,
    packets_written: u64,
    bytes_written: u64,
    start_time: Option<Instant>,
    last_packet_time: Option<Instant>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<M: Muxer> StreamingMuxer<M> {
    /// Creates a new streaming muxer with default configuration.
    pub const fn new(inner: M) -> Self {
        Self::with_config(inner, StreamingMuxerConfig::new())
    }

    /// Creates a new streaming muxer with custom configuration.
    pub const fn with_config(inner: M, streaming_config: StreamingMuxerConfig) -> Self {
        Self {
            inner,
            streaming_config,
            packets_written: 0,
            bytes_written: 0,
            start_time: None,
            last_packet_time: None,
        }
    }

    /// Returns the number of packets written.
    #[must_use]
    pub const fn packets_written(&self) -> u64 {
        self.packets_written
    }

    /// Returns the number of bytes written.
    #[must_use]
    pub const fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Returns the elapsed time since muxing started.
    #[must_use]
    pub fn elapsed(&self) -> Option<Duration> {
        self.start_time.map(|start| start.elapsed())
    }

    /// Returns a reference to the inner muxer.
    #[must_use]
    pub const fn inner(&self) -> &M {
        &self.inner
    }

    /// Returns a mutable reference to the inner muxer.
    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.inner
    }

    /// Unwraps and returns the inner muxer.
    #[must_use]
    pub fn into_inner(self) -> M {
        self.inner
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
impl<M: Muxer> Muxer for StreamingMuxer<M> {
    fn add_stream(&mut self, info: StreamInfo) -> OxiResult<usize> {
        self.inner.add_stream(info)
    }

    async fn write_header(&mut self) -> OxiResult<()> {
        self.start_time = Some(Instant::now());
        self.inner.write_header().await
    }

    async fn write_packet(&mut self, packet: &Packet) -> OxiResult<()> {
        let now = Instant::now();
        self.last_packet_time = Some(now);
        self.packets_written += 1;
        self.bytes_written += packet.size() as u64;
        self.inner.write_packet(packet).await
    }

    async fn write_trailer(&mut self) -> OxiResult<()> {
        self.inner.write_trailer().await
    }

    fn streams(&self) -> &[StreamInfo] {
        self.inner.streams()
    }

    fn config(&self) -> &MuxerConfig {
        self.inner.config()
    }
}

/// Packet sender for background muxing.
#[cfg(not(target_arch = "wasm32"))]
pub struct PacketSender {
    tx: mpsc::UnboundedSender<Packet>,
}

#[cfg(not(target_arch = "wasm32"))]
impl PacketSender {
    /// Creates a new packet sender.
    const fn new(tx: mpsc::UnboundedSender<Packet>) -> Self {
        Self { tx }
    }

    /// Sends a packet to the background muxer.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the background muxer task has terminated.
    pub fn send(&self, packet: Packet) -> Result<(), mpsc::error::SendError<Packet>> {
        self.tx.send(packet)
    }

    /// Tries to send a packet without blocking.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the background muxer task has terminated.
    pub fn try_send(&self, packet: Packet) -> Result<(), mpsc::error::SendError<Packet>> {
        self.tx.send(packet)
    }
}

/// Spawns a background task for muxing.
///
/// This function creates a background task that continuously receives packets
/// from a channel and writes them to the muxer. This is useful for streaming
/// scenarios where you want to decouple packet production from muxing.
///
/// # Arguments
///
/// * `muxer` - The muxer to run in the background
///
/// # Returns
///
/// A `PacketSender` that can be used to send packets to the background task.
///
/// # Errors
///
/// Returns `Err` if writing the container header fails.
///
/// # Examples
///
/// ```ignore
/// let muxer = MatroskaMuxer::new(sink, config);
/// let sender = spawn_muxer(muxer).await?;
///
/// for packet in packets {
///     sender.send(packet)?;
/// }
/// ```
#[cfg(not(target_arch = "wasm32"))]
pub async fn spawn_muxer<M: Muxer + Send + 'static>(mut muxer: M) -> OxiResult<PacketSender> {
    // Write header first
    muxer.write_header().await?;

    let (tx, mut rx) = mpsc::unbounded_channel();

    tokio::spawn(async move {
        while let Some(packet) = rx.recv().await {
            if muxer.write_packet(&packet).await.is_err() {
                break;
            }
        }
        let _ = muxer.write_trailer().await;
    });

    Ok(PacketSender::new(tx))
}

/// Statistics for streaming muxing.
#[derive(Debug, Clone, Copy, Default)]
pub struct MuxingStats {
    /// Total packets written.
    pub packets_written: u64,
    /// Total bytes written.
    pub bytes_written: u64,
    /// Average bitrate in bits per second.
    pub avg_bitrate: f64,
    /// Current bitrate in bits per second.
    pub current_bitrate: f64,
    /// Total duration in seconds.
    pub duration_secs: f64,
}

impl MuxingStats {
    /// Creates new statistics.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            packets_written: 0,
            bytes_written: 0,
            avg_bitrate: 0.0,
            current_bitrate: 0.0,
            duration_secs: 0.0,
        }
    }

    /// Updates statistics with a new packet.
    pub fn update(&mut self, packet_size: usize, duration_secs: f64) {
        self.packets_written += 1;
        self.bytes_written += packet_size as u64;
        self.duration_secs = duration_secs;

        if duration_secs > 0.0 {
            #[allow(clippy::cast_precision_loss)]
            {
                self.avg_bitrate = (self.bytes_written as f64 * 8.0) / duration_secs;
            }
        }
    }

    /// Sets the current bitrate.
    pub fn set_current_bitrate(&mut self, bitrate: f64) {
        self.current_bitrate = bitrate;
    }
}

/// Latency monitor for streaming.
#[derive(Debug)]
pub struct LatencyMonitor {
    target_latency: Duration,
    measurements: Vec<Duration>,
    max_measurements: usize,
}

impl LatencyMonitor {
    /// Creates a new latency monitor.
    #[must_use]
    pub fn new(target_latency: Duration) -> Self {
        Self {
            target_latency,
            measurements: Vec::with_capacity(100),
            max_measurements: 100,
        }
    }

    /// Records a latency measurement.
    pub fn record(&mut self, latency: Duration) {
        if self.measurements.len() >= self.max_measurements {
            self.measurements.remove(0);
        }
        self.measurements.push(latency);
    }

    /// Returns the average latency.
    #[must_use]
    pub fn average_latency(&self) -> Option<Duration> {
        if self.measurements.is_empty() {
            return None;
        }

        let sum: Duration = self.measurements.iter().sum();
        #[allow(clippy::cast_possible_truncation)]
        let count = self.measurements.len() as u32;
        Some(sum / count)
    }

    /// Returns true if latency is within target.
    #[must_use]
    pub fn is_within_target(&self) -> bool {
        self.average_latency()
            .map_or(true, |avg| avg <= self.target_latency)
    }

    /// Returns the target latency.
    #[must_use]
    pub const fn target_latency(&self) -> Duration {
        self.target_latency
    }
}

// ─── CMAF Chunked Transfer ─────────────────────────────────────────────────

/// CMAF chunked transfer encoding mode for low-latency streaming.
///
/// Implements chunked transfer as defined in ISO/IEC 23000-19 CMAF,
/// enabling sub-segment delivery where each chunk contains one or more
/// complete samples and can be delivered independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmafChunkMode {
    /// Standard CMAF segments (no chunking).
    Standard,
    /// Chunked transfer: each chunk is a partial segment containing
    /// one or more complete moof+mdat pairs.
    Chunked,
    /// Low-latency chunked transfer: each chunk is exactly one sample
    /// (frame) for minimum latency.
    LowLatencyChunked,
}

impl Default for CmafChunkMode {
    fn default() -> Self {
        Self::Standard
    }
}

/// Configuration for CMAF chunked transfer.
#[derive(Debug, Clone)]
pub struct CmafChunkedConfig {
    /// Chunk delivery mode.
    pub mode: CmafChunkMode,
    /// Target chunk duration in milliseconds.
    /// Only relevant for `Chunked` mode.
    pub chunk_duration_ms: u32,
    /// Maximum number of samples per chunk.
    /// Only relevant for `Chunked` mode.
    pub max_samples_per_chunk: u32,
    /// Whether to include `mfra` (Movie Fragment Random Access) box.
    pub include_mfra: bool,
    /// Whether to signal low-latency in the `ftyp` compatible brands.
    pub signal_low_latency: bool,
    /// Part target duration for LL-HLS (in milliseconds).
    pub part_target_duration_ms: Option<u32>,
}

impl Default for CmafChunkedConfig {
    fn default() -> Self {
        Self {
            mode: CmafChunkMode::Standard,
            chunk_duration_ms: 500,
            max_samples_per_chunk: 5,
            include_mfra: false,
            signal_low_latency: false,
            part_target_duration_ms: None,
        }
    }
}

impl CmafChunkedConfig {
    /// Creates a new default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the chunk mode.
    #[must_use]
    pub const fn with_mode(mut self, mode: CmafChunkMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets the target chunk duration in milliseconds.
    #[must_use]
    pub const fn with_chunk_duration_ms(mut self, ms: u32) -> Self {
        self.chunk_duration_ms = ms;
        self
    }

    /// Sets the maximum samples per chunk.
    #[must_use]
    pub const fn with_max_samples_per_chunk(mut self, max: u32) -> Self {
        self.max_samples_per_chunk = max;
        self
    }

    /// Enables MFRA box.
    #[must_use]
    pub const fn with_mfra(mut self, include: bool) -> Self {
        self.include_mfra = include;
        self
    }

    /// Enables low-latency signalling.
    #[must_use]
    pub const fn with_low_latency_signal(mut self, signal: bool) -> Self {
        self.signal_low_latency = signal;
        self
    }

    /// Sets the part target duration for LL-HLS.
    #[must_use]
    pub const fn with_part_target_duration_ms(mut self, ms: u32) -> Self {
        self.part_target_duration_ms = Some(ms);
        self
    }
}

/// A single CMAF chunk ready for delivery.
#[derive(Debug, Clone)]
pub struct CmafChunk {
    /// Chunk sequence number (monotonically increasing).
    pub sequence_number: u32,
    /// Segment sequence this chunk belongs to.
    pub segment_sequence: u32,
    /// Chunk index within the segment (0-based).
    pub chunk_index: u32,
    /// Start PTS in timescale ticks.
    pub start_pts: u64,
    /// Duration in timescale ticks.
    pub duration: u64,
    /// Number of samples in this chunk.
    pub sample_count: u32,
    /// Total data size in bytes.
    pub data_size: usize,
    /// Whether this chunk starts with a keyframe.
    pub starts_with_keyframe: bool,
    /// Whether this is the last chunk in the segment.
    pub is_last_in_segment: bool,
    /// Raw chunk bytes (moof+mdat).
    pub data: Vec<u8>,
    /// Whether this chunk is independently decodable.
    pub is_independent: bool,
}

/// A pending sample for chunk assembly.
#[derive(Debug, Clone)]
pub struct ChunkSample {
    /// Presentation timestamp in timescale ticks.
    pub pts: u64,
    /// Decode timestamp in timescale ticks.
    pub dts: u64,
    /// Duration in timescale ticks.
    pub duration: u32,
    /// Sample data.
    pub data: Vec<u8>,
    /// Whether this is a keyframe.
    pub is_keyframe: bool,
    /// Track ID.
    pub track_id: u32,
}

/// CMAF chunked transfer encoder.
///
/// Receives samples and produces CMAF chunks according to the configured
/// chunking strategy. Designed for low-latency live streaming.
///
/// # Example
///
/// ```ignore
/// let config = CmafChunkedConfig::new()
///     .with_mode(CmafChunkMode::LowLatencyChunked);
/// let mut encoder = CmafChunkedEncoder::new(config, 90000);
///
/// // Feed samples
/// encoder.push_sample(sample);
///
/// // Collect chunks
/// while let Some(chunk) = encoder.pop_chunk() {
///     deliver(chunk.data);
/// }
/// ```
#[derive(Debug)]
pub struct CmafChunkedEncoder {
    config: CmafChunkedConfig,
    timescale: u32,
    /// Pending samples for the current chunk.
    pending_samples: Vec<ChunkSample>,
    /// Accumulated duration of pending samples (in timescale ticks).
    pending_duration: u64,
    /// Completed chunks waiting to be consumed.
    completed_chunks: Vec<CmafChunk>,
    /// Global chunk sequence counter.
    chunk_sequence: u32,
    /// Current segment sequence number.
    segment_sequence: u32,
    /// Chunk index within current segment.
    chunk_index_in_segment: u32,
    /// Total bytes produced.
    total_bytes_produced: u64,
    /// Total chunks produced.
    total_chunks_produced: u64,
}

impl CmafChunkedEncoder {
    /// Creates a new chunked encoder.
    #[must_use]
    pub fn new(config: CmafChunkedConfig, timescale: u32) -> Self {
        Self {
            config,
            timescale,
            pending_samples: Vec::new(),
            pending_duration: 0,
            completed_chunks: Vec::new(),
            chunk_sequence: 1,
            segment_sequence: 1,
            chunk_index_in_segment: 0,
            total_bytes_produced: 0,
            total_chunks_produced: 0,
        }
    }

    /// Returns the timescale.
    #[must_use]
    pub const fn timescale(&self) -> u32 {
        self.timescale
    }

    /// Returns the total number of chunks produced.
    #[must_use]
    pub const fn total_chunks_produced(&self) -> u64 {
        self.total_chunks_produced
    }

    /// Returns the total bytes produced.
    #[must_use]
    pub const fn total_bytes_produced(&self) -> u64 {
        self.total_bytes_produced
    }

    /// Returns the current segment sequence number.
    #[must_use]
    pub const fn current_segment_sequence(&self) -> u32 {
        self.segment_sequence
    }

    /// Returns the number of pending samples.
    #[must_use]
    pub fn pending_sample_count(&self) -> usize {
        self.pending_samples.len()
    }

    /// Returns the number of completed chunks waiting to be consumed.
    #[must_use]
    pub fn available_chunks(&self) -> usize {
        self.completed_chunks.len()
    }

    /// Pushes a sample into the encoder.
    ///
    /// If the sample triggers a chunk boundary (based on the configured
    /// chunk mode), one or more chunks will be produced.
    pub fn push_sample(&mut self, sample: ChunkSample) {
        let duration = u64::from(sample.duration);
        self.pending_samples.push(sample);
        self.pending_duration += duration;

        match self.config.mode {
            CmafChunkMode::Standard => {
                // No chunking; caller manually flushes segments
            }
            CmafChunkMode::Chunked => {
                self.try_emit_chunk_by_duration();
            }
            CmafChunkMode::LowLatencyChunked => {
                // One sample per chunk for minimum latency
                self.emit_current_chunk(false);
            }
        }
    }

    /// Tries to emit a chunk if the accumulated duration exceeds the target.
    fn try_emit_chunk_by_duration(&mut self) {
        let target_ticks = if self.timescale == 0 {
            0
        } else {
            u64::from(self.config.chunk_duration_ms) * u64::from(self.timescale) / 1000
        };

        if self.pending_duration >= target_ticks
            || self.pending_samples.len() >= self.config.max_samples_per_chunk as usize
        {
            self.emit_current_chunk(false);
        }
    }

    /// Emits the current pending samples as a chunk.
    fn emit_current_chunk(&mut self, is_last: bool) {
        if self.pending_samples.is_empty() {
            return;
        }

        let start_pts = self.pending_samples.first().map_or(0, |s| s.dts);

        let starts_with_keyframe = self
            .pending_samples
            .first()
            .map_or(false, |s| s.is_keyframe);

        let sample_count = self.pending_samples.len() as u32;
        let duration = self.pending_duration;

        // Build moof+mdat for this chunk
        let chunk_data = self.build_chunk_boxes();
        let data_size = chunk_data.len();

        let chunk = CmafChunk {
            sequence_number: self.chunk_sequence,
            segment_sequence: self.segment_sequence,
            chunk_index: self.chunk_index_in_segment,
            start_pts,
            duration,
            sample_count,
            data_size,
            starts_with_keyframe,
            is_last_in_segment: is_last,
            data: chunk_data,
            is_independent: starts_with_keyframe,
        };

        self.completed_chunks.push(chunk);
        self.chunk_sequence += 1;
        self.chunk_index_in_segment += 1;
        self.total_chunks_produced += 1;
        self.total_bytes_produced += data_size as u64;

        self.pending_samples.clear();
        self.pending_duration = 0;
    }

    /// Builds moof+mdat boxes for the current pending samples.
    ///
    /// Uses a two-pass build for `trun.data_offset`, mirroring the pattern in
    /// [`crate::mux::cmaf::CmafMuxer::emit_segment`]: the `moof` is first
    /// assembled with a placeholder offset of `0` purely to measure its total
    /// size, then rebuilt with the real offset. Because `tfhd` sets
    /// `default-base-is-moof` (flags `0x020000`), `trun.data_offset` is
    /// relative to the first byte of `moof`; the `mdat` box header is always
    /// 8 bytes, so the first sample byte sits at `moof.len() + 8`. Leaving
    /// the placeholder unpatched (as a previous revision of this function
    /// did) produces a `trun` that always points at offset 0 — i.e. at the
    /// `moof` box's own size field — which is not a valid low-latency
    /// fragment.
    fn build_chunk_boxes(&self) -> Vec<u8> {
        use crate::mux::cmaf::{write_box, write_full_box, write_u32_be, write_u64_be};

        if self.pending_samples.is_empty() {
            return Vec::new();
        }

        // Collect mdat payload
        let mut mdat_payload = Vec::new();
        for sample in &self.pending_samples {
            mdat_payload.extend_from_slice(&sample.data);
        }

        let track_id = self.pending_samples.first().map_or(1, |s| s.track_id);
        let base_dts = self.pending_samples.first().map_or(0, |s| s.dts);

        // Builds the complete moof for a given trun.data_offset value. Every
        // other field is independent of the offset, so this can be called
        // twice: once with a placeholder to measure the moof size, then again
        // with the real, now-known offset.
        let build_moof = |data_offset: u32| -> Vec<u8> {
            // flags: data_offset_present (0x1) | duration_present (0x100) | size_present (0x200)
            let trun_flags: u32 = 0x000301;
            let mut trun_content = Vec::new();
            trun_content.extend_from_slice(&write_u32_be(self.pending_samples.len() as u32));
            trun_content.extend_from_slice(&write_u32_be(data_offset));
            for sample in &self.pending_samples {
                trun_content.extend_from_slice(&write_u32_be(sample.duration));
                trun_content.extend_from_slice(&write_u32_be(sample.data.len() as u32));
            }
            let trun = write_full_box(b"trun", 0, trun_flags, &trun_content);

            let mut tfhd_content = Vec::new();
            tfhd_content.extend_from_slice(&write_u32_be(track_id));
            let tfhd = write_full_box(b"tfhd", 0, 0x020000, &tfhd_content);

            let mut tfdt_content = Vec::new();
            tfdt_content.extend_from_slice(&write_u64_be(base_dts));
            let tfdt = write_full_box(b"tfdt", 1, 0, &tfdt_content);

            let mut traf_content = Vec::new();
            traf_content.extend(tfhd);
            traf_content.extend(tfdt);
            traf_content.extend(trun);
            let traf = write_box(b"traf", &traf_content);

            let mut mfhd_content = Vec::new();
            mfhd_content.extend_from_slice(&write_u32_be(self.chunk_sequence));
            let mfhd = write_full_box(b"mfhd", 0, 0, &mfhd_content);

            let mut moof_content = Vec::new();
            moof_content.extend(mfhd);
            moof_content.extend(traf);
            write_box(b"moof", &moof_content)
        };

        // Pass 1: measure moof size with a placeholder data_offset.
        let moof_placeholder = build_moof(0);
        // Pass 2: rebuild with the real, now-known offset.
        let data_offset = moof_placeholder.len() as u32 + 8;
        let moof = build_moof(data_offset);

        let mdat = write_box(b"mdat", &mdat_payload);

        let mut output = Vec::with_capacity(moof.len() + mdat.len());
        output.extend(moof);
        output.extend(mdat);
        output
    }

    /// Flushes the current segment, emitting any remaining samples as the
    /// last chunk and advancing the segment sequence number.
    ///
    /// Returns the chunks produced for this final part of the segment.
    pub fn flush_segment(&mut self) -> Vec<CmafChunk> {
        if !self.pending_samples.is_empty() {
            self.emit_current_chunk(true);
        } else if let Some(last) = self.completed_chunks.last_mut() {
            last.is_last_in_segment = true;
        }

        let chunks = std::mem::take(&mut self.completed_chunks);
        self.segment_sequence += 1;
        self.chunk_index_in_segment = 0;
        chunks
    }

    /// Pops the oldest completed chunk, if available.
    pub fn pop_chunk(&mut self) -> Option<CmafChunk> {
        if self.completed_chunks.is_empty() {
            None
        } else {
            Some(self.completed_chunks.remove(0))
        }
    }

    /// Drains all completed chunks.
    pub fn drain_chunks(&mut self) -> Vec<CmafChunk> {
        std::mem::take(&mut self.completed_chunks)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CmafChunkWriter: simple low-latency chunk writer (one sample → one chunk)
// ─────────────────────────────────────────────────────────────────────────────

/// A single media sample to be written by [`CmafChunkWriter`].
#[derive(Debug, Clone)]
pub struct CmafSample {
    /// Presentation timestamp in timescale ticks.
    pub pts: i64,
    /// Duration in timescale ticks.
    pub duration: u32,
    /// Whether this is a keyframe (sync sample).
    pub is_keyframe: bool,
    /// Raw encoded sample bytes.
    pub data: Vec<u8>,
}

impl CmafSample {
    /// Creates a new `CmafSample`.
    #[must_use]
    pub fn new(pts: i64, duration: u32, is_keyframe: bool, data: Vec<u8>) -> Self {
        Self {
            pts,
            duration,
            is_keyframe,
            data,
        }
    }
}

/// A single CMAF chunk: one `moof`+`mdat` box pair ready for delivery.
#[derive(Debug, Clone)]
pub struct CmafChunkOwned {
    /// Raw bytes: `moof` + `mdat` boxes.
    pub data: Vec<u8>,
    /// Monotonically-increasing chunk index (0-based).
    pub chunk_idx: u64,
    /// PTS of the first sample in this chunk (timescale ticks).
    pub pts_start: i64,
    /// PTS of the last sample plus its duration (timescale ticks).
    pub pts_end: i64,
    /// Whether the chunk contains a keyframe (sync sample).
    pub is_keyframe: bool,
}

/// A complete CMAF segment: an ordered collection of [`CmafChunkOwned`] values
/// that together form one independently decodable CMAF track segment.
#[derive(Debug, Clone)]
pub struct CmafSegment {
    /// All chunks in this segment, in decode order.
    pub chunks: Vec<CmafChunkOwned>,
    /// 1-based segment index that advances with each call to
    /// [`CmafChunkWriter::flush_segment`].
    pub segment_idx: u64,
}

/// Simple CMAF low-latency chunk writer.
///
/// Each [`CmafSample`] pushed via [`write_sample`] is immediately serialised
/// into a `moof`+`mdat` box pair and returned as a [`CmafChunkOwned`].  Chunks
/// accumulate internally until [`flush_segment`] is called, which drains them
/// into a [`CmafSegment`] and advances the segment counter.
///
/// This one-sample-per-chunk design delivers the lowest possible latency: a
/// player can begin decoding as soon as the first chunk arrives on the wire.
///
/// # Example
///
/// ```ignore
/// let mut writer = CmafChunkWriter::new(500);
///
/// if let Some(chunk) = writer.write_sample(&sample) {
///     deliver_to_cdn(chunk.data);
/// }
///
/// // At segment boundary:
/// if let Some(seg) = writer.flush_segment() {
///     archive_segment(seg);
/// }
/// ```
///
/// [`write_sample`]: CmafChunkWriter::write_sample
/// [`flush_segment`]: CmafChunkWriter::flush_segment
#[derive(Debug)]
pub struct CmafChunkWriter {
    /// Nominal target chunk duration in milliseconds (informational only;
    /// actual chunking is one sample per chunk).
    pub chunk_duration_ms: u32,
    /// Chunks accumulated since the last [`flush_segment`] call.
    current_chunk: Vec<CmafChunkOwned>,
    /// Monotonically-increasing global chunk counter.
    chunk_count: u64,
    /// 1-based current segment index.
    segment_count: u64,
}

impl CmafChunkWriter {
    /// Creates a new writer with the given nominal chunk duration in
    /// milliseconds.
    #[must_use]
    pub fn new(chunk_duration_ms: u32) -> Self {
        Self {
            chunk_duration_ms,
            current_chunk: Vec::new(),
            chunk_count: 0,
            segment_count: 1,
        }
    }

    /// Writes `sample`, serialises it into a `moof`+`mdat` chunk, and returns
    /// that chunk.  The chunk is also retained internally until
    /// [`flush_segment`] is called.
    ///
    /// Always returns `Some` (the `Option` wrapper follows the interface
    /// contract for chunk writers that may buffer samples).
    ///
    /// [`flush_segment`]: CmafChunkWriter::flush_segment
    pub fn write_sample(&mut self, sample: &CmafSample) -> Option<CmafChunkOwned> {
        let chunk = self.build_chunk(sample);
        self.current_chunk.push(chunk.clone());
        Some(chunk)
    }

    /// Flushes any in-progress partial chunk.
    ///
    /// In this one-sample-per-chunk implementation samples are emitted
    /// immediately, so this is always a no-op and returns `None`.
    pub fn flush_chunk(&mut self) -> Option<CmafChunkOwned> {
        None
    }

    /// Drains all accumulated chunks into a [`CmafSegment`] and advances the
    /// segment counter.
    ///
    /// Returns `None` if no samples have been written since the last call.
    pub fn flush_segment(&mut self) -> Option<CmafSegment> {
        if self.current_chunk.is_empty() {
            return None;
        }
        let chunks = std::mem::take(&mut self.current_chunk);
        let seg = CmafSegment {
            chunks,
            segment_idx: self.segment_count,
        };
        self.segment_count += 1;
        Some(seg)
    }

    /// Returns the total number of chunks written so far.
    #[must_use]
    pub fn chunk_count(&self) -> u64 {
        self.chunk_count
    }

    /// Returns the current 1-based segment index (incremented by each
    /// [`flush_segment`] call).
    ///
    /// [`flush_segment`]: CmafChunkWriter::flush_segment
    #[must_use]
    pub fn segment_count(&self) -> u64 {
        self.segment_count
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    /// Serialises `sample` into a `moof`+`mdat` box pair.
    ///
    /// Uses the same two-pass `trun.data_offset` patch as
    /// [`CmafChunkedEncoder::build_chunk_boxes`] and
    /// [`crate::mux::cmaf::CmafMuxer::emit_segment`]: `moof` is built once
    /// with a placeholder offset to measure its size, then rebuilt with the
    /// real offset (`moof.len() + 8`, since `tfhd` sets
    /// `default-base-is-moof` and the `mdat` header is 8 bytes).
    fn build_chunk(&mut self, sample: &CmafSample) -> CmafChunkOwned {
        use crate::mux::cmaf::{write_box, write_full_box, write_u32_be, write_u64_be};

        let chunk_idx = self.chunk_count;
        self.chunk_count += 1;

        // Sequence numbers in ISO BMFF are 1-based.
        let sequence_number = (chunk_idx as u32).wrapping_add(1);
        let track_id: u32 = 1;

        // ── mdat ──────────────────────────────────────────────────────────
        let mdat = write_box(b"mdat", &sample.data);

        let dts: u64 = if sample.pts >= 0 {
            sample.pts as u64
        } else {
            0
        };

        // Builds the complete moof for a given trun.data_offset value; see
        // the two-pass explanation on the doc comment above.
        let build_moof = |data_offset: u32| -> Vec<u8> {
            // ── trun (track run box) ────────────────────────────────────
            // flags: data-offset-present (0x001) | sample-duration-present (0x100)
            //        | sample-size-present (0x200)
            let trun_flags: u32 = 0x0000_0301;
            let mut trun_content = Vec::new();
            trun_content.extend_from_slice(&write_u32_be(1_u32)); // sample_count
            trun_content.extend_from_slice(&write_u32_be(data_offset));
            trun_content.extend_from_slice(&write_u32_be(sample.duration));
            trun_content.extend_from_slice(&write_u32_be(sample.data.len() as u32));
            let trun = write_full_box(b"trun", 0, trun_flags, &trun_content);

            // ── tfhd (track fragment header) ────────────────────────────
            // flags: default-base-is-moof (0x020000)
            let mut tfhd_content = Vec::new();
            tfhd_content.extend_from_slice(&write_u32_be(track_id));
            let tfhd = write_full_box(b"tfhd", 0, 0x0002_0000, &tfhd_content);

            // ── tfdt (track fragment decode time, version 1 = 64-bit) ───
            let mut tfdt_content = Vec::new();
            tfdt_content.extend_from_slice(&write_u64_be(dts));
            let tfdt = write_full_box(b"tfdt", 1, 0, &tfdt_content);

            // ── traf (track fragment) ────────────────────────────────────
            let mut traf_content = Vec::new();
            traf_content.extend_from_slice(&tfhd);
            traf_content.extend_from_slice(&tfdt);
            traf_content.extend_from_slice(&trun);
            let traf = write_box(b"traf", &traf_content);

            // ── mfhd (movie fragment header) ─────────────────────────────
            let mut mfhd_content = Vec::new();
            mfhd_content.extend_from_slice(&write_u32_be(sequence_number));
            let mfhd = write_full_box(b"mfhd", 0, 0, &mfhd_content);

            // ── moof (movie fragment) ─────────────────────────────────────
            let mut moof_content = Vec::new();
            moof_content.extend_from_slice(&mfhd);
            moof_content.extend_from_slice(&traf);
            write_box(b"moof", &moof_content)
        };

        // Pass 1: measure moof size with a placeholder data_offset.
        let moof_placeholder = build_moof(0);
        // Pass 2: rebuild with the real, now-known offset.
        let data_offset = moof_placeholder.len() as u32 + 8;
        let moof = build_moof(data_offset);

        // ── combine ───────────────────────────────────────────────────────
        let mut data = Vec::with_capacity(moof.len() + mdat.len());
        data.extend_from_slice(&moof);
        data.extend_from_slice(&mdat);

        let pts_end = sample.pts.saturating_add(i64::from(sample.duration));

        CmafChunkOwned {
            data,
            chunk_idx,
            pts_start: sample.pts,
            pts_end,
            is_keyframe: sample.is_keyframe,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Finds the first direct child box named `fourcc` within `data[start..end]`.
    ///
    /// Returns the absolute `(start, end)` byte range of the matched box
    /// (including its own 8-byte header), or `None` if no such child exists
    /// before `end`. Used to walk the `moof > traf > trun` hierarchy in
    /// structural tests without hardcoding byte offsets that would silently
    /// go stale if box contents change.
    fn find_child_box(
        data: &[u8],
        start: usize,
        end: usize,
        fourcc: &[u8; 4],
    ) -> Option<(usize, usize)> {
        let mut offset = start;
        while offset + 8 <= end {
            let size = u32::from_be_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            if size < 8 || offset + size > end {
                break;
            }
            if &data[offset + 4..offset + 8] == fourcc {
                return Some((offset, offset + size));
            }
            offset += size;
        }
        None
    }

    /// Extracts the `trun.data_offset` field (a `u32`, per version-0 `trun`)
    /// from a `moof` box located at `data[moof_start..moof_end]`, descending
    /// through its first `traf` child.
    fn trun_data_offset(data: &[u8], moof_start: usize, moof_end: usize) -> u32 {
        let (traf_start, traf_end) = find_child_box(data, moof_start + 8, moof_end, b"traf")
            .expect("moof must contain a traf box");
        let (trun_start, _trun_end) = find_child_box(data, traf_start + 8, traf_end, b"trun")
            .expect("traf must contain a trun box");
        // trun layout: [size:4][fourcc:4][version:1][flags:3][sample_count:4][data_offset:4]...
        let field_start = trun_start + 8 + 4 + 4;
        u32::from_be_bytes(
            data[field_start..field_start + 4]
                .try_into()
                .expect("4 bytes"),
        )
    }

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingMuxerConfig::default();
        assert_eq!(config.target_latency_ms, 1000);
        assert!(!config.low_latency);
        assert!(config.fragment_duration_ms.is_none());
        assert!(!config.realtime);
    }

    #[test]
    fn test_streaming_config_builder() {
        let config = StreamingMuxerConfig::new()
            .with_low_latency(true)
            .with_target_latency(500)
            .with_fragment_duration(2000)
            .with_realtime(true);

        assert!(config.low_latency);
        assert_eq!(config.target_latency_ms, 500);
        assert_eq!(config.fragment_duration_ms, Some(2000));
        assert!(config.realtime);
    }

    #[test]
    fn test_muxing_stats() {
        let mut stats = MuxingStats::new();
        assert_eq!(stats.packets_written, 0);
        assert_eq!(stats.bytes_written, 0);

        stats.update(1000, 1.0);
        assert_eq!(stats.packets_written, 1);
        assert_eq!(stats.bytes_written, 1000);
        assert!(stats.avg_bitrate > 0.0);

        stats.update(2000, 2.0);
        assert_eq!(stats.packets_written, 2);
        assert_eq!(stats.bytes_written, 3000);
    }

    #[test]
    fn test_latency_monitor() {
        let mut monitor = LatencyMonitor::new(Duration::from_millis(100));

        monitor.record(Duration::from_millis(50));
        monitor.record(Duration::from_millis(60));
        monitor.record(Duration::from_millis(70));

        let avg = monitor.average_latency().expect("operation should succeed");
        assert!(avg >= Duration::from_millis(59) && avg <= Duration::from_millis(61));
        assert!(monitor.is_within_target());
    }

    // ── CMAF Chunked Config tests ───────────────────────────────────────

    #[test]
    fn test_cmaf_chunked_config_default() {
        let config = CmafChunkedConfig::new();
        assert_eq!(config.mode, CmafChunkMode::Standard);
        assert_eq!(config.chunk_duration_ms, 500);
        assert_eq!(config.max_samples_per_chunk, 5);
        assert!(!config.include_mfra);
        assert!(!config.signal_low_latency);
        assert!(config.part_target_duration_ms.is_none());
    }

    #[test]
    fn test_cmaf_chunked_config_builder() {
        let config = CmafChunkedConfig::new()
            .with_mode(CmafChunkMode::LowLatencyChunked)
            .with_chunk_duration_ms(200)
            .with_max_samples_per_chunk(1)
            .with_mfra(true)
            .with_low_latency_signal(true)
            .with_part_target_duration_ms(333);

        assert_eq!(config.mode, CmafChunkMode::LowLatencyChunked);
        assert_eq!(config.chunk_duration_ms, 200);
        assert_eq!(config.max_samples_per_chunk, 1);
        assert!(config.include_mfra);
        assert!(config.signal_low_latency);
        assert_eq!(config.part_target_duration_ms, Some(333));
    }

    // ── CMAF Chunked Encoder tests ──────────────────────────────────────

    fn make_sample(pts: u64, duration: u32, keyframe: bool) -> ChunkSample {
        ChunkSample {
            pts,
            dts: pts,
            duration,
            data: vec![0xAA; 100],
            is_keyframe: keyframe,
            track_id: 1,
        }
    }

    #[test]
    fn test_encoder_new() {
        let encoder = CmafChunkedEncoder::new(CmafChunkedConfig::new(), 90000);
        assert_eq!(encoder.timescale(), 90000);
        assert_eq!(encoder.total_chunks_produced(), 0);
        assert_eq!(encoder.total_bytes_produced(), 0);
        assert_eq!(encoder.pending_sample_count(), 0);
        assert_eq!(encoder.available_chunks(), 0);
    }

    #[test]
    fn test_standard_mode_no_auto_chunks() {
        let config = CmafChunkedConfig::new().with_mode(CmafChunkMode::Standard);
        let mut encoder = CmafChunkedEncoder::new(config, 90000);

        encoder.push_sample(make_sample(0, 3000, true));
        encoder.push_sample(make_sample(3000, 3000, false));
        encoder.push_sample(make_sample(6000, 3000, false));

        // Standard mode doesn't auto-emit chunks
        assert_eq!(encoder.available_chunks(), 0);
        assert_eq!(encoder.pending_sample_count(), 3);
    }

    #[test]
    fn test_low_latency_one_sample_per_chunk() {
        let config = CmafChunkedConfig::new().with_mode(CmafChunkMode::LowLatencyChunked);
        let mut encoder = CmafChunkedEncoder::new(config, 90000);

        encoder.push_sample(make_sample(0, 3000, true));
        assert_eq!(encoder.available_chunks(), 1);

        encoder.push_sample(make_sample(3000, 3000, false));
        assert_eq!(encoder.available_chunks(), 2);

        encoder.push_sample(make_sample(6000, 3000, false));
        assert_eq!(encoder.available_chunks(), 3);
    }

    #[test]
    fn test_chunked_mode_duration_based() {
        // 90kHz timescale, chunk_duration_ms=100, which is 9000 ticks
        let config = CmafChunkedConfig::new()
            .with_mode(CmafChunkMode::Chunked)
            .with_chunk_duration_ms(100)
            .with_max_samples_per_chunk(100);
        let mut encoder = CmafChunkedEncoder::new(config, 90000);

        // Each sample is 3000 ticks (~33ms), need 3 samples to exceed 9000 ticks
        encoder.push_sample(make_sample(0, 3000, true));
        assert_eq!(encoder.available_chunks(), 0);

        encoder.push_sample(make_sample(3000, 3000, false));
        assert_eq!(encoder.available_chunks(), 0);

        encoder.push_sample(make_sample(6000, 3000, false));
        // 9000 ticks >= 9000 target, should emit
        assert_eq!(encoder.available_chunks(), 1);
    }

    #[test]
    fn test_chunked_mode_max_samples() {
        let config = CmafChunkedConfig::new()
            .with_mode(CmafChunkMode::Chunked)
            .with_chunk_duration_ms(99999) // very high duration
            .with_max_samples_per_chunk(2);
        let mut encoder = CmafChunkedEncoder::new(config, 90000);

        encoder.push_sample(make_sample(0, 3000, true));
        assert_eq!(encoder.available_chunks(), 0);

        encoder.push_sample(make_sample(3000, 3000, false));
        // Max samples per chunk = 2, should emit
        assert_eq!(encoder.available_chunks(), 1);
    }

    #[test]
    fn test_pop_chunk() {
        let config = CmafChunkedConfig::new().with_mode(CmafChunkMode::LowLatencyChunked);
        let mut encoder = CmafChunkedEncoder::new(config, 90000);

        encoder.push_sample(make_sample(0, 3000, true));
        encoder.push_sample(make_sample(3000, 3000, false));

        let chunk1 = encoder.pop_chunk().expect("should have chunk");
        assert_eq!(chunk1.sequence_number, 1);
        assert!(chunk1.starts_with_keyframe);

        let chunk2 = encoder.pop_chunk().expect("should have chunk");
        assert_eq!(chunk2.sequence_number, 2);
        assert!(!chunk2.starts_with_keyframe);

        assert!(encoder.pop_chunk().is_none());
    }

    #[test]
    fn test_drain_chunks() {
        let config = CmafChunkedConfig::new().with_mode(CmafChunkMode::LowLatencyChunked);
        let mut encoder = CmafChunkedEncoder::new(config, 90000);

        for i in 0..5 {
            encoder.push_sample(make_sample(i * 3000, 3000, i == 0));
        }

        let chunks = encoder.drain_chunks();
        assert_eq!(chunks.len(), 5);
        assert_eq!(encoder.available_chunks(), 0);
    }

    #[test]
    fn test_flush_segment() {
        let config = CmafChunkedConfig::new().with_mode(CmafChunkMode::Standard);
        let mut encoder = CmafChunkedEncoder::new(config, 90000);

        encoder.push_sample(make_sample(0, 3000, true));
        encoder.push_sample(make_sample(3000, 3000, false));

        let chunks = encoder.flush_segment();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_last_in_segment);
        assert_eq!(chunks[0].sample_count, 2);

        // Segment sequence should advance
        assert_eq!(encoder.current_segment_sequence(), 2);
    }

    #[test]
    fn test_flush_segment_with_existing_chunks() {
        let config = CmafChunkedConfig::new().with_mode(CmafChunkMode::LowLatencyChunked);
        let mut encoder = CmafChunkedEncoder::new(config, 90000);

        encoder.push_sample(make_sample(0, 3000, true));
        encoder.push_sample(make_sample(3000, 3000, false));
        // 2 chunks auto-emitted, plus pending is empty

        let chunks = encoder.flush_segment();
        assert_eq!(chunks.len(), 2);
        assert!(chunks.last().map_or(false, |c| c.is_last_in_segment));
    }

    #[test]
    fn test_chunk_contains_moof_mdat() {
        let config = CmafChunkedConfig::new().with_mode(CmafChunkMode::LowLatencyChunked);
        let mut encoder = CmafChunkedEncoder::new(config, 90000);

        encoder.push_sample(make_sample(0, 3000, true));
        let chunk = encoder.pop_chunk().expect("should have chunk");

        assert!(!chunk.data.is_empty());
        let has_moof = chunk.data.windows(4).any(|w| w == b"moof");
        let has_mdat = chunk.data.windows(4).any(|w| w == b"mdat");
        assert!(has_moof, "chunk must contain moof box");
        assert!(has_mdat, "chunk must contain mdat box");
    }

    #[test]
    fn test_chunk_metadata() {
        let config = CmafChunkedConfig::new().with_mode(CmafChunkMode::LowLatencyChunked);
        let mut encoder = CmafChunkedEncoder::new(config, 90000);

        encoder.push_sample(make_sample(9000, 3000, true));
        let chunk = encoder.pop_chunk().expect("should have chunk");

        assert_eq!(chunk.start_pts, 9000);
        assert_eq!(chunk.duration, 3000);
        assert_eq!(chunk.sample_count, 1);
        assert!(chunk.starts_with_keyframe);
        assert!(chunk.is_independent);
        assert_eq!(chunk.segment_sequence, 1);
        assert_eq!(chunk.chunk_index, 0);
    }

    // 4a. trun.data_offset must be patched to the real mdat payload offset,
    // not left at the placeholder 0. This is a structural read-back test:
    // it parses the produced moof/traf/trun boxes and mdat position, then
    // proves the two agree (both as a numeric equality AND by confirming the
    // sample bytes are actually found at that exact offset).
    #[test]
    fn test_chunked_encoder_data_offset_is_patched() {
        let config = CmafChunkedConfig::new().with_mode(CmafChunkMode::LowLatencyChunked);
        let mut encoder = CmafChunkedEncoder::new(config, 90000);
        encoder.push_sample(make_sample(0, 3000, true));
        let chunk = encoder.pop_chunk().expect("should have chunk");

        let (moof_start, moof_end) =
            find_child_box(&chunk.data, 0, chunk.data.len(), b"moof").expect("moof present");
        let (mdat_start, mdat_end) =
            find_child_box(&chunk.data, 0, chunk.data.len(), b"mdat").expect("mdat present");
        assert_eq!(moof_start, 0, "moof must be the first box");
        assert_eq!(mdat_start, moof_end, "mdat must immediately follow moof");

        let data_offset = trun_data_offset(&chunk.data, moof_start, moof_end);
        assert_ne!(
            data_offset, 0,
            "data_offset must not be left at the unpatched placeholder value"
        );
        assert_eq!(
            data_offset as usize,
            moof_end - moof_start + 8,
            "data_offset must equal moof size + 8-byte mdat header"
        );

        // The offset must point at the real sample bytes: mdat's payload
        // (100 bytes of 0xAA, per make_sample) starts at mdat_start + 8.
        assert_eq!(mdat_start + 8, moof_start + data_offset as usize);
        let sample_bytes = &chunk.data[moof_start + data_offset as usize..mdat_end];
        assert_eq!(sample_bytes, vec![0xAAu8; 100].as_slice());
    }

    #[test]
    fn test_chunk_indices_increment() {
        let config = CmafChunkedConfig::new().with_mode(CmafChunkMode::LowLatencyChunked);
        let mut encoder = CmafChunkedEncoder::new(config, 90000);

        for i in 0..3 {
            encoder.push_sample(make_sample(i * 3000, 3000, i == 0));
        }

        let chunks = encoder.drain_chunks();
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[1].chunk_index, 1);
        assert_eq!(chunks[2].chunk_index, 2);
    }

    #[test]
    fn test_segment_sequence_advances() {
        let config = CmafChunkedConfig::new().with_mode(CmafChunkMode::LowLatencyChunked);
        let mut encoder = CmafChunkedEncoder::new(config, 90000);

        encoder.push_sample(make_sample(0, 3000, true));
        encoder.flush_segment();
        assert_eq!(encoder.current_segment_sequence(), 2);

        encoder.push_sample(make_sample(3000, 3000, true));
        encoder.flush_segment();
        assert_eq!(encoder.current_segment_sequence(), 3);
    }

    #[test]
    fn test_total_bytes_tracked() {
        let config = CmafChunkedConfig::new().with_mode(CmafChunkMode::LowLatencyChunked);
        let mut encoder = CmafChunkedEncoder::new(config, 90000);

        encoder.push_sample(make_sample(0, 3000, true));
        encoder.push_sample(make_sample(3000, 3000, false));

        assert_eq!(encoder.total_chunks_produced(), 2);
        assert!(encoder.total_bytes_produced() > 0);
    }

    #[test]
    fn test_flush_empty_encoder() {
        let config = CmafChunkedConfig::new().with_mode(CmafChunkMode::Standard);
        let mut encoder = CmafChunkedEncoder::new(config, 90000);

        let chunks = encoder.flush_segment();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_multiple_segments() {
        let config = CmafChunkedConfig::new().with_mode(CmafChunkMode::LowLatencyChunked);
        let mut encoder = CmafChunkedEncoder::new(config, 90000);

        // Segment 1: 3 chunks
        for i in 0..3 {
            encoder.push_sample(make_sample(i * 3000, 3000, i == 0));
        }
        let seg1 = encoder.flush_segment();
        assert_eq!(seg1.len(), 3);
        assert_eq!(seg1[0].segment_sequence, 1);

        // Segment 2: 2 chunks
        for i in 0..2 {
            encoder.push_sample(make_sample(9000 + i * 3000, 3000, i == 0));
        }
        let seg2 = encoder.flush_segment();
        assert_eq!(seg2.len(), 2);
        assert_eq!(seg2[0].segment_sequence, 2);
        assert_eq!(seg2[0].chunk_index, 0); // reset per segment
    }

    // ── CmafChunkWriter tests ─────────────────────────────────────────────

    fn make_cmaf_sample(pts: i64, duration: u32, is_keyframe: bool) -> CmafSample {
        CmafSample::new(pts, duration, is_keyframe, vec![0xBB; 256])
    }

    #[test]
    fn test_chunk_writer_basic_moof_mdat() {
        let mut writer = CmafChunkWriter::new(500);
        let s = make_cmaf_sample(0, 3000, true);
        let chunk = writer.write_sample(&s).expect("should return chunk");
        assert!(!chunk.data.is_empty());
        assert!(
            chunk.data.windows(4).any(|w| w == b"moof"),
            "chunk must contain moof box"
        );
        assert!(
            chunk.data.windows(4).any(|w| w == b"mdat"),
            "chunk must contain mdat box"
        );
    }

    // trun.data_offset must be patched for CmafChunkWriter's output too
    // (same underlying bug/fix as CmafChunkedEncoder, different call site).
    #[test]
    fn test_chunk_writer_data_offset_is_patched() {
        let mut writer = CmafChunkWriter::new(500);
        let s = make_cmaf_sample(0, 3000, true); // 256 bytes of 0xBB, see make_cmaf_sample
        let chunk = writer.write_sample(&s).expect("should return chunk");

        let (moof_start, moof_end) =
            find_child_box(&chunk.data, 0, chunk.data.len(), b"moof").expect("moof present");
        let (mdat_start, mdat_end) =
            find_child_box(&chunk.data, 0, chunk.data.len(), b"mdat").expect("mdat present");
        assert_eq!(mdat_start, moof_end, "mdat must immediately follow moof");

        let data_offset = trun_data_offset(&chunk.data, moof_start, moof_end);
        assert_ne!(
            data_offset, 0,
            "data_offset must not stay at the placeholder"
        );
        assert_eq!(data_offset as usize, moof_end - moof_start + 8);

        let sample_bytes = &chunk.data[moof_start + data_offset as usize..mdat_end];
        assert_eq!(sample_bytes, vec![0xBBu8; 256].as_slice());
    }

    #[test]
    fn test_chunk_writer_pts_range() {
        let mut writer = CmafChunkWriter::new(500);
        let s = make_cmaf_sample(9000, 3000, false);
        let chunk = writer.write_sample(&s).expect("should return chunk");
        assert_eq!(chunk.pts_start, 9000);
        assert_eq!(chunk.pts_end, 12000);
        assert!(!chunk.is_keyframe);
    }

    #[test]
    fn test_chunk_writer_keyframe_flag() {
        let mut writer = CmafChunkWriter::new(500);
        let s = make_cmaf_sample(0, 3000, true);
        let chunk = writer.write_sample(&s).expect("should return chunk");
        assert!(chunk.is_keyframe);
    }

    #[test]
    fn test_chunk_writer_idx_increments() {
        let mut writer = CmafChunkWriter::new(500);
        for i in 0..4_i64 {
            let s = make_cmaf_sample(i * 3000, 3000, i == 0);
            let chunk = writer.write_sample(&s).expect("should return chunk");
            assert_eq!(chunk.chunk_idx, i as u64);
        }
        assert_eq!(writer.chunk_count(), 4);
    }

    #[test]
    fn test_chunk_writer_flush_segment_collects() {
        let mut writer = CmafChunkWriter::new(500);
        for i in 0..3_i64 {
            writer.write_sample(&make_cmaf_sample(i * 3000, 3000, i == 0));
        }
        let seg = writer.flush_segment().expect("should have segment");
        assert_eq!(seg.chunks.len(), 3);
        assert_eq!(seg.segment_idx, 1);
        assert_eq!(writer.segment_count(), 2);
    }

    #[test]
    fn test_chunk_writer_flush_segment_advances_counter() {
        let mut writer = CmafChunkWriter::new(500);
        writer.write_sample(&make_cmaf_sample(0, 3000, true));
        let seg1 = writer.flush_segment().expect("seg1");
        assert_eq!(seg1.segment_idx, 1);

        writer.write_sample(&make_cmaf_sample(3000, 3000, true));
        let seg2 = writer.flush_segment().expect("seg2");
        assert_eq!(seg2.segment_idx, 2);
        assert_eq!(writer.segment_count(), 3);
    }

    #[test]
    fn test_chunk_writer_flush_empty_is_none() {
        let mut writer = CmafChunkWriter::new(500);
        assert!(writer.flush_segment().is_none());
        // After writing and flushing, a second flush with no new data is None
        writer.write_sample(&make_cmaf_sample(0, 3000, true));
        writer.flush_segment();
        assert!(writer.flush_segment().is_none());
    }

    #[test]
    fn test_chunk_writer_flush_chunk_noop() {
        let mut writer = CmafChunkWriter::new(500);
        // flush_chunk is always None in this implementation
        assert!(writer.flush_chunk().is_none());
        writer.write_sample(&make_cmaf_sample(0, 3000, true));
        assert!(writer.flush_chunk().is_none());
    }

    #[test]
    fn test_chunk_writer_negative_pts_clamped() {
        let mut writer = CmafChunkWriter::new(500);
        // Negative PTS should not panic (DTS is clamped to 0 internally)
        let s = make_cmaf_sample(-1000, 3000, false);
        let chunk = writer.write_sample(&s).expect("should return chunk");
        assert_eq!(chunk.pts_start, -1000);
        assert_eq!(chunk.pts_end, 2000);
    }
}
