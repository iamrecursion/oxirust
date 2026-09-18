//! Fragmented MP4 (fMP4) support for DASH/HLS.
//!
//! Implements fragmented MP4 format used in adaptive streaming protocols
//! like MPEG-DASH and HLS.

#![forbid(unsafe_code)]

use oximedia_core::{CodecId, MediaType, OxiError, OxiResult, Rational, Timestamp};
use std::time::Duration;

use crate::mux::cmaf::{CmafMuxer, CmafSample as MuxCmafSample, CmafTrack, TrackType};
use crate::{Packet, StreamInfo};

// ─── StreamInfo/Packet → CmafTrack/CmafSample conversion ──────────────────────
//
// `FragmentedMp4Builder` below delegates all real ISO-BMFF box emission to
// `crate::mux::cmaf::CmafMuxer`, which already produces a complete,
// spec-conformant fMP4/CMAF init segment (ftyp+moov with mvex/trex, trak,
// mdia, minf incl. vmhd/smhd+dinf+stbl) and media segments (moof+mdat with a
// correctly two-pass-patched trun.data_offset). These helpers adapt this
// module's `StreamInfo`/`Packet` types to the muxer's `CmafTrack`/`CmafSample`.

/// Maps a codec to its ISO-BMFF sample entry four-character code.
///
/// Mirrors the canonical mapping in
/// `crate::mux::mp4::writer::boxes::codec_to_fourcc` (kept as a small local
/// copy since that function is private to the `mux::mp4::writer` module
/// tree).
fn codec_fourcc(codec: CodecId) -> [u8; 4] {
    match codec {
        CodecId::Av1 => *b"av01",
        CodecId::Vp9 => *b"vp09",
        CodecId::Vp8 => *b"vp08",
        CodecId::Opus => *b"Opus",
        CodecId::Flac => *b"fLaC",
        CodecId::Vorbis => *b"vorb",
        CodecId::Apv => *b"apv1",
        CodecId::Mjpeg => *b"jpeg",
        _ => *b"unkn",
    }
}

/// Maps a stream's [`MediaType`] to the coarser [`TrackType`] classification
/// `CmafMuxer` needs (video / audio / everything else grouped as subtitle).
const fn track_type_for(media_type: MediaType) -> TrackType {
    match media_type {
        MediaType::Video => TrackType::Video,
        MediaType::Audio => TrackType::Audio,
        MediaType::Subtitle | MediaType::Data | MediaType::Attachment => TrackType::Subtitle,
    }
}

/// Builds the [`CmafTrack`] descriptor `CmafMuxer` needs to emit this
/// stream's `trak`/`trex` boxes, from this module's own [`StreamInfo`].
fn build_cmaf_track(track_id: u32, info: &StreamInfo) -> CmafTrack {
    CmafTrack {
        track_id,
        track_type: track_type_for(info.media_type),
        codec_fourcc: codec_fourcc(info.codec),
        timescale: u32::try_from(info.timebase.den).unwrap_or(1),
        width: info.codec_params.width,
        height: info.codec_params.height,
        sample_rate: info.codec_params.sample_rate,
        channels: info.codec_params.channels,
        extradata: info
            .codec_params
            .extradata
            .as_ref()
            .map(|b| b.to_vec())
            .unwrap_or_default(),
    }
}

/// Converts a [`Packet`] into the [`MuxCmafSample`] `CmafMuxer` needs to
/// emit `trun`/`mdat` entries. Negative PTS/DTS (e.g. from B-frame reorder
/// buffers, or a caller that hasn't normalized timestamps yet) are clamped
/// to `0` rather than wrapping, matching the same clamp used by
/// [`crate::streaming::mux::CmafChunkWriter::build_chunk`].
fn build_cmaf_sample(packet: &Packet) -> MuxCmafSample {
    #[allow(clippy::cast_possible_truncation)]
    let track_id = (packet.stream_index as u32).wrapping_add(1);
    let pts = u64::try_from(packet.pts()).unwrap_or(0);
    let dts = packet
        .dts()
        .and_then(|d| u64::try_from(d).ok())
        .unwrap_or(pts);
    let duration = packet
        .duration()
        .and_then(|d| u32::try_from(d).ok())
        .unwrap_or(0);

    MuxCmafSample {
        track_id,
        pts,
        dts,
        duration,
        data: packet.data.to_vec(),
        keyframe: packet.is_keyframe(),
    }
}

/// Configuration for fragmented MP4.
#[derive(Clone, Debug)]
pub struct FragmentedMp4Config {
    /// Fragment duration in milliseconds.
    pub fragment_duration_ms: u64,
    /// Enable separate initialization segment.
    pub separate_init_segment: bool,
    /// Enable self-initializing segments.
    pub self_initializing: bool,
    /// Fragment sequence number (increments for each fragment).
    pub sequence_number: u32,
    /// Enable single fragment per segment.
    pub single_fragment: bool,
}

impl Default for FragmentedMp4Config {
    fn default() -> Self {
        Self {
            fragment_duration_ms: 2000, // 2 seconds
            separate_init_segment: true,
            self_initializing: false,
            sequence_number: 1,
            single_fragment: true,
        }
    }
}

impl FragmentedMp4Config {
    /// Creates a new configuration with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            fragment_duration_ms: 2000,
            separate_init_segment: true,
            self_initializing: false,
            sequence_number: 1,
            single_fragment: true,
        }
    }

    /// Sets the fragment duration.
    #[must_use]
    pub const fn with_fragment_duration(mut self, duration_ms: u64) -> Self {
        self.fragment_duration_ms = duration_ms;
        self
    }

    /// Enables separate initialization segment.
    #[must_use]
    pub const fn with_separate_init(mut self, enabled: bool) -> Self {
        self.separate_init_segment = enabled;
        self
    }

    /// Enables self-initializing segments.
    #[must_use]
    pub const fn with_self_initializing(mut self, enabled: bool) -> Self {
        self.self_initializing = enabled;
        self
    }

    /// Sets the starting sequence number.
    #[must_use]
    pub const fn with_sequence_number(mut self, number: u32) -> Self {
        self.sequence_number = number;
        self
    }
}

/// Type of MP4 fragment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentType {
    /// Initialization segment (ftyp + moov).
    Init,
    /// Media fragment (moof + mdat).
    Media,
}

/// An MP4 fragment.
#[derive(Debug, Clone)]
pub struct Mp4Fragment {
    /// Type of fragment.
    pub fragment_type: FragmentType,
    /// Sequence number.
    pub sequence: u32,
    /// Fragment data.
    pub data: Vec<u8>,
    /// Duration of the fragment in microseconds.
    pub duration_us: u64,
    /// Start timestamp.
    pub start_timestamp: Timestamp,
    /// Stream indices included in this fragment.
    pub stream_indices: Vec<usize>,
    /// Whether this fragment contains a keyframe.
    pub has_keyframe: bool,
}

impl Mp4Fragment {
    /// Creates a new MP4 fragment.
    #[must_use]
    pub fn new(fragment_type: FragmentType, sequence: u32) -> Self {
        Self {
            fragment_type,
            sequence,
            data: Vec::new(),
            duration_us: 0,
            start_timestamp: Timestamp::new(0, Rational::new(1, 1)),
            stream_indices: Vec::new(),
            has_keyframe: false,
        }
    }

    /// Returns the size of the fragment in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns the duration as a `Duration`.
    #[must_use]
    pub const fn duration(&self) -> Duration {
        Duration::from_micros(self.duration_us)
    }

    /// Returns true if this is an initialization fragment.
    #[must_use]
    pub const fn is_init(&self) -> bool {
        matches!(self.fragment_type, FragmentType::Init)
    }

    /// Returns true if this is a media fragment.
    #[must_use]
    pub const fn is_media(&self) -> bool {
        matches!(self.fragment_type, FragmentType::Media)
    }
}

/// Builder for constructing fragmented MP4 streams.
///
/// Real ISO-BMFF box emission (`ftyp`/`moov` for the init segment,
/// `moof`/`mdat` for each media fragment) is delegated to an internal
/// [`CmafMuxer`], which already implements a complete, spec-conformant fMP4
/// writer (including the two-pass `trun.data_offset` patch). This builder's
/// own job is bookkeeping: tracking fragment boundaries, durations, and
/// keyframe/stream-index metadata around that muxer.
#[derive(Debug)]
pub struct FragmentedMp4Builder {
    config: FragmentedMp4Config,
    streams: Vec<StreamInfo>,
    #[allow(dead_code)]
    current_fragment: Option<Mp4Fragment>,
    fragment_start_time: Option<i64>,
    packets_in_fragment: Vec<Packet>,
    /// Emits the real `ftyp`/`moov`/`moof`/`mdat` boxes; kept in lock-step
    /// with `streams` (one `add_track` per `add_stream`) and with
    /// `config.sequence_number` (seeded in [`Self::new`] so `moof.mfhd`
    /// reflects whatever starting sequence number the caller configured).
    cmaf_muxer: CmafMuxer,
}

impl FragmentedMp4Builder {
    /// Creates a new builder with the given configuration.
    #[must_use]
    pub fn new(config: FragmentedMp4Config) -> Self {
        let mut cmaf_muxer = CmafMuxer::new(crate::mux::cmaf::CmafConfig::default());
        cmaf_muxer.set_sequence_number(config.sequence_number);
        Self {
            config,
            streams: Vec::new(),
            current_fragment: None,
            fragment_start_time: None,
            packets_in_fragment: Vec::new(),
            cmaf_muxer,
        }
    }

    /// Adds a stream to the builder.
    pub fn add_stream(&mut self, info: StreamInfo) -> usize {
        let index = self.streams.len();
        #[allow(clippy::cast_possible_truncation)]
        let track_id = (index as u32).wrapping_add(1);
        self.cmaf_muxer.add_track(build_cmaf_track(track_id, &info));
        self.streams.push(info);
        index
    }

    /// Returns the streams.
    #[must_use]
    pub fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    /// Generates the initialization segment.
    ///
    /// # Errors
    ///
    /// Returns `Err` if no streams have been added.
    pub fn build_init_segment(&self) -> OxiResult<Mp4Fragment> {
        if self.streams.is_empty() {
            return Err(OxiError::InvalidData("No streams added".into()));
        }

        let mut fragment = Mp4Fragment::new(FragmentType::Init, 0);
        // Real ftyp+moov (mvhd, mvex/trex, trak per stream with full
        // mdia/minf/stbl), built by the shared CmafMuxer box writer.
        fragment.data = self.cmaf_muxer.write_init_segment();

        Ok(fragment)
    }

    /// Adds a packet to the current fragment.
    ///
    /// Returns `Some(fragment)` if the fragment is complete.
    ///
    /// # Errors
    ///
    /// Returns `Err` if fragment finalization fails.
    pub fn add_packet(&mut self, packet: Packet) -> OxiResult<Option<Mp4Fragment>> {
        // Initialize fragment start time if needed
        if self.fragment_start_time.is_none() {
            self.fragment_start_time = Some(packet.pts());
        }

        self.packets_in_fragment.push(packet);

        // Check if fragment is complete
        if self.should_close_fragment() {
            self.finalize_fragment()
        } else {
            Ok(None)
        }
    }

    /// Checks if the current fragment should be closed.
    fn should_close_fragment(&self) -> bool {
        if self.packets_in_fragment.is_empty() {
            return false;
        }

        // Close on keyframe after minimum duration
        if let Some(last_packet) = self.packets_in_fragment.last() {
            if let Some(start_time) = self.fragment_start_time {
                let duration_ms = (last_packet.pts() - start_time) / 1000;
                #[allow(clippy::cast_sign_loss)]
                {
                    if duration_ms as u64 >= self.config.fragment_duration_ms
                        && last_packet.is_keyframe()
                    {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Finalizes the current fragment.
    fn finalize_fragment(&mut self) -> OxiResult<Option<Mp4Fragment>> {
        if self.packets_in_fragment.is_empty() {
            return Ok(None);
        }

        let sequence = self.config.sequence_number;
        let mut fragment = Mp4Fragment::new(FragmentType::Media, sequence);

        // Calculate duration and collect stream indices
        let start_pts = self
            .fragment_start_time
            .ok_or_else(|| OxiError::InvalidData("No start time".into()))?;
        let end_pts = self
            .packets_in_fragment
            .last()
            .map_or(start_pts, super::super::packet::Packet::pts);

        #[allow(clippy::cast_sign_loss)]
        {
            fragment.duration_us = ((end_pts - start_pts) * 1000) as u64;
        }

        // Set start timestamp
        if let Some(first_packet) = self.packets_in_fragment.first() {
            fragment.start_timestamp = first_packet.timestamp;
        }

        // Collect unique stream indices
        let mut stream_indices: Vec<usize> = self
            .packets_in_fragment
            .iter()
            .map(|p| p.stream_index)
            .collect();
        stream_indices.sort_unstable();
        stream_indices.dedup();
        fragment.stream_indices = stream_indices;

        // Check for keyframe
        fragment.has_keyframe = self
            .packets_in_fragment
            .iter()
            .any(super::super::packet::Packet::is_keyframe);

        // Real moof+mdat: converts the buffered packets into CmafSamples and
        // lets CmafMuxer emit correctly offset-patched trun/traf/moof boxes
        // plus mdat. mfhd.sequence_number is driven by the muxer's own
        // internal counter, which build_init_segment/add_stream seeded from
        // `config.sequence_number` — the `sequence` field below (read BEFORE
        // that counter advances) stays in lock-step with it.
        let cmaf_samples: Vec<MuxCmafSample> = self
            .packets_in_fragment
            .iter()
            .map(build_cmaf_sample)
            .collect();
        fragment.data = self.cmaf_muxer.write_media_segment(&cmaf_samples);

        // Clear state
        self.packets_in_fragment.clear();
        self.fragment_start_time = None;
        self.config.sequence_number += 1;

        Ok(Some(fragment))
    }

    /// Forces finalization of the current fragment.
    ///
    /// # Errors
    ///
    /// Returns `Err` if fragment finalization fails.
    pub fn flush(&mut self) -> OxiResult<Option<Mp4Fragment>> {
        self.finalize_fragment()
    }
}

/// Fragmented MP4 track information.
#[derive(Debug, Clone)]
pub struct FragmentedTrack {
    /// Track ID (1-based).
    pub track_id: u32,
    /// Stream index.
    pub stream_index: usize,
    /// Track information.
    pub stream_info: StreamInfo,
    /// Default sample duration.
    pub default_sample_duration: Option<u32>,
    /// Default sample size.
    pub default_sample_size: Option<u32>,
}

impl FragmentedTrack {
    /// Creates a new fragmented track.
    #[must_use]
    pub const fn new(track_id: u32, stream_index: usize, stream_info: StreamInfo) -> Self {
        Self {
            track_id,
            stream_index,
            stream_info,
            default_sample_duration: None,
            default_sample_size: None,
        }
    }

    /// Sets the default sample duration.
    #[must_use]
    pub const fn with_default_duration(mut self, duration: u32) -> Self {
        self.default_sample_duration = Some(duration);
        self
    }

    /// Sets the default sample size.
    #[must_use]
    pub const fn with_default_size(mut self, size: u32) -> Self {
        self.default_sample_size = Some(size);
        self
    }
}

// ─── Fragmented MP4 live ingest ────────────────────────────────────────────────

/// Parser for fragmented MP4 live ingest streams (moof+mdat).
///
/// Handles incoming fMP4 data from live sources, tracking sequence numbers
/// and validating fragment boundaries.
#[derive(Debug)]
pub struct FragmentedMp4Ingest {
    /// Expected next sequence number.
    expected_sequence: u32,
    /// Total fragments received.
    fragments_received: u64,
    /// Total bytes received.
    bytes_received: u64,
    /// Fragments received out of order.
    out_of_order_count: u64,
    /// Whether we have received an init segment.
    init_received: bool,
}

impl FragmentedMp4Ingest {
    /// Creates a new ingest handler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            expected_sequence: 1,
            fragments_received: 0,
            bytes_received: 0,
            out_of_order_count: 0,
            init_received: false,
        }
    }

    /// Ingests raw fragment data and attempts to parse it.
    ///
    /// Returns the parsed fragment on success.
    ///
    /// # Errors
    ///
    /// Returns `OxiError` if the data is not a valid fragment.
    pub fn ingest(&mut self, data: &[u8]) -> OxiResult<IngestResult> {
        if data.len() < 8 {
            return Err(OxiError::InvalidData("Fragment data too short".into()));
        }

        let box_type = &data[4..8];

        if box_type == b"ftyp" {
            self.init_received = true;
            self.bytes_received += data.len() as u64;
            return Ok(IngestResult::InitSegment);
        }

        if box_type == b"moof" {
            if !self.init_received {
                return Err(OxiError::InvalidData(
                    "Received moof before init segment".into(),
                ));
            }

            // Parse sequence number from moof > mfhd
            let sequence = self
                .parse_mfhd_sequence(data)
                .unwrap_or(self.expected_sequence);

            if sequence != self.expected_sequence {
                self.out_of_order_count += 1;
            }

            self.expected_sequence = sequence + 1;
            self.fragments_received += 1;
            self.bytes_received += data.len() as u64;

            return Ok(IngestResult::MediaFragment { sequence });
        }

        // For mdat and other boxes, just track bytes
        self.bytes_received += data.len() as u64;
        Ok(IngestResult::OtherBox)
    }

    /// Attempts to parse the sequence number from a mfhd box within moof.
    fn parse_mfhd_sequence(&self, data: &[u8]) -> Option<u32> {
        // moof box structure: size(4) + "moof"(4) + children
        // mfhd is first child: size(4) + "mfhd"(4) + version+flags(4) + sequence(4)
        if data.len() < 24 {
            return None;
        }
        // Check for mfhd at offset 8
        if &data[12..16] != b"mfhd" {
            return None;
        }
        // Sequence number at offset 20 (after version+flags)
        let seq = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
        Some(seq)
    }

    /// Returns the total number of fragments received.
    #[must_use]
    pub fn fragments_received(&self) -> u64 {
        self.fragments_received
    }

    /// Returns the total bytes received.
    #[must_use]
    pub fn bytes_received(&self) -> u64 {
        self.bytes_received
    }

    /// Returns the number of out-of-order fragments.
    #[must_use]
    pub fn out_of_order_count(&self) -> u64 {
        self.out_of_order_count
    }

    /// Returns whether an init segment has been received.
    #[must_use]
    pub fn init_received(&self) -> bool {
        self.init_received
    }
}

impl Default for FragmentedMp4Ingest {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of ingesting a fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestResult {
    /// An initialization segment (ftyp+moov).
    InitSegment,
    /// A media fragment (moof+mdat) with its sequence number.
    MediaFragment {
        /// Sequence number from the mfhd box.
        sequence: u32,
    },
    /// Some other MP4 box (mdat, styp, etc.).
    OtherBox,
}

// ─── CMAF chunk generation ────────────────────────────────────────────────────

/// CMAF (Common Media Application Format, ISO/IEC 23000-19) chunk types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmafChunkType {
    /// Regular CMAF chunk (contains one or more complete samples).
    Regular,
    /// Low-latency CMAF chunk (may contain a single sample for LL-DASH/LL-HLS).
    LowLatency,
}

/// Configuration for CMAF chunk generation.
#[derive(Debug, Clone)]
pub struct CmafConfig {
    /// Target chunk duration in milliseconds.
    pub chunk_duration_ms: u64,
    /// Chunk type (regular or low-latency).
    pub chunk_type: CmafChunkType,
    /// Whether to add styp box to each chunk.
    pub add_styp: bool,
    /// Brand for styp box.
    pub brand: String,
    /// Enable chunked transfer encoding hints.
    pub chunked_transfer: bool,
}

impl Default for CmafConfig {
    fn default() -> Self {
        Self {
            chunk_duration_ms: 2000,
            chunk_type: CmafChunkType::Regular,
            add_styp: true,
            brand: "cmfc".into(),
            chunked_transfer: false,
        }
    }
}

impl CmafConfig {
    /// Creates a new CMAF config for regular chunks.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a low-latency CMAF config with short chunks.
    #[must_use]
    pub fn low_latency() -> Self {
        Self {
            chunk_duration_ms: 500,
            chunk_type: CmafChunkType::LowLatency,
            add_styp: true,
            brand: "cmfl".into(),
            chunked_transfer: true,
        }
    }

    /// Sets the chunk duration.
    #[must_use]
    pub fn with_chunk_duration_ms(mut self, ms: u64) -> Self {
        self.chunk_duration_ms = ms;
        self
    }

    /// Sets the chunk type.
    #[must_use]
    pub fn with_chunk_type(mut self, ct: CmafChunkType) -> Self {
        self.chunk_type = ct;
        self
    }

    /// Enables chunked transfer encoding.
    #[must_use]
    pub fn with_chunked_transfer(mut self, enabled: bool) -> Self {
        self.chunked_transfer = enabled;
        self
    }
}

/// A generated CMAF chunk.
#[derive(Debug, Clone)]
pub struct CmafChunk {
    /// Chunk sequence number.
    pub sequence: u32,
    /// Chunk data (styp + moof + mdat or just moof + mdat).
    pub data: Vec<u8>,
    /// Duration of this chunk in microseconds.
    pub duration_us: u64,
    /// Whether this chunk starts with a keyframe.
    pub starts_with_keyframe: bool,
    /// Chunk type.
    pub chunk_type: CmafChunkType,
    /// Whether this chunk is independently decodable.
    pub independent: bool,
}

impl CmafChunk {
    /// Creates a new CMAF chunk.
    #[must_use]
    pub fn new(sequence: u32, chunk_type: CmafChunkType) -> Self {
        Self {
            sequence,
            data: Vec::new(),
            duration_us: 0,
            starts_with_keyframe: false,
            chunk_type,
            independent: false,
        }
    }

    /// Returns the size of the chunk in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns the duration as a `Duration`.
    #[must_use]
    pub fn duration(&self) -> Duration {
        Duration::from_micros(self.duration_us)
    }
}

/// Builder for CMAF chunks from fragmented MP4 fragments.
#[derive(Debug)]
pub struct CmafChunkBuilder {
    config: CmafConfig,
    sequence: u32,
}

impl CmafChunkBuilder {
    /// Creates a new CMAF chunk builder.
    #[must_use]
    pub fn new(config: CmafConfig) -> Self {
        Self {
            config,
            sequence: 1,
        }
    }

    /// Converts an `Mp4Fragment` into one or more CMAF chunks.
    pub fn fragment_to_chunks(&mut self, fragment: &Mp4Fragment) -> Vec<CmafChunk> {
        if fragment.is_init() {
            return Vec::new();
        }

        let mut chunk = CmafChunk::new(self.sequence, self.config.chunk_type);
        chunk.duration_us = fragment.duration_us;
        chunk.starts_with_keyframe = fragment.has_keyframe;
        chunk.independent = fragment.has_keyframe;

        // Build chunk data
        let mut data = Vec::new();

        // Add styp box if configured
        if self.config.add_styp {
            let brand_bytes = self.config.brand.as_bytes();
            let brand = if brand_bytes.len() >= 4 {
                [
                    brand_bytes[0],
                    brand_bytes[1],
                    brand_bytes[2],
                    brand_bytes[3],
                ]
            } else {
                [b'c', b'm', b'f', b'c']
            };
            let styp_size: u32 = 16; // size + "styp" + brand + minor_version
            data.extend_from_slice(&styp_size.to_be_bytes());
            data.extend_from_slice(b"styp");
            data.extend_from_slice(&brand);
            data.extend_from_slice(&0u32.to_be_bytes()); // minor version
        }

        data.extend_from_slice(&fragment.data);
        chunk.data = data;

        self.sequence += 1;
        vec![chunk]
    }

    /// Returns the current sequence number.
    #[must_use]
    pub fn current_sequence(&self) -> u32 {
        self.sequence
    }

    /// Returns the CMAF configuration.
    #[must_use]
    pub fn config(&self) -> &CmafConfig {
        &self.config
    }
}

// ─── Fragment boundary detection ──────────────────────────────────────────────

/// Detects and validates fragment boundaries in a byte stream.
///
/// Scans for moof/mdat box pairs and validates their structural integrity.
#[derive(Debug, Clone)]
pub struct FragmentBoundaryDetector {
    /// Detected fragment boundaries (byte offsets).
    boundaries: Vec<FragmentBoundary>,
}

/// A detected fragment boundary in a byte stream.
#[derive(Debug, Clone)]
pub struct FragmentBoundary {
    /// Byte offset of the moof box start.
    pub moof_offset: u64,
    /// Size of the moof box.
    pub moof_size: u32,
    /// Byte offset of the mdat box start (if found).
    pub mdat_offset: Option<u64>,
    /// Size of the mdat box (if found).
    pub mdat_size: Option<u32>,
    /// Whether this boundary appears structurally valid.
    pub valid: bool,
}

impl FragmentBoundaryDetector {
    /// Creates a new detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            boundaries: Vec::new(),
        }
    }

    /// Scans a byte buffer for fragment boundaries.
    ///
    /// Finds all moof boxes and their associated mdat boxes.
    pub fn scan(&mut self, data: &[u8]) {
        self.boundaries.clear();
        let mut offset = 0u64;

        while (offset as usize) + 8 <= data.len() {
            let pos = offset as usize;
            let size = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            let box_type = &data[pos + 4..pos + 8];

            if box_type == b"moof" && size >= 8 {
                let mut boundary = FragmentBoundary {
                    moof_offset: offset,
                    moof_size: size,
                    mdat_offset: None,
                    mdat_size: None,
                    valid: true,
                };

                // Look for mdat immediately after moof
                let mdat_pos = (offset + u64::from(size)) as usize;
                if mdat_pos + 8 <= data.len() {
                    let mdat_size = u32::from_be_bytes([
                        data[mdat_pos],
                        data[mdat_pos + 1],
                        data[mdat_pos + 2],
                        data[mdat_pos + 3],
                    ]);
                    if &data[mdat_pos + 4..mdat_pos + 8] == b"mdat" {
                        boundary.mdat_offset = Some(offset + u64::from(size));
                        boundary.mdat_size = Some(mdat_size);
                    }
                }

                self.boundaries.push(boundary);
            }

            if size < 8 {
                break;
            }
            offset += u64::from(size);
        }
    }

    /// Returns the detected boundaries.
    #[must_use]
    pub fn boundaries(&self) -> &[FragmentBoundary] {
        &self.boundaries
    }

    /// Returns the number of detected fragments.
    #[must_use]
    pub fn fragment_count(&self) -> usize {
        self.boundaries.len()
    }

    /// Returns `true` if all detected boundaries are structurally valid.
    #[must_use]
    pub fn all_valid(&self) -> bool {
        self.boundaries
            .iter()
            .all(|b| b.valid && b.mdat_offset.is_some())
    }
}

impl Default for FragmentBoundaryDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::Rational;

    #[test]
    fn test_config_default() {
        let config = FragmentedMp4Config::default();
        assert_eq!(config.fragment_duration_ms, 2000);
        assert!(config.separate_init_segment);
        assert!(!config.self_initializing);
        assert_eq!(config.sequence_number, 1);
    }

    #[test]
    fn test_config_builder() {
        let config = FragmentedMp4Config::new()
            .with_fragment_duration(3000)
            .with_separate_init(false)
            .with_self_initializing(true)
            .with_sequence_number(10);

        assert_eq!(config.fragment_duration_ms, 3000);
        assert!(!config.separate_init_segment);
        assert!(config.self_initializing);
        assert_eq!(config.sequence_number, 10);
    }

    #[test]
    fn test_fragment_creation() {
        let fragment = Mp4Fragment::new(FragmentType::Init, 0);
        assert!(fragment.is_init());
        assert!(!fragment.is_media());
        assert_eq!(fragment.sequence, 0);
        assert_eq!(fragment.size(), 0);
    }

    #[test]
    fn test_builder() {
        let config = FragmentedMp4Config::default();
        let mut builder = FragmentedMp4Builder::new(config);

        // This would normally be a real StreamInfo
        let mut stream_info =
            StreamInfo::new(0, oximedia_core::CodecId::Opus, Rational::new(1, 48000));
        stream_info.codec_params = crate::stream::CodecParams::audio(48000, 2);

        let index = builder.add_stream(stream_info);
        assert_eq!(index, 0);
        assert_eq!(builder.streams().len(), 1);
    }

    // ── Real ISO-BMFF box emission tests ─────────────────────────────────────
    //
    // These prove FragmentedMp4Builder no longer writes literal 4-byte ASCII
    // markers (the fixed bug): every fragment must contain fully-formed,
    // nested boxes rather than just the fourcc bytes on their own.

    fn video_stream(index: usize) -> StreamInfo {
        let mut info = StreamInfo::new(index, CodecId::Av1, Rational::new(1, 90000));
        info.codec_params = crate::stream::CodecParams::video(1280, 720);
        info
    }

    fn keyframe_packet(stream_index: usize, pts: i64, payload: u8) -> Packet {
        Packet::new(
            stream_index,
            bytes::Bytes::from(vec![payload; 32]),
            Timestamp::new(pts, Rational::new(1, 90000)),
            crate::PacketFlags::KEYFRAME,
        )
    }

    #[test]
    fn test_build_init_segment_emits_real_boxes_not_a_literal_marker() {
        let mut builder = FragmentedMp4Builder::new(FragmentedMp4Config::default());
        builder.add_stream(video_stream(0));

        let fragment = builder
            .build_init_segment()
            .expect("init segment should build");

        assert!(fragment.is_init());
        // The old bug wrote exactly these 4 bytes and nothing else.
        assert_ne!(
            fragment.data,
            b"ftyp".to_vec(),
            "must not regress to the literal 4-byte marker"
        );
        assert!(fragment.data.len() > 8, "must be a real, sized box tree");
        assert_eq!(
            &fragment.data[4..8],
            b"ftyp",
            "must start with a real ftyp box (4-byte size prefix + fourcc)"
        );
        for fourcc in [
            b"moov", b"mvhd", b"mvex", b"trex", b"trak", b"mdia", b"stbl",
        ] {
            assert!(
                fragment.data.windows(4).any(|w| w == fourcc),
                "init segment must contain a real {:?} box",
                std::str::from_utf8(fourcc).unwrap_or("?")
            );
        }

        // The declared ftyp box size must actually match its content: not
        // just a fourcc dropped into the stream with no real framing.
        let ftyp_size = u32::from_be_bytes([
            fragment.data[0],
            fragment.data[1],
            fragment.data[2],
            fragment.data[3],
        ]) as usize;
        assert!(ftyp_size >= 8 && ftyp_size <= fragment.data.len());
    }

    #[test]
    fn test_build_init_segment_multi_track_has_two_trak_boxes() {
        let mut builder = FragmentedMp4Builder::new(FragmentedMp4Config::default());
        builder.add_stream(video_stream(0));
        let mut audio = StreamInfo::new(1, CodecId::Opus, Rational::new(1, 48000));
        audio.codec_params = crate::stream::CodecParams::audio(48000, 2);
        builder.add_stream(audio);

        let fragment = builder
            .build_init_segment()
            .expect("init segment should build");
        let trak_count = fragment.data.windows(4).filter(|w| *w == b"trak").count();
        assert_eq!(trak_count, 2, "one trak per registered stream");
    }

    #[test]
    fn test_finalize_fragment_emits_real_moof_mdat_not_a_literal_marker() {
        let config = FragmentedMp4Config::new().with_fragment_duration(10);
        let mut builder = FragmentedMp4Builder::new(config);
        builder.add_stream(video_stream(0));

        // First keyframe opens the fragment.
        assert!(builder
            .add_packet(keyframe_packet(0, 0, 0xAA))
            .expect("add_packet ok")
            .is_none());
        // Second keyframe, 50_000 "ticks" later: should_close_fragment divides
        // by 1000 to get an elapsed-ms estimate, so this clears the 10ms
        // configured fragment_duration_ms and closes the fragment.
        let fragment = builder
            .add_packet(keyframe_packet(0, 50_000, 0xBB))
            .expect("add_packet ok")
            .expect("fragment must close");

        assert!(fragment.is_media());
        assert_ne!(
            fragment.data,
            b"moof".to_vec(),
            "must not regress to the literal 4-byte marker"
        );
        assert_eq!(
            &fragment.data[4..8],
            b"moof",
            "must start with a real moof box"
        );
        for fourcc in [b"mfhd", b"traf", b"tfhd", b"tfdt", b"trun", b"mdat"] {
            assert!(
                fragment.data.windows(4).any(|w| w == fourcc),
                "media fragment must contain a real {:?} box",
                std::str::from_utf8(fourcc).unwrap_or("?")
            );
        }

        // Structural validation via the crate's own moof/mdat pairing
        // detector: confirms sizes are self-consistent, not just that the
        // fourcc bytes happen to appear somewhere in the buffer.
        let mut detector = FragmentBoundaryDetector::new();
        detector.scan(&fragment.data);
        assert_eq!(detector.fragment_count(), 1);
        assert!(
            detector.all_valid(),
            "moof/mdat pair must be structurally valid (real sizes, mdat present)"
        );
        let boundary = &detector.boundaries()[0];
        assert_eq!(boundary.moof_offset, 0);
        let mdat_offset = boundary.mdat_offset.expect("mdat must be found");
        assert_eq!(
            mdat_offset,
            u64::from(boundary.moof_size),
            "mdat must immediately follow moof"
        );

        // The sample payload itself must be present in mdat, not dropped.
        assert!(fragment.data.windows(32).any(|w| w == [0xAAu8; 32]));
        assert!(fragment.data.windows(32).any(|w| w == [0xBBu8; 32]));
    }

    #[test]
    fn test_finalize_fragment_sequence_number_matches_configured_start() {
        // A non-default starting sequence number must actually reach the
        // moof's mfhd.sequence_number, not just the Mp4Fragment.sequence
        // bookkeeping field.
        let config = FragmentedMp4Config::new()
            .with_fragment_duration(10)
            .with_sequence_number(7);
        let mut builder = FragmentedMp4Builder::new(config);
        builder.add_stream(video_stream(0));

        builder
            .add_packet(keyframe_packet(0, 0, 1))
            .expect("add_packet ok");
        let fragment = builder
            .add_packet(keyframe_packet(0, 50_000, 2))
            .expect("add_packet ok")
            .expect("fragment must close");

        assert_eq!(
            fragment.sequence, 7,
            "Mp4Fragment.sequence must reflect the configured starting sequence"
        );

        let ingest = FragmentedMp4Ingest::new();
        let mfhd_seq = ingest
            .parse_mfhd_sequence(&fragment.data)
            .expect("mfhd sequence must be parseable from a real moof box");
        assert_eq!(
            mfhd_seq, 7,
            "moof.mfhd.sequence_number must match Mp4Fragment.sequence, proving \
             CmafMuxer's internal counter was seeded correctly rather than \
             left at its own default of 1"
        );
    }

    #[test]
    fn test_finalize_fragment_sequence_number_increments_across_fragments() {
        let config = FragmentedMp4Config::new().with_fragment_duration(10);
        let mut builder = FragmentedMp4Builder::new(config);
        builder.add_stream(video_stream(0));
        let ingest = FragmentedMp4Ingest::new();

        builder.add_packet(keyframe_packet(0, 0, 1)).expect("ok");
        let f1 = builder
            .add_packet(keyframe_packet(0, 50_000, 2))
            .expect("ok")
            .expect("fragment 1 closes");
        assert_eq!(f1.sequence, 1);
        assert_eq!(ingest.parse_mfhd_sequence(&f1.data), Some(1));

        builder
            .add_packet(keyframe_packet(0, 50_000, 3))
            .expect("ok");
        let f2 = builder
            .add_packet(keyframe_packet(0, 100_000, 4))
            .expect("ok")
            .expect("fragment 2 closes");
        assert_eq!(f2.sequence, 2);
        assert_eq!(ingest.parse_mfhd_sequence(&f2.data), Some(2));
    }

    #[test]
    fn test_fragmented_track() {
        let mut stream_info =
            StreamInfo::new(0, oximedia_core::CodecId::Opus, Rational::new(1, 48000));
        stream_info.codec_params = crate::stream::CodecParams::audio(48000, 2);

        let track = FragmentedTrack::new(1, 0, stream_info)
            .with_default_duration(960)
            .with_default_size(100);

        assert_eq!(track.track_id, 1);
        assert_eq!(track.stream_index, 0);
        assert_eq!(track.default_sample_duration, Some(960));
        assert_eq!(track.default_sample_size, Some(100));
    }

    // ── FragmentedMp4Ingest tests ────────────────────────────────────────────

    #[test]
    fn test_ingest_new() {
        let ingest = FragmentedMp4Ingest::new();
        assert!(!ingest.init_received());
        assert_eq!(ingest.fragments_received(), 0);
        assert_eq!(ingest.bytes_received(), 0);
    }

    #[test]
    fn test_ingest_too_short() {
        let mut ingest = FragmentedMp4Ingest::new();
        let result = ingest.ingest(&[0u8; 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ingest_ftyp() {
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(&20u32.to_be_bytes());
        data[4..8].copy_from_slice(b"ftyp");
        data[8..12].copy_from_slice(b"iso5");

        let mut ingest = FragmentedMp4Ingest::new();
        let result = ingest.ingest(&data).expect("ingest ok");
        assert_eq!(result, IngestResult::InitSegment);
        assert!(ingest.init_received());
    }

    #[test]
    fn test_ingest_moof_before_init() {
        let mut data = vec![0u8; 24];
        data[0..4].copy_from_slice(&24u32.to_be_bytes());
        data[4..8].copy_from_slice(b"moof");

        let mut ingest = FragmentedMp4Ingest::new();
        let result = ingest.ingest(&data);
        assert!(result.is_err()); // No init yet
    }

    #[test]
    fn test_ingest_moof_after_init() {
        let mut ingest = FragmentedMp4Ingest::new();

        // Send init first
        let mut ftyp = vec![0u8; 20];
        ftyp[0..4].copy_from_slice(&20u32.to_be_bytes());
        ftyp[4..8].copy_from_slice(b"ftyp");
        ingest.ingest(&ftyp).expect("ftyp ok");

        // Now send moof with mfhd
        let mut moof = vec![0u8; 24];
        moof[0..4].copy_from_slice(&24u32.to_be_bytes());
        moof[4..8].copy_from_slice(b"moof");
        moof[8..12].copy_from_slice(&16u32.to_be_bytes()); // mfhd size
        moof[12..16].copy_from_slice(b"mfhd");
        moof[16..20].copy_from_slice(&0u32.to_be_bytes()); // version+flags
        moof[20..24].copy_from_slice(&1u32.to_be_bytes()); // sequence = 1

        let result = ingest.ingest(&moof).expect("moof ok");
        assert_eq!(result, IngestResult::MediaFragment { sequence: 1 });
        assert_eq!(ingest.fragments_received(), 1);
    }

    #[test]
    fn test_ingest_other_box() {
        let mut ingest = FragmentedMp4Ingest::new();
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(&16u32.to_be_bytes());
        data[4..8].copy_from_slice(b"mdat");

        let result = ingest.ingest(&data).expect("ingest ok");
        assert_eq!(result, IngestResult::OtherBox);
    }

    // ── CMAF tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_cmaf_config_default() {
        let cfg = CmafConfig::default();
        assert_eq!(cfg.chunk_duration_ms, 2000);
        assert_eq!(cfg.chunk_type, CmafChunkType::Regular);
        assert!(cfg.add_styp);
        assert!(!cfg.chunked_transfer);
    }

    #[test]
    fn test_cmaf_config_low_latency() {
        let cfg = CmafConfig::low_latency();
        assert_eq!(cfg.chunk_duration_ms, 500);
        assert_eq!(cfg.chunk_type, CmafChunkType::LowLatency);
        assert!(cfg.chunked_transfer);
    }

    #[test]
    fn test_cmaf_config_builder() {
        let cfg = CmafConfig::new()
            .with_chunk_duration_ms(1000)
            .with_chunk_type(CmafChunkType::LowLatency)
            .with_chunked_transfer(true);
        assert_eq!(cfg.chunk_duration_ms, 1000);
        assert_eq!(cfg.chunk_type, CmafChunkType::LowLatency);
        assert!(cfg.chunked_transfer);
    }

    #[test]
    fn test_cmaf_chunk_new() {
        let chunk = CmafChunk::new(1, CmafChunkType::Regular);
        assert_eq!(chunk.sequence, 1);
        assert_eq!(chunk.size(), 0);
        assert!(!chunk.starts_with_keyframe);
    }

    #[test]
    fn test_cmaf_chunk_builder_from_fragment() {
        let mut fragment = Mp4Fragment::new(FragmentType::Media, 1);
        fragment.data = b"moof_mdat_data".to_vec();
        fragment.duration_us = 2_000_000;
        fragment.has_keyframe = true;

        let cfg = CmafConfig::new();
        let mut builder = CmafChunkBuilder::new(cfg);
        let chunks = builder.fragment_to_chunks(&fragment);

        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].starts_with_keyframe);
        assert_eq!(chunks[0].duration_us, 2_000_000);
        // Should have styp prefix
        assert!(chunks[0].data.len() > fragment.data.len());
        assert_eq!(builder.current_sequence(), 2);
    }

    #[test]
    fn test_cmaf_chunk_builder_skips_init() {
        let fragment = Mp4Fragment::new(FragmentType::Init, 0);
        let cfg = CmafConfig::new();
        let mut builder = CmafChunkBuilder::new(cfg);
        let chunks = builder.fragment_to_chunks(&fragment);
        assert!(chunks.is_empty());
    }

    // ── Fragment boundary detection tests ────────────────────────────────────

    #[test]
    fn test_boundary_detector_empty() {
        let mut detector = FragmentBoundaryDetector::new();
        detector.scan(&[]);
        assert_eq!(detector.fragment_count(), 0);
        assert!(detector.all_valid());
    }

    #[test]
    fn test_boundary_detector_single_moof_mdat() {
        let mut data = Vec::new();
        // moof box
        data.extend_from_slice(&16u32.to_be_bytes());
        data.extend_from_slice(b"moof");
        data.extend_from_slice(&[0u8; 8]); // padding
                                           // mdat box
        data.extend_from_slice(&12u32.to_be_bytes());
        data.extend_from_slice(b"mdat");
        data.extend_from_slice(&[0u8; 4]); // data

        let mut detector = FragmentBoundaryDetector::new();
        detector.scan(&data);

        assert_eq!(detector.fragment_count(), 1);
        let b = &detector.boundaries()[0];
        assert_eq!(b.moof_offset, 0);
        assert_eq!(b.moof_size, 16);
        assert!(b.mdat_offset.is_some());
        assert_eq!(b.mdat_size, Some(12));
        assert!(detector.all_valid());
    }

    #[test]
    fn test_boundary_detector_moof_without_mdat() {
        let mut data = Vec::new();
        // moof box only
        data.extend_from_slice(&16u32.to_be_bytes());
        data.extend_from_slice(b"moof");
        data.extend_from_slice(&[0u8; 8]);

        let mut detector = FragmentBoundaryDetector::new();
        detector.scan(&data);

        assert_eq!(detector.fragment_count(), 1);
        assert!(detector.boundaries()[0].mdat_offset.is_none());
        assert!(!detector.all_valid()); // No mdat → not fully valid
    }

    #[test]
    fn test_boundary_detector_two_fragments() {
        let mut data = Vec::new();
        for _ in 0..2 {
            data.extend_from_slice(&16u32.to_be_bytes());
            data.extend_from_slice(b"moof");
            data.extend_from_slice(&[0u8; 8]);
            data.extend_from_slice(&12u32.to_be_bytes());
            data.extend_from_slice(b"mdat");
            data.extend_from_slice(&[0u8; 4]);
        }

        let mut detector = FragmentBoundaryDetector::new();
        detector.scan(&data);
        assert_eq!(detector.fragment_count(), 2);
        assert!(detector.all_valid());
    }
}
