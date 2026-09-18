//! FLAC muxer implementation.

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

use async_trait::async_trait;
use oximedia_core::{CodecId, OxiError, OxiResult};
use oximedia_io::MediaSource;
use std::io::SeekFrom;

use crate::mux::traits::{Muxer, MuxerConfig};
use crate::{Packet, StreamInfo};

// ============================================================================
// Constants
// ============================================================================

/// FLAC stream marker.
const FLAC_MARKER: &[u8; 4] = b"fLaC";

/// STREAMINFO block type.
const BLOCK_TYPE_STREAMINFO: u8 = 0;

/// PADDING block type.
#[allow(dead_code)]
const BLOCK_TYPE_PADDING: u8 = 1;

/// `VORBIS_COMMENT` block type.
const BLOCK_TYPE_VORBIS_COMMENT: u8 = 4;

/// SEEKTABLE block type.
const BLOCK_TYPE_SEEKTABLE: u8 = 3;

/// Last block flag (OR'd with block type).
const LAST_BLOCK_FLAG: u8 = 0x80;

/// STREAMINFO block size.
const STREAMINFO_SIZE: usize = 34;

/// Default seek table entry count.
const DEFAULT_SEEK_TABLE_SIZE: usize = 100;

/// Seek point placeholder (marks empty entry).
const SEEKPOINT_PLACEHOLDER: u64 = 0xFFFF_FFFF_FFFF_FFFF;

// ============================================================================
// FLAC Stream Info
// ============================================================================

/// FLAC stream information.
///
/// Contains essential parameters for the FLAC stream as defined in the
/// STREAMINFO metadata block specification.
#[derive(Clone, Debug)]
pub struct FlacStreamInfo {
    /// Minimum block size in samples.
    pub min_block_size: u16,

    /// Maximum block size in samples.
    pub max_block_size: u16,

    /// Minimum frame size in bytes (0 means unknown).
    pub min_frame_size: u32,

    /// Maximum frame size in bytes (0 means unknown).
    pub max_frame_size: u32,

    /// Sample rate in Hz.
    pub sample_rate: u32,

    /// Number of channels (1-8).
    pub channels: u8,

    /// Bits per sample (4-32).
    pub bits_per_sample: u8,

    /// Total samples (0 means unknown).
    pub total_samples: u64,

    /// MD5 signature of unencoded audio data.
    ///
    /// Per the FLAC specification, this is the MD5 of the raw (unencoded)
    /// interleaved PCM samples. [`FlacMuxer`] only ever sees already-encoded
    /// FLAC frames (`Packet::data`), so it cannot compute this value itself;
    /// callers that have the original PCM must hash it externally and pass
    /// the result to [`FlacMuxer::set_md5_signature`]. Otherwise this field
    /// stays all-zero, which the spec explicitly sanctions as "MD5 signature
    /// unknown" (§ Stream / `METADATA_BLOCK_STREAMINFO`) — a spec-legal
    /// "unknown" beats a confidently wrong hash of the encoded bytes.
    pub md5_signature: [u8; 16],
}

impl FlacStreamInfo {
    /// Creates a new FLAC stream info with default values.
    #[must_use]
    pub fn new(sample_rate: u32, channels: u8, bits_per_sample: u8) -> Self {
        Self {
            min_block_size: 4096,
            max_block_size: 4096,
            min_frame_size: 0,
            max_frame_size: 0,
            sample_rate,
            channels,
            bits_per_sample,
            total_samples: 0,
            md5_signature: [0u8; 16],
        }
    }

    /// Encodes the STREAMINFO block data.
    ///
    /// Packs all stream parameters into a 34-byte block according to the
    /// FLAC specification.
    #[must_use]
    pub fn encode(&self) -> [u8; STREAMINFO_SIZE] {
        let mut data = [0u8; STREAMINFO_SIZE];

        // Min/max block size (bytes 0-3)
        data[0..2].copy_from_slice(&self.min_block_size.to_be_bytes());
        data[2..4].copy_from_slice(&self.max_block_size.to_be_bytes());

        // Min/max frame size (bytes 4-9, 24-bit each)
        data[4] = (self.min_frame_size >> 16) as u8;
        data[5] = (self.min_frame_size >> 8) as u8;
        data[6] = self.min_frame_size as u8;
        data[7] = (self.max_frame_size >> 16) as u8;
        data[8] = (self.max_frame_size >> 8) as u8;
        data[9] = self.max_frame_size as u8;

        // Sample rate (20 bits), channels (3 bits), bits per sample (5 bits), total samples (36 bits)
        // This spans bytes 10-17
        let sample_rate_20 = self.sample_rate & 0xFFFFF;
        let channels_3 = (self.channels - 1) & 0x7;
        let bps_5 = (self.bits_per_sample - 1) & 0x1F;
        let total_samples_36 = self.total_samples & 0xF_FFFF_FFFF;

        // Pack the bit fields
        data[10] = (sample_rate_20 >> 12) as u8;
        data[11] = (sample_rate_20 >> 4) as u8;
        data[12] =
            ((sample_rate_20 << 4) | (u32::from(channels_3) << 1) | (u32::from(bps_5) >> 4)) as u8;
        data[13] = ((u32::from(bps_5) << 4) | ((total_samples_36 >> 32) as u32)) as u8;
        data[14] = (total_samples_36 >> 24) as u8;
        data[15] = (total_samples_36 >> 16) as u8;
        data[16] = (total_samples_36 >> 8) as u8;
        data[17] = total_samples_36 as u8;

        // MD5 signature (bytes 18-33)
        data[18..34].copy_from_slice(&self.md5_signature);

        data
    }
}

impl Default for FlacStreamInfo {
    fn default() -> Self {
        Self::new(44100, 2, 16)
    }
}

// ============================================================================
// Seek Point
// ============================================================================

/// A seek point in the SEEKTABLE.
///
/// Seek points allow efficient random access to specific samples in the stream.
#[derive(Clone, Copy, Debug)]
pub struct SeekPoint {
    /// Sample number of first sample in target frame.
    pub sample_number: u64,

    /// Offset in bytes from first frame header.
    pub stream_offset: u64,

    /// Number of samples in target frame.
    pub frame_samples: u16,
}

impl SeekPoint {
    /// Creates a new seek point.
    #[must_use]
    pub const fn new(sample_number: u64, stream_offset: u64, frame_samples: u16) -> Self {
        Self {
            sample_number,
            stream_offset,
            frame_samples,
        }
    }

    /// Creates a placeholder seek point.
    ///
    /// Placeholder entries are used to reserve space in the seek table.
    #[must_use]
    pub const fn placeholder() -> Self {
        Self {
            sample_number: SEEKPOINT_PLACEHOLDER,
            stream_offset: 0,
            frame_samples: 0,
        }
    }

    /// Returns true if this is a placeholder.
    #[must_use]
    #[allow(dead_code)]
    pub const fn is_placeholder(&self) -> bool {
        self.sample_number == SEEKPOINT_PLACEHOLDER
    }

    /// Encodes the seek point to bytes.
    #[must_use]
    pub fn encode(&self) -> [u8; 18] {
        let mut data = [0u8; 18];
        data[0..8].copy_from_slice(&self.sample_number.to_be_bytes());
        data[8..16].copy_from_slice(&self.stream_offset.to_be_bytes());
        data[16..18].copy_from_slice(&self.frame_samples.to_be_bytes());
        data
    }
}

// ============================================================================
// FLAC Muxer
// ============================================================================

/// FLAC native container muxer.
///
/// Creates native FLAC files from FLAC-encoded audio packets.
///
/// # MD5 Signature
///
/// The FLAC specification requires an MD5 signature of the unencoded (PCM) audio data
/// in the STREAMINFO block. This muxer only has access to encoded FLAC frames
/// (`Packet::data`), never the raw PCM samples that produced them, so it cannot compute
/// a spec-correct MD5 on its own. Hashing the *encoded* bytes instead — which earlier
/// revisions of this muxer did — produces a value that decoders would treat as a
/// legitimate MD5 of the PCM, silently failing any integrity check that relies on it.
/// That is worse than not signing at all, so this muxer:
///
/// - Writes the all-zero MD5 (the spec-sanctioned "unknown"/unsigned sentinel) by
///   default.
/// - Uses whatever value was passed to [`FlacMuxer::set_md5_signature`] if the caller
///   computed the real MD5 of the raw PCM externally (e.g. before encoding) and
///   supplied it before [`write_trailer`](Muxer::write_trailer) runs.
///
/// A future enhancement could compute this automatically by decoding each FLAC frame
/// back to PCM and hashing that, if/when a FLAC decoder becomes available at this layer.
///
/// # Example
///
/// ```ignore
/// use oximedia_container::mux::{FlacMuxer, Muxer, MuxerConfig};
///
/// let config = MuxerConfig::new()
///     .with_title("My Song")
///     .with_muxing_app("OxiMedia");
///
/// let mut muxer = FlacMuxer::new(sink, config);
/// muxer.add_stream(audio_info)?;
/// muxer.write_header().await?;
///
/// for packet in packets {
///     muxer.write_packet(&packet).await?;
/// }
///
/// muxer.write_trailer().await?;
/// ```
pub struct FlacMuxer<W> {
    /// The underlying writer.
    sink: W,

    /// Muxer configuration.
    config: MuxerConfig,

    /// Registered streams (FLAC only supports one stream).
    streams: Vec<StreamInfo>,

    /// FLAC stream info.
    stream_info: Option<FlacStreamInfo>,

    /// Whether header has been written.
    header_written: bool,

    /// Current write position.
    position: u64,

    /// Position of STREAMINFO block (for fixup).
    streaminfo_position: u64,

    /// Position of SEEKTABLE block (for fixup).
    seektable_position: Option<u64>,

    /// Seek points.
    seek_points: Vec<SeekPoint>,

    /// First frame position (after metadata).
    first_frame_position: u64,

    /// Total samples written.
    total_samples: u64,

    /// Minimum frame size seen.
    min_frame_size: u32,

    /// Maximum frame size seen.
    max_frame_size: u32,

    /// Whether [`set_md5_signature`](Self::set_md5_signature) was called with
    /// a real (externally-computed) PCM MD5. When `false`, `write_trailer`
    /// writes the all-zero "unknown" MD5 rather than fabricating one from
    /// encoded frame bytes.
    md5_explicit: bool,
}

impl<W> FlacMuxer<W> {
    /// Creates a new FLAC muxer.
    ///
    /// # Arguments
    ///
    /// * `sink` - The writer to output to
    /// * `config` - Muxer configuration
    #[must_use]
    pub fn new(sink: W, config: MuxerConfig) -> Self {
        Self {
            sink,
            config,
            streams: Vec::new(),
            stream_info: None,
            header_written: false,
            position: 0,
            streaminfo_position: 0,
            seektable_position: None,
            seek_points: Vec::with_capacity(DEFAULT_SEEK_TABLE_SIZE),
            first_frame_position: 0,
            total_samples: 0,
            min_frame_size: 0,
            max_frame_size: 0,
            md5_explicit: false,
        }
    }

    /// Returns a reference to the underlying sink.
    #[must_use]
    pub const fn sink(&self) -> &W {
        &self.sink
    }

    /// Returns a mutable reference to the underlying sink.
    pub fn sink_mut(&mut self) -> &mut W {
        &mut self.sink
    }

    /// Consumes the muxer and returns the underlying sink.
    #[must_use]
    #[allow(dead_code)]
    pub fn into_sink(self) -> W {
        self.sink
    }

    /// Returns the total samples written.
    #[must_use]
    pub const fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Sets the MD5 signature if known from external source.
    ///
    /// This should be called before `write_trailer()` if you have calculated
    /// the MD5 of the raw PCM audio data externally (this muxer never sees
    /// raw PCM itself — only already-encoded FLAC frames — so it cannot
    /// compute the spec-correct MD5 on its own). Calling this marks the
    /// signature as explicit, so `write_trailer()` preserves `md5` verbatim
    /// instead of falling back to the all-zero "unknown" sentinel.
    pub fn set_md5_signature(&mut self, md5: [u8; 16]) {
        if let Some(ref mut info) = self.stream_info {
            info.md5_signature = md5;
        }
        self.md5_explicit = true;
    }
}

impl<W: MediaSource> FlacMuxer<W> {
    /// Writes bytes to the sink.
    async fn write_bytes(&mut self, data: &[u8]) -> OxiResult<()> {
        self.sink.write_all(data).await?;
        self.position += data.len() as u64;
        Ok(())
    }

    /// Builds stream info from a stream.
    fn build_stream_info(stream: &StreamInfo) -> FlacStreamInfo {
        let sample_rate = stream.codec_params.sample_rate.unwrap_or(44100);
        let channels = stream.codec_params.channels.unwrap_or(2);

        FlacStreamInfo::new(sample_rate, channels, 16)
    }

    /// Writes the FLAC stream marker.
    async fn write_marker(&mut self) -> OxiResult<()> {
        self.write_bytes(FLAC_MARKER).await
    }

    /// Writes a metadata block header.
    async fn write_block_header(
        &mut self,
        block_type: u8,
        size: u32,
        is_last: bool,
    ) -> OxiResult<()> {
        let header_byte = if is_last {
            block_type | LAST_BLOCK_FLAG
        } else {
            block_type
        };

        self.write_bytes(&[header_byte]).await?;
        self.write_bytes(&size.to_be_bytes()[1..4]).await // 24-bit size
    }

    /// Writes the STREAMINFO block.
    async fn write_streaminfo(&mut self, is_last: bool) -> OxiResult<()> {
        let stream_info = self
            .stream_info
            .clone()
            .ok_or_else(|| OxiError::InvalidData("Stream info not configured".into()))?;

        self.streaminfo_position = self.position;
        self.write_block_header(BLOCK_TYPE_STREAMINFO, STREAMINFO_SIZE as u32, is_last)
            .await?;
        self.write_bytes(&stream_info.encode()).await
    }

    /// Writes a SEEKTABLE block.
    async fn write_seektable(&mut self, is_last: bool) -> OxiResult<()> {
        // Pre-allocate seek points
        let seek_table_size = DEFAULT_SEEK_TABLE_SIZE * 18;

        self.seektable_position = Some(self.position);
        self.write_block_header(BLOCK_TYPE_SEEKTABLE, seek_table_size as u32, is_last)
            .await?;

        // Write placeholder seek points
        for _ in 0..DEFAULT_SEEK_TABLE_SIZE {
            self.write_bytes(&SeekPoint::placeholder().encode()).await?;
        }

        Ok(())
    }

    /// Writes a `VORBIS_COMMENT` block.
    async fn write_vorbis_comment(&mut self, is_last: bool) -> OxiResult<()> {
        let mut content = Vec::new();

        // Vendor string
        let vendor = self.config.muxing_app.as_deref().unwrap_or("OxiMedia");
        content.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        content.extend_from_slice(vendor.as_bytes());

        // User comments
        let mut comments = Vec::new();
        if let Some(ref title) = self.config.title {
            comments.push(format!("TITLE={title}"));
        }

        content.extend_from_slice(&(comments.len() as u32).to_le_bytes());
        for comment in comments {
            content.extend_from_slice(&(comment.len() as u32).to_le_bytes());
            content.extend_from_slice(comment.as_bytes());
        }

        self.write_block_header(BLOCK_TYPE_VORBIS_COMMENT, content.len() as u32, is_last)
            .await?;
        self.write_bytes(&content).await
    }

    /// Fixes up the STREAMINFO block with final values.
    async fn fixup_streaminfo(&mut self) -> OxiResult<()> {
        if let Some(ref mut stream_info) = self.stream_info {
            stream_info.total_samples = self.total_samples;
            stream_info.min_frame_size = self.min_frame_size;
            stream_info.max_frame_size = self.max_frame_size;

            // Only keep a caller-supplied MD5 of the real PCM data (set via
            // `set_md5_signature`). Otherwise write the spec-sanctioned
            // all-zero "unknown" MD5 rather than a confidently wrong hash of
            // the encoded frame bytes.
            if !self.md5_explicit {
                stream_info.md5_signature = [0u8; 16];
            }

            let data = stream_info.encode();

            // Save current position
            let current_pos = self.position;

            // Seek to STREAMINFO data (skip 4-byte header)
            self.sink
                .seek(SeekFrom::Start(self.streaminfo_position + 4))
                .await?;
            self.sink.write_all(&data).await?;

            // Seek back
            self.sink.seek(SeekFrom::Start(current_pos)).await?;
        }

        Ok(())
    }

    /// Fixes up the SEEKTABLE block with recorded seek points.
    async fn fixup_seektable(&mut self) -> OxiResult<()> {
        if let Some(seektable_pos) = self.seektable_position {
            // Save current position
            let current_pos = self.position;

            // Seek to SEEKTABLE data (skip 4-byte header)
            self.sink.seek(SeekFrom::Start(seektable_pos + 4)).await?;

            // Write seek points (pad with placeholders)
            for i in 0..DEFAULT_SEEK_TABLE_SIZE {
                let point = self
                    .seek_points
                    .get(i)
                    .copied()
                    .unwrap_or_else(SeekPoint::placeholder);
                self.sink.write_all(&point.encode()).await?;
            }

            // Seek back
            self.sink.seek(SeekFrom::Start(current_pos)).await?;
        }

        Ok(())
    }
}

#[async_trait]
impl<W: MediaSource> Muxer for FlacMuxer<W> {
    fn add_stream(&mut self, info: StreamInfo) -> OxiResult<usize> {
        if self.header_written {
            return Err(OxiError::InvalidData(
                "Cannot add stream after header is written".into(),
            ));
        }

        // FLAC only supports one stream
        if !self.streams.is_empty() {
            return Err(OxiError::unsupported(
                "FLAC format only supports a single audio stream",
            ));
        }

        // Validate codec is FLAC
        if info.codec != CodecId::Flac {
            return Err(OxiError::unsupported(format!(
                "FLAC muxer only supports FLAC codec, got {:?}",
                info.codec
            )));
        }

        // Build stream info
        self.stream_info = Some(Self::build_stream_info(&info));

        let index = self.streams.len();
        self.streams.push(info);
        Ok(index)
    }

    async fn write_header(&mut self) -> OxiResult<()> {
        if self.header_written {
            return Err(OxiError::InvalidData("Header already written".into()));
        }

        if self.streams.is_empty() {
            return Err(OxiError::InvalidData("No streams configured".into()));
        }

        // Write FLAC marker
        self.write_marker().await?;

        // Write metadata blocks
        self.write_streaminfo(false).await?;

        if self.config.write_cues {
            self.write_seektable(false).await?;
        }

        self.write_vorbis_comment(true).await?;

        self.first_frame_position = self.position;
        self.header_written = true;
        Ok(())
    }

    async fn write_packet(&mut self, packet: &Packet) -> OxiResult<()> {
        if !self.header_written {
            return Err(OxiError::InvalidData("Header not written".into()));
        }

        if packet.stream_index >= self.streams.len() {
            return Err(OxiError::InvalidData(format!(
                "Invalid stream index: {}",
                packet.stream_index
            )));
        }

        let frame_size = packet.data.len() as u32;

        // Update frame size statistics
        if self.min_frame_size == 0 || frame_size < self.min_frame_size {
            self.min_frame_size = frame_size;
        }
        if frame_size > self.max_frame_size {
            self.max_frame_size = frame_size;
        }

        // Deliberately not hashing `packet.data` into an MD5 here: it is
        // encoded FLAC frame data, not the raw PCM the spec requires. See
        // `set_md5_signature` / the `FlacMuxer` "MD5 Signature" doc section.

        // Record seek point at regular intervals
        if self.config.write_cues {
            let seek_interval = self.stream_info.as_ref().map_or(44100, |s| s.sample_rate);
            let block_size = self
                .stream_info
                .as_ref()
                .map_or(4096, |s| u64::from(s.max_block_size));

            if self.total_samples % u64::from(seek_interval) < block_size {
                let offset = self.position - self.first_frame_position;
                let point = SeekPoint::new(self.total_samples, offset, block_size as u16);
                if self.seek_points.len() < DEFAULT_SEEK_TABLE_SIZE {
                    self.seek_points.push(point);
                }
            }
        }

        // Write frame data
        self.write_bytes(&packet.data).await?;

        // Update sample count (estimate from frame duration)
        #[allow(clippy::cast_sign_loss)]
        if let Some(duration) = packet.duration() {
            self.total_samples += duration as u64;
        } else {
            // Estimate from block size
            let block_size = self
                .stream_info
                .as_ref()
                .map_or(4096, |s| u64::from(s.max_block_size));
            self.total_samples += block_size;
        }

        Ok(())
    }

    async fn write_trailer(&mut self) -> OxiResult<()> {
        if !self.header_written {
            return Err(OxiError::InvalidData("Header not written".into()));
        }

        // Fix up STREAMINFO with final values
        self.fixup_streaminfo().await?;

        // Fix up SEEKTABLE with recorded seek points
        self.fixup_seektable().await?;

        Ok(())
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn config(&self) -> &MuxerConfig {
        &self.config
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use oximedia_core::{Rational, Timestamp};
    use oximedia_io::MemorySource;

    fn create_flac_stream() -> StreamInfo {
        let mut stream = StreamInfo::new(0, CodecId::Flac, Rational::new(1, 44100));
        stream.codec_params.sample_rate = Some(44100);
        stream.codec_params.channels = Some(2);
        stream
    }

    #[test]
    fn test_flac_stream_info_new() {
        let info = FlacStreamInfo::new(48000, 2, 24);
        assert_eq!(info.sample_rate, 48000);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits_per_sample, 24);
    }

    #[test]
    fn test_flac_stream_info_encode() {
        let info = FlacStreamInfo::new(44100, 2, 16);
        let data = info.encode();
        assert_eq!(data.len(), STREAMINFO_SIZE);
    }

    #[test]
    fn test_seek_point_new() {
        let point = SeekPoint::new(48000, 1000, 4096);
        assert_eq!(point.sample_number, 48000);
        assert_eq!(point.stream_offset, 1000);
        assert_eq!(point.frame_samples, 4096);
    }

    #[test]
    fn test_seek_point_placeholder() {
        let point = SeekPoint::placeholder();
        assert!(point.is_placeholder());
    }

    #[test]
    fn test_seek_point_encode() {
        let point = SeekPoint::new(48000, 1000, 4096);
        let data = point.encode();
        assert_eq!(data.len(), 18);
    }

    #[tokio::test]
    async fn test_muxer_new() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let muxer = FlacMuxer::new(sink, config);

        assert!(!muxer.header_written);
        assert!(muxer.streams.is_empty());
    }

    #[tokio::test]
    async fn test_muxer_add_stream() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let mut muxer = FlacMuxer::new(sink, config);

        let flac = create_flac_stream();
        let idx = muxer.add_stream(flac).expect("operation should succeed");

        assert_eq!(idx, 0);
        assert_eq!(muxer.streams.len(), 1);
    }

    #[tokio::test]
    async fn test_muxer_add_multiple_streams_fails() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let mut muxer = FlacMuxer::new(sink, config);

        let flac1 = create_flac_stream();
        let flac2 = create_flac_stream();

        muxer.add_stream(flac1).expect("operation should succeed");
        let result = muxer.add_stream(flac2);

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_muxer_add_non_flac_fails() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let mut muxer = FlacMuxer::new(sink, config);

        let opus = StreamInfo::new(0, CodecId::Opus, Rational::new(1, 48000));
        let result = muxer.add_stream(opus);

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_muxer_write_header() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let mut muxer = FlacMuxer::new(sink, config);

        let flac = create_flac_stream();
        muxer.add_stream(flac).expect("operation should succeed");

        let result = muxer.write_header().await;
        assert!(result.is_ok());
        assert!(muxer.header_written);
    }

    #[tokio::test]
    async fn test_muxer_write_packet() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let mut muxer = FlacMuxer::new(sink, config);

        let flac = create_flac_stream();
        muxer.add_stream(flac).expect("operation should succeed");
        muxer
            .write_header()
            .await
            .expect("operation should succeed");

        // Create a minimal FLAC frame (just test data)
        let packet = Packet::new(
            0,
            Bytes::from(vec![0xFF, 0xF8, 0x00, 0x00]), // FLAC frame sync
            Timestamp::new(0, Rational::new(1, 44100)),
            crate::PacketFlags::KEYFRAME,
        );

        let result = muxer.write_packet(&packet).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_muxer_write_trailer() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let mut muxer = FlacMuxer::new(sink, config);

        let flac = create_flac_stream();
        muxer.add_stream(flac).expect("operation should succeed");
        muxer
            .write_header()
            .await
            .expect("operation should succeed");

        let result = muxer.write_trailer().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_muxer_with_title() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new().with_title("Test Song");
        let mut muxer = FlacMuxer::new(sink, config);

        let flac = create_flac_stream();
        muxer.add_stream(flac).expect("operation should succeed");

        let result = muxer.write_header().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_muxer_md5_zero_without_explicit_pcm_hash() {
        // Without an externally-supplied PCM MD5, the muxer must never
        // fabricate a hash from the encoded frame bytes: FLAC decoders treat
        // the STREAMINFO MD5 as authoritative for integrity checks, so a
        // wrong-but-nonzero value is worse than the spec-sanctioned
        // all-zero "unknown" sentinel.
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let mut muxer = FlacMuxer::new(sink, config);

        let flac = create_flac_stream();
        muxer.add_stream(flac).expect("operation should succeed");
        muxer
            .write_header()
            .await
            .expect("operation should succeed");

        for i in 0..10 {
            let packet = Packet::new(
                0,
                Bytes::from(vec![0xFF, 0xF8, i as u8, i as u8]),
                Timestamp::new(i64::from(i) * 4096, Rational::new(1, 44_100)),
                crate::PacketFlags::KEYFRAME,
            );
            muxer
                .write_packet(&packet)
                .await
                .expect("operation should succeed");
        }

        muxer
            .write_trailer()
            .await
            .expect("operation should succeed");

        let info = muxer
            .stream_info
            .as_ref()
            .expect("stream_info must be set after add_stream");
        assert_eq!(
            info.md5_signature, [0u8; 16],
            "MD5 must stay the all-zero 'unknown' sentinel when no real PCM hash was supplied"
        );

        // The all-zero MD5 must also have reached the actual bytes written
        // to the sink (not just the in-memory struct), at STREAMINFO's
        // fixed MD5 offset: block header (4 bytes) + 18 bytes of preceding
        // STREAMINFO fields.
        let written = muxer.sink().written_data();
        let md5_offset = (muxer.streaminfo_position + 4 + 18) as usize;
        assert_eq!(&written[md5_offset..md5_offset + 16], &[0u8; 16][..]);
    }

    #[tokio::test]
    async fn test_muxer_md5_explicit_signature_preserved_through_trailer() {
        // When the caller supplies a real MD5 of the raw PCM (computed
        // externally, e.g. before encoding), write_trailer must preserve it
        // verbatim, both in memory and in the bytes written to the sink.
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let mut muxer = FlacMuxer::new(sink, config);

        let flac = create_flac_stream();
        muxer.add_stream(flac).expect("operation should succeed");
        muxer
            .write_header()
            .await
            .expect("operation should succeed");

        let real_pcm_md5: [u8; 16] = [
            0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE,
            0xFF, 0x01,
        ];
        muxer.set_md5_signature(real_pcm_md5);

        let packet = Packet::new(
            0,
            Bytes::from(vec![0xFF, 0xF8, 0x00, 0x00]),
            Timestamp::new(0, Rational::new(1, 44_100)),
            crate::PacketFlags::KEYFRAME,
        );
        muxer
            .write_packet(&packet)
            .await
            .expect("operation should succeed");

        muxer
            .write_trailer()
            .await
            .expect("operation should succeed");

        let info = muxer
            .stream_info
            .as_ref()
            .expect("stream_info must be set after add_stream");
        assert_eq!(info.md5_signature, real_pcm_md5);

        let written = muxer.sink().written_data();
        let md5_offset = (muxer.streaminfo_position + 4 + 18) as usize;
        assert_eq!(&written[md5_offset..md5_offset + 16], &real_pcm_md5[..]);
    }
}
