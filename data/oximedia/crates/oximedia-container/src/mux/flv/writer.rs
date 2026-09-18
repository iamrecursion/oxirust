//! Adobe FLV container muxer.
//!
//! Writes an FLV (Flash Video) byte stream conforming to *Adobe Video File
//! Format Specification Version 10* (2008).  Only patent-free codecs from the
//! Green List are accepted.
//!
//! # Supported codecs
//!
//! | Domain | Codec              | FLV code |
//! |--------|--------------------|----------|
//! | Audio  | MP3                | 2        |
//! | Audio  | PCM (platform)     | 0        |
//! | Audio  | PCM (little-endian)| 3        |
//! | Video  | Sorenson H.263     | 2        |
//!
//! AAC and AVC/H.264 are **explicitly rejected** per workspace patent policy.

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

use async_trait::async_trait;
use oximedia_core::{CodecId, OxiError, OxiResult};
use oximedia_io::MediaSource;

use crate::mux::traits::{Muxer, MuxerConfig};
use crate::{Packet, PacketFlags, StreamInfo};

use super::amf0;

// ============================================================================
// Internal helpers — FLV tag writing
// ============================================================================

/// Tag type byte for audio tags.
const TAG_TYPE_AUDIO: u8 = 8;
/// Tag type byte for video tags.
const TAG_TYPE_VIDEO: u8 = 9;
/// Tag type byte for script (metadata) tags.
const TAG_TYPE_SCRIPT: u8 = 18;

/// FLV file header size (always 9 for version-1 files).
const FLV_HEADER_SIZE: u32 = 9;

/// Build the 4-byte `PreviousTagSize` value for a tag whose body is `data_len`
/// bytes: 11 (fixed tag header) + data_len.
#[inline]
fn prev_tag_size(data_len: usize) -> u32 {
    11u32 + data_len as u32
}

// ============================================================================
// FLV Muxer
// ============================================================================

/// Adobe FLV container muxer.
///
/// # Lifecycle
///
/// 1. Construct with [`FlvMuxer::new`].
/// 2. Register streams via [`Muxer::add_stream`].
/// 3. Call [`Muxer::write_header`] to emit the FLV file header and an
///    `onMetaData` script tag when dimension info is available.
/// 4. Call [`Muxer::write_packet`] for each compressed packet.
/// 5. Call [`Muxer::write_trailer`] (no-op; FLV has no trailer).
pub struct FlvMuxer<S: MediaSource> {
    /// Underlying media sink (writable).
    sink: S,
    /// Muxer configuration.
    config: MuxerConfig,
    /// Registered stream descriptors (indexed by stream index).
    streams: Vec<StreamInfo>,
    /// `true` once the FLV file header has been written.
    header_written: bool,
    /// `PreviousTagSize` value that precedes the *next* tag — reflects the
    /// total byte length (11 + DataSize) of the most recently written tag,
    /// or 0 if no tag has been written yet.
    prev_tag_size: u32,
}

impl<S: MediaSource> FlvMuxer<S> {
    /// Creates a new `FlvMuxer` around the given writable sink.
    ///
    /// # Arguments
    ///
    /// * `sink` – writable media sink (e.g. `MemorySource::new_writable(…)`)
    /// * `config` – muxer metadata and options
    #[must_use]
    pub fn new(sink: S, config: MuxerConfig) -> Self {
        Self {
            sink,
            config,
            streams: Vec::new(),
            header_written: false,
            prev_tag_size: 0,
        }
    }

    /// Returns a reference to the underlying sink.
    #[must_use]
    pub const fn sink(&self) -> &S {
        &self.sink
    }

    /// Returns a mutable reference to the underlying sink.
    pub fn sink_mut(&mut self) -> &mut S {
        &mut self.sink
    }

    /// Consumes the muxer and returns the underlying sink.
    #[must_use]
    pub fn into_sink(self) -> S {
        self.sink
    }
}

impl<S: MediaSource> FlvMuxer<S> {
    // =========================================================================
    // Low-level write helpers
    // =========================================================================

    /// Writes raw bytes to the underlying sink.
    async fn write_bytes(&mut self, data: &[u8]) -> OxiResult<()> {
        self.sink.write_all(data).await
    }

    // =========================================================================
    // FLV tag serialisation
    // =========================================================================

    /// Writes a complete FLV tag followed by its `PreviousTagSize` trailer.
    ///
    /// # Layout
    ///
    /// ```text
    /// [TagType        (1)]
    /// [DataSize       (3)]   ← big-endian, 24-bit
    /// [Timestamp      (3)]   ← low 24 bits of timestamp_ms, big-endian
    /// [TimestampExt   (1)]   ← high 8 bits of timestamp_ms
    /// [StreamID       (3)]   ← always 0
    /// [data         (DataSize bytes)]
    /// [PreviousTagSize (4)]  ← 11 + DataSize, big-endian
    /// ```
    ///
    /// The demuxer reads `PreviousTagSize` *after* the tag, so it comes
    /// last in the byte stream.
    async fn write_tag(&mut self, tag_type: u8, data: &[u8], timestamp_ms: u32) -> OxiResult<()> {
        let data_size = data.len() as u32;

        // Tag type byte.
        self.write_bytes(&[tag_type]).await?;

        // DataSize: 24-bit big-endian.
        self.write_bytes(&[
            ((data_size >> 16) & 0xFF) as u8,
            ((data_size >> 8) & 0xFF) as u8,
            (data_size & 0xFF) as u8,
        ])
        .await?;

        // Timestamp: low 24 bits big-endian, then extended high byte.
        let ts_lo = timestamp_ms & 0x00FF_FFFF;
        let ts_ext = (timestamp_ms >> 24) as u8;
        self.write_bytes(&[
            ((ts_lo >> 16) & 0xFF) as u8,
            ((ts_lo >> 8) & 0xFF) as u8,
            (ts_lo & 0xFF) as u8,
            ts_ext,
        ])
        .await?;

        // StreamID: always 0x00 0x00 0x00.
        self.write_bytes(&[0x00, 0x00, 0x00]).await?;

        // Tag body.
        self.write_bytes(data).await?;

        // PreviousTagSize: 11 (fixed tag header) + DataSize.
        let pts = prev_tag_size(data.len());
        self.write_bytes(&pts.to_be_bytes()).await?;
        self.prev_tag_size = pts;

        Ok(())
    }

    // =========================================================================
    // Audio body construction
    // =========================================================================

    /// Builds the body for an FLV audio tag from a packet.
    ///
    /// Audio tag body layout:
    /// ```text
    /// [AudioHeader (1)] ← (format_code << 4) | (rate_code << 2) | (size_code << 1) | stereo
    /// [payload …]
    /// ```
    fn build_audio_body(stream: &StreamInfo, data: &[u8]) -> OxiResult<Vec<u8>> {
        // Determine FLV sound-format code.
        let format_code: u8 = match stream.codec {
            CodecId::Mp3 => 2,
            CodecId::Pcm => 0, // Platform-endian PCM (FLV code 0)
            _ => {
                return Err(OxiError::unsupported(format!(
                    "FLV muxer: audio codec {:?} is not supported (patent policy or unimplemented)",
                    stream.codec
                )));
            }
        };

        // Rate code: 0=5.5 kHz, 1=11 kHz, 2=22 kHz, 3=44.1 kHz.
        let rate_hz = stream.codec_params.sample_rate.unwrap_or(44_100);
        let rate_code: u8 = if rate_hz <= 5_512 {
            0
        } else if rate_hz <= 11_025 {
            1
        } else if rate_hz <= 22_050 {
            2
        } else {
            3
        };

        // Size code: 0=8-bit, 1=16-bit.
        // CodecParams does not carry bits_per_sample; default to 16-bit.
        let size_code: u8 = 1;

        // Stereo flag: 0=mono, 1=stereo.
        let channels = stream.codec_params.channels.unwrap_or(1);
        let stereo: u8 = if channels >= 2 { 1 } else { 0 };

        let header = (format_code << 4) | (rate_code << 2) | (size_code << 1) | stereo;

        let mut body = Vec::with_capacity(1 + data.len());
        body.push(header);
        body.extend_from_slice(data);
        Ok(body)
    }

    // =========================================================================
    // Video body construction
    // =========================================================================

    /// Builds the body for an FLV video tag from a packet.
    ///
    /// Video tag body layout:
    /// ```text
    /// [VideoHeader (1)] ← (frame_type << 4) | codec_code
    /// [payload …]
    /// ```
    fn build_video_body(
        stream: &StreamInfo,
        data: &[u8],
        flags: PacketFlags,
    ) -> OxiResult<Vec<u8>> {
        // Reject AVC/H.264 explicitly with a clear patent-policy message.
        // (AVC is not in CodecId, but we guard any video codec that would map
        //  to FLV codec code 7.)
        let codec_code: u8 = match stream.codec {
            CodecId::H263 => 2, // Sorenson H.263 — patents expired 2019
            other => {
                return Err(OxiError::unsupported(format!(
                    "FLV muxer: video codec {other:?} is not supported \
                     (H.264/AVC and other patent-encumbered codecs are excluded by policy)"
                )));
            }
        };

        // Frame type: 1=keyframe, 2=inter.
        let frame_type: u8 = if flags.contains(PacketFlags::KEYFRAME) {
            1
        } else {
            2
        };

        let header = (frame_type << 4) | codec_code;

        let mut body = Vec::with_capacity(1 + data.len());
        body.push(header);
        body.extend_from_slice(data);
        Ok(body)
    }

    // =========================================================================
    // onMetaData script tag
    // =========================================================================

    /// Writes an `onMetaData` script tag derived from the registered streams.
    async fn write_on_metadata(&mut self) -> OxiResult<()> {
        let has_video = self.streams.iter().any(|s| s.is_video());
        let has_audio = self.streams.iter().any(|s| s.is_audio());

        // Collect video dimensions from the first video stream, if present.
        let (width, height) = self
            .streams
            .iter()
            .find(|s| s.is_video())
            .map(|s| {
                (
                    s.codec_params.width.unwrap_or(0) as f64,
                    s.codec_params.height.unwrap_or(0) as f64,
                )
            })
            .unwrap_or((0.0, 0.0));

        let metadata_body =
            amf0::write_on_metadata(0.0, width, height, 0.0, 0.0, has_video, has_audio);

        self.write_tag(TAG_TYPE_SCRIPT, &metadata_body, 0).await
    }

    // =========================================================================
    // Timestamp conversion
    // =========================================================================

    /// Converts a packet PTS (in the stream's timebase) to milliseconds.
    ///
    /// FLV stores timestamps as 32-bit unsigned millisecond values.
    /// Negative PTS values are clamped to 0.
    fn pts_to_ms(packet: &Packet, stream: &StreamInfo) -> u32 {
        let pts = packet.pts();
        if pts <= 0 {
            return 0;
        }

        // pts * (num / den) * 1000 → ms
        // Use i128 to avoid overflow for large timestamps.
        let num = stream.timebase.num as i128;
        let den = stream.timebase.den as i128;

        if den == 0 {
            return 0;
        }

        let ms = pts as i128 * num * 1000 / den;
        ms.clamp(0, u32::MAX as i128) as u32
    }
}

// ============================================================================
// Muxer trait implementation
// ============================================================================

#[async_trait]
impl<S: MediaSource> Muxer for FlvMuxer<S> {
    fn add_stream(&mut self, info: StreamInfo) -> OxiResult<usize> {
        if self.header_written {
            return Err(OxiError::InvalidData(
                "Cannot add stream after header is written".into(),
            ));
        }

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

        // ── FLV file header (9 bytes) ────────────────────────────────────────
        // Signature.
        self.write_bytes(b"FLV").await?;
        // Version = 1.
        self.write_bytes(&[1u8]).await?;

        // Flags: bit 2 = has_audio, bit 0 = has_video.
        let has_audio = self.streams.iter().any(|s| s.is_audio());
        let has_video = self.streams.iter().any(|s| s.is_video());
        let flags: u8 =
            (if has_audio { 0x04 } else { 0x00 }) | (if has_video { 0x01 } else { 0x00 });
        self.write_bytes(&[flags]).await?;

        // DataOffset (4 bytes BE) = 9.
        self.write_bytes(&FLV_HEADER_SIZE.to_be_bytes()).await?;

        // PreviousTagSize0 = 0.
        self.write_bytes(&0u32.to_be_bytes()).await?;

        self.header_written = true;

        // ── onMetaData script tag ────────────────────────────────────────────
        self.write_on_metadata().await
    }

    async fn write_packet(&mut self, packet: &Packet) -> OxiResult<()> {
        if !self.header_written {
            return Err(OxiError::InvalidData("Header not written".into()));
        }

        let stream_idx = packet.stream_index;
        if stream_idx >= self.streams.len() {
            return Err(OxiError::InvalidData(format!(
                "Invalid stream index: {stream_idx}"
            )));
        }

        let stream = self.streams[stream_idx].clone();
        let timestamp_ms = Self::pts_to_ms(packet, &stream);

        if stream.is_audio() {
            let body = Self::build_audio_body(&stream, &packet.data)?;
            self.write_tag(TAG_TYPE_AUDIO, &body, timestamp_ms).await
        } else if stream.is_video() {
            let body = Self::build_video_body(&stream, &packet.data, packet.flags)?;
            self.write_tag(TAG_TYPE_VIDEO, &body, timestamp_ms).await
        } else {
            // Subtitle / attachment / data streams are not muxable into FLV.
            Err(OxiError::unsupported(format!(
                "FLV muxer: stream type {:?} is not supported",
                stream.media_type
            )))
        }
    }

    async fn write_trailer(&mut self) -> OxiResult<()> {
        // FLV has no trailer.  The last PreviousTagSize will be written
        // at the start of the *next* tag; for a streaming producer that
        // may suffice.  A seek-fixup pass is not required by the spec.
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
    use crate::demux::flv::{FlvDemuxer, FlvTag};
    use bytes::Bytes;
    use oximedia_core::{Rational, Timestamp};
    use oximedia_io::MemorySource;
    use std::io::Cursor;

    // ── Helpers ─────────────────────────────────────────────────────────────

    fn mp3_stream() -> StreamInfo {
        let mut s = StreamInfo::new(0, CodecId::Mp3, Rational::new(1, 1000));
        s.codec_params.sample_rate = Some(44_100);
        s.codec_params.channels = Some(2);
        s
    }

    fn h263_stream() -> StreamInfo {
        let mut s = StreamInfo::new(1, CodecId::H263, Rational::new(1, 1000));
        s.codec_params.width = Some(320);
        s.codec_params.height = Some(240);
        s
    }

    fn audio_packet(stream_index: usize, pts_ms: i64, data: &[u8]) -> Packet {
        Packet::new(
            stream_index,
            Bytes::copy_from_slice(data),
            Timestamp::new(pts_ms, Rational::new(1, 1000)),
            PacketFlags::KEYFRAME,
        )
    }

    fn video_keyframe(stream_index: usize, pts_ms: i64, data: &[u8]) -> Packet {
        Packet::new(
            stream_index,
            Bytes::copy_from_slice(data),
            Timestamp::new(pts_ms, Rational::new(1, 1000)),
            PacketFlags::KEYFRAME,
        )
    }

    fn video_inter(stream_index: usize, pts_ms: i64, data: &[u8]) -> Packet {
        Packet::new(
            stream_index,
            Bytes::copy_from_slice(data),
            Timestamp::new(pts_ms, Rational::new(1, 1000)),
            PacketFlags::empty(),
        )
    }

    // ── Test 1: FLV header magic ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_flv_header_magic() {
        let sink = MemorySource::new_writable(65536);
        let mut muxer = FlvMuxer::new(sink, MuxerConfig::new());
        muxer.add_stream(mp3_stream()).expect("add stream");
        muxer.write_header().await.expect("write_header");
        let bytes = muxer.into_sink().written_data().to_vec();

        assert!(bytes.len() >= 4, "output too short");
        assert_eq!(&bytes[..3], b"FLV", "FLV signature mismatch");
        assert_eq!(bytes[3], 1u8, "FLV version must be 1");
    }

    // ── Test 2: audio round-trip ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_flv_audio_round_trip() {
        let sink = MemorySource::new_writable(65536);
        let mut muxer = FlvMuxer::new(sink, MuxerConfig::new());
        muxer.add_stream(mp3_stream()).expect("add stream");
        muxer.write_header().await.expect("write_header");
        for i in 0u8..3 {
            let pkt = audio_packet(0, i64::from(i) * 100, &[0xAA, i]);
            muxer.write_packet(&pkt).await.expect("write_packet");
        }
        muxer.write_trailer().await.expect("write_trailer");
        let bytes = muxer.into_sink().written_data().to_vec();

        let mut demux = FlvDemuxer::new(Cursor::new(bytes));
        let hdr = demux.read_header().expect("read_header");
        assert!(hdr.has_audio);

        // First tag is onMetaData script tag, then 3 audio tags.
        let t0 = demux.next_tag().expect("tag 0").expect("some");
        assert!(matches!(t0, FlvTag::Script(_)), "expected onMetaData");

        let mut audio_count = 0u32;
        let mut timestamps = Vec::new();
        while let Some(tag) = demux.next_tag().expect("next") {
            match tag {
                FlvTag::Audio(a) => {
                    timestamps.push(a.timestamp_ms);
                    audio_count += 1;
                }
                other => panic!("unexpected tag: {other:?}"),
            }
        }
        assert_eq!(audio_count, 3, "expected 3 audio tags");
        assert_eq!(timestamps, vec![0, 100, 200]);
    }

    // ── Test 3: video round-trip ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_flv_video_round_trip() {
        let sink = MemorySource::new_writable(65536);
        let mut muxer = FlvMuxer::new(sink, MuxerConfig::new());
        muxer.add_stream(h263_stream()).expect("add stream");
        muxer.write_header().await.expect("write_header");

        let kf = video_keyframe(0, 0, &[0x01, 0x02]);
        muxer.write_packet(&kf).await.expect("keyframe");

        let ip = video_inter(0, 40, &[0x03, 0x04]);
        muxer.write_packet(&ip).await.expect("inter");

        muxer.write_trailer().await.expect("trailer");
        let bytes = muxer.into_sink().written_data().to_vec();

        let mut demux = FlvDemuxer::new(Cursor::new(bytes));
        demux.read_header().expect("header");

        // Skip onMetaData.
        let _ = demux.next_tag().expect("meta").expect("some");

        let t1 = demux.next_tag().expect("v1").expect("some");
        let t2 = demux.next_tag().expect("v2").expect("some");

        match (&t1, &t2) {
            (FlvTag::Video(v1), FlvTag::Video(v2)) => {
                use crate::demux::flv::FrameType;
                assert_eq!(v1.frame_type, FrameType::Key, "first must be keyframe");
                assert_eq!(v2.frame_type, FrameType::Inter, "second must be inter");
                assert_eq!(v1.timestamp_ms, 0);
                assert_eq!(v2.timestamp_ms, 40);
                // Payload: original data without the 1-byte FLV video header.
                assert_eq!(v1.payload, &[0x01, 0x02]);
                assert_eq!(v2.payload, &[0x03, 0x04]);
            }
            _ => panic!("expected two video tags"),
        }
    }

    // ── Test 4: mixed round-trip ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_flv_mixed_round_trip() {
        // mp3_stream has index=0, h263_stream has index=1 in StreamInfo::new,
        // but FlvMuxer assigns indices based on add_stream order.
        // We therefore add them with matching stream_index values.
        let mut audio_stream = mp3_stream();
        audio_stream.index = 0;
        let mut video_stream = h263_stream();
        video_stream.index = 1;

        let sink = MemorySource::new_writable(65536);
        let mut muxer = FlvMuxer::new(sink, MuxerConfig::new());
        muxer.add_stream(audio_stream).expect("audio");
        muxer.add_stream(video_stream).expect("video");
        muxer.write_header().await.expect("header");

        muxer
            .write_packet(&audio_packet(0, 0, &[0xAA]))
            .await
            .expect("a0");
        muxer
            .write_packet(&video_keyframe(1, 0, &[0xBB]))
            .await
            .expect("v0");
        muxer
            .write_packet(&audio_packet(0, 40, &[0xCC]))
            .await
            .expect("a1");
        muxer
            .write_packet(&video_inter(1, 40, &[0xDD]))
            .await
            .expect("v1");

        muxer.write_trailer().await.expect("trailer");
        let bytes = muxer.into_sink().written_data().to_vec();

        let mut demux = FlvDemuxer::new(Cursor::new(bytes));
        let hdr = demux.read_header().expect("header");
        assert!(hdr.has_audio);
        assert!(hdr.has_video);

        // Collect all tags.
        let mut tags = Vec::new();
        while let Some(tag) = demux.next_tag().expect("next") {
            tags.push(tag);
        }

        // 1 script + 2 audio + 2 video = 5.
        assert_eq!(tags.len(), 5);
        assert!(matches!(tags[0], FlvTag::Script(_)));
        assert!(matches!(tags[1], FlvTag::Audio(_)));
        assert!(matches!(tags[2], FlvTag::Video(_)));
        assert!(matches!(tags[3], FlvTag::Audio(_)));
        assert!(matches!(tags[4], FlvTag::Video(_)));
    }

    // ── Test 5: extended timestamp ───────────────────────────────────────────

    #[tokio::test]
    async fn test_timestamp_extended() {
        // A timestamp just above 0x00FF_FFFF (16 777 215 ms ≈ 4.7 hours).
        let big_ts: i64 = 0x0100_0000; // = 16 777 216 ms

        let sink = MemorySource::new_writable(65536);
        let mut muxer = FlvMuxer::new(sink, MuxerConfig::new());
        muxer.add_stream(mp3_stream()).expect("stream");
        muxer.write_header().await.expect("header");
        let pkt = audio_packet(0, big_ts, &[0xBB]);
        muxer.write_packet(&pkt).await.expect("packet");
        muxer.write_trailer().await.expect("trailer");
        let bytes = muxer.into_sink().written_data().to_vec();

        let mut demux = FlvDemuxer::new(Cursor::new(bytes));
        demux.read_header().expect("header");
        // Skip onMetaData.
        let _ = demux.next_tag().expect("meta").expect("some");

        let tag = demux.next_tag().expect("tag").expect("some");
        assert_eq!(tag.timestamp_ms(), big_ts as u32);
    }

    // ── Test 6: probe FLV ────────────────────────────────────────────────────

    #[test]
    fn test_probe_flv() {
        use crate::{probe_format, ContainerFormat};

        // Hand-craft the 9-byte FLV header + PreviousTagSize0 = minimal valid FLV.
        let mut v = Vec::new();
        v.extend_from_slice(b"FLV");
        v.push(1u8); // version
        v.push(0x04u8); // has_audio flag
        v.extend_from_slice(&9u32.to_be_bytes()); // DataOffset
        v.extend_from_slice(&0u32.to_be_bytes()); // PreviousTagSize0

        let result = probe_format(&v).expect("probe");
        assert_eq!(result.format, ContainerFormat::Flv);
    }

    // ── Test 7: AVC/unsupported codec rejected ───────────────────────────────

    #[tokio::test]
    async fn test_unsupported_video_codec_rejected() {
        // AV1 is the test stand-in for "a video codec not in FLV's Green List".
        let mut s = StreamInfo::new(0, CodecId::Av1, Rational::new(1, 1000));
        s.codec_params.width = Some(320);
        s.codec_params.height = Some(240);

        let sink = MemorySource::new_writable(4096);
        let mut muxer = FlvMuxer::new(sink, MuxerConfig::new());
        muxer.add_stream(s).expect("add stream");
        muxer.write_header().await.expect("header");

        let pkt = video_keyframe(0, 0, &[0x00]);
        let result = muxer.write_packet(&pkt).await;
        assert!(result.is_err(), "AV1 must be rejected by FLV muxer");
    }
}
