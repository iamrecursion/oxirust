//! Type conversions for WASM bindings.
//!
//! This module provides JavaScript-compatible wrappers for `OxiMedia` types.

use crate::container::{CodecParams, Metadata, Packet, StreamInfo};
use oximedia_core::MediaType;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// WASM-compatible stream information.
///
/// Contains information about a stream in a media container.
///
/// # JavaScript Example
///
/// ```javascript
/// const streams = demuxer.streams();
/// for (const stream of streams) {
///     console.log(`Stream ${stream.index()}: ${stream.codec()}`);
///     console.log(`  Duration: ${stream.duration_seconds()} seconds`);
/// }
/// ```
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmStreamInfo {
    inner: StreamInfo,
}

#[wasm_bindgen]
impl WasmStreamInfo {
    /// Returns the stream index (0-based).
    #[must_use]
    pub fn index(&self) -> usize {
        self.inner.index
    }

    /// Returns the codec name.
    ///
    /// Possible values: `"AV1"`, `"VP9"`, `"VP8"`, `"Opus"`, `"Vorbis"`, `"FLAC"`, etc.
    #[must_use]
    pub fn codec(&self) -> String {
        format!("{:?}", self.inner.codec)
    }

    /// Returns the media type.
    ///
    /// Possible values: `"Video"`, `"Audio"`, `"Subtitle"`, `"Data"`.
    #[must_use]
    pub fn media_type(&self) -> String {
        match self.inner.media_type {
            MediaType::Video => "Video".to_string(),
            MediaType::Audio => "Audio".to_string(),
            MediaType::Subtitle => "Subtitle".to_string(),
            MediaType::Data => "Data".to_string(),
            MediaType::Attachment => "Attachment".to_string(),
        }
    }

    /// Returns true if this is a video stream.
    #[must_use]
    pub fn is_video(&self) -> bool {
        self.inner.is_video()
    }

    /// Returns true if this is an audio stream.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        self.inner.is_audio()
    }

    /// Returns the duration in seconds, if known.
    #[must_use]
    pub fn duration_seconds(&self) -> Option<f64> {
        self.inner.duration_seconds()
    }

    /// Returns the timebase numerator.
    #[must_use]
    pub fn timebase_num(&self) -> i64 {
        self.inner.timebase.num
    }

    /// Returns the timebase denominator.
    #[must_use]
    pub fn timebase_den(&self) -> i64 {
        self.inner.timebase.den
    }

    /// Returns codec parameters.
    #[must_use]
    pub fn codec_params(&self) -> WasmCodecParams {
        WasmCodecParams {
            inner: self.inner.codec_params.clone(),
        }
    }

    /// Returns metadata.
    #[must_use]
    pub fn metadata(&self) -> WasmMetadata {
        WasmMetadata {
            inner: self.inner.metadata.clone(),
        }
    }
}

impl From<StreamInfo> for WasmStreamInfo {
    fn from(inner: StreamInfo) -> Self {
        Self { inner }
    }
}

/// WASM-compatible codec parameters.
///
/// Contains format-specific information needed for decoding.
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmCodecParams {
    inner: CodecParams,
}

#[wasm_bindgen]
impl WasmCodecParams {
    /// Returns video width in pixels, if applicable.
    #[must_use]
    pub fn width(&self) -> Option<u32> {
        self.inner.width
    }

    /// Returns video height in pixels, if applicable.
    #[must_use]
    pub fn height(&self) -> Option<u32> {
        self.inner.height
    }

    /// Returns audio sample rate in Hz, if applicable.
    #[must_use]
    pub fn sample_rate(&self) -> Option<u32> {
        self.inner.sample_rate
    }

    /// Returns number of audio channels, if applicable.
    #[must_use]
    pub fn channels(&self) -> Option<u8> {
        self.inner.channels
    }

    /// Returns true if this has video parameters.
    #[must_use]
    pub fn has_video_params(&self) -> bool {
        self.inner.has_video_params()
    }

    /// Returns true if this has audio parameters.
    #[must_use]
    pub fn has_audio_params(&self) -> bool {
        self.inner.has_audio_params()
    }

    /// Returns extradata as a byte array, if present.
    #[must_use]
    pub fn extradata(&self) -> Option<Vec<u8>> {
        self.inner.extradata.as_ref().map(|b| b.to_vec())
    }
}

/// WASM-compatible metadata.
///
/// Contains textual metadata such as title, artist, album.
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmMetadata {
    inner: Metadata,
}

#[wasm_bindgen]
impl WasmMetadata {
    /// Returns the title, if present.
    #[must_use]
    pub fn title(&self) -> Option<String> {
        self.inner.title.clone()
    }

    /// Returns the artist, if present.
    #[must_use]
    pub fn artist(&self) -> Option<String> {
        self.inner.artist.clone()
    }

    /// Returns the album, if present.
    #[must_use]
    pub fn album(&self) -> Option<String> {
        self.inner.album.clone()
    }

    /// Gets a metadata value by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<String> {
        self.inner.get(key).map(|s| s.to_string())
    }

    /// Returns true if the metadata is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// WASM-compatible packet.
///
/// Represents a compressed media packet from a container.
///
/// # JavaScript Example
///
/// ```javascript
/// const packet = await demuxer.read_packet();
/// console.log(`Stream: ${packet.stream_index()}`);
/// console.log(`Size: ${packet.size()} bytes`);
/// console.log(`Keyframe: ${packet.is_keyframe()}`);
/// console.log(`PTS: ${packet.pts()}`);
/// const data = packet.data();
/// ```
#[wasm_bindgen]
pub struct WasmPacket {
    inner: Packet,
}

#[wasm_bindgen]
impl WasmPacket {
    /// Returns the stream index this packet belongs to.
    #[must_use]
    pub fn stream_index(&self) -> usize {
        self.inner.stream_index
    }

    /// Returns the size of the packet data in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.inner.size()
    }

    /// Returns the packet data as a byte array.
    #[must_use]
    pub fn data(&self) -> Vec<u8> {
        self.inner.data.to_vec()
    }

    /// Returns true if this is a keyframe.
    #[must_use]
    pub fn is_keyframe(&self) -> bool {
        self.inner.is_keyframe()
    }

    /// Returns true if the packet may be corrupt.
    #[must_use]
    pub fn is_corrupt(&self) -> bool {
        self.inner.is_corrupt()
    }

    /// Returns the presentation timestamp.
    #[must_use]
    pub fn pts(&self) -> i64 {
        self.inner.pts()
    }

    /// Returns the decode timestamp, if present.
    #[must_use]
    pub fn dts(&self) -> Option<i64> {
        self.inner.dts()
    }

    /// Returns the packet duration, if present.
    #[must_use]
    pub fn duration(&self) -> Option<i64> {
        self.inner.duration()
    }
}

impl From<Packet> for WasmPacket {
    fn from(inner: Packet) -> Self {
        Self { inner }
    }
}

/// Serializable stream information for JavaScript consumers.
///
/// This plain-data struct is serialized to a JavaScript object via `serde_json`
/// and returned as `JsValue` from [`probe_media`](crate::probe::probe_media).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JsStreamInfo {
    /// Stream index (0-based)
    pub index: usize,
    /// Codec name (e.g. `"Av1"`, `"Opus"`)
    pub codec: String,
    /// Media type: `"Video"`, `"Audio"`, `"Subtitle"`, `"Data"`, or `"Attachment"`
    pub media_type: String,
    /// Duration in seconds, if known
    pub duration_seconds: Option<f64>,
    /// Timebase numerator
    pub timebase_num: i64,
    /// Timebase denominator
    pub timebase_den: i64,
    /// Video width in pixels, if applicable
    pub width: Option<u32>,
    /// Video height in pixels, if applicable
    pub height: Option<u32>,
    /// Audio sample rate in Hz, if applicable
    pub sample_rate: Option<u32>,
    /// Number of audio channels, if applicable
    pub channels: Option<u8>,
}

impl From<&WasmStreamInfo> for JsStreamInfo {
    fn from(s: &WasmStreamInfo) -> Self {
        Self {
            index: s.index(),
            codec: s.codec(),
            media_type: s.media_type(),
            duration_seconds: s.duration_seconds(),
            timebase_num: s.timebase_num(),
            timebase_den: s.timebase_den(),
            width: s.codec_params().width(),
            height: s.codec_params().height(),
            sample_rate: s.codec_params().sample_rate(),
            channels: s.codec_params().channels(),
        }
    }
}

/// Complete media information returned by [`probe_media`](crate::probe::probe_media).
///
/// Serialized to a plain JavaScript object so callers can read properties
/// directly without calling wrapper methods.
///
/// # JavaScript Example
///
/// ```javascript
/// import * as oximedia from 'oximedia-wasm';
///
/// const data = new Uint8Array([0x1A, 0x45, 0xDF, 0xA3, 0, 0, 0, 0]);
/// const info = oximedia.probe_media(data);
/// console.log('Format:', info.format);
/// console.log('Confidence:', info.confidence);
/// for (const stream of info.streams) {
///     console.log(`Stream ${stream.index}: ${stream.codec} (${stream.media_type})`);
/// }
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MediaInfo {
    /// Detected container format name (e.g. `"Matroska"`, `"Ogg"`)
    pub format: String,
    /// Human-readable format description
    pub format_description: String,
    /// Detection confidence from 0.0 to 1.0
    pub confidence: f32,
    /// Whether the container supports video
    pub is_video_container: bool,
    /// Whether the container is audio-only
    pub is_audio_only: bool,
    /// Stream information for each stream found
    pub streams: Vec<JsStreamInfo>,
    /// Number of streams
    pub stream_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::PacketFlags;
    use bytes::Bytes;
    use oximedia_core::{CodecId, Rational, Timestamp};

    #[test]
    fn test_stream_info() {
        let stream = StreamInfo::new(0, CodecId::Av1, Rational::new(1, 1000));
        let wasm_stream = WasmStreamInfo::from(stream);

        assert_eq!(wasm_stream.index(), 0);
        assert_eq!(wasm_stream.codec(), "Av1");
        assert_eq!(wasm_stream.media_type(), "Video");
        assert!(wasm_stream.is_video());
        assert!(!wasm_stream.is_audio());
    }

    #[test]
    fn test_codec_params() {
        let params = CodecParams::video(1920, 1080);
        let wasm_params = WasmCodecParams { inner: params };

        assert_eq!(wasm_params.width(), Some(1920));
        assert_eq!(wasm_params.height(), Some(1080));
        assert!(wasm_params.has_video_params());
        assert!(!wasm_params.has_audio_params());
    }

    #[test]
    fn test_metadata() {
        let mut metadata = Metadata::new();
        metadata.title = Some("Test".to_string());
        metadata.artist = Some("Artist".to_string());
        let wasm_metadata = WasmMetadata { inner: metadata };

        assert_eq!(wasm_metadata.title(), Some("Test".to_string()));
        assert_eq!(wasm_metadata.artist(), Some("Artist".to_string()));
        assert!(!wasm_metadata.is_empty());
    }

    #[test]
    fn test_packet() {
        let packet = Packet::new(
            0,
            Bytes::from_static(&[1, 2, 3, 4]),
            Timestamp::new(1000, Rational::new(1, 1000)),
            PacketFlags::KEYFRAME,
        );
        let wasm_packet = WasmPacket::from(packet);

        assert_eq!(wasm_packet.stream_index(), 0);
        assert_eq!(wasm_packet.size(), 4);
        assert!(wasm_packet.is_keyframe());
        assert_eq!(wasm_packet.pts(), 1000);
    }
}
