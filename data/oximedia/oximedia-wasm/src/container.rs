//! Container format definitions and probing.
//!
//! This module provides simplified container handling for WASM,
//! without depending on the full oximedia-container crate (which
//! requires tokio and async runtime).

use bitflags::bitflags;
use bytes::Bytes;
use oximedia_core::{CodecId, OxiError, Rational, Timestamp};

/// Container format types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContainerFormat {
    /// Matroska/`WebM` container
    Matroska,
    /// Ogg container
    Ogg,
    /// FLAC audio
    Flac,
    /// WAV audio
    Wav,
    /// MP4/ISOBMFF container
    Mp4,
}

/// Result of format probing.
#[derive(Clone, Debug)]
pub struct ProbeResult {
    /// Detected container format
    pub format: ContainerFormat,
    /// Confidence score from 0.0 to 1.0
    pub confidence: f32,
}

/// Probe the container format from raw bytes.
pub fn probe_format(data: &[u8]) -> Result<ProbeResult, OxiError> {
    if data.len() < 4 {
        return Err(OxiError::UnknownFormat);
    }

    // Check Matroska/WebM (EBML header)
    if data.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        return Ok(ProbeResult {
            format: ContainerFormat::Matroska,
            confidence: 0.95,
        });
    }

    // Check Ogg
    if data.starts_with(b"OggS") {
        return Ok(ProbeResult {
            format: ContainerFormat::Ogg,
            confidence: 0.99,
        });
    }

    // Check FLAC
    if data.starts_with(b"fLaC") {
        return Ok(ProbeResult {
            format: ContainerFormat::Flac,
            confidence: 0.99,
        });
    }

    // Check WAV (RIFF + WAVE)
    if data.len() >= 12 && data.starts_with(b"RIFF") && &data[8..12] == b"WAVE" {
        return Ok(ProbeResult {
            format: ContainerFormat::Wav,
            confidence: 0.99,
        });
    }

    // Check ISOBMFF/MP4 (ftyp box)
    if data.len() >= 8 && &data[4..8] == b"ftyp" {
        return Ok(ProbeResult {
            format: ContainerFormat::Mp4,
            confidence: 0.90,
        });
    }

    Err(OxiError::UnknownFormat)
}

bitflags! {
    /// Flags indicating packet properties.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct PacketFlags: u32 {
        /// Packet contains a keyframe
        const KEYFRAME = 0x0001;
        /// Packet data may be corrupt
        const CORRUPT = 0x0002;
        /// Packet should be discarded
        const DISCARD = 0x0004;
    }
}

impl Default for PacketFlags {
    fn default() -> Self {
        Self::empty()
    }
}

/// A compressed media packet.
#[derive(Clone, Debug)]
pub struct Packet {
    /// Index of the stream this packet belongs to
    pub stream_index: usize,
    /// Compressed packet data
    pub data: Bytes,
    /// Presentation and decode timestamps
    pub timestamp: Timestamp,
    /// Packet flags
    pub flags: PacketFlags,
}

impl Packet {
    /// Creates a new packet.
    #[must_use]
    pub const fn new(
        stream_index: usize,
        data: Bytes,
        timestamp: Timestamp,
        flags: PacketFlags,
    ) -> Self {
        Self {
            stream_index,
            data,
            timestamp,
            flags,
        }
    }

    /// Returns true if this packet is a keyframe.
    #[must_use]
    pub const fn is_keyframe(&self) -> bool {
        self.flags.contains(PacketFlags::KEYFRAME)
    }

    /// Returns true if this packet may be corrupt.
    #[must_use]
    pub const fn is_corrupt(&self) -> bool {
        self.flags.contains(PacketFlags::CORRUPT)
    }

    /// Returns the size of the packet data in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns the presentation timestamp.
    #[must_use]
    pub const fn pts(&self) -> i64 {
        self.timestamp.pts
    }

    /// Returns the decode timestamp if available.
    #[must_use]
    pub const fn dts(&self) -> Option<i64> {
        self.timestamp.dts
    }

    /// Returns the packet duration if available.
    #[must_use]
    pub const fn duration(&self) -> Option<i64> {
        self.timestamp.duration
    }
}

/// Information about a stream in a container.
#[derive(Clone, Debug)]
pub struct StreamInfo {
    /// Stream index (0-based)
    pub index: usize,
    /// Codec used for this stream
    pub codec: CodecId,
    /// Type of media in this stream
    pub media_type: oximedia_core::MediaType,
    /// Timebase for interpreting timestamps
    pub timebase: Rational,
    /// Duration of the stream in timebase units, if known
    pub duration: Option<i64>,
    /// Codec-specific parameters
    pub codec_params: CodecParams,
    /// Stream metadata
    pub metadata: Metadata,
}

impl StreamInfo {
    /// Creates a new `StreamInfo`.
    #[must_use]
    pub fn new(index: usize, codec: CodecId, timebase: Rational) -> Self {
        Self {
            index,
            codec,
            media_type: codec.media_type(),
            timebase,
            duration: None,
            codec_params: CodecParams::default(),
            metadata: Metadata::default(),
        }
    }

    /// Returns the duration in seconds, if known.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> Option<f64> {
        self.duration.map(|d| {
            if self.timebase.den == 0 {
                0.0
            } else {
                (d as f64 * self.timebase.num as f64) / self.timebase.den as f64
            }
        })
    }

    /// Returns true if this is a video stream.
    #[must_use]
    pub fn is_video(&self) -> bool {
        matches!(self.media_type, oximedia_core::MediaType::Video)
    }

    /// Returns true if this is an audio stream.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        matches!(self.media_type, oximedia_core::MediaType::Audio)
    }
}

/// Codec-specific parameters.
#[derive(Clone, Debug, Default)]
pub struct CodecParams {
    /// Video width in pixels
    pub width: Option<u32>,
    /// Video height in pixels
    pub height: Option<u32>,
    /// Audio sample rate in Hz
    pub sample_rate: Option<u32>,
    /// Number of audio channels
    pub channels: Option<u8>,
    /// Codec-specific extra data
    pub extradata: Option<Bytes>,
}

impl CodecParams {
    /// Creates video codec parameters.
    #[must_use]
    pub const fn video(width: u32, height: u32) -> Self {
        Self {
            width: Some(width),
            height: Some(height),
            sample_rate: None,
            channels: None,
            extradata: None,
        }
    }

    /// Creates audio codec parameters.
    #[must_use]
    pub const fn audio(sample_rate: u32, channels: u8) -> Self {
        Self {
            width: None,
            height: None,
            sample_rate: Some(sample_rate),
            channels: Some(channels),
            extradata: None,
        }
    }

    /// Returns true if video dimensions are set.
    #[must_use]
    pub const fn has_video_params(&self) -> bool {
        self.width.is_some() && self.height.is_some()
    }

    /// Returns true if audio parameters are set.
    #[must_use]
    pub const fn has_audio_params(&self) -> bool {
        self.sample_rate.is_some() && self.channels.is_some()
    }
}

/// Stream and container metadata.
#[derive(Clone, Debug, Default)]
pub struct Metadata {
    /// Stream or container title
    pub title: Option<String>,
    /// Artist or author
    pub artist: Option<String>,
    /// Album name (for audio)
    pub album: Option<String>,
    /// Additional metadata entries
    pub entries: Vec<(String, String)>,
}

impl Metadata {
    /// Creates empty metadata.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            title: None,
            artist: None,
            album: None,
            entries: Vec::new(),
        }
    }

    /// Gets a metadata value by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        let key_lower = key.to_lowercase();
        match key_lower.as_str() {
            "title" => self.title.as_deref(),
            "artist" | "author" => self.artist.as_deref(),
            "album" => self.album.as_deref(),
            _ => self
                .entries
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(key))
                .map(|(_, v)| v.as_str()),
        }
    }

    /// Returns true if the metadata is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.title.is_none()
            && self.artist.is_none()
            && self.album.is_none()
            && self.entries.is_empty()
    }
}
