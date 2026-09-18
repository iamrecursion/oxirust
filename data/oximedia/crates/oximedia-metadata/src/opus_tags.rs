//! Opus-specific metadata support per RFC 7845.
//!
//! This module handles Opus audio metadata, including:
//!
//! - **OpusHead** header parsing (output gain, channel mapping, pre-skip)
//! - **R128 gain tags** (track gain, album gain per R128 standard)
//! - **Output gain** from the Opus header
//! - Integration with Vorbis Comments (OpusTags packet)
//!
//! # RFC 7845 Overview
//!
//! Opus streams in Ogg containers have two mandatory header packets:
//!
//! 1. **OpusHead** (identification header): Contains sample rate, channel count,
//!    pre-skip, output gain, and channel mapping family.
//! 2. **OpusTags** (comment header): Contains Vorbis Comment-style tags with
//!    additional Opus-specific fields like `R128_TRACK_GAIN` and `R128_ALBUM_GAIN`.
//!
//! The R128 gain values are stored as Q7.8 fixed-point integers in units of
//! 1/256 dB. They represent the gain required to normalize the decoded audio
//! to -23 LUFS (EBU R 128 reference level).
//!
//! # Example
//!
//! ```
//! use oximedia_metadata::opus_tags::{OpusHeader, R128Gain};
//!
//! // Parse an Opus header
//! let header = OpusHeader::new()
//!     .with_output_gain_q78(0)  // 0 dB output gain
//!     .with_sample_rate(48000)
//!     .with_channels(2)
//!     .with_pre_skip(312);
//!
//! assert_eq!(header.output_gain_db(), 0.0);
//! assert_eq!(header.sample_rate(), 48000);
//!
//! // Work with R128 gain
//! let r128 = R128Gain::from_db(-3.5);
//! assert!((r128.as_db() - (-3.5)).abs() < 0.01);
//! ```

use crate::{Error, Metadata, MetadataFormat, MetadataValue};

/// Reference loudness for R128 in LUFS.
const R128_REFERENCE_LUFS: f64 = -23.0;

/// Opus identification header (OpusHead) metadata.
///
/// Contains codec configuration and the output gain value that should be
/// applied to decoded audio before playback.
#[derive(Debug, Clone, PartialEq)]
pub struct OpusHeader {
    /// Opus version number (should be 1).
    version: u8,
    /// Number of output channels (1-255).
    channels: u8,
    /// Number of samples to skip at the beginning (encoder delay).
    pre_skip: u16,
    /// Original input sample rate in Hz (informational, Opus always decodes at 48000).
    sample_rate: u32,
    /// Output gain in Q7.8 format (1/256 dB units, signed).
    output_gain_q78: i16,
    /// Channel mapping family (0=mono/stereo, 1=Vorbis order, 255=custom).
    channel_mapping_family: u8,
    /// Stream count (for families > 0).
    stream_count: u8,
    /// Coupled stream count (for families > 0).
    coupled_stream_count: u8,
    /// Channel mapping table (for families > 0).
    channel_mapping: Vec<u8>,
}

impl Default for OpusHeader {
    fn default() -> Self {
        Self {
            version: 1,
            channels: 2,
            pre_skip: 312, // Typical pre-skip for Opus
            sample_rate: 48000,
            output_gain_q78: 0,
            channel_mapping_family: 0,
            stream_count: 1,
            coupled_stream_count: 1,
            channel_mapping: Vec::new(),
        }
    }
}

impl OpusHeader {
    /// Create a new default Opus header (stereo, 48kHz, 0 dB gain).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of channels.
    pub fn with_channels(mut self, channels: u8) -> Self {
        self.channels = channels;
        self
    }

    /// Set the pre-skip in samples.
    pub fn with_pre_skip(mut self, pre_skip: u16) -> Self {
        self.pre_skip = pre_skip;
        self
    }

    /// Set the original input sample rate.
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Set the output gain in Q7.8 format (1/256 dB).
    pub fn with_output_gain_q78(mut self, gain: i16) -> Self {
        self.output_gain_q78 = gain;
        self
    }

    /// Set the output gain from a dB value.
    pub fn with_output_gain_db(mut self, gain_db: f64) -> Self {
        self.output_gain_q78 = db_to_q78(gain_db);
        self
    }

    /// Set the channel mapping family.
    pub fn with_channel_mapping_family(mut self, family: u8) -> Self {
        self.channel_mapping_family = family;
        self
    }

    /// Set the stream count.
    pub fn with_stream_count(mut self, count: u8) -> Self {
        self.stream_count = count;
        self
    }

    /// Set the coupled stream count.
    pub fn with_coupled_stream_count(mut self, count: u8) -> Self {
        self.coupled_stream_count = count;
        self
    }

    /// Set the channel mapping table.
    pub fn with_channel_mapping(mut self, mapping: Vec<u8>) -> Self {
        self.channel_mapping = mapping;
        self
    }

    // ---- Getters ----

    /// Opus version.
    pub fn version(&self) -> u8 {
        self.version
    }

    /// Number of channels.
    pub fn channels(&self) -> u8 {
        self.channels
    }

    /// Pre-skip in samples.
    pub fn pre_skip(&self) -> u16 {
        self.pre_skip
    }

    /// Original input sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Output gain in Q7.8 format.
    pub fn output_gain_q78(&self) -> i16 {
        self.output_gain_q78
    }

    /// Output gain in dB.
    pub fn output_gain_db(&self) -> f64 {
        q78_to_db(self.output_gain_q78)
    }

    /// Channel mapping family.
    pub fn channel_mapping_family(&self) -> u8 {
        self.channel_mapping_family
    }

    /// Stream count.
    pub fn stream_count(&self) -> u8 {
        self.stream_count
    }

    /// Coupled stream count.
    pub fn coupled_stream_count(&self) -> u8 {
        self.coupled_stream_count
    }

    /// Channel mapping table.
    pub fn channel_mapping(&self) -> &[u8] {
        &self.channel_mapping
    }

    /// Pre-skip duration in milliseconds at 48kHz.
    pub fn pre_skip_ms(&self) -> f64 {
        f64::from(self.pre_skip) / 48.0
    }

    /// Whether this is a mono stream.
    pub fn is_mono(&self) -> bool {
        self.channels == 1
    }

    /// Whether this is a stereo stream.
    pub fn is_stereo(&self) -> bool {
        self.channels == 2
    }

    /// Whether this uses Vorbis channel order (family 1).
    pub fn is_vorbis_channel_order(&self) -> bool {
        self.channel_mapping_family == 1
    }

    /// Whether this uses Ambisonics (family 2 or 3).
    pub fn is_ambisonics(&self) -> bool {
        self.channel_mapping_family == 2 || self.channel_mapping_family == 3
    }

    /// Parse an OpusHead packet from raw bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is too short or has an invalid magic signature.
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        // Minimum size: 8 (magic) + 1 (version) + 1 (channels) + 2 (pre-skip) +
        //               4 (sample rate) + 2 (output gain) + 1 (mapping family) = 19
        if data.len() < 19 {
            return Err(Error::ParseError(format!(
                "OpusHead too short: {} bytes (minimum 19)",
                data.len()
            )));
        }

        // Check magic signature "OpusHead"
        if &data[..8] != b"OpusHead" {
            return Err(Error::ParseError(
                "Invalid OpusHead magic signature".to_string(),
            ));
        }

        let version = data[8];
        if version > 15 {
            return Err(Error::ParseError(format!(
                "Unsupported Opus version: {version}"
            )));
        }

        let channels = data[9];
        if channels == 0 {
            return Err(Error::ParseError("Opus channel count is 0".to_string()));
        }

        let pre_skip = u16::from_le_bytes([data[10], data[11]]);
        let sample_rate = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        let output_gain_q78 = i16::from_le_bytes([data[16], data[17]]);
        let channel_mapping_family = data[18];

        let mut header = OpusHeader {
            version,
            channels,
            pre_skip,
            sample_rate,
            output_gain_q78,
            channel_mapping_family,
            stream_count: 1,
            coupled_stream_count: if channels >= 2 { 1 } else { 0 },
            channel_mapping: Vec::new(),
        };

        // Parse channel mapping table for families > 0
        if channel_mapping_family > 0 {
            if data.len() < 21 {
                return Err(Error::ParseError(
                    "OpusHead too short for channel mapping table".to_string(),
                ));
            }
            header.stream_count = data[19];
            header.coupled_stream_count = data[20];

            let mapping_len = channels as usize;
            if data.len() < 21 + mapping_len {
                return Err(Error::ParseError(
                    "OpusHead channel mapping table truncated".to_string(),
                ));
            }
            header.channel_mapping = data[21..21 + mapping_len].to_vec();
        }

        Ok(header)
    }

    /// Serialize to OpusHead packet bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(27 + self.channel_mapping.len());

        // Magic
        buf.extend_from_slice(b"OpusHead");
        // Version
        buf.push(self.version);
        // Channels
        buf.push(self.channels);
        // Pre-skip (LE)
        buf.extend_from_slice(&self.pre_skip.to_le_bytes());
        // Sample rate (LE)
        buf.extend_from_slice(&self.sample_rate.to_le_bytes());
        // Output gain (LE)
        buf.extend_from_slice(&self.output_gain_q78.to_le_bytes());
        // Channel mapping family
        buf.push(self.channel_mapping_family);

        if self.channel_mapping_family > 0 {
            buf.push(self.stream_count);
            buf.push(self.coupled_stream_count);
            buf.extend_from_slice(&self.channel_mapping);
        }

        buf
    }

    /// Write header fields to a `Metadata` container.
    pub fn to_metadata(&self, metadata: &mut Metadata) {
        metadata.insert(
            "opus:version".to_string(),
            MetadataValue::Integer(i64::from(self.version)),
        );
        metadata.insert(
            "opus:channels".to_string(),
            MetadataValue::Integer(i64::from(self.channels)),
        );
        metadata.insert(
            "opus:pre_skip".to_string(),
            MetadataValue::Integer(i64::from(self.pre_skip)),
        );
        metadata.insert(
            "opus:sample_rate".to_string(),
            MetadataValue::Integer(i64::from(self.sample_rate)),
        );
        metadata.insert(
            "opus:output_gain_q78".to_string(),
            MetadataValue::Integer(i64::from(self.output_gain_q78)),
        );
        metadata.insert(
            "opus:output_gain_db".to_string(),
            MetadataValue::Float(self.output_gain_db()),
        );
        metadata.insert(
            "opus:channel_mapping_family".to_string(),
            MetadataValue::Integer(i64::from(self.channel_mapping_family)),
        );
    }
}

/// R128 gain value (Q7.8 fixed-point, in 1/256 dB).
///
/// Per RFC 7845, the R128 gain represents the gain required to normalize
/// audio to -23 LUFS. The value is a signed 16-bit integer where each
/// unit represents 1/256 of a dB.
///
/// Two tags are defined:
/// - `R128_TRACK_GAIN`: per-track normalization gain
/// - `R128_ALBUM_GAIN`: per-album normalization gain
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct R128Gain {
    /// Raw Q7.8 value (1/256 dB units).
    raw: i16,
}

impl R128Gain {
    /// Create from raw Q7.8 value.
    pub fn from_raw(raw: i16) -> Self {
        Self { raw }
    }

    /// Create from a dB value.
    pub fn from_db(db: f64) -> Self {
        Self { raw: db_to_q78(db) }
    }

    /// Get the raw Q7.8 value.
    pub fn raw(&self) -> i16 {
        self.raw
    }

    /// Convert to dB.
    pub fn as_db(&self) -> f64 {
        q78_to_db(self.raw)
    }

    /// Convert to linear gain factor.
    pub fn as_linear(&self) -> f64 {
        10.0f64.powf(self.as_db() / 20.0)
    }

    /// Format as the string used in Vorbis Comment tags (e.g., "-896").
    pub fn to_tag_string(&self) -> String {
        self.raw.to_string()
    }

    /// Parse from a Vorbis Comment tag string (e.g., "-896").
    ///
    /// # Errors
    ///
    /// Returns an error if the string is not a valid integer.
    pub fn from_tag_string(s: &str) -> Result<Self, Error> {
        let raw: i16 = s
            .trim()
            .parse()
            .map_err(|_| Error::ParseError(format!("Invalid R128 gain value: '{s}'")))?;
        Ok(Self { raw })
    }

    /// Reference loudness in LUFS.
    pub fn reference_lufs() -> f64 {
        R128_REFERENCE_LUFS
    }

    /// Check if the gain is zero (no adjustment needed).
    pub fn is_zero(&self) -> bool {
        self.raw == 0
    }

    /// Clamp the gain to the valid range for Opus (-32768 to 32767).
    pub fn clamped(self) -> Self {
        self // Already an i16, inherently clamped
    }
}

impl Default for R128Gain {
    fn default() -> Self {
        Self { raw: 0 }
    }
}

/// Opus metadata container combining header and tag information.
#[derive(Debug, Clone)]
pub struct OpusMetadata {
    /// Opus identification header.
    pub header: OpusHeader,
    /// R128 track gain (from OpusTags).
    pub r128_track_gain: Option<R128Gain>,
    /// R128 album gain (from OpusTags).
    pub r128_album_gain: Option<R128Gain>,
}

impl OpusMetadata {
    /// Create from an OpusHeader.
    pub fn new(header: OpusHeader) -> Self {
        Self {
            header,
            r128_track_gain: None,
            r128_album_gain: None,
        }
    }

    /// Set the R128 track gain.
    pub fn with_track_gain(mut self, gain: R128Gain) -> Self {
        self.r128_track_gain = Some(gain);
        self
    }

    /// Set the R128 album gain.
    pub fn with_album_gain(mut self, gain: R128Gain) -> Self {
        self.r128_album_gain = Some(gain);
        self
    }

    /// Compute the total playback gain in dB.
    ///
    /// Per RFC 7845, the total gain is: output_gain + R128_TRACK_GAIN
    /// (or R128_ALBUM_GAIN when album mode is active).
    pub fn total_track_gain_db(&self) -> f64 {
        let output = self.header.output_gain_db();
        let r128 = self.r128_track_gain.map_or(0.0, |g| g.as_db());
        output + r128
    }

    /// Compute the total album gain in dB.
    pub fn total_album_gain_db(&self) -> Option<f64> {
        self.r128_album_gain
            .map(|g| self.header.output_gain_db() + g.as_db())
    }

    /// Compute the total playback gain as a linear factor.
    pub fn total_track_gain_linear(&self) -> f64 {
        10.0f64.powf(self.total_track_gain_db() / 20.0)
    }

    /// Whether any R128 gain data is present.
    pub fn has_r128_data(&self) -> bool {
        self.r128_track_gain.is_some() || self.r128_album_gain.is_some()
    }

    /// Write all Opus metadata to a `Metadata` container.
    pub fn to_metadata(&self, format: MetadataFormat) -> Metadata {
        let mut metadata = Metadata::new(format);

        self.header.to_metadata(&mut metadata);

        if let Some(gain) = self.r128_track_gain {
            metadata.insert(
                "R128_TRACK_GAIN".to_string(),
                MetadataValue::Text(gain.to_tag_string()),
            );
        }
        if let Some(gain) = self.r128_album_gain {
            metadata.insert(
                "R128_ALBUM_GAIN".to_string(),
                MetadataValue::Text(gain.to_tag_string()),
            );
        }

        metadata
    }

    /// Extract Opus metadata from a `Metadata` container.
    pub fn from_metadata(metadata: &Metadata) -> Self {
        let mut opus = OpusMetadata::new(OpusHeader::new());

        // Extract header fields
        if let Some(ch) = metadata.get("opus:channels").and_then(|v| v.as_integer()) {
            opus.header.channels = ch as u8;
        }
        if let Some(rate) = metadata
            .get("opus:sample_rate")
            .and_then(|v| v.as_integer())
        {
            opus.header.sample_rate = rate as u32;
        }
        if let Some(gain) = metadata
            .get("opus:output_gain_q78")
            .and_then(|v| v.as_integer())
        {
            opus.header.output_gain_q78 = gain as i16;
        }
        if let Some(ps) = metadata.get("opus:pre_skip").and_then(|v| v.as_integer()) {
            opus.header.pre_skip = ps as u16;
        }

        // Extract R128 gains
        if let Some(text) = metadata.get("R128_TRACK_GAIN").and_then(|v| v.as_text()) {
            if let Ok(gain) = R128Gain::from_tag_string(text) {
                opus.r128_track_gain = Some(gain);
            }
        }
        if let Some(text) = metadata.get("R128_ALBUM_GAIN").and_then(|v| v.as_text()) {
            if let Ok(gain) = R128Gain::from_tag_string(text) {
                opus.r128_album_gain = Some(gain);
            }
        }

        opus
    }

    /// Validate the Opus metadata.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.header.channels == 0 {
            issues.push("Channel count is 0".to_string());
        }

        // RFC 7845: version must be in range 0..15
        if self.header.version > 15 {
            issues.push(format!("Unsupported version: {}", self.header.version));
        }

        // Check that mapping family 0 is only used for mono/stereo
        if self.header.channel_mapping_family == 0 && self.header.channels > 2 {
            issues.push(format!(
                "Channel mapping family 0 supports max 2 channels, got {}",
                self.header.channels
            ));
        }

        // Check for extreme R128 gain values
        if let Some(gain) = self.r128_track_gain {
            let db = gain.as_db();
            if db < -128.0 || db > 127.0 {
                issues.push(format!("R128 track gain {db:.2} dB exceeds typical range"));
            }
        }
        if let Some(gain) = self.r128_album_gain {
            let db = gain.as_db();
            if db < -128.0 || db > 127.0 {
                issues.push(format!("R128 album gain {db:.2} dB exceeds typical range"));
            }
        }

        issues
    }
}

// ---- Q7.8 helpers ----

/// Convert Q7.8 fixed-point to dB.
fn q78_to_db(q78: i16) -> f64 {
    f64::from(q78) / 256.0
}

/// Convert dB to Q7.8 fixed-point.
fn db_to_q78(db: f64) -> i16 {
    let raw = (db * 256.0).round();
    // Clamp to i16 range
    if raw > f64::from(i16::MAX) {
        i16::MAX
    } else if raw < f64::from(i16::MIN) {
        i16::MIN
    } else {
        raw as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Q7.8 conversion tests ----

    #[test]
    fn test_q78_to_db_zero() {
        assert!((q78_to_db(0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_q78_to_db_positive() {
        // 256 units = 1.0 dB
        assert!((q78_to_db(256) - 1.0).abs() < f64::EPSILON);
        // 128 units = 0.5 dB
        assert!((q78_to_db(128) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_q78_to_db_negative() {
        // -256 units = -1.0 dB
        assert!((q78_to_db(-256) - (-1.0)).abs() < f64::EPSILON);
        // -896 units = -3.5 dB
        assert!((q78_to_db(-896) - (-3.5)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_db_to_q78_round_trip() {
        let test_values = [-3.5, 0.0, 1.0, -10.25, 6.75];
        for db in &test_values {
            let q78 = db_to_q78(*db);
            let restored = q78_to_db(q78);
            assert!(
                (restored - db).abs() < 0.01,
                "Round-trip failed for {db}: got {restored}"
            );
        }
    }

    #[test]
    fn test_db_to_q78_clamping() {
        // Values that exceed i16 range
        let q78 = db_to_q78(200.0);
        assert_eq!(q78, i16::MAX);
        let q78 = db_to_q78(-200.0);
        assert_eq!(q78, i16::MIN);
    }

    // ---- R128Gain tests ----

    #[test]
    fn test_r128_gain_from_raw() {
        let g = R128Gain::from_raw(-896);
        assert_eq!(g.raw(), -896);
        assert!((g.as_db() - (-3.5)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_r128_gain_from_db() {
        let g = R128Gain::from_db(-3.5);
        assert_eq!(g.raw(), -896);
    }

    #[test]
    fn test_r128_gain_linear() {
        let g = R128Gain::from_db(0.0);
        assert!((g.as_linear() - 1.0).abs() < 1e-10);

        let g2 = R128Gain::from_db(-20.0);
        assert!((g2.as_linear() - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_r128_gain_tag_string_round_trip() {
        let g = R128Gain::from_raw(-896);
        let tag = g.to_tag_string();
        assert_eq!(tag, "-896");

        let parsed = R128Gain::from_tag_string(&tag).expect("should parse");
        assert_eq!(parsed.raw(), -896);
    }

    #[test]
    fn test_r128_gain_tag_string_invalid() {
        let result = R128Gain::from_tag_string("not_a_number");
        assert!(result.is_err());
    }

    #[test]
    fn test_r128_gain_is_zero() {
        assert!(R128Gain::from_raw(0).is_zero());
        assert!(!R128Gain::from_raw(-256).is_zero());
    }

    #[test]
    fn test_r128_gain_reference_lufs() {
        assert!((R128Gain::reference_lufs() - (-23.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_r128_gain_default() {
        let g = R128Gain::default();
        assert_eq!(g.raw(), 0);
        assert!(g.is_zero());
    }

    // ---- OpusHeader tests ----

    #[test]
    fn test_opus_header_default() {
        let h = OpusHeader::new();
        assert_eq!(h.version(), 1);
        assert_eq!(h.channels(), 2);
        assert_eq!(h.pre_skip(), 312);
        assert_eq!(h.sample_rate(), 48000);
        assert_eq!(h.output_gain_q78(), 0);
        assert!((h.output_gain_db() - 0.0).abs() < f64::EPSILON);
        assert_eq!(h.channel_mapping_family(), 0);
        assert!(h.is_stereo());
        assert!(!h.is_mono());
    }

    #[test]
    fn test_opus_header_builders() {
        let h = OpusHeader::new()
            .with_channels(1)
            .with_pre_skip(480)
            .with_sample_rate(44100)
            .with_output_gain_db(-3.5)
            .with_channel_mapping_family(0);

        assert_eq!(h.channels(), 1);
        assert_eq!(h.pre_skip(), 480);
        assert_eq!(h.sample_rate(), 44100);
        assert!((h.output_gain_db() - (-3.5)).abs() < 0.01);
        assert!(h.is_mono());
        assert!(!h.is_stereo());
    }

    #[test]
    fn test_opus_header_pre_skip_ms() {
        let h = OpusHeader::new().with_pre_skip(480);
        assert!((h.pre_skip_ms() - 10.0).abs() < 0.01); // 480/48 = 10ms
    }

    #[test]
    fn test_opus_header_channel_types() {
        let h = OpusHeader::new().with_channel_mapping_family(1);
        assert!(h.is_vorbis_channel_order());
        assert!(!h.is_ambisonics());

        let h2 = OpusHeader::new().with_channel_mapping_family(2);
        assert!(h2.is_ambisonics());
        assert!(!h2.is_vorbis_channel_order());
    }

    #[test]
    fn test_opus_header_parse_valid() {
        let h = OpusHeader::new()
            .with_channels(2)
            .with_pre_skip(312)
            .with_sample_rate(48000)
            .with_output_gain_q78(-256);

        let bytes = h.to_bytes();
        let parsed = OpusHeader::parse(&bytes).expect("should parse");

        assert_eq!(parsed.version(), 1);
        assert_eq!(parsed.channels(), 2);
        assert_eq!(parsed.pre_skip(), 312);
        assert_eq!(parsed.sample_rate(), 48000);
        assert_eq!(parsed.output_gain_q78(), -256);
    }

    #[test]
    fn test_opus_header_parse_mono() {
        let h = OpusHeader::new().with_channels(1);
        let bytes = h.to_bytes();
        let parsed = OpusHeader::parse(&bytes).expect("should parse");
        assert_eq!(parsed.channels(), 1);
    }

    #[test]
    fn test_opus_header_parse_too_short() {
        let result = OpusHeader::parse(&[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_opus_header_parse_bad_magic() {
        let mut data = vec![0u8; 19];
        data[..8].copy_from_slice(b"BadMagic");
        let result = OpusHeader::parse(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_opus_header_parse_zero_channels() {
        let mut h = OpusHeader::new();
        h.channels = 0;
        let mut bytes = h.to_bytes();
        bytes[9] = 0; // channels = 0
        let result = OpusHeader::parse(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_opus_header_with_channel_mapping() {
        let h = OpusHeader::new()
            .with_channels(6)
            .with_channel_mapping_family(1)
            .with_stream_count(4)
            .with_coupled_stream_count(2)
            .with_channel_mapping(vec![0, 4, 1, 2, 3, 5]);

        let bytes = h.to_bytes();
        let parsed = OpusHeader::parse(&bytes).expect("should parse");

        assert_eq!(parsed.channels(), 6);
        assert_eq!(parsed.channel_mapping_family(), 1);
        assert_eq!(parsed.stream_count(), 4);
        assert_eq!(parsed.coupled_stream_count(), 2);
        assert_eq!(parsed.channel_mapping(), &[0, 4, 1, 2, 3, 5]);
    }

    #[test]
    fn test_opus_header_metadata_round_trip() {
        let h = OpusHeader::new()
            .with_channels(2)
            .with_sample_rate(44100)
            .with_output_gain_q78(-512);

        let mut metadata = Metadata::new(MetadataFormat::VorbisComments);
        h.to_metadata(&mut metadata);

        assert_eq!(
            metadata.get("opus:channels").and_then(|v| v.as_integer()),
            Some(2)
        );
        assert_eq!(
            metadata
                .get("opus:sample_rate")
                .and_then(|v| v.as_integer()),
            Some(44100)
        );
        assert_eq!(
            metadata
                .get("opus:output_gain_q78")
                .and_then(|v| v.as_integer()),
            Some(-512)
        );
    }

    // ---- OpusMetadata tests ----

    #[test]
    fn test_opus_metadata_new() {
        let om = OpusMetadata::new(OpusHeader::new());
        assert!(!om.has_r128_data());
    }

    #[test]
    fn test_opus_metadata_with_gains() {
        let om = OpusMetadata::new(OpusHeader::new())
            .with_track_gain(R128Gain::from_db(-3.5))
            .with_album_gain(R128Gain::from_db(-4.0));

        assert!(om.has_r128_data());
        assert!((om.r128_track_gain.expect("track gain").as_db() - (-3.5)).abs() < 0.01);
        assert!((om.r128_album_gain.expect("album gain").as_db() - (-4.0)).abs() < 0.01);
    }

    #[test]
    fn test_opus_metadata_total_track_gain() {
        let om = OpusMetadata::new(OpusHeader::new().with_output_gain_db(-1.0))
            .with_track_gain(R128Gain::from_db(-3.5));

        let total = om.total_track_gain_db();
        assert!((total - (-4.5)).abs() < 0.02);
    }

    #[test]
    fn test_opus_metadata_total_album_gain() {
        let om = OpusMetadata::new(OpusHeader::new().with_output_gain_db(-1.0))
            .with_album_gain(R128Gain::from_db(-5.0));

        let total = om.total_album_gain_db().expect("should have album gain");
        assert!((total - (-6.0)).abs() < 0.02);
    }

    #[test]
    fn test_opus_metadata_total_album_gain_none() {
        let om = OpusMetadata::new(OpusHeader::new());
        assert!(om.total_album_gain_db().is_none());
    }

    #[test]
    fn test_opus_metadata_total_track_gain_linear() {
        let om = OpusMetadata::new(OpusHeader::new()); // 0 dB output, no R128
        let linear = om.total_track_gain_linear();
        assert!((linear - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_opus_metadata_round_trip() {
        let om = OpusMetadata::new(
            OpusHeader::new()
                .with_channels(2)
                .with_sample_rate(48000)
                .with_output_gain_q78(-256),
        )
        .with_track_gain(R128Gain::from_raw(-896))
        .with_album_gain(R128Gain::from_raw(-1024));

        let metadata = om.to_metadata(MetadataFormat::VorbisComments);
        let restored = OpusMetadata::from_metadata(&metadata);

        assert_eq!(restored.header.channels(), 2);
        assert_eq!(restored.header.sample_rate(), 48000);
        assert_eq!(restored.header.output_gain_q78(), -256);
        assert_eq!(restored.r128_track_gain.expect("track gain").raw(), -896);
        assert_eq!(restored.r128_album_gain.expect("album gain").raw(), -1024);
    }

    #[test]
    fn test_opus_metadata_from_empty_metadata() {
        let metadata = Metadata::new(MetadataFormat::VorbisComments);
        let om = OpusMetadata::from_metadata(&metadata);
        assert!(!om.has_r128_data());
    }

    #[test]
    fn test_opus_metadata_validate_ok() {
        let om = OpusMetadata::new(OpusHeader::new()).with_track_gain(R128Gain::from_db(-5.0));
        assert!(om.validate().is_empty());
    }

    #[test]
    fn test_opus_metadata_validate_zero_channels() {
        let mut h = OpusHeader::new();
        h.channels = 0;
        let om = OpusMetadata::new(h);
        let issues = om.validate();
        assert!(!issues.is_empty());
        assert!(issues.iter().any(|i| i.contains("Channel count")));
    }

    #[test]
    fn test_opus_metadata_validate_family0_too_many_channels() {
        let mut h = OpusHeader::new();
        h.channels = 6;
        h.channel_mapping_family = 0;
        let om = OpusMetadata::new(h);
        let issues = om.validate();
        assert!(issues.iter().any(|i| i.contains("family 0")));
    }

    #[test]
    fn test_opus_header_bytes_round_trip_stereo() {
        let original = OpusHeader::new()
            .with_channels(2)
            .with_pre_skip(480)
            .with_sample_rate(44100)
            .with_output_gain_q78(-512);

        let bytes = original.to_bytes();
        let parsed = OpusHeader::parse(&bytes).expect("should parse");

        assert_eq!(parsed.channels(), original.channels());
        assert_eq!(parsed.pre_skip(), original.pre_skip());
        assert_eq!(parsed.sample_rate(), original.sample_rate());
        assert_eq!(parsed.output_gain_q78(), original.output_gain_q78());
    }
}
