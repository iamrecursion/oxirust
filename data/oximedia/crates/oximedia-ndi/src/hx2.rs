//! NDI|HX2 compressed mode support.
//!
//! NDI|HX2 is a compressed transport variant that uses AV1 as its video codec,
//! providing substantially lower bandwidth than uncompressed or SpeedHQ at the
//! cost of a few frames of encoder latency.
//!
//! This module provides:
//! - [`Hx2Config`] — encoder/decoder configuration
//! - [`Hx2Preset`] — quality/latency trade-off presets
//! - [`Hx2Encoder`] — frame encoder stub with AV1 integration points
//! - [`Hx2Decoder`] — frame decoder stub with AV1 integration points

#![allow(dead_code)]

use crate::av_buffer::NdiVideoFrame;
use crate::{NdiError, Result};

// ---------------------------------------------------------------------------
// Hx2Preset
// ---------------------------------------------------------------------------

/// Quality/latency trade-off presets for NDI|HX2 encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Hx2Preset {
    /// Minimise encode latency at the expense of compression efficiency.
    /// Suitable for live production where frames must arrive with sub-frame
    /// latency.
    LowLatency,
    /// Balanced preset — moderate latency, good compression ratio.
    /// This is the default for most NDI|HX2 deployments.
    Balanced,
    /// Maximise compression quality (e.g., for archive/transfer workflows).
    /// Latency may reach several frames.
    HighQuality,
}

impl Hx2Preset {
    /// Return the target keyframe interval (in frames) for this preset.
    pub fn default_keyframe_interval(self) -> u32 {
        match self {
            Self::LowLatency => 30,
            Self::Balanced => 60,
            Self::HighQuality => 120,
        }
    }

    /// Return the default target bitrate in kbps for 1080p at 30 fps.
    pub fn default_bitrate_kbps(self) -> u32 {
        match self {
            Self::LowLatency => 8_000,
            Self::Balanced => 4_000,
            Self::HighQuality => 2_000,
        }
    }

    /// Returns the AV1 speed setting (0 = slowest/best, 8 = fastest/worst).
    pub fn av1_speed(self) -> u8 {
        match self {
            Self::LowLatency => 8,
            Self::Balanced => 5,
            Self::HighQuality => 2,
        }
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::LowLatency => "low-latency",
            Self::Balanced => "balanced",
            Self::HighQuality => "high-quality",
        }
    }
}

// ---------------------------------------------------------------------------
// Hx2Config
// ---------------------------------------------------------------------------

/// Configuration for the NDI|HX2 encoder/decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hx2Config {
    /// Target bitrate in kilobits per second.
    pub bitrate_kbps: u32,
    /// Number of frames between forced keyframes (IDR frames).
    pub keyframe_interval: u32,
    /// Encoding preset that controls the speed/quality trade-off.
    pub preset: Hx2Preset,
    /// Enable rate-distortion optimisation lookahead (increases latency).
    pub lookahead_frames: u32,
    /// AV1 tile columns (log2).  0 means single tile column.
    pub tile_cols_log2: u8,
    /// AV1 tile rows (log2).  0 means single tile row.
    pub tile_rows_log2: u8,
}

impl Hx2Config {
    /// Create a new config from a preset.
    ///
    /// All other fields are derived from the preset's defaults.
    pub fn from_preset(preset: Hx2Preset) -> Self {
        Self {
            bitrate_kbps: preset.default_bitrate_kbps(),
            keyframe_interval: preset.default_keyframe_interval(),
            preset,
            lookahead_frames: 0,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
        }
    }

    /// Validate the configuration.  Returns an error if any field is
    /// outside its valid range.
    pub fn validate(&self) -> Result<()> {
        if self.bitrate_kbps == 0 {
            return Err(NdiError::Codec("HX2 bitrate_kbps must be > 0".to_string()));
        }
        if self.keyframe_interval == 0 {
            return Err(NdiError::Codec(
                "HX2 keyframe_interval must be > 0".to_string(),
            ));
        }
        if self.tile_cols_log2 > 6 {
            return Err(NdiError::Codec(
                "HX2 tile_cols_log2 must be in 0..=6".to_string(),
            ));
        }
        if self.tile_rows_log2 > 6 {
            return Err(NdiError::Codec(
                "HX2 tile_rows_log2 must be in 0..=6".to_string(),
            ));
        }
        Ok(())
    }

    /// Estimate the maximum compressed frame size in bytes for the given
    /// resolution and frame rate.
    ///
    /// This is a conservative upper bound used for buffer pre-allocation;
    /// actual frames will typically be much smaller.
    pub fn max_frame_bytes(&self, width: u32, height: u32, fps: f64) -> usize {
        // bits per frame = bitrate / fps; add 20 % headroom for keyframes
        let bits_per_frame = (f64::from(self.bitrate_kbps) * 1000.0 / fps) * 1.20;
        // Keyframe overhead: use 3× the average frame size
        let keyframe_bits = bits_per_frame * 3.0;
        let _ = (width, height); // used in a more complete implementation
        (keyframe_bits / 8.0) as usize + 4096 // +4 KiB for headers
    }
}

impl Default for Hx2Config {
    fn default() -> Self {
        Self::from_preset(Hx2Preset::Balanced)
    }
}

// ---------------------------------------------------------------------------
// Hx2Encoder
// ---------------------------------------------------------------------------

/// NDI|HX2 video encoder.
///
/// The encoder accepts raw [`NdiVideoFrame`] descriptors together with
/// associated pixel data and returns AV1-encoded byte payloads.
///
/// # AV1 integration note
///
/// A full AV1 encoder integration would call into a crate such as `rav1e` or
/// `dav1d` (via safe Rust bindings).  This stub generates a deterministic
/// synthetic payload whose size scales with the configured bitrate so that
/// downstream buffer-management and scheduling code can be tested without a
/// real codec dependency.
pub struct Hx2Encoder {
    config: Hx2Config,
    frame_count: u64,
}

impl Hx2Encoder {
    /// Create a new encoder from `config`.
    pub fn new(config: Hx2Config) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            frame_count: 0,
        })
    }

    /// Return whether the next frame should be a keyframe.
    pub fn is_next_keyframe(&self) -> bool {
        self.frame_count % u64::from(self.config.keyframe_interval) == 0
    }

    /// Encode a single video frame.
    ///
    /// # Arguments
    ///
    /// * `frame` — Metadata descriptor of the frame.
    /// * `pixels` — Raw pixel data (must be `frame.data_size_bytes` bytes).
    ///
    /// # Returns
    ///
    /// A byte vector containing the AV1 OBU payload ready for network
    /// transmission in an NDI|HX2 packet.
    pub fn encode(&mut self, frame: &NdiVideoFrame, pixels: &[u8]) -> Result<Vec<u8>> {
        let expected = frame.data_size_bytes as usize;
        if pixels.len() < expected {
            return Err(NdiError::Codec(format!(
                "HX2 encode: pixel buffer too small ({} < {})",
                pixels.len(),
                expected
            )));
        }

        let is_key = self.is_next_keyframe();
        self.frame_count += 1;

        // --- AV1 integration point ---
        // In a production implementation this would call into rav1e:
        //
        //   let mut enc = rav1e::Context::new(&rav1e::Config::default());
        //   enc.send_frame(...);
        //   let packet = enc.receive_packet()?;
        //   return Ok(packet.data.to_vec());
        //
        // For now we produce a synthetic payload that:
        // - Starts with an AV1 Temporal Unit Delimiter OBU (0x12 0x00).
        // - Has a size proportional to the configured bitrate.
        // - Embeds the frame dimensions and timecode for round-trip tests.

        let approx_fps = frame.frame_rate() as f64;
        let target_bytes =
            self.config
                .max_frame_bytes(frame.width, frame.height, approx_fps.max(1.0));

        // Scale down for non-key frames.
        let payload_len = if is_key {
            target_bytes.min(1 << 20) // cap at 1 MiB
        } else {
            (target_bytes / 8).max(64)
        };

        let mut payload = Vec::with_capacity(payload_len + 24);
        // AV1 Temporal Unit Delimiter OBU header (simplified).
        payload.extend_from_slice(&[0x12, 0x00]);
        // Frame type tag: 0x01 = key, 0x00 = delta.
        payload.push(if is_key { 0x01 } else { 0x00 });
        // Embed frame metadata for decoder round-trip tests.
        payload.extend_from_slice(&frame.width.to_le_bytes());
        payload.extend_from_slice(&frame.height.to_le_bytes());
        payload.extend_from_slice(&frame.timecode.to_le_bytes());
        payload.extend_from_slice(&frame.frame_rate_n.to_le_bytes());
        payload.extend_from_slice(&frame.frame_rate_d.to_le_bytes());
        // Pad to approximate the target compressed size.
        let header_len = payload.len();
        if payload_len > header_len {
            payload.resize(payload_len, 0xABu8);
        }

        Ok(payload)
    }

    /// Return a reference to the encoder configuration.
    pub fn config(&self) -> &Hx2Config {
        &self.config
    }

    /// Return the number of frames encoded so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Reset the frame counter (e.g. after a stream restart).
    pub fn reset(&mut self) {
        self.frame_count = 0;
    }
}

// ---------------------------------------------------------------------------
// Hx2Decoder
// ---------------------------------------------------------------------------

/// Decoded frame metadata returned by [`Hx2Decoder::decode`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hx2DecodedFrame {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// 100-nanosecond NDI timecode.
    pub timecode: i64,
    /// Frame-rate numerator.
    pub frame_rate_n: u32,
    /// Frame-rate denominator.
    pub frame_rate_d: u32,
    /// Whether this was a keyframe (IDR).
    pub is_keyframe: bool,
    /// Reconstructed pixel data (empty in stub — full AV1 decode omitted).
    pub pixels: Vec<u8>,
}

/// NDI|HX2 video decoder.
///
/// Parses the synthetic AV1 OBU payload produced by [`Hx2Encoder`] and
/// reconstructs frame metadata.  Pixel reconstruction requires a real AV1
/// decoder library integration (e.g. `dav1d`).
pub struct Hx2Decoder {
    frames_decoded: u64,
}

impl Hx2Decoder {
    /// Create a new decoder.
    pub fn new() -> Self {
        Self { frames_decoded: 0 }
    }

    /// Decode an AV1 OBU payload produced by [`Hx2Encoder`].
    ///
    /// # Errors
    ///
    /// Returns [`NdiError::Codec`] if the payload is too short or if the
    /// magic bytes do not match the expected AV1 TUD OBU header.
    pub fn decode(&mut self, data: &[u8]) -> Result<Hx2DecodedFrame> {
        // Minimum header: 2 (OBU hdr) + 1 (key flag) + 4+4+8+4+4 = 27 bytes.
        const MIN_LEN: usize = 27;
        if data.len() < MIN_LEN {
            return Err(NdiError::Codec(format!(
                "HX2 decode: payload too short ({} < {MIN_LEN})",
                data.len()
            )));
        }
        // Validate AV1 Temporal Unit Delimiter OBU magic.
        if data[0] != 0x12 || data[1] != 0x00 {
            return Err(NdiError::Codec(
                "HX2 decode: invalid AV1 OBU magic bytes".to_string(),
            ));
        }
        let is_keyframe = data[2] == 0x01;
        let mut cursor = 3usize;

        let width = u32::from_le_bytes(
            data[cursor..cursor + 4]
                .try_into()
                .map_err(|_| NdiError::Codec("HX2 decode: width parse error".to_string()))?,
        );
        cursor += 4;
        let height = u32::from_le_bytes(
            data[cursor..cursor + 4]
                .try_into()
                .map_err(|_| NdiError::Codec("HX2 decode: height parse error".to_string()))?,
        );
        cursor += 4;
        let timecode = i64::from_le_bytes(
            data[cursor..cursor + 8]
                .try_into()
                .map_err(|_| NdiError::Codec("HX2 decode: timecode parse error".to_string()))?,
        );
        cursor += 8;
        let frame_rate_n =
            u32::from_le_bytes(data[cursor..cursor + 4].try_into().map_err(|_| {
                NdiError::Codec("HX2 decode: frame_rate_n parse error".to_string())
            })?);
        cursor += 4;
        let frame_rate_d =
            u32::from_le_bytes(data[cursor..cursor + 4].try_into().map_err(|_| {
                NdiError::Codec("HX2 decode: frame_rate_d parse error".to_string())
            })?);

        self.frames_decoded += 1;

        Ok(Hx2DecodedFrame {
            width,
            height,
            timecode,
            frame_rate_n,
            frame_rate_d,
            is_keyframe,
            pixels: Vec::new(), // full AV1 pixel decode is a future integration point
        })
    }

    /// Return the number of frames decoded since construction (or last reset).
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }

    /// Reset the internal counter.
    pub fn reset(&mut self) {
        self.frames_decoded = 0;
    }
}

impl Default for Hx2Decoder {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av_buffer::NdiVideoFrame;

    fn make_frame(w: u32, h: u32) -> NdiVideoFrame {
        NdiVideoFrame::new(w, h, 30, 1, 123_456_789, w * 4, w * h * 4)
    }

    // -----------------------------------------------------------------------
    // Hx2Config
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default_is_balanced() {
        let cfg = Hx2Config::default();
        assert_eq!(cfg.preset, Hx2Preset::Balanced);
    }

    #[test]
    fn test_config_validation_zero_bitrate() {
        let mut cfg = Hx2Config::from_preset(Hx2Preset::LowLatency);
        cfg.bitrate_kbps = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_zero_keyframe_interval() {
        let mut cfg = Hx2Config::from_preset(Hx2Preset::Balanced);
        cfg.keyframe_interval = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_tile_cols_out_of_range() {
        let mut cfg = Hx2Config::from_preset(Hx2Preset::HighQuality);
        cfg.tile_cols_log2 = 7;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_valid_presets() {
        for preset in [
            Hx2Preset::LowLatency,
            Hx2Preset::Balanced,
            Hx2Preset::HighQuality,
        ] {
            assert!(Hx2Config::from_preset(preset).validate().is_ok());
        }
    }

    #[test]
    fn test_max_frame_bytes_positive() {
        let cfg = Hx2Config::from_preset(Hx2Preset::Balanced);
        let bytes = cfg.max_frame_bytes(1920, 1080, 30.0);
        assert!(bytes > 0);
    }

    // -----------------------------------------------------------------------
    // Hx2Preset
    // -----------------------------------------------------------------------

    #[test]
    fn test_preset_labels() {
        assert_eq!(Hx2Preset::LowLatency.label(), "low-latency");
        assert_eq!(Hx2Preset::Balanced.label(), "balanced");
        assert_eq!(Hx2Preset::HighQuality.label(), "high-quality");
    }

    #[test]
    fn test_preset_av1_speed_ordering() {
        // Low-latency must be the fastest (highest speed value)
        assert!(Hx2Preset::LowLatency.av1_speed() > Hx2Preset::HighQuality.av1_speed());
    }

    // -----------------------------------------------------------------------
    // Hx2Encoder
    // -----------------------------------------------------------------------

    #[test]
    fn test_encoder_keyframe_at_start() {
        let enc = Hx2Encoder::new(Hx2Config::default()).expect("encoder creation failed");
        assert!(enc.is_next_keyframe());
    }

    #[test]
    fn test_encoder_encode_returns_non_empty() {
        let mut enc = Hx2Encoder::new(Hx2Config::default()).expect("encoder creation failed");
        let frame = make_frame(1920, 1080);
        let pixels = vec![0u8; frame.data_size_bytes as usize];
        let encoded = enc.encode(&frame, &pixels).expect("encode failed");
        assert!(!encoded.empty_check());
        // AV1 OBU magic bytes
        assert_eq!(encoded[0], 0x12);
        assert_eq!(encoded[1], 0x00);
    }

    #[test]
    fn test_encoder_encode_pixel_buffer_too_small() {
        let mut enc = Hx2Encoder::new(Hx2Config::default()).expect("encoder creation failed");
        let frame = make_frame(1920, 1080);
        let pixels = vec![0u8; 10]; // way too small
        assert!(enc.encode(&frame, &pixels).is_err());
    }

    #[test]
    fn test_encoder_frame_count_increments() {
        let mut enc = Hx2Encoder::new(Hx2Config::default()).expect("encoder creation failed");
        let frame = make_frame(640, 480);
        let pixels = vec![0u8; frame.data_size_bytes as usize];
        enc.encode(&frame, &pixels).expect("encode 1 failed");
        enc.encode(&frame, &pixels).expect("encode 2 failed");
        assert_eq!(enc.frame_count(), 2);
    }

    #[test]
    fn test_encoder_keyframe_marking() {
        let cfg = Hx2Config {
            keyframe_interval: 3,
            ..Hx2Config::from_preset(Hx2Preset::LowLatency)
        };
        let mut enc = Hx2Encoder::new(cfg).expect("encoder creation failed");
        let frame = make_frame(320, 240);
        let pixels = vec![0u8; frame.data_size_bytes as usize];

        let kf0 = enc.encode(&frame, &pixels).expect("encode kf0 failed"); // frame 0 = key
        let df1 = enc.encode(&frame, &pixels).expect("encode df1 failed"); // frame 1 = delta
        let df2 = enc.encode(&frame, &pixels).expect("encode df2 failed"); // frame 2 = delta
        let kf3 = enc.encode(&frame, &pixels).expect("encode kf3 failed"); // frame 3 = key

        assert_eq!(kf0[2], 0x01, "frame 0 should be keyframe");
        assert_eq!(df1[2], 0x00, "frame 1 should be delta");
        assert_eq!(df2[2], 0x00, "frame 2 should be delta");
        assert_eq!(kf3[2], 0x01, "frame 3 should be keyframe");
    }

    #[test]
    fn test_encoder_keyframe_larger_than_delta() {
        let mut enc = Hx2Encoder::new(Hx2Config::default()).expect("encoder creation failed");
        let frame = make_frame(1280, 720);
        let pixels = vec![0u8; frame.data_size_bytes as usize];
        let key = enc.encode(&frame, &pixels).expect("keyframe encode failed");
        let delta = enc.encode(&frame, &pixels).expect("delta encode failed");
        assert!(
            key.len() > delta.len(),
            "keyframe should be larger than delta"
        );
    }

    // -----------------------------------------------------------------------
    // Hx2Decoder
    // -----------------------------------------------------------------------

    #[test]
    fn test_decoder_decode_round_trip() {
        let mut enc = Hx2Encoder::new(Hx2Config::default()).expect("encoder creation failed");
        let mut dec = Hx2Decoder::new();
        let frame = make_frame(1920, 1080);
        let pixels = vec![42u8; frame.data_size_bytes as usize];
        let payload = enc.encode(&frame, &pixels).expect("encode failed");
        let decoded = dec.decode(&payload).expect("decode failed");

        assert_eq!(decoded.width, 1920);
        assert_eq!(decoded.height, 1080);
        assert_eq!(decoded.timecode, frame.timecode);
        assert_eq!(decoded.frame_rate_n, 30);
        assert_eq!(decoded.frame_rate_d, 1);
        assert!(decoded.is_keyframe);
    }

    #[test]
    fn test_decoder_rejects_short_payload() {
        let mut dec = Hx2Decoder::new();
        assert!(dec.decode(&[0x12, 0x00]).is_err());
    }

    #[test]
    fn test_decoder_rejects_invalid_magic() {
        let mut dec = Hx2Decoder::new();
        let mut bad = vec![0x00u8; 30];
        bad[0] = 0xFF; // wrong magic
        assert!(dec.decode(&bad).is_err());
    }

    #[test]
    fn test_decoder_frame_count() {
        let mut enc = Hx2Encoder::new(Hx2Config::default()).expect("encoder creation failed");
        let mut dec = Hx2Decoder::new();
        let frame = make_frame(640, 480);
        let pixels = vec![0u8; frame.data_size_bytes as usize];
        let p = enc.encode(&frame, &pixels).expect("encode failed");
        dec.decode(&p).expect("decode 1 failed");
        dec.decode(&p).expect("decode 2 failed");
        assert_eq!(dec.frames_decoded(), 2);
    }
}

// ---------------------------------------------------------------------------
// Helper extension for tests
// ---------------------------------------------------------------------------

trait IsEmpty {
    fn empty_check(&self) -> bool;
}
impl IsEmpty for Vec<u8> {
    fn empty_check(&self) -> bool {
        self.is_empty()
    }
}
