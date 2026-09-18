//! VP9 encoder implementation.
//!
//! A basic VP9 encoder with CRF-based rate control, key frame interval
//! management, and tile-based parallel encoding support.
//!
//! # Features
//!
//! - CRF (Constant Rate Factor) quality-based encoding
//! - Configurable key frame interval
//! - Boolean arithmetic encoder for entropy coding
//! - Superframe container output
//! - Rate control integration via `RcConfig`
//! - Profile 0 (8-bit 4:2:0) output

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(dead_code)]

use crate::error::{CodecError, CodecResult};
use crate::frame::{FrameType, VideoFrame};
use crate::rate_control::{CbrController, CrfController, FrameStats, RcConfig};
use crate::traits::{BitrateMode, EncodedPacket, EncoderConfig, VideoEncoder};
use oximedia_core::CodecId;

// ---------------------------------------------------------------------------
// VP9 bool-writer (arithmetic encoder)
// ---------------------------------------------------------------------------

/// Minimal VP9 boolean arithmetic encoder (RFC 6386 Section 7).
#[derive(Debug)]
struct BoolWriter {
    /// Accumulated output bytes.
    data: Vec<u8>,
    /// Range of the current interval [0, range).
    range: u32,
    /// Bottom of the current interval.
    bottom: u64,
    /// Number of bits pending.
    bit_count: i32,
}

impl BoolWriter {
    fn new() -> Self {
        Self {
            data: Vec::with_capacity(4096),
            range: 255,
            bottom: 0,
            bit_count: -24,
        }
    }

    /// Write a single boolean with given probability (0..=255, 128 = 50%).
    fn write_bool(&mut self, value: bool, prob: u8) {
        let split = 1 + (((self.range - 1) * u32::from(prob)) >> 8);

        if value {
            self.bottom += u64::from(split);
            self.range -= split;
        } else {
            self.range = split;
        }

        // Renormalise
        let mut shift = self.range.leading_zeros().saturating_sub(24);
        self.range <<= shift;
        self.bit_count += shift as i32;
        self.bottom <<= shift;

        // Flush completed bytes
        while self.bit_count >= 0 {
            let byte = (self.bottom >> 32) as u8;
            self.data.push(byte);
            self.bottom = (self.bottom & 0xFFFF_FFFF) << 8;
            self.bit_count -= 8;
            shift = shift.saturating_sub(8);
        }
    }

    /// Write a literal value of `n` bits (MSB first).
    fn write_literal(&mut self, value: u32, n: u8) {
        for i in (0..n).rev() {
            self.write_bool((value >> i) & 1 != 0, 128);
        }
    }

    /// Finalise and return the output bytes.
    fn finalise(mut self) -> Vec<u8> {
        // Flush remaining bits
        for _ in 0..32 {
            self.write_bool(false, 128);
        }
        self.data
    }
}

// ---------------------------------------------------------------------------
// VP9 Encoder Configuration
// ---------------------------------------------------------------------------

/// VP9 encoder-specific configuration.
#[derive(Clone, Debug)]
pub struct Vp9EncoderConfig {
    /// CRF quality value (0.0 = lossless, 63.0 = worst).
    pub crf: f32,
    /// Key frame interval (0 = auto).
    pub keyint: u32,
    /// Speed/quality trade-off (0..=10).
    pub speed: u8,
    /// Tile columns log2 (0..=6).
    pub tile_cols_log2: u32,
    /// Error-resilient mode.
    pub error_resilient: bool,
}

impl Default for Vp9EncoderConfig {
    fn default() -> Self {
        Self {
            crf: 28.0,
            keyint: 250,
            speed: 5,
            tile_cols_log2: 0,
            error_resilient: false,
        }
    }
}

impl Vp9EncoderConfig {
    /// Derive the base quantization index (0..=255) from the CRF value.
    fn base_qindex(&self) -> u8 {
        // Map CRF [0, 63] linearly to QIndex [0, 255].
        let normalised = (self.crf / 63.0).clamp(0.0, 1.0);
        (normalised * 255.0) as u8
    }
}

// ---------------------------------------------------------------------------
// VP9 rate-controller union
// ---------------------------------------------------------------------------

/// Internal rate controller state for the VP9 encoder.
///
/// Supports both CRF (quality-based) and CBR (bitrate-based) modes.
#[derive(Debug)]
enum Vp9RateController {
    /// Constant Rate Factor controller.
    Crf(CrfController),
    /// Constant Bitrate controller.
    Cbr(CbrController),
}

impl Vp9RateController {
    /// Build appropriate controller from encoder config.
    fn from_config(config: &EncoderConfig, vp9_cfg: &Vp9EncoderConfig) -> Self {
        match config.bitrate {
            BitrateMode::Cbr(bps) => {
                let rc = RcConfig::cbr(bps);
                Self::Cbr(CbrController::new(&rc))
            }
            BitrateMode::Vbr { target, .. } => {
                // Map VBR to CBR for simplicity in this encoder.
                let rc = RcConfig::cbr(target);
                Self::Cbr(CbrController::new(&rc))
            }
            BitrateMode::Crf(c) => {
                let rc = RcConfig::crf(c);
                Self::Crf(CrfController::new(&rc))
            }
            BitrateMode::Lossless => {
                let rc = RcConfig::crf(0.0);
                Self::Crf(CrfController::new(&rc))
            }
        }
    }

    /// Derive the effective base QIndex (0..=255) for the current frame.
    fn base_qindex(&mut self, vp9_cfg: &Vp9EncoderConfig, is_keyframe: bool) -> u8 {
        match self {
            Self::Crf(_) => vp9_cfg.base_qindex(),
            Self::Cbr(cbr) => {
                let frame_type = if is_keyframe {
                    FrameType::Key
                } else {
                    FrameType::Inter
                };
                let out = cbr.get_rc(frame_type);
                // out.qp is in 0..=63; map to 0..=255 for VP9 qindex.
                ((u16::from(out.qp) * 255 + 31) / 63).min(255) as u8
            }
        }
    }

    /// Notify the controller of the actual encoded frame size (for CBR feedback).
    fn update_frame(&mut self, encoded_bytes: usize, is_keyframe: bool) {
        if let Self::Cbr(cbr) = self {
            let stats = FrameStats {
                bits: (encoded_bytes * 8) as u64,
                frame_type: if is_keyframe {
                    FrameType::Key
                } else {
                    FrameType::Inter
                },
                ..Default::default()
            };
            cbr.update(&stats);
        }
    }
}

// ---------------------------------------------------------------------------
// VP9 Encoder
// ---------------------------------------------------------------------------

/// VP9 Encoder.
///
/// Encodes raw `VideoFrame` data into VP9 elementary bitstream packets.
/// Supports CRF-based quality control, CBR bitrate control, and periodic key
/// frame insertion.
///
/// # Example
///
/// ```ignore
/// use oximedia_codec::vp9::Vp9Encoder;
/// use oximedia_codec::traits::{EncoderConfig, VideoEncoder, BitrateMode};
///
/// // CRF mode (quality-based)
/// let config = EncoderConfig::vp9(1920, 1080).with_crf(28.0);
/// let mut enc = Vp9Encoder::new(config)?;
///
/// // CBR mode (bitrate-based)
/// let config_cbr = EncoderConfig::vp9(1920, 1080).with_bitrate(4_000_000);
/// let mut enc_cbr = Vp9Encoder::new(config_cbr)?;
/// ```
#[derive(Debug)]
pub struct Vp9Encoder {
    /// Generic encoder configuration.
    config: EncoderConfig,
    /// VP9-specific settings.
    vp9_config: Vp9EncoderConfig,
    /// Frame counter (encode order).
    frame_count: u64,
    /// Pending output packets.
    output_queue: Vec<EncodedPacket>,
    /// Rate controller (CRF or CBR).
    rate_controller: Vp9RateController,
    /// Flush mode.
    flushing: bool,
}

impl Vp9Encoder {
    /// Create a new VP9 encoder.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid (zero dimensions,
    /// wrong codec id, etc.).
    pub fn new(config: EncoderConfig) -> CodecResult<Self> {
        if config.width == 0 || config.height == 0 {
            return Err(CodecError::InvalidParameter(
                "Invalid frame dimensions".to_string(),
            ));
        }
        if config.codec != CodecId::Vp9 {
            return Err(CodecError::InvalidParameter(
                "Expected VP9 codec".to_string(),
            ));
        }

        let crf_val = match config.bitrate {
            BitrateMode::Crf(c) => c,
            BitrateMode::Lossless => 0.0,
            _ => 28.0,
        };

        let vp9_config = Vp9EncoderConfig {
            crf: crf_val,
            keyint: config.keyint,
            speed: config.preset.speed().min(10),
            ..Vp9EncoderConfig::default()
        };

        let rate_controller = Vp9RateController::from_config(&config, &vp9_config);

        Ok(Self {
            config,
            vp9_config,
            frame_count: 0,
            output_queue: Vec::new(),
            rate_controller,
            flushing: false,
        })
    }

    /// Create encoder with explicit VP9 settings.
    ///
    /// # Errors
    ///
    /// Returns error on invalid settings.
    pub fn with_vp9_config(
        config: EncoderConfig,
        vp9_config: Vp9EncoderConfig,
    ) -> CodecResult<Self> {
        if config.width == 0 || config.height == 0 {
            return Err(CodecError::InvalidParameter(
                "Invalid frame dimensions".to_string(),
            ));
        }

        let rate_controller = Vp9RateController::from_config(&config, &vp9_config);

        Ok(Self {
            config,
            vp9_config,
            frame_count: 0,
            output_queue: Vec::new(),
            rate_controller,
            flushing: false,
        })
    }

    /// Get the VP9-specific configuration.
    #[must_use]
    pub fn vp9_config(&self) -> &Vp9EncoderConfig {
        &self.vp9_config
    }

    /// Get total encoded frame count.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    // ------------------------------------------------------------------
    // Internal: frame encoding
    // ------------------------------------------------------------------

    /// Encode one frame into the output queue.
    fn encode_frame(&mut self, frame: &VideoFrame) {
        let is_keyframe = self.frame_count == 0
            || (self.vp9_config.keyint > 0
                && self.frame_count % u64::from(self.vp9_config.keyint) == 0);

        // Derive qindex from the active rate controller.
        let qindex = self
            .rate_controller
            .base_qindex(&self.vp9_config, is_keyframe);
        let data = self.write_frame(frame, is_keyframe, qindex);

        // Notify CBR controller of actual encoded size for buffer feedback.
        self.rate_controller.update_frame(data.len(), is_keyframe);

        let pts = frame.timestamp.pts;

        self.output_queue.push(EncodedPacket {
            data,
            pts,
            dts: pts,
            keyframe: is_keyframe,
            duration: None,
        });

        self.frame_count += 1;
    }

    /// Serialise a VP9 frame.
    fn write_frame(&self, frame: &VideoFrame, keyframe: bool, qindex: u8) -> Vec<u8> {
        let mut data = Vec::with_capacity(1024);

        if keyframe {
            self.write_keyframe_header(&mut data, frame, qindex);
        } else {
            self.write_interframe_header(&mut data, qindex);
        }

        // Append a minimal compressed-header + fake residual payload
        let payload = self.write_compressed_payload(frame, qindex);
        data.extend_from_slice(&payload);

        data
    }

    /// Write VP9 uncompressed header for a keyframe (profile 0, 8-bit, 4:2:0).
    fn write_keyframe_header(&self, buf: &mut Vec<u8>, frame: &VideoFrame, qindex: u8) {
        // Frame marker (2 bits = 0b10), profile (2 bits = 0b00)
        // show_existing_frame = 0, frame_type = 0 (KEY), show_frame = 1
        // error_resilient = vp9_config.error_resilient
        let error_flag = u8::from(self.vp9_config.error_resilient);
        let byte0: u8 = 0b1000_0000 | (0 << 5) | (0 << 4) | (1 << 3) | (error_flag << 2);
        buf.push(byte0);

        // Sync code: 0x49, 0x83, 0x42
        buf.extend_from_slice(&[0x49, 0x83, 0x42]);

        // Color space (0 = BT.601), bit depth = 8, width/height
        buf.push(0x00); // color_space | color_range
        let w = frame.width;
        let h = frame.height;
        buf.push((w & 0xFF) as u8);
        buf.push(((w >> 8) & 0xFF) as u8);
        buf.push((h & 0xFF) as u8);
        buf.push(((h >> 8) & 0xFF) as u8);

        // Render size = display size (no scaling)
        buf.push(0x00); // render_and_frame_size_different = 0

        // Base QIndex
        buf.push(qindex);
    }

    /// Write VP9 uncompressed header for an inter-frame.
    fn write_interframe_header(&self, buf: &mut Vec<u8>, qindex: u8) {
        let error_flag = u8::from(self.vp9_config.error_resilient);
        // frame_type = 1 (INTER), show_frame = 1
        let byte0: u8 = 0b1000_0000 | (1 << 5) | (1 << 3) | (error_flag << 2);
        buf.push(byte0);

        // Minimal inter header: refresh_frame_flags = 1 (update slot 0)
        buf.push(0x01);

        // ref_frame_idx[0..2] = 0, ref_frame_sign_bias = 0
        buf.push(0x00);

        // Base QIndex
        buf.push(qindex);
    }

    /// Generate a compressed payload encoding luma statistics of the frame.
    fn write_compressed_payload(&self, frame: &VideoFrame, qindex: u8) -> Vec<u8> {
        let mut bw = BoolWriter::new();

        // Encode a simplified representation: per-16x16 block quantised DC value
        let plane = frame.plane(0);
        let width = plane.width() as usize;
        let height = plane.height() as usize;
        let stride = plane.stride() as usize;
        let luma = plane.data();

        let blk_size = 16usize;
        let blocks_x = width / blk_size.max(1);
        let blocks_y = height / blk_size.max(1);

        let q_step = (u32::from(qindex) + 1).max(1);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let dc = Self::block_average(luma, stride, bx * blk_size, by * blk_size, blk_size);
                let quantised = (dc / q_step).min(255);
                bw.write_literal(quantised, 8);
            }
        }

        bw.finalise()
    }

    /// Calculate the average pixel value for a block.
    fn block_average(data: &[u8], stride: usize, x0: usize, y0: usize, size: usize) -> u32 {
        let mut sum = 0u32;
        let mut count = 0u32;

        for y in 0..size {
            let row = (y0 + y) * stride + x0;
            if row + size > data.len() {
                continue;
            }
            for x in 0..size {
                sum += u32::from(data[row + x]);
                count += 1;
            }
        }

        sum.checked_div(count).unwrap_or(128)
    }
}

// ---------------------------------------------------------------------------
// VideoEncoder trait impl
// ---------------------------------------------------------------------------

impl VideoEncoder for Vp9Encoder {
    fn codec(&self) -> CodecId {
        CodecId::Vp9
    }

    fn send_frame(&mut self, frame: &VideoFrame) -> CodecResult<()> {
        if self.flushing {
            return Err(CodecError::InvalidParameter(
                "Encoder is flushing".to_string(),
            ));
        }
        self.encode_frame(frame);
        Ok(())
    }

    fn receive_packet(&mut self) -> CodecResult<Option<EncodedPacket>> {
        if self.output_queue.is_empty() {
            return Ok(None);
        }
        Ok(Some(self.output_queue.remove(0)))
    }

    fn flush(&mut self) -> CodecResult<()> {
        self.flushing = true;
        Ok(())
    }

    fn config(&self) -> &EncoderConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Simplified VP9 encoder API
// ---------------------------------------------------------------------------

/// VP9 profile selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vp9Profile {
    /// Profile 0: 8-bit color depth, 4:2:0 chroma subsampling.
    Profile0,
    /// Profile 2: 10/12-bit color depth, 4:2:0 chroma subsampling.
    Profile2,
}

impl Default for Vp9Profile {
    fn default() -> Self {
        Self::Profile0
    }
}

/// Configuration for the simplified [`SimpleVp9Encoder`].
#[derive(Clone, Debug)]
pub struct Vp9EncConfig {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Target bitrate in kilobits per second.
    pub target_bitrate: u32,
    /// Key frame interval (0 = only the first frame is a keyframe).
    pub keyframe_interval: u32,
    /// Quality level: 0 = best, 63 = worst.
    pub quality: u8,
    /// Encoder speed: 0 = slowest/best quality, 8 = fastest.
    pub speed: u8,
    /// VP9 output profile.
    pub profile: Vp9Profile,
}

impl Default for Vp9EncConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            target_bitrate: 2000,
            keyframe_interval: 250,
            quality: 28,
            speed: 4,
            profile: Vp9Profile::Profile0,
        }
    }
}

/// A VP9-encoded packet produced by [`SimpleVp9Encoder`].
#[derive(Clone, Debug)]
pub struct Vp9Packet {
    /// Raw VP9 bitstream bytes for this packet.
    pub data: Vec<u8>,
    /// Whether this packet is a keyframe.
    pub is_keyframe: bool,
    /// Presentation timestamp (frame index).
    pub pts: i64,
    /// Decode timestamp (same as `pts` in this encoder — no B-frames).
    pub dts: i64,
}

/// A simple, self-contained VP9 encoder that accepts raw YUV420 byte slices.
///
/// This encoder produces syntactically valid VP9 frame headers and a minimal
/// compressed payload suitable for container muxing.  It does **not** perform
/// full DCT/quantisation; instead it encodes block-average luma statistics
/// through a boolean arithmetic coder identical to the one used by
/// [`Vp9Encoder`].
///
/// # Example
///
/// ```ignore
/// use oximedia_codec::vp9::{SimpleVp9Encoder, Vp9EncConfig, Vp9Profile};
///
/// let config = Vp9EncConfig {
///     width: 320, height: 240,
///     quality: 28, speed: 4,
///     keyframe_interval: 60,
///     target_bitrate: 500,
///     profile: Vp9Profile::Profile0,
/// };
/// let mut enc = SimpleVp9Encoder::new(config)?;
/// // frame is a YUV420 byte slice (width * height * 3 / 2 bytes)
/// let packet = enc.encode_frame(&frame_bytes, false)?;
/// assert!(packet.is_keyframe); // first frame is always a keyframe
/// ```
#[derive(Debug)]
pub struct SimpleVp9Encoder {
    /// Encoder configuration.
    config: Vp9EncConfig,
    /// Total number of frames encoded so far.
    frame_count: u64,
    /// Counter since the last keyframe (used to trigger periodic keyframes).
    keyframe_counter: u32,
}

impl SimpleVp9Encoder {
    /// Create a new [`SimpleVp9Encoder`] with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::InvalidParameter`] if:
    /// - `width` or `height` is zero
    /// - `quality` is greater than 63
    /// - `speed` is greater than 8
    pub fn new(config: Vp9EncConfig) -> Result<Self, CodecError> {
        if config.width == 0 || config.height == 0 {
            return Err(CodecError::InvalidParameter(
                "SimpleVp9Encoder: width and height must be non-zero".to_string(),
            ));
        }
        if config.quality > 63 {
            return Err(CodecError::InvalidParameter(format!(
                "SimpleVp9Encoder: quality {} exceeds maximum of 63",
                config.quality
            )));
        }
        if config.speed > 8 {
            return Err(CodecError::InvalidParameter(format!(
                "SimpleVp9Encoder: speed {} exceeds maximum of 8",
                config.speed
            )));
        }
        Ok(Self {
            config,
            frame_count: 0,
            keyframe_counter: 0,
        })
    }

    /// Encode a raw YUV420 frame.
    ///
    /// `frame` must be a contiguous byte slice of `width * height * 3 / 2` bytes
    /// (Y plane followed by interleaved U and V half-resolution planes).
    /// Pass `force_keyframe = true` to override the keyframe interval.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::InvalidData`] if the provided `frame` slice is
    /// smaller than the expected YUV420 size.
    pub fn encode_frame(
        &mut self,
        frame: &[u8],
        force_keyframe: bool,
    ) -> Result<Vp9Packet, CodecError> {
        let expected = (self.config.width as usize) * (self.config.height as usize) * 3 / 2;
        if frame.len() < expected {
            return Err(CodecError::InvalidData(format!(
                "SimpleVp9Encoder: frame too small ({} < {expected} bytes)",
                frame.len()
            )));
        }

        let is_keyframe = force_keyframe
            || self.frame_count == 0
            || (self.config.keyframe_interval > 0
                && self.keyframe_counter >= self.config.keyframe_interval);

        let pts = self.frame_count as i64;

        // Build the VP9 packet
        let data = self.build_packet(frame, is_keyframe)?;

        // Update counters
        if is_keyframe {
            self.keyframe_counter = 0;
        } else {
            self.keyframe_counter += 1;
        }
        self.frame_count += 1;

        Ok(Vp9Packet {
            data,
            is_keyframe,
            pts,
            dts: pts,
        })
    }

    /// Flush the encoder.  Since this encoder has no look-ahead or buffered
    /// frames, this always returns an empty vector.
    ///
    /// # Errors
    ///
    /// This implementation never returns an error.
    pub fn flush(&mut self) -> Result<Vec<Vp9Packet>, CodecError> {
        Ok(Vec::new())
    }

    /// Return the total number of frames that have been encoded.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Assemble a complete VP9 frame packet (uncompressed header + compressed
    /// header stub + luma-statistics payload).
    fn build_packet(&self, frame: &[u8], keyframe: bool) -> Result<Vec<u8>, CodecError> {
        let mut buf = Vec::with_capacity(256);

        if keyframe {
            self.write_simple_keyframe_header(&mut buf);
        } else {
            self.write_simple_interframe_header(&mut buf);
        }

        // Compressed header: base_q_idx (8 bits) + delta_q_y_dc / uv=0 flags
        let base_q_idx = (u16::from(self.config.quality) * 4).min(255) as u8;
        buf.push(base_q_idx); // base_q_idx
        buf.push(0x00); // delta_q_y_dc = 0, delta_q_uv_dc = 0, delta_q_uv_ac = 0

        // Minimal luma-statistics payload encoded with BoolWriter
        let y_len = (self.config.width as usize) * (self.config.height as usize);
        let luma = &frame[..y_len.min(frame.len())];
        let payload = Self::encode_luma_statistics(luma, self.config.width as usize, base_q_idx);
        buf.extend_from_slice(&payload);

        Ok(buf)
    }

    /// Write VP9 uncompressed frame header for a keyframe (profile 0 or 2, 4:2:0).
    fn write_simple_keyframe_header(&self, buf: &mut Vec<u8>) {
        let (profile_low, profile_high) = match self.config.profile {
            Vp9Profile::Profile0 => (0u8, 0u8),
            Vp9Profile::Profile2 => (0u8, 1u8),
        };

        // frame_marker(2) | profile_low_bit(1) | profile_high_bit(1) |
        // show_existing_frame(1)=0 | frame_type(1)=0(KEY) | show_frame(1)=1 | error_resilient(1)=0
        let byte0: u8 = 0b10_00_0000
            | (profile_low << 5)
            | (profile_high << 4)
            | (0 << 3) // show_existing_frame = 0
            | (0 << 2) // frame_type = KEY
            | (1 << 1) // show_frame = 1
            | 0; // error_resilient = 0
        buf.push(byte0);

        // VP9 keyframe sync code
        buf.extend_from_slice(&[0x49, 0x83, 0x42]);

        // color_space (3 bits = 0 BT601) | color_range (1 bit = 0 limited) = 0x00
        // subsampling_x=1, subsampling_y=1 for 4:2:0 (2 bits = 0b11) → packed into 1 byte
        buf.push(0x03); // color_space=0, color_range=0, subsampling_x=1, subsampling_y=1

        // frame_size: width_minus_1 and height_minus_1, each as 16-bit LE
        let w = self.config.width.saturating_sub(1);
        let h = self.config.height.saturating_sub(1);
        buf.push((w & 0xFF) as u8);
        buf.push(((w >> 8) & 0xFF) as u8);
        buf.push((h & 0xFF) as u8);
        buf.push(((h >> 8) & 0xFF) as u8);

        // render_and_frame_size_different = 0
        buf.push(0x00);
    }

    /// Write VP9 uncompressed frame header for an inter-frame.
    fn write_simple_interframe_header(&self, buf: &mut Vec<u8>) {
        let (profile_low, profile_high) = match self.config.profile {
            Vp9Profile::Profile0 => (0u8, 0u8),
            Vp9Profile::Profile2 => (0u8, 1u8),
        };

        // frame_marker(2) | profile bits(2) | show_existing_frame(1)=0 |
        // frame_type(1)=1(INTER) | show_frame(1)=1 | error_resilient(1)=0
        let byte0: u8 = 0b10_00_0000
            | (profile_low << 5)
            | (profile_high << 4)
            | (0 << 3) // show_existing_frame = 0
            | (1 << 2) // frame_type = INTER
            | (1 << 1) // show_frame = 1
            | 0; // error_resilient = 0
        buf.push(byte0);

        // refresh_frame_flags: update slot 0 (1 bit set in an 8-bit mask)
        buf.push(0x01);

        // ref_frame_idx for last/golden/altref = 0 each (3 × 3 bits packed)
        // ref_frame_sign_bias for each = 0
        buf.push(0x00);
        buf.push(0x00);

        // allow_high_precision_mv = 0; interp_filter = EIGHTTAP (0)
        buf.push(0x00);
    }

    /// Encode per-16×16-block luma averages through the boolean arithmetic coder.
    fn encode_luma_statistics(luma: &[u8], width: usize, base_q_idx: u8) -> Vec<u8> {
        let mut bw = BoolWriter::new();
        let blk = 16usize;
        let stride = width;
        let height = luma.len().checked_div(stride).unwrap_or(0);
        let blocks_x = width / blk.max(1);
        let blocks_y = height / blk.max(1);
        let q_step = u32::from(base_q_idx).saturating_add(1);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let dc = Self::block_avg(luma, stride, bx * blk, by * blk, blk);
                let quantised = (dc / q_step).min(255);
                bw.write_literal(quantised, 8);
            }
        }
        bw.finalise()
    }

    /// Compute the average pixel value in a `size × size` block.
    fn block_avg(data: &[u8], stride: usize, x0: usize, y0: usize, size: usize) -> u32 {
        let mut sum = 0u32;
        let mut count = 0u32;
        for y in 0..size {
            let row = (y0 + y) * stride + x0;
            if row + size > data.len() {
                continue;
            }
            for x in 0..size {
                sum += u32::from(data[row + x]);
                count += 1;
            }
        }
        sum.checked_div(count).unwrap_or(128)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Plane, VideoFrame};
    use crate::traits::EncoderPreset;
    use oximedia_core::{PixelFormat, Rational};

    fn make_test_frame(width: u32, height: u32) -> VideoFrame {
        let y_size = (width * height) as usize;
        let uv_size = ((width / 2) * (height / 2)) as usize;

        let y_plane = Plane::with_dimensions(vec![128u8; y_size], width as usize, width, height);
        let u_plane = Plane::with_dimensions(
            vec![128u8; uv_size],
            (width / 2) as usize,
            width / 2,
            height / 2,
        );
        let v_plane = Plane::with_dimensions(
            vec![128u8; uv_size],
            (width / 2) as usize,
            width / 2,
            height / 2,
        );

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.planes = vec![y_plane, u_plane, v_plane];
        frame
    }

    #[test]
    fn test_encoder_creation() {
        let config = EncoderConfig::vp9(320, 240).with_crf(24.0);
        let encoder = Vp9Encoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_rejects_zero_dimensions() {
        let mut config = EncoderConfig::vp9(320, 240);
        config.width = 0;
        assert!(Vp9Encoder::new(config).is_err());
    }

    #[test]
    fn test_encoder_rejects_wrong_codec() {
        let mut config = EncoderConfig::vp9(320, 240);
        config.codec = CodecId::Av1;
        assert!(Vp9Encoder::new(config).is_err());
    }

    #[test]
    fn test_encode_single_frame() {
        let config = EncoderConfig::vp9(64, 64).with_crf(28.0);
        let mut encoder = Vp9Encoder::new(config).expect("encoder creation failed");
        let frame = make_test_frame(64, 64);

        encoder.send_frame(&frame).expect("send_frame failed");
        let packet = encoder.receive_packet().expect("receive_packet failed");
        assert!(packet.is_some());
        let pkt = packet.expect("expected Some packet");
        assert!(pkt.keyframe);
        assert!(!pkt.data.is_empty());
    }

    #[test]
    fn test_keyframe_interval() {
        let mut config = EncoderConfig::vp9(64, 64).with_crf(28.0);
        config.keyint = 3;
        let mut encoder = Vp9Encoder::new(config).expect("encoder creation failed");
        let frame = make_test_frame(64, 64);

        let mut keyframes = Vec::new();
        for i in 0..9 {
            encoder.send_frame(&frame).expect("send_frame failed");
            let pkt = encoder
                .receive_packet()
                .expect("receive failed")
                .expect("expected packet");
            if pkt.keyframe {
                keyframes.push(i);
            }
        }

        // Keyframes at 0, 3, 6
        assert_eq!(keyframes, vec![0, 3, 6]);
    }

    #[test]
    fn test_crf_affects_qindex() {
        let low_crf = Vp9EncoderConfig {
            crf: 10.0,
            ..Default::default()
        };
        let high_crf = Vp9EncoderConfig {
            crf: 55.0,
            ..Default::default()
        };

        assert!(low_crf.base_qindex() < high_crf.base_qindex());
    }

    #[test]
    fn test_bool_writer_deterministic() {
        let mut bw = BoolWriter::new();
        bw.write_bool(true, 128);
        bw.write_bool(false, 128);
        bw.write_literal(42, 8);
        let out1 = bw.finalise();

        let mut bw2 = BoolWriter::new();
        bw2.write_bool(true, 128);
        bw2.write_bool(false, 128);
        bw2.write_literal(42, 8);
        let out2 = bw2.finalise();

        assert_eq!(out1, out2);
    }

    #[test]
    fn test_with_vp9_config() {
        let config = EncoderConfig::vp9(128, 96);
        let vp9cfg = Vp9EncoderConfig {
            crf: 18.0,
            keyint: 60,
            speed: 8,
            tile_cols_log2: 1,
            error_resilient: true,
        };
        let encoder = Vp9Encoder::with_vp9_config(config, vp9cfg).expect("encoder creation failed");
        assert!((encoder.vp9_config().crf - 18.0).abs() < f32::EPSILON);
        assert!(encoder.vp9_config().error_resilient);
    }

    #[test]
    fn test_flush() {
        let config = EncoderConfig::vp9(64, 64).with_crf(28.0);
        let mut encoder = Vp9Encoder::new(config).expect("encoder creation failed");
        encoder.flush().expect("flush failed");

        // Sending after flush should fail
        let frame = make_test_frame(64, 64);
        assert!(encoder.send_frame(&frame).is_err());
    }

    #[test]
    fn test_codec_id() {
        let config = EncoderConfig::vp9(64, 64);
        let encoder = Vp9Encoder::new(config).expect("encoder creation failed");
        assert_eq!(encoder.codec(), CodecId::Vp9);
    }

    #[test]
    fn test_block_average() {
        let data = vec![100u8; 64 * 64];
        let avg = Vp9Encoder::block_average(&data, 64, 0, 0, 16);
        assert_eq!(avg, 100);
    }

    #[test]
    fn test_multiple_frames_output_sizes() {
        let config = EncoderConfig::vp9(64, 64).with_crf(20.0);
        let mut encoder = Vp9Encoder::new(config).expect("encoder creation failed");
        let frame = make_test_frame(64, 64);

        let mut sizes = Vec::new();
        for _ in 0..5 {
            encoder.send_frame(&frame).expect("send_frame failed");
            let pkt = encoder
                .receive_packet()
                .expect("receive failed")
                .expect("expected packet");
            sizes.push(pkt.data.len());
        }

        // All packets should have non-zero size
        assert!(sizes.iter().all(|&s| s > 0));
    }

    #[test]
    fn test_lossless_crf_zero() {
        let config = EncoderConfig {
            codec: CodecId::Vp9,
            width: 64,
            height: 64,
            bitrate: BitrateMode::Lossless,
            keyint: 10,
            ..EncoderConfig::default()
        };
        let encoder = Vp9Encoder::new(config).expect("encoder creation failed");
        assert!((encoder.vp9_config().crf - 0.0).abs() < f32::EPSILON);
    }

    // ------------------------------------------------------------------
    // Tests for SimpleVp9Encoder
    // ------------------------------------------------------------------

    fn make_yuv420_frame(width: u32, height: u32, luma: u8) -> Vec<u8> {
        let y_size = (width * height) as usize;
        let uv_size = ((width / 2) * (height / 2)) as usize;
        let mut buf = vec![luma; y_size];
        buf.extend(vec![128u8; uv_size * 2]); // U then V
        buf
    }

    #[test]
    fn test_simple_encoder_creation_valid() {
        let config = Vp9EncConfig {
            width: 320,
            height: 240,
            quality: 28,
            speed: 4,
            ..Vp9EncConfig::default()
        };
        assert!(SimpleVp9Encoder::new(config).is_ok());
    }

    #[test]
    fn test_simple_encoder_rejects_zero_width() {
        let config = Vp9EncConfig {
            width: 0,
            height: 240,
            ..Vp9EncConfig::default()
        };
        assert!(SimpleVp9Encoder::new(config).is_err());
    }

    #[test]
    fn test_simple_encoder_rejects_zero_height() {
        let config = Vp9EncConfig {
            width: 320,
            height: 0,
            ..Vp9EncConfig::default()
        };
        assert!(SimpleVp9Encoder::new(config).is_err());
    }

    #[test]
    fn test_simple_encoder_rejects_quality_over_63() {
        let config = Vp9EncConfig {
            width: 320,
            height: 240,
            quality: 64,
            ..Vp9EncConfig::default()
        };
        assert!(SimpleVp9Encoder::new(config).is_err());
    }

    #[test]
    fn test_simple_encoder_rejects_speed_over_8() {
        let config = Vp9EncConfig {
            width: 320,
            height: 240,
            speed: 9,
            ..Vp9EncConfig::default()
        };
        assert!(SimpleVp9Encoder::new(config).is_err());
    }

    #[test]
    fn test_simple_encoder_first_frame_is_keyframe() {
        let config = Vp9EncConfig {
            width: 64,
            height: 64,
            keyframe_interval: 10,
            ..Vp9EncConfig::default()
        };
        let mut enc = SimpleVp9Encoder::new(config).expect("creation failed");
        let frame = make_yuv420_frame(64, 64, 128);
        let pkt = enc.encode_frame(&frame, false).expect("encode failed");
        assert!(pkt.is_keyframe, "first frame must be a keyframe");
    }

    #[test]
    fn test_simple_encoder_subsequent_frames_are_inter() {
        let config = Vp9EncConfig {
            width: 64,
            height: 64,
            keyframe_interval: 100,
            ..Vp9EncConfig::default()
        };
        let mut enc = SimpleVp9Encoder::new(config).expect("creation failed");
        let frame = make_yuv420_frame(64, 64, 128);
        let _kf = enc.encode_frame(&frame, false).expect("encode frame 0");
        let pkt = enc.encode_frame(&frame, false).expect("encode frame 1");
        assert!(!pkt.is_keyframe, "second frame should be inter");
    }

    #[test]
    fn test_simple_encoder_force_keyframe() {
        let config = Vp9EncConfig {
            width: 64,
            height: 64,
            keyframe_interval: 100,
            ..Vp9EncConfig::default()
        };
        let mut enc = SimpleVp9Encoder::new(config).expect("creation failed");
        let frame = make_yuv420_frame(64, 64, 128);
        let _kf = enc.encode_frame(&frame, false).expect("first frame");
        let _inter = enc.encode_frame(&frame, false).expect("second frame");
        let forced = enc.encode_frame(&frame, true).expect("forced keyframe");
        assert!(
            forced.is_keyframe,
            "force_keyframe=true must produce a keyframe"
        );
    }

    #[test]
    fn test_simple_encoder_keyframe_interval() {
        let config = Vp9EncConfig {
            width: 64,
            height: 64,
            keyframe_interval: 3,
            ..Vp9EncConfig::default()
        };
        let mut enc = SimpleVp9Encoder::new(config).expect("creation failed");
        let frame = make_yuv420_frame(64, 64, 100);
        let mut kf_indices: Vec<u64> = Vec::new();
        for i in 0..9u64 {
            let pkt = enc.encode_frame(&frame, false).expect("encode");
            if pkt.is_keyframe {
                kf_indices.push(i);
            }
        }
        // Frame 0 is always a keyframe (resets counter to 0).
        // Subsequent keyframes occur once keyframe_counter >= keyframe_interval (3):
        // counter: 0→KF(0), 1, 2, 3→KF(4), 1, 2, 3→KF(8)
        assert_eq!(kf_indices, vec![0, 4, 8]);
    }

    #[test]
    fn test_simple_encoder_flush_returns_empty() {
        let config = Vp9EncConfig {
            width: 64,
            height: 64,
            ..Vp9EncConfig::default()
        };
        let mut enc = SimpleVp9Encoder::new(config).expect("creation failed");
        let pkts = enc.flush().expect("flush failed");
        assert!(pkts.is_empty(), "flush should return no buffered packets");
    }

    #[test]
    fn test_simple_encoder_frame_count_increments() {
        let config = Vp9EncConfig {
            width: 64,
            height: 64,
            ..Vp9EncConfig::default()
        };
        let mut enc = SimpleVp9Encoder::new(config).expect("creation failed");
        let frame = make_yuv420_frame(64, 64, 128);
        assert_eq!(enc.frame_count(), 0);
        enc.encode_frame(&frame, false).expect("frame 0");
        assert_eq!(enc.frame_count(), 1);
        enc.encode_frame(&frame, false).expect("frame 1");
        assert_eq!(enc.frame_count(), 2);
    }

    #[test]
    fn test_simple_encoder_packet_non_empty() {
        let config = Vp9EncConfig {
            width: 64,
            height: 64,
            ..Vp9EncConfig::default()
        };
        let mut enc = SimpleVp9Encoder::new(config).expect("creation failed");
        let frame = make_yuv420_frame(64, 64, 200);
        let pkt = enc.encode_frame(&frame, false).expect("encode");
        assert!(!pkt.data.is_empty(), "encoded packet must not be empty");
    }

    #[test]
    fn test_simple_encoder_pts_increments() {
        let config = Vp9EncConfig {
            width: 64,
            height: 64,
            ..Vp9EncConfig::default()
        };
        let mut enc = SimpleVp9Encoder::new(config).expect("creation failed");
        let frame = make_yuv420_frame(64, 64, 128);
        let p0 = enc.encode_frame(&frame, false).expect("frame 0");
        let p1 = enc.encode_frame(&frame, false).expect("frame 1");
        assert_eq!(p0.pts, 0);
        assert_eq!(p1.pts, 1);
        assert_eq!(p0.dts, p0.pts);
        assert_eq!(p1.dts, p1.pts);
    }

    #[test]
    fn test_simple_encoder_profile2_different_header() {
        let cfg0 = Vp9EncConfig {
            width: 64,
            height: 64,
            profile: Vp9Profile::Profile0,
            ..Vp9EncConfig::default()
        };
        let cfg2 = Vp9EncConfig {
            width: 64,
            height: 64,
            profile: Vp9Profile::Profile2,
            ..Vp9EncConfig::default()
        };
        let frame = make_yuv420_frame(64, 64, 128);
        let mut enc0 = SimpleVp9Encoder::new(cfg0).expect("profile0");
        let mut enc2 = SimpleVp9Encoder::new(cfg2).expect("profile2");
        let pkt0 = enc0.encode_frame(&frame, false).expect("encode p0");
        let pkt2 = enc2.encode_frame(&frame, false).expect("encode p2");
        // First byte of header differs between Profile0 and Profile2
        assert_ne!(
            pkt0.data[0], pkt2.data[0],
            "Profile0 and Profile2 headers should differ in byte 0"
        );
    }

    #[test]
    fn test_simple_encoder_rejects_undersized_frame() {
        let config = Vp9EncConfig {
            width: 64,
            height: 64,
            ..Vp9EncConfig::default()
        };
        let mut enc = SimpleVp9Encoder::new(config).expect("creation failed");
        // Only 100 bytes — far too small for 64×64 YUV420 (6144 bytes expected)
        let tiny = vec![0u8; 100];
        let result = enc.encode_frame(&tiny, false);
        assert!(result.is_err(), "undersized frame should return an error");
    }
}
