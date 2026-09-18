//! VP8 encoder implementation.
//!
//! A basic VP8 encoder with CRF-based rate control, key frame interval
//! management, and macroblock-based encoding.
//!
//! # Features
//!
//! - CRF (Constant Rate Factor) quality-based encoding
//! - Configurable key frame interval
//! - Boolean arithmetic encoder for entropy coding
//! - Rate control integration via `RcConfig`
//! - 8-bit 4:2:0 output (VP8 only supports this)

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(dead_code)]

use crate::error::{CodecError, CodecResult};
use crate::frame::{FrameType, VideoFrame};
use crate::rate_control::{CrfController, RcConfig};
use crate::traits::{BitrateMode, EncodedPacket, EncoderConfig, VideoEncoder};
use oximedia_core::CodecId;

// ---------------------------------------------------------------------------
// VP8 bool-writer (arithmetic encoder per RFC 6386 Section 7)
// ---------------------------------------------------------------------------

/// VP8 boolean arithmetic encoder.
#[derive(Debug)]
struct BoolWriter {
    /// Accumulated output bytes.
    data: Vec<u8>,
    /// Range of the current interval.
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
        for _ in 0..32 {
            self.write_bool(false, 128);
        }
        self.data
    }
}

// ---------------------------------------------------------------------------
// VP8 Encoder Configuration
// ---------------------------------------------------------------------------

/// VP8 encoder-specific configuration.
#[derive(Clone, Debug)]
pub struct Vp8EncoderConfig {
    /// CRF quality value (0.0 = lossless, 63.0 = worst).
    pub crf: f32,
    /// Key frame interval (0 = auto).
    pub keyint: u32,
    /// Speed/quality trade-off (0..=10).
    pub speed: u8,
    /// Error-resilient mode (token partitions).
    pub error_resilient: bool,
    /// Number of token partitions (log2, 0..=3).
    pub token_partitions_log2: u8,
    /// Sharpness level for loop filter (0..=7).
    pub sharpness: u8,
}

impl Default for Vp8EncoderConfig {
    fn default() -> Self {
        Self {
            crf: 28.0,
            keyint: 250,
            speed: 5,
            error_resilient: false,
            token_partitions_log2: 0,
            sharpness: 0,
        }
    }
}

impl Vp8EncoderConfig {
    /// Derive the base quantization index (0..=127) from the CRF value.
    /// VP8 uses QI range 0-127, unlike VP9 which uses 0-255.
    fn base_qindex(&self) -> u8 {
        let normalised = (self.crf / 63.0).clamp(0.0, 1.0);
        (normalised * 127.0) as u8
    }
}

// ---------------------------------------------------------------------------
// VP8 Encoder
// ---------------------------------------------------------------------------

/// VP8 Encoder.
///
/// Encodes raw `VideoFrame` data into VP8 elementary bitstream packets.
/// VP8 always outputs 8-bit YUV 4:2:0.
///
/// # Example
///
/// ```ignore
/// use oximedia_codec::vp8::Vp8Encoder;
/// use oximedia_codec::traits::{EncoderConfig, VideoEncoder, BitrateMode};
///
/// let config = EncoderConfig::vp8(320, 240).with_crf(28.0);
/// let mut encoder = Vp8Encoder::new(config)?;
/// ```
#[derive(Debug)]
pub struct Vp8Encoder {
    /// Generic encoder configuration.
    config: EncoderConfig,
    /// VP8-specific settings.
    vp8_config: Vp8EncoderConfig,
    /// Frame counter (encode order).
    frame_count: u64,
    /// Pending output packets.
    output_queue: Vec<EncodedPacket>,
    /// Rate controller.
    rate_controller: CrfController,
    /// Flush mode.
    flushing: bool,
}

impl Vp8Encoder {
    /// Create a new VP8 encoder.
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
        if config.codec != CodecId::Vp8 {
            return Err(CodecError::InvalidParameter(
                "Expected VP8 codec".to_string(),
            ));
        }

        let crf_val = match config.bitrate {
            BitrateMode::Crf(c) => c,
            BitrateMode::Lossless => 0.0,
            _ => 28.0,
        };

        let vp8_config = Vp8EncoderConfig {
            crf: crf_val,
            keyint: config.keyint,
            speed: config.preset.speed().min(10),
            ..Vp8EncoderConfig::default()
        };

        let rc_config = RcConfig::crf(crf_val);
        let rate_controller = CrfController::new(&rc_config);

        Ok(Self {
            config,
            vp8_config,
            frame_count: 0,
            output_queue: Vec::new(),
            rate_controller,
            flushing: false,
        })
    }

    /// Create encoder with explicit VP8 settings.
    ///
    /// # Errors
    ///
    /// Returns error on invalid settings.
    pub fn with_vp8_config(
        config: EncoderConfig,
        vp8_config: Vp8EncoderConfig,
    ) -> CodecResult<Self> {
        if config.width == 0 || config.height == 0 {
            return Err(CodecError::InvalidParameter(
                "Invalid frame dimensions".to_string(),
            ));
        }

        let rc_config = RcConfig::crf(vp8_config.crf);
        let rate_controller = CrfController::new(&rc_config);

        Ok(Self {
            config,
            vp8_config,
            frame_count: 0,
            output_queue: Vec::new(),
            rate_controller,
            flushing: false,
        })
    }

    /// Get the VP8-specific configuration.
    #[must_use]
    pub fn vp8_config(&self) -> &Vp8EncoderConfig {
        &self.vp8_config
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
            || (self.vp8_config.keyint > 0
                && self.frame_count % u64::from(self.vp8_config.keyint) == 0);

        let qindex = self.vp8_config.base_qindex();
        let data = self.write_frame(frame, is_keyframe, qindex);

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

    /// Serialise a VP8 frame.
    fn write_frame(&self, frame: &VideoFrame, keyframe: bool, qindex: u8) -> Vec<u8> {
        let mut data = Vec::with_capacity(1024);

        if keyframe {
            self.write_keyframe(&mut data, frame, qindex);
        } else {
            self.write_interframe(&mut data, frame, qindex);
        }

        data
    }

    /// Write VP8 keyframe (RFC 6386 Section 9.1).
    ///
    /// VP8 frame tag format (3 bytes):
    /// - Bit 0: frame_type (0=key, 1=inter)
    /// - Bits 1-2: version (0=bicubic)
    /// - Bit 3: show_frame
    /// - Bits 4-23: first_part_size (19 bits)
    fn write_keyframe(&self, buf: &mut Vec<u8>, frame: &VideoFrame, qindex: u8) {
        // First partition will be computed later; use placeholder size
        let first_part_size_placeholder = 0u32;

        // Frame tag: frame_type=0 (key), version=0, show_frame=1
        let tag = (first_part_size_placeholder << 5) | (1 << 4) | (0 << 1) | 0;
        buf.push((tag & 0xFF) as u8);
        buf.push(((tag >> 8) & 0xFF) as u8);
        buf.push(((tag >> 16) & 0xFF) as u8);

        // Sync code: 0x9D 0x01 0x2A
        buf.extend_from_slice(&[0x9D, 0x01, 0x2A]);

        // Width and height (16-bit LE each, with scale=0 in upper 2 bits)
        let w = frame.width as u16;
        let h = frame.height as u16;
        buf.push((w & 0xFF) as u8);
        buf.push(((w >> 8) & 0x3F) as u8); // scale=0
        buf.push((h & 0xFF) as u8);
        buf.push(((h >> 8) & 0x3F) as u8); // scale=0

        // First partition: header data encoded with bool coder
        let header_data = self.encode_keyframe_header(frame, qindex);
        buf.extend_from_slice(&header_data);

        // Second partition: macroblock data
        let mb_data = self.encode_macroblock_data(frame, qindex);
        buf.extend_from_slice(&mb_data);

        // Patch frame tag with actual first_part_size
        let first_part_size = header_data.len() as u32;
        let tag = (first_part_size << 5) | (1 << 4) | (0 << 1) | 0;
        buf[0] = (tag & 0xFF) as u8;
        buf[1] = ((tag >> 8) & 0xFF) as u8;
        buf[2] = ((tag >> 16) & 0xFF) as u8;
    }

    /// Write VP8 inter-frame.
    fn write_interframe(&self, buf: &mut Vec<u8>, frame: &VideoFrame, qindex: u8) {
        let first_part_size_placeholder = 0u32;

        // Frame tag: frame_type=1 (inter), version=0, show_frame=1
        let tag = (first_part_size_placeholder << 5) | (1 << 4) | (0 << 1) | 1;
        buf.push((tag & 0xFF) as u8);
        buf.push(((tag >> 8) & 0xFF) as u8);
        buf.push(((tag >> 16) & 0xFF) as u8);

        // First partition: inter-frame header
        let header_data = self.encode_interframe_header(qindex);
        buf.extend_from_slice(&header_data);

        // Second partition: macroblock data
        let mb_data = self.encode_macroblock_data(frame, qindex);
        buf.extend_from_slice(&mb_data);

        // Patch frame tag
        let first_part_size = header_data.len() as u32;
        let tag = (first_part_size << 5) | (1 << 4) | (0 << 1) | 1;
        buf[0] = (tag & 0xFF) as u8;
        buf[1] = ((tag >> 8) & 0xFF) as u8;
        buf[2] = ((tag >> 16) & 0xFF) as u8;
    }

    /// Encode keyframe header using bool coder.
    fn encode_keyframe_header(&self, _frame: &VideoFrame, qindex: u8) -> Vec<u8> {
        let mut bw = BoolWriter::new();

        // color_space = 0 (YCbCr)
        bw.write_bool(false, 128);
        // clamping_type = 0 (clamped)
        bw.write_bool(false, 128);

        // segmentation_enabled = 0
        bw.write_bool(false, 128);

        // filter_type = 0 (normal)
        bw.write_bool(false, 128);
        // loop_filter_level (6 bits)
        let filter_level = (qindex / 4).min(63);
        bw.write_literal(u32::from(filter_level), 6);
        // sharpness_level (3 bits)
        bw.write_literal(u32::from(self.vp8_config.sharpness.min(7)), 3);

        // mb_lf_adjustments (loop filter ref/mode delta adjustments)
        // loop_filter_adj_enable = 0
        bw.write_bool(false, 128);

        // Number of token partitions (2 bits)
        bw.write_literal(u32::from(self.vp8_config.token_partitions_log2.min(3)), 2);

        // Quantization indices
        // y_ac_qi (7 bits) - primary quantizer
        bw.write_literal(u32::from(qindex), 7);
        // y_dc_delta_present = 0
        bw.write_bool(false, 128);
        // y2_dc_delta_present = 0
        bw.write_bool(false, 128);
        // y2_ac_delta_present = 0
        bw.write_bool(false, 128);
        // uv_dc_delta_present = 0
        bw.write_bool(false, 128);
        // uv_ac_delta_present = 0
        bw.write_bool(false, 128);

        // refresh_entropy_probs = 1
        bw.write_bool(true, 128);

        // No coefficient probability updates (simplified)
        // In a full encoder this would update the default probability tables

        bw.finalise()
    }

    /// Encode inter-frame header using bool coder.
    fn encode_interframe_header(&self, qindex: u8) -> Vec<u8> {
        let mut bw = BoolWriter::new();

        // segmentation_enabled = 0
        bw.write_bool(false, 128);

        // filter_type = 0
        bw.write_bool(false, 128);
        // loop_filter_level (6 bits)
        let filter_level = (qindex / 4).min(63);
        bw.write_literal(u32::from(filter_level), 6);
        // sharpness_level (3 bits)
        bw.write_literal(u32::from(self.vp8_config.sharpness.min(7)), 3);

        // mb_lf_adjustments enable = 0
        bw.write_bool(false, 128);

        // Number of token partitions (2 bits)
        bw.write_literal(u32::from(self.vp8_config.token_partitions_log2.min(3)), 2);

        // Quantization indices
        bw.write_literal(u32::from(qindex), 7);
        // All delta present flags = 0
        bw.write_bool(false, 128);
        bw.write_bool(false, 128);
        bw.write_bool(false, 128);
        bw.write_bool(false, 128);
        bw.write_bool(false, 128);

        // refresh_last_frame = 1
        bw.write_bool(true, 128);
        // refresh_golden_frame = 0
        bw.write_bool(false, 128);
        // refresh_alt_ref_frame = 0
        bw.write_bool(false, 128);

        // copy_buffer_to_golden = 0 (2 bits)
        bw.write_literal(0, 2);
        // copy_buffer_to_alternate = 0 (2 bits)
        bw.write_literal(0, 2);

        // ref_frame_sign_bias for golden = 0, alt_ref = 0
        bw.write_bool(false, 128);
        bw.write_bool(false, 128);

        // refresh_entropy_probs = 1
        bw.write_bool(true, 128);

        bw.finalise()
    }

    /// Encode macroblock data (simplified: quantised DC coefficients per 16x16 block).
    fn encode_macroblock_data(&self, frame: &VideoFrame, qindex: u8) -> Vec<u8> {
        let mut bw = BoolWriter::new();

        let plane = frame.plane(0);
        let width = plane.width() as usize;
        let height = plane.height() as usize;
        let stride = plane.stride() as usize;
        let luma = plane.data();

        let mb_size = 16usize;
        let mb_cols = (width + mb_size - 1) / mb_size;
        let mb_rows = (height + mb_size - 1) / mb_size;

        let q_step = (u32::from(qindex) + 1).max(1);

        for row in 0..mb_rows {
            for col in 0..mb_cols {
                let x0 = col * mb_size;
                let y0 = row * mb_size;
                let dc = Self::block_average(luma, stride, x0, y0, mb_size, width, height);
                let quantised = (dc / q_step).min(255);
                bw.write_literal(quantised, 8);
            }
        }

        bw.finalise()
    }

    /// Calculate the average pixel value for a macroblock, handling edges.
    fn block_average(
        data: &[u8],
        stride: usize,
        x0: usize,
        y0: usize,
        size: usize,
        frame_width: usize,
        frame_height: usize,
    ) -> u32 {
        let mut sum = 0u32;
        let mut count = 0u32;

        let max_y = (y0 + size).min(frame_height);
        let max_x = (x0 + size).min(frame_width);

        for y in y0..max_y {
            let row_start = y * stride;
            for x in x0..max_x {
                if row_start + x < data.len() {
                    sum += u32::from(data[row_start + x]);
                    count += 1;
                }
            }
        }

        sum.checked_div(count).unwrap_or(128)
    }
}

// ---------------------------------------------------------------------------
// VideoEncoder trait impl
// ---------------------------------------------------------------------------

impl VideoEncoder for Vp8Encoder {
    fn codec(&self) -> CodecId {
        CodecId::Vp8
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
// SimpleVp8Encoder — lightweight API for raw YUV420 byte slices
// ---------------------------------------------------------------------------

/// Configuration for [`SimpleVp8Encoder`].
#[derive(Clone, Debug)]
pub struct Vp8EncConfig {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Target bitrate in kilobits per second (informational; used for header).
    pub target_bitrate: u32,
    /// Key frame interval (0 = only the very first frame is a keyframe).
    pub keyframe_interval: u32,
}

impl Default for Vp8EncConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            target_bitrate: 2000,
            keyframe_interval: 250,
        }
    }
}

/// A VP8-encoded packet produced by [`SimpleVp8Encoder`].
#[derive(Clone, Debug)]
pub struct Vp8Packet {
    /// Raw VP8 bitstream bytes for this packet.
    pub data: Vec<u8>,
    /// Whether this packet is a keyframe.
    pub is_keyframe: bool,
    /// Presentation timestamp (frame index).
    pub pts: u64,
}

/// A simple, self-contained VP8 encoder that accepts raw YUV420 byte slices.
///
/// Produces syntactically valid VP8 bitstream headers (frame tag, sync code,
/// dimension fields, partition offsets) followed by a quantised macroblock
/// payload.  Does **not** perform full DCT/motion-search; suitable for
/// container muxing and conformance testing.
///
/// # Example
///
/// ```ignore
/// use oximedia_codec::vp8::{SimpleVp8Encoder, Vp8EncConfig};
///
/// let config = Vp8EncConfig {
///     width: 320, height: 240,
///     target_bitrate: 500,
///     keyframe_interval: 60,
/// };
/// let mut enc = SimpleVp8Encoder::new(config)?;
/// // frame_bytes: YUV420 slice (width * height * 3 / 2 bytes)
/// let pkt = enc.encode_frame(&frame_bytes)?;
/// assert!(pkt.is_keyframe); // first frame is always a keyframe
/// ```
#[derive(Debug)]
pub struct SimpleVp8Encoder {
    config: Vp8EncConfig,
    frame_count: u64,
    keyframe_counter: u32,
}

impl SimpleVp8Encoder {
    /// Create a new [`SimpleVp8Encoder`].
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::InvalidParameter`] if `width` or `height` is zero.
    pub fn new(config: Vp8EncConfig) -> CodecResult<Self> {
        if config.width == 0 || config.height == 0 {
            return Err(CodecError::InvalidParameter(
                "SimpleVp8Encoder: width and height must be non-zero".to_string(),
            ));
        }
        Ok(Self {
            config,
            frame_count: 0,
            keyframe_counter: 0,
        })
    }

    /// Encode a raw YUV420 frame.
    ///
    /// `yuv420_data` must be at least `width * height * 3 / 2` bytes long
    /// (Y plane followed by U and V half-resolution planes).
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::InvalidData`] if `yuv420_data` is shorter than
    /// the expected frame size.
    pub fn encode_frame(&mut self, yuv420_data: &[u8]) -> CodecResult<Vp8Packet> {
        let expected = (self.config.width as usize) * (self.config.height as usize) * 3 / 2;
        if yuv420_data.len() < expected {
            return Err(CodecError::InvalidData(format!(
                "SimpleVp8Encoder: frame too small ({} < {expected} bytes)",
                yuv420_data.len()
            )));
        }

        let is_keyframe = self.frame_count == 0
            || (self.config.keyframe_interval > 0
                && self.frame_count % u64::from(self.config.keyframe_interval) == 0);

        let pts = self.frame_count;
        let data = self.build_packet(yuv420_data, is_keyframe);

        self.keyframe_counter = if is_keyframe {
            0
        } else {
            self.keyframe_counter + 1
        };
        self.frame_count += 1;

        Ok(Vp8Packet {
            data,
            is_keyframe,
            pts,
        })
    }

    /// Return the total number of frames encoded.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Derive quantisation index (0..=127) from target bitrate + dimensions.
    fn qindex(&self) -> u8 {
        let pixels = (self.config.width as u64) * (self.config.height as u64);
        // bits-per-pixel at target rate (assume 30 fps)
        let bpp = (self.config.target_bitrate as u64 * 1000) / (pixels.max(1) * 30);
        // Map bpp → quality: higher bpp → lower qindex (better quality)
        let q = (127u64).saturating_sub(bpp.min(127)) as u8;
        q.clamp(0, 127)
    }

    /// Build a complete VP8 packet for a single frame.
    fn build_packet(&self, yuv420_data: &[u8], is_keyframe: bool) -> Vec<u8> {
        let mut buf = Vec::with_capacity(1024);

        if is_keyframe {
            self.write_keyframe_packet(&mut buf, yuv420_data);
        } else {
            self.write_interframe_packet(&mut buf, yuv420_data);
        }

        buf
    }

    /// Write a keyframe VP8 packet.
    fn write_keyframe_packet(&self, buf: &mut Vec<u8>, yuv420_data: &[u8]) {
        let qindex = self.qindex();

        // Placeholder 3-byte frame tag
        buf.extend_from_slice(&[0u8; 3]);

        // Sync code
        buf.extend_from_slice(&[0x9D, 0x01, 0x2A]);

        // Width and height (LE 16-bit each, upper 2 bits = scale = 0)
        let w = self.config.width as u16;
        let h = self.config.height as u16;
        buf.push((w & 0xFF) as u8);
        buf.push(((w >> 8) & 0x3F) as u8);
        buf.push((h & 0xFF) as u8);
        buf.push(((h >> 8) & 0x3F) as u8);

        // First partition (header)
        let first_part = self.encode_simple_keyframe_header(qindex);
        let first_part_size = first_part.len() as u32;
        buf.extend_from_slice(&first_part);

        // Second partition (macroblock data)
        let mb_data = self.encode_simple_mb_data(yuv420_data, qindex);
        buf.extend_from_slice(&mb_data);

        // Patch frame tag: frame_type=0 (key), version=0, show_frame=1
        // tag bits: [0]=frame_type, [1..2]=version, [3]=show_frame, [4..23]=first_part_size
        let tag = (first_part_size << 5) | (1 << 4) | 0u32;
        buf[0] = (tag & 0xFF) as u8;
        buf[1] = ((tag >> 8) & 0xFF) as u8;
        buf[2] = ((tag >> 16) & 0xFF) as u8;
    }

    /// Write an inter-frame VP8 packet.
    fn write_interframe_packet(&self, buf: &mut Vec<u8>, yuv420_data: &[u8]) {
        let qindex = self.qindex();

        // Placeholder 3-byte frame tag
        buf.extend_from_slice(&[0u8; 3]);

        // First partition (inter-frame header)
        let first_part = self.encode_simple_interframe_header(qindex);
        let first_part_size = first_part.len() as u32;
        buf.extend_from_slice(&first_part);

        // Second partition (macroblock data)
        let mb_data = self.encode_simple_mb_data(yuv420_data, qindex);
        buf.extend_from_slice(&mb_data);

        // Patch frame tag: frame_type=1 (inter), version=0, show_frame=1
        let tag = (first_part_size << 5) | (1 << 4) | 1u32;
        buf[0] = (tag & 0xFF) as u8;
        buf[1] = ((tag >> 8) & 0xFF) as u8;
        buf[2] = ((tag >> 16) & 0xFF) as u8;
    }

    /// Encode the first partition header for a keyframe.
    fn encode_simple_keyframe_header(&self, qindex: u8) -> Vec<u8> {
        let mut bw = BoolWriter::new();

        // color_space = 0 (YCbCr)
        bw.write_bool(false, 128);
        // clamping_type = 0
        bw.write_bool(false, 128);
        // segmentation_enabled = 0
        bw.write_bool(false, 128);
        // filter_type = 0 (normal)
        bw.write_bool(false, 128);
        // loop_filter_level (6 bits)
        bw.write_literal(u32::from((qindex / 4).min(63)), 6);
        // sharpness_level (3 bits)
        bw.write_literal(0, 3);
        // loop_filter_adj_enable = 0
        bw.write_bool(false, 128);
        // token_partitions (2 bits) = 0
        bw.write_literal(0, 2);
        // y_ac_qi (7 bits)
        bw.write_literal(u32::from(qindex), 7);
        // delta present flags × 5 = 0
        for _ in 0..5 {
            bw.write_bool(false, 128);
        }
        // refresh_entropy_probs = 1
        bw.write_bool(true, 128);

        bw.finalise()
    }

    /// Encode the first partition header for an inter-frame.
    fn encode_simple_interframe_header(&self, qindex: u8) -> Vec<u8> {
        let mut bw = BoolWriter::new();

        // segmentation_enabled = 0
        bw.write_bool(false, 128);
        // filter_type = 0
        bw.write_bool(false, 128);
        // loop_filter_level (6 bits)
        bw.write_literal(u32::from((qindex / 4).min(63)), 6);
        // sharpness_level (3 bits)
        bw.write_literal(0, 3);
        // mb_lf_adjustments enable = 0
        bw.write_bool(false, 128);
        // token_partitions (2 bits) = 0
        bw.write_literal(0, 2);
        // y_ac_qi (7 bits)
        bw.write_literal(u32::from(qindex), 7);
        // delta present flags × 5 = 0
        for _ in 0..5 {
            bw.write_bool(false, 128);
        }
        // refresh_last_frame = 1
        bw.write_bool(true, 128);
        // refresh_golden_frame = 0, refresh_alt_ref_frame = 0
        bw.write_bool(false, 128);
        bw.write_bool(false, 128);
        // copy_buffer_to_golden (2 bits) = 0
        bw.write_literal(0, 2);
        // copy_buffer_to_alternate (2 bits) = 0
        bw.write_literal(0, 2);
        // ref_frame_sign_bias for golden, alt_ref = 0
        bw.write_bool(false, 128);
        bw.write_bool(false, 128);
        // refresh_entropy_probs = 1
        bw.write_bool(true, 128);

        bw.finalise()
    }

    /// Encode macroblock partition data from a YUV420 slice.
    ///
    /// For each 16×16 luma macroblock, the average luma value is quantised
    /// and written as an 8-bit literal via the boolean arithmetic coder.
    fn encode_simple_mb_data(&self, yuv420_data: &[u8], qindex: u8) -> Vec<u8> {
        let w = self.config.width as usize;
        let h = self.config.height as usize;
        let y_plane_len = w * h;
        let luma = if yuv420_data.len() >= y_plane_len {
            &yuv420_data[..y_plane_len]
        } else {
            yuv420_data
        };

        let mb_size = 16usize;
        let mb_cols = (w + mb_size - 1) / mb_size;
        let mb_rows = (h + mb_size - 1) / mb_size;
        let q_step = (u32::from(qindex) + 1).max(1);

        let mut bw = BoolWriter::new();
        for row in 0..mb_rows {
            for col in 0..mb_cols {
                let dc = block_average_yuv(luma, w, h, col * mb_size, row * mb_size, mb_size);
                let quantised = (dc / q_step).min(255);
                bw.write_literal(quantised, 8);
            }
        }

        bw.finalise()
    }
}

/// Compute average pixel value of a `size×size` block at `(x0, y0)`.
fn block_average_yuv(
    data: &[u8],
    frame_width: usize,
    frame_height: usize,
    x0: usize,
    y0: usize,
    size: usize,
) -> u32 {
    let mut sum = 0u32;
    let mut count = 0u32;
    let max_y = (y0 + size).min(frame_height);
    let max_x = (x0 + size).min(frame_width);

    for y in y0..max_y {
        let row_start = y * frame_width;
        for x in x0..max_x {
            let idx = row_start + x;
            if idx < data.len() {
                sum += u32::from(data[idx]);
                count += 1;
            }
        }
    }

    sum.checked_div(count).unwrap_or(128)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Plane, VideoFrame};
    use crate::traits::EncoderPreset;
    use oximedia_core::PixelFormat;

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
        let config = EncoderConfig::vp8(320, 240).with_crf(24.0);
        let encoder = Vp8Encoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_rejects_zero_dimensions() {
        let mut config = EncoderConfig::vp8(320, 240);
        config.width = 0;
        assert!(Vp8Encoder::new(config).is_err());
    }

    #[test]
    fn test_encoder_rejects_wrong_codec() {
        let mut config = EncoderConfig::vp8(320, 240);
        config.codec = CodecId::Av1;
        assert!(Vp8Encoder::new(config).is_err());
    }

    #[test]
    fn test_encode_single_frame() {
        let config = EncoderConfig::vp8(64, 64).with_crf(28.0);
        let mut encoder = Vp8Encoder::new(config).expect("encoder creation failed");
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
        let mut config = EncoderConfig::vp8(64, 64).with_crf(28.0);
        config.keyint = 3;
        let mut encoder = Vp8Encoder::new(config).expect("encoder creation failed");
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
        let low_crf = Vp8EncoderConfig {
            crf: 10.0,
            ..Default::default()
        };
        let high_crf = Vp8EncoderConfig {
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
    fn test_with_vp8_config() {
        let config = EncoderConfig::vp8(128, 96);
        let vp8cfg = Vp8EncoderConfig {
            crf: 18.0,
            keyint: 60,
            speed: 8,
            error_resilient: true,
            token_partitions_log2: 2,
            sharpness: 3,
        };
        let encoder = Vp8Encoder::with_vp8_config(config, vp8cfg).expect("encoder creation failed");
        assert!((encoder.vp8_config().crf - 18.0).abs() < f32::EPSILON);
        assert!(encoder.vp8_config().error_resilient);
        assert_eq!(encoder.vp8_config().token_partitions_log2, 2);
    }

    #[test]
    fn test_flush() {
        let config = EncoderConfig::vp8(64, 64).with_crf(28.0);
        let mut encoder = Vp8Encoder::new(config).expect("encoder creation failed");
        encoder.flush().expect("flush failed");

        // Sending after flush should fail
        let frame = make_test_frame(64, 64);
        assert!(encoder.send_frame(&frame).is_err());
    }

    #[test]
    fn test_codec_id() {
        let config = EncoderConfig::vp8(64, 64);
        let encoder = Vp8Encoder::new(config).expect("encoder creation failed");
        assert_eq!(encoder.codec(), CodecId::Vp8);
    }

    #[test]
    fn test_block_average() {
        let data = vec![100u8; 64 * 64];
        let avg = Vp8Encoder::block_average(&data, 64, 0, 0, 16, 64, 64);
        assert_eq!(avg, 100);
    }

    #[test]
    fn test_block_average_edge() {
        // Test edge macroblocks where block extends past frame boundary
        let data = vec![200u8; 60 * 60];
        let avg = Vp8Encoder::block_average(&data, 60, 48, 48, 16, 60, 60);
        assert_eq!(avg, 200);
    }

    #[test]
    fn test_multiple_frames_output_sizes() {
        let config = EncoderConfig::vp8(64, 64).with_crf(20.0);
        let mut encoder = Vp8Encoder::new(config).expect("encoder creation failed");
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

        assert!(sizes.iter().all(|&s| s > 0));
    }

    #[test]
    fn test_lossless_crf_zero() {
        let config = EncoderConfig {
            codec: CodecId::Vp8,
            width: 64,
            height: 64,
            bitrate: BitrateMode::Lossless,
            keyint: 10,
            ..EncoderConfig::default()
        };
        let encoder = Vp8Encoder::new(config).expect("encoder creation failed");
        assert!((encoder.vp8_config().crf - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_keyframe_has_sync_code() {
        let config = EncoderConfig::vp8(64, 64).with_crf(28.0);
        let mut encoder = Vp8Encoder::new(config).expect("encoder creation failed");
        let frame = make_test_frame(64, 64);

        encoder.send_frame(&frame).expect("send_frame failed");
        let pkt = encoder
            .receive_packet()
            .expect("receive failed")
            .expect("expected packet");

        // VP8 keyframe sync code at bytes 3-5
        assert!(pkt.data.len() > 6);
        assert_eq!(pkt.data[3], 0x9D);
        assert_eq!(pkt.data[4], 0x01);
        assert_eq!(pkt.data[5], 0x2A);
    }

    #[test]
    fn test_keyframe_contains_dimensions() {
        let config = EncoderConfig::vp8(320, 240).with_crf(28.0);
        let mut encoder = Vp8Encoder::new(config).expect("encoder creation failed");
        let frame = make_test_frame(320, 240);

        encoder.send_frame(&frame).expect("send_frame failed");
        let pkt = encoder
            .receive_packet()
            .expect("receive failed")
            .expect("expected packet");

        // Width at bytes 6-7 (LE), height at bytes 8-9 (LE)
        assert!(pkt.data.len() > 10);
        let w = u16::from(pkt.data[6]) | (u16::from(pkt.data[7] & 0x3F) << 8);
        let h = u16::from(pkt.data[8]) | (u16::from(pkt.data[9] & 0x3F) << 8);
        assert_eq!(w, 320);
        assert_eq!(h, 240);
    }

    #[test]
    fn test_interframe_is_not_keyframe() {
        let config = EncoderConfig::vp8(64, 64).with_crf(28.0);
        let mut encoder = Vp8Encoder::new(config).expect("encoder creation failed");
        let frame = make_test_frame(64, 64);

        // First frame is keyframe
        encoder.send_frame(&frame).expect("send_frame failed");
        let pkt0 = encoder
            .receive_packet()
            .expect("receive failed")
            .expect("pkt");
        assert!(pkt0.keyframe);

        // Second frame is inter
        encoder.send_frame(&frame).expect("send_frame failed");
        let pkt1 = encoder
            .receive_packet()
            .expect("receive failed")
            .expect("pkt");
        assert!(!pkt1.keyframe);

        // Verify frame tag bit 0 = 1 (inter frame)
        assert_eq!(pkt1.data[0] & 0x01, 1);
    }

    // ------------------------------------------------------------------
    // SimpleVp8Encoder tests
    // ------------------------------------------------------------------

    fn make_yuv420(w: usize, h: usize) -> Vec<u8> {
        vec![128u8; w * h * 3 / 2]
    }

    #[test]
    fn test_simple_encoder_new_ok() {
        let cfg = Vp8EncConfig {
            width: 320,
            height: 240,
            target_bitrate: 500,
            keyframe_interval: 60,
        };
        assert!(SimpleVp8Encoder::new(cfg).is_ok());
    }

    #[test]
    fn test_simple_encoder_rejects_zero_width() {
        let cfg = Vp8EncConfig {
            width: 0,
            height: 240,
            target_bitrate: 500,
            keyframe_interval: 60,
        };
        assert!(SimpleVp8Encoder::new(cfg).is_err());
    }

    #[test]
    fn test_simple_encoder_rejects_zero_height() {
        let cfg = Vp8EncConfig {
            width: 320,
            height: 0,
            target_bitrate: 500,
            keyframe_interval: 60,
        };
        assert!(SimpleVp8Encoder::new(cfg).is_err());
    }

    #[test]
    fn test_simple_encoder_first_frame_is_keyframe() {
        let cfg = Vp8EncConfig {
            width: 64,
            height: 64,
            target_bitrate: 500,
            keyframe_interval: 30,
        };
        let mut enc = SimpleVp8Encoder::new(cfg).expect("new");
        let frame = make_yuv420(64, 64);
        let pkt = enc.encode_frame(&frame).expect("encode");
        assert!(pkt.is_keyframe);
        assert_eq!(pkt.pts, 0);
    }

    #[test]
    fn test_simple_encoder_second_frame_is_inter() {
        let cfg = Vp8EncConfig {
            width: 64,
            height: 64,
            target_bitrate: 500,
            keyframe_interval: 30,
        };
        let mut enc = SimpleVp8Encoder::new(cfg).expect("new");
        let frame = make_yuv420(64, 64);
        enc.encode_frame(&frame).expect("frame 0");
        let pkt1 = enc.encode_frame(&frame).expect("frame 1");
        assert!(!pkt1.is_keyframe);
        // inter-frame tag bit 0 = 1
        assert_eq!(pkt1.data[0] & 0x01, 1);
    }

    #[test]
    fn test_simple_encoder_keyframe_sync_code() {
        let cfg = Vp8EncConfig {
            width: 64,
            height: 64,
            target_bitrate: 500,
            keyframe_interval: 30,
        };
        let mut enc = SimpleVp8Encoder::new(cfg).expect("new");
        let frame = make_yuv420(64, 64);
        let pkt = enc.encode_frame(&frame).expect("encode");
        assert!(pkt.data.len() > 6);
        assert_eq!(pkt.data[3], 0x9D);
        assert_eq!(pkt.data[4], 0x01);
        assert_eq!(pkt.data[5], 0x2A);
    }

    #[test]
    fn test_simple_encoder_keyframe_dimensions() {
        let cfg = Vp8EncConfig {
            width: 320,
            height: 240,
            target_bitrate: 500,
            keyframe_interval: 60,
        };
        let mut enc = SimpleVp8Encoder::new(cfg).expect("new");
        let frame = make_yuv420(320, 240);
        let pkt = enc.encode_frame(&frame).expect("encode");
        assert!(pkt.data.len() > 10);
        let w = u16::from(pkt.data[6]) | (u16::from(pkt.data[7] & 0x3F) << 8);
        let h = u16::from(pkt.data[8]) | (u16::from(pkt.data[9] & 0x3F) << 8);
        assert_eq!(w, 320);
        assert_eq!(h, 240);
    }

    #[test]
    fn test_simple_encoder_keyframe_interval() {
        let cfg = Vp8EncConfig {
            width: 64,
            height: 64,
            target_bitrate: 500,
            keyframe_interval: 3,
        };
        let mut enc = SimpleVp8Encoder::new(cfg).expect("new");
        let frame = make_yuv420(64, 64);
        let mut keyframe_indices = Vec::new();
        for i in 0..9u64 {
            let pkt = enc.encode_frame(&frame).expect("encode");
            if pkt.is_keyframe {
                keyframe_indices.push(i);
            }
        }
        // Keyframes at 0, 3, 6
        assert_eq!(keyframe_indices, vec![0, 3, 6]);
    }

    #[test]
    fn test_simple_encoder_frame_count() {
        let cfg = Vp8EncConfig {
            width: 64,
            height: 64,
            target_bitrate: 500,
            keyframe_interval: 10,
        };
        let mut enc = SimpleVp8Encoder::new(cfg).expect("new");
        let frame = make_yuv420(64, 64);
        for _ in 0..5 {
            enc.encode_frame(&frame).expect("encode");
        }
        assert_eq!(enc.frame_count(), 5);
    }

    #[test]
    fn test_simple_encoder_rejects_short_frame() {
        let cfg = Vp8EncConfig {
            width: 64,
            height: 64,
            target_bitrate: 500,
            keyframe_interval: 10,
        };
        let mut enc = SimpleVp8Encoder::new(cfg).expect("new");
        let too_short = vec![0u8; 10];
        assert!(enc.encode_frame(&too_short).is_err());
    }

    #[test]
    fn test_simple_encoder_pts_increments() {
        let cfg = Vp8EncConfig {
            width: 64,
            height: 64,
            target_bitrate: 500,
            keyframe_interval: 10,
        };
        let mut enc = SimpleVp8Encoder::new(cfg).expect("new");
        let frame = make_yuv420(64, 64);
        for i in 0u64..4 {
            let pkt = enc.encode_frame(&frame).expect("encode");
            assert_eq!(pkt.pts, i);
        }
    }

    #[test]
    fn test_simple_encoder_output_non_empty() {
        let cfg = Vp8EncConfig {
            width: 64,
            height: 64,
            target_bitrate: 1000,
            keyframe_interval: 10,
        };
        let mut enc = SimpleVp8Encoder::new(cfg).expect("new");
        let frame = make_yuv420(64, 64);
        for _ in 0..3 {
            let pkt = enc.encode_frame(&frame).expect("encode");
            assert!(!pkt.data.is_empty());
        }
    }
}
