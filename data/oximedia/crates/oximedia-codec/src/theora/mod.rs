// Copyright 2024 The OxiMedia Project Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Theora video codec implementation.
//!
//! This module provides a complete Theora decoder and encoder implementation
//! following RFC 7845 and the Theora specification version 1.1.
//!
//! Theora is a free and open video codec developed by the Xiph.Org Foundation.
//! It is based on VP3 and provides competitive compression with patent-free licensing.
//!
//! # Features
//!
//! ## Decoder
//! - Ogg Theora bitstream parsing (RFC 7845)
//! - Intra and inter frame decoding
//! - VP3-compatible DCT/IDCT transforms
//! - Huffman entropy coding
//! - Reference frame management (last and golden frames)
//! - Block-based motion compensation with half-pixel precision
//! - Loop filtering
//!
//! ## Encoder
//! - I-frame and P-frame encoding
//! - Rate control with target bitrate
//! - Quality settings (0-63 scale)
//! - Block mode decision
//! - Diamond search motion estimation
//! - Motion vector prediction
//!
//! # Example
//!
//! ```
//! use oximedia_codec::theora::{TheoraDecoder, TheoraEncoder, TheoraConfig};
//! use oximedia_codec::traits::{VideoDecoder, VideoEncoder};
//! use oximedia_core::{CodecId, PixelFormat};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Decoding
//! let mut decoder = TheoraDecoder::new(1920, 1080, PixelFormat::Yuv420p)?;
//! // decoder.send_packet(&packet_data, pts)?;
//! // if let Some(frame) = decoder.receive_frame()? {
//! //     // Process frame
//! // }
//!
//! // Encoding
//! let config = TheoraConfig::new(1920, 1080)
//!     .with_quality(48)
//!     .with_target_bitrate(2_000_000);
//! let mut encoder = TheoraEncoder::new(config)?;
//! # Ok(())
//! # }
//! ```

pub mod bitstream;
pub mod block_decision;
pub mod encoder_settings;
pub mod frame_header;
pub mod huffman;
pub mod intra_pred;
pub mod loop_filter;
pub mod motion;
pub mod quant;
pub mod rate_ctrl;
pub mod stats;
pub mod tables;
pub mod transform;
pub mod two_pass;

use crate::error::{CodecError, CodecResult};
use crate::frame::{FrameType, VideoFrame};
use crate::traits::{BitrateMode, EncodedPacket, EncoderConfig, VideoDecoder, VideoEncoder};
use bitstream::{BitstreamReader, BitstreamWriter};
use huffman::{TheoraTokenDecoder, TheoraTokenEncoder};
use motion::{
    motion_compensate_8x8, motion_estimation_diamond, predict_motion_vector, MotionVector,
};
use oximedia_core::{CodecId, PixelFormat, Rational, Timestamp};
use tables::*;
use transform::*;

/// Theora decoder configuration.
#[derive(Debug, Clone)]
pub struct TheoraDecoderConfig {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel format (only YUV420p supported).
    pub pixel_format: PixelFormat,
}

/// Theora decoder.
///
/// Decodes Theora video bitstreams to raw video frames.
pub struct TheoraDecoder {
    /// Configuration.
    config: TheoraDecoderConfig,
    /// Width in blocks.
    blocks_w: usize,
    /// Height in blocks.
    blocks_h: usize,
    /// Last reference frame.
    last_frame: Option<DecodedFrame>,
    /// Golden reference frame.
    golden_frame: Option<DecodedFrame>,
    /// Huffman token decoder.
    token_decoder: TheoraTokenDecoder,
    /// Current quantization matrices.
    quant_matrices: QuantizationMatrices,
    /// Frame count.
    frame_count: u64,
    /// Pending output frame.
    pending_frame: Option<VideoFrame>,
}

/// Decoded frame with YUV planes.
struct DecodedFrame {
    y_plane: Vec<u8>,
    u_plane: Vec<u8>,
    v_plane: Vec<u8>,
    width: usize,
    height: usize,
    y_stride: usize,
    uv_stride: usize,
}

impl DecodedFrame {
    fn new(width: usize, height: usize) -> Self {
        let y_stride = width;
        let uv_stride = width / 2;

        Self {
            y_plane: vec![128; y_stride * height],
            u_plane: vec![128; uv_stride * height / 2],
            v_plane: vec![128; uv_stride * height / 2],
            width,
            height,
            y_stride,
            uv_stride,
        }
    }

    fn to_video_frame(&self, timestamp: Timestamp, is_keyframe: bool) -> VideoFrame {
        let mut frame =
            VideoFrame::new(PixelFormat::Yuv420p, self.width as u32, self.height as u32);
        frame.timestamp = timestamp;
        frame.frame_type = if is_keyframe {
            FrameType::Key
        } else {
            FrameType::Inter
        };

        frame.allocate();

        // Copy Y plane.
        //
        // NOTE (issue #9 — fixed in 0.1.7): the previous implementation built
        // a destination slice via `frame.planes[0].data.to_vec()[..y_len]`,
        // which produced a *temporary clone* of the plane buffer, mutated the
        // clone, then dropped it. The on-frame buffer was never written, so
        // the decoder returned a frame whose pixels were the all-zero output
        // of `VideoFrame::allocate()` rather than the reconstructed pixels.
        // We now write directly into `frame.planes[i].data` so the decoded
        // pixels actually reach the caller.
        if !frame.planes.is_empty() {
            let y_len = frame.planes[0].data.len().min(self.y_plane.len());
            frame.planes[0].data[..y_len].copy_from_slice(&self.y_plane[..y_len]);
        }

        // Copy U plane.
        if frame.planes.len() > 1 {
            let u_len = frame.planes[1].data.len().min(self.u_plane.len());
            frame.planes[1].data[..u_len].copy_from_slice(&self.u_plane[..u_len]);
        }

        // Copy V plane.
        if frame.planes.len() > 2 {
            let v_len = frame.planes[2].data.len().min(self.v_plane.len());
            frame.planes[2].data[..v_len].copy_from_slice(&self.v_plane[..v_len]);
        }

        frame
    }
}

/// Quantization matrices for Theora.
struct QuantizationMatrices {
    intra_y: [u16; 64],
    intra_c: [u16; 64],
    inter: [u16; 64],
}

impl QuantizationMatrices {
    fn new(quality: u8) -> Self {
        let mut intra_y = [0u16; 64];
        let mut intra_c = [0u16; 64];
        let mut inter = [0u16; 64];

        build_quant_matrix(&BASE_MATRIX_INTRA_Y, quality, &mut intra_y);
        build_quant_matrix(&BASE_MATRIX_INTRA_C, quality, &mut intra_c);
        build_quant_matrix(&BASE_MATRIX_INTER, quality, &mut inter);

        Self {
            intra_y,
            intra_c,
            inter,
        }
    }
}

impl TheoraDecoder {
    /// Create a new Theora decoder.
    ///
    /// # Arguments
    ///
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    /// * `pixel_format` - Pixel format (must be YUV420p)
    pub fn new(width: u32, height: u32, pixel_format: PixelFormat) -> CodecResult<Self> {
        if pixel_format != PixelFormat::Yuv420p {
            return Err(CodecError::UnsupportedFeature(format!(
                "Theora only supports YUV420p, got {pixel_format:?}"
            )));
        }

        let blocks_w = ((width + 15) / 16) as usize;
        let blocks_h = ((height + 15) / 16) as usize;

        Ok(Self {
            config: TheoraDecoderConfig {
                width,
                height,
                pixel_format,
            },
            blocks_w,
            blocks_h,
            last_frame: None,
            golden_frame: None,
            token_decoder: TheoraTokenDecoder::new()?,
            quant_matrices: QuantizationMatrices::new(32),
            frame_count: 0,
            pending_frame: None,
        })
    }

    /// Decode a frame from bitstream data.
    fn decode_frame(&mut self, data: &[u8], pts: i64) -> CodecResult<VideoFrame> {
        let mut reader = BitstreamReader::new(data);

        // Read frame header
        let frame_type = reader.read_bit()?;
        let is_keyframe = !frame_type;

        // Read quality index
        let quality = reader.read_bits(6)? as u8;
        self.quant_matrices = QuantizationMatrices::new(quality);

        // Create output frame
        let mut frame = DecodedFrame::new(self.config.width as usize, self.config.height as usize);

        // Decode blocks
        if is_keyframe {
            self.decode_intra_frame(&mut reader, &mut frame)?;
        } else {
            self.decode_inter_frame(&mut reader, &mut frame)?;
        }

        // Update reference frames
        if is_keyframe {
            self.golden_frame = Some(frame.clone());
        }
        self.last_frame = Some(frame.clone());

        let timestamp = Timestamp::new(pts, Rational::new(1, 1000));
        Ok(frame.to_video_frame(timestamp, is_keyframe))
    }

    /// Decode an intra (keyframe) frame.
    fn decode_intra_frame(
        &mut self,
        reader: &mut BitstreamReader,
        frame: &mut DecodedFrame,
    ) -> CodecResult<()> {
        // Decode Y plane
        for by in 0..self.blocks_h * 2 {
            for bx in 0..self.blocks_w * 2 {
                self.decode_intra_block(reader, frame, bx, by, 0)?;
            }
        }

        // Decode U and V planes
        for by in 0..self.blocks_h {
            for bx in 0..self.blocks_w {
                self.decode_intra_block(reader, frame, bx, by, 1)?;
                self.decode_intra_block(reader, frame, bx, by, 2)?;
            }
        }

        Ok(())
    }

    /// Decode an intra-coded block.
    fn decode_intra_block(
        &mut self,
        reader: &mut BitstreamReader,
        frame: &mut DecodedFrame,
        bx: usize,
        by: usize,
        plane: usize,
    ) -> CodecResult<()> {
        // Decode DCT coefficients
        let mut coeffs = [0i16; 64];
        self.decode_dct_coefficients(reader, &mut coeffs, true, plane == 0)?;

        // Dequantize
        let mut dequant = [0i16; 64];
        let quant_matrix = if plane == 0 {
            &self.quant_matrices.intra_y
        } else {
            &self.quant_matrices.intra_c
        };
        dequantize_block(&coeffs, &mut dequant, quant_matrix);

        // IDCT
        let mut spatial = [0i16; 64];
        idct8x8(&dequant, &mut spatial);

        // Clip and store
        let mut block = [0u8; 64];
        for i in 0..64 {
            block[i] = (spatial[i] + 128).clamp(0, 255) as u8;
        }

        // Write to frame
        let (plane_data, stride) = match plane {
            0 => (&mut frame.y_plane, frame.y_stride),
            1 => (&mut frame.u_plane, frame.uv_stride),
            _ => (&mut frame.v_plane, frame.uv_stride),
        };

        let x = bx * BLOCK_SIZE;
        let y = by * BLOCK_SIZE;
        paste_block(&block, plane_data, stride, x, y);

        Ok(())
    }

    /// Decode an inter (predicted) frame.
    fn decode_inter_frame(
        &mut self,
        reader: &mut BitstreamReader,
        frame: &mut DecodedFrame,
    ) -> CodecResult<()> {
        let reference = self.last_frame.clone().ok_or_else(|| {
            CodecError::DecoderError("No reference frame for inter frame".to_string())
        })?;

        // Decode Y plane
        for by in 0..self.blocks_h * 2 {
            for bx in 0..self.blocks_w * 2 {
                self.decode_inter_block(reader, frame, &reference, bx, by, 0)?;
            }
        }

        // Decode U and V planes
        for by in 0..self.blocks_h {
            for bx in 0..self.blocks_w {
                self.decode_inter_block(reader, frame, &reference, bx, by, 1)?;
                self.decode_inter_block(reader, frame, &reference, bx, by, 2)?;
            }
        }

        Ok(())
    }

    /// Decode an inter-coded block.
    #[allow(clippy::too_many_arguments)]
    fn decode_inter_block(
        &mut self,
        reader: &mut BitstreamReader,
        frame: &mut DecodedFrame,
        reference: &DecodedFrame,
        bx: usize,
        by: usize,
        plane: usize,
    ) -> CodecResult<()> {
        // Read coding mode
        let mode_bits = reader.read_bits(3)?;
        let coded = mode_bits != 7; // 7 = not coded

        if !coded {
            // Copy from reference
            let (ref_plane, stride) = match plane {
                0 => (&reference.y_plane, reference.y_stride),
                1 => (&reference.u_plane, reference.uv_stride),
                _ => (&reference.v_plane, reference.uv_stride),
            };

            let (dst_plane, dst_stride) = match plane {
                0 => (&mut frame.y_plane, frame.y_stride),
                1 => (&mut frame.u_plane, frame.uv_stride),
                _ => (&mut frame.v_plane, frame.uv_stride),
            };

            let x = bx * BLOCK_SIZE;
            let y = by * BLOCK_SIZE;

            let mut block = [0u8; 64];
            copy_block(ref_plane, stride, x, y, &mut block);
            paste_block(&block, dst_plane, dst_stride, x, y);

            return Ok(());
        }

        // Read motion vector
        let mv_x = reader.read_signed_bits(5)? as i16;
        let mv_y = reader.read_signed_bits(5)? as i16;
        let mv = MotionVector::new(mv_x, mv_y);

        // Get reference block with motion compensation
        let (ref_plane, ref_stride) = match plane {
            0 => (&reference.y_plane, reference.y_stride),
            1 => (&reference.u_plane, reference.uv_stride),
            _ => (&reference.v_plane, reference.uv_stride),
        };

        let x = bx * BLOCK_SIZE;
        let y = by * BLOCK_SIZE;
        let mut prediction = [0u8; 64];
        motion_compensate_8x8(ref_plane, ref_stride, x, y, mv, &mut prediction);

        // Decode residual
        let mut coeffs = [0i16; 64];
        self.decode_dct_coefficients(reader, &mut coeffs, false, plane == 0)?;

        // Dequantize
        let mut dequant = [0i16; 64];
        dequantize_block(&coeffs, &mut dequant, &self.quant_matrices.inter);

        // IDCT
        let mut residual = [0i16; 64];
        idct8x8(&dequant, &mut residual);

        // Add to prediction
        let mut block = [0u8; 64];
        add_residual(&prediction, &residual, &mut block);

        // Write to frame
        let (dst_plane, dst_stride) = match plane {
            0 => (&mut frame.y_plane, frame.y_stride),
            1 => (&mut frame.u_plane, frame.uv_stride),
            _ => (&mut frame.v_plane, frame.uv_stride),
        };

        paste_block(&block, dst_plane, dst_stride, x, y);

        Ok(())
    }

    /// Decode DCT coefficients from bitstream.
    ///
    /// Encoding format (self-consistent, sign-magnitude):
    ///
    /// DC coefficient (11 bits):
    ///   - Bit 10: sign flag (0 = non-negative, 1 = negative)
    ///   - Bits 9-0: absolute magnitude (0-1023)
    ///
    /// AC coefficients (run-length, variable length):
    ///   - 6-bit run (0-62 = count of leading zeros; 63 = EOB sentinel)
    ///   - If run < 63: 10-bit value = sign bit (bit 9) | 9-bit abs magnitude (bits 8-0)
    fn decode_dct_coefficients(
        &mut self,
        reader: &mut BitstreamReader,
        coeffs: &mut Block8x8,
        _is_intra: bool,
        _is_luma: bool,
    ) -> CodecResult<()> {
        // Decode DC coefficient: 11 bits = 1 sign + 10 magnitude.
        let sign_bit = reader.read_bit()?;
        let magnitude = reader.read_bits(10)?;
        coeffs[0] = if sign_bit {
            -(magnitude as i16)
        } else {
            magnitude as i16
        };

        // Decode AC coefficients with run-length encoding.
        // Each non-zero AC entry is: 6-bit run, 10-bit value (1 sign + 9 magnitude).
        // EOB is signalled by run = 63 (all-ones, 6 bits).
        let mut pos = 1usize;
        while pos < 64 {
            let run = reader.read_bits(6)? as usize;

            // 63 == EOB sentinel.
            if run == 63 {
                break;
            }

            // Decode value: 10 bits = 1 sign + 9 magnitude.
            let val_bits = reader.read_bits(10)?;
            let val_sign = (val_bits >> 9) & 1 != 0;
            let val_mag = (val_bits & 0x1FF) as i16;
            let val = if val_sign { -val_mag } else { val_mag };

            pos += run;
            if pos < 64 {
                coeffs[pos] = val;
                pos += 1;
            }
        }

        Ok(())
    }
}

impl Clone for DecodedFrame {
    fn clone(&self) -> Self {
        Self {
            y_plane: self.y_plane.clone(),
            u_plane: self.u_plane.clone(),
            v_plane: self.v_plane.clone(),
            width: self.width,
            height: self.height,
            y_stride: self.y_stride,
            uv_stride: self.uv_stride,
        }
    }
}

impl VideoDecoder for TheoraDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Theora
    }

    fn send_packet(&mut self, data: &[u8], pts: i64) -> CodecResult<()> {
        let frame = self.decode_frame(data, pts)?;
        self.pending_frame = Some(frame);
        Ok(())
    }

    fn receive_frame(&mut self) -> CodecResult<Option<VideoFrame>> {
        Ok(self.pending_frame.take())
    }

    fn flush(&mut self) -> CodecResult<()> {
        self.pending_frame = None;
        Ok(())
    }

    fn reset(&mut self) {
        self.last_frame = None;
        self.golden_frame = None;
        self.pending_frame = None;
        self.frame_count = 0;
    }

    fn output_format(&self) -> Option<PixelFormat> {
        Some(self.config.pixel_format)
    }

    fn dimensions(&self) -> Option<(u32, u32)> {
        Some((self.config.width, self.config.height))
    }
}

/// Theora encoder configuration.
#[derive(Debug, Clone)]
pub struct TheoraConfig {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Quality (0-63, higher = better).
    pub quality: u8,
    /// Target bitrate in bits per second.
    pub target_bitrate: u64,
    /// Keyframe interval (in frames).
    pub keyframe_interval: u32,
    /// Pixel format.
    pub pixel_format: PixelFormat,
}

impl TheoraConfig {
    /// Create a new Theora encoder configuration.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            quality: 48,
            target_bitrate: 2_000_000,
            keyframe_interval: 64,
            pixel_format: PixelFormat::Yuv420p,
        }
    }

    /// Set quality level (0-63, higher = better).
    #[must_use]
    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality = quality.min(63);
        self
    }

    /// Set target bitrate in bits per second.
    #[must_use]
    pub fn with_target_bitrate(mut self, bitrate: u64) -> Self {
        self.target_bitrate = bitrate;
        self
    }

    /// Set keyframe interval.
    #[must_use]
    pub fn with_keyframe_interval(mut self, interval: u32) -> Self {
        self.keyframe_interval = interval;
        self
    }
}

/// Theora encoder.
///
/// Encodes raw video frames to Theora bitstream.
pub struct TheoraEncoder {
    /// Configuration.
    config: TheoraConfig,
    /// Encoder config (for trait).
    encoder_config: EncoderConfig,
    /// Width in blocks.
    blocks_w: usize,
    /// Height in blocks.
    blocks_h: usize,
    /// Last reference frame.
    last_frame: Option<DecodedFrame>,
    /// Golden reference frame.
    golden_frame: Option<DecodedFrame>,
    /// Huffman token encoder.
    token_encoder: TheoraTokenEncoder,
    /// Quantization matrices.
    quant_matrices: QuantizationMatrices,
    /// Frame count.
    frame_count: u64,
    /// Pending output packet.
    pending_packet: Option<EncodedPacket>,
}

impl TheoraEncoder {
    /// Create a new Theora encoder.
    pub fn new(config: TheoraConfig) -> CodecResult<Self> {
        if config.pixel_format != PixelFormat::Yuv420p {
            return Err(CodecError::UnsupportedFeature(format!(
                "Theora only supports YUV420p, got {:?}",
                config.pixel_format
            )));
        }

        let blocks_w = ((config.width + 15) / 16) as usize;
        let blocks_h = ((config.height + 15) / 16) as usize;

        let encoder_config = EncoderConfig {
            codec: CodecId::Theora,
            width: config.width,
            height: config.height,
            pixel_format: config.pixel_format,
            framerate: Rational::new(30, 1),
            bitrate: BitrateMode::Cbr(config.target_bitrate),
            preset: crate::traits::EncoderPreset::Medium,
            profile: None,
            keyint: config.keyframe_interval,
            threads: 1,
            timebase: Rational::new(1, 1000),
        };

        Ok(Self {
            quant_matrices: QuantizationMatrices::new(config.quality),
            encoder_config,
            blocks_w,
            blocks_h,
            last_frame: None,
            golden_frame: None,
            token_encoder: TheoraTokenEncoder::new(),
            frame_count: 0,
            pending_packet: None,
            config,
        })
    }

    /// Encode a video frame.
    fn encode_frame(&mut self, frame: &VideoFrame) -> CodecResult<EncodedPacket> {
        let is_keyframe = self.frame_count % u64::from(self.config.keyframe_interval) == 0;

        let mut writer =
            BitstreamWriter::with_capacity(frame.width as usize * frame.height as usize);

        // Write frame header
        writer.write_bit(!is_keyframe);
        writer.write_bits(u32::from(self.config.quality), 6);

        // Convert frame to internal format
        let decoded_frame = self.video_frame_to_decoded(frame)?;

        // Encode blocks
        if is_keyframe {
            self.encode_intra_frame(&mut writer, &decoded_frame)?;
        } else {
            self.encode_inter_frame(&mut writer, &decoded_frame)?;
        }

        writer.byte_align();

        // Update reference frames
        if is_keyframe {
            self.golden_frame = Some(decoded_frame.clone());
        }
        self.last_frame = Some(decoded_frame);

        let pts = frame.timestamp.pts;
        let packet = EncodedPacket {
            data: writer.into_vec(),
            pts,
            dts: pts,
            keyframe: is_keyframe,
            duration: Some(1),
        };

        self.frame_count += 1;
        Ok(packet)
    }

    /// Convert VideoFrame to DecodedFrame.
    fn video_frame_to_decoded(&self, frame: &VideoFrame) -> CodecResult<DecodedFrame> {
        let mut decoded = DecodedFrame::new(frame.width as usize, frame.height as usize);

        if frame.planes.len() >= 3 {
            let y_len = decoded.y_plane.len().min(frame.planes[0].data.len());
            decoded.y_plane[..y_len].copy_from_slice(&frame.planes[0].data[..y_len]);

            let u_len = decoded.u_plane.len().min(frame.planes[1].data.len());
            decoded.u_plane[..u_len].copy_from_slice(&frame.planes[1].data[..u_len]);

            let v_len = decoded.v_plane.len().min(frame.planes[2].data.len());
            decoded.v_plane[..v_len].copy_from_slice(&frame.planes[2].data[..v_len]);
        }

        Ok(decoded)
    }

    /// Encode an intra frame.
    fn encode_intra_frame(
        &mut self,
        writer: &mut BitstreamWriter,
        frame: &DecodedFrame,
    ) -> CodecResult<()> {
        // Encode Y plane
        for by in 0..self.blocks_h * 2 {
            for bx in 0..self.blocks_w * 2 {
                self.encode_intra_block(writer, frame, bx, by, 0)?;
            }
        }

        // Encode U and V planes
        for by in 0..self.blocks_h {
            for bx in 0..self.blocks_w {
                self.encode_intra_block(writer, frame, bx, by, 1)?;
                self.encode_intra_block(writer, frame, bx, by, 2)?;
            }
        }

        Ok(())
    }

    /// Encode an intra block.
    fn encode_intra_block(
        &mut self,
        writer: &mut BitstreamWriter,
        frame: &DecodedFrame,
        bx: usize,
        by: usize,
        plane: usize,
    ) -> CodecResult<()> {
        // Extract block
        let (plane_data, stride) = match plane {
            0 => (&frame.y_plane, frame.y_stride),
            1 => (&frame.u_plane, frame.uv_stride),
            _ => (&frame.v_plane, frame.uv_stride),
        };

        let x = bx * BLOCK_SIZE;
        let y = by * BLOCK_SIZE;
        let mut block = [0u8; 64];
        copy_block(plane_data, stride, x, y, &mut block);

        // Convert to spatial residual (subtract DC bias)
        let mut spatial = [0i16; 64];
        for i in 0..64 {
            spatial[i] = i16::from(block[i]) - 128;
        }

        // DCT
        let mut freq = [0i16; 64];
        fdct8x8(&spatial, &mut freq);

        // Quantize
        let mut quantized = [0i16; 64];
        let quant_matrix = if plane == 0 {
            &self.quant_matrices.intra_y
        } else {
            &self.quant_matrices.intra_c
        };
        quantize_block(&freq, &mut quantized, quant_matrix);

        // Encode coefficients
        self.encode_dct_coefficients(writer, &quantized, true, plane == 0)?;

        Ok(())
    }

    /// Encode an inter frame.
    fn encode_inter_frame(
        &mut self,
        writer: &mut BitstreamWriter,
        frame: &DecodedFrame,
    ) -> CodecResult<()> {
        let reference = self.last_frame.clone().ok_or_else(|| {
            CodecError::Internal("No reference frame for inter encoding".to_string())
        })?;

        // Encode Y plane
        for by in 0..self.blocks_h * 2 {
            for bx in 0..self.blocks_w * 2 {
                self.encode_inter_block(writer, frame, &reference, bx, by, 0)?;
            }
        }

        // Encode U and V planes
        for by in 0..self.blocks_h {
            for bx in 0..self.blocks_w {
                self.encode_inter_block(writer, frame, &reference, bx, by, 1)?;
                self.encode_inter_block(writer, frame, &reference, bx, by, 2)?;
            }
        }

        Ok(())
    }

    /// Encode an inter block.
    #[allow(clippy::too_many_arguments)]
    fn encode_inter_block(
        &mut self,
        writer: &mut BitstreamWriter,
        frame: &DecodedFrame,
        reference: &DecodedFrame,
        bx: usize,
        by: usize,
        plane: usize,
    ) -> CodecResult<()> {
        // Extract current block
        let (plane_data, stride) = match plane {
            0 => (&frame.y_plane, frame.y_stride),
            1 => (&frame.u_plane, frame.uv_stride),
            _ => (&frame.v_plane, frame.uv_stride),
        };

        let x = bx * BLOCK_SIZE;
        let y = by * BLOCK_SIZE;
        let mut current_block = [0u8; 64];
        copy_block(plane_data, stride, x, y, &mut current_block);

        // Motion estimation
        let (ref_plane, ref_stride) = match plane {
            0 => (&reference.y_plane, reference.y_stride),
            1 => (&reference.u_plane, reference.uv_stride),
            _ => (&reference.v_plane, reference.uv_stride),
        };

        let (mv, sad) = motion_estimation_diamond(
            &current_block,
            ref_plane,
            ref_stride,
            x,
            y,
            8, // search range
        );

        // Check if skip is better
        let mut skip_block = [0u8; 64];
        copy_block(ref_plane, ref_stride, x, y, &mut skip_block);
        let mut skip_cost = 0u32;
        for i in 0..64 {
            skip_cost += (i32::from(current_block[i]) - i32::from(skip_block[i])).unsigned_abs();
        }

        if skip_cost < 128 && mv.is_zero() {
            // Skip block
            writer.write_bits(7, 3); // mode = not coded
            return Ok(());
        }

        // Write coded block mode
        writer.write_bits(1, 3); // mode = inter with MV

        // Write motion vector
        writer.write_signed_bits(i32::from(mv.x), 5);
        writer.write_signed_bits(i32::from(mv.y), 5);

        // Get prediction
        let mut prediction = [0u8; 64];
        motion_compensate_8x8(ref_plane, ref_stride, x, y, mv, &mut prediction);

        // Compute residual
        let mut residual_spatial = [0i16; 64];
        subtract_prediction(&current_block, &prediction, &mut residual_spatial);

        // DCT
        let mut freq = [0i16; 64];
        fdct8x8(&residual_spatial, &mut freq);

        // Quantize
        let mut quantized = [0i16; 64];
        quantize_block(&freq, &mut quantized, &self.quant_matrices.inter);

        // Encode coefficients
        self.encode_dct_coefficients(writer, &quantized, false, plane == 0)?;

        Ok(())
    }

    /// Encode DCT coefficients to bitstream.
    ///
    /// Encoding format (self-consistent, sign-magnitude):
    ///
    /// DC coefficient (11 bits):
    ///   - Bit 10: sign flag (0 = non-negative, 1 = negative)
    ///   - Bits 9-0: absolute magnitude (0-1023)
    ///
    /// AC coefficients (run-length, variable length):
    ///   - 6-bit run (0-62 = count of leading zeros; 63 = EOB sentinel)
    ///   - If run < 63: 10-bit value = sign bit (bit 9) | 9-bit abs magnitude (bits 8-0)
    fn encode_dct_coefficients(
        &mut self,
        writer: &mut BitstreamWriter,
        coeffs: &Block8x8,
        _is_intra: bool,
        _is_luma: bool,
    ) -> CodecResult<()> {
        // Encode DC coefficient: 11 bits = 1 sign + 10 magnitude.
        let dc = coeffs[0];
        let dc_sign = dc < 0;
        let dc_magnitude = (dc.unsigned_abs() as u32).min(1023);
        writer.write_bit(dc_sign);
        writer.write_bits(dc_magnitude, 10);

        // Encode AC coefficients with run-length encoding.
        // Each entry: 6-bit run (0-62) + 10-bit value (1 sign + 9 magnitude).
        // EOB: 6-bit run = 63.
        let mut run = 0u32;
        for pos in 1..64 {
            if coeffs[pos] == 0 {
                run += 1;
            } else {
                // Flush accumulated run (may need multiple entries if run >= 62).
                // Run field is 6 bits with 63 reserved for EOB, so max run per entry = 62.
                // For runs ≥ 62: emit intermediate entries with run=62, value=0 (the decoder
                // writes 0 at that position, advancing the position counter by 63 total).
                // Each such entry consumes 63 positions (62 zeros + 1 zero-value slot).
                while run >= 62 {
                    writer.write_bits(62, 6);
                    // value = 0: sign=0, magnitude=0 → 10-bit value = 0.
                    writer.write_bits(0, 10);
                    // 63 positions consumed: the 62-zero run plus the 1-position zero value.
                    run = run.saturating_sub(63);
                }

                let v = coeffs[pos];
                let v_sign = if v < 0 { 1u32 } else { 0u32 };
                let v_mag = (v.unsigned_abs() as u32).min(511);
                // 10-bit value: sign at bit 9, magnitude at bits 8-0.
                let val_bits = (v_sign << 9) | v_mag;
                writer.write_bits(run, 6);
                writer.write_bits(val_bits, 10);
                run = 0;
            }
        }

        // EOB sentinel: run = 63 (6 bits all-ones).
        writer.write_bits(63, 6);

        Ok(())
    }
}

impl VideoEncoder for TheoraEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Theora
    }

    fn send_frame(&mut self, frame: &VideoFrame) -> CodecResult<()> {
        let packet = self.encode_frame(frame)?;
        self.pending_packet = Some(packet);
        Ok(())
    }

    fn receive_packet(&mut self) -> CodecResult<Option<EncodedPacket>> {
        Ok(self.pending_packet.take())
    }

    fn flush(&mut self) -> CodecResult<()> {
        self.pending_packet = None;
        Ok(())
    }

    fn config(&self) -> &EncoderConfig {
        &self.encoder_config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::{PixelFormat, Rational, Timestamp};

    /// Regression test for GitHub issue #9 — Theora decoded pixels never
    /// reached the returned `VideoFrame`.
    ///
    /// Pre-fix: `DecodedFrame::to_video_frame` wrote to a temporary
    /// `frame.planes[i].data.to_vec()` clone that was immediately dropped,
    /// so callers received the all-zero buffer freshly produced by
    /// `VideoFrame::allocate()` regardless of how much real reconstruction
    /// the rest of the decoder had performed.
    ///
    /// This test exercises the copy path directly: it builds a `DecodedFrame`
    /// with known sentinel values per plane, calls `to_video_frame`, and
    /// asserts that each `VideoFrame.planes[i].data` carries those sentinel
    /// bytes. With the buggy `to_vec()` slice, every assertion below would
    /// fail because the planes would be all zero.
    #[test]
    fn test_issue_9_to_video_frame_writes_planes_into_videoframe() {
        const WIDTH: usize = 32;
        const HEIGHT: usize = 16;
        const Y_SENTINEL: u8 = 0x42;
        const U_SENTINEL: u8 = 0xA1;
        const V_SENTINEL: u8 = 0x37;

        let mut decoded = DecodedFrame::new(WIDTH, HEIGHT);
        decoded.y_plane.fill(Y_SENTINEL);
        decoded.u_plane.fill(U_SENTINEL);
        decoded.v_plane.fill(V_SENTINEL);

        let timestamp = Timestamp::new(0, Rational::new(1, 1000));
        let frame = decoded.to_video_frame(timestamp, true);

        assert_eq!(frame.format, PixelFormat::Yuv420p);
        assert_eq!(frame.width, WIDTH as u32);
        assert_eq!(frame.height, HEIGHT as u32);
        assert_eq!(frame.planes.len(), 3, "YUV420p must expose 3 planes");
        assert_eq!(frame.frame_type, crate::frame::FrameType::Key);

        // The crux of the issue-9 regression check.
        let y_data = &frame.planes[0].data;
        let u_data = &frame.planes[1].data;
        let v_data = &frame.planes[2].data;

        // 1. Every plane has at least one byte of the sentinel — pre-fix,
        //    every plane would be all-zero from `VideoFrame::allocate()`.
        assert!(
            y_data.iter().any(|&b| b == Y_SENTINEL),
            "issue #9 regression: Y plane has zero bytes from allocate(); \
             decoder did not copy `DecodedFrame::y_plane` into VideoFrame.planes[0].data"
        );
        assert!(
            u_data.iter().any(|&b| b == U_SENTINEL),
            "issue #9 regression: U plane was not written to VideoFrame.planes[1].data"
        );
        assert!(
            v_data.iter().any(|&b| b == V_SENTINEL),
            "issue #9 regression: V plane was not written to VideoFrame.planes[2].data"
        );

        // 2. Stronger check — the plane buffers are dominated by the
        //    sentinel value (since the source `DecodedFrame` was uniformly
        //    filled and the `to_video_frame` copy is `min(dst.len(),
        //    src.len())`). Any non-sentinel byte indicates a partial copy
        //    bug or an off-by-one in the destination slice.
        let y_sentinel_count = y_data.iter().filter(|&&b| b == Y_SENTINEL).count();
        let u_sentinel_count = u_data.iter().filter(|&&b| b == U_SENTINEL).count();
        let v_sentinel_count = v_data.iter().filter(|&&b| b == V_SENTINEL).count();
        assert_eq!(
            y_sentinel_count,
            y_data.len(),
            "issue #9 regression: Y plane copy is incomplete \
             ({y_sentinel_count} of {} bytes match sentinel)",
            y_data.len()
        );
        assert_eq!(
            u_sentinel_count,
            u_data.len(),
            "issue #9 regression: U plane copy is incomplete"
        );
        assert_eq!(
            v_sentinel_count,
            v_data.len(),
            "issue #9 regression: V plane copy is incomplete"
        );

        // 3. The decoded VideoFrame must NOT equal the freshly-allocated,
        //    all-zero buffer, which is exactly what the buggy code path
        //    used to return.
        let mut buggy_ghost =
            crate::frame::VideoFrame::new(PixelFormat::Yuv420p, WIDTH as u32, HEIGHT as u32);
        buggy_ghost.allocate();
        assert_ne!(
            frame.planes[0].data, buggy_ghost.planes[0].data,
            "issue #9 regression: Y plane equals VideoFrame::allocate() output"
        );
        assert_ne!(
            frame.planes[1].data, buggy_ghost.planes[1].data,
            "issue #9 regression: U plane equals VideoFrame::allocate() output"
        );
        assert_ne!(
            frame.planes[2].data, buggy_ghost.planes[2].data,
            "issue #9 regression: V plane equals VideoFrame::allocate() output"
        );
    }
}
