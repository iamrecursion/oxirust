//! VP8 key-frame header parsing (RFC 6386 §9, §19).
//!
//! WebP lossy frames are always key frames, so only the key-frame branch of the
//! VP8 frame header is implemented here. The header consists of:
//! - a 3-byte uncompressed "frame tag" (frame type, version, show flag, first
//!   partition length),
//! - the 7-byte uncompressed key-frame start code + dimensions,
//! - a boolean-coded section: colour space, clamping, segmentation, loop-filter
//!   parameters, partition count, quantiser indices and probability updates.

use super::bool_decoder::BoolDecoder;
use super::tables::{COEFF_UPDATE_PROBS, DEFAULT_COEFF_PROBS, KF_UV_MODE_PROB, KF_YMODE_PROB};
use crate::error::{ImageError, ImageResult};

/// Number of independently-decodable segments (RFC 6386 §10).
pub const MAX_SEGMENTS: usize = 4;
/// Number of macroblock "reference frame" loop-filter delta slots.
pub const MAX_REF_LF_DELTAS: usize = 4;
/// Number of macroblock "mode" loop-filter delta slots.
pub const MAX_MODE_LF_DELTAS: usize = 4;

/// Per-segment quantiser / loop-filter feature data.
#[derive(Debug, Clone, Default)]
pub struct SegmentHeader {
    /// Whether segmentation is enabled for this frame.
    pub enabled: bool,
    /// Whether the per-MB segment map is transmitted this frame.
    pub update_map: bool,
    /// Absolute (true) vs delta (false) interpretation of segment features.
    pub abs_delta: bool,
    /// Per-segment quantiser values (absolute or delta).
    pub quantizer: [i32; MAX_SEGMENTS],
    /// Per-segment loop-filter levels (absolute or delta).
    pub filter_strength: [i32; MAX_SEGMENTS],
    /// Probabilities for the per-MB segment-id tree.
    pub tree_probs: [u8; 3],
}

/// In-loop deblocking-filter header (RFC 6386 §9.4, §15).
#[derive(Debug, Clone, Default)]
pub struct LoopFilterHeader {
    /// `true` selects the simple filter, `false` the normal filter.
    pub simple: bool,
    /// Base filter level for the frame.
    pub level: i32,
    /// Sharpness control (0..7) feeding the interior-limit derivation.
    pub sharpness: i32,
    /// Whether per-MB loop-filter deltas are in use.
    pub delta_enabled: bool,
    /// Per-reference-frame loop-filter level deltas.
    pub ref_deltas: [i32; MAX_REF_LF_DELTAS],
    /// Per-prediction-mode loop-filter level deltas.
    pub mode_deltas: [i32; MAX_MODE_LF_DELTAS],
}

/// Quantiser-index header (RFC 6386 §9.6).
#[derive(Debug, Clone, Default)]
pub struct QuantHeader {
    /// Base AC quantiser index for the luma plane.
    pub y_ac_qi: i32,
    /// Delta applied to the luma DC quantiser index.
    pub y_dc_delta: i32,
    /// Delta applied to the Y2 (WHT) DC quantiser index.
    pub y2_dc_delta: i32,
    /// Delta applied to the Y2 (WHT) AC quantiser index.
    pub y2_ac_delta: i32,
    /// Delta applied to the chroma DC quantiser index.
    pub uv_dc_delta: i32,
    /// Delta applied to the chroma AC quantiser index.
    pub uv_ac_delta: i32,
}

/// Fully-parsed VP8 key-frame header plus decoder-ready probability state.
pub struct FrameHeader {
    /// Decoded frame width in pixels.
    pub width: u32,
    /// Decoded frame height in pixels.
    pub height: u32,
    /// Horizontal upscaling code (0 = none).
    pub horizontal_scale: u8,
    /// Vertical upscaling code (0 = none).
    pub vertical_scale: u8,
    /// Colour-space flag (0 = YUV; only 0 is defined).
    pub color_space: u8,
    /// Pixel-clamping flag (0 = decoder must clamp).
    pub clamping_required: bool,
    /// Segmentation header.
    pub segment: SegmentHeader,
    /// Loop-filter header.
    pub loop_filter: LoopFilterHeader,
    /// Quantiser header.
    pub quant: QuantHeader,
    /// DCT-token probability table after any header updates.
    pub coeff_probs: [[[[u8; 11]; 3]; 8]; 4],
    /// Whether the per-MB `mb_skip_coeff` flag is coded (skipping enabled).
    pub mb_no_skip_coeff: bool,
    /// Probability used by the `mb_skip_coeff` flag when skipping is enabled.
    pub prob_skip_false: u8,
    /// 16x16 luma mode probabilities (key frames use the fixed `KF_YMODE_PROB`).
    pub y_mode_probs: [u8; 4],
    /// Chroma mode probabilities (key frames use the fixed `KF_UV_MODE_PROB`).
    pub uv_mode_probs: [u8; 3],
    /// Byte offset of the first DCT-token partition, relative to the start of
    /// the VP8 payload.
    pub partitions_start: usize,
    /// Length of the first (header) partition in bytes.
    pub first_partition_size: usize,
    /// Number of DCT-token partitions (1, 2, 4 or 8).
    pub num_token_partitions: usize,
}

impl FrameHeader {
    /// Parses a VP8 key-frame header from `data` (the raw `VP8 ` chunk payload).
    ///
    /// Returns the parsed header and the boolean decoder positioned immediately
    /// after the header section, ready to decode macroblock prediction modes.
    ///
    /// # Errors
    /// Fails if the payload is too short, is not a key frame, has a bad start
    /// code, or declares unsupported parameters.
    pub fn parse(data: &[u8]) -> ImageResult<(Self, BoolDecoder<'_>)> {
        if data.len() < 10 {
            return Err(ImageError::invalid_format("VP8: payload too small"));
        }

        // --- 3-byte uncompressed frame tag (RFC 6386 §9.1) ---
        let tag = u32::from(data[0]) | (u32::from(data[1]) << 8) | (u32::from(data[2]) << 16);
        let key_frame = (tag & 1) == 0;
        let version = ((tag >> 1) & 0x7) as u8;
        let _show_frame = ((tag >> 4) & 1) != 0;
        let first_partition_size = ((tag >> 5) & 0x7_FFFF) as usize;

        if !key_frame {
            return Err(ImageError::unsupported(
                "VP8: inter frames are not valid in still WebP",
            ));
        }
        if version > 3 {
            return Err(ImageError::invalid_format("VP8: unsupported version"));
        }

        // --- 7-byte uncompressed key-frame header (RFC 6386 §9.1) ---
        // Start code: 0x9d 0x01 0x2a.
        if data[3] != 0x9d || data[4] != 0x01 || data[5] != 0x2a {
            return Err(ImageError::invalid_format("VP8: bad key-frame start code"));
        }
        let dim0 = u32::from(data[6]) | (u32::from(data[7]) << 8);
        let dim1 = u32::from(data[8]) | (u32::from(data[9]) << 8);
        let width = dim0 & 0x3FFF;
        let horizontal_scale = (dim0 >> 14) as u8;
        let height = dim1 & 0x3FFF;
        let vertical_scale = (dim1 >> 14) as u8;

        if width == 0 || height == 0 {
            return Err(ImageError::invalid_format("VP8: zero dimension"));
        }

        // First partition starts after the 10-byte uncompressed header.
        let header_end = 10usize;
        let partitions_start = header_end
            .checked_add(first_partition_size)
            .ok_or_else(|| ImageError::invalid_format("VP8: partition size overflow"))?;
        if first_partition_size == 0 || partitions_start > data.len() {
            return Err(ImageError::invalid_format(
                "VP8: first partition exceeds payload",
            ));
        }

        // The boolean decoder operates on the first (header) partition only.
        let header_partition = &data[header_end..partitions_start];
        let mut bd = BoolDecoder::new(header_partition);

        // --- colour space and clamping (RFC 6386 §9.2) ---
        let color_space = u8::from(bd.get_flag());
        let clamping_required = !bd.get_flag(); // 0 => decoder must clamp

        // --- segmentation header (RFC 6386 §9.3, §10) ---
        let segment = parse_segmentation(&mut bd);

        // --- loop-filter header (RFC 6386 §9.4) ---
        let loop_filter = parse_loop_filter(&mut bd);

        // --- token partition count (RFC 6386 §9.5) ---
        let log2_partitions = bd.get_literal(2);
        let num_token_partitions = 1usize << log2_partitions;

        // --- quantiser indices (RFC 6386 §9.6) ---
        let quant = parse_quant(&mut bd);

        // --- refresh flags (key frame: golden / altref refresh forced) ---
        // For a key frame only `refresh_entropy_probs` is coded here.
        let _refresh_entropy_probs = bd.get_flag();

        // --- DCT-token probability updates (RFC 6386 §9.7, §13.4) ---
        let mut coeff_probs = DEFAULT_COEFF_PROBS;
        for (i, plane) in COEFF_UPDATE_PROBS.iter().enumerate() {
            for (j, band) in plane.iter().enumerate() {
                for (k, ctx) in band.iter().enumerate() {
                    for (t, &update_prob) in ctx.iter().enumerate() {
                        if bd.get_bool(update_prob) {
                            coeff_probs[i][j][k][t] = bd.get_literal(8) as u8;
                        }
                    }
                }
            }
        }

        // --- mb_no_skip_coeff (RFC 6386 §9.10) ---
        let mb_no_skip_coeff = bd.get_flag();
        let prob_skip_false = if mb_no_skip_coeff {
            bd.get_literal(8) as u8
        } else {
            0
        };

        let header = Self {
            width,
            height,
            horizontal_scale,
            vertical_scale,
            color_space,
            clamping_required,
            segment,
            loop_filter,
            quant,
            coeff_probs,
            mb_no_skip_coeff,
            prob_skip_false,
            y_mode_probs: KF_YMODE_PROB,
            uv_mode_probs: KF_UV_MODE_PROB,
            partitions_start,
            first_partition_size,
            num_token_partitions,
        };
        Ok((header, bd))
    }
}

/// Parses the segmentation sub-header (RFC 6386 §9.3).
fn parse_segmentation(bd: &mut BoolDecoder<'_>) -> SegmentHeader {
    let mut seg = SegmentHeader {
        tree_probs: [255, 255, 255],
        ..SegmentHeader::default()
    };
    seg.enabled = bd.get_flag();
    if !seg.enabled {
        return seg;
    }
    seg.update_map = bd.get_flag();
    let update_data = bd.get_flag();
    if update_data {
        seg.abs_delta = bd.get_flag();
        // Quantiser feature: signed 7-bit magnitude per segment.
        for q in &mut seg.quantizer {
            *q = if bd.get_flag() {
                bd.get_signed_literal(7)
            } else {
                0
            };
        }
        // Loop-filter feature: signed 6-bit magnitude per segment.
        for f in &mut seg.filter_strength {
            *f = if bd.get_flag() {
                bd.get_signed_literal(6)
            } else {
                0
            };
        }
    }
    if seg.update_map {
        for p in &mut seg.tree_probs {
            *p = if bd.get_flag() {
                bd.get_literal(8) as u8
            } else {
                255
            };
        }
    }
    seg
}

/// Parses the loop-filter sub-header (RFC 6386 §9.4).
fn parse_loop_filter(bd: &mut BoolDecoder<'_>) -> LoopFilterHeader {
    let mut lf = LoopFilterHeader {
        simple: bd.get_flag(),
        level: bd.get_literal(6) as i32,
        sharpness: bd.get_literal(3) as i32,
        ..LoopFilterHeader::default()
    };
    lf.delta_enabled = bd.get_flag();
    if lf.delta_enabled {
        // `loop_filter_delta_update`
        if bd.get_flag() {
            for d in &mut lf.ref_deltas {
                if bd.get_flag() {
                    *d = bd.get_signed_literal(6);
                }
            }
            for d in &mut lf.mode_deltas {
                if bd.get_flag() {
                    *d = bd.get_signed_literal(6);
                }
            }
        }
    }
    lf
}

/// Parses the quantiser sub-header (RFC 6386 §9.6).
fn parse_quant(bd: &mut BoolDecoder<'_>) -> QuantHeader {
    // Reads a flagged signed 4-bit delta (used for every delta field).
    fn delta(bd: &mut BoolDecoder<'_>) -> i32 {
        if bd.get_flag() {
            bd.get_signed_literal(4)
        } else {
            0
        }
    }
    QuantHeader {
        y_ac_qi: bd.get_literal(7) as i32,
        y_dc_delta: delta(bd),
        y2_dc_delta: delta(bd),
        y2_ac_delta: delta(bd),
        uv_dc_delta: delta(bd),
        uv_ac_delta: delta(bd),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rejects_short_payload() {
        assert!(FrameHeader::parse(&[0u8; 4]).is_err());
    }

    #[test]
    fn test_parse_rejects_bad_start_code() {
        // 10-byte payload, key frame tag, wrong start code.
        let mut data = vec![0u8; 64];
        let part_size: u32 = 32;
        let tag = part_size << 5;
        data[0] = (tag & 0xFF) as u8;
        data[1] = ((tag >> 8) & 0xFF) as u8;
        data[2] = ((tag >> 16) & 0xFF) as u8;
        data[3] = 0x00; // wrong start code
        assert!(FrameHeader::parse(&data).is_err());
    }

    #[test]
    fn test_parse_rejects_inter_frame() {
        let mut data = vec![0u8; 64];
        data[0] = 1; // bit0 = 1 => inter frame
        assert!(FrameHeader::parse(&data).is_err());
    }

    #[test]
    fn test_segment_header_default() {
        let seg = SegmentHeader::default();
        assert!(!seg.enabled);
        assert_eq!(seg.quantizer, [0; MAX_SEGMENTS]);
    }
}
