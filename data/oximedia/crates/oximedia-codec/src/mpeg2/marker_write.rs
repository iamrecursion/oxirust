//! MPEG-2 header writers (ISO/IEC 13818-2 §6.2 / §6.3), the encoder-side
//! inverse of [`super::headers`].
//!
//! Each writer emits a header **payload** with exactly the field order and bit
//! widths the matching parser in [`super::headers`] consumes. Start codes
//! themselves are emitted by the caller via
//! [`BitWriter::write_start_code`](super::bitwriter::BitWriter::write_start_code).
//!
//! Scope: the writers cover what the I-frame / 4:2:0 / progressive-frame decoder
//! reads. Optional matrix downloads and the composite-display extension are
//! written with their "absent" flags (load flags `0`, composite flag `0`) so the
//! decoder falls back to the default quant matrices — which is exactly what the
//! encoder quantises against.

use super::bitwriter::BitWriter;
use super::dequant::{DEFAULT_INTRA_MATRIX, DEFAULT_NON_INTRA_MATRIX};
use super::zigzag::SCAN_PROGRESSIVE;

/// `picture_structure` value for a frame (progressive) picture.
const PICTURE_STRUCTURE_FRAME: u32 = 3;
/// `picture_coding_type` value for an I (intra) picture.
const PICTURE_CODING_TYPE_I: u32 = 1;
/// `chroma_format` value for 4:2:0 (6 blocks/MB).
pub const CHROMA_FORMAT_420: u32 = 1;
/// `chroma_format` value for 4:2:2 (8 blocks/MB).
pub const CHROMA_FORMAT_422: u32 = 2;
/// `chroma_format` value for 4:4:4 (12 blocks/MB).
pub const CHROMA_FORMAT_444: u32 = 3;
/// Sequence Extension identifier (`0001`).
const SEQUENCE_EXTENSION_ID: u32 = 0b0001;
/// Picture Coding Extension identifier (`1000`).
const PICTURE_CODING_EXTENSION_ID: u32 = 0b1000;

/// Parameters needed to write a `sequence_header`.
#[derive(Debug, Clone, Copy)]
pub struct SequenceHeaderParams {
    /// Full luminance width (low 12 bits go in the base header).
    pub width: u32,
    /// Full luminance height (low 12 bits go in the base header).
    pub height: u32,
    /// `aspect_ratio_information` (4 bits).
    pub aspect_ratio_information: u8,
    /// `frame_rate_code` (4 bits).
    pub frame_rate_code: u8,
    /// `bit_rate_value` (18 bits, units of 400 bit/s). 0x3FFFF = unconstrained.
    pub bit_rate_value: u32,
    /// `vbv_buffer_size_value` (10 bits).
    pub vbv_buffer_size_value: u32,
    /// Whether to download the (default) intra/non-intra quant matrices.
    /// When `false` the load flags are `0` and the decoder uses its defaults.
    pub load_default_matrices: bool,
}

/// Write a 64-entry quant matrix in **zig-zag** (Figure 7-2) order, 8 bits each,
/// from a raster-order source — exactly what `read_quant_matrix` expects.
fn write_quant_matrix_zigzag(writer: &mut BitWriter, matrix_raster: &[u8; 64]) {
    for &raster_pos in SCAN_PROGRESSIVE.iter() {
        writer.write_bits(u32::from(matrix_raster[raster_pos]), 8);
    }
}

/// Write a `sequence_header` payload (after the `0xB3` start code).
pub fn write_sequence_header(writer: &mut BitWriter, params: &SequenceHeaderParams) {
    writer.write_bits(params.width & 0xFFF, 12);
    writer.write_bits(params.height & 0xFFF, 12);
    writer.write_bits(u32::from(params.aspect_ratio_information), 4);
    writer.write_bits(u32::from(params.frame_rate_code), 4);
    writer.write_bits(params.bit_rate_value & 0x3_FFFF, 18);
    writer.write_bit(true); // marker bit
    writer.write_bits(params.vbv_buffer_size_value & 0x3FF, 10);
    writer.write_bit(false); // constrained_parameters_flag

    if params.load_default_matrices {
        writer.write_bit(true); // load_intra_quantiser_matrix
        write_quant_matrix_zigzag(writer, &DEFAULT_INTRA_MATRIX);
        writer.write_bit(true); // load_non_intra_quantiser_matrix
        write_quant_matrix_zigzag(writer, &DEFAULT_NON_INTRA_MATRIX);
    } else {
        writer.write_bit(false); // load_intra_quantiser_matrix = 0 → default
        writer.write_bit(false); // load_non_intra_quantiser_matrix = 0 → default
    }
}

/// Parameters for the `sequence_extension`.
#[derive(Debug, Clone, Copy)]
pub struct SequenceExtensionParams {
    /// `profile_and_level_indication` (8 bits). 0x44 = Main@Main.
    pub profile_and_level_indication: u8,
    /// `progressive_sequence` flag.
    pub progressive_sequence: bool,
    /// `chroma_format` (2 bits): 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4.
    pub chroma_format: u8,
    /// High 2 bits of the luminance width.
    pub horizontal_size_extension: u8,
    /// High 2 bits of the luminance height.
    pub vertical_size_extension: u8,
    /// `bit_rate_extension` (12 bits).
    pub bit_rate_extension: u32,
    /// `frame_rate_extension_n` (2 bits).
    pub frame_rate_extension_n: u8,
    /// `frame_rate_extension_d` (5 bits).
    pub frame_rate_extension_d: u8,
}

/// Write a `sequence_extension` payload (after the `0xB5` start code).
///
/// `chroma_format` may be 1 (4:2:0), 2 (4:2:2) or 3 (4:4:4); any other value
/// is masked to the low two bits and may be rejected by a strict decoder.
pub fn write_sequence_extension(writer: &mut BitWriter, params: &SequenceExtensionParams) {
    writer.write_bits(SEQUENCE_EXTENSION_ID, 4);
    writer.write_bits(u32::from(params.profile_and_level_indication), 8);
    writer.write_bit(params.progressive_sequence);
    writer.write_bits(u32::from(params.chroma_format) & 0x3, 2);
    writer.write_bits(u32::from(params.horizontal_size_extension), 2);
    writer.write_bits(u32::from(params.vertical_size_extension), 2);
    writer.write_bits(params.bit_rate_extension & 0xFFF, 12);
    writer.write_bit(true); // marker bit
    writer.write_bits(0, 8); // vbv_buffer_size_extension
    writer.write_bit(false); // low_delay
    writer.write_bits(u32::from(params.frame_rate_extension_n), 2);
    writer.write_bits(u32::from(params.frame_rate_extension_d), 5);
}

/// Write an I-picture `picture_header` payload (after the `0x00` start code).
pub fn write_picture_header(writer: &mut BitWriter, temporal_reference: u32, vbv_delay: u32) {
    writer.write_bits(temporal_reference & 0x3FF, 10);
    writer.write_bits(PICTURE_CODING_TYPE_I, 3);
    writer.write_bits(vbv_delay & 0xFFFF, 16);
    // I-pictures carry no full_pel_*_vector or *_f_code fields.
}

/// Parameters for the `picture_coding_extension`.
#[derive(Debug, Clone, Copy)]
pub struct PictureCodingExtensionParams {
    /// `intra_dc_precision` (2 bits): 0..=3 → 8/9/10/11-bit DC.
    pub intra_dc_precision: u8,
    /// `q_scale_type` flag.
    pub q_scale_type: bool,
    /// `intra_vlc_format` flag: 0 → Table B-14, 1 → Table B-15.
    pub intra_vlc_format: bool,
    /// `alternate_scan` flag.
    pub alternate_scan: bool,
    /// `progressive_frame` flag.
    pub progressive_frame: bool,
}

/// Write a `picture_coding_extension` payload (after the `0xB5` start code).
///
/// For an I-frame the four f-codes are all `0xF` (unused / forbidden for intra),
/// `picture_structure` is frame, and `frame_pred_frame_dct` is `1`.
pub fn write_picture_coding_extension(
    writer: &mut BitWriter,
    params: &PictureCodingExtensionParams,
) {
    writer.write_bits(PICTURE_CODING_EXTENSION_ID, 4);
    // f_code[forward/backward][horizontal/vertical] = 0xF each (unused for I).
    for _ in 0..4 {
        writer.write_bits(0xF, 4);
    }
    writer.write_bits(u32::from(params.intra_dc_precision & 0x3), 2);
    writer.write_bits(PICTURE_STRUCTURE_FRAME, 2);
    writer.write_bit(false); // top_field_first
    writer.write_bit(true); // frame_pred_frame_dct
    writer.write_bit(false); // concealment_motion_vectors
    writer.write_bit(params.q_scale_type);
    writer.write_bit(params.intra_vlc_format);
    writer.write_bit(params.alternate_scan);
    writer.write_bit(false); // repeat_first_field
    writer.write_bit(true); // chroma_420_type
    writer.write_bit(params.progressive_frame);
    writer.write_bit(false); // composite_display_flag
}

/// Write a slice header payload (after the slice start code, which itself
/// encodes the vertical position). Emits `quantiser_scale_code` (5 bits) and a
/// `0` `extra_bit_slice` so there is no slice-extension chain.
pub fn write_slice_header(writer: &mut BitWriter, quantiser_scale_code: u8) {
    writer.write_bits(u32::from(quantiser_scale_code & 0x1F), 5);
    writer.write_bit(false); // extra_bit_slice = 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpeg2::bitreader::BitReader;
    use crate::mpeg2::headers::{
        parse_picture_coding_extension, parse_picture_header, parse_sequence_extension,
        parse_sequence_header, parse_slice_header,
    };

    #[test]
    fn sequence_header_round_trips_defaults() {
        let params = SequenceHeaderParams {
            width: 32,
            height: 16,
            aspect_ratio_information: 1,
            frame_rate_code: 3,
            bit_rate_value: 0x3FFFF,
            vbv_buffer_size_value: 112,
            load_default_matrices: false,
        };
        let mut w = BitWriter::new();
        write_sequence_header(&mut w, &params);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let sh = parse_sequence_header(&mut r).expect("parse");
        assert_eq!(sh.horizontal_size_value, 32);
        assert_eq!(sh.vertical_size_value, 16);
        assert_eq!(sh.aspect_ratio_information, 1);
        assert_eq!(sh.frame_rate_code, 3);
        assert_eq!(sh.intra_quantiser_matrix, DEFAULT_INTRA_MATRIX);
    }

    #[test]
    fn sequence_header_round_trips_with_downloaded_matrices() {
        let params = SequenceHeaderParams {
            width: 64,
            height: 48,
            aspect_ratio_information: 2,
            frame_rate_code: 4,
            bit_rate_value: 1000,
            vbv_buffer_size_value: 200,
            load_default_matrices: true,
        };
        let mut w = BitWriter::new();
        write_sequence_header(&mut w, &params);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let sh = parse_sequence_header(&mut r).expect("parse");
        // Downloaded matrices equal the defaults we wrote.
        assert_eq!(sh.intra_quantiser_matrix, DEFAULT_INTRA_MATRIX);
        assert_eq!(sh.non_intra_quantiser_matrix, DEFAULT_NON_INTRA_MATRIX);
    }

    #[test]
    fn sequence_extension_round_trips() {
        let params = SequenceExtensionParams {
            profile_and_level_indication: 0x44,
            progressive_sequence: true,
            chroma_format: 1,
            horizontal_size_extension: 0,
            vertical_size_extension: 0,
            bit_rate_extension: 0,
            frame_rate_extension_n: 0,
            frame_rate_extension_d: 0,
        };
        let mut w = BitWriter::new();
        write_sequence_extension(&mut w, &params);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let se = parse_sequence_extension(&mut r).expect("parse");
        assert_eq!(se.chroma_format, 1);
        assert_eq!(se.profile_and_level_indication, 0x44);
        assert!(se.progressive_sequence);
    }

    #[test]
    fn sequence_extension_writes_all_chroma_formats() {
        for cf in 1u8..=3u8 {
            let params = SequenceExtensionParams {
                profile_and_level_indication: 0x44,
                progressive_sequence: true,
                chroma_format: cf,
                horizontal_size_extension: 0,
                vertical_size_extension: 0,
                bit_rate_extension: 0,
                frame_rate_extension_n: 0,
                frame_rate_extension_d: 0,
            };
            let mut w = BitWriter::new();
            write_sequence_extension(&mut w, &params);
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            let se = parse_sequence_extension(&mut r).expect("parse");
            assert_eq!(se.chroma_format, cf);
        }
    }

    #[test]
    fn picture_header_round_trips() {
        let mut w = BitWriter::new();
        write_picture_header(&mut w, 0, 0xFFFF);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let ph = parse_picture_header(&mut r).expect("parse");
        assert_eq!(ph.picture_coding_type, 1);
        assert_eq!(ph.vbv_delay, 0xFFFF);
    }

    #[test]
    fn picture_coding_extension_round_trips() {
        let params = PictureCodingExtensionParams {
            intra_dc_precision: 2,
            q_scale_type: true,
            intra_vlc_format: true,
            alternate_scan: false,
            progressive_frame: true,
        };
        let mut w = BitWriter::new();
        write_picture_coding_extension(&mut w, &params);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let pce = parse_picture_coding_extension(&mut r).expect("parse");
        assert_eq!(pce.intra_dc_precision, 2);
        assert_eq!(pce.picture_structure, 3);
        assert!(pce.q_scale_type);
        assert!(pce.intra_vlc_format);
        assert!(!pce.alternate_scan);
        assert!(pce.progressive_frame);
    }

    #[test]
    fn slice_header_round_trips() {
        let mut w = BitWriter::new();
        write_slice_header(&mut w, 9);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let sh = parse_slice_header(&mut r, 1).expect("parse");
        assert_eq!(sh.quantiser_scale_code, 9);
        assert_eq!(sh.slice_vertical_position, 1);
    }
}
