//! Integration tests for the MPEG-2 I-frame decoder (`mpeg2` feature).
//!
//! These tests build minimal but spec-conformant synthetic bitstreams by
//! writing the exact bits the decoder consumes (the test controls both sides),
//! then exercise the public decode path end-to-end.

#![cfg(feature = "mpeg2")]

use oximedia_codec::mpeg2::bitreader::{
    BitReader, EXTENSION_START_CODE, PICTURE_START_CODE, SEQUENCE_HEADER_CODE,
};
use oximedia_codec::mpeg2::dequant::{dequantize_intra, intra_dc_mult, DEFAULT_INTRA_MATRIX};
use oximedia_codec::mpeg2::headers::parse_sequence_header;
use oximedia_codec::mpeg2::idct::idct_8x8;
use oximedia_codec::mpeg2::Mpeg2Decoder;

/// Minimal MSB-first bit writer for assembling test bitstreams.
#[derive(Default)]
struct BitWriter {
    bits: Vec<u8>,
}

impl BitWriter {
    fn new() -> Self {
        Self::default()
    }

    /// Append the low `len` bits of `value`, MSB first.
    fn put(&mut self, value: u32, len: u8) {
        for i in (0..len).rev() {
            self.bits.push(((value >> i) & 1) as u8);
        }
    }

    /// Pad with zero bits to the next byte boundary.
    fn align(&mut self) {
        while self.bits.len() % 8 != 0 {
            self.bits.push(0);
        }
    }

    /// Collapse the accumulated bits into bytes (must be byte-aligned).
    fn into_bytes(mut self) -> Vec<u8> {
        self.align();
        self.bits
            .chunks(8)
            .map(|c| c.iter().fold(0u8, |acc, &b| (acc << 1) | b))
            .collect()
    }
}

/// Push a 4-byte start code `00 00 01 <code>` into `out`.
fn push_start_code(out: &mut Vec<u8>, code: u8) {
    out.extend_from_slice(&[0x00, 0x00, 0x01, code]);
}

/// Build a sequence-header payload (without the start code) for a
/// `width × height` 4:2:0 stream with default quant matrices.
fn sequence_header_payload(width: u32, height: u32) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put(width & 0xFFF, 12); // horizontal_size_value
    w.put(height & 0xFFF, 12); // vertical_size_value
    w.put(1, 4); // aspect_ratio_information
    w.put(3, 4); // frame_rate_code (25 fps)
    w.put(0x3FFFF, 18); // bit_rate_value
    w.put(1, 1); // marker bit
    w.put(112, 10); // vbv_buffer_size_value
    w.put(0, 1); // constrained_parameters_flag
    w.put(0, 1); // load_intra_quantiser_matrix = 0
    w.put(0, 1); // load_non_intra_quantiser_matrix = 0
    w.into_bytes()
}

/// Build a sequence-extension payload (without start code) for 4:2:0.
fn sequence_extension_payload() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put(0b0001, 4); // extension id = Sequence Extension
    w.put(0x44, 8); // profile_and_level (Main@Main)
    w.put(0, 1); // progressive_sequence
    w.put(1, 2); // chroma_format = 4:2:0
    w.put(0, 2); // horizontal_size_extension
    w.put(0, 2); // vertical_size_extension
    w.put(0, 12); // bit_rate_extension
    w.put(1, 1); // marker bit
    w.put(0, 8); // vbv_buffer_size_extension
    w.put(0, 1); // low_delay
    w.put(0, 2); // frame_rate_extension_n
    w.put(0, 5); // frame_rate_extension_d
    w.into_bytes()
}

/// Build a picture-header payload (without start code) for an I-picture.
fn picture_header_payload() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put(0, 10); // temporal_reference
    w.put(1, 3); // picture_coding_type = I
    w.put(0xFFFF, 16); // vbv_delay
    w.into_bytes()
}

/// Build a picture-coding-extension payload (without start code).
///
/// `intra_dc_precision`, `intra_vlc_format`, `alternate_scan` are configurable.
fn picture_coding_extension_payload(
    intra_dc_precision: u8,
    intra_vlc_format: bool,
    alternate_scan: bool,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put(0b1000, 4); // extension id = Picture Coding Extension
    w.put(0xF, 4); // f_code[0][0] (unused for I)
    w.put(0xF, 4); // f_code[0][1]
    w.put(0xF, 4); // f_code[1][0]
    w.put(0xF, 4); // f_code[1][1]
    w.put(u32::from(intra_dc_precision), 2);
    w.put(3, 2); // picture_structure = frame
    w.put(0, 1); // top_field_first
    w.put(1, 1); // frame_pred_frame_dct
    w.put(0, 1); // concealment_motion_vectors
    w.put(0, 1); // q_scale_type (linear)
    w.put(u32::from(intra_vlc_format), 1);
    w.put(u32::from(alternate_scan), 1);
    w.put(0, 1); // repeat_first_field
    w.put(1, 1); // chroma_420_type
    w.put(1, 1); // progressive_frame
    w.put(0, 1); // composite_display_flag
    w.into_bytes()
}

/// Append a single grey (DC-only, zero-differential) intra macroblock to `w`.
///
/// `quantiser_scale_code` is written in the slice header by the caller; here we
/// only emit `macroblock_address_increment = 1`, `macroblock_type = Intra`, and
/// the six DC-only blocks (size 0 → DC differential 0, then AC EOB).
fn put_grey_macroblock(w: &mut BitWriter) {
    w.put(0b1, 1); // macroblock_address_increment = 1
    w.put(0b1, 1); // macroblock_type = Intra (no quant change)
    for blk in 0..6 {
        if blk < 4 {
            // Luma DC size 0 → code `100`.
            w.put(0b100, 3);
        } else {
            // Chroma DC size 0 → code `00`.
            w.put(0b00, 2);
        }
        // AC end-of-block (Table B-14) → `10`.
        w.put(0b10, 2);
    }
}

/// Assemble a full single-slice grey I-frame elementary stream.
fn build_grey_iframe(width: u32, height: u32) -> Vec<u8> {
    let mut stream = Vec::new();

    push_start_code(&mut stream, SEQUENCE_HEADER_CODE);
    stream.extend_from_slice(&sequence_header_payload(width, height));

    push_start_code(&mut stream, EXTENSION_START_CODE);
    stream.extend_from_slice(&sequence_extension_payload());

    push_start_code(&mut stream, PICTURE_START_CODE);
    stream.extend_from_slice(&picture_header_payload());

    push_start_code(&mut stream, EXTENSION_START_CODE);
    stream.extend_from_slice(&picture_coding_extension_payload(0, false, false));

    // One slice covering macroblock row 1 (slice_start_code == vertical pos).
    push_start_code(&mut stream, 0x01);
    let mut slice = BitWriter::new();
    slice.put(2, 5); // quantiser_scale_code = 2 → linear scale 4
    slice.put(0, 1); // no slice extension flag

    // Number of macroblocks in the row = ceil(width / 16).
    let mb_cols = (width as usize).div_ceil(16);
    for _ in 0..mb_cols {
        put_grey_macroblock(&mut slice);
    }
    stream.extend_from_slice(&slice.into_bytes());

    // Sequence end code.
    push_start_code(&mut stream, 0xB7);

    stream
}

#[test]
fn parse_sequence_header_synthetic() {
    let payload = sequence_header_payload(16, 16);
    let mut reader = BitReader::new(&payload);
    let sh = parse_sequence_header(&mut reader).expect("parse sequence header");
    assert_eq!(sh.horizontal_size_value, 16);
    assert_eq!(sh.vertical_size_value, 16);
    assert_eq!(sh.aspect_ratio_information, 1);
    assert_eq!(sh.frame_rate_code, 3);
    // Default intra matrix when not downloaded.
    assert_eq!(sh.intra_quantiser_matrix, DEFAULT_INTRA_MATRIX);
}

#[test]
fn idct_dc_only_block() {
    // A block with only DC set produces a flat spatial block equal to DC / 8.
    let mut coeffs = [0i32; 64];
    coeffs[0] = 1024; // 1024 / 8 = 128
    let out = idct_8x8(&coeffs);
    for &v in &out {
        assert!((v - 128).abs() <= 1, "expected flat ~128, got {v}");
    }

    // A different DC value.
    let mut coeffs = [0i32; 64];
    coeffs[0] = 512;
    let out = idct_8x8(&coeffs);
    for &v in &out {
        assert!((v - 64).abs() <= 1, "expected flat ~64, got {v}");
    }
}

#[test]
fn dequant_intra_dc_precision() {
    // F[0][0] = intra_dc_mult * QF[0][0] for each precision (within saturation).
    for (prec, mult) in [(0u8, 8i32), (1, 4), (2, 2), (3, 1)] {
        assert_eq!(intra_dc_mult(prec), mult);
        let mut q = [0i32; 64];
        q[0] = 5;
        let f = dequantize_intra(&q, &DEFAULT_INTRA_MATRIX, prec, 4);
        assert_eq!(f[0], 5 * mult, "precision {prec}");
    }
}

#[test]
fn decode_intra_macroblock_constant_grey() {
    // A 16×16 single-macroblock grey I-frame. Every block is DC-only with zero
    // differential, so the DC predictor (reset to 128) gives a flat mid-grey.
    let stream = build_grey_iframe(16, 16);
    let decoder = Mpeg2Decoder::new();
    let frame = decoder.decode(&stream).expect("decode grey frame");

    assert_eq!(frame.width, 16);
    assert_eq!(frame.height, 16);
    assert_eq!(frame.y.len(), 16 * 16);
    assert_eq!(frame.cb.len(), 8 * 8);
    assert_eq!(frame.cr.len(), 8 * 8);

    // All luma samples within ±2 LSB of mid-grey 128.
    for (i, &v) in frame.y.iter().enumerate() {
        assert!(
            (i32::from(v) - 128).abs() <= 2,
            "luma[{i}] = {v}, expected ~128"
        );
    }
    for (i, &v) in frame.cb.iter().enumerate() {
        assert!(
            (i32::from(v) - 128).abs() <= 2,
            "cb[{i}] = {v}, expected ~128"
        );
    }
    for (i, &v) in frame.cr.iter().enumerate() {
        assert!(
            (i32::from(v) - 128).abs() <= 2,
            "cr[{i}] = {v}, expected ~128"
        );
    }
}

#[test]
fn decode_wider_grey_frame() {
    // 32×16 → two macroblocks in a single row, exercising mb_address increment.
    let stream = build_grey_iframe(32, 16);
    let decoder = Mpeg2Decoder::new();
    let frame = decoder.decode(&stream).expect("decode 32x16 grey");
    assert_eq!(frame.width, 32);
    assert_eq!(frame.height, 16);
    assert_eq!(frame.y.len(), 32 * 16);
    for &v in &frame.y {
        assert!((i32::from(v) - 128).abs() <= 2, "luma {v} expected ~128");
    }
}

#[test]
fn reject_truncated_stream() {
    // A stream cut off after the sequence header start code prefix must error,
    // not panic.
    let mut stream = build_grey_iframe(16, 16);
    stream.truncate(8);
    let decoder = Mpeg2Decoder::new();
    assert!(decoder.decode(&stream).is_err());

    // Empty input.
    assert!(Mpeg2Decoder::new().decode(&[]).is_err());

    // Garbage with no start codes.
    assert!(Mpeg2Decoder::new()
        .decode(&[0xDE, 0xAD, 0xBE, 0xEF])
        .is_err());
}

#[test]
fn reject_p_picture() {
    // Build a stream whose picture header declares a P-picture; decode must
    // return Err rather than attempting motion compensation.
    let mut stream = Vec::new();
    push_start_code(&mut stream, SEQUENCE_HEADER_CODE);
    stream.extend_from_slice(&sequence_header_payload(16, 16));
    push_start_code(&mut stream, EXTENSION_START_CODE);
    stream.extend_from_slice(&sequence_extension_payload());
    push_start_code(&mut stream, PICTURE_START_CODE);
    // P-picture header.
    let mut ph = BitWriter::new();
    ph.put(0, 10); // temporal_reference
    ph.put(2, 3); // picture_coding_type = P
    ph.put(0, 16); // vbv_delay
    ph.put(0, 1); // full_pel_forward_vector
    ph.put(0b111, 3); // forward_f_code
    stream.extend_from_slice(&ph.into_bytes());

    let decoder = Mpeg2Decoder::new();
    assert!(decoder.decode(&stream).is_err());
}
