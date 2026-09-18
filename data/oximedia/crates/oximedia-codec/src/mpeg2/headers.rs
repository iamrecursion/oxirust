//! MPEG-2 header parsing (ISO/IEC 13818-2 §6.2 / §6.3).
//!
//! The intra-only decoder parses the following structures:
//!
//! - `sequence_header` (§6.2.2.1 / §6.3.3)
//! - `sequence_extension` (§6.2.2.3 / §6.3.5)
//! - `group_of_pictures_header` (§6.2.2.6 / §6.3.8) — fields skipped, presence
//!   detected by the GOP start code
//! - `picture_header` (§6.2.3 / §6.3.9)
//! - `picture_coding_extension` (§6.2.3.1 / §6.3.10)
//! - `slice` header (§6.2.4 / §6.3.16)
//!
//! Start codes themselves are located by the caller (`decode.rs`); the
//! functions here take a [`BitReader`] positioned **immediately after** the
//! relevant start code (i.e. at the first payload bit).

use super::bitreader::BitReader;
use super::dequant::{DEFAULT_INTRA_MATRIX, DEFAULT_NON_INTRA_MATRIX};
use super::zigzag::SCAN_PROGRESSIVE;
use super::Mpeg2Error;
use super::Mpeg2Result;

/// Parsed `sequence_header` together with the matrices that may be downloaded
/// in it.
#[derive(Debug, Clone)]
pub struct SequenceHeader {
    /// `horizontal_size_value` — low 12 bits of the luminance width.
    pub horizontal_size_value: u32,
    /// `vertical_size_value` — low 12 bits of the luminance height.
    pub vertical_size_value: u32,
    /// `aspect_ratio_information` (4 bits).
    pub aspect_ratio_information: u8,
    /// `frame_rate_code` (4 bits).
    pub frame_rate_code: u8,
    /// `bit_rate_value` (18 bits) — in units of 400 bit/s.
    pub bit_rate_value: u32,
    /// `vbv_buffer_size_value` (10 bits).
    pub vbv_buffer_size_value: u32,
    /// `constrained_parameters_flag`.
    pub constrained_parameters_flag: bool,
    /// Active intra quantiser matrix (raster order), default or downloaded.
    pub intra_quantiser_matrix: [u8; 64],
    /// Active non-intra quantiser matrix (raster order), default or downloaded.
    pub non_intra_quantiser_matrix: [u8; 64],
}

/// Parsed `sequence_extension`.
#[derive(Debug, Clone)]
pub struct SequenceExtension {
    /// `profile_and_level_indication` (8 bits).
    pub profile_and_level_indication: u8,
    /// `progressive_sequence` flag.
    pub progressive_sequence: bool,
    /// `chroma_format` (2 bits): 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4.
    pub chroma_format: u8,
    /// `horizontal_size_extension` (2 bits, high bits of width).
    pub horizontal_size_extension: u8,
    /// `vertical_size_extension` (2 bits, high bits of height).
    pub vertical_size_extension: u8,
    /// `bit_rate_extension` (12 bits).
    pub bit_rate_extension: u32,
    /// `frame_rate_extension_n` (2 bits).
    pub frame_rate_extension_n: u8,
    /// `frame_rate_extension_d` (5 bits).
    pub frame_rate_extension_d: u8,
}

/// Parsed `picture_header`.
#[derive(Debug, Clone)]
pub struct PictureHeader {
    /// `temporal_reference` (10 bits).
    pub temporal_reference: u32,
    /// `picture_coding_type` (3 bits): 1 = I, 2 = P, 3 = B.
    pub picture_coding_type: u8,
    /// `vbv_delay` (16 bits).
    pub vbv_delay: u32,
}

/// Parsed `picture_coding_extension`.
#[derive(Debug, Clone)]
pub struct PictureCodingExtension {
    /// f-codes \[forward/backward\]\[horizontal/vertical\]. Not used for I-frames.
    pub f_code: [[u8; 2]; 2],
    /// `intra_dc_precision` (2 bits): 0..=3 → 8/9/10/11-bit DC.
    pub intra_dc_precision: u8,
    /// `picture_structure` (2 bits): 3 = frame picture.
    pub picture_structure: u8,
    /// `top_field_first` flag.
    pub top_field_first: bool,
    /// `frame_pred_frame_dct` flag.
    pub frame_pred_frame_dct: bool,
    /// `concealment_motion_vectors` flag.
    pub concealment_motion_vectors: bool,
    /// `q_scale_type` flag (selects linear vs non-linear quantiser scale).
    pub q_scale_type: bool,
    /// `intra_vlc_format` flag: 0 → Table B-14, 1 → Table B-15.
    pub intra_vlc_format: bool,
    /// `alternate_scan` flag: selects the alternate (Fig 7-3) inverse scan.
    pub alternate_scan: bool,
    /// `repeat_first_field` flag.
    pub repeat_first_field: bool,
    /// `progressive_frame` flag.
    pub progressive_frame: bool,
}

/// Parsed slice header (fields before the macroblock data).
#[derive(Debug, Clone)]
pub struct SliceHeader {
    /// 1-based macroblock row (`slice_vertical_position`), derived from the
    /// slice start code value.
    pub slice_vertical_position: u8,
    /// `quantiser_scale_code` (5 bits).
    pub quantiser_scale_code: u8,
    /// Number of payload bits consumed by the header (so the caller knows where
    /// the macroblock data begins relative to the slice payload).
    pub header_bits: usize,
}

const PICTURE_STRUCTURE_FRAME: u8 = 3;

/// Parse a `sequence_header`. `reader` must be positioned just after the
/// `sequence_header_code` (`0xB3`).
///
/// # Errors
///
/// Returns [`Mpeg2Error::InvalidData`] on a missing marker bit or truncation,
/// [`Mpeg2Error::UnexpectedEof`] if the bitstream is too short.
pub fn parse_sequence_header(reader: &mut BitReader<'_>) -> Mpeg2Result<SequenceHeader> {
    let horizontal_size_value = reader.read_bits(12)?;
    let vertical_size_value = reader.read_bits(12)?;
    let aspect_ratio_information = reader.read_bits(4)? as u8;
    let frame_rate_code = reader.read_bits(4)? as u8;
    let bit_rate_value = reader.read_bits(18)?;
    expect_marker_bit(reader, "sequence_header.marker")?;
    let vbv_buffer_size_value = reader.read_bits(10)?;
    let constrained_parameters_flag = reader.read_bit()?;

    let load_intra = reader.read_bit()?;
    let mut intra_quantiser_matrix = DEFAULT_INTRA_MATRIX;
    if load_intra {
        read_quant_matrix(reader, &mut intra_quantiser_matrix)?;
    }

    let load_non_intra = reader.read_bit()?;
    let mut non_intra_quantiser_matrix = DEFAULT_NON_INTRA_MATRIX;
    if load_non_intra {
        read_quant_matrix(reader, &mut non_intra_quantiser_matrix)?;
    }

    if horizontal_size_value == 0 || vertical_size_value == 0 {
        return Err(Mpeg2Error::InvalidData(
            "sequence_header has zero dimension".into(),
        ));
    }

    Ok(SequenceHeader {
        horizontal_size_value,
        vertical_size_value,
        aspect_ratio_information,
        frame_rate_code,
        bit_rate_value,
        vbv_buffer_size_value,
        constrained_parameters_flag,
        intra_quantiser_matrix,
        non_intra_quantiser_matrix,
    })
}

/// Parse a `sequence_extension`. `reader` must be positioned just after the
/// `extension_start_code` (`0xB5`) **and** the 4-bit extension start code
/// identifier has NOT yet been consumed — this function reads it and verifies
/// it is the Sequence Extension ID (`0001`).
///
/// # Errors
///
/// Returns [`Mpeg2Error::Unsupported`] if the chroma format is not 4:2:0, or
/// [`Mpeg2Error::InvalidData`] on a bad extension id / marker.
pub fn parse_sequence_extension(reader: &mut BitReader<'_>) -> Mpeg2Result<SequenceExtension> {
    let ext_id = reader.read_bits(4)? as u8;
    if ext_id != 0b0001 {
        return Err(Mpeg2Error::InvalidData(format!(
            "expected Sequence Extension id 0001, got {ext_id:04b}"
        )));
    }
    let profile_and_level_indication = reader.read_bits(8)? as u8;
    let progressive_sequence = reader.read_bit()?;
    let chroma_format = reader.read_bits(2)? as u8;
    let horizontal_size_extension = reader.read_bits(2)? as u8;
    let vertical_size_extension = reader.read_bits(2)? as u8;
    let bit_rate_extension = reader.read_bits(12)?;
    expect_marker_bit(reader, "sequence_extension.marker")?;
    let _vbv_buffer_size_extension = reader.read_bits(8)?;
    let _low_delay = reader.read_bit()?;
    let frame_rate_extension_n = reader.read_bits(2)? as u8;
    let frame_rate_extension_d = reader.read_bits(5)? as u8;

    // chroma_format: 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4 (ISO/IEC 13818-2 §6.1.1.4).
    // All three are now supported (Wave 10). The value `0` is reserved.
    if !(1..=3).contains(&chroma_format) {
        return Err(Mpeg2Error::Unsupported(format!(
            "chroma_format {chroma_format} (must be 1, 2 or 3)"
        )));
    }

    Ok(SequenceExtension {
        profile_and_level_indication,
        progressive_sequence,
        chroma_format,
        horizontal_size_extension,
        vertical_size_extension,
        bit_rate_extension,
        frame_rate_extension_n,
        frame_rate_extension_d,
    })
}

/// Parse a `picture_header`. `reader` must be positioned just after the
/// `picture_start_code` (`0x00`).
///
/// Rejects non-I pictures (`picture_coding_type != 1`) with an `Err`.
///
/// # Errors
///
/// Returns [`Mpeg2Error::Unsupported`] for P/B pictures, or
/// [`Mpeg2Error::InvalidData`] for reserved coding types.
pub fn parse_picture_header(reader: &mut BitReader<'_>) -> Mpeg2Result<PictureHeader> {
    let temporal_reference = reader.read_bits(10)?;
    let picture_coding_type = reader.read_bits(3)? as u8;
    let vbv_delay = reader.read_bits(16)?;

    match picture_coding_type {
        1 => {}
        2 | 3 => {
            return Err(Mpeg2Error::Unsupported(format!(
                "picture_coding_type {picture_coding_type} (P/B not supported; I-frames only)"
            )));
        }
        other => {
            return Err(Mpeg2Error::InvalidData(format!(
                "reserved picture_coding_type {other}"
            )));
        }
    }

    // I-pictures carry no further conditional fields (no full_pel_*_vector or
    // *_f_code) before the byte alignment / extensions.

    Ok(PictureHeader {
        temporal_reference,
        picture_coding_type,
        vbv_delay,
    })
}

/// Parse a `picture_coding_extension`. `reader` must be positioned just after
/// the `extension_start_code` (`0xB5`); this reads the 4-bit extension id and
/// verifies it is the Picture Coding Extension ID (`1000`).
///
/// # Errors
///
/// Returns [`Mpeg2Error::Unsupported`] for non-frame picture structures, or
/// [`Mpeg2Error::InvalidData`] for a bad extension id.
pub fn parse_picture_coding_extension(
    reader: &mut BitReader<'_>,
) -> Mpeg2Result<PictureCodingExtension> {
    let ext_id = reader.read_bits(4)? as u8;
    if ext_id != 0b1000 {
        return Err(Mpeg2Error::InvalidData(format!(
            "expected Picture Coding Extension id 1000, got {ext_id:04b}"
        )));
    }

    let mut f_code = [[0u8; 2]; 2];
    for dir in &mut f_code {
        for c in dir.iter_mut() {
            *c = reader.read_bits(4)? as u8;
        }
    }

    let intra_dc_precision = reader.read_bits(2)? as u8;
    let picture_structure = reader.read_bits(2)? as u8;
    let top_field_first = reader.read_bit()?;
    let frame_pred_frame_dct = reader.read_bit()?;
    let concealment_motion_vectors = reader.read_bit()?;
    let q_scale_type = reader.read_bit()?;
    let intra_vlc_format = reader.read_bit()?;
    let alternate_scan = reader.read_bit()?;
    let repeat_first_field = reader.read_bit()?;
    let _chroma_420_type = reader.read_bit()?;
    let progressive_frame = reader.read_bit()?;
    let composite_display_flag = reader.read_bit()?;
    if composite_display_flag {
        // v_axis(1) field_sequence(3) sub_carrier(1) burst_amplitude(7) sub_carrier_phase(8)
        let _ = reader.read_bits(20)?;
    }

    if picture_structure != PICTURE_STRUCTURE_FRAME {
        return Err(Mpeg2Error::Unsupported(format!(
            "picture_structure {picture_structure} (only frame pictures == 3 supported)"
        )));
    }

    Ok(PictureCodingExtension {
        f_code,
        intra_dc_precision,
        picture_structure,
        top_field_first,
        frame_pred_frame_dct,
        concealment_motion_vectors,
        q_scale_type,
        intra_vlc_format,
        alternate_scan,
        repeat_first_field,
        progressive_frame,
    })
}

/// Parse a slice header. `reader` must be positioned just after the slice start
/// code; `slice_start_code` is the start-code byte value (which encodes the
/// vertical position for streams ≤ 2800 lines).
///
/// # Errors
///
/// Returns [`Mpeg2Error::UnexpectedEof`] on truncation.
pub fn parse_slice_header(
    reader: &mut BitReader<'_>,
    slice_start_code: u8,
) -> Mpeg2Result<SliceHeader> {
    let start_bits = reader.remaining_bits();

    // For vertical_size <= 2800, slice_vertical_position == slice_start_code.
    let slice_vertical_position = slice_start_code;

    let quantiser_scale_code = reader.read_bits(5)? as u8;

    // Optional slice extension flag chain: if the next bit is 1, there is
    // intra_slice_flag + intra_slice + reserved + extra_information_slice loop.
    let extra_bit_slice_present = reader.read_bit()?;
    if extra_bit_slice_present {
        // slice_extension_flag was 1: read intra_slice_flag etc.
        let _intra_slice = reader.read_bit()?;
        let _reserved_bits = reader.read_bits(7)?;
        // extra_information_slice loop: each iteration is extra_bit_slice(1) +
        // extra_information(8). Loop while extra_bit_slice == 1.
        loop {
            let extra = reader.read_bit()?;
            if !extra {
                break;
            }
            let _ = reader.read_bits(8)?;
        }
    }

    let header_bits = start_bits - reader.remaining_bits();

    Ok(SliceHeader {
        slice_vertical_position,
        quantiser_scale_code,
        header_bits,
    })
}

/// Combined luminance width from base value plus 2-bit extension.
#[must_use]
pub fn full_horizontal_size(seq: &SequenceHeader, ext: &SequenceExtension) -> u32 {
    (u32::from(ext.horizontal_size_extension) << 12) | seq.horizontal_size_value
}

/// Combined luminance height from base value plus 2-bit extension.
#[must_use]
pub fn full_vertical_size(seq: &SequenceHeader, ext: &SequenceExtension) -> u32 {
    (u32::from(ext.vertical_size_extension) << 12) | seq.vertical_size_value
}

/// Read a 64-entry quantiser matrix from the bitstream (8 bits per entry) and
/// store it in **raster** order. The stream stores it in zig-zag scan order.
fn read_quant_matrix(reader: &mut BitReader<'_>, out: &mut [u8; 64]) -> Mpeg2Result<()> {
    for scan_idx in 0..64 {
        let value = reader.read_bits(8)? as u8;
        out[SCAN_PROGRESSIVE[scan_idx]] = value;
    }
    Ok(())
}

/// Read one marker bit and require it to be `1` (ISO/IEC 13818-2 markers are
/// always `1`).
fn expect_marker_bit(reader: &mut BitReader<'_>, what: &str) -> Mpeg2Result<()> {
    if reader.read_bit()? {
        Ok(())
    } else {
        Err(Mpeg2Error::InvalidData(format!(
            "{what}: expected marker bit = 1"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Append `value` as `len` bits (MSB-first) into a growing bit vector.
    fn push_bits(bits: &mut Vec<u8>, value: u32, len: u8) {
        for i in (0..len).rev() {
            bits.push(((value >> i) & 1) as u8);
        }
    }

    fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
        let mut padded = bits.to_vec();
        while padded.len() % 8 != 0 {
            padded.push(0);
        }
        padded
            .chunks(8)
            .map(|c| c.iter().fold(0u8, |acc, &b| (acc << 1) | b))
            .collect()
    }

    #[test]
    fn parse_minimal_sequence_header() {
        let mut bits = Vec::new();
        push_bits(&mut bits, 16, 12); // horizontal
        push_bits(&mut bits, 16, 12); // vertical
        push_bits(&mut bits, 1, 4); // aspect ratio
        push_bits(&mut bits, 3, 4); // frame rate code
        push_bits(&mut bits, 0x3FFFF, 18); // bit rate
        push_bits(&mut bits, 1, 1); // marker
        push_bits(&mut bits, 100, 10); // vbv buffer
        push_bits(&mut bits, 0, 1); // constrained
        push_bits(&mut bits, 0, 1); // load_intra = 0
        push_bits(&mut bits, 0, 1); // load_non_intra = 0
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        let sh = parse_sequence_header(&mut r).expect("parse");
        assert_eq!(sh.horizontal_size_value, 16);
        assert_eq!(sh.vertical_size_value, 16);
        assert_eq!(sh.aspect_ratio_information, 1);
        assert_eq!(sh.frame_rate_code, 3);
        assert_eq!(sh.bit_rate_value, 0x3FFFF);
        assert_eq!(sh.intra_quantiser_matrix, DEFAULT_INTRA_MATRIX);
    }

    #[test]
    fn sequence_header_zero_dimension_errors() {
        let mut bits = Vec::new();
        push_bits(&mut bits, 0, 12); // horizontal = 0 → invalid
        push_bits(&mut bits, 16, 12);
        push_bits(&mut bits, 1, 4);
        push_bits(&mut bits, 3, 4);
        push_bits(&mut bits, 0, 18);
        push_bits(&mut bits, 1, 1);
        push_bits(&mut bits, 0, 10);
        push_bits(&mut bits, 0, 1);
        push_bits(&mut bits, 0, 1);
        push_bits(&mut bits, 0, 1);
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        assert!(parse_sequence_header(&mut r).is_err());
    }

    #[test]
    fn sequence_header_bad_marker_errors() {
        let mut bits = Vec::new();
        push_bits(&mut bits, 16, 12);
        push_bits(&mut bits, 16, 12);
        push_bits(&mut bits, 1, 4);
        push_bits(&mut bits, 3, 4);
        push_bits(&mut bits, 0, 18);
        push_bits(&mut bits, 0, 1); // marker = 0 → invalid
        push_bits(&mut bits, 0, 10);
        push_bits(&mut bits, 0, 1);
        push_bits(&mut bits, 0, 1);
        push_bits(&mut bits, 0, 1);
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        assert!(parse_sequence_header(&mut r).is_err());
    }

    #[test]
    fn parse_sequence_extension_420() {
        let mut bits = Vec::new();
        push_bits(&mut bits, 0b0001, 4); // ext id
        push_bits(&mut bits, 0x44, 8); // profile/level
        push_bits(&mut bits, 0, 1); // progressive_sequence
        push_bits(&mut bits, 1, 2); // chroma_format = 4:2:0
        push_bits(&mut bits, 0, 2); // hsize ext
        push_bits(&mut bits, 0, 2); // vsize ext
        push_bits(&mut bits, 0, 12); // bit_rate_ext
        push_bits(&mut bits, 1, 1); // marker
        push_bits(&mut bits, 0, 8); // vbv buffer ext
        push_bits(&mut bits, 0, 1); // low delay
        push_bits(&mut bits, 0, 2); // fr ext n
        push_bits(&mut bits, 0, 5); // fr ext d
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        let se = parse_sequence_extension(&mut r).expect("parse");
        assert_eq!(se.chroma_format, 1);
        assert_eq!(se.profile_and_level_indication, 0x44);
    }

    #[test]
    fn sequence_extension_accepts_all_chroma_formats() {
        // Wave 10: 4:2:0 / 4:2:2 / 4:4:4 are all valid; only `0` is reserved.
        for chroma_format in 1u8..=3u8 {
            let mut bits = Vec::new();
            push_bits(&mut bits, 0b0001, 4);
            push_bits(&mut bits, 0x44, 8);
            push_bits(&mut bits, 0, 1);
            push_bits(&mut bits, u32::from(chroma_format), 2);
            push_bits(&mut bits, 0, 2);
            push_bits(&mut bits, 0, 2);
            push_bits(&mut bits, 0, 12);
            push_bits(&mut bits, 1, 1);
            push_bits(&mut bits, 0, 8);
            push_bits(&mut bits, 0, 1);
            push_bits(&mut bits, 0, 2);
            push_bits(&mut bits, 0, 5);
            let bytes = bits_to_bytes(&bits);
            let mut r = BitReader::new(&bytes);
            let se = parse_sequence_extension(&mut r).expect("parse");
            assert_eq!(se.chroma_format, chroma_format);
        }
    }

    #[test]
    fn sequence_extension_rejects_reserved_chroma_format() {
        let mut bits = Vec::new();
        push_bits(&mut bits, 0b0001, 4);
        push_bits(&mut bits, 0x44, 8);
        push_bits(&mut bits, 0, 1);
        push_bits(&mut bits, 0, 2); // chroma_format = 0 (reserved) → reject
        push_bits(&mut bits, 0, 2);
        push_bits(&mut bits, 0, 2);
        push_bits(&mut bits, 0, 12);
        push_bits(&mut bits, 1, 1);
        push_bits(&mut bits, 0, 8);
        push_bits(&mut bits, 0, 1);
        push_bits(&mut bits, 0, 2);
        push_bits(&mut bits, 0, 5);
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        assert!(matches!(
            parse_sequence_extension(&mut r),
            Err(Mpeg2Error::Unsupported(_))
        ));
    }

    #[test]
    fn parse_i_picture_header() {
        let mut bits = Vec::new();
        push_bits(&mut bits, 0, 10); // temporal_reference
        push_bits(&mut bits, 1, 3); // picture_coding_type = I
        push_bits(&mut bits, 0xFFFF, 16); // vbv_delay
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        let ph = parse_picture_header(&mut r).expect("parse");
        assert_eq!(ph.picture_coding_type, 1);
        assert_eq!(ph.vbv_delay, 0xFFFF);
    }

    #[test]
    fn picture_header_rejects_p_frame() {
        let mut bits = Vec::new();
        push_bits(&mut bits, 0, 10);
        push_bits(&mut bits, 2, 3); // P-frame → reject
        push_bits(&mut bits, 0, 16);
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        assert!(matches!(
            parse_picture_header(&mut r),
            Err(Mpeg2Error::Unsupported(_))
        ));
    }

    #[test]
    fn parse_picture_coding_extension_frame() {
        let mut bits = Vec::new();
        push_bits(&mut bits, 0b1000, 4); // ext id
        push_bits(&mut bits, 0xF, 4); // f_code[0][0]
        push_bits(&mut bits, 0xF, 4); // f_code[0][1]
        push_bits(&mut bits, 0xF, 4); // f_code[1][0]
        push_bits(&mut bits, 0xF, 4); // f_code[1][1]
        push_bits(&mut bits, 2, 2); // intra_dc_precision = 2 (10-bit)
        push_bits(&mut bits, 3, 2); // picture_structure = frame
        push_bits(&mut bits, 1, 1); // top_field_first
        push_bits(&mut bits, 1, 1); // frame_pred_frame_dct
        push_bits(&mut bits, 0, 1); // concealment mv
        push_bits(&mut bits, 1, 1); // q_scale_type
        push_bits(&mut bits, 1, 1); // intra_vlc_format
        push_bits(&mut bits, 1, 1); // alternate_scan
        push_bits(&mut bits, 0, 1); // repeat_first_field
        push_bits(&mut bits, 1, 1); // chroma_420_type
        push_bits(&mut bits, 1, 1); // progressive_frame
        push_bits(&mut bits, 0, 1); // composite_display_flag = 0
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        let pce = parse_picture_coding_extension(&mut r).expect("parse");
        assert_eq!(pce.intra_dc_precision, 2);
        assert_eq!(pce.picture_structure, 3);
        assert!(pce.q_scale_type);
        assert!(pce.intra_vlc_format);
        assert!(pce.alternate_scan);
    }

    #[test]
    fn picture_coding_extension_rejects_field() {
        let mut bits = Vec::new();
        push_bits(&mut bits, 0b1000, 4);
        for _ in 0..4 {
            push_bits(&mut bits, 0xF, 4);
        }
        push_bits(&mut bits, 0, 2); // intra_dc_precision
        push_bits(&mut bits, 1, 2); // picture_structure = top field → reject
        push_bits(&mut bits, 0, 1);
        push_bits(&mut bits, 0, 1);
        push_bits(&mut bits, 0, 1);
        push_bits(&mut bits, 0, 1);
        push_bits(&mut bits, 0, 1);
        push_bits(&mut bits, 0, 1);
        push_bits(&mut bits, 0, 1);
        push_bits(&mut bits, 0, 1);
        push_bits(&mut bits, 0, 1);
        push_bits(&mut bits, 0, 1);
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        assert!(matches!(
            parse_picture_coding_extension(&mut r),
            Err(Mpeg2Error::Unsupported(_))
        ));
    }

    #[test]
    fn parse_simple_slice_header() {
        let mut bits = Vec::new();
        push_bits(&mut bits, 9, 5); // quantiser_scale_code
        push_bits(&mut bits, 0, 1); // no slice extension flag
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        let sh = parse_slice_header(&mut r, 1).expect("parse");
        assert_eq!(sh.slice_vertical_position, 1);
        assert_eq!(sh.quantiser_scale_code, 9);
        assert_eq!(sh.header_bits, 6);
    }

    #[test]
    fn full_size_combines_extensions() {
        let seq = SequenceHeader {
            horizontal_size_value: 0x080,
            vertical_size_value: 0x040,
            aspect_ratio_information: 1,
            frame_rate_code: 3,
            bit_rate_value: 0,
            vbv_buffer_size_value: 0,
            constrained_parameters_flag: false,
            intra_quantiser_matrix: DEFAULT_INTRA_MATRIX,
            non_intra_quantiser_matrix: DEFAULT_NON_INTRA_MATRIX,
        };
        let ext = SequenceExtension {
            profile_and_level_indication: 0,
            progressive_sequence: false,
            chroma_format: 1,
            horizontal_size_extension: 1,
            vertical_size_extension: 0,
            bit_rate_extension: 0,
            frame_rate_extension_n: 0,
            frame_rate_extension_d: 0,
        };
        assert_eq!(full_horizontal_size(&seq, &ext), (1 << 12) | 0x080);
        assert_eq!(full_vertical_size(&seq, &ext), 0x040);
    }
}
