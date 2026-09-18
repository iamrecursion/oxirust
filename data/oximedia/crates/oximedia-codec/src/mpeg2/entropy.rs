//! Intra macroblock entropy decode for MPEG-2 (ISO/IEC 13818-2 §7.2).
//!
//! For a 4:2:0 intra macroblock six 8×8 blocks are coded in order: four
//! luminance blocks (Y0..Y3) then one Cb and one Cr block. For each block:
//!
//! 1. **DC**: the `dct_dc_size` category is read via Table B-12 (luma) or B-13
//!    (chroma). If the size is non-zero, that many `dct_dc_differential` bits
//!    follow in offset-binary form; the differential is added to the per
//!    component DC predictor.
//! 2. **AC**: run/level pairs are read via Table B-14 (or B-15 when
//!    `intra_vlc_format == 1`) until an end-of-block code. Each non-EOB code
//!    carries a zero `run` then a `level` whose sign bit follows; the escape
//!    code (`0000 01`) is followed by a 6-bit run and a 12-bit signed level.
//!
//! The decoded quantised coefficients are returned in **raster** order via the
//! active inverse scan, ready for [`super::dequant::dequantize_intra`].
//!
//! DC predictors are reset to `1 << (7 + intra_dc_precision)` at the start of
//! every slice and whenever a non-intra-coded run would interrupt the
//! prediction chain (not applicable here as every macroblock is intra).

use super::bitreader::BitReader;
use super::vlc_tables::{
    match_dc_size, match_vlc, AcSymbol, AcTablePtr, DcTablePtr, AC_TABLE_B14, AC_TABLE_B15,
    DC_SIZE_CHROMA, DC_SIZE_LUMA,
};
use super::zigzag::{place_in_raster, scan_table};
use super::Mpeg2Error;
use super::Mpeg2Result;

/// Per-component DC predictors (Y, Cb, Cr) used by the DPCM DC reconstruction.
#[derive(Debug, Clone, Copy)]
pub struct DcPredictors {
    /// Luminance DC predictor.
    pub y: i32,
    /// Cb DC predictor.
    pub cb: i32,
    /// Cr DC predictor.
    pub cr: i32,
}

impl DcPredictors {
    /// Create predictors reset to the value mandated by `intra_dc_precision`:
    /// `1 << (7 + intra_dc_precision)` (128 for 8-bit DC, 256 for 9-bit, …).
    #[must_use]
    pub fn reset(intra_dc_precision: u8) -> Self {
        let reset_value = 1i32 << (7 + i32::from(intra_dc_precision & 0x03));
        Self {
            y: reset_value,
            cb: reset_value,
            cr: reset_value,
        }
    }
}

/// Which block of the macroblock is being decoded (selects the DC table and the
/// predictor).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockComponent {
    /// One of the four luminance blocks.
    Luma,
    /// The Cb chrominance block.
    Cb,
    /// The Cr chrominance block.
    Cr,
}

/// Decode one DC coefficient (the quantised `QF[0][0]`) using DPCM from the
/// relevant predictor in `predictors`, updating that predictor in place.
///
/// Returns the new quantised DC value for the block.
pub fn decode_dc(
    reader: &mut BitReader<'_>,
    predictors: &mut DcPredictors,
    component: BlockComponent,
) -> Mpeg2Result<i32> {
    let dc_table: DcTablePtr = match component {
        BlockComponent::Luma => DC_SIZE_LUMA,
        BlockComponent::Cb | BlockComponent::Cr => DC_SIZE_CHROMA,
    };

    let peek = reader.peek_bits_msb_aligned();
    let (dc_size, consumed) = match_dc_size(dc_table, peek)?;
    reader.skip_bits(consumed)?;

    let dct_dc_differential = if dc_size == 0 {
        0i32
    } else {
        let raw = reader.read_bits(dc_size)? as i32;
        // Offset-binary: if the leading bit is 0 the value is negative.
        if (raw >> (dc_size - 1)) & 1 == 1 {
            raw
        } else {
            raw - ((1i32 << dc_size) - 1)
        }
    };

    let predictor = match component {
        BlockComponent::Luma => &mut predictors.y,
        BlockComponent::Cb => &mut predictors.cb,
        BlockComponent::Cr => &mut predictors.cr,
    };
    *predictor += dct_dc_differential;
    Ok(*predictor)
}

/// Decode the AC coefficients of one intra block into `block` (raster order).
///
/// `block[0]` (DC) must already be set by the caller. The active AC table and
/// scan are selected by `intra_vlc_format` and `alternate_scan` respectively.
pub fn decode_ac(
    reader: &mut BitReader<'_>,
    block: &mut [i32; 64],
    intra_vlc_format: bool,
    alternate_scan: bool,
) -> Mpeg2Result<()> {
    let ac_table: AcTablePtr = if intra_vlc_format {
        AC_TABLE_B15
    } else {
        AC_TABLE_B14
    };
    let scan = scan_table(alternate_scan);

    // Scan position 0 is DC; AC fills 1..=63.
    let mut scan_index: usize = 1;

    loop {
        if scan_index >= 64 {
            break;
        }
        let peek = reader.peek_bits_msb_aligned();
        let symbol = match_vlc(ac_table, peek)?;
        match symbol {
            AcSymbol::EndOfBlock { bits } => {
                reader.skip_bits(bits)?;
                break;
            }
            AcSymbol::RunLevel { run, level, bits } => {
                reader.skip_bits(bits)?;
                let sign = reader.read_bit()?;
                scan_index += run as usize;
                if scan_index >= 64 {
                    return Err(Mpeg2Error::VlcDecode(format!(
                        "AC run overflowed block (scan_index {scan_index})"
                    )));
                }
                let value = if sign {
                    -i32::from(level)
                } else {
                    i32::from(level)
                };
                place_in_raster(block, scan, scan_index, value);
                scan_index += 1;
            }
            AcSymbol::Escape { bits } => {
                reader.skip_bits(bits)?;
                // Fixed-length escape (MPEG-2): 6-bit run, 12-bit signed level.
                let run = reader.read_bits(6)? as usize;
                let level_raw = reader.read_bits(12)? as i32;
                // 12-bit two's complement sign extension.
                let level = if level_raw & 0x800 != 0 {
                    level_raw - 0x1000
                } else {
                    level_raw
                };
                if level == 0 {
                    return Err(Mpeg2Error::VlcDecode("escape level 0 is forbidden".into()));
                }
                scan_index += run;
                if scan_index >= 64 {
                    return Err(Mpeg2Error::VlcDecode(format!(
                        "escape run overflowed block (scan_index {scan_index})"
                    )));
                }
                place_in_raster(block, scan, scan_index, level);
                scan_index += 1;
            }
        }
    }

    Ok(())
}

/// Decode one full intra block (DC + AC) into raster-ordered quantised
/// coefficients.
///
/// `block[0]` is set to the DPCM-reconstructed DC; AC coefficients are placed
/// by the active inverse scan.
pub fn decode_intra_block(
    reader: &mut BitReader<'_>,
    predictors: &mut DcPredictors,
    component: BlockComponent,
    intra_vlc_format: bool,
    alternate_scan: bool,
) -> Mpeg2Result<[i32; 64]> {
    let mut block = [0i32; 64];
    block[0] = decode_dc(reader, predictors, component)?;
    decode_ac(reader, &mut block, intra_vlc_format, alternate_scan)?;
    Ok(block)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Append `value` as `len` bits (MSB-first).
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
    fn dc_predictor_reset_values() {
        assert_eq!(DcPredictors::reset(0).y, 128);
        assert_eq!(DcPredictors::reset(1).y, 256);
        assert_eq!(DcPredictors::reset(2).y, 512);
        assert_eq!(DcPredictors::reset(3).y, 1024);
    }

    #[test]
    fn decode_dc_zero_size() {
        // Luma DC size 1 has code `00` (len 2); size 0 has code `100`.
        // Use size 0 (code 100) → diff 0 → DC == predictor.
        let mut bits = Vec::new();
        push_bits(&mut bits, 0b100, 3); // dct_dc_size_luminance = 0
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        let mut preds = DcPredictors::reset(0);
        let dc = decode_dc(&mut r, &mut preds, BlockComponent::Luma).expect("dc");
        assert_eq!(dc, 128);
        assert_eq!(preds.y, 128);
    }

    #[test]
    fn decode_dc_positive_diff() {
        // Luma size 2 (code `01`, len 2), then 2 differential bits = 0b11 = 3.
        // Leading bit 1 → positive → diff = 3. DC = 128 + 3 = 131.
        let mut bits = Vec::new();
        push_bits(&mut bits, 0b01, 2); // size 2
        push_bits(&mut bits, 0b11, 2); // differential 3
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        let mut preds = DcPredictors::reset(0);
        let dc = decode_dc(&mut r, &mut preds, BlockComponent::Luma).expect("dc");
        assert_eq!(dc, 131);
    }

    #[test]
    fn decode_dc_negative_diff() {
        // Luma size 2 (code `01`), differential bits = 0b00.
        // Leading bit 0 → negative → diff = 0 - (2^2 - 1) = -3. DC = 125.
        let mut bits = Vec::new();
        push_bits(&mut bits, 0b01, 2);
        push_bits(&mut bits, 0b00, 2);
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        let mut preds = DcPredictors::reset(0);
        let dc = decode_dc(&mut r, &mut preds, BlockComponent::Luma).expect("dc");
        assert_eq!(dc, 125);
    }

    #[test]
    fn decode_dc_chroma_uses_b13() {
        // Chroma size 0 has code `00` (len 2) → diff 0 → DC = predictor.
        let mut bits = Vec::new();
        push_bits(&mut bits, 0b00, 2);
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        let mut preds = DcPredictors::reset(0);
        let dc = decode_dc(&mut r, &mut preds, BlockComponent::Cb).expect("dc");
        assert_eq!(dc, 128);
    }

    #[test]
    fn decode_ac_immediate_eob() {
        // B-14 EOB is `10`. No AC coefficients → all zero.
        let mut bits = Vec::new();
        push_bits(&mut bits, 0b10, 2);
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        let mut block = [0i32; 64];
        block[0] = 100;
        decode_ac(&mut r, &mut block, false, false).expect("ac");
        assert_eq!(block[0], 100);
        assert!(block[1..].iter().all(|&v| v == 0));
    }

    #[test]
    fn decode_ac_one_run_level_then_eob() {
        // B-14: (run=0, level=1) code `11`, then sign bit 0 (positive),
        // then EOB `10`.
        let mut bits = Vec::new();
        push_bits(&mut bits, 0b11, 2); // run 0, level 1
        push_bits(&mut bits, 0, 1); // sign + (positive)
        push_bits(&mut bits, 0b10, 2); // EOB
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        let mut block = [0i32; 64];
        decode_ac(&mut r, &mut block, false, false).expect("ac");
        // Progressive scan index 1 → raster position 1.
        assert_eq!(block[1], 1);
        assert!(block[2..].iter().all(|&v| v == 0));
    }

    #[test]
    fn decode_ac_negative_level() {
        // (run=0, level=1) `11`, sign bit 1 (negative), EOB.
        let mut bits = Vec::new();
        push_bits(&mut bits, 0b11, 2);
        push_bits(&mut bits, 1, 1); // negative
        push_bits(&mut bits, 0b10, 2);
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        let mut block = [0i32; 64];
        decode_ac(&mut r, &mut block, false, false).expect("ac");
        assert_eq!(block[1], -1);
    }

    #[test]
    fn decode_ac_run_skips_positions() {
        // (run=1, level=1) code `011`, sign 0, EOB.
        // Scan index advances by run (1) → index 2 → raster pos (progressive) 8.
        let mut bits = Vec::new();
        push_bits(&mut bits, 0b011, 3);
        push_bits(&mut bits, 0, 1);
        push_bits(&mut bits, 0b10, 2);
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        let mut block = [0i32; 64];
        decode_ac(&mut r, &mut block, false, false).expect("ac");
        // run=1 → scan index 1+1 = 2 → SCAN_PROGRESSIVE[2] = 8.
        assert_eq!(block[8], 1);
        assert_eq!(block[1], 0);
    }

    #[test]
    fn decode_ac_escape_sequence() {
        // Escape `0000 01`, run 0 (6 bits), level 100 (12-bit signed), then EOB.
        let mut bits = Vec::new();
        push_bits(&mut bits, 0b000001, 6); // escape
        push_bits(&mut bits, 0, 6); // run = 0
        push_bits(&mut bits, 100, 12); // level = 100
        push_bits(&mut bits, 0b10, 2); // EOB
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        let mut block = [0i32; 64];
        decode_ac(&mut r, &mut block, false, false).expect("ac");
        assert_eq!(block[1], 100);
    }

    #[test]
    fn decode_ac_escape_negative_level() {
        // Escape, run 0, level = -50 in 12-bit two's complement (0x1000 - 50).
        let neg = (0x1000 - 50) as u32;
        let mut bits = Vec::new();
        push_bits(&mut bits, 0b000001, 6);
        push_bits(&mut bits, 0, 6);
        push_bits(&mut bits, neg, 12);
        push_bits(&mut bits, 0b10, 2);
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        let mut block = [0i32; 64];
        decode_ac(&mut r, &mut block, false, false).expect("ac");
        assert_eq!(block[1], -50);
    }

    #[test]
    fn decode_intra_block_dc_only() {
        // Luma DC size 0 (code 100) → DC = predictor; AC EOB immediately.
        let mut bits = Vec::new();
        push_bits(&mut bits, 0b100, 3); // dc size 0
        push_bits(&mut bits, 0b10, 2); // AC EOB
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        let mut preds = DcPredictors::reset(0);
        let block = decode_intra_block(&mut r, &mut preds, BlockComponent::Luma, false, false)
            .expect("block");
        assert_eq!(block[0], 128);
        assert!(block[1..].iter().all(|&v| v == 0));
    }

    #[test]
    fn decode_ac_uses_b15_when_intra_vlc_set() {
        // B-15 EOB is `0110`. With intra_vlc_format=true, that should be EOB.
        let mut bits = Vec::new();
        push_bits(&mut bits, 0b0110, 4);
        let bytes = bits_to_bytes(&bits);
        let mut r = BitReader::new(&bytes);
        let mut block = [0i32; 64];
        decode_ac(&mut r, &mut block, true, false).expect("ac");
        assert!(block[1..].iter().all(|&v| v == 0));
    }
}
