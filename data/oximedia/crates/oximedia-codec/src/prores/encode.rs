//! Slice-level encode pipeline for ProRes 422.
//!
//! For each slice:
//! 1. Extract 8×8 blocks from luma and chroma planes.
//! 2. Center input samples (subtract 512 for Y, Cb, Cr each).
//! 3. Apply fdct_8x8 to each block.
//! 4. Quantize with the appropriate matrix and qscale.
//! 5. Forward zigzag scan to get coefficients in scan order.
//! 6. Entropy-encode each plane's block stream.
//! 7. Write the slice header (header_size, qscale, per-plane sizes).
//! 8. Concatenate header + luma payload + Cb payload + Cr payload.
//!
//! The output bytes are exactly what [`super::decode::decode_slice_to_yuv422`]
//! can consume.

use super::bitwriter::BitWriter;
use super::entropy_encode::encode_block;
use super::fdct::fdct_8x8;
use super::quantize::quantize_block;
use super::zigzag::PROGRESSIVE_ZIGZAG;

/// Encode a single ProRes 422 slice to bytes.
///
/// The slice covers `mb_width` macroblocks (each 16×16 luma, 8×16 chroma).
/// For 4:2:2 the luma plane is `mb_width * 16` samples wide and 16 rows tall;
/// each chroma plane is `mb_width * 8` samples wide and 16 rows tall.
///
/// `luma` / `cb` / `cr` are 10-bit samples stored as `u16`, in raster order.
/// `luma_stride` is the row stride in *samples* for the luma plane;
/// `chroma_stride` is the row stride in *samples* for each chroma plane.
///
/// `qscale` is the quantization parameter for this slice (1..=224).
/// `luma_matrix` and `chroma_matrix` are the 64-element quantization matrices.
///
/// Returns the complete slice bytes (header + Y payload + Cb payload + Cr payload).
pub fn encode_slice(
    luma: &[u16],
    cb: &[u16],
    cr: &[u16],
    luma_stride: usize,
    chroma_stride: usize,
    mb_width: usize,
    qscale: u8,
    luma_matrix: &[u8; 64],
    chroma_matrix: &[u8; 64],
) -> Vec<u8> {
    let luma_bits = encode_plane(
        luma,
        luma_stride,
        mb_width,
        PlaneKind::Luma,
        qscale,
        luma_matrix,
    );
    let cb_bits = encode_plane(
        cb,
        chroma_stride,
        mb_width,
        PlaneKind::Chroma,
        qscale,
        chroma_matrix,
    );
    let cr_bits = encode_plane(
        cr,
        chroma_stride,
        mb_width,
        PlaneKind::Chroma,
        qscale,
        chroma_matrix,
    );

    let luma_size = luma_bits.len() as u16;
    let cb_size = cb_bits.len() as u16;
    let cr_size = cr_bits.len() as u16;

    // Build slice header (8 bytes, no alpha).
    // byte 0: header_size (8) in high nibble, low nibble reserved/0.
    // byte 1: qscale
    // bytes 2-3: luma_data_size BE
    // bytes 4-5: cb_data_size BE
    // bytes 6-7: cr_data_size BE
    let mut out = Vec::with_capacity(8 + luma_bits.len() + cb_bits.len() + cr_bits.len());
    out.push(8u8 << 4); // header_size = 8, stored in high nibble
    out.push(qscale);
    out.extend_from_slice(&luma_size.to_be_bytes());
    out.extend_from_slice(&cb_size.to_be_bytes());
    out.extend_from_slice(&cr_size.to_be_bytes());
    out.extend_from_slice(&luma_bits);
    out.extend_from_slice(&cb_bits);
    out.extend_from_slice(&cr_bits);
    out
}

/// Encode a single ProRes 4:4:4 slice to bytes.
///
/// The slice covers `mb_width` macroblocks (each 16×16 luma *and* 16×16 per
/// chroma). For 4:4:4 (`'ap4h'` / `'ap4x'`) every plane — luma, Cb, Cr — is
/// full-resolution and contributes 4 blocks per MB in the identical 2×2
/// raster layout. The per-slice chroma byte budget is therefore roughly
/// double that of a 4:2:2 slice.
///
/// `luma` / `cb` / `cr` are 10-bit samples stored as `u16`, in raster order.
/// `luma_stride` / `chroma_stride` are the row strides in *samples*; for
/// 4:4:4 they are equal (`mb_width * 16`).
///
/// Returns the complete slice bytes (8-byte header + Y + Cb + Cr payloads).
/// The DCT-coded alpha plane is not emitted.
#[allow(clippy::too_many_arguments)]
pub fn encode_slice_444(
    luma: &[u16],
    cb: &[u16],
    cr: &[u16],
    luma_stride: usize,
    chroma_stride: usize,
    mb_width: usize,
    qscale: u8,
    luma_matrix: &[u8; 64],
    chroma_matrix: &[u8; 64],
) -> Vec<u8> {
    let luma_bits = encode_plane(
        luma,
        luma_stride,
        mb_width,
        PlaneKind::Luma,
        qscale,
        luma_matrix,
    );
    // 4:4:4 chroma uses the full-resolution luma block layout.
    let cb_bits = encode_plane(
        cb,
        chroma_stride,
        mb_width,
        PlaneKind::Chroma444,
        qscale,
        chroma_matrix,
    );
    let cr_bits = encode_plane(
        cr,
        chroma_stride,
        mb_width,
        PlaneKind::Chroma444,
        qscale,
        chroma_matrix,
    );

    let luma_size = luma_bits.len() as u16;
    let cb_size = cb_bits.len() as u16;
    let cr_size = cr_bits.len() as u16;

    let mut out = Vec::with_capacity(8 + luma_bits.len() + cb_bits.len() + cr_bits.len());
    out.push(8u8 << 4); // header_size = 8, stored in high nibble
    out.push(qscale);
    out.extend_from_slice(&luma_size.to_be_bytes());
    out.extend_from_slice(&cb_size.to_be_bytes());
    out.extend_from_slice(&cr_size.to_be_bytes());
    out.extend_from_slice(&luma_bits);
    out.extend_from_slice(&cb_bits);
    out.extend_from_slice(&cr_bits);
    out
}

#[derive(Clone, Copy)]
enum PlaneKind {
    Luma,
    Chroma,
    /// 4:4:4 chroma: full-resolution, 4 blocks per MB in the same 2×2 raster
    /// layout as luma.
    Chroma444,
}

impl PlaneKind {
    fn blocks_per_mb(self) -> usize {
        match self {
            Self::Luma | Self::Chroma444 => 4,
            Self::Chroma => 2,
        }
    }

    /// Column offset in samples for block `b` (0..) within macroblock `mb_x`.
    fn col_offset(self, mb_x: usize, block_in_mb: usize) -> usize {
        match self {
            Self::Luma | Self::Chroma444 => mb_x * 16 + (block_in_mb & 1) * 8,
            Self::Chroma => mb_x * 8,
        }
    }

    /// Row offset for block `b` (0..) within macroblock.
    fn row_offset(self, block_in_mb: usize) -> usize {
        match self {
            Self::Luma | Self::Chroma444 => (block_in_mb / 2) * 8,
            Self::Chroma => block_in_mb * 8,
        }
    }
}

/// Encode all blocks for one plane of one slice, returning the compressed bytes.
fn encode_plane(
    plane: &[u16],
    stride: usize,
    mb_width: usize,
    kind: PlaneKind,
    qscale: u8,
    matrix: &[u8; 64],
) -> Vec<u8> {
    let mut writer = BitWriter::new();
    let mut running_dc: i16 = 0;
    let blocks_per_mb = kind.blocks_per_mb();

    for mb_x in 0..mb_width {
        for b in 0..blocks_per_mb {
            let col = kind.col_offset(mb_x, b);
            let row = kind.row_offset(b);

            // Extract 8×8 block of samples, converting to signed i32.
            // ProRes uses 10-bit offset binary; center around 0 by subtracting 512.
            let mut block = [0i32; 64];
            for r in 0..8 {
                for c in 0..8 {
                    let idx = (row + r) * stride + (col + c);
                    let sample = if idx < plane.len() { plane[idx] } else { 512 };
                    block[r * 8 + c] = i32::from(sample) - 512;
                }
            }

            // Forward DCT.
            let freq = fdct_8x8(&block);

            // Quantize (raster order).
            let quantized_raster = quantize_block(&freq, matrix, qscale);

            // Forward zigzag scan: raster → scan order.
            let mut quantized_scan = [0i16; 64];
            for (scan_idx, &raster_idx) in PROGRESSIVE_ZIGZAG.iter().enumerate() {
                quantized_scan[scan_idx] = quantized_raster[raster_idx as usize];
            }

            // Entropy-encode this block. The encoder returns the new running DC
            // (absolute value), but we track it as i16 for the differential coder.
            running_dc = encode_block(&mut writer, &quantized_scan, running_dc);
        }
    }

    writer.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prores::decode::{
        decode_slice_to_yuv422, decode_slice_to_yuv444, split_slice_planes,
    };
    use crate::prores::picture::parse_slice_header;
    use crate::prores::quant::{DEFAULT_CHROMA_QUANT_MATRIX, DEFAULT_LUMA_QUANT_MATRIX};

    fn flat_plane(val: u16, width: usize, height: usize) -> Vec<u16> {
        vec![val; width * height]
    }

    #[test]
    fn encode_slice_header_format_is_correct() {
        let mb_width = 8usize;
        let luma_w = mb_width * 16;
        let chroma_w = mb_width * 8;
        let height = 16;
        let luma = flat_plane(512, luma_w, height);
        let cb = flat_plane(512, chroma_w, height);
        let cr = flat_plane(512, chroma_w, height);

        let slice_bytes = encode_slice(
            &luma,
            &cb,
            &cr,
            luma_w,
            chroma_w,
            mb_width,
            6,
            &DEFAULT_LUMA_QUANT_MATRIX,
            &DEFAULT_CHROMA_QUANT_MATRIX,
        );

        // Parse the slice header to verify the format is correct.
        let (hdr, payload) = parse_slice_header(&slice_bytes, false).expect("parse_slice_header");
        assert_eq!(hdr.header_size, 8, "header_size should be 8");
        assert_eq!(hdr.quant_scale, 6, "quant_scale should match");
        // Check that declared plane sizes add up to the payload length.
        assert_eq!(
            hdr.data_size(),
            payload.len(),
            "sum of plane sizes should match payload"
        );
    }

    #[test]
    fn encode_decode_slice_flat_color_roundtrip() {
        let mb_width = 2usize;
        let luma_w = mb_width * 16;
        let chroma_w = mb_width * 8;
        let height = 16;

        // Flat color: Y=700, Cb=512, Cr=300.
        let luma = flat_plane(700, luma_w, height);
        let cb = flat_plane(512, chroma_w, height);
        let cr = flat_plane(300, chroma_w, height);

        let slice_bytes = encode_slice(
            &luma,
            &cb,
            &cr,
            luma_w,
            chroma_w,
            mb_width,
            6,
            &DEFAULT_LUMA_QUANT_MATRIX,
            &DEFAULT_CHROMA_QUANT_MATRIX,
        );

        let (hdr, payload) = parse_slice_header(&slice_bytes, false).expect("parse_slice_header");
        let sd = split_slice_planes(
            payload,
            hdr.luma_data_size,
            hdr.cb_data_size,
            hdr.cr_data_size,
            None,
        )
        .expect("split");

        let mut dst_luma = vec![0u16; luma_w * height];
        let mut dst_cb = vec![0u16; chroma_w * height];
        let mut dst_cr = vec![0u16; chroma_w * height];

        decode_slice_to_yuv422(
            sd,
            &DEFAULT_LUMA_QUANT_MATRIX,
            &DEFAULT_CHROMA_QUANT_MATRIX,
            hdr.quant_scale,
            mb_width,
            &mut dst_luma,
            luma_w,
            &mut dst_cb,
            chroma_w,
            &mut dst_cr,
            chroma_w,
        )
        .expect("decode");

        // Each output pixel should be close to the input value.
        for &v in &dst_luma {
            assert!(
                (v as i32 - 700).abs() <= 16,
                "luma error too large: got {v}"
            );
        }
        for &v in &dst_cb {
            assert!((v as i32 - 512).abs() <= 16, "Cb error too large: got {v}");
        }
        for &v in &dst_cr {
            assert!((v as i32 - 300).abs() <= 16, "Cr error too large: got {v}");
        }
    }

    #[test]
    fn encode_decode_slice_444_flat_color_roundtrip() {
        // For 4:4:4 the chroma planes are full-resolution (same width as luma).
        let mb_width = 2usize;
        let plane_w = mb_width * 16; // luma == chroma width in 4:4:4
        let height = 16;

        // Flat color: Y=620, Cb=480, Cr=350.
        let luma = flat_plane(620, plane_w, height);
        let cb = flat_plane(480, plane_w, height);
        let cr = flat_plane(350, plane_w, height);

        let slice_bytes = encode_slice_444(
            &luma,
            &cb,
            &cr,
            plane_w,
            plane_w,
            mb_width,
            6,
            &DEFAULT_LUMA_QUANT_MATRIX,
            &DEFAULT_CHROMA_QUANT_MATRIX,
        );

        let (hdr, payload) = parse_slice_header(&slice_bytes, false).expect("parse_slice_header");
        let sd = split_slice_planes(
            payload,
            hdr.luma_data_size,
            hdr.cb_data_size,
            hdr.cr_data_size,
            None,
        )
        .expect("split");

        let mut dst_luma = vec![0u16; plane_w * height];
        let mut dst_cb = vec![0u16; plane_w * height];
        let mut dst_cr = vec![0u16; plane_w * height];

        decode_slice_to_yuv444(
            sd,
            &DEFAULT_LUMA_QUANT_MATRIX,
            &DEFAULT_CHROMA_QUANT_MATRIX,
            hdr.quant_scale,
            mb_width,
            &mut dst_luma,
            plane_w,
            &mut dst_cb,
            plane_w,
            &mut dst_cr,
            plane_w,
        )
        .expect("decode 4:4:4");

        for &v in &dst_luma {
            assert!(
                (v as i32 - 620).abs() <= 16,
                "luma error too large: got {v}"
            );
        }
        for &v in &dst_cb {
            assert!((v as i32 - 480).abs() <= 16, "Cb error too large: got {v}");
        }
        for &v in &dst_cr {
            assert!((v as i32 - 350).abs() <= 16, "Cr error too large: got {v}");
        }
    }

    #[test]
    fn encode_slice_444_chroma_payload_larger_than_422() {
        // The same MB width with a non-flat chroma signal: 4:4:4 carries
        // double the chroma blocks, so its chroma payload must exceed 4:2:2's.
        let mb_width = 4usize;
        let luma_w = mb_width * 16;
        let chroma_w_422 = mb_width * 8;
        let chroma_w_444 = mb_width * 16;
        let height = 16;

        // Build a ramp chroma signal (non-flat → real AC coefficients).
        let ramp = |w: usize, h: usize| -> Vec<u16> {
            (0..w * h)
                .map(|i| 200u16 + ((i as u16 * 7) % 600))
                .collect()
        };
        let luma = ramp(luma_w, height);

        let s422 = encode_slice(
            &luma,
            &ramp(chroma_w_422, height),
            &ramp(chroma_w_422, height),
            luma_w,
            chroma_w_422,
            mb_width,
            6,
            &DEFAULT_LUMA_QUANT_MATRIX,
            &DEFAULT_CHROMA_QUANT_MATRIX,
        );
        let s444 = encode_slice_444(
            &luma,
            &ramp(chroma_w_444, height),
            &ramp(chroma_w_444, height),
            luma_w,
            chroma_w_444,
            mb_width,
            6,
            &DEFAULT_LUMA_QUANT_MATRIX,
            &DEFAULT_CHROMA_QUANT_MATRIX,
        );

        let (h422, _) = parse_slice_header(&s422, false).expect("422 header");
        let (h444, _) = parse_slice_header(&s444, false).expect("444 header");
        // Luma is identical between the two; chroma must be strictly larger.
        assert!(
            h444.cb_data_size > h422.cb_data_size,
            "4:4:4 Cb ({}) should exceed 4:2:2 Cb ({})",
            h444.cb_data_size,
            h422.cb_data_size
        );
        assert!(h444.cr_data_size > h422.cr_data_size);
    }
}
