//! DNxHD frame decoder — top-level pipeline.
//!
//! Implements `DnxhdDecoder::decode()` which processes a complete VC-3 frame
//! buffer and produces a `DecodedFrame` with planar YUV 4:2:2 output.
//!
//! # Pipeline
//!
//! ```text
//! raw bytes
//!    │
//!    ▼  parse_frame_header        (26 bytes, fixed)
//! FrameHeader + slice size table offset
//!    │
//!    ▼  read slice size table     (4 bytes × num_slices)
//! per-slice byte offsets
//!    │
//!    ▼  for each slice:
//!       read per-plane byte counts from slice header
//!       for each macroblock (4 luma + 2 chroma 8×8 blocks):
//!         decode DC (DPCM) + decode AC (MPEG-2 VLC)
//!         dequantize → inverse zigzag → IDCT → finalize
//!         blit 8×8 block into output plane
//! ```
//!
//! # Output
//!
//! - `PixelFormat::Yuv422p` for 8-bit profiles (CID 1237, 1238, 1242, 1243).
//! - `PixelFormat::Yuv422p10le` for 10-bit profiles (CID 1235, 1241).
//! Planes are ordered Y, Cb, Cr in planar layout.

use oximedia_core::PixelFormat;

use super::bitreader::BitReader;
use super::entropy::{
    dc_table_entries_8bit, decode_ac_coefficients, decode_dc_sequential, dequantize_block,
    QUANT_MATRIX_CHROMA_8BIT, QUANT_MATRIX_LUMA_8BIT,
};
use super::frame_header::{parse_frame_header, DnxhdProfile, FrameHeader};
use super::idct::{finalize_10bit, finalize_8bit, idct_8x8};
use super::vlc_tables::build_ac_table;
use super::zigzag::inverse_zigzag;
use super::DecodeError;

/// A decoded DNxHD frame, containing planar YUV data.
#[derive(Debug)]
pub struct DecodedFrame {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Profile that was decoded.
    pub profile: DnxhdProfile,
    /// Planar YUV data: Y plane (w×h), Cb plane (w/2×h), Cr plane (w/2×h).
    /// For 8-bit output: each sample is 1 byte.
    /// For 10-bit output: each sample is 2 bytes (little-endian u16, low 10 bits).
    pub yuv_data: Vec<u8>,
    /// Pixel format of the output.
    pub pixel_format: PixelFormat,
}

/// DNxHD (VC-3 / SMPTE ST 2019-1) decoder.
///
/// Only decodes 4:2:2 progressive frames. Supports 8-bit (DNxHD 145, 220,
/// 100, 60) and 10-bit (DNxHD 145x, 220x) profiles.
pub struct DnxhdDecoder;

impl DnxhdDecoder {
    /// Decode a complete DNxHD frame from `data`.
    ///
    /// # Errors
    ///
    /// Returns `DecodeError` if the frame is malformed, the CID is unknown,
    /// or the profile is not supported (4:4:4 or interlaced).
    pub fn decode(data: &[u8]) -> Result<DecodedFrame, DecodeError> {
        // ── 1. Parse frame header ────────────────────────────────────────────
        let (header, hdr_len) = parse_frame_header(data)?;

        // Reject 4:4:4 and unknown profiles for now.
        if header.chroma_format == 0x48 {
            return Err(DecodeError::UnsupportedProfile(header.profile));
        }
        if matches!(header.profile, DnxhdProfile::Unknown(_)) {
            return Err(DecodeError::UnknownCid(header.cid));
        }

        let width = header.width as usize;
        let height = header.height as usize;
        let num_slices = header.num_slices as usize;
        let is_10bit = header.bits_per_pixel == 10;

        // Defend against an allocation bomb: DNxHD width/height are 16-bit
        // header fields that drive the Y/Cb/Cr output plane allocations below;
        // reject impossibly large frames against the shared dimension ceiling.
        crate::limits::check_dimensions(width, height).map_err(DecodeError::InvalidData)?;

        // ── 2. Read the slice size table ─────────────────────────────────────
        // Each entry is a 4-byte big-endian u32 giving the byte length of the
        // corresponding slice (including the slice header).
        let slice_table_offset = hdr_len;
        let slice_table_bytes = num_slices * 4;
        let slice_data_start = slice_table_offset + slice_table_bytes;

        if data.len() < slice_data_start {
            return Err(DecodeError::BufferTooSmall {
                need: slice_data_start,
                have: data.len(),
            });
        }

        let mut slice_sizes = Vec::with_capacity(num_slices);
        for i in 0..num_slices {
            let off = slice_table_offset + i * 4;
            let sz = u32::from_be_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            slice_sizes.push(sz as usize);
        }

        // ── 3. Allocate output planes ────────────────────────────────────────
        // Planar YUV 4:2:2: Y = w×h, Cb = w/2×h, Cr = w/2×h.
        let chroma_w = width / 2;
        let bytes_per_sample: usize = if is_10bit { 2 } else { 1 };

        let y_size = width * height * bytes_per_sample;
        let cb_size = chroma_w * height * bytes_per_sample;
        let cr_size = chroma_w * height * bytes_per_sample;

        let mut yuv_data = vec![0u8; y_size + cb_size + cr_size];
        let (y_plane, rest) = yuv_data.split_at_mut(y_size);
        let (cb_plane, cr_plane) = rest.split_at_mut(cb_size);

        // Build shared decode tables.
        let dc_entries = dc_table_entries_8bit();
        let ac_table = build_ac_table();

        // DC DPCM state per component (Y, Cb, Cr).
        let mut dc_y: i16 = 0;
        let mut dc_cb: i16 = 0;
        let mut dc_cr: i16 = 0;

        // Number of macroblock rows per slice. DNxHD typically uses 8 or 16 MB rows
        // per slice. The slice header tells us the actual size.
        let mb_height = 16usize; // 1 macroblock = 16 luma lines
        let mb_rows_per_slice = {
            let total_mb_rows = height.div_ceil(mb_height);
            total_mb_rows.div_ceil(num_slices.max(1))
        };

        // ── 4. Decode each slice ─────────────────────────────────────────────
        let mut slice_data_offset = slice_data_start;

        for slice_idx in 0..num_slices {
            let slice_len = slice_sizes[slice_idx];
            if slice_data_offset + slice_len > data.len() {
                return Err(DecodeError::BufferTooSmall {
                    need: slice_data_offset + slice_len,
                    have: data.len(),
                });
            }
            let slice_bytes = &data[slice_data_offset..slice_data_offset + slice_len];
            slice_data_offset += slice_len;

            // Slice header: per-plane byte counts (4 bytes each for Y, Cb, Cr,
            // then the compressed data follows).
            if slice_bytes.len() < 12 {
                continue; // Empty or degenerate slice — skip.
            }
            let y_bytes = u32::from_be_bytes([
                slice_bytes[0],
                slice_bytes[1],
                slice_bytes[2],
                slice_bytes[3],
            ]) as usize;
            let cb_bytes = u32::from_be_bytes([
                slice_bytes[4],
                slice_bytes[5],
                slice_bytes[6],
                slice_bytes[7],
            ]) as usize;
            let cr_bytes = u32::from_be_bytes([
                slice_bytes[8],
                slice_bytes[9],
                slice_bytes[10],
                slice_bytes[11],
            ]) as usize;
            let payload_off = 12usize;

            if payload_off + y_bytes + cb_bytes + cr_bytes > slice_bytes.len() {
                continue; // Slice payload too short; skip.
            }

            let y_data = &slice_bytes[payload_off..payload_off + y_bytes];
            let cb_data = &slice_bytes[payload_off + y_bytes..payload_off + y_bytes + cb_bytes];
            let cr_data = &slice_bytes
                [payload_off + y_bytes + cb_bytes..payload_off + y_bytes + cb_bytes + cr_bytes];

            // Macroblock row range for this slice.
            let mb_row_start = slice_idx * mb_rows_per_slice;
            let mb_row_end = ((slice_idx + 1) * mb_rows_per_slice).min(height / mb_height);

            let mb_cols = width / 16;

            // ── Decode luma (Y) ──────────────────────────────────────────────
            {
                let mut y_reader = BitReader::new(y_data);
                for mb_row in mb_row_start..mb_row_end {
                    for mb_col in 0..mb_cols {
                        // 4 luma 8×8 blocks per macroblock (2×2 in 16×16 MB).
                        for block_row in 0..2usize {
                            for block_col in 0..2usize {
                                dc_y = decode_dc_sequential(&mut y_reader, &dc_entries, dc_y)?;
                                let mut ac_coeffs =
                                    decode_ac_coefficients(&mut y_reader, &ac_table)?;
                                ac_coeffs[0] = dc_y;

                                let dequant =
                                    dequantize_block(&ac_coeffs, &QUANT_MATRIX_LUMA_8BIT, 1);
                                let raster = inverse_zigzag(&dequant);
                                let spatial = idct_8x8(&raster);

                                // Blit into Y plane.
                                let top_y = (mb_row * 16 + block_row * 8) * width;
                                let left_x = mb_col * 16 + block_col * 8;
                                blit_8x8_block(
                                    &spatial,
                                    y_plane,
                                    top_y + left_x,
                                    width,
                                    is_10bit,
                                    bytes_per_sample,
                                );
                            }
                        }
                    }
                }
            }

            // ── Decode Cb chroma ─────────────────────────────────────────────
            {
                let mut cb_reader = BitReader::new(cb_data);
                for mb_row in mb_row_start..mb_row_end {
                    for mb_col in 0..mb_cols {
                        // 1 chroma 8×8 block per macroblock in 4:2:2.
                        dc_cb = decode_dc_sequential(&mut cb_reader, &dc_entries, dc_cb)?;
                        let mut ac_coeffs = decode_ac_coefficients(&mut cb_reader, &ac_table)?;
                        ac_coeffs[0] = dc_cb;

                        let dequant = dequantize_block(&ac_coeffs, &QUANT_MATRIX_CHROMA_8BIT, 1);
                        let raster = inverse_zigzag(&dequant);
                        let spatial = idct_8x8(&raster);

                        // Blit into Cb plane (chroma at half horizontal width).
                        let top_y = mb_row * 16 * chroma_w;
                        let left_x = mb_col * 8;
                        blit_8x8_block(
                            &spatial,
                            cb_plane,
                            top_y + left_x,
                            chroma_w,
                            is_10bit,
                            bytes_per_sample,
                        );
                    }
                }
            }

            // ── Decode Cr chroma ─────────────────────────────────────────────
            {
                let mut cr_reader = BitReader::new(cr_data);
                for mb_row in mb_row_start..mb_row_end {
                    for mb_col in 0..mb_cols {
                        dc_cr = decode_dc_sequential(&mut cr_reader, &dc_entries, dc_cr)?;
                        let mut ac_coeffs = decode_ac_coefficients(&mut cr_reader, &ac_table)?;
                        ac_coeffs[0] = dc_cr;

                        let dequant = dequantize_block(&ac_coeffs, &QUANT_MATRIX_CHROMA_8BIT, 1);
                        let raster = inverse_zigzag(&dequant);
                        let spatial = idct_8x8(&raster);

                        let top_y = mb_row * 16 * chroma_w;
                        let left_x = mb_col * 8;
                        blit_8x8_block(
                            &spatial,
                            cr_plane,
                            top_y + left_x,
                            chroma_w,
                            is_10bit,
                            bytes_per_sample,
                        );
                    }
                }
            }
        }

        let pixel_format = if is_10bit {
            PixelFormat::Yuv422p10le
        } else {
            PixelFormat::Yuv422p
        };

        Ok(DecodedFrame {
            width: width as u32,
            height: height as u32,
            profile: header.profile,
            yuv_data,
            pixel_format,
        })
    }
}

/// Blit an 8×8 IDCT output block into a plane buffer.
///
/// `origin` is the linear index of the top-left sample of this block in the
/// plane. `plane_stride` is the number of samples per row.
fn blit_8x8_block(
    spatial: &[i32; 64],
    plane: &mut [u8],
    origin: usize,
    plane_stride: usize,
    is_10bit: bool,
    bytes_per_sample: usize,
) {
    for row in 0..8 {
        for col in 0..8 {
            let sample_idx = origin + row * plane_stride + col;
            let coeff = spatial[row * 8 + col];

            if is_10bit {
                // 10-bit: store as u16 little-endian. dc_offset = 0 (centred by IDCT).
                let pix = finalize_10bit(coeff, 0);
                let byte_idx = sample_idx * bytes_per_sample;
                if byte_idx + 1 < plane.len() {
                    let le = pix.to_le_bytes();
                    plane[byte_idx] = le[0];
                    plane[byte_idx + 1] = le[1];
                }
            } else {
                // 8-bit.
                let pix = finalize_8bit(coeff, 0);
                if sample_idx < plane.len() {
                    plane[sample_idx] = pix;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dnxhd::vlc_tables::build_dc_table_8bit;

    // ── VLC lookup test ──────────────────────────────────────────────────────

    #[test]
    fn vlc_dc_table_lookup_all_sizes() {
        use crate::dnxhd::vlc_tables::DC_TABLE_8BIT;
        let table = build_dc_table_8bit();
        for (size, entry) in DC_TABLE_8BIT.iter().enumerate() {
            let code_msb: u32 = (entry.code as u32) << 16;
            let result = table.lookup(code_msb);
            assert!(result.is_some(), "size {size}: lookup failed");
            let (val, len) = result.unwrap();
            assert_eq!(val as usize, size, "size {size}: wrong value");
            assert_eq!(len, entry.len, "size {size}: wrong len");
        }
    }

    // ── IDCT DC-only test ────────────────────────────────────────────────────

    #[test]
    fn idct_dc_only_uniform() {
        use crate::dnxhd::idct::idct_8x8;
        // DC = 128*8: after IDCT both passes, spatial ≈ 128 for all 64 samples.
        let mut coeffs = [0i32; 64];
        coeffs[0] = 128 * 8;
        let out = idct_8x8(&coeffs);
        let first = out[0];
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - first).abs() <= 2,
                "idct_dc_only: sample[{i}]={v} != first={first}"
            );
        }
    }

    // ── Zigzag round-trip test ───────────────────────────────────────────────

    #[test]
    fn zigzag_round_trip() {
        use crate::dnxhd::zigzag::{inverse_zigzag, ZIGZAG_SCAN};
        // Fill a known raster array: value[i] = i.
        let raster: [i32; 64] = std::array::from_fn(|i| i as i32);
        // Forward zigzag: scan_buf[scan_idx] = raster[ZIGZAG_SCAN[scan_idx]].
        let mut scan_buf = [0i32; 64];
        for (scan_idx, &raster_idx) in ZIGZAG_SCAN.iter().enumerate() {
            scan_buf[scan_idx] = raster[raster_idx];
        }
        // Inverse back to raster.
        let recovered = inverse_zigzag(&scan_buf);
        assert_eq!(recovered, raster, "zigzag round-trip failed");
    }

    // ── Frame header parse test ──────────────────────────────────────────────

    fn make_test_header_inline(cid: u32, width: u16, height: u16, bpp_marker: u16) -> Vec<u8> {
        use crate::dnxhd::frame_header::FRAME_MAGIC;
        const FRAME_MARKER: [u8; 4] = [0x00u8, 0x00, 0x00, 0x01];
        let mut h = vec![0u8; 40];
        h[0..4].copy_from_slice(&FRAME_MAGIC);
        h[4..8].copy_from_slice(&FRAME_MARKER);
        h[8..12].copy_from_slice(&cid.to_be_bytes());
        h[12..14].copy_from_slice(&width.to_be_bytes());
        h[14..16].copy_from_slice(&height.to_be_bytes());
        h[16..18].copy_from_slice(&height.to_be_bytes());
        h[19] = 0x58;
        h[20..22].copy_from_slice(&bpp_marker.to_be_bytes());
        let ns = (height / 16).max(1);
        h[22..24].copy_from_slice(&ns.to_be_bytes());
        let mbw = (width / 16).max(1);
        h[24..26].copy_from_slice(&mbw.to_be_bytes());
        h
    }

    #[test]
    fn frame_header_parse_dnxhd145() {
        let data = make_test_header_inline(1237, 1440, 1080, 0x5814);
        let (hdr, consumed) = parse_frame_header(&data).unwrap();
        assert_eq!(hdr.profile, DnxhdProfile::Dnxhd145);
        assert_eq!(hdr.bits_per_pixel, 8);
        assert_eq!(consumed, 26);
    }

    // ── Minimal frame decode test (bad magic rejection) ──────────────────────

    #[test]
    fn decode_rejects_bad_magic() {
        let mut data = make_test_header_inline(1238, 16, 16, 0x5814);
        data[0] = 0xFF;
        let result = DnxhdDecoder::decode(&data);
        assert!(
            matches!(result, Err(crate::dnxhd::DecodeError::InvalidMagic)),
            "expected InvalidMagic"
        );
    }
}
