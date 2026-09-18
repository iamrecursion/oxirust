//! Top-level MPEG-2 I-frame decode pipeline (ISO/IEC 13818-2).
//!
//! ```text
//! elementary stream bytes
//!    │
//!    ▼  find sequence_header_code (0xB3)  → parse_sequence_header
//!    ▼  find extension_start_code (0xB5)  → parse_sequence_extension  (4:2:0/4:2:2/4:4:4)
//!    ▼  find picture_start_code   (0x00)  → parse_picture_header      (I only)
//!    ▼  find extension_start_code (0xB5)  → parse_picture_coding_extension
//!    ▼  for each slice_start_code (0x01..0xAF):
//!         parse_slice_header
//!         for each intra macroblock:
//!           decode 6/8/12 blocks (depending on chroma_format): DC + AC
//!           dequantize_intra → idct_8x8 → +128 level shift → clip → blit
//!    ▼  assemble YUV planar (Yuv420p / Yuv422p / Yuv444p depending on cf)
//! ```
//!
//! Only one I-picture is decoded per [`Mpeg2Decoder::decode`] call (the first
//! one found). The output is a [`Mpeg2Frame`] holding Y/Cb/Cr `Vec<u8>` planes.
//!
//! # Chroma layout (ISO/IEC 13818-2 §6.1.1.4 Table 6-10)
//!
//! - **4:2:0** (`chroma_format == 1`): 4 luma + 1 Cb + 1 Cr = 6 blocks/MB.
//!   Cb/Cr is one 8×8 covering the full 16×16 MB area.
//! - **4:2:2** (`chroma_format == 2`): 4 luma + 2 Cb + 2 Cr = 8 blocks/MB.
//!   Chroma is 8×16 per MB; the two stacked 8×8 are `Cb_top, Cb_bot, Cr_top,
//!   Cr_bot`.
//! - **4:4:4** (`chroma_format == 3`): 4 luma + 4 Cb + 4 Cr = 12 blocks/MB.
//!   Chroma is 16×16 per MB, arranged like luma in a 2×2 raster
//!   `top-left, top-right, bottom-left, bottom-right`.

use oximedia_core::{CodecId, PixelFormat};

use crate::error::{CodecError, CodecResult};
use crate::frame::{Plane, VideoFrame};
use crate::traits::VideoDecoder;

use super::bitreader::{
    find_specific_start_code, find_start_code, BitReader, EXTENSION_START_CODE, PICTURE_START_CODE,
    SEQUENCE_HEADER_CODE, SLICE_START_CODE_MAX, SLICE_START_CODE_MIN,
};
use super::dequant::{dequantize_intra, quantiser_scale};
use super::entropy::{decode_intra_block, BlockComponent, DcPredictors};
use super::headers::{
    full_horizontal_size, full_vertical_size, parse_picture_coding_extension, parse_picture_header,
    parse_sequence_extension, parse_sequence_header, parse_slice_header, PictureCodingExtension,
    SequenceExtension, SequenceHeader,
};
use super::idct::{clip_to_u8, idct_8x8};
use super::Mpeg2Error;
use super::Mpeg2Result;

/// A decoded MPEG-2 I-frame in planar YUV form (4:2:0, 4:2:2 or 4:4:4).
#[derive(Debug, Clone)]
pub struct Mpeg2Frame {
    /// Display width in pixels (luminance width).
    pub width: u32,
    /// Display height in pixels (luminance height).
    pub height: u32,
    /// `chroma_format` of the source (1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4).
    pub chroma_format: u8,
    /// Luminance plane, `width * height` bytes (raster order).
    pub y: Vec<u8>,
    /// Cb plane. Size depends on `chroma_format`:
    /// - 4:2:0 → `(width/2) * (height/2)`
    /// - 4:2:2 → `(width/2) * height`
    /// - 4:4:4 → `width * height`
    pub cb: Vec<u8>,
    /// Cr plane (same dimensions as Cb).
    pub cr: Vec<u8>,
}

/// MPEG-2 video decoder (intra / I-frame only).
///
/// Supports 4:2:0, 4:2:2 and 4:4:4 chroma formats (Wave 10).
#[derive(Debug, Default)]
pub struct Mpeg2Decoder {
    decoded_queue: Vec<VideoFrame>,
    last_dimensions: Option<(u32, u32)>,
    last_pixel_format: Option<PixelFormat>,
}

/// Compute the chroma-plane width relative to the (padded) luma width.
fn chroma_plane_width(luma_w: usize, chroma_format: u8) -> usize {
    match chroma_format {
        2 | 3 => {
            if chroma_format == 3 {
                luma_w
            } else {
                luma_w / 2
            }
        }
        _ => luma_w / 2, // 4:2:0
    }
}

/// Compute the chroma-plane height relative to the (padded) luma height.
fn chroma_plane_height(luma_h: usize, chroma_format: u8) -> usize {
    match chroma_format {
        2 | 3 => {
            if chroma_format == 3 {
                luma_h
            } else {
                luma_h
            }
        }
        _ => luma_h / 2, // 4:2:0
    }
}

/// Display-resolution chroma width: `width / 2` for 4:2:0/4:2:2, `width` for 4:4:4.
fn display_chroma_width(width: usize, chroma_format: u8) -> usize {
    if chroma_format == 3 {
        width
    } else {
        width.div_ceil(2)
    }
}

/// Display-resolution chroma height: `height / 2` for 4:2:0, `height` for 4:2:2/4:4:4.
fn display_chroma_height(height: usize, chroma_format: u8) -> usize {
    if chroma_format == 1 {
        height.div_ceil(2)
    } else {
        height
    }
}

/// Translate `chroma_format` (1/2/3) to the matching planar `PixelFormat`.
fn pixel_format_for_chroma(chroma_format: u8) -> PixelFormat {
    match chroma_format {
        2 => PixelFormat::Yuv422p,
        3 => PixelFormat::Yuv444p,
        _ => PixelFormat::Yuv420p,
    }
}

impl Mpeg2Decoder {
    /// Construct a new decoder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode the first I-frame found in `data`.
    ///
    /// # Errors
    ///
    /// Returns [`Mpeg2Error`] if the stream is truncated, the picture is not an
    /// I-frame, the chroma format is reserved (not 1/2/3), or any header/VLC is
    /// malformed.
    pub fn decode(&self, data: &[u8]) -> Mpeg2Result<Mpeg2Frame> {
        // ── 1. Sequence header ──────────────────────────────────────────────
        let seq_sc = find_specific_start_code(data, 0, SEQUENCE_HEADER_CODE)
            .ok_or(Mpeg2Error::StartCodeNotFound(SEQUENCE_HEADER_CODE))?;
        let mut reader = BitReader::new_at(data, seq_sc.payload_offset);
        let sequence_header = parse_sequence_header(&mut reader)?;

        // ── 2. Sequence extension (must immediately follow as an extension) ──
        let seq_ext = find_specific_start_code(data, seq_sc.payload_offset, EXTENSION_START_CODE)
            .ok_or(Mpeg2Error::StartCodeNotFound(EXTENSION_START_CODE))?;
        let mut reader = BitReader::new_at(data, seq_ext.payload_offset);
        let sequence_extension = parse_sequence_extension(&mut reader)?;
        let chroma_format = sequence_extension.chroma_format;

        // ── 3. Picture header (first picture_start_code after the seq ext) ───
        let pic_sc = find_specific_start_code(data, seq_ext.payload_offset, PICTURE_START_CODE)
            .ok_or(Mpeg2Error::StartCodeNotFound(PICTURE_START_CODE))?;
        let mut reader = BitReader::new_at(data, pic_sc.payload_offset);
        let _picture_header = parse_picture_header(&mut reader)?;

        // ── 4. Picture coding extension ─────────────────────────────────────
        let pce_sc = find_specific_start_code(data, pic_sc.payload_offset, EXTENSION_START_CODE)
            .ok_or(Mpeg2Error::StartCodeNotFound(EXTENSION_START_CODE))?;
        let mut reader = BitReader::new_at(data, pce_sc.payload_offset);
        let picture_coding_extension = parse_picture_coding_extension(&mut reader)?;

        // ── 5. Geometry ─────────────────────────────────────────────────────
        let width = full_horizontal_size(&sequence_header, &sequence_extension);
        let height = full_vertical_size(&sequence_header, &sequence_extension);
        if width == 0 || height == 0 {
            return Err(Mpeg2Error::InvalidData("zero frame dimension".into()));
        }
        // Defend against an allocation bomb: reject frame dimensions above the
        // shared decode ceiling before sizing the Y/Cb/Cr plane allocations.
        crate::limits::check_dimensions(width as usize, height as usize)
            .map_err(Mpeg2Error::InvalidData)?;
        let mb_cols = (width as usize).div_ceil(16);
        let mb_rows = (height as usize).div_ceil(16);
        // Coded (padded) luma dimensions, multiple of 16.
        let coded_w = mb_cols * 16;
        let coded_h = mb_rows * 16;
        let coded_cw = chroma_plane_width(coded_w, chroma_format);
        let coded_ch = chroma_plane_height(coded_h, chroma_format);

        let mut y_plane = vec![0u8; coded_w * coded_h];
        let mut cb_plane = vec![0u8; coded_cw * coded_ch];
        let mut cr_plane = vec![0u8; coded_cw * coded_ch];

        // ── 6. Decode slices ────────────────────────────────────────────────
        let mut search_from = pce_sc.payload_offset;
        let mut decoded_any = false;
        loop {
            let Some(slice_sc) = find_start_code(data, search_from) else {
                break;
            };
            // Stop at the next picture / sequence boundary.
            if slice_sc.code < SLICE_START_CODE_MIN || slice_sc.code > SLICE_START_CODE_MAX {
                break;
            }

            // Slice payload spans up to the next start code prefix.
            let slice_end = find_start_code(data, slice_sc.payload_offset)
                .map_or(data.len(), |next| next.prefix_offset);
            let slice_bytes = &data[slice_sc.payload_offset..slice_end];

            decode_slice(
                slice_bytes,
                slice_sc.code,
                &sequence_header,
                &picture_coding_extension,
                mb_cols,
                coded_w,
                coded_cw,
                chroma_format,
                &mut y_plane,
                &mut cb_plane,
                &mut cr_plane,
            )?;
            decoded_any = true;
            search_from = slice_end;
        }

        if !decoded_any {
            return Err(Mpeg2Error::StartCodeNotFound(SLICE_START_CODE_MIN));
        }

        // ── 7. Crop padding to the display size ─────────────────────────────
        let y = crop_plane(&y_plane, coded_w, width as usize, height as usize);
        let cw = display_chroma_width(width as usize, chroma_format);
        let ch = display_chroma_height(height as usize, chroma_format);
        let cb = crop_plane(&cb_plane, coded_cw, cw, ch);
        let cr = crop_plane(&cr_plane, coded_cw, cw, ch);

        Ok(Mpeg2Frame {
            width,
            height,
            chroma_format,
            y,
            cb,
            cr,
        })
    }
}

/// Crop a (possibly padded) plane of stride `src_stride` down to
/// `dst_w × dst_h` raster bytes.
fn crop_plane(src: &[u8], src_stride: usize, dst_w: usize, dst_h: usize) -> Vec<u8> {
    let mut out = vec![0u8; dst_w * dst_h];
    for row in 0..dst_h {
        let src_off = row * src_stride;
        let dst_off = row * dst_w;
        if src_off + dst_w <= src.len() {
            out[dst_off..dst_off + dst_w].copy_from_slice(&src[src_off..src_off + dst_w]);
        }
    }
    out
}

/// One entry in the macroblock block list: which component, and the
/// `chroma_sub_index` for chroma blocks (0..=1 for 4:2:2, 0..=3 for 4:4:4).
#[derive(Debug, Clone, Copy)]
struct MbBlockSpec {
    component: BlockComponent,
    /// For luma blocks: 0..=3 in the 2×2 raster (Y0..Y3).
    /// For chroma blocks: 0..=N-1 sub-index within the chroma block group.
    sub_index: usize,
}

/// Build the per-MB block list for the given `chroma_format`.
///
/// - 4:2:0 → 6 entries: Y0..Y3, Cb, Cr.
/// - 4:2:2 → 8 entries: Y0..Y3, Cb_top, Cb_bot, Cr_top, Cr_bot.
/// - 4:4:4 → 12 entries: Y0..Y3, Cb0..Cb3, Cr0..Cr3 (2×2 raster, top-left,
///   top-right, bottom-left, bottom-right).
fn macroblock_block_list(chroma_format: u8) -> Vec<MbBlockSpec> {
    let mut out = Vec::with_capacity(12);
    for sub in 0..4 {
        out.push(MbBlockSpec {
            component: BlockComponent::Luma,
            sub_index: sub,
        });
    }
    let chroma_blocks = match chroma_format {
        2 => 2,
        3 => 4,
        _ => 1, // 4:2:0
    };
    for sub in 0..chroma_blocks {
        out.push(MbBlockSpec {
            component: BlockComponent::Cb,
            sub_index: sub,
        });
    }
    for sub in 0..chroma_blocks {
        out.push(MbBlockSpec {
            component: BlockComponent::Cr,
            sub_index: sub,
        });
    }
    out
}

/// Compute the `(origin_x, origin_y)` of the chroma 8×8 block within its
/// plane, given the macroblock coordinates, the chroma sub-index and the
/// chroma format.
///
/// For 4:2:0 there is only one sub-block (`sub_index == 0`) covering the full
/// MB (origin = `mb_col*8, mb_row*8`).
///
/// For 4:2:2 the chroma is 8×16; sub-indices map to:
/// - 0 → top  (8×8 at `mb_col*8, mb_row*16`)
/// - 1 → bottom (8×8 at `mb_col*8, mb_row*16 + 8`)
///
/// For 4:4:4 the chroma is 16×16, identical to luma layout; sub-indices map
/// the 2×2 raster `(0,0), (0,1), (1,0), (1,1)`:
/// - 0 → top-left,     1 → top-right,
/// - 2 → bottom-left,  3 → bottom-right.
fn chroma_block_origin(
    mb_col: usize,
    mb_row: usize,
    sub_index: usize,
    chroma_format: u8,
) -> (usize, usize) {
    match chroma_format {
        2 => {
            // 4:2:2 — chroma is 8×16 per MB, two stacked 8×8.
            let origin_x = mb_col * 8;
            let origin_y = mb_row * 16 + sub_index * 8;
            (origin_x, origin_y)
        }
        3 => {
            // 4:4:4 — chroma is 16×16 per MB, four 8×8 in 2×2 raster.
            let block_x = (sub_index & 1) * 8;
            let block_y = (sub_index >> 1) * 8;
            let origin_x = mb_col * 16 + block_x;
            let origin_y = mb_row * 16 + block_y;
            (origin_x, origin_y)
        }
        _ => {
            // 4:2:0 — one 8×8 per MB covering the full chroma area.
            (mb_col * 8, mb_row * 8)
        }
    }
}

/// Decode one slice's worth of intra macroblocks into the output planes.
#[allow(clippy::too_many_arguments)]
fn decode_slice(
    slice_bytes: &[u8],
    slice_start_code: u8,
    sequence_header: &SequenceHeader,
    pce: &PictureCodingExtension,
    mb_cols: usize,
    coded_w: usize,
    coded_cw: usize,
    chroma_format: u8,
    y_plane: &mut [u8],
    cb_plane: &mut [u8],
    cr_plane: &mut [u8],
) -> Mpeg2Result<()> {
    let mut reader = BitReader::new(slice_bytes);
    let slice_header = parse_slice_header(&mut reader, slice_start_code)?;

    // slice_vertical_position is 1-based; macroblock row = position - 1.
    let mb_row = (slice_header.slice_vertical_position as usize).saturating_sub(1);
    let mut q_scale = quantiser_scale(slice_header.quantiser_scale_code, pce.q_scale_type);

    // DC predictors reset at slice start, per-component (Y, Cb, Cr). They are
    // independent of `chroma_format`: §7.2.1 mandates one predictor per
    // component reset to `1 << (7+intra_dc_precision)`, and that single
    // predictor is reused across all chroma blocks of the same component in
    // the slice (so 2 or 4 chroma blocks per MB still share `cb` / `cr`).
    let mut predictors = DcPredictors::reset(pce.intra_dc_precision);

    let block_list = macroblock_block_list(chroma_format);

    // Macroblock address within the row, 0-based. First MB increment is read
    // from the bitstream and is normally 1 (placing us at column 0).
    let mut mb_col: isize = -1;

    loop {
        if reader.remaining_bits() < 2 {
            break;
        }
        // ── macroblock_address_increment (Table B-1) ───────────────────────
        let increment = match read_macroblock_address_increment(&mut reader)? {
            Some(inc) => inc,
            None => break, // end of slice / unparseable → stop.
        };
        mb_col += increment as isize;
        if mb_col < 0 || mb_col as usize >= mb_cols {
            break;
        }

        // ── macroblock_type (Table B-2, I-picture) ─────────────────────────
        let macroblock_quant = read_macroblock_type_intra(&mut reader)?;
        if macroblock_quant {
            let new_code = reader.read_bits(5)? as u8;
            q_scale = quantiser_scale(new_code, pce.q_scale_type);
        }

        // ── 6 / 8 / 12 blocks per macroblock (dispatched on chroma_format) ──
        for spec in &block_list {
            let quantised = decode_intra_block(
                &mut reader,
                &mut predictors,
                spec.component,
                pce.intra_vlc_format,
                pce.alternate_scan,
            )?;
            let recon = dequantize_intra(
                &quantised,
                &sequence_header.intra_quantiser_matrix,
                pce.intra_dc_precision,
                q_scale,
            );
            let spatial = idct_8x8(&recon);

            blit_block(
                &spatial,
                *spec,
                mb_row,
                mb_col as usize,
                coded_w,
                coded_cw,
                chroma_format,
                y_plane,
                cb_plane,
                cr_plane,
            );
        }
    }

    Ok(())
}

/// Blit one reconstructed 8×8 spatial block into the appropriate output plane.
#[allow(clippy::too_many_arguments)]
fn blit_block(
    spatial: &[i32; 64],
    spec: MbBlockSpec,
    mb_row: usize,
    mb_col: usize,
    coded_w: usize,
    coded_cw: usize,
    chroma_format: u8,
    y_plane: &mut [u8],
    cb_plane: &mut [u8],
    cr_plane: &mut [u8],
) {
    match spec.component {
        BlockComponent::Luma => {
            // Luma blocks are arranged 2×2 within a 16×16 macroblock.
            let block_x = (spec.sub_index & 1) * 8;
            let block_y = (spec.sub_index >> 1) * 8;
            let origin_x = mb_col * 16 + block_x;
            let origin_y = mb_row * 16 + block_y;
            write_block(spatial, y_plane, coded_w, origin_x, origin_y);
        }
        BlockComponent::Cb => {
            let (origin_x, origin_y) =
                chroma_block_origin(mb_col, mb_row, spec.sub_index, chroma_format);
            write_block(spatial, cb_plane, coded_cw, origin_x, origin_y);
        }
        BlockComponent::Cr => {
            let (origin_x, origin_y) =
                chroma_block_origin(mb_col, mb_row, spec.sub_index, chroma_format);
            write_block(spatial, cr_plane, coded_cw, origin_x, origin_y);
        }
    }
}

/// Write an 8×8 spatial block into `plane` at `(origin_x, origin_y)`, clipping
/// to `[0, 255]`.
///
/// For MPEG-2 **intra** blocks the inverse DCT output is the reconstructed
/// sample value directly (the intra DC predictor is reset to `2^(7+precision)`,
/// i.e. mid-grey, in the quantised domain), so **no** additional level shift is
/// applied here — only clipping (ISO/IEC 13818-2 §7.4.4 / §7.6).
fn write_block(
    spatial: &[i32; 64],
    plane: &mut [u8],
    stride: usize,
    origin_x: usize,
    origin_y: usize,
) {
    for r in 0..8 {
        let py = origin_y + r;
        for c in 0..8 {
            let px = origin_x + c;
            let idx = py * stride + px;
            if idx < plane.len() {
                plane[idx] = clip_to_u8(spatial[r * 8 + c]);
            }
        }
    }
}

/// Decode `macroblock_address_increment` (Table B-1).
///
/// Returns `Ok(Some(increment))` on success, `Ok(None)` if the bits cannot be
/// matched (treated as the end of the slice). Supports increments up to a small
/// range plus the escape/stuffing codes that map to no increment.
fn read_macroblock_address_increment(reader: &mut BitReader<'_>) -> Mpeg2Result<Option<u32>> {
    // Table B-1 (subset sufficient for sequential intra macroblocks):
    // 1            -> 1
    // 011          -> 2
    // 010          -> 3
    // 0011         -> 4
    // 0010         -> 5
    // 00011        -> 6
    // 00010        -> 7
    // 0000111      -> 8
    // 0000110      -> 9
    // macroblock_escape = 0000 0001 000 (+33), macroblock_stuffing = 0000 0001 111.
    let peek = reader.peek_bits_msb_aligned();
    // Most common case: increment 1 (single `1` bit).
    if peek >> 31 == 1 {
        reader.skip_bits(1)?;
        return Ok(Some(1));
    }
    // Match remaining short codes.
    const TABLE: &[(u32, u8, u32)] = &[
        (0b011, 3, 2),
        (0b010, 3, 3),
        (0b0011, 4, 4),
        (0b0010, 4, 5),
        (0b00011, 5, 6),
        (0b00010, 5, 7),
        (0b0000111, 7, 8),
        (0b0000110, 7, 9),
        (0b00001011, 8, 10),
        (0b00001010, 8, 11),
        (0b00001001, 8, 12),
        (0b00001000, 8, 13),
        (0b00000111, 8, 14),
        (0b00000110, 8, 15),
        (0b0000010111, 10, 16),
    ];
    for &(code, len, inc) in TABLE {
        if peek >> (32 - u32::from(len)) == u32::from(code) {
            reader.skip_bits(len)?;
            return Ok(Some(inc));
        }
    }
    Ok(None)
}

/// Decode `macroblock_type` for an I-picture (Table B-2).
///
/// Returns `Ok(macroblock_quant)`: `false` for plain Intra (code `1`), `true`
/// for Intra with a new quantiser scale (code `01`).
fn read_macroblock_type_intra(reader: &mut BitReader<'_>) -> Mpeg2Result<bool> {
    let first = reader.read_bit()?;
    if first {
        // `1` → Intra, no quant change.
        return Ok(false);
    }
    let second = reader.read_bit()?;
    if second {
        // `01` → Intra + quant.
        Ok(true)
    } else {
        Err(Mpeg2Error::InvalidData(
            "invalid I-picture macroblock_type (expected `1` or `01`)".into(),
        ))
    }
}

// ── VideoDecoder trait (push-pull) ───────────────────────────────────────────

impl VideoDecoder for Mpeg2Decoder {
    fn codec(&self) -> CodecId {
        CodecId::Mpeg2
    }

    fn send_packet(&mut self, data: &[u8], pts: i64) -> CodecResult<()> {
        let frame = self
            .decode(data)
            .map_err(|e| CodecError::InvalidBitstream(e.to_string()))?;

        let w = frame.width;
        let h = frame.height;
        let cw = display_chroma_width(w as usize, frame.chroma_format) as u32;
        let ch = display_chroma_height(h as usize, frame.chroma_format) as u32;
        let pix_fmt = pixel_format_for_chroma(frame.chroma_format);
        self.last_dimensions = Some((w, h));
        self.last_pixel_format = Some(pix_fmt);

        let mut vf = VideoFrame::new(pix_fmt, w, h);
        vf.timestamp.pts = pts;
        vf.planes = vec![
            Plane::with_dimensions(frame.y, w as usize, w, h),
            Plane::with_dimensions(frame.cb, cw as usize, cw, ch),
            Plane::with_dimensions(frame.cr, cw as usize, cw, ch),
        ];

        self.decoded_queue.push(vf);
        Ok(())
    }

    fn receive_frame(&mut self) -> CodecResult<Option<VideoFrame>> {
        if self.decoded_queue.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.decoded_queue.remove(0)))
        }
    }

    fn flush(&mut self) -> CodecResult<()> {
        // Intra-only: nothing is buffered for reordering.
        Ok(())
    }

    fn reset(&mut self) {
        self.decoded_queue.clear();
        self.last_dimensions = None;
        self.last_pixel_format = None;
    }

    fn output_format(&self) -> Option<PixelFormat> {
        // Falls back to 4:2:0 until a frame has been seen, since the chroma
        // format is per-stream and only known after decoding the sequence
        // extension.
        Some(self.last_pixel_format.unwrap_or(PixelFormat::Yuv420p))
    }

    fn dimensions(&self) -> Option<(u32, u32)> {
        self.last_dimensions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_plane_extracts_top_left() {
        // 4-wide padded plane, crop to 2×2.
        let src = vec![
            1, 2, 9, 9, //
            3, 4, 9, 9, //
            9, 9, 9, 9,
        ];
        let out = crop_plane(&src, 4, 2, 2);
        assert_eq!(out, vec![1, 2, 3, 4]);
    }

    #[test]
    fn write_block_clips_without_level_shift() {
        let mut plane = vec![0u8; 64];
        let mut spatial = [0i32; 64];
        spatial[0] = 128; // direct sample value
        spatial[1] = 300; // clip 255
        spatial[2] = -10; // clip 0
        write_block(&spatial, &mut plane, 8, 0, 0);
        assert_eq!(plane[0], 128);
        assert_eq!(plane[1], 255);
        assert_eq!(plane[2], 0);
    }

    #[test]
    fn mb_increment_one_is_single_bit() {
        let bytes = [0b1000_0000u8];
        let mut r = BitReader::new(&bytes);
        let inc = read_macroblock_address_increment(&mut r).expect("inc");
        assert_eq!(inc, Some(1));
        assert_eq!(r.byte_pos(), 0);
        assert_eq!(r.remaining_bits(), 7);
    }

    #[test]
    fn mb_type_intra_plain() {
        let bytes = [0b1000_0000u8];
        let mut r = BitReader::new(&bytes);
        assert!(!read_macroblock_type_intra(&mut r).expect("type"));
    }

    #[test]
    fn mb_type_intra_quant() {
        let bytes = [0b0100_0000u8];
        let mut r = BitReader::new(&bytes);
        assert!(read_macroblock_type_intra(&mut r).expect("type"));
    }

    #[test]
    fn codec_id_is_mpeg2() {
        let dec = Mpeg2Decoder::new();
        assert_eq!(dec.codec(), CodecId::Mpeg2);
        assert_eq!(dec.output_format(), Some(PixelFormat::Yuv420p));
    }

    #[test]
    fn decode_missing_sequence_header_errors() {
        let dec = Mpeg2Decoder::new();
        let data = [0xDE, 0xAD, 0xBE, 0xEF];
        assert!(matches!(
            dec.decode(&data),
            Err(Mpeg2Error::StartCodeNotFound(_))
        ));
    }
}
