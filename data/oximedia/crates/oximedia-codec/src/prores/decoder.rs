//! ProRes 422 frame-level decoder.
//!
//! Wraps the low-level per-slice pipeline ([`super::decode`]) in a
//! frame-oriented API that accepts a raw `'icpf'` byte slice and returns a
//! [`ProResFrame`] with fully assembled YUV 4:2:2 planar output.
//!
//! ## Pipeline
//!
//! ```text
//!   raw bytes  ('icpf' frame)
//!     │
//!     ▼  FrameContainer::parse
//!   container payload
//!     │
//!     ▼  parse_frame_header
//!   FrameHeader  (quant matrices, dimensions, interlace_mode)
//!     │
//!     ▼  parse_picture_header   ×  pictures_per_frame
//!   PictureHeader (slice_count, log2_slice_mb_width)
//!     │
//!     ▼  for each slice:
//!       parse_slice_header  →  split_slice_planes
//!       decode_slice_to_yuv422  →  blit into full-frame planes
//!     │
//!     ▼  ProResFrame
//!   y: Vec<u8>, cb: Vec<u8>, cr: Vec<u8>   (8-bit, downscaled from 10-bit)
//! ```
//!
//! ## 10-bit → 8-bit Conversion
//!
//! ProRes is natively 10-bit. The decoder outputs 8-bit per-channel
//! (`Vec<u8>`) by right-shifting each 10-bit sample by 2 (drop the two LSBs).
//! This is the conventional fast 10→8 mapping used by FFmpeg and most display
//! pipelines.
//!
//! ## Profiles
//!
//! All five 4:2:2 sub-profiles (`apco`/`apcs`/`apcn`/`apch`) plus the two
//! 4:4:4 profiles (`ap4h`/`ap4x`) share the same bitstream syntax — only the
//! chroma plane geometry differs. The decoder auto-detects the profile and
//! the chroma format from the `'icpf'` frame header. When
//! [`ProResDecoderConfig::profile`] is set to `Some(p)`, the decoder verifies
//! that the stream matches `p` and returns an error if it does not.
//!
//! ## 4:4:4 (ProRes 4444 / 4444 XQ)
//!
//! For `chroma_format == Yuv444` each macroblock carries 4 luma + 4 Cb + 4 Cr
//! 8×8 blocks (full-resolution chroma), so the chroma planes are the same
//! width as luma and the per-slice chroma byte budget doubles relative to
//! 4:2:2. The DCT-coded picture is decoded fully; the optional run-length
//! coded **alpha** plane of `'4444'` streams is currently dropped.
//!
//! ## Interlaced streams
//!
//! Top-field-first and bottom-field-first interlaced frames contain two
//! pictures. Both pictures are decoded and assembled into a single interleaved
//! output frame (even rows = field 0, odd rows = field 1 for TFF; reversed for
//! BFF). The output [`ProResFrame::is_interlaced`] flag is set accordingly.

use crate::error::{CodecError, CodecResult};
use crate::frame::{Plane, VideoFrame};
use crate::traits::VideoDecoder;
use oximedia_core::{CodecId, PixelFormat};

use super::decode::{decode_slice_to_yuv422, decode_slice_to_yuv444, split_slice_planes};
use super::frame::{
    parse_frame_header, ChromaFormat, FrameContainer, InterlaceMode, ProResProfile,
};
use super::picture::{parse_picture_header, parse_slice_header};

// ─── Configuration ────────────────────────────────────────────────────────────

/// ProRes 422 decoder configuration.
#[derive(Debug, Clone, Default)]
pub struct ProResDecoderConfig {
    /// Expected ProRes profile (Proxy, LT, Standard, HQ).
    ///
    /// When `Some`, the decoder validates that the stream's embedded profile
    /// FourCC matches and returns [`CodecError::InvalidParameter`] when it
    /// does not. When `None` (the default), the profile is auto-detected from
    /// the stream.
    pub profile: Option<ProResProfile>,
}

// ─── Output frame ─────────────────────────────────────────────────────────────

/// A fully decoded ProRes video frame.
///
/// Planes are 8-bit per sample (10-bit stream values right-shifted by 2).
/// For 4:2:2 streams the chroma planes are half the horizontal luma
/// resolution; for 4:4:4 streams they are full resolution. Inspect
/// [`ProResFrame::chroma_format`] (or [`ProResFrame::chroma_width`]) to tell
/// which.
#[derive(Debug, Clone)]
pub struct ProResFrame {
    /// Luma width in pixels.
    pub width: u32,
    /// Luma height in pixels.
    pub height: u32,
    /// `true` when the source stream was interlaced (top- or bottom-field-first).
    pub is_interlaced: bool,
    /// Which ProRes profile this frame was encoded with.
    pub profile: ProResProfile,
    /// Chroma subsampling of this frame (`Yuv422` or `Yuv444`).
    pub chroma_format: ChromaFormat,
    /// Y (luma) plane, row-major, `width × height` 8-bit samples.
    pub y: Vec<u8>,
    /// Cb (chroma-blue) plane, row-major, `chroma_width × height` 8-bit samples.
    pub cb: Vec<u8>,
    /// Cr (chroma-red) plane, row-major, `chroma_width × height` 8-bit samples.
    pub cr: Vec<u8>,
}

impl ProResFrame {
    /// Width of the chroma (`cb` / `cr`) planes in samples.
    ///
    /// Equals `width` for 4:4:4 frames and `width / 2` for 4:2:2 frames.
    #[must_use]
    pub fn chroma_width(&self) -> u32 {
        match self.chroma_format {
            ChromaFormat::Yuv444 => self.width,
            ChromaFormat::Yuv422 => self.width / 2,
        }
    }
}

// ─── Decoder ──────────────────────────────────────────────────────────────────

/// ProRes 422 decoder.
///
/// Decodes a complete Apple ProRes 422 `'icpf'` frame bitstream from a
/// `&[u8]` byte slice into 8-bit YUV 4:2:2 planar output.
///
/// # Usage
///
/// ```ignore
/// use oximedia_codec::prores::{ProResDecoder, ProResDecoderConfig};
///
/// let compressed: Vec<u8> = /* your ProRes 422 frame bytes */;
/// let decoder = ProResDecoder::new();
/// let frame = ProResDecoder::decode(&compressed)?;
/// println!("{}×{} {:?}", frame.width, frame.height, frame.profile);
/// ```
pub struct ProResDecoder {
    config: ProResDecoderConfig,
    /// Packets pushed via the [`VideoDecoder`] trait that have not yet been
    /// consumed by [`VideoDecoder::receive_frame`].
    pending_pts: Vec<i64>,
    decoded_queue: Vec<VideoFrame>,
    /// Pixel format of the most recently decoded packet. `None` until the
    /// first packet is seen, since the chroma format is a per-stream property.
    last_pixel_format: Option<PixelFormat>,
}

impl Default for ProResDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ProResDecoder {
    /// Create a decoder with default configuration (auto-detect profile).
    pub fn new() -> Self {
        Self {
            config: ProResDecoderConfig::default(),
            pending_pts: Vec::new(),
            decoded_queue: Vec::new(),
            last_pixel_format: None,
        }
    }

    /// Create a decoder with explicit configuration.
    pub fn with_config(config: ProResDecoderConfig) -> Self {
        Self {
            config,
            pending_pts: Vec::new(),
            decoded_queue: Vec::new(),
            last_pixel_format: None,
        }
    }

    /// Decode a complete ProRes 422 `'icpf'` frame from `data`.
    ///
    /// Returns a [`ProResFrame`] with 8-bit YUV 4:2:2 planar output.
    ///
    /// # Errors
    ///
    /// - [`CodecError::InvalidBitstream`] — the container tag is wrong, the
    ///   frame header is truncated or malformed, or slice parsing fails.
    /// - [`CodecError::InvalidParameter`] — a non-`None` `config.profile` was
    ///   specified but the stream's profile does not match.
    pub fn decode(data: &[u8]) -> CodecResult<ProResFrame> {
        let decoder = Self::new();
        decoder.decode_impl(data)
    }

    /// Decode with the decoder's own configuration.
    pub fn decode_with_config(&self, data: &[u8]) -> CodecResult<ProResFrame> {
        self.decode_impl(data)
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    fn decode_impl(&self, data: &[u8]) -> CodecResult<ProResFrame> {
        // ── 1. Parse frame container ──────────────────────────────────────────
        if data.len() < 8 {
            return Err(CodecError::InvalidBitstream(format!(
                "ProRes frame too short: {} bytes (minimum 8)",
                data.len()
            )));
        }
        let (container, _rest) =
            FrameContainer::parse(data).map_err(|e| CodecError::InvalidBitstream(e.to_string()))?;

        // ── 2. Parse frame header ─────────────────────────────────────────────
        let (fhdr, after_fhdr) = parse_frame_header(container.payload)
            .map_err(|e| CodecError::InvalidBitstream(e.to_string()))?;

        // Validate against requested profile.
        if let Some(expected) = self.config.profile {
            if fhdr.profile != expected {
                return Err(CodecError::InvalidParameter(format!(
                    "ProRes decoder: stream profile {:?} != expected {:?}",
                    fhdr.profile, expected
                )));
            }
        }

        let width = fhdr.width as usize;
        let height = fhdr.height as usize;
        // 4:2:2 → chroma is half-width; 4:4:4 → chroma is full-width.
        let chroma_width = match fhdr.chroma_format {
            ChromaFormat::Yuv422 => width / 2,
            ChromaFormat::Yuv444 => width,
        };
        let is_interlaced = !matches!(fhdr.interlace_mode, InterlaceMode::Progressive);
        let has_alpha = fhdr.alpha_channel_type != 0;

        // Defend against an allocation bomb: ProRes width/height are 16-bit
        // header fields (up to 0xFFFF each) that drive the Y/Cb/Cr plane
        // allocations below; reject impossibly large frames before allocating.
        crate::limits::checked_dims(width, height, 1, 2).map_err(CodecError::InvalidData)?;

        // Allocate full-frame 10-bit planes (u16 samples). Chroma planes are
        // sized per the chroma format (full-resolution for 4:4:4).
        let mut y_plane = vec![0u16; width * height];
        let mut cb_plane = vec![0u16; chroma_width * height];
        let mut cr_plane = vec![0u16; chroma_width * height];

        // ── 3. Decode picture(s) ──────────────────────────────────────────────
        let num_pictures = fhdr.pictures_per_frame();
        let mut picture_payload = after_fhdr;

        for pic_idx in 0..num_pictures {
            // Interlaced: first picture = field 0, second = field 1.
            let field_row_stride = if is_interlaced { num_pictures } else { 1 };
            let field_row_offset = if is_interlaced {
                match fhdr.interlace_mode {
                    InterlaceMode::TopFieldFirst => pic_idx,
                    InterlaceMode::BottomFieldFirst => 1 - pic_idx,
                    InterlaceMode::Progressive => 0,
                }
            } else {
                0
            };

            picture_payload = decode_picture(
                picture_payload,
                &fhdr.luma_quant_matrix,
                &fhdr.chroma_quant_matrix,
                fhdr.chroma_format,
                has_alpha,
                width,
                height,
                field_row_offset,
                field_row_stride,
                &mut y_plane,
                &mut cb_plane,
                &mut cr_plane,
            )?;
        }

        // ── 4. Convert 10-bit planes to 8-bit output ──────────────────────────
        let y_out: Vec<u8> = y_plane.iter().map(|&s| (s >> 2) as u8).collect();
        let cb_out: Vec<u8> = cb_plane.iter().map(|&s| (s >> 2) as u8).collect();
        let cr_out: Vec<u8> = cr_plane.iter().map(|&s| (s >> 2) as u8).collect();

        Ok(ProResFrame {
            width: width as u32,
            height: height as u32,
            is_interlaced,
            profile: fhdr.profile,
            chroma_format: fhdr.chroma_format,
            y: y_out,
            cb: cb_out,
            cr: cr_out,
        })
    }
}

// ─── VideoDecoder trait impl ──────────────────────────────────────────────────

impl VideoDecoder for ProResDecoder {
    fn codec(&self) -> CodecId {
        CodecId::ProRes
    }

    fn send_packet(&mut self, data: &[u8], pts: i64) -> CodecResult<()> {
        let frame = self.decode_impl(data)?;

        // Convert ProResFrame (8-bit u8 planes) to VideoFrame. The output
        // pixel format and chroma plane geometry track the stream's chroma
        // format (4:2:2 → Yuv422p / half-width chroma; 4:4:4 → Yuv444p /
        // full-width chroma).
        let w = frame.width;
        let h = frame.height;
        let cw = frame.chroma_width();
        let pix_fmt = pixel_format_for_chroma(frame.chroma_format);

        let mut vf = VideoFrame::new(pix_fmt, w, h);
        vf.timestamp.pts = pts;
        vf.planes = vec![
            Plane::with_dimensions(frame.y, w as usize, w, h),
            Plane::with_dimensions(frame.cb, cw as usize, cw, h),
            Plane::with_dimensions(frame.cr, cw as usize, cw, h),
        ];

        self.last_pixel_format = Some(pix_fmt);
        self.pending_pts.push(pts);
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
        // ProRes is intra-only; no buffered frames to flush.
        Ok(())
    }

    fn reset(&mut self) {
        self.pending_pts.clear();
        self.decoded_queue.clear();
        self.last_pixel_format = None;
    }

    fn output_format(&self) -> Option<PixelFormat> {
        // The chroma format is a per-stream property only known after a
        // packet has been decoded. Until then, report the 4:2:2 default
        // (the overwhelmingly common ProRes case).
        Some(self.last_pixel_format.unwrap_or(PixelFormat::Yuv422p))
    }

    fn dimensions(&self) -> Option<(u32, u32)> {
        None
    }
}

// ─── Chroma-format helpers ────────────────────────────────────────────────────

/// Map a ProRes [`ChromaFormat`] to the matching planar [`PixelFormat`].
fn pixel_format_for_chroma(chroma: ChromaFormat) -> PixelFormat {
    match chroma {
        ChromaFormat::Yuv422 => PixelFormat::Yuv422p,
        ChromaFormat::Yuv444 => PixelFormat::Yuv444p,
    }
}

// ─── Picture decoder ──────────────────────────────────────────────────────────

/// Decode one picture (field) from `payload` and blit slice output into the
/// full-frame planes at the appropriate row offsets.
///
/// Handles both 4:2:2 and 4:4:4 chroma formats: `chroma_format` selects the
/// chroma plane geometry (half-width vs full-width) and the per-slice chroma
/// decoder.
///
/// `field_row_offset` and `field_row_stride` implement interlaced field
/// interleaving:
///   - Progressive: `offset = 0`, `stride = 1` → each row maps 1:1.
///   - Top-field-first, field 0: `offset = 0`, `stride = 2` → rows 0,2,4,…
///   - Top-field-first, field 1: `offset = 1`, `stride = 2` → rows 1,3,5,…
///   - Bottom-field-first reverses the field→offset mapping.
///
/// Returns the remaining bytes of the container payload after this picture.
#[allow(clippy::too_many_arguments)]
fn decode_picture<'a>(
    payload: &'a [u8],
    luma_matrix: &[u8; 64],
    chroma_matrix: &[u8; 64],
    chroma_format: ChromaFormat,
    has_alpha: bool,
    frame_width: usize,
    frame_height: usize,
    field_row_offset: usize,
    field_row_stride: usize,
    y_plane: &mut [u16],
    cb_plane: &mut [u16],
    cr_plane: &mut [u16],
) -> CodecResult<&'a [u8]> {
    let (pic_hdr, after_pic_hdr) =
        parse_picture_header(payload).map_err(|e| CodecError::InvalidBitstream(e.to_string()))?;

    let is_444 = chroma_format == ChromaFormat::Yuv444;
    let slice_count = pic_hdr.slice_count as usize;
    let mb_width_per_slice = 1usize << pic_hdr.log2_slice_mb_width;
    let mb_cols_total = frame_width / 16;
    let mb_rows_total = frame_height / 16;
    let slices_per_row = (mb_cols_total + mb_width_per_slice - 1) / mb_width_per_slice;
    // 4:4:4 chroma is full-width; 4:2:2 chroma is half-width.
    let chroma_frame_width = if is_444 { frame_width } else { frame_width / 2 };

    // ── Slice offset table ────────────────────────────────────────────────────
    // The table immediately follows the picture header: slice_count × 2-byte
    // big-endian entries giving the byte size of each slice (header + data).
    let offset_table_bytes = slice_count * 2;
    if after_pic_hdr.len() < offset_table_bytes {
        return Err(CodecError::InvalidBitstream(format!(
            "ProRes picture: slice offset table truncated (need {} bytes, have {})",
            offset_table_bytes,
            after_pic_hdr.len()
        )));
    }

    let mut slice_sizes = Vec::with_capacity(slice_count);
    for i in 0..slice_count {
        let base = i * 2;
        let sz = u16::from_be_bytes([after_pic_hdr[base], after_pic_hdr[base + 1]]) as usize;
        slice_sizes.push(sz);
    }

    // Slice data starts after the offset table.
    let slice_data = &after_pic_hdr[offset_table_bytes..];

    // ── Iterate slices ────────────────────────────────────────────────────────
    let mut cursor = 0usize;

    for slice_idx in 0..slice_count {
        let mb_row = slice_idx / slices_per_row;
        let slice_col = slice_idx % slices_per_row;

        // Guard against malformed slice_count that exceeds the frame grid.
        if mb_row >= mb_rows_total {
            break;
        }

        let sz = slice_sizes[slice_idx];
        if cursor + sz > slice_data.len() {
            return Err(CodecError::InvalidBitstream(format!(
                "ProRes slice {}: data overrun (need {} bytes at offset {}, have {})",
                slice_idx,
                sz,
                cursor,
                slice_data.len()
            )));
        }

        let slice_bytes = &slice_data[cursor..cursor + sz];
        cursor += sz;

        // Parse slice header.
        let (shdr, slice_payload) = parse_slice_header(slice_bytes, has_alpha)
            .map_err(|e| CodecError::InvalidBitstream(e.to_string()))?;

        // Split payload into per-plane sub-slices.
        let slice_data_ref = split_slice_planes(
            slice_payload,
            shdr.luma_data_size,
            shdr.cb_data_size,
            shdr.cr_data_size,
            shdr.alpha_data_size,
        )
        .map_err(|e| CodecError::InvalidBitstream(e.to_string()))?;

        // Macroblock dimensions for this slice.
        let mb_x_start = slice_col * mb_width_per_slice;
        let mb_x_end = (mb_x_start + mb_width_per_slice).min(mb_cols_total);
        let this_mb_width = mb_x_end - mb_x_start;
        let slice_luma_w = this_mb_width * 16;
        // 4:4:4 chroma occupies the full 16-sample MB width; 4:2:2 only 8.
        let slice_chroma_w = if is_444 {
            this_mb_width * 16
        } else {
            this_mb_width * 8
        };

        // Temporary buffers for this slice's output.
        let mut dst_luma = vec![0u16; slice_luma_w * 16];
        let mut dst_cb = vec![0u16; slice_chroma_w * 16];
        let mut dst_cr = vec![0u16; slice_chroma_w * 16];

        if is_444 {
            decode_slice_to_yuv444(
                slice_data_ref,
                luma_matrix,
                chroma_matrix,
                shdr.quant_scale,
                this_mb_width,
                &mut dst_luma,
                slice_luma_w,
                &mut dst_cb,
                slice_chroma_w,
                &mut dst_cr,
                slice_chroma_w,
            )
            .map_err(|e| CodecError::DecoderError(e.to_string()))?;
        } else {
            decode_slice_to_yuv422(
                slice_data_ref,
                luma_matrix,
                chroma_matrix,
                shdr.quant_scale,
                this_mb_width,
                &mut dst_luma,
                slice_luma_w,
                &mut dst_cb,
                slice_chroma_w,
                &mut dst_cr,
                slice_chroma_w,
            )
            .map_err(|e| CodecError::DecoderError(e.to_string()))?;
        }

        // Blit slice output into the full-frame planes.
        //
        // The slice covers rows [mb_row*16 .. mb_row*16+16) (in picture
        // coordinates). For interlaced output these map to every
        // `field_row_stride`-th row of the frame starting at
        // `mb_row * 16 * field_row_stride + field_row_offset`.
        let frame_col_start_luma = mb_x_start * 16;
        // Chroma column origin: 16 samples per MB in 4:4:4, 8 in 4:2:2.
        let frame_col_start_chroma = if is_444 {
            mb_x_start * 16
        } else {
            mb_x_start * 8
        };

        for slice_row in 0..16 {
            let frame_row =
                mb_row * 16 * field_row_stride + slice_row * field_row_stride + field_row_offset;
            if frame_row >= frame_height {
                break;
            }

            // Luma row.
            let src_luma_base = slice_row * slice_luma_w;
            let dst_luma_base = frame_row * frame_width + frame_col_start_luma;
            let copy_luma = slice_luma_w.min(frame_width.saturating_sub(frame_col_start_luma));
            if dst_luma_base + copy_luma <= y_plane.len() {
                y_plane[dst_luma_base..dst_luma_base + copy_luma]
                    .copy_from_slice(&dst_luma[src_luma_base..src_luma_base + copy_luma]);
            }

            // Chroma rows (4:2:2 — same number of rows as luma).
            let src_chroma_base = slice_row * slice_chroma_w;
            let dst_chroma_base = frame_row * chroma_frame_width + frame_col_start_chroma;
            let copy_chroma =
                slice_chroma_w.min(chroma_frame_width.saturating_sub(frame_col_start_chroma));
            if dst_chroma_base + copy_chroma <= cb_plane.len() {
                cb_plane[dst_chroma_base..dst_chroma_base + copy_chroma]
                    .copy_from_slice(&dst_cb[src_chroma_base..src_chroma_base + copy_chroma]);
                cr_plane[dst_chroma_base..dst_chroma_base + copy_chroma]
                    .copy_from_slice(&dst_cr[src_chroma_base..src_chroma_base + copy_chroma]);
            }
        }
    }

    // Return the bytes after this picture's data.
    // picture_size covers header + offset_table + slice_data.
    let picture_total = pic_hdr.picture_size as usize;
    let consumed = picture_total.min(payload.len());
    Ok(&payload[consumed..])
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Plane, VideoFrame};
    use crate::prores::encoder::{ProResEncoder, ProResEncoderConfig};
    use crate::prores::frame::ProResProfile;
    use crate::traits::VideoEncoder;

    /// Build a flat-grey Yuv422p10le frame of the given dimensions.
    /// All luma samples = `y_val`, chroma = 512 (10-bit mid-grey).
    fn flat_grey_frame(width: u32, height: u32, y_val: u16) -> VideoFrame {
        let w = width as usize;
        let h = height as usize;
        let cw = w / 2;

        let y_bytes: Vec<u8> = (0..w * h).flat_map(|_| y_val.to_le_bytes()).collect();
        let chroma_bytes: Vec<u8> = (0..cw * h).flat_map(|_| 512u16.to_le_bytes()).collect();

        let mut frame = VideoFrame::new(PixelFormat::Yuv422p10le, width, height);
        frame.planes = vec![
            Plane::with_dimensions(y_bytes, w * 2, width, height),
            Plane::with_dimensions(chroma_bytes.clone(), cw * 2, width / 2, height),
            Plane::with_dimensions(chroma_bytes, cw * 2, width / 2, height),
        ];
        frame
    }

    /// Encode a frame with ProResEncoder and return the packet bytes.
    fn encode_frame(frame: &VideoFrame) -> Vec<u8> {
        let cfg = ProResEncoderConfig::new(ProResProfile::Standard, frame.width, frame.height);
        let mut enc = ProResEncoder::new(cfg).expect("encoder");
        enc.send_frame(frame).expect("send_frame");
        enc.receive_packet().expect("receive").expect("packet").data
    }

    // ─── Smoke / unit tests ───────────────────────────────────────────────────

    #[test]
    fn decoder_new_and_default() {
        let _d1 = ProResDecoder::new();
        let _d2 = ProResDecoder::default();
        // Both should construct without panic.
    }

    #[test]
    fn decode_rejects_too_short() {
        let result = ProResDecoder::decode(&[0u8; 4]);
        assert!(result.is_err(), "should reject frames shorter than 8 bytes");
    }

    #[test]
    fn decode_rejects_bad_container_tag() {
        // Build a valid-looking header but corrupt the 'icpf' tag.
        let mut buf = vec![0u8; 32];
        // frame_size = 32 (big-endian)
        buf[0] = 0;
        buf[1] = 0;
        buf[2] = 0;
        buf[3] = 32;
        // Bad tag: 'XXXX' instead of 'icpf'
        buf[4] = b'X';
        buf[5] = b'X';
        buf[6] = b'X';
        buf[7] = b'X';
        assert!(ProResDecoder::decode(&buf).is_err());
    }

    #[test]
    fn profile_from_fourcc_all_variants() {
        // Test that all known profile FourCCs round-trip through from_fourcc.
        let cases: &[(&[u8; 4], ProResProfile)] = &[
            (b"apco", ProResProfile::Proxy),
            (b"apcs", ProResProfile::Lt),
            (b"apcn", ProResProfile::Standard),
            (b"apch", ProResProfile::Hq),
            (b"ap4h", ProResProfile::P4444),
            (b"ap4x", ProResProfile::P4444Xq),
        ];
        for (fcc, expected) in cases {
            let got = ProResProfile::from_fourcc(fcc).expect("known FourCC");
            assert_eq!(got, *expected, "fourcc {:?}", fcc);
        }
        // Unknown FourCC must error.
        assert!(ProResProfile::from_fourcc(b"xxxx").is_err());
    }

    #[test]
    fn decode_rejects_profile_mismatch() {
        // Encode with Standard profile, then attempt to decode expecting Proxy.
        let frame = flat_grey_frame(32, 16, 400);
        let pkt = encode_frame(&frame);

        let config = ProResDecoderConfig {
            profile: Some(ProResProfile::Proxy),
        };
        let dec = ProResDecoder::with_config(config);
        let result = dec.decode_with_config(&pkt);
        assert!(
            result.is_err(),
            "should reject Standard-encoded stream when Proxy expected"
        );
    }

    #[test]
    fn encode_decode_constant_grey() {
        // Encode a 64×16 constant-luma frame and decode it.
        // Luma value = 400 (10-bit) → expected 8-bit output ≈ 400 >> 2 = 100.
        let y_val_10bit: u16 = 400;
        let frame = flat_grey_frame(64, 16, y_val_10bit);
        let pkt = encode_frame(&frame);

        let decoded = ProResDecoder::decode(&pkt).expect("decode");

        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 16);
        assert_eq!(decoded.profile, ProResProfile::Standard);
        assert!(!decoded.is_interlaced);
        assert_eq!(decoded.y.len(), 64 * 16);
        assert_eq!(decoded.cb.len(), 32 * 16);
        assert_eq!(decoded.cr.len(), 32 * 16);

        // Verify luma values within ±4 LSB of the expected 8-bit value.
        let expected_y8 = (y_val_10bit >> 2) as i32;
        let tolerance = 4i32;
        for (i, &sample) in decoded.y.iter().enumerate() {
            let err = (sample as i32 - expected_y8).abs();
            assert!(
                err <= tolerance,
                "luma sample {} = {} deviates from expected {} by {} (tolerance {})",
                i,
                sample,
                expected_y8,
                err,
                tolerance
            );
        }
    }

    #[test]
    fn video_decoder_trait_send_receive() {
        let frame = flat_grey_frame(32, 16, 512);
        let pkt = encode_frame(&frame);

        let mut dec = ProResDecoder::new();
        dec.send_packet(&pkt, 42).expect("send_packet");
        let vf = dec.receive_frame().expect("receive_frame").expect("Some");
        assert_eq!(vf.timestamp.pts, 42);
        assert_eq!(vf.width, 32);
        assert_eq!(vf.height, 16);
        assert_eq!(vf.format, PixelFormat::Yuv422p);
        // Queue is now drained.
        assert!(dec.receive_frame().expect("second receive").is_none());
    }

    #[test]
    fn flush_is_noop() {
        let mut dec = ProResDecoder::new();
        dec.flush().expect("flush should not error");
    }

    #[test]
    fn reset_clears_queue() {
        let frame = flat_grey_frame(32, 16, 512);
        let pkt = encode_frame(&frame);

        let mut dec = ProResDecoder::new();
        dec.send_packet(&pkt, 0).expect("send");
        dec.reset();
        // After reset, queue must be empty.
        assert!(dec.receive_frame().expect("receive after reset").is_none());
    }

    #[test]
    fn output_format_is_yuv422p() {
        let dec = ProResDecoder::new();
        assert_eq!(dec.output_format(), Some(PixelFormat::Yuv422p));
    }

    #[test]
    fn codec_id_is_prores() {
        let dec = ProResDecoder::new();
        assert_eq!(dec.codec(), CodecId::ProRes);
    }
}
