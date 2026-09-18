//! MPEG-2 video **I-frame** encoder (ISO/IEC 13818-2 / ITU-T H.262).
//!
//! The encoder-side counterpart to [`super::decode::Mpeg2Decoder`]. It produces
//! an elementary-stream byte buffer that the existing decoder reconstructs to
//! within the quantiser/IDCT tolerance.
//!
//! # Scope (Wave 10)
//!
//! - **I-frames only** (`picture_coding_type == 1`, every macroblock intra).
//! - **4:2:0**, **4:2:2** *and* **4:4:4** chroma formats (the configured
//!   [`Mpeg2EncoderConfig::chroma_format`] is written to `sequence_extension`).
//! - **Progressive frame** pictures (`picture_structure == 3`).
//!
//! P/B frames and field pictures remain out of scope.
//!
//! # Pipeline
//!
//! ```text
//! YUV planes (Yuv420p / Yuv422p / Yuv444p depending on cfg.chroma_format)
//!    │  pad luma/chroma up to a 16×16 / per-format chroma macroblock grid
//!    ▼  emit sequence_header + sequence_extension(chroma_format)
//!    ▼  emit picture_header (I) + picture_coding_extension
//!    ▼  for each macroblock row → one slice:
//!         emit slice header (quantiser_scale_code)
//!         for each macroblock (address_increment = 1, type = Intra):
//!           for each block in the 6/8/12-entry list:
//!             FDCT → forward intra quant → zig-zag scan
//!             DC DPCM differential + AC run/level → VLC
//!    ▼  emit sequence_end_code (0xB7)
//! ```
//!
//! The DC predictor reset is per-component and shared across all chroma blocks
//! of a macroblock — exactly as on the decoder side. The absence of a spatial
//! level shift, the FDCT/IDCT mismatch, and the VLC table inversion are all
//! matched bit-for-bit to the decoder.

use crate::error::{CodecError, CodecResult};
use crate::frame::VideoFrame;
use crate::traits::{EncodedPacket, EncoderConfig, VideoEncoder};
use oximedia_core::{CodecId, PixelFormat};

use super::bitreader::{
    EXTENSION_START_CODE, PICTURE_START_CODE, SEQUENCE_END_CODE, SEQUENCE_HEADER_CODE,
};
use super::bitwriter::BitWriter;
use super::dequant::{quantiser_scale, DEFAULT_INTRA_MATRIX};
use super::fdct::fdct_8x8;
use super::marker_write::{
    write_picture_coding_extension, write_picture_header, write_sequence_extension,
    write_sequence_header, write_slice_header, PictureCodingExtensionParams,
    SequenceExtensionParams, SequenceHeaderParams,
};
use super::quantize_fwd::quantize_intra;
use super::vlc_encode::{encode_ac_run_level, encode_dc, encode_eob};
use super::vlc_tables::{AC_TABLE_B14, DC_SIZE_CHROMA, DC_SIZE_LUMA};
use super::zigzag::scan_table;
use super::Mpeg2Error;
use super::Mpeg2Result;

/// Largest macroblock-row count addressable by a slice start code
/// (`slice_start_code` spans `0x01..=0xAF`, i.e. up to 175 rows ⇒ 2800 lines).
const MAX_SLICE_ROWS: usize = 0xAF;

/// Configuration for the MPEG-2 I-frame encoder.
#[derive(Debug, Clone, Copy)]
pub struct Mpeg2EncoderConfig {
    /// Luminance width in pixels (any positive value; padded to a 16-grid).
    pub width: u32,
    /// Luminance height in pixels (any positive value; padded to a 16-grid).
    pub height: u32,
    /// `quantiser_scale_code` in `1..=31` (lower = higher quality).
    pub qscale: u8,
    /// `intra_dc_precision` in `0..=3` (8/9/10/11-bit DC).
    pub intra_dc_precision: u8,
    /// `frame_rate_code` (4 bits) written in the sequence header. 3 = 25 fps.
    pub frame_rate: u8,
    /// `aspect_ratio_information` (4 bits) written in the sequence header.
    pub aspect_ratio: u8,
    /// `chroma_format` (2 bits): 1 = 4:2:0 (6 blocks/MB), 2 = 4:2:2 (8),
    /// 3 = 4:4:4 (12). Defaults to `1` for backward compatibility.
    pub chroma_format: u8,
}

impl Default for Mpeg2EncoderConfig {
    fn default() -> Self {
        Self {
            width: 16,
            height: 16,
            qscale: 8,
            intra_dc_precision: 0,
            frame_rate: 3,
            aspect_ratio: 1,
            chroma_format: 1,
        }
    }
}

impl Mpeg2EncoderConfig {
    /// Create a 4:2:0 configuration for `width × height` at the given `qscale`.
    #[must_use]
    pub fn new(width: u32, height: u32, qscale: u8) -> Self {
        Self {
            width,
            height,
            qscale,
            ..Self::default()
        }
    }

    /// Shortcut for a 4:2:0 (Yuv420p) configuration.
    #[must_use]
    pub fn yuv420p(width: u32, height: u32, qscale: u8) -> Self {
        Self {
            chroma_format: 1,
            ..Self::new(width, height, qscale)
        }
    }

    /// Shortcut for a 4:2:2 (Yuv422p) configuration.
    #[must_use]
    pub fn yuv422p(width: u32, height: u32, qscale: u8) -> Self {
        Self {
            chroma_format: 2,
            ..Self::new(width, height, qscale)
        }
    }

    /// Shortcut for a 4:4:4 (Yuv444p) configuration.
    #[must_use]
    pub fn yuv444p(width: u32, height: u32, qscale: u8) -> Self {
        Self {
            chroma_format: 3,
            ..Self::new(width, height, qscale)
        }
    }

    /// Map `chroma_format` (1/2/3) to the matching planar `PixelFormat`.
    #[must_use]
    pub fn pixel_format(&self) -> PixelFormat {
        match self.chroma_format {
            2 => PixelFormat::Yuv422p,
            3 => PixelFormat::Yuv444p,
            _ => PixelFormat::Yuv420p,
        }
    }

    /// Chroma plane width given the (padded) luma width.
    fn chroma_w(&self, luma_w: usize) -> usize {
        if self.chroma_format == 3 {
            luma_w
        } else {
            luma_w.div_ceil(2)
        }
    }

    /// Chroma plane height given the (padded) luma height.
    fn chroma_h(&self, luma_h: usize) -> usize {
        if self.chroma_format == 1 {
            luma_h.div_ceil(2)
        } else {
            luma_h
        }
    }

    /// Number of chroma blocks per component, per macroblock (1, 2, or 4).
    fn chroma_blocks_per_mb(&self) -> usize {
        match self.chroma_format {
            2 => 2,
            3 => 4,
            _ => 1, // 4:2:0
        }
    }

    /// Validate the configuration.
    fn validate(&self) -> Mpeg2Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(Mpeg2Error::InvalidConfig(
                "frame dimensions must be non-zero".into(),
            ));
        }
        if self.width > 0xFFFF || self.height > 0xFFFF {
            return Err(Mpeg2Error::InvalidConfig(format!(
                "dimensions {}×{} exceed the 14-bit MPEG-2 size limit",
                self.width, self.height
            )));
        }
        if !(1..=31).contains(&self.qscale) {
            return Err(Mpeg2Error::InvalidConfig(format!(
                "qscale {} out of range 1..=31",
                self.qscale
            )));
        }
        if self.intra_dc_precision > 3 {
            return Err(Mpeg2Error::InvalidConfig(format!(
                "intra_dc_precision {} out of range 0..=3",
                self.intra_dc_precision
            )));
        }
        if !(1..=3).contains(&self.chroma_format) {
            return Err(Mpeg2Error::InvalidConfig(format!(
                "chroma_format {} out of range 1..=3",
                self.chroma_format
            )));
        }
        let mb_rows = (self.height as usize).div_ceil(16);
        if mb_rows > MAX_SLICE_ROWS {
            return Err(Mpeg2Error::InvalidConfig(format!(
                "height {} needs {mb_rows} macroblock rows (> {MAX_SLICE_ROWS})",
                self.height
            )));
        }
        Ok(())
    }
}

/// MPEG-2 I-frame encoder (intra, 4:2:0 / 4:2:2 / 4:4:4, progressive).
#[derive(Debug)]
pub struct Mpeg2Encoder {
    config: Mpeg2EncoderConfig,
    enc_config: EncoderConfig,
    frame_count: u64,
    output_queue: Vec<EncodedPacket>,
}

impl Mpeg2Encoder {
    /// Create a new encoder from a [`Mpeg2EncoderConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`Mpeg2Error::InvalidConfig`] if the configuration is invalid.
    pub fn new(config: Mpeg2EncoderConfig) -> Mpeg2Result<Self> {
        config.validate()?;
        let enc_config = EncoderConfig {
            codec: CodecId::Mpeg2,
            width: config.width,
            height: config.height,
            pixel_format: config.pixel_format(),
            ..EncoderConfig::default()
        };
        Ok(Self {
            config,
            enc_config,
            frame_count: 0,
            output_queue: Vec::new(),
        })
    }

    /// Borrow the encoder configuration.
    #[must_use]
    pub fn config(&self) -> &Mpeg2EncoderConfig {
        &self.config
    }

    /// Encode one YUV frame from raw planes into an MPEG-2 elementary stream.
    ///
    /// - `y` is `width × height` bytes (raster order).
    /// - `cb` and `cr` are each `chroma_w × chroma_h` bytes, where the chroma
    ///   plane dimensions depend on `cfg.chroma_format`:
    ///   - 4:2:0 → `ceil(width/2) × ceil(height/2)`
    ///   - 4:2:2 → `ceil(width/2) × height`
    ///   - 4:4:4 → `width × height`
    ///
    /// # Errors
    ///
    /// Returns [`Mpeg2Error::Encode`] if a plane is too small for the configured
    /// dimensions, or a VLC/header write fails.
    pub fn encode_planes(&self, y: &[u8], cb: &[u8], cr: &[u8]) -> Mpeg2Result<Vec<u8>> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let chroma_w = self.config.chroma_w(width);
        let chroma_h = self.config.chroma_h(height);

        if y.len() < width * height {
            return Err(Mpeg2Error::Encode(format!(
                "luma plane too small: have {}, need {}",
                y.len(),
                width * height
            )));
        }
        if cb.len() < chroma_w * chroma_h || cr.len() < chroma_w * chroma_h {
            return Err(Mpeg2Error::Encode(format!(
                "chroma plane too small: have cb={} cr={}, need {}",
                cb.len(),
                cr.len(),
                chroma_w * chroma_h
            )));
        }

        let plane = SourcePlanes {
            y,
            cb,
            cr,
            width,
            height,
            chroma_w,
            chroma_h,
        };
        self.encode_planes_inner(&plane)
    }

    /// Internal encode taking a bundled [`SourcePlanes`] view.
    fn encode_planes_inner(&self, planes: &SourcePlanes<'_>) -> Mpeg2Result<Vec<u8>> {
        let cfg = &self.config;
        let mb_cols = planes.width.div_ceil(16);
        let mb_rows = planes.height.div_ceil(16);

        let mut writer = BitWriter::with_capacity(planes.width * planes.height);

        // ── Sequence header + extension ─────────────────────────────────────
        let hsize_ext = ((cfg.width >> 12) & 0x3) as u8;
        let vsize_ext = ((cfg.height >> 12) & 0x3) as u8;

        writer.write_start_code(SEQUENCE_HEADER_CODE);
        write_sequence_header(
            &mut writer,
            &SequenceHeaderParams {
                width: cfg.width,
                height: cfg.height,
                aspect_ratio_information: cfg.aspect_ratio,
                frame_rate_code: cfg.frame_rate,
                bit_rate_value: 0x3_FFFF,
                vbv_buffer_size_value: 112,
                load_default_matrices: false,
            },
        );

        writer.write_start_code(EXTENSION_START_CODE);
        write_sequence_extension(
            &mut writer,
            &SequenceExtensionParams {
                profile_and_level_indication: 0x44,
                progressive_sequence: true,
                chroma_format: cfg.chroma_format,
                horizontal_size_extension: hsize_ext,
                vertical_size_extension: vsize_ext,
                bit_rate_extension: 0,
                frame_rate_extension_n: 0,
                frame_rate_extension_d: 0,
            },
        );

        // ── Picture header + coding extension ───────────────────────────────
        writer.write_start_code(PICTURE_START_CODE);
        write_picture_header(&mut writer, 0, 0xFFFF);

        writer.write_start_code(EXTENSION_START_CODE);
        write_picture_coding_extension(
            &mut writer,
            &PictureCodingExtensionParams {
                intra_dc_precision: cfg.intra_dc_precision,
                q_scale_type: false,
                intra_vlc_format: false,
                alternate_scan: false,
                progressive_frame: true,
            },
        );

        // ── One slice per macroblock row ────────────────────────────────────
        let q_scale = quantiser_scale(cfg.qscale, false);
        let ac_table = AC_TABLE_B14; // intra_vlc_format == 0
        let scan = scan_table(false); // alternate_scan == 0

        for mb_row in 0..mb_rows {
            // slice_start_code == slice_vertical_position == mb_row + 1.
            let slice_code = (mb_row + 1) as u8;
            writer.write_start_code(slice_code);
            write_slice_header(&mut writer, cfg.qscale);

            // DC predictors reset at the start of every slice.
            let mut dc_pred = DcPredictorsFwd::reset(cfg.intra_dc_precision);

            for mb_col in 0..mb_cols {
                // macroblock_address_increment = 1 (code `1`).
                writer.write_bit(true);
                // macroblock_type = Intra, no quant change (code `1`).
                writer.write_bit(true);

                self.encode_macroblock(
                    &mut writer,
                    planes,
                    mb_row,
                    mb_col,
                    q_scale,
                    ac_table,
                    scan,
                    &mut dc_pred,
                )?;
            }
        }

        // ── Sequence end ────────────────────────────────────────────────────
        writer.write_start_code(SEQUENCE_END_CODE);

        Ok(writer.into_bytes())
    }

    /// Encode the 6 / 8 / 12 8×8 blocks of one macroblock (luma + chroma,
    /// count depending on `cfg.chroma_format`).
    ///
    /// Block ordering (ISO/IEC 13818-2 §6.1.1.4 Table 6-10):
    /// - 4:2:0 → Y0, Y1, Y2, Y3, Cb, Cr
    /// - 4:2:2 → Y0, Y1, Y2, Y3, Cb_top, Cb_bot, Cr_top, Cr_bot
    /// - 4:4:4 → Y0..Y3, Cb0..Cb3, Cr0..Cr3 (chroma in 2×2 raster)
    ///
    /// The DC predictors are kept per-component (Y / Cb / Cr); the same Cb
    /// predictor is shared across all Cb blocks of a macroblock (same for Cr).
    #[allow(clippy::too_many_arguments)]
    fn encode_macroblock(
        &self,
        writer: &mut BitWriter,
        planes: &SourcePlanes<'_>,
        mb_row: usize,
        mb_col: usize,
        q_scale: i32,
        ac_table: super::vlc_tables::AcTablePtr,
        scan: &[usize; 64],
        dc_pred: &mut DcPredictorsFwd,
    ) -> Mpeg2Result<()> {
        let chroma_format = self.config.chroma_format;
        let chroma_blocks = self.config.chroma_blocks_per_mb();

        // ── Four luma blocks arranged 2×2 within the 16×16 macroblock ──────
        for blk in 0..4usize {
            let block_x = (blk & 1) * 8;
            let block_y = (blk >> 1) * 8;
            let origin_x = mb_col * 16 + block_x;
            let origin_y = mb_row * 16 + block_y;
            let samples = gather_block(planes.y, planes.width, planes.height, origin_x, origin_y);
            self.encode_block(
                writer,
                &samples,
                q_scale,
                DC_SIZE_LUMA,
                ac_table,
                scan,
                &mut dc_pred.y,
            )?;
        }

        // ── Chroma blocks: 1 / 2 / 4 per component depending on chroma_format ──
        // Cb first, then Cr (matches the decoder's block_list ordering).
        for sub in 0..chroma_blocks {
            let (cx, cy) = encoder_chroma_origin(mb_col, mb_row, sub, chroma_format);
            let cb_samples = gather_block(planes.cb, planes.chroma_w, planes.chroma_h, cx, cy);
            self.encode_block(
                writer,
                &cb_samples,
                q_scale,
                DC_SIZE_CHROMA,
                ac_table,
                scan,
                &mut dc_pred.cb,
            )?;
        }
        for sub in 0..chroma_blocks {
            let (cx, cy) = encoder_chroma_origin(mb_col, mb_row, sub, chroma_format);
            let cr_samples = gather_block(planes.cr, planes.chroma_w, planes.chroma_h, cx, cy);
            self.encode_block(
                writer,
                &cr_samples,
                q_scale,
                DC_SIZE_CHROMA,
                ac_table,
                scan,
                &mut dc_pred.cr,
            )?;
        }

        Ok(())
    }

    /// Encode one 8×8 block: FDCT → forward quant → DC DPCM + AC run/level VLC.
    #[allow(clippy::too_many_arguments)]
    fn encode_block(
        &self,
        writer: &mut BitWriter,
        samples: &[i32; 64],
        q_scale: i32,
        dc_table: super::vlc_tables::DcTablePtr,
        ac_table: super::vlc_tables::AcTablePtr,
        scan: &[usize; 64],
        dc_predictor: &mut i32,
    ) -> Mpeg2Result<()> {
        let freq = fdct_8x8(samples);
        let qf = quantize_intra(
            &freq,
            &DEFAULT_INTRA_MATRIX,
            self.config.intra_dc_precision,
            q_scale,
        );

        // DC DPCM differential against the running predictor.
        let dc = qf[0];
        let diff = dc - *dc_predictor;
        *dc_predictor = dc;
        encode_dc(writer, dc_table, diff)?;

        // AC run/level in scan order (scan index 1..=63).
        encode_ac_coefficients(writer, &qf, scan, ac_table)?;

        Ok(())
    }

    /// Drive [`encode_planes`](Self::encode_planes) from a [`VideoFrame`],
    /// respecting per-plane stride. Accepts any of the three planar formats
    /// (`Yuv420p` / `Yuv422p` / `Yuv444p`) provided the frame's format matches
    /// the encoder's configured `chroma_format`.
    fn encode_video_frame(&self, frame: &VideoFrame) -> CodecResult<Vec<u8>> {
        let expected = self.config.pixel_format();
        if frame.format != expected {
            return Err(CodecError::InvalidParameter(format!(
                "MPEG-2 encoder: expected {expected:?}, got {:?}",
                frame.format
            )));
        }
        if frame.width != self.config.width || frame.height != self.config.height {
            return Err(CodecError::InvalidParameter(format!(
                "MPEG-2 encoder: frame {}×{} != configured {}×{}",
                frame.width, frame.height, self.config.width, self.config.height
            )));
        }
        if frame.planes.len() < 3 {
            return Err(CodecError::InvalidParameter(format!(
                "MPEG-2 encoder: {expected:?} frame needs 3 planes"
            )));
        }

        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let chroma_w = self.config.chroma_w(width);
        let chroma_h = self.config.chroma_h(height);

        // Repack each plane to a contiguous, stride-free buffer.
        let y = repack_plane(&frame.planes[0], width, height);
        let cb = repack_plane(&frame.planes[1], chroma_w, chroma_h);
        let cr = repack_plane(&frame.planes[2], chroma_w, chroma_h);

        self.encode_planes(&y, &cb, &cr)
            .map_err(|e| CodecError::Internal(e.to_string()))
    }
}

/// Compute the `(origin_x, origin_y)` of a chroma 8×8 block within its plane,
/// given the macroblock coordinates, chroma sub-index and chroma format.
///
/// Mirror of the decoder-side helper.
fn encoder_chroma_origin(
    mb_col: usize,
    mb_row: usize,
    sub_index: usize,
    chroma_format: u8,
) -> (usize, usize) {
    match chroma_format {
        2 => {
            // 4:2:2: chroma 8×16 per MB, two stacked 8×8.
            (mb_col * 8, mb_row * 16 + sub_index * 8)
        }
        3 => {
            // 4:4:4: chroma 16×16 per MB, four 8×8 in 2×2 raster.
            let block_x = (sub_index & 1) * 8;
            let block_y = (sub_index >> 1) * 8;
            (mb_col * 16 + block_x, mb_row * 16 + block_y)
        }
        _ => {
            // 4:2:0: one 8×8 per MB covering 8×8 of the chroma plane.
            (mb_col * 8, mb_row * 8)
        }
    }
}

/// Bundled borrowed plane views for one frame.
struct SourcePlanes<'a> {
    y: &'a [u8],
    cb: &'a [u8],
    cr: &'a [u8],
    width: usize,
    height: usize,
    chroma_w: usize,
    chroma_h: usize,
}

/// Forward DC predictors (one per component), reset per slice.
struct DcPredictorsFwd {
    y: i32,
    cb: i32,
    cr: i32,
}

impl DcPredictorsFwd {
    /// Reset to `1 << (7 + intra_dc_precision)`, matching the decoder.
    fn reset(intra_dc_precision: u8) -> Self {
        let v = 1i32 << (7 + i32::from(intra_dc_precision & 0x3));
        Self { y: v, cb: v, cr: v }
    }
}

/// Gather an 8×8 block from `plane` at `(origin_x, origin_y)`, clamping reads to
/// the plane edge (so partial macroblocks at the right/bottom edge replicate the
/// last valid sample). Returns signed `i32` samples (raw 0..=255 values).
fn gather_block(
    plane: &[u8],
    plane_w: usize,
    plane_h: usize,
    origin_x: usize,
    origin_y: usize,
) -> [i32; 64] {
    let mut out = [0i32; 64];
    for r in 0..8 {
        let sy = (origin_y + r).min(plane_h.saturating_sub(1));
        for c in 0..8 {
            let sx = (origin_x + c).min(plane_w.saturating_sub(1));
            let idx = sy * plane_w + sx;
            let sample = plane.get(idx).copied().unwrap_or(128);
            out[r * 8 + c] = i32::from(sample);
        }
    }
    out
}

/// Encode the AC coefficients of one quantised block in scan order.
///
/// Walks scan indices `1..=63`; for each non-zero coefficient emits the run of
/// preceding zeros together with the level. Terminates with an EOB code unless
/// the final coefficient sits at scan index 63 (where the decoder stops on its
/// own without consuming an EOB).
fn encode_ac_coefficients(
    writer: &mut BitWriter,
    qf: &[i32; 64],
    scan: &[usize; 64],
    ac_table: super::vlc_tables::AcTablePtr,
) -> Mpeg2Result<()> {
    // Find the highest scan index holding a non-zero AC coefficient.
    let mut last_nz: Option<usize> = None;
    for scan_index in 1..64 {
        if qf[scan[scan_index]] != 0 {
            last_nz = Some(scan_index);
        }
    }

    let Some(last_nz) = last_nz else {
        // No AC energy → emit EOB immediately.
        return encode_eob(writer, ac_table);
    };

    let mut run: u8 = 0;
    for scan_index in 1..=last_nz {
        let level = qf[scan[scan_index]];
        if level == 0 {
            run += 1;
            continue;
        }
        encode_ac_run_level(writer, ac_table, run, level)?;
        run = 0;
    }

    // If the last non-zero coefficient is not at the very end, the decoder will
    // read an EOB to know the block terminated early.
    if last_nz < 63 {
        encode_eob(writer, ac_table)?;
    }
    Ok(())
}

/// Copy a (possibly strided) plane into a contiguous `width × height` buffer,
/// padding missing rows/columns with mid-grey 128.
fn repack_plane(plane: &crate::frame::Plane, width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![128u8; width * height];
    let stride = plane.stride.max(width);
    for row in 0..height {
        let src_off = row * stride;
        let dst_off = row * width;
        let copy = width.min(plane.data.len().saturating_sub(src_off));
        if copy > 0 {
            out[dst_off..dst_off + copy].copy_from_slice(&plane.data[src_off..src_off + copy]);
        }
    }
    out
}

impl VideoEncoder for Mpeg2Encoder {
    fn codec(&self) -> CodecId {
        CodecId::Mpeg2
    }

    fn send_frame(&mut self, frame: &VideoFrame) -> CodecResult<()> {
        let data = self.encode_video_frame(frame)?;
        let pts = self.frame_count as i64;
        self.output_queue.push(EncodedPacket {
            data,
            pts,
            dts: pts,
            keyframe: true,
            duration: Some(1),
        });
        self.frame_count += 1;
        Ok(())
    }

    fn receive_packet(&mut self) -> CodecResult<Option<EncodedPacket>> {
        if self.output_queue.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.output_queue.remove(0)))
        }
    }

    fn flush(&mut self) -> CodecResult<()> {
        // Intra-only: nothing buffered for reordering.
        Ok(())
    }

    fn config(&self) -> &EncoderConfig {
        &self.enc_config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validation() {
        assert!(Mpeg2Encoder::new(Mpeg2EncoderConfig::new(16, 16, 8)).is_ok());
        assert!(Mpeg2Encoder::new(Mpeg2EncoderConfig::new(0, 16, 8)).is_err());
        assert!(Mpeg2Encoder::new(Mpeg2EncoderConfig::new(16, 16, 0)).is_err());
        assert!(Mpeg2Encoder::new(Mpeg2EncoderConfig::new(16, 16, 32)).is_err());
        let mut bad = Mpeg2EncoderConfig::new(16, 16, 8);
        bad.intra_dc_precision = 4;
        assert!(Mpeg2Encoder::new(bad).is_err());
        // chroma_format outside 1..=3 → reject.
        let mut bad_cf = Mpeg2EncoderConfig::new(16, 16, 8);
        bad_cf.chroma_format = 0;
        assert!(Mpeg2Encoder::new(bad_cf).is_err());
        let mut bad_cf2 = Mpeg2EncoderConfig::new(16, 16, 8);
        bad_cf2.chroma_format = 4;
        assert!(Mpeg2Encoder::new(bad_cf2).is_err());
    }

    #[test]
    fn factory_shortcuts_pick_correct_pixel_format() {
        let c420 = Mpeg2EncoderConfig::yuv420p(16, 16, 4);
        assert_eq!(c420.chroma_format, 1);
        assert_eq!(c420.pixel_format(), PixelFormat::Yuv420p);
        let c422 = Mpeg2EncoderConfig::yuv422p(16, 16, 4);
        assert_eq!(c422.chroma_format, 2);
        assert_eq!(c422.pixel_format(), PixelFormat::Yuv422p);
        let c444 = Mpeg2EncoderConfig::yuv444p(16, 16, 4);
        assert_eq!(c444.chroma_format, 3);
        assert_eq!(c444.pixel_format(), PixelFormat::Yuv444p);
    }

    #[test]
    fn chroma_plane_dims_match_format() {
        let c420 = Mpeg2EncoderConfig::yuv420p(32, 16, 4);
        assert_eq!((c420.chroma_w(32), c420.chroma_h(16)), (16, 8));
        let c422 = Mpeg2EncoderConfig::yuv422p(32, 16, 4);
        assert_eq!((c422.chroma_w(32), c422.chroma_h(16)), (16, 16));
        let c444 = Mpeg2EncoderConfig::yuv444p(32, 16, 4);
        assert_eq!((c444.chroma_w(32), c444.chroma_h(16)), (32, 16));
    }

    #[test]
    fn encoder_chroma_origin_dispatches() {
        // 4:2:0: one 8×8 per MB, always at (mb_col*8, mb_row*8).
        assert_eq!(encoder_chroma_origin(2, 3, 0, 1), (16, 24));
        // 4:2:2: two stacked 8×8 per MB.
        assert_eq!(encoder_chroma_origin(2, 3, 0, 2), (16, 48));
        assert_eq!(encoder_chroma_origin(2, 3, 1, 2), (16, 56));
        // 4:4:4: 2×2 raster of 8×8 per MB.
        assert_eq!(encoder_chroma_origin(2, 3, 0, 3), (32, 48));
        assert_eq!(encoder_chroma_origin(2, 3, 1, 3), (40, 48));
        assert_eq!(encoder_chroma_origin(2, 3, 2, 3), (32, 56));
        assert_eq!(encoder_chroma_origin(2, 3, 3, 3), (40, 56));
    }

    #[test]
    fn gather_block_clamps_to_edge() {
        // 2×2 plane, gather an 8×8 block at origin (0,0): all reads clamp.
        let plane = vec![10u8, 20, 30, 40];
        let block = gather_block(&plane, 2, 2, 0, 0);
        // Top-left is plane[0]=10, bottom-right replicates plane[3]=40.
        assert_eq!(block[0], 10);
        assert_eq!(block[63], 40);
    }

    #[test]
    fn encode_planes_too_small_errors() {
        let enc = Mpeg2Encoder::new(Mpeg2EncoderConfig::new(16, 16, 8)).expect("enc");
        let small = vec![0u8; 10];
        assert!(matches!(
            enc.encode_planes(&small, &small, &small),
            Err(Mpeg2Error::Encode(_))
        ));
    }

    #[test]
    fn encode_grey_frame_produces_stream() {
        let enc = Mpeg2Encoder::new(Mpeg2EncoderConfig::new(16, 16, 4)).expect("enc");
        let y = vec![128u8; 16 * 16];
        let c = vec![128u8; 8 * 8];
        let stream = enc.encode_planes(&y, &c, &c).expect("encode");
        // Must begin with the sequence header start code.
        assert_eq!(&stream[0..4], &[0x00, 0x00, 0x01, SEQUENCE_HEADER_CODE]);
        // Must end with the sequence end code.
        let n = stream.len();
        assert_eq!(&stream[n - 4..], &[0x00, 0x00, 0x01, SEQUENCE_END_CODE]);
    }

    #[test]
    fn dc_predictors_reset_matches_decoder() {
        assert_eq!(DcPredictorsFwd::reset(0).y, 128);
        assert_eq!(DcPredictorsFwd::reset(1).y, 256);
        assert_eq!(DcPredictorsFwd::reset(2).y, 512);
        assert_eq!(DcPredictorsFwd::reset(3).y, 1024);
    }

    #[test]
    fn encode_ac_all_zero_emits_eob_only() {
        let mut w = BitWriter::new();
        let qf = [0i32; 64];
        let scan = scan_table(false);
        encode_ac_coefficients(&mut w, &qf, scan, AC_TABLE_B14).expect("ac");
        // B-14 EOB is `10` (2 bits) → first byte starts 0b10....
        let bytes = w.into_bytes();
        assert_eq!(bytes[0] >> 6, 0b10);
    }
}
