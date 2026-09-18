//! ProRes 422 progressive encoder.
//!
//! Accepts `Yuv422p10le` input frames and produces ProRes 422 `'icpf'`
//! compressed frames. Each input frame encodes as a complete intra-frame
//! (no inter-frame prediction — ProRes is always intra).
//!
//! ## Usage
//!
//! ```ignore
//! use oximedia_codec::prores::{ProResEncoder, ProResEncoderConfig};
//! use oximedia_codec::prores::frame::ProResProfile;
//! use oximedia_codec::traits::VideoEncoder;
//!
//! let cfg = ProResEncoderConfig {
//!     profile: ProResProfile::Standard,
//!     width: 1920,
//!     height: 1080,
//!     qscale: 0, // auto
//!     log2_slice_mb_width: 3, // 8 MBs per slice
//! };
//! let mut encoder = ProResEncoder::new(cfg)?;
//! encoder.send_frame(&yuv_frame)?;
//! if let Some(pkt) = encoder.receive_packet()? {
//!     // pkt.data is the complete 'icpf' frame bytes
//! }
//! ```

use crate::error::{CodecError, CodecResult};
use crate::frame::VideoFrame;
use crate::traits::{EncodedPacket, EncoderConfig, VideoEncoder};
use oximedia_core::{CodecId, PixelFormat};

use super::encode::{encode_slice, encode_slice_444};
use super::frame::{ChromaFormat, InterlaceMode, ProResProfile};
use super::frame_write::write_frame;
use super::quant::{DEFAULT_CHROMA_QUANT_MATRIX, DEFAULT_LUMA_QUANT_MATRIX};

/// ProRes encoder configuration.
#[derive(Debug, Clone)]
pub struct ProResEncoderConfig {
    /// ProRes profile (Proxy / LT / Standard / HQ / 4444 / 4444 XQ).
    pub profile: ProResProfile,
    /// Frame width in luma samples. Must be a multiple of 16.
    pub width: u32,
    /// Frame height in luma samples. Must be a multiple of 16.
    pub height: u32,
    /// Per-slice quantization scale. `0` = use profile default.
    /// Valid range: 1..=224. Higher value = more compression, lower quality.
    pub qscale: u8,
    /// Log2 of macroblocks per slice row. Default 3 = 8 MB wide slices.
    /// `1 << log2_slice_mb_width` is the number of MBs per horizontal slice.
    pub log2_slice_mb_width: u8,
    /// Frame rate code per RDD 36 Table 6.
    /// 0x00 = unspecified, 0x05 = 30 fps, 0x06 = 25 fps, 0x03 ≈ 29.97 fps.
    pub frame_rate_code: u8,
    /// Chroma subsampling. [`ChromaFormat::Yuv422`] (the default) expects
    /// `Yuv422p10le` input; [`ChromaFormat::Yuv444`] expects `Yuv444p10le`
    /// input and emits a full-resolution-chroma stream. The 4:4:4 path emits
    /// no alpha plane.
    pub chroma_format: ChromaFormat,
}

impl ProResEncoderConfig {
    /// Create a configuration with profile-appropriate default qscale.
    ///
    /// The chroma format defaults to 4:2:2. Use [`ProResEncoderConfig::yuv444`]
    /// for a 4:4:4 stream.
    #[must_use]
    pub fn new(profile: ProResProfile, width: u32, height: u32) -> Self {
        Self {
            profile,
            width,
            height,
            qscale: 0, // auto
            log2_slice_mb_width: 3,
            frame_rate_code: 0,
            chroma_format: ChromaFormat::Yuv422,
        }
    }

    /// Create a 4:4:4 configuration (ProRes 4444 / 4444 XQ).
    ///
    /// The encoder will expect `Yuv444p10le` input frames and produce a
    /// full-resolution-chroma `'ap4h'` / `'ap4x'` stream.
    #[must_use]
    pub fn yuv444(profile: ProResProfile, width: u32, height: u32) -> Self {
        Self {
            chroma_format: ChromaFormat::Yuv444,
            ..Self::new(profile, width, height)
        }
    }

    /// Effective qscale: uses the profile's default when `qscale == 0`.
    #[must_use]
    pub fn effective_qscale(&self) -> u8 {
        if self.qscale != 0 {
            return self.qscale;
        }
        // FFmpeg reference defaults for ProRes 422 profiles.
        match self.profile {
            ProResProfile::Proxy => 16,
            ProResProfile::Lt => 9,
            ProResProfile::Standard => 6,
            ProResProfile::Hq => 4,
            ProResProfile::P4444 | ProResProfile::P4444Xq => 4,
        }
    }
}

/// ProRes 422 progressive encoder.
pub struct ProResEncoder {
    config: ProResEncoderConfig,
    enc_config: EncoderConfig,
    frame_count: u64,
    output_queue: Vec<EncodedPacket>,
}

impl ProResEncoder {
    /// Create a new ProRes encoder from a [`ProResEncoderConfig`].
    pub fn new(config: ProResEncoderConfig) -> CodecResult<Self> {
        if config.width == 0 || config.height == 0 {
            return Err(CodecError::InvalidParameter(
                "ProRes encoder: frame dimensions must be non-zero".to_string(),
            ));
        }
        if config.width % 16 != 0 || config.height % 16 != 0 {
            return Err(CodecError::InvalidParameter(format!(
                "ProRes encoder: width ({}) and height ({}) must be multiples of 16",
                config.width, config.height,
            )));
        }
        // The chroma format must match the profile family: the two 4444
        // profiles ('ap4h'/'ap4x') are 4:4:4; the four 422 profiles are 4:2:2.
        match config.chroma_format {
            ChromaFormat::Yuv444 if !config.profile.is_4444() => {
                return Err(CodecError::InvalidParameter(format!(
                    "ProRes encoder: 4:4:4 chroma requires a 4444 profile, got {:?}",
                    config.profile
                )));
            }
            ChromaFormat::Yuv422 if config.profile.is_4444() => {
                return Err(CodecError::InvalidParameter(format!(
                    "ProRes encoder: a 4444 profile ({:?}) requires 4:4:4 chroma",
                    config.profile
                )));
            }
            _ => {}
        }
        let pixel_format = match config.chroma_format {
            ChromaFormat::Yuv422 => PixelFormat::Yuv422p10le,
            ChromaFormat::Yuv444 => PixelFormat::Yuv444p10le,
        };
        let enc_config = EncoderConfig {
            codec: CodecId::ProRes,
            width: config.width,
            height: config.height,
            pixel_format,
            ..EncoderConfig::default()
        };
        Ok(Self {
            config,
            enc_config,
            frame_count: 0,
            output_queue: Vec::new(),
        })
    }

    /// Encode one VideoFrame and push a packet into the output queue.
    ///
    /// Accepts `Yuv422p10le` for 4:2:2 configs and `Yuv444p10le` for 4:4:4
    /// configs (see [`ProResEncoderConfig::chroma_format`]).
    fn encode_frame_inner(&mut self, frame: &VideoFrame) -> CodecResult<()> {
        let is_444 = self.config.chroma_format == ChromaFormat::Yuv444;
        let expected_format = if is_444 {
            PixelFormat::Yuv444p10le
        } else {
            PixelFormat::Yuv422p10le
        };
        if frame.format != expected_format {
            return Err(CodecError::InvalidParameter(format!(
                "ProRes encoder: expected {:?}, got {:?}",
                expected_format, frame.format
            )));
        }
        if frame.width != self.config.width || frame.height != self.config.height {
            return Err(CodecError::InvalidParameter(format!(
                "ProRes encoder: frame dimensions {}×{} != configured {}×{}",
                frame.width, frame.height, self.config.width, self.config.height
            )));
        }
        if frame.planes.len() < 3 {
            return Err(CodecError::InvalidParameter(format!(
                "ProRes encoder: {expected_format:?} frame needs 3 planes"
            )));
        }

        let width = frame.width as usize;
        let height = frame.height as usize;
        // 4:4:4 chroma is full-width; 4:2:2 chroma is half-width.
        let chroma_width = if is_444 { width } else { width / 2 };
        // Per-MB chroma sample width: 16 for 4:4:4, 8 for 4:2:2.
        let chroma_mb_width = if is_444 { 16 } else { 8 };

        let qscale = self.config.effective_qscale();
        let mb_width_per_slice = 1usize << self.config.log2_slice_mb_width;

        // Convert plane bytes (LE u16) to u16 sample slices.
        let luma_samples = plane_bytes_to_u16(&frame.planes[0].data, width * height);
        let cb_samples = plane_bytes_to_u16(&frame.planes[1].data, chroma_width * height);
        let cr_samples = plane_bytes_to_u16(&frame.planes[2].data, chroma_width * height);

        // Compute slice grid.
        // Each macroblock is 16×16 luma / 8×16 (4:2:2) or 16×16 (4:4:4) chroma.
        let mb_cols_total = width / 16;
        let mb_rows_total = height / 16;
        let slices_per_row = (mb_cols_total + mb_width_per_slice - 1) / mb_width_per_slice;

        let mut encoded_slices: Vec<Vec<u8>> = Vec::with_capacity(mb_rows_total * slices_per_row);

        for mb_row in 0..mb_rows_total {
            for slice_col in 0..slices_per_row {
                let mb_x_start = slice_col * mb_width_per_slice;
                let mb_x_end = (mb_x_start + mb_width_per_slice).min(mb_cols_total);
                let this_mb_width = mb_x_end - mb_x_start;

                // Build luma sub-plane view for this slice (16 rows tall).
                let luma_row_start = mb_row * 16;
                let luma_col_start = mb_x_start * 16;
                let slice_luma_w = this_mb_width * 16;
                let slice_luma = extract_sub_plane(
                    &luma_samples,
                    width,
                    luma_row_start,
                    luma_col_start,
                    slice_luma_w,
                    16,
                );

                // Chroma sub-plane.
                let chroma_col_start = mb_x_start * chroma_mb_width;
                let slice_chroma_w = this_mb_width * chroma_mb_width;
                let slice_cb = extract_sub_plane(
                    &cb_samples,
                    chroma_width,
                    luma_row_start,
                    chroma_col_start,
                    slice_chroma_w,
                    16,
                );
                let slice_cr = extract_sub_plane(
                    &cr_samples,
                    chroma_width,
                    luma_row_start,
                    chroma_col_start,
                    slice_chroma_w,
                    16,
                );

                let slice_bytes = if is_444 {
                    encode_slice_444(
                        &slice_luma,
                        &slice_cb,
                        &slice_cr,
                        slice_luma_w,
                        slice_chroma_w,
                        this_mb_width,
                        qscale,
                        &DEFAULT_LUMA_QUANT_MATRIX,
                        &DEFAULT_CHROMA_QUANT_MATRIX,
                    )
                } else {
                    encode_slice(
                        &slice_luma,
                        &slice_cb,
                        &slice_cr,
                        slice_luma_w,
                        slice_chroma_w,
                        this_mb_width,
                        qscale,
                        &DEFAULT_LUMA_QUANT_MATRIX,
                        &DEFAULT_CHROMA_QUANT_MATRIX,
                    )
                };
                encoded_slices.push(slice_bytes);
            }
        }

        let frame_bytes = write_frame(
            &encoded_slices,
            self.config.profile,
            width as u16,
            height as u16,
            self.config.frame_rate_code,
            self.config.chroma_format,
            InterlaceMode::Progressive,
            0,
            self.config.log2_slice_mb_width,
        );

        let pts = self.frame_count as i64;
        self.output_queue.push(EncodedPacket {
            data: frame_bytes,
            pts,
            dts: pts,
            keyframe: true,
            duration: Some(1),
        });
        self.frame_count += 1;
        Ok(())
    }
}

impl VideoEncoder for ProResEncoder {
    fn codec(&self) -> CodecId {
        CodecId::ProRes
    }

    fn send_frame(&mut self, frame: &VideoFrame) -> CodecResult<()> {
        self.encode_frame_inner(frame)
    }

    fn receive_packet(&mut self) -> CodecResult<Option<EncodedPacket>> {
        if self.output_queue.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.output_queue.remove(0)))
        }
    }

    fn flush(&mut self) -> CodecResult<()> {
        // ProRes is intra-only; nothing to flush.
        Ok(())
    }

    fn config(&self) -> &EncoderConfig {
        &self.enc_config
    }
}

/// Convert a `u8` byte slice (packed LE u16 samples) to a `Vec<u16>`.
/// Each pair of bytes `[lo, hi]` becomes `u16::from_le_bytes([lo, hi])`.
/// Truncates to at most `max_samples` u16 values.
fn plane_bytes_to_u16(data: &[u8], max_samples: usize) -> Vec<u16> {
    let sample_count = (data.len() / 2).min(max_samples);
    let mut out = Vec::with_capacity(sample_count);
    for i in 0..sample_count {
        let lo = data[i * 2];
        let hi = data[i * 2 + 1];
        out.push(u16::from_le_bytes([lo, hi]));
    }
    out
}

/// Extract a contiguous sub-plane from `src` (given full width `src_stride`)
/// starting at `(row_start, col_start)`, of size `out_width × out_height`.
/// Pads with 512 (10-bit midgrey) if the source is too small.
fn extract_sub_plane(
    src: &[u16],
    src_stride: usize,
    row_start: usize,
    col_start: usize,
    out_width: usize,
    out_height: usize,
) -> Vec<u16> {
    let mut out = Vec::with_capacity(out_width * out_height);
    for r in 0..out_height {
        let src_row = row_start + r;
        for c in 0..out_width {
            let src_col = col_start + c;
            let idx = src_row * src_stride + src_col;
            let sample = if idx < src.len() { src[idx] } else { 512 };
            out.push(sample);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Plane, VideoFrame};
    use crate::prores::decode::{decode_slice_to_yuv422, split_slice_planes};
    use crate::prores::frame::{parse_frame_header, FrameContainer};
    use crate::prores::picture::{parse_picture_header, parse_slice_header};

    /// Build a synthetic Yuv422p10le VideoFrame of given dimensions,
    /// filled with a simple ramp pattern.
    fn make_yuv422p10le_frame(width: u32, height: u32) -> VideoFrame {
        let w = width as usize;
        let h = height as usize;
        let cw = w / 2;

        // Y plane: ramp 64..768 across the frame.
        let mut y_bytes = vec![0u8; w * h * 2];
        for row in 0..h {
            for col in 0..w {
                let val = 64u16 + ((row * w + col) as u16 % 704);
                let bytes = val.to_le_bytes();
                let idx = (row * w + col) * 2;
                y_bytes[idx] = bytes[0];
                y_bytes[idx + 1] = bytes[1];
            }
        }

        // Cb/Cr: flat mid-grey (512).
        let chroma_bytes: Vec<u8> = (0..cw * h).flat_map(|_| 512u16.to_le_bytes()).collect();

        let mut frame = VideoFrame::new(PixelFormat::Yuv422p10le, width, height);
        frame.planes = vec![
            Plane::with_dimensions(y_bytes, w * 2, width, height),
            Plane::with_dimensions(chroma_bytes.clone(), cw * 2, width / 2, height),
            Plane::with_dimensions(chroma_bytes, cw * 2, width / 2, height),
        ];
        frame
    }

    #[test]
    fn encoder_new_rejects_odd_dimensions() {
        let cfg = ProResEncoderConfig::new(ProResProfile::Standard, 15, 16);
        assert!(
            ProResEncoder::new(cfg).is_err(),
            "width=15 is not multiple of 16"
        );
    }

    #[test]
    fn encoder_new_rejects_zero_dimensions() {
        let cfg = ProResEncoderConfig::new(ProResProfile::Standard, 0, 0);
        assert!(ProResEncoder::new(cfg).is_err());
    }

    #[test]
    fn encoder_produces_parseable_icpf_frame() {
        let w = 32u32;
        let h = 16u32;
        let cfg = ProResEncoderConfig::new(ProResProfile::Standard, w, h);
        let mut enc = ProResEncoder::new(cfg).expect("encoder");
        let frame = make_yuv422p10le_frame(w, h);
        enc.send_frame(&frame).expect("send_frame");
        let pkt = enc
            .receive_packet()
            .expect("receive_packet")
            .expect("Some packet");

        // Parse the output — should not error.
        let (container, rest) = FrameContainer::parse(&pkt.data).expect("container parse");
        assert!(rest.is_empty());
        let (fhdr, after_fhdr) = parse_frame_header(container.payload).expect("frame hdr");
        assert_eq!(fhdr.width, w as u16);
        assert_eq!(fhdr.height, h as u16);
        assert_eq!(fhdr.profile, ProResProfile::Standard);
        let (_pic_hdr, _) = parse_picture_header(after_fhdr).expect("pic hdr");
    }

    #[test]
    fn encoder_full_roundtrip_small_frame() {
        let w = 32u32;
        let h = 16u32;
        let cfg = ProResEncoderConfig::new(ProResProfile::Standard, w, h);
        let mut enc = ProResEncoder::new(cfg).expect("encoder");
        let frame = make_yuv422p10le_frame(w, h);
        enc.send_frame(&frame).expect("send_frame");
        let pkt = enc
            .receive_packet()
            .expect("receive_packet")
            .expect("Some packet");

        // Decode all slices and check per-pixel error.
        let (container, _) = FrameContainer::parse(&pkt.data).expect("container");
        let (fhdr, after_fhdr) = parse_frame_header(container.payload).expect("fhdr");
        let (pic_hdr, after_pic) = parse_picture_header(after_fhdr).expect("pichdr");

        let slice_count = pic_hdr.slice_count as usize;
        // Skip the offset table.
        let offset_table_bytes = slice_count * 2;
        let slice_data_start = &after_pic[offset_table_bytes..];

        // Collect slice sizes from the offset table.
        let mut slice_sizes = Vec::with_capacity(slice_count);
        for i in 0..slice_count {
            let ofs = i * 2;
            let sz = u16::from_be_bytes([after_pic[ofs], after_pic[ofs + 1]]) as usize;
            slice_sizes.push(sz);
        }

        let luma_w = w as usize;
        let chroma_w = w as usize / 2;
        let luma_stride = luma_w;
        let chroma_stride = chroma_w;
        let mb_width_per_slice = 1usize << pic_hdr.log2_slice_mb_width;

        let mut cursor = 0usize;
        let mb_rows = h as usize / 16;
        let mb_cols_total = w as usize / 16;
        let slices_per_row = (mb_cols_total + mb_width_per_slice - 1) / mb_width_per_slice;

        for mb_row in 0..mb_rows {
            for _slice_col in 0..slices_per_row {
                let idx = mb_row * slices_per_row + _slice_col;
                let sz = slice_sizes[idx];
                let slice_bytes = &slice_data_start[cursor..cursor + sz];
                cursor += sz;

                let (shdr, payload) = parse_slice_header(slice_bytes, false).expect("slice hdr");
                let sd = split_slice_planes(
                    payload,
                    shdr.luma_data_size,
                    shdr.cb_data_size,
                    shdr.cr_data_size,
                    None,
                )
                .expect("split");

                let sl_mb_w =
                    mb_width_per_slice.min(mb_cols_total - _slice_col * mb_width_per_slice);
                let sl_luma_w = sl_mb_w * 16;
                let sl_chroma_w = sl_mb_w * 8;

                let mut dst_luma = vec![0u16; sl_luma_w * 16];
                let mut dst_cb = vec![0u16; sl_chroma_w * 16];
                let mut dst_cr = vec![0u16; sl_chroma_w * 16];

                decode_slice_to_yuv422(
                    sd,
                    &fhdr.luma_quant_matrix,
                    &fhdr.chroma_quant_matrix,
                    shdr.quant_scale,
                    sl_mb_w,
                    &mut dst_luma,
                    sl_luma_w,
                    &mut dst_cb,
                    sl_chroma_w,
                    &mut dst_cr,
                    sl_chroma_w,
                )
                .expect("decode");

                // Check per-pixel error is within acceptable quantization tolerance.
                let col_offset = _slice_col * mb_width_per_slice * 16;
                let row_offset = mb_row * 16;
                let y_plane = &frame.planes[0];
                for r in 0..16 {
                    for c in 0..sl_luma_w {
                        let src_idx = (row_offset + r) * luma_stride + col_offset + c;
                        let src_lo = y_plane.data[src_idx * 2];
                        let src_hi = y_plane.data[src_idx * 2 + 1];
                        let src_val = u16::from_le_bytes([src_lo, src_hi]);
                        let dst_val = dst_luma[r * sl_luma_w + c];
                        let err = (src_val as i32 - dst_val as i32).abs();
                        assert!(
                            err <= 32,
                            "luma round-trip error too large at ({}, {}): src={}, dst={}, err={}",
                            row_offset + r,
                            col_offset + c,
                            src_val,
                            dst_val,
                            err
                        );
                    }
                }
            }
        }

        // Second call to receive_packet should return None.
        assert!(enc.receive_packet().expect("no error").is_none());
    }

    #[test]
    fn default_qscale_by_profile() {
        let check = |profile: ProResProfile, expected_qs: u8| {
            let cfg = ProResEncoderConfig::new(profile, 16, 16);
            assert_eq!(cfg.effective_qscale(), expected_qs, "profile {:?}", profile);
        };
        check(ProResProfile::Proxy, 16);
        check(ProResProfile::Lt, 9);
        check(ProResProfile::Standard, 6);
        check(ProResProfile::Hq, 4);
    }

    #[test]
    fn explicit_qscale_overrides_default() {
        let mut cfg = ProResEncoderConfig::new(ProResProfile::Standard, 16, 16);
        cfg.qscale = 12;
        assert_eq!(cfg.effective_qscale(), 12);
    }

    /// Build a synthetic `Yuv444p10le` VideoFrame: luma is a ramp, chroma
    /// planes are full-resolution with their own gentle gradients.
    fn make_yuv444p10le_frame(width: u32, height: u32) -> VideoFrame {
        let w = width as usize;
        let h = height as usize;

        let mut y_bytes = vec![0u8; w * h * 2];
        let mut cb_bytes = vec![0u8; w * h * 2];
        let mut cr_bytes = vec![0u8; w * h * 2];
        for row in 0..h {
            for col in 0..w {
                let idx = (row * w + col) * 2;
                let y = 64u16 + ((row * w + col) as u16 % 704);
                let cb = 512u16; // flat chroma keeps the round-trip tight
                let cr = 512u16;
                y_bytes[idx..idx + 2].copy_from_slice(&y.to_le_bytes());
                cb_bytes[idx..idx + 2].copy_from_slice(&cb.to_le_bytes());
                cr_bytes[idx..idx + 2].copy_from_slice(&cr.to_le_bytes());
            }
        }

        let mut frame = VideoFrame::new(PixelFormat::Yuv444p10le, width, height);
        frame.planes = vec![
            Plane::with_dimensions(y_bytes, w * 2, width, height),
            Plane::with_dimensions(cb_bytes, w * 2, width, height),
            Plane::with_dimensions(cr_bytes, w * 2, width, height),
        ];
        frame
    }

    #[test]
    fn yuv444_config_rejects_non_4444_profile() {
        let cfg = ProResEncoderConfig::yuv444(ProResProfile::Standard, 32, 16);
        assert!(
            ProResEncoder::new(cfg).is_err(),
            "4:4:4 chroma must require a 4444 profile"
        );
    }

    #[test]
    fn yuv422_config_rejects_4444_profile() {
        let cfg = ProResEncoderConfig::new(ProResProfile::P4444, 32, 16);
        assert!(
            ProResEncoder::new(cfg).is_err(),
            "a 4444 profile must require 4:4:4 chroma"
        );
    }

    #[test]
    fn encoder_444_produces_parseable_ap4h_frame() {
        let w = 32u32;
        let h = 16u32;
        let cfg = ProResEncoderConfig::yuv444(ProResProfile::P4444, w, h);
        let mut enc = ProResEncoder::new(cfg).expect("4:4:4 encoder");
        let frame = make_yuv444p10le_frame(w, h);
        enc.send_frame(&frame).expect("send_frame");
        let pkt = enc
            .receive_packet()
            .expect("receive_packet")
            .expect("Some packet");

        let (container, rest) = FrameContainer::parse(&pkt.data).expect("container parse");
        assert!(rest.is_empty());
        let (fhdr, after_fhdr) = parse_frame_header(container.payload).expect("frame hdr");
        assert_eq!(fhdr.width, w as u16);
        assert_eq!(fhdr.height, h as u16);
        assert_eq!(fhdr.profile, ProResProfile::P4444);
        assert_eq!(fhdr.chroma_format, ChromaFormat::Yuv444);
        let (_pic_hdr, _) = parse_picture_header(after_fhdr).expect("pic hdr");
    }

    #[test]
    fn encoder_444_rejects_wrong_input_format() {
        let cfg = ProResEncoderConfig::yuv444(ProResProfile::P4444, 32, 16);
        let mut enc = ProResEncoder::new(cfg).expect("encoder");
        // Feeding a 4:2:2 frame to a 4:4:4 encoder must error.
        let frame = make_yuv422p10le_frame(32, 16);
        assert!(enc.send_frame(&frame).is_err());
    }
}
