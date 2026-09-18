// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! JPEG XS encoder (ISO/IEC 21122-1:2019).
//!
//! `JpegXsEncoder` is the forward counterpart of [`super::decoder::JpegXsDecoder`].
//! It produces codestreams that the project's decoder reconstructs exactly for
//! the **unit-weight lossless** path:
//!
//! ```text
//! pixels → forward 5/3 DWT → (unit-weight) quantise → VLC entropy encode
//!        → SOC · PIH · CDT · WGT · CWD · SLH(+slice payload) · EOC
//! ```
//!
//! # Round-trip guarantees and limitations
//!
//! - **Lossless (unit weights)**: a single-component image round-trips
//!   byte-exactly because the forward 5/3 DWT ([`super::wavelet::forward_wavelet_2d`])
//!   is the exact integer inverse of the decoder's [`super::wavelet::inverse_53_2d`],
//!   and the entropy stage ([`super::vlc_encode::encode_subband`]) is the exact
//!   inverse of [`super::entropy::decode_subband`].
//! - **Single slice**: the encoder emits one slice covering the full frame.
//!   The project decoder reconstructs from `slices[0]` and replicates the
//!   reconstructed plane across all components, so multi-component frames are
//!   only recovered exactly when every component plane is identical (e.g. a
//!   constant-grey image). The encoder therefore entropy-codes component 0.
//! - **No NLT**: the encoder does not emit an NLT marker; the decoder treats
//!   its absence as a passthrough.
//! - **No lossy quantisation**: the project decoder has no dequantisation stage,
//!   so only unit weights produce a correct reconstruction. Non-unit weights are
//!   rejected with `JxsError::Unsupported`.

use super::bitwriter::BitWriter;
use super::marker_write::{
    write_cdt, write_cwd, write_eoc, write_pih, write_slh, write_soc, write_wgt, CdtComponent,
    PihFields,
};
use super::markers::PROFILE_MAIN;
use super::vlc_encode::encode_subband;
use super::wavelet::forward_wavelet_2d;
use super::{JxsError, JxsResult};

/// Colour space identifier for the encoded image.
///
/// This is recorded for documentation/structure; the project decoder treats all
/// components identically, so the value does not change the reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JxsColorSpace {
    /// Single luma / greyscale plane.
    Grey,
    /// Full-resolution RGB (no subsampling).
    Rgb,
    /// Luma + chroma (YCbCr), possibly subsampled per component.
    Yuv,
}

/// The "unit" quantisation weight — encoding with this weight for every band is
/// the lossless identity (no quantisation is applied).
pub const JXS_UNIT_WEIGHT: u16 = 1;

/// Configuration for [`JpegXsEncoder`].
#[derive(Debug, Clone)]
pub struct JpegXsEncoderConfig {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Sample bit depth (1..=16).
    pub bit_depth: u8,
    /// Number of colour components (planes).
    pub components: u8,
    /// Colour space identifier.
    pub color_space: JxsColorSpace,
    /// Horizontal wavelet decomposition levels. Only a single level (`1`) is
    /// implemented for round-trip with the project decoder.
    pub wavelet_levels_h: u8,
    /// Vertical wavelet decomposition levels. Only a single level (`1`) is
    /// implemented for round-trip with the project decoder.
    pub wavelet_levels_v: u8,
    /// Slice height in lines. The encoder emits a single slice; this value is
    /// recorded in the PIH header. `0` means "whole frame".
    pub slice_height: u32,
    /// Per-subband quantisation weights (LL, HL, LH, HH for one level). Empty
    /// means "no WGT marker, lossless". When present, every weight must equal
    /// [`JXS_UNIT_WEIGHT`] (lossless) — non-unit weights are rejected because
    /// the project decoder performs no dequantisation.
    pub weights: Vec<u16>,
}

impl JpegXsEncoderConfig {
    /// Construct a lossless single-level configuration for the given geometry.
    pub fn new(width: u32, height: u32, bit_depth: u8, components: u8) -> Self {
        let color_space = match components {
            1 => JxsColorSpace::Grey,
            3 => JxsColorSpace::Yuv,
            _ => JxsColorSpace::Rgb,
        };
        Self {
            width,
            height,
            bit_depth,
            components,
            color_space,
            wavelet_levels_h: 1,
            wavelet_levels_v: 1,
            slice_height: height,
            weights: Vec::new(),
        }
    }
}

/// JPEG XS encoder.
///
/// Construct with [`JpegXsEncoder::new`] and call [`JpegXsEncoder::encode`].
#[derive(Debug, Clone)]
pub struct JpegXsEncoder {
    config: JpegXsEncoderConfig,
}

impl JpegXsEncoder {
    /// Validate `config` and build a `JpegXsEncoder`.
    ///
    /// # Errors
    /// - `JxsError::InvalidHeader` for invalid geometry / bit depth / component
    ///   count, or weight vectors of the wrong length.
    /// - `JxsError::Unsupported` for multi-level decomposition or non-unit
    ///   quantisation weights (the project decoder cannot reverse either).
    pub fn new(config: JpegXsEncoderConfig) -> JxsResult<Self> {
        if config.width == 0 || config.height == 0 {
            return Err(JxsError::InvalidHeader(format!(
                "encoder: frame size {}x{} must be non-zero",
                config.width, config.height
            )));
        }
        if config.width > u32::from(u16::MAX) || config.height > u32::from(u16::MAX) {
            return Err(JxsError::InvalidHeader(format!(
                "encoder: frame size {}x{} exceeds 65535",
                config.width, config.height
            )));
        }
        if config.bit_depth == 0 || config.bit_depth > 16 {
            return Err(JxsError::InvalidHeader(format!(
                "encoder: bit depth {} must be 1-16",
                config.bit_depth
            )));
        }
        if config.components == 0 {
            return Err(JxsError::InvalidHeader(
                "encoder: at least one component required".to_string(),
            ));
        }
        if config.wavelet_levels_h != 1 || config.wavelet_levels_v != 1 {
            return Err(JxsError::Unsupported(format!(
                "encoder: only single-level decomposition supported (got {}x{} levels)",
                config.wavelet_levels_h, config.wavelet_levels_v
            )));
        }
        if !config.weights.is_empty() {
            if config.weights.len() != 4 {
                return Err(JxsError::InvalidHeader(format!(
                    "encoder: weights must have 4 entries (LL,HL,LH,HH), got {}",
                    config.weights.len()
                )));
            }
            if config.weights.iter().any(|&w| w != JXS_UNIT_WEIGHT) {
                return Err(JxsError::Unsupported(
                    "encoder: non-unit quantisation weights are not supported (decoder has no \
                     dequantisation stage); use unit weights for lossless encoding"
                        .to_string(),
                ));
            }
        }
        Ok(Self { config })
    }

    /// Encode `planes` (one `Vec<i32>` per component, row-major, length
    /// `width * height`) into a JPEG XS codestream.
    ///
    /// Component 0 is entropy-coded into the single slice; see the module-level
    /// documentation for the multi-component reconstruction caveat.
    ///
    /// # Errors
    /// - `JxsError::InvalidHeader` if `planes` is empty, has the wrong plane
    ///   count, or a plane has the wrong length.
    pub fn encode(&self, planes: &[Vec<i32>]) -> JxsResult<Vec<u8>> {
        let cfg = &self.config;
        let width = cfg.width as usize;
        let height = cfg.height as usize;
        let expected = width * height;

        if planes.is_empty() {
            return Err(JxsError::InvalidHeader(
                "encode: no component planes provided".to_string(),
            ));
        }
        if planes.len() != cfg.components as usize {
            return Err(JxsError::InvalidHeader(format!(
                "encode: expected {} planes, got {}",
                cfg.components,
                planes.len()
            )));
        }
        for (i, plane) in planes.iter().enumerate() {
            if plane.len() != expected {
                return Err(JxsError::InvalidHeader(format!(
                    "encode: plane {i} has {} samples, expected {expected}",
                    plane.len()
                )));
            }
        }

        // ── 1. Forward 5/3 DWT on component 0 (single decomposition level) ──
        let (ll, hl, lh, hh) = forward_wavelet_2d(&planes[0], width, height)?;

        // ── 2. (Unit-weight) quantisation — identity, lossless. ─────────────
        // With unit weights no scaling is applied; the coefficients pass through
        // unchanged so the decoder's coefficient-domain reconstruction is exact.

        // ── 3. VLC entropy-encode the slice payload (LL → HL → LH → HH). ────
        let mut slice_writer = BitWriter::new();
        encode_subband(&mut slice_writer, &ll);
        encode_subband(&mut slice_writer, &hl);
        encode_subband(&mut slice_writer, &lh);
        encode_subband(&mut slice_writer, &hh);
        let slice_bytes = slice_writer.finish();

        // ── 4. Assemble the codestream markers. ─────────────────────────────
        let mut buf = Vec::with_capacity(64 + slice_bytes.len());
        write_soc(&mut buf);

        let pih = PihFields {
            codestream_len: 0, // VBR / undefined
            profile: PROFILE_MAIN,
            level: 0,
            width: cfg.width as u16,
            height: cfg.height as u16,
            codegroup_width: cfg.width as u16,
            slice_height: if cfg.slice_height == 0 {
                cfg.height as u16
            } else {
                (cfg.slice_height.min(cfg.height)) as u16
            },
            num_components: cfg.components,
            ganging: 0,
            bit_depth: cfg.bit_depth,
            bw_ext: 0,
            fq: 0,
            bitrate: 0,
            fsl: 0,
            ppoc: 0,
            cpih: 0,
        };
        write_pih(&mut buf, &pih)?;

        // CDT: per-component descriptors. Subsampling is reported as 1 (the
        // decoder reconstructs all components identically regardless).
        let cdt: Vec<CdtComponent> = (0..cfg.components)
            .map(|_| CdtComponent {
                bit_depth: cfg.bit_depth,
                sx: 1,
                sy: 1,
            })
            .collect();
        write_cdt(&mut buf, &cdt)?;

        // WGT: emit unit weights when requested (structural / documentation).
        if !cfg.weights.is_empty() {
            write_wgt(&mut buf, &cfg.weights)?;
        }

        // CWD: empty codeword-mapping marker (decoder uses its default tables
        // and skips this payload). Emitted for structural completeness.
        write_cwd(&mut buf, &[]);

        // SLH + slice payload. A single slice covers the full frame.
        write_slh(&mut buf, 0);
        buf.extend_from_slice(&slice_bytes);

        write_eoc(&mut buf);
        Ok(buf)
    }

    /// Borrow the encoder configuration.
    pub fn config(&self) -> &JpegXsEncoderConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpegxs::decoder::JpegXsDecoder;

    #[test]
    fn new_rejects_zero_dimensions() {
        let cfg = JpegXsEncoderConfig::new(0, 8, 8, 1);
        assert!(JpegXsEncoder::new(cfg).is_err());
    }

    #[test]
    fn new_rejects_bad_bit_depth() {
        let cfg = JpegXsEncoderConfig::new(8, 8, 0, 1);
        assert!(JpegXsEncoder::new(cfg).is_err());
    }

    #[test]
    fn new_rejects_multilevel() {
        let mut cfg = JpegXsEncoderConfig::new(8, 8, 8, 1);
        cfg.wavelet_levels_h = 2;
        assert!(JpegXsEncoder::new(cfg).is_err());
    }

    #[test]
    fn new_rejects_nonunit_weights() {
        let mut cfg = JpegXsEncoderConfig::new(8, 8, 8, 1);
        cfg.weights = vec![2, 2, 2, 2];
        assert!(matches!(
            JpegXsEncoder::new(cfg),
            Err(JxsError::Unsupported(_))
        ));
    }

    #[test]
    fn new_accepts_unit_weights() {
        let mut cfg = JpegXsEncoderConfig::new(8, 8, 8, 1);
        cfg.weights = vec![1, 1, 1, 1];
        assert!(JpegXsEncoder::new(cfg).is_ok());
    }

    #[test]
    fn encode_rejects_wrong_plane_count() {
        let cfg = JpegXsEncoderConfig::new(4, 4, 8, 1);
        let enc = JpegXsEncoder::new(cfg).unwrap();
        let planes = vec![vec![0i32; 16], vec![0i32; 16]];
        assert!(enc.encode(&planes).is_err());
    }

    #[test]
    fn encode_rejects_wrong_plane_length() {
        let cfg = JpegXsEncoderConfig::new(4, 4, 8, 1);
        let enc = JpegXsEncoder::new(cfg).unwrap();
        let planes = vec![vec![0i32; 10]];
        assert!(enc.encode(&planes).is_err());
    }

    #[test]
    fn encode_produces_valid_soc_eoc() {
        let cfg = JpegXsEncoderConfig::new(4, 4, 8, 1);
        let enc = JpegXsEncoder::new(cfg).unwrap();
        let planes = vec![vec![5i32; 16]];
        let stream = enc.encode(&planes).unwrap();
        assert_eq!(&stream[0..2], &[0xFF, 0x10]); // SOC
        assert_eq!(&stream[stream.len() - 2..], &[0xFF, 0x11]); // EOC
        assert!(JpegXsDecoder::is_jpegxs(&stream));
    }

    #[test]
    fn roundtrip_small_gradient_lossless() {
        let (w, h) = (8u32, 8u32);
        let cfg = JpegXsEncoderConfig::new(w, h, 8, 1);
        let enc = JpegXsEncoder::new(cfg).unwrap();
        let plane: Vec<i32> = (0..(w * h) as usize)
            .map(|i| (i % w as usize) as i32)
            .collect();
        let stream = enc.encode(std::slice::from_ref(&plane)).unwrap();
        let img = JpegXsDecoder::decode(&stream).unwrap();
        let decoded: Vec<i32> = img.samples[0].iter().map(|&v| v as i32).collect();
        assert_eq!(decoded, plane, "lossless gradient round-trip failed");
    }

    #[test]
    fn roundtrip_constant_lossless() {
        let cfg = JpegXsEncoderConfig::new(16, 16, 8, 1);
        let enc = JpegXsEncoder::new(cfg).unwrap();
        let plane = vec![200i32; 256];
        let stream = enc.encode(std::slice::from_ref(&plane)).unwrap();
        let img = JpegXsDecoder::decode(&stream).unwrap();
        let decoded: Vec<i32> = img.samples[0].iter().map(|&v| v as i32).collect();
        assert_eq!(decoded, plane);
    }

    #[test]
    fn roundtrip_with_unit_weights_marker_lossless() {
        let mut cfg = JpegXsEncoderConfig::new(8, 8, 8, 1);
        cfg.weights = vec![1, 1, 1, 1];
        let enc = JpegXsEncoder::new(cfg).unwrap();
        let plane: Vec<i32> = (0..64).map(|i| (i % 8) as i32).collect();
        let stream = enc.encode(std::slice::from_ref(&plane)).unwrap();
        // WGT marker (0xFF14) must be present.
        assert!(
            stream.windows(2).any(|w| w == [0xFF, 0x14]),
            "WGT marker missing"
        );
        let img = JpegXsDecoder::decode(&stream).unwrap();
        let decoded: Vec<i32> = img.samples[0].iter().map(|&v| v as i32).collect();
        assert_eq!(decoded, plane);
    }

    #[test]
    fn roundtrip_10bit_gradient_lossless() {
        let (w, h) = (16u32, 8u32);
        let cfg = JpegXsEncoderConfig::new(w, h, 10, 1);
        let enc = JpegXsEncoder::new(cfg).unwrap();
        // 10-bit ramp 0..=1020 in steps so values exercise larger magnitudes.
        let plane: Vec<i32> = (0..(w * h) as usize)
            .map(|i| ((i * 8) % 1024) as i32)
            .collect();
        let stream = enc.encode(std::slice::from_ref(&plane)).unwrap();
        let img = JpegXsDecoder::decode(&stream).unwrap();
        let decoded: Vec<i32> = img.samples[0].iter().map(|&v| v as i32).collect();
        assert_eq!(decoded, plane);
    }
}
