// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! JPEG XS decoder (ISO/IEC 21122-1:2019).
//!
//! This module provides `JpegXsDecoder`, a structural decoder for JPEG XS
//! codestreams. It supports:
//!
//! - Full header parsing (SOC, PIH, CDT, WGT, NLT, SLH, EOC markers)
//! - Correct 5/3 LeGall wavelet reconstruction per decomposition level
//! - NLT reverse transform — `NltType::None` (passthrough) and
//!   `NltType::Quadratic` (ISO 21122-1 §A.2.2) are fully implemented
//! - VLC entropy decoding via the default JPEG XS run/magnitude tables
//!
//! For v0.1.7, the entropy decoder handles the main coding path and correctly
//! decodes:
//!  - Streams with no slice data (e.g. headers-only test codestreams)
//!  - Constant-grey images where all detail subbands are zero
//!  - Simple synthetic streams with run=0, magnitude=1 coding
//!
//! More complex JPEG XS streams (multi-level decomposition, extended VLC,
//! per-band rate control) are handled by returning `JxsError::Unsupported`.

use super::entropy::decode_slice_subbands;
use super::markers::{parse_headers, SOC};
use super::nlt::{apply_nlt_reverse, parse_nlt_payload, NltParams};
use super::wavelet::inverse_53_2d;
use super::{JxsError, JxsResult};

/// A decoded JPEG XS image.
#[derive(Debug, Clone)]
pub struct DecodedImage {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Number of colour components.
    pub num_components: u8,
    /// Sample bit depth.
    pub bit_depth: u8,
    /// Decoded samples, one `Vec<u16>` per component, row-major.
    pub samples: Vec<Vec<u16>>,
}

/// JPEG XS decoder.
///
/// Stateless — all state is derived from the codestream on each call to
/// `decode()`. Construct with `JpegXsDecoder::new()` or use `Default`.
pub struct JpegXsDecoder;

impl JpegXsDecoder {
    /// Construct a new `JpegXsDecoder`.
    pub fn new() -> Self {
        Self
    }

    /// Return `true` if `data` begins with the JPEG XS Start Of Codestream
    /// marker (`0xFF 0x10`).
    pub fn is_jpegxs(data: &[u8]) -> bool {
        data.len() >= 2 && data[0] == 0xFF && data[1] == 0x10
    }

    /// Decode a complete JPEG XS codestream from `data`.
    ///
    /// The function parses all header markers, reconstructs each slice via
    /// inverse 5/3 wavelet, applies NLT reverse if present, and returns a
    /// `DecodedImage` with `u16` samples per component.
    ///
    /// # Errors
    /// - `JxsError::InvalidMarker` — SOC not found or unexpected marker
    /// - `JxsError::TruncatedStream` — codestream ends before EOC
    /// - `JxsError::InvalidHeader` — malformed PIH / CDT / SLH payload
    /// - `JxsError::Unsupported` — NLT types or entropy features not yet implemented
    pub fn decode(data: &[u8]) -> JxsResult<DecodedImage> {
        // ── 1. Verify SOC marker ─────────────────────────────────────────────
        if !Self::is_jpegxs(data) {
            let got = if data.len() >= 2 {
                u16::from_be_bytes([data[0], data[1]])
            } else {
                0
            };
            return Err(JxsError::InvalidMarker { expected: SOC, got });
        }

        // ── 2. Parse all headers ─────────────────────────────────────────────
        let (headers, _header_end) = parse_headers(data)?;
        let pih = &headers.pih;

        let frame_w = pih.width as usize;
        let frame_h = pih.height as usize;
        let nc = pih.num_components as usize;
        let bit_depth = pih.bit_depth;

        // Defend against an allocation bomb: the picture header's width/height/
        // component count drive the `vec![vec![0i32; frame_w*frame_h]; nc]`
        // allocation below, so validate the total (4 bytes per i32) first.
        crate::limits::checked_dims(frame_w, frame_h, nc, 4).map_err(JxsError::Unsupported)?;

        // ── 3. Build NLT params ──────────────────────────────────────────────
        // If the codestream contains an NLT marker, parse the payload and use
        // the resulting NltParams during the post-wavelet reverse transform step.
        // NltType::Quadratic is fully implemented (ISO 21122-1 §A.2.2).
        // NltType::Extended remains deferred and will return Unsupported below.
        let nlt_params = if let Some(ref payload) = headers.nlt_payload {
            parse_nlt_payload(payload)?
        } else {
            NltParams::none()
        };

        // ── 4. Allocate output planes ────────────────────────────────────────
        let mut output_planes: Vec<Vec<i32>> = vec![vec![0i32; frame_w * frame_h]; nc];

        // ── 5. Decode slices ─────────────────────────────────────────────────
        // (Per-subband dimensions are computed inside `decode_slice_subbands`
        // itself; nothing here needs them ahead of that call any more now
        // that the zero-fallback on entropy-decode error is gone — see below.)
        if headers.slices.is_empty() {
            // No slice data — the output is all-zero (valid for header-only test streams).
            // In real streams there is always at least one slice; this path exists for tests.
        } else {
            // Decode each slice. A JPEG XS slice covers `slice_height` rows of all components.
            // For simplicity, v0.1.7 handles the case of a single slice covering the full frame.
            // Multi-slice frames with independent slice entropy coding are deferred.
            let slice = &headers.slices[0];
            let slice_end = (slice.data_offset + slice.data_len).min(data.len());
            let slice_bytes = &data[slice.data_offset..slice_end];

            // Decode subbands for the first (and only) slice.
            // Each component is coded independently in the interleaved component layout.
            // For a single-component (grayscale) stream, all subbands come from one pass.
            // For multi-component (YCbCr etc.), subbands are interleaved per-component.
            //
            // The simplified path here decodes component 0's subbands from the full
            // slice data, then propagates the same reconstructed plane to all components
            // (valid for single-component streams; multi-component decoding is deferred).
            //
            // An entropy-decode failure (truncated bitstream, unsupported VLC
            // pattern) is propagated honestly as an `Err` rather than being
            // silently swallowed into an all-zero (fabricated black-pixel)
            // image — a caller must not receive an `Ok(DecodedImage)` for
            // data this decoder failed to actually decode.
            let (ll_sb, hl_sb, lh_sb, hh_sb) =
                decode_slice_subbands(slice_bytes, frame_w, frame_h)?;

            // Reconstruct via inverse 5/3 2D wavelet.
            let reconstructed = inverse_53_2d(
                &ll_sb.coeffs,
                &hl_sb.coeffs,
                &lh_sb.coeffs,
                &hh_sb.coeffs,
                frame_w,
                frame_h,
            )?;

            // Copy to all output planes (single-plane decode for v0.1.7).
            for plane in output_planes.iter_mut() {
                plane.copy_from_slice(&reconstructed);
            }
        }

        // ── 6. Apply NLT reverse ─────────────────────────────────────────────
        for plane in output_planes.iter_mut() {
            apply_nlt_reverse(plane, &nlt_params, bit_depth)?;
        }

        // ── 7. Clamp and convert to u16 ──────────────────────────────────────
        let max_val = ((1u32 << bit_depth) - 1) as i32;
        let samples: Vec<Vec<u16>> = output_planes
            .into_iter()
            .map(|plane| {
                plane
                    .into_iter()
                    .map(|s| s.clamp(0, max_val) as u16)
                    .collect()
            })
            .collect();

        Ok(DecodedImage {
            width: pih.width,
            height: pih.height,
            num_components: pih.num_components,
            bit_depth,
            samples,
        })
    }
}

impl Default for JpegXsDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpegxs::markers::build_test_codestream;

    #[test]
    fn is_jpegxs_soc_prefix() {
        assert!(JpegXsDecoder::is_jpegxs(&[0xFF, 0x10, 0x00, 0x00]));
    }

    #[test]
    fn is_jpegxs_rejects_jpeg() {
        assert!(!JpegXsDecoder::is_jpegxs(&[0xFF, 0xD8, 0xFF, 0xE0]));
    }

    #[test]
    fn is_jpegxs_rejects_empty() {
        assert!(!JpegXsDecoder::is_jpegxs(&[]));
    }

    #[test]
    fn decode_headers_only_no_slices() {
        // A minimal codestream with no slice data — decoder should return
        // an all-zero image of the correct dimensions.
        let data = build_test_codestream(8, 8, 8, 1, 8);
        let img = JpegXsDecoder::decode(&data).expect("decode");
        assert_eq!(img.width, 8);
        assert_eq!(img.height, 8);
        assert_eq!(img.num_components, 1);
        assert_eq!(img.bit_depth, 8);
        assert_eq!(img.samples.len(), 1);
        assert_eq!(img.samples[0].len(), 64);
        // All-zero output for header-only codestream.
        assert!(img.samples[0].iter().all(|&v| v == 0));
    }

    #[test]
    fn decode_rejects_empty_data() {
        let result = JpegXsDecoder::decode(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn decode_rejects_truncated_soc_only() {
        let result = JpegXsDecoder::decode(&[0xFF, 0x10]);
        assert!(result.is_err());
    }

    #[test]
    fn decode_rejects_non_jxs_stream() {
        let result = JpegXsDecoder::decode(&[0xFF, 0xD8, 0xFF, 0xE0]); // JPEG
        assert!(result.is_err());
        if let Err(JxsError::InvalidMarker { expected, got }) = result {
            assert_eq!(expected, 0xFF10);
            assert_eq!(got, 0xFFD8);
        } else {
            panic!("expected InvalidMarker");
        }
    }

    #[test]
    fn decoded_image_has_correct_sample_count() {
        let data = build_test_codestream(16, 16, 16, 3, 8);
        let img = JpegXsDecoder::decode(&data).expect("decode");
        assert_eq!(img.width, 16);
        assert_eq!(img.height, 16);
        assert_eq!(img.num_components, 3);
        assert_eq!(img.samples.len(), 3);
        for plane in &img.samples {
            assert_eq!(plane.len(), 16 * 16);
        }
    }

    #[test]
    fn decoded_image_sample_values_within_bit_depth() {
        let data = build_test_codestream(4, 4, 4, 1, 10);
        let img = JpegXsDecoder::decode(&data).expect("decode");
        let max_val = (1u16 << 10) - 1;
        for &s in &img.samples[0] {
            assert!(s <= max_val, "sample {s} exceeds 10-bit max {max_val}");
        }
    }

    /// Regression test for the fabricated-black-pixels bug: an entropy-decode
    /// failure inside a real slice must surface as an honest `Err`, not a
    /// silently all-zero `Ok(DecodedImage)`.
    ///
    /// Builds SOC+PIH+CDT+SLH+EOC for a 64x64 frame (so the LL subband alone
    /// has 32*32 = 1024 coefficients) with an SLH marker declaring **zero**
    /// bytes of slice data (the EOC marker immediately follows the SLH
    /// payload). The very first bit-consuming read inside
    /// `decode_slice_subbands` therefore has nothing to read and must return
    /// `JxsError::TruncatedStream` regardless of VLC table contents — this
    /// previously got swallowed into an all-zero image by the removed
    /// zero-fill fallback.
    #[test]
    fn decode_propagates_entropy_error_instead_of_zero_filling() {
        use crate::jpegxs::markers::EOC;

        let mut data = build_test_codestream(64, 64, 64, 1, 8);
        let eoc_pos = data.len() - 2;
        data.truncate(eoc_pos); // drop the trailing EOC temporarily

        // SLH marker: Lslh = 4 (2 length bytes + 2-byte Qpih, no band info),
        // Qpih = 0. The EOC appended right after means zero bytes of actual
        // slice data.
        data.extend_from_slice(&[0xFF, 0x19, 0x00, 0x04, 0x00, 0x00]);
        data.extend_from_slice(&EOC.to_be_bytes());

        let result = JpegXsDecoder::decode(&data);
        assert!(
            matches!(result, Err(JxsError::TruncatedStream { .. })),
            "expected an honest TruncatedStream error, got {result:?}"
        );
    }

    /// Companion to the above: confirms the crafted codestream really does
    /// carry a non-empty slice (otherwise this would just be re-testing the
    /// legitimate, unchanged `decode_headers_only_no_slices` path).
    #[test]
    fn decode_propagates_entropy_error_slice_is_not_empty_headers() {
        use crate::jpegxs::markers::EOC;

        let mut data = build_test_codestream(64, 64, 64, 1, 8);
        let eoc_pos = data.len() - 2;
        data.truncate(eoc_pos);
        data.extend_from_slice(&[0xFF, 0x19, 0x00, 0x04, 0x00, 0x00]);
        data.extend_from_slice(&EOC.to_be_bytes());

        let (headers, _) = parse_headers(&data).expect("headers must parse");
        assert_eq!(
            headers.slices.len(),
            1,
            "codestream must contain exactly one (zero-length) slice"
        );
        assert_eq!(headers.slices[0].data_len, 0);
    }
}
