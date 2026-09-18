// Copyright 2024 The OxiMedia Project Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Theora encode → decode round-trip integration tests.
//!
//! These tests verify that the Theora encoder produces a bitstream that the
//! Theora decoder can reproduce within the expected DCT/quantisation error
//! tolerance, promoting Theora from "Bitstream-parsing" to "Functional".

#[cfg(feature = "theora")]
mod theora_roundtrip {
    use oximedia_codec::frame::VideoFrame;
    use oximedia_codec::theora::{TheoraConfig, TheoraDecoder, TheoraEncoder};
    use oximedia_codec::traits::{EncodedPacket, VideoDecoder, VideoEncoder};
    use oximedia_core::{PixelFormat, Rational, Timestamp};

    /// Build a 32×16 YUV420p ramp frame:
    ///   Y[x, y] = (x + y * 16) % 256
    ///   U[x, y] = (x * 2 + y * 8) % 256
    ///   V[x, y] = (x * 3 + y * 4 + 64) % 256
    fn make_ramp_frame(width: u32, height: u32, pts: i64) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.timestamp = Timestamp::new(pts, Rational::new(1, 1000));
        frame.allocate();

        let w = width as usize;
        let h = height as usize;

        // Y plane: full resolution
        if !frame.planes.is_empty() {
            for y in 0..h {
                for x in 0..w {
                    frame.planes[0].data[y * w + x] = ((x + y * 16) % 256) as u8;
                }
            }
        }

        // U plane: half width, half height
        if frame.planes.len() > 1 {
            let uw = w / 2;
            let uh = h / 2;
            for y in 0..uh {
                for x in 0..uw {
                    frame.planes[1].data[y * uw + x] = ((x * 2 + y * 8) % 256) as u8;
                }
            }
        }

        // V plane: half width, half height
        if frame.planes.len() > 2 {
            let vw = w / 2;
            let vh = h / 2;
            for y in 0..vh {
                for x in 0..vw {
                    frame.planes[2].data[y * vw + x] = ((x * 3 + y * 4 + 64) % 256) as u8;
                }
            }
        }

        frame
    }

    /// Compute per-plane maximum absolute error and mean absolute error.
    fn plane_error_stats(original: &VideoFrame, decoded: &VideoFrame, plane: usize) -> (u32, f64) {
        if original.planes.len() <= plane || decoded.planes.len() <= plane {
            return (0, 0.0);
        }

        let orig = &original.planes[plane].data;
        let dec = &decoded.planes[plane].data;
        let len = orig.len().min(dec.len());

        if len == 0 {
            return (0, 0.0);
        }

        let mut max_err: u32 = 0;
        let mut sum_err: u64 = 0;

        for i in 0..len {
            let err = (i32::from(orig[i]) - i32::from(dec[i])).unsigned_abs();
            if err > max_err {
                max_err = err;
            }
            sum_err += u64::from(err);
        }

        let mean = sum_err as f64 / len as f64;
        (max_err, mean)
    }

    /// Find the first pixel that exceeds the tolerance, for diagnostics.
    fn first_mismatch(
        original: &VideoFrame,
        decoded: &VideoFrame,
        plane: usize,
        tolerance: u32,
    ) -> Option<(usize, u8, u8)> {
        if original.planes.len() <= plane || decoded.planes.len() <= plane {
            return None;
        }

        let orig = &original.planes[plane].data;
        let dec = &decoded.planes[plane].data;
        let len = orig.len().min(dec.len());

        for i in 0..len {
            let err = (i32::from(orig[i]) - i32::from(dec[i])).unsigned_abs();
            if err > tolerance {
                return Some((i, orig[i], dec[i]));
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Test: I-frame round-trip 32×16, quality 48
    // -----------------------------------------------------------------------

    /// An I-frame (keyframe) at quality 48 must round-trip within 8 luma
    /// units of error per pixel.  DCT coefficients are lossy-quantised, so
    /// pixel-exact fidelity is not expected; the tolerance covers the
    /// quantisation error at this quality setting.
    #[test]
    fn i_frame_yuv420p_32x16_quality48() {
        const WIDTH: u32 = 32;
        const HEIGHT: u32 = 16;
        const QUALITY: u8 = 48;
        const LUMA_TOLERANCE: u32 = 8;
        const CHROMA_TOLERANCE: u32 = 12;

        // --- Encode ---
        let config = TheoraConfig::new(WIDTH, HEIGHT).with_quality(QUALITY);
        let mut encoder = TheoraEncoder::new(config).expect("encoder creation must succeed");

        let original = make_ramp_frame(WIDTH, HEIGHT, 0);
        encoder
            .send_frame(&original)
            .expect("send_frame must succeed");

        let packet: EncodedPacket = encoder
            .receive_packet()
            .expect("receive_packet must not error")
            .expect("packet must be present after send_frame");

        assert!(packet.keyframe, "first frame must be a keyframe");
        assert!(!packet.data.is_empty(), "encoded packet must not be empty");

        // --- Decode ---
        let mut decoder = TheoraDecoder::new(WIDTH, HEIGHT, PixelFormat::Yuv420p)
            .expect("decoder creation must succeed");

        decoder
            .send_packet(&packet.data, packet.pts)
            .expect("send_packet must succeed");

        let decoded_frame = decoder
            .receive_frame()
            .expect("receive_frame must not error")
            .expect("frame must be present after send_packet");

        assert_eq!(decoded_frame.width, WIDTH);
        assert_eq!(decoded_frame.height, HEIGHT);
        assert_eq!(decoded_frame.format, PixelFormat::Yuv420p);
        assert_eq!(decoded_frame.planes.len(), 3, "YUV420p must have 3 planes");

        // --- Verify per-plane error ---
        for plane in 0..3 {
            let tolerance = if plane == 0 {
                LUMA_TOLERANCE
            } else {
                CHROMA_TOLERANCE
            };
            let plane_name = ["Y", "U", "V"][plane];

            let (max_err, mean_err) = plane_error_stats(&original, &decoded_frame, plane);

            // Diagnostic info on failure
            let mismatch = first_mismatch(&original, &decoded_frame, plane, tolerance);

            assert!(
                max_err <= tolerance,
                "Plane {plane_name}: max per-pixel error {max_err} exceeds tolerance {tolerance}. \
                 Mean error = {mean_err:.2}. \
                 First mismatch: {mismatch:?}. \
                 Encoded packet size = {} bytes.",
                packet.data.len()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test: I-frame round-trip 32×16, quality 32 (lower quality = larger
    //       quantisation step — tolerance is proportionally wider)
    // -----------------------------------------------------------------------

    #[test]
    fn i_frame_yuv420p_32x16_quality32() {
        const WIDTH: u32 = 32;
        const HEIGHT: u32 = 16;
        const QUALITY: u8 = 32;
        const LUMA_TOLERANCE: u32 = 16;
        const CHROMA_TOLERANCE: u32 = 20;

        let config = TheoraConfig::new(WIDTH, HEIGHT).with_quality(QUALITY);
        let mut encoder = TheoraEncoder::new(config).expect("encoder creation must succeed");

        let original = make_ramp_frame(WIDTH, HEIGHT, 1000);
        encoder
            .send_frame(&original)
            .expect("send_frame must succeed");

        let packet = encoder
            .receive_packet()
            .expect("receive_packet must not error")
            .expect("packet must be present");

        assert!(packet.keyframe);

        let mut decoder = TheoraDecoder::new(WIDTH, HEIGHT, PixelFormat::Yuv420p)
            .expect("decoder creation must succeed");

        decoder
            .send_packet(&packet.data, packet.pts)
            .expect("send_packet must succeed");

        let decoded_frame = decoder
            .receive_frame()
            .expect("receive_frame must not error")
            .expect("frame must be present");

        for plane in 0..3 {
            let tolerance = if plane == 0 {
                LUMA_TOLERANCE
            } else {
                CHROMA_TOLERANCE
            };
            let plane_name = ["Y", "U", "V"][plane];

            let (max_err, mean_err) = plane_error_stats(&original, &decoded_frame, plane);
            let mismatch = first_mismatch(&original, &decoded_frame, plane, tolerance);

            assert!(
                max_err <= tolerance,
                "Plane {plane_name}: max per-pixel error {max_err} exceeds tolerance {tolerance}. \
                 Mean error = {mean_err:.2}. \
                 First mismatch: {mismatch:?}."
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test: uniform flat frame (DC-only blocks — zero AC energy).
    // For a flat 128-grey frame, each 8×8 block has a single DC coefficient;
    // all AC coefficients are zero.  The round-trip must be pixel-exact (or
    // very close: tolerance 2 to allow rounding in the DCT/IDCT).
    // -----------------------------------------------------------------------

    #[test]
    fn i_frame_flat_grey_pixel_near_exact() {
        const WIDTH: u32 = 16;
        const HEIGHT: u32 = 16;
        const QUALITY: u8 = 48;
        const TOLERANCE: u32 = 2;

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, WIDTH, HEIGHT);
        frame.timestamp = Timestamp::new(0, Rational::new(1, 1000));
        frame.allocate();

        // Fill all planes with constant 128.
        for plane in &mut frame.planes {
            plane.data.fill(128);
        }

        let config = TheoraConfig::new(WIDTH, HEIGHT).with_quality(QUALITY);
        let mut encoder = TheoraEncoder::new(config).expect("encoder creation must succeed");

        encoder.send_frame(&frame).expect("send_frame must succeed");
        let packet = encoder
            .receive_packet()
            .expect("receive_packet must not error")
            .expect("packet must be present");

        let mut decoder = TheoraDecoder::new(WIDTH, HEIGHT, PixelFormat::Yuv420p)
            .expect("decoder creation must succeed");

        decoder
            .send_packet(&packet.data, packet.pts)
            .expect("send_packet must succeed");

        let decoded = decoder
            .receive_frame()
            .expect("receive_frame must not error")
            .expect("frame must be present");

        for plane in 0..3 {
            let plane_name = ["Y", "U", "V"][plane];
            let (max_err, mean_err) = plane_error_stats(&frame, &decoded, plane);
            assert!(
                max_err <= TOLERANCE,
                "Flat grey plane {plane_name}: max error {max_err} > {TOLERANCE}. \
                 Mean = {mean_err:.2}."
            );
        }
    }
}
