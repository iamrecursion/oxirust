//! JPEG-LS encoder → decoder round-trip integration tests (ISO 14495-1).
//!
//! Every test encodes with the public `JpegLsEncoder` and decodes the result
//! with the existing `JpegLsDecoder`, asserting losslessness (NEAR = 0) or the
//! `±NEAR` bound (NEAR > 0). This validates that the forward LOCO-I path is the
//! exact inverse of the decoder for all interleave modes.

#[cfg(feature = "jpegls")]
mod jpegls_encode_tests {
    use oximedia_codec::jpegls::{JpegLsDecoder, JpegLsEncoder, JpegLsEncoderConfig};

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Build a deterministic gradient plane of `w * h` 8-bit samples.
    fn gradient_plane(w: usize, h: usize) -> Vec<u16> {
        let mut v = Vec::with_capacity(w * h);
        for row in 0..h {
            for col in 0..w {
                v.push((((row * 7 + col * 3 + 11) & 0xFF) as u16).min(255));
            }
        }
        v
    }

    // ── test 1: lossless greyscale gradient (32×24) ───────────────────────────

    #[test]
    fn roundtrip_lossless_greyscale_gradient() {
        let (w, h) = (32usize, 24usize);
        let samples = gradient_plane(w, h);

        let enc = JpegLsEncoder::new(JpegLsEncoderConfig::greyscale(w as u32, h as u32, 8))
            .expect("encoder config");
        let bytes = enc.encode_greyscale(&samples).expect("encode");

        assert!(
            JpegLsDecoder::is_jpegls(&bytes),
            "output must be detected as JPEG-LS"
        );
        let decoded = JpegLsDecoder::decode(&bytes).expect("decode");

        assert_eq!(decoded.width, w as u32);
        assert_eq!(decoded.height, h as u32);
        assert_eq!(decoded.num_components, 1);
        assert_eq!(decoded.precision, 8);
        assert_eq!(decoded.samples.len(), 1);

        for (i, (&orig, &rec)) in samples.iter().zip(decoded.samples[0].iter()).enumerate() {
            assert_eq!(
                rec, orig,
                "pixel {i}: lossless mismatch (got {rec}, want {orig})"
            );
        }
    }

    // ── test 2: constant grey 16×16 (flat-region path) ────────────────────────

    #[test]
    fn roundtrip_lossless_16x16_constant() {
        let (w, h) = (16usize, 16usize);
        let fill = 200u16;
        let samples = vec![fill; w * h];

        let enc = JpegLsEncoder::new(JpegLsEncoderConfig::greyscale(w as u32, h as u32, 8))
            .expect("encoder config");
        let bytes = enc.encode_greyscale(&samples).expect("encode");
        let decoded = JpegLsDecoder::decode(&bytes).expect("decode");

        assert_eq!(decoded.samples.len(), 1);
        for (i, &rec) in decoded.samples[0].iter().enumerate() {
            assert_eq!(
                rec, fill,
                "pixel {i}: constant region mismatch (got {rec}, want {fill})"
            );
        }
    }

    // ── test 3: near-lossless NEAR=2 within bound ─────────────────────────────

    #[test]
    fn roundtrip_near_lossless_near2() {
        let (w, h) = (24usize, 18usize);
        let near = 2u8;
        // A source with sharp transitions to exercise quantised residuals.
        let mut samples = Vec::with_capacity(w * h);
        for row in 0..h {
            for col in 0..w {
                let v = ((row * 17 + col * 23 + 5) % 256) as u16;
                samples.push(v);
            }
        }

        let cfg = JpegLsEncoderConfig::greyscale(w as u32, h as u32, 8).with_near(near);
        let enc = JpegLsEncoder::new(cfg).expect("encoder config");
        let bytes = enc.encode_greyscale(&samples).expect("encode");
        let decoded = JpegLsDecoder::decode(&bytes).expect("decode");

        assert_eq!(decoded.samples.len(), 1);
        for (i, (&orig, &rec)) in samples.iter().zip(decoded.samples[0].iter()).enumerate() {
            let diff = (orig as i32 - rec as i32).abs();
            assert!(
                diff <= near as i32,
                "pixel {i}: |{orig} - {rec}| = {diff} exceeds NEAR={near}"
            );
        }
    }

    // ── shared RGB fixture ────────────────────────────────────────────────────

    fn rgb_planes(w: usize, h: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let n = w * h;
        let r: Vec<u16> = (0..n).map(|i| ((i * 17 + 3) % 256) as u16).collect();
        let g: Vec<u16> = (0..n).map(|i| ((i * 31 + 50) % 256) as u16).collect();
        let b: Vec<u16> = (0..n).map(|i| ((i * 7 + 120) % 256) as u16).collect();
        (r, g, b)
    }

    fn assert_rgb_lossless(
        decoded: &oximedia_codec::jpegls::DecodedImage,
        r: &[u16],
        g: &[u16],
        b: &[u16],
        label: &str,
    ) {
        assert_eq!(decoded.num_components, 3, "{label}: component count");
        assert_eq!(decoded.samples.len(), 3, "{label}: plane count");
        for (ci, expected) in [r, g, b].iter().enumerate() {
            for (pi, (&orig, &rec)) in expected.iter().zip(decoded.samples[ci].iter()).enumerate() {
                assert_eq!(
                    rec, orig,
                    "{label} comp {ci} pixel {pi}: expected {orig}, got {rec}"
                );
            }
        }
    }

    // ── test 4: multicomponent RGB, ILV=0 ─────────────────────────────────────

    #[test]
    fn roundtrip_multicomponent_rgb_ilv0() {
        let (w, h) = (8usize, 6usize);
        let (r, g, b) = rgb_planes(w, h);

        let cfg = JpegLsEncoderConfig::multicomponent(w as u32, h as u32, 3, 8, 0);
        let enc = JpegLsEncoder::new(cfg).expect("encoder config");
        let bytes = enc.encode_planes(&[&r, &g, &b]).expect("encode");
        let decoded = JpegLsDecoder::decode(&bytes).expect("decode");

        assert_eq!(decoded.width, w as u32);
        assert_eq!(decoded.height, h as u32);
        assert_rgb_lossless(&decoded, &r, &g, &b, "ILV=0");
    }

    // ── test 5: multicomponent RGB, ILV=1 (line-interleaved) ──────────────────

    #[test]
    fn roundtrip_multicomponent_rgb_ilv1() {
        let (w, h) = (8usize, 6usize);
        let (r, g, b) = rgb_planes(w, h);

        let cfg = JpegLsEncoderConfig::multicomponent(w as u32, h as u32, 3, 8, 1);
        let enc = JpegLsEncoder::new(cfg).expect("encoder config");
        let bytes = enc.encode_planes(&[&r, &g, &b]).expect("encode");
        let decoded = JpegLsDecoder::decode(&bytes).expect("decode");

        assert_rgb_lossless(&decoded, &r, &g, &b, "ILV=1");
    }

    // ── test 6: multicomponent RGB, ILV=2 (sample-interleaved) ────────────────

    #[test]
    fn roundtrip_multicomponent_rgb_ilv2() {
        let (w, h) = (5usize, 4usize);
        let (r, g, b) = rgb_planes(w, h);

        let cfg = JpegLsEncoderConfig::multicomponent(w as u32, h as u32, 3, 8, 2);
        let enc = JpegLsEncoder::new(cfg).expect("encoder config");
        let bytes = enc.encode_planes(&[&r, &g, &b]).expect("encode");
        let decoded = JpegLsDecoder::decode(&bytes).expect("decode");

        assert_rgb_lossless(&decoded, &r, &g, &b, "ILV=2");
    }

    // ── extra: 12-bit lossless single-row exercises overflow escape ───────────

    #[test]
    fn roundtrip_lossless_12bit_gradient() {
        let (w, h) = (16usize, 4usize);
        let samples: Vec<u16> = (0..(w * h) as u16).map(|i| (i * 257) % 4096).collect();

        let enc = JpegLsEncoder::new(JpegLsEncoderConfig::greyscale(w as u32, h as u32, 12))
            .expect("encoder config");
        let bytes = enc.encode_greyscale(&samples).expect("encode");
        let decoded = JpegLsDecoder::decode(&bytes).expect("decode");

        assert_eq!(decoded.precision, 12);
        for (i, (&orig, &rec)) in samples.iter().zip(decoded.samples[0].iter()).enumerate() {
            assert_eq!(
                rec, orig,
                "pixel {i}: 12-bit lossless mismatch (got {rec}, want {orig})"
            );
        }
    }

    // ── extra: 1×1 single pixel edge case ─────────────────────────────────────

    #[test]
    fn roundtrip_single_pixel() {
        let enc = JpegLsEncoder::new(JpegLsEncoderConfig::greyscale(1, 1, 8)).expect("config");
        let bytes = enc.encode_greyscale(&[123u16]).expect("encode");
        let decoded = JpegLsDecoder::decode(&bytes).expect("decode");
        assert_eq!(decoded.samples[0][0], 123);
    }

    // ── extra: u8 convenience API round-trips ─────────────────────────────────

    #[test]
    fn roundtrip_u8_helper_ilv2() {
        let (w, h) = (4usize, 4usize);
        let n = w * h;
        let r: Vec<u8> = (0..n).map(|i| (i * 9 + 1) as u8).collect();
        let g: Vec<u8> = (0..n).map(|i| (i * 5 + 40) as u8).collect();
        let b: Vec<u8> = (0..n).map(|i| (i * 3 + 80) as u8).collect();

        let cfg = JpegLsEncoderConfig::multicomponent(w as u32, h as u32, 3, 8, 2);
        let enc = JpegLsEncoder::new(cfg).expect("config");
        let bytes = enc.encode_planes_u8(&[&r, &g, &b]).expect("encode");
        let decoded = JpegLsDecoder::decode(&bytes).expect("decode");

        for (ci, expected) in [&r, &g, &b].iter().enumerate() {
            for (pi, (&orig, &rec)) in expected.iter().zip(decoded.samples[ci].iter()).enumerate() {
                assert_eq!(rec, orig as u16, "u8 comp {ci} pixel {pi}");
            }
        }
    }
}
