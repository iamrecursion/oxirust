//! JPEG-LS §A.7 RUN-mode round-trip integration tests (Wave 10 Slice 3).
//!
//! These tests exercise the RUN-mode path added in Wave 10:
//!
//! - Flat regions (lossless): all three raw gradients vanish, so the
//!   encoder enters RUN mode and emits compact length tokens drawn from
//!   `J[]`.  The corresponding decoder reads them back and reconstructs
//!   the constant region byte-exactly.
//! - Flat regions (near-lossless, NEAR > 0): the entry condition
//!   widens to `|d_i| <= NEAR`.  The decoded image is within `±NEAR`.
//! - Mixed regions: stripes and column transitions force the encoder to
//!   emit a `0` interruption bit, the residual length, and the
//!   termination sample under the right RUN-interruption context
//!   (365 for `Ra == Rb`, 366 for `Ra != Rb`).
//! - Multi-component: each component keeps its own `RunState` and its
//!   own context arrays.
//! - Zero-length runs at line start: the first pixel of each new row
//!   often breaks the entry condition because the top row has different
//!   neighbour values.  This validates that the encoder/decoder reset
//!   `run_index = 0` at the start of every line.
//!
//! All assertions are deterministic: the production encoder is paired
//! with the production decoder and both must round-trip byte-exact
//! (lossless) or within `±NEAR` (near-lossless).

#[cfg(feature = "jpegls")]
mod jpegls_runmode_tests {
    use oximedia_codec::jpegls::{JpegLsDecoder, JpegLsEncoder, JpegLsEncoderConfig};

    /// Helper: build a constant-colour single-component greyscale plane.
    fn constant_plane(w: usize, h: usize, fill: u16) -> Vec<u16> {
        vec![fill; w * h]
    }

    /// Helper: encode → decode lossless greyscale and return both the
    /// encoded byte length and the decoded samples.
    fn roundtrip_lossless_greyscale(
        samples: &[u16],
        w: u32,
        h: u32,
        bit_depth: u8,
    ) -> (usize, Vec<u16>) {
        let cfg = JpegLsEncoderConfig::greyscale(w, h, bit_depth);
        let enc = JpegLsEncoder::new(cfg).expect("encoder config");
        let bytes = enc.encode_greyscale(samples).expect("encode");
        let decoded = JpegLsDecoder::decode(&bytes).expect("decode");
        assert_eq!(decoded.samples.len(), 1, "single component expected");
        (bytes.len(), decoded.samples[0].clone())
    }

    // ── test 1: constant 32×32 lossless — RUN mode shrinks the output ────────

    #[test]
    fn roundtrip_constant_32x32() {
        let (w, h) = (32u32, 32u32);
        let samples = constant_plane(w as usize, h as usize, 128);
        let (encoded_len, decoded) = roundtrip_lossless_greyscale(&samples, w, h, 8);

        // Pixel round-trip is byte-exact.
        for (i, &px) in decoded.iter().enumerate() {
            assert_eq!(px, 128, "pixel {i} decoded as {px}, expected 128");
        }

        // Wave 8 regular-mode-only would have used one regular-mode
        // Golomb code per sample (~6 bits/sample after `k` ramps up), so
        // 32*32 = 1024 samples ⇒ ~6144 scan bits ⇒ ~770 byte scan plus
        // ~30 bytes of headers ⇒ ~800 bytes overall.  RUN mode collapses
        // the flat rows into a handful of length tokens per row (~12
        // bits/row), shrinking the stream by an order of magnitude.
        // Row 0 is more expensive because its seeded `runval = 0`
        // disagrees with the source `128`, forcing one termination
        // sample with a fat overflow-Golomb code (~41 bits); even so
        // the encoded length must be well under 200 bytes.
        assert!(
            encoded_len < 200,
            "RUN-mode encoded length {encoded_len} ≥ 200 — RUN mode not active?"
        );
    }

    // ── test 2: alternating stripes — termination at row boundaries ──────────

    #[test]
    fn roundtrip_stripes_32x32() {
        let (w, h) = (32usize, 32usize);
        let mut samples = vec![0u16; w * h];
        for row in 0..h {
            let fill = if row % 2 == 0 { 100 } else { 150 };
            for col in 0..w {
                samples[row * w + col] = fill;
            }
        }
        let (_encoded_len, decoded) = roundtrip_lossless_greyscale(&samples, w as u32, h as u32, 8);

        // Lossless: every pixel must round-trip byte-exact.  Each row is
        // flat (so RUN mode triggers within the row), but the value
        // alternates between rows — the run terminates at the row
        // boundary.  Tests both the EOL `1`-bit emission and the
        // line-start reset of `run_index`.
        for (i, (&src, &rec)) in samples.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(rec, src, "stripes pixel {i}: expected {src}, got {rec}");
        }
    }

    // ── test 3: column transitions — RUN-interruption ctx 366 (Ra != Rb) ─────

    #[test]
    fn roundtrip_two_color_columns() {
        let (w, h) = (32usize, 32usize);
        let mut samples = vec![0u16; w * h];
        // Two halves: left 16 columns = 80, right 16 columns = 200.  Most
        // of the interior pixels have Ra != Rb at the column boundary
        // (Ra = 80 from left, Rb = 200 from above — only when crossing
        // the boundary), exercising ctx 366.
        for row in 0..h {
            for col in 0..w {
                samples[row * w + col] = if col < 16 { 80 } else { 200 };
            }
        }
        let (_encoded_len, decoded) = roundtrip_lossless_greyscale(&samples, w as u32, h as u32, 8);
        for (i, (&src, &rec)) in samples.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(
                rec, src,
                "two-color column pixel {i}: expected {src}, got {rec}"
            );
        }
    }

    // ── test 4: near-lossless NEAR=2 constant 24×24 ──────────────────────────

    #[test]
    fn roundtrip_near_lossless_constant_24x24() {
        let (w, h) = (24u32, 24u32);
        let fill = 200u16;
        let samples = constant_plane(w as usize, h as usize, fill);
        let near = 2u8;

        let cfg = JpegLsEncoderConfig::greyscale(w, h, 8).with_near(near);
        let enc = JpegLsEncoder::new(cfg).expect("encoder config");
        let bytes = enc.encode_greyscale(&samples).expect("encode");
        let decoded = JpegLsDecoder::decode(&bytes).expect("decode");
        assert_eq!(decoded.samples.len(), 1);

        for (i, &px) in decoded.samples[0].iter().enumerate() {
            let diff = (px as i32 - fill as i32).abs();
            assert!(
                diff <= near as i32,
                "pixel {i}: |{px} - {fill}| = {diff} > NEAR={near}"
            );
        }
    }

    // ── test 5: interleaved (ILV=1) RGB constant — per-component RUN state ──

    #[test]
    fn roundtrip_ilv1_rgb_constant() {
        let (w, h) = (16u32, 16u32);
        let r_plane = constant_plane(w as usize, h as usize, 90);
        let g_plane = constant_plane(w as usize, h as usize, 130);
        let b_plane = constant_plane(w as usize, h as usize, 200);

        let cfg = JpegLsEncoderConfig::multicomponent(w, h, 3, 8, 1);
        let enc = JpegLsEncoder::new(cfg).expect("encoder config");
        let bytes = enc
            .encode_planes(&[&r_plane, &g_plane, &b_plane])
            .expect("encode");
        let decoded = JpegLsDecoder::decode(&bytes).expect("decode");

        assert_eq!(decoded.num_components, 3);
        for (i, &px) in decoded.samples[0].iter().enumerate() {
            assert_eq!(px, 90, "R pixel {i}: got {px}");
        }
        for (i, &px) in decoded.samples[1].iter().enumerate() {
            assert_eq!(px, 130, "G pixel {i}: got {px}");
        }
        for (i, &px) in decoded.samples[2].iter().enumerate() {
            assert_eq!(px, 200, "B pixel {i}: got {px}");
        }
    }

    // ── test 6: zero-length run at line start (run_index reset) ──────────────

    #[test]
    fn roundtrip_zero_run_at_line_start() {
        // Pattern: first column is always 50, all other columns are 200
        // and copy from the previous row.  The first pixel of every row
        // after row 0 differs from `Rb` (above is 50, left is 200 from
        // wrap-around — but at col 0 there is no left neighbour, so
        // `a = recon[(row-1)*w] = 50`, which equals the source.  That
        // means d1 = d-b = 200-200 = 0 (d is wrap-replicate of b),
        // d2 = b-c = 50-50 = 0, d3 = c-a = 50-50 = 0 — RUN entry succeeds
        // and immediately yields run_length = 0 because column 0 sample is
        // 50 == runval=50, and column 1 is 200 != 50, so the run
        // terminates immediately at col 1.  This exercises a degenerate
        // residual-zero termination right at line start and confirms
        // run_index resets correctly.
        let (w, h) = (8usize, 4usize);
        let mut samples = vec![0u16; w * h];
        for row in 0..h {
            for col in 0..w {
                samples[row * w + col] = if col == 0 { 50 } else { 200 };
            }
        }
        let (_encoded_len, decoded) = roundtrip_lossless_greyscale(&samples, w as u32, h as u32, 8);
        for (i, (&src, &rec)) in samples.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(rec, src, "zero-run pixel {i}: expected {src}, got {rec}");
        }
    }

    // ── extra sanity: long run that spans many full tokens ───────────────────

    #[test]
    fn roundtrip_long_run_64x64() {
        // 64×64 constant region — at run_index=11 J[11]=3 ⇒ threshold 8;
        // each full token advances 8 samples.  64 samples per row means
        // 8 full-length-8 tokens, after which run_index = 19 ⇒ J[19]=6
        // ⇒ threshold 64; the next row starts fresh because of the
        // line-start reset.  Decoder must agree on every step.
        let (w, h) = (64u32, 64u32);
        let samples = constant_plane(w as usize, h as usize, 77);
        let (encoded_len, decoded) = roundtrip_lossless_greyscale(&samples, w, h, 8);

        for (i, &px) in decoded.iter().enumerate() {
            assert_eq!(px, 77, "long-run pixel {i}: got {px}");
        }
        // 64*64 = 4096 flat samples — RUN mode must compress to well
        // under 4096 / 4 bytes = 1024 bytes.  Each row encodes in
        // ~13 bits + a small per-row overhead; header ~30 bytes.
        assert!(
            encoded_len < 500,
            "long-run encoded length {encoded_len} ≥ 500 — RUN mode not collapsing flats?"
        );
    }

    // ── extra sanity: gradient-with-flat-tail still round-trips ──────────────

    #[test]
    fn roundtrip_gradient_then_flat() {
        // Left half is a sample-by-sample gradient (no RUN entry), right
        // half is constant (RUN entry triggers mid-row).  Exercises the
        // transition into RUN mode from a non-flat region in the middle
        // of a line.
        let (w, h) = (32usize, 8usize);
        let mut samples = vec![0u16; w * h];
        for row in 0..h {
            for col in 0..w {
                samples[row * w + col] = if col < 16 { (col as u16) * 8 } else { 180 };
            }
        }
        let (_encoded_len, decoded) = roundtrip_lossless_greyscale(&samples, w as u32, h as u32, 8);
        for (i, (&src, &rec)) in samples.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(
                rec, src,
                "gradient-then-flat pixel {i}: expected {src}, got {rec}"
            );
        }
    }
}
