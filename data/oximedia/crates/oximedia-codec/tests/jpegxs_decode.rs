// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Integration tests for the JPEG XS decoder (`oximedia-codec` feature `jpegxs`).

#[cfg(feature = "jpegxs")]
mod tests {
    use oximedia_codec::jpegxs::decoder::JpegXsDecoder;
    use oximedia_codec::jpegxs::markers::{
        build_test_codestream, build_test_codestream_with_nlt, parse_headers, EOC, SOC,
    };

    // ── Marker / stream detection ─────────────────────────────────────────────

    #[test]
    fn soc_marker_detection() {
        let data = [0xFF, 0x10, 0x00, 0x00];
        assert!(JpegXsDecoder::is_jpegxs(&data));
    }

    #[test]
    fn non_jpegxs_stream_rejected() {
        let data = [0xFF, 0xD8, 0xFF, 0xE0]; // JPEG SOI
        assert!(!JpegXsDecoder::is_jpegxs(&data));
    }

    #[test]
    fn av1_stream_rejected() {
        let data = [0x0A, 0x0B, 0x00, 0x00]; // not JPEG XS
        assert!(!JpegXsDecoder::is_jpegxs(&data));
    }

    // ── Error cases ───────────────────────────────────────────────────────────

    #[test]
    fn empty_stream_rejected() {
        let result = JpegXsDecoder::decode(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn truncated_stream_rejected() {
        let data = [0xFF, 0x10]; // SOC but nothing else
        let result = JpegXsDecoder::decode(&data);
        assert!(result.is_err());
    }

    #[test]
    fn wrong_magic_rejected() {
        let data = [0xDE, 0xAD, 0xBE, 0xEF];
        let result = JpegXsDecoder::decode(&data);
        assert!(result.is_err());
    }

    // ── Header parsing ────────────────────────────────────────────────────────

    #[test]
    fn parse_minimal_640x480_3comp_8bit() {
        let data = build_test_codestream(640, 480, 16, 3, 8);
        let (headers, _end) = parse_headers(&data).expect("parse_headers");
        assert_eq!(headers.pih.width, 640);
        assert_eq!(headers.pih.height, 480);
        assert_eq!(headers.pih.slice_height, 16);
        assert_eq!(headers.pih.num_components, 3);
        assert_eq!(headers.pih.bit_depth, 8);
    }

    #[test]
    fn parse_1920x1080_1comp_10bit() {
        let data = build_test_codestream(1920, 1080, 32, 1, 10);
        let (headers, _) = parse_headers(&data).expect("parse_headers");
        assert_eq!(headers.pih.width, 1920);
        assert_eq!(headers.pih.height, 1080);
        assert_eq!(headers.pih.num_components, 1);
        assert_eq!(headers.pih.bit_depth, 10);
    }

    #[test]
    fn parsed_components_match_num_components() {
        let data = build_test_codestream(64, 64, 8, 4, 8);
        let (headers, _) = parse_headers(&data).expect("parse_headers");
        assert_eq!(headers.components.len(), 4);
        for comp in &headers.components {
            assert_eq!(comp.sx, 1);
            assert_eq!(comp.sy, 1);
        }
    }

    #[test]
    fn soc_constant_value() {
        assert_eq!(SOC, 0xFF10u16);
    }

    #[test]
    fn eoc_constant_value() {
        assert_eq!(EOC, 0xFF11u16);
    }

    // ── Decode ────────────────────────────────────────────────────────────────

    #[test]
    fn decode_headers_only_returns_zero_image() {
        let data = build_test_codestream(8, 8, 8, 1, 8);
        let img = JpegXsDecoder::decode(&data).expect("decode");
        assert_eq!(img.width, 8);
        assert_eq!(img.height, 8);
        assert_eq!(img.num_components, 1);
        assert_eq!(img.bit_depth, 8);
        assert_eq!(img.samples[0].len(), 64);
    }

    #[test]
    fn decode_multicomponent_correct_plane_count() {
        let data = build_test_codestream(16, 16, 16, 3, 8);
        let img = JpegXsDecoder::decode(&data).expect("decode");
        assert_eq!(img.samples.len(), 3);
        for plane in &img.samples {
            assert_eq!(plane.len(), 256);
        }
    }

    #[test]
    fn decoded_sample_values_within_8bit_range() {
        let data = build_test_codestream(4, 4, 4, 1, 8);
        let img = JpegXsDecoder::decode(&data).expect("decode");
        for &s in &img.samples[0] {
            assert!(s <= 255, "sample {s} exceeds 8-bit max");
        }
    }

    #[test]
    fn decoded_sample_values_within_10bit_range() {
        let data = build_test_codestream(4, 4, 4, 1, 10);
        let img = JpegXsDecoder::decode(&data).expect("decode");
        let max_val = (1u16 << 10) - 1; // 1023
        for &s in &img.samples[0] {
            assert!(s <= max_val, "sample {s} exceeds 10-bit max");
        }
    }

    #[test]
    fn decode_with_nlt_quadratic_marker_succeeds() {
        // Codestream with NLT quadratic marker (T1=64, T2=192) — no slice data,
        // so output is all-zero. The NLT reverse of 0 (which is ≤ T1) is 0 (identity).
        let data = build_test_codestream_with_nlt(8, 8, 8, 1, 8, 64, 192);
        let img = JpegXsDecoder::decode(&data).expect("decode with NLT should succeed");
        assert_eq!(img.width, 8);
        assert_eq!(img.height, 8);
        assert_eq!(img.num_components, 1);
        assert_eq!(img.bit_depth, 8);
        // All-zero output (no slice data), low region → identity, so all zeros.
        assert!(img.samples[0].iter().all(|&v| v == 0));
    }

    #[test]
    fn decode_with_nlt_marker_sample_values_in_range() {
        // Even with an NLT marker, output must be within bit-depth range.
        let data = build_test_codestream_with_nlt(4, 4, 4, 1, 8, 32, 200);
        let img = JpegXsDecoder::decode(&data).expect("decode");
        for &s in &img.samples[0] {
            assert!(s <= 255, "sample {s} exceeds 8-bit max after NLT reverse");
        }
    }

    // ── BitReader ─────────────────────────────────────────────────────────────

    #[test]
    fn bitreader_basic() {
        use oximedia_codec::jpegxs::bitreader::BitReader;
        let data = [0b1010_0011u8, 0b1111_0000u8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bit().unwrap(), 1);
        assert_eq!(r.read_bit().unwrap(), 0);
        assert_eq!(r.read_bit().unwrap(), 1);
        assert_eq!(r.read_bit().unwrap(), 0);
        // Next 4 bits: 0011
        assert_eq!(r.read_bits_u32(4).unwrap(), 0b0011);
    }

    #[test]
    fn bitreader_read_u16_be() {
        use oximedia_codec::jpegxs::bitreader::BitReader;
        let data = [0x12u8, 0x34u8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_u16_be().unwrap(), 0x1234);
    }

    #[test]
    fn bitreader_truncated_error() {
        use oximedia_codec::jpegxs::bitreader::BitReader;
        let data = [0xFFu8];
        let mut r = BitReader::new(&data);
        let _ = r.read_bits_u32(8).unwrap();
        assert!(r.read_bit().is_err());
    }

    // ── Wavelet ───────────────────────────────────────────────────────────────

    #[test]
    fn wavelet_53_2d_constant() {
        use oximedia_codec::jpegxs::wavelet::{forward_53_1d, inverse_53_2d};

        let n = 4usize;
        let constant = 64i32;

        let n_low_w = (n + 1) / 2;
        let n_high_w = n / 2;
        let n_low_h = (n + 1) / 2;
        let n_high_h = n / 2;

        // Build true LL via forward transform of constant image.
        let image = vec![constant; n * n];
        let mut row_lows = vec![vec![0i32; n_low_w]; n];
        let mut row_highs = vec![vec![0i32; n_high_w]; n];
        for row in 0..n {
            let (l, h) = forward_53_1d(&image[row * n..(row + 1) * n]);
            row_lows[row] = l;
            row_highs[row] = h;
        }
        let mut ll = vec![0i32; n_low_h * n_low_w];
        let mut hl = vec![0i32; n_low_h * n_high_w];
        for col in 0..n_low_w {
            let col_vals: Vec<i32> = (0..n).map(|r| row_lows[r][col]).collect();
            let (l, _) = forward_53_1d(&col_vals);
            for (r, &v) in l.iter().enumerate() {
                ll[r * n_low_w + col] = v;
            }
        }
        for col in 0..n_high_w {
            let col_vals: Vec<i32> = (0..n).map(|r| row_highs[r][col]).collect();
            let (l, _) = forward_53_1d(&col_vals);
            for (r, &v) in l.iter().enumerate() {
                hl[r * n_high_w + col] = v;
            }
        }
        let lh = vec![0i32; n_high_h * n_low_w];
        let hh = vec![0i32; n_high_h * n_high_w];

        let result = inverse_53_2d(&ll, &hl, &lh, &hh, n, n).unwrap();
        assert_eq!(result.len(), n * n);
        for &v in &result {
            assert_eq!(v, constant, "expected all samples to be {constant}");
        }
    }

    // ── NLT ──────────────────────────────────────────────────────────────────

    #[test]
    fn nlt_none_is_identity() {
        use oximedia_codec::jpegxs::nlt::{apply_nlt_reverse, NltParams};
        let mut s = vec![10i32, 20, 30];
        apply_nlt_reverse(&mut s, &NltParams::none(), 8).unwrap();
        assert_eq!(s, [10, 20, 30]);
    }

    #[test]
    fn nlt_quadratic_low_region_identity() {
        use oximedia_codec::jpegxs::nlt::{apply_nlt_reverse, NltParams};
        // s = 50 < T1 = 64 → low region → identity
        let mut s = vec![50i32];
        apply_nlt_reverse(&mut s, &NltParams::quadratic(64, 192), 8).unwrap();
        assert_eq!(s[0], 50);
    }

    #[test]
    fn nlt_quadratic_succeeds_for_valid_params() {
        use oximedia_codec::jpegxs::nlt::{apply_nlt_reverse, NltParams};
        // Verify the quadratic reverse transform no longer returns an error.
        let mut s = vec![100i32, 150, 200];
        let result = apply_nlt_reverse(&mut s, &NltParams::quadratic(64, 192), 8);
        assert!(result.is_ok(), "expected Ok, got {result:?}");
    }

    #[test]
    fn nlt_quadratic_invalid_params_returns_error() {
        use oximedia_codec::jpegxs::nlt::{apply_nlt_reverse, NltParams};
        // T1 == T2 is invalid; must return InvalidHeader.
        let mut s = vec![100i32];
        let result = apply_nlt_reverse(&mut s, &NltParams::quadratic(128, 128), 8);
        assert!(result.is_err());
    }

    // ── VLC tables ────────────────────────────────────────────────────────────

    #[test]
    fn vlc_run_table_run0_is_one_zero_bit() {
        use oximedia_codec::jpegxs::vlc::default_run_table;
        let table = default_run_table();
        // Top bit = 0 → run = 0, consumed = 1
        let result = table.lookup(0x0000_0000u32);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.value, 0);
        assert_eq!(r.bits_consumed, 1);
    }

    #[test]
    fn vlc_magnitude_table_level1_is_one_zero_bit() {
        use oximedia_codec::jpegxs::vlc::default_magnitude_table;
        let table = default_magnitude_table();
        let result = table.lookup(0x0000_0000u32);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.value, 1);
        assert_eq!(r.bits_consumed, 1);
    }

    #[test]
    fn vlc_significance_table_codes_zero_and_one() {
        use oximedia_codec::jpegxs::vlc::default_significance_table;
        let table = default_significance_table();
        let r0 = table.lookup(0x0000_0000u32).unwrap();
        assert_eq!(r0.value, 0);
        let r1 = table.lookup(0x8000_0000u32).unwrap();
        assert_eq!(r1.value, 1);
    }
}
