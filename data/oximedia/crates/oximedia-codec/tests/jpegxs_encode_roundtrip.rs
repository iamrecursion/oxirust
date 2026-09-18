// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Integration tests for the JPEG XS **encoder** (`oximedia-codec` feature
//! `jpegxs`). Each test exercises the forward path and, where applicable,
//! decodes the result with the existing `JpegXsDecoder` to verify the encoder
//! and decoder agree.

#![cfg(feature = "jpegxs")]

use oximedia_codec::jpegxs::bitreader::BitReader;
use oximedia_codec::jpegxs::bitwriter::BitWriter;
use oximedia_codec::jpegxs::decoder::JpegXsDecoder;
use oximedia_codec::jpegxs::entropy::decode_subband;
use oximedia_codec::jpegxs::marker_write::{
    write_cdt, write_eoc, write_pih, write_soc, CdtComponent, PihFields,
};
use oximedia_codec::jpegxs::markers::{parse_headers, PROFILE_MAIN};
use oximedia_codec::jpegxs::vlc::{default_magnitude_table, default_run_table};
use oximedia_codec::jpegxs::vlc_encode::encode_subband;
use oximedia_codec::jpegxs::wavelet::{forward_wavelet_2d, inverse_53_2d};
use oximedia_codec::jpegxs::{JpegXsEncoder, JpegXsEncoderConfig};

// ── markers_pih_write_then_parse ────────────────────────────────────────────

#[test]
fn markers_pih_write_then_parse() {
    // Write a PIH (16x16, Bw=8, Nc=3, Csp=YUV), parse with the existing parser,
    // and assert all fields are equal.
    let mut buf = Vec::new();
    write_soc(&mut buf);
    let pih = PihFields {
        codestream_len: 0,
        profile: PROFILE_MAIN,
        level: 0,
        width: 16,
        height: 16,
        codegroup_width: 16,
        slice_height: 16,
        num_components: 3,
        ganging: 0,
        bit_depth: 8,
        bw_ext: 0,
        fq: 0,
        bitrate: 0,
        fsl: 0,
        ppoc: 0,
        cpih: 0,
    };
    write_pih(&mut buf, &pih).expect("write_pih");
    write_cdt(
        &mut buf,
        &[CdtComponent {
            bit_depth: 8,
            sx: 1,
            sy: 1,
        }; 3],
    )
    .expect("write_cdt");
    write_eoc(&mut buf);

    let (headers, _) = parse_headers(&buf).expect("parse_headers");
    assert_eq!(headers.pih.width, 16);
    assert_eq!(headers.pih.height, 16);
    assert_eq!(headers.pih.slice_height, 16);
    assert_eq!(headers.pih.num_components, 3);
    assert_eq!(headers.pih.bit_depth, 8);
    assert_eq!(headers.pih.profile, PROFILE_MAIN);
    assert_eq!(headers.pih.level, 0);
    assert_eq!(headers.components.len(), 3);
    for c in &headers.components {
        assert_eq!(c.bit_depth, 8);
        assert_eq!(c.sx, 1);
        assert_eq!(c.sy, 1);
    }
}

// ── wavelet_53_forward_inverse_roundtrip ────────────────────────────────────

#[test]
fn wavelet_53_forward_inverse_roundtrip() {
    // Gradient + pseudo-random i32 signal: forward then inverse 5/3 == identity,
    // byte-exact, across several dimensions (even, odd, rectangular).
    let cases = [(8usize, 8usize), (7, 5), (32, 16), (15, 11)];
    for &(w, h) in &cases {
        // Gradient.
        let grad: Vec<i32> = (0..w * h)
            .map(|i| (2 * (i % w) + 3 * (i / w)) as i32)
            .collect();
        let (ll, hl, lh, hh) = forward_wavelet_2d(&grad, w, h).expect("forward");
        let rec = inverse_53_2d(&ll, &hl, &lh, &hh, w, h).expect("inverse");
        assert_eq!(rec, grad, "gradient {w}x{h} forward→inverse not identity");

        // Deterministic pseudo-random.
        let mut state: u32 = 0xDEAD_BEEF ^ (w as u32) ^ ((h as u32) << 16);
        let rnd: Vec<i32> = (0..w * h)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (((state >> 16) & 0x3FF) as i32) - 200
            })
            .collect();
        let (ll, hl, lh, hh) = forward_wavelet_2d(&rnd, w, h).expect("forward");
        let rec = inverse_53_2d(&ll, &hl, &lh, &hh, w, h).expect("inverse");
        assert_eq!(rec, rnd, "random {w}x{h} forward→inverse not identity");
    }
}

// ── vlc_encode_decode_roundtrip ─────────────────────────────────────────────

#[test]
fn vlc_encode_decode_roundtrip() {
    // Every run length (0..=7 unary plus escape) and every magnitude level
    // (1..=8 unary plus escape) must encode then decode back through the real
    // decoder tables.
    let run_t = default_run_table();
    let mag_t = default_magnitude_table();

    // Build a subband that contains, in sequence: each run length 0..=10 each
    // followed by a non-zero coefficient covering each magnitude level 1..=12,
    // with alternating signs. This exercises both unary and escape paths.
    let mut coeffs: Vec<i32> = Vec::new();
    let mut level = 1i32;
    let mut sign = 1i32;
    for run in 0u32..=10 {
        for _ in 0..run {
            coeffs.push(0);
        }
        coeffs.push(sign * level);
        level = if level >= 12 { 1 } else { level + 1 };
        sign = -sign;
    }

    let n = coeffs.len();
    let mut w = BitWriter::new();
    encode_subband(&mut w, &coeffs);
    let bytes = w.finish();

    let mut r = BitReader::new(&bytes);
    let decoded = decode_subband(&mut r, &run_t, &mag_t, n, 1).expect("decode_subband");
    assert_eq!(decoded.coeffs, coeffs, "VLC encode→decode mismatch");

    // Exhaustively cover each magnitude level 1..=20 individually (positive and
    // negative), each as a lone non-zero at index 0 of a 1-coefficient subband.
    for level in 1i32..=20 {
        for &val in &[level, -level] {
            let mut w = BitWriter::new();
            encode_subband(&mut w, &[val]);
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            let d = decode_subband(&mut r, &run_t, &mag_t, 1, 1).expect("decode");
            assert_eq!(d.coeffs[0], val, "single-coeff level {val} round-trip");
        }
    }
}

// ── roundtrip_gradient_32x16_unit_weights ───────────────────────────────────

#[test]
fn roundtrip_gradient_32x16_unit_weights() {
    // Encode a gradient with unit weights, decode with the existing decoder,
    // assert byte-exact (5/3 is lossless). Single component because the project
    // decoder replicates the reconstructed plane across all components.
    let (w, h) = (32u32, 16u32);
    let mut cfg = JpegXsEncoderConfig::new(w, h, 8, 1);
    cfg.weights = vec![1, 1, 1, 1]; // explicit unit weights → WGT marker emitted

    let enc = JpegXsEncoder::new(cfg).expect("encoder");
    // A genuine 2D gradient over [0, 255].
    let plane: Vec<i32> = (0..(w * h) as usize)
        .map(|i| {
            let x = (i % w as usize) as i32;
            let y = (i / w as usize) as i32;
            (x * 7 + y * 5) % 256
        })
        .collect();

    let stream = enc.encode(std::slice::from_ref(&plane)).expect("encode");
    assert!(JpegXsDecoder::is_jpegxs(&stream));

    let img = JpegXsDecoder::decode(&stream).expect("decode");
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.num_components, 1);
    let decoded: Vec<i32> = img.samples[0].iter().map(|&v| v as i32).collect();
    assert_eq!(decoded, plane, "32x16 gradient must round-trip byte-exact");
}

// ── roundtrip_constant_grey_16x16_yuv422 ────────────────────────────────────

#[test]
fn roundtrip_constant_grey_16x16_yuv422() {
    // Encode a constant-grey 16x16 YUV (3-component) image, decode, and assert
    // the result is within ±2 LSB. (With unit weights it is in fact exact.)
    let grey = 137i32;
    let cfg = JpegXsEncoderConfig::new(16, 16, 8, 3);
    assert_eq!(
        cfg.color_space,
        oximedia_codec::jpegxs::encoder::JxsColorSpace::Yuv
    );

    let enc = JpegXsEncoder::new(cfg).expect("encoder");
    let plane = vec![grey; 16 * 16];
    let planes = vec![plane.clone(), plane.clone(), plane.clone()];

    let stream = enc.encode(&planes).expect("encode");
    let img = JpegXsDecoder::decode(&stream).expect("decode");
    assert_eq!(img.num_components, 3);
    for (c, comp) in img.samples.iter().enumerate() {
        assert_eq!(comp.len(), 256);
        for (i, &s) in comp.iter().enumerate() {
            let diff = (i32::from(s) - grey).abs();
            assert!(
                diff <= 2,
                "component {c} sample {i}: decoded {s} differs from {grey} by {diff} (> 2 LSB)"
            );
        }
    }
}

// ── extra: end-to-end through the public encoder/decoder API ────────────────

#[test]
fn roundtrip_random_64x32_lossless() {
    // Larger pseudo-random single-component frame, byte-exact lossless.
    let (w, h) = (64u32, 32u32);
    let cfg = JpegXsEncoderConfig::new(w, h, 8, 1);
    let enc = JpegXsEncoder::new(cfg).expect("encoder");

    let mut state: u32 = 0x0BAD_F00D;
    let plane: Vec<i32> = (0..(w * h) as usize)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 24) & 0xFF) as i32
        })
        .collect();

    let stream = enc.encode(std::slice::from_ref(&plane)).expect("encode");
    let img = JpegXsDecoder::decode(&stream).expect("decode");
    let decoded: Vec<i32> = img.samples[0].iter().map(|&v| v as i32).collect();
    assert_eq!(
        decoded, plane,
        "64x32 random frame must round-trip byte-exact"
    );
}

#[test]
fn roundtrip_odd_dimensions_lossless() {
    // Odd width and height exercise the asymmetric subband sizing.
    let (w, h) = (17u32, 9u32);
    let cfg = JpegXsEncoderConfig::new(w, h, 8, 1);
    let enc = JpegXsEncoder::new(cfg).expect("encoder");
    let plane: Vec<i32> = (0..(w * h) as usize)
        .map(|i| ((i * 11) % 200) as i32)
        .collect();
    let stream = enc.encode(std::slice::from_ref(&plane)).expect("encode");
    let img = JpegXsDecoder::decode(&stream).expect("decode");
    let decoded: Vec<i32> = img.samples[0].iter().map(|&v| v as i32).collect();
    assert_eq!(decoded, plane, "17x9 frame must round-trip byte-exact");
}
