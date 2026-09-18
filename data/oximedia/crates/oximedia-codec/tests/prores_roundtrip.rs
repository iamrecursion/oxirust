//! ProRes 422 encoder + decoder round-trip integration tests.
//!
//! These tests exercise the full `ProResEncoder → ProResDecoder` pipeline
//! for correctness. ProRes is visually lossless (not mathematically lossless),
//! so we verify that decoded samples are within a small tolerance of the input.

#![cfg(feature = "prores")]

use oximedia_codec::frame::{Plane, VideoFrame};
use oximedia_codec::prores::{
    ChromaFormat, ProResDecoder, ProResDecoderConfig, ProResEncoder, ProResEncoderConfig,
    ProResProfile,
};
use oximedia_codec::traits::{VideoDecoder, VideoEncoder};
use oximedia_core::PixelFormat;

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Build a flat-grey Yuv422p10le frame: every luma sample = `y_val`,
/// every chroma sample = 512 (10-bit neutral grey).
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

/// Encode a `VideoFrame` with the default `Standard` profile and return the
/// `'icpf'` packet bytes.
fn encode_frame(frame: &VideoFrame) -> Vec<u8> {
    let cfg = ProResEncoderConfig::new(ProResProfile::Standard, frame.width, frame.height);
    let mut enc = ProResEncoder::new(cfg).expect("encoder init");
    enc.send_frame(frame).expect("send_frame");
    enc.receive_packet()
        .expect("receive_packet")
        .expect("Some(packet)")
        .data
}

/// Encode with the given profile.
fn encode_frame_profile(frame: &VideoFrame, profile: ProResProfile) -> Vec<u8> {
    let cfg = ProResEncoderConfig::new(profile, frame.width, frame.height);
    let mut enc = ProResEncoder::new(cfg).expect("encoder init");
    enc.send_frame(frame).expect("send_frame");
    enc.receive_packet()
        .expect("receive_packet")
        .expect("Some(packet)")
        .data
}

// ─── Tests ────────────────────────────────────────────────────────────────────

/// Verify that `ProResDecoder::new()` and `ProResDecoder::default()` both
/// construct without error and produce the same observable behaviour.
#[test]
fn decoder_new_and_default() {
    let _d1 = ProResDecoder::new();
    let _d2 = ProResDecoder::default();
    // Both should complete without panic.
}

/// Verify that a truncated input (fewer than 8 bytes) returns `Err`.
#[test]
fn decode_rejects_too_short() {
    let result = ProResDecoder::decode(&[0u8; 3]);
    assert!(result.is_err(), "should reject < 8 bytes");

    // Exactly 7 bytes — still too short.
    let result2 = ProResDecoder::decode(&[0u8; 7]);
    assert!(result2.is_err(), "should reject 7-byte input");
}

/// Verify that a buffer with a wrong container tag ('XXXX' instead of 'icpf')
/// returns `Err`.
#[test]
fn decode_rejects_bad_container_tag() {
    // Construct a plausible-looking 32-byte buffer with a bad tag.
    let mut buf = vec![0u8; 32];
    buf[0] = 0;
    buf[1] = 0;
    buf[2] = 0;
    buf[3] = 32; // frame_size = 32
    buf[4] = b'X';
    buf[5] = b'X';
    buf[6] = b'X';
    buf[7] = b'X'; // tag = 'XXXX'

    let result = ProResDecoder::decode(&buf);
    assert!(result.is_err(), "bad container tag should be rejected");
}

/// Test that `ProResProfile::from_fourcc` maps every known FourCC correctly,
/// and returns `Err` for an unknown one.
#[test]
fn profile_from_fourcc() {
    use oximedia_codec::prores::frame::ProResProfile;

    let cases: &[(&[u8; 4], ProResProfile)] = &[
        (b"apco", ProResProfile::Proxy),
        (b"apcs", ProResProfile::Lt),
        (b"apcn", ProResProfile::Standard),
        (b"apch", ProResProfile::Hq),
        (b"ap4h", ProResProfile::P4444),
        (b"ap4x", ProResProfile::P4444Xq),
    ];
    for (fourcc, expected) in cases {
        let got = ProResProfile::from_fourcc(fourcc).expect("known FourCC");
        assert_eq!(got, *expected, "fourcc {:?}", fourcc);
    }
    // Unknown FourCC must be rejected.
    assert!(
        ProResProfile::from_fourcc(b"xxxx").is_err(),
        "unknown FourCC should return Err"
    );
}

/// Encode a constant-grey 64×16 frame and verify that decoded luma samples are
/// within ±4 LSB of the expected 8-bit value.
///
/// The tolerance of ±4 accounts for the quantization error inherent in the
/// ProRes lossy-but-visually-lossless codec.
#[test]
fn encode_decode_constant_grey() {
    // Choose a luma value in a range that exercises mid-range quantization.
    let y_val_10bit: u16 = 400;
    let frame = flat_grey_frame(64, 16, y_val_10bit);
    let pkt = encode_frame(&frame);

    let decoded = ProResDecoder::decode(&pkt).expect("decode constant-grey frame");

    // Structural checks.
    assert_eq!(decoded.width, 64, "width");
    assert_eq!(decoded.height, 16, "height");
    assert_eq!(decoded.profile, ProResProfile::Standard, "profile");
    assert!(!decoded.is_interlaced, "should be progressive");
    assert_eq!(decoded.y.len(), 64 * 16, "luma plane size");
    assert_eq!(decoded.cb.len(), 32 * 16, "Cb plane size");
    assert_eq!(decoded.cr.len(), 32 * 16, "Cr plane size");

    // Per-sample luma accuracy.
    let expected_y8 = (y_val_10bit >> 2) as i32; // ~100
    let tolerance = 4i32;
    for (i, &sample) in decoded.y.iter().enumerate() {
        let err = (sample as i32 - expected_y8).abs();
        assert!(
            err <= tolerance,
            "luma sample[{}] = {} differs from expected {} by {} (tolerance {})",
            i,
            sample,
            expected_y8,
            err,
            tolerance
        );
    }
}

/// Encode a 32×16 constant-grey frame with the Hq profile and confirm the
/// decoded frame records `ProResProfile::Hq`.
#[test]
fn encode_decode_hq_profile() {
    let frame = flat_grey_frame(32, 16, 600);
    let pkt = encode_frame_profile(&frame, ProResProfile::Hq);

    let decoded = ProResDecoder::decode(&pkt).expect("decode HQ frame");
    assert_eq!(decoded.profile, ProResProfile::Hq);
    assert_eq!(decoded.width, 32);
    assert_eq!(decoded.height, 16);
}

/// Encode a 32×16 constant-grey frame with the LT profile and confirm
/// the decoded frame records `ProResProfile::Lt`.
#[test]
fn encode_decode_lt_profile() {
    let frame = flat_grey_frame(32, 16, 200);
    let pkt = encode_frame_profile(&frame, ProResProfile::Lt);

    let decoded = ProResDecoder::decode(&pkt).expect("decode LT frame");
    assert_eq!(decoded.profile, ProResProfile::Lt);
}

/// Encode a 32×16 constant-grey frame with the Proxy profile and confirm the
/// decoded frame records `ProResProfile::Proxy`.
#[test]
fn encode_decode_proxy_profile() {
    let frame = flat_grey_frame(32, 16, 300);
    let pkt = encode_frame_profile(&frame, ProResProfile::Proxy);

    let decoded = ProResDecoder::decode(&pkt).expect("decode Proxy frame");
    assert_eq!(decoded.profile, ProResProfile::Proxy);
}

/// When `ProResDecoderConfig::profile` is set to `Some(Proxy)` but the stream
/// was encoded with `Standard`, the decoder must return `Err`.
#[test]
fn profile_config_mismatch_returns_error() {
    let frame = flat_grey_frame(32, 16, 512);
    let pkt = encode_frame_profile(&frame, ProResProfile::Standard);

    let config = ProResDecoderConfig {
        profile: Some(ProResProfile::Proxy),
    };
    let dec = ProResDecoder::with_config(config);
    let result = dec.decode_with_config(&pkt);
    assert!(
        result.is_err(),
        "should reject Standard stream when Proxy is required"
    );
}

/// When `ProResDecoderConfig::profile` matches the stream, decoding succeeds.
#[test]
fn profile_config_matching_succeeds() {
    let frame = flat_grey_frame(32, 16, 512);
    let pkt = encode_frame_profile(&frame, ProResProfile::Hq);

    let config = ProResDecoderConfig {
        profile: Some(ProResProfile::Hq),
    };
    let dec = ProResDecoder::with_config(config);
    dec.decode_with_config(&pkt)
        .expect("should decode matching-profile stream");
}

/// Larger frame (128×32) round-trip: encode with Standard profile, decode, and
/// check that all 128×32 luma samples are within ±4 LSB of expected.
#[test]
fn encode_decode_larger_frame() {
    let y_val: u16 = 700;
    let frame = flat_grey_frame(128, 32, y_val);
    let pkt = encode_frame(&frame);

    let decoded = ProResDecoder::decode(&pkt).expect("decode larger frame");
    assert_eq!(decoded.width, 128);
    assert_eq!(decoded.height, 32);
    assert_eq!(decoded.y.len(), 128 * 32);

    let expected = (y_val >> 2) as i32;
    for &sample in &decoded.y {
        let err = (sample as i32 - expected).abs();
        assert!(
            err <= 4,
            "luma {} expected ~{} err={}",
            sample,
            expected,
            err
        );
    }
}

// ─── 4:4:4 (ProRes 4444 / `ap4h`) round-trip tests ─────────────────────────────

/// Build a flat-colour `Yuv444p10le` frame: every luma sample = `y_val`,
/// every chroma sample = `c_val`. All three planes are full-resolution.
fn flat_yuv444p10le_frame(width: u32, height: u32, y_val: u16, c_val: u16) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;

    let y_bytes: Vec<u8> = (0..w * h).flat_map(|_| y_val.to_le_bytes()).collect();
    let c_bytes: Vec<u8> = (0..w * h).flat_map(|_| c_val.to_le_bytes()).collect();

    let mut frame = VideoFrame::new(PixelFormat::Yuv444p10le, width, height);
    frame.planes = vec![
        Plane::with_dimensions(y_bytes, w * 2, width, height),
        Plane::with_dimensions(c_bytes.clone(), w * 2, width, height),
        Plane::with_dimensions(c_bytes, w * 2, width, height),
    ];
    frame
}

/// Encode a `Yuv444p10le` frame as ProRes 4444 (`ap4h`) and return the packet
/// bytes.
fn encode_frame_444(frame: &VideoFrame, profile: ProResProfile) -> Vec<u8> {
    let cfg = ProResEncoderConfig::yuv444(profile, frame.width, frame.height);
    let mut enc = ProResEncoder::new(cfg).expect("4:4:4 encoder init");
    enc.send_frame(frame).expect("send_frame");
    enc.receive_packet()
        .expect("receive_packet")
        .expect("Some(packet)")
        .data
}

/// A constant-colour 4:4:4 frame must decode to a `Yuv444p`-flavoured
/// `ProResFrame` of the right shape, with full-resolution chroma planes and
/// samples within a small tolerance of the input.
#[test]
fn encode_decode_444_constant_color() {
    let y_val: u16 = 600;
    let c_val: u16 = 440;
    let frame = flat_yuv444p10le_frame(64, 32, y_val, c_val);
    let pkt = encode_frame_444(&frame, ProResProfile::P4444);

    let decoded = ProResDecoder::decode(&pkt).expect("decode 4:4:4 frame");

    // Structural checks: 4:4:4 chroma is full-resolution.
    assert_eq!(decoded.width, 64, "width");
    assert_eq!(decoded.height, 32, "height");
    assert_eq!(decoded.profile, ProResProfile::P4444, "profile");
    assert_eq!(decoded.chroma_format, ChromaFormat::Yuv444, "chroma format");
    assert_eq!(decoded.chroma_width(), 64, "chroma width == luma width");
    assert!(!decoded.is_interlaced, "should be progressive");
    assert_eq!(decoded.y.len(), 64 * 32, "luma plane size");
    assert_eq!(decoded.cb.len(), 64 * 32, "Cb plane is full-resolution");
    assert_eq!(decoded.cr.len(), 64 * 32, "Cr plane is full-resolution");

    // Per-sample accuracy (10-bit → 8-bit via >> 2).
    let expect_y8 = (y_val >> 2) as i32;
    let expect_c8 = (c_val >> 2) as i32;
    let tol = 4i32;
    for &s in &decoded.y {
        assert!(
            (s as i32 - expect_y8).abs() <= tol,
            "luma {s} differs from expected {expect_y8}"
        );
    }
    for &s in &decoded.cb {
        assert!(
            (s as i32 - expect_c8).abs() <= tol,
            "Cb {s} differs from expected {expect_c8}"
        );
    }
    for &s in &decoded.cr {
        assert!(
            (s as i32 - expect_c8).abs() <= tol,
            "Cr {s} differs from expected {expect_c8}"
        );
    }
}

/// A 4:4:4 frame with three *distinct* flat plane values must preserve each
/// plane independently — the decoder must not cross luma/chroma wires.
#[test]
fn encode_decode_444_distinct_plane_values() {
    let w = 48u32;
    let h = 16u32;
    // Three clearly different values so a plane mix-up is obvious.
    let y_val = 720u16;
    let cb_val = 300u16;
    let cr_val = 540u16;

    let mut frame = VideoFrame::new(PixelFormat::Yuv444p10le, w, h);
    let wu = w as usize;
    let hu = h as usize;
    frame.planes = vec![
        Plane::with_dimensions(
            (0..wu * hu).flat_map(|_| y_val.to_le_bytes()).collect(),
            wu * 2,
            w,
            h,
        ),
        Plane::with_dimensions(
            (0..wu * hu).flat_map(|_| cb_val.to_le_bytes()).collect(),
            wu * 2,
            w,
            h,
        ),
        Plane::with_dimensions(
            (0..wu * hu).flat_map(|_| cr_val.to_le_bytes()).collect(),
            wu * 2,
            w,
            h,
        ),
    ];

    let pkt = encode_frame_444(&frame, ProResProfile::P4444Xq);
    let decoded = ProResDecoder::decode(&pkt).expect("decode 4:4:4 frame");

    assert_eq!(decoded.profile, ProResProfile::P4444Xq);
    assert_eq!(decoded.chroma_format, ChromaFormat::Yuv444);

    let near = |plane: &[u8], expect10: u16, name: &str| {
        let expect8 = (expect10 >> 2) as i32;
        for &s in plane {
            assert!(
                (s as i32 - expect8).abs() <= 6,
                "{name} sample {s} far from expected {expect8}"
            );
        }
    };
    near(&decoded.y, y_val, "Y");
    near(&decoded.cb, cb_val, "Cb");
    near(&decoded.cr, cr_val, "Cr");
}

/// A 4:4:4 frame with a *textured* luma plane (a fine-grained pattern that
/// gives every 8×8 block real AC energy) and flat chroma must round-trip:
/// every full-resolution luma block exercises the 4:4:4 AC decode path, while
/// the chroma planes — also full-resolution — must come back near mid-grey.
#[test]
fn encode_decode_444_textured_luma() {
    let w = 64u32;
    let h = 32u32;
    let wu = w as usize;
    let hu = h as usize;

    // Fine-grained luma texture in the 64..768 band. The `% 704` keeps the
    // block-to-block DC swing modest while still giving every block AC
    // content, exactly like the proven 4:2:2 round-trip generator.
    let luma_at = |row: usize, col: usize| -> u16 { 64u16 + ((row * wu + col) as u16 % 704) };
    let mut y_bytes = vec![0u8; wu * hu * 2];
    for row in 0..hu {
        for col in 0..wu {
            let idx = (row * wu + col) * 2;
            y_bytes[idx..idx + 2].copy_from_slice(&luma_at(row, col).to_le_bytes());
        }
    }
    // Flat chroma at 10-bit mid-grey.
    let c_bytes: Vec<u8> = (0..wu * hu).flat_map(|_| 512u16.to_le_bytes()).collect();

    let mut frame = VideoFrame::new(PixelFormat::Yuv444p10le, w, h);
    frame.planes = vec![
        Plane::with_dimensions(y_bytes, wu * 2, w, h),
        Plane::with_dimensions(c_bytes.clone(), wu * 2, w, h),
        Plane::with_dimensions(c_bytes, wu * 2, w, h),
    ];

    let pkt = encode_frame_444(&frame, ProResProfile::P4444);
    let decoded = ProResDecoder::decode(&pkt).expect("decode 4:4:4 textured frame");

    assert_eq!(decoded.chroma_format, ChromaFormat::Yuv444);
    assert_eq!(decoded.y.len(), wu * hu);
    assert_eq!(decoded.cb.len(), wu * hu, "Cb is full-resolution");
    assert_eq!(decoded.cr.len(), wu * hu, "Cr is full-resolution");

    // The decoded luma should track the input texture (8-bit domain) within
    // ProRes's lossy-but-visually-lossless tolerance.
    let mut max_err = 0i32;
    for row in 0..hu {
        for col in 0..wu {
            let expect8 = (luma_at(row, col) >> 2) as i32;
            let got = decoded.y[row * wu + col] as i32;
            max_err = max_err.max((got - expect8).abs());
        }
    }
    assert!(
        max_err <= 24,
        "4:4:4 textured luma max error {max_err} exceeds tolerance"
    );

    // The full-resolution chroma planes must stay near mid-grey everywhere —
    // the luma texture must not bleed into Cb/Cr.
    for &s in &decoded.cb {
        assert!((s as i32 - 128).abs() <= 8, "Cb drifted: {s}");
    }
    for &s in &decoded.cr {
        assert!((s as i32 - 128).abs() <= 8, "Cr drifted: {s}");
    }
}

/// The `VideoDecoder` trait path must report `Yuv444p` and emit three
/// full-resolution planes for a 4:4:4 stream.
#[test]
fn video_decoder_trait_444_reports_yuv444p() {
    let frame = flat_yuv444p10le_frame(32, 16, 512, 512);
    let pkt = encode_frame_444(&frame, ProResProfile::P4444);

    let mut dec = ProResDecoder::new();
    // Before any packet, the decoder defaults to the 4:2:2 format.
    assert_eq!(dec.output_format(), Some(PixelFormat::Yuv422p));

    dec.send_packet(&pkt, 7).expect("send_packet");
    let vf = dec.receive_frame().expect("receive_frame").expect("Some");
    assert_eq!(vf.timestamp.pts, 7);
    assert_eq!(vf.width, 32);
    assert_eq!(vf.height, 16);
    assert_eq!(vf.format, PixelFormat::Yuv444p, "4:4:4 → Yuv444p");
    assert_eq!(vf.planes.len(), 3);
    // All three planes are full-resolution for 4:4:4.
    assert_eq!(vf.planes[0].data.len(), 32 * 16, "Y plane");
    assert_eq!(vf.planes[1].data.len(), 32 * 16, "Cb plane (full-res)");
    assert_eq!(vf.planes[2].data.len(), 32 * 16, "Cr plane (full-res)");

    // After seeing a 4:4:4 packet, output_format reports Yuv444p.
    assert_eq!(dec.output_format(), Some(PixelFormat::Yuv444p));
}

/// A 4:4:4 round-trip must not regress 4:2:2: decoding a 4:2:2 stream still
/// yields half-width chroma and a `Yuv422` chroma format.
#[test]
fn decode_422_still_reports_yuv422_after_444_support() {
    let frame = flat_grey_frame(64, 16, 480);
    let pkt = encode_frame(&frame); // 4:2:2 Standard

    let decoded = ProResDecoder::decode(&pkt).expect("decode 4:2:2 frame");
    assert_eq!(decoded.chroma_format, ChromaFormat::Yuv422);
    assert_eq!(decoded.chroma_width(), 32, "4:2:2 chroma is half-width");
    assert_eq!(decoded.cb.len(), 32 * 16);
    assert_eq!(decoded.cr.len(), 32 * 16);
}

/// A larger 4:4:4 frame (128×48) round-trips with correctly sized planes.
#[test]
fn encode_decode_444_larger_frame() {
    let frame = flat_yuv444p10le_frame(128, 48, 700, 512);
    let pkt = encode_frame_444(&frame, ProResProfile::P4444);

    let decoded = ProResDecoder::decode(&pkt).expect("decode larger 4:4:4 frame");
    assert_eq!(decoded.width, 128);
    assert_eq!(decoded.height, 48);
    assert_eq!(decoded.chroma_format, ChromaFormat::Yuv444);
    assert_eq!(decoded.y.len(), 128 * 48);
    assert_eq!(decoded.cb.len(), 128 * 48);
    assert_eq!(decoded.cr.len(), 128 * 48);

    let expect = (700u16 >> 2) as i32;
    for &s in &decoded.y {
        assert!((s as i32 - expect).abs() <= 4, "luma {s} far from {expect}");
    }
}
