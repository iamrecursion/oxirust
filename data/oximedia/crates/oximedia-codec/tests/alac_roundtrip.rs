//! ALAC (Apple Lossless) encoder ↔ decoder round-trip integration tests.
//!
//! Every test asserts byte-exact reconstruction: the decoded interleaved PCM
//! must equal the original input. Noise is generated with a tiny xorshift PRNG
//! so the tests stay dependency-free.

#![cfg(feature = "alac")]

use oximedia_codec::alac::{AlacDecoder, AlacEncoder, AlacEncoderConfig, AlacSpecificConfig};

/// Tiny xorshift32 PRNG for dependency-free noise generation.
struct XorShift32 {
    state: u32,
}

impl XorShift32 {
    fn new(seed: u32) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }
}

/// Encode one block and decode it back, asserting exact equality.
fn assert_roundtrip(pcm: &[i32], channels: u8, bit_depth: u8) {
    let frame_length = (pcm.len() / channels as usize) as u32;
    let cfg = AlacEncoderConfig {
        frame_length,
        sample_rate: 44_100,
        channels,
        bit_depth,
    };
    let mut encoder = AlacEncoder::new(cfg).expect("encoder");
    let cookie = encoder.magic_cookie();
    let frame = encoder.encode_frame(pcm).expect("encode");

    let mut decoder = AlacDecoder::new(&cookie).expect("decoder");
    let decoded = decoder.decode_packet(&frame).expect("decode");
    assert_eq!(decoded, pcm, "byte-exact round-trip failed");
}

#[test]
fn config_cookie_roundtrip() {
    let cfg = AlacSpecificConfig::new(4096, 48_000, 2, 24);
    let bytes = cfg.serialize();
    let parsed = AlacSpecificConfig::parse(&bytes).expect("parse");
    assert_eq!(parsed, cfg);
    // Reserialization is stable.
    assert_eq!(parsed.serialize(), bytes);
}

#[test]
fn roundtrip_mono_16bit_sine() {
    let pcm: Vec<i32> = (0..4096)
        .map(|i| ((i as f64 * 0.07).sin() * 12_000.0) as i32)
        .collect();
    assert_roundtrip(&pcm, 1, 16);
}

#[test]
fn roundtrip_stereo_16bit_decorrelated() {
    // Highly correlated L/R (a small inter-channel offset) exercises the
    // mid/side decorrelation path.
    let mut pcm = Vec::with_capacity(4096 * 2);
    for i in 0..4096 {
        let base = ((i as f64 * 0.05).sin() * 10_000.0) as i32;
        let left = base;
        let right = base + ((i as f64 * 0.05).cos() * 200.0) as i32;
        pcm.push(left);
        pcm.push(right);
    }
    assert_roundtrip(&pcm, 2, 16);
}

#[test]
fn roundtrip_24bit_stereo() {
    let mut pcm = Vec::with_capacity(2048 * 2);
    for i in 0..2048 {
        let base = ((i as f64 * 0.013).sin() * 3_000_000.0) as i32;
        pcm.push(base);
        pcm.push(-base / 2 + ((i as f64 * 0.013).cos() * 5_000.0) as i32);
    }
    assert_roundtrip(&pcm, 2, 24);
}

#[test]
fn roundtrip_20bit_mono() {
    let pcm: Vec<i32> = (0..2048)
        .map(|i| ((i as f64 * 0.02).sin() * 400_000.0) as i32)
        .collect();
    assert_roundtrip(&pcm, 1, 20);
}

#[test]
fn roundtrip_constant_block() {
    // Constant samples stress the predictor + zero-run path.
    let pcm = vec![7777i32; 4096];
    assert_roundtrip(&pcm, 1, 16);

    let silence = vec![0i32; 2048 * 2];
    assert_roundtrip(&silence, 2, 16);
}

#[test]
fn roundtrip_random_noise_escape() {
    // High-entropy 16-bit noise: incompressible, must still be byte-exact and
    // should trigger the uncompressed/escape fallback.
    let mut rng = XorShift32::new(0xDEAD_BEEF);
    let pcm: Vec<i32> = (0..4096)
        .map(|_| {
            let v = (rng.next_u32() & 0xFFFF) as i32 - 0x8000;
            v
        })
        .collect();
    assert_roundtrip(&pcm, 1, 16);

    // Stereo noise too.
    let mut rng2 = XorShift32::new(0x1234_5678);
    let stereo: Vec<i32> = (0..2048 * 2)
        .map(|_| (rng2.next_u32() & 0xFFFF) as i32 - 0x8000)
        .collect();
    assert_roundtrip(&stereo, 2, 16);
}

#[test]
fn roundtrip_shifted_low_bits() {
    // Force the bytes_shifted (uncompressed low-bits) decode path.
    let mut pcm = Vec::with_capacity(1024 * 2);
    let mut rng = XorShift32::new(0xABCD_1234);
    for i in 0..1024 {
        let base = ((i as f64 * 0.02).sin() * 2_000_000.0) as i32;
        // Add noisy low bits so shifting them off is meaningful.
        let noise = (rng.next_u32() & 0xFF) as i32;
        pcm.push(base + noise);
        pcm.push(base / 2 + ((rng.next_u32() & 0xFF) as i32));
    }
    let cfg = AlacEncoderConfig {
        frame_length: 1024,
        sample_rate: 44_100,
        channels: 2,
        bit_depth: 24,
    };
    let mut encoder = AlacEncoder::new(cfg).expect("encoder");
    encoder.set_bytes_shifted(1).expect("shift");
    let cookie = encoder.magic_cookie();
    let frame = encoder.encode_frame(&pcm).expect("encode");

    let mut decoder = AlacDecoder::new(&cookie).expect("decoder");
    let decoded = decoder.decode_packet(&frame).expect("decode");
    assert_eq!(decoded, pcm, "shifted-bits round-trip failed");
}

#[test]
fn rice_encode_decode_each_k() {
    // A parameter sweep: ramps of growing magnitude exercise a wide range of
    // adaptive-Golomb parameters within a single mono frame each.
    for scale in [1i32, 4, 16, 64, 256, 1024] {
        let pcm: Vec<i32> = (0..1024).map(|i| (i % 13 - 6) * scale).collect();
        assert_roundtrip(&pcm, 1, 24);
    }
}

#[test]
fn reject_truncated_frame() {
    let cfg = AlacEncoderConfig {
        frame_length: 1024,
        sample_rate: 44_100,
        channels: 2,
        bit_depth: 16,
    };
    let mut encoder = AlacEncoder::new(cfg).expect("encoder");
    let cookie = encoder.magic_cookie();
    let pcm: Vec<i32> = (0..1024 * 2)
        .map(|i| ((i as f64 * 0.03).sin() * 9_000.0) as i32)
        .collect();
    let frame = encoder.encode_frame(&pcm).expect("encode");

    let mut decoder = AlacDecoder::new(&cookie).expect("decoder");
    // Truncate the frame to half its length: must error, never panic.
    let truncated = &frame[..frame.len() / 2];
    let result = decoder.decode_packet(truncated);
    assert!(result.is_err(), "truncated frame must return Err");

    // An empty frame must also error.
    assert!(decoder.decode_packet(&[]).is_err());
}

#[test]
fn roundtrip_partial_frame() {
    // A block shorter than frame_length triggers the partial_frame header path.
    let cfg = AlacEncoderConfig {
        frame_length: 4096,
        sample_rate: 44_100,
        channels: 1,
        bit_depth: 16,
    };
    let mut encoder = AlacEncoder::new(cfg).expect("encoder");
    let cookie = encoder.magic_cookie();
    let pcm: Vec<i32> = (0..1000)
        .map(|i| ((i as f64 * 0.09).sin() * 8_000.0) as i32)
        .collect();
    let frame = encoder.encode_frame(&pcm).expect("encode");

    let mut decoder = AlacDecoder::new(&cookie).expect("decoder");
    let decoded = decoder.decode_packet(&frame).expect("decode");
    assert_eq!(decoded, pcm);
}
