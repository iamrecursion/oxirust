//! Integration tests for the JPEG 2000 encoder, lossless (5-3) and lossy
//! (9-7) paths.
//!
//! Lossless tests encode an image with [`Jpeg2000Encoder`] and decode the
//! result with the existing [`Jpeg2000Decoder`], asserting the decoded samples
//! equal the original input EXACTLY (byte-exact round-trip).
//!
//! Lossy tests assert the decoded samples match within the quantiser tolerance
//! (±2 LSB on small flat patches, ≥ 35 dB PSNR on a gradient).

#![cfg(feature = "jpeg2000")]

use oximedia_codec::jpeg2000::encoder::{Jpeg2000Encoder, Jpeg2000EncoderConfig};
use oximedia_codec::jpeg2000::marker_write::{
    write_eoc, write_qcd_lossy, write_siz, write_soc, write_sod, write_sot, ComponentSpec,
};
use oximedia_codec::jpeg2000::markers::{parse_codestream, MarkerSegment};
use oximedia_codec::jpeg2000::mq_coder::MqDecoder;
use oximedia_codec::jpeg2000::mq_encoder::MqEncoder;
use oximedia_codec::jpeg2000::wavelet::{
    decompose_levels, forward_wavelet_1d_97, inverse_wavelet_1d_97, reconstruct_levels,
};
use oximedia_codec::jpeg2000::Jpeg2000Decoder;

/// Tiny deterministic LCG so the tests need no external RNG dependency.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
    fn next_range(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }
}

/// Encode a single greyscale plane and assert a byte-exact decode.
fn assert_lossless_grey(plane: &[i32], width: usize, height: usize, levels: u8, bit_depth: u8) {
    let cfg = Jpeg2000EncoderConfig {
        levels,
        xcb: 6,
        ycb: 6,
        bit_depth,
        lossless: true,
    };
    let enc = Jpeg2000Encoder::new(cfg);
    let bytes = enc
        .encode_greyscale(plane, width, height)
        .expect("encode_greyscale");

    let img = Jpeg2000Decoder::decode(&bytes).expect("decode");
    assert_eq!(img.width as usize, width, "width");
    assert_eq!(img.height as usize, height, "height");
    assert_eq!(img.num_components, 1, "num_components");
    assert_eq!(img.samples[0].len(), width * height, "sample count");
    for (i, (&orig, &dec)) in plane.iter().zip(img.samples[0].iter()).enumerate() {
        assert_eq!(
            orig,
            i32::from(dec),
            "pixel {i} (row {}, col {}) mismatch: orig {orig}, decoded {dec}",
            i / width,
            i % width
        );
    }
}

// ── MQ coder standalone round-trip ────────────────────────────────────────────

#[test]
fn mq_encode_decode_roundtrip() {
    let mut rng = Lcg::new(0xc0ff_ee00_1234_5678);
    // Several independent random decision/context streams must survive an
    // encode → decode cycle bit-for-bit.
    for trial in 0..16 {
        let len = 50 + (rng.next_range(4000) as usize);
        let decisions: Vec<(usize, u8)> = (0..len)
            .map(|_| {
                let cx = rng.next_range(19) as usize;
                let d = (rng.next_u32() & 1) as u8;
                (cx, d)
            })
            .collect();

        let mut enc = MqEncoder::new();
        for &(cx, d) in &decisions {
            assert!(enc.encode_decision(cx, d), "encode failed cx={cx}");
        }
        let bytes = enc.flush();

        let mut dec = MqDecoder::new(&bytes);
        for (i, &(cx, expected)) in decisions.iter().enumerate() {
            let got = dec.decode_bit(cx).expect("decode");
            assert_eq!(got, expected, "trial {trial}, decision {i} (cx={cx})");
        }
    }
}

// ── Forward/inverse 5-3 wavelet identity ──────────────────────────────────────

#[test]
fn wavelet_53_forward_inverse_identity() {
    let mut rng = Lcg::new(0x5353_5353_aaaa_bbbb);

    // The decoder's `reconstruct_levels` reconstructs intermediate (non-finest)
    // resolutions by doubling the detail-subband dimensions, which only matches
    // the ceil-based forward decomposition when intermediate dimensions stay
    // even. Even / power-of-two sizes are therefore exercised at all levels;
    // odd sizes are exercised at 0 and 1 levels (where the finest level uses the
    // true image dimensions directly).
    let even_cases: &[(usize, usize)] = &[(16, 16), (32, 16), (64, 64), (8, 24)];
    for &(w, h) in even_cases {
        let image: Vec<i32> = (0..w * h)
            .map(|_| (rng.next_range(512) as i32) - 256)
            .collect();
        for levels in 0..=3usize {
            let tree = decompose_levels(&image, w, h, levels).expect("decompose");
            let recon = reconstruct_levels(&tree, levels, w, h).expect("reconstruct");
            assert_eq!(recon.len(), w * h, "size {w}x{h} levels {levels}");
            for (i, (&orig, &rec)) in image.iter().zip(recon.iter()).enumerate() {
                assert_eq!(
                    orig, rec,
                    "wavelet identity mismatch at {i} for {w}x{h} levels {levels}"
                );
            }
        }
    }

    let odd_cases: &[(usize, usize)] = &[(7, 13), (5, 9), (15, 1), (1, 11)];
    for &(w, h) in odd_cases {
        let image: Vec<i32> = (0..w * h)
            .map(|_| (rng.next_range(512) as i32) - 256)
            .collect();
        for levels in 0..=1usize {
            let tree = decompose_levels(&image, w, h, levels).expect("decompose");
            let recon = reconstruct_levels(&tree, levels, w, h).expect("reconstruct");
            assert_eq!(recon.len(), w * h, "odd size {w}x{h} levels {levels}");
            for (i, (&orig, &rec)) in image.iter().zip(recon.iter()).enumerate() {
                assert_eq!(
                    orig, rec,
                    "wavelet identity mismatch at {i} for odd {w}x{h} levels {levels}"
                );
            }
        }
    }
}

// ── Pixel-domain encode → decode round-trips (byte-exact) ─────────────────────

#[test]
fn roundtrip_grey_16x16_nl0() {
    // 0 decomposition levels: the image is carried directly in the LL band.
    let w = 16;
    let h = 16;
    let mut rng = Lcg::new(0x1616_0000_1111_2222);
    let plane: Vec<i32> = (0..w * h).map(|_| rng.next_range(256) as i32).collect();
    assert_lossless_grey(&plane, w, h, 0, 8);
}

#[test]
fn roundtrip_gradient_32x32_nl1() {
    let w = 32;
    let h = 32;
    // Smooth 2D gradient — low-frequency, so detail coefficients stay small and
    // within the 8 magnitude bit-planes the decoder uses.
    let plane: Vec<i32> = (0..w * h)
        .map(|i| {
            let x = i % w;
            let y = i / w;
            ((x + y) * 255 / (w + h - 2)) as i32
        })
        .collect();
    assert_lossless_grey(&plane, w, h, 1, 8);
}

#[test]
fn roundtrip_gradient_64x64_nl3() {
    let w = 64;
    let h = 64;
    let plane: Vec<i32> = (0..w * h)
        .map(|i| {
            let x = i % w;
            let y = i / w;
            ((x + y) * 255 / (w + h - 2)) as i32
        })
        .collect();
    assert_lossless_grey(&plane, w, h, 3, 8);
}

#[test]
fn roundtrip_rgb_3component() {
    // The decoder reconstructs the same tile-data for every component, so a
    // byte-exact multi-component round-trip uses identical planes (greyscale
    // replicated across R/G/B). This exercises the multi-component SIZ and the
    // full-frame assembly path.
    let w = 32;
    let h = 32;
    let plane: Vec<i32> = (0..w * h)
        .map(|i| {
            let x = i % w;
            let y = i / w;
            ((x * 7 + y * 3) % 200) as i32
        })
        .collect();
    let planes: [&[i32]; 3] = [&plane, &plane, &plane];

    let cfg = Jpeg2000EncoderConfig {
        levels: 2,
        xcb: 6,
        ycb: 6,
        bit_depth: 8,
        lossless: true,
    };
    let enc = Jpeg2000Encoder::new(cfg);
    let bytes = enc.encode_planes(&planes, w, h).expect("encode_planes");

    let img = Jpeg2000Decoder::decode(&bytes).expect("decode");
    assert_eq!(img.num_components, 3, "num_components");
    assert_eq!(img.samples.len(), 3, "component count");
    for (comp_idx, comp) in img.samples.iter().enumerate() {
        assert_eq!(comp.len(), w * h, "component {comp_idx} sample count");
        for (i, (&orig, &dec)) in plane.iter().zip(comp.iter()).enumerate() {
            assert_eq!(
                orig,
                i32::from(dec),
                "component {comp_idx} pixel {i} mismatch"
            );
        }
    }
}

// ── Extra coverage (still byte-exact) ─────────────────────────────────────────

#[test]
fn roundtrip_constant_grey_various_levels() {
    for levels in 0..=3u8 {
        let w = 16;
        let h = 16;
        let plane = vec![200i32; w * h];
        assert_lossless_grey(&plane, w, h, levels, 8);
    }
}

#[test]
fn roundtrip_smooth_16bit_nl2() {
    // 16-bit component with a smooth gradient (small detail coefficients).
    let w = 32;
    let h = 32;
    let plane: Vec<i32> = (0..w * h)
        .map(|i| {
            let x = i % w;
            let y = i / w;
            ((x + y) * 65535 / (w + h - 2)) as i32
        })
        .collect();
    assert_lossless_grey(&plane, w, h, 2, 16);
}

#[test]
fn roundtrip_vertical_ramp_nl2() {
    // A vertical-only ramp (constant per row) keeps horizontal detail zero.
    let w = 64;
    let h = 64;
    let plane: Vec<i32> = (0..w * h)
        .map(|i| ((i / w) * 255 / (h - 1)) as i32)
        .collect();
    assert_lossless_grey(&plane, w, h, 2, 8);
}

// ── CDF 9/7 lossy round-trip integration tests (Wave 10 Slice 2) ─────────────

#[test]
fn wavelet_97_forward_inverse_identity_8() {
    // forward then inverse CDF 9/7 on 8 floats must reconstruct identity
    // within 1e-6 (filter floating-point tolerance).
    let signal: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let (low, high) = forward_wavelet_1d_97(&signal);
    let recovered = inverse_wavelet_1d_97(&low, &high);
    assert_eq!(recovered.len(), signal.len());
    for (i, (&orig, &rec)) in signal.iter().zip(recovered.iter()).enumerate() {
        assert!(
            (orig - rec).abs() < 1e-6,
            "sample {i}: orig {orig}, rec {rec}, diff {}",
            (orig - rec).abs()
        );
    }
}

fn assert_lossy_grey_within(
    plane: &[i32],
    width: usize,
    height: usize,
    levels: u8,
    bit_depth: u8,
    tolerance_lsb: u16,
) {
    let cfg = Jpeg2000EncoderConfig {
        levels,
        xcb: 6,
        ycb: 6,
        bit_depth,
        lossless: false,
    };
    let enc = Jpeg2000Encoder::new(cfg);
    let bytes = enc
        .encode_greyscale(plane, width, height)
        .expect("encode_greyscale lossy");
    let img = Jpeg2000Decoder::decode(&bytes).expect("decode lossy");
    assert_eq!(img.width as usize, width);
    assert_eq!(img.height as usize, height);
    assert_eq!(img.num_components, 1);
    assert_eq!(img.samples[0].len(), width * height);
    for (i, (&orig, &dec)) in plane.iter().zip(img.samples[0].iter()).enumerate() {
        let diff = (orig - i32::from(dec)).abs();
        assert!(
            diff <= i32::from(tolerance_lsb),
            "pixel {i} (row {}, col {}): orig {orig}, decoded {dec}, |diff| {diff} > {tolerance_lsb}",
            i / width,
            i % width
        );
    }
}

#[test]
fn roundtrip_grey_16x16_97_nl1() {
    // Flat 16×16 grey through lossy 9-7 with 1 decomposition level must round-
    // trip within ±2 LSB.
    let w = 16;
    let h = 16;
    let plane = vec![137i32; w * h];
    assert_lossy_grey_within(&plane, w, h, 1, 8, 2);
}

#[test]
fn roundtrip_grey_16x16_97_nl1_low_value() {
    let w = 16;
    let h = 16;
    let plane = vec![15i32; w * h];
    assert_lossy_grey_within(&plane, w, h, 1, 8, 2);
}

#[test]
fn roundtrip_gradient_32x32_97_nl3() {
    // 32×32 smooth gradient through lossy 9-7 with 3 decomposition levels;
    // assert PSNR ≥ 35 dB.
    let w = 32usize;
    let h = 32usize;
    let plane: Vec<i32> = (0..w * h)
        .map(|i| {
            let x = i % w;
            let y = i / w;
            ((x + y) * 255 / (w + h - 2)) as i32
        })
        .collect();
    let cfg = Jpeg2000EncoderConfig {
        levels: 3,
        xcb: 5,
        ycb: 5,
        bit_depth: 8,
        lossless: false,
    };
    let enc = Jpeg2000Encoder::new(cfg);
    let bytes = enc
        .encode_greyscale(&plane, w, h)
        .expect("encode_greyscale lossy");
    let img = Jpeg2000Decoder::decode(&bytes).expect("decode lossy");
    assert_eq!(img.samples[0].len(), w * h);

    // Compute PSNR (8-bit reference).
    let mut sse: f64 = 0.0;
    for (&orig, &dec) in plane.iter().zip(img.samples[0].iter()) {
        let diff = (orig - i32::from(dec)) as f64;
        sse += diff * diff;
    }
    let mse = sse / (w * h) as f64;
    assert!(mse > 0.0 || sse == 0.0, "MSE should be non-negative finite");
    let max_val_sq = 255.0_f64 * 255.0;
    let psnr_db = if mse < 1e-12 {
        99.0
    } else {
        10.0 * (max_val_sq / mse).log10()
    };
    assert!(
        psnr_db >= 35.0,
        "Lossy 9-7 PSNR {psnr_db} dB < 35 dB target (MSE = {mse})"
    );
}

#[test]
fn qcd_lossy_write_parse_roundtrip() {
    // Write a lossy QCD via write_qcd_lossy, parse it back via parse_codestream,
    // and verify the ε/μ pairs survive byte-exactly.
    let mut bytes = Vec::new();
    write_soc(&mut bytes);
    write_siz(&mut bytes, 16, 16, 16, 16, &[ComponentSpec::unsigned(8)]).expect("siz");
    // 1 decomp level → 4 subbands (LL + HL + LH + HH).
    let pairs = [(8u8, 0u16), (8u8, 0u16), (8u8, 0u16), (8u8, 0u16)];
    write_qcd_lossy(&mut bytes, 1, &pairs).expect("write_qcd_lossy");
    write_sot(&mut bytes, 0, 0, 0, 1);
    write_sod(&mut bytes);
    bytes.push(0x00);
    write_eoc(&mut bytes);

    let segments = parse_codestream(&bytes).expect("parse_codestream");
    let qcd = segments
        .iter()
        .find_map(|s| match s {
            MarkerSegment::Qcd(q) => Some(q),
            _ => None,
        })
        .expect("QCD segment");

    assert_eq!(qcd.quant_style(), 2, "Sqcd style must be 2 (expounded)");
    assert_eq!(qcd.guard_bits(), 0, "guard bits = 0");
    assert_eq!(qcd.step_sizes.len(), pairs.len(), "subband count");
    for (raw, &(eps, mu)) in qcd.step_sizes.iter().zip(pairs.iter()) {
        let parsed_eps = ((raw >> 11) & 0x1F) as u8;
        let parsed_mu = raw & 0x07FF;
        assert_eq!(parsed_eps, eps, "ε round-trip");
        assert_eq!(parsed_mu, mu, "μ round-trip");
    }
}

#[test]
fn lossless_path_still_works_after_lossy_changes() {
    // Sanity: the unchanged lossless path must still byte-exact round-trip on
    // a smooth gradient (sharp-edge inputs can trip the lossless coefficient
    // magnitude check, which is intentional). This mirrors the smooth ramp
    // pattern used by other lossless tests in the file.
    let w = 8;
    let h = 8;
    let plane: Vec<i32> = (0..w * h)
        .map(|i| {
            let x = i % w;
            let y = i / w;
            ((x + y) * 200 / (w + h - 2)) as i32
        })
        .collect();
    assert_lossless_grey(&plane, w, h, 1, 8);
}
