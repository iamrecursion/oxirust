//! CPU smoke tests — must pass in CI without any CUDA device.
//!
//! These tests exercise:
//! - Deterministic seed replay (same seed → same sequence)
//! - Uniform distribution range [0.0, 1.0)
//! - Normal distribution approximate statistics (N=10,000)
//! - Bernoulli proportion (N=10,000)

use tensorlogic_oxicuda_rng::{RngEngine, RngEngineKind};

/// Same seed on two independently constructed engines must produce identical
/// output sequences.
#[test]
fn deterministic_seed_replay() {
    let mut rng1 = RngEngine::new(RngEngineKind::Philox, 42).unwrap();
    let mut rng2 = RngEngine::new(RngEngineKind::Philox, 42).unwrap();
    let mut out1 = vec![0f32; 100];
    let mut out2 = vec![0f32; 100];
    rng1.uniform_f32(&mut out1).unwrap();
    rng2.uniform_f32(&mut out2).unwrap();
    assert_eq!(out1, out2, "same seed must produce same sequence");
}

/// Different engine kinds with the same seed should generally differ (the PCG
/// stream discriminator is derived from the seed, not the kind — but each kind
/// with the *same* seed but different usage order may diverge over multiple
/// calls).  At minimum they must both produce valid output.
#[test]
fn different_kinds_construct_ok() {
    for kind in [
        RngEngineKind::Philox,
        RngEngineKind::Xorwow,
        RngEngineKind::Mrg32k3a,
    ] {
        let mut rng = RngEngine::new(kind, 7).unwrap();
        let mut out = vec![0f32; 64];
        rng.uniform_f32(&mut out).unwrap();
        assert!(
            out.iter().all(|&v| (0.0..1.0).contains(&v)),
            "kind {kind} produced out-of-range uniform sample"
        );
    }
}

/// All uniform samples must lie in `[0.0, 1.0)`.
#[test]
fn uniform_range() {
    let mut rng = RngEngine::new(RngEngineKind::Xorwow, 1234).unwrap();
    let mut out = vec![0f32; 10_000];
    rng.uniform_f32(&mut out).unwrap();
    for &v in &out {
        assert!((0.0..1.0).contains(&v), "uniform out of range: {v}");
    }
}

/// Sequential calls must advance the stream (not reset or repeat).
#[test]
fn uniform_sequential_calls_advance_stream() {
    let mut rng = RngEngine::new(RngEngineKind::Philox, 999).unwrap();
    let mut a = vec![0f32; 16];
    let mut b = vec![0f32; 16];
    rng.uniform_f32(&mut a).unwrap();
    rng.uniform_f32(&mut b).unwrap();
    // It is astronomically unlikely that two 16-element blocks are identical.
    assert_ne!(a, b, "sequential uniform calls must advance the stream");
}

/// Normal distribution with N=10,000 should have mean ≈ 0 and variance ≈ 1
/// (tolerance: ±0.05).
#[test]
fn normal_approximate_stats() {
    let mut rng = RngEngine::new(RngEngineKind::Mrg32k3a, 999).unwrap();
    let mut out = vec![0f32; 10_000];
    rng.normal_f32(&mut out, 0.0, 1.0).unwrap();

    let n = out.len() as f32;
    let mean: f32 = out.iter().sum::<f32>() / n;
    let var: f32 = out.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;

    assert!(mean.abs() < 0.05, "mean too far from 0: {mean}");
    assert!((var - 1.0).abs() < 0.05, "variance too far from 1: {var}");
}

/// Normal distribution with non-standard parameters should shift mean and
/// scale variance accordingly.
#[test]
fn normal_nonstandard_params() {
    let target_mean = 5.0_f32;
    let target_std = 2.0_f32;

    let mut rng = RngEngine::new(RngEngineKind::Xorwow, 4242).unwrap();
    let mut out = vec![0f32; 10_000];
    rng.normal_f32(&mut out, target_mean, target_std).unwrap();

    let n = out.len() as f32;
    let mean: f32 = out.iter().sum::<f32>() / n;
    let var: f32 = out.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();

    assert!(
        (mean - target_mean).abs() < 0.1,
        "mean {mean} too far from target {target_mean}"
    );
    assert!(
        (std - target_std).abs() < 0.1,
        "std {std} too far from target {target_std}"
    );
}

/// All normal samples must be finite (no NaN, no ±inf).
#[test]
fn normal_all_finite() {
    let mut rng = RngEngine::new(RngEngineKind::Philox, 101).unwrap();
    let mut out = vec![0f32; 5_000];
    rng.normal_f32(&mut out, 0.0, 1.0).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "element {i} is not finite: {v}");
    }
}

/// An odd-length output must be filled completely (tests trailing-element
/// branch of the Box-Muller loop).
#[test]
fn normal_odd_length_complete() {
    let mut rng = RngEngine::new(RngEngineKind::Mrg32k3a, 13).unwrap();
    let odd = 2_001_usize;
    let mut out = vec![f32::NAN; odd];
    rng.normal_f32(&mut out, 0.0, 1.0).unwrap();
    let nans = out.iter().filter(|v| v.is_nan()).count();
    assert_eq!(
        nans, 0,
        "{nans} NaN(s) remain after normal_f32 with odd length {odd}"
    );
}

/// Bernoulli p=0.3: observed proportion must be within ±0.03 of 0.3.
#[test]
fn bernoulli_proportion() {
    let mut rng = RngEngine::new(RngEngineKind::Philox, 777).unwrap();
    let mut out = vec![0u8; 10_000];
    rng.bernoulli(&mut out, 0.3).unwrap();

    let ones = out.iter().filter(|&&b| b == 1).count() as f32 / 10_000.0;
    assert!(
        (ones - 0.3).abs() < 0.03,
        "proportion {ones} too far from 0.3"
    );
}

/// Bernoulli p=0.0: all outputs must be 0.
#[test]
fn bernoulli_p_zero_all_zero() {
    let mut rng = RngEngine::new(RngEngineKind::Xorwow, 1).unwrap();
    let mut out = vec![1u8; 1_000];
    rng.bernoulli(&mut out, 0.0).unwrap();
    assert!(out.iter().all(|&b| b == 0), "p=0 must produce all zeros");
}

/// Bernoulli p=1.0: all outputs must be 1.
#[test]
fn bernoulli_p_one_all_one() {
    let mut rng = RngEngine::new(RngEngineKind::Mrg32k3a, 2).unwrap();
    let mut out = vec![0u8; 1_000];
    rng.bernoulli(&mut out, 1.0).unwrap();
    assert!(out.iter().all(|&b| b == 1), "p=1 must produce all ones");
}

/// Bernoulli outputs must only ever be 0 or 1.
#[test]
fn bernoulli_values_are_binary() {
    let mut rng = RngEngine::new(RngEngineKind::Philox, 321).unwrap();
    let mut out = vec![255u8; 5_000];
    rng.bernoulli(&mut out, 0.5).unwrap();
    for (i, &b) in out.iter().enumerate() {
        assert!(b == 0 || b == 1, "element {i} has non-binary value {b}");
    }
}
