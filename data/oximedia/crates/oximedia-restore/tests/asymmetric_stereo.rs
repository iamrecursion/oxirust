//! Asymmetric-corruption stereo restoration tests.
//!
//! Verifies that `RestoreChain::process_stereo` treats the two channels
//! independently: corrupting only the *left* channel with clicks must clean
//! the left while leaving the *right* (clean) channel bit-for-bit untouched —
//! i.e. there is no cross-channel state leak through the shared detector /
//! remover.

use oximedia_restore::click::{ClickDetector, ClickDetectorConfig, ClickRemover};
use oximedia_restore::utils::interpolation::InterpolationMethod;
use oximedia_restore::{RestorationStep, RestoreChain};
use std::f32::consts::PI;

/// Pearson correlation coefficient between two equal-length signals.
fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let inv = 1.0 / n as f32;
    let mean_a: f32 = a[..n].iter().sum::<f32>() * inv;
    let mean_b: f32 = b[..n].iter().sum::<f32>() * inv;
    let mut cov = 0.0f32;
    let mut va = 0.0f32;
    let mut vb = 0.0f32;
    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    let denom = (va * vb).sqrt();
    if denom <= f32::EPSILON {
        0.0
    } else {
        cov / denom
    }
}

/// Maximum absolute first-difference over the window `[idx-r, idx+r]`.
fn local_max_first_diff(signal: &[f32], idx: usize, r: usize) -> f32 {
    let lo = idx.saturating_sub(r);
    let hi = (idx + r).min(signal.len().saturating_sub(1));
    let mut m = 0.0f32;
    let mut i = lo + 1;
    while i <= hi {
        m = m.max((signal[i] - signal[i - 1]).abs());
        i += 1;
    }
    m
}

#[test]
fn clicks_left_only_right_preserved_no_bleed() {
    const SR: u32 = 44100;
    const N: usize = 20000;
    const CLICKS: [usize; 3] = [4000, 9000, 15000];

    // Identical 330 Hz base tone on both channels.
    let base: Vec<f32> = (0..N)
        .map(|i| 0.4 * (2.0 * PI * 330.0 * i as f32 / SR as f32).sin())
        .collect();

    // Left = base + click spikes; right = base (perfectly clean).
    let mut left = base.clone();
    for (k, &p) in CLICKS.iter().enumerate() {
        left[p] += if k % 2 == 0 { 0.9 } else { -0.9 };
    }
    let right = base.clone();

    // Click-removal-only chain.
    let mut chain = RestoreChain::new();
    chain.add_step(RestorationStep::ClickRemoval {
        detector: ClickDetector::new(ClickDetectorConfig::default()),
        remover: ClickRemover::new(InterpolationMethod::Cubic, 2),
    });

    let (out_l, out_r) = chain
        .process_stereo(&left, &right, SR)
        .expect("stereo click removal should succeed");

    assert_eq!(out_l.len(), left.len(), "left length preserved");
    assert_eq!(out_r.len(), right.len(), "right length preserved");

    // The clean right channel must be returned bit-close: NO cross-channel
    // bleed from processing the left channel's clicks.
    let max_right_dev = out_r
        .iter()
        .zip(right.iter())
        .map(|(&o, &r)| (o - r).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_right_dev < 1e-6,
        "clean right channel must be untouched (max deviation = {max_right_dev}); \
         any larger value indicates a cross-channel state-leak bug in process_stereo"
    );

    // The left channel's clicks must be flattened.
    for &p in &CLICKS {
        let before = local_max_first_diff(&left, p, 4);
        let after = local_max_first_diff(&out_l, p, 4);
        // A single additive ±0.9 spike yields a first-difference of ≈0.9
        // (one sample jumps by 0.9, then back) — sanity-check it is clearly
        // present before restoration.
        assert!(
            before > 0.5,
            "left click at {p} should be visible before (Δ={before})"
        );
        assert!(
            after < 0.3,
            "left click at {p} not removed: local |Δ| after = {after}"
        );
    }

    // The cleaned left channel should track the base tone very closely.
    let corr = pearson(&out_l, &base);
    assert!(
        corr > 0.95,
        "cleaned left channel should correlate with base tone (got {corr})"
    );
}
