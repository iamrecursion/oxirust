//! Unit tests for speculative decoding.
//!
//! The decisive test is [`empirical_distribution_matches_target`]: it runs the
//! decoder 10 000 times with a **biased** draft and a known target, then
//! verifies via a chi-square goodness-of-fit statistic that the empirical
//! distribution over emitted tokens matches `p_target`.  If the rejection
//! sampling math is wrong, this test fails.
//!
//! Because we cannot add a statistics dependency, the chi-square critical
//! value for `df = vocab_size - 1` at the 0.01 significance level is
//! hard-coded in [`chi_sq_critical_01`] from the standard table.

use scirs2_core::random::{SeedableRng, StdRng};

use super::acceptance::{accept, adjusted_distribution, resample_from_adjusted_target};
use super::engine::{SpeculativeDecoder, SpeculativeDecoderConfig};
use super::error::SpeculativeDecodingError;
use super::metrics::SpeculativeMetrics;
use super::mock_models::{FixedDistDraftModel, FixedDistTargetModel};
use super::rng::SpecRng;

/// Upper-tail chi-square critical value at α = 0.01 for degrees of freedom
/// 1..=10, taken from standard statistical tables.  A fit is "not rejected
/// at the 0.01 level" when the computed statistic is *below* this threshold.
const CHI_SQ_CRIT_01: [f64; 10] = [
    6.635,  // df=1
    9.210,  // df=2
    11.345, // df=3
    13.277, // df=4
    15.086, // df=5
    16.812, // df=6
    18.475, // df=7
    20.090, // df=8
    21.666, // df=9
    23.209, // df=10
];

/// Return the chi-square critical value at α = 0.01 for degrees of freedom
/// `df` in the range `1..=10`.  Panics for `df` outside the table — tests
/// must stay within it.
fn chi_sq_critical_01(df: usize) -> f64 {
    assert!(
        df >= 1 && df <= CHI_SQ_CRIT_01.len(),
        "df out of range: {}",
        df
    );
    CHI_SQ_CRIT_01[df - 1]
}

/// Compute the Pearson chi-square statistic between observed counts and an
/// expected probability distribution.
fn chi_square(observed: &[usize], expected_probs: &[f64]) -> f64 {
    assert_eq!(observed.len(), expected_probs.len());
    let n: f64 = observed.iter().sum::<usize>() as f64;
    let mut stat = 0.0;
    for (o, p) in observed.iter().zip(expected_probs.iter()) {
        let e = p * n;
        if e > 0.0 {
            let diff = *o as f64 - e;
            stat += diff * diff / e;
        }
    }
    stat
}

// -- Core acceptance rule -----------------------------------------------------

#[test]
fn accept_when_draft_equals_target_always_accepts() {
    let mut rng = StdRng::seed_from_u64(11);
    // Same distribution ⇒ ratio = 1 ⇒ always accept.
    for _ in 0..1000 {
        assert!(accept(-0.5, -0.5, &mut rng));
        assert!(accept(-1.3, -1.3, &mut rng));
    }
}

#[test]
fn accept_miscalibrated_draft_mixes() {
    let mut rng = StdRng::seed_from_u64(13);
    let mut n_accept = 0;
    let mut n_reject = 0;
    for _ in 0..2000 {
        // ratio = exp(-1.5) ≈ 0.22.
        if accept(-0.2, -1.7, &mut rng) {
            n_accept += 1;
        } else {
            n_reject += 1;
        }
    }
    assert!(n_accept > 0 && n_reject > 0);
    let rate = n_accept as f64 / 2000.0;
    // Expected acceptance ~0.22; allow 0.1 slack.
    assert!((rate - 0.22).abs() < 0.1, "empirical accept rate {}", rate);
}

// -- Adjusted distribution math ----------------------------------------------

#[test]
fn adjusted_distribution_is_nonnegative_and_normalized() {
    // p_target = (0.1, 0.2, 0.3, 0.4)
    let tgt: Vec<f64> = vec![0.1f64.ln(), 0.2f64.ln(), 0.3f64.ln(), 0.4f64.ln()];
    // p_draft = (0.25, 0.25, 0.25, 0.25)
    let drf: Vec<f64> = vec![0.25f64.ln(); 4];
    let q = adjusted_distribution(&tgt, &drf).expect("adjusted");
    for &p in &q {
        assert!(p >= 0.0);
    }
    let sum: f64 = q.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9);
    // The mass should be concentrated on indices where p_target > p_draft.
    // max(0, 0.1 - 0.25) = 0, max(0, 0.2 - 0.25) = 0,
    // max(0, 0.3 - 0.25) = 0.05, max(0, 0.4 - 0.25) = 0.15 → normalized:
    // [0, 0, 0.25, 0.75].
    assert!((q[0] - 0.0).abs() < 1e-9);
    assert!((q[1] - 0.0).abs() < 1e-9);
    assert!((q[2] - 0.25).abs() < 1e-6);
    assert!((q[3] - 0.75).abs() < 1e-6);
}

#[test]
fn resample_returns_token_in_support() {
    // p_target places all probability on index 3; p_draft is uniform.
    let tgt: Vec<f64> = vec![f64::NEG_INFINITY; 3]
        .into_iter()
        .chain(std::iter::once(0.0))
        .collect();
    let drf: Vec<f64> = vec![0.25f64.ln(); 4];
    let mut rng = StdRng::seed_from_u64(23);
    for _ in 0..100 {
        let idx = resample_from_adjusted_target(&tgt, &drf, &mut rng).expect("resample");
        assert_eq!(idx, 3, "adjusted distribution should point at index 3");
    }
}

// -- Metrics updates ----------------------------------------------------------

#[test]
fn metrics_update_on_round() {
    let mut m = SpeculativeMetrics::new();
    m.record_round(4, 2, 3, 4);
    assert!((m.accept_rate - 0.5).abs() < 1e-6);
    assert!((m.tokens_per_step_avg - 3.0).abs() < 1e-6);
    assert_eq!(m.rounds, 1);
}

// -- Mock models end-to-end ---------------------------------------------------

#[test]
fn mock_decoder_emits_tokens_when_draft_equals_target() {
    let probs = vec![0.1, 0.2, 0.3, 0.4];
    let draft = FixedDistDraftModel::new(probs.clone()).expect("draft");
    let target = FixedDistTargetModel::new(probs).expect("target");
    let mut dec = SpeculativeDecoder::new(draft, target, SpeculativeDecoderConfig::default())
        .expect("decoder");
    let out = dec.generate(&[0], 32).expect("generate");
    assert_eq!(out.len(), 32);
    // Since draft==target, accept rate should be 1.0.
    assert!(dec.metrics().accept_rate > 0.999);
}

#[test]
fn k_equals_one_degenerate_case_works() {
    let probs = vec![0.25, 0.25, 0.25, 0.25];
    let draft = FixedDistDraftModel::new(probs.clone()).expect("draft");
    let target = FixedDistTargetModel::new(probs).expect("target");
    let cfg = SpeculativeDecoderConfig::default().with_k(1);
    let mut dec = SpeculativeDecoder::new(draft, target, cfg).expect("decoder");
    let out = dec.generate(&[0], 10).expect("generate");
    assert_eq!(out.len(), 10);
}

#[test]
fn empty_prefix_errors_out() {
    let probs = vec![0.5, 0.5];
    let draft = FixedDistDraftModel::new(probs.clone()).expect("draft");
    let target = FixedDistTargetModel::new(probs).expect("target");
    let mut dec = SpeculativeDecoder::new(draft, target, SpeculativeDecoderConfig::default())
        .expect("decoder");
    let err = dec.generate(&[], 5).expect_err("empty prefix");
    assert!(matches!(err, SpeculativeDecodingError::EmptyPrefix));
}

#[test]
fn vocab_mismatch_errors_out() {
    let draft = FixedDistDraftModel::new(vec![0.5, 0.5]).expect("draft");
    let target = FixedDistTargetModel::new(vec![0.25; 4]).expect("target");
    let err = SpeculativeDecoder::new(draft, target, SpeculativeDecoderConfig::default())
        .expect_err("vocab mismatch");
    assert!(matches!(
        err,
        SpeculativeDecodingError::VocabMismatch { .. }
    ));
}

#[test]
fn miscalibrated_draft_still_emits_correct_count() {
    // Draft favors index 0 heavily; target is uniform.  Resampling path must
    // still emit the requested number of tokens.
    let draft = FixedDistDraftModel::new(vec![0.8, 0.05, 0.05, 0.1]).expect("draft");
    let target = FixedDistTargetModel::new(vec![0.25; 4]).expect("target");
    let mut dec = SpeculativeDecoder::new(draft, target, SpeculativeDecoderConfig::default())
        .expect("decoder");
    let out = dec.generate(&[0], 64).expect("generate");
    assert_eq!(out.len(), 64);
    // Accept rate should be strictly below 1.0 since draft != target.
    assert!(dec.metrics().accept_rate < 0.99);
}

// -- The correctness theorem test --------------------------------------------

#[test]
fn empirical_distribution_matches_target() {
    // The most important test: under arbitrary draft, the emitted-token
    // distribution must match p_target.  We take 10_000 samples of the very
    // first emitted token across independent runs.
    //
    // Using 4-element vocab so df = 3 → chi-square critical value 11.345.
    let target_probs = vec![0.1, 0.2, 0.3, 0.4];
    let draft_probs = vec![0.4, 0.3, 0.2, 0.1]; // deliberately anti-correlated.
    let n_samples: usize = 10_000;

    let target = FixedDistTargetModel::new(target_probs.clone()).expect("target");
    let draft = FixedDistDraftModel::new(draft_probs).expect("draft");
    let mut dec =
        SpeculativeDecoder::new(draft, target, SpeculativeDecoderConfig::default().with_k(4))
            .expect("decoder");

    let mut rng = StdRng::seed_from_u64(2026);
    let mut counts = vec![0usize; 4];
    for _ in 0..n_samples {
        let out = dec
            .generate_with_rng(&[0], 1, &mut rng as &mut dyn SpecRng)
            .expect("generate");
        assert_eq!(out.len(), 1);
        counts[out[0]] += 1;
    }
    let stat = chi_square(&counts, &target_probs);
    let crit = chi_sq_critical_01(target_probs.len() - 1);
    assert!(
        stat < crit,
        "chi-square {} >= critical {}; counts {:?} expected {:?}",
        stat,
        crit,
        counts,
        target_probs
    );
    // Also assert the accept rate is meaningfully less than 1 (proving the
    // resampling path actually fired).
    assert!(dec.metrics().accept_rate < 0.9);
}

#[test]
fn empirical_distribution_matches_target_multi_step() {
    // Same guarantee, but we let the decoder emit many tokens in one go.
    // Because the target is context-free, every emitted position should
    // marginally match p_target.
    let target_probs = vec![0.1, 0.2, 0.3, 0.4];
    let draft_probs = vec![0.5, 0.3, 0.15, 0.05];
    let target = FixedDistTargetModel::new(target_probs.clone()).expect("target");
    let draft = FixedDistDraftModel::new(draft_probs).expect("draft");
    let mut dec =
        SpeculativeDecoder::new(draft, target, SpeculativeDecoderConfig::default().with_k(4))
            .expect("decoder");

    let out = dec.generate(&[0], 10_000).expect("gen");
    assert_eq!(out.len(), 10_000);
    let mut counts = vec![0usize; 4];
    for &t in &out {
        counts[t] += 1;
    }
    let stat = chi_square(&counts, &target_probs);
    let crit = chi_sq_critical_01(target_probs.len() - 1);
    assert!(
        stat < crit,
        "multi-step chi-square {} >= critical {}; counts {:?}",
        stat,
        crit,
        counts
    );
}

#[test]
fn speedup_estimate_positive_after_rounds() {
    let draft = FixedDistDraftModel::new(vec![0.25; 4]).expect("draft");
    let target = FixedDistTargetModel::new(vec![0.25; 4]).expect("target");
    let mut dec = SpeculativeDecoder::new(draft, target, SpeculativeDecoderConfig::default())
        .expect("decoder");
    dec.generate(&[0], 50).expect("gen");
    assert!(dec.metrics().rounds > 0);
    assert!(dec.metrics().speedup_estimate > 0.0);
}

#[test]
fn decoder_config_eos_triggers_early_stop() {
    // Target forces index 0 (the EOS), draft matches → we stop immediately.
    let target = FixedDistTargetModel::new(vec![1.0, 1e-9, 1e-9, 1e-9]).expect("t");
    let draft = FixedDistDraftModel::new(vec![1.0, 1e-9, 1e-9, 1e-9]).expect("d");
    let cfg = SpeculativeDecoderConfig::default().with_eos(0);
    let mut dec = SpeculativeDecoder::new(draft, target, cfg).expect("decoder");
    let out = dec.generate(&[5], 64).expect("gen");
    assert!(out.len() <= 64);
    assert!(out.contains(&0));
}
