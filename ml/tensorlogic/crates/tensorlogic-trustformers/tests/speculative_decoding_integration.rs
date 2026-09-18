//! End-to-end integration test for model-level speculative decoding.
//!
//! This test exercises the public re-exports from the crate root and walks
//! through the canonical Leviathan et al. protocol:
//!
//! 1. Build a mock draft + target pair with **mismatched** distributions.
//! 2. Run `generate` for a long continuation.
//! 3. Confirm that (a) the emitted sequence has exactly the requested length,
//!    (b) the running accept-rate is below 1 (rejection branch fired) and
//!    above 0 (acceptance branch fired), (c) the empirical distribution of
//!    emitted tokens is statistically indistinguishable from the target's
//!    fixed distribution at the 0.01 significance level.

use scirs2_core::random::{SeedableRng, StdRng};
use tensorlogic_trustformers::{
    FixedDistDraftModel, FixedDistTargetModel, SpecRng, SpeculativeDecoder,
    SpeculativeDecoderConfig,
};

const CHI_SQ_CRIT_01_DF_3: f64 = 11.345;

fn chi_square(observed: &[usize], expected_probs: &[f64]) -> f64 {
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

#[test]
fn end_to_end_speculative_generation_matches_target_distribution() {
    let target_probs = vec![0.1, 0.2, 0.3, 0.4];
    let draft_probs = vec![0.4, 0.3, 0.2, 0.1];

    let draft = FixedDistDraftModel::new(draft_probs).expect("draft");
    let target = FixedDistTargetModel::new(target_probs.clone()).expect("target");
    let cfg = SpeculativeDecoderConfig::default().with_k(4);
    let mut dec = SpeculativeDecoder::new(draft, target, cfg).expect("decoder");

    let mut rng = StdRng::seed_from_u64(4711);
    let output = dec
        .generate_with_rng(&[0], 10_000, &mut rng as &mut dyn SpecRng)
        .expect("generate");

    assert_eq!(output.len(), 10_000);

    // Sanity: both branches of rejection sampling ran.
    assert!(dec.metrics().accept_rate > 0.0);
    assert!(dec.metrics().accept_rate < 1.0);
    assert!(dec.metrics().rounds > 0);
    assert!(dec.metrics().total_committed as usize == output.len());

    // Empirical distribution test.
    let mut counts = vec![0usize; 4];
    for &t in &output {
        counts[t] += 1;
    }
    let stat = chi_square(&counts, &target_probs);
    assert!(
        stat < CHI_SQ_CRIT_01_DF_3,
        "integration chi-square {} >= critical {}; counts {:?}",
        stat,
        CHI_SQ_CRIT_01_DF_3,
        counts
    );
}

#[test]
fn end_to_end_respects_k_config_and_bonus_position() {
    // With target == draft everything accepts, so each round commits k + 1
    // tokens.  Over M rounds we therefore expect (k + 1) * M tokens — which
    // the engine should respect exactly.
    let probs = vec![0.25, 0.25, 0.25, 0.25];
    let draft = FixedDistDraftModel::new(probs.clone()).expect("d");
    let target = FixedDistTargetModel::new(probs).expect("t");
    let cfg = SpeculativeDecoderConfig::default().with_k(4);
    let mut dec = SpeculativeDecoder::new(draft, target, cfg).expect("decoder");

    let out = dec.generate(&[0], 25).expect("generate");
    assert_eq!(out.len(), 25);
    // Draft == target ⇒ accept_rate must be 1.0 and tokens_per_step ≈ 5.
    assert!(dec.metrics().accept_rate > 0.9999);
    assert!(dec.metrics().tokens_per_step_avg > 4.9);
}
