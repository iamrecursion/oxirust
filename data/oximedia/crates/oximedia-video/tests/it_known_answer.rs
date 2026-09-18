//! Deterministic known-answer integration tests for `oximedia-video`
//! (Wave 29 / Slice 7 — pure test-hardening, no production change).
//!
//! These tests pin closed-form / hand-computed oracles for three subsystems:
//!
//! * **Motion estimation** ([`motion_compensation_ext::estimate_frame_motion_parallel`]).
//!   Sign convention: the engine searches the *reference* for the current
//!   block (candidate `rx = mb_x + dx`), so content shifted RIGHT by 8 px in
//!   the current frame matches the reference at `dx = -8`, i.e. MV `(-8, 0)`.
//!   Verified against a brute-force SAD oracle in Python before writing.
//!
//! * **Integral image** ([`integral_image::IntegralImage`]). `build(&[u8], w, h)`
//!   over an 8×8 gradient `px(x,y) = y*8 + x` (values 0..=63):
//!   `rect_sum == 2016`, `rect_sum_sq == 85344`, `rect_variance == 341.25`
//!   (= `(64² − 1)/12`, the population variance of a discrete uniform 0..=63).
//!   Upper bounds are EXCLUSIVE; `rect_variance` is population variance (÷N).
//!
//! * **Pulldown cadence detection** — two independent modules:
//!   [`pulldown_detect::detect_cadence`] and
//!   [`cadence_confidence::CadenceConfidenceScorer`].
//!
//! # Known latent issues (Wave 29) — DOCUMENTED, NOT FIXED
//!
//! Two real cross-module inconsistencies were confirmed while writing these
//! tests. They are surfaced here (and in the slice report) rather than fixed,
//! because picking a canonical phase / threshold is a design decision that is
//! out of scope for a pure test-hardening slice.
//!
//! * **Bug #1 — divergent 3:2 reference phase.**
//!   [`pulldown_detect::detect_cadence`] matches the 3:2 cadence against the
//!   pattern `[L, L, H, L, H]`
//!   (`pulldown_detect.rs:205`, `let pulldown_32 = [false, false, true, false, true];`),
//!   whereas [`cadence_confidence::cadence_pattern`] uses `[H, L, H, L, L]`
//!   for the same `Cadence::Pulldown32`
//!   (`cadence_confidence.rs:83`, `Cadence::Pulldown32 => &[HIGH, LOW, HIGH, LOW, LOW]`).
//!   These are different phase rotations of the identical underlying cadence,
//!   so a caller that mixes the two modules (e.g. detects phase with one and
//!   scores confidence with the other) sees an inconsistent phase offset of 2
//!   frames. Each module is internally self-consistent; the tests below feed
//!   each module its OWN reference phase.
//!
//! * **Bug #2 — divergent LOW / High threshold.**
//!   [`pulldown_detect::detect_cadence`] classifies a frame as "High" when
//!   `combing_score > 0.04` (`pulldown_detect.rs:195`,
//!   `.map(|m| m.combing_score > 0.04)`), but
//!   [`cadence_confidence`] uses a nominal `LOW = 0.05`
//!   (`cadence_confidence.rs:72`, `const LOW: f32 = 0.05;`).
//!   Feeding `cadence_confidence`'s nominal LOW value of `0.05` straight into
//!   `detect_cadence` reads as High on *every* frame (`0.05 > 0.04`), which
//!   would mis-classify a clean 3:2 sequence as `Interlaced`. The
//!   `detect_cadence` tests below therefore deliberately keep LOW values at or
//!   below `0.04` so that LOW reads as Low.

use oximedia_video::cadence_confidence::CadenceConfidenceScorer;
use oximedia_video::integral_image::IntegralImage;
use oximedia_video::motion_compensation_ext::estimate_frame_motion_parallel;
use oximedia_video::pulldown_detect::{detect_cadence, Cadence, FieldMetrics};

// ===========================================================================
// (a) Motion estimation — known-answer block displacement
// ===========================================================================

/// A 48×16 luma frame whose column texture is strongly shift-sensitive, with
/// `current` equal to `reference` shifted RIGHT by 8 px (edge-replicated on
/// the left). The engine searches the reference for each current block, so a
/// right-shift of 8 px is recovered as motion vector `(-8, 0)`.
///
/// Macroblock layout (block = 16): three columns at `mb_x ∈ {0, 16, 32}`,
/// one row. Block 0 (`mb_x = 0`) CANNOT resolve to `(-8, 0)` because the
/// candidate `rx = 0 + (-8) = -8` is out of range and skipped by the search;
/// only the two interior blocks have an unambiguous SAD-zero match at
/// `(-8, 0)`, so we assert those.
fn build_shift_right_8_frames() -> (Vec<u8>, Vec<u8>, usize, usize) {
    let (w, h) = (48usize, 16usize);
    // Deterministic per-column texture: distinct, high-contrast, shift-sensitive.
    let reference: Vec<u8> = (0..w * h)
        .map(|i| {
            let x = i % w;
            ((x * 37 + 11) % 256) as u8
        })
        .collect();

    // current[x] = reference[x - 8]; left 8 columns replicate column 0.
    let mut current = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let sx = x.saturating_sub(8);
            current[y * w + x] = reference[y * w + sx];
        }
    }
    (reference, current, w, h)
}

#[test]
fn motion_right_shift_8px_yields_minus8_mv() {
    let (reference, current, w, h) = build_shift_right_8_frames();
    let block_size = 16usize;
    let search_range = 8i32;

    let mvs = estimate_frame_motion_parallel(&reference, &current, w, h, block_size, search_range);

    // 48/16 = 3 blocks across, 16/16 = 1 down → 3 macroblocks in raster order.
    assert_eq!(mvs.len(), 3, "expected 3 macroblocks, got {}", mvs.len());

    // Interior blocks (mb_x = 16 and mb_x = 32) match the reference exactly at
    // dx = -8 (SAD = 0): content shifted RIGHT by 8 px is recovered as (-8, 0).
    assert_eq!(
        mvs[1],
        (-8, 0),
        "middle block: right-shift of 8 px must recover MV (-8, 0), got {:?}",
        mvs[1]
    );
    assert_eq!(
        mvs[2],
        (-8, 0),
        "right block: right-shift of 8 px must recover MV (-8, 0), got {:?}",
        mvs[2]
    );

    // Block 0 cannot reach dx = -8 (candidate rx = -8 is out of range and
    // skipped), so it resolves to some other in-range vector. We only assert
    // it is a valid in-range vector, not a specific value.
    let (dx0, dy0) = mvs[0];
    assert!(
        dx0 >= -search_range && dx0 <= search_range,
        "block 0 dx {dx0} out of search range"
    );
    assert!(
        dy0 >= -search_range && dy0 <= search_range,
        "block 0 dy {dy0} out of search range"
    );
}

#[test]
fn motion_identical_frames_all_zero() {
    // Identical frames → every macroblock has a zero-SAD match at (0, 0).
    let (w, h) = (32usize, 32usize);
    let frame: Vec<u8> = (0..w * h).map(|i| ((i * 13 + 7) % 256) as u8).collect();
    let mvs = estimate_frame_motion_parallel(&frame, &frame, w, h, 8, 4);
    // 32/8 * 32/8 = 16 macroblocks.
    assert_eq!(mvs.len(), 16);
    for (idx, mv) in mvs.iter().enumerate() {
        assert_eq!(*mv, (0, 0), "macroblock {idx}: identical frames → (0, 0)");
    }
}

// ===========================================================================
// (b) Integral image — closed-form summed-area-table oracles
// ===========================================================================

#[test]
fn integral_uniform_8x8_sum_and_zero_variance() {
    // 8×8 all-100 → sum = 100 * 64 = 6400, population variance = 0.
    let frame = vec![100u8; 8 * 8];
    let ii = IntegralImage::build(&frame, 8, 8);

    assert_eq!(ii.rect_sum(0, 0, 8, 8), 6400, "8×8 uniform-100 sum");

    let var = ii.rect_variance(0, 0, 8, 8);
    assert!(
        var.abs() < 1e-9,
        "uniform image variance must be 0.0, got {var}"
    );
}

#[test]
fn integral_gradient_8x8_closed_form() {
    // 8×8 gradient px(x,y) = y*8 + x → values 0..=63.
    let frame: Vec<u8> = (0u8..64u8).collect();
    let ii = IntegralImage::build(&frame, 8, 8);

    // Σ 0..=63 = 2016.
    assert_eq!(ii.rect_sum(0, 0, 8, 8), 2016, "gradient full-frame sum");

    // Σ i² for i in 0..=63 = 85344.
    assert_eq!(
        ii.rect_sum_sq(0, 0, 8, 8),
        85344,
        "gradient full-frame sum-of-squares"
    );

    // Population variance of discrete uniform 0..=63 = (64² − 1)/12 = 341.25.
    let var = ii.rect_variance(0, 0, 8, 8);
    let expected = (64.0 * 64.0 - 1.0) / 12.0; // 341.25
    assert!(
        (var - expected).abs() < 1e-9,
        "gradient population variance: expected {expected}, got {var}"
    );
    assert!(
        (var - 341.25).abs() < 1e-9,
        "gradient population variance must equal 341.25, got {var}"
    );
}

#[test]
fn integral_gradient_subrect_exclusive_upper_bound() {
    // Subrect rect_sum(2, 1, 5, 4): rows y ∈ {1, 2, 3}, cols x ∈ {2, 3, 4}
    // (exclusive upper bound). px(x,y) = y*8 + x.
    // y=1: 10+11+12 = 33; y=2: 18+19+20 = 57; y=3: 26+27+28 = 81 → total 171.
    let frame: Vec<u8> = (0u8..64u8).collect();
    let ii = IntegralImage::build(&frame, 8, 8);
    assert_eq!(
        ii.rect_sum(2, 1, 5, 4),
        171,
        "3×3 subrect at (2,1)-(5,4) exclusive must sum to 171"
    );
}

// ===========================================================================
// (c) Pulldown cadence — detect_cadence (module: pulldown_detect)
// ===========================================================================
//
// detect_cadence uses the 3:2 reference phase [L, L, H, L, H] and the High
// threshold `combing_score > 0.04` (see bugs #1 and #2 in the module doc
// above). We therefore feed its OWN phase with LOW values ≤ 0.04.

fn metrics_from_scores(scores: &[f32]) -> Vec<FieldMetrics> {
    scores
        .iter()
        .enumerate()
        .map(|(i, &s)| FieldMetrics {
            frame_number: i as u64,
            combing_score: s,
            tff: true,
        })
        .collect()
}

#[test]
fn detect_cadence_32_pattern() {
    // [L, L, H, L, H] with LOW = 0.0 (≤ 0.04 → Low) and HIGH = 0.5 (> 0.04).
    let metrics = metrics_from_scores(&[0.0, 0.0, 0.5, 0.0, 0.5]);
    let refs: Vec<&FieldMetrics> = metrics.iter().collect();
    assert_eq!(
        detect_cadence(&refs),
        Cadence::Pulldown32,
        "[L,L,H,L,H] with LOW≤0.04 must classify as Pulldown32"
    );
}

#[test]
fn detect_cadence_low_threshold_boundary() {
    // Bug #2 pin: exactly 0.04 reads as Low (NOT High), since the test is
    // strict `> 0.04`. An all-0.04 sequence is therefore Progressive, not
    // Interlaced. Feeding cadence_confidence's nominal LOW (0.05) instead
    // would read as all-High (→ Interlaced) — the documented inconsistency.
    let at_threshold = metrics_from_scores(&[0.04, 0.04, 0.04, 0.04, 0.04]);
    let refs: Vec<&FieldMetrics> = at_threshold.iter().collect();
    assert_eq!(
        detect_cadence(&refs),
        Cadence::Progressive,
        "score == 0.04 must read as Low (strict > 0.04), so all-0.04 → Progressive"
    );

    // And 0.05 (cadence_confidence's nominal LOW) reads as High everywhere.
    let above = metrics_from_scores(&[0.05, 0.05, 0.05, 0.05, 0.05]);
    let refs_above: Vec<&FieldMetrics> = above.iter().collect();
    assert_eq!(
        detect_cadence(&refs_above),
        Cadence::Interlaced,
        "score 0.05 > 0.04 reads as High on every frame → Interlaced (bug #2)"
    );
}

// ===========================================================================
// (c) Pulldown cadence — CadenceConfidenceScorer (module: cadence_confidence)
// ===========================================================================
//
// The confidence scorer uses the 3:2 reference phase [H, L, H, L, L] with
// nominal HIGH = 0.70 / LOW = 0.05 and sigma = 0.12. Feeding exactly this
// phase yields winner = Pulldown32 with probability ≈ 1.0.

const CC_HIGH: f32 = 0.70;
const CC_LOW: f32 = 0.05;

fn push_pulldown32_pattern(scorer: &mut CadenceConfidenceScorer, frames: usize) {
    // cadence_confidence's own 3:2 phase: [H, L, H, L, L].
    let pattern = [CC_HIGH, CC_LOW, CC_HIGH, CC_LOW, CC_LOW];
    for i in 0..frames {
        scorer.push(FieldMetrics {
            frame_number: i as u64,
            combing_score: pattern[i % pattern.len()],
            tff: true,
        });
    }
}

#[test]
fn confidence_scorer_32_winner_high_probability() {
    // Repeat the [H,L,H,L,L] cycle for 5, 10, 20, and 40 frames; the winner
    // must be Pulldown32 with probability > 0.9 (≈ 1.0) in every case.
    for &frames in &[5usize, 10, 20, 40] {
        let mut scorer = CadenceConfidenceScorer::new();
        push_pulldown32_pattern(&mut scorer, frames);

        let score = scorer
            .score()
            .expect("scorer has at least min_frames (5) observations");

        assert_eq!(
            score.winner,
            Cadence::Pulldown32,
            "[H,L,H,L,L]×{frames} must elect Pulldown32 as winner"
        );
        assert_eq!(
            score.best_cadence(),
            Cadence::Pulldown32,
            "best_cadence() must agree with winner for {frames} frames"
        );
        assert!(
            score.winner_probability > 0.9,
            "winner probability for {frames} frames must be > 0.9, got {}",
            score.winner_probability
        );
        assert_eq!(
            score.frame_count, frames,
            "frame_count must equal the number of pushed frames"
        );
    }
}

#[test]
fn confidence_scorer_probabilities_sum_to_one() {
    // Sanity: the softmax posterior over the five hypotheses must sum to ≈ 1.0.
    let mut scorer = CadenceConfidenceScorer::new();
    push_pulldown32_pattern(&mut scorer, 20);
    let score = scorer.score().expect("enough frames");
    let sum: f32 = score.hypotheses.iter().map(|h| h.probability).sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "posterior probabilities must sum to 1.0, got {sum}"
    );
}
