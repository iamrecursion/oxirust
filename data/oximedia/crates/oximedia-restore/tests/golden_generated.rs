//! Generated golden test for single-click removal.
//!
//! Rather than ship a binary fixture, this test *computes* the expected
//! outcome of removing one known click from a known clean tone and asserts
//! against those analytic expectations:
//!
//!  1. the residual energy in the small window around the click collapses,
//!  2. the repaired sample at the click index matches the clean tone closely,
//!  3. every sample *outside* the touched window is left bit-for-bit identical
//!     to the dirty input (the remover must be strictly local).

use oximedia_restore::click::{ClickDetector, ClickDetectorConfig, ClickRemover};
use oximedia_restore::utils::interpolation::InterpolationMethod;
use oximedia_restore::{RestorationStep, RestoreChain};
use std::f32::consts::PI;

#[test]
fn golden_single_click_removed_at_known_index() {
    const SR: u32 = 44100;
    const N: usize = 8000;
    const CLICK_IDX: usize = 4000;
    // Window guaranteed to enclose the detector region (click region ±2 padding
    // around index 4000 spans roughly [3997, 4003]).
    const WIN_LO: usize = 3990;
    const WIN_HI: usize = 4010;

    // Pristine 220 Hz tone, amplitude 0.3.
    let clean: Vec<f32> = (0..N)
        .map(|i| 0.3 * (2.0 * PI * 220.0 * i as f32 / SR as f32).sin())
        .collect();

    // Inject exactly one +0.95 click.
    let mut dirty = clean.clone();
    dirty[CLICK_IDX] += 0.95;

    // Click-removal-only chain.
    let mut chain = RestoreChain::new();
    chain.add_step(RestorationStep::ClickRemoval {
        detector: ClickDetector::new(ClickDetectorConfig::default()),
        remover: ClickRemover::new(InterpolationMethod::Cubic, 2),
    });

    let out = chain
        .process(&dirty, SR)
        .expect("golden click removal should succeed");

    assert_eq!(out.len(), dirty.len(), "length preserved");

    // (1) Residual energy in the click window collapses near zero.
    let residual: f32 = (WIN_LO..WIN_HI).map(|i| (out[i] - clean[i]).powi(2)).sum();
    assert!(
        residual < 1e-3,
        "residual energy in click window should collapse (got {residual})"
    );

    // (2) The repaired sample matches the clean tone closely.
    let err = (out[CLICK_IDX] - clean[CLICK_IDX]).abs();
    assert!(
        err < 0.05,
        "repaired sample at {CLICK_IDX} should match clean tone (err = {err})"
    );

    // (3) Everything OUTSIDE the window is untouched (bit-close to dirty).
    for i in 0..N {
        if (WIN_LO..WIN_HI).contains(&i) {
            continue;
        }
        let dev = (out[i] - dirty[i]).abs();
        assert!(
            dev < 1e-6,
            "sample {i} outside click window must be untouched (deviation = {dev})"
        );
    }
}
