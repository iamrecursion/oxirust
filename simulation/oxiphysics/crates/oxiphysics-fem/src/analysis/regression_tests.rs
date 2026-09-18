//! Regression tests for the dense analysis drivers.

use super::*;

/// A 500 MPa steel: 250 MPa amplitude lasts 10⁶ cycles, 200 MPa is the
/// endurance limit.
fn steel_fatigue_model() -> FatigueLifeEstimator {
    FatigueLifeEstimator::new(500e6, 250e6, 1e6, 200e6).expect("valid fatigue parameters")
}

#[test]
fn test_fatigue_life_below_endurance() {
    let est = steel_fatigue_model();
    let life = est.cycles_to_failure(100e6);
    assert!(
        life == f64::INFINITY,
        "below endurance should return infinity: {life}"
    );
    // The endurance limit itself is a run-out as well.
    assert_eq!(est.cycles_to_failure(200e6), f64::INFINITY);
    assert_eq!(est.cycles_to_failure(0.0), f64::INFINITY);
    assert_eq!(est.cycles_to_failure(-5e6), f64::INFINITY);
}

#[test]
fn test_fatigue_life_at_ultimate() {
    let est = steel_fatigue_model();
    assert_eq!(est.cycles_to_failure(500e6), 1.0);
    assert_eq!(est.cycles_to_failure(900e6), 1.0);
}

#[test]
fn test_fatigue_life_passes_through_the_reference_point() {
    // N(σ_ref) = N_ref: the former prefactor gave N_ref² here (1e12 cycles).
    let est = steel_fatigue_model();
    let life = est.cycles_to_failure(250e6);
    assert!(
        (life / 1e6 - 1.0).abs() < 1e-9,
        "N(sigma_ref) = {life}, expected 1e6"
    );
    // The same must hold for an anchor that is not half the UTS.
    let other = FatigueLifeEstimator::new(900e6, 360e6, 2e5, 300e6).expect("valid");
    let life = other.cycles_to_failure(360e6);
    assert!((life / 2e5 - 1.0).abs() < 1e-9, "N = {life}");
}

#[test]
fn test_fatigue_life_is_continuous_at_the_ultimate_strength() {
    // The old formula returned N_ref just below the UTS and then jumped to one.
    let est = steel_fatigue_model();
    // N = (σ_uts/σ)^m ≈ 1 + m·ε/σ_uts with m ≈ 19.9, so N − 1 stays below
    // 25·ε/σ_uts and tends to zero with ε.
    for epsilon in [1e-3, 1.0, 1e3, 1e5] {
        let life = est.cycles_to_failure(500e6 - epsilon);
        let bound = 1.0 + 25.0 * epsilon / 500e6;
        assert!(
            (1.0..=bound).contains(&life),
            "N(sigma_uts - {epsilon}) = {life}, bound {bound}"
        );
    }
}

#[test]
fn test_fatigue_life_decreases_monotonically_with_amplitude() {
    let est = steel_fatigue_model();
    let mut previous = f64::INFINITY;
    for step in 1..=29 {
        let amplitude = 200e6 + step as f64 * 10e6;
        let life = est.cycles_to_failure(amplitude);
        assert!(
            life.is_finite() && life < previous,
            "life({amplitude}) = {life}"
        );
        previous = life;
    }
}

#[test]
fn test_basquin_exponent_follows_the_two_point_fit() {
    let est = steel_fatigue_model();
    // ln(1e6) / ln(500/250): the old code hard-wired σ_ref = σ_uts / 2.
    let expected = (1e6_f64).ln() / 2.0_f64.ln();
    assert!((est.basquin_exponent() - expected).abs() < 1e-12);
    let other = FatigueLifeEstimator::new(900e6, 360e6, 2e5, 300e6).expect("valid");
    let expected = (2e5_f64).ln() / (900.0_f64 / 360.0).ln();
    assert!((other.basquin_exponent() - expected).abs() < 1e-12);
    // Both anchors of the S-N line are reproduced by N = (σ_uts / σ)^m.
    let life = (other.sigma_uts / other.sigma_ref).powf(other.basquin_exponent());
    assert!((life / other.n_ref - 1.0).abs() < 1e-12);
}

#[test]
fn test_fatigue_endurance_limit_may_equal_the_reference_amplitude() {
    // A knee at the reference point: finite life above, run-out at and below.
    let est = FatigueLifeEstimator::new(500e6, 250e6, 1e6, 250e6).expect("valid");
    assert_eq!(est.cycles_to_failure(250e6), f64::INFINITY);
    let just_above = est.cycles_to_failure(250e6 * (1.0 + 1e-9));
    assert!((just_above / 1e6 - 1.0).abs() < 1e-6, "{just_above}");
}

#[test]
fn test_fatigue_nan_amplitude_is_nan() {
    assert!(steel_fatigue_model().cycles_to_failure(f64::NAN).is_nan());
}

#[test]
fn test_fatigue_constructor_rejects_invalid_parameters_without_panicking() {
    let invalid: [(f64, f64, f64, f64); 12] = [
        (0.0, 250e6, 1e6, 0.0),             // zero UTS
        (-500e6, 250e6, 1e6, 0.0),          // negative UTS
        (500e6, 0.0, 1e6, 0.0),             // zero reference amplitude
        (500e6, -1.0, 1e6, 0.0),            // negative reference amplitude
        (500e6, 500e6, 1e6, 0.0),           // reference amplitude not below the UTS
        (500e6, 600e6, 1e6, 0.0),           // reference amplitude above the UTS
        (500e6, 250e6, 1.0, 0.0),           // one-cycle reference life: m = 0
        (500e6, 250e6, 0.5, 0.0),           // sub-cycle reference life
        (500e6, 250e6, 1e6, -1.0),          // negative endurance limit
        (500e6, 250e6, 1e6, 300e6),         // endurance limit above sigma_ref
        (f64::NAN, 250e6, 1e6, 100e6),      // NaN
        (500e6, 250e6, f64::INFINITY, 0.0), // infinite reference life
    ];
    for (uts, reference, n_ref, endurance) in invalid {
        let result = FatigueLifeEstimator::new(uts, reference, n_ref, endurance);
        assert!(
            matches!(result, Err(FemError::InvalidInput(_))),
            "({uts}, {reference}, {n_ref}, {endurance}) must be rejected, got {result:?}"
        );
    }
    // The boundary of the valid region is accepted: no endurance limit at all.
    assert!(FatigueLifeEstimator::new(500e6, 250e6, 1e6, 0.0).is_ok());
}
