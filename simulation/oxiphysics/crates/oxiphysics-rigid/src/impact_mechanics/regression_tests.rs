// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the impact-mechanics models.
//!
//! The Taylor-anvil tests pin the dynamic-yield inversion: the previous
//! implementation collapsed algebraically to `ρ·c_L·v₀` for every final
//! length, so the reported dynamic yield strength ignored the measured
//! deformation altogether.

use super::*;

/// Copper-like rod: L₀ = 100 mm, D₀ = 10 mm, ρ = 8900 kg/m³, c_L = 4600 m/s,
/// Y = 200 MPa.
fn copper_rod() -> TaylorImpact {
    TaylorImpact::new(0.10, 0.01, 8900.0, 4600.0, 200e6)
}

#[test]
fn taylor_analysis_recovers_the_yield_strength_of_the_rod_model() {
    let rod = copper_rod();
    // Impact speeds for which the final-length model is not clamped.
    for &v0 in &[40.0, 90.0, 150.0, 180.0] {
        let result = rod.analyze(v0);
        assert!(
            result.final_length < rod.initial_length && result.final_length > rod.diameter,
            "v0 = {v0}: final length {} is clamped",
            result.final_length
        );
        let rel = (result.dynamic_yield_strength - rod.yield_strength).abs() / rod.yield_strength;
        assert!(
            rel < 1e-9,
            "v0 = {v0}: Y_d = {} vs Y = {}",
            result.dynamic_yield_strength,
            rod.yield_strength
        );
    }
}

#[test]
fn taylor_dynamic_yield_depends_on_the_final_length() {
    let rod = copper_rod();
    let v0 = 150.0;
    let y_short = rod.dynamic_yield_from_geometry(v0, 0.03);
    let y_long = rod.dynamic_yield_from_geometry(v0, 0.06);
    // Y_d ∝ 1 / (L₀ − L_f): a rod that shortened less is the stronger one.
    assert!(y_long > y_short);
    let expected_ratio = (0.10 - 0.03) / (0.10 - 0.06);
    assert!(
        ((y_long / y_short) - expected_ratio).abs() < 1e-12,
        "ratio {} vs {expected_ratio}",
        y_long / y_short
    );
    // Closed form ρ v₀² L₀ / (2 (L₀ − L_f)).
    let expected = 8900.0 * v0 * v0 * 0.10 / (2.0 * 0.04);
    assert!((y_long - expected).abs() < 1e-6 * expected);
}

#[test]
fn taylor_dynamic_yield_is_independent_of_the_elastic_wave_speed() {
    let slow = TaylorImpact::new(0.10, 0.01, 8900.0, 2000.0, 200e6);
    let fast = TaylorImpact::new(0.10, 0.01, 8900.0, 6000.0, 200e6);
    let a = slow.dynamic_yield_from_geometry(150.0, 0.05);
    let b = fast.dynamic_yield_from_geometry(150.0, 0.05);
    assert_eq!(a, b);
    // The old result ρ·c·v₀ would be 6.1e9 Pa for c = 4600 m/s; the physical
    // value for this deformation is of the order of the 200 MPa yield stress.
    assert!(a < 1.0e9, "Y_d = {a}");
}

#[test]
fn taylor_dynamic_yield_without_shortening_is_the_nominal_yield_strength() {
    let rod = copper_rod();
    assert_eq!(
        rod.dynamic_yield_from_geometry(100.0, rod.initial_length),
        rod.yield_strength
    );
    assert_eq!(
        rod.dynamic_yield_taylor(100.0, rod.initial_length, 0.05),
        rod.yield_strength
    );
}

#[test]
fn taylor_relation_with_undeformed_length_reduces_to_the_rod_model() {
    let rod = copper_rod();
    let (v0, lf) = (150.0, 0.05);
    let limit = rod.dynamic_yield_from_geometry(v0, lf);
    // X → L₀ (including the exact limit X = L₀).
    for &gap in &[1e-3, 1e-6, 1e-9, 0.0] {
        let y = rod.dynamic_yield_taylor(v0, lf, rod.initial_length * (1.0 - gap));
        assert!(
            (y - limit).abs() < 1e-3 * limit,
            "gap = {gap}: {y} vs {limit}"
        );
    }
    // A larger rigid (undeformed) section lowers the inferred strength.
    let y_half = rod.dynamic_yield_taylor(v0, lf, 0.5 * rod.initial_length);
    assert!(y_half < limit);
}

#[test]
fn taylor_relation_matches_the_closed_form_for_a_known_undeformed_length() {
    let rod = copper_rod();
    let (v0, lf) = (150.0, 0.05);
    // X = L₀/e gives ln(L₀/X) = 1 and (L₀ − X) = L₀·(1 − 1/e).
    let x = rod.initial_length / std::f64::consts::E;
    let y = rod.dynamic_yield_taylor(v0, lf, x);
    let expected =
        8900.0 * v0 * v0 * rod.initial_length * (1.0 - 1.0 / std::f64::consts::E) / (2.0 * 0.05);
    assert!(
        (y - expected).abs() < 1e-9 * expected,
        "Y_d = {y}, expected {expected}"
    );
}

#[test]
fn taylor_relation_clamps_the_undeformed_length_into_the_rod() {
    let rod = copper_rod();
    let y_over = rod.dynamic_yield_taylor(150.0, 0.05, 1.0);
    let y_full = rod.dynamic_yield_taylor(150.0, 0.05, rod.initial_length);
    assert_eq!(y_over, y_full);
    let y_zero = rod.dynamic_yield_taylor(150.0, 0.05, 0.0);
    assert!(y_zero.is_finite() && y_zero >= 0.0, "Y_d = {y_zero}");
}
