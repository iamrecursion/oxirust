// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of [`try_gjk_intersect`]: a boolean GJK run that exhausts its iteration
//! budget must be reported as [`Error::GjkNoConverge`] instead of being
//! conflated with "no intersection".

use super::test_support::Rng;
use super::*;
use crate::{Error, ErrorKind, RecoveryHint, Severity};

fn sphere(center: [f64; 3], radius: f64) -> GjkSphere {
    GjkSphere { center, radius }
}

fn cuboid(center: [f64; 3], half_extents: [f64; 3]) -> GjkBox {
    GjkBox {
        center,
        half_extents,
    }
}

#[test]
fn separated_shapes_are_reported_as_not_intersecting() {
    let a = sphere([0.0; 3], 1.0);
    let b = sphere([3.0, 0.0, 0.0], 1.0);
    assert!(matches!(try_gjk_intersect(&a, &b), Ok(false)));
    let box_a = cuboid([0.0; 3], [0.5; 3]);
    let box_b = cuboid([1.2, 0.3, -0.2], [0.5; 3]);
    assert!(matches!(try_gjk_intersect(&box_a, &box_b), Ok(false)));
}

#[test]
fn overlapping_shapes_are_reported_as_intersecting() {
    let a = sphere([0.0; 3], 1.0);
    let b = sphere([1.5, 0.0, 0.0], 1.0);
    assert!(matches!(try_gjk_intersect(&a, &b), Ok(true)));
    let box_a = cuboid([0.0; 3], [0.5; 3]);
    let box_b = cuboid([0.3, 0.2, 0.1], [0.5; 3]);
    assert!(matches!(try_gjk_intersect(&box_a, &box_b), Ok(true)));
    // Containment.
    let small = sphere([0.1, 0.0, 0.0], 0.2);
    assert!(matches!(try_gjk_intersect(&a, &small), Ok(true)));
}

#[test]
fn the_answer_agrees_with_gjk_intersect_whenever_the_run_converges() {
    let mut rng = Rng(0x5EED_1234_ABCD_9876);
    let mut decided = 0;
    for _ in 0..400 {
        let a = cuboid(rng.vec(-1.0, 1.0), rng.vec(0.2, 1.0));
        let b = sphere(rng.vec(-2.0, 2.0), rng.range(0.2, 1.2));
        match try_gjk_intersect(&a, &b) {
            Ok(answer) => {
                assert_eq!(answer, gjk_intersect(&a, &b));
                decided += 1;
            }
            Err(error) => {
                // The boolean test reports "no intersection" for an undecided run.
                assert!(matches!(error, Error::GjkNoConverge { .. }), "{error:?}");
                assert!(!gjk_intersect(&a, &b));
            }
        }
    }
    assert!(decided > 300, "only {decided} of 400 runs converged");
}

#[test]
fn exhausting_the_iteration_cap_is_a_convergence_error() {
    // Two unit spheres whose centres are 1.5 apart: they overlap by 0.5, but GJK
    // needs more than a step or two to prove it.
    let a = sphere([0.0; 3], 1.0);
    let b = sphere([1.5, 0.0, 0.0], 1.0);
    assert!(matches!(try_gjk_intersect_capped(&a, &b, 64), Ok(true)));

    // No step at all: the simplex is the first support point of A - B, 0.5 away
    // from the origin along x.
    match try_gjk_intersect_capped(&a, &b, 0) {
        Err(Error::GjkNoConverge { iterations, dist }) => {
            assert_eq!(iterations, 0);
            assert!((dist - 0.5).abs() < 1e-12, "dist = {dist}");
        }
        other => panic!("expected a convergence error, got {other:?}"),
    }
    // One step: the simplex is a segment through the origin.
    match try_gjk_intersect_capped(&a, &b, 1) {
        Err(Error::GjkNoConverge { iterations, dist }) => {
            assert_eq!(iterations, 1);
            assert!(
                dist.is_finite() && (0.0..=0.5).contains(&dist),
                "dist = {dist}"
            );
        }
        other => panic!("expected a convergence error, got {other:?}"),
    }
}

#[test]
fn the_convergence_error_is_classified_for_triage() {
    let a = sphere([0.0; 3], 1.0);
    let b = sphere([1.5, 0.0, 0.0], 1.0);
    let error = try_gjk_intersect_capped(&a, &b, 0).expect_err("zero steps cannot decide");
    assert_eq!(error.kind(), ErrorKind::Convergence);
    assert!(error.is_convergence());
    assert_eq!(error.severity(), Severity::Warning);
    assert_eq!(error.recovery_hint(), RecoveryHint::TreatAsSeparated);
    assert!(error.to_string().contains("GJK did not converge"));
}
