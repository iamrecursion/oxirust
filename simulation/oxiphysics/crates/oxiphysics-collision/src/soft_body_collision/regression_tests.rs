// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the exact edge-edge CCD and the soft-body BVH.
//!
//! The edge tests pin the time of impact to analytic values and to independent
//! oracles (dense time sampling, a golden-section search for the distance of two
//! segments, brute-force pair enumeration). The BVH tests check balance on
//! adversarial input orders, the structural invariants of the flat array and the
//! query results against brute force.

use super::edge_ccd::{segment_pair, unit_interval_roots};
use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn vec_close(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    len3(sub3(a, b)) <= tol
}

/// Deterministic xorshift generator for the randomised oracle tests.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    fn vec(&mut self, lo: f64, hi: f64) -> [f64; 3] {
        [self.range(lo, hi), self.range(lo, hi), self.range(lo, hi)]
    }

    /// Uniform index below `n`.
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
}

/// Distance of the point `p` to the segment `q0..q1`.
fn point_segment_distance(p: [f64; 3], q0: [f64; 3], q1: [f64; 3]) -> f64 {
    let d = sub3(q1, q0);
    let length_sq = dot3(d, d);
    let u = if length_sq > 0.0 {
        (dot3(sub3(p, q0), d) / length_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    len3(sub3(p, add3(q0, scale3(d, u))))
}

/// Distance of two segments by golden-section search over the first one: the
/// distance of a point moving linearly to a fixed segment is a convex function.
fn segment_distance_oracle(a0: [f64; 3], a1: [f64; 3], b0: [f64; 3], b1: [f64; 3]) -> f64 {
    let at = |s: f64| point_segment_distance(add3(a0, scale3(sub3(a1, a0), s)), b0, b1);
    let golden = 0.5 * (5.0_f64.sqrt() - 1.0);
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    let mut x1 = hi - golden * (hi - lo);
    let mut x2 = lo + golden * (hi - lo);
    let (mut f1, mut f2) = (at(x1), at(x2));
    for _ in 0..200 {
        if f1 <= f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - golden * (hi - lo);
            f1 = at(x1);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + golden * (hi - lo);
            f2 = at(x2);
        }
    }
    at(0.0).min(at(1.0)).min(f1).min(f2)
}

/// The four moving end points of an edge pair, as the arguments of
/// [`edge_edge_ccd`] in order `p0a p1a q0a q1a p0b p1b q0b q1b`.
type EdgePair = [[f64; 3]; 8];

fn pair_toi(pair: &EdgePair, radius: f64) -> Option<f64> {
    edge_edge_ccd(
        pair[0], pair[1], pair[2], pair[3], pair[4], pair[5], pair[6], pair[7], radius,
    )
}

/// Distance between the segments of the edge pair at normalised time `t`.
fn pair_distance_at(pair: &EdgePair, t: f64) -> f64 {
    let lerp = |a: [f64; 3], b: [f64; 3]| add3(scale3(a, 1.0 - t), scale3(b, t));
    let (p_a, q_a) = (lerp(pair[0], pair[1]), lerp(pair[2], pair[3]));
    let (p_b, q_b) = (lerp(pair[4], pair[5]), lerp(pair[6], pair[7]));
    segment_pair(p_a, q_a, p_b, q_b).distance
}

/// Apply the similarity `x -> scale * rotation(x) + offset` to every point.
fn transformed(pair: &EdgePair, scale: f64, offset: [f64; 3]) -> EdgePair {
    // Rotation by 0.7 rad about the axis (1, 2, 3) / sqrt(14), by Rodrigues' formula.
    let axis = scale3([1.0, 2.0, 3.0], 1.0 / 14.0_f64.sqrt());
    let (sin, cos) = 0.7_f64.sin_cos();
    let rotate = |p: [f64; 3]| {
        add3(
            add3(scale3(p, cos), scale3(cross3(axis, p), sin)),
            scale3(axis, dot3(axis, p) * (1.0 - cos)),
        )
    };
    pair.map(|p| add3(scale3(rotate(p), scale), offset))
}

// ─────────────────────────────────────────────────────────────────────────────
// Cubic roots
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn unit_interval_roots_finds_every_root_of_a_cubic() {
    // (t - 0.2)(t - 0.5)(t - 0.9)
    let roots = unit_interval_roots([1.0, -1.6, 0.73, -0.09]);
    assert_eq!(roots.len(), 3, "{roots:?}");
    for (root, expected) in roots.iter().zip([0.2, 0.5, 0.9]) {
        assert!(close(*root, expected, 1e-12), "{roots:?}");
    }
    // (t - 1.5)(t + 0.3)(t - 0.4): only 0.4 lies in [0, 1].
    let roots = unit_interval_roots([1.0, -1.6, 0.03, 0.18]);
    assert_eq!(roots.len(), 1, "{roots:?}");
    assert!(close(roots[0], 0.4, 1e-12));
    // t^3 + t + 1 has its only real root at a negative t.
    assert!(unit_interval_roots([1.0, 0.0, 1.0, 1.0]).is_empty());
}

#[test]
fn unit_interval_roots_handles_double_roots_and_the_interval_ends() {
    // (t - 0.5)^2 (t - 2): the cubic touches zero at 0.5 without changing sign.
    let roots = unit_interval_roots([1.0, -3.0, 2.25, -0.5]);
    assert_eq!(roots.len(), 1, "{roots:?}");
    assert!(close(roots[0], 0.5, 1e-12));
    // Roots exactly at 0 and 1: t (t - 1) (t + 1).
    let roots = unit_interval_roots([1.0, 0.0, -1.0, 0.0]);
    assert_eq!(roots, vec![0.0, 1.0]);
}

#[test]
fn unit_interval_roots_degrades_to_the_quadratic_and_linear_cases() {
    // t^2 - 0.7 t + 0.1 = (t - 0.2)(t - 0.5)
    let roots = unit_interval_roots([0.0, 1.0, -0.7, 0.1]);
    assert_eq!(roots.len(), 2, "{roots:?}");
    assert!(close(roots[0], 0.2, 1e-12) && close(roots[1], 0.5, 1e-12));
    // 2 t - 1
    let roots = unit_interval_roots([0.0, 0.0, 2.0, -1.0]);
    assert_eq!(roots.len(), 1);
    assert!(close(roots[0], 0.5, 1e-15));
    // A nonzero constant has no root.
    assert!(unit_interval_roots([0.0, 0.0, 0.0, 3.0]).is_empty());
}

#[test]
fn unit_interval_roots_separates_roots_that_lie_close_together() {
    // (t - a)(t - b)(t - c) with a and b only 2e-4 apart.
    let (a, b, c) = (0.5, 0.500_2, 0.9);
    let roots = unit_interval_roots([1.0, -(a + b + c), a * b + a * c + b * c, -a * b * c]);
    assert_eq!(roots.len(), 3, "{roots:?}");
    for (root, expected) in roots.iter().zip([a, b, c]) {
        assert!(close(*root, expected, 1e-9), "{roots:?}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Segment distance
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn segment_pair_matches_the_golden_section_oracle_on_random_segments() {
    let mut rng = Rng::new(0x5EED_1001);
    for case in 0..2000 {
        let a0 = rng.vec(-1.0, 1.0);
        let mut a1 = rng.vec(-1.0, 1.0);
        let b0 = rng.vec(-1.0, 1.0);
        let mut b1 = rng.vec(-1.0, 1.0);
        match rng.below(5) {
            // A segment that is a single point.
            0 => a1 = a0,
            1 => b1 = b0,
            // Parallel segments, shifted sideways and along their direction.
            2 => b1 = add3(b0, scale3(sub3(a1, a0), rng.range(-2.0, 2.0))),
            _ => {}
        }
        let got = segment_pair(a0, a1, b0, b1);
        let expected = segment_distance_oracle(a0, a1, b0, b1);
        assert!(
            close(got.distance, expected, 1e-9),
            "case {case}: {} expected {expected}",
            got.distance
        );
        assert!((0.0..=1.0).contains(&got.s) && (0.0..=1.0).contains(&got.u));
        assert!(vec_close(
            got.closest_a,
            add3(a0, scale3(sub3(a1, a0), got.s)),
            1e-12
        ));
        assert!(vec_close(
            got.closest_b,
            add3(b0, scale3(sub3(b1, b0), got.u)),
            1e-12
        ));
        assert!(close(
            got.distance,
            len3(sub3(got.closest_a, got.closest_b)),
            1e-15
        ));
    }
}

#[test]
fn segment_pair_has_no_absolute_length_threshold() {
    // Two crossing segments 1e-9 long, one above the other by 1e-9: an absolute
    // threshold on the squared length would treat both as points.
    let tiny = 1e-9;
    let crossing = segment_pair(
        [0.0, 0.0, 0.0],
        [tiny, 0.0, 0.0],
        [0.5 * tiny, -0.5 * tiny, tiny],
        [0.5 * tiny, 0.5 * tiny, tiny],
    );
    assert!(close(crossing.distance, tiny, 1e-24), "{crossing:?}");
    assert!(close(crossing.s, 0.5, 1e-9) && close(crossing.u, 0.5, 1e-9));
}

// ─────────────────────────────────────────────────────────────────────────────
// Pair CCD
// ─────────────────────────────────────────────────────────────────────────────

/// Edge A is static along +x from the origin to (1, 0, 0). Edge B runs along y
/// from -0.5 to 0.5 and sweeps from (501.5, ., -5) to (-498.5, ., 5). The four
/// end points are coplanar only at t = 0.5, when B is 0.5 beyond the end of A,
/// far outside the contact radius 0.01. B then passes the end point (1, 0, 0) of
/// A at a height of 0.005 (t = 0.5005) and stays within the radius of A for only
/// 5e-4 of the step, which no uniform sampling of the step with 128 or fewer
/// samples sees.
#[test]
fn a_brief_graze_over_an_end_point_is_found_between_time_samples() {
    let pair: EdgePair = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [501.5, -0.5, -5.0],
        [-498.5, -0.5, 5.0],
        [501.5, 0.5, -5.0],
        [-498.5, 0.5, 5.0],
    ];
    let radius = 0.01;
    let toi = pair_toi(&pair, radius).expect("the edges pass within the contact radius");
    // x(t) = 501.5 - 1000 t, z(t) = -5 + 10 t. With u = t - 0.5005 the end point
    // of A is reached at u = 0, where B is 0.005 above it, and the distance to
    // it is the radius where 1000100 u^2 + 0.1 u - 7.5e-5 = 0.
    let discriminant = 0.1_f64 * 0.1 + 4.0 * 1_000_100.0 * 7.5e-5;
    let expected = 0.5005 + (-0.1 - discriminant.sqrt()) / (2.0 * 1_000_100.0);
    assert!(close(toi, expected, 1e-9), "toi {toi} expected {expected}");
    assert!(close(pair_distance_at(&pair, toi), radius, 1e-6));
    // Nothing sampled on the coarse grid is inside the shell.
    assert!((0..=128).all(|k| pair_distance_at(&pair, k as f64 / 128.0) > radius));
    // With a contact radius below the closest approach (0.005 above the end
    // point) the same motion is a miss.
    assert_eq!(pair_toi(&pair, 0.002), None);
}

#[test]
fn a_fast_crossing_between_time_samples_is_found_at_the_crossing_time() {
    // Edge A is static along +x. Edge B (along y at x = 0.5) drops through it from
    // z = 5.2 to z = -4.8 in one step: the edges cross at t = 0.52.
    let pair: EdgePair = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, -0.5, 5.2],
        [0.5, -0.5, -4.8],
        [0.5, 0.5, 5.2],
        [0.5, 0.5, -4.8],
    ];
    for radius in [0.0, 1e-6, 0.01] {
        let toi = pair_toi(&pair, radius).expect("the edges cross");
        assert!(close(toi, 0.52, 1e-9), "radius {radius}: toi {toi}");
    }
}

#[test]
fn a_crossing_is_found_for_any_scale_position_and_orientation() {
    let pair: EdgePair = [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.1, 0.0],
        [0.5, -0.5, 1.0],
        [0.6, -0.4, -2.0],
        [0.5, 0.5, 1.5],
        [0.4, 0.6, -1.0],
    ];
    let reference = pair_toi(&pair, 0.0).expect("the edges cross");
    assert!(reference > 0.0 && reference < 1.0);
    for scale in [1e-6, 1e-3, 1.0, 1e3, 1e6] {
        for offset in [
            [0.0; 3],
            scale3([1.0, -2.0, 0.5], 1e3 * scale),
            scale3([-3.0, 1.0, 2.0], 1e4 * scale),
        ] {
            let moved = transformed(&pair, scale, offset);
            let toi = pair_toi(&moved, 0.0).expect("the edges still cross");
            assert!(
                close(toi, reference, 1e-9),
                "scale {scale} offset {offset:?}: toi {toi} expected {reference}"
            );
        }
    }
}

#[test]
fn the_crossing_time_matches_an_independent_root_search_on_random_edges() {
    let mut rng = Rng::new(0x5EED_1002);
    let (mut crossings, mut misses) = (0, 0);
    for case in 0..3000 {
        let pair: EdgePair = std::array::from_fn(|_| rng.vec(-1.0, 1.0));
        // Signed volume of the four end points (zero when they are coplanar),
        // evaluated directly from the moving points.
        let volume_at = |t: f64| {
            let lerp = |a: [f64; 3], b: [f64; 3]| add3(scale3(a, 1.0 - t), scale3(b, t));
            let edge_a = sub3(lerp(pair[2], pair[3]), lerp(pair[0], pair[1]));
            let edge_b = sub3(lerp(pair[6], pair[7]), lerp(pair[4], pair[5]));
            let link = sub3(lerp(pair[4], pair[5]), lerp(pair[0], pair[1]));
            dot3(edge_a, cross3(edge_b, link))
        };
        // Roots by dense sampling followed by bisection.
        let samples = 4000;
        let volumes: Vec<f64> = (0..=samples)
            .map(|k| volume_at(k as f64 / samples as f64))
            .collect();
        let mut expected: Option<f64> = None;
        let mut nearest_miss = f64::INFINITY;
        for k in 0..samples {
            let (mut lo, mut hi) = (k as f64 / samples as f64, (k + 1) as f64 / samples as f64);
            if (volumes[k] < 0.0) == (volumes[k + 1] < 0.0) {
                continue;
            }
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                if (volume_at(mid) < 0.0) == (volume_at(lo) < 0.0) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let distance = pair_distance_at(&pair, lo);
            if distance <= 1e-9 {
                expected = expected.or(Some(lo));
            } else {
                nearest_miss = nearest_miss.min(distance);
            }
        }
        // Skip scenes in which a root passes the edges at a distance that is
        // neither a crossing nor a clear miss.
        if nearest_miss < 1e-6 {
            continue;
        }
        let got = pair_toi(&pair, 0.0);
        match expected {
            Some(t) => {
                crossings += 1;
                let got = got.unwrap_or(f64::NAN);
                assert!(close(got, t, 1e-7), "case {case}: toi {got} expected {t}");
            }
            None => {
                misses += 1;
                assert!(got.is_none(), "case {case}: false contact at {got:?}");
            }
        }
    }
    assert!(
        crossings > 50 && misses > 50,
        "scenes: {crossings} crossings, {misses} misses"
    );
}

#[test]
fn edges_moving_in_one_plane_report_the_first_entry_into_the_contact_shell() {
    // Everything stays in z = 0, so the four end points are coplanar at every
    // time and the coplanarity cubic vanishes identically. Edge A is static along
    // x; edge B (along y) slides in +x from -3 to 3 and gets within 0.05 of A when
    // its x reaches -0.05.
    let pair: EdgePair = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-3.0, -0.5, 0.0],
        [3.0, -0.5, 0.0],
        [-3.0, 0.5, 0.0],
        [3.0, 0.5, 0.0],
    ];
    let toi = pair_toi(&pair, 0.05).expect("B slides across A");
    assert!(close(toi, (3.0 - 0.05) / 6.0, 1e-9), "toi {toi}");
}

#[test]
fn parallel_edges_approaching_each_other_stop_at_the_contact_distance() {
    // Edge B is parallel to edge A and falls through it from y = 1 to y = -1.
    let pair: EdgePair = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
    ];
    let toi = pair_toi(&pair, 0.05).expect("B reaches A");
    assert!(close(toi, 0.475, 1e-9), "toi {toi}");
    // The same fall beside A, out of its reach in x, never touches it.
    let mut aside = pair;
    for point in &mut aside[4..] {
        point[0] += 3.0;
    }
    assert_eq!(pair_toi(&aside, 0.05), None);
}

#[test]
fn edges_that_start_within_the_contact_radius_touch_at_time_zero() {
    // Two edges that share an end point.
    let pair: EdgePair = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    assert_eq!(pair_toi(&pair, 0.01), Some(0.0));
    assert_eq!(pair_toi(&pair, 0.0), Some(0.0));
}

#[test]
fn invalid_input_reports_no_contact() {
    let pair: EdgePair = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, -0.5, 1.0],
        [0.5, -0.5, -1.0],
        [0.5, 0.5, 1.0],
        [0.5, 0.5, -1.0],
    ];
    assert_eq!(
        pair_toi(&pair, 0.01).map(|t| close(t, 0.5, 1e-9)),
        Some(true)
    );
    assert_eq!(pair_toi(&pair, -0.01), None);
    assert_eq!(pair_toi(&pair, f64::NAN), None);
    for coordinate in [f64::NAN, f64::INFINITY] {
        for index in 0..8 {
            let mut broken = pair;
            broken[index][1] = coordinate;
            assert_eq!(
                pair_toi(&broken, 0.01),
                None,
                "point {index} = {coordinate}"
            );
        }
    }
}

#[test]
fn the_pair_ccd_agrees_with_dense_time_sampling_on_random_edges() {
    let mut rng = Rng::new(0x5EED_1003);
    let (mut hits, mut misses) = (0, 0);
    for case in 0..1500 {
        let start: [[f64; 3]; 4] = std::array::from_fn(|_| rng.vec(-1.0, 1.0));
        let pair: EdgePair = [
            start[0],
            add3(start[0], rng.vec(-1.5, 1.5)),
            start[1],
            add3(start[1], rng.vec(-1.5, 1.5)),
            start[2],
            add3(start[2], rng.vec(-1.5, 1.5)),
            start[3],
            add3(start[3], rng.vec(-1.5, 1.5)),
        ];
        let radius = rng.range(0.0, 0.2);
        let samples = 4000;
        let sampled_minimum = (0..=samples)
            .map(|k| pair_distance_at(&pair, k as f64 / samples as f64))
            .fold(f64::INFINITY, f64::min);
        match pair_toi(&pair, radius) {
            Some(t) => {
                hits += 1;
                assert!((0.0..=1.0).contains(&t), "case {case}: toi {t}");
                assert!(
                    pair_distance_at(&pair, t) <= radius + 1e-8,
                    "case {case}: distance {} at toi {t} exceeds {radius}",
                    pair_distance_at(&pair, t)
                );
            }
            None => {
                misses += 1;
                assert!(
                    sampled_minimum > radius - 1e-9,
                    "case {case}: missed a dip to {sampled_minimum} (radius {radius})"
                );
            }
        }
    }
    assert!(
        hits > 100 && misses > 100,
        "scenes: {hits} contacts, {misses} misses"
    );
}

/// Edge A is static along x from the origin to the end point `E = (1, 0, 0)`.
/// Edge B runs along y (half length 0.5) and its centre moves in the xz plane at
/// a speed of 50 to 200 units per step along a straight line that passes `E` at
/// the distance `passing` without crossing A. The closest approach of the edges is
/// therefore exactly `passing`, reached at a time `t*` in `[0.2, 0.8]`, and the
/// edges spend only a few thousandths of the step within a contact radius of that
/// size.
fn grazing_scene(rng: &mut Rng, passing: f64) -> EdgePair {
    let angle = rng.range(0.15, std::f64::consts::PI - 0.15);
    let (sin, cos) = angle.sin_cos();
    // Direction of motion (downwards, so the line crosses the x axis beyond E) and
    // the unit vector from E to the closest point of the line.
    let along = [cos, 0.0, -sin];
    let away = [sin, 0.0, cos];
    let speed = rng.range(50.0, 200.0);
    let closest_time = rng.range(0.2, 0.8);
    let foot = add3([1.0, 0.0, 0.0], scale3(away, passing));
    let start = sub3(foot, scale3(along, speed * closest_time));
    let end = add3(foot, scale3(along, speed * (1.0 - closest_time)));
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [start[0], -0.5, start[2]],
        [end[0], -0.5, end[2]],
        [start[0], 0.5, start[2]],
        [end[0], 0.5, end[2]],
    ]
}

#[test]
fn the_pair_ccd_separates_brief_grazes_from_misses_at_high_speed() {
    let mut rng = Rng::new(0x5EED_1004);
    let radius = 0.02;
    let (mut hits, mut misses) = (0, 0);
    for case in 0..300 {
        let ratio = rng.range(0.4, 1.6);
        let mut pair = grazing_scene(&mut rng, ratio * radius);
        if case % 2 == 1 {
            pair = transformed(&pair, 1.0, [3.0, -2.0, 1.0]);
        }
        let toi = pair_toi(&pair, radius);
        // The closest approach is `ratio * radius`, so the pair is in contact
        // exactly when the ratio is below one; stay clear of the borderline.
        if ratio < 0.8 {
            assert!(toi.is_some(), "case {case}: ratio {ratio} missed");
        }
        if ratio > 1.2 {
            assert!(toi.is_none(), "case {case}: ratio {ratio} reported {toi:?}");
        }
        match toi {
            Some(t) => {
                hits += 1;
                assert!(
                    pair_distance_at(&pair, t) <= radius + 1e-6,
                    "case {case}: distance {} at toi {t}",
                    pair_distance_at(&pair, t)
                );
            }
            None => misses += 1,
        }
    }
    assert!(
        hits > 80 && misses > 80,
        "scenes: {hits} contacts, {misses} misses"
    );
}

/// Edge A is static along x from 0 to 1. Edge B is parallel to it, `clearance`
/// above the contact radius, and slides along x by 1000 units, passing over A.
fn sliding_pair(radius: f64, clearance: f64) -> EdgePair {
    let height = radius + clearance;
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-500.0, height, 0.0],
        [500.0, height, 0.0],
        [-499.0, height, 0.0],
        [501.0, height, 0.0],
    ]
}

#[test]
fn a_parallel_pair_sliding_past_at_a_safe_clearance_is_certified_apart() {
    // The distance is a constant 0.01 above the radius while B overlaps A, and
    // the subdivision proves that for every instant of the overlap.
    assert_eq!(pair_toi(&sliding_pair(0.01, 0.01), 0.01), None);
    // Sliding at the radius itself touches, from the moment B overlaps A (the
    // right end of B reaches the left end of A at t = 0.499).
    let toi = pair_toi(&sliding_pair(0.01, 0.0), 0.01).expect("B slides at the contact distance");
    assert!(close(toi, 0.499, 1e-6), "toi {toi}");
}

#[test]
fn a_pair_hovering_a_hair_above_the_contact_radius_is_reported_as_touching() {
    // The clearance is a millionth of the radius, far below what a bounded
    // search can certify at a relative speed of 1000 units per step: the
    // pair is reported in contact, at the start of the overlap (t = 0.499 to
    // 0.501), instead of being risked as a miss.
    let pair = sliding_pair(0.01, 1e-8);
    let toi = pair_toi(&pair, 0.01).expect("no gap can be certified");
    assert!(toi > 0.499 - 1e-4 && toi < 0.501, "toi {toi}");
    assert!(pair_distance_at(&pair, toi) <= 0.01 + 1e-6);
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch CCD
// ─────────────────────────────────────────────────────────────────────────────

/// Two meshes of one edge each with the same index layout (vertices 0 and 1):
/// the positions of both vertices of each mesh at the start and the end of the
/// step.
struct CrossingMeshes {
    a_start: [[f64; 3]; 2],
    a_end: [[f64; 3]; 2],
    b_start: [[f64; 3]; 2],
    b_end: [[f64; 3]; 2],
}

/// Edge A is static along x; edge B is along y and falls through it from z = 1 to
/// z = -1, crossing at t = 0.5.
fn crossing_meshes() -> CrossingMeshes {
    let mesh_a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    CrossingMeshes {
        a_start: mesh_a,
        a_end: mesh_a,
        b_start: [[0.5, -0.5, 1.0], [0.5, 0.5, 1.0]],
        b_end: [[0.5, -0.5, -1.0], [0.5, 0.5, -1.0]],
    }
}

#[test]
fn edges_of_two_meshes_with_identical_index_layouts_are_not_skipped() {
    let CrossingMeshes {
        a_start,
        a_end,
        b_start,
        b_end,
    } = crossing_meshes();
    let edges = [(0, 1)];
    // Both meshes use vertex indices 0 and 1, which the old shared-vertex test
    // mistook for neighbouring edges of one mesh.
    let results = batch_edge_edge_ccd(
        &edges, &a_start, &a_end, &edges, &b_start, &b_end, 0.01, false,
    );
    assert_eq!(results.len(), 1, "{results:?}");
    let hit = &results[0];
    assert_eq!((hit.edge_a, hit.edge_b), (0, 0));
    assert!(close(hit.toi, 0.5, 1e-9), "toi {}", hit.toi);
    // Edge A starts below edge B, so the normal from B towards A points down.
    assert!(
        vec_close(hit.normal, [0.0, 0.0, -1.0], 1e-9),
        "{:?}",
        hit.normal
    );
    // The single-pair function agrees.
    let toi = edge_edge_ccd(
        a_start[0], a_end[0], a_start[1], a_end[1], b_start[0], b_end[0], b_start[1], b_end[1],
        0.01,
    );
    assert!(toi.is_some_and(|t| close(t, hit.toi, 1e-12)));
    // Flagging the sets as one mesh does skip the pair: the indices are shared.
    let one_mesh = batch_edge_edge_ccd(
        &edges, &a_start, &a_end, &edges, &b_start, &b_end, 0.01, true,
    );
    assert!(one_mesh.is_empty());
}

#[test]
fn the_normal_is_oriented_by_the_side_edge_a_started_on() {
    // Edge A is along x and edge B along y; their heights cross at t = 0.5 in all
    // four combinations of rising and falling. The normal points from B towards A
    // on the side A came from: up if A started above B, down if below.
    let edges = [(0, 1)];
    for (a_from, a_to, b_from, b_to, expected_z) in [
        (2.0, -2.0, 1.0, -1.0, 1.0),
        (-2.0, 2.0, 1.0, -1.0, -1.0),
        (2.0, -2.0, -1.0, 1.0, 1.0),
        (-2.0, 2.0, -1.0, 1.0, -1.0),
    ] {
        let a_start = [[0.0, 0.0, a_from], [1.0, 0.0, a_from]];
        let a_end = [[0.0, 0.0, a_to], [1.0, 0.0, a_to]];
        let b_start = [[0.5, -0.5, b_from], [0.5, 0.5, b_from]];
        let b_end = [[0.5, -0.5, b_to], [0.5, 0.5, b_to]];
        let results = batch_edge_edge_ccd(
            &edges, &a_start, &a_end, &edges, &b_start, &b_end, 0.0, false,
        );
        assert_eq!(results.len(), 1, "{results:?}");
        assert!(close(results[0].toi, 0.5, 1e-9), "toi {}", results[0].toi);
        assert!(
            vec_close(results[0].normal, [0.0, 0.0, expected_z], 1e-9),
            "A from {a_from} to {a_to}, B from {b_from} to {b_to}: {:?}",
            results[0].normal
        );
    }
}

#[test]
fn the_normal_of_a_shell_graze_is_the_separation_of_the_closest_points() {
    // Edge A: x from 0 to 10 at y = z = 0. Edge B runs along z at x = 5 and moves
    // from y = 1 to y = 0.2. It never crosses A (y stays above 0) but enters the
    // contact shell of radius 0.5 at y = 0.5, i.e. t = 0.625, where the closest
    // points are (5, 0, 0) on A and (5, 0.5, 0) on B. The old normal was taken
    // between the first vertices of the edges, (0, 0, 0) and (5, 0.5, -1).
    let a = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
    let b_start = [[5.0, 1.0, -1.0], [5.0, 1.0, 1.0]];
    let b_end = [[5.0, 0.2, -1.0], [5.0, 0.2, 1.0]];
    let edges = [(0, 1)];
    let results = batch_edge_edge_ccd(&edges, &a, &a, &edges, &b_start, &b_end, 0.5, false);
    assert_eq!(results.len(), 1, "{results:?}");
    assert!(close(results[0].toi, 0.625, 1e-9), "toi {}", results[0].toi);
    assert!(
        vec_close(results[0].normal, [0.0, -1.0, 0.0], 1e-9),
        "{:?}",
        results[0].normal
    );
}

#[test]
fn neighbouring_edges_of_one_mesh_touch_at_time_zero_unless_skipped() {
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
    let edges = [(0, 1), (1, 2)];
    // Reported as in contact when the sets are not declared to be one mesh.
    let contact = batch_edge_edge_ccd(
        &edges, &positions, &positions, &edges, &positions, &positions, 0.01, false,
    );
    assert_eq!(contact.len(), 4, "{contact:?}");
    assert!(contact.iter().all(|r| r.toi == 0.0));
    // Skipped for the shared vertex when they are.
    let skipped = batch_edge_edge_ccd(
        &edges, &positions, &positions, &edges, &positions, &positions, 0.01, true,
    );
    assert!(skipped.is_empty(), "{skipped:?}");
}

#[test]
fn batch_results_index_the_input_lists_and_skip_invalid_edges() {
    let CrossingMeshes {
        a_start,
        a_end,
        b_start,
        b_end,
    } = crossing_meshes();
    // The invalid edges sit before the valid one, so the reported indices differ
    // from the position in the list of valid edges.
    let edges_a = [(0, 5), (0, 1)];
    let edges_b = [(9, 1), (7, 7), (0, 1)];
    let results = batch_edge_edge_ccd(
        &edges_a, &a_start, &a_end, &edges_b, &b_start, &b_end, 0.01, false,
    );
    assert_eq!(results.len(), 1, "{results:?}");
    assert_eq!((results[0].edge_a, results[0].edge_b), (1, 2));
    // Position arrays of different length for the start and end of the step.
    let short = batch_edge_edge_ccd(
        &[(0, 1)],
        &a_start,
        &a_end[..1],
        &[(0, 1)],
        &b_start,
        &b_end,
        0.01,
        false,
    );
    assert!(short.is_empty());
    // A negative or NaN radius can never be reached.
    for radius in [-1.0, f64::NAN] {
        let none = batch_edge_edge_ccd(
            &[(0, 1)],
            &a_start,
            &a_end,
            &[(0, 1)],
            &b_start,
            &b_end,
            radius,
            false,
        );
        assert!(none.is_empty());
    }
}

#[test]
fn the_batch_reports_exactly_the_pairs_the_pair_function_finds() {
    let mut rng = Rng::new(0x5EED_1005);
    let radius = 0.04;
    let (mut contacts, mut checked) = (0, 0);
    for _ in 0..40 {
        // Two small meshes of six vertices that move randomly; every vertex pair
        // is an edge, so edge lists share vertices inside each mesh.
        let verts0_a: Vec<[f64; 3]> = (0..6).map(|_| rng.vec(-1.0, 1.0)).collect();
        let verts1_a: Vec<[f64; 3]> = verts0_a
            .iter()
            .map(|&p| add3(p, rng.vec(-0.6, 0.6)))
            .collect();
        let verts0_b: Vec<[f64; 3]> = (0..6).map(|_| rng.vec(-1.0, 1.0)).collect();
        let verts1_b: Vec<[f64; 3]> = verts0_b
            .iter()
            .map(|&p| add3(p, rng.vec(-0.6, 0.6)))
            .collect();
        let edges: Vec<(usize, usize)> = (0..6)
            .flat_map(|i| (i + 1..6).map(move |j| (i, j)))
            .collect();
        let results = batch_edge_edge_ccd(
            &edges, &verts0_a, &verts1_a, &edges, &verts0_b, &verts1_b, radius, false,
        );
        for (ia, &(a0, a1)) in edges.iter().enumerate() {
            for (ib, &(b0, b1)) in edges.iter().enumerate() {
                let expected = edge_edge_ccd(
                    verts0_a[a0],
                    verts1_a[a0],
                    verts0_a[a1],
                    verts1_a[a1],
                    verts0_b[b0],
                    verts1_b[b0],
                    verts0_b[b1],
                    verts1_b[b1],
                    radius,
                );
                let found = results.iter().find(|r| r.edge_a == ia && r.edge_b == ib);
                checked += 1;
                match (expected, found) {
                    (Some(toi), Some(result)) => {
                        contacts += 1;
                        assert!(close(toi, result.toi, 1e-12));
                        assert!(close(len3(result.normal), 1.0, 1e-9), "{:?}", result.normal);
                    }
                    (None, None) => {}
                    (expected, found) => panic!(
                        "pair ({ia}, {ib}): pair function {expected:?}, batch {:?}",
                        found.map(|r| r.toi)
                    ),
                }
            }
        }
    }
    assert!(contacts > 50, "{contacts} contacts in {checked} pairs");
}

// ─────────────────────────────────────────────────────────────────────────────
// SoftBodyBvh
// ─────────────────────────────────────────────────────────────────────────────

/// `n` small triangles scattered in the cube `[-extent, extent]^3`; triangle `i`
/// uses the vertices `3 i`, `3 i + 1` and `3 i + 2`.
fn triangle_soup(
    rng: &mut Rng,
    n: usize,
    extent: f64,
    size: f64,
) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut positions = Vec::with_capacity(3 * n);
    let mut triangles = Vec::with_capacity(n);
    for i in 0..n {
        let center = rng.vec(-extent, extent);
        for _ in 0..3 {
            positions.push(add3(center, scale3(rng.vec(-1.0, 1.0), size)));
        }
        triangles.push([3 * i, 3 * i + 1, 3 * i + 2]);
    }
    (positions, triangles)
}

/// Number of nodes on the longest root-to-leaf path.
fn depth(bvh: &SoftBodyBvh) -> usize {
    let mut deepest = 0;
    let mut stack = Vec::new();
    if !bvh.nodes.is_empty() {
        stack.push((0usize, 1usize));
    }
    while let Some((index, level)) = stack.pop() {
        deepest = deepest.max(level);
        let node = &bvh.nodes[index];
        if !node.is_leaf() {
            stack.push((node.left, level + 1));
            stack.push((node.right, level + 1));
        }
    }
    deepest
}

/// Triangles whose margin-grown box overlaps the box of the sphere, by brute force.
fn brute_force_query(
    positions: &[[f64; 3]],
    triangles: &[[usize; 3]],
    margin: f64,
    center: [f64; 3],
    radius: f64,
) -> Vec<usize> {
    let sphere = BvhAabb::new(sub3(center, [radius; 3]), add3(center, [radius; 3]));
    triangles
        .iter()
        .enumerate()
        .filter(|(_, tri)| {
            let [a, b, c] = tri.map(|v| positions[v]);
            BvhAabb::from_triangle(a, b, c, margin).overlaps(&sphere)
        })
        .map(|(i, _)| i)
        .collect()
}

fn sorted(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v
}

#[test]
fn a_hundred_thousand_triangles_in_random_order_give_a_balanced_tree() {
    let mut rng = Rng::new(0x5EED_2001);
    let n = 100_000;
    let (positions, triangles) = triangle_soup(&mut rng, n, 50.0, 0.1);
    let bvh = SoftBodyBvh::build(&triangles, &positions, 0.01);
    assert_eq!(bvh.nodes.len(), 2 * n - 1);
    let levels = depth(&bvh);
    let log2 = (n as f64).log2();
    assert!(
        levels as f64 <= 2.0 * log2,
        "depth {levels} for {n} triangles (2 log2 n = {})",
        2.0 * log2
    );
    // A median split is perfectly balanced.
    assert_eq!(levels, log2.ceil() as usize + 1);
}

#[test]
fn every_input_order_gives_a_balanced_tree_with_correct_queries() {
    let mut rng = Rng::new(0x5EED_2002);
    let n = 3000;
    let (positions, mut triangles) = triangle_soup(&mut rng, n, 20.0, 0.3);
    let centroid_x = |tri: &[usize; 3]| {
        (positions[tri[0]][0] + positions[tri[1]][0] + positions[tri[2]][0]) / 3.0
    };
    let mut orders: Vec<(&str, Vec<[usize; 3]>)> = vec![("random", triangles.clone())];
    triangles.sort_by(|p, q| centroid_x(p).total_cmp(&centroid_x(q)));
    orders.push(("ascending", triangles.clone()));
    triangles.reverse();
    orders.push(("descending", triangles.clone()));
    let margin = 0.05;
    for (label, order) in &orders {
        let bvh = SoftBodyBvh::build(order, &positions, margin);
        let levels = depth(&bvh);
        assert!(
            levels as f64 <= 2.0 * (n as f64).log2(),
            "{label}: depth {levels}"
        );
        for query in 0..200 {
            let center = rng.vec(-20.0, 20.0);
            let radius = rng.range(0.0, 6.0);
            let got = sorted(bvh.query_sphere(center, radius));
            let expected = brute_force_query(&positions, order, margin, center, radius);
            assert_eq!(got, expected, "{label}, query {query}");
        }
    }
}

#[test]
fn the_flat_array_is_a_valid_depth_first_tree_rooted_at_node_zero() {
    let mut rng = Rng::new(0x5EED_2003);
    let n = 1000;
    let margin = 0.02;
    let (positions, triangles) = triangle_soup(&mut rng, n, 10.0, 0.4);
    let bvh = SoftBodyBvh::build(&triangles, &positions, margin);
    assert_eq!(bvh.nodes.len(), 2 * n - 1);
    let mut visits = vec![0usize; bvh.nodes.len()];
    let mut leaf_triangles = Vec::new();
    let mut stack = vec![0usize];
    while let Some(index) = stack.pop() {
        visits[index] += 1;
        let node = &bvh.nodes[index];
        if node.is_leaf() {
            assert_eq!(node.right, usize::MAX);
            leaf_triangles.push(node.tri_index);
            let [a, b, c] = triangles[node.tri_index].map(|v| positions[v]);
            let exact = BvhAabb::from_triangle(a, b, c, margin);
            assert_eq!((node.aabb.min, node.aabb.max), (exact.min, exact.max));
            continue;
        }
        // Children follow their parent, and the left one directly.
        assert_eq!(node.left, index + 1);
        assert!(node.right > node.left && node.right < bvh.nodes.len());
        let union = bvh.nodes[node.left].aabb.merge(&bvh.nodes[node.right].aabb);
        assert_eq!((node.aabb.min, node.aabb.max), (union.min, union.max));
        stack.push(node.right);
        stack.push(node.left);
    }
    // Every node is reached exactly once and every triangle sits in one leaf.
    assert!(visits.iter().all(|&v| v == 1));
    assert_eq!(sorted(leaf_triangles), (0..n).collect::<Vec<_>>());
}

#[test]
fn identical_triangles_do_not_unbalance_the_tree() {
    // All centroids coincide, so a split by the centroid value alone would put
    // everything on one side; the median split still halves the set.
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let n = 5000;
    let triangles = vec![[0, 1, 2]; n];
    let bvh = SoftBodyBvh::build(&triangles, &positions, 0.0);
    assert_eq!(bvh.nodes.len(), 2 * n - 1);
    assert_eq!(depth(&bvh), (n as f64).log2().ceil() as usize + 1);
    assert_eq!(
        sorted(bvh.query_sphere([0.25, 0.25, 0.0], 0.1)),
        (0..n).collect::<Vec<_>>()
    );
    assert!(bvh.query_sphere([5.0, 5.0, 5.0], 0.1).is_empty());
}

#[test]
fn triangles_with_out_of_range_vertices_are_left_out_and_indices_are_kept() {
    let positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [5.0, 5.0, 5.0],
        [6.0, 5.0, 5.0],
        [5.0, 6.0, 5.0],
    ];
    let triangles = [[0, 1, 2], [0, 1, 99], [3, 4, 5], [6, 0, 1]];
    let bvh = SoftBodyBvh::build(&triangles, &positions, 0.01);
    assert_eq!(bvh.nodes.len(), 3);
    assert_eq!(sorted(bvh.query_sphere([0.0; 3], 100.0)), vec![0, 2]);
    assert_eq!(bvh.query_sphere([5.2, 5.2, 5.0], 0.1), vec![2]);
}

#[test]
fn an_empty_mesh_gives_an_empty_tree_that_matches_nothing() {
    let bvh = SoftBodyBvh::build(&[], &[], 0.1);
    assert!(bvh.nodes.is_empty());
    assert!(bvh.query_sphere([0.0; 3], 1e9).is_empty());
    let all_invalid = SoftBodyBvh::build(&[[0, 1, 2]], &[[0.0; 3]], 0.1);
    assert!(all_invalid.nodes.is_empty());
    assert!(all_invalid.query_sphere([0.0; 3], 1e9).is_empty());
}

#[test]
fn a_single_triangle_is_a_single_leaf_and_a_bad_query_matches_nothing() {
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let bvh = SoftBodyBvh::build(&[[0, 1, 2]], &positions, 0.0);
    assert_eq!(bvh.nodes.len(), 1);
    assert!(bvh.nodes[0].is_leaf());
    assert_eq!(bvh.query_sphere([0.2, 0.2, 0.0], 0.0), vec![0]);
    assert!(bvh.query_sphere([0.2, 0.2, 0.0], -1.0).is_empty());
    assert!(bvh.query_sphere([f64::NAN, 0.0, 0.0], 1.0).is_empty());
}
