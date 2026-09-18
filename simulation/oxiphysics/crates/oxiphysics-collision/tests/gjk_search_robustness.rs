// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the GJK overlap search (`Gjk::intersect`,
//! `Gjk::query`, `Gjk::intersect_scaled`, `WarmStartGjk`, `GjkDebugTrace`).
//!
//! The boolean search used to oscillate between simplices of nearly touching
//! shapes until its iteration budget ran out and then report an overlap, so
//! spheres a relative `1e-5` apart were "colliding". Every verdict here is
//! cross-checked against an exact reference (analytic sphere distance,
//! segment-segment distance for capsules, 15-axis SAT for boxes), on both
//! sides of contact, at several scales and far from the origin. No assertion is
//! made inside the rounding band of contact.

use oxiphysics_collision::Gjk;
use oxiphysics_collision::narrowphase::{GjkDebugTrace, GjkResult, WarmStartGjk};
use oxiphysics_core::Transform;
use oxiphysics_core::math::{UnitQuaternion, Vec3};
use oxiphysics_geometry::{BoxShape, Capsule, Sphere};

/// Small deterministic generator.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    fn unit(&mut self) -> Vec3 {
        loop {
            let v = Vec3::new(
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            );
            let n = v.norm();
            if n > 0.1 && n < 1.0 {
                return v / n;
            }
        }
    }

    fn rotation(&mut self) -> UnitQuaternion<f64> {
        let axis = self.unit();
        let angle = self.range(0.0, 3.0);
        Transform::from_axis_angle(Vec3::zeros(), axis, angle).rotation
    }
}

fn posed(position: Vec3, rotation: UnitQuaternion<f64>) -> Transform {
    let mut transform = Transform::from_position(position);
    transform.rotation = rotation;
    transform
}

/// Unit quaternion from its `(x, y, z, w)` coordinates.
fn quat(x: f64, y: f64, z: f64, w: f64) -> UnitQuaternion<f64> {
    let mut q = UnitQuaternion::<f64>::identity().into_inner();
    q.coords.x = x;
    q.coords.y = y;
    q.coords.z = z;
    q.coords.w = w;
    UnitQuaternion::new_normalize(q)
}

/// Distance between segments `p1 q1` and `p2 q2` (Ericson 5.1.9).
fn segment_distance(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> f64 {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.dot(&d1);
    let e = d2.dot(&d2);
    let f = d2.dot(&r);
    let c = d1.dot(&r);
    let b = d1.dot(&d2);
    let denom = a * e - b * b;
    let mut s = if denom > 1e-300 {
        ((b * f - c * e) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut t = (b * s + f) / e;
    if t < 0.0 {
        t = 0.0;
        s = (-c / a).clamp(0.0, 1.0);
    } else if t > 1.0 {
        t = 1.0;
        s = ((b - c) / a).clamp(0.0, 1.0);
    }
    ((p1 + d1 * s) - (p2 + d2 * t)).norm()
}

/// Largest separation of two oriented boxes over the 15 SAT axes (positive:
/// separated by at least that much along a face axis; for an edge axis it is
/// the exact gap of that axis).
fn sat_gap(
    ca: Vec3,
    ra: UnitQuaternion<f64>,
    ha: Vec3,
    cb: Vec3,
    rb: UnitQuaternion<f64>,
    hb: Vec3,
) -> f64 {
    let axes_a = [ra * Vec3::x(), ra * Vec3::y(), ra * Vec3::z()];
    let axes_b = [rb * Vec3::x(), rb * Vec3::y(), rb * Vec3::z()];
    let mut axes: Vec<Vec3> = axes_a.iter().chain(axes_b.iter()).copied().collect();
    for a in &axes_a {
        for b in &axes_b {
            let c = a.cross(b);
            if c.norm() > 1e-9 {
                axes.push(c / c.norm());
            }
        }
    }
    let offset = cb - ca;
    let radius = |axes: &[Vec3; 3], h: &Vec3, n: &Vec3| {
        h.x * axes[0].dot(n).abs() + h.y * axes[1].dot(n).abs() + h.z * axes[2].dot(n).abs()
    };
    axes.iter()
        .map(|n| offset.dot(n).abs() - radius(&axes_a, &ha, n) - radius(&axes_b, &hb, n))
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Relative gaps (negative: penetration) used by the sweeps, all well outside
/// the rounding band of contact.
const GAPS: [f64; 8] = [1e-1, 1e-4, 1e-7, 1e-9, -1e-9, -1e-7, -1e-4, -1e-1];

#[test]
fn nearly_touching_spheres_are_not_reported_as_overlapping() {
    // The configuration the old boolean search got wrong: unit spheres a
    // hundred-thousandth apart.
    let sphere = Sphere::new(1.0);
    let a = Transform::from_position(Vec3::zeros());
    for direction in [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.6, 0.8, 0.0),
        Vec3::new(0.48, 0.6, 0.64),
    ] {
        for gap in [1e-5, 1e-7, 1e-9] {
            let b = Transform::from_position(direction * (2.0 + gap));
            assert!(
                !Gjk::intersect(&sphere, &a, &sphere, &b),
                "spheres {gap} apart along {direction:?} reported as overlapping"
            );
            match Gjk::query(&sphere, &a, &sphere, &b) {
                GjkResult::Separated { distance, .. } => {
                    assert!(
                        (distance - gap).abs() <= 1e-9 * 2.0 + 1e-3 * gap,
                        "{distance}"
                    );
                }
                GjkResult::Intersecting(_) => panic!("query reported overlap at gap {gap}"),
            }
            let c = Transform::from_position(direction * (2.0 - gap));
            assert!(
                Gjk::intersect(&sphere, &a, &sphere, &c),
                "spheres overlapping by {gap} reported as separated"
            );
        }
    }
}

#[test]
fn sphere_pairs_match_the_analytic_verdict_at_every_scale() {
    let mut rng = Lcg(0xA11CE);
    for scale in [1e-3, 1.0, 1e3] {
        for gap_rel in GAPS {
            for _ in 0..40 {
                let centre = Vec3::new(rng.range(-10.0, 10.0), rng.range(-10.0, 10.0), 0.0) * scale;
                let ra = rng.range(0.3, 1.0) * scale;
                let rb = rng.range(0.3, 1.0) * scale;
                let other = centre + rng.unit() * (ra + rb + gap_rel * scale);
                let gap = (other - centre).norm() - ra - rb;
                let hit = Gjk::intersect(
                    &Sphere::new(ra),
                    &Transform::from_position(centre),
                    &Sphere::new(rb),
                    &Transform::from_position(other),
                );
                assert_eq!(hit, gap < 0.0, "scale {scale} gap {gap:e}");
            }
        }
    }
}

#[test]
fn thin_capsule_pairs_match_the_segment_distance_verdict() {
    let mut rng = Lcg(0xCA95);
    let mut checked = 0;
    for scale in [1e-3, 1.0, 1e3] {
        for gap_rel in GAPS {
            for _ in 0..30 {
                let ra = rng.range(0.001, 0.02) * scale;
                let rb = rng.range(0.001, 0.02) * scale;
                let ha = rng.range(0.5, 3.0) * scale;
                let hb = rng.range(0.5, 3.0) * scale;
                let centre_a = Vec3::new(rng.range(-5.0, 5.0), rng.range(-5.0, 5.0), 0.0) * scale;
                let rot_a = rng.rotation();
                let rot_b = rng.rotation();
                let axis_a = rot_a * Vec3::y();
                let axis_b = rot_b * Vec3::y();
                let (pa, qa) = (centre_a - axis_a * ha, centre_a + axis_a * ha);
                let start = centre_a + rng.unit() * (rng.range(0.0, 1.0) * ha);
                let push = rng.unit();
                // Slide B along `push` until the segment distance hits the target.
                let target = ra + rb + gap_rel * scale;
                let (mut lo, mut hi) = (0.0, 20.0 * scale);
                for _ in 0..200 {
                    let mid = 0.5 * (lo + hi);
                    let c = start + push * mid;
                    if segment_distance(pa, qa, c - axis_b * hb, c + axis_b * hb) < target {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let centre_b = start + push * hi;
                let gap = segment_distance(pa, qa, centre_b - axis_b * hb, centre_b + axis_b * hb)
                    - ra
                    - rb;
                // Skip placements where the bisection could not reach the target.
                if (gap - gap_rel * scale).abs() > 0.5 * (gap_rel * scale).abs() {
                    continue;
                }
                checked += 1;
                let hit = Gjk::intersect(
                    &Capsule::new(ra, ha),
                    &posed(centre_a, rot_a),
                    &Capsule::new(rb, hb),
                    &posed(centre_b, rot_b),
                );
                assert_eq!(hit, gap < 0.0, "scale {scale} gap {gap:e}");
            }
        }
    }
    assert!(checked >= 120, "only {checked} capsule pairs checked");
}

#[test]
fn rotated_box_pairs_match_the_sat_verdict() {
    let mut rng = Lcg(0xB0C5);
    let mut checked = 0;
    for scale in [1e-3, 1.0, 1e3] {
        for gap_rel in GAPS {
            for _ in 0..30 {
                let ha = Vec3::new(
                    rng.range(0.3, 1.0),
                    rng.range(0.3, 1.0),
                    rng.range(0.3, 1.0),
                ) * scale;
                let hb = Vec3::new(
                    rng.range(0.3, 1.0),
                    rng.range(0.3, 1.0),
                    rng.range(0.3, 1.0),
                ) * scale;
                let centre_a = Vec3::new(rng.range(-5.0, 5.0), 0.0, rng.range(-5.0, 5.0)) * scale;
                let rot_a = rng.rotation();
                let rot_b = if rng.next_f64() < 0.3 {
                    rot_a
                } else {
                    rng.rotation()
                };
                let push = rng.unit();
                let gap_target = gap_rel * scale;
                let (mut lo, mut hi) = (0.0, 10.0 * scale);
                for _ in 0..200 {
                    let mid = 0.5 * (lo + hi);
                    if sat_gap(centre_a, rot_a, ha, centre_a + push * mid, rot_b, hb) < gap_target {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let centre_b = centre_a + push * hi;
                let gap = sat_gap(centre_a, rot_a, ha, centre_b, rot_b, hb);
                if (gap - gap_target).abs() > 0.5 * gap_target.abs() {
                    continue;
                }
                checked += 1;
                let hit = Gjk::intersect(
                    &BoxShape::new(ha),
                    &posed(centre_a, rot_a),
                    &BoxShape::new(hb),
                    &posed(centre_b, rot_b),
                );
                assert_eq!(hit, gap < 0.0, "scale {scale} gap {gap:e}");
            }
        }
    }
    assert!(checked >= 500, "only {checked} box pairs checked");
}

#[test]
fn far_from_the_origin_the_verdict_holds_above_the_coordinate_noise() {
    // At a million units from the origin the coordinates themselves carry an
    // absolute rounding error of about 2e-10, so gaps of 1e-5 and more must
    // still be classified correctly.
    let mut rng = Lcg(0xFA12);
    let far = Vec3::new(1e6, -1e6, 5e5);
    for gap in [1e-3, 1e-5, -1e-5, -1e-3] {
        for _ in 0..40 {
            let centre = far + Vec3::new(rng.range(-5.0, 5.0), rng.range(-5.0, 5.0), 0.0);
            let ra = rng.range(0.3, 1.0);
            let rb = rng.range(0.3, 1.0);
            let other = centre + rng.unit() * (ra + rb + gap);
            let exact = (other - centre).norm() - ra - rb;
            let hit = Gjk::intersect(
                &Sphere::new(ra),
                &Transform::from_position(centre),
                &Sphere::new(rb),
                &Transform::from_position(other),
            );
            assert_eq!(hit, exact < 0.0, "sphere gap {exact:e}");

            let rot = rng.rotation();
            let half = Vec3::new(0.5, 0.7, 0.9);
            let offset = rot * Vec3::new(1.0 + gap, rng.range(-0.5, 0.5), rng.range(-0.5, 0.5));
            let hit = Gjk::intersect(
                &BoxShape::new(half),
                &posed(centre, rot),
                &BoxShape::new(half),
                &posed(centre + offset, rot),
            );
            assert_eq!(hit, gap < 0.0, "box gap {gap:e}");
        }
    }
}

#[test]
fn shallow_overlaps_of_thin_capsules_with_sliver_simplices_are_found() {
    // Two configurations in which the search ends on a sliver simplex (vertices
    // clustered within 1e-8 next to a far vertex). A tolerance-based closest-
    // point routine treated the enclosing tetrahedron as flat and the stalled
    // search then reported these overlaps (depth 1e-12 and 1e-8) as separated.
    let cases = [
        (
            (1.3229017368362011e-5, 0.0022292518279844755),
            (9.142492001672167e-6, 0.0022384716740328093),
            Vec3::new(
                -0.0005726349611669712,
                -0.00815620516294022,
                -0.004860567687439761,
            ),
            quat(
                0.12778351737362773,
                -0.05625757360297918,
                -0.17290209237314869,
                0.9749929869249914,
            ),
            Vec3::new(
                -0.0005599800977261198,
                -0.0057473463799579474,
                -0.004006401100097009,
            ),
            quat(
                0.14168071061873233,
                0.04978170544856549,
                0.5110978905779872,
                0.8463021353440715,
            ),
        ),
        (
            (5.290032621956261, 538.7105142589755),
            (19.511447319961626, 2552.304199045919),
            Vec3::new(-8695.89687854019, 5672.511005366634, -5228.521227104652),
            quat(
                0.3483305680237109,
                -0.18062619152761378,
                0.4579291393783429,
                0.7977097828300208,
            ),
            Vec3::new(-8551.333264295972, 5271.803834112866, -5482.068932020784),
            quat(
                0.771989486282171,
                -0.43363074908333077,
                0.4193741520065833,
                0.20030458593846112,
            ),
        ),
    ];
    for ((ra, ha), (rb, hb), pa, qa, pb, qb) in cases {
        let ta = posed(pa, qa);
        let tb = posed(pb, qb);
        let axis_a = ta.rotation * Vec3::y();
        let axis_b = tb.rotation * Vec3::y();
        let gap = segment_distance(
            pa - axis_a * ha,
            pa + axis_a * ha,
            pb - axis_b * hb,
            pb + axis_b * hb,
        ) - ra
            - rb;
        assert!(gap < 0.0, "the reference configuration overlaps ({gap:e})");
        assert!(Gjk::intersect(
            &Capsule::new(ra, ha),
            &ta,
            &Capsule::new(rb, hb),
            &tb
        ));
    }
}

#[test]
fn scaled_intersection_uses_the_same_robust_search() {
    let sphere = Sphere::new(0.5);
    let a = Transform::from_position(Vec3::zeros());
    // Scaled by two, the radii become one: centres 2 + 1e-6 apart are separated.
    let apart = Transform::from_position(Vec3::new(2.0 + 1e-6, 0.0, 0.0));
    assert!(!Gjk::intersect_scaled(
        &sphere, &a, 2.0, &sphere, &apart, 2.0
    ));
    let overlapping = Transform::from_position(Vec3::new(2.0 - 1e-6, 0.0, 0.0));
    assert!(Gjk::intersect_scaled(
        &sphere,
        &a,
        2.0,
        &sphere,
        &overlapping,
        2.0
    ));
}

#[test]
fn warm_started_queries_do_not_reuse_stale_support_points() {
    let mut warm = WarmStartGjk::new();
    let sphere = Sphere::new(1.0);
    let a = Transform::from_position(Vec3::zeros());
    // Frame 1: deep overlap.
    let near = Transform::from_position(Vec3::new(0.5, 0.0, 0.0));
    assert!(matches!(
        warm.query(&sphere, &a, &sphere, &near),
        GjkResult::Intersecting(_)
    ));
    // Frame 2: B jumped away; the previous simplex is no longer part of A - B.
    let far = Transform::from_position(Vec3::new(0.0, 2.0 + 1e-6, 0.0));
    match warm.query(&sphere, &a, &sphere, &far) {
        GjkResult::Separated { distance, .. } => {
            assert!((distance - 1e-6).abs() < 1e-9, "distance {distance}");
        }
        GjkResult::Intersecting(_) => panic!("stale warm start reported an overlap"),
    }
    // Frame 3: nearly touching again, warm-started from frame 2's direction.
    let close = Transform::from_position(Vec3::new(0.0, 2.0 + 1e-8, 0.0));
    assert!(matches!(
        warm.query(&sphere, &a, &sphere, &close),
        GjkResult::Separated { .. }
    ));
    assert_eq!(warm.query_count(), 3);
}

#[test]
fn debug_trace_agrees_with_the_intersection_test() {
    let mut rng = Lcg(0xDEB6);
    for _ in 0..60 {
        let box_a = BoxShape::new(Vec3::new(0.5, 0.8, 0.3));
        let capsule = Capsule::new(0.2, 0.6);
        let ta = posed(Vec3::zeros(), rng.rotation());
        let tb = posed(rng.unit() * rng.range(0.5, 2.5), rng.rotation());
        let trace = GjkDebugTrace::collect(&box_a, &ta, &capsule, &tb);
        assert!(trace.n_iterations() > 0);
        assert_eq!(
            trace.terminated_as_intersecting(),
            Gjk::intersect(&box_a, &ta, &capsule, &tb)
        );
        // Every recorded direction is a unit vector.
        for direction in trace.directions() {
            assert!((direction.norm() - 1.0).abs() < 1e-12);
        }
    }
}
