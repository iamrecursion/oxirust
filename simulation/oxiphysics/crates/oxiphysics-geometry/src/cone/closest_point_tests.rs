// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the closest-point queries of [`Cone`] and for the
//! signed distance, SDF and sphere tests derived from them.
//!
//! The cone of radius `r` and half-height `hh` is a body of revolution about
//! the Y axis; in the meridian half-plane `(rho, y)` its boundary is the slant
//! segment from the apex `(0, hh)` to the rim `(r, -hh)` and the base segment
//! from `(0, -hh)` to the rim.

use super::Cone;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn assert_point_close(got: [f64; 3], want: [f64; 3], tol: f64) {
    for axis in 0..3 {
        assert!(
            approx_eq(got[axis], want[axis], tol),
            "got {got:?}, expected {want:?}"
        );
    }
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn rho_of(p: [f64; 3]) -> f64 {
    (p[0] * p[0] + p[2] * p[2]).sqrt()
}

/// Deterministic xorshift generator.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// Distance from the meridian point `(rho, y)` to the boundary of `cone`,
/// found by dense sampling of the slant and the base segments.  It is an upper
/// bound of the exact distance that is within about `1e-3` of it for the cones
/// used here.
fn sampled_boundary_distance(cone: &Cone, rho: f64, y: f64) -> f64 {
    const SAMPLES: usize = 4000;
    let mut best = f64::INFINITY;
    for i in 0..=SAMPLES {
        let t = i as f64 / SAMPLES as f64;
        let slant = [
            t * cone.radius,
            cone.half_height - 2.0 * t * cone.half_height,
        ];
        let base = [t * cone.radius, -cone.half_height];
        for q in [slant, base] {
            best = best.min(((rho - q[0]).powi(2) + (y - q[1]).powi(2)).sqrt());
        }
    }
    best
}

/// `true` if `q` lies on the boundary surface of `cone` (slant or base).
fn on_boundary(cone: &Cone, q: [f64; 3]) -> bool {
    let rho = rho_of(q);
    let on_base = approx_eq(q[1], -cone.half_height, 1e-12) && rho <= cone.radius + 1e-12;
    let on_slant = q[1] >= -cone.half_height - 1e-12
        && q[1] <= cone.half_height + 1e-12
        && approx_eq(rho, cone.radius_at_y(q[1]), 1e-12);
    on_base || on_slant
}

#[test]
fn far_point_projects_onto_the_base_rim_not_horizontally() {
    // Cone(1, 1): apex (0, 1, 0), base disc at y = -1 with radius 1.
    let cone = Cone::new(1.0, 1.0);
    let cp = cone.closest_point([5.0, 0.0, 0.0]);
    assert_point_close(cp, [1.0, -1.0, 0.0], 1e-12);
    // The old horizontal projection (0.5, 0, 0) is 4.5 away; the rim is
    // sqrt(17) = 4.123 away.
    let d = cone.signed_distance([5.0, 0.0, 0.0]);
    assert!(approx_eq(d, 17.0_f64.sqrt(), 1e-12), "distance {d}");
}

#[test]
fn point_beside_the_slant_projects_orthogonally_onto_it() {
    // For p = (2, 0, 0) the foot on the slant line 2 rho + y = 1 is (0.8, -0.6).
    let cone = Cone::new(1.0, 1.0);
    assert_point_close(cone.closest_point([2.0, 0.0, 0.0]), [0.8, -0.6, 0.0], 1e-12);
    // The projection keeps the azimuth of the query point.
    assert_point_close(cone.closest_point([0.0, 0.0, 2.0]), [0.0, -0.6, 0.8], 1e-12);
    assert_point_close(
        cone.closest_point([-2.0, 0.0, 0.0]),
        [-0.8, -0.6, 0.0],
        1e-12,
    );
    let diagonal = 2.0 / 2.0_f64.sqrt();
    let cp = cone.closest_point([diagonal, 0.0, diagonal]);
    let want = 0.8 / 2.0_f64.sqrt();
    assert_point_close(cp, [want, -0.6, want], 1e-12);
    // |p - cp| is the distance to the slant line: |2*2 + 0 - 1| / sqrt(5).
    assert!(approx_eq(
        cone.signed_distance([2.0, 0.0, 0.0]),
        3.0 / 5.0_f64.sqrt(),
        1e-12
    ));
}

#[test]
fn points_on_the_axis_project_to_apex_base_or_slant() {
    let cone = Cone::new(1.0, 1.0);
    assert_point_close(cone.closest_point([0.0, 5.0, 0.0]), [0.0, 1.0, 0.0], 1e-12);
    assert_point_close(
        cone.closest_point([0.0, -5.0, 0.0]),
        [0.0, -1.0, 0.0],
        1e-12,
    );
    // Inside, on the axis: the slant (0.447 away) is nearer than the base (1).
    let cp = cone.closest_point([0.0, 0.0, 0.0]);
    assert_point_close(cp, [0.4, 0.2, 0.0], 1e-12);
    assert!(approx_eq(
        cone.signed_distance([0.0, 0.0, 0.0]),
        -1.0 / 5.0_f64.sqrt(),
        1e-12
    ));
}

#[test]
fn closest_point_is_a_nearest_boundary_point_everywhere() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    for &(radius, half_height) in &[(1.0, 1.0), (0.5, 2.0), (3.0, 0.7), (2.0, 2.0), (0.1, 1.5)] {
        let cone = Cone::new(radius, half_height);
        let reach = 2.0 * radius.max(half_height);
        for _ in 0..600 {
            let p = [
                rng.range(-reach, reach),
                rng.range(-reach, reach),
                rng.range(-reach, reach),
            ];
            let cp = cone.closest_point(p);
            let label = format!("cone ({radius}, {half_height}) point {p:?} -> {cp:?}");
            assert!(on_boundary(&cone, cp), "not on the boundary: {label}");
            let found = distance(p, cp);
            let sampled = sampled_boundary_distance(&cone, rho_of(p), p[1]);
            assert!(found <= sampled + 1e-9, "not nearest: {label} ({sampled})");
            assert!(sampled - found <= 1e-3, "sampling disagrees: {label}");
            // The nearest point lies in the meridian plane of the query point.
            let cross = p[0] * cp[2] - p[2] * cp[0];
            assert!(cross.abs() <= 1e-9 * (1.0 + found), "azimuth: {label}");
            // Signed distance: exact magnitude, sign from containment.
            let signed = cone.signed_distance(p);
            assert!(approx_eq(signed.abs(), found, 1e-12), "{label}");
            assert_eq!(
                signed < 0.0,
                cone.contains_point(p) && found > 0.0,
                "{label}"
            );
        }
    }
}

#[test]
fn interior_distance_is_not_inflated_by_the_slant() {
    // The interior distance used to be measured horizontally to the slant, which
    // overstates it by 1 / cos(half-angle).  Cone(1, 1) at (0, -0.2, 0): the slant
    // is (2 * 0 + (-0.2) - 1) / sqrt(5) = 0.537 away, the base 0.8.
    let cone = Cone::new(1.0, 1.0);
    let d = cone.signed_distance([0.0, -0.2, 0.0]);
    assert!(approx_eq(d, -1.2 / 5.0_f64.sqrt(), 1e-12), "distance {d}");
    assert!(approx_eq(cone.sdf([0.0, -0.2, 0.0]), d, 0.0));
    // Close to the base the base cap is the nearest surface.
    let near_base = cone.signed_distance([0.1, -0.9, 0.0]);
    assert!(approx_eq(near_base, -0.1, 1e-12), "distance {near_base}");
}

#[test]
fn sphere_tests_use_the_exact_distance() {
    let cone = Cone::new(1.0, 1.0);
    let rim_distance = 17.0_f64.sqrt();
    // A sphere at (5, 0, 0) reaches the cone once its radius passes sqrt(17).
    assert!(cone.intersects_sphere([5.0, 0.0, 0.0], 4.3));
    assert!(cone.intersects_sphere([5.0, 0.0, 0.0], rim_distance + 1e-9));
    assert!(!cone.intersects_sphere([5.0, 0.0, 0.0], 4.1));
    assert!(!cone.intersects_sphere([5.0, 0.0, 0.0], 4.0));
    // A small sphere whose center is inside the cone overlaps it.
    assert!(cone.intersects_sphere([0.0, 0.0, 0.0], 0.01));
    assert!(cone.intersects_sphere([0.0, 0.0, 0.0], 0.0));
    // It lies completely inside while its radius stays below the distance
    // 1 / sqrt(5) = 0.447 to the slant.
    assert!(cone.sphere_inside_cone([0.0, 0.0, 0.0], 0.44));
    assert!(!cone.sphere_inside_cone([0.0, 0.0, 0.0], 0.46));
}

#[test]
fn degenerate_cones_give_finite_answers() {
    // Zero radius: a needle along the axis; zero height: a disc of radius 1.
    let needle = Cone::new(0.0, 1.0);
    assert_point_close(
        needle.closest_point([1.0, 0.0, 0.0]),
        [0.0, 0.0, 0.0],
        1e-12,
    );
    assert_point_close(
        needle.closest_point([0.0, 3.0, 0.0]),
        [0.0, 1.0, 0.0],
        1e-12,
    );
    let flat = Cone::new(1.0, 0.0);
    for p in [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, -2.0, 0.4]] {
        let cp = flat.closest_point(p);
        assert!(cp.iter().all(|c| c.is_finite()), "{p:?} -> {cp:?}");
    }
    assert_point_close(flat.closest_point([2.0, 0.0, 0.0]), [1.0, 0.0, 0.0], 1e-12);
}
