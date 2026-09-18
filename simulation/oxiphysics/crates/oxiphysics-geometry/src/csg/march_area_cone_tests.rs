// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the relaxed sphere march, the grid surface-area
//! estimator, the non-cubic marching-cubes sampler and the exact signed
//! distance of the CSG cone.

use super::*;
use std::f64::consts::PI;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

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

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    length(sub(a, b))
}

/// Field that counts how often it is evaluated.
struct CountingField<S: ImplicitSurface> {
    inner: S,
    evaluations: Arc<AtomicUsize>,
}

impl<S: ImplicitSurface> CountingField<S> {
    fn new(inner: S) -> (Self, Arc<AtomicUsize>) {
        let evaluations = Arc::new(AtomicUsize::new(0));
        (
            Self {
                inner,
                evaluations: Arc::clone(&evaluations),
            },
            evaluations,
        )
    }
}

impl<S: ImplicitSurface> ImplicitSurface for CountingField<S> {
    fn sdf(&self, p: [f64; 3]) -> f64 {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        self.inner.sdf(p)
    }

    fn gradient(&self, p: [f64; 3]) -> [f64; 3] {
        self.inner.gradient(p)
    }
}

/// The thick slab `1 <= x <= 5` with a field that overestimates the distance
/// by `scale` (so it is not a valid signed distance field for `scale > 1`).
struct ScaledSlab {
    scale: f64,
}

impl ImplicitSurface for ScaledSlab {
    fn sdf(&self, p: [f64; 3]) -> f64 {
        self.scale * (1.0 - p[0]).max(p[0] - 5.0)
    }

    fn gradient(&self, _p: [f64; 3]) -> [f64; 3] {
        [-1.0, 0.0, 0.0]
    }
}

/// A field that is not a number anywhere.
struct NanField;

impl ImplicitSurface for NanField {
    fn sdf(&self, _p: [f64; 3]) -> f64 {
        f64::NAN
    }

    fn gradient(&self, _p: [f64; 3]) -> [f64; 3] {
        [0.0; 3]
    }
}

// ── sphere_march_relaxed ──────────────────────────────────────────────────

#[test]
fn relaxed_march_does_not_tunnel_through_a_thin_slab() {
    // A slab of half-thickness 0.01 at x = 1.01.  From the origin the first
    // relaxed step (1.0 * 1.9) lands 0.88 beyond the far face.
    let slab = SdfBox::new([1.01, 0.0, 0.0], [0.01, 5.0, 5.0]);
    let (hit, normal) = sphere_march_relaxed(&slab, [0.0; 3], [1.0, 0.0, 0.0], 100.0, 256, 1.9)
        .expect("the relaxed march must hit the slab");
    assert!((hit[0] - 1.0).abs() < 1e-4, "hit at {hit:?}");
    assert!(hit[1].abs() < 1e-9 && hit[2].abs() < 1e-9, "hit at {hit:?}");
    assert!((normal[0] + 1.0).abs() < 1e-3, "normal {normal:?}");
    // The plain march finds the same point.
    let (plain, _) = sphere_march(&slab, [0.0; 3], [1.0, 0.0, 0.0], 100.0, 256)
        .expect("plain sphere tracing hits the slab");
    assert!(distance(hit, plain) < 1e-4, "{hit:?} vs {plain:?}");
}

#[test]
fn relaxed_march_hits_the_near_side_when_a_step_lands_on_the_far_side() {
    // From x = -5 the first step of 4 * 1.5 = 6 lands exactly on the far side of
    // the unit sphere; the hit must still be the near side at x = -1.
    let sphere = SdfSphere::new([0.0; 3], 1.0);
    let (hit, normal) =
        sphere_march_relaxed(&sphere, [-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0, 256, 1.5)
            .expect("hit");
    assert!((hit[0] + 1.0).abs() < 1e-4, "hit at {hit:?}");
    assert!((normal[0] + 1.0).abs() < 1e-3, "normal {normal:?}");
}

#[test]
fn relaxed_march_agrees_with_plain_tracing_for_every_omega() {
    let slab = SdfBox::new([1.01, 0.0, 0.0], [0.01, 5.0, 5.0]);
    let sphere = SdfSphere::new([0.0; 3], 1.0);
    let origin_slab = [0.0; 3];
    let origin_sphere = [-5.0, 0.0, 0.0];
    let dir = [1.0, 0.0, 0.0];
    let slab_hit = sphere_march(&slab, origin_slab, dir, 100.0, 512).expect("slab");
    let sphere_hit = sphere_march(&sphere, origin_sphere, dir, 100.0, 512).expect("sphere");
    for omega in [0.5, 1.0, 1.2, 1.5, 1.9, 2.5, 4.0, 10.0] {
        let slab_relaxed = sphere_march_relaxed(&slab, origin_slab, dir, 100.0, 512, omega)
            .unwrap_or_else(|| panic!("omega {omega}: slab missed"));
        assert!(
            distance(slab_relaxed.0, slab_hit.0) < 1e-4,
            "omega {omega}: {:?} vs {:?}",
            slab_relaxed.0,
            slab_hit.0
        );
        let sphere_relaxed = sphere_march_relaxed(&sphere, origin_sphere, dir, 100.0, 512, omega)
            .unwrap_or_else(|| panic!("omega {omega}: sphere missed"));
        assert!(
            distance(sphere_relaxed.0, sphere_hit.0) < 1e-4,
            "omega {omega}: {:?} vs {:?}",
            sphere_relaxed.0,
            sphere_hit.0
        );
    }
}

#[test]
fn relaxed_march_refines_a_step_that_ends_inside_by_bisection() {
    // The field of the slab 1 <= x <= 5 is scaled by 1.2, so the unbounding
    // spheres of the first two samples overlap although the second sample
    // (x = 1.44, field -0.528) is already inside: the surface at x = 1 must be
    // found by bisection between the samples.
    let slab = ScaledSlab { scale: 1.2 };
    let (hit, normal) =
        sphere_march_relaxed(&slab, [0.0; 3], [1.0, 0.0, 0.0], 100.0, 64, 1.2).expect("hit");
    assert!((hit[0] - 1.0).abs() < 1e-5, "hit at {hit:?}");
    assert!(slab.sdf(hit).abs() < 1e-5);
    assert!((normal[0] + 1.0).abs() < 1e-3, "normal {normal:?}");
}

#[test]
fn relaxed_march_from_inside_hits_at_the_origin() {
    let sphere = SdfSphere::new([0.0; 3], 2.0);
    let origin = [0.25, -0.5, 0.125];
    let (hit, _) = sphere_march_relaxed(&sphere, origin, [0.0, 1.0, 0.0], 100.0, 64, 1.6)
        .expect("a ray that starts inside hits at its origin");
    assert_eq!(hit, origin);
}

#[test]
fn relaxed_march_respects_its_limits_and_bad_input() {
    let sphere = SdfSphere::new([0.0; 3], 1.0);
    let dir = [1.0, 0.0, 0.0];
    // A ray that passes beside the sphere misses.
    assert!(sphere_march_relaxed(&sphere, [-5.0, 3.0, 0.0], dir, 100.0, 256, 1.5).is_none());
    // The sphere is further away than `max_dist`.
    assert!(sphere_march_relaxed(&sphere, [-5.0, 0.0, 0.0], dir, 3.0, 256, 1.5).is_none());
    // One step is not enough to arrive.
    assert!(sphere_march_relaxed(&sphere, [-5.0, 0.0, 0.0], dir, 100.0, 1, 1.0).is_none());
    // A NaN field never hits.
    assert!(sphere_march_relaxed(&NanField, [0.0; 3], dir, 100.0, 64, 1.5).is_none());
    // A non-positive or non-finite relaxation factor is plain tracing.
    for omega in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let (hit, _) = sphere_march_relaxed(&sphere, [-5.0, 0.0, 0.0], dir, 100.0, 256, omega)
            .unwrap_or_else(|| panic!("omega {omega} must behave like plain tracing"));
        assert!((hit[0] + 1.0).abs() < 1e-4, "omega {omega}: {hit:?}");
    }
}

#[test]
fn relaxed_march_matches_plain_tracing_on_thin_feature_scenes() {
    // Two thin plates and a small sphere; plain sphere tracing never skips a
    // surface, so every relaxed result must agree with it.
    let plate = SdfBox::new([0.0; 3], [0.02, 1.5, 1.5]);
    let plank = SdfBox::new([0.5, 0.5, 0.0], [0.3, 0.015, 0.8]);
    let ball = SdfSphere::new([-1.0, 0.0, 0.7], 0.3);
    let scene = CsgUnion::new(
        Box::new(CsgUnion::new(Box::new(plate), Box::new(plank))),
        Box::new(ball),
    );
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut hits = 0;
    for ray in 0..400 {
        let origin = [
            rng.range(-6.0, 6.0),
            rng.range(-6.0, 6.0),
            rng.range(-6.0, 6.0),
        ];
        let target = [
            rng.range(-1.5, 1.5),
            rng.range(-1.5, 1.5),
            rng.range(-1.5, 1.5),
        ];
        let dir = normalize(sub(target, origin));
        let plain = sphere_march(&scene, origin, dir, 50.0, 6000);
        let relaxed = sphere_march_relaxed(&scene, origin, dir, 50.0, 6000, 1.7);
        match (plain, relaxed) {
            (Some((a, _)), Some((b, _))) => {
                hits += 1;
                assert!(
                    distance(a, b) < 5e-3,
                    "ray {ray}: plain hit {a:?}, relaxed hit {b:?}"
                );
            }
            (None, None) => {}
            (plain, relaxed) => {
                panic!("ray {ray}: plain {plain:?} but relaxed {relaxed:?}");
            }
        }
    }
    assert!(hits > 100, "only {hits} of 400 rays hit the scene");
}

// ── csg_surface_area ──────────────────────────────────────────────────────

#[test]
fn unit_sphere_area_is_within_two_percent() {
    let sphere = CsgTree::leaf(SdfSphere::new([0.0; 3], 1.0));
    let want = 4.0 * PI;
    let area = csg_surface_area(&sphere, [-1.5; 3], [1.5; 3], 64);
    assert!(
        (area - want).abs() < 0.02 * want,
        "area {area} vs {want} (the unweighted crossing count gave 1.5 * {want})"
    );
    let default_resolution = csg_node_surface_area(&sphere, [-1.5; 3], [1.5; 3]);
    assert!(
        (default_resolution - want).abs() < 0.02 * want,
        "default-resolution area {default_resolution} vs {want}"
    );
}

#[test]
fn sphere_area_is_accurate_at_every_resolution() {
    // The weighted crossings are a smooth quadrature of n_x^2 + n_y^2 + n_z^2, so
    // even coarse grids are close; the unweighted count was 50 % too large at
    // every resolution.
    let sphere = CsgTree::leaf(SdfSphere::new([0.0; 3], 1.0));
    let want = 4.0 * PI;
    for n in [8, 16, 32, 64] {
        let area = csg_surface_area(&sphere, [-1.5; 3], [1.5; 3], n);
        let error = (area - want).abs() / want;
        assert!(
            error < 0.015,
            "n = {n}: area {area}, relative error {error}"
        );
    }
}

#[test]
fn grid_aligned_box_area_is_exact() {
    // Cell faces coincide with the faces of the box 2 x 1 x 0.5, whose area is
    // 2 * (2 * 1 + 2 * 0.5 + 1 * 0.5) = 7.
    let block = CsgTree::leaf(SdfBox::new([0.0; 3], [1.0, 0.5, 0.25]));
    let area = csg_surface_area(&block, [-2.0; 3], [2.0; 3], 16);
    assert!((area - 7.0).abs() < 1e-9, "area {area}");
}

#[test]
fn composite_areas_follow_the_boolean_operations() {
    let radius = 1.0;
    // Two disjoint spheres: the areas add.
    let pair = CsgTree::union(
        CsgTree::leaf(SdfSphere::new([-2.0, 0.0, 0.0], radius)),
        CsgTree::leaf(SdfSphere::new([2.0, 0.0, 0.0], radius)),
    );
    let area = csg_surface_area(&pair, [-3.5, -1.5, -1.5], [3.5, 1.5, 1.5], 96);
    let want = 2.0 * 4.0 * PI;
    assert!(
        (area - want).abs() < 0.03 * want,
        "pair area {area} vs {want}"
    );
    // A shell: outer sphere minus an inner one, both surfaces count.
    let shell = CsgTree::difference(
        CsgTree::leaf(SdfSphere::new([0.0; 3], 1.0)),
        CsgTree::leaf(SdfSphere::new([0.0; 3], 0.5)),
    );
    let area = csg_surface_area(&shell, [-1.5; 3], [1.5; 3], 64);
    let want = 4.0 * PI * (1.0 + 0.25);
    assert!(
        (area - want).abs() < 0.03 * want,
        "shell area {area} vs {want}"
    );
}

#[test]
fn area_estimator_handles_degenerate_boxes_and_counts_samples_once() {
    let sphere = CsgTree::leaf(SdfSphere::new([0.0; 3], 1.0));
    // No extent along an axis, or a non-finite bound: no area.
    assert_eq!(
        csg_surface_area(&sphere, [-1.0, -1.0, 0.0], [1.0, 1.0, 0.0], 16),
        0.0
    );
    assert_eq!(csg_surface_area(&sphere, [f64::NAN; 3], [1.0; 3], 16), 0.0);
    assert_eq!(
        csg_surface_area(&sphere, [-1.0; 3], [f64::INFINITY; 3], 16),
        0.0
    );
    // Reversed bounds describe the same box.
    let forward = csg_surface_area(&sphere, [-1.5; 3], [1.5; 3], 24);
    let reversed = csg_surface_area(&sphere, [1.5; 3], [-1.5; 3], 24);
    assert_eq!(forward, reversed);
    // One field evaluation per cell plus six per crossing (the central
    // differences), not four per cell.
    let (counted, evaluations) = CountingField::new(SdfSphere::new([0.0; 3], 0.4));
    let tree = CsgTree::leaf(counted);
    let n = 24_usize;
    let area = csg_surface_area(&tree, [-1.0; 3], [1.0; 3], n);
    assert!(area > 0.0);
    let used = evaluations.load(Ordering::Relaxed);
    assert!(used >= n * n * n, "{used} evaluations");
    assert!(used < n * n * n + 6 * 3 * n * n, "{used} evaluations");
}

// ── marching_cubes_sample ─────────────────────────────────────────────────

#[test]
fn non_cubic_bounds_are_covered_to_the_far_end() {
    // The box is four times longer along x.  With the smallest step on every
    // axis the grid reached only x = 1 and the sphere at x = 3.5 was lost.
    let sphere = SdfSphere::new([3.5, 0.5, 0.5], 0.4);
    let cells = marching_cubes_sample(&sphere, [0.0; 3], [4.0, 1.0, 1.0], 8);
    assert!(!cells.is_empty(), "the sphere at the far end was lost");
    for cell in &cells {
        assert_eq!(cell.step, [0.5, 0.125, 0.125]);
        assert!(cell.min[0] >= 3.0 - 1e-12, "cell at {:?}", cell.min);
        for axis in 0..3 {
            let upper = [4.0, 1.0, 1.0][axis];
            assert!(cell.min[axis] >= 0.0 && cell.min[axis] + cell.step[axis] <= upper + 1e-12);
        }
    }
    // At a finer resolution the edge vertices lie on the sphere to within the
    // linear interpolation error of the field.
    let vertices = marching_cubes_vertices(&sphere, [0.0; 3], [4.0, 1.0, 1.0], 32);
    assert!(vertices.len() > 100, "{} vertices", vertices.len());
    for v in &vertices {
        assert!(sphere.sdf(*v).abs() < 0.02, "vertex {v:?} off the sphere");
        assert!(
            v[0] > 3.0 && v[0] < 4.0,
            "vertex {v:?} outside the sphere's span"
        );
    }
}

#[test]
fn cells_tile_a_non_cubic_box_exactly() {
    // The half-space x < 2 has its boundary on the cell face at x = 2, so the
    // surface cells are exactly the 8 x 8 cells of the layer [1.5, 2.0].
    struct HalfSpace;
    impl ImplicitSurface for HalfSpace {
        fn sdf(&self, p: [f64; 3]) -> f64 {
            p[0] - 2.0
        }
        fn gradient(&self, _p: [f64; 3]) -> [f64; 3] {
            [1.0, 0.0, 0.0]
        }
    }
    let cells = marching_cubes_sample(&HalfSpace, [0.0; 3], [4.0, 1.0, 1.0], 8);
    assert_eq!(cells.len(), 64);
    let mut seen = std::collections::BTreeSet::new();
    for cell in &cells {
        assert_eq!(cell.min[0], 1.5);
        assert_eq!(cell.step, [0.5, 0.125, 0.125]);
        seen.insert((
            (cell.min[1] * 8.0).round() as i64,
            (cell.min[2] * 8.0).round() as i64,
        ));
    }
    assert_eq!(
        seen.len(),
        64,
        "every (y, z) cell of the layer appears once"
    );
}

#[test]
fn every_corner_is_sampled_exactly_once() {
    let (counted, evaluations) = CountingField::new(SdfSphere::new([0.5; 3], 0.3));
    let n = 10_usize;
    let cells = marching_cubes_sample(&counted, [0.0; 3], [1.0, 1.0, 1.0], n);
    assert!(!cells.is_empty());
    assert_eq!(
        evaluations.load(Ordering::Relaxed),
        (n + 1) * (n + 1) * (n + 1)
    );
    // A degenerate box has no cells and costs nothing.
    let (counted, evaluations) = CountingField::new(SdfSphere::new([0.5; 3], 0.3));
    assert!(marching_cubes_sample(&counted, [0.0; 3], [1.0, 0.0, 1.0], n).is_empty());
    assert!(marching_cubes_sample(&counted, [f64::NAN; 3], [1.0; 3], n).is_empty());
    assert_eq!(evaluations.load(Ordering::Relaxed), 0);
}

#[test]
fn cell_corner_order_and_per_axis_steps_place_the_edge_vertices() {
    // Only corner 0, the minimum corner, is inside; its three edges go to
    // corners 1 (+x), 3 (+y) and 4 (+z).  With values -1 and +1 every vertex is
    // at the middle of its edge, which is half a step along that axis.
    let cell = MarchingCell {
        min: [10.0, 20.0, 30.0],
        step: [2.0, 1.0, 0.5],
        corner_values: [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    };
    assert!(cell.has_surface());
    let vertices = cell.extract_vertices();
    assert_eq!(vertices.len(), 3);
    for want in [[11.0, 20.0, 30.0], [10.0, 20.5, 30.0], [10.0, 20.0, 30.25]] {
        assert!(
            vertices.iter().any(|v| distance(*v, want) < 1e-12),
            "missing vertex {want:?} in {vertices:?}"
        );
    }
}

#[test]
fn cell_classification_treats_negative_zero_like_the_surface_test() {
    // `-0.0` is not negative: `has_surface` counts it as outside, so the edges
    // to the inside corners must produce vertices.
    let cell = MarchingCell {
        min: [0.0; 3],
        step: [1.0; 3],
        corner_values: [-0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    };
    assert!(cell.has_surface());
    assert_eq!(cell.extract_vertices().len(), 3);
}

// ── SdfCone ───────────────────────────────────────────────────────────────

/// Distance from `p` to the boundary of the cone with apex `apex`, half-angle
/// `alpha` and height `height`, found by sampling the slant and the base in the
/// meridian plane of `p`.
fn sampled_cone_distance(apex: [f64; 3], alpha: f64, height: f64, p: [f64; 3]) -> f64 {
    const SAMPLES: usize = 3000;
    let rho = (p[0] - apex[0]).hypot(p[2] - apex[2]);
    let y = p[1] - apex[1];
    let rim = height * alpha.tan();
    let mut best = f64::INFINITY;
    for i in 0..=SAMPLES {
        let t = i as f64 / SAMPLES as f64;
        let slant = [t * rim, t * height];
        let base = [t * rim, height];
        for q in [slant, base] {
            best = best.min((rho - q[0]).hypot(y - q[1]));
        }
    }
    best
}

fn cone_contains(apex: [f64; 3], alpha: f64, height: f64, p: [f64; 3]) -> bool {
    let rho = (p[0] - apex[0]).hypot(p[2] - apex[2]);
    let y = p[1] - apex[1];
    y >= 0.0 && y <= height && rho <= y * alpha.tan()
}

#[test]
fn cone_below_the_apex_is_not_overestimated() {
    // Apex at the origin, half-angle 45 degrees.  The point (10, -1, 0) lies
    // below the apex; the nearest point of the cone is the foot (4.5, 4.5, 0) on
    // the slant, 5.5 * sqrt(2) = 7.778 away.  The old code returned |p| = 10.05.
    let cone = SdfCone::new([0.0; 3], PI / 4.0, 100.0);
    let d = cone.sdf([10.0, -1.0, 0.0]);
    assert!((d - 5.5 * 2.0_f64.sqrt()).abs() < 1e-9, "distance {d}");
    // Straight below the apex the nearest point is the apex itself.
    assert!((cone.sdf([0.0, -3.0, 0.0]) - 3.0).abs() < 1e-12);
}

#[test]
fn cone_matches_the_distance_to_the_sampled_boundary() {
    let mut rng = Rng(0x0123_4567_89AB_CDEF);
    for &(alpha, height) in &[(PI / 4.0, 2.0), (0.3, 3.0), (1.2, 0.8), (PI / 6.0, 1.0)] {
        let apex = [
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ];
        let cone = SdfCone::new(apex, alpha, height);
        let reach = 1.5 * (height + height * alpha.tan());
        for _ in 0..400 {
            let p = [
                apex[0] + rng.range(-reach, reach),
                apex[1] + rng.range(-reach, reach),
                apex[2] + rng.range(-reach, reach),
            ];
            let d = cone.sdf(p);
            let sampled = sampled_cone_distance(apex, alpha, height, p);
            let label =
                format!("alpha {alpha}, height {height}, p {p:?}: sdf {d}, sampled {sampled}");
            assert!(d.abs() <= sampled + 1e-9, "{label}");
            assert!(sampled - d.abs() < 2e-3, "{label}");
            assert_eq!(d < 0.0, cone_contains(apex, alpha, height, p), "{label}");
        }
    }
}

#[test]
fn cone_has_no_dead_zone_above_the_base_or_beside_the_slant() {
    let cone = SdfCone::new([1.0, 2.0, 3.0], PI / 6.0, 1.0);
    // Above the base disc the distance is the height above the base plane.
    for lift in [0.01, 0.5, 4.0] {
        let d = cone.sdf([1.0, 3.0 + lift, 3.0]);
        assert!((d - lift).abs() < 1e-12, "lift {lift}: {d}");
    }
    // On the axis, inside, the nearer of slant and base wins.
    let inside = cone.sdf([1.0, 2.5, 3.0]);
    assert!((inside + 0.25).abs() < 1e-12, "inside {inside}");
    // The surface itself is at distance zero, e.g. the apex and the rim.
    assert!(cone.sdf([1.0, 2.0, 3.0]).abs() < 1e-12);
    let rim = (PI / 6.0).tan();
    assert!(cone.sdf([1.0 + rim, 3.0, 3.0]).abs() < 1e-12);
}

#[test]
fn cone_gradient_points_away_from_the_surface() {
    let cone = SdfCone::new([0.0; 3], PI / 6.0, 1.0);
    // Beside the slant, in the x-y plane: the outward normal of the generator
    // is (cos a, -sin a).
    let g = cone.gradient([1.0, 0.4, 0.0]);
    assert!((g[0] - (PI / 6.0).cos()).abs() < 1e-4, "gradient {g:?}");
    assert!((g[1] + (PI / 6.0).sin()).abs() < 1e-4, "gradient {g:?}");
    // Above the base the gradient is +Y.
    let up = cone.gradient([0.0, 2.0, 0.0]);
    assert!((up[1] - 1.0).abs() < 1e-6, "gradient {up:?}");
}
