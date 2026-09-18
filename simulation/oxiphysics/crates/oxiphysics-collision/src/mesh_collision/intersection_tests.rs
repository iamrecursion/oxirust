// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the triangle-triangle intersection test and the swept
//! triangle query.
//!
//! * `triangle_triangle_intersect` used to evaluate `0/0` for coplanar pairs and
//!   reported every one of them as intersecting (two triangles 100 m apart
//!   included); a vertex lying exactly on the other triangle's plane produced a
//!   one-point interval and lost real intersections.
//! * `swept_triangle_vs_mesh` only swept the vertices of the moving triangle
//!   against the static faces, so edge-first and spike-first contacts tunnelled.
//! * `vertex_triangle_ccd` sampled the path 64 times; a crossing that lands on a
//!   sample was missed and every time of impact was quantised to 1/64.

use super::*;

// ── helpers ──────────────────────────────────────────────────────────────────

fn tri(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> Triangle {
    Triangle::new(a, b, c)
}

fn translated(t: &Triangle, d: [f64; 3]) -> Triangle {
    Triangle::new(vec3_add(t.v0, d), vec3_add(t.v1, d), vec3_add(t.v2, d))
}

/// `t` moved by `s * displacement`.
fn moved(t: &Triangle, displacement: [f64; 3], s: f64) -> Triangle {
    translated(t, vec3_scale(displacement, s))
}

/// Apply `p -> offset + scale * p` to every vertex.
fn affine(t: &Triangle, scale: f64, offset: [f64; 3]) -> Triangle {
    let map = |p: [f64; 3]| vec3_add(offset, vec3_scale(p, scale));
    Triangle::new(map(t.v0), map(t.v1), map(t.v2))
}

/// Minimal deterministic generator so the randomised checks are reproducible.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn point(&mut self) -> [f64; 3] {
        [
            2.0 * self.next_f64() - 1.0,
            2.0 * self.next_f64() - 1.0,
            2.0 * self.next_f64() - 1.0,
        ]
    }
}

/// Independent oracle for generic (non-coplanar) pairs: two triangles intersect
/// iff an edge of one pierces the other.
fn edge_pierces(a: &Triangle, b: &Triangle) -> bool {
    [(a.v0, a.v1), (a.v1, a.v2), (a.v2, a.v0)]
        .iter()
        .any(|&(p, q)| ray_triangle_mt(p, vec3_sub(q, p), b, 0.0, 1.0).is_some())
}

fn oracle(a: &Triangle, b: &Triangle) -> bool {
    edge_pierces(a, b) || edge_pierces(b, a)
}

// ── coplanar pairs (F0135) ───────────────────────────────────────────────────

#[test]
fn coplanar_triangles_100_m_apart_do_not_intersect() {
    let t1 = tri([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let t2 = tri([100.0, 0.0, 0.0], [101.0, 0.0, 0.0], [100.0, 1.0, 0.0]);
    assert!(!triangle_triangle_intersect(&t1, &t2));
    assert!(!triangle_triangle_intersect(&t2, &t1));
}

#[test]
fn coplanar_triangles_with_overlapping_boxes_but_no_common_point_are_apart() {
    // The boxes share the square [3, 4]^2, the triangles lie on either side of
    // the line x + y = 4 .. 6.
    let t1 = tri([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]);
    let t2 = tri([3.0, 3.0, 0.0], [5.0, 3.0, 0.0], [3.0, 5.0, 0.0]);
    assert!(t1.aabb().overlaps(&t2.aabb()));
    assert!(!triangle_triangle_intersect(&t1, &t2));
    assert!(!triangle_triangle_intersect(&t2, &t1));
}

#[test]
fn coplanar_triangles_with_a_small_gap_do_not_intersect() {
    let t1 = tri([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let t2 = tri([1.001, 0.0, 0.0], [2.0, 0.0, 0.0], [1.001, 1.0, 0.0]);
    assert!(!triangle_triangle_intersect(&t1, &t2));
    assert!(!triangle_triangle_intersect(&t2, &t1));
}

#[test]
fn coplanar_overlapping_triangles_intersect() {
    let t1 = tri([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]);
    let t2 = tri([1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 2.0, 0.0]);
    assert!(triangle_triangle_intersect(&t1, &t2));
    assert!(triangle_triangle_intersect(&t2, &t1));
}

#[test]
fn coplanar_triangle_contained_in_another_intersects() {
    let big = tri([-4.0, -4.0, 0.0], [4.0, -4.0, 0.0], [0.0, 5.0, 0.0]);
    let small = tri([-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]);
    assert!(triangle_triangle_intersect(&big, &small));
    assert!(triangle_triangle_intersect(&small, &big));
}

#[test]
fn coplanar_triangles_in_a_tilted_plane_follow_the_same_rules() {
    // Plane x + y + z = 0 spanned by u = (1, -1, 0) and v = (0, 1, -1).
    let u = [1.0, -1.0, 0.0];
    let v = [0.0, 1.0, -1.0];
    let at = |a: f64, b: f64| vec3_add(vec3_scale(u, a), vec3_scale(v, b));
    let t1 = tri(at(0.0, 0.0), at(2.0, 0.0), at(0.0, 2.0));
    let overlapping = tri(at(1.0, 0.0), at(3.0, 0.0), at(1.0, 2.0));
    let apart = tri(at(5.0, 5.0), at(6.0, 5.0), at(5.0, 6.0));
    assert!(triangle_triangle_intersect(&t1, &overlapping));
    assert!(!triangle_triangle_intersect(&t1, &apart));
}

#[test]
fn coplanar_triangles_touching_at_a_vertex_or_sharing_an_edge_intersect() {
    let t1 = tri([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let at_vertex = tri([1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0]);
    let shared_edge = tri([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]);
    assert!(triangle_triangle_intersect(&t1, &at_vertex));
    assert!(triangle_triangle_intersect(&at_vertex, &t1));
    assert!(triangle_triangle_intersect(&t1, &shared_edge));
    assert!(triangle_triangle_intersect(&shared_edge, &t1));
}

#[test]
fn interlocked_coplanar_triangles_intersect_without_any_vertex_inside() {
    // Star of David: every vertex of one triangle is outside the other, only the
    // edges cross.
    let up = tri([-1.0, -0.5, 0.0], [1.0, -0.5, 0.0], [0.0, 1.232_050_8, 0.0]);
    let down = tri([-1.0, 0.5, 0.0], [1.0, 0.5, 0.0], [0.0, -1.232_050_8, 0.0]);
    assert!(triangle_triangle_intersect(&up, &down));
    assert!(triangle_triangle_intersect(&down, &up));
}

// ── generic pairs ────────────────────────────────────────────────────────────

#[test]
fn parallel_planes_do_not_intersect() {
    let t1 = tri([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let t2 = tri([0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]);
    assert!(!triangle_triangle_intersect(&t1, &t2));
}

#[test]
fn crossing_planes_with_disjoint_footprints_do_not_intersect() {
    // t2 crosses the plane z = 0, but 10 m away from t1.
    let t1 = tri([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let t2 = tri([10.0, 0.0, -1.0], [10.0, 1.0, -1.0], [10.0, 0.5, 1.0]);
    assert!(!triangle_triangle_intersect(&t1, &t2));
    assert!(!triangle_triangle_intersect(&t2, &t1));
}

#[test]
fn an_edge_piercing_a_face_intersects() {
    let t1 = tri([-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]);
    let t2 = tri([0.0, -1.0, -1.0], [0.0, 0.5, -1.0], [0.0, 0.0, 1.0]);
    assert!(triangle_triangle_intersect(&t1, &t2));
    assert!(triangle_triangle_intersect(&t2, &t1));
}

#[test]
fn a_vertex_on_the_other_plane_does_not_lose_the_intersection() {
    // t2 has one vertex exactly on the plane of t1 (z = 0) and one vertex on each
    // side of it. Its footprint on the plane is the segment from (0, -3, 0) to
    // (0, 0.5, 0), which overlaps t1 along y in [-2, 0.5].
    let t1 = tri([-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]);
    let t2 = tri([0.0, -3.0, 0.0], [0.0, 0.5, 1.0], [0.0, 0.5, -1.0]);
    assert!(triangle_triangle_intersect(&t1, &t2));
    assert!(triangle_triangle_intersect(&t2, &t1));
}

#[test]
fn a_vertex_touching_a_face_intersects() {
    let t1 = tri([-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]);
    let touching = tri([0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]);
    let hovering = tri([0.0, 0.0, 0.01], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]);
    assert!(triangle_triangle_intersect(&t1, &touching));
    assert!(!triangle_triangle_intersect(&t1, &hovering));
}

#[test]
fn random_pairs_agree_with_the_edge_piercing_oracle_and_are_symmetric() {
    let mut rng = Lcg(0x5EED_1234_ABCD_0042);
    let (mut hits, mut misses) = (0usize, 0usize);
    for _ in 0..4000 {
        let a = tri(rng.point(), rng.point(), rng.point());
        let b = tri(rng.point(), rng.point(), rng.point());
        if a.area() < 1e-2 || b.area() < 1e-2 {
            continue;
        }
        let expected = oracle(&a, &b);
        assert_eq!(triangle_triangle_intersect(&a, &b), expected, "{a:?} {b:?}");
        assert_eq!(triangle_triangle_intersect(&b, &a), expected, "{b:?} {a:?}");
        if expected {
            hits += 1;
        } else {
            misses += 1;
        }
    }
    assert!(hits > 200 && misses > 200, "hits {hits}, misses {misses}");
}

#[test]
fn the_result_does_not_depend_on_scale_or_position() {
    let t1 = tri([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]);
    let overlapping = tri([1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 2.0, 0.0]);
    let apart = tri([3.0, 3.0, 0.0], [5.0, 3.0, 0.0], [3.0, 5.0, 0.0]);
    let crossing = tri([0.5, 0.5, -1.0], [0.5, 1.0, 1.0], [1.0, 0.25, 0.5]);
    for &scale in &[1e-6, 1.0, 1e6] {
        for &offset in &[[0.0; 3], [1e3, -2e3, 5e2], [-7.0, 0.0, 4.0]] {
            let off = vec3_scale(offset, scale);
            let m = |t: &Triangle| affine(t, scale, off);
            assert!(triangle_triangle_intersect(&m(&t1), &m(&overlapping)));
            assert!(!triangle_triangle_intersect(&m(&t1), &m(&apart)));
            assert!(triangle_triangle_intersect(&m(&t1), &m(&crossing)));
        }
    }
}

// ── degenerate triangles ─────────────────────────────────────────────────────

#[test]
fn a_zero_area_triangle_acts_as_its_segment() {
    let face = tri([-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]);
    // A needle through the face, one along it, one beside it and one far above.
    let piercing = tri([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]);
    let lying_in = tri([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    let beside = tri([5.0, 0.0, -1.0], [5.0, 0.0, 1.0], [5.0, 0.0, 0.0]);
    let above = tri([0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.5, 0.0, 1.0]);
    assert!(triangle_triangle_intersect(&face, &piercing));
    assert!(triangle_triangle_intersect(&piercing, &face));
    assert!(triangle_triangle_intersect(&face, &lying_in));
    assert!(!triangle_triangle_intersect(&face, &beside));
    assert!(!triangle_triangle_intersect(&beside, &face));
    assert!(!triangle_triangle_intersect(&face, &above));
}

#[test]
fn a_point_triangle_is_inside_or_outside() {
    let face = tri([-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]);
    let inside = tri([0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0]);
    let outside = tri([0.5, 0.0, 0.5], [0.5, 0.0, 0.5], [0.5, 0.0, 0.5]);
    assert!(triangle_triangle_intersect(&face, &inside));
    assert!(triangle_triangle_intersect(&inside, &face));
    assert!(!triangle_triangle_intersect(&face, &outside));
    assert!(!triangle_triangle_intersect(&outside, &face));
}

#[test]
fn two_zero_area_triangles_meet_where_their_segments_meet() {
    let along_x = tri([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    let crossing = tri([0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 0.0]);
    let skew = tri([0.0, -1.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.5, 0.5]);
    let collinear_overlap = tri([0.5, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
    let collinear_apart = tri([2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.5, 0.0, 0.0]);
    assert!(triangle_triangle_intersect(&along_x, &crossing));
    assert!(!triangle_triangle_intersect(&along_x, &skew));
    assert!(triangle_triangle_intersect(&along_x, &collinear_overlap));
    assert!(!triangle_triangle_intersect(&along_x, &collinear_apart));
    assert!(!triangle_triangle_intersect(&collinear_apart, &along_x));
}

#[test]
fn non_finite_vertices_never_intersect() {
    let t1 = tri([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let nan = tri([f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let inf = tri([0.0, 0.0, 0.0], [f64::INFINITY, 0.0, 0.0], [0.0, 1.0, 0.0]);
    assert!(!triangle_triangle_intersect(&t1, &nan));
    assert!(!triangle_triangle_intersect(&nan, &t1));
    assert!(!triangle_triangle_intersect(&t1, &inf));
}

// ── mesh level ───────────────────────────────────────────────────────────────

#[test]
fn narrow_phase_drops_far_apart_coplanar_candidates() {
    let a = TriMeshBvh::build(vec![tri([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])]);
    let b = TriMeshBvh::build(vec![tri(
        [100.0, 0.0, 0.0],
        [101.0, 0.0, 0.0],
        [100.0, 1.0, 0.0],
    )]);
    let candidates = [TriPair { tri_a: 0, tri_b: 0 }];
    assert!(mesh_narrow_phase(&a, &b, &candidates).is_empty());
    assert!(mesh_mesh_collision(&a, &b).is_empty());
}

#[test]
fn coplanar_meshes_collide_only_where_their_triangles_overlap() {
    // a[0] overlaps b, a[1] only shares b's bounding box (it lies beyond the line
    // x + y = 3 that bounds b) and a[2] is far away.
    let a = TriMeshBvh::build(vec![
        tri([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]),
        tri([2.5, 1.5, 0.0], [3.5, 1.5, 0.0], [2.5, 2.5, 0.0]),
        tri([10.0, 0.0, 0.0], [12.0, 0.0, 0.0], [10.0, 2.0, 0.0]),
    ]);
    let b = TriMeshBvh::build(vec![tri([1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 2.0, 0.0])]);
    assert!(
        mesh_broad_phase(&a, &b).iter().any(|p| p.tri_a == 1),
        "the second triangle must be a broad-phase candidate"
    );
    let pairs = mesh_mesh_collision(&a, &b);
    assert_eq!(pairs.len(), 1);
    assert_eq!((pairs[0].tri_a, pairs[0].tri_b), (0, 0));
}

// ── swept triangle vs mesh (F0143) ───────────────────────────────────────────

const TOI_TOL: f64 = 1e-6;

fn single(t: Triangle) -> TriMeshBvh {
    TriMeshBvh::build(vec![t])
}

/// The invariant that ties the sweep to the static test: just before the time of
/// impact the triangles are apart, just after they intersect.
fn assert_first_contact(moving: &Triangle, displacement: [f64; 3], fixed: &Triangle, toi: f64) {
    let before = moved(moving, displacement, toi - 1e-4);
    let after = moved(moving, displacement, toi + 1e-4);
    assert!(
        !triangle_triangle_intersect(&before, fixed),
        "already touching before the reported time {toi}"
    );
    assert!(
        triangle_triangle_intersect(&after, fixed),
        "still apart after the reported time {toi}"
    );
}

#[test]
fn a_triangle_sweeping_edge_first_across_a_static_edge_is_detected() {
    // The moving triangle stands in the plane x = 0; its bottom edge (z = 3, from
    // y = -5 to y = 5) falls onto the horizontal static triangle. Its vertices
    // stay outside the static footprint and no static vertex lies in its plane,
    // so only edge-edge contact exists.
    let fixed = tri([-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.5, 1.0, 0.0]);
    let moving = tri([0.0, -5.0, 3.0], [0.0, 5.0, 3.0], [0.0, 0.0, 8.0]);
    let displacement = [0.0, 0.0, -4.0];
    let toi = swept_triangle_vs_mesh(&moving, displacement, &single(fixed.clone()))
        .expect("the bottom edge crosses the static triangle");
    assert!((toi - 0.75).abs() < TOI_TOL, "toi {toi}");
    assert_first_contact(&moving, displacement, &fixed, toi);
}

#[test]
fn a_static_spike_piercing_the_moving_face_is_detected() {
    // A thin vertical static triangle whose apex (0.1, 0, 1) pushes into a large
    // horizontal triangle falling from z = 2.
    let spike = tri([0.0, 0.0, -1.0], [0.2, 0.0, -1.0], [0.1, 0.0, 1.0]);
    let moving = tri([-2.0, -2.0, 2.0], [2.0, -2.0, 2.0], [0.0, 2.0, 2.0]);
    let displacement = [0.0, 0.0, -2.5];
    let toi = swept_triangle_vs_mesh(&moving, displacement, &single(spike.clone()))
        .expect("the spike apex meets the falling face");
    assert!((toi - 0.4).abs() < TOI_TOL, "toi {toi}");
    assert_first_contact(&moving, displacement, &spike, toi);
}

#[test]
fn a_crossing_that_lands_on_a_former_sample_time_is_not_missed() {
    // Face-on approach: every vertex reaches z = 0 at t = 0.5 exactly.
    let fixed = tri([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let moving = tri([0.2, 0.2, 1.0], [0.4, 0.2, 1.0], [0.2, 0.4, 1.0]);
    let toi = swept_triangle_vs_mesh(&moving, [0.0, 0.0, -2.0], &single(fixed))
        .expect("the sheets meet at t = 0.5");
    assert!((toi - 0.5).abs() < TOI_TOL, "toi {toi}");
}

#[test]
fn a_triangle_that_already_intersects_reports_time_zero() {
    let fixed = tri([-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]);
    let moving = tri([0.0, -1.0, -1.0], [0.0, 0.5, -1.0], [0.0, 0.0, 1.0]);
    assert_eq!(
        swept_triangle_vs_mesh(&moving, [0.0, 0.0, -3.0], &single(fixed.clone())),
        Some(0.0)
    );
    // Without any motion an overlap is still a contact at t = 0.
    assert_eq!(
        swept_triangle_vs_mesh(&moving, [0.0; 3], &single(fixed)),
        Some(0.0)
    );
}

#[test]
fn a_sweep_that_never_touches_returns_none() {
    let fixed = tri([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    // Falls past the static triangle without overlapping its footprint.
    let beside = tri([5.0, 5.0, 1.0], [6.0, 5.0, 1.0], [5.0, 6.0, 1.0]);
    assert_eq!(
        swept_triangle_vs_mesh(&beside, [0.0, 0.0, -2.0], &single(fixed.clone())),
        None
    );
    // Stops short of the static triangle.
    let short = tri([0.2, 0.2, 3.0], [0.4, 0.2, 3.0], [0.2, 0.4, 3.0]);
    assert_eq!(
        swept_triangle_vs_mesh(&short, [0.0, 0.0, -1.0], &single(fixed.clone())),
        None
    );
    // No motion, no overlap.
    assert_eq!(
        swept_triangle_vs_mesh(&short, [0.0; 3], &single(fixed.clone())),
        None
    );
    // The swept box overlaps the static triangle, but the sheet crosses the plane
    // z = 0 (at t = 0.5) beyond the hypotenuse x + y = 1, so nothing touches.
    let diagonal = tri([0.8, 0.8, 1.0], [0.9, 0.8, 1.0], [0.8, 0.9, 1.0]);
    let displacement = [-0.5, -0.5, -2.0];
    let mut swept_box = diagonal.aabb();
    swept_box.expand_aabb(&translated(&diagonal, displacement).aabb());
    assert!(swept_box.overlaps(&fixed.aabb()));
    assert_eq!(
        swept_triangle_vs_mesh(&diagonal, displacement, &single(fixed)),
        None
    );
}

#[test]
fn the_earliest_hit_over_all_candidate_triangles_wins() {
    let floor_high = tri([-5.0, -5.0, 0.0], [5.0, -5.0, 0.0], [0.0, 5.0, 0.0]);
    let floor_low = tri([-5.0, -5.0, -1.0], [5.0, -5.0, -1.0], [0.0, 5.0, -1.0]);
    let mesh = TriMeshBvh::build(vec![floor_low, floor_high]);
    let moving = tri([0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]);
    let toi = swept_triangle_vs_mesh(&moving, [0.0, 0.0, -3.0], &mesh).expect("falls onto both");
    assert!((toi - 1.0 / 3.0).abs() < TOI_TOL, "toi {toi}");
}

#[test]
fn a_non_finite_sweep_has_no_time_of_impact() {
    let fixed = tri([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let moving = tri([0.2, 0.2, 1.0], [0.4, 0.2, 1.0], [0.2, 0.4, 1.0]);
    let mesh = single(fixed);
    assert_eq!(
        swept_triangle_vs_mesh(&moving, [0.0, 0.0, f64::NAN], &mesh),
        None
    );
    assert_eq!(
        swept_triangle_vs_mesh(&moving, [f64::INFINITY, 0.0, 0.0], &mesh),
        None
    );
}

// ── vertex_triangle_ccd (F0139 / F0140) ──────────────────────────────────────

/// The analytic solver reports the exact root; the margin only absorbs the
/// contact shell of the fallback search.
const VERTEX_TOI_TOL: f64 = 1e-7;

fn unit_triangle() -> Triangle {
    tri([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
}

/// `vertex_triangle_ccd` of a vertex path against a triangle that stands still.
fn vertex_ccd_static(v0: [f64; 3], v1: [f64; 3], t: &Triangle) -> Option<CcdHit> {
    vertex_triangle_ccd(v0, v1, t.v0, t.v0, t.v1, t.v1, t.v2, t.v2)
}

#[test]
fn vertex_ccd_time_of_impact_is_exact_not_quantised() {
    let hit = vertex_ccd_static([0.25, 0.25, 1.0], [0.25, 0.25, -2.0], &unit_triangle())
        .expect("the vertex pierces the triangle");
    // The old sampler answered 21/64 = 0.328125.
    assert!(
        (hit.toi - 1.0 / 3.0).abs() < VERTEX_TOI_TOL,
        "toi {}",
        hit.toi
    );
    assert!((hit.u - 0.25).abs() < 1e-6 && (hit.v - 0.25).abs() < 1e-6);
    assert!((hit.normal[2].abs() - 1.0).abs() < 1e-12);
}

#[test]
fn vertex_ccd_finds_a_crossing_that_falls_on_a_sample() {
    // z reaches 0 at t = 0.5, a former sample time where the old test saw a zero
    // distance and skipped the sign change.
    let hit = vertex_ccd_static([0.25, 0.25, 1.0], [0.25, 0.25, -1.0], &unit_triangle())
        .expect("a crossing exactly at t = 0.5 must be found");
    assert!((hit.toi - 0.5).abs() < VERTEX_TOI_TOL, "toi {}", hit.toi);
}

#[test]
fn vertex_ccd_misses_a_path_beside_the_triangle() {
    assert!(vertex_ccd_static([2.0, 2.0, 1.0], [2.0, 2.0, -1.0], &unit_triangle()).is_none());
    assert!(vertex_ccd_static([0.25, 0.25, 1.0], [0.25, 0.25, 0.5], &unit_triangle()).is_none());
}

#[test]
fn vertex_ccd_handles_a_moving_triangle() {
    // The triangle rises through a vertex that stays at (0.25, 0.25, 0.5).
    let v = [0.25, 0.25, 0.5];
    let hit = vertex_triangle_ccd(
        v,
        v,
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, -1.0],
        [0.0, 1.0, 1.0],
    )
    .expect("the rising sheet meets the vertex");
    // The sheet is at z = -1 + 2 s and reaches z = 0.5 at s = 0.75.
    assert!((hit.toi - 0.75).abs() < VERTEX_TOI_TOL, "toi {}", hit.toi);
}
