// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the incremental Garland–Heckbert decimation engine
//! ([`ProgressiveMesh`], [`simplify_with_features`], [`QemDecimation`]).

use super::functions::{
    compute_vertex_quadrics, edge_collapse_cost, find_boundary_edges, mesh_volume,
    simplify_with_features,
};
use super::qem_engine::{CollapsePolicy, QemEngine};
use super::types::{
    EdgePriorityQueue, MeshEdge, ProgressiveMesh, QemDecimation, QuadricMatrix, SimpleMesh,
};
use crate::test_meshes;
use std::collections::HashSet;
use std::f64::consts::PI;
use std::time::Instant;

/// Unit icosphere with outward, counter-clockwise triangles
/// (`20 * 4^subdivisions` triangles).
fn icosphere(subdivisions: usize) -> SimpleMesh {
    let (vertices, triangles) = test_meshes::icosphere(subdivisions);
    SimpleMesh {
        vertices,
        triangles,
    }
}

/// An open `n x n` quad grid in the plane z = 0, two triangles per quad.
fn flat_grid(n: usize) -> SimpleMesh {
    let (vertices, triangles) = test_meshes::flat_grid(n, n, 1.0);
    SimpleMesh {
        vertices,
        triangles,
    }
}

fn unit_cube() -> SimpleMesh {
    let mut m = SimpleMesh::new();
    for &v in &[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ] {
        m.add_vertex(v);
    }
    for t in [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
    ] {
        m.add_triangle(t[0], t[1], t[2]);
    }
    m
}

fn referenced_vertices(mesh: &SimpleMesh) -> HashSet<usize> {
    mesh.triangles.iter().flatten().copied().collect()
}

fn undirected_edge_count(mesh: &SimpleMesh) -> usize {
    let mut edges: HashSet<(usize, usize)> = HashSet::new();
    for &[a, b, c] in &mesh.triangles {
        for (p, q) in [(a, b), (b, c), (c, a)] {
            edges.insert((p.min(q), p.max(q)));
        }
    }
    edges.len()
}

fn assert_valid(mesh: &SimpleMesh) {
    for &[a, b, c] in &mesh.triangles {
        assert!(a < mesh.vertices.len() && b < mesh.vertices.len() && c < mesh.vertices.len());
        assert!(
            a != b && b != c && a != c,
            "degenerate triangle {a} {b} {c}"
        );
    }
    for v in &mesh.vertices {
        assert!(v.iter().all(|c| c.is_finite()), "non-finite vertex {v:?}");
    }
}

#[test]
fn icosphere_helper_is_closed_and_outward() {
    let m = icosphere(2);
    assert_eq!(m.triangle_count(), 320);
    assert_eq!(m.vertex_count(), 162);
    assert!(find_boundary_edges(&m).is_empty());
    assert!(mesh_volume(&m) > 0.0, "triangles must face outwards");
}

#[test]
fn twenty_thousand_triangle_sphere_decimates_to_two_thousand() {
    let mesh = icosphere(5);
    assert_eq!(mesh.triangle_count(), 20480);
    let start = Instant::now();
    let mut pm = ProgressiveMesh::new(mesh);
    pm.decimate_to(2000);
    let elapsed = start.elapsed().as_secs_f64();
    // The incremental engine needs well under a second (about 0.1 s measured on a
    // machine under heavy load) while the old per-collapse rebuild was quadratic in
    // the triangle count; the bound leaves room for a loaded CI machine.
    // `engine_work_grows_linearly_with_the_mesh_size` is the load-independent
    // check of the complexity.
    assert!(elapsed < 5.0, "decimation took {elapsed:.2} s");

    let reduced = &pm.mesh;
    assert!(
        (1990..=2000).contains(&reduced.triangle_count()),
        "{} triangles",
        reduced.triangle_count()
    );
    assert_valid(reduced);
    assert_eq!(pm.history.len(), (20480 - reduced.triangle_count()) / 2);

    // The surviving vertices stay on the sphere, so the right collapses were chosen.
    for v in referenced_vertices(reduced) {
        let p = reduced.vertices[v];
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        assert!((r - 1.0).abs() < 0.02, "vertex {v} at radius {r}");
    }
    let volume = mesh_volume(reduced);
    let area = reduced.surface_area();
    assert!(
        (volume / (4.0 / 3.0 * PI) - 1.0).abs() < 0.03,
        "volume {volume}"
    );
    assert!((area / (4.0 * PI) - 1.0).abs() < 0.03, "area {area}");
    // Still a sphere topologically: V - E + F = 2.
    let v = referenced_vertices(reduced).len() as i64;
    let e = undirected_edge_count(reduced) as i64;
    let f = reduced.triangle_count() as i64;
    assert_eq!(v - e + f, 2, "Euler characteristic of the reduced sphere");
}

#[test]
fn engine_work_grows_linearly_with_the_mesh_size() {
    let evaluations = |subdivisions: usize, keep_divisor: usize| -> (usize, usize) {
        let mesh = icosphere(subdivisions);
        let faces = mesh.triangle_count();
        let Some(mut engine) = QemEngine::new(&mesh, CollapsePolicy::Unrestricted) else {
            panic!("valid mesh");
        };
        while engine.live_triangles() > faces / keep_divisor {
            assert!(engine.collapse_cheapest(f64::INFINITY).is_some());
        }
        (faces, engine.cost_evaluations())
    };
    let (small_faces, small) = evaluations(3, 10);
    let (large_faces, large) = evaluations(5, 10);
    assert_eq!(large_faces, 16 * small_faces);
    // Each collapse re-costs only the edges around the surviving vertex.
    assert!(
        large < 12 * large_faces,
        "{large} cost evaluations for {large_faces} triangles"
    );
    // Sixteen times the triangles must cost about sixteen times the work (the old
    // algorithm scaled quadratically, i.e. by a factor of 256).
    let ratio = large as f64 / small as f64;
    assert!(
        ratio < 24.0,
        "work grew by {ratio:.1}x for 16x the triangles"
    );
}

#[test]
fn collapse_merges_the_quadrics_of_its_end_points() {
    let mesh = icosphere(2);
    let Some(mut engine) = QemEngine::new(&mesh, CollapsePolicy::Unrestricted) else {
        panic!("valid mesh");
    };
    let before = compute_vertex_quadrics(&mesh);
    // Quadrics are summed over incident faces exactly as the reference does.
    assert_eq!(before.len(), mesh.vertex_count());
    for (v, reference) in before.iter().enumerate() {
        for i in 0..4 {
            for j in 0..4 {
                assert!((engine.quadric(v).q[i][j] - reference.q[i][j]).abs() < 1e-12);
            }
        }
    }
    let Some(outcome) = engine.collapse_cheapest(f64::INFINITY) else {
        panic!("a sphere has collapsible edges");
    };
    for i in 0..4 {
        for j in 0..4 {
            let expected = before[outcome.kept].q[i][j] + before[outcome.removed].q[i][j];
            assert!(
                (engine.quadric(outcome.kept).q[i][j] - expected).abs() < 1e-12,
                "Q[keep] must equal Q[keep] + Q[removed]"
            );
        }
    }
    assert_eq!(engine.live_triangles(), mesh.triangle_count() - 2);
}

#[test]
fn cheapest_collapse_matches_an_exhaustive_search() {
    let mesh = icosphere(2);
    let quadrics = compute_vertex_quadrics(&mesh);
    let mut best = (f64::INFINITY, usize::MAX, usize::MAX);
    let mut edges: HashSet<(usize, usize)> = HashSet::new();
    for &[a, b, c] in &mesh.triangles {
        for (p, q) in [(a, b), (b, c), (c, a)] {
            edges.insert((p.min(q), p.max(q)));
        }
    }
    let mut sorted: Vec<(usize, usize)> = edges.into_iter().collect();
    sorted.sort_unstable();
    for (a, b) in sorted {
        let (cost, _) = edge_collapse_cost(
            mesh.vertices[a],
            mesh.vertices[b],
            &quadrics[a],
            &quadrics[b],
        );
        if cost < best.0 {
            best = (cost, a, b);
        }
    }
    let Some(mut engine) = QemEngine::new(&mesh, CollapsePolicy::Unrestricted) else {
        panic!("valid mesh");
    };
    let Some(outcome) = engine.collapse_cheapest(f64::INFINITY) else {
        panic!("a sphere has collapsible edges");
    };
    assert_eq!((outcome.kept, outcome.removed), (best.1, best.2));
}

#[test]
fn decimation_is_deterministic_in_the_presence_of_ties() {
    // The cube and the icosphere are full of exactly tied edge costs.
    for mesh in [unit_cube(), icosphere(2)] {
        let target = mesh.triangle_count() / 3;
        let run = || {
            let mut pm = ProgressiveMesh::new(mesh.clone());
            pm.decimate_to(target);
            pm
        };
        let reference = run();
        for _ in 0..12 {
            let again = run();
            assert_eq!(reference.mesh.triangles, again.mesh.triangles);
            let bits = |m: &SimpleMesh| -> Vec<[u64; 3]> {
                m.vertices
                    .iter()
                    .map(|p| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()])
                    .collect()
            };
            assert_eq!(bits(&reference.mesh), bits(&again.mesh));
            assert_eq!(reference.history.len(), again.history.len());
            for (a, b) in reference.history.iter().zip(again.history.iter()) {
                assert_eq!((a.v_kept, a.v_removed), (b.v_kept, b.v_removed));
            }
        }
    }
}

#[test]
fn collapse_one_records_the_collapse_it_performs() {
    let mesh = icosphere(1);
    let original = mesh.clone();
    let mut pm = ProgressiveMesh::new(mesh);
    assert!(pm.collapse_one());
    assert_eq!(pm.history.len(), 1);
    let rec = pm.history[0].clone();
    assert_eq!(pm.mesh.triangle_count(), original.triangle_count() - 2);
    assert_eq!(rec.old_pos, original.vertices[rec.v_kept]);
    assert_eq!(rec.new_pos, pm.mesh.vertices[rec.v_kept]);
    assert!(
        !referenced_vertices(&pm.mesh).contains(&rec.v_removed),
        "the removed vertex must no longer be used"
    );
    assert_valid(&pm.mesh);
}

#[test]
fn collapse_one_is_the_first_step_of_decimate_to() {
    let mesh = icosphere(2);
    let mut stepwise = ProgressiveMesh::new(mesh.clone());
    for _ in 0..5 {
        assert!(stepwise.collapse_one());
    }
    let mut bulk = ProgressiveMesh::new(mesh.clone());
    bulk.decimate_to(mesh.triangle_count() - 10);
    assert_eq!(stepwise.mesh.triangles.len(), bulk.mesh.triangles.len());
    assert_eq!(stepwise.history.len(), 5);
    assert_eq!(bulk.history.len(), 5);
    // The very first collapse is identical.
    assert_eq!(
        (stepwise.history[0].v_kept, stepwise.history[0].v_removed),
        (bulk.history[0].v_kept, bulk.history[0].v_removed)
    );
}

#[test]
fn decimation_stops_at_three_triangles_and_ignores_invalid_meshes() {
    let mut pm = ProgressiveMesh::new(unit_cube());
    pm.decimate_to(0);
    assert!(pm.mesh.triangle_count() <= 4 && pm.mesh.triangle_count() >= 2);

    let mut bad = SimpleMesh::new();
    bad.add_vertex([0.0; 3]);
    bad.add_vertex([1.0, 0.0, 0.0]);
    bad.add_vertex([0.0, 1.0, 0.0]);
    for _ in 0..5 {
        bad.add_triangle(0, 1, 2);
    }
    bad.add_triangle(0, 1, 9);
    let snapshot = bad.triangles.clone();
    let mut pm = ProgressiveMesh::new(bad.clone());
    assert!(!pm.collapse_one());
    pm.decimate_to(1);
    assert_eq!(pm.mesh.triangles, snapshot);
    let mut mesh = bad;
    simplify_with_features(&mut mesh, &[], 1);
    assert_eq!(mesh.triangles, snapshot);
    assert_eq!(EdgePriorityQueue::from_mesh(&mesh).len(), 0);
}

#[test]
fn simplify_with_features_never_removes_flagged_vertices() {
    let mut mesh = icosphere(3);
    assert_eq!(mesh.triangle_count(), 1280);
    let flagged: Vec<bool> = mesh.vertices.iter().map(|p| p[2] > 0.6).collect();
    let n_flagged = flagged.iter().filter(|&&f| f).count();
    assert!(n_flagged > 10);
    simplify_with_features(&mut mesh, &flagged, 400);
    assert_valid(&mesh);
    assert!(mesh.triangle_count() <= 400);
    let used = referenced_vertices(&mesh);
    for (v, &f) in flagged.iter().enumerate() {
        if f {
            assert!(used.contains(&v), "flagged vertex {v} was removed");
        }
    }
}

#[test]
fn simplify_with_features_cannot_collapse_between_flagged_vertices() {
    let mut mesh = icosphere(1);
    let all = vec![true; mesh.vertex_count()];
    let before = mesh.triangles.clone();
    simplify_with_features(&mut mesh, &all, 0);
    assert_eq!(
        mesh.triangles, before,
        "every edge joins two flagged vertices"
    );

    // A short flag slice leaves the remaining vertices unflagged.
    let mut mesh = icosphere(1);
    simplify_with_features(&mut mesh, &[true, true], 20);
    assert!(mesh.triangle_count() <= 20);
    assert_valid(&mesh);
}

#[test]
fn simplify_with_features_is_deterministic() {
    let base = icosphere(2);
    let flags: Vec<bool> = base.vertices.iter().map(|p| p[0] > 0.8).collect();
    let run = || {
        let mut m = base.clone();
        simplify_with_features(&mut m, &flags, 120);
        m
    };
    let reference = run();
    for _ in 0..8 {
        let again = run();
        assert_eq!(reference.triangles, again.triangles);
        assert_eq!(reference.vertices, again.vertices);
    }
}

#[test]
fn preserve_boundary_never_collapses_a_boundary_edge() {
    let mesh = flat_grid(8);
    let Some(mut engine) = QemEngine::new(&mesh, CollapsePolicy::SkipBoundaryEdges) else {
        panic!("valid mesh");
    };
    let mut collapses = 0;
    loop {
        let boundary: HashSet<(usize, usize)> = {
            let mut snapshot = mesh.clone();
            engine.write_back(&mut snapshot);
            find_boundary_edges(&snapshot).into_iter().collect()
        };
        let Some(outcome) = engine.collapse_cheapest(f64::INFINITY) else {
            break;
        };
        let key = (
            outcome.kept.min(outcome.removed),
            outcome.kept.max(outcome.removed),
        );
        assert!(!boundary.contains(&key), "boundary edge {key:?} collapsed");
        collapses += 1;
    }
    assert!(collapses > 0, "the flat interior must simplify");
}

#[test]
fn preserve_boundary_simplifies_a_flat_patch_without_corrupting_it() {
    let mut qd = QemDecimation::new(flat_grid(6));
    let before = qd.mesh.triangle_count();
    let n = qd.preserve_boundary(1e-9);
    assert!(n > 0, "flat interior edges cost nothing");
    assert!(qd.mesh.triangle_count() < before);
    assert_valid(&qd.mesh);
    for v in &qd.mesh.vertices {
        assert!(v[2].abs() < 1e-9, "vertex left the plane: {v:?}");
    }
    assert_eq!(qd.preserve_boundary(-1.0), 0);
    assert_eq!(qd.preserve_boundary(f64::NAN), 0);
}

#[test]
fn edge_priority_queue_orders_edges_by_qem_cost() {
    // Two unit quads folded along a 90 degree crease.
    let mut m = SimpleMesh::new();
    for p in [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ] {
        m.add_vertex(p);
    }
    m.add_triangle(0, 1, 2);
    m.add_triangle(0, 2, 3);
    m.add_triangle(0, 1, 4);
    m.add_triangle(1, 5, 4);
    let quadrics = compute_vertex_quadrics(&m);
    let mut expected: Vec<f64> = Vec::new();
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    for &[a, b, c] in &m.triangles {
        for (p, q) in [(a, b), (b, c), (a, c)] {
            let key = (p.min(q), p.max(q));
            if seen.insert(key) {
                let (cost, _) = edge_collapse_cost(
                    m.vertices[key.0],
                    m.vertices[key.1],
                    &quadrics[key.0],
                    &quadrics[key.1],
                );
                expected.push(cost);
            }
        }
    }
    expected.sort_by(f64::total_cmp);
    let mut q = EdgePriorityQueue::from_mesh(&m);
    assert_eq!(q.len(), expected.len());
    let mut popped = Vec::new();
    while let Some(e) = q.pop() {
        popped.push(e.cost);
    }
    for (got, want) in popped.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
    assert!(
        popped.windows(2).all(|w| w[0] <= w[1]),
        "edges must pop cheapest first"
    );
}

#[test]
fn mesh_edge_ties_pop_in_vertex_pair_order() {
    let mut q = EdgePriorityQueue::new();
    for &(v0, v1) in &[(2, 3), (0, 2), (1, 2), (0, 1), (1, 3)] {
        q.push(MeshEdge { v0, v1, cost: 0.5 });
    }
    q.push(MeshEdge {
        v0: 9,
        v1: 9,
        cost: 0.25,
    });
    let order: Vec<(usize, usize)> = std::iter::from_fn(|| q.pop().map(|e| (e.v0, e.v1))).collect();
    assert_eq!(order, vec![(9, 9), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]);
}

/// Deterministic pseudo-random value in `[0, 1)` for the counter `x` (SplitMix64).
fn hash01(x: u64) -> f64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1_u64 << 53) as f64
}

/// A sphere with a random radial bump on every vertex, so that edge costs are
/// pairwise different.
fn bumpy_sphere(subdivisions: usize, amplitude: f64) -> SimpleMesh {
    let mut mesh = icosphere(subdivisions);
    for (i, p) in mesh.vertices.iter_mut().enumerate() {
        let scale = 1.0 + amplitude * (hash01(i as u64) - 0.5);
        *p = [p[0] * scale, p[1] * scale, p[2] * scale];
    }
    mesh
}

/// Which end point of the edge `(lo, hi)` survives, or `None` if it may not
/// collapse; given the current triangle list.
type EndpointRule<'a> = &'a dyn Fn(usize, usize, &[[usize; 3]]) -> Option<(usize, usize)>;

/// Cheapest admissible collapse found by rescanning every edge of the current
/// triangle list: `(cost, kept, removed, position)`.
type NaiveCollapse = (f64, usize, usize, [f64; 3]);

/// The textbook Garland–Heckbert algorithm without any acceleration structure: every
/// step rescans all edges of the current mesh (`O(F)` per collapse) and merges
/// the quadrics with `Q[keep] += Q[removed]`.  It is the reference the
/// incremental engine must reproduce collapse for collapse.
struct NaiveQem {
    positions: Vec<[f64; 3]>,
    quadrics: Vec<QuadricMatrix>,
    triangles: Vec<[usize; 3]>,
}

impl NaiveQem {
    fn new(mesh: &SimpleMesh) -> Self {
        Self {
            positions: mesh.vertices.clone(),
            quadrics: compute_vertex_quadrics(mesh),
            triangles: mesh.triangles.clone(),
        }
    }

    fn cheapest(&self, rule: EndpointRule<'_>) -> Option<NaiveCollapse> {
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for &[a, b, c] in &self.triangles {
            for (p, q) in [(a, b), (b, c), (c, a)] {
                edges.push((p.min(q), p.max(q)));
            }
        }
        edges.sort_unstable();
        edges.dedup();
        let mut best: Option<NaiveCollapse> = None;
        for (lo, hi) in edges {
            let Some((kept, removed)) = rule(lo, hi, &self.triangles) else {
                continue;
            };
            let (cost, pos) = edge_collapse_cost(
                self.positions[kept],
                self.positions[removed],
                &self.quadrics[kept],
                &self.quadrics[removed],
            );
            if !cost.is_finite() {
                continue;
            }
            // The edges are visited in (lo, hi) order, so a strict comparison keeps
            // the smallest vertex pair among equal costs.
            if best.is_none_or(|b| cost.total_cmp(&b.0).is_lt()) {
                best = Some((cost, kept, removed, pos));
            }
        }
        best
    }

    fn collapse(&mut self, kept: usize, removed: usize, pos: [f64; 3]) {
        self.positions[kept] = pos;
        let merged = std::mem::replace(&mut self.quadrics[removed], QuadricMatrix::zero());
        self.quadrics[kept] = self.quadrics[kept].add(&merged);
        for tri in &mut self.triangles {
            for v in tri.iter_mut() {
                if *v == removed {
                    *v = kept;
                }
            }
        }
        self.triangles
            .retain(|&[a, b, c]| a != b && b != c && a != c);
    }
}

/// Runs the engine and the naive reference side by side until `target`
/// triangles are left (or no edge may collapse) and compares every step.
fn assert_engine_matches_reference(
    mesh: &SimpleMesh,
    policy: CollapsePolicy,
    rule: EndpointRule<'_>,
    target: usize,
) -> usize {
    let Some(mut engine) = QemEngine::new(mesh, policy) else {
        panic!("valid mesh");
    };
    let mut reference = NaiveQem::new(mesh);
    let mut steps = 0;
    while engine.live_triangles() > target {
        match (
            reference.cheapest(rule),
            engine.collapse_cheapest(f64::INFINITY),
        ) {
            (Some((_, kept, removed, pos)), Some(outcome)) => {
                assert_eq!(
                    (outcome.kept, outcome.removed),
                    (kept, removed),
                    "collapse {steps} picked a different edge"
                );
                assert_eq!(outcome.new_pos, pos, "collapse {steps}: position");
                reference.collapse(kept, removed, pos);
                assert_eq!(
                    engine.live_triangles(),
                    reference.triangles.len(),
                    "collapse {steps}: triangle count"
                );
            }
            (None, None) => break,
            (expected, got) => panic!("collapse {steps}: reference {expected:?}, engine {got:?}"),
        }
        steps += 1;
    }
    let mut result = mesh.clone();
    engine.write_back(&mut result);
    assert_eq!(result.triangles, reference.triangles, "final triangle list");
    steps
}

fn edge_has_one_face(lo: usize, hi: usize, triangles: &[[usize; 3]]) -> bool {
    triangles
        .iter()
        .filter(|tri| tri.contains(&lo) && tri.contains(&hi))
        .count()
        == 1
}

#[test]
fn incremental_engine_reproduces_the_naive_reference_collapse_for_collapse() {
    let sphere = bumpy_sphere(2, 0.3);
    let unrestricted = |lo: usize, hi: usize, _: &[[usize; 3]]| Some((lo, hi));
    let steps =
        assert_engine_matches_reference(&sphere, CollapsePolicy::Unrestricted, &unrestricted, 40);
    // 320 triangles down to 40 triangles: two triangles vanish per collapse.
    assert!(steps >= 100, "{steps} collapses");

    let flags: Vec<bool> = sphere.vertices.iter().map(|p| p[2] > 0.5).collect();
    let flagged = |lo: usize, hi: usize, _: &[[usize; 3]]| match (flags[lo], flags[hi]) {
        (true, true) => None,
        (_, false) => Some((lo, hi)),
        (false, true) => Some((hi, lo)),
    };
    let steps = assert_engine_matches_reference(
        &sphere,
        CollapsePolicy::KeepFlagged(flags.clone()),
        &flagged,
        60,
    );
    assert!(steps > 50, "{steps} collapses");

    // A bumpy open patch: boundary edges never collapse, so the run ends when
    // only boundary edges are left.
    let mut patch = flat_grid(7);
    for (i, p) in patch.vertices.iter_mut().enumerate() {
        p[2] = 0.3 * (hash01(1000 + i as u64) - 0.5);
    }
    let interior = |lo: usize, hi: usize, triangles: &[[usize; 3]]| {
        if edge_has_one_face(lo, hi, triangles) {
            None
        } else {
            Some((lo, hi))
        }
    };
    let steps =
        assert_engine_matches_reference(&patch, CollapsePolicy::SkipBoundaryEdges, &interior, 0);
    assert!(steps > 10, "{steps} collapses");
}

#[test]
fn engine_honours_the_cost_limit_in_cheapest_first_order() {
    let sphere = bumpy_sphere(2, 0.3);
    let unrestricted = |lo: usize, hi: usize, _: &[[usize; 3]]| Some((lo, hi));
    // Costs of the first collapses of the reference run and the triangle count
    // left after each of them.
    let mut reference = NaiveQem::new(&sphere);
    let mut costs = Vec::new();
    let mut remaining = Vec::new();
    for _ in 0..40 {
        let Some((cost, kept, removed, pos)) = reference.cheapest(&unrestricted) else {
            break;
        };
        costs.push(cost);
        reference.collapse(kept, removed, pos);
        remaining.push(reference.triangles.len());
    }
    assert!(costs.len() == 40 && costs[9] < costs[39]);
    // The engine follows the reference sequence, so a limit equal to the 10th
    // cost admits exactly the leading collapses that do not exceed it.
    let limit = costs[9];
    let admitted = costs.iter().take_while(|&&c| c <= limit).count();
    assert!(
        (10..40).contains(&admitted),
        "{admitted} admitted collapses"
    );

    let Some(mut engine) = QemEngine::new(&sphere, CollapsePolicy::Unrestricted) else {
        panic!("valid mesh");
    };
    let mut performed = 0;
    while engine.collapse_cheapest(limit).is_some() {
        performed += 1;
    }
    assert_eq!(performed, admitted);
    assert_eq!(engine.live_triangles(), remaining[admitted - 1]);
    // A NaN or negative-infinite limit admits nothing; an infinite one everything.
    assert!(engine.collapse_cheapest(f64::NAN).is_none());
    assert!(engine.collapse_cheapest(f64::NEG_INFINITY).is_none());
    assert!(engine.collapse_cheapest(f64::INFINITY).is_some());
}
