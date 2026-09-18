// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the lipid-bilayer geometry: icosphere construction,
//! discrete curvatures computed from the mesh, geometric Helfrich/ADE energies
//! and the vesicle step.

use std::collections::BTreeMap;

use super::*;

const KAPPA: f64 = 2.0e-19;

/// Icosphere with the given spontaneous curvature and Gaussian modulus.
fn sphere(radius: f64, subdivisions: usize, c0: f64, kappa_g: f64) -> LipidBilayer {
    LipidBilayer::new_icosphere(radius, subdivisions, KAPPA, kappa_g, c0, 0.24, 4.0e-9)
}

fn positions_of(bilayer: &LipidBilayer) -> Vec<Vec3> {
    bilayer.vertices.iter().map(|v| v.position).collect()
}

fn mean_edge_length(bilayer: &LipidBilayer) -> f64 {
    let mut total = 0.0;
    let mut count = 0.0;
    for tri in &bilayer.triangles {
        for k in 0..3 {
            let a = bilayer.vertices[tri[k]].position;
            let b = bilayer.vertices[tri[(k + 1) % 3]].position;
            total += norm3(sub3(a, b));
            count += 1.0;
        }
    }
    total / count
}

/// Open cylinder of the given radius with outward normals (rows of `around` vertices).
fn open_cylinder(radius: f64, around: usize, rows: usize, length: f64) -> LipidBilayer {
    let mut vertices = Vec::new();
    for j in 0..=rows {
        for i in 0..around {
            let theta = 2.0 * PI * i as f64 / around as f64;
            vertices.push(MembraneVertex::new([
                radius * theta.cos(),
                radius * theta.sin(),
                length * j as f64 / rows as f64,
            ]));
        }
    }
    let mut triangles = Vec::new();
    for j in 0..rows {
        for i in 0..around {
            let a = j * around + i;
            let b = j * around + (i + 1) % around;
            let c = (j + 1) * around + i;
            let d = (j + 1) * around + (i + 1) % around;
            triangles.push([a, b, d]);
            triangles.push([a, d, c]);
        }
    }
    LipidBilayer {
        kappa: KAPPA,
        kappa_gaussian: 0.0,
        c0: 0.0,
        k_area: 0.24,
        area0: 0.0,
        ade_alpha: 1.0,
        thickness: 4.0e-9,
        vertices,
        triangles,
    }
}

/// Flat n x n grid with deterministic interior jitter (contains obtuse triangles).
fn jittered_flat_grid(n: usize, jitter: f64) -> (Vec<Vec3>, Vec<[usize; 3]>) {
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as f64 / f64::from(1_u32 << 31) - 0.5
    };
    let mut positions = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let interior = i > 0 && j > 0 && i + 1 < n && j + 1 < n;
            let (dx, dy) = if interior {
                (jitter * next(), jitter * next())
            } else {
                (0.0, 0.0)
            };
            positions.push([i as f64 + dx, j as f64 + dy, 0.0]);
        }
    }
    let mut triangles = Vec::new();
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let a = j * n + i;
            triangles.push([a, a + 1, a + n + 1]);
            triangles.push([a, a + n + 1, a + n]);
        }
    }
    (positions, triangles)
}

fn circumcenter(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> [f64; 2] {
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    let (a2, b2, c2) = (
        a[0] * a[0] + a[1] * a[1],
        b[0] * b[0] + b[1] * b[1],
        c[0] * c[0] + c[1] * c[1],
    );
    [
        (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d,
        (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d,
    ]
}

fn polygon_area(points: &[[f64; 2]]) -> f64 {
    let mut twice = 0.0;
    for k in 0..points.len() {
        let (p, q) = (points[k], points[(k + 1) % points.len()]);
        twice += p[0] * q[1] - q[0] * p[1];
    }
    0.5 * twice.abs()
}

// ---------------------------------------------------------------------------
// Icosphere construction (F4187)
// ---------------------------------------------------------------------------

#[test]
fn icosphere_has_subdivision_counts_and_euler_characteristic_two() {
    for subdivisions in 0..=3_u32 {
        let bilayer = sphere(1.0, subdivisions as usize, 0.0, 0.0);
        let faces = 20 * 4_usize.pow(subdivisions);
        let vertices = 10 * 4_usize.pow(subdivisions) + 2;
        assert_eq!(bilayer.triangles.len(), faces);
        assert_eq!(bilayer.vertices.len(), vertices);
        let mut uses: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for tri in &bilayer.triangles {
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        assert!(
            uses.values().all(|&count| count == 2),
            "closed manifold: every edge is shared by exactly two triangles"
        );
        assert_eq!(uses.len(), 30 * 4_usize.pow(subdivisions));
        assert_eq!(vertices as isize - uses.len() as isize + faces as isize, 2);
    }
}

#[test]
fn icosphere_vertices_lie_on_the_sphere_and_faces_point_outward() {
    for &radius in &[1.0, 5.0e-6, 5.0e-8] {
        let bilayer = sphere(radius, 2, 0.0, 0.0);
        for vertex in &bilayer.vertices {
            assert!((norm3(vertex.position) / radius - 1.0).abs() < 1e-12);
        }
        for tri in &bilayer.triangles {
            let p = [
                bilayer.vertices[tri[0]].position,
                bilayer.vertices[tri[1]].position,
                bilayer.vertices[tri[2]].position,
            ];
            let centroid = scale3(vec3_sum(&p), 1.0 / 3.0);
            assert!(
                dot3(triangle_normal(p[0], p[1], p[2]), centroid) > 0.0,
                "triangle {tri:?} is wound inwards"
            );
        }
        assert!(bilayer.signed_volume() > 0.0);
    }
}

fn vec3_sum(points: &[Vec3]) -> Vec3 {
    points.iter().fold([0.0; 3], |acc, p| {
        [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]]
    })
}

#[test]
fn icosphere_construction_is_deterministic() {
    let first = sphere(1.0e-6, 3, 0.0, 0.0);
    let second = sphere(1.0e-6, 3, 0.0, 0.0);
    assert_eq!(first.triangles, second.triangles);
    for (a, b) in first.vertices.iter().zip(&second.vertices) {
        assert_eq!(a.position, b.position, "bitwise identical vertices");
    }
}

#[test]
fn icosphere_area_and_volume_approach_the_sphere_from_below() {
    let radius = 1.0;
    let mut previous = (0.0, 0.0);
    for subdivisions in 0..=4 {
        let bilayer = sphere(radius, subdivisions, 0.0, 0.0);
        let area_ratio = bilayer.total_area() / sphere_area(radius);
        let volume_ratio = bilayer.signed_volume() / sphere_volume(radius);
        assert!(
            area_ratio < 1.0 && volume_ratio < 1.0,
            "inscribed polyhedron"
        );
        assert!(area_ratio > previous.0 && volume_ratio > previous.1);
        previous = (area_ratio, volume_ratio);
        assert_eq!(
            bilayer.area0,
            bilayer.total_area(),
            "area0 is the mesh area"
        );
        if subdivisions == 3 {
            assert!(area_ratio > 0.99 && volume_ratio > 0.99);
        }
    }
}

#[test]
fn new_sphere_is_a_closed_nearly_spherical_vesicle() {
    let radius = 5.0e-6;
    let vesicle = VesicleSimulation::new_sphere(radius, KAPPA, 0.24, 1.0e10);
    let reduced = vesicle.reduced_volume();
    assert!(
        (0.98..=1.0).contains(&reduced),
        "reduced volume of a sphere: {reduced}"
    );
    assert_eq!(vesicle.shape_class(), "sphere");
    let analytic = sphere_volume(radius);
    assert!(
        (vesicle.enclosed_volume() - analytic).abs() < 0.03 * analytic,
        "volume {} vs analytic {analytic}",
        vesicle.enclosed_volume()
    );
    assert_eq!(vesicle.volume0, vesicle.enclosed_volume());
    assert_eq!(vesicle.volume_energy(), 0.0);
    assert_eq!(vesicle.bilayer.area_elastic_energy(), 0.0);
    assert!(vesicle.total_energy() > 0.0, "bending energy of a sphere");
}

#[test]
fn flat_patch_encloses_no_volume() {
    // An open sheet is not a vesicle: its divergence-theorem volume vanishes.
    let patch = LipidBilayer::new_circular_patch(1.0e-6, 4, KAPPA, 0.0, 0.0, 0.24, 4.0e-9);
    assert!(patch.signed_volume().abs() < 1e-30);
}

// ---------------------------------------------------------------------------
// Curvatures from geometry (F4183)
// ---------------------------------------------------------------------------

#[test]
fn sphere_curvatures_match_inverse_radius_and_inverse_radius_squared() {
    for &radius in &[1.0, 5.0e-6, 5.0e-8] {
        let bilayer = sphere(radius, 2, 0.0, 0.0);
        for vertex in &bilayer.vertices {
            let h = vertex.mean_curvature * radius;
            let k = vertex.gaussian_curvature * radius * radius;
            assert!((h - 1.0).abs() < 0.05, "H R = {h} at radius {radius}");
            assert!((k - 1.0).abs() < 0.05, "K R^2 = {k} at radius {radius}");
            assert!((norm3(vertex.normal) - 1.0).abs() < 1e-12, "unit normal");
            let outward = dot3(vertex.normal, scale3(vertex.position, 1.0 / radius));
            assert!(outward > 0.999, "outward normal, cos = {outward}");
        }
    }
}

#[test]
fn finer_icosphere_curvatures_are_more_accurate() {
    let radius = 1.0;
    let worst_k_error = |subdivisions: usize| {
        let bilayer = sphere(radius, subdivisions, 0.0, 0.0);
        bilayer
            .vertices
            .iter()
            .map(|v| (v.gaussian_curvature - 1.0).abs())
            .fold(0.0_f64, f64::max)
    };
    let (coarse, medium, fine) = (worst_k_error(1), worst_k_error(2), worst_k_error(3));
    assert!(coarse > medium && medium > fine);
    assert!(fine < 0.01, "K error at level 3: {fine}");
}

#[test]
fn gauss_bonnet_and_area_partition_hold_on_closed_meshes() {
    for subdivisions in 1..=3 {
        let bilayer = sphere(2.0e-6, subdivisions, 0.0, 0.0);
        let integrated_k: f64 = bilayer
            .vertices
            .iter()
            .map(|v| v.gaussian_curvature * v.area)
            .sum();
        assert!(
            (integrated_k - 4.0 * PI).abs() < 1e-10 * 4.0 * PI,
            "sum K A = {integrated_k}"
        );
        let area_sum: f64 = bilayer.vertices.iter().map(|v| v.area).sum();
        assert!((area_sum - bilayer.total_area()).abs() < 1e-12 * bilayer.total_area());
    }
}

#[test]
fn sphere_bending_energy_is_eight_pi_kappa_plus_gaussian_term() {
    let kappa_g = -0.5 * KAPPA;
    let bilayer = sphere(1.0e-6, 3, 0.0, kappa_g);
    let expected = 8.0 * PI * KAPPA + 4.0 * PI * kappa_g;
    let energy = bilayer.bending_energy();
    assert!(
        (energy - expected).abs() < 0.02 * expected,
        "E = {energy}, expected {expected}"
    );
}

#[test]
fn spontaneous_curvature_follows_the_outward_normal_convention() {
    let radius = 1.0e-6;
    let reference = 8.0 * PI * KAPPA;
    // c0 = +2/R equals 2H of a sphere with outward normals: (2H - c0)^2 ~ 0.
    let matched = sphere(radius, 3, 2.0 / radius, 0.0).bending_energy();
    assert!(matched < 0.01 * reference, "matched c0 energy {matched}");
    // c0 = -2/R: (2H - c0)^2 = (4/R)^2, E = 2 kappa A / R^2 * 4 ~ 32 pi kappa.
    let opposed = sphere(radius, 3, -2.0 / radius, 0.0).bending_energy();
    let expected = 4.0 * reference;
    assert!(
        (opposed - expected).abs() < 0.03 * expected,
        "opposed c0 energy {opposed} vs {expected}"
    );
}

#[test]
fn flat_patch_has_no_curvature_and_no_energy_even_with_gaussian_modulus() {
    let side = 1.0e-6;
    let patch = LipidBilayer::new_circular_patch(side, 5, KAPPA, -0.5 * KAPPA, 0.0, 0.24, 4.0e-9);
    for vertex in &patch.vertices {
        assert!(
            vertex.mean_curvature.abs() * side < 1e-9,
            "H = {}",
            vertex.mean_curvature
        );
        assert!(
            vertex.gaussian_curvature.abs() * side * side < 1e-9,
            "K = {}",
            vertex.gaussian_curvature
        );
    }
    assert!(patch.bending_energy().abs() < 1e-30);
    assert!(patch.ade_energy().abs() < 1e-30);
    let area_sum: f64 = patch.vertices.iter().map(|v| v.area).sum();
    assert!((area_sum - 4.0 * side * side).abs() < 1e-12 * 4.0 * side * side);
}

#[test]
fn open_cylinder_has_half_inverse_radius_mean_curvature_and_zero_gaussian() {
    let radius = 3.0e-6;
    let mut cylinder = open_cylinder(radius, 32, 6, 4.0e-6);
    cylinder.update_curvatures();
    for vertex in &cylinder.vertices {
        // H = (1/R + 0)/2 on interior and free-edge rows alike (normal projection).
        assert!(
            (vertex.mean_curvature * radius - 0.5).abs() < 0.005,
            "H R = {}",
            vertex.mean_curvature * radius
        );
        assert!(
            vertex.gaussian_curvature.abs() * radius * radius < 1e-9,
            "K R^2 = {}",
            vertex.gaussian_curvature * radius * radius
        );
    }
    // Energy of a cylinder: (kappa/2) (2H)^2 A = kappa A / (2 R^2).
    let expected = KAPPA * cylinder.total_area() / (2.0 * radius * radius);
    assert!((cylinder.bending_energy() - expected).abs() < 0.01 * expected);
}

#[test]
fn rolled_flat_patch_stores_cylinder_bending_energy() {
    let flat = LipidBilayer::new_circular_patch(1.0e-6, 15, KAPPA, 0.0, 0.0, 0.24, 4.0e-9);
    let rolled = |radius: f64| {
        let mut bilayer = flat.clone();
        for vertex in &mut bilayer.vertices {
            let angle = vertex.position[0] / radius;
            vertex.position = [
                radius * angle.sin(),
                vertex.position[1],
                radius * (1.0 - angle.cos()),
            ];
        }
        bilayer
    };
    let (loose, tight) = (rolled(2.0e-6), rolled(1.0e-6));
    for (bilayer, radius) in [(&loose, 2.0e-6), (&tight, 1.0e-6)] {
        let expected = KAPPA * bilayer.total_area() / (2.0 * radius * radius);
        let energy = bilayer.bending_energy();
        assert!(
            (energy - expected).abs() < 0.05 * expected,
            "radius {radius}: {energy} vs {expected}"
        );
    }
    assert!(tight.bending_energy() > 3.0 * loose.bending_energy());
}

#[test]
fn bending_energy_ignores_stored_curvature_values() {
    let mut bilayer = sphere(1.0e-6, 2, 0.0, -0.5 * KAPPA);
    let bending = bilayer.bending_energy();
    let ade = bilayer.ade_energy();
    bilayer.assign_curvatures(1.0e9, 1.0e18);
    assert_eq!(bilayer.bending_energy(), bending);
    assert_eq!(bilayer.ade_energy(), ade);
    bilayer.update_curvatures();
    assert!((bilayer.vertices[7].mean_curvature * 1.0e-6 - 1.0).abs() < 0.01);
    assert_eq!(bilayer.bending_energy(), bending);
}

#[test]
fn bending_energy_follows_the_deformed_geometry() {
    let round = sphere(1.0e-6, 3, 0.0, 0.0);
    // Willmore energy is scale invariant ...
    let mut small = round.clone();
    for vertex in &mut small.vertices {
        vertex.position = scale3(vertex.position, 0.5);
    }
    assert!((small.bending_energy() / round.bending_energy() - 1.0).abs() < 1e-9);
    // ... but grows when the sphere is stretched into a prolate spheroid
    // (analytic Willmore energy ratio for an axis ratio 1.5 is 1.0806).
    let mut prolate = round.clone();
    for vertex in &mut prolate.vertices {
        vertex.position[2] *= 1.5;
    }
    let ratio = prolate.bending_energy() / round.bending_energy();
    assert!((ratio - 1.0806).abs() < 0.02, "energy ratio {ratio}");
}

#[test]
fn ade_energy_of_a_sphere_is_size_independent() {
    let small = sphere(5.0e-8, 3, 0.0, 0.0);
    let large = sphere(5.0e-6, 3, 0.0, 0.0);
    let (e_small, e_large) = (small.ade_energy(), large.ade_energy());
    assert!((e_small / e_large - 1.0).abs() < 1e-9);
    let expected = 8.0 * PI * PI * KAPPA; // alpha = 1
    assert!(
        (e_large - expected).abs() < 0.02 * expected,
        "E_ADE = {e_large}, expected {expected}"
    );
    // Delta A = 8 pi R d for a sphere.
    let delta_a = large.area_difference();
    let analytic = 8.0 * PI * 5.0e-6 * 4.0e-9;
    assert!((delta_a - analytic).abs() < 0.02 * analytic);
}

#[test]
fn update_curvatures_replaces_synthetic_values_by_geometric_ones() {
    let mut bilayer = sphere(1.0, 2, 0.0, 0.0);
    bilayer.assign_curvatures(7.0, 9.0);
    assert_eq!(bilayer.vertices[3].mean_curvature, 7.0);
    bilayer.update_curvatures();
    for vertex in &bilayer.vertices {
        assert!((vertex.mean_curvature - 1.0).abs() < 0.05);
        assert!((vertex.gaussian_curvature - 1.0).abs() < 0.05);
    }
}

#[test]
fn normals_are_unit_at_nanometre_scale() {
    // Face areas of a 50 nm vesicle are ~1e-17 m^2, far below any absolute threshold.
    let mut bilayer = sphere(5.0e-8, 2, 0.0, 0.0);
    for vertex in &mut bilayer.vertices {
        vertex.normal = [0.0; 3];
    }
    bilayer.update_normals();
    for vertex in &bilayer.vertices {
        assert!((norm3(vertex.normal) - 1.0).abs() < 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Cotangent operators (F4188)
// ---------------------------------------------------------------------------

#[test]
fn mixed_voronoi_areas_partition_acute_and_obtuse_triangles() {
    // Acute triangle: compare with the circumcentric Voronoi cells.
    let acute = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.4, 0.8, 0.0]];
    let operators = SurfaceOperators::new(&acute, &[[0, 1, 2]]);
    let total: f64 = operators.areas.iter().sum();
    assert!((total - 0.4).abs() < 1e-12, "areas sum to {total}");
    let planar = acute.map(|p| [p[0], p[1]]);
    let centre = circumcenter(planar[0], planar[1], planar[2]);
    for k in 0..3 {
        let (next, prev) = (planar[(k + 1) % 3], planar[(k + 2) % 3]);
        let mid = |q: [f64; 2]| [0.5 * (planar[k][0] + q[0]), 0.5 * (planar[k][1] + q[1])];
        let cell = polygon_area(&[planar[k], mid(next), centre, mid(prev)]);
        assert!(
            (operators.areas[k] - cell).abs() < 1e-12,
            "vertex {k}: {} vs Voronoi cell {cell}",
            operators.areas[k]
        );
    }
    // Obtuse triangle: the obtuse corner gets half of the area, the others a quarter.
    let obtuse = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [1.0, 0.5, 0.0]];
    let operators = SurfaceOperators::new(&obtuse, &[[0, 1, 2]]);
    assert!((operators.areas[2] - 0.5).abs() < 1e-12);
    assert!((operators.areas[0] - 0.25).abs() < 1e-12);
    assert!((operators.areas[1] - 0.25).abs() < 1e-12);
}

#[test]
fn cotangent_laplacian_reproduces_linear_functions_on_an_irregular_flat_mesh() {
    let n = 9;
    // Jitter below 0.3 keeps every triangle positively oriented (no folds).
    let (positions, triangles) = jittered_flat_grid(n, 0.3);
    for tri in &triangles {
        let (a, b, c) = (positions[tri[0]], positions[tri[1]], positions[tri[2]]);
        assert!(cross3(sub3(b, a), sub3(c, a))[2] > 0.0, "folded triangle");
    }
    let obtuse = triangles
        .iter()
        .filter(|tri| {
            (0..3).any(|k| {
                let c = positions[tri[k]];
                let u = sub3(positions[tri[(k + 1) % 3]], c);
                let w = sub3(positions[tri[(k + 2) % 3]], c);
                dot3(u, w) < 0.0
            })
        })
        .count();
    assert!(obtuse > 0, "the test mesh must contain obtuse triangles");
    let operators = SurfaceOperators::new(&positions, &triangles);
    let laplacian = operators.laplace_beltrami(&triangles, &positions);
    for j in 1..n - 1 {
        for i in 1..n - 1 {
            let size = norm3(laplacian[j * n + i]);
            assert!(size < 1e-9, "interior vertex ({i},{j}): |Delta x| = {size}");
        }
    }
    // Boundary vertices are flagged, interior ones are not.
    for j in 0..n {
        for i in 0..n {
            let edge = i == 0 || j == 0 || i == n - 1 || j == n - 1;
            assert_eq!(operators.on_boundary[j * n + i], edge);
        }
    }
}

#[test]
fn degenerate_triangles_do_not_contribute_to_the_operators() {
    let (mut positions, mut triangles) = jittered_flat_grid(4, 0.2);
    let reference = SurfaceOperators::new(&positions, &triangles);
    // A collinear (zero-area) triangle spanning existing vertices.
    positions.push([0.5, 0.0, 0.0]);
    let extra = positions.len() - 1;
    triangles.push([0, 1, extra]);
    let operators = SurfaceOperators::new(&positions, &triangles);
    for (with, without) in operators.areas.iter().zip(&reference.areas) {
        assert!((with - without).abs() < 1e-15);
    }
    let laplacian = operators.laplace_beltrami(&triangles, &positions);
    assert!(laplacian.iter().all(|l| l.iter().all(|c| c.is_finite())));
    assert_eq!(operators.areas[extra], 0.0);
    assert_eq!(laplacian[extra], [0.0; 3]);
}

// ---------------------------------------------------------------------------
// Vesicle dynamics (F4189 follow-up of the shared helper)
// ---------------------------------------------------------------------------

/// Dimensionless unit vesicle (R = 1, kappa = 1) with the given constraints.
fn unit_vesicle(k_area: f64, k_volume: f64) -> VesicleSimulation {
    let bilayer = LipidBilayer::new_icosphere(1.0, 2, 1.0, 0.0, 0.0, k_area, 0.01);
    VesicleSimulation::from_bilayer(bilayer, k_volume)
}

/// Deterministic pseudo-random radial roughness of relative amplitude `amplitude`.
fn roughen(vesicle: &mut VesicleSimulation, amplitude: f64) {
    for (i, vertex) in vesicle.bilayer.vertices.iter_mut().enumerate() {
        let wobble = ((i * 37 + 11) % 17) as f64 / 8.0 - 1.0;
        vertex.position = scale3(vertex.position, 1.0 + amplitude * wobble);
    }
    vesicle.bilayer.update_curvatures();
}

fn rms_radius_deviation(bilayer: &LipidBilayer) -> f64 {
    let radii: Vec<f64> = bilayer.vertices.iter().map(|v| norm3(v.position)).collect();
    let mean = radii.iter().sum::<f64>() / radii.len() as f64;
    (radii.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / radii.len() as f64).sqrt()
}

#[test]
fn step_refreshes_the_stored_geometry_and_advances_time() {
    let mut vesicle = unit_vesicle(0.0, 0.0);
    roughen(&mut vesicle, 0.02);
    let dt = 0.005 * mean_edge_length(&vesicle.bilayer).powi(2);
    for _ in 0..5 {
        vesicle.step(dt);
    }
    assert!((vesicle.time - 5.0 * dt).abs() < 1e-15);
    let fresh = vesicle.bilayer.curvature_field();
    for (stored, geometry) in vesicle.bilayer.vertices.iter().zip(&fresh) {
        assert_eq!(stored.mean_curvature, geometry.mean);
        assert_eq!(stored.gaussian_curvature, geometry.gaussian);
        assert_eq!(stored.area, geometry.area);
        assert_eq!(stored.normal, geometry.normal);
        assert!(stored.position.iter().all(|c| c.is_finite()));
    }
}

#[test]
fn bending_flow_smooths_a_rough_sphere_without_moving_its_centre() {
    let mut vesicle = unit_vesicle(0.0, 0.0);
    roughen(&mut vesicle, 0.03);
    let before = rms_radius_deviation(&vesicle.bilayer);
    let energy_before = vesicle.bilayer.bending_energy();
    let centre = |v: &VesicleSimulation| {
        scale3(
            vec3_sum(&positions_of(&v.bilayer)),
            1.0 / v.bilayer.vertices.len() as f64,
        )
    };
    let centre_before = centre(&vesicle);
    let dt = 0.005 * mean_edge_length(&vesicle.bilayer).powi(2);
    for _ in 0..60 {
        vesicle.step(dt);
    }
    let after = rms_radius_deviation(&vesicle.bilayer);
    assert!(
        after.is_finite() && after < 0.5 * before,
        "{before} -> {after}"
    );
    assert!(vesicle.bilayer.bending_energy() < energy_before);
    // The bending force is translation invariant: it exerts no net force.
    let shift = norm3(sub3(centre(&vesicle), centre_before));
    assert!(shift < 1e-12, "centre of the vesicle moved by {shift}");
}

#[test]
fn bending_energy_after_steps_is_still_that_of_a_sphere() {
    // Realistic parameters: the stiff area/volume penalties and the tiny bending
    // rigidity keep the icosphere round, and the refreshed curvatures give 8 pi kappa.
    let radius = 1.0e-6;
    let mut vesicle = VesicleSimulation::new_sphere(radius, KAPPA, 0.24, 1.0e5);
    let (kappa, kappa_g) = (vesicle.bilayer.kappa, vesicle.bilayer.kappa_gaussian);
    let expected = 8.0 * PI * kappa + 4.0 * PI * kappa_g;
    for _ in 0..5 {
        vesicle.step(1.0e-2);
    }
    let energy = vesicle.bilayer.bending_energy();
    assert!(
        (energy - expected).abs() < 0.05 * expected,
        "E = {energy}, expected {expected}"
    );
    for vertex in &vesicle.bilayer.vertices {
        assert!((vertex.mean_curvature * radius - 1.0).abs() < 0.05);
    }
    // Squashing the vertices anisotropically changes the energy.
    let mut squashed = vesicle.clone();
    for vertex in &mut squashed.bilayer.vertices {
        vertex.position[2] *= 0.6;
    }
    assert!((squashed.bilayer.bending_energy() / energy - 1.0).abs() > 0.02);
}

#[test]
fn bending_flow_obeys_the_documented_step_limit() {
    // step() documents stability for dt below about 0.05 h^2 / kappa.
    let run = |dt_factor: f64, steps: usize| {
        let mut vesicle = unit_vesicle(0.0, 0.0);
        roughen(&mut vesicle, 0.03);
        let h = mean_edge_length(&vesicle.bilayer);
        let before = rms_radius_deviation(&vesicle.bilayer);
        for _ in 0..steps {
            vesicle.step(dt_factor * h * h);
        }
        rms_radius_deviation(&vesicle.bilayer) / before
    };
    let stable = run(0.025, 30);
    assert!(
        stable.is_finite() && stable < 0.5,
        "below the limit: {stable}"
    );
    // Four times the limit: the roughness is amplified instead of damped.
    let unstable = run(0.2, 20);
    assert!(
        !unstable.is_finite() || unstable > 20.0,
        "above the limit the roughness must grow: {unstable}"
    );
}

#[test]
fn bending_force_pulls_a_bump_on_a_flat_sheet_back() {
    let mut patch = LipidBilayer::new_circular_patch(1.0, 8, 1.0, 0.0, 0.0, 0.0, 0.01);
    let stride = 10; // n_rings + 2 vertices per row
    let apex = 5 * stride + 5;
    patch.vertices[apex].position[2] = 0.05;
    let mut sheet = VesicleSimulation::from_bilayer(patch, 0.0);
    let dt = 0.005 * mean_edge_length(&sheet.bilayer).powi(2);
    sheet.step(dt);
    let z = sheet.bilayer.vertices[apex].position[2];
    assert!(
        z > 0.0 && z < 0.05,
        "the bump relaxes towards the plane: z = {z}"
    );
    // Neighbours are dragged up by the (translation-invariant) bending force.
    let dragged = sheet
        .bilayer
        .vertices
        .iter()
        .enumerate()
        .any(|(i, v)| i != apex && v.position[2] > 1e-6);
    assert!(dragged);
    let total_lift: f64 = sheet.bilayer.vertices.iter().map(|v| v.position[2]).sum();
    assert!((total_lift - 0.05).abs() < 1e-12, "momentum: {total_lift}");
}

#[test]
fn volume_penalty_drives_a_compressed_vesicle_back_to_its_target_volume() {
    for flip_winding in [false, true] {
        let mut bilayer = LipidBilayer::new_icosphere(1.0e-6, 2, 1.0e-30, 0.0, 0.0, 0.0, 4.0e-9);
        if flip_winding {
            for tri in &mut bilayer.triangles {
                tri.swap(1, 2);
            }
        }
        let mut vesicle = VesicleSimulation::from_bilayer(bilayer, 0.0);
        assert!(vesicle.bilayer.signed_volume().is_sign_negative() == flip_winding);
        // Penalty stiffness chosen so that dt * lambda_V |grad V|^2 = 0.1.
        let dt = 1.0e-3;
        let grad_sq: f64 = vesicle
            .bilayer
            .vertices
            .iter()
            .map(|v| v.area * v.area)
            .sum();
        vesicle.k_volume = 0.1 * vesicle.volume0 / (dt * grad_sq);
        for vertex in &mut vesicle.bilayer.vertices {
            vertex.position = scale3(vertex.position, 0.9);
        }
        let mut deficit = (vesicle.enclosed_volume() - vesicle.volume0).abs();
        let initial = deficit;
        assert!(initial > 0.2 * vesicle.volume0);
        for _ in 0..20 {
            vesicle.step(dt);
            let next = (vesicle.enclosed_volume() - vesicle.volume0).abs();
            assert!(
                next < deficit,
                "volume error must shrink (flip {flip_winding})"
            );
            deficit = next;
        }
        assert!(deficit < 0.5 * initial, "{initial} -> {deficit}");
    }
}

#[test]
fn area_penalty_drives_a_stretched_vesicle_back_to_its_target_area() {
    let mut bilayer = LipidBilayer::new_icosphere(1.0e-6, 2, 1.0e-30, 0.0, 0.0, 0.0, 4.0e-9);
    let dt = 1.0e-3;
    // dt * lambda_A |grad A|^2 = 0.1 with |grad A|^2 = sum (2 A_i / R)^2.
    let grad_sq: f64 = bilayer
        .vertices
        .iter()
        .map(|v| (2.0 * v.area / 1.0e-6).powi(2))
        .sum();
    bilayer.k_area = 0.1 * bilayer.area0 / (dt * grad_sq);
    let mut vesicle = VesicleSimulation::from_bilayer(bilayer, 0.0);
    for vertex in &mut vesicle.bilayer.vertices {
        vertex.position = scale3(vertex.position, 1.2);
    }
    let area0 = vesicle.bilayer.area0;
    let mut deficit = (vesicle.bilayer.total_area() - area0).abs();
    let initial = deficit;
    assert!(initial > 0.2 * area0);
    for _ in 0..20 {
        vesicle.step(dt);
        let next = (vesicle.bilayer.total_area() - area0).abs();
        assert!(next < deficit, "area error must shrink");
        deficit = next;
    }
    assert!(deficit < 0.5 * initial, "{initial} -> {deficit}");
}

#[test]
fn osmotic_pressure_difference_shrinks_the_vesicle() {
    // Negligible bending rigidity: only the pressure term moves the surface.
    let bilayer = LipidBilayer::new_icosphere(1.0, 2, 1.0e-12, 0.0, 0.0, 0.0, 0.01);
    let mut vesicle = VesicleSimulation::from_bilayer(bilayer, 0.0);
    vesicle.delta_p = 1.0;
    let before = vesicle.enclosed_volume();
    vesicle.step(1.0e-4);
    let after = vesicle.enclosed_volume();
    assert!(
        after < before,
        "positive delta_p lowers the energy by shrinking"
    );
    // Work balance to first order: dV = -dt * dp * |grad V|^2 with |grad V|^2 ~ sum A_i^2.
    let grad_sq: f64 = vesicle
        .bilayer
        .vertices
        .iter()
        .map(|v| v.area * v.area)
        .sum();
    let predicted = 1.0e-4 * grad_sq;
    let observed = before - after;
    assert!(
        observed > 0.5 * predicted && observed < 2.0 * predicted,
        "observed {observed} vs predicted {predicted}"
    );
}

#[test]
fn step_on_an_empty_bilayer_only_advances_time() {
    let bilayer = LipidBilayer {
        kappa: KAPPA,
        kappa_gaussian: 0.0,
        c0: 0.0,
        k_area: 0.24,
        area0: 0.0,
        ade_alpha: 1.0,
        thickness: 4.0e-9,
        vertices: Vec::new(),
        triangles: Vec::new(),
    };
    let mut vesicle = VesicleSimulation::from_bilayer(bilayer, 1.0);
    vesicle.step(0.5);
    assert_eq!(vesicle.time, 0.5);
    assert_eq!(vesicle.enclosed_volume(), 0.0);
}
