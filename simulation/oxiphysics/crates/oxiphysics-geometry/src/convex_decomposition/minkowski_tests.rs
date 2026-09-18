// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the exact point containment of [`MinkowskiSum`].
//!
//! A point lies in `A ⊕ B` exactly when `A` meets `p − B`.  The tests compare
//! the GJK-based answer with two independent oracles built from the explicit
//! convex hull of all pairwise sums `a + b`: the facet-plane test for the
//! membership and the point-to-triangle distance for the tolerance.

use super::*;

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

    fn point(&mut self, half: f64) -> Vec3 {
        [
            self.range(-half, half),
            self.range(-half, half),
            self.range(-half, half),
        ]
    }
}

fn octahedron() -> Vec<Vec3> {
    vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
}

fn cube(half: f64) -> Vec<Vec3> {
    let mut corners = Vec::new();
    for x in [-half, half] {
        for y in [-half, half] {
            for z in [-half, half] {
                corners.push([x, y, z]);
            }
        }
    }
    corners
}

fn minkowski(a: Vec<Vec3>, b: Vec<Vec3>) -> MinkowskiSum {
    MinkowskiSum::new(SupportFunction::new(a), SupportFunction::new(b))
}

fn pairwise_sums(a: &[Vec3], b: &[Vec3]) -> Vec<Vec3> {
    a.iter()
        .flat_map(|&x| b.iter().map(move |&y| add(x, y)))
        .collect()
}

/// Largest signed distance of `p` to a facet plane of `hull`: negative inside,
/// positive outside, and the exact distance to the nearest facet plane.
fn plane_margin(hull: &ConvexHull3d, p: Vec3) -> f64 {
    hull.faces
        .iter()
        .enumerate()
        .map(|(face, plane)| dot(plane.normal, sub(p, hull.vertices[hull.triangles[face][0]])))
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Closest point of the triangle `(a, b, c)` to `p` (Ericson, Real-Time
/// Collision Detection, section 5.1.5).
fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return add(a, scale(ab, d1 / (d1 - d3)));
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return add(a, scale(ac, d2 / (d2 - d6)));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(sub(c, b), w));
    }
    let denom = 1.0 / (va + vb + vc);
    add(a, add(scale(ab, vb * denom), scale(ac, vc * denom)))
}

/// Distance from `p` to the surface of `hull`.
fn surface_distance(hull: &ConvexHull3d, p: Vec3) -> f64 {
    hull.triangles
        .iter()
        .map(|tri| {
            let q = closest_point_on_triangle(
                p,
                hull.vertices[tri[0]],
                hull.vertices[tri[1]],
                hull.vertices[tri[2]],
            );
            norm(sub(p, q))
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn octahedron_sum_rejects_points_the_ten_fixed_directions_let_through() {
    // O ⊕ O = { |x| + |y| + |z| <= 2 }.  (0.9, 0.9, 0.9) has |.|₁ = 2.7 and is
    // outside, but it passes the test along each of the old ten directions.
    let sum = minkowski(octahedron(), octahedron());
    assert!(!sum.contains_point([0.9, 0.9, 0.9], 1e-9));
    assert!(sum.contains_point([0.6, 0.6, 0.6], 1e-9));
    assert!(sum.contains_point([0.0; 3], 1e-9));
    // The boundary is inside: a vertex, an edge point and a facet point.
    assert!(sum.contains_point([2.0, 0.0, 0.0], 1e-9));
    assert!(sum.contains_point([1.0, 1.0, 0.0], 1e-9));
    assert!(sum.contains_point([0.5, 0.5, 1.0], 1e-9));
    // And points just outside the boundary are not.
    assert!(!sum.contains_point([2.0 + 1e-6, 0.0, 0.0], 1e-9));
    assert!(!sum.contains_point([1.0, 1.0, 1e-6], 1e-12));
    assert!(!sum.contains_point([1.0, 1.0, 0.1], 1e-9));
    // A second point outside along the diagonal that the old directions also accepted.
    assert!(!sum.contains_point([0.7, 0.7, 0.7], 1e-9));
}

#[test]
fn tolerance_is_the_distance_from_the_sum() {
    // Outside the octahedron sum along +x the distance of (2.1, 0, 0) is 0.1.
    let sum = minkowski(octahedron(), octahedron());
    let p = [2.1, 0.0, 0.0];
    assert!(!sum.contains_point(p, 0.0));
    assert!(!sum.contains_point(p, 0.099));
    assert!(sum.contains_point(p, 0.101));
    // A negative or NaN tolerance counts as zero: deep inside is still inside,
    // outside is still outside.
    assert!(sum.contains_point([0.1, 0.0, 0.0], -1.0));
    assert!(sum.contains_point([0.1, 0.0, 0.0], f64::NAN));
    assert!(!sum.contains_point(p, -1.0));
    assert!(!sum.contains_point(p, f64::NAN));
}

#[test]
fn cube_sum_boundary_is_inside_with_zero_tolerance() {
    // [-1, 1]³ ⊕ [-1, 1]³ = [-2, 2]³: a corner, edge and face point, and a few
    // just outside.
    let sum = minkowski(cube(1.0), cube(1.0));
    for boundary in [
        [2.0, 2.0, 2.0],
        [2.0, 2.0, 0.37],
        [2.0, 0.3, -0.7],
        [-2.0, -1.5, 1.9],
    ] {
        assert!(sum.contains_point(boundary, 0.0), "{boundary:?}");
    }
    for outside in [[2.0 + 1e-7, 2.0, 2.0], [0.0, -2.0 - 1e-7, 0.0]] {
        assert!(!sum.contains_point(outside, 0.0), "{outside:?}");
    }
    assert!(sum.contains_point([2.0 + 1e-7, 2.0, 2.0], 1e-6));
}

#[test]
fn agrees_with_the_hull_of_the_pairwise_sums_on_random_polytopes() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut inside_cases = 0;
    let mut outside_cases = 0;
    for round in 0..14 {
        let a: Vec<Vec3> = (0..rng.range(5.0, 14.0) as usize)
            .map(|_| rng.point(1.0))
            .collect();
        let b: Vec<Vec3> = (0..rng.range(5.0, 12.0) as usize)
            .map(|_| rng.point(0.7))
            .collect();
        let sums = pairwise_sums(&a, &b);
        let hull = ConvexHull3d::build(&sums);
        assert!(!hull.triangles.is_empty(), "round {round}: flat sums");
        let sum = minkowski(a, b);
        for _ in 0..150 {
            let p = rng.point(1.7);
            let margin = plane_margin(&hull, p);
            if margin.abs() < 1e-7 {
                continue;
            }
            let label = format!("round {round}, p {p:?}, plane margin {margin}");
            if margin < 0.0 {
                inside_cases += 1;
                assert!(sum.contains_point(p, 0.0), "inside: {label}");
            } else {
                outside_cases += 1;
                assert!(!sum.contains_point(p, 0.0), "outside: {label}");
                // The tolerance is the Euclidean distance to the sum.
                let distance = surface_distance(&hull, p);
                assert!(distance >= margin - 1e-9, "{label}");
                if distance > 1e-6 {
                    assert!(!sum.contains_point(p, 0.999 * distance), "{label}");
                    assert!(sum.contains_point(p, 1.001 * distance), "{label}");
                }
            }
        }
    }
    assert!(inside_cases > 200, "only {inside_cases} inside probes");
    assert!(outside_cases > 300, "only {outside_cases} outside probes");
}

#[test]
fn flat_and_degenerate_operands_are_handled() {
    // Two coplanar squares: a flat square [-1.5, 1.5]² in the plane z = 0.
    let square = |half: f64| -> Vec<Vec3> {
        vec![
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ]
    };
    let flat = minkowski(square(1.0), square(0.5));
    assert!(flat.contains_point([1.4, 1.4, 0.0], 1e-9));
    assert!(flat.contains_point([-1.5, 0.2, 0.0], 1e-9));
    assert!(!flat.contains_point([1.6, 0.0, 0.0], 1e-9));
    assert!(!flat.contains_point([0.0, 0.0, 0.5], 1e-9));
    assert!(flat.contains_point([0.0, 0.0, 0.5], 0.5001));
    assert!(flat.contains_point([1.6, 0.0, 0.0], 0.1001));
    assert!(!flat.contains_point([1.6, 0.0, 0.0], 0.0999));

    // A single point plus a cube is the translated cube.
    let moved_cube = minkowski(vec![[1.0, 2.0, 3.0]], cube(1.0));
    assert!(moved_cube.contains_point([1.5, 2.5, 3.5], 1e-9));
    assert!(moved_cube.contains_point([2.0, 3.0, 4.0], 1e-9));
    assert!(!moved_cube.contains_point([2.5, 2.0, 3.0], 1e-9));
    assert!(!moved_cube.contains_point([0.0, 0.0, 0.0], 1e-9));

    // Two perpendicular segments span a rectangle.
    let rectangle = minkowski(
        vec![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        vec![[0.0, -2.0, 0.0], [0.0, 2.0, 0.0]],
    );
    assert!(rectangle.contains_point([0.5, 1.5, 0.0], 1e-9));
    assert!(rectangle.contains_point([1.0, -2.0, 0.0], 1e-9));
    assert!(!rectangle.contains_point([0.5, 2.5, 0.0], 1e-9));
    assert!(!rectangle.contains_point([0.5, 0.0, 0.1], 1e-9));
}

#[test]
fn empty_operands_and_non_finite_input_contain_nothing() {
    let cube_support = SupportFunction::new(cube(1.0));
    let empty = SupportFunction::new(Vec::new());
    let no_a = MinkowskiSum::new(empty.clone(), cube_support.clone());
    let no_b = MinkowskiSum::new(cube_support, empty);
    for sum in [&no_a, &no_b] {
        assert!(!sum.contains_point([0.0; 3], 1e-9));
        assert!(!sum.contains_point([0.0; 3], f64::INFINITY));
    }
    let sum = minkowski(cube(1.0), cube(1.0));
    assert!(!sum.contains_point([f64::NAN, 0.0, 0.0], 1e-9));
    assert!(!sum.contains_point([0.0, f64::INFINITY, 0.0], 1e-9));
    let broken = minkowski(vec![[f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0]], cube(1.0));
    assert!(!broken.contains_point([0.0; 3], 1e-9));
}

#[test]
fn many_vertex_operands_converge() {
    // Fibonacci points on spheres of radius 1 and 0.5: the sum is nearly the
    // ball of radius 1.5.
    let fibonacci = |count: usize, radius: f64| -> Vec<Vec3> {
        let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        (0..count)
            .map(|i| {
                let y = 1.0 - 2.0 * (i as f64 + 0.5) / count as f64;
                let r = (1.0 - y * y).sqrt();
                let angle = golden * i as f64;
                [
                    radius * r * angle.cos(),
                    radius * y,
                    radius * r * angle.sin(),
                ]
            })
            .collect()
    };
    let sum = minkowski(fibonacci(400, 1.0), fibonacci(300, 0.5));
    let mut rng = Rng(7);
    for _ in 0..60 {
        let direction = normalize(rng.point(1.0));
        if norm(direction) == 0.0 {
            continue;
        }
        assert!(sum.contains_point(scale(direction, 1.3), 1e-9));
        assert!(!sum.contains_point(scale(direction, 1.6), 1e-9));
    }
}
