// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Signed-volume distance sub-algorithm for GJK.
//!
//! Given a simplex of one to four points, find the point of its convex hull
//! that is closest to the origin together with the smallest sub-simplex that
//! carries it. This is the method of Montanari, Petrinic and Barbieri
//! ("Improving the GJK algorithm for faster and more reliable distance queries
//! between convex objects", ACM TOG 36(3), 2017): instead of Johnson's
//! sub-determinants with tolerance-based degeneracy tests it decides every
//! Voronoi region by the *sign* of a signed length, area or volume, computed
//! on the coordinate axis or plane on which the simplex has the largest
//! projection. Comparing signs needs no tolerance, so long slivers and nearly
//! flat tetrahedra (which a relative flatness threshold would discard) are
//! still resolved correctly as long as their orientation is determined by the
//! floating-point coordinates.

use oxiphysics_core::math::{Real, Vec3};

/// Closest point of a simplex to the origin.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SimplexClosest {
    /// The closest point: the barycentric combination of the carrying vertices.
    pub(crate) point: Vec3,
    /// Indices into the input simplex of the carrying vertices.
    pub(crate) keep: [usize; 4],
    /// Number of carrying vertices (1 to 4). Four means that the origin lies
    /// inside the tetrahedron.
    pub(crate) len: usize,
}

impl SimplexClosest {
    fn vertex(point: &Vec3, index: usize) -> Self {
        Self {
            point: *point,
            keep: [index, 0, 0, 0],
            len: 1,
        }
    }

    /// Keep the candidate with the smaller distance to the origin.
    fn keep_closer(best: &mut Option<Self>, candidate: Self) {
        let closer = match best {
            Some(current) => candidate.point.norm_squared() < current.point.norm_squared(),
            None => true,
        };
        if closer {
            *best = Some(candidate);
        }
    }
}

/// Whether `a` and `b` are both strictly positive or both strictly negative.
fn same_sign(a: Real, b: Real) -> bool {
    (a > 0.0 && b > 0.0) || (a < 0.0 && b < 0.0)
}

/// Index of the component of `v` with the largest magnitude.
fn dominant_axis(v: &Vec3) -> usize {
    let (x, y, z) = (v.x.abs(), v.y.abs(), v.z.abs());
    if x >= y && x >= z {
        0
    } else if y >= z {
        1
    } else {
        2
    }
}

/// Closest point to the origin of the hull of `points` (1 to 4 entries; any
/// beyond the fourth are ignored).
pub(crate) fn closest_on_simplex(points: &[Vec3]) -> SimplexClosest {
    match points {
        [] => SimplexClosest::vertex(&Vec3::zeros(), 0),
        [a] => SimplexClosest::vertex(a, 0),
        [a, b] => closest_on_segment(a, b, [0, 1]),
        [a, b, c] => closest_on_triangle([a, b, c], [0, 1, 2]),
        [a, b, c, d, ..] => closest_on_tetrahedron([a, b, c, d]),
    }
}

/// Signed-length sub-algorithm on the segment `ab` (input indices `ids`).
fn closest_on_segment(a: &Vec3, b: &Vec3, ids: [usize; 2]) -> SimplexClosest {
    let edge = b - a;
    let length_sq = edge.norm_squared();
    if !(length_sq.is_finite() && length_sq > 0.0) {
        return SimplexClosest::vertex(a, ids[0]);
    }
    // Orthogonal projection of the origin onto the line, then barycentric
    // weights read off the axis of the largest extent.
    let along = -a.dot(&edge) / length_sq;
    let projection = a + edge * along;
    let axis = dominant_axis(&edge);
    let mu = edge[axis];
    let weight_a = b[axis] - projection[axis];
    let weight_b = projection[axis] - a[axis];
    let inside_a = same_sign(mu, weight_a);
    let inside_b = same_sign(mu, weight_b);
    if inside_a && inside_b {
        let (lambda_a, lambda_b) = (weight_a / mu, weight_b / mu);
        SimplexClosest {
            point: a * lambda_a + b * lambda_b,
            keep: [ids[0], ids[1], 0, 0],
            len: 2,
        }
    } else if inside_b {
        // The projection lies beyond b.
        SimplexClosest::vertex(b, ids[1])
    } else {
        SimplexClosest::vertex(a, ids[0])
    }
}

/// Twice the signed area of the triangle `uvw` projected on the coordinate
/// plane `(i, j)`.
fn projected_area(u: &Vec3, v: &Vec3, w: &Vec3, i: usize, j: usize) -> Real {
    (v[i] - u[i]) * (w[j] - u[j]) - (v[j] - u[j]) * (w[i] - u[i])
}

/// Signed-area sub-algorithm on the triangle `tri` (input indices `ids`).
fn closest_on_triangle(tri: [&Vec3; 3], ids: [usize; 3]) -> SimplexClosest {
    let [a, b, c] = tri;
    let normal = (b - a).cross(&(c - a));
    let normal_sq = normal.norm_squared();
    let mut best: Option<SimplexClosest> = None;
    if normal_sq.is_finite() && normal_sq > 0.0 {
        // Projection of the origin onto the plane of the triangle.
        let projection = normal * (a.dot(&normal) / normal_sq);
        let k = dominant_axis(&normal);
        let (i, j) = ((k + 1) % 3, (k + 2) % 3);
        let mu = projected_area(a, b, c, i, j);
        let cofactors = [
            projected_area(&projection, b, c, i, j),
            projected_area(a, &projection, c, i, j),
            projected_area(a, b, &projection, i, j),
        ];
        if cofactors.iter().all(|&cofactor| same_sign(mu, cofactor)) {
            let weights = [cofactors[0] / mu, cofactors[1] / mu, cofactors[2] / mu];
            return SimplexClosest {
                point: a * weights[0] + b * weights[1] + c * weights[2],
                keep: [ids[0], ids[1], ids[2], 0],
                len: 3,
            };
        }
        // The origin projects outside: the closest point lies on an edge
        // opposite a vertex whose weight has the wrong sign.
        for (opposite, &cofactor) in cofactors.iter().enumerate() {
            if same_sign(mu, cofactor) {
                continue;
            }
            let (p, q) = edge_without(opposite);
            SimplexClosest::keep_closer(
                &mut best,
                closest_on_segment(tri[p], tri[q], [ids[p], ids[q]]),
            );
        }
    } else {
        // Collinear (or non-finite) triangle: every edge is a candidate.
        for opposite in 0..3 {
            let (p, q) = edge_without(opposite);
            SimplexClosest::keep_closer(
                &mut best,
                closest_on_segment(tri[p], tri[q], [ids[p], ids[q]]),
            );
        }
    }
    best.unwrap_or_else(|| SimplexClosest::vertex(a, ids[0]))
}

/// The two vertex positions of a triangle other than `opposite`.
fn edge_without(opposite: usize) -> (usize, usize) {
    match opposite {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

/// Signed volume (six times the volume) of the tetrahedron `pqrs`.
fn signed_volume(p: &Vec3, q: &Vec3, r: &Vec3, s: &Vec3) -> Real {
    (q - p).dot(&(r - p).cross(&(s - p)))
}

/// Signed-volume sub-algorithm on the tetrahedron `tet`.
fn closest_on_tetrahedron(tet: [&Vec3; 4]) -> SimplexClosest {
    let [a, b, c, d] = tet;
    let origin = Vec3::zeros();
    let mu = signed_volume(a, b, c, d);
    let cofactors = [
        signed_volume(&origin, b, c, d),
        signed_volume(a, &origin, c, d),
        signed_volume(a, b, &origin, d),
        signed_volume(a, b, c, &origin),
    ];
    let flat = !(mu.is_finite() && mu != 0.0);
    if !flat && cofactors.iter().all(|&cofactor| same_sign(mu, cofactor)) {
        let weights = cofactors.map(|cofactor| cofactor / mu);
        return SimplexClosest {
            point: a * weights[0] + b * weights[1] + c * weights[2] + d * weights[3],
            keep: [0, 1, 2, 3],
            len: 4,
        };
    }
    let mut best: Option<SimplexClosest> = None;
    for (opposite, &cofactor) in cofactors.iter().enumerate() {
        if !flat && same_sign(mu, cofactor) {
            continue;
        }
        let face = match opposite {
            0 => [1, 2, 3],
            1 => [0, 2, 3],
            2 => [0, 1, 3],
            _ => [0, 1, 2],
        };
        SimplexClosest::keep_closer(
            &mut best,
            closest_on_triangle([tet[face[0]], tet[face[1]], tet[face[2]]], face),
        );
    }
    best.unwrap_or_else(|| SimplexClosest::vertex(a, 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: Real, y: Real, z: Real) -> Vec3 {
        Vec3::new(x, y, z)
    }

    fn closest(points: &[Vec3]) -> (Vec3, usize) {
        let result = closest_on_simplex(points);
        (result.point, result.len)
    }

    #[test]
    fn segment_regions() {
        let (p, n) = closest(&[v(1.0, -1.0, 0.0), v(1.0, 1.0, 0.0)]);
        assert_eq!(n, 2);
        assert!((p - v(1.0, 0.0, 0.0)).norm() < 1e-15);
        let (p, n) = closest(&[v(1.0, 1.0, 0.0), v(1.0, 2.0, 0.0)]);
        assert_eq!(n, 1);
        assert!((p - v(1.0, 1.0, 0.0)).norm() < 1e-15);
        let (p, n) = closest(&[v(1.0, 2.0, 0.0), v(1.0, 1.0, 0.0)]);
        assert_eq!(n, 1);
        assert!((p - v(1.0, 1.0, 0.0)).norm() < 1e-15);
    }

    #[test]
    fn triangle_regions() {
        let tri = [v(-1.0, -1.0, 2.0), v(3.0, -1.0, 2.0), v(-1.0, 3.0, 2.0)];
        let (p, n) = closest(&tri);
        assert_eq!(n, 3);
        assert!((p - v(0.0, 0.0, 2.0)).norm() < 1e-14);
        let tri = [v(1.0, 1.0, 0.0), v(3.0, 1.0, 0.0), v(1.0, 3.0, 0.0)];
        let (p, n) = closest(&tri);
        assert_eq!(n, 1);
        assert!((p - v(1.0, 1.0, 0.0)).norm() < 1e-15);
        let tri = [v(1.0, -1.0, 0.0), v(1.0, 1.0, 0.0), v(3.0, 0.0, 0.0)];
        let (p, n) = closest(&tri);
        assert_eq!(n, 2);
        assert!((p - v(1.0, 0.0, 0.0)).norm() < 1e-15);
    }

    #[test]
    fn collinear_triangle_falls_back_to_its_edges() {
        let tri = [v(1.0, -1.0, 0.0), v(1.0, 0.5, 0.0), v(1.0, 1.0, 0.0)];
        let (p, n) = closest(&tri);
        assert_eq!(n, 2);
        assert!((p - v(1.0, 0.0, 0.0)).norm() < 1e-15);
    }

    #[test]
    fn tetrahedron_encloses_or_projects_to_a_face() {
        let tet = [
            v(1.0, 1.0, 1.0),
            v(-1.0, -1.0, 1.0),
            v(-1.0, 1.0, -1.0),
            v(1.0, -1.0, -1.0),
        ];
        let result = closest_on_simplex(&tet);
        assert_eq!(result.len, 4);
        assert!(result.point.norm() < 1e-14);
        let shifted = tet.map(|p| p + v(0.0, 0.0, 5.0));
        let (p, n) = closest(&shifted);
        // The bottom edge of the shifted tetrahedron (z = 4) is closest.
        assert_eq!(n, 2);
        assert!((p - v(0.0, 0.0, 4.0)).norm() < 1e-14);
    }

    #[test]
    fn sliver_tetrahedron_around_the_origin_is_enclosed() {
        // Three vertices clustered within 1e-8 of each other and one far away:
        // the volume is tiny relative to the longest edge, but its sign (and
        // the signs of the cofactors) is well determined, so the enclosed
        // origin is recognised.
        let h = 1e-12;
        let tet = [
            v(-1e-3, 0.0, 0.0),
            v(1e-8, -1e-8, -h),
            v(1e-8, 1e-8, -h),
            v(1e-8, 0.0, 3.0 * h),
        ];
        let result = closest_on_simplex(&tet);
        assert_eq!(result.len, 4, "{result:?}");
    }
}
