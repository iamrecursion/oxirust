// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Exact distance between the convex hulls of two finite point sets.
//!
//! This is the Gilbert-Johnson-Keerthi algorithm specialised to polytopes given
//! by their vertices. The support function of such a shape is an exact `argmax`
//! over the vertex list, the Minkowski difference of the two hulls is the hull
//! of all pairwise differences, and the algorithm terminates after finitely many
//! iterations with the exact closest pair of points (up to floating-point
//! rounding), including for degenerate hulls (a single point, a segment, a flat
//! polygon).
//!
//! The closest point of the current simplex (at most four vertices of the
//! Minkowski difference) to the origin is found by examining every sub-simplex
//! (vertex, edge, triangle, tetrahedron): the origin is projected onto the
//! affine hull of each one, the projections that fall inside their sub-simplex
//! are convex combinations of the vertices, and the nearest of them is the
//! closest point. Choosing by distance, rather than by an optimality test such
//! as `c . q >= c . c` for all simplex vertices `q`, matters when the hulls
//! touch or overlap: the closest point is then the origin itself, every dot
//! product in such a test is pure rounding noise and no tolerance proportional
//! to `|c|` can accept it. Termination follows van den Bergen: the iteration
//! stops when the support point in direction `-c` no longer improves on the
//! closest point `c` of the current simplex, or when rounding stops the distance
//! from decreasing, in which case the best simplex found so far is returned.

use super::{add3, dot3, len3, scale3, sub3};

/// Upper bound on GJK iterations. A polytope with `n` vertices needs far fewer.
const MAX_ITERATIONS: usize = 128;

/// One vertex of the Minkowski difference `A - B`, remembering its sources.
#[derive(Clone, Copy)]
struct CsoVertex {
    /// Index of the support point in shape A.
    index_a: usize,
    /// Index of the support point in shape B.
    index_b: usize,
    /// Support point of A (already translated).
    point_a: [f64; 3],
    /// Support point of B.
    point_b: [f64; 3],
    /// `point_a - point_b`.
    point: [f64; 3],
}

/// Distance query result for two convex hulls.
#[derive(Debug, Clone)]
pub(super) struct HullDistance {
    /// Euclidean distance between the hulls (zero if they intersect).
    pub distance: f64,
    /// Closest point on hull A (a point common to both hulls if they intersect).
    pub point_a: [f64; 3],
    /// Closest point on hull B (a point common to both hulls if they intersect).
    pub point_b: [f64; 3],
    /// True if the hulls intersect or touch.
    pub intersecting: bool,
}

/// Index of the first point of `points` that maximises `p . dir`.
fn support_index(points: &[[f64; 3]], dir: [f64; 3]) -> usize {
    points
        .iter()
        .enumerate()
        .fold((0usize, f64::NEG_INFINITY), |(best, best_value), (i, p)| {
            let value = dot3(*p, dir);
            if value > best_value {
                (i, value)
            } else {
                (best, best_value)
            }
        })
        .0
}

/// Mean of the points (`None` if empty).
pub(super) fn centroid(points: &[[f64; 3]]) -> Option<[f64; 3]> {
    if points.is_empty() {
        return None;
    }
    let sum = points.iter().fold([0.0; 3], |acc, p| add3(acc, *p));
    Some(scale3(sum, 1.0 / points.len() as f64))
}

/// Smallest accepted size of an edge of a sub-simplex, relative to its own
/// length, once its components along the previous edges are removed. A
/// sub-simplex below it is affinely dependent (numerically flat) and a smaller
/// sub-simplex covers the same points.
const DEPENDENT_EDGE: f64 = 1e-10;

/// Barycentric weights (aligned with `verts`) of the point of the affine hull of
/// the sub-simplex `members` that is closest to the origin, or `None` when the
/// sub-simplex is affinely dependent or the point lies outside the sub-simplex.
///
/// With `first` the first member and `e_i` the edges from it to the other
/// members, the point is `first + sum(mu_i e_i)` where `mu` minimises its length.
/// The edges are orthonormalised (modified Gram-Schmidt, applied twice) and `mu`
/// is found by back substitution on the triangular factor. Solving the normal
/// equations instead would square the condition number: for a tetrahedron as
/// thin as a small box lying against a very large triangle that loses about
/// eight digits, enough to miss that the tetrahedron encloses the origin.
fn subset_weights(verts: &[CsoVertex], members: &[usize]) -> Option<[f64; 4]> {
    let first = verts.get(*members.first()?)?.point;
    let k = members.len() - 1;
    let mut weights = [0.0; 4];
    if k == 0 {
        *weights.get_mut(members[0])? = 1.0;
        return Some(weights);
    }

    // `basis` holds the orthonormal vectors; `factor[j][i]` is the component of
    // edge `i` along basis vector `j` for `j < i`, and `factor[i][i]` the length
    // left of edge `i` after removing its components along the earlier ones.
    let mut basis = [[0.0; 3]; 3];
    let mut factor = [[0.0; 3]; 3];
    for (i, &m) in members.iter().skip(1).enumerate() {
        let edge = sub3(verts.get(m)?.point, first);
        let mut rest = edge;
        for _ in 0..2 {
            for (q, row) in basis.iter().zip(factor.iter_mut()).take(i) {
                let along = dot3(*q, rest);
                row[i] += along;
                rest = sub3(rest, scale3(*q, along));
            }
        }
        let rest_len = len3(rest);
        if rest_len.is_nan() || rest_len <= DEPENDENT_EDGE * len3(edge) {
            return None;
        }
        factor[i][i] = rest_len;
        basis[i] = scale3(rest, 1.0 / rest_len);
    }

    // Solve `factor * mu = -basis^T first` from the last unknown to the first.
    let mut mu = [0.0; 3];
    for i in (0..k).rev() {
        let beyond: f64 = factor[i][i + 1..k]
            .iter()
            .zip(&mu[i + 1..k])
            .map(|(r, m)| r * m)
            .sum();
        mu[i] = (-dot3(basis[i], first) - beyond) / factor[i][i];
    }

    let mut sum = 0.0;
    for (slot, &m) in members.iter().enumerate().skip(1) {
        *weights.get_mut(m)? = mu[slot - 1];
        sum += mu[slot - 1];
    }
    *weights.get_mut(members[0])? = 1.0 - sum;
    // Weights that are negative beyond rounding put the point outside the face.
    if weights.iter().any(|&w| w < -1e-12) {
        return None;
    }
    let mut total = 0.0;
    for w in weights.iter_mut() {
        *w = w.max(0.0);
        total += *w;
    }
    if total.is_nan() || total <= 0.0 {
        return None;
    }
    for w in weights.iter_mut() {
        *w /= total;
    }
    Some(weights)
}

/// Closest point to the origin of the convex hull of `verts` (one to four
/// vertices), with barycentric weights aligned with `verts`.
///
/// The closest point of a simplex hull is the projection of the origin onto the
/// affine hull of one of its faces (a vertex, an edge, a triangle or the whole
/// tetrahedron), taken from a face that contains its own projection. Every such
/// projection is a convex combination of the vertices, so none of them can be
/// nearer to the origin than the hull is, and the nearest one is the closest
/// point. A vertex is always a valid face, so there is always a candidate, and
/// rounding errors can only make the result marginally too far, never too near.
fn closest_to_origin(verts: &[CsoVertex]) -> ([f64; 3], [f64; 4]) {
    let n = verts.len().min(4);
    let verts = &verts[..n];
    let Some(first) = verts.first() else {
        return ([0.0; 3], [0.0; 4]);
    };

    let mut best_point = first.point;
    let mut best_weights = [1.0, 0.0, 0.0, 0.0];
    let mut best_sq = f64::INFINITY;
    // Sub-simplices in order of increasing size, so that of two equally good
    // candidates the one with fewer vertices wins.
    for size in 1..=n {
        for mask in 1u32..(1u32 << n) {
            if mask.count_ones() as usize != size {
                continue;
            }
            let mut members = [0usize; 4];
            let mut count = 0;
            for i in 0..n {
                if mask & (1 << i) != 0 {
                    members[count] = i;
                    count += 1;
                }
            }
            let Some(weights) = subset_weights(verts, &members[..count]) else {
                continue;
            };
            let mut point = [0.0; 3];
            for (v, &w) in verts.iter().zip(weights.iter()) {
                point = add3(point, scale3(v.point, w));
            }
            let point_sq = dot3(point, point);
            if point_sq < best_sq * (1.0 - 1e-12) {
                best_point = point;
                best_weights = weights;
                best_sq = point_sq;
            }
        }
    }
    (best_point, best_weights)
}

/// Assemble the witnesses of the active vertices `(vertex, weight)`.
fn assemble(active: &[(CsoVertex, f64)], closest_sq: f64, intersecting: bool) -> HullDistance {
    let mut point_a = [0.0; 3];
    let mut point_b = [0.0; 3];
    for &(v, w) in active {
        point_a = add3(point_a, scale3(v.point_a, w));
        point_b = add3(point_b, scale3(v.point_b, w));
    }
    HullDistance {
        distance: if intersecting { 0.0 } else { closest_sq.sqrt() },
        point_a,
        point_b,
        intersecting,
    }
}

/// Distance between the convex hull of `points_a`, translated by `offset_a`, and
/// the convex hull of `points_b`.
///
/// Returns `None` if either point set is empty.
pub(super) fn hull_distance(
    points_a: &[[f64; 3]],
    offset_a: [f64; 3],
    points_b: &[[f64; 3]],
) -> Option<HullDistance> {
    let centroid_a = add3(centroid(points_a)?, offset_a);
    let centroid_b = centroid(points_b)?;

    let support = |dir: [f64; 3]| -> Option<CsoVertex> {
        let index_a = support_index(points_a, dir);
        let index_b = support_index(points_b, scale3(dir, -1.0));
        let point_a = add3(*points_a.get(index_a)?, offset_a);
        let point_b = *points_b.get(index_b)?;
        Some(CsoVertex {
            index_a,
            index_b,
            point_a,
            point_b,
            point: sub3(point_a, point_b),
        })
    };

    let mut direction = sub3(centroid_b, centroid_a);
    if !direction.iter().all(|c| c.is_finite()) || dot3(direction, direction) <= 0.0 {
        direction = [1.0, 0.0, 0.0];
    }

    let mut simplex: Vec<CsoVertex> = Vec::with_capacity(4);
    simplex.push(support(direction)?);
    let mut scale = len3(simplex[0].point);
    // The result of the iteration with the smallest distance so far, together
    // with its squared distance.
    let mut best: Option<(HullDistance, f64)> = None;

    for _ in 0..MAX_ITERATIONS {
        let (closest, weights) = closest_to_origin(&simplex);
        // Keep only the vertices that carry the closest point.
        let mut active: Vec<(CsoVertex, f64)> = simplex
            .iter()
            .zip(weights.iter())
            .filter(|&(_, &w)| w > 0.0)
            .map(|(v, &w)| (*v, w))
            .collect();
        if active.is_empty() {
            active.push((simplex[0], 1.0));
        }
        simplex = active.iter().map(|&(v, _)| v).collect();

        let closest_sq = dot3(closest, closest);
        let zero = 1e-12 * scale.max(1e-300);
        if closest_sq <= zero * zero {
            return Some(assemble(&active, closest_sq, true));
        }
        // In exact arithmetic every iteration brings the closest point nearer
        // to the origin; if it does not, rounding has stalled the iteration and
        // the previous simplex is the better answer.
        if best
            .as_ref()
            .is_some_and(|&(_, best_sq)| closest_sq >= best_sq)
        {
            break;
        }
        best = Some((assemble(&active, closest_sq, false), closest_sq));

        let candidate = support(scale3(closest, -1.0))?;
        scale = scale.max(len3(candidate.point));
        let gap = closest_sq - dot3(closest, candidate.point);
        let tolerance =
            1e-13 * closest_sq + 16.0 * f64::EPSILON * len3(closest) * len3(candidate.point);
        let repeated = simplex
            .iter()
            .any(|v| v.index_a == candidate.index_a && v.index_b == candidate.index_b);
        if gap <= tolerance || repeated || simplex.len() >= 4 {
            break;
        }
        simplex.push(candidate);
    }
    best.map(|(result, _)| result)
}
