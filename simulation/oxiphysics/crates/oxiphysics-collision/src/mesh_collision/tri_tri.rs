// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Triangle-triangle intersection by the separating axis theorem.
//!
//! Two triangles (flat convex polygons in 3-D) are disjoint exactly when some
//! axis separates their projections. Every such pair is separated by one of
//!
//! * the two face normals,
//! * the nine cross products of an edge of one with an edge of the other, or
//! * for (nearly) coplanar pairs, where all of the above degenerate, the in-plane
//!   edge normals `n x e`.
//!
//! The in-plane axes are always generated, for every non-degenerate normal and
//! every edge of both triangles. They are valid for any configuration, and a
//! candidate axis can only ever prove a *true* separation (the projections of
//! all six vertices are compared, nothing is assumed about the axis), so extra
//! or slightly rotated axes cost time but never produce a wrong answer. There
//! is no division anywhere, hence no `0/0` for coplanar or touching pairs.
//!
//! A triangle whose height over its longest edge is below the contact tolerance
//! is a segment (or a point). Against a proper triangle it is handled by the
//! same axes; two such triangles are compared as segments.
//!
//! Touching counts as intersecting: the pair is separated only if the gap along
//! an axis exceeds `CONTACT_REL_TOL` times the extent of the six vertices, which
//! absorbs the rounding of the projections.

use super::{vec3_cross, vec3_dot, vec3_len, vec3_scale, vec3_sub};

type V3 = [f64; 3];

/// Gaps up to this fraction of the extent of the two triangles count as contact.
const CONTACT_REL_TOL: f64 = 1e-12;
/// A candidate axis shorter than this fraction of the product of the lengths of
/// the vectors it was crossed from is parallel to them and carries no
/// information.
const PARALLEL_REL_TOL: f64 = 1e-12;
/// Two face normals, nine edge pairs and twelve in-plane axes (two normals times
/// six edges).
const MAX_AXES: usize = 23;

/// One input triangle, expressed relative to the common origin.
struct Tri {
    /// Vertices relative to the origin.
    v: [V3; 3],
    /// Edge `i` runs from `v[i]` to `v[(i + 1) % 3]`.
    edges: [V3; 3],
    /// Unit face normal, `None` when the triangle is a segment or a point.
    normal: Option<V3>,
    /// End points of the longest edge (equal for a point): the convex hull of a
    /// degenerate triangle.
    hull: (V3, V3),
}

impl Tri {
    fn new(v: [V3; 3], tol: f64) -> Self {
        let edges = [
            vec3_sub(v[1], v[0]),
            vec3_sub(v[2], v[1]),
            vec3_sub(v[0], v[2]),
        ];
        let lengths = edges.map(vec3_len);
        let mut longest = 0;
        for (i, &len) in lengths.iter().enumerate() {
            if len > lengths[longest] {
                longest = i;
            }
        }
        let longest_len = lengths[longest];
        // (v1 - v0) x (v2 - v0); its length is twice the area.
        let cross = vec3_cross(edges[0], vec3_scale(edges[2], -1.0));
        let twice_area = vec3_len(cross);
        // Height over the longest edge is twice_area / longest_len.
        let degenerate = longest_len <= tol || twice_area <= tol * longest_len;
        let normal = if degenerate {
            None
        } else {
            Some(vec3_scale(cross, 1.0 / twice_area))
        };
        let hull = if longest_len <= tol {
            (v[0], v[0])
        } else {
            (v[longest], v[(longest + 1) % 3])
        };
        Self {
            v,
            edges,
            normal,
            hull,
        }
    }
}

/// Range of `axis . p` over the three vertices.
fn project(v: &[V3; 3], axis: V3) -> (f64, f64) {
    let d = [
        vec3_dot(v[0], axis),
        vec3_dot(v[1], axis),
        vec3_dot(v[2], axis),
    ];
    (d[0].min(d[1]).min(d[2]), d[0].max(d[1]).max(d[2]))
}

/// Squared distance between the segments `p1..q1` and `p2..q2` (either may be a
/// point), after Ericson, *Real-Time Collision Detection*, 5.1.9.
fn segment_segment_dist_sq(p1: V3, q1: V3, p2: V3, q2: V3) -> f64 {
    let d1 = vec3_sub(q1, p1);
    let d2 = vec3_sub(q2, p2);
    let r = vec3_sub(p1, p2);
    let a = vec3_dot(d1, d1);
    let e = vec3_dot(d2, d2);
    let f = vec3_dot(d2, r);
    let point_1 = a <= f64::MIN_POSITIVE;
    let point_2 = e <= f64::MIN_POSITIVE;
    let (s, t) = if point_1 && point_2 {
        (0.0, 0.0)
    } else if point_1 {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = vec3_dot(d1, r);
        if point_2 {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = vec3_dot(d1, d2);
            let denom = a * e - b * b;
            // Parallel segments (denom == 0) may start anywhere on segment 1.
            let s0 = if denom > 0.0 {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let t0 = (b * s0 + f) / e;
            if t0 < 0.0 {
                ((-c / a).clamp(0.0, 1.0), 0.0)
            } else if t0 > 1.0 {
                (((b - c) / a).clamp(0.0, 1.0), 1.0)
            } else {
                (s0, t0)
            }
        }
    };
    let c1 = [p1[0] + d1[0] * s, p1[1] + d1[1] * s, p1[2] + d1[2] * s];
    let c2 = [p2[0] + d2[0] * t, p2[1] + d2[1] * t, p2[2] + d2[2] * t];
    let gap = vec3_sub(c1, c2);
    vec3_dot(gap, gap)
}

/// Whether the closed triangles with vertices `t1` and `t2` have a common point.
///
/// Non-finite coordinates never intersect anything.
pub(super) fn triangles_intersect(t1: &[V3; 3], t2: &[V3; 3]) -> bool {
    if !t1.iter().chain(t2.iter()).flatten().all(|c| c.is_finite()) {
        return false;
    }
    // Work relative to a vertex of the first triangle: coordinates far from the
    // origin would otherwise cost digits in every difference below.
    let origin = t1[0];
    let local_1 = t1.map(|p| vec3_sub(p, origin));
    let local_2 = t2.map(|p| vec3_sub(p, origin));
    let extent = local_1
        .iter()
        .chain(local_2.iter())
        .map(|&p| vec3_len(p))
        .fold(0.0_f64, f64::max);
    let tol = CONTACT_REL_TOL * extent;
    let a = Tri::new(local_1, tol);
    let b = Tri::new(local_2, tol);

    if a.normal.is_none() && b.normal.is_none() {
        // Two segments / points: no face normal exists to build the full axis set.
        return segment_segment_dist_sq(a.hull.0, a.hull.1, b.hull.0, b.hull.1) <= tol * tol;
    }

    let mut axes = [[0.0_f64; 3]; MAX_AXES];
    let mut count = 0;
    let mut add = |axis: V3, reference: f64| {
        let len = vec3_len(axis);
        if len > 0.0 && len > PARALLEL_REL_TOL * reference && count < MAX_AXES {
            axes[count] = vec3_scale(axis, 1.0 / len);
            count += 1;
        }
    };
    let normals = [a.normal, b.normal];
    for n in normals.into_iter().flatten() {
        add(n, 1.0);
    }
    for &ea in &a.edges {
        for &eb in &b.edges {
            add(vec3_cross(ea, eb), vec3_len(ea) * vec3_len(eb));
        }
    }
    for n in normals.into_iter().flatten() {
        for &e in a.edges.iter().chain(b.edges.iter()) {
            add(vec3_cross(n, e), vec3_len(e));
        }
    }

    axes[..count].iter().all(|&axis| {
        let (a_lo, a_hi) = project(&a.v, axis);
        let (b_lo, b_hi) = project(&b.v, axis);
        // Not separated along this axis.
        a_hi + tol >= b_lo && b_hi + tol >= a_lo
    })
}
