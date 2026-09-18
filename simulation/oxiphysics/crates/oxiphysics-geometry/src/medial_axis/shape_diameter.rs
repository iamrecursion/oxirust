// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Shape diameter function of a triangle mesh.
//!
//! The shape diameter function (Shapira, Shamir & Cohen-Or, "Consistent mesh
//! partitioning and skeletonization using the shape diameter function", 2008)
//! measures the local thickness of a solid at a point of its surface: rays are
//! cast into the solid inside a cone around the inward normal and the lengths
//! up to the opposite wall are aggregated. For a cylinder of radius `R` the
//! mean ray length over a cone is exactly the diameter `2R`, for a slab it is
//! the thickness along the axis ray.

use crate::mesh_boolean::SimpleMesh;
use std::f64::consts::{FRAC_PI_2, PI};

/// Barycentric slack of the ray-triangle test: a ray that passes exactly
/// through a mesh edge or vertex must not slip between the triangles that
/// share it.
const BARYCENTRIC_TOLERANCE: f64 = 1e-9;

/// A ray is treated as parallel to a triangle when `|d · N|` is below this
/// fraction of its maximum `|N|` (`N` = unnormalised triangle normal).
const PARALLEL_TOLERANCE: f64 = 1e-12;

/// The ray origin is moved this fraction of the mesh diagonal into the solid
/// so that the triangle containing the query point is not hit again.
const ORIGIN_OFFSET: f64 = 1e-9;

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

fn is_finite3(a: V3) -> bool {
    a.iter().all(|c| c.is_finite())
}

/// A non-degenerate mesh triangle prepared for ray casting.
struct Triangle {
    v0: V3,
    e1: V3,
    e2: V3,
    /// Unnormalised counter-clockwise normal `e1 × e2` (outward for an
    /// outward-oriented mesh).
    normal: V3,
    /// `|e1| · |e2|`, the scale of `|normal|`.
    scale: f64,
}

impl Triangle {
    /// Intersect the ray `origin + t · dir` (unit `dir`) with the triangle.
    ///
    /// Returns the ray parameter `t > 0` and whether the ray leaves the solid
    /// through the triangle, i.e. crosses it along the outward normal
    /// (`normal · dir > 0`).
    fn hit(&self, origin: V3, dir: V3) -> Option<(f64, bool)> {
        let h = cross(dir, self.e2);
        let det = dot(self.e1, h);
        if det.abs() <= PARALLEL_TOLERANCE * self.scale {
            return None;
        }
        let inv = 1.0 / det;
        let s = sub(origin, self.v0);
        let u = inv * dot(s, h);
        if !(-BARYCENTRIC_TOLERANCE..=1.0 + BARYCENTRIC_TOLERANCE).contains(&u) {
            return None;
        }
        let q = cross(s, self.e1);
        let v = inv * dot(dir, q);
        if v < -BARYCENTRIC_TOLERANCE || u + v > 1.0 + BARYCENTRIC_TOLERANCE {
            return None;
        }
        let t = inv * dot(self.e2, q);
        if t > 0.0 {
            Some((t, dot(self.normal, dir) > 0.0))
        } else {
            None
        }
    }
}

/// Collect the usable triangles of `mesh` (valid indices, finite
/// coordinates, non-zero area) and the diagonal of their bounding box.
fn prepare_triangles(mesh: &SimpleMesh) -> (Vec<Triangle>, f64) {
    let mut triangles = Vec::with_capacity(mesh.triangles.len());
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for tri in &mesh.triangles {
        let (Some(&a), Some(&b), Some(&c)) = (
            mesh.vertices.get(tri[0]),
            mesh.vertices.get(tri[1]),
            mesh.vertices.get(tri[2]),
        ) else {
            continue;
        };
        if !(is_finite3(a) && is_finite3(b) && is_finite3(c)) {
            continue;
        }
        let e1 = sub(b, a);
        let e2 = sub(c, a);
        let normal = cross(e1, e2);
        let scale = norm(e1) * norm(e2);
        if norm(normal) <= PARALLEL_TOLERANCE * scale {
            continue;
        }
        for corner in [a, b, c] {
            for axis in 0..3 {
                lo[axis] = lo[axis].min(corner[axis]);
                hi[axis] = hi[axis].max(corner[axis]);
            }
        }
        triangles.push(Triangle {
            v0: a,
            e1,
            e2,
            normal,
            scale,
        });
    }
    let diagonal = if triangles.is_empty() {
        0.0
    } else {
        norm(sub(hi, lo))
    };
    (triangles, diagonal)
}

/// Estimate the local thickness of a solid, the *shape diameter*, at the
/// surface point `p` with outward normal `n`.
///
/// `n_rays` rays are cast into the solid from just inside `p`. Their
/// directions fill the cone of half-angle `cone_half_angle` (radians) around
/// the inward normal `-n` evenly in solid angle (a Fibonacci lattice, so the
/// result is deterministic); with `cone_half_angle == 0` every ray follows
/// `-n`. For each ray the nearest intersection with the mesh is found. A ray
/// counts only if it leaves the solid there, i.e. the hit triangle's outward
/// normal points along the ray; a ray whose nearest hit is a front face (it
/// has already left the solid, e.g. at a convex edge or through a hole) or that
/// hits nothing is discarded. The result is the median length of the counted
/// rays, which is robust against the stray rays that slip through gaps.
///
/// Ray lengths are not corrected for obliquity, as in the original definition:
/// an oblique ray through a slab of thickness `T` is longer than `T` (use a
/// zero cone angle to read the thickness along the normal). For a long
/// cylinder of radius `R` the mean ray length is exactly the diameter `2R` for
/// any cone and the median returned here stays within 2 % of it up to a cone
/// half-angle of 60°.
///
/// The mesh must be a closed surface whose triangles are wound
/// counter-clockwise seen from outside (outward normals); triangles with
/// out-of-range indices, non-finite coordinates or zero area are ignored. The
/// cost is `O(n_rays · triangles)`.
///
/// Returns `None` when no ray counts, and also for invalid input: no
/// triangles, `n_rays == 0`, a non-finite `p` or `n`, a zero `n`, or a
/// `cone_half_angle` outside `[0, π/2)`.
pub fn shape_diameter(
    mesh: &SimpleMesh,
    p: [f64; 3],
    n: [f64; 3],
    n_rays: usize,
    cone_half_angle: f64,
) -> Option<f64> {
    if n_rays == 0 || !is_finite3(p) || !is_finite3(n) {
        return None;
    }
    if !(0.0..FRAC_PI_2).contains(&cone_half_angle) {
        return None;
    }
    let n_len = norm(n);
    if n_len == 0.0 || !n_len.is_finite() {
        return None;
    }
    let (triangles, diagonal) = prepare_triangles(mesh);
    if triangles.is_empty() {
        return None;
    }

    // Cone axis (into the solid) and an orthonormal frame around it.
    let axis = [-n[0] / n_len, -n[1] / n_len, -n[2] / n_len];
    let helper = if axis[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let t1_raw = cross(axis, helper);
    let t1_len = norm(t1_raw);
    let t1 = [t1_raw[0] / t1_len, t1_raw[1] / t1_len, t1_raw[2] / t1_len];
    let t2 = cross(axis, t1);

    let offset = ORIGIN_OFFSET * diagonal;
    let origin = [
        p[0] + offset * axis[0],
        p[1] + offset * axis[1],
        p[2] + offset * axis[2],
    ];
    let golden_angle = PI * (3.0 - 5.0_f64.sqrt());
    let cos_cone = cone_half_angle.cos();

    let mut lengths: Vec<f64> = Vec::with_capacity(n_rays);
    for k in 0..n_rays {
        // Uniform in solid angle: cos(theta) is uniform on [cos(cone), 1].
        let cos_theta = 1.0 - (1.0 - cos_cone) * (k as f64 + 0.5) / n_rays as f64;
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = golden_angle * k as f64;
        let (sin_phi, cos_phi) = phi.sin_cos();
        let dir = [
            cos_theta * axis[0] + sin_theta * (cos_phi * t1[0] + sin_phi * t2[0]),
            cos_theta * axis[1] + sin_theta * (cos_phi * t1[1] + sin_phi * t2[1]),
            cos_theta * axis[2] + sin_theta * (cos_phi * t1[2] + sin_phi * t2[2]),
        ];

        let mut nearest: Option<(f64, bool)> = None;
        for triangle in &triangles {
            if let Some((t, leaving)) = triangle.hit(origin, dir)
                && nearest.is_none_or(|(best, _)| t < best)
            {
                nearest = Some((t, leaving));
            }
        }
        if let Some((t, true)) = nearest {
            let hit = [
                origin[0] + t * dir[0],
                origin[1] + t * dir[1],
                origin[2] + t * dir[2],
            ];
            lengths.push(norm(sub(hit, p)));
        }
    }
    if lengths.is_empty() {
        return None;
    }

    lengths.sort_by(f64::total_cmp);
    let mid = lengths.len() / 2;
    Some(if lengths.len() % 2 == 1 {
        lengths[mid]
    } else {
        0.5 * (lengths[mid - 1] + lengths[mid])
    })
}
