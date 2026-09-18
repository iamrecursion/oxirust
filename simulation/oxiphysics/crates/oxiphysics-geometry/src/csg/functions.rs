//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{CsgOp, CsgTree, MarchingCell, PlaneSide};

/// Type alias for a mesh returned as (vertices, triangle-index-lists).
type VertsTris = (Vec<[f64; 3]>, Vec<[usize; 3]>);
/// Type alias for a list of (point, normal) pairs.
type PointNormals = Vec<([f64; 3], [f64; 3])>;
/// Type alias for an AABB of an implicit surface: (min, max) corner pair.
type SurfaceAabb = ([f64; 3], [f64; 3]);

#[inline]
pub(super) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub(super) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub(super) fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub(super) fn scale(a: [f64; 3], t: f64) -> [f64; 3] {
    [a[0] * t, a[1] * t, a[2] * t]
}
#[inline]
pub(super) fn length(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
#[inline]
pub(super) fn normalize(a: [f64; 3]) -> [f64; 3] {
    let l = length(a);
    if l > 1e-15 {
        scale(a, 1.0 / l)
    } else {
        [0.0, 0.0, 0.0]
    }
}

#[inline]
pub(super) fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}
#[inline]
pub(super) fn mix(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}
#[inline]
pub(super) fn lerp3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}
/// Signed distance field interface.
///
/// `sdf` returns a negative value inside the surface, 0 on the surface, and a
/// positive value outside.  `gradient` returns the unit-outward normal (or an
/// approximation thereof).
pub trait ImplicitSurface: Send + Sync {
    /// Signed distance from `p` to the nearest point on the surface.
    fn sdf(&self, p: [f64; 3]) -> f64;
    /// Gradient of the SDF at `p` (should be approximately unit-length on the surface).
    fn gradient(&self, p: [f64; 3]) -> [f64; 3];
}
/// Sphere-march a ray against an implicit surface.
///
/// Advances along `ray_dir` by `sdf(current_pos)` at each step.  Returns
/// `Some((hit_point, surface_normal))` when the ray comes within `1e-5` of the
/// surface, or `None` if `max_dist` or `max_steps` is exceeded.
///
/// The normal is computed via central differences of the SDF.
pub fn sphere_march(
    surface: &dyn ImplicitSurface,
    ray_origin: [f64; 3],
    ray_dir: [f64; 3],
    max_dist: f64,
    max_steps: usize,
) -> Option<([f64; 3], [f64; 3])> {
    pub(super) const HIT_EPS: f64 = 1e-5;
    let dir = normalize(ray_dir);
    let mut t = 0.0_f64;
    for _ in 0..max_steps {
        let p = add(ray_origin, scale(dir, t));
        let d = surface.sdf(p);
        if d.abs() < HIT_EPS {
            let normal = central_diff_normal(surface, p);
            return Some((p, normal));
        }
        if d < 0.0 {
            t -= d.abs();
            let p2 = add(ray_origin, scale(dir, t));
            let normal = central_diff_normal(surface, p2);
            return Some((p2, normal));
        }
        t += d;
        if t > max_dist {
            return None;
        }
    }
    None
}
/// Sphere march with over-relaxation and a fail-safe against skipped surfaces.
///
/// Like [`sphere_march`], but the step taken from a sample is `omega * sdf`
/// instead of `sdf`, which converges in fewer steps for `omega > 1`.  A relaxed
/// step can jump over a thin feature, so the march keeps the previous sample
/// (ray parameter `t_prev`, distance `d_prev`) and checks the *unbounding
/// spheres* of the two latest samples (Keinert et al., "Enhanced Sphere
/// Tracing", 2014): if `|d| + d_prev` is smaller than the step length the two
/// spheres do not overlap and the step may have skipped a surface.  The march
/// then retreats to `t_prev + d_prev`, the farthest point that is safe, ignores
/// the sample at the discarded position and continues with `omega = 1`.
///
/// A step that ends inside the solid despite overlapping spheres is refined by
/// bisection between the previous sample (outside) and the current one, so the
/// returned point lies on the surface to within `1e-5`.  A ray that starts
/// inside the solid hits at its origin.  `omega = 1` is plain sphere tracing,
/// `0 < omega < 1` under-relaxes the steps, and a non-positive or non-finite
/// `omega` is treated as `1`.
///
/// Returns `Some((hit_point, surface_normal))` (the normal from central
/// differences of the SDF), or `None` if the ray leaves `max_dist`, runs out of
/// `max_steps`, or the SDF returns `NaN`.
pub fn sphere_march_relaxed(
    surface: &dyn ImplicitSurface,
    ray_origin: [f64; 3],
    ray_dir: [f64; 3],
    max_dist: f64,
    max_steps: usize,
    omega: f64,
) -> Option<([f64; 3], [f64; 3])> {
    const HIT_EPS: f64 = 1e-5;
    /// Halvings of the bracket when refining a hit inside the solid: after 64 of
    /// them the bracket is at machine precision for any ray length.
    const BISECTION_STEPS: usize = 64;
    let dir = normalize(ray_dir);
    let point_at = |t: f64| add(ray_origin, scale(dir, t));
    let mut omega = if omega.is_finite() && omega > 0.0 {
        omega
    } else {
        1.0
    };
    let mut t = 0.0_f64;
    let mut t_prev = 0.0_f64;
    let mut d_prev = 0.0_f64;
    for _ in 0..max_steps {
        let d = surface.sdf(point_at(t));
        if d.is_nan() {
            return None;
        }
        if omega > 1.0 && d.abs() + d_prev < t - t_prev {
            // The relaxed step may have skipped the surface: take the safe step
            // from the previous sample instead and stop relaxing.
            t = t_prev + d_prev;
            omega = 1.0;
            continue;
        }
        if d.abs() < HIT_EPS {
            let p = point_at(t);
            return Some((p, central_diff_normal(surface, p)));
        }
        if d < 0.0 {
            // The step ended inside the solid: the surface lies between the
            // previous sample, which is outside, and this one.
            let t_hit = if t > t_prev {
                let (mut outside, mut inside) = (t_prev, t);
                let mut found = None;
                for _ in 0..BISECTION_STEPS {
                    let mid = 0.5 * (outside + inside);
                    let value = surface.sdf(point_at(mid));
                    if value.abs() < HIT_EPS {
                        found = Some(mid);
                        break;
                    }
                    if value < 0.0 {
                        inside = mid;
                    } else {
                        outside = mid;
                    }
                }
                found.unwrap_or(0.5 * (outside + inside))
            } else {
                t
            };
            if t_hit > max_dist {
                return None;
            }
            let p = point_at(t_hit);
            return Some((p, central_diff_normal(surface, p)));
        }
        t_prev = t;
        d_prev = d;
        t += d * omega;
        if t > max_dist {
            return None;
        }
    }
    None
}
/// Compute the SDF gradient at `p` via central differences.
pub(super) fn central_diff_normal(surface: &dyn ImplicitSurface, p: [f64; 3]) -> [f64; 3] {
    pub(super) const EPS: f64 = 1e-4;
    let dx = surface.sdf([p[0] + EPS, p[1], p[2]]) - surface.sdf([p[0] - EPS, p[1], p[2]]);
    let dy = surface.sdf([p[0], p[1] + EPS, p[2]]) - surface.sdf([p[0], p[1] - EPS, p[2]]);
    let dz = surface.sdf([p[0], p[1], p[2] + EPS]) - surface.sdf([p[0], p[1], p[2] - EPS]);
    normalize([dx, dy, dz])
}
/// Test whether a ray hits a CSG object using sphere marching.
///
/// Returns the hit distance along the ray, or `None` on miss.
pub fn csg_ray_test(
    surface: &dyn ImplicitSurface,
    ray_origin: [f64; 3],
    ray_dir: [f64; 3],
    max_dist: f64,
) -> Option<f64> {
    let dir = normalize(ray_dir);
    sphere_march(surface, ray_origin, dir, max_dist, 256).map(|(hit, _)| {
        let d = sub(hit, ray_origin);
        length(d)
    })
}
/// Test whether a point is inside a CSG object (SDF < 0).
pub fn csg_contains_point(surface: &dyn ImplicitSurface, point: [f64; 3]) -> bool {
    surface.sdf(point) < 0.0
}
/// Classify a point with respect to a plane defined by `normal` and `d`.
pub fn classify_point(point: [f64; 3], normal: [f64; 3], d: f64, tolerance: f64) -> PlaneSide {
    let dist = dot(point, normal) - d;
    if dist > tolerance {
        PlaneSide::Front
    } else if dist < -tolerance {
        PlaneSide::Back
    } else {
        PlaneSide::OnPlane
    }
}
/// Classify an entire polygon (triangle) with respect to a plane.
///
/// Returns `(front_count, back_count, on_count)`.
pub fn classify_triangle(
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
    normal: [f64; 3],
    d: f64,
    tolerance: f64,
) -> (usize, usize, usize) {
    let sides = [
        classify_point(v0, normal, d, tolerance),
        classify_point(v1, normal, d, tolerance),
        classify_point(v2, normal, d, tolerance),
    ];
    let front = sides.iter().filter(|&&s| s == PlaneSide::Front).count();
    let back = sides.iter().filter(|&&s| s == PlaneSide::Back).count();
    let on = sides.iter().filter(|&&s| s == PlaneSide::OnPlane).count();
    (front, back, on)
}
/// Clip a polygon (list of vertices) against a plane.
///
/// Returns the vertices on the front side (including intersections with the plane).
/// Uses the Sutherland-Hodgman algorithm variant.
pub fn clip_polygon_front(vertices: &[[f64; 3]], normal: [f64; 3], d: f64) -> Vec<[f64; 3]> {
    if vertices.is_empty() {
        return vec![];
    }
    let mut output = Vec::new();
    let n = vertices.len();
    for i in 0..n {
        let current = vertices[i];
        let next = vertices[(i + 1) % n];
        let d_current = dot(current, normal) - d;
        let d_next = dot(next, normal) - d;
        if d_current >= 0.0 {
            output.push(current);
            if d_next < 0.0 {
                let t = d_current / (d_current - d_next);
                output.push(lerp3(current, next, t));
            }
        } else if d_next >= 0.0 {
            let t = d_current / (d_current - d_next);
            output.push(lerp3(current, next, t));
        }
    }
    output
}
/// Clip a polygon against a plane, keeping the back side.
pub fn clip_polygon_back(vertices: &[[f64; 3]], normal: [f64; 3], d: f64) -> Vec<[f64; 3]> {
    clip_polygon_front(vertices, [-normal[0], -normal[1], -normal[2]], -d)
}
/// Magnitude that non-finite field values are clamped to by [`SdfField`]: large
/// enough to dominate any real distance, small enough that the difference of two
/// clamped values cannot overflow.
const FIELD_LIMIT: f64 = f64::MAX / 4.0;
/// Presents an [`ImplicitSurface`] of this module to the marching-cubes extractor
/// of [`crate::implicit_surfaces`].
///
/// Non-finite field values are mapped to large finite ones (`NaN` and `+inf` read
/// as far outside, `-inf` as deep inside), so that no edge crossing is ever `NaN`.
struct SdfField<'a>(&'a dyn ImplicitSurface);
impl crate::implicit_surfaces::ImplicitSurface for SdfField<'_> {
    fn eval(&self, p: [f64; 3]) -> f64 {
        let value = self.0.sdf(p);
        if value.is_nan() {
            FIELD_LIMIT
        } else {
            value.clamp(-FIELD_LIMIT, FIELD_LIMIT)
        }
    }
    fn gradient(&self, p: [f64; 3]) -> [f64; 3] {
        self.0.gradient(p)
    }
}
/// Extract a triangle mesh of the zero level set of an implicit surface with the
/// marching-cubes algorithm.
///
/// The box `[bounds_min, bounds_max]` is divided into `n = max(resolution, 2)`
/// cells per axis (the cells need not be cubes) and the field is sampled exactly
/// once at each of the `(n + 1)^3` grid nodes.  Triangles come from the standard
/// 256-case Lorensen & Cline table and the vertices are welded: every grid edge
/// that crosses the surface yields one vertex, shared by all triangles around it,
/// so a surface that stays inside the box gives a watertight indexed mesh in which
/// every edge belongs to exactly two triangles.  A surface that leaves the box is
/// left open at the box faces.
///
/// A node is *inside* when `sdf < 0`.  Triangles are wound counter-clockwise when
/// seen from outside, so their normals point towards increasing `sdf`.
///
/// Returns `(vertices, triangles)` where each triangle is `[i0, i1, i2]`.  The
/// mesh is empty if the surface does not cross the box, if a bound is not finite
/// or the box has no extent along an axis (a reversed pair of bounds is swapped),
/// or if the resolution is so large that the vertex ids would not fit 32 bits.
/// The node values and the edge-vertex table of the whole grid are kept in memory,
/// so the memory use grows with `(n + 1)^3`.
pub fn extract_mesh_simple(
    surface: &dyn ImplicitSurface,
    bounds_min: [f64; 3],
    bounds_max: [f64; 3],
    resolution: usize,
) -> VertsTris {
    let finite = bounds_min
        .iter()
        .chain(bounds_max.iter())
        .all(|c| c.is_finite());
    let bounds: [[f64; 2]; 3] = std::array::from_fn(|axis| {
        [
            bounds_min[axis].min(bounds_max[axis]),
            bounds_min[axis].max(bounds_max[axis]),
        ]
    });
    let has_extent = bounds.iter().all(|b| {
        let extent = b[1] - b[0];
        extent.is_finite() && extent > 0.0
    });
    if !(finite && has_extent) {
        return (Vec::new(), Vec::new());
    }
    crate::implicit_surfaces::marching_cubes(&SdfField(surface), bounds, resolution.max(2))
}
/// Sample a regular grid over the given bounding box and return all grid
/// points where `|sdf(p)| < threshold`.
///
/// `resolution` is the number of divisions per axis, so the total number of
/// probed points is `(resolution + 1)^3`.
///
/// The threshold is chosen as half the grid spacing to capture at least one
/// layer of points near the zero-level set.
pub fn sample_surface_points(
    surface: &dyn ImplicitSurface,
    bounds_min: [f64; 3],
    bounds_max: [f64; 3],
    resolution: usize,
) -> Vec<[f64; 3]> {
    let n = resolution.max(1);
    let step = [
        (bounds_max[0] - bounds_min[0]) / n as f64,
        (bounds_max[1] - bounds_min[1]) / n as f64,
        (bounds_max[2] - bounds_min[2]) / n as f64,
    ];
    let threshold = step[0].max(step[1]).max(step[2]) * 0.55;
    let mut points = Vec::new();
    for ix in 0..=n {
        let x = bounds_min[0] + ix as f64 * step[0];
        for iy in 0..=n {
            let y = bounds_min[1] + iy as f64 * step[1];
            for iz in 0..=n {
                let z = bounds_min[2] + iz as f64 * step[2];
                let p = [x, y, z];
                if surface.sdf(p).abs() < threshold {
                    points.push(p);
                }
            }
        }
    }
    points
}
/// Sample surface points with projected normals.
///
/// Returns `(point, normal)` pairs for all near-surface grid points.
pub fn sample_surface_points_with_normals(
    surface: &dyn ImplicitSurface,
    bounds_min: [f64; 3],
    bounds_max: [f64; 3],
    resolution: usize,
) -> PointNormals {
    let points = sample_surface_points(surface, bounds_min, bounds_max, resolution);
    points
        .into_iter()
        .map(|p| {
            let n = surface.gradient(p);
            (p, n)
        })
        .collect()
}
/// Compute the approximate volume of the interior region using grid sampling.
///
/// Returns the volume estimate (count of interior cells * cell_volume).
pub fn estimate_volume(
    surface: &dyn ImplicitSurface,
    bounds_min: [f64; 3],
    bounds_max: [f64; 3],
    resolution: usize,
) -> f64 {
    let n = resolution.max(1);
    let step = [
        (bounds_max[0] - bounds_min[0]) / n as f64,
        (bounds_max[1] - bounds_min[1]) / n as f64,
        (bounds_max[2] - bounds_min[2]) / n as f64,
    ];
    let cell_volume = step[0] * step[1] * step[2];
    let mut count = 0usize;
    for ix in 0..=n {
        let x = bounds_min[0] + ix as f64 * step[0];
        for iy in 0..=n {
            let y = bounds_min[1] + iy as f64 * step[1];
            for iz in 0..=n {
                let z = bounds_min[2] + iz as f64 * step[2];
                if surface.sdf([x, y, z]) < 0.0 {
                    count += 1;
                }
            }
        }
    }
    count as f64 * cell_volume
}
/// Evaluate the SDF of a `CsgTree` at point `p`.
///
/// Recursively descends the tree, combining child SDF values according to the
/// boolean operation at each interior node.
pub fn csg_sdf(node: &CsgTree, p: [f64; 3]) -> f64 {
    match node {
        CsgTree::Leaf(s) => s.sdf(p),
        CsgTree::Node { op, left, right } => {
            let da = csg_sdf(left, p);
            let db = csg_sdf(right, p);
            match op {
                CsgOp::Union => da.min(db),
                CsgOp::Intersection => da.max(db),
                CsgOp::Difference => da.max(-db),
            }
        }
    }
}
/// Compute the gradient (approximate unit normal) of a `CsgTree` SDF at `p`.
///
/// Uses central differences of [`csg_sdf`].
pub fn csg_gradient(node: &CsgTree, p: [f64; 3]) -> [f64; 3] {
    pub(super) const EPS: f64 = 1e-4;
    let dx = csg_sdf(node, [p[0] + EPS, p[1], p[2]]) - csg_sdf(node, [p[0] - EPS, p[1], p[2]]);
    let dy = csg_sdf(node, [p[0], p[1] + EPS, p[2]]) - csg_sdf(node, [p[0], p[1] - EPS, p[2]]);
    let dz = csg_sdf(node, [p[0], p[1], p[2] + EPS]) - csg_sdf(node, [p[0], p[1], p[2] - EPS]);
    normalize([dx, dy, dz])
}
/// Sphere-march a ray against a `CsgTree`.
///
/// Advances the ray by `csg_sdf(node, current_pos)` at each step.
/// Returns `Some(t)` — the ray parameter of the hit — when the SDF drops below
/// `1e-4`, or `None` if `max_dist` or `max_steps` is exceeded.
pub fn csg_ray_march(
    node: &CsgTree,
    origin: [f64; 3],
    dir: [f64; 3],
    max_dist: f64,
    max_steps: usize,
) -> Option<f64> {
    pub(super) const HIT_EPS: f64 = 1e-4;
    let d = normalize(dir);
    let mut t = 0.0_f64;
    for _ in 0..max_steps {
        let p = add(origin, scale(d, t));
        let dist = csg_sdf(node, p);
        if dist.abs() < HIT_EPS {
            return Some(t);
        }
        if dist < 0.0 {
            return Some(t);
        }
        t += dist;
        if t > max_dist {
            return None;
        }
    }
    None
}
/// Test whether a world-space point is inside a `CsgTree` (SDF < 0).
pub fn csg_tree_contains(node: &CsgTree, p: [f64; 3]) -> bool {
    csg_sdf(node, p) < 0.0
}
/// Estimate the surface area of the zero level set of a `CsgTree` on a grid.
///
/// The box `[bbox_min, bbox_max]` is divided into `max(n_div, 2)` cells per axis
/// and the field is sampled once at every cell centre.  Two neighbouring samples
/// of different sign (inside is negative) mean that the surface crosses the grid
/// edge between them.  The edge along axis `a` stands for the cross-section
/// `A_a` of a cell perpendicular to `a`, and the crossing is weighted by the
/// component of the unit surface normal along that axis: it adds `A_a * |n_a|`,
/// with the crossing point found by linear interpolation of the field between
/// the two samples and the normal from central differences of the field
/// ([`csg_gradient`]) there.
///
/// Surface crossings of the edges along `a` occur with a density of
/// `|n_a| / A_a` per unit of surface area, so the sum over the crossings
/// converges to the integral of `n_a^2` over the surface; the three axes together
/// give the integral of `n_x^2 + n_y^2 + n_z^2 = 1`, which is the area.  Counting
/// every crossing with its full cross-section, as an unweighted sign-change count
/// does, would integrate `|n_x| + |n_y| + |n_z|` instead and overestimate the
/// area (by a factor of 1.5 for a sphere).  The estimate is a deterministic
/// quadrature, not a Monte-Carlo sample, and approaches the area as `n_div`
/// grows; the part of the surface outside the box is not counted.
///
/// The cost is one field evaluation per cell plus a few per crossing; the
/// samples are held two layers at a time, so the memory use is `O(n_div^2)`.
/// A non-finite bound or a box without extent along an axis gives `0.0`.  The
/// normals come from [`csg_gradient`], whose finite-difference step is `1e-4`, so
/// the cells should be large compared with that.
pub fn csg_surface_area(
    node: &CsgTree,
    bbox_min: [f64; 3],
    bbox_max: [f64; 3],
    n_div: usize,
) -> f64 {
    let n = n_div.max(2);
    let lower: [f64; 3] = std::array::from_fn(|axis| bbox_min[axis].min(bbox_max[axis]));
    let upper: [f64; 3] = std::array::from_fn(|axis| bbox_min[axis].max(bbox_max[axis]));
    let step: [f64; 3] = std::array::from_fn(|axis| (upper[axis] - lower[axis]) / n as f64);
    if step.iter().any(|s| !s.is_finite() || *s <= 0.0) {
        return 0.0;
    }
    let cell_area = [step[1] * step[2], step[0] * step[2], step[0] * step[1]];
    let centre = |ix: usize, iy: usize, iz: usize| -> [f64; 3] {
        [
            lower[0] + (ix as f64 + 0.5) * step[0],
            lower[1] + (iy as f64 + 0.5) * step[1],
            lower[2] + (iz as f64 + 0.5) * step[2],
        ]
    };
    // Field values of one layer of cell centres, indexed `iy * n + iz`.
    let sample_layer = |ix: usize| -> Vec<f64> {
        (0..n * n)
            .map(|cell| csg_sdf(node, centre(ix, cell / n, cell % n)))
            .collect()
    };
    // Area contributed by the grid edge from `p` (field `d`) to `pn` (field `dn`)
    // along `axis`: zero unless the sign changes.
    let crossing_area = |p: [f64; 3], d: f64, pn: [f64; 3], dn: f64, axis: usize| -> f64 {
        if (d < 0.0) == (dn < 0.0) {
            return 0.0;
        }
        let t = d / (d - dn);
        let at_crossing = lerp3(p, pn, t);
        cell_area[axis] * csg_gradient(node, at_crossing)[axis].abs()
    };
    let mut area = 0.0_f64;
    let mut layer = sample_layer(0);
    for ix in 0..n {
        let next = (ix + 1 < n).then(|| sample_layer(ix + 1));
        for iy in 0..n {
            for iz in 0..n {
                let cell = iy * n + iz;
                let d = layer[cell];
                let p = centre(ix, iy, iz);
                if let Some(next) = &next {
                    area += crossing_area(p, d, centre(ix + 1, iy, iz), next[cell], 0);
                }
                if iy + 1 < n {
                    area += crossing_area(p, d, centre(ix, iy + 1, iz), layer[cell + n], 1);
                }
                if iz + 1 < n {
                    area += crossing_area(p, d, centre(ix, iy, iz + 1), layer[cell + 1], 2);
                }
            }
        }
        if let Some(next) = next {
            layer = next;
        }
    }
    area
}
/// Evaluate the SDF of an offset `CsgTree` at point `p`.
///
/// `f_offset(p) = csg_sdf(node, p) - offset`
pub fn csg_offset_sdf(node: &CsgTree, p: [f64; 3], offset: f64) -> f64 {
    csg_sdf(node, p) - offset
}
/// Returns `true` if a point lies inside the offset surface.
pub fn csg_offset_contains(node: &CsgTree, p: [f64; 3], offset: f64) -> bool {
    csg_offset_sdf(node, p, offset) < 0.0
}
/// Compute the gradient of the offset SDF (same direction as the original,
/// since the offset is a constant shift of the level set).
pub fn csg_offset_gradient(node: &CsgTree, p: [f64; 3]) -> [f64; 3] {
    csg_gradient(node, p)
}
/// A simplified representation of a `CsgTree` — here we perform constant
/// folding: if the root is a Union/Intersection/Difference of two leaves
/// whose bounding boxes do not overlap, we can determine the result
/// analytically.
///
/// Returns `true` if the tree was recognized as a trivially-simplified form
/// (the caller can then use a simpler primitive).  This is a best-effort
/// static analysis pass.
///
/// Strategy:
/// - A `Union(A, B)` where neither A nor B can overlap (conservative check
///   via sample points) collapses to the one that is "larger".
/// - An `Intersection(A, B)` where the two are disjoint collapses to empty.
///
/// For the general case this returns `false` and the original tree is used.
pub fn csg_tree_simplify_check(node: &CsgTree, probe_center: [f64; 3]) -> bool {
    match node {
        CsgTree::Leaf(_) => true,
        CsgTree::Node { op, left, right } => {
            let da = csg_sdf(left, probe_center);
            let db = csg_sdf(right, probe_center);
            match op {
                CsgOp::Union => da < 0.0 || db < 0.0,
                CsgOp::Intersection => da < 0.0 && db < 0.0,
                CsgOp::Difference => da < 0.0 && db >= 0.0,
            }
        }
    }
}
/// Recursive surface area estimator for a `CsgTree` using a user-supplied bounding box.
///
/// Delegates to `csg_surface_area` with a default resolution of 16 divisions.
pub fn csg_node_surface_area(node: &CsgTree, bbox_min: [f64; 3], bbox_max: [f64; 3]) -> f64 {
    csg_surface_area(node, bbox_min, bbox_max, 16)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::csg::CsgDifference;
    use crate::csg::CsgIntersection;
    use crate::csg::CsgSmoothDifference;
    use crate::csg::CsgSmoothIntersection;
    use crate::csg::CsgSmoothUnion;

    use crate::csg::CsgUnion;
    use crate::csg::SdfBox;
    use crate::csg::SdfCylinder;
    use crate::csg::SdfSphere;
    use crate::csg::SdfTorus;

    #[test]
    fn sphere_sdf_center_is_minus_radius() {
        let s = SdfSphere::new([0.0, 0.0, 0.0], 2.0);
        assert!((s.sdf([0.0, 0.0, 0.0]) - (-2.0)).abs() < 1e-12);
    }
    #[test]
    fn sphere_sdf_on_surface_is_zero() {
        let s = SdfSphere::new([0.0, 0.0, 0.0], 1.5);
        assert!(s.sdf([1.5, 0.0, 0.0]).abs() < 1e-12);
    }
    #[test]
    fn sphere_sdf_outside_is_positive() {
        let s = SdfSphere::new([0.0, 0.0, 0.0], 1.0);
        assert!(s.sdf([3.0, 0.0, 0.0]) > 0.0);
    }
    #[test]
    fn union_inside_either_sphere_is_negative() {
        let a = Box::new(SdfSphere::new([-2.0, 0.0, 0.0], 1.0));
        let b = Box::new(SdfSphere::new([2.0, 0.0, 0.0], 1.0));
        let u = CsgUnion::new(a, b);
        assert!(u.sdf([-2.0, 0.0, 0.0]) < 0.0);
        assert!(u.sdf([2.0, 0.0, 0.0]) < 0.0);
        assert!(u.sdf([0.0, 0.0, 0.0]) > 0.0);
    }
    #[test]
    fn intersection_only_inside_both_is_negative() {
        let a = Box::new(SdfSphere::new([0.0, 0.0, 0.0], 2.0));
        let b = Box::new(SdfSphere::new([1.0, 0.0, 0.0], 2.0));
        let i = CsgIntersection::new(a, b);
        assert!(i.sdf([0.5, 0.0, 0.0]) < 0.0);
        assert!(i.sdf([-1.5, 0.0, 0.0]) > 0.0);
        assert!(i.sdf([2.5, 0.0, 0.0]) > 0.0);
    }
    #[test]
    fn difference_carved_out_region_is_positive() {
        let a = Box::new(SdfSphere::new([0.0, 0.0, 0.0], 3.0));
        let b = Box::new(SdfSphere::new([0.0, 0.0, 0.0], 1.5));
        let d = CsgDifference::new(a, b);
        assert!(d.sdf([0.0, 0.0, 0.0]) > 0.0);
        assert!(d.sdf([2.5, 0.0, 0.0]) < 0.0);
        assert!(d.sdf([5.0, 0.0, 0.0]) > 0.0);
    }
    #[test]
    fn smooth_union_blends_surfaces() {
        let a = Box::new(SdfSphere::new([-0.5, 0.0, 0.0], 1.0));
        let b = Box::new(SdfSphere::new([0.5, 0.0, 0.0], 1.0));
        let su = CsgSmoothUnion::new(a, b, 0.5);
        let sharp_a = Box::new(SdfSphere::new([-0.5, 0.0, 0.0], 1.0));
        let sharp_b = Box::new(SdfSphere::new([0.5, 0.0, 0.0], 1.0));
        let sharp = CsgUnion::new(sharp_a, sharp_b);
        assert!(su.sdf([0.0, 0.0, 0.0]) <= sharp.sdf([0.0, 0.0, 0.0]) + 1e-10);
    }
    #[test]
    fn smooth_intersection_blends() {
        let a = Box::new(SdfSphere::new([0.0, 0.0, 0.0], 2.0));
        let b = Box::new(SdfSphere::new([1.0, 0.0, 0.0], 2.0));
        let si = CsgSmoothIntersection::new(a, b, 0.3);
        assert!(si.sdf([0.5, 0.0, 0.0]) < 0.0);
    }
    #[test]
    fn smooth_difference_carves() {
        let a = Box::new(SdfSphere::new([0.0, 0.0, 0.0], 3.0));
        let b = Box::new(SdfSphere::new([0.0, 0.0, 0.0], 1.0));
        let sd = CsgSmoothDifference::new(a, b, 0.2);
        assert!(sd.sdf([0.0, 0.0, 0.0]) > 0.0);
        assert!(sd.sdf([2.5, 0.0, 0.0]) < 0.0);
    }
    #[test]
    fn sphere_march_hits_sphere() {
        let s = SdfSphere::new([0.0, 0.0, 0.0], 1.0);
        let origin = [-5.0, 0.0, 0.0];
        let dir = [1.0, 0.0, 0.0];
        let result = sphere_march(&s, origin, dir, 100.0, 256);
        assert!(result.is_some(), "expected a hit");
        let (hit, _normal) = result.unwrap();
        assert!(
            s.sdf(hit).abs() < 1e-4,
            "hit point not on surface: sdf={}",
            s.sdf(hit)
        );
    }
    #[test]
    fn sphere_march_misses_sphere() {
        let s = SdfSphere::new([0.0, 0.0, 0.0], 1.0);
        let origin = [-5.0, 5.0, 0.0];
        let dir = [1.0, 0.0, 0.0];
        let result = sphere_march(&s, origin, dir, 20.0, 256);
        assert!(result.is_none());
    }
    #[test]
    fn sphere_march_relaxed_hits() {
        let s = SdfSphere::new([0.0, 0.0, 0.0], 1.0);
        let result = sphere_march_relaxed(&s, [-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0, 256, 1.5);
        assert!(result.is_some());
    }
    #[test]
    fn box_sdf_corner_region() {
        let b = SdfBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(b.sdf([0.0, 0.0, 0.0]) < 0.0);
        assert!(b.sdf([1.0, 0.0, 0.0]).abs() < 1e-12);
        let p = [2.0, 2.0, 2.0];
        let expected = ((1.0_f64).powi(2) * 3.0).sqrt();
        assert!((b.sdf(p) - expected).abs() < 1e-10);
    }
    #[test]
    fn cylinder_sdf_on_axis() {
        let c = SdfCylinder::new([0.0, 0.0, 0.0], 1.0);
        assert!((c.sdf([0.0, 0.0, 0.0]) - (-1.0)).abs() < 1e-12);
    }
    #[test]
    fn cylinder_sdf_on_surface() {
        let c = SdfCylinder::new([0.0, 0.0, 0.0], 1.0);
        assert!(c.sdf([1.0, 5.0, 0.0]).abs() < 1e-12);
    }
    #[test]
    fn torus_sdf_on_surface() {
        let t = SdfTorus::new(2.0, 0.5);
        assert!(t.sdf([2.5, 0.0, 0.0]).abs() < 1e-12);
        assert!(t.sdf([1.5, 0.0, 0.0]).abs() < 1e-12);
    }
    #[test]
    fn sample_surface_points_returns_nonempty_for_sphere() {
        let s = SdfSphere::new([0.0, 0.0, 0.0], 0.5);
        let pts = sample_surface_points(&s, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], 20);
        assert!(!pts.is_empty(), "expected surface points, got none");
    }
    #[test]
    fn sample_surface_points_with_normals_has_unit_normals() {
        let s = SdfSphere::new([0.0, 0.0, 0.0], 1.0);
        let pts = sample_surface_points_with_normals(&s, [-1.5, -1.5, -1.5], [1.5, 1.5, 1.5], 10);
        for (_, n) in &pts {
            let len = length(*n);
            assert!((len - 1.0).abs() < 0.1, "normal length = {len}");
        }
    }
    #[test]
    fn estimate_volume_sphere() {
        let s = SdfSphere::new([0.0, 0.0, 0.0], 1.0);
        let vol = estimate_volume(&s, [-1.5, -1.5, -1.5], [1.5, 1.5, 1.5], 20);
        let expected = 4.0 / 3.0 * std::f64::consts::PI;
        assert!(
            (vol - expected).abs() < 1.0,
            "estimated volume = {vol}, expected ~ {expected}"
        );
    }
    #[test]
    fn csg_ray_test_hits() {
        let s = SdfSphere::new([0.0, 0.0, 0.0], 1.0);
        let dist = csg_ray_test(&s, [-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0);
        assert!(dist.is_some());
        let d = dist.unwrap();
        assert!((d - 4.0).abs() < 0.1, "hit distance = {d}");
    }
    #[test]
    fn csg_contains_point_inside() {
        let s = SdfSphere::new([0.0, 0.0, 0.0], 1.0);
        assert!(csg_contains_point(&s, [0.0, 0.0, 0.0]));
        assert!(!csg_contains_point(&s, [2.0, 0.0, 0.0]));
    }
    #[test]
    fn classify_point_front_back() {
        let normal = [0.0, 1.0, 0.0];
        let d = 0.0;
        assert_eq!(
            classify_point([0.0, 1.0, 0.0], normal, d, 1e-6),
            PlaneSide::Front
        );
        assert_eq!(
            classify_point([0.0, -1.0, 0.0], normal, d, 1e-6),
            PlaneSide::Back
        );
        assert_eq!(
            classify_point([0.0, 0.0, 0.0], normal, d, 1e-6),
            PlaneSide::OnPlane
        );
    }
    #[test]
    fn classify_triangle_straddling() {
        let normal = [0.0, 1.0, 0.0];
        let d = 0.0;
        let (f, b, _on) = classify_triangle(
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            normal,
            d,
            1e-6,
        );
        assert_eq!(f, 1);
        assert_eq!(b, 1);
    }
    #[test]
    fn clip_polygon_front_basic() {
        let tri = [[0.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]];
        let result = clip_polygon_front(&tri, [0.0, 1.0, 0.0], 0.0);
        assert!(
            result.len() >= 3,
            "clipped polygon has {} verts",
            result.len()
        );
        for v in &result {
            assert!(dot(*v, [0.0, 1.0, 0.0]) >= -1e-10);
        }
    }
    #[test]
    fn clip_polygon_back_basic() {
        let tri = [[0.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]];
        let result = clip_polygon_back(&tri, [0.0, 1.0, 0.0], 0.0);
        assert!(result.len() >= 3);
        for v in &result {
            assert!(dot(*v, [0.0, 1.0, 0.0]) <= 1e-10);
        }
    }
    #[test]
    fn extract_mesh_simple_nonempty() {
        let s = SdfSphere::new([0.0, 0.0, 0.0], 1.0);
        let (verts, _tris) = extract_mesh_simple(&s, [-1.5, -1.5, -1.5], [1.5, 1.5, 1.5], 20);
        assert!(!verts.is_empty(), "mesh has no vertices");
    }
}
#[cfg(test)]
mod tests_csg_tree {

    use crate::csg::CsgTree;

    use crate::csg::SdfSphere;

    use crate::csg::csg_gradient;
    use crate::csg::csg_ray_march;
    use crate::csg::csg_sdf;
    use crate::csg::csg_tree_contains;
    #[test]
    fn csg_tree_leaf_sphere_sdf() {
        let tree = CsgTree::leaf(SdfSphere::new([0.0, 0.0, 0.0], 1.0));
        assert!(csg_sdf(&tree, [0.0, 0.0, 0.0]) < 0.0);
        assert!(csg_sdf(&tree, [3.0, 0.0, 0.0]) > 0.0);
        assert!(csg_sdf(&tree, [1.0, 0.0, 0.0]).abs() < 1e-12);
    }
    #[test]
    fn csg_tree_union_inside_either() {
        let a = CsgTree::leaf(SdfSphere::new([-2.0, 0.0, 0.0], 1.0));
        let b = CsgTree::leaf(SdfSphere::new([2.0, 0.0, 0.0], 1.0));
        let u = CsgTree::union(a, b);
        assert!(csg_sdf(&u, [-2.0, 0.0, 0.0]) < 0.0);
        assert!(csg_sdf(&u, [2.0, 0.0, 0.0]) < 0.0);
        assert!(csg_sdf(&u, [0.0, 0.0, 0.0]) > 0.0);
    }
    #[test]
    fn csg_tree_intersection_only_shared_region() {
        let a = CsgTree::leaf(SdfSphere::new([0.0, 0.0, 0.0], 2.0));
        let b = CsgTree::leaf(SdfSphere::new([1.0, 0.0, 0.0], 2.0));
        let i = CsgTree::intersection(a, b);
        assert!(csg_sdf(&i, [0.5, 0.0, 0.0]) < 0.0);
        assert!(csg_sdf(&i, [-1.5, 0.0, 0.0]) > 0.0);
        assert!(csg_sdf(&i, [2.5, 0.0, 0.0]) > 0.0);
    }
    #[test]
    fn csg_tree_difference_carves_out() {
        let a = CsgTree::leaf(SdfSphere::new([0.0, 0.0, 0.0], 3.0));
        let b = CsgTree::leaf(SdfSphere::new([0.0, 0.0, 0.0], 1.5));
        let d = CsgTree::difference(a, b);
        assert!(csg_sdf(&d, [0.0, 0.0, 0.0]) > 0.0);
        assert!(csg_sdf(&d, [2.5, 0.0, 0.0]) < 0.0);
        assert!(csg_sdf(&d, [5.0, 0.0, 0.0]) > 0.0);
    }
    #[test]
    fn csg_tree_nested_union_of_three_spheres() {
        let a = CsgTree::leaf(SdfSphere::new([-3.0, 0.0, 0.0], 1.0));
        let b = CsgTree::leaf(SdfSphere::new([0.0, 0.0, 0.0], 1.0));
        let c = CsgTree::leaf(SdfSphere::new([3.0, 0.0, 0.0], 1.0));
        let ab = CsgTree::union(a, b);
        let abc = CsgTree::union(ab, c);
        assert!(csg_sdf(&abc, [-3.0, 0.0, 0.0]) < 0.0);
        assert!(csg_sdf(&abc, [0.0, 0.0, 0.0]) < 0.0);
        assert!(csg_sdf(&abc, [3.0, 0.0, 0.0]) < 0.0);
        assert!(csg_sdf(&abc, [1.5, 0.0, 0.0]) > 0.0);
    }
    #[test]
    fn csg_gradient_leaf_sphere_is_unit() {
        let tree = CsgTree::leaf(SdfSphere::new([0.0, 0.0, 0.0], 1.0));
        let g = csg_gradient(&tree, [2.0, 0.0, 0.0]);
        let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
        assert!((len - 1.0).abs() < 0.01, "gradient length = {len}");
    }
    #[test]
    fn csg_ray_march_hits_sphere() {
        let tree = CsgTree::leaf(SdfSphere::new([0.0, 0.0, 0.0], 1.0));
        let t = csg_ray_march(&tree, [-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0, 256);
        assert!(t.is_some(), "should hit sphere");
        let t = t.unwrap();
        assert!((t - 4.0).abs() < 0.1, "expected t≈4, got {t}");
    }
    #[test]
    fn csg_ray_march_misses_sphere() {
        let tree = CsgTree::leaf(SdfSphere::new([0.0, 0.0, 0.0], 1.0));
        let t = csg_ray_march(&tree, [-5.0, 10.0, 0.0], [1.0, 0.0, 0.0], 20.0, 256);
        assert!(t.is_none(), "should miss sphere");
    }
    #[test]
    fn csg_ray_march_union_hits_either() {
        let a = CsgTree::leaf(SdfSphere::new([-5.0, 0.0, 0.0], 1.0));
        let b = CsgTree::leaf(SdfSphere::new([5.0, 0.0, 0.0], 1.0));
        let u = CsgTree::union(a, b);
        let t = csg_ray_march(&u, [-10.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0, 512);
        assert!(t.is_some(), "should hit union");
    }
    #[test]
    fn csg_ray_march_difference_misses_carved_region() {
        let big = CsgTree::leaf(SdfSphere::new([0.0, 0.0, 0.0], 3.0));
        let cut = CsgTree::leaf(SdfSphere::new([0.0, 0.0, 0.0], 2.9));
        let shell = CsgTree::difference(big, cut);
        let t = csg_ray_march(&shell, [-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], 20.0, 512);
        assert!(t.is_some(), "should hit shell");
    }
    #[test]
    fn csg_tree_contains_inside() {
        let tree = CsgTree::leaf(SdfSphere::new([0.0, 0.0, 0.0], 2.0));
        assert!(csg_tree_contains(&tree, [0.0, 0.0, 0.0]));
        assert!(!csg_tree_contains(&tree, [3.0, 0.0, 0.0]));
    }
    #[test]
    fn csg_tree_contains_difference() {
        let a = CsgTree::leaf(SdfSphere::new([0.0, 0.0, 0.0], 3.0));
        let b = CsgTree::leaf(SdfSphere::new([0.0, 0.0, 0.0], 1.5));
        let d = CsgTree::difference(a, b);
        assert!(!csg_tree_contains(&d, [0.0, 0.0, 0.0]));
        assert!(csg_tree_contains(&d, [2.5, 0.0, 0.0]));
    }
}
/// Compute a conservative axis-aligned bounding box for a `CsgTree`.
///
/// For leaf nodes, the bounding box must be provided via a user-supplied
/// `bounds` function.  For combination nodes:
/// - **Union**: union of child bounding boxes.
/// - **Intersection**: intersection of child bounding boxes (may be empty).
/// - **Difference**: use the left child's bounding box (conservative).
pub fn csg_tree_aabb<F>(node: &CsgTree, bounds: &F) -> ([f64; 3], [f64; 3])
where
    F: Fn(&dyn ImplicitSurface) -> SurfaceAabb,
{
    match node {
        CsgTree::Leaf(s) => bounds(s.as_ref()),
        CsgTree::Node { op, left, right } => {
            let (l_min, l_max) = csg_tree_aabb(left, bounds);
            let (r_min, r_max) = csg_tree_aabb(right, bounds);
            match op {
                CsgOp::Union => {
                    let mn = [
                        l_min[0].min(r_min[0]),
                        l_min[1].min(r_min[1]),
                        l_min[2].min(r_min[2]),
                    ];
                    let mx = [
                        l_max[0].max(r_max[0]),
                        l_max[1].max(r_max[1]),
                        l_max[2].max(r_max[2]),
                    ];
                    (mn, mx)
                }
                CsgOp::Intersection => {
                    let mn = [
                        l_min[0].max(r_min[0]),
                        l_min[1].max(r_min[1]),
                        l_min[2].max(r_min[2]),
                    ];
                    let mx = [
                        l_max[0].min(r_max[0]),
                        l_max[1].min(r_max[1]),
                        l_max[2].min(r_max[2]),
                    ];
                    (mn, mx)
                }
                CsgOp::Difference => (l_min, l_max),
            }
        }
    }
}
/// Sample an SDF on a regular grid and collect all `MarchingCell`s that
/// contain a surface crossing.
///
/// `resolution` is the number of cells per axis (at least 2), so the grid has
/// `resolution^3` cells and `(resolution + 1)^3` corners.  The cells divide the
/// box `[bounds_min, bounds_max]` evenly along each axis, so a box that is longer
/// along one axis gets longer cells there (see [`MarchingCell::step`]) and the
/// whole box is covered.  The field is evaluated exactly once at every corner,
/// not once per cell corner, and the corner layers are held two at a time, so the
/// memory use apart from the returned cells is `O(resolution^2)`.  A cell is
/// returned if its corners are not all on the same side of the surface (inside is
/// negative).  A non-finite bound or a box without extent along an axis gives no
/// cells.
pub fn marching_cubes_sample(
    surface: &dyn ImplicitSurface,
    bounds_min: [f64; 3],
    bounds_max: [f64; 3],
    resolution: usize,
) -> Vec<MarchingCell> {
    let n = resolution.max(2);
    let lower: [f64; 3] = std::array::from_fn(|axis| bounds_min[axis].min(bounds_max[axis]));
    let upper: [f64; 3] = std::array::from_fn(|axis| bounds_min[axis].max(bounds_max[axis]));
    let step: [f64; 3] = std::array::from_fn(|axis| (upper[axis] - lower[axis]) / n as f64);
    if step.iter().any(|s| !s.is_finite() || *s <= 0.0) {
        return Vec::new();
    }
    let corners = n + 1;
    // Field values of one layer of corners, indexed `iy * corners + iz`.
    let sample_layer = |ix: usize| -> Vec<f64> {
        let x = lower[0] + ix as f64 * step[0];
        let mut values = Vec::with_capacity(corners * corners);
        for iy in 0..corners {
            let y = lower[1] + iy as f64 * step[1];
            for iz in 0..corners {
                let z = lower[2] + iz as f64 * step[2];
                values.push(surface.sdf([x, y, z]));
            }
        }
        values
    };
    let mut cells = Vec::new();
    let mut near = sample_layer(0);
    for ix in 0..n {
        let far = sample_layer(ix + 1);
        let x0 = lower[0] + ix as f64 * step[0];
        for iy in 0..n {
            let y0 = lower[1] + iy as f64 * step[1];
            for iz in 0..n {
                let z0 = lower[2] + iz as f64 * step[2];
                let at = |layer: &[f64], dy: usize, dz: usize| layer[(iy + dy) * corners + iz + dz];
                // Corner order of the Lorensen-Cline marching-cubes tables.
                let corner_values = [
                    at(&near, 0, 0),
                    at(&far, 0, 0),
                    at(&far, 1, 0),
                    at(&near, 1, 0),
                    at(&near, 0, 1),
                    at(&far, 0, 1),
                    at(&far, 1, 1),
                    at(&near, 1, 1),
                ];
                let cell = MarchingCell {
                    min: [x0, y0, z0],
                    step,
                    corner_values,
                };
                if cell.has_surface() {
                    cells.push(cell);
                }
            }
        }
        near = far;
    }
    cells
}
/// Extract a flat list of surface vertex positions from a marching-cubes grid.
///
/// Returns all edge-crossing positions (one per sign-change edge per cell).
pub fn marching_cubes_vertices(
    surface: &dyn ImplicitSurface,
    bounds_min: [f64; 3],
    bounds_max: [f64; 3],
    resolution: usize,
) -> Vec<[f64; 3]> {
    let cells = marching_cubes_sample(surface, bounds_min, bounds_max, resolution);
    cells.iter().flat_map(|c| c.extract_vertices()).collect()
}
