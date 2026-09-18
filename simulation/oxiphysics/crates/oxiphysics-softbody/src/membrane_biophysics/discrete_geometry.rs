// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Discrete differential geometry of triangle meshes for the membrane models.
//!
//! * [`SurfaceOperators`] — cotangent Laplace-Beltrami operator together with
//!   the mixed Voronoi vertex areas of Meyer, Desbrun, Schroeder and Barr
//!   (2003), the interior-angle sums and the mesh-boundary flags.
//! * [`vertex_curvatures`] — per-vertex unit normal, area, mean curvature and
//!   Gaussian curvature derived from those operators.
//! * [`icosphere_mesh`] — deterministic subdivided-icosahedron sphere.
//!
//! All tolerances are relative (to edge lengths or accumulated areas), so the
//! results do not depend on the length unit: a vesicle of 50 nm and one of
//! 50 um are treated identically.

use std::collections::HashMap;
use std::f64::consts::PI;

use super::{Vec3, cross3, dot3, norm3, scale3, sub3, triangle_normal};

/// A triangle whose doubled area is below this fraction of its squared longest
/// edge is considered degenerate (its corner angles and cotangents are not
/// defined) and does not contribute to any operator.
const DEGENERATE_SINE: f64 = 1e-12;

/// Corner angles, cotangents and mixed Voronoi vertex areas of one triangle.
struct TriangleMeasure {
    /// Interior angle at each corner (rad).
    angles: [f64; 3],
    /// Cotangent of the interior angle at each corner.
    cotangents: [f64; 3],
    /// Share of the triangle's area assigned to each corner (m^2).
    corner_areas: [f64; 3],
}

impl TriangleMeasure {
    /// Measure a triangle; `None` when it is degenerate.
    fn new(corners: [Vec3; 3]) -> Option<Self> {
        // edge[k] is the edge opposite corner k.
        let edge = [
            sub3(corners[2], corners[1]),
            sub3(corners[0], corners[2]),
            sub3(corners[1], corners[0]),
        ];
        let length_sq = [
            dot3(edge[0], edge[0]),
            dot3(edge[1], edge[1]),
            dot3(edge[2], edge[2]),
        ];
        let twice_area = norm3(cross3(edge[1], edge[2]));
        let longest_sq = length_sq[0].max(length_sq[1]).max(length_sq[2]);
        let non_degenerate = twice_area > DEGENERATE_SINE * longest_sq;
        if !non_degenerate {
            return None;
        }

        // |to_next x to_prev| equals the doubled area at every corner.
        let mut angles = [0.0; 3];
        let mut cotangents = [0.0; 3];
        for (k, (angle, cotangent)) in angles.iter_mut().zip(cotangents.iter_mut()).enumerate() {
            let to_next = sub3(corners[(k + 1) % 3], corners[k]);
            let to_prev = sub3(corners[(k + 2) % 3], corners[k]);
            let cosine_part = dot3(to_next, to_prev);
            *cotangent = cosine_part / twice_area;
            *angle = twice_area.atan2(cosine_part);
        }

        let area = 0.5 * twice_area;
        let corner_areas = match cotangents.iter().position(|&cot| cot < 0.0) {
            // Obtuse corner: it gets half of the area, the others a quarter each.
            Some(obtuse) => {
                let mut shares = [0.25 * area; 3];
                shares[obtuse] = 0.5 * area;
                shares
            }
            // Non-obtuse: exact Voronoi share (1/8) sum |edge|^2 cot(opposite angle)
            // over the two edges meeting at the corner.
            None => std::array::from_fn(|k| {
                let next = (k + 2) % 3; // edge k -> k+1 is opposite corner k+2
                let prev = (k + 1) % 3; // edge k -> k+2 is opposite corner k+1
                0.125 * (length_sq[next] * cotangents[next] + length_sq[prev] * cotangents[prev])
            }),
        };
        Some(Self {
            angles,
            cotangents,
            corner_areas,
        })
    }
}

/// Flags the vertices that lie on a boundary edge (an edge used by exactly one
/// triangle).
fn boundary_vertices(vertex_count: usize, triangles: &[[usize; 3]]) -> Vec<bool> {
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(3 * triangles.len());
    for tri in triangles {
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            edges.push((a.min(b), a.max(b)));
        }
    }
    edges.sort_unstable();
    let mut on_boundary = vec![false; vertex_count];
    for run in edges.chunk_by(|a, b| a == b) {
        if let [(a, b)] = *run {
            on_boundary[a] = true;
            on_boundary[b] = true;
        }
    }
    on_boundary
}

/// Cotangent-discretised surface operators of a triangle mesh.
///
/// For vertex `i` with one-ring neighbours `j` the Laplace-Beltrami operator
/// is `(Delta f)_i = 1 / (2 A_i) sum_j (cot a_ij + cot b_ij) (f_j - f_i)`,
/// where `a_ij`, `b_ij` are the two angles opposite the edge `ij` and `A_i` is
/// the mixed Voronoi area. The cotangents are signed (not clamped), so the
/// operator reproduces linear functions exactly on any flat mesh and is
/// positive semi-definite as a stiffness matrix.
pub(super) struct SurfaceOperators {
    /// Per triangle, the cotangents of its three corner angles (all zero for a
    /// degenerate triangle).
    cotangents: Vec<[f64; 3]>,
    /// Mixed Voronoi area of each vertex (m^2); the areas of all vertices sum
    /// to the surface area of the (non-degenerate) triangles.
    pub(super) areas: Vec<f64>,
    /// Sum of the interior angles of the triangles around each vertex (rad).
    pub(super) angle_sums: Vec<f64>,
    /// Whether the vertex lies on the boundary of an open mesh.
    pub(super) on_boundary: Vec<bool>,
}

impl SurfaceOperators {
    /// Build the operators for `triangles` indexing into `positions`.
    pub(super) fn new(positions: &[Vec3], triangles: &[[usize; 3]]) -> Self {
        let vertex_count = positions.len();
        let mut cotangents = Vec::with_capacity(triangles.len());
        let mut areas = vec![0.0; vertex_count];
        let mut angle_sums = vec![0.0; vertex_count];
        for tri in triangles {
            let corners = [positions[tri[0]], positions[tri[1]], positions[tri[2]]];
            match TriangleMeasure::new(corners) {
                Some(measure) => {
                    for (k, &vertex) in tri.iter().enumerate() {
                        areas[vertex] += measure.corner_areas[k];
                        angle_sums[vertex] += measure.angles[k];
                    }
                    cotangents.push(measure.cotangents);
                }
                None => cotangents.push([0.0; 3]),
            }
        }
        Self {
            cotangents,
            areas,
            angle_sums,
            on_boundary: boundary_vertices(vertex_count, triangles),
        }
    }

    /// Cotangent stiffness product `(S f)_i = 1/2 sum_j (cot a_ij + cot b_ij) (f_j - f_i)`.
    ///
    /// `S` is symmetric positive semi-definite and `S x` is the gradient of the
    /// surface area with respect to the vertex positions.
    pub(super) fn stiffness_product(&self, triangles: &[[usize; 3]], values: &[Vec3]) -> Vec<Vec3> {
        let mut product = vec![[0.0; 3]; values.len()];
        for (tri, cot) in triangles.iter().zip(&self.cotangents) {
            for (k, &cotangent) in cot.iter().enumerate() {
                // The edge opposite corner k joins corners k+1 and k+2.
                let i = tri[(k + 1) % 3];
                let j = tri[(k + 2) % 3];
                let weight = 0.5 * cotangent;
                for axis in 0..3 {
                    let flux = weight * (values[j][axis] - values[i][axis]);
                    product[i][axis] += flux;
                    product[j][axis] -= flux;
                }
            }
        }
        product
    }

    /// Laplace-Beltrami operator `Delta f = M^-1 S f` with `M = diag(A_i)`.
    ///
    /// Vertices without any non-degenerate triangle (zero area) get zero.
    pub(super) fn laplace_beltrami(&self, triangles: &[[usize; 3]], values: &[Vec3]) -> Vec<Vec3> {
        let mut laplacian = self.stiffness_product(triangles, values);
        for (entry, &area) in laplacian.iter_mut().zip(&self.areas) {
            *entry = if area > 0.0 {
                scale3(*entry, 1.0 / area)
            } else {
                [0.0; 3]
            };
        }
        laplacian
    }
}

/// Area-weighted unit vertex normals (zero for a vertex whose incident face
/// normals cancel or that has no incident triangle).
pub(super) fn vertex_normals(positions: &[Vec3], triangles: &[[usize; 3]]) -> Vec<Vec3> {
    let mut sums = vec![[0.0; 3]; positions.len()];
    let mut weights = vec![0.0; positions.len()];
    for tri in triangles {
        let face = triangle_normal(positions[tri[0]], positions[tri[1]], positions[tri[2]]);
        let weight = norm3(face);
        for &vertex in tri {
            for axis in 0..3 {
                sums[vertex][axis] += face[axis];
            }
            weights[vertex] += weight;
        }
    }
    sums.iter()
        .zip(&weights)
        .map(|(sum, &weight)| {
            let length = norm3(*sum);
            if weight > 0.0 && length > 1e-12 * weight {
                scale3(*sum, 1.0 / length)
            } else {
                [0.0; 3]
            }
        })
        .collect()
}

/// Gradient of a triangle's area with respect to its three corners.
///
/// With `n = (p_b - p_a) x (p_c - p_a)` the corner gradients are
/// `dA/dp_a = n x (p_c - p_b) / (2 |n|)` and the cyclic permutations. Returns
/// `None` for a degenerate triangle, where the area is not differentiable.
pub(super) fn triangle_area_gradients(pa: Vec3, pb: Vec3, pc: Vec3) -> Option<[Vec3; 3]> {
    let normal = cross3(sub3(pb, pa), sub3(pc, pa));
    let twice_area = norm3(normal);
    let (ab, bc, ca) = (sub3(pb, pa), sub3(pc, pb), sub3(pa, pc));
    let longest_sq = dot3(ab, ab).max(dot3(bc, bc)).max(dot3(ca, ca));
    let non_degenerate = twice_area > DEGENERATE_SINE * longest_sq;
    if !non_degenerate {
        return None;
    }
    let unit = scale3(normal, 1.0 / twice_area);
    Some([
        scale3(cross3(unit, bc), 0.5),
        scale3(cross3(unit, ca), 0.5),
        scale3(cross3(unit, ab), 0.5),
    ])
}

/// Differential-geometry quantities at one mesh vertex.
#[derive(Debug, Clone, Copy)]
pub(super) struct VertexCurvature {
    /// Area-weighted unit normal.
    pub(super) normal: Vec3,
    /// Mixed Voronoi area (m^2).
    pub(super) area: f64,
    /// Mean curvature `H = (c1 + c2) / 2` (1/m); positive where the surface
    /// curves away from its normal, so a sphere with outward normals has
    /// `H = +1/R`.
    pub(super) mean: f64,
    /// Gaussian curvature `K = c1 c2` (1/m^2) from the angle deficit; zero on
    /// boundary vertices.
    pub(super) gaussian: f64,
}

/// Sorted, de-duplicated one-ring neighbours of every vertex.
fn vertex_neighbours(vertex_count: usize, triangles: &[[usize; 3]]) -> Vec<Vec<usize>> {
    let mut neighbours = vec![Vec::new(); vertex_count];
    for tri in triangles {
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            neighbours[a].push(b);
            neighbours[b].push(a);
        }
    }
    for list in &mut neighbours {
        list.sort_unstable();
        list.dedup();
    }
    neighbours
}

/// Compute normal, area, mean and Gaussian curvature at every vertex.
///
/// * `H_i = -1/2 (Delta x)_i . n_i` — the normal component of the
///   Laplace-Beltrami vector of the coordinates (`Delta x = -2 H n`). The
///   tangential part of `Delta x` only reflects mesh irregularity and, on open
///   meshes, the boundary conormal, so it is deliberately not counted as
///   curvature (a flat patch has `H = 0` everywhere, boundary included).
///   A boundary vertex has only a one-sided stencil: its tangential part is
///   of order `1 / h`, so the small tilt of its vertex normal would pollute
///   `H` at order one. Boundary vertices therefore take the mean of the `H`
///   of their interior neighbours (their own value only if they have none).
/// * `K_i = (2 pi - sum of corner angles) / A_i` on interior vertices. On a
///   boundary vertex the angle deficit `pi - sum` measures the geodesic
///   turning of the free edge, not Gaussian curvature of the surface, so `K`
///   is reported as zero there (the integrated `K` of an open piecewise-flat
///   surface lives on its interior vertices).
pub(super) fn vertex_curvatures(
    positions: &[Vec3],
    triangles: &[[usize; 3]],
) -> Vec<VertexCurvature> {
    let operators = SurfaceOperators::new(positions, triangles);
    let normals = vertex_normals(positions, triangles);
    let laplacian = operators.laplace_beltrami(triangles, positions);
    let mut mean: Vec<f64> = laplacian
        .iter()
        .zip(&normals)
        .map(|(l, n)| -0.5 * dot3(*l, *n))
        .collect();
    if operators.on_boundary.iter().any(|&flag| flag) {
        let neighbours = vertex_neighbours(positions.len(), triangles);
        for i in 0..positions.len() {
            if !operators.on_boundary[i] {
                continue;
            }
            let (sum, count) = neighbours[i]
                .iter()
                .filter(|&&j| !operators.on_boundary[j])
                .fold((0.0, 0_usize), |(sum, count), &j| {
                    (sum + mean[j], count + 1)
                });
            if count > 0 {
                mean[i] = sum / count as f64;
            }
        }
    }
    (0..positions.len())
        .map(|i| {
            let area = operators.areas[i];
            let interior = area > 0.0 && !operators.on_boundary[i];
            VertexCurvature {
                normal: normals[i],
                area,
                mean: mean[i],
                gaussian: if interior {
                    (2.0 * PI - operators.angle_sums[i]) / area
                } else {
                    0.0
                },
            }
        })
        .collect()
}

/// Faces of the regular icosahedron built from [`icosahedron_vertices`],
/// wound counter-clockwise seen from outside.
const ICOSAHEDRON_FACES: [[usize; 3]; 20] = [
    [0, 11, 5],
    [0, 5, 1],
    [0, 1, 7],
    [0, 7, 10],
    [0, 10, 11],
    [1, 5, 9],
    [5, 11, 4],
    [11, 10, 2],
    [10, 7, 6],
    [7, 1, 8],
    [3, 9, 4],
    [3, 4, 2],
    [3, 2, 6],
    [3, 6, 8],
    [3, 8, 9],
    [4, 9, 5],
    [2, 4, 11],
    [6, 2, 10],
    [8, 6, 7],
    [9, 8, 1],
];

/// The 12 vertices of an icosahedron (unnormalised, circumradius `sqrt(1 + phi^2)`).
fn icosahedron_vertices() -> [Vec3; 12] {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    [
        [-1.0, phi, 0.0],
        [1.0, phi, 0.0],
        [-1.0, -phi, 0.0],
        [1.0, -phi, 0.0],
        [0.0, -1.0, phi],
        [0.0, 1.0, phi],
        [0.0, -1.0, -phi],
        [0.0, 1.0, -phi],
        [phi, 0.0, -1.0],
        [phi, 0.0, 1.0],
        [-phi, 0.0, -1.0],
        [-phi, 0.0, 1.0],
    ]
}

/// Radially project `point` onto the sphere of the given radius about the origin.
fn project_to_sphere(point: Vec3, radius: f64) -> Vec3 {
    scale3(point, radius / norm3(point))
}

/// Index of the (projected) midpoint of edge `a`-`b`, created on first use.
///
/// `cache` is only ever queried by key, never iterated, so the vertex order
/// depends solely on the (deterministic) triangle order.
fn edge_midpoint(
    vertices: &mut Vec<Vec3>,
    cache: &mut HashMap<(usize, usize), usize>,
    a: usize,
    b: usize,
    radius: f64,
) -> usize {
    let key = (a.min(b), a.max(b));
    if let Some(&index) = cache.get(&key) {
        return index;
    }
    let (pa, pb) = (vertices[a], vertices[b]);
    let middle = [
        0.5 * (pa[0] + pb[0]),
        0.5 * (pa[1] + pb[1]),
        0.5 * (pa[2] + pb[2]),
    ];
    let index = vertices.len();
    vertices.push(project_to_sphere(middle, radius));
    cache.insert(key, index);
    index
}

/// Closed, outward-oriented triangulation of a sphere.
///
/// Starts from the icosahedron and `subdivisions` times splits every triangle
/// into four through its edge midpoints, projecting the new vertices onto the
/// sphere of the given `radius`. The result has `10 * 4^subdivisions + 2`
/// vertices and `20 * 4^subdivisions` triangles; the construction is fully
/// deterministic.
pub(super) fn icosphere_mesh(radius: f64, subdivisions: usize) -> (Vec<Vec3>, Vec<[usize; 3]>) {
    let mut vertices: Vec<Vec3> = icosahedron_vertices()
        .iter()
        .map(|&p| project_to_sphere(p, radius))
        .collect();
    let mut triangles: Vec<[usize; 3]> = ICOSAHEDRON_FACES.to_vec();
    for _ in 0..subdivisions {
        let mut midpoints: HashMap<(usize, usize), usize> =
            HashMap::with_capacity(3 * triangles.len() / 2);
        let mut refined = Vec::with_capacity(4 * triangles.len());
        for tri in &triangles {
            let ab = edge_midpoint(&mut vertices, &mut midpoints, tri[0], tri[1], radius);
            let bc = edge_midpoint(&mut vertices, &mut midpoints, tri[1], tri[2], radius);
            let ca = edge_midpoint(&mut vertices, &mut midpoints, tri[2], tri[0], radius);
            refined.push([tri[0], ab, ca]);
            refined.push([tri[1], bc, ab]);
            refined.push([tri[2], ca, bc]);
            refined.push([ab, bc, ca]);
        }
        triangles = refined;
    }
    // Enforce outward orientation: the face normal must point away from the centre.
    for tri in &mut triangles {
        let (p0, p1, p2) = (vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]);
        let centroid = [
            (p0[0] + p1[0] + p2[0]) / 3.0,
            (p0[1] + p1[1] + p2[1]) / 3.0,
            (p0[2] + p1[2] + p2[2]) / 3.0,
        ];
        if dot3(triangle_normal(p0, p1, p2), centroid) < 0.0 {
            tri.swap(1, 2);
        }
    }
    (vertices, triangles)
}
