// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Voxelization of closed triangle meshes into a signed distance grid.
//!
//! The distance to the mesh is the exact point-to-triangle distance.  The
//! *sign* (negative inside, positive outside) cannot be taken from the plane of
//! the nearest triangle, because near an edge or a vertex the nearest
//! triangle's plane says nothing reliable about which side of the *solid* the
//! point is on.  It is derived instead from angle-weighted pseudo-normals
//! (Bærentzen and Aanæs, *Signed distance computation using the angle weighted
//! pseudonormal*, IEEE TVCG 11(3), 2005): the closest feature of the mesh
//! (face interior, edge or vertex) determines which normal to test against,
//!
//! * face: the face normal,
//! * edge: the normalised sum of the normals of the faces sharing the edge,
//! * vertex: the sum of the normals of the faces around the vertex, each
//!   weighted by the face's interior angle at the vertex,
//!
//! and the sign is that of `dot(p - closest, pseudo_normal)`.  For a closed,
//! consistently oriented (counter-clockwise seen from outside) manifold mesh
//! this is exact.
//!
//! Exact distances are only evaluated in a narrow band around every triangle
//! (its bounding box grown by two cell sizes); the rest of the grid is filled
//! by [`FastSweeping`] from the band and takes its sign from the adjacent band
//! voxels.

use std::collections::{HashMap, VecDeque};

use super::{FastSweeping, SdfVoxelGrid, VoxelCoord, add3, cross3, dist_sq3, dot3, scale3, sub3};
use crate::{Error, Result};

/// Width of the exact-distance band around every triangle, in cell sizes.
const BAND_CELLS: f64 = 2.0;

/// A triangle with a smaller area than this fraction of its squared longest
/// edge is treated as degenerate: it has no defined normal.
const DEGENERATE_AREA_RATIO: f64 = 1e-12;

/// The feature of a triangle that is closest to a query point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriangleFeature {
    /// The interior of the triangle.
    Face,
    /// The interior of edge `k` (`0` = `a→b`, `1` = `b→c`, `2` = `c→a`).
    Edge(usize),
    /// The vertex `k` (`0` = `a`, `1` = `b`, `2` = `c`).
    Vertex(usize),
}

/// The point of a triangle that is closest to a query point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangleClosestPoint {
    /// The closest point on the triangle.
    pub point: [f64; 3],
    /// Squared distance from the query point to [`Self::point`].
    pub dist_sq: f64,
    /// The face, edge or vertex of the triangle that contains [`Self::point`].
    ///
    /// A closest point that coincides with a corner is always reported as
    /// [`TriangleFeature::Vertex`], never as an edge.
    pub feature: TriangleFeature,
}

/// Voxelizer for triangle meshes.
///
/// Rasterizes a triangle mesh into an `SdfVoxelGrid` using exact
/// point-to-triangle distances and angle-weighted pseudo-normals for the sign.
pub struct MeshVoxelizer;

impl MeshVoxelizer {
    /// Compute the squared distance from point `p` to triangle `(a, b, c)`.
    pub fn point_triangle_dist_sq(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
        Self::closest_point_on_triangle(p, a, b, c).dist_sq
    }

    /// Closest point of triangle `(a, b, c)` to `p`, together with the
    /// triangle feature (face, edge or vertex) it lies on.
    ///
    /// Degenerate (zero-area) triangles are handled by falling back to the
    /// nearest corner.
    pub fn closest_point_on_triangle(
        p: [f64; 3],
        a: [f64; 3],
        b: [f64; 3],
        c: [f64; 3],
    ) -> TriangleClosestPoint {
        let ab = sub3(b, a);
        let ac = sub3(c, a);
        let ap = sub3(p, a);
        let d1 = dot3(ab, ap);
        let d2 = dot3(ac, ap);
        if d1 <= 0.0 && d2 <= 0.0 {
            return closest_at_vertex(p, a, 0);
        }

        let bp = sub3(p, b);
        let d3 = dot3(ab, bp);
        let d4 = dot3(ac, bp);
        if d3 >= 0.0 && d4 <= d3 {
            return closest_at_vertex(p, b, 1);
        }

        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            // Edge a-b.
            let t = edge_parameter(d1, d1 - d3);
            return closest_on_edge(p, a, ab, t, [0, 1], 0);
        }

        let cp = sub3(p, c);
        let d5 = dot3(ab, cp);
        let d6 = dot3(ac, cp);
        if d6 >= 0.0 && d5 <= d6 {
            return closest_at_vertex(p, c, 2);
        }

        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            // Edge a-c, which is edge 2 (`c→a`) of the triangle.
            let t = edge_parameter(d2, d2 - d6);
            return closest_on_edge(p, a, ac, t, [0, 2], 2);
        }

        let va = d3 * d6 - d5 * d4;
        let dp = d4 - d3;
        let ep = d5 - d6;
        if va <= 0.0 && dp >= 0.0 && ep >= 0.0 {
            // Edge b-c.
            let t = edge_parameter(dp, dp + ep);
            return closest_on_edge(p, b, sub3(c, b), t, [1, 2], 1);
        }

        let total = va + vb + vc;
        if !total.is_finite() || total <= 0.0 {
            // Zero-area triangle: no interior, use the nearest corner.
            let candidates = [(a, 0), (b, 1), (c, 2)];
            let mut best = closest_at_vertex(p, a, 0);
            for &(corner, k) in &candidates[1..] {
                let cand = closest_at_vertex(p, corner, k);
                if cand.dist_sq < best.dist_sq {
                    best = cand;
                }
            }
            return best;
        }
        let denom = 1.0 / total;
        let v = vb * denom;
        let w = vc * denom;
        let point = add3(a, add3(scale3(ab, v), scale3(ac, w)));
        TriangleClosestPoint {
            point,
            dist_sq: dist_sq3(p, point),
            feature: TriangleFeature::Face,
        }
    }

    /// Return which side of the *plane* of triangle `(a, b, c)` the point `p`
    /// is on: `+1.0` on the positive-normal side (`(b - a) × (c - a)`), `-1.0`
    /// otherwise.
    ///
    /// This is a plane test only.  It is **not** an inside/outside test for a
    /// mesh: close to an edge or a vertex of the mesh the plane of one adjacent
    /// triangle can put an outside point on its negative side.  Use
    /// [`MeshVoxelizer::voxelize`], which classifies with pseudo-normals.
    pub fn sign_winding(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
        let n = cross3(sub3(b, a), sub3(c, a));
        let d = dot3(n, sub3(p, a));
        if d >= 0.0 { 1.0 } else { -1.0 }
    }

    /// Voxelize a closed triangle mesh and write signed distances into `grid`.
    ///
    /// `vertices` is a flat list of `[f64; 3]` positions; `triangles` is a list
    /// of index triples `(i0, i1, i2)` into `vertices`, wound
    /// counter-clockwise when seen from outside.  Every voxel of `grid` is
    /// overwritten with the signed distance from its centre to the mesh
    /// (negative inside).
    ///
    /// The distance is exact within two cell sizes of the surface; farther away
    /// it is extended with the first-order fast sweeping method, and the sign
    /// is carried over from the adjacent voxels (the surface never separates
    /// two such voxels).  Vertices at exactly the same position are treated as
    /// one when the edge and vertex pseudo-normals are accumulated, so meshes
    /// stored as triangle soup are handled.  Zero-area triangles carry no
    /// normal and are ignored.  If the mesh has no triangle (or only
    /// degenerate ones) every voxel is set to `f64::INFINITY`.
    ///
    /// # Errors
    ///
    /// * [`Error::IndexOutOfRange`] if a triangle references a vertex that does
    ///   not exist,
    /// * [`Error::NonFinite`] if a referenced vertex has a NaN or infinite
    ///   coordinate,
    /// * [`Error::DegenerateShape`] if the grid's cell size is not a positive
    ///   finite number,
    /// * [`Error::General`] if `grid.data` does not hold `nx * ny * nz` values.
    ///
    /// On error the grid is left untouched.
    pub fn voxelize(
        grid: &mut SdfVoxelGrid,
        vertices: &[[f64; 3]],
        triangles: &[[usize; 3]],
    ) -> Result<()> {
        let cell = grid.cell_size;
        if !cell.is_finite() || cell <= 0.0 {
            return Err(Error::DegenerateShape {
                detail: format!("voxel cell size must be a positive finite number, got {cell}"),
            });
        }
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let n = nx * ny * nz;
        if grid.data.len() != n {
            return Err(Error::General(format!(
                "voxel grid holds {} values but its dimensions {nx} x {ny} x {nz} need {n}",
                grid.data.len()
            )));
        }
        let mesh = PreparedMesh::new(vertices, triangles)?;
        if n == 0 {
            return Ok(());
        }
        if mesh.is_empty() {
            grid.data.fill(f64::INFINITY);
            return Ok(());
        }

        let band = BAND_CELLS * cell;
        let band_sq = band * band;
        let mut best_dist_sq = vec![f64::INFINITY; n];
        let mut best_sign = vec![1_i8; n];

        // Exact distance and sign in a band around every triangle.
        for (face, &[a, b, c]) in mesh.corners.iter().enumerate() {
            let mut ranges = [(0usize, 0usize); 3];
            let mut reachable = true;
            for axis in 0..3 {
                let lo = a[axis].min(b[axis]).min(c[axis]) - band;
                let hi = a[axis].max(b[axis]).max(c[axis]) + band;
                let extent = [nx, ny, nz][axis];
                match voxel_range(lo, hi, grid.origin[axis], cell, extent) {
                    Some(range) => ranges[axis] = range,
                    None => {
                        reachable = false;
                        break;
                    }
                }
            }
            if !reachable {
                continue;
            }
            for iz in ranges[2].0..=ranges[2].1 {
                for iy in ranges[1].0..=ranges[1].1 {
                    for ix in ranges[0].0..=ranges[0].1 {
                        let center = grid.voxel_center(ix, iy, iz);
                        let closest = Self::closest_point_on_triangle(center, a, b, c);
                        if closest.dist_sq > band_sq {
                            // Inside the grown box but not close enough to be exact.
                            continue;
                        }
                        let idx = ix + nx * (iy + ny * iz);
                        if closest.dist_sq < best_dist_sq[idx] {
                            best_dist_sq[idx] = closest.dist_sq;
                            let side = mesh.side(face, closest.feature, center, closest.point);
                            best_sign[idx] = if side < 0.0 { -1 } else { 1 };
                        }
                    }
                }
            }
        }

        // Voxels with a distance of at most `band` are exact: any triangle that
        // is that close to a voxel centre has the centre inside its grown box.
        let mut seeds: Vec<(VoxelCoord, f64)> = Vec::new();
        let mut sign = vec![0_i8; n];
        let mut queue: VecDeque<usize> = VecDeque::new();
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let idx = ix + nx * (iy + ny * iz);
                    if best_dist_sq[idx] <= band_sq {
                        let (Ok(cx), Ok(cy), Ok(cz)) =
                            (i32::try_from(ix), i32::try_from(iy), i32::try_from(iz))
                        else {
                            continue;
                        };
                        seeds.push((VoxelCoord::new(cx, cy, cz), best_dist_sq[idx].sqrt()));
                        sign[idx] = best_sign[idx];
                        queue.push_back(idx);
                    }
                }
            }
        }

        if seeds.is_empty() {
            // The mesh is farther than the band from every voxel centre, so
            // there is no exact data to extend: evaluate the whole grid.
            for iz in 0..nz {
                for iy in 0..ny {
                    for ix in 0..nx {
                        let center = grid.voxel_center(ix, iy, iz);
                        grid.data[ix + nx * (iy + ny * iz)] = mesh.signed_distance(center);
                    }
                }
            }
            return Ok(());
        }

        let unsigned = FastSweeping::compute(grid, &seeds);

        // Carry the sign into the unresolved voxels, breadth first from the band.
        let plane = nx * ny;
        while let Some(idx) = queue.pop_front() {
            let ix = idx % nx;
            let iy = (idx / nx) % ny;
            let iz = idx / plane;
            let mut visit = |neighbour: usize| {
                if sign[neighbour] == 0 {
                    sign[neighbour] = sign[idx];
                    queue.push_back(neighbour);
                }
            };
            if ix > 0 {
                visit(idx - 1);
            }
            if ix + 1 < nx {
                visit(idx + 1);
            }
            if iy > 0 {
                visit(idx - nx);
            }
            if iy + 1 < ny {
                visit(idx + nx);
            }
            if iz > 0 {
                visit(idx - plane);
            }
            if iz + 1 < nz {
                visit(idx + plane);
            }
        }

        for (idx, value) in grid.data.iter_mut().enumerate() {
            let side = if sign[idx] < 0 { -1.0 } else { 1.0 };
            *value = side * unsigned[idx];
        }
        Ok(())
    }
}

/// Closest-point record for a query point whose closest point is the corner
/// `corner` (index `k` of the triangle).
fn closest_at_vertex(p: [f64; 3], corner: [f64; 3], k: usize) -> TriangleClosestPoint {
    TriangleClosestPoint {
        point: corner,
        dist_sq: dist_sq3(p, corner),
        feature: TriangleFeature::Vertex(k),
    }
}

/// Clamped parameter of the closest point along an edge, `numerator / denominator`.
fn edge_parameter(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        (numerator / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Closest-point record for the point `start + t * direction` on an edge whose
/// end points are the triangle corners `ends`; `edge` is the edge's index.
fn closest_on_edge(
    p: [f64; 3],
    start: [f64; 3],
    direction: [f64; 3],
    t: f64,
    ends: [usize; 2],
    edge: usize,
) -> TriangleClosestPoint {
    let point = add3(start, scale3(direction, t));
    let feature = if t <= 0.0 {
        TriangleFeature::Vertex(ends[0])
    } else if t >= 1.0 {
        TriangleFeature::Vertex(ends[1])
    } else {
        TriangleFeature::Edge(edge)
    };
    TriangleClosestPoint {
        point,
        dist_sq: dist_sq3(p, point),
        feature,
    }
}

/// Inclusive range of voxel indices along one axis whose centres lie in
/// `[lo, hi]`, clamped to `0..extent`; `None` if there is none.
fn voxel_range(lo: f64, hi: f64, origin: f64, cell: f64, extent: usize) -> Option<(usize, usize)> {
    if extent == 0 {
        return None;
    }
    let first = ((lo - origin) / cell - 0.5).ceil();
    let last = ((hi - origin) / cell - 0.5).floor();
    let top = (extent - 1) as f64;
    if !first.is_finite() || !last.is_finite() || first > last || last < 0.0 || first > top {
        return None;
    }
    // `as usize` saturates, and both values are within `0..=top` here.
    Some((first.max(0.0) as usize, last.min(top) as usize))
}

/// Unit vector along `v`, or `None` if `v` is (numerically) zero.
fn unit(v: [f64; 3]) -> Option<[f64; 3]> {
    let len = dot3(v, v).sqrt();
    if len > 0.0 && len.is_finite() {
        Some(scale3(v, 1.0 / len))
    } else {
        None
    }
}

/// Interior angle at `apex` between the directions to `p` and `q`.
fn corner_angle(apex: [f64; 3], p: [f64; 3], q: [f64; 3]) -> f64 {
    let u = sub3(p, apex);
    let v = sub3(q, apex);
    dot3(cross3(u, v), cross3(u, v)).sqrt().atan2(dot3(u, v))
}

/// Hashable key of a position; `-0.0` and `0.0` weld together.
fn weld_key(p: [f64; 3]) -> [u64; 3] {
    [
        (p[0] + 0.0).to_bits(),
        (p[1] + 0.0).to_bits(),
        (p[2] + 0.0).to_bits(),
    ]
}

/// A validated mesh with its face, edge and vertex pseudo-normals.
///
/// Only non-degenerate triangles are kept.  The pseudo-normals are stored per
/// face corner, so the sign query in the voxelization loop needs no lookups.
pub(super) struct PreparedMesh {
    /// Corner positions of every kept triangle.
    pub(super) corners: Vec<[[f64; 3]; 3]>,
    /// Unit face normals, `(b - a) × (c - a)` normalised.
    face_normals: Vec<[f64; 3]>,
    /// Edge pseudo-normals; entry `k` belongs to edge `k` of the triangle.
    edge_normals: Vec<[[f64; 3]; 3]>,
    /// Angle-weighted vertex pseudo-normals; entry `k` belongs to corner `k`.
    vertex_normals: Vec<[[f64; 3]; 3]>,
}

impl PreparedMesh {
    /// Validate `triangles` against `vertices` and build the pseudo-normals.
    pub(super) fn new(vertices: &[[f64; 3]], triangles: &[[usize; 3]]) -> Result<Self> {
        let mut ids: HashMap<[u64; 3], usize> = HashMap::new();
        let mut canonical: Vec<Option<usize>> = vec![None; vertices.len()];

        let mut corners: Vec<[[f64; 3]; 3]> = Vec::with_capacity(triangles.len());
        let mut corner_ids: Vec<[usize; 3]> = Vec::with_capacity(triangles.len());
        let mut face_normals: Vec<[f64; 3]> = Vec::with_capacity(triangles.len());

        for tri in triangles {
            let mut pos = [[0.0_f64; 3]; 3];
            let mut cid = [0usize; 3];
            for k in 0..3 {
                let vi = tri[k];
                let Some(&v) = vertices.get(vi) else {
                    return Err(Error::IndexOutOfRange {
                        index: vi,
                        len: vertices.len(),
                    });
                };
                if !v.iter().all(|x| x.is_finite()) {
                    return Err(Error::NonFinite {
                        context: format!("MeshVoxelizer::voxelize vertex {vi}"),
                    });
                }
                pos[k] = v;
                cid[k] = match canonical[vi] {
                    Some(id) => id,
                    None => {
                        let next = ids.len();
                        let id = *ids.entry(weld_key(v)).or_insert(next);
                        canonical[vi] = Some(id);
                        id
                    }
                };
            }
            let [a, b, c] = pos;
            let normal = cross3(sub3(b, a), sub3(c, a));
            let longest_sq = dist_sq3(a, b).max(dist_sq3(b, c)).max(dist_sq3(c, a));
            let area2 = dot3(normal, normal).sqrt();
            let welded = cid[0] == cid[1] || cid[1] == cid[2] || cid[2] == cid[0];
            if welded || area2.is_nan() || area2 <= DEGENERATE_AREA_RATIO * longest_sq {
                continue;
            }
            let Some(unit_normal) = unit(normal) else {
                continue;
            };
            corners.push(pos);
            corner_ids.push(cid);
            face_normals.push(unit_normal);
        }

        let mut vertex_sum: Vec<[f64; 3]> = vec![[0.0; 3]; ids.len()];
        let mut edge_sum: HashMap<(usize, usize), [f64; 3]> = HashMap::new();
        for ((pos, cid), normal) in corners.iter().zip(&corner_ids).zip(&face_normals) {
            for k in 0..3 {
                let next = (k + 1) % 3;
                let prev = (k + 2) % 3;
                let angle = corner_angle(pos[k], pos[next], pos[prev]);
                vertex_sum[cid[k]] = add3(vertex_sum[cid[k]], scale3(*normal, angle));
                let key = (cid[k].min(cid[next]), cid[k].max(cid[next]));
                let entry = edge_sum.entry(key).or_insert([0.0; 3]);
                *entry = add3(*entry, *normal);
            }
        }

        let mut edge_normals = Vec::with_capacity(corners.len());
        let mut vertex_normals = Vec::with_capacity(corners.len());
        for (cid, face_normal) in corner_ids.iter().zip(&face_normals) {
            let mut edges = [*face_normal; 3];
            let mut verts = [*face_normal; 3];
            for k in 0..3 {
                let next = (k + 1) % 3;
                let key = (cid[k].min(cid[next]), cid[k].max(cid[next]));
                if let Some(n) = edge_sum.get(&key).copied().and_then(unit) {
                    edges[k] = n;
                }
                if let Some(n) = unit(vertex_sum[cid[k]]) {
                    verts[k] = n;
                }
            }
            edge_normals.push(edges);
            vertex_normals.push(verts);
        }

        Ok(Self {
            corners,
            face_normals,
            edge_normals,
            vertex_normals,
        })
    }

    /// `true` if no triangle was kept.
    pub(super) fn is_empty(&self) -> bool {
        self.corners.is_empty()
    }

    /// The pseudo-normal of `feature` of triangle `face`.
    pub(super) fn pseudo_normal(&self, face: usize, feature: TriangleFeature) -> [f64; 3] {
        match feature {
            TriangleFeature::Face => self.face_normals[face],
            TriangleFeature::Edge(k) => self.edge_normals[face][k % 3],
            TriangleFeature::Vertex(k) => self.vertex_normals[face][k % 3],
        }
    }

    /// `+1.0` if `p` is outside the solid, `-1.0` if inside, given the closest
    /// point `closest` of triangle `face` and the feature it lies on.
    pub(super) fn side(
        &self,
        face: usize,
        feature: TriangleFeature,
        p: [f64; 3],
        closest: [f64; 3],
    ) -> f64 {
        let n = self.pseudo_normal(face, feature);
        if dot3(sub3(p, closest), n) >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    /// Exact signed distance from `p` to the whole mesh (`O(triangles)`).
    pub(super) fn signed_distance(&self, p: [f64; 3]) -> f64 {
        let mut best = f64::INFINITY;
        let mut sign = 1.0;
        for (face, &[a, b, c]) in self.corners.iter().enumerate() {
            let closest = MeshVoxelizer::closest_point_on_triangle(p, a, b, c);
            if closest.dist_sq < best {
                best = closest.dist_sq;
                sign = self.side(face, closest.feature, p, closest.point);
            }
        }
        sign * best.sqrt()
    }
}
