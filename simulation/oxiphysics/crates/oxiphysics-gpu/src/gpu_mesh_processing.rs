// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! GPU triangle mesh processing (CPU mock implementation).
//!
//! Provides vertex normal computation, Laplacian smoothing, edge collapse,
//! Loop subdivision, decimation, AABB, surface area, volume, and vertex welding.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

// ── Mesh data structure ──────────────────────────────────────────────────────

/// Triangle mesh stored in GPU-friendly interleaved layout.
///
/// Each vertex holds a position, a normal, and a UV texture coordinate.
/// Triangles are stored as `[i, j, k]` index triples.
#[derive(Debug, Clone)]
pub struct GpuMesh {
    /// Vertex positions `[x, y, z]`.
    pub vertices: Vec<[f32; 3]>,
    /// Per-vertex normals `[nx, ny, nz]`.
    pub normals: Vec<[f32; 3]>,
    /// Triangle index triples.
    pub indices: Vec<[u32; 3]>,
    /// Per-vertex UV texture coordinates `[u, v]`.
    pub tex_coords: Vec<[f32; 2]>,
}

impl GpuMesh {
    /// Create an empty mesh.
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
            tex_coords: Vec::new(),
        }
    }

    /// Add a vertex with position, normal, and UV.
    pub fn add_vertex(&mut self, pos: [f32; 3], normal: [f32; 3], uv: [f32; 2]) {
        self.vertices.push(pos);
        self.normals.push(normal);
        self.tex_coords.push(uv);
    }

    /// Add a triangle by vertex indices.
    pub fn add_triangle(&mut self, i: u32, j: u32, k: u32) {
        self.indices.push([i, j, k]);
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }
}

impl Default for GpuMesh {
    fn default() -> Self {
        Self::new()
    }
}

// ── Math helpers ─────────────────────────────────────────────────────────────

/// Compute the area of triangle `(a, b, c)` using the cross-product formula.
///
/// Returns `0.5 * |AB × AC|`.
pub fn triangle_area(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let cross = cross3f(ab, ac);
    0.5 * length3f(cross)
}

/// Compute the unit normal of triangle `(a, b, c)`.
///
/// Returns the zero vector if the triangle is degenerate.
pub fn triangle_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    normalize3f(cross3f(ab, ac))
}

// ── Normal computation ───────────────────────────────────────────────────────

/// Recompute per-vertex normals using area-weighted face normals.
///
/// Each vertex normal is the normalised sum of the normals of its incident
/// triangles, weighted by the triangle area.
pub fn gpu_compute_normals(mesh: &mut GpuMesh) {
    let nv = mesh.vertex_count();
    let mut acc = vec![[0.0f32; 3]; nv];

    for tri in &mesh.indices {
        let [i, j, k] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let a = mesh.vertices[i];
        let b = mesh.vertices[j];
        let c = mesh.vertices[k];
        let area = triangle_area(a, b, c);
        let n = triangle_normal(a, b, c);
        for idx in [i, j, k] {
            acc[idx][0] += n[0] * area;
            acc[idx][1] += n[1] * area;
            acc[idx][2] += n[2] * area;
        }
    }

    for (v_idx, n) in mesh.normals.iter_mut().enumerate() {
        *n = normalize3f(acc[v_idx]);
    }
}

/// Laplacian normal smoothing: average each vertex normal with its neighbours.
///
/// Runs `n_iter` passes.
pub fn gpu_smooth_normals(mesh: &mut GpuMesh, n_iter: usize) {
    let nv = mesh.vertex_count();
    for _ in 0..n_iter {
        let mut acc = vec![[0.0f32; 3]; nv];
        let mut count = vec![0u32; nv];
        for tri in &mesh.indices {
            for &vi in tri.iter() {
                let vi = vi as usize;
                for &vj in tri.iter() {
                    let vj = vj as usize;
                    acc[vi][0] += mesh.normals[vj][0];
                    acc[vi][1] += mesh.normals[vj][1];
                    acc[vi][2] += mesh.normals[vj][2];
                    count[vi] += 1;
                }
            }
        }
        for i in 0..nv {
            let c = count[i].max(1) as f32;
            mesh.normals[i] = normalize3f([acc[i][0] / c, acc[i][1] / c, acc[i][2] / c]);
        }
    }
}

// ── Edge collapse (simplified QEM) ──────────────────────────────────────────

/// Collapse edges whose midpoint error is below `error_threshold`.
///
/// Returns the number of removed vertices.  Uses a simplified cost metric
/// (edge length squared) as a proxy for the Quadric Error Metric.  Vertices
/// joined by a short edge are merged (union-find with path halving); triangles
/// that become degenerate, and triangles that name a vertex that does not
/// exist, are removed from the index buffer.
pub fn gpu_edge_collapse(mesh: &mut GpuMesh, error_threshold: f32) -> usize {
    let mut removed = 0usize;
    let nv = mesh.vertex_count();
    let mut merge_target: Vec<usize> = (0..nv).collect(); // union-find parents
    let threshold_sq = error_threshold * error_threshold;

    // Build the edge list from the triangles and merge short edges.
    for tri in &mesh.indices {
        let Some(verts) = triangle_vertices(tri, nv) else {
            continue;
        };
        for (i, j) in [
            (verts[0], verts[1]),
            (verts[1], verts[2]),
            (verts[0], verts[2]),
        ] {
            if dist2_3f(mesh.vertices[i], mesh.vertices[j]) < threshold_sq {
                // Merge j into i
                let root_i = find_root(&mut merge_target, i);
                let root_j = find_root(&mut merge_target, j);
                if root_i != root_j {
                    merge_target[root_j] = root_i;
                    removed += 1;
                }
            }
        }
    }

    // Drop triangles that reference missing vertices, then remap the rest.
    mesh.indices
        .retain(|tri| triangle_vertices(tri, nv).is_some());
    for tri in mesh.indices.iter_mut() {
        for idx in tri.iter_mut() {
            *idx = find_root(&mut merge_target, *idx as usize) as u32;
        }
    }
    // Remove degenerate triangles.
    mesh.indices
        .retain(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2]);
    removed
}

/// The vertex indices of `tri` as `usize`, or `None` when one of them is not
/// below the vertex count `vertex_count`.
fn triangle_vertices(tri: &[u32; 3], vertex_count: usize) -> Option<[usize; 3]> {
    let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
    (a < vertex_count && b < vertex_count && c < vertex_count).then_some([a, b, c])
}

/// Union-find root of `i` with path halving: every other node on the way up is
/// re-pointed at its grandparent, which keeps repeated lookups near-constant.
fn find_root(target: &mut [usize], mut i: usize) -> usize {
    while target[i] != i {
        target[i] = target[target[i]];
        i = target[i];
    }
    i
}

// ── Loop subdivision ─────────────────────────────────────────────────────────

/// One undirected edge of the mesh seen by Loop subdivision.
#[derive(Debug, Clone, Copy)]
struct LoopEdge {
    /// Smaller endpoint.
    a: u32,
    /// Larger endpoint.
    b: u32,
    /// Vertices opposite the edge in its first two incident triangles.
    opposite: [u32; 2],
    /// Number of incident triangles.
    faces: u32,
}

impl LoopEdge {
    /// Interior edges have exactly two incident triangles; boundary edges (one)
    /// and non-manifold edges (three or more) are treated as creases.
    fn is_crease(&self) -> bool {
        self.faces != 2
    }
}

/// Loop's vertex weight `β(n) = (1/n) (5/8 − (3/8 + cos(2π/n)/4)²)` for an
/// interior vertex of valence `n` (`3/16` for `n = 3`, `1/16` for `n = 6`).
fn loop_beta(valence: usize) -> f64 {
    let n = valence as f64;
    let c = 0.375 + (2.0 * std::f64::consts::PI / n).cos() * 0.25;
    (0.625 - c * c) / n
}

fn weighted_sum3(terms: &[(f32, [f32; 3])]) -> [f32; 3] {
    let mut sum = [0.0_f32; 3];
    for (weight, v) in terms {
        sum[0] += weight * v[0];
        sum[1] += weight * v[1];
        sum[2] += weight * v[2];
    }
    sum
}

/// Perform one step of Loop subdivision, returning a new mesh.
///
/// Each triangle is split into four sub-triangles (winding is preserved) and
/// the vertices follow Loop's scheme:
///
/// * a new *odd* vertex on an interior edge `(a, b)` with opposite vertices `c`
///   and `d` is placed at `3/8 (a + b) + 1/8 (c + d)`; on a boundary edge (or an
///   edge with other than two incident triangles) it is the midpoint
///   `1/2 (a + b)`;
/// * an original *even* vertex of valence `n` with neighbours `n_i` moves to
///   `(1 − nβ) v + β Σ n_i` with `β = (1/n)(5/8 − (3/8 + cos(2π/n)/4)²)`; a
///   boundary vertex moves to `3/4 v + 1/8 (n_0 + n_1)` where `n_0` and `n_1`
///   are its two neighbours along boundary edges; a vertex with any other
///   number of boundary edges (a corner or a non-manifold vertex) and a vertex
///   that no triangle uses keep their position.
///
/// Texture coordinates of new vertices are the average of the edge's end
/// points; original vertices keep theirs.  Vertex normals are recomputed from
/// the subdivided surface (vertices that no triangle uses keep their normal).
/// Triangles that name a vertex that does not exist or repeat a vertex are
/// dropped.  If the result would need more vertices than `u32` indices can
/// address, the mesh is returned unchanged.
pub fn gpu_loop_subdivision(mesh: &GpuMesh) -> GpuMesh {
    let nv = mesh.vertices.len();
    let triangles: Vec<[u32; 3]> = mesh
        .indices
        .iter()
        .copied()
        .filter(|tri| {
            triangle_vertices(tri, nv).is_some()
                && tri[0] != tri[1]
                && tri[1] != tri[2]
                && tri[0] != tri[2]
        })
        .collect();

    // Edge table in order of first appearance (deterministic), with the
    // edges of every triangle as (a,b), (b,c), (c,a).
    let mut edges: Vec<LoopEdge> = Vec::new();
    let mut edge_index: HashMap<(u32, u32), usize> = HashMap::new();
    let mut triangle_edges: Vec<[usize; 3]> = Vec::with_capacity(triangles.len());
    for tri in &triangles {
        let mut ids = [0_usize; 3];
        let sides = [
            (tri[0], tri[1], tri[2]),
            (tri[1], tri[2], tri[0]),
            (tri[2], tri[0], tri[1]),
        ];
        for (id, (u, v, opposite)) in ids.iter_mut().zip(sides) {
            let key = if u < v { (u, v) } else { (v, u) };
            let slot = *edge_index.entry(key).or_insert_with(|| {
                edges.push(LoopEdge {
                    a: key.0,
                    b: key.1,
                    opposite: [0; 2],
                    faces: 0,
                });
                edges.len() - 1
            });
            let edge = &mut edges[slot];
            if let Some(entry) = edge.opposite.get_mut(edge.faces as usize) {
                *entry = opposite;
            }
            edge.faces += 1;
            *id = slot;
        }
        triangle_edges.push(ids);
    }

    if nv
        .checked_add(edges.len())
        .is_none_or(|total| u32::try_from(total).is_err())
    {
        return mesh.clone();
    }

    // Vertex adjacency (all neighbours, and neighbours across crease edges).
    let mut neighbours: Vec<Vec<u32>> = vec![Vec::new(); nv];
    let mut crease_neighbours: Vec<Vec<u32>> = vec![Vec::new(); nv];
    for edge in &edges {
        neighbours[edge.a as usize].push(edge.b);
        neighbours[edge.b as usize].push(edge.a);
        if edge.is_crease() {
            crease_neighbours[edge.a as usize].push(edge.b);
            crease_neighbours[edge.b as usize].push(edge.a);
        }
    }

    // Even vertices: smoothed originals.
    let mut vertices: Vec<[f32; 3]> = Vec::with_capacity(nv + edges.len());
    for (v, position) in mesh.vertices.iter().enumerate() {
        let ring = &neighbours[v];
        let creases = &crease_neighbours[v];
        let moved = if ring.is_empty() {
            *position
        } else if creases.is_empty() {
            let beta = loop_beta(ring.len()) as f32;
            let mut sum = [0.0_f32; 3];
            for &n in ring {
                let p = mesh.vertices[n as usize];
                sum = [sum[0] + p[0], sum[1] + p[1], sum[2] + p[2]];
            }
            weighted_sum3(&[(1.0 - ring.len() as f32 * beta, *position), (beta, sum)])
        } else if let [n0, n1] = creases[..] {
            weighted_sum3(&[
                (0.75, *position),
                (0.125, mesh.vertices[n0 as usize]),
                (0.125, mesh.vertices[n1 as usize]),
            ])
        } else {
            *position
        };
        vertices.push(moved);
    }

    // Odd vertices: one per edge.
    let uv_of = |v: u32| mesh.tex_coords.get(v as usize).copied().unwrap_or([0.0; 2]);
    let mut tex_coords: Vec<[f32; 2]> = (0..nv as u32).map(uv_of).collect();
    for edge in &edges {
        let pa = mesh.vertices[edge.a as usize];
        let pb = mesh.vertices[edge.b as usize];
        let position = if edge.is_crease() {
            weighted_sum3(&[(0.5, pa), (0.5, pb)])
        } else {
            weighted_sum3(&[
                (0.375, pa),
                (0.375, pb),
                (0.125, mesh.vertices[edge.opposite[0] as usize]),
                (0.125, mesh.vertices[edge.opposite[1] as usize]),
            ])
        };
        vertices.push(position);
        let (ua, ub) = (uv_of(edge.a), uv_of(edge.b));
        tex_coords.push([(ua[0] + ub[0]) * 0.5, (ua[1] + ub[1]) * 0.5]);
    }

    // Four sub-triangles per triangle: three corners and the middle one.
    let mut indices: Vec<[u32; 3]> = Vec::with_capacity(triangles.len() * 4);
    for (tri, ids) in triangles.iter().zip(triangle_edges.iter()) {
        let [a, b, c] = *tri;
        let [ab, bc, ca] = ids.map(|slot| (nv + slot) as u32);
        indices.push([a, ab, ca]);
        indices.push([b, bc, ab]);
        indices.push([c, ca, bc]);
        indices.push([ab, bc, ca]);
    }

    let mut result = GpuMesh {
        normals: vec![[0.0; 3]; vertices.len()],
        vertices,
        indices,
        tex_coords,
    };
    gpu_compute_normals(&mut result);
    // Vertices no triangle touches keep the normal they came with.
    for (v, ring) in neighbours.iter().enumerate() {
        if ring.is_empty()
            && let Some(&normal) = mesh.normals.get(v)
        {
            result.normals[v] = normal;
        }
    }
    result
}

// ── Decimation ───────────────────────────────────────────────────────────────

/// An edge collapse may not leave a triangle whose normal turns by more than
/// `acos(MIN_COLLAPSE_NORMAL_COSINE)` (about 78°) relative to before; a smaller
/// cosine means the triangle would fold over or become a sliver.
const MIN_COLLAPSE_NORMAL_COSINE: f32 = 0.2;

/// A queued edge collapse: `remove` is merged into `keep`.
///
/// Ordered by cost, then by the endpoints and versions, so ties are resolved
/// deterministically.  Entries are never updated in place: when a vertex moves
/// or its neighbourhood changes its `version` is bumped and entries that carry
/// the old version are discarded when popped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CollapseCandidate {
    /// IEEE-754 bits of the squared edge length: for finite non-negative floats
    /// the integer order equals the numeric order.
    cost: u32,
    keep: u32,
    remove: u32,
    keep_version: u32,
    remove_version: u32,
}

/// Where the merged vertex goes, and what it carries.
#[derive(Debug, Clone, Copy)]
struct CollapsePlan {
    position: [f32; 3],
    uv: [f32; 2],
}

/// Whether a triangle whose corners move from `old` to `new` during an edge
/// collapse stays a healthy triangle: its normal must not turn by more than
/// `acos(`[`MIN_COLLAPSE_NORMAL_COSINE`]`)` and its area must not vanish.  A
/// triangle that already had zero area carries nothing to preserve and always
/// passes.
fn triangle_survives_collapse(old: [[f32; 3]; 3], new: [[f32; 3]; 3]) -> bool {
    let old_normal = cross3f(sub3f(old[1], old[0]), sub3f(old[2], old[0]));
    let old_len = length3f(old_normal);
    if old_len == 0.0 {
        return true;
    }
    let new_normal = cross3f(sub3f(new[1], new[0]), sub3f(new[2], new[0]));
    let new_len = length3f(new_normal);
    if new_len.is_nan() || new_len <= 0.0 {
        return false;
    }
    dot3f(old_normal, new_normal) > MIN_COLLAPSE_NORMAL_COSINE * old_len * new_len
}

/// Working state of [`gpu_mesh_decimate`]: positions, triangles and the
/// vertex → triangle adjacency, updated incrementally as edges collapse.
struct Decimator {
    positions: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    triangles: Vec<[u32; 3]>,
    triangle_alive: Vec<bool>,
    /// Live triangles incident to each vertex (dead ones are removed eagerly).
    vertex_triangles: Vec<Vec<u32>>,
    vertex_alive: Vec<bool>,
    version: Vec<u32>,
    alive_triangles: usize,
}

impl Decimator {
    /// Build the working state; triangles that name a missing vertex or repeat
    /// a vertex are not carried over.
    fn new(mesh: &GpuMesh) -> Self {
        let nv = mesh.vertices.len();
        let triangles: Vec<[u32; 3]> = mesh
            .indices
            .iter()
            .copied()
            .filter(|tri| {
                triangle_vertices(tri, nv).is_some()
                    && tri[0] != tri[1]
                    && tri[1] != tri[2]
                    && tri[0] != tri[2]
            })
            .collect();
        let mut vertex_triangles: Vec<Vec<u32>> = vec![Vec::new(); nv];
        for (t, tri) in triangles.iter().enumerate() {
            for &v in tri {
                vertex_triangles[v as usize].push(t as u32);
            }
        }
        Self {
            positions: mesh.vertices.clone(),
            uvs: (0..nv)
                .map(|v| mesh.tex_coords.get(v).copied().unwrap_or([0.0; 2]))
                .collect(),
            alive_triangles: triangles.len(),
            triangle_alive: vec![true; triangles.len()],
            triangles,
            vertex_triangles,
            vertex_alive: vec![true; nv],
            version: vec![0; nv],
        }
    }

    /// Distinct vertices that share a live triangle with `v`, ascending.
    fn neighbours(&self, v: u32) -> Vec<u32> {
        let mut out: Vec<u32> = self.vertex_triangles[v as usize]
            .iter()
            .flat_map(|&t| self.triangles[t as usize])
            .filter(|&other| other != v)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// True when some edge at `v` has other than two incident triangles.
    fn is_boundary(&self, v: u32) -> bool {
        let mut others: Vec<u32> = self.vertex_triangles[v as usize]
            .iter()
            .flat_map(|&t| self.triangles[t as usize])
            .filter(|&other| other != v)
            .collect();
        others.sort_unstable();
        // Each neighbour appears once per incident triangle on the edge to it.
        others.chunk_by(|a, b| a == b).any(|run| run.len() != 2)
    }

    fn squared_length(&self, a: u32, b: u32) -> f32 {
        dist2_3f(self.positions[a as usize], self.positions[b as usize])
    }

    fn candidate(&self, a: u32, b: u32) -> Option<CollapseCandidate> {
        let (keep, remove) = if a < b { (a, b) } else { (b, a) };
        let length_sq = self.squared_length(keep, remove);
        length_sq.is_finite().then(|| CollapseCandidate {
            cost: length_sq.to_bits(),
            keep,
            remove,
            keep_version: self.version[keep as usize],
            remove_version: self.version[remove as usize],
        })
    }

    /// Queue every distinct edge of the live triangles.
    fn initial_candidates(&self) -> Vec<Reverse<CollapseCandidate>> {
        let mut edges: Vec<(u32, u32)> = Vec::with_capacity(self.triangles.len() * 3);
        for tri in &self.triangles {
            for (u, v) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                edges.push(if u < v { (u, v) } else { (v, u) });
            }
        }
        edges.sort_unstable();
        edges.dedup();
        edges
            .into_iter()
            .filter_map(|(a, b)| self.candidate(a, b).map(Reverse))
            .collect()
    }

    /// A popped entry is current when both endpoints are alive and neither has
    /// changed since the entry was queued.
    fn is_current(&self, c: &CollapseCandidate) -> bool {
        let (k, r) = (c.keep as usize, c.remove as usize);
        self.vertex_alive[k]
            && self.vertex_alive[r]
            && self.version[k] == c.keep_version
            && self.version[r] == c.remove_version
    }

    /// Check whether collapsing `remove` into `keep` keeps the surface valid
    /// and, if so, where the merged vertex goes.
    ///
    /// * The edge must be a manifold edge (one or two triangles).
    /// * Link condition: the vertices adjacent to both endpoints must be
    ///   exactly the vertices opposite the edge — otherwise the collapse would
    ///   pinch the surface or create duplicate triangles.
    /// * Two boundary vertices may only merge along a boundary edge.
    /// * A closed tetrahedron is never collapsed (it would flatten).
    /// * No triangle around the edge may flip or degenerate: its normal must
    ///   stay within [`MIN_COLLAPSE_NORMAL_COSINE`] of the old one.
    ///
    /// The merged vertex sits at the midpoint of the edge, or at the boundary
    /// vertex when exactly one endpoint is on the boundary, so the boundary
    /// curve is not pulled inwards.
    fn plan_collapse(&self, keep: u32, remove: u32) -> Option<CollapsePlan> {
        let shared: Vec<u32> = self.vertex_triangles[remove as usize]
            .iter()
            .copied()
            .filter(|&t| self.triangles[t as usize].contains(&keep))
            .collect();
        if shared.is_empty() || shared.len() > 2 {
            return None;
        }
        let mut opposite: Vec<u32> = shared
            .iter()
            .flat_map(|&t| self.triangles[t as usize])
            .filter(|&v| v != keep && v != remove)
            .collect();
        opposite.sort_unstable();

        let around_keep: Vec<u32> = self
            .neighbours(keep)
            .into_iter()
            .filter(|&v| v != remove)
            .collect();
        let around_remove: Vec<u32> = self
            .neighbours(remove)
            .into_iter()
            .filter(|&v| v != keep)
            .collect();
        let common: Vec<u32> = around_keep
            .iter()
            .copied()
            .filter(|v| around_remove.binary_search(v).is_ok())
            .collect();
        if common != opposite {
            return None;
        }

        let keep_boundary = self.is_boundary(keep);
        let remove_boundary = self.is_boundary(remove);
        if keep_boundary && remove_boundary && shared.len() == 2 {
            return None;
        }
        if shared.len() == 2 && !keep_boundary && !remove_boundary {
            let mut union = around_keep.clone();
            union.extend_from_slice(&around_remove);
            union.sort_unstable();
            union.dedup();
            if union.len() <= 2 {
                return None;
            }
        }

        let plan = {
            let (ka, ra) = (keep as usize, remove as usize);
            let (ku, ru) = (self.uvs[ka], self.uvs[ra]);
            match (keep_boundary, remove_boundary) {
                (true, false) => CollapsePlan {
                    position: self.positions[ka],
                    uv: ku,
                },
                (false, true) => CollapsePlan {
                    position: self.positions[ra],
                    uv: ru,
                },
                _ => CollapsePlan {
                    position: midpoint3f(self.positions[ka], self.positions[ra]),
                    uv: [(ku[0] + ru[0]) * 0.5, (ku[1] + ru[1]) * 0.5],
                },
            }
        };

        // Every triangle that survives and touches either endpoint moves its
        // corner to the merged position: it must not flip or collapse.
        for &t in self.vertex_triangles[keep as usize]
            .iter()
            .chain(self.vertex_triangles[remove as usize].iter())
        {
            if shared.contains(&t) {
                continue;
            }
            let tri = self.triangles[t as usize];
            let old = tri.map(|v| self.positions[v as usize]);
            let new = tri.map(|v| {
                if v == keep || v == remove {
                    plan.position
                } else {
                    self.positions[v as usize]
                }
            });
            if !triangle_survives_collapse(old, new) {
                return None;
            }
        }
        Some(plan)
    }

    /// Merge `remove` into `keep`: kill the triangles that contain both, re-point
    /// the others at `keep`, and move `keep` to the planned position.
    fn collapse(&mut self, keep: u32, remove: u32, plan: &CollapsePlan) {
        let incident = std::mem::take(&mut self.vertex_triangles[remove as usize]);
        for t in incident {
            let tri = self.triangles[t as usize];
            if tri.contains(&keep) {
                self.triangle_alive[t as usize] = false;
                self.alive_triangles -= 1;
                for v in tri {
                    if v != remove {
                        self.vertex_triangles[v as usize].retain(|&q| q != t);
                    }
                }
            } else {
                if let Some(slot) = self.triangles[t as usize]
                    .iter_mut()
                    .find(|v| **v == remove)
                {
                    *slot = keep;
                }
                self.vertex_triangles[keep as usize].push(t);
            }
        }
        self.vertex_alive[remove as usize] = false;
        self.positions[keep as usize] = plan.position;
        self.uvs[keep as usize] = plan.uv;
    }

    /// After a collapse every edge around the merged vertex has changed
    /// (length, or the topology that decides whether it may collapse): bump the
    /// versions of the merged vertex and of its neighbours — which discards all
    /// stale entries that touch them — and queue their edges afresh.
    fn requeue_around(&mut self, merged: u32, heap: &mut BinaryHeap<Reverse<CollapseCandidate>>) {
        let mut ring = self.neighbours(merged);
        ring.push(merged);
        for &v in &ring {
            let version = &mut self.version[v as usize];
            *version = version.wrapping_add(1);
        }
        let mut edges: Vec<(u32, u32)> = Vec::new();
        for &v in &ring {
            for n in self.neighbours(v) {
                edges.push(if v < n { (v, n) } else { (n, v) });
            }
        }
        edges.sort_unstable();
        edges.dedup();
        for (a, b) in edges {
            if let Some(candidate) = self.candidate(a, b) {
                heap.push(Reverse(candidate));
            }
        }
    }

    /// Compact to the vertices that a live triangle uses (original order) and
    /// recompute the vertex normals.
    fn into_mesh(self) -> GpuMesh {
        let mut used = vec![false; self.positions.len()];
        for (tri, &alive) in self.triangles.iter().zip(self.triangle_alive.iter()) {
            if alive {
                for &v in tri {
                    used[v as usize] = true;
                }
            }
        }
        let mut new_index = vec![0_u32; self.positions.len()];
        let mut vertices = Vec::new();
        let mut tex_coords = Vec::new();
        for (v, &is_used) in used.iter().enumerate() {
            if is_used {
                new_index[v] = vertices.len() as u32;
                vertices.push(self.positions[v]);
                tex_coords.push(self.uvs[v]);
            }
        }
        let indices: Vec<[u32; 3]> = self
            .triangles
            .iter()
            .zip(self.triangle_alive.iter())
            .filter(|&(_, &alive)| alive)
            .map(|(tri, _)| tri.map(|v| new_index[v as usize]))
            .collect();
        let mut mesh = GpuMesh {
            normals: vec![[0.0; 3]; vertices.len()],
            vertices,
            indices,
            tex_coords,
        };
        gpu_compute_normals(&mut mesh);
        mesh
    }
}

/// Simplify a mesh to approximately `target_triangles`.
///
/// Repeatedly collapses the shortest edge — merging its end points at the edge
/// midpoint — until the triangle count is at or below `target_triangles` or no
/// further collapses are possible.  A mesh that already has at most
/// `target_triangles` triangles is returned unchanged.
///
/// Edges are kept in a priority queue with lazy invalidation (per-vertex version
/// counters) and the vertex → triangle adjacency is updated per collapse, so the
/// cost is `O(T log T)` instead of a full rescan per collapse.  Collapses that
/// would change the topology (link condition), flip or degenerate a triangle, or
/// pull a boundary inwards are skipped — so a closed manifold stays a closed
/// manifold of the same genus and never shrinks below a tetrahedron.  A
/// boundary vertex is only ever merged with a boundary vertex along a boundary
/// edge (at the edge midpoint, which rounds off corners of the outline) or
/// absorbs an interior neighbour (staying where it is).
///
/// The result is compacted: only vertices used by a remaining triangle are
/// kept (in their original order), their texture coordinates follow the merge,
/// and the vertex normals are recomputed.  Triangles of the input that name a
/// missing vertex or repeat a vertex are dropped when any decimation happens.
pub fn gpu_mesh_decimate(mesh: &GpuMesh, target_triangles: usize) -> GpuMesh {
    if mesh.triangle_count() <= target_triangles {
        return mesh.clone();
    }
    let mut state = Decimator::new(mesh);
    let mut heap: BinaryHeap<Reverse<CollapseCandidate>> =
        state.initial_candidates().into_iter().collect();
    while state.alive_triangles > target_triangles {
        let Some(Reverse(candidate)) = heap.pop() else {
            break;
        };
        if !state.is_current(&candidate) {
            continue;
        }
        let Some(plan) = state.plan_collapse(candidate.keep, candidate.remove) else {
            continue;
        };
        state.collapse(candidate.keep, candidate.remove, &plan);
        state.requeue_around(candidate.keep, &mut heap);
    }
    state.into_mesh()
}

// ── Bounding box & measurements ──────────────────────────────────────────────

/// Compute the axis-aligned bounding box of the mesh.
///
/// Returns `(min, max)` corner arrays.  If the mesh has no vertices the
/// result is `([0,0,0], [0,0,0])`.
pub fn gpu_compute_aabb(mesh: &GpuMesh) -> ([f32; 3], [f32; 3]) {
    if mesh.vertices.is_empty() {
        return ([0.0; 3], [0.0; 3]);
    }
    let mut mn = [f32::MAX; 3];
    let mut mx = [f32::MIN; 3];
    for v in &mesh.vertices {
        for axis in 0..3 {
            mn[axis] = mn[axis].min(v[axis]);
            mx[axis] = mx[axis].max(v[axis]);
        }
    }
    (mn, mx)
}

/// Compute total surface area as the sum of triangle areas.
pub fn gpu_compute_surface_area(mesh: &GpuMesh) -> f32 {
    mesh.indices
        .iter()
        .map(|t| {
            let a = mesh.vertices[t[0] as usize];
            let b = mesh.vertices[t[1] as usize];
            let c = mesh.vertices[t[2] as usize];
            triangle_area(a, b, c)
        })
        .sum()
}

/// Compute the signed volume of the mesh via the divergence theorem.
///
/// Assumes the mesh forms a closed manifold.
pub fn gpu_compute_volume(mesh: &GpuMesh) -> f32 {
    let mut vol = 0.0f32;
    for t in &mesh.indices {
        let a = mesh.vertices[t[0] as usize];
        let b = mesh.vertices[t[1] as usize];
        let c = mesh.vertices[t[2] as usize];
        // Signed tetrahedral volume from origin.
        vol += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))
            / 6.0;
    }
    vol
}

// ── Vertex welding ────────────────────────────────────────────────────────────

/// Merge vertices that are within `tol` distance of each other.
///
/// Returns the number of vertices that were merged.  Vertices are visited in
/// index order; every vertex that has not itself been merged absorbs all later,
/// not yet merged vertices closer than `tol` (distance strictly below `tol`).
/// The merged vertices stay in the vertex arrays but are no longer referenced:
/// the index buffer is remapped, triangles that become degenerate are removed,
/// and so are triangles that name a vertex that does not exist.  A tolerance
/// that is not a positive finite number merges nothing.
///
/// Candidates come from a spatial hash with cells of the tolerance size, so the
/// work grows linearly with the vertex count for a bounded point density
/// instead of with its square.
pub fn gpu_weld_vertices(mesh: &mut GpuMesh, tol: f32) -> usize {
    let nv = mesh.vertex_count();
    let mut remap: Vec<usize> = (0..nv).collect();
    let mut merged = 0usize;

    if tol.is_finite() && tol > 0.0 {
        let tol_sq = tol * tol;
        // A cell edge a hair above `tol`, so that two points closer than `tol`
        // always fall into the same or adjacent cells despite rounding.
        let cell_edge = f64::from(tol) * (1.0 + 1.0e-5);
        let cell_of = |p: [f32; 3]| {
            [
                (f64::from(p[0]) / cell_edge).floor() as i64,
                (f64::from(p[1]) / cell_edge).floor() as i64,
                (f64::from(p[2]) / cell_edge).floor() as i64,
            ]
        };
        let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::with_capacity(nv);
        for (index, &position) in mesh.vertices.iter().enumerate() {
            grid.entry(cell_of(position)).or_default().push(index);
        }

        for (i, &pi) in mesh.vertices.iter().enumerate() {
            if remap[i] != i {
                continue;
            }
            let [cx, cy, cz] = cell_of(pi);
            for dx in -1..=1_i64 {
                for dy in -1..=1_i64 {
                    for dz in -1..=1_i64 {
                        let key = [
                            cx.saturating_add(dx),
                            cy.saturating_add(dy),
                            cz.saturating_add(dz),
                        ];
                        let Some(candidates) = grid.get(&key) else {
                            continue;
                        };
                        for &j in candidates {
                            if j > i && remap[j] == j && dist2_3f(pi, mesh.vertices[j]) < tol_sq {
                                remap[j] = i;
                                merged += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    mesh.indices
        .retain(|tri| triangle_vertices(tri, nv).is_some());
    for tri in mesh.indices.iter_mut() {
        for idx in tri.iter_mut() {
            *idx = remap[*idx as usize] as u32;
        }
    }
    mesh.indices
        .retain(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2]);
    merged
}

// ── Internal math helpers ─────────────────────────────────────────────────────

fn cross3f(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub3f(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3f(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length3f(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn normalize3f(v: [f32; 3]) -> [f32; 3] {
    let len = length3f(v);
    if len < 1e-9 {
        return [0.0; 3];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

fn dist2_3f(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

fn midpoint3f(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ]
}

#[cfg(test)]
mod tests;
