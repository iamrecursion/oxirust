// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Quadric-error-metric (QEM) edge-collapse decimation of a [`ProcessMesh`]
//! (Garland & Heckbert 1997).
//!
//! The decimator keeps its state incrementally instead of rescanning the mesh for
//! every collapse:
//!
//! * the vertex quadrics are computed once and merged on a collapse,
//!   `Q[keep] += Q[removed]`;
//! * a vertex -> faces adjacency lets a collapse rewrite only the faces around
//!   the two end points, and dead faces are never visited again;
//! * a binary heap holds the candidate edges.  Entries carry version stamps of
//!   both end points and are discarded lazily once a collapse has invalidated
//!   them, so only the edges around the surviving vertex are re-costed.
//!
//! A pass over a mesh with `F` triangles therefore costs `O(F log F)`.
//!
//! Every collapse is validated before it is applied:
//!
//! * the *link condition* (the end points of the edge share exactly the vertices
//!   opposite to it) and the manifold boundary rules keep the surface a manifold
//!   without pinched vertices or duplicated faces;
//! * the merged vertex of an interior edge is placed at the position that
//!   minimises the combined quadric (a 3x3 solve; the edge midpoint and both end
//!   points are the fall-backs when the system is singular or the optimum is far
//!   from the edge), and the cheapest candidate that passes the next check wins;
//! * a collapse is rejected when it would flip the normal of (or collapse to a
//!   line) any triangle around the merged vertex.
//!
//! Open boundaries are preserved twice over: heavily weighted planes through the
//! boundary edges, perpendicular to their triangle (Garland & Heckbert 1998), make
//! bending the boundary expensive, and a collapse that involves a boundary vertex
//! never invents a new position: a boundary vertex that absorbs an interior one
//! stays where it is, and the two end points of a boundary edge merge into one of
//! them.  The boundary of the result therefore consists of boundary vertices of
//! the input only.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use super::functions::{
    Face, Vertex, vec3_add, vec3_cross, vec3_dot, vec3_len, vec3_normalize, vec3_scale, vec3_sub,
    vertex_quadrics,
};
use super::types::{ProcessMesh, Quadric};

/// Weight of the boundary-preserving planes relative to the face planes.
const BOUNDARY_WEIGHT: f64 = 1000.0;
/// Weight of the squared edge length in the collapse priority.
///
/// Collapses that are equally cheap in the quadric metric (every collapse inside
/// a flat region costs zero) are resolved in favour of the shorter edge.
const EDGE_LENGTH_TIE_BREAK: f64 = 1e-9;
/// The quadric optimum is only used when it lies within this many edge lengths of
/// the edge midpoint; a nearly singular system otherwise puts it far away.
const MAX_OPTIMUM_DISTANCE: f64 = 2.0;
/// The 3x3 system of the optimum is treated as singular when its determinant is
/// below this fraction of `(trace / 3)^3`, the largest determinant a positive
/// semi-definite matrix of that trace can have.
const SINGULAR_DETERMINANT_RATIO: f64 = 1e-10;
/// A triangle around a collapse may not shrink below this fraction of its
/// squared area (twice-area vector length squared), which rejects collapses that
/// turn a triangle into a line.
const MIN_SQUARED_AREA_RATIO: f64 = 1e-16;

/// A candidate edge collapse in the priority queue.
struct Candidate {
    /// Quadric error of the collapse plus the edge-length tie-break.
    priority: f64,
    /// Smaller vertex index of the edge; this vertex survives the collapse.
    lo: usize,
    /// Larger vertex index of the edge; this vertex is merged into `lo`.
    hi: usize,
    /// Version stamp of `lo` when the entry was created.
    lo_version: u32,
    /// Version stamp of `hi` when the entry was created.
    hi_version: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    /// Reversed so that `BinaryHeap` pops the smallest `(priority, lo, hi)`
    /// first; ties are broken on the vertex pair, which keeps the result
    /// independent of hash-map iteration order.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .priority
            .total_cmp(&self.priority)
            .then_with(|| other.lo.cmp(&self.lo))
            .then_with(|| other.hi.cmp(&self.hi))
    }
}

/// A validated collapse: where the merged vertex goes and what it costs.
struct CollapsePlan {
    /// Position of the merged vertex.
    position: Vertex,
    /// Priority-queue key of the collapse.
    priority: f64,
}

/// The incremental edge-collapse state of one decimation run.
struct Decimator {
    /// Working vertex positions.
    positions: Vec<Vertex>,
    /// Accumulated quadric of every vertex.
    quadrics: Vec<Quadric>,
    /// Working triangle list (indices never move; dead triangles stay in place).
    faces: Vec<Face>,
    /// Whether each triangle is still part of the mesh.
    face_alive: Vec<bool>,
    /// Number of triangles that are still alive.
    live_faces: usize,
    /// Live triangles incident to every vertex.
    vertex_faces: Vec<Vec<usize>>,
    /// Whether each vertex has not been merged into another one.
    vertex_alive: Vec<bool>,
    /// Version stamp of each vertex, bumped whenever its quadric or position changes.
    version: Vec<u32>,
    /// Candidate collapses, cheapest first.
    heap: BinaryHeap<Candidate>,
}

/// Scale every coefficient of `quadric` by `weight`.
fn scaled_quadric(quadric: &Quadric, weight: f64) -> Quadric {
    let mut out = *quadric;
    for coefficient in out.q.iter_mut() {
        *coefficient *= weight;
    }
    out
}

/// Twice-area normal vector `(p1 - p0) x (p2 - p0)` of a triangle.
fn area_vector(points: [Vertex; 3]) -> Vertex {
    vec3_cross(
        vec3_sub(points[1], points[0]),
        vec3_sub(points[2], points[0]),
    )
}

/// The position that minimises the quadric `q`, or `None` when the 3x3 system is
/// (numerically) singular.
fn optimal_position(q: &Quadric) -> Option<Vertex> {
    let [a00, a01, a02, b0, a11, a12, b1, a22, b2, _] = q.q;
    let cof00 = a11 * a22 - a12 * a12;
    let cof01 = a02 * a12 - a01 * a22;
    let cof02 = a01 * a12 - a02 * a11;
    let det = a00 * cof00 + a01 * cof01 + a02 * cof02;
    let mean_diagonal = (a00 + a11 + a22) / 3.0;
    if det.is_nan() || det.abs() <= SINGULAR_DETERMINANT_RATIO * mean_diagonal.powi(3) {
        return None;
    }
    let cof11 = a00 * a22 - a02 * a02;
    let cof12 = a01 * a02 - a00 * a12;
    let cof22 = a00 * a11 - a01 * a01;
    let inv_det = 1.0 / det;
    Some([
        -inv_det * (cof00 * b0 + cof01 * b1 + cof02 * b2),
        -inv_det * (cof01 * b0 + cof11 * b1 + cof12 * b2),
        -inv_det * (cof02 * b0 + cof12 * b1 + cof22 * b2),
    ])
}

impl Decimator {
    /// Build the decimator for `mesh`.
    ///
    /// Returns `None` when a triangle references a vertex outside the mesh.
    /// Triangles with a repeated vertex index are dropped (they carry no area).
    fn new(mesh: &ProcessMesh) -> Option<Self> {
        let nv = mesh.verts.len();
        if mesh.faces.iter().flatten().any(|&v| v >= nv) {
            return None;
        }
        let faces = mesh.faces.clone();
        let face_alive: Vec<bool> = faces
            .iter()
            .map(|&[a, b, c]| a != b && b != c && a != c)
            .collect();
        let live_faces = face_alive.iter().filter(|&&alive| alive).count();

        let mut vertex_faces: Vec<Vec<usize>> = vec![Vec::new(); nv];
        let mut edge_uses: HashMap<(usize, usize), u32> = HashMap::new();
        for (f, &[a, b, c]) in faces.iter().enumerate() {
            if !face_alive[f] {
                continue;
            }
            for v in [a, b, c] {
                vertex_faces[v].push(f);
            }
            for (u, v) in [(a, b), (b, c), (c, a)] {
                *edge_uses.entry((u.min(v), u.max(v))).or_insert(0) += 1;
            }
        }

        let mut quadrics = vertex_quadrics(mesh);
        // Pin the open boundary: a plane through every boundary edge,
        // perpendicular to its triangle, makes moving a boundary vertex off the
        // boundary expensive.  The faces are visited in order so the summation
        // order (and with it the result) is deterministic.
        for (f, &[a, b, c]) in faces.iter().enumerate() {
            if !face_alive[f] {
                continue;
            }
            let normal = vec3_normalize(area_vector([mesh.verts[a], mesh.verts[b], mesh.verts[c]]));
            if vec3_len(normal) < 0.5 {
                continue;
            }
            for (u, v) in [(a, b), (b, c), (c, a)] {
                if edge_uses.get(&(u.min(v), u.max(v))) != Some(&1) {
                    continue;
                }
                let edge = vec3_sub(mesh.verts[v], mesh.verts[u]);
                let wall = vec3_normalize(vec3_cross(edge, normal));
                if vec3_len(wall) < 0.5 {
                    continue;
                }
                let offset = -vec3_dot(wall, mesh.verts[u]);
                let plane = scaled_quadric(
                    &Quadric::from_plane(wall[0], wall[1], wall[2], offset),
                    BOUNDARY_WEIGHT,
                );
                quadrics[u] = quadrics[u].add(&plane);
                quadrics[v] = quadrics[v].add(&plane);
            }
        }

        Some(Self {
            positions: mesh.verts.clone(),
            quadrics,
            faces,
            face_alive,
            live_faces,
            vertex_faces,
            vertex_alive: vec![true; nv],
            version: vec![0; nv],
            heap: BinaryHeap::new(),
        })
    }

    /// Live triangles that contain both `a` and `b`.
    fn shared_faces(&self, a: usize, b: usize) -> Vec<usize> {
        self.vertex_faces[a]
            .iter()
            .copied()
            .filter(|&f| self.faces[f].contains(&b))
            .collect()
    }

    /// Sorted vertices that share a live triangle with `v`.
    fn neighbours(&self, v: usize) -> Vec<usize> {
        let mut out: Vec<usize> = self.vertex_faces[v]
            .iter()
            .flat_map(|&f| self.faces[f])
            .filter(|&w| w != v)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Whether `v` lies on an open boundary, i.e. one of its edges belongs to a
    /// single live triangle.
    fn is_boundary_vertex(&self, v: usize) -> bool {
        self.vertex_faces[v].iter().any(|&f| {
            self.faces[f].iter().any(|&w| {
                w != v
                    && self.vertex_faces[v]
                        .iter()
                        .filter(|&&g| self.faces[g].contains(&w))
                        .count()
                        == 1
            })
        })
    }

    /// Whether collapsing the edge `(lo, hi)` keeps the surface a manifold;
    /// `lo_boundary` and `hi_boundary` tell which end points lie on a boundary.
    fn topology_allows_collapse(
        &self,
        lo: usize,
        hi: usize,
        shared: &[usize],
        lo_boundary: bool,
        hi_boundary: bool,
    ) -> bool {
        // Non-manifold edges (more than two triangles) never collapse.
        if shared.is_empty() || shared.len() > 2 {
            return false;
        }
        // Link condition: the end points have exactly the vertices opposite to
        // the edge in common, otherwise the collapse pinches the surface.
        let lo_neighbours = self.neighbours(lo);
        let hi_neighbours = self.neighbours(hi);
        let common: Vec<usize> = lo_neighbours
            .iter()
            .copied()
            .filter(|&w| w != hi && hi_neighbours.binary_search(&w).is_ok())
            .collect();
        let mut opposite: Vec<usize> = shared
            .iter()
            .filter_map(|&f| self.faces[f].iter().copied().find(|&w| w != lo && w != hi))
            .collect();
        opposite.sort_unstable();
        opposite.dedup();
        if common != opposite {
            return false;
        }
        // Two boundary vertices joined by an interior edge would be welded
        // together, closing a handle across the boundary.
        if lo_boundary && hi_boundary && shared.len() == 2 {
            return false;
        }
        // The merged vertex keeps the triangles of both end points except the
        // ones on the edge; an interior vertex needs a closed fan of at least
        // three of them.
        let surviving =
            self.vertex_faces[lo].len() + self.vertex_faces[hi].len() - 2 * shared.len();
        let required = if lo_boundary || hi_boundary { 1 } else { 3 };
        surviving >= required
    }

    /// Whether moving the merged vertex to `target` keeps every surrounding
    /// triangle oriented as before and non-degenerate.
    fn collapse_keeps_faces_valid(
        &self,
        lo: usize,
        hi: usize,
        shared: &[usize],
        target: Vertex,
    ) -> bool {
        for endpoint in [lo, hi] {
            for &f in &self.vertex_faces[endpoint] {
                if shared.contains(&f) {
                    continue;
                }
                let tri = self.faces[f];
                let before = area_vector(tri.map(|v| self.positions[v]));
                let before_sq = vec3_dot(before, before);
                if before_sq <= f64::MIN_POSITIVE {
                    // A triangle that has no orientation cannot be flipped.
                    continue;
                }
                let after = area_vector(tri.map(|v| {
                    if v == lo || v == hi {
                        target
                    } else {
                        self.positions[v]
                    }
                }));
                let after_sq = vec3_dot(after, after);
                if after_sq <= MIN_SQUARED_AREA_RATIO * before_sq || vec3_dot(before, after) <= 0.0
                {
                    return false;
                }
            }
        }
        true
    }

    /// Validate the collapse of `(lo, hi)` against the current mesh and pick the
    /// cheapest admissible position for the merged vertex.
    fn plan_collapse(&self, lo: usize, hi: usize) -> Option<CollapsePlan> {
        if !self.vertex_alive[lo] || !self.vertex_alive[hi] {
            return None;
        }
        let shared = self.shared_faces(lo, hi);
        let lo_boundary = self.is_boundary_vertex(lo);
        let hi_boundary = self.is_boundary_vertex(hi);
        if !self.topology_allows_collapse(lo, hi, &shared, lo_boundary, hi_boundary) {
            return None;
        }
        let p_lo = self.positions[lo];
        let p_hi = self.positions[hi];
        let midpoint = vec3_scale(vec3_add(p_lo, p_hi), 0.5);
        let length = vec3_len(vec3_sub(p_hi, p_lo));
        let q = self.quadrics[lo].add(&self.quadrics[hi]);

        // Candidates in order of preference; on a tie the earlier one wins.  The
        // boundary is never given a new position: a boundary vertex that absorbs
        // an interior one stays put, and the two end points of a boundary edge
        // (the topology check rules out any other edge between boundary
        // vertices) merge into one of them.
        let mut candidates: Vec<Vertex> = Vec::with_capacity(4);
        match (lo_boundary, hi_boundary) {
            (false, false) => {
                if let Some(optimum) = optimal_position(&q)
                    && optimum.iter().all(|c| c.is_finite())
                    && vec3_len(vec3_sub(optimum, midpoint)) <= MAX_OPTIMUM_DISTANCE * length
                {
                    candidates.push(optimum);
                }
                candidates.extend([midpoint, p_lo, p_hi]);
            }
            (true, false) => candidates.push(p_lo),
            (false, true) => candidates.push(p_hi),
            (true, true) => candidates.extend([p_lo, p_hi]),
        }

        let mut best: Option<(f64, Vertex)> = None;
        for position in candidates {
            let raw_cost = q.evaluate(position);
            if !raw_cost.is_finite() {
                continue;
            }
            let cost = raw_cost.max(0.0);
            if best.is_some_and(|(best_cost, _)| cost >= best_cost) {
                continue;
            }
            if self.collapse_keeps_faces_valid(lo, hi, &shared, position) {
                best = Some((cost, position));
            }
        }
        best.map(|(cost, position)| CollapsePlan {
            position,
            priority: cost + EDGE_LENGTH_TIE_BREAK * length * length,
        })
    }

    /// Queue the edge `(lo, hi)` (with `lo < hi`) if it may collapse.
    fn push_edge(&mut self, lo: usize, hi: usize) {
        if let Some(plan) = self.plan_collapse(lo, hi) {
            self.heap.push(Candidate {
                priority: plan.priority,
                lo,
                hi,
                lo_version: self.version[lo],
                hi_version: self.version[hi],
            });
        }
    }

    /// Replace the heap by every edge of the live mesh, in a deterministic order.
    fn seed_heap(&mut self) {
        self.heap.clear();
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for (f, &[a, b, c]) in self.faces.iter().enumerate() {
            if self.face_alive[f] {
                for (p, q) in [(a, b), (b, c), (c, a)] {
                    edges.push((p.min(q), p.max(q)));
                }
            }
        }
        edges.sort_unstable();
        edges.dedup();
        for (lo, hi) in edges {
            self.push_edge(lo, hi);
        }
    }

    /// Whether a queued entry still describes the current mesh.
    fn is_current(&self, entry: &Candidate) -> bool {
        self.vertex_alive[entry.lo]
            && self.vertex_alive[entry.hi]
            && self.version[entry.lo] == entry.lo_version
            && self.version[entry.hi] == entry.hi_version
    }

    /// Merge `hi` into `lo` at `position` and queue the edges around `lo` again.
    fn apply_collapse(&mut self, lo: usize, hi: usize, position: Vertex) {
        let incident = std::mem::take(&mut self.vertex_faces[hi]);
        for f in incident {
            let tri = self.faces[f];
            if tri.contains(&lo) {
                // The triangle contained the collapsed edge: it vanishes.
                self.face_alive[f] = false;
                self.live_faces -= 1;
                for v in tri {
                    if v != hi {
                        self.vertex_faces[v].retain(|&g| g != f);
                    }
                }
            } else {
                self.faces[f] = tri.map(|v| if v == hi { lo } else { v });
                self.vertex_faces[lo].push(f);
            }
        }
        self.vertex_alive[hi] = false;
        self.positions[lo] = position;
        let merged = self.quadrics[lo].add(&self.quadrics[hi]);
        self.quadrics[lo] = merged;
        self.quadrics[hi] = Quadric::zero();
        self.version[lo] = self.version[lo].wrapping_add(1);
        for w in self.neighbours(lo) {
            self.push_edge(lo.min(w), lo.max(w));
        }
    }

    /// Collapse the cheapest admissible edges until at most `target` triangles
    /// are left or no admissible edge remains.
    fn decimate_to(&mut self, target: usize) {
        loop {
            self.seed_heap();
            let mut progressed = false;
            while self.live_faces > target {
                let Some(entry) = self.heap.pop() else {
                    break;
                };
                if !self.is_current(&entry) {
                    continue;
                }
                // The triangles around the edge may have changed since the entry
                // was queued, so validate the collapse against the current mesh.
                let Some(plan) = self.plan_collapse(entry.lo, entry.hi) else {
                    continue;
                };
                // If it got more expensive than a waiting collapse, requeue it.
                let overtaken = plan.priority > entry.priority
                    && self
                        .heap
                        .peek()
                        .is_some_and(|next| plan.priority > next.priority);
                if overtaken {
                    self.heap.push(Candidate {
                        priority: plan.priority,
                        ..entry
                    });
                    continue;
                }
                self.apply_collapse(entry.lo, entry.hi, plan.position);
                progressed = true;
            }
            // An exhausted heap does not prove that no collapse is left: edges
            // rejected earlier may have become admissible.  Start another pass
            // until a whole pass makes no progress.
            if self.live_faces <= target || !progressed {
                break;
            }
        }
    }

    /// The decimated mesh: live triangles in their original order and the
    /// vertices they reference, in their original order.
    fn into_mesh(self) -> ProcessMesh {
        let nv = self.positions.len();
        let mut referenced = vec![false; nv];
        for (tri, &alive) in self.faces.iter().zip(self.face_alive.iter()) {
            if alive {
                for &v in tri {
                    referenced[v] = true;
                }
            }
        }
        let mut remap = vec![usize::MAX; nv];
        let mut verts: Vec<Vertex> = Vec::new();
        for v in 0..nv {
            if referenced[v] {
                remap[v] = verts.len();
                verts.push(self.positions[v]);
            }
        }
        let faces: Vec<Face> = self
            .faces
            .iter()
            .zip(self.face_alive.iter())
            .filter(|&(_, &alive)| alive)
            .map(|(&[a, b, c], _)| [remap[a], remap[b], remap[c]])
            .collect();
        ProcessMesh::new(verts, faces)
    }
}

/// Decimate a mesh to at most `target_faces` triangles using quadric error
/// metrics with greedy edge collapses (Garland & Heckbert).
///
/// The cheapest edge is collapsed first; the merged vertex is placed where the
/// combined quadric of its end points is minimal.  The cost of a pass is
/// `O(F log F)` for `F` triangles.  Collapses that would pinch the surface,
/// flip or flatten a triangle are never performed, and the vertices of an open
/// boundary are never moved (only removed by merging them into a neighbouring
/// boundary vertex), so decimating a manifold mesh gives a manifold whose
/// boundary is made of boundary vertices of the input.  Decimation stops early,
/// with more than `target_faces` triangles left, when no admissible collapse
/// remains (a closed mesh keeps at least four triangles, for example).
///
/// The output is compacted: vertices that no remaining triangle references are
/// dropped.  Per-vertex normals and UVs are not carried over.  A mesh that is
/// already small enough, or that references vertices which do not exist, is
/// returned unchanged.
pub fn qem_decimate(mesh: &ProcessMesh, target_faces: usize) -> ProcessMesh {
    if mesh.faces.len() <= target_faces {
        return mesh.clone();
    }
    let Some(mut decimator) = Decimator::new(mesh) else {
        return mesh.clone();
    };
    decimator.decimate_to(target_faces);
    decimator.into_mesh()
}
