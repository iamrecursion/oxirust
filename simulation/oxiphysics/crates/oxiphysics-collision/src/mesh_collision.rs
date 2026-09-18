// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Mesh collision detection: BVH for triangle meshes (AABB tree build/query),
//! ray-triangle (Möller-Trumbore) and triangle-triangle (separating axis)
//! intersection, mesh-mesh broad+narrow phase,
//! deformable mesh collision (BVH refit), cloth self-collision, continuous collision
//! detection for deformables, proximity query (distance field), swept triangle vs
//! static mesh, penetration depth estimation, and contact manifold generation.

use crate::deformable_collision::{ccd_edge_edge_analytic, ccd_vertex_face_analytic};

mod cloth_grid;
#[cfg(test)]
mod intersection_tests;
#[cfg(test)]
mod regression_tests;
mod tri_tri;

// ─────────────────────────────────────────────────────────────────────────────
// Vector math (no nalgebra — use [f64; 3] arrays)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a - b component-wise.
#[inline]
pub fn vec3_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Compute a + b component-wise.
#[inline]
pub fn vec3_add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scale a vector.
#[inline]
pub fn vec3_scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Dot product.
#[inline]
pub fn vec3_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product.
#[inline]
pub fn vec3_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Squared length.
#[inline]
pub fn vec3_len_sq(a: [f64; 3]) -> f64 {
    vec3_dot(a, a)
}

/// Length.
#[inline]
pub fn vec3_len(a: [f64; 3]) -> f64 {
    vec3_len_sq(a).sqrt()
}

/// Normalise; returns zero vector if near-zero input.
#[inline]
pub fn vec3_norm(a: [f64; 3]) -> [f64; 3] {
    let l = vec3_len(a);
    if l < 1e-30 {
        [0.0; 3]
    } else {
        vec3_scale(a, 1.0 / l)
    }
}

/// Component-wise minimum.
#[inline]
pub fn vec3_min(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])]
}

/// Component-wise maximum.
#[inline]
pub fn vec3_max(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])]
}

/// Linear interpolation.
#[inline]
pub fn vec3_lerp(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    vec3_add(vec3_scale(a, 1.0 - t), vec3_scale(b, t))
}

// ─────────────────────────────────────────────────────────────────────────────
// AABB (Axis-Aligned Bounding Box)
// ─────────────────────────────────────────────────────────────────────────────

/// An axis-aligned bounding box in 3D.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshAabb {
    /// Minimum corner.
    pub min: [f64; 3],
    /// Maximum corner.
    pub max: [f64; 3],
}

impl MeshAabb {
    /// Create an empty (inverted) AABB.
    pub fn empty() -> Self {
        Self {
            min: [f64::INFINITY; 3],
            max: [f64::NEG_INFINITY; 3],
        }
    }

    /// Create an AABB from min and max corners.
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        Self { min, max }
    }

    /// Expand to include a point.
    pub fn expand_point(&mut self, p: [f64; 3]) {
        self.min = vec3_min(self.min, p);
        self.max = vec3_max(self.max, p);
    }

    /// Expand to include another AABB.
    pub fn expand_aabb(&mut self, other: &MeshAabb) {
        self.min = vec3_min(self.min, other.min);
        self.max = vec3_max(self.max, other.max);
    }

    /// Centre of the AABB.
    pub fn centre(&self) -> [f64; 3] {
        vec3_scale(vec3_add(self.min, self.max), 0.5)
    }

    /// Half-extents.
    pub fn half_extents(&self) -> [f64; 3] {
        vec3_scale(vec3_sub(self.max, self.min), 0.5)
    }

    /// Surface area.
    pub fn surface_area(&self) -> f64 {
        let d = vec3_sub(self.max, self.min);
        2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
    }

    /// Does this AABB overlap with `other`?
    pub fn overlaps(&self, other: &MeshAabb) -> bool {
        self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
    }

    /// Does this AABB contain the point `p`?
    pub fn contains_point(&self, p: [f64; 3]) -> bool {
        p[0] >= self.min[0]
            && p[0] <= self.max[0]
            && p[1] >= self.min[1]
            && p[1] <= self.max[1]
            && p[2] >= self.min[2]
            && p[2] <= self.max[2]
    }

    /// Build AABB for a triangle.
    pub fn from_triangle(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> Self {
        let mut aabb = Self::empty();
        aabb.expand_point(a);
        aabb.expand_point(b);
        aabb.expand_point(c);
        aabb
    }

    /// Expand by a margin (fat AABB for CCD).
    pub fn fattened(&self, margin: f64) -> Self {
        Self {
            min: [
                self.min[0] - margin,
                self.min[1] - margin,
                self.min[2] - margin,
            ],
            max: [
                self.max[0] + margin,
                self.max[1] + margin,
                self.max[2] + margin,
            ],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Triangle
// ─────────────────────────────────────────────────────────────────────────────

/// A triangle defined by three vertices.
#[derive(Clone, Debug)]
pub struct Triangle {
    /// First vertex.
    pub v0: [f64; 3],
    /// Second vertex.
    pub v1: [f64; 3],
    /// Third vertex.
    pub v2: [f64; 3],
}

impl Triangle {
    /// Create a new [`Triangle`].
    pub fn new(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3]) -> Self {
        Self { v0, v1, v2 }
    }

    /// Outward unit normal.
    pub fn normal(&self) -> [f64; 3] {
        let e1 = vec3_sub(self.v1, self.v0);
        let e2 = vec3_sub(self.v2, self.v0);
        vec3_norm(vec3_cross(e1, e2))
    }

    /// Area of the triangle.
    pub fn area(&self) -> f64 {
        let e1 = vec3_sub(self.v1, self.v0);
        let e2 = vec3_sub(self.v2, self.v0);
        vec3_len(vec3_cross(e1, e2)) * 0.5
    }

    /// Centroid.
    pub fn centroid(&self) -> [f64; 3] {
        [
            (self.v0[0] + self.v1[0] + self.v2[0]) / 3.0,
            (self.v0[1] + self.v1[1] + self.v2[1]) / 3.0,
            (self.v0[2] + self.v1[2] + self.v2[2]) / 3.0,
        ]
    }

    /// AABB of the triangle.
    pub fn aabb(&self) -> MeshAabb {
        MeshAabb::from_triangle(self.v0, self.v1, self.v2)
    }

    /// Point at barycentric coordinates (u, v).
    pub fn barycentric_point(&self, u: f64, v: f64) -> [f64; 3] {
        let w = 1.0 - u - v;
        [
            w * self.v0[0] + u * self.v1[0] + v * self.v2[0],
            w * self.v0[1] + u * self.v1[1] + v * self.v2[1],
            w * self.v0[2] + u * self.v1[2] + v * self.v2[2],
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Möller-Trumbore ray-triangle intersection
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a ray-triangle intersection test.
#[derive(Clone, Debug)]
pub struct RayTriHit {
    /// Ray parameter t at the hit point.
    pub t: f64,
    /// Barycentric u coordinate.
    pub u: f64,
    /// Barycentric v coordinate.
    pub v: f64,
}

/// Möller-Trumbore ray-triangle intersection.
///
/// Returns `Some(RayTriHit)` if the ray origin + t * direction hits the triangle
/// for t in \[t_min, t_max\].
pub fn ray_triangle_mt(
    origin: [f64; 3],
    direction: [f64; 3],
    tri: &Triangle,
    t_min: f64,
    t_max: f64,
) -> Option<RayTriHit> {
    const EPS: f64 = 1e-10;
    let e1 = vec3_sub(tri.v1, tri.v0);
    let e2 = vec3_sub(tri.v2, tri.v0);
    let h = vec3_cross(direction, e2);
    let a = vec3_dot(e1, h);
    if a.abs() < EPS {
        return None;
    }
    let f = 1.0 / a;
    let s = vec3_sub(origin, tri.v0);
    let u = f * vec3_dot(s, h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = vec3_cross(s, e1);
    let v = f * vec3_dot(direction, q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * vec3_dot(e2, q);
    if t < t_min || t > t_max {
        return None;
    }
    Some(RayTriHit { t, u, v })
}

// ─────────────────────────────────────────────────────────────────────────────
// Triangle-Triangle intersection (separating axis test)
// ─────────────────────────────────────────────────────────────────────────────

/// Test whether two triangles intersect.
///
/// The triangles are treated as closed sets: touching at a vertex, along an edge
/// or inside a shared plane counts as intersecting. The test is the separating
/// axis theorem over the two face normals, the nine edge-edge cross products and
/// the in-plane edge normals, so coplanar pairs need no special case (the old
/// signed-distance interval test divided `0/0` there and reported every coplanar
/// pair as intersecting). A gap of at most `1e-12` times the extent of the two
/// triangles counts as contact.
///
/// A zero-area triangle acts as the segment (or point) it collapses to, and a
/// triangle with a non-finite coordinate never intersects anything.
pub fn triangle_triangle_intersect(t1: &Triangle, t2: &Triangle) -> bool {
    tri_tri::triangles_intersect(&[t1.v0, t1.v1, t1.v2], &[t2.v0, t2.v1, t2.v2])
}

// ─────────────────────────────────────────────────────────────────────────────
// BVH node
// ─────────────────────────────────────────────────────────────────────────────

/// A node in a BVH tree.
#[derive(Clone, Debug)]
pub struct BvhNode {
    /// AABB of this node.
    pub aabb: MeshAabb,
    /// Left child index (usize::MAX if leaf).
    pub left: usize,
    /// Right child index (usize::MAX if leaf).
    pub right: usize,
    /// Triangle index (only valid for leaves).
    pub tri_idx: usize,
    /// Is this a leaf node?
    pub is_leaf: bool,
}

impl BvhNode {
    /// Create a leaf node.
    pub fn leaf(aabb: MeshAabb, tri_idx: usize) -> Self {
        Self {
            aabb,
            left: usize::MAX,
            right: usize::MAX,
            tri_idx,
            is_leaf: true,
        }
    }

    /// Create an internal node.
    pub fn internal(aabb: MeshAabb, left: usize, right: usize) -> Self {
        Self {
            aabb,
            left,
            right,
            tri_idx: usize::MAX,
            is_leaf: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Triangle mesh BVH
// ─────────────────────────────────────────────────────────────────────────────

/// A triangle mesh with an AABB BVH for fast collision queries.
pub struct TriMeshBvh {
    /// Triangle vertices in flat form: \[tri_idx\]\[vert_idx\]\[xyz\].
    pub triangles: Vec<Triangle>,
    /// BVH nodes.
    pub nodes: Vec<BvhNode>,
    /// Root node index.
    pub root: usize,
}

impl TriMeshBvh {
    /// Build a BVH from a list of triangles using the SAH-inspired median split.
    ///
    /// An empty triangle list yields an empty tree: `nodes` is empty and `root`
    /// is the `usize::MAX` sentinel. Every traversal ([`Self::ray_cast`],
    /// [`Self::query_aabb`], [`Self::refit`], [`mesh_broad_phase`]) treats a node
    /// index outside `nodes` as "nothing to visit", so queries on an empty mesh
    /// simply return no hits.
    pub fn build(triangles: Vec<Triangle>) -> Self {
        let n = triangles.len();
        if n == 0 {
            return Self {
                triangles,
                nodes: Vec::new(),
                root: usize::MAX,
            };
        }
        let mut nodes: Vec<BvhNode> = Vec::with_capacity(2 * n);
        let indices: Vec<usize> = (0..n).collect();
        let root = Self::build_recursive(&triangles, &indices, &mut nodes);
        Self {
            triangles,
            nodes,
            root,
        }
    }

    fn build_recursive(tris: &[Triangle], indices: &[usize], nodes: &mut Vec<BvhNode>) -> usize {
        // An empty index range has no node; `usize::MAX` is the "no node"
        // sentinel that every traversal already skips. Without this guard the
        // median split below would recurse on the same empty slice forever.
        if indices.is_empty() {
            return usize::MAX;
        }
        // Compute AABB for all triangles in this node
        let mut aabb = MeshAabb::empty();
        for &i in indices {
            aabb.expand_aabb(&tris[i].aabb());
        }

        if indices.len() == 1 {
            let node = BvhNode::leaf(aabb, indices[0]);
            let idx = nodes.len();
            nodes.push(node);
            return idx;
        }

        // Find longest axis to split along
        let ext = vec3_sub(aabb.max, aabb.min);
        let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] {
            0
        } else if ext[1] >= ext[2] {
            1
        } else {
            2
        };

        // Sort by centroid on chosen axis
        let mut sorted: Vec<usize> = indices.to_vec();
        sorted.sort_by(|&a, &b| {
            tris[a].centroid()[axis]
                .partial_cmp(&tris[b].centroid()[axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mid = sorted.len() / 2;
        let left = Self::build_recursive(tris, &sorted[..mid], nodes);
        let right = Self::build_recursive(tris, &sorted[mid..], nodes);

        let node = BvhNode::internal(aabb, left, right);
        let idx = nodes.len();
        nodes.push(node);
        idx
    }

    /// Ray cast into the BVH. Returns the closest hit.
    pub fn ray_cast(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        t_max: f64,
    ) -> Option<(RayTriHit, usize)> {
        let mut best: Option<(RayTriHit, usize)> = None;
        let mut stack = vec![self.root];
        let mut t_cur = t_max;

        while let Some(node_idx) = stack.pop() {
            if node_idx >= self.nodes.len() {
                continue;
            }
            let node = &self.nodes[node_idx];
            // Ray-AABB test
            if !ray_aabb_intersect(origin, direction, &node.aabb, 0.0, t_cur) {
                continue;
            }
            if node.is_leaf {
                let tri = &self.triangles[node.tri_idx];
                if let Some(hit) = ray_triangle_mt(origin, direction, tri, 0.0, t_cur) {
                    t_cur = hit.t;
                    best = Some((hit, node.tri_idx));
                }
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
        best
    }

    /// Collect all triangle indices whose AABB overlaps the query AABB.
    pub fn query_aabb(&self, query: &MeshAabb) -> Vec<usize> {
        let mut result = Vec::new();
        let mut stack = vec![self.root];
        while let Some(node_idx) = stack.pop() {
            if node_idx >= self.nodes.len() {
                continue;
            }
            let node = &self.nodes[node_idx];
            if !node.aabb.overlaps(query) {
                continue;
            }
            if node.is_leaf {
                result.push(node.tri_idx);
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
        result
    }

    /// Refit BVH after vertex positions have changed (for deformable meshes).
    pub fn refit(&mut self) {
        Self::refit_recursive(&self.triangles, &mut self.nodes, self.root);
    }

    fn refit_recursive(tris: &[Triangle], nodes: &mut Vec<BvhNode>, node_idx: usize) {
        if node_idx >= nodes.len() {
            return;
        }
        let is_leaf = nodes[node_idx].is_leaf;
        if is_leaf {
            let tri_idx = nodes[node_idx].tri_idx;
            nodes[node_idx].aabb = tris[tri_idx].aabb();
            return;
        }
        let left = nodes[node_idx].left;
        let right = nodes[node_idx].right;
        Self::refit_recursive(tris, nodes, left);
        Self::refit_recursive(tris, nodes, right);
        let mut aabb = nodes[left].aabb.clone();
        aabb.expand_aabb(&nodes[right].aabb.clone());
        nodes[node_idx].aabb = aabb;
    }

    /// Number of triangles.
    pub fn num_triangles(&self) -> usize {
        self.triangles.len()
    }

    /// Number of BVH nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

/// Ray-AABB intersection test (slab method).
pub fn ray_aabb_intersect(
    origin: [f64; 3],
    direction: [f64; 3],
    aabb: &MeshAabb,
    t_min: f64,
    t_max: f64,
) -> bool {
    let mut tmin = t_min;
    let mut tmax = t_max;
    for i in 0..3 {
        let inv = if direction[i].abs() > 1e-30 {
            1.0 / direction[i]
        } else {
            f64::INFINITY
        };
        let t1 = (aabb.min[i] - origin[i]) * inv;
        let t2 = (aabb.max[i] - origin[i]) * inv;
        let (lo, hi) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        tmin = tmin.max(lo);
        tmax = tmax.min(hi);
        if tmax < tmin {
            return false;
        }
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Mesh-mesh broad phase + narrow phase
// ─────────────────────────────────────────────────────────────────────────────

/// A candidate pair of triangle indices for narrow phase testing.
#[derive(Clone, Debug)]
pub struct TriPair {
    /// Triangle index in mesh A.
    pub tri_a: usize,
    /// Triangle index in mesh B.
    pub tri_b: usize,
}

/// Broad phase: find candidate triangle pairs between two BVH meshes.
pub fn mesh_broad_phase(a: &TriMeshBvh, b: &TriMeshBvh) -> Vec<TriPair> {
    let mut pairs = Vec::new();
    mesh_broad_phase_recursive(a, b, a.root, b.root, &mut pairs);
    pairs
}

fn mesh_broad_phase_recursive(
    a: &TriMeshBvh,
    b: &TriMeshBvh,
    na: usize,
    nb: usize,
    pairs: &mut Vec<TriPair>,
) {
    if na >= a.nodes.len() || nb >= b.nodes.len() {
        return;
    }
    if !a.nodes[na].aabb.overlaps(&b.nodes[nb].aabb) {
        return;
    }

    let a_leaf = a.nodes[na].is_leaf;
    let b_leaf = b.nodes[nb].is_leaf;

    match (a_leaf, b_leaf) {
        (true, true) => {
            pairs.push(TriPair {
                tri_a: a.nodes[na].tri_idx,
                tri_b: b.nodes[nb].tri_idx,
            });
        }
        (false, true) => {
            mesh_broad_phase_recursive(a, b, a.nodes[na].left, nb, pairs);
            mesh_broad_phase_recursive(a, b, a.nodes[na].right, nb, pairs);
        }
        (true, false) => {
            mesh_broad_phase_recursive(a, b, na, b.nodes[nb].left, pairs);
            mesh_broad_phase_recursive(a, b, na, b.nodes[nb].right, pairs);
        }
        (false, false) => {
            // Descend into the larger node
            let sa = a.nodes[na].aabb.surface_area();
            let sb = b.nodes[nb].aabb.surface_area();
            if sa >= sb {
                mesh_broad_phase_recursive(a, b, a.nodes[na].left, nb, pairs);
                mesh_broad_phase_recursive(a, b, a.nodes[na].right, nb, pairs);
            } else {
                mesh_broad_phase_recursive(a, b, na, b.nodes[nb].left, pairs);
                mesh_broad_phase_recursive(a, b, na, b.nodes[nb].right, pairs);
            }
        }
    }
}

/// Narrow phase: run triangle-triangle tests on candidate pairs.
pub fn mesh_narrow_phase(a: &TriMeshBvh, b: &TriMeshBvh, pairs: &[TriPair]) -> Vec<TriPair> {
    pairs
        .iter()
        .filter(|p| triangle_triangle_intersect(&a.triangles[p.tri_a], &b.triangles[p.tri_b]))
        .cloned()
        .collect()
}

/// Full mesh-mesh collision: broad + narrow phase.
pub fn mesh_mesh_collision(a: &TriMeshBvh, b: &TriMeshBvh) -> Vec<TriPair> {
    let candidates = mesh_broad_phase(a, b);
    mesh_narrow_phase(a, b, &candidates)
}

// ─────────────────────────────────────────────────────────────────────────────
// Cloth self-collision
// ─────────────────────────────────────────────────────────────────────────────

/// A cloth vertex.
#[derive(Clone, Debug)]
pub struct ClothVertex {
    /// Current position.
    pub pos: [f64; 3],
    /// Previous position (for Verlet).
    pub prev_pos: [f64; 3],
    /// Velocity.
    pub vel: [f64; 3],
    /// Mass.
    pub mass: f64,
    /// Is this vertex fixed (pinned)?
    pub pinned: bool,
}

impl ClothVertex {
    /// Create a new [`ClothVertex`].
    pub fn new(pos: [f64; 3], mass: f64) -> Self {
        Self {
            pos,
            prev_pos: pos,
            vel: [0.0; 3],
            mass,
            pinned: false,
        }
    }
}

/// A cloth edge constraint.
#[derive(Clone, Debug)]
pub struct ClothEdge {
    /// Index of first vertex.
    pub i: usize,
    /// Index of second vertex.
    pub j: usize,
    /// Rest length.
    pub rest_length: f64,
    /// Stiffness ∈ \[0, 1\].
    pub stiffness: f64,
}

impl ClothEdge {
    /// Create a new [`ClothEdge`].
    pub fn new(i: usize, j: usize, rest_length: f64, stiffness: f64) -> Self {
        Self {
            i,
            j,
            rest_length,
            stiffness,
        }
    }

    /// Project constraint (PBD-style).
    pub fn project(&self, verts: &mut [ClothVertex]) {
        if verts[self.i].pinned && verts[self.j].pinned {
            return;
        }
        let pi = verts[self.i].pos;
        let pj = verts[self.j].pos;
        let diff = vec3_sub(pi, pj);
        let l = vec3_len(diff);
        if l < 1e-15 {
            return;
        }
        let correction = self.stiffness * (l - self.rest_length) / l;
        let delta = vec3_scale(diff, correction);
        let wi = if verts[self.i].pinned {
            0.0
        } else {
            1.0 / verts[self.i].mass
        };
        let wj = if verts[self.j].pinned {
            0.0
        } else {
            1.0 / verts[self.j].mass
        };
        let w_sum = wi + wj;
        if w_sum < 1e-30 {
            return;
        }
        if !verts[self.i].pinned {
            let c = vec3_scale(delta, -wi / w_sum);
            verts[self.i].pos = vec3_add(verts[self.i].pos, c);
        }
        if !verts[self.j].pinned {
            let c = vec3_scale(delta, wj / w_sum);
            verts[self.j].pos = vec3_add(verts[self.j].pos, c);
        }
    }
}

/// Cloth self-collision response: push overlapping vertices apart.
///
/// Applies a position correction when two vertices that do not share a triangle
/// come closer than `thickness`: they are moved apart along the line joining
/// them by half the penetration each (a pinned vertex stays where it is).
/// Pairs are examined in ascending `(i, j)` order and every correction is seen
/// by the pairs examined after it, so the outcome is identical to the
/// brute-force double loop over all vertex pairs.
///
/// Returns the number of corrected pairs. A `thickness` that is not a finite
/// number above `1e-15` produces no collisions. Triangle corners that are not
/// valid vertex indices are ignored.
///
/// # Complexity
///
/// The vertices live in a uniform grid whose cells are at least `thickness`
/// wide, so a vertex is only compared with the vertices of the 27 cells around
/// it, and the triangle adjacency is precomputed once as sorted neighbour
/// lists. The cost is `O(V + T + k)` for `V` vertices, `T` triangles and `k`
/// candidate pairs, instead of the `O(V^2 T)` of scanning every triangle for
/// every vertex pair.
pub fn cloth_self_collision(
    vertices: &mut [ClothVertex],
    triangles: &[[usize; 3]],
    thickness: f64,
) -> usize {
    let n = vertices.len();
    if n < 2 || !thickness.is_finite() || thickness <= 1e-15 {
        return 0;
    }
    let adjacency = cloth_grid::vertex_adjacency(n, triangles);
    let mut grid = cloth_grid::VertexGrid::new(thickness);
    for (index, vertex) in vertices.iter().enumerate() {
        grid.insert(index, vertex.pos);
    }

    let mut n_collisions = 0;
    let mut partners: Vec<usize> = Vec::new();
    for i in 0..n {
        // The partners of `i` are examined in ascending index order, exactly like
        // the inner loop of the brute-force scan. The candidate list only has to
        // be rebuilt when a correction moves `i` into another grid cell: the
        // other vertices that move are partners that were already examined, so
        // they cannot enter the neighbourhood as new candidates.
        let mut cell = grid.cell_of(vertices[i].pos);
        grid.partners_after(vertices[i].pos, i, &adjacency[i], &mut partners);
        let mut cursor = 0;
        while cursor < partners.len() {
            let j = partners[cursor];
            cursor += 1;
            let diff = vec3_sub(vertices[i].pos, vertices[j].pos);
            let dist = vec3_len(diff);
            let colliding = dist < thickness && dist > 1e-15;
            if !colliding {
                continue;
            }
            let push = vec3_scale(vec3_norm(diff), (thickness - dist) * 0.5);
            let (old_i, old_j) = (vertices[i].pos, vertices[j].pos);
            if !vertices[i].pinned {
                vertices[i].pos = vec3_add(vertices[i].pos, push);
            }
            if !vertices[j].pinned {
                vertices[j].pos = vec3_sub(vertices[j].pos, push);
            }
            grid.relocate(i, old_i, vertices[i].pos);
            grid.relocate(j, old_j, vertices[j].pos);
            n_collisions += 1;

            let moved_cell = grid.cell_of(vertices[i].pos);
            if moved_cell != cell {
                cell = moved_cell;
                grid.partners_after(vertices[i].pos, j, &adjacency[i], &mut partners);
                cursor = 0;
            }
        }
    }
    n_collisions
}

// ─────────────────────────────────────────────────────────────────────────────
// Continuous collision detection (CCD) for deformables
// ─────────────────────────────────────────────────────────────────────────────

/// CCD result for a vertex-triangle collision.
#[derive(Clone, Debug)]
pub struct CcdHit {
    /// Time of impact ∈ \[0, 1\].
    pub toi: f64,
    /// Impact normal.
    pub normal: [f64; 3],
    /// Barycentric u coordinate on triangle.
    pub u: f64,
    /// Barycentric v coordinate on triangle.
    pub v: f64,
}

/// Relative width of the contact shell used by the analytic time-of-impact
/// solver, as a fraction of the size of the geometry involved.
const CCD_SHELL_REL: f64 = 1e-9;
/// Absolute floor of the shell, as a fraction of the largest coordinate: it keeps
/// the shell above the rounding of positions far from the origin.
const CCD_SHELL_FLOOR: f64 = 1e-14;

/// Contact shell for a continuous query over `points` (start and end positions of
/// everything involved).
///
/// The analytic solver validates a coplanarity root by the distance of the
/// primitives at that time, which is zero only up to rounding, so a shell of
/// exactly zero would reject true roots.
fn ccd_shell(points: &[[f64; 3]]) -> f64 {
    let lo: [f64; 3] =
        std::array::from_fn(|k| points.iter().map(|p| p[k]).fold(f64::INFINITY, f64::min));
    let hi: [f64; 3] = std::array::from_fn(|k| {
        points
            .iter()
            .map(|p| p[k])
            .fold(f64::NEG_INFINITY, f64::max)
    });
    let diameter = (0..3).map(|k| hi[k] - lo[k]).fold(0.0_f64, f64::max);
    let magnitude = lo
        .iter()
        .chain(hi.iter())
        .fold(0.0_f64, |m, c| m.max(c.abs()));
    CCD_SHELL_REL * diameter + CCD_SHELL_FLOOR * magnitude
}

/// Continuous collision detection between a moving vertex and a moving triangle.
///
/// Every point moves linearly from its start to its end position over the time
/// interval \[0, 1\]. Returns `Some(CcdHit)` for the earliest time at which the
/// vertex touches the triangle, `None` if it never does.
///
/// The time of impact is the earliest root of the coplanarity cubic of the four
/// points whose projection lies inside the triangle, found analytically (see
/// [`crate::deformable_collision::ccd_vertex_face_analytic`]); it is exact, not
/// quantised, and a crossing is found wherever it falls in the interval. A vertex
/// that already touches the triangle at the start reports time `0`.
///
/// `CcdHit::normal` is the unit geometric normal `(q - p) x (r - p)` of the
/// triangle at the time of impact (it is not flipped towards the vertex), and
/// `u`, `v` are the barycentric weights of the second and third triangle vertices
/// at the contact point.
pub fn vertex_triangle_ccd(
    vp0: [f64; 3],
    vp1: [f64; 3], // vertex start/end
    tp0: [f64; 3],
    tp1: [f64; 3], // tri v0 start/end
    tq0: [f64; 3],
    tq1: [f64; 3], // tri v1 start/end
    tr0: [f64; 3],
    tr1: [f64; 3], // tri v2 start/end
) -> Option<CcdHit> {
    let points = [vp0, vp1, tp0, tp1, tq0, tq1, tr0, tr1];
    if !points.iter().flatten().all(|c| c.is_finite()) {
        return None;
    }
    let toi = ccd_vertex_face_analytic(vp0, vp1, tp0, tp1, tq0, tq1, tr0, tr1, ccd_shell(&points))?;
    let vertex = vec3_lerp(vp0, vp1, toi);
    let tri = Triangle::new(
        vec3_lerp(tp0, tp1, toi),
        vec3_lerp(tq0, tq1, toi),
        vec3_lerp(tr0, tr1, toi),
    );
    let bary = point_triangle_barycentric(vertex, &tri);
    Some(CcdHit {
        toi,
        normal: tri.normal(),
        u: bary[0],
        v: bary[1],
    })
}

/// Compute barycentric coordinates of point p projected onto triangle.
pub fn point_triangle_barycentric(p: [f64; 3], tri: &Triangle) -> [f64; 3] {
    let v0 = vec3_sub(tri.v1, tri.v0);
    let v1 = vec3_sub(tri.v2, tri.v0);
    let v2 = vec3_sub(p, tri.v0);
    let d00 = vec3_dot(v0, v0);
    let d01 = vec3_dot(v0, v1);
    let d11 = vec3_dot(v1, v1);
    let d20 = vec3_dot(v2, v0);
    let d21 = vec3_dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-30 {
        return [1.0 / 3.0; 3];
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    [v, w, u]
}

// ─────────────────────────────────────────────────────────────────────────────
// Proximity query: signed distance field
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the squared distance from point `p` to the triangle `tri`.
pub fn point_triangle_dist_sq(p: [f64; 3], tri: &Triangle) -> f64 {
    let ab = vec3_sub(tri.v1, tri.v0);
    let ac = vec3_sub(tri.v2, tri.v0);
    let ap = vec3_sub(p, tri.v0);

    let d1 = vec3_dot(ab, ap);
    let d2 = vec3_dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return vec3_len_sq(ap);
    }

    let bp = vec3_sub(p, tri.v1);
    let d3 = vec3_dot(ab, bp);
    let d4 = vec3_dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return vec3_len_sq(bp);
    }

    let cp = vec3_sub(p, tri.v2);
    let d5 = vec3_dot(ab, cp);
    let d6 = vec3_dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return vec3_len_sq(cp);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let proj = vec3_add(tri.v0, vec3_scale(ab, v));
        return vec3_len_sq(vec3_sub(p, proj));
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let proj = vec3_add(tri.v0, vec3_scale(ac, w));
        return vec3_len_sq(vec3_sub(p, proj));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let proj = vec3_add(tri.v1, vec3_scale(vec3_sub(tri.v2, tri.v1), w));
        return vec3_len_sq(vec3_sub(p, proj));
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let proj = vec3_add(tri.v0, vec3_add(vec3_scale(ab, v), vec3_scale(ac, w)));
    vec3_len_sq(vec3_sub(p, proj))
}

/// Compute the closest point on a triangle to point p.
pub fn closest_point_on_triangle(p: [f64; 3], tri: &Triangle) -> [f64; 3] {
    let bary = point_triangle_barycentric(p, tri);
    let u = bary[0].clamp(0.0, 1.0);
    let v = bary[1].clamp(0.0, 1.0);
    let w_total = u + v;
    let (u, v) = if w_total > 1.0 {
        (u / w_total, v / w_total)
    } else {
        (u, v)
    };
    tri.barycentric_point(u, v)
}

/// Build a proximity distance field on a grid relative to a triangle mesh.
pub fn build_proximity_field(
    mesh: &[Triangle],
    origin: [f64; 3],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
) -> Vec<f64> {
    let total = nx * ny * nz;
    let mut field = vec![f64::INFINITY; total];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let p = [
                    origin[0] + ix as f64 * dx,
                    origin[1] + iy as f64 * dx,
                    origin[2] + iz as f64 * dx,
                ];
                let mut min_d2 = f64::INFINITY;
                for tri in mesh {
                    let d2 = point_triangle_dist_sq(p, tri);
                    if d2 < min_d2 {
                        min_d2 = d2;
                    }
                }
                field[ix * ny * nz + iy * nz + iz] = min_d2.sqrt();
            }
        }
    }
    field
}

// ─────────────────────────────────────────────────────────────────────────────
// Swept triangle vs static mesh
// ─────────────────────────────────────────────────────────────────────────────

/// Earliest time in \[0, 1\] at which the triangle translating from `start` to
/// `end` touches the static triangle `fixed`, or `None`.
///
/// Two triangles first touch either vertex-first (a vertex of one meets the face
/// of the other) or edge-first (an edge of one crosses an edge of the other), so
/// all three kinds of primitive pair are swept: the three moving vertices against
/// the static face, the three static vertices against the moving face and the
/// nine edge pairs. A pair that already intersects at the start touches at `0`.
fn sweep_pair_toi(
    start: &[[f64; 3]; 3],
    end: &[[f64; 3]; 3],
    fixed: &[[f64; 3]; 3],
) -> Option<f64> {
    if tri_tri::triangles_intersect(start, fixed) {
        return Some(0.0);
    }
    let shell = ccd_shell(&[
        start[0], start[1], start[2], end[0], end[1], end[2], fixed[0], fixed[1], fixed[2],
    ]);

    let mut earliest: Option<f64> = None;
    let mut consider = |toi: Option<f64>| {
        if let Some(t) = toi {
            earliest = Some(earliest.map_or(t, |best| best.min(t)));
        }
    };
    // Moving vertices against the static face.
    for (&s, &e) in start.iter().zip(end.iter()) {
        consider(ccd_vertex_face_analytic(
            s, e, fixed[0], fixed[0], fixed[1], fixed[1], fixed[2], fixed[2], shell,
        ));
    }
    // Static vertices against the moving face.
    for &p in fixed {
        consider(ccd_vertex_face_analytic(
            p, p, start[0], end[0], start[1], end[1], start[2], end[2], shell,
        ));
    }
    // Moving edges against static edges.
    const EDGES: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];
    for &(i, j) in &EDGES {
        for &(k, l) in &EDGES {
            consider(ccd_edge_edge_analytic(
                start[i], end[i], start[j], end[j], fixed[k], fixed[k], fixed[l], fixed[l], shell,
            ));
        }
    }
    earliest
}

/// Test a swept triangle (linear motion by displacement `d`) against a static mesh BVH.
///
/// The triangle translates rigidly from its position to `moving_tri + displacement`;
/// the result is the earliest time of impact ∈ \[0, 1\] (a fraction of
/// `displacement`) with any triangle of the mesh, `0` if it already intersects one,
/// and `None` if it never touches the mesh or the input is not finite.
///
/// Each candidate triangle (from the swept bounding box) is tested for contact
/// vertex-first in both directions and edge-first, see [`vertex_triangle_ccd`];
/// the minimum over all primitive pairs and candidates is returned.
pub fn swept_triangle_vs_mesh(
    moving_tri: &Triangle,
    displacement: [f64; 3],
    static_mesh: &TriMeshBvh,
) -> Option<f64> {
    let start = [moving_tri.v0, moving_tri.v1, moving_tri.v2];
    if !start
        .iter()
        .flatten()
        .chain(displacement.iter())
        .all(|c| c.is_finite())
    {
        return None;
    }
    let end = start.map(|p| vec3_add(p, displacement));

    // Broad phase: the box swept by the triangle, grown by the contact shell.
    let mut swept_aabb = moving_tri.aabb();
    swept_aabb.expand_aabb(&MeshAabb::from_triangle(end[0], end[1], end[2]));
    let swept_aabb = swept_aabb.fattened(ccd_shell(&[
        start[0], start[1], start[2], end[0], end[1], end[2],
    ]));

    let mut earliest: Option<f64> = None;
    for tri_idx in static_mesh.query_aabb(&swept_aabb) {
        let Some(static_tri) = static_mesh.triangles.get(tri_idx) else {
            continue;
        };
        let fixed = [static_tri.v0, static_tri.v1, static_tri.v2];
        if let Some(toi) = sweep_pair_toi(&start, &end, &fixed) {
            if toi <= 0.0 {
                return Some(0.0);
            }
            earliest = Some(earliest.map_or(toi, |best| best.min(toi)));
        }
    }
    earliest
}

// ─────────────────────────────────────────────────────────────────────────────
// Penetration depth estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the penetration depth between two overlapping meshes using
/// a sampling approach on the contact region.
pub fn penetration_depth_estimate(a: &TriMeshBvh, b: &TriMeshBvh, n_samples: usize) -> f64 {
    let pairs = mesh_mesh_collision(a, b);
    if pairs.is_empty() {
        return 0.0;
    }

    let mut max_depth = 0.0_f64;
    for pair in &pairs {
        let tri_a = &a.triangles[pair.tri_a];
        let tri_b = &b.triangles[pair.tri_b];
        // Sample points on tri_a and measure distance to tri_b's plane
        for k in 0..n_samples {
            let t = k as f64 / (n_samples as f64 - 1.0).max(1.0);
            let u = t * 0.5;
            let v = (1.0 - t) * 0.5;
            let p = tri_a.barycentric_point(u, v);
            let d2 = point_triangle_dist_sq(p, tri_b);
            if d2 > max_depth {
                max_depth = d2;
            }
        }
    }
    max_depth.sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Contact manifold generation for meshes
// ─────────────────────────────────────────────────────────────────────────────

/// A contact point for mesh collisions.
#[derive(Clone, Debug)]
pub struct MeshContact {
    /// Contact position in world space.
    pub position: [f64; 3],
    /// Contact normal (pointing from B to A).
    pub normal: [f64; 3],
    /// Penetration depth.
    pub depth: f64,
    /// Triangle index in mesh A.
    pub tri_a: usize,
    /// Triangle index in mesh B.
    pub tri_b: usize,
}

/// A contact manifold for two colliding meshes.
#[derive(Clone, Debug, Default)]
pub struct MeshContactManifold {
    /// Contact points.
    pub contacts: Vec<MeshContact>,
    /// Overall contact normal (average).
    pub avg_normal: [f64; 3],
    /// Maximum penetration depth.
    pub max_depth: f64,
}

impl MeshContactManifold {
    /// Create an empty manifold.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a contact point.
    pub fn add_contact(&mut self, c: MeshContact) {
        self.max_depth = self.max_depth.max(c.depth);
        self.contacts.push(c);
        self.update_avg_normal();
    }

    fn update_avg_normal(&mut self) {
        if self.contacts.is_empty() {
            return;
        }
        let mut n = [0.0; 3];
        for c in &self.contacts {
            n = vec3_add(n, c.normal);
        }
        self.avg_normal = vec3_norm(n);
    }

    /// Number of contact points.
    pub fn num_contacts(&self) -> usize {
        self.contacts.len()
    }
}

/// Generate a contact manifold for two colliding mesh BVHs.
pub fn generate_mesh_contact_manifold(a: &TriMeshBvh, b: &TriMeshBvh) -> MeshContactManifold {
    let pairs = mesh_mesh_collision(a, b);
    let mut manifold = MeshContactManifold::new();

    for pair in &pairs {
        let tri_a = &a.triangles[pair.tri_a];
        let tri_b = &b.triangles[pair.tri_b];

        let na = tri_a.normal();
        let nb = tri_b.normal();

        // Use the average normal
        let normal = vec3_norm(vec3_sub(na, nb));

        // Contact position: midpoint of centroids
        let ca = tri_a.centroid();
        let cb = tri_b.centroid();
        let position = vec3_scale(vec3_add(ca, cb), 0.5);

        // Depth: distance from centroid of a to plane of b
        let n2 = tri_b.normal();
        let d = vec3_dot(n2, vec3_sub(ca, tri_b.v0)).abs();

        manifold.add_contact(MeshContact {
            position,
            normal,
            depth: d,
            tri_a: pair.tri_a,
            tri_b: pair.tri_b,
        });
    }
    manifold
}

// ─────────────────────────────────────────────────────────────────────────────
// Deformable mesh (BVH + vertex array)
// ─────────────────────────────────────────────────────────────────────────────

/// A deformable triangle mesh backed by a dynamic vertex array and a refittable BVH.
pub struct DeformableMesh {
    /// Vertex positions.
    pub vertices: Vec<[f64; 3]>,
    /// Triangle index triples.
    pub face_indices: Vec<[usize; 3]>,
    /// The BVH (refitted each step).
    pub bvh: TriMeshBvh,
}

impl DeformableMesh {
    /// Build from vertices and face indices.
    pub fn build(vertices: Vec<[f64; 3]>, face_indices: Vec<[usize; 3]>) -> Self {
        let triangles: Vec<Triangle> = face_indices
            .iter()
            .map(|&[a, b, c]| Triangle::new(vertices[a], vertices[b], vertices[c]))
            .collect();
        let bvh = TriMeshBvh::build(triangles);
        Self {
            vertices,
            face_indices,
            bvh,
        }
    }

    /// Update vertex position and refit BVH.
    pub fn update_vertex(&mut self, idx: usize, new_pos: [f64; 3]) {
        self.vertices[idx] = new_pos;
        // Sync triangles
        for (fi, &[a, b, c]) in self.face_indices.iter().enumerate() {
            if a == idx || b == idx || c == idx {
                self.bvh.triangles[fi] =
                    Triangle::new(self.vertices[a], self.vertices[b], self.vertices[c]);
            }
        }
        self.bvh.refit();
    }

    /// Apply a displacement to all vertices and refit.
    pub fn apply_displacement(&mut self, displacements: &[[f64; 3]]) {
        for (i, d) in displacements.iter().enumerate() {
            if i < self.vertices.len() {
                self.vertices[i] = vec3_add(self.vertices[i], *d);
            }
        }
        for (fi, &[a, b, c]) in self.face_indices.iter().enumerate() {
            self.bvh.triangles[fi] =
                Triangle::new(self.vertices[a], self.vertices[b], self.vertices[c]);
        }
        self.bvh.refit();
    }

    /// Number of faces.
    pub fn num_faces(&self) -> usize {
        self.face_indices.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Signed Distance Field (SDF) helpers
// ─────────────────────────────────────────────────────────────────────────────

/// A simple analytic SDF for a sphere.
pub fn sdf_sphere(p: [f64; 3], centre: [f64; 3], radius: f64) -> f64 {
    vec3_len(vec3_sub(p, centre)) - radius
}

/// A simple analytic SDF for an AABB.
pub fn sdf_aabb(p: [f64; 3], aabb: &MeshAabb) -> f64 {
    let c = aabb.centre();
    let h = aabb.half_extents();
    let d = [
        (p[0] - c[0]).abs() - h[0],
        (p[1] - c[1]).abs() - h[1],
        (p[2] - c[2]).abs() - h[2],
    ];
    let outside = [d[0].max(0.0), d[1].max(0.0), d[2].max(0.0)];
    let inside = d[0].max(d[1]).max(d[2]).min(0.0);
    vec3_len(outside) + inside
}

/// Sample the SDF of a triangle mesh on a uniform grid.
///
/// Uses the proximity field with sign estimated from face normals.
pub fn mesh_sdf_grid(
    mesh: &[Triangle],
    origin: [f64; 3],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
) -> Vec<f64> {
    let total = nx * ny * nz;
    let mut field = vec![0.0_f64; total];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let p = [
                    origin[0] + ix as f64 * dx,
                    origin[1] + iy as f64 * dx,
                    origin[2] + iz as f64 * dx,
                ];
                let mut min_d2 = f64::INFINITY;
                let mut closest_n = [0.0; 3];
                let mut closest_diff = [0.0; 3];
                for tri in mesh {
                    let d2 = point_triangle_dist_sq(p, tri);
                    if d2 < min_d2 {
                        min_d2 = d2;
                        closest_n = tri.normal();
                        closest_diff = vec3_sub(p, tri.centroid());
                    }
                }
                let sign = if vec3_dot(closest_diff, closest_n) >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                field[ix * ny * nz + iy * nz + iz] = sign * min_d2.sqrt();
            }
        }
    }
    field
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Vector math ───────────────────────────────────────────────────────────

    #[test]
    fn test_vec3_dot_orthogonal() {
        assert!((vec3_dot([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])).abs() < 1e-15);
    }

    #[test]
    fn test_vec3_cross_unit() {
        let c = vec3_cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!((c[2] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_vec3_norm_unit_length() {
        let n = vec3_norm([3.0, 4.0, 0.0]);
        assert!((vec3_len(n) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_vec3_lerp_midpoint() {
        let m = vec3_lerp([0.0; 3], [2.0, 2.0, 2.0], 0.5);
        for &c in &m {
            assert!((c - 1.0).abs() < 1e-14);
        }
    }

    // ── AABB ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_aabb_overlaps() {
        let a = MeshAabb::new([0.0; 3], [1.0; 3]);
        let b = MeshAabb::new([0.5; 3], [1.5; 3]);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_aabb_not_overlaps() {
        let a = MeshAabb::new([0.0; 3], [1.0; 3]);
        let b = MeshAabb::new([2.0; 3], [3.0; 3]);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_aabb_contains_point() {
        let a = MeshAabb::new([0.0; 3], [1.0; 3]);
        assert!(a.contains_point([0.5, 0.5, 0.5]));
        assert!(!a.contains_point([1.5, 0.5, 0.5]));
    }

    #[test]
    fn test_aabb_from_triangle() {
        let a = MeshAabb::from_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(a.min[2].abs() < 1e-15);
        assert!((a.max[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_aabb_fattened() {
        let a = MeshAabb::new([0.0; 3], [1.0; 3]);
        let fat = a.fattened(0.1);
        for i in 0..3 {
            assert!((fat.min[i] - (-0.1)).abs() < 1e-14);
            assert!((fat.max[i] - 1.1).abs() < 1e-14);
        }
    }

    // ── Triangle ──────────────────────────────────────────────────────────────

    #[test]
    fn test_triangle_normal_unit() {
        let tri = Triangle::new([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let n = tri.normal();
        assert!((vec3_len(n) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_triangle_area() {
        let tri = Triangle::new([0.0; 3], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]);
        assert!((tri.area() - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_triangle_centroid() {
        let tri = Triangle::new([0.0; 3], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]);
        let c = tri.centroid();
        assert!((c[0] - 1.0).abs() < 1e-14);
        assert!((c[1] - 1.0).abs() < 1e-14);
    }

    // ── Möller-Trumbore ───────────────────────────────────────────────────────

    #[test]
    fn test_ray_triangle_hit() {
        let tri = Triangle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let hit = ray_triangle_mt([0.25, 0.25, 1.0], [0.0, 0.0, -1.0], &tri, 0.0, 10.0);
        assert!(hit.is_some(), "Should hit the triangle");
        let h = hit.unwrap();
        assert!((h.t - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ray_triangle_miss() {
        let tri = Triangle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let hit = ray_triangle_mt([5.0, 5.0, 1.0], [0.0, 0.0, -1.0], &tri, 0.0, 10.0);
        assert!(hit.is_none());
    }

    #[test]
    fn test_ray_triangle_back_face_culled_by_t() {
        // Ray in same direction, t should be negative → should miss
        let tri = Triangle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let hit = ray_triangle_mt([0.25, 0.25, -1.0], [0.0, 0.0, -1.0], &tri, 0.0, 10.0);
        assert!(hit.is_none(), "Ray moving away from triangle");
    }

    // ── Ray-AABB ──────────────────────────────────────────────────────────────

    #[test]
    fn test_ray_aabb_hit() {
        let aabb = MeshAabb::new([0.0; 3], [1.0; 3]);
        assert!(ray_aabb_intersect(
            [0.5, 0.5, 2.0],
            [0.0, 0.0, -1.0],
            &aabb,
            0.0,
            10.0
        ));
    }

    #[test]
    fn test_ray_aabb_miss() {
        let aabb = MeshAabb::new([0.0; 3], [1.0; 3]);
        assert!(!ray_aabb_intersect(
            [5.0, 5.0, 2.0],
            [0.0, 0.0, -1.0],
            &aabb,
            0.0,
            10.0
        ));
    }

    // ── Triangle-Triangle ─────────────────────────────────────────────────────

    #[test]
    fn test_tri_tri_intersect_coplanar_overlapping() {
        let t1 = Triangle::new([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]);
        let t2 = Triangle::new([1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 2.0, 0.0]);
        // These triangles share edge/overlap region
        let _ = triangle_triangle_intersect(&t1, &t2); // should not panic
    }

    #[test]
    fn test_tri_tri_separated() {
        // t1 in xy-plane at z=0, t2 in xy-plane at z=2.0 (clearly separated by a gap)
        let t1 = Triangle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let t2 = Triangle::new([0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]);
        assert!(!triangle_triangle_intersect(&t1, &t2));
    }

    // ── BVH build/query ───────────────────────────────────────────────────────

    #[test]
    fn test_bvh_build_single_triangle() {
        let tris = vec![Triangle::new([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])];
        let bvh = TriMeshBvh::build(tris);
        assert_eq!(bvh.num_triangles(), 1);
        assert!(bvh.num_nodes() >= 1);
    }

    #[test]
    fn test_bvh_build_multiple_triangles() {
        let tris: Vec<Triangle> = (0..8)
            .map(|i| {
                let x = i as f64;
                Triangle::new([x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x, 1.0, 0.0])
            })
            .collect();
        let bvh = TriMeshBvh::build(tris);
        assert_eq!(bvh.num_triangles(), 8);
    }

    #[test]
    fn test_bvh_ray_cast_hits() {
        let tris = vec![Triangle::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        )];
        let bvh = TriMeshBvh::build(tris);
        let result = bvh.ray_cast([0.25, 0.25, 2.0], [0.0, 0.0, -1.0], 10.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_bvh_ray_cast_miss() {
        let tris = vec![Triangle::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        )];
        let bvh = TriMeshBvh::build(tris);
        let result = bvh.ray_cast([5.0, 5.0, 2.0], [0.0, 0.0, -1.0], 10.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_bvh_query_aabb() {
        let tris: Vec<Triangle> = (0..4)
            .map(|i| {
                let x = i as f64 * 3.0;
                Triangle::new([x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x, 1.0, 0.0])
            })
            .collect();
        let bvh = TriMeshBvh::build(tris);
        let q = MeshAabb::new([0.0; 3], [1.5, 1.5, 1.5]);
        let hits = bvh.query_aabb(&q);
        assert!(!hits.is_empty());
    }

    // ── Mesh-mesh collision ───────────────────────────────────────────────────

    #[test]
    fn test_mesh_broad_phase_overlapping() {
        let tris_a = vec![Triangle::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        )];
        let tris_b = vec![Triangle::new(
            [0.3, 0.3, -0.1],
            [0.6, 0.3, 0.0],
            [0.3, 0.6, 0.0],
        )];
        let bvh_a = TriMeshBvh::build(tris_a);
        let bvh_b = TriMeshBvh::build(tris_b);
        let pairs = mesh_broad_phase(&bvh_a, &bvh_b);
        assert!(!pairs.is_empty());
    }

    #[test]
    fn test_mesh_broad_phase_separated() {
        let tris_a = vec![Triangle::new([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])];
        let tris_b = vec![Triangle::new(
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
        )];
        let bvh_a = TriMeshBvh::build(tris_a);
        let bvh_b = TriMeshBvh::build(tris_b);
        let pairs = mesh_broad_phase(&bvh_a, &bvh_b);
        assert!(pairs.is_empty());
    }

    // ── Point-triangle distance ───────────────────────────────────────────────

    #[test]
    fn test_point_triangle_dist_vertex_case() {
        let tri = Triangle::new([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let d2 = point_triangle_dist_sq([-1.0, 0.0, 0.0], &tri);
        assert!((d2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_triangle_dist_above() {
        let tri = Triangle::new([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let d2 = point_triangle_dist_sq([0.25, 0.25, 2.0], &tri);
        assert!((d2 - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_closest_point_on_triangle_inside() {
        let tri = Triangle::new([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let cp = closest_point_on_triangle([0.25, 0.25, 1.0], &tri);
        // Should project onto the triangle plane
        assert!(
            (cp[2]).abs() < 0.1,
            "Closest point z should be near 0, got {}",
            cp[2]
        );
    }

    // ── Cloth self-collision ──────────────────────────────────────────────────

    #[test]
    fn test_cloth_self_collision_separates_vertices() {
        let mut verts = vec![
            ClothVertex::new([0.0, 0.0, 0.0], 1.0),
            ClothVertex::new([0.001, 0.0, 0.0], 1.0),
        ];
        let tris: Vec<[usize; 3]> = vec![];
        let n = cloth_self_collision(&mut verts, &tris, 0.05);
        assert!(n > 0);
        let dist = vec3_len(vec3_sub(verts[0].pos, verts[1].pos));
        assert!(dist >= 0.049, "Vertices should be separated, dist={}", dist);
    }

    #[test]
    fn test_cloth_edge_constraint_projection() {
        let mut verts = vec![
            ClothVertex::new([0.0, 0.0, 0.0], 1.0),
            ClothVertex::new([2.0, 0.0, 0.0], 1.0),
        ];
        let edge = ClothEdge::new(0, 1, 1.0, 1.0);
        edge.project(&mut verts);
        let d = vec3_len(vec3_sub(verts[0].pos, verts[1].pos));
        assert!(
            (d - 1.0).abs() < 0.01,
            "After projection edge length should be ~1, got {}",
            d
        );
    }

    // ── Deformable mesh ───────────────────────────────────────────────────────

    #[test]
    fn test_deformable_mesh_build() {
        let verts = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0, 1, 2]];
        let dm = DeformableMesh::build(verts, faces);
        assert_eq!(dm.num_faces(), 1);
    }

    #[test]
    fn test_deformable_mesh_update_vertex() {
        let verts = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0, 1, 2]];
        let mut dm = DeformableMesh::build(verts, faces);
        dm.update_vertex(0, [0.5, 0.5, 0.0]);
        assert!((dm.vertices[0][0] - 0.5).abs() < 1e-14);
    }

    // ── SDF helpers ───────────────────────────────────────────────────────────

    #[test]
    fn test_sdf_sphere_outside() {
        let d = sdf_sphere([2.0, 0.0, 0.0], [0.0; 3], 1.0);
        assert!((d - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_sdf_sphere_inside() {
        let d = sdf_sphere([0.5, 0.0, 0.0], [0.0; 3], 1.0);
        assert!(d < 0.0, "Point inside sphere → negative SDF");
    }

    #[test]
    fn test_sdf_aabb_outside() {
        let aabb = MeshAabb::new([0.0; 3], [1.0; 3]);
        let d = sdf_aabb([2.0, 0.5, 0.5], &aabb);
        assert!((d - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_sdf_aabb_inside() {
        let aabb = MeshAabb::new([0.0; 3], [2.0; 3]);
        let d = sdf_aabb([1.0, 1.0, 1.0], &aabb);
        assert!(d < 0.0, "Centre of AABB → negative SDF");
    }

    // ── Contact manifold ──────────────────────────────────────────────────────

    #[test]
    fn test_contact_manifold_add_contact() {
        let mut m = MeshContactManifold::new();
        m.add_contact(MeshContact {
            position: [0.0; 3],
            normal: [0.0, 0.0, 1.0],
            depth: 0.05,
            tri_a: 0,
            tri_b: 0,
        });
        assert_eq!(m.num_contacts(), 1);
        assert!((m.max_depth - 0.05).abs() < 1e-14);
    }

    #[test]
    fn test_generate_manifold_no_collision() {
        let tris_a = vec![Triangle::new([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])];
        let tris_b = vec![Triangle::new(
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
        )];
        let bvh_a = TriMeshBvh::build(tris_a);
        let bvh_b = TriMeshBvh::build(tris_b);
        let manifold = generate_mesh_contact_manifold(&bvh_a, &bvh_b);
        assert_eq!(manifold.num_contacts(), 0);
    }

    // ── Penetration depth ─────────────────────────────────────────────────────

    #[test]
    fn test_penetration_depth_no_overlap() {
        let tris_a = vec![Triangle::new([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])];
        let tris_b = vec![Triangle::new(
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
        )];
        let a = TriMeshBvh::build(tris_a);
        let b = TriMeshBvh::build(tris_b);
        let d = penetration_depth_estimate(&a, &b, 4);
        assert!((d - 0.0).abs() < 1e-14);
    }

    // ── BVH refit ─────────────────────────────────────────────────────────────

    #[test]
    fn test_bvh_refit_after_vertex_move() {
        let verts = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0, 1, 2]];
        let mut dm = DeformableMesh::build(verts, faces);
        dm.update_vertex(1, [5.0, 0.0, 0.0]);
        let root_aabb = &dm.bvh.nodes[dm.bvh.root].aabb;
        assert!(root_aabb.max[0] >= 4.9, "BVH should cover moved vertex");
    }

    // ── Proximity field ───────────────────────────────────────────────────────

    #[test]
    fn test_proximity_field_near_triangle() {
        let tris = vec![Triangle::new([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])];
        let field = build_proximity_field(&tris, [0.0, 0.0, 0.0], 3, 3, 3, 0.5);
        // Point (0,0,0) is on the triangle → dist ≈ 0
        assert!(field[0] < 1e-10, "Point on triangle → zero distance");
    }

    // ── Barycentric coords ────────────────────────────────────────────────────

    #[test]
    fn test_barycentric_centroid() {
        let tri = Triangle::new([0.0; 3], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]);
        let c = tri.centroid();
        let bary = point_triangle_barycentric(c, &tri);
        // All barycentric coords should be near 1/3
        assert!((bary[0] + bary[1] + bary[2] - 1.0).abs() < 0.1);
    }
}
