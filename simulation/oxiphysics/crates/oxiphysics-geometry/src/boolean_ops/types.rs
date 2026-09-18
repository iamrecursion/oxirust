//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
use super::operands::{
    from_simple_mesh, mesh_op, owned_by_first, require_closed_oriented, snap_operands, tidy_result,
    to_simple_mesh,
};
use crate::error::{Error, Result};
use crate::mesh_boolean::mesh_boolean;
use crate::triangle_mesh::TriangleMesh;

/// A node in a Binary Space Partitioning tree.
///
/// Each node divides space with a plane; polygons are classified as front,
/// back, or coplanar with the plane.
pub struct BspNode {
    /// The dividing plane.
    pub plane: BspPlane,
    /// Coplanar polygons stored at this node.
    pub polygons: Vec<Vec<[f64; 3]>>,
    /// Subtree in front of the plane.
    pub front: Option<Box<BspNode>>,
    /// Subtree behind the plane.
    pub back: Option<Box<BspNode>>,
}
impl BspNode {
    /// Create a new BSP node with the given plane, no polygons, no children.
    pub fn new(plane: BspPlane) -> Self {
        Self {
            plane,
            polygons: Vec::new(),
            front: None,
            back: None,
        }
    }
    /// Classify a point relative to this node's plane.
    pub fn classify_point(&self, p: [f64; 3]) -> PlaneClass {
        let d = self.plane.signed_dist(p);
        if d > 1e-8 {
            PlaneClass::Front
        } else if d < -1e-8 {
            PlaneClass::Back
        } else {
            PlaneClass::OnPlane
        }
    }
    /// Insert a polygon into the BSP tree.
    ///
    /// The polygon is split if it straddles the plane, and each fragment is
    /// routed to the front or back subtree.
    pub fn insert_polygon(&mut self, polygon: Vec<[f64; 3]>) {
        if polygon.is_empty() {
            return;
        }
        let classes: Vec<PlaneClass> = polygon.iter().map(|&p| self.classify_point(p)).collect();
        let all_front = classes
            .iter()
            .all(|&c| c == PlaneClass::Front || c == PlaneClass::OnPlane);
        let all_back = classes
            .iter()
            .all(|&c| c == PlaneClass::Back || c == PlaneClass::OnPlane);
        if all_front && all_back {
            self.polygons.push(polygon);
        } else if all_front {
            match &mut self.front {
                Some(f) => f.insert_polygon(polygon),
                None => {
                    let mut node = Box::new(BspNode::new(self.plane));
                    node.polygons.push(polygon);
                    self.front = Some(node);
                }
            }
        } else if all_back {
            match &mut self.back {
                Some(b) => b.insert_polygon(polygon),
                None => {
                    let mut node = Box::new(BspNode::new(self.plane));
                    node.polygons.push(polygon);
                    self.back = Some(node);
                }
            }
        } else {
            let (front_poly, back_poly) = split_polygon_by_plane(&polygon, &self.plane);
            if !front_poly.is_empty() {
                match &mut self.front {
                    Some(f) => f.insert_polygon(front_poly),
                    None => {
                        let mut node = Box::new(BspNode::new(self.plane));
                        node.polygons.push(front_poly);
                        self.front = Some(node);
                    }
                }
            }
            if !back_poly.is_empty() {
                match &mut self.back {
                    Some(b) => b.insert_polygon(back_poly),
                    None => {
                        let mut node = Box::new(BspNode::new(self.plane));
                        node.polygons.push(back_poly);
                        self.back = Some(node);
                    }
                }
            }
        }
    }
    /// Collect all polygons in front of the plane (recursively).
    pub fn collect_front_polygons(&self) -> Vec<Vec<[f64; 3]>> {
        let mut result = self.polygons.clone();
        if let Some(f) = &self.front {
            result.extend(f.collect_front_polygons());
        }
        result
    }
    /// Collect all polygons behind the plane (recursively).
    pub fn collect_back_polygons(&self) -> Vec<Vec<[f64; 3]>> {
        let mut result = self.polygons.clone();
        if let Some(b) = &self.back {
            result.extend(b.collect_back_polygons());
        }
        result
    }
    /// Count all polygons in the tree.
    pub fn count_polygons(&self) -> usize {
        let mut count = self.polygons.len();
        if let Some(f) = &self.front {
            count += f.count_polygons();
        }
        if let Some(b) = &self.back {
            count += b.count_polygons();
        }
        count
    }
}
/// Performs CSG boolean operations while tracking which material each
/// resulting triangle comes from.
///
/// The operation runs on the exact surface-arrangement pipeline of
/// [`mesh_boolean`], so the result is split along the intersection curves of
/// the operands and every result triangle is a piece of exactly one input
/// triangle; it carries the material of that triangle's operand.  A coincident
/// face of the two operands that survives the operation is the first operand's
/// copy and carries the first material.
pub struct CsgWithMaterials;
impl CsgWithMaterials {
    /// Execute a boolean operation, returning `(result_mesh, per_triangle_materials)`.
    ///
    /// `per_triangle_materials` is parallel to `result_mesh.indices`.
    ///
    /// # Errors
    ///
    /// The operand validation errors of [`MeshBoolean::execute`].
    pub fn execute(
        op: BooleanOp,
        mesh_a: &TriangleMesh,
        mat_a: CsgMaterial,
        mesh_b: &TriangleMesh,
        mat_b: CsgMaterial,
    ) -> Result<(TriangleMesh, Vec<CsgMaterial>)> {
        let a = to_simple_mesh("mesh_a", mesh_a)?;
        let b = to_simple_mesh("mesh_b", mesh_b)?;
        let result = mesh_boolean(&a, &b, mesh_op(op));
        let materials = owned_by_first(&result, &a)
            .into_iter()
            .map(|from_a| if from_a { mat_a.clone() } else { mat_b.clone() })
            .collect();
        Ok((from_simple_mesh(&result), materials))
    }
}
/// A manifold check result.
#[derive(Debug, Clone)]
pub struct ManifoldCheckResult {
    /// Whether the mesh is manifold (every edge shared by exactly 2 triangles).
    pub is_manifold: bool,
    /// Number of boundary edges (not shared by 2 triangles).
    pub n_boundary_edges: usize,
    /// Number of non-manifold edges (shared by >2 triangles).
    pub n_non_manifold_edges: usize,
}
/// Performs boolean operations between two closed triangle meshes.
///
/// This is a front-end for the exact surface-arrangement pipeline of
/// [`mesh_boolean`]: the two surfaces are
/// intersected, every triangle that is cut is re-triangulated along the
/// intersection curves (points on shared edges are propagated to both incident
/// triangles, so the seam stays conforming), and each resulting piece, which
/// can no longer straddle the other surface, is kept or dropped by an exact
/// inside/outside decision.  Pieces lying in a face of the other operand are
/// resolved by relative orientation, so shared faces neither duplicate nor
/// leave internal walls.
///
/// For closed, consistently oriented operands in general position, and for
/// exactly coincident faces, the result is a closed, consistently oriented
/// surface with shared vertices.  Near-degenerate configurations (faces that
/// are almost but not exactly coplanar) can leave a few non-manifold seam
/// edges; [`RobustMeshBoolean`] snaps such geometry with a tolerance first.
///
/// The operands must be closed and consistently oriented (outward normals);
/// [`ManifoldBoolean`] verifies this before operating.
pub struct MeshBoolean;
impl MeshBoolean {
    /// Execute a boolean operation between two meshes.
    ///
    /// # Arguments
    /// * `op` – the type of operation.
    /// * `mesh_a` – the first operand.
    /// * `mesh_b` – the second operand.
    ///
    /// # Returns
    /// A new [`TriangleMesh`] representing the result (empty when the result
    /// has no surface, e.g. the intersection of disjoint solids).
    ///
    /// # Errors
    ///
    /// * [`Error::TooFewPoints`] when an operand has no vertices.
    /// * [`Error::InvalidMesh`] when an operand has no triangles or a
    ///   non-finite vertex.
    /// * [`Error::IndexOutOfBounds`] when a triangle references a missing vertex.
    pub fn execute(
        op: BooleanOp,
        mesh_a: &TriangleMesh,
        mesh_b: &TriangleMesh,
    ) -> Result<TriangleMesh> {
        let a = to_simple_mesh("mesh_a", mesh_a)?;
        let b = to_simple_mesh("mesh_b", mesh_b)?;
        Ok(from_simple_mesh(&mesh_boolean(&a, &b, mesh_op(op))))
    }
}
/// A half-edge mesh structure for efficient topological queries.
pub struct HalfEdgeMesh {
    /// Vertex positions.
    pub vertices: Vec<[f64; 3]>,
    /// Half-edges.
    pub half_edges: Vec<HalfEdge>,
    /// For each face, the index of one of its half-edges.
    pub face_starts: Vec<usize>,
}
impl Default for HalfEdgeMesh {
    fn default() -> Self {
        Self::new()
    }
}
impl HalfEdgeMesh {
    /// Build a half-edge mesh from a triangle mesh given as vertices + face indices.
    pub fn from_triangle_mesh(verts: &[[f64; 3]], tris: &[[usize; 3]]) -> Self {
        let mut half_edges: Vec<HalfEdge> = Vec::new();
        let mut face_starts: Vec<usize> = Vec::with_capacity(tris.len());
        for (fi, tri) in tris.iter().enumerate() {
            let base = half_edges.len();
            face_starts.push(base);
            for k in 0..3 {
                half_edges.push(HalfEdge {
                    vertex: tri[(k + 1) % 3],
                    face: fi,
                    next: base + (k + 1) % 3,
                    twin: usize::MAX,
                });
            }
        }
        let mut edge_map: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        for (hi, he) in half_edges.iter().enumerate() {
            let base = face_starts[he.face];
            let local = hi - base;
            let prev_local = (local + 2) % 3;
            let from_v = tris[he.face][prev_local];
            let to_v = he.vertex;
            edge_map.entry((from_v, to_v)).or_insert(hi);
        }
        let edge_map_clone = edge_map.clone();
        for (&(from, to), &hi) in &edge_map_clone {
            if let Some(&twin_hi) = edge_map.get(&(to, from)) {
                half_edges[hi].twin = twin_hi;
                half_edges[twin_hi].twin = hi;
            }
        }
        HalfEdgeMesh {
            vertices: verts.to_vec(),
            half_edges,
            face_starts,
        }
    }
    /// Return the number of faces.
    pub fn n_faces(&self) -> usize {
        self.face_starts.len()
    }
    /// Return the number of vertices.
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }
    /// Return the three vertex indices for face `fi`.
    pub fn face_vertices(&self, fi: usize) -> [usize; 3] {
        let base = self.face_starts[fi];
        let he0 = &self.half_edges[base];
        let he1 = &self.half_edges[he0.next];
        let he2 = &self.half_edges[he1.next];
        [he2.vertex, he0.vertex, he1.vertex]
    }
    /// Compute the area of face `fi`.
    pub fn face_area(&self, fi: usize) -> f64 {
        let [v0, v1, v2] = self.face_vertices(fi);
        let a = self.vertices[v0];
        let b = self.vertices[v1];
        let c = self.vertices[v2];
        let n = cross3(sub3(b, a), sub3(c, a));
        len3(n) * 0.5
    }
    /// Return all boundary edges (half-edges with no twin).
    pub fn boundary_half_edges(&self) -> Vec<usize> {
        self.half_edges
            .iter()
            .enumerate()
            .filter(|(_, he)| he.twin == usize::MAX)
            .map(|(i, _)| i)
            .collect()
    }
    /// Check whether the mesh is manifold (every edge has exactly one twin).
    pub fn is_manifold(&self) -> bool {
        self.boundary_half_edges().is_empty()
    }
    /// Create an empty half-edge mesh.
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            half_edges: Vec::new(),
            face_starts: Vec::new(),
        }
    }
    /// Add a vertex to the mesh. Returns the vertex index.
    pub fn add_vertex(&mut self, pos: [f64; 3]) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(pos);
        idx
    }
    /// Add a triangle face (given vertex indices). Returns the face index.
    pub fn add_triangle(&mut self, v0: usize, v1: usize, v2: usize) -> usize {
        let fi = self.face_starts.len();
        let base = self.half_edges.len();
        self.face_starts.push(base);
        self.half_edges.push(HalfEdge {
            vertex: v1,
            face: fi,
            next: base + 1,
            twin: usize::MAX,
        });
        self.half_edges.push(HalfEdge {
            vertex: v2,
            face: fi,
            next: base + 2,
            twin: usize::MAX,
        });
        self.half_edges.push(HalfEdge {
            vertex: v0,
            face: fi,
            next: base,
            twin: usize::MAX,
        });
        fi
    }
    /// Rebuild twin half-edge links from scratch.
    pub fn build_twin_links(&mut self) {
        // Reset all twins
        for he in self.half_edges.iter_mut() {
            he.twin = usize::MAX;
        }
        let mut edge_map: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        // First pass: collect edge data immutably
        let edge_data: Vec<(usize, usize, usize)> = self
            .half_edges
            .iter()
            .enumerate()
            .map(|(hi, he)| {
                let fi = he.face;
                let base = self.face_starts[fi];
                let local = hi - base;
                let prev_local = (local + 2) % 3;
                let from_v = self.half_edges[base + prev_local].vertex;
                let to_v = he.vertex;
                (hi, from_v, to_v)
            })
            .collect();
        // Second pass: apply twin links mutably
        for (hi, from_v, to_v) in edge_data {
            if let Some(&twin_hi) = edge_map.get(&(to_v, from_v)) {
                self.half_edges[hi].twin = twin_hi;
                self.half_edges[twin_hi].twin = hi;
            }
            edge_map.insert((from_v, to_v), hi);
        }
    }
    /// Alias for `n_faces()`.
    pub fn num_faces(&self) -> usize {
        self.n_faces()
    }
    /// Alias for `n_faces()`.
    pub fn face_count(&self) -> usize {
        self.n_faces()
    }
    /// Alias for `n_vertices()`.
    pub fn vertex_count(&self) -> usize {
        self.n_vertices()
    }
    /// Alias for `n_vertices()` (no deletion tracking in this implementation).
    pub fn active_vertex_count(&self) -> usize {
        self.n_vertices()
    }
    /// Alias for `n_faces()` (no deletion tracking in this implementation).
    pub fn active_face_count(&self) -> usize {
        self.n_faces()
    }
    /// Compute the unit normal for face `fi`.
    pub fn face_normal(&self, fi: usize) -> [f64; 3] {
        let [v0, v1, v2] = self.face_vertices(fi);
        let a = self.vertices[v0];
        let b = self.vertices[v1];
        let c = self.vertices[v2];
        let n = cross3(sub3(b, a), sub3(c, a));
        normalize3(n)
    }
    /// Compute the centroid of face `fi`.
    pub fn face_centroid(&self, fi: usize) -> [f64; 3] {
        let [v0, v1, v2] = self.face_vertices(fi);
        let a = self.vertices[v0];
        let b = self.vertices[v1];
        let c = self.vertices[v2];
        [
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ]
    }
    /// Return boundary edges as `(from_vertex, to_vertex)` pairs.
    pub fn boundary_edges(&self) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        for (hi, he) in self.half_edges.iter().enumerate() {
            if he.twin == usize::MAX {
                let fi = he.face;
                let base = self.face_starts[fi];
                let local = hi - base;
                let prev_local = (local + 2) % 3;
                let from_v = self.half_edges[base + prev_local].vertex;
                let to_v = he.vertex;
                result.push((from_v, to_v));
            }
        }
        result
    }
    /// Group boundary edges into connected loops (lists of vertex indices).
    pub fn boundary_loops(&self) -> Vec<Vec<usize>> {
        let edges = self.boundary_edges();
        let mut adj: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &(from, to) in &edges {
            adj.insert(from, to);
        }
        let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut loops = Vec::new();
        for &(start, _) in &edges {
            if visited.contains(&start) {
                continue;
            }
            let mut loop_verts = Vec::new();
            let mut cur = start;
            loop {
                if visited.contains(&cur) {
                    break;
                }
                visited.insert(cur);
                loop_verts.push(cur);
                if let Some(&next) = adj.get(&cur) {
                    cur = next;
                } else {
                    break;
                }
            }
            if !loop_verts.is_empty() {
                loops.push(loop_verts);
            }
        }
        loops
    }
    /// Check whether vertex `v` is on the boundary (has at least one boundary half-edge).
    pub fn is_boundary_vertex(&self, v: usize) -> bool {
        for he in &self.half_edges {
            if he.vertex == v && he.twin == usize::MAX {
                return true;
            }
        }
        // Also check if vertex is the "from" vertex of a boundary half-edge
        for (hi, he) in self.half_edges.iter().enumerate() {
            if he.twin == usize::MAX {
                let fi = he.face;
                let base = self.face_starts[fi];
                let local = hi - base;
                let prev_local = (local + 2) % 3;
                if self.half_edges[base + prev_local].vertex == v {
                    return true;
                }
            }
        }
        false
    }
    /// Find all vertex neighbors of vertex `v` (vertices connected by an edge).
    pub fn vertex_neighbors(&self, v: usize) -> Vec<usize> {
        let mut neighbors = std::collections::HashSet::new();
        for he in &self.half_edges {
            let fi = he.face;
            let base = self.face_starts[fi];
            let local_idx = (0..3).find(|&k| {
                let idx = base + k;
                idx < self.half_edges.len() && self.half_edges[idx].vertex == v
            });
            if local_idx.is_some() {
                // v is one of the vertices of this face
                let fv = self.face_vertices(fi);
                for &fvi in &fv {
                    if fvi != v {
                        neighbors.insert(fvi);
                    }
                }
            }
        }
        neighbors.into_iter().collect()
    }
    /// Extract the mesh as (vertices, triangles) arrays.
    pub fn extract_triangles(&self) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
        let verts = self.vertices.clone();
        let tris: Vec<[usize; 3]> = (0..self.n_faces())
            .map(|fi| self.face_vertices(fi))
            .collect();
        (verts, tris)
    }
}
/// Classification of a point relative to a BSP plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneClass {
    /// Point is in front (positive half-space).
    Front,
    /// Point is behind (negative half-space).
    Back,
    /// Point is on the plane.
    OnPlane,
}
/// Material properties attached to a mesh for CSG operations.
#[derive(Debug, Clone)]
pub struct CsgMaterial {
    /// Unique material ID.
    pub id: u32,
    /// Material name.
    pub name: String,
    /// Density in kg/m³.
    pub density: f64,
}
/// The type of boolean operation to perform between two meshes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp {
    /// A ∪ B — all geometry covered by at least one mesh.
    Union,
    /// A ∩ B — geometry common to both meshes.
    Intersection,
    /// A \ B — geometry in A but not in B.
    Difference,
}
/// A BSP plane: `normal·x = offset`.
#[derive(Debug, Clone, Copy)]
pub struct BspPlane {
    /// Outward-facing unit normal.
    pub normal: [f64; 3],
    /// Plane offset: `normal·p = offset` for points on the plane.
    pub offset: f64,
}
impl BspPlane {
    /// Create a new BSP plane.
    pub fn new(normal: [f64; 3], offset: f64) -> Self {
        Self {
            normal: normalize3(normal),
            offset,
        }
    }
    /// Signed distance from a point to the plane.
    #[inline]
    pub fn signed_dist(&self, p: [f64; 3]) -> f64 {
        dot3(self.normal, p) - self.offset
    }
    /// Classify a point: positive = front, negative = back, zero = on plane.
    pub fn classify(&self, p: [f64; 3]) -> f64 {
        self.signed_dist(p)
    }
}
/// A half-edge in a triangle mesh.
///
/// Half-edges form the basis for traversal and boolean operations on meshes.
#[derive(Debug, Clone)]
pub struct HalfEdge {
    /// Index of the vertex this half-edge points to.
    pub vertex: usize,
    /// Index of the face this half-edge belongs to.
    pub face: usize,
    /// Index of the next half-edge in the same face.
    pub next: usize,
    /// Index of the twin half-edge (opposite direction), or `usize::MAX` if boundary.
    pub twin: usize,
}
/// Performs boolean operations with a configurable epsilon tolerance for
/// near-coincident geometry.
///
/// The exact arrangement of [`MeshBoolean`] decides every coincidence exactly,
/// so geometry that coincides only up to round-off (CAD exports, transformed
/// copies) is treated as nearly parallel, nearly touching surfaces, and the
/// seams it leaves can be open.  Before the arrangement runs, the vertices of
/// both operands are therefore cleaned at the tolerance `epsilon`:
///
/// 1. all vertices closer than `epsilon` are merged (one consistent merge of
///    the two vertex sets, earlier vertices winning, those of the first operand
///    first), so nearly coincident corners, and the edges and faces they span,
///    become exactly coincident;
/// 2. every coordinate is rounded to a multiple of `epsilon` (snap rounding),
///    so nearly coplanar faces that are aligned with a coordinate plane get
///    exactly equal coordinates and are resolved as coincident faces.  Tilted
///    faces that coincide only up to `epsilon` are not made coplanar by this.
///
/// No vertex moves by `2 * epsilon` or more.  The result is merged at `epsilon`
/// again, which collapses the steps and slivers thinner than `epsilon` that
/// remain, so no two distinct result vertices are closer than `epsilon`.
pub struct RobustMeshBoolean;
impl RobustMeshBoolean {
    /// Execute a boolean operation with the vertex snapping tolerance `epsilon`.
    ///
    /// `epsilon = 0` disables the snapping and is equivalent to
    /// [`MeshBoolean::execute`].
    ///
    /// # Errors
    ///
    /// * [`Error::OutOfRange`] when `epsilon` is negative or not finite.
    /// * The operand validation errors of [`MeshBoolean::execute`].
    pub fn execute(
        op: BooleanOp,
        mesh_a: &TriangleMesh,
        mesh_b: &TriangleMesh,
        epsilon: f64,
    ) -> Result<TriangleMesh> {
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(Error::out_of_range("epsilon", epsilon, 0.0, f64::MAX));
        }
        let a = to_simple_mesh("mesh_a", mesh_a)?;
        let b = to_simple_mesh("mesh_b", mesh_b)?;
        let (a, b) = snap_operands(&a, &b, epsilon);
        let result = mesh_boolean(&a, &b, mesh_op(op));
        Ok(from_simple_mesh(&tidy_result(result, epsilon)))
    }
}
/// Boolean operations specialised for (and validated against) manifold meshes.
///
/// Both inputs must be closed, consistently oriented 2-manifolds: every edge is
/// shared by exactly two triangles that traverse it in opposite directions.
/// Anything else is rejected with an error instead of being processed.
pub struct ManifoldBoolean;
impl ManifoldBoolean {
    /// Execute a boolean operation, validating manifold correctness.
    ///
    /// Non-manifold input returns an error; this is
    /// [`ManifoldBoolean::execute_strict`].  Use [`MeshBoolean::execute`] for
    /// the unchecked operation.
    pub fn execute(
        op: BooleanOp,
        mesh_a: &TriangleMesh,
        mesh_b: &TriangleMesh,
    ) -> Result<TriangleMesh> {
        Self::execute_strict(op, mesh_a, mesh_b)
    }
    /// Execute boolean operation only if both meshes are strictly manifold.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidMesh`] when an operand has boundary edges, edges
    ///   shared by more than two triangles, or inconsistently oriented
    ///   triangles.
    /// * The operand validation errors of [`MeshBoolean::execute`].
    pub fn execute_strict(
        op: BooleanOp,
        mesh_a: &TriangleMesh,
        mesh_b: &TriangleMesh,
    ) -> Result<TriangleMesh> {
        let a = to_simple_mesh("mesh_a", mesh_a)?;
        let b = to_simple_mesh("mesh_b", mesh_b)?;
        require_closed_oriented("mesh_a", &mesh_a.indices)?;
        require_closed_oriented("mesh_b", &mesh_b.indices)?;
        Ok(from_simple_mesh(&mesh_boolean(&a, &b, mesh_op(op))))
    }
}
