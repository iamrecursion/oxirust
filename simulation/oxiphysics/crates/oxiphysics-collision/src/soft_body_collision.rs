// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Soft body and cloth collision detection and response.
//!
//! Covers:
//! - Vertex-triangle distance queries for deformable surface meshes
//! - Cloth self-collision using spatial hashing
//! - Collision response impulses for deformable bodies
//! - Continuous collision detection (CCD) for cloth edges
//! - Penalty force springs for interpenetration
//! - Bounding volume hierarchy (BVH) for dynamic meshes
//! - Contact manifold generation for soft bodies
//! - Friction model for cloth-rigid contact
//! - Tearing and cutting along collision boundaries (conceptual)
//! - GPU-friendly flat collision data structures

use std::collections::HashMap;

mod edge_ccd;

// ─────────────────────────────────────────────────────────────────────────────
// Internal vector helpers — no nalgebra
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn len3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

#[inline]
fn normalize3(a: [f64; 3]) -> [f64; 3] {
    let l = len3(a);
    if l < 1e-14 {
        [0.0, 0.0, 1.0]
    } else {
        scale3(a, 1.0 / l)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vertex-triangle collision
// ─────────────────────────────────────────────────────────────────────────────

/// Closest point on a triangle (a, b, c) to point p, plus barycentric coords.
///
/// Returns (squared_distance, barycentric \[u, v, w\]).
pub fn vertex_triangle_closest(
    p: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
) -> (f64, [f64; 3]) {
    let ab = sub3(b, a);
    let ac = sub3(c, a);
    let ap = sub3(p, a);

    let d1 = dot3(ab, ap);
    let d2 = dot3(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        let diff = sub3(p, a);
        return (dot3(diff, diff), [1.0, 0.0, 0.0]);
    }

    let bp = sub3(p, b);
    let d3 = dot3(ab, bp);
    let d4 = dot3(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        let diff = sub3(p, b);
        return (dot3(diff, diff), [0.0, 1.0, 0.0]);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let closest = add3(a, scale3(ab, v));
        let diff = sub3(p, closest);
        return (dot3(diff, diff), [1.0 - v, v, 0.0]);
    }

    let cp = sub3(p, c);
    let d5 = dot3(ab, cp);
    let d6 = dot3(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        let diff = sub3(p, c);
        return (dot3(diff, diff), [0.0, 0.0, 1.0]);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let closest = add3(a, scale3(ac, w));
        let diff = sub3(p, closest);
        return (dot3(diff, diff), [1.0 - w, 0.0, w]);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let bc = sub3(c, b);
        let closest = add3(b, scale3(bc, w));
        let diff = sub3(p, closest);
        return (dot3(diff, diff), [0.0, 1.0 - w, w]);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let closest = add3(add3(a, scale3(ab, v)), scale3(ac, w));
    let diff = sub3(p, closest);
    (dot3(diff, diff), [1.0 - v - w, v, w])
}

/// A vertex-triangle contact for a soft body surface mesh.
#[derive(Debug, Clone)]
pub struct VertexTriangleContact {
    /// Vertex index (in the deformable mesh).
    pub vertex_index: usize,
    /// Triangle index (in the rigid or static mesh).
    pub triangle_index: usize,
    /// Contact normal pointing from triangle to vertex.
    pub normal: [f64; 3],
    /// Penetration depth (positive = interpenetrating).
    pub depth: f64,
    /// Barycentric coordinates of the closest point on the triangle.
    pub bary: [f64; 3],
}

/// Detect contacts between deformable vertices and a triangle mesh.
///
/// Returns all vertex-triangle pairs where the vertex is within `thickness`
/// of the triangle surface.
pub fn detect_vertex_triangle_contacts(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
    tri_positions: &[[f64; 3]],
    thickness: f64,
) -> Vec<VertexTriangleContact> {
    let threshold_sq = thickness * thickness;
    let mut contacts = Vec::new();

    for (vi, &vp) in vertices.iter().enumerate() {
        for (ti, &tri) in triangles.iter().enumerate() {
            let a = tri_positions[tri[0]];
            let b = tri_positions[tri[1]];
            let c = tri_positions[tri[2]];

            let (dist_sq, bary) = vertex_triangle_closest(vp, a, b, c);
            if dist_sq < threshold_sq {
                let closest = add3(
                    add3(scale3(a, bary[0]), scale3(b, bary[1])),
                    scale3(c, bary[2]),
                );
                let to_vert = sub3(vp, closest);
                let dist = dist_sq.sqrt();
                let normal = if dist > 1e-14 {
                    scale3(to_vert, 1.0 / dist)
                } else {
                    let ab = sub3(b, a);
                    let ac = sub3(c, a);
                    normalize3(cross3(ab, ac))
                };
                contacts.push(VertexTriangleContact {
                    vertex_index: vi,
                    triangle_index: ti,
                    normal,
                    depth: thickness - dist,
                    bary,
                });
            }
        }
    }
    contacts
}

// ─────────────────────────────────────────────────────────────────────────────
// Cloth self-collision — spatial hashing
// ─────────────────────────────────────────────────────────────────────────────

/// Spatial hash for cloth vertex pairs, used for self-collision detection.
pub struct ClothSpatialHash {
    /// Cell size.
    pub cell_size: f64,
    /// Map from grid cell key to vertex indices.
    cells: HashMap<(i64, i64, i64), Vec<usize>>,
}

impl ClothSpatialHash {
    /// Construct an empty spatial hash with given cell size.
    pub fn new(cell_size: f64) -> Self {
        ClothSpatialHash {
            cell_size,
            cells: HashMap::new(),
        }
    }

    fn cell_of(&self, p: [f64; 3]) -> (i64, i64, i64) {
        (
            (p[0] / self.cell_size).floor() as i64,
            (p[1] / self.cell_size).floor() as i64,
            (p[2] / self.cell_size).floor() as i64,
        )
    }

    /// Insert all vertices into the spatial hash.
    pub fn build(&mut self, positions: &[[f64; 3]]) {
        self.cells.clear();
        for (i, &p) in positions.iter().enumerate() {
            let key = self.cell_of(p);
            self.cells.entry(key).or_default().push(i);
        }
    }

    /// Query all candidate self-collision vertex pairs within cell radius 1.
    ///
    /// Returns pairs (i, j) with i < j whose cells are adjacent.
    pub fn candidate_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        let all_keys: Vec<(i64, i64, i64)> = self.cells.keys().copied().collect();
        for &(cx, cy, cz) in &all_keys {
            for dx in -1_i64..=1 {
                for dy in -1_i64..=1 {
                    for dz in -1_i64..=1 {
                        let neighbor = (cx + dx, cy + dy, cz + dz);
                        if let Some(other) = self.cells.get(&neighbor)
                            && let Some(this) = self.cells.get(&(cx, cy, cz))
                        {
                            for &i in this {
                                for &j in other {
                                    if i < j {
                                        pairs.push((i, j));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    /// Detect self-collision contacts for all candidate pairs.
    ///
    /// Returns pairs with distances below `radius`.
    pub fn detect_self_collisions(
        &self,
        positions: &[[f64; 3]],
        radius: f64,
    ) -> Vec<(usize, usize, [f64; 3], f64)> {
        let candidates = self.candidate_pairs();
        let mut contacts = Vec::new();
        for (i, j) in candidates {
            if i >= positions.len() || j >= positions.len() {
                continue;
            }
            let diff = sub3(positions[i], positions[j]);
            let dist = len3(diff);
            if dist < radius && dist > 1e-14 {
                let normal = scale3(diff, 1.0 / dist);
                contacts.push((i, j, normal, radius - dist));
            }
        }
        contacts
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Collision response impulse for deformable bodies
// ─────────────────────────────────────────────────────────────────────────────

/// Velocity impulse to apply to a vertex after collision resolution.
#[derive(Debug, Clone)]
pub struct DeformableImpulse {
    /// Target vertex index.
    pub vertex_index: usize,
    /// Velocity delta to apply.
    pub delta_v: [f64; 3],
    /// Position correction vector.
    pub correction: [f64; 3],
}

/// Compute collision response impulses for vertex-triangle contacts.
///
/// Uses a simple inelastic impulse model.  `restitution` ∈ \[0, 1\].
pub fn compute_contact_impulses(
    contacts: &[VertexTriangleContact],
    velocities: &[[f64; 3]],
    masses: &[f64],
    restitution: f64,
    friction_coeff: f64,
) -> Vec<DeformableImpulse> {
    let mut impulses = Vec::new();
    for c in contacts {
        let vi = c.vertex_index;
        if vi >= velocities.len() || vi >= masses.len() {
            continue;
        }
        let v = velocities[vi];
        let m = masses[vi];
        if m <= 0.0 {
            continue;
        }

        let v_normal = dot3(v, c.normal);
        if v_normal >= 0.0 {
            // Already separating
            continue;
        }

        // Normal impulse
        let j_n = -(1.0 + restitution) * v_normal * m;
        let impulse_n = scale3(c.normal, j_n / m);

        // Tangential friction impulse
        let v_tang = sub3(v, scale3(c.normal, v_normal));
        let v_tang_len = len3(v_tang);
        let impulse_t = if v_tang_len > 1e-14 {
            let friction_mag = friction_coeff * j_n.abs() / m;
            scale3(v_tang, -friction_mag.min(v_tang_len) / v_tang_len)
        } else {
            [0.0; 3]
        };

        let delta_v = add3(impulse_n, impulse_t);
        let correction = scale3(c.normal, c.depth.max(0.0));

        impulses.push(DeformableImpulse {
            vertex_index: vi,
            delta_v,
            correction,
        });
    }
    impulses
}

// ─────────────────────────────────────────────────────────────────────────────
// CCD for cloth edges
// ─────────────────────────────────────────────────────────────────────────────

/// Edge-edge CCD result for cloth collision.
#[derive(Debug, Clone)]
pub struct EdgeCcdResult {
    /// Edge index in mesh A.
    pub edge_a: usize,
    /// Edge index in mesh B (or same mesh for self-collision).
    pub edge_b: usize,
    /// Time of impact in \[0, 1\].
    pub toi: f64,
    /// Contact normal at impact.
    pub normal: [f64; 3],
}

/// Continuous collision detection for a pair of linearly-moving edges.
///
/// Edge A joins vertex `p` and vertex `q`; each vertex is given by its position
/// at the start (`*0*`) and at the end (`*1*`) of the step and moves linearly in
/// between. The argument pairs are, in order, `(p0a, p1a)` and `(q0a, q1a)` for
/// edge A and `(p0b, p1b)` and `(q0b, q1b)` for edge B, `radius` being the
/// contact distance.
///
/// Returns the earliest time of impact `t ∈ [0, 1]`, or `None` if the edges stay
/// further apart than `radius` during the whole step. The time is exact rather
/// than sampled, so a fast crossing that lies entirely between two time samples
/// is not missed:
///
/// * it is `0` if the edges start within `radius` of each other;
/// * otherwise it is the earliest root of the coplanarity cubic of the four end
///   points at which the segments are within `radius` (Provot 1997, Bridson et
///   al. 2002); for edges that cross, the time at which they cross;
/// * if there is no such root (the edges come within `radius` of each other
///   without ever being coplanar, or they move in a common plane) it is the
///   first time the distance between the segments reaches `radius`, located by
///   interval subdivision with a Lipschitz bound on the speed of that distance,
///   which cannot step over a brief dip of the distance. (A pair that hovers
///   just above `radius` for the whole step cannot be certified apart within the
///   bounded search; it is reported as touching rather than risk a missed
///   contact.)
///
/// A `radius` of zero detects exact crossings (the radius used is never smaller
/// than a rounding floor of `1e-10` times the extent of the pair). A negative or
/// NaN `radius`, or a non-finite coordinate, gives `None`. The result does not
/// depend on the units or the position of the mesh.
pub fn edge_edge_ccd(
    p0a: [f64; 3],
    p1a: [f64; 3],
    q0a: [f64; 3],
    q1a: [f64; 3],
    p0b: [f64; 3],
    p1b: [f64; 3],
    q0b: [f64; 3],
    q1b: [f64; 3],
    radius: f64,
) -> Option<f64> {
    let edge_a = edge_ccd::MovingEdge {
        start: [p0a, q0a],
        end: [p1a, q1a],
    };
    let edge_b = edge_ccd::MovingEdge {
        start: [p0b, q0b],
        end: [p1b, q1b],
    };
    edge_ccd::edge_pair_contact(&edge_a, &edge_b, radius).map(|contact| contact.toi)
}

/// Batch CCD for all edge pairs of two edge sets over one step.
///
/// `edges_a` and `edges_b` are (start, end) vertex index pairs into the position
/// arrays of their mesh: `verts0_*` hold the positions at the start of the step,
/// `verts1_*` those at its end, and every vertex moves linearly in between. Each
/// pair `(edge_a, edge_b)` that comes into contact yields one [`EdgeCcdResult`]
/// whose indices refer to `edges_a` and `edges_b`. Its `toi` is the time of
/// impact of [`edge_edge_ccd`] with `radius` as the contact distance, and its
/// `normal` is the unit contact normal pointing from edge B towards edge A: the
/// direction between the closest points of the two edges at the time of impact,
/// or, when those coincide (the edges cross), the common perpendicular of the
/// edges oriented towards the side on which edge A started the step.
///
/// `same_mesh` tells whether both edge sets index the same vertex arrays, as in
/// cloth self-collision. Edges of one mesh that share a vertex always touch
/// there, so with `same_mesh` set such pairs are skipped. Without it the sets
/// belong to different meshes, equal vertex indices name different vertices, and
/// no pair is skipped for its indices (if the same mesh is passed that way, edges
/// that are neighbours are reported in contact at `toi = 0`).
///
/// Edges with a vertex index outside the position arrays of their mesh are
/// skipped. A negative or NaN `radius` reports nothing.
pub fn batch_edge_edge_ccd(
    edges_a: &[(usize, usize)],
    verts0_a: &[[f64; 3]],
    verts1_a: &[[f64; 3]],
    edges_b: &[(usize, usize)],
    verts0_b: &[[f64; 3]],
    verts1_b: &[[f64; 3]],
    radius: f64,
    same_mesh: bool,
) -> Vec<EdgeCcdResult> {
    edge_ccd::edge_edge_batch(
        radius, same_mesh, edges_a, verts0_a, verts1_a, edges_b, verts0_b, verts1_b,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Penalty force springs for interpenetration
// ─────────────────────────────────────────────────────────────────────────────

/// Penalty spring configuration for interpenetration resolution.
pub struct PenaltySpring {
    /// Spring stiffness (force per unit penetration depth).
    pub stiffness: f64,
    /// Damping coefficient.
    pub damping: f64,
}

impl PenaltySpring {
    /// Construct a penalty spring with given stiffness and damping.
    pub fn new(stiffness: f64, damping: f64) -> Self {
        PenaltySpring { stiffness, damping }
    }

    /// Compute penalty force for a single interpenetrating vertex.
    ///
    /// `depth` is the penetration depth (positive), `normal` is the contact
    /// normal pointing outward, `rel_velocity` is the relative velocity at the
    /// contact point.  Returns the penalty force vector.
    pub fn force(&self, depth: f64, normal: [f64; 3], rel_velocity: [f64; 3]) -> [f64; 3] {
        if depth <= 0.0 {
            return [0.0; 3];
        }
        let v_n = dot3(rel_velocity, normal);
        let f_mag = self.stiffness * depth - self.damping * v_n;
        scale3(normal, f_mag.max(0.0))
    }

    /// Apply penalty forces to all vertex-triangle contacts.
    ///
    /// Returns force vectors indexed by vertex index (accumulated).
    pub fn apply_all(
        &self,
        contacts: &[VertexTriangleContact],
        velocities: &[[f64; 3]],
    ) -> Vec<(usize, [f64; 3])> {
        let mut forces = Vec::new();
        for c in contacts {
            let vi = c.vertex_index;
            let vel = if vi < velocities.len() {
                velocities[vi]
            } else {
                [0.0; 3]
            };
            let f = self.force(c.depth, c.normal, vel);
            forces.push((vi, f));
        }
        forces
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BVH for dynamic meshes
// ─────────────────────────────────────────────────────────────────────────────

/// Axis-aligned bounding box for BVH nodes.
#[derive(Debug, Clone)]
pub struct BvhAabb {
    /// Minimum corner.
    pub min: [f64; 3],
    /// Maximum corner.
    pub max: [f64; 3],
}

impl BvhAabb {
    /// Construct from min/max corners.
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        BvhAabb { min, max }
    }

    /// Construct from three triangle vertices with an optional margin.
    pub fn from_triangle(a: [f64; 3], b: [f64; 3], c: [f64; 3], margin: f64) -> Self {
        let min = [
            a[0].min(b[0]).min(c[0]) - margin,
            a[1].min(b[1]).min(c[1]) - margin,
            a[2].min(b[2]).min(c[2]) - margin,
        ];
        let max = [
            a[0].max(b[0]).max(c[0]) + margin,
            a[1].max(b[1]).max(c[1]) + margin,
            a[2].max(b[2]).max(c[2]) + margin,
        ];
        BvhAabb { min, max }
    }

    /// Test overlap with another AABB.
    pub fn overlaps(&self, other: &BvhAabb) -> bool {
        self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
    }

    /// Merge two AABBs into their union.
    pub fn merge(&self, other: &BvhAabb) -> BvhAabb {
        BvhAabb {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    /// Surface area of the AABB (for SAH cost).
    pub fn surface_area(&self) -> f64 {
        let dx = self.max[0] - self.min[0];
        let dy = self.max[1] - self.min[1];
        let dz = self.max[2] - self.min[2];
        2.0 * (dx * dy + dy * dz + dz * dx)
    }
}

/// A flat BVH node for dynamic soft body mesh.
#[derive(Debug, Clone)]
pub struct BvhNode {
    /// Bounding box.
    pub aabb: BvhAabb,
    /// Left child index (usize::MAX = leaf).
    pub left: usize,
    /// Right child index (usize::MAX = leaf).
    pub right: usize,
    /// Triangle index for leaf nodes.
    pub tri_index: usize,
}

impl BvhNode {
    /// Is this a leaf node?
    pub fn is_leaf(&self) -> bool {
        self.left == usize::MAX
    }
}

/// One triangle waiting to be placed in a [`SoftBodyBvh`].
struct BvhLeaf {
    /// Index of the triangle in the input.
    tri_index: usize,
    /// Bounding box of the triangle, grown by the collision margin.
    aabb: BvhAabb,
    /// Centroid of the triangle, the key the tree is partitioned by.
    centroid: [f64; 3],
}

/// A BVH built over a triangle mesh (flat array representation).
///
/// The nodes are stored in depth-first order with the root at index 0 (the
/// array is empty when there is no triangle): the left child of an internal node
/// directly follows it, so the children of a node always have larger indices
/// than the node. A leaf holds one triangle, in `tri_index`.
pub struct SoftBodyBvh {
    /// Flat array of nodes (root = index 0).
    pub nodes: Vec<BvhNode>,
}

impl SoftBodyBvh {
    /// Build a BVH for a set of triangles with a collision margin.
    ///
    /// Every node is split at the median of the triangle centroids along the
    /// longest axis of their bounding box, so the tree is balanced whatever the
    /// order of the input: it has `2 n - 1` nodes for `n` triangles and a depth
    /// of `ceil(log2(n)) + 1` nodes. Triangles with a vertex index outside
    /// `positions` are left out of the tree. The build uses an explicit stack, so
    /// the size of the mesh is not limited by the depth of the call stack.
    pub fn build(triangles: &[[usize; 3]], positions: &[[f64; 3]], margin: f64) -> Self {
        let mut leaves: Vec<BvhLeaf> = triangles
            .iter()
            .enumerate()
            .filter_map(|(tri_index, tri)| {
                let a = *positions.get(tri[0])?;
                let b = *positions.get(tri[1])?;
                let c = *positions.get(tri[2])?;
                Some(BvhLeaf {
                    tri_index,
                    aabb: BvhAabb::from_triangle(a, b, c, margin),
                    centroid: scale3(add3(add3(a, b), c), 1.0 / 3.0),
                })
            })
            .collect();

        let mut nodes = Vec::with_capacity((2 * leaves.len()).saturating_sub(1));
        // Ranges of `leaves` still to be turned into subtrees, as
        // `(node index, start, end)`. The left half of a range is pushed last, so
        // it is processed first and the nodes come out in depth-first order.
        let mut pending = Vec::new();
        if !leaves.is_empty() {
            pending.push((0, 0, leaves.len()));
        }
        while let Some((index, start, end)) = pending.pop() {
            let Some(range) = leaves.get_mut(start..end) else {
                continue;
            };
            let Some(first) = range.first() else {
                continue;
            };
            if range.len() == 1 {
                nodes.push(BvhNode {
                    aabb: first.aabb.clone(),
                    left: usize::MAX,
                    right: usize::MAX,
                    tri_index: first.tri_index,
                });
                continue;
            }

            let aabb = range
                .iter()
                .skip(1)
                .fold(first.aabb.clone(), |union, leaf| union.merge(&leaf.aabb));
            // Split along the axis on which the centroids are spread the most.
            let mut lo = first.centroid;
            let mut hi = first.centroid;
            for leaf in range.iter() {
                for ((l, h), &c) in lo.iter_mut().zip(hi.iter_mut()).zip(leaf.centroid.iter()) {
                    *l = l.min(c);
                    *h = h.max(c);
                }
            }
            let mut axis = 0;
            let mut widest = hi[0] - lo[0];
            for k in 1..3 {
                if hi[k] - lo[k] > widest {
                    axis = k;
                    widest = hi[k] - lo[k];
                }
            }
            // Median split: the left half gets the `len / 2` smallest centroids.
            let left_len = range.len() / 2;
            range.select_nth_unstable_by(left_len, |p, q| {
                p.centroid[axis].total_cmp(&q.centroid[axis])
            });

            // The left subtree has `2 * left_len - 1` nodes and starts right
            // after this node, so the right subtree starts `2 * left_len` later.
            let left_index = index + 1;
            let right_index = index + 2 * left_len;
            nodes.push(BvhNode {
                aabb,
                left: left_index,
                right: right_index,
                tri_index: usize::MAX,
            });
            pending.push((right_index, start + left_len, end));
            pending.push((left_index, start, start + left_len));
        }
        SoftBodyBvh { nodes }
    }

    /// Query which triangles have AABBs that overlap the bounding box of a
    /// sphere.
    ///
    /// Returns the input indices of the triangles, in tree order. The traversal
    /// uses an explicit stack.
    pub fn query_sphere(&self, center: [f64; 3], radius: f64) -> Vec<usize> {
        let sphere_aabb = BvhAabb {
            min: [center[0] - radius, center[1] - radius, center[2] - radius],
            max: [center[0] + radius, center[1] + radius, center[2] + radius],
        };
        let mut result = Vec::new();
        let mut stack = Vec::new();
        if !self.nodes.is_empty() {
            stack.push(0);
        }
        while let Some(index) = stack.pop() {
            let Some(node) = self.nodes.get(index) else {
                continue;
            };
            if !node.aabb.overlaps(&sphere_aabb) {
                continue;
            }
            if node.is_leaf() {
                result.push(node.tri_index);
            } else {
                // The right child goes first so that the left one is popped first.
                stack.push(node.right);
                stack.push(node.left);
            }
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Contact manifold for soft bodies
// ─────────────────────────────────────────────────────────────────────────────

/// A single contact point in a soft-body contact manifold.
#[derive(Debug, Clone)]
pub struct SoftContact {
    /// Vertex index on the deformable mesh.
    pub vertex_index: usize,
    /// Contact normal (pointing away from obstacle).
    pub normal: [f64; 3],
    /// Penetration depth.
    pub depth: f64,
    /// Contact point position in world space.
    pub position: [f64; 3],
}

/// Contact manifold for a soft body.
pub struct SoftContactManifold {
    /// All active contact points.
    pub contacts: Vec<SoftContact>,
    /// Maximum allowed contacts.
    pub max_contacts: usize,
}

impl SoftContactManifold {
    /// Create an empty manifold with a capacity limit.
    pub fn new(max_contacts: usize) -> Self {
        SoftContactManifold {
            contacts: Vec::new(),
            max_contacts,
        }
    }

    /// Add a contact, removing the shallowest if at capacity.
    pub fn add(&mut self, contact: SoftContact) {
        if self.contacts.len() < self.max_contacts {
            self.contacts.push(contact);
        } else {
            // Replace shallowest contact
            if let Some(idx) = self
                .contacts
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.depth
                        .partial_cmp(&b.1.depth)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                && self.contacts[idx].depth < contact.depth
            {
                self.contacts[idx] = contact;
            }
        }
    }

    /// Clear all contacts.
    pub fn clear(&mut self) {
        self.contacts.clear();
    }

    /// Return the deepest contact, if any.
    pub fn deepest(&self) -> Option<&SoftContact> {
        self.contacts.iter().max_by(|a, b| {
            a.depth
                .partial_cmp(&b.depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tearing / cutting along collision boundaries (conceptual)
// ─────────────────────────────────────────────────────────────────────────────

/// Edge in a mesh that can potentially be torn.
#[derive(Debug, Clone, PartialEq)]
pub struct TearableEdge {
    /// Index of vertex A.
    pub va: usize,
    /// Index of vertex B.
    pub vb: usize,
    /// Current accumulated stress on the edge.
    pub stress: f64,
    /// Stress threshold above which the edge tears.
    pub threshold: f64,
}

impl TearableEdge {
    /// Construct a tearable edge.
    pub fn new(va: usize, vb: usize, threshold: f64) -> Self {
        TearableEdge {
            va,
            vb,
            stress: 0.0,
            threshold,
        }
    }

    /// Returns true if the edge should tear.
    pub fn should_tear(&self) -> bool {
        self.stress >= self.threshold
    }

    /// Accumulate stress from a contact force magnitude.
    pub fn accumulate_stress(&mut self, force_magnitude: f64) {
        self.stress += force_magnitude;
    }
}

/// Simulate one step of tearing propagation.
///
/// Marks edges exceeding their threshold as torn and returns their indices.
pub fn propagate_tears(edges: &mut [TearableEdge]) -> Vec<usize> {
    let mut torn = Vec::new();
    for (i, edge) in edges.iter_mut().enumerate() {
        if edge.should_tear() {
            torn.push(i);
        }
    }
    torn
}

/// Apply stress to edges adjacent to a collision contact.
///
/// Increases stress proportional to the contact depth for edges that share
/// the contacted vertex.
pub fn apply_collision_stress(
    edges: &mut [TearableEdge],
    vertex_index: usize,
    contact_depth: f64,
    stress_factor: f64,
) {
    for edge in edges.iter_mut() {
        if edge.va == vertex_index || edge.vb == vertex_index {
            edge.accumulate_stress(contact_depth * stress_factor);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU-friendly flat collision data structures
// ─────────────────────────────────────────────────────────────────────────────

/// GPU-friendly flat vertex buffer for cloth simulation.
///
/// All data is stored in structure-of-arrays (SoA) layout for coalesced
/// GPU memory access.
pub struct GpuClothBuffer {
    /// X positions of all vertices.
    pub pos_x: Vec<f32>,
    /// Y positions of all vertices.
    pub pos_y: Vec<f32>,
    /// Z positions of all vertices.
    pub pos_z: Vec<f32>,
    /// X velocities.
    pub vel_x: Vec<f32>,
    /// Y velocities.
    pub vel_y: Vec<f32>,
    /// Z velocities.
    pub vel_z: Vec<f32>,
    /// Inverse masses (0 = pinned).
    pub inv_mass: Vec<f32>,
}

impl GpuClothBuffer {
    /// Construct from Rust `f64` vertex arrays.
    pub fn from_vertices(positions: &[[f64; 3]], velocities: &[[f64; 3]], masses: &[f64]) -> Self {
        let n = positions.len();
        let mut buf = GpuClothBuffer {
            pos_x: Vec::with_capacity(n),
            pos_y: Vec::with_capacity(n),
            pos_z: Vec::with_capacity(n),
            vel_x: Vec::with_capacity(n),
            vel_y: Vec::with_capacity(n),
            vel_z: Vec::with_capacity(n),
            inv_mass: Vec::with_capacity(n),
        };
        for &p in positions.iter() {
            buf.pos_x.push(p[0] as f32);
            buf.pos_y.push(p[1] as f32);
            buf.pos_z.push(p[2] as f32);
        }
        for &v in velocities.iter() {
            buf.vel_x.push(v[0] as f32);
            buf.vel_y.push(v[1] as f32);
            buf.vel_z.push(v[2] as f32);
        }
        for &m in masses {
            buf.inv_mass
                .push(if m > 0.0 { (1.0 / m) as f32 } else { 0.0 });
        }
        buf
    }

    /// Number of vertices.
    pub fn len(&self) -> usize {
        self.pos_x.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.pos_x.is_empty()
    }

    /// Read a vertex position back to `[f64; 3]`.
    pub fn position(&self, i: usize) -> [f64; 3] {
        [
            self.pos_x[i] as f64,
            self.pos_y[i] as f64,
            self.pos_z[i] as f64,
        ]
    }
}

/// GPU-friendly flat contact buffer (SoA layout).
///
/// Suitable for uploading to a GPU for parallel impulse resolution.
pub struct GpuContactBuffer {
    /// Vertex indices for each contact.
    pub vertex_indices: Vec<u32>,
    /// Contact normal X.
    pub normal_x: Vec<f32>,
    /// Contact normal Y.
    pub normal_y: Vec<f32>,
    /// Contact normal Z.
    pub normal_z: Vec<f32>,
    /// Penetration depths.
    pub depths: Vec<f32>,
}

impl GpuContactBuffer {
    /// Construct an empty GPU contact buffer.
    pub fn new() -> Self {
        GpuContactBuffer {
            vertex_indices: Vec::new(),
            normal_x: Vec::new(),
            normal_y: Vec::new(),
            normal_z: Vec::new(),
            depths: Vec::new(),
        }
    }

    /// Populate from vertex-triangle contacts.
    pub fn from_contacts(contacts: &[VertexTriangleContact]) -> Self {
        let mut buf = Self::new();
        for c in contacts {
            buf.vertex_indices.push(c.vertex_index as u32);
            buf.normal_x.push(c.normal[0] as f32);
            buf.normal_y.push(c.normal[1] as f32);
            buf.normal_z.push(c.normal[2] as f32);
            buf.depths.push(c.depth as f32);
        }
        buf
    }

    /// Number of contacts.
    pub fn len(&self) -> usize {
        self.vertex_indices.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.vertex_indices.is_empty()
    }
}

impl Default for GpuContactBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU-friendly edge buffer (SoA) for CCD.
pub struct GpuEdgeBuffer {
    /// Start vertex indices.
    pub start: Vec<u32>,
    /// End vertex indices.
    pub end: Vec<u32>,
}

impl GpuEdgeBuffer {
    /// Construct from edge list.
    pub fn from_edges(edges: &[(usize, usize)]) -> Self {
        let start = edges.iter().map(|&(a, _)| a as u32).collect();
        let end = edges.iter().map(|&(_, b)| b as u32).collect();
        GpuEdgeBuffer { start, end }
    }

    /// Number of edges.
    pub fn len(&self) -> usize {
        self.start.len()
    }

    /// Returns true if no edges.
    pub fn is_empty(&self) -> bool {
        self.start.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod regression_tests;

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    // ── vertex_triangle_closest ───────────────────────────────────────────────

    #[test]
    fn test_vtc_vertex_above_centre() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let p = [0.25, 0.25, 1.0];
        let (sq_d, bary) = vertex_triangle_closest(p, a, b, c);
        // closest point is (0.25, 0.25, 0) → dist = 1
        assert!((sq_d - 1.0).abs() < 1e-8);
        assert!((bary[0] + bary[1] + bary[2] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_vtc_vertex_at_corner() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let (sq_d, _bary) = vertex_triangle_closest(a, a, b, c);
        assert!(sq_d.abs() < EPS);
    }

    #[test]
    fn test_vtc_vertex_outside_edge() {
        // Point projected off AB edge
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let p = [0.5, -1.0, 0.0];
        let (sq_d, _) = vertex_triangle_closest(p, a, b, c);
        // closest on edge AB at (0.5, 0, 0), dist = 1
        assert!((sq_d - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_vtc_bary_sum_to_one() {
        let a = [0.0, 0.0, 0.0];
        let b = [2.0, 0.0, 0.0];
        let c = [0.0, 2.0, 0.0];
        let p = [0.5, 0.5, 0.5];
        let (_, bary) = vertex_triangle_closest(p, a, b, c);
        let sum = bary[0] + bary[1] + bary[2];
        assert!((sum - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_vtc_distance_non_negative() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let p = [5.0, 5.0, 5.0];
        let (sq_d, _) = vertex_triangle_closest(p, a, b, c);
        assert!(sq_d >= 0.0);
    }

    // ── detect_vertex_triangle_contacts ───────────────────────────────────────

    #[test]
    fn test_detect_contacts_finds_close_vertex() {
        let verts = vec![[0.25, 0.25, 0.05]];
        let tris = vec![[0, 1, 2]];
        let tri_pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let contacts = detect_vertex_triangle_contacts(&verts, &tris, &tri_pos, 0.1);
        assert_eq!(contacts.len(), 1);
        assert!(contacts[0].depth > 0.0);
    }

    #[test]
    fn test_detect_contacts_misses_far_vertex() {
        let verts = vec![[0.25, 0.25, 5.0]];
        let tris = vec![[0, 1, 2]];
        let tri_pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let contacts = detect_vertex_triangle_contacts(&verts, &tris, &tri_pos, 0.1);
        assert_eq!(contacts.len(), 0);
    }

    #[test]
    fn test_detect_contacts_normal_points_away_from_tri() {
        let verts = vec![[0.25, 0.25, 0.05]];
        let tris = vec![[0, 1, 2]];
        let tri_pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let contacts = detect_vertex_triangle_contacts(&verts, &tris, &tri_pos, 0.1);
        assert_eq!(contacts.len(), 1);
        // Normal should point toward vertex (positive z component)
        assert!(contacts[0].normal[2] > 0.0);
    }

    // ── ClothSpatialHash ───────────────────────────────────────────────────────

    #[test]
    fn test_spatial_hash_finds_nearby_pair() {
        let mut sh = ClothSpatialHash::new(0.5);
        let positions = vec![[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [10.0, 0.0, 0.0]];
        sh.build(&positions);
        let contacts = sh.detect_self_collisions(&positions, 0.3);
        assert_eq!(contacts.len(), 1);
        assert_eq!((contacts[0].0, contacts[0].1), (0, 1));
    }

    #[test]
    fn test_spatial_hash_no_contacts_far_apart() {
        let mut sh = ClothSpatialHash::new(1.0);
        let positions = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        sh.build(&positions);
        let contacts = sh.detect_self_collisions(&positions, 0.3);
        assert_eq!(contacts.len(), 0);
    }

    #[test]
    fn test_spatial_hash_contact_normal_unit_length() {
        let mut sh = ClothSpatialHash::new(0.5);
        let positions = vec![[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]];
        sh.build(&positions);
        let contacts = sh.detect_self_collisions(&positions, 0.3);
        for (_, _, n, _) in &contacts {
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((l - 1.0).abs() < 1e-8);
        }
    }

    // ── compute_contact_impulses ──────────────────────────────────────────────

    #[test]
    fn test_impulse_separates_penetrating_vertex() {
        let contacts = vec![VertexTriangleContact {
            vertex_index: 0,
            triangle_index: 0,
            normal: [0.0, 1.0, 0.0],
            depth: 0.05,
            bary: [0.33, 0.33, 0.34],
        }];
        let velocities = vec![[0.0, -1.0, 0.0]]; // moving into surface
        let masses = vec![1.0];
        let impulses = compute_contact_impulses(&contacts, &velocities, &masses, 0.0, 0.0);
        assert_eq!(impulses.len(), 1);
        assert!(impulses[0].delta_v[1] > 0.0); // velocity should flip
    }

    #[test]
    fn test_impulse_no_response_for_separating_vertex() {
        let contacts = vec![VertexTriangleContact {
            vertex_index: 0,
            triangle_index: 0,
            normal: [0.0, 1.0, 0.0],
            depth: 0.05,
            bary: [0.33, 0.33, 0.34],
        }];
        let velocities = vec![[0.0, 1.0, 0.0]]; // moving away
        let masses = vec![1.0];
        let impulses = compute_contact_impulses(&contacts, &velocities, &masses, 0.0, 0.0);
        assert_eq!(impulses.len(), 0);
    }

    // ── edge_edge_ccd ─────────────────────────────────────────────────────────

    #[test]
    fn test_edge_ccd_detects_approaching_edges() {
        // Edge A: from (0,0,0)→(1,0,0) moving toward edge B
        // Edge B: from (0.5,1,0)→(0.5,-1,0) moving toward edge A
        let toi = edge_edge_ccd(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.02, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.02, 0.5],
            0.1,
        );
        assert!(toi.is_some());
    }

    #[test]
    fn test_edge_ccd_no_collision_when_parallel() {
        // Two parallel non-approaching edges
        let toi = edge_edge_ccd(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
            [1.0, 10.0, 0.0],
            [1.0, 10.0, 0.0],
            0.1,
        );
        assert!(toi.is_none());
    }

    #[test]
    fn test_edge_ccd_detects_fast_crossing_between_time_samples() {
        // Edge A is static along +X. Edge B (along Y at x = 0.5) drops through it
        // from z = 5.2 to z = -4.8 in one step, i.e. at 10 units per step. With a
        // contact radius of 0.01 it stays inside the shell for only 0.002 of the
        // step, which falls between the 1/64 time samples of the old sampler
        // (t = 33/64 gives z = 0.04375, t = 34/64 gives z = -0.1125).
        let toi = edge_edge_ccd(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, -0.5, 5.2],
            [0.5, -0.5, -4.8],
            [0.5, 0.5, 5.2],
            [0.5, 0.5, -4.8],
            0.01,
        )
        .expect("the edges cross at t = 0.52");
        assert!((toi - 0.52).abs() < 1e-6, "toi={toi}");
    }

    #[test]
    fn test_edge_ccd_crossing_near_the_end_points() {
        // B crosses A 2 cm from A's end point; the edge midpoints are never close.
        let toi = edge_edge_ccd(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.98, -0.5, 1.0],
            [0.98, -0.5, -1.0],
            [0.98, 0.5, 1.0],
            [0.98, 0.5, -1.0],
            0.05,
        )
        .expect("the edges cross at t = 0.5");
        assert!((toi - 0.5).abs() < 1e-6, "toi={toi}");
    }

    #[test]
    fn test_edge_ccd_misses_when_the_crossing_is_beyond_the_edge_end() {
        let toi = edge_edge_ccd(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, -0.5, 1.0],
            [1.5, -0.5, -1.0],
            [1.5, 0.5, 1.0],
            [1.5, 0.5, -1.0],
            0.05,
        );
        assert!(toi.is_none());
    }

    // ── PenaltySpring ─────────────────────────────────────────────────────────

    #[test]
    fn test_penalty_force_zero_at_zero_depth() {
        let ps = PenaltySpring::new(1000.0, 10.0);
        let f = ps.force(0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(f[0].abs() < EPS && f[1].abs() < EPS && f[2].abs() < EPS);
    }

    #[test]
    fn test_penalty_force_positive_for_penetration() {
        let ps = PenaltySpring::new(1000.0, 0.0);
        let f = ps.force(0.01, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(f[1] > 0.0);
    }

    #[test]
    fn test_penalty_force_direction_matches_normal() {
        let ps = PenaltySpring::new(500.0, 0.0);
        let normal = normalize3([1.0, 1.0, 0.0]);
        let f = ps.force(0.05, normal, [0.0, 0.0, 0.0]);
        let f_len = len3(f);
        if f_len > 1e-14 {
            let f_dir = scale3(f, 1.0 / f_len);
            assert!((f_dir[0] - normal[0]).abs() < 1e-6);
            assert!((f_dir[1] - normal[1]).abs() < 1e-6);
        }
    }

    // ── BvhAabb ────────────────────────────────────────────────────────────────

    #[test]
    fn test_bvh_aabb_overlaps() {
        let a = BvhAabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = BvhAabb::new([0.5, 0.5, 0.5], [1.5, 1.5, 1.5]);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_bvh_aabb_no_overlap() {
        let a = BvhAabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = BvhAabb::new([2.0, 2.0, 2.0], [3.0, 3.0, 3.0]);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_bvh_aabb_merge() {
        let a = BvhAabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = BvhAabb::new([2.0, 2.0, 2.0], [3.0, 3.0, 3.0]);
        let merged = a.merge(&b);
        assert_eq!(merged.min, [0.0, 0.0, 0.0]);
        assert_eq!(merged.max, [3.0, 3.0, 3.0]);
    }

    // ── SoftBodyBvh ────────────────────────────────────────────────────────────

    #[test]
    fn test_bvh_build_and_query() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [5.0, 5.0, 5.0],
            [6.0, 5.0, 5.0],
            [5.0, 6.0, 5.0],
        ];
        let triangles = vec![[0, 1, 2], [3, 4, 5]];
        let bvh = SoftBodyBvh::build(&triangles, &positions, 0.01);
        let hits = bvh.query_sphere([0.25, 0.25, 0.0], 0.5);
        assert!(!hits.is_empty());
    }

    #[test]
    fn test_bvh_query_misses_distant_triangle() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let triangles = vec![[0, 1, 2]];
        let bvh = SoftBodyBvh::build(&triangles, &positions, 0.01);
        let hits = bvh.query_sphere([100.0, 100.0, 100.0], 0.5);
        assert_eq!(hits.len(), 0);
    }

    // ── SoftContactManifold ────────────────────────────────────────────────────

    #[test]
    fn test_manifold_add_and_deepest() {
        let mut manifold = SoftContactManifold::new(4);
        manifold.add(SoftContact {
            vertex_index: 0,
            normal: [0.0, 1.0, 0.0],
            depth: 0.1,
            position: [0.0, 0.0, 0.0],
        });
        manifold.add(SoftContact {
            vertex_index: 1,
            normal: [0.0, 1.0, 0.0],
            depth: 0.5,
            position: [1.0, 0.0, 0.0],
        });
        let deepest = manifold.deepest().unwrap();
        assert_eq!(deepest.vertex_index, 1);
    }

    #[test]
    fn test_manifold_capacity_limit() {
        let mut manifold = SoftContactManifold::new(2);
        for i in 0..5 {
            manifold.add(SoftContact {
                vertex_index: i,
                normal: [0.0, 1.0, 0.0],
                depth: i as f64 * 0.1,
                position: [0.0, 0.0, 0.0],
            });
        }
        assert!(manifold.contacts.len() <= 2);
    }

    #[test]
    fn test_manifold_clear() {
        let mut manifold = SoftContactManifold::new(4);
        manifold.add(SoftContact {
            vertex_index: 0,
            normal: [0.0, 1.0, 0.0],
            depth: 0.1,
            position: [0.0, 0.0, 0.0],
        });
        manifold.clear();
        assert!(manifold.contacts.is_empty());
    }

    // ── TearableEdge / propagate_tears ─────────────────────────────────────────

    #[test]
    fn test_edge_tears_when_stress_exceeded() {
        let mut edge = TearableEdge::new(0, 1, 1.0);
        edge.accumulate_stress(1.5);
        assert!(edge.should_tear());
    }

    #[test]
    fn test_edge_does_not_tear_below_threshold() {
        let mut edge = TearableEdge::new(0, 1, 1.0);
        edge.accumulate_stress(0.5);
        assert!(!edge.should_tear());
    }

    #[test]
    fn test_propagate_tears_returns_torn_indices() {
        let mut edges = vec![TearableEdge::new(0, 1, 1.0), TearableEdge::new(1, 2, 1.0)];
        edges[0].stress = 2.0;
        edges[1].stress = 0.1;
        let torn = propagate_tears(&mut edges);
        assert_eq!(torn, vec![0]);
    }

    #[test]
    fn test_apply_collision_stress() {
        let mut edges = vec![TearableEdge::new(0, 1, 10.0), TearableEdge::new(2, 3, 10.0)];
        apply_collision_stress(&mut edges, 0, 0.5, 2.0);
        // Edge 0 shares vertex 0, should have stress 1.0
        assert!((edges[0].stress - 1.0).abs() < EPS);
        // Edge 1 does not share vertex 0
        assert!(edges[1].stress.abs() < EPS);
    }

    // ── GpuClothBuffer ─────────────────────────────────────────────────────────

    #[test]
    fn test_gpu_cloth_buffer_len() {
        let pos = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let vel = vec![[0.0; 3]; 2];
        let mass = vec![1.0, 2.0];
        let buf = GpuClothBuffer::from_vertices(&pos, &vel, &mass);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_gpu_cloth_buffer_position_roundtrip() {
        let pos = vec![[1.5, -2.5, 3.0]];
        let vel = vec![[0.0; 3]];
        let mass = vec![1.0];
        let buf = GpuClothBuffer::from_vertices(&pos, &vel, &mass);
        let p = buf.position(0);
        assert!((p[0] - 1.5).abs() < 1e-5);
        assert!((p[1] - (-2.5)).abs() < 1e-5);
        assert!((p[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_cloth_buffer_pinned_vertex() {
        let pos = vec![[0.0; 3]];
        let vel = vec![[0.0; 3]];
        let mass = vec![0.0]; // pinned
        let buf = GpuClothBuffer::from_vertices(&pos, &vel, &mass);
        assert_eq!(buf.inv_mass[0], 0.0);
    }

    // ── GpuContactBuffer ───────────────────────────────────────────────────────

    #[test]
    fn test_gpu_contact_buffer_from_contacts() {
        let contacts = vec![VertexTriangleContact {
            vertex_index: 3,
            triangle_index: 0,
            normal: [0.0, 1.0, 0.0],
            depth: 0.05,
            bary: [0.33, 0.33, 0.34],
        }];
        let buf = GpuContactBuffer::from_contacts(&contacts);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf.vertex_indices[0], 3);
        assert!((buf.depths[0] - 0.05).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_contact_buffer_empty() {
        let buf = GpuContactBuffer::new();
        assert!(buf.is_empty());
    }

    // ── GpuEdgeBuffer ──────────────────────────────────────────────────────────

    #[test]
    fn test_gpu_edge_buffer() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let buf = GpuEdgeBuffer::from_edges(&edges);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.start[0], 0);
        assert_eq!(buf.end[2], 0);
    }

    // ── batch_edge_edge_ccd ────────────────────────────────────────────────────

    #[test]
    fn test_batch_ccd_empty_edges() {
        let results = batch_edge_edge_ccd(&[], &[], &[], &[], &[], &[], 0.01, false);
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_ccd_skips_shared_vertices_within_one_mesh() {
        // Edges sharing vertex 0 of the same mesh: no collision should be reported
        let edges_a = vec![(0, 1)];
        let edges_b = vec![(0, 2)];
        let v0 = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let v1 = v0.clone();
        let results = batch_edge_edge_ccd(&edges_a, &v0, &v1, &edges_b, &v0, &v1, 0.01, true);
        assert!(results.is_empty());
    }
}
