//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::borrow::Cow;
use std::sync::OnceLock;

use oxiphysics_core::math::{Real, Vec3};

use super::functions::*;
use super::quickhull_core::{HullTolerances, build_hull, cloud_extent};
use super::solid::HullSolid;
use super::topology::HullTopology;

/// A convex hull defined by a set of vertices (uses nalgebra `Vec3`).
///
/// The [`Shape`](crate::shape::Shape) implementation treats the hull as the
/// uniform solid spanned by `vertices`: volume, centre of mass and inertia
/// tensor come from exact polyhedral integration over the hull facets, and
/// ray casts intersect the facets.  The facets are computed lazily on first
/// use and cached.  Vertex sets that span no volume (fewer than four points,
/// or all points coplanar or collinear, or a non-finite coordinate) have zero
/// volume and inertia, their centre of mass is the vertex mean and ray casts
/// never hit.
///
/// The polygon faces and edges that contact generation needs are available from
/// [`topology`](Self::topology); they are derived from `vertices` once and cached
/// on the hull.
#[derive(Clone)]
pub struct ConvexHull {
    /// Vertices of the convex hull.
    pub vertices: Vec<Vec3>,
    /// Facets and mass properties derived from `vertices`.
    solid: OnceLock<SolidCache>,
    /// Polygon faces and edges derived from `vertices`.
    pub(super) topology: OnceLock<TopologyCache>,
}

/// Derived polyhedron together with the vertex list it was built from.
///
/// `vertices` is a public field, so the cache remembers its source and is
/// ignored if the vertices were changed after it was filled.
#[derive(Clone)]
struct SolidCache {
    source: Vec<Vec3>,
    solid: Option<HullSolid>,
}

/// Derived topology together with the vertex list it was built from.
///
/// Like [`SolidCache`], it is ignored if the public `vertices` were changed after
/// it was filled.  A hull that spans no volume is cached too (`topology` is
/// `None`), so it does not run the hull builder again on every query.
#[derive(Clone)]
pub(super) struct TopologyCache {
    pub(super) source: Vec<Vec3>,
    pub(super) topology: Option<HullTopology>,
}

impl TopologyCache {
    /// Build the topology of `vertices` and remember them as its source.
    fn build(vertices: &[Vec3]) -> Self {
        Self {
            source: vertices.to_vec(),
            topology: topology_of(vertices),
        }
    }
}

/// Topology of the hull spanned by `vertices` (`None` if they span no volume).
fn topology_of(vertices: &[Vec3]) -> Option<HullTopology> {
    let points: Vec<[f64; 3]> = vertices.iter().map(|v| [v.x, v.y, v.z]).collect();
    HullTopology::build(&points)
}

impl std::fmt::Debug for ConvexHull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvexHull")
            .field("vertices", &self.vertices)
            .finish_non_exhaustive()
    }
}

impl ConvexHull {
    /// Create a new convex hull from vertices.
    pub fn new(vertices: Vec<Vec3>) -> Self {
        Self {
            vertices,
            solid: OnceLock::new(),
            topology: OnceLock::new(),
        }
    }
    /// Polygon faces and edges of the hull spanned by the current `vertices`,
    /// in the local frame of the hull (`None` if they span no volume, or if the
    /// hull builder returned an inconsistent surface, see [`HullTopology::build`]).
    ///
    /// The topology is derived on the first call and cached, so repeated queries
    /// (every contact query of a hull in the narrow phase) borrow it for the cost
    /// of an `O(n)` comparison of `vertices` with the vertices it was built from.
    /// `vertices` is a public field: if it was changed after the cache was
    /// filled, the topology is rebuilt (and returned owned) on every call, so
    /// build a new hull with [`ConvexHull::new`] after editing the vertices.
    pub fn topology(&self) -> Option<Cow<'_, HullTopology>> {
        let cache = self
            .topology
            .get_or_init(|| TopologyCache::build(&self.vertices));
        if cache.source == self.vertices {
            cache.topology.as_ref().map(Cow::Borrowed)
        } else {
            topology_of(&self.vertices).map(Cow::Owned)
        }
    }
    /// Run `f` on the polyhedron spanned by the current `vertices`
    /// (`None` if they span no volume).
    pub(super) fn with_solid<R>(&self, f: impl FnOnce(Option<&HullSolid>) -> R) -> R {
        let cache = self.solid.get_or_init(|| SolidCache {
            source: self.vertices.clone(),
            solid: HullSolid::build(&self.vertices),
        });
        if cache.source == self.vertices {
            f(cache.solid.as_ref())
        } else {
            f(HullSolid::build(&self.vertices).as_ref())
        }
    }
    /// Mean of the vertices (zero for an empty hull).
    pub(super) fn centroid(&self) -> Vec3 {
        if self.vertices.is_empty() {
            return Vec3::zeros();
        }
        let sum: Vec3 = self.vertices.iter().sum();
        sum / self.vertices.len() as Real
    }
    /// Find the support point (vertex with maximum dot product in the given direction).
    pub fn support_point(&self, direction: &Vec3) -> Vec3 {
        let mut best = Vec3::zeros();
        let mut best_dot = f64::NEG_INFINITY;
        for v in &self.vertices {
            let d = v.dot(direction);
            if d > best_dot {
                best_dot = d;
                best = *v;
            }
        }
        best
    }
    /// Compute the axis-aligned bounding box.
    pub fn bounding_box(&self) -> ConvexHullAabb {
        let mut min = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for v in &self.vertices {
            for k in 0..3 {
                if v[k] < min[k] {
                    min[k] = v[k];
                }
                if v[k] > max[k] {
                    max[k] = v[k];
                }
            }
        }
        ConvexHullAabb { min, max }
    }
}
/// Axis-aligned bounding box for ConvexHull.
#[derive(Debug, Clone)]
pub struct ConvexHullAabb {
    /// Minimum corner.
    pub min: Vec3,
    /// Maximum corner.
    pub max: Vec3,
}
/// 3-D convex hull computed via the Quickhull algorithm.
///
/// Uses plain `[f64; 3]` arrays — no nalgebra dependency.
#[derive(Debug, Clone)]
pub struct ConvexHull3D {
    /// Unique vertices of the hull (subset of the input point cloud).
    pub vertices: Vec<[f64; 3]>,
    /// Triangular faces as triples of indices into `vertices` (CCW outward winding).
    pub faces: Vec<[usize; 3]>,
}
impl ConvexHull3D {
    /// Build a convex hull from a point cloud using the Quickhull algorithm.
    ///
    /// Returns `None` if the input has fewer than 4 non-coplanar points or if
    /// any coordinate is not finite.  The degeneracy and visibility tolerances
    /// scale with the extent of the cloud, so a cloud and a scaled copy of it
    /// yield the same hull.
    ///
    /// The hull grows from a seed tetrahedron of extreme points by repeatedly
    /// adding the point farthest in front of some face; each face keeps the
    /// points in front of it (its outside set), the faces a new point sees are
    /// found over the face adjacency and replaced by a fan of faces from their
    /// horizon, and the points of the removed faces are handed on to the new
    /// faces.  Points that are in front of no face are never revisited, so the
    /// expected cost is close to `O(n log n)` for `n` points instead of
    /// testing every point against every face.  A point is added only if it is
    /// more than `1e-10` times the longest side of the bounding box in front of
    /// the plane of some face; points closer to the hull than that (the noise of
    /// a flat face, rounding error) may be left out, which keeps the surface free
    /// of faces between nearly coplanar or collinear points.  A face is removed
    /// for an added point when the point is in front of it by more than a
    /// hundredth of that distance, and also when keeping it would leave an edge
    /// that is reflex by more than the tolerance; the rounded planes of thin
    /// faces are not trusted and the exact orientation predicate decides for them
    /// (see the module documentation of the Quickhull core).  For input whose
    /// noise is of the size of the tolerance itself the surface stays closed and
    /// consistently wound but can be slightly non-convex.
    ///
    /// The result lists the input points that are corners of a face, in input
    /// order, and the triangles of the surface (counter-clockwise seen from
    /// outside, closed and consistently wound).  Every corner of the hull is a
    /// vertex.  The surface is a triangulation: for input with many coplanar or
    /// collinear points a point inside a flat face or on an edge may be kept as a
    /// vertex, and where a flat face has more than three corners its
    /// triangulation depends on the order of the construction.
    pub fn build(points: &[[f64; 3]]) -> Option<ConvexHull3D> {
        let extent = cloud_extent(points)?.longest_side;
        // Tolerances relative to the size of the cloud: the seed tetrahedron
        // needs a vertex `seed_distance` off the seed line and plane, and a
        // point is in front of a face if it is more than `visible_distance`
        // above its plane (a geometric distance, so thin sliver faces are
        // judged like fat ones).
        let tolerances = HullTolerances {
            visible_distance: 1e-10 * extent,
            seed_distance: 1e-12 * extent,
        };
        let surface = build_hull(points, tolerances)?;
        Some(ConvexHull3D {
            vertices: surface.vertices,
            faces: surface.faces,
        })
    }
    /// Volume via signed tetrahedral decomposition from the centroid.
    pub fn volume(&self) -> f64 {
        if self.faces.is_empty() || self.vertices.is_empty() {
            return 0.0;
        }
        let c = self.centroid();
        let mut vol = 0.0f64;
        for face in &self.faces {
            let a = sub(self.vertices[face[0]], c);
            let b = sub(self.vertices[face[1]], c);
            let cc = sub(self.vertices[face[2]], c);
            vol += dot(a, cross(b, cc));
        }
        (vol / 6.0).abs()
    }
    /// Surface area as the sum of triangle areas.
    pub fn surface_area(&self) -> f64 {
        let mut area = 0.0f64;
        for face in &self.faces {
            let n = face_normal(
                self.vertices[face[0]],
                self.vertices[face[1]],
                self.vertices[face[2]],
            );
            area += dot(n, n).sqrt() * 0.5;
        }
        area
    }
    /// Returns `true` if `p` is inside (or on the boundary of) the convex hull.
    ///
    /// A point is inside iff it is on the non-positive side of every face plane.
    pub fn contains_point(&self, p: [f64; 3]) -> bool {
        for face in &self.faces {
            let n = face_normal(
                self.vertices[face[0]],
                self.vertices[face[1]],
                self.vertices[face[2]],
            );
            if face_plane_dist(n, self.vertices[face[0]], p) > 1e-9 {
                return false;
            }
        }
        true
    }
    /// Support function: returns the furthest vertex in direction `dir` (for GJK).
    pub fn support(&self, dir: [f64; 3]) -> [f64; 3] {
        self.vertices
            .iter()
            .copied()
            .max_by(|a, b| {
                dot(*a, dir)
                    .partial_cmp(&dot(*b, dir))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or([0.0, 0.0, 0.0])
    }
    /// Number of hull vertices.
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }
    /// Number of triangular faces.
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }
    fn centroid(&self) -> [f64; 3] {
        let n = self.vertices.len() as f64;
        let mut s = [0.0f64; 3];
        for v in &self.vertices {
            s = add(s, *v);
        }
        scale(s, 1.0 / n)
    }
}
impl ConvexHull3D {
    /// Compute the volume of the convex hull via the divergence theorem.
    ///
    /// Each triangular face contributes a signed tetrahedral volume measured from
    /// the vertex mean, which lies inside the hull, so all contributions have the
    /// sign of the facet orientation and nothing cancels however far the hull is
    /// from the origin.  The absolute value gives the total enclosed volume.
    pub fn compute_volume(&self) -> f64 {
        self.volume()
    }
    /// Compute the center of mass (centroid) of the solid convex hull.
    ///
    /// Uses the signed-tetrahedral decomposition: each triangle + the vertex mean
    /// (an interior reference point) forms a tetrahedron whose centroid is
    /// `(a + b + c + reference) / 4`.  The weighted average of the tetrahedron
    /// centroids (by signed volume) is the COM; it does not depend on the facet
    /// orientation, and measuring from an interior point keeps the sums free of
    /// cancellation for hulls far from the origin.
    ///
    /// Returns `[0.0; 3]` for an empty hull and the vertex mean for a hull
    /// without volume.
    pub fn compute_center_of_mass(&self) -> [f64; 3] {
        if self.faces.is_empty() || self.vertices.is_empty() {
            return [0.0; 3];
        }
        let reference = self.centroid();
        let mut total_vol = 0.0f64;
        let mut weighted = [0.0f64; 3];
        for face in &self.faces {
            let a = sub(self.vertices[face[0]], reference);
            let b = sub(self.vertices[face[1]], reference);
            let c = sub(self.vertices[face[2]], reference);
            let signed_vol = dot(a, cross(b, c)) / 6.0;
            // Centroid of the tetrahedron (reference, a, b, c), relative to `reference`.
            let tet_com = scale(add(add(a, b), c), 0.25);
            total_vol += signed_vol;
            weighted = add(weighted, scale(tet_com, signed_vol));
        }
        if total_vol.abs() > 1e-24 {
            add(reference, scale(weighted, 1.0 / total_vol))
        } else {
            reference
        }
    }
    /// Compute the 3×3 inertia tensor of the solid convex hull, assuming
    /// uniform density with total `mass`.
    ///
    /// The tensor is returned in row-major order as `[[f64; 3\]; 3]`.
    /// Off-diagonal terms (products of inertia) are included.  It is taken about
    /// the centre of mass and does not depend on whether the facets wind outward
    /// or inward.  A non-positive `mass` or a hull without volume gives zeros.
    ///
    /// Uses the tetrahedral decomposition method from Polyhedral Mass Properties
    /// (David Eberly, Geometric Tools).
    pub fn compute_moment_of_inertia(&self, mass: f64) -> [[f64; 3]; 3] {
        if self.faces.is_empty() || mass <= 0.0 {
            return [[0.0; 3]; 3];
        }
        let com = self.compute_center_of_mass();
        let mut ixx = 0.0f64;
        let mut iyy = 0.0f64;
        let mut izz = 0.0f64;
        let mut ixy = 0.0f64;
        let mut ixz = 0.0f64;
        let mut iyz = 0.0f64;
        let mut total_vol = 0.0f64;
        for face in &self.faces {
            let a = sub(self.vertices[face[0]], com);
            let b = sub(self.vertices[face[1]], com);
            let c = sub(self.vertices[face[2]], com);
            let signed_vol = dot(a, cross(b, c)) / 6.0;
            total_vol += signed_vol;
            let ax = a[0];
            let ay = a[1];
            let az = a[2];
            let bx = b[0];
            let by = b[1];
            let bz = b[2];
            let cx = c[0];
            let cy = c[1];
            let cz = c[2];
            let f2x = ax * ax + bx * bx + cx * cx + ax * bx + ax * cx + bx * cx;
            let f2y = ay * ay + by * by + cy * cy + ay * by + ay * cy + by * cy;
            let f2z = az * az + bz * bz + cz * cz + az * bz + az * cz + bz * cz;
            let fxy = ax * ay
                + bx * by
                + cx * cy
                + 0.5 * (ax * by + ay * bx + ax * cy + ay * cx + bx * cy + by * cx);
            let fxz = ax * az
                + bx * bz
                + cx * cz
                + 0.5 * (ax * bz + az * bx + ax * cz + az * cx + bx * cz + bz * cx);
            let fyz = ay * az
                + by * bz
                + cy * cz
                + 0.5 * (ay * bz + az * by + ay * cz + az * cy + by * cz + bz * cy);
            let w = signed_vol / 10.0;
            ixx += w * (f2y + f2z);
            iyy += w * (f2x + f2z);
            izz += w * (f2x + f2y);
            ixy -= w * fxy;
            ixz -= w * fxz;
            iyz -= w * fyz;
        }
        // Every face sum above carries the sign of the facet orientation, so
        // dividing by the *signed* volume cancels it: the tensor is the same
        // whether the facets wind outward or inward.
        if total_vol.abs() < 1e-24 {
            return [[0.0; 3]; 3];
        }
        let density = mass / total_vol;
        [
            [density * ixx, density * ixy, density * ixz],
            [density * ixy, density * iyy, density * iyz],
            [density * ixz, density * iyz, density * izz],
        ]
    }
}
impl ConvexHull3D {
    /// Compute the centroid (arithmetic mean) of hull vertices.
    pub fn vertex_centroid(&self) -> [f64; 3] {
        self.centroid()
    }
    /// Compute a signed-tetrahedral volume estimate and centroid together.
    ///
    /// More accurate than the vertex centroid for volume purposes.
    /// Returns `(volume, centroid)`.
    pub fn volume_centroid(&self) -> (f64, [f64; 3]) {
        if self.faces.is_empty() {
            return (0.0, self.centroid());
        }
        let c = self.centroid();
        let mut total_vol = 0.0f64;
        let mut weighted = [0.0f64; 3];
        for face in &self.faces {
            let a = sub(self.vertices[face[0]], c);
            let b = sub(self.vertices[face[1]], c);
            let cc = sub(self.vertices[face[2]], c);
            let signed_vol = dot(a, cross(b, cc)) / 6.0;
            let tet_c = scale(
                add(
                    add(add(c, self.vertices[face[0]]), self.vertices[face[1]]),
                    self.vertices[face[2]],
                ),
                0.25,
            );
            total_vol += signed_vol;
            weighted = add(weighted, scale(tet_c, signed_vol));
        }
        if total_vol.abs() > 1e-24 {
            (total_vol.abs(), scale(weighted, 1.0 / total_vol))
        } else {
            (0.0, c)
        }
    }
}
/// Build an approximate convex hull after optionally deduplicating and
/// thinning the input, and return it together with quality metrics.
pub struct HullWithQuality {
    /// The convex hull.
    pub hull: ConvexHull3D,
    /// Sphericity ∈ (0, 1].
    pub sphericity: f64,
    /// Aspect ratio (1 = cube-like, ∞ = degenerate).
    pub aspect_ratio: f64,
}
impl HullWithQuality {
    /// Build the hull and compute quality metrics in one step.
    pub fn build(points: &[[f64; 3]], tolerance: f64) -> Option<Self> {
        let unique = deduplicate_points(points, tolerance);
        let hull = ConvexHull3D::build(&unique)?;
        let sphericity = hull_sphericity(&hull);
        let aspect_ratio = hull_aspect_ratio(&hull);
        Some(HullWithQuality {
            hull,
            sphericity,
            aspect_ratio,
        })
    }
}
/// Incremental 3D convex hull builder.
///
/// Allows adding points one at a time and maintains a valid convex hull at all
/// times.  Based on the Quickhull increment approach: start with a seed
/// tetrahedron and add points using the existing `ConvexHull3D::build` logic
/// incrementally by maintaining a mutable point cloud and rebuilding.
///
/// For large datasets, prefer `ConvexHull3D::build` directly.  This builder is
/// useful when points arrive in a stream.
pub struct IncrementalConvexHull {
    /// All points added so far.
    pub(super) points: Vec<[f64; 3]>,
    /// Current best hull (if at least 4 non-coplanar points exist).
    pub(super) hull: Option<ConvexHull3D>,
}
impl IncrementalConvexHull {
    /// Create an empty incremental hull builder.
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            hull: None,
        }
    }
    /// Add a single point and update the convex hull.
    ///
    /// If the point is already inside the current hull, the hull is unchanged.
    /// Otherwise, the hull is rebuilt from all points.
    pub fn add_point(&mut self, p: [f64; 3]) {
        if let Some(h) = &self.hull
            && h.contains_point(p)
        {
            return;
        }
        self.points.push(p);
        self.hull = ConvexHull3D::build(&self.points);
    }
    /// Add multiple points at once.
    pub fn add_points(&mut self, pts: &[[f64; 3]]) {
        for &p in pts {
            self.add_point(p);
        }
    }
    /// Get a reference to the current hull.
    pub fn hull(&self) -> Option<&ConvexHull3D> {
        self.hull.as_ref()
    }
    /// Number of points added so far.
    pub fn n_points(&self) -> usize {
        self.points.len()
    }
    /// Remove all points and reset.
    pub fn clear(&mut self) {
        self.points.clear();
        self.hull = None;
    }
    /// Compute the current volume, or 0 if not yet a valid hull.
    pub fn volume(&self) -> f64 {
        self.hull.as_ref().map(|h| h.volume()).unwrap_or(0.0)
    }
}
