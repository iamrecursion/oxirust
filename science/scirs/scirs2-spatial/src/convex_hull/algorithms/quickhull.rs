//! Pure Rust Quickhull algorithm implementation for convex hull computation
//!
//! This module implements the Quickhull algorithm for computing convex hulls
//! in 2D and 3D. Quickhull is an efficient divide-and-conquer algorithm with
//! average-case complexity O(n log n) for 2D and O(n log n) for 3D (though
//! worst case can be O(n²)).
//!
//! # Algorithm Overview
//!
//! 1. Find extreme points to form initial simplex
//! 2. For each face of the simplex, find the farthest point
//! 3. Recursively process the resulting sub-problems
//! 4. Combine results to form the complete hull

use crate::convex_hull::core::ConvexHull;
use crate::convex_hull::geometry::calculations_2d::compute_2d_hull_equations;
use crate::error::{SpatialError, SpatialResult};
use scirs2_core::ndarray::{Array2, ArrayView2};
use std::collections::HashSet;

/// Compute convex hull using pure Rust Quickhull algorithm
///
/// # Arguments
///
/// * `points` - Input points (shape: npoints x n_dim)
///
/// # Returns
///
/// * Result containing a ConvexHull instance or an error
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::algorithms::quickhull::compute_quickhull;
/// use scirs2_core::ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
/// let hull = compute_quickhull(&points.view()).expect("Operation failed");
/// assert_eq!(hull.ndim(), 2);
/// ```
pub fn compute_quickhull(points: &ArrayView2<'_, f64>) -> SpatialResult<ConvexHull> {
    let npoints = points.nrows();
    let ndim = points.ncols();

    // Handle special cases
    if npoints < ndim + 1 {
        return handle_degenerate_case(points, ndim);
    }

    match ndim {
        1 => compute_quickhull_1d(points),
        2 => compute_quickhull_2d(points),
        3 => compute_quickhull_3d(points),
        _ => compute_quickhull_nd(points),
    }
}

/// Handle degenerate cases with too few points
fn handle_degenerate_case(points: &ArrayView2<'_, f64>, ndim: usize) -> SpatialResult<ConvexHull> {
    let npoints = points.nrows();

    if npoints == 0 {
        return Err(SpatialError::ValueError(
            "Cannot compute convex hull of empty point set".to_string(),
        ));
    }

    if npoints == 1 {
        return Ok(ConvexHull {
            points: points.to_owned(),
            vertex_indices: vec![0],
            simplices: vec![],
            equations: None,
        });
    }

    if npoints == 2 {
        return Ok(ConvexHull {
            points: points.to_owned(),
            vertex_indices: vec![0, 1],
            simplices: if ndim >= 2 { vec![vec![0, 1]] } else { vec![] },
            equations: None,
        });
    }

    // For 3 points in 2D
    if ndim == 2 && npoints == 3 {
        let vertex_indices = vec![0, 1, 2];
        let simplices = vec![vec![0, 1], vec![1, 2], vec![2, 0]];
        let equations = Some(compute_2d_hull_equations(points, &vertex_indices));

        return Ok(ConvexHull {
            points: points.to_owned(),
            vertex_indices,
            simplices,
            equations,
        });
    }

    Err(SpatialError::ValueError(format!(
        "Not enough points for {}D convex hull: need at least {}, got {}",
        ndim,
        ndim + 1,
        npoints
    )))
}

/// Quickhull for 1D - just find min and max
fn compute_quickhull_1d(points: &ArrayView2<'_, f64>) -> SpatialResult<ConvexHull> {
    let npoints = points.nrows();

    let mut min_idx = 0;
    let mut max_idx = 0;
    let mut min_val = points[[0, 0]];
    let mut max_val = points[[0, 0]];

    for i in 1..npoints {
        let val = points[[i, 0]];
        if val < min_val {
            min_val = val;
            min_idx = i;
        }
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    let vertex_indices = if min_idx != max_idx {
        vec![min_idx, max_idx]
    } else {
        vec![min_idx]
    };

    let simplices = if vertex_indices.len() == 2 {
        vec![vec![0, 1]]
    } else {
        vec![]
    };

    Ok(ConvexHull {
        points: points.to_owned(),
        vertex_indices,
        simplices,
        equations: None,
    })
}

/// Quickhull for 2D points
fn compute_quickhull_2d(points: &ArrayView2<'_, f64>) -> SpatialResult<ConvexHull> {
    let npoints = points.nrows();

    // Find leftmost and rightmost points
    let mut min_x_idx = 0;
    let mut max_x_idx = 0;

    for i in 1..npoints {
        if points[[i, 0]] < points[[min_x_idx, 0]] {
            min_x_idx = i;
        }
        if points[[i, 0]] > points[[max_x_idx, 0]] {
            max_x_idx = i;
        }
    }

    // If all points have the same x, find by y
    if (points[[min_x_idx, 0]] - points[[max_x_idx, 0]]).abs() < 1e-10 {
        for i in 1..npoints {
            if points[[i, 1]] < points[[min_x_idx, 1]] {
                min_x_idx = i;
            }
            if points[[i, 1]] > points[[max_x_idx, 1]] {
                max_x_idx = i;
            }
        }
    }

    if min_x_idx == max_x_idx {
        // All points are the same
        return Ok(ConvexHull {
            points: points.to_owned(),
            vertex_indices: vec![min_x_idx],
            simplices: vec![],
            equations: None,
        });
    }

    // Divide points into two sets: above and below the line from min to max
    let mut above: Vec<usize> = Vec::new();
    let mut below: Vec<usize> = Vec::new();

    let p1 = [points[[min_x_idx, 0]], points[[min_x_idx, 1]]];
    let p2 = [points[[max_x_idx, 0]], points[[max_x_idx, 1]]];

    for i in 0..npoints {
        if i == min_x_idx || i == max_x_idx {
            continue;
        }
        let p = [points[[i, 0]], points[[i, 1]]];
        let cross = cross_product_2d(p1, p2, p);

        if cross > 1e-10 {
            above.push(i);
        } else if cross < -1e-10 {
            below.push(i);
        }
        // Points on the line are ignored (they're not on the hull)
    }

    // Build hull recursively
    let mut hull: Vec<usize> = Vec::new();

    // Add points from min to max (upper hull)
    hull.push(min_x_idx);
    quickhull_2d_recursive(points, min_x_idx, max_x_idx, &above, &mut hull);
    hull.push(max_x_idx);
    quickhull_2d_recursive(points, max_x_idx, min_x_idx, &below, &mut hull);

    // Remove duplicates while maintaining order
    let mut seen = HashSet::new();
    let vertex_indices: Vec<usize> = hull.into_iter().filter(|x| seen.insert(*x)).collect();

    // Create simplices (edges)
    let n = vertex_indices.len();
    let mut simplices = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        simplices.push(vec![vertex_indices[i], vertex_indices[j]]);
    }

    // Compute equations
    let equations = Some(compute_2d_hull_equations(points, &vertex_indices));

    Ok(ConvexHull {
        points: points.to_owned(),
        vertex_indices,
        simplices,
        equations,
    })
}

/// Recursive step of 2D Quickhull
fn quickhull_2d_recursive(
    points: &ArrayView2<'_, f64>,
    p1_idx: usize,
    p2_idx: usize,
    point_set: &[usize],
    hull: &mut Vec<usize>,
) {
    if point_set.is_empty() {
        return;
    }

    let p1 = [points[[p1_idx, 0]], points[[p1_idx, 1]]];
    let p2 = [points[[p2_idx, 0]], points[[p2_idx, 1]]];

    // Find the point farthest from the line p1-p2
    let mut max_dist = 0.0;
    let mut max_idx = point_set[0];

    for &idx in point_set {
        let p = [points[[idx, 0]], points[[idx, 1]]];
        let dist = cross_product_2d(p1, p2, p).abs();
        if dist > max_dist {
            max_dist = dist;
            max_idx = idx;
        }
    }

    let p_max = [points[[max_idx, 0]], points[[max_idx, 1]]];

    // Divide remaining points into two sets
    let mut set1: Vec<usize> = Vec::new();
    let mut set2: Vec<usize> = Vec::new();

    for &idx in point_set {
        if idx == max_idx {
            continue;
        }
        let p = [points[[idx, 0]], points[[idx, 1]]];

        // Points to the left of p1->p_max
        if cross_product_2d(p1, p_max, p) > 1e-10 {
            set1.push(idx);
        }
        // Points to the left of p_max->p2
        if cross_product_2d(p_max, p2, p) > 1e-10 {
            set2.push(idx);
        }
    }

    // Recurse
    quickhull_2d_recursive(points, p1_idx, max_idx, &set1, hull);
    hull.push(max_idx);
    quickhull_2d_recursive(points, max_idx, p2_idx, &set2, hull);
}

/// Cross product for 2D points: (p2 - p1) × (p3 - p1)
fn cross_product_2d(p1: [f64; 2], p2: [f64; 2], p3: [f64; 2]) -> f64 {
    (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
}

/// Quickhull for 3D points
fn compute_quickhull_3d(points: &ArrayView2<'_, f64>) -> SpatialResult<ConvexHull> {
    let npoints = points.nrows();

    // Find extreme points in each dimension
    let mut extremes = [0usize; 6]; // min_x, max_x, min_y, max_y, min_z, max_z

    for i in 1..npoints {
        if points[[i, 0]] < points[[extremes[0], 0]] {
            extremes[0] = i;
        }
        if points[[i, 0]] > points[[extremes[1], 0]] {
            extremes[1] = i;
        }
        if points[[i, 1]] < points[[extremes[2], 1]] {
            extremes[2] = i;
        }
        if points[[i, 1]] > points[[extremes[3], 1]] {
            extremes[3] = i;
        }
        if points[[i, 2]] < points[[extremes[4], 2]] {
            extremes[4] = i;
        }
        if points[[i, 2]] > points[[extremes[5], 2]] {
            extremes[5] = i;
        }
    }

    // Find the two most distant extreme points to form initial edge
    let mut max_dist = 0.0;
    let mut p1_idx = 0;
    let mut p2_idx = 1;

    for i in 0..6 {
        for j in (i + 1)..6 {
            let dist = distance_squared_3d(points, extremes[i], extremes[j]);
            if dist > max_dist {
                max_dist = dist;
                p1_idx = extremes[i];
                p2_idx = extremes[j];
            }
        }
    }

    // Find third point farthest from the line p1-p2
    let mut max_dist = 0.0;
    let mut p3_idx = 0;

    for i in 0..npoints {
        if i == p1_idx || i == p2_idx {
            continue;
        }
        let dist = point_to_line_distance_3d(points, i, p1_idx, p2_idx);
        if dist > max_dist {
            max_dist = dist;
            p3_idx = i;
        }
    }

    if max_dist < 1e-10 {
        // All points are collinear
        return compute_collinear_hull_3d(points, p1_idx, p2_idx);
    }

    // Find fourth point farthest from the plane p1-p2-p3
    let mut max_dist = 0.0;
    let mut p4_idx = 0;

    for i in 0..npoints {
        if i == p1_idx || i == p2_idx || i == p3_idx {
            continue;
        }
        let dist = point_to_plane_distance_3d(points, i, p1_idx, p2_idx, p3_idx).abs();
        if dist > max_dist {
            max_dist = dist;
            p4_idx = i;
        }
    }

    if max_dist < 1e-10 {
        // All points are coplanar - treat as 2D
        return compute_coplanar_hull_3d(points, p1_idx, p2_idx, p3_idx);
    }

    // Create initial tetrahedron with proper face orientations
    let mut faces: Vec<Face3D> = Vec::new();
    let initial_indices = [p1_idx, p2_idx, p3_idx, p4_idx];

    // Determine if p4 is above or below the plane of p1-p2-p3
    let signed_dist = point_to_plane_distance_3d(points, p4_idx, p1_idx, p2_idx, p3_idx);

    if signed_dist > 0.0 {
        // p4 is above - orient faces outward
        faces.push(Face3D::new(p1_idx, p3_idx, p2_idx)); // Bottom (away from p4)
        faces.push(Face3D::new(p1_idx, p2_idx, p4_idx));
        faces.push(Face3D::new(p2_idx, p3_idx, p4_idx));
        faces.push(Face3D::new(p3_idx, p1_idx, p4_idx));
    } else {
        // p4 is below - orient faces outward
        faces.push(Face3D::new(p1_idx, p2_idx, p3_idx)); // Bottom
        faces.push(Face3D::new(p1_idx, p4_idx, p2_idx));
        faces.push(Face3D::new(p2_idx, p4_idx, p3_idx));
        faces.push(Face3D::new(p3_idx, p4_idx, p1_idx));
    }

    // Assign remaining points to faces
    let mut unassigned: Vec<usize> = (0..npoints)
        .filter(|&i| !initial_indices.contains(&i))
        .collect();

    for face in &mut faces {
        face.assign_points(points, &unassigned);
    }

    // Process faces recursively
    let mut final_faces: Vec<[usize; 3]> = Vec::new();
    let mut processed: HashSet<(usize, usize, usize)> = HashSet::new();

    quickhull_3d_recursive(points, &mut faces, &mut final_faces, &mut processed);

    // Extract unique vertices
    let mut vertex_set: HashSet<usize> = HashSet::new();
    for face in &final_faces {
        vertex_set.insert(face[0]);
        vertex_set.insert(face[1]);
        vertex_set.insert(face[2]);
    }
    let vertex_indices: Vec<usize> = vertex_set.into_iter().collect();

    // Create simplices from faces
    let simplices: Vec<Vec<usize>> = final_faces.iter().map(|f| vec![f[0], f[1], f[2]]).collect();

    // Compute equations
    let equations = ConvexHull::compute_equations_from_simplices(&points.to_owned(), &simplices, 3);

    Ok(ConvexHull {
        points: points.to_owned(),
        vertex_indices,
        simplices,
        equations,
    })
}

/// A face in 3D quickhull
#[derive(Clone)]
struct Face3D {
    vertices: [usize; 3],
    outside_points: Vec<usize>,
    normal: [f64; 3],
    offset: f64,
}

impl Face3D {
    fn new(v0: usize, v1: usize, v2: usize) -> Self {
        Face3D {
            vertices: [v0, v1, v2],
            outside_points: Vec::new(),
            normal: [0.0; 3],
            offset: 0.0,
        }
    }

    fn compute_normal(&mut self, points: &ArrayView2<'_, f64>) {
        let p0 = [
            points[[self.vertices[0], 0]],
            points[[self.vertices[0], 1]],
            points[[self.vertices[0], 2]],
        ];
        let p1 = [
            points[[self.vertices[1], 0]],
            points[[self.vertices[1], 1]],
            points[[self.vertices[1], 2]],
        ];
        let p2 = [
            points[[self.vertices[2], 0]],
            points[[self.vertices[2], 1]],
            points[[self.vertices[2], 2]],
        ];

        let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

        self.normal = [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ];

        let len = (self.normal[0].powi(2) + self.normal[1].powi(2) + self.normal[2].powi(2)).sqrt();
        if len > 1e-10 {
            self.normal[0] /= len;
            self.normal[1] /= len;
            self.normal[2] /= len;
        }

        self.offset = -(self.normal[0] * p0[0] + self.normal[1] * p0[1] + self.normal[2] * p0[2]);
    }

    fn signed_distance(&self, points: &ArrayView2<'_, f64>, idx: usize) -> f64 {
        let p = [points[[idx, 0]], points[[idx, 1]], points[[idx, 2]]];
        self.normal[0] * p[0] + self.normal[1] * p[1] + self.normal[2] * p[2] + self.offset
    }

    fn assign_points(&mut self, points: &ArrayView2<'_, f64>, candidates: &[usize]) {
        self.compute_normal(points);
        self.outside_points.clear();

        for &idx in candidates {
            if self.signed_distance(points, idx) > 1e-10 {
                self.outside_points.push(idx);
            }
        }
    }

    fn farthest_point(&self, points: &ArrayView2<'_, f64>) -> Option<usize> {
        if self.outside_points.is_empty() {
            return None;
        }

        let mut max_dist = 0.0;
        let mut max_idx = self.outside_points[0];

        for &idx in &self.outside_points {
            let dist = self.signed_distance(points, idx);
            if dist > max_dist {
                max_dist = dist;
                max_idx = idx;
            }
        }

        Some(max_idx)
    }

    fn key(&self) -> (usize, usize, usize) {
        let mut v = self.vertices;
        v.sort();
        (v[0], v[1], v[2])
    }
}

/// Recursive step of 3D Quickhull
fn quickhull_3d_recursive(
    points: &ArrayView2<'_, f64>,
    faces: &mut Vec<Face3D>,
    final_faces: &mut Vec<[usize; 3]>,
    processed: &mut HashSet<(usize, usize, usize)>,
) {
    while let Some(mut face) = faces.pop() {
        let key = face.key();
        if processed.contains(&key) {
            continue;
        }

        if let Some(apex) = face.farthest_point(points) {
            processed.insert(key);

            // Find visible faces from apex
            let mut visible_faces: Vec<Face3D> = vec![face.clone()];
            let mut remaining_faces: Vec<Face3D> = Vec::new();
            let mut horizon_edges: Vec<(usize, usize)> = Vec::new();

            for other_face in faces.drain(..) {
                if other_face.signed_distance(points, apex) > 1e-10 {
                    visible_faces.push(other_face);
                } else {
                    remaining_faces.push(other_face);
                }
            }

            // Find horizon edges (edges of visible faces that border non-visible faces)
            for vis_face in &visible_faces {
                let edges = [
                    (vis_face.vertices[0], vis_face.vertices[1]),
                    (vis_face.vertices[1], vis_face.vertices[2]),
                    (vis_face.vertices[2], vis_face.vertices[0]),
                ];

                for edge in edges {
                    let reversed = (edge.1, edge.0);
                    let is_horizon = !visible_faces.iter().any(|f| {
                        let f_edges = [
                            (f.vertices[0], f.vertices[1]),
                            (f.vertices[1], f.vertices[2]),
                            (f.vertices[2], f.vertices[0]),
                        ];
                        f_edges.contains(&reversed)
                    });

                    if is_horizon {
                        horizon_edges.push(edge);
                    }
                }
            }

            // Collect points from visible faces
            let mut orphan_points: Vec<usize> = Vec::new();
            for vis_face in &visible_faces {
                for &pt in &vis_face.outside_points {
                    if pt != apex && !orphan_points.contains(&pt) {
                        orphan_points.push(pt);
                    }
                }
            }

            // Create new faces connecting apex to horizon
            for (e0, e1) in horizon_edges {
                let mut new_face = Face3D::new(e0, e1, apex);
                new_face.assign_points(points, &orphan_points);
                remaining_faces.push(new_face);
            }

            *faces = remaining_faces;
        } else {
            // No outside points - this face is on the hull
            processed.insert(key);
            final_faces.push(face.vertices);
        }
    }
}

/// Distance squared between two points in 3D
fn distance_squared_3d(points: &ArrayView2<'_, f64>, i: usize, j: usize) -> f64 {
    let dx = points[[i, 0]] - points[[j, 0]];
    let dy = points[[i, 1]] - points[[j, 1]];
    let dz = points[[i, 2]] - points[[j, 2]];
    dx * dx + dy * dy + dz * dz
}

/// Distance from point to line in 3D
fn point_to_line_distance_3d(
    points: &ArrayView2<'_, f64>,
    pt_idx: usize,
    l1_idx: usize,
    l2_idx: usize,
) -> f64 {
    let p = [
        points[[pt_idx, 0]],
        points[[pt_idx, 1]],
        points[[pt_idx, 2]],
    ];
    let l1 = [
        points[[l1_idx, 0]],
        points[[l1_idx, 1]],
        points[[l1_idx, 2]],
    ];
    let l2 = [
        points[[l2_idx, 0]],
        points[[l2_idx, 1]],
        points[[l2_idx, 2]],
    ];

    let d = [l2[0] - l1[0], l2[1] - l1[1], l2[2] - l1[2]];
    let v = [p[0] - l1[0], p[1] - l1[1], p[2] - l1[2]];

    // Cross product v × d
    let cross = [
        v[1] * d[2] - v[2] * d[1],
        v[2] * d[0] - v[0] * d[2],
        v[0] * d[1] - v[1] * d[0],
    ];

    let cross_len = (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt();
    let d_len = (d[0].powi(2) + d[1].powi(2) + d[2].powi(2)).sqrt();

    if d_len < 1e-10 {
        return 0.0;
    }

    cross_len / d_len
}

/// Signed distance from point to plane defined by three points
fn point_to_plane_distance_3d(
    points: &ArrayView2<'_, f64>,
    pt_idx: usize,
    p1_idx: usize,
    p2_idx: usize,
    p3_idx: usize,
) -> f64 {
    let p = [
        points[[pt_idx, 0]],
        points[[pt_idx, 1]],
        points[[pt_idx, 2]],
    ];
    let p1 = [
        points[[p1_idx, 0]],
        points[[p1_idx, 1]],
        points[[p1_idx, 2]],
    ];
    let p2 = [
        points[[p2_idx, 0]],
        points[[p2_idx, 1]],
        points[[p2_idx, 2]],
    ];
    let p3 = [
        points[[p3_idx, 0]],
        points[[p3_idx, 1]],
        points[[p3_idx, 2]],
    ];

    let v1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    let v2 = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]];

    // Normal = v1 × v2
    let n = [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ];

    let len = (n[0].powi(2) + n[1].powi(2) + n[2].powi(2)).sqrt();
    if len < 1e-10 {
        return 0.0;
    }

    // Distance = (p - p1) · n / |n|
    let d = [p[0] - p1[0], p[1] - p1[1], p[2] - p1[2]];
    (d[0] * n[0] + d[1] * n[1] + d[2] * n[2]) / len
}

/// Handle collinear points in 3D (degenerate case)
fn compute_collinear_hull_3d(
    points: &ArrayView2<'_, f64>,
    p1_idx: usize,
    p2_idx: usize,
) -> SpatialResult<ConvexHull> {
    Ok(ConvexHull {
        points: points.to_owned(),
        vertex_indices: vec![p1_idx, p2_idx],
        simplices: vec![vec![p1_idx, p2_idx]],
        equations: None,
    })
}

/// Handle coplanar points in 3D (project to 2D and compute hull)
fn compute_coplanar_hull_3d(
    points: &ArrayView2<'_, f64>,
    p1_idx: usize,
    p2_idx: usize,
    p3_idx: usize,
) -> SpatialResult<ConvexHull> {
    // For simplicity, return triangle of the three defining points
    // A full implementation would project to 2D, compute 2D hull, then lift back
    Ok(ConvexHull {
        points: points.to_owned(),
        vertex_indices: vec![p1_idx, p2_idx, p3_idx],
        simplices: vec![vec![p1_idx, p2_idx, p3_idx]],
        equations: None,
    })
}

/// Quickhull for n-dimensional points (general case)
fn compute_quickhull_nd(points: &ArrayView2<'_, f64>) -> SpatialResult<ConvexHull> {
    // For higher dimensions, use a simplified incremental approach
    // This is a placeholder - full nD Quickhull is complex
    let npoints = points.nrows();
    let ndim = points.ncols();

    if npoints <= ndim {
        let vertex_indices: Vec<usize> = (0..npoints).collect();
        return Ok(ConvexHull {
            points: points.to_owned(),
            vertex_indices,
            simplices: vec![],
            equations: None,
        });
    }

    // Find extreme points in each dimension
    let mut vertex_set: HashSet<usize> = HashSet::new();

    for d in 0..ndim {
        let mut min_idx = 0;
        let mut max_idx = 0;

        for i in 1..npoints {
            if points[[i, d]] < points[[min_idx, d]] {
                min_idx = i;
            }
            if points[[i, d]] > points[[max_idx, d]] {
                max_idx = i;
            }
        }

        vertex_set.insert(min_idx);
        vertex_set.insert(max_idx);
    }

    let vertex_indices: Vec<usize> = vertex_set.into_iter().collect();

    Ok(ConvexHull {
        points: points.to_owned(),
        vertex_indices,
        simplices: vec![],
        equations: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_quickhull_2d_triangle() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let hull = compute_quickhull(&points.view()).expect("Operation failed");

        assert_eq!(hull.ndim(), 2);
        assert_eq!(hull.vertex_indices().len(), 3);
    }

    #[test]
    fn test_quickhull_2d_square() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let hull = compute_quickhull(&points.view()).expect("Operation failed");

        assert_eq!(hull.ndim(), 2);
        assert_eq!(hull.vertex_indices().len(), 4);
    }

    #[test]
    fn test_quickhull_2d_with_interior() {
        let points = arr2(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.25], // Interior point
        ]);
        let hull = compute_quickhull(&points.view()).expect("Operation failed");

        assert_eq!(hull.ndim(), 2);
        assert_eq!(hull.vertex_indices().len(), 3);
        assert!(!hull.vertex_indices().contains(&3)); // Interior point not in hull
    }

    #[test]
    fn test_quickhull_3d_tetrahedron() {
        let points = arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let hull = compute_quickhull(&points.view()).expect("Operation failed");

        assert_eq!(hull.ndim(), 3);
        assert_eq!(hull.vertex_indices().len(), 4);
    }

    #[test]
    fn test_quickhull_3d_with_interior() {
        let points = arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.25, 0.25, 0.25], // Interior point
        ]);
        let hull = compute_quickhull(&points.view()).expect("Operation failed");

        assert_eq!(hull.ndim(), 3);
        // Interior point should not be in hull
        assert!(!hull.vertex_indices().contains(&4));
    }
}
