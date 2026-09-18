//! Bowyer-Watson algorithm for 3D Delaunay triangulation

use crate::error::SpatialResult;
use scirs2_core::ndarray::Array2;

/// Bowyer-Watson algorithm for 3D Delaunay triangulation
///
/// Similar to 2D, but uses tetrahedra and circumspheres
pub(crate) fn bowyer_watson_3d(points: &Array2<f64>) -> SpatialResult<Vec<Vec<usize>>> {
    let npoints = points.nrows();

    // Find bounding box
    let mut min_coords = [f64::INFINITY; 3];
    let mut max_coords = [f64::NEG_INFINITY; 3];

    for i in 0..npoints {
        for j in 0..3 {
            min_coords[j] = min_coords[j].min(points[[i, j]]);
            max_coords[j] = max_coords[j].max(points[[i, j]]);
        }
    }

    // Create super-tetrahedron that contains all points
    let dx = max_coords[0] - min_coords[0];
    let dy = max_coords[1] - min_coords[1];
    let dz = max_coords[2] - min_coords[2];
    let delta_max = dx.max(dy).max(dz);
    let mid = [
        (min_coords[0] + max_coords[0]) / 2.0,
        (min_coords[1] + max_coords[1]) / 2.0,
        (min_coords[2] + max_coords[2]) / 2.0,
    ];

    // Super-tetrahedron vertices (with large margin)
    let margin = 100.0;
    let super_p0 = [
        mid[0] - margin * delta_max,
        mid[1] - margin * delta_max,
        mid[2] - margin * delta_max,
    ];
    let super_p1 = [
        mid[0] + margin * delta_max,
        mid[1] - margin * delta_max,
        mid[2] - margin * delta_max,
    ];
    let super_p2 = [
        mid[0],
        mid[1] + margin * delta_max,
        mid[2] - margin * delta_max,
    ];
    let super_p3 = [mid[0], mid[1], mid[2] + margin * delta_max];

    // Extended point list
    let mut all_points: Vec<[f64; 3]> = Vec::with_capacity(npoints + 4);
    for i in 0..npoints {
        all_points.push([points[[i, 0]], points[[i, 1]], points[[i, 2]]]);
    }
    all_points.push(super_p0);
    all_points.push(super_p1);
    all_points.push(super_p2);
    all_points.push(super_p3);

    // Initialize with super-tetrahedron
    let mut tetrahedra: Vec<[usize; 4]> = vec![[npoints, npoints + 1, npoints + 2, npoints + 3]];

    // Add points one at a time
    for point_idx in 0..npoints {
        let point = all_points[point_idx];

        // Find all tetrahedra whose circumsphere contains this point
        let mut bad_tetrahedra: Vec<usize> = Vec::new();
        for (tet_idx, tet) in tetrahedra.iter().enumerate() {
            let p0 = all_points[tet[0]];
            let p1 = all_points[tet[1]];
            let p2 = all_points[tet[2]];
            let p3 = all_points[tet[3]];

            if point_in_circumsphere_3d(point, p0, p1, p2, p3) {
                bad_tetrahedra.push(tet_idx);
            }
        }

        // Find the boundary faces of the cavity
        let mut boundary_faces: Vec<[usize; 3]> = Vec::new();
        for &tet_idx in &bad_tetrahedra {
            let tet = tetrahedra[tet_idx];
            // Each tetrahedron has 4 triangular faces
            let faces = [
                [tet[0], tet[1], tet[2]],
                [tet[0], tet[1], tet[3]],
                [tet[0], tet[2], tet[3]],
                [tet[1], tet[2], tet[3]],
            ];

            for face in &faces {
                // Check if this face is shared with another bad tetrahedron
                let mut is_shared = false;
                for &other_idx in &bad_tetrahedra {
                    if other_idx == tet_idx {
                        continue;
                    }
                    if tetrahedron_has_face(&tetrahedra[other_idx], face) {
                        is_shared = true;
                        break;
                    }
                }

                if !is_shared {
                    boundary_faces.push(*face);
                }
            }
        }

        // Remove bad tetrahedra (in reverse order to maintain indices)
        bad_tetrahedra.sort_by(|a, b| b.cmp(a));
        for tet_idx in bad_tetrahedra {
            tetrahedra.remove(tet_idx);
        }

        // Create new tetrahedra from boundary faces to the new point
        for face in boundary_faces {
            tetrahedra.push([point_idx, face[0], face[1], face[2]]);
        }
    }

    // Remove tetrahedra that have a vertex from the super-tetrahedron
    let super_vertices = [npoints, npoints + 1, npoints + 2, npoints + 3];
    tetrahedra.retain(|tet| !tet.iter().any(|&v| super_vertices.contains(&v)));

    // Convert to Vec<Vec<usize>>
    let simplices: Vec<Vec<usize>> = tetrahedra.iter().map(|t| t.to_vec()).collect();

    Ok(simplices)
}

/// Check if a tetrahedron has a specific triangular face
fn tetrahedron_has_face(tet: &[usize; 4], face: &[usize; 3]) -> bool {
    let faces = [
        [tet[0], tet[1], tet[2]],
        [tet[0], tet[1], tet[3]],
        [tet[0], tet[2], tet[3]],
        [tet[1], tet[2], tet[3]],
    ];

    let mut sorted_face: Vec<usize> = face.to_vec();
    sorted_face.sort();

    for f in &faces {
        let mut sorted_f: Vec<usize> = f.to_vec();
        sorted_f.sort();
        if sorted_f == sorted_face {
            return true;
        }
    }
    false
}

/// Check if a point is inside the circumsphere of a tetrahedron
fn point_in_circumsphere_3d(
    p: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    d: [f64; 3],
) -> bool {
    // Using the 5x5 determinant method
    let ax = a[0] - p[0];
    let ay = a[1] - p[1];
    let az = a[2] - p[2];
    let bx = b[0] - p[0];
    let by = b[1] - p[1];
    let bz = b[2] - p[2];
    let cx = c[0] - p[0];
    let cy = c[1] - p[1];
    let cz = c[2] - p[2];
    let dx = d[0] - p[0];
    let dy = d[1] - p[1];
    let dz = d[2] - p[2];

    let as_ = ax * ax + ay * ay + az * az;
    let bs = bx * bx + by * by + bz * bz;
    let cs = cx * cx + cy * cy + cz * cz;
    let ds = dx * dx + dy * dy + dz * dz;

    // 4x4 determinant for in-sphere test
    let det = as_ * det3x3(bx, by, bz, cx, cy, cz, dx, dy, dz)
        - bs * det3x3(ax, ay, az, cx, cy, cz, dx, dy, dz)
        + cs * det3x3(ax, ay, az, bx, by, bz, dx, dy, dz)
        - ds * det3x3(ax, ay, az, bx, by, bz, cx, cy, cz);

    // Check orientation of tetrahedron
    let orientation = det3x3(
        b[0] - a[0],
        b[1] - a[1],
        b[2] - a[2],
        c[0] - a[0],
        c[1] - a[1],
        c[2] - a[2],
        d[0] - a[0],
        d[1] - a[1],
        d[2] - a[2],
    );

    if orientation > 0.0 {
        det > 0.0
    } else {
        det < 0.0
    }
}

/// Compute 3x3 determinant
fn det3x3(
    a11: f64,
    a12: f64,
    a13: f64,
    a21: f64,
    a22: f64,
    a23: f64,
    a31: f64,
    a32: f64,
    a33: f64,
) -> f64 {
    a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)
}
