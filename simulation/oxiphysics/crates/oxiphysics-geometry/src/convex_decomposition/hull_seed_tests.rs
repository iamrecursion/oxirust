// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the seed of [`ConvexHull3d::build`]: it must not panic
//! for fewer than four points and must not start from four coplanar points, so
//! the order of the input (for example a cube listed bottom face first) cannot
//! turn the hull into four degenerate faces.

use super::*;

/// Corners of the unit cube listed with the whole bottom face (z = 0) first, so
/// the first four points are coplanar.
fn unit_cube_bottom_face_first() -> Vec<Vec3> {
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
}

#[test]
fn unit_cube_listed_bottom_face_first_has_volume_one() {
    let points = unit_cube_bottom_face_first();
    let hull = ConvexHull3d::build(&points);
    assert!(
        (hull.volume() - 1.0).abs() < 1e-12,
        "volume {}",
        hull.volume()
    );
    assert_eq!(hull.triangles.len(), 12);
    assert!(hull.all_points_inside_or_on(&points, 1e-12));
    assert!(hull.all_points_on_boundary(&points, 1e-12));
    // Every facet looks away from the center of the cube.
    let center = [0.5, 0.5, 0.5];
    for (face, triangle) in hull.faces.iter().zip(&hull.triangles) {
        let corner = hull.vertices[triangle[0]];
        assert!(dot(face.normal, sub(corner, center)) > 0.0, "{triangle:?}");
    }
    // The same cube with the top face first, and in reverse order.
    let mut top_first = points.clone();
    top_first.rotate_left(4);
    let mut reversed = points.clone();
    reversed.reverse();
    for order in [top_first, reversed] {
        let hull = ConvexHull3d::build(&order);
        assert!(
            (hull.volume() - 1.0).abs() < 1e-12,
            "volume {}",
            hull.volume()
        );
        assert_eq!(hull.triangles.len(), 12);
    }
}

#[test]
fn convex_part_of_the_cube_is_convex_whatever_the_first_four_points() {
    let points = unit_cube_bottom_face_first();
    assert!(ConvexPart::new(points.clone()).is_convex(1e-9));
    // A buried vertex is still detected although the first four points are
    // coplanar.
    let mut buried = points;
    buried.push([0.5, 0.5, 0.5]);
    assert!(!ConvexPart::new(buried).is_convex(1e-9));
}

#[test]
fn fewer_than_four_points_have_no_hull_and_do_not_panic() {
    for count in 0..4 {
        let points: Vec<Vec3> = (0..count)
            .map(|i| [i as f64, (i * i) as f64, 0.5])
            .collect();
        let hull = ConvexHull3d::build(&points);
        assert!(hull.triangles.is_empty(), "{count} points");
        assert!(hull.faces.is_empty(), "{count} points");
        assert_eq!(hull.volume(), 0.0, "{count} points");
        assert_eq!(hull.vertices.len(), count);
        assert!(ConvexPart::new(points).is_convex(1e-9), "{count} points");
    }
}

#[test]
fn coplanar_points_before_the_apex_still_give_the_pyramid() {
    // A square base of four points followed by an apex, and the apex first.
    let base = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ];
    let apex = [1.0, 1.0, 3.0];
    let mut apex_last = base.to_vec();
    apex_last.push(apex);
    let mut apex_first = vec![apex];
    apex_first.extend_from_slice(&base);
    for order in [apex_last, apex_first] {
        let hull = ConvexHull3d::build(&order);
        // Base 2 x 2 times height 3 over 3.
        assert!(
            (hull.volume() - 4.0).abs() < 1e-12,
            "volume {}",
            hull.volume()
        );
        assert_eq!(hull.triangles.len(), 6);
        assert!(hull.all_points_inside_or_on(&order, 1e-12));
    }
}
