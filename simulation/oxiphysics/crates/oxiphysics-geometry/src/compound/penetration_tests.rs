//! Regression tests for the signed sphere-vs-child penetration depth of
//! [`CompoundShape`] and for the closest surface point of capsule children.

use super::types::CompoundShape;

const TOL: f64 = 1e-12;

fn assert_depth(result: Option<(f64, usize)>, want_depth: f64, want_child: usize) {
    let (depth, child) = result.expect("the sphere penetrates the compound");
    assert_eq!(child, want_child, "deepest child");
    assert!(
        (depth - want_depth).abs() < TOL,
        "depth {depth} expected {want_depth}"
    );
}

#[test]
fn sphere_centred_in_a_box_child_penetrates_by_depth_plus_radius() {
    // The centre is 2 below every face of the box (half extents 2); the sphere
    // of radius 0.5 has to travel 2 + 0.5 to get clear.  The unsigned distance
    // to the nearest surface point used to give +1.5, i.e. "no penetration".
    let mut cs = CompoundShape::new();
    cs.add_box([0.0; 3], [2.0, 2.0, 2.0]);
    assert_depth(cs.penetration_depth_sphere([0.0; 3], 0.5), -2.5, 0);
}

#[test]
fn off_centre_sphere_in_a_box_uses_the_nearest_face() {
    // Centre 0.5 inside the +x face of a 2x2x2 half-extent box.
    let mut cs = CompoundShape::new();
    cs.add_box([0.0; 3], [2.0, 2.0, 2.0]);
    assert_depth(cs.penetration_depth_sphere([1.5, 0.0, 0.0], 0.5), -1.0, 0);
    // Same depth from the opposite side and along another axis.
    assert_depth(cs.penetration_depth_sphere([-1.5, 0.3, 0.2], 0.5), -1.0, 0);
    assert_depth(cs.penetration_depth_sphere([0.0, 0.0, 1.5], 0.5), -1.0, 0);
}

#[test]
fn box_child_outside_distance_is_euclidean_at_the_corner() {
    // Centre (3, 3, 0) lies diagonally off the edge of the unit box: its
    // distance to the box is sqrt(2^2 + 2^2) = 2 sqrt 2, not the per-axis 2.
    let mut cs = CompoundShape::new();
    cs.add_box([0.0; 3], [1.0, 1.0, 1.0]);
    let corner_gap = 2.0 * 2.0_f64.sqrt();
    assert!(cs.penetration_depth_sphere([3.0, 3.0, 0.0], 2.5).is_none());
    assert_depth(
        cs.penetration_depth_sphere([3.0, 3.0, 0.0], 3.0),
        corner_gap - 3.0,
        0,
    );
}

#[test]
fn sphere_child_contains_the_query_centre() {
    let mut cs = CompoundShape::new();
    cs.add_sphere([1.0, 0.0, 0.0], 1.0);
    // Centre of the child: 1 below the surface.
    assert_depth(cs.penetration_depth_sphere([1.0, 0.0, 0.0], 0.25), -1.25, 0);
    // 0.4 from the child's centre: 0.6 below the surface.
    assert_depth(cs.penetration_depth_sphere([1.4, 0.0, 0.0], 0.25), -0.85, 0);
    // Outside the child and clear of it.
    assert!(cs.penetration_depth_sphere([3.0, 0.0, 0.0], 0.5).is_none());
    // Exactly touching is not a penetration.
    assert!(cs.penetration_depth_sphere([3.0, 0.0, 0.0], 1.0).is_none());
}

#[test]
fn capsule_child_contains_the_query_centre_in_cylinder_and_caps() {
    let mut cs = CompoundShape::new();
    cs.add_capsule([0.0; 3], 0.5, 1.0);
    // Inside the cylinder: 0.1 off the axis, 0.4 below the surface.
    assert_depth(cs.penetration_depth_sphere([0.1, 0.5, 0.0], 0.2), -0.6, 0);
    // Inside the upper cap: 0.2 beyond the core segment, 0.3 below the surface.
    assert_depth(cs.penetration_depth_sphere([0.0, 1.2, 0.0], 0.2), -0.5, 0);
    // On the axis: the full radius below the surface.
    assert_depth(cs.penetration_depth_sphere([0.0, 0.0, 0.0], 0.2), -0.7, 0);
    // Above the cap the distance runs along the axis, 1.0 - 0.5 = 0.5 away.
    assert!(cs.penetration_depth_sphere([0.0, 2.0, 0.0], 0.2).is_none());
    assert_depth(cs.penetration_depth_sphere([0.0, 2.0, 0.0], 0.7), -0.2, 0);
}

#[test]
fn deepest_child_wins_and_shallower_children_are_ignored() {
    let mut cs = CompoundShape::new();
    cs.add_sphere([0.0; 3], 1.0);
    cs.add_box([0.0; 3], [3.0, 3.0, 3.0]);
    cs.add_capsule([20.0, 0.0, 0.0], 0.5, 1.0);
    // The centre is 1 below the sphere's surface and 3 below the box's: the
    // box is the deeper child.  The far capsule does not touch the query.
    assert_depth(cs.penetration_depth_sphere([0.0; 3], 0.5), -3.5, 1);
}

#[test]
fn empty_compound_has_no_penetration() {
    let cs = CompoundShape::new();
    assert!(cs.penetration_depth_sphere([0.0; 3], 1.0).is_none());
}

#[test]
fn capsule_closest_point_over_the_cap_follows_the_3d_direction() {
    let mut cs = CompoundShape::new();
    cs.add_capsule([0.0; 3], 0.5, 1.0);
    // Straight above the cap the nearest surface point is the pole, not a
    // point of the equator rim.
    let (pole, child) = cs.closest_point([0.0, 5.0, 0.0]);
    assert_eq!(child, 0);
    for (got, want) in pole.iter().zip([0.0, 1.5, 0.0]) {
        assert!((got - want).abs() < TOL, "pole {pole:?}");
    }
    // Diagonally above the cap: radius 0.5 along the direction (1, 1, 0).
    let (cap, _) = cs.closest_point([1.0, 2.0, 0.0]);
    let along = 0.5 / 2.0_f64.sqrt();
    for (got, want) in cap.iter().zip([along, 1.0 + along, 0.0]) {
        assert!((got - want).abs() < TOL, "cap point {cap:?}");
    }
    // The distance to that point is the distance to the capsule surface.
    let (_, d2, _) = cs.closest_point_with_dist2([1.0, 2.0, 0.0]);
    assert!((d2.sqrt() - (2.0_f64.sqrt() - 0.5)).abs() < TOL);
    // Over the cylinder the nearest point is still the radial projection.
    let (side, _) = cs.closest_point([3.0, 0.4, 0.0]);
    for (got, want) in side.iter().zip([0.5, 0.4, 0.0]) {
        assert!((got - want).abs() < TOL, "side point {side:?}");
    }
}
