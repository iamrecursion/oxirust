use approx::assert_relative_eq;
use scirs2_core::ndarray::array;
use std::f64::consts::PI;

use scirs2_spatial::transform::{Rotation, RotationSpline, Slerp};

#[test]
#[allow(dead_code)]
fn test_rotation_basic() {
    // Create a rotation from various representations
    let rot_identity = Rotation::identity();

    // Apply to a test point
    let point_arr = array![1.0, 2.0, 3.0];
    let point = point_arr.view();
    let rotated = rot_identity.apply(&point).expect("Operation failed");

    // Identity rotation should return the same point
    assert_relative_eq!(rotated[0], point[0], epsilon = 1e-10);
    assert_relative_eq!(rotated[1], point[1], epsilon = 1e-10);
    assert_relative_eq!(rotated[2], point[2], epsilon = 1e-10);

    // Create a 90-degree rotation around Z
    let euler_z_arr = array![0.0, 0.0, PI / 2.0];
    let euler_z = euler_z_arr.view();
    let rot_z = Rotation::from_euler(&euler_z, "xyz").expect("Operation failed");
    let p_x_arr = array![1.0, 0.0, 0.0];
    let p_x = p_x_arr.view();
    let rotated_z = rot_z.apply(&p_x).expect("Operation failed");

    // Should map [1, 0, 0] to approximately [0, 1, 0]
    assert_relative_eq!(rotated_z[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_z[1], 1.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_z[2], 0.0, epsilon = 1e-10);

    // Test inverse rotation
    let rot_z_inv = rot_z.inv();
    let rotated_z_view = rotated_z.view();
    let point_back = rot_z_inv.apply(&rotated_z_view).expect("Operation failed");

    // Should get back original point [1, 0, 0]
    assert_relative_eq!(point_back[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(point_back[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(point_back[2], 0.0, epsilon = 1e-10);
}

#[test]
#[allow(dead_code)]
fn test_rotation_euler_conventions() {
    let test_conventions = ["xyz", "zyx", "xyx", "xzx", "yxy", "yzy", "zxz", "zyz"];

    // Test each Euler angle convention
    for &convention in &test_conventions {
        // Create a simple rotation using this convention
        let angles_arr = match convention {
            "xyz" | "zyx" => array![PI / 4.0, 0.0, 0.0], // 45 degrees around X
            "xyx" | "xzx" => array![0.0, PI / 4.0, 0.0], // 45 degrees around Y (middle axis)
            "yxy" | "yzy" => array![PI / 4.0, 0.0, 0.0], // 45 degrees around Y (first axis)
            "zxz" | "zyz" => array![0.0, PI / 4.0, 0.0], // 45 degrees around Z (middle axis)
            _ => unreachable!(),
        };
        let angles = angles_arr.view();

        let rotation = Rotation::from_euler(&angles, convention).expect("Operation failed");

        // Get back the Euler angles
        let angles_back = rotation.as_euler(convention).expect("Operation failed");
        let angles_back_view = angles_back.view();

        // For simple rotations, the angles should be recoverable
        // (allowing for different but equivalent representations)
        let point_arr = array![1.0, 1.0, 1.0];
        let point = point_arr.view();
        let rotated1 = rotation.apply(&point).expect("Operation failed");

        let rotation2 =
            Rotation::from_euler(&angles_back_view, convention).expect("Operation failed");
        let rotated2 = rotation2.apply(&point).expect("Operation failed");

        // Both rotations should produce the same result
        assert_relative_eq!(rotated1[0], rotated2[0], epsilon = 1e-10);
        assert_relative_eq!(rotated1[1], rotated2[1], epsilon = 1e-10);
        assert_relative_eq!(rotated1[2], rotated2[2], epsilon = 1e-10);
    }
}

#[test]
#[allow(dead_code)]
fn test_slerp_basic() {
    // Create two rotations
    let rot1 = Rotation::identity();
    let euler_pi_arr = array![0.0, 0.0, PI];
    let euler_pi = euler_pi_arr.view();
    let rot2 = Rotation::from_euler(&euler_pi, "xyz").expect("Operation failed"); // 180 degrees around Z

    // Create a slerp interpolator
    let slerp = Slerp::new(rot1, rot2).expect("Operation failed");

    // Test interpolation at various parameters
    let test_point_arr = array![1.0, 0.0, 0.0];
    let test_point = test_point_arr.view();

    // Test t=0 and t=1 (endpoints)
    let rot_0 = slerp.interpolate(0.0);
    let rot_1 = slerp.interpolate(1.0);

    let rotated_0 = rot_0.apply(&test_point).expect("Operation failed");
    let rotated_1 = rot_1.apply(&test_point).expect("Operation failed");

    // Should match the original rotations
    assert_relative_eq!(rotated_0[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_0[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_0[2], 0.0, epsilon = 1e-10);

    assert_relative_eq!(rotated_1[0], -1.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_1[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_1[2], 0.0, epsilon = 1e-10);

    // Test midpoint (should be 90 degrees around Z)
    let rot_half = slerp.interpolate(0.5);
    let rotated_half = rot_half.apply(&test_point).expect("Operation failed");

    assert_relative_eq!(rotated_half[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_half[1], 1.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_half[2], 0.0, epsilon = 1e-10);
}

#[test]
#[allow(dead_code)]
fn test_rotation_spline_slerp() {
    // Create a rotation spline
    let rotations = vec![
        Rotation::identity(),
        {
            let euler_90_arr = array![0.0, 0.0, PI / 2.0];
            let euler_90 = euler_90_arr.view();
            Rotation::from_euler(&euler_90, "xyz").expect("Operation failed")
        },
        {
            let euler_180_arr = array![0.0, 0.0, PI];
            let euler_180 = euler_180_arr.view();
            Rotation::from_euler(&euler_180, "xyz").expect("Operation failed")
        },
    ];
    let times = vec![0.0, 1.0, 2.0];

    let spline = RotationSpline::new(&rotations, &times).expect("Operation failed");

    // Test that the default interpolation type is "slerp"
    assert_eq!(spline.interpolation_type(), "slerp");

    // Test interpolation at key times
    let test_point_arr = array![1.0, 0.0, 0.0];
    let test_point = test_point_arr.view();

    let rot_0 = spline.interpolate(0.0);
    let rot_1 = spline.interpolate(1.0);
    let rot_2 = spline.interpolate(2.0);

    let rotated_0 = rot_0.apply(&test_point).expect("Operation failed");
    let rotated_1 = rot_1.apply(&test_point).expect("Operation failed");
    let rotated_2 = rot_2.apply(&test_point).expect("Operation failed");

    // Check that they match the original rotations
    assert_relative_eq!(rotated_0[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_0[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_0[2], 0.0, epsilon = 1e-10);

    assert_relative_eq!(rotated_1[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_1[1], 1.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_1[2], 0.0, epsilon = 1e-10);

    assert_relative_eq!(rotated_2[0], -1.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_2[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_2[2], 0.0, epsilon = 1e-10);

    // Test interpolation at midpoints
    let rot_05 = spline.interpolate(0.5);
    let rot_15 = spline.interpolate(1.5);

    let rotated_05 = rot_05.apply(&test_point).expect("Operation failed");
    let rotated_15 = rot_15.apply(&test_point).expect("Operation failed");

    // Check that they produce expected intermediate rotations
    // t=0.5 should be 45 degrees around Z
    assert_relative_eq!(
        rotated_05[0],
        std::f64::consts::FRAC_1_SQRT_2,
        epsilon = 0.001
    );
    assert_relative_eq!(
        rotated_05[1],
        std::f64::consts::FRAC_1_SQRT_2,
        epsilon = 0.001
    );
    assert_relative_eq!(rotated_05[2], 0.0, epsilon = 0.001);

    // t=1.5 should be 135 degrees around Z
    assert_relative_eq!(
        rotated_15[0],
        -std::f64::consts::FRAC_1_SQRT_2,
        epsilon = 0.001
    );
    assert_relative_eq!(
        rotated_15[1],
        std::f64::consts::FRAC_1_SQRT_2,
        epsilon = 0.001
    );
    assert_relative_eq!(rotated_15[2], 0.0, epsilon = 0.001);
}

#[test]
#[allow(dead_code)]
fn test_rotation_spline_cubic() {
    // Create a rotation spline
    let rotations = vec![
        Rotation::identity(),
        {
            let euler_90_arr = array![0.0, 0.0, PI / 2.0];
            let euler_90 = euler_90_arr.view();
            Rotation::from_euler(&euler_90, "xyz").expect("Operation failed")
        },
        {
            let euler_180_arr = array![0.0, 0.0, PI];
            let euler_180 = euler_180_arr.view();
            Rotation::from_euler(&euler_180, "xyz").expect("Operation failed")
        },
    ];
    let times = vec![0.0, 1.0, 2.0];

    let mut spline = RotationSpline::new(&rotations, &times).expect("Operation failed");

    // Switch to cubic interpolation
    spline
        .set_interpolation_type("cubic")
        .expect("Operation failed");
    assert_eq!(spline.interpolation_type(), "cubic");

    // Test that the velocities have been computed
    // (This is an implementation detail, so we don't directly test the velocities)

    // Test interpolation at key times
    let test_point_arr = array![1.0, 0.0, 0.0];
    let test_point = test_point_arr.view();

    let rot_0 = spline.interpolate(0.0);
    let rot_1 = spline.interpolate(1.0);
    let rot_2 = spline.interpolate(2.0);

    let rotated_0 = rot_0.apply(&test_point).expect("Operation failed");
    let rotated_1 = rot_1.apply(&test_point).expect("Operation failed");
    let rotated_2 = rot_2.apply(&test_point).expect("Operation failed");

    // Check that they match the original rotations
    assert_relative_eq!(rotated_0[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_0[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_0[2], 0.0, epsilon = 1e-10);

    assert_relative_eq!(rotated_1[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_1[1], 1.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_1[2], 0.0, epsilon = 1e-10);

    assert_relative_eq!(rotated_2[0], -1.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_2[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(rotated_2[2], 0.0, epsilon = 1e-10);

    // Test interpolation at midpoints
    // The results will be different from SLERP due to the cubic interpolation
    // But we can't directly test the values without knowing the implementation details
    // Instead, we make sure that it produces valid rotations
    let rot_05 = spline.interpolate(0.5);
    let rot_15 = spline.interpolate(1.5);

    let rotated_05 = rot_05.apply(&test_point).expect("Operation failed");
    let rotated_15 = rot_15.apply(&test_point).expect("Operation failed");

    // Check that the results are unit vectors (valid rotations)
    let norm_05 = (rotated_05[0] * rotated_05[0]
        + rotated_05[1] * rotated_05[1]
        + rotated_05[2] * rotated_05[2])
        .sqrt();
    let norm_15 = (rotated_15[0] * rotated_15[0]
        + rotated_15[1] * rotated_15[1]
        + rotated_15[2] * rotated_15[2])
        .sqrt();

    assert_relative_eq!(norm_05, 1.0, epsilon = 1e-10);
    assert_relative_eq!(norm_15, 1.0, epsilon = 1e-10);
}

#[test]
#[allow(dead_code)]
fn test_rotation_spline_angular_velocity() {
    // Create a simple rotation spline (rotation around Z-axis)
    let rotations = vec![
        Rotation::identity(),
        {
            let euler_180_arr = array![0.0, 0.0, PI];
            let euler_180 = euler_180_arr.view();
            Rotation::from_euler(&euler_180, "xyz").expect("Operation failed")
        }, // 180 degrees around Z in 1 second
    ];
    let times = vec![0.0, 1.0];

    let spline = RotationSpline::new(&rotations, &times).expect("Operation failed");

    // Test angular velocity at midpoint
    let velocity = spline.angular_velocity(0.5).expect("Operation failed");

    // For 180-degree rotation around Z-axis in 1 second, angular velocity should be [0, 0, PI]
    assert_relative_eq!(velocity[0], 0.0, epsilon = 1e-3);
    assert_relative_eq!(velocity[1], 0.0, epsilon = 1e-3);
    assert_relative_eq!(velocity[2], PI, epsilon = 1e-3);

    // Test at the endpoints (should be zero)
    let velocity_start = spline.angular_velocity(0.0).expect("Operation failed");
    let velocity_end = spline.angular_velocity(1.0).expect("Operation failed");

    assert_relative_eq!(
        (velocity_start.dot(&velocity_start)).sqrt(),
        0.0,
        epsilon = 1e-10
    );
    assert_relative_eq!(
        (velocity_end.dot(&velocity_end)).sqrt(),
        0.0,
        epsilon = 1e-10
    );

    // Test with cubic interpolation
    let mut cubic_spline = RotationSpline::new(&rotations, &times).expect("Operation failed");
    cubic_spline
        .set_interpolation_type("cubic")
        .expect("Operation failed");

    // Velocity with cubic interpolation
    let cubic_velocity = cubic_spline
        .angular_velocity(0.5)
        .expect("Operation failed");

    // Cubic interpolation should also produce valid angular velocities
    let cubic_magnitude = (cubic_velocity.dot(&cubic_velocity)).sqrt();
    assert!(cubic_magnitude > 0.0); // Should have non-zero velocity at midpoint
}

#[test]
#[allow(dead_code)]
fn test_rotation_spline_angular_acceleration() {
    // Create a rotation spline
    let rotations = vec![
        Rotation::identity(),
        {
            let euler_90_arr = array![0.0, 0.0, PI / 2.0];
            let euler_90 = euler_90_arr.view();
            Rotation::from_euler(&euler_90, "xyz").expect("Operation failed")
        },
        {
            let euler_180_arr = array![0.0, 0.0, PI];
            let euler_180 = euler_180_arr.view();
            Rotation::from_euler(&euler_180, "xyz").expect("Operation failed")
        },
    ];
    let times = vec![0.0, 1.0, 2.0];

    // With SLERP, acceleration should be zero
    let spline = RotationSpline::new(&rotations, &times).expect("Operation failed");
    let accel = spline.angular_acceleration(0.5);

    assert_relative_eq!(accel[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(accel[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(accel[2], 0.0, epsilon = 1e-10);

    // With cubic interpolation, acceleration should be non-zero
    let mut cubic_spline = RotationSpline::new(&rotations, &times).expect("Operation failed");
    cubic_spline
        .set_interpolation_type("cubic")
        .expect("Operation failed");

    // Create a more complex rotation sequence to better test acceleration
    let complex_rotations = vec![
        Rotation::identity(),
        {
            let euler_1_arr = array![0.0, PI / 4.0, 0.0];
            let euler_1 = euler_1_arr.view();
            Rotation::from_euler(&euler_1, "xyz").expect("Operation failed")
        },
        {
            let euler_2_arr = array![PI / 4.0, PI / 4.0, 0.0];
            let euler_2 = euler_2_arr.view();
            Rotation::from_euler(&euler_2, "xyz").expect("Operation failed")
        },
        {
            let euler_3_arr = array![0.0, 0.0, 0.0];
            let euler_3 = euler_3_arr.view();
            Rotation::from_euler(&euler_3, "xyz").expect("Operation failed")
        },
    ];
    let complex_times = vec![0.0, 1.0, 2.0, 3.0];

    let mut complex_spline =
        RotationSpline::new(&complex_rotations, &complex_times).expect("Operation failed");
    complex_spline
        .set_interpolation_type("cubic")
        .expect("Operation failed");

    // Test at a few points
    let accel_1 = complex_spline.angular_acceleration(0.5);
    let accel_2 = complex_spline.angular_acceleration(1.5);
    let accel_3 = complex_spline.angular_acceleration(2.5);

    // Accelerations should be non-zero for a complex spline
    // We don't know the exact values, so just check that at least one component is non-zero
    let has_accel_1 = accel_1[0].abs() > 1e-6 || accel_1[1].abs() > 1e-6 || accel_1[2].abs() > 1e-6;
    let has_accel_2 = accel_2[0].abs() > 1e-6 || accel_2[1].abs() > 1e-6 || accel_2[2].abs() > 1e-6;
    let has_accel_3 = accel_3[0].abs() > 1e-6 || accel_3[1].abs() > 1e-6 || accel_3[2].abs() > 1e-6;

    // At least one of the tested points should have non-zero acceleration
    assert!(has_accel_1 || has_accel_2 || has_accel_3);
}
