//! Closed-form checks of the capsule inertia tensor.
//!
//! A solid capsule of mass `m` has a cylinder of height `h = 2 * half_height`
//! and the two hemispherical caps (together one sphere).  With the mass shared
//! by volume (`m_cyl`, `m_caps`) the moments about the centre are
//!
//! * `I_xz = m_cyl (3 r² + h²) / 12 + m_caps (2 r² / 5 + h² / 4 + 3 h r / 8)`
//! * `I_y  = m_cyl r² / 2 + m_caps 2 r² / 5`
//!
//! The cap term is `m_h (83/320 r² + (h/2 + 3r/8)²)` per hemisphere (parallel
//! axis theorem from the hemisphere's own centre of mass).  The earlier
//! `0.4 m r² + m (h/2 + 3r/8)²` used the flat-face moment of a *sphere* about
//! the hemisphere's centre of mass and so counted an extra `9/64 m_caps r²`.

use super::Capsule;
use crate::Shape;
use std::f64::consts::PI;

/// `(I_xz, I_y)` from the closed form of the module documentation.
fn closed_form(radius: f64, half_height: f64, mass: f64) -> (f64, f64) {
    let r = radius;
    let h = 2.0 * half_height;
    let cylinder_volume = PI * r * r * h;
    let sphere_volume = 4.0 / 3.0 * PI * r * r * r;
    let total = cylinder_volume + sphere_volume;
    let m_cylinder = mass * cylinder_volume / total;
    let m_caps = mass * sphere_volume / total;
    let ixz = m_cylinder * (3.0 * r * r + h * h) / 12.0
        + m_caps * (2.0 * r * r / 5.0 + h * h / 4.0 + 3.0 * h * r / 8.0);
    let iy = m_cylinder * r * r / 2.0 + m_caps * 2.0 * r * r / 5.0;
    (ixz, iy)
}

#[test]
fn radius_half_and_half_height_one_have_the_pinned_closed_form_moments() {
    // r = 0.5, h = 2: the cylinder carries 3/4 of the mass, the caps 1/4.
    //   I_xz = 0.75 (0.75 + 4) / 12 + 0.25 (0.1 + 1 + 0.375) = 0.665625 m
    //   I_y  = 0.75 * 0.125 + 0.25 * 0.1                     = 0.11875 m
    let capsule = Capsule::new(0.5, 1.0);
    let (ixz, iy) = closed_form(0.5, 1.0, 1.0);
    assert!((ixz - 0.665_625).abs() < 1e-12, "closed form {ixz}");
    assert!((iy - 0.118_75).abs() < 1e-12, "closed form {iy}");

    let array = capsule.inertia_tensor_array(1.0);
    assert!((array[0][0] - 0.665_625).abs() < 1e-12, "{}", array[0][0]);
    assert!((array[1][1] - 0.118_75).abs() < 1e-12, "{}", array[1][1]);
    assert!((array[2][2] - 0.665_625).abs() < 1e-12, "{}", array[2][2]);

    // The `Shape` implementation takes the same path (it used to carry its own
    // copy of the formula).
    let tensor = capsule.inertia_tensor(1.0);
    assert!((tensor[(0, 0)] - 0.665_625).abs() < 1e-12);
    assert!((tensor[(1, 1)] - 0.118_75).abs() < 1e-12);
    assert!((tensor[(2, 2)] - 0.665_625).abs() < 1e-12);

    // The over-counted value is clearly excluded: the extra term is
    // 9/64 * m_caps * r² = 9/64 * 0.25 * 0.25.
    let spurious = 9.0 / 64.0 * 0.25 * 0.25;
    assert!((array[0][0] - (0.665_625 + spurious)).abs() > 8e-3);
}

#[test]
fn all_three_entry_points_match_the_closed_form_for_assorted_capsules() {
    let cases = [
        (0.5, 1.0, 1.0),
        (1.0, 2.5, 7.0),
        (0.3, 0.1, 0.4),
        (2.0, 0.0, 3.0),
        (0.05, 3.0, 12.0),
    ];
    for (radius, half_height, mass) in cases {
        let capsule = Capsule::new(radius, half_height);
        let (ixz, iy) = closed_form(radius, half_height, mass);
        let scale = ixz.max(iy);
        let array = capsule.inertia_tensor_array(mass);
        let raw = capsule.inertia_tensor_raw(mass);
        let tensor = capsule.inertia_tensor(mass);
        let (transverse, axial) = capsule.principal_moments(mass);
        for (name, a, b) in [
            ("array xx", array[0][0], ixz),
            ("array yy", array[1][1], iy),
            ("array zz", array[2][2], ixz),
            ("raw xx", raw[0][0], ixz),
            ("raw yy", raw[1][1], iy),
            ("shape xx", tensor[(0, 0)], ixz),
            ("shape yy", tensor[(1, 1)], iy),
            ("shape zz", tensor[(2, 2)], ixz),
            ("transverse", transverse, ixz),
            ("axial", axial, iy),
        ] {
            assert!(
                (a - b).abs() <= 1e-12 * scale,
                "r={radius} hh={half_height} m={mass}: {name} {a} vs {b}"
            );
        }
        // Principal axes: no products of inertia.
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            assert_eq!(array[i][j], 0.0);
            assert_eq!(tensor[(i, j)], 0.0);
        }
    }
}

#[test]
fn inertia_is_linear_in_the_mass() {
    let capsule = Capsule::new(0.7, 1.3);
    let single = capsule.inertia_tensor_array(1.0);
    let triple = capsule.inertia_tensor_array(3.0);
    for i in 0..3 {
        assert!((triple[i][i] - 3.0 * single[i][i]).abs() < 1e-12);
    }
}
