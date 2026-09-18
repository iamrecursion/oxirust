// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the cable-net force model, the solar position
//! convention of [`ParametricFacade`] and the PCA plane fit.

use super::*;

// ---------------------------------------------------------------------------
// TensileStructure: constant cable tension
// ---------------------------------------------------------------------------

/// Sag of the single free node of a 3×3 net whose four cables each carry the
/// constant tension `tension`.
///
/// Vertical balance of the tributary load `q a²` against the four cables,
/// `4 T z / sqrt(a² + z²) = q a²`, gives `z = q a³ / sqrt(16 T² − q² a⁴)`.
fn closed_form_sag(spacing: f64, tension: f64, pressure: f64) -> f64 {
    pressure * spacing.powi(3)
        / (16.0 * tension * tension - pressure * pressure * spacing.powi(4)).sqrt()
}

#[test]
fn cable_net_form_finding_matches_closed_form_sag() {
    // The same tension and load on nets of different mesh spacing: a force of
    // T / L per cable (the former behaviour) makes the sag depend on the
    // spacing and misses every one of these closed-form values.
    for &(spacing, tension, pressure) in
        &[(1.0, 100.0, 50.0), (2.5, 1000.0, 50.0), (0.4, 250.0, 120.0)]
    {
        let mut net = TensileStructure::flat_net(3, 3, 2.0 * spacing, 2.0 * spacing, tension);
        let free: Vec<usize> = (0..net.nodes.len())
            .filter(|i| !net.anchors.contains(i))
            .collect();
        assert_eq!(free, vec![4], "only the centre node is free");
        // Unit nodal mass and linearised stiffness 4 T / a: a time step of
        // 0.1 / omega keeps the explicit scheme far inside its stability limit.
        let omega = (4.0 * tension / spacing).sqrt();
        let max_sag = net.form_find(pressure, 200_000, 0.1 / omega, 0.1, 1e-10);
        let expected = closed_form_sag(spacing, tension, pressure);
        let centre = net.nodes[4];
        assert!(
            (max_sag - expected).abs() < 1e-6,
            "spacing {spacing}: sag {max_sag} expected {expected}"
        );
        assert!(
            (centre.z + expected).abs() < 1e-6,
            "pressure acts in -z, centre z = {} expected {}",
            centre.z,
            -expected
        );
        assert!((centre.x - spacing).abs() < 1e-9, "x = {}", centre.x);
        assert!((centre.y - spacing).abs() < 1e-9, "y = {}", centre.y);
    }
}

#[test]
fn cable_net_equilibrium_cable_force_has_magnitude_of_the_prestress() {
    // At the converged state the four cables meeting at the free node pull
    // with the prestress T each: their upward components balance the load.
    let (spacing, tension, pressure) = (2.0, 800.0, 40.0);
    let mut net = TensileStructure::flat_net(3, 3, 2.0 * spacing, 2.0 * spacing, tension);
    let omega = (4.0 * tension / spacing).sqrt();
    net.form_find(pressure, 200_000, 0.1 / omega, 0.1, 1e-10);
    let centre = net.nodes[4];
    let mut vertical = 0.0;
    for &(a, b) in &net.cables {
        if a != 4 && b != 4 {
            continue;
        }
        let other = if a == 4 { net.nodes[b] } else { net.nodes[a] };
        let cable = other.sub(&centre);
        // Unit vector from the free node along the cable times the tension.
        vertical += tension * cable.z / cable.norm();
    }
    let load = pressure * net.trib_areas[4];
    assert!(
        (vertical - load).abs() < 1e-6 * load,
        "cables lift with {vertical}, the load is {load}"
    );
}

// ---------------------------------------------------------------------------
// ParametricFacade: solar azimuth from south, positive west
// ---------------------------------------------------------------------------

/// Independent evaluation of the documented convention: the azimuth from
/// north via `acos` converted to degrees from south, signed by the hour angle.
fn azimuth_from_south_via_acos(facade: &ParametricFacade, day: f64, hour: f64) -> f64 {
    let lat = facade.latitude.to_radians();
    let decl = 23.45_f64.to_radians() * (360.0 / 365.0 * (day - 81.0)).to_radians().sin();
    let hour_angle = (hour - 12.0) * 15.0;
    let sin_alt = lat.sin() * decl.sin() + lat.cos() * decl.cos() * hour_angle.to_radians().cos();
    let cos_alt = sin_alt.clamp(-1.0, 1.0).asin().cos();
    let cos_az_north =
        ((decl.sin() - lat.sin() * sin_alt) / (lat.cos() * cos_alt)).clamp(-1.0, 1.0);
    let az_south = 180.0 - cos_az_north.acos().to_degrees();
    if hour_angle > 0.0 {
        az_south
    } else {
        -az_south
    }
}

fn facade_facing(azimuth_from_north: f64) -> ParametricFacade {
    ParametricFacade {
        facade_azimuth: azimuth_from_north,
        ..ParametricFacade::tokyo_south()
    }
}

#[test]
fn solar_noon_sun_is_due_south_and_hits_the_south_facade() {
    let facade = ParametricFacade::tokyo_south();
    let alt = facade.solar_altitude(172.0, 12.0);
    let az = facade.solar_azimuth(172.0, 12.0);
    assert!(az.abs() < 1e-4, "noon azimuth from south = {az}");
    // Facing the sun: the incidence angle on the vertical south facade equals
    // the solar altitude, so the incident cosine is positive.
    let incidence = facade.incidence_angle(alt, az);
    assert!(
        (incidence - alt).abs() < 1e-4,
        "incidence {incidence} alt {alt}"
    );
    assert!(incidence.to_radians().cos() > 0.2);
    // The same instant seen from a north facade: no direct sun.
    let north = facade_facing(0.0);
    assert!(north.incidence_angle(alt, az) > 90.0);
}

#[test]
fn equinox_afternoon_azimuth_matches_the_hand_computed_value() {
    // Declination 0 (day 81), hour angle 45°, latitude 35.7°:
    // A = atan2(sin 45°, cos 45° sin 35.7°) = 59.7347° west of south.
    let facade = ParametricFacade::tokyo_south();
    let az = facade.solar_azimuth(81.0, 15.0);
    assert!((az - 59.734_677).abs() < 1e-4, "azimuth = {az}");
    let morning = facade.solar_azimuth(81.0, 9.0);
    assert!((morning + 59.734_677).abs() < 1e-4, "azimuth = {morning}");
}

#[test]
fn solar_azimuth_agrees_with_the_acos_form_and_mirrors_about_noon() {
    let facade = ParametricFacade::tokyo_south();
    for &day in &[20.0, 81.0, 120.0, 172.0, 266.0, 355.0] {
        for half_hours in 0..=44 {
            // 01:00 to 23:00 in half-hour steps (the sun is also tracked below the horizon).
            let hour = f64::from(half_hours) * 0.5 + 1.0;
            let az = facade.solar_azimuth(day, hour);
            let reference = azimuth_from_south_via_acos(&facade, day, hour);
            if (hour - 12.0).abs() > 1e-9 {
                assert!(
                    (az - reference).abs() < 1e-3,
                    "day {day} hour {hour}: {az} vs {reference}"
                );
                // East of south in the morning, west of south in the afternoon.
                assert_eq!(az > 0.0, hour > 12.0, "day {day} hour {hour}: {az}");
            }
            let mirrored = facade.solar_azimuth(day, 24.0 - hour);
            assert!(
                (az + mirrored).abs() < 1e-9,
                "day {day} hour {hour}: {az} vs {mirrored}"
            );
        }
    }
}

#[test]
fn facades_see_the_sun_on_the_side_it_is_on() {
    // Equinox afternoon: the sun is south-west, so a west facade is lit at a
    // 45° incidence angle and an east facade is in its own shadow.
    let base = ParametricFacade::tokyo_south();
    let alt = base.solar_altitude(81.0, 15.0);
    let az = base.solar_azimuth(81.0, 15.0);
    let west = facade_facing(270.0).incidence_angle(alt, az);
    let east = facade_facing(90.0).incidence_angle(alt, az);
    assert!((west - 45.0).abs() < 1e-6, "west incidence = {west}");
    assert!((east - 135.0).abs() < 1e-6, "east incidence = {east}");
    // Morning: the roles swap.
    let alt = base.solar_altitude(81.0, 9.0);
    let az = base.solar_azimuth(81.0, 9.0);
    assert!((facade_facing(90.0).incidence_angle(alt, az) - 45.0).abs() < 1e-6);
    assert!((facade_facing(270.0).incidence_angle(alt, az) - 135.0).abs() < 1e-6);
}

#[test]
fn annual_exposure_orders_the_facade_orientations() {
    let south = facade_facing(180.0).annual_solar_exposure();
    let east = facade_facing(90.0).annual_solar_exposure();
    let west = facade_facing(270.0).annual_solar_exposure();
    let north = facade_facing(0.0).annual_solar_exposure();
    // The sampled hours are symmetric about noon, so east and west agree.
    assert!((east - west).abs() < 1e-9 * east, "east {east} west {west}");
    assert!(south > east, "south {south} east {east}");
    assert!(east > north, "east {east} north {north}");
    // At 35.7 degrees north the north facade only sees the early and late
    // summer sun: a tiny fraction of the south facade's exposure.
    assert!(south > 10.0 * north, "south {south} north {north}");
    assert!(north >= 0.0);
}

// ---------------------------------------------------------------------------
// fit_plane: smallest-eigenvalue eigenvector of the covariance
// ---------------------------------------------------------------------------

/// Deterministic xorshift64 generator mapped to `[-1, 1)`.
struct Noise(u64);

impl Noise {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1_u64 << 52) as f64 - 1.0
    }
}

/// A 7×7 grid of parameters `(0.5 i, 0.7 j)` mapped through `place`.
fn plane_points(mut place: impl FnMut(f64, f64) -> V3) -> Vec<V3> {
    (-3..=3)
        .flat_map(|i| (-3..=3).map(move |j| (0.5 * f64::from(i), 0.7 * f64::from(j))))
        .map(|(u, v)| place(u, v))
        .collect()
}

fn assert_normal_close(normal: V3, expected: V3, tol: f64) {
    assert!(
        (normal.norm() - 1.0).abs() < 1e-12,
        "|n| = {}",
        normal.norm()
    );
    assert!(
        (normal.x - expected.x).abs() < tol
            && (normal.y - expected.y).abs() < tol
            && (normal.z - expected.z).abs() < tol,
        "normal ({}, {}, {}) expected ({}, {}, {})",
        normal.x,
        normal.y,
        normal.z,
        expected.x,
        expected.y,
        expected.z
    );
}

#[test]
fn fit_plane_of_a_constant_y_plane_has_a_y_normal() {
    let pts = plane_points(|u, v| V3::new(u, 2.0, v));
    let (centroid, normal) = fit_plane(&pts);
    assert_normal_close(normal, V3::new(0.0, 1.0, 0.0), 1e-12);
    assert!((centroid.y - 2.0).abs() < 1e-12);
}

#[test]
fn fit_plane_of_a_constant_x_plane_has_an_x_normal() {
    let pts = plane_points(|u, v| V3::new(-1.0, u, v));
    let (centroid, normal) = fit_plane(&pts);
    assert_normal_close(normal, V3::new(1.0, 0.0, 0.0), 1e-12);
    assert!((centroid.x + 1.0).abs() < 1e-12);
}

#[test]
fn fit_plane_of_a_constant_z_plane_has_a_z_normal() {
    let pts = plane_points(|u, v| V3::new(u, v, -0.75));
    let (centroid, normal) = fit_plane(&pts);
    assert_normal_close(normal, V3::new(0.0, 0.0, 1.0), 1e-12);
    assert!((centroid.z + 0.75).abs() < 1e-12);
}

#[test]
fn fit_plane_recovers_a_tilted_plane_from_noisy_points() {
    // z = 0.3 x + 0.2 y with deterministic noise of amplitude 1e-4 on z.
    let mut noise = Noise(0x9E37_79B9_7F4A_7C15);
    let pts = plane_points(|x, y| V3::new(x, y, 0.3 * x + 0.2 * y + 1e-4 * noise.uniform()));
    let (_, normal) = fit_plane(&pts);
    let len = (0.3_f64 * 0.3 + 0.2 * 0.2 + 1.0).sqrt();
    assert_normal_close(normal, V3::new(-0.3 / len, -0.2 / len, 1.0 / len), 1e-3);
}

#[test]
fn fit_plane_of_exact_points_in_general_position_is_exact() {
    let n = V3::new(1.0, 2.0, 3.0).normalize();
    let u = n.cross(&V3::new(1.0, 0.0, 0.0)).normalize();
    let w = n.cross(&u);
    let origin = V3::new(0.4, -1.3, 2.2);
    let pts = plane_points(|a, b| origin.add(&u.scale(a)).add(&w.scale(b)));
    let (centroid, normal) = fit_plane(&pts);
    assert!(
        normal.cross(&n).norm() < 1e-12,
        "normal is not parallel to ({}, {}, {})",
        n.x,
        n.y,
        n.z
    );
    assert!(centroid.sub(&origin).dot(&n).abs() < 1e-12);
    for p in &pts {
        assert!(point_to_plane_distance(*p, centroid, normal) < 1e-12);
    }
}

#[test]
fn fit_plane_orients_the_normal_by_its_dominant_component() {
    let n = V3::new(-1.0, 2.0, -6.0).normalize();
    let u = n.cross(&V3::new(0.0, 1.0, 0.0)).normalize();
    let w = n.cross(&u);
    let pts = plane_points(|a, b| u.scale(a).add(&w.scale(b)));
    let (_, normal) = fit_plane(&pts);
    // The z component dominates and is negative in `n`, so the reported
    // normal is the flipped one.
    assert_normal_close(normal, n.scale(-1.0), 1e-12);
}

#[test]
fn fit_plane_degenerate_inputs_fall_back_to_the_z_axis() {
    let z = V3::new(0.0, 0.0, 1.0);
    let (centroid, normal) = fit_plane(&[V3::new(1.0, 2.0, 3.0), V3::new(4.0, 5.0, 6.0)]);
    assert_eq!(normal, z);
    assert_eq!(centroid, V3::zero());
    let same = vec![V3::new(1.0, -2.0, 0.5); 5];
    let (centroid, normal) = fit_plane(&same);
    assert_eq!(normal, z);
    assert!(centroid.sub(&V3::new(1.0, -2.0, 0.5)).norm() < 1e-15);
    // Collinear points: any unit normal perpendicular to the line is valid.
    let direction = V3::new(1.0, 1.0, 2.0).normalize();
    let line: Vec<V3> = (0..6).map(|k| direction.scale(f64::from(k))).collect();
    let (_, normal) = fit_plane(&line);
    assert!((normal.norm() - 1.0).abs() < 1e-12);
    assert!(normal.dot(&direction).abs() < 1e-12);
}

#[test]
fn symmetric_eigen_3x3_returns_orthonormal_eigenpairs() {
    let a = [[4.0, 1.0, 0.5], [1.0, 3.0, -0.2], [0.5, -0.2, 2.0]];
    let (values, vectors) = symmetric_eigen_3x3(a);
    assert!((values.iter().sum::<f64>() - 9.0).abs() < 1e-12, "trace");
    for (k, &lambda) in values.iter().enumerate() {
        let v = [vectors[0][k], vectors[1][k], vectors[2][k]];
        for (row, &v_row) in a.iter().zip(v.iter()) {
            let av: f64 = row.iter().zip(v.iter()).map(|(m, x)| m * x).sum();
            assert!(
                (av - lambda * v_row).abs() < 1e-12,
                "A v = lambda v, k = {k}"
            );
        }
        let norm: f64 = v.iter().map(|x| x * x).sum();
        assert!((norm - 1.0).abs() < 1e-12, "unit length, k = {k}");
    }
    for p in 0..3 {
        for q in (p + 1)..3 {
            let dot: f64 = vectors.iter().map(|row| row[p] * row[q]).sum();
            assert!(dot.abs() < 1e-12, "columns {p} and {q} are orthogonal");
        }
    }
    // Repeated eigenvalues and an already diagonal matrix.
    let (values, _) = symmetric_eigen_3x3([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 5.0]]);
    assert_eq!(values, [2.0, 2.0, 5.0]);
}
