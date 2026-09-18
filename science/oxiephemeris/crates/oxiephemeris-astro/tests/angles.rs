//! Cross-checks for the chart angles.
//!
//! The strongest tests here never trust the closed forms: the ascendant
//! and vertex are re-derived through explicit rotation matrices
//! ([`oxiephemeris_bodies::math`]) — build the ecliptic → horizon
//! rotation, intersect the two planes as a cross product of their poles,
//! pick the eastern/western branch geometrically, and demand agreement
//! with the `atan2` closed forms. The defining conditions (RA of the MC
//! equals the RAMC; the ascendant sits on the horizon *and* in the east;
//! the vertex sits on the prime vertical in the west) are asserted
//! directly, plus continuity across the `θ = 0/90/180/270°` quadrant
//! boundaries and the documented degenerate cases.

use core::f64::consts::{FRAC_PI_2, PI};
use oxiephemeris_astro::angles::{ascendant, east_point, local_sidereal_time, mc, ramc, vertex};
use oxiephemeris_bodies::math::{cross, dot, r1, r2, r3, Mat3, Vec3};

const TWO_PI: f64 = 2.0 * PI;
const DEG: f64 = PI / 180.0;
/// True obliquity used across the grid (a contemporary value).
const EPS_TEST: f64 = 23.4367 * DEG;

/// Latitudes strictly below the polar circle for `EPS_TEST` (66.5633°).
const SAFE_PHIS_DEG: [f64; 9] = [-60.0, -35.0, -20.0, 0.0, 20.0, 35.0, 51.5, 60.0, 66.0];

/// Sidereal times including the exact quadrant boundaries.
const THETAS_DEG: [f64; 12] = [
    0.0, 17.3, 62.3, 90.0, 107.3, 152.3, 180.0, 197.3, 242.3, 270.0, 287.3, 332.3,
];

/// Smallest angular distance between two directions, radians.
fn wrap_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(TWO_PI);
    if d > PI {
        TWO_PI - d
    } else {
        d
    }
}

/// Equatorial unit vector of the ecliptic point of longitude `lambda`.
fn ecl_unit(lambda: f64, eps: f64) -> Vec3 {
    [
        lambda.cos(),
        lambda.sin() * eps.cos(),
        lambda.sin() * eps.sin(),
    ]
}

/// Zenith direction (equatorial frame) for sidereal time / latitude.
fn zenith(theta: f64, phi: f64) -> Vec3 {
    [phi.cos() * theta.cos(), phi.cos() * theta.sin(), phi.sin()]
}

/// East point of the horizon (equatorial frame).
fn east_dir(theta: f64) -> Vec3 {
    [-theta.sin(), theta.cos(), 0.0]
}

/// North point of the horizon (equatorial frame) — the prime-vertical
/// pole.
fn north_point(theta: f64, phi: f64) -> Vec3 {
    [
        -phi.sin() * theta.cos(),
        -phi.sin() * theta.sin(),
        phi.cos(),
    ]
}

fn ok<E: std::fmt::Display>(r: Result<f64, E>) -> f64 {
    match r {
        Ok(v) => v,
        Err(e) => panic!("unexpected angles error: {e}"),
    }
}

/// Right ascension of an ecliptic longitude, from the equatorial vector.
fn right_ascension(lambda: f64, eps: f64) -> f64 {
    let p = ecl_unit(lambda, eps);
    p[1].atan2(p[0])
}

/// Ecliptic → horizon (south, east, zenith) rotation, built from the
/// `bodies` crate's rotation matrices: `R2(π/2 − φ) · R3(θ) · R1(−ε)`.
fn ecl_to_hor(theta: f64, phi: f64, eps: f64) -> Mat3 {
    r2(FRAC_PI_2 - phi).mul(&r3(theta)).mul(&r1(-eps))
}

/// Independent matrix derivation of the ascendant / vertex: intersect
/// the ecliptic (pole `q` in the horizon frame) with the horizon plane
/// (`z = 0`) or the prime vertical (`x = 0`), then pick the branch with
/// the requested sign of the east component.
fn angle_by_matrices(theta: f64, phi: f64, eps: f64, plane_normal: Vec3, want_east: bool) -> f64 {
    let m = ecl_to_hor(theta, phi, eps);
    let q = m.apply([0.0, 0.0, 1.0]);
    let mut d = cross(plane_normal, q);
    if (d[1] > 0.0) != want_east {
        d = [-d[0], -d[1], -d[2]];
    }
    // `m` maps ecliptic to horizon coordinates, so the transpose brings
    // the intersection direction straight back to the ecliptic frame.
    let back = m.transpose().apply(d);
    back[1].atan2(back[0]).rem_euclid(TWO_PI)
}

#[test]
fn horizon_frame_orientation_is_south_east_zenith() {
    // Self-check of the test harness itself: the equatorial → horizon
    // map must send the zenith to +z and the east point to +y.
    for &phi_deg in &SAFE_PHIS_DEG {
        for &theta_deg in &THETAS_DEG {
            let (theta, phi) = (theta_deg * DEG, phi_deg * DEG);
            let m = r2(FRAC_PI_2 - phi).mul(&r3(theta));
            let z = m.apply(zenith(theta, phi));
            let e = m.apply(east_dir(theta));
            let n = m.apply(north_point(theta, phi));
            assert!(wrap_len(z, [0.0, 0.0, 1.0]) < 1e-14, "zenith image {z:?}");
            assert!(wrap_len(e, [0.0, 1.0, 0.0]) < 1e-14, "east image {e:?}");
            assert!(wrap_len(n, [-1.0, 0.0, 0.0]) < 1e-14, "north image {n:?}");
        }
    }
}

/// Max component difference of two vectors.
fn wrap_len(a: Vec3, b: Vec3) -> f64 {
    let mut worst = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        worst = worst.max((x - y).abs());
    }
    worst
}

#[test]
fn mc_right_ascension_equals_ramc() {
    for &eps in &[0.0, 10.0 * DEG, EPS_TEST, 30.0 * DEG] {
        for &theta_deg in &THETAS_DEG {
            let theta = theta_deg * DEG;
            let lambda_mc = mc(theta, eps);
            let ra = right_ascension(lambda_mc, eps);
            assert!(
                wrap_diff(ra, ramc(theta)) < 1e-12,
                "RA(MC) != RAMC at theta={theta_deg}, eps={eps}: {ra} vs {theta}"
            );
        }
    }
}

#[test]
fn zero_obliquity_continuous_limits() {
    for &theta_deg in &THETAS_DEG {
        let theta = theta_deg * DEG;
        assert!(wrap_diff(mc(theta, 0.0), theta) < 1e-12);
        assert!(wrap_diff(east_point(theta, 0.0), theta + FRAC_PI_2) < 1e-12);
        // The ascendant limit is latitude-independent for eps = 0.
        for &phi_deg in &[-60.0, 0.0, 51.5] {
            let asc = ok(ascendant(theta, phi_deg * DEG, 0.0));
            assert!(
                wrap_diff(asc, theta + FRAC_PI_2) < 1e-12,
                "asc(eps=0) at theta={theta_deg}, phi={phi_deg}"
            );
        }
    }
}

#[test]
fn ascendant_lies_on_horizon_and_east() {
    for &phi_deg in &SAFE_PHIS_DEG {
        for &theta_deg in &THETAS_DEG {
            let (theta, phi) = (theta_deg * DEG, phi_deg * DEG);
            let asc = ok(ascendant(theta, phi, EPS_TEST));
            let p = ecl_unit(asc, EPS_TEST);
            let altitude_sin = dot(p, zenith(theta, phi));
            assert!(
                altitude_sin.abs() < 1e-10,
                "asc off horizon at phi={phi_deg}, theta={theta_deg}: {altitude_sin:e}"
            );
            assert!(
                dot(p, east_dir(theta)) > 0.0,
                "asc not east at phi={phi_deg}, theta={theta_deg}"
            );
        }
    }
}

#[test]
fn ascendant_matches_matrix_derivation() {
    for &phi_deg in &SAFE_PHIS_DEG {
        for &theta_deg in &THETAS_DEG {
            let (theta, phi) = (theta_deg * DEG, phi_deg * DEG);
            let closed = ok(ascendant(theta, phi, EPS_TEST));
            let matrix = angle_by_matrices(theta, phi, EPS_TEST, [0.0, 0.0, 1.0], true);
            assert!(
                wrap_diff(closed, matrix) < 1e-10,
                "asc mismatch at phi={phi_deg}, theta={theta_deg}: {closed} vs {matrix}"
            );
        }
    }
}

#[test]
fn east_point_ra_is_ramc_plus_90_and_equals_equator_ascendant() {
    for &theta_deg in &THETAS_DEG {
        let theta = theta_deg * DEG;
        let ep = east_point(theta, EPS_TEST);
        let ra = right_ascension(ep, EPS_TEST);
        assert!(
            wrap_diff(ra, theta + FRAC_PI_2) < 1e-12,
            "RA(EP) != RAMC + 90 deg at theta={theta_deg}"
        );
        let asc0 = ok(ascendant(theta, 0.0, EPS_TEST));
        assert!(
            wrap_diff(ep, asc0) < 1e-12,
            "EP != equator ascendant at theta={theta_deg}"
        );
    }
}

#[test]
fn vertex_on_prime_vertical_everywhere() {
    // Membership (ecliptic /\ prime vertical) holds at every latitude,
    // including the tropics and the equator where the western branch is
    // not guaranteed.
    for &phi_deg in &[-60.0, -35.0, -10.0, 0.0, 10.0, 20.0, 35.0, 51.5, 66.0, 80.0] {
        for &theta_deg in &THETAS_DEG {
            let (theta, phi) = (theta_deg * DEG, phi_deg * DEG);
            let vtx = ok(vertex(theta, phi, EPS_TEST));
            let p = ecl_unit(vtx, EPS_TEST);
            let off_plane = dot(p, north_point(theta, phi));
            assert!(
                off_plane.abs() < 1e-10,
                "vertex off prime vertical at phi={phi_deg}, theta={theta_deg}: {off_plane:e}"
            );
        }
    }
}

#[test]
fn vertex_is_west_above_tropics_and_matches_matrices() {
    // |phi| > eps: the returned branch is documented to be western.
    for &phi_deg in &[-60.0, -35.0, 35.0, 51.5, 66.0, 80.0] {
        for &theta_deg in &THETAS_DEG {
            let (theta, phi) = (theta_deg * DEG, phi_deg * DEG);
            let vtx = ok(vertex(theta, phi, EPS_TEST));
            let p = ecl_unit(vtx, EPS_TEST);
            assert!(
                dot(p, east_dir(theta)) < 0.0,
                "vertex not west at phi={phi_deg}, theta={theta_deg}"
            );
            let matrix = angle_by_matrices(theta, phi, EPS_TEST, [1.0, 0.0, 0.0], false);
            assert!(
                wrap_diff(vtx, matrix) < 1e-10,
                "vertex mismatch at phi={phi_deg}, theta={theta_deg}: {vtx} vs {matrix}"
            );
        }
    }
}

#[test]
fn degenerate_cases_behave_as_documented() {
    // Exact poles: the continuous limits pi (north) and 0 (south).
    for &theta_deg in &THETAS_DEG {
        let theta = theta_deg * DEG;
        let north = ok(ascendant(theta, FRAC_PI_2, EPS_TEST));
        assert!(wrap_diff(north, PI) < 1e-9, "north-pole asc {north}");
        let south = ok(ascendant(theta, -FRAC_PI_2, EPS_TEST));
        assert!(wrap_diff(south, 0.0) < 1e-9, "south-pole asc {south}");
    }
    // Horizon/ecliptic coincidence at the polar circle.
    assert!(ascendant(3.0 * FRAC_PI_2, FRAC_PI_2 - EPS_TEST, EPS_TEST).is_err());
    assert!(ascendant(FRAC_PI_2, EPS_TEST - FRAC_PI_2, EPS_TEST).is_err());
    // Prime-vertical/ecliptic coincidence at |phi| = eps.
    assert!(vertex(FRAC_PI_2, EPS_TEST, EPS_TEST).is_err());
    assert!(vertex(3.0 * FRAC_PI_2, -EPS_TEST, EPS_TEST).is_err());
    // Nudged off the exact degeneracy, both are defined again.
    assert!(ascendant(3.0 * FRAC_PI_2, FRAC_PI_2 - EPS_TEST - 1e-4, EPS_TEST).is_ok());
    assert!(vertex(FRAC_PI_2, EPS_TEST + 1e-4, EPS_TEST).is_ok());
    // Vertex at the equator: the prime vertical is the celestial
    // equator, and the vertex is whichever equinox is currently west.
    for &theta_deg in &[17.3, 62.3, 107.3, 152.3] {
        let vtx = ok(vertex(theta_deg * DEG, 0.0, EPS_TEST));
        assert!(
            wrap_diff(vtx, 0.0) < 1e-9,
            "equator vertex {vtx} (sin theta > 0)"
        );
    }
    for &theta_deg in &[197.3, 242.3, 287.3, 332.3] {
        let vtx = ok(vertex(theta_deg * DEG, 0.0, EPS_TEST));
        assert!(
            wrap_diff(vtx, PI) < 1e-9,
            "equator vertex {vtx} (sin theta < 0)"
        );
    }
}

/// Sweeps `f` over the full circle of `theta` and returns the smallest
/// and largest forward increment plus the accumulated total.
fn sweep(steps: i32, f: impl Fn(f64) -> f64) -> (f64, f64, f64) {
    let step = TWO_PI / f64::from(steps);
    let mut previous = f(0.0);
    let (mut smallest, mut largest, mut total) = (f64::MAX, f64::MIN, 0.0);
    for i in 1..=steps {
        let value = f(f64::from(i) * step);
        let inc = (value - previous).rem_euclid(TWO_PI);
        smallest = smallest.min(inc);
        largest = largest.max(inc);
        total += inc;
        previous = value;
    }
    (smallest, largest, total)
}

#[test]
fn continuity_across_quadrant_boundaries() {
    // 0.05 deg sampling => step 8.727e-4 rad. Bounds are the numerically
    // probed slope extrema of each curve times the step, with margin:
    // MC slope in [cos eps, 1/cos eps] = [0.9175, 1.0899];
    // asc slope max: 1.09 (phi=0), 2.40 (51.5), 41.4 (66).
    let steps = 7200;
    let step = TWO_PI / f64::from(steps);
    let (mc_min, mc_max, mc_total) = sweep(steps, |t| mc(t, EPS_TEST));
    assert!(
        mc_min > 0.85 * step && mc_max < 1.15 * step,
        "MC jump: {mc_min:e}..{mc_max:e}"
    );
    assert!((mc_total - TWO_PI).abs() < 1e-9, "MC winding {mc_total}");
    for (phi_deg, slope_bound) in [(0.0, 1.3), (35.0, 1.9), (51.5, 2.9), (66.0, 50.0)] {
        let phi = phi_deg * DEG;
        let (asc_min, asc_max, asc_total) = sweep(steps, |t| ok(ascendant(t, phi, EPS_TEST)));
        assert!(
            asc_min > 0.0 && asc_max < slope_bound * step,
            "asc jump at phi={phi_deg}: {asc_min:e}..{asc_max:e}"
        );
        assert!(
            (asc_total - TWO_PI).abs() < 1e-9,
            "asc winding at phi={phi_deg}: {asc_total}"
        );
    }
}

#[test]
fn local_sidereal_time_and_ramc_normalize() {
    let last = local_sidereal_time(6.0, 1.0);
    assert!((last - (7.0 - TWO_PI)).abs() < 1e-15);
    let last2 = local_sidereal_time(0.25, -1.0);
    assert!((last2 - (TWO_PI - 0.75)).abs() < 1e-12);
    assert!((ramc(-0.5) - (TWO_PI - 0.5)).abs() < 1e-12);
    assert!((ramc(TWO_PI + 0.5) - 0.5).abs() < 1e-12);
}
