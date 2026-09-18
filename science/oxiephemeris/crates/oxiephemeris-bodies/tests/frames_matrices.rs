//! Structural and cross-check tests for the frame matrices.
//!
//! The strongest test here rebuilds the GCRS → mean-of-date rotation on a
//! completely different route — the equinox-based IAU 2006 precession
//! angles `zeta_A`, `z_A`, `theta_A` (Capitaine, Wallace & Chapront 2003,
//! A&A 412, 567, eq. 40) composed with the explicit frame-bias matrix
//! (IERS TN36 eqs. 5.21/5.33) — and requires agreement with the
//! Fukushima–Williams route at the sub-milliarcsecond level over
//! `t` in `[-2, +2]` centuries. A sign error in any rotation convention
//! makes this explode.

use oxiephemeris_bodies::frames::{
    fw_angles_iau2006, gcrs_to_true_of_date, mean_obliquity_iau2006, mean_of_date_to_ecliptic,
    nutation_iau2000a_truncated, nutation_matrix, precession_bias_matrix, true_of_date_to_ecliptic,
};
use oxiephemeris_bodies::math::{cartesian_to_spherical, r1, r2, r3, spherical_to_cartesian, Mat3};

const ARCSEC_TO_RAD: f64 = std::f64::consts::PI / (180.0 * 3600.0);

/// Max absolute element difference between two matrices.
fn max_diff(a: &Mat3, b: &Mat3) -> f64 {
    let mut worst = 0.0_f64;
    for (row_a, row_b) in a.0.iter().zip(b.0.iter()) {
        for (x, y) in row_a.iter().zip(row_b.iter()) {
            worst = worst.max((x - y).abs());
        }
    }
    worst
}

/// Max absolute element of `m^T m - I` and `|det - 1|`.
fn orthonormality_error(m: &Mat3) -> (f64, f64) {
    let gram = m.transpose().mul(m);
    let residual = max_diff(&gram, &Mat3::IDENTITY);
    let det_err = (m.det() - 1.0).abs();
    (residual, det_err)
}

const EPOCHS: [f64; 9] = [-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0];

#[test]
fn rotation_matrices_are_orthonormal() {
    for &angle in &[-3.0_f64, -0.7, 0.0, 1e-8, 0.3, 2.9] {
        for m in [r1(angle), r2(angle), r3(angle)] {
            let (residual, det_err) = orthonormality_error(&m);
            assert!(residual < 1e-15, "R residual {residual:e} at {angle}");
            assert!(det_err < 1e-15, "R det error {det_err:e} at {angle}");
        }
    }
}

#[test]
fn composed_frame_matrices_are_orthonormal() {
    for &t in &EPOCHS {
        for (name, m) in [
            ("PB", precession_bias_matrix(t)),
            ("N", nutation_matrix(t)),
            ("NPB", gcrs_to_true_of_date(t)),
            ("ECL_MEAN", mean_of_date_to_ecliptic(t)),
            ("ECL_TRUE", true_of_date_to_ecliptic(t)),
        ] {
            let (residual, det_err) = orthonormality_error(&m);
            assert!(residual < 1e-15, "{name} residual {residual:e} at t={t}");
            assert!(det_err < 1e-15, "{name} det error {det_err:e} at t={t}");
        }
    }
}

#[test]
fn mean_obliquity_at_j2000_is_exact() {
    let expected = 84_381.406 * ARCSEC_TO_RAD;
    let got = mean_obliquity_iau2006(0.0);
    // Identical up to one rounding of the arcsec->rad product.
    assert!(
        (got - expected).abs() <= 2.0 * f64::EPSILON * expected,
        "eps_A(0) = {got:.18} rad, expected {expected:.18} rad"
    );
}

#[test]
fn precession_bias_at_j2000_is_the_frame_bias() {
    let pb0 = precession_bias_matrix(0.0);
    let mut largest_off_diag = 0.0_f64;
    for (i, row) in pb0.0.iter().enumerate() {
        for (j, &element) in row.iter().enumerate() {
            if i == j {
                continue;
            }
            let magnitude = element.abs();
            // Bias angles are tens of mas: well below 2e-6 rad ...
            assert!(magnitude < 2e-6, "PB(0)[{i}][{j}] = {element:e} too large");
            largest_off_diag = largest_off_diag.max(magnitude);
        }
    }
    // ... but PB(0) must NOT be the identity: the frame bias is real.
    assert!(
        largest_off_diag > 1e-8,
        "PB(0) looks like the identity; frame bias missing (max off-diag {largest_off_diag:e})"
    );
}

#[test]
fn fw_angles_at_j2000_match_tn36_constants() {
    // TN36 eq. (5.40) constant terms.
    let fw = fw_angles_iau2006(0.0);
    let cases = [
        (fw.gamma_bar_rad, -0.052_928),
        (fw.phi_bar_rad, 84_381.412_819),
        (fw.psi_bar_rad, -0.041_775),
        (fw.eps_a_rad, 84_381.406),
    ];
    for (got_rad, expected_arcsec) in cases {
        let expected = expected_arcsec * ARCSEC_TO_RAD;
        assert!(
            (got_rad - expected).abs() <= 2.0 * f64::EPSILON * expected.abs(),
            "FW angle {got_rad:.18} rad vs {expected:.18} rad"
        );
    }
}

#[test]
fn nutation_envelope_1800_to_2200() {
    // |dpsi| < 20", |deps| < 12" for all dates 1800..2200 (5-day steps).
    let mut k = 0_i32;
    loop {
        let t = -2.0 + f64::from(k) * (5.0 / 36_525.0);
        if t > 2.0 {
            break;
        }
        let nut = nutation_iau2000a_truncated(t);
        let dpsi_arcsec = nut.dpsi_rad / ARCSEC_TO_RAD;
        let deps_arcsec = nut.deps_rad / ARCSEC_TO_RAD;
        assert!(dpsi_arcsec.abs() < 20.0, "dpsi = {dpsi_arcsec}\" at t={t}");
        assert!(deps_arcsec.abs() < 12.0, "deps = {deps_arcsec}\" at t={t}");
        k += 1;
    }
}

#[test]
fn npb_equals_nutation_times_precession_bias() {
    for &t in &EPOCHS {
        let combined = gcrs_to_true_of_date(t);
        let product = nutation_matrix(t).mul(&precession_bias_matrix(t));
        let diff = max_diff(&combined, &product);
        assert!(diff < 1e-15, "NPB vs N*PB diff {diff:e} at t={t}");
    }
}

#[test]
fn ecliptic_rotations_move_the_pole_correctly() {
    // The mean ecliptic pole, expressed in the mean equatorial frame of
    // date, is (0, -sin eps_A, cos eps_A)^T rotated back: check that
    // R1(eps_A) maps it to (0, 0, 1).
    for &t in &EPOCHS {
        let eps = mean_obliquity_iau2006(t);
        let pole_equatorial = [0.0, -eps.sin(), eps.cos()];
        let pole_ecliptic = mean_of_date_to_ecliptic(t).apply(pole_equatorial);
        assert!(pole_ecliptic[0].abs() < 1e-15, "x {:e}", pole_ecliptic[0]);
        assert!(pole_ecliptic[1].abs() < 1e-15, "y {:e}", pole_ecliptic[1]);
        assert!(
            (pole_ecliptic[2] - 1.0).abs() < 1e-15,
            "z {:e}",
            pole_ecliptic[2]
        );
        // True variant differs from the mean one by the (small) nutation
        // in obliquity only.
        let diff = max_diff(&true_of_date_to_ecliptic(t), &mean_of_date_to_ecliptic(t));
        assert!(diff < 1e-4, "true vs mean ecliptic diff {diff:e} at t={t}");
    }
}

#[test]
fn spherical_cartesian_round_trip() {
    let samples = [
        (0.0, 0.0, 1.0),
        (1.234, -0.5, 2.5),
        (-2.9, 1.2, 0.001),
        (3.0, -1.55, 12_345.678),
    ];
    for &(lon, lat, radius) in &samples {
        let v = spherical_to_cartesian(lon, lat, radius);
        let (lon2, lat2, r2v) = cartesian_to_spherical(v);
        assert!((lon - lon2).abs() < 1e-13, "lon {lon} vs {lon2}");
        assert!((lat - lat2).abs() < 1e-13, "lat {lat} vs {lat2}");
        assert!(
            ((radius - r2v) / radius).abs() < 1e-14,
            "r {radius} vs {r2v}"
        );
    }
    let (lon0, lat0, r0) = cartesian_to_spherical([0.0, 0.0, 0.0]);
    assert!(lon0.abs() < 1e-300 && lat0.abs() < 1e-300 && r0.abs() < 1e-300);
}

// ---------------------------------------------------------------------------
// Cross-check: equinox-based IAU 2006 precession x explicit frame bias.
// ---------------------------------------------------------------------------

/// Equinox-based precession angles, IAU 2006 (P03): Capitaine, Wallace &
/// Chapront (2003), A&A 412, 567, eq. (40). Arcseconds -> radians.
fn equinox_angles_p03(t: f64) -> (f64, f64, f64) {
    let zeta = 2.650_545
        + t * (2_306.083_227
            + t * (0.298_849_9
                + t * (0.018_018_28 + t * (-0.000_005_971 + t * (-0.000_000_317_3)))));
    let z = -2.650_545
        + t * (2_306.077_181
            + t * (1.092_734_8
                + t * (0.018_268_37 + t * (-0.000_028_596 + t * (-0.000_000_290_4)))));
    let theta = t
        * (2_004.191_903
            + t * (-0.429_493_4
                + t * (-0.041_822_64 + t * (-0.000_007_089 + t * (-0.000_000_127_4)))));
    (
        zeta * ARCSEC_TO_RAD,
        z * ARCSEC_TO_RAD,
        theta * ARCSEC_TO_RAD,
    )
}

/// Equinox-based precession matrix `P = R3(-z_A) . R2(theta_A) . R3(-zeta_A)`
/// (J2000 mean equator/equinox -> mean equator/equinox of date; classical
/// composition, cf. Capitaine et al. 2003 §4.1 / Lieske et al. 1977).
fn equinox_precession_p03(t: f64) -> Mat3 {
    let (zeta, z, theta) = equinox_angles_p03(t);
    r3(-z).mul(&r2(theta)).mul(&r3(-zeta))
}

/// GCRS frame-bias matrix `B = R1(-eta_0) . R2(xi_0) . R3(d_alpha_0)` with
/// the ICRS pole offsets `xi_0 = -0.0166170"`, `eta_0 = -0.0068192"`
/// (TN36 eq. 5.33) and the equinox offset `d_alpha_0 = -0.01460"`
/// (TN36 eq. 5.21); composition per TN36 fig. 5.1.
fn frame_bias_tn36() -> Mat3 {
    let d_alpha_0 = -0.014_60 * ARCSEC_TO_RAD;
    let xi_0 = -0.016_617_0 * ARCSEC_TO_RAD;
    let eta_0 = -0.006_819_2 * ARCSEC_TO_RAD;
    r1(-eta_0).mul(&r2(xi_0)).mul(&r3(d_alpha_0))
}

#[test]
fn fukushima_williams_matches_equinox_based_precession_times_bias() {
    let bias = frame_bias_tn36();
    let mut worst = 0.0_f64;
    let mut worst_t = 0.0_f64;
    let mut k = -200_i32;
    while k <= 200 {
        let t = f64::from(k) / 100.0;
        let pb_fw = precession_bias_matrix(t);
        let pb_eq = equinox_precession_p03(t).mul(&bias);
        let diff = max_diff(&pb_fw, &pb_eq);
        if diff > worst {
            worst = diff;
            worst_t = t;
        }
        k += 1;
    }
    println!("FW vs equinox-based PB: max element diff {worst:e} at t={worst_t}");
    assert!(
        worst < 5e-10,
        "FW and equinox-based bias-precession disagree: {worst:e} at t={worst_t}"
    );
}

#[test]
fn frame_bias_first_order_structure() {
    // First-order layout of B (TN36 eq. 5.20/5.21/5.33): check PB(0)
    // against the explicit bias matrix element by element.
    let diff = max_diff(&precession_bias_matrix(0.0), &frame_bias_tn36());
    // FW angles reproduce the bias to ~1 microarcsecond (5e-12 rad); allow
    // a wide margin over that.
    assert!(diff < 5e-10, "PB(0) vs B diff {diff:e}");
}
