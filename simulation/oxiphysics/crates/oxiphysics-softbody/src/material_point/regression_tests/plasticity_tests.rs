//! F4175 / F4176 / F4174: snow clamping, von Mises and Drucker-Prager return
//! mappings on top of the real SVD and polar decomposition.

use super::*;

#[test]
fn f4174_von_mises_return_mapping_keeps_the_rotation_of_fe() {
    // The mapping scales the deviatoric part of the stretch S in Fe = R S and keeps R,
    // which needs a correct polar factor also for symmetric (unrotated) input.
    let (mu, lambda) = (1.0e5, 2.0e5);
    let stretch = mat3_diag([1.4, 1.0, 0.9]);
    let deviator_norm = |s: [[f64; 3]; 3]| {
        let mean = mat3_trace(s) / 3.0;
        frobenius(mat3_add(s, mat3_scale(mat3_identity(), -mean)))
    };
    for rotation in [
        mat3_identity(),
        rot_z(0.6),
        mat3_mul(rot_x(0.4), rot_y(-0.9)),
    ] {
        let fe = mat3_mul(rotation, stretch);
        let out = von_mises_return_mapping(fe, mu, lambda, 1.0e3);
        assert!(
            max_diff(out, fe) > 1e-3,
            "trial state must exceed the yield stress"
        );
        let (r_out, s_out) = polar_decomp(out);
        assert!(max_diff(r_out, rotation) < 1e-10, "R = {r_out:?}");
        assert!((mat3_trace(s_out) - mat3_trace(stretch)).abs() < 1e-10);
        assert!(deviator_norm(s_out) < 0.5 * deviator_norm(stretch));
    }
    // Below the yield stress nothing changes.
    let fe = mat3_mul(rot_z(0.6), stretch);
    assert_eq!(von_mises_return_mapping(fe, mu, lambda, 1.0e12), fe);
}

#[test]
fn f4175_snow_update_leaves_gradients_inside_the_band_untouched() {
    // sigma = (1.005, 1, 1) lies inside [0.975, 1.0075]; the fake SVD returned
    // diag(1.010025, 1, 1) here.
    let fe = mat3_diag([1.005, 1.0, 1.0]);
    let (fe_new, fp_new) = snow_plastic_update(fe, mat3_identity(), fe, THETA_C, THETA_S);
    assert_eq!(fe_new, fe);
    assert!(max_diff(fp_new, mat3_identity()) < 1e-14, "Fp = {fp_new:?}");

    // The same holds for a rotated gradient with all stretches inside the band.
    let rotated = mat3_mul(rot_z(0.4), mat3_diag([1.005, 0.99, 1.0]));
    let (fe_new, _) = snow_plastic_update(rotated, mat3_identity(), rotated, THETA_C, THETA_S);
    assert_eq!(fe_new, rotated);
}

#[test]
fn f4175_snow_update_clamps_singular_values_and_keeps_the_rotation() {
    let rotation = rot_z(0.4);
    let fe = mat3_mul(rotation, mat3_diag([1.05, 0.9, 1.0]));
    // A pre-existing plastic part: F = Fe Fp.
    let f_total = mat3_mul(fe, mat3_diag([1.02, 1.0, 0.98]));
    let (fe_new, fp_new) = snow_plastic_update(fe, mat3_identity(), f_total, THETA_C, THETA_S);
    let expected = mat3_mul(rotation, mat3_diag([1.0 + THETA_S, 1.0 - THETA_C, 1.0]));
    assert!(max_diff(fe_new, expected) < 1e-12, "Fe = {fe_new:?}");
    // F = Fe Fp stays consistent.
    assert!(max_diff(mat3_mul(fe_new, fp_new), f_total) < 1e-12);
    // The plastic part carries the clamped excess: det Fp = det F / det Fe.
    let jp = mat3_det(fp_new);
    let expected_jp = mat3_det(f_total) / mat3_det(fe_new);
    assert!((jp - expected_jp).abs() < 1e-12);
}

#[test]
fn f4175_snow_update_repairs_inverted_gradients() {
    let fe = mat3_diag([1.0, 1.0, -1.0]);
    let (fe_new, fp_new) = snow_plastic_update(fe, mat3_identity(), fe, THETA_C, THETA_S);
    assert!(mat3_det(fe_new) > 0.0, "Fe still inverted: {fe_new:?}");
    let (_, sigma, _) = mat3_svd(fe_new);
    assert!((sigma[2] - (1.0 - THETA_C)).abs() < 1e-12, "σ = {sigma:?}");
    assert!(max_diff(mat3_mul(fe_new, fp_new), fe) < 1e-12);
}

#[test]
fn f4175_snow_update_always_lands_inside_the_band_for_random_gradients() {
    let mut rng = TestRng::new(0x5A0_0F4175);
    for _ in 0..300 {
        let fe = mat3_add(mat3_identity(), mat3_scale(rng.mat3(-1.0, 1.0), 0.2));
        let (fe_new, fp_new) = snow_plastic_update(fe, mat3_identity(), fe, THETA_C, THETA_S);
        let (_, sigma, _) = mat3_svd(fe_new);
        for s in sigma {
            assert!(
                (1.0 - THETA_C - 1e-12..=1.0 + THETA_S + 1e-12).contains(&s),
                "σ {s} outside the band"
            );
        }
        assert!(max_diff(mat3_mul(fe_new, fp_new), fe) < 1e-10);
    }
}

#[test]
fn f4175_update_deformation_gradient_keeps_in_band_snow_stretch_elastic() {
    let dt = 0.01;
    let mut p = MpmParticle::new(
        [0.5; 3],
        [0.0; 3],
        1.0,
        1.0e-3,
        MaterialType::Snow,
        58333.0,
        38194.0,
    );
    // I + dt C = diag(1.005, 1, 1).
    p.affine_c = mat3_diag([0.5, 0.0, 0.0]);
    let mut particles = [p];
    update_deformation_gradient(&mut particles, dt, THETA_C, THETA_S);
    let p = &particles[0];
    assert!(
        max_diff(p.fe, mat3_diag([1.005, 1.0, 1.0])) < 1e-14,
        "Fe = {:?}",
        p.fe
    );
    assert!(max_diff(p.fp, mat3_identity()) < 1e-14, "Fp = {:?}", p.fp);
    assert!(max_diff(p.deformation_gradient, p.fe) < 1e-14);
}

const SAND_MU: f64 = 1.0e5;
const SAND_LAMBDA: f64 = 2.0e5;

/// Friction constant `α = √(2/3) · 2 sin φ / (3 - sin φ)` of the cone.
fn cone_alpha(friction_deg: f64) -> f64 {
    let sin_phi = friction_deg.to_radians().sin();
    (2.0_f64 / 3.0).sqrt() * 2.0 * sin_phi / (3.0 - sin_phi)
}

/// Yield function `‖dev ε‖ + α (3λ + 2μ)/(2μ) tr ε` of a gradient (`<= 0` is elastic).
fn drucker_prager_yield(fe: [[f64; 3]; 3], mu: f64, lambda: f64, friction_deg: f64) -> f64 {
    let (_, sigma, _) = mat3_svd(fe);
    let eps = sigma.map(|s| s.abs().ln());
    let tr: f64 = eps.iter().sum();
    let dev = eps.map(|e| e - tr / 3.0);
    len3(dev) + cone_alpha(friction_deg) * (3.0 * lambda + 2.0 * mu) / (2.0 * mu) * tr
}

/// Klar et al. (2016) return mapping on known principal stretches, written
/// straight from the paper and independent of any SVD.
fn reference_stretches(stretch: [f64; 3], mu: f64, lambda: f64, friction_deg: f64) -> [f64; 3] {
    let eps = stretch.map(f64::ln);
    let tr: f64 = eps.iter().sum();
    if tr > 0.0 {
        return [1.0; 3];
    }
    let dev = eps.map(|e| e - tr / 3.0);
    let dev_norm = len3(dev);
    let delta_gamma =
        dev_norm + (3.0 * lambda + 2.0 * mu) / (2.0 * mu) * cone_alpha(friction_deg) * tr;
    if dev_norm == 0.0 || delta_gamma <= 0.0 {
        return stretch;
    }
    std::array::from_fn(|k| (eps[k] - delta_gamma * dev[k] / dev_norm).exp())
}

#[test]
fn f4176_hydrostatic_tension_projects_to_the_rotation_of_fe() {
    let rotation = rot_z(0.5);
    // The old code returned the identity here and so dropped the rotation.
    let out = drucker_prager_return_mapping(mat3_scale(rotation, 1.1), SAND_MU, SAND_LAMBDA, 30.0);
    assert!(max_diff(out, rotation) < 1e-12, "Fe = {out:?}");
    assert!((mat3_det(out) - 1.0).abs() < 1e-12);

    // Any state with tr(ln sigma) > 0 resets all stretches to one.
    let fe = mat3_mul(rotation, mat3_diag([1.4, 0.9, 1.2]));
    let out = drucker_prager_return_mapping(fe, SAND_MU, SAND_LAMBDA, 30.0);
    assert!(max_diff(out, rotation) < 1e-12, "Fe = {out:?}");
}

#[test]
fn f4176_compression_inside_the_cone_is_returned_unchanged() {
    let states = [
        mat3_scale(mat3_identity(), 0.9),
        mat3_scale(rot_z(0.3), 0.7),
        mat3_mul(rot_z(0.3), mat3_diag([0.8, 0.82, 0.78])),
        mat3_identity(),
    ];
    for fe in states {
        assert!(drucker_prager_yield(fe, SAND_MU, SAND_LAMBDA, 30.0) <= 1e-12);
        let out = drucker_prager_return_mapping(fe, SAND_MU, SAND_LAMBDA, 30.0);
        assert_eq!(out, fe, "elastic state must be returned bit-for-bit");
    }
}

#[test]
fn f4176_yielding_state_is_projected_onto_the_cone_without_changing_the_volume() {
    let rotation = rot_z(0.3);
    let stretch = [0.5, 1.5, 1.0];
    let fe = mat3_mul(rotation, mat3_diag(stretch));
    assert!(drucker_prager_yield(fe, SAND_MU, SAND_LAMBDA, 30.0) > 0.1);

    let out = drucker_prager_return_mapping(fe, SAND_MU, SAND_LAMBDA, 30.0);
    // The principal directions are untouched: Rᵀ out is diagonal ...
    let in_frame = mat3_mul(mat3_transpose(rotation), out);
    let expected = reference_stretches(stretch, SAND_MU, SAND_LAMBDA, 30.0);
    assert!(
        max_diff(in_frame, mat3_diag(expected)) < 1e-12,
        "Rᵀ Fe = {in_frame:?}, expected diag{expected:?}"
    );
    // ... the state sits exactly on the yield surface ...
    assert!(drucker_prager_yield(out, SAND_MU, SAND_LAMBDA, 30.0).abs() < 1e-10);
    // ... and tr(ln sigma) is preserved (the old code subtracted alpha/3 from it).
    assert!((mat3_det(out) - mat3_det(fe)).abs() < 1e-12);
}

#[test]
fn f4176_return_mapping_uses_the_lame_parameters() {
    let stretch = [0.5, 1.5, 1.0];
    let fe = mat3_mul(rot_z(0.3), mat3_diag(stretch));
    let hard = drucker_prager_return_mapping(fe, SAND_MU, SAND_LAMBDA, 30.0);
    let soft = drucker_prager_return_mapping(fe, SAND_MU, 0.0, 30.0);
    assert!(
        max_diff(hard, soft) > 1e-2,
        "(3 lambda + 2 mu)/(2 mu) must change the plastic multiplier"
    );
    for (out, lambda) in [(hard, SAND_LAMBDA), (soft, 0.0)] {
        let expected = reference_stretches(stretch, SAND_MU, lambda, 30.0);
        let in_frame = mat3_mul(mat3_transpose(rot_z(0.3)), out);
        for k in 0..3 {
            assert!(
                (in_frame[k][k] - expected[k]).abs() < 1e-12,
                "lambda = {lambda}"
            );
        }
        assert!(drucker_prager_yield(out, SAND_MU, lambda, 30.0).abs() < 1e-10);
    }
    // A stiffer shear modulus makes the same strain more plastic as well.
    let stiff = drucker_prager_return_mapping(fe, 4.0 * SAND_MU, SAND_LAMBDA, 30.0);
    let expected = reference_stretches(stretch, 4.0 * SAND_MU, SAND_LAMBDA, 30.0);
    let in_frame = mat3_mul(mat3_transpose(rot_z(0.3)), stiff);
    for k in 0..3 {
        assert!((in_frame[k][k] - expected[k]).abs() < 1e-12);
    }
}

#[test]
fn f4176_zero_friction_removes_all_deviatoric_strain() {
    let fe = mat3_mul(rot_z(0.3), mat3_diag([0.5, 1.5, 1.0]));
    let out = drucker_prager_return_mapping(fe, SAND_MU, SAND_LAMBDA, 0.0);
    let scale = mat3_det(fe).cbrt();
    assert!(
        max_diff(out, mat3_scale(rot_z(0.3), scale)) < 1e-12,
        "Fe = {out:?}"
    );
}

#[test]
fn f4176_return_mapping_is_an_idempotent_projection_for_random_states() {
    let mut rng = TestRng::new(0xD2A6_4176);
    for case in 0..500 {
        let q1 = random_rotation(&mut rng);
        let q2 = random_rotation(&mut rng);
        let stretch = rng.vec3(0.4, 1.4);
        let fe = mat3_mul(mat3_mul(q1, mat3_diag(stretch)), q2);
        let friction = rng.range(5.0, 50.0);
        let out = drucker_prager_return_mapping(fe, SAND_MU, SAND_LAMBDA, friction);

        assert!(out.iter().flatten().all(|x| x.is_finite()), "#{case}");
        // Never left outside the yield surface.
        let f_out = drucker_prager_yield(out, SAND_MU, SAND_LAMBDA, friction);
        assert!(
            f_out <= 1e-9,
            "#{case}: yield function {f_out} after return"
        );
        // Projection: applying it again changes nothing.
        let again = drucker_prager_return_mapping(out, SAND_MU, SAND_LAMBDA, friction);
        assert!(max_diff(again, out) < 1e-8, "#{case}: not idempotent");
        // Elastic states are untouched.
        let (_, sigma, _) = mat3_svd(fe);
        let tr: f64 = sigma.iter().map(|s| s.abs().ln()).sum();
        if tr <= 0.0 && drucker_prager_yield(fe, SAND_MU, SAND_LAMBDA, friction) < -1e-9 {
            assert_eq!(out, fe, "#{case}: elastic state modified");
        }
    }
}

#[test]
fn f4176_inverted_and_singular_gradients_are_repaired_not_propagated() {
    // Inverted but elastic: the reflection is removed, the magnitude kept.
    let inverted = mat3_diag([0.9, 0.9, -0.9]);
    let out = drucker_prager_return_mapping(inverted, SAND_MU, SAND_LAMBDA, 30.0);
    assert!(
        max_diff(out, mat3_scale(mat3_identity(), 0.9)) < 1e-12,
        "Fe = {out:?}"
    );
    assert!(mat3_det(out) > 0.0);

    // Singular gradient: finite, invertible result.
    let singular = mat3_diag([1.0, 1.0, 0.0]);
    let out = drucker_prager_return_mapping(singular, SAND_MU, SAND_LAMBDA, 30.0);
    assert!(out.iter().flatten().all(|x| x.is_finite()));
    assert!(mat3_det(out) > 0.0, "Fe = {out:?}");

    // Invalid parameters never panic or produce NaN.
    let fe = mat3_diag([0.5, 1.5, 1.0]);
    for (mu, lambda, phi) in [
        (0.0, 1.0, 30.0),
        (-1.0, 1.0, 30.0),
        (1.0, f64::NAN, 30.0),
        (1.0, 1.0, f64::NAN),
    ] {
        let out = drucker_prager_return_mapping(fe, mu, lambda, phi);
        assert_eq!(out, fe);
    }
    // A NaN gradient stays NaN instead of being replaced by a valid matrix.
    let nan = mat3_diag([f64::NAN, 1.0, 1.0]);
    let out = drucker_prager_return_mapping(nan, SAND_MU, SAND_LAMBDA, 30.0);
    assert!(out[0][0].is_nan());
}

#[test]
fn f4176_update_deformation_gradient_applies_the_sand_return_mapping() {
    let mut p = MpmParticle::new(
        [0.5; 3],
        [0.0; 3],
        1.0,
        1.0e-3,
        MaterialType::Sand,
        SAND_MU,
        SAND_LAMBDA,
    );
    p.yield_param = 30.0;
    // A rotated volume-preserving shear: tr(ln sigma) = 0 puts it on the cone tip.
    p.fe = mat3_mul(rot_z(0.3), mat3_diag([0.5, 2.0, 1.0]));
    p.deformation_gradient = p.fe;
    let mut particles = [p];
    update_deformation_gradient(&mut particles, 0.0, 0.025, 0.0075);
    let p = &particles[0];
    assert!(max_diff(p.fe, rot_z(0.3)) < 1e-12, "Fe = {:?}", p.fe);
    assert!(max_diff(mat3_mul(p.fe, p.fp), p.deformation_gradient) < 1e-12);
}
