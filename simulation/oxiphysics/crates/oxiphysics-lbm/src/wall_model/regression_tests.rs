//! Regression tests for the wall-model solvers: the friction-velocity solver
//! [`u_tau_newton`] with the wall models built on it, and the Werner-Wengle model.
//!
//! The friction velocity is checked against an independent bisection oracle on the
//! `y+` form of the law of the wall and against the defining identity
//! `|u| / u_tau = f(y * u_tau / nu)`; the Werner-Wengle stress is checked against
//! its defining power law, dimensional scaling and numerical quadrature of the
//! two-layer profile.

use super::*;

const KAPPA: f64 = 0.41;
const B: f64 = 5.2;

/// `(y, nu)` pairs used to turn a Reynolds number `|u| y / nu` into dimensional data.
const GEOMETRIES: [(f64, f64); 3] = [(1e-3, 1e-5), (0.05, 1.5e-5), (2e-4, 1e-6)];

/// Independent oracle for the friction velocity: bisection on `y+ f(y+) = |u| y / nu`.
fn reference_u_tau(u: f64, y: f64, nu: f64, kappa: f64, b: f64) -> f64 {
    let re_y = u.abs() * y / nu;
    let y_plus_couette = re_y.sqrt();
    if y_plus_couette <= 11.6 {
        return y_plus_couette * nu / y;
    }
    // On the log branch h(y+) = y+ (ln y+ / kappa + b) - Re_y increases from
    // h(11.6) < 0 to h(2 Re_y) > 0.
    let mut lo = 11.6_f64;
    let mut hi = 2.0 * re_y;
    for _ in 0..300 {
        let mid = 0.5 * (lo + hi);
        if mid * (mid.ln() / kappa + b) - re_y < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi) * nu / y
}

/// Residual of the law of the wall, `|u| / u_tau - f(y * u_tau / nu)`.
fn law_residual(u: f64, y: f64, nu: f64, u_tau: f64) -> f64 {
    u.abs() / u_tau - law_of_the_wall_velocity(y * u_tau / nu, KAPPA, B)
}

/// Relative agreement helper.
fn assert_rel(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tol * expected.abs(),
        "{what}: {actual} vs {expected} (tol {tol})"
    );
}

// ---------------------------------------------------------------------------
// u_tau_newton
// ---------------------------------------------------------------------------

#[test]
fn u_tau_newton_solves_the_log_layer_cases_from_the_audit() {
    // Both cases have |u| y / nu = 1000, i.e. y+ f(y+) = 1000 -> y+ ~ 65: the
    // previous Jacobian drove u_tau to the 1e-15 clamp.
    for (u, y, nu, u_tau_expected) in [(10.0, 1e-3, 1e-5, 0.650116), (1.0, 0.1, 1e-4, 0.0650116)] {
        let u_tau = u_tau_newton(u, y, nu, KAPPA, B, 50);
        assert_rel(u_tau, u_tau_expected, 1e-5, "u_tau");
        let residual = law_residual(u, y, nu, u_tau);
        assert!(
            residual.abs() < 1e-8,
            "law-of-the-wall residual {residual} for u={u}, y={y}, nu={nu}"
        );
        assert_rel(u_tau, reference_u_tau(u, y, nu, KAPPA, B), 1e-12, "oracle");
        let y_plus_val = y * u_tau / nu;
        assert!(y_plus_val > 60.0 && y_plus_val < 70.0, "y+ = {y_plus_val}");
    }
}

#[test]
fn u_tau_newton_matches_the_oracle_over_sixteen_decades_of_reynolds_number() {
    for (y, nu) in GEOMETRIES {
        for k in 0..=76_u32 {
            let re_y = 10f64.powf(-3.0 + 0.25 * f64::from(k));
            let u = re_y * nu / y;
            // Ten iterations are enough for every Reynolds number up to 1e16.
            let u_tau = u_tau_newton(u, y, nu, KAPPA, B, 10);
            assert!(u_tau.is_finite() && u_tau > 0.0, "Re_y = {re_y}: {u_tau}");
            assert_rel(
                u_tau,
                reference_u_tau(u, y, nu, KAPPA, B),
                1e-11,
                &format!("oracle at Re_y = {re_y:e}"),
            );
            let f = law_of_the_wall_velocity(y * u_tau / nu, KAPPA, B);
            let residual = law_residual(u, y, nu, u_tau);
            assert!(
                residual.abs() < 1e-10 * (1.0 + f),
                "Re_y = {re_y:e}: residual {residual}, u+ = {f}"
            );
        }
    }
}

#[test]
fn u_tau_newton_is_exactly_couette_in_the_viscous_sublayer() {
    let (y, nu) = (1e-3, 1e-5);
    for re_y in [1e-3, 1.0, 50.0, 129.0, 134.0] {
        let u = re_y * nu / y;
        let u_tau = u_tau_newton(u, y, nu, KAPPA, B, 50);
        assert_rel(u_tau * u_tau, nu * u / y, 1e-14, "tau_w / rho");
        assert!(
            y * u_tau / nu <= 11.6,
            "Re_y = {re_y} must stay in the sublayer"
        );
    }
}

#[test]
fn u_tau_newton_returns_the_viscous_root_where_both_branches_have_one() {
    // The two branches of the law do not meet: u+ drops from 11.6 to about 11.2 at
    // y+ = 11.6, so Re_y in (129.7, 134.6) has a viscous and a log root.
    let (y, nu) = (1e-3, 1e-5);
    let u = 132.0 * nu / y;
    let u_tau = u_tau_newton(u, y, nu, KAPPA, B, 50);
    let y_plus_val = y * u_tau / nu;
    assert!(
        (y_plus_val - 132.0_f64.sqrt()).abs() < 1e-12,
        "y+ = {y_plus_val}"
    );
    assert!(law_residual(u, y, nu, u_tau).abs() < 1e-10);
}

#[test]
fn u_tau_newton_is_odd_symmetric_and_monotone_in_the_velocity() {
    let (y, nu) = (1e-3, 1e-5);
    let mut previous = 0.0_f64;
    for k in 1..=120_u32 {
        let u = 0.01 * 1.15_f64.powf(f64::from(k));
        let u_tau = u_tau_newton(u, y, nu, KAPPA, B, 50);
        assert_eq!(u_tau, u_tau_newton(-u, y, nu, KAPPA, B, 50));
        if previous > 0.0 && (u * y / nu) > 134.56 {
            assert!(u_tau > previous, "u = {u}: {u_tau} <= {previous}");
        }
        previous = u_tau;
    }
}

#[test]
fn u_tau_newton_rejects_degenerate_input_without_nan() {
    let nan = f64::NAN;
    let inf = f64::INFINITY;
    let cases = [
        (0.0, 1e-3, 1e-5, KAPPA, B),
        (10.0, 0.0, 1e-5, KAPPA, B),
        (10.0, 1e-3, 0.0, KAPPA, B),
        (10.0, 1e-3, -1e-5, KAPPA, B),
        (10.0, 1e-3, 1e-5, 0.0, B),
        (10.0, 1e-3, 1e-5, -0.41, B),
        (nan, 1e-3, 1e-5, KAPPA, B),
        (10.0, nan, 1e-5, KAPPA, B),
        (10.0, 1e-3, nan, KAPPA, B),
        (10.0, 1e-3, 1e-5, nan, B),
        (10.0, 1e-3, 1e-5, KAPPA, nan),
        (inf, 1e-3, 1e-5, KAPPA, B),
    ];
    for (u, y, nu, kappa, b) in cases {
        assert_eq!(
            u_tau_newton(u, y, nu, kappa, b, 50),
            0.0,
            "{u} {y} {nu} {kappa} {b}"
        );
    }
}

/// Log-layer solver data for a Reynolds number `|u| y / nu` (standard constants).
fn log_layer_solve(re_y: f64, y: f64, nu: f64) -> LogLawSolve {
    LogLawSolve {
        u_abs: re_y * nu / y,
        y,
        nu,
        kappa: KAPPA,
        b: B,
    }
}

#[test]
fn log_layer_newton_converges_in_a_handful_of_iterations_without_the_fallback() {
    // Reynolds numbers above the sublayer limit 11.6^2 = 134.56 up to 1e16.
    for (y, nu) in GEOMETRIES {
        for k in 0..=56_u32 {
            let re_y = 10f64.powf(2.25 + 0.25 * f64::from(k));
            let solve = log_layer_solve(re_y, y, nu);
            let couette = (nu * solve.u_abs / y).sqrt();
            let newton = solve.newton(couette, 8);
            let u_tau = newton.unwrap_or(f64::NAN);
            assert_rel(
                u_tau,
                reference_u_tau(solve.u_abs, y, nu, KAPPA, B),
                1e-11,
                &format!("Newton at Re_y = {re_y:e}"),
            );
        }
    }
}

#[test]
fn log_layer_bisection_alone_matches_the_oracle() {
    for (y, nu) in GEOMETRIES {
        for k in 0..=56_u32 {
            let re_y = 10f64.powf(2.25 + 0.25 * f64::from(k));
            let solve = log_layer_solve(re_y, y, nu);
            assert_rel(
                solve.bisection(),
                reference_u_tau(solve.u_abs, y, nu, KAPPA, B),
                1e-11,
                &format!("bisection at Re_y = {re_y:e}"),
            );
        }
    }
}

#[test]
fn u_tau_newton_is_accurate_for_every_iteration_cap() {
    // A cap too small for Newton hands over to the bracketed bisection; the result
    // is the same root for any cap, including 0.
    let (y, nu) = (1e-3, 1e-5);
    for re_y in [150.0, 1e3, 1e5, 1e9, 1e14] {
        let u = re_y * nu / y;
        let expected = reference_u_tau(u, y, nu, KAPPA, B);
        for max_iter in [0_usize, 1, 2, 3, 50] {
            let u_tau = u_tau_newton(u, y, nu, KAPPA, B, max_iter);
            assert_rel(
                u_tau,
                expected,
                1e-11,
                &format!("Re_y {re_y:e}, cap {max_iter}"),
            );
        }
    }
}

#[test]
fn u_tau_newton_settles_at_the_switch_when_the_log_law_lies_above_the_sublayer_law() {
    // kappa = 0.35, b = 6: u+(11.6) = 13.0 > 11.6, so for Re_y slightly above 11.6^2
    // the residual |u| / u_tau - f changes sign exactly at y+ = 11.6 (no root).
    let (kappa, b) = (0.35, 6.0);
    let (y, nu) = (1e-3, 1e-5);
    let u = 140.0 * nu / y;
    let u_tau = u_tau_newton(u, y, nu, kappa, b, 50);
    assert_rel(u_tau, 11.6 * nu / y, 1e-12, "switch u_tau");
    // The bracketed fallback reaches the same switch point.
    let solve = LogLawSolve {
        u_abs: u,
        y,
        nu,
        kappa,
        b,
    };
    assert_rel(
        solve.bisection(),
        11.6 * nu / y,
        1e-12,
        "bisection at the switch",
    );
    let below =
        u / (u_tau * (1.0 - 1e-9)) - law_of_the_wall_velocity(11.6 * (1.0 - 1e-9), kappa, b);
    let above =
        u / (u_tau * (1.0 + 1e-9)) - law_of_the_wall_velocity(11.6 * (1.0 + 1e-9), kappa, b);
    assert!(below > 0.0 && above < 0.0, "sign change: {below} / {above}");
}

// ---------------------------------------------------------------------------
// Wall models built on u_tau_newton
// ---------------------------------------------------------------------------

#[test]
fn wall_models_use_the_converged_friction_velocity() {
    let (u, y, nu) = (10.0, 1e-3, 1e-5);
    let u_tau = reference_u_tau(u, y, nu, KAPPA, B);
    let tau = u_tau * u_tau;

    assert_rel(
        WallModeledLes::new(nu).wall_stress(u, y),
        tau,
        1e-10,
        "WMLES stress",
    );
    assert_rel(
        WallModeledLes::new(nu).y_plus_at_match(y, u),
        y * u_tau / nu,
        1e-10,
        "WMLES y+",
    );
    assert_rel(
        EquilibriumWallModel::new(nu).wall_stress(u, y),
        tau,
        1e-10,
        "EWM stress",
    );
    assert_rel(
        LogLawWallModel::new(KAPPA, B, nu).wall_shear_stress_newton(u, y),
        tau,
        1e-10,
        "log-law stress",
    );
    // The log law evaluated at the solved u_tau reproduces the matching velocity.
    assert_rel(
        EquilibriumWallModel::new(nu).predicted_velocity(u_tau, y),
        u,
        1e-10,
        "predicted velocity",
    );
    let bc = WallFunctionBc::new(y, nu, LogLawWallModel::new(KAPPA, B, nu));
    assert_rel(
        bc.wall_velocity_bc(u),
        u,
        1e-10,
        "wall-function BC velocity",
    );

    // Eddy viscosity kappa y u_tau D(y+) of the WMLES and EWM (molecular part added).
    let yp = y * u_tau / nu;
    let nu_t = KAPPA * y * u_tau * van_driest_damping(yp, 26.0);
    assert_rel(
        WallModeledLes::new(nu).eddy_viscosity(y, u),
        nu_t,
        1e-10,
        "WMLES nu_t",
    );
    assert_rel(
        EquilibriumWallModel::new(nu).effective_viscosity(u, y, 0.1),
        nu + nu_t,
        1e-10,
        "EWM nu_eff",
    );
}

#[test]
fn non_equilibrium_wall_model_is_a_moderate_correction_of_the_equilibrium_stress() {
    let (u, y, nu) = (10.0, 1e-3, 1e-5);
    let u_tau_eq = reference_u_tau(u, y, nu, KAPPA, B);
    let tau_eq = u_tau_eq * u_tau_eq;
    let model = NonEquilibriumWallModel::new(nu, 0.1);

    assert_rel(
        model.wall_stress(u, y, 0.0),
        tau_eq,
        1e-10,
        "no pressure gradient",
    );

    // dp_ds / rho = +-100 m/s^2 shifts the matching velocity by dp_ds * y / u_tau.
    for dp_ds in [100.0_f64, -100.0] {
        let u_eff = u - dp_ds * y / u_tau_eq;
        let expected = reference_u_tau(u_eff, y, nu, KAPPA, B).powi(2);
        let tau = model.wall_stress(u, y, dp_ds);
        assert_rel(tau, expected, 1e-10, "pressure-gradient stress");
        let ratio = tau / tau_eq;
        if dp_ds > 0.0 {
            assert!(
                ratio > 0.95 && ratio < 1.0,
                "adverse gradient ratio {ratio}"
            );
        } else {
            assert!(
                ratio > 1.0 && ratio < 1.05,
                "favourable gradient ratio {ratio}"
            );
        }
    }
}

#[test]
fn wall_model_switcher_selects_the_branch_by_the_equilibrium_y_plus() {
    let switcher = WallModelSwitcher::new(1e-5, 5.0);
    // Viscous: Re_y = 10, y+ = 3.16 <= 5 -> equilibrium stress nu u / y = 0.1.
    assert_rel(
        switcher.compute_stress(1.0, 1e-4),
        0.1,
        1e-12,
        "viscous stress",
    );
    // Log layer: y+ = 65 > 5 -> non-equilibrium branch, which equals the
    // equilibrium stress for dp_ds = 0.
    let u_tau = reference_u_tau(10.0, 1e-3, 1e-5, KAPPA, B);
    assert_rel(
        switcher.compute_stress(10.0, 1e-3),
        u_tau * u_tau,
        1e-10,
        "log stress",
    );
}

#[test]
fn thermal_wall_model_heat_flux_follows_the_solved_friction_velocity() {
    let (u, y, nu, pr_t) = (10.0, 1e-3, 1e-5, 0.85);
    let u_tau = reference_u_tau(u, y, nu, KAPPA, B);
    // T+ = Pr_t * u+ = Pr_t * |u| / u_tau  =>  q = u_tau^2 * dT / (Pr_t * |u|).
    let expected = u_tau * u_tau * (300.0 - 350.0) / (pr_t * u);
    let q = ThermalWallModel::new(nu, pr_t).wall_heat_flux(300.0, 350.0, y, u);
    assert_rel(q, expected, 1e-9, "wall heat flux");
    assert!(q < 0.0, "heat flows toward the colder wall: {q}");
}

// ---------------------------------------------------------------------------
// Werner-Wengle
// ---------------------------------------------------------------------------

#[test]
fn werner_wengle_switch_is_the_intersection_of_the_two_laws() {
    let y_plus_m = WernerWengle::y_plus_switch();
    assert!(y_plus_m > 11.80 && y_plus_m < 11.82, "y+_m = {y_plus_m}");
    // A * y+^B = y+ exactly at the switch.
    assert_rel(
        WernerWengle::A * y_plus_m.powf(WernerWengle::B),
        y_plus_m,
        1e-13,
        "u+ at the switch",
    );
}

#[test]
fn werner_wengle_viscous_branch_is_couette() {
    let tau = WernerWengle::wall_shear_stress(1e-4, 1e-3, 1e-4);
    assert_rel(tau, 1e-4 * 1e-4 / 1e-3, 1e-13, "tau_w / rho");
    // Odd symmetry: the magnitude of the stress does not depend on the sign of u.
    assert_eq!(tau, WernerWengle::wall_shear_stress(-1e-4, 1e-3, 1e-4));
}

#[test]
fn werner_wengle_power_law_branch_satisfies_the_defining_identity() {
    let (y, nu) = (1e-3, 1e-5);
    for re_y in [200.0, 500.0, 1e3, 1e4, 1e6] {
        let u = re_y * nu / y;
        let tau = WernerWengle::wall_shear_stress(u, y, nu);
        let u_tau = tau.sqrt();
        let u_plus = u / u_tau;
        let power_law = WernerWengle::A * (y * u_tau / nu).powf(WernerWengle::B);
        assert_rel(
            u_plus,
            power_law,
            1e-12,
            &format!("u/u_tau at Re_y = {re_y}"),
        );
        assert!(y * u_tau / nu > WernerWengle::y_plus_switch());
    }
}

#[test]
fn werner_wengle_stress_is_continuous_and_monotone_across_the_switch() {
    let (y, nu) = (1e-3, 1e-5);
    let switch_re = WernerWengle::y_plus_switch().powi(2);
    let tau_below = WernerWengle::wall_shear_stress(switch_re * (1.0 - 1e-12) * nu / y, y, nu);
    let tau_above = WernerWengle::wall_shear_stress(switch_re * (1.0 + 1e-12) * nu / y, y, nu);
    assert!(
        (tau_above - tau_below).abs() < 1e-10 * tau_below,
        "jump {} at the switch",
        tau_above - tau_below
    );
    let mut previous = 0.0_f64;
    for k in 1..=200_u32 {
        let u = 0.05 * 1.05_f64.powf(f64::from(k));
        let tau = WernerWengle::wall_shear_stress(u, y, nu);
        assert!(tau > previous, "u = {u}: {tau} <= {previous}");
        previous = tau;
    }
}

#[test]
fn werner_wengle_stress_has_the_dimensions_of_velocity_squared() {
    // Rescaling velocity by a, length by b and kinematic viscosity by a*b must scale
    // tau_w / rho by a^2.  The former (8 nu / y)^(1/4) (u/2)^(3/4) scaled like a.
    let bases = [
        (1e-4, 1e-3, 1e-4),
        (10.0, 1e-3, 1e-5),
        (1.0, 1e-2, 1e-5),
        (0.3, 5e-3, 2e-5),
    ];
    for (u, y, nu) in bases {
        let tau = WernerWengle::wall_shear_stress(u, y, nu);
        let tau_cell = WernerWengle::wall_shear_stress_cell_averaged(u, 2.0 * y, nu);
        for (a, b) in [(0.5, 2.0), (3.0, 0.1), (10.0, 7.0)] {
            let scaled = WernerWengle::wall_shear_stress(a * u, b * y, a * b * nu);
            assert_rel(
                scaled,
                a * a * tau,
                1e-12,
                &format!("pointwise a={a} b={b}"),
            );
            let scaled_cell =
                WernerWengle::wall_shear_stress_cell_averaged(a * u, 2.0 * b * y, a * b * nu);
            assert_rel(
                scaled_cell,
                a * a * tau_cell,
                1e-12,
                &format!("cell a={a} b={b}"),
            );
        }
    }
}

#[test]
fn werner_wengle_agrees_with_the_log_law_in_the_overlap_region() {
    // The 1/7 power law is a fit of the log law: u_tau within 3% for 30 <= y+ <= 1000.
    let (y, nu) = (1e-3, 1e-5);
    for y_plus_target in [30.0, 100.0, 300.0, 1000.0] {
        let u_tau_ref = y_plus_target * nu / y;
        let u = u_tau_ref * law_of_the_wall_velocity(y_plus_target, KAPPA, B);
        let u_tau_ww = WernerWengle::wall_shear_stress(u, y, nu).sqrt();
        let ratio = u_tau_ww / u_tau_newton(u, y, nu, KAPPA, B, 50);
        assert!(
            (ratio - 1.0).abs() < 0.03,
            "y+ = {y_plus_target}: ratio {ratio}"
        );
    }
}

#[test]
fn werner_wengle_rejects_degenerate_input() {
    let nan = f64::NAN;
    for (u, y, nu) in [
        (0.0, 1e-3, 1e-5),
        (1.0, 0.0, 1e-5),
        (1.0, -1e-3, 1e-5),
        (1.0, 1e-3, 0.0),
        (1.0, 1e-3, -1e-5),
        (nan, 1e-3, 1e-5),
        (1.0, nan, 1e-5),
        (1.0, 1e-3, nan),
        (f64::INFINITY, 1e-3, 1e-5),
    ] {
        assert_eq!(
            WernerWengle::wall_shear_stress(u, y, nu),
            0.0,
            "{u} {y} {nu}"
        );
        assert_eq!(
            WernerWengle::wall_shear_stress_cell_averaged(u, y, nu),
            0.0,
            "cell: {u} {y} {nu}"
        );
    }
}

/// Composite Simpson rule with `n` (even) intervals.
fn simpson<F: Fn(f64) -> f64>(f: F, lo: f64, hi: f64, n: usize) -> f64 {
    let step = (hi - lo) / n as f64;
    let mut sum = f(lo) + f(hi);
    for i in 1..n {
        let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += weight * f(lo + i as f64 * step);
    }
    sum * step / 3.0
}

/// Mean over `[0, h]` of the two-layer Werner-Wengle velocity profile with friction
/// velocity `u_tau`, by quadrature (log-spaced for the power-law layer).
fn werner_wengle_cell_mean(u_tau: f64, h: f64, nu: f64) -> f64 {
    let (a, b) = (WernerWengle::A, WernerWengle::B);
    let y_m = WernerWengle::y_plus_switch() * nu / u_tau;
    let viscous = |y: f64| u_tau * u_tau * y / nu;
    if h <= y_m {
        return simpson(viscous, 0.0, h, 4) / h;
    }
    let power_in_log_y = |t: f64| {
        let y = t.exp();
        u_tau * a * (y * u_tau / nu).powf(b) * y
    };
    let total = simpson(viscous, 0.0, y_m, 4) + simpson(power_in_log_y, y_m.ln(), h.ln(), 4000);
    total / h
}

#[test]
fn werner_wengle_cell_averaged_inverts_the_integrated_two_layer_profile() {
    // (u_tau, cell height, nu): cell fully viscous, partially viscous, and deep in
    // the power-law layer (cell y+ of 0.5, 5, 20, 100 and 1e4).
    let cases = [
        (0.05, 1e-4, 1e-5),
        (0.05, 1e-3, 1e-5),
        (0.05, 4e-3, 1e-5),
        (0.3, 5e-3, 1.5e-5),
        (0.5, 0.2, 1e-5),
    ];
    for (u_tau, h, nu) in cases {
        let u_mean = werner_wengle_cell_mean(u_tau, h, nu);
        let tau = WernerWengle::wall_shear_stress_cell_averaged(u_mean, h, nu);
        assert_rel(
            tau,
            u_tau * u_tau,
            1e-8,
            &format!("h+ = {}", h * u_tau / nu),
        );
    }
}

#[test]
fn werner_wengle_cell_averaged_is_continuous_and_matches_the_node_form_in_the_sublayer() {
    let (h, nu) = (2e-3, 1e-5);
    let u_switch = 0.5 * (nu / h) * WernerWengle::y_plus_switch().powi(2);
    let below = WernerWengle::wall_shear_stress_cell_averaged(u_switch * (1.0 - 1e-12), h, nu);
    let above = WernerWengle::wall_shear_stress_cell_averaged(u_switch * (1.0 + 1e-12), h, nu);
    assert!(
        (above - below).abs() < 1e-10 * below,
        "jump {}",
        above - below
    );
    // Cell mean of the sublayer profile equals its value at the cell centre y = h / 2.
    let u = 0.7 * u_switch;
    assert_rel(
        WernerWengle::wall_shear_stress_cell_averaged(u, h, nu),
        WernerWengle::wall_shear_stress(u, 0.5 * h, nu),
        1e-13,
        "cell vs node form",
    );
}
