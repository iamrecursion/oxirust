//! Regression tests for the hemodynamics LBM: unit-consistent rheology (the
//! relaxation time follows from the SI viscosity and the lattice scales) and
//! wall-bounded streaming (walls reflect, they never inject populations).

use super::*;

// ── rheology: SI viscosity → lattice relaxation time ─────────────────────────

/// Lattice spacing (m) of the blood tests (100 µm).
const DX: f64 = 1.0e-4;
/// Blood density (kg/m³).
const RHO: f64 = 1060.0;

/// Time step (s) that puts the asymptotic Casson/Carreau–Yasuda viscosity
/// `mu_inf = 0.00345` Pa·s at the relaxation time `tau_inf`.
fn dt_for_tau_inf(tau_inf: f64) -> f64 {
    lattice_time_step(DX, RHO, 0.00345, tau_inf).expect("valid scales")
}

#[test]
fn casson_without_yield_stress_is_a_newtonian_fluid_of_viscosity_mu_inf() {
    let dt = dt_for_tau_inf(0.8);
    let model = CassonModel::new(0.0, 0.00345);
    let expected = 0.5 + 3.0 * 0.00345 / RHO * dt / (DX * DX);
    assert!(
        (expected - 0.8).abs() < 1e-12,
        "time step must realise tau = 0.8: {expected}"
    );
    for &gamma_lu in &[0.0, 1.0e-4, 1.0e-3, 1.0e-2, 0.2] {
        let tau = model.relaxation_time(gamma_lu, DX, dt, RHO);
        assert!(
            (tau - expected).abs() < 1e-12,
            "gamma_lu = {gamma_lu}: tau = {tau}, expected 0.5 + 3 mu_inf/rho dt/dx^2 = {expected}"
        );
    }
}

#[test]
fn casson_relaxation_time_uses_the_shear_rate_in_si_units() {
    let dt = dt_for_tau_inf(0.6);
    let model = CassonModel::new_blood();
    let gamma_lu = 2.0e-3;
    let gamma_si = gamma_lu / dt;
    assert!(gamma_si > 1.0, "test setup: SI shear rate {gamma_si} 1/s");
    let mu = model.viscosity(gamma_si);
    let nu_lu = mu / RHO * dt / (DX * DX);
    let expected = 0.5 + 3.0 * nu_lu;
    let tau = model.relaxation_time(gamma_lu, DX, dt, RHO);
    assert!(
        (tau - expected).abs() < 1e-12,
        "tau = {tau}, expected {expected} from mu({gamma_si} 1/s) = {mu}"
    );
    // Shear thinning: a faster shear gives a smaller relaxation time.
    let tau_fast = model.relaxation_time(10.0 * gamma_lu, DX, dt, RHO);
    assert!(
        tau_fast < tau,
        "tau must fall with shear: {tau_fast} vs {tau}"
    );
    assert!(tau_fast > 0.6, "the limit is tau_inf = 0.6: {tau_fast}");
}

#[test]
fn carreau_yasuda_limits_in_lattice_units() {
    let dt = dt_for_tau_inf(0.55);
    let model = CarreauYasudaModel::new_blood();
    let tau_inf = 0.5 + 3.0 * model.mu_inf / RHO * dt / (DX * DX);
    let tau_0 = 0.5 + 3.0 * model.mu_0 / RHO * dt / (DX * DX);
    assert!((tau_inf - 0.55).abs() < 1e-12);
    assert!(
        tau_0 > 1.0 && tau_0 < TAU_MAX,
        "test setup: the zero-shear tau {tau_0} must lie inside the clamp"
    );
    let at_rest = model.relaxation_time(0.0, DX, dt, RHO);
    assert!(
        (at_rest - tau_0).abs() < 1e-12,
        "zero shear: tau = {at_rest}, expected {tau_0}"
    );
    // gamma_si = 1e7 1/s is far above the Carreau–Yasuda knee 1/lambda.
    let fast = model.relaxation_time(1.0e7 * dt, DX, dt, RHO);
    assert!(
        (fast - tau_inf).abs() < 1e-3,
        "infinite shear: tau = {fast}, expected {tau_inf}"
    );
    let mut previous = f64::INFINITY;
    for k in 0..12 {
        let gamma_lu = 1.0e-7 * 10.0_f64.powi(k);
        let tau = model.relaxation_time(gamma_lu, DX, dt, RHO);
        assert!(
            tau <= previous + 1e-15,
            "tau must not increase with shear: {tau} after {previous}"
        );
        previous = tau;
    }
}

#[test]
fn relaxation_time_is_clamped_to_the_stable_range() {
    let dt = dt_for_tau_inf(0.8);
    // An unyielded Casson plug has an enormous viscosity.
    let plug = CassonModel::new_blood().relaxation_time(0.0, DX, dt, RHO);
    assert_eq!(plug, TAU_MAX);
    // A nearly inviscid fluid would give tau = 1/2 exactly.
    let thin = CassonModel::new(0.0, 1.0e-12).relaxation_time(1.0e-3, DX, dt, RHO);
    assert_eq!(thin, TAU_MIN);
    // Degenerate scales never produce NaN.
    let degenerate = CassonModel::new_blood().relaxation_time(0.1, 0.0, dt, RHO);
    assert_eq!(degenerate, TAU_MAX);
}

#[test]
fn unit_scales_reproduce_lattice_unit_viscosities() {
    // dx = dt = rho = 1: the model viscosity is the lattice viscosity.
    let model = CassonModel::new(0.0, 0.02);
    assert!((model.relaxation_time(0.05, 1.0, 1.0, 1.0) - (0.5 + 3.0 * 0.02)).abs() < 1e-12);
    let solver = HemodynamicsLbm::new(6, 6, 0.1);
    assert_eq!((solver.dx, solver.dt, solver.rho_phys), (1.0, 1.0, 1.0));
}

#[test]
fn blood_presets_in_si_units_give_stable_relaxation_times() {
    // With physical scales both blood presets stay inside the BGK stability
    // window and approach tau_inf = 0.6 at high shear; without the unit
    // conversion the same viscosities gave tau = 0.5 + mu/cs^2 ~ 0.51.
    let dt = dt_for_tau_inf(0.6);
    let casson = CassonModel::new_blood();
    let carreau = CarreauYasudaModel::new_blood();
    for k in 0..8 {
        let gamma_si = 0.1 * 10.0_f64.powi(k);
        let gamma_lu = gamma_si * dt;
        for tau in [
            casson.relaxation_time(gamma_lu, DX, dt, RHO),
            carreau.relaxation_time(gamma_lu, DX, dt, RHO),
        ] {
            assert!(
                (TAU_MIN..=TAU_MAX).contains(&tau),
                "gamma = {gamma_si} 1/s: tau = {tau}"
            );
        }
    }
    let tau_fast = casson.relaxation_time(1.0e6 * dt, DX, dt, RHO);
    assert!((tau_fast - 0.6).abs() < 0.01, "high-shear tau {tau_fast}");
    let unscaled = 0.5 + casson.mu_inf / CS2;
    assert!(
        unscaled < 0.52 && tau_fast > 0.59,
        "the old SI-as-lattice conversion gave {unscaled}, the scaled one {tau_fast}"
    );
}

/// Install the linear shear `u_x = shear * y` in the velocity field of an
/// all-fluid solver.
fn install_linear_shear(solver: &mut HemodynamicsLbm, shear: f64) {
    for column in solver.vel.iter_mut() {
        for (y, velocity) in column.iter_mut().enumerate() {
            *velocity = [shear * y as f64, 0.0];
        }
    }
}

#[test]
fn solver_updates_tau_with_its_physical_scales() {
    let (nx, ny) = (12, 12);
    let dt = dt_for_tau_inf(0.6);
    let mut solver =
        HemodynamicsLbm::from_physical_units(nx, ny, DX, dt, RHO, 0.00345).expect("scales");
    assert!((solver.tau_base - 0.6).abs() < 1e-12, "tau_base");
    let shear = 2.0e-3;
    install_linear_shear(&mut solver, shear);
    // compute_shear_rate of u_x = S y is |S| in lattice units.
    assert!((solver.compute_shear_rate(6, 6) - shear).abs() < 1e-15);

    let casson = CassonModel::new_blood();
    solver.update_tau_casson(&casson);
    let expected = casson.relaxation_time(shear, DX, dt, RHO);
    assert!(
        expected > 0.6 && expected < TAU_MAX,
        "test setup: tau = {expected}"
    );
    assert!((solver.tau[6][6] - expected).abs() < 1e-14);

    let carreau = CarreauYasudaModel::new_blood();
    solver.update_tau_carreau_yasuda(&carreau);
    let expected_cy = carreau.relaxation_time(shear, DX, dt, RHO);
    assert!((solver.tau[6][6] - expected_cy).abs() < 1e-14);

    // Doubling the shear thins the fluid.
    install_linear_shear(&mut solver, 2.0 * shear);
    solver.update_tau_casson(&casson);
    assert!(solver.tau[6][6] < expected, "shear thinning in the solver");
}

#[test]
fn physical_scales_are_validated() {
    let mut solver = HemodynamicsLbm::new(8, 8, 0.1);
    for (dx, dt, rho, bad) in [
        (-1.0e-4, 1.0e-4, 1060.0, "dx"),
        (1.0e-4, 0.0, 1060.0, "dt"),
        (1.0e-4, 1.0e-4, f64::NAN, "rho_phys"),
    ] {
        match solver.set_physical_units(dx, dt, rho) {
            Err(Error::ParameterOutOfBounds { name, .. }) => assert_eq!(name, bad),
            other => panic!("expected ParameterOutOfBounds for {bad}, got {other:?}"),
        }
    }
    assert_eq!((solver.dx, solver.dt, solver.rho_phys), (1.0, 1.0, 1.0));
    solver
        .set_physical_units(1.0e-4, 2.0e-4, 1060.0)
        .expect("valid scales");
    assert_eq!(
        (solver.dx, solver.dt, solver.rho_phys),
        (1.0e-4, 2.0e-4, 1060.0)
    );
    assert!(HemodynamicsLbm::from_physical_units(8, 8, 1.0e-4, 1.0e-4, 1060.0, 0.0).is_err());
    assert!(lattice_time_step(DX, RHO, 0.00345, 0.5).is_err());
    assert!(lattice_time_step(0.0, RHO, 0.00345, 0.8).is_err());
}

// ── streaming: walls reflect, they never inject populations ─────────────────

/// Deterministic pseudo-random numbers in `[-1, 1)` (64-bit LCG).
fn next_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

#[test]
fn wall_nodes_reflect_instead_of_injecting_populations() {
    let n = 5;
    let mut solver = HemodynamicsLbm::new(n, n, 0.1);
    for column in solver.cell_type.iter_mut() {
        column.fill(CellType::Wall);
    }
    solver.cell_type[2][2] = CellType::Fluid;
    let mut state = 7_u64;
    // Poison every wall population; give the pocket distinct values.
    for column in solver.f.iter_mut() {
        for cell in column.iter_mut() {
            for value in cell.iter_mut() {
                *value = 10.0 + next_unit(&mut state);
            }
        }
    }
    let pocket: [f64; 9] = std::array::from_fn(|i| 0.1 * (i as f64 + 1.0));
    solver.f[2][2] = pocket;
    let walls_before = solver.f.clone();
    solver.stream();
    for (i, &opposite) in D2Q9_OPP.iter().enumerate() {
        assert_eq!(
            solver.f[2][2][i], pocket[opposite],
            "population {i} must be the reflected population {opposite}"
        );
    }
    for (x, (after, before)) in solver.f.iter().zip(&walls_before).enumerate() {
        for (y, (cell_after, cell_before)) in after.iter().zip(before).enumerate() {
            if (x, y) != (2, 2) {
                assert_eq!(cell_after, cell_before, "wall ({x},{y}) changed");
            }
        }
    }
}

/// An all-wall rim around a fluid interior.
fn closed_box(nx: usize, ny: usize) -> HemodynamicsLbm {
    let mut solver = HemodynamicsLbm::new(nx, ny, 0.1);
    for x in 0..nx {
        for y in 0..ny {
            if x == 0 || y == 0 || x + 1 == nx || y + 1 == ny {
                solver.cell_type[x][y] = CellType::Wall;
            }
        }
    }
    solver
}

fn fluid_mass(solver: &HemodynamicsLbm) -> f64 {
    let mut mass = 0.0;
    for x in 0..solver.nx {
        for y in 0..solver.ny {
            if solver.cell_type[x][y] != CellType::Wall {
                mass += solver.f[x][y].iter().sum::<f64>();
            }
        }
    }
    mass
}

#[test]
fn closed_box_conserves_mass_and_bounds_the_flow() {
    let mut solver = closed_box(18, 14);
    let mut state = 42_u64;
    for x in 1..17 {
        for y in 1..13 {
            let rho = 1.0 + 0.02 * next_unit(&mut state);
            let u = [0.03 * next_unit(&mut state), 0.03 * next_unit(&mut state)];
            solver.f[x][y] = HemodynamicsLbm::equilibrium_static(rho, u);
        }
    }
    let mass_before = fluid_mass(&solver);
    for _ in 0..400 {
        solver.step();
    }
    let mass_after = fluid_mass(&solver);
    assert!(
        (mass_after - mass_before).abs() < 1e-12 * mass_before,
        "mass {mass_before} -> {mass_after}"
    );
    solver.compute_macroscopic();
    let speed_max = solver
        .vel
        .iter()
        .flatten()
        .map(|&u| len2(u))
        .fold(0.0_f64, f64::max);
    assert!(
        speed_max < 0.1,
        "the closed box must stay quiescent: {speed_max}"
    );
}

/// Assert that the flow of a straight channel with the centre row `centre` is
/// mirror-symmetric (`u_x` even, `u_y` odd in the offset from the centre row)
/// and that the wall shear stress of both walls agrees; returns the centre-line
/// speed at mid-length.
fn assert_symmetric_channel(
    solver: &mut HemodynamicsLbm,
    geometry: &ArterialGeometry,
    centre: usize,
) -> f64 {
    solver.compute_macroscopic();
    let nx = solver.nx;
    for x in 1..nx - 1 {
        for d in 1..=6 {
            let (upper, lower) = (solver.vel[x][centre + d], solver.vel[x][centre - d]);
            assert!(
                (upper[0] - lower[0]).abs() < 1e-10,
                "x = {x}, d = {d}: u_x {} vs {}",
                upper[0],
                lower[0]
            );
            assert!(
                (upper[1] + lower[1]).abs() < 1e-10,
                "x = {x}, d = {d}: u_y {} vs {}",
                upper[1],
                lower[1]
            );
        }
    }
    let (wss_lower, wss_upper) = compute_wall_shear_stress(solver, geometry, 0.003);
    for x in 1..nx - 1 {
        assert!(
            (wss_lower[x] - wss_upper[x]).abs() < 1e-10,
            "x = {x}: wall shear stress lower {} vs upper {}",
            wss_lower[x],
            wss_upper[x]
        );
    }
    assert!(
        wss_lower[nx / 2] > 0.0,
        "wall shear stress must be non-zero"
    );
    solver.vel[nx / 2][centre][0]
}

#[test]
fn straight_channel_profile_is_symmetric_and_wall_shear_stress_matches() {
    let (nx, ny) = (40, 20);
    let geometry = ArterialGeometry::straight(nx, ny, 7.0);
    let mut solver = HemodynamicsLbm::new(nx, ny, 0.1);
    solver.set_arterial_geometry(&geometry);
    solver.init_poiseuille(0.04, &geometry);
    let inflow = PulsatileFlow::new_simple(0.02, 72.0, 0.0);
    let centre = ny / 2;
    // The fed channel decays slowly (the equilibrium inlet and the
    // zero-gradient outlet impose no pressure drop), but it stays non-trivial
    // and must stay mirror-symmetric throughout.
    let mut early_speed = 0.0;
    for step in 0..2000 {
        solver.step();
        solver.apply_pulsatile_inlet(&inflow, &geometry, step as f64);
        solver.apply_outlet();
        if step == 299 {
            early_speed = assert_symmetric_channel(&mut solver, &geometry, centre);
        }
    }
    let late_speed = assert_symmetric_channel(&mut solver, &geometry, centre);
    assert!(early_speed > 0.015, "flow after 300 steps: {early_speed}");
    assert!(late_speed > 1.0e-3, "flow after 2000 steps: {late_speed}");
}
