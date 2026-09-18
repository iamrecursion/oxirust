//! Regression tests for the soil SPH module: Drucker-Prager return mapping,
//! Terzaghi pore-pressure diffusion, the Bishop slope-stability factor of safety
//! and the neighbour search of [`SoilSphSolver`].

use super::*;

/// Shear modulus of the return-mapping tests \[Pa\].
const G: f64 = 1.0e7;
/// Bulk modulus of the return-mapping tests \[Pa\].
const K: f64 = 2.0e7;

/// Deterministic uniform pseudo-random numbers in `[0, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

fn diagonal(a: f64, b: f64, c: f64) -> Tensor3 {
    [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]]
}

fn cone() -> DruckerPrager {
    DruckerPrager::from_mohr_coulomb(5000.0, 25.0_f64.to_radians())
}

/// `sqrt(J2)` of a tensor.
fn sqrt_j2(t: &Tensor3) -> f64 {
    tensor_j2(&tensor_deviatoric(t)).sqrt()
}

/// Largest entry-wise difference of two tensors.
fn max_difference(a: &Tensor3, b: &Tensor3) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            worst = worst.max((a[i][j] - b[i][j]).abs());
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// F4642: Drucker-Prager return mapping
// ---------------------------------------------------------------------------

#[test]
fn return_mapping_lands_on_the_yield_surface() {
    let dp = cone();
    let sigma = diagonal(1.0e7, -5.0e6, 0.0);
    let f_before = dp.yield_function(&sigma);
    assert!(
        f_before > 0.0,
        "the trial state must violate the yield condition"
    );
    let (corrected, d_lambda) = dp.return_mapping(&sigma, G, K);
    assert!(d_lambda > 0.0);
    let f_after = dp.yield_function(&corrected);
    assert!(
        f_after.abs() < 1e-8 * dp.k.max(1.0),
        "f after the return mapping must vanish, got {f_after} (k = {})",
        dp.k
    );
}

#[test]
fn return_mapping_volumetric_correction_is_three_k_alpha_dlambda() {
    let dp = cone();
    let sigma = diagonal(1.0e7, -5.0e6, 0.0);
    let f_before = dp.yield_function(&sigma);
    let (corrected, d_lambda) = dp.return_mapping(&sigma, G, K);

    // Consistency condition: dlambda = f / (G + 9 K alpha^2).
    let expected_dl = f_before / (G + 9.0 * K * dp.alpha * dp.alpha);
    assert!((d_lambda - expected_dl).abs() < 1e-12 * expected_dl);

    // Associated flow: delta eps_v = 3 alpha dlambda, so delta p = -3 K alpha dlambda.
    let p_trial = tensor_trace(&sigma) / 3.0;
    let p_new = tensor_trace(&corrected) / 3.0;
    let expected_p = p_trial - 3.0 * K * dp.alpha * d_lambda;
    assert!(
        (p_new - expected_p).abs() < 1e-9 * p_trial.abs(),
        "mean stress {p_new} but p_trial - 3 K alpha dlambda = {expected_p}"
    );

    // The deviator shrinks by G dlambda: sqrt(J2) -> sqrt(J2) - G dlambda.
    let before = sqrt_j2(&sigma);
    let after = sqrt_j2(&corrected);
    assert!((after - (before - G * d_lambda)).abs() < 1e-9 * before);
}

#[test]
fn return_mapping_returns_hydrostatic_tension_to_the_apex() {
    let dp = cone();
    let sigma = tensor_scale(&tensor_identity(), 1.0e6);
    assert!(dp.yield_function(&sigma) > 0.0);
    let (corrected, d_lambda) = dp.return_mapping(&sigma, G, K);
    let p_apex = dp.k / (3.0 * dp.alpha);
    let expected = tensor_scale(&tensor_identity(), p_apex);
    assert!(
        max_difference(&corrected, &expected) < 1e-9 * p_apex,
        "apex return must give p = k / (3 alpha) = {p_apex}, got {corrected:?}"
    );
    let expected_dl = (dp.alpha * 3.0e6 - dp.k) / (9.0 * K * dp.alpha * dp.alpha);
    assert!((d_lambda - expected_dl).abs() < 1e-12 * expected_dl);
    assert!(dp.yield_function(&corrected).abs() < 1e-8 * dp.k.max(1.0));
}

#[test]
fn return_mapping_removes_a_small_deviator_in_the_apex_region() {
    let dp = cone();
    // Large mean tension with a small deviator: sqrt(J2) = 1e5.
    let sigma = diagonal(1.0e6 + 1.0e5, 1.0e6 - 1.0e5, 1.0e6);
    let (corrected, _) = dp.return_mapping(&sigma, G, K);
    let p_apex = dp.k / (3.0 * dp.alpha);
    assert!(
        max_difference(&corrected, &tensor_scale(&tensor_identity(), p_apex)) < 1e-9 * p_apex,
        "the whole deviator must be removed, got {corrected:?}"
    );

    // A larger deviator keeps the cone return: it survives (scaled) and f = 0.
    let strong = diagonal(1.0e6 + 2.0e6, 1.0e6 - 2.0e6, 1.0e6);
    let (corrected, d_lambda) = dp.return_mapping(&strong, G, K);
    let remaining = sqrt_j2(&corrected);
    assert!(remaining > 1e4, "cone return keeps part of the deviator");
    assert!(remaining < sqrt_j2(&strong));
    assert!(dp.yield_function(&corrected).abs() < 1e-8 * dp.k.max(1.0));
    assert!(d_lambda > 0.0);
}

#[test]
fn plastic_multiplier_and_stress_are_continuous_across_the_apex_boundary() {
    let dp = cone();
    // Mean stress 1e6 (I1 = 3e6): the apex region is sqrt(J2) <= a_star.
    let i1 = 3.0e6;
    let a_star = G * (dp.alpha * i1 - dp.k) / (9.0 * K * dp.alpha * dp.alpha);
    let state = |a: f64| diagonal(i1 / 3.0 + a, i1 / 3.0 - a, i1 / 3.0);
    let (cone_side, dl_cone) = dp.return_mapping(&state(a_star * (1.0 + 1e-7)), G, K);
    let (apex_side, dl_apex) = dp.return_mapping(&state(a_star * (1.0 - 1e-7)), G, K);
    assert!(
        (dl_cone - dl_apex).abs() < 1e-6 * dl_apex,
        "multiplier jumps across the apex boundary: {dl_cone} vs {dl_apex}"
    );
    assert!(
        max_difference(&cone_side, &apex_side) < 1e-5 * (i1 + a_star),
        "stress jumps across the apex boundary: {cone_side:?} vs {apex_side:?}"
    );
}

#[test]
fn return_mapping_leaves_elastic_states_untouched() {
    let dp = cone();
    let sigma = diagonal(-2.0e5, -3.0e5, -2.5e5);
    assert!(dp.yield_function(&sigma) < 0.0);
    let (unchanged, d_lambda) = dp.return_mapping(&sigma, G, K);
    assert_eq!(unchanged, sigma);
    assert_eq!(d_lambda, 0.0);
    // On the surface itself nothing happens either.
    let hydrostatic = tensor_scale(&tensor_identity(), dp.k / (3.0 * dp.alpha));
    let (same, dl) = dp.return_mapping(&hydrostatic, G, K);
    assert!(dl <= 1e-12);
    assert!(max_difference(&same, &hydrostatic) < 1e-6 * dp.k / dp.alpha);
}

#[test]
fn return_mapping_hands_back_a_corrupted_stress_unchanged() {
    let dp = cone();
    let mut sigma = diagonal(1.0e6, 1.0e6, 1.0e6);
    sigma[0][1] = f64::NAN;
    let (out, dl) = dp.return_mapping(&sigma, G, K);
    assert_eq!(dl, 0.0);
    assert!(out[0][1].is_nan());
    assert_eq!(out[0][0], 1.0e6);
}

#[test]
fn return_mapping_sweep_is_consistent_on_both_branches() {
    let dp = cone();
    let mut rng = Lcg(31337);
    let (mut cone_returns, mut apex_returns, mut elastic) = (0_u32, 0_u32, 0_u32);
    for _ in 0..20_000 {
        // Random symmetric trial stress: mean stress in [-3e6, 3e6] and a deviator
        // of either large (3e6) or small (3e5) magnitude, so that the sweep reaches
        // the elastic region, the smooth part of the cone and the apex region.
        let p = rng.range(-3.0e6, 3.0e6);
        let dev_scale = if rng.next_f64() < 0.5 { 3.0e6 } else { 3.0e5 };
        let [xx, yy, zz, xy, xz, yz] = [0; 6].map(|_| rng.range(-1.0, 1.0) * dev_scale);
        let dev = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]];
        let trial = tensor_add(
            &tensor_deviatoric(&dev),
            &tensor_scale(&tensor_identity(), p),
        );
        let f_before = dp.yield_function(&trial);
        let (corrected, d_lambda) = dp.return_mapping(&trial, G, K);
        if f_before <= 0.0 {
            assert_eq!(corrected, trial);
            assert_eq!(d_lambda, 0.0);
            elastic += 1;
            continue;
        }
        let scale = 1.0 + tensor_trace(&trial).abs() + sqrt_j2(&trial) + dp.k;
        let f_after = dp.yield_function(&corrected);
        assert!(
            f_after.abs() < 1e-9 * scale,
            "trial {trial:?}: f after return is {f_after}"
        );
        assert!(d_lambda >= 0.0, "negative plastic multiplier {d_lambda}");
        // Plastic flow lowers the mean stress and never grows the deviator.
        assert!(tensor_trace(&corrected) <= tensor_trace(&trial) + 1e-9 * scale);
        let (j_before, j_after) = (sqrt_j2(&trial), sqrt_j2(&corrected));
        assert!(j_after <= j_before * (1.0 + 1e-12));
        // The deviator keeps its direction.
        let ratio = if j_before > 0.0 {
            j_after / j_before
        } else {
            0.0
        };
        let dev_trial = tensor_deviatoric(&trial);
        let dev_new = tensor_deviatoric(&corrected);
        let expected_dev = tensor_scale(&dev_trial, ratio);
        assert!(max_difference(&dev_new, &expected_dev) <= 1e-9 * (1.0 + j_before));
        if j_after < 1e-9 * (1.0 + j_before) {
            apex_returns += 1;
        } else {
            cone_returns += 1;
        }
    }
    assert!(
        cone_returns > 1000,
        "only {cone_returns} cone returns sampled"
    );
    assert!(
        apex_returns > 100,
        "only {apex_returns} apex returns sampled"
    );
    assert!(elastic > 1000, "only {elastic} elastic states sampled");
}

#[test]
fn return_mapping_without_friction_is_a_finite_von_mises_return() {
    // phi = 0: alpha = 0, the yield surface is a cylinder, no volumetric correction.
    let dp = DruckerPrager::from_mohr_coulomb(2.0e4, 0.0);
    assert_eq!(dp.alpha, 0.0);
    let sigma = diagonal(1.0e5, -1.0e5, 0.0);
    let (corrected, d_lambda) = dp.return_mapping(&sigma, G, K);
    assert!(d_lambda > 0.0 && d_lambda.is_finite());
    assert!((sqrt_j2(&corrected) - dp.k).abs() < 1e-8 * dp.k);
    assert!(tensor_trace(&corrected).abs() < 1e-6);

    // phi = 0 and c = 0: nothing but the hydrostatic axis is admissible; the
    // result must stay finite (no 0/0 from an apex at k / (3 alpha)).
    let degenerate = DruckerPrager::from_mohr_coulomb(0.0, 0.0);
    let (flat, dl) = degenerate.return_mapping(&diagonal(3.0e4, -1.0e4, 5.0e3), G, K);
    assert!(dl.is_finite());
    for row in flat.iter() {
        for v in row.iter() {
            assert!(v.is_finite());
        }
    }
    assert!(sqrt_j2(&flat) < 1e-6);
    // No volumetric plastic strain without friction: the mean stress is kept.
    let mean_in = (3.0e4 - 1.0e4 + 5.0e3) / 3.0;
    assert!((tensor_trace(&flat) / 3.0 - mean_in).abs() < 1e-6 * mean_in);
}

#[test]
fn solver_update_returns_yielding_particles_to_the_effective_stress_surface() {
    let params = SoilParams::saturated_clay();
    let mut particle = SoilParticle::new([0.0; 3], 1.0, 1900.0, 0.1);
    particle.stress = diagonal(1.0e7, -5.0e6, 0.0);
    particle.pore_pressure = 2.0e5;
    let mut solver = SoilSphSolver::new(vec![particle], params, SoilSphConfig::default());
    solver.build_neighbors();
    let eff_before = effective_stress(&solver.particles[0].stress, 2.0e5);
    assert!(solver.yield_surface.yield_function(&eff_before) > 0.0);
    solver.update_stresses();
    let eff = effective_stress(&solver.particles[0].stress, 2.0e5);
    let f = solver.yield_surface.yield_function(&eff);
    assert!(
        f.abs() < 1e-8 * solver.yield_surface.k.max(1.0),
        "effective stress must sit on the yield surface, f = {f}"
    );
    assert!(solver.particles[0].plastic_strain > 0.0);
}

// ---------------------------------------------------------------------------
// F4645: pore-pressure diffusion is a Laplacian
// ---------------------------------------------------------------------------

/// `n`³ particles on a cubic lattice of spacing `dx` starting at the origin.
fn lattice(n: usize, dx: f64, h: f64, density: f64) -> Vec<SoilParticle> {
    let mass = density * dx.powi(3);
    let mut particles = Vec::with_capacity(n * n * n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                particles.push(SoilParticle::new(
                    [i as f64 * dx, j as f64 * dx, k as f64 * dx],
                    mass,
                    density,
                    h,
                ));
            }
        }
    }
    particles
}

fn solver_with_lists(particles: Vec<SoilParticle>) -> SoilSphSolver {
    let mut solver = SoilSphSolver::new(
        particles,
        SoilParams::saturated_clay(),
        SoilSphConfig::default(),
    );
    solver.build_neighbors();
    solver
}

#[test]
fn diffusion_preserves_a_uniform_pore_pressure() {
    let mut particles = lattice(5, 0.1, 0.15, 1700.0);
    for p in &mut particles {
        p.pore_pressure = 5.0e4;
    }
    let mut solver = solver_with_lists(particles);
    pore_pressure_diffusion_step(&mut solver.particles, &solver.neighbors, 2.0e-3, 0.1);
    for p in &solver.particles {
        assert!((p.pore_pressure - 5.0e4).abs() < 1e-9);
    }
}

#[test]
fn diffusion_rate_is_cv_times_the_laplacian_in_pascal_per_second() {
    let dx = 0.1;
    let n = 9;
    let mut particles = lattice(n, dx, 1.4 * dx, 1700.0);
    // u = a (x^2 + 2 y^2 + 3 z^2) about the lattice centre: laplacian = 12 a.
    let a = 1.0e6;
    let centre_coord = 4.0 * dx;
    for p in &mut particles {
        let (x, y, z) = (
            p.position[0] - centre_coord,
            p.position[1] - centre_coord,
            p.position[2] - centre_coord,
        );
        p.pore_pressure = a * (x * x + 2.0 * y * y + 3.0 * z * z);
    }
    let centre = (4 * n + 4) * n + 4;
    assert!(particles[centre].pore_pressure.abs() < 1e-9);
    let cv = 2.0e-3;
    let dt = 1.0e-3;
    let mut solver = solver_with_lists(particles);
    pore_pressure_diffusion_step(&mut solver.particles, &solver.neighbors, cv, dt);
    let rate = solver.particles[centre].pore_pressure / dt;
    let expected = cv * 12.0 * a;
    // The cubic-spline lattice sum reproduces the Laplacian to about 0.5 % for h = 1.4 dx.
    assert!(
        (rate - expected).abs() < 0.015 * expected,
        "du/dt at the centre is {rate} Pa/s but c_v laplacian(u) = {expected} Pa/s"
    );
}

#[test]
fn a_pressure_peak_decays_into_its_neighbours_and_the_total_is_conserved() {
    let dx = 0.1;
    let n = 7;
    let mut particles = lattice(n, dx, 1.4 * dx, 1700.0);
    let centre = (3 * n + 3) * n + 3;
    particles[centre].pore_pressure = 1.0e5;
    let mut solver = solver_with_lists(particles);
    let total_before: f64 = solver.particles.iter().map(|p| p.pore_pressure).sum();
    // dt * cv * sum_j w_ij is about 0.16 < 1: the explicit update is a convex combination.
    pore_pressure_diffusion_step(&mut solver.particles, &solver.neighbors, 2.0e-3, 0.3);
    let peak = solver.particles[centre].pore_pressure;
    assert!(peak < 1.0e5, "the peak must decay, got {peak}");
    assert!(
        peak > 0.5e5,
        "one stable step cannot remove most of the peak"
    );
    for &j in &solver.neighbors[centre] {
        assert!(
            solver.particles[j].pore_pressure > 0.0,
            "neighbour {j} must gain pressure"
        );
    }
    let minimum = solver
        .particles
        .iter()
        .map(|p| p.pore_pressure)
        .fold(f64::INFINITY, f64::min);
    assert!(minimum >= 0.0, "a stable step creates no negative pressure");
    let total_after: f64 = solver.particles.iter().map(|p| p.pore_pressure).sum();
    assert!(
        (total_after - total_before).abs() < 1e-9 * total_before,
        "diffusion conserves the total pore pressure: {total_before} -> {total_after}"
    );
}

#[test]
fn boundary_particles_keep_their_prescribed_pore_pressure() {
    let mut fluid = SoilParticle::new([0.0; 3], 1.0, 1700.0, 0.1);
    fluid.pore_pressure = 0.0;
    let mut wall = SoilParticle::new_boundary([0.05, 0.0, 0.0], 1.0, 1700.0, 0.1);
    wall.pore_pressure = 5.0e4;
    let mut solver = solver_with_lists(vec![fluid, wall]);
    pore_pressure_diffusion_step(&mut solver.particles, &solver.neighbors, 1.0e-3, 0.01);
    assert_eq!(solver.particles[1].pore_pressure, 5.0e4);
    assert!(
        solver.particles[0].pore_pressure > 0.0,
        "fluid drains towards the wall value"
    );
}

#[test]
fn solver_step_dissipates_a_pore_pressure_peak_only_when_enabled() {
    let dx = 0.1;
    let n = 7;
    let centre = (3 * n + 3) * n + 3;
    let run = |enabled: bool| {
        let mut particles = lattice(n, dx, 1.4 * dx, 1700.0);
        particles[centre].pore_pressure = 1.0e5;
        let config = SoilSphConfig {
            gravity: [0.0; 3],
            enable_pore_pressure: enabled,
            // dt * c_v * sum_j w_ij is about 0.27: a stable explicit step.
            consolidation_cv: 10.0,
            ..SoilSphConfig::default()
        };
        let mut solver = SoilSphSolver::new(particles, SoilParams::saturated_clay(), config);
        solver.step();
        solver
    };

    let diffusing = run(true);
    let peak = diffusing.particles[centre].pore_pressure;
    assert!(
        peak < 0.85e5 && peak > 0.6e5,
        "one step must dissipate about a quarter of the peak, got {peak}"
    );
    for &j in &diffusing.neighbors[centre] {
        assert!(diffusing.particles[j].pore_pressure > 0.0);
    }

    let frozen = run(false);
    assert_eq!(frozen.particles[centre].pore_pressure, 1.0e5);
}

/// One-dimensional consolidation column made of a 3-D SPH lattice.
///
/// `layers` real particles form the column `0 <= z <= layers * dx`; every other
/// particle is a boundary (ghost) particle that is never updated by the solver:
/// the lateral columns and the layers beyond both ends are images whose pore
/// pressure is synchronised from the real column before each step.  The images
/// beyond the bottom are *even* reflections (zero flux, impermeable) and those
/// beyond the top are *odd* reflections (`u = 0` at `z = H`, drained), which is
/// the method of images for Terzaghi's boundary conditions.
struct ConsolidationColumn {
    solver: SoilSphSolver,
    /// Indices of the real particles, bottom to top.
    real: Vec<usize>,
    /// `(ghost index, source layer, sign)`.
    ghosts: Vec<(usize, usize, f64)>,
}

impl ConsolidationColumn {
    fn new(layers: usize, dx: f64, h: f64, density: f64, u0: f64) -> Self {
        let reach = 3_i32;
        let n_layers = layers as i32;
        let mass = density * dx.powi(3);
        let mut particles = Vec::new();
        let mut real = Vec::new();
        let mut ghosts = Vec::new();
        for kz in -reach..(n_layers + reach) {
            for ix in -reach..=reach {
                for iy in -reach..=reach {
                    let pos = [
                        f64::from(ix) * dx,
                        f64::from(iy) * dx,
                        (f64::from(kz) + 0.5) * dx,
                    ];
                    let index = particles.len();
                    let in_column = ix == 0 && iy == 0 && (0..n_layers).contains(&kz);
                    if in_column {
                        let mut p = SoilParticle::new(pos, mass, density, h);
                        p.pore_pressure = u0;
                        particles.push(p);
                        real.push(index);
                    } else {
                        particles.push(SoilParticle::new_boundary(pos, mass, density, h));
                        let (source, sign) = if (0..n_layers).contains(&kz) {
                            (kz, 1.0)
                        } else if kz < 0 {
                            (-1 - kz, 1.0)
                        } else {
                            (2 * n_layers - 1 - kz, -1.0)
                        };
                        ghosts.push((index, source as usize, sign));
                    }
                }
            }
        }
        let mut solver = SoilSphSolver::new(
            particles,
            SoilParams::saturated_clay(),
            SoilSphConfig::default(),
        );
        solver.build_neighbors();
        let mut column = Self {
            solver,
            real,
            ghosts,
        };
        column.sync_ghosts();
        column
    }

    fn sync_ghosts(&mut self) {
        let column: Vec<f64> = self
            .real
            .iter()
            .map(|&i| self.solver.particles[i].pore_pressure)
            .collect();
        for &(index, source, sign) in &self.ghosts {
            self.solver.particles[index].pore_pressure = sign * column[source];
        }
    }

    fn step(&mut self, cv: f64, dt: f64) {
        self.sync_ghosts();
        pore_pressure_diffusion_step(&mut self.solver.particles, &self.solver.neighbors, cv, dt);
    }

    /// Average degree of consolidation `1 - mean(u) / u0` of the column.
    fn average_degree(&self, u0: f64) -> f64 {
        let sum: f64 = self
            .real
            .iter()
            .map(|&i| self.solver.particles[i].pore_pressure)
            .sum();
        1.0 - sum / (self.real.len() as f64 * u0)
    }
}

#[test]
fn terzaghi_column_follows_the_analytic_degree_of_consolidation() {
    let layers = 16;
    let dx = 0.0625;
    let height = layers as f64 * dx;
    let u0 = 1.0e5;
    let cv = 2.0e-3;
    // dt cv / dx^2 = 0.15 keeps the explicit scheme stable (limit ~ 0.41 for h = 1.5 dx).
    let dt = 0.15 * dx * dx / cv;
    let mut column = ConsolidationColumn::new(layers, dx, 1.5 * dx, 1700.0, u0);
    assert!(column.average_degree(u0).abs() < 1e-12);

    let checkpoints = [86_usize, 171, 341, 683, 1365];
    let mut step = 0;
    let mut previous = 0.0;
    for &target in &checkpoints {
        while step < target {
            column.step(cv, dt);
            step += 1;
        }
        let time_factor = cv * (step as f64 * dt) / (height * height);
        let measured = column.average_degree(u0);
        let analytic = degree_of_consolidation(time_factor);
        assert!(
            (measured - analytic).abs() <= 0.03 * analytic,
            "T_v = {time_factor:.4}: SPH degree of consolidation {measured:.5} vs Terzaghi {analytic:.5}"
        );
        assert!(
            measured > previous,
            "consolidation must progress monotonically"
        );
        previous = measured;
    }
    // By T_v ~ 0.8 the column is about 90 % consolidated.
    assert!(
        previous > 0.85 && previous < 0.95,
        "final degree {previous}"
    );
}

#[test]
fn terzaghi_column_pressure_profile_decays_away_from_the_drain() {
    let layers = 16;
    let dx = 0.0625;
    let u0 = 1.0e5;
    let cv = 2.0e-3;
    let dt = 0.15 * dx * dx / cv;
    let mut column = ConsolidationColumn::new(layers, dx, 1.5 * dx, 1700.0, u0);
    for _ in 0..341 {
        column.step(cv, dt);
    }
    let profile: Vec<f64> = column
        .real
        .iter()
        .map(|&i| column.solver.particles[i].pore_pressure)
        .collect();
    // Excess pressure is highest at the impermeable bottom and lowest at the drain.
    for pair in profile.windows(2) {
        assert!(pair[0] >= pair[1], "profile not monotone: {profile:?}");
    }
    // Analytic values at T_v = 0.2: 0.77 u0 at the impermeable bottom, 0.04 u0 half a
    // particle spacing below the drain.
    assert!(
        profile[0] > 0.7 * u0,
        "bottom stays loaded, got {}",
        profile[0]
    );
    assert!(
        profile[0] < 0.85 * u0,
        "bottom must have started to drain, got {}",
        profile[0]
    );
    assert!(profile[layers - 1] < 0.1 * u0, "drain end is relieved");
    assert!(profile.iter().all(|&u| (0.0..=u0).contains(&u)));
}

// ---------------------------------------------------------------------------
// F4646: bishop_simplified honours its unit weight
// ---------------------------------------------------------------------------

#[test]
fn bishop_unit_weight_changes_the_factor_of_safety() {
    // Purely cohesive slices (phi = 0): FoS = sum(c b / cos a) / sum(gamma h b sin a),
    // which is inversely proportional to the unit weight.
    let slices = vec![
        (0.0, 0.0, 5.0, 1.0, 0.1, 50_000.0, 0.0, 0.0),
        (1.0, 0.0, 4.0, 1.0, 0.15, 50_000.0, 0.0, 0.0),
    ];
    let fos_18 = bishop_simplified(&slices, 18_000.0);
    let fos_20 = bishop_simplified(&slices, 20_000.0);
    assert!(
        fos_18 > fos_20,
        "a heavier soil is less stable: {fos_18} vs {fos_20}"
    );
    assert!(
        (fos_18 * 18_000.0 - fos_20 * 20_000.0).abs() < 1e-9 * fos_18 * 18_000.0,
        "FoS * gamma must be constant for cohesive soil"
    );
    let resisting: f64 = slices.iter().map(|s| s.5 * s.3 / s.4.cos()).sum();
    let driving: f64 = slices
        .iter()
        .map(|s| s.2 * s.3 * 18_000.0 * s.4.sin())
        .sum();
    assert!((fos_18 - resisting / driving).abs() < 1e-9 * fos_18);
}

#[test]
fn bishop_is_invariant_under_consistent_unit_scaling() {
    // N/m^3 with Pa versus kN/m^3 with kPa describe the same slope.
    let slices_si = vec![
        (0.0, 0.0, 6.0, 1.5, 0.35, 12_000.0, 0.45, 8_000.0),
        (1.5, 0.0, 5.0, 1.5, 0.20, 12_000.0, 0.45, 15_000.0),
        (3.0, 0.0, 3.5, 1.5, 0.05, 12_000.0, 0.45, 4_000.0),
    ];
    let slices_kilo: Vec<_> = slices_si
        .iter()
        .map(|s| (s.0, s.1, s.2, s.3, s.4, s.5 / 1000.0, s.6, s.7 / 1000.0))
        .collect();
    let fos_si = bishop_simplified(&slices_si, 18_000.0);
    let fos_kilo = bishop_simplified(&slices_kilo, 18.0);
    assert!(fos_si.is_finite() && fos_si > 0.0);
    assert!(
        (fos_si - fos_kilo).abs() < 1e-9,
        "unit systems disagree: {fos_si} vs {fos_kilo}"
    );
    // ... and the old hard-coded 18 can no longer hide a different unit weight.
    let fos_light = bishop_simplified(&slices_si, 14_000.0);
    assert!((fos_si - fos_light).abs() > 1e-3);
}

#[test]
fn bishop_single_slice_matches_the_infinite_slope_solution() {
    let (alpha, phi) = (0.3_f64, 0.5_f64);
    // c = 0, u = 0: FoS = tan(phi) / tan(alpha), independent of the weight.
    let slice = vec![(0.0, 0.0, 2.0, 1.0, alpha, 0.0, phi, 0.0)];
    let expected = phi.tan() / alpha.tan();
    for weight in [1.0, 18_000.0] {
        let fos = bishop_simplified(&slice, weight);
        assert!(
            (fos - expected).abs() < 1e-5,
            "weight {weight}: FoS {fos} vs tan(phi)/tan(alpha) = {expected}"
        );
    }
    // With pore pressure: FoS = tan(phi) (cos^2 a - u b / W) / (sin a cos a),
    // and now the unit weight matters through u b / W.
    let (h, b, u) = (2.0, 1.0, 10_000.0);
    let wet = vec![(0.0, 0.0, h, b, alpha, 0.0, phi, u)];
    for weight in [18_000.0, 20_000.0] {
        let w = h * b * weight;
        let expected_wet =
            phi.tan() * (alpha.cos().powi(2) - u * b / w) / (alpha.sin() * alpha.cos());
        let fos = bishop_simplified(&wet, weight);
        assert!(
            (fos - expected_wet).abs() < 1e-5,
            "weight {weight}: FoS {fos} vs closed form {expected_wet}"
        );
    }
    assert!(
        (bishop_simplified(&wet, 18_000.0) - bishop_simplified(&wet, 20_000.0)).abs() > 1e-3,
        "a different unit weight must change the pore-pressure effect"
    );
}

#[test]
fn bishop_without_a_driving_moment_is_infinitely_safe() {
    let flat = vec![(0.0, 0.0, 3.0, 1.0, 0.0, 10_000.0, 0.4, 0.0)];
    assert_eq!(bishop_simplified(&flat, 18_000.0), f64::INFINITY);
}

// ---------------------------------------------------------------------------
// F4647: neighbour search
// ---------------------------------------------------------------------------

/// The all-pairs search the solver used before: `j` is listed for `i` when
/// `|x_i - x_j| < 2 h_i`.
fn all_pairs_neighbors(particles: &[SoilParticle]) -> Vec<Vec<usize>> {
    let n = particles.len();
    (0..n)
        .map(|i| {
            let cutoff = 2.0 * particles[i].h;
            (0..n)
                .filter(|&j| j != i)
                .filter(|&j| {
                    vec3_norm(vec3_sub(particles[i].position, particles[j].position)) < cutoff
                })
                .collect()
        })
        .collect()
}

fn random_cloud(n: usize, seed: u64, h_range: (f64, f64)) -> Vec<SoilParticle> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|_| {
            let position = [
                rng.range(0.0, 1.0),
                rng.range(0.0, 1.0),
                rng.range(0.0, 1.0),
            ];
            let h = rng.range(h_range.0, h_range.1);
            SoilParticle::new(position, 1.0, 1700.0, h)
        })
        .collect()
}

#[test]
fn build_neighbors_matches_the_all_pairs_search_for_random_clouds() {
    for (seed, h_range) in [(1_u64, (0.08, 0.08)), (2, (0.04, 0.12)), (3, (0.02, 0.2))] {
        let particles = random_cloud(300, seed, h_range);
        let expected = all_pairs_neighbors(&particles);
        let mut solver =
            SoilSphSolver::new(particles, SoilParams::dry_sand(), SoilSphConfig::default());
        solver.build_neighbors();
        assert_eq!(solver.neighbors.len(), 300);
        let mut total = 0;
        for (i, want) in expected.iter().enumerate() {
            assert_eq!(
                &solver.neighbors[i], want,
                "seed {seed}: neighbour list of particle {i}"
            );
            total += want.len();
        }
        assert!(total > 500, "seed {seed}: only {total} neighbour entries");
    }
}

#[test]
fn build_neighbors_is_rebuilt_from_scratch_every_call() {
    let mut solver = SoilSphSolver::new(
        random_cloud(120, 9, (0.06, 0.1)),
        SoilParams::dry_sand(),
        SoilSphConfig::default(),
    );
    solver.build_neighbors();
    // Move the particles and drop a stale, oversized list: it must be replaced.
    for p in &mut solver.particles {
        p.position[0] += 0.37;
    }
    solver.neighbors.push(vec![1, 2, 3]);
    solver.build_neighbors();
    assert_eq!(solver.neighbors.len(), 120);
    assert_eq!(solver.neighbors, all_pairs_neighbors(&solver.particles));
}

#[test]
fn build_neighbors_support_edge_is_exclusive() {
    let h = 0.0625;
    let inside = SoilParticle::new([0.0; 3], 1.0, 1700.0, h);
    let at_edge = SoilParticle::new([2.0 * h, 0.0, 0.0], 1.0, 1700.0, h);
    let just_inside = SoilParticle::new([0.0, 2.0 * h * (1.0 - 1e-9), 0.0], 1.0, 1700.0, h);
    let mut solver = SoilSphSolver::new(
        vec![inside, at_edge, just_inside],
        SoilParams::dry_sand(),
        SoilSphConfig::default(),
    );
    solver.build_neighbors();
    assert_eq!(
        solver.neighbors[0],
        vec![2],
        "|dx| = 2h is outside the support"
    );
    assert!(!solver.neighbors[1].contains(&0));
    assert_eq!(solver.neighbors[2], vec![0]);
}

#[test]
fn build_neighbors_uses_each_particles_own_support() {
    // A wide particle sees a narrow one that cannot see it back.
    let wide = SoilParticle::new([0.0; 3], 1.0, 1700.0, 0.1);
    let narrow = SoilParticle::new([0.15, 0.0, 0.0], 1.0, 1700.0, 0.05);
    let mut solver = SoilSphSolver::new(
        vec![wide, narrow],
        SoilParams::dry_sand(),
        SoilSphConfig::default(),
    );
    solver.build_neighbors();
    assert_eq!(solver.neighbors[0], vec![1]);
    assert!(solver.neighbors[1].is_empty());
}

#[test]
fn build_neighbors_survives_a_blown_up_particle() {
    let mut particles = random_cloud(150, 5, (0.08, 0.08));
    particles.push(SoilParticle::new([f64::NAN, 0.5, 0.5], 1.0, 1700.0, 0.08));
    particles.push(SoilParticle::new(
        [0.5, f64::INFINITY, 0.5],
        1.0,
        1700.0,
        0.08,
    ));
    particles.push(SoilParticle::new([1.0e300, 0.5, 0.5], 1.0, 1700.0, 0.08));
    particles.push(SoilParticle::new([0.5, 0.5, 0.5], 1.0, 1700.0, f64::NAN));
    particles.push(SoilParticle::new([0.5, 0.5, 0.5001], 1.0, 1700.0, -1.0));
    let expected = all_pairs_neighbors(&particles);
    let mut solver =
        SoilSphSolver::new(particles, SoilParams::dry_sand(), SoilSphConfig::default());
    solver.build_neighbors();
    assert_eq!(solver.neighbors, expected);
    for bad in 150..153 {
        assert!(solver.neighbors[bad].is_empty());
        assert!(solver.neighbors.iter().all(|list| !list.contains(&bad)));
    }
    // Invalid smoothing lengths see nobody but may still be seen.
    assert!(solver.neighbors[153].is_empty());
    assert!(solver.neighbors[154].is_empty());
}

#[test]
fn build_neighbors_handles_empty_and_degenerate_inputs() {
    let mut empty =
        SoilSphSolver::new(Vec::new(), SoilParams::dry_sand(), SoilSphConfig::default());
    empty.build_neighbors();
    assert!(empty.neighbors.is_empty());

    let zero_h = vec![
        SoilParticle::new([0.0; 3], 1.0, 1700.0, 0.0),
        SoilParticle::new([0.0; 3], 1.0, 1700.0, 0.0),
    ];
    let mut solver = SoilSphSolver::new(zero_h, SoilParams::dry_sand(), SoilSphConfig::default());
    solver.build_neighbors();
    assert_eq!(solver.neighbors, vec![Vec::<usize>::new(), Vec::new()]);
}

#[test]
fn build_neighbors_interior_particle_sees_exactly_the_lattice_stencil() {
    // 20^3 lattice: an interior particle has exactly the lattice points with
    // |offset| < 2.8 (h = 1.4 dx) as neighbours.
    let dx = 0.05;
    let n = 20;
    let particles = lattice(n, dx, 1.4 * dx, 1700.0);
    let mut solver =
        SoilSphSolver::new(particles, SoilParams::dry_sand(), SoilSphConfig::default());
    solver.build_neighbors();
    let mut expected = 0;
    for a in -3_i32..=3 {
        for b in -3_i32..=3 {
            for c in -3_i32..=3 {
                if (a, b, c) != (0, 0, 0) && f64::from(a * a + b * b + c * c) < 2.8 * 2.8 {
                    expected += 1;
                }
            }
        }
    }
    let centre = (10 * n + 10) * n + 10;
    assert_eq!(solver.neighbors[centre].len(), expected);
    assert!(
        solver.neighbors[0].len() < expected,
        "a corner has fewer neighbours"
    );
}

#[test]
fn solver_updates_keep_the_neighbor_lists_in_place() {
    let dx = 0.05;
    let mut solver = SoilSphSolver::new(
        lattice(5, dx, 1.4 * dx, 1700.0),
        SoilParams::dry_sand(),
        SoilSphConfig::default(),
    );
    solver.build_neighbors();
    let lists = solver.neighbors.clone();
    assert!(lists.iter().any(|l| !l.is_empty()));

    solver.update_densities();
    assert_eq!(
        solver.neighbors, lists,
        "update_densities must restore the lists"
    );
    for p in &solver.particles {
        assert!(p.density.is_finite() && p.density > 0.0);
    }
    solver.update_stresses();
    assert_eq!(
        solver.neighbors, lists,
        "update_stresses must restore the lists"
    );
    solver.compute_accelerations();
    assert_eq!(
        solver.neighbors, lists,
        "compute_accelerations must restore the lists"
    );
    for p in &solver.particles {
        assert!(p.acceleration.iter().all(|a| a.is_finite()));
    }

    // A whole step builds the lists from the positions it starts with and keeps them.
    let before_step = solver.particles.clone();
    solver.step();
    assert_eq!(solver.neighbors, all_pairs_neighbors(&before_step));
    assert!(solver.time > 0.0);
}
