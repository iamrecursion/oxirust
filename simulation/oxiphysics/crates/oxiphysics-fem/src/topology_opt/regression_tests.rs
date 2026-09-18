//! Regression tests for the topology-optimisation audit fixes.
//!
//! * `filter_sensitivities_pde` performed one explicit sweep whose diagonal
//!   already contained the Laplacian's `−2φ`, so constants shrank by
//!   `1 / (1 + 4 r² / h²)` and `(1 − r²∇²) φ = ψ` was never solved.
//! * `update_densities_mma` was the optimality-criteria heuristic with the
//!   asymptotes used as box bounds.
//! * `OcOptimizer::has_converged` measured the distance to the *initial* uniform
//!   density instead of the change made by the last step.

use super::functions::filter_sensitivities_pde;
use super::types::{OcOptimizer, SimpParams, TopologyGrid};

/// Deterministic generator (64-bit LCG, high bits).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn signed_unit(&mut self) -> f64 {
        (self.next() % 20_001) as f64 / 10_000.0 - 1.0
    }
}

fn grid(nx: usize, ny: usize, dx: f64, dy: f64) -> TopologyGrid {
    TopologyGrid::new_uniform(nx, ny, 0.5, 1.0, dx, dy)
}

// ---------------------------------------------------------------------------
// Helmholtz (PDE) filter
// ---------------------------------------------------------------------------

/// `(A φ)ᵢ` of the finite-volume Helmholtz operator with zero-flux boundaries,
/// written independently of the production assembly.
fn apply_helmholtz(phi: &[f64], nx: usize, ny: usize, dx: f64, dy: f64, r: f64) -> Vec<f64> {
    let mut out = vec![0.0; nx * ny];
    for row in 0..ny {
        for col in 0..nx {
            let i = row * nx + col;
            let mut flux = 0.0;
            if col > 0 {
                flux += (phi[i] - phi[i - 1]) / (dx * dx);
            }
            if col + 1 < nx {
                flux += (phi[i] - phi[i + 1]) / (dx * dx);
            }
            if row > 0 {
                flux += (phi[i] - phi[i - nx]) / (dy * dy);
            }
            if row + 1 < ny {
                flux += (phi[i] - phi[i + nx]) / (dy * dy);
            }
            out[i] = phi[i] + r * r * flux;
        }
    }
    out
}

#[test]
fn pde_filter_preserves_constant_fields() {
    // The single explicit sweep returned c / (1 + 4 r² / h²) for a constant c.
    for (nx, ny, dx, dy) in [
        (5, 7, 1.0, 1.0),
        (6, 4, 0.5, 2.0),
        (1, 6, 1.0, 0.25),
        (6, 1, 3.0, 1.0),
        (1, 1, 1.0, 1.0),
        (12, 12, 0.1, 0.1),
    ] {
        let g = grid(nx, ny, dx, dy);
        for r in [0.0, 0.3, 1.0, 2.5, 10.0] {
            let psi = vec![-3.7; nx * ny];
            let phi = filter_sensitivities_pde(&g, &psi, r);
            assert_eq!(phi.len(), nx * ny);
            for (i, v) in phi.iter().enumerate() {
                assert!(
                    (v + 3.7).abs() < 1e-12 * 3.7,
                    "{nx}x{ny} h=({dx},{dy}) r={r}: phi[{i}] = {v}"
                );
            }
        }
    }
}

#[test]
fn pde_filter_conserves_the_integral_of_the_field() {
    // 1ᵀ A = 1ᵀ for the zero-flux operator, so Σφ = Σψ.
    let mut rng = Lcg(42);
    for (nx, ny, dx, dy, r) in [
        (8, 6, 1.0, 1.0, 1.5),
        (7, 9, 0.5, 2.0, 0.8),
        (20, 3, 1.0, 0.5, 4.0),
    ] {
        let g = grid(nx, ny, dx, dy);
        let psi: Vec<f64> = (0..nx * ny).map(|_| rng.signed_unit()).collect();
        let phi = filter_sensitivities_pde(&g, &psi, r);
        let sum_psi: f64 = psi.iter().sum();
        let sum_phi: f64 = phi.iter().sum();
        let scale: f64 = psi.iter().map(|v| v.abs()).sum();
        assert!(
            (sum_phi - sum_psi).abs() < 1e-10 * scale,
            "{nx}x{ny}: sum phi = {sum_phi}, sum psi = {sum_psi}"
        );
    }
}

#[test]
fn pde_filter_solves_the_discrete_helmholtz_equation() {
    let mut rng = Lcg(7);
    for (nx, ny, dx, dy, r) in [
        (9, 9, 1.0, 1.0, 1.2),
        (6, 11, 0.4, 1.3, 0.9),
        (30, 30, 1.0, 1.0, 12.0),
    ] {
        let g = grid(nx, ny, dx, dy);
        let psi: Vec<f64> = (0..nx * ny).map(|_| rng.signed_unit()).collect();
        let phi = filter_sensitivities_pde(&g, &psi, r);
        let back = apply_helmholtz(&phi, nx, ny, dx, dy, r);
        let worst = psi
            .iter()
            .zip(back.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst < 1e-9,
            "{nx}x{ny} r={r}: max |A phi - psi| = {worst:e}"
        );
    }
}

#[test]
fn pde_filter_impulse_response_is_symmetric_positive_and_decaying() {
    let (nx, ny) = (11, 11);
    let g = grid(nx, ny, 1.0, 1.0);
    let centre = (ny / 2) * nx + nx / 2;
    let mut psi = vec![0.0; nx * ny];
    psi[centre] = 1.0;
    let phi = filter_sensitivities_pde(&g, &psi, 1.5);
    assert!(
        phi.iter().all(|&v| v > 0.0),
        "inverse of an M-matrix is positive"
    );
    assert!(phi[centre] < 1.0, "the impulse is smoothed");
    assert!((phi.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    for row in 0..ny {
        for col in 0..nx {
            let v = phi[row * nx + col];
            let mirrored_x = phi[row * nx + (nx - 1 - col)];
            let mirrored_y = phi[(ny - 1 - row) * nx + col];
            let transposed = phi[col * nx + row];
            assert!((v - mirrored_x).abs() < 1e-9, "x symmetry at ({row},{col})");
            assert!((v - mirrored_y).abs() < 1e-9, "y symmetry at ({row},{col})");
            assert!(
                (v - transposed).abs() < 1e-9,
                "diagonal symmetry at ({row},{col})"
            );
        }
    }
    // Strictly decreasing along the centre row away from the impulse.
    let row = ny / 2;
    for col in nx / 2..nx - 1 {
        assert!(phi[row * nx + col] > phi[row * nx + col + 1], "col {col}");
    }
}

#[test]
fn pde_filter_handles_degenerate_arguments() {
    let g = grid(4, 3, 1.0, 1.0);
    let psi: Vec<f64> = (0..12).map(|i| i as f64 - 5.0).collect();
    // r = 0 is the identity; so is a non-finite radius.
    assert_eq!(filter_sensitivities_pde(&g, &psi, 0.0), psi);
    assert_eq!(filter_sensitivities_pde(&g, &psi, f64::NAN), psi);
    assert_eq!(filter_sensitivities_pde(&g, &psi, f64::INFINITY), psi);
    // Zero input stays zero.
    assert_eq!(filter_sensitivities_pde(&g, &[0.0; 12], 1.0), vec![0.0; 12]);
    // Non-finite sensitivities count as zero and never poison the result.
    let mut poisoned = psi.clone();
    poisoned[3] = f64::NAN;
    poisoned[8] = f64::INFINITY;
    let mut cleaned = psi.clone();
    cleaned[3] = 0.0;
    cleaned[8] = 0.0;
    let a = filter_sensitivities_pde(&g, &poisoned, 1.3);
    let b = filter_sensitivities_pde(&g, &cleaned, 1.3);
    assert!(a.iter().all(|v| v.is_finite()));
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-12);
    }
    // An empty grid has an empty field.
    let empty = TopologyGrid::new_uniform(0, 0, 0.5, 1.0, 1.0, 1.0);
    assert!(filter_sensitivities_pde(&empty, &[], 1.0).is_empty());
    // A negative radius acts like its magnitude (only r² enters).
    let plus = filter_sensitivities_pde(&g, &psi, 0.9);
    let minus = filter_sensitivities_pde(&g, &psi, -0.9);
    assert_eq!(plus, minus);
}

#[test]
fn pde_filter_tends_to_the_mean_for_a_huge_radius() {
    let (nx, ny) = (10, 10);
    let g = grid(nx, ny, 1.0, 1.0);
    let mut rng = Lcg(3);
    let psi: Vec<f64> = (0..nx * ny).map(|_| rng.signed_unit()).collect();
    let mean = psi.iter().sum::<f64>() / psi.len() as f64;
    let phi = filter_sensitivities_pde(&g, &psi, 1.0e3);
    for (i, v) in phi.iter().enumerate() {
        assert!((v - mean).abs() < 1e-3, "phi[{i}] = {v}, mean {mean}");
    }
}

#[test]
#[should_panic(expected = "expected one sensitivity per element")]
fn pde_filter_rejects_a_sensitivity_vector_of_the_wrong_length() {
    let g = grid(3, 3, 1.0, 1.0);
    let _ = filter_sensitivities_pde(&g, &[1.0; 8], 1.0);
}

// ---------------------------------------------------------------------------
// MMA density update
// ---------------------------------------------------------------------------

fn simp_params(volume_fraction: f64, move_limit: f64) -> SimpParams {
    SimpParams {
        e0: 1.0,
        emin: 1e-9,
        penalty: 3.0,
        volume_fraction,
        filter_radius: 1.5,
        move_limit,
    }
}

fn densities(g: &TopologyGrid) -> Vec<f64> {
    g.elements.iter().map(|e| e.rho).collect()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

#[test]
fn mma_meets_the_volume_target_inside_the_trust_region() {
    let params = simp_params(0.4, 0.2);
    let mut g = grid(6, 6, 1.0, 1.0);
    let (lo, hi) = g.mma_initial_asymptotes();
    g.update_densities_mma(&vec![-1.0; 36], &params, &lo, &hi);
    let rho = densities(&g);
    // The volume constraint is imposed exactly on the updated design.
    assert!((mean(&rho) - 0.4).abs() < 1e-9, "mean = {}", mean(&rho));
    for (i, &r) in rho.iter().enumerate() {
        assert!((0.3 - 1e-12..=0.7 + 1e-12).contains(&r), "rho[{i}] = {r}");
        assert!(
            (r - rho[0]).abs() < 1e-12,
            "identical elements stay identical"
        );
    }
    for e in &g.elements {
        let expected = params.emin + e.rho.powf(3.0) * (params.e0 - params.emin);
        assert!(
            (e.stiffness - expected).abs() < 1e-15,
            "stiffness is refreshed"
        );
    }
}

#[test]
fn mma_converges_to_the_analytic_optimum_of_a_separable_compliance() {
    // min Σ wᵢ / xᵢ  s.t.  mean(x) = V has x*ᵢ = n V √wᵢ / Σⱼ √wⱼ.  With the lower
    // asymptote at 0 the MMA term q/(x − 0) reproduces wᵢ/xᵢ exactly.
    let w = [1.0, 4.0, 9.0, 16.0];
    let volume = 0.5;
    let optimum = [0.2, 0.4, 0.6, 0.8];
    let params = simp_params(volume, 0.2);
    let mut g = grid(4, 1, 1.0, 1.0);
    let lower = [0.0; 4];
    let upper = [1.0; 4];
    let objective = |x: &[f64]| -> f64 { x.iter().zip(w.iter()).map(|(xi, wi)| wi / xi).sum() };
    let mut previous = objective(&densities(&g));
    let mut converged_at = None;
    for iteration in 0..200 {
        let x = densities(&g);
        let sens: Vec<f64> = x
            .iter()
            .zip(w.iter())
            .map(|(xi, wi)| -wi / (xi * xi))
            .collect();
        g.update_densities_mma(&sens, &params, &lower, &upper);
        let x_new = densities(&g);
        assert!(
            (mean(&x_new) - volume).abs() < 1e-9,
            "volume at {iteration}"
        );
        let value = objective(&x_new);
        assert!(
            value <= previous + 1e-9,
            "objective rose: {previous} -> {value}"
        );
        previous = value;
        let change = x
            .iter()
            .zip(x_new.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        if change < 1e-12 {
            converged_at = Some(iteration);
            break;
        }
    }
    let converged_at = converged_at.expect("MMA must reach a fixed point");
    assert!(
        converged_at < 150,
        "slow convergence: {converged_at} iterations"
    );
    for (got, want) in densities(&g).iter().zip(optimum.iter()) {
        assert!((got - want).abs() < 1e-6, "{got} vs analytic {want}");
    }
}

#[test]
fn mma_moves_density_towards_the_stiffer_elements() {
    let params = simp_params(0.5, 0.2);
    let mut g = grid(4, 1, 1.0, 1.0);
    let (lo, hi) = g.mma_initial_asymptotes();
    // Elements 0 and 1 have a much larger compliance sensitivity.
    g.update_densities_mma(&[-40.0, -40.0, -0.5, -0.5], &params, &lo, &hi);
    let rho = densities(&g);
    assert!(rho[0] > 0.5 && rho[1] > 0.5, "{rho:?}");
    assert!(rho[2] < 0.5 && rho[3] < 0.5, "{rho:?}");
    assert!((mean(&rho) - 0.5).abs() < 1e-9);
}

#[test]
fn mma_design_does_not_depend_on_the_scale_of_the_sensitivities() {
    let params = simp_params(0.45, 0.2);
    let base: Vec<f64> = (0..12).map(|i| -(1.0 + i as f64)).collect();
    let mut reference = grid(4, 3, 1.0, 1.0);
    let (lo, hi) = reference.mma_initial_asymptotes();
    reference.update_densities_mma(&base, &params, &lo, &hi);
    for factor in [1e-20, 1e-6, 1e6, 1e20] {
        let mut scaled = grid(4, 3, 1.0, 1.0);
        let sens: Vec<f64> = base.iter().map(|s| s * factor).collect();
        scaled.update_densities_mma(&sens, &params, &lo, &hi);
        for (a, b) in densities(&reference).iter().zip(densities(&scaled).iter()) {
            assert!((a - b).abs() < 1e-9, "factor {factor}: {a} vs {b}");
        }
    }
}

#[test]
fn mma_leaves_an_inactive_volume_constraint_alone() {
    // Every element wants more material and the target (0.95) is far away: the
    // update goes to the upper trust-region bound min(1, 0.5 + 0.2, safeguard) = 0.7.
    let params = simp_params(0.95, 0.2);
    let mut g = grid(3, 3, 1.0, 1.0);
    let (lo, hi) = g.mma_initial_asymptotes();
    g.update_densities_mma(&[-1.0; 9], &params, &lo, &hi);
    for (i, r) in densities(&g).iter().enumerate() {
        assert!((r - 0.7).abs() < 1e-12, "rho[{i}] = {r}");
    }
    // Zero sensitivities carry no information: nothing moves while the constraint holds.
    let mut idle = grid(3, 3, 1.0, 1.0);
    let params = simp_params(0.5, 0.2);
    let (lo, hi) = idle.mma_initial_asymptotes();
    idle.update_densities_mma(&[0.0; 9], &params, &lo, &hi);
    for r in densities(&idle) {
        assert!((r - 0.5).abs() < 1e-12, "{r}");
    }
}

#[test]
fn mma_returns_the_lowest_reachable_design_when_the_target_is_out_of_reach() {
    // Target 0.01 is unreachable in one step with a move limit of 0.2: every
    // element goes to its lower trust-region bound 0.5 - 0.2 = 0.3.
    let params = simp_params(0.01, 0.2);
    let mut g = grid(3, 3, 1.0, 1.0);
    let (lo, hi) = g.mma_initial_asymptotes();
    g.update_densities_mma(&[-1.0; 9], &params, &lo, &hi);
    for r in densities(&g) {
        assert!((r - 0.3).abs() < 1e-12, "{r}");
    }
}

#[test]
fn mma_lowers_density_where_more_material_hurts_the_objective() {
    // dc/drho > 0: the objective grows with the density, so the design drops to
    // the lower trust-region bound.
    let params = simp_params(0.5, 0.2);
    let mut g = grid(3, 2, 1.0, 1.0);
    let (lo, hi) = g.mma_initial_asymptotes();
    g.update_densities_mma(&[1.0; 6], &params, &lo, &hi);
    for r in densities(&g) {
        assert!((r - 0.3).abs() < 1e-12, "{r}");
    }
}

#[test]
fn mma_replaces_missing_or_invalid_asymptotes_by_the_initial_ones() {
    let params = simp_params(0.4, 0.2);
    let sens: Vec<f64> = (0..8).map(|i| -(0.5 + i as f64)).collect();
    let mut reference = grid(4, 2, 1.0, 1.0);
    let (lo, hi) = reference.mma_initial_asymptotes();
    reference.update_densities_mma(&sens, &params, &lo, &hi);
    // Missing entries.
    let mut missing = grid(4, 2, 1.0, 1.0);
    missing.update_densities_mma(&sens, &params, &[], &[]);
    // Asymptotes that do not bracket the density (or are not finite).
    let mut invalid = grid(4, 2, 1.0, 1.0);
    let bad_lower = [0.9, f64::NAN, 0.5, 0.7, f64::NEG_INFINITY, 2.0, 0.6, 0.5];
    let bad_upper = [0.1, f64::INFINITY, 0.5, 0.2, f64::NAN, -1.0, 0.4, 0.5];
    invalid.update_densities_mma(&sens, &params, &bad_lower, &bad_upper);
    for ((a, b), c) in densities(&reference)
        .iter()
        .zip(densities(&missing).iter())
        .zip(densities(&invalid).iter())
    {
        assert!(
            (a - b).abs() < 1e-12 && (a - c).abs() < 1e-12,
            "{a} {b} {c}"
        );
    }
}

#[test]
fn mma_asymptotes_far_from_the_iterate_allow_larger_steps_than_close_ones() {
    // Close asymptotes make the sub-problem conservative: the safeguard
    // L + 0.1 (x - L) / U - 0.1 (U - x) shrinks the trust region.
    let params = simp_params(0.3, 0.45);
    let sens = [-1.0; 4];
    let mut wide = grid(2, 2, 1.0, 1.0);
    let (lo, hi) = wide.mma_initial_asymptotes();
    wide.update_densities_mma(&sens, &params, &lo, &hi);
    let mut tight = grid(2, 2, 1.0, 1.0);
    let lo_tight = [0.45; 4];
    let hi_tight = [0.55; 4];
    tight.update_densities_mma(&sens, &params, &lo_tight, &hi_tight);
    // Target 0.3 needs a decrease of 0.2: reachable for wide asymptotes, but
    // the tight safeguard 0.45 + 0.1 * 0.05 = 0.455 stops at 0.455.
    assert!((mean(&densities(&wide)) - 0.3).abs() < 1e-9);
    for r in densities(&tight) {
        assert!((r - 0.455).abs() < 1e-12, "{r}");
    }
}

#[test]
#[should_panic(expected = "expected one sensitivity per element")]
fn mma_rejects_a_sensitivity_vector_of_the_wrong_length() {
    let mut g = grid(2, 2, 1.0, 1.0);
    let (lo, hi) = g.mma_initial_asymptotes();
    g.update_densities_mma(&[-1.0; 3], &simp_params(0.5, 0.2), &lo, &hi);
}

// ---------------------------------------------------------------------------
// OC optimizer convergence metric
// ---------------------------------------------------------------------------

fn set_sensitivities(opt: &mut OcOptimizer, value: f64) {
    for e in opt.grid.elements.iter_mut() {
        e.sensitivity = value;
    }
}

#[test]
fn a_fresh_optimizer_is_not_converged() {
    // The old metric |rho - volume_fraction| is zero for the uniform start.
    let opt = OcOptimizer::new(4, 4, SimpParams::default_steel());
    let changes = opt.has_converged(1e9);
    assert_eq!(
        changes.len(),
        16,
        "no step yet: never reported as converged"
    );
    assert!(changes.iter().all(|&c| c == 0.0));
    assert_eq!(opt.max_density_change(), f64::INFINITY);
}

#[test]
fn uniform_sensitivities_are_a_fixed_point_and_converge() {
    let mut opt = OcOptimizer::new(4, 4, SimpParams::default_steel());
    set_sensitivities(&mut opt, -1.0);
    opt.step();
    opt.step();
    // rho = volume fraction with uniform sensitivities is the OC fixed point; the
    // 50-step bisection on [0, 1e9] only resolves the multiplier to ~1e-6.
    assert!(
        opt.has_converged(1e-5).is_empty(),
        "changes: {:?}",
        opt.has_converged(1e-5)
    );
    assert!(
        opt.max_density_change() < 1e-5,
        "{}",
        opt.max_density_change()
    );
    // A tolerance below the achievable change reports the changes instead.
    let strict = opt.has_converged(0.0);
    assert_eq!(strict.len(), 16);
}

#[test]
fn a_converged_design_may_be_far_from_the_initial_volume_fraction() {
    // Solid left half (strong sensitivity), void right half (weak sensitivity):
    // a saturated 0/1 design is stationary although every element differs from
    // the volume fraction by about 0.5, so the old metric never converged.
    let (nx, ny) = (4, 4);
    let mut params = SimpParams::default_steel();
    params.filter_radius = 0.1; // below the element spacing: no neighbour mixing
    let mut opt = OcOptimizer::new(nx, ny, params);
    for (i, e) in opt.grid.elements.iter_mut().enumerate() {
        let solid = i % nx < nx / 2;
        e.rho = if solid { 1.0 } else { 1e-3 };
        e.sensitivity = if solid { -1.0e6 } else { -1.0 };
    }
    opt.params.volume_fraction = opt.grid.volume_fraction();
    for _ in 0..3 {
        opt.step();
    }
    let vf = opt.params.volume_fraction;
    assert!(
        opt.grid.elements.iter().all(|e| (e.rho - vf).abs() > 0.4),
        "design must be far from the volume fraction"
    );
    assert!(
        opt.has_converged(1e-6).is_empty(),
        "changes: {:?}",
        opt.has_converged(1e-6)
    );
    assert!(opt.max_density_change() < 1e-6);
}

#[test]
fn has_converged_reports_the_changes_of_the_last_step() {
    let mut opt = OcOptimizer::new(4, 4, SimpParams::default_steel());
    for (i, e) in opt.grid.elements.iter_mut().enumerate() {
        e.sensitivity = -(1.0 + i as f64);
    }
    let before: Vec<f64> = opt.density_field();
    opt.step();
    let after = opt.density_field();
    let expected: Vec<f64> = before
        .iter()
        .zip(after.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();
    assert!(expected.iter().any(|&c| c > 1e-3), "the design must move");
    let reported = opt.has_converged(1e-3);
    assert_eq!(reported.len(), 16);
    for (got, want) in reported.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-15, "{got} vs {want}");
    }
    let largest = expected.iter().cloned().fold(0.0_f64, f64::max);
    assert!((opt.max_density_change() - largest).abs() < 1e-15);
}

#[test]
fn the_change_is_measured_against_the_previous_step_not_the_start() {
    let mut opt = OcOptimizer::new(3, 3, SimpParams::default_steel());
    for (i, e) in opt.grid.elements.iter_mut().enumerate() {
        e.sensitivity = -(1.0 + i as f64);
    }
    let start = opt.density_field();
    opt.step();
    let after_first = opt.density_field();
    opt.step();
    let after_second = opt.density_field();
    let reported = opt.has_converged(0.0);
    assert_eq!(reported.len(), 9);
    let mut differs_from_the_start = false;
    for i in 0..9 {
        let last_step = (after_second[i] - after_first[i]).abs();
        let since_start = (after_second[i] - start[i]).abs();
        assert!(
            (reported[i] - last_step).abs() < 1e-15,
            "element {i}: reported {} vs last-step change {last_step}",
            reported[i]
        );
        if (last_step - since_start).abs() > 1e-6 {
            differs_from_the_start = true;
        }
    }
    assert!(
        differs_from_the_start,
        "the test must distinguish the last step from the distance to the start"
    );
}
