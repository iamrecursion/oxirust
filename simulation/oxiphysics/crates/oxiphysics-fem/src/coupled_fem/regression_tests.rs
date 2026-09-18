//! Regression tests for the coupled-FEM findings of the production-readiness audit:
//! F1106 (`ConformingTransfer`), F1107 (explicit-step stability errors) and
//! F1108 (`PodRom` POD basis).

use super::*;
use crate::Error;

/// Deterministic splitmix64 generator for reproducible pseudo-random test data.
struct SplitMix(u64);

impl SplitMix {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Message of an `Error::InvalidInput`; any other outcome fails the test.
fn invalid_input_message<T: std::fmt::Debug>(result: crate::Result<T>) -> String {
    match result {
        Err(Error::InvalidInput(message)) => message,
        other => panic!("expected Error::InvalidInput, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// F1107 — time-step functions return errors instead of panicking
// ---------------------------------------------------------------------------

#[test]
fn chemo_diffusion_step_rejects_unstable_dt_and_keeps_state() {
    let mut prob = ChemoMechanicalProblem::new_1d(11, 1.0, 1.0);
    prob.concentration[3] = 2.5;
    let dt_max = prob.max_stable_dt();
    assert!((dt_max - 0.5 * prob.h * prob.h / prob.diff_coeff).abs() <= 1e-12 * dt_max);
    let before = prob.concentration.clone();
    let message = invalid_input_message(prob.diffusion_step(1.5 * dt_max, 1.0, 1.0));
    assert!(message.contains("Fourier number"), "{message}");
    assert!(
        message.contains(&format!("{dt_max:e}")),
        "the error must carry the stable limit {dt_max:e}: {message}"
    );
    assert_eq!(
        prob.concentration, before,
        "state must be untouched on error"
    );
}

#[test]
fn chemo_coupled_step_error_leaves_concentration_and_stress_untouched() {
    let mut prob = ChemoMechanicalProblem::new_1d(11, 1.0, 1.0);
    prob.concentration[5] = 3.0;
    prob.update_stress_from_concentration(1.0);
    let (c_before, s_before) = (prob.concentration.clone(), prob.hydrostatic_stress.clone());
    let result = prob.coupled_step(2.0 * prob.max_stable_dt(), 1.0, 1.0, 1.0);
    assert!(matches!(result, Err(Error::InvalidInput(_))));
    assert_eq!(prob.concentration, c_before);
    assert_eq!(prob.hydrostatic_stress, s_before);
    // A stable step goes through and refreshes the stress field.
    prob.coupled_step(prob.max_stable_dt(), 1.0, 1.0, 1.0)
        .unwrap();
    assert_ne!(prob.concentration, c_before);
}

#[test]
fn chemo_diffusion_step_accepts_exact_stability_limit() {
    // dt = max_stable_dt() gives Fo = ½ up to rounding and must not be rejected;
    // the update is then a convex average, so the extrema cannot grow.
    let mut rng = SplitMix(11);
    for &(n, length, diff) in &[(11usize, 1.0, 1e-14), (21, 0.37, 3.3e-9), (6, 12.0, 0.7)] {
        let mut prob = ChemoMechanicalProblem::new_1d(n, length, 0.0);
        prob.diff_coeff = diff;
        for c in prob.concentration.iter_mut() {
            *c = rng.uniform(0.0, 1.0);
        }
        let dt = prob.max_stable_dt();
        prob.diffusion_step(dt, 0.3, 0.7).unwrap();
        assert!(
            prob.concentration.iter().all(|c| (0.0..=1.0).contains(c)),
            "maximum principle violated at the stability limit: {:?}",
            prob.concentration
        );
        // A step just above the limit is rejected.
        assert!(prob.diffusion_step(dt * (1.0 + 1e-6), 0.3, 0.7).is_err());
    }
}

#[test]
fn chemo_diffusion_step_rejects_invalid_arguments() {
    let mut prob = ChemoMechanicalProblem::new_1d(11, 1.0, 1.0);
    let before = prob.concentration.clone();
    for dt in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        assert!(
            matches!(
                prob.diffusion_step(dt, 1.0, 1.0),
                Err(Error::InvalidInput(_))
            ),
            "dt = {dt}"
        );
    }
    assert!(prob.diffusion_step(1.0, f64::NAN, 1.0).is_err());
    assert!(prob.diffusion_step(1.0, 1.0, f64::INFINITY).is_err());
    // Degenerate grids used to underflow `n_nodes - 2` (debug) or index out of bounds.
    let mut single = ChemoMechanicalProblem::new_1d(11, 1.0, 1.0);
    single.n_nodes = 1;
    single.concentration = vec![1.0];
    assert!(single.diffusion_step(1.0, 1.0, 1.0).is_err());
    let mut ragged = ChemoMechanicalProblem::new_1d(11, 1.0, 1.0);
    ragged.concentration.truncate(5);
    assert!(ragged.diffusion_step(1.0, 1.0, 1.0).is_err());
    let mut anti = ChemoMechanicalProblem::new_1d(11, 1.0, 1.0);
    anti.diff_coeff = -1e-14;
    assert!(anti.diffusion_step(1.0, 1.0, 1.0).is_err());
    assert_eq!(prob.concentration, before);
    // A non-diffusing species has no stability limit.
    let mut inert = ChemoMechanicalProblem::new_1d(11, 1.0, 1.0);
    inert.diff_coeff = 0.0;
    assert!(inert.max_stable_dt().is_infinite());
    inert.diffusion_step(1e30, 1.0, 1.0).unwrap();
}

#[test]
fn chemo_diffusion_matches_analytic_mode_when_sub_cycled() {
    // c(x, t) = sin(πx/L)·exp(−D (π/L)² t) for c = 0 at both ends.  The physical
    // step is far above the explicit limit, so it is sub-cycled with max_stable_dt.
    let n = 21;
    let mut prob = ChemoMechanicalProblem::new_1d(n, 1.0, 0.0);
    for i in 0..n {
        prob.concentration[i] = (std::f64::consts::PI * i as f64 * prob.h).sin();
    }
    let k2 = std::f64::consts::PI.powi(2);
    let t_end = 0.5 / (prob.diff_coeff * k2);
    let sub_steps = (t_end / prob.max_stable_dt()).ceil() as usize;
    assert!(sub_steps >= 30, "the test must exercise sub-cycling");
    let dt = t_end / sub_steps as f64;
    for _ in 0..sub_steps {
        prob.diffusion_step(dt, 0.0, 0.0).unwrap();
    }
    let mid = prob.concentration[n / 2];
    let exact = (-0.5_f64).exp();
    assert!(
        (mid - exact).abs() <= 2e-2 * exact,
        "mid-point concentration {mid}, analytic {exact}"
    );
}

fn pulse_cavity(n_nodes: usize) -> AcousticCavity1D {
    let mut cav = AcousticCavity1D::new(n_nodes, 1.0, 343.0, 1.225);
    for i in 0..n_nodes {
        let x = i as f64 * cav.h - 0.5;
        cav.pressure[i] = 100.0 * (-(x * x) / 0.0025).exp();
        cav.velocity[i] = 0.01 * x;
    }
    cav
}

#[test]
fn acoustic_step_rejects_cfl_violation_and_keeps_state() {
    let mut cav = pulse_cavity(41);
    let dt_max = cav.max_stable_dt();
    assert!((dt_max - cav.h / cav.c_sound).abs() <= 1e-15 * dt_max);
    let (p0, v0) = (cav.pressure.clone(), cav.velocity.clone());
    let message = invalid_input_message(cav.step(1.1 * dt_max));
    assert!(message.contains("CFL"), "{message}");
    assert!(
        message.contains(&format!("{dt_max:e}")),
        "the error must carry the stable limit {dt_max:e}: {message}"
    );
    assert_eq!(cav.pressure, p0);
    assert_eq!(cav.velocity, v0);
}

#[test]
fn acoustic_rigid_wall_step_error_leaves_wall_nodes_untouched() {
    let mut cav = pulse_cavity(41);
    cav.pressure[0] = 7.0;
    cav.velocity[40] = -0.3;
    let (p0, v0) = (cav.pressure.clone(), cav.velocity.clone());
    assert!(matches!(
        cav.step_rigid_walls(2.0 * cav.max_stable_dt()),
        Err(Error::InvalidInput(_))
    ));
    assert_eq!(
        cav.pressure, p0,
        "wall pressures must not be modified on error"
    );
    assert_eq!(
        cav.velocity, v0,
        "wall velocities must not be modified on error"
    );
    // A valid step still works and enforces the rigid-wall velocity.
    cav.step_rigid_walls(0.5 * cav.max_stable_dt()).unwrap();
    assert_eq!(cav.velocity[0], 0.0);
    assert_eq!(cav.velocity[40], 0.0);
}

#[test]
fn acoustic_step_accepts_exact_cfl_limit_for_many_parameter_sets() {
    // c·(h/c)/h evaluates to 1 + O(ε) for many parameter sets; the limit itself
    // must remain usable.  r = 1 makes the Lax-Wendroff update an exact shift.
    let mut rng = SplitMix(5);
    for _ in 0..200 {
        let n = 5 + (rng.next_f64() * 60.0) as usize;
        let length = rng.uniform(0.1, 10.0);
        let c_sound = rng.uniform(50.0, 6000.0);
        let mut cav = AcousticCavity1D::new(n, length, c_sound, 1.2);
        cav.pressure[n / 2] = 1.0;
        let dt = cav.max_stable_dt();
        cav.step(dt).unwrap();
        cav.step_rigid_walls(dt).unwrap();
        assert!(cav.pressure.iter().all(|p| p.is_finite()));
    }
}

#[test]
fn acoustic_step_rejects_invalid_arguments_without_panicking() {
    let mut cav = pulse_cavity(21);
    let before = (cav.pressure.clone(), cav.velocity.clone());
    for dt in [f64::NAN, f64::INFINITY, 0.0, -1e-6] {
        assert!(
            matches!(cav.step(dt), Err(Error::InvalidInput(_))),
            "dt = {dt}"
        );
        assert!(cav.step_rigid_walls(dt).is_err(), "rigid walls, dt = {dt}");
    }
    let mut mismatched = pulse_cavity(21);
    mismatched.velocity.pop();
    assert!(mismatched.step(1e-6).is_err());
    let mut unphysical = pulse_cavity(21);
    unphysical.rho_f = 0.0;
    assert!(unphysical.step(1e-6).is_err());
    assert_eq!((cav.pressure, cav.velocity), before);
}

#[test]
fn acoustic_cavity_without_interior_is_a_validated_no_op() {
    // Two nodes (or none) have no interior; the former `n_nodes - 1` range underflowed.
    let mut two = AcousticCavity1D::new(2, 1.0, 343.0, 1.225);
    two.pressure = vec![3.0, 4.0];
    let dt = two.max_stable_dt();
    two.step(dt).unwrap();
    two.step_rigid_walls(dt).unwrap();
    assert_eq!(two.pressure, vec![3.0, 4.0]);
    assert!(two.step(10.0 * dt).is_err(), "the CFL check still applies");
    let mut empty = AcousticCavity1D::new(2, 1.0, 343.0, 1.225);
    empty.n_nodes = 0;
    empty.pressure.clear();
    empty.velocity.clear();
    empty.step(1e-6).unwrap();
    assert!(empty.max_stable_dt() > 0.0);
}

// ---------------------------------------------------------------------------
// F1106 — ConformingTransfer
// ---------------------------------------------------------------------------

/// Trapezoid (tributary-length) weights of a 1-D node distribution.
fn tributary_lengths(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|i| {
            let left = if i > 0 { x[i] - x[i - 1] } else { 0.0 };
            let right = if i + 1 < n { x[i + 1] - x[i] } else { 0.0 };
            0.5 * (left + right)
        })
        .collect()
}

#[test]
fn conforming_transfer_rejects_degenerate_source_meshes() {
    // n_src < 2 with a target used to underflow `n_src - 2` (debug) / read
    // out of bounds (release).
    assert!(matches!(
        ConformingTransfer::linear_interpolation(vec![0.5], vec![0.2]),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        ConformingTransfer::linear_interpolation(Vec::new(), vec![0.2]),
        Err(Error::InvalidInput(_))
    ));
    // Unsorted, repeated and non-finite source positions.
    for bad in [
        vec![0.0, 0.5, 0.4, 1.0],
        vec![0.0, 0.5, 0.5, 1.0],
        vec![0.0, f64::NAN, 1.0],
        vec![0.0, 1.0, f64::INFINITY],
    ] {
        assert!(
            ConformingTransfer::linear_interpolation(bad.clone(), vec![0.3]).is_err(),
            "{bad:?}"
        );
    }
    assert!(ConformingTransfer::linear_interpolation(vec![0.0, 1.0], vec![f64::NAN]).is_err());
    // Two source nodes are the minimum and work.
    let tf = ConformingTransfer::linear_interpolation(vec![0.0, 1.0], vec![0.25]).unwrap();
    assert!((tf.transfer(&[2.0, 6.0]).unwrap()[0] - 3.0).abs() < 1e-14);
}

#[test]
fn conforming_transfer_reproduces_linear_fields_exactly() {
    let src = vec![0.0, 0.3, 0.45, 1.0, 1.7];
    let tgt = vec![-0.2, 0.0, 0.1, 0.3, 0.77, 1.0, 1.45, 1.7, 2.0];
    let tf = ConformingTransfer::linear_interpolation(src.clone(), tgt.clone()).unwrap();
    let field = |x: f64| 2.5 * x - 1.0;
    let values: Vec<f64> = src.iter().map(|&x| field(x)).collect();
    let out = tf.transfer(&values).unwrap();
    assert_eq!(out.len(), tgt.len());
    for (o, &x) in out.iter().zip(tgt.iter()) {
        assert!(
            (o - field(x)).abs() <= 1e-14,
            "x = {x}: {o} vs {}",
            field(x)
        );
    }
}

#[test]
fn conforming_transfer_stores_a_sparse_stencil_and_densifies_on_demand() {
    let src: Vec<f64> = (0..50).map(|i| i as f64 * 0.02).collect();
    let tgt: Vec<f64> = (0..37).map(|i| i as f64 * 0.0271).collect();
    let tf = ConformingTransfer::linear_interpolation(src.clone(), tgt.clone()).unwrap();
    assert_eq!(tf.stencil.len(), tgt.len(), "one stencil entry per target");
    assert_eq!((tf.n_src(), tf.n_tgt()), (src.len(), tgt.len()));
    let dense = tf.weights_dense();
    assert_eq!(dense.len(), tgt.len());
    let mut rng = SplitMix(3);
    let values: Vec<f64> = src.iter().map(|_| rng.uniform(-1.0, 1.0)).collect();
    let direct = tf.transfer(&values).unwrap();
    for (row, &d) in dense.iter().zip(direct.iter()) {
        assert_eq!(row.len(), src.len());
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() <= 1e-14, "weights must sum to one: {sum}");
        assert!(row.iter().filter(|w| **w != 0.0).count() <= 2);
        assert!((dot(row, &values) - d).abs() <= 1e-14);
    }
}

#[test]
fn conforming_transfer_rejects_wrong_value_counts() {
    let tf =
        ConformingTransfer::linear_interpolation(vec![0.0, 0.5, 1.0], vec![0.25, 0.75]).unwrap();
    assert!(matches!(
        tf.transfer(&[1.0, 2.0]),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        tf.transfer(&[1.0, 2.0, 3.0, 4.0]),
        Err(Error::InvalidInput(_))
    ));
    // The coupling residual propagates the mismatch instead of truncating silently.
    assert!(interface_residual(&[0.0, 0.0], &[1.0, 2.0], &tf).is_err());
    assert!(interface_residual(&[0.0], &[1.0, 2.0, 3.0], &tf).is_err());
    let r = interface_residual(&[1.0, 2.0], &[0.0, 2.0, 4.0], &tf).unwrap();
    assert!((r[0] - 0.0).abs() < 1e-14 && (r[1] - (2.0 - 3.0)).abs() < 1e-14);
}

#[test]
fn conservative_transfer_preserves_the_integral_to_machine_precision() {
    let src: Vec<f64> = (0..11).map(|i| i as f64 * 0.1).collect();
    let tgt = vec![0.0, 0.13, 0.31, 0.5, 0.62, 0.85, 1.0];
    let a_src = tributary_lengths(&src);
    let a_tgt = tributary_lengths(&tgt);
    let tf = ConformingTransfer::linear_interpolation(src.clone(), tgt.clone()).unwrap();
    let values: Vec<f64> = src.iter().map(|&x| 1.0 + (3.0 * x).sin()).collect();
    let source_integral = dot(&values, &a_src);
    // Plain interpolation does not conserve: the two meshes integrate differently.
    let raw = tf.transfer(&values).unwrap();
    assert!(
        (dot(&raw, &a_tgt) - source_integral).abs() > 1e-6 * source_integral,
        "the plain transfer already conserves; the test would not discriminate"
    );
    let conservative = tf.conservative_transfer(&values, &a_src, &a_tgt).unwrap();
    let target_integral = dot(&conservative, &a_tgt);
    assert!(
        (target_integral - source_integral).abs() <= 1e-12 * source_integral,
        "integral {target_integral} vs {source_integral}"
    );
    // Same field shape: the result is the interpolated field times a constant.
    let factor = conservative[3] / raw[3];
    for (c, r) in conservative.iter().zip(raw.iter()) {
        assert!((c - factor * r).abs() <= 1e-12 * c.abs().max(1.0));
    }
}

#[test]
fn conservative_transfer_handles_sign_changing_fields() {
    let src: Vec<f64> = (0..9).map(|i| i as f64 * 0.125).collect();
    let tgt = vec![0.0, 0.2, 0.55, 0.9, 1.0];
    let a_src = tributary_lengths(&src);
    let a_tgt = tributary_lengths(&tgt);
    let tf = ConformingTransfer::linear_interpolation(src.clone(), tgt).unwrap();
    let values: Vec<f64> = src.iter().map(|&x| 0.8 - x * x * 2.0).collect();
    let source_integral = dot(&values, &a_src);
    assert!(source_integral.abs() > 1e-3);
    let out = tf.conservative_transfer(&values, &a_src, &a_tgt).unwrap();
    assert!((dot(&out, &a_tgt) - source_integral).abs() <= 1e-12 * source_integral.abs());
}

#[test]
fn conservative_transfer_reports_inconsistent_inputs() {
    let tf = ConformingTransfer::linear_interpolation(vec![0.0, 1.0], vec![0.0, 0.5, 1.0]).unwrap();
    let ok_values = [1.0, 1.0];
    // Wrong number of areas.
    assert!(
        tf.conservative_transfer(&ok_values, &[0.5], &[0.25, 0.5, 0.25])
            .is_err()
    );
    assert!(
        tf.conservative_transfer(&ok_values, &[0.5, 0.5], &[0.25, 0.5])
            .is_err()
    );
    assert!(
        tf.conservative_transfer(&ok_values, &[0.5, f64::NAN], &[0.25, 0.5, 0.25])
            .is_err()
    );
    // Non-zero source integral but a target that integrates to zero: no scale
    // factor can restore it, so this must not silently return the raw values.
    let message =
        invalid_input_message(tf.conservative_transfer(&ok_values, &[0.5, 0.5], &[0.0, 0.0, 0.0]));
    assert!(message.contains("cannot"), "{message}");
    // If both integrals vanish the interpolated field is returned unchanged.
    let zero = tf
        .conservative_transfer(&[0.0, 0.0], &[0.5, 0.5], &[0.25, 0.5, 0.25])
        .unwrap();
    assert!(zero.iter().all(|v| *v == 0.0));
}

// ---------------------------------------------------------------------------
// F1108 — PodRom is a proper orthogonal decomposition
// ---------------------------------------------------------------------------

fn combination(coeffs: &[f64], vectors: &[&[f64]]) -> Vec<f64> {
    let n = vectors[0].len();
    let mut out = vec![0.0; n];
    for (c, v) in coeffs.iter().zip(vectors.iter()) {
        for (o, x) in out.iter_mut().zip(v.iter()) {
            *o += c * x;
        }
    }
    out
}

fn random_vector(rng: &mut SplitMix, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect()
}

/// Gram-Schmidt orthonormalisation of the given vectors.
fn orthonormalise(vectors: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::new();
    for v in vectors {
        let mut w = v.clone();
        for q in &out {
            let overlap = dot(&w, q);
            for (wi, qi) in w.iter_mut().zip(q.iter()) {
                *wi -= overlap * qi;
            }
        }
        let norm = dot(&w, &w).sqrt();
        out.push(w.iter().map(|x| x / norm).collect());
    }
    out
}

fn max_orthonormality_defect(basis: &[Vec<f64>]) -> f64 {
    let mut worst = 0.0_f64;
    for (i, bi) in basis.iter().enumerate() {
        for (j, bj) in basis.iter().enumerate() {
            let expected = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((dot(bi, bj) - expected).abs());
        }
    }
    worst
}

#[test]
fn pod_rom_recovers_a_two_dimensional_subspace_despite_collinear_leading_snapshots() {
    // Five snapshots spanning two vectors, the first two collinear.  The former
    // Gram-Schmidt of the first n snapshots produced a degenerate basis here.
    let mut rng = SplitMix(21);
    let a = random_vector(&mut rng, 7);
    let b = random_vector(&mut rng, 7);
    let snapshots = vec![
        combination(&[1.0], &[&a]),
        combination(&[2.0], &[&a]),
        combination(&[1.0], &[&b]),
        combination(&[1.0, 1.0], &[&a, &b]),
        combination(&[3.0, -2.0], &[&a, &b]),
    ];
    let rom = PodRom::from_snapshots(&snapshots, 2).unwrap();
    assert_eq!(rom.n_modes, 2);
    assert_eq!(rom.basis.len(), 2);
    assert!(max_orthonormality_defect(&rom.basis) < 1e-12);
    for (i, s) in snapshots.iter().enumerate() {
        let err = rom.projection_error(s);
        assert!(err < 1e-10, "snapshot {i}: projection error {err:e}");
    }
    assert!(
        rom.energy_fraction() > 1.0 - 1e-12,
        "two modes capture the whole rank-2 data: {}",
        rom.energy_fraction()
    );
    assert!(rom.singular_values[0] >= rom.singular_values[1]);
    assert!(rom.singular_values[1] > 0.0);
}

#[test]
fn pod_rom_matches_the_singular_value_decomposition_of_the_snapshot_matrix() {
    // S = U Σ Vᵀ with orthonormal U (6×3) and V (5×3): the POD must return the
    // singular values and (up to sign) the columns of U.
    let mut rng = SplitMix(2024);
    let u = orthonormalise(&[
        random_vector(&mut rng, 6),
        random_vector(&mut rng, 6),
        random_vector(&mut rng, 6),
    ]);
    let v = orthonormalise(&[
        random_vector(&mut rng, 5),
        random_vector(&mut rng, 5),
        random_vector(&mut rng, 5),
    ]);
    let sigma = [10.0, 3.0, 0.5];
    let snapshots: Vec<Vec<f64>> = (0..5)
        .map(|j| {
            let coeffs: Vec<f64> = (0..3).map(|k| sigma[k] * v[k][j]).collect();
            combination(&coeffs, &[&u[0], &u[1], &u[2]])
        })
        .collect();
    let rom = PodRom::from_snapshots(&snapshots, 3).unwrap();
    assert_eq!(rom.n_modes, 3);
    for k in 0..3 {
        assert!(
            (rom.singular_values[k] - sigma[k]).abs() <= 1e-9 * sigma[k],
            "σ_{k} = {} expected {}",
            rom.singular_values[k],
            sigma[k]
        );
        let alignment = dot(&rom.basis[k], &u[k]).abs();
        assert!(
            alignment > 1.0 - 1e-10,
            "mode {k} is not the k-th left singular vector: |φ·u| = {alignment}"
        );
    }
    let truncated = PodRom::from_snapshots(&snapshots, 2).unwrap();
    let expected = (100.0 + 9.0) / (100.0 + 9.0 + 0.25);
    assert!((truncated.energy_fraction() - expected).abs() < 1e-12);
    assert!((truncated.total_energy - 109.25).abs() < 1e-9);
}

#[test]
fn pod_rom_truncation_is_energy_optimal_unlike_leading_snapshot_gram_schmidt() {
    // Leading snapshots nearly collinear; Σ‖s − ΦΦᵀs‖² equals the discarded
    // energy Σ_{k>n} σ_k² for the POD basis (Eckart-Young) and is strictly larger
    // for the old Gram-Schmidt basis of the first n snapshots.
    let mut rng = SplitMix(7);
    let a = random_vector(&mut rng, 8);
    let b = random_vector(&mut rng, 8);
    let c = random_vector(&mut rng, 8);
    let snapshots = vec![
        a.clone(),
        combination(&[1.0, 1e-3], &[&a, &b]),
        combination(&[2.0, -1.5], &[&b, &c]),
        combination(&[1.0, 2.0], &[&c, &a]),
    ];
    let rom = PodRom::from_snapshots(&snapshots, 2).unwrap();
    let pod_error: f64 = snapshots
        .iter()
        .map(|s| (rom.projection_error(s) * dot(s, s).sqrt()).powi(2))
        .sum();
    let discarded = rom.total_energy * (1.0 - rom.energy_fraction());
    assert!(
        (pod_error - discarded).abs() <= 1e-9 * rom.total_energy,
        "POD error {pod_error} vs discarded energy {discarded}"
    );
    let gram_schmidt = orthonormalise(&snapshots[..2]);
    let gs_error: f64 = snapshots
        .iter()
        .map(|s| {
            let projected = combination(
                &[dot(s, &gram_schmidt[0]), dot(s, &gram_schmidt[1])],
                &[&gram_schmidt[0], &gram_schmidt[1]],
            );
            s.iter()
                .zip(projected.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
        })
        .sum();
    assert!(
        gs_error > 1.5 * pod_error,
        "first-snapshots basis error {gs_error} should clearly exceed POD error {pod_error}"
    );
}

#[test]
fn pod_rom_is_invariant_to_the_snapshot_scale() {
    // The Jacobi eigen-solver has absolute tolerances; tiny or huge snapshot
    // magnitudes must still give the same basis and errors.
    let mut rng = SplitMix(99);
    let a = random_vector(&mut rng, 9);
    let b = random_vector(&mut rng, 9);
    let c = random_vector(&mut rng, 9);
    let base = vec![
        combination(&[1.0, 0.2], &[&a, &b]),
        combination(&[0.3, 1.0], &[&a, &c]),
        combination(&[-0.5, 0.8], &[&b, &c]),
        combination(&[0.9, -0.4, 0.1], &[&a, &b, &c]),
    ];
    let reference = PodRom::from_snapshots(&base, 2).unwrap();
    for scale in [1e-9, 1e-4, 1e6] {
        let scaled: Vec<Vec<f64>> = base
            .iter()
            .map(|s| s.iter().map(|x| x * scale).collect())
            .collect();
        let rom = PodRom::from_snapshots(&scaled, 2).unwrap();
        assert_eq!(rom.n_modes, reference.n_modes, "scale {scale:e}");
        assert!(
            (rom.energy_fraction() - reference.energy_fraction()).abs() < 1e-10,
            "scale {scale:e}"
        );
        for k in 0..2 {
            let alignment = dot(&rom.basis[k], &reference.basis[k]).abs();
            assert!(
                alignment > 1.0 - 1e-9,
                "scale {scale:e}, mode {k}: {alignment}"
            );
            assert!(
                (rom.singular_values[k] / scale - reference.singular_values[k]).abs()
                    <= 1e-9 * reference.singular_values[k]
            );
        }
        for (s, original) in scaled.iter().zip(base.iter()) {
            assert!(
                (rom.projection_error(s) - reference.projection_error(original)).abs() < 1e-9,
                "scale {scale:e}: relative projection error changed"
            );
        }
    }
}

#[test]
fn pod_rom_drops_numerically_dependent_directions() {
    // Rank-two data with four modes requested: only the two real modes survive.
    let mut rng = SplitMix(5);
    let a = random_vector(&mut rng, 6);
    let b = random_vector(&mut rng, 6);
    let snapshots = vec![
        combination(&[1.0, 0.0], &[&a, &b]),
        combination(&[0.4, 1.0], &[&a, &b]),
        combination(&[-1.2, 0.7], &[&a, &b]),
        combination(&[2.0, 2.0], &[&a, &b]),
    ];
    let rom = PodRom::from_snapshots(&snapshots, 4).unwrap();
    assert_eq!(
        rom.n_modes, 2,
        "rank-2 data cannot yield {} modes",
        rom.n_modes
    );
    assert_eq!(rom.singular_values.len(), 2);
    assert_eq!(rom.q.len(), 2);
    assert_eq!(rom.k_rom.len(), 2);
}

#[test]
fn pod_rom_basis_stays_orthonormal_for_a_steep_singular_spectrum() {
    let mut rng = SplitMix(1234);
    let u = orthonormalise(&[
        random_vector(&mut rng, 10),
        random_vector(&mut rng, 10),
        random_vector(&mut rng, 10),
    ]);
    let v = orthonormalise(&[
        random_vector(&mut rng, 6),
        random_vector(&mut rng, 6),
        random_vector(&mut rng, 6),
    ]);
    let sigma = [1.0, 1e-3, 1e-5];
    let snapshots: Vec<Vec<f64>> = (0..6)
        .map(|j| {
            let coeffs: Vec<f64> = (0..3).map(|k| sigma[k] * v[k][j]).collect();
            combination(&coeffs, &[&u[0], &u[1], &u[2]])
        })
        .collect();
    let rom = PodRom::from_snapshots(&snapshots, 3).unwrap();
    assert_eq!(rom.n_modes, 3);
    assert!(
        max_orthonormality_defect(&rom.basis) < 1e-12,
        "defect {:e}",
        max_orthonormality_defect(&rom.basis)
    );
    for s in &snapshots {
        assert!(rom.projection_error(s) < 1e-6);
    }
}

#[test]
fn pod_rom_rejects_invalid_snapshot_sets() {
    assert!(matches!(
        PodRom::from_snapshots(&[], 2),
        Err(Error::InvalidInput(_))
    ));
    assert!(PodRom::from_snapshots(&[vec![1.0, 2.0]], 0).is_err());
    assert!(PodRom::from_snapshots(&[vec![1.0, 2.0], vec![1.0]], 1).is_err());
    assert!(PodRom::from_snapshots(&[vec![1.0, f64::NAN]], 1).is_err());
    assert!(PodRom::from_snapshots(&[vec![]], 1).is_err());
    assert!(matches!(
        PodRom::from_snapshots(&[vec![0.0; 4], vec![0.0; 4]], 1),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn pod_rom_projects_the_full_order_system() {
    // 1-D Laplacian K = tridiag(−1, 2, −1), M = I; the snapshots live in the span
    // of its first two eigenvectors φ_k(j) = sin(kπ j/(n+1)).
    let n = 8;
    let phi = |k: usize| -> Vec<f64> {
        (1..=n)
            .map(|j| (k as f64 * std::f64::consts::PI * j as f64 / (n as f64 + 1.0)).sin())
            .collect()
    };
    let lambda = |k: usize| 2.0 - 2.0 * (k as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos();
    let mut k_full = vec![vec![0.0; n]; n];
    let mut m_full = vec![vec![0.0; n]; n];
    for i in 0..n {
        k_full[i][i] = 2.0;
        m_full[i][i] = 1.0;
        if i + 1 < n {
            k_full[i][i + 1] = -1.0;
            k_full[i + 1][i] = -1.0;
        }
    }
    let (p1, p2) = (phi(1), phi(2));
    let snapshots = vec![
        combination(&[5.0], &[&p1]),
        combination(&[2.0], &[&p2]),
        combination(&[1.0, 1.0], &[&p1, &p2]),
    ];
    let mut rom = PodRom::from_snapshots(&snapshots, 2).unwrap();
    assert!(
        rom.k_rom.iter().flatten().all(|v| *v == 0.0),
        "reduced operators are empty until the system is projected"
    );
    let f_full = random_vector(&mut SplitMix(8), n);
    rom.project_system(&k_full, &m_full, &f_full).unwrap();
    // Independent evaluation of Φᵀ K Φ.
    for i in 0..2 {
        for j in 0..2 {
            let k_phi_j: Vec<f64> = k_full.iter().map(|row| dot(row, &rom.basis[j])).collect();
            let expected = dot(&rom.basis[i], &k_phi_j);
            assert!((rom.k_rom[i][j] - expected).abs() < 1e-12);
            assert!((rom.k_rom[i][j] - rom.k_rom[j][i]).abs() < 1e-12);
            let identity = if i == j { 1.0 } else { 0.0 };
            assert!(
                (rom.m_rom[i][j] - identity).abs() < 1e-12,
                "Φᵀ M Φ must be I"
            );
        }
        assert!((rom.f_rom[i] - dot(&rom.basis[i], &f_full)).abs() < 1e-12);
    }
    // The POD space is invariant under K, so the reduced matrix has the
    // eigenvalues λ₁, λ₂ of the full operator.
    let trace = rom.k_rom[0][0] + rom.k_rom[1][1];
    let det = rom.k_rom[0][0] * rom.k_rom[1][1] - rom.k_rom[0][1] * rom.k_rom[1][0];
    assert!(
        (trace - (lambda(1) + lambda(2))).abs() < 1e-10,
        "trace {trace}"
    );
    assert!((det - lambda(1) * lambda(2)).abs() < 1e-10, "det {det}");
    // Dimension errors leave the stored operators untouched.
    let stored = rom.k_rom.clone();
    assert!(matches!(
        rom.project_system(&k_full[..n - 1], &m_full, &f_full),
        Err(Error::InvalidInput(_))
    ));
    assert!(
        rom.project_system(&k_full, &m_full, &f_full[..n - 1])
            .is_err()
    );
    assert_eq!(rom.k_rom, stored);
}
