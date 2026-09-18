//! Regression tests for the stochastic FEM audit fixes.
//!
//! * The PCE variance omitted the `n!` normalisation of the unnormalised
//!   probabilists' Hermite basis (`E[He_n²] = n!`), which under-estimated the
//!   variance (and mis-scaled the Sobol fractions) by `2×` for order 2, `6×` for
//!   order 3, ...
//! * The Karhunen–Loève modes came from a power iteration started at the first
//!   coordinate vector, which missed every mode orthogonal to it (a diagonal
//!   covariance reported `λ = 1` and never found `λ = 5`).
//! * The Morris screening kept the base output of a trajectory while the point
//!   moved on, so the `k`-th effect was a cumulative difference over `k`
//!   coordinates; it also drew from an unseeded RNG.

use super::functions::*;
use super::types::*;
use crate::eigen::jacobi_eigen_dense;
use std::f64::consts::{PI, SQRT_2};

// ---------------------------------------------------------------------------
// PCE variance and Sobol fractions
// ---------------------------------------------------------------------------

/// `E[f²] − E[f]²` of a Hermite expansion by Gauss–Hermite quadrature, which is
/// independent of the closed-form `Σ cₙ² n!`.
fn quadrature_variance(coeffs: &[f64]) -> f64 {
    let (nodes, weights) = gauss_hermite_quadrature(24);
    let max_order = coeffs.len() - 1;
    let second_moment: f64 = nodes
        .iter()
        .zip(weights.iter())
        .map(|(&x, &w)| {
            let value = polynomial_chaos_expansion(coeffs, SQRT_2 * x, max_order);
            w * value * value
        })
        .sum::<f64>()
        / PI.sqrt();
    second_moment - coeffs[0] * coeffs[0]
}

#[test]
fn pce_variance_weights_each_order_by_its_hermite_norm() {
    // 2² · 1! + 3² · 2! = 22 (the unweighted sum of squares gave 13).
    assert!((pce_variance(&[0.0, 2.0, 3.0]) - 22.0).abs() < 1e-14);
    // Order 3 carries 3! = 6: 1² · 6 = 6, not 1.
    assert!((pce_variance(&[0.0, 0.0, 0.0, 1.0]) - 6.0).abs() < 1e-14);
    // Order 4 carries 4! = 24.
    assert!((pce_variance(&[9.0, 0.0, 0.0, 0.0, 0.5]) - 6.0).abs() < 1e-14);
    // The mean never enters the variance.
    assert_eq!(pce_variance(&[3.0]), 0.0);
    assert_eq!(pce_variance(&[]), 0.0);
}

#[test]
fn pce_variance_matches_the_gauss_hermite_second_moment() {
    let coeffs = [0.7, -1.3, 0.4, 0.9, -0.25, 0.1];
    // 1.3² · 1 + 0.4² · 2 + 0.9² · 6 + 0.25² · 24 + 0.1² · 120 = 9.57.
    let closed_form = pce_variance(&coeffs);
    assert!((closed_form - 9.57).abs() < 1e-12, "{closed_form}");
    let reference = quadrature_variance(&coeffs);
    assert!(
        (closed_form - reference).abs() < 1e-11 * reference,
        "closed form {closed_form} vs quadrature {reference}"
    );
    let pce = PolynomialChaosExpansion::new(coeffs.to_vec());
    assert!((pce.variance() - reference).abs() < 1e-11 * reference);
    assert!((pce.std_dev() - reference.sqrt()).abs() < 1e-11);
    assert!((pce.cov() - reference.sqrt() / 0.7).abs() < 1e-11);
}

#[test]
fn pce_variance_contributions_are_squared_coefficients_times_factorials() {
    let contributions = pce_variance_contributions(&[5.0, 2.0, 0.0, -1.0]);
    assert_eq!(contributions.len(), 3);
    assert!((contributions[0] - 4.0).abs() < 1e-14); // 2² · 1!
    assert_eq!(contributions[1], 0.0); // zero coefficient
    assert!((contributions[2] - 6.0).abs() < 1e-14); // (−1)² · 3!
    assert!(pce_variance_contributions(&[1.0]).is_empty());
    assert!(pce_variance_contributions(&[]).is_empty());
}

#[test]
fn sobol_fractions_use_the_factorial_weighted_contributions() {
    let pce = PolynomialChaosExpansion::new(vec![0.0, 3.0, 4.0]);
    let sobol = pce.sobol_indices();
    assert_eq!(sobol.len(), 2);
    // Var = 9 + 32 = 41.
    assert!((sobol[0] - 9.0 / 41.0).abs() < 1e-14, "{}", sobol[0]);
    assert!((sobol[1] - 32.0 / 41.0).abs() < 1e-14, "{}", sobol[1]);
    assert!((pce.total_sobol_index() - 1.0).abs() < 1e-14);
    // A pure order-3 response puts everything in S₃.
    let cubic = PolynomialChaosExpansion::new(vec![1.0, 0.0, 0.0, 2.0]).sobol_indices();
    assert!(cubic[0].abs() < 1e-15 && cubic[1].abs() < 1e-15);
    assert!((cubic[2] - 1.0).abs() < 1e-14);
    // No variance: all zero fractions.
    assert_eq!(
        PolynomialChaosExpansion::new(vec![1.0, 0.0, 0.0]).sobol_indices(),
        vec![0.0, 0.0]
    );
}

#[test]
fn projected_lognormal_variance_matches_the_closed_form() {
    // Var[exp(a ξ)] = e^{a²} (e^{a²} − 1).
    let a = 0.3_f64;
    let exact = (a * a).exp() * ((a * a).exp() - 1.0);
    let coarse = PolynomialChaosExpansion::from_quadrature(|x| (a * x).exp(), 6, 5);
    assert!(
        (coarse.variance() - exact).abs() < 1e-3,
        "5-point projection: {} vs {exact}",
        coarse.variance()
    );
    let fine = PolynomialChaosExpansion::from_quadrature(|x| (a * x).exp(), 8, 14);
    assert!(
        (fine.variance() - exact).abs() < 1e-9,
        "14-point projection: {} vs {exact}",
        fine.variance()
    );
    // The mean is E[exp(aξ)] = e^{a²/2}.
    assert!((fine.mean() - (0.5 * a * a).exp()).abs() < 1e-12);
    // Without the n! weights the variance would be off by about 4 % here.
    let unweighted: f64 = fine.coefficients[1..].iter().map(|c| c * c).sum();
    assert!((unweighted - exact).abs() > 1e-3);
}

// ---------------------------------------------------------------------------
// Karhunen–Loève modes
// ---------------------------------------------------------------------------

fn matvec(flat: &[f64], n: usize, v: &[f64]) -> Vec<f64> {
    (0..n)
        .map(|i| (0..n).map(|j| flat[i * n + j] * v[j]).sum())
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Largest `|C φ − λ φ|` over the retained modes.
fn max_eigen_residual(kle: &KarhunenLoeveExpansion, flat: &[f64], n: usize) -> f64 {
    kle.eigenvalues
        .iter()
        .zip(kle.eigenvectors.iter())
        .map(|(&lambda, phi)| {
            matvec(flat, n, phi)
                .iter()
                .zip(phi.iter())
                .map(|(c, p)| (c - lambda * p).abs())
                .fold(0.0_f64, f64::max)
        })
        .fold(0.0_f64, f64::max)
}

fn assert_orthonormal(kle: &KarhunenLoeveExpansion, tol: f64) {
    for (i, phi_i) in kle.eigenvectors.iter().enumerate() {
        for (j, phi_j) in kle.eigenvectors.iter().enumerate() {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot(phi_i, phi_j) - expected).abs() < tol,
                "<phi_{i}, phi_{j}> = {}",
                dot(phi_i, phi_j)
            );
        }
    }
}

fn exponential_covariance(n: usize, sigma: f64, corr_length: f64) -> Vec<f64> {
    let problem = StochasticFemProblem::new_uniform(n, 1.0, 0.0, sigma, corr_length);
    problem
        .covariance_matrix_exponential()
        .into_iter()
        .flatten()
        .collect()
}

#[test]
fn kl_modes_of_a_diagonal_covariance_are_all_found_in_descending_order() {
    // The power iteration from e_0 returned lambda = 1 first and never found 5.
    let cov = [1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 3.0];
    let kle = KarhunenLoeveExpansion::from_covariance(&cov, 3, 3, 100);
    assert_eq!(kle.eigenvalues.len(), 3);
    for (got, want) in kle.eigenvalues.iter().zip([5.0, 3.0, 1.0]) {
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
    // Eigenvectors are the coordinate vectors e_1, e_2, e_0 with positive sign.
    for (mode, axis) in [1usize, 2, 0].into_iter().enumerate() {
        for (component, value) in kle.eigenvectors[mode].iter().enumerate() {
            let expected = if component == axis { 1.0 } else { 0.0 };
            assert!((value - expected).abs() < 1e-12, "mode {mode}: {value}");
        }
    }
    assert!((kle.total_variance() - 9.0).abs() < 1e-12);
}

#[test]
fn kl_modes_of_a_block_covariance_cover_every_block() {
    // [[2, 1], [1, 2]] ⊕ [4]: eigenvalues 4, 3, 1.
    let cov = [2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 4.0];
    let kle = KarhunenLoeveExpansion::from_covariance(&cov, 3, 3, 10);
    for (got, want) in kle.eigenvalues.iter().zip([4.0, 3.0, 1.0]) {
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
    let r = std::f64::consts::FRAC_1_SQRT_2;
    let expected = [[0.0, 0.0, 1.0], [r, r, 0.0], [r, -r, 0.0]];
    for (phi, want) in kle.eigenvectors.iter().zip(expected.iter()) {
        // Eigenvectors are normalised with the largest component positive; the
        // two equal components of mode 2 are ambiguous, so compare up to sign.
        let aligned = dot(phi, want).abs();
        assert!((aligned - 1.0).abs() < 1e-12, "{phi:?} vs {want:?}");
    }
    assert_orthonormal(&kle, 1e-12);
}

#[test]
fn kl_exponential_kernel_matches_the_jacobi_reference() {
    let n = 30;
    let cov = exponential_covariance(n, 1.5, 0.4);
    let rows: Vec<Vec<f64>> = cov.chunks(n).map(<[f64]>::to_vec).collect();
    let (values, _) = jacobi_eigen_dense(&rows, n);
    let kle = KarhunenLoeveExpansion::from_covariance(&cov, n, 8, 100);
    assert_eq!(kle.eigenvalues.len(), 8);
    for (mode, &lambda) in kle.eigenvalues.iter().enumerate() {
        let reference = values[n - 1 - mode];
        assert!(
            (lambda - reference).abs() < 1e-10,
            "mode {mode}: {lambda} vs Jacobi {reference}"
        );
    }
    for pair in kle.eigenvalues.windows(2) {
        assert!(pair[0] >= pair[1], "eigenvalues must descend");
    }
    // They are genuine eigenpairs of the covariance.
    assert!(max_eigen_residual(&kle, &cov, n) < 1e-10);
    assert_orthonormal(&kle, 1e-10);
}

#[test]
fn kl_with_all_modes_retains_the_total_variance() {
    let n = 24;
    let sigma = 2.0;
    let cov = exponential_covariance(n, sigma, 0.3);
    let kle = KarhunenLoeveExpansion::from_covariance(&cov, n, n, 100);
    assert_eq!(kle.eigenvalues.len(), n);
    // trace(C) = n σ² = sum of eigenvalues.
    let trace = n as f64 * sigma * sigma;
    assert!((kle.total_variance() - trace).abs() < 1e-10 * trace);
    assert!((kle.cumulative_energy(n) - 1.0).abs() < 1e-12);
    // The truncated field reproduces the covariance when every mode is kept:
    // Σ λ φ φᵀ = C.
    for i in 0..n {
        for j in 0..n {
            let rebuilt: f64 = kle
                .eigenvalues
                .iter()
                .zip(kle.eigenvectors.iter())
                .map(|(&l, phi)| l * phi[i] * phi[j])
                .sum();
            assert!(
                (rebuilt - cov[i * n + j]).abs() < 1e-10,
                "({i},{j}): {rebuilt} vs {}",
                cov[i * n + j]
            );
        }
    }
}

#[test]
fn kl_is_independent_of_the_units_of_the_field() {
    // The Jacobi tolerances are absolute: a covariance of 1e-16 would otherwise
    // be treated as already diagonal.
    let n = 12;
    let base = exponential_covariance(n, 1.0, 0.5);
    let tiny: Vec<f64> = base.iter().map(|c| c * 1e-16).collect();
    let reference = KarhunenLoeveExpansion::from_covariance(&base, n, 5, 100);
    let scaled = KarhunenLoeveExpansion::from_covariance(&tiny, n, 5, 100);
    assert_eq!(scaled.eigenvalues.len(), 5);
    for (a, b) in reference.eigenvalues.iter().zip(scaled.eigenvalues.iter()) {
        assert!((a * 1e-16 - b).abs() < 1e-12 * a * 1e-16, "{a} vs {b}");
    }
    assert!(max_eigen_residual(&scaled, &tiny, n) < 1e-12 * 1e-16 * n as f64);
}

#[test]
fn kl_skips_the_numerical_null_space_and_rejects_bad_input() {
    // Rank one: only one positive eigenvalue (4 for the all-ones 4×4 matrix).
    let ones = [1.0; 16];
    let kle = KarhunenLoeveExpansion::from_covariance(&ones, 4, 3, 50);
    assert_eq!(kle.eigenvalues.len(), 1);
    assert!((kle.eigenvalues[0] - 4.0).abs() < 1e-12);
    for component in &kle.eigenvectors[0] {
        assert!((component - 0.5).abs() < 1e-12);
    }
    // The zero matrix has no mode.
    assert!(
        KarhunenLoeveExpansion::from_covariance(&[0.0; 9], 3, 3, 50)
            .eigenvalues
            .is_empty()
    );
    // Wrong length, zero sizes and non-finite entries return an empty expansion.
    assert!(
        KarhunenLoeveExpansion::from_covariance(&[1.0, 0.0, 0.0], 2, 1, 50)
            .eigenvalues
            .is_empty()
    );
    assert!(
        KarhunenLoeveExpansion::from_covariance(&[], 0, 1, 50)
            .eigenvalues
            .is_empty()
    );
    assert!(
        KarhunenLoeveExpansion::from_covariance(&[1.0], 1, 0, 50)
            .eigenvalues
            .is_empty()
    );
    let nan = [1.0, f64::NAN, f64::NAN, 1.0];
    assert!(
        KarhunenLoeveExpansion::from_covariance(&nan, 2, 2, 50)
            .eigenvalues
            .is_empty()
    );
    // Asking for more modes than the dimension keeps all n.
    let identity = [1.0, 0.0, 0.0, 1.0];
    assert_eq!(
        KarhunenLoeveExpansion::from_covariance(&identity, 2, 7, 50)
            .eigenvalues
            .len(),
        2
    );
}

#[test]
fn kl_symmetrises_a_slightly_asymmetric_covariance() {
    let cov = [2.0, 1.0 + 1e-9, 1.0 - 1e-9, 2.0];
    let kle = KarhunenLoeveExpansion::from_covariance(&cov, 2, 2, 50);
    assert!((kle.eigenvalues[0] - 3.0).abs() < 1e-12);
    assert!((kle.eigenvalues[1] - 1.0).abs() < 1e-12);
}

#[test]
fn lobpcg_dominant_pairs_match_the_dense_solver() {
    let n = 40;
    let cov = exponential_covariance(n, 1.0, 0.35);
    let rows: Vec<Vec<f64>> = cov.chunks(n).map(<[f64]>::to_vec).collect();
    let dense = dominant_eigenpairs_dense(&rows, n, 5);
    let iterative = dominant_eigenpairs_lobpcg(&rows, n, 5, 500)
        .expect("LOBPCG must converge on a small exponential kernel");
    assert_eq!(iterative.len(), 5);
    for ((l_it, v_it), (l_de, v_de)) in iterative.iter().zip(dense.iter()) {
        assert!(
            (l_it - l_de).abs() < 1e-8 * dense[0].0,
            "{l_it} vs dense {l_de}"
        );
        assert!(
            (dot(v_it, v_de).abs() - 1.0).abs() < 1e-6,
            "eigenvector mismatch for lambda {l_de}"
        );
    }
}

/// Block-diagonal covariance of `blocks` 2×2 blocks `[[3p/4, p/4], [p/4, 3p/4]]`
/// with `p = 10 · 0.85^k`: eigenvalues `p` (vector `(1, 1)/√2`) and `p/2`
/// (vector `(1, −1)/√2`) per block, so the spectrum is known exactly.
fn block_covariance(blocks: usize) -> (Vec<f64>, Vec<f64>) {
    let n = 2 * blocks;
    let mut cov = vec![0.0; n * n];
    let mut spectrum = Vec::new();
    for k in 0..blocks {
        let p = 10.0 * 0.85_f64.powi(k as i32);
        let (i, j) = (2 * k, 2 * k + 1);
        cov[i * n + i] = 0.75 * p;
        cov[j * n + j] = 0.75 * p;
        cov[i * n + j] = 0.25 * p;
        cov[j * n + i] = 0.25 * p;
        spectrum.push(p);
        spectrum.push(0.5 * p);
    }
    spectrum.sort_by(|a, b| b.total_cmp(a));
    (cov, spectrum)
}

#[test]
fn large_covariances_use_lobpcg_and_return_exact_dominant_modes() {
    // n = 520 > 500 selects the LOBPCG branch.
    let blocks = 260;
    let n = 2 * blocks;
    let (cov, spectrum) = block_covariance(blocks);
    let rows: Vec<Vec<f64>> = cov.chunks(n).map(<[f64]>::to_vec).collect();
    let direct = dominant_eigenpairs_lobpcg(&rows, n, 4, 500)
        .expect("LOBPCG must converge for a well separated spectrum");
    assert_eq!(direct.len(), 4);
    let kle = KarhunenLoeveExpansion::from_covariance(&cov, n, 4, 500);
    assert_eq!(kle.eigenvalues.len(), 4);
    for mode in 0..4 {
        assert!(
            (direct[mode].0 - spectrum[mode]).abs() < 1e-8,
            "direct mode {mode}: {} vs {}",
            direct[mode].0,
            spectrum[mode]
        );
        assert!(
            (kle.eigenvalues[mode] - spectrum[mode]).abs() < 1e-8,
            "mode {mode}: {} vs {}",
            kle.eigenvalues[mode],
            spectrum[mode]
        );
    }
    // LOBPCG stops at a relative residual of 1e-8 of the scaled matrix (entries up to 7.5).
    assert!(max_eigen_residual(&kle, &cov, n) < 1e-6);
    assert_orthonormal(&kle, 1e-8);
}

// ---------------------------------------------------------------------------
// Morris screening
// ---------------------------------------------------------------------------

#[test]
fn morris_effects_of_a_linear_model_are_its_coefficients() {
    // With a stale base output the k-th effect of a trajectory accumulated the
    // k preceding steps, so mu* exceeded |a| and sigma was non-zero.
    let a = [1.0, -2.0, 3.0];
    let model = |x: &[f64]| a.iter().zip(x.iter()).map(|(c, v)| c * v).sum::<f64>();
    let analysis = SensitivityAnalysis::new(3, 25);
    let (mu_star, sigma) = analysis.morris_screening_seeded(&model, 0.5, 7);
    for i in 0..3 {
        assert!(
            (mu_star[i] - a[i].abs()).abs() < 1e-12,
            "mu*[{i}] = {} expected {}",
            mu_star[i],
            a[i].abs()
        );
        assert!(sigma[i] < 1e-12, "sigma[{i}] = {}", sigma[i]);
    }
    // The entropy-seeded entry point agrees: a linear model has no randomness.
    let (mu_free, sigma_free) = analysis.morris_screening(&model, 0.5);
    for i in 0..3 {
        assert!((mu_free[i] - a[i].abs()).abs() < 1e-12);
        assert!(sigma_free[i] < 1e-12);
    }
}

#[test]
fn morris_effects_of_a_quadratic_model_follow_the_visited_points() {
    // For f = Σ xᵢ² the effect of input i between two neighbouring points is
    // ((xᵢ + s)² − xᵢ²) / s = 2 xᵢ + s with xᵢ the start value (each input is
    // perturbed once) and s = +delta for xᵢ ≤ 1 − delta, else −delta.  Hence every
    // effect lies in [delta, 2 − delta] and, for a uniform start,
    // E[EE] = 0.75 · 1 + 0.25 · 1.5 = 1.125 for delta = 0.25.  The stale base
    // output accumulated the preceding steps and gave about 2.8.
    let model = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
    let delta = 0.25;
    let (mu_star, sigma) =
        SensitivityAnalysis::new(4, 400).morris_screening_seeded(&model, delta, 11);
    for i in 0..4 {
        assert!(
            (delta..=2.0 - delta).contains(&mu_star[i]),
            "mu*[{i}] = {}",
            mu_star[i]
        );
        assert!(
            (mu_star[i] - 1.125).abs() < 0.12,
            "mu*[{i}] = {} expected about 1.125",
            mu_star[i]
        );
        assert!(
            sigma[i] > 0.2 && sigma[i] < 0.8,
            "a non-linear input has spread: {}",
            sigma[i]
        );
    }
}

#[test]
fn morris_is_reproducible_for_a_given_seed() {
    let model = |x: &[f64]| x[0] * x[1] + x[2] * x[2] - x[0];
    let analysis = SensitivityAnalysis::new(3, 15);
    let first = analysis.morris_screening_seeded(&model, 0.4, 2024);
    let second = analysis.morris_screening_seeded(&model, 0.4, 2024);
    assert_eq!(first, second, "same seed must give bit-identical output");
    let other = analysis.morris_screening_seeded(&model, 0.4, 2025);
    assert_ne!(
        first, other,
        "a different seed must change the trajectories"
    );
    // The generic entry point draws from the supplied generator.
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(2024);
    let via_rng = analysis.morris_screening_with_rng(&model, 0.4, &mut rng);
    assert_eq!(first, via_rng);
}

#[test]
fn morris_rejects_steps_that_cannot_stay_inside_the_unit_cube() {
    let model = |x: &[f64]| x[0] + x[1];
    let analysis = SensitivityAnalysis::new(2, 5);
    for delta in [0.0, -0.1, 0.51, 1.0, f64::NAN, f64::INFINITY] {
        let (mu_star, sigma) = analysis.morris_screening_seeded(&model, delta, 1);
        assert_eq!(mu_star.len(), 2);
        assert!(
            mu_star.iter().chain(sigma.iter()).all(|v| v.is_nan()),
            "delta = {delta} must yield NaN"
        );
    }
}

#[test]
fn morris_evaluates_every_perturbed_point_inside_the_unit_cube() {
    let seen_outside = std::cell::Cell::new(false);
    let model = |x: &[f64]| {
        if x.iter().any(|v| !(0.0..=1.0).contains(v)) {
            seen_outside.set(true);
        }
        x.iter().sum::<f64>()
    };
    let _ = SensitivityAnalysis::new(5, 100).morris_screening_seeded(&model, 0.5, 3);
    assert!(!seen_outside.get(), "a trajectory left the unit hypercube");
}
