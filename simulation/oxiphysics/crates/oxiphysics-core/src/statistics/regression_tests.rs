//! Regression tests for the statistics fixes: PCA, the Shapiro-Wilk statistic,
//! log-gamma based densities, the Student-t p-values of the t-tests and the
//! Anderson-Darling statistic.
//!
//! Reference values were computed independently: p-values and A² with 50-digit
//! arithmetic, log-gamma based densities with `mpmath`, and the Shapiro-Wilk
//! values with a line-by-line port of the AS R94 routine of R's `swilk.c`.

use super::functions::{
    anderson_darling_statistic, covariance_matrix, kurtosis, log_factorial, mean, pacf,
    pacf_from_acf, pca, pca_transform, sample_kurtosis, sample_skewness, shapiro_wilk_w, skewness,
    student_t_two_sided_p, t_test_one_sample, t_test_one_sample_full, t_test_two_sample, variance,
    welch_t_test,
};
use super::poisson::poisson_ptrs;
use super::types::{BetaDistribution, PoissonDistribution, StatRng};
use crate::random::student_t_cdf;

/// Asserts `|actual - expected| <= rel * |expected| + abs`.
fn assert_close(actual: f64, expected: f64, rel: f64, abs: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= rel * expected.abs() + abs,
        "{what}: got {actual}, expected {expected}"
    );
}

// ---------------------------------------------------------------------------
// pca
// ---------------------------------------------------------------------------

/// Observations `±sigma_k e_k` whose sample covariance is exactly
/// `diag(variances)`.
fn diagonal_covariance_data(variances: &[f64]) -> Vec<Vec<f64>> {
    let d = variances.len();
    let scale = (2 * d - 1) as f64 / 2.0;
    let mut data = Vec::with_capacity(2 * d);
    for (k, variance) in variances.iter().enumerate() {
        for sign in [1.0, -1.0] {
            let mut row = vec![0.0; d];
            row[k] = sign * (variance * scale).sqrt();
            data.push(row);
        }
    }
    data
}

#[test]
fn pca_finds_the_dominant_axis_even_when_it_is_not_the_first() {
    // Power iteration started at e0 returned e0 (variance 1) for this data.
    let data = diagonal_covariance_data(&[1.0, 5.0]);
    let result = pca(&data, 2).expect("well-posed data");
    assert_close(
        result.explained_variance[0],
        5.0,
        1e-12,
        0.0,
        "variance of PC1",
    );
    assert_close(
        result.explained_variance[1],
        1.0,
        1e-12,
        0.0,
        "variance of PC2",
    );
    assert_close(result.components[0][0], 0.0, 0.0, 1e-12, "PC1 x");
    assert_close(result.components[0][1], 1.0, 1e-12, 0.0, "PC1 y");
    assert_close(result.components[1][0], 1.0, 1e-12, 0.0, "PC2 x");
    assert_close(result.components[1][1], 0.0, 0.0, 1e-12, "PC2 y");
}

#[test]
fn pca_returns_distinct_orthonormal_components_after_deflation() {
    // Deflation left A e0 = 0 for diag(3, 2, 1) and the old loop returned e0 twice.
    let data = diagonal_covariance_data(&[3.0, 2.0, 1.0]);
    let result = pca(&data, 3).expect("well-posed data");
    for (k, expected) in [3.0, 2.0, 1.0].iter().enumerate() {
        assert_close(
            result.explained_variance[k],
            *expected,
            1e-12,
            0.0,
            "variance",
        );
        for (j, &entry) in result.components[k].iter().enumerate() {
            let target = if j == k { 1.0 } else { 0.0 };
            assert_close(entry, target, 1e-12, 1e-12, "component entry");
        }
    }
}

#[test]
fn pca_aligns_with_a_tilted_principal_axis() {
    let c = 5.0_f64.sqrt().recip();
    let u = [c, 2.0 * c];
    let w = [-2.0 * c, c];
    let along = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let across = [0.1, -0.2, 0.1, 0.0, -0.1, 0.2, -0.1];
    let data: Vec<Vec<f64>> = along
        .iter()
        .zip(&across)
        .map(|(a, b)| vec![a * u[0] + b * w[0], a * u[1] + b * w[1]])
        .collect();
    let result = pca(&data, 2).expect("well-posed data");
    assert_close(
        result.explained_variance[0],
        28.0 / 6.0,
        1e-12,
        0.0,
        "major variance",
    );
    assert_close(
        result.explained_variance[1],
        0.02,
        1e-9,
        0.0,
        "minor variance",
    );
    assert_close(result.components[0][0], u[0], 1e-10, 0.0, "PC1 x");
    assert_close(result.components[0][1], u[1], 1e-10, 0.0, "PC1 y");
    assert_close(result.components[1][0], -w[0], 1e-10, 0.0, "PC2 x");
    assert_close(result.components[1][1], -w[1], 1e-10, 0.0, "PC2 y");
}

#[test]
fn pca_solves_the_eigenproblem_of_correlated_data() {
    let mut rng = StatRng::new(77);
    let mut data = Vec::new();
    for _ in 0..200 {
        let factors = [
            3.0 * rng.next_normal(),
            1.5 * rng.next_normal(),
            0.7 * rng.next_normal(),
        ];
        let mut row = Vec::new();
        for j in 0..6 {
            let jf = j as f64;
            let parity = if j % 2 == 0 { 1.0 } else { -1.0 };
            row.push(
                factors[0] * (jf + 1.0) * 0.3
                    + factors[1] * (jf - 2.5) * 0.4
                    + factors[2] * parity
                    + 0.05 * rng.next_normal(),
            );
        }
        data.push(row);
    }
    let cov = covariance_matrix(&data);
    let result = pca(&data, 6).expect("well-posed data");
    assert_eq!(result.components.len(), 6);
    let trace: f64 = (0..6).map(|i| cov[i][i]).sum();
    let total: f64 = result.explained_variance.iter().sum();
    assert_close(total, trace, 1e-10, 0.0, "total variance");
    let lambda_max = result.explained_variance[0];
    assert!(result.explained_variance.windows(2).all(|w| w[0] >= w[1]));
    for (k, v) in result.components.iter().enumerate() {
        let lambda = result.explained_variance[k];
        for (i, row) in cov.iter().enumerate() {
            let cv: f64 = row.iter().zip(v).map(|(c, x)| c * x).sum();
            assert!(
                (cv - lambda * v[i]).abs() <= 1e-9 * lambda_max,
                "component {k} is not an eigenvector (row {i})"
            );
        }
        for (l, other) in result.components.iter().enumerate() {
            let dot: f64 = v.iter().zip(other).map(|(a, b)| a * b).sum();
            let target = if k == l { 1.0 } else { 0.0 };
            assert!((dot - target).abs() < 1e-10, "components {k}, {l}: {dot}");
        }
    }
    // Three latent factors: the remaining variances are the noise floor.
    assert!(result.explained_variance[3] < 0.02);
    // Scores are uncorrelated with the component variances on the diagonal.
    let scores: Vec<Vec<f64>> = data.iter().map(|x| pca_transform(x, &result)).collect();
    let score_cov = covariance_matrix(&scores);
    for (i, row) in score_cov.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            if i == j {
                assert_close(
                    value,
                    result.explained_variance[i],
                    1e-9,
                    1e-12,
                    "score variance",
                );
            } else {
                assert!(
                    value.abs() <= 1e-9 * lambda_max,
                    "scores {i}, {j} correlated"
                );
            }
        }
    }
    // Every component is oriented with its largest entry positive.
    for v in &result.components {
        let largest = v
            .iter()
            .fold(0.0_f64, |m, &x| if x.abs() > m.abs() { x } else { m });
        assert!(largest > 0.0);
    }
}

#[test]
fn pca_handles_rank_deficient_data_and_component_counts() {
    // The third column is constant: its direction carries zero variance.
    let data: Vec<Vec<f64>> = (0..12)
        .map(|i| vec![f64::from(i), f64::from(i % 3) - 1.0, 7.0])
        .collect();
    let all = pca(&data, 10).expect("well-posed data");
    assert_eq!(all.components.len(), 3, "at most `d` components");
    assert!(all.explained_variance.iter().all(|&v| v >= 0.0));
    assert!(all.explained_variance[2] < 1e-12);
    let top = pca(&data, 1).expect("well-posed data");
    assert_eq!(top.components.len(), 1);
    assert_close(
        top.explained_variance[0],
        all.explained_variance[0],
        1e-12,
        0.0,
        "top variance",
    );
    assert!(
        pca(&data, 0)
            .expect("zero components are allowed")
            .components
            .is_empty()
    );
}

#[test]
fn pca_rejects_malformed_input() {
    assert!(pca(&[], 1).is_none());
    assert!(pca(&[vec![1.0, 2.0]], 1).is_none());
    assert!(pca(&[vec![], vec![]], 1).is_none());
    assert!(pca(&[vec![1.0, 2.0], vec![3.0]], 1).is_none());
    assert!(pca(&[vec![1.0, f64::NAN], vec![3.0, 4.0]], 1).is_none());
    assert!(pca(&[vec![1.0, f64::INFINITY], vec![3.0, 4.0]], 1).is_none());
}

// ---------------------------------------------------------------------------
// shapiro_wilk_w
// ---------------------------------------------------------------------------

/// Blom quantiles `Phi^-1((i - 3/8) / (n + 1/4))`: a textbook "perfectly normal" sample.
fn blom_sample(n: usize) -> Vec<f64> {
    (1..=n)
        .map(|i| {
            let p = (i as f64 - 0.375) / (n as f64 + 0.25);
            crate::random::normal_ppf(p)
        })
        .collect()
}

#[test]
fn shapiro_wilk_matches_the_reference_routine() {
    let weights = [
        148.0, 154.0, 158.0, 160.0, 161.0, 162.0, 166.0, 170.0, 182.0, 195.0, 236.0,
    ];
    let squares: Vec<f64> = (1..=12).map(|i| f64::from(i * i)).collect();
    let logs: Vec<f64> = (1..=20).map(|i| f64::from(i).ln()).collect();
    let ramp: Vec<f64> = (1..=100).map(f64::from).collect();
    let cases: [(&str, Vec<f64>, f64); 9] = [
        // Shapiro and Wilk (1965): men's weights, W = 0.79.
        (
            "Shapiro-Wilk 1965 example",
            weights.to_vec(),
            0.7888146948353876,
        ),
        ("n = 3", vec![1.0, 2.0, 4.0], 0.9642857142857142),
        ("n = 4", vec![1.0, 2.0, 4.0, 8.0], 0.92020267879194),
        ("n = 5", vec![2.0, 4.0, 5.0, 7.0, 30.0], 0.7010554316967945),
        (
            "n = 6",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 20.0],
            0.6890610841684206,
        ),
        (
            "n = 7",
            vec![10.2, 11.1, 9.8, 10.5, 12.9, 10.0, 9.7],
            0.8037076397512073,
        ),
        ("n = 12 squares", squares, 0.9162924414742841),
        ("n = 20 logs", logs, 0.8883490714723817),
        ("n = 100 ramp", ramp, 0.9547247449917191),
    ];
    for (name, data, expected) in cases {
        let w = shapiro_wilk_w(&data);
        assert!(
            (w - expected).abs() < 1e-7,
            "{name}: W = {w}, reference {expected}"
        );
    }
    assert_close(
        shapiro_wilk_w(&blom_sample(1000)),
        0.9999031336873113,
        0.0,
        1e-7,
        "n = 1000 Blom",
    );
}

#[test]
fn shapiro_wilk_is_at_most_one_and_separates_normal_from_skewed_samples() {
    let normal = blom_sample(10);
    let w_normal = shapiro_wilk_w(&normal);
    assert!(
        w_normal > 0.95 && w_normal <= 1.0,
        "W of normal quantiles = {w_normal}"
    );
    // The unnormalised coefficients used to give W of the order of n.
    let skewed: Vec<f64> = (0..50)
        .map(|i| -(1.0 - (f64::from(i) + 0.5) / 50.0).ln())
        .collect();
    let w_skewed = shapiro_wilk_w(&skewed);
    assert!(
        w_skewed > 0.0 && w_skewed < 0.9,
        "W of exponential quantiles = {w_skewed}"
    );
    assert!(shapiro_wilk_w(&blom_sample(50)) > w_skewed + 0.1);
}

#[test]
fn shapiro_wilk_for_three_points_is_the_closed_form() {
    // W = (x3 - x1)^2 / (2 SS) for n = 3.
    for data in [
        [0.0, 1.0, 2.0],
        [0.0, 0.0, 1.0],
        [1.0, 2.0, 4.0],
        [-3.0, 0.5, 9.0],
    ] {
        let mean = data.iter().sum::<f64>() / 3.0;
        let ss: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();
        let expected = 0.5 * (data[2] - data[0]).powi(2) / ss;
        assert_close(shapiro_wilk_w(&data), expected, 1e-12, 0.0, "n = 3");
    }
}

#[test]
fn shapiro_wilk_is_bounded_and_invariant_under_affine_maps_and_reordering() {
    let mut rng = StatRng::new(5);
    for n in [3_usize, 4, 5, 6, 7, 10, 11, 25, 60, 200] {
        let data: Vec<f64> = (0..n)
            .map(|_| {
                let g = rng.next_normal();
                g * g * g + 0.3 * g
            })
            .collect();
        let w = shapiro_wilk_w(&data);
        assert!(w > 0.0 && w <= 1.0, "n = {n}: W = {w}");
        for (scale, shift) in [(3.7, -12.0), (1.0e-3, 4.0), (-2.5, 1.0e3)] {
            let mapped: Vec<f64> = data.iter().map(|x| scale * x + shift).collect();
            assert_close(shapiro_wilk_w(&mapped), w, 1e-9, 0.0, "affine invariance");
        }
        let mut reversed = data.clone();
        reversed.reverse();
        assert_eq!(shapiro_wilk_w(&reversed), w, "order must not matter");
    }
}

#[test]
fn shapiro_wilk_degenerate_inputs() {
    assert_eq!(shapiro_wilk_w(&[]), 1.0);
    assert_eq!(shapiro_wilk_w(&[1.0, 2.0]), 1.0);
    assert_eq!(shapiro_wilk_w(&[4.0, 4.0, 4.0, 4.0]), 1.0);
    assert_eq!(shapiro_wilk_w(&[1.0e-30, 1.0e-30, 1.0e-30]), 1.0);
    assert!(shapiro_wilk_w(&[1.0, f64::NAN, 3.0, 4.0]).is_nan());
    assert!(shapiro_wilk_w(&[1.0, f64::INFINITY, 3.0, 4.0]).is_nan());
    // Tiny but non-constant samples are scale-free.
    let tiny = [1.0e-30, 2.0e-30, 4.0e-30];
    assert_close(
        shapiro_wilk_w(&tiny),
        0.9642857142857142,
        1e-9,
        0.0,
        "tiny sample",
    );
}

// ---------------------------------------------------------------------------
// log-gamma based densities
// ---------------------------------------------------------------------------

#[test]
fn log_factorial_matches_exact_values_across_the_switchover() {
    assert_eq!(log_factorial(0), 0.0);
    assert_eq!(log_factorial(1), 0.0);
    assert_close(log_factorial(5), 120.0_f64.ln(), 1e-14, 0.0, "ln 5!");
    assert_close(log_factorial(20), 42.335616460753485, 1e-14, 0.0, "ln 20!");
    assert_close(log_factorial(21), 45.38013889847691, 1e-13, 0.0, "ln 21!");
    assert_close(log_factorial(30), 74.65823634883016, 1e-13, 0.0, "ln 30!");
    assert_close(log_factorial(200), 863.2319871924054, 1e-13, 0.0, "ln 200!");
    // Gamma recurrence across the k = 20 / 21 boundary.
    assert_close(
        log_factorial(21) - log_factorial(20),
        21.0_f64.ln(),
        0.0,
        1e-12,
        "recurrence",
    );
}

#[test]
fn beta_pdf_is_correctly_normalised() {
    let beta_2_3 = BetaDistribution::new(2.0, 3.0);
    for (x, expected) in [(0.1, 0.972), (0.4, 1.728), (0.5, 1.5), (0.9, 0.108)] {
        assert_close(beta_2_3.pdf(x), expected, 1e-12, 0.0, "Beta(2, 3) pdf");
    }
    // Composite Simpson rule over (0, 1) with 2000 panels.
    let panels = 2000;
    let h = 1.0 / f64::from(panels);
    let mut integral = beta_2_3.pdf(0.0) + beta_2_3.pdf(1.0);
    for i in 1..panels {
        let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
        integral += weight * beta_2_3.pdf(f64::from(i) * h);
    }
    integral *= h / 3.0;
    assert!(
        (integral - 1.0).abs() < 1e-6,
        "integral of the Beta(2, 3) pdf = {integral}"
    );
    assert_close(
        BetaDistribution::new(0.5, 0.5).pdf(0.3),
        0.6946091180428566,
        1e-12,
        0.0,
        "Beta(1/2, 1/2)",
    );
    assert_close(
        BetaDistribution::new(7.5, 2.25).pdf(0.3),
        0.02500453877374873,
        1e-11,
        0.0,
        "Beta(7.5, 2.25)",
    );
    assert_close(
        BetaDistribution::new(30.0, 40.0).pdf(0.3),
        0.5922745153911992,
        1e-10,
        0.0,
        "Beta(30, 40)",
    );
}

#[test]
fn poisson_pmf_matches_reference_values_and_sums_to_one() {
    let poisson = PoissonDistribution::new(30.0);
    let reference = [
        (0_u64, 9.357622968840175e-14),
        (5, 1.8949186511901354e-08),
        (20, 0.01341115001283779),
        (21, 0.019158785732625413),
        (25, 0.051115337428941324),
        (30, 0.07263452647159149),
        (50, 0.0002208784807212655),
        (100, 5.167581801838418e-24),
        (200, 3.1515764515926386e-93),
    ];
    for (k, expected) in reference {
        assert_close(
            poisson.pmf(k),
            expected,
            1e-11,
            0.0,
            &format!("Poisson(30) pmf({k})"),
        );
    }
    let total: f64 = (0..=200).map(|k| poisson.pmf(k)).sum();
    assert!((total - 1.0).abs() < 1e-9, "sum of the pmf = {total}");
    let mean: f64 = (0..=200).map(|k| k as f64 * poisson.pmf(k)).sum();
    assert!((mean - 30.0).abs() < 1e-8, "mean of the pmf = {mean}");
}

// ---------------------------------------------------------------------------
// Student-t p-values of the t-tests
// ---------------------------------------------------------------------------

#[test]
fn student_t_two_sided_p_matches_the_incomplete_beta_reference() {
    let reference = [
        (2.776445105197793, 4.0, 0.05),
        (2.0, 4.0, 0.1161165235168156),
        (1.0, 1.0, 0.5),
        (0.0, 7.0, 1.0),
        (10.0, 30.0, 4.575251408229613e-11),
        (50.0, 3.0, 1.7617152041271975e-05),
        (0.5, 2.5, 0.6576979198697147),
        (3.0, 12.75, 0.010431195738589558),
        (2.0, 1.0e5, 0.045502963457506464),
    ];
    for (t, nu, expected) in reference {
        assert_close(
            student_t_two_sided_p(t, nu),
            expected,
            1e-9,
            1e-15,
            &format!("p(t = {t}, nu = {nu})"),
        );
        assert_close(
            student_t_two_sided_p(-t, nu),
            expected,
            1e-9,
            1e-15,
            "p must be even in t",
        );
    }
    // The cdf-based identity p = 2 (1 - F(|t|)).
    assert_close(
        student_t_two_sided_p(1.7, 6.0),
        2.0 * (1.0 - student_t_cdf(1.7, 6.0)),
        1e-12,
        0.0,
        "tail identity",
    );
}

#[test]
fn one_sample_t_test_uses_the_t_distribution_with_n_minus_one_degrees_of_freedom() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let (t, p) = t_test_one_sample(&data, 0.0);
    assert_close(t, 4.242640687119285, 1e-12, 0.0, "t");
    assert_close(p, 0.01323559956368269, 1e-9, 0.0, "p");
    let (t, p) = t_test_one_sample(&data, 1.5);
    assert_close(t, 2.1213203435596424, 1e-12, 0.0, "t");
    assert_close(p, 0.10119150721829545, 1e-9, 0.0, "p");
    let (t, p) = t_test_one_sample(&data, 3.0);
    assert_eq!((t, p), (0.0, 1.0));
    let sample = [4.1, 5.3, 6.2, 5.8, 7.4, 6.6, 5.1, 6.9];
    let (t, p) = t_test_one_sample(&sample, 5.0);
    assert_close(t, 2.4359281865504485, 1e-12, 0.0, "t");
    assert_close(p, 0.04502714021725965, 1e-9, 0.0, "p");
    assert_eq!(t_test_one_sample_full(&sample, 5.0), (t, p));
}

#[test]
fn one_sample_t_test_at_the_critical_value_gives_five_percent() {
    // Five observations with mean 2.7764451 / sqrt(5) and unit sample variance
    // have t = 2.7764451 = t_{0.975, 4}; the normal approximation said p = 0.0055.
    let c = 0.4_f64.sqrt();
    let mean = 2.776445105197793 / 5.0_f64.sqrt();
    let data: Vec<f64> = [-2.0, -1.0, 0.0, 1.0, 2.0]
        .iter()
        .map(|k| mean + c * k)
        .collect();
    let (t, p) = t_test_one_sample(&data, 0.0);
    assert_close(t, 2.776445105197793, 1e-12, 0.0, "t");
    assert!((p - 0.05).abs() < 1e-8, "p = {p}");
}

#[test]
fn welch_t_test_uses_the_welch_satterthwaite_degrees_of_freedom() {
    let cases: [(&[f64], &[f64], f64, f64); 3] = [
        (
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
            -2.905487990874559,
            0.016533893656960716,
        ),
        (
            &[5.1, 4.9, 5.6, 5.8, 6.0, 5.2, 4.7],
            &[6.3, 6.8, 7.1, 6.5, 7.4, 6.9, 7.0],
            -6.661529028102187,
            3.207824123786516e-05,
        ),
        (
            &[10.0, 10.2, 9.9, 10.1, 10.3],
            &[10.0, 10.4, 9.8, 10.6, 9.6, 10.2, 10.1, 9.7, 10.5, 10.3],
            -0.15461042160902136,
            0.8795039209744359,
        ),
    ];
    for (x, y, expected_t, expected_p) in cases {
        let (t, p) = t_test_two_sample(x, y);
        assert_close(t, expected_t, 1e-12, 0.0, "t");
        assert_close(p, expected_p, 1e-9, 0.0, "p");
        assert_eq!(welch_t_test(x, y), (t, p));
        // Swapping the samples flips the sign of t and keeps the p-value.
        let (t_swapped, p_swapped) = t_test_two_sample(y, x);
        assert_close(t_swapped, -t, 1e-12, 0.0, "swapped t");
        assert_close(p_swapped, p, 1e-12, 1e-300, "swapped p");
    }
}

#[test]
fn t_tests_return_the_neutral_result_for_degenerate_samples() {
    assert_eq!(t_test_one_sample(&[1.0], 0.0), (0.0, 1.0));
    assert_eq!(t_test_one_sample(&[2.0, 2.0, 2.0], 0.0), (0.0, 1.0));
    assert_eq!(t_test_two_sample(&[1.0], &[1.0, 2.0]), (0.0, 1.0));
    assert_eq!(t_test_two_sample(&[3.0, 3.0], &[3.0, 3.0, 3.0]), (0.0, 1.0));
}

#[test]
fn t_test_p_values_approach_the_normal_limit_for_large_samples() {
    // 2001 observations: df = 2000, so the Student-t and normal tails nearly
    // agree.  Their difference is not a fixed fraction of the p-value: to first
    // order in 1 / df the two-sided tail exceeds the normal one by
    // phi(t) |t| (t^2 + 1) / (2 df), a relative 0.3 % at |t| = 2.  The
    // expansion is checked together with the plain closeness of the limits.
    let mut rng = StatRng::new(11);
    let data: Vec<f64> = (0..2001).map(|_| rng.next_normal()).collect();
    let (t, p) = t_test_one_sample(&data, 0.0);
    let normal_p = 2.0 * student_t_cdf(-t.abs(), f64::INFINITY);
    let df = 2000.0;
    let density = (-0.5 * t * t).exp() / std::f64::consts::TAU.sqrt();
    let first_order = normal_p + density * t.abs() * (t * t + 1.0) / (2.0 * df);
    assert!(
        (p - first_order).abs() < 1e-6,
        "t = {t}: p = {p}, first-order expansion {first_order}"
    );
    assert!(
        (p - normal_p).abs() < 3e-4,
        "t = {t}: p = {p}, normal limit {normal_p}"
    );
}

// ---------------------------------------------------------------------------
// anderson_darling_statistic
// ---------------------------------------------------------------------------

fn exact_normal_cdf(x: f64) -> f64 {
    student_t_cdf(x, f64::INFINITY)
}

#[test]
fn anderson_darling_matches_reference_values() {
    let unit_cdf = |x: f64| x;
    let uniform_100: Vec<f64> = (1..=100).map(|i| f64::from(i) / 101.0).collect();
    assert_close(
        anderson_darling_statistic(&uniform_100, unit_cdf),
        0.029_212_557_845_249_1,
        1e-9,
        0.0,
        "uniform n = 100",
    );
    let uniform_20: Vec<f64> = (1..=20).map(|i| f64::from(i) / 21.0).collect();
    assert_close(
        anderson_darling_statistic(&uniform_20, unit_cdf),
        0.09402320491525216,
        1e-9,
        0.0,
        "uniform n = 20",
    );
    let exponential = [0.1, 0.35, 0.5, 0.8, 1.1, 1.5, 2.2, 2.9, 4.0, 6.5];
    assert_close(
        anderson_darling_statistic(&exponential, |x| -(-x).exp_m1()),
        2.3740116870546895,
        1e-9,
        0.0,
        "exponential sample",
    );
    let normal = [-1.2, -0.7, -0.3, -0.1, 0.0, 0.2, 0.4, 0.9, 1.3, 2.1];
    assert_close(
        anderson_darling_statistic(&normal, exact_normal_cdf),
        0.4158631095782303,
        1e-9,
        0.0,
        "normal sample",
    );
    let shifted: Vec<f64> = normal.iter().map(|x| x + 1.5).collect();
    assert_close(
        anderson_darling_statistic(&shifted, exact_normal_cdf),
        14.716462287164777,
        1e-9,
        0.0,
        "shifted normal",
    );
}

#[test]
fn anderson_darling_is_small_for_a_good_fit_and_large_for_a_bad_one() {
    let good: Vec<f64> = (1..=100).map(|i| f64::from(i) / 101.0).collect();
    let bad: Vec<f64> = good.iter().map(|x| 0.5 * x).collect();
    let a2_good = anderson_darling_statistic(&good, |x| x);
    let a2_bad = anderson_darling_statistic(&bad, |x| x);
    assert!(a2_good < 0.1, "good fit A^2 = {a2_good}");
    assert_close(a2_bad, 38.72381991488973, 1e-9, 0.0, "bad fit");
    assert!(a2_bad > 100.0 * a2_good);
}

#[test]
fn anderson_darling_ignores_input_order_and_floors_out_of_support_points() {
    let normal = [-1.2, -0.7, -0.3, -0.1, 0.0, 0.2, 0.4, 0.9, 1.3, 2.1];
    let mut scrambled = normal;
    scrambled.reverse();
    scrambled.swap(2, 7);
    assert_eq!(
        anderson_darling_statistic(&normal, exact_normal_cdf),
        anderson_darling_statistic(&scrambled, exact_normal_cdf)
    );
    // Points outside the support have F = 0 or 1: the floor keeps A^2 finite.
    let clamp = |x: f64| x.clamp(0.0, 1.0);
    let outside = anderson_darling_statistic(&[-1.0, 0.5, 2.0], clamp);
    assert!(outside.is_finite() && outside > 10.0, "A^2 = {outside}");
    assert_eq!(anderson_darling_statistic(&[0.5], clamp), 0.0);
    assert_eq!(anderson_darling_statistic(&[], clamp), 0.0);
}

// ---------------------------------------------------------------------------
// skewness / kurtosis
// ---------------------------------------------------------------------------

#[test]
fn skewness_and_kurtosis_are_the_bias_corrected_g1_and_g2() {
    // Reference values from 50-digit arithmetic; they equal scipy.stats.skew and
    // scipy.stats.kurtosis with bias=False.  The old code returned 0.2706 and
    // -1.9787 for the first data set (the Bessel standard deviation was used
    // inside population moments).
    let first = [2.0, 8.0, 0.0, 4.0, 1.0, 9.0, 9.0, 0.0];
    assert_close(skewness(&first), 0.330_582_180_407_974_66, 1e-12, 0.0, "G1");
    assert_close(kurtosis(&first), -2.098_602_258_096_087, 1e-12, 0.0, "G2");
    let second = [3.1, 4.1, 5.9, 2.6, 5.3, 5.8, 9.7, 9.3, 2.3, 8.4];
    assert_close(skewness(&second), 0.339_432_721_195_207_7, 1e-12, 0.0, "G1");
    assert_close(
        kurtosis(&second),
        -1.322_660_171_317_440_6,
        1e-12,
        0.0,
        "G2",
    );
    let outlier = [1.0, 2.0, 3.0, 4.0, 100.0];
    assert_close(
        skewness(&outlier),
        2.232_395_911_636_457_5,
        1e-12,
        0.0,
        "G1",
    );
    assert_close(kurtosis(&outlier), 4.986_865_957_200_654, 1e-12, 0.0, "G2");
    // The named aliases are the same estimators.
    assert_eq!(sample_skewness(&first), skewness(&first));
    assert_eq!(sample_kurtosis(&first), kurtosis(&first));
}

#[test]
fn symmetric_data_has_zero_skewness_and_uniform_data_negative_excess_kurtosis() {
    let uniform = [1.0, 2.0, 3.0, 4.0, 5.0];
    assert!(skewness(&uniform).abs() < 1e-14);
    assert_close(kurtosis(&uniform), -1.2, 1e-12, 0.0, "G2 of 1..5");
}

#[test]
fn sample_kurtosis_of_small_normal_samples_is_unbiased() {
    // E[G2] = 0 for normal samples of any size n >= 4.  The uncorrected
    // population kurtosis g2 averages about -0.55 at n = 10, which is what the
    // old implementation (about -0.57 here) came close to.
    let mut rng = StatRng::new(4242);
    let samples = 20_000;
    let mut sum_g2 = 0.0;
    let mut sum_g1 = 0.0;
    for _ in 0..samples {
        let x: Vec<f64> = (0..10).map(|_| rng.next_normal()).collect();
        sum_g2 += kurtosis(&x);
        sum_g1 += skewness(&x);
    }
    let mean_g2 = sum_g2 / f64::from(samples);
    let mean_g1 = sum_g1 / f64::from(samples);
    // The standard errors are 1.34 / sqrt(20000) = 0.0095 and 0.0071.
    assert!(mean_g2.abs() < 0.05, "mean G2 = {mean_g2}");
    assert!(mean_g1.abs() < 0.05, "mean G1 = {mean_g1}");
}

#[test]
fn skewness_and_kurtosis_are_invariant_under_affine_maps() {
    let data = [2.0, 8.0, 0.0, 4.0, 1.0, 9.0, 9.0, 0.0];
    let moved: Vec<f64> = data.iter().map(|x| 1.0e8 + 3.0 * x).collect();
    assert_close(skewness(&moved), skewness(&data), 1e-7, 0.0, "G1 shifted");
    assert_close(kurtosis(&moved), kurtosis(&data), 1e-7, 0.0, "G2 shifted");
    let mirrored: Vec<f64> = data.iter().map(|x| -x).collect();
    assert_close(
        skewness(&mirrored),
        -skewness(&data),
        1e-12,
        0.0,
        "G1 mirrored",
    );
    assert_close(
        kurtosis(&mirrored),
        kurtosis(&data),
        1e-12,
        0.0,
        "G2 mirrored",
    );
}

#[test]
fn constant_data_has_zero_skewness_and_kurtosis() {
    // The mean of [0.1, 0.1, 0.1] is 1 ulp away from 0.1, so the deviations are
    // 1e-17 rounding noise and their standardised moments used to come out as
    // garbage (about -1 for the skewness).
    assert_eq!(skewness(&[0.1, 0.1, 0.1]), 0.0);
    assert_eq!(skewness(&[0.1; 10]), 0.0);
    assert_eq!(kurtosis(&[0.1, 0.1, 0.1, 0.1]), 0.0);
    assert_eq!(kurtosis(&[0.1; 25]), 0.0);
    assert_eq!(skewness(&[7.0; 5]), 0.0);
    assert_eq!(kurtosis(&[-3.5; 6]), 0.0);
    assert_eq!(kurtosis(&[0.0; 6]), 0.0);
    // A long constant series: a plainly summed mean drifts by far more than the
    // data resolution (about 4e-14 relative for 1e5 copies of 1/3).
    assert_eq!(skewness(&vec![0.3; 10_000]), 0.0);
    assert_eq!(kurtosis(&vec![1.0 / 3.0; 100_000]), 0.0);
    // A tight cluster far from the origin is not constant.
    let cluster: Vec<f64> = (1..=5).map(|i| 1.0e8 + f64::from(i)).collect();
    assert_close(kurtosis(&cluster), -1.2, 1e-6, 0.0, "G2 of 1e8 + 1..5");
}

#[test]
fn skewness_and_kurtosis_of_short_series_are_zero() {
    assert_eq!(skewness(&[]), 0.0);
    assert_eq!(skewness(&[1.0, 2.0]), 0.0);
    assert_eq!(kurtosis(&[1.0, 2.0, 4.0]), 0.0);
    assert!(skewness(&[1.0, 2.0, 4.0]).is_finite());
    assert!(kurtosis(&[1.0, 2.0, 4.0, 9.0]).is_finite());
}

// ---------------------------------------------------------------------------
// pacf (Durbin-Levinson)
// ---------------------------------------------------------------------------

/// Exact autocorrelations `rho_0..rho_max_lag` of the stationary AR(2) process
/// `x_t = phi1 x_{t-1} + phi2 x_{t-2} + e_t`.
fn ar2_acf(phi1: f64, phi2: f64, max_lag: usize) -> Vec<f64> {
    let mut rho = vec![1.0, phi1 / (1.0 - phi2)];
    for k in 2..=max_lag {
        rho.push(phi1 * rho[k - 1] + phi2 * rho[k - 2]);
    }
    rho
}

/// `phi_kk`: the last coefficient of the order-`k` Yule-Walker solution,
/// obtained by Gaussian elimination with partial pivoting.
fn yule_walker_last_coefficient(rho: &[f64], k: usize) -> f64 {
    let mut a: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            let mut row: Vec<f64> = (0..k).map(|j| rho[i.abs_diff(j)]).collect();
            row.push(rho[i + 1]);
            row
        })
        .collect();
    for col in 0..k {
        let pivot = (col..k)
            .max_by(|&p, &q| a[p][col].abs().total_cmp(&a[q][col].abs()))
            .expect("non-empty pivot range");
        a.swap(col, pivot);
        for row in col + 1..k {
            let factor = a[row][col] / a[col][col];
            let (upper, lower) = a.split_at_mut(row);
            for (target, pivot) in lower[0].iter_mut().zip(&upper[col]).skip(col) {
                *target -= factor * pivot;
            }
        }
    }
    let mut solution = vec![0.0; k];
    for i in (0..k).rev() {
        let tail: f64 = (i + 1..k).map(|j| a[i][j] * solution[j]).sum();
        solution[i] = (a[i][k] - tail) / a[i][i];
    }
    solution[k - 1]
}

#[test]
fn pacf_of_an_ar2_acf_cuts_off_after_lag_two() {
    // The old recursion reused the PACF diagonal as the predictor coefficients
    // and returned -0.481 at lag 3 and 0.411 at lag 4 (both must vanish).
    let rho = ar2_acf(0.5, 0.3, 12);
    let partial = pacf_from_acf(&rho);
    assert_eq!(partial.len(), rho.len());
    assert_eq!(partial[0], 1.0);
    assert_close(partial[1], 0.5 / 0.7, 1e-14, 0.0, "pacf[1]");
    assert_close(partial[2], 0.3, 1e-13, 0.0, "pacf[2]");
    for (k, value) in partial.iter().enumerate().skip(3) {
        assert!(value.abs() < 1e-12, "pacf[{k}] = {value}");
    }
}

#[test]
fn pacf_of_an_ar1_acf_cuts_off_after_lag_one() {
    let rho: Vec<f64> = (0..8).map(|k| 0.6_f64.powi(k)).collect();
    let partial = pacf_from_acf(&rho);
    assert_close(partial[1], 0.6, 1e-14, 0.0, "pacf[1]");
    for (k, value) in partial.iter().enumerate().skip(2) {
        assert!(value.abs() < 1e-13, "pacf[{k}] = {value}");
    }
}

#[test]
fn pacf_matches_the_direct_yule_walker_solution_for_an_arma_acf() {
    // ARMA(1,1): rho_1 = (1 + phi theta)(phi + theta) / (1 + 2 phi theta + theta^2),
    // rho_k = phi rho_{k-1}; its PACF decays but never cuts off.
    let (phi, theta) = (0.7, 0.4);
    let rho1 = (1.0 + phi * theta) * (phi + theta) / (1.0 + 2.0 * phi * theta + theta * theta);
    let mut rho = vec![1.0, rho1];
    for k in 2..=8 {
        rho.push(phi * rho[k - 1]);
    }
    let partial = pacf_from_acf(&rho);
    for (k, value) in partial.iter().enumerate().skip(1) {
        let direct = yule_walker_last_coefficient(&rho, k);
        assert!(
            (value - direct).abs() < 1e-10,
            "pacf[{k}] = {value}, Yule-Walker gives {direct}"
        );
    }
    assert!(partial[5].abs() > 1e-4, "an ARMA PACF does not cut off");
}

#[test]
fn pacf_of_an_ar2_series_recovers_the_cut_off() {
    let (phi1, phi2) = (0.5, 0.3);
    let mut rng = StatRng::new(31);
    let mut series = Vec::with_capacity(40_000);
    let (mut x1, mut x2) = (0.0_f64, 0.0_f64);
    for t in 0..40_000 {
        let x = phi1 * x1 + phi2 * x2 + rng.next_normal();
        x2 = x1;
        x1 = x;
        if t >= 100 {
            series.push(x);
        }
    }
    let partial = pacf(&series, 6);
    assert_eq!(partial.len(), 7);
    // Standard error 1 / sqrt(40000) = 0.005.
    assert!(
        (partial[1] - 0.5 / 0.7).abs() < 0.03,
        "pacf[1] = {}",
        partial[1]
    );
    assert!((partial[2] - 0.3).abs() < 0.03, "pacf[2] = {}", partial[2]);
    for (k, value) in partial.iter().enumerate().skip(3) {
        assert!(value.abs() < 0.03, "pacf[{k}] = {value}");
    }
}

#[test]
fn pacf_handles_degenerate_inputs() {
    assert_eq!(pacf_from_acf(&[]), vec![1.0]);
    assert_eq!(pacf_from_acf(&[1.0]), vec![1.0]);
    assert_eq!(pacf(&[3.0], 4), vec![1.0]);
    // A first-order sequence has just its lag-one coefficient.
    assert_eq!(pacf_from_acf(&[1.0, 0.25]), vec![1.0, 0.25]);
    // A constant series has a unit ACF; the vanishing prediction-error
    // denominator must not produce NaN.
    let constant = pacf(&[2.0; 10], 4);
    assert_eq!(constant, vec![1.0, 1.0, 0.0, 0.0, 0.0]);
}

// ---------------------------------------------------------------------------
// Poisson sampling (PTRS)
// ---------------------------------------------------------------------------

/// Sample mean and unbiased variance of `n` draws from `Poisson(lambda)`.
fn poisson_moments(lambda: f64, n: usize, seed: u64) -> (f64, f64) {
    let mut rng = StatRng::new(seed);
    let distribution = PoissonDistribution::new(lambda);
    let draws: Vec<f64> = (0..n)
        .map(|_| distribution.sample(&mut rng) as f64)
        .collect();
    (mean(&draws), variance(&draws))
}

/// Upper 0.1 % point of the chi-square distribution with `dof` degrees of
/// freedom (Wilson-Hilferty approximation).
fn chi_square_critical(dof: usize) -> f64 {
    let k = dof as f64;
    let z_999 = 3.090_232_306_167_813;
    let t = 2.0 / (9.0 * k);
    k * (1.0 - t + z_999 * t.sqrt()).powi(3)
}

#[test]
fn poisson_sample_with_a_large_mean_is_not_capped_near_745() {
    // Knuth's loop returned about 745 for every lambda above that, as exp(-lambda)
    // underflows to 0.
    let (sample_mean, sample_variance) = poisson_moments(1000.0, 40_000, 99);
    assert!(
        (sample_mean - 1000.0).abs() < 10.0,
        "mean = {sample_mean}, expected 1000 within 1 %"
    );
    assert!(
        (sample_variance - 1000.0).abs() < 50.0,
        "variance = {sample_variance}, expected 1000 within 5 %"
    );
    let mut rng = StatRng::new(5);
    let largest = (0..2_000)
        .map(|_| PoissonDistribution::new(5000.0).sample(&mut rng))
        .max()
        .expect("non-empty sample");
    assert!(
        largest > 5000,
        "the draws must exceed the old cap, got {largest}"
    );
}

#[test]
fn poisson_moments_are_right_on_both_sides_of_the_ptrs_threshold() {
    for lambda in [0.5, 5.0, 9.99, 10.0, 10.01, 30.0, 200.0, 5000.0, 1.0e6] {
        let n = 100_000;
        let (sample_mean, sample_variance) = poisson_moments(lambda, n, 17);
        let standard_error = (lambda / n as f64).sqrt();
        assert!(
            (sample_mean - lambda).abs() < 5.0 * standard_error,
            "lambda = {lambda}: mean {sample_mean}"
        );
        // Var[s^2] = lambda (2 lambda (n - 1) / n + 1) ... / n for a Poisson law.
        let variance_error = lambda * ((2.0 + 1.0 / lambda) / n as f64).sqrt();
        assert!(
            (sample_variance - lambda).abs() < 5.0 * variance_error,
            "lambda = {lambda}: variance {sample_variance}"
        );
    }
}

/// Pearson chi-square statistic of `draws` draws against the Poisson pmf, with
/// the right tail pooled into one cell and cells below 5 expected counts merged.
fn poisson_chi_square(lambda: f64, draws: u32, seed: u64) -> (f64, usize) {
    let distribution = PoissonDistribution::new(lambda);
    let mut rng = StatRng::new(seed);
    let cells = (lambda + 12.0 * lambda.sqrt() + 12.0) as usize;
    let mut observed = vec![0_u32; cells + 1];
    for _ in 0..draws {
        let k = (distribution.sample(&mut rng) as usize).min(cells);
        observed[k] += 1;
    }
    let mut chi_square = 0.0;
    let mut used_cells = 0;
    let (mut pooled_observed, mut pooled_expected) = (0.0, 0.0);
    let mut cumulative = 0.0;
    for (k, &count) in observed.iter().enumerate().take(cells) {
        let expected = f64::from(draws) * distribution.pmf(k as u64);
        cumulative += distribution.pmf(k as u64);
        pooled_observed += f64::from(count);
        pooled_expected += expected;
        if pooled_expected >= 5.0 {
            chi_square += (pooled_observed - pooled_expected).powi(2) / pooled_expected;
            used_cells += 1;
            pooled_observed = 0.0;
            pooled_expected = 0.0;
        }
    }
    // Everything left, including the pooled right tail, forms the last cell.
    pooled_observed += f64::from(observed[cells]);
    pooled_expected += f64::from(draws) * (1.0 - cumulative);
    if pooled_expected > 0.0 {
        chi_square += (pooled_observed - pooled_expected).powi(2) / pooled_expected;
        used_cells += 1;
    }
    (chi_square, used_cells - 1)
}

#[test]
fn poisson_samples_follow_the_probability_mass_function() {
    // 9.5 uses Knuth's method, the others PTRS (the squeeze and the log test).
    for lambda in [9.5, 10.0, 12.0, 30.0, 150.0] {
        let (chi_square, dof) = poisson_chi_square(lambda, 400_000, 2_024);
        assert!(
            chi_square < chi_square_critical(dof),
            "lambda = {lambda}: chi^2 = {chi_square} with {dof} degrees of freedom \
             exceeds the 0.1 % point {}",
            chi_square_critical(dof)
        );
    }
}

#[test]
fn poisson_ptrs_costs_a_constant_number_of_uniforms() {
    let mut rng = StatRng::new(8);
    let mut calls = 0_u64;
    let draws = 20_000_u64;
    let mut uniform = || {
        calls += 1;
        rng.next_f64()
    };
    for _ in 0..draws {
        let _ = poisson_ptrs(1.0e5, &mut uniform);
    }
    // Hormann's acceptance rate is above 90 %: about 2.3 uniforms per variate,
    // against roughly 1e5 for Knuth's product method.
    let per_draw = calls as f64 / draws as f64;
    assert!(per_draw < 3.0, "{per_draw} uniforms per draw");
}

#[test]
fn poisson_ptrs_rejects_non_positive_and_non_finite_means_without_drawing() {
    let mut never = || -> f64 { panic!("no uniform may be drawn") };
    for lambda in [
        0.0,
        -1.0,
        -1.0e300,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert_eq!(poisson_ptrs(lambda, &mut never), 0, "lambda = {lambda}");
    }
    let mut rng = StatRng::new(1);
    assert_eq!(PoissonDistribution::new(0.0).sample(&mut rng), 0);
    assert_eq!(PoissonDistribution::new(-3.0).sample(&mut rng), 0);
    assert_eq!(PoissonDistribution::new(f64::NAN).sample(&mut rng), 0);
}

#[test]
fn poisson_ptrs_handles_extreme_means() {
    // Beyond 1e9 the normal approximation takes over; the draws stay centred on
    // lambda with a spread of sqrt(lambda).
    for lambda in [1.0e9, 1.0e12, 1.0e15] {
        let mut rng = StatRng::new(3);
        let n = 4_000;
        let draws: Vec<f64> = (0..n)
            .map(|_| poisson_ptrs(lambda, &mut || rng.next_f64()) as f64)
            .collect();
        let sigma = lambda.sqrt();
        assert!(
            (mean(&draws) - lambda).abs() < 5.0 * sigma / (n as f64).sqrt(),
            "lambda = {lambda}: mean {}",
            mean(&draws)
        );
        assert!(
            (variance(&draws).sqrt() / sigma - 1.0).abs() < 0.1,
            "lambda = {lambda}: sd {}",
            variance(&draws).sqrt()
        );
    }
    // A mean beyond the u64 range saturates instead of wrapping.
    let mut rng = StatRng::new(4);
    assert_eq!(poisson_ptrs(1.0e30, &mut || rng.next_f64()), u64::MAX);
}

#[test]
fn poisson_sampling_is_deterministic_per_seed() {
    let distribution = PoissonDistribution::new(250.0);
    let run = |seed: u64| -> Vec<u64> {
        let mut rng = StatRng::new(seed);
        (0..32).map(|_| distribution.sample(&mut rng)).collect()
    };
    assert_eq!(run(11), run(11));
    assert_ne!(run(11), run(12));
}
