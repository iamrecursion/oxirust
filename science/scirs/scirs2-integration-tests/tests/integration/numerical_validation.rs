//! Numerical validation tests: verify accuracy against analytically known values.
//! These tests serve as v1.0.0 quality gate for numerical correctness.

// =================== Special Functions ===================

#[cfg(test)]
mod special_functions {
    use scirs2_special::{beta, erf, erfc, gamma, j0};

    /// gamma(n) = (n-1)! for positive integers
    #[test]
    fn test_gamma_integers() {
        let cases: [(f64, f64); 7] = [
            (1.0, 1.0),
            (2.0, 1.0),
            (3.0, 2.0),
            (4.0, 6.0),
            (5.0, 24.0),
            (6.0, 120.0),
            (7.0, 720.0),
        ];
        for (x, expected) in cases {
            let got = gamma(x);
            let rel_err = (got - expected).abs() / expected;
            assert!(
                rel_err < 1e-10,
                "gamma({x}) = {got}, expected {expected}, rel_err = {rel_err}"
            );
        }
    }

    /// Half-integer gamma values: gamma(1/2) = sqrt(pi), etc.
    #[test]
    fn test_gamma_half_integers() {
        let sqrt_pi = std::f64::consts::PI.sqrt();

        let g_half = gamma(0.5f64);
        assert!(
            (g_half - sqrt_pi).abs() < 1e-12,
            "gamma(0.5) = {g_half}, expected {sqrt_pi}"
        );

        let g_3half = gamma(1.5f64);
        let expected_3half = sqrt_pi / 2.0;
        assert!(
            (g_3half - expected_3half).abs() < 1e-12,
            "gamma(1.5) = {g_3half}, expected {expected_3half}"
        );

        let g_5half = gamma(2.5f64);
        let expected_5half = 3.0 * sqrt_pi / 4.0;
        assert!(
            (g_5half - expected_5half).abs() < 1e-12,
            "gamma(2.5) = {g_5half}, expected {expected_5half}"
        );
    }

    /// Recurrence: gamma(x+1) = x * gamma(x)
    #[test]
    fn test_gamma_recurrence() {
        let xs = [0.3f64, 0.7, 1.2, 1.8, 2.5, 3.1, 4.4];
        for x in xs {
            let lhs = gamma(x + 1.0);
            let rhs = x * gamma(x);
            let err = (lhs - rhs).abs();
            assert!(
                err < 1e-11,
                "gamma recurrence failed at x={x}: gamma(x+1)={lhs}, x*gamma(x)={rhs}, err={err}"
            );
        }
    }

    /// erf known values cross-checked against reference tables
    #[test]
    fn test_erf_known_values() {
        let cases: [(f64, f64); 5] = [
            (0.0, 0.0),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (3.0, 0.999_977_909_503_001_4),
            (-1.0, -0.842_700_792_949_714_9),
        ];
        for (x, expected) in cases {
            let got = erf(x);
            // scirs2-special erf has ~7 digit accuracy
            assert!(
                (got - expected).abs() < 1e-6,
                "erf({x}) = {got}, expected {expected}"
            );
        }
    }

    /// erf is an odd function: erf(-x) = -erf(x)
    #[test]
    fn test_erf_odd_symmetry() {
        let xs = [0.1f64, 0.5, 1.0, 2.0, 3.5];
        for x in xs {
            let lhs = erf(-x);
            let rhs = -erf(x);
            assert!(
                (lhs - rhs).abs() < 1e-14,
                "erf odd symmetry failed at x={x}: erf(-x)={lhs}, -erf(x)={rhs}"
            );
        }
    }

    /// erf(x) + erfc(x) = 1
    #[test]
    fn test_erf_erfc_complement() {
        let xs = [0.0f64, 0.5, 1.0, 2.0, 4.0];
        for x in xs {
            let sum = erf(x) + erfc(x);
            assert!(
                (sum - 1.0).abs() < 1e-14,
                "erf({x}) + erfc({x}) = {sum}, expected 1.0"
            );
        }
    }

    /// J0(0) = 1 and J0 at its first zero (~2.4048) is approximately 0
    #[test]
    fn test_j0_known_values() {
        let j0_at_zero = j0(0.0f64);
        assert!(
            (j0_at_zero - 1.0).abs() < 1e-14,
            "J0(0) = {j0_at_zero}, expected 1.0"
        );

        // First zero of J0 is 2.404825557695773
        let first_zero = 2.404_825_557_695_773_f64;
        let j0_at_first_zero = j0(first_zero);
        assert!(
            j0_at_first_zero.abs() < 1e-10,
            "J0(first_zero) = {j0_at_first_zero}, expected ~0"
        );
    }

    /// Beta symmetry: beta(a, b) = beta(b, a)
    #[test]
    fn test_beta_symmetry() {
        let pairs: [(f64, f64); 4] = [(1.5, 2.5), (3.0, 1.0), (0.5, 3.5), (2.0, 4.0)];
        for (a, b) in pairs {
            let b12 = beta(a, b);
            let b21 = beta(b, a);
            assert!(
                (b12 - b21).abs() < 1e-14,
                "beta({a},{b}) = {b12} != beta({b},{a}) = {b21}"
            );
        }
    }

    /// beta(1, 1) = 1
    #[test]
    fn test_beta_unit_value() {
        let b11 = beta(1.0f64, 1.0f64);
        assert!((b11 - 1.0).abs() < 1e-14, "beta(1,1) = {b11}");
    }

    /// beta(n, 1) = 1/n
    #[test]
    fn test_beta_integer_one_arg() {
        for n in [2u32, 3, 5, 10] {
            let expected = 1.0 / n as f64;
            let got = beta(n as f64, 1.0f64);
            assert!(
                (got - expected).abs() < 1e-14,
                "beta({n}, 1) = {got}, expected {expected}"
            );
        }
    }

    /// beta(a, b) = gamma(a)*gamma(b)/gamma(a+b)
    #[test]
    fn test_beta_gamma_relation() {
        let pairs: [(f64, f64); 3] = [(2.0, 3.0), (0.5, 0.5), (1.5, 2.5)];
        for (a, b) in pairs {
            let from_beta = beta(a, b);
            let from_gamma = gamma(a) * gamma(b) / gamma(a + b);
            let rel_err = (from_beta - from_gamma).abs() / from_beta.abs().max(1e-30);
            assert!(
                rel_err < 1e-10,
                "beta({a},{b}) via beta={from_beta}, via gamma={from_gamma}, rel_err={rel_err}"
            );
        }
    }
}

// =================== Statistics ===================

#[cfg(test)]
mod statistics_validation {
    use scirs2_stats::distributions::{chi_square::ChiSquare, normal::Normal, poisson::Poisson};

    /// Standard normal pdf at 0 = 1/sqrt(2*pi)
    #[test]
    fn test_normal_pdf_at_zero() {
        let n = Normal::new(0.0f64, 1.0).expect("Normal::new");
        let pdf_0 = n.pdf(0.0);
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!(
            (pdf_0 - expected).abs() < 1e-14,
            "N(0,1) pdf at 0: {pdf_0}, expected {expected}"
        );
    }

    /// Standard normal pdf at 1 = exp(-0.5)/sqrt(2*pi)
    #[test]
    fn test_normal_pdf_at_one() {
        let n = Normal::new(0.0f64, 1.0).expect("Normal::new");
        let pdf_1 = n.pdf(1.0);
        let expected = (-0.5f64).exp() / (2.0 * std::f64::consts::PI).sqrt();
        assert!(
            (pdf_1 - expected).abs() < 1e-14,
            "N(0,1) pdf at 1: {pdf_1}, expected {expected}"
        );
    }

    /// Standard normal cdf at 0 = 0.5
    #[test]
    fn test_normal_cdf_at_zero() {
        let n = Normal::new(0.0f64, 1.0).expect("Normal::new");
        let cdf_0 = n.cdf(0.0);
        assert!(
            (cdf_0 - 0.5).abs() < 1e-14,
            "N(0,1) cdf at 0: {cdf_0}, expected 0.5"
        );
    }

    /// Normal cdf symmetry: Phi(x) + Phi(-x) = 1
    #[test]
    fn test_normal_cdf_symmetry() {
        let n = Normal::new(0.0f64, 1.0).expect("Normal::new");
        for x in [0.5f64, 1.0, 1.5, 1.96, 2.576] {
            let sum = n.cdf(x) + n.cdf(-x);
            assert!(
                (sum - 1.0).abs() < 1e-13,
                "N cdf symmetry at x={x}: Phi(x)+Phi(-x)={sum}"
            );
        }
    }

    /// Normal cdf at large positive x approaches 1
    #[test]
    fn test_normal_cdf_tail() {
        let n = Normal::new(0.0f64, 1.0).expect("Normal::new");
        assert!(
            (n.cdf(100.0) - 1.0).abs() < 1e-10,
            "N cdf at +100 should be ~1"
        );
    }

    /// P(X=0; lambda=3) = exp(-3)
    #[test]
    fn test_poisson_pmf_at_zero() {
        let p = Poisson::new(3.0f64, 0.0).expect("Poisson::new");
        let expected = (-3.0f64).exp();
        let got = p.pmf(0.0);
        assert!(
            (got - expected).abs() < 1e-14,
            "Poisson(3) pmf at 0: {got}, expected {expected}"
        );
    }

    /// P(X=3; lambda=3) = exp(-3) * 27/6
    #[test]
    fn test_poisson_pmf_at_mode() {
        let p = Poisson::new(3.0f64, 0.0).expect("Poisson::new");
        let expected = (-3.0f64).exp() * 27.0 / 6.0;
        let got = p.pmf(3.0);
        assert!(
            (got - expected).abs() < 1e-13,
            "Poisson(3) pmf at 3: {got}, expected {expected}"
        );
    }

    /// P(X=1; lambda=1) = exp(-1)
    #[test]
    fn test_poisson_pmf_unit_rate() {
        let p = Poisson::new(1.0f64, 0.0).expect("Poisson::new");
        let expected = (-1.0f64).exp();
        let got = p.pmf(1.0);
        assert!(
            (got - expected).abs() < 1e-14,
            "Poisson(1) pmf at 1: {got}, expected {expected}"
        );
    }

    /// Chi-squared df=2: pdf(x) = 0.5 * exp(-x/2)
    #[test]
    fn test_chi_squared_df2_pdf() {
        let chi2 = ChiSquare::new(2.0f64, 0.0, 1.0).expect("ChiSquare::new");
        let xs = [0.5f64, 1.0, 2.0, 3.0, 4.0];
        for x in xs {
            let got = chi2.pdf(x);
            let expected = 0.5 * (-x / 2.0).exp();
            assert!(
                (got - expected).abs() < 1e-11,
                "Chi2(df=2) pdf({x}) = {got}, expected {expected}"
            );
        }
    }

    /// Chi-squared pdf is non-negative for positive x
    #[test]
    fn test_chi_squared_pdf_non_negative() {
        let chi2 = ChiSquare::new(4.0f64, 0.0, 1.0).expect("ChiSquare::new");
        for x in [0.1f64, 1.0, 5.0, 10.0] {
            let pdf_val = chi2.pdf(x);
            assert!(
                pdf_val >= 0.0,
                "Chi2 pdf should be non-negative, got {pdf_val} at x={x}"
            );
        }
    }
}

// =================== Linear Algebra ===================

#[cfg(test)]
mod linalg_validation {
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};
    use scirs2_linalg::{eigh, solve, svd};

    /// Solve 2x2 system with known exact solution
    #[test]
    fn test_matrix_solve_exact_2x2() {
        // [[2, 1], [1, 3]] * x = [5, 10] => x = [1, 3]
        let a: Array2<f64> = array![[2.0, 1.0], [1.0, 3.0]];
        let b = array![5.0, 10.0];
        let x = solve(&a.view(), &b.view(), None).expect("solve failed");
        assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1], 3.0, epsilon = 1e-12);
    }

    /// Solve 3x3 diagonal system: diag(2,3,4) * x = [4, 9, 16] => x = [2, 3, 4]
    #[test]
    fn test_matrix_solve_diagonal_3x3() {
        let a: Array2<f64> = array![[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
        let b = array![4.0, 9.0, 16.0];
        let x = solve(&a.view(), &b.view(), None).expect("solve diagonal failed");
        assert_abs_diff_eq!(x[0], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[2], 4.0, epsilon = 1e-12);
    }

    /// Symmetric 2x2 matrix [[2,1],[1,2]] has eigenvalues 1 and 3
    #[test]
    fn test_eigh_2x2_known_eigenvalues() {
        let a: Array2<f64> = array![[2.0, 1.0], [1.0, 2.0]];
        let (evals, _evecs) = eigh(&a.view(), None).expect("eigh failed");
        let mut sorted = evals.to_vec();
        sorted.sort_by(|p, q| p.partial_cmp(q).expect("NaN in eigenvalues"));
        assert_abs_diff_eq!(sorted[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(sorted[1], 3.0, epsilon = 1e-12);
    }

    /// Identity matrix eigenvalues are all 1
    #[test]
    fn test_eigh_identity_eigenvalues() {
        let a: Array2<f64> = Array2::eye(3);
        let (evals, _evecs) = eigh(&a.view(), None).expect("eigh identity failed");
        for (i, &ev) in evals.iter().enumerate() {
            assert!(
                (ev - 1.0).abs() < 1e-12,
                "eigenvalue[{i}] = {ev}, expected 1.0"
            );
        }
    }

    /// SVD reconstruction: A = U * diag(s) * Vt
    #[test]
    fn test_svd_reconstruction_2x3() {
        let a: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        // full_matrices=false => U is 2x2, s has 2 entries, Vt is 2x3
        let (u, s, vt) = svd(&a.view(), false, None).expect("svd failed");
        // Build Sigma as 2x2 diagonal
        let sigma = Array2::from_diag(&s);
        let reconstructed = u.dot(&sigma.dot(&vt));
        for i in 0..2 {
            for j in 0..3 {
                let diff = (reconstructed[[i, j]] - a[[i, j]]).abs();
                assert!(
                    diff < 1e-10,
                    "SVD reconstruction mismatch at [{i},{j}]: got {}, expected {}, diff={diff}",
                    reconstructed[[i, j]],
                    a[[i, j]]
                );
            }
        }
    }

    /// SVD singular values are non-negative and in descending order
    #[test]
    fn test_svd_singular_values_ordered() {
        let a: Array2<f64> = array![[3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [1.0, 1.0, 1.0]];
        let (_u, s, _vt) = svd(&a.view(), false, None).expect("svd ordered failed");
        for i in 0..(s.len() - 1) {
            assert!(
                s[i] >= s[i + 1],
                "singular values not ordered: s[{i}]={} < s[{}]={}",
                s[i],
                i + 1,
                s[i + 1]
            );
            assert!(s[i] >= 0.0, "singular value s[{i}]={} is negative", s[i]);
        }
    }
}

// =================== FFT Validation ===================

#[cfg(test)]
mod fft_validation {
    use approx::assert_abs_diff_eq;
    use scirs2_fft::{fft, ifft};

    /// FFT of constant signal: all energy in DC bin
    #[test]
    fn test_fft_dc_component() {
        let n = 8usize;
        let c = 3.0f64;
        let signal: Vec<f64> = vec![c; n];
        let spectrum = fft(&signal, None).expect("fft failed");

        // DC component = sum = N * c
        assert_abs_diff_eq!(spectrum[0].re, n as f64 * c, epsilon = 1e-10);
        assert_abs_diff_eq!(spectrum[0].im, 0.0, epsilon = 1e-10);

        // All non-DC components should be negligible
        for (k, s) in spectrum.iter().enumerate().skip(1) {
            assert!(
                s.norm() < 1e-10,
                "non-DC component at k={k}: |X[{k}]| = {}",
                s.norm()
            );
        }
    }

    /// Parseval's theorem: sum |x[n]|^2 = (1/N) * sum |X[k]|^2
    #[test]
    fn test_fft_parseval() {
        let n = 16usize;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let spectrum = fft(&signal, None).expect("fft parseval failed");

        let time_energy: f64 = signal.iter().map(|&x| x * x).sum();
        let freq_energy: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum::<f64>() / n as f64;

        assert_abs_diff_eq!(time_energy, freq_energy, epsilon = 1e-10);
    }

    /// FFT of a single pure tone: energy concentrated at frequency bin k0
    #[test]
    fn test_fft_pure_tone_localization() {
        let n = 16usize;
        let k0 = 2usize; // frequency bin index
                         // Signal = cos(2*pi*k0*t/N) has spectral lines at ±k0
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * k0 as f64 * i as f64 / n as f64).cos())
            .collect();
        let spectrum = fft(&signal, None).expect("fft tone failed");

        // Magnitude at k0 should be n/2 (for real cosine)
        let mag_k0 = spectrum[k0].norm();
        assert!(
            (mag_k0 - n as f64 / 2.0).abs() < 1e-9,
            "Expected magnitude {}, got {} at bin k0={k0}",
            n as f64 / 2.0,
            mag_k0
        );

        // All other bins (except mirror at n-k0) should be near zero
        let mirror = n - k0;
        for (k, s) in spectrum.iter().enumerate() {
            if k != k0 && k != mirror {
                assert!(
                    s.norm() < 1e-10,
                    "Unexpected energy at bin k={k}: |X[{k}]| = {}",
                    s.norm()
                );
            }
        }
    }

    /// IFFT(FFT(x)) = x: roundtrip recovers the original signal
    #[test]
    fn test_fft_ifft_roundtrip() {
        let signal: Vec<f64> = vec![1.0, 2.0, -1.0, 3.0, 0.5, -2.0, 1.5, 0.0];
        let n = signal.len();
        let spectrum = fft(&signal, None).expect("fft roundtrip failed");
        let recovered = ifft(&spectrum, Some(n)).expect("ifft roundtrip failed");
        for (i, (orig, rec)) in signal.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec.re).abs() < 1e-12,
                "roundtrip mismatch at index {i}: orig={orig}, rec.re={}",
                rec.re
            );
            assert!(
                rec.im.abs() < 1e-12,
                "non-zero imaginary at index {i}: rec.im={}",
                rec.im
            );
        }
    }

    /// FFT linearity: FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)
    #[test]
    fn test_fft_linearity() {
        let n = 8usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| (n - i) as f64).collect();
        let a = 2.0f64;
        let b = -1.0f64;

        let combined: Vec<f64> = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| a * xi + b * yi)
            .collect();

        let fft_x = fft(&x, None).expect("fft_x failed");
        let fft_y = fft(&y, None).expect("fft_y failed");
        let fft_combined = fft(&combined, None).expect("fft_combined failed");

        for k in 0..n {
            let expected_re = a * fft_x[k].re + b * fft_y[k].re;
            let expected_im = a * fft_x[k].im + b * fft_y[k].im;
            assert_abs_diff_eq!(fft_combined[k].re, expected_re, epsilon = 1e-10);
            assert_abs_diff_eq!(fft_combined[k].im, expected_im, epsilon = 1e-10);
        }
    }
}

// =================== Signal Processing Validation ===================

#[cfg(test)]
mod signal_validation {
    use scirs2_fft::fft;

    /// A signal composed of two sinusoids has positive energy
    #[test]
    fn test_mixed_signal_energy_positive() {
        let n = 256usize;
        let fs = 1000.0f64;
        let low_freq = 50.0f64;
        let high_freq = 400.0f64;

        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * std::f64::consts::PI * low_freq * t).sin()
                    + (2.0 * std::f64::consts::PI * high_freq * t).sin()
            })
            .collect();

        let energy: f64 = signal.iter().map(|&x| x * x).sum::<f64>() / n as f64;
        assert!(
            energy > 0.5,
            "Signal energy should be positive, got {energy}"
        );
    }

    /// FFT of a dual-tone signal has energy at both frequency bins
    #[test]
    fn test_dual_tone_fft_spectrum() {
        let n = 128usize;
        let k1 = 4usize;
        let k2 = 16usize;

        let signal: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * k1 as f64 * i as f64 / n as f64).cos()
                    + (2.0 * std::f64::consts::PI * k2 as f64 * i as f64 / n as f64).cos()
            })
            .collect();

        let spectrum = fft(&signal, None).expect("dual tone fft failed");

        // Bins k1 and k2 should have large magnitude (~n/2 each)
        let mag_k1 = spectrum[k1].norm();
        let mag_k2 = spectrum[k2].norm();
        let expected = n as f64 / 2.0;

        assert!(
            (mag_k1 - expected).abs() < 1e-8,
            "Expected magnitude {expected} at bin k1={k1}, got {mag_k1}"
        );
        assert!(
            (mag_k2 - expected).abs() < 1e-8,
            "Expected magnitude {expected} at bin k2={k2}, got {mag_k2}"
        );
    }

    /// Constant (DC) signal has zero energy at all non-DC frequency bins
    #[test]
    fn test_dc_signal_spectrum() {
        let n = 64usize;
        let signal: Vec<f64> = vec![2.5f64; n];
        let spectrum = fft(&signal, None).expect("dc signal fft failed");

        for (k, s) in spectrum.iter().enumerate().skip(1) {
            assert!(
                s.norm() < 1e-10,
                "DC signal has non-zero spectrum at k={k}: {}",
                s.norm()
            );
        }
    }
}

// =================== Descriptive Statistics Validation ===================

#[cfg(test)]
mod descriptive_stats_validation {
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;
    use scirs2_stats::{mean, median, var};

    /// Mean of [1, 2, 3, 4, 5] = 3.0
    #[test]
    fn test_mean_arithmetic() {
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let m = mean(&data.view()).expect("mean failed");
        assert_abs_diff_eq!(m, 3.0, epsilon = 1e-14);
    }

    /// Mean is linear: mean(a*x + b) = a*mean(x) + b
    #[test]
    fn test_mean_linearity() {
        let data = array![1.0f64, 3.0, 5.0, 7.0, 9.0];
        let a = 2.0f64;
        let b = -1.0f64;
        let shifted: scirs2_core::ndarray::Array1<f64> = data.mapv(|x| a * x + b);
        let mean_data = mean(&data.view()).expect("mean data failed");
        let mean_shifted = mean(&shifted.view()).expect("mean shifted failed");
        let expected = a * mean_data + b;
        assert_abs_diff_eq!(mean_shifted, expected, epsilon = 1e-12);
    }

    /// Median of [1, 2, 3, 4, 5] = 3.0 (odd count)
    #[test]
    fn test_median_odd_count() {
        let data = array![5.0f64, 1.0, 3.0, 2.0, 4.0];
        let med = median(&data.view()).expect("median failed");
        assert_abs_diff_eq!(med, 3.0, epsilon = 1e-14);
    }

    /// Population variance of [1,2,3,4,5] = 2.0
    #[test]
    fn test_variance_known_value() {
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
        // ddof=0 => population variance
        let v = var(&data.view(), 0, None).expect("var failed");
        assert_abs_diff_eq!(v, 2.0, epsilon = 1e-12);
    }

    /// Variance is non-negative for any input
    #[test]
    fn test_variance_non_negative() {
        let data = array![3.1f64, 3.1, 3.1, 3.1, 3.1];
        let v = var(&data.view(), 1, None).expect("var non-negative failed");
        assert!(v >= 0.0, "Variance should be non-negative, got {v}");
    }
}
