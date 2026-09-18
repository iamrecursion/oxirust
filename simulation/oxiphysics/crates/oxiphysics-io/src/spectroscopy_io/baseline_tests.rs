// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the asymmetric-least-squares baseline (second-difference
//! penalty, banded solver) and the moving average.

use super::types::SpectralAnalysis;

/// Deterministic xorshift generator (no dev-dependency needed).
struct XorShift(u64);

impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform value in `[-1, 1)`.
    fn symmetric(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

/// Dense Gaussian elimination with partial pivoting (reference solver).
fn dense_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| a[r1][col].abs().total_cmp(&a[r2][col].abs()))
            .expect("non-empty range");
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            for k in col..n {
                let subtract = factor * a[col][k];
                a[row][k] -= subtract;
            }
            let subtract = factor * b[col];
            b[row] -= subtract;
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for (k, xk) in x.iter().enumerate().skip(row + 1) {
            sum -= a[row][k] * xk;
        }
        x[row] = sum / a[row][row];
    }
    x
}

/// Dense matrix of the pentadiagonal bands.
fn dense_from_bands(main: &[f64], off1: &[f64], off2: &[f64]) -> Vec<Vec<f64>> {
    let n = main.len();
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        a[i][i] = main[i];
        if i + 1 < n {
            a[i][i + 1] = off1[i];
            a[i + 1][i] = off1[i];
        }
        if i + 2 < n {
            a[i][i + 2] = off2[i];
            a[i + 2][i] = off2[i];
        }
    }
    a
}

// --- penalty matrix ------------------------------------------------------------

#[test]
fn penalty_bands_match_the_dense_second_difference_product_for_every_size() {
    for n in 3..=12 {
        let (main, off1, off2) = SpectralAnalysis::second_difference_penalty(n);
        assert_eq!((main.len(), off1.len(), off2.len()), (n, n - 1, n - 2));
        // D2'D2 computed densely from the (n - 2) x n operator.
        let mut expected = vec![vec![0.0; n]; n];
        for row in 0..n - 2 {
            let stencil = [(row, 1.0), (row + 1, -2.0), (row + 2, 1.0)];
            for &(ci, vi) in &stencil {
                for &(cj, vj) in &stencil {
                    expected[ci][cj] += vi * vj;
                }
            }
        }
        assert_eq!(dense_from_bands(&main, &off1, &off2), expected, "n = {n}");
    }
}

#[test]
fn penalty_bands_have_the_textbook_values() {
    let (main, off1, off2) = SpectralAnalysis::second_difference_penalty(3);
    assert_eq!(main, vec![1.0, 4.0, 1.0]);
    assert_eq!(off1, vec![-2.0, -2.0]);
    assert_eq!(off2, vec![1.0]);

    let (main, off1, off2) = SpectralAnalysis::second_difference_penalty(7);
    assert_eq!(main, vec![1.0, 5.0, 6.0, 6.0, 6.0, 5.0, 1.0]);
    assert_eq!(off1, vec![-2.0, -4.0, -4.0, -4.0, -4.0, -2.0]);
    assert_eq!(off2, vec![1.0; 5]);
}

// --- banded solver -------------------------------------------------------------

#[test]
fn pentadiagonal_solver_matches_dense_elimination_on_random_spd_systems() {
    let mut rng = XorShift(0x2545_F491_4F6C_DD1D);
    for &n in &[1usize, 2, 3, 4, 5, 6, 17, 64] {
        let off1: Vec<f64> = (0..n.saturating_sub(1)).map(|_| rng.symmetric()).collect();
        let off2: Vec<f64> = (0..n.saturating_sub(2)).map(|_| rng.symmetric()).collect();
        // Strict diagonal dominance guarantees a symmetric positive-definite matrix.
        let main: Vec<f64> = (0..n)
            .map(|i| {
                let mut row_sum = 1.0 + rng.symmetric().abs();
                if i >= 1 {
                    row_sum += off1[i - 1].abs();
                }
                if i < off1.len() {
                    row_sum += off1[i].abs();
                }
                if i >= 2 {
                    row_sum += off2[i - 2].abs();
                }
                if i < off2.len() {
                    row_sum += off2[i].abs();
                }
                row_sum
            })
            .collect();
        let rhs: Vec<f64> = (0..n).map(|_| 10.0 * rng.symmetric()).collect();
        let expected = dense_solve(dense_from_bands(&main, &off1, &off2), rhs.clone());
        let solved = SpectralAnalysis::pentadiagonal_solve(&main, &off1, &off2, &rhs)
            .expect("diagonally dominant system is positive definite");
        for (got, want) in solved.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-10, "n = {n}: {got} vs {want}");
        }
    }
}

#[test]
fn pentadiagonal_solver_rejects_indefinite_and_malformed_systems() {
    // Indefinite: the second pivot is 1 - 2^2 / 1 < 0.
    assert!(
        SpectralAnalysis::pentadiagonal_solve(
            &[1.0, 1.0, 1.0],
            &[2.0, 2.0],
            &[0.0],
            &[1.0, 1.0, 1.0]
        )
        .is_none()
    );
    // NaN on the diagonal.
    assert!(
        SpectralAnalysis::pentadiagonal_solve(&[1.0, f64::NAN], &[0.0], &[], &[1.0, 1.0]).is_none()
    );
    // Band or right-hand-side lengths that do not fit.
    assert!(SpectralAnalysis::pentadiagonal_solve(&[1.0, 1.0], &[], &[], &[1.0, 1.0]).is_none());
    assert!(SpectralAnalysis::pentadiagonal_solve(&[1.0, 1.0], &[0.0], &[], &[1.0]).is_none());
    assert!(SpectralAnalysis::pentadiagonal_solve(&[], &[], &[], &[]).is_none());
}

// --- ALS baseline --------------------------------------------------------------

/// Reference output of the Eilers & Boelens algorithm (scipy sparse solve of
/// `(W + lam D2'D2) z = W y`), `lam = 10`, `p = 0.05`, 8 iterations.
#[test]
fn als_baseline_matches_a_reference_implementation() {
    let y = [1.0, 1.2, 0.9, 1.5, 3.0, 5.5, 3.2, 1.8, 1.4, 1.1, 1.0, 1.3];
    let reference_baseline = [
        0.910711169502274,
        1.0357333326974452,
        1.1612019400451048,
        1.2883847690342543,
        1.39373541284961,
        1.4547655408307174,
        1.4570181452528734,
        1.4062623906872227,
        1.316982350978646,
        1.2056307880185893,
        1.0890755519436044,
        0.9741495680284773,
    ];
    let corrected = SpectralAnalysis::als_baseline(&y, 10.0, 0.05, 8);
    assert_eq!(corrected.len(), y.len());
    for i in 0..y.len() {
        let baseline = y[i] - corrected[i];
        assert!(
            (baseline - reference_baseline[i]).abs() < 1e-9,
            "baseline[{i}] = {baseline}, reference {}",
            reference_baseline[i]
        );
    }
}

#[test]
fn als_baseline_follows_a_linear_ramp_below_a_peak() {
    let n = 500;
    let ramp: Vec<f64> = (0..n).map(|i| 0.01 * i as f64).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let d = (i as f64 - 250.0) / 10.0;
            ramp[i] + 5.0 * (-d * d).exp()
        })
        .collect();
    let corrected = SpectralAnalysis::als_baseline(&y, 1e5, 0.01, 10);
    assert_eq!(corrected.len(), n);
    let ramp_range = ramp[n - 1] - ramp[0];
    for i in 0..n {
        if (i as f64 - 250.0).abs() <= 50.0 {
            continue;
        }
        let baseline = y[i] - corrected[i];
        assert!(
            (baseline - ramp[i]).abs() < 0.01 * ramp_range,
            "baseline[{i}] = {baseline} departs from the ramp {}",
            ramp[i]
        );
    }
    // The peak itself is kept in the corrected signal (reference value 4.943).
    assert!(
        (corrected[250] - 4.943400667311371).abs() < 1e-6,
        "corrected peak = {}",
        corrected[250]
    );
}

#[test]
fn als_baseline_leaves_no_residue_on_a_pure_ramp() {
    // Straight lines lie in the null space of the second-difference penalty, so
    // a ramp is its own baseline; a first-difference penalty would flatten it
    // (residue of several units for this ramp).
    let y: Vec<f64> = (0..500).map(|i| 0.01 * i as f64).collect();
    let corrected = SpectralAnalysis::als_baseline(&y, 1e5, 0.01, 10);
    let worst = corrected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(worst < 1e-6, "largest residue {worst}");
}

#[test]
fn als_baseline_of_a_three_point_signal_is_a_straight_line() {
    let y = [1.0, 3.0, 2.0];
    let corrected = SpectralAnalysis::als_baseline(&y, 1e3, 0.01, 10);
    assert_eq!(corrected.len(), 3);
    let baseline: Vec<f64> = y.iter().zip(&corrected).map(|(a, b)| a - b).collect();
    // Reference (scipy): [1.00753767, 1.5075414, 2.00753767].
    assert!((baseline[0] - 1.00753767).abs() < 1e-6);
    assert!((baseline[1] - 1.5075414).abs() < 1e-6);
    // Nearly straight: the data's own second difference is -3, the baseline's ~ -7e-6.
    let curvature = baseline[0] - 2.0 * baseline[1] + baseline[2];
    assert!(curvature.abs() < 1e-4, "curvature {curvature}");
}

#[test]
fn als_baseline_returns_the_input_when_no_baseline_can_be_estimated() {
    assert!(SpectralAnalysis::als_baseline(&[], 1e5, 0.01, 10).is_empty());
    assert_eq!(
        SpectralAnalysis::als_baseline(&[4.0], 1e5, 0.01, 10),
        vec![4.0]
    );
    assert_eq!(
        SpectralAnalysis::als_baseline(&[4.0, 5.0], 1e5, 0.01, 10),
        vec![4.0, 5.0]
    );
    let y = [1.0, 2.0, 4.0, 3.0, 2.0];
    assert_eq!(SpectralAnalysis::als_baseline(&y, 1e5, 0.01, 0), y.to_vec());
    assert_eq!(
        SpectralAnalysis::als_baseline(&y, -1.0, 0.01, 10),
        y.to_vec()
    );
    assert_eq!(
        SpectralAnalysis::als_baseline(&y, f64::NAN, 0.01, 10),
        y.to_vec()
    );
    assert_eq!(
        SpectralAnalysis::als_baseline(&y, f64::INFINITY, 0.01, 10),
        y.to_vec()
    );
    assert_eq!(SpectralAnalysis::als_baseline(&y, 1e3, 1.5, 10), y.to_vec());
    assert_eq!(
        SpectralAnalysis::als_baseline(&y, 1e3, -0.1, 10),
        y.to_vec()
    );
    assert_eq!(
        SpectralAnalysis::als_baseline(&y, 1e3, f64::NAN, 10),
        y.to_vec()
    );
}

#[test]
fn als_baseline_with_zero_penalty_reproduces_the_signal() {
    let y = [1.0, 2.0, 4.0, 3.0, 2.0, 5.0];
    let corrected = SpectralAnalysis::als_baseline(&y, 0.0, 0.1, 5);
    for (i, c) in corrected.iter().enumerate() {
        assert!(c.abs() < 1e-12, "corrected[{i}] = {c}");
    }
}

// --- moving average ------------------------------------------------------------

fn naive_moving_average(y: &[f64], half: usize) -> Vec<f64> {
    let mut out = y.to_vec();
    if y.len() < 2 * half + 1 {
        return out;
    }
    for i in half..y.len() - half {
        let sum: f64 = y[i - half..=i + half].iter().sum();
        out[i] = sum / (2 * half + 1) as f64;
    }
    out
}

#[test]
fn moving_average_returns_short_signals_unchanged_instead_of_panicking() {
    assert!(SpectralAnalysis::moving_average(&[], 1).is_empty());
    assert!(SpectralAnalysis::moving_average(&[], 0).is_empty());
    let y = [1.0, 5.0, 2.0];
    assert_eq!(SpectralAnalysis::moving_average(&y, 5), y.to_vec());
    // One sample short of the window of five.
    let four = [1.0, 5.0, 2.0, 7.0];
    assert_eq!(SpectralAnalysis::moving_average(&four, 2), four.to_vec());
    // A window width that does not fit in usize.
    assert_eq!(SpectralAnalysis::moving_average(&y, usize::MAX), y.to_vec());
    assert_eq!(
        SpectralAnalysis::moving_average(&y, usize::MAX / 2),
        y.to_vec()
    );
}

#[test]
fn moving_average_of_a_signal_exactly_as_long_as_the_window_averages_the_middle() {
    let y = [1.0, 5.0, 3.0];
    let out = SpectralAnalysis::moving_average(&y, 1);
    assert_eq!(out, vec![1.0, 3.0, 3.0]);
}

#[test]
fn moving_average_with_an_empty_half_window_is_the_identity() {
    let y = [1.0, -2.5, 3.25, 0.0];
    assert_eq!(SpectralAnalysis::moving_average(&y, 0), y.to_vec());
}

#[test]
fn moving_average_matches_the_naive_window_sums_on_random_data() {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let y: Vec<f64> = (0..1000).map(|_| 100.0 * rng.symmetric()).collect();
    for half in [1usize, 7, 50] {
        let fast = SpectralAnalysis::moving_average(&y, half);
        let slow = naive_moving_average(&y, half);
        for (i, (a, b)) in fast.iter().zip(slow.iter()).enumerate() {
            assert!((a - b).abs() < 1e-12, "half = {half}, i = {i}: {a} vs {b}");
        }
    }
}

#[test]
fn moving_average_keeps_the_boundary_samples() {
    let y: Vec<f64> = (0..20).map(|i| (i * i) as f64).collect();
    let out = SpectralAnalysis::moving_average(&y, 3);
    assert_eq!(&out[..3], &y[..3]);
    assert_eq!(&out[17..], &y[17..]);
    // Quadratic data: the 7-point mean of i^2 is i^2 + 4.
    for i in 3..17 {
        assert!((out[i] - (y[i] + 4.0)).abs() < 1e-9, "i = {i}: {}", out[i]);
    }
}

#[test]
fn moving_average_is_not_polluted_by_a_large_value_that_left_the_window() {
    // A plain running sum would keep the rounding error of adding 0.1 to 1e16.
    let mut rng = XorShift(0xDEAD_BEEF_CAFE_F00D);
    let mut y: Vec<f64> = (0..200).map(|_| rng.symmetric().abs()).collect();
    y[0] = 1e16;
    let fast = SpectralAnalysis::moving_average(&y, 2);
    let slow = naive_moving_average(&y, 2);
    for (i, (a, b)) in fast.iter().zip(slow.iter()).enumerate() {
        // The first windows contain the 1e16 spike and carry its rounding error.
        let tolerance = if i <= 2 { 1.0 } else { 1e-12 };
        assert!((a - b).abs() <= tolerance, "i = {i}: {a} vs {b}");
    }
}

#[test]
fn moving_average_confines_non_finite_samples_to_their_windows() {
    let mut y: Vec<f64> = (0..30).map(|i| i as f64).collect();
    y[10] = f64::NAN;
    y[20] = f64::INFINITY;
    let out = SpectralAnalysis::moving_average(&y, 2);
    let naive = naive_moving_average(&y, 2);
    for i in 0..30 {
        let near_nan = (8..=12).contains(&i);
        let near_inf = (18..=22).contains(&i);
        if near_nan {
            assert!(out[i].is_nan(), "out[{i}] = {}", out[i]);
        } else if near_inf {
            assert_eq!(out[i], f64::INFINITY, "out[{i}]");
        } else {
            assert!(out[i].is_finite(), "out[{i}] = {}", out[i]);
            assert!((out[i] - naive[i]).abs() < 1e-12);
        }
    }
}
