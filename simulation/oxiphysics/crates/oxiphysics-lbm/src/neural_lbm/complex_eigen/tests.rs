// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests of the dense complex eigensolver and least-squares routine.

use super::*;
use oxiphysics_core::numerical_methods::symmetric_eigen_n;
use std::f64::consts::PI;

/// Deterministic pseudo-random numbers in `[-1, 1)`.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn random_matrix(n: usize, seed: u64, complex: bool) -> ComplexMatrix {
    let mut state = seed;
    let mut a = ComplexMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            let re = lcg(&mut state);
            let im = if complex { lcg(&mut state) } else { 0.0 };
            a[(i, j)] = Complex::new(re, im);
        }
    }
    a
}

fn real_matrix(n: usize, entries: &[f64]) -> ComplexMatrix {
    ComplexMatrix::from_real_row_major(n, entries).unwrap()
}

fn mat_vec(a: &ComplexMatrix, v: &[Complex]) -> Vec<Complex> {
    (0..a.dim())
        .map(|i| (0..a.dim()).fold(Complex::zero(), |acc, j| acc + a[(i, j)] * v[j]))
        .collect()
}

/// Largest `‖A v_k − λ_k v_k‖₂` over the computed eigenpairs.
fn max_residual(a: &ComplexMatrix, e: &EigenDecomposition) -> f64 {
    let mut worst = 0.0_f64;
    for (&lambda, v) in e.values.iter().zip(e.vectors.iter()) {
        let av = mat_vec(a, v);
        let diff: Vec<Complex> = av
            .iter()
            .zip(v.iter())
            .map(|(&x, &y)| x - lambda * y)
            .collect();
        worst = worst.max(vector_norm(&diff));
    }
    worst
}

/// Every expected eigenvalue is matched by a distinct computed one within `tol`.
fn assert_same_spectrum(got: &[Complex], expected: &[Complex], tol: f64) {
    assert_eq!(got.len(), expected.len(), "spectrum size");
    let mut used = vec![false; got.len()];
    for &want in expected {
        let best = got
            .iter()
            .enumerate()
            .filter(|(k, _)| !used[*k])
            .min_by(|(_, a), (_, b)| modulus(**a - want).total_cmp(&modulus(**b - want)))
            .map(|(k, _)| k);
        let k = best.expect("an unmatched eigenvalue must be left");
        assert!(
            modulus(got[k] - want) <= tol,
            "eigenvalue {want:?} not found, closest {:?}",
            got[k]
        );
        used[k] = true;
    }
}

#[test]
fn one_by_one_matrix() {
    let mut a = ComplexMatrix::zeros(1);
    a[(0, 0)] = Complex::new(3.0, -2.0);
    let e = eigen_decomposition(&a).unwrap();
    assert_eq!(e.values, vec![Complex::new(3.0, -2.0)]);
    assert_eq!(e.vectors, vec![vec![Complex::one()]]);
}

#[test]
fn rotation_block_has_unit_modulus_eigenvalues() {
    let theta: f64 = 0.7;
    let (c, s) = (theta.cos(), theta.sin());
    let a = real_matrix(2, &[c, -s, s, c]);
    let e = eigen_decomposition(&a).unwrap();
    let expected = [Complex::new(c, s), Complex::new(c, -s)];
    assert_same_spectrum(&e.values, &expected, 1e-14);
    assert!(max_residual(&a, &e) < 1e-14);
    // Conjugate pairs are listed with the positive imaginary part first.
    assert!(e.values[0].im > 0.0 && e.values[1].im < 0.0);
}

#[test]
fn damped_rotation_spectrum_is_rho_exp_plus_minus_i_theta() {
    let (rho, theta): (f64, f64) = (0.93, 0.35);
    let (c, s) = (rho * theta.cos(), rho * theta.sin());
    let a = real_matrix(2, &[c, -s, s, c]);
    let e = eigen_decomposition(&a).unwrap();
    let expected = [Complex::new(c, s), Complex::new(c, -s)];
    assert_same_spectrum(&e.values, &expected, 1e-14);
}

#[test]
fn diagonal_matrix_returns_its_diagonal_ordered_by_modulus() {
    let mut a = ComplexMatrix::zeros(4);
    for (k, d) in [0.5, -2.0, 1.0, 0.0].into_iter().enumerate() {
        a[(k, k)] = Complex::new(d, 0.0);
    }
    let e = eigen_decomposition(&a).unwrap();
    let moduli: Vec<f64> = e.values.iter().map(|&z| modulus(z)).collect();
    assert_eq!(moduli, vec![2.0, 1.0, 0.5, 0.0]);
    assert_eq!(e.values[0], Complex::new(-2.0, 0.0));
    // The eigenvector of the eigenvalue d_k is the unit vector e_k.
    for (value, v) in e.values.iter().zip(e.vectors.iter()) {
        let expected_index = [0.5, -2.0, 1.0, 0.0]
            .iter()
            .position(|&d| d == value.re)
            .unwrap();
        for (i, &component) in v.iter().enumerate() {
            let want = if i == expected_index { 1.0 } else { 0.0 };
            assert!((component.re - want).abs() < 1e-15 && component.im.abs() < 1e-15);
        }
    }
}

#[test]
fn upper_triangular_matrix_has_its_diagonal_as_spectrum() {
    #[rustfmt::skip]
    let a = real_matrix(4, &[
        1.5, 2.0, -1.0, 0.5,
        0.0, -0.7, 3.0, 1.0,
        0.0, 0.0, 2.5, -2.0,
        0.0, 0.0, 0.0, 0.1,
    ]);
    let e = eigen_decomposition(&a).unwrap();
    let diagonal = [1.5, -0.7, 2.5, 0.1].map(|d| Complex::new(d, 0.0));
    assert_same_spectrum(&e.values, &diagonal, 1e-14);
    assert!(max_residual(&a, &e) < 1e-13);
}

#[test]
fn random_real_matrices_have_small_residuals_and_the_right_trace() {
    for n in 1..=16 {
        for seed in [1_u64, 77, 12345] {
            let a = random_matrix(n, seed.wrapping_mul(n as u64 + 3), false);
            let e = eigen_decomposition(&a).unwrap();
            assert_eq!(e.values.len(), n);
            let residual = max_residual(&a, &e);
            assert!(
                residual < 1e-11,
                "n = {n}, seed = {seed}: residual {residual}"
            );
            let trace = (0..n).fold(Complex::zero(), |acc, i| acc + a[(i, i)]);
            let sum = e.values.iter().fold(Complex::zero(), |acc, &z| acc + z);
            assert!(
                modulus(sum - trace) < 1e-11,
                "n = {n}: trace {sum:?} vs {trace:?}"
            );
        }
    }
}

#[test]
fn random_complex_matrices_have_small_residuals() {
    for n in [2, 3, 5, 8, 13] {
        let a = random_matrix(n, 4242 + n as u64, true);
        let e = eigen_decomposition(&a).unwrap();
        let residual = max_residual(&a, &e);
        assert!(residual < 1e-11, "n = {n}: residual {residual}");
        let trace = (0..n).fold(Complex::zero(), |acc, i| acc + a[(i, i)]);
        let sum = e.values.iter().fold(Complex::zero(), |acc, &z| acc + z);
        assert!(modulus(sum - trace) < 1e-11);
    }
}

#[test]
fn a_larger_dense_matrix_converges() {
    let n = 40;
    let a = random_matrix(n, 99, false);
    let e = eigen_decomposition(&a).unwrap();
    assert!(max_residual(&a, &e) < 1e-10);
}

#[test]
fn symmetric_matrices_agree_with_the_symmetric_solver() {
    let n = 7;
    let raw = random_matrix(n, 2024, false);
    let mut rows = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            rows[i][j] = 0.5 * (raw[(i, j)].re + raw[(j, i)].re);
        }
    }
    let (reference, _) = symmetric_eigen_n(&rows);
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    let a = real_matrix(n, &flat);
    let e = eigen_decomposition(&a).unwrap();
    let mut values: Vec<Complex> = e.values.clone();
    values.sort_by(|p, q| p.re.total_cmp(&q.re));
    for (got, want) in values.iter().zip(reference.iter()) {
        assert!(got.im.abs() < 1e-12, "imaginary part {}", got.im);
        assert!((got.re - want).abs() < 1e-12, "{} vs {want}", got.re);
    }
    // Symmetric input: the eigenvectors are mutually orthogonal.
    for p in 0..n {
        for q in p + 1..n {
            if (e.values[p].re - e.values[q].re).abs() > 1e-6 {
                assert!(modulus(inner(&e.vectors[p], &e.vectors[q])) < 1e-10);
            }
        }
    }
}

#[test]
fn cyclic_permutation_matrices_are_solved() {
    // The companion matrices of x^n - 1 (periodic translation by one cell): the
    // plain Wilkinson shift is exactly zero for them and stalls.
    for n in 2..=12 {
        let mut a = ComplexMatrix::zeros(n);
        for j in 0..n {
            a[((j + 1) % n, j)] = Complex::one();
        }
        let e = eigen_decomposition(&a).unwrap();
        let roots: Vec<Complex> = (0..n)
            .map(|k| Complex::from_polar(1.0, 2.0 * PI * k as f64 / n as f64))
            .collect();
        assert_same_spectrum(&e.values, &roots, 1e-12);
        assert!(max_residual(&a, &e) < 1e-12, "n = {n}");
    }
}

#[test]
fn companion_matrix_recovers_known_polynomial_roots() {
    let roots = [
        Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(-3.0, 0.0),
        Complex::new(0.5, 0.8),
        Complex::new(0.5, -0.8),
    ];
    let n = roots.len();
    // Monic polynomial prod (x - r): coefficients c[0] + c[1] x + ... + x^n.
    let mut coefficients = vec![Complex::one()];
    for &r in &roots {
        let mut next = vec![Complex::zero(); coefficients.len() + 1];
        for (k, &c) in coefficients.iter().enumerate() {
            next[k + 1] = next[k + 1] + c;
            next[k] = next[k] - c * r;
        }
        coefficients = next;
    }
    let mut a = ComplexMatrix::zeros(n);
    for i in 1..n {
        a[(i, i - 1)] = Complex::one();
    }
    for i in 0..n {
        a[(i, n - 1)] = -coefficients[i];
    }
    let e = eigen_decomposition(&a).unwrap();
    assert_same_spectrum(&e.values, &roots, 1e-9);
}

#[test]
fn defective_jordan_block_has_a_double_spectrum_and_small_residuals() {
    #[rustfmt::skip]
    let a = real_matrix(3, &[
        2.0, 1.0, 0.0,
        0.0, 2.0, 1.0,
        0.0, 0.0, 2.0,
    ]);
    let e = eigen_decomposition(&a).unwrap();
    for z in &e.values {
        assert_eq!(*z, Complex::new(2.0, 0.0));
    }
    assert!(max_residual(&a, &e) < 1e-12);
    for v in &e.vectors {
        assert!((vector_norm(v) - 1.0).abs() < 1e-14);
    }
}

#[test]
fn block_diagonal_rotations_are_solved_block_by_block() {
    // Two damped rotations: the Hessenberg matrix is reducible (a zero coupling
    // in the middle), so the iteration deflates the two blocks independently.
    let (rho1, theta1): (f64, f64) = (0.9, 0.4);
    let (rho2, theta2): (f64, f64) = (0.6, 1.9);
    let (c1, s1) = (rho1 * theta1.cos(), rho1 * theta1.sin());
    let (c2, s2) = (rho2 * theta2.cos(), rho2 * theta2.sin());
    #[rustfmt::skip]
    let a = real_matrix(4, &[
        c1, -s1, 0.0, 0.0,
        s1, c1, 0.0, 0.0,
        0.0, 0.0, c2, -s2,
        0.0, 0.0, s2, c2,
    ]);
    let e = eigen_decomposition(&a).unwrap();
    let expected = [
        Complex::new(c1, s1),
        Complex::new(c1, -s1),
        Complex::new(c2, s2),
        Complex::new(c2, -s2),
    ];
    assert_same_spectrum(&e.values, &expected, 1e-14);
    assert!(max_residual(&a, &e) < 1e-14);
}

#[test]
fn a_repeated_but_diagonalizable_eigenvalue_gets_independent_eigenvectors() {
    #[rustfmt::skip]
    let a = real_matrix(3, &[
        2.0, 0.0, 1.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 3.0,
    ]);
    let e = eigen_decomposition(&a).unwrap();
    assert!(max_residual(&a, &e) < 1e-14);
    let doubles: Vec<&Vec<Complex>> = e
        .values
        .iter()
        .zip(e.vectors.iter())
        .filter(|(value, _)| (value.re - 2.0).abs() < 1e-12)
        .map(|(_, vector)| vector)
        .collect();
    assert_eq!(doubles.len(), 2);
    let overlap = modulus(inner(doubles[0], doubles[1]));
    assert!(
        overlap < 1e-12,
        "the two eigenvectors of lambda = 2 coincide: {overlap}"
    );
}

#[test]
fn zero_and_identity_matrices() {
    let zero = ComplexMatrix::zeros(3);
    let e = eigen_decomposition(&zero).unwrap();
    assert!(e.values.iter().all(|&z| z == Complex::zero()));
    let identity = ComplexMatrix::identity(4);
    let e = eigen_decomposition(&identity).unwrap();
    assert!(e.values.iter().all(|&z| z == Complex::one()));
    assert!(max_residual(&identity, &e) < 1e-15);
    // The empty matrix has an empty spectrum.
    let e = eigen_decomposition(&ComplexMatrix::zeros(0)).unwrap();
    assert!(e.values.is_empty() && e.vectors.is_empty());
}

#[test]
fn non_finite_entries_are_rejected() {
    let mut a = ComplexMatrix::identity(3);
    a[(1, 2)] = Complex::new(f64::NAN, 0.0);
    assert!(matches!(
        eigen_decomposition(&a),
        Err(Error::NonFiniteField { index: 5, .. })
    ));
    let mut b = ComplexMatrix::identity(2);
    b[(0, 1)] = Complex::new(0.0, f64::INFINITY);
    assert!(eigen_decomposition(&b).is_err());
    assert!(ComplexMatrix::from_real_row_major(2, &[1.0, 2.0, 3.0]).is_err());
}

#[test]
fn eigenvectors_are_unit_vectors_with_a_real_positive_dominant_component() {
    let a = random_matrix(9, 31337, false);
    let e = eigen_decomposition(&a).unwrap();
    for v in &e.vectors {
        assert!((vector_norm(v) - 1.0).abs() < 1e-13);
        let dominant = v
            .iter()
            .copied()
            .max_by(|p, q| modulus(*p).total_cmp(&modulus(*q)))
            .unwrap();
        assert!(dominant.re > 0.0 && dominant.im.abs() < 1e-13 * dominant.re);
    }
}

#[test]
fn real_matrices_have_conjugate_eigenpairs() {
    let a = random_matrix(8, 555, false);
    let e = eigen_decomposition(&a).unwrap();
    for (k, &lambda) in e.values.iter().enumerate() {
        if lambda.im.abs() < 1e-8 {
            continue;
        }
        let partner = e
            .values
            .iter()
            .position(|&mu| modulus(mu - lambda.conj()) < 1e-9)
            .expect("conjugate eigenvalue");
        for (&x, &y) in e.vectors[k].iter().zip(e.vectors[partner].iter()) {
            assert!(
                modulus(x.conj() - y) < 1e-8,
                "eigenvectors are not conjugate"
            );
        }
    }
}

#[test]
fn the_spectrum_scales_with_the_matrix() {
    let base = random_matrix(6, 8, false);
    let reference = eigen_decomposition(&base).unwrap();
    for factor in [1.0e-100, 1.0e100] {
        let mut scaled = base.clone();
        for i in 0..6 {
            for j in 0..6 {
                scaled[(i, j)] = scale(base[(i, j)], factor);
            }
        }
        let e = eigen_decomposition(&scaled).unwrap();
        let expected: Vec<Complex> = reference.values.iter().map(|&z| scale(z, factor)).collect();
        assert_same_spectrum(&e.values, &expected, 1e-10 * factor);
    }
}

// ---------------------------------------------------------------------------
// Least squares
// ---------------------------------------------------------------------------

fn random_columns(rows: usize, cols: usize, seed: u64) -> Vec<Vec<Complex>> {
    let mut state = seed;
    (0..cols)
        .map(|_| {
            (0..rows)
                .map(|_| Complex::new(lcg(&mut state), lcg(&mut state)))
                .collect()
        })
        .collect()
}

fn combine(columns: &[Vec<Complex>], coefficients: &[Complex]) -> Vec<Complex> {
    let rows = columns[0].len();
    let mut out = vec![Complex::zero(); rows];
    for (column, &b) in columns.iter().zip(coefficients.iter()) {
        for (slot, &entry) in out.iter_mut().zip(column.iter()) {
            *slot = *slot + b * entry;
        }
    }
    out
}

#[test]
fn least_squares_solves_a_square_system() {
    let columns = random_columns(5, 5, 11);
    let truth = [
        Complex::new(1.0, -2.0),
        Complex::new(-0.5, 0.25),
        Complex::new(3.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(2.0, 2.0),
    ];
    let rhs = combine(&columns, &truth);
    let solution = least_squares(&columns, &rhs, 1e-10);
    for (got, want) in solution.iter().zip(truth.iter()) {
        assert!(modulus(*got - *want) < 1e-11, "{got:?} vs {want:?}");
    }
}

#[test]
fn least_squares_residual_is_orthogonal_to_the_columns() {
    let columns = random_columns(12, 4, 21);
    let mut state = 5_u64;
    let rhs: Vec<Complex> = (0..12)
        .map(|_| Complex::new(lcg(&mut state), lcg(&mut state)))
        .collect();
    let solution = least_squares(&columns, &rhs, 1e-10);
    let fitted = combine(&columns, &solution);
    let residual: Vec<Complex> = rhs
        .iter()
        .zip(fitted.iter())
        .map(|(&r, &f)| r - f)
        .collect();
    for column in &columns {
        assert!(modulus(inner(column, &residual)) < 1e-12);
    }
}

#[test]
fn dependent_and_zero_columns_receive_zero_coefficients() {
    let base = random_columns(6, 3, 3);
    let zero = vec![Complex::zero(); 6];
    // Columns: a, zero, b, a again (dependent), c.
    let columns = vec![
        base[0].clone(),
        zero,
        base[1].clone(),
        base[0].clone(),
        base[2].clone(),
    ];
    let truth = [
        Complex::new(1.0, 1.0),
        Complex::new(-2.0, 0.5),
        Complex::new(0.0, 3.0),
    ];
    let rhs = combine(&base, &truth);
    let solution = least_squares(&columns, &rhs, 1e-10);
    assert_eq!(solution[1], Complex::zero(), "zero column");
    assert_eq!(solution[3], Complex::zero(), "dependent duplicate");
    let fitted = combine(&columns, &solution);
    let error: Vec<Complex> = rhs
        .iter()
        .zip(fitted.iter())
        .map(|(&r, &f)| r - f)
        .collect();
    assert!(vector_norm(&error) < 1e-11);
}

#[test]
fn least_squares_without_columns_or_with_zero_rhs() {
    assert!(least_squares(&[], &[Complex::one()], 1e-10).is_empty());
    let columns = random_columns(4, 2, 1);
    let solution = least_squares(&columns, &[Complex::zero(); 4], 1e-10);
    assert!(solution.iter().all(|z| modulus(*z) < 1e-15));
}
