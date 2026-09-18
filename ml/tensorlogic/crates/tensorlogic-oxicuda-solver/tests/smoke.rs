//! Smoke tests for tensorlogic-oxicuda-solver — CPU path, no GPU required.
//!
//! These run in CI with the default `cpu` feature only.

use tensorlogic_oxicuda_solver::{cg_solve, solve_cholesky, solve_lu, solve_qr_lstsq};

// ---------------------------------------------------------------------------
// LU tests
// ---------------------------------------------------------------------------

#[test]
fn solve_lu_identity_3x3() {
    // I * x = [1,2,3] → x = [1,2,3]
    let a = vec![1f32, 0., 0., 0., 1., 0., 0., 0., 1.];
    let b = vec![1f32, 2., 3.];
    let x = solve_lu(&a, 3, &b).unwrap();
    for (xi, bi) in x.iter().zip(b.iter()) {
        assert!((xi - bi).abs() < 1e-5, "expected {bi} got {xi}");
    }
}

#[test]
fn solve_lu_known_2x2() {
    // A = [[3, 1],[1, 2]], b = [9, 8] → x = [2, 3]
    let a = vec![3f32, 1., 1., 2.];
    let b = vec![9f32, 8.];
    let x = solve_lu(&a, 2, &b).unwrap();
    assert!((x[0] - 2.0).abs() < 1e-4, "x[0]={}", x[0]);
    assert!((x[1] - 3.0).abs() < 1e-4, "x[1]={}", x[1]);
}

#[test]
fn solve_lu_residual_4x4() {
    // Non-trivial 4×4 — verify A*x ≈ b rather than a known exact solution.
    #[rustfmt::skip]
    let a = vec![
        10f32, 2.,  1.,  0.,
         2., 8.,  3.,  1.,
         1., 3., 12.,  2.,
         0., 1.,  2.,  6.,
    ];
    let b = vec![13f32, 14., 18., 9.];
    let x = solve_lu(&a, 4, &b).unwrap();

    // Compute residual r = A*x - b
    let residual: Vec<f32> = (0..4)
        .map(|i| {
            let ax_i: f32 = (0..4).map(|j| a[i * 4 + j] * x[j]).sum();
            ax_i - b[i]
        })
        .collect();
    let max_err = residual.iter().map(|r| r.abs()).fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-4,
        "max residual={max_err:.2e}, residual={residual:?}"
    );
}

#[test]
fn solve_lu_singular_error() {
    // Rank-deficient 2×2: rows are proportional.
    let a = vec![1f32, 2., 2., 4.];
    let b = vec![1f32, 2.];
    assert!(
        solve_lu(&a, 2, &b).is_err(),
        "expected error for singular matrix"
    );
}

#[test]
fn solve_lu_dim_mismatch_error() {
    // b has wrong length
    let a = vec![1f32, 0., 0., 1.];
    let b = vec![1f32, 2., 3.]; // length 3, not 2
    assert!(solve_lu(&a, 2, &b).is_err(), "expected DimMismatch error");
}

// ---------------------------------------------------------------------------
// Cholesky tests
// ---------------------------------------------------------------------------

#[test]
fn solve_cholesky_spd_2x2() {
    // A = [[4,2],[2,3]] (SPD), b = [6,5] → x = [1,1]
    let a = vec![4f32, 2., 2., 3.];
    let b = vec![6f32, 5.];
    let x = solve_cholesky(&a, 2, &b).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-4, "x[0]={}", x[0]);
    assert!((x[1] - 1.0).abs() < 1e-4, "x[1]={}", x[1]);
}

#[test]
fn solve_cholesky_identity_3x3() {
    let a = vec![1f32, 0., 0., 0., 1., 0., 0., 0., 1.];
    let b = vec![5f32, -2., 7.];
    let x = solve_cholesky(&a, 3, &b).unwrap();
    for (xi, bi) in x.iter().zip(b.iter()) {
        assert!((xi - bi).abs() < 1e-5, "expected {bi} got {xi}");
    }
}

#[test]
fn solve_cholesky_larger_spd() {
    // Diagonally dominant 4×4 (guaranteed SPD): D = diag(4,4,4,4), off-diag = 1
    #[rustfmt::skip]
    let a = vec![
        4f32, 1., 0., 0.,
        1., 4., 1., 0.,
        0., 1., 4., 1.,
        0., 0., 1., 4.,
    ];
    let b = vec![5f32, 6., 6., 5.];
    let x = solve_cholesky(&a, 4, &b).unwrap();
    // verify residual
    let residual_max: f32 = (0..4)
        .map(|i| {
            let ax_i: f32 = (0..4).map(|j| a[i * 4 + j] * x[j]).sum();
            (ax_i - b[i]).abs()
        })
        .fold(0.0f32, f32::max);
    assert!(residual_max < 1e-4, "max residual={residual_max:.2e}");
}

#[test]
fn solve_cholesky_non_spd_returns_error() {
    // [[1,2],[2,1]] → eigenvalues 3 and -1 → not SPD
    let a = vec![1f32, 2., 2., 1.];
    let b = vec![3f32, 3.];
    assert!(
        solve_cholesky(&a, 2, &b).is_err(),
        "expected error for non-SPD matrix"
    );
}

// ---------------------------------------------------------------------------
// QR least-squares tests
// ---------------------------------------------------------------------------

#[test]
fn solve_qr_lstsq_square_2x2() {
    let a = vec![4f32, 2., 2., 3.];
    let b = vec![6f32, 5.];
    let x = solve_qr_lstsq(&a, 2, 2, &b).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-4, "x[0]={}", x[0]);
    assert!((x[1] - 1.0).abs() < 1e-4, "x[1]={}", x[1]);
}

#[test]
fn solve_qr_lstsq_overdetermined_exact() {
    // A (3×2) = [[1,0],[0,1],[1,1]], b = [1,1,2] → x = [1,1] exactly
    let a = vec![1f32, 0., 0., 1., 1., 1.];
    let b = vec![1f32, 1., 2.];
    let x = solve_qr_lstsq(&a, 3, 2, &b).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-4, "x[0]={}", x[0]);
    assert!((x[1] - 1.0).abs() < 1e-4, "x[1]={}", x[1]);
}

#[test]
fn solve_qr_lstsq_overdetermined_approx() {
    // Overdetermined 4×2 system with an approximate (least-squares) solution.
    // A = [[1,1],[1,2],[1,3],[1,4]], b = [2,3,4,5]
    // The exact LS solution is x = [1, 1] (y-intercept=1, slope=1).
    let a = vec![1f32, 1., 1., 2., 1., 3., 1., 4.];
    let b = vec![2f32, 3., 4., 5.];
    let x = solve_qr_lstsq(&a, 4, 2, &b).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-3, "x[0]={}", x[0]);
    assert!((x[1] - 1.0).abs() < 1e-3, "x[1]={}", x[1]);
}

// ---------------------------------------------------------------------------
// CG tests
// ---------------------------------------------------------------------------

#[test]
fn cg_solve_spd_2x2() {
    // Same 2×2 SPD system
    let a = vec![4f32, 2., 2., 3.];
    let b = vec![6f32, 5.];
    let x = cg_solve(&a, 2, &b, 100, 1e-6).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-4, "x[0]={}", x[0]);
    assert!((x[1] - 1.0).abs() < 1e-4, "x[1]={}", x[1]);
}

#[test]
fn cg_solve_identity() {
    let a = vec![1f32, 0., 0., 0., 1., 0., 0., 0., 1.];
    let b = vec![3f32, 7., -2.];
    let x = cg_solve(&a, 3, &b, 20, 1e-6).unwrap();
    for (xi, bi) in x.iter().zip(b.iter()) {
        assert!((xi - bi).abs() < 1e-4, "expected {bi} got {xi}");
    }
}

#[test]
fn cg_solve_diagonal_spd() {
    // diag(2,4,6) * x = [2,4,6] → x = [1,1,1]
    #[rustfmt::skip]
    let a = vec![
        2f32, 0., 0.,
        0.,  4., 0.,
        0.,  0., 6.,
    ];
    let b = vec![2f32, 4., 6.];
    let x = cg_solve(&a, 3, &b, 30, 1e-6).unwrap();
    for (xi, bi) in x.iter().zip([1.0f32, 1.0, 1.0].iter()) {
        assert!((xi - bi).abs() < 1e-4, "expected {bi} got {xi}");
    }
}

#[test]
fn cg_solve_max_iter_zero_fails() {
    let a = vec![4f32, 2., 2., 3.];
    let b = vec![6f32, 5.];
    // With 0 iterations the algorithm cannot converge for a non-trivial RHS.
    assert!(
        cg_solve(&a, 2, &b, 0, 1e-6).is_err(),
        "expected DidNotConverge"
    );
}
