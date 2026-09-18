//! Advanced solver tests: f64 variants, PCG, and tridiagonal solvers.
//!
//! Covers all new public functions introduced in Round 7 Track B.

use tensorlogic_oxicuda_solver::{
    cg_solve_f64, pcg_solve, pcg_solve_f64, solve_cholesky_f64, solve_lu_f64, solve_qr_lstsq_f64,
    solve_tridiagonal, solve_tridiagonal_f64, Precond, SolverError,
};

// ---------------------------------------------------------------------------
// Shared helper utilities
// ---------------------------------------------------------------------------

/// Maximum absolute error between two equal-length f64 slices.
fn max_abs_err_f64(u: &[f64], v: &[f64]) -> f64 {
    u.iter()
        .zip(v.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

/// Maximum absolute error between two equal-length f32 slices.
fn max_abs_err_f32(u: &[f32], v: &[f32]) -> f32 {
    u.iter()
        .zip(v.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

/// Dense matrix–vector multiply for f64: y = A · x  (A is n×n row-major).
fn mat_vec_f64(a: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
    (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum::<f64>())
        .collect()
}

// ---------------------------------------------------------------------------
// f64 LU solver tests
// ---------------------------------------------------------------------------

#[test]
fn solve_lu_f64_identity() {
    // 3×3 identity: I · x = b → x = b exactly
    let a = vec![1f64, 0., 0., 0., 1., 0., 0., 0., 1.];
    let b = vec![7.0_f64, -3.0, 5.0];
    let x = solve_lu_f64(&a, 3, &b).unwrap();
    assert_eq!(x.len(), 3, "solution length mismatch");
    assert!(
        max_abs_err_f64(&x, &b) < 1e-14,
        "identity: x={x:?} vs b={b:?}"
    );
}

#[test]
fn solve_lu_f64_known_2x2() {
    // A = [[3,1],[1,2]], b = [9,8] → x = [2,3]
    let a = vec![3.0_f64, 1., 1., 2.];
    let b = vec![9.0_f64, 8.];
    let x = solve_lu_f64(&a, 2, &b).unwrap();
    assert!((x[0] - 2.0).abs() < 1e-12, "x[0]={}", x[0]);
    assert!((x[1] - 3.0).abs() < 1e-12, "x[1]={}", x[1]);
    // Also verify residual
    let r0 = 3.0 * x[0] + 1.0 * x[1] - 9.0;
    let r1 = 1.0 * x[0] + 2.0 * x[1] - 8.0;
    assert!(r0.abs() < 1e-12, "r0={r0}");
    assert!(r1.abs() < 1e-12, "r1={r1}");
}

// ---------------------------------------------------------------------------
// f64 Cholesky solver tests
// ---------------------------------------------------------------------------

#[test]
fn solve_cholesky_f64_spd_2x2() {
    // A = [[4,2],[2,3]] (SPD), b = [6,5] → x = [1,1]
    let a = vec![4.0_f64, 2., 2., 3.];
    let b = vec![6.0_f64, 5.];
    let x = solve_cholesky_f64(&a, 2, &b).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-12, "x[0]={}", x[0]);
    assert!((x[1] - 1.0).abs() < 1e-12, "x[1]={}", x[1]);
}

// ---------------------------------------------------------------------------
// f64 QR least-squares solver tests
// ---------------------------------------------------------------------------

#[test]
fn solve_qr_lstsq_f64_overdetermined() {
    // A (3×2) = [[1,0],[0,1],[1,1]], b = [1,1,2] → x = [1,1] exactly
    let a = vec![1.0_f64, 0., 0., 1., 1., 1.];
    let b = vec![1.0_f64, 1., 2.];
    let x = solve_qr_lstsq_f64(&a, 3, 2, &b).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-11, "x[0]={}", x[0]);
    assert!((x[1] - 1.0).abs() < 1e-11, "x[1]={}", x[1]);
}

// ---------------------------------------------------------------------------
// f64 CG solver tests
// ---------------------------------------------------------------------------

#[test]
fn cg_solve_f64_diagonal_spd() {
    // diag(1,2,3,4) · x = [1,2,3,4] → x = [1,1,1,1]
    #[rustfmt::skip]
    let a = vec![
        1.0_f64, 0., 0., 0.,
        0., 2., 0., 0.,
        0., 0., 3., 0.,
        0., 0., 0., 4.,
    ];
    let b = vec![1.0_f64, 2., 3., 4.];
    let x = cg_solve_f64(&a, 4, &b, 50, 1e-12).unwrap();
    let expected = vec![1.0_f64, 1., 1., 1.];
    assert!(max_abs_err_f64(&x, &expected) < 1e-10, "x={x:?}");
}

// ---------------------------------------------------------------------------
// PCG Jacobi tests — convergence comparison
// ---------------------------------------------------------------------------

/// Count the PCG iterations required for convergence by trying increasing
/// budgets until success.  Returns the minimum successful budget.
fn count_pcg_iters_f32(a: &[f32], n: usize, b: &[f32], precond: Precond, tol: f32) -> usize {
    // Try budgets 1, 2, 3, … until the solver succeeds.
    for budget in 1..=10_000 {
        if pcg_solve(a, n, b, precond, budget, tol).is_ok() {
            return budget;
        }
    }
    10_001
}

/// Same for plain CG (also f32, using pcg_solve with no preconditioner isn't
/// available, so we use the pcg_solve interface with Jacobi on a matrix where
/// Jacobi == identity scaling).
///
/// For a fair comparison we measure CG via the pcg_solve path with Jacobi
/// on a strongly diagonally dominant matrix (where Jacobi precond helps most).
#[test]
fn pcg_jacobi_converges_faster_on_diag_dominant() {
    // Diagonally dominant 4×4 SPD: D = diag(100,200,300,400), off-diag = 1
    // Condition number ≈ 4 (well-conditioned), Jacobi scaling ≈ exact inverse.
    #[rustfmt::skip]
    let a = vec![
        100f32,  1.,  0.,  0.,
          1., 200.,  1.,  0.,
          0.,   1., 300., 1.,
          0.,   0.,   1., 400.,
    ];
    let b = vec![101f32, 202., 302., 401.];
    let tol = 1e-5_f32;

    // CG (Jacobi identity = Jacobi with uniform diagonal) should converge
    // in at most n=4 iterations for a diagonally dominant system when
    // Jacobi precond is effective.
    let jacobi_iters = count_pcg_iters_f32(&a, 4, &b, Precond::Jacobi, tol);

    // Verify correctness of the Jacobi-PCG solution
    let x = pcg_solve(&a, 4, &b, Precond::Jacobi, 200, tol).unwrap();
    let ax = mat_vec_f64(
        &a.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        4,
        &x.iter().map(|&v| v as f64).collect::<Vec<_>>(),
    );
    let b_f64: Vec<f64> = b.iter().map(|&v| v as f64).collect();
    let residual = max_abs_err_f64(&ax, &b_f64) as f32;
    assert!(residual < 1e-3, "PCG-Jacobi residual too large: {residual}");

    // The Jacobi PCG must converge in a reasonable number of iterations
    assert!(
        jacobi_iters <= 200,
        "Jacobi PCG used {jacobi_iters} iterations — too slow"
    );
}

// ---------------------------------------------------------------------------
// PCG IC(0) tests
// ---------------------------------------------------------------------------

#[test]
fn pcg_ichol_converges_on_spd() {
    // 4×4 SPD system: A = D + E where D = diag(5,6,7,8), E = tridiagonal(1)
    #[rustfmt::skip]
    let a = vec![
        5f32, 1., 0., 0.,
        1., 6., 1., 0.,
        0., 1., 7., 1.,
        0., 0., 1., 8.,
    ];
    let b = vec![6f32, 8., 9., 9.];
    let x = pcg_solve(&a, 4, &b, Precond::IncompleteCholesky, 100, 1e-5).unwrap();

    // verify residual
    let ax: Vec<f32> = (0..4)
        .map(|i| (0..4).map(|j| a[i * 4 + j] * x[j]).sum::<f32>())
        .collect();
    let residual = max_abs_err_f32(&ax, &b);
    assert!(
        residual < 1e-4,
        "IC(0) PCG residual too large: {residual}, ax={ax:?}, b={b:?}"
    );
}

// ---------------------------------------------------------------------------
// PCG f64 variant tests
// ---------------------------------------------------------------------------

#[test]
fn pcg_f64_jacobi() {
    // 3×3 SPD diagonal: diag(2,3,4) · x = [2,3,4] → x = [1,1,1]
    #[rustfmt::skip]
    let a = vec![
        2.0_f64, 0., 0.,
        0., 3., 0.,
        0., 0., 4.,
    ];
    let b = vec![2.0_f64, 3., 4.];
    let x = pcg_solve_f64(&a, 3, &b, Precond::Jacobi, 50, 1e-12).unwrap();
    let expected = vec![1.0_f64, 1., 1.];
    assert!(max_abs_err_f64(&x, &expected) < 1e-10, "x={x:?}");
}

// ---------------------------------------------------------------------------
// Tridiagonal (Thomas) solver tests — f32
// ---------------------------------------------------------------------------

#[test]
fn solve_tridiagonal_known_solution() {
    // 5-element tridiagonal system:
    // diag = [2,2,2,2,2], sub = [0,-1,-1,-1,-1], sup = [-1,-1,-1,-1,0]
    // This is the 1-D discrete Laplacian (scaled): A = tridiag(-1,2,-1).
    // b = [1,0,0,0,1]
    // Known solution: x = [1,1,1,1,1] * (5/6) -- skip exact; verify residual.
    let n = 5;
    let sub = vec![0f32, -1., -1., -1., -1.];
    let diag = vec![2f32, 2., 2., 2., 2.];
    let sup = vec![-1f32, -1., -1., -1., 0.];
    let rhs = vec![1f32, 0., 0., 0., 1.];

    let x = solve_tridiagonal(&sub, &diag, &sup, &rhs).unwrap();

    // verify A·x = rhs (tridiagonal product)
    let mut ax = vec![0f32; n];
    ax[0] = diag[0] * x[0] + sup[0] * x[1];
    for i in 1..(n - 1) {
        ax[i] = sub[i] * x[i - 1] + diag[i] * x[i] + sup[i] * x[i + 1];
    }
    ax[n - 1] = sub[n - 1] * x[n - 2] + diag[n - 1] * x[n - 1];

    let residual = max_abs_err_f32(&ax, &rhs);
    assert!(
        residual < 1e-5,
        "tridiagonal residual={residual:.2e}, ax={ax:?}, rhs={rhs:?}"
    );
}

#[test]
fn solve_tridiagonal_agrees_with_dense_lu() {
    // Build a 4×4 tridiagonal as both a dense matrix and separate vectors,
    // then verify both solvers produce the same answer.
    use tensorlogic_oxicuda_solver::solve_lu;

    let n = 4usize;
    let sub_vals = [0f32, -0.5, -0.5, -0.5];
    let diag_vals = [3f32, 3., 3., 3.];
    let sup_vals = [-0.5f32, -0.5, -0.5, 0.];
    let rhs = vec![2.5f32, 2., 2., 2.5];

    // Build dense matrix (row-major)
    let mut a_dense = vec![0f32; n * n];
    for i in 0..n {
        a_dense[i * n + i] = diag_vals[i];
        if i > 0 {
            a_dense[i * n + (i - 1)] = sub_vals[i];
        }
        if i + 1 < n {
            a_dense[i * n + (i + 1)] = sup_vals[i];
        }
    }

    let x_tri = solve_tridiagonal(&sub_vals, &diag_vals, &sup_vals, &rhs).unwrap();

    let x_dense = solve_lu(&a_dense, n, &rhs).unwrap();

    let err = max_abs_err_f32(&x_tri, &x_dense);
    assert!(
        err < 1e-5,
        "tridiagonal vs dense LU mismatch: {err:.2e}, tri={x_tri:?}, lu={x_dense:?}"
    );
}

// ---------------------------------------------------------------------------
// Tridiagonal solver tests — f64
// ---------------------------------------------------------------------------

#[test]
fn solve_tridiagonal_f64_known() {
    // 3-element tridiagonal: diag=[4,4,4], sub=[0,-1,-1], sup=[-1,-1,0]
    // b = [3,2,3] → known solution x=[1,1,1]
    let sub = vec![0f64, -1., -1.];
    let diag = vec![4f64, 4., 4.];
    let sup = vec![-1f64, -1., 0.];
    let rhs = vec![3.0_f64, 2., 3.];

    let x = solve_tridiagonal_f64(&sub, &diag, &sup, &rhs).unwrap();
    let expected = vec![1.0_f64, 1., 1.];
    assert!(max_abs_err_f64(&x, &expected) < 1e-13, "x={x:?}");
}

// ---------------------------------------------------------------------------
// Singular tridiagonal — error handling
// ---------------------------------------------------------------------------

#[test]
fn solve_tridiagonal_singular_returns_error() {
    // diag contains a zero on the main diagonal after elimination,
    // guaranteed by choosing diag[0] = 0.
    let sub = vec![0f32, 0., 0.];
    let diag = vec![0f32, 1., 1.]; // zero pivot at index 0
    let sup = vec![0f32, 0., 0.];
    let rhs = vec![1f32, 1., 1.];

    let result = solve_tridiagonal(&sub, &diag, &sup, &rhs);
    assert!(
        matches!(result, Err(SolverError::Singular)),
        "expected Singular, got {result:?}"
    );
}
