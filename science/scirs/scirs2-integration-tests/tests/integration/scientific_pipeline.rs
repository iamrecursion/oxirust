// Integration tests for scientific computing pipeline:
//   scirs2_integrate (ODE) + scirs2_linalg (solve/eigh) + scirs2_sparse (CsrMatrix)

use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{array, Array1, Array2, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};
use scirs2_linalg::{eigh, solve};
use scirs2_sparse::CsrMatrix;

use crate::common::*;

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: ODE solve — dy/dt = -y, y(0) = 1  →  analytical solution y(t) = e^(-t)
// ─────────────────────────────────────────────────────────────────────────────

/// Solve a simple scalar exponential-decay ODE and verify against e^(-t).
#[test]
fn test_scientific_pipeline_ode_solve() -> TestResult<()> {
    // Define ODE: dy/dt = -y
    let f = |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { array![-y[0]] };

    let options = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-8,
        atol: 1e-10,
        max_steps: 2000,
        ..Default::default()
    };

    let result = solve_ivp(f, [0.0_f64, 5.0_f64], array![1.0_f64], Some(options))
        .map_err(|e| format!("solve_ivp failed: {}", e))?;

    assert!(
        result.success,
        "ODE solver did not succeed: {:?}",
        result.message
    );

    // Verify solution at all output time points against analytical y(t) = e^{-t}
    // Allow relative error tolerance matching rtol (1e-8) with some slack
    let tol = 1e-5;
    for (t_val, y_vec) in result.t.iter().zip(result.y.iter()) {
        let analytical = (-t_val).exp();
        let computed = y_vec[0];
        assert!(
            (computed - analytical).abs() < tol,
            "ODE solution at t={}: computed={}, analytical={}, diff={}",
            t_val,
            computed,
            analytical,
            (computed - analytical).abs()
        );
    }

    // Check final time and final value
    let t_final = result.t.last().ok_or("No time points in result")?;
    let y_final = result.y.last().ok_or("No solution points in result")?;
    let analytical_final = (-t_final).exp();

    assert_abs_diff_eq!(y_final[0], analytical_final, epsilon = tol);

    println!(
        "ODE solve: t=[0,5], {} steps, y(5)={:.8} (analytical={:.8})",
        result.n_steps, y_final[0], analytical_final
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Sparse tridiagonal system → dense solve verification
//
// Build a tridiagonal system T·x = b using CsrMatrix, compute b = T·x_true,
// then solve T·x_solve = b using scirs2_linalg::solve (dense), verify x_solve ≈ x_true.
// ─────────────────────────────────────────────────────────────────────────────

/// Build Poisson-1D tridiagonal SPD matrix: diag=2, off-diag=-1.
fn build_tridiagonal_system(n: usize) -> TestResult<(CsrMatrix<f64>, Array2<f64>)> {
    let mut row_idx: Vec<usize> = Vec::new();
    let mut col_idx: Vec<usize> = Vec::new();
    let mut data: Vec<f64> = Vec::new();

    let mut dense = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        row_idx.push(i);
        col_idx.push(i);
        data.push(2.0);
        dense[[i, i]] = 2.0;

        if i > 0 {
            row_idx.push(i);
            col_idx.push(i - 1);
            data.push(-1.0);
            dense[[i, i - 1]] = -1.0;
        }
        if i + 1 < n {
            row_idx.push(i);
            col_idx.push(i + 1);
            data.push(-1.0);
            dense[[i, i + 1]] = -1.0;
        }
    }

    let sparse = CsrMatrix::new(data, row_idx, col_idx, (n, n))
        .map_err(|e| format!("CsrMatrix::new failed: {}", e))?;

    Ok((sparse, dense))
}

/// Build the tridiagonal system, compute rhs b via sparse matvec, solve densely.
#[test]
fn test_scientific_pipeline_linear_system() -> TestResult<()> {
    let n = 20usize;
    let (sparse, dense) = build_tridiagonal_system(n)?;

    // x_true = 1, 2, 3, ..., n
    let x_true: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let x_true_arr = Array1::from_vec(x_true.clone());

    // Compute b = sparse * x_true
    let b_vec = sparse
        .dot(&x_true)
        .map_err(|e| format!("sparse matvec failed: {}", e))?;
    let b = Array1::from_vec(b_vec);

    // Solve dense system: dense * x_solve = b
    let x_solve =
        solve(&dense.view(), &b.view(), None).map_err(|e| format!("dense solve failed: {}", e))?;

    // Verify x_solve ≈ x_true
    let tol = 1e-8;
    for i in 0..n {
        assert_abs_diff_eq!(x_solve[i], x_true_arr[i], epsilon = tol);
    }

    // Compute residual norm
    let mut residual = 0.0f64;
    for i in 0..n {
        residual += (x_solve[i] - x_true_arr[i]).powi(2);
    }
    let residual_norm = residual.sqrt();

    assert!(
        residual_norm < 1e-7,
        "Residual norm too large: {}",
        residual_norm
    );

    println!(
        "Linear system pipeline: {}×{} tridiagonal, residual_norm={:.2e}",
        n, n, residual_norm
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Eigenvalues of a 5×5 symmetric tridiagonal matrix
//
// Tridiagonal with diag=2, off-diag=-1 (same structure as Poisson-1D).
// Analytical eigenvalues: λ_k = 2 - 2*cos(k*π/(n+1))  for k = 1..n
// ─────────────────────────────────────────────────────────────────────────────

/// Compute eigenvalues of a small symmetric tridiagonal matrix and compare to analytical values.
#[test]
fn test_scientific_pipeline_eigenvalue() -> TestResult<()> {
    let n = 5usize;
    let (_sparse, dense) = build_tridiagonal_system(n)?;

    // Analytical eigenvalues for the n×n Poisson tridiagonal: λ_k = 2 - 2cos(kπ/(n+1))
    let pi = std::f64::consts::PI;
    let mut expected: Vec<f64> = (1..=n)
        .map(|k| 2.0 - 2.0 * (k as f64 * pi / (n + 1) as f64).cos())
        .collect();
    expected.sort_by(|a, b| a.partial_cmp(b).expect("comparison failed"));

    // Compute eigenvalues with eigh (symmetric → real eigenvalues)
    let (eigenvalues, _eigenvectors) =
        eigh(&dense.view(), None).map_err(|e| format!("eigh failed: {}", e))?;

    // Sort computed eigenvalues ascending
    let mut computed: Vec<f64> = eigenvalues.to_vec();
    computed.sort_by(|a, b| a.partial_cmp(b).expect("comparison failed"));

    assert_eq!(
        computed.len(),
        n,
        "Expected {} eigenvalues, got {}",
        n,
        computed.len()
    );

    // All eigenvalues must be positive (SPD matrix)
    for (i, &ev) in computed.iter().enumerate() {
        assert!(ev > 0.0, "Eigenvalue {} is not positive: {}", i, ev);
    }

    // Compare to analytical values
    let tol = 1e-8;
    for (i, (&comp, &analyt)) in computed.iter().zip(expected.iter()).enumerate() {
        assert_abs_diff_eq!(comp, analyt, epsilon = tol);
        println!(
            "  λ_{}: computed={:.10}, analytical={:.10}, diff={:.2e}",
            i + 1,
            comp,
            analyt,
            (comp - analyt).abs()
        );
    }

    println!(
        "Eigenvalue pipeline: {}×{} tridiagonal, all {} eigenvalues match analytical",
        n, n, n
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Sparse-dense workflow consistency
//
// Build same matrix as sparse and dense, apply matvec with the same vector,
// compare results.
// ─────────────────────────────────────────────────────────────────────────────

/// Verify that sparse matvec and dense matrix-vector product give identical results.
#[test]
fn test_scientific_pipeline_sparse_dense_consistency() -> TestResult<()> {
    let n = 15usize;
    let (sparse, dense) = build_tridiagonal_system(n)?;

    let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();

    // Sparse matvec
    let sparse_result = sparse
        .dot(&x)
        .map_err(|e| format!("sparse matvec: {}", e))?;

    // Dense matrix-vector product
    let x_arr = Array1::from_vec(x.clone());
    let dense_result: Vec<f64> = (0..n).map(|i| dense.row(i).dot(&x_arr.view())).collect();

    // Compare element-wise
    let tol = 1e-12;
    for i in 0..n {
        assert_abs_diff_eq!(sparse_result[i], dense_result[i], epsilon = tol);
    }

    println!(
        "Sparse-dense consistency: {}×{} tridiagonal matvec verified",
        n, n
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: ODE solve — 2D coupled system for sanity
//
// y' = [[0, 1], [-1, 0]] * y  (simple harmonic oscillator)
// y(0) = [1, 0]
// Analytical: y_0(t) = cos(t), y_1(t) = -sin(t)
// ─────────────────────────────────────────────────────────────────────────────

/// Solve a 2D coupled oscillator ODE and verify cos/sin analytical solution.
#[test]
fn test_scientific_pipeline_ode_oscillator() -> TestResult<()> {
    // y' = [y1, -y0]
    let f = |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { array![y[1], -y[0]] };

    let options = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-9,
        atol: 1e-11,
        max_steps: 5000,
        ..Default::default()
    };

    // Integrate over [0, 2π] — one full period
    let t_end = 2.0 * std::f64::consts::PI;
    let result = solve_ivp(f, [0.0_f64, t_end], array![1.0_f64, 0.0_f64], Some(options))
        .map_err(|e| format!("solve_ivp (oscillator) failed: {}", e))?;

    assert!(result.success, "Oscillator ODE solver did not succeed");

    // Check a few intermediate time points
    let tol = 1e-5;
    for (t_val, y_vec) in result.t.iter().zip(result.y.iter()) {
        let expected_y0 = t_val.cos();
        let expected_y1 = -t_val.sin();
        assert!(
            (y_vec[0] - expected_y0).abs() < tol,
            "y0 at t={:.4}: computed={:.8}, expected={:.8}",
            t_val,
            y_vec[0],
            expected_y0
        );
        assert!(
            (y_vec[1] - expected_y1).abs() < tol,
            "y1 at t={:.4}: computed={:.8}, expected={:.8}",
            t_val,
            y_vec[1],
            expected_y1
        );
    }

    // After one full period y(2π) ≈ y(0) = [1, 0]
    let y_final = result.y.last().ok_or("No solution points")?;
    assert_abs_diff_eq!(y_final[0], 1.0, epsilon = tol);
    assert_abs_diff_eq!(y_final[1], 0.0, epsilon = tol);

    println!(
        "Oscillator ODE: t=[0,2π], {} steps, y(2π)=[{:.8},{:.8}]",
        result.n_steps, y_final[0], y_final[1]
    );
    Ok(())
}
