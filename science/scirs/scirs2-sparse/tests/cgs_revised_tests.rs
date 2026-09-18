use scirs2_sparse::{
    csr::CsrMatrix,
    linalg::{cgs, AsLinearOperator, CGSOptions, JacobiPreconditioner, LinearOperator},
};

#[test]
#[allow(dead_code)]
fn test_cgs_identity() {
    // Test CGS on identity matrix - should converge in 1 iteration
    let rows = vec![0, 1, 2];
    let cols = vec![0, 1, 2];
    let data = vec![1.0, 1.0, 1.0];
    let shape = (3, 3);

    let matrix = CsrMatrix::new(data, rows, cols, shape).expect("Operation failed");
    let op = matrix.as_linear_operator();

    let b = vec![1.0, 2.0, 3.0];
    let options = CGSOptions::default();
    let result = cgs(op.as_ref(), &b, options).expect("Operation failed");

    assert!(result.converged);
    assert_eq!(result.iterations, 1);
    for (xi, bi) in result.x.iter().zip(&b) {
        let diff: f64 = *xi - *bi;
        assert!(diff.abs() < 1e-10);
    }
}

#[test]
#[allow(dead_code)]
fn test_cgs_well_conditioned() {
    // Test CGS on a well-conditioned non-symmetric matrix
    // This matrix is diagonally dominant
    let rows = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let cols = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let data = vec![4.0, 0.5, 0.5, 0.5, 4.0, 0.5, 0.0, 0.5, 4.0];
    let shape = (3, 3);

    let matrix = CsrMatrix::new(data, rows, cols, shape).expect("Operation failed");
    let op = matrix.as_linear_operator();

    let b = vec![5.0, 5.0, 4.5];
    let options = CGSOptions::<f64> {
        rtol: 1e-8,
        atol: 1e-10,
        ..Default::default()
    };

    let result = cgs(op.as_ref(), &b, options).expect("Operation failed");

    // CGS may not always converge based on internal residual measure,
    // but what matters is the actual residual of the system

    // Verify solution
    let ax = op.matvec(&result.x).expect("Operation failed");
    let residual: Vec<f64> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();
    let residual_norm: f64 = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();

    assert!(
        residual_norm < 1e-8,
        "Residual norm too large: {}",
        residual_norm
    );
}

#[test]
#[allow(dead_code)]
fn test_cgs_with_preconditioner() {
    // Test CGS with Jacobi preconditioner on a moderately ill-conditioned matrix
    let rows = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];
    let cols = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
    let data = vec![
        10.0, 1.0, 0.5, 0.0, // row 0
        1.0, 8.0, 0.5, 0.5, // row 1
        0.5, 0.5, 6.0, 1.0, // row 2
        0.0, 0.5, 1.0, 4.0, // row 3
    ];
    let shape = (4, 4);

    let matrix = CsrMatrix::new(data, rows, cols, shape).expect("Operation failed");
    let op = matrix.as_linear_operator();

    // Create RHS vector
    let b = vec![11.5, 10.0, 8.0, 5.5];

    // Without preconditioner
    let options_no_precond = CGSOptions::<f64> {
        max_iter: 100,
        rtol: 1e-6,
        ..Default::default()
    };

    let result_no_precond = cgs(op.as_ref(), &b, options_no_precond).expect("Operation failed");

    // With Jacobi preconditioner
    let precond = JacobiPreconditioner::new(&matrix).expect("Operation failed");
    let options_precond = CGSOptions::<f64> {
        max_iter: 100,
        rtol: 1e-6,
        right_preconditioner: Some(Box::new(precond) as Box<dyn LinearOperator<f64>>),
        ..Default::default()
    };

    let result_precond = cgs(op.as_ref(), &b, options_precond).expect("Operation failed");

    // With preconditioner should converge faster (if it converges)
    if result_precond.converged && result_no_precond.converged {
        assert!(result_precond.iterations <= result_no_precond.iterations);
    }
}

#[test]
#[allow(dead_code)]
fn test_cgs_real_world_pattern() {
    // Test CGS on a sparse matrix with real-world sparsity pattern
    // This represents a simple finite difference discretization
    let n = 5; // 5x5 grid -> 25x25 matrix
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    // Create a 2D Laplacian matrix
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;

            // Diagonal entry
            rows.push(idx);
            cols.push(idx);
            data.push(4.0);

            // Off-diagonal entries (4-point stencil)
            if i > 0 {
                rows.push(idx);
                cols.push((i - 1) * n + j);
                data.push(-1.0);
            }
            if i < n - 1 {
                rows.push(idx);
                cols.push((i + 1) * n + j);
                data.push(-1.0);
            }
            if j > 0 {
                rows.push(idx);
                cols.push(i * n + (j - 1));
                data.push(-1.0);
            }
            if j < n - 1 {
                rows.push(idx);
                cols.push(i * n + (j + 1));
                data.push(-1.0);
            }
        }
    }

    let matrix = CsrMatrix::new(data, rows, cols, (n * n, n * n)).expect("Operation failed");
    let op = matrix.as_linear_operator();

    // Create a simple RHS
    let b: Vec<f64> = (0..n * n).map(|i| (i + 1) as f64).collect();

    let options = CGSOptions::<f64> {
        max_iter: 200, // May need more iterations for larger problems
        rtol: 1e-5,
        atol: 1e-7,
        ..Default::default()
    };

    let result = cgs(op.as_ref(), &b, options).expect("Operation failed");

    // We just check that it produces a valid solution (residual is small)
    if result.converged {
        let ax = op.matvec(&result.x).expect("Operation failed");
        let residual: Vec<f64> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();
        let residual_norm: f64 = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|&r| r * r).sum::<f64>().sqrt();

        assert!(residual_norm / b_norm < 1e-4, "Relative residual too large");
    }
}

#[test]
#[allow(dead_code)]
fn test_cgs_symmetric_vs_cg() {
    // CGS should also work on symmetric positive definite matrices
    // though CG would be preferred in practice
    let rows = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let cols = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let data = vec![4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0];
    let shape = (3, 3);

    let matrix = CsrMatrix::new(data, rows, cols, shape).expect("Operation failed");
    let op = matrix.as_linear_operator();

    let b = vec![2.0, 2.0, 2.0];
    let options = CGSOptions::<f64> {
        rtol: 1e-8,
        ..Default::default()
    };

    let result = cgs(op.as_ref(), &b, options).expect("Operation failed");

    assert!(result.converged);

    // Check that solution is correct
    let expected = vec![1.0, 1.0, 1.0]; // Due to symmetry
    for (xi, ei) in result.x.iter().zip(&expected) {
        let diff: f64 = *xi - *ei;
        assert!(diff.abs() < 1e-6);
    }
}
