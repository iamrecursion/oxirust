use approx::assert_relative_eq;
use scirs2_sparse::csr::CsrMatrix;
use scirs2_sparse::linalg::{
    cg, AsLinearOperator, CGOptions, JacobiPreconditioner, LinearOperator,
};

#[test]
#[allow(dead_code)]
fn test_cg_with_jacobi_preconditioner() {
    // Create a positive definite matrix with diagonal dominance:
    // [[4, -1, 0], [-1, 4, -1], [0, -1, 4]]
    let rows = vec![0, 0, 1, 1, 1, 2, 2];
    let cols = vec![0, 1, 0, 1, 2, 1, 2];
    let data = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
    let matrix = CsrMatrix::new(data, rows, cols, (3, 3)).expect("Operation failed");

    // Create Jacobi preconditioner
    let precond = JacobiPreconditioner::new(&matrix).expect("Operation failed");

    // Convert matrix to LinearOperator
    let op = matrix.as_linear_operator();

    // Right-hand side
    let b = vec![2.0, 1.0, 2.0];

    // Solve with preconditioner
    let options = CGOptions::<f64> {
        preconditioner: Some(Box::new(precond)),
        ..Default::default()
    };

    let result = cg(op.as_ref(), &b, options).expect("Operation failed");

    // Check convergence
    assert!(result.converged);
    assert!(result.iterations < 20); // Should converge faster with preconditioner

    // Verify solution by computing Ax and comparing with b
    let ax = op.matvec(&result.x).expect("Operation failed");
    for i in 0..3 {
        assert_relative_eq!(ax[i], b[i], epsilon = 1e-5);
    }
}

#[test]
#[allow(dead_code)]
fn test_jacobi_preconditioner_simple() {
    // Create a simple diagonal matrix for testing
    let rows = vec![0, 1, 2];
    let cols = vec![0, 1, 2];
    let data = vec![2.0, 3.0, 4.0];
    let matrix = CsrMatrix::new(data, rows, cols, (3, 3)).expect("Operation failed");

    // Create Jacobi preconditioner
    let precond = JacobiPreconditioner::new(&matrix).expect("Operation failed");

    // Test application
    let x = vec![2.0, 6.0, 12.0];
    let result = precond.matvec(&x).expect("Operation failed");

    // Should be [1.0, 2.0, 3.0] (element-wise division by diagonal)
    assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(result[1], 2.0, epsilon = 1e-10);
    assert_relative_eq!(result[2], 3.0, epsilon = 1e-10);
}
