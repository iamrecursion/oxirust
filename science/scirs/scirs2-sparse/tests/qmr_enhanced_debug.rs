use scirs2_sparse::linalg::{qmr, IdentityOperator, QMROptions};

#[test]
#[allow(dead_code)]
fn test_qmr_identity_step_by_step() {
    // Test QMR implementation step-by-step on the identity matrix
    println!("=== QMR Step-by-Step Debug ===");

    // Test QMR on identity matrix
    let identity = IdentityOperator::<f64>::new(3);
    let b = vec![1.0, 2.0, 3.0];
    let x0 = vec![0.0, 0.0, 0.0];

    // Show how the algorithm would break down
    let r_initial = b.clone();
    println!("Initial residual: {:?}", r_initial);

    let r_star = r_initial.clone();
    println!("rstar: {:?}", r_star);

    let rho_initial: f64 = r_initial.iter().zip(&r_star).map(|(a, b)| a * b).sum();
    println!("Initial rho = dot(r, r_star) = {}", rho_initial);

    // Show the steps that would lead to breakdown
    let v_tilde_0: Vec<f64> = r_initial.iter().map(|&x| x / rho_initial).collect();
    println!("v_tilde_0 = r / rho = {:?}", v_tilde_0);

    let y_0: Vec<f64> = r_star.iter().map(|&x| x / rho_initial).collect();
    println!("y_0 = r_star / rho = {:?}", y_0);

    // This demonstrates the mathematical issue with the identity matrix
    println!("\n=== Running actual QMR function ===");
    let options = QMROptions {
        x0: Some(x0),
        max_iter: 1,
        rtol: 1e-8,
        atol: 1e-12,
        ..Default::default()
    };

    // Test that QMR works correctly on the identity matrix
    match qmr(&identity, &b, options) {
        Ok(result) => {
            println!("QMR converged in {} iterations", result.iterations);
            // The solution should be the same as b for the identity matrix
            for (i, b_val) in b.iter().enumerate().take(3) {
                assert!(
                    (result.x[i] - b_val).abs() < 1e-10,
                    "Solution x[{}] = {} should match b[{}] = {}",
                    i,
                    result.x[i],
                    i,
                    b_val
                );
            }
            assert!(result.converged, "QMR should converge for identity matrix");
        }
        Err(e) => panic!("QMR should succeed for identity matrix, got: {:?}", e),
    }
}
