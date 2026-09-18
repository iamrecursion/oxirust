use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_linalg::compat::{self, UPLO};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // This example demonstrates the SciPy-compatible API
    println!("SciPy-compatible API example for scirs2-linalg");

    // Create a test matrix
    let a: Array2<f64> = array![[4.0, 2.0], [2.0, 5.0]];
    println!("Matrix A:");
    println!("{}", a);

    // Compute determinant
    let det = compat::det(&a.view())?;
    println!("\nDeterminant: {}", det);

    // Compute inverse
    let inv = compat::inv(&a.view())?;
    println!("\nInverse:");
    println!("{}", inv);

    // LU decomposition
    let (p, l, u) = compat::lu(&a.view())?;
    println!("\nLU decomposition:");
    println!("P =\n{}", p);
    println!("L =\n{}", l);
    println!("U =\n{}", u);

    // QR decomposition
    let (q, r) = compat::qr(&a.view())?;
    println!("\nQR decomposition:");
    println!("Q =\n{}", q);
    println!("R =\n{}", r);

    // SVD decomposition
    let (u, s, vt) = compat::svd(&a.view(), true)?;
    println!("\nSVD decomposition:");
    println!("U =\n{}", u);
    println!("S =\n{}", s);
    println!("Vt =\n{}", vt);

    // Cholesky decomposition (for positive definite matrix)
    let l = compat::cholesky(&a.view(), UPLO::Lower)?;
    println!("\nCholesky decomposition:");
    println!("L =\n{}", l);

    // Solve linear system
    let b: Array1<f64> = array![1.0, 2.0];
    println!("\nVector b:");
    println!("{}", b);

    let x = compat::compat_solve(&a.view(), &b.view())?;
    println!("\nSolution to Ax = b:");
    println!("{}", x);

    // Eigenvalue decomposition (for symmetric matrix)
    let (eigenvalues, eigenvectors) = compat::eigh(&a.view(), UPLO::Lower)?;
    println!("\nEigenvalue decomposition for symmetric matrix:");
    println!("Eigenvalues =\n{}", eigenvalues);
    println!("Eigenvectors =\n{}", eigenvectors);

    println!("\nNote: The compat module provides SciPy-compatible function signatures");
    println!("while delegating to the efficient scirs2-linalg implementations.");

    Ok(())
}
