// Enhanced DPSS (Discrete Prolate Spheroidal Sequences) implementation
//
// This module provides a corrected and validated implementation of DPSS
// computation following SciPy's approach and Percival & Walden (1993).
// Updated to remove ndarray-linalg dependency.

use crate::error::{SignalError, SignalResult};
use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use scirs2_core::validation::check_positive;
use std::f64::consts::PI;

#[allow(unused_imports)]
/// Enhanced DPSS computation with proper SciPy-compatible implementation
///
/// Computes the Discrete Prolate Spheroidal Sequences (Slepian sequences)
/// using the tridiagonal matrix formulation.
///
/// # Arguments
///
/// * `n` - Sequence length
/// * `nw` - Time-bandwidth product (typically 2-4)
/// * `k` - Number of sequences to compute (default: 2*nw - 1)
/// * `return_ratios` - Whether to return concentration ratios
///
/// # Returns
///
/// * Tuple of (tapers, concentration_ratios)
#[allow(dead_code)]
pub fn dpss_enhanced(
    n: usize,
    nw: f64,
    k: usize,
    return_ratios: bool,
) -> SignalResult<(Array2<f64>, Option<Array1<f64>>)> {
    // Validate inputs
    check_positive(n, "n")?;
    check_positive(nw, "nw")?;
    check_positive(k, "k")?;

    if k > n {
        return Err(SignalError::ValueError(format!(
            "k ({}) must not exceed n ({})",
            k, n
        )));
    }

    // Maximum useful number of tapers
    let k_max = (2.0 * nw).floor() as usize;
    if k > k_max {
        eprintln!(
            "Warning: k ({}) is greater than 2*NW-1 ({}). The higher order tapers will have poor concentration.",
            k, k_max - 1
        );
    }

    // Compute normalized frequency
    let w = nw / n as f64;

    // Build the Toeplitz concentration matrix B directly
    // B[i,j] = sin(2*pi*W*(i-j)) / (pi*(i-j)) for i!=j, B[i,i] = 2*W
    // The eigenvectors of B are the DPSS sequences and eigenvalues
    // are the concentration ratios.
    // For small n (up to ~128), dense Jacobi eigendecomposition is practical.
    // For larger n, use the faster tridiagonal QR algorithm.
    let (eigenvalues, eigenvectors) = if n <= 128 {
        solve_concentration_matrix(n, w)?
    } else {
        // For larger n, use the tridiagonal formulation
        let (diagonal, off_diagonal) = build_tridiagonal_matrix(n, w);
        solve_tridiagonal_symmetric(&diagonal, &off_diagonal)?
    };

    // Sort by eigenvalue (descending, largest first)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| {
        eigenvalues[j]
            .partial_cmp(&eigenvalues[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build final tapers with correct ordering and sign convention
    let mut tapers = Array2::zeros((k, n));
    let mut final_ratios = Array1::zeros(k);

    for i in 0..k {
        let idx = indices[i];
        let mut eigvec = eigenvectors.column(idx).to_owned();
        normalize_eigenvector(&mut eigvec);
        apply_sign_convention(&mut eigvec, i);

        tapers.row_mut(i).assign(&eigvec);
        // For the concentration matrix, eigenvalues ARE the concentration ratios
        final_ratios[i] = eigenvalues[idx].clamp(0.0, 1.0);
    }

    // If we used the tridiagonal approach (large n), compute concentration
    // ratios from the Toeplitz matrix
    let ratios = if return_ratios {
        if n > 128 {
            Some(compute_concentration_ratios(&tapers, w, n)?)
        } else {
            Some(final_ratios)
        }
    } else {
        None
    };

    Ok((tapers, ratios))
}

/// Build the tridiagonal matrix for the eigenvalue problem
#[allow(dead_code)]
fn build_tridiagonal_matrix(n: usize, w: f64) -> (Vec<f64>, Vec<f64>) {
    let cos_2pi_w = (2.0 * PI * w).cos();

    // Diagonal elements: ((n-1-2i)/2)^2 * cos(2πW)
    let diagonal: Vec<f64> = (0..n)
        .map(|i| {
            let term = (n as f64 - 1.0 - 2.0 * i as f64) / 2.0;
            term * term * cos_2pi_w
        })
        .collect();

    // Off-diagonal elements: i(n-i)/2 for i = 1, 2, ..., n-1
    let off_diagonal: Vec<f64> = (1..n).map(|i| (i as f64 * (n - i) as f64) / 2.0).collect();

    (diagonal, off_diagonal)
}

/// Build and solve the Toeplitz concentration matrix eigenvalue problem directly.
/// This gives more accurate DPSS sequences than the tridiagonal approach for small n.
#[allow(dead_code)]
fn solve_concentration_matrix(n: usize, w: f64) -> SignalResult<(Vec<f64>, Array2<f64>)> {
    // Build the symmetric Toeplitz matrix B
    // B[i,j] = sin(2*pi*W*(i-j)) / (pi*(i-j)) for i!=j
    // B[i,i] = 2*W
    let mut b_mat = Array2::zeros((n, n));
    for i in 0..n {
        b_mat[[i, i]] = 2.0 * w;
        for j in (i + 1)..n {
            let d = (j - i) as f64;
            let val = (2.0 * PI * w * d).sin() / (PI * d);
            b_mat[[i, j]] = val;
            b_mat[[j, i]] = val;
        }
    }

    // Solve using Jacobi eigenvalue algorithm for symmetric matrices
    jacobi_eigendecomposition(&mut b_mat)
}

/// Jacobi eigenvalue algorithm for symmetric matrices (cyclic sweep variant)
/// Returns eigenvalues and eigenvectors (columns of the returned matrix)
#[allow(dead_code)]
fn jacobi_eigendecomposition(a: &mut Array2<f64>) -> SignalResult<(Vec<f64>, Array2<f64>)> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(SignalError::ValueError("Matrix must be square".to_string()));
    }

    let mut v = Array2::eye(n);
    let max_sweeps = 100;
    let eps = f64::EPSILON;

    for _sweep in 0..max_sweeps {
        // Compute off-diagonal norm
        let mut off_norm = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off_norm += a[[i, j]] * a[[i, j]];
            }
        }
        off_norm = off_norm.sqrt();

        // Check convergence
        if off_norm < eps * 100.0 {
            break;
        }

        // Threshold for this sweep (classical Jacobi with threshold)
        let threshold = if _sweep < 3 {
            0.2 * off_norm / (n * n) as f64
        } else {
            0.0
        };

        // Cyclic sweep: process all (i,j) pairs
        for p in 0..n - 1 {
            for q in (p + 1)..n {
                let apq = a[[p, q]];

                // Skip small elements
                if apq.abs() < threshold {
                    continue;
                }

                // For very small elements, just set to zero
                if apq.abs() < eps * (a[[p, p]].abs() + a[[q, q]].abs()) * 0.01 {
                    a[[p, q]] = 0.0;
                    a[[q, p]] = 0.0;
                    continue;
                }

                let app = a[[p, p]];
                let aqq = a[[q, q]];

                // Compute rotation angle
                let (c, s) = if (app - aqq).abs() < eps * (app.abs() + aqq.abs()) {
                    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
                    (inv_sqrt2, if apq >= 0.0 { inv_sqrt2 } else { -inv_sqrt2 })
                } else {
                    let tau = (aqq - app) / (2.0 * apq);
                    let t = if tau >= 0.0 {
                        1.0 / (tau + (1.0 + tau * tau).sqrt())
                    } else {
                        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                    };
                    let c_val = 1.0 / (1.0 + t * t).sqrt();
                    (c_val, t * c_val)
                };

                // Apply rotation: A' = J^T * A * J
                for i in 0..n {
                    if i != p && i != q {
                        let aip = a[[i, p]];
                        let aiq = a[[i, q]];
                        a[[i, p]] = c * aip - s * aiq;
                        a[[p, i]] = a[[i, p]];
                        a[[i, q]] = s * aip + c * aiq;
                        a[[q, i]] = a[[i, q]];
                    }
                }

                let new_app = c * c * app - 2.0 * c * s * apq + s * s * aqq;
                let new_aqq = s * s * app + 2.0 * c * s * apq + c * c * aqq;
                a[[p, p]] = new_app;
                a[[q, q]] = new_aqq;
                a[[p, q]] = 0.0;
                a[[q, p]] = 0.0;

                // Accumulate eigenvectors
                for i in 0..n {
                    let vip = v[[i, p]];
                    let viq = v[[i, q]];
                    v[[i, p]] = c * vip - s * viq;
                    v[[i, q]] = s * vip + c * viq;
                }
            }
        }
    }

    // Extract eigenvalues from diagonal
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[[i, i]]).collect();

    Ok((eigenvalues, v))
}

/// Solve symmetric tridiagonal eigenvalue problem using implicit QR algorithm
/// with Wilkinson shift (Golub & Van Loan, Chapter 8)
#[allow(dead_code)]
fn solve_tridiagonal_symmetric(
    diagonal: &[f64],
    off_diagonal: &[f64],
) -> SignalResult<(Vec<f64>, Array2<f64>)> {
    let n = diagonal.len();

    if n == 0 {
        return Ok((vec![], Array2::zeros((0, 0))));
    }
    if n == 1 {
        let mut q = Array2::zeros((1, 1));
        q[[0, 0]] = 1.0;
        return Ok((diagonal.to_vec(), q));
    }

    // Copy arrays for modification
    let mut d = diagonal.to_vec();
    let mut e = off_diagonal.to_vec();

    // Initialize eigenvector matrix as identity
    let mut z = Array2::eye(n);

    // Implicit QR algorithm for symmetric tridiagonal matrices
    // The QR algorithm typically converges in O(n) iterations per eigenvalue
    // with Wilkinson shifts. Use 30*n as a safe upper bound.
    let max_iterations = 30 * n;
    let eps = f64::EPSILON;

    let mut m_end = n - 1; // active submatrix upper index
    let mut converged = false;

    for _iter in 0..max_iterations {
        // Check if we've deflated everything
        if m_end == 0 {
            converged = true;
            break;
        }

        // Find the largest m_start such that e[m_start..m_end] are all non-negligible
        // (i.e., find the unreduced block)
        let mut m_start = m_end;
        while m_start > 0 {
            let test_val = e[m_start - 1].abs();
            let threshold = eps * (d[m_start - 1].abs() + d[m_start].abs());
            if test_val <= threshold {
                e[m_start - 1] = 0.0;
                break;
            }
            m_start -= 1;
        }

        if m_start == m_end {
            // 1x1 block has converged, deflate
            m_end -= 1;
            continue;
        }

        // Wilkinson shift: eigenvalue of trailing 2x2 submatrix closer to d[m_end]
        let shift = wilkinson_shift_val(d[m_end - 1], d[m_end], e[m_end - 1]);

        // Implicit QR step with shift (chase the bulge)
        implicit_qr_step(&mut d, &mut e, &mut z, m_start, m_end, shift);
    }

    if !converged {
        eprintln!(
            "Warning: QR algorithm did not fully converge after {} iterations (n={})",
            max_iterations, n
        );
    }

    Ok((d, z))
}

/// Refine eigenvectors using inverse iteration with deflation
/// For each eigenvector, solve (T - lambda*I) * x = v iteratively
/// to get a more accurate eigenvector, then orthogonalize.
#[allow(dead_code)]
fn refine_eigenvectors_inverse_iteration(
    diag: &[f64],
    offdiag: &[f64],
    eigenvalues: &[f64],
    z: &mut Array2<f64>,
) {
    let n = diag.len();
    if n < 2 {
        return;
    }

    // Sort eigenvalues by descending value to process most important first
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&i, &j| {
        eigenvalues[j]
            .partial_cmp(&eigenvalues[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n_refine_iters = 5;

    // Collect refined eigenvectors
    let mut refined_vecs: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut refined_indices: Vec<usize> = Vec::with_capacity(n);

    for &col in &sorted_indices {
        let lambda = eigenvalues[col];
        // Use a small perturbation proportional to machine epsilon
        let perturbation = f64::EPSILON * lambda.abs().max(1.0) * 10.0;
        let sigma = lambda + perturbation;

        // Start with random-ish initial vector (use current QR eigenvector)
        let mut x: Vec<f64> = (0..n).map(|i| z[[i, col]]).collect();

        for _iter in 0..n_refine_iters {
            // Solve (T - sigma*I) * y = x using Thomas algorithm
            let mut d_s: Vec<f64> = diag.iter().map(|&d| d - sigma).collect();
            let mut rhs = x.clone();

            // Add small regularization to diagonal to prevent exact singularity
            for d in d_s.iter_mut() {
                if d.abs() < f64::EPSILON * 1e6 {
                    *d = f64::EPSILON * 1e6 * if *d >= 0.0 { 1.0 } else { -1.0 };
                }
            }

            // Forward elimination (Thomas algorithm)
            let mut c = offdiag.to_vec();
            for i in 1..n {
                let factor = offdiag[i - 1] / d_s[i - 1];
                d_s[i] -= factor * c[i - 1];
                rhs[i] -= factor * rhs[i - 1];
            }

            // Back substitution
            x[n - 1] = rhs[n - 1] / d_s[n - 1];
            for i in (0..n - 1).rev() {
                x[i] = (rhs[i] - c[i] * x[i + 1]) / d_s[i];
            }

            // Orthogonalize against previously refined eigenvectors
            for prev_vec in &refined_vecs {
                let dot: f64 = x.iter().zip(prev_vec.iter()).map(|(&a, &b)| a * b).sum();
                for (xi, &pi) in x.iter_mut().zip(prev_vec.iter()) {
                    *xi -= dot * pi;
                }
            }

            // Normalize
            let norm: f64 = x.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if norm > f64::EPSILON {
                for v in x.iter_mut() {
                    *v /= norm;
                }
            }
        }

        refined_vecs.push(x);
        refined_indices.push(col);
    }

    // Store refined eigenvectors back into z
    for (vec, &col) in refined_vecs.iter().zip(refined_indices.iter()) {
        for i in 0..n {
            z[[i, col]] = vec[i];
        }
    }
}

/// Compute Wilkinson shift for the trailing 2x2 submatrix
#[allow(dead_code)]
fn wilkinson_shift_val(a: f64, b: f64, e: f64) -> f64 {
    // 2x2 matrix: [[a, e], [e, b]]
    // Eigenvalue closer to b
    let delta = (a - b) / 2.0;
    if delta.abs() < f64::EPSILON {
        b - e.abs()
    } else {
        let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
        b - e * e / (delta + sign * (delta * delta + e * e).sqrt())
    }
}

/// Perform one implicit QR step on the tridiagonal submatrix d[m_start..=m_end]
/// using Givens rotations to chase the bulge.
#[allow(dead_code)]
fn implicit_qr_step(
    d: &mut [f64],
    e: &mut [f64],
    z: &mut Array2<f64>,
    m_start: usize,
    m_end: usize,
    shift: f64,
) {
    let n = d.len();

    // Initial bulge creation
    let mut x = d[m_start] - shift;
    let mut y = e[m_start];

    for k in m_start..m_end {
        // Compute Givens rotation to zero out y
        let (c, s) = givens_rotation(x, y);

        // Apply rotation from both sides to the tridiagonal matrix
        // This is a similarity transform, so the matrix stays symmetric
        if k > m_start {
            e[k - 1] = c * x + s * y;
        }

        let d_k = d[k];
        let d_k1 = d[k + 1];
        let e_k = e[k];

        // Update diagonal and off-diagonal elements
        d[k] = c * c * d_k + 2.0 * c * s * e_k + s * s * d_k1;
        d[k + 1] = s * s * d_k - 2.0 * c * s * e_k + c * c * d_k1;
        e[k] = c * s * (d_k1 - d_k) + (c * c - s * s) * e_k;

        // Compute next bulge element
        if k + 1 < m_end {
            x = e[k];
            y = s * e[k + 1];
            e[k + 1] = c * e[k + 1];
        }

        // Accumulate eigenvectors: Z = Z * G_k
        for j in 0..n {
            let temp = c * z[[j, k]] + s * z[[j, k + 1]];
            z[[j, k + 1]] = -s * z[[j, k]] + c * z[[j, k + 1]];
            z[[j, k]] = temp;
        }
    }
}

/// Compute Givens rotation parameters to zero out b in [a, b]
/// Returns (c, s) such that [c, s; -s, c]^T * [a; b] = [r; 0]
#[allow(dead_code)]
fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
    if b.abs() < f64::EPSILON * a.abs().max(1.0) {
        return (1.0, 0.0);
    }

    if a.abs() < f64::EPSILON * b.abs().max(1.0) {
        return (0.0, if b >= 0.0 { 1.0 } else { -1.0 });
    }

    let r = a.hypot(b);
    (a / r, b / r)
}

/// Normalize eigenvector to unit norm
#[allow(dead_code)]
fn normalize_eigenvector(eigvec: &mut Array1<f64>) {
    let norm = eigvec.dot(eigvec).sqrt();
    if norm > 1e-10 {
        *eigvec /= norm;
    }
}

/// Apply sign convention to ensure consistency
#[allow(dead_code)]
fn apply_sign_convention(eigvec: &mut Array1<f64>, order: usize) {
    let n = eigvec.len();

    if order % 2 == 0 {
        // Even order: ensure symmetric taper has positive average
        let sum: f64 = eigvec.sum();
        if sum < 0.0 {
            *eigvec *= -1.0;
        }
    } else {
        // Odd order: ensure antisymmetric taper starts positive
        let mid = n / 2;
        if eigvec[0] < 0.0 || (n % 2 == 1 && eigvec[mid] < 0.0) {
            *eigvec *= -1.0;
        }
    }
}

/// Compute concentration ratios using the direct quadratic form method
///
/// The concentration ratio (eigenvalue) lambda_k is defined as:
/// lambda_k = v_k^T * B * v_k
/// where B[i,j] = sin(2*pi*W*(i-j)) / (pi*(i-j)) for i != j
/// and B[i,i] = 2*W
#[allow(dead_code)]
fn compute_concentration_ratios(
    tapers: &Array2<f64>,
    w: f64,
    n: usize,
) -> SignalResult<Array1<f64>> {
    let k = tapers.nrows();
    let mut ratios = Array1::zeros(k);

    // Precompute the sinc-like kernel: b[d] = sin(2*pi*W*d) / (pi*d) for d > 0
    // and b[0] = 2*W
    let mut b = vec![0.0; n];
    b[0] = 2.0 * w;
    for d in 1..n {
        let x = d as f64 * PI;
        b[d] = (2.0 * PI * w * d as f64).sin() / x;
    }

    for i in 0..k {
        let taper = tapers.row(i);

        // Compute v^T * B * v using the Toeplitz structure more efficiently
        // B is a symmetric Toeplitz matrix, so B[i,j] = b[|i-j|]
        // Optimize from O(n²) to O(n²) but with better constants by computing
        // diagonal-wise instead of element-wise
        let mut concentration = 0.0;

        // Diagonal term: sum_p b[0] * v[p]^2
        for p in 0..n {
            concentration += b[0] * taper[p] * taper[p];
        }

        // Off-diagonal terms: for each diagonal d, sum_p v[p] * v[p+d]
        // We exploit symmetry: B[p,p+d] = B[p+d,p] = b[d]
        // So we compute each d once and multiply by 2
        for d in 1..n {
            let mut diag_sum = 0.0;
            for p in 0..(n - d) {
                diag_sum += taper[p] * taper[p + d];
            }
            concentration += 2.0 * b[d] * diag_sum;
        }

        ratios[i] = concentration.clamp(0.0, 1.0);
    }

    Ok(ratios)
}

/// Validate DPSS computation against known values
#[allow(dead_code)]
pub fn validate_dpss_implementation() -> SignalResult<bool> {
    // Test case from SciPy documentation
    let n = 64;
    let nw = 4.0;
    let k = 7;

    let (tapers, ratios) = dpss_enhanced(n, nw, k, true)?;
    let ratios = ratios.expect("Operation failed");

    // Expected concentration ratios (verified against SciPy 1.x for N=64, NW=4.0)
    // These are the exact eigenvalues of the Toeplitz concentration matrix B
    // where B[i,j] = sin(2*pi*W*(i-j)) / (pi*(i-j)), W = NW/N = 0.0625
    let expected_ratios = vec![
        0.999999999746,
        0.999999975397,
        0.999998895190,
        0.999969657680,
        0.999436549707,
        0.992710115932,
        0.937468604926,
    ];

    // Check concentration ratios against SciPy reference values
    for i in 0..k {
        let error = (ratios[i] - expected_ratios[i]).abs();
        if error > 1e-6 {
            eprintln!(
                "Concentration ratio mismatch at index {}: expected {:.12}, got {:.12}",
                i, expected_ratios[i], ratios[i]
            );
            return Ok(false);
        }
    }

    // Check orthogonality
    for i in 0..k {
        for j in i + 1..k {
            let dot_product = tapers.row(i).dot(&tapers.row(j));
            if dot_product.abs() > 1e-8 {
                eprintln!(
                    "Orthogonality violated: tapers {} and {} have dot product {:.2e}",
                    i, j, dot_product
                );
                return Ok(false);
            }
        }
    }

    // Check normalization
    for i in 0..k {
        let norm = tapers.row(i).dot(&tapers.row(i)).sqrt();
        if ((norm - 1.0) as f64).abs() > 1e-10 {
            eprintln!("Taper {} not normalized: norm = {:.10}", i, norm);
            return Ok(false);
        }
    }

    Ok(true)
}

/// Generate reference values for testing
#[allow(dead_code)]
pub fn generate_reference_values() -> SignalResult<()> {
    println!("DPSS Reference Values:");
    println!("======================");

    // Standard test cases
    let test_cases = vec![
        (16, 2.5, 3),
        (32, 3.0, 5),
        (64, 4.0, 7),
        (128, 4.0, 7),
        (256, 3.5, 6),
    ];

    for (n, nw, k) in test_cases {
        println!("\nCase: n={}, NW={}, k={}", n, nw, k);

        let (tapers, ratios) = dpss_enhanced(n, nw, k, true)?;
        let ratios = ratios.expect("Operation failed");

        println!("Concentration ratios:");
        for i in 0..k {
            println!("  λ[{}] = {:.12}", i, ratios[i]);
        }

        // Print first few values of first taper
        println!("First taper (first 8 values):");
        for i in 0..8.min(n) {
            println!("  v[0][{}] = {:.12}", i, tapers[[0, i]]);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    #[test]
    fn test_dpss_basic() {
        let (tapers, ratios) = dpss_enhanced(64, 4.0, 7, true).expect("Operation failed");

        assert_eq!(tapers.nrows(), 7);
        assert_eq!(tapers.ncols(), 64);
        assert!(ratios.is_some());
    }

    #[test]
    fn test_dpss_orthogonality() {
        let (tapers, _) = dpss_enhanced(128, 4.0, 7, false).expect("Operation failed");

        // Check orthogonality
        for i in 0..7 {
            for j in i + 1..7 {
                let dot = tapers.row(i).dot(&tapers.row(j));
                assert_abs_diff_eq!(dot, 0.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_dpss_normalization() {
        let (tapers_, _) = dpss_enhanced(128, 4.0, 7, false).expect("Operation failed");

        // Check unit norm
        for i in 0..7 {
            let norm_sq = tapers_.row(i).dot(&tapers_.row(i));
            assert_abs_diff_eq!(norm_sq, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_concentration_ratios() {
        let (_, ratios) = dpss_enhanced(64, 4.0, 7, true).expect("Operation failed");
        let ratios = ratios.expect("Operation failed");

        // All ratios should be between 0 and 1
        for &ratio in ratios.iter() {
            assert!(ratio >= 0.0 && ratio <= 1.0);
        }

        // Ratios should decrease
        for i in 1..ratios.len() {
            assert!(ratios[i] <= ratios[i - 1]);
        }
    }

    #[test]
    fn test_validation() {
        assert!(validate_dpss_implementation().expect("Operation failed"));
    }
}
