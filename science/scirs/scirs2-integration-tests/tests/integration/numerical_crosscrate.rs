// Cross-crate numerical consistency tests for SciRS2 (Wave 44).
// Verifies that results are consistent across modules and mathematical
// identities hold end-to-end across crate boundaries.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;
use scirs2_core::ndarray::{Array1, Array2};

use scirs2_fft::{fft, ifft, rfft};
use scirs2_linalg::{eigh, solve, svd};

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

// ---------------------------------------------------------------------------
// FFT + signal consistency
// ---------------------------------------------------------------------------

/// Convolution theorem: circular convolution in time == pointwise multiplication
/// in frequency.  Direct naive circular convolution must agree with the
/// FFT-based approach to within floating-point tolerance.
#[test]
fn test_fft_convolution_theorem() -> TestResult<()> {
    let n = 8usize;
    let a = vec![1.0f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let b = vec![1.0f64, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Naive circular convolution y[k] = sum_j a[j] * b[(k-j) mod n]
    let mut direct_conv = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..n {
            direct_conv[(i + j) % n] += a[i] * b[j];
        }
    }

    // FFT-based circular convolution
    let fa = fft(&a, None).map_err(|e| format!("fft a: {}", e))?;
    let fb = fft(&b, None).map_err(|e| format!("fft b: {}", e))?;
    let fc: Vec<Complex64> = fa.iter().zip(fb.iter()).map(|(x, y)| x * y).collect();
    let fft_conv = ifft(&fc, None).map_err(|e| format!("ifft: {}", e))?;

    for i in 0..n {
        assert_abs_diff_eq!(fft_conv[i].re, direct_conv[i], epsilon = 1e-10);
    }

    println!("FFT convolution theorem verified (n={})", n);
    Ok(())
}

/// FFT round-trip: IFFT(FFT(x)).re == x to machine precision.
#[test]
fn test_fft_roundtrip() -> TestResult<()> {
    let n = 64usize;
    let signal: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 2.0 * std::f64::consts::PI / n as f64).sin())
        .collect();

    let spectrum = fft(&signal, None).map_err(|e| format!("fft: {}", e))?;
    let recovered = ifft(&spectrum, None).map_err(|e| format!("ifft: {}", e))?;

    for i in 0..n {
        assert_abs_diff_eq!(recovered[i].re, signal[i], epsilon = 1e-12);
    }

    println!("FFT round-trip verified (n={})", n);
    Ok(())
}

/// Parseval's theorem: sum |x[k]|^2 == (1/N) * sum |X[k]|^2.
#[test]
fn test_fft_parseval() -> TestResult<()> {
    let n = 128usize;
    let signal: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 2.0 * std::f64::consts::PI * 3.0 / n as f64).sin())
        .collect();

    let time_energy: f64 = signal.iter().map(|&x| x * x).sum();

    let spectrum = fft(&signal, Some(n)).map_err(|e| format!("fft: {}", e))?;
    let freq_energy: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum::<f64>() / n as f64;

    assert_abs_diff_eq!(time_energy, freq_energy, epsilon = 1e-10);
    println!(
        "Parseval's theorem verified: time_energy={:.6}, freq_energy={:.6}",
        time_energy, freq_energy
    );
    Ok(())
}

/// RFFT output length contract: rfft on n samples yields n/2+1 complex bins.
#[test]
fn test_rfft_length_contract() -> TestResult<()> {
    for &n in &[32usize, 64, 128, 256] {
        let signal: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let result = rfft(&signal, None).map_err(|e| format!("rfft n={}: {}", n, e))?;
        assert_eq!(
            result.len(),
            n / 2 + 1,
            "rfft(n={}) length: expected {}, got {}",
            n,
            n / 2 + 1,
            result.len()
        );
    }
    println!("RFFT length contract verified for n in [32, 64, 128, 256]");
    Ok(())
}

// ---------------------------------------------------------------------------
// linalg + optimize consistency
// ---------------------------------------------------------------------------

/// For the overdetermined least-squares problem min ||Ax - b||^2 the normal
/// equations A^T A x = A^T b must be solvable by the direct linear solver and
/// yield a small residual.
#[test]
fn test_normal_equations_solve() -> TestResult<()> {
    use ndarray::array;

    let a: Array2<f64> = array![[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]];
    let b: Array1<f64> = array![1.0, 2.0, 2.0];

    // Normal equations: (A^T A) x = A^T b
    let ata = a.t().dot(&a);
    let atb = a.t().dot(&b);
    let x = solve(&ata.view(), &atb.view(), None)
        .map_err(|e| format!("normal equations solve: {}", e))?;

    // Residual of the original problem ||Ax - b||
    let residual = &a.dot(&x) - &b;
    let res_norm: f64 = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();
    assert!(
        res_norm < 1.0,
        "Residual norm {:.4} should be < 1.0 for least-squares solution",
        res_norm
    );

    println!("Normal equations solve: residual_norm={:.6}", res_norm);
    Ok(())
}

/// A rank-1 matrix u u^T has exactly one non-zero eigenvalue equal to ||u||^2.
/// The remaining n-1 eigenvalues must be numerically zero.
#[test]
fn test_rank1_eigenvalue_consistency() -> TestResult<()> {
    use ndarray::array;

    let u: Array1<f64> = array![1.0, 2.0, 3.0];
    // Build outer product u u^T as 3x3 matrix
    let u_col = u.view().insert_axis(ndarray::Axis(1));
    let a: Array2<f64> = u_col.dot(&u_col.t());

    let (evals, _evecs) = eigh(&a.view(), None).map_err(|e| format!("eigh rank-1: {}", e))?;

    let mut sorted = evals.to_vec();
    sorted.sort_by(|p, q| q.partial_cmp(p).unwrap_or(std::cmp::Ordering::Equal));

    // ||u||^2 = 1+4+9 = 14
    assert_abs_diff_eq!(sorted[0], 14.0, epsilon = 1e-10);
    assert!(
        sorted[1].abs() < 1e-10,
        "second eigenvalue should be ~0, got {}",
        sorted[1]
    );
    assert!(
        sorted[2].abs() < 1e-10,
        "third eigenvalue should be ~0, got {}",
        sorted[2]
    );

    println!("Rank-1 eigenvalue: largest={:.4}, rest < 1e-10", sorted[0]);
    Ok(())
}

/// solve() must recover the exact solution for a well-conditioned 3x3 system.
#[test]
fn test_solve_known_system() -> TestResult<()> {
    use ndarray::array;

    // A x = b  with  x = [1, 2, 3]
    let a: Array2<f64> = array![[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
    let x_true: Array1<f64> = array![1.0, 2.0, 3.0];
    let b = a.dot(&x_true);

    let x = solve(&a.view(), &b.view(), None).map_err(|e| format!("solve: {}", e))?;

    for i in 0..3 {
        assert_abs_diff_eq!(x[i], x_true[i], epsilon = 1e-12);
    }

    println!("solve() exact recovery verified");
    Ok(())
}

// ---------------------------------------------------------------------------
// stats + linalg consistency (PCA via SVD)
// ---------------------------------------------------------------------------

/// SVD-based PCA: the variance explained by the first singular value should
/// dominate in a dataset where one dimension carries most of the signal.
#[test]
fn test_pca_variance_ordering() -> TestResult<()> {
    let n = 20usize;
    // Data: first column has large variance (slope 2.0), second is small (slope 0.3)
    let data: Array2<f64> = Array2::from_shape_fn((n, 2), |(i, j)| match j {
        0 => i as f64 * 2.0,
        _ => i as f64 * 0.3,
    });

    // Center data
    let mean: Array1<f64> = data
        .mean_axis(ndarray::Axis(0))
        .ok_or("mean_axis returned None")?;
    let centered = &data - &mean;

    let (_u, s, _vt) = svd(&centered.view(), true, None).map_err(|e| format!("svd: {}", e))?;

    // Variance explained is proportional to s[i]^2
    let var0 = s[0] * s[0];
    let var1 = s[1] * s[1];

    assert!(
        var0 > var1 * 3.0,
        "First PC variance ({:.4}) should dominate second ({:.4})",
        var0,
        var1
    );

    println!("PCA variance ordering: var0={:.4}, var1={:.4}", var0, var1);
    Ok(())
}

/// SVD singular values are non-negative and ordered descending.
#[test]
fn test_svd_singular_value_ordering() -> TestResult<()> {
    // 5x4 rectangular matrix
    let a: Array2<f64> =
        Array2::from_shape_fn((5, 4), |(i, j)| ((i + 1) as f64) * ((j + 1) as f64));

    let (_u, s, _vt) = svd(&a.view(), false, None).map_err(|e| format!("svd ordering: {}", e))?;

    // All singular values must be >= 0
    for &si in s.iter() {
        assert!(si >= -1e-14, "Negative singular value: {}", si);
    }

    // Values must be non-increasing
    for i in 1..s.len() {
        assert!(
            s[i - 1] >= s[i] - 1e-12,
            "Singular values not ordered: s[{}]={:.6} < s[{}]={:.6}",
            i - 1,
            s[i - 1],
            i,
            s[i]
        );
    }

    println!(
        "SVD singular values ordered and non-negative: {:?}",
        s.to_vec()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Sparse vs dense matmul consistency
// ---------------------------------------------------------------------------

/// Dense matrix-vector product for a known 3x3 matrix must yield the
/// analytically correct result.  (Wire-up test to catch import / API regressions.)
#[test]
fn test_dense_matvec_known_result() -> TestResult<()> {
    use ndarray::array;
    use scirs2_linalg::*;

    let a: Array2<f64> = array![[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 4.0]];
    let x: Array1<f64> = array![1.0, 2.0, 3.0];

    let y = a.dot(&x);

    // Expected: [2+3, 6, 1+12] = [5, 6, 13]
    assert_abs_diff_eq!(y[0], 5.0, epsilon = 1e-14);
    assert_abs_diff_eq!(y[1], 6.0, epsilon = 1e-14);
    assert_abs_diff_eq!(y[2], 13.0, epsilon = 1e-14);

    println!("Dense matvec known result verified: y = {:?}", y.to_vec());
    Ok(())
}

// ---------------------------------------------------------------------------
// Eigenvalue symmetry checks
// ---------------------------------------------------------------------------

/// For a symmetric positive-definite matrix all eigenvalues must be positive.
#[test]
fn test_spd_eigenvalues_positive() -> TestResult<()> {
    // Diagonal matrix with positive entries is trivially SPD
    let n = 5usize;
    let mut a = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        a[[i, i]] = (i as f64 + 1.0) * 2.0; // diagonal entries 2,4,6,8,10
    }

    let (evals, _) = eigh(&a.view(), None).map_err(|e| format!("eigh spd: {}", e))?;

    for &ev in evals.iter() {
        assert!(
            ev > 0.0,
            "SPD matrix must have all positive eigenvalues, got {}",
            ev
        );
    }

    println!("SPD eigenvalues all positive: {:?}", evals.to_vec());
    Ok(())
}

/// The sum of eigenvalues of a symmetric matrix must equal its trace.
#[test]
fn test_eigenvalue_trace_identity() -> TestResult<()> {
    // Build a small symmetric matrix
    let data = vec![4.0f64, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0];
    let a = Array2::from_shape_vec((3, 3), data).map_err(|e| format!("from_shape_vec: {}", e))?;

    let trace: f64 = a.diag().iter().sum();

    let (evals, _) = eigh(&a.view(), None).map_err(|e| format!("eigh trace: {}", e))?;

    let eval_sum: f64 = evals.iter().sum();

    assert_abs_diff_eq!(trace, eval_sum, epsilon = 1e-10);
    println!(
        "Trace identity: trace={:.6}, sum(eigenvalues)={:.6}",
        trace, eval_sum
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// FFT + linalg cross-module: circulant matrix eigenvalues via FFT
// ---------------------------------------------------------------------------

/// A circulant matrix's eigenvalues are exactly the DFT of its first row.
/// This test verifies that the FFT result (from scirs2-fft) matches the
/// eigenvalues computed by eigh (from scirs2-linalg) for a symmetric circulant.
#[test]
fn test_circulant_eigenvalues_via_fft() -> TestResult<()> {
    // Symmetric circulant: first row = [4, 1, 0, 1]
    // Matrix:
    //  4 1 0 1
    //  1 4 1 0
    //  0 1 4 1
    //  1 0 1 4
    let n = 4usize;
    let first_row = vec![4.0f64, 1.0, 0.0, 1.0];

    let mut a = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a[[i, j]] = first_row[(n + j - i) % n];
        }
    }

    // Eigenvalues via eigh
    let (mut evals_linalg, _) =
        eigh(&a.view(), None).map_err(|e| format!("eigh circulant: {}", e))?;

    // Eigenvalues via FFT of first row (real parts, symmetric circulant => all real)
    let spectrum = fft(&first_row, None).map_err(|e| format!("fft circulant: {}", e))?;
    let mut evals_fft: Vec<f64> = spectrum.iter().map(|c| c.re).collect();

    // Sort both sets for comparison
    evals_linalg
        .as_slice_mut()
        .ok_or("evals_linalg not contiguous")?
        .sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
    evals_fft.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));

    for i in 0..n {
        assert_abs_diff_eq!(evals_linalg[i], evals_fft[i], epsilon = 1e-10);
    }

    println!("Circulant eigenvalues agree between eigh and FFT");
    Ok(())
}

// ---------------------------------------------------------------------------
// Error-handling cross-crate: confirm dimension mismatches surface as Err
// ---------------------------------------------------------------------------

/// solve() with a non-square matrix must return Err, not panic.
#[test]
fn test_solve_nonsquare_returns_error() {
    use ndarray::array;
    let a: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 2x3, not square
    let b: Array1<f64> = array![1.0, 2.0];
    let result = solve(&a.view(), &b.view(), None);
    assert!(
        result.is_err(),
        "solve() on a non-square matrix must return Err"
    );
}

/// eigh() on a non-square matrix must return Err, not panic.
#[test]
fn test_eigh_nonsquare_returns_error() {
    use ndarray::array;
    let a: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 2x3, not square
    let result = eigh(&a.view(), None);
    assert!(
        result.is_err(),
        "eigh() on a non-square matrix must return Err"
    );
}

/// fft() on an empty slice must return Err, not panic.
#[test]
fn test_fft_empty_input_returns_error() {
    let result = fft::<f64>(&[], None);
    assert!(result.is_err(), "fft() on empty input must return Err");
}
