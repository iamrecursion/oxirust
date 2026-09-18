//! Advanced tests for extended sparse matrix functionality (Track C, Round 7).
//!
//! Tests cover:
//! - Generic SparseCsr<f64>, SparseCsc<f32/f64>
//! - Transpose and CSR↔CSC round-trips
//! - f64 SpMV and SpMM
//! - Batched SpMV
//! - from_dense / threshold filtering

use tensorlogic_oxicuda_sparse::{
    spmm_f64, spmv_batched, spmv_f64, SparseCsc, SparseCsr, SparseError,
};

// ---------------------------------------------------------------------------
// Helper utilities
// ---------------------------------------------------------------------------

/// Element-wise comparison of two `f32` slices with tolerance.
fn assert_f32_slice_eq(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (av - bv).abs() <= tol,
            "{label}[{i}]: got {av}, expected {bv} (tol={tol})"
        );
    }
}

/// Element-wise comparison of two `f64` slices with tolerance.
fn assert_f64_slice_eq(a: &[f64], b: &[f64], tol: f64, label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (av - bv).abs() <= tol,
            "{label}[{i}]: got {av}, expected {bv} (tol={tol})"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 1: CSR transpose twice is identity
// ---------------------------------------------------------------------------

/// Build a 3×4 sparse matrix, transpose it twice, and verify it equals the
/// original dense representation.
#[test]
fn csr_transpose_twice_is_identity() {
    // 3×4 matrix:
    // [ 1  0  2  0 ]
    // [ 0  3  0  4 ]
    // [ 5  0  6  0 ]
    let a: SparseCsr<f32> = SparseCsr::from_triplets(
        3,
        4,
        &[0, 0, 1, 1, 2, 2],
        &[0, 2, 1, 3, 0, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    let dense_original = a.to_dense();

    let at = a.transpose().unwrap();
    let att = at.transpose().unwrap();

    let dense_roundtrip = att.to_dense();

    assert_eq!(
        att.shape(),
        a.shape(),
        "shape must be restored after transpose^2"
    );
    assert_f32_slice_eq(
        &dense_roundtrip,
        &dense_original,
        1e-6,
        "transpose^2 identity",
    );
}

// ---------------------------------------------------------------------------
// Test 2: CSR → CSC → CSR round-trip
// ---------------------------------------------------------------------------

/// Convert a CSR matrix to CSC and back, verifying the dense representation is
/// preserved throughout.
#[test]
fn csr_to_csc_to_csr_roundtrip() {
    // 4×3 sparse matrix (non-trivial structure).
    let a: SparseCsr<f32> = SparseCsr::from_triplets(
        4,
        3,
        &[0, 1, 2, 3, 0, 2],
        &[0, 1, 2, 0, 2, 1],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    let dense_original = a.to_dense();

    let csc = a.to_csc().unwrap();
    assert_eq!(csc.shape(), (4, 3), "CSC shape should match CSR");

    let dense_csc = csc.to_dense();
    assert_f32_slice_eq(&dense_csc, &dense_original, 1e-6, "CSC dense match");

    let csr2 = csc.to_csr().unwrap();
    assert_eq!(csr2.shape(), (4, 3), "re-converted CSR shape must match");

    let dense_csr2 = csr2.to_dense();
    assert_f32_slice_eq(
        &dense_csr2,
        &dense_original,
        1e-6,
        "CSR roundtrip dense match",
    );
}

// ---------------------------------------------------------------------------
// Test 3: spmv_f64 matches dense matrix-vector product
// ---------------------------------------------------------------------------

/// Build a 3×3 dense and sparse (f64) matrix, compare SpMV outputs exactly.
///
/// Matrix:
/// [ 1  2  0 ]
/// [ 0  3  4 ]
/// [ 5  0  6 ]
///
/// x = [1, 2, 3]
/// Expected y = A*x = [5, 18, 23]
#[test]
fn spmv_f64_matches_dense() {
    let a: SparseCsr<f64> = SparseCsr::from_triplets(
        3,
        3,
        &[0, 0, 1, 1, 2, 2],
        &[0, 1, 1, 2, 0, 2],
        &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    let x = vec![1.0_f64, 2.0, 3.0];
    let mut y = vec![0.0_f64; 3];
    spmv_f64(&a, &x, 1.0, 0.0, &mut y).unwrap();

    // y[0] = 1*1 + 2*2 = 5
    // y[1] = 3*2 + 4*3 = 18
    // y[2] = 5*1 + 6*3 = 23
    assert_f64_slice_eq(&y, &[5.0, 18.0, 23.0], 1e-12, "spmv_f64");
}

// ---------------------------------------------------------------------------
// Test 4: spmm_f64 with identity matrix
// ---------------------------------------------------------------------------

/// SpMM(I, B) = B for f64: multiplying any dense matrix by a sparse identity
/// must reproduce the original matrix.
#[test]
fn spmm_f64_identity_times_vector() {
    let identity: SparseCsr<f64> =
        SparseCsr::from_triplets(3, 3, &[0, 1, 2], &[0, 1, 2], &[1.0_f64, 1.0, 1.0]).unwrap();

    // B is a 3×2 row-major dense matrix.
    let b = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut c = vec![0.0_f64; 6]; // 3×2 output

    spmm_f64(&identity, &b, 2, 1.0, 0.0, &mut c).unwrap();

    // C = I * B = B
    assert_f64_slice_eq(&c, &b, 1e-12, "spmm_f64 identity");
}

// ---------------------------------------------------------------------------
// Test 5: spmv_batched equals column-wise spmv
// ---------------------------------------------------------------------------

/// Batched SpMV(A, [x1|x2]) must give the same result as SpMV(A, x1) and
/// SpMV(A, x2) called individually.
#[test]
fn spmv_batched_equals_column_wise() {
    use tensorlogic_oxicuda_sparse::spmv;

    // 3×3 diagonal matrix diag(2, 3, 4).
    let a: SparseCsr<f32> =
        SparseCsr::from_triplets(3, 3, &[0, 1, 2], &[0, 1, 2], &[2.0_f32, 3.0, 4.0]).unwrap();

    let x1 = [1.0_f32, 0.0, 0.0];
    let x2 = [0.0_f32, 1.0, 0.0];

    // Expected results from individual SpMV.
    let mut y1_expected = vec![0.0_f32; 3];
    let mut y2_expected = vec![0.0_f32; 3];
    spmv(&a, &x1, 1.0, 0.0, &mut y1_expected).unwrap();
    spmv(&a, &x2, 1.0, 0.0, &mut y2_expected).unwrap();

    // Build x_batch and y_batch in column-major (row-major with stride=2).
    // x_batch has shape (3, 2) row-major: row i, col j → index i*2 + j
    let x_batch = vec![
        x1[0], x2[0], // row 0
        x1[1], x2[1], // row 1
        x1[2], x2[2], // row 2
    ];
    let mut y_batch = vec![0.0_f32; 6]; // shape (3, 2) row-major

    spmv_batched(&a, &x_batch, 2, 1.0, 0.0, &mut y_batch).unwrap();

    // Extract columns from y_batch.
    let y1_got = [y_batch[0], y_batch[2], y_batch[4]]; // col 0
    let y2_got = [y_batch[1], y_batch[3], y_batch[5]]; // col 1

    assert_f32_slice_eq(&y1_got, &y1_expected, 1e-6, "batched col 0");
    assert_f32_slice_eq(&y2_got, &y2_expected, 1e-6, "batched col 1");
}

// ---------------------------------------------------------------------------
// Test 6: CSR from_dense round-trip
// ---------------------------------------------------------------------------

/// Build a sparse CSR from a dense matrix, convert back to dense, compare.
#[test]
fn csr_from_dense_roundtrip() {
    #[rustfmt::skip]
    let dense = vec![
        1.0_f32, 0.0,     2.0,
        0.0,     3.0,     0.0,
        4.0,     0.0,     5.0,
    ];

    let a: SparseCsr<f32> = SparseCsr::from_dense(&dense, 3, 3, 0.0).unwrap();
    assert_eq!(a.nnz(), 5, "expected 5 non-zeros");

    let recovered = a.to_dense();
    assert_f32_slice_eq(&recovered, &dense, 1e-7, "from_dense roundtrip");
}

// ---------------------------------------------------------------------------
// Test 7: CSC from_triplets and csc_spmv
// ---------------------------------------------------------------------------

/// Build a small CSC matrix from triplets and verify its SpMV result matches
/// the expected hand-computed values.
///
/// Matrix:
/// [ 1  0  3 ]
/// [ 2  4  0 ]
///
/// x = [1, 1, 1]
/// y = A*x = [4, 6]
#[test]
fn csc_from_triplets_and_spmv() {
    // (row, col, val): (0,0,1), (1,0,2), (1,1,4), (0,2,3)
    let a: SparseCsc<f32> = SparseCsc::from_triplets(
        2,
        3,
        &[0, 1, 1, 0],
        &[0, 0, 1, 2],
        &[1.0_f32, 2.0, 4.0, 3.0],
    )
    .unwrap();

    assert_eq!(a.nnz(), 4);
    assert_eq!(a.shape(), (2, 3));

    let x = vec![1.0_f32, 1.0, 1.0];
    let mut y = vec![0.0_f32; 2];
    a.csc_spmv(&x, 1.0, 0.0, &mut y).unwrap();

    // y[0] = 1*1 + 3*1 = 4
    // y[1] = 2*1 + 4*1 = 6
    assert_f32_slice_eq(&y, &[4.0, 6.0], 1e-6, "csc_spmv");
}

// ---------------------------------------------------------------------------
// Test 8: CSC from CSR transpose
// ---------------------------------------------------------------------------

/// `a.to_csc()` should give the same matrix as `SparseCsc::from_triplets` with
/// the same non-zero pattern.  Verify by comparing dense representations.
#[test]
fn csc_from_csr_transpose() {
    // Original CSR (2×3):
    // [ 1  0  3 ]
    // [ 2  4  0 ]
    let a: SparseCsr<f32> = SparseCsr::from_triplets(
        2,
        3,
        &[0, 0, 1, 1],
        &[0, 2, 0, 1],
        &[1.0_f32, 3.0, 2.0, 4.0],
    )
    .unwrap();

    let csc_from_csr = a.to_csc().unwrap();

    // Build the same matrix directly as CSC.
    let csc_direct: SparseCsc<f32> = SparseCsc::from_triplets(
        2,
        3,
        &[0, 1, 1, 0],
        &[0, 0, 1, 2],
        &[1.0_f32, 2.0, 4.0, 3.0],
    )
    .unwrap();

    let dense_from_csr = csc_from_csr.to_dense();
    let dense_direct = csc_direct.to_dense();

    assert_f32_slice_eq(
        &dense_from_csr,
        &dense_direct,
        1e-6,
        "csc_from_csr vs direct",
    );
    // Also match the original CSR dense.
    let dense_original = a.to_dense();
    assert_f32_slice_eq(
        &dense_from_csr,
        &dense_original,
        1e-6,
        "csc_from_csr vs csr dense",
    );
}

// ---------------------------------------------------------------------------
// Test 9: CSR from_dense threshold drops small values
// ---------------------------------------------------------------------------

/// Values whose absolute value is ≤ threshold must be omitted from the sparse
/// representation, while values strictly above threshold are kept.
#[test]
fn csr_from_dense_threshold_drops_small() {
    // Dense matrix with values 0.001 (should be dropped for threshold=0.01)
    // and 1.0, 2.0 (kept).
    #[rustfmt::skip]
    let dense = vec![
        1.0_f32,  0.001,
        0.001,    2.0_f32,
    ];

    let threshold = 0.01_f32;
    let a: SparseCsr<f32> = SparseCsr::from_dense(&dense, 2, 2, threshold).unwrap();

    // Only the 1.0 and 2.0 entries survive (|0.001| <= 0.01).
    assert_eq!(a.nnz(), 2, "only 2 values exceed threshold");

    // The dense reconstruction should have zeros where small values were.
    let recovered = a.to_dense();
    #[rustfmt::skip]
    let expected = vec![
        1.0_f32, 0.0,
        0.0,     2.0_f32,
    ];
    assert_f32_slice_eq(&recovered, &expected, 1e-7, "threshold filter");
}

// ---------------------------------------------------------------------------
// Test 10: SparseCsc → to_csr() → to_dense() round-trip
// ---------------------------------------------------------------------------

/// Build a CSC matrix, convert to CSR, convert both to dense, and compare.
#[test]
fn csc_to_csr_roundtrip() {
    // 3×3 non-symmetric sparse matrix.
    // [ 0  2  0 ]
    // [ 1  0  4 ]
    // [ 0  3  5 ]
    let csc: SparseCsc<f32> = SparseCsc::from_triplets(
        3,
        3,
        &[1, 0, 2, 1, 2],
        &[0, 1, 1, 2, 2],
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0],
    )
    .unwrap();

    let dense_csc = csc.to_dense();

    let csr = csc.to_csr().unwrap();
    let dense_csr = csr.to_dense();

    assert_f32_slice_eq(&dense_csr, &dense_csc, 1e-6, "csc→csr roundtrip");

    // Spot-check the expected values.
    // dense is row-major 3×3, indices: row*3 + col
    assert!((dense_csr[1] - 2.0).abs() < 1e-6); // (0,1) = 2
    assert!((dense_csr[3] - 1.0).abs() < 1e-6); // (1,0) = 1
    assert!((dense_csr[5] - 4.0).abs() < 1e-6); // (1,2) = 4
    assert!((dense_csr[2 * 3 + 1] - 3.0).abs() < 1e-6); // (2,1) = 3
    assert!((dense_csr[2 * 3 + 2] - 5.0).abs() < 1e-6); // (2,2) = 5

    // Verify nnz.
    assert_eq!(csr.nnz(), 5, "nnz must be preserved through CSC→CSR");
}

// ---------------------------------------------------------------------------
// Error-path tests
// ---------------------------------------------------------------------------

/// from_dense on a wrongly-sized slice must return ShapeMismatch.
#[test]
fn csr_from_dense_size_mismatch() {
    let dense = vec![1.0_f32, 2.0, 3.0]; // only 3 elements, need 4 for 2×2
    let result: Result<SparseCsr<f32>, SparseError> = SparseCsr::from_dense(&dense, 2, 2, 0.0);
    assert!(
        result.is_err(),
        "expected ShapeMismatch for wrong-size dense"
    );
}

/// csc_spmv with wrong x length must return ShapeMismatch.
#[test]
fn csc_spmv_shape_mismatch_x() {
    let a: SparseCsc<f32> = SparseCsc::from_triplets(2, 3, &[0], &[0], &[1.0_f32]).unwrap();
    let x = vec![0.0_f32; 2]; // should be 3
    let mut y = vec![0.0_f32; 2];
    let err = a.csc_spmv(&x, 1.0, 0.0, &mut y);
    assert!(err.is_err(), "expected ShapeMismatch for x.len() mismatch");
}

/// csc_spmv with wrong y length must return ShapeMismatch.
#[test]
fn csc_spmv_shape_mismatch_y() {
    let a: SparseCsc<f32> = SparseCsc::from_triplets(2, 3, &[0], &[0], &[1.0_f32]).unwrap();
    let x = vec![0.0_f32; 3];
    let mut y = vec![0.0_f32; 5]; // should be 2
    let err = a.csc_spmv(&x, 1.0, 0.0, &mut y);
    assert!(err.is_err(), "expected ShapeMismatch for y.len() mismatch");
}

/// spmv_batched with wrong x_batch length must return ShapeMismatch.
#[test]
fn spmv_batched_shape_mismatch() {
    let a: SparseCsr<f32> = SparseCsr::from_triplets(2, 3, &[0], &[0], &[1.0_f32]).unwrap();
    let x_batch = vec![0.0_f32; 4]; // should be 3*2=6 for batch_size=2
    let mut y_batch = vec![0.0_f32; 4];
    let err = spmv_batched(&a, &x_batch, 2, 1.0, 0.0, &mut y_batch);
    assert!(err.is_err(), "expected ShapeMismatch for x_batch length");
}
