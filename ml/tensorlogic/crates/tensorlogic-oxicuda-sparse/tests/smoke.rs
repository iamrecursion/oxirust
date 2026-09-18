//! Smoke tests for tensorlogic-oxicuda-sparse (CPU path, no GPU required).
//!
//! These tests exercise the public API of `SparseCsr`, `spmv`, and `spmm`
//! without relying on any CUDA infrastructure — they run in CI under the
//! default `cpu` feature.

use tensorlogic_oxicuda_sparse::{spmm, spmv, SparseCsr};

// ---------------------------------------------------------------------------
// SparseCsr construction tests
// ---------------------------------------------------------------------------

/// Identity 3×3 matrix: from_triplets + spmv gives back the same vector.
#[test]
fn csr_from_triplets_and_spmv() {
    let a = SparseCsr::from_triplets(3, 3, &[0, 1, 2], &[0, 1, 2], &[1.0, 1.0, 1.0]).unwrap();

    assert_eq!(a.nnz(), 3);
    assert_eq!(a.shape(), (3, 3));

    let x = vec![1.0f32, 2.0, 3.0];
    let mut y = vec![0.0f32; 3];
    spmv(&a, &x, 1.0, 0.0, &mut y).unwrap();

    assert_eq!(y, vec![1.0, 2.0, 3.0]);
}

/// Verify alpha/beta scaling: y = alpha*A*x + beta*y_init.
#[test]
fn spmv_alpha_beta() {
    // Diagonal matrix diag(2, 3).
    let a = SparseCsr::from_triplets(2, 2, &[0, 1], &[0, 1], &[2.0, 3.0]).unwrap();

    let x = vec![1.0f32, 1.0];
    let mut y = vec![10.0f32, 10.0];

    spmv(&a, &x, 1.0, 0.5, &mut y).unwrap();

    // y[0] = 1.0 * 2.0 * 1.0 + 0.5 * 10.0 = 7.0
    // y[1] = 1.0 * 3.0 * 1.0 + 0.5 * 10.0 = 8.0
    assert!((y[0] - 7.0).abs() < 1e-5, "y[0]={}", y[0]);
    assert!((y[1] - 8.0).abs() < 1e-5, "y[1]={}", y[1]);
}

/// to_dense must produce the correct row-major dense representation.
#[test]
fn to_dense_roundtrip() {
    // 2×3 matrix:
    //   row 0: (0,1) = 5
    //   row 1: (1,0) = 6, (1,2) = 7
    let a = SparseCsr::from_triplets(2, 3, &[0, 1, 1], &[1, 0, 2], &[5.0, 6.0, 7.0]).unwrap();

    let dense = a.to_dense();
    assert_eq!(dense, vec![0.0, 5.0, 0.0, 6.0, 0.0, 7.0]);
}

// ---------------------------------------------------------------------------
// Edge-case tests
// ---------------------------------------------------------------------------

/// A matrix with no non-zeros is valid (empty slices, nnz = 0).
#[test]
fn empty_matrix_spmv() {
    let a = SparseCsr::from_triplets(4, 4, &[], &[], &[]).unwrap();
    assert_eq!(a.nnz(), 0);

    let x = vec![1.0f32; 4];
    let mut y = vec![5.0f32; 4];

    // y = 1.0 * 0 * x + 0.5 * y  →  all 2.5
    spmv(&a, &x, 1.0, 0.5, &mut y).unwrap();
    for &v in &y {
        assert!((v - 2.5).abs() < 1e-6, "expected 2.5 got {v}");
    }
}

/// Duplicate (row, col) entries must be summed.
#[test]
fn duplicate_entries_are_summed() {
    // (0,0) appears twice with values 1.0 and 2.0 → should give 3.0.
    let a = SparseCsr::from_triplets(2, 2, &[0, 0, 1], &[0, 0, 1], &[1.0, 2.0, 4.0]).unwrap();
    assert_eq!(a.nnz(), 2, "duplicates should be merged");

    let x = vec![1.0f32, 1.0];
    let mut y = vec![0.0f32; 2];
    spmv(&a, &x, 1.0, 0.0, &mut y).unwrap();

    // y[0] = (1+2)*1 = 3, y[1] = 4*1 = 4
    assert!((y[0] - 3.0).abs() < 1e-6, "y[0]={}", y[0]);
    assert!((y[1] - 4.0).abs() < 1e-6, "y[1]={}", y[1]);
}

/// Shape-mismatch errors are correctly reported.
#[test]
fn spmv_shape_mismatch_x() {
    let a = SparseCsr::from_triplets(2, 3, &[0], &[0], &[1.0]).unwrap();
    let x = vec![0.0f32; 2]; // wrong length
    let mut y = vec![0.0f32; 2];
    let err = spmv(&a, &x, 1.0, 0.0, &mut y);
    assert!(err.is_err(), "expected ShapeMismatch error for x length");
}

#[test]
fn spmv_shape_mismatch_y() {
    let a = SparseCsr::from_triplets(2, 3, &[0], &[0], &[1.0]).unwrap();
    let x = vec![0.0f32; 3];
    let mut y = vec![0.0f32; 5]; // wrong length
    let err = spmv(&a, &x, 1.0, 0.0, &mut y);
    assert!(err.is_err(), "expected ShapeMismatch error for y length");
}

/// from_triplets rejects out-of-bounds row index.
#[test]
fn from_triplets_oob_row() {
    let err = SparseCsr::from_triplets(2, 2, &[5], &[0], &[1.0]);
    assert!(err.is_err(), "expected IndexError for row=5 in 2×2 matrix");
}

/// from_triplets rejects out-of-bounds column index.
#[test]
fn from_triplets_oob_col() {
    let err = SparseCsr::from_triplets(2, 2, &[0], &[9], &[1.0]);
    assert!(err.is_err(), "expected IndexError for col=9 in 2×2 matrix");
}

/// from_triplets rejects mismatched triplet slice lengths.
#[test]
fn from_triplets_mismatched_lengths() {
    let err = SparseCsr::from_triplets(2, 2, &[0, 1], &[0], &[1.0, 2.0]);
    assert!(
        err.is_err(),
        "expected ShapeMismatch for mismatched triplet lengths"
    );
}

// ---------------------------------------------------------------------------
// SpMM tests
// ---------------------------------------------------------------------------

/// Identity A, identity B → C = identity (row-major).
#[test]
fn spmm_identity_times_identity() {
    // 3×3 identity sparse.
    let a = SparseCsr::from_triplets(3, 3, &[0, 1, 2], &[0, 1, 2], &[1.0, 1.0, 1.0]).unwrap();

    // Dense B = 3×3 identity (row-major).
    #[rustfmt::skip]
    let b = vec![
        1.0f32, 0.0, 0.0,
        0.0,    1.0, 0.0,
        0.0,    0.0, 1.0,
    ];
    let mut c = vec![0.0f32; 9];

    spmm(&a, &b, 3, 1.0, 0.0, &mut c).unwrap();

    // C should equal B.
    for (i, (&got, &exp)) in c.iter().zip(b.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "c[{i}]: got {got}, expected {exp}"
        );
    }
}

/// SpMM with alpha/beta scaling.
#[test]
fn spmm_alpha_beta_scaling() {
    // A = 2×2 diagonal diag(2, 3)
    let a = SparseCsr::from_triplets(2, 2, &[0, 1], &[0, 1], &[2.0, 3.0]).unwrap();

    // B = [[1, 0], [0, 1]] (identity, row-major)
    let b = vec![1.0f32, 0.0, 0.0, 1.0];
    // C_init = [[1, 1], [1, 1]]
    let mut c = vec![1.0f32; 4];

    // C = 2.0 * A * B + 0.5 * C_init
    // A * B = [[2, 0], [0, 3]]
    // 2.0 * [[2,0],[0,3]] = [[4,0],[0,6]]
    // 0.5 * [[1,1],[1,1]] = [[0.5,0.5],[0.5,0.5]]
    // C = [[4.5, 0.5], [0.5, 6.5]]
    spmm(&a, &b, 2, 2.0, 0.5, &mut c).unwrap();

    assert!((c[0] - 4.5).abs() < 1e-5, "c[0]={}", c[0]);
    assert!((c[1] - 0.5).abs() < 1e-5, "c[1]={}", c[1]);
    assert!((c[2] - 0.5).abs() < 1e-5, "c[2]={}", c[2]);
    assert!((c[3] - 6.5).abs() < 1e-5, "c[3]={}", c[3]);
}

/// SpMM shape-mismatch error.
#[test]
fn spmm_shape_mismatch_b() {
    let a = SparseCsr::from_triplets(2, 3, &[0], &[0], &[1.0]).unwrap();
    let b = vec![0.0f32; 4]; // should be 3 * b_cols = 6 for b_cols=2
    let mut c = vec![0.0f32; 4];
    let err = spmm(&a, &b, 2, 1.0, 0.0, &mut c);
    assert!(err.is_err(), "expected ShapeMismatch for B length");
}
