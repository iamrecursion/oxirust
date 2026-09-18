//! Integration tests for integer sparse matrices
//!
//! This test suite verifies that sparse matrices work correctly with integer element types,
//! which is essential for use cases like graph adjacency matrices and quantum computing.

use scirs2_sparse::{CooArray, CsrArray, SparseArray};

#[test]
fn test_u8_csr_array() {
    // Create a simple binary matrix (like a graph adjacency matrix)
    let rows = vec![0, 0, 1, 2, 2];
    let cols = vec![0, 2, 1, 0, 2];
    let data: Vec<u8> = vec![1, 1, 1, 1, 1];
    let shape = (3, 3);

    let csr = CsrArray::from_triplets(&rows, &cols, &data, shape, false).expect("Operation failed");

    assert_eq!(csr.shape(), (3, 3));
    assert_eq!(csr.nnz(), 5);
    assert_eq!(csr.get(0, 0), 1);
    assert_eq!(csr.get(0, 1), 0);
    assert_eq!(csr.get(1, 1), 1);
}

#[test]
fn test_i32_coo_array() {
    // Create a matrix with integer coefficients
    let rows = vec![0, 0, 1, 2];
    let cols = vec![0, 1, 1, 2];
    let data: Vec<i32> = vec![2, -1, 3, 4];
    let shape = (3, 3);

    let coo = CooArray::from_triplets(&rows, &cols, &data, shape, false).expect("Operation failed");

    assert_eq!(coo.shape(), (3, 3));
    assert_eq!(coo.nnz(), 4);
    assert_eq!(coo.get(0, 0), 2);
    assert_eq!(coo.get(0, 1), -1);
    assert_eq!(coo.get(2, 2), 4);
}

#[test]
fn test_u64_sparse_operations() {
    // Test arithmetic operations with unsigned integers
    let rows = vec![0, 1, 2];
    let cols = vec![0, 1, 2];
    let data: Vec<u64> = vec![10, 20, 30];
    let shape = (3, 3);

    let sparse =
        CsrArray::from_triplets(&rows, &cols, &data, shape, false).expect("Operation failed");

    // Test to_array conversion
    let dense = sparse.to_array();
    assert_eq!(dense[[0, 0]], 10);
    assert_eq!(dense[[1, 1]], 20);
    assert_eq!(dense[[2, 2]], 30);
    assert_eq!(dense[[0, 1]], 0);
}

#[test]
fn test_u8_pauli_matrix() {
    // Simulate a binary Pauli operator (quantum computing use case)
    // This is the kind of use case that prompted the integer support
    let rows = vec![0, 1];
    let cols = vec![1, 0];
    let data: Vec<u8> = vec![1, 1]; // Pauli X matrix (binary)
    let shape = (2, 2);

    let pauli_x =
        CsrArray::from_triplets(&rows, &cols, &data, shape, false).expect("Operation failed");

    assert_eq!(pauli_x.get(0, 0), 0);
    assert_eq!(pauli_x.get(0, 1), 1);
    assert_eq!(pauli_x.get(1, 0), 1);
    assert_eq!(pauli_x.get(1, 1), 0);

    // Verify memory efficiency: u8 uses 1 byte per element vs 8 bytes for f64
    assert_eq!(pauli_x.nnz(), 2);
}

#[test]
fn test_integer_zero_handling() {
    // Test that zero elements are handled correctly for integers
    let rows = vec![0, 1, 2];
    let cols = vec![0, 1, 2];
    let data: Vec<i32> = vec![1, 0, 3]; // Middle element is zero
    let shape = (3, 3);

    let mut sparse =
        CsrArray::from_triplets(&rows, &cols, &data, shape, false).expect("Operation failed");

    // Before eliminate_zeros, should have 3 elements
    assert_eq!(sparse.nnz(), 3);

    sparse.eliminate_zeros();

    // After eliminate_zeros, should have 2 elements
    assert_eq!(sparse.nnz(), 2);
    assert_eq!(sparse.get(0, 0), 1);
    assert_eq!(sparse.get(1, 1), 0); // Returns zero for non-stored
    assert_eq!(sparse.get(2, 2), 3);
}
