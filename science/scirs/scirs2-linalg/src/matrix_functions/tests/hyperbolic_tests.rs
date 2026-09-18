//! Tests for hyperbolic matrix functions

use super::super::hyperbolic::*;
use scirs2_core::ndarray::array;

#[test]
fn test_coshm_zero() {
    let zero = array![[0.0_f64, 0.0], [0.0, 0.0]];
    let result = coshm(&zero.view()).expect("Test: operation failed");
    let identity = array![[1.0_f64, 0.0], [0.0, 1.0]];

    for i in 0..2 {
        for j in 0..2 {
            assert!((result[[i, j]] - identity[[i, j]]).abs() < 1e-10);
        }
    }
}

#[test]
fn test_sinhm_zero() {
    let zero = array![[0.0_f64, 0.0], [0.0, 0.0]];
    let result = sinhm(&zero.view()).expect("Test: operation failed");

    for i in 0..2 {
        for j in 0..2 {
            assert!(result[[i, j]].abs() < 1e-10);
        }
    }
}
