// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Lossless conversions between the CPU reference CSR matrix of
//! [`sparse_gpu`](crate::sparse_gpu) and the canonical
//! [`oxiphysics_core::sparse::CsrMatrix`].
//!
//! The two store the same arrays (row pointers, column indices, values); only the names
//! of the dimensions differ (`n_rows`/`n_cols` here, `nrows`/`ncols` in the core type).
//! The conversion by value moves the arrays without copying them and the conversion of a
//! reference clones them.  The data are not inspected: whatever invariants the source
//! holds are carried over.

use crate::sparse_gpu::CsrMatrix;
use oxiphysics_core::sparse::CsrMatrix as CoreCsr;

impl From<CoreCsr> for CsrMatrix {
    fn from(matrix: CoreCsr) -> Self {
        Self {
            n_rows: matrix.nrows,
            n_cols: matrix.ncols,
            row_ptr: matrix.row_ptr,
            col_idx: matrix.col_idx,
            values: matrix.values,
        }
    }
}

impl From<&CoreCsr> for CsrMatrix {
    fn from(matrix: &CoreCsr) -> Self {
        Self {
            n_rows: matrix.nrows,
            n_cols: matrix.ncols,
            row_ptr: matrix.row_ptr.clone(),
            col_idx: matrix.col_idx.clone(),
            values: matrix.values.clone(),
        }
    }
}

impl From<CsrMatrix> for CoreCsr {
    fn from(matrix: CsrMatrix) -> Self {
        Self {
            nrows: matrix.n_rows,
            ncols: matrix.n_cols,
            row_ptr: matrix.row_ptr,
            col_idx: matrix.col_idx,
            values: matrix.values,
        }
    }
}

impl From<&CsrMatrix> for CoreCsr {
    fn from(matrix: &CsrMatrix) -> Self {
        Self {
            nrows: matrix.n_rows,
            ncols: matrix.n_cols,
            row_ptr: matrix.row_ptr.clone(),
            col_idx: matrix.col_idx.clone(),
            values: matrix.values.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CsrMatrix {
        CsrMatrix::from_dense(&[
            vec![2.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, -3.0, 0.0, 5.0],
        ])
    }

    #[test]
    fn conversion_to_the_core_type_preserves_shape_and_products() {
        let gpu = sample();
        let x = [1.0, 2.0, 3.0, 4.0];
        let expected = gpu.spmv(&x);
        assert_eq!(expected, vec![5.0, 0.0, 14.0]);

        let core = CoreCsr::from(&gpu);
        assert_eq!((core.nrows, core.ncols), (3, 4));
        assert_eq!(core.matvec(&x), expected);
        assert_eq!(core.get(2, 1), -3.0);

        let back = CsrMatrix::from(core);
        assert_eq!((back.n_rows, back.n_cols), (3, 4));
        assert_eq!(back.row_ptr, gpu.row_ptr);
        assert_eq!(back.col_idx, gpu.col_idx);
        assert_eq!(back.values, gpu.values);
        assert_eq!(back.spmv(&x), expected);
    }

    #[test]
    fn conversions_by_value_move_the_arrays_and_by_reference_clone_them() {
        let gpu = sample();
        let values_ptr = gpu.values.as_ptr();
        let borrowed = CoreCsr::from(&gpu);
        assert_ne!(borrowed.values.as_ptr(), values_ptr);
        assert_eq!(borrowed.values, gpu.values);

        let moved = CoreCsr::from(gpu);
        assert_eq!(moved.values.as_ptr(), values_ptr);

        let again = CsrMatrix::from(&moved);
        assert_eq!(again.col_idx, moved.col_idx);
        assert_eq!(again.nnz(), 4);
    }
}
