// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Lossless conversions between the softbody CSR matrix and the canonical
//! [`oxiphysics_core::sparse::CsrMatrix`].
//!
//! Both store `nrows`, `ncols`, the row pointers (`row_ptr`, length `nrows + 1`), the
//! column indices (`col_idx`) and the values in the same layout, so the conversion by
//! value moves the arrays without copying them and the conversion of a reference clones
//! them.  The data are not inspected: whatever invariants the source holds (sorted
//! columns, no duplicate entries) are carried over.

use super::types::CsrMatrix;
use oxiphysics_core::sparse::CsrMatrix as CoreCsr;

impl From<CoreCsr> for CsrMatrix {
    fn from(matrix: CoreCsr) -> Self {
        Self {
            nrows: matrix.nrows,
            ncols: matrix.ncols,
            row_ptr: matrix.row_ptr,
            col_idx: matrix.col_idx,
            values: matrix.values,
        }
    }
}

impl From<&CoreCsr> for CsrMatrix {
    fn from(matrix: &CoreCsr) -> Self {
        Self {
            nrows: matrix.nrows,
            ncols: matrix.ncols,
            row_ptr: matrix.row_ptr.clone(),
            col_idx: matrix.col_idx.clone(),
            values: matrix.values.clone(),
        }
    }
}

impl From<CsrMatrix> for CoreCsr {
    fn from(matrix: CsrMatrix) -> Self {
        Self {
            nrows: matrix.nrows,
            ncols: matrix.ncols,
            row_ptr: matrix.row_ptr,
            col_idx: matrix.col_idx,
            values: matrix.values,
        }
    }
}

impl From<&CsrMatrix> for CoreCsr {
    fn from(matrix: &CsrMatrix) -> Self {
        Self {
            nrows: matrix.nrows,
            ncols: matrix.ncols,
            row_ptr: matrix.row_ptr.clone(),
            col_idx: matrix.col_idx.clone(),
            values: matrix.values.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3 x 3 matrix `[[4, -1, 0], [-1, 4, -1], [0, -1, 4]]` with a duplicate COO entry.
    fn sample() -> CsrMatrix {
        CsrMatrix::from_coo(
            3,
            3,
            &[0, 0, 1, 1, 1, 2, 2, 2],
            &[0, 1, 0, 1, 2, 1, 2, 2],
            &[4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 1.0, 3.0],
        )
    }

    #[test]
    fn conversion_to_the_core_type_preserves_the_matrix() {
        let soft = sample();
        let x = [1.0, 2.0, 3.0];
        let expected = soft.matvec(&x);
        assert_eq!(expected, vec![2.0, 4.0, 10.0]);

        let core = CoreCsr::from(&soft);
        assert_eq!(core.matvec(&x), expected);
        assert_eq!(core.get(1, 2), -1.0);
        assert_eq!(core.nnz(), soft.values.len());

        let back = CsrMatrix::from(core);
        assert_eq!(back.nrows, 3);
        assert_eq!(back.ncols, 3);
        assert_eq!(back.row_ptr, soft.row_ptr);
        assert_eq!(back.col_idx, soft.col_idx);
        assert_eq!(back.values, soft.values);
        assert_eq!(back.matvec(&x), expected);
    }

    #[test]
    fn conversions_by_value_move_the_arrays_and_by_reference_clone_them() {
        let soft = sample();
        let values_ptr = soft.values.as_ptr();
        let borrowed = CoreCsr::from(&soft);
        assert_ne!(borrowed.values.as_ptr(), values_ptr);
        assert_eq!(borrowed.values, soft.values);

        let moved = CoreCsr::from(soft);
        assert_eq!(moved.values.as_ptr(), values_ptr);

        let again = CsrMatrix::from(&moved);
        assert_eq!(again.col_idx, moved.col_idx);
        assert_eq!(again.diagonal(), vec![4.0, 4.0, 4.0]);
    }
}
