//! Compressed Sparse Row (CSR) matrix stored on the host.
//!
//! [`SparseCsr`] is the primary data structure for this crate.  All index
//! arrays use `i32` (matching the convention of `oxicuda-sparse`) and values
//! use a generic floating-point type `T` (defaulting to `f32`).  The GPU path
//! uploads these arrays to device memory; the CPU path operates on them
//! directly.

use scirs2_core::numeric::{Float, NumCast};

use crate::error::SparseError;

/// A sparse matrix in Compressed Sparse Row (CSR) format.
///
/// Index arrays use `i32` to match the convention of `oxicuda-sparse`.
/// The value type `T` defaults to `f32` for backward compatibility.
/// Storage is on the **host**; the GPU path uploads data to device memory on
/// demand.
///
/// # Invariants
///
/// - `indptr.len() == rows + 1`
/// - `indptr[0] == 0` and `indptr[rows] == nnz as i32`
/// - `indptr` is non-decreasing
/// - `indices.len() == data.len() == nnz`
/// - Every value in `indices` satisfies `0 <= idx < cols`
pub struct SparseCsr<T = f32> {
    /// Number of rows in the matrix.
    pub rows: usize,
    /// Number of columns in the matrix.
    pub cols: usize,
    /// Row pointer array, length `rows + 1`.  `indptr[i]` is the index into
    /// `indices`/`data` where row `i` begins.
    pub(crate) indptr: Vec<i32>,
    /// Column indices of non-zero entries, length `nnz`.
    pub(crate) indices: Vec<i32>,
    /// Non-zero values, length `nnz`.
    pub(crate) data: Vec<T>,
}

impl<T: Float + NumCast + PartialOrd + Copy> SparseCsr<T> {
    /// Constructs a [`SparseCsr`] from coordinate (triplet) format.
    ///
    /// Duplicate `(row, col)` entries are summed together.  The resulting
    /// matrix has the non-zeros stored in column-sorted order within each row.
    ///
    /// # Arguments
    ///
    /// * `rows`    – number of rows.
    /// * `cols`    – number of columns.
    /// * `row_idx` – row index of each non-zero entry (0-indexed).
    /// * `col_idx` – column index of each non-zero entry (0-indexed).
    /// * `values`  – value of each non-zero entry.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::ShapeMismatch`] when `row_idx`, `col_idx`, and
    /// `values` do not all have the same length.
    ///
    /// Returns [`SparseError::IndexError`] when any row or column index is out
    /// of bounds for the declared shape, or when the shape is zero in a
    /// dimension while entries are present.
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        row_idx: &[usize],
        col_idx: &[usize],
        values: &[T],
    ) -> Result<Self, SparseError> {
        // Length consistency check.
        if row_idx.len() != col_idx.len() || row_idx.len() != values.len() {
            return Err(SparseError::ShapeMismatch(format!(
                "row_idx ({}), col_idx ({}), and values ({}) must all have the same length",
                row_idx.len(),
                col_idx.len(),
                values.len(),
            )));
        }

        let nnz_input = row_idx.len();

        // Validate all indices before doing any work.
        for (k, (&r, &c)) in row_idx.iter().zip(col_idx.iter()).enumerate() {
            if r >= rows {
                return Err(SparseError::IndexError(format!(
                    "row_idx[{k}]={r} out of bounds for rows={rows}",
                )));
            }
            if c >= cols {
                return Err(SparseError::IndexError(format!(
                    "col_idx[{k}]={c} out of bounds for cols={cols}",
                )));
            }
        }

        // ------------------------------------------------------------------
        // Build the CSR structure.
        //
        // Strategy:
        //   1. Count non-zeros per row  → indptr (histogram pass).
        //   2. Prefix-sum indptr.
        //   3. Scatter (col, val) pairs into their row buckets, then
        //      sort each row's entries by column index.
        //   4. Sum duplicates within each row.
        // ------------------------------------------------------------------

        // Step 1: count nnz per row.
        let mut indptr = vec![0i32; rows + 1];
        for &r in row_idx {
            indptr[r + 1] += 1;
        }

        // Step 2: prefix sum → indptr[i] = start of row i.
        for i in 0..rows {
            indptr[i + 1] += indptr[i];
        }

        // Step 3: scatter into contiguous buckets.
        //
        // We use a mutable copy of indptr as write cursors.  After this pass,
        // `cursor[i]` equals the original `indptr[i + 1]`.
        let mut cursor = indptr[..rows].to_vec();
        let total_nnz = indptr[rows] as usize;
        let mut raw_col = vec![0i32; total_nnz];
        let mut raw_val = vec![T::zero(); total_nnz];

        for k in 0..nnz_input {
            let r = row_idx[k];
            let pos = cursor[r] as usize;
            raw_col[pos] = col_idx[k] as i32;
            raw_val[pos] = values[k];
            cursor[r] += 1;
        }

        // Step 4: sort each row's entries by column index, then sum duplicates.
        let mut final_col: Vec<i32> = Vec::with_capacity(total_nnz);
        let mut final_val: Vec<T> = Vec::with_capacity(total_nnz);
        let mut new_indptr = vec![0i32; rows + 1];

        for r in 0..rows {
            let start = indptr[r] as usize;
            let end = indptr[r + 1] as usize;

            // Collect (col, val) pairs for this row.
            let mut row_entries: Vec<(i32, T)> = raw_col[start..end]
                .iter()
                .copied()
                .zip(raw_val[start..end].iter().copied())
                .collect();

            // Sort by column index for canonical ordering.
            row_entries.sort_unstable_by_key(|&(c, _)| c);

            // Sum duplicate column entries.
            let mut prev_col: Option<i32> = None;
            for (c, v) in row_entries {
                match prev_col {
                    Some(pc) if pc == c => {
                        // Accumulate into the last entry.
                        let last = final_val.len() - 1;
                        final_val[last] = final_val[last] + v;
                    }
                    _ => {
                        final_col.push(c);
                        final_val.push(v);
                        prev_col = Some(c);
                    }
                }
            }

            new_indptr[r + 1] = final_col.len() as i32;
        }

        Ok(Self {
            rows,
            cols,
            indptr: new_indptr,
            indices: final_col,
            data: final_val,
        })
    }

    /// Returns the number of stored non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Returns the shape of the matrix as `(rows, cols)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Converts the sparse matrix to a dense row-major layout.
    ///
    /// The returned `Vec<T>` has length `rows * cols`.  Element `(i, j)` is
    /// at index `i * cols + j`.
    pub fn to_dense(&self) -> Vec<T> {
        let mut dense = vec![T::zero(); self.rows * self.cols];
        for row in 0..self.rows {
            let start = self.indptr[row] as usize;
            let end = self.indptr[row + 1] as usize;
            for k in start..end {
                let col = self.indices[k] as usize;
                dense[row * self.cols + col] = self.data[k];
            }
        }
        dense
    }

    /// Build a [`SparseCsr`] from a row-major dense matrix, treating values
    /// with `|v| > threshold` as non-zero.
    ///
    /// # Arguments
    ///
    /// * `dense`     – Row-major dense values, length `rows * cols`.
    /// * `rows`      – Number of rows.
    /// * `cols`      – Number of columns.
    /// * `threshold` – Values with absolute value ≤ `threshold` are treated as zero.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::ShapeMismatch`] when `dense.len() != rows * cols`.
    pub fn from_dense(
        dense: &[T],
        rows: usize,
        cols: usize,
        threshold: T,
    ) -> Result<Self, SparseError> {
        if dense.len() != rows * cols {
            return Err(SparseError::ShapeMismatch(format!(
                "from_dense: dense.len()={} but rows*cols={}*{}={}",
                dense.len(),
                rows,
                cols,
                rows * cols,
            )));
        }

        let mut row_idx: Vec<usize> = Vec::new();
        let mut col_idx: Vec<usize> = Vec::new();
        let mut values: Vec<T> = Vec::new();

        for r in 0..rows {
            for c in 0..cols {
                let v = dense[r * cols + c];
                // |v| > threshold
                if v.abs() > threshold {
                    row_idx.push(r);
                    col_idx.push(c);
                    values.push(v);
                }
            }
        }

        Self::from_triplets(rows, cols, &row_idx, &col_idx, &values)
    }

    /// Return the transpose as a new [`SparseCsr`].
    ///
    /// Implemented by collecting `(col, row, val)` triplets from `self`
    /// and rebuilding with rows and cols swapped.
    ///
    /// # Errors
    ///
    /// Propagates any [`SparseError`] from `from_triplets` (in practice this
    /// should never fail for a well-formed matrix).
    pub fn transpose(&self) -> Result<SparseCsr<T>, SparseError> {
        let nnz = self.nnz();
        let mut new_row: Vec<usize> = Vec::with_capacity(nnz);
        let mut new_col: Vec<usize> = Vec::with_capacity(nnz);
        let mut vals: Vec<T> = Vec::with_capacity(nnz);

        for r in 0..self.rows {
            let start = self.indptr[r] as usize;
            let end = self.indptr[r + 1] as usize;
            for k in start..end {
                // In the transpose, the old column becomes the new row.
                new_row.push(self.indices[k] as usize);
                new_col.push(r);
                vals.push(self.data[k]);
            }
        }

        SparseCsr::from_triplets(self.cols, self.rows, &new_row, &new_col, &vals)
    }

    /// Convert this CSR matrix to Compressed Sparse Column (CSC) format.
    ///
    /// # Errors
    ///
    /// Propagates any [`SparseError`] from the CSC builder.
    pub fn to_csc(&self) -> Result<crate::csc::SparseCsc<T>, SparseError> {
        let nnz = self.nnz();
        let mut row_idx: Vec<usize> = Vec::with_capacity(nnz);
        let mut col_idx: Vec<usize> = Vec::with_capacity(nnz);
        let mut vals: Vec<T> = Vec::with_capacity(nnz);

        for r in 0..self.rows {
            let start = self.indptr[r] as usize;
            let end = self.indptr[r + 1] as usize;
            for k in start..end {
                row_idx.push(r);
                col_idx.push(self.indices[k] as usize);
                vals.push(self.data[k]);
            }
        }

        crate::csc::SparseCsc::from_triplets(self.rows, self.cols, &row_idx, &col_idx, &vals)
    }
}
