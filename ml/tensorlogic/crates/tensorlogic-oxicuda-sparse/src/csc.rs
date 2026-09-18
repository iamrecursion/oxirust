//! Compressed Sparse Column (CSC) format.
//!
//! [`SparseCsc`] is the column-oriented counterpart to [`crate::SparseCsr`].  Within
//! column `j`, `indptr[j]..indptr[j+1]` gives the slice of `indices` (row
//! indices) and `data` (values) for all non-zero entries in that column.

use scirs2_core::numeric::{Float, NumCast};

use crate::error::SparseError;

/// A sparse matrix in Compressed Sparse Column (CSC) format.
///
/// Analogous to [`SparseCsr`](crate::csr::SparseCsr) but column-indexed:
/// `indptr[j]..indptr[j+1]` gives the range of non-zero entries in column
/// `j`, with their row indices stored in `indices` and values in `data`.
///
/// Index arrays use `i32` to match the convention of `oxicuda-sparse`.
/// The value type `T` defaults to `f32` for backward compatibility.
///
/// # Invariants
///
/// - `indptr.len() == cols + 1`
/// - `indptr[0] == 0` and `indptr[cols] == nnz as i32`
/// - `indptr` is non-decreasing
/// - `indices.len() == data.len() == nnz`
/// - Every value in `indices` satisfies `0 <= idx < rows`
pub struct SparseCsc<T = f32> {
    /// Number of rows in the matrix.
    pub rows: usize,
    /// Number of columns in the matrix.
    pub cols: usize,
    /// Column pointer array, length `cols + 1`.  `indptr[j]` is the index
    /// into `indices`/`data` where column `j` begins.
    pub(crate) indptr: Vec<i32>,
    /// Row indices of non-zero entries, length `nnz`.
    pub(crate) indices: Vec<i32>,
    /// Non-zero values, length `nnz`.
    pub(crate) data: Vec<T>,
}

impl<T: Float + NumCast + PartialOrd + Copy> SparseCsc<T> {
    /// Constructs a [`SparseCsc`] from coordinate (triplet) format.
    ///
    /// Duplicate `(row, col)` entries are summed together.  The resulting
    /// matrix has the non-zeros stored in row-sorted order within each column.
    ///
    /// # Arguments
    ///
    /// * `rows`    – number of rows.
    /// * `cols`    – number of columns.
    /// * `row_ind` – row index of each non-zero entry (0-indexed).
    /// * `col_ind` – column index of each non-zero entry (0-indexed).
    /// * `values`  – value of each non-zero entry.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::ShapeMismatch`] when the three slices do not all
    /// have the same length.
    ///
    /// Returns [`SparseError::IndexError`] when any row or column index is out
    /// of bounds for the declared shape.
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        row_ind: &[usize],
        col_ind: &[usize],
        values: &[T],
    ) -> Result<Self, SparseError> {
        // Length consistency check.
        if row_ind.len() != col_ind.len() || row_ind.len() != values.len() {
            return Err(SparseError::ShapeMismatch(format!(
                "row_ind ({}), col_ind ({}), and values ({}) must all have the same length",
                row_ind.len(),
                col_ind.len(),
                values.len(),
            )));
        }

        let nnz_input = row_ind.len();

        // Validate all indices before doing any work.
        for (k, (&r, &c)) in row_ind.iter().zip(col_ind.iter()).enumerate() {
            if r >= rows {
                return Err(SparseError::IndexError(format!(
                    "row_ind[{k}]={r} out of bounds for rows={rows}",
                )));
            }
            if c >= cols {
                return Err(SparseError::IndexError(format!(
                    "col_ind[{k}]={c} out of bounds for cols={cols}",
                )));
            }
        }

        // ------------------------------------------------------------------
        // Build the CSC structure using a counting-sort approach.
        //
        // Strategy (analogous to CSR but histogramming over columns):
        //   1. Count non-zeros per column  → indptr (histogram pass).
        //   2. Prefix-sum indptr.
        //   3. Scatter (row, val) pairs into their column buckets, then
        //      sort each column's entries by row index.
        //   4. Sum duplicates within each column.
        // ------------------------------------------------------------------

        // Step 1: count nnz per column.
        let mut indptr = vec![0i32; cols + 1];
        for &c in col_ind {
            indptr[c + 1] += 1;
        }

        // Step 2: prefix sum → indptr[j] = start of column j.
        for j in 0..cols {
            indptr[j + 1] += indptr[j];
        }

        // Step 3: scatter into contiguous buckets using cursor array.
        let mut cursor = indptr[..cols].to_vec();
        let total_nnz = indptr[cols] as usize;
        let mut raw_row = vec![0i32; total_nnz];
        let mut raw_val = vec![T::zero(); total_nnz];

        for k in 0..nnz_input {
            let c = col_ind[k];
            let pos = cursor[c] as usize;
            raw_row[pos] = row_ind[k] as i32;
            raw_val[pos] = values[k];
            cursor[c] += 1;
        }

        // Step 4: sort each column's entries by row index, then sum duplicates.
        let mut final_row: Vec<i32> = Vec::with_capacity(total_nnz);
        let mut final_val: Vec<T> = Vec::with_capacity(total_nnz);
        let mut new_indptr = vec![0i32; cols + 1];

        for j in 0..cols {
            let start = indptr[j] as usize;
            let end = indptr[j + 1] as usize;

            // Collect (row, val) pairs for this column.
            let mut col_entries: Vec<(i32, T)> = raw_row[start..end]
                .iter()
                .copied()
                .zip(raw_val[start..end].iter().copied())
                .collect();

            // Sort by row index for canonical ordering.
            col_entries.sort_unstable_by_key(|&(r, _)| r);

            // Sum duplicate row entries within this column.
            let mut prev_row: Option<i32> = None;
            for (r, v) in col_entries {
                match prev_row {
                    Some(pr) if pr == r => {
                        // Accumulate into the last entry.
                        let last = final_val.len() - 1;
                        final_val[last] = final_val[last] + v;
                    }
                    _ => {
                        final_row.push(r);
                        final_val.push(v);
                        prev_row = Some(r);
                    }
                }
            }

            new_indptr[j + 1] = final_row.len() as i32;
        }

        Ok(Self {
            rows,
            cols,
            indptr: new_indptr,
            indices: final_row,
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
        for j in 0..self.cols {
            let start = self.indptr[j] as usize;
            let end = self.indptr[j + 1] as usize;
            for k in start..end {
                let row = self.indices[k] as usize;
                dense[row * self.cols + j] = self.data[k];
            }
        }
        dense
    }

    /// Sparse matrix-vector product: `y = alpha * A * x + beta * y`.
    ///
    /// Column-oriented: first apply `y *= beta`, then for each column `j` and
    /// each entry `k` in that column, accumulate `alpha * data[k] * x[j]` into
    /// `y[indices[k]]`.
    ///
    /// # Arguments
    ///
    /// * `x`     – Input dense vector; must have length `self.cols`.
    /// * `alpha` – Scalar multiplier for `A * x`.
    /// * `beta`  – Scalar multiplier for the current content of `y`.
    /// * `y`     – In-out dense vector; must have length `self.rows`.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::ShapeMismatch`] when lengths are inconsistent.
    pub fn csc_spmv(&self, x: &[T], alpha: T, beta: T, y: &mut [T]) -> Result<(), SparseError> {
        if x.len() != self.cols {
            return Err(SparseError::ShapeMismatch(format!(
                "csc_spmv: x.len()={} but A.cols={}",
                x.len(),
                self.cols,
            )));
        }
        if y.len() != self.rows {
            return Err(SparseError::ShapeMismatch(format!(
                "csc_spmv: y.len()={} but A.rows={}",
                y.len(),
                self.rows,
            )));
        }

        // Scale y by beta first.
        for y_i in y.iter_mut() {
            *y_i = beta * *y_i;
        }

        // Column-oriented accumulation: for each column j, scatter contributions.
        for (j, &x_j) in x.iter().enumerate() {
            let start = self.indptr[j] as usize;
            let end = self.indptr[j + 1] as usize;
            for k in start..end {
                let row = self.indices[k] as usize;
                y[row] = y[row] + alpha * self.data[k] * x_j;
            }
        }

        Ok(())
    }

    /// Convert this CSC matrix to Compressed Sparse Row (CSR) format.
    ///
    /// # Errors
    ///
    /// Propagates any [`SparseError`] from the CSR builder.
    pub fn to_csr(&self) -> Result<crate::csr::SparseCsr<T>, SparseError> {
        let nnz = self.nnz();
        let mut row_idx: Vec<usize> = Vec::with_capacity(nnz);
        let mut col_idx: Vec<usize> = Vec::with_capacity(nnz);
        let mut vals: Vec<T> = Vec::with_capacity(nnz);

        for j in 0..self.cols {
            let start = self.indptr[j] as usize;
            let end = self.indptr[j + 1] as usize;
            for k in start..end {
                row_idx.push(self.indices[k] as usize);
                col_idx.push(j);
                vals.push(self.data[k]);
            }
        }

        crate::csr::SparseCsr::from_triplets(self.rows, self.cols, &row_idx, &col_idx, &vals)
    }

    /// Build a [`SparseCsc`] from a row-major dense matrix, treating values
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
}
