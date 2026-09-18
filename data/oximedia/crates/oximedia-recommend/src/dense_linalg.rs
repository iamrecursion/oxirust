//! Pure Rust dense linear algebra types replacing ndarray dependency.
//!
//! Provides `DenseMatrix` (2D: rows x cols) and `DenseVector` (1D) for
//! recommendation algorithms without external linear algebra crates.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Cache-block (tile) size for `matmul` / `matmul_parallel`.
///
/// 64×64 f32 tiles fit comfortably in L1 cache on most hardware
/// (64 × 64 × 4 bytes = 16 KiB per matrix slice; 3 slices = 48 KiB).
const TILE: usize = 64;

/// A 2-dimensional dense matrix backed by a flat `Vec<f32>`.
///
/// This replaces `ndarray::Array2<f32>` for storing factor matrices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseMatrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl DenseMatrix {
    /// Create a new matrix filled with zeros.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0f32; rows * cols],
            rows,
            cols,
        }
    }

    /// Get a value at (row, col).
    #[inline]
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    /// Set a value at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }

    /// Get the number of rows.
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Get a row as a `Vec<f32>`.
    #[must_use]
    pub fn row_vec(&self, row: usize) -> Vec<f32> {
        let start = row * self.cols;
        self.data[start..start + self.cols].to_vec()
    }

    /// Get a row as a slice.
    #[must_use]
    pub fn row_slice(&self, row: usize) -> &[f32] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Get a column as a `Vec<f32>`.
    #[must_use]
    pub fn col_vec(&self, col: usize) -> Vec<f32> {
        (0..self.rows)
            .map(|r| self.data[r * self.cols + col])
            .collect()
    }

    /// Concatenate rows (add rows from another matrix with same number of columns).
    #[must_use]
    pub fn concat_rows(&self, other: &Self) -> Self {
        debug_assert_eq!(self.cols, other.cols);
        let mut new_data = self.data.clone();
        new_data.extend_from_slice(&other.data);
        Self {
            data: new_data,
            rows: self.rows + other.rows,
            cols: self.cols,
        }
    }

    /// Concatenate columns (add columns from another matrix with same number of rows).
    #[must_use]
    pub fn concat_cols(&self, other: &Self) -> Self {
        debug_assert_eq!(self.rows, other.rows);
        let new_cols = self.cols + other.cols;
        let mut new_data = vec![0.0f32; self.rows * new_cols];

        for r in 0..self.rows {
            // Copy self row
            let dst_start = r * new_cols;
            let src_start = r * self.cols;
            new_data[dst_start..dst_start + self.cols]
                .copy_from_slice(&self.data[src_start..src_start + self.cols]);
            // Copy other row
            let other_src_start = r * other.cols;
            new_data[dst_start + self.cols..dst_start + new_cols]
                .copy_from_slice(&other.data[other_src_start..other_src_start + other.cols]);
        }

        Self {
            data: new_data,
            rows: self.rows,
            cols: new_cols,
        }
    }

    /// Cache-blocked (tiled) general matrix multiplication: `self` (M×K) × `rhs` (K×N) → (M×N).
    ///
    /// Uses `TILE`-sized blocks to maximise L1-cache reuse.  The inner
    /// accumulation reads a `TILE×TILE` strip of `self` and `rhs` at a time,
    /// which keeps both strips warm in cache across the `l` loop.
    ///
    /// # Panics
    ///
    /// Panics if `self.ncols() != rhs.nrows()` (inner dimensions must agree).
    #[must_use]
    pub fn matmul(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.cols, rhs.rows,
            "matmul: inner dimensions must match ({} vs {})",
            self.cols, rhs.rows
        );

        let m = self.rows;
        let n = rhs.cols;
        let k = self.cols;
        let mut result = Self::zeros(m, n);

        for i0 in (0..m).step_by(TILE) {
            let i_end = (i0 + TILE).min(m);
            for l0 in (0..k).step_by(TILE) {
                let l_end = (l0 + TILE).min(k);
                for j0 in (0..n).step_by(TILE) {
                    let j_end = (j0 + TILE).min(n);
                    // Inner micro-kernel: accumulate the (i0..i_end, j0..j_end) tile.
                    for i in i0..i_end {
                        for l in l0..l_end {
                            let a = self.data[i * k + l];
                            let res_row_start = i * n;
                            let rhs_row_start = l * n;
                            for j in j0..j_end {
                                result.data[res_row_start + j] += a * rhs.data[rhs_row_start + j];
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Parallel cache-blocked matrix multiplication via `rayon`.
    ///
    /// Partitions the output row-dimension into `TILE`-sized blocks and
    /// processes each block on a separate rayon thread.  The result is
    /// bit-exact with [`Self::matmul`].
    ///
    /// # Panics
    ///
    /// Panics if `self.ncols() != rhs.nrows()` (inner dimensions must agree).
    #[must_use]
    pub fn matmul_parallel(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.cols, rhs.rows,
            "matmul_parallel: inner dimensions must match ({} vs {})",
            self.cols, rhs.rows
        );

        let m = self.rows;
        let n = rhs.cols;
        let k = self.cols;

        // Collect row-block indices; process each block in parallel.
        let row_blocks: Vec<usize> = (0..m).step_by(TILE).collect();

        // Each block computes a flat Vec<f32> for its rows, then we stitch them.
        let blocks: Vec<(usize, usize, Vec<f32>)> = row_blocks
            .par_iter()
            .map(|&i0| {
                let i_end = (i0 + TILE).min(m);
                let block_rows = i_end - i0;
                let mut block = vec![0.0f32; block_rows * n];

                for l0 in (0..k).step_by(TILE) {
                    let l_end = (l0 + TILE).min(k);
                    for j0 in (0..n).step_by(TILE) {
                        let j_end = (j0 + TILE).min(n);
                        for (bi, i) in (i0..i_end).enumerate() {
                            for l in l0..l_end {
                                let a = self.data[i * k + l];
                                let rhs_row_start = l * n;
                                for j in j0..j_end {
                                    block[bi * n + j] += a * rhs.data[rhs_row_start + j];
                                }
                            }
                        }
                    }
                }
                (i0, i_end, block)
            })
            .collect();

        // Stitch blocks into the output matrix.
        let mut result = Self::zeros(m, n);
        for (i0, i_end, block) in blocks {
            let block_rows = i_end - i0;
            let dst_start = i0 * n;
            let dst_end = dst_start + block_rows * n;
            result.data[dst_start..dst_end].copy_from_slice(&block);
        }
        result
    }
}

impl PartialEq for DenseMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
    }
}

/// A 1-dimensional dense vector backed by a `Vec<f32>`.
///
/// This replaces `ndarray::Array1<f32>`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DenseVector {
    data: Vec<f32>,
}

impl DenseVector {
    /// Create a new vector from a `Vec<f32>`.
    #[must_use]
    pub fn from_vec(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Create a vector of zeros.
    #[must_use]
    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![0.0f32; len],
        }
    }

    /// Get the length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a value at index.
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> f32 {
        self.data[index]
    }

    /// Set a value at index.
    #[inline]
    pub fn set(&mut self, index: usize, value: f32) {
        self.data[index] = value;
    }

    /// Get the underlying data as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Convert to a `Vec<f32>`.
    #[must_use]
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }

    /// Iterate over the values.
    pub fn iter(&self) -> std::slice::Iter<'_, f32> {
        self.data.iter()
    }

    /// Compute dot product with another vector.
    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_matrix_zeros() {
        let m = DenseMatrix::zeros(3, 4);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 4);
        assert!((m.get(0, 0) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dense_matrix_set_get() {
        let mut m = DenseMatrix::zeros(3, 3);
        m.set(1, 2, 5.0);
        assert!((m.get(1, 2) - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dense_matrix_row_vec() {
        let mut m = DenseMatrix::zeros(2, 3);
        m.set(1, 0, 1.0);
        m.set(1, 1, 2.0);
        m.set(1, 2, 3.0);
        assert_eq!(m.row_vec(1), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dense_matrix_col_vec() {
        let mut m = DenseMatrix::zeros(3, 2);
        m.set(0, 1, 10.0);
        m.set(1, 1, 20.0);
        m.set(2, 1, 30.0);
        assert_eq!(m.col_vec(1), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_dense_matrix_concat_rows() {
        let m1 = DenseMatrix::zeros(2, 3);
        let m2 = DenseMatrix::zeros(3, 3);
        let result = m1.concat_rows(&m2);
        assert_eq!(result.nrows(), 5);
        assert_eq!(result.ncols(), 3);
    }

    #[test]
    fn test_dense_matrix_concat_cols() {
        let m1 = DenseMatrix::zeros(2, 3);
        let m2 = DenseMatrix::zeros(2, 4);
        let result = m1.concat_cols(&m2);
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 7);
    }

    #[test]
    fn test_dense_vector_from_vec() {
        let v = DenseVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        assert!((v.get(1) - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dense_vector_dot() {
        let v1 = DenseVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = DenseVector::from_vec(vec![4.0, 5.0, 6.0]);
        assert!((v1.dot(&v2) - 32.0).abs() < f32::EPSILON);
    }

    // ---- matmul tests ----

    /// Naive triple-loop reference GEMM for correctness verification.
    fn naive_matmul(a: &DenseMatrix, b: &DenseMatrix) -> DenseMatrix {
        let m = a.nrows();
        let k = a.ncols();
        let n = b.ncols();
        let mut c = DenseMatrix::zeros(m, n);
        for i in 0..m {
            for l in 0..k {
                for j in 0..n {
                    let prev = c.get(i, j);
                    c.set(i, j, prev + a.get(i, l) * b.get(l, j));
                }
            }
        }
        c
    }

    #[test]
    fn test_matmul_square() {
        // 3×3 × 3×3 — compare against naive implementation on known values.
        let mut a = DenseMatrix::zeros(3, 3);
        let mut b = DenseMatrix::zeros(3, 3);
        let vals_a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let vals_b = [9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        for r in 0..3 {
            for c in 0..3 {
                a.set(r, c, vals_a[r * 3 + c]);
                b.set(r, c, vals_b[r * 3 + c]);
            }
        }

        let tiled = a.matmul(&b);
        let naive = naive_matmul(&a, &b);

        assert_eq!(tiled.nrows(), 3);
        assert_eq!(tiled.ncols(), 3);
        for r in 0..3 {
            for c in 0..3 {
                let diff = (tiled.get(r, c) - naive.get(r, c)).abs();
                assert!(
                    diff < 1e-4,
                    "mismatch at ({r},{c}): tiled={}, naive={}",
                    tiled.get(r, c),
                    naive.get(r, c)
                );
            }
        }
    }

    #[test]
    fn test_matmul_non_square() {
        // 2×3 × 3×4 → 2×4
        let mut a = DenseMatrix::zeros(2, 3);
        let mut b = DenseMatrix::zeros(3, 4);
        // Fill with deterministic values.
        for r in 0..2 {
            for c in 0..3 {
                a.set(r, c, (r * 3 + c + 1) as f32);
            }
        }
        for r in 0..3 {
            for c in 0..4 {
                b.set(r, c, (r * 4 + c + 1) as f32);
            }
        }

        let result = a.matmul(&b);
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 4);

        let naive = naive_matmul(&a, &b);
        for r in 0..2 {
            for c in 0..4 {
                let diff = (result.get(r, c) - naive.get(r, c)).abs();
                assert!(diff < 1e-4, "mismatch at ({r},{c})");
            }
        }
    }

    #[test]
    fn test_matmul_non_tile_multiple() {
        // 70×130 × 130×90 — dimensions cross tile boundaries (TILE=64).
        let m = 70usize;
        let k = 130usize;
        let n = 90usize;
        let mut a = DenseMatrix::zeros(m, k);
        let mut b = DenseMatrix::zeros(k, n);
        for r in 0..m {
            for c in 0..k {
                a.set(r, c, ((r + c) % 7) as f32 * 0.1);
            }
        }
        for r in 0..k {
            for c in 0..n {
                b.set(r, c, ((r + c) % 5) as f32 * 0.2);
            }
        }

        let result = a.matmul(&b);
        assert_eq!(result.nrows(), m, "output rows must be {m}");
        assert_eq!(result.ncols(), n, "output cols must be {n}");

        // Spot-check element (0,0) against naive.
        let naive = naive_matmul(&a, &b);
        let diff = (result.get(0, 0) - naive.get(0, 0)).abs();
        assert!(
            diff < 1e-3,
            "spot-check (0,0): tiled={}, naive={}",
            result.get(0, 0),
            naive.get(0, 0)
        );

        // Also verify a corner element.
        let diff2 = (result.get(m - 1, n - 1) - naive.get(m - 1, n - 1)).abs();
        assert!(diff2 < 1e-3, "spot-check ({},{}) failed", m - 1, n - 1);
    }

    #[test]
    fn test_matmul_identity() {
        // A × I = A for any matrix A.
        let mut a = DenseMatrix::zeros(4, 4);
        for r in 0..4 {
            for c in 0..4 {
                a.set(r, c, (r * 4 + c + 1) as f32);
            }
        }

        let mut identity = DenseMatrix::zeros(4, 4);
        for i in 0..4 {
            identity.set(i, i, 1.0);
        }

        let result = a.matmul(&identity);
        for r in 0..4 {
            for c in 0..4 {
                let diff = (result.get(r, c) - a.get(r, c)).abs();
                assert!(diff < f32::EPSILON, "A×I should equal A at ({r},{c})");
            }
        }
    }

    #[test]
    fn test_matmul_parallel_matches_sequential() {
        // matmul_parallel must be bit-exact with matmul.
        let m = 100usize;
        let k = 80usize;
        let n = 60usize;
        let mut a = DenseMatrix::zeros(m, k);
        let mut b = DenseMatrix::zeros(k, n);
        for r in 0..m {
            for c in 0..k {
                a.set(r, c, ((r * k + c) % 11) as f32 * 0.3);
            }
        }
        for r in 0..k {
            for c in 0..n {
                b.set(r, c, ((r * n + c) % 13) as f32 * 0.2);
            }
        }

        let seq = a.matmul(&b);
        let par = a.matmul_parallel(&b);

        assert_eq!(seq.nrows(), par.nrows());
        assert_eq!(seq.ncols(), par.ncols());
        for r in 0..m {
            for c in 0..n {
                let diff = (seq.get(r, c) - par.get(r, c)).abs();
                assert!(diff < 1e-4, "parallel/sequential mismatch at ({r},{c})");
            }
        }
    }
}
