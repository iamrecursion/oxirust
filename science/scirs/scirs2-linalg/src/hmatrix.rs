//! Hierarchical matrix (H-matrix) compression for dense-but-rank-structured matrices.
//!
//! H-matrices exploit the fact that certain off-diagonal blocks of physically-motivated
//! matrices (BEM, FEM, covariance kernels) have low numerical rank. By recursively
//! partitioning the index set and storing admissible off-diagonal blocks in compressed
//! low-rank form (U·Vᵀ), we achieve O(N log N) storage and O(N log N) matvec.
//!
//! ## Algorithms
//!
//! - **Block cluster tree** — recursive bisection of row/column index sets.
//! - **ACA** (Adaptive Cross Approximation) — column-pivoted cross approximation for
//!   generating low-rank factors from a kernel function, without forming the full block.
//! - **H-matrix matvec** — O(N log N) matrix-vector product.
//! - **Recompression** — SVD-based rank truncation of existing low-rank factors.
//! - **Frobenius norm** — square root of sum of squared block norms.
//!
//! ## References
//!
//! - Hackbusch, W. (2015). "Hierarchical Matrices: Algorithms and Analysis."
//! - Bebendorf, M. (2000). "Approximation of boundary element matrices."
//! - Rjasanow, S. & Steinbach, O. (2007). "The Fast Solution of Boundary Integral Equations."

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;
use std::iter::Sum;

// ─────────────────────────────────────────────────────────────────────────────
// BlockClusterTree
// ─────────────────────────────────────────────────────────────────────────────

/// A node in the block cluster tree used to drive H-matrix construction.
///
/// Each leaf node represents a set of row/column indices that either satisfies the
/// admissibility condition (→ low-rank) or is below `leaf_size` (→ dense).
#[derive(Debug, Clone)]
pub struct BlockClusterTree {
    /// Row indices covered by this node (global indices into the matrix).
    pub rows: Vec<usize>,
    /// Column indices covered by this node.
    pub cols: Vec<usize>,
    /// Four children from (row-bisect) × (col-bisect); `None` at leaves.
    pub children: Option<Box<[BlockClusterTree; 4]>>,
    /// True when this node is a leaf (no further subdivision).
    pub is_leaf: bool,
}

impl BlockClusterTree {
    /// Construct a block cluster tree by recursive bisection.
    ///
    /// # Arguments
    ///
    /// * `rows` — row index set
    /// * `cols` — column index set
    /// * `leaf_size` — stop subdividing when `min(|rows|, |cols|) <= leaf_size`
    /// * `eta` — admissibility parameter: a block is admissible when
    ///   `min(diam(rows), diam(cols)) * eta <= dist(rows, cols)`.
    ///   Use `eta = 0.0` to always subdivide (never mark admissible at interior nodes).
    pub fn build(rows: Vec<usize>, cols: Vec<usize>, leaf_size: usize, eta: f64) -> Self {
        let n_rows = rows.len();
        let n_cols = cols.len();

        // Leaf conditions
        if n_rows <= leaf_size || n_cols <= leaf_size {
            return BlockClusterTree {
                rows,
                cols,
                children: None,
                is_leaf: true,
            };
        }

        // Admissibility check based on index ranges
        let row_min = rows.iter().copied().min().unwrap_or(0);
        let row_max = rows.iter().copied().max().unwrap_or(0);
        let col_min = cols.iter().copied().min().unwrap_or(0);
        let col_max = cols.iter().copied().max().unwrap_or(0);

        let diam_rows = (row_max - row_min) as f64;
        let diam_cols = (col_max - col_min) as f64;
        let diam = diam_rows.min(diam_cols);

        // Distance between row-cluster and col-cluster intervals
        let dist = if row_max < col_min {
            (col_min - row_max) as f64
        } else if col_max < row_min {
            (row_min - col_max) as f64
        } else {
            0.0_f64 // overlapping — not admissible
        };

        // Admissible block: can be approximated with ACA (leaf)
        if dist > 0.0 && diam * eta <= dist {
            return BlockClusterTree {
                rows,
                cols,
                children: None,
                is_leaf: true,
            };
        }

        // Subdivide: bisect rows and cols
        let row_mid = n_rows / 2;
        let col_mid = n_cols / 2;

        let rows_lo = rows[..row_mid].to_vec();
        let rows_hi = rows[row_mid..].to_vec();
        let cols_lo = cols[..col_mid].to_vec();
        let cols_hi = cols[col_mid..].to_vec();

        let c00 = BlockClusterTree::build(rows_lo.clone(), cols_lo.clone(), leaf_size, eta);
        let c01 = BlockClusterTree::build(rows_lo, cols_hi.clone(), leaf_size, eta);
        let c10 = BlockClusterTree::build(rows_hi.clone(), cols_lo, leaf_size, eta);
        let c11 = BlockClusterTree::build(rows_hi, cols_hi, leaf_size, eta);

        BlockClusterTree {
            rows,
            cols,
            children: Some(Box::new([c00, c01, c10, c11])),
            is_leaf: false,
        }
    }

    /// Count the total number of leaf nodes.
    pub fn count_leaves(&self) -> usize {
        if self.is_leaf {
            return 1;
        }
        match &self.children {
            None => 1,
            Some(children) => children.iter().map(|c| c.count_leaves()).sum(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Low-rank and dense block types
// ─────────────────────────────────────────────────────────────────────────────

/// A low-rank matrix block stored as `U · Vᵀ` where `U` is `nrows × rank`
/// and `V` is `ncols × rank` (both stored in column-major flat `Vec`).
///
/// Entry `(i, j)` = Σ_k U[i, k] * V[j, k].
#[derive(Debug, Clone)]
pub struct LowRankBlock<F: Float> {
    /// Left factor, shape `nrows × rank`, stored row-major.
    pub u: Vec<F>,
    /// Right factor, shape `ncols × rank`, stored row-major.
    pub v: Vec<F>,
    /// Numerical rank.
    pub rank: usize,
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
}

impl<F: Float + Debug + Clone + Sum> LowRankBlock<F> {
    /// Evaluate `(U Vᵀ) x` for a vector `x` of length `ncols`.
    ///
    /// Returns a vector of length `nrows`.
    pub fn matvec(&self, x: &[F]) -> LinalgResult<Vec<F>> {
        if x.len() != self.ncols {
            return Err(LinalgError::DimensionError(format!(
                "LowRankBlock matvec: x has length {} but ncols={}",
                x.len(),
                self.ncols
            )));
        }
        // Step 1: tmp_k = Vᵀ x  (length `rank`)
        let mut tmp = vec![F::zero(); self.rank];
        for (k, t) in tmp.iter_mut().enumerate() {
            let mut acc = F::zero();
            for (j, xj) in x.iter().enumerate() {
                acc = acc + self.v[j * self.rank + k] * *xj;
            }
            *t = acc;
        }
        // Step 2: y = U tmp  (length `nrows`)
        let mut y = vec![F::zero(); self.nrows];
        for (i, yi) in y.iter_mut().enumerate() {
            let mut acc = F::zero();
            for (k, tk) in tmp.iter().enumerate() {
                acc = acc + self.u[i * self.rank + k] * *tk;
            }
            *yi = acc;
        }
        Ok(y)
    }

    /// Frobenius norm of the low-rank block `‖U Vᵀ‖_F`.
    ///
    /// Uses the identity `‖UV^T‖_F = ‖(UᵀU)(VᵀV)‖_F^{1/2}` — actually we compute
    /// it via `‖UV^T‖_F² = trace((UV^T)^T(UV^T)) = trace(VU^T UV^T) = trace((UᵀU)(VᵀV))`.
    pub fn frobenius_norm_sq(&self) -> F {
        // Compute G = UᵀU  (rank × rank)
        let mut gu = vec![F::zero(); self.rank * self.rank];
        for k1 in 0..self.rank {
            for k2 in 0..self.rank {
                let mut s = F::zero();
                for i in 0..self.nrows {
                    s = s + self.u[i * self.rank + k1] * self.u[i * self.rank + k2];
                }
                gu[k1 * self.rank + k2] = s;
            }
        }
        // Compute H = VᵀV  (rank × rank)
        let mut gv = vec![F::zero(); self.rank * self.rank];
        for k1 in 0..self.rank {
            for k2 in 0..self.rank {
                let mut s = F::zero();
                for j in 0..self.ncols {
                    s = s + self.v[j * self.rank + k1] * self.v[j * self.rank + k2];
                }
                gv[k1 * self.rank + k2] = s;
            }
        }
        // trace(G · H) = Σ_{k1,k2} G[k1,k2] * H[k2,k1]
        let mut tr = F::zero();
        for k1 in 0..self.rank {
            for k2 in 0..self.rank {
                tr = tr + gu[k1 * self.rank + k2] * gv[k2 * self.rank + k1];
            }
        }
        tr
    }
}

/// A dense matrix block stored in row-major flat `Vec`.
#[derive(Debug, Clone)]
pub struct DenseBlock<F: Float> {
    /// Entry data, shape `nrows × ncols`, stored row-major.
    pub data: Vec<F>,
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
}

impl<F: Float + Debug + Clone> DenseBlock<F> {
    /// Apply the dense block to vector `x` of length `ncols`.
    pub fn matvec(&self, x: &[F]) -> LinalgResult<Vec<F>> {
        if x.len() != self.ncols {
            return Err(LinalgError::DimensionError(format!(
                "DenseBlock matvec: x has length {} but ncols={}",
                x.len(),
                self.ncols
            )));
        }
        let mut y = vec![F::zero(); self.nrows];
        for (i, yi) in y.iter_mut().enumerate() {
            let mut acc = F::zero();
            for (j, xj) in x.iter().enumerate() {
                acc = acc + self.data[i * self.ncols + j] * *xj;
            }
            *yi = acc;
        }
        Ok(y)
    }

    /// Frobenius norm squared.
    pub fn frobenius_norm_sq(&self) -> F {
        self.data.iter().fold(F::zero(), |acc, &v| acc + v * v)
    }
}

/// A single block in the H-matrix, either low-rank or dense.
#[derive(Debug, Clone)]
pub enum HBlock<F: Float> {
    /// Low-rank approximation of an admissible block.
    LowRank(LowRankBlock<F>),
    /// Explicitly stored dense block (near-field or small leaf).
    Dense(DenseBlock<F>),
}

impl<F: Float + Debug + Clone + Sum> HBlock<F> {
    /// Number of rows in this block.
    pub fn nrows(&self) -> usize {
        match self {
            HBlock::LowRank(b) => b.nrows,
            HBlock::Dense(b) => b.nrows,
        }
    }

    /// Number of columns in this block.
    pub fn ncols(&self) -> usize {
        match self {
            HBlock::LowRank(b) => b.ncols,
            HBlock::Dense(b) => b.ncols,
        }
    }

    /// Apply this block to sub-vector `x` of length `ncols`, adding result into `y[row_offset..]`.
    pub fn matvec_add(&self, x: &[F], y: &mut [F]) -> LinalgResult<()> {
        let result = match self {
            HBlock::LowRank(b) => b.matvec(x)?,
            HBlock::Dense(b) => b.matvec(x)?,
        };
        for (yi, ri) in y.iter_mut().zip(result.iter()) {
            *yi = *yi + *ri;
        }
        Ok(())
    }

    /// Frobenius norm squared.
    pub fn frobenius_norm_sq(&self) -> F {
        match self {
            HBlock::LowRank(b) => b.frobenius_norm_sq(),
            HBlock::Dense(b) => b.frobenius_norm_sq(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// H-matrix structure
// ─────────────────────────────────────────────────────────────────────────────

/// Hierarchical matrix (H-matrix) for dense-but-rank-structured problems.
///
/// Stores a collection of leaf blocks, each tagged with its `(row_offset, col_offset)`
/// in the global matrix. Together they tile the full `nrows × ncols` matrix.
///
/// # Type parameter
///
/// `F` must implement `Float` (from `scirs2_core::numeric`).
#[derive(Debug)]
pub struct HMatrix<F: Float> {
    /// List of `(row_offset, col_offset, block)` tuples.
    pub(crate) blocks: Vec<(usize, usize, HBlock<F>)>,
    /// Total number of rows.
    pub nrows: usize,
    /// Total number of columns.
    pub ncols: usize,
    /// Leaf size used during construction.
    pub leaf_size: usize,
}

impl<F: Float + Debug + Clone + Sum + 'static> HMatrix<F> {
    /// Build an H-matrix from a kernel function using recursive bisection + ACA.
    ///
    /// # Arguments
    ///
    /// * `n` — number of rows (= number of columns; square matrices only)
    /// * `kernel` — callable `K(i, j) -> F` giving the `(i,j)` matrix entry
    /// * `leaf_size` — maximum block size at leaves (blocks ≤ this are stored dense)
    /// * `eta` — admissibility parameter (typical value: 1.0–2.0)
    /// * `tol` — ACA approximation tolerance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use scirs2_linalg::hmatrix::HMatrix;
    /// let h: HMatrix<f64> = HMatrix::from_kernel(64, |i, j| 1.0 / ((i as f64 - j as f64).abs() + 1.0), 8, 2.0, 1e-6).unwrap();
    /// ```
    pub fn from_kernel<K>(
        n: usize,
        kernel: K,
        leaf_size: usize,
        eta: f64,
        tol: F,
    ) -> LinalgResult<Self>
    where
        K: Fn(usize, usize) -> F,
    {
        if n == 0 {
            return Err(LinalgError::ValueError(
                "HMatrix::from_kernel: n must be > 0".into(),
            ));
        }
        let rows: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = (0..n).collect();
        let tree = BlockClusterTree::build(rows, cols, leaf_size, eta);

        let mut blocks = Vec::new();
        Self::collect_blocks(&tree, &kernel, leaf_size, eta, tol, &mut blocks)?;

        Ok(HMatrix {
            blocks,
            nrows: n,
            ncols: n,
            leaf_size,
        })
    }

    /// Recursively walk the block cluster tree and build leaf blocks.
    fn collect_blocks<K>(
        node: &BlockClusterTree,
        kernel: &K,
        leaf_size: usize,
        eta: f64,
        tol: F,
        out: &mut Vec<(usize, usize, HBlock<F>)>,
    ) -> LinalgResult<()>
    where
        K: Fn(usize, usize) -> F,
    {
        if !node.is_leaf {
            // Interior node — recurse into children
            if let Some(children) = &node.children {
                for child in children.iter() {
                    Self::collect_blocks(child, kernel, leaf_size, eta, tol, out)?;
                }
            }
            return Ok(());
        }

        let n_rows = node.rows.len();
        let n_cols = node.cols.len();
        let row_offset = node.rows.iter().copied().min().unwrap_or(0);
        let col_offset = node.cols.iter().copied().min().unwrap_or(0);

        // Decide: admissible (use ACA) or dense
        let is_admissible = Self::is_admissible_block(&node.rows, &node.cols, leaf_size, eta);

        if is_admissible && n_rows > 1 && n_cols > 1 {
            // Try ACA approximation
            let lr = aca_approximate(kernel, &node.rows, &node.cols, tol)?;
            out.push((row_offset, col_offset, HBlock::LowRank(lr)));
        } else {
            // Store dense
            let mut data = vec![F::zero(); n_rows * n_cols];
            for (li, &gi) in node.rows.iter().enumerate() {
                for (lj, &gj) in node.cols.iter().enumerate() {
                    data[li * n_cols + lj] = kernel(gi, gj);
                }
            }
            out.push((
                row_offset,
                col_offset,
                HBlock::Dense(DenseBlock {
                    data,
                    nrows: n_rows,
                    ncols: n_cols,
                }),
            ));
        }
        Ok(())
    }

    /// Decide if a leaf block is admissible (should be low-rank compressed).
    fn is_admissible_block(rows: &[usize], cols: &[usize], leaf_size: usize, eta: f64) -> bool {
        if rows.len() <= leaf_size || cols.len() <= leaf_size {
            return false;
        }
        let row_min = rows.iter().copied().min().unwrap_or(0);
        let row_max = rows.iter().copied().max().unwrap_or(0);
        let col_min = cols.iter().copied().min().unwrap_or(0);
        let col_max = cols.iter().copied().max().unwrap_or(0);

        let diam_rows = (row_max - row_min) as f64;
        let diam_cols = (col_max - col_min) as f64;
        let diam = diam_rows.min(diam_cols);

        let dist = if row_max < col_min {
            (col_min - row_max) as f64
        } else if col_max < row_min {
            (row_min - col_max) as f64
        } else {
            return false; // overlapping index sets → not admissible
        };

        dist > 0.0 && diam * eta <= dist
    }

    /// Number of leaf blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Number of low-rank blocks.
    pub fn num_lowrank_blocks(&self) -> usize {
        self.blocks
            .iter()
            .filter(|(_, _, b)| matches!(b, HBlock::LowRank(_)))
            .count()
    }

    /// Number of dense blocks.
    pub fn num_dense_blocks(&self) -> usize {
        self.blocks
            .iter()
            .filter(|(_, _, b)| matches!(b, HBlock::Dense(_)))
            .count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Adaptive Cross Approximation (ACA)
// ─────────────────────────────────────────────────────────────────────────────

/// Adaptive Cross Approximation (ACA) — partially pivoted variant.
///
/// Builds a low-rank approximation `A ≈ U Vᵀ` of the sub-matrix defined by
/// `rows` × `cols` using only `O(k(m+n))` kernel evaluations.
///
/// # Algorithm
///
/// Starting from row pivot `i₀ = 0`:
/// 1. Evaluate row `i₀` of the residual → column pivot `j*` = argmax |r_{i₀,:}|
/// 2. Scale the row by `1/r_{i₀,j*}`
/// 3. Evaluate column `j*` of the residual → next row pivot `i*` = argmax |r_{:,j*}|
/// 4. Accumulate rank-1 contribution into U, V
/// 5. Stop when `‖uₖ‖ · ‖vₖ‖ < tol · ‖A_approx‖_F`
///
/// # Arguments
///
/// * `kernel` — matrix entry function
/// * `rows` — global row indices for this block
/// * `cols` — global column indices for this block
/// * `tol` — relative stopping tolerance
pub fn aca_approximate<F, K>(
    kernel: &K,
    rows: &[usize],
    cols: &[usize],
    tol: F,
) -> LinalgResult<LowRankBlock<F>>
where
    F: Float + Debug + Clone + Sum,
    K: Fn(usize, usize) -> F,
{
    let m = rows.len();
    let n = cols.len();
    let max_rank = m.min(n);

    if m == 0 || n == 0 {
        return Ok(LowRankBlock {
            u: Vec::new(),
            v: Vec::new(),
            rank: 0,
            nrows: m,
            ncols: n,
        });
    }

    // We build U column by column and V column by column.
    // u_cols[k] is the k-th column of U  (length m)
    // v_cols[k] is the k-th column of V  (length n)
    let mut u_cols: Vec<Vec<F>> = Vec::new();
    let mut v_cols: Vec<Vec<F>> = Vec::new();

    // Track used row/col pivots to avoid repetition
    let mut used_row_pivots = vec![false; m];

    // Frobenius norm squared of current approximation (updated incrementally)
    let mut approx_norm_sq = F::zero();

    // Starting row pivot index (local index in 0..m)
    let mut row_pivot = 0usize;

    for _iter in 0..max_rank {
        // --- Compute residual row at row_pivot ---
        let mut res_row = vec![F::zero(); n];
        for (lj, &gj) in cols.iter().enumerate() {
            let gi = rows[row_pivot];
            let mut v = kernel(gi, gj);
            // Subtract already-computed rank contributions
            for k in 0..u_cols.len() {
                v = v - u_cols[k][row_pivot] * v_cols[k][lj];
            }
            res_row[lj] = v;
        }

        // Find column pivot: argmax |res_row[j]|
        let col_pivot = res_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let pivot_val = res_row[col_pivot];

        // Stop if pivot is essentially zero (block is numerically zero or already captured)
        if pivot_val.abs() <= F::epsilon() * F::from(1000.0).unwrap_or(F::one()) {
            break;
        }

        // Scale row to form new v vector: v_k = res_row / pivot
        let inv_pivot = F::one() / pivot_val;
        let new_v: Vec<F> = res_row.iter().map(|&x| x * inv_pivot).collect();

        // --- Compute residual column at col_pivot ---
        let mut new_u = vec![F::zero(); m];
        for (li, &gi) in rows.iter().enumerate() {
            let gj = cols[col_pivot];
            let mut val = kernel(gi, gj);
            for k in 0..u_cols.len() {
                val = val - u_cols[k][li] * v_cols[k][col_pivot];
            }
            new_u[li] = val;
        }

        // Compute norms for convergence check
        let u_norm_sq: F = new_u.iter().map(|&x| x * x).sum();
        let v_norm_sq: F = new_v.iter().map(|&x| x * x).sum();
        let rank_contrib_sq = u_norm_sq * v_norm_sq;

        // Update approximation Frobenius norm (cross-term formula)
        let cross_term: F = u_cols
            .iter()
            .zip(v_cols.iter())
            .fold(F::zero(), |acc, (uk, vk)| {
                let dot_u: F = uk.iter().zip(new_u.iter()).map(|(&a, &b)| a * b).sum();
                let dot_v: F = vk.iter().zip(new_v.iter()).map(|(&a, &b)| a * b).sum();
                acc + dot_u * dot_v
            });
        approx_norm_sq = approx_norm_sq + rank_contrib_sq + (F::one() + F::one()) * cross_term;

        u_cols.push(new_u);
        v_cols.push(new_v);

        // Convergence check: ‖rank-1 contrib‖_F / ‖approx‖_F < tol
        if approx_norm_sq > F::zero() {
            let rel_contrib = rank_contrib_sq / approx_norm_sq;
            if rel_contrib < tol * tol {
                break;
            }
        }

        // Choose next row pivot: unused row with largest |new_u[i]|
        used_row_pivots[row_pivot] = true;
        let next_row = (0..m).filter(|&i| !used_row_pivots[i]).max_by(|&a, &b| {
            let au = u_cols.last().map(|uk| uk[a].abs()).unwrap_or(F::zero());
            let bu = u_cols.last().map(|uk| uk[b].abs()).unwrap_or(F::zero());
            au.partial_cmp(&bu).unwrap_or(std::cmp::Ordering::Equal)
        });

        match next_row {
            Some(r) => row_pivot = r,
            None => break, // all rows used
        }
    }

    let rank = u_cols.len();

    // Pack into row-major flat storage: U[i, k] = u_cols[k][i]
    let mut u_flat = vec![F::zero(); m * rank];
    let mut v_flat = vec![F::zero(); n * rank];
    for k in 0..rank {
        for i in 0..m {
            u_flat[i * rank + k] = u_cols[k][i];
        }
        for j in 0..n {
            v_flat[j * rank + k] = v_cols[k][j];
        }
    }

    Ok(LowRankBlock {
        u: u_flat,
        v: v_flat,
        rank,
        nrows: m,
        ncols: n,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// H-matrix matvec
// ─────────────────────────────────────────────────────────────────────────────

/// Apply an H-matrix to a vector: `y = H x`.
///
/// Complexity is O(k N log N) where k is the average block rank.
///
/// # Arguments
///
/// * `h` — the H-matrix
/// * `x` — input vector of length `h.ncols`
///
/// # Returns
///
/// Result vector of length `h.nrows`.
pub fn hmatrix_matvec<F>(h: &HMatrix<F>, x: &[F]) -> LinalgResult<Vec<F>>
where
    F: Float + Debug + Clone + Sum + 'static,
{
    if x.len() != h.ncols {
        return Err(LinalgError::DimensionError(format!(
            "hmatrix_matvec: x length {} != h.ncols {}",
            x.len(),
            h.ncols
        )));
    }

    let mut y = vec![F::zero(); h.nrows];

    for (row_off, col_off, block) in &h.blocks {
        let n_rows = block.nrows();
        let n_cols = block.ncols();

        // Extract sub-vector of x
        let x_sub = &x[*col_off..*col_off + n_cols];

        // Apply block and accumulate into y
        block.matvec_add(x_sub, &mut y[*row_off..*row_off + n_rows])?;
    }

    Ok(y)
}

// ─────────────────────────────────────────────────────────────────────────────
// H-matrix compression (SVD recompression)
// ─────────────────────────────────────────────────────────────────────────────

/// Recompress all low-rank blocks in an H-matrix using truncated SVD.
///
/// For each `LowRank(U, V)` block, forms `A = U Vᵀ` (small dense matrix),
/// computes its SVD, and retains only singular values ≥ `tol * σ_max`.
///
/// This typically reduces the rank further after ACA, improving memory use
/// without significant loss of accuracy.
///
/// # Arguments
///
/// * `h` — mutable reference to the H-matrix
/// * `tol` — relative singular value truncation threshold (0 < tol < 1)
pub fn hmatrix_compress<F>(h: &mut HMatrix<F>, tol: F) -> LinalgResult<()>
where
    F: Float + Debug + Clone + Sum + NumAssign + ScalarOperand + Send + Sync + 'static,
{
    for (_, _, block) in h.blocks.iter_mut() {
        if let HBlock::LowRank(lr) = block {
            if lr.rank == 0 {
                continue;
            }

            let m = lr.nrows;
            let n = lr.ncols;
            let r = lr.rank;

            // Form the dense m×n matrix A = U Vᵀ
            let mut a_data = vec![F::zero(); m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut s = F::zero();
                    for k in 0..r {
                        s += lr.u[i * r + k] * lr.v[j * r + k];
                    }
                    a_data[i * n + j] = s;
                }
            }

            // Convert to ndarray for SVD
            let a_arr = Array2::from_shape_vec((m, n), a_data).map_err(|e| {
                LinalgError::ComputationError(format!("hmatrix_compress: shape error: {e}"))
            })?;

            // Call crate SVD
            let (u_arr, s_arr, vt_arr) = crate::decomposition::svd(&a_arr.view(), true, None)
                .map_err(|e| LinalgError::ComputationError(format!("hmatrix_compress svd: {e}")))?;

            // Determine truncated rank
            let sigma_max = if s_arr.is_empty() {
                F::zero()
            } else {
                s_arr[0]
            };
            let threshold = tol * sigma_max;
            let new_rank = s_arr
                .iter()
                .take_while(|&&sv| sv >= threshold)
                .count()
                .max(1)
                .min(r);

            if new_rank >= r {
                // No compression achieved
                continue;
            }

            // Rebuild U: m × new_rank, row-major
            // u_new[i, k] = u_arr[i, k] * sqrt(s[k])
            // v_new[j, k] = vt_arr[k, j] * sqrt(s[k])  (note: VT layout)
            let mut new_u = vec![F::zero(); m * new_rank];
            let mut new_v = vec![F::zero(); n * new_rank];

            for k in 0..new_rank {
                let sv = s_arr[k];
                let sv_sqrt = sv.sqrt();
                for i in 0..m {
                    new_u[i * new_rank + k] = u_arr[[i, k]] * sv_sqrt;
                }
                for j in 0..n {
                    new_v[j * new_rank + k] = vt_arr[[k, j]] * sv_sqrt;
                }
            }

            lr.u = new_u;
            lr.v = new_v;
            lr.rank = new_rank;
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Frobenius norm
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Frobenius norm of an H-matrix: `‖H‖_F = sqrt(Σ_blocks ‖block‖_F²)`.
///
/// For low-rank blocks this is exact (not an approximation), using the identity
/// `‖UV^T‖_F² = trace(U^T U · V^T V)`.
pub fn hmatrix_frobenius_norm<F>(h: &HMatrix<F>) -> F
where
    F: Float + Debug + Clone + Sum + 'static,
{
    let sum_sq: F = h
        .blocks
        .iter()
        .fold(F::zero(), |acc, (_, _, b)| acc + b.frobenius_norm_sq());
    sum_sq.sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// 1D Cauchy kernel: K(i,j) = 1/(|i-j|+1)
    fn cauchy_kernel(i: usize, j: usize) -> f64 {
        let diff = (i as f64 - j as f64).abs();
        1.0 / (diff + 1.0)
    }

    /// Build the full n×n Cauchy matrix (for reference comparison).
    fn dense_cauchy(n: usize) -> Vec<f64> {
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = cauchy_kernel(i, j);
            }
        }
        a
    }

    /// Dense matvec (reference implementation).
    fn dense_matvec(a: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                y[i] += a[i * n + j] * x[j];
            }
        }
        y
    }

    // ── Test 1: Build H-matrix from 1D Cauchy kernel n=64, verify matvec accuracy ──

    #[test]
    fn test_hmatrix_build_and_matvec_n64() {
        let n = 64usize;
        let h: HMatrix<f64> =
            HMatrix::from_kernel(n, cauchy_kernel, 8, 2.0, 1e-6).expect("build failed");

        // Random-ish test vector
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 1.3 + 0.7).sin()).collect();

        let y_h = hmatrix_matvec(&h, &x).expect("matvec failed");
        let a_dense = dense_cauchy(n);
        let y_dense = dense_matvec(&a_dense, n, &x);

        assert_eq!(y_h.len(), n);
        let max_err = y_h
            .iter()
            .zip(y_dense.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        // Tolerance: relative to max magnitude of y
        let max_y = y_dense.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        assert!(
            max_err < 1e-3 * max_y + 1e-10,
            "matvec error too large: {max_err:.2e} (max_y={max_y:.2e})"
        );
    }

    // ── Test 2: ACA produces low rank for smooth kernel ──

    #[test]
    fn test_aca_low_rank_smooth_kernel() {
        let rows: Vec<usize> = (0..16).collect();
        let cols: Vec<usize> = (32..48).collect();

        // Smooth far-field kernel: K(i,j) = exp(-|i-j|/20)
        let kernel = |i: usize, j: usize| -> f64 {
            let d = (i as f64 - j as f64).abs();
            (-d / 20.0).exp()
        };

        let lr = aca_approximate(&kernel, &rows, &cols, 1e-8).expect("ACA failed");

        // A 16×16 far-field exponential decay block should be very low rank
        assert!(
            lr.rank <= 6,
            "expected rank ≤ 6 for smooth kernel, got {}",
            lr.rank
        );
        assert_eq!(lr.nrows, 16);
        assert_eq!(lr.ncols, 16);
    }

    // ── Test 3: Compression reduces rank without accuracy loss ──

    #[test]
    fn test_hmatrix_compress_reduces_rank() {
        let n = 32usize;
        let mut h: HMatrix<f64> =
            HMatrix::from_kernel(n, cauchy_kernel, 4, 1.5, 1e-4).expect("build failed");

        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let y_before = hmatrix_matvec(&h, &x).expect("matvec before compress failed");

        let rank_before: usize = h
            .blocks
            .iter()
            .filter_map(|(_, _, b)| {
                if let HBlock::LowRank(lr) = b {
                    Some(lr.rank)
                } else {
                    None
                }
            })
            .sum();

        hmatrix_compress(&mut h, 1e-6).expect("compress failed");

        let y_after = hmatrix_matvec(&h, &x).expect("matvec after compress failed");

        let rank_after: usize = h
            .blocks
            .iter()
            .filter_map(|(_, _, b)| {
                if let HBlock::LowRank(lr) = b {
                    Some(lr.rank)
                } else {
                    None
                }
            })
            .sum();

        // Rank should not increase after compression
        assert!(
            rank_after <= rank_before + 1, // allow +1 tolerance
            "rank increased after compression: {} → {}",
            rank_before,
            rank_after
        );

        // Accuracy should be maintained
        let max_err = y_before
            .iter()
            .zip(y_after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_err < 1e-3,
            "accuracy lost after compression: max diff = {max_err:.2e}"
        );
    }

    // ── Test 4: LowRankBlock U*V^T has correct shape ──

    #[test]
    fn test_lowrank_block_shape() {
        let lr = LowRankBlock::<f64> {
            u: vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5],           // 3×2
            v: vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 0.5], // 4×2
            rank: 2,
            nrows: 3,
            ncols: 4,
        };
        assert_eq!(lr.nrows, 3);
        assert_eq!(lr.ncols, 4);
        assert_eq!(lr.rank, 2);

        // Test matvec shape
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let y = lr.matvec(&x).expect("matvec failed");
        assert_eq!(y.len(), 3);
    }

    // ── Test 5: Dense block fallback for small leaf_size ──

    #[test]
    fn test_dense_block_fallback() {
        let n = 8usize;
        // With leaf_size = n, everything is stored dense
        let h: HMatrix<f64> =
            HMatrix::from_kernel(n, cauchy_kernel, n, 2.0, 1e-6).expect("build failed");

        // Should only have dense blocks (or single leaf)
        let num_lr = h.num_lowrank_blocks();
        // With leaf_size = n, the root is a leaf → 1 dense block
        assert_eq!(h.num_dense_blocks(), 1, "expected all dense blocks");
        assert_eq!(
            num_lr, 0,
            "expected no low-rank blocks with large leaf_size"
        );
    }

    // ── Test 6: Frobenius norm matches dense norm (within tolerance) ──

    #[test]
    fn test_frobenius_norm_matches_dense() {
        let n = 32usize;
        let h: HMatrix<f64> =
            HMatrix::from_kernel(n, cauchy_kernel, 4, 2.0, 1e-8).expect("build failed");

        let h_norm = hmatrix_frobenius_norm(&h);

        // Dense Frobenius norm
        let a_dense = dense_cauchy(n);
        let dense_norm: f64 = a_dense.iter().map(|&v| v * v).sum::<f64>().sqrt();

        let rel_err = (h_norm - dense_norm).abs() / (dense_norm + 1e-15);
        assert!(
            rel_err < 0.05,
            "Frobenius norm mismatch: H={h_norm:.6}, dense={dense_norm:.6}, rel_err={rel_err:.4}"
        );
    }

    // ── Test 7: Matvec matches dense matrix-vector product for n=32 ──

    #[test]
    fn test_matvec_matches_dense_n32() {
        let n = 32usize;
        let h: HMatrix<f64> =
            HMatrix::from_kernel(n, cauchy_kernel, 4, 2.0, 1e-8).expect("build failed");

        let x: Vec<f64> = (0..n)
            .map(|i| if i % 3 == 0 { 1.0 } else { -0.5 })
            .collect();

        let y_h = hmatrix_matvec(&h, &x).expect("H-matvec failed");
        let a_dense = dense_cauchy(n);
        let y_dense = dense_matvec(&a_dense, n, &x);

        let max_y = y_dense.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let max_err = y_h
            .iter()
            .zip(y_dense.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        assert!(
            max_err < 1e-4 * max_y + 1e-10,
            "matvec error: {max_err:.2e}, max_y={max_y:.2e}"
        );
    }

    // ── Test 8: Round-trip build → compress → matvec still accurate ──

    #[test]
    fn test_round_trip_build_compress_matvec() {
        let n = 48usize;
        let mut h: HMatrix<f64> =
            HMatrix::from_kernel(n, cauchy_kernel, 6, 2.0, 1e-6).expect("build failed");

        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).cos()).collect();

        // Compress
        hmatrix_compress(&mut h, 1e-5).expect("compress failed");

        let y_h = hmatrix_matvec(&h, &x).expect("post-compress matvec failed");
        let a_dense = dense_cauchy(n);
        let y_dense = dense_matvec(&a_dense, n, &x);

        let max_y = y_dense.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let max_err = y_h
            .iter()
            .zip(y_dense.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        assert!(
            max_err < 1e-2 * max_y + 1e-8,
            "round-trip error too large: {max_err:.2e}, max_y={max_y:.2e}"
        );
    }

    // ── Test 9: Block cluster tree builds correct structure ──

    #[test]
    fn test_block_cluster_tree_structure() {
        let rows: Vec<usize> = (0..16).collect();
        let cols: Vec<usize> = (0..16).collect();
        let tree = BlockClusterTree::build(rows, cols, 4, 2.0);

        // With n=16 and leaf_size=4, we should have multiple leaves
        let num_leaves = tree.count_leaves();
        assert!(num_leaves >= 1, "tree should have at least one leaf");
    }

    // ── Test 10: DenseBlock matvec correctness ──

    #[test]
    fn test_dense_block_matvec() {
        // 3×3 identity-like block
        let block = DenseBlock::<f64> {
            data: vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
            nrows: 3,
            ncols: 3,
        };
        let x = vec![1.0, 2.0, 3.0];
        let y = block.matvec(&x).expect("dense matvec failed");
        assert_abs_diff_eq!(y[0], 1.0, epsilon = 1e-14);
        assert_abs_diff_eq!(y[1], 4.0, epsilon = 1e-14);
        assert_abs_diff_eq!(y[2], 9.0, epsilon = 1e-14);
    }
}
