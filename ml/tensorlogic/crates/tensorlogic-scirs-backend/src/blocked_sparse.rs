//! Blocked Sparse Row (BSR) format tensor operations.
//!
//! This module provides a Blocked Sparse Row (BSR) representation for tensors
//! where non-zero data is organized in fixed-size dense blocks. BSR is particularly
//! efficient when there is block-level sparsity — the matrix is sparse at the block
//! granularity but dense within each stored block (e.g., neural network weight matrices
//! pruned with structured/block sparsity).
//!
//! # Format Overview
//!
//! For a matrix of shape `(M, K)` with block size `(br, bc)`:
//! - The logical matrix is divided into `(M/br) × (K/bc)` blocks.
//! - Only blocks where at least one element exceeds a threshold are stored.
//! - `block_row_ptr[i]` .. `block_row_ptr[i+1]` are the indices into `block_col_idx`
//!   and `data` for block-row `i` (CSR-style indirection).
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::ndarray::Array2;
//! use tensorlogic_scirs_backend::blocked_sparse::{BlockedSparseTensor, BlockSparsityStats};
//!
//! let dense = Array2::<f64>::eye(4);
//! let bst = BlockedSparseTensor::from_dense(&dense, 2, 2, 1e-10).unwrap();
//! println!("NNZ blocks: {}", bst.nnz_blocks());
//! println!("Sparsity: {:.2}", bst.sparsity());
//!
//! let stats = BlockSparsityStats::compute(&bst);
//! println!("Compression ratio: {:.2}", stats.compression_ratio);
//! ```

use scirs2_core::ndarray::{s, Array2, ArrayD};
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise from blocked sparse tensor operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockedSparseError {
    /// The matrix dimension is not evenly divisible by the requested block size.
    DimensionNotDivisibleByBlock { dim: usize, block: usize },
    /// A block index (block_row, block_col) is out of range for the tensor.
    BlockIndexOutOfBounds { row: usize, col: usize },
    /// Shape mismatch between two operands.
    ShapeMismatch {
        expected: (usize, usize),
        got: (usize, usize),
    },
    /// Incompatible inner dimensions for matrix multiplication.
    IncompatibleDimensions { lhs_cols: usize, rhs_rows: usize },
    /// The matrix has no rows or no columns.
    EmptyMatrix,
    /// Block size of zero was requested.
    ZeroBlockSize,
}

impl fmt::Display for BlockedSparseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlockedSparseError::DimensionNotDivisibleByBlock { dim, block } => {
                write!(f, "dimension {dim} is not divisible by block size {block}")
            }
            BlockedSparseError::BlockIndexOutOfBounds { row, col } => {
                write!(f, "block index ({row}, {col}) is out of bounds")
            }
            BlockedSparseError::ShapeMismatch { expected, got } => {
                write!(
                    f,
                    "shape mismatch: expected ({}, {}), got ({}, {})",
                    expected.0, expected.1, got.0, got.1
                )
            }
            BlockedSparseError::IncompatibleDimensions { lhs_cols, rhs_rows } => {
                write!(
                    f,
                    "incompatible dimensions for matmul: lhs has {lhs_cols} columns but rhs has {rhs_rows} rows"
                )
            }
            BlockedSparseError::EmptyMatrix => write!(f, "matrix has zero rows or zero columns"),
            BlockedSparseError::ZeroBlockSize => write!(f, "block size must be greater than zero"),
        }
    }
}

impl std::error::Error for BlockedSparseError {}

// ─────────────────────────────────────────────────────────────────────────────
// Core BSR struct
// ─────────────────────────────────────────────────────────────────────────────

/// Blocked Sparse Row (BSR) format tensor.
///
/// Stores non-zero blocks of fixed size `(block_rows × block_cols)`.
/// The indexing structure mirrors the Compressed Sparse Row (CSR) format,
/// but each "element" is a dense 2-D block instead of a scalar.
///
/// # Invariants
///
/// - `nrows % block_rows == 0`
/// - `ncols % block_cols == 0`
/// - `block_row_ptr.len() == num_block_rows() + 1`
/// - `block_col_idx.len() == data.len() == nnz_blocks()`
/// - Every block in `data` has shape `(block_rows, block_cols)`
/// - Column indices within each block-row are sorted in ascending order
#[derive(Debug, Clone)]
pub struct BlockedSparseTensor {
    /// Total number of rows in the logical matrix.
    pub nrows: usize,
    /// Total number of columns in the logical matrix.
    pub ncols: usize,
    /// Height of each block (in scalar elements).
    pub block_rows: usize,
    /// Width of each block (in scalar elements).
    pub block_cols: usize,
    /// CSR-style row pointer into `block_col_idx` / `data`.
    /// Length is `num_block_rows() + 1`.
    pub block_row_ptr: Vec<usize>,
    /// Block-column index for each stored block.
    pub block_col_idx: Vec<usize>,
    /// Dense data for each stored block; shape is `(block_rows, block_cols)`.
    pub data: Vec<Array2<f64>>,
}

impl BlockedSparseTensor {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Create a BSR tensor from a dense 2-D array.
    ///
    /// Blocks whose maximum absolute value is ≤ `threshold` are treated as
    /// zero and **not** stored.
    pub fn from_dense(
        dense: &Array2<f64>,
        block_rows: usize,
        block_cols: usize,
        threshold: f64,
    ) -> Result<Self, BlockedSparseError> {
        if block_rows == 0 || block_cols == 0 {
            return Err(BlockedSparseError::ZeroBlockSize);
        }

        let (nrows, ncols) = dense.dim();

        if nrows == 0 || ncols == 0 {
            return Err(BlockedSparseError::EmptyMatrix);
        }

        if !nrows.is_multiple_of(block_rows) {
            return Err(BlockedSparseError::DimensionNotDivisibleByBlock {
                dim: nrows,
                block: block_rows,
            });
        }

        if !ncols.is_multiple_of(block_cols) {
            return Err(BlockedSparseError::DimensionNotDivisibleByBlock {
                dim: ncols,
                block: block_cols,
            });
        }

        let nbr = nrows / block_rows;
        let nbc = ncols / block_cols;

        let mut block_row_ptr = vec![0usize; nbr + 1];
        let mut block_col_idx: Vec<usize> = Vec::new();
        let mut data: Vec<Array2<f64>> = Vec::new();

        for br in 0..nbr {
            let row_start = br * block_rows;
            let row_end = row_start + block_rows;

            for bc in 0..nbc {
                let col_start = bc * block_cols;
                let col_end = col_start + block_cols;

                let block = dense.slice(s![row_start..row_end, col_start..col_end]);

                // Check if the block is non-zero (any |entry| > threshold).
                let is_nonzero = block.iter().any(|&v| v.abs() > threshold);
                if is_nonzero {
                    block_col_idx.push(bc);
                    data.push(block.to_owned());
                }
            }

            block_row_ptr[br + 1] = block_col_idx.len();
        }

        Ok(BlockedSparseTensor {
            nrows,
            ncols,
            block_rows,
            block_cols,
            block_row_ptr,
            block_col_idx,
            data,
        })
    }

    /// Create an empty BSR tensor with the given dimensions and block sizes.
    ///
    /// No blocks are stored initially; use `set_block` to populate.
    pub fn empty(
        nrows: usize,
        ncols: usize,
        block_rows: usize,
        block_cols: usize,
    ) -> Result<Self, BlockedSparseError> {
        if block_rows == 0 || block_cols == 0 {
            return Err(BlockedSparseError::ZeroBlockSize);
        }
        if nrows == 0 || ncols == 0 {
            return Err(BlockedSparseError::EmptyMatrix);
        }
        if !nrows.is_multiple_of(block_rows) {
            return Err(BlockedSparseError::DimensionNotDivisibleByBlock {
                dim: nrows,
                block: block_rows,
            });
        }
        if !ncols.is_multiple_of(block_cols) {
            return Err(BlockedSparseError::DimensionNotDivisibleByBlock {
                dim: ncols,
                block: block_cols,
            });
        }

        let nbr = nrows / block_rows;
        Ok(BlockedSparseTensor {
            nrows,
            ncols,
            block_rows,
            block_cols,
            block_row_ptr: vec![0usize; nbr + 1],
            block_col_idx: Vec::new(),
            data: Vec::new(),
        })
    }

    // ── Dimension helpers ────────────────────────────────────────────────────

    /// Number of block-rows in the matrix.
    #[inline]
    pub fn num_block_rows(&self) -> usize {
        self.nrows / self.block_rows
    }

    /// Number of block-columns in the matrix.
    #[inline]
    pub fn num_block_cols(&self) -> usize {
        self.ncols / self.block_cols
    }

    /// Number of stored (non-zero) blocks.
    #[inline]
    pub fn nnz_blocks(&self) -> usize {
        self.data.len()
    }

    /// Fraction of blocks that are **zero** (not stored).
    ///
    /// Returns `1.0` when no blocks are stored, `0.0` when all blocks are stored.
    pub fn sparsity(&self) -> f64 {
        let total = self.num_block_rows() * self.num_block_cols();
        if total == 0 {
            return 1.0;
        }
        let nnz = self.nnz_blocks();
        1.0 - (nnz as f64 / total as f64)
    }

    /// Approximate memory usage in bytes (stack + heap).
    ///
    /// Counts:
    /// - `block_row_ptr` vector elements
    /// - `block_col_idx` vector elements
    /// - all dense block data (`block_rows × block_cols × 8` bytes each)
    pub fn memory_bytes(&self) -> usize {
        let ptr_bytes = self.block_row_ptr.len() * std::mem::size_of::<usize>();
        let idx_bytes = self.block_col_idx.len() * std::mem::size_of::<usize>();
        let block_element_bytes = self.block_rows * self.block_cols * std::mem::size_of::<f64>();
        let data_bytes = self.nnz_blocks() * block_element_bytes;
        ptr_bytes + idx_bytes + data_bytes
    }

    // ── Block access ─────────────────────────────────────────────────────────

    /// Retrieve a reference to the stored block at logical block position
    /// `(block_row, block_col)`.
    ///
    /// Returns `None` if the block is not stored (i.e. it is a zero block).
    pub fn get_block(&self, block_row: usize, block_col: usize) -> Option<&Array2<f64>> {
        if block_row >= self.num_block_rows() || block_col >= self.num_block_cols() {
            return None;
        }
        let start = self.block_row_ptr[block_row];
        let end = self.block_row_ptr[block_row + 1];
        // Column indices within a block-row are sorted; use binary search.
        match self.block_col_idx[start..end].binary_search(&block_col) {
            Ok(relative_pos) => Some(&self.data[start + relative_pos]),
            Err(_) => None,
        }
    }

    /// Insert or replace the block at `(block_row, block_col)`.
    ///
    /// If a block already exists at that position it is replaced in-place.
    /// Otherwise the block is inserted while maintaining the sorted column-index
    /// invariant within the block-row.
    ///
    /// The provided `block` must have shape `(block_rows, block_cols)`.
    pub fn set_block(
        &mut self,
        block_row: usize,
        block_col: usize,
        block: Array2<f64>,
    ) -> Result<(), BlockedSparseError> {
        if block_row >= self.num_block_rows() || block_col >= self.num_block_cols() {
            return Err(BlockedSparseError::BlockIndexOutOfBounds {
                row: block_row,
                col: block_col,
            });
        }

        let block_shape = block.dim();
        if block_shape != (self.block_rows, self.block_cols) {
            return Err(BlockedSparseError::ShapeMismatch {
                expected: (self.block_rows, self.block_cols),
                got: block_shape,
            });
        }

        let start = self.block_row_ptr[block_row];
        let end = self.block_row_ptr[block_row + 1];

        match self.block_col_idx[start..end].binary_search(&block_col) {
            Ok(relative_pos) => {
                // Replace existing block.
                self.data[start + relative_pos] = block;
            }
            Err(insert_offset) => {
                // Insert new block, maintaining sorted order.
                let abs_pos = start + insert_offset;
                self.block_col_idx.insert(abs_pos, block_col);
                self.data.insert(abs_pos, block);
                // Increment all row pointers beyond this block-row.
                for ptr in self.block_row_ptr[block_row + 1..].iter_mut() {
                    *ptr += 1;
                }
            }
        }

        Ok(())
    }

    // ── Dense conversion ─────────────────────────────────────────────────────

    /// Convert the BSR tensor to a dense `Array2<f64>`.
    pub fn to_dense(&self) -> Array2<f64> {
        let mut dense = Array2::<f64>::zeros((self.nrows, self.ncols));

        for br in 0..self.num_block_rows() {
            let row_start = br * self.block_rows;
            let row_end = row_start + self.block_rows;
            let start = self.block_row_ptr[br];
            let end = self.block_row_ptr[br + 1];

            for idx in start..end {
                let bc = self.block_col_idx[idx];
                let col_start = bc * self.block_cols;
                let col_end = col_start + self.block_cols;

                dense
                    .slice_mut(s![row_start..row_end, col_start..col_end])
                    .assign(&self.data[idx]);
            }
        }

        dense
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Arithmetic operations
// ─────────────────────────────────────────────────────────────────────────────

/// Multiply a blocked sparse matrix `A` (M × K) by a dense matrix `B` (K × N),
/// producing a dense result `C` (M × N).
///
/// This routine iterates only over stored (non-zero) blocks of `A`, performing a
/// GEMM-style product for each and accumulating into the output.
pub fn blocked_sparse_dense_mm(
    a: &BlockedSparseTensor,
    b: &Array2<f64>,
) -> Result<Array2<f64>, BlockedSparseError> {
    let (b_rows, b_cols) = b.dim();

    if a.ncols != b_rows {
        return Err(BlockedSparseError::IncompatibleDimensions {
            lhs_cols: a.ncols,
            rhs_rows: b_rows,
        });
    }

    let mut c = Array2::<f64>::zeros((a.nrows, b_cols));

    for br in 0..a.num_block_rows() {
        let row_start = br * a.block_rows;
        let row_end = row_start + a.block_rows;
        let ptr_start = a.block_row_ptr[br];
        let ptr_end = a.block_row_ptr[br + 1];

        for idx in ptr_start..ptr_end {
            let bc = a.block_col_idx[idx];
            let col_start = bc * a.block_cols;
            let col_end = col_start + a.block_cols;

            // a_block: (block_rows, block_cols)
            // b_slice: (block_cols, b_cols)
            let a_block = &a.data[idx];
            let b_slice = b.slice(s![col_start..col_end, ..]);

            // GEMM: c_slice += a_block @ b_slice
            let product = a_block.dot(&b_slice);
            c.slice_mut(s![row_start..row_end, ..])
                .scaled_add(1.0, &product);
        }
    }

    Ok(c)
}

/// Multiply two blocked sparse matrices `A` (M × K) and `B` (K × N),
/// producing a blocked sparse result `C` (M × N).
///
/// Both operands must use the **same block sizes**. The output uses those same
/// block sizes. Blocks in `C` that accumulate to all-zeros (below a very tight
/// threshold of `f64::EPSILON`) are **not** stored.
pub fn blocked_sparse_mm(
    a: &BlockedSparseTensor,
    b: &BlockedSparseTensor,
) -> Result<BlockedSparseTensor, BlockedSparseError> {
    if a.ncols != b.nrows {
        return Err(BlockedSparseError::IncompatibleDimensions {
            lhs_cols: a.ncols,
            rhs_rows: b.nrows,
        });
    }
    if a.block_rows != b.block_rows || a.block_cols != b.block_cols {
        // Block sizes must match so that block products are conformable.
        return Err(BlockedSparseError::ShapeMismatch {
            expected: (a.block_rows, a.block_cols),
            got: (b.block_rows, b.block_cols),
        });
    }

    let nbr_a = a.num_block_rows();
    let nbc_b = b.num_block_cols();
    let _nbc_a = a.num_block_cols(); // == nbr_b (kept for documentation)

    // We accumulate dense block results into a 2-D array of Option<Array2<f64>>.
    // This avoids the complexity of building CSR incrementally in arbitrary order.
    let mut acc: Vec<Vec<Option<Array2<f64>>>> = (0..nbr_a)
        .map(|_| (0..nbc_b).map(|_| None).collect())
        .collect();

    // For each block-column in B that block-row br_a touches, accumulate.
    // We need index access to both `a.block_row_ptr` and `acc`, so we use an
    // index-based loop here; suppress clippy's needless_range_loop lint since
    // we intentionally index multiple data structures by the same counter.
    #[allow(clippy::needless_range_loop)]
    for br_a in 0..nbr_a {
        let a_start = a.block_row_ptr[br_a];
        let a_end = a.block_row_ptr[br_a + 1];

        for a_idx in a_start..a_end {
            let bc_a = a.block_col_idx[a_idx]; // == br_b
            let a_block = &a.data[a_idx];

            // Walk the matching block-row of B (bc_a is the shared index).
            if bc_a >= b.num_block_rows() {
                continue;
            }
            let b_start = b.block_row_ptr[bc_a];
            let b_end = b.block_row_ptr[bc_a + 1];

            for b_idx in b_start..b_end {
                let bc_b = b.block_col_idx[b_idx];
                let b_block = &b.data[b_idx];

                let product = a_block.dot(b_block);

                match &mut acc[br_a][bc_b] {
                    Some(existing) => {
                        *existing = &*existing + &product;
                    }
                    slot @ None => {
                        *slot = Some(product);
                    }
                }
            }
        }
    }

    // Build the output BSR from the accumulation.
    let mut c = BlockedSparseTensor::empty(a.nrows, b.ncols, a.block_rows, a.block_cols)?;

    #[allow(clippy::needless_range_loop)]
    for br in 0..nbr_a {
        for bc in 0..nbc_b {
            if let Some(block) = acc[br][bc].take() {
                // Only store the block if it has at least one significant entry.
                let is_nonzero = block.iter().any(|&v| v.abs() > f64::EPSILON);
                if is_nonzero {
                    c.set_block(br, bc, block)?;
                }
            }
        }
    }

    Ok(c)
}

/// Element-wise addition of two BSR tensors with identical shape and block sizes.
///
/// Blocks present in either operand are included. Blocks present in both are summed.
pub fn blocked_sparse_add(
    a: &BlockedSparseTensor,
    b: &BlockedSparseTensor,
) -> Result<BlockedSparseTensor, BlockedSparseError> {
    if a.nrows != b.nrows || a.ncols != b.ncols {
        return Err(BlockedSparseError::ShapeMismatch {
            expected: (a.nrows, a.ncols),
            got: (b.nrows, b.ncols),
        });
    }
    if a.block_rows != b.block_rows || a.block_cols != b.block_cols {
        return Err(BlockedSparseError::ShapeMismatch {
            expected: (a.block_rows, a.block_cols),
            got: (b.block_rows, b.block_cols),
        });
    }

    let mut result = a.clone();

    for br in 0..b.num_block_rows() {
        let b_start = b.block_row_ptr[br];
        let b_end = b.block_row_ptr[br + 1];

        for idx in b_start..b_end {
            let bc = b.block_col_idx[idx];
            let b_block = &b.data[idx];

            // Check if 'result' already has this block.
            let r_start = result.block_row_ptr[br];
            let r_end = result.block_row_ptr[br + 1];

            match result.block_col_idx[r_start..r_end].binary_search(&bc) {
                Ok(relative) => {
                    // Add in-place.
                    let abs = r_start + relative;
                    result.data[abs] = &result.data[abs] + b_block;
                }
                Err(_) => {
                    // Block only in B — insert it.
                    result.set_block(br, bc, b_block.clone())?;
                }
            }
        }
    }

    Ok(result)
}

/// Scale every stored element of a BSR tensor by a scalar factor, returning a new tensor.
pub fn blocked_sparse_scale(tensor: &BlockedSparseTensor, scalar: f64) -> BlockedSparseTensor {
    let scaled_data: Vec<Array2<f64>> = tensor.data.iter().map(|block| block * scalar).collect();

    BlockedSparseTensor {
        nrows: tensor.nrows,
        ncols: tensor.ncols,
        block_rows: tensor.block_rows,
        block_cols: tensor.block_cols,
        block_row_ptr: tensor.block_row_ptr.clone(),
        block_col_idx: tensor.block_col_idx.clone(),
        data: scaled_data,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Detailed statistics about the block-level sparsity of a BSR tensor.
#[derive(Debug, Clone)]
pub struct BlockSparsityStats {
    /// Total number of possible blocks.
    pub total_blocks: usize,
    /// Number of stored (non-zero) blocks.
    pub nnz_blocks: usize,
    /// Fraction of blocks that are zero (1 − density).
    pub sparsity: f64,
    /// Fraction of blocks that are non-zero (1 − sparsity).
    pub density: f64,
    /// Actual memory used by the BSR representation (bytes).
    pub memory_bytes: usize,
    /// Memory a fully dense `f64` matrix would require (bytes).
    pub theoretical_dense_bytes: usize,
    /// `theoretical_dense_bytes / memory_bytes`. Values > 1.0 indicate savings.
    pub compression_ratio: f64,
    /// Mean Frobenius norm of stored blocks.
    pub avg_block_norm: f64,
    /// Maximum Frobenius norm among stored blocks.
    pub max_block_norm: f64,
}

impl BlockSparsityStats {
    /// Compute statistics for the given BSR tensor.
    pub fn compute(tensor: &BlockedSparseTensor) -> Self {
        let total_blocks = tensor.num_block_rows() * tensor.num_block_cols();
        let nnz_blocks = tensor.nnz_blocks();
        let sparsity = tensor.sparsity();
        let density = 1.0 - sparsity;
        let memory_bytes = tensor.memory_bytes();
        let theoretical_dense_bytes = tensor.nrows * tensor.ncols * std::mem::size_of::<f64>();

        let compression_ratio = if memory_bytes == 0 {
            f64::INFINITY
        } else {
            theoretical_dense_bytes as f64 / memory_bytes as f64
        };

        // Frobenius norms of stored blocks.
        let block_norms: Vec<f64> = tensor
            .data
            .iter()
            .map(|block| {
                let sq_sum: f64 = block.iter().map(|&v| v * v).sum();
                sq_sum.sqrt()
            })
            .collect();

        let avg_block_norm = if block_norms.is_empty() {
            0.0
        } else {
            block_norms.iter().sum::<f64>() / block_norms.len() as f64
        };

        let max_block_norm = block_norms.iter().cloned().fold(0.0_f64, f64::max);

        BlockSparsityStats {
            total_blocks,
            nnz_blocks,
            sparsity,
            density,
            memory_bytes,
            theoretical_dense_bytes,
            compression_ratio,
            avg_block_norm,
            max_block_norm,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sparsity pattern
// ─────────────────────────────────────────────────────────────────────────────

/// Binary pattern showing which blocks of a BSR tensor are non-zero.
///
/// This is useful for structural analysis (symmetry detection, diagonal coverage,
/// bandwidth estimation) without loading the actual block data.
#[derive(Debug, Clone)]
pub struct BlockSparsityPattern {
    /// Number of block-rows.
    pub nblock_rows: usize,
    /// Number of block-columns.
    pub nblock_cols: usize,
    /// `pattern[br][bc]` is `true` iff block `(br, bc)` is stored.
    pub pattern: Vec<Vec<bool>>,
}

impl BlockSparsityPattern {
    /// Derive the sparsity pattern from a BSR tensor.
    pub fn from_tensor(tensor: &BlockedSparseTensor) -> Self {
        let nbr = tensor.num_block_rows();
        let nbc = tensor.num_block_cols();

        let mut pattern = vec![vec![false; nbc]; nbr];

        // Index-based loop needed because we access tensor.block_row_ptr by index.
        #[allow(clippy::needless_range_loop)]
        for br in 0..nbr {
            let start = tensor.block_row_ptr[br];
            let end = tensor.block_row_ptr[br + 1];
            for idx in start..end {
                let bc = tensor.block_col_idx[idx];
                if bc < nbc {
                    pattern[br][bc] = true;
                }
            }
        }

        BlockSparsityPattern {
            nblock_rows: nbr,
            nblock_cols: nbc,
            pattern,
        }
    }

    /// Fraction of blocks that are non-zero.
    pub fn density(&self) -> f64 {
        let total = self.nblock_rows * self.nblock_cols;
        if total == 0 {
            return 0.0;
        }
        let nnz: usize = self
            .pattern
            .iter()
            .map(|row| row.iter().filter(|&&v| v).count())
            .sum();
        nnz as f64 / total as f64
    }

    /// Returns `true` if the block pattern is square and symmetric, i.e.
    /// `pattern[i][j] == pattern[j][i]` for all valid `(i, j)`.
    pub fn is_symmetric(&self) -> bool {
        if self.nblock_rows != self.nblock_cols {
            return false;
        }
        let n = self.nblock_rows;
        for i in 0..n {
            for j in 0..n {
                if self.pattern[i][j] != self.pattern[j][i] {
                    return false;
                }
            }
        }
        true
    }

    /// Returns `true` if every diagonal block `(i, i)` is present.
    ///
    /// Only meaningful for square block patterns.
    pub fn has_diagonal_blocks(&self) -> bool {
        let n = self.nblock_rows.min(self.nblock_cols);
        (0..n).all(|i| self.pattern[i][i])
    }

    /// Render the sparsity pattern as an ASCII string.
    ///
    /// `'#'` represents a non-zero block; `'.'` represents a zero block.
    /// Each block-row occupies one text line.
    pub fn to_ascii(&self) -> String {
        let mut output = String::with_capacity(self.nblock_rows * (self.nblock_cols + 1));
        for row in &self.pattern {
            for &present in row {
                output.push(if present { '#' } else { '.' });
            }
            output.push('\n');
        }
        output
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports for convenience
// ─────────────────────────────────────────────────────────────────────────────

/// Convenience alias for `ndarray::ArrayD<f64>` (dynamic-dimensional tensor).
pub type BlockedSparseDynTensor = ArrayD<f64>;

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build a 4×4 matrix where every element equals its linear index as f64.
    fn make_4x4() -> Array2<f64> {
        Array2::from_shape_fn((4, 4), |(r, c)| (r * 4 + c) as f64 + 1.0)
    }

    /// Build a 4×4 identity matrix.
    fn make_identity_4() -> Array2<f64> {
        Array2::<f64>::eye(4)
    }

    // ── Construction tests ────────────────────────────────────────────────────

    #[test]
    fn test_from_dense_all_nonzero() {
        let dense = make_4x4();
        let bst = BlockedSparseTensor::from_dense(&dense, 2, 2, 1e-10)
            .expect("from_dense should succeed");
        // 4×4 matrix with 2×2 blocks → 4 blocks total, all non-zero.
        assert_eq!(bst.nnz_blocks(), 4, "all 4 blocks should be stored");
    }

    #[test]
    fn test_from_dense_threshold_drops_blocks() {
        // Create 4×4 where the bottom-right 2×2 block is near-zero.
        let mut dense = make_4x4();
        dense.slice_mut(s![2..4, 2..4]).fill(1e-15);

        let bst = BlockedSparseTensor::from_dense(&dense, 2, 2, 1e-10)
            .expect("from_dense should succeed");
        // Only 3 blocks should be stored (bottom-right dropped).
        assert_eq!(bst.nnz_blocks(), 3, "near-zero block should be dropped");
    }

    #[test]
    fn test_to_dense_roundtrip() {
        let dense = make_4x4();
        let bst = BlockedSparseTensor::from_dense(&dense, 2, 2, 1e-10)
            .expect("from_dense should succeed");
        let recovered = bst.to_dense();

        for ((r, c), &original_val) in dense.indexed_iter() {
            let recovered_val = recovered[(r, c)];
            assert!(
                (original_val - recovered_val).abs() < 1e-12,
                "mismatch at ({r},{c}): {original_val} vs {recovered_val}"
            );
        }
    }

    #[test]
    fn test_get_block_exists() {
        let dense = make_4x4();
        let bst = BlockedSparseTensor::from_dense(&dense, 2, 2, 1e-10)
            .expect("from_dense should succeed");

        let block = bst.get_block(0, 0).expect("block (0,0) must be stored");
        // Top-left 2×2 of make_4x4() is [[1,2],[5,6]].
        assert!((block[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((block[(0, 1)] - 2.0).abs() < 1e-12);
        assert!((block[(1, 0)] - 5.0).abs() < 1e-12);
        assert!((block[(1, 1)] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_get_block_missing() {
        let mut dense = make_4x4();
        dense.slice_mut(s![2..4, 2..4]).fill(1e-15);

        let bst = BlockedSparseTensor::from_dense(&dense, 2, 2, 1e-10)
            .expect("from_dense should succeed");

        // Block (1,1) is the bottom-right block — it was zeroed out.
        let result = bst.get_block(1, 1);
        assert!(result.is_none(), "dropped block should return None");
    }

    #[test]
    fn test_set_block_inserts() {
        let mut bst =
            BlockedSparseTensor::empty(4, 4, 2, 2).expect("empty construction should succeed");
        assert_eq!(bst.nnz_blocks(), 0);

        let new_block = Array2::from_elem((2, 2), 7.0);
        bst.set_block(0, 1, new_block.clone())
            .expect("set_block should succeed");

        assert_eq!(bst.nnz_blocks(), 1);
        let retrieved = bst
            .get_block(0, 1)
            .expect("block must be present after set");
        assert!((retrieved[(0, 0)] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_nnz_blocks() {
        let identity = make_identity_4();
        // 4×4 identity with 2×2 blocks → 2 diagonal blocks non-zero.
        let bst = BlockedSparseTensor::from_dense(&identity, 2, 2, 1e-10)
            .expect("from_dense should succeed");
        assert_eq!(bst.nnz_blocks(), 2);
    }

    // ── Sparsity tests ────────────────────────────────────────────────────────

    #[test]
    fn test_sparsity_all_dense() {
        let dense = make_4x4();
        let bst = BlockedSparseTensor::from_dense(&dense, 2, 2, 1e-10)
            .expect("from_dense should succeed");
        assert!(
            bst.sparsity().abs() < 1e-12,
            "sparsity should be 0 when all blocks stored"
        );
    }

    #[test]
    fn test_sparsity_all_sparse() {
        // All near-zero → no blocks stored.
        let tiny = Array2::<f64>::from_elem((4, 4), 1e-15);
        let bst =
            BlockedSparseTensor::from_dense(&tiny, 2, 2, 1e-10).expect("from_dense should succeed");
        assert!(
            (bst.sparsity() - 1.0).abs() < 1e-12,
            "sparsity should be 1 when no blocks stored"
        );
    }

    // ── Matrix multiply tests ─────────────────────────────────────────────────

    #[test]
    fn test_blocked_sparse_dense_mm_correctness() {
        let a_dense = make_4x4();
        let b_dense = Array2::from_shape_fn((4, 3), |(r, c)| (r * 3 + c) as f64 * 0.1);
        let expected = a_dense.dot(&b_dense);

        let a_bsr = BlockedSparseTensor::from_dense(&a_dense, 2, 2, 1e-10)
            .expect("from_dense should succeed");
        let result =
            blocked_sparse_dense_mm(&a_bsr, &b_dense).expect("sparse-dense mm should succeed");

        for ((r, c), &exp_val) in expected.indexed_iter() {
            let got = result[(r, c)];
            assert!(
                (exp_val - got).abs() < 1e-9,
                "sparse-dense mm mismatch at ({r},{c}): expected {exp_val}, got {got}"
            );
        }
    }

    #[test]
    fn test_blocked_sparse_mm_correctness() {
        let a_dense = make_4x4();
        let b_dense = Array2::from_shape_fn((4, 4), |(r, c)| ((r + 1) * (c + 1)) as f64 * 0.5);
        let expected = a_dense.dot(&b_dense);

        let a_bsr = BlockedSparseTensor::from_dense(&a_dense, 2, 2, 1e-10).expect("from_dense A");
        let b_bsr = BlockedSparseTensor::from_dense(&b_dense, 2, 2, 1e-10).expect("from_dense B");

        let c_bsr = blocked_sparse_mm(&a_bsr, &b_bsr).expect("sparse-sparse mm should succeed");
        let c_dense = c_bsr.to_dense();

        for ((r, c), &exp_val) in expected.indexed_iter() {
            let got = c_dense[(r, c)];
            assert!(
                (exp_val - got).abs() < 1e-9,
                "sparse-sparse mm mismatch at ({r},{c}): expected {exp_val}, got {got}"
            );
        }
    }

    // ── Element-wise operation tests ──────────────────────────────────────────

    #[test]
    fn test_blocked_sparse_add() {
        let a_dense = make_4x4();
        let b_dense = Array2::from_shape_fn((4, 4), |(r, c)| (r + c) as f64);
        let expected = &a_dense + &b_dense;

        let a_bsr = BlockedSparseTensor::from_dense(&a_dense, 2, 2, 1e-10).expect("from_dense A");
        let b_bsr = BlockedSparseTensor::from_dense(&b_dense, 2, 2, 1e-10).expect("from_dense B");

        let c_bsr = blocked_sparse_add(&a_bsr, &b_bsr).expect("add should succeed");
        let c_dense = c_bsr.to_dense();

        for ((r, c), &exp_val) in expected.indexed_iter() {
            let got = c_dense[(r, c)];
            assert!(
                (exp_val - got).abs() < 1e-12,
                "add mismatch at ({r},{c}): expected {exp_val}, got {got}"
            );
        }
    }

    #[test]
    fn test_blocked_sparse_scale() {
        let a_dense = make_4x4();
        let a_bsr = BlockedSparseTensor::from_dense(&a_dense, 2, 2, 1e-10).expect("from_dense");

        let scaled_bsr = blocked_sparse_scale(&a_bsr, 2.0);
        let scaled_dense = scaled_bsr.to_dense();

        for ((r, c), &orig) in a_dense.indexed_iter() {
            let got = scaled_dense[(r, c)];
            assert!(
                (got - orig * 2.0).abs() < 1e-12,
                "scale mismatch at ({r},{c}): expected {}, got {got}",
                orig * 2.0
            );
        }
    }

    // ── BlockSparsityStats tests ──────────────────────────────────────────────

    #[test]
    fn test_block_sparsity_stats_density() {
        let dense = make_4x4();
        let bst = BlockedSparseTensor::from_dense(&dense, 2, 2, 1e-10).expect("from_dense");
        let stats = BlockSparsityStats::compute(&bst);
        let sum = stats.density + stats.sparsity;
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "density + sparsity must equal 1.0, got {sum}"
        );
    }

    #[test]
    fn test_block_sparsity_stats_compression_ratio() {
        // Identity matrix: 2 out of 4 blocks are non-zero → sparse → compression > 1.
        let identity = make_identity_4();
        let bst = BlockedSparseTensor::from_dense(&identity, 2, 2, 1e-10).expect("from_dense");
        let stats = BlockSparsityStats::compute(&bst);
        assert!(
            stats.compression_ratio > 1.0,
            "compression_ratio should be > 1.0 for a sparse matrix, got {}",
            stats.compression_ratio
        );
    }

    // ── BlockSparsityPattern tests ────────────────────────────────────────────

    #[test]
    fn test_block_sparsity_pattern_from_tensor() {
        let identity = make_identity_4();
        let bst = BlockedSparseTensor::from_dense(&identity, 2, 2, 1e-10).expect("from_dense");
        let pattern = BlockSparsityPattern::from_tensor(&bst);

        // density (pattern) == 1 - sparsity (tensor)
        let expected_density = 1.0 - bst.sparsity();
        assert!(
            (pattern.density() - expected_density).abs() < 1e-12,
            "pattern density {} should equal tensor density {}",
            pattern.density(),
            expected_density
        );
    }

    #[test]
    fn test_block_sparsity_pattern_symmetric() {
        // Build a symmetric dense matrix.
        let base = make_4x4();
        let sym = &base + &base.t().to_owned();
        let bst = BlockedSparseTensor::from_dense(&sym, 2, 2, 1e-10).expect("from_dense");
        let pattern = BlockSparsityPattern::from_tensor(&bst);
        // All 4 blocks are non-zero → symmetric.
        assert!(
            pattern.is_symmetric(),
            "fully dense pattern must be symmetric"
        );
    }

    #[test]
    fn test_block_sparsity_pattern_ascii_shape() {
        let dense = make_4x4();
        let bst = BlockedSparseTensor::from_dense(&dense, 2, 2, 1e-10).expect("from_dense");
        let pattern = BlockSparsityPattern::from_tensor(&bst);
        let ascii = pattern.to_ascii();

        // Should have exactly nblock_rows lines.
        let lines: Vec<&str> = ascii.lines().collect();
        assert_eq!(
            lines.len(),
            pattern.nblock_rows,
            "ASCII should have {} lines",
            pattern.nblock_rows
        );
        // Each line should have exactly nblock_cols characters.
        for line in &lines {
            assert_eq!(
                line.len(),
                pattern.nblock_cols,
                "each ASCII line should have {} characters",
                pattern.nblock_cols
            );
        }
    }

    // ── Error condition tests ─────────────────────────────────────────────────

    #[test]
    fn test_dimension_not_divisible_error() {
        // 5×4 matrix with block size 2 — row dimension not divisible.
        let odd = Array2::<f64>::zeros((5, 4));
        let result = BlockedSparseTensor::from_dense(&odd, 2, 2, 1e-10);
        match result {
            Err(BlockedSparseError::DimensionNotDivisibleByBlock { dim: 5, block: 2 }) => {}
            other => panic!("expected DimensionNotDivisibleByBlock, got {other:?}"),
        }
    }

    #[test]
    fn test_memory_bytes_positive() {
        let dense = make_4x4();
        let bst = BlockedSparseTensor::from_dense(&dense, 2, 2, 1e-10).expect("from_dense");
        assert!(
            bst.memory_bytes() > 0,
            "memory_bytes must be positive for non-empty tensor"
        );
    }
}
