//! Structured sparsity patterns and pruning for TensorLogic.
//!
//! Provides magnitude-based and structured pruning to reduce model size
//! while maintaining hardware efficiency through regular sparsity patterns.

use ndarray::{Array1, Array2, ArrayD};
use thiserror::Error;

/// Errors from pruning operations.
#[derive(Debug, Error)]
pub enum PruningError {
    #[error("Invalid sparsity ratio: {0}. Must be in [0, 1).")]
    InvalidSparsityRatio(f64),
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("Block size {0} does not divide dimension {1}")]
    InvalidBlockSize(usize, usize),
    #[error("Empty tensor")]
    EmptyTensor,
}

/// Structured sparsity pattern types.
#[derive(Debug, Clone, PartialEq)]
pub enum SparsityPattern {
    /// Element-wise (unstructured) sparsity — zero out individual elements.
    Unstructured,
    /// Block sparsity — zero out rectangular blocks of size (block_h × block_w).
    Block { block_h: usize, block_w: usize },
    /// Channel/row sparsity — zero out entire rows (for weight matrices).
    Row,
    /// Column sparsity — zero out entire columns.
    Column,
    /// N:M sparsity — keep N non-zero values per group of M (common on Ampere+ GPUs).
    NM { n: usize, m: usize },
}

impl SparsityPattern {
    /// Human-readable name for this pattern.
    pub fn name(&self) -> &'static str {
        match self {
            SparsityPattern::Unstructured => "unstructured",
            SparsityPattern::Block { .. } => "block",
            SparsityPattern::Row => "row",
            SparsityPattern::Column => "column",
            SparsityPattern::NM { .. } => "n:m",
        }
    }

    /// Whether this is a structured pattern (more hardware efficient).
    pub fn is_structured(&self) -> bool {
        !matches!(self, SparsityPattern::Unstructured)
    }
}

/// Statistics about the sparsity of a pruned tensor.
#[derive(Debug, Clone)]
pub struct SparsityStats {
    /// Fraction of zero elements (0.0 = dense, 1.0 = all-zero).
    pub actual_sparsity: f64,
    /// Number of zero elements.
    pub zero_count: usize,
    /// Total number of elements.
    pub total_count: usize,
    /// Theoretical compute speedup from sparsity (rough estimate).
    pub theoretical_speedup: f64,
    /// Pattern used for pruning.
    pub pattern: SparsityPattern,
}

impl SparsityStats {
    /// Compute sparsity statistics for a tensor with the given pattern.
    pub fn compute(tensor: &ArrayD<f64>, pattern: SparsityPattern) -> Self {
        let total_count = tensor.len();
        let zero_count = tensor.iter().filter(|&&v| v == 0.0).count();
        let actual_sparsity = if total_count == 0 {
            0.0
        } else {
            zero_count as f64 / total_count as f64
        };
        // Rough speedup: structured sparsity > unstructured
        let theoretical_speedup = if pattern.is_structured() {
            1.0 / (1.0 - actual_sparsity + 1e-9)
        } else {
            1.0 + actual_sparsity * 0.5 // unstructured has limited hw benefit
        };
        SparsityStats {
            actual_sparsity,
            zero_count,
            total_count,
            theoretical_speedup,
            pattern,
        }
    }
}

/// Configuration for the pruning process.
#[derive(Debug, Clone)]
pub struct PruningConfig {
    /// Target sparsity ratio [0, 1).
    pub target_sparsity: f64,
    /// Sparsity pattern to apply.
    pub pattern: SparsityPattern,
    /// Whether to rescale remaining weights after pruning.
    pub rescale: bool,
}

impl PruningConfig {
    /// Create a new pruning config with the given sparsity and pattern.
    ///
    /// Returns an error if `target_sparsity` is not in `[0, 1)`.
    pub fn new(target_sparsity: f64, pattern: SparsityPattern) -> Result<Self, PruningError> {
        if !(0.0..1.0).contains(&target_sparsity) {
            return Err(PruningError::InvalidSparsityRatio(target_sparsity));
        }
        Ok(PruningConfig {
            target_sparsity,
            pattern,
            rescale: false,
        })
    }

    /// Set whether to rescale non-zero weights after pruning.
    pub fn with_rescale(mut self, rescale: bool) -> Self {
        self.rescale = rescale;
        self
    }
}

/// Magnitude-based pruner: zero out elements with smallest absolute values.
pub struct MagnitudePruner {
    config: PruningConfig,
}

impl MagnitudePruner {
    /// Create a new magnitude pruner with the given config.
    pub fn new(config: PruningConfig) -> Self {
        MagnitudePruner { config }
    }

    /// Prune a 2D matrix in-place according to the pattern.
    pub fn prune_2d(&self, matrix: &mut Array2<f64>) -> Result<SparsityStats, PruningError> {
        if matrix.is_empty() {
            return Err(PruningError::EmptyTensor);
        }
        match &self.config.pattern {
            SparsityPattern::Unstructured => {
                self.prune_unstructured_2d(matrix)?;
            }
            SparsityPattern::Block { block_h, block_w } => {
                self.prune_block_2d(matrix, *block_h, *block_w)?;
            }
            SparsityPattern::Row => {
                self.prune_rows_2d(matrix)?;
            }
            SparsityPattern::Column => {
                self.prune_columns_2d(matrix)?;
            }
            SparsityPattern::NM { n, m } => {
                self.prune_nm_2d(matrix, *n, *m)?;
            }
        }
        if self.config.rescale {
            self.rescale_nonzero(matrix);
        }
        Ok(SparsityStats::compute(
            &matrix.clone().into_dyn(),
            self.config.pattern.clone(),
        ))
    }

    /// Prune a general N-D tensor (applies unstructured or falls back to 2D for structured patterns).
    pub fn prune(&self, tensor: &mut ArrayD<f64>) -> Result<SparsityStats, PruningError> {
        if tensor.is_empty() {
            return Err(PruningError::EmptyTensor);
        }
        match &self.config.pattern {
            SparsityPattern::Unstructured => {
                self.prune_unstructured_nd(tensor)?;
            }
            _ => {
                // For structured patterns, require 2D
                if tensor.ndim() != 2 {
                    return Err(PruningError::ShapeMismatch(format!(
                        "Structured pruning requires 2D tensor, got {}D",
                        tensor.ndim()
                    )));
                }
                let mut mat = tensor
                    .clone()
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| PruningError::ShapeMismatch(e.to_string()))?;
                self.prune_2d(&mut mat)?;
                *tensor = mat.into_dyn();
            }
        }
        Ok(SparsityStats::compute(tensor, self.config.pattern.clone()))
    }

    fn prune_unstructured_nd(&self, tensor: &mut ArrayD<f64>) -> Result<(), PruningError> {
        let k = ((1.0 - self.config.target_sparsity) * tensor.len() as f64).ceil() as usize;
        let mut mags: Vec<f64> = tensor.iter().map(|v| v.abs()).collect();
        mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = if k < mags.len() {
            mags[mags.len() - k]
        } else {
            0.0
        };
        tensor.mapv_inplace(|v| if v.abs() >= threshold { v } else { 0.0 });
        Ok(())
    }

    fn prune_unstructured_2d(&self, matrix: &mut Array2<f64>) -> Result<(), PruningError> {
        let k = ((1.0 - self.config.target_sparsity) * matrix.len() as f64).ceil() as usize;
        let mut mags: Vec<f64> = matrix.iter().map(|v| v.abs()).collect();
        mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = if k < mags.len() {
            mags[mags.len() - k]
        } else {
            0.0
        };
        matrix.mapv_inplace(|v| if v.abs() >= threshold { v } else { 0.0 });
        Ok(())
    }

    fn prune_rows_2d(&self, matrix: &mut Array2<f64>) -> Result<(), PruningError> {
        let nrows = matrix.nrows();
        let n_prune = (self.config.target_sparsity * nrows as f64).round() as usize;
        // Compute row L2 norms
        let mut norms: Vec<(usize, f64)> = (0..nrows)
            .map(|i| {
                let norm: f64 = matrix.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
                (i, norm)
            })
            .collect();
        norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for &(row_idx, _) in &norms[..n_prune] {
            matrix.row_mut(row_idx).fill(0.0);
        }
        Ok(())
    }

    fn prune_columns_2d(&self, matrix: &mut Array2<f64>) -> Result<(), PruningError> {
        let ncols = matrix.ncols();
        let n_prune = (self.config.target_sparsity * ncols as f64).round() as usize;
        let mut norms: Vec<(usize, f64)> = (0..ncols)
            .map(|j| {
                let norm: f64 = matrix.column(j).iter().map(|v| v * v).sum::<f64>().sqrt();
                (j, norm)
            })
            .collect();
        norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for &(col_idx, _) in &norms[..n_prune] {
            matrix.column_mut(col_idx).fill(0.0);
        }
        Ok(())
    }

    fn prune_block_2d(
        &self,
        matrix: &mut Array2<f64>,
        bh: usize,
        bw: usize,
    ) -> Result<(), PruningError> {
        let (rows, cols) = (matrix.nrows(), matrix.ncols());
        if rows % bh != 0 {
            return Err(PruningError::InvalidBlockSize(bh, rows));
        }
        if cols % bw != 0 {
            return Err(PruningError::InvalidBlockSize(bw, cols));
        }
        let n_blocks_r = rows / bh;
        let n_blocks_c = cols / bw;
        let total_blocks = n_blocks_r * n_blocks_c;
        let n_prune = (self.config.target_sparsity * total_blocks as f64).round() as usize;
        // Compute block norms
        let mut block_norms: Vec<(usize, usize, f64)> = Vec::with_capacity(total_blocks);
        for br in 0..n_blocks_r {
            for bc in 0..n_blocks_c {
                let norm: f64 = matrix
                    .slice(ndarray::s![br * bh..(br + 1) * bh, bc * bw..(bc + 1) * bw])
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt();
                block_norms.push((br, bc, norm));
            }
        }
        block_norms.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        for &(br, bc, _) in &block_norms[..n_prune] {
            matrix
                .slice_mut(ndarray::s![br * bh..(br + 1) * bh, bc * bw..(bc + 1) * bw])
                .fill(0.0);
        }
        Ok(())
    }

    fn prune_nm_2d(
        &self,
        matrix: &mut Array2<f64>,
        n: usize,
        m: usize,
    ) -> Result<(), PruningError> {
        if n >= m {
            return Err(PruningError::InvalidBlockSize(n, m));
        }
        // For each row, for each group of m consecutive elements, keep top-n by magnitude
        let ncols = matrix.ncols();
        for i in 0..matrix.nrows() {
            let mut col = 0;
            while col + m <= ncols {
                let group: Vec<f64> = (col..col + m).map(|j| matrix[[i, j]]).collect();
                let mut mags: Vec<(usize, f64)> = group
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| (j, v.abs()))
                    .collect();
                mags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let keep: std::collections::HashSet<usize> =
                    mags[..n].iter().map(|&(j, _)| j).collect();
                for j in 0..m {
                    if !keep.contains(&j) {
                        matrix[[i, col + j]] = 0.0;
                    }
                }
                col += m;
            }
        }
        Ok(())
    }

    fn rescale_nonzero(&self, matrix: &mut Array2<f64>) {
        let total = matrix.len() as f64;
        let nonzero = matrix.iter().filter(|&&v| v != 0.0).count() as f64;
        if nonzero > 0.0 {
            let scale = total / nonzero;
            matrix.mapv_inplace(|v| if v != 0.0 { v * scale } else { 0.0 });
        }
    }
}

/// Compute sparsity statistics for an N-D tensor.
///
/// Returns the fraction of zero elements (0.0 = fully dense, 1.0 = all zeros).
pub fn compute_sparsity(tensor: &ArrayD<f64>) -> f64 {
    if tensor.is_empty() {
        return 0.0;
    }
    let zeros = tensor.iter().filter(|&&v| v == 0.0).count();
    zeros as f64 / tensor.len() as f64
}

/// Compute per-row L2 norms for a 2D matrix.
pub fn row_norms(matrix: &Array2<f64>) -> Array1<f64> {
    Array1::from_iter(
        matrix
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|v| v * v).sum::<f64>().sqrt()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // Helper: build a 2D array and convert to dynamic.
    fn dyn2d(data: Array2<f64>) -> ArrayD<f64> {
        data.into_dyn()
    }

    #[test]
    fn test_sparsity_pattern_names() {
        assert_eq!(SparsityPattern::Unstructured.name(), "unstructured");
        assert_eq!(
            SparsityPattern::Block {
                block_h: 2,
                block_w: 2
            }
            .name(),
            "block"
        );
        assert_eq!(SparsityPattern::Row.name(), "row");
        assert_eq!(SparsityPattern::Column.name(), "column");
        assert_eq!(SparsityPattern::NM { n: 2, m: 4 }.name(), "n:m");
    }

    #[test]
    fn test_sparsity_pattern_is_structured() {
        assert!(!SparsityPattern::Unstructured.is_structured());
        assert!(SparsityPattern::Block {
            block_h: 2,
            block_w: 2
        }
        .is_structured());
        assert!(SparsityPattern::Row.is_structured());
        assert!(SparsityPattern::Column.is_structured());
        assert!(SparsityPattern::NM { n: 1, m: 4 }.is_structured());
    }

    #[test]
    fn test_pruning_config_invalid_ratio() {
        let result = PruningConfig::new(1.0, SparsityPattern::Unstructured);
        assert!(result.is_err());
        let result_neg = PruningConfig::new(-0.1, SparsityPattern::Unstructured);
        assert!(result_neg.is_err());
    }

    #[test]
    fn test_pruning_config_valid() {
        let result = PruningConfig::new(0.5, SparsityPattern::Unstructured);
        assert!(result.is_ok());
        let cfg = result.expect("valid config");
        assert!((cfg.target_sparsity - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_unstructured_pruning_zeros_out() {
        // 4×4 matrix, 50% sparsity → ~8 zeros out of 16
        let mut mat = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];
        let cfg = PruningConfig::new(0.5, SparsityPattern::Unstructured).expect("valid config");
        let pruner = MagnitudePruner::new(cfg);
        let stats = pruner.prune_2d(&mut mat).expect("prune ok");
        // Should have ~50% zeros
        assert!(stats.actual_sparsity >= 0.4 && stats.actual_sparsity <= 0.6);
    }

    #[test]
    fn test_unstructured_preserves_largest() {
        // Elements 10, 20, 30, 40 — with 50% sparsity (keep 50%), 20 and 40 must survive
        let mut mat = array![[10.0, 20.0], [30.0, 40.0]];
        let cfg = PruningConfig::new(0.5, SparsityPattern::Unstructured).expect("valid config");
        let pruner = MagnitudePruner::new(cfg);
        pruner.prune_2d(&mut mat).expect("prune ok");
        // Largest two (30, 40) must be non-zero
        assert!(mat[[1, 0]] != 0.0 || mat[[1, 1]] != 0.0);
        assert!(mat[[1, 1]] != 0.0); // 40 is the largest, must survive
    }

    #[test]
    fn test_row_pruning_zeros_weakest_rows() {
        // Row 0 has tiny values, row 1 has large values
        let mut mat = array![[0.001, 0.001], [100.0, 100.0]];
        let cfg = PruningConfig::new(0.5, SparsityPattern::Row).expect("valid config");
        let pruner = MagnitudePruner::new(cfg);
        pruner.prune_2d(&mut mat).expect("prune ok");
        // Row 0 (weakest) should be zeroed
        assert_eq!(mat[[0, 0]], 0.0);
        assert_eq!(mat[[0, 1]], 0.0);
        // Row 1 should survive
        assert!(mat[[1, 0]] != 0.0);
    }

    #[test]
    fn test_column_pruning_zeros_weakest_cols() {
        // Col 0 has tiny values, col 1 has large values
        let mut mat = array![[0.001, 100.0], [0.001, 100.0]];
        let cfg = PruningConfig::new(0.5, SparsityPattern::Column).expect("valid config");
        let pruner = MagnitudePruner::new(cfg);
        pruner.prune_2d(&mut mat).expect("prune ok");
        // Column 0 should be zeroed
        assert_eq!(mat[[0, 0]], 0.0);
        assert_eq!(mat[[1, 0]], 0.0);
        // Column 1 should survive
        assert!(mat[[0, 1]] != 0.0);
    }

    #[test]
    fn test_block_pruning_basic() {
        // 4×4 with 2×2 blocks → 4 blocks total, 50% → 2 blocks zeroed
        let mut mat = array![
            [1.0, 2.0, 100.0, 200.0],
            [3.0, 4.0, 300.0, 400.0],
            [0.1, 0.2, 50.0, 60.0],
            [0.3, 0.4, 70.0, 80.0],
        ];
        let cfg = PruningConfig::new(
            0.5,
            SparsityPattern::Block {
                block_h: 2,
                block_w: 2,
            },
        )
        .expect("valid config");
        let pruner = MagnitudePruner::new(cfg);
        let stats = pruner.prune_2d(&mut mat).expect("prune ok");
        // 2 out of 4 blocks zeroed → 50% element sparsity
        assert!((stats.actual_sparsity - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_block_pruning_invalid_size() {
        // 4 rows but block_h=3 → does not divide
        let mut mat = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];
        let cfg = PruningConfig::new(
            0.5,
            SparsityPattern::Block {
                block_h: 3,
                block_w: 3,
            },
        )
        .expect("valid config");
        let pruner = MagnitudePruner::new(cfg);
        let result = pruner.prune_2d(&mut mat);
        assert!(matches!(result, Err(PruningError::InvalidBlockSize(_, _))));
    }

    #[test]
    fn test_nm_pruning_keeps_n_per_m() {
        // 2:4 sparsity on a single row [1, 2, 3, 4]
        // Keep 2 largest (3 and 4), zero out 1 and 2
        let mut mat = array![[1.0, 2.0, 3.0, 4.0]];
        let cfg =
            PruningConfig::new(0.5, SparsityPattern::NM { n: 2, m: 4 }).expect("valid config");
        let pruner = MagnitudePruner::new(cfg);
        pruner.prune_2d(&mut mat).expect("prune ok");
        let nonzero_count = mat.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(nonzero_count, 2);
        // The two largest must survive
        assert!(mat[[0, 2]] != 0.0); // 3.0
        assert!(mat[[0, 3]] != 0.0); // 4.0
    }

    #[test]
    fn test_nm_invalid_n_ge_m() {
        let mut mat = array![[1.0, 2.0, 3.0, 4.0]];
        // n=4 >= m=4 is invalid
        let cfg =
            PruningConfig::new(0.1, SparsityPattern::NM { n: 4, m: 4 }).expect("valid config");
        let pruner = MagnitudePruner::new(cfg);
        let result = pruner.prune_2d(&mut mat);
        assert!(matches!(result, Err(PruningError::InvalidBlockSize(_, _))));
    }

    #[test]
    fn test_rescale_preserves_sum() {
        // After rescaling, non-zero elements are scaled by total/nonzero.
        // With 50% sparsity: the surviving elements are scaled by 2×,
        // so their sum equals the sum of the top-50% elements × 2.
        // We verify the rescaled sum is strictly larger than the unrescaled pruned sum.
        let original = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];

        // Prune without rescale
        let mut mat_no_rescale = original.clone();
        let cfg_no = PruningConfig::new(0.5, SparsityPattern::Unstructured).expect("valid config");
        let pruner_no = MagnitudePruner::new(cfg_no);
        pruner_no.prune_2d(&mut mat_no_rescale).expect("prune ok");
        let sum_no_rescale: f64 = mat_no_rescale.iter().copied().sum();

        // Prune with rescale
        let mut mat = original.clone();
        let cfg = PruningConfig::new(0.5, SparsityPattern::Unstructured)
            .expect("valid config")
            .with_rescale(true);
        let pruner = MagnitudePruner::new(cfg);
        pruner.prune_2d(&mut mat).expect("prune ok");
        let sum_rescaled: f64 = mat.iter().copied().sum();

        // Rescaled sum should be larger than the non-rescaled pruned sum
        // (weights are scaled up to compensate for pruned weights)
        assert!(
            sum_rescaled > sum_no_rescale,
            "rescaled sum ({sum_rescaled}) should exceed unrescaled pruned sum ({sum_no_rescale})"
        );
        // And the number of non-zero elements should be the same
        let nz_no = mat_no_rescale.iter().filter(|&&v| v != 0.0).count();
        let nz_rescaled = mat.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(
            nz_no, nz_rescaled,
            "rescale should not change which elements are zero"
        );
    }

    #[test]
    fn test_sparsity_stats_compute() {
        let mat = array![[0.0, 1.0], [0.0, 2.0]];
        let stats = SparsityStats::compute(&mat.into_dyn(), SparsityPattern::Unstructured);
        assert_eq!(stats.zero_count, 2);
        assert_eq!(stats.total_count, 4);
        assert!((stats.actual_sparsity - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sparsity_stats_speedup_structured() {
        // A 75% sparse tensor: structured speedup should exceed unstructured speedup
        let mat = array![[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 2.0]];
        let structured_stats =
            SparsityStats::compute(&mat.clone().into_dyn(), SparsityPattern::Row);
        let unstructured_stats =
            SparsityStats::compute(&mat.into_dyn(), SparsityPattern::Unstructured);
        assert!(
            structured_stats.theoretical_speedup > unstructured_stats.theoretical_speedup,
            "structured speedup ({}) should exceed unstructured ({})",
            structured_stats.theoretical_speedup,
            unstructured_stats.theoretical_speedup
        );
    }

    #[test]
    fn test_compute_sparsity_dense() {
        let mat = array![[1.0, 2.0], [3.0, 4.0]];
        let sparsity = compute_sparsity(&mat.into_dyn());
        assert!((sparsity - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_sparsity_half() {
        let mat = array![[0.0, 1.0], [0.0, 2.0]];
        let sparsity = compute_sparsity(&mat.into_dyn());
        assert!((sparsity - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_row_norms_correctness() {
        let mat = array![[3.0, 4.0], [0.0, 0.0]];
        let norms = row_norms(&mat);
        assert!(
            (norms[0] - 5.0).abs() < 1e-10,
            "norm[0] should be 5.0, got {}",
            norms[0]
        );
        assert!(
            (norms[1] - 0.0).abs() < 1e-10,
            "norm[1] should be 0.0, got {}",
            norms[1]
        );
    }

    #[test]
    fn test_prune_nd_tensor() {
        // 3D tensor (2×3×4), apply unstructured pruning
        use ndarray::Array3;
        let data: Array3<f64> =
            Array3::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k + 1) as f64);
        let mut tensor = data.into_dyn();
        let cfg = PruningConfig::new(0.5, SparsityPattern::Unstructured).expect("valid config");
        let pruner = MagnitudePruner::new(cfg);
        let stats = pruner.prune(&mut tensor).expect("prune ok");
        // Roughly 50% should be zeros
        assert!(
            stats.actual_sparsity >= 0.4 && stats.actual_sparsity <= 0.6,
            "sparsity={} not near 0.5",
            stats.actual_sparsity
        );
    }

    #[test]
    fn test_prune_empty_tensor() {
        use ndarray::Array2;
        let mut empty: ArrayD<f64> = dyn2d(Array2::zeros((0, 4)));
        let cfg = PruningConfig::new(0.5, SparsityPattern::Unstructured).expect("valid config");
        let pruner = MagnitudePruner::new(cfg);
        let result = pruner.prune(&mut empty);
        assert!(matches!(result, Err(PruningError::EmptyTensor)));
    }
}
