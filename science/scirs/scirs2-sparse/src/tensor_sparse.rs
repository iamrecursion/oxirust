//! Tensor-based sparse operations
//!
//! This module provides operations for sparse tensors (multi-dimensional sparse arrays):
//! - Sparse tensor construction and manipulation
//! - Tensor contractions and products
//! - Mode-n unfolding and folding
//! - Tucker and CP decompositions

use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::numeric::{Float, SparseElement, Zero};
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::Div;

/// Sparse tensor in COO (coordinate) format
#[derive(Debug, Clone)]
pub struct SparseTensor<T> {
    /// Indices for each non-zero element (one vector per dimension)
    pub indices: Vec<Vec<usize>>,
    /// Values of non-zero elements
    pub values: Vec<T>,
    /// Shape of the tensor
    pub shape: Vec<usize>,
}

impl<T> SparseTensor<T>
where
    T: Float + SparseElement + Debug + Copy + std::iter::Sum + 'static,
{
    /// Create a new sparse tensor
    pub fn new(indices: Vec<Vec<usize>>, values: Vec<T>, shape: Vec<usize>) -> SparseResult<Self> {
        // Validate inputs
        if indices.is_empty() {
            return Err(SparseError::ValueError(
                "Indices cannot be empty".to_string(),
            ));
        }

        let ndim = indices.len();
        if ndim != shape.len() {
            return Err(SparseError::ValueError(
                "Number of index dimensions must match shape dimensions".to_string(),
            ));
        }

        let nnz = values.len();
        for idx_dim in &indices {
            if idx_dim.len() != nnz {
                return Err(SparseError::ValueError(
                    "All index dimensions must have same length as values".to_string(),
                ));
            }
        }

        // Validate indices are within bounds
        for (dim, idx_vec) in indices.iter().enumerate() {
            for &idx in idx_vec {
                if idx >= shape[dim] {
                    return Err(SparseError::ValueError(format!(
                        "Index {} in dimension {} exceeds shape {}",
                        idx, dim, shape[dim]
                    )));
                }
            }
        }

        Ok(Self {
            indices,
            values,
            shape,
        })
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get the total number of elements
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get an element at the specified indices
    pub fn get(&self, indices: &[usize]) -> T {
        if indices.len() != self.ndim() {
            return T::sparse_zero();
        }

        // Search for the element
        for i in 0..self.nnz() {
            let mut found = true;
            for (dim, &idx) in indices.iter().enumerate() {
                if self.indices[dim][i] != idx {
                    found = false;
                    break;
                }
            }
            if found {
                return self.values[i];
            }
        }

        T::sparse_zero()
    }

    /// Mode-n unfolding (matricization)
    ///
    /// Unfolds the tensor along the specified mode into a matrix.
    pub fn unfold(&self, mode: usize) -> SparseResult<CsrArray<T>> {
        if mode >= self.ndim() {
            return Err(SparseError::ValueError(format!(
                "Mode {} exceeds tensor dimensions {}",
                mode,
                self.ndim()
            )));
        }

        // Calculate matrix dimensions
        let nrows = self.shape[mode];
        let ncols: usize = self
            .shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != mode)
            .map(|(_, &s)| s)
            .product();

        // Build row/col/data for matrix
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();

        for elem_idx in 0..self.nnz() {
            let row = self.indices[mode][elem_idx];

            // Calculate column index from other dimensions
            let mut col = 0;
            let mut stride = 1;

            for dim in (0..self.ndim()).rev() {
                if dim != mode {
                    col += self.indices[dim][elem_idx] * stride;
                    stride *= self.shape[dim];
                }
            }

            rows.push(row);
            cols.push(col);
            data.push(self.values[elem_idx]);
        }

        CsrArray::from_triplets(&rows, &cols, &data, (nrows, ncols), false)
    }

    /// Fold a matrix back into a tensor along the specified mode
    pub fn fold(matrix: &dyn SparseArray<T>, shape: Vec<usize>, mode: usize) -> SparseResult<Self> {
        if mode >= shape.len() {
            return Err(SparseError::ValueError(format!(
                "Mode {} exceeds tensor dimensions {}",
                mode,
                shape.len()
            )));
        }

        let (nrows, ncols) = matrix.shape();

        if nrows != shape[mode] {
            return Err(SparseError::ValueError(
                "Matrix rows must match mode dimension".to_string(),
            ));
        }

        let expected_cols: usize = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != mode)
            .map(|(_, &s)| s)
            .product();

        if ncols != expected_cols {
            return Err(SparseError::ValueError(
                "Matrix columns must match product of other dimensions".to_string(),
            ));
        }

        // Get non-zero elements from matrix
        let (mat_rows, mat_cols, mat_values) = matrix.find();

        let ndim = shape.len();
        let mut indices = vec![Vec::new(); ndim];
        let mut values = Vec::new();

        for (i, (&row, &col)) in mat_rows.iter().zip(mat_cols.iter()).enumerate() {
            // Set mode index
            indices[mode].push(row);

            // Decode column into other dimension indices
            let mut remaining = col;
            let mut other_dims: Vec<usize> = (0..ndim).filter(|&d| d != mode).collect();
            other_dims.reverse();

            for &dim in &other_dims {
                let idx = remaining % shape[dim];
                indices[dim].push(idx);
                remaining /= shape[dim];
            }

            values.push(mat_values[i]);
        }

        Self::new(indices, values, shape)
    }

    /// Tensor-matrix product along specified mode
    ///
    /// Multiplies the tensor by a matrix along the given mode.
    pub fn mode_product(&self, matrix: &CsrArray<T>, mode: usize) -> SparseResult<Self> {
        if mode >= self.ndim() {
            return Err(SparseError::ValueError(format!(
                "Mode {} exceeds tensor dimensions {}",
                mode,
                self.ndim()
            )));
        }

        let (mat_rows, mat_cols) = matrix.shape();
        if mat_cols != self.shape[mode] {
            return Err(SparseError::ValueError(
                "Matrix columns must match tensor mode dimension".to_string(),
            ));
        }

        // Unfold tensor along mode
        let unfolded = self.unfold(mode)?;

        // Multiply: result = matrix * unfolded
        let result_matrix = matrix.dot(&unfolded)?;

        // Update shape with new mode dimension
        let mut new_shape = self.shape.clone();
        new_shape[mode] = mat_rows;

        // Fold back into tensor
        Self::fold(result_matrix.as_ref(), new_shape, mode)
    }

    /// Inner product of two sparse tensors
    pub fn inner_product(&self, other: &Self) -> SparseResult<T> {
        if self.shape != other.shape {
            return Err(SparseError::ValueError(
                "Tensors must have the same shape for inner product".to_string(),
            ));
        }

        let mut result = T::sparse_zero();

        // Build index map for efficient lookup
        let mut index_map: HashMap<Vec<usize>, T> = HashMap::new();
        for i in 0..other.nnz() {
            let indices: Vec<usize> = (0..self.ndim()).map(|d| other.indices[d][i]).collect();
            index_map.insert(indices, other.values[i]);
        }

        // Sum products of matching non-zeros
        for i in 0..self.nnz() {
            let indices: Vec<usize> = (0..self.ndim()).map(|d| self.indices[d][i]).collect();

            if let Some(&other_val) = index_map.get(&indices) {
                result = result + self.values[i] * other_val;
            }
        }

        Ok(result)
    }

    /// Frobenius norm of the tensor
    pub fn frobenius_norm(&self) -> T {
        let sum_sq: T = self.values.iter().map(|&v| v * v).sum();
        sum_sq.sqrt()
    }
}

/// Tucker decomposition result
#[derive(Debug, Clone)]
pub struct TuckerDecomposition<T>
where
    T: SparseElement + Div<Output = T> + PartialOrd + Zero + 'static,
{
    /// Core tensor
    pub core: SparseTensor<T>,
    /// Factor matrices (one per mode)
    pub factors: Vec<CsrArray<T>>,
}

/// CP (CANDECOMP/PARAFAC) decomposition result
#[derive(Debug, Clone)]
pub struct CPDecomposition<T>
where
    T: SparseElement + Div<Output = T> + PartialOrd + Zero + 'static,
{
    /// Weights of rank-1 components
    pub weights: Vec<T>,
    /// Factor matrices (one per mode)
    pub factors: Vec<CsrArray<T>>,
    /// Rank of decomposition
    pub rank: usize,
}

/// Compute Khatri-Rao product of two matrices
///
/// The Khatri-Rao product is a column-wise Kronecker product.
pub fn khatri_rao_product<T>(a: &CsrArray<T>, b: &CsrArray<T>) -> SparseResult<CsrArray<T>>
where
    T: Float + SparseElement + Debug + Copy + std::iter::Sum + 'static,
{
    let (rows_a, cols_a) = a.shape();
    let (rows_b, cols_b) = b.shape();

    if cols_a != cols_b {
        return Err(SparseError::ValueError(
            "Matrices must have the same number of columns for Khatri-Rao product".to_string(),
        ));
    }

    let ncols = cols_a;
    let nrows = rows_a * rows_b;

    let mut result_rows = Vec::new();
    let mut result_cols = Vec::new();
    let mut result_data = Vec::new();

    // For each column
    for col in 0..ncols {
        // Get column vectors
        let mut col_a = vec![T::sparse_zero(); rows_a];
        let mut col_b = vec![T::sparse_zero(); rows_b];

        for row in 0..rows_a {
            col_a[row] = a.get(row, col);
        }

        for row in 0..rows_b {
            col_b[row] = b.get(row, col);
        }

        // Compute Kronecker product of columns
        for i in 0..rows_a {
            for j in 0..rows_b {
                let value = col_a[i] * col_b[j];
                if !scirs2_core::SparseElement::is_zero(&value) {
                    result_rows.push(i * rows_b + j);
                    result_cols.push(col);
                    result_data.push(value);
                }
            }
        }
    }

    CsrArray::from_triplets(
        &result_rows,
        &result_cols,
        &result_data,
        (nrows, ncols),
        false,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_tensor() -> SparseTensor<f64> {
        // Create a simple 2x3x4 sparse tensor with a few non-zero elements
        let indices = vec![
            vec![0, 0, 1, 1], // dimension 0
            vec![0, 1, 0, 2], // dimension 1
            vec![0, 1, 2, 3], // dimension 2
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 3, 4];

        SparseTensor::new(indices, values, shape).expect("Failed to create tensor")
    }

    #[test]
    fn test_tensor_creation() {
        let tensor = create_test_tensor();

        assert_eq!(tensor.ndim(), 3);
        assert_eq!(tensor.nnz(), 4);
        assert_eq!(tensor.size(), 24);
        assert_eq!(tensor.shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_tensor_get() {
        let tensor = create_test_tensor();

        assert_relative_eq!(tensor.get(&[0, 0, 0]), 1.0);
        assert_relative_eq!(tensor.get(&[0, 1, 1]), 2.0);
        assert_relative_eq!(tensor.get(&[1, 0, 2]), 3.0);
        assert_relative_eq!(tensor.get(&[1, 2, 3]), 4.0);
        assert_relative_eq!(tensor.get(&[0, 0, 1]), 0.0); // zero element
    }

    #[test]
    fn test_unfold() {
        let tensor = create_test_tensor();

        // Unfold along mode 0
        let unfolded = tensor.unfold(0).expect("Failed to unfold");
        assert_eq!(unfolded.shape(), (2, 12)); // 2 x (3*4)

        // Unfold along mode 1
        let unfolded1 = tensor.unfold(1).expect("Failed to unfold");
        assert_eq!(unfolded1.shape(), (3, 8)); // 3 x (2*4)

        // Unfold along mode 2
        let unfolded2 = tensor.unfold(2).expect("Failed to unfold");
        assert_eq!(unfolded2.shape(), (4, 6)); // 4 x (2*3)
    }

    #[test]
    fn test_fold_unfold_roundtrip() {
        let tensor = create_test_tensor();

        for mode in 0..tensor.ndim() {
            let unfolded = tensor.unfold(mode).expect("Failed to unfold");
            let refolded =
                SparseTensor::fold(&unfolded, tensor.shape.clone(), mode).expect("Failed to fold");

            // Check that we get the same values back
            assert_eq!(refolded.nnz(), tensor.nnz());

            for i in 0..tensor.nnz() {
                let indices: Vec<usize> =
                    (0..tensor.ndim()).map(|d| tensor.indices[d][i]).collect();
                assert_relative_eq!(
                    tensor.get(&indices),
                    refolded.get(&indices),
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_inner_product() {
        let tensor1 = create_test_tensor();
        let tensor2 = create_test_tensor();

        let ip = tensor1.inner_product(&tensor2).expect("Failed");

        // Inner product with itself should equal sum of squares
        let sum_sq: f64 = tensor1.values.iter().map(|&v| v * v).sum();
        assert_relative_eq!(ip, sum_sq, epsilon = 1e-10);
    }

    #[test]
    fn test_frobenius_norm() {
        let tensor = create_test_tensor();

        let norm = tensor.frobenius_norm();

        // Should be sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(30)
        let expected = (1.0f64 + 4.0 + 9.0 + 16.0).sqrt();
        assert_relative_eq!(norm, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_khatri_rao_product() {
        // Create two small matrices
        let rows_a = vec![0, 0, 1];
        let cols_a = vec![0, 1, 0];
        let data_a = vec![1.0, 2.0, 3.0];
        let a = CsrArray::from_triplets(&rows_a, &cols_a, &data_a, (2, 2), false).expect("Failed");

        let rows_b = vec![0, 1, 1];
        let cols_b = vec![0, 0, 1];
        let data_b = vec![4.0, 5.0, 6.0];
        let b = CsrArray::from_triplets(&rows_b, &cols_b, &data_b, (2, 2), false).expect("Failed");

        let result = khatri_rao_product(&a, &b).expect("Failed");

        // Result should be 4x2 (2*2 rows, same columns)
        assert_eq!(result.shape(), (4, 2));
        assert!(result.nnz() > 0);
    }
}
