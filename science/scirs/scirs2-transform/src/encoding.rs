//! Categorical data encoding utilities
//!
//! This module provides methods for encoding categorical data into numerical
//! formats suitable for machine learning algorithms.

use scirs2_core::ndarray::{Array2, ArrayBase, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use std::collections::HashMap;

use crate::error::{Result, TransformError};

/// Simple sparse matrix representation in COO (Coordinate) format
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    /// Shape of the matrix (rows, cols)
    pub shape: (usize, usize),
    /// Row indices of non-zero values
    pub row_indices: Vec<usize>,
    /// Column indices of non-zero values
    pub col_indices: Vec<usize>,
    /// Non-zero values
    pub values: Vec<f64>,
}

impl SparseMatrix {
    /// Create a new empty sparse matrix
    pub fn new(shape: (usize, usize)) -> Self {
        SparseMatrix {
            shape,
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Add a non-zero value at (row, col)
    pub fn push(&mut self, row: usize, col: usize, value: f64) {
        if row < self.shape.0 && col < self.shape.1 && value != 0.0 {
            self.row_indices.push(row);
            self.col_indices.push(col);
            self.values.push(value);
        }
    }

    /// Convert to dense Array2
    pub fn to_dense(&self) -> Array2<f64> {
        let mut dense = Array2::zeros(self.shape);
        for ((&row, &col), &val) in self
            .row_indices
            .iter()
            .zip(self.col_indices.iter())
            .zip(self.values.iter())
        {
            dense[[row, col]] = val;
        }
        dense
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

/// Output format for encoded data
#[derive(Debug, Clone)]
pub enum EncodedOutput {
    /// Dense matrix representation
    Dense(Array2<f64>),
    /// Sparse matrix representation
    Sparse(SparseMatrix),
}

impl EncodedOutput {
    /// Convert to dense matrix (creates copy if sparse)
    pub fn to_dense(&self) -> Array2<f64> {
        match self {
            EncodedOutput::Dense(arr) => arr.clone(),
            EncodedOutput::Sparse(sparse) => sparse.to_dense(),
        }
    }

    /// Get shape of the output
    pub fn shape(&self) -> (usize, usize) {
        match self {
            EncodedOutput::Dense(arr) => (arr.nrows(), arr.ncols()),
            EncodedOutput::Sparse(sparse) => sparse.shape,
        }
    }
}

/// OneHotEncoder for converting categorical features to binary features
///
/// This transformer converts categorical features into a one-hot encoded representation,
/// where each category is represented by a binary feature.
pub struct OneHotEncoder {
    /// Categories for each feature (learned during fit)
    categories_: Option<Vec<Vec<u64>>>,
    /// Whether to drop one category per feature to avoid collinearity
    drop: Option<String>,
    /// Whether to handle unknown categories
    handleunknown: String,
    /// Whether to return sparse matrix output
    sparse: bool,
}

impl OneHotEncoder {
    /// Creates a new OneHotEncoder
    ///
    /// # Arguments
    /// * `drop` - Strategy for dropping categories ('first', 'if_binary', or None)
    /// * `handleunknown` - How to handle unknown categories ('error' or 'ignore')
    /// * `sparse` - Whether to return sparse arrays
    ///
    /// # Returns
    /// * A new OneHotEncoder instance
    pub fn new(_drop: Option<String>, handleunknown: &str, sparse: bool) -> Result<Self> {
        if let Some(ref drop_strategy) = _drop {
            if drop_strategy != "first" && drop_strategy != "if_binary" {
                return Err(TransformError::InvalidInput(
                    "_drop must be 'first', 'if_binary', or None".to_string(),
                ));
            }
        }

        if handleunknown != "error" && handleunknown != "ignore" {
            return Err(TransformError::InvalidInput(
                "handleunknown must be 'error' or 'ignore'".to_string(),
            ));
        }

        Ok(OneHotEncoder {
            categories_: None,
            drop: _drop,
            handleunknown: handleunknown.to_string(),
            sparse,
        })
    }

    /// Creates a OneHotEncoder with default settings
    pub fn with_defaults() -> Self {
        Self::new(None, "error", false).expect("Operation failed")
    }

    /// Fits the OneHotEncoder to the input data
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = NumCast::from(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        let mut categories = Vec::with_capacity(n_features);

        for j in 0..n_features {
            // Collect unique values for this feature
            let mut unique_values: Vec<u64> = x_u64.column(j).to_vec();
            unique_values.sort_unstable();
            unique_values.dedup();

            categories.push(unique_values);
        }

        self.categories_ = Some(categories);
        Ok(())
    }

    /// Transforms the input data using the fitted OneHotEncoder
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<EncodedOutput>` - The one-hot encoded data (dense or sparse)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<EncodedOutput>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = NumCast::from(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if self.categories_.is_none() {
            return Err(TransformError::TransformationError(
                "OneHotEncoder has not been fitted".to_string(),
            ));
        }

        let categories = self.categories_.as_ref().expect("Operation failed");

        if n_features != categories.len() {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but OneHotEncoder was fitted with {} features",
                n_features,
                categories.len()
            )));
        }

        // Calculate total number of output features
        let mut total_features = 0;
        for (j, feature_categories) in categories.iter().enumerate() {
            let n_cats = feature_categories.len();

            // Apply drop strategy
            let n_output_cats = match &self.drop {
                Some(strategy) if strategy == "first" => n_cats.saturating_sub(1),
                Some(strategy) if strategy == "if_binary" && n_cats == 2 => 1,
                _ => n_cats,
            };

            if n_output_cats == 0 {
                return Err(TransformError::InvalidInput(format!(
                    "Feature {j} has only one category after dropping"
                )));
            }

            total_features += n_output_cats;
        }

        // Create mappings from category values to column indices
        let mut category_mappings = Vec::new();
        let mut current_col = 0;

        for feature_categories in categories.iter() {
            let mut mapping = HashMap::new();
            let n_cats = feature_categories.len();

            // Determine how many categories to keep
            let (start_idx, n_output_cats) = match &self.drop {
                Some(strategy) if strategy == "first" => (1, n_cats.saturating_sub(1)),
                Some(strategy) if strategy == "if_binary" && n_cats == 2 => (0, 1),
                _ => (0, n_cats),
            };

            for (cat_idx, &category) in feature_categories.iter().enumerate() {
                if cat_idx >= start_idx && cat_idx < start_idx + n_output_cats {
                    mapping.insert(category, current_col + cat_idx - start_idx);
                }
            }

            category_mappings.push(mapping);
            current_col += n_output_cats;
        }

        // Create output based on sparse setting
        if self.sparse {
            // Sparse output
            let mut sparse_matrix = SparseMatrix::new((n_samples, total_features));

            for i in 0..n_samples {
                for j in 0..n_features {
                    let value = x_u64[[i, j]];

                    if let Some(&col_idx) = category_mappings[j].get(&value) {
                        sparse_matrix.push(i, col_idx, 1.0);
                    } else {
                        // Check if this is a dropped category
                        let feature_categories = &categories[j];
                        let is_dropped_category = match &self.drop {
                            Some(strategy) if strategy == "first" => {
                                !feature_categories.is_empty() && value == feature_categories[0]
                            }
                            Some(strategy)
                                if strategy == "if_binary" && feature_categories.len() == 2 =>
                            {
                                feature_categories.len() == 2 && value == feature_categories[1]
                            }
                            _ => false,
                        };

                        if !is_dropped_category && self.handleunknown == "error" {
                            return Err(TransformError::InvalidInput(format!(
                                "Found unknown category {value} in feature {j}"
                            )));
                        }
                        // If it's a dropped category or handleunknown == "ignore", we don't add anything (sparse)
                    }
                }
            }

            Ok(EncodedOutput::Sparse(sparse_matrix))
        } else {
            // Dense output
            let mut transformed = Array2::zeros((n_samples, total_features));

            for i in 0..n_samples {
                for j in 0..n_features {
                    let value = x_u64[[i, j]];

                    if let Some(&col_idx) = category_mappings[j].get(&value) {
                        transformed[[i, col_idx]] = 1.0;
                    } else {
                        // Check if this is a dropped category
                        let feature_categories = &categories[j];
                        let is_dropped_category = match &self.drop {
                            Some(strategy) if strategy == "first" => {
                                !feature_categories.is_empty() && value == feature_categories[0]
                            }
                            Some(strategy)
                                if strategy == "if_binary" && feature_categories.len() == 2 =>
                            {
                                feature_categories.len() == 2 && value == feature_categories[1]
                            }
                            _ => false,
                        };

                        if !is_dropped_category && self.handleunknown == "error" {
                            return Err(TransformError::InvalidInput(format!(
                                "Found unknown category {value} in feature {j}"
                            )));
                        }
                        // If it's a dropped category or handleunknown == "ignore", we just leave it as 0
                    }
                }
            }

            Ok(EncodedOutput::Dense(transformed))
        }
    }

    /// Fits the OneHotEncoder to the input data and transforms it
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<EncodedOutput>` - The one-hot encoded data (dense or sparse)
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<EncodedOutput>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Convenience method that always returns dense output for backward compatibility
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The one-hot encoded data as dense matrix
    pub fn transform_dense<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        Ok(self.transform(x)?.to_dense())
    }

    /// Convenience method that fits and transforms returning dense output
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The one-hot encoded data as dense matrix
    pub fn fit_transform_dense<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        Ok(self.fit_transform(x)?.to_dense())
    }

    /// Returns the categories for each feature
    ///
    /// # Returns
    /// * `Option<&Vec<Vec<u64>>>` - The categories for each feature
    pub fn categories(&self) -> Option<&Vec<Vec<u64>>> {
        self.categories_.as_ref()
    }

    /// Gets the feature names for the transformed output
    ///
    /// # Arguments
    /// * `inputfeatures` - Names of input features
    ///
    /// # Returns
    /// * `Result<Vec<String>>` - Names of output features
    pub fn get_feature_names(&self, inputfeatures: Option<&[String]>) -> Result<Vec<String>> {
        if self.categories_.is_none() {
            return Err(TransformError::TransformationError(
                "OneHotEncoder has not been fitted".to_string(),
            ));
        }

        let categories = self.categories_.as_ref().expect("Operation failed");
        let mut feature_names = Vec::new();

        for (j, feature_categories) in categories.iter().enumerate() {
            let feature_name = if let Some(names) = inputfeatures {
                if j < names.len() {
                    names[j].clone()
                } else {
                    format!("x{j}")
                }
            } else {
                format!("x{j}")
            };

            let n_cats = feature_categories.len();

            // Determine which categories to include based on drop strategy
            let (start_idx, n_output_cats) = match &self.drop {
                Some(strategy) if strategy == "first" => (1, n_cats.saturating_sub(1)),
                Some(strategy) if strategy == "if_binary" && n_cats == 2 => (0, 1),
                _ => (0, n_cats),
            };

            for &category in feature_categories
                .iter()
                .skip(start_idx)
                .take(n_output_cats)
            {
                feature_names.push(format!("{feature_name}_cat_{category}"));
            }
        }

        Ok(feature_names)
    }
}

/// OrdinalEncoder for converting categorical features to ordinal integers
///
/// This transformer converts categorical features into ordinal integers,
/// where each category is assigned a unique integer.
pub struct OrdinalEncoder {
    /// Categories for each feature (learned during fit)
    categories_: Option<Vec<Vec<u64>>>,
    /// How to handle unknown categories
    handleunknown: String,
    /// Value to use for unknown categories
    unknownvalue: Option<f64>,
}

impl OrdinalEncoder {
    /// Creates a new OrdinalEncoder
    ///
    /// # Arguments
    /// * `handleunknown` - How to handle unknown categories ('error' or 'use_encoded_value')
    /// * `unknownvalue` - Value to use for unknown categories (when handleunknown='use_encoded_value')
    ///
    /// # Returns
    /// * A new OrdinalEncoder instance
    pub fn new(handleunknown: &str, unknownvalue: Option<f64>) -> Result<Self> {
        if handleunknown != "error" && handleunknown != "use_encoded_value" {
            return Err(TransformError::InvalidInput(
                "handleunknown must be 'error' or 'use_encoded_value'".to_string(),
            ));
        }

        if handleunknown == "use_encoded_value" && unknownvalue.is_none() {
            return Err(TransformError::InvalidInput(
                "unknownvalue must be specified when handleunknown='use_encoded_value'".to_string(),
            ));
        }

        Ok(OrdinalEncoder {
            categories_: None,
            handleunknown: handleunknown.to_string(),
            unknownvalue,
        })
    }

    /// Creates an OrdinalEncoder with default settings
    pub fn with_defaults() -> Self {
        Self::new("error", None).expect("Operation failed")
    }

    /// Fits the OrdinalEncoder to the input data
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = NumCast::from(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        let mut categories = Vec::with_capacity(n_features);

        for j in 0..n_features {
            // Collect unique values for this feature
            let mut unique_values: Vec<u64> = x_u64.column(j).to_vec();
            unique_values.sort_unstable();
            unique_values.dedup();

            categories.push(unique_values);
        }

        self.categories_ = Some(categories);
        Ok(())
    }

    /// Transforms the input data using the fitted OrdinalEncoder
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The ordinally encoded data
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = NumCast::from(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if self.categories_.is_none() {
            return Err(TransformError::TransformationError(
                "OrdinalEncoder has not been fitted".to_string(),
            ));
        }

        let categories = self.categories_.as_ref().expect("Operation failed");

        if n_features != categories.len() {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but OrdinalEncoder was fitted with {} features",
                n_features,
                categories.len()
            )));
        }

        let mut transformed = Array2::zeros((n_samples, n_features));

        // Create mappings from category values to ordinal values
        let mut category_mappings = Vec::new();
        for feature_categories in categories {
            let mut mapping = HashMap::new();
            for (ordinal, &category) in feature_categories.iter().enumerate() {
                mapping.insert(category, ordinal as f64);
            }
            category_mappings.push(mapping);
        }

        // Fill the transformed array
        for i in 0..n_samples {
            for j in 0..n_features {
                let value = x_u64[[i, j]];

                if let Some(&ordinal_value) = category_mappings[j].get(&value) {
                    transformed[[i, j]] = ordinal_value;
                } else if self.handleunknown == "error" {
                    return Err(TransformError::InvalidInput(format!(
                        "Found unknown category {value} in feature {j}"
                    )));
                } else {
                    // handleunknown == "use_encoded_value"
                    transformed[[i, j]] = self.unknownvalue.expect("Operation failed");
                }
            }
        }

        Ok(transformed)
    }

    /// Fits the OrdinalEncoder to the input data and transforms it
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The ordinally encoded data
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the categories for each feature
    ///
    /// # Returns
    /// * `Option<&Vec<Vec<u64>>>` - The categories for each feature
    pub fn categories(&self) -> Option<&Vec<Vec<u64>>> {
        self.categories_.as_ref()
    }
}

/// TargetEncoder for supervised categorical encoding
///
/// This encoder transforms categorical features using the target variable values,
/// encoding each category with a statistic (mean, median, etc.) of the target values
/// for that category. This is useful for high-cardinality categorical features.
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_transform::encoding::TargetEncoder;
///
/// let x = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]).expect("Operation failed");
/// let y = vec![1.0, 2.0, 3.0, 1.5, 2.5, 3.5];
///
/// let mut encoder = TargetEncoder::new("mean", 1.0, 0.0).expect("Operation failed");
/// let encoded = encoder.fit_transform(&x, &y).expect("Operation failed");
/// ```
#[derive(Debug, Clone)]
pub struct TargetEncoder {
    /// Encoding strategy ('mean', 'median', 'count', 'sum')
    strategy: String,
    /// Smoothing parameter for regularization (higher = more smoothing toward global mean)
    smoothing: f64,
    /// Global statistic to use for smoothing and unknown categories
    globalstat: f64,
    /// Mappings from categories to encoded values for each feature
    encodings_: Option<Vec<HashMap<u64, f64>>>,
    /// Whether the encoder has been fitted
    is_fitted: bool,
    /// Global mean of target values (computed during fit)
    global_mean_: f64,
}

impl TargetEncoder {
    /// Creates a new TargetEncoder
    ///
    /// # Arguments
    /// * `strategy` - Encoding strategy ('mean', 'median', 'count', 'sum')
    /// * `smoothing` - Smoothing parameter (0.0 = no smoothing, higher = more smoothing)
    /// * `globalstat` - Global statistic fallback for unknown categories
    ///
    /// # Returns
    /// * A new TargetEncoder instance
    pub fn new(_strategy: &str, smoothing: f64, globalstat: f64) -> Result<Self> {
        if !["mean", "median", "count", "sum"].contains(&_strategy) {
            return Err(TransformError::InvalidInput(
                "_strategy must be 'mean', 'median', 'count', or 'sum'".to_string(),
            ));
        }

        if smoothing < 0.0 {
            return Err(TransformError::InvalidInput(
                "smoothing parameter must be non-negative".to_string(),
            ));
        }

        Ok(TargetEncoder {
            strategy: _strategy.to_string(),
            smoothing,
            globalstat,
            encodings_: None,
            is_fitted: false,
            global_mean_: 0.0,
        })
    }

    /// Creates a TargetEncoder with mean strategy and default smoothing
    pub fn with_mean(smoothing: f64) -> Self {
        TargetEncoder {
            strategy: "mean".to_string(),
            smoothing,
            globalstat: 0.0,
            encodings_: None,
            is_fitted: false,
            global_mean_: 0.0,
        }
    }

    /// Creates a TargetEncoder with median strategy
    pub fn with_median(smoothing: f64) -> Self {
        TargetEncoder {
            strategy: "median".to_string(),
            smoothing,
            globalstat: 0.0,
            encodings_: None,
            is_fitted: false,
            global_mean_: 0.0,
        }
    }

    /// Fits the TargetEncoder to the input data and target values
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    /// * `y` - The target values, length n_samples
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>, y: &[f64]) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = NumCast::from(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        if y.len() != n_samples {
            return Err(TransformError::InvalidInput(
                "Number of target values must match number of samples".to_string(),
            ));
        }

        // Compute global mean for smoothing
        self.global_mean_ = y.iter().sum::<f64>() / y.len() as f64;

        let mut encodings = Vec::with_capacity(n_features);

        for j in 0..n_features {
            // Group target values by category for this feature
            let mut category_targets: HashMap<u64, Vec<f64>> = HashMap::new();

            for i in 0..n_samples {
                let category = x_u64[[i, j]];
                category_targets.entry(category).or_default().push(y[i]);
            }

            // Compute encoding for each category
            let mut category_encoding = HashMap::new();

            for (category, targets) in category_targets.iter() {
                let encoded_value = match self.strategy.as_str() {
                    "mean" => {
                        let category_mean = targets.iter().sum::<f64>() / targets.len() as f64;
                        let count = targets.len() as f64;

                        // Apply smoothing: (count * category_mean + smoothing * global_mean) / (count + smoothing)
                        if self.smoothing > 0.0 {
                            (count * category_mean + self.smoothing * self.global_mean_)
                                / (count + self.smoothing)
                        } else {
                            category_mean
                        }
                    }
                    "median" => {
                        let mut sorted_targets = targets.clone();
                        sorted_targets.sort_by(|a, b| a.partial_cmp(b).expect("Operation failed"));

                        let median = if sorted_targets.len() % 2 == 0 {
                            let mid = sorted_targets.len() / 2;
                            (sorted_targets[mid - 1] + sorted_targets[mid]) / 2.0
                        } else {
                            sorted_targets[sorted_targets.len() / 2]
                        };

                        // Apply smoothing toward global mean
                        if self.smoothing > 0.0 {
                            let count = targets.len() as f64;
                            (count * median + self.smoothing * self.global_mean_)
                                / (count + self.smoothing)
                        } else {
                            median
                        }
                    }
                    "count" => targets.len() as f64,
                    "sum" => targets.iter().sum::<f64>(),
                    _ => unreachable!(),
                };

                category_encoding.insert(*category, encoded_value);
            }

            encodings.push(category_encoding);
        }

        self.encodings_ = Some(encodings);
        self.is_fitted = true;
        Ok(())
    }

    /// Transforms the input data using the fitted TargetEncoder
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The target-encoded data
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if !self.is_fitted {
            return Err(TransformError::TransformationError(
                "TargetEncoder has not been fitted".to_string(),
            ));
        }

        let x_u64 = x.mapv(|x| {
            let val_f64 = NumCast::from(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        let encodings = self.encodings_.as_ref().expect("Operation failed");

        if n_features != encodings.len() {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but TargetEncoder was fitted with {} features",
                n_features,
                encodings.len()
            )));
        }

        let mut transformed = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                let category = x_u64[[i, j]];

                if let Some(&encoded_value) = encodings[j].get(&category) {
                    transformed[[i, j]] = encoded_value;
                } else {
                    // Use global statistic for unknown categories
                    transformed[[i, j]] = if self.globalstat != 0.0 {
                        self.globalstat
                    } else {
                        self.global_mean_
                    };
                }
            }
        }

        Ok(transformed)
    }

    /// Fits the TargetEncoder and transforms the data in one step
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    /// * `y` - The target values, length n_samples
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The target-encoded data
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>, y: &[f64]) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Returns the learned encodings for each feature
    ///
    /// # Returns
    /// * `Option<&Vec<HashMap<u64, f64>>>` - The category encodings for each feature
    pub fn encodings(&self) -> Option<&Vec<HashMap<u64, f64>>> {
        self.encodings_.as_ref()
    }

    /// Returns whether the encoder has been fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Returns the global mean computed during fitting
    pub fn global_mean(&self) -> f64 {
        self.global_mean_
    }

    /// Applies cross-validation target encoding to prevent overfitting
    ///
    /// This method performs k-fold cross-validation to compute target encodings,
    /// which helps prevent overfitting when the same data is used for both
    /// fitting and transforming.
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    /// * `y` - The target values, length n_samples
    /// * `cv_folds` - Number of cross-validation folds (default: 5)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The cross-validated target-encoded data
    pub fn fit_transform_cv<S>(
        &mut self,
        x: &ArrayBase<S, Ix2>,
        y: &[f64],
        cv_folds: usize,
    ) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = NumCast::from(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        if y.len() != n_samples {
            return Err(TransformError::InvalidInput(
                "Number of target values must match number of samples".to_string(),
            ));
        }

        if cv_folds < 2 {
            return Err(TransformError::InvalidInput(
                "cv_folds must be at least 2".to_string(),
            ));
        }

        let mut transformed = Array2::zeros((n_samples, n_features));

        // Compute global mean
        self.global_mean_ = y.iter().sum::<f64>() / y.len() as f64;

        // Create fold indices
        let fold_size = n_samples / cv_folds;
        let mut fold_indices = Vec::new();
        for fold in 0..cv_folds {
            let start = fold * fold_size;
            let end = if fold == cv_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };
            fold_indices.push((start, end));
        }

        // For each fold, train on other _folds and predict on this fold
        for fold in 0..cv_folds {
            let (val_start, val_end) = fold_indices[fold];

            // Collect training data (all _folds except current)
            let mut train_indices = Vec::new();
            for (other_fold, &(start, end)) in fold_indices.iter().enumerate().take(cv_folds) {
                if other_fold != fold {
                    train_indices.extend(start..end);
                }
            }

            // For each feature, compute encodings on training data
            for j in 0..n_features {
                let mut category_targets: HashMap<u64, Vec<f64>> = HashMap::new();

                // Collect target values by category for training data
                for &train_idx in &train_indices {
                    let category = x_u64[[train_idx, j]];
                    category_targets
                        .entry(category)
                        .or_default()
                        .push(y[train_idx]);
                }

                // Compute encodings for this fold
                let mut category_encoding = HashMap::new();
                for (category, targets) in category_targets.iter() {
                    let encoded_value = match self.strategy.as_str() {
                        "mean" => {
                            let category_mean = targets.iter().sum::<f64>() / targets.len() as f64;
                            let count = targets.len() as f64;

                            if self.smoothing > 0.0 {
                                (count * category_mean + self.smoothing * self.global_mean_)
                                    / (count + self.smoothing)
                            } else {
                                category_mean
                            }
                        }
                        "median" => {
                            let mut sorted_targets = targets.clone();
                            sorted_targets
                                .sort_by(|a, b| a.partial_cmp(b).expect("Operation failed"));

                            let median = if sorted_targets.len() % 2 == 0 {
                                let mid = sorted_targets.len() / 2;
                                (sorted_targets[mid - 1] + sorted_targets[mid]) / 2.0
                            } else {
                                sorted_targets[sorted_targets.len() / 2]
                            };

                            if self.smoothing > 0.0 {
                                let count = targets.len() as f64;
                                (count * median + self.smoothing * self.global_mean_)
                                    / (count + self.smoothing)
                            } else {
                                median
                            }
                        }
                        "count" => targets.len() as f64,
                        "sum" => targets.iter().sum::<f64>(),
                        _ => unreachable!(),
                    };

                    category_encoding.insert(*category, encoded_value);
                }

                // Apply encodings to validation fold
                for val_idx in val_start..val_end {
                    let category = x_u64[[val_idx, j]];

                    if let Some(&encoded_value) = category_encoding.get(&category) {
                        transformed[[val_idx, j]] = encoded_value;
                    } else {
                        // Use global mean for unknown categories
                        transformed[[val_idx, j]] = self.global_mean_;
                    }
                }
            }
        }

        // Now fit on the full data for future transforms
        self.fit(x, y)?;

        Ok(transformed)
    }
}

/// BinaryEncoder for converting categorical features to binary representations
///
/// This transformer converts categorical features into binary representations,
/// where each category is encoded as a unique binary number. This is more
/// memory-efficient than one-hot encoding for high-cardinality categorical features.
///
/// For n unique categories, ceil(log2(n)) binary features are created.
#[derive(Debug, Clone)]
pub struct BinaryEncoder {
    /// Mappings from categories to binary codes for each feature
    categories_: Option<Vec<HashMap<u64, Vec<u8>>>>,
    /// Number of binary features per original feature
    n_binary_features_: Option<Vec<usize>>,
    /// Whether to handle unknown categories
    handleunknown: String,
    /// Whether the encoder has been fitted
    is_fitted: bool,
}

impl BinaryEncoder {
    /// Creates a new BinaryEncoder
    ///
    /// # Arguments
    /// * `handleunknown` - How to handle unknown categories ('error' or 'ignore')
    ///   - 'error': Raise an error if unknown categories are encountered
    ///   - 'ignore': Encode unknown categories as all zeros
    ///
    /// # Returns
    /// * `Result<BinaryEncoder>` - The new encoder instance
    pub fn new(handleunknown: &str) -> Result<Self> {
        if handleunknown != "error" && handleunknown != "ignore" {
            return Err(TransformError::InvalidInput(
                "handleunknown must be 'error' or 'ignore'".to_string(),
            ));
        }

        Ok(BinaryEncoder {
            categories_: None,
            n_binary_features_: None,
            handleunknown: handleunknown.to_string(),
            is_fitted: false,
        })
    }

    /// Creates a BinaryEncoder with default settings (handleunknown='error')
    pub fn with_defaults() -> Self {
        Self::new("error").expect("Operation failed")
    }

    /// Fits the BinaryEncoder to the input data
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = NumCast::from(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        let mut categories = Vec::with_capacity(n_features);
        let mut n_binary_features = Vec::with_capacity(n_features);

        for j in 0..n_features {
            // Collect unique categories for this feature
            let mut unique_categories: Vec<u64> = x_u64.column(j).to_vec();
            unique_categories.sort_unstable();
            unique_categories.dedup();

            if unique_categories.is_empty() {
                return Err(TransformError::InvalidInput(
                    "Feature has no valid categories".to_string(),
                ));
            }

            // Calculate number of binary features needed
            let n_cats = unique_categories.len();
            let nbits = if n_cats <= 1 {
                1
            } else {
                (n_cats as f64).log2().ceil() as usize
            };

            // Create binary mappings
            let mut category_map = HashMap::new();
            for (idx, &category) in unique_categories.iter().enumerate() {
                let binary_code = Self::int_to_binary(idx, nbits);
                category_map.insert(category, binary_code);
            }

            categories.push(category_map);
            n_binary_features.push(nbits);
        }

        self.categories_ = Some(categories);
        self.n_binary_features_ = Some(n_binary_features);
        self.is_fitted = true;

        Ok(())
    }

    /// Transforms the input data using the fitted encoder
    ///
    /// # Arguments
    /// * `x` - The input categorical data to transform
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The binary-encoded data
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if !self.is_fitted {
            return Err(TransformError::InvalidInput(
                "Encoder has not been fitted yet".to_string(),
            ));
        }

        let categories = self.categories_.as_ref().expect("Operation failed");
        let n_binary_features = self.n_binary_features_.as_ref().expect("Operation failed");

        let x_u64 = x.mapv(|x| {
            let val_f64 = NumCast::from(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if n_features != categories.len() {
            return Err(TransformError::InvalidInput(format!(
                "Number of features ({}) does not match fitted features ({})",
                n_features,
                categories.len()
            )));
        }

        // Calculate total number of output features
        let total_binary_features: usize = n_binary_features.iter().sum();
        let mut result = Array2::<f64>::zeros((n_samples, total_binary_features));

        let mut output_col = 0;
        for j in 0..n_features {
            let category_map = &categories[j];
            let nbits = n_binary_features[j];

            for i in 0..n_samples {
                let category = x_u64[[i, j]];

                if let Some(binary_code) = category_map.get(&category) {
                    // Known category: use binary code
                    for (bit_idx, &bit_val) in binary_code.iter().enumerate() {
                        result[[i, output_col + bit_idx]] = bit_val as f64;
                    }
                } else {
                    // Unknown category
                    match self.handleunknown.as_str() {
                        "error" => {
                            return Err(TransformError::InvalidInput(format!(
                                "Unknown category {category} in feature {j}"
                            )));
                        }
                        "ignore" => {
                            // Set all bits to zero (already initialized)
                        }
                        _ => unreachable!(),
                    }
                }
            }

            output_col += nbits;
        }

        Ok(result)
    }

    /// Fits the encoder and transforms the data in one step
    ///
    /// # Arguments
    /// * `x` - The input categorical data
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The binary-encoded data
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns whether the encoder has been fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Returns the category mappings if fitted
    pub fn categories(&self) -> Option<&Vec<HashMap<u64, Vec<u8>>>> {
        self.categories_.as_ref()
    }

    /// Returns the number of binary features per original feature
    pub fn n_binary_features(&self) -> Option<&Vec<usize>> {
        self.n_binary_features_.as_ref()
    }

    /// Returns the total number of output features
    pub fn n_output_features(&self) -> Option<usize> {
        self.n_binary_features_.as_ref().map(|v| v.iter().sum())
    }

    /// Converts an integer to binary representation
    fn int_to_binary(_value: usize, nbits: usize) -> Vec<u8> {
        let mut binary = Vec::with_capacity(nbits);
        let mut val = _value;

        for _ in 0..nbits {
            binary.push((val & 1) as u8);
            val >>= 1;
        }

        binary.reverse(); // Most significant bit first
        binary
    }
}

/// FrequencyEncoder for converting categorical features to frequency counts
///
/// This transformer converts categorical features into their frequency of occurrence
/// in the training data. High-frequency categories get higher values, which can be
/// useful for models that can leverage frequency information.
#[derive(Debug, Clone)]
pub struct FrequencyEncoder {
    /// Frequency mappings for each feature
    frequency_maps_: Option<Vec<HashMap<u64, f64>>>,
    /// Whether to normalize frequencies to [0, 1]
    normalize: bool,
    /// How to handle unknown categories
    handleunknown: String,
    /// Value to use for unknown categories (when handleunknown="use_encoded_value")
    unknownvalue: f64,
    /// Whether the encoder has been fitted
    is_fitted: bool,
}

impl FrequencyEncoder {
    /// Creates a new FrequencyEncoder
    ///
    /// # Arguments
    /// * `normalize` - Whether to normalize frequencies to [0, 1] range
    /// * `handleunknown` - How to handle unknown categories ('error', 'ignore', or 'use_encoded_value')
    /// * `unknownvalue` - Value to use for unknown categories (when handleunknown="use_encoded_value")
    ///
    /// # Returns
    /// * `Result<FrequencyEncoder>` - The new encoder instance
    pub fn new(normalize: bool, handleunknown: &str, unknownvalue: f64) -> Result<Self> {
        if !["error", "ignore", "use_encoded_value"].contains(&handleunknown) {
            return Err(TransformError::InvalidInput(
                "handleunknown must be 'error', 'ignore', or 'use_encoded_value'".to_string(),
            ));
        }

        Ok(FrequencyEncoder {
            frequency_maps_: None,
            normalize,
            handleunknown: handleunknown.to_string(),
            unknownvalue,
            is_fitted: false,
        })
    }

    /// Creates a FrequencyEncoder with default settings
    pub fn with_defaults() -> Self {
        Self::new(false, "error", 0.0).expect("Operation failed")
    }

    /// Creates a FrequencyEncoder with normalized frequencies
    pub fn with_normalization() -> Self {
        Self::new(true, "error", 0.0).expect("Operation failed")
    }

    /// Fits the FrequencyEncoder to the input data
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = NumCast::from(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        let mut frequency_maps = Vec::with_capacity(n_features);

        for j in 0..n_features {
            // Count frequency of each category
            let mut category_counts: HashMap<u64, usize> = HashMap::new();
            for i in 0..n_samples {
                let category = x_u64[[i, j]];
                *category_counts.entry(category).or_insert(0) += 1;
            }

            // Convert counts to frequencies
            let mut frequency_map = HashMap::new();
            for (category, count) in category_counts {
                let frequency = if self.normalize {
                    count as f64 / n_samples as f64
                } else {
                    count as f64
                };
                frequency_map.insert(category, frequency);
            }

            frequency_maps.push(frequency_map);
        }

        self.frequency_maps_ = Some(frequency_maps);
        self.is_fitted = true;
        Ok(())
    }

    /// Transforms the input data using the fitted FrequencyEncoder
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The frequency-encoded data
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if !self.is_fitted {
            return Err(TransformError::TransformationError(
                "FrequencyEncoder has not been fitted".to_string(),
            ));
        }

        let frequency_maps = self.frequency_maps_.as_ref().expect("Operation failed");

        let x_u64 = x.mapv(|x| {
            let val_f64 = NumCast::from(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if n_features != frequency_maps.len() {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but FrequencyEncoder was fitted with {} features",
                n_features,
                frequency_maps.len()
            )));
        }

        let mut transformed = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                let category = x_u64[[i, j]];

                if let Some(&frequency) = frequency_maps[j].get(&category) {
                    transformed[[i, j]] = frequency;
                } else {
                    // Handle unknown category
                    match self.handleunknown.as_str() {
                        "error" => {
                            return Err(TransformError::InvalidInput(format!(
                                "Unknown category {category} in feature {j}"
                            )));
                        }
                        "ignore" => {
                            transformed[[i, j]] = 0.0;
                        }
                        "use_encoded_value" => {
                            transformed[[i, j]] = self.unknownvalue;
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }

        Ok(transformed)
    }

    /// Fits the encoder and transforms the data in one step
    ///
    /// # Arguments
    /// * `x` - The input categorical data
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The frequency-encoded data
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns whether the encoder has been fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Returns the learned frequency mappings if fitted
    pub fn frequency_maps(&self) -> Option<&Vec<HashMap<u64, f64>>> {
        self.frequency_maps_.as_ref()
    }
}

/// Weight of Evidence (WOE) Encoder for converting categorical features using target information
///
/// WOE encoding transforms categorical features based on the relationship between
/// each category and a binary target variable. It's particularly useful for credit
/// scoring and other binary classification tasks.
///
/// WOE = ln(P(target=1|category) / P(target=0|category))
#[derive(Debug, Clone)]
pub struct WOEEncoder {
    /// WOE mappings for each feature
    woe_maps_: Option<Vec<HashMap<u64, f64>>>,
    /// Information Value (IV) for each feature
    information_values_: Option<Vec<f64>>,
    /// Regularization parameter to handle categories with zero events/non-events
    regularization: f64,
    /// How to handle unknown categories
    handleunknown: String,
    /// Value to use for unknown categories (when handleunknown="use_encoded_value")
    unknownvalue: f64,
    /// Global WOE value for unknown categories (computed as overall log-odds)
    global_woe_: f64,
    /// Whether the encoder has been fitted
    is_fitted: bool,
}

impl WOEEncoder {
    /// Creates a new WOEEncoder
    ///
    /// # Arguments
    /// * `regularization` - Small value added to prevent division by zero (default: 0.5)
    /// * `handleunknown` - How to handle unknown categories ('error', 'global_woe', or 'use_encoded_value')
    /// * `unknownvalue` - Value to use for unknown categories (when handleunknown="use_encoded_value")
    ///
    /// # Returns
    /// * `Result<WOEEncoder>` - The new encoder instance
    pub fn new(regularization: f64, handleunknown: &str, unknownvalue: f64) -> Result<Self> {
        if regularization < 0.0 {
            return Err(TransformError::InvalidInput(
                "regularization must be non-negative".to_string(),
            ));
        }

        if !["error", "global_woe", "use_encoded_value"].contains(&handleunknown) {
            return Err(TransformError::InvalidInput(
                "handleunknown must be 'error', 'global_woe', or 'use_encoded_value'".to_string(),
            ));
        }

        Ok(WOEEncoder {
            woe_maps_: None,
            information_values_: None,
            regularization,
            handleunknown: handleunknown.to_string(),
            unknownvalue,
            global_woe_: 0.0,
            is_fitted: false,
        })
    }

    /// Creates a WOEEncoder with default settings
    pub fn with_defaults() -> Self {
        Self::new(0.5, "global_woe", 0.0).expect("Operation failed")
    }

    /// Creates a WOEEncoder with custom regularization
    pub fn with_regularization(regularization: f64) -> Result<Self> {
        Self::new(regularization, "global_woe", 0.0)
    }

    /// Fits the WOEEncoder to the input data
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    /// * `y` - The binary target values (0 or 1), length n_samples
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>, y: &[f64]) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = NumCast::from(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        if y.len() != n_samples {
            return Err(TransformError::InvalidInput(
                "Number of target values must match number of samples".to_string(),
            ));
        }

        // Validate that target is binary
        for &target in y {
            if target != 0.0 && target != 1.0 {
                return Err(TransformError::InvalidInput(
                    "Target values must be binary (0 or 1)".to_string(),
                ));
            }
        }

        // Calculate global statistics
        let total_events: f64 = y.iter().sum();
        let total_non_events = n_samples as f64 - total_events;

        if total_events == 0.0 || total_non_events == 0.0 {
            return Err(TransformError::InvalidInput(
                "Target must contain both 0 and 1 values".to_string(),
            ));
        }

        // Global WOE (overall log-odds)
        self.global_woe_ = (total_events / total_non_events).ln();

        let mut woe_maps = Vec::with_capacity(n_features);
        let mut information_values = Vec::with_capacity(n_features);

        for j in 0..n_features {
            // Collect target values by category
            let mut category_stats: HashMap<u64, (f64, f64)> = HashMap::new(); // (events, non_events)

            for i in 0..n_samples {
                let category = x_u64[[i, j]];
                let target = y[i];

                let (events, non_events) = category_stats.entry(category).or_insert((0.0, 0.0));
                if target == 1.0 {
                    *events += 1.0;
                } else {
                    *non_events += 1.0;
                }
            }

            // Calculate WOE and IV for each category
            let mut woe_map = HashMap::new();
            let mut feature_iv = 0.0;

            for (category, (events, non_events)) in category_stats.iter() {
                // Add regularization to handle zero counts
                let reg_events = events + self.regularization;
                let reg_non_events = non_events + self.regularization;
                let reg_total_events =
                    total_events + self.regularization * category_stats.len() as f64;
                let reg_total_non_events =
                    total_non_events + self.regularization * category_stats.len() as f64;

                // Calculate distribution percentages
                let event_rate = reg_events / reg_total_events;
                let non_event_rate = reg_non_events / reg_total_non_events;

                // Calculate WOE
                let woe = (event_rate / non_event_rate).ln();
                woe_map.insert(*category, woe);

                // Calculate Information Value contribution
                let iv_contribution = (event_rate - non_event_rate) * woe;
                feature_iv += iv_contribution;
            }

            woe_maps.push(woe_map);
            information_values.push(feature_iv);
        }

        self.woe_maps_ = Some(woe_maps);
        self.information_values_ = Some(information_values);
        self.is_fitted = true;
        Ok(())
    }

    /// Transforms the input data using the fitted WOEEncoder
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The WOE-encoded data
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if !self.is_fitted {
            return Err(TransformError::TransformationError(
                "WOEEncoder has not been fitted".to_string(),
            ));
        }

        let woe_maps = self.woe_maps_.as_ref().expect("Operation failed");

        let x_u64 = x.mapv(|x| {
            let val_f64 = NumCast::from(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if n_features != woe_maps.len() {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but WOEEncoder was fitted with {} features",
                n_features,
                woe_maps.len()
            )));
        }

        let mut transformed = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                let category = x_u64[[i, j]];

                if let Some(&woe_value) = woe_maps[j].get(&category) {
                    transformed[[i, j]] = woe_value;
                } else {
                    // Handle unknown category
                    match self.handleunknown.as_str() {
                        "error" => {
                            return Err(TransformError::InvalidInput(format!(
                                "Unknown category {category} in feature {j}"
                            )));
                        }
                        "global_woe" => {
                            transformed[[i, j]] = self.global_woe_;
                        }
                        "use_encoded_value" => {
                            transformed[[i, j]] = self.unknownvalue;
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }

        Ok(transformed)
    }

    /// Fits the encoder and transforms the data in one step
    ///
    /// # Arguments
    /// * `x` - The input categorical data
    /// * `y` - The binary target values
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The WOE-encoded data
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>, y: &[f64]) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Returns whether the encoder has been fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Returns the learned WOE mappings if fitted
    pub fn woe_maps(&self) -> Option<&Vec<HashMap<u64, f64>>> {
        self.woe_maps_.as_ref()
    }

    /// Returns the Information Values for each feature if fitted
    ///
    /// Information Value interpretation:
    /// - < 0.02: Not useful for prediction
    /// - 0.02 - 0.1: Weak predictive power
    /// - 0.1 - 0.3: Medium predictive power  
    /// - 0.3 - 0.5: Strong predictive power
    /// - > 0.5: Suspicious, too good to be true
    pub fn information_values(&self) -> Option<&Vec<f64>> {
        self.information_values_.as_ref()
    }

    /// Returns the global WOE value (overall log-odds)
    pub fn global_woe(&self) -> f64 {
        self.global_woe_
    }

    /// Returns features ranked by Information Value (descending order)
    ///
    /// # Returns
    /// * `Option<Vec<(usize, f64)>>` - Vector of (feature_index, information_value) pairs
    pub fn feature_importance_ranking(&self) -> Option<Vec<(usize, f64)>> {
        self.information_values_.as_ref().map(|ivs| {
            let mut ranking: Vec<(usize, f64)> =
                ivs.iter().enumerate().map(|(idx, &iv)| (idx, iv)).collect();
            ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            ranking
        })
    }
}

#[cfg(test)]
#[path = "encoding_tests.rs"]
mod tests;
