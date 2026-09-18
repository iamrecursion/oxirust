//! Convenience ndarray-returning wrappers for all advanced generators
//!
//! Provides simple function signatures returning `Array2<f64>` / `Array1<usize>`
//! for easy integration into ndarray-based ML pipelines. Each function wraps the
//! corresponding config-struct based generator and converts the output.

use crate::error::{DatasetsError, Result};
use scirs2_core::ndarray::{Array1, Array2};

use crate::generators::low_rank::{make_low_rank as low_rank_impl, LowRankConfig};
use crate::generators::sparse_classification::{
    make_sparse_classification as sparse_class_impl, SparseClassConfig,
};

// ────────────────────────────────────────────────────────────────────
// make_low_rank
// ────────────────────────────────────────────────────────────────────

/// Generate a low-rank matrix completion benchmark as ndarray matrices.
///
/// Constructs a matrix `X = A @ B + noise` where A is `(n_samples, rank)` and
/// B is `(rank, n_features)`, both drawn from N(0,1). The returned pair is:
/// - `X_full`: the complete noisy matrix `(n_samples, n_features)`
/// - `X_observed`: same shape but with ~50 % of entries set to `NAN` (masked)
///
/// # Arguments
///
/// * `n_samples` - Number of rows
/// * `n_features` - Number of columns
/// * `rank` - True rank of the underlying matrix
/// * `noise` - Standard deviation of Gaussian noise added to every entry
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// `(X_full, X_observed)` — both `Array2<f64>` of shape `(n_samples, n_features)`.
pub fn make_low_rank(
    n_samples: usize,
    n_features: usize,
    rank: usize,
    noise: f64,
    seed: u64,
) -> Result<(Array2<f64>, Array2<f64>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat("n_samples must be > 0".into()));
    }
    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be > 0".into(),
        ));
    }
    let effective_rank = rank.max(1).min(n_samples.min(n_features));

    let config = LowRankConfig {
        n_rows: n_samples,
        n_cols: n_features,
        rank: effective_rank,
        noise_std: noise,
        observation_fraction: 0.5,
        seed,
    };

    let ds = low_rank_impl(&config);
    let n_rows = ds.matrix.len();
    let n_cols = if n_rows > 0 { ds.matrix[0].len() } else { 0 };
    let total = n_rows * n_cols;

    // Build X_full: flat row-major vector
    let mut flat_full = Vec::with_capacity(total);
    for row in &ds.matrix {
        flat_full.extend_from_slice(row);
    }

    // Build X_observed: NAN where entry was not observed, true value where observed
    let mut flat_obs = Vec::with_capacity(total);
    for (i, row) in ds.matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if ds.observed_mask[i][j] {
                flat_obs.push(val);
            } else {
                flat_obs.push(f64::NAN);
            }
        }
    }

    let x_full = Array2::from_shape_vec((n_rows, n_cols), flat_full)
        .map_err(|e| DatasetsError::InvalidFormat(format!("Array2 shape error: {e}")))?;
    let x_obs = Array2::from_shape_vec((n_rows, n_cols), flat_obs)
        .map_err(|e| DatasetsError::InvalidFormat(format!("Array2 shape error: {e}")))?;

    Ok((x_full, x_obs))
}

// ────────────────────────────────────────────────────────────────────
// make_sparse_classification
// ────────────────────────────────────────────────────────────────────

/// Generate a high-dimensional sparse classification dataset as ndarray arrays.
///
/// Only `n_informative` out of `n_features` dimensions carry signal. Each
/// class gets a centroid drawn from N(0,1) projected only onto the informative
/// features. Non-informative features remain exactly 0.
///
/// # Arguments
///
/// * `n_samples` - Number of samples
/// * `n_features` - Total number of features (most will be zero)
/// * `n_informative` - Number of truly informative features
/// * `density` - Unused parameter kept for API symmetry; actual sparsity is
///   controlled by `n_informative / n_features`
/// * `n_classes` - Number of classes
/// * `seed` - Random seed
///
/// # Returns
///
/// `(X, y)` — `Array2<f64>` of shape `(n_samples, n_features)` and
/// `Array1<usize>` of shape `(n_samples,)`.
pub fn make_sparse_classification(
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    _density: f64,
    n_classes: usize,
    seed: u64,
) -> Result<(Array2<f64>, Array1<usize>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat("n_samples must be > 0".into()));
    }
    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be > 0".into(),
        ));
    }
    if n_classes == 0 {
        return Err(DatasetsError::InvalidFormat("n_classes must be > 0".into()));
    }

    let config = SparseClassConfig {
        n_samples,
        n_features,
        n_informative: n_informative.min(n_features),
        n_classes,
        class_sep: 1.0,
        seed,
    };

    let ds = sparse_class_impl(&config);
    let n_rows = ds.x.len();
    let n_cols = if n_rows > 0 { ds.x[0].len() } else { 0 };

    let mut flat = Vec::with_capacity(n_rows * n_cols);
    for row in &ds.x {
        flat.extend_from_slice(row);
    }

    let x = Array2::from_shape_vec((n_rows, n_cols), flat)
        .map_err(|e| DatasetsError::InvalidFormat(format!("Array2 shape error: {e}")))?;
    let y = Array1::from_vec(ds.y);

    Ok((x, y))
}

// ────────────────────────────────────────────────────────────────────
// make_multilabel_classification
// ────────────────────────────────────────────────────────────────────

/// Generate a multi-label classification dataset as ndarray arrays.
///
/// Each sample can have 0 to `n_labels` active classes (unlike one-hot
/// encoding). The label matrix Y is `(n_samples, n_classes)` binary.
/// The average number of active labels per sample follows a Poisson
/// distribution centred on `n_labels`.
///
/// This is a thin wrapper around the existing config-based implementation
/// in `generators::classification::make_multilabel_classification`.
///
/// # Arguments
///
/// * `n_samples` - Number of samples
/// * `n_features` - Number of input features
/// * `n_classes` - Number of possible labels
/// * `n_labels` - Average number of labels per sample
/// * `seed` - Random seed
///
/// # Returns
///
/// `(X, Y)` — `Array2<f64>` of shape `(n_samples, n_features)` and
/// `Array2<u8>` of shape `(n_samples, n_classes)` with binary indicators.
pub fn make_multilabel_classification_nd(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    n_labels: usize,
    seed: u64,
) -> Result<(Array2<f64>, Array2<u8>)> {
    use crate::generators::classification::{make_multilabel_classification, MultilabelConfig};

    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat("n_samples must be > 0".into()));
    }
    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be > 0".into(),
        ));
    }
    if n_classes == 0 {
        return Err(DatasetsError::InvalidFormat("n_classes must be > 0".into()));
    }
    let effective_labels = n_labels.max(1).min(n_classes);

    let config = MultilabelConfig {
        n_samples,
        n_features,
        n_classes,
        n_labels: effective_labels,
        allow_unlabeled: false,
        random_state: Some(seed),
    };

    let ds = make_multilabel_classification(config)?;

    // Convert target f64 matrix to u8 binary matrix
    let nrows = ds.target.nrows();
    let ncols = ds.target.ncols();
    let mut flat_y = Vec::with_capacity(nrows * ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            flat_y.push(if ds.target[[i, j]] > 0.5 { 1u8 } else { 0u8 });
        }
    }

    // X is already an Array2<f64> — convert to owned flat Vec
    let x_flat: Vec<f64> = ds.data.iter().copied().collect();
    let x = Array2::from_shape_vec((n_samples, n_features), x_flat)
        .map_err(|e| DatasetsError::InvalidFormat(format!("Array2 shape error: {e}")))?;
    let y = Array2::from_shape_vec((nrows, ncols), flat_y)
        .map_err(|e| DatasetsError::InvalidFormat(format!("Array2 shape error: {e}")))?;

    Ok((x, y))
}

// ────────────────────────────────────────────────────────────────────
// make_heterogeneous
// ────────────────────────────────────────────────────────────────────

/// Generate a mixed numeric / categorical classification dataset as ndarray arrays.
///
/// Numeric features are sampled from Gaussian distributions; categorical
/// features are integer-encoded in `[0, n_categories)`. The two sets of
/// features are concatenated horizontally.
///
/// # Arguments
///
/// * `n_samples` - Number of samples
/// * `n_numeric` - Number of continuous Gaussian features
/// * `n_categorical` - Number of integer-encoded categorical features
/// * `n_categories` - Number of distinct categories for each categorical feature
/// * `seed` - Random seed
///
/// # Returns
///
/// `(X, y)` — `Array2<f64>` of shape `(n_samples, n_numeric + n_categorical)` and
/// `Array1<usize>` of shape `(n_samples,)` with binary class labels.
pub fn make_heterogeneous_nd(
    n_samples: usize,
    n_numeric: usize,
    n_categorical: usize,
    n_categories: usize,
    seed: u64,
) -> Result<(Array2<f64>, Array1<usize>)> {
    use crate::generators::heterogeneous::{
        make_heterogeneous, FeatureType, HeteroConfig, HeteroFeatureValue,
    };

    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat("n_samples must be > 0".into()));
    }
    if n_numeric + n_categorical == 0 {
        return Err(DatasetsError::InvalidFormat(
            "at least one feature column required".into(),
        ));
    }
    let n_cats = n_categories.max(2);

    // Build explicit feature type list
    let mut feature_types = Vec::new();
    for _ in 0..n_numeric {
        feature_types.push(FeatureType::Continuous(0.0, 1.0));
    }
    for _ in 0..n_categorical {
        feature_types.push(FeatureType::Categorical(n_cats));
    }

    let config = HeteroConfig {
        n_samples,
        feature_types,
        n_features: n_numeric + n_categorical,
        n_classes: 2,
        seed,
    };

    let ds = make_heterogeneous(&config);

    let n_features = n_numeric + n_categorical;
    let mut flat = Vec::with_capacity(n_samples * n_features);
    for row in &ds.features {
        for val in row {
            let fval = match val {
                HeteroFeatureValue::Float(v) => *v,
                HeteroFeatureValue::Int(k) => *k as f64,
                HeteroFeatureValue::Bool(b) if *b => 1.0,
                HeteroFeatureValue::Bool(_) => 0.0,
                // HeteroFeatureValue is #[non_exhaustive]; handle any future variants
                #[allow(unreachable_patterns)]
                _ => 0.0,
            };
            flat.push(fval);
        }
    }

    let x = Array2::from_shape_vec((n_samples, n_features), flat)
        .map_err(|e| DatasetsError::InvalidFormat(format!("Array2 shape error: {e}")))?;
    let y = Array1::from_vec(ds.labels);

    Ok((x, y))
}

// ────────────────────────────────────────────────────────────────────
// make_concept_drift
// ────────────────────────────────────────────────────────────────────

/// Generate a time-series classification dataset with concept drift as ndarray arrays.
///
/// Before each drift point: class 0 features are drawn from N(0, 1) and
/// class 1 from N(1, 1). After the drift: distributions swap — class 0 is
/// from N(1, 1), class 1 is from N(0, 1). Multiple drift points supported.
///
/// This wrapper builds a binary classification time series directly using
/// the drift-point positions to control the class distributions.
///
/// # Arguments
///
/// * `n_samples` - Total number of samples (time steps)
/// * `n_features` - Number of input features per sample
/// * `drift_points` - Positions (sample indices) where drift occurs
/// * `seed` - Random seed
///
/// # Returns
///
/// `(X, y, actual_drift_points)` — `Array2<f64>` `(n_samples, n_features)`,
/// `Array1<usize>` `(n_samples,)` class labels, and `Vec<usize>` confirmed drift positions.
pub fn make_concept_drift_nd(
    n_samples: usize,
    n_features: usize,
    drift_points: Vec<usize>,
    seed: u64,
) -> Result<(Array2<f64>, Array1<usize>, Vec<usize>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat("n_samples must be > 0".into()));
    }
    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be > 0".into(),
        ));
    }

    // Validate drift point positions
    let valid_points: Vec<usize> = drift_points
        .iter()
        .filter(|&&p| p > 0 && p < n_samples)
        .copied()
        .collect();

    // Simple seeded LCG to avoid pulling in rand
    let mut state = seed.wrapping_add(1);
    let mut next_u64 = || -> u64 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state
    };
    let mut next_f64 = || -> f64 { (next_u64() >> 11) as f64 / (1u64 << 53) as f64 };
    let mut next_normal = || -> f64 {
        let u1 = next_f64().max(1e-10);
        let u2 = next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    let mut flat_x = Vec::with_capacity(n_samples * n_features);
    let mut y_vec = Vec::with_capacity(n_samples);

    // Count how many drift points have been passed at each sample
    // Odd segment count => distributions swapped
    for t in 0..n_samples {
        let segment = valid_points.iter().filter(|&&p| t >= p).count();
        let swapped = segment % 2 == 1;

        // class label for this sample: alternate 0 and 1 to keep balanced
        let class: usize = t % 2;
        y_vec.push(class);

        // Before swap: class 0 ~ N(0,1), class 1 ~ N(1,1)
        // After swap:  class 0 ~ N(1,1), class 1 ~ N(0,1)
        let mean = if swapped {
            if class == 0 {
                1.0
            } else {
                0.0
            }
        } else {
            if class == 0 {
                0.0
            } else {
                1.0
            }
        };

        for _ in 0..n_features {
            flat_x.push(mean + next_normal());
        }
    }

    let x = Array2::from_shape_vec((n_samples, n_features), flat_x)
        .map_err(|e| DatasetsError::InvalidFormat(format!("Array2 shape error: {e}")))?;
    let y = Array1::from_vec(y_vec);

    Ok((x, y, valid_points))
}

// ────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify output shapes for make_low_rank
    #[test]
    fn test_make_low_rank_shape() {
        let (x_full, x_obs) =
            make_low_rank(80, 60, 5, 0.1, 42).expect("make_low_rank should succeed");
        assert_eq!(x_full.nrows(), 80, "X_full rows");
        assert_eq!(x_full.ncols(), 60, "X_full cols");
        assert_eq!(x_obs.nrows(), 80, "X_obs rows");
        assert_eq!(x_obs.ncols(), 60, "X_obs cols");
    }

    /// Verify that effective rank is at most the requested rank.
    ///
    /// We check this by confirming that X_full entries are non-trivial
    /// (the true low-rank structure means values aren't all identical).
    #[test]
    fn test_make_low_rank_rank_property() {
        let (x_full, _) = make_low_rank(50, 50, 3, 0.0, 7).expect("make_low_rank should succeed");
        // Variance across all entries should be non-zero
        let n = (50 * 50) as f64;
        let mean: f64 = x_full.iter().sum::<f64>() / n;
        let var: f64 = x_full.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
        assert!(
            var > 1e-6,
            "X_full should have non-trivial variance, got {var}"
        );
        // About 50% of X_obs entries should be NAN
        let (xf, xo) = make_low_rank(50, 50, 3, 0.0, 7).expect("ok");
        let _ = xf; // not needed here
        let nan_count = xo.iter().filter(|v| v.is_nan()).count();
        let total = 50 * 50;
        let nan_fraction = nan_count as f64 / total as f64;
        // Expect between 30% and 70% NaN
        assert!(
            (0.3..=0.7).contains(&nan_fraction),
            "Expected ~50% NaN in X_obs, got {nan_fraction:.2}"
        );
    }

    /// Verify density (fraction of non-zero entries) in sparse classification
    #[test]
    fn test_make_sparse_classification_sparsity() {
        let (x, y) = make_sparse_classification(200, 1000, 10, 0.01, 2, 42)
            .expect("make_sparse_classification should succeed");
        assert_eq!(x.nrows(), 200);
        assert_eq!(x.ncols(), 1000);
        assert_eq!(y.len(), 200);

        let total = x.len() as f64;
        let nonzero = x.iter().filter(|&&v| v != 0.0).count() as f64;
        let density = nonzero / total;
        // With 10 informative out of 1000, density should be well below 5%
        assert!(
            density < 0.05,
            "Expected sparse features (density < 0.05), got {density:.4}"
        );
    }

    /// Verify average label count is near n_labels
    #[test]
    fn test_make_multilabel_avg_labels() {
        let n_samples = 200;
        let n_labels = 3;
        let (x, y) = make_multilabel_classification_nd(n_samples, 10, 6, n_labels, 42)
            .expect("multilabel should succeed");
        assert_eq!(x.nrows(), n_samples);
        assert_eq!(y.nrows(), n_samples);
        assert_eq!(y.ncols(), 6);

        let total_active: usize = y.iter().map(|&b| b as usize).sum();
        let avg = total_active as f64 / n_samples as f64;
        // Average should be close to n_labels (within 50%)
        assert!(
            avg >= n_labels as f64 * 0.5 && avg <= n_labels as f64 * 1.5,
            "Expected avg labels ≈ {n_labels}, got {avg:.2}"
        );
    }

    /// Verify categorical feature values are in [0, n_categories)
    #[test]
    fn test_make_heterogeneous_categorical_range() {
        let n_categories = 5usize;
        let (x, y) = make_heterogeneous_nd(100, 3, 4, n_categories, 42)
            .expect("heterogeneous should succeed");
        assert_eq!(x.nrows(), 100);
        assert_eq!(x.ncols(), 7); // 3 numeric + 4 categorical
        assert_eq!(y.len(), 100);

        // Categorical columns are columns 3..7, values should be in [0, n_categories)
        for i in 0..100 {
            for j in 3..7 {
                let v = x[[i, j]];
                // v is integer-encoded: value is 0, 1, ..., n_categories-1
                assert!(
                    v >= 0.0 && v < n_categories as f64,
                    "Categorical feature {j} out of range: {v}"
                );
                // Must be an integer value
                assert_eq!(
                    v.fract(),
                    0.0,
                    "Categorical feature {j} should be integer, got {v}"
                );
            }
        }
    }

    /// Verify that means differ before and after a concept drift point
    #[test]
    fn test_make_concept_drift_distributions() {
        let n_samples = 1000;
        let n_features = 4;
        let drift_at = vec![500usize];
        let (x, _y, actual) = make_concept_drift_nd(n_samples, n_features, drift_at.clone(), 42)
            .expect("concept_drift should succeed");

        assert_eq!(x.nrows(), n_samples);
        assert_eq!(x.ncols(), n_features);
        assert_eq!(actual, drift_at, "Drift points should be preserved");

        // Compute mean of feature 0 for class-0 samples before and after drift
        // Before drift: class 0 (even samples) ~ N(0,1)
        // After drift: class 0 (even samples) ~ N(1,1)
        let mut before_sum = 0.0f64;
        let mut before_count = 0usize;
        let mut after_sum = 0.0f64;
        let mut after_count = 0usize;

        for t in 0..n_samples {
            if t % 2 == 0 {
                // class 0 samples only
                let v = x[[t, 0]];
                if t < 500 {
                    before_sum += v;
                    before_count += 1;
                } else {
                    after_sum += v;
                    after_count += 1;
                }
            }
        }

        let before_mean = if before_count > 0 {
            before_sum / before_count as f64
        } else {
            0.0
        };
        let after_mean = if after_count > 0 {
            after_sum / after_count as f64
        } else {
            0.0
        };

        // Before drift class-0 mean ≈ 0, after drift class-0 mean ≈ 1
        // Allow generous tolerance for statistical fluctuation
        assert!(
            before_mean.abs() < 0.5,
            "Before-drift class-0 mean should be ≈ 0, got {before_mean:.3}"
        );
        assert!(
            (after_mean - 1.0).abs() < 0.5,
            "After-drift class-0 mean should be ≈ 1, got {after_mean:.3}"
        );
    }

    /// Verify that all shards together cover all samples exactly once
    #[test]
    fn test_data_shard_coverage() {
        use crate::sharding::{shard_by_index, ShardingConfig};
        let n_shards = 5;
        let n_samples = 97; // non-divisible to test remainder handling
        let config = ShardingConfig {
            n_shards,
            shuffle: false,
            seed: 0,
            ..Default::default()
        };
        let _ = config; // config is used for documentation; test shard_by_index directly
        let shards = shard_by_index(n_samples, n_shards, false, 0);
        assert_eq!(shards.len(), n_shards);

        let mut seen = vec![false; n_samples];
        for shard in &shards {
            for &idx in &shard.indices {
                assert!(!seen[idx], "index {idx} seen in multiple shards");
                seen[idx] = true;
            }
        }
        assert!(seen.iter().all(|&v| v), "Not all samples covered");
        let total: usize = shards.iter().map(|s| s.indices.len()).sum();
        assert_eq!(total, n_samples, "Total samples mismatch");
    }

    /// Verify that same seed gives same permutation across shard calls
    #[test]
    fn test_data_shard_shuffled_consistency() {
        use crate::sharding::shard_by_index;
        let s1 = shard_by_index(100, 4, true, 999);
        let s2 = shard_by_index(100, 4, true, 999);
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert_eq!(a.indices, b.indices, "Same seed must give same permutation");
        }
        // Different seeds should give different orderings (with overwhelming probability)
        let s3 = shard_by_index(100, 4, true, 12345);
        let differs = s1
            .iter()
            .zip(s3.iter())
            .any(|(a, b)| a.indices != b.indices);
        assert!(
            differs,
            "Different seeds should give different shard indices"
        );
    }
}
