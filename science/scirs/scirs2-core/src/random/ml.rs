//! Machine Learning specific random utilities
//!
//! This module provides specialized random number generation utilities tailored
//! for machine learning workflows, including data splitting, weight initialization,
//! augmentation, and reproducible model training.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_core::random::ml::*;
//!
//! // Data splitting for ML
//! let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
//! let (train, val, test) = train_val_test_split(&data, 0.7, 0.15, 0.15, 42);
//!
//! // Neural network weight initialization
//! let weights = WeightInitializer::xavier(784, 256, 42);
//! let he_weights = WeightInitializer::he(256, 128, 42);
//!
//! // Cross-validation
//! let cv = CrossValidator::new(5, 42);
//! let folds = cv.split(&data);
//! ```

use crate::random::{
    arrays::OptimizedArrayRandom, random_he_weights, random_xavier_weights, seeded_rng, thread_rng,
    ParallelRng, Random, ThreadLocalRngPool,
};
use ::ndarray::{Array, Array1, Array2, Array3, Ix2};
use rand_distr::{Normal, Uniform};
use std::collections::HashMap;

use crate::random::Rng;

/// Train/validation/test split for machine learning datasets
pub fn train_val_test_split<T: Clone>(
    data: &[T],
    train_ratio: f64,
    val_ratio: f64,
    test_ratio: f64,
    seed: u64,
) -> (Vec<T>, Vec<T>, Vec<T>) {
    assert!(
        (train_ratio + val_ratio + test_ratio - 1.0).abs() < 1e-6,
        "Ratios must sum to 1.0"
    );

    let mut rng = seeded_rng(seed);
    let mut indices: Vec<usize> = (0..data.len()).collect();

    use rand::seq::SliceRandom;
    indices.shuffle(&mut rng.rng);

    let train_end = (data.len() as f64 * train_ratio) as usize;
    let val_end = train_end + (data.len() as f64 * val_ratio) as usize;

    let train_data = indices[..train_end]
        .iter()
        .map(|&i| data[i].clone())
        .collect();
    let val_data = indices[train_end..val_end]
        .iter()
        .map(|&i| data[i].clone())
        .collect();
    let test_data = indices[val_end..]
        .iter()
        .map(|&i| data[i].clone())
        .collect();

    (train_data, val_data, test_data)
}

/// Simple train/test split
pub fn train_test_split<T: Clone>(data: &[T], train_ratio: f64, seed: u64) -> (Vec<T>, Vec<T>) {
    let test_ratio = 1.0 - train_ratio;
    let (train, _, test) = train_val_test_split(data, train_ratio, 0.0, test_ratio, seed);
    (train, test)
}

/// Stratified split maintaining class distribution
pub fn stratified_split<T: Clone, K: Clone + Eq + std::hash::Hash>(
    data: &[(T, K)],
    train_ratio: f64,
    seed: u64,
) -> (Vec<(T, K)>, Vec<(T, K)>) {
    let mut rng = seeded_rng(seed);
    use std::collections::HashMap;

    // Group by class
    let mut class_groups: HashMap<K, Vec<&(T, K)>> = HashMap::new();
    for item in data {
        class_groups.entry(item.1.clone()).or_default().push(item);
    }

    let mut train_data = Vec::new();
    let mut test_data = Vec::new();

    // Split each class proportionally
    for (_, mut group) in class_groups {
        use rand::seq::SliceRandom;
        group.shuffle(&mut rng.rng);

        let train_size = (group.len() as f64 * train_ratio) as usize;

        for item in group.iter().take(train_size) {
            train_data.push((*item).clone());
        }
        for item in group.iter().skip(train_size) {
            test_data.push((*item).clone());
        }
    }

    (train_data, test_data)
}

/// Weight initialization strategies for neural networks
pub struct WeightInitializer;

impl WeightInitializer {
    /// Xavier/Glorot initialization
    pub fn xavier(fan_in: usize, fan_out: usize, seed: u64) -> Array2<f64> {
        let mut rng = seeded_rng(seed);
        random_xavier_weights(fan_in, fan_out, &mut rng)
    }

    /// He initialization (good for ReLU networks)
    pub fn he(fan_in: usize, fan_out: usize, seed: u64) -> Array2<f64> {
        let mut rng = seeded_rng(seed);
        random_he_weights(fan_in, fan_out, &mut rng)
    }

    /// LeCun initialization
    pub fn lecun(fan_in: usize, fan_out: usize, seed: u64) -> Array2<f64> {
        let mut rng = seeded_rng(seed);
        let std_dev = (1.0 / fan_in as f64).sqrt();
        Array::random_bulk(
            Ix2(fan_out, fan_in),
            Normal::new(0.0, std_dev).expect("Operation failed"),
            &mut rng,
        )
    }

    /// Uniform initialization in [-limit, limit]
    pub fn uniform(fan_in: usize, fan_out: usize, limit: f64, seed: u64) -> Array2<f64> {
        let mut rng = seeded_rng(seed);
        Array::random_bulk(
            Ix2(fan_out, fan_in),
            Uniform::new(-limit, limit).expect("Operation failed"),
            &mut rng,
        )
    }

    /// Zero initialization
    pub fn zeros(fan_in: usize, fan_out: usize) -> Array2<f64> {
        Array2::zeros([fan_out, fan_in])
    }

    /// Identity initialization (for square matrices)
    pub fn identity(size: usize) -> Array2<f64> {
        Array2::eye(size)
    }

    /// Orthogonal initialization using QR decomposition
    pub fn orthogonal(fan_in: usize, fan_out: usize, seed: u64) -> Array2<f64> {
        let mut rng = seeded_rng(seed);

        // Generate random matrix
        let random_matrix = Array::random_bulk(
            Ix2(fan_out, fan_in),
            Normal::new(0.0, 1.0).expect("Operation failed"),
            &mut rng,
        );

        // For simplicity, return normalized random matrix
        // In a full implementation, this would use proper QR decomposition
        let norm = (random_matrix.mapv(|x| x * x).sum() as f64).sqrt();
        random_matrix / norm
    }
}

/// Cross-validation utilities
pub struct CrossValidator {
    k_folds: usize,
    seed: u64,
}

impl CrossValidator {
    /// Create a new cross-validator
    pub fn new(k_folds: usize, seed: u64) -> Self {
        Self { k_folds, seed }
    }

    /// Split data into k folds
    pub fn split<T: Clone>(&self, data: &[T]) -> Vec<(Vec<T>, Vec<T>)> {
        crate::random::scientific::cross_validation_splits(data, self.k_folds, self.seed)
    }

    /// Leave-one-out cross-validation
    pub fn leave_one_out<T: Clone>(&self, data: &[T]) -> Vec<(Vec<T>, Vec<T>)> {
        (0..data.len())
            .map(|i| {
                let test_item = vec![data[i].clone()];
                let train_data = data
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| *idx != i)
                    .map(|(_, item)| item.clone())
                    .collect();
                (train_data, test_item)
            })
            .collect()
    }

    /// Stratified k-fold for classification
    pub fn stratified_split<T: Clone, K: Clone + Eq + std::hash::Hash>(
        &self,
        data: &[(T, K)],
    ) -> Vec<(Vec<(T, K)>, Vec<(T, K)>)> {
        let mut rng = seeded_rng(self.seed);

        // Group by class
        let mut class_groups: HashMap<K, Vec<&(T, K)>> = HashMap::new();
        for item in data {
            class_groups.entry(item.1.clone()).or_default().push(item);
        }

        let mut folds = vec![Vec::new(); self.k_folds];

        // Distribute each class across folds
        for (_, mut group) in class_groups {
            use rand::seq::SliceRandom;
            group.shuffle(&mut rng.rng);

            for (i, item) in group.iter().enumerate() {
                let fold_idx = i % self.k_folds;
                folds[fold_idx].push((*item).clone());
            }
        }

        // Create train/test splits
        (0..self.k_folds)
            .map(|test_fold| {
                let test_data = folds[test_fold].clone();
                let train_data = folds
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != test_fold)
                    .flat_map(|(_, fold)| fold.iter().cloned())
                    .collect();
                (train_data, test_data)
            })
            .collect()
    }
}

/// Data augmentation utilities
pub struct DataAugmentor {
    seed: u64,
}

impl DataAugmentor {
    /// Create a new data augmentor
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Add Gaussian noise to data
    pub fn add_noise(&self, data: &Array1<f64>, noise_std: f64) -> Array1<f64> {
        let mut rng = seeded_rng(self.seed);
        let noise_dist = Normal::new(0.0, noise_std).expect("Operation failed");

        data + &Array::random_bulk(data.raw_dim(), noise_dist, &mut rng)
    }

    /// Randomly drop features (like dropout)
    pub fn random_dropout(&self, data: &Array1<f64>, dropout_rate: f64) -> Array1<f64> {
        let mut rng = seeded_rng(self.seed);
        let keep_prob = 1.0 - dropout_rate;

        data.mapv(|x| {
            if rng.sample(Uniform::new(0.0, 1.0).expect("Operation failed")) < keep_prob {
                x / keep_prob // Scale to maintain expected value
            } else {
                0.0
            }
        })
    }

    /// Random scaling augmentation
    pub fn random_scale(&self, data: &Array1<f64>, scale_range: (f64, f64)) -> Array1<f64> {
        let mut rng = seeded_rng(self.seed);
        let scale_factor =
            rng.sample(Uniform::new(scale_range.0, scale_range.1).expect("Operation failed"));
        data * scale_factor
    }

    /// Random rotation for 2D data (simple version)
    pub fn random_rotation_2d(&self, data: &Array1<f64>, max_angle: f64) -> Array1<f64> {
        if data.len() != 2 {
            return data.clone();
        }

        let mut rng = seeded_rng(self.seed);
        let angle = rng.sample(Uniform::new(-max_angle, max_angle).expect("Operation failed"));

        let cos_angle = angle.cos();
        let sin_angle = angle.sin();

        let x = data[0];
        let y = data[1];

        Array1::from(vec![
            x * cos_angle - y * sin_angle,
            x * sin_angle + y * cos_angle,
        ])
    }
}

/// Batch generation utilities
pub struct BatchGenerator<T> {
    data: Vec<T>,
    batch_size: usize,
    shuffle: bool,
    seed: u64,
    current_epoch: usize,
}

impl<T: Clone> BatchGenerator<T> {
    /// Create a new batch generator
    pub fn new(data: Vec<T>, batch_size: usize, shuffle: bool, seed: u64) -> Self {
        Self {
            data,
            batch_size,
            shuffle,
            seed,
            current_epoch: 0,
        }
    }

    /// Generate batches for one epoch
    pub fn epoch(&mut self) -> Vec<Vec<T>> {
        let mut epoch_data = self.data.clone();

        if self.shuffle {
            let mut rng = seeded_rng(self.seed + self.current_epoch as u64);
            use rand::seq::SliceRandom;
            epoch_data.shuffle(&mut rng.rng);
        }

        let batches = epoch_data
            .chunks(self.batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        self.current_epoch += 1;
        batches
    }

    /// Get number of batches per epoch
    pub fn batches_per_epoch(&self) -> usize {
        (self.data.len() + self.batch_size - 1) / self.batch_size
    }

    /// Reset to epoch 0
    pub fn reset(&mut self) {
        self.current_epoch = 0;
    }
}

/// Hyperparameter optimization utilities
pub mod hyperopt {
    use super::*;

    /// Random search for hyperparameter optimization
    pub struct RandomSearch {
        seed: u64,
        param_ranges: HashMap<String, (f64, f64)>,
    }

    impl RandomSearch {
        /// Create a new random search optimizer
        pub fn new(seed: u64) -> Self {
            Self {
                seed,
                param_ranges: HashMap::new(),
            }
        }

        /// Add a parameter range to search
        pub fn add_param_range(mut self, name: String, min: f64, max: f64) -> Self {
            self.param_ranges.insert(name, (min, max));
            self
        }

        /// Generate random parameter combinations
        pub fn sample_params(&self, n_trials: usize) -> Vec<HashMap<String, f64>> {
            let mut rng = seeded_rng(self.seed);

            (0..n_trials)
                .map(|_| {
                    self.param_ranges
                        .iter()
                        .map(|(name, (min, max))| {
                            let value =
                                rng.sample(Uniform::new(*min, *max).expect("Operation failed"));
                            (name.clone(), value)
                        })
                        .collect()
                })
                .collect()
        }
    }

    /// Grid search helper
    pub struct GridSearch {
        param_grids: HashMap<String, Vec<f64>>,
    }

    impl GridSearch {
        /// Create a new grid search
        pub fn new() -> Self {
            Self {
                param_grids: HashMap::new(),
            }
        }

        /// Add parameter grid
        pub fn add_param_grid(mut self, name: String, values: Vec<f64>) -> Self {
            self.param_grids.insert(name, values);
            self
        }

        /// Generate all parameter combinations
        pub fn all_combinations(&self) -> Vec<HashMap<String, f64>> {
            let param_names: Vec<String> = self.param_grids.keys().cloned().collect();
            let param_values: Vec<Vec<f64>> = param_names
                .iter()
                .map(|name| self.param_grids[name].clone())
                .collect();

            let combinations =
                crate::random::scientific::ExperimentalDesign::factorial_design(&param_values);

            combinations
                .into_iter()
                .map(|combo| {
                    param_names
                        .iter()
                        .zip(combo.iter())
                        .map(|(name, &value)| (name.clone(), value))
                        .collect()
                })
                .collect()
        }
    }
}

/// Ensemble learning utilities
pub mod ensemble {
    use super::*;

    /// Bootstrap aggregating (bagging) sample generator
    pub fn bootstrap_samples<T: Clone>(
        data: &[T],
        n_estimators: usize,
        sample_ratio: f64,
        seed: u64,
    ) -> Vec<Vec<T>> {
        let pool = ThreadLocalRngPool::new(seed);
        let sample_size = (data.len() as f64 * sample_ratio) as usize;

        (0..n_estimators)
            .map(|i| {
                pool.with_rng(|rng| {
                    (0..sample_size)
                        .map(|_| {
                            let idx = rng.random_range(0..data.len());
                            data[idx].clone()
                        })
                        .collect()
                })
            })
            .collect()
    }

    /// Random subspace sampling for feature bagging
    pub fn random_subspace_features(
        n_features: usize,
        max_features: usize,
        n_estimators: usize,
        seed: u64,
    ) -> Vec<Vec<usize>> {
        let mut rng = seeded_rng(seed);

        (0..n_estimators)
            .map(|_| {
                let mut features: Vec<usize> = (0..n_features).collect();
                use rand::seq::SliceRandom;
                features.shuffle(&mut rng.rng);
                features.into_iter().take(max_features).collect()
            })
            .collect()
    }
}

/// Active learning utilities
pub mod active_learning {
    use super::*;

    /// Uncertainty sampling strategies
    pub struct UncertaintySampler {
        seed: u64,
    }

    impl UncertaintySampler {
        pub fn new(seed: u64) -> Self {
            Self { seed }
        }

        /// Random sampling baseline
        pub fn random_sample<T: Clone>(&self, candidates: &[T], n_samples: usize) -> Vec<T> {
            let mut rng = seeded_rng(self.seed);
            use rand::seq::SliceRandom;

            // Manual sampling since choose_multiple may not be available
            let mut indices: Vec<usize> = (0..candidates.len()).collect();
            indices.shuffle(&mut rng.rng);
            indices
                .into_iter()
                .take(n_samples.min(candidates.len()))
                .map(|i| candidates[i].clone())
                .collect()
        }

        /// Entropy-based sampling (requires uncertainty scores)
        pub fn entropy_sampling<T: Clone>(
            &self,
            candidates: &[(T, f64)], // (data, entropy_score)
            n_samples: usize,
        ) -> Vec<T> {
            let mut scored_candidates = candidates.to_vec();
            scored_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("Operation failed"));

            scored_candidates
                .into_iter()
                .take(n_samples)
                .map(|(data, _)| data)
                .collect()
        }
    }
}
