//! User-item matrix for collaborative filtering.

use crate::dense_linalg::DenseMatrix;
use crate::error::RecommendResult;
use crate::{ContentMetadata, Recommendation, RecommendationReason, RecommendationRequest};
use std::collections::HashMap;
use uuid::Uuid;

/// User-item interaction matrix
#[derive(Debug, Clone)]
pub struct UserItemMatrix {
    /// Matrix data (users x items)
    data: DenseMatrix,
    /// User ID to index mapping
    user_to_index: HashMap<Uuid, usize>,
    /// Item ID to index mapping
    item_to_index: HashMap<Uuid, usize>,
    /// Index to user ID mapping
    index_to_user: Vec<Uuid>,
    /// Index to item ID mapping
    index_to_item: Vec<Uuid>,
}

impl UserItemMatrix {
    /// Create a new user-item matrix
    #[must_use]
    pub fn new(num_users: usize, num_items: usize) -> Self {
        Self {
            data: DenseMatrix::zeros(num_users, num_items),
            user_to_index: HashMap::new(),
            item_to_index: HashMap::new(),
            index_to_user: Vec::new(),
            index_to_item: Vec::new(),
        }
    }

    /// Add a user
    pub fn add_user(&mut self, user_id: Uuid) -> usize {
        if let Some(&index) = self.user_to_index.get(&user_id) {
            return index;
        }

        let index = self.index_to_user.len();
        self.user_to_index.insert(user_id, index);
        self.index_to_user.push(user_id);

        // Expand matrix if needed
        if index >= self.data.nrows() {
            let new_rows = index + 1 - self.data.nrows();
            let zeros = DenseMatrix::zeros(new_rows, self.data.ncols());
            self.data = self.data.concat_rows(&zeros);
        }

        index
    }

    /// Add an item
    pub fn add_item(&mut self, item_id: Uuid) -> usize {
        if let Some(&index) = self.item_to_index.get(&item_id) {
            return index;
        }

        let index = self.index_to_item.len();
        self.item_to_index.insert(item_id, index);
        self.index_to_item.push(item_id);

        // Expand matrix if needed
        if index >= self.data.ncols() {
            let new_cols = index + 1 - self.data.ncols();
            let zeros = DenseMatrix::zeros(self.data.nrows(), new_cols);
            self.data = self.data.concat_cols(&zeros);
        }

        index
    }

    /// Set rating for user-item pair
    pub fn set_rating(&mut self, user_id: Uuid, item_id: Uuid, rating: f32) {
        let user_idx = self.add_user(user_id);
        let item_idx = self.add_item(item_id);
        self.data.set(user_idx, item_idx, rating);
    }

    /// Get rating for user-item pair
    #[must_use]
    pub fn get_rating(&self, user_id: Uuid, item_id: Uuid) -> Option<f32> {
        let user_idx = self.user_to_index.get(&user_id)?;
        let item_idx = self.item_to_index.get(&item_id)?;
        Some(self.data.get(*user_idx, *item_idx))
    }

    /// Get user's ratings vector
    #[must_use]
    pub fn get_user_ratings(&self, user_id: Uuid) -> Option<Vec<f32>> {
        let user_idx = self.user_to_index.get(&user_id)?;
        Some(self.data.row_vec(*user_idx))
    }

    /// Get item's ratings vector
    #[must_use]
    pub fn get_item_ratings(&self, item_id: Uuid) -> Option<Vec<f32>> {
        let item_idx = self.item_to_index.get(&item_id)?;
        Some(self.data.col_vec(*item_idx))
    }

    /// Get number of rows in the underlying data matrix
    #[must_use]
    pub fn data_nrows(&self) -> usize {
        self.data.nrows()
    }

    /// Get number of columns in the underlying data matrix
    #[must_use]
    pub fn data_ncols(&self) -> usize {
        self.data.ncols()
    }

    /// Get a value from the underlying data matrix by indices
    #[must_use]
    pub fn data_get(&self, row: usize, col: usize) -> f32 {
        self.data.get(row, col)
    }

    /// Get a row from the underlying data matrix as a Vec
    #[must_use]
    pub fn data_row_vec(&self, row: usize) -> Vec<f32> {
        self.data.row_vec(row)
    }

    /// Get number of users
    #[must_use]
    pub fn num_users(&self) -> usize {
        self.index_to_user.len()
    }

    /// Get number of items
    #[must_use]
    pub fn num_items(&self) -> usize {
        self.index_to_item.len()
    }

    /// Get item ID by index
    #[must_use]
    pub fn get_item_id(&self, index: usize) -> Option<Uuid> {
        self.index_to_item.get(index).copied()
    }

    /// Get user ID by index
    #[must_use]
    pub fn get_user_id(&self, index: usize) -> Option<Uuid> {
        self.index_to_user.get(index).copied()
    }

    /// Find items rated by user
    #[must_use]
    pub fn get_rated_items(&self, user_id: Uuid) -> Vec<(Uuid, f32)> {
        let Some(&user_idx) = self.user_to_index.get(&user_id) else {
            return Vec::new();
        };

        let row = self.data.row_vec(user_idx);
        row.iter()
            .enumerate()
            .filter(|(_, &rating)| rating > 0.0)
            .filter_map(|(item_idx, &rating)| {
                self.index_to_item
                    .get(item_idx)
                    .map(|&item_id| (item_id, rating))
            })
            .collect()
    }
}

/// Configuration for incremental matrix factorization.
#[derive(Debug, Clone)]
pub struct IncrementalMfConfig {
    /// Number of latent factors (embedding dimension).
    pub num_factors: usize,
    /// Learning rate for SGD updates.
    pub learning_rate: f32,
    /// Regularization parameter.
    pub regularization: f32,
    /// Number of SGD passes per incremental update.
    pub update_iterations: usize,
}

impl Default for IncrementalMfConfig {
    fn default() -> Self {
        Self {
            num_factors: 16,
            learning_rate: 0.01,
            regularization: 0.02,
            update_iterations: 5,
        }
    }
}

/// Latent factor model trained via incremental SGD.
///
/// User factors: `num_users x num_factors`
/// Item factors: `num_items x num_factors`
#[derive(Debug, Clone)]
pub struct LatentFactorModel {
    /// User factor matrix
    user_factors: DenseMatrix,
    /// Item factor matrix
    item_factors: DenseMatrix,
    /// Global mean rating
    global_mean: f32,
    /// User biases
    user_bias: Vec<f32>,
    /// Item biases
    item_bias: Vec<f32>,
    /// Total number of ratings ingested
    total_ratings: u64,
    /// Running sum of all ratings (for incremental mean)
    rating_sum: f64,
    /// Precomputed per-user embedding cache (user_idx → factor row).
    ///
    /// Populated by [`LatentFactorModel::precompute_user_embeddings`] and
    /// cleared whenever the model is retrained.
    user_embedding_cache: Option<HashMap<usize, Vec<f32>>>,
}

impl LatentFactorModel {
    /// Create a new empty latent factor model.
    #[must_use]
    pub fn new(num_factors: usize) -> Self {
        Self {
            user_factors: DenseMatrix::zeros(0, num_factors),
            item_factors: DenseMatrix::zeros(0, num_factors),
            global_mean: 0.0,
            user_bias: Vec::new(),
            item_bias: Vec::new(),
            total_ratings: 0,
            rating_sum: 0.0,
            user_embedding_cache: None,
        }
    }

    /// Number of users currently tracked by this model.
    #[must_use]
    pub fn num_users(&self) -> usize {
        self.user_factors.nrows()
    }

    /// Number of items currently tracked by this model.
    #[must_use]
    pub fn num_items(&self) -> usize {
        self.item_factors.nrows()
    }

    /// Extract the embedding (factor row) for a user.
    #[must_use]
    fn user_factors_row(&self, user_idx: usize) -> Vec<f32> {
        self.user_factors.row_vec(user_idx)
    }

    /// Extract the embedding (factor row) for an item.
    #[must_use]
    fn item_factors_row(&self, item_idx: usize) -> Vec<f32> {
        self.item_factors.row_vec(item_idx)
    }

    /// Precompute and cache every user's embedding row.
    ///
    /// After calling this, [`Self::recommend_precomputed`] can serve
    /// top-k recommendations with a single cache lookup + dot products.
    /// The cache is invalidated automatically when the model is retrained
    /// (see [`CollaborativeEngine::retrain`]).
    pub fn precompute_user_embeddings(&mut self) {
        let mut cache = HashMap::with_capacity(self.user_factors.nrows());
        for user_idx in 0..self.user_factors.nrows() {
            cache.insert(user_idx, self.user_factors_row(user_idx));
        }
        self.user_embedding_cache = Some(cache);
    }

    /// Recommend the top-`top_k` item indices for a user using the precomputed
    /// embedding cache.
    ///
    /// Returns an empty vec if `user_idx` is not in the cache.
    ///
    /// # Errors (panics)
    ///
    /// Panics with a clear message if [`Self::precompute_user_embeddings`] has
    /// not been called yet — this is an explicit programming-model violation
    /// rather than a runtime error that callers are expected to handle silently.
    #[must_use]
    pub fn recommend_precomputed(&self, user_idx: usize, top_k: usize) -> Vec<usize> {
        let cache = self
            .user_embedding_cache
            .as_ref()
            .expect("call precompute_user_embeddings() before recommend_precomputed()");

        let user_vec = match cache.get(&user_idx) {
            Some(v) => v,
            None => return vec![],
        };

        let num_items = self.item_factors.nrows();
        let mut scores: Vec<(usize, f32)> = (0..num_items)
            .map(|item_idx| {
                let item_vec = self.item_factors_row(item_idx);
                let dot: f32 = user_vec
                    .iter()
                    .zip(item_vec.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let bu = self.user_bias.get(user_idx).copied().unwrap_or(0.0);
                let bi = self.item_bias.get(item_idx).copied().unwrap_or(0.0);
                let score = self.global_mean + bu + bi + dot;
                (item_idx, score)
            })
            .collect();

        scores.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.into_iter().take(top_k).map(|(i, _)| i).collect()
    }

    /// Recommend top-`top_k` items for `user_idx` using the on-demand (non-cached) path.
    ///
    /// This is equivalent to [`Self::recommend_precomputed`] but always re-reads from
    /// the factor matrices.  Use it to cross-validate the cache.
    #[must_use]
    pub fn recommend_on_demand(&self, user_idx: usize, top_k: usize) -> Vec<usize> {
        if user_idx >= self.user_factors.nrows() {
            return vec![];
        }
        let num_items = self.item_factors.nrows();
        let mut scores: Vec<(usize, f32)> = (0..num_items)
            .map(|item_idx| (item_idx, self.predict(user_idx, item_idx)))
            .collect();
        scores.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.into_iter().take(top_k).map(|(i, _)| i).collect()
    }

    /// Ensure user row exists, expanding factors if needed.
    fn ensure_user(&mut self, user_idx: usize, num_factors: usize) {
        while user_idx >= self.user_factors.nrows() {
            // Append a row with small initial values
            let init_row = DenseMatrix::zeros(1, num_factors);
            let mut init = init_row;
            let seed = (self.user_factors.nrows() as f32 + 1.0) * 0.01;
            for j in 0..num_factors {
                init.set(0, j, seed / (j as f32 + 1.0));
            }
            self.user_factors = self.user_factors.concat_rows(&init);
            self.user_bias.push(0.0);
        }
    }

    /// Ensure item row exists, expanding factors if needed.
    fn ensure_item(&mut self, item_idx: usize, num_factors: usize) {
        while item_idx >= self.item_factors.nrows() {
            let init_row = DenseMatrix::zeros(1, num_factors);
            let mut init = init_row;
            let seed = (self.item_factors.nrows() as f32 + 1.0) * 0.01;
            for j in 0..num_factors {
                init.set(0, j, seed / (j as f32 + 1.0));
            }
            self.item_factors = self.item_factors.concat_rows(&init);
            self.item_bias.push(0.0);
        }
    }

    /// Predict rating for a user-item pair.
    #[must_use]
    pub fn predict(&self, user_idx: usize, item_idx: usize) -> f32 {
        if user_idx >= self.user_factors.nrows() || item_idx >= self.item_factors.nrows() {
            return self.global_mean;
        }
        let num_factors = self.user_factors.ncols();
        let mut dot = 0.0f32;
        for k in 0..num_factors {
            dot += self.user_factors.get(user_idx, k) * self.item_factors.get(item_idx, k);
        }
        let bu = self.user_bias.get(user_idx).copied().unwrap_or(0.0);
        let bi = self.item_bias.get(item_idx).copied().unwrap_or(0.0);
        self.global_mean + bu + bi + dot
    }

    /// Perform a single SGD step on one (user, item, rating) observation.
    fn sgd_step(&mut self, user_idx: usize, item_idx: usize, rating: f32, lr: f32, reg: f32) {
        let num_factors = self.user_factors.ncols();
        let pred = self.predict(user_idx, item_idx);
        let error = rating - pred;

        // Update biases
        if let Some(bu) = self.user_bias.get_mut(user_idx) {
            *bu += lr * (error - reg * *bu);
        }
        if let Some(bi) = self.item_bias.get_mut(item_idx) {
            *bi += lr * (error - reg * *bi);
        }

        // Update latent factors
        for k in 0..num_factors {
            let pu = self.user_factors.get(user_idx, k);
            let qi = self.item_factors.get(item_idx, k);
            self.user_factors
                .set(user_idx, k, pu + lr * (error * qi - reg * pu));
            self.item_factors
                .set(item_idx, k, qi + lr * (error * pu - reg * qi));
        }
    }

    /// Number of latent factors.
    #[must_use]
    pub fn num_factors(&self) -> usize {
        self.user_factors.ncols()
    }

    /// Global mean rating.
    #[must_use]
    pub fn global_mean(&self) -> f32 {
        self.global_mean
    }

    /// Total ratings processed.
    #[must_use]
    pub fn total_ratings(&self) -> u64 {
        self.total_ratings
    }
}

/// Collaborative filtering engine
pub struct CollaborativeEngine {
    /// User-item matrix
    matrix: UserItemMatrix,
    /// Content metadata
    content_metadata: HashMap<Uuid, ContentMetadata>,
    /// K-nearest neighbors calculator
    knn: super::knn::KnnCalculator,
    /// Latent factor model for incremental MF
    factor_model: LatentFactorModel,
    /// Incremental MF configuration
    mf_config: IncrementalMfConfig,
}

impl CollaborativeEngine {
    /// Create a new collaborative engine
    #[must_use]
    pub fn new() -> Self {
        let mf_config = IncrementalMfConfig::default();
        let factor_model = LatentFactorModel::new(mf_config.num_factors);
        Self {
            matrix: UserItemMatrix::new(0, 0),
            content_metadata: HashMap::new(),
            knn: super::knn::KnnCalculator::new(10),
            factor_model,
            mf_config,
        }
    }

    /// Create a collaborative engine with custom MF configuration.
    #[must_use]
    pub fn with_mf_config(config: IncrementalMfConfig) -> Self {
        let factor_model = LatentFactorModel::new(config.num_factors);
        Self {
            matrix: UserItemMatrix::new(0, 0),
            content_metadata: HashMap::new(),
            knn: super::knn::KnnCalculator::new(10),
            factor_model,
            mf_config: config,
        }
    }

    /// Add a rating (updates both the raw matrix and the latent factor model incrementally).
    pub fn add_rating(&mut self, user_id: Uuid, content_id: Uuid, rating: f32) {
        self.matrix.set_rating(user_id, content_id, rating);
        self.incremental_update(user_id, content_id, rating);
    }

    /// Perform incremental matrix factorization update for a single new rating.
    ///
    /// Runs several SGD passes on the single observation to fold the new
    /// data point into the latent factor model without retraining from scratch.
    fn incremental_update(&mut self, user_id: Uuid, content_id: Uuid, rating: f32) {
        let user_idx = self
            .matrix
            .user_to_index
            .get(&user_id)
            .copied()
            .unwrap_or(0);
        let item_idx = self
            .matrix
            .item_to_index
            .get(&content_id)
            .copied()
            .unwrap_or(0);

        let num_factors = self.mf_config.num_factors;
        self.factor_model.ensure_user(user_idx, num_factors);
        self.factor_model.ensure_item(item_idx, num_factors);

        // Update global mean incrementally
        self.factor_model.rating_sum += f64::from(rating);
        self.factor_model.total_ratings += 1;
        self.factor_model.global_mean =
            (self.factor_model.rating_sum / self.factor_model.total_ratings as f64) as f32;

        let lr = self.mf_config.learning_rate;
        let reg = self.mf_config.regularization;
        let iters = self.mf_config.update_iterations;

        for _ in 0..iters {
            self.factor_model
                .sgd_step(user_idx, item_idx, rating, lr, reg);
        }
    }

    /// Predict rating using the latent factor model.
    #[must_use]
    pub fn predict_rating(&self, user_id: Uuid, content_id: Uuid) -> f32 {
        let user_idx = self
            .matrix
            .user_to_index
            .get(&user_id)
            .copied()
            .unwrap_or(0);
        let item_idx = self
            .matrix
            .item_to_index
            .get(&content_id)
            .copied()
            .unwrap_or(0);
        self.factor_model.predict(user_idx, item_idx)
    }

    /// Full retrain of the latent factor model from the current matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is empty.
    pub fn retrain(&mut self, epochs: usize) -> RecommendResult<()> {
        if self.matrix.num_users() == 0 || self.matrix.num_items() == 0 {
            return Err(crate::error::RecommendError::insufficient_data(
                "Cannot retrain with empty matrix",
            ));
        }

        let num_factors = self.mf_config.num_factors;
        self.factor_model = LatentFactorModel::new(num_factors);
        // Cache is implicitly cleared by replacing the entire model above.

        // Collect all non-zero ratings
        let mut observations: Vec<(usize, usize, f32)> = Vec::new();
        for u in 0..self.matrix.num_users() {
            for i in 0..self.matrix.num_items() {
                let val = self.matrix.data_get(u, i);
                if val.abs() > f32::EPSILON {
                    observations.push((u, i, val));
                }
            }
        }

        // Update global mean
        let sum: f64 = observations.iter().map(|(_, _, r)| f64::from(*r)).sum();
        self.factor_model.total_ratings = observations.len() as u64;
        self.factor_model.rating_sum = sum;
        if !observations.is_empty() {
            self.factor_model.global_mean = (sum / observations.len() as f64) as f32;
        }

        // Ensure all user/item rows exist
        for &(u, i, _) in &observations {
            self.factor_model.ensure_user(u, num_factors);
            self.factor_model.ensure_item(i, num_factors);
        }

        let lr = self.mf_config.learning_rate;
        let reg = self.mf_config.regularization;

        for _ in 0..epochs {
            for &(u, i, r) in &observations {
                self.factor_model.sgd_step(u, i, r, lr, reg);
            }
        }

        Ok(())
    }

    /// Access the latent factor model (read-only).
    #[must_use]
    pub fn factor_model(&self) -> &LatentFactorModel {
        &self.factor_model
    }

    /// Add content metadata
    pub fn add_content(&mut self, content_id: Uuid, metadata: ContentMetadata) {
        self.content_metadata.insert(content_id, metadata);
    }

    /// Get collaborative recommendations
    ///
    /// # Errors
    ///
    /// Returns an error if recommendation generation fails
    pub fn recommend(
        &self,
        request: &RecommendationRequest,
    ) -> RecommendResult<Vec<Recommendation>> {
        // Find similar users
        let similar_users = self
            .knn
            .find_similar_users(&self.matrix, request.user_id, 20)?;

        // Get items rated by similar users
        let mut candidate_items: HashMap<Uuid, f32> = HashMap::new();

        for (similar_user, similarity) in similar_users {
            let rated_items = self.matrix.get_rated_items(similar_user);
            for (item_id, rating) in rated_items {
                // Skip items already rated by the user
                if self.matrix.get_rating(request.user_id, item_id).is_some() {
                    continue;
                }

                *candidate_items.entry(item_id).or_insert(0.0) += rating * similarity;
            }
        }

        // Convert to recommendations
        let mut recommendations: Vec<Recommendation> = candidate_items
            .into_iter()
            .filter_map(|(content_id, score)| {
                self.content_metadata
                    .get(&content_id)
                    .map(|metadata| Recommendation {
                        content_id,
                        score,
                        rank: 0,
                        reasons: vec![RecommendationReason::CollaborativeFiltering {
                            confidence: score,
                        }],
                        metadata: metadata.clone(),
                        explanation: None,
                    })
            })
            .collect();

        // Sort by score
        recommendations.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign ranks
        for (idx, rec) in recommendations.iter_mut().enumerate() {
            rec.rank = idx + 1;
        }

        recommendations.truncate(request.limit);

        Ok(recommendations)
    }
}

impl Default for CollaborativeEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_item_matrix_creation() {
        let matrix = UserItemMatrix::new(10, 20);
        assert_eq!(matrix.data_nrows(), 10);
        assert_eq!(matrix.data_ncols(), 20);
    }

    #[test]
    fn test_add_user() {
        let mut matrix = UserItemMatrix::new(0, 0);
        let user_id = Uuid::new_v4();
        let index = matrix.add_user(user_id);
        assert_eq!(index, 0);

        let index2 = matrix.add_user(user_id);
        assert_eq!(index2, 0); // Same index for same user
    }

    #[test]
    fn test_add_item() {
        let mut matrix = UserItemMatrix::new(0, 0);
        let item_id = Uuid::new_v4();
        let index = matrix.add_item(item_id);
        assert_eq!(index, 0);
    }

    #[test]
    fn test_set_get_rating() {
        let mut matrix = UserItemMatrix::new(0, 0);
        let user_id = Uuid::new_v4();
        let item_id = Uuid::new_v4();

        matrix.set_rating(user_id, item_id, 4.5);
        let rating = matrix.get_rating(user_id, item_id);
        assert_eq!(rating, Some(4.5));
    }

    #[test]
    fn test_collaborative_engine_creation() {
        let engine = CollaborativeEngine::new();
        assert_eq!(engine.matrix.num_users(), 0);
        assert_eq!(engine.matrix.num_items(), 0);
    }

    #[test]
    fn test_add_rating_to_engine() {
        let mut engine = CollaborativeEngine::new();
        let user_id = Uuid::new_v4();
        let content_id = Uuid::new_v4();

        engine.add_rating(user_id, content_id, 5.0);
        let rating = engine.matrix.get_rating(user_id, content_id);
        assert_eq!(rating, Some(5.0));
    }

    // ---- Incremental MF tests ----

    #[test]
    fn test_incremental_mf_config_default() {
        let config = IncrementalMfConfig::default();
        assert_eq!(config.num_factors, 16);
        assert!(config.learning_rate > 0.0);
        assert!(config.regularization > 0.0);
    }

    #[test]
    fn test_latent_factor_model_new() {
        let model = LatentFactorModel::new(8);
        assert_eq!(model.num_factors(), 8);
        assert_eq!(model.total_ratings(), 0);
        assert!((model.global_mean() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_incremental_update_single_rating() {
        let mut engine = CollaborativeEngine::new();
        let u = Uuid::new_v4();
        let i = Uuid::new_v4();
        engine.add_rating(u, i, 4.0);

        assert_eq!(engine.factor_model.total_ratings(), 1);
        assert!((engine.factor_model.global_mean() - 4.0).abs() < f32::EPSILON);

        // Prediction should move toward 4.0
        let pred = engine.predict_rating(u, i);
        assert!(pred.is_finite());
    }

    #[test]
    fn test_incremental_update_multiple_ratings() {
        let mut engine = CollaborativeEngine::new();
        let u1 = Uuid::new_v4();
        let u2 = Uuid::new_v4();
        let i1 = Uuid::new_v4();
        let i2 = Uuid::new_v4();

        engine.add_rating(u1, i1, 5.0);
        engine.add_rating(u1, i2, 3.0);
        engine.add_rating(u2, i1, 4.0);

        assert_eq!(engine.factor_model.total_ratings(), 3);
        let mean = engine.factor_model.global_mean();
        assert!((mean - 4.0).abs() < f32::EPSILON);

        let pred = engine.predict_rating(u2, i2);
        assert!(pred.is_finite());
    }

    #[test]
    fn test_retrain_empty_matrix_errors() {
        let mut engine = CollaborativeEngine::new();
        let result = engine.retrain(10);
        assert!(result.is_err());
    }

    #[test]
    fn test_retrain_converges() {
        let config = IncrementalMfConfig {
            num_factors: 4,
            learning_rate: 0.05,
            regularization: 0.01,
            update_iterations: 1,
        };
        let mut engine = CollaborativeEngine::with_mf_config(config);

        let u1 = Uuid::new_v4();
        let u2 = Uuid::new_v4();
        let i1 = Uuid::new_v4();
        let i2 = Uuid::new_v4();
        let i3 = Uuid::new_v4();

        // Build matrix without incremental (use matrix directly)
        engine.matrix.set_rating(u1, i1, 5.0);
        engine.matrix.set_rating(u1, i2, 3.0);
        engine.matrix.set_rating(u2, i1, 4.0);
        engine.matrix.set_rating(u2, i3, 2.0);

        let result = engine.retrain(50);
        assert!(result.is_ok());

        // After training, predictions for known pairs should be
        // reasonably close to the actual ratings.
        let pred_u1_i1 = engine.predict_rating(u1, i1);
        assert!(
            (pred_u1_i1 - 5.0).abs() < 1.5,
            "pred={pred_u1_i1}, expected ~5.0"
        );
    }

    #[test]
    fn test_with_mf_config() {
        let config = IncrementalMfConfig {
            num_factors: 32,
            learning_rate: 0.005,
            regularization: 0.01,
            update_iterations: 10,
        };
        let engine = CollaborativeEngine::with_mf_config(config);
        assert_eq!(engine.factor_model.num_factors(), 32);
    }

    #[test]
    fn test_predict_unknown_user_item() {
        let engine = CollaborativeEngine::new();
        let pred = engine.predict_rating(Uuid::new_v4(), Uuid::new_v4());
        // Should return global mean (0.0 for empty model)
        assert!(pred.is_finite());
    }

    #[test]
    fn test_incremental_update_preserves_matrix() {
        let mut engine = CollaborativeEngine::new();
        let u = Uuid::new_v4();
        let i1 = Uuid::new_v4();
        let i2 = Uuid::new_v4();

        engine.add_rating(u, i1, 5.0);
        engine.add_rating(u, i2, 2.0);

        assert_eq!(engine.matrix.get_rating(u, i1), Some(5.0));
        assert_eq!(engine.matrix.get_rating(u, i2), Some(2.0));
    }

    // ---- Precomputed embedding tests ----

    /// Build a small engine trained on 3 users × 4 items.
    fn build_small_engine() -> CollaborativeEngine {
        let config = IncrementalMfConfig {
            num_factors: 4,
            learning_rate: 0.05,
            regularization: 0.01,
            update_iterations: 1,
        };
        let mut engine = CollaborativeEngine::with_mf_config(config);

        let u0 = Uuid::from_u128(0x1000);
        let u1 = Uuid::from_u128(0x2000);
        let u2 = Uuid::from_u128(0x3000);
        let i0 = Uuid::from_u128(0xA000);
        let i1 = Uuid::from_u128(0xB000);
        let i2 = Uuid::from_u128(0xC000);
        let i3 = Uuid::from_u128(0xD000);

        engine.matrix.set_rating(u0, i0, 5.0);
        engine.matrix.set_rating(u0, i1, 3.0);
        engine.matrix.set_rating(u1, i1, 4.0);
        engine.matrix.set_rating(u1, i2, 2.0);
        engine.matrix.set_rating(u2, i0, 1.0);
        engine.matrix.set_rating(u2, i3, 5.0);

        engine
            .retrain(30)
            .expect("retrain must not fail on non-empty matrix");
        engine
    }

    #[test]
    fn test_precomputed_recommend_matches_on_demand() {
        let mut engine = build_small_engine();
        let model = &mut engine.factor_model;

        // Ensure there is at least one user to query.
        if model.num_users() == 0 {
            return;
        }

        model.precompute_user_embeddings();

        // Compare top-3 recommendations from both paths for every user.
        let num_users = model.num_users();
        for user_idx in 0..num_users {
            let precomputed = model.recommend_precomputed(user_idx, 3);
            let on_demand = model.recommend_on_demand(user_idx, 3);
            assert_eq!(
                precomputed, on_demand,
                "precomputed and on-demand must agree for user {user_idx}"
            );
        }
    }

    #[test]
    fn test_cache_invalidated_after_refit() {
        let mut engine = build_small_engine();

        // Precompute embeddings so the cache is populated.
        engine.factor_model.precompute_user_embeddings();
        assert!(
            engine.factor_model.user_embedding_cache.is_some(),
            "cache must be Some after precompute"
        );

        // Retrain — this replaces the whole LatentFactorModel, clearing the cache.
        engine
            .retrain(5)
            .expect("retrain should succeed on non-empty matrix");
        assert!(
            engine.factor_model.user_embedding_cache.is_none(),
            "cache must be None after retrain"
        );
    }

    #[test]
    fn test_precompute_returns_empty_for_unknown_user() {
        let mut engine = build_small_engine();
        engine.factor_model.precompute_user_embeddings();

        // user index 9999 is well outside any trained range.
        let result = engine.factor_model.recommend_precomputed(9999, 5);
        assert!(result.is_empty(), "unknown user should return empty vec");
    }
}
