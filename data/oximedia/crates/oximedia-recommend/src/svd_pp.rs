//! SVD++ — Enhanced Matrix Factorization with Implicit Feedback.
//!
//! SVD++ extends Funk-SVD by incorporating the *implicit* signal of which
//! items a user has interacted with, regardless of the explicit rating value.
//! The prediction rule is:
//!
//! ```text
//! r̂_ui = μ + b_u + b_i + q_i^T (p_u + |N(u)|^{-½} Σ_{j∈N(u)} y_j)
//! ```
//!
//! where
//! - μ  = global mean rating
//! - b_u = user bias
//! - b_i = item bias
//! - q_i = item latent factor vector
//! - p_u = user latent factor vector
//! - y_j = implicit feedback factor for item j
//! - N(u) = set of items user u has interacted with
//!
//! Training uses SGD with separate learning-rate schedules for biases and
//! factors, plus independent L2 regularisation for each parameter group.
//!
//! # Example
//!
//! ```
//! use oximedia_recommend::svd_pp::{SvdPpConfig, SvdPpModel};
//! use oximedia_recommend::als::Rating;
//!
//! let ratings = vec![
//!     Rating { user_id: 0, item_id: 0, rating: 5.0 },
//!     Rating { user_id: 0, item_id: 1, rating: 3.0 },
//!     Rating { user_id: 1, item_id: 0, rating: 4.0 },
//!     Rating { user_id: 1, item_id: 2, rating: 2.0 },
//!     Rating { user_id: 2, item_id: 1, rating: 5.0 },
//!     Rating { user_id: 2, item_id: 2, rating: 4.0 },
//! ];
//!
//! let config = SvdPpConfig { n_factors: 4, n_epochs: 10, ..SvdPpConfig::default() };
//! let model = SvdPpModel::fit(&ratings, config).expect("training failed");
//! assert!(model.predict(0, 1).is_some());
//! ```

use std::collections::HashMap;

use crate::als::{AlsError, Rating};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Hyper-parameters for SVD++ training.
#[derive(Debug, Clone)]
pub struct SvdPpConfig {
    /// Number of latent factors.  Default: 10.
    pub n_factors: usize,
    /// Number of SGD epochs.  Default: 20.
    pub n_epochs: usize,
    /// SGD learning rate for factor updates.  Default: 0.007.
    pub learning_rate: f32,
    /// L2 regularisation coefficient for factor parameters.  Default: 0.02.
    pub regularization: f32,
    /// L2 regularisation coefficient for bias parameters.  Default: 0.005.
    pub bias_regularization: f32,
    /// Random seed for reproducible initialisation.  Default: 42.
    pub seed: u64,
}

impl Default for SvdPpConfig {
    fn default() -> Self {
        Self {
            n_factors: 10,
            n_epochs: 20,
            learning_rate: 0.007,
            regularization: 0.02,
            bias_regularization: 0.005,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Seeded PRNG (same LC generator used by als.rs)
// ---------------------------------------------------------------------------

struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.state >> 33) as u32;
        (bits as f32 + 0.5) / 2_147_483_648.0
    }

    fn next_init(&mut self) -> f32 {
        (self.next_f32() - 0.5) * 0.1
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// A trained SVD++ model.
#[derive(Debug)]
pub struct SvdPpModel {
    /// Global mean rating μ.
    pub global_mean: f32,
    /// User bias b_u  (indexed by internal row).
    user_bias: Vec<f32>,
    /// Item bias b_i  (indexed by internal row).
    item_bias: Vec<f32>,
    /// User latent factors p_u  [n_users × n_factors].
    user_factors: Vec<Vec<f32>>,
    /// Item latent factors q_i  [n_items × n_factors].
    item_factors: Vec<Vec<f32>>,
    /// Implicit feedback factors y_j  [n_items × n_factors].
    implicit_factors: Vec<Vec<f32>>,
    /// External user_id → internal row index.
    user_index: HashMap<u32, usize>,
    /// External item_id → internal row index.
    item_index: HashMap<u32, usize>,
    /// Reverse: internal row → external user_id.
    user_ids: Vec<u32>,
    /// Reverse: internal row → external item_id.
    item_ids: Vec<u32>,
    /// Implicit feedback set N(u): user row → sorted vec of item rows.
    implicit_feedback: HashMap<usize, Vec<usize>>,
    /// Config snapshot (for n_factors accessor).
    config: SvdPpConfig,
}

impl SvdPpModel {
    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    /// Train a SVD++ model from an explicit rating dataset.
    ///
    /// # Errors
    ///
    /// Returns [`AlsError::InsufficientData`] when there are fewer than 2 users
    /// or items.
    pub fn fit(ratings: &[Rating], config: SvdPpConfig) -> Result<Self, AlsError> {
        if ratings.is_empty() {
            return Err(AlsError::InsufficientData(2));
        }

        // Build sorted, deduplicated index maps
        let mut user_set: Vec<u32> = ratings.iter().map(|r| r.user_id).collect();
        user_set.sort_unstable();
        user_set.dedup();
        let mut item_set: Vec<u32> = ratings.iter().map(|r| r.item_id).collect();
        item_set.sort_unstable();
        item_set.dedup();

        if user_set.len() < 2 || item_set.len() < 2 {
            return Err(AlsError::InsufficientData(2));
        }

        let n_users = user_set.len();
        let n_items = item_set.len();
        let n_factors = config.n_factors;

        let user_index: HashMap<u32, usize> = user_set
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();
        let item_index: HashMap<u32, usize> = item_set
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        // Global mean
        let global_mean = ratings.iter().map(|r| r.rating).sum::<f32>() / ratings.len() as f32;

        // Initialise all parameters to small random values
        let mut rng = Lcg64::new(config.seed);

        let user_bias = vec![0.0_f32; n_users];
        let item_bias = vec![0.0_f32; n_items];
        let mut user_factors: Vec<Vec<f32>> = (0..n_users)
            .map(|_| (0..n_factors).map(|_| rng.next_init()).collect())
            .collect();
        let mut item_factors: Vec<Vec<f32>> = (0..n_items)
            .map(|_| (0..n_factors).map(|_| rng.next_init()).collect())
            .collect();
        let mut implicit_factors: Vec<Vec<f32>> = (0..n_items)
            .map(|_| (0..n_factors).map(|_| rng.next_init()).collect())
            .collect();

        let mut user_bias = user_bias;
        let mut item_bias = item_bias;

        // Convert ratings to internal indices
        let indexed_ratings: Vec<(usize, usize, f32)> = ratings
            .iter()
            .map(|r| (user_index[&r.user_id], item_index[&r.item_id], r.rating))
            .collect();

        // Build N(u): the implicit feedback set for each user
        let mut implicit_feedback: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(u, i, _) in &indexed_ratings {
            implicit_feedback.entry(u).or_default().push(i);
        }
        // Deduplicate and sort each set
        for items in implicit_feedback.values_mut() {
            items.sort_unstable();
            items.dedup();
        }

        let lr = config.learning_rate;
        let reg = config.regularization;
        let bias_reg = config.bias_regularization;

        // SGD training loop
        for _epoch in 0..config.n_epochs {
            for &(u, i, rating) in &indexed_ratings {
                // Compute |N(u)|^{-½}
                let nu: &[usize] = implicit_feedback.get(&u).map(Vec::as_slice).unwrap_or(&[]);
                let sqrt_nu = if nu.is_empty() {
                    1.0_f32
                } else {
                    (nu.len() as f32).sqrt().recip()
                };

                // Sum of implicit factors: Σ_{j∈N(u)} y_j
                let mut implicit_sum = vec![0.0_f32; n_factors];
                for &j in nu {
                    for f in 0..n_factors {
                        implicit_sum[f] += implicit_factors[j][f];
                    }
                }

                // Effective user vector: p_u + sqrt_nu * implicit_sum
                let effective_user: Vec<f32> = (0..n_factors)
                    .map(|f| user_factors[u][f] + sqrt_nu * implicit_sum[f])
                    .collect();

                // Prediction
                let dot: f32 = (0..n_factors)
                    .map(|f| item_factors[i][f] * effective_user[f])
                    .sum();
                let pred = global_mean + user_bias[u] + item_bias[i] + dot;
                let err = rating - pred;

                // Update biases
                user_bias[u] += lr * (err - bias_reg * user_bias[u]);
                item_bias[i] += lr * (err - bias_reg * item_bias[i]);

                // Update user factors p_u and item factors q_i
                for f in 0..n_factors {
                    let puf = user_factors[u][f];
                    let qif = item_factors[i][f];
                    user_factors[u][f] += lr * (err * qif - reg * puf);
                    item_factors[i][f] += lr * (err * effective_user[f] - reg * qif);
                }

                // Update implicit factors y_j for all j ∈ N(u)
                let nu_owned: Vec<usize> = nu.to_vec();
                for j in nu_owned {
                    for f in 0..n_factors {
                        let yjf = implicit_factors[j][f];
                        implicit_factors[j][f] +=
                            lr * (err * sqrt_nu * item_factors[i][f] - reg * yjf);
                    }
                }
            }
        }

        Ok(Self {
            global_mean,
            user_bias,
            item_bias,
            user_factors,
            item_factors,
            implicit_factors,
            user_index,
            item_index,
            user_ids: user_set,
            item_ids: item_set,
            implicit_feedback,
            config,
        })
    }

    // -----------------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------------

    /// Predict the rating for a (user_id, item_id) pair.
    ///
    /// Returns `None` for unknown user or item IDs.
    #[must_use]
    pub fn predict(&self, user_id: u32, item_id: u32) -> Option<f32> {
        let u = *self.user_index.get(&user_id)?;
        let i = *self.item_index.get(&item_id)?;
        Some(self.predict_internal(u, i))
    }

    /// Top-`n` recommendations for `user_id`.
    ///
    /// No items are excluded; call site may filter further.
    /// Returns an empty vector for unknown users.
    #[must_use]
    pub fn recommend(&self, user_id: u32, n: usize) -> Vec<(u32, f32)> {
        let u = match self.user_index.get(&user_id) {
            Some(&u) => u,
            None => return Vec::new(),
        };

        let mut scored: Vec<(u32, f32)> = self
            .item_ids
            .iter()
            .enumerate()
            .map(|(i, &iid)| (iid, self.predict_internal(u, i)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored
    }

    /// RMSE on a held-out test set.  Unknown pairs are skipped.
    #[must_use]
    pub fn rmse(&self, test_ratings: &[Rating]) -> f32 {
        let mut sum_sq = 0.0_f32;
        let mut count = 0_usize;
        for r in test_ratings {
            if let Some(pred) = self.predict(r.user_id, r.item_id) {
                let diff = r.rating - pred;
                sum_sq += diff * diff;
                count += 1;
            }
        }
        if count == 0 {
            return 0.0;
        }
        (sum_sq / count as f32).sqrt()
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Number of users.
    #[must_use]
    pub fn n_users(&self) -> usize {
        self.user_factors.len()
    }

    /// Number of items.
    #[must_use]
    pub fn n_items(&self) -> usize {
        self.item_factors.len()
    }

    /// Number of latent factors.
    #[must_use]
    pub fn n_factors(&self) -> usize {
        self.config.n_factors
    }

    /// User bias vector (internal indices).
    #[must_use]
    pub fn user_bias(&self) -> &[f32] {
        &self.user_bias
    }

    /// Item bias vector (internal indices).
    #[must_use]
    pub fn item_bias(&self) -> &[f32] {
        &self.item_bias
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn predict_internal(&self, u: usize, i: usize) -> f32 {
        let n_factors = self.config.n_factors;
        let nu: &[usize] = self
            .implicit_feedback
            .get(&u)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let sqrt_nu = if nu.is_empty() {
            1.0_f32
        } else {
            (nu.len() as f32).sqrt().recip()
        };

        let mut implicit_sum = vec![0.0_f32; n_factors];
        for &j in nu {
            for f in 0..n_factors {
                implicit_sum[f] += self.implicit_factors[j][f];
            }
        }

        let dot: f32 = (0..n_factors)
            .map(|f| {
                self.item_factors[i][f] * (self.user_factors[u][f] + sqrt_nu * implicit_sum[f])
            })
            .sum();

        self.global_mean + self.user_bias[u] + self.item_bias[i] + dot
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ratings() -> Vec<Rating> {
        vec![
            Rating {
                user_id: 0,
                item_id: 0,
                rating: 5.0,
            },
            Rating {
                user_id: 0,
                item_id: 1,
                rating: 4.0,
            },
            Rating {
                user_id: 0,
                item_id: 2,
                rating: 1.0,
            },
            Rating {
                user_id: 1,
                item_id: 0,
                rating: 4.0,
            },
            Rating {
                user_id: 1,
                item_id: 1,
                rating: 5.0,
            },
            Rating {
                user_id: 1,
                item_id: 3,
                rating: 2.0,
            },
            Rating {
                user_id: 2,
                item_id: 2,
                rating: 5.0,
            },
            Rating {
                user_id: 2,
                item_id: 3,
                rating: 4.0,
            },
            Rating {
                user_id: 2,
                item_id: 4,
                rating: 3.0,
            },
            Rating {
                user_id: 3,
                item_id: 1,
                rating: 2.0,
            },
            Rating {
                user_id: 3,
                item_id: 3,
                rating: 5.0,
            },
            Rating {
                user_id: 3,
                item_id: 4,
                rating: 4.0,
            },
            Rating {
                user_id: 4,
                item_id: 0,
                rating: 3.0,
            },
            Rating {
                user_id: 4,
                item_id: 2,
                rating: 4.0,
            },
            Rating {
                user_id: 4,
                item_id: 4,
                rating: 5.0,
            },
        ]
    }

    fn default_config() -> SvdPpConfig {
        SvdPpConfig {
            n_factors: 4,
            n_epochs: 10,
            ..SvdPpConfig::default()
        }
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_svdpp_fit_basic() {
        let result = SvdPpModel::fit(&sample_ratings(), default_config());
        assert!(result.is_ok(), "SVD++ fit should succeed on valid data");
    }

    #[test]
    fn test_svdpp_dimensions() {
        let model =
            SvdPpModel::fit(&sample_ratings(), default_config()).expect("SVD++ fit should succeed");
        assert_eq!(model.n_users(), 5);
        assert_eq!(model.n_items(), 5);
        assert_eq!(model.n_factors(), 4);
    }

    #[test]
    fn test_svdpp_global_mean_sensible() {
        let model =
            SvdPpModel::fit(&sample_ratings(), default_config()).expect("SVD++ fit should succeed");
        assert!(
            (0.0..=5.0).contains(&model.global_mean),
            "global mean out of rating range: {}",
            model.global_mean
        );
    }

    // -----------------------------------------------------------------------
    // Predict
    // -----------------------------------------------------------------------

    #[test]
    fn test_svdpp_predict_known_pair() {
        let model =
            SvdPpModel::fit(&sample_ratings(), default_config()).expect("SVD++ fit should succeed");
        assert!(model.predict(0, 0).is_some());
        assert!(model.predict(2, 4).is_some());
    }

    #[test]
    fn test_svdpp_predict_unknown_user() {
        let model =
            SvdPpModel::fit(&sample_ratings(), default_config()).expect("SVD++ fit should succeed");
        assert!(model.predict(99, 0).is_none());
    }

    #[test]
    fn test_svdpp_predict_unknown_item() {
        let model =
            SvdPpModel::fit(&sample_ratings(), default_config()).expect("SVD++ fit should succeed");
        assert!(model.predict(0, 99).is_none());
    }

    #[test]
    fn test_svdpp_predict_finite() {
        let model =
            SvdPpModel::fit(&sample_ratings(), default_config()).expect("SVD++ fit should succeed");
        for r in &sample_ratings() {
            let pred = model
                .predict(r.user_id, r.item_id)
                .expect("prediction should exist");
            assert!(pred.is_finite(), "prediction not finite: {pred}");
        }
    }

    // -----------------------------------------------------------------------
    // Recommend
    // -----------------------------------------------------------------------

    #[test]
    fn test_svdpp_recommend_returns_n() {
        let model =
            SvdPpModel::fit(&sample_ratings(), default_config()).expect("SVD++ fit should succeed");
        let recs = model.recommend(0, 3);
        assert!(recs.len() <= 3);
    }

    #[test]
    fn test_svdpp_recommend_sorted() {
        let model =
            SvdPpModel::fit(&sample_ratings(), default_config()).expect("SVD++ fit should succeed");
        let recs = model.recommend(0, 5);
        let scores: Vec<f32> = recs.iter().map(|&(_, s)| s).collect();
        let sorted = scores.windows(2).all(|w| w[0] >= w[1]);
        assert!(sorted, "recommendations not sorted descending");
    }

    #[test]
    fn test_svdpp_recommend_unknown_user() {
        let model =
            SvdPpModel::fit(&sample_ratings(), default_config()).expect("SVD++ fit should succeed");
        assert!(model.recommend(999, 5).is_empty());
    }

    // -----------------------------------------------------------------------
    // RMSE
    // -----------------------------------------------------------------------

    #[test]
    fn test_svdpp_rmse_training_set() {
        let ratings = sample_ratings();
        let config = SvdPpConfig {
            n_factors: 6,
            n_epochs: 30,
            learning_rate: 0.01,
            regularization: 0.01,
            bias_regularization: 0.001,
            ..SvdPpConfig::default()
        };
        let model = SvdPpModel::fit(&ratings, config).expect("SVD++ fit should succeed");
        let rmse = model.rmse(&ratings);
        assert!(rmse.is_finite(), "RMSE must be finite");
        assert!(rmse < 3.0, "training RMSE too high: {rmse}");
    }

    #[test]
    fn test_svdpp_rmse_empty() {
        let model =
            SvdPpModel::fit(&sample_ratings(), default_config()).expect("SVD++ fit should succeed");
        let rmse = model.rmse(&[Rating {
            user_id: 99,
            item_id: 99,
            rating: 3.0,
        }]);
        assert!((rmse).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_svdpp_empty_ratings_error() {
        let result = SvdPpModel::fit(&[], default_config());
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AlsError::InsufficientData(_)));
    }

    #[test]
    fn test_svdpp_single_user_error() {
        let ratings = vec![
            Rating {
                user_id: 0,
                item_id: 0,
                rating: 5.0,
            },
            Rating {
                user_id: 0,
                item_id: 1,
                rating: 3.0,
            },
        ];
        let result = SvdPpModel::fit(&ratings, default_config());
        assert!(matches!(result.unwrap_err(), AlsError::InsufficientData(_)));
    }

    #[test]
    fn test_svdpp_single_item_error() {
        let ratings = vec![
            Rating {
                user_id: 0,
                item_id: 0,
                rating: 5.0,
            },
            Rating {
                user_id: 1,
                item_id: 0,
                rating: 3.0,
            },
        ];
        let result = SvdPpModel::fit(&ratings, default_config());
        assert!(matches!(result.unwrap_err(), AlsError::InsufficientData(_)));
    }

    #[test]
    fn test_svdpp_deterministic() {
        let ratings = sample_ratings();
        let ca = SvdPpConfig {
            seed: 77,
            n_factors: 4,
            n_epochs: 5,
            ..SvdPpConfig::default()
        };
        let cb = SvdPpConfig {
            seed: 77,
            n_factors: 4,
            n_epochs: 5,
            ..SvdPpConfig::default()
        };
        let ma = SvdPpModel::fit(&ratings, ca).expect("SVD++ fit A should succeed");
        let mb = SvdPpModel::fit(&ratings, cb).expect("SVD++ fit B should succeed");
        let pa = ma.predict(0, 0).expect("prediction A should exist");
        let pb = mb.predict(0, 0).expect("prediction B should exist");
        assert!(
            (pa - pb).abs() < 1e-6,
            "SVD++ not deterministic: {pa} vs {pb}"
        );
    }

    #[test]
    fn test_svdpp_config_default() {
        let cfg = SvdPpConfig::default();
        assert_eq!(cfg.n_factors, 10);
        assert_eq!(cfg.n_epochs, 20);
        assert!((cfg.learning_rate - 0.007).abs() < 1e-6);
        assert!((cfg.regularization - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_svdpp_bias_vectors_correct_size() {
        let model =
            SvdPpModel::fit(&sample_ratings(), default_config()).expect("SVD++ fit should succeed");
        assert_eq!(model.user_bias().len(), model.n_users());
        assert_eq!(model.item_bias().len(), model.n_items());
    }
}
