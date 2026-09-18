//! Alternating Least Squares (ALS) Matrix Factorization for collaborative filtering.
//!
//! Decomposes a user-item rating matrix R ≈ U × V^T where:
//! - U = user factor matrix (n_users × n_factors)
//! - V = item factor matrix (n_items × n_factors)
//!
//! Supports both **explicit** (numeric ratings) and **implicit** feedback
//! (binary interaction data with confidence weighting).
//!
//! # Example
//!
//! ```
//! use oximedia_recommend::als::{AlsConfig, AlsModel, Rating};
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
//! let config = AlsConfig { n_factors: 4, n_iterations: 5, ..AlsConfig::default() };
//! let model = AlsModel::fit(&ratings, config).expect("training failed");
//! let pred = model.predict(0, 1);
//! assert!(pred.is_some());
//! ```

use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Public data types
// ---------------------------------------------------------------------------

/// A single sparse rating entry (user × item × value).
#[derive(Debug, Clone, PartialEq)]
pub struct Rating {
    /// External user identifier (arbitrary u32 key).
    pub user_id: u32,
    /// External item identifier (arbitrary u32 key).
    pub item_id: u32,
    /// Rating value.
    ///
    /// For explicit feedback: typically 0.0–5.0.
    /// For implicit feedback: typically 0.0 (no interaction) or 1.0 (interaction).
    pub rating: f32,
}

/// Configuration for ALS training.
#[derive(Debug, Clone)]
pub struct AlsConfig {
    /// Number of latent factors (embedding dimension).  Default: 10.
    pub n_factors: usize,
    /// Number of alternating-least-squares iterations.  Default: 10.
    pub n_iterations: usize,
    /// L2 regularisation coefficient λ.  Default: 0.1.
    pub regularization: f32,
    /// Confidence scaling α used in implicit-feedback mode.
    ///
    /// The confidence for an interaction r is computed as `1 + α·r`.
    /// Default: 40.0.
    pub alpha: f32,
    /// Use implicit-feedback weighting instead of direct rating regression.
    pub use_implicit: bool,
    /// Random seed for factor initialisation (deterministic LC PRNG).
    pub seed: u64,
}

impl Default for AlsConfig {
    fn default() -> Self {
        Self {
            n_factors: 10,
            n_iterations: 10,
            regularization: 0.1,
            alpha: 40.0,
            use_implicit: false,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise during ALS training or inference.
#[derive(Debug, Error)]
pub enum AlsError {
    /// Not enough distinct users or items to train.
    #[error("insufficient data: need at least {0} users and items")]
    InsufficientData(usize),
    /// Factor dimension mismatch during computation.
    #[error("factor dimension mismatch")]
    DimensionMismatch,
    /// A linear system encountered during ALS was singular (or near-singular).
    #[error("singular matrix in ALS solve")]
    SingularMatrix,
}

// ---------------------------------------------------------------------------
// Internal helpers: a tiny seeded pseudo-random number generator
// ---------------------------------------------------------------------------

/// Linear-congruential PRNG state.
struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Return next f32 in (0, 1).
    fn next_f32(&mut self) -> f32 {
        // Knuth multiplicative LCG
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.state >> 33) as u32;
        // Map to (0, 1) by dividing by 2^31
        (bits as f32 + 0.5) / 2_147_483_648.0
    }

    /// Return small non-zero initialisation value centred near zero.
    fn next_init(&mut self) -> f32 {
        (self.next_f32() - 0.5) * 0.1
    }
}

// ---------------------------------------------------------------------------
// ALS Model
// ---------------------------------------------------------------------------

/// A trained ALS matrix-factorisation model.
///
/// After calling [`AlsModel::fit`] the model stores user and item embedding
/// matrices and can answer prediction, recommendation, and similarity queries.
#[derive(Debug)]
pub struct AlsModel {
    /// User embedding matrix: `user_factors[u][f]`.
    pub user_factors: Vec<Vec<f32>>,
    /// Item embedding matrix: `item_factors[i][f]`.
    pub item_factors: Vec<Vec<f32>>,
    /// Mapping from external user_id → row index.
    user_index: HashMap<u32, usize>,
    /// Mapping from external item_id → row index.
    item_index: HashMap<u32, usize>,
    /// Reverse mapping: row index → external user_id.
    user_ids: Vec<u32>,
    /// Reverse mapping: row index → external item_id.
    item_ids: Vec<u32>,
    /// Ratings per user (for exclude_rated lookups): user_row → set of item_row.
    user_rated: Vec<Vec<usize>>,
    /// Configuration used during training.
    config: AlsConfig,
}

impl AlsModel {
    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    /// Train an ALS model from a slice of [`Rating`]s.
    ///
    /// # Errors
    ///
    /// Returns [`AlsError::InsufficientData`] when there are fewer than 2 users
    /// or items, or [`AlsError::SingularMatrix`] when a linear solve fails.
    pub fn fit(ratings: &[Rating], config: AlsConfig) -> Result<Self, AlsError> {
        if ratings.is_empty() {
            return Err(AlsError::InsufficientData(2));
        }

        // Build index maps
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

        // Initialise factor matrices with small random values
        let mut rng = Lcg64::new(config.seed);
        let mut user_factors: Vec<Vec<f32>> = (0..n_users)
            .map(|_| (0..n_factors).map(|_| rng.next_init()).collect())
            .collect();
        let mut item_factors: Vec<Vec<f32>> = (0..n_items)
            .map(|_| (0..n_factors).map(|_| rng.next_init()).collect())
            .collect();

        // Build per-user and per-item rating lists (internal indices)
        let mut user_items: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_users];
        let mut item_users: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_items];
        for r in ratings {
            let ui = user_index[&r.user_id];
            let ii = item_index[&r.item_id];
            let effective_rating = if config.use_implicit {
                // binarise: any positive value → 1.0
                if r.rating > 0.0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                r.rating
            };
            user_items[ui].push((ii, effective_rating));
            item_users[ii].push((ui, effective_rating));
        }

        // Build user_rated: user → sorted item indices rated
        let user_rated: Vec<Vec<usize>> = user_items
            .iter()
            .map(|v| {
                let mut indices: Vec<usize> = v.iter().map(|&(i, _)| i).collect();
                indices.sort_unstable();
                indices
            })
            .collect();

        // ALS iterations
        for _iter in 0..config.n_iterations {
            // Fix item factors, solve for user factors
            for u in 0..n_users {
                let ratings_u = if config.use_implicit {
                    Self::build_implicit_ratings(&user_items[u], &config)
                } else {
                    user_items[u].clone()
                };
                user_factors[u] = Self::solve_als_row(
                    &item_factors,
                    &ratings_u,
                    config.regularization,
                    n_factors,
                    config.use_implicit,
                    n_items,
                )?;
            }

            // Fix user factors, solve for item factors
            for i in 0..n_items {
                let ratings_i = if config.use_implicit {
                    Self::build_implicit_ratings(&item_users[i], &config)
                } else {
                    item_users[i].clone()
                };
                item_factors[i] = Self::solve_als_row(
                    &user_factors,
                    &ratings_i,
                    config.regularization,
                    n_factors,
                    config.use_implicit,
                    n_users,
                )?;
            }
        }

        Ok(Self {
            user_factors,
            item_factors,
            user_index,
            item_index,
            user_ids: user_set,
            item_ids: item_set,
            user_rated,
            config,
        })
    }

    // -----------------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------------

    /// Predict the rating for a (user_id, item_id) pair.
    ///
    /// Returns `None` if either ID was not seen during training.
    #[must_use]
    pub fn predict(&self, user_id: u32, item_id: u32) -> Option<f32> {
        let u = *self.user_index.get(&user_id)?;
        let i = *self.item_index.get(&item_id)?;
        Some(Self::dot_product(
            &self.user_factors[u],
            &self.item_factors[i],
        ))
    }

    /// Return the top-`n` recommended items for `user_id`, excluding any item
    /// whose `item_id` appears in `exclude_rated`.
    ///
    /// Returns an empty vector for unknown users.
    #[must_use]
    pub fn recommend(&self, user_id: u32, n: usize, exclude_rated: &[u32]) -> Vec<(u32, f32)> {
        let u = match self.user_index.get(&user_id) {
            Some(&u) => u,
            None => return Vec::new(),
        };

        let exclude_set: std::collections::HashSet<u32> = exclude_rated.iter().copied().collect();

        let user_vec = &self.user_factors[u];
        let mut scored: Vec<(u32, f32)> = self
            .item_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &iid)| {
                if exclude_set.contains(&iid) {
                    return None;
                }
                let score = Self::dot_product(user_vec, &self.item_factors[i]);
                Some((iid, score))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored
    }

    /// Return the top-`n` items most similar to `item_id` in factor space
    /// (cosine similarity).  The query item itself is excluded from results.
    ///
    /// Returns an empty vector for unknown items.
    #[must_use]
    pub fn similar_items(&self, item_id: u32, n: usize) -> Vec<(u32, f32)> {
        let qi = match self.item_index.get(&item_id) {
            Some(&i) => i,
            None => return Vec::new(),
        };
        let query = &self.item_factors[qi];
        let query_norm = Self::l2_norm(query);
        if query_norm < f32::EPSILON {
            return Vec::new();
        }

        let mut scored: Vec<(u32, f32)> = self
            .item_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &iid)| {
                if i == qi {
                    return None;
                }
                let candidate = &self.item_factors[i];
                let cand_norm = Self::l2_norm(candidate);
                if cand_norm < f32::EPSILON {
                    return None;
                }
                let cos = Self::dot_product(query, candidate) / (query_norm * cand_norm);
                Some((iid, cos))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored
    }

    /// Return the top-`n` users most similar to `user_id` in factor space
    /// (cosine similarity).  The query user itself is excluded from results.
    ///
    /// Returns an empty vector for unknown users.
    #[must_use]
    pub fn similar_users(&self, user_id: u32, n: usize) -> Vec<(u32, f32)> {
        let qu = match self.user_index.get(&user_id) {
            Some(&u) => u,
            None => return Vec::new(),
        };
        let query = &self.user_factors[qu];
        let query_norm = Self::l2_norm(query);
        if query_norm < f32::EPSILON {
            return Vec::new();
        }

        let mut scored: Vec<(u32, f32)> = self
            .user_ids
            .iter()
            .enumerate()
            .filter_map(|(u, &uid)| {
                if u == qu {
                    return None;
                }
                let candidate = &self.user_factors[u];
                let cand_norm = Self::l2_norm(candidate);
                if cand_norm < f32::EPSILON {
                    return None;
                }
                let cos = Self::dot_product(query, candidate) / (query_norm * cand_norm);
                Some((uid, cos))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored
    }

    /// Compute Root Mean Squared Error on a held-out test set.
    ///
    /// Ratings for unknown (user, item) pairs are skipped.
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

    /// Number of users in the model.
    #[must_use]
    pub fn n_users(&self) -> usize {
        self.user_factors.len()
    }

    /// Number of items in the model.
    #[must_use]
    pub fn n_items(&self) -> usize {
        self.item_factors.len()
    }

    /// Number of latent factors.
    #[must_use]
    pub fn n_factors(&self) -> usize {
        self.config.n_factors
    }

    /// The user index mapping (external id → internal row).
    #[must_use]
    pub fn user_index(&self) -> &HashMap<u32, usize> {
        &self.user_index
    }

    /// The item index mapping (external id → internal row).
    #[must_use]
    pub fn item_index(&self) -> &HashMap<u32, usize> {
        &self.item_index
    }

    /// The rated item indices for each user (internal indices), sorted.
    #[must_use]
    pub fn user_rated(&self) -> &[Vec<usize>] {
        &self.user_rated
    }

    // -----------------------------------------------------------------------
    // Internal linear algebra helpers
    // -----------------------------------------------------------------------

    /// Solve the weighted least-squares normal equation for one row.
    ///
    /// Computes:
    /// ```text
    ///   A = X_S^T C_u X_S + λI
    ///   b = X_S^T C_u p_u
    ///   → u = A⁻¹ b
    /// ```
    ///
    /// where `X_S` is the sub-matrix of `fixed_factors` for items that the
    /// user has interacted with, `C_u` is a diagonal confidence matrix (all
    /// ones for explicit, `1 + α·r` for implicit), and `p_u` is the preference
    /// vector (equals the rating for explicit, 1.0 for implicit).
    ///
    /// Uses Gaussian elimination with partial pivoting.
    fn solve_als_row(
        fixed_factors: &[Vec<f32>],
        ratings_for_row: &[(usize, f32)],
        regularization: f32,
        n_factors: usize,
        use_implicit: bool,
        n_fixed: usize,
    ) -> Result<Vec<f32>, AlsError> {
        // Build the (n_factors × n_factors) matrix A = X^T W X + λI
        // and the right-hand side b = X^T W p.
        let mut a = vec![0.0_f32; n_factors * n_factors];
        let mut b = vec![0.0_f32; n_factors];

        if use_implicit {
            // Full pass over all items (X^T C X) — O(n_items × n_factors²)
            // First accumulate the "uniform" part C_0 = 1 for all items.
            for j in 0..n_fixed {
                let xj = &fixed_factors[j];
                for f in 0..n_factors {
                    for g in 0..n_factors {
                        a[f * n_factors + g] += xj[f] * xj[g];
                    }
                }
            }
            // Then add the extra confidence for observed interactions.
            for &(j, r) in ratings_for_row {
                let c_extra = r; // c_ui - 1 = alpha * r_ui (r already scaled)
                let xj = &fixed_factors[j];
                for f in 0..n_factors {
                    for g in 0..n_factors {
                        a[f * n_factors + g] += c_extra * xj[f] * xj[g];
                    }
                    // Preference p_ui = 1 for any observed item.
                    b[f] += (1.0 + c_extra) * xj[f];
                }
            }
        } else {
            // Explicit: only iterate over observed ratings (sparse).
            for &(j, r) in ratings_for_row {
                let xj = &fixed_factors[j];
                for f in 0..n_factors {
                    for g in 0..n_factors {
                        a[f * n_factors + g] += xj[f] * xj[g];
                    }
                    b[f] += r * xj[f];
                }
            }
        }

        // Regularisation: A += λI
        for f in 0..n_factors {
            a[f * n_factors + f] += regularization;
        }

        // Solve A x = b via Gaussian elimination with partial pivoting.
        Self::gaussian_solve(n_factors, &mut a, &mut b)
    }

    /// Gaussian elimination with partial pivoting to solve A x = b in-place.
    ///
    /// On success returns the solution vector x.
    fn gaussian_solve(n: usize, a: &mut Vec<f32>, b: &mut Vec<f32>) -> Result<Vec<f32>, AlsError> {
        // Forward elimination
        for col in 0..n {
            // Find pivot
            let pivot_row = (col..n)
                .max_by(|&r1, &r2| {
                    a[r1 * n + col]
                        .abs()
                        .partial_cmp(&a[r2 * n + col].abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .ok_or(AlsError::SingularMatrix)?;

            if a[pivot_row * n + col].abs() < 1e-12 {
                return Err(AlsError::SingularMatrix);
            }

            // Swap rows
            if pivot_row != col {
                for k in 0..n {
                    a.swap(col * n + k, pivot_row * n + k);
                }
                b.swap(col, pivot_row);
            }

            let pivot_val = a[col * n + col];
            for row in (col + 1)..n {
                let factor = a[row * n + col] / pivot_val;
                for k in col..n {
                    let sub = factor * a[col * n + k];
                    a[row * n + k] -= sub;
                }
                let sub_b = factor * b[col];
                b[row] -= sub_b;
            }
        }

        // Back substitution
        let mut x = vec![0.0_f32; n];
        for i in (0..n).rev() {
            let mut sum = b[i];
            for j in (i + 1)..n {
                sum -= a[i * n + j] * x[j];
            }
            if a[i * n + i].abs() < 1e-12 {
                return Err(AlsError::SingularMatrix);
            }
            x[i] = sum / a[i * n + i];
        }

        Ok(x)
    }

    /// Scale implicit ratings: `(confidence - 1) = alpha * r`.
    fn build_implicit_ratings(raw: &[(usize, f32)], config: &AlsConfig) -> Vec<(usize, f32)> {
        raw.iter()
            .filter(|&&(_, r)| r > 0.0)
            .map(|&(i, r)| (i, config.alpha * r))
            .collect()
    }

    /// Dot product of two equal-length slices.
    #[inline]
    fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// L2 norm of a slice.
    #[inline]
    fn l2_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small but well-conditioned 5-user × 5-item rating matrix.
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

    fn default_config() -> AlsConfig {
        AlsConfig {
            n_factors: 4,
            n_iterations: 10,
            regularization: 0.1,
            ..AlsConfig::default()
        }
    }

    // -----------------------------------------------------------------------
    // Construction & dimension checks
    // -----------------------------------------------------------------------

    #[test]
    fn test_als_fit_basic() {
        let model = AlsModel::fit(&sample_ratings(), default_config());
        assert!(model.is_ok(), "ALS fit should succeed on valid data");
    }

    #[test]
    fn test_als_dimensions() {
        let model =
            AlsModel::fit(&sample_ratings(), default_config()).expect("ALS fit should succeed");
        assert_eq!(model.n_users(), 5);
        assert_eq!(model.n_items(), 5);
        assert_eq!(model.n_factors(), 4);
        assert_eq!(model.user_factors.len(), 5);
        assert_eq!(model.item_factors.len(), 5);
        assert!(model.user_factors.iter().all(|v| v.len() == 4));
        assert!(model.item_factors.iter().all(|v| v.len() == 4));
    }

    // -----------------------------------------------------------------------
    // Predict
    // -----------------------------------------------------------------------

    #[test]
    fn test_als_predict_known_pair() {
        let model =
            AlsModel::fit(&sample_ratings(), default_config()).expect("ALS fit should succeed");
        // All (user, item) pairs that appear in training should return Some.
        assert!(model.predict(0, 0).is_some());
        assert!(model.predict(2, 4).is_some());
        assert!(model.predict(4, 2).is_some());
    }

    #[test]
    fn test_als_predict_unknown_user() {
        let model =
            AlsModel::fit(&sample_ratings(), default_config()).expect("ALS fit should succeed");
        assert!(model.predict(99, 0).is_none());
    }

    #[test]
    fn test_als_predict_unknown_item() {
        let model =
            AlsModel::fit(&sample_ratings(), default_config()).expect("ALS fit should succeed");
        assert!(model.predict(0, 99).is_none());
    }

    #[test]
    fn test_als_predict_reasonable_range() {
        // After convergence predictions should be in a sane range (not NaN/inf).
        let config = AlsConfig {
            n_factors: 4,
            n_iterations: 20,
            regularization: 0.1,
            ..AlsConfig::default()
        };
        let model = AlsModel::fit(&sample_ratings(), config).expect("ALS fit should succeed");
        let pred = model.predict(0, 0).expect("prediction should exist");
        assert!(pred.is_finite(), "prediction must be finite: {pred}");
    }

    // -----------------------------------------------------------------------
    // Recommend top-N
    // -----------------------------------------------------------------------

    #[test]
    fn test_als_recommend_returns_n() {
        let model =
            AlsModel::fit(&sample_ratings(), default_config()).expect("ALS fit should succeed");
        let recs = model.recommend(0, 3, &[]);
        assert!(recs.len() <= 3);
    }

    #[test]
    fn test_als_recommend_excludes_rated() {
        let model =
            AlsModel::fit(&sample_ratings(), default_config()).expect("ALS fit should succeed");
        // User 0 rated items 0, 1, 2
        let recs = model.recommend(0, 5, &[0, 1, 2]);
        let rec_ids: Vec<u32> = recs.iter().map(|&(id, _)| id).collect();
        assert!(!rec_ids.contains(&0));
        assert!(!rec_ids.contains(&1));
        assert!(!rec_ids.contains(&2));
    }

    #[test]
    fn test_als_recommend_unknown_user() {
        let model =
            AlsModel::fit(&sample_ratings(), default_config()).expect("ALS fit should succeed");
        assert!(model.recommend(999, 5, &[]).is_empty());
    }

    #[test]
    fn test_als_recommend_sorted_by_score() {
        let model =
            AlsModel::fit(&sample_ratings(), default_config()).expect("ALS fit should succeed");
        let recs = model.recommend(0, 5, &[]);
        let scores: Vec<f32> = recs.iter().map(|&(_, s)| s).collect();
        let is_sorted = scores.windows(2).all(|w| w[0] >= w[1]);
        assert!(is_sorted, "recommendations must be sorted descending");
    }

    // -----------------------------------------------------------------------
    // Similar items
    // -----------------------------------------------------------------------

    #[test]
    fn test_similar_items_returns_n() {
        let model =
            AlsModel::fit(&sample_ratings(), default_config()).expect("ALS fit should succeed");
        let sims = model.similar_items(0, 3);
        assert!(sims.len() <= 3);
    }

    #[test]
    fn test_similar_items_excludes_query() {
        let model =
            AlsModel::fit(&sample_ratings(), default_config()).expect("ALS fit should succeed");
        let sims = model.similar_items(0, 5);
        let ids: Vec<u32> = sims.iter().map(|&(id, _)| id).collect();
        assert!(!ids.contains(&0), "query item must not appear in results");
    }

    #[test]
    fn test_similar_items_unknown_item() {
        let model =
            AlsModel::fit(&sample_ratings(), default_config()).expect("ALS fit should succeed");
        assert!(model.similar_items(999, 5).is_empty());
    }

    #[test]
    fn test_similar_items_cosine_range() {
        let model =
            AlsModel::fit(&sample_ratings(), default_config()).expect("ALS fit should succeed");
        let sims = model.similar_items(0, 5);
        for &(_, cos) in &sims {
            assert!(
                (-1.001..=1.001).contains(&cos),
                "cosine similarity out of range: {cos}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Similar users
    // -----------------------------------------------------------------------

    #[test]
    fn test_similar_users_returns_n() {
        let model =
            AlsModel::fit(&sample_ratings(), default_config()).expect("ALS fit should succeed");
        let sims = model.similar_users(0, 3);
        assert!(sims.len() <= 3);
    }

    #[test]
    fn test_similar_users_excludes_query() {
        let model =
            AlsModel::fit(&sample_ratings(), default_config()).expect("ALS fit should succeed");
        let sims = model.similar_users(0, 5);
        let ids: Vec<u32> = sims.iter().map(|&(id, _)| id).collect();
        assert!(!ids.contains(&0));
    }

    #[test]
    fn test_similar_users_unknown_user() {
        let model =
            AlsModel::fit(&sample_ratings(), default_config()).expect("ALS fit should succeed");
        assert!(model.similar_users(999, 5).is_empty());
    }

    // -----------------------------------------------------------------------
    // RMSE
    // -----------------------------------------------------------------------

    #[test]
    fn test_als_rmse_on_training_set_low() {
        let ratings = sample_ratings();
        let config = AlsConfig {
            n_factors: 5,
            n_iterations: 20,
            regularization: 0.01,
            ..AlsConfig::default()
        };
        let model = AlsModel::fit(&ratings, config).expect("ALS fit should succeed");
        let rmse = model.rmse(&ratings);
        // With sufficient factors and low regularisation training RMSE < 2.0
        assert!(rmse < 2.0, "training RMSE too high: {rmse}");
    }

    #[test]
    fn test_als_rmse_empty() {
        let model =
            AlsModel::fit(&sample_ratings(), default_config()).expect("ALS fit should succeed");
        // No overlap between test_ratings and training → all skipped → 0.0
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
    fn test_als_empty_ratings_error() {
        let result = AlsModel::fit(&[], default_config());
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AlsError::InsufficientData(_)));
    }

    #[test]
    fn test_als_single_user_error() {
        // All ratings from the same user
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
        let result = AlsModel::fit(&ratings, default_config());
        assert!(matches!(result.unwrap_err(), AlsError::InsufficientData(_)));
    }

    #[test]
    fn test_als_single_item_error() {
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
        let result = AlsModel::fit(&ratings, default_config());
        assert!(matches!(result.unwrap_err(), AlsError::InsufficientData(_)));
    }

    #[test]
    fn test_als_implicit_feedback() {
        let ratings = vec![
            Rating {
                user_id: 0,
                item_id: 0,
                rating: 1.0,
            },
            Rating {
                user_id: 0,
                item_id: 1,
                rating: 1.0,
            },
            Rating {
                user_id: 1,
                item_id: 1,
                rating: 1.0,
            },
            Rating {
                user_id: 1,
                item_id: 2,
                rating: 1.0,
            },
            Rating {
                user_id: 2,
                item_id: 0,
                rating: 1.0,
            },
            Rating {
                user_id: 2,
                item_id: 2,
                rating: 1.0,
            },
        ];
        let config = AlsConfig {
            n_factors: 3,
            n_iterations: 5,
            use_implicit: true,
            ..AlsConfig::default()
        };
        let result = AlsModel::fit(&ratings, config);
        assert!(result.is_ok());
        let model = result.expect("ALS fit should succeed");
        assert!(model.predict(0, 0).is_some());
    }

    #[test]
    fn test_als_deterministic_with_seed() {
        let ratings = sample_ratings();
        let config_a = AlsConfig {
            seed: 123,
            n_factors: 4,
            n_iterations: 5,
            ..AlsConfig::default()
        };
        let config_b = AlsConfig {
            seed: 123,
            n_factors: 4,
            n_iterations: 5,
            ..AlsConfig::default()
        };
        let model_a = AlsModel::fit(&ratings, config_a).expect("ALS fit A should succeed");
        let model_b = AlsModel::fit(&ratings, config_b).expect("ALS fit B should succeed");
        // Predictions must be identical given same seed.
        let p_a = model_a.predict(0, 0).expect("prediction A should exist");
        let p_b = model_b.predict(0, 0).expect("prediction B should exist");
        assert!(
            (p_a - p_b).abs() < 1e-6,
            "models not deterministic: {p_a} vs {p_b}"
        );
    }

    #[test]
    fn test_als_config_default() {
        let cfg = AlsConfig::default();
        assert_eq!(cfg.n_factors, 10);
        assert_eq!(cfg.n_iterations, 10);
        assert!((cfg.regularization - 0.1).abs() < f32::EPSILON);
        assert!((cfg.alpha - 40.0).abs() < f32::EPSILON);
        assert!(!cfg.use_implicit);
    }

    #[test]
    fn test_als_error_display() {
        let e = AlsError::InsufficientData(2);
        assert!(e.to_string().contains("insufficient data"));
        let e2 = AlsError::SingularMatrix;
        assert!(e2.to_string().contains("singular"));
        let e3 = AlsError::DimensionMismatch;
        assert!(e3.to_string().contains("dimension"));
    }
}
