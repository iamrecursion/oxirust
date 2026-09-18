//! Collaborative filtering utilities: user-item matrix operations and similarity metrics.
//!
//! This module provides standalone collaborative filtering helpers distinct from
//! the `collaborative` directory module.  It focuses on lightweight similarity
//! computations and neighbourhood-based prediction that can be reused across the
//! crate without pulling in heavy SVD machinery.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// User-Item rating entry
// ---------------------------------------------------------------------------

/// A single user-item rating tuple.
#[derive(Debug, Clone, PartialEq)]
pub struct RatingEntry {
    /// User identifier.
    pub user_id: Uuid,
    /// Item identifier.
    pub item_id: Uuid,
    /// Rating value (0.0–5.0 scale).
    pub rating: f32,
}

impl RatingEntry {
    /// Create a new rating entry.
    #[must_use]
    pub fn new(user_id: Uuid, item_id: Uuid, rating: f32) -> Self {
        Self {
            user_id,
            item_id,
            rating,
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse user-item matrix
// ---------------------------------------------------------------------------

/// Sparse representation of the user-item rating matrix.
///
/// Ratings are stored as `user_id → (item_id → rating)` for fast per-user
/// lookup, and additionally as `item_id → (user_id → rating)` for fast
/// per-item lookup.
#[derive(Debug, Clone, Default)]
pub struct SparseMatrix {
    by_user: HashMap<Uuid, HashMap<Uuid, f32>>,
    by_item: HashMap<Uuid, HashMap<Uuid, f32>>,
}

impl SparseMatrix {
    /// Create an empty sparse matrix.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or update a rating.
    pub fn insert(&mut self, user_id: Uuid, item_id: Uuid, rating: f32) {
        self.by_user
            .entry(user_id)
            .or_default()
            .insert(item_id, rating);
        self.by_item
            .entry(item_id)
            .or_default()
            .insert(user_id, rating);
    }

    /// Retrieve a rating if it exists.
    #[must_use]
    pub fn get(&self, user_id: Uuid, item_id: Uuid) -> Option<f32> {
        self.by_user.get(&user_id)?.get(&item_id).copied()
    }

    /// Return all items rated by `user_id`.
    #[must_use]
    pub fn items_for_user(&self, user_id: Uuid) -> Vec<(Uuid, f32)> {
        self.by_user
            .get(&user_id)
            .map(|m| m.iter().map(|(id, &r)| (*id, r)).collect())
            .unwrap_or_default()
    }

    /// Return all users who rated `item_id`.
    #[must_use]
    pub fn users_for_item(&self, item_id: Uuid) -> Vec<(Uuid, f32)> {
        self.by_item
            .get(&item_id)
            .map(|m| m.iter().map(|(id, &r)| (*id, r)).collect())
            .unwrap_or_default()
    }

    /// Number of ratings stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_user
            .values()
            .map(std::collections::HashMap::len)
            .sum()
    }

    /// Returns `true` when the matrix contains no ratings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// All distinct user IDs in the matrix.
    #[must_use]
    pub fn user_ids(&self) -> Vec<Uuid> {
        self.by_user.keys().copied().collect()
    }

    /// All distinct item IDs in the matrix.
    #[must_use]
    pub fn item_ids(&self) -> Vec<Uuid> {
        self.by_item.keys().copied().collect()
    }
}

// ---------------------------------------------------------------------------
// Similarity metrics
// ---------------------------------------------------------------------------

/// Compute the **Pearson correlation coefficient** between two rating maps.
///
/// Returns `None` when there are fewer than two co-rated items or when
/// variance is zero.
#[must_use]
pub fn pearson_similarity(a: &HashMap<Uuid, f32>, b: &HashMap<Uuid, f32>) -> Option<f32> {
    // Collect co-rated items.
    let common: Vec<Uuid> = a.keys().filter(|id| b.contains_key(*id)).copied().collect();
    if common.len() < 2 {
        return None;
    }

    let n = common.len() as f32;
    let sum_a: f32 = common.iter().map(|id| a[id]).sum();
    let sum_b: f32 = common.iter().map(|id| b[id]).sum();
    let mean_a = sum_a / n;
    let mean_b = sum_b / n;

    let mut num = 0.0_f32;
    let mut den_a = 0.0_f32;
    let mut den_b = 0.0_f32;
    for id in &common {
        let da = a[id] - mean_a;
        let db = b[id] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }

    let denom = (den_a * den_b).sqrt();
    if denom < f32::EPSILON {
        None
    } else {
        Some(num / denom)
    }
}

/// Compute the **cosine similarity** between two rating vectors represented
/// as sparse maps.
#[must_use]
pub fn cosine_similarity(a: &HashMap<Uuid, f32>, b: &HashMap<Uuid, f32>) -> f32 {
    let dot: f32 = a
        .iter()
        .filter_map(|(id, &va)| b.get(id).map(|&vb| va * vb))
        .sum();
    let norm_a: f32 = a.values().map(|v| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = b.values().map(|v| v * v).sum::<f32>().sqrt();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

// ---------------------------------------------------------------------------
// Neighbourhood-based prediction
// ---------------------------------------------------------------------------

/// Parameters governing neighbourhood selection.
#[derive(Debug, Clone)]
pub struct NeighbourhoodConfig {
    /// Maximum number of neighbours to consider.
    pub k: usize,
    /// Minimum similarity threshold (neighbours below this are ignored).
    pub min_similarity: f32,
}

impl Default for NeighbourhoodConfig {
    fn default() -> Self {
        Self {
            k: 10,
            min_similarity: 0.1,
        }
    }
}

/// Predict the rating that `user_id` would assign to `item_id` using
/// user-based collaborative filtering.
///
/// Returns `None` if there are no valid neighbours.
#[must_use]
pub fn predict_rating(
    matrix: &SparseMatrix,
    user_id: Uuid,
    item_id: Uuid,
    config: &NeighbourhoodConfig,
) -> Option<f32> {
    let user_ratings = matrix.by_user.get(&user_id)?;

    // Collect neighbours who also rated `item_id`.
    let mut neighbours: Vec<(Uuid, f32)> = matrix
        .users_for_item(item_id)
        .into_iter()
        .filter(|(uid, _)| *uid != user_id)
        .filter_map(|(uid, _)| {
            let other_ratings = matrix.by_user.get(&uid)?;
            let sim = pearson_similarity(user_ratings, other_ratings)?;
            if sim >= config.min_similarity {
                Some((uid, sim))
            } else {
                None
            }
        })
        .collect();

    if neighbours.is_empty() {
        return None;
    }

    // Sort descending by similarity, keep top-k.
    neighbours.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    neighbours.truncate(config.k);

    // Weighted average.
    let mut num = 0.0_f32;
    let mut den = 0.0_f32;
    for (uid, sim) in &neighbours {
        if let Some(r) = matrix.get(*uid, item_id) {
            num += sim * r;
            den += sim.abs();
        }
    }

    if den < f32::EPSILON {
        None
    } else {
        Some((num / den).clamp(0.0, 5.0))
    }
}

// ---------------------------------------------------------------------------
// Top-N recommendation generation
// ---------------------------------------------------------------------------

/// Return the top-`n` items predicted for `user_id` from unseen items in
/// `candidate_items`.
#[must_use]
pub fn top_n_for_user(
    matrix: &SparseMatrix,
    user_id: Uuid,
    candidate_items: &[Uuid],
    n: usize,
    config: &NeighbourhoodConfig,
) -> Vec<(Uuid, f32)> {
    let seen: std::collections::HashSet<Uuid> = matrix
        .items_for_user(user_id)
        .into_iter()
        .map(|(id, _)| id)
        .collect();

    let mut scored: Vec<(Uuid, f32)> = candidate_items
        .iter()
        .filter(|id| !seen.contains(id))
        .filter_map(|id| predict_rating(matrix, user_id, *id, config).map(|score| (*id, score)))
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(n);
    scored
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uid() -> Uuid {
        Uuid::new_v4()
    }

    // --- RatingEntry ---

    #[test]
    fn test_rating_entry_new() {
        let u = uid();
        let i = uid();
        let e = RatingEntry::new(u, i, 3.5);
        assert_eq!(e.user_id, u);
        assert_eq!(e.item_id, i);
        assert!((e.rating - 3.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_rating_entry_clone() {
        let e = RatingEntry::new(uid(), uid(), 2.0);
        let e2 = e.clone();
        assert_eq!(e, e2);
    }

    // --- SparseMatrix ---

    #[test]
    fn test_sparse_matrix_insert_get() {
        let mut m = SparseMatrix::new();
        let u = uid();
        let i = uid();
        m.insert(u, i, 4.0);
        assert_eq!(m.get(u, i), Some(4.0));
    }

    #[test]
    fn test_sparse_matrix_missing() {
        let m = SparseMatrix::new();
        assert_eq!(m.get(uid(), uid()), None);
    }

    #[test]
    fn test_sparse_matrix_len_and_is_empty() {
        let mut m = SparseMatrix::new();
        assert!(m.is_empty());
        m.insert(uid(), uid(), 1.0);
        assert_eq!(m.len(), 1);
        assert!(!m.is_empty());
    }

    #[test]
    fn test_items_for_user() {
        let mut m = SparseMatrix::new();
        let u = uid();
        let i1 = uid();
        let i2 = uid();
        m.insert(u, i1, 3.0);
        m.insert(u, i2, 4.5);
        let items = m.items_for_user(u);
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_users_for_item() {
        let mut m = SparseMatrix::new();
        let item = uid();
        let u1 = uid();
        let u2 = uid();
        m.insert(u1, item, 2.0);
        m.insert(u2, item, 5.0);
        assert_eq!(m.users_for_item(item).len(), 2);
    }

    #[test]
    fn test_user_ids_and_item_ids() {
        let mut m = SparseMatrix::new();
        let u = uid();
        let i = uid();
        m.insert(u, i, 1.0);
        assert!(m.user_ids().contains(&u));
        assert!(m.item_ids().contains(&i));
    }

    // --- Similarity metrics ---

    #[test]
    fn test_cosine_similarity_identical() {
        let id1 = uid();
        let id2 = uid();
        let mut map: HashMap<Uuid, f32> = HashMap::new();
        map.insert(id1, 3.0);
        map.insert(id2, 4.0);
        let sim = cosine_similarity(&map, &map);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let id1 = uid();
        let id2 = uid();
        let mut a: HashMap<Uuid, f32> = HashMap::new();
        a.insert(id1, 1.0);
        let mut b: HashMap<Uuid, f32> = HashMap::new();
        b.insert(id2, 1.0);
        assert!((cosine_similarity(&a, &b)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pearson_similarity_perfect() {
        let ids: Vec<Uuid> = (0..5).map(|_| uid()).collect();
        let mut a: HashMap<Uuid, f32> = HashMap::new();
        let mut b: HashMap<Uuid, f32> = HashMap::new();
        for (k, id) in ids.iter().enumerate() {
            a.insert(*id, k as f32);
            b.insert(*id, k as f32 * 2.0);
        }
        let sim = pearson_similarity(&a, &b).expect("expected Some");
        assert!((sim - 1.0).abs() < 1e-4, "sim={sim}");
    }

    #[test]
    fn test_pearson_similarity_none_when_insufficient() {
        let id = uid();
        let mut a: HashMap<Uuid, f32> = HashMap::new();
        a.insert(id, 3.0);
        let mut b: HashMap<Uuid, f32> = HashMap::new();
        b.insert(id, 4.0);
        // Only one common item – should return None.
        assert!(pearson_similarity(&a, &b).is_none());
    }

    // --- Prediction ---

    #[test]
    fn test_predict_rating_basic() {
        let mut m = SparseMatrix::new();
        let target_user = uid();
        let item_a = uid();
        let item_b = uid();

        // target user has rated item_a
        m.insert(target_user, item_a, 4.0);

        // neighbour user rated both items similarly to target
        let neighbour = uid();
        m.insert(neighbour, item_a, 3.8);
        m.insert(neighbour, item_b, 5.0);

        // Add a second item to target user's ratings so pearson can compute
        let item_c = uid();
        m.insert(target_user, item_c, 3.0);
        m.insert(neighbour, item_c, 2.9);

        let cfg = NeighbourhoodConfig {
            k: 5,
            min_similarity: 0.0,
        };
        let pred = predict_rating(&m, target_user, item_b, &cfg);
        assert!(pred.is_some());
        let val = pred.expect("should succeed in test");
        assert!((0.0..=5.0).contains(&val));
    }

    #[test]
    fn test_top_n_for_user_excludes_seen() {
        let mut m = SparseMatrix::new();
        let u = uid();
        let seen_item = uid();
        m.insert(u, seen_item, 5.0);

        let cfg = NeighbourhoodConfig::default();
        let result = top_n_for_user(&m, u, &[seen_item], 10, &cfg);
        // seen_item should be excluded
        assert!(result.is_empty());
    }

    #[test]
    fn test_neighbourhood_config_default() {
        let cfg = NeighbourhoodConfig::default();
        assert_eq!(cfg.k, 10);
        assert!((cfg.min_similarity - 0.1).abs() < f32::EPSILON);
    }
}
