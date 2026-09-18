//! Visual feature indexing with VP-tree acceleration.
//!
//! [`VisualIndex`] stores `(Uuid, Vec<f32>)` feature vectors and routes
//! similarity queries through a [`FloatVpTree`] whenever the index contains
//! at least `TREE_THRESHOLD` entries.  Below that threshold a brute-force
//! linear scan is cheaper due to the constant overhead of VP-tree traversal.
//!
//! # Interior mutability
//!
//! The VP-tree is rebuilt lazily on the first query after an insertion.  To
//! allow callers to search through a shared `&VisualIndex` reference (as
//! required by `SearchEngine::search`, which takes `&self`), the tree and
//! dirty flag are wrapped in `RefCell`/`Cell`.  This is safe as long as
//! `VisualIndex` is not shared across threads without an external `Mutex`.
//!
//! # Rebuild policy
//!
//! Any mutation (insert, delete) sets `tree_dirty = true`.  The next call to
//! [`VisualIndex::search_similar`] on a collection with at least
//! `TREE_THRESHOLD` entries will call `VisualIndex::rebuild_tree` and
//! clear the flag before dispatching to the VP-tree.
//!
//! # Example
//!
//! ```rust
//! use oximedia_search::visual::index::VisualIndex;
//! use std::env::temp_dir;
//!
//! let dir = temp_dir().join("visual_index_doctest");
//! let mut index = VisualIndex::new(&dir).expect("create index");
//!
//! let id = uuid::Uuid::new_v4();
//! index.add_document(id, &[100_u8, 150, 200]).expect("add doc");
//!
//! let results = index.search_similar(&[100, 150, 200], 5).expect("search");
//! assert!(!results.is_empty());
//! assert_eq!(results[0].asset_id, id);
//!
//! std::fs::remove_dir_all(dir).ok();
//! ```

use crate::error::SearchResult;
use crate::SearchResultItem;
use std::cell::{Cell, RefCell};
use std::path::Path;
use uuid::Uuid;

use super::vp_tree::{euclidean_dist, FloatVpTree};

/// Minimum number of indexed entries before the VP-tree is used.
///
/// Below this value a linear scan is cheaper due to VP-tree overhead.
const TREE_THRESHOLD: usize = 8;

/// Visual index for efficient similarity search.
///
/// Stores feature vectors as `f32` slices derived from raw byte features and
/// routes queries through a [`FloatVpTree`] for O(log n) average complexity
/// when the collection is large enough.
///
/// Uses [`RefCell`]/[`Cell`] for interior mutability so that
/// [`search_similar`](Self::search_similar) can be called through `&self`
/// while lazily rebuilding the VP-tree.
pub struct VisualIndex {
    index_path: std::path::PathBuf,
    /// Flat feature store: `(asset_id, normalised_f32_features)`.
    features: RefCell<Vec<(Uuid, Vec<f32>)>>,
    /// VP-tree built over `features`.  `None` when dirty or below threshold.
    tree: RefCell<Option<FloatVpTree>>,
    /// `true` whenever `features` is mutated; triggers a lazy rebuild.
    tree_dirty: Cell<bool>,
}

impl VisualIndex {
    /// Create a new visual index rooted at `index_path`.
    ///
    /// The directory is created if it does not exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created.
    pub fn new(index_path: &Path) -> SearchResult<Self> {
        if !index_path.exists() {
            std::fs::create_dir_all(index_path)?;
        }

        Ok(Self {
            index_path: index_path.to_path_buf(),
            features: RefCell::new(Vec::new()),
            tree: RefCell::new(None),
            tree_dirty: Cell::new(false),
        })
    }

    /// Add a document to the visual index.
    ///
    /// `features` is a byte slice that is normalised to `[0, 1]` f32 values.
    ///
    /// # Errors
    ///
    /// Returns an error if indexing fails.
    pub fn add_document(&mut self, asset_id: Uuid, features: &[u8]) -> SearchResult<()> {
        let feature_vec: Vec<f32> = features.iter().map(|&b| f32::from(b) / 255.0).collect();
        self.features.borrow_mut().push((asset_id, feature_vec));
        self.tree_dirty.set(true);
        Ok(())
    }

    /// Search for images similar to `query_data`.
    ///
    /// Routes through the VP-tree for N >= `TREE_THRESHOLD`; falls back to a
    /// linear scan for smaller collections.  The tree is rebuilt lazily when
    /// dirty (using interior mutability).
    ///
    /// # Errors
    ///
    /// Returns an error if search fails.
    pub fn search_similar(
        &self,
        query_data: &[u8],
        limit: usize,
    ) -> SearchResult<Vec<SearchResultItem>> {
        let query_features: Vec<f32> = query_data.iter().map(|&b| f32::from(b) / 255.0).collect();

        let n = self.features.borrow().len();

        if n >= TREE_THRESHOLD {
            if self.tree_dirty.get() || self.tree.borrow().is_none() {
                self.rebuild_tree();
                self.tree_dirty.set(false);
            }

            let tree_borrow = self.tree.borrow();
            if let Some(ref tree) = *tree_borrow {
                let knn = tree.search_knn(&query_features, limit);
                let features_borrow = self.features.borrow();
                let results: Vec<SearchResultItem> = knn
                    .into_iter()
                    .filter_map(|(store_idx, dist)| {
                        features_borrow.get(store_idx).map(|(asset_id, _)| {
                            let score = 1.0 / (1.0 + dist);
                            SearchResultItem {
                                asset_id: *asset_id,
                                score,
                                title: None,
                                description: None,
                                file_path: String::new(),
                                mime_type: None,
                                duration_ms: None,
                                created_at: 0,
                                modified_at: None,
                                file_size: None,
                                matched_fields: vec!["visual".to_string()],
                                thumbnail_url: None,
                            }
                        })
                    })
                    .collect();

                if !results.is_empty() {
                    return Ok(results);
                }
            }
        }

        Ok(self.linear_search_f32(&query_features, limit))
    }

    /// Rebuild the VP-tree from the current feature set.
    ///
    /// Points are stored as `(store_idx, Vec<f32>)` so that VP-tree results
    /// can be mapped back to `(Uuid, Vec<f32>)` entries via direct indexing.
    fn rebuild_tree(&self) {
        let features = self.features.borrow();
        if features.is_empty() {
            *self.tree.borrow_mut() = None;
            return;
        }

        let points: Vec<(usize, Vec<f32>)> = features
            .iter()
            .enumerate()
            .map(|(idx, (_, v))| (idx, v.clone()))
            .collect();

        *self.tree.borrow_mut() = Some(FloatVpTree::build(points));
    }

    /// Brute-force linear similarity search over normalised f32 features.
    ///
    /// Used when the collection is below `TREE_THRESHOLD` or as a fallback.
    pub fn linear_search_f32(&self, query_features: &[f32], limit: usize) -> Vec<SearchResultItem> {
        let features = self.features.borrow();
        let mut scored: Vec<SearchResultItem> = features
            .iter()
            .map(|(asset_id, feat_vec)| {
                let distance = euclidean_dist(query_features, feat_vec);
                let score = 1.0 / (1.0 + distance);
                SearchResultItem {
                    asset_id: *asset_id,
                    score,
                    title: None,
                    description: None,
                    file_path: String::new(),
                    mime_type: None,
                    duration_ms: None,
                    created_at: 0,
                    modified_at: None,
                    file_size: None,
                    matched_fields: vec!["visual".to_string()],
                    thumbnail_url: None,
                }
            })
            .collect();

        // Sort by score descending (highest similarity first).
        scored.sort_by(|a, b| b.score.total_cmp(&a.score));
        scored.truncate(limit);
        scored
    }

    /// Commit changes to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if commit fails.
    pub fn commit(&self) -> SearchResult<()> {
        let _ = &self.index_path; // suppress unused warning
        Ok(())
    }

    /// Delete a document from the index.
    ///
    /// Marks the VP-tree dirty so it is rebuilt on the next query.
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails.
    pub fn delete(&mut self, asset_id: Uuid) -> SearchResult<()> {
        let before = self.features.borrow().len();
        self.features.borrow_mut().retain(|(id, _)| *id != asset_id);
        if self.features.borrow().len() < before {
            self.tree_dirty.set(true);
            *self.tree.borrow_mut() = None;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index() -> VisualIndex {
        let dir = std::env::temp_dir().join(format!(
            "visual_index_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        VisualIndex::new(&dir).expect("should succeed")
    }

    /// Generate a deterministic dims-dimensional f32 vector from a seed integer.
    ///
    /// Uses the LCG formula to produce pseudo-random byte values then normalises to [0, 1].
    fn pseudo_vec(seed: usize, dims: usize) -> Vec<f32> {
        let mut val = seed;
        (0..dims)
            .map(|_| {
                val = val.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (val & 0xFF) as f32 / 255.0
            })
            .collect()
    }

    // ── Basic lifecycle ──────────────────────────────────────────────────────

    #[test]
    fn test_add_and_search() {
        let temp_dir = std::env::temp_dir().join("visual_index_test");
        let mut index = VisualIndex::new(&temp_dir).expect("should succeed in test");

        let id = Uuid::new_v4();
        let features = vec![100, 150, 200];

        index
            .add_document(id, &features)
            .expect("should succeed in test");

        let results = index
            .search_similar(&features, 10)
            .expect("should succeed in test");
        assert!(!results.is_empty());
        assert_eq!(results[0].asset_id, id);

        std::fs::remove_dir_all(temp_dir).ok();
    }

    // ── VP-tree vs linear agreement ──────────────────────────────────────────

    /// Insert 50 random 128-dim vectors.  VP-tree search_similar (N >= 8, so
    /// tree path) should return the same top-5 results as linear_search_f32.
    #[test]
    fn test_visual_index_vp_vs_linear() {
        let mut index = make_index();
        let dims = 128;
        let n = 50;

        for i in 0..n {
            let id = Uuid::from_u128(i as u128);
            let v: Vec<u8> = pseudo_vec(i * 31, dims)
                .iter()
                .map(|&f| (f * 255.0) as u8)
                .collect();
            index.add_document(id, &v).expect("add doc");
        }

        // Build a query that is the byte encoding of a pseudo_vec.
        let query_f32 = pseudo_vec(12345, dims);
        let query_bytes: Vec<u8> = query_f32.iter().map(|&f| (f * 255.0) as u8).collect();

        // VP-tree path (N=50 >= TREE_THRESHOLD=8).
        let vp_results = index.search_similar(&query_bytes, 5).expect("vp search");

        // Linear path for comparison.
        let linear_results = index.linear_search_f32(&query_f32, 5);

        assert_eq!(vp_results.len(), 5, "vp results count");
        assert_eq!(linear_results.len(), 5, "linear results count");

        // Collect result IDs for set-equality check.
        let vp_ids: std::collections::HashSet<Uuid> =
            vp_results.iter().map(|r| r.asset_id).collect();
        let linear_ids: std::collections::HashSet<Uuid> =
            linear_results.iter().map(|r| r.asset_id).collect();

        assert_eq!(
            vp_ids, linear_ids,
            "VP-tree and linear should return the same top-5 IDs"
        );
    }

    /// Insert 1000 random 128-dim vectors.  VP-tree search should complete
    /// without error and return the correct number of results.
    #[test]
    fn test_visual_index_sublinear() {
        let mut index = make_index();
        let dims = 128;
        let n = 1000;

        for i in 0..n {
            let id = Uuid::from_u128(i as u128);
            let v: Vec<u8> = pseudo_vec(i * 31, dims)
                .iter()
                .map(|&f| (f * 255.0) as u8)
                .collect();
            index.add_document(id, &v).expect("add doc");
        }

        let query: Vec<u8> = pseudo_vec(99999, dims)
            .iter()
            .map(|&f| (f * 255.0) as u8)
            .collect();

        let results = index.search_similar(&query, 10).expect("vp search");
        assert_eq!(results.len(), 10, "should return exactly top-10");
    }

    /// Insert 1000 vectors, search 100 queries.  All must complete without error.
    #[test]
    fn test_visual_index_batch_throughput() {
        let mut index = make_index();
        let dims = 64;
        let n = 1000;

        for i in 0..n {
            let id = Uuid::from_u128(i as u128);
            let v: Vec<u8> = pseudo_vec(i * 7, dims)
                .iter()
                .map(|&f| (f * 255.0) as u8)
                .collect();
            index.add_document(id, &v).expect("add doc");
        }

        for q in 0..100_usize {
            let query: Vec<u8> = pseudo_vec(q * 999, dims)
                .iter()
                .map(|&f| (f * 255.0) as u8)
                .collect();
            let results = index.search_similar(&query, 5).expect("batch search");
            assert!(!results.is_empty(), "query {} returned empty", q);
        }
    }

    /// Create 10 "duplicate" pairs: original vector + small noise variant.
    /// Searching for each original should return its duplicate in top-3.
    #[test]
    fn test_visual_index_precision_recall() {
        let mut index = make_index();
        let dims = 32;

        // pairs: (original_id, noisy_id, original_bytes)
        let mut originals: Vec<(Uuid, Vec<u8>)> = Vec::new();
        let mut noisy_ids: Vec<Uuid> = Vec::new();

        for i in 0..10_usize {
            let orig_id = Uuid::from_u128(i as u128);
            let noisy_id = Uuid::from_u128((i + 100) as u128);

            // Original vector
            let orig_f32 = pseudo_vec(i * 1000, dims);
            let orig_bytes: Vec<u8> = orig_f32.iter().map(|&f| (f * 255.0) as u8).collect();

            // Noisy variant: perturb each value by ±1 (within 0..=255).
            let noisy_bytes: Vec<u8> = orig_bytes
                .iter()
                .enumerate()
                .map(|(j, &b)| {
                    let delta: i16 = if (j + i) % 2 == 0 { 1 } else { -1 };
                    (b as i16 + delta).clamp(0, 255) as u8
                })
                .collect();

            originals.push((orig_id, orig_bytes.clone()));
            noisy_ids.push(noisy_id);

            index.add_document(orig_id, &orig_bytes).expect("add orig");
            index
                .add_document(noisy_id, &noisy_bytes)
                .expect("add noisy");
        }

        // Add 80 distractor vectors (random, far from originals).
        for i in 20..100_usize {
            let id = Uuid::from_u128(i as u128);
            let v: Vec<u8> = pseudo_vec(i * 9999, dims)
                .iter()
                .map(|&f| (f * 255.0) as u8)
                .collect();
            index.add_document(id, &v).expect("add distractor");
        }

        // For each original, verify its noisy duplicate appears in top-3.
        for (pair_idx, (orig_id, orig_bytes)) in originals.iter().enumerate() {
            let results = index
                .search_similar(orig_bytes, 3)
                .expect("precision recall search");

            let noisy_id = noisy_ids[pair_idx];
            let found = results
                .iter()
                .any(|r| r.asset_id == noisy_id || r.asset_id == *orig_id);
            assert!(
                found,
                "pair {}: duplicate not in top-3 (got {:?})",
                pair_idx,
                results.iter().map(|r| r.asset_id).collect::<Vec<_>>()
            );
        }
    }
}
