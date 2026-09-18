//! Result deduplication and diversity enforcement utilities.
//!
//! Provides:
//! - [`SimilarityMatrix`] – sparse pairwise similarity storage
//! - [`MaximalMarginalRelevance`] – MMR-based diverse selection
//! - [`DuplicateDetector`] – Jaccard-based near-duplicate detection
//! - [`CategoryDiversifier`] – per-category capping

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Similarity matrix
// ---------------------------------------------------------------------------

/// Sparse symmetric matrix for pairwise similarity scores.
///
/// Scores are stored only for pairs `(i, j)` where `i <= j` to avoid
/// redundancy.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct SimilarityMatrix {
    data: HashMap<(u64, u64), f32>,
}

impl SimilarityMatrix {
    /// Create an empty similarity matrix.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Store the similarity between items `i` and `j`.
    pub fn set(&mut self, i: u64, j: u64, sim: f32) {
        let key = if i <= j { (i, j) } else { (j, i) };
        self.data.insert(key, sim.clamp(0.0, 1.0));
    }

    /// Retrieve the similarity between items `i` and `j`.
    ///
    /// Returns `0.0` if the pair has not been stored.
    #[must_use]
    pub fn get(&self, i: u64, j: u64) -> f32 {
        let key = if i <= j { (i, j) } else { (j, i) };
        *self.data.get(&key).unwrap_or(&0.0)
    }

    /// Return the number of stored pairs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return `true` if no pairs are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Maximal Marginal Relevance
// ---------------------------------------------------------------------------

/// Maximal Marginal Relevance (MMR) for diverse candidate selection.
///
/// Given a list of `(item_id, relevance_score)` candidates and a similarity
/// matrix, iteratively selects the item that maximises:
///
/// ```text
/// MMR_score = lambda * relevance - (1 - lambda) * max_sim_to_selected
/// ```
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MaximalMarginalRelevance {
    /// Trade-off between relevance (1.0) and diversity (0.0)
    pub lambda: f32,
}

impl MaximalMarginalRelevance {
    /// Create a new MMR selector with the given lambda.
    #[must_use]
    pub fn new(lambda: f32) -> Self {
        Self {
            lambda: lambda.clamp(0.0, 1.0),
        }
    }

    /// Select up to `k` diverse items from `candidates`.
    ///
    /// `candidates` is a slice of `(item_id, relevance)` pairs.
    /// Items are returned in the order they were greedily selected.
    #[must_use]
    pub fn select(
        &self,
        candidates: &[(u64, f32)],
        similarity: &SimilarityMatrix,
        k: usize,
    ) -> Vec<u64> {
        if candidates.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut selected: Vec<u64> = Vec::with_capacity(k);
        let mut remaining: Vec<(u64, f32)> = candidates.to_vec();

        while selected.len() < k && !remaining.is_empty() {
            let best_idx = remaining
                .iter()
                .enumerate()
                .map(|(i, &(id, rel))| {
                    let max_sim = selected
                        .iter()
                        .map(|&sel| similarity.get(id, sel))
                        .fold(0.0_f32, f32::max);
                    let mmr = self.lambda * rel - (1.0 - self.lambda) * max_sim;
                    (i, mmr)
                })
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i);

            let (chosen_id, _) = remaining.remove(best_idx);
            selected.push(chosen_id);
        }

        selected
    }
}

impl Default for MaximalMarginalRelevance {
    fn default() -> Self {
        Self::new(0.7)
    }
}

// ---------------------------------------------------------------------------
// Duplicate detector
// ---------------------------------------------------------------------------

/// Near-duplicate detector based on Jaccard similarity of fingerprints.
///
/// Each item is registered with a set of 32-bit tokens (e.g. perceptual hash
/// buckets).  Two items are considered duplicates when their Jaccard
/// similarity exceeds a configurable threshold.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct DuplicateDetector {
    fingerprints: HashMap<u64, HashSet<u32>>,
}

impl DuplicateDetector {
    /// Create an empty duplicate detector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an item with its fingerprint tokens.
    pub fn add(&mut self, id: u64, fingerprint: Vec<u32>) {
        self.fingerprints
            .insert(id, fingerprint.into_iter().collect());
    }

    /// Return all pairs `(a, b)` with `a < b` whose Jaccard similarity is
    /// at least `threshold`.
    #[must_use]
    pub fn find_duplicates(&self, threshold: f32) -> Vec<(u64, u64)> {
        let items: Vec<u64> = self.fingerprints.keys().copied().collect();
        let mut result = Vec::new();

        for (i, &a) in items.iter().enumerate() {
            for &b in &items[i + 1..] {
                let sim = self.jaccard(a, b);
                if sim >= threshold {
                    let pair = if a < b { (a, b) } else { (b, a) };
                    result.push(pair);
                }
            }
        }

        result.sort_unstable();
        result
    }

    /// Compute the Jaccard similarity between two registered items.
    #[must_use]
    fn jaccard(&self, a: u64, b: u64) -> f32 {
        let fa = match self.fingerprints.get(&a) {
            Some(f) => f,
            None => return 0.0,
        };
        let fb = match self.fingerprints.get(&b) {
            Some(f) => f,
            None => return 0.0,
        };

        let intersection = fa.intersection(fb).count();
        let union = fa.union(fb).count();

        if union == 0 {
            return 0.0;
        }

        intersection as f32 / union as f32
    }
}

// ---------------------------------------------------------------------------
// Category diversifier
// ---------------------------------------------------------------------------

/// Ensures that no single category dominates the result list.
///
/// Items are processed in the order supplied; the first `max_per_category`
/// items belonging to a given category pass through, the rest are dropped.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CategoryDiversifier {
    /// Maximum items allowed per category
    pub max_per_category: usize,
}

impl CategoryDiversifier {
    /// Create a new diversifier.
    #[must_use]
    pub fn new(max_per_category: usize) -> Self {
        Self { max_per_category }
    }

    /// Filter `items` so that each category appears at most
    /// `max_per_category` times.
    ///
    /// `items` is a slice of `(media_id, category)` pairs.
    /// Returns the media IDs that survived the cap, in input order.
    #[must_use]
    pub fn diversify(&self, items: &[(u64, &str)], max_per_category: usize) -> Vec<u64> {
        let cap = max_per_category;
        let mut counts: HashMap<&str, usize> = HashMap::new();
        let mut result = Vec::new();

        for &(id, cat) in items {
            let count = counts.entry(cat).or_insert(0);
            if *count < cap {
                *count += 1;
                result.push(id);
            }
        }

        result
    }
}

impl Default for CategoryDiversifier {
    fn default() -> Self {
        Self::new(3)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SimilarityMatrix --

    #[test]
    fn test_similarity_matrix_set_get() {
        let mut m = SimilarityMatrix::new();
        m.set(1, 2, 0.75);
        assert!((m.get(1, 2) - 0.75).abs() < 1e-6);
        assert!((m.get(2, 1) - 0.75).abs() < 1e-6); // symmetric
    }

    #[test]
    fn test_similarity_matrix_missing_is_zero() {
        let m = SimilarityMatrix::new();
        assert_eq!(m.get(10, 20), 0.0);
    }

    #[test]
    fn test_similarity_matrix_clamps() {
        let mut m = SimilarityMatrix::new();
        m.set(1, 2, 1.5); // should clamp to 1.0
        assert!((m.get(1, 2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_matrix_len() {
        let mut m = SimilarityMatrix::new();
        m.set(1, 2, 0.5);
        m.set(3, 4, 0.8);
        assert_eq!(m.len(), 2);
    }

    // -- MaximalMarginalRelevance --

    #[test]
    fn test_mmr_empty_candidates() {
        let mmr = MaximalMarginalRelevance::new(0.7);
        let sim = SimilarityMatrix::new();
        let result = mmr.select(&[], &sim, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_mmr_selects_top_k() {
        let mmr = MaximalMarginalRelevance::new(0.7);
        let sim = SimilarityMatrix::new();
        let candidates = vec![(1u64, 0.9), (2u64, 0.8), (3u64, 0.7)];
        let result = mmr.select(&candidates, &sim, 2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_mmr_diversity() {
        let mmr = MaximalMarginalRelevance::new(0.5);
        let mut sim = SimilarityMatrix::new();
        // Items 1 and 2 are very similar; item 3 is different
        sim.set(1, 2, 0.95);
        sim.set(1, 3, 0.1);
        sim.set(2, 3, 0.1);
        // Item 1 is clearly the top candidate; item 3 should beat item 2 after 1 is chosen
        let candidates = vec![(1u64, 1.0), (2u64, 0.9), (3u64, 0.85)];
        let result = mmr.select(&candidates, &sim, 2);
        // Item 1 has the highest relevance, so it is selected first.
        // After that, item 3 is more diverse than item 2 (which is near-duplicate of 1).
        assert!(result.contains(&1));
        assert!(result.contains(&3));
    }

    // -- DuplicateDetector --

    #[test]
    fn test_duplicate_detector_no_duplicates() {
        let mut dd = DuplicateDetector::new();
        dd.add(1, vec![1, 2, 3]);
        dd.add(2, vec![4, 5, 6]);
        assert!(dd.find_duplicates(0.5).is_empty());
    }

    #[test]
    fn test_duplicate_detector_exact_duplicate() {
        let mut dd = DuplicateDetector::new();
        dd.add(1, vec![1, 2, 3, 4]);
        dd.add(2, vec![1, 2, 3, 4]);
        let dups = dd.find_duplicates(0.9);
        assert!(!dups.is_empty());
        assert!(dups.contains(&(1, 2)));
    }

    #[test]
    fn test_duplicate_detector_partial_overlap() {
        let mut dd = DuplicateDetector::new();
        dd.add(10, vec![1, 2, 3, 4]);
        dd.add(20, vec![3, 4, 5, 6]);
        // intersection={3,4}, union={1,2,3,4,5,6} → Jaccard=2/6≈0.33
        assert!(dd.find_duplicates(0.5).is_empty());
        assert_eq!(dd.find_duplicates(0.3).len(), 1);
    }

    #[test]
    fn test_duplicate_detector_threshold() {
        let mut dd = DuplicateDetector::new();
        dd.add(1, vec![1, 2, 3]);
        dd.add(2, vec![1, 2, 3]);
        // Jaccard = 1.0
        assert_eq!(dd.find_duplicates(1.0).len(), 1);
        assert_eq!(dd.find_duplicates(1.1).len(), 0);
    }

    // -- CategoryDiversifier --

    #[test]
    fn test_category_diversifier_basic() {
        let div = CategoryDiversifier::new(2);
        let items: Vec<(u64, &str)> = vec![
            (1, "action"),
            (2, "action"),
            (3, "action"), // should be dropped
            (4, "drama"),
        ];
        let result = div.diversify(&items, 2);
        assert_eq!(result, vec![1, 2, 4]);
    }

    #[test]
    fn test_category_diversifier_all_different() {
        let div = CategoryDiversifier::default();
        let items: Vec<(u64, &str)> = vec![(1, "a"), (2, "b"), (3, "c")];
        let result = div.diversify(&items, 1);
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_category_diversifier_empty() {
        let div = CategoryDiversifier::default();
        let result = div.diversify(&[], 3);
        assert!(result.is_empty());
    }
}
