#![allow(dead_code)]
//! Federated search across multiple independent search indices.
//!
//! This module enables combining results from multiple index shards or
//! heterogeneous backends (text, visual, audio) into a single unified
//! result set with configurable merging and deduplication strategies.

use std::collections::HashMap;

/// Identifier for a search backend or index shard.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BackendId(String);

impl BackendId {
    /// Create a new backend identifier.
    #[must_use]
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }

    /// Return the identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Strategy for merging results from multiple backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Interleave results round-robin from each backend.
    RoundRobin,
    /// Take all results from highest-weight backend first.
    WeightedPriority,
    /// Merge by score across all backends.
    ScoreMerge,
    /// Reciprocal rank fusion.
    ReciprocalRankFusion,
}

/// Strategy for handling duplicate items across backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeduplicationStrategy {
    /// Keep the first occurrence.
    KeepFirst,
    /// Keep the highest-scored occurrence.
    KeepHighestScore,
    /// Average scores of duplicates.
    AverageScores,
}

/// A single federated result item.
#[derive(Debug, Clone)]
pub struct FederatedItem {
    /// Unique document identifier.
    pub doc_id: String,
    /// Relevance score from backend.
    pub score: f64,
    /// Which backend produced this result.
    pub source: BackendId,
    /// Optional title.
    pub title: Option<String>,
    /// Backend-specific metadata.
    pub metadata: HashMap<String, String>,
}

impl FederatedItem {
    /// Create a new federated item.
    #[must_use]
    pub fn new(doc_id: &str, score: f64, source: BackendId) -> Self {
        Self {
            doc_id: doc_id.to_string(),
            score,
            source,
            title: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the title.
    #[must_use]
    pub fn with_title(mut self, title: &str) -> Self {
        self.title = Some(title.to_string());
        self
    }
}

/// Configuration for a single backend in the federation.
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Backend identifier.
    pub id: BackendId,
    /// Weight for score merging (higher = more important).
    pub weight: f64,
    /// Maximum results to request from this backend.
    pub max_results: usize,
    /// Whether this backend is enabled.
    pub enabled: bool,
}

impl BackendConfig {
    /// Create a new backend configuration.
    #[must_use]
    pub fn new(id: &str, weight: f64, max_results: usize) -> Self {
        Self {
            id: BackendId::new(id),
            weight,
            max_results,
            enabled: true,
        }
    }
}

/// Result of a federated search across all backends.
#[derive(Debug, Clone)]
pub struct FederatedResults {
    /// Merged result items.
    pub items: Vec<FederatedItem>,
    /// Total results before limit.
    pub total_count: usize,
    /// Per-backend result counts.
    pub backend_counts: HashMap<String, usize>,
    /// Whether any backends failed.
    pub partial: bool,
    /// Names of backends that failed.
    pub failed_backends: Vec<String>,
}

/// Federated search coordinator that merges results from multiple backends.
#[derive(Debug)]
pub struct FederatedSearch {
    /// Backend configurations.
    backends: Vec<BackendConfig>,
    /// Merge strategy.
    merge_strategy: MergeStrategy,
    /// Deduplication strategy.
    dedup_strategy: DeduplicationStrategy,
    /// RRF constant (k) for reciprocal rank fusion.
    rrf_k: f64,
    /// Global result limit.
    result_limit: usize,
}

impl FederatedSearch {
    /// Create a new federated search coordinator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
            merge_strategy: MergeStrategy::ScoreMerge,
            dedup_strategy: DeduplicationStrategy::KeepHighestScore,
            rrf_k: 60.0,
            result_limit: 100,
        }
    }

    /// Add a backend configuration.
    pub fn add_backend(&mut self, config: BackendConfig) {
        self.backends.push(config);
    }

    /// Set the merge strategy.
    #[must_use]
    pub fn with_merge_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.merge_strategy = strategy;
        self
    }

    /// Set the deduplication strategy.
    #[must_use]
    pub fn with_dedup_strategy(mut self, strategy: DeduplicationStrategy) -> Self {
        self.dedup_strategy = strategy;
        self
    }

    /// Set the RRF constant.
    #[must_use]
    pub fn with_rrf_k(mut self, k: f64) -> Self {
        self.rrf_k = k;
        self
    }

    /// Set the global result limit.
    #[must_use]
    pub fn with_result_limit(mut self, limit: usize) -> Self {
        self.result_limit = limit;
        self
    }

    /// Return the number of configured backends.
    #[must_use]
    pub fn backend_count(&self) -> usize {
        self.backends.len()
    }

    /// Merge result sets from multiple backends.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn merge(&self, backend_results: &HashMap<String, Vec<FederatedItem>>) -> FederatedResults {
        let mut backend_counts = HashMap::new();
        let mut failed_backends = Vec::new();

        for backend in &self.backends {
            let id = backend.id.as_str();
            if let Some(results) = backend_results.get(id) {
                backend_counts.insert(id.to_string(), results.len());
            } else if backend.enabled {
                failed_backends.push(id.to_string());
                backend_counts.insert(id.to_string(), 0);
            }
        }

        let merged = match self.merge_strategy {
            MergeStrategy::RoundRobin => self.merge_round_robin(backend_results),
            MergeStrategy::WeightedPriority => self.merge_weighted_priority(backend_results),
            MergeStrategy::ScoreMerge => self.merge_by_score(backend_results),
            MergeStrategy::ReciprocalRankFusion => self.merge_rrf(backend_results),
        };

        let deduped = self.deduplicate(merged);
        let total_count = deduped.len();
        let items: Vec<FederatedItem> = deduped.into_iter().take(self.result_limit).collect();

        FederatedResults {
            items,
            total_count,
            backend_counts,
            partial: !failed_backends.is_empty(),
            failed_backends,
        }
    }

    /// Merge by interleaving round-robin.
    fn merge_round_robin(
        &self,
        backend_results: &HashMap<String, Vec<FederatedItem>>,
    ) -> Vec<FederatedItem> {
        let mut iters: Vec<std::slice::Iter<FederatedItem>> = self
            .backends
            .iter()
            .filter_map(|b| backend_results.get(b.id.as_str()).map(|v| v.iter()))
            .collect();

        let mut merged = Vec::new();
        let mut any_remaining = true;
        while any_remaining {
            any_remaining = false;
            for iter in &mut iters {
                if let Some(item) = iter.next() {
                    merged.push(item.clone());
                    any_remaining = true;
                }
            }
        }
        merged
    }

    /// Merge by weighted priority: backends with higher weight go first.
    ///
    /// Scores are scaled by the backend weight so that items from
    /// higher-priority backends sort above items from lower-priority ones
    /// when scores are otherwise equal.
    fn merge_weighted_priority(
        &self,
        backend_results: &HashMap<String, Vec<FederatedItem>>,
    ) -> Vec<FederatedItem> {
        let weight_map: HashMap<&str, f64> = self
            .backends
            .iter()
            .map(|b| (b.id.as_str(), b.weight))
            .collect();

        let mut sorted_backends: Vec<&BackendConfig> = self.backends.iter().collect();
        sorted_backends.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut merged = Vec::new();
        for backend in sorted_backends {
            if let Some(results) = backend_results.get(backend.id.as_str()) {
                let weight = weight_map.get(backend.id.as_str()).copied().unwrap_or(1.0);
                merged.extend(results.iter().map(|item| {
                    let mut scored = item.clone();
                    scored.score *= weight;
                    scored
                }));
            }
        }
        merged
    }

    /// Merge all results and sort by score.
    #[allow(clippy::cast_precision_loss)]
    fn merge_by_score(
        &self,
        backend_results: &HashMap<String, Vec<FederatedItem>>,
    ) -> Vec<FederatedItem> {
        let weight_map: HashMap<&str, f64> = self
            .backends
            .iter()
            .map(|b| (b.id.as_str(), b.weight))
            .collect();

        let mut merged: Vec<FederatedItem> = backend_results
            .iter()
            .flat_map(|(backend_id, items)| {
                let weight = weight_map.get(backend_id.as_str()).copied().unwrap_or(1.0);
                items.iter().map(move |item| {
                    let mut scored = item.clone();
                    scored.score *= weight;
                    scored
                })
            })
            .collect();

        merged.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged
    }

    /// Merge using Reciprocal Rank Fusion.
    #[allow(clippy::cast_precision_loss)]
    fn merge_rrf(
        &self,
        backend_results: &HashMap<String, Vec<FederatedItem>>,
    ) -> Vec<FederatedItem> {
        let mut rrf_scores: HashMap<String, f64> = HashMap::new();
        let mut item_map: HashMap<String, FederatedItem> = HashMap::new();

        for items in backend_results.values() {
            for (rank, item) in items.iter().enumerate() {
                let rrf_score = 1.0 / (self.rrf_k + (rank + 1) as f64);
                *rrf_scores.entry(item.doc_id.clone()).or_insert(0.0) += rrf_score;
                item_map
                    .entry(item.doc_id.clone())
                    .or_insert_with(|| item.clone());
            }
        }

        let mut merged: Vec<FederatedItem> = item_map
            .into_values()
            .map(|mut item| {
                item.score = rrf_scores.get(&item.doc_id).copied().unwrap_or(0.0);
                item
            })
            .collect();

        merged.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged
    }

    /// Deduplicate items based on configured strategy.
    fn deduplicate(&self, items: Vec<FederatedItem>) -> Vec<FederatedItem> {
        let mut seen: HashMap<String, FederatedItem> = HashMap::new();

        for item in items {
            match self.dedup_strategy {
                DeduplicationStrategy::KeepFirst => {
                    seen.entry(item.doc_id.clone()).or_insert(item);
                }
                DeduplicationStrategy::KeepHighestScore => {
                    seen.entry(item.doc_id.clone())
                        .and_modify(|existing| {
                            if item.score > existing.score {
                                *existing = item.clone();
                            }
                        })
                        .or_insert(item);
                }
                DeduplicationStrategy::AverageScores => {
                    seen.entry(item.doc_id.clone())
                        .and_modify(|existing| {
                            existing.score = (existing.score + item.score) / 2.0;
                        })
                        .or_insert(item);
                }
            }
        }

        let mut result: Vec<FederatedItem> = seen.into_values().collect();
        result.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }
}

impl Default for FederatedSearch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_items(backend_name: &str, ids: &[&str], scores: &[f64]) -> Vec<FederatedItem> {
        ids.iter()
            .zip(scores.iter())
            .map(|(id, &score)| FederatedItem::new(id, score, BackendId::new(backend_name)))
            .collect()
    }

    #[test]
    fn test_federated_search_creation() {
        let fs = FederatedSearch::new();
        assert_eq!(fs.backend_count(), 0);
    }

    #[test]
    fn test_add_backend() {
        let mut fs = FederatedSearch::new();
        fs.add_backend(BackendConfig::new("text", 1.0, 50));
        fs.add_backend(BackendConfig::new("visual", 0.8, 30));
        assert_eq!(fs.backend_count(), 2);
    }

    #[test]
    fn test_score_merge() {
        let mut fs = FederatedSearch::new();
        fs.add_backend(BackendConfig::new("a", 1.0, 50));
        fs.add_backend(BackendConfig::new("b", 1.0, 50));
        let fs = fs.with_merge_strategy(MergeStrategy::ScoreMerge);

        let mut backend_results = HashMap::new();
        backend_results.insert("a".to_string(), make_items("a", &["d1", "d2"], &[0.9, 0.7]));
        backend_results.insert(
            "b".to_string(),
            make_items("b", &["d3", "d4"], &[0.85, 0.6]),
        );

        let results = fs.merge(&backend_results);
        assert_eq!(results.items.len(), 4);
        // First item should be the highest score
        assert_eq!(results.items[0].doc_id, "d1");
    }

    #[test]
    fn test_round_robin_merge() {
        let mut fs = FederatedSearch::new();
        fs.add_backend(BackendConfig::new("a", 1.0, 50));
        fs.add_backend(BackendConfig::new("b", 1.0, 50));
        let fs = fs.with_merge_strategy(MergeStrategy::RoundRobin);

        let mut backend_results = HashMap::new();
        backend_results.insert("a".to_string(), make_items("a", &["a1", "a2"], &[1.0, 0.9]));
        backend_results.insert("b".to_string(), make_items("b", &["b1", "b2"], &[1.0, 0.9]));

        let results = fs.merge(&backend_results);
        assert_eq!(results.items.len(), 4);
    }

    #[test]
    fn test_weighted_priority_merge() {
        let mut fs = FederatedSearch::new();
        fs.add_backend(BackendConfig::new("low", 0.5, 50));
        fs.add_backend(BackendConfig::new("high", 2.0, 50));
        let fs = fs.with_merge_strategy(MergeStrategy::WeightedPriority);

        let mut backend_results = HashMap::new();
        backend_results.insert("low".to_string(), make_items("low", &["l1"], &[1.0]));
        backend_results.insert("high".to_string(), make_items("high", &["h1"], &[1.0]));

        let results = fs.merge(&backend_results);
        assert_eq!(results.items.len(), 2);
        // high-weight backend items come first
        assert_eq!(results.items[0].source.as_str(), "high");
    }

    #[test]
    fn test_rrf_merge() {
        let mut fs = FederatedSearch::new();
        fs.add_backend(BackendConfig::new("a", 1.0, 50));
        fs.add_backend(BackendConfig::new("b", 1.0, 50));
        let fs = fs
            .with_merge_strategy(MergeStrategy::ReciprocalRankFusion)
            .with_rrf_k(60.0);

        let mut backend_results = HashMap::new();
        backend_results.insert("a".to_string(), make_items("a", &["d1", "d2"], &[0.9, 0.5]));
        backend_results.insert("b".to_string(), make_items("b", &["d1", "d3"], &[0.8, 0.3]));

        let results = fs.merge(&backend_results);
        // d1 appears in both, should have highest RRF score
        assert_eq!(results.items[0].doc_id, "d1");
    }

    #[test]
    fn test_dedup_keep_first() {
        let mut fs = FederatedSearch::new();
        fs.add_backend(BackendConfig::new("a", 1.0, 50));
        let fs = fs
            .with_merge_strategy(MergeStrategy::ScoreMerge)
            .with_dedup_strategy(DeduplicationStrategy::KeepFirst);

        let mut backend_results = HashMap::new();
        backend_results.insert("a".to_string(), make_items("a", &["d1", "d1"], &[0.9, 0.5]));

        let results = fs.merge(&backend_results);
        assert_eq!(results.items.len(), 1);
    }

    #[test]
    fn test_dedup_keep_highest() {
        let mut fs = FederatedSearch::new();
        fs.add_backend(BackendConfig::new("a", 1.0, 50));
        let fs = fs
            .with_merge_strategy(MergeStrategy::ScoreMerge)
            .with_dedup_strategy(DeduplicationStrategy::KeepHighestScore);

        let mut backend_results = HashMap::new();
        backend_results.insert("a".to_string(), make_items("a", &["d1", "d1"], &[0.3, 0.9]));

        let results = fs.merge(&backend_results);
        assert_eq!(results.items.len(), 1);
        assert!((results.items[0].score - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dedup_average() {
        let mut fs = FederatedSearch::new();
        fs.add_backend(BackendConfig::new("a", 1.0, 50));
        let fs = fs
            .with_merge_strategy(MergeStrategy::ScoreMerge)
            .with_dedup_strategy(DeduplicationStrategy::AverageScores);

        let mut backend_results = HashMap::new();
        backend_results.insert("a".to_string(), make_items("a", &["d1", "d1"], &[0.4, 0.8]));

        let results = fs.merge(&backend_results);
        assert_eq!(results.items.len(), 1);
        assert!((results.items[0].score - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_result_limit() {
        let mut fs = FederatedSearch::new();
        fs.add_backend(BackendConfig::new("a", 1.0, 50));
        let fs = fs.with_result_limit(2);

        let mut backend_results = HashMap::new();
        backend_results.insert(
            "a".to_string(),
            make_items("a", &["d1", "d2", "d3", "d4"], &[0.9, 0.8, 0.7, 0.6]),
        );

        let results = fs.merge(&backend_results);
        assert_eq!(results.items.len(), 2);
        assert_eq!(results.total_count, 4);
    }

    #[test]
    fn test_partial_results_on_missing_backend() {
        let mut fs = FederatedSearch::new();
        fs.add_backend(BackendConfig::new("a", 1.0, 50));
        fs.add_backend(BackendConfig::new("missing", 1.0, 50));

        let mut backend_results = HashMap::new();
        backend_results.insert("a".to_string(), make_items("a", &["d1"], &[0.9]));
        // "missing" backend has no results entry

        let results = fs.merge(&backend_results);
        assert!(results.partial);
        assert!(results.failed_backends.contains(&"missing".to_string()));
    }

    #[test]
    fn test_federated_item_with_title() {
        let item = FederatedItem::new("doc1", 0.95, BackendId::new("text")).with_title("My Video");
        assert_eq!(item.title.as_deref(), Some("My Video"));
        assert_eq!(item.doc_id, "doc1");
    }

    #[test]
    fn test_backend_id_as_str() {
        let id = BackendId::new("visual");
        assert_eq!(id.as_str(), "visual");
    }
}
