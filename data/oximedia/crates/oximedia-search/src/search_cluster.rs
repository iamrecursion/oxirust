#![allow(dead_code)]
//! Query and result clustering for `oximedia-search`.
//!
//! Groups similar queries or search result documents into clusters so that
//! diverse result sets can be presented to users (e.g. "Videos", "Images",
//! "Audio clips").  Uses simple centroid-based clustering on feature vectors.

use std::collections::HashMap;

/// Strategy for assigning a document to a cluster.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClusterStrategy {
    /// Assign by MIME type category (video / audio / image / document).
    MimeCategory,
    /// Assign by keyword overlap with cluster label terms.
    KeywordOverlap,
    /// Assign by nearest centroid in a feature vector space.
    NearestCentroid,
}

/// A cluster of related search results.
#[derive(Debug, Clone)]
pub struct ResultCluster {
    /// Human-readable label for the cluster.
    pub label: String,
    /// Asset IDs belonging to this cluster.
    pub members: Vec<String>,
    /// Representative score for the cluster (average of member scores).
    pub avg_score: f32,
}

impl ResultCluster {
    /// Create an empty cluster with the given label.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            members: Vec::new(),
            avg_score: 0.0,
        }
    }

    /// Add a member asset and its score.
    pub fn add_member(&mut self, asset_id: impl Into<String>, score: f32) {
        self.members.push(asset_id.into());
        // Recompute average score.
        let n = self.members.len();
        #[allow(clippy::cast_precision_loss)]
        {
            self.avg_score = (self.avg_score * (n as f32 - 1.0) + score) / n as f32;
        }
    }

    /// Returns the cluster size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.members.len()
    }

    /// Returns `true` if the cluster has no members.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

/// A document to be clustered.
#[derive(Debug, Clone)]
pub struct ClusterDocument {
    /// Asset identifier.
    pub asset_id: String,
    /// MIME type string (e.g. `"video/mp4"`).
    pub mime_type: String,
    /// Keywords / tags associated with this asset.
    pub keywords: Vec<String>,
    /// Relevance score from search.
    pub score: f32,
    /// Optional feature vector for centroid clustering.
    pub features: Vec<f32>,
}

impl ClusterDocument {
    /// Create a minimal document with just an ID, MIME type, and score.
    pub fn new(asset_id: impl Into<String>, mime_type: impl Into<String>, score: f32) -> Self {
        Self {
            asset_id: asset_id.into(),
            mime_type: mime_type.into(),
            keywords: Vec::new(),
            score,
            features: Vec::new(),
        }
    }

    /// Add keywords to this document.
    #[must_use]
    pub fn with_keywords(mut self, keywords: Vec<String>) -> Self {
        self.keywords = keywords;
        self
    }

    /// Add a feature vector for centroid-based clustering.
    #[must_use]
    pub fn with_features(mut self, features: Vec<f32>) -> Self {
        self.features = features;
        self
    }
}

/// Configuration for the `SearchClusterer`.
#[derive(Debug, Clone)]
pub struct ClustererConfig {
    /// Clustering strategy to use.
    pub strategy: ClusterStrategy,
    /// Maximum number of clusters.
    pub max_clusters: usize,
    /// Minimum cluster size — clusters smaller than this are merged into "Other".
    pub min_cluster_size: usize,
}

impl Default for ClustererConfig {
    fn default() -> Self {
        Self {
            strategy: ClusterStrategy::MimeCategory,
            max_clusters: 6,
            min_cluster_size: 1,
        }
    }
}

/// Groups search result documents into labelled clusters.
#[derive(Debug)]
pub struct SearchClusterer {
    config: ClustererConfig,
}

impl SearchClusterer {
    /// Create a clusterer with default (MIME-category) configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ClustererConfig::default(),
        }
    }

    /// Create a clusterer with a custom configuration.
    #[must_use]
    pub fn with_config(config: ClustererConfig) -> Self {
        Self { config }
    }

    /// Cluster a list of documents and return the resulting clusters.
    ///
    /// Clusters with fewer than `config.min_cluster_size` members are merged
    /// into an "Other" bucket.  The returned vec is sorted by `avg_score` desc.
    #[must_use]
    pub fn cluster(&self, documents: Vec<ClusterDocument>) -> Vec<ResultCluster> {
        let clusters = match self.config.strategy {
            ClusterStrategy::MimeCategory => self.cluster_by_mime(documents),
            ClusterStrategy::KeywordOverlap => self.cluster_by_keyword(documents),
            ClusterStrategy::NearestCentroid => self.cluster_by_centroid(documents),
        };
        self.merge_small_clusters(clusters)
    }

    /// Cluster by top-level MIME type category.
    fn cluster_by_mime(&self, documents: Vec<ClusterDocument>) -> Vec<ResultCluster> {
        let mut map: HashMap<String, ResultCluster> = HashMap::new();
        for doc in documents {
            let label = mime_category(&doc.mime_type);
            map.entry(label.clone())
                .or_insert_with(|| ResultCluster::new(&label))
                .add_member(&doc.asset_id, doc.score);
        }
        map.into_values().collect()
    }

    /// Cluster by first keyword match against predefined category labels.
    fn cluster_by_keyword(&self, documents: Vec<ClusterDocument>) -> Vec<ResultCluster> {
        let categories = ["interview", "tutorial", "promo", "documentary", "music"];
        let mut map: HashMap<String, ResultCluster> = HashMap::new();
        for doc in documents {
            let label = doc
                .keywords
                .iter()
                .find_map(|kw| {
                    categories
                        .iter()
                        .find(|&&c| kw.to_lowercase().contains(c))
                        .map(|&c| c.to_string())
                })
                .unwrap_or_else(|| "general".to_string());
            map.entry(label.clone())
                .or_insert_with(|| ResultCluster::new(&label))
                .add_member(&doc.asset_id, doc.score);
        }
        map.into_values().collect()
    }

    /// Cluster by nearest centroid using Euclidean distance.
    fn cluster_by_centroid(&self, documents: Vec<ClusterDocument>) -> Vec<ResultCluster> {
        if documents.is_empty() {
            return Vec::new();
        }
        let k = self.config.max_clusters.min(documents.len());
        // Seed centroids from the first k documents.
        let mut centroids: Vec<Vec<f32>> =
            documents[..k].iter().map(|d| d.features.clone()).collect();

        let mut assignments = vec![0usize; documents.len()];

        // Run a fixed number of iterations (simplified k-means).
        for _ in 0..10 {
            // Assign each document to the nearest centroid.
            for (i, doc) in documents.iter().enumerate() {
                assignments[i] = nearest_centroid(&doc.features, &centroids);
            }
            // Update centroids.
            for (ci, centroid) in centroids.iter_mut().enumerate() {
                let members: Vec<&Vec<f32>> = documents
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| assignments[*i] == ci)
                    .map(|(_, d)| &d.features)
                    .collect();
                if !members.is_empty() {
                    *centroid = mean_vector(&members);
                }
            }
        }

        let mut clusters: Vec<ResultCluster> = (0..k)
            .map(|i| ResultCluster::new(format!("cluster-{i}")))
            .collect();
        for (i, doc) in documents.iter().enumerate() {
            clusters[assignments[i]].add_member(&doc.asset_id, doc.score);
        }
        clusters
    }

    /// Merge clusters smaller than `min_cluster_size` into "Other".
    fn merge_small_clusters(&self, mut clusters: Vec<ResultCluster>) -> Vec<ResultCluster> {
        let min = self.config.min_cluster_size;
        let mut other = ResultCluster::new("Other");
        clusters.retain(|c| {
            if c.size() < min {
                for m in &c.members {
                    other.add_member(m, c.avg_score);
                }
                false
            } else {
                true
            }
        });
        if !other.is_empty() {
            clusters.push(other);
        }
        clusters.sort_by(|a, b| {
            b.avg_score
                .partial_cmp(&a.avg_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        clusters
    }
}

impl Default for SearchClusterer {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns a human-readable MIME category label.
fn mime_category(mime: &str) -> String {
    if mime.starts_with("video/") {
        "Video".to_string()
    } else if mime.starts_with("audio/") {
        "Audio".to_string()
    } else if mime.starts_with("image/") {
        "Image".to_string()
    } else {
        "Document".to_string()
    }
}

/// Returns the index of the nearest centroid for a feature vector.
#[allow(clippy::cast_precision_loss)]
fn nearest_centroid(features: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, euclidean_sq(features, c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i)
}

/// Computes squared Euclidean distance between two vectors.
fn euclidean_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Computes the element-wise mean of a list of vectors.
#[allow(clippy::cast_precision_loss)]
fn mean_vector(vecs: &[&Vec<f32>]) -> Vec<f32> {
    if vecs.is_empty() {
        return Vec::new();
    }
    let len = vecs[0].len();
    let n = vecs.len() as f32;
    (0..len)
        .map(|i| {
            vecs.iter()
                .map(|v| v.get(i).copied().unwrap_or(0.0))
                .sum::<f32>()
                / n
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn video_doc(id: &str, score: f32) -> ClusterDocument {
        ClusterDocument::new(id, "video/mp4", score)
    }

    fn audio_doc(id: &str, score: f32) -> ClusterDocument {
        ClusterDocument::new(id, "audio/aac", score)
    }

    fn image_doc(id: &str, score: f32) -> ClusterDocument {
        ClusterDocument::new(id, "image/jpeg", score)
    }

    #[test]
    fn test_cluster_by_mime_groups_correctly() {
        let c = SearchClusterer::new();
        let docs = vec![
            video_doc("v1", 0.9),
            audio_doc("a1", 0.7),
            video_doc("v2", 0.8),
        ];
        let clusters = c.cluster(docs);
        let video_cluster = clusters.iter().find(|c| c.label == "Video");
        assert!(video_cluster.is_some());
        assert_eq!(video_cluster.expect("should succeed in test").size(), 2);
    }

    #[test]
    fn test_cluster_audio_group() {
        let c = SearchClusterer::new();
        let docs = vec![audio_doc("a1", 0.5), audio_doc("a2", 0.6)];
        let clusters = c.cluster(docs);
        let audio_cluster = clusters.iter().find(|cl| cl.label == "Audio");
        assert!(audio_cluster.is_some());
        assert_eq!(audio_cluster.expect("should succeed in test").size(), 2);
    }

    #[test]
    fn test_cluster_image_group() {
        let c = SearchClusterer::new();
        let docs = vec![image_doc("i1", 0.4)];
        let clusters = c.cluster(docs);
        let img_cluster = clusters.iter().find(|cl| cl.label == "Image");
        assert!(img_cluster.is_some());
    }

    #[test]
    fn test_cluster_sorted_by_avg_score_desc() {
        let c = SearchClusterer::new();
        let docs = vec![audio_doc("a1", 0.3), video_doc("v1", 0.9)];
        let clusters = c.cluster(docs);
        for w in clusters.windows(2) {
            assert!(w[0].avg_score >= w[1].avg_score);
        }
    }

    #[test]
    fn test_result_cluster_add_member_avg_score() {
        let mut cl = ResultCluster::new("test");
        cl.add_member("a", 0.8);
        cl.add_member("b", 0.2);
        assert!((cl.avg_score - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_result_cluster_size() {
        let mut cl = ResultCluster::new("test");
        assert_eq!(cl.size(), 0);
        cl.add_member("x", 0.5);
        assert_eq!(cl.size(), 1);
    }

    #[test]
    fn test_result_cluster_is_empty() {
        let cl = ResultCluster::new("empty");
        assert!(cl.is_empty());
    }

    #[test]
    fn test_empty_documents_returns_empty() {
        let c = SearchClusterer::new();
        let clusters = c.cluster(vec![]);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_merge_small_clusters_into_other() {
        let config = ClustererConfig {
            strategy: ClusterStrategy::MimeCategory,
            max_clusters: 6,
            min_cluster_size: 2,
        };
        let c = SearchClusterer::with_config(config);
        // One doc for image → should be merged into "Other".
        let docs = vec![
            image_doc("i1", 0.5),
            video_doc("v1", 0.8),
            video_doc("v2", 0.7),
        ];
        let clusters = c.cluster(docs);
        let other = clusters.iter().find(|cl| cl.label == "Other");
        assert!(other.is_some());
        assert_eq!(other.expect("should succeed in test").size(), 1);
    }

    #[test]
    fn test_keyword_cluster_strategy() {
        let config = ClustererConfig {
            strategy: ClusterStrategy::KeywordOverlap,
            max_clusters: 6,
            min_cluster_size: 1,
        };
        let c = SearchClusterer::with_config(config);
        let doc = ClusterDocument::new("d1", "video/mp4", 0.8)
            .with_keywords(vec!["interview".to_string()]);
        let clusters = c.cluster(vec![doc]);
        let interview = clusters.iter().find(|cl| cl.label == "interview");
        assert!(interview.is_some());
    }

    #[test]
    fn test_centroid_cluster_strategy() {
        let config = ClustererConfig {
            strategy: ClusterStrategy::NearestCentroid,
            max_clusters: 2,
            min_cluster_size: 1,
        };
        let c = SearchClusterer::with_config(config);
        let docs = vec![
            ClusterDocument::new("a", "video/mp4", 0.9).with_features(vec![0.0, 0.0]),
            ClusterDocument::new("b", "audio/aac", 0.8).with_features(vec![10.0, 10.0]),
            ClusterDocument::new("c", "image/jpeg", 0.7).with_features(vec![0.1, 0.1]),
        ];
        let clusters = c.cluster(docs);
        // Should produce 2 non-empty clusters.
        let non_empty: Vec<_> = clusters.iter().filter(|cl| !cl.is_empty()).collect();
        assert_eq!(non_empty.len(), 2);
    }

    #[test]
    fn test_mime_category_video() {
        assert_eq!(mime_category("video/mp4"), "Video");
    }

    #[test]
    fn test_mime_category_audio() {
        assert_eq!(mime_category("audio/wav"), "Audio");
    }

    #[test]
    fn test_mime_category_document() {
        assert_eq!(mime_category("application/pdf"), "Document");
    }

    #[test]
    fn test_euclidean_sq_same_vector() {
        assert!((euclidean_sq(&[1.0, 2.0], &[1.0, 2.0])).abs() < 1e-5);
    }
}
