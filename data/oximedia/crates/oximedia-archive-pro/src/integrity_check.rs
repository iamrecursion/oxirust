//! Archive integrity monitoring: scheduled verification, hash trees, and
//! anomaly detection.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// The algorithm used for integrity hashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum HashAlgorithm {
    /// MD5 – fast, weaker collision resistance
    Md5,
    /// SHA-256
    Sha256,
    /// SHA-512
    Sha512,
    /// BLAKE3 – fast and cryptographically strong
    Blake3,
}

impl HashAlgorithm {
    /// Digest length in bytes.
    #[must_use]
    pub fn digest_bytes(&self) -> usize {
        match self {
            Self::Md5 => 16,
            Self::Sha256 => 32,
            Self::Sha512 => 64,
            Self::Blake3 => 32,
        }
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Md5 => "MD5",
            Self::Sha256 => "SHA-256",
            Self::Sha512 => "SHA-512",
            Self::Blake3 => "BLAKE3",
        }
    }
}

/// Represents a node in a Merkle hash tree.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HashNode {
    /// Hex-encoded digest of this node
    pub digest: String,
    /// Left child index (None for leaves)
    pub left: Option<usize>,
    /// Right child index (None for leaves)
    pub right: Option<usize>,
    /// True if this is a leaf node
    pub is_leaf: bool,
}

/// A simple binary Merkle tree over asset chunk digests.
#[derive(Debug, Clone)]
pub struct MerkleTree {
    nodes: Vec<HashNode>,
    algorithm: HashAlgorithm,
}

impl MerkleTree {
    /// Build a Merkle tree from a list of leaf digests.
    /// Leaf digests must be hex strings.
    #[must_use]
    pub fn from_leaves(leaves: Vec<String>, algorithm: HashAlgorithm) -> Self {
        if leaves.is_empty() {
            return Self {
                nodes: vec![],
                algorithm,
            };
        }

        let mut nodes: Vec<HashNode> = leaves
            .into_iter()
            .map(|d| HashNode {
                digest: d,
                left: None,
                right: None,
                is_leaf: true,
            })
            .collect();

        let mut level_start = 0usize;
        let mut level_len = nodes.len();

        while level_len > 1 {
            let next_level_start = nodes.len();
            let mut i = 0;
            while i < level_len {
                let left_idx = level_start + i;
                let right_idx = if i + 1 < level_len {
                    level_start + i + 1
                } else {
                    // Odd leaf – duplicate
                    level_start + i
                };
                let combined = format!("{}{}", nodes[left_idx].digest, nodes[right_idx].digest);
                // Use a simple deterministic "hash" for the tree structure
                // In production this would use the actual hash algorithm.
                let parent_digest = format!("{:016x}", fxhash(&combined));
                nodes.push(HashNode {
                    digest: parent_digest,
                    left: Some(left_idx),
                    right: Some(right_idx),
                    is_leaf: false,
                });
                i += 2;
            }
            let new_level_len = nodes.len() - next_level_start;
            level_start = next_level_start;
            level_len = new_level_len;
        }

        Self { nodes, algorithm }
    }

    /// Root digest, or empty string if tree is empty.
    #[must_use]
    pub fn root_digest(&self) -> &str {
        self.nodes.last().map_or("", |n| &n.digest)
    }

    /// Number of nodes in the tree.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Algorithm used.
    #[must_use]
    pub fn algorithm(&self) -> HashAlgorithm {
        self.algorithm
    }

    /// Number of leaf nodes.
    #[must_use]
    pub fn leaf_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf).count()
    }
}

/// A trivial polynomial hash for deterministic test purposes only.
fn fxhash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Result of a single integrity verification run.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Asset identifier
    pub asset_id: String,
    /// Whether the check passed
    pub passed: bool,
    /// Details about failures, empty if passed
    pub failures: Vec<String>,
    /// Unix timestamp of check
    pub checked_at: u64,
}

impl VerificationResult {
    /// Create a passing result.
    #[must_use]
    pub fn pass(asset_id: impl Into<String>, checked_at: u64) -> Self {
        Self {
            asset_id: asset_id.into(),
            passed: true,
            failures: vec![],
            checked_at,
        }
    }

    /// Create a failing result.
    #[must_use]
    pub fn fail(asset_id: impl Into<String>, failures: Vec<String>, checked_at: u64) -> Self {
        Self {
            asset_id: asset_id.into(),
            passed: false,
            failures,
            checked_at,
        }
    }
}

/// Schedule configuration for periodic integrity checks.
#[derive(Debug, Clone)]
pub struct VerificationSchedule {
    /// Interval in seconds between checks
    pub interval_secs: u64,
    /// Maximum allowed consecutive failures before alerting
    pub max_consecutive_failures: u32,
}

impl VerificationSchedule {
    /// Default schedule: daily checks, alert after 3 failures.
    #[must_use]
    pub fn daily() -> Self {
        Self {
            interval_secs: 86_400,
            max_consecutive_failures: 3,
        }
    }

    /// Weekly schedule.
    #[must_use]
    pub fn weekly() -> Self {
        Self {
            interval_secs: 7 * 86_400,
            max_consecutive_failures: 2,
        }
    }

    /// Whether a check is due given last check and current time.
    #[must_use]
    pub fn is_due(&self, last_checked: u64, now: u64) -> bool {
        now >= last_checked + self.interval_secs
    }
}

/// Anomaly detector over a series of verification results.
#[derive(Debug, Default)]
pub struct AnomalyDetector {
    /// asset_id → list of (timestamp, passed)
    history: HashMap<String, Vec<(u64, bool)>>,
}

impl AnomalyDetector {
    /// Create a new detector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a verification result.
    pub fn record(&mut self, result: &VerificationResult) {
        self.history
            .entry(result.asset_id.clone())
            .or_default()
            .push((result.checked_at, result.passed));
    }

    /// Current consecutive failure count for an asset.
    #[must_use]
    pub fn consecutive_failures(&self, asset_id: &str) -> u32 {
        let Some(hist) = self.history.get(asset_id) else {
            return 0;
        };
        hist.iter().rev().take_while(|(_, passed)| !passed).count() as u32
    }

    /// Assets with consecutive failures exceeding the given threshold.
    #[must_use]
    pub fn anomalous_assets(&self, threshold: u32) -> Vec<&str> {
        self.history
            .keys()
            .filter(|id| self.consecutive_failures(id) >= threshold)
            .map(String::as_str)
            .collect()
    }

    /// Overall pass rate across all recorded results.
    #[must_use]
    pub fn global_pass_rate(&self) -> f64 {
        let total: usize = self.history.values().map(|v| v.len()).sum();
        if total == 0 {
            return 1.0;
        }
        let passed: usize = self
            .history
            .values()
            .flat_map(|v| v.iter())
            .filter(|(_, ok)| *ok)
            .count();
        passed as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_algorithm_digest_bytes() {
        assert_eq!(HashAlgorithm::Md5.digest_bytes(), 16);
        assert_eq!(HashAlgorithm::Sha256.digest_bytes(), 32);
        assert_eq!(HashAlgorithm::Sha512.digest_bytes(), 64);
        assert_eq!(HashAlgorithm::Blake3.digest_bytes(), 32);
    }

    #[test]
    fn test_hash_algorithm_names() {
        assert_eq!(HashAlgorithm::Blake3.name(), "BLAKE3");
        assert_eq!(HashAlgorithm::Sha256.name(), "SHA-256");
    }

    #[test]
    fn test_merkle_tree_single_leaf() {
        let tree = MerkleTree::from_leaves(vec!["abcd1234".into()], HashAlgorithm::Blake3);
        assert_eq!(tree.leaf_count(), 1);
        assert_eq!(tree.root_digest(), "abcd1234");
    }

    #[test]
    fn test_merkle_tree_two_leaves() {
        let tree = MerkleTree::from_leaves(vec!["aaa".into(), "bbb".into()], HashAlgorithm::Sha256);
        assert_eq!(tree.leaf_count(), 2);
        // Root should be a 16-char hex string produced by fxhash
        assert_eq!(tree.root_digest().len(), 16);
    }

    #[test]
    fn test_merkle_tree_odd_leaves() {
        let leaves = vec!["a".into(), "b".into(), "c".into()];
        let tree = MerkleTree::from_leaves(leaves, HashAlgorithm::Sha256);
        assert_eq!(tree.leaf_count(), 3);
        // Root must be non-empty
        assert!(!tree.root_digest().is_empty());
    }

    #[test]
    fn test_merkle_tree_empty() {
        let tree = MerkleTree::from_leaves(vec![], HashAlgorithm::Sha256);
        assert_eq!(tree.node_count(), 0);
        assert_eq!(tree.root_digest(), "");
    }

    #[test]
    fn test_merkle_tree_algorithm() {
        let tree = MerkleTree::from_leaves(vec!["x".into()], HashAlgorithm::Md5);
        assert_eq!(tree.algorithm(), HashAlgorithm::Md5);
    }

    #[test]
    fn test_verification_result_pass() {
        let r = VerificationResult::pass("asset-1", 9999);
        assert!(r.passed);
        assert!(r.failures.is_empty());
    }

    #[test]
    fn test_verification_result_fail() {
        let r = VerificationResult::fail("asset-2", vec!["chunk 3 mismatch".into()], 9999);
        assert!(!r.passed);
        assert_eq!(r.failures.len(), 1);
    }

    #[test]
    fn test_schedule_is_due() {
        let sched = VerificationSchedule::daily();
        assert!(sched.is_due(0, 86_400));
        assert!(!sched.is_due(0, 86_399));
    }

    #[test]
    fn test_schedule_weekly() {
        let sched = VerificationSchedule::weekly();
        assert_eq!(sched.interval_secs, 7 * 86_400);
    }

    #[test]
    fn test_anomaly_detector_consecutive_failures() {
        let mut det = AnomalyDetector::new();
        det.record(&VerificationResult::pass("a", 1));
        det.record(&VerificationResult::fail("a", vec!["err".into()], 2));
        det.record(&VerificationResult::fail("a", vec!["err".into()], 3));
        assert_eq!(det.consecutive_failures("a"), 2);
    }

    #[test]
    fn test_anomaly_detector_no_history() {
        let det = AnomalyDetector::new();
        assert_eq!(det.consecutive_failures("unknown"), 0);
    }

    #[test]
    fn test_anomaly_detector_anomalous_assets() {
        let mut det = AnomalyDetector::new();
        for i in 0..3u64 {
            det.record(&VerificationResult::fail("bad-asset", vec!["x".into()], i));
        }
        let anomalous = det.anomalous_assets(3);
        assert!(anomalous.contains(&"bad-asset"));
    }

    #[test]
    fn test_anomaly_detector_global_pass_rate() {
        let mut det = AnomalyDetector::new();
        det.record(&VerificationResult::pass("x", 1));
        det.record(&VerificationResult::pass("x", 2));
        det.record(&VerificationResult::fail("x", vec!["e".into()], 3));
        let rate = det.global_pass_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_anomaly_detector_empty_pass_rate() {
        let det = AnomalyDetector::new();
        assert!((det.global_pass_rate() - 1.0).abs() < 1e-9);
    }
}
