//! Offline evaluation metrics for recommendation quality.
//!
//! Provides standard information retrieval metrics for evaluating the quality
//! of recommendation lists against ground truth relevant items:
//!
//! - **Precision@K**: fraction of recommended items that are relevant
//! - **Recall@K**: fraction of relevant items that are recommended
//! - **NDCG@K**: Normalised Discounted Cumulative Gain
//! - **MAP**: Mean Average Precision
//! - **Hit Rate@K**: whether at least one relevant item appears in top-K
//! - **MRR**: Mean Reciprocal Rank

use std::collections::HashSet;

/// A single evaluation instance: a list of recommended item IDs and
/// the set of ground-truth relevant item IDs for one user/query.
#[derive(Debug, Clone)]
pub struct EvalInstance {
    /// Ordered list of recommended item IDs (most relevant first).
    pub recommended: Vec<String>,
    /// Set of ground-truth relevant item IDs.
    pub relevant: HashSet<String>,
}

impl EvalInstance {
    /// Create a new evaluation instance.
    pub fn new(recommended: Vec<String>, relevant: HashSet<String>) -> Self {
        Self {
            recommended,
            relevant,
        }
    }

    /// Convenience: create from slices of &str.
    pub fn from_strs(recommended: &[&str], relevant: &[&str]) -> Self {
        Self {
            recommended: recommended.iter().map(|s| s.to_string()).collect(),
            relevant: relevant.iter().map(|s| s.to_string()).collect(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-instance metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Precision@K for a single instance.
///
/// Precision@K = |{recommended ∩ relevant}| / K
#[must_use]
pub fn precision_at_k(instance: &EvalInstance, k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let top_k = instance.recommended.iter().take(k);
    let hits = top_k
        .filter(|item| instance.relevant.contains(*item))
        .count();
    hits as f64 / k as f64
}

/// Compute Recall@K for a single instance.
///
/// Recall@K = |{recommended ∩ relevant}| / |relevant|
#[must_use]
pub fn recall_at_k(instance: &EvalInstance, k: usize) -> f64 {
    if instance.relevant.is_empty() {
        return 0.0;
    }
    let top_k = instance.recommended.iter().take(k);
    let hits = top_k
        .filter(|item| instance.relevant.contains(*item))
        .count();
    hits as f64 / instance.relevant.len() as f64
}

/// Compute NDCG@K (Normalised Discounted Cumulative Gain) for a single instance.
///
/// Uses binary relevance: rel(i) = 1 if item is relevant, 0 otherwise.
///
/// DCG@K = Σ_{i=1}^{K} rel(i) / log2(i + 1)
/// IDCG@K = DCG of the ideal ranking (all relevant items first)
/// NDCG@K = DCG@K / IDCG@K
#[must_use]
pub fn ndcg_at_k(instance: &EvalInstance, k: usize) -> f64 {
    if instance.relevant.is_empty() || k == 0 {
        return 0.0;
    }

    // DCG
    let mut dcg = 0.0;
    for (i, item) in instance.recommended.iter().take(k).enumerate() {
        if instance.relevant.contains(item) {
            dcg += 1.0 / ((i as f64) + 2.0).log2();
        }
    }

    // IDCG: best possible DCG with min(|relevant|, k) relevant items at top
    let ideal_count = instance.relevant.len().min(k);
    let mut idcg = 0.0;
    for i in 0..ideal_count {
        idcg += 1.0 / ((i as f64) + 2.0).log2();
    }

    if idcg < f64::EPSILON {
        return 0.0;
    }

    dcg / idcg
}

/// Compute Average Precision for a single instance.
///
/// AP = (1/|relevant|) * Σ_{k=1}^{n} (Precision@k * rel(k))
#[must_use]
pub fn average_precision(instance: &EvalInstance) -> f64 {
    if instance.relevant.is_empty() {
        return 0.0;
    }

    let mut hits = 0;
    let mut sum_precision = 0.0;

    for (i, item) in instance.recommended.iter().enumerate() {
        if instance.relevant.contains(item) {
            hits += 1;
            sum_precision += hits as f64 / (i + 1) as f64;
        }
    }

    sum_precision / instance.relevant.len() as f64
}

/// Compute Hit Rate@K for a single instance (1.0 if any relevant item in top-K, else 0.0).
#[must_use]
pub fn hit_rate_at_k(instance: &EvalInstance, k: usize) -> f64 {
    let has_hit = instance
        .recommended
        .iter()
        .take(k)
        .any(|item| instance.relevant.contains(item));
    if has_hit {
        1.0
    } else {
        0.0
    }
}

/// Compute Reciprocal Rank for a single instance.
///
/// RR = 1 / rank of first relevant item (0.0 if none found).
#[must_use]
pub fn reciprocal_rank(instance: &EvalInstance) -> f64 {
    for (i, item) in instance.recommended.iter().enumerate() {
        if instance.relevant.contains(item) {
            return 1.0 / (i + 1) as f64;
        }
    }
    0.0
}

// ─────────────────────────────────────────────────────────────────────────────
// Aggregated (mean) metrics over multiple instances
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Mean Average Precision (MAP) over multiple instances.
#[must_use]
pub fn mean_average_precision(instances: &[EvalInstance]) -> f64 {
    if instances.is_empty() {
        return 0.0;
    }
    let total: f64 = instances.iter().map(average_precision).sum();
    total / instances.len() as f64
}

/// Compute mean Precision@K over multiple instances.
#[must_use]
pub fn mean_precision_at_k(instances: &[EvalInstance], k: usize) -> f64 {
    if instances.is_empty() {
        return 0.0;
    }
    let total: f64 = instances.iter().map(|inst| precision_at_k(inst, k)).sum();
    total / instances.len() as f64
}

/// Compute mean Recall@K over multiple instances.
#[must_use]
pub fn mean_recall_at_k(instances: &[EvalInstance], k: usize) -> f64 {
    if instances.is_empty() {
        return 0.0;
    }
    let total: f64 = instances.iter().map(|inst| recall_at_k(inst, k)).sum();
    total / instances.len() as f64
}

/// Compute mean NDCG@K over multiple instances.
#[must_use]
pub fn mean_ndcg_at_k(instances: &[EvalInstance], k: usize) -> f64 {
    if instances.is_empty() {
        return 0.0;
    }
    let total: f64 = instances.iter().map(|inst| ndcg_at_k(inst, k)).sum();
    total / instances.len() as f64
}

/// Compute Mean Reciprocal Rank (MRR) over multiple instances.
#[must_use]
pub fn mean_reciprocal_rank(instances: &[EvalInstance]) -> f64 {
    if instances.is_empty() {
        return 0.0;
    }
    let total: f64 = instances.iter().map(reciprocal_rank).sum();
    total / instances.len() as f64
}

/// Compute mean Hit Rate@K over multiple instances.
#[must_use]
pub fn mean_hit_rate_at_k(instances: &[EvalInstance], k: usize) -> f64 {
    if instances.is_empty() {
        return 0.0;
    }
    let total: f64 = instances.iter().map(|inst| hit_rate_at_k(inst, k)).sum();
    total / instances.len() as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluation report
// ─────────────────────────────────────────────────────────────────────────────

/// Comprehensive evaluation report.
#[derive(Debug, Clone)]
pub struct EvaluationReport {
    /// Precision@K for various K values.
    pub precision: Vec<(usize, f64)>,
    /// Recall@K for various K values.
    pub recall: Vec<(usize, f64)>,
    /// NDCG@K for various K values.
    pub ndcg: Vec<(usize, f64)>,
    /// Mean Average Precision.
    pub map: f64,
    /// Mean Reciprocal Rank.
    pub mrr: f64,
    /// Hit Rate@K for various K values.
    pub hit_rate: Vec<(usize, f64)>,
    /// Number of evaluation instances.
    pub num_instances: usize,
}

/// Generate a full evaluation report over multiple instances.
///
/// Computes all metrics at k = 1, 3, 5, 10, 20.
#[must_use]
pub fn generate_report(instances: &[EvalInstance]) -> EvaluationReport {
    let k_values = [1, 3, 5, 10, 20];

    let precision: Vec<(usize, f64)> = k_values
        .iter()
        .map(|&k| (k, mean_precision_at_k(instances, k)))
        .collect();

    let recall: Vec<(usize, f64)> = k_values
        .iter()
        .map(|&k| (k, mean_recall_at_k(instances, k)))
        .collect();

    let ndcg: Vec<(usize, f64)> = k_values
        .iter()
        .map(|&k| (k, mean_ndcg_at_k(instances, k)))
        .collect();

    let hit_rate: Vec<(usize, f64)> = k_values
        .iter()
        .map(|&k| (k, mean_hit_rate_at_k(instances, k)))
        .collect();

    EvaluationReport {
        precision,
        recall,
        ndcg,
        map: mean_average_precision(instances),
        mrr: mean_reciprocal_rank(instances),
        hit_rate,
        num_instances: instances.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_instance(rec: &[&str], rel: &[&str]) -> EvalInstance {
        EvalInstance::from_strs(rec, rel)
    }

    // ---- Precision@K ----

    #[test]
    fn test_precision_at_k_perfect() {
        let inst = make_instance(&["a", "b", "c"], &["a", "b", "c"]);
        assert!((precision_at_k(&inst, 3) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_precision_at_k_half() {
        let inst = make_instance(&["a", "x", "b", "y"], &["a", "b"]);
        assert!((precision_at_k(&inst, 4) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_precision_at_k_zero() {
        let inst = make_instance(&["x", "y", "z"], &["a", "b"]);
        assert!((precision_at_k(&inst, 3)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_precision_at_k_zero_k() {
        let inst = make_instance(&["a"], &["a"]);
        assert!((precision_at_k(&inst, 0)).abs() < f64::EPSILON);
    }

    // ---- Recall@K ----

    #[test]
    fn test_recall_at_k_perfect() {
        let inst = make_instance(&["a", "b"], &["a", "b"]);
        assert!((recall_at_k(&inst, 2) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_recall_at_k_partial() {
        let inst = make_instance(&["a", "x"], &["a", "b"]);
        assert!((recall_at_k(&inst, 2) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_recall_at_k_empty_relevant() {
        let inst = make_instance(&["a"], &[]);
        assert!((recall_at_k(&inst, 1)).abs() < f64::EPSILON);
    }

    // ---- NDCG@K ----

    #[test]
    fn test_ndcg_perfect_ranking() {
        let inst = make_instance(&["a", "b", "c"], &["a", "b", "c"]);
        assert!((ndcg_at_k(&inst, 3) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ndcg_worst_ranking() {
        // All relevant items are outside top-K
        let inst = make_instance(&["x", "y", "z", "a", "b"], &["a", "b"]);
        assert!((ndcg_at_k(&inst, 3)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ndcg_partial_ranking() {
        // Relevant item at position 2 (0-indexed)
        let inst = make_instance(&["x", "y", "a"], &["a"]);
        let ndcg = ndcg_at_k(&inst, 3);
        // DCG = 1/log2(4) = 0.5; IDCG = 1/log2(2) = 1.0 → NDCG = 0.5
        assert!((ndcg - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_ndcg_empty() {
        let inst = make_instance(&[], &["a"]);
        assert!((ndcg_at_k(&inst, 5)).abs() < f64::EPSILON);
    }

    // ---- Average Precision ----

    #[test]
    fn test_average_precision_perfect() {
        let inst = make_instance(&["a", "b", "c"], &["a", "b", "c"]);
        // AP = (1/3)(1/1 + 2/2 + 3/3) = 1.0
        assert!((average_precision(&inst) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_average_precision_one_relevant() {
        let inst = make_instance(&["x", "y", "a"], &["a"]);
        // AP = (1/1)(1/3) = 1/3
        assert!((average_precision(&inst) - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_average_precision_empty_relevant() {
        let inst = make_instance(&["a", "b"], &[]);
        assert!((average_precision(&inst)).abs() < f64::EPSILON);
    }

    // ---- Hit Rate@K ----

    #[test]
    fn test_hit_rate_hit() {
        let inst = make_instance(&["x", "a", "y"], &["a"]);
        assert!((hit_rate_at_k(&inst, 3) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hit_rate_miss() {
        let inst = make_instance(&["x", "y", "z"], &["a"]);
        assert!((hit_rate_at_k(&inst, 3)).abs() < f64::EPSILON);
    }

    // ---- Reciprocal Rank ----

    #[test]
    fn test_reciprocal_rank_first() {
        let inst = make_instance(&["a", "b", "c"], &["a"]);
        assert!((reciprocal_rank(&inst) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reciprocal_rank_third() {
        let inst = make_instance(&["x", "y", "a"], &["a"]);
        assert!((reciprocal_rank(&inst) - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_reciprocal_rank_none() {
        let inst = make_instance(&["x", "y", "z"], &["a"]);
        assert!((reciprocal_rank(&inst)).abs() < f64::EPSILON);
    }

    // ---- Mean metrics ----

    #[test]
    fn test_mean_average_precision() {
        let instances = vec![
            make_instance(&["a", "b", "c"], &["a", "b", "c"]),
            make_instance(&["x", "y", "a"], &["a"]),
        ];
        let map = mean_average_precision(&instances);
        // User1 AP = 1.0, User2 AP = 1/3 → MAP = 2/3
        assert!((map - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_precision_at_k() {
        let instances = vec![
            make_instance(&["a", "x"], &["a"]),
            make_instance(&["a", "b"], &["a", "b"]),
        ];
        // P@2 user1 = 0.5, P@2 user2 = 1.0 → mean = 0.75
        assert!((mean_precision_at_k(&instances, 2) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_mean_reciprocal_rank() {
        let instances = vec![
            make_instance(&["a", "b"], &["a"]), // RR = 1
            make_instance(&["x", "a"], &["a"]), // RR = 0.5
        ];
        assert!((mean_reciprocal_rank(&instances) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_mean_metrics_empty() {
        let instances: Vec<EvalInstance> = vec![];
        assert!((mean_average_precision(&instances)).abs() < f64::EPSILON);
        assert!((mean_precision_at_k(&instances, 5)).abs() < f64::EPSILON);
        assert!((mean_recall_at_k(&instances, 5)).abs() < f64::EPSILON);
        assert!((mean_ndcg_at_k(&instances, 5)).abs() < f64::EPSILON);
        assert!((mean_reciprocal_rank(&instances)).abs() < f64::EPSILON);
        assert!((mean_hit_rate_at_k(&instances, 5)).abs() < f64::EPSILON);
    }

    // ---- Report ----

    #[test]
    fn test_generate_report() {
        let instances = vec![
            make_instance(&["a", "b", "c", "d", "e"], &["a", "c"]),
            make_instance(&["x", "y", "a", "b", "c"], &["a", "b"]),
        ];
        let report = generate_report(&instances);
        assert_eq!(report.num_instances, 2);
        assert!(!report.precision.is_empty());
        assert!(!report.recall.is_empty());
        assert!(!report.ndcg.is_empty());
        assert!(!report.hit_rate.is_empty());
        assert!(report.map >= 0.0);
        assert!(report.mrr >= 0.0);
    }

    #[test]
    fn test_generate_report_empty() {
        let report = generate_report(&[]);
        assert_eq!(report.num_instances, 0);
        assert!((report.map).abs() < f64::EPSILON);
    }

    #[test]
    fn test_eval_instance_new() {
        let rec = vec!["a".to_string(), "b".to_string()];
        let mut rel = HashSet::new();
        rel.insert("a".to_string());
        let inst = EvalInstance::new(rec, rel);
        assert_eq!(inst.recommended.len(), 2);
        assert_eq!(inst.relevant.len(), 1);
    }
}
