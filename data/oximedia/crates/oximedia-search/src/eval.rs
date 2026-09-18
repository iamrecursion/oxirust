//! Lightweight, set-based Information Retrieval evaluation metrics.
//!
//! This module offers a *minimal, allocation-free* alternative to the richer
//! [`crate::ir_evaluation`] module. Where [`crate::ir_evaluation`] works against
//! a graded *qrel* judgement set ([`crate::ir_evaluation::RelevanceJudgements`])
//! and computes a whole bundle of metrics (NDCG, R-precision, MRR, …), the
//! functions here operate directly on:
//!
//! - a **ranked list** of document identifiers (rank 1 == first element), and
//! - a **set of relevant identifiers** ([`std::collections::HashSet`]).
//!
//! That makes them convenient building blocks for ad-hoc benchmarking, property
//! tests, and quick relevance checks where binary relevance is sufficient and a
//! full judgement matrix is overkill.
//!
//! All functions are generic over the identifier type `Id` (anything that is
//! [`Eq`] + [`Hash`]), so they work with `&str`, `String`,
//! `Uuid`, integer ids, etc.
//!
//! # Edge cases
//!
//! Every function is total — it never panics and never calls `unwrap`:
//!
//! - `k == 0` ⇒ precision/recall at k are `0.0`.
//! - `k > ranked.len()` ⇒ the effective cut-off is clamped to `ranked.len()`.
//! - empty `relevant` set ⇒ recall and average precision are `0.0`
//!   (there is nothing to retrieve), while precision is still well-defined.
//! - empty `ranked` list ⇒ all metrics are `0.0`.
//!
//! # Examples
//!
//! ```rust
//! use std::collections::HashSet;
//! use oximedia_search::eval::{precision_at_k, recall_at_k, average_precision};
//!
//! let ranked = ["a", "b", "c", "d", "e"];
//! let relevant: HashSet<&str> = ["a", "c"].into_iter().collect();
//!
//! // Relevant hits at ranks 1 and 3.
//! assert!((precision_at_k(&ranked, &relevant, 3) - 2.0 / 3.0).abs() < 1e-12);
//! assert!((recall_at_k(&ranked, &relevant, 5) - 1.0).abs() < 1e-12);
//! // AP = (1/1 + 2/3) / 2 = 0.8333…
//! assert!((average_precision(&ranked, &relevant) - 0.833_333_333_333).abs() < 1e-9);
//! ```

use std::collections::HashSet;
use std::hash::Hash;

/// Precision at rank cut-off `k`: the fraction of the top-`k` ranked items that
/// are relevant.
///
/// Formally `|{relevant} ∩ top_k| / k_eff`, where `k_eff = min(k, ranked.len())`.
///
/// Returns `0.0` when `k == 0` or `ranked` is empty (the cut-off contains no
/// documents, so precision is conventionally `0`).
///
/// # Examples
///
/// ```rust
/// use std::collections::HashSet;
/// use oximedia_search::eval::precision_at_k;
///
/// let ranked = ["a", "b", "c"];
/// let relevant: HashSet<&str> = ["a", "c"].into_iter().collect();
/// assert!((precision_at_k(&ranked, &relevant, 2) - 0.5).abs() < 1e-12);
/// ```
#[must_use]
pub fn precision_at_k<Id>(ranked: &[Id], relevant: &HashSet<Id>, k: usize) -> f64
where
    Id: Eq + Hash,
{
    let cutoff = k.min(ranked.len());
    if cutoff == 0 {
        return 0.0;
    }
    let hits = ranked[..cutoff]
        .iter()
        .filter(|id| relevant.contains(*id))
        .count();
    // `cutoff >= 1` here, so the division is safe and exact for small counts.
    hits as f64 / cutoff as f64
}

/// Recall at rank cut-off `k`: the fraction of all relevant items that appear in
/// the top-`k` ranked items.
///
/// Formally `|{relevant} ∩ top_k| / |relevant|`, where
/// `top_k = ranked[..min(k, ranked.len())]`.
///
/// Returns `0.0` when `relevant` is empty (there is nothing to recall) or when
/// `k == 0` / `ranked` is empty.
///
/// # Examples
///
/// ```rust
/// use std::collections::HashSet;
/// use oximedia_search::eval::recall_at_k;
///
/// let ranked = ["a", "b", "c", "d"];
/// let relevant: HashSet<&str> = ["a", "c", "x"].into_iter().collect();
/// // 2 of the 3 relevant ids are in the top 4 ("x" is absent).
/// assert!((recall_at_k(&ranked, &relevant, 4) - 2.0 / 3.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn recall_at_k<Id>(ranked: &[Id], relevant: &HashSet<Id>, k: usize) -> f64
where
    Id: Eq + Hash,
{
    let total = relevant.len();
    if total == 0 {
        return 0.0;
    }
    let cutoff = k.min(ranked.len());
    if cutoff == 0 {
        return 0.0;
    }
    let hits = ranked[..cutoff]
        .iter()
        .filter(|id| relevant.contains(*id))
        .count();
    hits as f64 / total as f64
}

/// Average Precision (AP) over the **full** ranked list.
///
/// AP is the mean of the precision evaluated at each rank where a relevant item
/// is retrieved, divided by the total number of relevant items:
///
/// ```text
/// AP = (1 / |relevant|) * Σ_{r : ranked[r] relevant} precision@(r+1)
/// ```
///
/// This is the canonical single-query summary of the precision-recall curve.
/// Relevant items that never appear in `ranked` still count toward `|relevant|`
/// in the denominator (so missing a relevant document lowers AP), matching the
/// standard TREC definition.
///
/// Returns `0.0` when `relevant` is empty or `ranked` is empty.
///
/// # Examples
///
/// ```rust
/// use std::collections::HashSet;
/// use oximedia_search::eval::average_precision;
///
/// // Relevant hits at ranks 1 and 3 of a length-5 ranking, |relevant| = 2.
/// // AP = (1/1 + 2/3) / 2 = 0.8333…
/// let ranked = ["a", "x", "b", "y", "z"];
/// let relevant: HashSet<&str> = ["a", "b"].into_iter().collect();
/// assert!((average_precision(&ranked, &relevant) - 0.833_333_333_333).abs() < 1e-9);
/// ```
#[must_use]
pub fn average_precision<Id>(ranked: &[Id], relevant: &HashSet<Id>) -> f64
where
    Id: Eq + Hash,
{
    let total = relevant.len();
    if total == 0 {
        return 0.0;
    }
    let mut hits = 0usize;
    let mut precision_sum = 0.0_f64;
    for (i, id) in ranked.iter().enumerate() {
        if relevant.contains(id) {
            hits += 1;
            // Precision at this relevant hit: hits so far / (1-based rank).
            precision_sum += hits as f64 / (i + 1) as f64;
        }
    }
    precision_sum / total as f64
}

/// Mean Average Precision (MAP) over a set of queries.
///
/// Each query is a `(ranked, relevant)` pair; MAP is the arithmetic mean of the
/// per-query [`average_precision`] values:
///
/// ```text
/// MAP = (1 / |queries|) * Σ_q AP(q)
/// ```
///
/// Returns `0.0` for an empty query set.
///
/// # Examples
///
/// ```rust
/// use std::collections::HashSet;
/// use oximedia_search::eval::mean_average_precision;
///
/// let q1 = (vec!["a", "b"], ["a"].into_iter().collect::<HashSet<_>>()); // AP = 1.0
/// let q2 = (vec!["x", "b"], ["b"].into_iter().collect::<HashSet<_>>()); // AP = 0.5
/// let queries = [q1, q2];
/// assert!((mean_average_precision(&queries) - 0.75).abs() < 1e-12);
/// ```
#[must_use]
pub fn mean_average_precision<Id>(queries: &[(Vec<Id>, HashSet<Id>)]) -> f64
where
    Id: Eq + Hash,
{
    let n = queries.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = queries
        .iter()
        .map(|(ranked, relevant)| average_precision(ranked, relevant))
        .sum();
    sum / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel<'a>(ids: &[&'a str]) -> HashSet<&'a str> {
        ids.iter().copied().collect()
    }

    // ── precision_at_k ──────────────────────────────────────────────────────

    #[test]
    fn precision_basic() {
        let ranked = ["a", "b", "c", "d", "e"];
        let relevant = rel(&["a", "c"]);
        // top-3 holds a (rel), b, c (rel) → 2/3.
        assert!((precision_at_k(&ranked, &relevant, 3) - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn precision_full_list() {
        let ranked = ["a", "b"];
        let relevant = rel(&["a", "b"]);
        assert!((precision_at_k(&ranked, &relevant, 2) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn precision_k_zero_is_zero() {
        let ranked = ["a", "b"];
        let relevant = rel(&["a"]);
        assert_eq!(precision_at_k(&ranked, &relevant, 0), 0.0);
    }

    #[test]
    fn precision_k_larger_than_len_clamps() {
        // k=10 but only 2 ranked items; one relevant → 1/2.
        let ranked = ["a", "b"];
        let relevant = rel(&["a"]);
        assert!((precision_at_k(&ranked, &relevant, 10) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn precision_empty_ranked_is_zero() {
        let ranked: [&str; 0] = [];
        let relevant = rel(&["a"]);
        assert_eq!(precision_at_k(&ranked, &relevant, 5), 0.0);
    }

    #[test]
    fn precision_empty_relevant_is_zero() {
        let ranked = ["a", "b"];
        let relevant: HashSet<&str> = HashSet::new();
        assert_eq!(precision_at_k(&ranked, &relevant, 2), 0.0);
    }

    // ── recall_at_k ─────────────────────────────────────────────────────────

    #[test]
    fn recall_basic() {
        let ranked = ["a", "b", "c", "d"];
        let relevant = rel(&["a", "c", "x"]); // "x" never retrieved
        assert!((recall_at_k(&ranked, &relevant, 4) - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn recall_full_when_all_found() {
        let ranked = ["a", "b", "c"];
        let relevant = rel(&["a", "c"]);
        assert!((recall_at_k(&ranked, &relevant, 3) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn recall_cutoff_limits_hits() {
        // relevant "c" is at rank 3, but k=2 excludes it → 1/2.
        let ranked = ["a", "b", "c"];
        let relevant = rel(&["a", "c"]);
        assert!((recall_at_k(&ranked, &relevant, 2) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn recall_empty_relevant_is_zero() {
        let ranked = ["a", "b"];
        let relevant: HashSet<&str> = HashSet::new();
        assert_eq!(recall_at_k(&ranked, &relevant, 2), 0.0);
    }

    #[test]
    fn recall_k_zero_is_zero() {
        let ranked = ["a", "b"];
        let relevant = rel(&["a"]);
        assert_eq!(recall_at_k(&ranked, &relevant, 0), 0.0);
    }

    // ── average_precision ───────────────────────────────────────────────────

    #[test]
    fn ap_hits_at_1_and_3() {
        // The canonical golden value from the task description.
        let ranked = ["a", "x", "b", "y", "z"];
        let relevant = rel(&["a", "b"]);
        // AP = (1/1 + 2/3) / 2 = 0.8333…
        assert!((average_precision(&ranked, &relevant) - 0.833_333_333_333).abs() < 1e-9);
    }

    #[test]
    fn ap_perfect_ranking_is_one() {
        let ranked = ["a", "b", "c"];
        let relevant = rel(&["a", "b"]);
        // AP = (1/1 + 2/2) / 2 = 1.0
        assert!((average_precision(&ranked, &relevant) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ap_missing_relevant_lowers_score() {
        // Two relevant, only one retrieved (at rank 1); the missing one still
        // counts in the denominator: AP = (1/1) / 2 = 0.5.
        let ranked = ["a", "x", "y"];
        let relevant = rel(&["a", "missing"]);
        assert!((average_precision(&ranked, &relevant) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn ap_empty_relevant_is_zero() {
        let ranked = ["a", "b"];
        let relevant: HashSet<&str> = HashSet::new();
        assert_eq!(average_precision(&ranked, &relevant), 0.0);
    }

    #[test]
    fn ap_empty_ranked_is_zero() {
        let ranked: [&str; 0] = [];
        let relevant = rel(&["a"]);
        assert_eq!(average_precision(&ranked, &relevant), 0.0);
    }

    // ── mean_average_precision ──────────────────────────────────────────────

    #[test]
    fn map_two_queries() {
        let queries = [
            (vec!["a", "b"], rel(&["a"])), // AP = 1.0
            (vec!["x", "b"], rel(&["b"])), // AP = 1/2 = 0.5
        ];
        assert!((mean_average_precision(&queries) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn map_empty_query_set_is_zero() {
        let queries: [(Vec<&str>, HashSet<&str>); 0] = [];
        assert_eq!(mean_average_precision(&queries), 0.0);
    }

    #[test]
    fn map_generic_over_integer_ids() {
        // Confirms the generic bound works for non-string ids too.
        let q1: (Vec<u32>, HashSet<u32>) = (vec![1, 2, 3], [1, 3].into_iter().collect());
        // hits at ranks 1 and 3 → AP = (1/1 + 2/3)/2 = 0.8333…
        let queries = [q1];
        assert!((mean_average_precision(&queries) - 0.833_333_333_333).abs() < 1e-9);
    }
}
