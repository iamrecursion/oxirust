//! Relevance ranking — TF-IDF, BM25 and result sorting utilities.
#![allow(dead_code)]

// ── BM25Params ────────────────────────────────────────────────────────────────

/// Parameters for the BM25 ranking function.
#[derive(Debug, Clone)]
pub struct BM25Params {
    /// Term-frequency saturation parameter (typical range 1.2–2.0).
    pub k1: f32,
    /// Field-length normalisation parameter (0.0 = no normalisation, 1.0 = full).
    pub b: f32,
    /// Average document length in tokens (corpus-level statistic).
    pub avg_doc_len: f32,
}

impl BM25Params {
    /// Recommended defaults from the BM25 literature.
    #[must_use]
    pub fn default_params() -> Self {
        Self {
            k1: 1.5,
            b: 0.75,
            avg_doc_len: 100.0,
        }
    }

    /// Compute the BM25 term weight for a single term occurrence.
    ///
    /// - `tf` — raw term frequency in the document.
    /// - `idf` — inverse document frequency (natural log-based).
    /// - `doc_len` — number of tokens in the document.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn score(&self, tf: f32, idf: f32, doc_len: usize) -> f32 {
        let dl = doc_len as f32;
        let normalised_tf = (tf * (self.k1 + 1.0))
            / (tf + self.k1 * (1.0 - self.b + self.b * dl / self.avg_doc_len));
        idf * normalised_tf
    }
}

impl Default for BM25Params {
    fn default() -> Self {
        Self::default_params()
    }
}

// ── TfIdfScore ────────────────────────────────────────────────────────────────

/// Helper for computing TF-IDF scores.
#[derive(Debug, Clone, Default)]
pub struct TfIdfScore;

impl TfIdfScore {
    /// Compute term frequency (TF) for a term appearing `term_count` times
    /// in a document of length `doc_len`.
    ///
    /// Uses raw-count normalisation: `TF = count / doc_len`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn compute_tf(term_count: usize, doc_len: usize) -> f32 {
        if doc_len == 0 {
            return 0.0;
        }
        term_count as f32 / doc_len as f32
    }

    /// Compute inverse document frequency (IDF) using the standard log formula:
    /// `IDF = ln(N / df + 1)`.
    ///
    /// - `num_docs` — total number of documents in the corpus.
    /// - `doc_freq` — number of documents containing the term.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn compute_idf(num_docs: usize, doc_freq: usize) -> f32 {
        if num_docs == 0 || doc_freq == 0 {
            return 0.0;
        }
        ((num_docs as f32 / doc_freq as f32) + 1.0).ln()
    }

    /// Compute the final TF-IDF score.
    #[must_use]
    pub fn score(tf: f32, idf: f32) -> f32 {
        tf * idf
    }

    /// Convenience: compute TF-IDF in one call.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn compute(term_count: usize, doc_len: usize, num_docs: usize, doc_freq: usize) -> f32 {
        let tf = Self::compute_tf(term_count, doc_len);
        let idf = Self::compute_idf(num_docs, doc_freq);
        Self::score(tf, idf)
    }
}

// ── RankResult ────────────────────────────────────────────────────────────────

/// A scored search result ready for ranking.
#[derive(Debug, Clone)]
pub struct RankResult {
    /// Document identifier.
    pub doc_id: String,
    /// Relevance score (higher = better).
    pub score: f32,
    /// Optional human-readable title for display.
    pub title: Option<String>,
}

impl RankResult {
    /// Create a new ranked result.
    #[must_use]
    pub fn new(doc_id: impl Into<String>, score: f32) -> Self {
        Self {
            doc_id: doc_id.into(),
            score,
            title: None,
        }
    }

    /// Attach a title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Sort a mutable slice of results by descending score in-place.
    pub fn rank_by_score(results: &mut [RankResult]) {
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Return a new `Vec` of results sorted by descending score.
    #[must_use]
    pub fn ranked(mut results: Vec<RankResult>) -> Vec<RankResult> {
        Self::rank_by_score(&mut results);
        results
    }

    /// Apply a multiplicative boost to the score.
    pub fn boost(&mut self, factor: f32) {
        self.score *= factor;
    }

    /// Normalise scores in a slice so the highest score becomes 1.0.
    pub fn normalise(results: &mut [RankResult]) {
        let max = results
            .iter()
            .map(|r| r.score)
            .fold(f32::NEG_INFINITY, f32::max);
        if max > 0.0 {
            for r in results.iter_mut() {
                r.score /= max;
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // TF-IDF ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_compute_tf_basic() {
        let tf = TfIdfScore::compute_tf(3, 10);
        assert!((tf - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_compute_tf_zero_doc_len() {
        let tf = TfIdfScore::compute_tf(5, 0);
        assert!((tf - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_idf_basic() {
        // ln(100/1 + 1) = ln(101) ≈ 4.615
        let idf = TfIdfScore::compute_idf(100, 1);
        assert!((idf - 101_f32.ln()).abs() < 1e-4);
    }

    #[test]
    fn test_compute_idf_zero_docs() {
        let idf = TfIdfScore::compute_idf(0, 0);
        assert!((idf - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compute_idf_all_docs_contain_term() {
        // Every document contains the term → low IDF
        let idf = TfIdfScore::compute_idf(100, 100);
        assert!(idf < 2.0); // ln(2) ≈ 0.693
    }

    #[test]
    fn test_tfidf_score() {
        let score = TfIdfScore::score(0.3, 4.0);
        assert!((score - 1.2).abs() < 1e-5);
    }

    #[test]
    fn test_tfidf_compute_convenience() {
        let s = TfIdfScore::compute(2, 10, 1000, 5);
        assert!(s > 0.0);
    }

    // BM25 ────────────────────────────────────────────────────────────────────

    #[test]
    fn test_bm25_default_params() {
        let p = BM25Params::default_params();
        assert!((p.k1 - 1.5).abs() < f32::EPSILON);
        assert!((p.b - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bm25_score_positive() {
        let p = BM25Params::default_params();
        let s = p.score(2.0, 3.0, 50);
        assert!(s > 0.0);
    }

    #[test]
    fn test_bm25_score_zero_idf() {
        let p = BM25Params::default_params();
        let s = p.score(5.0, 0.0, 100);
        assert!((s - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bm25_longer_doc_lower_score() {
        let p = BM25Params::default_params();
        let short = p.score(2.0, 3.0, 20);
        let long = p.score(2.0, 3.0, 500);
        assert!(short > long);
    }

    // RankResult ──────────────────────────────────────────────────────────────

    #[test]
    fn test_rank_result_new() {
        let r = RankResult::new("doc1", 0.9);
        assert_eq!(r.doc_id, "doc1");
        assert!((r.score - 0.9).abs() < f32::EPSILON);
        assert!(r.title.is_none());
    }

    #[test]
    fn test_rank_result_with_title() {
        let r = RankResult::new("doc1", 1.0).with_title("My Video");
        assert_eq!(r.title.as_deref(), Some("My Video"));
    }

    #[test]
    fn test_rank_by_score_descending() {
        let mut results = vec![
            RankResult::new("a", 0.3),
            RankResult::new("b", 0.9),
            RankResult::new("c", 0.6),
        ];
        RankResult::rank_by_score(&mut results);
        assert_eq!(results[0].doc_id, "b");
        assert_eq!(results[1].doc_id, "c");
        assert_eq!(results[2].doc_id, "a");
    }

    #[test]
    fn test_ranked_returns_sorted_vec() {
        let results = vec![RankResult::new("x", 0.1), RankResult::new("y", 0.8)];
        let ranked = RankResult::ranked(results);
        assert_eq!(ranked[0].doc_id, "y");
    }

    #[test]
    fn test_boost_multiplies_score() {
        let mut r = RankResult::new("doc1", 2.0);
        r.boost(3.0);
        assert!((r.score - 6.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalise_max_becomes_one() {
        let mut results = vec![
            RankResult::new("a", 4.0),
            RankResult::new("b", 2.0),
            RankResult::new("c", 0.0),
        ];
        RankResult::normalise(&mut results);
        assert!((results[0].score - 1.0).abs() < 1e-5);
        assert!((results[1].score - 0.5).abs() < 1e-5);
    }
}
