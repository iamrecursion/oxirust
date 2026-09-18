#![allow(dead_code)]

//! Relevance scoring algorithms for search results.
//!
//! Implements TF-IDF, BM25, and custom field-boosted scoring to rank
//! search results by their relevance to a query.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Term frequency helpers
// ---------------------------------------------------------------------------

/// Compute raw term frequency of `term` in `tokens`.
#[must_use]
pub fn term_frequency(term: &str, tokens: &[&str]) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    let count = tokens.iter().filter(|&&t| t == term).count();
    count as f64
}

/// Compute log-normalised term frequency: 1 + ln(tf) if tf > 0.
#[must_use]
pub fn log_term_frequency(term: &str, tokens: &[&str]) -> f64 {
    let tf = term_frequency(term, tokens);
    if tf > 0.0 {
        1.0 + tf.ln()
    } else {
        0.0
    }
}

/// Compute inverse document frequency: ln(N / df).
///
/// `total_docs` is the total number of documents in the collection.
/// `doc_freq` is the number of documents containing the term.
#[must_use]
pub fn inverse_document_frequency(total_docs: usize, doc_freq: usize) -> f64 {
    if doc_freq == 0 || total_docs == 0 {
        return 0.0;
    }
    (total_docs as f64 / doc_freq as f64).ln()
}

// ---------------------------------------------------------------------------
// TF-IDF scorer
// ---------------------------------------------------------------------------

/// TF-IDF based relevance scorer.
#[derive(Debug, Clone)]
pub struct TfIdfScorer {
    /// Total number of documents in the collection.
    total_docs: usize,
    /// Document frequency for each term (how many docs contain it).
    doc_frequencies: HashMap<String, usize>,
}

impl TfIdfScorer {
    /// Create a new TF-IDF scorer.
    #[must_use]
    pub fn new(total_docs: usize, doc_frequencies: HashMap<String, usize>) -> Self {
        Self {
            total_docs,
            doc_frequencies,
        }
    }

    /// Score a document (represented by its tokens) against query terms.
    #[must_use]
    pub fn score(&self, query_terms: &[&str], doc_tokens: &[&str]) -> f64 {
        let mut total = 0.0;
        for &qt in query_terms {
            let tf = log_term_frequency(qt, doc_tokens);
            let df = self.doc_frequencies.get(qt).copied().unwrap_or(0);
            let idf = inverse_document_frequency(self.total_docs, df);
            total += tf * idf;
        }
        total
    }

    /// Return total docs.
    #[must_use]
    pub fn total_docs(&self) -> usize {
        self.total_docs
    }
}

// ---------------------------------------------------------------------------
// BM25 scorer
// ---------------------------------------------------------------------------

/// BM25 relevance scoring algorithm.
///
/// BM25 is a probabilistic ranking function that improves on TF-IDF
/// by incorporating document length normalization.
#[derive(Debug, Clone)]
pub struct Bm25Scorer {
    /// Free parameter controlling term frequency saturation (typically 1.2-2.0).
    pub k1: f64,
    /// Free parameter controlling document length normalization (typically 0.75).
    pub b: f64,
    /// Total number of documents in the collection.
    total_docs: usize,
    /// Average document length across the collection.
    avg_doc_len: f64,
    /// Document frequency for each term.
    doc_frequencies: HashMap<String, usize>,
}

impl Bm25Scorer {
    /// Create a new BM25 scorer with standard parameters.
    #[must_use]
    pub fn new(
        total_docs: usize,
        avg_doc_len: f64,
        doc_frequencies: HashMap<String, usize>,
    ) -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            total_docs,
            avg_doc_len,
            doc_frequencies,
        }
    }

    /// Create a BM25 scorer with custom k1 and b parameters.
    #[must_use]
    pub fn with_params(
        k1: f64,
        b: f64,
        total_docs: usize,
        avg_doc_len: f64,
        doc_frequencies: HashMap<String, usize>,
    ) -> Self {
        Self {
            k1,
            b,
            total_docs,
            avg_doc_len,
            doc_frequencies,
        }
    }

    /// Compute IDF component for BM25 (uses the Robertson-Sparck Jones formula).
    fn idf(&self, doc_freq: usize) -> f64 {
        let n = self.total_docs as f64;
        let df = doc_freq as f64;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Score a document against query terms.
    ///
    /// `doc_tokens` are the tokens in the document, `doc_len` is the
    /// total number of tokens in the document.
    #[must_use]
    pub fn score(&self, query_terms: &[&str], doc_tokens: &[&str], doc_len: usize) -> f64 {
        let mut total = 0.0;
        let dl = doc_len as f64;

        for &qt in query_terms {
            let tf = term_frequency(qt, doc_tokens);
            let df = self.doc_frequencies.get(qt).copied().unwrap_or(0);
            let idf = self.idf(df);
            let numerator = tf * (self.k1 + 1.0);
            let denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avg_doc_len);
            total += idf * numerator / denominator;
        }
        total
    }

    /// Return total docs count.
    #[must_use]
    pub fn total_docs(&self) -> usize {
        self.total_docs
    }

    /// Return average document length.
    #[must_use]
    pub fn avg_doc_len(&self) -> f64 {
        self.avg_doc_len
    }
}

// ---------------------------------------------------------------------------
// Field-boosted scorer
// ---------------------------------------------------------------------------

/// Per-field boost weights for multi-field scoring.
#[derive(Debug, Clone)]
pub struct FieldBoost {
    /// Map of field name to boost weight.
    pub weights: HashMap<String, f64>,
    /// Default weight for fields not in the map.
    pub default_weight: f64,
}

impl Default for FieldBoost {
    fn default() -> Self {
        Self {
            weights: HashMap::new(),
            default_weight: 1.0,
        }
    }
}

impl FieldBoost {
    /// Create a field boost with the given field weights.
    #[must_use]
    pub fn new(weights: HashMap<String, f64>) -> Self {
        Self {
            weights,
            default_weight: 1.0,
        }
    }

    /// Get the boost weight for a given field.
    #[must_use]
    pub fn weight_for(&self, field: &str) -> f64 {
        self.weights
            .get(field)
            .copied()
            .unwrap_or(self.default_weight)
    }

    /// Compute the combined score across multiple fields.
    ///
    /// Each entry in `field_scores` is `(field_name, raw_score)`.
    #[must_use]
    pub fn combined_score(&self, field_scores: &[(&str, f64)]) -> f64 {
        let mut total = 0.0;
        for &(field, score) in field_scores {
            total += score * self.weight_for(field);
        }
        total
    }
}

// ---------------------------------------------------------------------------
// Score normalization
// ---------------------------------------------------------------------------

/// Normalization method for final score output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationMethod {
    /// No normalization.
    None,
    /// Min-max normalization to [0, 1].
    MinMax,
    /// Z-score normalization (mean=0, std=1).
    ZScore,
}

/// Normalize a list of scores using the specified method.
pub fn normalize_scores(scores: &[f64], method: NormalizationMethod) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }

    match method {
        NormalizationMethod::None => scores.to_vec(),
        NormalizationMethod::MinMax => {
            let min = scores.iter().copied().fold(f64::INFINITY, f64::min);
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let range = max - min;
            if range.abs() < f64::EPSILON {
                vec![0.5; scores.len()]
            } else {
                scores.iter().map(|&s| (s - min) / range).collect()
            }
        }
        NormalizationMethod::ZScore => {
            let n = scores.len() as f64;
            let mean = scores.iter().sum::<f64>() / n;
            let variance = scores.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / n;
            let std_dev = variance.sqrt();
            if std_dev.abs() < f64::EPSILON {
                vec![0.0; scores.len()]
            } else {
                scores.iter().map(|&s| (s - mean) / std_dev).collect()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_frequency_basic() {
        let tokens = vec!["the", "quick", "brown", "fox", "the"];
        assert!((term_frequency("the", &tokens) - 2.0).abs() < f64::EPSILON);
        assert!((term_frequency("fox", &tokens) - 1.0).abs() < f64::EPSILON);
        assert!((term_frequency("cat", &tokens) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_term_frequency_empty() {
        let tokens: Vec<&str> = Vec::new();
        assert!((term_frequency("any", &tokens) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_log_term_frequency() {
        let tokens = vec!["a", "b", "a", "a"];
        let ltf = log_term_frequency("a", &tokens);
        // 1 + ln(3)
        assert!((ltf - (1.0 + 3.0_f64.ln())).abs() < 1e-9);
    }

    #[test]
    fn test_log_term_frequency_zero() {
        let tokens = vec!["a"];
        assert!((log_term_frequency("b", &tokens) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_idf() {
        let idf = inverse_document_frequency(1000, 10);
        assert!((idf - (100.0_f64).ln()).abs() < 1e-9);
    }

    #[test]
    fn test_idf_zero_df() {
        assert!((inverse_document_frequency(100, 0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tfidf_scorer() {
        let mut df = HashMap::new();
        df.insert("fox".to_string(), 5);
        df.insert("quick".to_string(), 10);
        let scorer = TfIdfScorer::new(100, df);
        let doc_tokens = vec!["the", "quick", "brown", "fox"];
        let score = scorer.score(&["fox"], &doc_tokens);
        assert!(score > 0.0);
        assert_eq!(scorer.total_docs(), 100);
    }

    #[test]
    fn test_bm25_scorer() {
        let mut df = HashMap::new();
        df.insert("fox".to_string(), 5);
        let scorer = Bm25Scorer::new(100, 50.0, df);
        let doc_tokens = vec!["the", "quick", "brown", "fox"];
        let score = scorer.score(&["fox"], &doc_tokens, 4);
        assert!(score > 0.0);
        assert_eq!(scorer.total_docs(), 100);
        assert!((scorer.avg_doc_len() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bm25_custom_params() {
        let df = HashMap::new();
        let scorer = Bm25Scorer::with_params(2.0, 0.5, 10, 20.0, df);
        assert!((scorer.k1 - 2.0).abs() < f64::EPSILON);
        assert!((scorer.b - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_field_boost_combined() {
        let mut weights = HashMap::new();
        weights.insert("title".to_string(), 3.0);
        weights.insert("body".to_string(), 1.0);
        let boost = FieldBoost::new(weights);
        let field_scores = vec![("title", 0.5), ("body", 0.8)];
        let combined = boost.combined_score(&field_scores);
        // 0.5*3.0 + 0.8*1.0 = 2.3
        assert!((combined - 2.3).abs() < 1e-9);
    }

    #[test]
    fn test_field_boost_default_weight() {
        let boost = FieldBoost::default();
        assert!((boost.weight_for("unknown") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_normalize_minmax() {
        let scores = vec![10.0, 20.0, 30.0];
        let normed = normalize_scores(&scores, NormalizationMethod::MinMax);
        assert!((normed[0] - 0.0).abs() < 1e-9);
        assert!((normed[1] - 0.5).abs() < 1e-9);
        assert!((normed[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_normalize_zscore() {
        let scores = vec![1.0, 2.0, 3.0];
        let normed = normalize_scores(&scores, NormalizationMethod::ZScore);
        // mean=2, std~0.816
        assert!(normed[0] < 0.0);
        assert!((normed[1]).abs() < 1e-9);
        assert!(normed[2] > 0.0);
    }

    #[test]
    fn test_normalize_none() {
        let scores = vec![5.0, 10.0];
        let normed = normalize_scores(&scores, NormalizationMethod::None);
        assert!((normed[0] - 5.0).abs() < f64::EPSILON);
        assert!((normed[1] - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_normalize_empty() {
        let normed = normalize_scores(&[], NormalizationMethod::MinMax);
        assert!(normed.is_empty());
    }

    #[test]
    fn test_normalize_single_value() {
        let normed = normalize_scores(&[42.0], NormalizationMethod::MinMax);
        assert_eq!(normed.len(), 1);
        // range is 0, should return 0.5
        assert!((normed[0] - 0.5).abs() < f64::EPSILON);
    }
}
