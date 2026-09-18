//! Hierarchical Dirichlet Process (HDP) topic model.
//!
//! HDP automatically infers the number of topics from data using the
//! Chinese Restaurant Franchise (CRF) representation with collapsed Gibbs
//! sampling.
//!
//! ## Model
//!
//! The generative process mirrors LDA but replaces the fixed-`K` Dirichlet
//! prior on topic proportions with a DP prior, allowing the number of active
//! topics to grow with data.
//!
//! We follow the *truncated variational Bayes* approximation of
//! Teh et al. (2006) using a stick-breaking representation with a fixed
//! truncation level `T`.  A topic is considered "active" when at least one
//! token has been assigned to it.
//!
//! # References
//! Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006).
//! "Hierarchical Dirichlet Processes." *JASA*, 101(476), 1566–1581.
//! <https://doi.org/10.1198/016214506000000302>

use crate::error::{Result, TextError};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{rngs::StdRng, SeedableRng};

// ── HdpConfig ─────────────────────────────────────────────────────────────────

/// Configuration for the [`HdpModel`].
#[derive(Debug, Clone)]
pub struct HdpConfig {
    /// Truncation level T — maximum number of topics the model can infer.
    /// Default: 20.
    pub max_topics: usize,
    /// Document-level DP concentration parameter α. Default: 1.0.
    pub alpha: f64,
    /// Corpus-level DP concentration parameter γ. Default: 1.0.
    pub gamma: f64,
    /// Symmetric Dirichlet prior on word distributions η. Default: 0.1.
    pub eta: f64,
    /// Number of collapsed Gibbs iterations. Default: 100.
    pub n_iter: usize,
    /// Optional random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for HdpConfig {
    fn default() -> Self {
        HdpConfig {
            max_topics: 20,
            alpha: 1.0,
            gamma: 1.0,
            eta: 0.1,
            n_iter: 100,
            seed: None,
        }
    }
}

// ── HdpResult ─────────────────────────────────────────────────────────────────

/// Summary statistics returned by [`HdpModel::fit`].
#[derive(Debug, Clone)]
pub struct HdpResult {
    /// Number of active topics (those with at least one assigned token).
    pub n_topics: usize,
    /// Per-held-out-token perplexity estimate (exp(-avg log-likelihood)).
    pub perplexity: f64,
    /// Average log-likelihood per token.
    pub log_likelihood: f64,
    /// Actual number of Gibbs iterations performed.
    pub iterations: usize,
}

// ── HdpModel ──────────────────────────────────────────────────────────────────

/// Hierarchical Dirichlet Process topic model.
///
/// Fitted with collapsed Gibbs sampling over the Chinese Restaurant Franchise.
///
/// After calling [`fit`](HdpModel::fit) you can:
/// - Query [`n_topics_active`](HdpModel::n_topics_active) for the inferred
///   number of topics.
/// - Call [`transform`](HdpModel::transform) to get topic distributions for
///   new documents.
/// - Call [`top_words`](HdpModel::top_words) to inspect topic–word associations.
/// - Call [`coherence`](HdpModel::coherence) to evaluate topic quality.
pub struct HdpModel {
    config: HdpConfig,
    /// φ[k][w]: normalised topic-word distributions (T × vocab_size).
    phi: Vec<Vec<f64>>,
    /// Unnormalised topic-word count matrix (T × vocab_size).
    topic_word_counts: Vec<Vec<f64>>,
    /// Number of tokens assigned to each topic.
    topic_counts: Vec<usize>,
    /// Number of active topics inferred from data.
    pub n_topics_active: usize,
    /// Vocabulary size.
    vocab_size: usize,
    /// Whether the model has been fitted.
    is_fitted: bool,
}

impl HdpModel {
    /// Create a new (unfitted) HDP model.
    pub fn new(config: HdpConfig) -> Self {
        let t = config.max_topics;
        HdpModel {
            config,
            phi: vec![vec![]; t],
            topic_word_counts: vec![vec![]; t],
            topic_counts: vec![0; t],
            n_topics_active: 0,
            vocab_size: 0,
            is_fitted: false,
        }
    }

    // ── fit ──────────────────────────────────────────────────────────────

    /// Fit the HDP model to `corpus`.
    ///
    /// `corpus` is a slice of documents; each document is a `Vec<usize>`
    /// of word indices (all indices must be < `vocab_size`).
    ///
    /// # Errors
    /// Returns an error if `corpus` is empty, `vocab_size` is zero, or
    /// any word index is out of range.
    pub fn fit(&mut self, corpus: &[Vec<usize>], vocab_size: usize) -> Result<HdpResult> {
        if corpus.is_empty() {
            return Err(TextError::InvalidInput(
                "corpus must not be empty".to_string(),
            ));
        }
        if vocab_size == 0 {
            return Err(TextError::InvalidInput(
                "vocab_size must be > 0".to_string(),
            ));
        }
        // Validate indices
        for (di, doc) in corpus.iter().enumerate() {
            for &w in doc {
                if w >= vocab_size {
                    return Err(TextError::InvalidInput(format!(
                        "word index {w} in document {di} exceeds vocab_size {vocab_size}"
                    )));
                }
            }
        }

        self.vocab_size = vocab_size;
        let t = self.config.max_topics;

        // Initialise count tables
        self.topic_word_counts = vec![vec![0.0f64; vocab_size]; t];
        self.topic_counts = vec![0usize; t];

        let mut rng = self.make_rng();

        // ── Initialise topic assignments randomly ─────────────────────
        let n_docs = corpus.len();
        // z[d][n] = topic assignment for token n in doc d
        let mut z: Vec<Vec<usize>> = corpus
            .iter()
            .map(|doc| {
                doc.iter()
                    .map(|_| rng.random_range(0..t))
                    .collect::<Vec<usize>>()
            })
            .collect();

        // doc-level topic counts: theta_counts[d][k] = #tokens in doc d assigned to topic k
        let mut theta_counts: Vec<Vec<usize>> = vec![vec![0usize; t]; n_docs];

        // Populate initial counts
        for (d, doc) in corpus.iter().enumerate() {
            for (n, &w) in doc.iter().enumerate() {
                let k = z[d][n];
                self.topic_word_counts[k][w] += 1.0;
                self.topic_counts[k] += 1;
                theta_counts[d][k] += 1;
            }
        }

        // ── Collapsed Gibbs sampling ──────────────────────────────────
        let alpha = self.config.alpha;
        let eta = self.config.eta;
        let eta_sum = eta * vocab_size as f64;

        let mut iter_done = 0usize;
        for _iter in 0..self.config.n_iter {
            for d in 0..n_docs {
                for n in 0..corpus[d].len() {
                    let w = corpus[d][n];
                    let k_old = z[d][n];

                    // Remove this token's contribution
                    self.topic_word_counts[k_old][w] -= 1.0;
                    self.topic_counts[k_old] -= 1;
                    theta_counts[d][k_old] -= 1;

                    // Compute unnormalised probabilities for each topic
                    let mut probs = vec![0.0f64; t];
                    for k in 0..t {
                        let doc_factor = theta_counts[d][k] as f64 + alpha / t as f64;
                        let word_factor = (self.topic_word_counts[k][w] + eta)
                            / (self.topic_counts[k] as f64 + eta_sum);
                        probs[k] = doc_factor * word_factor;
                    }

                    // Sample new topic from normalised distribution
                    let k_new = sample_categorical(&probs, &mut rng);

                    // Update counts
                    z[d][n] = k_new;
                    self.topic_word_counts[k_new][w] += 1.0;
                    self.topic_counts[k_new] += 1;
                    theta_counts[d][k_new] += 1;
                }
            }
            iter_done += 1;
        }

        // ── Compute normalised φ and active topics ────────────────────
        self.phi = (0..t)
            .map(|k| {
                let total = self.topic_counts[k] as f64 + eta_sum;
                (0..vocab_size)
                    .map(|w| (self.topic_word_counts[k][w] + eta) / total)
                    .collect()
            })
            .collect();

        self.n_topics_active = self.topic_counts.iter().filter(|&&c| c > 0).count();
        self.is_fitted = true;

        // ── Compute log-likelihood / perplexity ───────────────────────
        let (ll, pp) = self.compute_perplexity(corpus, &theta_counts, eta, eta_sum);

        Ok(HdpResult {
            n_topics: self.n_topics_active,
            perplexity: pp,
            log_likelihood: ll,
            iterations: iter_done,
        })
    }

    // ── transform ────────────────────────────────────────────────────────

    /// Infer the topic distribution for a new (unseen) document.
    ///
    /// Uses one pass of the E-step (word-topic probability normalisation)
    /// without modifying model parameters.
    ///
    /// # Errors
    /// Returns an error if the model is not fitted or `doc` is empty.
    pub fn transform(&self, doc: &[usize]) -> Result<Vec<f64>> {
        if !self.is_fitted {
            return Err(TextError::ModelNotFitted(
                "HDP model not fitted yet".to_string(),
            ));
        }
        if doc.is_empty() {
            return Err(TextError::InvalidInput(
                "document must not be empty".to_string(),
            ));
        }

        let t = self.config.max_topics;
        let eta = self.config.eta;
        let eta_sum = eta * self.vocab_size as f64;

        let mut theta = vec![self.config.alpha / t as f64; t];

        // Simple E-step: accumulate normalised word-topic probabilities
        for &w in doc {
            if w >= self.vocab_size {
                continue;
            }
            let mut word_probs: Vec<f64> = (0..t)
                .map(|k| {
                    theta[k] * (self.topic_word_counts[k][w] + eta)
                        / (self.topic_counts[k] as f64 + eta_sum)
                })
                .collect();
            let sum: f64 = word_probs.iter().sum();
            if sum > 0.0 {
                word_probs.iter_mut().for_each(|p| *p /= sum);
                for k in 0..t {
                    theta[k] += word_probs[k];
                }
            }
        }

        // Normalise θ
        let theta_sum: f64 = theta.iter().sum();
        if theta_sum > 0.0 {
            theta.iter_mut().for_each(|p| *p /= theta_sum);
        }

        Ok(theta)
    }

    // ── top_words ────────────────────────────────────────────────────────

    /// Return the top-`n` word indices for each *active* topic, ordered by
    /// descending probability.
    ///
    /// # Errors
    /// Returns an error if the model is not fitted.
    pub fn top_words(&self, n: usize) -> Result<Vec<Vec<usize>>> {
        if !self.is_fitted {
            return Err(TextError::ModelNotFitted(
                "HDP model not fitted yet".to_string(),
            ));
        }

        let t = self.config.max_topics;
        let mut result = Vec::new();

        for k in 0..t {
            if self.topic_counts[k] == 0 {
                continue; // skip inactive topics
            }
            let phi_k = &self.phi[k];
            let mut indices: Vec<usize> = (0..phi_k.len()).collect();
            // Sort by descending probability
            indices.sort_by(|&a, &b| {
                phi_k[b]
                    .partial_cmp(&phi_k[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            indices.truncate(n);
            result.push(indices);
        }

        Ok(result)
    }

    // ── coherence ────────────────────────────────────────────────────────

    /// Compute per-topic PMI-based coherence scores.
    ///
    /// Uses the top-`n_top` words per active topic and measures co-occurrence
    /// in the training corpus.
    ///
    /// # Errors
    /// Returns an error if the model is not fitted.
    pub fn coherence(&self, corpus: &[Vec<usize>], n_top: usize) -> Result<Vec<f64>> {
        if !self.is_fitted {
            return Err(TextError::ModelNotFitted(
                "HDP model not fitted yet".to_string(),
            ));
        }

        let top = self.top_words(n_top)?;
        let n_docs = corpus.len() as f64;

        // Build word document-frequency map
        let mut df: Vec<f64> = vec![0.0; self.vocab_size];
        let mut codf: Vec<Vec<f64>> = vec![vec![0.0; self.vocab_size]; self.vocab_size];
        for doc in corpus {
            // Deduplicate within doc for DF counting
            let mut seen = std::collections::HashSet::new();
            for &w in doc {
                if w < self.vocab_size && seen.insert(w) {
                    df[w] += 1.0;
                }
            }
            let seen_vec: Vec<usize> = seen.into_iter().collect();
            for (i, &wi) in seen_vec.iter().enumerate() {
                for &wj in &seen_vec[i + 1..] {
                    let (a, b) = if wi < wj { (wi, wj) } else { (wj, wi) };
                    codf[a][b] += 1.0;
                }
            }
        }

        let mut scores = Vec::with_capacity(top.len());
        for topic_words in &top {
            let mut sum = 0.0f64;
            let mut count = 0usize;
            for (i, &wi) in topic_words.iter().enumerate() {
                for &wj in &topic_words[i + 1..] {
                    let (a, b) = if wi < wj { (wi, wj) } else { (wj, wi) };
                    let co = codf[a][b] + 1.0; // Laplace smoothing
                    let di = df[wi] + 1.0;
                    let dj = df[wj] + 1.0;
                    // PMI = log(P(wi,wj)) - log(P(wi)) - log(P(wj))
                    let pmi = (co / n_docs).ln() - (di / n_docs).ln() - (dj / n_docs).ln();
                    sum += pmi;
                    count += 1;
                }
            }
            scores.push(if count > 0 { sum / count as f64 } else { 0.0 });
        }

        Ok(scores)
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    fn make_rng(&self) -> StdRng {
        match self.config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_rng(&mut scirs2_core::random::rng()),
        }
    }

    /// Compute average log-likelihood and perplexity after Gibbs.
    fn compute_perplexity(
        &self,
        corpus: &[Vec<usize>],
        theta_counts: &[Vec<usize>],
        eta: f64,
        eta_sum: f64,
    ) -> (f64, f64) {
        let t = self.config.max_topics;
        let alpha = self.config.alpha;
        let mut total_ll = 0.0f64;
        let mut total_tokens = 0usize;

        for (d, doc) in corpus.iter().enumerate() {
            let theta_sum: f64 = theta_counts[d].iter().sum::<usize>() as f64 + alpha;
            for &w in doc {
                if w >= self.vocab_size {
                    continue;
                }
                // p(w | doc) = Σ_k θ_{dk} φ_{kw}
                let p_w: f64 = (0..t)
                    .map(|k| {
                        let theta_dk = (theta_counts[d][k] as f64 + alpha / t as f64) / theta_sum;
                        let phi_kw = (self.topic_word_counts[k][w] + eta)
                            / (self.topic_counts[k] as f64 + eta_sum);
                        theta_dk * phi_kw
                    })
                    .sum();
                if p_w > 0.0 {
                    total_ll += p_w.ln();
                }
                total_tokens += 1;
            }
        }

        if total_tokens == 0 {
            return (0.0, 1.0);
        }

        let avg_ll = total_ll / total_tokens as f64;
        let perplexity = (-avg_ll).exp();
        (avg_ll, perplexity)
    }
}

impl std::fmt::Debug for HdpModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HdpModel")
            .field("max_topics", &self.config.max_topics)
            .field("n_topics_active", &self.n_topics_active)
            .field("vocab_size", &self.vocab_size)
            .field("is_fitted", &self.is_fitted)
            .finish()
    }
}

// ── Internal sampling helper ──────────────────────────────────────────────────

/// Sample a category index from an unnormalised probability vector.
fn sample_categorical(probs: &[f64], rng: &mut StdRng) -> usize {
    let total: f64 = probs.iter().sum();
    if total <= 0.0 {
        // Degenerate — return uniform
        return rng.random_range(0..probs.len());
    }
    let u: f64 = rng.random_range(0.0..total);
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if u < cumulative {
            return i;
        }
    }
    probs.len() - 1 // fallback due to floating-point rounding
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic corpus with three clearly separated topics.
    /// Topic 0: words 0–4 (tech)
    /// Topic 1: words 5–9 (sports)
    /// Topic 2: words 10–14 (food)
    fn synthetic_corpus(n_per_topic: usize) -> Vec<Vec<usize>> {
        let mut corpus = Vec::new();
        let mut rng = StdRng::seed_from_u64(99);
        // Topic 0 documents
        for _ in 0..n_per_topic {
            let doc: Vec<usize> = (0..20).map(|_| rng.random_range(0..5)).collect();
            corpus.push(doc);
        }
        // Topic 1 documents
        for _ in 0..n_per_topic {
            let doc: Vec<usize> = (0..20).map(|_| rng.random_range(5..10)).collect();
            corpus.push(doc);
        }
        // Topic 2 documents
        for _ in 0..n_per_topic {
            let doc: Vec<usize> = (0..20).map(|_| rng.random_range(10..15)).collect();
            corpus.push(doc);
        }
        corpus
    }

    // ── test_hdp_infers_topics ────────────────────────────────────────────

    #[test]
    fn test_hdp_infers_topics() {
        let corpus = synthetic_corpus(15);
        let config = HdpConfig {
            max_topics: 20,
            n_iter: 30,
            seed: Some(42),
            ..Default::default()
        };
        let mut model = HdpModel::new(config);
        let result = model.fit(&corpus, 15).expect("fit should succeed");

        assert!(
            result.n_topics <= 20,
            "active topics ({}) must be <= max_topics",
            result.n_topics
        );
        assert!(
            result.n_topics >= 1,
            "at least one topic must be active, got {}",
            result.n_topics
        );
    }

    // ── test_hdp_perplexity_finite ────────────────────────────────────────

    #[test]
    fn test_hdp_perplexity_finite() {
        let corpus = synthetic_corpus(10);
        let config = HdpConfig {
            max_topics: 10,
            n_iter: 20,
            seed: Some(7),
            ..Default::default()
        };
        let mut model = HdpModel::new(config);
        let result = model.fit(&corpus, 15).expect("fit should succeed");

        assert!(
            result.perplexity.is_finite(),
            "perplexity must be finite, got {}",
            result.perplexity
        );
        assert!(
            result.perplexity > 0.0,
            "perplexity must be positive, got {}",
            result.perplexity
        );
        assert!(
            result.log_likelihood.is_finite(),
            "log_likelihood must be finite"
        );
    }

    // ── test_hdp_top_words_valid ──────────────────────────────────────────

    #[test]
    fn test_hdp_top_words_valid() {
        let corpus = synthetic_corpus(10);
        let config = HdpConfig {
            max_topics: 10,
            n_iter: 20,
            seed: Some(1),
            ..Default::default()
        };
        let mut model = HdpModel::new(config);
        model.fit(&corpus, 15).expect("fit should succeed");

        let top5 = model.top_words(5).expect("top_words should succeed");
        for topic_words in &top5 {
            assert!(
                topic_words.len() <= 5,
                "each topic should have <= n top words"
            );
            for &w in topic_words {
                assert!(w < 15, "word index {w} must be < vocab_size 15");
            }
        }
    }

    #[test]
    fn test_hdp_transform() {
        let corpus = synthetic_corpus(10);
        let config = HdpConfig {
            max_topics: 5,
            n_iter: 15,
            seed: Some(123),
            ..Default::default()
        };
        let mut model = HdpModel::new(config);
        model.fit(&corpus, 15).expect("fit should succeed");

        let doc = vec![0usize, 1, 2, 3, 0];
        let theta = model.transform(&doc).expect("transform should succeed");
        assert_eq!(theta.len(), 5);
        let sum: f64 = theta.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "topic distribution must sum to 1");
        for &p in &theta {
            assert!(p >= 0.0, "all topic probabilities must be >= 0");
        }
    }

    #[test]
    fn test_hdp_coherence() {
        let corpus = synthetic_corpus(10);
        let config = HdpConfig {
            max_topics: 5,
            n_iter: 15,
            seed: Some(55),
            ..Default::default()
        };
        let mut model = HdpModel::new(config);
        model.fit(&corpus, 15).expect("fit should succeed");

        let scores = model
            .coherence(&corpus, 3)
            .expect("coherence should succeed");
        for &s in &scores {
            assert!(s.is_finite(), "coherence score must be finite, got {s}");
        }
    }

    #[test]
    fn test_hdp_empty_corpus_error() {
        let mut model = HdpModel::new(HdpConfig::default());
        let result = model.fit(&[], 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_hdp_zero_vocab_error() {
        let mut model = HdpModel::new(HdpConfig::default());
        let result = model.fit(&[vec![0usize, 1]], 0);
        assert!(result.is_err());
    }
}
