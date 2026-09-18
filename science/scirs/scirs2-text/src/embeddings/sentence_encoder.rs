//! Sentence-level encoder with SimCSE-style contrastive training.
//!
//! Produces fixed-length sentence vectors from sequences of token embeddings
//! (represented as `Vec<Vec<f64>>`) using several pooling strategies.  A
//! lightweight linear projection reduces the token-embedding dimension to the
//! desired sentence-embedding dimension.
//!
//! # Design
//!
//! This module is intentionally *framework-free*: it operates on plain
//! `Vec<f64>` slices and does not depend on ndarray or any ML library.
//!
//! # References
//!
//! - Gao et al. (2021) "SimCSE: Simple Contrastive Learning of Sentence
//!   Embeddings."  <https://arxiv.org/abs/2104.08821>
//! - Reimers & Gurevych (2019) "Sentence-BERT: Sentence Embeddings using
//!   Siamese BERT-Networks."  <https://arxiv.org/abs/1908.10084>

use crate::error::{Result, TextError};

// ── PoolingStrategy ───────────────────────────────────────────────────────────

/// Strategy for aggregating per-token embeddings into a single sentence vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Arithmetic mean of all token embeddings.
    Mean,
    /// Element-wise maximum across all token embeddings.
    Max,
    /// First-token (CLS) embedding.
    Cls,
    /// TF-IDF–style weighted mean: token weight is its index-based IDF
    /// approximation `1 / (1 + rank)` so earlier, potentially rarer tokens
    /// get slightly more weight.  Falls back to `Mean` for single-token
    /// inputs.
    Weighted,
}

// ── SentenceEncoder ───────────────────────────────────────────────────────────

/// Projects sequences of token embeddings to a single sentence-level vector.
///
/// Internally the encoder applies:
/// 1. **Pooling** — aggregate token embeddings with the chosen strategy.
/// 2. **Projection** — a learnable `embedding_dim × projection_dim` linear
///    layer (bias included) maps the pooled vector to the output space.
/// 3. **Optional L2 normalisation** to unit length.
///
/// Weights are initialised from a deterministic LCG seeded by `seed`.
pub struct SentenceEncoder {
    embedding_dim: usize,
    projection_dim: usize,
    /// Flat row-major matrix of shape `embedding_dim × projection_dim`.
    projection: Vec<f64>,
    bias: Vec<f64>,
    pooling: PoolingStrategy,
    normalize: bool,
}

impl SentenceEncoder {
    /// Create a new `SentenceEncoder` with LCG-initialised weights.
    ///
    /// # Parameters
    /// - `embedding_dim` — dimensionality of token embeddings fed to `encode`.
    /// - `projection_dim` — output dimensionality of sentence embeddings.
    /// - `pooling` — pooling strategy.
    /// - `seed` — deterministic PRNG seed.
    pub fn new(
        embedding_dim: usize,
        projection_dim: usize,
        pooling: PoolingStrategy,
        seed: u64,
    ) -> Self {
        let proj_size = embedding_dim * projection_dim;
        let mut projection = Vec::with_capacity(proj_size);
        let scale = (2.0_f64 / embedding_dim as f64).sqrt();
        for i in 0..proj_size {
            projection.push((lcg_f64(seed, i as u64) * 2.0 - 1.0) * scale);
        }

        let mut bias = Vec::with_capacity(projection_dim);
        for i in 0..projection_dim {
            bias.push((lcg_f64(seed.wrapping_add(1), i as u64) * 2.0 - 1.0) * 0.01);
        }

        SentenceEncoder {
            embedding_dim,
            projection_dim,
            projection,
            bias,
            pooling,
            normalize: true,
        }
    }

    /// Enable or disable L2 normalisation of output embeddings.
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Encode a sequence of token embeddings into a single sentence vector.
    ///
    /// Returns a `Vec<f64>` of length `projection_dim`.
    ///
    /// # Errors
    /// Returns an error when `token_embeddings` is empty or any token
    /// embedding has a dimension other than `embedding_dim`.
    pub fn encode(&self, token_embeddings: &[Vec<f64>]) -> Result<Vec<f64>> {
        if token_embeddings.is_empty() {
            return Err(TextError::InvalidInput(
                "token_embeddings must not be empty".to_string(),
            ));
        }
        for (i, tok) in token_embeddings.iter().enumerate() {
            if tok.len() != self.embedding_dim {
                return Err(TextError::InvalidInput(format!(
                    "token {} has dimension {} but expected {}",
                    i,
                    tok.len(),
                    self.embedding_dim
                )));
            }
        }

        let pooled = self.pool(token_embeddings);
        let mut projected = self.project(&pooled);

        if self.normalize {
            Self::normalize(&mut projected);
        }

        Ok(projected)
    }

    // ── Pooling helpers ───────────────────────────────────────────────────────

    fn pool(&self, tokens: &[Vec<f64>]) -> Vec<f64> {
        match self.pooling {
            PoolingStrategy::Mean => {
                let n = tokens.len() as f64;
                let dim = self.embedding_dim;
                let mut out = vec![0.0f64; dim];
                for tok in tokens {
                    for (j, &v) in tok.iter().enumerate() {
                        out[j] += v;
                    }
                }
                out.iter_mut().for_each(|x| *x /= n);
                out
            }

            PoolingStrategy::Max => {
                let dim = self.embedding_dim;
                let mut out = tokens[0].clone();
                out.resize(dim, f64::NEG_INFINITY);
                for tok in tokens.iter().skip(1) {
                    for (j, &v) in tok.iter().enumerate() {
                        if j < dim && v > out[j] {
                            out[j] = v;
                        }
                    }
                }
                out
            }

            PoolingStrategy::Cls => tokens[0].clone(),

            PoolingStrategy::Weighted => {
                // Weight token i by 1 / (1 + i) (higher rank → lower weight)
                let dim = self.embedding_dim;
                let mut out = vec![0.0f64; dim];
                let mut total_weight = 0.0f64;
                for (i, tok) in tokens.iter().enumerate() {
                    let w = 1.0 / (1.0 + i as f64);
                    total_weight += w;
                    for (j, &v) in tok.iter().enumerate() {
                        out[j] += v * w;
                    }
                }
                if total_weight > 0.0 {
                    out.iter_mut().for_each(|x| *x /= total_weight);
                }
                out
            }
        }
    }

    // ── Projection helper ─────────────────────────────────────────────────────

    fn project(&self, v: &[f64]) -> Vec<f64> {
        let d_in = self.embedding_dim;
        let d_out = self.projection_dim;
        let mut out = vec![0.0f64; d_out];
        for j in 0..d_out {
            let mut sum = self.bias[j];
            for i in 0..d_in {
                sum += v[i] * self.projection[i * d_out + j];
            }
            out[j] = sum;
        }
        out
    }

    // ── Public utilities ──────────────────────────────────────────────────────

    /// Cosine similarity between two sentence embeddings.
    ///
    /// Returns a value in `[-1, 1]`.  Returns `0.0` when either vector has
    /// zero norm.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na < 1e-12 || nb < 1e-12 {
            return 0.0;
        }
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }

    /// Encode multiple sentences and return the `n × n` cosine-similarity
    /// matrix.
    ///
    /// Each element of `sentences` is a `Vec<Vec<f64>>` (token embeddings for
    /// one sentence).
    ///
    /// # Errors
    /// Propagates any error from [`encode`](Self::encode).
    pub fn similarity_matrix(&self, sentences: &[Vec<Vec<f64>>]) -> Result<Vec<Vec<f64>>> {
        let embeddings: Vec<Vec<f64>> = sentences
            .iter()
            .map(|s| self.encode(s))
            .collect::<Result<Vec<_>>>()?;

        let n = embeddings.len();
        let mut matrix = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                matrix[i][j] = Self::cosine_similarity(&embeddings[i], &embeddings[j]);
            }
        }
        Ok(matrix)
    }

    /// L2-normalise a vector in place.  A zero-norm vector is left unchanged.
    pub fn normalize(v: &mut [f64]) {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 && norm.is_finite() {
            v.iter_mut().for_each(|x| *x /= norm);
        }
    }

    /// The output (projection) dimension.
    pub fn projection_dim(&self) -> usize {
        self.projection_dim
    }

    /// The input (token embedding) dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

impl std::fmt::Debug for SentenceEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SentenceEncoder")
            .field("embedding_dim", &self.embedding_dim)
            .field("projection_dim", &self.projection_dim)
            .field("pooling", &self.pooling)
            .field("normalize", &self.normalize)
            .finish()
    }
}

// ── SimCseConfig ──────────────────────────────────────────────────────────────

/// Configuration for the SimCSE-style contrastive trainer.
#[derive(Debug, Clone)]
pub struct SimCseConfig {
    /// Temperature parameter τ for the InfoNCE loss.  Default: 0.05.
    pub temperature: f64,
    /// Learning rate for SGD weight update.  Default: 1e-3.
    pub learning_rate: f64,
}

impl Default for SimCseConfig {
    fn default() -> Self {
        SimCseConfig {
            temperature: 0.05,
            learning_rate: 1e-3,
        }
    }
}

// ── SimCseTrainer ─────────────────────────────────────────────────────────────

/// SimCSE-style contrastive trainer.
///
/// Given batches of (anchor, positive) pairs, it computes the InfoNCE loss
/// using in-batch negatives and performs a single SGD step on the projection
/// matrix.
///
/// # Loss
///
/// ```text
/// ℓ = -log( exp(sim(a, p) / τ) / Σⱼ exp(sim(a, eⱼ) / τ) )
/// ```
///
/// where the denominator sums over the positive and all other batch embeddings
/// treated as negatives.
pub struct SimCseTrainer {
    config: SimCseConfig,
    encoder: SentenceEncoder,
    step_count: usize,
}

impl SimCseTrainer {
    /// Create a new trainer wrapping the given encoder.
    pub fn new(encoder: SentenceEncoder, config: SimCseConfig) -> Self {
        SimCseTrainer {
            config,
            encoder,
            step_count: 0,
        }
    }

    /// Compute the InfoNCE contrastive loss for a batch of `(anchor, positive)`
    /// pairs.
    ///
    /// Both `anchors` and `positives` must have the same length (≥ 1).
    ///
    /// # Errors
    /// Returns an error when the batch is empty or `encode` fails.
    pub fn contrastive_loss(
        &self,
        anchors: &[Vec<Vec<f64>>],
        positives: &[Vec<Vec<f64>>],
    ) -> Result<f64> {
        if anchors.is_empty() {
            return Err(TextError::InvalidInput(
                "batch must contain at least one pair".to_string(),
            ));
        }
        if anchors.len() != positives.len() {
            return Err(TextError::InvalidInput(format!(
                "anchors length ({}) differs from positives length ({})",
                anchors.len(),
                positives.len()
            )));
        }

        let tau = self.config.temperature;

        // Encode all anchors and positives
        let a_embs: Vec<Vec<f64>> = anchors
            .iter()
            .map(|a| self.encoder.encode(a))
            .collect::<Result<_>>()?;
        let p_embs: Vec<Vec<f64>> = positives
            .iter()
            .map(|p| self.encoder.encode(p))
            .collect::<Result<_>>()?;

        // All positives form the "keys" pool (in-batch negatives)
        let n = a_embs.len();
        let mut total_loss = 0.0f64;

        for i in 0..n {
            let ai = &a_embs[i];
            let sim_pos = SentenceEncoder::cosine_similarity(ai, &p_embs[i]) / tau;

            // Denominator: sum over all positives including the matching one
            let denom: f64 = p_embs
                .iter()
                .map(|pk| (SentenceEncoder::cosine_similarity(ai, pk) / tau).exp())
                .sum();

            if denom > 0.0 && denom.is_finite() {
                total_loss += -sim_pos + denom.ln();
            }
        }

        Ok(total_loss / n as f64)
    }

    /// Perform a single SGD step: compute the contrastive loss, approximate
    /// gradients via finite differences on the projection matrix, and update
    /// weights.
    ///
    /// Returns the loss *before* the update.
    ///
    /// # Errors
    /// Propagates errors from `contrastive_loss`.
    pub fn step(&mut self, anchors: &[Vec<Vec<f64>>], positives: &[Vec<Vec<f64>>]) -> Result<f64> {
        let loss_before = self.contrastive_loss(anchors, positives)?;

        let lr = self.config.learning_rate;
        let eps = 1e-5_f64;
        let proj_len = self.encoder.projection.len();

        // Finite-difference gradient estimate on projection weights.
        // For efficiency we only update a random subset of weights each step
        // to avoid O(proj_size) forward passes; here we do a full pass but
        // with early exit when loss is already very small.
        if loss_before < 1e-8 {
            self.step_count += 1;
            return Ok(loss_before);
        }

        let mut grad = vec![0.0f64; proj_len];
        for k in 0..proj_len {
            let orig = self.encoder.projection[k];
            self.encoder.projection[k] = orig + eps;
            let loss_plus = self
                .contrastive_loss(anchors, positives)
                .unwrap_or(loss_before);
            self.encoder.projection[k] = orig;

            // Central-difference: (f(x+h) - f(x)) / h  (forward diff)
            grad[k] = (loss_plus - loss_before) / eps;
        }

        // SGD update
        for k in 0..proj_len {
            self.encoder.projection[k] -= lr * grad[k];
        }

        // Also update bias
        let bias_len = self.encoder.bias.len();
        for j in 0..bias_len {
            let orig = self.encoder.bias[j];
            self.encoder.bias[j] = orig + eps;
            let loss_plus = self
                .contrastive_loss(anchors, positives)
                .unwrap_or(loss_before);
            self.encoder.bias[j] = orig;
            let g = (loss_plus - loss_before) / eps;
            self.encoder.bias[j] -= lr * g;
        }

        self.step_count += 1;
        Ok(loss_before)
    }

    /// Borrow the underlying encoder.
    pub fn encoder(&self) -> &SentenceEncoder {
        &self.encoder
    }

    /// Number of update steps taken so far.
    pub fn step_count(&self) -> usize {
        self.step_count
    }
}

impl std::fmt::Debug for SimCseTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimCseTrainer")
            .field("step_count", &self.step_count)
            .field("temperature", &self.config.temperature)
            .finish()
    }
}

// ── SemanticSimilarity ────────────────────────────────────────────────────────

/// Embedding-based semantic search over a document corpus.
///
/// Documents are encoded on insertion and compared via cosine similarity at
/// query time.
pub struct SemanticSimilarity {
    encoder: SentenceEncoder,
    corpus_embeddings: Vec<Vec<f64>>,
    corpus_keys: Vec<String>,
}

impl SemanticSimilarity {
    /// Create an empty search index.
    pub fn new(encoder: SentenceEncoder) -> Self {
        SemanticSimilarity {
            encoder,
            corpus_embeddings: Vec::new(),
            corpus_keys: Vec::new(),
        }
    }

    /// Encode `token_embeddings` and add the resulting vector to the index
    /// under `key`.
    ///
    /// Silently skips documents that fail to encode (e.g. empty sequences).
    pub fn add_document(&mut self, key: String, token_embeddings: Vec<Vec<f64>>) {
        match self.encoder.encode(&token_embeddings) {
            Ok(emb) => {
                self.corpus_embeddings.push(emb);
                self.corpus_keys.push(key);
            }
            Err(_) => {
                // Skip unencodable documents silently
            }
        }
    }

    /// Return the `top_k` most-similar documents to the query, ordered by
    /// descending cosine similarity.
    ///
    /// If `top_k` exceeds the corpus size, all documents are returned.
    ///
    /// # Errors
    /// Returns an error when the query fails to encode.
    pub fn search(
        &self,
        query_embeddings: &[Vec<f64>],
        top_k: usize,
    ) -> Result<Vec<(String, f64)>> {
        let query_emb = self.encoder.encode(query_embeddings)?;

        let mut scored: Vec<(usize, f64)> = self
            .corpus_embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| {
                let sim = SentenceEncoder::cosine_similarity(&query_emb, emb);
                (i, sim)
            })
            .collect();

        // Sort by descending similarity
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = top_k.min(scored.len());
        Ok(scored[..k]
            .iter()
            .map(|(i, sim)| (self.corpus_keys[*i].clone(), *sim))
            .collect())
    }

    /// Number of documents currently in the index.
    pub fn len(&self) -> usize {
        self.corpus_keys.len()
    }

    /// Returns `true` when the index contains no documents.
    pub fn is_empty(&self) -> bool {
        self.corpus_keys.is_empty()
    }
}

impl std::fmt::Debug for SemanticSimilarity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemanticSimilarity")
            .field("corpus_size", &self.corpus_keys.len())
            .finish()
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Linear congruential generator; returns a pseudo-random value in `[0, 1)`.
fn lcg_f64(seed: u64, offset: u64) -> f64 {
    const A: u64 = 6_364_136_223_846_793_005;
    const C: u64 = 1_442_695_040_888_963_407;
    let state = A.wrapping_mul(seed.wrapping_add(offset)).wrapping_add(C);
    ((state >> 12) as f64) / ((1u64 << 52) as f64)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple encoder for tests.
    fn make_encoder(pooling: PoolingStrategy) -> SentenceEncoder {
        SentenceEncoder::new(8, 16, pooling, 42)
    }

    /// Create `n` random token embeddings of dimension `dim` seeded by `base`.
    fn rand_tokens(n: usize, dim: usize, base: u64) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|j| lcg_f64(base + i as u64, j as u64) * 2.0 - 1.0)
                    .collect()
            })
            .collect()
    }

    // ── SentenceEncoder ───────────────────────────────────────────────────────

    #[test]
    fn cosine_similarity_identical() {
        let v = vec![1.0f64, 2.0, 3.0, 4.0];
        let sim = SentenceEncoder::cosine_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "cosine sim of identical vectors must be 1.0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0f64, 0.0, 0.0];
        let b = vec![0.0f64, 1.0, 0.0];
        let sim = SentenceEncoder::cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-10,
            "cosine sim of orthogonal vectors must be 0.0, got {sim}"
        );
    }

    #[test]
    fn encode_output_has_projection_dim() {
        let enc = make_encoder(PoolingStrategy::Mean);
        let toks = rand_tokens(5, 8, 1);
        let emb = enc.encode(&toks).expect("encode must succeed");
        assert_eq!(
            emb.len(),
            16,
            "output length must equal projection_dim (16), got {}",
            emb.len()
        );
    }

    #[test]
    fn encode_normalized_has_unit_norm() {
        let enc = make_encoder(PoolingStrategy::Mean);
        let toks = rand_tokens(4, 8, 99);
        let emb = enc.encode(&toks).expect("encode must succeed");
        let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-9,
            "normalized embedding must have unit norm, got {norm}"
        );
    }

    #[test]
    fn similarity_matrix_is_symmetric() {
        let enc = make_encoder(PoolingStrategy::Mean);
        let sentences: Vec<Vec<Vec<f64>>> = (0..4_u64).map(|s| rand_tokens(3, 8, s * 10)).collect();
        let mat = enc
            .similarity_matrix(&sentences)
            .expect("similarity_matrix must succeed");
        let n = mat.len();
        assert_eq!(n, 4, "matrix must be 4 × 4");
        for i in 0..n {
            for j in 0..n {
                let diff = (mat[i][j] - mat[j][i]).abs();
                assert!(
                    diff < 1e-10,
                    "matrix[{i}][{j}]={} != matrix[{j}][{i}]={} (diff={diff})",
                    mat[i][j],
                    mat[j][i]
                );
            }
        }
    }

    #[test]
    fn similarity_matrix_diagonal_is_one() {
        let enc = make_encoder(PoolingStrategy::Max);
        let sentences: Vec<Vec<Vec<f64>>> =
            (0..3_u64).map(|s| rand_tokens(4, 8, s * 7 + 5)).collect();
        let mat = enc
            .similarity_matrix(&sentences)
            .expect("similarity_matrix must succeed");
        for i in 0..3 {
            assert!(
                (mat[i][i] - 1.0).abs() < 1e-9,
                "diagonal entry mat[{i}][{i}] must be 1.0, got {}",
                mat[i][i]
            );
        }
    }

    #[test]
    fn encode_empty_tokens_returns_error() {
        let enc = make_encoder(PoolingStrategy::Cls);
        let result = enc.encode(&[]);
        assert!(
            result.is_err(),
            "encode of empty tokens must return an error"
        );
    }

    #[test]
    fn encode_wrong_dim_returns_error() {
        let enc = make_encoder(PoolingStrategy::Mean);
        // encoder expects dim=8 but we supply dim=4
        let bad_tok = vec![vec![1.0f64; 4]];
        let result = enc.encode(&bad_tok);
        assert!(
            result.is_err(),
            "encode of wrong-dim token must return an error"
        );
    }

    // ── SimCseTrainer ─────────────────────────────────────────────────────────

    #[test]
    fn contrastive_loss_is_nonneg_and_finite() {
        let enc = make_encoder(PoolingStrategy::Mean);
        let trainer = SimCseTrainer::new(enc, SimCseConfig::default());

        let anchors: Vec<Vec<Vec<f64>>> = (0..4_u64).map(|s| rand_tokens(3, 8, s)).collect();
        let positives: Vec<Vec<Vec<f64>>> =
            (0..4_u64).map(|s| rand_tokens(3, 8, s + 100)).collect();

        let loss = trainer
            .contrastive_loss(&anchors, &positives)
            .expect("loss must succeed");
        assert!(loss >= 0.0, "contrastive loss must be >= 0, got {loss}");
        assert!(loss.is_finite(), "contrastive loss must be finite");
    }

    #[test]
    fn simcse_step_returns_loss() {
        let enc = make_encoder(PoolingStrategy::Mean);
        let mut trainer = SimCseTrainer::new(
            enc,
            SimCseConfig {
                temperature: 0.05,
                learning_rate: 1e-4,
            },
        );

        // Use consistent anchor = positive to drive loss down
        let data: Vec<Vec<Vec<f64>>> = (0..2_u64).map(|s| rand_tokens(2, 8, s)).collect();
        let loss = trainer.step(&data, &data).expect("step must succeed");
        assert!(loss.is_finite(), "step must return finite loss");
        assert_eq!(trainer.step_count(), 1);
    }

    // ── SemanticSimilarity ────────────────────────────────────────────────────

    #[test]
    fn search_returns_top_k_in_descending_order() {
        let enc = make_encoder(PoolingStrategy::Mean);
        let mut index = SemanticSimilarity::new(enc);

        for i in 0..5_u64 {
            index.add_document(format!("doc{i}"), rand_tokens(3, 8, i * 13));
        }

        let query = rand_tokens(2, 8, 99);
        let results = index.search(&query, 3).expect("search must succeed");

        assert_eq!(results.len(), 3, "must return exactly top_k=3 results");

        // Check descending order
        for w in results.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "results must be in descending similarity order: {} < {}",
                w[0].1,
                w[1].1
            );
        }
    }

    #[test]
    fn search_empty_corpus_returns_empty() {
        let enc = make_encoder(PoolingStrategy::Mean);
        let index = SemanticSimilarity::new(enc);
        let query = rand_tokens(2, 8, 7);
        let results = index.search(&query, 5).expect("search must succeed");
        assert!(
            results.is_empty(),
            "search on empty corpus must return empty"
        );
    }

    #[test]
    fn search_top_k_exceeds_corpus_returns_all() {
        let enc = make_encoder(PoolingStrategy::Mean);
        let mut index = SemanticSimilarity::new(enc);
        for i in 0..3_u64 {
            index.add_document(format!("d{i}"), rand_tokens(2, 8, i));
        }
        let query = rand_tokens(1, 8, 200);
        let results = index
            .search(&query, 10)
            .expect("search must succeed when top_k > corpus");
        assert_eq!(
            results.len(),
            3,
            "search must return all 3 docs when top_k>corpus"
        );
    }
}
