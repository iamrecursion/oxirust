//! Universal Sentence Encoder-style fixed-length sentence embeddings.
//!
//! Produces fixed-length embeddings via word-vector averaging and position-weighted
//! mean pooling. Fully self-contained — no external neural model required.
//!
//! # References
//! Cer et al. (2018) "Universal Sentence Encoder"
//! <https://arxiv.org/abs/1803.11175>

use std::collections::HashMap;

// ── PoolingStrategy ───────────────────────────────────────────────────────────

/// Strategy for aggregating per-word embeddings into a sentence vector.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PoolingStrategy {
    /// Average of all token embeddings.
    #[default]
    Mean,
    /// Element-wise maximum across token embeddings.
    Max,
    /// Position-weighted mean (later tokens slightly up-weighted).
    WeightedMean,
    /// CLS-style: use only the first token's representation.
    FirstToken,
}

// ── SentenceEncoderConfig ─────────────────────────────────────────────────────

/// Configuration for [`SentenceEncoder`].
#[derive(Debug, Clone)]
pub struct SentenceEncoderConfig {
    /// Output embedding dimensionality. Default: 128.
    pub embedding_dim: usize,
    /// Maximum sequence length (tokens beyond this are truncated). Default: 128.
    pub max_seq_len: usize,
    /// Pooling strategy for aggregating token embeddings. Default: Mean.
    pub pooling: PoolingStrategy,
    /// Whether to L2-normalise the output vector. Default: true.
    pub normalize: bool,
}

impl Default for SentenceEncoderConfig {
    fn default() -> Self {
        SentenceEncoderConfig {
            embedding_dim: 128,
            max_seq_len: 128,
            pooling: PoolingStrategy::Mean,
            normalize: true,
        }
    }
}

// ── SentenceEncoder ───────────────────────────────────────────────────────────

/// Encodes sentences to fixed-length float vectors via word-embedding lookup
/// and pooling.
///
/// Words not found in the vocabulary receive an OOV vector (all zeros by
/// default, but they are excluded from mean pooling when all words in the
/// sentence would otherwise be OOV — in that case a zero vector is returned).
pub struct SentenceEncoder {
    config: SentenceEncoderConfig,
    /// Word → embedding vector lookup table.
    embeddings: HashMap<String, Vec<f32>>,
    /// Cached embedding dimensionality (equals `config.embedding_dim`).
    embedding_dim: usize,
}

impl SentenceEncoder {
    // ── Constructors ──────────────────────────────────────────────────────

    /// Create a `SentenceEncoder` with **randomly initialised** embeddings for
    /// every word in `vocab`.
    ///
    /// Embeddings are initialised deterministically from a seeded LCG so that
    /// results are reproducible without importing any RNG crate.
    pub fn new(vocab: &[String], config: SentenceEncoderConfig) -> Self {
        let dim = config.embedding_dim;
        let mut embeddings = HashMap::with_capacity(vocab.len());
        for (word_idx, word) in vocab.iter().enumerate() {
            let vec: Vec<f32> = (0..dim)
                .map(|d| lcg_f32(42, word_idx as u64 * dim as u64 + d as u64))
                .collect();
            embeddings.insert(word.clone(), vec);
        }
        SentenceEncoder {
            config,
            embeddings,
            embedding_dim: dim,
        }
    }

    /// Create a `SentenceEncoder` from a pre-built token-to-vector map.
    ///
    /// All vectors must have the same length, which must equal
    /// `config.embedding_dim`.  If the map is empty the encoder still works
    /// but will return zero vectors for every sentence.
    pub fn from_vectors(vectors: HashMap<String, Vec<f32>>, config: SentenceEncoderConfig) -> Self {
        let dim = config.embedding_dim;
        SentenceEncoder {
            config,
            embeddings: vectors,
            embedding_dim: dim,
        }
    }

    // ── Encoding ─────────────────────────────────────────────────────────

    /// Encode a single sentence to a fixed-length `Vec<f32>`.
    ///
    /// The sentence is split on whitespace (after lower-casing). Tokens
    /// beyond `max_seq_len` are dropped.  Words not found in the vocabulary
    /// are ignored (treated as if absent) in mean/weighted-mean pooling.
    /// For max pooling, missing words contribute a zero vector.
    pub fn encode(&self, sentence: &str) -> Vec<f32> {
        let tokens = self.tokenize(sentence);
        self.pool(&tokens)
    }

    /// Encode a batch of sentences.
    pub fn encode_batch(&self, sentences: &[&str]) -> Vec<Vec<f32>> {
        sentences.iter().map(|s| self.encode(s)).collect()
    }

    // ── Similarity / search ───────────────────────────────────────────────

    /// Cosine similarity between two embedding vectors.
    ///
    /// Returns a value in `[-1.0, 1.0]`, or `0.0` when either vector has zero
    /// norm.
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        cosine_sim(a, b)
    }

    /// Find the `top_k` sentences most similar to `query` (by cosine
    /// similarity), returned in descending similarity order.
    pub fn most_similar<'a>(
        &self,
        query: &str,
        sentences: &[&'a str],
        top_k: usize,
    ) -> Vec<(&'a str, f32)> {
        let q_emb = self.encode(query);
        let mut scored: Vec<(&'a str, f32)> = sentences
            .iter()
            .map(|&s| {
                let emb = self.encode(s);
                let sim = cosine_sim(&q_emb, &emb);
                (s, sim)
            })
            .collect();

        // Sort descending by similarity (NaN-safe: NaN treated as -∞)
        scored.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    /// Simple whitespace tokenizer with lower-casing + length truncation.
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .take(self.config.max_seq_len)
            .map(|t| t.to_string())
            .collect()
    }

    /// Pool token embeddings according to the configured strategy.
    fn pool(&self, tokens: &[String]) -> Vec<f32> {
        let dim = self.embedding_dim;

        if tokens.is_empty() {
            return vec![0.0f32; dim];
        }

        let result = match self.config.pooling {
            PoolingStrategy::Mean => {
                let mut sum = vec![0.0f32; dim];
                let mut count = 0usize;
                for token in tokens {
                    if let Some(emb) = self.embeddings.get(token) {
                        for (s, e) in sum.iter_mut().zip(emb.iter()) {
                            *s += e;
                        }
                        count += 1;
                    }
                }
                if count == 0 {
                    return vec![0.0f32; dim];
                }
                let n = count as f32;
                sum.iter_mut().for_each(|v| *v /= n);
                sum
            }

            PoolingStrategy::Max => {
                let mut max_vec = vec![f32::NEG_INFINITY; dim];
                let mut any_hit = false;
                for token in tokens {
                    let emb = self
                        .embeddings
                        .get(token)
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);
                    if emb.len() == dim {
                        any_hit = true;
                        for (m, &e) in max_vec.iter_mut().zip(emb.iter()) {
                            if e > *m {
                                *m = e;
                            }
                        }
                    }
                }
                if !any_hit {
                    return vec![0.0f32; dim];
                }
                // Replace any remaining -inf with 0.0 (from OOV tokens)
                max_vec.iter_mut().for_each(|v| {
                    if v.is_infinite() {
                        *v = 0.0
                    }
                });
                max_vec
            }

            PoolingStrategy::WeightedMean => {
                // Later tokens receive linearly higher weight:
                // weight[i] = i + 1  (1-based position)
                let n = tokens.len();
                let mut sum = vec![0.0f32; dim];
                let mut total_weight = 0.0f32;
                for (i, token) in tokens.iter().enumerate() {
                    if let Some(emb) = self.embeddings.get(token) {
                        let w = (i + 1) as f32;
                        for (s, e) in sum.iter_mut().zip(emb.iter()) {
                            *s += e * w;
                        }
                        total_weight += w;
                    }
                }
                let _ = n; // consumed above implicitly
                if total_weight < 1e-12 {
                    return vec![0.0f32; dim];
                }
                sum.iter_mut().for_each(|v| *v /= total_weight);
                sum
            }

            PoolingStrategy::FirstToken => {
                for token in tokens {
                    if let Some(emb) = self.embeddings.get(token) {
                        return if self.config.normalize {
                            l2_norm_f32(emb.clone())
                        } else {
                            emb.clone()
                        };
                    }
                }
                return vec![0.0f32; dim];
            }
        };

        if self.config.normalize {
            l2_norm_f32(result)
        } else {
            result
        }
    }

    /// Return the embedding dimensionality.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Mutable access to the embeddings map for in-place updates.
    pub fn embeddings_mut(&mut self) -> &mut HashMap<String, Vec<f32>> {
        &mut self.embeddings
    }
}

impl std::fmt::Debug for SentenceEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SentenceEncoder")
            .field("embedding_dim", &self.embedding_dim)
            .field("vocab_size", &self.embeddings.len())
            .field("pooling", &self.config.pooling)
            .finish()
    }
}

// ── Free functions ────────────────────────────────────────────────────────────

/// Cosine similarity between two f32 slices.  Returns 0.0 when either is zero.
pub(crate) fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// In-place L2 normalisation of an f32 vector.
pub(crate) fn l2_norm_f32(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 && norm.is_finite() {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

/// LCG pseudo-random float in (-1, 1) — no external crate needed.
fn lcg_f32(seed: u64, offset: u64) -> f32 {
    const A: u64 = 6_364_136_223_846_793_005;
    const C: u64 = 1_442_695_040_888_963_407;
    let state = A.wrapping_mul(seed.wrapping_add(offset)).wrapping_add(C);
    let frac = ((state >> 12) as f64) / ((1u64 << 52) as f64); // [0, 1)
    (frac as f32) * 2.0 - 1.0
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_vocab(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("word{i}")).collect()
    }

    fn build_encoder(pooling: PoolingStrategy) -> SentenceEncoder {
        let vocab = build_vocab(100);
        SentenceEncoder::new(
            &vocab,
            SentenceEncoderConfig {
                embedding_dim: 32,
                max_seq_len: 64,
                pooling,
                normalize: true,
            },
        )
    }

    // ── test_sentence_encoder_output_dim ──────────────────────────────────

    #[test]
    fn test_sentence_encoder_output_dim() {
        let enc = build_encoder(PoolingStrategy::Mean);
        let emb = enc.encode("word0 word1 word2");
        assert_eq!(emb.len(), 32, "output dim must equal embedding_dim");
    }

    // ── test_sentence_encoder_similarity_self ────────────────────────────

    #[test]
    fn test_sentence_encoder_similarity_self() {
        let enc = build_encoder(PoolingStrategy::Mean);
        let s = "word0 word1 word2";
        let emb = enc.encode(s);
        let sim = enc.similarity(&emb, &emb);
        assert!(
            (sim - 1.0_f32).abs() < 1e-5,
            "self-similarity must be ~1.0, got {sim}"
        );
    }

    // ── test_sentence_encoder_most_similar_returns_topk ──────────────────

    #[test]
    fn test_sentence_encoder_most_similar_returns_topk() {
        let enc = build_encoder(PoolingStrategy::Mean);
        let candidates = &[
            "word0 word1",
            "word2 word3",
            "word4 word5",
            "word6 word7",
            "word8 word9",
        ];
        let top3 = enc.most_similar("word0 word1", candidates, 3);
        assert_eq!(top3.len(), 3, "should return exactly top_k results");
        // Results must be in descending similarity order
        for pair in top3.windows(2) {
            assert!(pair[0].1 >= pair[1].1, "results must be sorted descending");
        }
    }

    #[test]
    fn test_max_pooling_output_dim() {
        let enc = build_encoder(PoolingStrategy::Max);
        let emb = enc.encode("word0 word3 word7");
        assert_eq!(emb.len(), 32);
    }

    #[test]
    fn test_weighted_mean_pooling_output_dim() {
        let enc = build_encoder(PoolingStrategy::WeightedMean);
        let emb = enc.encode("word0 word1 word2 word3");
        assert_eq!(emb.len(), 32);
    }

    #[test]
    fn test_empty_sentence_returns_zero_vec() {
        let enc = build_encoder(PoolingStrategy::Mean);
        let emb = enc.encode("");
        assert_eq!(emb.len(), 32);
        assert!(emb.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_normalize_unit_norm() {
        let enc = build_encoder(PoolingStrategy::Mean);
        let emb = enc.encode("word0 word1 word2");
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0_f32).abs() < 1e-5, "normalised vector norm ~1.0");
    }
}
