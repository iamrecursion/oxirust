//! Sampling strategies for generative model token selection.
//!
//! Provides greedy decoding, temperature sampling, top-k, top-p (nucleus),
//! and a configurable sampler combining all of the above with repetition penalty.

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during sampling operations.
#[derive(Debug, Clone)]
pub enum SamplingError {
    /// The logit/probability vector was empty.
    EmptyDistribution,
    /// Temperature value was not strictly positive.
    InvalidTemperature(f64),
    /// Top-p value was outside (0, 1].
    InvalidTopP { p: f64 },
    /// Top-k value was zero.
    InvalidTopK { k: usize },
    /// Normalization of the distribution failed (e.g., all-zero softmax).
    NormalizationFailure,
    /// The probability array contained invalid values (NaN, negative, etc.).
    InvalidProbabilities(String),
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDistribution => write!(f, "logit/probability vector is empty"),
            Self::InvalidTemperature(t) => {
                write!(f, "temperature must be > 0.0, got {t}")
            }
            Self::InvalidTopP { p } => {
                write!(f, "top_p must be in (0, 1], got {p}")
            }
            Self::InvalidTopK { k } => {
                write!(f, "top_k must be >= 1, got {k}")
            }
            Self::NormalizationFailure => {
                write!(
                    f,
                    "probability distribution could not be normalised (all-zero or NaN)"
                )
            }
            Self::InvalidProbabilities(msg) => {
                write!(f, "invalid probability array: {msg}")
            }
        }
    }
}

impl std::error::Error for SamplingError {}

// ---------------------------------------------------------------------------
// Result of a single sampling step
// ---------------------------------------------------------------------------

/// The result of sampling a single token.
#[derive(Debug, Clone)]
pub struct SampledToken {
    /// Index of the selected token in the vocabulary.
    pub token_id: usize,
    /// Natural-log probability of the selected token: ln(prob).
    pub log_prob: f64,
    /// Linear probability of the selected token after softmax.
    pub prob: f64,
}

// ---------------------------------------------------------------------------
// SamplingConfig
// ---------------------------------------------------------------------------

/// Configuration for the [`ConfigurableSampler`].
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Scale applied to logits before softmax. 1.0 = no scaling.
    pub temperature: f64,
    /// If `Some(k)`, only the top-k logits participate in sampling.
    pub top_k: Option<usize>,
    /// If `Some(p)`, nucleus sampling keeps the fewest tokens whose
    /// cumulative probability meets or exceeds `p`.
    pub top_p: Option<f64>,
    /// Penalty applied to tokens that already appear in the context.
    /// Values > 1.0 reduce the probability of repetition; 1.0 = no effect.
    pub repetition_penalty: f64,
    /// Optional seed for the internal LCG RNG.
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            seed: None,
        }
    }
}

// ---------------------------------------------------------------------------
// SimpleRng – minimal LCG, no `rand` dependency
// ---------------------------------------------------------------------------

/// A minimal Linear Congruential Generator for reproducible sampling.
///
/// This avoids any external `rand` crate dependency while still providing
/// adequate statistical quality for token-sampling purposes.
#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        // Mix the seed slightly so seed=0 is not degenerate.
        let state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        Self { state }
    }

    /// Advance the LCG and return the raw 64-bit value.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state >> 11
    }

    /// Return a uniform float in [0.0, 1.0).
    fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa of f64.
        (self.next_u64() & ((1u64 << 53) - 1)) as f64 / (1u64 << 53) as f64
    }

    /// Draw from a categorical distribution defined by `probs` (must sum ≈ 1).
    ///
    /// Uses inverse CDF (linear scan); robust to small floating-point errors.
    fn sample_categorical(&mut self, probs: &[f64]) -> usize {
        let u = self.next_f64();
        let mut cumsum = 0.0_f64;
        for (idx, &p) in probs.iter().enumerate() {
            cumsum += p;
            if u < cumsum {
                return idx;
            }
        }
        // Fallback: return last non-zero index in case of rounding errors.
        probs
            .iter()
            .enumerate()
            .rev()
            .find(|(_, &p)| p > 0.0)
            .map(|(i, _)| i)
            .unwrap_or(probs.len().saturating_sub(1))
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Compute softmax with the log-sum-exp trick for numerical stability.
///
/// Returns a probability vector that sums to 1.0 (unless the input is empty).
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut exps: Vec<f64> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
}

/// Compute log-softmax: log(softmax(x)) with the log-sum-exp trick.
pub fn log_softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_sum_exp: f64 = logits
        .iter()
        .map(|&x| (x - max_val).exp())
        .sum::<f64>()
        .ln()
        + max_val;
    logits.iter().map(|&x| x - log_sum_exp).collect()
}

/// Shannon entropy of a probability distribution (in nats).
///
/// Tokens with zero probability contribute 0 to the sum (0 · ln 0 = 0).
pub fn entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

/// Perplexity: exp(mean negative log-prob) over a sequence of log-probabilities.
pub fn perplexity(log_probs: &[f64]) -> f64 {
    if log_probs.is_empty() {
        return 1.0;
    }
    let mean_nll = -log_probs.iter().sum::<f64>() / log_probs.len() as f64;
    mean_nll.exp()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Scale `logits` by `1 / temperature` and return a new `Vec<f64>`.
fn scale_by_temperature(logits: &[f64], temperature: f64) -> Vec<f64> {
    logits.iter().map(|&x| x / temperature).collect()
}

/// Given a probability vector, sample one token; return `(token_id, prob, log_prob)`.
fn sample_from_probs(probs: &[f64], rng: &mut SimpleRng) -> Result<SampledToken, SamplingError> {
    let sum: f64 = probs.iter().sum();
    if sum <= 0.0 || sum.is_nan() {
        return Err(SamplingError::NormalizationFailure);
    }
    let token_id = rng.sample_categorical(probs);
    let prob = probs[token_id];
    let log_prob = if prob > 0.0 {
        prob.ln()
    } else {
        f64::NEG_INFINITY
    };
    Ok(SampledToken {
        token_id,
        log_prob,
        prob,
    })
}

// ---------------------------------------------------------------------------
// GreedyDecoder
// ---------------------------------------------------------------------------

/// Always selects the token with the highest logit (argmax decoding).
#[derive(Debug, Clone)]
pub struct GreedyDecoder;

impl GreedyDecoder {
    /// Create a new `GreedyDecoder`.
    pub fn new() -> Self {
        Self
    }

    /// Decode a single logit vector, returning the argmax token.
    pub fn decode(&self, logits: &[f64]) -> Result<SampledToken, SamplingError> {
        if logits.is_empty() {
            return Err(SamplingError::EmptyDistribution);
        }
        let token_id = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or(SamplingError::EmptyDistribution)?;

        let probs = softmax(logits);
        let prob = probs[token_id];
        let log_prob = if prob > 0.0 {
            prob.ln()
        } else {
            f64::NEG_INFINITY
        };
        Ok(SampledToken {
            token_id,
            log_prob,
            prob,
        })
    }

    /// Decode a batch of logit vectors, one argmax per row.
    pub fn decode_batch(&self, logits: &[Vec<f64>]) -> Result<Vec<SampledToken>, SamplingError> {
        logits.iter().map(|row| self.decode(row)).collect()
    }
}

impl Default for GreedyDecoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TemperatureSampler
// ---------------------------------------------------------------------------

/// Samples from a softmax distribution after dividing logits by `temperature`.
///
/// - `temperature > 1.0`: flatter distribution (more randomness).
/// - `temperature < 1.0`: sharper distribution (more peaked).
/// - `temperature = 1.0`: unmodified softmax.
#[derive(Debug)]
pub struct TemperatureSampler {
    /// The temperature value used to scale logits.
    pub temperature: f64,
    rng: SimpleRng,
}

impl TemperatureSampler {
    /// Construct a `TemperatureSampler`.
    ///
    /// Returns `Err(SamplingError::InvalidTemperature)` if `temperature <= 0.0`.
    pub fn new(temperature: f64, seed: u64) -> Result<Self, SamplingError> {
        if temperature <= 0.0 || temperature.is_nan() {
            return Err(SamplingError::InvalidTemperature(temperature));
        }
        Ok(Self {
            temperature,
            rng: SimpleRng::new(seed),
        })
    }

    /// Sample one token from `logits`.
    pub fn sample(&mut self, logits: &[f64]) -> Result<SampledToken, SamplingError> {
        if logits.is_empty() {
            return Err(SamplingError::EmptyDistribution);
        }
        let scaled = scale_by_temperature(logits, self.temperature);
        let probs = softmax(&scaled);
        sample_from_probs(&probs, &mut self.rng)
    }

    /// Sample one token for each row in a batch.
    pub fn sample_batch(
        &mut self,
        logits: &[Vec<f64>],
    ) -> Result<Vec<SampledToken>, SamplingError> {
        logits.iter().map(|row| self.sample(row)).collect()
    }
}

// ---------------------------------------------------------------------------
// TopKSampler
// ---------------------------------------------------------------------------

/// Zeroes out all logits except the top-k, then applies temperature sampling.
#[derive(Debug)]
pub struct TopKSampler {
    /// Number of top tokens to keep.
    pub k: usize,
    /// Temperature applied after the top-k filter.
    pub temperature: f64,
    rng: SimpleRng,
}

impl TopKSampler {
    /// Construct a `TopKSampler`.
    ///
    /// Fails if `k == 0` or `temperature <= 0.0`.
    pub fn new(k: usize, temperature: f64, seed: u64) -> Result<Self, SamplingError> {
        if k == 0 {
            return Err(SamplingError::InvalidTopK { k });
        }
        if temperature <= 0.0 || temperature.is_nan() {
            return Err(SamplingError::InvalidTemperature(temperature));
        }
        Ok(Self {
            k,
            temperature,
            rng: SimpleRng::new(seed),
        })
    }

    /// Sample one token from `logits` using the top-k filter.
    pub fn sample(&mut self, logits: &[f64]) -> Result<SampledToken, SamplingError> {
        if logits.is_empty() {
            return Err(SamplingError::EmptyDistribution);
        }
        let filtered = Self::apply_top_k(logits, self.k);
        let scaled = scale_by_temperature(&filtered, self.temperature);
        let probs = softmax(&scaled);
        sample_from_probs(&probs, &mut self.rng)
    }

    /// Return a copy of `logits` where all but the top-`k` entries are
    /// set to `f64::NEG_INFINITY`.
    pub fn apply_top_k(logits: &[f64], k: usize) -> Vec<f64> {
        if logits.is_empty() || k == 0 {
            return logits.to_vec();
        }
        let effective_k = k.min(logits.len());

        // Build a list of (value, original_index), sort descending by value.
        let mut indexed: Vec<(f64, usize)> = logits
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Collect the indices of the top-k tokens.
        let top_k_indices: std::collections::HashSet<usize> =
            indexed.iter().take(effective_k).map(|&(_, i)| i).collect();

        logits
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if top_k_indices.contains(&i) {
                    v
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// TopPSampler
// ---------------------------------------------------------------------------

/// Nucleus (top-p) sampler: keeps the smallest set of tokens whose cumulative
/// probability is at least `p`, then samples from that nucleus.
#[derive(Debug)]
pub struct TopPSampler {
    /// Cumulative probability threshold in (0, 1].
    pub p: f64,
    /// Temperature applied before the nucleus filter.
    pub temperature: f64,
    rng: SimpleRng,
}

impl TopPSampler {
    /// Construct a `TopPSampler`.
    ///
    /// Fails if `p <= 0.0 || p > 1.0` or if `temperature <= 0.0`.
    pub fn new(p: f64, temperature: f64, seed: u64) -> Result<Self, SamplingError> {
        if p <= 0.0 || p > 1.0 || p.is_nan() {
            return Err(SamplingError::InvalidTopP { p });
        }
        if temperature <= 0.0 || temperature.is_nan() {
            return Err(SamplingError::InvalidTemperature(temperature));
        }
        Ok(Self {
            p,
            temperature,
            rng: SimpleRng::new(seed),
        })
    }

    /// Sample one token from `logits` using nucleus sampling.
    pub fn sample(&mut self, logits: &[f64]) -> Result<SampledToken, SamplingError> {
        if logits.is_empty() {
            return Err(SamplingError::EmptyDistribution);
        }
        let scaled = scale_by_temperature(logits, self.temperature);
        let probs = softmax(&scaled);
        let filtered_logits = Self::apply_top_p(&probs, self.p);
        let filtered_probs = softmax(&filtered_logits);
        sample_from_probs(&filtered_probs, &mut self.rng)
    }

    /// Given a probability vector `probs`, return a logit vector in which
    /// tokens outside the nucleus are set to `f64::NEG_INFINITY`.
    ///
    /// Algorithm:
    /// 1. Sort indices by descending probability.
    /// 2. Accumulate until the sum >= `p`.
    /// 3. All tokens beyond the cutoff become `NEG_INFINITY`.
    pub fn apply_top_p(probs: &[f64], p: f64) -> Vec<f64> {
        if probs.is_empty() {
            return Vec::new();
        }
        // Sort indices by descending probability.
        let mut sorted_indices: Vec<usize> = (0..probs.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find nucleus: smallest prefix whose cumulative prob >= p.
        let mut cumsum = 0.0_f64;
        let mut nucleus: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for &idx in &sorted_indices {
            nucleus.insert(idx);
            cumsum += probs[idx];
            if cumsum >= p {
                break;
            }
        }

        // Build output: nucleus tokens keep their log-prob; others = NEG_INFINITY.
        probs
            .iter()
            .enumerate()
            .map(|(i, &prob)| {
                if nucleus.contains(&i) {
                    // Convert probability back to logit space (log-prob is suitable).
                    if prob > 0.0 {
                        prob.ln()
                    } else {
                        f64::NEG_INFINITY
                    }
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ConfigurableSampler
// ---------------------------------------------------------------------------

/// A sampler that combines temperature scaling, top-k filtering, top-p
/// (nucleus) filtering, and repetition penalty into a single configurable
/// pipeline.
///
/// Pipeline order: repetition_penalty → temperature → top-k → top-p → sample.
#[derive(Debug)]
pub struct ConfigurableSampler {
    /// The complete sampling configuration.
    pub config: SamplingConfig,
    rng: SimpleRng,
}

impl ConfigurableSampler {
    /// Construct a `ConfigurableSampler` from a [`SamplingConfig`].
    ///
    /// Validates temperature, top_k, and top_p at construction time.
    pub fn new(config: SamplingConfig) -> Result<Self, SamplingError> {
        if config.temperature <= 0.0 || config.temperature.is_nan() {
            return Err(SamplingError::InvalidTemperature(config.temperature));
        }
        if let Some(k) = config.top_k {
            if k == 0 {
                return Err(SamplingError::InvalidTopK { k });
            }
        }
        if let Some(p) = config.top_p {
            if p <= 0.0 || p > 1.0 || p.is_nan() {
                return Err(SamplingError::InvalidTopP { p });
            }
        }
        let seed = config.seed.unwrap_or(42);
        Ok(Self {
            config,
            rng: SimpleRng::new(seed),
        })
    }

    /// Construct a `ConfigurableSampler` with the default configuration.
    ///
    /// This is equivalent to `ConfigurableSampler::new(SamplingConfig::default())` but
    /// is infallible because the defaults are always valid.
    pub fn with_default() -> Self {
        Self {
            config: SamplingConfig::default(),
            rng: SimpleRng::new(42),
        }
    }

    /// Apply repetition penalty in-place.
    ///
    /// For each token that appears in `context`:
    /// - If the logit is positive → divide by `penalty` (move toward 0).
    /// - If the logit is negative → multiply by `penalty` (move away from 0).
    ///
    /// A `penalty` of 1.0 is a no-op.
    pub fn apply_repetition_penalty(logits: &mut [f64], context: &[usize], penalty: f64) {
        if (penalty - 1.0).abs() < f64::EPSILON {
            return; // Fast path: no penalty.
        }
        for &token_id in context {
            if token_id < logits.len() {
                let v = logits[token_id];
                logits[token_id] = if v >= 0.0 { v / penalty } else { v * penalty };
            }
        }
    }

    /// Run the full sampling pipeline:
    ///
    /// 1. Apply repetition penalty to `logits` for tokens in `context`.
    /// 2. Scale by temperature.
    /// 3. Apply top-k filter (if configured).
    /// 4. Apply top-p (nucleus) filter (if configured).
    /// 5. Sample from the resulting distribution.
    pub fn sample(
        &mut self,
        logits: &[f64],
        context: &[usize],
    ) -> Result<SampledToken, SamplingError> {
        if logits.is_empty() {
            return Err(SamplingError::EmptyDistribution);
        }

        // Step 1: repetition penalty.
        let mut working = logits.to_vec();
        Self::apply_repetition_penalty(&mut working, context, self.config.repetition_penalty);

        // Step 2: temperature scaling.
        let mut working = scale_by_temperature(&working, self.config.temperature);

        // Step 3: top-k.
        if let Some(k) = self.config.top_k {
            working = TopKSampler::apply_top_k(&working, k);
        }

        // Step 4: top-p (operate on probabilities derived from current logits).
        if let Some(p) = self.config.top_p {
            let probs = softmax(&working);
            working = TopPSampler::apply_top_p(&probs, p);
        }

        // Step 5: sample.
        let probs = softmax(&working);
        sample_from_probs(&probs, &mut self.rng)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a small logit vector.
    fn logits_5() -> Vec<f64> {
        vec![0.1, 3.5, 1.2, -1.0, 2.0]
    }

    // ------------------------------------------------------------------
    // GreedyDecoder
    // ------------------------------------------------------------------

    #[test]
    fn test_greedy_decoder_argmax() {
        let decoder = GreedyDecoder::new();
        // Index 1 has the highest logit (3.5).
        let token = decoder.decode(&logits_5()).expect("decode should succeed");
        assert_eq!(token.token_id, 1);
    }

    #[test]
    fn test_greedy_decoder_empty() {
        let decoder = GreedyDecoder::new();
        let result = decoder.decode(&[]);
        assert!(
            matches!(result, Err(SamplingError::EmptyDistribution)),
            "expected EmptyDistribution, got {result:?}"
        );
    }

    // ------------------------------------------------------------------
    // TemperatureSampler
    // ------------------------------------------------------------------

    #[test]
    fn test_temperature_sampler_valid() {
        let sampler = TemperatureSampler::new(1.0, 0);
        assert!(sampler.is_ok(), "construction with temp=1.0 should succeed");
    }

    #[test]
    fn test_temperature_sampler_zero_temp_error() {
        let result = TemperatureSampler::new(0.0, 0);
        assert!(
            matches!(result, Err(SamplingError::InvalidTemperature(t)) if t == 0.0),
            "expected InvalidTemperature, got {result:?}"
        );
    }

    #[test]
    fn test_temperature_sampler_sample_returns_valid_token() {
        let mut sampler = TemperatureSampler::new(1.0, 42).expect("valid");
        let lgs = logits_5();
        let token = sampler.sample(&lgs).expect("sample should succeed");
        assert!(token.token_id < lgs.len(), "token_id out of vocab");
    }

    #[test]
    fn test_temperature_sampler_prob_in_range() {
        let mut sampler = TemperatureSampler::new(1.0, 7).expect("valid");
        let token = sampler.sample(&logits_5()).expect("sample should succeed");
        assert!(
            (0.0..=1.0).contains(&token.prob),
            "prob {} is out of [0, 1]",
            token.prob
        );
    }

    // ------------------------------------------------------------------
    // TopKSampler / apply_top_k
    // ------------------------------------------------------------------

    #[test]
    fn test_top_k_apply_filter_keeps_k() {
        let logits = logits_5();
        let k = 2_usize;
        let filtered = TopKSampler::apply_top_k(&logits, k);
        let finite_count = filtered.iter().filter(|&&v| v.is_finite()).count();
        assert_eq!(
            finite_count, k,
            "expected exactly {k} finite values, got {finite_count}"
        );
    }

    #[test]
    fn test_top_k_sampler_sample_within_vocab() {
        let mut sampler = TopKSampler::new(3, 1.0, 99).expect("valid");
        let lgs = logits_5();
        let token = sampler.sample(&lgs).expect("sample should succeed");
        assert!(token.token_id < lgs.len(), "token_id out of vocab");
    }

    #[test]
    fn test_top_k_zero_k_error() {
        let result = TopKSampler::new(0, 1.0, 0);
        assert!(
            matches!(result, Err(SamplingError::InvalidTopK { k: 0 })),
            "expected InvalidTopK, got {result:?}"
        );
    }

    // ------------------------------------------------------------------
    // TopPSampler / apply_top_p
    // ------------------------------------------------------------------

    #[test]
    fn test_top_p_apply_filter() {
        // Use a peaked distribution so that nucleus is well-defined.
        let probs = vec![0.5, 0.3, 0.15, 0.04, 0.01];
        let p = 0.8_f64;
        let filtered_logits = TopPSampler::apply_top_p(&probs, p);
        // The sum of exp of finite entries should be >= p of the total.
        let nucleus_prob_sum: f64 = filtered_logits
            .iter()
            .filter(|&&v| v.is_finite())
            .map(|&v| v.exp())
            .sum();
        // The nucleus should account for at least p of the total mass.
        assert!(
            nucleus_prob_sum >= p - 1e-9,
            "nucleus prob sum {nucleus_prob_sum} < p={p}"
        );
    }

    #[test]
    fn test_top_p_sampler_sample_valid() {
        let mut sampler = TopPSampler::new(0.9, 1.0, 1).expect("valid");
        let lgs = logits_5();
        let token = sampler.sample(&lgs).expect("sample should succeed");
        assert!(token.token_id < lgs.len());
    }

    #[test]
    fn test_top_p_invalid_p_error() {
        let result = TopPSampler::new(1.5, 1.0, 0);
        assert!(
            matches!(result, Err(SamplingError::InvalidTopP { p }) if p == 1.5),
            "expected InvalidTopP, got {result:?}"
        );
    }

    // ------------------------------------------------------------------
    // ConfigurableSampler
    // ------------------------------------------------------------------

    #[test]
    fn test_configurable_sampler_default() {
        let sampler = ConfigurableSampler::with_default();
        assert_eq!(sampler.config.temperature, 1.0);
    }

    #[test]
    fn test_configurable_sampler_with_top_k() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: Some(5),
            top_p: None,
            repetition_penalty: 1.0,
            seed: Some(0),
        };
        let mut sampler = ConfigurableSampler::new(config).expect("valid config");
        let lgs = logits_5();
        let token = sampler.sample(&lgs, &[]).expect("sample should succeed");
        assert!(token.token_id < lgs.len());
    }

    #[test]
    fn test_repetition_penalty_reduces_seen_tokens() {
        let logits = vec![1.0, 2.0, 3.0];
        let mut working = logits.clone();
        let context = vec![2_usize]; // token 2 has logit 3.0
        ConfigurableSampler::apply_repetition_penalty(&mut working, &context, 2.0);
        // Positive logit → divided by penalty.
        assert!(
            working[2] < logits[2],
            "expected logit[2] to decrease; was {}, now {}",
            logits[2],
            working[2]
        );
        // Token 0 and 1 should be unchanged.
        assert_eq!(working[0], logits[0]);
        assert_eq!(working[1], logits[1]);
    }

    // ------------------------------------------------------------------
    // Utility functions
    // ------------------------------------------------------------------

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 0.5, -1.0];
        let probs = softmax(&logits);
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "softmax sum={total}");
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Very large values must not produce NaN or infinity.
        let logits = vec![1000.0, 999.0, 998.0];
        let probs = softmax(&logits);
        for &p in &probs {
            assert!(p.is_finite() && p >= 0.0, "non-finite probability: {p}");
        }
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "softmax sum={total}");
    }

    #[test]
    fn test_log_softmax_matches_log_of_softmax() {
        let logits = vec![0.5, -1.0, 2.3, 0.0];
        let sm = softmax(&logits);
        let lsm = log_softmax(&logits);
        for (s, ls) in sm.iter().zip(lsm.iter()) {
            let expected = s.ln();
            assert!(
                (expected - ls).abs() < 1e-10,
                "log(softmax)={expected} vs log_softmax={ls}"
            );
        }
    }

    #[test]
    fn test_entropy_uniform() {
        // entropy([0.5, 0.5]) in nats = ln(2) ≈ 0.693147
        let probs = vec![0.5, 0.5];
        let h = entropy(&probs);
        let expected = (2.0_f64).ln();
        assert!(
            (h - expected).abs() < 1e-12,
            "entropy={h} expected={expected}"
        );
    }

    #[test]
    fn test_perplexity_basic() {
        // perplexity([-1.0]) = exp(1.0) ≈ 2.71828
        let log_probs = vec![-1.0_f64];
        let ppl = perplexity(&log_probs);
        let expected = 1.0_f64.exp();
        assert!(
            (ppl - expected).abs() < 1e-12,
            "perplexity={ppl} expected={expected}"
        );
    }
}
