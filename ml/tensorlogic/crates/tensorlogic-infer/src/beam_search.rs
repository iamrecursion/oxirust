//! Beam Search Decoder for sequence generation.
//!
//! Implements beam search, a heuristic search algorithm that explores the best-B
//! candidate sequences (beams) at each decoding step, keeping only the highest-scoring
//! hypotheses by cumulative log-probability.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for beam search decoding.
#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
    /// Number of beams to keep at each step.
    pub beam_width: usize,
    /// Maximum sequence length (inclusive).
    pub max_length: usize,
    /// Optional end-of-sequence token ID. When a beam generates this token it is
    /// moved to the completed list (subject to `min_length`).
    pub eos_token_id: Option<usize>,
    /// Length penalty exponent α: `score = log_prob / length^α`.
    /// `1.0` gives simple length normalisation; `0.0` disables it.
    pub length_penalty: f64,
    /// Minimum sequence length before EOS is allowed to terminate a beam.
    pub min_length: usize,
    /// Number of tokens in the vocabulary.
    pub vocab_size: usize,
    /// Temperature for logit scaling before softmax.  `1.0` = no change.
    pub temperature: f64,
    /// Optional top-k filter: only the top-k logits are kept per step.
    pub top_k_filter: Option<usize>,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_length: 50,
            eos_token_id: None,
            length_penalty: 1.0,
            min_length: 1,
            vocab_size: 1000,
            temperature: 1.0,
            top_k_filter: None,
        }
    }
}

// ---------------------------------------------------------------------------
// BeamHypothesis
// ---------------------------------------------------------------------------

/// A single hypothesis (candidate sequence) tracked during beam search.
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    /// Token IDs generated so far (including the seed/BOS token).
    pub tokens: Vec<usize>,
    /// Cumulative log-probability of the sequence.
    pub log_prob: f64,
    /// Length-penalised score used for ranking.
    pub score: f64,
    /// Whether this hypothesis has terminated (EOS or max length reached).
    pub is_done: bool,
}

impl BeamHypothesis {
    /// Create a new hypothesis seeded with a single token.
    pub fn new(initial_token: usize, log_prob: f64) -> Self {
        let tokens = vec![initial_token];
        let score = log_prob; // length = 1, any alpha => log_prob / 1.0
        Self {
            tokens,
            log_prob,
            score,
            is_done: false,
        }
    }

    /// Extend this hypothesis by one token, returning a new hypothesis.
    pub fn extend(&self, token: usize, token_log_prob: f64) -> Self {
        let mut tokens = self.tokens.clone();
        tokens.push(token);
        let log_prob = self.log_prob + token_log_prob;
        let score = log_prob; // score is updated by caller via length_penalized_score
        Self {
            tokens,
            log_prob,
            score,
            is_done: false,
        }
    }

    /// Compute the length-penalised score: `log_prob / length^alpha`.
    pub fn length_penalized_score(&self, alpha: f64) -> f64 {
        let len = self.tokens.len() as f64;
        if alpha == 0.0 || len == 0.0 {
            self.log_prob
        } else {
            self.log_prob / len.powf(alpha)
        }
    }

    /// Number of tokens in this hypothesis.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns `true` if the hypothesis contains no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

// ---------------------------------------------------------------------------
// BeamStepInput
// ---------------------------------------------------------------------------

/// Log-probabilities (or raw logits) for each beam at one decoding step.
///
/// Shape: `[beam_width][vocab_size]` — each row is the distribution for one beam.
pub struct BeamStepInput {
    /// `log_probs[beam_i][vocab_j]` = log P(token j | history of beam i).
    pub log_probs: Vec<Vec<f64>>,
}

impl BeamStepInput {
    /// Construct directly from pre-computed log-probabilities.
    pub fn new(log_probs: Vec<Vec<f64>>) -> Self {
        Self { log_probs }
    }

    /// Construct from raw logits: apply temperature scaling then log-softmax.
    pub fn from_logits(logits: Vec<Vec<f64>>, temperature: f64) -> Self {
        let log_probs = logits
            .into_iter()
            .map(|row| {
                let scaled = BeamSearchDecoder::apply_temperature(&row, temperature);
                BeamSearchDecoder::log_softmax(&scaled)
            })
            .collect();
        Self { log_probs }
    }

    /// Number of beams (rows) in this input.
    pub fn num_beams(&self) -> usize {
        self.log_probs.len()
    }

    /// Vocabulary size inferred from the first row (0 if empty).
    pub fn vocab_size(&self) -> usize {
        self.log_probs.first().map(|r| r.len()).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// BeamState
// ---------------------------------------------------------------------------

/// Complete state of a beam search at a given decoding step.
#[derive(Debug, Clone)]
pub struct BeamState {
    /// Currently active (non-terminated) beams.
    pub beams: Vec<BeamHypothesis>,
    /// Completed beams (those that emitted EOS or were finalised at max length).
    pub completed: Vec<BeamHypothesis>,
    /// Current step index (0-based; incremented after each call to `step`).
    pub step: usize,
}

impl BeamState {
    /// Create an initial state: `beam_width` identical hypotheses seeded with `bos_token_id`.
    pub fn initial(beam_width: usize, bos_token_id: usize) -> Self {
        let beams = (0..beam_width)
            .map(|_| BeamHypothesis::new(bos_token_id, 0.0))
            .collect();
        Self {
            beams,
            completed: Vec::new(),
            step: 0,
        }
    }

    /// Returns `true` if search is complete: enough completed beams or step limit reached.
    pub fn is_done(&self, config: &BeamSearchConfig) -> bool {
        self.completed.len() >= config.beam_width || self.step >= config.max_length
    }

    /// Return the highest-scored hypothesis across active and completed beams.
    pub fn best_hypothesis(&self) -> Option<&BeamHypothesis> {
        let all = self.beams.iter().chain(self.completed.iter());
        all.max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal))
    }
}

// ---------------------------------------------------------------------------
// Candidate (internal helper for BinaryHeap)
// ---------------------------------------------------------------------------

/// Internal candidate used for ranking during a beam step.
#[derive(Debug)]
struct Candidate {
    beam_idx: usize,
    token_id: usize,
    log_prob: f64, // cumulative log-prob if this candidate is chosen
    score: f64,    // penalised score (used for ranking)
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// BeamSearchError
// ---------------------------------------------------------------------------

/// Errors that can arise during beam search.
#[derive(Debug, Clone)]
pub enum BeamSearchError {
    /// No active beams remain.
    EmptyBeams,
    /// The number of beams in `BeamStepInput` does not match the state.
    BeamWidthMismatch { expected: usize, got: usize },
    /// Vocabulary size in `BeamStepInput` does not match configuration.
    VocabSizeMismatch { expected: usize, got: usize },
    /// `beam_width` is zero, which is invalid.
    ZeroBeamWidth,
    /// `max_length` is too short to produce any output.
    MaxLengthTooShort,
    /// The user-supplied scoring function returned an error.
    ScoringFunctionError(String),
    /// Temperature must be positive.
    InvalidTemperature(f64),
}

impl std::fmt::Display for BeamSearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BeamSearchError::EmptyBeams => write!(f, "beam search has no active beams"),
            BeamSearchError::BeamWidthMismatch { expected, got } => write!(
                f,
                "beam width mismatch: expected {expected} beams, got {got}"
            ),
            BeamSearchError::VocabSizeMismatch { expected, got } => write!(
                f,
                "vocab size mismatch: expected {expected} tokens, got {got}"
            ),
            BeamSearchError::ZeroBeamWidth => write!(f, "beam_width must be at least 1"),
            BeamSearchError::MaxLengthTooShort => {
                write!(f, "max_length must be at least 1")
            }
            BeamSearchError::ScoringFunctionError(msg) => {
                write!(f, "scoring function error: {msg}")
            }
            BeamSearchError::InvalidTemperature(t) => {
                write!(f, "temperature must be positive, got {t}")
            }
        }
    }
}

impl std::error::Error for BeamSearchError {}

// ---------------------------------------------------------------------------
// BeamSearchStats
// ---------------------------------------------------------------------------

/// Statistics collected over a complete beam search run.
#[derive(Debug, Clone)]
pub struct BeamSearchStats {
    /// Total number of decoding steps taken.
    pub total_steps: usize,
    /// Number of beams that ended by generating the EOS token.
    pub num_completed_at_eos: usize,
    /// Number of beams that ended by reaching `max_length`.
    pub num_completed_at_max_length: usize,
    /// Mean sequence length across all final hypotheses.
    pub avg_sequence_length: f64,
    /// `(min_score, max_score)` across all final hypotheses.
    pub score_range: (f64, f64),
}

// ---------------------------------------------------------------------------
// BeamSearchResult
// ---------------------------------------------------------------------------

/// Final output of a complete beam search run.
#[derive(Debug, Clone)]
pub struct BeamSearchResult {
    /// All final hypotheses, sorted by score descending.
    pub hypotheses: Vec<BeamHypothesis>,
    /// Token sequence of the best hypothesis.
    pub best_sequence: Vec<usize>,
    /// Score of the best hypothesis.
    pub best_score: f64,
    /// Aggregate statistics.
    pub stats: BeamSearchStats,
}

impl BeamSearchResult {
    /// Return a reference to the best hypothesis, or `None` if empty.
    pub fn best(&self) -> Option<&BeamHypothesis> {
        self.hypotheses.first()
    }
}

// ---------------------------------------------------------------------------
// BeamSearchDecoder
// ---------------------------------------------------------------------------

/// Beam search decoder.
pub struct BeamSearchDecoder {
    /// Configuration controlling all search behaviour.
    pub config: BeamSearchConfig,
}

impl BeamSearchDecoder {
    /// Create a decoder with the supplied configuration.
    pub fn new(config: BeamSearchConfig) -> Self {
        Self { config }
    }

    /// Create a decoder with default configuration.
    pub fn with_default() -> Self {
        Self::new(BeamSearchConfig::default())
    }

    /// Create an initial `BeamState` seeded with `bos_token_id`.
    pub fn initial_state(&self, bos_token_id: usize) -> BeamState {
        BeamState::initial(self.config.beam_width, bos_token_id)
    }

    /// Perform one step of beam search.
    ///
    /// Given the current `state` and per-beam log-probabilities `input`, advance
    /// each beam by one token, prune to the top-`beam_width` candidates, and
    /// handle EOS/completion logic.
    pub fn step(
        &self,
        mut state: BeamState,
        input: &BeamStepInput,
    ) -> Result<BeamState, BeamSearchError> {
        if self.config.beam_width == 0 {
            return Err(BeamSearchError::ZeroBeamWidth);
        }
        if state.beams.is_empty() {
            // All beams may have already completed; nothing to advance.
            state.step += 1;
            return Ok(state);
        }

        // Validate input dimensions.
        if input.num_beams() != state.beams.len() {
            return Err(BeamSearchError::BeamWidthMismatch {
                expected: state.beams.len(),
                got: input.num_beams(),
            });
        }
        let vocab_size = self.config.vocab_size;
        for (i, row) in input.log_probs.iter().enumerate() {
            if row.len() != vocab_size {
                return Err(BeamSearchError::VocabSizeMismatch {
                    expected: vocab_size,
                    got: row.len(),
                });
            }
            let _ = i;
        }

        // Build a max-heap of all (beam, token) candidates.
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();

        for (beam_idx, beam) in state.beams.iter().enumerate() {
            let mut lp: Vec<f64> = input.log_probs[beam_idx].clone();

            // Apply top-k filter if configured.
            if let Some(k) = self.config.top_k_filter {
                Self::top_k_filter_logits(&mut lp, k);
            }

            for (token_id, &token_lp) in lp.iter().enumerate() {
                // Skip -inf entries (filtered out by top-k).
                if token_lp == f64::NEG_INFINITY {
                    continue;
                }
                let new_log_prob = beam.log_prob + token_lp;
                // Compute penalised score based on hypothetical new length.
                let new_len = (beam.tokens.len() + 1) as f64;
                let score = if self.config.length_penalty == 0.0 {
                    new_log_prob
                } else {
                    new_log_prob / new_len.powf(self.config.length_penalty)
                };

                heap.push(Candidate {
                    beam_idx,
                    token_id,
                    log_prob: new_log_prob,
                    score,
                });
            }
        }

        // Select top beam_width candidates.
        let desired = self.config.beam_width;
        let mut new_beams: Vec<BeamHypothesis> = Vec::with_capacity(desired);
        let mut new_completed: Vec<BeamHypothesis> = state.completed.clone();
        let mut eos_count: usize = 0;
        let mut taken: usize = 0;

        while taken < desired {
            let candidate = match heap.pop() {
                Some(c) => c,
                None => break,
            };

            let parent = &state.beams[candidate.beam_idx];
            let mut hyp = parent.extend(candidate.token_id, 0.0);
            // Override the log_prob computed in extend (which adds 0.0) with the real value.
            hyp.log_prob = candidate.log_prob;
            hyp.score = candidate.score;

            // Check whether this is an EOS token.
            let is_eos = self
                .config
                .eos_token_id
                .map(|eos| candidate.token_id == eos)
                .unwrap_or(false);

            if is_eos && hyp.len() > self.config.min_length {
                // +1 because len includes the BOS token.
                hyp.is_done = true;
                eos_count += 1;
                new_completed.push(hyp);
            } else {
                new_beams.push(hyp);
            }
            taken += 1;
        }

        // Finalise any beams that have reached max_length.
        let (kept_beams, maxlen_beams): (Vec<_>, Vec<_>) = new_beams
            .into_iter()
            .partition(|b| b.len() < self.config.max_length);
        let new_beams = kept_beams;
        for mut beam in maxlen_beams {
            beam.is_done = true;
            new_completed.push(beam);
        }

        let _ = eos_count; // suppress unused warning

        Ok(BeamState {
            beams: new_beams,
            completed: new_completed,
            step: state.step + 1,
        })
    }

    /// Run a full beam search.
    ///
    /// `score_fn` is called at each step with the current beam sequences and must
    /// return log-probabilities of shape `[num_active_beams][vocab_size]`.
    pub fn decode<F>(
        &self,
        bos_token_id: usize,
        score_fn: F,
    ) -> Result<BeamSearchResult, BeamSearchError>
    where
        F: Fn(&[&[usize]]) -> Result<Vec<Vec<f64>>, String>,
    {
        if self.config.beam_width == 0 {
            return Err(BeamSearchError::ZeroBeamWidth);
        }
        if self.config.max_length == 0 {
            return Err(BeamSearchError::MaxLengthTooShort);
        }
        if self.config.temperature <= 0.0 {
            return Err(BeamSearchError::InvalidTemperature(self.config.temperature));
        }

        let mut state = self.initial_state(bos_token_id);
        while !state.is_done(&self.config) {
            if state.beams.is_empty() {
                break;
            }

            // Build token slices for the score function.
            let beam_seqs: Vec<&[usize]> =
                state.beams.iter().map(|b| b.tokens.as_slice()).collect();

            let raw_logits = score_fn(&beam_seqs).map_err(BeamSearchError::ScoringFunctionError)?;

            // Apply temperature and convert to log-probabilities.
            let log_probs: Vec<Vec<f64>> = raw_logits
                .into_iter()
                .map(|row| {
                    let scaled = Self::apply_temperature(&row, self.config.temperature);
                    Self::log_softmax(&scaled)
                })
                .collect();

            let input = BeamStepInput::new(log_probs);
            state = self.step(state, &input)?;
        }

        // Finalise any remaining active beams.
        let remaining: Vec<BeamHypothesis> = state.beams.drain(..).collect();
        for mut beam in remaining {
            beam.is_done = true;
            state.completed.push(beam);
        }

        // Count completion reasons.
        let mut eos_completed: usize = 0;
        let mut max_len_completed: usize = 0;
        for hyp in &state.completed {
            if let Some(eos) = self.config.eos_token_id {
                if hyp.tokens.last().copied() == Some(eos) {
                    eos_completed += 1;
                } else {
                    max_len_completed += 1;
                }
            } else {
                max_len_completed += 1;
            }
        }

        let total_steps = state.step;

        // Sort completed by score descending.
        let mut hypotheses = state.completed;
        hypotheses.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        let best_sequence = hypotheses
            .first()
            .map(|h| h.tokens.clone())
            .unwrap_or_default();
        let best_score = hypotheses
            .first()
            .map(|h| h.score)
            .unwrap_or(f64::NEG_INFINITY);

        let avg_sequence_length = if hypotheses.is_empty() {
            0.0
        } else {
            hypotheses.iter().map(|h| h.len() as f64).sum::<f64>() / hypotheses.len() as f64
        };

        let score_range = if hypotheses.is_empty() {
            (0.0, 0.0)
        } else {
            let min_score = hypotheses
                .iter()
                .map(|h| h.score)
                .fold(f64::INFINITY, f64::min);
            let max_score = hypotheses
                .iter()
                .map(|h| h.score)
                .fold(f64::NEG_INFINITY, f64::max);
            (min_score, max_score)
        };

        let stats = BeamSearchStats {
            total_steps,
            num_completed_at_eos: eos_completed,
            num_completed_at_max_length: max_len_completed,
            avg_sequence_length,
            score_range,
        };

        Ok(BeamSearchResult {
            hypotheses,
            best_sequence,
            best_score,
            stats,
        })
    }

    /// Extract the top-`k` hypotheses from a beam state, sorted by score descending.
    pub fn top_k_results(&self, state: &BeamState, k: usize) -> Vec<BeamHypothesis> {
        let mut all: Vec<BeamHypothesis> = state
            .beams
            .iter()
            .chain(state.completed.iter())
            .cloned()
            .collect();
        all.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        all.truncate(k);
        all
    }

    /// Apply temperature scaling to logits: `logits[i] / temperature`.
    pub fn apply_temperature(logits: &[f64], temperature: f64) -> Vec<f64> {
        if temperature == 1.0 {
            return logits.to_vec();
        }
        let t = if temperature == 0.0 {
            1e-8
        } else {
            temperature
        };
        logits.iter().map(|&x| x / t).collect()
    }

    /// Compute numerically stable log-softmax.
    ///
    /// Uses the log-sum-exp trick:
    /// `lse = max + log(sum(exp(x - max)))`,  `log_softmax(x_i) = x_i - lse`.
    pub fn log_softmax(logits: &[f64]) -> Vec<f64> {
        if logits.is_empty() {
            return Vec::new();
        }
        let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = logits.iter().map(|&x| (x - max_val).exp()).sum();
        let log_sum_exp = max_val + sum_exp.ln();
        logits.iter().map(|&x| x - log_sum_exp).collect()
    }

    /// Zero out all but the top-`k` logits in place (set others to `NEG_INFINITY`).
    pub fn top_k_filter_logits(logits: &mut [f64], k: usize) {
        if k == 0 || logits.is_empty() {
            for v in logits.iter_mut() {
                *v = f64::NEG_INFINITY;
            }
            return;
        }
        if k >= logits.len() {
            return; // Nothing to filter.
        }

        // Find the k-th largest value as a threshold.
        let mut sorted: Vec<f64> = logits.to_owned();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let threshold = sorted[k - 1];

        // Keep only those >= threshold (tie-breaking: we allow exactly k through).
        let mut kept = 0usize;
        for v in logits.iter_mut() {
            if *v >= threshold && kept < k {
                kept += 1;
            } else {
                *v = f64::NEG_INFINITY;
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

    /// Helper: build a constant score function that returns uniform log-probs.
    fn uniform_score_fn(
        vocab_size: usize,
    ) -> impl Fn(&[&[usize]]) -> Result<Vec<Vec<f64>>, String> {
        let lp = -(vocab_size as f64).ln();
        move |beams: &[&[usize]]| Ok(beams.iter().map(|_| vec![lp; vocab_size]).collect())
    }

    #[test]
    fn test_beam_search_config_default() {
        let cfg = BeamSearchConfig::default();
        assert_eq!(cfg.beam_width, 4);
        assert_eq!(cfg.max_length, 50);
        assert_eq!(cfg.eos_token_id, None);
        assert_eq!(cfg.length_penalty, 1.0);
        assert_eq!(cfg.temperature, 1.0);
    }

    #[test]
    fn test_beam_hypothesis_new() {
        let h = BeamHypothesis::new(0, -0.5);
        assert_eq!(h.len(), 1);
        assert_eq!(h.tokens, vec![0]);
        assert!(!h.is_done);
    }

    #[test]
    fn test_beam_hypothesis_extend() {
        let h = BeamHypothesis::new(0, -0.5);
        let h2 = h.extend(7, -1.0);
        assert_eq!(h2.len(), 2);
        assert_eq!(h2.tokens, vec![0, 7]);
        assert!((h2.log_prob - (-1.5)).abs() < 1e-10);
        assert!(!h2.is_done);
    }

    #[test]
    fn test_beam_hypothesis_length_penalized_score_no_penalty() {
        // alpha = 1.0: score = log_prob / length
        let h = BeamHypothesis::new(0, 0.0);
        let h2 = h.extend(1, -2.0);
        // log_prob = -2.0, length = 2
        let score = h2.length_penalized_score(1.0);
        assert!((score - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_beam_step_input_from_logits() {
        let logits = vec![vec![1.0, 2.0, 3.0], vec![0.5, 0.5, 0.5]];
        let input = BeamStepInput::from_logits(logits, 1.0);
        // Each row's exp should sum to ~1.
        for row in &input.log_probs {
            let sum: f64 = row.iter().map(|&lp| lp.exp()).sum();
            assert!((sum - 1.0).abs() < 1e-9, "sum was {sum}");
        }
    }

    #[test]
    fn test_beam_step_input_vocab_size() {
        let lp = vec![vec![0.1, 0.2, 0.7]; 3];
        let input = BeamStepInput::new(lp);
        assert_eq!(input.vocab_size(), 3);
        assert_eq!(input.num_beams(), 3);
    }

    #[test]
    fn test_beam_state_initial() {
        let state = BeamState::initial(4, 0);
        assert_eq!(state.beams.len(), 4);
        assert_eq!(state.completed.len(), 0);
        assert_eq!(state.step, 0);
        for b in &state.beams {
            assert_eq!(b.tokens, vec![0]);
        }
    }

    #[test]
    fn test_beam_state_is_done_max_length() {
        let cfg = BeamSearchConfig {
            max_length: 3,
            ..BeamSearchConfig::default()
        };
        let mut state = BeamState::initial(4, 0);
        assert!(!state.is_done(&cfg));
        state.step = 3;
        assert!(state.is_done(&cfg));
    }

    #[test]
    fn test_decoder_step_advances_state() {
        let cfg = BeamSearchConfig {
            beam_width: 2,
            vocab_size: 5,
            ..BeamSearchConfig::default()
        };
        let decoder = BeamSearchDecoder::new(cfg);
        let state = decoder.initial_state(0);
        let lp = BeamSearchDecoder::log_softmax(&[1.0; 5]);
        let input = BeamStepInput::new(vec![lp.clone(), lp]);
        let new_state = decoder.step(state, &input).expect("step failed");
        assert_eq!(new_state.step, 1);
    }

    #[test]
    fn test_decoder_step_beam_count() {
        let beam_width = 3;
        let vocab_size = 10;
        let cfg = BeamSearchConfig {
            beam_width,
            vocab_size,
            ..BeamSearchConfig::default()
        };
        let decoder = BeamSearchDecoder::new(cfg);
        let state = decoder.initial_state(0);
        let lp = BeamSearchDecoder::log_softmax(&vec![1.0; vocab_size]);
        let input = BeamStepInput::new(vec![lp; beam_width]);
        let new_state = decoder.step(state, &input).expect("step failed");
        // Active + completed should together contain beam_width hypotheses.
        assert_eq!(
            new_state.beams.len() + new_state.completed.len(),
            beam_width
        );
    }

    #[test]
    fn test_decoder_step_eos_moves_to_completed() {
        let eos = 1_usize;
        let vocab_size = 5;
        let beam_width = 2;
        let cfg = BeamSearchConfig {
            beam_width,
            vocab_size,
            eos_token_id: Some(eos),
            min_length: 1,
            ..BeamSearchConfig::default()
        };
        let decoder = BeamSearchDecoder::new(cfg);
        let state = decoder.initial_state(0);

        // Strongly bias logits towards token 1 (EOS) for all beams.
        let mut logits = vec![f64::NEG_INFINITY; vocab_size];
        logits[eos] = 100.0; // overwhelming preference for EOS
        let lp = BeamSearchDecoder::log_softmax(&logits);
        let input = BeamStepInput::new(vec![lp; beam_width]);

        let new_state = decoder.step(state, &input).expect("step failed");
        // With EOS strongly preferred, we expect some beams to complete.
        // (All beam_width candidates emit EOS, so they all go to completed.)
        assert!(!new_state.completed.is_empty(), "expected completed beams");
    }

    #[test]
    fn test_decoder_step_vocab_size_mismatch() {
        let cfg = BeamSearchConfig {
            beam_width: 2,
            vocab_size: 10,
            ..BeamSearchConfig::default()
        };
        let decoder = BeamSearchDecoder::new(cfg);
        let state = decoder.initial_state(0);
        // Provide 5 tokens instead of 10.
        let lp = vec![0.2; 5];
        let input = BeamStepInput::new(vec![lp; 2]);
        let result = decoder.step(state, &input);
        assert!(matches!(
            result,
            Err(BeamSearchError::VocabSizeMismatch { .. })
        ));
    }

    #[test]
    fn test_decoder_log_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let lsp = BeamSearchDecoder::log_softmax(&logits);
        let sum: f64 = lsp.iter().map(|&x| x.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum = {sum}");
    }

    #[test]
    fn test_decoder_top_k_filter() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        BeamSearchDecoder::top_k_filter_logits(&mut logits, 2);
        // Only the two largest (5.0 at index 1, 4.0 at index 4) should survive.
        let non_neg_inf: Vec<usize> = logits
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != f64::NEG_INFINITY)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(non_neg_inf.len(), 2);
        // Indices 1 and 4 are kept.
        assert!(non_neg_inf.contains(&1));
        assert!(non_neg_inf.contains(&4));
    }

    #[test]
    fn test_decoder_decode_simple() {
        let vocab_size = 8;
        let cfg = BeamSearchConfig {
            beam_width: 2,
            max_length: 5,
            vocab_size,
            ..BeamSearchConfig::default()
        };
        let decoder = BeamSearchDecoder::new(cfg);
        let score_fn = uniform_score_fn(vocab_size);
        let result = decoder.decode(0, score_fn);
        assert!(result.is_ok(), "decode returned error: {:?}", result.err());
    }

    #[test]
    fn test_beam_search_result_best() {
        let h1 = BeamHypothesis {
            tokens: vec![0, 1],
            log_prob: -1.0,
            score: -1.0,
            is_done: true,
        };
        let h2 = BeamHypothesis {
            tokens: vec![0, 2],
            log_prob: -0.5,
            score: -0.5,
            is_done: true,
        };
        let result = BeamSearchResult {
            best_sequence: h2.tokens.clone(),
            best_score: h2.score,
            hypotheses: vec![h2.clone(), h1.clone()],
            stats: BeamSearchStats {
                total_steps: 1,
                num_completed_at_eos: 0,
                num_completed_at_max_length: 2,
                avg_sequence_length: 2.0,
                score_range: (-1.0, -0.5),
            },
        };
        let best = result.best().expect("should have best");
        assert_eq!(best.score, -0.5);
    }

    #[test]
    fn test_beam_search_stats() {
        let vocab_size = 4;
        let cfg = BeamSearchConfig {
            beam_width: 2,
            max_length: 4,
            vocab_size,
            ..BeamSearchConfig::default()
        };
        let decoder = BeamSearchDecoder::new(cfg);
        let score_fn = uniform_score_fn(vocab_size);
        let result = decoder.decode(0, score_fn).expect("decode failed");
        assert!(result.stats.total_steps > 0);
    }

    #[test]
    fn test_top_k_results_sorted() {
        let decoder = BeamSearchDecoder::with_default();
        let make_hyp = |score: f64| BeamHypothesis {
            tokens: vec![0],
            log_prob: score,
            score,
            is_done: false,
        };
        let state = BeamState {
            beams: vec![make_hyp(-2.0), make_hyp(-0.5), make_hyp(-3.0)],
            completed: vec![make_hyp(-1.0)],
            step: 1,
        };
        let top = decoder.top_k_results(&state, 3);
        assert_eq!(top.len(), 3);
        // Sorted descending.
        assert!(top[0].score >= top[1].score);
        assert!(top[1].score >= top[2].score);
        assert!((top[0].score - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_decoder_temperature_scaling() {
        let logits = vec![1.0, 2.0, 3.0];
        let lp1 =
            BeamSearchDecoder::log_softmax(&BeamSearchDecoder::apply_temperature(&logits, 1.0));
        let lp2 =
            BeamSearchDecoder::log_softmax(&BeamSearchDecoder::apply_temperature(&logits, 2.0));
        // Higher temperature => flatter distribution (less spread in log-probs).
        let spread1 = lp1.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - lp1.iter().cloned().fold(f64::INFINITY, f64::min);
        let spread2 = lp2.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - lp2.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            spread2 < spread1,
            "temperature=2.0 should flatten distribution"
        );
    }

    #[test]
    fn test_beam_search_error_display() {
        let errors = vec![
            BeamSearchError::EmptyBeams,
            BeamSearchError::BeamWidthMismatch {
                expected: 4,
                got: 2,
            },
            BeamSearchError::VocabSizeMismatch {
                expected: 1000,
                got: 500,
            },
            BeamSearchError::ZeroBeamWidth,
            BeamSearchError::MaxLengthTooShort,
            BeamSearchError::ScoringFunctionError("test error".to_string()),
            BeamSearchError::InvalidTemperature(-1.0),
        ];
        for err in &errors {
            let s = err.to_string();
            assert!(!s.is_empty(), "display for {err:?} was empty");
        }
    }
}
