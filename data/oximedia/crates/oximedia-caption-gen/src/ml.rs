//! ML-assisted caption encoding backed by [`oximedia_ml::OnnxModel`].
//!
//! This module provides [`CaptionEncoder`], a thin, generic wrapper around a
//! pre-loaded ONNX model that returns the raw logits produced by a
//! caption / sequence-to-sequence encoder.  It is deliberately **additive**
//! — nothing in the existing speech-alignment / Knuth–Plass line-breaking /
//! WCAG / speaker-diarization surface depends on it, and the entire module
//! is gated behind the crate's `onnx` Cargo feature so default builds remain
//! free of ONNX symbols.
//!
//! # Design
//!
//! End-to-end caption generation is intentionally out of scope for this
//! module: real captioning stacks differ wildly in tokenisation,
//! preprocessing, beam-search strategy, language model, and post-processing.
//! What [`CaptionEncoder`] *does* commit to is a narrow contract:
//!
//! * Caller supplies a preprocessed `&[f32]` input tensor plus its
//!   `&[usize]` shape.
//! * The encoder runs forward inference and returns [`EncoderOutput`]
//!   containing the flat `Vec<f32>` logits and their `Vec<usize>` shape.
//! * Downstream code decodes those logits into token ids via
//!   [`greedy_decode`] or [`top_k_sample`] — both pure helpers that live
//!   below and do not touch the ONNX session.
//!
//! Preprocessing (audio feature extraction, spectrograms, tokenisation)
//! and decoding (beam search, language model fusion, post-edit) are the
//! **caller's** responsibility.  This layer is the smallest reusable core
//! that every caption pipeline needs — load a model, run it, get logits
//! back — and nothing more.
//!
//! A fuller AutoCaption end-to-end encoder-decoder pipeline is tracked in a
//! separate wave (Wave 2C in the 0.1.5 programme) and will build on top of
//! the primitives exposed here.
//!
//! # Error mapping
//!
//! Every fallible operation returns [`crate::CaptionGenResult`].  The
//! [`oximedia_ml::MlError`] type is folded into [`crate::CaptionGenError`]
//! via `thiserror`'s `#[from]` conversion declared on the `Ml` variant.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_caption_gen::{CaptionEncoder, greedy_decode};
//! use oximedia_ml::DeviceType;
//!
//! # fn run() -> oximedia_caption_gen::CaptionGenResult<()> {
//! let encoder = CaptionEncoder::from_path(
//!     "caption_encoder.onnx",
//!     DeviceType::auto(),
//! )?;
//!
//! // Pretend we already preprocessed audio into a log-mel spectrogram.
//! let tensor = vec![0.0_f32; 1 * 80 * 3000];
//! let shape = [1_usize, 80, 3000];
//! let out = encoder.encode(&tensor, &shape)?;
//!
//! // Shape: [batch=1, seq_len, vocab]. Greedy decode picks argmax per step.
//! let seq_len = out.shape.get(1).copied().unwrap_or(0);
//! let vocab = out.shape.last().copied().unwrap_or(0);
//! let tokens = greedy_decode(&out.logits, vocab, seq_len)?;
//! println!("decoded {} tokens", tokens.len());
//! # Ok(()) }
//! ```

use std::path::Path;
use std::sync::Arc;

use oximedia_ml::{
    argmax, softmax, top_k, DeviceType, MlError, ModelCache, OnnxModel, PipelineInfo, PipelineTask,
};

use crate::{CaptionGenError, CaptionGenResult};

/// Sentinel input-tensor name used when the model advertises no inputs.
///
/// Well-formed caption encoders always expose at least one input, so this
/// string only surfaces if a malformed graph is loaded — in which case
/// [`CaptionEncoder::encode`] will fail loudly with an `Ml` error before
/// the sentinel ever reaches disk.
const FALLBACK_INPUT_NAME: &str = "input";

/// Raw logits returned by [`CaptionEncoder::encode`].
///
/// The layout is intentionally generic: `shape` captures the model's
/// declared output rank (e.g. `[batch, seq_len, vocab]` for a typical
/// autoregressive encoder-decoder) and `logits` is the flat row-major
/// buffer.  Downstream decoders ([`greedy_decode`], [`top_k_sample`], or
/// any bespoke beam search) interpret the last dimension as the token
/// dimension.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderOutput {
    /// Flat row-major logits buffer.
    pub logits: Vec<f32>,
    /// Shape of `logits` as declared by the model's output tensor.
    pub shape: Vec<usize>,
}

impl EncoderOutput {
    /// Number of elements in the logits buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.logits.len()
    }

    /// `true` when the output has no logits.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.logits.is_empty()
    }
}

/// Opt-in ML caption encoder that produces raw logits from a preprocessed
/// input tensor.
///
/// Wraps an [`OnnxModel`] directly rather than a typed pipeline so any ONNX
/// caption / seq2seq encoder (Whisper encoder, custom conformer, …) can be
/// used.  Callers remain responsible for preprocessing inputs into the
/// shape expected by the model, and for decoding the returned logits into
/// token strings.
pub struct CaptionEncoder {
    model: Arc<OnnxModel>,
    input_name: String,
    output_name: String,
}

impl CaptionEncoder {
    /// Load a caption encoder ONNX model from disk.
    ///
    /// Default input / output tensor names are resolved from the model's
    /// [`oximedia_ml::ModelInfo`] (first input / first output).  Override
    /// them via [`Self::with_input_name`] / [`Self::with_output_name`]
    /// when a model has auxiliary heads.
    ///
    /// # Errors
    ///
    /// * Returns [`CaptionGenError::Ml`] wrapping
    ///   [`oximedia_ml::MlError::ModelLoad`] if the ONNX file cannot be
    ///   opened.
    /// * Returns [`CaptionGenError::Ml`] wrapping
    ///   [`oximedia_ml::MlError::DeviceUnavailable`] if the requested
    ///   device is not compiled in or unavailable at runtime.
    pub fn from_path(model_path: impl AsRef<Path>, device: DeviceType) -> CaptionGenResult<Self> {
        let path = model_path.as_ref();
        let model = Arc::new(OnnxModel::load(path, device)?);
        Ok(Self::build(model))
    }

    /// Build an encoder from a shared [`OnnxModel`] (typically resolved via
    /// a [`ModelCache`]).
    ///
    /// Useful when multiple pipelines share the same weights — for example
    /// live captioning and offline transcript generation both dispatching
    /// through the same encoder instance.
    ///
    /// # Errors
    ///
    /// The current implementation does not fail, but the signature returns
    /// [`CaptionGenResult`] to preserve API stability in case future builders
    /// need to validate model metadata at construction time (e.g. reject
    /// models lacking a configured output tensor).  Callers should use `?` as
    /// if it were fallible.
    pub fn from_shared_model(model: Arc<OnnxModel>) -> CaptionGenResult<Self> {
        Ok(Self::build(model))
    }

    /// Resolve an encoder against a [`ModelCache`], sharing the
    /// `OnnxModel` with any other caller that loaded the same path.
    ///
    /// # Errors
    ///
    /// Propagates any [`oximedia_ml::MlError`] raised by the cache loader.
    pub fn from_cache(
        cache: &ModelCache,
        model_path: impl AsRef<Path>,
        device: DeviceType,
    ) -> CaptionGenResult<Self> {
        let model = cache.get_or_load(model_path.as_ref(), device)?;
        Self::from_shared_model(model)
    }

    fn build(model: Arc<OnnxModel>) -> Self {
        let info = model.info();
        let input_name = info
            .inputs
            .first()
            .map(|spec| spec.name.clone())
            .unwrap_or_else(|| FALLBACK_INPUT_NAME.to_string());
        let output_name = info
            .outputs
            .first()
            .map(|spec| spec.name.clone())
            .unwrap_or_default();
        Self {
            model,
            input_name,
            output_name,
        }
    }

    /// Builder-style setter overriding the input tensor name.
    #[must_use]
    pub fn with_input_name(mut self, name: impl Into<String>) -> Self {
        self.input_name = name.into();
        self
    }

    /// Builder-style setter overriding the output tensor name read back
    /// as logits.  Necessary for multi-head models whose first output is
    /// not the caption head.
    #[must_use]
    pub fn with_output_name(mut self, name: impl Into<String>) -> Self {
        self.output_name = name.into();
        self
    }

    /// Currently configured input tensor name.
    #[must_use]
    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    /// Currently configured output tensor name.
    #[must_use]
    pub fn output_name(&self) -> &str {
        &self.output_name
    }

    /// Shared handle to the underlying [`OnnxModel`].
    #[must_use]
    pub fn shared_model(&self) -> Arc<OnnxModel> {
        self.model.clone()
    }

    /// Static description of the encoder, conforming to the
    /// [`oximedia_ml::TypedPipeline::info`] convention used by every other
    /// pipeline in the `oximedia-ml` zoo.
    ///
    /// The encoder is intentionally declared as
    /// [`PipelineTask::Custom`] because caption encoders do not fall into
    /// any of the built-in pipeline categories (classification, detection,
    /// segmentation, …) — they are generic token-logit producers.
    #[must_use]
    pub fn info(&self) -> PipelineInfo {
        PipelineInfo {
            id: "caption-gen/custom-encoder",
            name: "Caption Encoder",
            task: PipelineTask::Custom,
            input_size: None,
        }
    }

    /// Run forward inference on a preprocessed input tensor and return the
    /// raw logits plus their shape.
    ///
    /// `tensor.len()` must equal `shape.iter().product::<usize>()`.  The
    /// logits buffer is copied out of the ONNX session; no internal state
    /// from the model is retained.
    ///
    /// # Errors
    ///
    /// * [`CaptionGenError::Ml`] wrapping any error raised by the
    ///   underlying ONNX session (shape mismatch, runtime failure, …).
    /// * [`CaptionGenError::Ml`] if the configured output tensor name is
    ///   missing from the model's output map.
    pub fn encode(&self, tensor: &[f32], shape: &[usize]) -> CaptionGenResult<EncoderOutput> {
        let shape_vec = shape.to_vec();
        let data = tensor.to_vec();
        let mut outputs = self.model.run_single(&self.input_name, data, shape_vec)?;
        let raw = outputs.remove(&self.output_name).ok_or_else(|| {
            CaptionGenError::Ml(MlError::pipeline(
                "caption-gen",
                format!("output '{}' missing from model", self.output_name),
            ))
        })?;
        // The oximedia-ml surface returns only the flat f32 buffer, not the
        // tensor shape.  For decoder pipelines that need a shape, we fall
        // back to a [1, raw.len()] interpretation and let callers override
        // via model metadata when they have it.  Most encoder-decoder
        // models expose a static output shape through `ModelInfo`, which we
        // consult here to preserve rank information whenever it's static.
        let info = self.model.info();
        let declared_shape: Option<Vec<usize>> = info
            .outputs
            .iter()
            .find(|spec| spec.name == self.output_name)
            .and_then(|spec| {
                let mut dims = Vec::with_capacity(spec.shape.len());
                for d in &spec.shape {
                    match d {
                        Some(v) if *v > 0 => dims.push(*v as usize),
                        _ => return None,
                    }
                }
                if dims.iter().product::<usize>() == raw.len() {
                    Some(dims)
                } else {
                    None
                }
            });
        let out_shape = declared_shape.unwrap_or_else(|| vec![raw.len()]);
        Ok(EncoderOutput {
            logits: raw,
            shape: out_shape,
        })
    }
}

impl std::fmt::Debug for CaptionEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CaptionEncoder")
            .field("input_name", &self.input_name)
            .field("output_name", &self.output_name)
            .finish()
    }
}

// ── Decoding helpers ────────────────────────────────────────────────────────

/// Greedy-decode a flat logits buffer into a sequence of token ids.
///
/// Interprets `logits` as a `[seq_len, vocab_size]` row-major matrix and
/// picks `argmax` per row (ties break to the lowest index via
/// [`oximedia_ml::argmax`]).  Batch dimension handling is the caller's
/// responsibility — slice your batch before calling this.
///
/// # Errors
///
/// * [`CaptionGenError::InvalidParameter`] if `vocab_size == 0` or
///   `seq_len == 0`.
/// * [`CaptionGenError::InvalidParameter`] if
///   `logits.len() != seq_len * vocab_size`.
/// * [`CaptionGenError::InvalidParameter`] if a row is rejected by
///   [`argmax`] (this should be impossible because rows are non-empty by
///   construction, but we surface the underlying error message for
///   diagnostics).
pub fn greedy_decode(
    logits: &[f32],
    vocab_size: usize,
    seq_len: usize,
) -> CaptionGenResult<Vec<u32>> {
    if vocab_size == 0 {
        return Err(CaptionGenError::InvalidParameter(
            "vocab_size must be > 0".into(),
        ));
    }
    if seq_len == 0 {
        return Err(CaptionGenError::InvalidParameter(
            "seq_len must be > 0".into(),
        ));
    }
    let expected = seq_len.checked_mul(vocab_size).ok_or_else(|| {
        CaptionGenError::InvalidParameter("seq_len * vocab_size overflows usize".into())
    })?;
    if logits.len() != expected {
        return Err(CaptionGenError::InvalidParameter(format!(
            "logits len {} does not match seq_len ({}) * vocab_size ({}) = {}",
            logits.len(),
            seq_len,
            vocab_size,
            expected,
        )));
    }

    let mut out: Vec<u32> = Vec::with_capacity(seq_len);
    for step in 0..seq_len {
        let start = step * vocab_size;
        let end = start + vocab_size;
        let row = &logits[start..end];
        let idx = argmax(row).map_err(|e| {
            CaptionGenError::InvalidParameter(format!(
                "greedy_decode: argmax failed on step {step}: {e:?}"
            ))
        })?;
        out.push(u32_from_usize(idx)?);
    }
    Ok(out)
}

/// Sample token ids from the top-`k` logits of each step using a tiny
/// xorshift64\* PRNG.
///
/// Interprets `logits` as a `[seq_len, vocab_size]` row-major matrix.  For
/// each step:
///   1. The top-`k` entries (by logit value, with the lowest index winning
///      ties — see [`oximedia_ml::top_k`]) are taken from the row.
///   2. Softmax is applied over those `k` logits to obtain a proper
///      probability distribution.
///   3. A pseudo-random `f32` in `[0, 1)` drawn from a xorshift64\* PRNG
///      seeded from `seed + step` selects a token via inverse-CDF.
///
/// The PRNG is deterministic: calling with the same `seed` on the same
/// logits always produces the same token sequence, which matters for
/// reproducible caption generation in testing and replay scenarios.
///
/// If `k == 0` or `k > vocab_size`, the function clamps `k` to
/// `min(k, vocab_size)` and, for `k == 0`, falls back to
/// [`greedy_decode`] semantics (i.e. argmax).
///
/// # Errors
///
/// * [`CaptionGenError::InvalidParameter`] if `vocab_size == 0` or
///   `seq_len == 0`.
/// * [`CaptionGenError::InvalidParameter`] if
///   `logits.len() != seq_len * vocab_size`.
pub fn top_k_sample(
    logits: &[f32],
    vocab_size: usize,
    seq_len: usize,
    k: usize,
    seed: u64,
) -> CaptionGenResult<Vec<u32>> {
    if vocab_size == 0 {
        return Err(CaptionGenError::InvalidParameter(
            "vocab_size must be > 0".into(),
        ));
    }
    if seq_len == 0 {
        return Err(CaptionGenError::InvalidParameter(
            "seq_len must be > 0".into(),
        ));
    }
    let expected = seq_len.checked_mul(vocab_size).ok_or_else(|| {
        CaptionGenError::InvalidParameter("seq_len * vocab_size overflows usize".into())
    })?;
    if logits.len() != expected {
        return Err(CaptionGenError::InvalidParameter(format!(
            "logits len {} does not match seq_len ({}) * vocab_size ({}) = {}",
            logits.len(),
            seq_len,
            vocab_size,
            expected,
        )));
    }
    if k == 0 {
        return greedy_decode(logits, vocab_size, seq_len);
    }
    let effective_k = k.min(vocab_size);

    let mut out: Vec<u32> = Vec::with_capacity(seq_len);
    for step in 0..seq_len {
        let start = step * vocab_size;
        let end = start + vocab_size;
        let row = &logits[start..end];
        let top = top_k(row, effective_k).map_err(|e| {
            CaptionGenError::InvalidParameter(format!(
                "top_k_sample: top_k failed on step {step}: {e:?}"
            ))
        })?;
        // If by some pathology `top_k` returned nothing, fall back to
        // argmax semantics for this step.
        if top.is_empty() {
            let idx = argmax(row).map_err(|e| {
                CaptionGenError::InvalidParameter(format!(
                    "top_k_sample: fallback argmax failed on step {step}: {e:?}"
                ))
            })?;
            out.push(u32_from_usize(idx)?);
            continue;
        }
        // Softmax over the top-k logits only — produces a proper pmf.
        let top_logits: Vec<f32> = top.iter().map(|(_, v)| *v).collect();
        let pmf = softmax(&top_logits);

        // Draw a uniform sample deterministically from seed + step.
        let r = xorshift_uniform_f32(seed.wrapping_add(step as u64));
        // Inverse-CDF sampling.
        let mut acc: f32 = 0.0;
        let mut chosen = top.len() - 1; // safe: non-empty by construction
        for (i, p) in pmf.iter().enumerate() {
            acc += *p;
            if r < acc {
                chosen = i;
                break;
            }
        }
        let picked_vocab_idx = top[chosen].0;
        out.push(u32_from_usize(picked_vocab_idx)?);
    }
    Ok(out)
}

/// Narrow `usize` → `u32` with an explicit error if the index exceeds
/// `u32::MAX`.  Caption vocabularies in real deployments are well under
/// this limit, but the check keeps the API honest against malformed graphs.
fn u32_from_usize(v: usize) -> CaptionGenResult<u32> {
    u32::try_from(v)
        .map_err(|_| CaptionGenError::InvalidParameter(format!("token index {v} exceeds u32::MAX")))
}

/// xorshift64\* step — deterministic, no deps, ~6 lines of state.
///
/// The constant `0x2545F4914F6CEDD1` is the standard xorshift64\*
/// multiplier (Vigna, 2014).  A zero input is replaced with a fixed
/// non-zero salt so that `seed == 0` does not collapse to all zeros.
fn xorshift64_star(seed: u64) -> u64 {
    let mut s = if seed == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        seed
    };
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    s.wrapping_mul(0x2545_F491_4F6C_EDD1)
}

/// Draw a uniform `f32` in `[0, 1)` from a xorshift64\* stream.
fn xorshift_uniform_f32(seed: u64) -> f32 {
    // Top 24 bits → f32 mantissa in [0, 1).
    let bits = (xorshift64_star(seed) >> 40) as u32;
    (bits as f32) / ((1u32 << 24) as f32)
}

// ── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_output_len_and_is_empty_match_buffer() {
        let o = EncoderOutput {
            logits: vec![1.0, 2.0, 3.0],
            shape: vec![1, 3],
        };
        assert_eq!(o.len(), 3);
        assert!(!o.is_empty());

        let empty = EncoderOutput {
            logits: Vec::new(),
            shape: vec![0],
        };
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn greedy_decode_picks_argmax_per_row() {
        // 2 steps, 3-token vocab.
        let logits = vec![
            0.1, 0.8, 0.1, // step 0 → token 1
            0.4, 0.2, 0.4, // step 1 → tie on 0 and 2; argmax picks lowest → 0
        ];
        let out = greedy_decode(&logits, 3, 2).expect("ok");
        assert_eq!(out, vec![1_u32, 0_u32]);
    }

    #[test]
    fn greedy_decode_rejects_zero_vocab_or_seq_len() {
        let e1 = greedy_decode(&[1.0, 2.0], 0, 1).expect_err("vocab=0");
        assert!(matches!(e1, CaptionGenError::InvalidParameter(_)));
        let e2 = greedy_decode(&[1.0, 2.0], 2, 0).expect_err("seq_len=0");
        assert!(matches!(e2, CaptionGenError::InvalidParameter(_)));
    }

    #[test]
    fn greedy_decode_rejects_mismatched_buffer_length() {
        // 2 steps × 3 tokens = 6 expected, but only 5 supplied.
        let e = greedy_decode(&[0.0, 0.0, 0.0, 0.0, 0.0], 3, 2).expect_err("mismatched len");
        assert!(matches!(e, CaptionGenError::InvalidParameter(_)));
    }

    #[test]
    fn top_k_sample_with_k_zero_matches_greedy_decode() {
        let logits = vec![0.1, 0.8, 0.1, 0.4, 0.2, 0.4];
        let greedy = greedy_decode(&logits, 3, 2).expect("ok");
        let sampled = top_k_sample(&logits, 3, 2, 0, 42).expect("ok");
        assert_eq!(greedy, sampled);
    }

    #[test]
    fn top_k_sample_is_deterministic_for_identical_seed() {
        let logits = vec![0.5, 0.2, 0.1, 0.2, 0.1, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0.4];
        let a = top_k_sample(&logits, 4, 3, 2, 12345).expect("ok");
        let b = top_k_sample(&logits, 4, 3, 2, 12345).expect("ok");
        assert_eq!(a, b);
        assert_eq!(a.len(), 3);
    }

    #[test]
    fn top_k_sample_only_emits_tokens_from_top_k_set() {
        // Vocab = 5; only the top-2 per step should ever be sampled.
        // Step 0: top-2 indices = [1, 3] (values 0.9, 0.7).
        // Step 1: top-2 indices = [0, 2] (values 0.6, 0.5).
        let logits = vec![
            0.1, 0.9, 0.3, 0.7, 0.2, // step 0
            0.6, 0.4, 0.5, 0.2, 0.1, // step 1
        ];
        // Try many seeds; every emitted token must stay in the allowed set.
        for seed in 0..64_u64 {
            let out = top_k_sample(&logits, 5, 2, 2, seed).expect("ok");
            assert!(out[0] == 1 || out[0] == 3, "step 0 outside top-2: {out:?}");
            assert!(out[1] == 0 || out[1] == 2, "step 1 outside top-2: {out:?}");
        }
    }

    #[test]
    fn top_k_sample_rejects_invalid_sizes() {
        let e1 = top_k_sample(&[1.0, 2.0], 0, 1, 1, 0).expect_err("vocab=0");
        assert!(matches!(e1, CaptionGenError::InvalidParameter(_)));
        let e2 = top_k_sample(&[1.0, 2.0], 2, 0, 1, 0).expect_err("seq_len=0");
        assert!(matches!(e2, CaptionGenError::InvalidParameter(_)));
        let e3 = top_k_sample(&[1.0, 2.0, 3.0], 2, 2, 1, 0).expect_err("len mismatch");
        assert!(matches!(e3, CaptionGenError::InvalidParameter(_)));
    }

    #[test]
    fn top_k_sample_clamps_k_greater_than_vocab() {
        // k = 100 on a 3-token vocab must be clamped, not panic.
        let logits = vec![0.1, 0.8, 0.1, 0.2, 0.3, 0.5];
        let out = top_k_sample(&logits, 3, 2, 100, 7).expect("ok");
        assert_eq!(out.len(), 2);
        for &t in &out {
            assert!(t < 3, "token {t} outside vocab size 3");
        }
    }

    #[test]
    fn xorshift64_star_is_nonzero_for_zero_seed() {
        // Regression guard: the zero-seed salt must not produce zero output.
        assert_ne!(xorshift64_star(0), 0);
    }

    #[test]
    fn xorshift_uniform_f32_stays_in_unit_interval() {
        for seed in 0..4096_u64 {
            let r = xorshift_uniform_f32(seed);
            assert!((0.0..1.0).contains(&r), "seed {seed} produced {r}");
        }
    }

    #[test]
    fn ml_error_roundtrips_into_caption_gen_error() {
        // Exercises `CaptionGenError: From<MlError>` so the `#[from]`
        // derive stays connected even when no other test path touches it.
        fn forward() -> CaptionGenResult<()> {
            Err(MlError::FeatureDisabled("onnx"))?;
            Ok(())
        }
        let err = forward().expect_err("must propagate");
        assert!(matches!(
            err,
            CaptionGenError::Ml(MlError::FeatureDisabled("onnx"))
        ));
    }

    #[test]
    fn from_path_missing_file_returns_ml_error() {
        let path = std::env::temp_dir().join("oximedia-caption-gen-nonexistent-encoder.onnx");
        if path.exists() {
            let _ = std::fs::remove_file(&path);
        }
        let err = CaptionEncoder::from_path(&path, DeviceType::Cpu)
            .expect_err("loading a missing model must fail");
        assert!(
            matches!(err, CaptionGenError::Ml(_)),
            "expected CaptionGenError::Ml, got {err:?}",
        );
    }
}
