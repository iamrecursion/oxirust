//! ML-assisted music tagging / genre classification via ONNX.
//!
//! This module provides [`MusicTagger`], a thin wrapper around
//! [`oximedia_ml::OnnxModel`] for audio classification workflows such as
//! MusicNN / PANNs / OpenL3-style genre, mood, or auto-tagging models.
//! The wrapper is purely **additive** — default builds remain free of
//! ONNX symbols, and none of the existing heuristic MIR pipelines
//! (tempo, beat, key, chord, melody, structure, Camelot, mood, genre
//! classifier) are affected.
//!
//! # Design
//!
//! Music tagging models vary widely in their input contract:
//!
//! * **Mel-spectrogram models** (MusicNN, PANNs): input shape `[N, 1, T, F]`
//!   or `[N, T, F]`, optionally log-compressed.
//! * **Waveform models** (sample-CNNs, Wav2Vec-style): input shape
//!   `[N, 1, T]` with raw normalised samples at a fixed sample rate.
//! * **Audio-embedding models** (OpenL3, VGGish, YAMNet): same as
//!   mel-spectrogram inputs but emit per-frame embeddings that downstream
//!   classifiers consume.
//!
//! To accommodate all of these without committing to a single
//! preprocessing convention, [`MusicTagger`] accepts a preprocessed
//! `Vec<f32>` plus its shape.  The caller is responsible for the feature
//! extraction pipeline — this module only wraps inference and
//! post-processing.
//!
//! # Activation modes
//!
//! Music taggers come in two flavours:
//!
//! * **Single-label** (e.g. top-1 genre): logits + softmax → probabilities.
//! * **Multi-label** (e.g. tags like "female-vocal", "guitar", "live"):
//!   per-class sigmoid → independent probabilities.
//!
//! [`TagActivation`] selects between the two; the default is
//! [`TagActivation::Softmax`] because single-label genre classification is
//! the most common opt-in case.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_mir::ml::{MusicTagger, TagActivation};
//! use oximedia_ml::DeviceType;
//!
//! # fn run() -> oximedia_mir::MirResult<()> {
//! let labels = vec![
//!     "blues".to_string(),
//!     "classical".to_string(),
//!     "country".to_string(),
//!     "disco".to_string(),
//!     "hiphop".to_string(),
//!     "jazz".to_string(),
//!     "metal".to_string(),
//!     "pop".to_string(),
//!     "reggae".to_string(),
//!     "rock".to_string(),
//! ];
//! let tagger = MusicTagger::from_path("musicnn.onnx", DeviceType::auto())?
//!     .with_labels(labels)
//!     .with_top_k(3)
//!     .with_activation(TagActivation::Softmax);
//!
//! // Pretend we already computed a mel-spectrogram of shape [1, 1, 96, 128].
//! let mel = vec![0.0_f32; 1 * 1 * 96 * 128];
//! let shape = vec![1, 1, 96, 128];
//! let tags = tagger.classify(&mel, &shape)?;
//! for tag in tags.top() {
//!     println!("{}: {:.3}", tag.label, tag.score);
//! }
//! # Ok(()) }
//! ```
//!
//! # Error mapping
//!
//! Every fallible operation returns [`crate::MirResult`].  The
//! [`oximedia_ml::MlError`] type is folded into [`crate::MirError`] via
//! `thiserror`'s `#[from]` conversion declared on the `Ml` variant.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use oximedia_ml::{DeviceType, ModelCache, OnnxModel};

use crate::error::{MirError, MirResult};

/// Sentinel input-tensor name used when the model advertises no inputs.
///
/// In practice every well-formed ONNX music-tagging model exposes at
/// least one input, so this string only surfaces if someone loads an
/// empty / malformed graph — in which case [`MusicTagger::classify`]
/// will fail loudly with an `Ml` error before the sentinel reaches the
/// backend.
const FALLBACK_INPUT_NAME: &str = "input";

/// Default number of top tags returned by [`MusicTagger::classify`] when
/// no explicit value has been configured via [`MusicTagger::with_top_k`].
pub const DEFAULT_TOP_K: usize = 5;

/// Activation function applied to the raw model logits before ranking.
///
/// Defaults to [`Self::Softmax`] because single-label genre
/// classification is the most common opt-in case.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TagActivation {
    /// Softmax over the class axis (single-label; probabilities sum to 1).
    #[default]
    Softmax,
    /// Sigmoid per class (multi-label; each score is independent in `[0, 1]`).
    Sigmoid,
    /// No activation — treat raw logits as the scores to rank.
    ///
    /// Useful for models that emit pre-normalised tag scores directly,
    /// for instance custom heads that already embed their own softmax.
    None,
}

/// A single tag activation with its human-readable label.
#[derive(Clone, Debug, PartialEq)]
pub struct TagActivationScore {
    /// Human-readable tag label.
    pub label: String,
    /// Activation score in `[0, 1]` for [`TagActivation::Softmax`] /
    /// [`TagActivation::Sigmoid`], or raw logit for
    /// [`TagActivation::None`].
    pub score: f32,
    /// Zero-based index into the original logit vector.  Useful for
    /// callers that want to correlate results back to the label space
    /// without re-computing hashes.
    pub index: usize,
}

/// Ranked music-tagging output produced by [`MusicTagger::classify`].
///
/// Tags are stored in descending score order and already truncated to
/// the configured `top_k`.  Retrieve them via [`Self::top`] or
/// [`Self::into_inner`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MusicTags {
    /// Ranked list of `(label, score)` pairs, descending by score.
    tags: Vec<TagActivationScore>,
    /// Activation function that produced these scores.
    activation: TagActivation,
}

impl MusicTags {
    /// Wrap a pre-ranked list of tags.
    ///
    /// The vector is **not** re-sorted — callers that build instances
    /// manually are responsible for maintaining the descending-score
    /// invariant.
    #[must_use]
    pub fn new(tags: Vec<TagActivationScore>, activation: TagActivation) -> Self {
        Self { tags, activation }
    }

    /// Read-only slice of the ranked tags.
    #[must_use]
    pub fn top(&self) -> &[TagActivationScore] {
        &self.tags
    }

    /// Number of tags returned (bounded by the configured `top_k`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.tags.len()
    }

    /// Return whether the result is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// Activation that produced these scores.
    #[must_use]
    pub fn activation(&self) -> TagActivation {
        self.activation
    }

    /// Consume the result and return the owned ranked vector.
    #[must_use]
    pub fn into_inner(self) -> Vec<TagActivationScore> {
        self.tags
    }

    /// Top-1 tag (highest score), or `None` if no tags were produced.
    #[must_use]
    pub fn best(&self) -> Option<&TagActivationScore> {
        self.tags.first()
    }
}

/// Opt-in ML music tagger / genre classifier backed by an ONNX model.
///
/// Wraps an [`OnnxModel`] directly rather than a typed pipeline so any
/// ONNX music-tagging graph (MusicNN, PANNs, OpenL3, sample-CNN, custom
/// heads, …) can be used.  Callers are responsible for preprocessing
/// audio into the shape the model expects.
pub struct MusicTagger {
    /// Shared ONNX session.
    model: Arc<OnnxModel>,
    /// Path the model was loaded from (retained for diagnostics).
    model_path: PathBuf,
    /// Name of the input tensor passed to the model.
    input_name: String,
    /// Name of the output tensor read back as the logits / scores.
    output_name: String,
    /// Label space (same length as the class axis of the model's output).
    ///
    /// When empty, [`Self::classify`] generates synthetic labels of the
    /// form `class_{i}` so downstream callers never have to special-case
    /// missing label data.
    labels: Vec<String>,
    /// Maximum number of top tags reported by [`Self::classify`].
    top_k: usize,
    /// Activation applied to raw logits before ranking.
    activation: TagActivation,
}

impl MusicTagger {
    /// Load a music-tagging ONNX model from disk.
    ///
    /// The default input/output tensor names are resolved from the
    /// model's [`oximedia_ml::ModelInfo`] (first input / first output).
    /// Override them via [`Self::with_input_name`] /
    /// [`Self::with_output_name`] when a model has auxiliary heads.
    ///
    /// # Errors
    ///
    /// * Returns [`MirError::Ml`] wrapping
    ///   [`oximedia_ml::MlError::ModelLoad`] if the ONNX model cannot be
    ///   opened.
    /// * Returns [`MirError::Ml`] wrapping
    ///   [`oximedia_ml::MlError::DeviceUnavailable`] if the requested
    ///   device is not compiled in or is unavailable at runtime.
    pub fn from_path(model_path: impl AsRef<Path>, device: DeviceType) -> MirResult<Self> {
        let path = model_path.as_ref().to_path_buf();
        let model = Arc::new(OnnxModel::load(&path, device)?);
        Ok(Self::build(model, path))
    }

    /// Build a tagger from a shared [`OnnxModel`] (typically resolved
    /// via [`ModelCache`]).
    ///
    /// Useful when multiple MIR tasks share the same music-tagging
    /// weights (for example, genre + mood heads sharing a common
    /// embedding backbone).
    #[must_use]
    pub fn from_shared_model(model: Arc<OnnxModel>, model_path: PathBuf) -> Self {
        Self::build(model, model_path)
    }

    /// Resolve a tagger against a [`ModelCache`], sharing the
    /// `OnnxModel` with any other caller that loaded the same path.
    ///
    /// # Errors
    ///
    /// Propagates any [`oximedia_ml::MlError`] raised by the cache
    /// loader.
    pub fn from_cache(
        cache: &ModelCache,
        model_path: impl AsRef<Path>,
        device: DeviceType,
    ) -> MirResult<Self> {
        let path = model_path.as_ref().to_path_buf();
        let model = cache.get_or_load(&path, device)?;
        Ok(Self::from_shared_model(model, path))
    }

    fn build(model: Arc<OnnxModel>, model_path: PathBuf) -> Self {
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
            model_path,
            input_name,
            output_name,
            labels: Vec::new(),
            top_k: DEFAULT_TOP_K,
            activation: TagActivation::default(),
        }
    }

    /// Builder-style setter overriding the input tensor name.
    #[must_use]
    pub fn with_input_name(mut self, name: impl Into<String>) -> Self {
        self.input_name = name.into();
        self
    }

    /// Builder-style setter overriding the output tensor name used as
    /// the logit source.  Necessary for multi-head models where the
    /// first output is not the classification head.
    #[must_use]
    pub fn with_output_name(mut self, name: impl Into<String>) -> Self {
        self.output_name = name.into();
        self
    }

    /// Builder-style setter installing the tag label space.
    ///
    /// When the vector length matches the model's class axis, each
    /// ranked score is paired with the corresponding label.  When the
    /// vector is empty, [`Self::classify`] falls back to synthetic
    /// `class_{i}` labels derived from the logit index — this keeps the
    /// contract usable with models where labels aren't known at
    /// construction time.
    #[must_use]
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = labels;
        self
    }

    /// Builder-style setter configuring the top-`k` cap.  Clamped
    /// silently to `1` when callers pass `0` so [`Self::classify`]
    /// always produces at least one tag on a non-empty model output.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k.max(1);
        self
    }

    /// Builder-style setter selecting the activation function applied
    /// to the raw logits before ranking.
    #[must_use]
    pub fn with_activation(mut self, activation: TagActivation) -> Self {
        self.activation = activation;
        self
    }

    /// Return the currently configured input tensor name.
    #[must_use]
    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    /// Return the currently configured output tensor name.
    #[must_use]
    pub fn output_name(&self) -> &str {
        &self.output_name
    }

    /// Path of the ONNX model that backs this tagger.
    #[must_use]
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Read-only slice of the label space.  Empty when no labels have
    /// been configured via [`Self::with_labels`].
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Currently configured top-`k` cap.
    #[must_use]
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Currently configured activation.
    #[must_use]
    pub fn activation(&self) -> TagActivation {
        self.activation
    }

    /// Shared handle to the underlying [`OnnxModel`].
    #[must_use]
    pub fn shared_model(&self) -> Arc<OnnxModel> {
        self.model.clone()
    }

    /// Run the music-tagging model on a preprocessed input tensor and
    /// return the top-`k` ranked tags.
    ///
    /// `tensor.len()` must equal `shape.iter().product::<usize>()`.  The
    /// logits are taken from the output tensor named
    /// [`Self::output_name`] (defaults to the first model output).
    ///
    /// The tensor data is cloned internally so callers can re-use the
    /// input buffer across multiple invocations.
    ///
    /// # Errors
    ///
    /// * [`MirError::Ml`] wrapping any error raised by the underlying
    ///   ONNX session.
    /// * [`MirError::Ml`] if the configured output tensor name is
    ///   missing from the model's output map.
    /// * [`MirError::AnalysisFailed`] if the model returns an empty
    ///   output buffer (which cannot be ranked).
    pub fn classify(&self, tensor: &[f32], shape: &[usize]) -> MirResult<MusicTags> {
        let mut outputs =
            self.model
                .run_single(&self.input_name, tensor.to_vec(), shape.to_vec())?;
        let logits = outputs.remove(&self.output_name).ok_or_else(|| {
            MirError::Ml(oximedia_ml::MlError::pipeline(
                "music-tag",
                format!(
                    "output '{}' missing from model '{}'",
                    self.output_name,
                    self.model_path.display(),
                ),
            ))
        })?;
        if logits.is_empty() {
            return Err(MirError::AnalysisFailed(format!(
                "music-tagging model '{}' returned an empty output tensor",
                self.model_path.display(),
            )));
        }
        let ranked = activate_and_rank(&logits, &self.labels, self.top_k, self.activation)?;
        Ok(MusicTags::new(ranked, self.activation))
    }
}

impl std::fmt::Debug for MusicTagger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MusicTagger")
            .field("input_name", &self.input_name)
            .field("output_name", &self.output_name)
            .field("model_path", &self.model_path)
            .field("labels_len", &self.labels.len())
            .field("top_k", &self.top_k)
            .field("activation", &self.activation)
            .finish()
    }
}

/// Pure activation-and-rank helper used by [`MusicTagger::classify`].
///
/// Separated out so unit tests can exercise the scoring / ranking logic
/// without needing to load an actual ONNX model.  When `labels` is
/// empty, synthetic `class_{i}` labels are generated from the logit
/// index.  When `labels.len() != logits.len()`, the shorter of the two
/// bounds is used and the remaining classes get synthetic labels — this
/// keeps the contract resilient to off-by-one mismatches between the
/// model and an external label file.
///
/// # Errors
///
/// Returns [`MirError::Ml`] wrapping any error from
/// [`oximedia_ml::postprocess::top_k`] (notably an empty `logits`
/// slice).
pub fn activate_and_rank(
    logits: &[f32],
    labels: &[String],
    top_k: usize,
    activation: TagActivation,
) -> MirResult<Vec<TagActivationScore>> {
    let scores = apply_activation(logits, activation);
    let effective_k = top_k.min(scores.len()).max(1);
    let ranked = oximedia_ml::postprocess::top_k(&scores, effective_k)?;
    let mut out = Vec::with_capacity(ranked.len());
    for (idx, score) in ranked {
        let label = labels
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("class_{idx}"));
        out.push(TagActivationScore {
            label,
            score,
            index: idx,
        });
    }
    Ok(out)
}

/// Apply the selected activation function to `logits`.
///
/// [`TagActivation::None`] returns the logits unmodified so callers
/// that already feed pre-normalised scores can bypass the extra pass.
#[must_use]
pub fn apply_activation(logits: &[f32], activation: TagActivation) -> Vec<f32> {
    match activation {
        TagActivation::Softmax => oximedia_ml::postprocess::softmax(logits),
        TagActivation::Sigmoid => oximedia_ml::postprocess::sigmoid_slice(logits),
        TagActivation::None => logits.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_ml::MlError;

    #[test]
    fn softmax_activation_sums_to_one() {
        let probs = apply_activation(&[1.0, 2.0, 3.0], TagActivation::Softmax);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax output must sum to 1, got {sum}",
        );
    }

    #[test]
    fn sigmoid_activation_stays_in_zero_one() {
        let probs = apply_activation(&[-10.0, 0.0, 10.0], TagActivation::Sigmoid);
        for p in &probs {
            assert!((0.0..=1.0).contains(p), "sigmoid out of range: {p}");
        }
        // Boundary checks — large negative → 0, zero → 0.5, large positive → 1.
        assert!(probs[0] < 0.001);
        assert!((probs[1] - 0.5).abs() < 1e-6);
        assert!(probs[2] > 0.999);
    }

    #[test]
    fn none_activation_is_identity() {
        let raw = vec![1.5_f32, -2.5, 0.0];
        let out = apply_activation(&raw, TagActivation::None);
        assert_eq!(out, raw);
    }

    #[test]
    fn activate_and_rank_sorts_descending() {
        let labels = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let logits = vec![0.1_f32, 5.0, 0.3, 0.2];
        let ranked = activate_and_rank(&logits, &labels, 4, TagActivation::Softmax).expect("ok");
        assert_eq!(ranked.len(), 4);
        // With softmax, class 1 dominates.
        assert_eq!(ranked[0].label, "b");
        assert_eq!(ranked[0].index, 1);
        // Remaining order must be non-increasing in score.
        for w in ranked.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "ranking violates descending invariant: {w:?}",
            );
        }
    }

    #[test]
    fn activate_and_rank_top_k_truncates() {
        let labels = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ];
        let logits = vec![0.1_f32, 0.5, 0.3, 0.7, 0.2];
        let ranked = activate_and_rank(&logits, &labels, 2, TagActivation::Softmax).expect("ok");
        assert_eq!(ranked.len(), 2);
        // Softmax preserves the argmax ordering.
        assert_eq!(ranked[0].index, 3);
        assert_eq!(ranked[1].index, 1);
    }

    #[test]
    fn activate_and_rank_missing_labels_generates_synthetic() {
        let labels: Vec<String> = Vec::new();
        let logits = vec![0.1_f32, 0.9];
        let ranked = activate_and_rank(&logits, &labels, 2, TagActivation::Softmax).expect("ok");
        assert_eq!(ranked.len(), 2);
        // Descending by score — index 1 (logit 0.9) wins.
        assert_eq!(ranked[0].index, 1);
        assert_eq!(ranked[0].label, "class_1");
        assert_eq!(ranked[1].index, 0);
        assert_eq!(ranked[1].label, "class_0");
    }

    #[test]
    fn activate_and_rank_shorter_labels_falls_back_to_synthetic() {
        // Three classes but only two labels — out-of-range index gets
        // a synthetic "class_2" label.
        let labels = vec!["a".to_string(), "b".to_string()];
        let logits = vec![0.1_f32, 0.2, 0.9];
        let ranked = activate_and_rank(&logits, &labels, 3, TagActivation::Softmax).expect("ok");
        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].index, 2);
        assert_eq!(ranked[0].label, "class_2");
    }

    #[test]
    fn activate_and_rank_empty_logits_errors() {
        let err = activate_and_rank(&[], &[], 3, TagActivation::Softmax).expect_err("must fail");
        assert!(
            matches!(err, MirError::Ml(MlError::Postprocess(_))),
            "expected MlError::Postprocess, got {err:?}",
        );
    }

    #[test]
    fn activate_and_rank_top_k_zero_is_clamped_to_one() {
        let labels = vec!["a".to_string(), "b".to_string()];
        let logits = vec![0.2_f32, 0.8];
        let ranked = activate_and_rank(&logits, &labels, 0, TagActivation::Softmax).expect("ok");
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].index, 1);
    }

    #[test]
    fn music_tags_getters_work() {
        let tags = MusicTags::new(
            vec![
                TagActivationScore {
                    label: "rock".into(),
                    score: 0.6,
                    index: 0,
                },
                TagActivationScore {
                    label: "jazz".into(),
                    score: 0.3,
                    index: 1,
                },
            ],
            TagActivation::Sigmoid,
        );
        assert_eq!(tags.len(), 2);
        assert!(!tags.is_empty());
        assert_eq!(tags.activation(), TagActivation::Sigmoid);
        assert_eq!(tags.best().map(|t| t.label.as_str()), Some("rock"));
        assert_eq!(tags.top().len(), 2);
        let owned = tags.into_inner();
        assert_eq!(owned.len(), 2);
    }

    #[test]
    fn music_tags_empty_reports_empty() {
        let tags = MusicTags::default();
        assert!(tags.is_empty());
        assert_eq!(tags.len(), 0);
        assert!(tags.best().is_none());
        assert_eq!(tags.top().len(), 0);
    }

    #[test]
    fn tag_activation_default_is_softmax() {
        assert_eq!(TagActivation::default(), TagActivation::Softmax);
    }

    #[test]
    fn mir_error_from_ml_error_is_wired() {
        // Exercises `MirError: From<MlError>` so the `#[from]` derive
        // stays connected even when no other test path touches it.
        let ml_err = MlError::FeatureDisabled("onnx");
        let mir_err: MirError = ml_err.into();
        assert!(
            matches!(mir_err, MirError::Ml(MlError::FeatureDisabled("onnx"))),
            "expected MirError::Ml(FeatureDisabled), got {mir_err:?}",
        );
    }

    #[test]
    fn sigmoid_multi_label_preserves_independence() {
        // Two logits that would both be high-probability with softmax
        // rivalry but are independently confident under sigmoid.
        let logits = vec![4.0_f32, 4.0];
        let probs = apply_activation(&logits, TagActivation::Sigmoid);
        // Both must exceed 0.9 — independence property.
        assert!(probs[0] > 0.9);
        assert!(probs[1] > 0.9);
    }

    #[test]
    fn from_path_missing_file_returns_ml_error() {
        let path = std::path::PathBuf::from("/does-not-exist-oximedia-mir-music-tagger.onnx");
        let err = MusicTagger::from_path(&path, DeviceType::Cpu)
            .expect_err("loading a nonexistent model must fail");
        assert!(
            matches!(err, MirError::Ml(_)),
            "expected MirError::Ml, got {err:?}",
        );
    }
}
