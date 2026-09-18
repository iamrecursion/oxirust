//! ML-assisted embedding-based content similarity.
//!
//! This module provides [`EmbeddingExtractor`], a thin, generic wrapper
//! around [`oximedia_ml::OnnxModel`] that produces L2-normalised
//! [`ContentEmbedding`] vectors suitable for cosine-similarity-driven
//! recommendations.  The wrapper is purely **additive**: it never
//! replaces or mutates the crate's existing content/collaborative
//! recommenders, and the whole module is gated behind the `onnx` Cargo
//! feature so default builds remain free of ONNX symbols.
//!
//! # Design
//!
//! Unlike [`oximedia_ml::pipelines::SceneClassifier`] or
//! [`oximedia_ml::pipelines::ShotBoundaryDetector`] — which are *typed*
//! pipelines tied to a specific model contract — content similarity in
//! recommendation systems is intentionally generic: any ONNX model that
//! maps a preprocessed input tensor to a 1-D embedding is fair game
//! (CLIP text/image towers, SimCLR, MobileFaceNet, a custom fine-tuned
//! bi-encoder, …).  [`EmbeddingExtractor`] therefore wraps the raw
//! [`OnnxModel`] and exposes a narrow contract:
//!
//! * Caller supplies a preprocessed `Vec<f32>` plus its `Vec<usize>` shape.
//! * The extractor returns a [`ContentEmbedding`] whose raw buffer is
//!   **always** L2-normalised, so cosine similarity collapses to a plain
//!   dot product and callers never have to think about it again.
//!
//! Preprocessing (resize, colour-space conversion, mean/std
//! normalisation) is the **caller's** responsibility.  Recommendation
//! systems vary too widely for a single preprocessing convention to make
//! sense — use [`oximedia_ml::ImagePreprocessor`] or your own pipeline,
//! then hand the result to [`EmbeddingExtractor::extract`].
//!
//! # Error mapping
//!
//! Every fallible operation returns [`crate::RecommendResult`].  The
//! [`oximedia_ml::MlError`] type is folded into [`crate::RecommendError`]
//! via `thiserror`'s `#[from]` conversion declared on the `Ml` variant.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_recommend::ml::{ContentEmbedding, EmbeddingExtractor};
//! use oximedia_ml::DeviceType;
//!
//! # fn run() -> oximedia_recommend::RecommendResult<()> {
//! let extractor = EmbeddingExtractor::from_path(
//!     "image_encoder.onnx",
//!     DeviceType::auto(),
//! )?;
//!
//! // Pretend we already preprocessed a 224x224 RGB image into NCHW f32.
//! let data = vec![0.0_f32; 1 * 3 * 224 * 224];
//! let shape = vec![1, 3, 224, 224];
//! let query: ContentEmbedding = extractor.extract(data, shape)?;
//!
//! let candidates: Vec<ContentEmbedding> = Vec::new(); // load from your store
//! let ranked = oximedia_recommend::ml::rank_by_similarity(&query, &candidates, 10);
//! for (idx, score) in ranked {
//!     println!("candidate {idx}: similarity={score:.3}");
//! }
//! # Ok(()) }
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use oximedia_ml::{DeviceType, ModelCache, OnnxModel};

use crate::error::{RecommendError, RecommendResult};

/// Sentinel input-tensor name used when the model advertises no inputs.
///
/// In practice every well-formed ONNX embedding model exposes at least
/// one input, so this string only surfaces if someone loads an empty /
/// malformed graph — in which case `extract` will fail loudly with an
/// `Ml` error before the sentinel reaches disk.
const FALLBACK_INPUT_NAME: &str = "input";

/// An L2-normalised content embedding.
///
/// Every constructor normalises the underlying buffer in place via
/// [`oximedia_ml::l2_normalize`], so [`Self::cosine_similarity`] reduces
/// to a plain dot product.  The original (pre-normalised) buffer is not
/// retained — callers who need the raw model output should work with the
/// `Vec<f32>` returned from [`OnnxModel::run_single`] directly.
#[derive(Clone, Debug, PartialEq)]
pub struct ContentEmbedding {
    /// Unit-norm embedding vector.
    vector: Vec<f32>,
}

impl ContentEmbedding {
    /// Wrap an existing embedding buffer, normalising it in place.
    ///
    /// Returns [`RecommendError::InvalidSimilarity`] if the input is
    /// empty or if every value is zero / non-finite (which would leave
    /// the vector un-normalisable).
    pub fn new(mut vector: Vec<f32>) -> RecommendResult<Self> {
        if vector.is_empty() {
            return Err(RecommendError::InvalidSimilarity(0.0));
        }
        let norm_sq: f32 = vector.iter().map(|x| x * x).sum();
        if !norm_sq.is_finite() || norm_sq <= 0.0 {
            return Err(RecommendError::InvalidSimilarity(norm_sq));
        }
        oximedia_ml::l2_normalize(&mut vector);
        Ok(Self { vector })
    }

    /// Construct a [`ContentEmbedding`] without re-normalising.
    ///
    /// Intended for trusted callers that have *already* invoked
    /// [`oximedia_ml::l2_normalize`] on the buffer (for example, the
    /// `face-embedder-sim` pipeline, which normalises inside the
    /// pipeline itself).  Prefer [`Self::new`] when in doubt.
    #[must_use]
    pub fn from_normalized(vector: Vec<f32>) -> Self {
        Self { vector }
    }

    /// Dimensionality of the embedding.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.vector.len()
    }

    /// Return the underlying unit-norm buffer as a read-only slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.vector
    }

    /// Consume the embedding and return the owned unit-norm buffer.
    #[must_use]
    pub fn into_inner(self) -> Vec<f32> {
        self.vector
    }

    /// Cosine similarity against another embedding in `[-1, 1]`.
    ///
    /// Returns `0.0` for dimension mismatches — callers that treat that
    /// as a bug should check dimensions first via [`Self::dim`].
    #[must_use]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        oximedia_ml::cosine_similarity(&self.vector, &other.vector)
    }

    /// Euclidean (L2) distance against another embedding.
    ///
    /// Because both embeddings are unit-norm, this is equivalent to
    /// `sqrt(2 - 2 * cosine_similarity)`.  Returns [`f32::INFINITY`] on
    /// dimension mismatch so the caller notices immediately without
    /// silently collapsing the distance to zero.
    #[must_use]
    pub fn euclidean_distance(&self, other: &Self) -> f32 {
        if self.vector.len() != other.vector.len() {
            return f32::INFINITY;
        }
        let sum_sq: f32 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(&a, &b)| {
                let d = a - b;
                d * d
            })
            .sum();
        sum_sq.max(0.0).sqrt()
    }
}

/// Opt-in ML embedding extractor that produces L2-normalised
/// [`ContentEmbedding`] vectors from a preprocessed `Vec<f32>` input
/// tensor.
///
/// Wraps an [`OnnxModel`] directly rather than a typed pipeline so any
/// ONNX embedding graph (CLIP, SimCLR, MobileFaceNet, custom bi-encoder,
/// …) can be used.  Callers are responsible for preprocessing inputs
/// into the shape the model expects.
pub struct EmbeddingExtractor {
    model: Arc<OnnxModel>,
    model_path: PathBuf,
    /// Name of the input tensor passed to the model.
    input_name: String,
    /// Name of the output tensor read back as the embedding.
    output_name: String,
}

impl EmbeddingExtractor {
    /// Load an embedding ONNX model from disk.
    ///
    /// The default input/output tensor names are resolved from the
    /// model's [`oximedia_ml::ModelInfo`] (first input / first output).
    /// Override them via [`Self::with_input_name`] /
    /// [`Self::with_output_name`] when a model has auxiliary heads.
    ///
    /// # Errors
    ///
    /// * Returns [`RecommendError::Ml`] wrapping
    ///   [`oximedia_ml::MlError::ModelLoad`] if the ONNX model cannot be
    ///   opened.
    /// * Returns [`RecommendError::Ml`] wrapping
    ///   [`oximedia_ml::MlError::DeviceUnavailable`] if the requested
    ///   device is not compiled in or is unavailable at runtime.
    pub fn from_path(model_path: impl AsRef<Path>, device: DeviceType) -> RecommendResult<Self> {
        let path = model_path.as_ref().to_path_buf();
        let model = Arc::new(OnnxModel::load(&path, device)?);
        Ok(Self::build(model, path))
    }

    /// Build an extractor from a shared [`OnnxModel`] (typically resolved
    /// via [`ModelCache`]).
    ///
    /// Useful when multiple recommenders share the same embedding
    /// weights file (for example, image and video tracks both using a
    /// CLIP image tower).
    #[must_use]
    pub fn from_shared_model(model: Arc<OnnxModel>, model_path: PathBuf) -> Self {
        Self::build(model, model_path)
    }

    /// Resolve an extractor against a [`ModelCache`], sharing the
    /// `OnnxModel` with any other caller that loaded the same path.
    ///
    /// # Errors
    ///
    /// Propagates any [`oximedia_ml::MlError`] raised by the cache loader.
    pub fn from_cache(
        cache: &ModelCache,
        model_path: impl AsRef<Path>,
        device: DeviceType,
    ) -> RecommendResult<Self> {
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
        }
    }

    /// Builder-style setter overriding the input tensor name.
    #[must_use]
    pub fn with_input_name(mut self, name: impl Into<String>) -> Self {
        self.input_name = name.into();
        self
    }

    /// Builder-style setter overriding the output tensor name used as
    /// the embedding.  Necessary for multi-head models where the first
    /// output is not the embedding.
    #[must_use]
    pub fn with_output_name(mut self, name: impl Into<String>) -> Self {
        self.output_name = name.into();
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

    /// Path of the ONNX model that backs this extractor.
    #[must_use]
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Shared handle to the underlying [`OnnxModel`].
    #[must_use]
    pub fn shared_model(&self) -> Arc<OnnxModel> {
        self.model.clone()
    }

    /// Run the embedding model on a preprocessed input tensor and
    /// return an L2-normalised [`ContentEmbedding`].
    ///
    /// `data.len()` must equal `shape.iter().product::<usize>()`.  The
    /// embedding is taken from the output tensor named
    /// [`Self::output_name`] (defaults to the first model output).
    ///
    /// # Errors
    ///
    /// * [`RecommendError::Ml`] wrapping any error raised by the
    ///   underlying ONNX session.
    /// * [`RecommendError::Ml`] if the configured output tensor name is
    ///   missing from the model's output map.
    /// * [`RecommendError::InvalidSimilarity`] if the returned embedding
    ///   has zero norm (which would leave it un-normalisable).
    pub fn extract(&self, data: Vec<f32>, shape: Vec<usize>) -> RecommendResult<ContentEmbedding> {
        let mut outputs = self.model.run_single(&self.input_name, data, shape)?;
        let raw = outputs.remove(&self.output_name).ok_or_else(|| {
            RecommendError::Ml(oximedia_ml::MlError::pipeline(
                "embed",
                format!(
                    "output '{}' missing from model '{}'",
                    self.output_name,
                    self.model_path.display(),
                ),
            ))
        })?;
        ContentEmbedding::new(raw)
    }
}

impl std::fmt::Debug for EmbeddingExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingExtractor")
            .field("input_name", &self.input_name)
            .field("output_name", &self.output_name)
            .field("model_path", &self.model_path)
            .finish()
    }
}

/// Rank `candidates` against `query` by descending cosine similarity and
/// return the top-`k` `(index, similarity)` pairs.
///
/// `index` refers to the position of the candidate in the input slice,
/// so callers can map it back to whatever content id they use.
///
/// Candidates with a dimensionality different from `query` contribute a
/// similarity of `0.0` (see [`ContentEmbedding::cosine_similarity`]) —
/// they effectively fall to the bottom of the ranking rather than
/// corrupting it.
#[must_use]
pub fn rank_by_similarity(
    query: &ContentEmbedding,
    candidates: &[ContentEmbedding],
    top_k: usize,
) -> Vec<(usize, f32)> {
    if candidates.is_empty() || top_k == 0 {
        return Vec::new();
    }
    let mut scored: Vec<(usize, f32)> = candidates
        .iter()
        .enumerate()
        .map(|(idx, c)| (idx, query.cosine_similarity(c)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    scored
}

/// Thin re-export layer for [`oximedia_ml::pipelines::FaceEmbedder`] to
/// support face-based content similarity out of the box.  Only compiled
/// when the `face-embedder-sim` feature is enabled.
#[cfg(feature = "face-embedder-sim")]
pub mod face {
    use super::{ContentEmbedding, RecommendResult};
    use oximedia_ml::pipelines::{FaceEmbedder, FaceImage};
    use oximedia_ml::{DeviceType, TypedPipeline};
    use std::path::Path;

    /// Opt-in face-based content similarity helper.
    ///
    /// Wraps [`FaceEmbedder`] and lifts its unit-norm
    /// [`oximedia_ml::FaceEmbedding`] output into a crate-native
    /// [`ContentEmbedding`].  Because the underlying pipeline already
    /// L2-normalises, this wrapper uses
    /// [`ContentEmbedding::from_normalized`] to avoid a redundant pass.
    pub struct FaceContentExtractor {
        embedder: FaceEmbedder,
    }

    impl FaceContentExtractor {
        /// Load a face-embedder ONNX model (ArcFace-compatible).
        ///
        /// # Errors
        ///
        /// Returns [`crate::RecommendError::Ml`] wrapping any
        /// [`oximedia_ml::MlError`] raised during load.
        pub fn from_path(path: impl AsRef<Path>, device: DeviceType) -> RecommendResult<Self> {
            let embedder = FaceEmbedder::load(path, device)?;
            Ok(Self { embedder })
        }

        /// Produce a unit-norm [`ContentEmbedding`] from an aligned face
        /// RGB crop (112×112 by default).
        ///
        /// # Errors
        ///
        /// * [`crate::RecommendError::Ml`] if the preprocessor/pipeline
        ///   rejects the crop or the ONNX session fails.
        pub fn extract(
            &self,
            pixels: Vec<u8>,
            width: u32,
            height: u32,
        ) -> RecommendResult<ContentEmbedding> {
            let image = FaceImage::new(pixels, width, height)?;
            let face = self.embedder.run(image)?;
            Ok(ContentEmbedding::from_normalized(face.into_inner()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_ml::MlError;

    #[test]
    fn content_embedding_new_normalises_vector() {
        let e = ContentEmbedding::new(vec![3.0, 4.0]).expect("ok");
        let s = e.as_slice();
        let norm: f32 = s.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        assert_eq!(e.dim(), 2);
    }

    #[test]
    fn content_embedding_new_rejects_empty() {
        let err = ContentEmbedding::new(Vec::<f32>::new()).expect_err("must fail");
        assert!(matches!(err, RecommendError::InvalidSimilarity(_)));
    }

    #[test]
    fn content_embedding_new_rejects_zero_vector() {
        let err = ContentEmbedding::new(vec![0.0_f32; 4]).expect_err("must fail");
        assert!(matches!(err, RecommendError::InvalidSimilarity(_)));
    }

    #[test]
    fn from_normalized_bypasses_normalisation() {
        // Deliberately not unit-norm — `from_normalized` trusts the caller.
        let raw = vec![1.0_f32, 2.0, 3.0];
        let e = ContentEmbedding::from_normalized(raw.clone());
        assert_eq!(e.as_slice(), raw.as_slice());
    }

    #[test]
    fn cosine_similarity_identical_is_one() {
        let a = ContentEmbedding::new(vec![1.0_f32, 0.0, 0.0]).expect("ok");
        let b = ContentEmbedding::new(vec![1.0_f32, 0.0, 0.0]).expect("ok");
        assert!((a.cosine_similarity(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_is_zero() {
        let a = ContentEmbedding::new(vec![1.0_f32, 0.0]).expect("ok");
        let b = ContentEmbedding::new(vec![0.0_f32, 1.0]).expect("ok");
        assert!(a.cosine_similarity(&b).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_antiparallel_is_minus_one() {
        let a = ContentEmbedding::new(vec![1.0_f32, 0.0]).expect("ok");
        let b = ContentEmbedding::new(vec![-1.0_f32, 0.0]).expect("ok");
        assert!((a.cosine_similarity(&b) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn euclidean_distance_on_unit_norm_obeys_identity() {
        // For unit-norm vectors, euclidean == sqrt(2 - 2 * cos).
        let a = ContentEmbedding::new(vec![1.0_f32, 0.0]).expect("ok");
        let b = ContentEmbedding::new(vec![0.0_f32, 1.0]).expect("ok");
        let cos = a.cosine_similarity(&b);
        let expected = (2.0 - 2.0 * cos).max(0.0).sqrt();
        assert!((a.euclidean_distance(&b) - expected).abs() < 1e-5);
    }

    #[test]
    fn euclidean_distance_dim_mismatch_is_infinity() {
        let a = ContentEmbedding::new(vec![1.0_f32, 0.0]).expect("ok");
        let b = ContentEmbedding::new(vec![1.0_f32, 0.0, 0.0]).expect("ok");
        assert!(a.euclidean_distance(&b).is_infinite());
    }

    #[test]
    fn rank_by_similarity_returns_descending_top_k() {
        let query = ContentEmbedding::new(vec![1.0_f32, 0.0, 0.0]).expect("ok");
        // cos([1,0,0], [1,0,0]) = 1.0 ; cos([1,0,0], [0,1,0]) = 0.0 ;
        // cos([1,0,0], [0.5, 0.5, 0.0]) = 0.5/sqrt(0.5) = 1/sqrt(2) ≈ 0.707.
        let candidates = vec![
            ContentEmbedding::new(vec![0.0_f32, 1.0, 0.0]).expect("ok"),
            ContentEmbedding::new(vec![1.0_f32, 0.0, 0.0]).expect("ok"),
            ContentEmbedding::new(vec![0.5_f32, 0.5, 0.0]).expect("ok"),
        ];
        let ranked = rank_by_similarity(&query, &candidates, 2);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].0, 1);
        assert!((ranked[0].1 - 1.0).abs() < 1e-5);
        assert_eq!(ranked[1].0, 2);
    }

    #[test]
    fn rank_by_similarity_empty_candidates_returns_empty() {
        let query = ContentEmbedding::new(vec![1.0_f32, 0.0]).expect("ok");
        assert!(rank_by_similarity(&query, &[], 5).is_empty());
    }

    #[test]
    fn rank_by_similarity_top_k_zero_returns_empty() {
        let query = ContentEmbedding::new(vec![1.0_f32, 0.0]).expect("ok");
        let c = vec![ContentEmbedding::new(vec![1.0_f32, 0.0]).expect("ok")];
        assert!(rank_by_similarity(&query, &c, 0).is_empty());
    }

    #[test]
    fn rank_by_similarity_larger_top_k_than_candidates_is_capped() {
        let query = ContentEmbedding::new(vec![1.0_f32, 0.0]).expect("ok");
        let c = vec![
            ContentEmbedding::new(vec![1.0_f32, 0.0]).expect("ok"),
            ContentEmbedding::new(vec![0.0_f32, 1.0]).expect("ok"),
        ];
        let ranked = rank_by_similarity(&query, &c, 100);
        assert_eq!(ranked.len(), 2);
    }

    #[test]
    fn ml_error_from_conversion_is_wired() {
        // Exercises `RecommendError: From<MlError>` so the `#[from]`
        // derive stays connected even when no other test path touches it.
        let ml_err = MlError::FeatureDisabled("onnx");
        let rec_err: RecommendError = ml_err.into();
        match rec_err {
            RecommendError::Ml(inner) => {
                assert!(matches!(inner, MlError::FeatureDisabled("onnx")));
            }
            other => panic!("unexpected conversion result: {other:?}"),
        }
    }

    #[test]
    fn from_path_missing_file_returns_ml_error() {
        let path = std::path::PathBuf::from("/does-not-exist-oximedia-recommend-embedding.onnx");
        let err = EmbeddingExtractor::from_path(&path, DeviceType::Cpu)
            .expect_err("loading a nonexistent model must fail");
        assert!(
            matches!(err, RecommendError::Ml(_)),
            "expected RecommendError::Ml, got {err:?}"
        );
    }
}
