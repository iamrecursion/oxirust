//! ML-assisted scene classification.
//!
//! This module provides [`MlSceneEnricher`], a thin wrapper around
//! [`oximedia_ml::pipelines::SceneClassifier`] that attaches scene-class
//! predictions (indoor/outdoor/nature/city/…) to frames already handled by
//! the crate's heuristic scene pipeline.  The wrapper is purely **additive**:
//! it never replaces or mutates the existing heuristic
//! [`crate::classify::scene::SceneClassifier`], and the whole module is
//! gated behind the `onnx` Cargo feature so default builds remain free of
//! ONNX symbols.
//!
//! # Design
//!
//! The heuristic classifier operates on raw RGB8 images shaped
//! `width × height × 3` — see
//! [`crate::classify::scene::SceneClassifier::classify`].  `MlSceneEnricher`
//! mirrors that contract: callers hand in the same byte buffer plus its
//! dimensions and receive top-K `(label, score)` predictions from the
//! ONNX-backed Places365-style pipeline living in `oximedia-ml`.
//!
//! # Error mapping
//!
//! Every fallible operation returns [`crate::SceneResult`].  The
//! [`oximedia_ml::MlError`] type is folded into [`crate::SceneError`] via
//! `thiserror`'s `#[from]` conversion declared on the `MlError` variant.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_scene::ml::MlSceneEnricher;
//! use oximedia_ml::DeviceType;
//!
//! # fn run() -> oximedia_scene::SceneResult<()> {
//! let labels = vec![
//!     "indoor".to_string(),
//!     "outdoor".to_string(),
//!     "nature".to_string(),
//! ];
//! let mut enricher = MlSceneEnricher::from_path(
//!     "places365.onnx",
//!     labels,
//!     DeviceType::auto(),
//! )?
//! .with_top_k(3);
//!
//! let rgb: Vec<u8> = vec![0; 224 * 224 * 3];
//! let predictions = enricher.classify_frame(&rgb, 224, 224)?;
//! for (label, score) in predictions {
//!     println!("{label}: {score:.3}");
//! }
//! # Ok(()) }
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use oximedia_ml::pipelines::{
    SceneClassification as MlSceneClassification, SceneClassifier as MlSceneClassifier,
    SceneClassifierConfig, SceneImage,
};
use oximedia_ml::{DeviceType, ModelCache, OnnxModel, PipelineInfo, TypedPipeline};

use crate::error::{SceneError, SceneResult};

/// Opt-in ML scene enricher that attaches ONNX-produced `(label, score)`
/// predictions to the heuristic scene-detection output.
///
/// Wraps [`oximedia_ml::pipelines::SceneClassifier`] and forwards the
/// crate's native frame buffer layout (raw RGB8, row-major, `w × h × 3`) to
/// it without any extra copy beyond what the preprocessor already needs.
pub struct MlSceneEnricher {
    classifier: MlSceneClassifier,
    /// Number of top predictions to return per classification.
    ///
    /// The underlying `SceneClassifier` reads its own `top_k` from its
    /// config; this mirror exists so [`Self::with_top_k`] can expose a
    /// builder-style setter without needing interior mutability on the
    /// pipeline itself.
    top_k: usize,
}

impl MlSceneEnricher {
    /// Load a scene-classifier ONNX model from disk and return a
    /// ready-to-use enricher.
    ///
    /// `labels` are the human-readable class names indexed by model class
    /// id.  They are stored inside the pipeline config so every
    /// [`Self::classify_frame`] call can resolve class indices to names.
    ///
    /// # Errors
    ///
    /// * Returns [`SceneError::MlError`] wrapping
    ///   [`oximedia_ml::MlError::ModelLoad`] if the ONNX model cannot be
    ///   opened.
    /// * Returns [`SceneError::MlError`] wrapping
    ///   [`oximedia_ml::MlError::DeviceUnavailable`] if the requested device
    ///   is not compiled in or is unavailable at runtime.
    pub fn from_path(
        model_path: impl AsRef<Path>,
        labels: Vec<String>,
        device: DeviceType,
    ) -> SceneResult<Self> {
        let config = SceneClassifierConfig {
            labels: Some(labels),
            ..SceneClassifierConfig::default()
        };
        let top_k = config.top_k;
        let classifier = MlSceneClassifier::load_with_config(model_path, device, config)?;
        Ok(Self { classifier, top_k })
    }

    /// Build an enricher from a shared [`OnnxModel`] (typically resolved
    /// via [`ModelCache`]).  Useful when multiple pipelines share the same
    /// weights file.
    ///
    /// This mirrors [`MlSceneClassifier::from_shared`] but returns the
    /// crate-native [`SceneResult`] type for ergonomics.
    #[must_use]
    pub fn from_shared_model(
        model: Arc<OnnxModel>,
        labels: Vec<String>,
        model_path: PathBuf,
    ) -> Self {
        let config = SceneClassifierConfig {
            labels: Some(labels),
            ..SceneClassifierConfig::default()
        };
        let top_k = config.top_k;
        let classifier = MlSceneClassifier::from_shared(model, config, model_path);
        Self { classifier, top_k }
    }

    /// Resolve an enricher against a [`ModelCache`], sharing the `OnnxModel`
    /// with any other caller that loaded the same path.
    ///
    /// # Errors
    ///
    /// Propagates any [`oximedia_ml::MlError`] raised by the cache loader.
    pub fn from_cache(
        cache: &ModelCache,
        model_path: impl AsRef<Path>,
        labels: Vec<String>,
        device: DeviceType,
    ) -> SceneResult<Self> {
        let path = model_path.as_ref().to_path_buf();
        let model = cache.get_or_load(&path, device)?;
        Ok(Self::from_shared_model(model, labels, path))
    }

    /// Builder-style setter overriding the top-K retained per call.
    #[must_use]
    pub fn with_top_k(mut self, k: usize) -> Self {
        // Rebuild the inner pipeline with the new top-K while reusing the
        // already-loaded weights.  This keeps the API chainable without
        // forcing a second disk read.
        let shared = self.classifier.shared_model();
        let path = self.classifier.model_path().to_path_buf();
        let mut config = self.classifier.config().clone();
        config.top_k = k;
        self.top_k = k;
        self.classifier = MlSceneClassifier::from_shared(shared, config, path);
        self
    }

    /// Return the currently configured top-K.
    #[must_use]
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Immutable view of the inner pipeline info — useful for logging /
    /// telemetry.
    #[must_use]
    pub fn info(&self) -> PipelineInfo {
        self.classifier.info()
    }

    /// Run ML classification on a raw RGB8 frame and return top-K
    /// `(label, score)` predictions.
    ///
    /// The frame layout matches the crate's native heuristic classifier:
    /// `rgb.len() == width * height * 3`, row-major, 8 bits per channel.
    ///
    /// When a class index has no label configured, the returned label
    /// falls back to `"class_<index>"` so downstream code can always
    /// display *some* identifier.
    ///
    /// # Errors
    ///
    /// * [`SceneError::InvalidDimensions`] if `rgb.len()` disagrees with
    ///   `width * height * 3`.
    /// * [`SceneError::MlError`] wrapping any failure raised by the
    ///   underlying ONNX pipeline (preprocess / inference / postprocess).
    pub fn classify_frame(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<Vec<(String, f32)>> {
        let expected = width
            .checked_mul(height)
            .and_then(|wh| wh.checked_mul(3))
            .ok_or_else(|| {
                SceneError::InvalidDimensions(format!(
                    "width*height*3 overflows usize: width={width} height={height}"
                ))
            })?;
        if rgb.len() != expected {
            return Err(SceneError::InvalidDimensions(format!(
                "expected {expected} bytes, got {}",
                rgb.len()
            )));
        }

        // Narrow the dimensions to the u32 range the ML pipeline expects.
        let w32 = u32::try_from(width).map_err(|_| {
            SceneError::InvalidDimensions(format!("width {width} does not fit in u32"))
        })?;
        let h32 = u32::try_from(height).map_err(|_| {
            SceneError::InvalidDimensions(format!("height {height} does not fit in u32"))
        })?;

        let image = SceneImage::new(rgb.to_vec(), w32, h32)?;
        let raw: Vec<MlSceneClassification> = self.classifier.run(image)?;
        Ok(raw.into_iter().map(label_score_pair).collect())
    }
}

impl std::fmt::Debug for MlSceneEnricher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlSceneEnricher")
            .field("top_k", &self.top_k)
            .field("model_path", &self.classifier.model_path())
            .finish()
    }
}

/// Convert one [`MlSceneClassification`] into a stable `(label, score)`
/// tuple, filling in a synthetic `"class_<idx>"` label when the model's
/// config carries no human-readable names.
fn label_score_pair(pred: MlSceneClassification) -> (String, f32) {
    let MlSceneClassification {
        class_index,
        label,
        score,
    } = pred;
    let label = label.unwrap_or_else(|| format!("class_{class_index}"));
    (label, score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_ml::MlError;
    use std::path::PathBuf;

    #[test]
    fn label_score_pair_uses_label_when_present() {
        let pred = MlSceneClassification {
            class_index: 7,
            label: Some("nature".to_string()),
            score: 0.42,
        };
        let (label, score) = label_score_pair(pred);
        assert_eq!(label, "nature");
        assert!((score - 0.42).abs() < f32::EPSILON);
    }

    #[test]
    fn label_score_pair_falls_back_to_class_index() {
        let pred = MlSceneClassification {
            class_index: 13,
            label: None,
            score: 0.77,
        };
        let (label, score) = label_score_pair(pred);
        assert_eq!(label, "class_13");
        assert!((score - 0.77).abs() < f32::EPSILON);
    }

    #[test]
    fn from_path_missing_file_returns_ml_error() {
        let labels = vec!["indoor".to_string(), "outdoor".to_string()];
        let path = PathBuf::from("/does-not-exist-oximedia-scene.onnx");
        let err = MlSceneEnricher::from_path(&path, labels, DeviceType::Cpu)
            .expect_err("loading a nonexistent model must fail");
        assert!(
            matches!(err, SceneError::MlError(_)),
            "expected SceneError::MlError, got {err:?}"
        );
    }

    #[test]
    fn ml_error_from_conversion_is_wired() {
        // Exercises `SceneError: From<MlError>` so the `#[from]` derive stays
        // connected even when no other test path touches it.
        let ml_err = MlError::FeatureDisabled("onnx");
        let scene_err: SceneError = ml_err.into();
        match scene_err {
            SceneError::MlError(inner) => {
                assert!(matches!(inner, MlError::FeatureDisabled("onnx")));
            }
            other => panic!("unexpected conversion result: {other:?}"),
        }
    }
}
