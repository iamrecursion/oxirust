//! ML-assisted shot boundary detection.
//!
//! This module provides [`MlShotDetector`], a thin wrapper around
//! [`oximedia_ml::pipelines::ShotBoundaryDetector`] (TransNet V2-compatible)
//! that attaches ONNX-driven shot boundary probabilities to the same
//! [`crate::FrameBuffer`] layout already consumed by the crate's heuristic
//! [`crate::ShotDetector`].  The wrapper is purely **additive**: it never
//! replaces or mutates the existing heuristic detector, and the whole
//! module is gated behind the `onnx` Cargo feature so default builds
//! remain free of ONNX symbols.
//!
//! # Design
//!
//! The heuristic detector operates on [`crate::FrameBuffer`] sequences
//! shaped `height × width × 3` — see
//! [`crate::ShotDetector::detect_shots`].  `MlShotDetector` mirrors that
//! contract: callers hand in a slice of [`FrameBuffer`]s and receive
//! [`oximedia_ml::pipelines::ShotBoundary`] entries (frame index plus
//! confidence in `[0, 1]`) from the ONNX-backed TransNet V2-style pipeline
//! living in `oximedia-ml`.
//!
//! # Error mapping
//!
//! Every fallible operation returns [`crate::ShotResult`].  The
//! [`oximedia_ml::MlError`] type is folded into [`crate::ShotError`] via
//! `thiserror`'s `#[from]` conversion declared on the `MlError` variant.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_shots::ml::MlShotDetector;
//! use oximedia_shots::FrameBuffer;
//! use oximedia_ml::DeviceType;
//!
//! # fn run() -> oximedia_shots::ShotResult<()> {
//! let mut detector = MlShotDetector::from_path(
//!     "transnet_v2.onnx",
//!     DeviceType::auto(),
//! )?
//! .with_threshold(0.6)
//! .with_window(100);
//!
//! let frames: Vec<FrameBuffer> = vec![FrameBuffer::zeros(27, 48, 3); 4];
//! let boundaries = detector.detect_boundaries(&frames)?;
//! for b in boundaries {
//!     println!("boundary at frame {}: confidence={:.3}", b.frame_index, b.confidence);
//! }
//! # Ok(()) }
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use oximedia_ml::pipelines::{
    ShotBoundary as MlShotBoundary, ShotBoundaryConfig as MlShotBoundaryConfig,
    ShotBoundaryDetector as MlShotBoundaryDetector, ShotFrame as MlShotFrame,
};
use oximedia_ml::{DeviceType, ModelCache, OnnxModel, PipelineInfo, TypedPipeline};

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::FrameBuffer;

/// Opt-in ML shot boundary detector that produces confidence-scored
/// boundaries from an ONNX TransNet V2-style model.
///
/// Wraps [`oximedia_ml::pipelines::ShotBoundaryDetector`] and forwards
/// the crate's native frame buffer layout (raw RGB8, row-major,
/// `h × w × 3`) to it without any extra copy beyond what the pipeline's
/// preprocessor already needs.
///
/// Because the underlying `ShotBoundaryDetector` currently exposes no
/// `shared_model()` / `config()` accessors, `MlShotDetector` retains its
/// own copies of the loaded [`OnnxModel`], the [`MlShotBoundaryConfig`],
/// and the model path so builder-style setters
/// ([`Self::with_threshold`], [`Self::with_window`]) can rebuild the
/// inner pipeline without touching disk a second time.
pub struct MlShotDetector {
    detector: MlShotBoundaryDetector,
    shared: Arc<OnnxModel>,
    config: MlShotBoundaryConfig,
    model_path: PathBuf,
    /// Mirrored threshold exposed via [`Self::threshold`].
    threshold: f32,
    /// Mirrored window size exposed via [`Self::window`].
    window: usize,
}

impl MlShotDetector {
    /// Load a shot-boundary ONNX model from disk and return a
    /// ready-to-use detector.
    ///
    /// # Errors
    ///
    /// * Returns [`ShotError::MlError`] wrapping [`oximedia_ml::MlError::ModelLoad`]
    ///   if the ONNX model cannot be opened.
    /// * Returns [`ShotError::MlError`] wrapping
    ///   [`oximedia_ml::MlError::DeviceUnavailable`] if the requested device is
    ///   not compiled in or is unavailable at runtime.
    pub fn from_path(model_path: impl AsRef<Path>, device: DeviceType) -> ShotResult<Self> {
        let path = model_path.as_ref().to_path_buf();
        let model = Arc::new(OnnxModel::load(&path, device)?);
        Ok(Self::build(model, MlShotBoundaryConfig::default(), path))
    }

    /// Build a detector from a shared [`OnnxModel`] (typically resolved
    /// via [`ModelCache`]).  Useful when multiple pipelines share the same
    /// weights file.
    ///
    /// This mirrors [`MlShotBoundaryDetector::from_shared`] but returns the
    /// crate-native [`MlShotDetector`] for ergonomics.
    #[must_use]
    pub fn from_shared_model(model: Arc<OnnxModel>, model_path: PathBuf) -> Self {
        Self::build(model, MlShotBoundaryConfig::default(), model_path)
    }

    /// Resolve a detector against a [`ModelCache`], sharing the `OnnxModel`
    /// with any other caller that loaded the same path.
    ///
    /// # Errors
    ///
    /// Propagates any [`oximedia_ml::MlError`] raised by the cache loader.
    pub fn from_cache(
        cache: &ModelCache,
        model_path: impl AsRef<Path>,
        device: DeviceType,
    ) -> ShotResult<Self> {
        let path = model_path.as_ref().to_path_buf();
        let model = cache.get_or_load(&path, device)?;
        Ok(Self::from_shared_model(model, path))
    }

    fn build(model: Arc<OnnxModel>, config: MlShotBoundaryConfig, model_path: PathBuf) -> Self {
        let threshold = config.threshold;
        let window = config.window;
        let detector =
            MlShotBoundaryDetector::from_shared(model.clone(), config.clone(), model_path.clone());
        Self {
            detector,
            shared: model,
            config,
            model_path,
            threshold,
            window,
        }
    }

    /// Builder-style setter overriding the per-frame confidence threshold
    /// above which a boundary is emitted.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        // Rebuild the inner pipeline with the new threshold while reusing the
        // already-loaded weights.  This keeps the API chainable without
        // forcing a second disk read.
        self.config.threshold = threshold;
        self.threshold = threshold;
        self.detector = MlShotBoundaryDetector::from_shared(
            self.shared.clone(),
            self.config.clone(),
            self.model_path.clone(),
        );
        self
    }

    /// Builder-style setter overriding the sliding-window size (number of
    /// frames the detector reasons over per call).
    #[must_use]
    pub fn with_window(mut self, window: usize) -> Self {
        self.config.window = window;
        self.window = window;
        self.detector = MlShotBoundaryDetector::from_shared(
            self.shared.clone(),
            self.config.clone(),
            self.model_path.clone(),
        );
        self
    }

    /// Return the currently configured threshold.
    #[must_use]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Return the currently configured window size.
    #[must_use]
    pub fn window(&self) -> usize {
        self.window
    }

    /// Immutable view of the inner pipeline info — useful for logging /
    /// telemetry.
    #[must_use]
    pub fn info(&self) -> PipelineInfo {
        self.detector.info()
    }

    /// Run ML shot-boundary detection on a slice of native
    /// [`FrameBuffer`]s and return the detected boundaries.
    ///
    /// Each frame layout matches the crate's native heuristic detector:
    /// `height × width × 3`, row-major, 8 bits per channel.  An empty
    /// slice returns an empty result without invoking the model.
    ///
    /// # Errors
    ///
    /// * [`ShotError::InvalidFrame`] if any frame does not have exactly
    ///   3 channels or cannot have its dimensions narrowed to `u32`.
    /// * [`ShotError::MlError`] wrapping any failure raised by the
    ///   underlying ONNX pipeline (preprocess / inference / postprocess).
    pub fn detect_boundaries(&self, frames: &[FrameBuffer]) -> ShotResult<Vec<MlShotBoundary>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        let mut shot_frames: Vec<MlShotFrame> = Vec::with_capacity(frames.len());
        for (idx, frame) in frames.iter().enumerate() {
            let (h, w, c) = frame.dim();
            if c != 3 {
                return Err(ShotError::InvalidFrame(format!(
                    "frame {idx}: expected 3 channels (RGB), got {c}"
                )));
            }
            let w32 = u32::try_from(w).map_err(|_| {
                ShotError::InvalidFrame(format!("frame {idx}: width {w} does not fit in u32"))
            })?;
            let h32 = u32::try_from(h).map_err(|_| {
                ShotError::InvalidFrame(format!("frame {idx}: height {h} does not fit in u32"))
            })?;
            let pixels = frame.as_slice().to_vec();
            shot_frames.push(MlShotFrame::new(pixels, w32, h32)?);
        }

        let boundaries = self.detector.run(shot_frames)?;
        Ok(boundaries)
    }
}

impl std::fmt::Debug for MlShotDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlShotDetector")
            .field("threshold", &self.threshold)
            .field("window", &self.window)
            .field("model_path", &self.model_path)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_ml::MlError;
    use std::path::PathBuf;

    #[test]
    fn from_path_missing_file_returns_ml_error() {
        let path = PathBuf::from("/does-not-exist-oximedia-shots.onnx");
        let err = MlShotDetector::from_path(&path, DeviceType::Cpu)
            .expect_err("loading a nonexistent model must fail");
        assert!(
            matches!(err, ShotError::MlError(_)),
            "expected ShotError::MlError, got {err:?}"
        );
    }

    #[test]
    fn ml_error_from_conversion_is_wired() {
        // Exercises `ShotError: From<MlError>` so the `#[from]` derive stays
        // connected even when no other test path touches it.
        let ml_err = MlError::FeatureDisabled("onnx");
        let shot_err: ShotError = ml_err.into();
        match shot_err {
            ShotError::MlError(inner) => {
                assert!(matches!(inner, MlError::FeatureDisabled("onnx")));
            }
            other => panic!("unexpected conversion result: {other:?}"),
        }
    }
}
