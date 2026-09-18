//! TransNet V2-style shot boundary detector pipeline.
//!
//! Consumes a sliding window of video frames and returns the per-frame
//! probabilities of a shot cut. When the `onnx` feature is disabled
//! this pipeline falls back to a deterministic no-model implementation
//! that scores boundaries via frame-difference heuristics, so downstream
//! callers can still get *something* useful in a Pure-Rust-only build.
//!
//! ## Contract
//!
//! | Stage         | Shape / value                                                                        |
//! |---------------|--------------------------------------------------------------------------------------|
//! | Input         | `&[ShotFrame]` (a sliding window), each RGB u8                                       |
//! | Preprocess    | NCHW f32 in `[0, 1]`, resized to `frame_size` (default 48×27)                        |
//! | ONNX input    | `[1, window, 3, H, W]` — TransNet V2 defaults to `window == 100`                     |
//! | ONNX output   | Per-frame logits, length `window`                                                    |
//! | Postprocess   | [`crate::postprocess::sigmoid_slice`] + threshold + `min_gap` spacing                |
//! | Output        | `Vec<ShotBoundary>`                                                                  |
//!
//! ## Compatible models
//!
//! TransNet V2 exports, or any custom boundary detector that emits one
//! logit per window frame. If the `onnx` feature is disabled, or you
//! construct the detector via [`ShotBoundaryDetector::heuristic`], the
//! pipeline uses frame-difference L1 scoring instead of the model.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::device::DeviceType;
use crate::error::{MlError, MlResult};
use crate::model::OnnxModel;
use crate::pipeline::{PipelineInfo, PipelineTask, TypedPipeline};
#[cfg(feature = "onnx")]
use crate::postprocess::sigmoid_slice;
use crate::preprocess::{ImagePreprocessor, InputRange, TensorLayout};

/// Configuration knobs for [`ShotBoundaryDetector`].
#[derive(Clone, Debug)]
pub struct ShotBoundaryConfig {
    /// Per-frame spatial size. TransNet-V2 uses 48×27.
    pub frame_size: (u32, u32),
    /// Number of frames in the sliding window (TransNet-V2 uses 100).
    pub window: usize,
    /// Probability threshold above which a boundary is emitted.
    pub threshold: f32,
    /// Minimum number of frames between successive boundaries.
    pub min_gap: usize,
    /// Model input tensor name (overrides auto-detection).
    pub input_name: Option<String>,
    /// Model output tensor name (overrides auto-detection).
    pub output_name: Option<String>,
}

impl Default for ShotBoundaryConfig {
    fn default() -> Self {
        Self {
            frame_size: (48, 27),
            window: 100,
            threshold: 0.5,
            min_gap: 3,
            input_name: None,
            output_name: None,
        }
    }
}

/// Single detected shot boundary.
#[derive(Clone, Copy, Debug)]
pub struct ShotBoundary {
    /// Index of the boundary frame within the window.
    pub frame_index: usize,
    /// Confidence in `[0, 1]` (1.0 = definite boundary).
    pub confidence: f32,
}

/// A frame for shot boundary detection.
#[derive(Clone, Debug)]
pub struct ShotFrame {
    /// RGB u8 pixels.
    pub pixels: Vec<u8>,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
}

impl ShotFrame {
    /// Create a frame, validating buffer length.
    pub fn new(pixels: Vec<u8>, width: u32, height: u32) -> MlResult<Self> {
        let expected = (width as usize) * (height as usize) * 3;
        if pixels.len() != expected {
            return Err(MlError::invalid_input(format!(
                "shot frame: expected {expected} bytes for {width}x{height} RGB, got {}",
                pixels.len()
            )));
        }
        Ok(Self {
            pixels,
            width,
            height,
        })
    }
}

/// Shot boundary detector pipeline.
///
/// Construct via [`Self::load`] / [`Self::load_with_config`] for the
/// ONNX path, or [`Self::heuristic`] for the no-model frame-difference
/// fallback. Implements [`TypedPipeline`] with
/// `Input = Vec<ShotFrame>` and `Output = Vec<ShotBoundary>`.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "shot-boundary")]
/// # fn demo() -> oximedia_ml::MlResult<()> {
/// use oximedia_ml::pipelines::{ShotBoundaryConfig, ShotBoundaryDetector, ShotFrame};
/// use oximedia_ml::TypedPipeline;
///
/// // Heuristic path — always available.
/// let detector = ShotBoundaryDetector::heuristic(ShotBoundaryConfig::default());
/// let frames = vec![
///     ShotFrame::new(vec![0_u8; 48 * 27 * 3], 48, 27)?,
///     ShotFrame::new(vec![255_u8; 48 * 27 * 3], 48, 27)?,
/// ];
/// let _boundaries = detector.run(frames)?;
/// # Ok(())
/// # }
/// ```
pub struct ShotBoundaryDetector {
    model: Option<Arc<OnnxModel>>,
    config: ShotBoundaryConfig,
    preprocessor: ImagePreprocessor,
    model_path: Option<PathBuf>,
}

impl ShotBoundaryDetector {
    /// Construct a heuristic detector that does not use any ONNX model.
    ///
    /// This variant is always available regardless of features.
    #[must_use]
    pub fn heuristic(config: ShotBoundaryConfig) -> Self {
        let (w, h) = config.frame_size;
        let preprocessor = ImagePreprocessor::new(w, h)
            .with_tensor_layout(TensorLayout::Nchw)
            .with_input_range(InputRange::U8);
        Self {
            model: None,
            config,
            preprocessor,
            model_path: None,
        }
    }

    /// Load an ONNX model for shot boundary detection.
    ///
    /// # Errors
    ///
    /// Propagates any error from
    /// [`OnnxModel::load`][crate::OnnxModel::load].
    pub fn load(path: impl AsRef<Path>, device: DeviceType) -> MlResult<Self> {
        Self::load_with_config(path, device, ShotBoundaryConfig::default())
    }

    /// Load an ONNX model with a custom configuration.
    ///
    /// # Errors
    ///
    /// Propagates any error from
    /// [`OnnxModel::load`][crate::OnnxModel::load].
    pub fn load_with_config(
        path: impl AsRef<Path>,
        device: DeviceType,
        config: ShotBoundaryConfig,
    ) -> MlResult<Self> {
        let model_path = path.as_ref().to_path_buf();
        let model = Arc::new(OnnxModel::load(&model_path, device)?);
        Ok(Self::build(Some(model), config, Some(model_path)))
    }

    /// Build from a pre-loaded shared model.
    #[must_use]
    pub fn from_shared(
        model: Arc<OnnxModel>,
        config: ShotBoundaryConfig,
        model_path: PathBuf,
    ) -> Self {
        Self::build(Some(model), config, Some(model_path))
    }

    fn build(
        model: Option<Arc<OnnxModel>>,
        config: ShotBoundaryConfig,
        model_path: Option<PathBuf>,
    ) -> Self {
        let (w, h) = config.frame_size;
        let preprocessor = ImagePreprocessor::new(w, h)
            .with_tensor_layout(TensorLayout::Nchw)
            .with_input_range(InputRange::U8);
        Self {
            model,
            config,
            preprocessor,
            model_path,
        }
    }

    /// Path to the ONNX model, if any.
    #[must_use]
    pub fn model_path(&self) -> Option<&Path> {
        self.model_path.as_deref()
    }

    /// Whether a real ONNX model is attached.
    #[must_use]
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Configured threshold.
    #[must_use]
    pub fn threshold(&self) -> f32 {
        self.config.threshold
    }

    fn filter_peaks(&self, probs: &[f32]) -> Vec<ShotBoundary> {
        let mut out = Vec::new();
        let mut last_emitted: Option<usize> = None;
        for (idx, &p) in probs.iter().enumerate() {
            if p < self.config.threshold {
                continue;
            }
            if let Some(prev) = last_emitted {
                if idx.saturating_sub(prev) < self.config.min_gap {
                    continue;
                }
            }
            out.push(ShotBoundary {
                frame_index: idx,
                confidence: p,
            });
            last_emitted = Some(idx);
        }
        out
    }

    fn heuristic_probs(&self, frames: &[ShotFrame]) -> MlResult<Vec<f32>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let mut probs = vec![0.0_f32; frames.len()];
        let mut previous: Option<Vec<f32>> = None;
        for (idx, frame) in frames.iter().enumerate() {
            let buf = self
                .preprocessor
                .process_u8_rgb(&frame.pixels, frame.width, frame.height)?;
            if let Some(prev) = &previous {
                // Normalized L1 distance between consecutive preprocessed frames.
                let mut sum = 0.0_f32;
                let mut max_possible = 0.0_f32;
                for (a, b) in prev.iter().zip(buf.iter()) {
                    sum += (a - b).abs();
                    max_possible += 1.0;
                }
                if max_possible > 0.0 {
                    let ratio = (sum / max_possible).clamp(0.0, 1.0);
                    probs[idx] = ratio;
                }
            }
            previous = Some(buf);
        }
        Ok(probs)
    }
}

impl TypedPipeline for ShotBoundaryDetector {
    type Input = Vec<ShotFrame>;
    type Output = Vec<ShotBoundary>;

    fn run(&self, input: Self::Input) -> MlResult<Self::Output> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        #[cfg(feature = "onnx")]
        {
            if let Some(model) = self.model.as_ref() {
                use oxionnx::Tensor;
                use std::collections::HashMap;

                let (w, h) = self.config.frame_size;
                let mut batched: Vec<f32> =
                    Vec::with_capacity((w as usize) * (h as usize) * 3 * input.len());
                for frame in &input {
                    let buf = self.preprocessor.process_u8_rgb(
                        &frame.pixels,
                        frame.width,
                        frame.height,
                    )?;
                    batched.extend(buf);
                }
                let shape = vec![1, input.len(), 3, h as usize, w as usize];
                let tensor = Tensor {
                    data: batched,
                    shape,
                };

                let input_name = self
                    .config
                    .input_name
                    .clone()
                    .or_else(|| model.info().inputs.first().map(|s| s.name.clone()))
                    .ok_or_else(|| MlError::invalid_input("model has no declared inputs"))?;
                let output_name = self
                    .config
                    .output_name
                    .clone()
                    .or_else(|| model.info().outputs.first().map(|s| s.name.clone()))
                    .ok_or_else(|| MlError::invalid_input("model has no declared outputs"))?;

                let mut inputs: HashMap<&str, Tensor> = HashMap::with_capacity(1);
                inputs.insert(input_name.as_str(), tensor);
                let outputs = model.run(&inputs)?;
                let out = outputs.get(&output_name).ok_or_else(|| {
                    MlError::postprocess(format!("output '{output_name}' missing from model run"))
                })?;
                let probs = sigmoid_slice(&out.data);
                return Ok(self.filter_peaks(&probs));
            }
        }

        // Heuristic fallback: available regardless of feature configuration.
        let probs = self.heuristic_probs(&input)?;
        Ok(self.filter_peaks(&probs))
    }

    fn info(&self) -> PipelineInfo {
        PipelineInfo {
            id: "shot-boundary/transnet-v2",
            name: "Shot Boundary Detector",
            task: PipelineTask::ShotBoundary,
            input_size: Some(self.config.frame_size),
        }
    }
}

impl std::fmt::Debug for ShotBoundaryDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShotBoundaryDetector")
            .field("model_path", &self.model_path)
            .field("threshold", &self.config.threshold)
            .field("window", &self.config.window)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(w: u32, h: u32, rgb: [u8; 3]) -> ShotFrame {
        let mut buf = Vec::with_capacity((w as usize) * (h as usize) * 3);
        for _ in 0..(w as usize * h as usize) {
            buf.extend_from_slice(&rgb);
        }
        ShotFrame::new(buf, w, h).expect("valid frame")
    }

    #[test]
    fn empty_input_returns_empty() {
        let det = ShotBoundaryDetector::heuristic(ShotBoundaryConfig::default());
        let out = det.run(Vec::new()).expect("ok");
        assert!(out.is_empty());
    }

    #[test]
    fn heuristic_detects_color_change() {
        let det = ShotBoundaryDetector::heuristic(ShotBoundaryConfig {
            threshold: 0.1,
            min_gap: 0,
            ..Default::default()
        });
        let frames = vec![
            solid_frame(48, 27, [0, 0, 0]),
            solid_frame(48, 27, [0, 0, 0]),
            solid_frame(48, 27, [255, 255, 255]),
            solid_frame(48, 27, [255, 255, 255]),
        ];
        let out = det.run(frames).expect("ok");
        // Expect a boundary at index 2 (large frame difference).
        assert!(out.iter().any(|b| b.frame_index == 2));
    }

    #[test]
    fn default_config_is_transnet_shaped() {
        let cfg = ShotBoundaryConfig::default();
        assert_eq!(cfg.frame_size, (48, 27));
        assert_eq!(cfg.window, 100);
    }

    #[test]
    fn shot_frame_rejects_wrong_buffer() {
        let err = ShotFrame::new(vec![0u8; 10], 48, 27).expect_err("must fail");
        assert!(matches!(err, MlError::InvalidInput(_)));
    }
}
