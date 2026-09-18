//! NIMA-style aesthetic quality scorer pipeline.
//!
//! Consumes a single RGB image and produces an [`AestheticScore`]:
//! the weighted mean of a 10-bin quality distribution in the range
//! `1.0..=10.0`.
//!
//! ## Contract
//!
//! | Stage         | Shape / value                                                                      |
//! |---------------|------------------------------------------------------------------------------------|
//! | Input         | RGB u8 [`AestheticImage`], resized to `input_size` (default 224×224)               |
//! | Preprocess    | NCHW f32 with ImageNet mean/std                                                    |
//! | ONNX input    | `[1, 3, 224, 224]`                                                                 |
//! | ONNX output   | `[1, 10]` — 10 bins corresponding to quality scores 1..=10                         |
//! | Postprocess   | Optional [`crate::postprocess::softmax`] (controlled by `apply_softmax`), then     |
//! |               | [`AestheticScore::from_distribution`] (NIMA weighted mean)                         |
//! | Output        | [`AestheticScore`] with `score() in [1.0, 10.0]` for a valid distribution          |
//!
//! ## Compatible models
//!
//! NIMA (Neural Image Assessment) exports — ResNet / MobileNet / VGG
//! variants — where the 10-way head represents probability of quality
//! bin `i ∈ 1..=10`. If your model already returns a normalised
//! distribution, set [`AestheticScorerConfig::apply_softmax`] to `false`
//! to skip the extra softmax.
//!
//! When the `onnx` feature is disabled,
//! [`crate::pipeline::TypedPipeline::run`] returns
//! [`MlError::FeatureDisabled`] so callers can degrade gracefully.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::device::DeviceType;
use crate::error::{MlError, MlResult};
use crate::model::OnnxModel;
use crate::pipeline::{PipelineInfo, PipelineTask, TypedPipeline};
#[cfg(feature = "onnx")]
use crate::postprocess::softmax;
use crate::preprocess::{ImagePreprocessor, TensorLayout};

use super::types::AestheticScore;

/// Input image for the aesthetic scorer.
#[derive(Clone, Debug)]
pub struct AestheticImage {
    /// RGB u8 pixels, row-major, length = `width * height * 3`.
    pub pixels: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl AestheticImage {
    /// Create a new aesthetic image, validating the buffer length.
    pub fn new(pixels: Vec<u8>, width: u32, height: u32) -> MlResult<Self> {
        let expected = (width as usize) * (height as usize) * 3;
        if pixels.len() != expected {
            return Err(MlError::invalid_input(format!(
                "aesthetic image: expected {expected} bytes for {width}x{height} RGB, got {}",
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

/// Configuration for [`AestheticScorer`].
#[derive(Clone, Debug)]
pub struct AestheticScorerConfig {
    /// Input size expected by the model. Defaults to 224×224 (NIMA).
    pub input_size: (u32, u32),
    /// Per-channel mean (ImageNet defaults).
    pub mean: [f32; 3],
    /// Per-channel std (ImageNet defaults).
    pub std: [f32; 3],
    /// Model input tensor name (defaults to the model's first input).
    pub input_name: Option<String>,
    /// Model output tensor name (defaults to the model's first output).
    pub output_name: Option<String>,
    /// If `true`, apply softmax to the model output before computing
    /// the weighted mean. Set to `false` when the model already emits
    /// a probability distribution. Defaults to `true`.
    pub apply_softmax: bool,
}

impl Default for AestheticScorerConfig {
    fn default() -> Self {
        Self {
            input_size: (224, 224),
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            input_name: None,
            output_name: None,
            apply_softmax: true,
        }
    }
}

/// NIMA-style aesthetic quality scorer.
///
/// Construct via [`Self::load`] or [`Self::load_with_config`]. Implements
/// [`TypedPipeline`] with `Input = AestheticImage`,
/// `Output = AestheticScore`.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(all(feature = "onnx", feature = "aesthetic-score"))]
/// # fn demo() -> oximedia_ml::MlResult<()> {
/// use oximedia_ml::pipelines::{AestheticImage, AestheticScorer};
/// use oximedia_ml::{DeviceType, TypedPipeline};
///
/// let scorer = AestheticScorer::load("nima.onnx", DeviceType::auto())?;
/// let image = AestheticImage::new(vec![0_u8; 224 * 224 * 3], 224, 224)?;
/// let score = scorer.run(image)?;
/// println!("aesthetic score = {:.2}", score.score());
/// # Ok(())
/// # }
/// ```
pub struct AestheticScorer {
    model: Arc<OnnxModel>,
    config: AestheticScorerConfig,
    preprocessor: ImagePreprocessor,
    model_path: PathBuf,
}

impl AestheticScorer {
    /// Load a scorer from an ONNX model path with default config.
    ///
    /// # Errors
    ///
    /// Propagates any error from
    /// [`OnnxModel::load`][crate::OnnxModel::load].
    pub fn load(path: impl AsRef<Path>, device: DeviceType) -> MlResult<Self> {
        Self::load_with_config(path, device, AestheticScorerConfig::default())
    }

    /// Load a scorer with a custom configuration.
    ///
    /// # Errors
    ///
    /// Propagates any error from
    /// [`OnnxModel::load`][crate::OnnxModel::load].
    pub fn load_with_config(
        path: impl AsRef<Path>,
        device: DeviceType,
        config: AestheticScorerConfig,
    ) -> MlResult<Self> {
        let model_path = path.as_ref().to_path_buf();
        let model = Arc::new(OnnxModel::load(&model_path, device)?);
        Ok(Self::build(model, config, model_path))
    }

    /// Build directly from a pre-loaded model.
    #[must_use]
    pub fn from_shared(
        model: Arc<OnnxModel>,
        config: AestheticScorerConfig,
        model_path: PathBuf,
    ) -> Self {
        Self::build(model, config, model_path)
    }

    fn build(model: Arc<OnnxModel>, config: AestheticScorerConfig, model_path: PathBuf) -> Self {
        let (w, h) = config.input_size;
        let preprocessor = ImagePreprocessor::new(w, h)
            .with_tensor_layout(TensorLayout::Nchw)
            .with_mean(config.mean)
            .with_std(config.std);
        Self {
            model,
            config,
            preprocessor,
            model_path,
        }
    }

    /// Path to the backing ONNX model.
    #[must_use]
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Shared handle to the underlying model.
    #[must_use]
    pub fn shared_model(&self) -> Arc<OnnxModel> {
        Arc::clone(&self.model)
    }

    /// Read-only view of the configuration.
    #[must_use]
    pub fn config(&self) -> &AestheticScorerConfig {
        &self.config
    }

    /// Expected input tensor shape (`[1, 3, H, W]`).
    #[must_use]
    pub fn expected_input_shape(&self) -> [usize; 4] {
        let (w, h) = self.config.input_size;
        [1, 3, h as usize, w as usize]
    }
}

impl TypedPipeline for AestheticScorer {
    type Input = AestheticImage;
    type Output = AestheticScore;

    fn run(&self, input: Self::Input) -> MlResult<Self::Output> {
        #[cfg(feature = "onnx")]
        {
            use oxionnx::Tensor;
            use std::collections::HashMap;

            let buf = self
                .preprocessor
                .process_u8_rgb(&input.pixels, input.width, input.height)?;
            let shape = self.preprocessor.batch_shape();
            let tensor = Tensor { data: buf, shape };

            let input_name = self
                .config
                .input_name
                .clone()
                .or_else(|| self.model.info().inputs.first().map(|s| s.name.clone()))
                .ok_or_else(|| MlError::invalid_input("model has no declared inputs"))?;
            let output_name = self
                .config
                .output_name
                .clone()
                .or_else(|| self.model.info().outputs.first().map(|s| s.name.clone()))
                .ok_or_else(|| MlError::invalid_input("model has no declared outputs"))?;

            let mut inputs: HashMap<&str, Tensor> = HashMap::with_capacity(1);
            inputs.insert(input_name.as_str(), tensor);
            let outputs = self.model.run(&inputs)?;
            let out = outputs.get(&output_name).ok_or_else(|| {
                MlError::postprocess(format!("output '{output_name}' missing from model run"))
            })?;

            if out.data.len() != 10 {
                return Err(MlError::postprocess(format!(
                    "aesthetic scorer expected output len 10, got {}",
                    out.data.len()
                )));
            }

            let probs = if self.config.apply_softmax {
                softmax(&out.data)
            } else {
                out.data.clone()
            };

            let mut dist = [0.0_f32; 10];
            for (slot, &p) in dist.iter_mut().zip(probs.iter()) {
                *slot = p;
            }
            Ok(AestheticScore::from_distribution(dist))
        }
        #[cfg(not(feature = "onnx"))]
        {
            let _ = input;
            Err(MlError::FeatureDisabled("onnx"))
        }
    }

    fn info(&self) -> PipelineInfo {
        PipelineInfo {
            id: "aesthetic-score/nima",
            name: "Aesthetic Scorer",
            task: PipelineTask::AestheticScoring,
            input_size: Some(self.config.input_size),
        }
    }
}

impl std::fmt::Debug for AestheticScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AestheticScorer")
            .field("model_path", &self.model_path)
            .field("input_size", &self.config.input_size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aesthetic_image_rejects_wrong_buffer() {
        let err = AestheticImage::new(vec![0u8; 10], 2, 2).expect_err("must fail");
        assert!(matches!(err, MlError::InvalidInput(_)));
    }

    #[test]
    fn aesthetic_image_accepts_correct_buffer() {
        let img = AestheticImage::new(vec![0u8; 3 * 3 * 3], 3, 3).expect("ok");
        assert_eq!(img.width, 3);
        assert_eq!(img.height, 3);
    }

    #[test]
    fn default_config_is_nima_224() {
        let cfg = AestheticScorerConfig::default();
        assert_eq!(cfg.input_size, (224, 224));
        assert!(cfg.apply_softmax);
        assert!((cfg.mean[0] - 0.485).abs() < 1e-6);
    }
}
