//! Places365-style scene classifier pipeline.
//!
//! Takes a raw RGB u8 image and returns a ranked list of `(label,
//! score)` predictions. When the `onnx` feature is disabled this
//! pipeline returns [`crate::error::MlError::FeatureDisabled`] from
//! every `run` call so callers can gracefully fall back to heuristic
//! scoring.
//!
//! ## Contract
//!
//! | Stage         | Shape / value                                                                |
//! |---------------|------------------------------------------------------------------------------|
//! | Input image   | RGB u8 buffer, resized to 224×224 by default                                 |
//! | Preprocess    | NCHW f32 with ImageNet mean/std, `batch_shape() == [1, 3, 224, 224]`        |
//! | ONNX input    | Single-tensor; name defaults to `model.info().inputs[0].name`                |
//! | ONNX output   | Logits of shape `[1, num_classes]` (Places365 has 365)                       |
//! | Postprocess   | [`crate::postprocess::softmax`] then [`crate::postprocess::top_k`] |
//! | Output        | `Vec<SceneClassification>`, sorted by descending score                       |
//!
//! ## Compatible models
//!
//! Any image classifier exported to ONNX that accepts ImageNet-style
//! NCHW inputs and returns logits along the last axis — e.g. the
//! Places365 ResNet-18/50 variants, or a fine-tuned generic ResNet /
//! EfficientNet / ConvNeXt. Provide a `labels` vector in
//! [`SceneClassifierConfig`] to populate [`SceneClassification::label`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::device::DeviceType;
use crate::error::{MlError, MlResult};
use crate::model::OnnxModel;
use crate::pipeline::{PipelineInfo, PipelineTask, TypedPipeline};
#[cfg(feature = "onnx")]
use crate::postprocess::{softmax, top_k};
use crate::preprocess::{ImagePreprocessor, TensorLayout};

/// Input image for the scene classifier.
#[derive(Clone, Debug)]
pub struct SceneImage {
    /// RGB u8 pixels, row-major, length = `width * height * 3`.
    pub pixels: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl SceneImage {
    /// Create a new scene image, validating buffer length.
    pub fn new(pixels: Vec<u8>, width: u32, height: u32) -> MlResult<Self> {
        let expected = (width as usize) * (height as usize) * 3;
        if pixels.len() != expected {
            return Err(MlError::invalid_input(format!(
                "scene image: expected {expected} bytes for {width}x{height} RGB, got {}",
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

/// One classification prediction.
#[derive(Clone, Debug)]
pub struct SceneClassification {
    /// Integer class index produced by the model.
    pub class_index: usize,
    /// Human-readable label, if the classifier knows one.
    pub label: Option<String>,
    /// Softmax score in `[0, 1]`.
    pub score: f32,
}

/// Configuration knobs for [`SceneClassifier`].
#[derive(Clone, Debug)]
pub struct SceneClassifierConfig {
    /// Input size expected by the model. Defaults to 224×224.
    pub input_size: (u32, u32),
    /// Per-channel mean (ImageNet defaults).
    pub mean: [f32; 3],
    /// Per-channel std (ImageNet defaults).
    pub std: [f32; 3],
    /// Optional human-readable labels, indexed by class.
    pub labels: Option<Vec<String>>,
    /// Number of top predictions to return per call.
    pub top_k: usize,
    /// Model input tensor name (defaults to the model's first input).
    pub input_name: Option<String>,
    /// Model output tensor name (defaults to the model's first output).
    pub output_name: Option<String>,
}

impl Default for SceneClassifierConfig {
    fn default() -> Self {
        Self {
            input_size: (224, 224),
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            labels: None,
            top_k: 5,
            input_name: None,
            output_name: None,
        }
    }
}

/// Scene classifier pipeline.
///
/// Construct via [`Self::load`] or [`Self::load_with_config`]. Implements
/// [`TypedPipeline`] with `Input = SceneImage`, `Output =
/// Vec<SceneClassification>`.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(all(feature = "onnx", feature = "scene-classifier"))]
/// # fn demo() -> oximedia_ml::MlResult<()> {
/// use oximedia_ml::pipelines::{SceneClassifier, SceneImage};
/// use oximedia_ml::{DeviceType, TypedPipeline};
///
/// let classifier = SceneClassifier::load("places365.onnx", DeviceType::auto())?;
/// let frame = SceneImage::new(vec![0u8; 224 * 224 * 3], 224, 224)?;
/// let top = classifier.run(frame)?;
/// assert!(top.len() <= classifier.top_k());
/// # Ok(())
/// # }
/// ```
pub struct SceneClassifier {
    model: Arc<OnnxModel>,
    config: SceneClassifierConfig,
    preprocessor: ImagePreprocessor,
    model_path: PathBuf,
}

impl SceneClassifier {
    /// Load a classifier from an ONNX model path with default config.
    ///
    /// # Errors
    ///
    /// Propagates any error from
    /// [`OnnxModel::load`][crate::OnnxModel::load] (feature disabled,
    /// device unavailable, parse error).
    pub fn load(path: impl AsRef<Path>, device: DeviceType) -> MlResult<Self> {
        Self::load_with_config(path, device, SceneClassifierConfig::default())
    }

    /// Load a classifier with a custom configuration.
    ///
    /// # Errors
    ///
    /// Propagates any error from
    /// [`OnnxModel::load`][crate::OnnxModel::load].
    pub fn load_with_config(
        path: impl AsRef<Path>,
        device: DeviceType,
        config: SceneClassifierConfig,
    ) -> MlResult<Self> {
        let model_path = path.as_ref().to_path_buf();
        let model = Arc::new(OnnxModel::load(&model_path, device)?);
        Ok(Self::build(model, config, model_path))
    }

    /// Build directly from a pre-loaded model (useful for the [`crate::ModelCache`]).
    #[must_use]
    pub fn from_shared(
        model: Arc<OnnxModel>,
        config: SceneClassifierConfig,
        model_path: PathBuf,
    ) -> Self {
        Self::build(model, config, model_path)
    }

    fn build(model: Arc<OnnxModel>, config: SceneClassifierConfig, model_path: PathBuf) -> Self {
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

    /// Return the backing model path.
    #[must_use]
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Return the configured top-k.
    #[must_use]
    pub fn top_k(&self) -> usize {
        self.config.top_k
    }

    /// Clone the shared model handle.
    #[must_use]
    pub fn shared_model(&self) -> Arc<OnnxModel> {
        Arc::clone(&self.model)
    }

    /// Read-only view of the configuration.
    #[must_use]
    pub fn config(&self) -> &SceneClassifierConfig {
        &self.config
    }

    /// Read-only view of the preprocessor.
    #[must_use]
    pub fn preprocessor(&self) -> &ImagePreprocessor {
        &self.preprocessor
    }
}

impl TypedPipeline for SceneClassifier {
    type Input = SceneImage;
    type Output = Vec<SceneClassification>;

    fn run(&self, input: Self::Input) -> MlResult<Self::Output> {
        #[cfg(feature = "onnx")]
        {
            use oxionnx::Tensor;
            use std::collections::HashMap;

            let input_buf =
                self.preprocessor
                    .process_u8_rgb(&input.pixels, input.width, input.height)?;
            let shape = self.preprocessor.batch_shape();
            let tensor = Tensor {
                data: input_buf,
                shape,
            };

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

            let probs = softmax(&out.data);
            let top = top_k(&probs, self.config.top_k)?;
            Ok(top
                .into_iter()
                .map(|(idx, score)| SceneClassification {
                    class_index: idx,
                    label: self
                        .config
                        .labels
                        .as_ref()
                        .and_then(|v| v.get(idx).cloned()),
                    score,
                })
                .collect())
        }
        #[cfg(not(feature = "onnx"))]
        {
            let _ = input;
            Err(MlError::FeatureDisabled("onnx"))
        }
    }

    fn info(&self) -> PipelineInfo {
        PipelineInfo {
            id: "scene-classifier/places365",
            name: "Scene Classifier",
            task: PipelineTask::SceneClassification,
            input_size: Some(self.config.input_size),
        }
    }
}

impl std::fmt::Debug for SceneClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SceneClassifier")
            .field("model_path", &self.model_path)
            .field("top_k", &self.config.top_k)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_image_rejects_wrong_buffer() {
        let err = SceneImage::new(vec![0u8; 10], 2, 2).expect_err("must fail");
        assert!(matches!(err, MlError::InvalidInput(_)));
    }

    #[test]
    fn scene_image_accepts_correct_buffer() {
        let img = SceneImage::new(vec![0u8; 2 * 2 * 3], 2, 2).expect("ok");
        assert_eq!(img.width, 2);
        assert_eq!(img.height, 2);
    }

    #[test]
    fn default_config_uses_imagenet() {
        let cfg = SceneClassifierConfig::default();
        assert_eq!(cfg.input_size, (224, 224));
        assert_eq!(cfg.top_k, 5);
        assert!((cfg.mean[0] - 0.485).abs() < 1e-6);
    }
}
