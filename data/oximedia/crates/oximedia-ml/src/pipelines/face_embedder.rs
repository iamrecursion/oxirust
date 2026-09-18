//! ArcFace / FaceNet-compatible face embedder pipeline.
//!
//! Consumes a single RGB face crop (typically 112×112 after upstream
//! alignment) and produces a unit-norm [`FaceEmbedding`] that can be
//! compared to other embeddings via cosine similarity.
//!
//! ## Contract
//!
//! | Stage         | Shape / value                                                                |
//! |---------------|------------------------------------------------------------------------------|
//! | Input         | Aligned RGB u8 [`FaceImage`], resized to `input_size` (default 112×112)     |
//! | Preprocess    | NCHW f32 with ImageNet mean/std                                              |
//! | ONNX input    | `[1, 3, 112, 112]`                                                           |
//! | ONNX output   | `[1, embedding_dim]` (default 512)                                           |
//! | Postprocess   | [`FaceEmbedding::from_raw`] (applies [`crate::postprocess::l2_normalize`])  |
//! | Output        | Unit-norm [`FaceEmbedding`]                                                  |
//!
//! Compare two embeddings via
//! [`FaceEmbedding::cosine_similarity`].
//!
//! ## Compatible models
//!
//! ArcFace / CosFace / FaceNet / InsightFace exports — any model whose
//! final layer is a plain L2-agnostic embedding vector. If you need a
//! non-512 embedding dimension, override
//! [`FaceEmbedderConfig::embedding_dim`] to match the output tensor the
//! model emits.
//!
//! When the `onnx` feature is disabled, the pipeline returns
//! [`MlError::FeatureDisabled`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::device::DeviceType;
use crate::error::{MlError, MlResult};
use crate::model::OnnxModel;
use crate::pipeline::{PipelineInfo, PipelineTask, TypedPipeline};
use crate::preprocess::{ImagePreprocessor, TensorLayout};

use super::types::FaceEmbedding;

/// Input aligned face crop.
#[derive(Clone, Debug)]
pub struct FaceImage {
    /// RGB u8 pixels, row-major, length = `width * height * 3`.
    pub pixels: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl FaceImage {
    /// Create a new face image, validating the buffer length.
    pub fn new(pixels: Vec<u8>, width: u32, height: u32) -> MlResult<Self> {
        let expected = (width as usize) * (height as usize) * 3;
        if pixels.len() != expected {
            return Err(MlError::invalid_input(format!(
                "face image: expected {expected} bytes for {width}x{height} RGB, got {}",
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

/// Configuration for [`FaceEmbedder`].
#[derive(Clone, Debug)]
pub struct FaceEmbedderConfig {
    /// Input size expected by the model. Defaults to 112×112 (ArcFace).
    pub input_size: (u32, u32),
    /// Per-channel mean (ImageNet defaults).
    pub mean: [f32; 3],
    /// Per-channel std (ImageNet defaults).
    pub std: [f32; 3],
    /// Expected embedding dimensionality. Defaults to 512.
    pub embedding_dim: usize,
    /// Model input tensor name (defaults to the model's first input).
    pub input_name: Option<String>,
    /// Model output tensor name (defaults to the model's first output).
    pub output_name: Option<String>,
}

impl Default for FaceEmbedderConfig {
    fn default() -> Self {
        Self {
            input_size: (112, 112),
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            embedding_dim: 512,
            input_name: None,
            output_name: None,
        }
    }
}

/// Face embedder pipeline.
///
/// Construct via [`Self::load`] or [`Self::load_with_config`]. Implements
/// [`TypedPipeline`] with `Input = FaceImage`,
/// `Output = FaceEmbedding`.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(all(feature = "onnx", feature = "face-embedder"))]
/// # fn demo() -> oximedia_ml::MlResult<()> {
/// use oximedia_ml::pipelines::{FaceEmbedder, FaceImage};
/// use oximedia_ml::{DeviceType, TypedPipeline};
///
/// let embedder = FaceEmbedder::load("arcface.onnx", DeviceType::auto())?;
/// let crop = FaceImage::new(vec![0_u8; 112 * 112 * 3], 112, 112)?;
/// let embedding = embedder.run(crop)?;
/// assert_eq!(embedding.len(), 512);
/// # Ok(())
/// # }
/// ```
pub struct FaceEmbedder {
    model: Arc<OnnxModel>,
    config: FaceEmbedderConfig,
    preprocessor: ImagePreprocessor,
    model_path: PathBuf,
}

impl FaceEmbedder {
    /// Load an embedder from an ONNX model path with default config.
    ///
    /// # Errors
    ///
    /// Propagates any error from
    /// [`OnnxModel::load`][crate::OnnxModel::load].
    pub fn load(path: impl AsRef<Path>, device: DeviceType) -> MlResult<Self> {
        Self::load_with_config(path, device, FaceEmbedderConfig::default())
    }

    /// Load an embedder with a custom configuration.
    ///
    /// # Errors
    ///
    /// Propagates any error from
    /// [`OnnxModel::load`][crate::OnnxModel::load].
    pub fn load_with_config(
        path: impl AsRef<Path>,
        device: DeviceType,
        config: FaceEmbedderConfig,
    ) -> MlResult<Self> {
        let model_path = path.as_ref().to_path_buf();
        let model = Arc::new(OnnxModel::load(&model_path, device)?);
        Ok(Self::build(model, config, model_path))
    }

    /// Build directly from a pre-loaded model.
    #[must_use]
    pub fn from_shared(
        model: Arc<OnnxModel>,
        config: FaceEmbedderConfig,
        model_path: PathBuf,
    ) -> Self {
        Self::build(model, config, model_path)
    }

    fn build(model: Arc<OnnxModel>, config: FaceEmbedderConfig, model_path: PathBuf) -> Self {
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
    pub fn config(&self) -> &FaceEmbedderConfig {
        &self.config
    }

    /// Expected input tensor shape (`[1, 3, H, W]`).
    #[must_use]
    pub fn expected_input_shape(&self) -> [usize; 4] {
        let (w, h) = self.config.input_size;
        [1, 3, h as usize, w as usize]
    }
}

impl TypedPipeline for FaceEmbedder {
    type Input = FaceImage;
    type Output = FaceEmbedding;

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

            if out.data.len() != self.config.embedding_dim {
                return Err(MlError::postprocess(format!(
                    "face embedder expected output dim {}, got {}",
                    self.config.embedding_dim,
                    out.data.len()
                )));
            }

            Ok(FaceEmbedding::from_raw(out.data.clone()))
        }
        #[cfg(not(feature = "onnx"))]
        {
            let _ = input;
            Err(MlError::FeatureDisabled("onnx"))
        }
    }

    fn info(&self) -> PipelineInfo {
        PipelineInfo {
            id: "face-embedder/arcface",
            name: "Face Embedder",
            task: PipelineTask::FaceEmbedding,
            input_size: Some(self.config.input_size),
        }
    }
}

impl std::fmt::Debug for FaceEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FaceEmbedder")
            .field("model_path", &self.model_path)
            .field("input_size", &self.config.input_size)
            .field("embedding_dim", &self.config.embedding_dim)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_image_rejects_wrong_buffer() {
        let err = FaceImage::new(vec![0u8; 10], 2, 2).expect_err("must fail");
        assert!(matches!(err, MlError::InvalidInput(_)));
    }

    #[test]
    fn face_image_accepts_correct_buffer() {
        let img = FaceImage::new(vec![0u8; 4 * 4 * 3], 4, 4).expect("ok");
        assert_eq!(img.width, 4);
    }

    #[test]
    fn default_config_is_arcface_112() {
        let cfg = FaceEmbedderConfig::default();
        assert_eq!(cfg.input_size, (112, 112));
        assert_eq!(cfg.embedding_dim, 512);
    }
}
