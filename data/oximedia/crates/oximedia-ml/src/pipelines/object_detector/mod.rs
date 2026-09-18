//! YOLOv8-compatible object detector pipeline.
//!
//! ## Contract
//!
//! | Stage         | Shape / value                                                                        |
//! |---------------|--------------------------------------------------------------------------------------|
//! | Input         | RGB u8 [`DetectorImage`], resized to [`ObjectDetectorConfig::input_size`]           |
//! | Preprocess    | NCHW f32 with ImageNet mean/std                                                      |
//! | ONNX input    | `[1, 3, H, W]` — default 640×640                                                     |
//! | ONNX output   | `[1, 4 + num_classes, A]` — channel-major, `A` anchors pre-emitted by the model      |
//! | Postprocess   | transpose → sigmoid on class logits → per-class threshold → [`crate::postprocess::nms`] |
//! | Output        | `Vec<Detection>` sorted by descending score                                          |
//!
//! Post-processing flow implemented by
//! [`crate::pipelines::decode_yolov8_output`]:
//!
//! 1. Transpose `[1, 84, A]` → `[A, 84]` (per-anchor rows).
//! 2. Split into 4-D box centres (`cx, cy, w, h`) and 80-D class
//!    logits.
//! 3. Apply sigmoid to class logits, take the per-anchor argmax.
//! 4. Threshold by [`ObjectDetectorConfig::conf_threshold`].
//! 5. Greedy NMS at [`ObjectDetectorConfig::iou_threshold`].
//!
//! ## Compatible models
//!
//! Any YOLOv8 ONNX export with the standard channel-major output layout
//! (Ultralytics default) — COCO-pretrained with 80 classes, or custom
//! fine-tunes with a different `num_classes`. Override
//! [`ObjectDetectorConfig::num_classes`] to match the training set.
//!
//! The returned [`Detection`]s carry corner-form [`BoundingBox`] values
//! in model-input coordinate space; scale them back to source pixels on
//! the caller side if needed.
//!
//! [`BoundingBox`]: crate::BoundingBox

mod decode;

pub use decode::{decode_yolov8_output, DecodeOptions};

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::device::DeviceType;
use crate::error::{MlError, MlResult};
use crate::model::OnnxModel;
use crate::pipeline::{PipelineInfo, PipelineTask, TypedPipeline};
use crate::preprocess::{ImagePreprocessor, TensorLayout};

use super::types::Detection;

/// Number of class logits in a COCO-pretrained YOLOv8.
pub const YOLOV8_NUM_CLASSES: usize = 80;

/// Total channel count per anchor (4 box centres + 80 class logits).
pub const YOLOV8_CHANNELS: usize = 4 + YOLOV8_NUM_CLASSES;

/// Input RGB image for the detector.
#[derive(Clone, Debug)]
pub struct DetectorImage {
    /// RGB u8 pixels, row-major, length = `width * height * 3`.
    pub pixels: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl DetectorImage {
    /// Create a new detector image, validating the buffer length.
    pub fn new(pixels: Vec<u8>, width: u32, height: u32) -> MlResult<Self> {
        let expected = (width as usize) * (height as usize) * 3;
        if pixels.len() != expected {
            return Err(MlError::invalid_input(format!(
                "detector image: expected {expected} bytes for {width}x{height} RGB, got {}",
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

/// Configuration for [`ObjectDetector`].
#[derive(Clone, Debug)]
pub struct ObjectDetectorConfig {
    /// Model input spatial size. YOLOv8 default is 640×640.
    pub input_size: (u32, u32),
    /// Per-channel mean (ImageNet defaults).
    pub mean: [f32; 3],
    /// Per-channel std (ImageNet defaults).
    pub std: [f32; 3],
    /// Expected number of classes. Defaults to 80 (COCO).
    pub num_classes: usize,
    /// Minimum post-sigmoid confidence to keep a detection.
    pub conf_threshold: f32,
    /// IoU threshold used by NMS.
    pub iou_threshold: f32,
    /// Model input tensor name (defaults to the model's first input).
    pub input_name: Option<String>,
    /// Model output tensor name (defaults to the model's first output).
    pub output_name: Option<String>,
    /// Optional class-name table (indexed by class id).
    pub class_names: Option<Vec<String>>,
}

impl Default for ObjectDetectorConfig {
    fn default() -> Self {
        Self {
            input_size: (640, 640),
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            num_classes: YOLOV8_NUM_CLASSES,
            conf_threshold: 0.25,
            iou_threshold: 0.45,
            input_name: None,
            output_name: None,
            class_names: None,
        }
    }
}

/// YOLOv8-compatible object detector.
///
/// Construct via [`Self::load`] or [`Self::load_with_config`]. Implements
/// [`TypedPipeline`] with `Input = DetectorImage`, `Output =
/// Vec<Detection>`.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(all(feature = "onnx", feature = "object-detector"))]
/// # fn demo() -> oximedia_ml::MlResult<()> {
/// use oximedia_ml::pipelines::{DetectorImage, ObjectDetector};
/// use oximedia_ml::{DeviceType, TypedPipeline};
///
/// let detector = ObjectDetector::load("yolov8n.onnx", DeviceType::auto())?;
/// let frame = DetectorImage::new(vec![0_u8; 640 * 640 * 3], 640, 640)?;
/// for det in detector.run(frame)? {
///     println!("class {} @ {:.3}", det.class_id, det.score);
/// }
/// # Ok(())
/// # }
/// ```
pub struct ObjectDetector {
    model: Arc<OnnxModel>,
    config: ObjectDetectorConfig,
    preprocessor: ImagePreprocessor,
    model_path: PathBuf,
}

impl ObjectDetector {
    /// Load a detector from an ONNX model path with default config.
    ///
    /// # Errors
    ///
    /// Propagates any error from
    /// [`OnnxModel::load`][crate::OnnxModel::load].
    pub fn load(path: impl AsRef<Path>, device: DeviceType) -> MlResult<Self> {
        Self::load_with_config(path, device, ObjectDetectorConfig::default())
    }

    /// Load a detector with a custom configuration.
    ///
    /// # Errors
    ///
    /// Propagates any error from
    /// [`OnnxModel::load`][crate::OnnxModel::load].
    pub fn load_with_config(
        path: impl AsRef<Path>,
        device: DeviceType,
        config: ObjectDetectorConfig,
    ) -> MlResult<Self> {
        let model_path = path.as_ref().to_path_buf();
        let model = Arc::new(OnnxModel::load(&model_path, device)?);
        Ok(Self::build(model, config, model_path))
    }

    /// Build from a pre-loaded shared model.
    #[must_use]
    pub fn from_shared(
        model: Arc<OnnxModel>,
        config: ObjectDetectorConfig,
        model_path: PathBuf,
    ) -> Self {
        Self::build(model, config, model_path)
    }

    /// Override the configured input size. Useful for YOLOv8 exports
    /// at non-default resolutions (320×320, 1280×1280, …).
    #[must_use]
    pub fn with_input_size(mut self, width: u32, height: u32) -> Self {
        self.config.input_size = (width, height);
        self.preprocessor = build_preprocessor(&self.config);
        self
    }

    fn build(model: Arc<OnnxModel>, config: ObjectDetectorConfig, model_path: PathBuf) -> Self {
        let preprocessor = build_preprocessor(&config);
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
    pub fn config(&self) -> &ObjectDetectorConfig {
        &self.config
    }

    /// Expected input tensor shape (`[1, 3, H, W]`).
    #[must_use]
    pub fn expected_input_shape(&self) -> [usize; 4] {
        let (w, h) = self.config.input_size;
        [1, 3, h as usize, w as usize]
    }
}

fn build_preprocessor(config: &ObjectDetectorConfig) -> ImagePreprocessor {
    let (w, h) = config.input_size;
    ImagePreprocessor::new(w, h)
        .with_tensor_layout(TensorLayout::Nchw)
        .with_mean(config.mean)
        .with_std(config.std)
}

impl TypedPipeline for ObjectDetector {
    type Input = DetectorImage;
    type Output = Vec<Detection>;

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

            let opts = DecodeOptions {
                num_classes: self.config.num_classes,
                conf_threshold: self.config.conf_threshold,
                iou_threshold: self.config.iou_threshold,
            };
            decode_yolov8_output(&out.data, &out.shape, &opts)
        }
        #[cfg(not(feature = "onnx"))]
        {
            let _ = input;
            Err(MlError::FeatureDisabled("onnx"))
        }
    }

    fn info(&self) -> PipelineInfo {
        PipelineInfo {
            id: "object-detector/yolov8",
            name: "Object Detector",
            task: PipelineTask::Detection,
            input_size: Some(self.config.input_size),
        }
    }
}

impl std::fmt::Debug for ObjectDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectDetector")
            .field("model_path", &self.model_path)
            .field("input_size", &self.config.input_size)
            .field("num_classes", &self.config.num_classes)
            .field("conf_threshold", &self.config.conf_threshold)
            .field("iou_threshold", &self.config.iou_threshold)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_image_rejects_wrong_buffer() {
        let err = DetectorImage::new(vec![0u8; 10], 2, 2).expect_err("must fail");
        assert!(matches!(err, MlError::InvalidInput(_)));
    }

    #[test]
    fn default_config_is_yolov8_640() {
        let cfg = ObjectDetectorConfig::default();
        assert_eq!(cfg.input_size, (640, 640));
        assert_eq!(cfg.num_classes, YOLOV8_NUM_CLASSES);
        assert!((cfg.conf_threshold - 0.25).abs() < 1e-6);
        assert!((cfg.iou_threshold - 0.45).abs() < 1e-6);
    }

    #[test]
    fn yolov8_channel_count_consistent() {
        assert_eq!(YOLOV8_CHANNELS, 84);
    }
}
