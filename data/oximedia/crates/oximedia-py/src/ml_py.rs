//! `oximedia.ml` submodule — Python bindings for the typed-pipeline ML stack.
//!
//! Wraps [`oximedia_ml`] behind PyO3 classes so Python callers can drive
//! the scene classifier, shot-boundary detector, aesthetic scorer, object
//! detector, and face embedder without crossing back into Rust themselves.
//!
//! Input images are accepted as numpy `uint8` arrays of shape `(H, W, 3)`
//! (or `(N, H, W, 3)` for the shot-boundary sliding window). Outputs are
//! plain Python objects with the same structure as the Rust value types.
//!
//! This module is gated behind the `ml` feature; callers that do not opt
//! in see no additional symbols on the Python side.

use std::path::PathBuf;
use std::sync::Arc;

use numpy::{IntoPyArray, PyReadonlyArray3, PyReadonlyArray4, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use oximedia_ml::pipelines::{
    AestheticImage, AestheticScore, AestheticScorer, Detection, DetectorImage, FaceEmbedder,
    FaceEmbedding, FaceImage, ObjectDetector, SceneClassification, SceneClassifier, SceneImage,
    ShotBoundary, ShotBoundaryConfig, ShotBoundaryDetector, ShotFrame,
};
use oximedia_ml::{
    DeviceCapabilities, DeviceType, MlError, MlResult, ModelEntry, ModelInfo, ModelZoo, OnnxModel,
    PipelineTask, TensorDType, TensorSpec, TypedPipeline,
};

// ---------------------------------------------------------------------------
// Error conversion
// ---------------------------------------------------------------------------

fn ml_err(err: MlError) -> PyErr {
    PyRuntimeError::new_err(format!("{err}"))
}

fn map_ml<T>(res: MlResult<T>) -> PyResult<T> {
    res.map_err(ml_err)
}

// ---------------------------------------------------------------------------
// Device type
// ---------------------------------------------------------------------------

/// Execution backend for ML inference.
///
/// Mirrors [`oximedia_ml::DeviceType`]. Use :meth:`auto` to pick the best
/// backend available in this build, or call the zero-argument constructors
/// on the individual variants (`MlDeviceType.cpu()`, etc.) to force one.
#[pyclass(name = "MlDeviceType")]
#[derive(Clone, Copy, Debug)]
pub struct PyMlDeviceType {
    inner: DeviceType,
}

impl PyMlDeviceType {
    fn new(inner: DeviceType) -> Self {
        Self { inner }
    }

    fn inner(&self) -> DeviceType {
        self.inner
    }
}

#[pymethods]
impl PyMlDeviceType {
    /// Pick the strongest available backend.
    #[staticmethod]
    fn auto() -> Self {
        Self::new(DeviceType::auto())
    }

    /// Force the pure-Rust CPU path.
    #[staticmethod]
    fn cpu() -> Self {
        Self::new(DeviceType::Cpu)
    }

    /// NVIDIA CUDA backend (feature-gated; fails at load time if unavailable).
    #[staticmethod]
    fn cuda() -> Self {
        Self::new(DeviceType::Cuda)
    }

    /// WebGPU/wgpu backend.
    #[staticmethod]
    fn webgpu() -> Self {
        Self::new(DeviceType::WebGpu)
    }

    /// Microsoft DirectML backend (Windows-only at runtime).
    #[staticmethod]
    fn directml() -> Self {
        Self::new(DeviceType::DirectMl)
    }

    /// Apple CoreML backend (reserved; never currently available).
    #[staticmethod]
    fn coreml() -> Self {
        Self::new(DeviceType::CoreMl)
    }

    /// Resolve a device by string name (`"cpu"`, `"cuda"`, `"webgpu"`,
    /// `"directml"`, `"coreml"`, or `"auto"`).
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        let normalised = name.trim().to_ascii_lowercase();
        let inner = match normalised.as_str() {
            "auto" | "" => DeviceType::auto(),
            "cpu" => DeviceType::Cpu,
            "cuda" | "gpu" => DeviceType::Cuda,
            "webgpu" | "wgpu" => DeviceType::WebGpu,
            "directml" | "dml" => DeviceType::DirectMl,
            "coreml" => DeviceType::CoreMl,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown device name: {other}"
                )));
            }
        };
        Ok(Self::new(inner))
    }

    /// Canonical short name (`"cpu"`, `"cuda"`, ...).
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    /// Human-facing label (`"CPU"`, `"CUDA"`, ...).
    #[getter]
    fn display_name(&self) -> &'static str {
        self.inner.display_name()
    }

    /// Whether this device is usable in the current build and runtime.
    fn is_available(&self) -> bool {
        self.inner.is_available()
    }

    /// Every device variant whose backend is currently usable. CPU is
    /// always present.
    #[staticmethod]
    fn list_available() -> Vec<Self> {
        DeviceType::list_available()
            .into_iter()
            .map(Self::new)
            .collect()
    }

    /// Probe this backend and return a rich capability record.
    fn capabilities(&self) -> PyMlDeviceCapabilities {
        PyMlDeviceCapabilities::from(DeviceCapabilities::probe(self.inner))
    }

    fn __repr__(&self) -> String {
        format!(
            "MlDeviceType({}, available={})",
            self.inner.name(),
            self.inner.is_available()
        )
    }

    fn __str__(&self) -> String {
        self.inner.name().to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Rich capability description for a single :class:`MlDeviceType`.
#[pyclass(name = "MlDeviceCapabilities")]
#[derive(Clone, Debug)]
pub struct PyMlDeviceCapabilities {
    /// Which device this record describes.
    #[pyo3(get)]
    pub device: PyMlDeviceType,
    /// Whether the device is currently available for inference.
    #[pyo3(get)]
    pub is_available: bool,
    /// Human-facing device name.
    #[pyo3(get)]
    pub device_name: String,
    /// Total device memory in bytes, if known.
    #[pyo3(get)]
    pub memory_total_bytes: Option<u64>,
    /// Free device memory in bytes, if known.
    #[pyo3(get)]
    pub memory_free_bytes: Option<u64>,
    /// Compute capability string (e.g. `"8.6"` for Ampere).
    #[pyo3(get)]
    pub compute_capability: Option<String>,
    /// FP16 (half precision) supported.
    #[pyo3(get)]
    pub supports_fp16: bool,
    /// BF16 (bfloat16) supported.
    #[pyo3(get)]
    pub supports_bf16: bool,
    /// INT8 quantised inference supported.
    #[pyo3(get)]
    pub supports_int8: bool,
}

impl From<DeviceCapabilities> for PyMlDeviceCapabilities {
    fn from(caps: DeviceCapabilities) -> Self {
        Self {
            device: PyMlDeviceType::new(caps.device_type),
            is_available: caps.is_available,
            device_name: caps.device_name,
            memory_total_bytes: caps.memory_total_bytes,
            memory_free_bytes: caps.memory_free_bytes,
            compute_capability: caps.compute_capability,
            supports_fp16: caps.supports_fp16,
            supports_bf16: caps.supports_bf16,
            supports_int8: caps.supports_int8,
        }
    }
}

#[pymethods]
impl PyMlDeviceCapabilities {
    /// Probe every compiled-in backend and return a list of capability records.
    #[staticmethod]
    fn probe_all() -> Vec<Self> {
        DeviceCapabilities::probe_all()
            .into_iter()
            .map(Self::from)
            .collect()
    }

    /// Return the capability record for the best currently-available device.
    #[staticmethod]
    fn best_available() -> Self {
        DeviceCapabilities::best_available().into()
    }

    fn __repr__(&self) -> String {
        format!(
            "MlDeviceCapabilities({}, available={})",
            self.device_name, self.is_available
        )
    }
}

// ---------------------------------------------------------------------------
// TensorDType / TensorSpec / ModelInfo / OnnxModel
// ---------------------------------------------------------------------------

/// Canonical scalar dtype advertised by a model input or output.
#[pyclass(name = "MlTensorDType")]
#[derive(Clone, Copy, Debug)]
pub struct PyMlTensorDType {
    inner: TensorDType,
}

impl From<TensorDType> for PyMlTensorDType {
    fn from(inner: TensorDType) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyMlTensorDType {
    /// Short canonical name matching ONNX nomenclature.
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        format!("MlTensorDType({})", self.inner.name())
    }

    fn __str__(&self) -> String {
        self.inner.name().to_string()
    }
}

/// Describes a single model input or output tensor.
#[pyclass(name = "MlTensorSpec")]
#[derive(Clone, Debug)]
pub struct PyMlTensorSpec {
    /// Tensor name as declared in the ONNX graph.
    #[pyo3(get)]
    pub name: String,
    /// Scalar dtype of the tensor.
    #[pyo3(get)]
    pub dtype: PyMlTensorDType,
    /// Declared shape — dynamic dimensions are `None`.
    #[pyo3(get)]
    pub shape: Vec<Option<i64>>,
    /// Number of dynamic (None) dimensions.
    #[pyo3(get)]
    pub dynamic_rank: usize,
}

impl From<&TensorSpec> for PyMlTensorSpec {
    fn from(spec: &TensorSpec) -> Self {
        Self {
            name: spec.name.clone(),
            dtype: spec.dtype.into(),
            shape: spec.shape.clone(),
            dynamic_rank: spec.dynamic_rank(),
        }
    }
}

#[pymethods]
impl PyMlTensorSpec {
    fn __repr__(&self) -> String {
        format!(
            "MlTensorSpec(name={:?}, dtype={}, shape={:?})",
            self.name,
            self.dtype.inner.name(),
            self.shape
        )
    }
}

/// Static metadata describing a loaded ONNX model.
#[pyclass(name = "MlModelInfo")]
#[derive(Clone, Debug)]
pub struct PyMlModelInfo {
    /// On-disk (or virtual) path the model was loaded from.
    #[pyo3(get)]
    pub path: String,
    /// Model input tensor specifications.
    #[pyo3(get)]
    pub inputs: Vec<PyMlTensorSpec>,
    /// Model output tensor specifications.
    #[pyo3(get)]
    pub outputs: Vec<PyMlTensorSpec>,
    /// Producer name as declared in the ONNX file.
    #[pyo3(get)]
    pub producer: Option<String>,
    /// Opset version, if reported by the backend.
    #[pyo3(get)]
    pub opset_version: Option<i64>,
}

impl From<&ModelInfo> for PyMlModelInfo {
    fn from(info: &ModelInfo) -> Self {
        Self {
            path: info.path.to_string_lossy().into_owned(),
            inputs: info.inputs.iter().map(PyMlTensorSpec::from).collect(),
            outputs: info.outputs.iter().map(PyMlTensorSpec::from).collect(),
            producer: info.producer.clone(),
            opset_version: info.opset_version,
        }
    }
}

#[pymethods]
impl PyMlModelInfo {
    fn __repr__(&self) -> String {
        format!(
            "MlModelInfo(path={:?}, inputs={}, outputs={})",
            self.path,
            self.inputs.len(),
            self.outputs.len()
        )
    }
}

/// Pure-Rust ONNX model handle.
#[pyclass(name = "OnnxModel")]
#[derive(Clone)]
pub struct PyOnnxModel {
    model: Arc<OnnxModel>,
}

impl PyOnnxModel {
    fn from_arc(model: Arc<OnnxModel>) -> Self {
        Self { model }
    }
}

#[pymethods]
impl PyOnnxModel {
    /// Load an ONNX model from disk onto the given device.
    #[staticmethod]
    #[pyo3(signature = (path, device=None))]
    fn load(py: Python<'_>, path: &str, device: Option<PyMlDeviceType>) -> PyResult<Self> {
        let dev = device.map(|d| d.inner()).unwrap_or_else(DeviceType::auto);
        let path_owned: String = path.to_string();
        // Loading parses & validates the model graph; release the GIL for it.
        let model = map_ml(py.detach(move || OnnxModel::load(&path_owned, dev)))?;
        Ok(Self::from_arc(Arc::new(model)))
    }

    /// Load an ONNX model from an in-memory byte buffer.
    #[staticmethod]
    #[pyo3(signature = (data, device=None, virtual_path=None))]
    fn load_from_bytes(
        py: Python<'_>,
        data: &[u8],
        device: Option<PyMlDeviceType>,
        virtual_path: Option<&str>,
    ) -> PyResult<Self> {
        let dev = device.map(|d| d.inner()).unwrap_or_else(DeviceType::auto);
        let vp = PathBuf::from(virtual_path.unwrap_or("<bytes>"));
        let buf: Vec<u8> = data.to_vec();
        let model = map_ml(py.detach(move || OnnxModel::load_from_bytes(&buf, dev, vp)))?;
        Ok(Self::from_arc(Arc::new(model)))
    }

    /// Return the loaded model metadata.
    fn info(&self) -> PyMlModelInfo {
        PyMlModelInfo::from(self.model.info())
    }

    /// Return the device this model was loaded onto.
    fn device(&self) -> PyMlDeviceType {
        PyMlDeviceType::new(self.model.device())
    }

    fn __repr__(&self) -> String {
        let info = self.model.info();
        format!(
            "OnnxModel(device={}, inputs={}, outputs={})",
            self.model.device().name(),
            info.inputs.len(),
            info.outputs.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Model zoo
// ---------------------------------------------------------------------------

/// Metadata entry describing a model compatible with a pipeline.
#[pyclass(name = "MlModelEntry")]
#[derive(Clone, Debug)]
pub struct PyMlModelEntry {
    /// Stable unique ID, e.g. `"places365/resnet18"`.
    #[pyo3(get)]
    pub id: &'static str,
    /// Human-readable name.
    #[pyo3(get)]
    pub name: &'static str,
    /// Pipeline task this model is intended for.
    #[pyo3(get)]
    pub task: &'static str,
    /// Expected `(width, height)` of the image input, if applicable.
    #[pyo3(get)]
    pub input_size: Option<(u32, u32)>,
    /// Number of output classes, if applicable.
    #[pyo3(get)]
    pub num_classes: Option<usize>,
    /// Short notes / citation for the user.
    #[pyo3(get)]
    pub notes: &'static str,
}

fn task_name(task: PipelineTask) -> &'static str {
    match task {
        PipelineTask::SceneClassification => "scene-classification",
        PipelineTask::ShotBoundary => "shot-boundary",
        PipelineTask::Detection => "detection",
        PipelineTask::Segmentation => "segmentation",
        PipelineTask::AestheticScoring => "aesthetic-scoring",
        PipelineTask::FaceEmbedding => "face-embedding",
        PipelineTask::Custom => "custom",
    }
}

impl From<&ModelEntry> for PyMlModelEntry {
    fn from(entry: &ModelEntry) -> Self {
        Self {
            id: entry.id,
            name: entry.name,
            task: task_name(entry.task),
            input_size: entry.input_size,
            num_classes: entry.num_classes,
            notes: entry.notes,
        }
    }
}

#[pymethods]
impl PyMlModelEntry {
    fn __repr__(&self) -> String {
        format!(
            "MlModelEntry(id={:?}, task={:?}, input_size={:?})",
            self.id, self.task, self.input_size
        )
    }
}

/// In-memory registry of known models.
#[pyclass(name = "MlModelZoo")]
#[derive(Clone, Debug)]
pub struct PyMlModelZoo {
    zoo: ModelZoo,
}

#[pymethods]
impl PyMlModelZoo {
    /// Create an empty model zoo.
    #[new]
    fn new() -> Self {
        Self {
            zoo: ModelZoo::new(),
        }
    }

    /// Create a zoo pre-populated with the built-in model entries.
    #[staticmethod]
    fn with_defaults() -> Self {
        Self {
            zoo: ModelZoo::with_defaults(),
        }
    }

    /// Look up an entry by ID.
    fn get(&self, id: &str) -> Option<PyMlModelEntry> {
        self.zoo.get(id).map(PyMlModelEntry::from)
    }

    /// Return every registered entry.
    fn entries(&self) -> Vec<PyMlModelEntry> {
        self.zoo.entries().map(PyMlModelEntry::from).collect()
    }

    /// Number of entries in the zoo.
    fn __len__(&self) -> usize {
        self.zoo.len()
    }

    fn __repr__(&self) -> String {
        format!("MlModelZoo(entries={})", self.zoo.len())
    }
}

// ---------------------------------------------------------------------------
// Image input helpers
// ---------------------------------------------------------------------------

/// Extract `(pixels, width, height)` from a 3D `(H, W, 3)` numpy array.
fn extract_image(array: &PyReadonlyArray3<'_, u8>) -> PyResult<(Vec<u8>, u32, u32)> {
    let dims = array.shape();
    if dims.len() != 3 || dims[2] != 3 {
        return Err(PyValueError::new_err(format!(
            "expected image shape (H, W, 3), got {dims:?}"
        )));
    }
    let height = dims[0] as u32;
    let width = dims[1] as u32;
    let view = array.as_array();
    let mut pixels = Vec::with_capacity(dims[0] * dims[1] * 3);
    for row in view.outer_iter() {
        for col in row.outer_iter() {
            for &v in col.iter() {
                pixels.push(v);
            }
        }
    }
    Ok((pixels, width, height))
}

// ---------------------------------------------------------------------------
// Scene classifier
// ---------------------------------------------------------------------------

/// One scene classification prediction.
#[pyclass(name = "SceneClassification")]
#[derive(Clone, Debug)]
pub struct PySceneClassification {
    /// Integer class index produced by the model.
    #[pyo3(get)]
    pub class_index: usize,
    /// Human-readable label (`None` unless the classifier was given a label table).
    #[pyo3(get)]
    pub label: Option<String>,
    /// Softmax score in `[0, 1]`.
    #[pyo3(get)]
    pub score: f32,
}

impl From<SceneClassification> for PySceneClassification {
    fn from(c: SceneClassification) -> Self {
        Self {
            class_index: c.class_index,
            label: c.label,
            score: c.score,
        }
    }
}

#[pymethods]
impl PySceneClassification {
    fn __repr__(&self) -> String {
        format!(
            "SceneClassification(class_index={}, label={:?}, score={:.4})",
            self.class_index, self.label, self.score
        )
    }
}

/// Places365-compatible scene classifier.
///
/// Construct with :meth:`load` and drive with :meth:`run`, passing in a
/// numpy RGB image of shape `(H, W, 3)`.
#[pyclass(name = "SceneClassifier")]
pub struct PySceneClassifier {
    inner: SceneClassifier,
}

#[pymethods]
impl PySceneClassifier {
    /// Load a classifier from an ONNX model path onto the given device.
    #[staticmethod]
    #[pyo3(signature = (path, device=None))]
    fn load(py: Python<'_>, path: &str, device: Option<PyMlDeviceType>) -> PyResult<Self> {
        let dev = device.map(|d| d.inner()).unwrap_or_else(DeviceType::auto);
        let path_owned: String = path.to_string();
        let inner = map_ml(py.detach(move || SceneClassifier::load(&path_owned, dev)))?;
        Ok(Self { inner })
    }

    /// Number of top-k predictions returned by :meth:`run`.
    #[getter]
    fn top_k(&self) -> usize {
        self.inner.top_k()
    }

    /// Path of the backing ONNX model.
    #[getter]
    fn model_path(&self) -> String {
        self.inner.model_path().to_string_lossy().into_owned()
    }

    /// Run the classifier on a single RGB image.
    ///
    /// `image` is a numpy uint8 array of shape `(H, W, 3)`.
    fn run(
        &self,
        py: Python<'_>,
        image: PyReadonlyArray3<'_, u8>,
    ) -> PyResult<Vec<PySceneClassification>> {
        let (pixels, width, height) = extract_image(&image)?;
        let scene = map_ml(SceneImage::new(pixels, width, height))?;
        let inner = &self.inner;
        let results = map_ml(py.detach(move || inner.run(scene)))?;
        Ok(results
            .into_iter()
            .map(PySceneClassification::from)
            .collect())
    }

    fn __repr__(&self) -> String {
        format!(
            "SceneClassifier(path={:?}, top_k={})",
            self.inner.model_path().to_string_lossy(),
            self.inner.top_k()
        )
    }
}

// ---------------------------------------------------------------------------
// Shot boundary detector
// ---------------------------------------------------------------------------

/// A single detected shot boundary.
#[pyclass(name = "ShotBoundary")]
#[derive(Clone, Copy, Debug)]
pub struct PyShotBoundary {
    /// Index of the boundary frame within the window.
    #[pyo3(get)]
    pub frame_index: usize,
    /// Confidence in `[0, 1]` (1.0 = definite boundary).
    #[pyo3(get)]
    pub confidence: f32,
}

impl From<ShotBoundary> for PyShotBoundary {
    fn from(b: ShotBoundary) -> Self {
        Self {
            frame_index: b.frame_index,
            confidence: b.confidence,
        }
    }
}

#[pymethods]
impl PyShotBoundary {
    fn __repr__(&self) -> String {
        format!(
            "ShotBoundary(frame_index={}, confidence={:.4})",
            self.frame_index, self.confidence
        )
    }
}

/// TransNet V2-compatible shot boundary detector.
///
/// Two construction paths:
///
/// * :meth:`load` — wraps an ONNX model.
/// * :meth:`heuristic` — uses the always-available frame-difference
///   fallback (no model required).
///
/// :meth:`run` takes a numpy uint8 array of shape `(N, H, W, 3)` — the
/// sliding window of frames.
#[pyclass(name = "ShotBoundaryDetector")]
pub struct PyShotBoundaryDetector {
    inner: ShotBoundaryDetector,
}

#[pymethods]
impl PyShotBoundaryDetector {
    /// Load an ONNX shot-boundary model from disk.
    #[staticmethod]
    #[pyo3(signature = (path, device=None))]
    fn load(py: Python<'_>, path: &str, device: Option<PyMlDeviceType>) -> PyResult<Self> {
        let dev = device.map(|d| d.inner()).unwrap_or_else(DeviceType::auto);
        let path_owned: String = path.to_string();
        let inner = map_ml(py.detach(move || ShotBoundaryDetector::load(&path_owned, dev)))?;
        Ok(Self { inner })
    }

    /// Build a heuristic frame-difference detector that does not require
    /// an ONNX model. Always available.
    #[staticmethod]
    fn heuristic() -> Self {
        Self {
            inner: ShotBoundaryDetector::heuristic(ShotBoundaryConfig::default()),
        }
    }

    /// Whether a real ONNX model is attached.
    #[getter]
    fn has_model(&self) -> bool {
        self.inner.has_model()
    }

    /// Configured boundary probability threshold.
    #[getter]
    fn threshold(&self) -> f32 {
        self.inner.threshold()
    }

    /// Run the detector over a window of RGB frames.
    ///
    /// `window` is a numpy uint8 array of shape `(N, H, W, 3)` where
    /// `N` is the number of frames in the sliding window.
    fn run(
        &self,
        py: Python<'_>,
        window: PyReadonlyArray4<'_, u8>,
    ) -> PyResult<Vec<PyShotBoundary>> {
        let dims = window.shape();
        if dims.len() != 4 || dims[3] != 3 {
            return Err(PyValueError::new_err(format!(
                "expected shot window shape (N, H, W, 3), got {dims:?}"
            )));
        }
        let n = dims[0];
        let height = dims[1] as u32;
        let width = dims[2] as u32;

        let view = window.as_array();
        let mut frames: Vec<ShotFrame> = Vec::with_capacity(n);
        for frame_view in view.outer_iter() {
            let mut pixels = Vec::with_capacity(dims[1] * dims[2] * 3);
            for row in frame_view.outer_iter() {
                for col in row.outer_iter() {
                    for &v in col.iter() {
                        pixels.push(v);
                    }
                }
            }
            let frame = map_ml(ShotFrame::new(pixels, width, height))?;
            frames.push(frame);
        }

        let inner = &self.inner;
        let boundaries = map_ml(py.detach(move || inner.run(frames)))?;
        Ok(boundaries.into_iter().map(PyShotBoundary::from).collect())
    }

    fn __repr__(&self) -> String {
        format!(
            "ShotBoundaryDetector(has_model={}, threshold={:.3})",
            self.inner.has_model(),
            self.inner.threshold()
        )
    }
}

// ---------------------------------------------------------------------------
// Aesthetic scorer
// ---------------------------------------------------------------------------

/// NIMA-style aesthetic quality score.
#[pyclass(name = "AestheticScore")]
#[derive(Clone, Copy, Debug)]
pub struct PyAestheticScore {
    /// Weighted mean score in `[1.0, 10.0]` for a well-formed distribution.
    #[pyo3(get)]
    pub score: f32,
    distribution: [f32; 10],
}

impl From<AestheticScore> for PyAestheticScore {
    fn from(s: AestheticScore) -> Self {
        Self {
            score: s.score(),
            distribution: *s.distribution(),
        }
    }
}

#[pymethods]
impl PyAestheticScore {
    /// Full 10-bin probability distribution as produced by the model.
    #[getter]
    fn distribution(&self) -> Vec<f32> {
        self.distribution.to_vec()
    }

    fn __repr__(&self) -> String {
        format!("AestheticScore(score={:.3})", self.score)
    }
}

/// NIMA-style aesthetic quality scorer.
#[pyclass(name = "AestheticScorer")]
pub struct PyAestheticScorer {
    inner: AestheticScorer,
}

#[pymethods]
impl PyAestheticScorer {
    /// Load a scorer from an ONNX model path.
    #[staticmethod]
    #[pyo3(signature = (path, device=None))]
    fn load(py: Python<'_>, path: &str, device: Option<PyMlDeviceType>) -> PyResult<Self> {
        let dev = device.map(|d| d.inner()).unwrap_or_else(DeviceType::auto);
        let path_owned: String = path.to_string();
        let inner = map_ml(py.detach(move || AestheticScorer::load(&path_owned, dev)))?;
        Ok(Self { inner })
    }

    /// Score a single RGB image of shape `(H, W, 3)`.
    fn run(&self, py: Python<'_>, image: PyReadonlyArray3<'_, u8>) -> PyResult<PyAestheticScore> {
        let (pixels, width, height) = extract_image(&image)?;
        let aesthetic = map_ml(AestheticImage::new(pixels, width, height))?;
        let inner = &self.inner;
        let score = map_ml(py.detach(move || inner.run(aesthetic)))?;
        Ok(score.into())
    }

    fn __repr__(&self) -> String {
        "AestheticScorer(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// Object detector
// ---------------------------------------------------------------------------

/// Single object-detection result.
#[pyclass(name = "Detection")]
#[derive(Clone, Debug)]
pub struct PyDetection {
    /// Class index produced by the detector.
    #[pyo3(get)]
    pub class_id: u32,
    /// Post-sigmoid confidence in `[0, 1]`.
    #[pyo3(get)]
    pub score: f32,
    /// Bounding box in corner form: `(x0, y0, x1, y1)`.
    #[pyo3(get)]
    pub bbox: (f32, f32, f32, f32),
}

impl From<Detection> for PyDetection {
    fn from(d: Detection) -> Self {
        Self {
            class_id: d.class_id,
            score: d.score,
            bbox: (d.bbox.x0, d.bbox.y0, d.bbox.x1, d.bbox.y1),
        }
    }
}

#[pymethods]
impl PyDetection {
    fn __repr__(&self) -> String {
        format!(
            "Detection(class_id={}, score={:.4}, bbox=({:.2},{:.2},{:.2},{:.2}))",
            self.class_id, self.score, self.bbox.0, self.bbox.1, self.bbox.2, self.bbox.3
        )
    }
}

/// YOLOv8-compatible object detector.
#[pyclass(name = "ObjectDetector")]
pub struct PyObjectDetector {
    inner: ObjectDetector,
}

#[pymethods]
impl PyObjectDetector {
    /// Load a detector from an ONNX model path.
    #[staticmethod]
    #[pyo3(signature = (path, device=None))]
    fn load(py: Python<'_>, path: &str, device: Option<PyMlDeviceType>) -> PyResult<Self> {
        let dev = device.map(|d| d.inner()).unwrap_or_else(DeviceType::auto);
        let path_owned: String = path.to_string();
        let inner = map_ml(py.detach(move || ObjectDetector::load(&path_owned, dev)))?;
        Ok(Self { inner })
    }

    /// Run the detector on a single RGB image of shape `(H, W, 3)`.
    fn run(&self, py: Python<'_>, image: PyReadonlyArray3<'_, u8>) -> PyResult<Vec<PyDetection>> {
        let (pixels, width, height) = extract_image(&image)?;
        let frame = map_ml(DetectorImage::new(pixels, width, height))?;
        let inner = &self.inner;
        let dets = map_ml(py.detach(move || inner.run(frame)))?;
        Ok(dets.into_iter().map(PyDetection::from).collect())
    }

    fn __repr__(&self) -> String {
        "ObjectDetector(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// Face embedder
// ---------------------------------------------------------------------------

/// L2-normalised face embedding.
#[pyclass(name = "FaceEmbedding")]
#[derive(Clone, Debug)]
pub struct PyFaceEmbedding {
    values: Vec<f32>,
}

impl From<FaceEmbedding> for PyFaceEmbedding {
    fn from(emb: FaceEmbedding) -> Self {
        Self {
            values: emb.into_inner(),
        }
    }
}

#[pymethods]
impl PyFaceEmbedding {
    /// Construct a [`FaceEmbedding`] from an L2-normalised sequence.
    ///
    /// The input is not re-normalised; pass pre-normalised values. Use
    /// :meth:`from_raw` if you want the wrapper to normalise for you.
    #[new]
    fn py_new(values: Vec<f32>) -> Self {
        Self { values }
    }

    /// Wrap a raw vector and L2-normalise it in place.
    #[staticmethod]
    fn from_raw(values: Vec<f32>) -> Self {
        FaceEmbedding::from_raw(values).into()
    }

    /// Dimensionality of the embedding.
    fn __len__(&self) -> usize {
        self.values.len()
    }

    /// Raw f32 values as a Python list.
    fn to_list(&self) -> Vec<f32> {
        self.values.clone()
    }

    /// Raw f32 values as a numpy array.
    fn to_numpy(&self, py: Python<'_>) -> Py<PyAny> {
        self.values.clone().into_pyarray(py).into_any().unbind()
    }

    /// Cosine similarity with another embedding. Returns `0.0` when
    /// dimensions mismatch.
    fn cosine_similarity(&self, other: &Self) -> f32 {
        oximedia_ml::cosine_similarity(&self.values, &other.values)
    }

    fn __repr__(&self) -> String {
        format!("FaceEmbedding(dim={})", self.values.len())
    }
}

/// ArcFace / FaceNet-compatible face embedder.
#[pyclass(name = "FaceEmbedder")]
pub struct PyFaceEmbedder {
    inner: FaceEmbedder,
}

#[pymethods]
impl PyFaceEmbedder {
    /// Load an embedder from an ONNX model path.
    #[staticmethod]
    #[pyo3(signature = (path, device=None))]
    fn load(py: Python<'_>, path: &str, device: Option<PyMlDeviceType>) -> PyResult<Self> {
        let dev = device.map(|d| d.inner()).unwrap_or_else(DeviceType::auto);
        let path_owned: String = path.to_string();
        let inner = map_ml(py.detach(move || FaceEmbedder::load(&path_owned, dev)))?;
        Ok(Self { inner })
    }

    /// Embed an aligned RGB face crop of shape `(H, W, 3)`.
    fn run(&self, py: Python<'_>, image: PyReadonlyArray3<'_, u8>) -> PyResult<PyFaceEmbedding> {
        let (pixels, width, height) = extract_image(&image)?;
        let face = map_ml(FaceImage::new(pixels, width, height))?;
        let inner = &self.inner;
        let emb = map_ml(py.detach(move || inner.run(face)))?;
        Ok(emb.into())
    }

    fn __repr__(&self) -> String {
        "FaceEmbedder(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// Module-level helpers
// ---------------------------------------------------------------------------

/// Return a list of every available backend as :class:`MlDeviceType` objects.
#[pyfunction]
fn available_devices(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    let devices: Vec<PyMlDeviceType> = DeviceType::list_available()
        .into_iter()
        .map(PyMlDeviceType::new)
        .collect();
    PyList::new(py, devices)
}

/// Pick the strongest available backend.
#[pyfunction]
fn auto_device() -> PyMlDeviceType {
    PyMlDeviceType::new(DeviceType::auto())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register the `oximedia.ml` submodule into the parent module.
pub fn register_submodule(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "ml")?;

    m.add_class::<PyMlDeviceType>()?;
    m.add_class::<PyMlDeviceCapabilities>()?;
    m.add_class::<PyMlTensorDType>()?;
    m.add_class::<PyMlTensorSpec>()?;
    m.add_class::<PyMlModelInfo>()?;
    m.add_class::<PyOnnxModel>()?;
    m.add_class::<PyMlModelEntry>()?;
    m.add_class::<PyMlModelZoo>()?;

    m.add_class::<PySceneClassification>()?;
    m.add_class::<PySceneClassifier>()?;

    m.add_class::<PyShotBoundary>()?;
    m.add_class::<PyShotBoundaryDetector>()?;

    m.add_class::<PyAestheticScore>()?;
    m.add_class::<PyAestheticScorer>()?;

    m.add_class::<PyDetection>()?;
    m.add_class::<PyObjectDetector>()?;

    m.add_class::<PyFaceEmbedding>()?;
    m.add_class::<PyFaceEmbedder>()?;

    m.add_function(wrap_pyfunction!(available_devices, &m)?)?;
    m.add_function(wrap_pyfunction!(auto_device, &m)?)?;

    parent.add_submodule(&m)?;
    Ok(())
}
