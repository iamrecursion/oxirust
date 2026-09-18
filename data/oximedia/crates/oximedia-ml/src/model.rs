//! Pure-Rust ONNX model wrapper.
//!
//! [`OnnxModel`] is a thin façade over `oxionnx::Session` that exposes a
//! stable, Pure-Rust-only surface. The intent is that the rest of
//! oximedia-ml (and its downstream pipelines) never imports `oxionnx`
//! symbols directly — everything goes through [`OnnxModel`],
//! [`TensorSpec`], and [`ModelInfo`].
//!
//! When the `onnx` feature is disabled, [`OnnxModel`] still exists but
//! its constructor returns [`crate::error::MlError::FeatureDisabled`].
//! Pipelines can then decide whether to fall back to a non-ML heuristic
//! or propagate the error upward.
//!
//! ## Single-input convenience
//!
//! Many classifier / detector models have a single input and one or more
//! float outputs. For these,
//! [`OnnxModel::run_single`][crate::OnnxModel::run_single] skips the
//! `HashMap<&str, Tensor>` boilerplate and returns flat `Vec<f32>`
//! buffers keyed on output name:
//!
//! ```no_run
//! # #[cfg(feature = "onnx")]
//! # fn demo() -> oximedia_ml::MlResult<()> {
//! use oximedia_ml::{DeviceType, OnnxModel};
//!
//! let model = OnnxModel::load("scene.onnx", DeviceType::auto())?;
//! let outputs = model.run_single(
//!     "input",
//!     vec![0.0_f32; 1 * 3 * 224 * 224],
//!     vec![1, 3, 224, 224],
//! )?;
//! // `outputs` is `HashMap<String, Vec<f32>>`.
//! # let _ = outputs;
//! # Ok(())
//! # }
//! ```
//!
//! ## Metadata
//!
//! [`ModelInfo`] exposes input/output [`TensorSpec`]s (name, dtype,
//! shape with dynamic dims as `None`), the ONNX producer name, and the
//! opset version. Callers can use [`TensorSpec::dynamic_rank`] to decide
//! whether dynamic shape plumbing is needed.

use std::path::{Path, PathBuf};

use crate::device::DeviceType;
use crate::error::MlResult;

/// Canonical scalar dtype advertised by a model input or output.
///
/// Mirrors a pragmatic subset of the ONNX dtype list. Internally
/// `oxionnx` stores tensor data as `f32`, so anything non-`F32` signals
/// a cast at the boundary (performed inside the pipeline layer).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorDType {
    /// 32-bit IEEE float.
    F32,
    /// 16-bit IEEE float.
    F16,
    /// 64-bit IEEE float.
    F64,
    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 16-bit integer.
    U16,
    /// Unsigned 32-bit integer.
    U32,
    /// Unsigned 64-bit integer.
    U64,
    /// Boolean.
    Bool,
}

impl TensorDType {
    /// Short canonical name matching ONNX nomenclature.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::F64 => "f64",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::Bool => "bool",
        }
    }
}

/// Describes a single model input or output tensor.
#[derive(Clone, Debug)]
pub struct TensorSpec {
    /// Tensor name as declared in the ONNX graph.
    pub name: String,
    /// Scalar dtype.
    pub dtype: TensorDType,
    /// Shape with dynamic (None) dimensions expressed as `None`.
    /// Static dimensions are positive integers; `i64` is used to match
    /// the ONNX specification convention.
    pub shape: Vec<Option<i64>>,
}

impl TensorSpec {
    /// Create a new [`TensorSpec`].
    #[must_use]
    pub fn new(name: impl Into<String>, dtype: TensorDType, shape: Vec<Option<i64>>) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape,
        }
    }

    /// Number of dynamic dimensions (those reported as `None`).
    #[must_use]
    pub fn dynamic_rank(&self) -> usize {
        self.shape.iter().filter(|d| d.is_none()).count()
    }
}

/// Static metadata describing a loaded ONNX model.
///
/// Returned by [`OnnxModel::info`][crate::OnnxModel::info]; inspect
/// [`Self::inputs`] / [`Self::outputs`] to validate the expected tensor
/// contract before running inference.
#[derive(Clone, Debug, Default)]
pub struct ModelInfo {
    /// Source path of the model.
    pub path: PathBuf,
    /// Model input tensor specifications.
    pub inputs: Vec<TensorSpec>,
    /// Model output tensor specifications.
    pub outputs: Vec<TensorSpec>,
    /// Producer name as declared in the ONNX file.
    pub producer: Option<String>,
    /// Opset version, if reported by the backend.
    pub opset_version: Option<i64>,
}

#[cfg(feature = "onnx")]
mod imp {
    use super::{ModelInfo, TensorDType, TensorSpec};
    use crate::device::DeviceType;
    use crate::error::{MlError, MlResult};
    use oxionnx::execution_providers::ProviderKind;
    use oxionnx::graph::TensorInfo;
    use oxionnx::DType;
    use oxionnx::{OptLevel, Session, SessionBuilder, Tensor};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    /// Pure-Rust ONNX model handle.
    pub struct OnnxModel {
        session: Mutex<Session>,
        info: ModelInfo,
        device: DeviceType,
    }

    impl std::fmt::Debug for OnnxModel {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("OnnxModel")
                .field("device", &self.device)
                .field("info", &self.info)
                .finish()
        }
    }

    impl OnnxModel {
        /// Load an ONNX model from disk.
        ///
        /// The session is configured with an ordered [`ProviderKind`] list
        /// derived from `device` (see `device_to_providers`).  CPU is
        /// always the implicit terminal fallback, so the session can never
        /// fail solely because a GPU backend is unavailable at runtime.
        ///
        /// # Errors
        ///
        /// * [`MlError::DeviceUnavailable`] if `device` is not available in
        ///   this build / runtime.
        /// * [`MlError::ModelLoad`] if the file cannot be parsed by the
        ///   OxiONNX runtime.
        pub fn load(path: impl AsRef<Path>, device: DeviceType) -> MlResult<Self> {
            let path_ref = path.as_ref();
            if !device.is_available() {
                return Err(MlError::DeviceUnavailable(device.name().to_string()));
            }

            let providers = device_to_providers(device);
            let session = SessionBuilder::new()
                .with_optimization_level(OptLevel::All)
                .with_provider_kinds(providers)
                .load(path_ref)
                .map_err(|e| MlError::ModelLoad {
                    path: PathBuf::from(path_ref),
                    reason: format!("{e:?}"),
                })?;

            let info = extract_info(&session, path_ref);

            Ok(Self {
                session: Mutex::new(session),
                info,
                device,
            })
        }

        /// Load an ONNX model from an in-memory byte buffer.
        ///
        /// `virtual_path` is a synthetic identifier used for
        /// [`ModelInfo::path`] and cache keying; it does not need to
        /// refer to a real file on disk.
        ///
        /// The session is configured with an ordered [`ProviderKind`] list
        /// derived from `device` (see `device_to_providers`).
        ///
        /// # Errors
        ///
        /// * [`MlError::DeviceUnavailable`] if `device` is not available.
        /// * [`MlError::ModelLoad`] if `bytes` is not a valid ONNX payload.
        pub fn load_from_bytes(
            bytes: &[u8],
            device: DeviceType,
            virtual_path: impl Into<PathBuf>,
        ) -> MlResult<Self> {
            if !device.is_available() {
                return Err(MlError::DeviceUnavailable(device.name().to_string()));
            }
            let path = virtual_path.into();
            let providers = device_to_providers(device);
            let session = SessionBuilder::new()
                .with_optimization_level(OptLevel::All)
                .with_provider_kinds(providers)
                .load_from_bytes(bytes)
                .map_err(|e| MlError::ModelLoad {
                    path: path.clone(),
                    reason: format!("{e:?}"),
                })?;

            let mut info = extract_info(&session, &path);
            info.path = path;

            Ok(Self {
                session: Mutex::new(session),
                info,
                device,
            })
        }

        /// Execute a forward pass.
        ///
        /// Inputs are keyed by input tensor name. Outputs are returned
        /// as a map from output name to tensor.
        ///
        /// # Errors
        ///
        /// * [`MlError::Pipeline`] with stage `"onnx"` if the session
        ///   mutex is poisoned.
        /// * [`MlError::OnnxRuntime`] if inference fails inside OxiONNX.
        pub fn run(&self, inputs: &HashMap<&str, Tensor>) -> MlResult<HashMap<String, Tensor>> {
            let guard = self
                .session
                .lock()
                .map_err(|_| MlError::pipeline("onnx", "session mutex poisoned"))?;
            guard
                .run(inputs)
                .map_err(|e| MlError::OnnxRuntime(format!("{e:?}")))
        }

        /// Execute a forward pass for a single-input model, returning the
        /// raw `f32` buffer of every output tensor.
        ///
        /// This is a convenience adapter for the common case where the
        /// caller has a single `Vec<f32>` payload and a shape, and does not
        /// want to import `oxionnx::Tensor` into its own crate.  Pure-Rust
        /// downstream crates (e.g. `oximedia-recommend`) can build their
        /// embedding pipelines on top of `OnnxModel` without touching the
        /// `oxionnx` symbol surface at all, preserving the encapsulation
        /// documented in the module header.
        ///
        /// # Errors
        ///
        /// * [`MlError::OnnxRuntime`] if the underlying `Session::run` fails.
        /// * [`MlError::pipeline`] if the session mutex is poisoned.
        pub fn run_single(
            &self,
            input_name: &str,
            data: Vec<f32>,
            shape: Vec<usize>,
        ) -> MlResult<HashMap<String, Vec<f32>>> {
            let expected = shape.iter().product::<usize>();
            if data.len() != expected {
                return Err(MlError::pipeline(
                    "onnx",
                    format!(
                        "run_single: data len {} does not match shape product {}",
                        data.len(),
                        expected,
                    ),
                ));
            }
            let tensor = Tensor { data, shape };
            let mut inputs: HashMap<&str, Tensor> = HashMap::with_capacity(1);
            inputs.insert(input_name, tensor);
            let outputs = self.run(&inputs)?;
            Ok(outputs
                .into_iter()
                .map(|(name, t)| (name, t.data))
                .collect())
        }

        /// Return the loaded model metadata.
        #[must_use]
        pub fn info(&self) -> &ModelInfo {
            &self.info
        }

        /// Return the device this model was loaded onto.
        #[must_use]
        pub fn device(&self) -> DeviceType {
            self.device
        }
    }

    /// Map a [`DeviceType`] to an ordered list of [`ProviderKind`] backends.
    ///
    /// The returned list is passed to
    /// [`SessionBuilder::with_provider_kinds`][oxionnx::SessionBuilder::with_provider_kinds]
    /// so that the oxionnx dispatch loop tries the requested backend first
    /// and falls back to CPU when it is unavailable or returns `None` for a
    /// specific operator.
    ///
    /// | DeviceType  | Provider priority list              |
    /// |-------------|--------------------------------------|
    /// | Cpu         | `[Cpu]`                             |
    /// | Cuda        | `[Cuda, Cpu]` (feature `cuda`)      |
    /// | WebGpu      | `[Gpu, Cpu]`  (feature `webgpu`)    |
    /// | DirectMl    | `[DirectMl, Cpu]` (feature `directml`) |
    /// | CoreMl      | `[Cpu]` (CoreML not yet wired)      |
    fn device_to_providers(device: DeviceType) -> Vec<ProviderKind> {
        match device {
            DeviceType::Cpu | DeviceType::CoreMl => vec![ProviderKind::Cpu],
            DeviceType::Cuda => {
                // cuda feature is required for ProviderKind::Cuda to exist.
                #[cfg(feature = "cuda")]
                {
                    vec![ProviderKind::Cuda, ProviderKind::Cpu]
                }
                #[cfg(not(feature = "cuda"))]
                {
                    vec![ProviderKind::Cpu]
                }
            }
            DeviceType::WebGpu => {
                // webgpu feature is required for ProviderKind::Gpu to exist.
                #[cfg(feature = "webgpu")]
                {
                    vec![ProviderKind::Gpu, ProviderKind::Cpu]
                }
                #[cfg(not(feature = "webgpu"))]
                {
                    vec![ProviderKind::Cpu]
                }
            }
            DeviceType::DirectMl => {
                // directml feature is required for ProviderKind::DirectMl to exist.
                #[cfg(feature = "directml")]
                {
                    vec![ProviderKind::DirectMl, ProviderKind::Cpu]
                }
                #[cfg(not(feature = "directml"))]
                {
                    vec![ProviderKind::Cpu]
                }
            }
        }
    }

    fn extract_info(session: &Session, path: &Path) -> ModelInfo {
        let inputs = session
            .input_info()
            .iter()
            .map(tensor_info_to_spec)
            .collect();
        let outputs = session
            .output_info()
            .iter()
            .map(tensor_info_to_spec)
            .collect();

        let meta = session.metadata();
        let producer = meta.producer_name.clone();
        let opset_version = if meta.ir_version == 0 {
            None
        } else {
            Some(meta.ir_version)
        };

        ModelInfo {
            path: PathBuf::from(path),
            inputs,
            outputs,
            producer: if producer.is_empty() {
                None
            } else {
                Some(producer)
            },
            opset_version,
        }
    }

    fn tensor_info_to_spec(info: &TensorInfo) -> TensorSpec {
        TensorSpec {
            name: info.name.clone(),
            dtype: dtype_to_public(info.dtype),
            shape: info.shape.iter().map(|d| d.map(|v| v as i64)).collect(),
        }
    }

    fn dtype_to_public(dtype: DType) -> TensorDType {
        match dtype {
            DType::F32 => TensorDType::F32,
            DType::F16 | DType::BF16 => TensorDType::F16,
            DType::F64 => TensorDType::F64,
            DType::I8 => TensorDType::I8,
            DType::I16 => TensorDType::I16,
            DType::I32 => TensorDType::I32,
            DType::I64 => TensorDType::I64,
            DType::U8 => TensorDType::U8,
            DType::U16 => TensorDType::U16,
            DType::U32 => TensorDType::U32,
            DType::U64 => TensorDType::U64,
            DType::Bool => TensorDType::Bool,
        }
    }
}

#[cfg(not(feature = "onnx"))]
mod imp {
    use super::ModelInfo;
    use crate::device::DeviceType;
    use crate::error::{MlError, MlResult};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    /// Stub ONNX model used when the `onnx` feature is disabled.
    ///
    /// All constructors return [`MlError::FeatureDisabled`] so that
    /// downstream code can degrade gracefully at runtime without special
    /// `cfg` handling of its own.
    #[derive(Debug)]
    pub struct OnnxModel {
        _priv: (),
    }

    impl OnnxModel {
        /// Always fails with [`MlError::FeatureDisabled`] in this build.
        pub fn load(_path: impl AsRef<Path>, _device: DeviceType) -> MlResult<Self> {
            Err(MlError::FeatureDisabled("onnx"))
        }

        /// Always fails with [`MlError::FeatureDisabled`] in this build.
        pub fn load_from_bytes(
            _bytes: &[u8],
            _device: DeviceType,
            _virtual_path: impl Into<PathBuf>,
        ) -> MlResult<Self> {
            Err(MlError::FeatureDisabled("onnx"))
        }

        /// Always fails with [`MlError::FeatureDisabled`] in this build.
        pub fn run_single(
            &self,
            _input_name: &str,
            _data: Vec<f32>,
            _shape: Vec<usize>,
        ) -> MlResult<HashMap<String, Vec<f32>>> {
            Err(MlError::FeatureDisabled("onnx"))
        }

        /// Always returns an empty [`ModelInfo`].
        #[must_use]
        pub fn info(&self) -> &ModelInfo {
            // Unreachable in practice — `load` never succeeds — but provides
            // a safe default so pattern matching compiles.
            static EMPTY: std::sync::OnceLock<ModelInfo> = std::sync::OnceLock::new();
            EMPTY.get_or_init(ModelInfo::default)
        }

        /// Device this stub was requested with (always CPU).
        #[must_use]
        pub fn device(&self) -> DeviceType {
            DeviceType::Cpu
        }
    }
}

pub use imp::OnnxModel;

/// Convenience wrapper: load a model with the auto-selected device.
///
/// Equivalent to
/// `OnnxModel::load(path, DeviceType::auto())`. Use when you do not
/// need to pin a specific backend.
///
/// # Errors
///
/// Propagates any error from
/// [`OnnxModel::load`][crate::OnnxModel::load].
pub fn load_auto(path: impl AsRef<Path>) -> MlResult<OnnxModel> {
    OnnxModel::load(path, DeviceType::auto())
}

/// Stable path-based hint used by [`crate::ModelCache`] as a key.
///
/// Canonicalises `path` when possible; falls back to the raw path if
/// canonicalisation fails (e.g. the target does not exist yet).
#[must_use]
pub fn canonical_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| PathBuf::from(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "onnx"))]
    use crate::error::MlError;

    #[test]
    fn tensor_spec_dynamic_rank_counts_nones() {
        let spec = TensorSpec::new(
            "x",
            TensorDType::F32,
            vec![None, Some(3), Some(224), Some(224)],
        );
        assert_eq!(spec.dynamic_rank(), 1);
    }

    #[test]
    fn dtype_names_are_canonical() {
        assert_eq!(TensorDType::F32.name(), "f32");
        assert_eq!(TensorDType::I64.name(), "i64");
        assert_eq!(TensorDType::Bool.name(), "bool");
    }

    #[cfg(not(feature = "onnx"))]
    #[test]
    fn load_without_onnx_feature_reports_feature_disabled() {
        let err =
            OnnxModel::load("does-not-matter.onnx", DeviceType::Cpu).expect_err("expected failure");
        matches!(err, MlError::FeatureDisabled("onnx"));
    }
}
