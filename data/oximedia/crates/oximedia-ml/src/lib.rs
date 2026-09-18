//! # oximedia-ml — Sovereign ML for Media
//!
//! `oximedia-ml` wraps the [Pure-Rust OxiONNX](https://crates.io/crates/oxionnx)
//! runtime in a set of typed pipelines tailored to multimedia workloads:
//! scene classification, shot boundary detection, aesthetic scoring,
//! object detection, face embedding, and more as the feature-gated zoo
//! grows.
//!
//! ## Design goals
//!
//! * **Default Pure Rust** — the default build pulls in *zero* ONNX
//!   symbols. Enable the `onnx` feature to opt in to inference.
//! * **Typed pipelines** — callers get domain-shaped inputs and outputs
//!   rather than raw tensors. Every pipeline implements
//!   [`TypedPipeline`].
//! * **No unwrap policy** — every fallible operation returns
//!   [`MlResult`]; doc-tests follow the same rule.
//! * **Cache-friendly** — loaded models can be shared across pipelines
//!   via [`ModelCache`] (bounded LRU keyed by canonical path).
//! * **Device portable** — [`DeviceType::auto`] picks the best available
//!   backend at runtime (CUDA → DirectML → WebGPU → CPU) and memoises
//!   the result.
//!
//! ## Quick start
//!
//! Load a Places365-compatible scene classifier, run it on a 224×224 RGB
//! frame, and print the top-5 predictions:
//!
//! ```no_run
//! # #[cfg(all(feature = "onnx", feature = "scene-classifier"))]
//! # fn demo() -> oximedia_ml::MlResult<()> {
//! use oximedia_ml::pipelines::{SceneClassifier, SceneImage};
//! use oximedia_ml::{DeviceType, TypedPipeline};
//!
//! let classifier = SceneClassifier::load("places365.onnx", DeviceType::auto())?;
//! let image = SceneImage::new(vec![0u8; 224 * 224 * 3], 224, 224)?;
//! for pred in classifier.run(image)? {
//!     println!("class {} @ {:.3}", pred.class_index, pred.score);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Feature matrix
//!
//! Backend features control which ONNX execution providers are compiled
//! in; pipeline features enable individual domain adapters. Everything
//! except `cuda` is WASM-compatible (see the support table below).
//!
//! | Feature              | Purpose                                                           | Notes                         |
//! |----------------------|-------------------------------------------------------------------|-------------------------------|
//! | `onnx`               | Enables the real [`OnnxModel`] backed by OxiONNX.                 | Required for any inference.   |
//! | `cuda`               | Additionally compile `oxionnx-cuda` for NVIDIA GPU execution.     | **Native only** (no WASM).    |
//! | `webgpu`             | Additionally compile `oxionnx-gpu` (wgpu backend).                | Works on native + browsers.   |
//! | `directml`           | Additionally compile `oxionnx-directml`.                          | Stub outside Windows.         |
//! | `serde`              | Derives `Serialize` on pipeline info/value types.                 | Opt-in; no runtime cost.      |
//! | `scene-classifier`   | Builds the `pipelines::SceneClassifier` pipeline.                 | Places365-compatible.         |
//! | `shot-boundary`      | Builds the `pipelines::ShotBoundaryDetector` pipeline.            | TransNet V2-compatible.       |
//! | `aesthetic-score`    | Builds the `pipelines::AestheticScorer` pipeline.                 | NIMA-compatible.              |
//! | `object-detector`    | Builds the `pipelines::ObjectDetector` pipeline.                  | YOLOv8-compatible.            |
//! | `face-embedder`      | Builds the `pipelines::FaceEmbedder` pipeline.                    | ArcFace-compatible.           |
//! | `all-pipelines`      | Shortcut enabling every pipeline above.                           | Implies `onnx`.               |
//!
//! ## Device selection
//!
//! Callers rarely need to hard-code a backend. [`DeviceType::auto`]
//! probes capabilities once (memoised in an `OnceLock`) and returns the
//! strongest available device:
//!
//! ```no_run
//! use oximedia_ml::{DeviceCapabilities, DeviceType};
//!
//! // Cached after the first call.
//! let device = DeviceType::auto();
//!
//! // Want the full capability report? (panic-safe probes.)
//! for cap in DeviceCapabilities::probe_all() {
//!     println!(
//!         "{:?}: {}",
//!         cap.device_type,
//!         if cap.is_available { "available" } else { "unavailable" },
//!     );
//! }
//! ```
//!
//! Each pipeline constructor accepts a [`DeviceType`]. Pass
//! [`DeviceType::Cpu`] to force the pure-Rust path, or pick a specific
//! GPU backend when you know the deployment target.
//!
//! ## Pipeline ecosystem
//!
//! All pipelines live under [`pipelines`] and implement
//! [`TypedPipeline`]. Each is gated behind its own feature so apps only
//! compile what they use:
//!
//! | Pipeline                              | Feature             | Input             | Output                        | Reference model   |
//! |---------------------------------------|---------------------|-------------------|-------------------------------|-------------------|
//! | `pipelines::SceneClassifier`          | `scene-classifier`  | 224×224 RGB frame | `Vec<SceneClassification>`    | Places365/ResNet  |
//! | `pipelines::ShotBoundaryDetector`     | `shot-boundary`     | 48×27 RGB window  | `Vec<ShotBoundary>`           | TransNet V2       |
//! | `pipelines::AestheticScorer`          | `aesthetic-score`   | 224×224 RGB frame | [`AestheticScore`]            | NIMA              |
//! | `pipelines::ObjectDetector`           | `object-detector`   | 640×640 RGB frame | `Vec<Detection>`              | YOLOv8 (80 COCO)  |
//! | `pipelines::FaceEmbedder`             | `face-embedder`     | 112×112 RGB face  | [`FaceEmbedding`] (512-dim)   | ArcFace           |
//!
//! Value types ([`AestheticScore`], [`Detection`], [`FaceEmbedding`]) are
//! always re-exported at the crate root so callers can handle results
//! even if they only consume them from another crate.
//!
//! ## WebAssembly (`wasm32-unknown-unknown`)
//!
//! `oximedia-ml` is validated for the WASM target on every release.  The
//! support matrix is:
//!
//! | Feature set                                                              | `wasm32-unknown-unknown` |
//! |--------------------------------------------------------------------------|--------------------------|
//! | *default* (no features)                                                  | builds                   |
//! | `onnx`                                                                   | builds                   |
//! | `onnx` + any subset of `scene-classifier`/`shot-boundary`/`aesthetic-score`/`object-detector`/`face-embedder`/`all-pipelines` | builds                   |
//! | `webgpu` (wgpu browser backend)                                          | builds                   |
//! | `directml` (stub on non-Windows)                                         | builds                   |
//! | `cuda`                                                                   | **does not build**       |
//!
//! The `cuda` feature transitively depends on `oxicuda-driver`, which uses
//! [`libloading`] to bind the NVIDIA driver at runtime.  `libloading` gates
//! its `Library` type behind `cfg(any(unix, windows))`, so the crate will
//! never compile on `wasm32-unknown-unknown`.  This is a fundamental
//! property of GPU driver loading rather than a limitation of this crate,
//! so `cuda` is treated as a **native-only** feature.
//!
//! Everything on WASM executes the pure-Rust CPU path ([`DeviceType::Cpu`]),
//! which is what browsers actually want anyway — the WebGPU backend is
//! opted into by enabling the `webgpu` feature.  There is no mock inference
//! path; if the `onnx` feature is disabled, `OnnxModel::load` returns
//! [`MlError::FeatureDisabled`] as on native.
//!
//! [`libloading`]: https://crates.io/crates/libloading

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![warn(missing_docs)]

pub mod cache;
pub mod device;
pub mod error;
pub mod model;
pub mod pipeline;
pub mod pipelines;
pub mod postprocess;
pub mod preprocess;
pub mod zoo;

pub use cache::{ModelCache, DEFAULT_CAPACITY};
pub use device::{DeviceCapabilities, DeviceType};
pub use error::{MlError, MlResult};
pub use model::{load_auto, ModelInfo, OnnxModel, TensorDType, TensorSpec};
pub use pipeline::{PipelineInfo, PipelineTask, TypedPipeline};
pub use pipelines::{AestheticScore, Detection, FaceEmbedding};
pub use postprocess::{
    argmax, cosine_similarity, iou, l2_normalize, nms, sigmoid, sigmoid_slice, softmax, top_k,
    BoundingBox,
};
pub use preprocess::{ImagePreprocessor, InputRange, PixelLayout, TensorLayout};
pub use zoo::{ModelEntry, ModelZoo};
