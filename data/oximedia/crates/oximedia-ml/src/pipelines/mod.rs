//! Built-in typed pipelines.
//!
//! Each pipeline lives behind a feature gate:
//!
//! * `scene-classifier` → `scene_classifier::SceneClassifier`
//! * `shot-boundary` → `shot_boundary::ShotBoundaryDetector`
//! * `aesthetic-score` → `aesthetic_score::AestheticScorer`
//! * `object-detector` → `object_detector::ObjectDetector`
//! * `face-embedder` → `face_embedder::FaceEmbedder`
//!
//! Every pipeline implements the [`crate::TypedPipeline`] trait, which
//! means callers can treat them uniformly (store them in
//! `Vec<Box<dyn TypedPipeline<…>>>`, drive them from a workflow
//! engine, etc.) as long as the `Input` / `Output` types line up.
//!
//! Shared value types ([`Detection`], [`FaceEmbedding`],
//! [`AestheticScore`]) live in [`types`] and are always compiled so
//! the crate-root re-exports remain stable regardless of the active
//! feature set.
//!
//! Browse each submodule for contract details (tensor shapes, required
//! postprocess helpers, compatible model families, and `# Examples`
//! blocks).

pub mod types;

pub use types::{AestheticScore, Detection, FaceEmbedding};

#[cfg(feature = "scene-classifier")]
pub mod scene_classifier;

#[cfg(feature = "shot-boundary")]
pub mod shot_boundary;

#[cfg(feature = "aesthetic-score")]
pub mod aesthetic_score;

#[cfg(feature = "object-detector")]
pub mod object_detector;

#[cfg(feature = "face-embedder")]
pub mod face_embedder;

pub mod auto_caption;

#[cfg(feature = "auto-caption")]
pub use auto_caption::{AutoCaptionConfig, AutoCaptionPipeline};

#[cfg(feature = "scene-classifier")]
pub use scene_classifier::{
    SceneClassification, SceneClassifier, SceneClassifierConfig, SceneImage,
};

#[cfg(feature = "shot-boundary")]
pub use shot_boundary::{ShotBoundary, ShotBoundaryConfig, ShotBoundaryDetector, ShotFrame};

#[cfg(feature = "aesthetic-score")]
pub use aesthetic_score::{AestheticImage, AestheticScorer, AestheticScorerConfig};

#[cfg(feature = "object-detector")]
pub use object_detector::{
    decode_yolov8_output, DecodeOptions, DetectorImage, ObjectDetector, ObjectDetectorConfig,
    YOLOV8_CHANNELS, YOLOV8_NUM_CLASSES,
};

#[cfg(feature = "face-embedder")]
pub use face_embedder::{FaceEmbedder, FaceEmbedderConfig, FaceImage};
