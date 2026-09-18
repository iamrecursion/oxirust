//! Typed ML pipeline abstraction.
//!
//! A "pipeline" in oximedia-ml is a typed adapter that:
//!   1. Accepts a domain-specific `Input` value (an image, a video
//!      frame, a sliding window, …).
//!   2. Runs pre-processing → ONNX inference → post-processing.
//!   3. Returns a domain-specific `Output` value (e.g. a
//!      `SceneClassification`, a `Vec<ShotBoundary>`).
//!
//! Pipelines share the [`TypedPipeline`] trait so that higher-level
//! code can treat them uniformly (think: workflow graphs, batching
//! layers, benchmark harnesses). The trait deliberately does not assume
//! a particular tensor type — the backend details live inside each
//! implementor.
//!
//! ## Implementing a custom pipeline
//!
//! ```
//! use oximedia_ml::{MlResult, PipelineInfo, PipelineTask, TypedPipeline};
//!
//! struct Identity;
//!
//! impl TypedPipeline for Identity {
//!     type Input = f32;
//!     type Output = f32;
//!
//!     fn run(&self, input: Self::Input) -> MlResult<Self::Output> {
//!         Ok(input)
//!     }
//!
//!     fn info(&self) -> PipelineInfo {
//!         PipelineInfo {
//!             id: "custom/identity",
//!             name: "Identity",
//!             task: PipelineTask::Custom,
//!             input_size: None,
//!         }
//!     }
//! }
//!
//! # fn main() -> MlResult<()> {
//! let p = Identity;
//! assert_eq!(p.run(3.5)?, 3.5);
//! assert_eq!(p.info().id, "custom/identity");
//! # Ok(())
//! # }
//! ```

use crate::error::MlResult;

/// Static description of a pipeline.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct PipelineInfo {
    /// Stable identifier, e.g. `"scene-classifier/places365"`.
    pub id: &'static str,
    /// Short human-readable name.
    pub name: &'static str,
    /// Pipeline task (classification, detection, …).
    pub task: PipelineTask,
    /// Input image size expected by the pipeline, if applicable.
    pub input_size: Option<(u32, u32)>,
}

/// Broad category of ML task handled by a pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum PipelineTask {
    /// Per-frame classification.
    SceneClassification,
    /// Shot boundary detection.
    ShotBoundary,
    /// Object detection.
    Detection,
    /// Segmentation.
    Segmentation,
    /// Aesthetic / quality scoring (e.g. NIMA).
    AestheticScoring,
    /// Face embedding extraction (e.g. ArcFace / FaceNet).
    FaceEmbedding,
    /// Generic custom pipeline.
    Custom,
}

/// Typed inference pipeline.
///
/// Implementors encapsulate all three stages (preprocess, inference,
/// postprocess) and expose a single [`TypedPipeline::run`] entry point.
/// The trait is object-safe, so callers can stash pipelines behind
/// `Box<dyn TypedPipeline<Input = _, Output = _>>` for heterogeneous
/// collections.
pub trait TypedPipeline {
    /// Input type consumed by [`TypedPipeline::run`].
    type Input;
    /// Output type produced by [`TypedPipeline::run`].
    type Output;

    /// Execute the pipeline end-to-end.
    ///
    /// # Errors
    ///
    /// Implementations return any applicable
    /// [`MlError`](crate::error::MlError) variant — commonly
    /// `InvalidInput`, `Preprocess`, `OnnxRuntime`, or `Postprocess`.
    fn run(&self, input: Self::Input) -> MlResult<Self::Output>;

    /// Static description of the pipeline.
    fn info(&self) -> PipelineInfo;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DoublePipeline;

    impl TypedPipeline for DoublePipeline {
        type Input = i32;
        type Output = i32;

        fn run(&self, input: Self::Input) -> MlResult<Self::Output> {
            Ok(input * 2)
        }

        fn info(&self) -> PipelineInfo {
            PipelineInfo {
                id: "test/double",
                name: "Double",
                task: PipelineTask::Custom,
                input_size: None,
            }
        }
    }

    #[test]
    fn trait_object_works() {
        let p: Box<dyn TypedPipeline<Input = i32, Output = i32>> = Box::new(DoublePipeline);
        let info = p.info();
        assert_eq!(info.id, "test/double");
        assert_eq!(p.run(21).expect("ok"), 42);
    }
}
