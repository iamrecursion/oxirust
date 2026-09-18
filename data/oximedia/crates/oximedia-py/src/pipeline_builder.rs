//! Python-friendly pipeline builder.
//!
//! Provides a simple, serialisable description of a media-processing pipeline
//! that can be constructed in Python and then handed to the Rust processing
//! engine for execution.

#![allow(dead_code)]

/// A stage in a media-processing pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelineStage {
    /// Decode compressed bitstream into raw frames.
    Decode,
    /// Rescale video resolution.
    Scale,
    /// Convert pixel colour space or range.
    ColorConvert,
    /// Encode raw frames into a compressed bitstream.
    Encode,
    /// Multiplex audio and video into a container.
    Mux,
}

impl PipelineStage {
    /// Returns a short human-readable description of the stage.
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            Self::Decode => "Decode compressed bitstream into raw frames",
            Self::Scale => "Rescale video resolution",
            Self::ColorConvert => "Convert pixel colour space or range",
            Self::Encode => "Encode raw frames into a compressed bitstream",
            Self::Mux => "Multiplex audio and video into a container",
        }
    }

    /// Returns `true` if this stage involves a codec (decode or encode).
    #[must_use]
    pub fn is_codec_stage(&self) -> bool {
        matches!(self, Self::Decode | Self::Encode)
    }
}

/// A single parameterised step in a pipeline.
#[derive(Clone, Debug)]
pub struct PyPipelineStep {
    /// The processing stage this step belongs to.
    pub stage: PipelineStage,
    /// Key-value parameters for the step (e.g. `[("crf", "28")]`).
    pub params: Vec<(String, String)>,
}

impl PyPipelineStep {
    /// Creates a new step for the given stage with no parameters.
    #[must_use]
    pub fn new(stage: PipelineStage) -> Self {
        Self {
            stage,
            params: Vec::new(),
        }
    }

    /// Adds a key-value parameter to this step.
    pub fn add_param(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.params.push((key.into(), value.into()));
    }

    /// Returns the value associated with `key`, or `None` if not found.
    #[must_use]
    pub fn get_param(&self, key: &str) -> Option<&str> {
        self.params
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// A named sequence of pipeline steps.
#[derive(Clone, Debug)]
pub struct PyPipeline {
    /// Ordered list of processing steps.
    pub steps: Vec<PyPipelineStep>,
    /// Human-readable name for this pipeline.
    pub name: String,
}

impl PyPipeline {
    /// Creates a new, empty pipeline with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            steps: Vec::new(),
            name: name.into(),
        }
    }

    /// Appends a step to the pipeline.
    pub fn add_step(&mut self, step: PyPipelineStep) {
        self.steps.push(step);
    }

    /// Returns the number of steps in the pipeline.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Returns `true` if the pipeline contains at least one step with the given stage.
    #[must_use]
    pub fn has_stage(&self, stage: &PipelineStage) -> bool {
        self.steps.iter().any(|s| &s.stage == stage)
    }

    /// Returns a human-readable summary of the pipeline.
    #[must_use]
    pub fn describe(&self) -> String {
        if self.steps.is_empty() {
            return format!("Pipeline '{}': (empty)", self.name);
        }
        let stage_names: Vec<&str> = self.steps.iter().map(|s| s.stage.description()).collect();
        format!("Pipeline '{}': {}", self.name, stage_names.join(" → "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── PipelineStage ──────────────────────────────────────────────────────

    #[test]
    fn test_decode_is_codec_stage() {
        assert!(PipelineStage::Decode.is_codec_stage());
    }

    #[test]
    fn test_encode_is_codec_stage() {
        assert!(PipelineStage::Encode.is_codec_stage());
    }

    #[test]
    fn test_scale_is_not_codec_stage() {
        assert!(!PipelineStage::Scale.is_codec_stage());
    }

    #[test]
    fn test_mux_is_not_codec_stage() {
        assert!(!PipelineStage::Mux.is_codec_stage());
    }

    #[test]
    fn test_color_convert_is_not_codec_stage() {
        assert!(!PipelineStage::ColorConvert.is_codec_stage());
    }

    #[test]
    fn test_description_non_empty() {
        for stage in [
            PipelineStage::Decode,
            PipelineStage::Scale,
            PipelineStage::ColorConvert,
            PipelineStage::Encode,
            PipelineStage::Mux,
        ] {
            assert!(!stage.description().is_empty());
        }
    }

    // ── PyPipelineStep ──────────────────────────────────────────────────────

    #[test]
    fn test_new_step_no_params() {
        let step = PyPipelineStep::new(PipelineStage::Decode);
        assert!(step.params.is_empty());
    }

    #[test]
    fn test_add_param_stores_value() {
        let mut step = PyPipelineStep::new(PipelineStage::Encode);
        step.add_param("crf", "28");
        assert_eq!(step.get_param("crf"), Some("28"));
    }

    #[test]
    fn test_get_param_missing_returns_none() {
        let step = PyPipelineStep::new(PipelineStage::Scale);
        assert!(step.get_param("width").is_none());
    }

    #[test]
    fn test_multiple_params() {
        let mut step = PyPipelineStep::new(PipelineStage::Scale);
        step.add_param("width", "1280");
        step.add_param("height", "720");
        assert_eq!(step.get_param("width"), Some("1280"));
        assert_eq!(step.get_param("height"), Some("720"));
    }

    #[test]
    fn test_stage_stored_correctly() {
        let step = PyPipelineStep::new(PipelineStage::Mux);
        assert_eq!(step.stage, PipelineStage::Mux);
    }

    // ── PyPipeline ──────────────────────────────────────────────────────────

    #[test]
    fn test_new_pipeline_empty() {
        let p = PyPipeline::new("test");
        assert_eq!(p.step_count(), 0);
    }

    #[test]
    fn test_pipeline_name_stored() {
        let p = PyPipeline::new("my_pipeline");
        assert_eq!(p.name, "my_pipeline");
    }

    #[test]
    fn test_add_step_increments_count() {
        let mut p = PyPipeline::new("p");
        p.add_step(PyPipelineStep::new(PipelineStage::Decode));
        assert_eq!(p.step_count(), 1);
        p.add_step(PyPipelineStep::new(PipelineStage::Encode));
        assert_eq!(p.step_count(), 2);
    }

    #[test]
    fn test_has_stage_true() {
        let mut p = PyPipeline::new("p");
        p.add_step(PyPipelineStep::new(PipelineStage::Decode));
        assert!(p.has_stage(&PipelineStage::Decode));
    }

    #[test]
    fn test_has_stage_false() {
        let mut p = PyPipeline::new("p");
        p.add_step(PyPipelineStep::new(PipelineStage::Decode));
        assert!(!p.has_stage(&PipelineStage::Mux));
    }

    #[test]
    fn test_describe_empty_pipeline() {
        let p = PyPipeline::new("empty");
        let desc = p.describe();
        assert!(desc.contains("empty"));
        assert!(desc.contains("empty)") || desc.contains("(empty)"));
    }

    #[test]
    fn test_describe_with_steps() {
        let mut p = PyPipeline::new("transcode");
        p.add_step(PyPipelineStep::new(PipelineStage::Decode));
        p.add_step(PyPipelineStep::new(PipelineStage::Encode));
        let desc = p.describe();
        assert!(desc.contains("transcode"));
        assert!(desc.contains("→"));
    }

    #[test]
    fn test_pipeline_string_name() {
        let name = String::from("dynamic_pipeline");
        let p = PyPipeline::new(name);
        assert_eq!(p.name, "dynamic_pipeline");
    }
}
