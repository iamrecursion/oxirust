#![allow(dead_code)]
//! Stage-based ingest pipeline for MAM.
//!
//! [`IngestPipeline`] runs a sequence of [`IngestStage`]s, collects per-stage
//! [`IngestResult`]s, and exposes helpers for inspecting failures.

// ---------------------------------------------------------------------------
// IngestStage
// ---------------------------------------------------------------------------

/// Identifies a discrete step in the ingest workflow.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IngestStage {
    /// Validate that the source file exists and is readable.
    FileValidation,
    /// Unwrap the container and identify streams.
    ContainerProbe,
    /// Extract and normalise metadata (EXIF, XMP, etc.).
    MetadataExtraction,
    /// Generate proxy and thumbnail media.
    ProxyGeneration,
    /// Compute audio/video quality metrics.
    QualityAnalysis,
    /// Register the asset in the database.
    DatabaseRegistration,
    /// Notify downstream systems of the new asset.
    DownstreamNotification,
    /// A custom, user-defined stage with an arbitrary name.
    Custom(String),
}

impl IngestStage {
    /// Zero-based canonical index for built-in stages (custom stages return
    /// `usize::MAX` as a sentinel).
    #[must_use]
    pub fn stage_index(&self) -> usize {
        match self {
            Self::FileValidation => 0,
            Self::ContainerProbe => 1,
            Self::MetadataExtraction => 2,
            Self::ProxyGeneration => 3,
            Self::QualityAnalysis => 4,
            Self::DatabaseRegistration => 5,
            Self::DownstreamNotification => 6,
            Self::Custom(_) => usize::MAX,
        }
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::FileValidation => "file_validation",
            Self::ContainerProbe => "container_probe",
            Self::MetadataExtraction => "metadata_extraction",
            Self::ProxyGeneration => "proxy_generation",
            Self::QualityAnalysis => "quality_analysis",
            Self::DatabaseRegistration => "database_registration",
            Self::DownstreamNotification => "downstream_notification",
            Self::Custom(n) => n.as_str(),
        }
    }
}

// ---------------------------------------------------------------------------
// IngestResult
// ---------------------------------------------------------------------------

/// Outcome of executing a single [`IngestStage`].
#[derive(Debug, Clone)]
pub struct IngestResult {
    /// The stage this result belongs to.
    pub stage: IngestStage,
    /// `true` if the stage succeeded.
    pub success: bool,
    /// Human-readable message (error reason or success note).
    pub message: String,
    /// Optional structured data produced by the stage (e.g. JSON blob).
    pub payload: Option<String>,
}

impl IngestResult {
    /// Create a success result.
    #[must_use]
    pub fn ok(stage: IngestStage, message: impl Into<String>) -> Self {
        Self {
            stage,
            success: true,
            message: message.into(),
            payload: None,
        }
    }

    /// Create a failure result.
    #[must_use]
    pub fn err(stage: IngestStage, message: impl Into<String>) -> Self {
        Self {
            stage,
            success: false,
            message: message.into(),
            payload: None,
        }
    }

    /// Attach a payload string and return `self` for chaining.
    #[must_use]
    pub fn with_payload(mut self, payload: impl Into<String>) -> Self {
        self.payload = Some(payload.into());
        self
    }

    /// Returns `true` when all previous results indicate success AND this
    /// result itself succeeded.  Because [`IngestResult`] is used per-stage,
    /// the pipeline-level completeness check is on
    /// [`IngestPipeline::is_complete`].
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.success
    }
}

// ---------------------------------------------------------------------------
// IngestPipeline
// ---------------------------------------------------------------------------

/// Ordered sequence of stages plus a log of their results.
#[derive(Debug, Default)]
pub struct IngestPipeline {
    stages: Vec<IngestStage>,
    results: Vec<IngestResult>,
}

impl IngestPipeline {
    /// Create an empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a stage to the pipeline.
    pub fn add_stage(&mut self, stage: IngestStage) {
        self.stages.push(stage);
    }

    /// Run all stages by invoking `runner(stage)` for each.
    ///
    /// The pipeline stops at the first failure (short-circuit behaviour).
    /// Call [`Self::run_all_force`] to run every stage regardless of failures.
    pub fn run_all<F>(&mut self, mut runner: F)
    where
        F: FnMut(&IngestStage) -> IngestResult,
    {
        self.results.clear();
        for stage in &self.stages {
            let result = runner(stage);
            let failed = !result.success;
            self.results.push(result);
            if failed {
                break;
            }
        }
    }

    /// Run all stages without short-circuiting on failure.
    pub fn run_all_force<F>(&mut self, mut runner: F)
    where
        F: FnMut(&IngestStage) -> IngestResult,
    {
        self.results.clear();
        for stage in &self.stages {
            self.results.push(runner(stage));
        }
    }

    /// Returns `true` if every registered stage has a corresponding success
    /// result (i.e. the pipeline ran to completion with no failures).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.results.len() == self.stages.len() && self.results.iter().all(|r| r.success)
    }

    /// Collect references to results for stages that failed.
    #[must_use]
    pub fn failed_stages(&self) -> Vec<&IngestResult> {
        self.results.iter().filter(|r| !r.success).collect()
    }

    /// Number of stages in the pipeline.
    #[must_use]
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Number of results recorded so far.
    #[must_use]
    pub fn result_count(&self) -> usize {
        self.results.len()
    }

    /// Iterate over results.
    pub fn results(&self) -> impl Iterator<Item = &IngestResult> {
        self.results.iter()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_index_file_validation() {
        assert_eq!(IngestStage::FileValidation.stage_index(), 0);
    }

    #[test]
    fn test_stage_index_database_registration() {
        assert_eq!(IngestStage::DatabaseRegistration.stage_index(), 5);
    }

    #[test]
    fn test_stage_index_custom_sentinel() {
        assert_eq!(
            IngestStage::Custom("my_stage".into()).stage_index(),
            usize::MAX
        );
    }

    #[test]
    fn test_stage_name_file_validation() {
        assert_eq!(IngestStage::FileValidation.name(), "file_validation");
    }

    #[test]
    fn test_stage_name_custom() {
        assert_eq!(IngestStage::Custom("ocr".into()).name(), "ocr");
    }

    #[test]
    fn test_ingest_result_ok_is_complete() {
        let r = IngestResult::ok(IngestStage::ContainerProbe, "ok");
        assert!(r.is_complete());
    }

    #[test]
    fn test_ingest_result_err_not_complete() {
        let r = IngestResult::err(IngestStage::ContainerProbe, "fail");
        assert!(!r.is_complete());
    }

    #[test]
    fn test_ingest_result_with_payload() {
        let r = IngestResult::ok(IngestStage::MetadataExtraction, "ok")
            .with_payload(r#"{"duration":120}"#);
        assert_eq!(
            r.payload.expect("should succeed in test"),
            r#"{"duration":120}"#
        );
    }

    #[test]
    fn test_pipeline_run_all_success() {
        let mut p = IngestPipeline::new();
        p.add_stage(IngestStage::FileValidation);
        p.add_stage(IngestStage::ContainerProbe);
        p.run_all(|stage| IngestResult::ok(stage.clone(), "ok"));
        assert!(p.is_complete());
        assert!(p.failed_stages().is_empty());
    }

    #[test]
    fn test_pipeline_run_all_short_circuit_on_failure() {
        let mut p = IngestPipeline::new();
        p.add_stage(IngestStage::FileValidation);
        p.add_stage(IngestStage::ContainerProbe);
        p.add_stage(IngestStage::MetadataExtraction);
        p.run_all(|stage| {
            if *stage == IngestStage::ContainerProbe {
                IngestResult::err(stage.clone(), "probe failed")
            } else {
                IngestResult::ok(stage.clone(), "ok")
            }
        });
        // Should have stopped after ContainerProbe
        assert_eq!(p.result_count(), 2);
        assert!(!p.is_complete());
    }

    #[test]
    fn test_pipeline_failed_stages() {
        let mut p = IngestPipeline::new();
        p.add_stage(IngestStage::FileValidation);
        p.add_stage(IngestStage::ContainerProbe);
        p.run_all_force(|stage| {
            if *stage == IngestStage::FileValidation {
                IngestResult::err(stage.clone(), "missing")
            } else {
                IngestResult::ok(stage.clone(), "ok")
            }
        });
        let failures = p.failed_stages();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].stage, IngestStage::FileValidation);
    }

    #[test]
    fn test_pipeline_run_all_force_runs_all_stages() {
        let mut p = IngestPipeline::new();
        p.add_stage(IngestStage::FileValidation);
        p.add_stage(IngestStage::ContainerProbe);
        p.add_stage(IngestStage::ProxyGeneration);
        p.run_all_force(|stage| IngestResult::err(stage.clone(), "fail"));
        assert_eq!(p.result_count(), 3);
    }

    #[test]
    fn test_pipeline_stage_count() {
        let mut p = IngestPipeline::new();
        p.add_stage(IngestStage::FileValidation);
        p.add_stage(IngestStage::QualityAnalysis);
        assert_eq!(p.stage_count(), 2);
    }

    #[test]
    fn test_pipeline_is_complete_empty() {
        let p = IngestPipeline::new();
        // No stages and no results: vacuously complete
        assert!(p.is_complete());
    }

    #[test]
    fn test_pipeline_result_iteration() {
        let mut p = IngestPipeline::new();
        p.add_stage(IngestStage::FileValidation);
        p.add_stage(IngestStage::DatabaseRegistration);
        p.run_all(|stage| IngestResult::ok(stage.clone(), "ok"));
        let names: Vec<&str> = p.results().map(|r| r.stage.name()).collect();
        assert_eq!(names, vec!["file_validation", "database_registration"]);
    }
}
