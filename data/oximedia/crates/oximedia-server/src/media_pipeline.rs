//! Media processing pipeline: chaining operations (upload -> analyze -> transcode -> notify).
//!
//! Provides a declarative, composable pipeline for media processing.
//! Each stage receives the output of the previous stage and can produce
//! output for the next. The pipeline tracks progress and handles failures
//! at each stage independently.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// The status of a pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageStatus {
    /// Waiting to be executed.
    Pending,
    /// Currently executing.
    Running,
    /// Completed successfully.
    Completed,
    /// Failed with an error.
    Failed,
    /// Skipped (due to condition or upstream failure).
    Skipped,
}

impl StageStatus {
    /// Returns a human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
        }
    }

    /// Returns `true` if the stage is in a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Skipped)
    }
}

/// A media processing stage definition.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Stage name (e.g. "upload", "analyze", "transcode", "notify").
    pub name: String,
    /// Stage type.
    pub stage_type: StageType,
    /// Whether to continue the pipeline if this stage fails.
    pub continue_on_failure: bool,
    /// Maximum execution time for this stage.
    pub timeout: Duration,
    /// Retry configuration.
    pub retries: RetryConfig,
    /// Dependencies on other stages (by name).
    pub depends_on: Vec<String>,
}

/// The type of processing a stage performs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageType {
    /// Upload / ingest media from a source.
    Upload,
    /// Analyze media metadata (codecs, duration, resolution, etc.).
    Analyze,
    /// Transcode to target format(s).
    Transcode {
        /// Target codec.
        target_codec: String,
        /// Target quality / CRF value.
        quality: u32,
    },
    /// Generate thumbnails or preview images.
    Thumbnail {
        /// Number of thumbnails to generate.
        count: u32,
    },
    /// Validate media against quality rules.
    Validate,
    /// Send notification (webhook, email, etc.).
    Notify {
        /// Notification target URL or address.
        target: String,
    },
    /// Custom processing stage.
    Custom {
        /// Custom stage identifier.
        handler: String,
    },
}

impl StageType {
    /// Returns a label for the stage type.
    pub fn label(&self) -> &str {
        match self {
            Self::Upload => "upload",
            Self::Analyze => "analyze",
            Self::Transcode { .. } => "transcode",
            Self::Thumbnail { .. } => "thumbnail",
            Self::Validate => "validate",
            Self::Notify { .. } => "notify",
            Self::Custom { handler } => handler.as_str(),
        }
    }
}

/// Retry configuration for a pipeline stage.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Base delay between retries.
    pub base_delay: Duration,
    /// Whether to use exponential backoff.
    pub exponential_backoff: bool,
    /// Maximum delay cap.
    pub max_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_secs(1),
            exponential_backoff: true,
            max_delay: Duration::from_secs(60),
        }
    }
}

impl RetryConfig {
    /// Calculates the delay for a given attempt number.
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        if self.exponential_backoff {
            let multiplier = 2u64.saturating_pow(attempt);
            let delay = self.base_delay.saturating_mul(multiplier as u32);
            if delay > self.max_delay {
                self.max_delay
            } else {
                delay
            }
        } else {
            self.base_delay
        }
    }
}

/// Runtime state for a stage execution.
#[derive(Debug, Clone)]
pub struct StageExecution {
    /// Stage name.
    pub name: String,
    /// Current status.
    pub status: StageStatus,
    /// When execution started.
    pub started_at: Option<Instant>,
    /// When execution finished.
    pub finished_at: Option<Instant>,
    /// Number of retry attempts used.
    pub attempts: u32,
    /// Error message if failed.
    pub error: Option<String>,
    /// Stage output metadata.
    pub output: HashMap<String, String>,
}

impl StageExecution {
    /// Creates a new pending execution.
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            status: StageStatus::Pending,
            started_at: None,
            finished_at: None,
            attempts: 0,
            error: None,
            output: HashMap::new(),
        }
    }

    /// Duration of execution (or time since start if still running).
    pub fn duration(&self) -> Duration {
        match (self.started_at, self.finished_at) {
            (Some(start), Some(end)) => end.duration_since(start),
            (Some(start), None) => start.elapsed(),
            _ => Duration::ZERO,
        }
    }

    /// Marks the stage as started.
    fn start(&mut self) {
        self.status = StageStatus::Running;
        self.started_at = Some(Instant::now());
        self.attempts += 1;
    }

    /// Marks the stage as completed.
    fn complete(&mut self, output: HashMap<String, String>) {
        self.status = StageStatus::Completed;
        self.finished_at = Some(Instant::now());
        self.output = output;
    }

    /// Marks the stage as failed.
    fn fail(&mut self, error: String) {
        self.status = StageStatus::Failed;
        self.finished_at = Some(Instant::now());
        self.error = Some(error);
    }

    /// Marks the stage as skipped.
    fn skip(&mut self) {
        self.status = StageStatus::Skipped;
        self.finished_at = Some(Instant::now());
    }
}

/// Overall pipeline status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStatus {
    /// Not yet started.
    Created,
    /// Pipeline is executing stages.
    Running,
    /// All stages completed successfully.
    Completed,
    /// At least one stage failed (and continue_on_failure was false).
    Failed,
    /// Pipeline was cancelled.
    Cancelled,
}

impl PipelineStatus {
    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// A media processing pipeline instance.
pub struct MediaPipeline {
    /// Pipeline ID.
    pub id: String,
    /// Pipeline name/description.
    pub name: String,
    /// Ordered stages.
    stages: Vec<PipelineStage>,
    /// Stage execution state.
    executions: HashMap<String, StageExecution>,
    /// Pipeline-level metadata (e.g. media_id, user_id).
    pub metadata: HashMap<String, String>,
    /// Overall status.
    status: PipelineStatus,
    /// When the pipeline was created.
    created_at: Instant,
    /// When execution started.
    started_at: Option<Instant>,
    /// When execution finished.
    finished_at: Option<Instant>,
}

impl MediaPipeline {
    /// Creates a new pipeline.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            stages: Vec::new(),
            executions: HashMap::new(),
            metadata: HashMap::new(),
            status: PipelineStatus::Created,
            created_at: Instant::now(),
            started_at: None,
            finished_at: None,
        }
    }

    /// Adds a stage to the pipeline.
    pub fn add_stage(&mut self, stage: PipelineStage) {
        let exec = StageExecution::new(&stage.name);
        self.executions.insert(stage.name.clone(), exec);
        self.stages.push(stage);
    }

    /// Returns the current pipeline status.
    pub fn status(&self) -> PipelineStatus {
        self.status
    }

    /// Returns the number of stages.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Returns all stage names in order.
    pub fn stage_names(&self) -> Vec<&str> {
        self.stages.iter().map(|s| s.name.as_str()).collect()
    }

    /// Gets the execution state for a specific stage.
    pub fn stage_execution(&self, name: &str) -> Option<&StageExecution> {
        self.executions.get(name)
    }

    /// Returns the number of completed stages.
    pub fn completed_count(&self) -> usize {
        self.executions
            .values()
            .filter(|e| e.status == StageStatus::Completed)
            .count()
    }

    /// Returns the number of failed stages.
    pub fn failed_count(&self) -> usize {
        self.executions
            .values()
            .filter(|e| e.status == StageStatus::Failed)
            .count()
    }

    /// Returns progress as a fraction (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        if self.stages.is_empty() {
            return 1.0;
        }
        let terminal = self
            .executions
            .values()
            .filter(|e| e.status.is_terminal())
            .count();
        terminal as f64 / self.stages.len() as f64
    }

    /// Starts the pipeline execution.
    pub fn start(&mut self) {
        self.status = PipelineStatus::Running;
        self.started_at = Some(Instant::now());
    }

    /// Starts a specific stage.
    pub fn start_stage(&mut self, name: &str) -> bool {
        if let Some(exec) = self.executions.get_mut(name) {
            exec.start();
            true
        } else {
            false
        }
    }

    /// Completes a stage with output.
    pub fn complete_stage(&mut self, name: &str, output: HashMap<String, String>) -> bool {
        if let Some(exec) = self.executions.get_mut(name) {
            exec.complete(output);
            self.maybe_update_pipeline_status();
            true
        } else {
            false
        }
    }

    /// Fails a stage.
    pub fn fail_stage(&mut self, name: &str, error: &str) -> bool {
        let continue_on_failure = self
            .stages
            .iter()
            .find(|s| s.name == name)
            .map(|s| s.continue_on_failure)
            .unwrap_or(false);

        if let Some(exec) = self.executions.get_mut(name) {
            exec.fail(error.to_string());

            if !continue_on_failure {
                // Skip remaining stages
                self.skip_remaining_stages(name);
                self.status = PipelineStatus::Failed;
                self.finished_at = Some(Instant::now());
            } else {
                self.maybe_update_pipeline_status();
            }
            true
        } else {
            false
        }
    }

    /// Skips a stage.
    pub fn skip_stage(&mut self, name: &str) -> bool {
        if let Some(exec) = self.executions.get_mut(name) {
            exec.skip();
            self.maybe_update_pipeline_status();
            true
        } else {
            false
        }
    }

    /// Cancels the pipeline.
    pub fn cancel(&mut self) {
        self.status = PipelineStatus::Cancelled;
        self.finished_at = Some(Instant::now());
        for exec in self.executions.values_mut() {
            if !exec.status.is_terminal() {
                exec.skip();
            }
        }
    }

    /// Total wall-clock time.
    pub fn total_duration(&self) -> Duration {
        match (self.started_at, self.finished_at) {
            (Some(start), Some(end)) => end.duration_since(start),
            (Some(start), None) => start.elapsed(),
            _ => Duration::ZERO,
        }
    }

    /// Returns the next stage to execute (first pending stage whose deps are met).
    pub fn next_stage(&self) -> Option<&PipelineStage> {
        for stage in &self.stages {
            if let Some(exec) = self.executions.get(&stage.name) {
                if exec.status != StageStatus::Pending {
                    continue;
                }
            }
            // Check dependencies
            let deps_met = stage.depends_on.iter().all(|dep| {
                self.executions
                    .get(dep)
                    .map(|e| e.status == StageStatus::Completed)
                    .unwrap_or(false)
            });
            if deps_met {
                return Some(stage);
            }
        }
        None
    }

    /// Returns a summary of all stage executions.
    pub fn execution_summary(&self) -> Vec<(&str, StageStatus, Duration)> {
        self.stages
            .iter()
            .filter_map(|stage| {
                self.executions
                    .get(&stage.name)
                    .map(|exec| (stage.name.as_str(), exec.status, exec.duration()))
            })
            .collect()
    }

    // ── Internal helpers ──

    fn skip_remaining_stages(&mut self, failed_stage: &str) {
        let mut found = false;
        for stage in &self.stages {
            if stage.name == failed_stage {
                found = true;
                continue;
            }
            if found {
                if let Some(exec) = self.executions.get_mut(&stage.name) {
                    if !exec.status.is_terminal() {
                        exec.skip();
                    }
                }
            }
        }
    }

    fn maybe_update_pipeline_status(&mut self) {
        let all_terminal = self.executions.values().all(|e| e.status.is_terminal());
        if all_terminal && self.status == PipelineStatus::Running {
            let has_failure = self
                .executions
                .values()
                .any(|e| e.status == StageStatus::Failed);
            self.status = if has_failure {
                PipelineStatus::Failed
            } else {
                PipelineStatus::Completed
            };
            self.finished_at = Some(Instant::now());
        }
    }
}

/// Builder for common pipeline templates.
pub struct PipelineBuilder;

impl PipelineBuilder {
    /// Creates a standard upload-analyze-transcode-notify pipeline.
    pub fn upload_to_notify(
        pipeline_id: &str,
        target_codec: &str,
        quality: u32,
        notify_url: &str,
    ) -> MediaPipeline {
        let mut pipeline = MediaPipeline::new(pipeline_id, "Upload to Notify");

        pipeline.add_stage(PipelineStage {
            name: "upload".to_string(),
            stage_type: StageType::Upload,
            continue_on_failure: false,
            timeout: Duration::from_secs(3600),
            retries: RetryConfig {
                max_retries: 2,
                ..Default::default()
            },
            depends_on: vec![],
        });

        pipeline.add_stage(PipelineStage {
            name: "analyze".to_string(),
            stage_type: StageType::Analyze,
            continue_on_failure: false,
            timeout: Duration::from_secs(120),
            retries: RetryConfig::default(),
            depends_on: vec!["upload".to_string()],
        });

        pipeline.add_stage(PipelineStage {
            name: "transcode".to_string(),
            stage_type: StageType::Transcode {
                target_codec: target_codec.to_string(),
                quality,
            },
            continue_on_failure: false,
            timeout: Duration::from_secs(7200),
            retries: RetryConfig {
                max_retries: 1,
                ..Default::default()
            },
            depends_on: vec!["analyze".to_string()],
        });

        pipeline.add_stage(PipelineStage {
            name: "thumbnail".to_string(),
            stage_type: StageType::Thumbnail { count: 5 },
            continue_on_failure: true,
            timeout: Duration::from_secs(120),
            retries: RetryConfig::default(),
            depends_on: vec!["transcode".to_string()],
        });

        pipeline.add_stage(PipelineStage {
            name: "notify".to_string(),
            stage_type: StageType::Notify {
                target: notify_url.to_string(),
            },
            continue_on_failure: true,
            timeout: Duration::from_secs(30),
            retries: RetryConfig {
                max_retries: 5,
                ..Default::default()
            },
            depends_on: vec!["transcode".to_string()],
        });

        pipeline
    }

    /// Creates a simple analyze-only pipeline.
    pub fn analyze_only(pipeline_id: &str) -> MediaPipeline {
        let mut pipeline = MediaPipeline::new(pipeline_id, "Analyze Only");
        pipeline.add_stage(PipelineStage {
            name: "analyze".to_string(),
            stage_type: StageType::Analyze,
            continue_on_failure: false,
            timeout: Duration::from_secs(120),
            retries: RetryConfig::default(),
            depends_on: vec![],
        });
        pipeline
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // StageStatus

    #[test]
    fn test_stage_status_labels() {
        assert_eq!(StageStatus::Pending.label(), "pending");
        assert_eq!(StageStatus::Running.label(), "running");
        assert_eq!(StageStatus::Completed.label(), "completed");
        assert_eq!(StageStatus::Failed.label(), "failed");
        assert_eq!(StageStatus::Skipped.label(), "skipped");
    }

    #[test]
    fn test_stage_status_is_terminal() {
        assert!(!StageStatus::Pending.is_terminal());
        assert!(!StageStatus::Running.is_terminal());
        assert!(StageStatus::Completed.is_terminal());
        assert!(StageStatus::Failed.is_terminal());
        assert!(StageStatus::Skipped.is_terminal());
    }

    // StageType

    #[test]
    fn test_stage_type_labels() {
        assert_eq!(StageType::Upload.label(), "upload");
        assert_eq!(StageType::Analyze.label(), "analyze");
        assert_eq!(
            StageType::Transcode {
                target_codec: "av1".into(),
                quality: 30
            }
            .label(),
            "transcode"
        );
    }

    // RetryConfig

    #[test]
    fn test_retry_delay_exponential() {
        let cfg = RetryConfig::default();
        assert_eq!(cfg.delay_for_attempt(0), Duration::from_secs(1));
        assert_eq!(cfg.delay_for_attempt(1), Duration::from_secs(2));
        assert_eq!(cfg.delay_for_attempt(2), Duration::from_secs(4));
    }

    #[test]
    fn test_retry_delay_capped() {
        let cfg = RetryConfig {
            max_delay: Duration::from_secs(5),
            ..Default::default()
        };
        assert_eq!(cfg.delay_for_attempt(10), Duration::from_secs(5));
    }

    #[test]
    fn test_retry_delay_linear() {
        let cfg = RetryConfig {
            exponential_backoff: false,
            base_delay: Duration::from_secs(3),
            ..Default::default()
        };
        assert_eq!(cfg.delay_for_attempt(0), Duration::from_secs(3));
        assert_eq!(cfg.delay_for_attempt(5), Duration::from_secs(3));
    }

    // PipelineStatus

    #[test]
    fn test_pipeline_status_labels() {
        assert_eq!(PipelineStatus::Created.label(), "created");
        assert_eq!(PipelineStatus::Running.label(), "running");
        assert_eq!(PipelineStatus::Completed.label(), "completed");
        assert_eq!(PipelineStatus::Failed.label(), "failed");
        assert_eq!(PipelineStatus::Cancelled.label(), "cancelled");
    }

    // MediaPipeline

    #[test]
    fn test_pipeline_creation() {
        let pipeline = MediaPipeline::new("p1", "Test Pipeline");
        assert_eq!(pipeline.id, "p1");
        assert_eq!(pipeline.status(), PipelineStatus::Created);
        assert_eq!(pipeline.stage_count(), 0);
    }

    #[test]
    fn test_add_stage() {
        let mut pipeline = MediaPipeline::new("p1", "Test");
        pipeline.add_stage(PipelineStage {
            name: "upload".to_string(),
            stage_type: StageType::Upload,
            continue_on_failure: false,
            timeout: Duration::from_secs(60),
            retries: RetryConfig::default(),
            depends_on: vec![],
        });
        assert_eq!(pipeline.stage_count(), 1);
        assert_eq!(pipeline.stage_names(), vec!["upload"]);
    }

    #[test]
    fn test_pipeline_execution_flow() {
        let mut pipeline = PipelineBuilder::analyze_only("p1");
        pipeline.start();
        assert_eq!(pipeline.status(), PipelineStatus::Running);

        // Find next stage
        let next = pipeline.next_stage().map(|s| s.name.clone());
        assert_eq!(next, Some("analyze".to_string()));

        // Start and complete
        pipeline.start_stage("analyze");
        let exec = pipeline.stage_execution("analyze").expect("should exist");
        assert_eq!(exec.status, StageStatus::Running);

        pipeline.complete_stage("analyze", HashMap::new());
        assert_eq!(pipeline.status(), PipelineStatus::Completed);
        assert!((pipeline.progress() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pipeline_failure_skips_remaining() {
        let mut pipeline =
            PipelineBuilder::upload_to_notify("p1", "av1", 30, "http://example.com/hook");
        pipeline.start();

        // Complete upload
        pipeline.start_stage("upload");
        pipeline.complete_stage("upload", HashMap::new());

        // Fail analyze (continue_on_failure=false)
        pipeline.start_stage("analyze");
        pipeline.fail_stage("analyze", "codec not supported");

        assert_eq!(pipeline.status(), PipelineStatus::Failed);
        // Remaining stages should be skipped
        let exec = pipeline.stage_execution("transcode").expect("should exist");
        assert_eq!(exec.status, StageStatus::Skipped);
    }

    #[test]
    fn test_pipeline_continue_on_failure() {
        let mut pipeline = MediaPipeline::new("p1", "Test");
        pipeline.add_stage(PipelineStage {
            name: "optional".to_string(),
            stage_type: StageType::Validate,
            continue_on_failure: true,
            timeout: Duration::from_secs(30),
            retries: RetryConfig::default(),
            depends_on: vec![],
        });
        pipeline.add_stage(PipelineStage {
            name: "next".to_string(),
            stage_type: StageType::Analyze,
            continue_on_failure: false,
            timeout: Duration::from_secs(30),
            retries: RetryConfig::default(),
            depends_on: vec![],
        });

        pipeline.start();
        pipeline.start_stage("optional");
        pipeline.fail_stage("optional", "validation warning");

        // Pipeline should still be running
        assert_eq!(pipeline.status(), PipelineStatus::Running);
        // "next" should still be pending
        let exec = pipeline.stage_execution("next").expect("should exist");
        assert_eq!(exec.status, StageStatus::Pending);
    }

    #[test]
    fn test_pipeline_cancel() {
        let mut pipeline =
            PipelineBuilder::upload_to_notify("p1", "av1", 30, "http://hook.example");
        pipeline.start();
        pipeline.cancel();
        assert_eq!(pipeline.status(), PipelineStatus::Cancelled);
        // All stages should be skipped
        for exec in pipeline.executions.values() {
            assert!(exec.status.is_terminal());
        }
    }

    #[test]
    fn test_pipeline_progress() {
        let mut pipeline =
            PipelineBuilder::upload_to_notify("p1", "av1", 30, "http://hook.example");
        pipeline.start();
        assert!((pipeline.progress() - 0.0).abs() < 1e-9);

        pipeline.start_stage("upload");
        pipeline.complete_stage("upload", HashMap::new());
        // 1 of 5 completed
        assert!((pipeline.progress() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_pipeline_next_stage_respects_deps() {
        let mut pipeline =
            PipelineBuilder::upload_to_notify("p1", "av1", 30, "http://hook.example");
        pipeline.start();

        // Only "upload" should be available (no deps)
        let next = pipeline.next_stage().expect("should have next");
        assert_eq!(next.name, "upload");
    }

    #[test]
    fn test_pipeline_next_stage_after_dep_completed() {
        let mut pipeline =
            PipelineBuilder::upload_to_notify("p1", "av1", 30, "http://hook.example");
        pipeline.start();

        pipeline.start_stage("upload");
        pipeline.complete_stage("upload", HashMap::new());

        let next = pipeline.next_stage().expect("should have next");
        assert_eq!(next.name, "analyze");
    }

    #[test]
    fn test_stage_execution_duration() {
        let mut exec = StageExecution::new("test");
        exec.start();
        std::thread::sleep(Duration::from_millis(5));
        exec.complete(HashMap::new());
        assert!(exec.duration() >= Duration::from_millis(4));
    }

    #[test]
    fn test_execution_summary() {
        let mut pipeline = PipelineBuilder::analyze_only("p1");
        pipeline.start();
        pipeline.start_stage("analyze");
        pipeline.complete_stage("analyze", HashMap::new());
        let summary = pipeline.execution_summary();
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].0, "analyze");
        assert_eq!(summary[0].1, StageStatus::Completed);
    }

    #[test]
    fn test_completed_and_failed_counts() {
        let mut pipeline = MediaPipeline::new("p1", "Test");
        pipeline.add_stage(PipelineStage {
            name: "a".to_string(),
            stage_type: StageType::Analyze,
            continue_on_failure: true,
            timeout: Duration::from_secs(30),
            retries: RetryConfig::default(),
            depends_on: vec![],
        });
        pipeline.add_stage(PipelineStage {
            name: "b".to_string(),
            stage_type: StageType::Validate,
            continue_on_failure: true,
            timeout: Duration::from_secs(30),
            retries: RetryConfig::default(),
            depends_on: vec![],
        });
        pipeline.start();
        pipeline.start_stage("a");
        pipeline.complete_stage("a", HashMap::new());
        pipeline.start_stage("b");
        pipeline.fail_stage("b", "error");
        assert_eq!(pipeline.completed_count(), 1);
        assert_eq!(pipeline.failed_count(), 1);
    }

    #[test]
    fn test_pipeline_builder_upload_to_notify() {
        let pipeline = PipelineBuilder::upload_to_notify("p1", "av1", 30, "http://hook.example");
        assert_eq!(pipeline.stage_count(), 5);
        assert_eq!(
            pipeline.stage_names(),
            vec!["upload", "analyze", "transcode", "thumbnail", "notify"]
        );
    }

    #[test]
    fn test_pipeline_metadata() {
        let mut pipeline = MediaPipeline::new("p1", "Test");
        pipeline
            .metadata
            .insert("media_id".to_string(), "m-42".to_string());
        assert_eq!(pipeline.metadata.get("media_id"), Some(&"m-42".to_string()));
    }

    #[test]
    fn test_stage_output_propagation() {
        let mut pipeline = PipelineBuilder::analyze_only("p1");
        pipeline.start();
        pipeline.start_stage("analyze");

        let mut output = HashMap::new();
        output.insert("codec".to_string(), "av1".to_string());
        output.insert("duration".to_string(), "120.5".to_string());
        pipeline.complete_stage("analyze", output);

        let exec = pipeline.stage_execution("analyze").expect("should exist");
        assert_eq!(exec.output.get("codec"), Some(&"av1".to_string()));
    }

    #[test]
    fn test_skip_stage() {
        let mut pipeline = PipelineBuilder::analyze_only("p1");
        pipeline.start();
        pipeline.skip_stage("analyze");
        let exec = pipeline.stage_execution("analyze").expect("should exist");
        assert_eq!(exec.status, StageStatus::Skipped);
    }
}
