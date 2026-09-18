// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Job pipeline stage management.
//!
//! Provides structures for tracking multi-stage job execution pipelines,
//! including per-stage status, timing, and overall pipeline progress.

#![allow(dead_code)]

/// Status of a single pipeline stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageStatus {
    /// Waiting to start.
    Pending,
    /// Currently executing.
    Running,
    /// Finished successfully.
    Completed,
    /// Finished with an error.
    Failed,
    /// Deliberately bypassed.
    Skipped,
}

impl StageStatus {
    /// Returns `true` if the status represents a terminal (final) state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Skipped)
    }

    /// Human-readable name of the status.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
        }
    }
}

/// A single stage within a job pipeline.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Unique stage identifier within the pipeline.
    pub id: u32,
    /// Human-readable stage name.
    pub name: String,
    /// Current status.
    pub status: StageStatus,
    /// Millisecond timestamp when the stage started.
    pub started_ms: Option<u64>,
    /// Millisecond timestamp when the stage finished.
    pub finished_ms: Option<u64>,
    /// Optional failure reason.
    pub failure_reason: Option<String>,
}

impl PipelineStage {
    /// Create a new pending stage.
    #[must_use]
    pub fn new(id: u32, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            status: StageStatus::Pending,
            started_ms: None,
            finished_ms: None,
            failure_reason: None,
        }
    }

    /// Elapsed duration in milliseconds, if both start and finish are set.
    #[must_use]
    pub fn duration_ms(&self) -> Option<u64> {
        match (self.started_ms, self.finished_ms) {
            (Some(s), Some(f)) => Some(f.saturating_sub(s)),
            _ => None,
        }
    }
}

/// A pipeline that groups multiple ordered stages for a single job.
#[derive(Debug, Clone)]
pub struct JobPipeline {
    /// ID of the owning job.
    pub job_id: u64,
    /// Ordered list of stages.
    pub stages: Vec<PipelineStage>,
    /// Index of the currently active stage.
    pub current_stage: usize,
}

impl JobPipeline {
    /// Create an empty pipeline for the given job.
    #[must_use]
    pub fn new(job_id: u64) -> Self {
        Self {
            job_id,
            stages: Vec::new(),
            current_stage: 0,
        }
    }

    /// Append a new stage to the end of the pipeline.
    pub fn add_stage(&mut self, name: impl Into<String>) {
        let id = self.stages.len() as u32;
        self.stages.push(PipelineStage::new(id, name));
    }

    /// Transition the current stage from `Pending` to `Running`.
    ///
    /// Returns `false` if there is no pending stage to start.
    pub fn start_stage(&mut self) -> bool {
        if let Some(stage) = self.stages.get_mut(self.current_stage) {
            if stage.status == StageStatus::Pending {
                stage.status = StageStatus::Running;
                return true;
            }
        }
        false
    }

    /// Transition the current stage from `Running` to `Completed` and advance.
    ///
    /// Returns `false` if the current stage is not running.
    pub fn complete_stage(&mut self) -> bool {
        if let Some(stage) = self.stages.get_mut(self.current_stage) {
            if stage.status == StageStatus::Running {
                stage.status = StageStatus::Completed;
                self.current_stage += 1;
                return true;
            }
        }
        false
    }

    /// Transition the current stage from `Running` to `Failed` and record reason.
    ///
    /// Returns `false` if the current stage is not running.
    pub fn fail_stage(&mut self, reason: String) -> bool {
        if let Some(stage) = self.stages.get_mut(self.current_stage) {
            if stage.status == StageStatus::Running {
                stage.status = StageStatus::Failed;
                stage.failure_reason = Some(reason);
                return true;
            }
        }
        false
    }

    /// Skip the current pending stage without executing it.
    ///
    /// Returns `false` if the current stage is not pending.
    pub fn skip_stage(&mut self) -> bool {
        if let Some(stage) = self.stages.get_mut(self.current_stage) {
            if stage.status == StageStatus::Pending {
                stage.status = StageStatus::Skipped;
                self.current_stage += 1;
                return true;
            }
        }
        false
    }

    /// Percentage of stages that are in a terminal state (0.0–100.0).
    #[must_use]
    pub fn progress_pct(&self) -> f64 {
        if self.stages.is_empty() {
            return 100.0;
        }
        let terminal = self
            .stages
            .iter()
            .filter(|s| s.status.is_terminal())
            .count();
        (terminal as f64 / self.stages.len() as f64) * 100.0
    }

    /// Returns `true` when every stage has reached a terminal state.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        !self.stages.is_empty() && self.stages.iter().all(|s| s.status.is_terminal())
    }

    /// Returns `true` if any stage is in the `Failed` state.
    #[must_use]
    pub fn has_failed(&self) -> bool {
        self.stages.iter().any(|s| s.status == StageStatus::Failed)
    }

    /// Number of completed stages.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.stages
            .iter()
            .filter(|s| s.status == StageStatus::Completed)
            .count()
    }

    /// Number of failed stages.
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.stages
            .iter()
            .filter(|s| s.status == StageStatus::Failed)
            .count()
    }

    /// Number of skipped stages.
    #[must_use]
    pub fn skipped_count(&self) -> usize {
        self.stages
            .iter()
            .filter(|s| s.status == StageStatus::Skipped)
            .count()
    }
}

/// A reusable template for creating identically-structured pipelines.
#[derive(Debug, Clone)]
pub struct PipelineTemplate {
    /// Template name.
    pub name: String,
    /// Ordered stage names that will be instantiated.
    pub stage_names: Vec<String>,
}

impl PipelineTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stage_names: Vec::new(),
        }
    }

    /// Add a stage name to the template.
    pub fn add_stage_name(&mut self, name: impl Into<String>) {
        self.stage_names.push(name.into());
    }

    /// Instantiate a fresh [`JobPipeline`] from this template for `job_id`.
    #[must_use]
    pub fn instantiate(&self, job_id: u64) -> JobPipeline {
        let mut pipeline = JobPipeline::new(job_id);
        for stage_name in &self.stage_names {
            pipeline.add_stage(stage_name.clone());
        }
        pipeline
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pipeline(num_stages: usize) -> JobPipeline {
        let mut p = JobPipeline::new(1);
        for i in 0..num_stages {
            p.add_stage(format!("stage-{i}"));
        }
        p
    }

    #[test]
    fn test_stage_status_is_terminal() {
        assert!(!StageStatus::Pending.is_terminal());
        assert!(!StageStatus::Running.is_terminal());
        assert!(StageStatus::Completed.is_terminal());
        assert!(StageStatus::Failed.is_terminal());
        assert!(StageStatus::Skipped.is_terminal());
    }

    #[test]
    fn test_stage_status_name() {
        assert_eq!(StageStatus::Pending.name(), "pending");
        assert_eq!(StageStatus::Running.name(), "running");
        assert_eq!(StageStatus::Completed.name(), "completed");
        assert_eq!(StageStatus::Failed.name(), "failed");
        assert_eq!(StageStatus::Skipped.name(), "skipped");
    }

    #[test]
    fn test_pipeline_stage_duration_none_when_not_started() {
        let stage = PipelineStage::new(0, "init");
        assert!(stage.duration_ms().is_none());
    }

    #[test]
    fn test_pipeline_stage_duration_none_when_not_finished() {
        let mut stage = PipelineStage::new(0, "init");
        stage.started_ms = Some(1000);
        assert!(stage.duration_ms().is_none());
    }

    #[test]
    fn test_pipeline_stage_duration_some() {
        let mut stage = PipelineStage::new(0, "init");
        stage.started_ms = Some(1000);
        stage.finished_ms = Some(3000);
        assert_eq!(stage.duration_ms(), Some(2000));
    }

    #[test]
    fn test_add_stage() {
        let mut p = JobPipeline::new(42);
        p.add_stage("encode");
        p.add_stage("package");
        assert_eq!(p.stages.len(), 2);
        assert_eq!(p.stages[0].name, "encode");
        assert_eq!(p.stages[1].name, "package");
    }

    #[test]
    fn test_start_stage_success() {
        let mut p = make_pipeline(2);
        assert!(p.start_stage());
        assert_eq!(p.stages[0].status, StageStatus::Running);
    }

    #[test]
    fn test_start_stage_not_pending_fails() {
        let mut p = make_pipeline(1);
        p.start_stage();
        // Already running – should return false
        assert!(!p.start_stage());
    }

    #[test]
    fn test_complete_stage_advances_index() {
        let mut p = make_pipeline(2);
        p.start_stage();
        assert!(p.complete_stage());
        assert_eq!(p.current_stage, 1);
        assert_eq!(p.stages[0].status, StageStatus::Completed);
    }

    #[test]
    fn test_fail_stage_records_reason() {
        let mut p = make_pipeline(2);
        p.start_stage();
        assert!(p.fail_stage("disk full".to_string()));
        assert_eq!(p.stages[0].status, StageStatus::Failed);
        assert_eq!(p.stages[0].failure_reason.as_deref(), Some("disk full"));
        assert!(p.has_failed());
    }

    #[test]
    fn test_progress_pct_empty_pipeline() {
        let p = JobPipeline::new(1);
        assert_eq!(p.progress_pct(), 100.0);
    }

    #[test]
    fn test_progress_pct_partial() {
        let mut p = make_pipeline(4);
        p.start_stage();
        p.complete_stage(); // 1 done
        let pct = p.progress_pct();
        assert!((pct - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_complete() {
        let mut p = make_pipeline(2);
        assert!(!p.is_complete());
        p.start_stage();
        p.complete_stage();
        p.start_stage();
        p.complete_stage();
        assert!(p.is_complete());
    }

    #[test]
    fn test_skip_stage_advances() {
        let mut p = make_pipeline(2);
        assert!(p.skip_stage());
        assert_eq!(p.current_stage, 1);
        assert_eq!(p.stages[0].status, StageStatus::Skipped);
    }

    #[test]
    fn test_pipeline_template_instantiate() {
        let mut tmpl = PipelineTemplate::new("video-processing");
        tmpl.add_stage_name("decode");
        tmpl.add_stage_name("filter");
        tmpl.add_stage_name("encode");

        let pipeline = tmpl.instantiate(99);
        assert_eq!(pipeline.job_id, 99);
        assert_eq!(pipeline.stages.len(), 3);
        assert_eq!(pipeline.stages[2].name, "encode");
    }

    #[test]
    fn test_completed_failed_skipped_counts() {
        let mut p = make_pipeline(3);
        // Stage 0: complete
        p.start_stage();
        p.complete_stage();
        // Stage 1: fail
        p.start_stage();
        p.fail_stage("error".to_string());
        // Stage 2: skip (need to reset current_stage since fail doesn't advance)
        // After fail current_stage stays at 1, so manually move it for test
        p.current_stage = 2;
        p.skip_stage();

        assert_eq!(p.completed_count(), 1);
        assert_eq!(p.failed_count(), 1);
        assert_eq!(p.skipped_count(), 1);
    }
}
