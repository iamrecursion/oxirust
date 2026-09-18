//! Approval workflow engine.

use crate::{
    approval::{stage::ApprovalStage, ApprovalId},
    error::ReviewResult,
    SessionId,
};
use serde::{Deserialize, Serialize};

/// Approval workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalWorkflow {
    /// Workflow ID.
    pub id: ApprovalId,
    /// Session ID.
    pub session_id: SessionId,
    /// Workflow name.
    pub name: String,
    /// Workflow stages.
    pub stages: Vec<ApprovalStage>,
    /// Current stage index.
    pub current_stage: usize,
    /// Workflow status.
    pub status: WorkflowStatus,
    /// Whether workflow is sequential or parallel.
    pub sequential: bool,
}

/// Workflow status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkflowStatus {
    /// Workflow is not started.
    NotStarted,
    /// Workflow is in progress.
    InProgress,
    /// Workflow is completed (all stages approved).
    Completed,
    /// Workflow is rejected (any stage rejected).
    Rejected,
    /// Workflow is cancelled.
    Cancelled,
}

impl ApprovalWorkflow {
    /// Create a new workflow.
    #[must_use]
    pub fn new(session_id: SessionId, name: String, sequential: bool) -> Self {
        Self {
            id: ApprovalId::new(),
            session_id,
            name,
            stages: Vec::new(),
            current_stage: 0,
            status: WorkflowStatus::NotStarted,
            sequential,
        }
    }

    /// Add a stage to the workflow.
    pub fn add_stage(&mut self, stage: ApprovalStage) {
        self.stages.push(stage);
    }

    /// Start the workflow.
    pub fn start(&mut self) {
        self.status = WorkflowStatus::InProgress;
    }

    /// Get the current stage.
    #[must_use]
    pub fn current_stage(&self) -> Option<&ApprovalStage> {
        self.stages.get(self.current_stage)
    }

    /// Advance to the next stage.
    ///
    /// # Errors
    ///
    /// Returns error if cannot advance.
    pub fn advance(&mut self) -> ReviewResult<()> {
        if self.current_stage + 1 < self.stages.len() {
            self.current_stage += 1;
            Ok(())
        } else {
            self.status = WorkflowStatus::Completed;
            Ok(())
        }
    }

    /// Mark workflow as rejected.
    pub fn reject(&mut self) {
        self.status = WorkflowStatus::Rejected;
    }

    /// Check if workflow is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.status == WorkflowStatus::Completed
    }

    /// Get progress percentage.
    #[must_use]
    pub fn progress_percentage(&self) -> f64 {
        if self.stages.is_empty() {
            return 0.0;
        }

        let completed_stages = self.stages.iter().filter(|s| s.is_complete()).count();
        (completed_stages as f64 / self.stages.len() as f64) * 100.0
    }
}

/// Create a simple approval workflow.
///
/// # Errors
///
/// Returns error if creation fails.
pub async fn create_simple_workflow(
    session_id: SessionId,
    name: String,
) -> ReviewResult<ApprovalWorkflow> {
    Ok(ApprovalWorkflow::new(session_id, name, true))
}

/// Create a multi-stage workflow.
///
/// # Errors
///
/// Returns error if creation fails.
pub async fn create_multistage_workflow(
    session_id: SessionId,
    name: String,
    stage_names: Vec<String>,
) -> ReviewResult<ApprovalWorkflow> {
    let mut workflow = ApprovalWorkflow::new(session_id, name, true);

    for (index, stage_name) in stage_names.iter().enumerate() {
        let stage = ApprovalStage::new(index, stage_name.clone());
        workflow.add_stage(stage);
    }

    Ok(workflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_creation() {
        let session_id = SessionId::new();
        let workflow = ApprovalWorkflow::new(session_id, "Test Workflow".to_string(), true);

        assert_eq!(workflow.status, WorkflowStatus::NotStarted);
        assert!(workflow.sequential);
        assert_eq!(workflow.stages.len(), 0);
    }

    #[test]
    fn test_workflow_start() {
        let session_id = SessionId::new();
        let mut workflow = ApprovalWorkflow::new(session_id, "Test".to_string(), true);

        workflow.start();
        assert_eq!(workflow.status, WorkflowStatus::InProgress);
    }

    #[test]
    fn test_workflow_advance() {
        let session_id = SessionId::new();
        let mut workflow = ApprovalWorkflow::new(session_id, "Test".to_string(), true);

        workflow.add_stage(ApprovalStage::new(0, "Stage 1".to_string()));
        workflow.add_stage(ApprovalStage::new(1, "Stage 2".to_string()));

        assert_eq!(workflow.current_stage, 0);

        workflow.advance().expect("should succeed in test");
        assert_eq!(workflow.current_stage, 1);

        workflow.advance().expect("should succeed in test");
        assert_eq!(workflow.status, WorkflowStatus::Completed);
    }

    #[test]
    fn test_workflow_progress() {
        let session_id = SessionId::new();
        let mut workflow = ApprovalWorkflow::new(session_id, "Test".to_string(), true);

        workflow.add_stage(ApprovalStage::new(0, "Stage 1".to_string()));
        workflow.add_stage(ApprovalStage::new(1, "Stage 2".to_string()));

        assert!((workflow.progress_percentage() - 0.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_create_multistage_workflow() {
        let session_id = SessionId::new();
        let stages = vec![
            "Review".to_string(),
            "Approval".to_string(),
            "Final".to_string(),
        ];

        let workflow = create_multistage_workflow(session_id, "Test".to_string(), stages)
            .await
            .expect("should succeed in test");

        assert_eq!(workflow.stages.len(), 3);
    }
}
