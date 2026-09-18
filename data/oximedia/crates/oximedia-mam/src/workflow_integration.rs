//! MAM workflow integration layer.
//!
//! Provides workflow templates, instances, and a lightweight in-process
//! workflow engine for orchestrating asset processing pipelines.

/// Event that triggers a workflow to start automatically.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum WorkflowTrigger {
    /// Triggered when a new asset is ingested.
    OnIngest,
    /// Triggered when an asset passes QC.
    OnQcPass,
    /// Triggered when an asset fails QC.
    OnQcFail,
    /// Triggered when an asset receives manual approval.
    OnApproval,
    /// Triggered on a schedule (cron-like).
    Scheduled,
    /// Triggered manually by a user.
    Manual,
}

impl WorkflowTrigger {
    /// Returns `true` for triggers that fire without user intervention.
    #[must_use]
    pub fn is_automatic(&self) -> bool {
        matches!(
            self,
            Self::OnIngest | Self::OnQcPass | Self::OnQcFail | Self::Scheduled
        )
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::OnIngest => "On Ingest",
            Self::OnQcPass => "On QC Pass",
            Self::OnQcFail => "On QC Fail",
            Self::OnApproval => "On Approval",
            Self::Scheduled => "Scheduled",
            Self::Manual => "Manual",
        }
    }
}

/// Template that defines the structure of a reusable workflow.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct WorkflowTemplate {
    /// Unique numeric identifier for the template.
    pub id: u64,
    /// Human-readable name.
    pub name: String,
    /// Event that activates this template automatically.
    pub trigger: WorkflowTrigger,
    /// Ordered list of step names.
    pub steps: Vec<String>,
}

impl WorkflowTemplate {
    /// Create a new workflow template.
    #[must_use]
    pub fn new(
        id: u64,
        name: impl Into<String>,
        trigger: WorkflowTrigger,
        steps: Vec<String>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            trigger,
            steps,
        }
    }

    /// Returns the number of steps in this template.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
}

/// Runtime status of a workflow instance.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum WfStatus {
    /// Workflow is actively executing.
    Running,
    /// All steps completed successfully.
    Completed,
    /// Workflow encountered an error.
    Failed,
    /// Workflow has been paused awaiting intervention.
    Paused,
    /// Workflow was cancelled before completion.
    Cancelled,
}

impl WfStatus {
    /// Returns `true` for statuses from which no further execution occurs.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// A running or finished instance of a workflow template applied to one asset.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct WorkflowInstance {
    /// ID of the template this instance was created from.
    pub template_id: u64,
    /// The asset being processed by this instance.
    pub asset_id: String,
    /// Millisecond timestamp when execution began.
    pub started_ms: u64,
    /// Zero-based index of the step currently being executed.
    pub current_step: usize,
    /// Current execution status.
    pub status: WfStatus,
}

impl WorkflowInstance {
    /// Create a new running workflow instance.
    #[must_use]
    pub fn new(template_id: u64, asset_id: impl Into<String>, started_ms: u64) -> Self {
        Self {
            template_id,
            asset_id: asset_id.into(),
            started_ms,
            current_step: 0,
            status: WfStatus::Running,
        }
    }

    /// Percentage of steps completed, given the template's total step count.
    ///
    /// Returns `0.0` when `total_steps` is zero.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn progress_pct(&self, total_steps: usize) -> f64 {
        if total_steps == 0 {
            return 0.0;
        }
        (self.current_step as f64 / total_steps as f64 * 100.0).clamp(0.0, 100.0)
    }

    /// Returns `true` while the instance is in `Running` status.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.status == WfStatus::Running
    }
}

/// In-process workflow engine that manages templates and running instances.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct WorkflowEngine {
    /// Registered workflow templates.
    pub templates: Vec<WorkflowTemplate>,
    /// All workflow instances (running, completed, etc.).
    pub instances: Vec<WorkflowInstance>,
}

impl WorkflowEngine {
    /// Create a new, empty workflow engine.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a workflow template with the engine.
    pub fn register_template(&mut self, template: WorkflowTemplate) {
        self.templates.push(template);
    }

    /// Start a new workflow instance for the given template and asset.
    ///
    /// Returns the index of the new instance, or `None` if the template is
    /// not found.
    pub fn start(&mut self, template_id: u64, asset_id: &str, now_ms: u64) -> Option<usize> {
        if !self.templates.iter().any(|t| t.id == template_id) {
            return None;
        }
        let instance = WorkflowInstance::new(template_id, asset_id, now_ms);
        let idx = self.instances.len();
        self.instances.push(instance);
        Some(idx)
    }

    /// Advance `instance_idx` by one step.
    ///
    /// Returns `true` when the step was advanced, `false` if the instance is
    /// not found, not running, or already on the last step.
    pub fn advance(&mut self, instance_idx: usize) -> bool {
        let Some(inst) = self.instances.get_mut(instance_idx) else {
            return false;
        };
        if inst.status != WfStatus::Running {
            return false;
        }
        // Find how many steps the template has.
        let total = self
            .templates
            .iter()
            .find(|t| t.id == inst.template_id)
            .map_or(0, |t| t.step_count());

        if inst.current_step + 1 >= total {
            return false;
        }
        inst.current_step += 1;
        true
    }

    /// Mark an instance as completed.
    pub fn complete(&mut self, instance_idx: usize) {
        if let Some(inst) = self.instances.get_mut(instance_idx) {
            inst.status = WfStatus::Completed;
        }
    }

    /// Returns the count of currently running instances.
    #[must_use]
    pub fn running_count(&self) -> usize {
        self.instances.iter().filter(|i| i.is_running()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_template() -> WorkflowTemplate {
        WorkflowTemplate::new(
            1,
            "Ingest Pipeline",
            WorkflowTrigger::OnIngest,
            vec![
                "transcode".to_string(),
                "thumbnail".to_string(),
                "metadata".to_string(),
            ],
        )
    }

    fn engine_with_template() -> WorkflowEngine {
        let mut engine = WorkflowEngine::new();
        engine.register_template(sample_template());
        engine
    }

    // --- WorkflowTrigger tests ---

    #[test]
    fn test_trigger_on_ingest_is_automatic() {
        assert!(WorkflowTrigger::OnIngest.is_automatic());
    }

    #[test]
    fn test_trigger_manual_not_automatic() {
        assert!(!WorkflowTrigger::Manual.is_automatic());
    }

    #[test]
    fn test_trigger_on_approval_not_automatic() {
        assert!(!WorkflowTrigger::OnApproval.is_automatic());
    }

    #[test]
    fn test_trigger_scheduled_is_automatic() {
        assert!(WorkflowTrigger::Scheduled.is_automatic());
    }

    // --- WorkflowTemplate tests ---

    #[test]
    fn test_template_step_count() {
        let t = sample_template();
        assert_eq!(t.step_count(), 3);
    }

    #[test]
    fn test_template_empty_steps() {
        let t = WorkflowTemplate::new(99, "Empty", WorkflowTrigger::Manual, vec![]);
        assert_eq!(t.step_count(), 0);
    }

    // --- WfStatus tests ---

    #[test]
    fn test_status_running_not_terminal() {
        assert!(!WfStatus::Running.is_terminal());
    }

    #[test]
    fn test_status_completed_is_terminal() {
        assert!(WfStatus::Completed.is_terminal());
    }

    #[test]
    fn test_status_failed_is_terminal() {
        assert!(WfStatus::Failed.is_terminal());
    }

    #[test]
    fn test_status_cancelled_is_terminal() {
        assert!(WfStatus::Cancelled.is_terminal());
    }

    #[test]
    fn test_status_paused_not_terminal() {
        assert!(!WfStatus::Paused.is_terminal());
    }

    // --- WorkflowEngine tests ---

    #[test]
    fn test_engine_start_unknown_template_returns_none() {
        let mut engine = WorkflowEngine::new();
        assert!(engine.start(999, "asset-x", 0).is_none());
    }

    #[test]
    fn test_engine_start_known_template_returns_index() {
        let mut engine = engine_with_template();
        let idx = engine.start(1, "asset-a", 1000);
        assert_eq!(idx, Some(0));
    }

    #[test]
    fn test_engine_running_count() {
        let mut engine = engine_with_template();
        engine.start(1, "a", 0);
        engine.start(1, "b", 0);
        assert_eq!(engine.running_count(), 2);
    }

    #[test]
    fn test_engine_advance_increases_step() {
        let mut engine = engine_with_template();
        let idx = engine
            .start(1, "asset-b", 0)
            .expect("should succeed in test");
        assert!(engine.advance(idx));
        assert_eq!(engine.instances[idx].current_step, 1);
    }

    #[test]
    fn test_engine_complete_reduces_running_count() {
        let mut engine = engine_with_template();
        let idx = engine
            .start(1, "asset-c", 0)
            .expect("should succeed in test");
        assert_eq!(engine.running_count(), 1);
        engine.complete(idx);
        assert_eq!(engine.running_count(), 0);
    }

    #[test]
    fn test_instance_progress_pct() {
        let mut inst = WorkflowInstance::new(1, "asset-d", 0);
        inst.current_step = 2;
        let pct = inst.progress_pct(4);
        assert!((pct - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_instance_progress_zero_total() {
        let inst = WorkflowInstance::new(1, "asset-e", 0);
        assert!((inst.progress_pct(0) - 0.0).abs() < f64::EPSILON);
    }
}
