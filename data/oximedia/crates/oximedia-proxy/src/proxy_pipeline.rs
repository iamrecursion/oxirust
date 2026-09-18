//! Proxy production pipeline with ordered steps and phase tracking.
//!
//! A [`ProxyPipeline`] models the sequence of operations required to move
//! a media asset from ingest through proxy creation, review, and online
//! finishing.  Each operation is a [`PipelineStep`] that belongs to one of
//! the five [`PipelinePhase`]s.

#![allow(dead_code)]

/// High-level phase within a proxy production pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PipelinePhase {
    /// Raw media acquisition and integrity verification.
    Ingest,
    /// Proxy transcoding and quality checks.
    ProxyCreation,
    /// Offline editorial using low-resolution proxies.
    OfflineEdit,
    /// Client or stakeholder review before lock.
    Review,
    /// Online conform, colour grade, and final delivery.
    OnlineFinish,
}

impl PipelinePhase {
    /// Return a short human-readable name for this phase.
    pub fn name(self) -> &'static str {
        match self {
            Self::Ingest => "Ingest",
            Self::ProxyCreation => "Proxy Creation",
            Self::OfflineEdit => "Offline Edit",
            Self::Review => "Review",
            Self::OnlineFinish => "Online Finish",
        }
    }

    /// Return the next phase, or `None` if this is the final phase.
    pub fn next(self) -> Option<Self> {
        match self {
            Self::Ingest => Some(Self::ProxyCreation),
            Self::ProxyCreation => Some(Self::OfflineEdit),
            Self::OfflineEdit => Some(Self::Review),
            Self::Review => Some(Self::OnlineFinish),
            Self::OnlineFinish => None,
        }
    }

    /// Return all phases in pipeline order.
    pub fn all() -> &'static [PipelinePhase] {
        &[
            Self::Ingest,
            Self::ProxyCreation,
            Self::OfflineEdit,
            Self::Review,
            Self::OnlineFinish,
        ]
    }
}

impl std::fmt::Display for PipelinePhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Completion status of a [`PipelineStep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStatus {
    /// Not yet started.
    Pending,
    /// Currently executing.
    InProgress,
    /// Completed successfully.
    Complete,
    /// Failed; pipeline may be blocked.
    Failed,
    /// Explicitly skipped (e.g. optional step).
    Skipped,
}

impl StepStatus {
    /// Return `true` if this step no longer needs to run.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Complete | Self::Failed | Self::Skipped)
    }
}

/// A single named operation within a [`PipelinePhase`].
#[derive(Debug, Clone)]
pub struct PipelineStep {
    /// Human-readable name of this step.
    pub name: String,
    /// The phase this step belongs to.
    pub phase: PipelinePhase,
    /// Whether this step must succeed for the pipeline to advance.
    pub required: bool,
    /// Current completion status.
    pub status: StepStatus,
    /// Optional error message set when `status` is [`StepStatus::Failed`].
    pub error: Option<String>,
}

impl PipelineStep {
    /// Create a new required step in `Pending` state.
    pub fn new(name: impl Into<String>, phase: PipelinePhase) -> Self {
        Self {
            name: name.into(),
            phase,
            required: true,
            status: StepStatus::Pending,
            error: None,
        }
    }

    /// Create an optional step (failure will not block the pipeline).
    pub fn optional(name: impl Into<String>, phase: PipelinePhase) -> Self {
        Self {
            required: false,
            ..Self::new(name, phase)
        }
    }

    /// Mark this step as complete.
    pub fn complete(&mut self) {
        self.status = StepStatus::Complete;
        self.error = None;
    }

    /// Mark this step as failed with a reason.
    pub fn fail(&mut self, reason: impl Into<String>) {
        self.status = StepStatus::Failed;
        self.error = Some(reason.into());
    }

    /// Mark this step as skipped.
    pub fn skip(&mut self) {
        self.status = StepStatus::Skipped;
    }

    /// Mark this step as in progress.
    pub fn start(&mut self) {
        self.status = StepStatus::InProgress;
    }

    /// Return `true` if this step blocks pipeline advancement when failed.
    pub fn is_blocking(&self) -> bool {
        self.required && self.status == StepStatus::Failed
    }
}

/// Ordered pipeline of [`PipelineStep`]s tracking proxy workflow progress.
#[derive(Debug, Default)]
pub struct ProxyPipeline {
    steps: Vec<PipelineStep>,
}

impl ProxyPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a pipeline pre-populated with a standard set of steps.
    pub fn standard() -> Self {
        let mut p = Self::new();
        p.add(PipelineStep::new("Media ingest", PipelinePhase::Ingest));
        p.add(PipelineStep::new(
            "Checksum verification",
            PipelinePhase::Ingest,
        ));
        p.add(PipelineStep::new(
            "Proxy transcode",
            PipelinePhase::ProxyCreation,
        ));
        p.add(PipelineStep::optional(
            "QC check",
            PipelinePhase::ProxyCreation,
        ));
        p.add(PipelineStep::new("Load in NLE", PipelinePhase::OfflineEdit));
        p.add(PipelineStep::new(
            "Picture lock",
            PipelinePhase::OfflineEdit,
        ));
        p.add(PipelineStep::new("Client screening", PipelinePhase::Review));
        p.add(PipelineStep::optional(
            "Revision notes",
            PipelinePhase::Review,
        ));
        p.add(PipelineStep::new(
            "Online conform",
            PipelinePhase::OnlineFinish,
        ));
        p.add(PipelineStep::new(
            "Final delivery",
            PipelinePhase::OnlineFinish,
        ));
        p
    }

    /// Append a step to the pipeline.
    pub fn add(&mut self, step: PipelineStep) {
        self.steps.push(step);
    }

    /// Return a slice of all steps.
    pub fn steps(&self) -> &[PipelineStep] {
        &self.steps
    }

    /// Return all steps belonging to `phase`.
    pub fn steps_for_phase(&self, phase: PipelinePhase) -> Vec<&PipelineStep> {
        self.steps.iter().filter(|s| s.phase == phase).collect()
    }

    /// Return the current active phase (the phase of the first non-terminal step).
    ///
    /// Returns `None` if all steps are terminal.
    pub fn current_phase(&self) -> Option<PipelinePhase> {
        self.steps
            .iter()
            .find(|s| !s.status.is_terminal())
            .map(|s| s.phase)
    }

    /// Return `true` if any required step has failed.
    pub fn is_blocked(&self) -> bool {
        self.steps.iter().any(|s| s.is_blocking())
    }

    /// Return `true` if all steps are terminal (complete, skipped, or failed).
    pub fn is_finished(&self) -> bool {
        self.steps.iter().all(|s| s.status.is_terminal())
    }

    /// Return `true` if all required steps have completed or been skipped.
    pub fn is_successful(&self) -> bool {
        self.steps
            .iter()
            .filter(|s| s.required)
            .all(|s| matches!(s.status, StepStatus::Complete | StepStatus::Skipped))
    }

    /// Count steps by status.
    pub fn count_by_status(&self, status: StepStatus) -> usize {
        self.steps.iter().filter(|s| s.status == status).count()
    }

    /// Mutable access to a step by index.
    pub fn step_mut(&mut self, idx: usize) -> Option<&mut PipelineStep> {
        self.steps.get_mut(idx)
    }

    /// Total number of steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Return `true` if the pipeline contains no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_phase_name() {
        assert_eq!(PipelinePhase::Ingest.name(), "Ingest");
        assert_eq!(PipelinePhase::OnlineFinish.name(), "Online Finish");
    }

    #[test]
    fn pipeline_phase_display() {
        assert_eq!(format!("{}", PipelinePhase::Review), "Review");
    }

    #[test]
    fn pipeline_phase_next_chain() {
        assert_eq!(
            PipelinePhase::Ingest.next(),
            Some(PipelinePhase::ProxyCreation)
        );
        assert_eq!(PipelinePhase::OnlineFinish.next(), None);
    }

    #[test]
    fn pipeline_phase_all_count() {
        assert_eq!(PipelinePhase::all().len(), 5);
    }

    #[test]
    fn step_status_terminal() {
        assert!(StepStatus::Complete.is_terminal());
        assert!(StepStatus::Failed.is_terminal());
        assert!(StepStatus::Skipped.is_terminal());
        assert!(!StepStatus::Pending.is_terminal());
        assert!(!StepStatus::InProgress.is_terminal());
    }

    #[test]
    fn pipeline_step_complete() {
        let mut step = PipelineStep::new("Test", PipelinePhase::Ingest);
        step.complete();
        assert_eq!(step.status, StepStatus::Complete);
        assert!(step.error.is_none());
    }

    #[test]
    fn pipeline_step_fail() {
        let mut step = PipelineStep::new("Test", PipelinePhase::Ingest);
        step.fail("disk full");
        assert_eq!(step.status, StepStatus::Failed);
        assert_eq!(step.error.as_deref(), Some("disk full"));
    }

    #[test]
    fn pipeline_step_is_blocking_required_failed() {
        let mut step = PipelineStep::new("Req", PipelinePhase::ProxyCreation);
        step.fail("err");
        assert!(step.is_blocking());
    }

    #[test]
    fn pipeline_step_optional_not_blocking_on_fail() {
        let mut step = PipelineStep::optional("Opt", PipelinePhase::Review);
        step.fail("warn");
        assert!(!step.is_blocking());
    }

    #[test]
    fn proxy_pipeline_standard_length() {
        let p = ProxyPipeline::standard();
        assert_eq!(p.len(), 10);
    }

    #[test]
    fn proxy_pipeline_current_phase_initial() {
        let p = ProxyPipeline::standard();
        assert_eq!(p.current_phase(), Some(PipelinePhase::Ingest));
    }

    #[test]
    fn proxy_pipeline_is_not_finished_initially() {
        let p = ProxyPipeline::standard();
        assert!(!p.is_finished());
    }

    #[test]
    fn proxy_pipeline_is_blocked_after_required_failure() {
        let mut p = ProxyPipeline::standard();
        p.step_mut(0)
            .expect("should succeed in test")
            .fail("checksum mismatch");
        assert!(p.is_blocked());
    }

    #[test]
    fn proxy_pipeline_count_by_status() {
        let mut p = ProxyPipeline::standard();
        p.step_mut(0).expect("should succeed in test").complete();
        assert_eq!(p.count_by_status(StepStatus::Complete), 1);
        assert_eq!(p.count_by_status(StepStatus::Pending), p.len() - 1);
    }

    #[test]
    fn proxy_pipeline_steps_for_phase() {
        let p = ProxyPipeline::standard();
        let ingest_steps = p.steps_for_phase(PipelinePhase::Ingest);
        assert_eq!(ingest_steps.len(), 2); // "Media ingest" + "Checksum verification"
    }

    #[test]
    fn proxy_pipeline_is_successful_when_all_complete() {
        let mut p = ProxyPipeline::new();
        p.add(PipelineStep::new("A", PipelinePhase::Ingest));
        p.add(PipelineStep::new("B", PipelinePhase::ProxyCreation));
        p.step_mut(0).expect("should succeed in test").complete();
        p.step_mut(1).expect("should succeed in test").complete();
        assert!(p.is_successful());
    }

    #[test]
    fn proxy_pipeline_empty() {
        let p = ProxyPipeline::new();
        assert!(p.is_empty());
        assert!(p.current_phase().is_none());
        assert!(p.is_finished());
    }
}
