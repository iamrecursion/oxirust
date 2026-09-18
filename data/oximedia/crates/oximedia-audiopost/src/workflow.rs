//! Workflow automation and batch processing for audio post-production.

use crate::error::{AudioPostError, AudioPostResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

/// Workflow task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowTask {
    /// Task ID
    pub id: Uuid,
    /// Task name
    pub name: String,
    /// Task type
    pub task_type: TaskType,
    /// Task status
    pub status: TaskStatus,
    /// Input files
    pub inputs: Vec<PathBuf>,
    /// Output files
    pub outputs: Vec<PathBuf>,
    /// Task parameters
    pub parameters: HashMap<String, TaskParameter>,
    /// Dependencies (task IDs that must complete first)
    pub dependencies: Vec<Uuid>,
    /// Created timestamp
    pub created: chrono::DateTime<chrono::Utc>,
    /// Started timestamp
    pub started: Option<chrono::DateTime<chrono::Utc>>,
    /// Completed timestamp
    pub completed: Option<chrono::DateTime<chrono::Utc>>,
    /// Error message (if failed)
    pub error: Option<String>,
}

impl WorkflowTask {
    /// Create a new workflow task
    #[must_use]
    pub fn new(name: &str, task_type: TaskType) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            task_type,
            status: TaskStatus::Pending,
            inputs: Vec::new(),
            outputs: Vec::new(),
            parameters: HashMap::new(),
            dependencies: Vec::new(),
            created: chrono::Utc::now(),
            started: None,
            completed: None,
            error: None,
        }
    }

    /// Add an input file
    pub fn add_input(&mut self, path: PathBuf) {
        self.inputs.push(path);
    }

    /// Add an output file
    pub fn add_output(&mut self, path: PathBuf) {
        self.outputs.push(path);
    }

    /// Set a parameter
    pub fn set_parameter(&mut self, name: &str, value: TaskParameter) {
        self.parameters.insert(name.to_string(), value);
    }

    /// Add a dependency
    pub fn add_dependency(&mut self, task_id: Uuid) {
        if !self.dependencies.contains(&task_id) {
            self.dependencies.push(task_id);
        }
    }

    /// Mark task as started
    pub fn mark_started(&mut self) {
        self.status = TaskStatus::Running;
        self.started = Some(chrono::Utc::now());
    }

    /// Mark task as completed
    pub fn mark_completed(&mut self) {
        self.status = TaskStatus::Completed;
        self.completed = Some(chrono::Utc::now());
    }

    /// Mark task as failed
    pub fn mark_failed(&mut self, error: String) {
        self.status = TaskStatus::Failed;
        self.error = Some(error);
        self.completed = Some(chrono::Utc::now());
    }

    /// Get elapsed time in seconds
    #[must_use]
    pub fn elapsed_time(&self) -> Option<f64> {
        if let (Some(started), Some(completed)) = (self.started, self.completed) {
            Some((completed - started).num_milliseconds() as f64 / 1000.0)
        } else {
            None
        }
    }
}

/// Task type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    /// Normalize audio
    Normalize,
    /// Convert format
    ConvertFormat,
    /// Apply loudness correction
    LoudnessCorrection,
    /// Noise reduction
    NoiseReduction,
    /// EQ processing
    Eq,
    /// Compression
    Compression,
    /// Reverb
    Reverb,
    /// Batch export
    BatchExport,
    /// Stem creation
    StemCreation,
    /// ADR alignment
    AdrAlignment,
    /// Custom processing
    Custom,
}

impl TaskType {
    /// Get display name
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Normalize => "Normalize",
            Self::ConvertFormat => "Convert Format",
            Self::LoudnessCorrection => "Loudness Correction",
            Self::NoiseReduction => "Noise Reduction",
            Self::Eq => "EQ",
            Self::Compression => "Compression",
            Self::Reverb => "Reverb",
            Self::BatchExport => "Batch Export",
            Self::StemCreation => "Stem Creation",
            Self::AdrAlignment => "ADR Alignment",
            Self::Custom => "Custom",
        }
    }
}

/// Task status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Task is waiting to start
    Pending,
    /// Task is currently running
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed
    Failed,
    /// Task was cancelled
    Cancelled,
}

/// Task parameter value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskParameter {
    /// String value
    String(String),
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Bool(bool),
    /// File path
    FilePath(PathBuf),
}

/// Workflow containing multiple tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    /// Workflow ID
    pub id: Uuid,
    /// Workflow name
    pub name: String,
    /// Tasks in this workflow
    tasks: HashMap<Uuid, WorkflowTask>,
    /// Task execution order
    execution_order: Vec<Uuid>,
    /// Workflow status
    pub status: WorkflowStatus,
    /// Created timestamp
    pub created: chrono::DateTime<chrono::Utc>,
}

impl Workflow {
    /// Create a new workflow
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            tasks: HashMap::new(),
            execution_order: Vec::new(),
            status: WorkflowStatus::NotStarted,
            created: chrono::Utc::now(),
        }
    }

    /// Add a task to the workflow
    pub fn add_task(&mut self, task: WorkflowTask) -> Uuid {
        let id = task.id;
        self.tasks.insert(id, task);
        self.execution_order.push(id);
        id
    }

    /// Get a task
    ///
    /// # Errors
    ///
    /// Returns an error if task is not found
    pub fn get_task(&self, id: &Uuid) -> AudioPostResult<&WorkflowTask> {
        self.tasks
            .get(id)
            .ok_or(AudioPostError::Generic("Task not found".to_string()))
    }

    /// Get a mutable task
    ///
    /// # Errors
    ///
    /// Returns an error if task is not found
    pub fn get_task_mut(&mut self, id: &Uuid) -> AudioPostResult<&mut WorkflowTask> {
        self.tasks
            .get_mut(id)
            .ok_or(AudioPostError::Generic("Task not found".to_string()))
    }

    /// Get all tasks in execution order
    #[must_use]
    pub fn get_tasks_ordered(&self) -> Vec<&WorkflowTask> {
        self.execution_order
            .iter()
            .filter_map(|id| self.tasks.get(id))
            .collect()
    }

    /// Get task count
    #[must_use]
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Get completed task count
    #[must_use]
    pub fn completed_task_count(&self) -> usize {
        self.tasks
            .values()
            .filter(|task| task.status == TaskStatus::Completed)
            .count()
    }

    /// Get progress percentage
    #[must_use]
    pub fn progress_percentage(&self) -> f32 {
        if self.tasks.is_empty() {
            return 0.0;
        }
        (self.completed_task_count() as f32 / self.tasks.len() as f32) * 100.0
    }

    /// Calculate execution order based on dependencies
    pub fn calculate_execution_order(&mut self) -> AudioPostResult<()> {
        let mut order = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut temp_mark = std::collections::HashSet::new();

        // Topological sort
        for task_id in self.tasks.keys() {
            if !visited.contains(task_id) {
                self.visit_task(*task_id, &mut visited, &mut temp_mark, &mut order)?;
            }
        }

        self.execution_order = order;
        Ok(())
    }

    fn visit_task(
        &self,
        task_id: Uuid,
        visited: &mut std::collections::HashSet<Uuid>,
        temp_mark: &mut std::collections::HashSet<Uuid>,
        order: &mut Vec<Uuid>,
    ) -> AudioPostResult<()> {
        if temp_mark.contains(&task_id) {
            return Err(AudioPostError::Generic(
                "Circular dependency detected".to_string(),
            ));
        }

        if visited.contains(&task_id) {
            return Ok(());
        }

        temp_mark.insert(task_id);

        if let Some(task) = self.tasks.get(&task_id) {
            for dep_id in &task.dependencies {
                self.visit_task(*dep_id, visited, temp_mark, order)?;
            }
        }

        temp_mark.remove(&task_id);
        visited.insert(task_id);
        order.push(task_id);

        Ok(())
    }
}

/// Workflow status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkflowStatus {
    /// Workflow has not started
    NotStarted,
    /// Workflow is running
    Running,
    /// Workflow completed successfully
    Completed,
    /// Workflow failed
    Failed,
    /// Workflow was paused
    Paused,
}

/// Batch processor
#[derive(Debug)]
pub struct BatchProcessor {
    /// Maximum parallel tasks
    pub max_parallel: usize,
    /// Active workflows
    workflows: HashMap<Uuid, Workflow>,
}

impl BatchProcessor {
    /// Create a new batch processor
    #[must_use]
    pub fn new(max_parallel: usize) -> Self {
        Self {
            max_parallel,
            workflows: HashMap::new(),
        }
    }

    /// Add a workflow
    pub fn add_workflow(&mut self, workflow: Workflow) -> Uuid {
        let id = workflow.id;
        self.workflows.insert(id, workflow);
        id
    }

    /// Get a workflow
    ///
    /// # Errors
    ///
    /// Returns an error if workflow is not found
    pub fn get_workflow(&self, id: &Uuid) -> AudioPostResult<&Workflow> {
        self.workflows
            .get(id)
            .ok_or(AudioPostError::Generic("Workflow not found".to_string()))
    }

    /// Get workflow count
    #[must_use]
    pub fn workflow_count(&self) -> usize {
        self.workflows.len()
    }

    /// Process a single task (placeholder)
    pub fn process_task(&self, _task: &WorkflowTask) -> AudioPostResult<()> {
        // Placeholder for actual task processing
        Ok(())
    }
}

/// Preset for common workflows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowPreset {
    /// Preset name
    pub name: String,
    /// Preset description
    pub description: String,
    /// Template tasks
    pub template_tasks: Vec<WorkflowTask>,
}

impl WorkflowPreset {
    /// Create a new workflow preset
    #[must_use]
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            template_tasks: Vec::new(),
        }
    }

    /// Create a preset for podcast post-production
    #[must_use]
    pub fn podcast_postproduction() -> Self {
        let mut preset = Self::new(
            "Podcast Post-Production",
            "Standard podcast cleanup and export",
        );

        let mut task1 = WorkflowTask::new("Noise Reduction", TaskType::NoiseReduction);
        task1.set_parameter("strength", TaskParameter::Float(0.7));

        let mut task2 = WorkflowTask::new("Normalize", TaskType::Normalize);
        task2.set_parameter("target", TaskParameter::Float(-16.0));

        let mut task3 = WorkflowTask::new("Export", TaskType::BatchExport);
        task3.set_parameter("format", TaskParameter::String("mp3".to_string()));
        task3.set_parameter("bitrate", TaskParameter::Int(192));

        preset.template_tasks.push(task1);
        preset.template_tasks.push(task2);
        preset.template_tasks.push(task3);

        preset
    }

    /// Create a preset for dialogue cleanup
    #[must_use]
    pub fn dialogue_cleanup() -> Self {
        let mut preset = Self::new("Dialogue Cleanup", "Clean up dialogue tracks for film/TV");

        let mut task1 = WorkflowTask::new("Noise Reduction", TaskType::NoiseReduction);
        task1.set_parameter("strength", TaskParameter::Float(0.6));

        let mut task2 = WorkflowTask::new("De-esser", TaskType::Eq);
        task2.set_parameter("frequency", TaskParameter::Float(6000.0));

        let mut task3 = WorkflowTask::new("Compression", TaskType::Compression);
        task3.set_parameter("ratio", TaskParameter::Float(3.0));
        task3.set_parameter("threshold", TaskParameter::Float(-20.0));

        preset.template_tasks.push(task1);
        preset.template_tasks.push(task2);
        preset.template_tasks.push(task3);

        preset
    }

    /// Create a workflow from this preset
    #[must_use]
    pub fn create_workflow(&self) -> Workflow {
        let mut workflow = Workflow::new(&self.name);

        for template in &self.template_tasks {
            workflow.add_task(template.clone());
        }

        workflow
    }
}

/// Job queue for managing multiple workflows
#[derive(Debug)]
pub struct JobQueue {
    /// Queue of workflow IDs
    queue: Vec<Uuid>,
    /// Workflows
    workflows: HashMap<Uuid, Workflow>,
    /// Maximum concurrent workflows
    pub max_concurrent: usize,
}

impl JobQueue {
    /// Create a new job queue
    #[must_use]
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            queue: Vec::new(),
            workflows: HashMap::new(),
            max_concurrent,
        }
    }

    /// Enqueue a workflow
    pub fn enqueue(&mut self, workflow: Workflow) -> Uuid {
        let id = workflow.id;
        self.workflows.insert(id, workflow);
        self.queue.push(id);
        id
    }

    /// Dequeue the next workflow
    pub fn dequeue(&mut self) -> Option<Workflow> {
        if let Some(id) = self.queue.first() {
            let workflow = self.workflows.remove(id);
            self.queue.remove(0);
            workflow
        } else {
            None
        }
    }

    /// Get queue length
    #[must_use]
    pub fn queue_length(&self) -> usize {
        self.queue.len()
    }

    /// Get running workflows count (placeholder)
    #[must_use]
    pub fn running_count(&self) -> usize {
        self.workflows
            .values()
            .filter(|w| w.status == WorkflowStatus::Running)
            .count()
    }
}

/// Progress tracker
#[derive(Debug, Clone)]
pub struct ProgressTracker {
    /// Total items
    pub total: usize,
    /// Completed items
    pub completed: usize,
    /// Failed items
    pub failed: usize,
    /// Start time
    pub start_time: Option<chrono::DateTime<chrono::Utc>>,
}

impl ProgressTracker {
    /// Create a new progress tracker
    #[must_use]
    pub fn new(total: usize) -> Self {
        Self {
            total,
            completed: 0,
            failed: 0,
            start_time: Some(chrono::Utc::now()),
        }
    }

    /// Mark an item as completed
    pub fn mark_completed(&mut self) {
        self.completed += 1;
    }

    /// Mark an item as failed
    pub fn mark_failed(&mut self) {
        self.failed += 1;
    }

    /// Get progress percentage
    #[must_use]
    pub fn progress_percentage(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        ((self.completed + self.failed) as f32 / self.total as f32) * 100.0
    }

    /// Get estimated time remaining in seconds
    #[must_use]
    pub fn estimated_time_remaining(&self) -> Option<f64> {
        if let Some(start) = self.start_time {
            let elapsed = (chrono::Utc::now() - start).num_milliseconds() as f64 / 1000.0;
            let completed = self.completed + self.failed;

            if completed > 0 {
                let rate = elapsed / completed as f64;
                let remaining = self.total - completed;
                return Some(rate * remaining as f64);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_task_creation() {
        let task = WorkflowTask::new("Test Task", TaskType::Normalize);
        assert_eq!(task.name, "Test Task");
        assert_eq!(task.status, TaskStatus::Pending);
    }

    #[test]
    fn test_task_add_input() {
        let mut task = WorkflowTask::new("Test", TaskType::Normalize);
        task.add_input(PathBuf::from("/input.wav"));
        assert_eq!(task.inputs.len(), 1);
    }

    #[test]
    fn test_task_add_output() {
        let mut task = WorkflowTask::new("Test", TaskType::Normalize);
        task.add_output(PathBuf::from("/output.wav"));
        assert_eq!(task.outputs.len(), 1);
    }

    #[test]
    fn test_task_set_parameter() {
        let mut task = WorkflowTask::new("Test", TaskType::Normalize);
        task.set_parameter("target", TaskParameter::Float(-16.0));
        assert_eq!(task.parameters.len(), 1);
    }

    #[test]
    fn test_task_add_dependency() {
        let mut task = WorkflowTask::new("Test", TaskType::Normalize);
        let dep_id = Uuid::new_v4();
        task.add_dependency(dep_id);
        assert_eq!(task.dependencies.len(), 1);
    }

    #[test]
    fn test_task_mark_started() {
        let mut task = WorkflowTask::new("Test", TaskType::Normalize);
        task.mark_started();
        assert_eq!(task.status, TaskStatus::Running);
        assert!(task.started.is_some());
    }

    #[test]
    fn test_task_mark_completed() {
        let mut task = WorkflowTask::new("Test", TaskType::Normalize);
        task.mark_completed();
        assert_eq!(task.status, TaskStatus::Completed);
        assert!(task.completed.is_some());
    }

    #[test]
    fn test_task_mark_failed() {
        let mut task = WorkflowTask::new("Test", TaskType::Normalize);
        task.mark_failed("Error message".to_string());
        assert_eq!(task.status, TaskStatus::Failed);
        assert_eq!(task.error, Some("Error message".to_string()));
    }

    #[test]
    fn test_workflow_creation() {
        let workflow = Workflow::new("Test Workflow");
        assert_eq!(workflow.name, "Test Workflow");
        assert_eq!(workflow.task_count(), 0);
    }

    #[test]
    fn test_workflow_add_task() {
        let mut workflow = Workflow::new("Test");
        let task = WorkflowTask::new("Task 1", TaskType::Normalize);
        workflow.add_task(task);
        assert_eq!(workflow.task_count(), 1);
    }

    #[test]
    fn test_workflow_get_task() {
        let mut workflow = Workflow::new("Test");
        let task = WorkflowTask::new("Task 1", TaskType::Normalize);
        let id = workflow.add_task(task);
        assert!(workflow.get_task(&id).is_ok());
    }

    #[test]
    fn test_workflow_progress() {
        let mut workflow = Workflow::new("Test");
        let mut task = WorkflowTask::new("Task 1", TaskType::Normalize);
        task.mark_completed();
        workflow.add_task(task);
        workflow.add_task(WorkflowTask::new("Task 2", TaskType::Normalize));
        assert_eq!(workflow.progress_percentage(), 50.0);
    }

    #[test]
    fn test_batch_processor_creation() {
        let processor = BatchProcessor::new(4);
        assert_eq!(processor.max_parallel, 4);
    }

    #[test]
    fn test_batch_processor_add_workflow() {
        let mut processor = BatchProcessor::new(4);
        let workflow = Workflow::new("Test");
        processor.add_workflow(workflow);
        assert_eq!(processor.workflow_count(), 1);
    }

    #[test]
    fn test_workflow_preset_podcast() {
        let preset = WorkflowPreset::podcast_postproduction();
        assert_eq!(preset.name, "Podcast Post-Production");
        assert_eq!(preset.template_tasks.len(), 3);
    }

    #[test]
    fn test_workflow_preset_dialogue() {
        let preset = WorkflowPreset::dialogue_cleanup();
        assert_eq!(preset.name, "Dialogue Cleanup");
        assert_eq!(preset.template_tasks.len(), 3);
    }

    #[test]
    fn test_workflow_preset_create_workflow() {
        let preset = WorkflowPreset::podcast_postproduction();
        let workflow = preset.create_workflow();
        assert_eq!(workflow.task_count(), 3);
    }

    #[test]
    fn test_job_queue_creation() {
        let queue = JobQueue::new(2);
        assert_eq!(queue.max_concurrent, 2);
    }

    #[test]
    fn test_job_queue_enqueue() {
        let mut queue = JobQueue::new(2);
        let workflow = Workflow::new("Test");
        queue.enqueue(workflow);
        assert_eq!(queue.queue_length(), 1);
    }

    #[test]
    fn test_job_queue_dequeue() {
        let mut queue = JobQueue::new(2);
        let workflow = Workflow::new("Test");
        queue.enqueue(workflow);
        let dequeued = queue.dequeue();
        assert!(dequeued.is_some());
        assert_eq!(queue.queue_length(), 0);
    }

    #[test]
    fn test_progress_tracker_creation() {
        let tracker = ProgressTracker::new(10);
        assert_eq!(tracker.total, 10);
        assert_eq!(tracker.completed, 0);
    }

    #[test]
    fn test_progress_tracker_mark_completed() {
        let mut tracker = ProgressTracker::new(10);
        tracker.mark_completed();
        assert_eq!(tracker.completed, 1);
    }

    #[test]
    fn test_progress_tracker_mark_failed() {
        let mut tracker = ProgressTracker::new(10);
        tracker.mark_failed();
        assert_eq!(tracker.failed, 1);
    }

    #[test]
    fn test_progress_tracker_percentage() {
        let mut tracker = ProgressTracker::new(10);
        tracker.mark_completed();
        tracker.mark_completed();
        assert_eq!(tracker.progress_percentage(), 20.0);
    }

    #[test]
    fn test_task_type_as_str() {
        assert_eq!(TaskType::Normalize.as_str(), "Normalize");
        assert_eq!(TaskType::ConvertFormat.as_str(), "Convert Format");
    }
}

// ── AudiopostPipeline ─────────────────────────────────────────────────────────

/// Configuration for each DSP stage in `AudiopostPipeline`.
#[derive(Debug, Clone)]
pub struct PipelineStageConfig {
    /// Enable the declick stage.
    pub declick: bool,
    /// Enable the spectral-subtraction denoise stage.
    pub denoise: bool,
    /// Enable the sound-design chain (reverb → chorus → distortion) stage.
    pub sound_design: bool,
    /// Enable the loudness normalisation stage.
    pub normalize_loudness: bool,
    /// Target integrated loudness in LUFS (used when `normalize_loudness` is true).
    pub target_lufs: f32,
    /// Maximum true-peak in dBTP (used when `normalize_loudness` is true).
    pub max_true_peak_dbtp: f32,
}

impl Default for PipelineStageConfig {
    fn default() -> Self {
        Self {
            declick: true,
            denoise: true,
            sound_design: false,
            normalize_loudness: true,
            target_lufs: -23.0,
            max_true_peak_dbtp: -1.0,
        }
    }
}

/// Summary output produced by `AudiopostPipeline::process`.
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    /// Processed audio samples.
    pub samples: Vec<f32>,
    /// Integrated loudness measured on the output (LUFS).
    pub integrated_lufs: f64,
    /// True-peak measured on the output (dBTP).
    pub true_peak_dbtp: f64,
    /// Number of clicks detected and repaired (0 if declick stage was bypassed).
    pub clicks_repaired: usize,
}

/// Sequential DSP pipeline: declick → denoise → sound_design → loudness normalisation → metering tap.
///
/// Each stage is individually bypassable via [`PipelineStageConfig`].
///
/// Note: `process` takes `&mut self` because the optional
/// [`crate::sound_design::SoundDesignChain`] holds mutable internal state
/// (delay-line buffers for chorus and reverb).
///
/// ```
/// # use oximedia_audiopost::workflow::{AudiopostPipeline, PipelineStageConfig};
/// let mut pipeline = AudiopostPipeline::new(48000, PipelineStageConfig::default())
///     .expect("valid sample rate");
/// ```
pub struct AudiopostPipeline {
    sample_rate: u32,
    config: PipelineStageConfig,
    /// Optional sound-design chain; present when `config.sound_design` is true.
    sound_chain: Option<crate::sound_design::SoundDesignChain>,
}

impl AudiopostPipeline {
    /// Create a new pipeline.
    ///
    /// When `config.sound_design` is `true` a default
    /// [`crate::sound_design::SoundDesignChain`] (all stages bypassed) is
    /// initialised.  You can add reverb / chorus / distortion afterwards by
    /// rebuilding the pipeline with a custom chain via
    /// [`AudiopostPipeline::with_sound_chain`].
    ///
    /// # Errors
    ///
    /// Returns [`AudioPostError::InvalidSampleRate`] for a zero sample rate.
    pub fn new(sample_rate: u32, config: PipelineStageConfig) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        // Build a pass-through chain only when the stage is enabled.
        let sound_chain = if config.sound_design {
            Some(crate::sound_design::SoundDesignChain::new(sample_rate)?)
        } else {
            None
        };
        Ok(Self {
            sample_rate,
            config,
            sound_chain,
        })
    }

    /// Replace (or install) the sound-design chain and enable the stage.
    ///
    /// This lets callers configure reverb / chorus / distortion before
    /// the first `process` call.
    #[must_use]
    pub fn with_sound_chain(mut self, chain: crate::sound_design::SoundDesignChain) -> Self {
        self.sound_chain = Some(chain);
        self.config.sound_design = true;
        self
    }

    /// Process `samples` through the enabled DSP stages.
    ///
    /// Returns a [`PipelineOutput`] containing the processed audio and
    /// metering data measured at the output.
    ///
    /// # Errors
    ///
    /// Propagates errors from the individual DSP stages.
    pub fn process(&mut self, samples: &[f32]) -> AudioPostResult<PipelineOutput> {
        if samples.is_empty() {
            return Err(AudioPostError::InvalidBufferSize(0));
        }

        let mut buf = samples.to_vec();
        let mut clicks_repaired = 0_usize;

        // Stage 1 – Declick (MAD-based transient detection + cubic Hermite interpolation).
        if self.config.declick {
            let click_cfg = crate::restoration::DeclickConfig::default();
            // Count detected clicks before repairing.
            let first_diff: Vec<f32> = buf.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
            let mut sorted_diffs = first_diff.clone();
            sorted_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let m = sorted_diffs.len();
            let mad_median = if m == 0 {
                1e-9_f32
            } else if m % 2 == 1 {
                sorted_diffs[m / 2]
            } else {
                (sorted_diffs[m / 2 - 1] + sorted_diffs[m / 2]) / 2.0
            }
            .max(1e-9);
            let threshold = click_cfg.mad_threshold * mad_median;
            clicks_repaired = first_diff.iter().filter(|&&d| d > threshold).count();

            buf = crate::restoration::declick(&buf, &click_cfg)?;
        }

        // Stage 2 – Spectral subtraction denoise.
        if self.config.denoise {
            let ss_cfg = crate::restoration::SpectralSubtractionConfig::default();
            buf = crate::restoration::spectral_subtract(&buf, &ss_cfg)?;
        }

        // Stage 3 – Sound-design chain (reverb → chorus → distortion).
        if self.config.sound_design {
            if let Some(ref mut chain) = self.sound_chain {
                buf = chain.process(&buf);
            }
        }

        // Stage 4 – Loudness normalisation.
        if self.config.normalize_loudness {
            let delivery_cfg = crate::delivery::DeliveryConfig {
                target_lufs: self.config.target_lufs,
                max_true_peak_dbtp: self.config.max_true_peak_dbtp,
                tolerance_lu: 0.5,
            };
            buf = crate::delivery::normalize_loudness(&buf, self.sample_rate, &delivery_cfg)?;
        }

        // Metering tap on the output.
        let measurement =
            oximedia_audio::loudness_gating::GatedLoudnessMeter::measure(&buf, self.sample_rate, 1);

        Ok(PipelineOutput {
            samples: buf,
            integrated_lufs: measurement.integrated_lufs,
            true_peak_dbtp: measurement.true_peak_dbtp,
            clicks_repaired,
        })
    }
}
