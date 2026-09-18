//! Python bindings for `oximedia-workflow` orchestration engine.
//!
//! Provides `PyWorkflow`, `PyWorkflowStep`, `PyWorkflowStatus`,
//! `PyWorkflowTemplate`, and standalone functions for workflow management.

use oximedia_transcode::TranscodePipeline;
use oximedia_workflow as owf;
use oximedia_workflow::TaskExecutor as _;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyWorkflowStep
// ---------------------------------------------------------------------------

/// A single step (task) in a workflow.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyWorkflowStep {
    /// Step identifier.
    #[pyo3(get)]
    pub step_id: String,

    /// Task type: transcode, qc, transfer, analysis, wait, notification.
    #[pyo3(get)]
    pub task_type: String,

    /// Human-readable description.
    #[pyo3(get, set)]
    pub description: String,

    /// IDs of steps this step depends on.
    #[pyo3(get)]
    pub depends_on: Vec<String>,

    /// Task parameters as a JSON-compatible dict.
    #[pyo3(get)]
    pub params: HashMap<String, String>,
}

#[pymethods]
impl PyWorkflowStep {
    /// Create a new workflow step.
    #[new]
    #[pyo3(signature = (step_id, task_type, description=None, depends_on=None))]
    fn new(
        step_id: &str,
        task_type: &str,
        description: Option<&str>,
        depends_on: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let valid_types = [
            "transcode",
            "qc",
            "transfer",
            "analysis",
            "wait",
            "notification",
        ];
        if !valid_types.contains(&task_type) {
            return Err(PyValueError::new_err(format!(
                "Unknown task type '{}'. Valid: {}",
                task_type,
                valid_types.join(", ")
            )));
        }
        Ok(Self {
            step_id: step_id.to_string(),
            task_type: task_type.to_string(),
            description: description.unwrap_or("").to_string(),
            depends_on: depends_on.unwrap_or_default(),
            params: HashMap::new(),
        })
    }

    /// Set a parameter on this step.
    fn set_param(&mut self, key: &str, value: &str) {
        self.params.insert(key.to_string(), value.to_string());
    }

    /// Get a parameter value.
    fn get_param(&self, key: &str) -> Option<String> {
        self.params.get(key).cloned()
    }

    fn __repr__(&self) -> String {
        let deps = if self.depends_on.is_empty() {
            "none".to_string()
        } else {
            self.depends_on.join(", ")
        };
        format!(
            "PyWorkflowStep(id='{}', type='{}', deps=[{}])",
            self.step_id, self.task_type, deps,
        )
    }
}

// ---------------------------------------------------------------------------
// PyWorkflowStatus
// ---------------------------------------------------------------------------

/// Status of a workflow execution.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyWorkflowStatus {
    /// Workflow state: idle, running, completed, failed, cancelled.
    #[pyo3(get)]
    pub state: String,

    /// Progress as a fraction (0.0 - 1.0).
    #[pyo3(get)]
    pub progress: f64,

    /// Number of completed tasks.
    #[pyo3(get)]
    pub tasks_completed: usize,

    /// Total number of tasks.
    #[pyo3(get)]
    pub tasks_total: usize,

    /// Error message (if failed).
    #[pyo3(get)]
    pub error: Option<String>,
}

#[pymethods]
impl PyWorkflowStatus {
    fn __repr__(&self) -> String {
        format!(
            "PyWorkflowStatus(state='{}', progress={:.1}%, tasks={}/{})",
            self.state,
            self.progress * 100.0,
            self.tasks_completed,
            self.tasks_total,
        )
    }
}

// ---------------------------------------------------------------------------
// PyWorkflowTemplate
// ---------------------------------------------------------------------------

/// A reusable workflow template.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyWorkflowTemplate {
    /// Template name.
    #[pyo3(get)]
    pub name: String,

    /// Template description.
    #[pyo3(get)]
    pub description: String,

    /// Steps in the template.
    steps: Vec<PyWorkflowStep>,
}

#[pymethods]
impl PyWorkflowTemplate {
    /// Create a new template.
    #[new]
    fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            steps: Vec::new(),
        }
    }

    /// Add a step to the template.
    fn add_step(&mut self, step: PyWorkflowStep) {
        self.steps.push(step);
    }

    /// Get all steps.
    fn steps(&self) -> Vec<PyWorkflowStep> {
        self.steps.clone()
    }

    /// Get step count.
    fn step_count(&self) -> usize {
        self.steps.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyWorkflowTemplate(name='{}', steps={})",
            self.name,
            self.steps.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// PyWorkflow
// ---------------------------------------------------------------------------

/// A media processing workflow.
#[pyclass]
pub struct PyWorkflow {
    name: String,
    steps: Vec<PyWorkflowStep>,
    state: String,
}

#[pymethods]
impl PyWorkflow {
    /// Create a new workflow.
    #[new]
    #[pyo3(signature = (name=None))]
    fn new(name: Option<&str>) -> Self {
        Self {
            name: name.unwrap_or("Untitled Workflow").to_string(),
            steps: Vec::new(),
            state: "idle".to_string(),
        }
    }

    /// Get the workflow name.
    fn name(&self) -> String {
        self.name.clone()
    }

    /// Add a step to the workflow.
    fn add_step(&mut self, step: PyWorkflowStep) -> PyResult<()> {
        // Validate that dependencies exist
        let existing_ids: Vec<&str> = self.steps.iter().map(|s| s.step_id.as_str()).collect();
        for dep in &step.depends_on {
            if !existing_ids.contains(&dep.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "Step '{}' depends on unknown step '{}'",
                    step.step_id, dep
                )));
            }
        }
        self.steps.push(step);
        Ok(())
    }

    /// Get all steps.
    fn steps(&self) -> Vec<PyWorkflowStep> {
        self.steps.clone()
    }

    /// Get step count.
    fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Get current workflow state.
    fn state(&self) -> String {
        self.state.clone()
    }

    /// Get the workflow status.
    fn status(&self) -> PyWorkflowStatus {
        PyWorkflowStatus {
            state: self.state.clone(),
            progress: 0.0,
            tasks_completed: 0,
            tasks_total: self.steps.len(),
            error: None,
        }
    }

    /// Validate the workflow DAG.
    fn validate(&self) -> PyResult<Vec<String>> {
        let mut issues = Vec::new();

        if self.steps.is_empty() {
            issues.push("Workflow has no steps".to_string());
            return Ok(issues);
        }

        let step_ids: Vec<&str> = self.steps.iter().map(|s| s.step_id.as_str()).collect();

        // Check for duplicate IDs
        let mut seen = std::collections::HashSet::new();
        for id in &step_ids {
            if !seen.insert(id) {
                issues.push(format!("Duplicate step ID: '{}'", id));
            }
        }

        // Check for missing dependencies
        for step in &self.steps {
            for dep in &step.depends_on {
                if !step_ids.contains(&dep.as_str()) {
                    issues.push(format!(
                        "Step '{}' depends on unknown step '{}'",
                        step.step_id, dep
                    ));
                }
            }
        }

        Ok(issues)
    }

    /// Serialize to JSON string.
    fn to_json(&self) -> PyResult<String> {
        let data = serde_json::json!({
            "name": self.name,
            "state": self.state,
            "steps": self.steps.iter().map(|s| {
                serde_json::json!({
                    "step_id": s.step_id,
                    "task_type": s.task_type,
                    "description": s.description,
                    "depends_on": s.depends_on,
                    "params": s.params,
                })
            }).collect::<Vec<_>>(),
        });
        serde_json::to_string_pretty(&data)
            .map_err(|e| PyRuntimeError::new_err(format!("JSON error: {e}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "PyWorkflow(name='{}', steps={}, state='{}')",
            self.name,
            self.steps.len(),
            self.state,
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a workflow from a list of steps.
#[pyfunction]
#[pyo3(signature = (steps, name=None))]
pub fn create_workflow(steps: Vec<PyWorkflowStep>, name: Option<&str>) -> PyResult<PyWorkflow> {
    let mut wf = PyWorkflow::new(name);
    for step in steps {
        wf.add_step(step)?;
    }
    Ok(wf)
}

/// List available built-in workflow templates.
#[pyfunction]
pub fn list_templates() -> Vec<PyWorkflowTemplate> {
    let mut templates = Vec::new();

    // Transcode template
    let mut t = PyWorkflowTemplate::new("transcode", "Validate -> Transcode -> Verify");
    if let Ok(s1) = PyWorkflowStep::new("validate", "qc", Some("Validate source"), None) {
        t.add_step(s1);
    }
    if let Ok(s2) = PyWorkflowStep::new(
        "transcode",
        "transcode",
        Some("Transcode"),
        Some(vec!["validate".to_string()]),
    ) {
        t.add_step(s2);
    }
    if let Ok(s3) = PyWorkflowStep::new(
        "verify",
        "qc",
        Some("Verify output"),
        Some(vec!["transcode".to_string()]),
    ) {
        t.add_step(s3);
    }
    templates.push(t);

    // Ingest template
    let mut t2 = PyWorkflowTemplate::new("ingest", "Copy -> Probe -> Generate proxy");
    if let Ok(s1) = PyWorkflowStep::new("copy", "transfer", Some("Copy to storage"), None) {
        t2.add_step(s1);
    }
    if let Ok(s2) = PyWorkflowStep::new(
        "probe",
        "analysis",
        Some("Probe format"),
        Some(vec!["copy".to_string()]),
    ) {
        t2.add_step(s2);
    }
    if let Ok(s3) = PyWorkflowStep::new(
        "proxy",
        "transcode",
        Some("Generate proxy"),
        Some(vec!["probe".to_string()]),
    ) {
        t2.add_step(s3);
    }
    templates.push(t2);

    // QC template
    let mut t3 = PyWorkflowTemplate::new("qc", "Format, quality, and loudness checks");
    if let Ok(s1) = PyWorkflowStep::new("format_check", "qc", Some("Format check"), None) {
        t3.add_step(s1);
    }
    if let Ok(s2) = PyWorkflowStep::new("quality_check", "qc", Some("Quality check"), None) {
        t3.add_step(s2);
    }
    if let Ok(s3) = PyWorkflowStep::new("loudness_check", "qc", Some("Loudness check"), None) {
        t3.add_step(s3);
    }
    templates.push(t3);

    templates
}

// ---------------------------------------------------------------------------
// Real workflow execution bridge
// ---------------------------------------------------------------------------

/// `task_type` values this build can genuinely execute end-to-end through the
/// real `oximedia-workflow` orchestration engine.
///
/// `"analysis"` and `"notification"` are intentionally excluded: no real
/// analysis engine (there are several candidate crates — `oximedia-scene`,
/// `oximedia-shots`, `oximedia-audio-analysis`, `oximedia-qc` — with no
/// documented convention for which one a given step means) or notification
/// transport (no HTTP client is wired for webhook/Slack/Discord/email
/// delivery) is connected to the workflow executor in this build. Accepting
/// them would mean either fabricating success (the exact bug this module
/// fixes) or silently guessing a backend, so `run_workflow` rejects them
/// up front with a clear error instead.
const EXECUTABLE_TASK_TYPES: &[&str] = &["transcode", "qc", "transfer", "wait"];

/// Fetch a required string parameter from a step, or return a descriptive
/// `PyValueError` naming the step and the missing key.
fn require_param(step: &PyWorkflowStep, key: &str) -> PyResult<String> {
    step.params.get(key).cloned().ok_or_else(|| {
        PyValueError::new_err(format!(
            "Step '{}' (task_type '{}') is missing required param '{key}'",
            step.step_id, step.task_type
        ))
    })
}

/// Convert a `PyWorkflowStep` into a real `oximedia_workflow::TaskType`.
///
/// Only called for task types in [`EXECUTABLE_TASK_TYPES`]; `run_workflow`
/// rejects every other `task_type` before reaching this function.
fn step_to_task_type(step: &PyWorkflowStep) -> PyResult<owf::TaskType> {
    match step.task_type.as_str() {
        "wait" => {
            let secs: f64 = match step.params.get("duration_secs") {
                Some(v) => v.parse().map_err(|_| {
                    PyValueError::new_err(format!(
                        "Step '{}': 'duration_secs' must be a number, got '{v}'",
                        step.step_id
                    ))
                })?,
                None => 0.0,
            };
            if !secs.is_finite() || secs < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "Step '{}': 'duration_secs' must be a non-negative finite number",
                    step.step_id
                )));
            }
            Ok(owf::TaskType::Wait {
                duration: std::time::Duration::from_secs_f64(secs),
            })
        }
        "transfer" => Ok(owf::TaskType::Transfer {
            source: require_param(step, "source")?,
            destination: require_param(step, "destination")?,
            protocol: owf::TransferProtocol::Local,
            options: HashMap::new(),
        }),
        "transcode" => Ok(owf::TaskType::Transcode {
            input: std::path::PathBuf::from(require_param(step, "input")?),
            output: std::path::PathBuf::from(require_param(step, "output")?),
            preset: step
                .params
                .get("preset")
                .cloned()
                .unwrap_or_else(|| "default".to_string()),
            params: HashMap::new(),
        }),
        "qc" => Ok(owf::TaskType::QualityControl {
            input: std::path::PathBuf::from(require_param(step, "input")?),
            profile: step
                .params
                .get("profile")
                .cloned()
                .unwrap_or_else(|| "default".to_string()),
            rules: Vec::new(),
        }),
        other => Err(PyValueError::new_err(format!(
            "Unsupported task_type '{other}' for step '{}'",
            step.step_id
        ))),
    }
}

/// Real transcode execution for `TaskType::Transcode` steps.
///
/// Routes to the same real `oximedia_transcode::TranscodePipeline` entry
/// point used by `PyProxyGenerator::generate` (see `proxy_py.rs`), instead of
/// `oximedia_workflow::DefaultTaskExecutor`'s intent-only stub, which
/// validates paths and reports success without actually producing any
/// output (see its doc comment: "the actual codec pipeline is implemented in
/// oximedia-transcode; this executor records the intent and succeeds").
async fn execute_transcode_task(
    task_id: owf::TaskId,
    input: &std::path::Path,
    output: &std::path::Path,
) -> Result<owf::TaskResult, owf::WorkflowError> {
    let start = std::time::Instant::now();

    let out_ext = output
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .unwrap_or_default();
    if !matches!(out_ext.as_str(), "mkv" | "webm" | "ogg" | "oga" | "opus") {
        return Err(owf::WorkflowError::generic(format!(
            "Unsupported transcode output container '.{out_ext}' for '{}'; \
             supported: mkv, webm, ogg, oga, opus",
            output.display()
        )));
    }

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                owf::WorkflowError::generic(format!("Failed to create output dir: {e}"))
            })?;
        }
    }

    let mut pipeline = TranscodePipeline::builder()
        .input(input.to_path_buf())
        .output(output.to_path_buf())
        .track_progress(false)
        .build()
        .map_err(|e| owf::WorkflowError::generic(format!("Pipeline build error: {e}")))?;

    let transcode_out = pipeline
        .execute()
        .await
        .map_err(|e| owf::WorkflowError::generic(format!("Pipeline exec error: {e}")))?;

    Ok(owf::TaskResult {
        task_id,
        status: owf::TaskState::Completed,
        data: Some(serde_json::json!({
            "output_path": transcode_out.output_path,
            "file_size": transcode_out.file_size,
        })),
        error: None,
        duration: start.elapsed(),
        outputs: vec![output.to_path_buf()],
    })
}

/// `TaskExecutor` bridging `PyWorkflow` steps to real work.
///
/// Non-`Transcode` task types delegate to `oximedia_workflow::DefaultTaskExecutor`,
/// which performs genuine work for them: real `tokio::time::sleep` for
/// `Wait`, real `tokio::fs::copy` for local `Transfer`, and a real
/// file-existence + metadata check for `QualityControl`. `Transcode` is
/// intercepted so it runs the real transcode pipeline instead of
/// `DefaultTaskExecutor`'s intent-only stub (see `execute_transcode_task`).
struct PyWorkflowTaskExecutor;

#[async_trait::async_trait]
impl owf::TaskExecutor for PyWorkflowTaskExecutor {
    async fn execute(&self, task: &owf::Task) -> Result<owf::TaskResult, owf::WorkflowError> {
        if let owf::TaskType::Transcode { input, output, .. } = &task.task_type {
            return execute_transcode_task(task.id, input, output).await;
        }
        owf::DefaultTaskExecutor.execute(task).await
    }
}

/// Run a workflow by actually executing each step through the real
/// `oximedia-workflow` engine's task types, returning a status that
/// reflects what really happened — not a hardcoded "completed".
///
/// Execution is driven as a single dependency-respecting sequential pass
/// over `oximedia_workflow::Workflow::topological_sort()`, rather than via
/// `oximedia_workflow::executor::WorkflowExecutor::execute()`. That engine's
/// scheduling loop has a real bug: its task iterator is shared across the
/// whole run, and when a task's dependency isn't satisfied yet at the
/// moment the loop reaches it, the loop does `continue` — which permanently
/// advances past that task without ever retrying it. Because a
/// just-spawned dependency can never have completed yet within the same
/// synchronous scan, this reliably drops *every* task that has an
/// unsatisfied dependency on the first pass (i.e. every non-root task in
/// any dependency chain) while still reporting `WorkflowState::Completed`,
/// since dropped tasks are neither completed nor failed. Verified directly:
/// a 2-step `a -> b` chain run through `WorkflowExecutor::execute()`
/// executes `a` and silently never runs `b`. A single forward pass over
/// `topological_sort()`'s output has no such race — every dependency of a
/// task is guaranteed to appear earlier in that order and is fully awaited
/// before the next task starts — so this sacrifices `WorkflowExecutor`'s
/// parallel fan-out (steps here run sequentially) in exchange for
/// correctness. Per-task execution still fully delegates to the real
/// `oximedia_workflow::TaskExecutor` trait / `DefaultTaskExecutor` (real
/// `Wait`/local `Transfer`/`QualityControl`) and the real transcode
/// pipeline for `Transcode` (see [`PyWorkflowTaskExecutor`]).
///
/// # Errors
///
/// Returns `PyValueError` if the workflow fails DAG validation, contains a
/// step whose `task_type` this build cannot honestly execute (see
/// [`EXECUTABLE_TASK_TYPES`]), or a step is missing a required parameter.
/// Returns `PyRuntimeError` if the workflow graph is invalid (e.g. a cycle),
/// or the async runtime bridge cannot be constructed.
///
/// Individual step failures (e.g. a transcode pipeline error, a copy I/O
/// error) do **not** raise — they are reflected honestly in the returned
/// `PyWorkflowStatus` (`state="failed"`, `error` set, `progress` < 1.0, only
/// genuinely-completed steps counted).
#[pyfunction]
pub fn run_workflow(workflow: &PyWorkflow) -> PyResult<PyWorkflowStatus> {
    let issues = workflow.validate()?;
    if !issues.is_empty() {
        return Err(PyValueError::new_err(format!(
            "Workflow validation failed: {}",
            issues.join("; ")
        )));
    }

    let steps = workflow.steps();

    for step in &steps {
        if !EXECUTABLE_TASK_TYPES.contains(&step.task_type.as_str()) {
            return Err(PyValueError::new_err(format!(
                "Workflow execution not yet supported from Python for task_type '{}' \
                 (step '{}'); this build only wires real execution for: {}",
                step.task_type,
                step.step_id,
                EXECUTABLE_TASK_TYPES.join(", "),
            )));
        }
    }

    // Build the real oximedia_workflow::Workflow / Task graph.
    let mut real_wf = owf::Workflow::new(workflow.name());
    let mut id_map: HashMap<String, owf::TaskId> = HashMap::new();

    for step in &steps {
        let task_type = step_to_task_type(step)?;
        let task = owf::Task::new(step.step_id.clone(), task_type);
        let task_id = real_wf.add_task(task);
        id_map.insert(step.step_id.clone(), task_id);
    }

    for step in &steps {
        let Some(&to_id) = id_map.get(&step.step_id) else {
            continue;
        };
        for dep in &step.depends_on {
            let Some(&from_id) = id_map.get(dep) else {
                return Err(PyValueError::new_err(format!(
                    "Step '{}' depends on unknown step '{}'",
                    step.step_id, dep
                )));
            };
            real_wf.add_edge(from_id, to_id).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to link workflow steps: {e}"))
            })?;
        }
    }

    // Real cycle detection (defense in depth; `PyWorkflow::add_step` already
    // rejects forward-referencing dependencies, so a cycle should be
    // unreachable, but this uses the real graph's own check rather than
    // assuming that).
    real_wf
        .validate()
        .map_err(|e| PyRuntimeError::new_err(format!("Workflow graph is invalid: {e}")))?;
    let task_order = real_wf
        .topological_sort()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to order workflow steps: {e}")))?;

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create async runtime: {e}")))?;

    let executor = PyWorkflowTaskExecutor;
    let task_results: HashMap<owf::TaskId, owf::TaskResult> = rt.block_on(async {
        let mut completed: std::collections::HashSet<owf::TaskId> =
            std::collections::HashSet::new();
        let mut failed: std::collections::HashSet<owf::TaskId> = std::collections::HashSet::new();
        let mut results: HashMap<owf::TaskId, owf::TaskResult> = HashMap::new();

        for task_id in &task_order {
            // A dependency that failed (or was skipped) means this task
            // cannot honestly run either — mark it failed rather than
            // attempting it, and never report a completion that never
            // happened.
            let deps = real_wf.get_dependencies(task_id);
            if deps.iter().any(|d| failed.contains(d)) {
                failed.insert(*task_id);
                continue;
            }
            let Some(task) = real_wf.get_task(task_id) else {
                continue;
            };
            let result = match executor.execute(task).await {
                Ok(r) => r,
                Err(e) => owf::TaskResult {
                    task_id: *task_id,
                    status: owf::TaskState::Failed,
                    data: None,
                    error: Some(e.to_string()),
                    duration: std::time::Duration::ZERO,
                    outputs: Vec::new(),
                },
            };
            if matches!(result.status, owf::TaskState::Completed) {
                completed.insert(*task_id);
            } else {
                failed.insert(*task_id);
            }
            results.insert(*task_id, result);
        }

        results
    });

    let tasks_total = steps.len();
    let tasks_completed = task_results
        .values()
        .filter(|r| matches!(r.status, owf::TaskState::Completed))
        .count();
    let error = task_results
        .values()
        .find(|r| !matches!(r.status, owf::TaskState::Completed))
        .and_then(|r| r.error.clone());

    let state = if tasks_completed == tasks_total {
        "completed"
    } else {
        "failed"
    }
    .to_string();

    let progress = if tasks_total == 0 {
        0.0
    } else {
        tasks_completed as f64 / tasks_total as f64
    };

    Ok(PyWorkflowStatus {
        state,
        progress,
        tasks_completed,
        tasks_total,
        error,
    })
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register workflow bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWorkflow>()?;
    m.add_class::<PyWorkflowStep>()?;
    m.add_class::<PyWorkflowStatus>()?;
    m.add_class::<PyWorkflowTemplate>()?;
    m.add_function(wrap_pyfunction!(create_workflow, m)?)?;
    m.add_function(wrap_pyfunction!(list_templates, m)?)?;
    m.add_function(wrap_pyfunction!(run_workflow, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Like `.expect()`, but never touches `PyErr`'s `Debug`/`Display` impl.
    /// Those internally require the Python GIL; in a bare `cargo test` /
    /// `nextest` process for this crate (no embedded Python interpreter),
    /// formatting a `PyErr` while already unwinding from a failed `.expect()`
    /// triggers a second panic ("interpreter not initialized") and aborts
    /// the process (SIGABRT) instead of printing a readable message. This
    /// panics with just `msg` on `Err`, which is always safe.
    trait PyResultTestExt<T> {
        fn expect_ok(self, msg: &str) -> T;
    }

    impl<T> PyResultTestExt<T> for PyResult<T> {
        // Deliberately discards the `PyErr` without formatting it (see the
        // trait doc comment above for why `.expect(msg)` is unsafe here).
        #[allow(clippy::match_wild_err_arm)]
        fn expect_ok(self, msg: &str) -> T {
            match self {
                Ok(v) => v,
                Err(_) => panic!("{msg}"),
            }
        }
    }

    #[test]
    fn test_workflow_step_new() {
        let step = PyWorkflowStep::new("s1", "transcode", Some("Transcode"), None);
        assert!(step.is_ok());
        let step = step.expect_ok("valid");
        assert_eq!(step.step_id, "s1");
        assert_eq!(step.task_type, "transcode");
    }

    #[test]
    fn test_workflow_step_invalid_type() {
        let step = PyWorkflowStep::new("s1", "unknown", None, None);
        assert!(step.is_err());
    }

    #[test]
    fn test_workflow_add_steps_and_validate() {
        let mut wf = PyWorkflow::new(Some("Test"));
        let s1 = PyWorkflowStep::new("s1", "qc", None, None).expect_ok("valid");
        let s2 = PyWorkflowStep::new("s2", "transcode", None, Some(vec!["s1".to_string()]))
            .expect_ok("valid");
        wf.add_step(s1).expect_ok("valid");
        wf.add_step(s2).expect_ok("valid");
        assert_eq!(wf.step_count(), 2);

        let issues = wf.validate().expect_ok("validate should succeed");
        assert!(issues.is_empty());
    }

    #[test]
    fn test_workflow_validate_empty() {
        let wf = PyWorkflow::new(None);
        let issues = wf.validate().expect_ok("validate should succeed");
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_list_templates_fn() {
        let templates = list_templates();
        assert!(templates.len() >= 3);
        assert!(templates.iter().any(|t| t.name == "transcode"));
        assert!(templates.iter().any(|t| t.name == "ingest"));
    }

    /// Unique per-call temp path so parallel tests never collide. Preserves
    /// `name`'s extension (if any) at the very end of the filename, since
    /// the transcode task-type dispatches on `Path::extension()`.
    fn unique_tmp(name: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::path::Path::new(name);
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or(name);
        let filename = match path.extension().and_then(|e| e.to_str()) {
            Some(ext) => format!("oximedia-py-workflow-{stem}-{nanos}.{ext}"),
            None => format!("oximedia-py-workflow-{stem}-{nanos}"),
        };
        std::env::temp_dir().join(filename)
    }

    // ── Regression tests: reject task types this build cannot honestly run ──

    #[test]
    fn test_run_workflow_rejects_analysis_task_type() {
        let mut wf = PyWorkflow::new(Some("analysis-wf"));
        let step = PyWorkflowStep::new("s1", "analysis", None, None).expect_ok("valid step");
        wf.add_step(step).expect_ok("add step");

        let result = run_workflow(&wf);
        assert!(
            result.is_err(),
            "'analysis' has no real backend wired in this build and must not \
             fabricate a completed status"
        );
    }

    #[test]
    fn test_run_workflow_rejects_notification_task_type() {
        let mut wf = PyWorkflow::new(Some("notification-wf"));
        let step = PyWorkflowStep::new("s1", "notification", None, None).expect_ok("valid step");
        wf.add_step(step).expect_ok("add step");

        let result = run_workflow(&wf);
        assert!(
            result.is_err(),
            "'notification' has no real delivery transport wired in this \
             build and must not fabricate a completed status"
        );
    }

    #[test]
    fn test_run_workflow_transcode_missing_params_returns_err() {
        let mut wf = PyWorkflow::new(Some("bad-transcode-wf"));
        // "transcode" is executable, but this step has no "input"/"output"
        // params set.
        let step = PyWorkflowStep::new("s1", "transcode", None, None).expect_ok("valid step");
        wf.add_step(step).expect_ok("add step");

        let result = run_workflow(&wf);
        assert!(
            result.is_err(),
            "a transcode step missing required params must return Err"
        );
    }

    // ── Regression tests: real execution, not a hardcoded "completed" ──────

    #[test]
    fn test_run_workflow_real_wait_step_completes() {
        let mut wf = PyWorkflow::new(Some("wait-wf"));
        let mut step = PyWorkflowStep::new("s1", "wait", None, None).expect_ok("valid step");
        step.set_param("duration_secs", "0");
        wf.add_step(step).expect_ok("add step");

        let status = run_workflow(&wf).expect_ok("a real wait step must succeed");
        assert_eq!(status.state, "completed");
        assert_eq!(status.tasks_completed, 1);
        assert_eq!(status.tasks_total, 1);
        assert!((status.progress - 1.0).abs() < f64::EPSILON);
        assert!(status.error.is_none());
    }

    #[test]
    fn test_run_workflow_real_transfer_step_copies_real_file() {
        let src = unique_tmp("transfer-src.bin");
        let dst = unique_tmp("transfer-dst.bin");
        std::fs::write(&src, b"real bytes moved by the real workflow engine")
            .expect("write test source");
        let _ = std::fs::remove_file(&dst);

        let mut wf = PyWorkflow::new(Some("transfer-wf"));
        let mut step = PyWorkflowStep::new("s1", "transfer", None, None).expect_ok("valid step");
        step.set_param("source", src.to_str().expect("valid utf8 path"));
        step.set_param("destination", dst.to_str().expect("valid utf8 path"));
        wf.add_step(step).expect_ok("add step");

        let status = run_workflow(&wf).expect_ok("a real local transfer must succeed");
        assert_eq!(status.state, "completed");
        assert_eq!(status.tasks_completed, 1);

        let copied = std::fs::read(&dst).expect("destination file must really exist");
        assert_eq!(copied, b"real bytes moved by the real workflow engine");

        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
    }

    #[test]
    fn test_run_workflow_transcode_failure_reported_honestly() {
        let mut wf = PyWorkflow::new(Some("transcode-fail-wf"));
        let mut step = PyWorkflowStep::new("s1", "transcode", None, None).expect_ok("valid step");
        step.set_param("input", "/nonexistent/oximedia-py-workflow-test-input.mkv");
        step.set_param(
            "output",
            unique_tmp("transcode-fail-out.mkv")
                .to_str()
                .expect("valid utf8 path"),
        );
        wf.add_step(step).expect_ok("add step");

        // run_workflow itself must still return Ok(status): the *workflow*
        // ran (there was nothing wrong with the request), but the *task*
        // failed for a real reason. That failure must be reported honestly,
        // never papered over as "completed".
        let status = run_workflow(&wf)
            .expect_ok("run_workflow returns Ok(status) even when a step fails for real");
        assert_eq!(status.state, "failed");
        assert_eq!(status.tasks_completed, 0);
        assert!(status.error.is_some());
    }

    #[test]
    fn test_run_workflow_transcode_success_produces_real_output() {
        use oximedia_container::{
            mux::{MatroskaMuxer, MuxerConfig},
            Muxer, Packet, PacketFlags, StreamInfo,
        };
        use oximedia_core::{CodecId, Rational, Timestamp};
        use oximedia_io::MemorySource;

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build test runtime");

        let input = unique_tmp("transcode-ok-in.mkv");
        rt.block_on(async {
            let in_buf = MemorySource::new_writable(64 * 1024);
            let mut muxer = MatroskaMuxer::new(in_buf, MuxerConfig::new());
            let mut video = StreamInfo::new(0, CodecId::Vp9, Rational::new(1, 1000));
            video.codec_params.width = Some(320);
            video.codec_params.height = Some(240);
            muxer.add_stream(video).expect("add stream");
            muxer.write_header().await.expect("write header");
            for i in 0u64..10 {
                let data = vec![0x42u8, 0x00, (i & 0xFF) as u8, 0x01];
                let pkt = Packet::new(
                    0,
                    bytes::Bytes::from(data),
                    Timestamp::new(i as i64 * 33, Rational::new(1, 1000)),
                    PacketFlags::KEYFRAME,
                );
                muxer.write_packet(&pkt).await.expect("write packet");
            }
            muxer.write_trailer().await.expect("write trailer");
            let sink = muxer.into_sink();
            tokio::fs::write(&input, sink.written_data())
                .await
                .expect("write real input file");
        });

        let output = unique_tmp("transcode-ok-out.webm");
        let _ = std::fs::remove_file(&output);

        let mut wf = PyWorkflow::new(Some("transcode-ok-wf"));
        let mut step = PyWorkflowStep::new("s1", "transcode", None, None).expect_ok("valid step");
        step.set_param("input", input.to_str().expect("valid utf8 path"));
        step.set_param("output", output.to_str().expect("valid utf8 path"));
        wf.add_step(step).expect_ok("add step");

        let status =
            run_workflow(&wf).expect_ok("a valid input through a supported container must succeed");
        assert_eq!(status.state, "completed");
        assert_eq!(status.tasks_completed, 1);
        assert!(status.error.is_none());

        let real_len = std::fs::metadata(&output)
            .expect("real output file must exist on disk")
            .len();
        assert!(real_len > 0, "real output file must be non-empty");

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// Regression test for a real bug found in
    /// `oximedia_workflow::executor::WorkflowExecutor::execute()`: its
    /// scheduling loop permanently drops any task whose dependency isn't
    /// satisfied on the first synchronous scan, so a naive bridge to that
    /// engine would silently execute only step "a" and never step "b" in a
    /// 2-step chain while still reporting success. `run_workflow` avoids
    /// that engine and must run *every* step in a dependency chain.
    #[test]
    fn test_run_workflow_dependency_chain_runs_all_steps() {
        let src = unique_tmp("chain-src.bin");
        let dst = unique_tmp("chain-dst.bin");
        std::fs::write(&src, b"chained dependency bytes").expect("write test source");
        let _ = std::fs::remove_file(&dst);

        let mut wf = PyWorkflow::new(Some("chain-wf"));
        let mut step_a = PyWorkflowStep::new("a", "wait", None, None).expect_ok("valid step");
        step_a.set_param("duration_secs", "0");
        wf.add_step(step_a).expect_ok("add step a");

        let mut step_b = PyWorkflowStep::new("b", "transfer", None, Some(vec!["a".to_string()]))
            .expect_ok("valid step");
        step_b.set_param("source", src.to_str().expect("valid utf8 path"));
        step_b.set_param("destination", dst.to_str().expect("valid utf8 path"));
        wf.add_step(step_b).expect_ok("add step b");

        let status = run_workflow(&wf).expect_ok("chained workflow must succeed");
        assert_eq!(status.state, "completed");
        assert_eq!(
            status.tasks_completed, 2,
            "both steps in the dependency chain must actually run, not just the root"
        );
        assert_eq!(status.tasks_total, 2);
        assert!(
            dst.exists(),
            "the dependent step must have really executed and copied the file"
        );

        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
    }
}
