//! Parallel and sequential step execution within workflow stages.
//!
//! A [`WorkflowStage`] is a logical grouping of [`WorkflowStep`]s that either
//! run **in parallel** (all steps spawned concurrently via `std::thread::scope`)
//! or **sequentially** (each step blocks the next).
//!
//! # Example
//!
//! ```rust
//! use oximedia_workflow::parallel_steps::{
//!     ParallelSteps, StepType, WorkflowStage, WorkflowStep,
//! };
//!
//! let mut parallel = ParallelSteps::new("encode-all");
//! parallel.add_step(WorkflowStep::new("pass-1", StepType::Compute { value: 1 }));
//! parallel.add_step(WorkflowStep::new("pass-2", StepType::Compute { value: 2 }));
//! let results = parallel.execute_parallel().expect("all steps should succeed");
//! assert_eq!(results.len(), 2);
//! ```

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Step types
// ---------------------------------------------------------------------------

/// The unit of work a [`WorkflowStep`] performs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepType {
    /// Produce an integer value (useful for testing).
    Compute {
        /// The integer to return as output.
        value: i64,
    },
    /// Fail intentionally with a reason string.
    Fail {
        /// Human-readable failure reason.
        reason: String,
    },
    /// Sleep for a duration (simulates I/O-bound work).
    Wait {
        /// Duration to sleep, in milliseconds.
        duration_ms: u64,
    },
    /// Multiply `value` by `factor` and return the product.
    Transform {
        /// Input value.
        value: i64,
        /// Multiplication factor.
        factor: i64,
    },
}

// ---------------------------------------------------------------------------
// WorkflowStep
// ---------------------------------------------------------------------------

/// A single unit of work inside a [`WorkflowStage`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    /// Human-readable step name.
    pub name: String,
    /// What this step does.
    pub step_type: StepType,
    /// Optional per-step timeout.  When `Some(ms)`, a `Wait` step that
    /// exceeds the timeout is marked as timed-out.
    pub timeout_ms: Option<u64>,
}

impl WorkflowStep {
    /// Create a step with no timeout.
    #[must_use]
    pub fn new(name: impl Into<String>, step_type: StepType) -> Self {
        Self {
            name: name.into(),
            step_type,
            timeout_ms: None,
        }
    }

    /// Set an optional timeout (milliseconds).
    #[must_use]
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
}

// ---------------------------------------------------------------------------
// StepResult
// ---------------------------------------------------------------------------

/// The outcome of executing a single [`WorkflowStep`].
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Name of the step that produced this result.
    pub name: String,
    /// Whether the step completed without error.
    pub success: bool,
    /// Integer output, if the step produced one.
    pub output: Option<i64>,
    /// Error message, if the step failed.
    pub error: Option<String>,
    /// Approximate wall-clock time the step took, in milliseconds.
    pub duration_ms: u64,
}

// ---------------------------------------------------------------------------
// ParallelStepError
// ---------------------------------------------------------------------------

/// Errors from parallel (or sequential) step execution.
#[derive(Debug, thiserror::Error)]
pub enum ParallelStepError {
    /// A single named step failed.
    #[error("Step '{name}' failed: {reason}")]
    StepFailed {
        /// Name of the failing step.
        name: String,
        /// Failure reason.
        reason: String,
    },
    /// Multiple steps failed (reported when `fail_fast = false`).
    #[error("{count} of {total} steps failed")]
    MultipleStepsFailed {
        /// Number of failed steps.
        count: usize,
        /// Total number of steps in the group.
        total: usize,
        /// Individual step results (both successes and failures).
        results: Vec<StepResult>,
    },
    /// A step exceeded its per-step timeout.
    #[error("Step timed out: {name}")]
    Timeout {
        /// Name of the step that timed out.
        name: String,
    },
}

// ---------------------------------------------------------------------------
// Step execution
// ---------------------------------------------------------------------------

/// Execute a single step and return a [`StepResult`].
///
/// This function is `Send + Sync`-safe and is designed to be called from
/// `std::thread::scope` worker threads.
pub fn execute_step(step: &WorkflowStep) -> StepResult {
    let start = Instant::now();

    let (success, output, error) = match &step.step_type {
        StepType::Compute { value } => (true, Some(*value), None),

        StepType::Fail { reason } => (false, None, Some(reason.clone())),

        StepType::Wait { duration_ms } => {
            let wait = Duration::from_millis(*duration_ms);
            // Honour per-step timeout if set.
            let timed_out = if let Some(limit_ms) = step.timeout_ms {
                *duration_ms > limit_ms
            } else {
                false
            };

            if timed_out {
                let elapsed = start.elapsed().as_millis() as u64;
                return StepResult {
                    name: step.name.clone(),
                    success: false,
                    output: None,
                    error: Some(format!("step '{}' timed out", step.name)),
                    duration_ms: elapsed,
                };
            }

            std::thread::sleep(wait);
            (true, None, None)
        }

        StepType::Transform { value, factor } => {
            let result = value.saturating_mul(*factor);
            (true, Some(result), None)
        }
    };

    let duration_ms = start.elapsed().as_millis() as u64;
    StepResult {
        name: step.name.clone(),
        success,
        output,
        error,
        duration_ms,
    }
}

// ---------------------------------------------------------------------------
// ParallelSteps
// ---------------------------------------------------------------------------

/// A named group of steps that execute concurrently.
pub struct ParallelSteps {
    /// Group name (for logging and error messages).
    pub name: String,
    /// The steps in this group.
    pub steps: Vec<WorkflowStep>,
    /// When `true` (the default): return an error as soon as the first
    /// failure is detected rather than waiting for all steps to finish.
    ///
    /// Note: because this implementation uses `std::thread::scope`, all
    /// spawned threads must complete before the scope returns regardless
    /// of `fail_fast`.  The flag controls only whether the *error type*
    /// returned is [`ParallelStepError::StepFailed`] (single, fail-fast)
    /// or [`ParallelStepError::MultipleStepsFailed`] (aggregated).
    pub fail_fast: bool,
}

impl ParallelSteps {
    /// Create a new group with `fail_fast = true` and no steps.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            steps: Vec::new(),
            fail_fast: true,
        }
    }

    /// Set the `fail_fast` flag.
    #[must_use]
    pub fn with_fail_fast(mut self, fail_fast: bool) -> Self {
        self.fail_fast = fail_fast;
        self
    }

    /// Append a step to the group.
    pub fn add_step(&mut self, step: WorkflowStep) -> &mut Self {
        self.steps.push(step);
        self
    }

    /// Execute all steps in parallel using `std::thread::scope`.
    ///
    /// All worker threads are joined before this function returns.
    /// Results are collected in the same order as [`Self::steps`].
    ///
    /// # Errors
    ///
    /// Returns [`ParallelStepError::StepFailed`] if `fail_fast = true` and
    /// one or more steps fail, or [`ParallelStepError::MultipleStepsFailed`]
    /// if `fail_fast = false` and multiple steps fail.
    pub fn execute_parallel(&self) -> Result<Vec<StepResult>, ParallelStepError> {
        if self.steps.is_empty() {
            return Ok(Vec::new());
        }

        // Collect results using thread::scope so all threads are joined before
        // we return.  We use a Mutex-protected Vec to gather results.
        let results: std::sync::Mutex<Vec<(usize, StepResult)>> =
            std::sync::Mutex::new(Vec::with_capacity(self.steps.len()));

        std::thread::scope(|scope| {
            for (idx, step) in self.steps.iter().enumerate() {
                let results_ref = &results;
                scope.spawn(move || {
                    let result = execute_step(step);
                    if let Ok(mut guard) = results_ref.lock() {
                        guard.push((idx, result));
                    }
                });
            }
        });

        // Reconstruct in original order.
        let mut collected = results.into_inner().unwrap_or_default();
        collected.sort_by_key(|(idx, _)| *idx);
        let ordered: Vec<StepResult> = collected.into_iter().map(|(_, r)| r).collect();

        // Check for failures.
        let failed: Vec<&StepResult> = ordered.iter().filter(|r| !r.success).collect();

        if failed.is_empty() {
            return Ok(ordered);
        }

        if self.fail_fast {
            // Return the first failure.
            let first = &failed[0];
            return Err(ParallelStepError::StepFailed {
                name: first.name.clone(),
                reason: first
                    .error
                    .clone()
                    .unwrap_or_else(|| "unknown error".to_string()),
            });
        }

        // Not fail-fast: return aggregated error.
        Err(ParallelStepError::MultipleStepsFailed {
            count: failed.len(),
            total: ordered.len(),
            results: ordered,
        })
    }
}

// ---------------------------------------------------------------------------
// WorkflowStage
// ---------------------------------------------------------------------------

/// A logical stage within a workflow, containing steps that run either
/// sequentially or in parallel.
pub enum WorkflowStage {
    /// Steps run one after another; fails on the first error.
    Sequential(Vec<WorkflowStep>),
    /// All steps start at the same time; results are joined.
    Parallel(Vec<WorkflowStep>),
}

impl WorkflowStage {
    /// Execute this stage and return all step results.
    ///
    /// # Errors
    ///
    /// Returns [`ParallelStepError`] if any step fails.
    pub fn execute(&self) -> Result<Vec<StepResult>, ParallelStepError> {
        match self {
            Self::Sequential(steps) => {
                let mut results = Vec::with_capacity(steps.len());
                for step in steps {
                    let r = execute_step(step);
                    if !r.success {
                        let reason = r
                            .error
                            .clone()
                            .unwrap_or_else(|| "unknown error".to_string());
                        let name = r.name.clone();
                        results.push(r);
                        return Err(ParallelStepError::StepFailed { name, reason });
                    }
                    results.push(r);
                }
                Ok(results)
            }
            Self::Parallel(steps) => {
                let mut group = ParallelSteps::new("stage");
                group.fail_fast = true;
                for step in steps {
                    group.steps.push(step.clone());
                }
                group.execute_parallel()
            }
        }
    }

    /// Return the number of steps in this stage.
    #[must_use]
    pub fn step_count(&self) -> usize {
        match self {
            Self::Sequential(steps) | Self::Parallel(steps) => steps.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ParallelSteps — success paths
    // -----------------------------------------------------------------------

    #[test]
    fn parallel_all_success_returns_results() {
        let mut group = ParallelSteps::new("grp");
        group.add_step(WorkflowStep::new("a", StepType::Compute { value: 10 }));
        group.add_step(WorkflowStep::new("b", StepType::Compute { value: 20 }));
        group.add_step(WorkflowStep::new(
            "c",
            StepType::Transform {
                value: 3,
                factor: 7,
            },
        ));
        let results = group.execute_parallel().expect("all should succeed");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].output, Some(10));
        assert_eq!(results[1].output, Some(20));
        assert_eq!(results[2].output, Some(21));
    }

    #[test]
    fn parallel_empty_steps_returns_empty() {
        let group = ParallelSteps::new("empty");
        let results = group
            .execute_parallel()
            .expect("empty group should succeed");
        assert!(results.is_empty());
    }

    #[test]
    fn parallel_single_step_success() {
        let mut group = ParallelSteps::new("single");
        group.add_step(WorkflowStep::new("only", StepType::Compute { value: 42 }));
        let results = group.execute_parallel().expect("should succeed");
        assert_eq!(results[0].output, Some(42));
        assert!(results[0].success);
    }

    // -----------------------------------------------------------------------
    // ParallelSteps — failure paths
    // -----------------------------------------------------------------------

    #[test]
    fn parallel_one_fail_fast_gives_step_failed_error() {
        let mut group = ParallelSteps::new("fail-fast-grp");
        group.fail_fast = true;
        group.add_step(WorkflowStep::new("ok", StepType::Compute { value: 1 }));
        group.add_step(WorkflowStep::new(
            "bad",
            StepType::Fail {
                reason: "oops".to_string(),
            },
        ));
        let err = group.execute_parallel().expect_err("should fail");
        assert!(matches!(err, ParallelStepError::StepFailed { .. }));
    }

    #[test]
    fn parallel_one_fail_not_fast_gives_multiple_failed() {
        let mut group = ParallelSteps::new("no-ff");
        group.fail_fast = false;
        group.add_step(WorkflowStep::new("ok", StepType::Compute { value: 1 }));
        group.add_step(WorkflowStep::new(
            "bad",
            StepType::Fail {
                reason: "nope".to_string(),
            },
        ));
        let err = group.execute_parallel().expect_err("should fail");
        if let ParallelStepError::MultipleStepsFailed { count, total, .. } = err {
            assert_eq!(count, 1);
            assert_eq!(total, 2);
        } else {
            panic!("expected MultipleStepsFailed");
        }
    }

    #[test]
    fn parallel_all_fail_no_fast_aggregates() {
        let mut group = ParallelSteps::new("all-fail");
        group.fail_fast = false;
        group.add_step(WorkflowStep::new(
            "a",
            StepType::Fail {
                reason: "err-a".to_string(),
            },
        ));
        group.add_step(WorkflowStep::new(
            "b",
            StepType::Fail {
                reason: "err-b".to_string(),
            },
        ));
        let err = group.execute_parallel().expect_err("should fail");
        if let ParallelStepError::MultipleStepsFailed { count, total, .. } = err {
            assert_eq!(count, 2);
            assert_eq!(total, 2);
        } else {
            panic!("expected MultipleStepsFailed");
        }
    }

    // -----------------------------------------------------------------------
    // execute_step — direct tests
    // -----------------------------------------------------------------------

    #[test]
    fn execute_compute_step() {
        let step = WorkflowStep::new("s", StepType::Compute { value: -7 });
        let r = execute_step(&step);
        assert!(r.success);
        assert_eq!(r.output, Some(-7));
        assert!(r.error.is_none());
    }

    #[test]
    fn execute_fail_step() {
        let step = WorkflowStep::new(
            "s",
            StepType::Fail {
                reason: "bad".to_string(),
            },
        );
        let r = execute_step(&step);
        assert!(!r.success);
        assert_eq!(r.error.as_deref(), Some("bad"));
    }

    #[test]
    fn execute_transform_step() {
        let step = WorkflowStep::new(
            "s",
            StepType::Transform {
                value: 6,
                factor: 9,
            },
        );
        let r = execute_step(&step);
        assert!(r.success);
        assert_eq!(r.output, Some(54));
    }

    #[test]
    fn execute_wait_step_timeout() {
        // A Wait step whose duration exceeds the per-step timeout should fail.
        let step = WorkflowStep::new("s", StepType::Wait { duration_ms: 500 }).with_timeout(10);
        let r = execute_step(&step);
        assert!(!r.success, "should time out");
        assert!(r.error.is_some());
    }

    // -----------------------------------------------------------------------
    // WorkflowStage
    // -----------------------------------------------------------------------

    #[test]
    fn sequential_stage_executes_in_order() {
        let steps = vec![
            WorkflowStep::new("first", StepType::Compute { value: 1 }),
            WorkflowStep::new("second", StepType::Compute { value: 2 }),
            WorkflowStep::new("third", StepType::Compute { value: 3 }),
        ];
        let stage = WorkflowStage::Sequential(steps);
        let results = stage.execute().expect("should succeed");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].name, "first");
        assert_eq!(results[1].name, "second");
        assert_eq!(results[2].name, "third");
    }

    #[test]
    fn sequential_stage_stops_on_first_failure() {
        let steps = vec![
            WorkflowStep::new("ok", StepType::Compute { value: 1 }),
            WorkflowStep::new(
                "bad",
                StepType::Fail {
                    reason: "stop".to_string(),
                },
            ),
            WorkflowStep::new("never", StepType::Compute { value: 99 }),
        ];
        let stage = WorkflowStage::Sequential(steps);
        let err = stage.execute().expect_err("should fail");
        assert!(matches!(err, ParallelStepError::StepFailed { .. }));
    }

    #[test]
    fn parallel_stage_all_success() {
        let steps = vec![
            WorkflowStep::new("p1", StepType::Compute { value: 10 }),
            WorkflowStep::new(
                "p2",
                StepType::Transform {
                    value: 3,
                    factor: 4,
                },
            ),
        ];
        let stage = WorkflowStage::Parallel(steps);
        let results = stage.execute().expect("should succeed");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn stage_step_count() {
        let seq = WorkflowStage::Sequential(vec![
            WorkflowStep::new("a", StepType::Compute { value: 0 }),
            WorkflowStep::new("b", StepType::Compute { value: 0 }),
        ]);
        assert_eq!(seq.step_count(), 2);

        let par =
            WorkflowStage::Parallel(vec![WorkflowStep::new("x", StepType::Compute { value: 0 })]);
        assert_eq!(par.step_count(), 1);
    }

    #[test]
    fn parallel_and_sequential_same_results_all_success() {
        let steps = vec![
            WorkflowStep::new("a", StepType::Compute { value: 5 }),
            WorkflowStep::new("b", StepType::Compute { value: 10 }),
        ];
        let seq_results = WorkflowStage::Sequential(steps.clone())
            .execute()
            .expect("seq ok");
        let par_results = WorkflowStage::Parallel(steps).execute().expect("par ok");

        let seq_outputs: Vec<Option<i64>> = seq_results.iter().map(|r| r.output).collect();
        let par_outputs: Vec<Option<i64>> = par_results.iter().map(|r| r.output).collect();
        assert_eq!(seq_outputs, par_outputs);
    }
}
