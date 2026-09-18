//! Progress callbacks for training and compilation.
//!
//! Provides Python-callable progress tracking for long-running TensorLogic operations.
//! Compatible with tqdm-style progress bars.

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::time::Instant;

/// A single training progress event dispatched to Python callbacks.
///
/// Compatible with tqdm's `update()` interface:
/// - `step` maps to tqdm `n`
/// - `total_steps` maps to tqdm `total`
#[pyclass(module = "pytensorlogic")]
#[derive(Debug, Clone)]
pub struct PyProgressEvent {
    #[pyo3(get)]
    pub step: usize,
    #[pyo3(get)]
    pub total_steps: usize,
    #[pyo3(get)]
    pub loss: f64,
    #[pyo3(get)]
    pub grad_norm: f64,
    #[pyo3(get)]
    pub elapsed_ms: f64,
}

#[pymethods]
impl PyProgressEvent {
    #[new]
    pub fn new(
        step: usize,
        total_steps: usize,
        loss: f64,
        grad_norm: f64,
        elapsed_ms: f64,
    ) -> Self {
        PyProgressEvent {
            step,
            total_steps,
            loss,
            grad_norm,
            elapsed_ms,
        }
    }

    /// Progress as a fraction in [0.0, 1.0].
    pub fn progress(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            (self.step as f64 / self.total_steps as f64).clamp(0.0, 1.0)
        }
    }

    /// Whether this is the final event (step == total_steps).
    pub fn is_complete(&self) -> bool {
        self.step >= self.total_steps
    }

    /// Estimated remaining time in milliseconds.
    pub fn estimated_remaining_ms(&self) -> f64 {
        if self.step == 0 {
            return f64::INFINITY;
        }
        let per_step = self.elapsed_ms / self.step as f64;
        per_step * (self.total_steps.saturating_sub(self.step)) as f64
    }

    fn __repr__(&self) -> String {
        format!(
            "ProgressEvent(step={}/{}, loss={:.6}, grad_norm={:.4}, elapsed={:.1}ms)",
            self.step, self.total_steps, self.loss, self.grad_norm, self.elapsed_ms
        )
    }

    /// Return a dict suitable for tqdm.set_postfix(**event.as_dict()).
    pub fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("loss", self.loss)?;
        d.set_item("grad_norm", self.grad_norm)?;
        d.set_item("progress", self.progress())?;
        Ok(d)
    }
}

/// A single compilation progress event.
#[pyclass(module = "pytensorlogic")]
#[derive(Debug, Clone)]
pub struct PyCompilationEvent {
    #[pyo3(get)]
    pub phase: String,
    #[pyo3(get)]
    pub progress_pct: f64,
    #[pyo3(get)]
    pub message: String,
    #[pyo3(get)]
    pub nodes_processed: usize,
    #[pyo3(get)]
    pub total_nodes: usize,
}

#[pymethods]
impl PyCompilationEvent {
    #[new]
    pub fn new(
        phase: String,
        progress_pct: f64,
        message: String,
        nodes_processed: usize,
        total_nodes: usize,
    ) -> Self {
        PyCompilationEvent {
            phase,
            progress_pct: progress_pct.clamp(0.0, 100.0),
            message,
            nodes_processed,
            total_nodes,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.progress_pct >= 100.0
    }

    fn __repr__(&self) -> String {
        format!(
            "CompilationEvent(phase='{}', {:.1}%, '{}')",
            self.phase, self.progress_pct, self.message
        )
    }
}

/// A mock training result.
#[pyclass(module = "pytensorlogic")]
#[derive(Debug, Clone)]
pub struct PyTrainingResult {
    #[pyo3(get)]
    pub losses: Vec<f64>,
    #[pyo3(get)]
    pub total_steps: usize,
    #[pyo3(get)]
    pub total_time_ms: f64,
}

#[pymethods]
impl PyTrainingResult {
    #[new]
    pub fn new(losses: Vec<f64>, total_steps: usize, total_time_ms: f64) -> Self {
        PyTrainingResult {
            losses,
            total_steps,
            total_time_ms,
        }
    }

    pub fn final_loss(&self) -> f64 {
        self.losses.last().copied().unwrap_or(f64::NAN)
    }

    pub fn average_loss(&self) -> f64 {
        if self.losses.is_empty() {
            return 0.0;
        }
        self.losses.iter().sum::<f64>() / self.losses.len() as f64
    }

    pub fn loss_reduction(&self) -> f64 {
        if self.losses.len() < 2 {
            return 0.0;
        }
        let first = self.losses[0];
        let last = self.losses[self.losses.len() - 1];
        if first == 0.0 {
            0.0
        } else {
            (first - last) / first
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TrainingResult(steps={}, final_loss={:.6}, avg_loss={:.6})",
            self.total_steps,
            self.final_loss(),
            self.average_loss()
        )
    }
}

/// A training loop that fires Python progress callbacks.
///
/// Simulates a training process with decaying loss for demonstration purposes.
/// In production this would wrap the actual TensorLogic execution graph.
#[pyclass(module = "pytensorlogic")]
pub struct PyTrainingLoop {
    initial_loss: f64,
    decay_rate: f64,
    noise_scale: f64,
}

#[pymethods]
impl PyTrainingLoop {
    #[new]
    #[pyo3(signature = (initial_loss=1.0, decay_rate=0.1, noise_scale=0.01))]
    pub fn new(initial_loss: f64, decay_rate: f64, noise_scale: f64) -> Self {
        PyTrainingLoop {
            initial_loss,
            decay_rate,
            noise_scale,
        }
    }

    /// Run training for `n_steps` steps, firing `callback` at each step.
    ///
    /// `callback` should accept a `ProgressEvent` argument:
    ///   def my_callback(event): print(f"Step {event.step}: loss={event.loss:.4f}")
    #[pyo3(signature = (n_steps, callback=None))]
    pub fn run(
        &self,
        py: Python<'_>,
        n_steps: usize,
        callback: Option<Py<PyAny>>,
    ) -> PyResult<PyTrainingResult> {
        let start = Instant::now();
        let mut losses = Vec::with_capacity(n_steps);
        let mut loss = self.initial_loss;

        for step in 1..=n_steps {
            // Simulate loss decay with small noise
            let noise = (step as f64 * 1.23456).sin() * self.noise_scale;
            loss = loss * (1.0 - self.decay_rate) + noise.abs();
            losses.push(loss);

            let grad_norm = loss * 2.0 + 0.01;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            if let Some(ref cb) = callback {
                let event = PyProgressEvent::new(step, n_steps, loss, grad_norm, elapsed_ms);
                cb.call1(py, (event,))?;
            }
        }

        let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok(PyTrainingResult::new(losses, n_steps, total_time_ms))
    }

    /// Iterate over compilation events for a given expression string.
    ///
    /// Yields `CompilationEvent` objects suitable for use in a for loop.
    pub fn compile_with_progress(&self, _expr: &str) -> PyResult<Vec<PyCompilationEvent>> {
        let phases = [
            ("parse", 10.0, "Parsing expression"),
            ("type_check", 25.0, "Type checking"),
            ("optimize", 50.0, "Optimizing"),
            ("lower", 75.0, "Lowering to IR"),
            ("codegen", 90.0, "Generating einsum graph"),
            ("finalize", 100.0, "Finalizing"),
        ];

        Ok(phases
            .iter()
            .enumerate()
            .map(|(i, (phase, pct, msg))| {
                PyCompilationEvent::new(
                    phase.to_string(),
                    *pct,
                    msg.to_string(),
                    i + 1,
                    phases.len(),
                )
            })
            .collect())
    }
}

/// Register all progress types with the Python module.
pub fn register_progress_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProgressEvent>()?;
    m.add_class::<PyCompilationEvent>()?;
    m.add_class::<PyTrainingResult>()?;
    m.add_class::<PyTrainingLoop>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_event_progress_zero_steps() {
        let e = PyProgressEvent::new(0, 0, 1.0, 0.1, 0.0);
        assert_eq!(e.progress(), 0.0);
    }

    #[test]
    fn test_progress_event_progress_fraction() {
        let e = PyProgressEvent::new(5, 10, 0.5, 0.1, 100.0);
        assert!((e.progress() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_progress_event_progress_clamped() {
        let e = PyProgressEvent::new(15, 10, 0.0, 0.0, 0.0);
        assert_eq!(e.progress(), 1.0);
    }

    #[test]
    fn test_progress_event_is_complete_true() {
        let e = PyProgressEvent::new(10, 10, 0.01, 0.01, 500.0);
        assert!(e.is_complete());
    }

    #[test]
    fn test_progress_event_is_complete_false() {
        let e = PyProgressEvent::new(3, 10, 0.5, 0.2, 300.0);
        assert!(!e.is_complete());
    }

    #[test]
    fn test_progress_event_estimated_remaining() {
        let e = PyProgressEvent::new(5, 10, 0.5, 0.1, 500.0);
        // 500ms elapsed for 5 steps = 100ms/step, 5 remaining = 500ms
        let remaining = e.estimated_remaining_ms();
        assert!((remaining - 500.0).abs() < 1.0);
    }

    #[test]
    fn test_compilation_event_clamped() {
        let e = PyCompilationEvent::new("test".to_string(), 150.0, "msg".to_string(), 1, 1);
        assert_eq!(e.progress_pct, 100.0);
    }

    #[test]
    fn test_compilation_event_is_complete() {
        let e = PyCompilationEvent::new("done".to_string(), 100.0, "".to_string(), 6, 6);
        assert!(e.is_complete());
    }

    #[test]
    fn test_compilation_event_not_complete() {
        let e = PyCompilationEvent::new("parse".to_string(), 10.0, "".to_string(), 1, 6);
        assert!(!e.is_complete());
    }

    #[test]
    fn test_training_result_final_loss() {
        let r = PyTrainingResult::new(vec![1.0, 0.5, 0.2], 3, 100.0);
        assert!((r.final_loss() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_training_result_average_loss() {
        let r = PyTrainingResult::new(vec![1.0, 0.0], 2, 50.0);
        assert!((r.average_loss() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_training_result_loss_reduction() {
        let r = PyTrainingResult::new(vec![1.0, 0.5], 2, 50.0);
        // (1.0 - 0.5) / 1.0 = 0.5
        assert!((r.loss_reduction() - 0.5).abs() < 1e-9);
    }
}
