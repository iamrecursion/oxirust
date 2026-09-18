//! Step-through executor wrapper that logs intermediate tensor statistics.
//!
//! `StepExecutor<E>` wraps any `TlExecutor` and records `IntermediateValue`
//! statistics for each operation, optionally guarded by `BreakpointCondition`s.

use ndarray::ArrayD;

use crate::ops::{ElemOp, ReduceOp};
use crate::traits::TlExecutor;

/// Conditions that decide whether an intermediate value is recorded.
#[derive(Debug, Clone)]
pub enum BreakpointCondition {
    /// Break at the operation with this sequential index.
    NodeIndex(usize),
    /// Break whenever the output contains NaN.
    OnNaN,
    /// Break whenever the output contains Inf.
    OnInf,
    /// Always record — equivalent to a full trace.
    Always,
}

/// Statistics snapshot of a tensor value at a specific execution step.
#[derive(Debug, Clone)]
pub struct IntermediateValue {
    /// Sequential operation index (0-based).
    pub step: usize,
    /// Human-readable name of the operation.
    pub operation: String,
    /// Shape of the tensor.
    pub shape: Vec<usize>,
    /// Minimum value (NaN-safe via fold).
    pub min: f64,
    /// Maximum value (NaN-safe via fold).
    pub max: f64,
    /// Mean value.
    pub mean: f64,
    /// Whether any element is NaN.
    pub has_nan: bool,
    /// Whether any element is ±Inf.
    pub has_inf: bool,
    /// Total number of elements.
    pub element_count: usize,
}

impl IntermediateValue {
    /// Build statistics from a tensor.
    pub fn from_tensor(step: usize, op: &str, tensor: &ArrayD<f64>) -> Self {
        let element_count = tensor.len();
        let has_nan = tensor.iter().any(|x| x.is_nan());
        let has_inf = tensor.iter().any(|x| x.is_infinite());

        let (min, max, sum) = tensor.iter().cloned().fold(
            (f64::INFINITY, f64::NEG_INFINITY, 0.0f64),
            |(mn, mx, s), v| (mn.min(v), mx.max(v), s + v),
        );

        let (min, max) = if element_count == 0 {
            (0.0, 0.0)
        } else {
            (min, max)
        };

        let mean = if element_count == 0 {
            0.0
        } else {
            sum / element_count as f64
        };

        Self {
            step,
            operation: op.to_owned(),
            shape: tensor.shape().to_vec(),
            min,
            max,
            mean,
            has_nan,
            has_inf,
            element_count,
        }
    }
}

/// Wraps any `TlExecutor` and logs `IntermediateValue` snapshots at each operation.
///
/// A snapshot is recorded when at least one active `BreakpointCondition` triggers.
/// If no conditions are added no logging occurs; add `BreakpointCondition::Always`
/// to capture every step.
pub struct StepExecutor<E> {
    /// The inner executor that performs actual computation.
    pub inner: E,
    conditions: Vec<BreakpointCondition>,
    /// Accumulated log of intermediate values.
    pub log: Vec<IntermediateValue>,
    step_count: usize,
}

impl<E> StepExecutor<E> {
    /// Create a new `StepExecutor` wrapping `inner` with no active conditions.
    pub fn new(inner: E) -> Self {
        Self {
            inner,
            conditions: Vec::new(),
            log: Vec::new(),
            step_count: 0,
        }
    }

    /// Add a breakpoint condition.
    pub fn add_condition(&mut self, cond: BreakpointCondition) {
        self.conditions.push(cond);
    }

    /// View the accumulated log.
    pub fn log(&self) -> &[IntermediateValue] {
        &self.log
    }

    /// Total number of operations executed so far.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Clear the accumulated log (step count is not reset).
    pub fn clear_log(&mut self) {
        self.log.clear();
    }

    /// Returns true if any logged entry contains NaN.
    pub fn has_nan_in_log(&self) -> bool {
        self.log.iter().any(|v| v.has_nan)
    }

    /// Returns true if any logged entry contains Inf.
    pub fn has_inf_in_log(&self) -> bool {
        self.log.iter().any(|v| v.has_inf)
    }

    /// One-line human-readable summary of the execution log.
    pub fn summary(&self) -> String {
        let nan_count = self.log.iter().filter(|v| v.has_nan).count();
        let inf_count = self.log.iter().filter(|v| v.has_inf).count();
        format!(
            "StepExecutor: {} steps executed, {} logged, {} NaN entries, {} Inf entries",
            self.step_count,
            self.log.len(),
            nan_count,
            inf_count,
        )
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn should_log(&self, step: usize, iv: &IntermediateValue) -> bool {
        self.conditions.iter().any(|cond| match cond {
            BreakpointCondition::Always => true,
            BreakpointCondition::NodeIndex(idx) => *idx == step,
            BreakpointCondition::OnNaN => iv.has_nan,
            BreakpointCondition::OnInf => iv.has_inf,
        })
    }

    fn record_if_triggered(&mut self, iv: IntermediateValue) {
        if self.should_log(iv.step, &iv) {
            self.log.push(iv);
        }
    }
}

/// `TlExecutor` implementation for executors whose tensor type is `ArrayD<f64>`.
impl<E> TlExecutor for StepExecutor<E>
where
    E: TlExecutor<Tensor = ArrayD<f64>>,
{
    type Tensor = ArrayD<f64>;
    type Error = E::Error;

    fn einsum(&mut self, spec: &str, inputs: &[Self::Tensor]) -> Result<Self::Tensor, Self::Error> {
        let step = self.step_count;
        self.step_count += 1;
        let result = self.inner.einsum(spec, inputs)?;
        let iv = IntermediateValue::from_tensor(step, &format!("einsum({})", spec), &result);
        self.record_if_triggered(iv);
        Ok(result)
    }

    fn elem_op(&mut self, op: ElemOp, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        let step = self.step_count;
        self.step_count += 1;
        let result = self.inner.elem_op(op, x)?;
        let iv = IntermediateValue::from_tensor(step, &format!("elem_op({:?})", op), &result);
        self.record_if_triggered(iv);
        Ok(result)
    }

    fn elem_op_binary(
        &mut self,
        op: ElemOp,
        x: &Self::Tensor,
        y: &Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error> {
        let step = self.step_count;
        self.step_count += 1;
        let result = self.inner.elem_op_binary(op, x, y)?;
        let iv =
            IntermediateValue::from_tensor(step, &format!("elem_op_binary({:?})", op), &result);
        self.record_if_triggered(iv);
        Ok(result)
    }

    fn reduce(
        &mut self,
        op: ReduceOp,
        x: &Self::Tensor,
        axes: &[usize],
    ) -> Result<Self::Tensor, Self::Error> {
        let step = self.step_count;
        self.step_count += 1;
        let result = self.inner.reduce(op, x, axes)?;
        let iv = IntermediateValue::from_tensor(step, &format!("reduce({:?})", op), &result);
        self.record_if_triggered(iv);
        Ok(result)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ExecutorError;
    use ndarray::{Array, IxDyn};

    // Minimal executor whose Tensor = ArrayD<f64> for testing StepExecutor.
    struct ArrayExecutor;

    impl TlExecutor for ArrayExecutor {
        type Tensor = ArrayD<f64>;
        type Error = ExecutorError;

        fn einsum(
            &mut self,
            _spec: &str,
            inputs: &[Self::Tensor],
        ) -> Result<Self::Tensor, Self::Error> {
            Ok(inputs[0].clone())
        }

        fn elem_op(&mut self, _op: ElemOp, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
            Ok(x.clone())
        }

        fn elem_op_binary(
            &mut self,
            _op: ElemOp,
            x: &Self::Tensor,
            _y: &Self::Tensor,
        ) -> Result<Self::Tensor, Self::Error> {
            Ok(x.clone())
        }

        fn reduce(
            &mut self,
            _op: ReduceOp,
            x: &Self::Tensor,
            _axes: &[usize],
        ) -> Result<Self::Tensor, Self::Error> {
            Ok(x.clone())
        }
    }

    fn make_tensor(data: &[f64]) -> ArrayD<f64> {
        Array::from_shape_vec(IxDyn(&[data.len()]), data.to_vec()).unwrap()
    }

    #[test]
    fn test_step_executor_creates() {
        let exec = StepExecutor::new(ArrayExecutor);
        assert_eq!(exec.step_count(), 0);
        assert!(exec.log().is_empty());
    }

    #[test]
    fn test_intermediate_value_from_tensor() {
        let t = make_tensor(&[1.0, 2.0, 3.0, 4.0]);
        let iv = IntermediateValue::from_tensor(0, "test_op", &t);
        assert_eq!(iv.step, 0);
        assert_eq!(iv.operation, "test_op");
        assert_eq!(iv.element_count, 4);
        assert!((iv.min - 1.0).abs() < 1e-10);
        assert!((iv.max - 4.0).abs() < 1e-10);
        assert!((iv.mean - 2.5).abs() < 1e-10);
        assert!(!iv.has_nan);
        assert!(!iv.has_inf);
    }

    #[test]
    fn test_always_condition_logs_all() {
        let mut exec = StepExecutor::new(ArrayExecutor);
        exec.add_condition(BreakpointCondition::Always);
        let t = make_tensor(&[1.0, 2.0]);
        exec.einsum("ij->ij", std::slice::from_ref(&t)).unwrap();
        exec.elem_op(ElemOp::Relu, &t).unwrap();
        exec.elem_op_binary(ElemOp::Add, &t, &t).unwrap();
        assert_eq!(exec.log().len(), 3, "all 3 ops should be logged");
        assert_eq!(exec.step_count(), 3);
    }

    #[test]
    fn test_nan_detection_in_log() {
        let mut exec = StepExecutor::new(ArrayExecutor);
        exec.add_condition(BreakpointCondition::OnNaN);
        // Normal tensor should not be logged.
        let normal = make_tensor(&[1.0, 2.0]);
        exec.einsum("i->i", &[normal]).unwrap();
        assert!(exec.log().is_empty(), "no NaN, should not log");

        // NaN tensor should be logged.
        let nan_tensor = make_tensor(&[f64::NAN, 1.0]);
        exec.einsum("i->i", &[nan_tensor]).unwrap();
        assert_eq!(exec.log().len(), 1, "NaN tensor should be logged");
        assert!(exec.has_nan_in_log());
    }

    #[test]
    fn test_step_count_and_clear() {
        let mut exec = StepExecutor::new(ArrayExecutor);
        exec.add_condition(BreakpointCondition::Always);
        let t = make_tensor(&[1.0]);
        exec.einsum("i->i", std::slice::from_ref(&t)).unwrap();
        exec.einsum("i->i", std::slice::from_ref(&t)).unwrap();
        assert_eq!(exec.step_count(), 2);
        assert_eq!(exec.log().len(), 2);
        exec.clear_log();
        assert_eq!(exec.log().len(), 0);
        assert_eq!(exec.step_count(), 2, "step_count preserved after clear");
        let summary = exec.summary();
        assert!(summary.contains("2 steps"));
    }
}
