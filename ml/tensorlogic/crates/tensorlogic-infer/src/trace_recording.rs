//! Execution trace recording for debugging and performance analysis.
//!
//! Records intermediate tensor shapes, timings, and operation details
//! during graph execution for post-hoc analysis.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A single operation trace entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedTraceEntry {
    pub step: usize,
    pub operation: String,
    pub device_id: Option<String>,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shape: Vec<usize>,
    pub duration_us: f64,
    pub output_elements: usize,
    pub memory_bytes: usize,
}

impl RecordedTraceEntry {
    /// Create a new trace entry with defaults.
    pub fn new(step: usize, operation: impl Into<String>) -> Self {
        RecordedTraceEntry {
            step,
            operation: operation.into(),
            device_id: None,
            input_shapes: Vec::new(),
            output_shape: Vec::new(),
            duration_us: 0.0,
            output_elements: 0,
            memory_bytes: 0,
        }
    }

    /// Set device identifier (builder pattern).
    pub fn with_device_id(mut self, device_id: impl Into<String>) -> Self {
        self.device_id = Some(device_id.into());
        self
    }

    /// Set input shapes (builder pattern).
    pub fn with_input_shapes(mut self, shapes: Vec<Vec<usize>>) -> Self {
        self.input_shapes = shapes;
        self
    }

    /// Set output shape and derive element count / memory (builder pattern).
    pub fn with_output_shape(mut self, shape: Vec<usize>) -> Self {
        self.output_elements = shape.iter().product();
        self.memory_bytes = self.output_elements * 8; // assume f64
        self.output_shape = shape;
        self
    }

    /// Set duration from a `Duration` (builder pattern).
    pub fn with_duration(mut self, d: Duration) -> Self {
        self.duration_us = d.as_secs_f64() * 1e6;
        self
    }

    /// Throughput in elements per microsecond.
    pub fn throughput_elements_per_us(&self) -> f64 {
        if self.duration_us < 1e-9 {
            0.0
        } else {
            self.output_elements as f64 / self.duration_us
        }
    }
}

/// Complete execution trace.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecordedExecutionTrace {
    pub entries: Vec<RecordedTraceEntry>,
    pub total_duration_us: f64,
    pub metadata: HashMap<String, String>,
}

impl RecordedExecutionTrace {
    /// Create an empty trace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a trace entry, accumulating total duration.
    pub fn add_entry(&mut self, entry: RecordedTraceEntry) {
        self.total_duration_us += entry.duration_us;
        self.entries.push(entry);
    }

    /// Attach metadata (builder pattern).
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Number of recorded steps.
    pub fn step_count(&self) -> usize {
        self.entries.len()
    }

    /// Total memory across all entries.
    pub fn total_memory_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.memory_bytes).sum()
    }

    /// Peak memory of any single entry.
    pub fn peak_memory_bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|e| e.memory_bytes)
            .max()
            .unwrap_or(0)
    }

    /// Return the N slowest operations, sorted descending by duration.
    pub fn slowest_ops(&self, n: usize) -> Vec<&RecordedTraceEntry> {
        let mut sorted: Vec<_> = self.entries.iter().collect();
        sorted.sort_by(|a, b| {
            b.duration_us
                .partial_cmp(&a.duration_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(n);
        sorted
    }

    /// Export to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Summary for one operation type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpSummary {
    pub count: usize,
    pub total_duration_us: f64,
    pub total_memory_bytes: usize,
}

/// Summary for one device.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeviceSummary {
    pub op_count: usize,
    pub total_duration_us: f64,
    pub total_memory_bytes: usize,
}

/// Communication hotspot in distributed traces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationBottleneck {
    pub operation: String,
    pub total_duration_us: f64,
    pub ratio_of_total: f64,
    pub call_count: usize,
}

/// Load balance metrics derived from per-device timing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalanceMetrics {
    pub device_count: usize,
    pub total_duration_us: f64,
    pub ideal_duration_us: f64,
    pub max_duration_us: f64,
    pub imbalance_ratio: f64,
    pub per_device_duration_us: Vec<(String, f64)>,
}

/// Trace analyzer for post-hoc inspection.
pub struct TraceAnalyzer;

impl TraceAnalyzer {
    /// Compute per-operation-type summary.
    pub fn operation_summary(trace: &RecordedExecutionTrace) -> HashMap<String, OpSummary> {
        let mut map: HashMap<String, OpSummary> = HashMap::new();
        for entry in &trace.entries {
            let s = map.entry(entry.operation.clone()).or_default();
            s.count += 1;
            s.total_duration_us += entry.duration_us;
            s.total_memory_bytes += entry.memory_bytes;
        }
        map
    }

    /// Find memory hotspots (ops using more than `threshold_bytes`).
    pub fn memory_hotspots(
        trace: &RecordedExecutionTrace,
        threshold_bytes: usize,
    ) -> Vec<&RecordedTraceEntry> {
        trace
            .entries
            .iter()
            .filter(|e| e.memory_bytes > threshold_bytes)
            .collect()
    }

    /// Compute average duration per operation type.
    pub fn avg_duration_by_op(trace: &RecordedExecutionTrace) -> HashMap<String, f64> {
        let summary = Self::operation_summary(trace);
        summary
            .into_iter()
            .map(|(k, v)| {
                let avg = if v.count > 0 {
                    v.total_duration_us / v.count as f64
                } else {
                    0.0
                };
                (k, avg)
            })
            .collect()
    }

    /// Compute per-device profile summary for traces with `device_id` set.
    pub fn per_device_summary(trace: &RecordedExecutionTrace) -> HashMap<String, DeviceSummary> {
        let mut map: HashMap<String, DeviceSummary> = HashMap::new();
        for entry in &trace.entries {
            if let Some(device) = &entry.device_id {
                let summary = map.entry(device.clone()).or_default();
                summary.op_count += 1;
                summary.total_duration_us += entry.duration_us;
                summary.total_memory_bytes += entry.memory_bytes;
            }
        }
        map
    }

    /// Compute load balancing metrics from per-device timings.
    pub fn load_balance_metrics(trace: &RecordedExecutionTrace) -> Option<LoadBalanceMetrics> {
        let summary = Self::per_device_summary(trace);
        if summary.len() < 2 {
            return None;
        }

        let mut per_device_duration_us: Vec<(String, f64)> = summary
            .iter()
            .map(|(device, s)| (device.clone(), s.total_duration_us))
            .collect();
        per_device_duration_us.sort_by(|a, b| a.0.cmp(&b.0));

        let total_duration_us: f64 = per_device_duration_us.iter().map(|(_, t)| *t).sum();
        let device_count = per_device_duration_us.len();
        let ideal_duration_us = total_duration_us / device_count as f64;
        let max_duration_us = per_device_duration_us
            .iter()
            .map(|(_, t)| *t)
            .fold(0.0_f64, f64::max);
        let imbalance_ratio = if ideal_duration_us > 0.0 {
            ((max_duration_us - ideal_duration_us) / ideal_duration_us).max(0.0)
        } else {
            0.0
        };

        Some(LoadBalanceMetrics {
            device_count,
            total_duration_us,
            ideal_duration_us,
            max_duration_us,
            imbalance_ratio,
            per_device_duration_us,
        })
    }

    /// Detect communication bottlenecks in distributed traces.
    ///
    /// A communication op is detected by operation names containing one of:
    /// `allreduce`, `all_gather`, `reduce_scatter`, `broadcast`, `send`, `recv`, `comm`.
    /// Returns ops whose cumulative time exceeds `min_ratio_of_total`.
    pub fn communication_bottlenecks(
        trace: &RecordedExecutionTrace,
        min_ratio_of_total: f64,
    ) -> Vec<CommunicationBottleneck> {
        let total_duration_us = trace.total_duration_us.max(1e-9);
        let mut aggregate: HashMap<String, (f64, usize)> = HashMap::new();

        for entry in &trace.entries {
            let op = entry.operation.to_ascii_lowercase();
            let is_comm = op.contains("allreduce")
                || op.contains("all_gather")
                || op.contains("reduce_scatter")
                || op.contains("broadcast")
                || op.contains("send")
                || op.contains("recv")
                || op.contains("comm");

            if is_comm {
                let agg = aggregate.entry(entry.operation.clone()).or_insert((0.0, 0));
                agg.0 += entry.duration_us;
                agg.1 += 1;
            }
        }

        let mut results: Vec<CommunicationBottleneck> = aggregate
            .into_iter()
            .map(
                |(operation, (duration, call_count))| CommunicationBottleneck {
                    operation,
                    total_duration_us: duration,
                    ratio_of_total: duration / total_duration_us,
                    call_count,
                },
            )
            .filter(|b| b.ratio_of_total >= min_ratio_of_total)
            .collect();

        results.sort_by(|a, b| {
            b.total_duration_us
                .partial_cmp(&a.total_duration_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Export a collapsed stack format string compatible with FlameGraph tools.
    pub fn flamegraph_collapsed(trace: &RecordedExecutionTrace) -> String {
        let mut aggregate: HashMap<String, u64> = HashMap::new();
        for entry in &trace.entries {
            let device = entry.device_id.as_deref().unwrap_or("unknown");
            let stack = format!("trace;{};{}", device, entry.operation);
            let weight = entry.duration_us.max(1.0).round() as u64;
            *aggregate.entry(stack).or_insert(0) += weight;
        }

        let mut lines: Vec<(String, u64)> = aggregate.into_iter().collect();
        lines.sort_by(|a, b| a.0.cmp(&b.0));

        lines
            .into_iter()
            .map(|(stack, weight)| format!("{} {}", stack, weight))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A recording session that tracks execution in real-time.
pub struct TraceRecorder {
    trace: RecordedExecutionTrace,
    current_step: usize,
    phase_start: Option<Instant>,
    current_op: Option<String>,
}

impl TraceRecorder {
    /// Create a new recorder.
    pub fn new() -> Self {
        TraceRecorder {
            trace: RecordedExecutionTrace::new(),
            current_step: 0,
            phase_start: None,
            current_op: None,
        }
    }

    /// Begin recording an operation. Ends the previous one if still active.
    pub fn begin_op(&mut self, op: impl Into<String>) {
        self.end_op(); // end previous if any
        self.current_op = Some(op.into());
        self.phase_start = Some(Instant::now());
    }

    /// End the current operation, recording input/output shapes.
    pub fn end_op_with_shapes(&mut self, input_shapes: Vec<Vec<usize>>, output_shape: Vec<usize>) {
        if let (Some(op), Some(start)) = (self.current_op.take(), self.phase_start.take()) {
            let entry = RecordedTraceEntry::new(self.current_step, op)
                .with_input_shapes(input_shapes)
                .with_output_shape(output_shape)
                .with_duration(start.elapsed());
            self.trace.add_entry(entry);
            self.current_step += 1;
        }
    }

    /// End the current operation without shape information.
    pub fn end_op(&mut self) {
        if self.current_op.is_some() {
            self.end_op_with_shapes(vec![], vec![]);
        }
    }

    /// Finish recording and return the completed trace.
    pub fn finish(mut self) -> RecordedExecutionTrace {
        self.end_op();
        self.trace
    }

    /// Peek at the trace built so far.
    pub fn current_trace(&self) -> &RecordedExecutionTrace {
        &self.trace
    }
}

impl Default for TraceRecorder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_entry_new() {
        let entry = RecordedTraceEntry::new(0, "matmul");
        assert_eq!(entry.step, 0);
        assert_eq!(entry.operation, "matmul");
        assert!(entry.device_id.is_none());
        assert!(entry.input_shapes.is_empty());
        assert!(entry.output_shape.is_empty());
        assert!((entry.duration_us - 0.0).abs() < f64::EPSILON);
        assert_eq!(entry.output_elements, 0);
        assert_eq!(entry.memory_bytes, 0);
    }

    #[test]
    fn test_trace_entry_builder() {
        let entry = RecordedTraceEntry::new(1, "conv2d")
            .with_device_id("gpu:0")
            .with_input_shapes(vec![vec![1, 3, 32, 32], vec![16, 3, 3, 3]])
            .with_output_shape(vec![1, 16, 30, 30])
            .with_duration(Duration::from_micros(500));
        assert_eq!(entry.device_id.as_deref(), Some("gpu:0"));
        assert_eq!(entry.input_shapes.len(), 2);
        assert_eq!(entry.output_shape, vec![1, 16, 30, 30]);
        assert_eq!(entry.output_elements, 16 * 30 * 30);
        assert_eq!(entry.memory_bytes, 16 * 30 * 30 * 8);
        assert!((entry.duration_us - 500.0).abs() < 1.0);
    }

    #[test]
    fn test_trace_entry_throughput() {
        let entry = RecordedTraceEntry::new(0, "add")
            .with_output_shape(vec![1000])
            .with_duration(Duration::from_micros(100));
        let tp = entry.throughput_elements_per_us();
        assert!((tp - 10.0).abs() < 0.1);

        // Zero duration yields zero throughput.
        let zero = RecordedTraceEntry::new(0, "noop");
        assert!((zero.throughput_elements_per_us() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trace_new_empty() {
        let trace = RecordedExecutionTrace::new();
        assert!(trace.entries.is_empty());
        assert!((trace.total_duration_us - 0.0).abs() < f64::EPSILON);
        assert!(trace.metadata.is_empty());
    }

    #[test]
    fn test_trace_add_entry() {
        let mut trace = RecordedExecutionTrace::new();
        let e1 = RecordedTraceEntry::new(0, "op_a").with_duration(Duration::from_micros(100));
        let e2 = RecordedTraceEntry::new(1, "op_b").with_duration(Duration::from_micros(200));
        trace.add_entry(e1);
        trace.add_entry(e2);
        assert_eq!(trace.entries.len(), 2);
        assert!((trace.total_duration_us - 300.0).abs() < 1.0);
    }

    #[test]
    fn test_trace_step_count() {
        let mut trace = RecordedExecutionTrace::new();
        assert_eq!(trace.step_count(), 0);
        trace.add_entry(RecordedTraceEntry::new(0, "a"));
        trace.add_entry(RecordedTraceEntry::new(1, "b"));
        trace.add_entry(RecordedTraceEntry::new(2, "c"));
        assert_eq!(trace.step_count(), 3);
    }

    #[test]
    fn test_trace_total_memory() {
        let mut trace = RecordedExecutionTrace::new();
        trace.add_entry(RecordedTraceEntry::new(0, "a").with_output_shape(vec![10]));
        trace.add_entry(RecordedTraceEntry::new(1, "b").with_output_shape(vec![20]));
        // 10*8 + 20*8 = 240
        assert_eq!(trace.total_memory_bytes(), 240);
    }

    #[test]
    fn test_trace_peak_memory() {
        let mut trace = RecordedExecutionTrace::new();
        trace.add_entry(RecordedTraceEntry::new(0, "a").with_output_shape(vec![10]));
        trace.add_entry(RecordedTraceEntry::new(1, "b").with_output_shape(vec![100]));
        trace.add_entry(RecordedTraceEntry::new(2, "c").with_output_shape(vec![50]));
        assert_eq!(trace.peak_memory_bytes(), 100 * 8);

        // Empty trace yields 0.
        let empty = RecordedExecutionTrace::new();
        assert_eq!(empty.peak_memory_bytes(), 0);
    }

    #[test]
    fn test_trace_slowest_ops() {
        let mut trace = RecordedExecutionTrace::new();
        trace
            .add_entry(RecordedTraceEntry::new(0, "fast").with_duration(Duration::from_micros(10)));
        trace.add_entry(
            RecordedTraceEntry::new(1, "slow").with_duration(Duration::from_micros(500)),
        );
        trace.add_entry(
            RecordedTraceEntry::new(2, "medium").with_duration(Duration::from_micros(100)),
        );
        let slowest = trace.slowest_ops(2);
        assert_eq!(slowest.len(), 2);
        assert_eq!(slowest[0].operation, "slow");
        assert_eq!(slowest[1].operation, "medium");
    }

    #[test]
    fn test_trace_to_json() {
        let mut trace = RecordedExecutionTrace::new();
        trace.add_entry(RecordedTraceEntry::new(0, "matmul").with_output_shape(vec![4, 4]));
        let json = trace.to_json().expect("serialization should succeed");
        assert!(json.contains("matmul"));
        assert!(json.contains("output_shape"));
        // Verify it round-trips.
        let parsed: RecordedExecutionTrace =
            serde_json::from_str(&json).expect("deserialization should succeed");
        assert_eq!(parsed.entries.len(), 1);
    }

    #[test]
    fn test_trace_metadata() {
        let trace = RecordedExecutionTrace::new()
            .with_metadata("model", "resnet50")
            .with_metadata("device", "cpu");
        assert_eq!(
            trace.metadata.get("model").map(|s| s.as_str()),
            Some("resnet50")
        );
        assert_eq!(
            trace.metadata.get("device").map(|s| s.as_str()),
            Some("cpu")
        );
    }

    #[test]
    fn test_analyzer_operation_summary() {
        let mut trace = RecordedExecutionTrace::new();
        trace.add_entry(
            RecordedTraceEntry::new(0, "matmul")
                .with_duration(Duration::from_micros(100))
                .with_output_shape(vec![10]),
        );
        trace.add_entry(
            RecordedTraceEntry::new(1, "matmul")
                .with_duration(Duration::from_micros(200))
                .with_output_shape(vec![20]),
        );
        trace.add_entry(
            RecordedTraceEntry::new(2, "relu")
                .with_duration(Duration::from_micros(50))
                .with_output_shape(vec![10]),
        );
        let summary = TraceAnalyzer::operation_summary(&trace);
        let mm = summary.get("matmul").expect("matmul should exist");
        assert_eq!(mm.count, 2);
        assert!((mm.total_duration_us - 300.0).abs() < 1.0);
        assert_eq!(mm.total_memory_bytes, (10 + 20) * 8);
        let relu = summary.get("relu").expect("relu should exist");
        assert_eq!(relu.count, 1);
    }

    #[test]
    fn test_analyzer_memory_hotspots() {
        let mut trace = RecordedExecutionTrace::new();
        trace.add_entry(RecordedTraceEntry::new(0, "small").with_output_shape(vec![10]));
        trace.add_entry(RecordedTraceEntry::new(1, "big").with_output_shape(vec![1000]));
        trace.add_entry(RecordedTraceEntry::new(2, "medium").with_output_shape(vec![100]));
        // threshold = 500 bytes => only "big" (1000*8=8000) and "medium" (100*8=800) qualify
        let hotspots = TraceAnalyzer::memory_hotspots(&trace, 500);
        assert_eq!(hotspots.len(), 2);
        // Only "big" with threshold 1000
        let hotspots2 = TraceAnalyzer::memory_hotspots(&trace, 1000);
        assert_eq!(hotspots2.len(), 1);
        assert_eq!(hotspots2[0].operation, "big");
    }

    #[test]
    fn test_analyzer_avg_duration() {
        let mut trace = RecordedExecutionTrace::new();
        trace
            .add_entry(RecordedTraceEntry::new(0, "add").with_duration(Duration::from_micros(100)));
        trace
            .add_entry(RecordedTraceEntry::new(1, "add").with_duration(Duration::from_micros(300)));
        trace
            .add_entry(RecordedTraceEntry::new(2, "mul").with_duration(Duration::from_micros(200)));
        let avgs = TraceAnalyzer::avg_duration_by_op(&trace);
        let add_avg = avgs.get("add").copied().unwrap_or(0.0);
        assert!((add_avg - 200.0).abs() < 1.0);
        let mul_avg = avgs.get("mul").copied().unwrap_or(0.0);
        assert!((mul_avg - 200.0).abs() < 1.0);
    }

    #[test]
    fn test_analyzer_per_device_summary() {
        let mut trace = RecordedExecutionTrace::new();
        trace.add_entry(
            RecordedTraceEntry::new(0, "matmul")
                .with_device_id("gpu:0")
                .with_duration(Duration::from_micros(100))
                .with_output_shape(vec![32, 32]),
        );
        trace.add_entry(
            RecordedTraceEntry::new(1, "relu")
                .with_device_id("gpu:0")
                .with_duration(Duration::from_micros(50))
                .with_output_shape(vec![32, 32]),
        );
        trace.add_entry(
            RecordedTraceEntry::new(2, "allreduce")
                .with_device_id("gpu:1")
                .with_duration(Duration::from_micros(250))
                .with_output_shape(vec![32, 32]),
        );

        let summary = TraceAnalyzer::per_device_summary(&trace);
        assert_eq!(summary.len(), 2);
        let gpu0 = summary.get("gpu:0").expect("gpu:0 summary must exist");
        assert_eq!(gpu0.op_count, 2);
        assert!((gpu0.total_duration_us - 150.0).abs() < 1.0);
        let gpu1 = summary.get("gpu:1").expect("gpu:1 summary must exist");
        assert_eq!(gpu1.op_count, 1);
        assert!((gpu1.total_duration_us - 250.0).abs() < 1.0);
    }

    #[test]
    fn test_analyzer_load_balance_metrics() {
        let mut trace = RecordedExecutionTrace::new();
        trace.add_entry(
            RecordedTraceEntry::new(0, "matmul")
                .with_device_id("gpu:0")
                .with_duration(Duration::from_micros(300)),
        );
        trace.add_entry(
            RecordedTraceEntry::new(1, "matmul")
                .with_device_id("gpu:1")
                .with_duration(Duration::from_micros(100)),
        );

        let metrics = TraceAnalyzer::load_balance_metrics(&trace)
            .expect("load balance metrics should be available for >=2 devices");
        assert_eq!(metrics.device_count, 2);
        assert!((metrics.total_duration_us - 400.0).abs() < 1.0);
        assert!((metrics.ideal_duration_us - 200.0).abs() < 1.0);
        assert!((metrics.max_duration_us - 300.0).abs() < 1.0);
        assert!(metrics.imbalance_ratio > 0.45 && metrics.imbalance_ratio < 0.55);
    }

    #[test]
    fn test_analyzer_communication_bottlenecks() {
        let mut trace = RecordedExecutionTrace::new();
        trace.add_entry(
            RecordedTraceEntry::new(0, "allreduce")
                .with_device_id("gpu:0")
                .with_duration(Duration::from_micros(600)),
        );
        trace.add_entry(
            RecordedTraceEntry::new(1, "matmul")
                .with_device_id("gpu:0")
                .with_duration(Duration::from_micros(200)),
        );
        trace.add_entry(
            RecordedTraceEntry::new(2, "broadcast")
                .with_device_id("gpu:1")
                .with_duration(Duration::from_micros(300)),
        );

        let bottlenecks = TraceAnalyzer::communication_bottlenecks(&trace, 0.2);
        assert_eq!(bottlenecks.len(), 2);
        assert_eq!(bottlenecks[0].operation, "allreduce");
        assert!(bottlenecks[0].ratio_of_total > 0.5);
    }

    #[test]
    fn test_analyzer_flamegraph_collapsed() {
        let mut trace = RecordedExecutionTrace::new();
        trace.add_entry(
            RecordedTraceEntry::new(0, "matmul")
                .with_device_id("gpu:0")
                .with_duration(Duration::from_micros(123)),
        );
        trace.add_entry(
            RecordedTraceEntry::new(1, "matmul")
                .with_device_id("gpu:0")
                .with_duration(Duration::from_micros(77)),
        );
        trace.add_entry(
            RecordedTraceEntry::new(2, "relu")
                .with_device_id("gpu:1")
                .with_duration(Duration::from_micros(50)),
        );

        let collapsed = TraceAnalyzer::flamegraph_collapsed(&trace);
        assert!(collapsed.contains("trace;gpu:0;matmul 200"));
        assert!(collapsed.contains("trace;gpu:1;relu 50"));
    }

    #[test]
    fn test_recorder_begin_end() {
        let mut recorder = TraceRecorder::new();
        recorder.begin_op("matmul");
        recorder.end_op_with_shapes(vec![vec![2, 3], vec![3, 4]], vec![2, 4]);
        let trace = recorder.finish();
        assert_eq!(trace.step_count(), 1);
        assert_eq!(trace.entries[0].operation, "matmul");
        assert_eq!(trace.entries[0].output_shape, vec![2, 4]);
        assert_eq!(trace.entries[0].output_elements, 8);
    }

    #[test]
    fn test_recorder_multiple_ops() {
        let mut recorder = TraceRecorder::new();
        recorder.begin_op("conv");
        recorder.end_op_with_shapes(vec![vec![1, 3, 8, 8]], vec![1, 16, 6, 6]);
        recorder.begin_op("relu");
        recorder.end_op_with_shapes(vec![vec![1, 16, 6, 6]], vec![1, 16, 6, 6]);
        recorder.begin_op("pool");
        recorder.end_op_with_shapes(vec![vec![1, 16, 6, 6]], vec![1, 16, 3, 3]);
        let trace = recorder.finish();
        assert_eq!(trace.step_count(), 3);
        assert_eq!(trace.entries[0].step, 0);
        assert_eq!(trace.entries[1].step, 1);
        assert_eq!(trace.entries[2].step, 2);
    }

    #[test]
    fn test_recorder_finish() {
        let mut recorder = TraceRecorder::new();
        recorder.begin_op("op_a");
        // Do NOT explicitly end — finish() should close it.
        let trace = recorder.finish();
        assert_eq!(trace.step_count(), 1);
        assert_eq!(trace.entries[0].operation, "op_a");
        assert!(trace.total_duration_us >= 0.0);
    }

    #[test]
    fn test_op_summary_default() {
        let summary = OpSummary::default();
        assert_eq!(summary.count, 0);
        assert!((summary.total_duration_us - 0.0).abs() < f64::EPSILON);
        assert_eq!(summary.total_memory_bytes, 0);
    }
}
