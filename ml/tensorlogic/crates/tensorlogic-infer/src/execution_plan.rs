//! Execution plan formatting and resource timeline visualization.
//!
//! Provides human-readable views of execution schedules, parallel opportunity
//! analysis, and memory usage timelines.

use std::collections::HashMap;
use std::fmt::Write;

/// A single step in an execution plan.
#[derive(Debug, Clone)]
pub struct PlanStep {
    /// Step index (execution order)
    pub index: usize,
    /// Operation name/description
    pub operation: String,
    /// Input tensor names
    pub inputs: Vec<String>,
    /// Output tensor name
    pub output: String,
    /// Estimated memory usage in bytes
    pub estimated_memory_bytes: usize,
    /// Estimated FLOPs
    pub estimated_flops: u64,
    /// Whether this step can run in parallel with the previous step
    pub parallelizable: bool,
    /// Level in the dependency graph (0 = no dependencies)
    pub dependency_level: usize,
}

impl PlanStep {
    /// Create a new plan step with default values for optional fields.
    pub fn new(index: usize, operation: impl Into<String>, output: impl Into<String>) -> Self {
        PlanStep {
            index,
            operation: operation.into(),
            inputs: Vec::new(),
            output: output.into(),
            estimated_memory_bytes: 0,
            estimated_flops: 0,
            parallelizable: false,
            dependency_level: 0,
        }
    }

    /// Set the input tensor names.
    pub fn with_inputs(mut self, inputs: Vec<String>) -> Self {
        self.inputs = inputs;
        self
    }

    /// Set the estimated memory usage in bytes.
    pub fn with_memory(mut self, bytes: usize) -> Self {
        self.estimated_memory_bytes = bytes;
        self
    }

    /// Set the estimated FLOPs.
    pub fn with_flops(mut self, flops: u64) -> Self {
        self.estimated_flops = flops;
        self
    }

    /// Set whether this step can run in parallel.
    pub fn with_parallel(mut self, p: bool) -> Self {
        self.parallelizable = p;
        self
    }

    /// Set the dependency level.
    pub fn with_level(mut self, l: usize) -> Self {
        self.dependency_level = l;
        self
    }
}

/// A complete execution plan containing ordered steps with dependency and
/// resource metadata.
#[derive(Debug, Clone, Default)]
pub struct ExecutionPlan {
    /// The ordered steps of this execution plan.
    pub steps: Vec<PlanStep>,
}

impl ExecutionPlan {
    /// Create a new empty execution plan.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a step to the execution plan.
    pub fn add_step(&mut self, step: PlanStep) {
        self.steps.push(step);
    }

    /// Total estimated FLOPs across all steps.
    pub fn total_flops(&self) -> u64 {
        self.steps.iter().map(|s| s.estimated_flops).sum()
    }

    /// Peak estimated memory (sum of all live tensors at the busiest level).
    ///
    /// Groups steps by dependency level, sums memory per level, and returns
    /// the maximum.
    pub fn peak_memory(&self) -> usize {
        let mut level_mem: HashMap<usize, usize> = HashMap::new();
        for step in &self.steps {
            *level_mem.entry(step.dependency_level).or_insert(0) += step.estimated_memory_bytes;
        }
        level_mem.values().copied().max().unwrap_or(0)
    }

    /// Number of steps that can be parallelized.
    pub fn parallel_count(&self) -> usize {
        self.steps.iter().filter(|s| s.parallelizable).count()
    }

    /// Maximum dependency depth (critical path length).
    ///
    /// Returns the number of distinct dependency levels, i.e. max level + 1.
    pub fn critical_path_length(&self) -> usize {
        self.steps
            .iter()
            .map(|s| s.dependency_level)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0)
    }

    /// Theoretical speedup from parallelism (total_steps / critical_path_length).
    pub fn parallel_speedup(&self) -> f64 {
        let cpl = self.critical_path_length();
        if cpl == 0 {
            return 1.0;
        }
        self.steps.len() as f64 / cpl as f64
    }
}

/// Formatter for rendering execution plans as human-readable strings.
pub struct PlanFormatter;

impl PlanFormatter {
    /// Format the plan as a table with columns for step index, operation,
    /// output, dependency level, memory, and parallelizability.
    pub fn format_table(plan: &ExecutionPlan) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "{:-<80}", "");
        let _ = writeln!(
            out,
            "{:<5} {:<20} {:<20} {:<8} {:<10} {:<5}",
            "Step", "Operation", "Output", "Level", "Memory", "Par?"
        );
        let _ = writeln!(out, "{:-<80}", "");
        for step in &plan.steps {
            let mem_str = format_bytes(step.estimated_memory_bytes);
            let par = if step.parallelizable { "yes" } else { "no" };
            let _ = writeln!(
                out,
                "{:<5} {:<20} {:<20} {:<8} {:<10} {:<5}",
                step.index,
                truncate(&step.operation, 19),
                truncate(&step.output, 19),
                step.dependency_level,
                mem_str,
                par
            );
        }
        let _ = writeln!(out, "{:-<80}", "");
        let _ = writeln!(
            out,
            "Total steps: {} | Critical path: {} | Parallel speedup: {:.1}x",
            plan.steps.len(),
            plan.critical_path_length(),
            plan.parallel_speedup()
        );
        let _ = writeln!(
            out,
            "Total FLOPs: {} | Peak memory: {}",
            plan.total_flops(),
            format_bytes(plan.peak_memory())
        );
        out
    }

    /// Format the plan as a level-grouped tree showing parallelism
    /// opportunities.
    pub fn format_tree(plan: &ExecutionPlan) -> String {
        let mut out = String::new();
        let max_level = plan
            .steps
            .iter()
            .map(|s| s.dependency_level)
            .max()
            .unwrap_or(0);
        for level in 0..=max_level {
            let steps_at_level: Vec<_> = plan
                .steps
                .iter()
                .filter(|s| s.dependency_level == level)
                .collect();
            let _ = writeln!(
                out,
                "Level {} ({} ops{}):",
                level,
                steps_at_level.len(),
                if steps_at_level.len() > 1 {
                    " \u{2014} parallelizable"
                } else {
                    ""
                }
            );
            for step in steps_at_level {
                let _ = writeln!(
                    out,
                    "  [{}] {} \u{2192} {}",
                    step.index, step.operation, step.output
                );
            }
        }
        out
    }
}

/// A single entry in a memory timeline, tracking allocations over time.
#[derive(Debug, Clone)]
pub struct MemoryTimelineEntry {
    /// The step index this entry corresponds to.
    pub step: usize,
    /// Bytes allocated at this step.
    pub allocated_bytes: usize,
    /// Bytes freed at this step.
    pub freed_bytes: usize,
    /// Total live bytes after this step.
    pub live_bytes: usize,
}

/// Compute a memory timeline from an execution plan.
///
/// Produces one [`MemoryTimelineEntry`] per step, tracking cumulative
/// allocations. In this simplified model no memory is freed between steps.
pub fn compute_memory_timeline(plan: &ExecutionPlan) -> Vec<MemoryTimelineEntry> {
    let mut live = 0usize;
    plan.steps
        .iter()
        .map(|step| {
            live = live.saturating_add(step.estimated_memory_bytes);
            MemoryTimelineEntry {
                step: step.index,
                allocated_bytes: step.estimated_memory_bytes,
                freed_bytes: 0,
                live_bytes: live,
            }
        })
        .collect()
}

/// Format a byte count as a human-readable string (B / KB / MB).
fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{}B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Truncate a string to at most `max` characters, appending an ellipsis if
/// truncated.
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let boundary = max.saturating_sub(1);
        // Find a valid char boundary at or before the target position
        let end = s
            .char_indices()
            .take_while(|&(i, _)| i < boundary)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        format!("{}\u{2026}", &s[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_plan() -> ExecutionPlan {
        let mut plan = ExecutionPlan::new();
        plan.add_step(
            PlanStep::new(0, "matmul", "t0")
                .with_inputs(vec!["a".into(), "b".into()])
                .with_memory(1024)
                .with_flops(2000)
                .with_level(0),
        );
        plan.add_step(
            PlanStep::new(1, "relu", "t1")
                .with_inputs(vec!["t0".into()])
                .with_memory(512)
                .with_flops(500)
                .with_parallel(true)
                .with_level(0),
        );
        plan.add_step(
            PlanStep::new(2, "add", "t2")
                .with_inputs(vec!["t0".into(), "t1".into()])
                .with_memory(2048)
                .with_flops(1000)
                .with_level(1),
        );
        plan
    }

    #[test]
    fn test_plan_step_new() {
        let step = PlanStep::new(0, "matmul", "out");
        assert_eq!(step.index, 0);
        assert_eq!(step.operation, "matmul");
        assert_eq!(step.output, "out");
        assert!(step.inputs.is_empty());
        assert_eq!(step.estimated_memory_bytes, 0);
        assert_eq!(step.estimated_flops, 0);
        assert!(!step.parallelizable);
        assert_eq!(step.dependency_level, 0);
    }

    #[test]
    fn test_plan_step_builder() {
        let step = PlanStep::new(1, "conv2d", "feat")
            .with_inputs(vec!["img".into()])
            .with_memory(4096)
            .with_flops(8000)
            .with_parallel(true)
            .with_level(2);
        assert_eq!(step.index, 1);
        assert_eq!(step.inputs, vec!["img".to_string()]);
        assert_eq!(step.estimated_memory_bytes, 4096);
        assert_eq!(step.estimated_flops, 8000);
        assert!(step.parallelizable);
        assert_eq!(step.dependency_level, 2);
    }

    #[test]
    fn test_plan_new_empty() {
        let plan = ExecutionPlan::new();
        assert!(plan.steps.is_empty());
        assert_eq!(plan.total_flops(), 0);
        assert_eq!(plan.peak_memory(), 0);
        assert_eq!(plan.critical_path_length(), 0);
    }

    #[test]
    fn test_plan_add_step() {
        let mut plan = ExecutionPlan::new();
        assert_eq!(plan.steps.len(), 0);
        plan.add_step(PlanStep::new(0, "op", "out"));
        assert_eq!(plan.steps.len(), 1);
        plan.add_step(PlanStep::new(1, "op2", "out2"));
        assert_eq!(plan.steps.len(), 2);
    }

    #[test]
    fn test_plan_total_flops() {
        let plan = sample_plan();
        // 2000 + 500 + 1000 = 3500
        assert_eq!(plan.total_flops(), 3500);
    }

    #[test]
    fn test_plan_peak_memory() {
        let plan = sample_plan();
        // Level 0: 1024 + 512 = 1536, Level 1: 2048 => peak = 2048
        assert_eq!(plan.peak_memory(), 2048);
    }

    #[test]
    fn test_plan_parallel_count() {
        let plan = sample_plan();
        // Only step 1 is parallelizable
        assert_eq!(plan.parallel_count(), 1);
    }

    #[test]
    fn test_plan_critical_path() {
        let plan = sample_plan();
        // Max level is 1, so critical path = 2
        assert_eq!(plan.critical_path_length(), 2);
    }

    #[test]
    fn test_plan_parallel_speedup() {
        let plan = sample_plan();
        // 3 steps / 2 levels = 1.5
        let speedup = plan.parallel_speedup();
        assert!((speedup - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_format_table_header() {
        let plan = sample_plan();
        let table = PlanFormatter::format_table(&plan);
        assert!(table.contains("Step"));
        assert!(table.contains("Operation"));
        assert!(table.contains("Output"));
        assert!(table.contains("Level"));
        assert!(table.contains("Memory"));
        assert!(table.contains("Par?"));
    }

    #[test]
    fn test_format_table_entries() {
        let plan = sample_plan();
        let table = PlanFormatter::format_table(&plan);
        // Step indices should appear
        assert!(table.contains("0"));
        assert!(table.contains("1"));
        assert!(table.contains("2"));
        // Operation names
        assert!(table.contains("matmul"));
        assert!(table.contains("relu"));
        assert!(table.contains("add"));
    }

    #[test]
    fn test_format_table_summary() {
        let plan = sample_plan();
        let table = PlanFormatter::format_table(&plan);
        assert!(table.contains("Total steps: 3"));
        assert!(table.contains("Critical path: 2"));
        assert!(table.contains("Parallel speedup: 1.5x"));
        assert!(table.contains("Total FLOPs: 3500"));
    }

    #[test]
    fn test_format_tree_levels() {
        let plan = sample_plan();
        let tree = PlanFormatter::format_tree(&plan);
        assert!(tree.contains("Level 0"));
        assert!(tree.contains("Level 1"));
        // Step 0 and 1 at level 0
        assert!(tree.contains("[0] matmul"));
        assert!(tree.contains("[1] relu"));
        // Step 2 at level 1
        assert!(tree.contains("[2] add"));
    }

    #[test]
    fn test_format_tree_parallel_note() {
        let plan = sample_plan();
        let tree = PlanFormatter::format_tree(&plan);
        // Level 0 has 2 ops, should show parallelizable note
        assert!(tree.contains("parallelizable"));
        // Level 1 has 1 op, should NOT show parallelizable for that line
        let lines: Vec<&str> = tree.lines().collect();
        let level1_line = lines
            .iter()
            .find(|l| l.starts_with("Level 1"))
            .expect("Level 1 line must exist");
        assert!(!level1_line.contains("parallelizable"));
    }

    #[test]
    fn test_memory_timeline_accumulates() {
        let plan = sample_plan();
        let timeline = compute_memory_timeline(&plan);
        // live_bytes should monotonically increase
        assert_eq!(timeline[0].live_bytes, 1024);
        assert_eq!(timeline[1].live_bytes, 1536);
        assert_eq!(timeline[2].live_bytes, 3584);
    }

    #[test]
    fn test_memory_timeline_length() {
        let plan = sample_plan();
        let timeline = compute_memory_timeline(&plan);
        assert_eq!(timeline.len(), plan.steps.len());
    }

    #[test]
    fn test_format_bytes_b() {
        assert_eq!(format_bytes(512), "512B");
        assert_eq!(format_bytes(0), "0B");
        assert_eq!(format_bytes(1023), "1023B");
    }

    #[test]
    fn test_format_bytes_kb() {
        assert_eq!(format_bytes(2048), "2.0KB");
        assert_eq!(format_bytes(1024), "1.0KB");
    }
}
