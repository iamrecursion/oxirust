#![allow(dead_code)]
//! Validate batch pipeline configurations before execution.
//!
//! Running a multi-step pipeline on thousands of files only to have it fail
//! on step 3 is expensive.  This module checks pipeline definitions for
//! common errors: missing inputs, circular dependencies, incompatible
//! codec/resolution combinations, and resource feasibility.

use std::collections::{HashMap, HashSet};

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Informational hint; the pipeline will still run.
    Info,
    /// Warning; the pipeline will run but may produce unexpected results.
    Warning,
    /// Error; the pipeline will not run.
    Error,
}

/// A single validation finding.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Severity level.
    pub severity: Severity,
    /// Machine-readable issue code.
    pub code: String,
    /// Human-readable description.
    pub message: String,
    /// The pipeline step index that triggered this issue (if applicable).
    pub step_index: Option<usize>,
}

impl ValidationIssue {
    /// Create a new validation issue.
    #[must_use]
    pub fn new(
        severity: Severity,
        code: impl Into<String>,
        message: impl Into<String>,
        step_index: Option<usize>,
    ) -> Self {
        Self {
            severity,
            code: code.into(),
            message: message.into(),
            step_index,
        }
    }

    /// Create an error-level issue.
    #[must_use]
    pub fn error(code: impl Into<String>, message: impl Into<String>, step: Option<usize>) -> Self {
        Self::new(Severity::Error, code, message, step)
    }

    /// Create a warning-level issue.
    #[must_use]
    pub fn warning(
        code: impl Into<String>,
        message: impl Into<String>,
        step: Option<usize>,
    ) -> Self {
        Self::new(Severity::Warning, code, message, step)
    }

    /// Create an info-level issue.
    #[must_use]
    pub fn info(code: impl Into<String>, message: impl Into<String>, step: Option<usize>) -> Self {
        Self::new(Severity::Info, code, message, step)
    }
}

/// Result of a full validation run.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// All issues found.
    pub issues: Vec<ValidationIssue>,
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationReport {
    /// Create an empty report.
    #[must_use]
    pub fn new() -> Self {
        Self { issues: Vec::new() }
    }

    /// Add an issue to the report.
    pub fn add(&mut self, issue: ValidationIssue) {
        self.issues.push(issue);
    }

    /// Whether the pipeline is valid (no errors).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.issues.iter().any(|i| i.severity == Severity::Error)
    }

    /// Count issues at a given severity.
    #[must_use]
    pub fn count(&self, severity: Severity) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == severity)
            .count()
    }

    /// Return only issues of a given severity.
    #[must_use]
    pub fn filter(&self, severity: Severity) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == severity)
            .collect()
    }

    /// Total number of issues.
    #[must_use]
    pub fn total(&self) -> usize {
        self.issues.len()
    }

    /// Whether the report has no issues at all.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }

    /// Merge another report into this one.
    pub fn merge(&mut self, other: &ValidationReport) {
        self.issues.extend(other.issues.iter().cloned());
    }
}

/// Describes a single step in a batch pipeline.
#[derive(Debug, Clone)]
pub struct PipelineStep {
    /// Step name.
    pub name: String,
    /// Step index.
    pub index: usize,
    /// Names of steps this step depends on.
    pub depends_on: Vec<String>,
    /// Input file extensions this step accepts.
    pub input_extensions: Vec<String>,
    /// Output file extension this step produces.
    pub output_extension: String,
    /// Whether this step requires a GPU.
    pub requires_gpu: bool,
    /// Estimated CPU cores needed.
    pub cpu_cores: f64,
    /// Estimated memory in MiB.
    pub memory_mib: u64,
}

impl PipelineStep {
    /// Create a new pipeline step.
    #[must_use]
    pub fn new(name: impl Into<String>, index: usize) -> Self {
        Self {
            name: name.into(),
            index,
            depends_on: Vec::new(),
            input_extensions: Vec::new(),
            output_extension: String::new(),
            requires_gpu: false,
            cpu_cores: 1.0,
            memory_mib: 256,
        }
    }

    /// Set dependencies.
    #[must_use]
    pub fn with_depends_on(mut self, deps: Vec<String>) -> Self {
        self.depends_on = deps;
        self
    }

    /// Set accepted input extensions.
    #[must_use]
    pub fn with_input_extensions(mut self, exts: Vec<String>) -> Self {
        self.input_extensions = exts;
        self
    }

    /// Set the output extension.
    #[must_use]
    pub fn with_output_extension(mut self, ext: impl Into<String>) -> Self {
        self.output_extension = ext.into();
        self
    }
}

/// A batch pipeline definition to be validated.
#[derive(Debug, Clone)]
pub struct PipelineDefinition {
    /// Pipeline name.
    pub name: String,
    /// Ordered list of steps.
    pub steps: Vec<PipelineStep>,
    /// Maximum total CPU cores available.
    pub max_cpu_cores: f64,
    /// Maximum total memory available in MiB.
    pub max_memory_mib: u64,
    /// Whether a GPU is available.
    pub gpu_available: bool,
}

impl PipelineDefinition {
    /// Create a new pipeline definition.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            steps: Vec::new(),
            max_cpu_cores: 8.0,
            max_memory_mib: 16384,
            gpu_available: false,
        }
    }

    /// Add a step.
    pub fn add_step(&mut self, step: PipelineStep) {
        self.steps.push(step);
    }
}

/// The pipeline validator.
#[derive(Debug, Clone)]
pub struct PipelineValidator {
    /// Whether to check resource limits.
    check_resources: bool,
    /// Whether to check extension compatibility between steps.
    check_extensions: bool,
}

impl Default for PipelineValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineValidator {
    /// Create a new validator with all checks enabled.
    #[must_use]
    pub fn new() -> Self {
        Self {
            check_resources: true,
            check_extensions: true,
        }
    }

    /// Enable or disable resource checking.
    #[must_use]
    pub fn with_resource_check(mut self, enabled: bool) -> Self {
        self.check_resources = enabled;
        self
    }

    /// Enable or disable extension compatibility checking.
    #[must_use]
    pub fn with_extension_check(mut self, enabled: bool) -> Self {
        self.check_extensions = enabled;
        self
    }

    /// Validate a pipeline definition.
    #[must_use]
    pub fn validate(&self, pipeline: &PipelineDefinition) -> ValidationReport {
        let mut report = ValidationReport::new();

        // Check: pipeline must have at least one step
        if pipeline.steps.is_empty() {
            report.add(ValidationIssue::error(
                "EMPTY_PIPELINE",
                "Pipeline has no steps",
                None,
            ));
            return report;
        }

        // Check: unique step names
        self.check_unique_names(pipeline, &mut report);

        // Check: dependency references exist
        self.check_dependencies(pipeline, &mut report);

        // Check: circular dependencies
        self.check_cycles(pipeline, &mut report);

        // Check: extension compatibility
        if self.check_extensions {
            self.check_extension_compatibility(pipeline, &mut report);
        }

        // Check: resource limits
        if self.check_resources {
            self.check_resource_limits(pipeline, &mut report);
        }

        report
    }

    /// Verify all step names are unique.
    fn check_unique_names(&self, pipeline: &PipelineDefinition, report: &mut ValidationReport) {
        let mut seen = HashSet::new();
        for step in &pipeline.steps {
            if !seen.insert(&step.name) {
                report.add(ValidationIssue::error(
                    "DUPLICATE_STEP",
                    format!("Duplicate step name: {}", step.name),
                    Some(step.index),
                ));
            }
        }
    }

    /// Verify dependency references point to existing steps.
    fn check_dependencies(&self, pipeline: &PipelineDefinition, report: &mut ValidationReport) {
        let names: HashSet<&str> = pipeline.steps.iter().map(|s| s.name.as_str()).collect();
        for step in &pipeline.steps {
            for dep in &step.depends_on {
                if !names.contains(dep.as_str()) {
                    report.add(ValidationIssue::error(
                        "MISSING_DEPENDENCY",
                        format!("Step '{}' depends on non-existent step '{dep}'", step.name),
                        Some(step.index),
                    ));
                }
            }
        }
    }

    /// Detect circular dependencies via DFS.
    fn check_cycles(&self, pipeline: &PipelineDefinition, report: &mut ValidationReport) {
        let name_to_deps: HashMap<&str, &[String]> = pipeline
            .steps
            .iter()
            .map(|s| (s.name.as_str(), s.depends_on.as_slice()))
            .collect();

        let mut visited = HashSet::new();
        let mut stack = HashSet::new();

        for step in &pipeline.steps {
            if self.dfs_has_cycle(&step.name, &name_to_deps, &mut visited, &mut stack) {
                report.add(ValidationIssue::error(
                    "CIRCULAR_DEPENDENCY",
                    format!(
                        "Circular dependency detected involving step '{}'",
                        step.name
                    ),
                    Some(step.index),
                ));
                break;
            }
        }
    }

    /// DFS cycle detection helper.
    fn dfs_has_cycle<'a>(
        &self,
        node: &'a str,
        graph: &HashMap<&str, &'a [String]>,
        visited: &mut HashSet<&'a str>,
        stack: &mut HashSet<&'a str>,
    ) -> bool {
        if stack.contains(node) {
            return true;
        }
        if visited.contains(node) {
            return false;
        }
        visited.insert(node);
        stack.insert(node);

        if let Some(deps) = graph.get(node) {
            for dep in *deps {
                if self.dfs_has_cycle(dep.as_str(), graph, visited, stack) {
                    return true;
                }
            }
        }

        stack.remove(node);
        false
    }

    /// Check that output extension of a dependency matches the input
    /// extensions of the dependent step.
    fn check_extension_compatibility(
        &self,
        pipeline: &PipelineDefinition,
        report: &mut ValidationReport,
    ) {
        let name_to_step: HashMap<&str, &PipelineStep> = pipeline
            .steps
            .iter()
            .map(|s| (s.name.as_str(), s))
            .collect();

        for step in &pipeline.steps {
            if step.input_extensions.is_empty() {
                continue;
            }
            for dep_name in &step.depends_on {
                if let Some(dep_step) = name_to_step.get(dep_name.as_str()) {
                    if !dep_step.output_extension.is_empty()
                        && !step.input_extensions.contains(&dep_step.output_extension)
                    {
                        report.add(ValidationIssue::warning(
                            "EXTENSION_MISMATCH",
                            format!(
                                "Step '{}' outputs '.{}' but step '{}' accepts {:?}",
                                dep_step.name,
                                dep_step.output_extension,
                                step.name,
                                step.input_extensions,
                            ),
                            Some(step.index),
                        ));
                    }
                }
            }
        }
    }

    /// Check resource limits.
    #[allow(clippy::cast_precision_loss)]
    fn check_resource_limits(&self, pipeline: &PipelineDefinition, report: &mut ValidationReport) {
        for step in &pipeline.steps {
            if step.cpu_cores > pipeline.max_cpu_cores {
                report.add(ValidationIssue::error(
                    "CPU_EXCEEDED",
                    format!(
                        "Step '{}' needs {} CPU cores but only {} available",
                        step.name, step.cpu_cores, pipeline.max_cpu_cores,
                    ),
                    Some(step.index),
                ));
            }
            if step.memory_mib > pipeline.max_memory_mib {
                report.add(ValidationIssue::error(
                    "MEMORY_EXCEEDED",
                    format!(
                        "Step '{}' needs {} MiB memory but only {} MiB available",
                        step.name, step.memory_mib, pipeline.max_memory_mib,
                    ),
                    Some(step.index),
                ));
            }
            if step.requires_gpu && !pipeline.gpu_available {
                report.add(ValidationIssue::error(
                    "GPU_REQUIRED",
                    format!("Step '{}' requires a GPU but none is available", step.name),
                    Some(step.index),
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_pipeline() -> PipelineDefinition {
        let mut p = PipelineDefinition::new("test");
        p.add_step(PipelineStep::new("decode", 0).with_output_extension("yuv"));
        p.add_step(
            PipelineStep::new("encode", 1)
                .with_depends_on(vec!["decode".into()])
                .with_input_extensions(vec!["yuv".into()])
                .with_output_extension("mp4"),
        );
        p
    }

    #[test]
    fn test_valid_pipeline() {
        let validator = PipelineValidator::new();
        let report = validator.validate(&simple_pipeline());
        assert!(report.is_valid());
    }

    #[test]
    fn test_empty_pipeline() {
        let validator = PipelineValidator::new();
        let p = PipelineDefinition::new("empty");
        let report = validator.validate(&p);
        assert!(!report.is_valid());
        assert_eq!(report.count(Severity::Error), 1);
    }

    #[test]
    fn test_duplicate_step_names() {
        let validator = PipelineValidator::new();
        let mut p = PipelineDefinition::new("dup");
        p.add_step(PipelineStep::new("step1", 0));
        p.add_step(PipelineStep::new("step1", 1));
        let report = validator.validate(&p);
        assert!(!report.is_valid());
        assert!(report.issues.iter().any(|i| i.code == "DUPLICATE_STEP"));
    }

    #[test]
    fn test_missing_dependency() {
        let validator = PipelineValidator::new();
        let mut p = PipelineDefinition::new("missing_dep");
        p.add_step(PipelineStep::new("encode", 0).with_depends_on(vec!["nonexistent".into()]));
        let report = validator.validate(&p);
        assert!(!report.is_valid());
        assert!(report.issues.iter().any(|i| i.code == "MISSING_DEPENDENCY"));
    }

    #[test]
    fn test_circular_dependency() {
        let validator = PipelineValidator::new();
        let mut p = PipelineDefinition::new("cycle");
        p.add_step(PipelineStep::new("a", 0).with_depends_on(vec!["b".into()]));
        p.add_step(PipelineStep::new("b", 1).with_depends_on(vec!["a".into()]));
        let report = validator.validate(&p);
        assert!(!report.is_valid());
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == "CIRCULAR_DEPENDENCY"));
    }

    #[test]
    fn test_extension_mismatch() {
        let validator = PipelineValidator::new();
        let mut p = PipelineDefinition::new("ext_mismatch");
        p.add_step(PipelineStep::new("decode", 0).with_output_extension("yuv"));
        p.add_step(
            PipelineStep::new("encode", 1)
                .with_depends_on(vec!["decode".into()])
                .with_input_extensions(vec!["pcm".into()]), // mismatch
        );
        let report = validator.validate(&p);
        assert!(report.is_valid()); // warning, not error
        assert_eq!(report.count(Severity::Warning), 1);
    }

    #[test]
    fn test_cpu_exceeded() {
        let validator = PipelineValidator::new();
        let mut p = PipelineDefinition::new("cpu");
        p.max_cpu_cores = 4.0;
        let mut step = PipelineStep::new("heavy", 0);
        step.cpu_cores = 16.0;
        p.add_step(step);
        let report = validator.validate(&p);
        assert!(!report.is_valid());
        assert!(report.issues.iter().any(|i| i.code == "CPU_EXCEEDED"));
    }

    #[test]
    fn test_memory_exceeded() {
        let validator = PipelineValidator::new();
        let mut p = PipelineDefinition::new("mem");
        p.max_memory_mib = 1024;
        let mut step = PipelineStep::new("big", 0);
        step.memory_mib = 8192;
        p.add_step(step);
        let report = validator.validate(&p);
        assert!(!report.is_valid());
        assert!(report.issues.iter().any(|i| i.code == "MEMORY_EXCEEDED"));
    }

    #[test]
    fn test_gpu_required_but_unavailable() {
        let validator = PipelineValidator::new();
        let mut p = PipelineDefinition::new("gpu");
        p.gpu_available = false;
        let mut step = PipelineStep::new("gpu_step", 0);
        step.requires_gpu = true;
        p.add_step(step);
        let report = validator.validate(&p);
        assert!(!report.is_valid());
        assert!(report.issues.iter().any(|i| i.code == "GPU_REQUIRED"));
    }

    #[test]
    fn test_validation_report_merge() {
        let mut r1 = ValidationReport::new();
        r1.add(ValidationIssue::info("I1", "info", None));
        let mut r2 = ValidationReport::new();
        r2.add(ValidationIssue::error("E1", "error", None));
        r1.merge(&r2);
        assert_eq!(r1.total(), 2);
        assert!(!r1.is_valid());
    }

    #[test]
    fn test_validation_report_filter() {
        let mut report = ValidationReport::new();
        report.add(ValidationIssue::info("I1", "info", None));
        report.add(ValidationIssue::warning("W1", "warn", None));
        report.add(ValidationIssue::error("E1", "error", None));
        assert_eq!(report.filter(Severity::Warning).len(), 1);
    }

    #[test]
    fn test_validation_report_is_clean() {
        let report = ValidationReport::new();
        assert!(report.is_clean());
    }

    #[test]
    fn test_validator_with_checks_disabled() {
        let validator = PipelineValidator::new()
            .with_resource_check(false)
            .with_extension_check(false);
        let mut p = PipelineDefinition::new("no_checks");
        p.max_cpu_cores = 1.0;
        let mut step = PipelineStep::new("heavy", 0);
        step.cpu_cores = 100.0;
        step.requires_gpu = true;
        p.add_step(step);
        let report = validator.validate(&p);
        // Resource/ext checks disabled, so no errors from those
        assert!(report.is_valid());
    }
}
