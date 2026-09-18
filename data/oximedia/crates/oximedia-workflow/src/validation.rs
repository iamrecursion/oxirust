//! Workflow and task validation.

use crate::error::{Result, WorkflowError};
use crate::task::{Task, TaskType};
use crate::workflow::Workflow;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Workflow validator.
pub struct WorkflowValidator {
    /// Validation rules.
    rules: Vec<Box<dyn ValidationRule>>,
}

impl WorkflowValidator {
    /// Create a new workflow validator.
    #[must_use]
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a validation rule.
    #[must_use]
    pub fn add_rule(mut self, rule: Box<dyn ValidationRule>) -> Self {
        self.rules.push(rule);
        self
    }

    /// Add default validation rules.
    #[must_use]
    pub fn with_defaults(self) -> Self {
        self.add_rule(Box::new(NoCyclesRule))
            .add_rule(Box::new(ValidEdgesRule))
            .add_rule(Box::new(ValidTaskTypesRule))
            .add_rule(Box::new(NoOrphanTasksRule))
            .add_rule(Box::new(ValidDependenciesRule))
    }

    /// Validate a workflow.
    pub fn validate(&self, workflow: &Workflow) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();

        for rule in &self.rules {
            if let Err(e) = rule.validate(workflow) {
                report.add_error(e.to_string());
            }
        }

        if let Err(e) = workflow.validate() {
            report.add_error(e.to_string());
        }

        Ok(report)
    }
}

impl Default for WorkflowValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation rule trait.
pub trait ValidationRule: Send + Sync {
    /// Validate a workflow.
    fn validate(&self, workflow: &Workflow) -> Result<()>;

    /// Get rule name.
    fn name(&self) -> &str;
}

/// Rule: No cycles in workflow DAG.
pub struct NoCyclesRule;

impl ValidationRule for NoCyclesRule {
    fn validate(&self, workflow: &Workflow) -> Result<()> {
        if workflow.has_cycle() {
            return Err(WorkflowError::CycleDetected);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "NoCyclesRule"
    }
}

/// Rule: All edges reference valid tasks.
pub struct ValidEdgesRule;

impl ValidationRule for ValidEdgesRule {
    fn validate(&self, workflow: &Workflow) -> Result<()> {
        for edge in &workflow.edges {
            if !workflow.tasks.contains_key(&edge.from) {
                return Err(WorkflowError::TaskNotFound(edge.from.to_string()));
            }
            if !workflow.tasks.contains_key(&edge.to) {
                return Err(WorkflowError::TaskNotFound(edge.to.to_string()));
            }
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "ValidEdgesRule"
    }
}

/// Rule: All task types are valid.
pub struct ValidTaskTypesRule;

impl ValidationRule for ValidTaskTypesRule {
    fn validate(&self, workflow: &Workflow) -> Result<()> {
        for task in workflow.tasks.values() {
            validate_task_type(&task.task_type)?;
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "ValidTaskTypesRule"
    }
}

/// Rule: No orphan tasks (all tasks should be reachable).
pub struct NoOrphanTasksRule;

impl ValidationRule for NoOrphanTasksRule {
    fn validate(&self, workflow: &Workflow) -> Result<()> {
        if workflow.tasks.is_empty() {
            return Ok(());
        }

        let root_tasks = workflow.get_root_tasks();
        if root_tasks.is_empty() {
            return Err(WorkflowError::InvalidConfiguration(
                "No root tasks found".to_string(),
            ));
        }

        // BFS to find all reachable tasks
        let mut reachable = HashSet::new();
        let mut queue = root_tasks.clone();

        while let Some(task_id) = queue.pop() {
            if reachable.insert(task_id) {
                let dependents = workflow.get_dependents(&task_id);
                queue.extend(dependents);
            }
        }

        // Check for unreachable tasks
        for task_id in workflow.tasks.keys() {
            if !reachable.contains(task_id) {
                return Err(WorkflowError::InvalidConfiguration(format!(
                    "Task {task_id} is not reachable"
                )));
            }
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "NoOrphanTasksRule"
    }
}

/// Rule: Task dependencies are valid.
pub struct ValidDependenciesRule;

impl ValidationRule for ValidDependenciesRule {
    fn validate(&self, workflow: &Workflow) -> Result<()> {
        for task in workflow.tasks.values() {
            for dep_id in &task.dependencies {
                if !workflow.tasks.contains_key(dep_id) {
                    return Err(WorkflowError::TaskNotFound(dep_id.to_string()));
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "ValidDependenciesRule"
    }
}

/// Validation report.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Validation errors.
    pub errors: Vec<String>,
    /// Validation warnings.
    pub warnings: Vec<String>,
}

impl ValidationReport {
    /// Create a new validation report.
    #[must_use]
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Add an error.
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }

    /// Add a warning.
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Check if validation passed.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get error count.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Get warning count.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate task type configuration.
fn validate_task_type(task_type: &TaskType) -> Result<()> {
    match task_type {
        TaskType::Transcode {
            input,
            output,
            preset,
            ..
        } => {
            if preset.is_empty() {
                return Err(WorkflowError::InvalidConfiguration(
                    "Transcode preset cannot be empty".to_string(),
                ));
            }
            validate_file_path(input)?;
            validate_file_path(output)?;
        }

        TaskType::QualityControl { input, profile, .. } => {
            if profile.is_empty() {
                return Err(WorkflowError::InvalidConfiguration(
                    "QC profile cannot be empty".to_string(),
                ));
            }
            validate_file_path(input)?;
        }

        TaskType::Transfer {
            source,
            destination,
            ..
        } => {
            if source.is_empty() {
                return Err(WorkflowError::InvalidConfiguration(
                    "Transfer source cannot be empty".to_string(),
                ));
            }
            if destination.is_empty() {
                return Err(WorkflowError::InvalidConfiguration(
                    "Transfer destination cannot be empty".to_string(),
                ));
            }
        }

        TaskType::Notification { message, .. } => {
            if message.is_empty() {
                return Err(WorkflowError::InvalidConfiguration(
                    "Notification message cannot be empty".to_string(),
                ));
            }
        }

        TaskType::CustomScript { script, .. } => {
            validate_file_path(script)?;
        }

        TaskType::Analysis { input, .. } => {
            validate_file_path(input)?;
        }

        TaskType::HttpRequest { url, .. } => {
            if url.is_empty() {
                return Err(WorkflowError::InvalidConfiguration(
                    "HTTP URL cannot be empty".to_string(),
                ));
            }
        }

        TaskType::Wait { duration } => {
            if duration.is_zero() {
                return Err(WorkflowError::InvalidConfiguration(
                    "Wait duration cannot be zero".to_string(),
                ));
            }
        }

        TaskType::Conditional { condition, .. } => {
            if condition.is_empty() {
                return Err(WorkflowError::InvalidConfiguration(
                    "Conditional expression cannot be empty".to_string(),
                ));
            }
        }
    }

    Ok(())
}

fn validate_file_path(path: &Path) -> Result<()> {
    let path_str = path.to_string_lossy();
    if path_str.is_empty() {
        return Err(WorkflowError::InvalidConfiguration(
            "File path cannot be empty".to_string(),
        ));
    }

    // Check for potentially dangerous paths
    if path_str.contains("..") {
        return Err(WorkflowError::InvalidConfiguration(
            "Path traversal detected".to_string(),
        ));
    }

    Ok(())
}

/// Task validator.
pub struct TaskValidator;

impl TaskValidator {
    /// Validate a task.
    pub fn validate(task: &Task) -> Result<()> {
        // Validate task name
        if task.name.is_empty() {
            return Err(WorkflowError::InvalidConfiguration(
                "Task name cannot be empty".to_string(),
            ));
        }

        // Validate task type
        validate_task_type(&task.task_type)?;

        // Validate timeout
        if task.timeout.is_zero() {
            return Err(WorkflowError::InvalidConfiguration(
                "Task timeout cannot be zero".to_string(),
            ));
        }

        // Validate retry policy
        if task.retry.max_attempts == 0 {
            return Err(WorkflowError::InvalidConfiguration(
                "Retry max_attempts must be at least 1".to_string(),
            ));
        }

        Ok(())
    }
}

/// Workflow complexity analyzer.
pub struct ComplexityAnalyzer;

impl ComplexityAnalyzer {
    /// Analyze workflow complexity.
    #[must_use]
    pub fn analyze(workflow: &Workflow) -> ComplexityMetrics {
        let task_count = workflow.tasks.len();
        let edge_count = workflow.edges.len();

        // Calculate depth (longest path)
        let depth = Self::calculate_depth(workflow);

        // Calculate width (maximum parallel tasks)
        let width = Self::calculate_width(workflow);

        // Calculate branching factor
        let branching_factor = if task_count > 0 {
            edge_count as f64 / task_count as f64
        } else {
            0.0
        };

        // Calculate cyclomatic complexity
        let cyclomatic_complexity = edge_count.saturating_sub(task_count) + 2;

        ComplexityMetrics {
            task_count,
            edge_count,
            depth,
            width,
            branching_factor,
            cyclomatic_complexity,
        }
    }

    fn calculate_depth(workflow: &Workflow) -> usize {
        let roots = workflow.get_root_tasks();
        let mut max_depth = 0;

        for root in roots {
            let depth = Self::dfs_depth(workflow, root, &mut HashSet::new());
            max_depth = max_depth.max(depth);
        }

        max_depth
    }

    fn dfs_depth(
        workflow: &Workflow,
        task_id: crate::task::TaskId,
        visited: &mut HashSet<crate::task::TaskId>,
    ) -> usize {
        if visited.contains(&task_id) {
            return 0;
        }

        visited.insert(task_id);

        let dependents = workflow.get_dependents(&task_id);
        if dependents.is_empty() {
            return 1;
        }

        let mut max_depth = 0;
        for dep in dependents {
            let depth = Self::dfs_depth(workflow, dep, visited);
            max_depth = max_depth.max(depth);
        }

        max_depth + 1
    }

    fn calculate_width(workflow: &Workflow) -> usize {
        // Calculate maximum number of tasks at any level
        let roots = workflow.get_root_tasks();
        let mut levels: HashMap<crate::task::TaskId, usize> = HashMap::new();
        let mut level_counts: HashMap<usize, usize> = HashMap::new();

        for root in roots {
            Self::assign_levels(workflow, root, 0, &mut levels);
        }

        for level in levels.values() {
            *level_counts.entry(*level).or_insert(0) += 1;
        }

        level_counts.values().max().copied().unwrap_or(0)
    }

    fn assign_levels(
        workflow: &Workflow,
        task_id: crate::task::TaskId,
        level: usize,
        levels: &mut HashMap<crate::task::TaskId, usize>,
    ) {
        if let Some(&existing_level) = levels.get(&task_id) {
            if level <= existing_level {
                return;
            }
        }

        levels.insert(task_id, level);

        for dep in workflow.get_dependents(&task_id) {
            Self::assign_levels(workflow, dep, level + 1, levels);
        }
    }
}

/// Workflow complexity metrics.
#[derive(Debug, Clone)]
pub struct ComplexityMetrics {
    /// Total number of tasks.
    pub task_count: usize,
    /// Total number of edges.
    pub edge_count: usize,
    /// Maximum depth (longest path).
    pub depth: usize,
    /// Maximum width (parallel tasks).
    pub width: usize,
    /// Average branching factor.
    pub branching_factor: f64,
    /// Cyclomatic complexity.
    pub cyclomatic_complexity: usize,
}

impl ComplexityMetrics {
    /// Get complexity score (0-100).
    #[must_use]
    pub fn score(&self) -> f64 {
        let task_score = (self.task_count as f64 / 100.0).min(1.0) * 25.0;
        let depth_score = (self.depth as f64 / 10.0).min(1.0) * 25.0;
        let branching_score = (self.branching_factor / 3.0).min(1.0) * 25.0;
        let cyclomatic_score = (self.cyclomatic_complexity as f64 / 20.0).min(1.0) * 25.0;

        task_score + depth_score + branching_score + cyclomatic_score
    }

    /// Get complexity level.
    #[must_use]
    pub fn level(&self) -> ComplexityLevel {
        let score = self.score();
        if score < 25.0 {
            ComplexityLevel::Low
        } else if score < 50.0 {
            ComplexityLevel::Medium
        } else if score < 75.0 {
            ComplexityLevel::High
        } else {
            ComplexityLevel::VeryHigh
        }
    }
}

/// Complexity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexityLevel {
    /// Low complexity.
    Low,
    /// Medium complexity.
    Medium,
    /// High complexity.
    High,
    /// Very high complexity.
    VeryHigh,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_task(name: &str) -> Task {
        Task::new(
            name,
            TaskType::Wait {
                duration: Duration::from_secs(1),
            },
        )
    }

    #[test]
    fn test_workflow_validator() {
        let validator = WorkflowValidator::new().with_defaults();
        let workflow = Workflow::new("test");

        let report = validator
            .validate(&workflow)
            .expect("should succeed in test");
        assert!(report.is_valid());
    }

    #[test]
    fn test_cycle_detection() {
        let mut workflow = Workflow::new("test");
        let task1 = create_test_task("task1");
        let task2 = create_test_task("task2");

        let id1 = workflow.add_task(task1);
        let id2 = workflow.add_task(task2);

        workflow.add_edge(id1, id2).expect("should succeed in test");
        workflow.add_edge(id2, id1).expect("should succeed in test");

        let rule = NoCyclesRule;
        assert!(rule.validate(&workflow).is_err());
    }

    #[test]
    fn test_valid_edges_rule() {
        let mut workflow = Workflow::new("test");
        let task = create_test_task("task1");
        workflow.add_task(task);

        let rule = ValidEdgesRule;
        assert!(rule.validate(&workflow).is_ok());
    }

    #[test]
    fn test_task_validator() {
        let task = create_test_task("test-task");
        assert!(TaskValidator::validate(&task).is_ok());
    }

    #[test]
    fn test_task_validator_empty_name() {
        let task = create_test_task("");
        assert!(TaskValidator::validate(&task).is_err());
    }

    #[test]
    fn test_complexity_analyzer() {
        let mut workflow = Workflow::new("test");
        let task1 = create_test_task("task1");
        let task2 = create_test_task("task2");
        let task3 = create_test_task("task3");

        let id1 = workflow.add_task(task1);
        let id2 = workflow.add_task(task2);
        let id3 = workflow.add_task(task3);

        workflow.add_edge(id1, id2).expect("should succeed in test");
        workflow.add_edge(id2, id3).expect("should succeed in test");

        let metrics = ComplexityAnalyzer::analyze(&workflow);

        assert_eq!(metrics.task_count, 3);
        assert_eq!(metrics.edge_count, 2);
        assert_eq!(metrics.depth, 3);
    }

    #[test]
    fn test_complexity_score() {
        let metrics = ComplexityMetrics {
            task_count: 10,
            edge_count: 12,
            depth: 5,
            width: 3,
            branching_factor: 1.2,
            cyclomatic_complexity: 4,
        };

        let score = metrics.score();
        assert!(score > 0.0 && score <= 100.0);
    }

    #[test]
    fn test_complexity_level() {
        let low_metrics = ComplexityMetrics {
            task_count: 2,
            edge_count: 1,
            depth: 2,
            width: 1,
            branching_factor: 0.5,
            cyclomatic_complexity: 1,
        };

        assert_eq!(low_metrics.level(), ComplexityLevel::Low);
    }

    #[test]
    fn test_validation_report() {
        let mut report = ValidationReport::new();
        assert!(report.is_valid());

        report.add_error("Test error".to_string());
        assert!(!report.is_valid());
        assert_eq!(report.error_count(), 1);

        report.add_warning("Test warning".to_string());
        assert_eq!(report.warning_count(), 1);
    }
}
