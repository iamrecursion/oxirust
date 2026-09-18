//! Utility functions and helpers.

use crate::error::{Result, WorkflowError};
use crate::task::TaskType;
use crate::workflow::Workflow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Parse duration from string (e.g., "1h30m", "90s", "1.5h").
pub fn parse_duration(s: &str) -> Result<Duration> {
    let s = s.trim();

    // Try parsing as seconds (plain number)
    if let Ok(secs) = s.parse::<u64>() {
        return Ok(Duration::from_secs(secs));
    }

    // Parse with units
    let mut total_secs = 0u64;
    let mut current_num = String::new();

    for ch in s.chars() {
        if ch.is_ascii_digit() || ch == '.' {
            current_num.push(ch);
        } else {
            if current_num.is_empty() {
                continue;
            }

            let value: f64 = current_num.parse().map_err(|_| {
                WorkflowError::InvalidConfiguration(format!("Invalid duration: {s}"))
            })?;

            let unit_secs = match ch {
                's' => 1,
                'm' => 60,
                'h' => 3600,
                'd' => 86400,
                _ => {
                    return Err(WorkflowError::InvalidConfiguration(format!(
                        "Invalid duration unit: {ch}"
                    )))
                }
            };

            total_secs += (value * f64::from(unit_secs)) as u64;
            current_num.clear();
        }
    }

    if total_secs == 0 {
        return Err(WorkflowError::InvalidConfiguration(format!(
            "Invalid duration: {s}"
        )));
    }

    Ok(Duration::from_secs(total_secs))
}

/// Format duration as human-readable string.
#[must_use]
pub fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();

    if secs < 60 {
        return format!("{secs}s");
    }

    if secs < 3600 {
        let mins = secs / 60;
        let secs = secs % 60;
        if secs == 0 {
            return format!("{mins}m");
        }
        return format!("{mins}m{secs}s");
    }

    if secs < 86400 {
        let hours = secs / 3600;
        let mins = (secs % 3600) / 60;
        if mins == 0 {
            return format!("{hours}h");
        }
        return format!("{hours}h{mins}m");
    }

    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    if hours == 0 {
        return format!("{days}d");
    }
    format!("{days}d{hours}h")
}

/// Sanitize task name (remove invalid characters).
#[must_use]
pub fn sanitize_task_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Generate a unique task name.
#[must_use]
pub fn generate_task_name(prefix: &str, index: usize) -> String {
    format!("{}-{}", sanitize_task_name(prefix), index)
}

/// Expand environment variables in a string.
#[must_use]
pub fn expand_env_vars(s: &str) -> String {
    let mut result = s.to_string();

    for (key, value) in std::env::vars() {
        let placeholder = format!("${{{key}}}");
        result = result.replace(&placeholder, &value);

        let placeholder = format!("${key}");
        result = result.replace(&placeholder, &value);
    }

    result
}

/// Expand template variables in a string.
#[must_use]
pub fn expand_template(template: &str, variables: &HashMap<String, String>) -> String {
    let mut result = template.to_string();

    for (key, value) in variables {
        let placeholder = format!("{{{key}}}");
        result = result.replace(&placeholder, value);

        let placeholder = format!("${{{key}}}");
        result = result.replace(&placeholder, value);
    }

    result
}

/// Calculate estimated workflow duration.
#[must_use]
pub fn estimate_workflow_duration(workflow: &Workflow) -> Duration {
    let mut max_duration = Duration::ZERO;

    // Get topological order
    if let Ok(sorted_tasks) = workflow.topological_sort() {
        let mut task_end_times: HashMap<crate::task::TaskId, Duration> = HashMap::new();

        for &task_id in &sorted_tasks {
            if let Some(task) = workflow.get_task(&task_id) {
                // Calculate start time based on graph edges (workflow.get_dependencies)
                // or the task-level dependency list, whichever is non-empty.
                let mut start_time = Duration::ZERO;

                let edge_deps = workflow.get_dependencies(&task_id);
                let all_deps: Vec<crate::task::TaskId> = if edge_deps.is_empty() {
                    task.dependencies.clone()
                } else {
                    edge_deps
                };

                for dep_id in &all_deps {
                    if let Some(&dep_end_time) = task_end_times.get(dep_id) {
                        start_time = start_time.max(dep_end_time);
                    }
                }

                // Calculate end time
                let end_time = start_time + task.timeout;
                task_end_times.insert(task_id, end_time);

                max_duration = max_duration.max(end_time);
            }
        }
    }

    max_duration
}

/// Find critical path in workflow.
#[must_use]
pub fn find_critical_path(workflow: &Workflow) -> Vec<crate::task::TaskId> {
    let mut task_durations: HashMap<_, _> = HashMap::new();
    let mut task_paths: HashMap<_, Vec<_>> = HashMap::new();

    if let Ok(sorted_tasks) = workflow.topological_sort() {
        for &task_id in &sorted_tasks {
            if let Some(task) = workflow.get_task(&task_id) {
                let deps = workflow.get_dependencies(&task_id);

                if deps.is_empty() {
                    task_durations.insert(task_id, task.timeout);
                    task_paths.insert(task_id, vec![task_id]);
                } else {
                    let mut max_duration = Duration::ZERO;
                    let mut max_path = Vec::new();

                    for dep_id in deps {
                        if let Some(&dep_duration) = task_durations.get(&dep_id) {
                            let total_duration = dep_duration + task.timeout;
                            if total_duration > max_duration {
                                max_duration = total_duration;
                                max_path = task_paths.get(&dep_id).cloned().unwrap_or_default();
                            }
                        }
                    }

                    max_path.push(task_id);
                    task_durations.insert(task_id, max_duration);
                    task_paths.insert(task_id, max_path);
                }
            }
        }

        // Find the path with maximum duration
        let mut critical_path = Vec::new();
        let mut max_duration = Duration::ZERO;

        for (task_id, &duration) in &task_durations {
            if duration > max_duration {
                max_duration = duration;
                critical_path = task_paths.get(task_id).cloned().unwrap_or_default();
            }
        }

        return critical_path;
    }

    Vec::new()
}

/// Merge workflow configurations.
#[must_use]
pub fn merge_configs(
    base: &crate::workflow::WorkflowConfig,
    override_config: &crate::workflow::WorkflowConfig,
) -> crate::workflow::WorkflowConfig {
    let mut merged = base.clone();

    if override_config.max_concurrent_tasks != base.max_concurrent_tasks {
        merged.max_concurrent_tasks = override_config.max_concurrent_tasks;
    }

    if override_config.global_timeout.is_some() {
        merged.global_timeout = override_config.global_timeout;
    }

    merged.fail_fast = override_config.fail_fast || base.fail_fast;
    merged.continue_on_error = override_config.continue_on_error || base.continue_on_error;

    // Merge variables
    for (key, value) in &override_config.variables {
        merged.variables.insert(key.clone(), value.clone());
    }

    merged
}

/// Clone a workflow with a new ID.
#[must_use]
pub fn clone_workflow(workflow: &Workflow, new_name: Option<String>) -> Workflow {
    let mut cloned = Workflow::new(new_name.unwrap_or_else(|| format!("{}-copy", workflow.name)));
    cloned.description = workflow.description.clone();
    cloned.config = workflow.config.clone();
    cloned.metadata = workflow.metadata.clone();

    // Clone tasks with ID mapping
    let mut id_map = HashMap::new();

    for task in workflow.tasks.values() {
        let mut new_task = task.clone();
        let old_id = new_task.id;
        new_task.id = crate::task::TaskId::new();
        id_map.insert(old_id, new_task.id);
        cloned.tasks.insert(new_task.id, new_task);
    }

    // Clone edges with remapped IDs
    for edge in &workflow.edges {
        if let (Some(&from_id), Some(&to_id)) = (id_map.get(&edge.from), id_map.get(&edge.to)) {
            cloned.edges.push(crate::workflow::Edge {
                from: from_id,
                to: to_id,
                condition: edge.condition.clone(),
            });
        }
    }

    cloned
}

/// Normalize file paths in workflow.
pub fn normalize_paths(workflow: &mut Workflow, base_path: &Path) -> Result<()> {
    for task in workflow.tasks.values_mut() {
        match &mut task.task_type {
            TaskType::Transcode { input, output, .. } => {
                *input = normalize_path(input, base_path);
                *output = normalize_path(output, base_path);
            }
            TaskType::QualityControl { input, .. } => {
                *input = normalize_path(input, base_path);
            }
            TaskType::Analysis { input, output, .. } => {
                *input = normalize_path(input, base_path);
                if let Some(out) = output {
                    *out = normalize_path(out, base_path);
                }
            }
            TaskType::CustomScript { script, .. } => {
                *script = normalize_path(script, base_path);
            }
            _ => {}
        }
    }

    Ok(())
}

fn normalize_path(path: &Path, base_path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_path.join(path)
    }
}

/// Calculate task parallelism opportunities.
#[must_use]
pub fn calculate_parallelism(workflow: &Workflow) -> Vec<Vec<crate::task::TaskId>> {
    let mut levels: Vec<Vec<crate::task::TaskId>> = Vec::new();

    if let Ok(sorted_tasks) = workflow.topological_sort() {
        let mut task_levels: HashMap<_, _> = HashMap::new();

        for &task_id in &sorted_tasks {
            let deps = workflow.get_dependencies(&task_id);

            if deps.is_empty() {
                task_levels.insert(task_id, 0);
            } else {
                let mut max_level = 0;
                for dep_id in deps {
                    if let Some(&level) = task_levels.get(&dep_id) {
                        max_level = max_level.max(level + 1);
                    }
                }
                task_levels.insert(task_id, max_level);
            }
        }

        // Group tasks by level
        let max_level = task_levels.values().max().copied().unwrap_or(0);
        levels = vec![Vec::new(); max_level + 1];

        for (task_id, level) in task_levels {
            levels[level].push(task_id);
        }
    }

    levels
}

/// Get workflow statistics.
#[must_use]
pub fn get_workflow_statistics(workflow: &Workflow) -> WorkflowStatistics {
    let task_count = workflow.tasks.len();
    let edge_count = workflow.edges.len();

    let mut task_type_counts: HashMap<String, usize> = HashMap::new();

    for task in workflow.tasks.values() {
        let type_name = match &task.task_type {
            TaskType::Transcode { .. } => "Transcode",
            TaskType::QualityControl { .. } => "QualityControl",
            TaskType::Transfer { .. } => "Transfer",
            TaskType::Notification { .. } => "Notification",
            TaskType::CustomScript { .. } => "CustomScript",
            TaskType::Analysis { .. } => "Analysis",
            TaskType::Conditional { .. } => "Conditional",
            TaskType::Wait { .. } => "Wait",
            TaskType::HttpRequest { .. } => "HttpRequest",
        };

        *task_type_counts.entry(type_name.to_string()).or_insert(0) += 1;
    }

    let root_count = workflow.get_root_tasks().len();
    let leaf_count = workflow.get_leaf_tasks().len();

    let estimated_duration = estimate_workflow_duration(workflow);
    let critical_path = find_critical_path(workflow);

    WorkflowStatistics {
        task_count,
        edge_count,
        root_task_count: root_count,
        leaf_task_count: leaf_count,
        task_type_counts,
        estimated_duration,
        critical_path_length: critical_path.len(),
    }
}

/// Workflow statistics.
#[derive(Debug, Clone)]
pub struct WorkflowStatistics {
    /// Total number of tasks.
    pub task_count: usize,
    /// Total number of edges.
    pub edge_count: usize,
    /// Number of root tasks.
    pub root_task_count: usize,
    /// Number of leaf tasks.
    pub leaf_task_count: usize,
    /// Task type distribution.
    pub task_type_counts: HashMap<String, usize>,
    /// Estimated duration.
    pub estimated_duration: Duration,
    /// Critical path length.
    pub critical_path_length: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::Task;

    #[test]
    fn test_parse_duration() {
        assert_eq!(
            parse_duration("60").expect("should succeed in test"),
            Duration::from_secs(60)
        );
        assert_eq!(
            parse_duration("1m").expect("should succeed in test"),
            Duration::from_secs(60)
        );
        assert_eq!(
            parse_duration("1h").expect("should succeed in test"),
            Duration::from_secs(3600)
        );
        assert_eq!(
            parse_duration("1h30m").expect("should succeed in test"),
            Duration::from_secs(5400)
        );
        assert_eq!(
            parse_duration("1d").expect("should succeed in test"),
            Duration::from_secs(86400)
        );
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(Duration::from_secs(60)), "1m");
        assert_eq!(format_duration(Duration::from_secs(90)), "1m30s");
        assert_eq!(format_duration(Duration::from_secs(3600)), "1h");
        assert_eq!(format_duration(Duration::from_secs(3660)), "1h1m");
    }

    #[test]
    fn test_sanitize_task_name() {
        assert_eq!(sanitize_task_name("hello world"), "hello_world");
        assert_eq!(sanitize_task_name("test-task"), "test-task");
        assert_eq!(sanitize_task_name("test@task!"), "test_task_");
    }

    #[test]
    fn test_generate_task_name() {
        assert_eq!(generate_task_name("test", 1), "test-1");
        assert_eq!(generate_task_name("my task", 5), "my_task-5");
    }

    #[test]
    fn test_expand_template() {
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "world".to_string());
        vars.insert("num".to_string(), "42".to_string());

        let result = expand_template("Hello {name}, answer is {num}", &vars);
        assert_eq!(result, "Hello world, answer is 42");
    }

    #[test]
    fn test_estimate_workflow_duration() {
        let mut workflow = Workflow::new("test");
        let task1 = Task::new(
            "task1",
            TaskType::Wait {
                duration: Duration::from_secs(10),
            },
        )
        .with_timeout(Duration::from_secs(10));

        let task2 = Task::new(
            "task2",
            TaskType::Wait {
                duration: Duration::from_secs(20),
            },
        )
        .with_timeout(Duration::from_secs(20));

        let id1 = workflow.add_task(task1);
        let id2 = workflow.add_task(task2);
        workflow.add_edge(id1, id2).expect("should succeed in test");

        let duration = estimate_workflow_duration(&workflow);
        assert_eq!(duration, Duration::from_secs(30));
    }

    #[test]
    fn test_clone_workflow() {
        let mut workflow = Workflow::new("original");
        let task = Task::new(
            "task1",
            TaskType::Wait {
                duration: Duration::from_secs(1),
            },
        );
        workflow.add_task(task);

        let cloned = clone_workflow(&workflow, Some("cloned".to_string()));

        assert_eq!(cloned.name, "cloned");
        assert_eq!(cloned.tasks.len(), workflow.tasks.len());
        assert_ne!(cloned.id, workflow.id);
    }

    #[test]
    fn test_calculate_parallelism() {
        let mut workflow = Workflow::new("test");

        let task1 = Task::new(
            "task1",
            TaskType::Wait {
                duration: Duration::from_secs(1),
            },
        );
        let task2 = Task::new(
            "task2",
            TaskType::Wait {
                duration: Duration::from_secs(1),
            },
        );
        let task3 = Task::new(
            "task3",
            TaskType::Wait {
                duration: Duration::from_secs(1),
            },
        );

        let id1 = workflow.add_task(task1);
        let id2 = workflow.add_task(task2);
        let id3 = workflow.add_task(task3);

        workflow.add_edge(id1, id3).expect("should succeed in test");
        workflow.add_edge(id2, id3).expect("should succeed in test");

        let levels = calculate_parallelism(&workflow);
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].len(), 2); // task1 and task2 can run in parallel
        assert_eq!(levels[1].len(), 1); // task3 runs after
    }

    #[test]
    fn test_get_workflow_statistics() {
        let mut workflow = Workflow::new("test");
        let task = Task::new(
            "task1",
            TaskType::Wait {
                duration: Duration::from_secs(1),
            },
        );
        workflow.add_task(task);

        let stats = get_workflow_statistics(&workflow);
        assert_eq!(stats.task_count, 1);
        assert_eq!(stats.edge_count, 0);
    }

    #[test]
    fn test_merge_configs() {
        let base = crate::workflow::WorkflowConfig {
            max_concurrent_tasks: 2,
            ..Default::default()
        };

        let override_config = crate::workflow::WorkflowConfig {
            max_concurrent_tasks: 4,
            fail_fast: true,
            ..Default::default()
        };

        let merged = merge_configs(&base, &override_config);
        assert_eq!(merged.max_concurrent_tasks, 4);
        assert!(merged.fail_fast);
    }
}
