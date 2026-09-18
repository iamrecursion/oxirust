//! Workflow diff: compare two workflow versions to identify changes.
//!
//! Provides structural comparison of workflow definitions, identifying
//! added, removed, and modified tasks and edges. Useful for version control,
//! change auditing, and migration planning.

use crate::task::{Task, TaskId};
use crate::workflow::Workflow;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Type of change detected between two workflow versions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeType {
    /// A new element was added.
    Added,
    /// An existing element was removed.
    Removed,
    /// An existing element was modified.
    Modified,
    /// No change.
    Unchanged,
}

impl std::fmt::Display for ChangeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Added => write!(f, "added"),
            Self::Removed => write!(f, "removed"),
            Self::Modified => write!(f, "modified"),
            Self::Unchanged => write!(f, "unchanged"),
        }
    }
}

/// A change to a specific task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskChange {
    /// Task name used for identification.
    pub task_name: String,
    /// Type of change.
    pub change_type: ChangeType,
    /// Detailed field-level changes (only for Modified).
    pub field_changes: Vec<FieldChange>,
    /// Task from the old version (if present).
    pub old_task_type: Option<String>,
    /// Task from the new version (if present).
    pub new_task_type: Option<String>,
}

/// A change to a specific field within a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldChange {
    /// Field name.
    pub field: String,
    /// Old value (as JSON string).
    pub old_value: String,
    /// New value (as JSON string).
    pub new_value: String,
}

/// A change to an edge (dependency).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeChange {
    /// Source task name.
    pub from_task: String,
    /// Target task name.
    pub to_task: String,
    /// Type of change.
    pub change_type: ChangeType,
    /// Old condition (if any).
    pub old_condition: Option<String>,
    /// New condition (if any).
    pub new_condition: Option<String>,
}

/// A change to workflow-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigChange {
    /// Config field name.
    pub field: String,
    /// Old value.
    pub old_value: String,
    /// New value.
    pub new_value: String,
}

/// Complete diff result between two workflow versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowDiff {
    /// Name of the old workflow.
    pub old_name: String,
    /// Name of the new workflow.
    pub new_name: String,
    /// Whether the workflow name changed.
    pub name_changed: bool,
    /// Whether the description changed.
    pub description_changed: bool,
    /// Old description.
    pub old_description: String,
    /// New description.
    pub new_description: String,
    /// Task-level changes.
    pub task_changes: Vec<TaskChange>,
    /// Edge-level changes.
    pub edge_changes: Vec<EdgeChange>,
    /// Configuration changes.
    pub config_changes: Vec<ConfigChange>,
    /// Metadata changes.
    pub metadata_changes: Vec<FieldChange>,
    /// Summary statistics.
    pub summary: DiffSummary,
}

/// Summary statistics of a diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffSummary {
    /// Number of tasks added.
    pub tasks_added: usize,
    /// Number of tasks removed.
    pub tasks_removed: usize,
    /// Number of tasks modified.
    pub tasks_modified: usize,
    /// Number of tasks unchanged.
    pub tasks_unchanged: usize,
    /// Number of edges added.
    pub edges_added: usize,
    /// Number of edges removed.
    pub edges_removed: usize,
    /// Number of edges modified.
    pub edges_modified: usize,
    /// Number of config fields changed.
    pub config_changes: usize,
    /// Whether there are any changes at all.
    pub has_changes: bool,
}

/// Compute the diff between two workflow versions.
///
/// Tasks are matched by name (not by ID), since IDs are typically
/// regenerated when importing/exporting workflows.
#[must_use]
pub fn diff_workflows(old: &Workflow, new: &Workflow) -> WorkflowDiff {
    let task_changes = diff_tasks(old, new);
    let edge_changes = diff_edges(old, new);
    let config_changes = diff_config(old, new);
    let metadata_changes = diff_metadata(old, new);

    let tasks_added = task_changes
        .iter()
        .filter(|c| c.change_type == ChangeType::Added)
        .count();
    let tasks_removed = task_changes
        .iter()
        .filter(|c| c.change_type == ChangeType::Removed)
        .count();
    let tasks_modified = task_changes
        .iter()
        .filter(|c| c.change_type == ChangeType::Modified)
        .count();
    let tasks_unchanged = task_changes
        .iter()
        .filter(|c| c.change_type == ChangeType::Unchanged)
        .count();
    let edges_added = edge_changes
        .iter()
        .filter(|c| c.change_type == ChangeType::Added)
        .count();
    let edges_removed = edge_changes
        .iter()
        .filter(|c| c.change_type == ChangeType::Removed)
        .count();
    let edges_modified = edge_changes
        .iter()
        .filter(|c| c.change_type == ChangeType::Modified)
        .count();

    let has_changes = tasks_added > 0
        || tasks_removed > 0
        || tasks_modified > 0
        || edges_added > 0
        || edges_removed > 0
        || edges_modified > 0
        || !config_changes.is_empty()
        || !metadata_changes.is_empty()
        || old.name != new.name
        || old.description != new.description;

    WorkflowDiff {
        old_name: old.name.clone(),
        new_name: new.name.clone(),
        name_changed: old.name != new.name,
        description_changed: old.description != new.description,
        old_description: old.description.clone(),
        new_description: new.description.clone(),
        task_changes,
        edge_changes,
        config_changes: config_changes.clone(),
        metadata_changes,
        summary: DiffSummary {
            tasks_added,
            tasks_removed,
            tasks_modified,
            tasks_unchanged,
            edges_added,
            edges_removed,
            edges_modified,
            config_changes: config_changes.len(),
            has_changes,
        },
    }
}

/// Diff tasks between old and new workflows (matched by name).
fn diff_tasks(old: &Workflow, new: &Workflow) -> Vec<TaskChange> {
    let old_by_name: HashMap<&str, &Task> =
        old.tasks.values().map(|t| (t.name.as_str(), t)).collect();
    let new_by_name: HashMap<&str, &Task> =
        new.tasks.values().map(|t| (t.name.as_str(), t)).collect();

    let old_names: HashSet<&str> = old_by_name.keys().copied().collect();
    let new_names: HashSet<&str> = new_by_name.keys().copied().collect();

    let mut changes = Vec::new();

    // Added tasks
    for &name in new_names.difference(&old_names) {
        let new_task = new_by_name[name];
        changes.push(TaskChange {
            task_name: name.to_string(),
            change_type: ChangeType::Added,
            field_changes: Vec::new(),
            old_task_type: None,
            new_task_type: Some(format!("{:?}", new_task.task_type)),
        });
    }

    // Removed tasks
    for &name in old_names.difference(&new_names) {
        let old_task = old_by_name[name];
        changes.push(TaskChange {
            task_name: name.to_string(),
            change_type: ChangeType::Removed,
            field_changes: Vec::new(),
            old_task_type: Some(format!("{:?}", old_task.task_type)),
            new_task_type: None,
        });
    }

    // Potentially modified tasks
    for &name in old_names.intersection(&new_names) {
        let old_task = old_by_name[name];
        let new_task = new_by_name[name];
        let field_changes = diff_task_fields(old_task, new_task);

        let change_type = if field_changes.is_empty() {
            ChangeType::Unchanged
        } else {
            ChangeType::Modified
        };

        changes.push(TaskChange {
            task_name: name.to_string(),
            change_type,
            field_changes,
            old_task_type: Some(format!("{:?}", old_task.task_type)),
            new_task_type: Some(format!("{:?}", new_task.task_type)),
        });
    }

    // Sort for deterministic output
    changes.sort_by(|a, b| a.task_name.cmp(&b.task_name));
    changes
}

/// Compare individual task fields.
fn diff_task_fields(old: &Task, new: &Task) -> Vec<FieldChange> {
    let mut changes = Vec::new();

    let old_type_str = format!("{:?}", old.task_type);
    let new_type_str = format!("{:?}", new.task_type);
    if old_type_str != new_type_str {
        changes.push(FieldChange {
            field: "task_type".to_string(),
            old_value: old_type_str,
            new_value: new_type_str,
        });
    }

    let old_priority = format!("{:?}", old.priority);
    let new_priority = format!("{:?}", new.priority);
    if old_priority != new_priority {
        changes.push(FieldChange {
            field: "priority".to_string(),
            old_value: old_priority,
            new_value: new_priority,
        });
    }

    if old.timeout != new.timeout {
        changes.push(FieldChange {
            field: "timeout".to_string(),
            old_value: format!("{:?}", old.timeout),
            new_value: format!("{:?}", new.timeout),
        });
    }

    if old.conditions != new.conditions {
        changes.push(FieldChange {
            field: "conditions".to_string(),
            old_value: format!("{:?}", old.conditions),
            new_value: format!("{:?}", new.conditions),
        });
    }

    if old.metadata != new.metadata {
        changes.push(FieldChange {
            field: "metadata".to_string(),
            old_value: format!("{:?}", old.metadata),
            new_value: format!("{:?}", new.metadata),
        });
    }

    changes
}

/// Diff edges between old and new workflows.
///
/// Edges are identified by (from_task_name, to_task_name).
fn diff_edges(old: &Workflow, new: &Workflow) -> Vec<EdgeChange> {
    let old_name_map: HashMap<TaskId, &str> = old
        .tasks
        .values()
        .map(|t| (t.id, t.name.as_str()))
        .collect();
    let new_name_map: HashMap<TaskId, &str> = new
        .tasks
        .values()
        .map(|t| (t.id, t.name.as_str()))
        .collect();

    // Build edge signature maps: (from_name, to_name) -> condition
    let old_edges: HashMap<(String, String), Option<String>> = old
        .edges
        .iter()
        .filter_map(|e| {
            let from = old_name_map.get(&e.from)?.to_string();
            let to = old_name_map.get(&e.to)?.to_string();
            Some(((from, to), e.condition.clone()))
        })
        .collect();

    let new_edges: HashMap<(String, String), Option<String>> = new
        .edges
        .iter()
        .filter_map(|e| {
            let from = new_name_map.get(&e.from)?.to_string();
            let to = new_name_map.get(&e.to)?.to_string();
            Some(((from, to), e.condition.clone()))
        })
        .collect();

    let old_keys: HashSet<&(String, String)> = old_edges.keys().collect();
    let new_keys: HashSet<&(String, String)> = new_edges.keys().collect();

    let mut changes = Vec::new();

    // Added edges
    for &key in new_keys.difference(&old_keys) {
        changes.push(EdgeChange {
            from_task: key.0.clone(),
            to_task: key.1.clone(),
            change_type: ChangeType::Added,
            old_condition: None,
            new_condition: new_edges.get(key).cloned().flatten(),
        });
    }

    // Removed edges
    for &key in old_keys.difference(&new_keys) {
        changes.push(EdgeChange {
            from_task: key.0.clone(),
            to_task: key.1.clone(),
            change_type: ChangeType::Removed,
            old_condition: old_edges.get(key).cloned().flatten(),
            new_condition: None,
        });
    }

    // Potentially modified edges (same endpoints, different condition)
    for &key in old_keys.intersection(&new_keys) {
        let old_cond = old_edges.get(key).cloned().flatten();
        let new_cond = new_edges.get(key).cloned().flatten();
        if old_cond != new_cond {
            changes.push(EdgeChange {
                from_task: key.0.clone(),
                to_task: key.1.clone(),
                change_type: ChangeType::Modified,
                old_condition: old_cond,
                new_condition: new_cond,
            });
        }
    }

    changes.sort_by(|a, b| (&a.from_task, &a.to_task).cmp(&(&b.from_task, &b.to_task)));
    changes
}

/// Diff workflow-level config.
fn diff_config(old: &Workflow, new: &Workflow) -> Vec<ConfigChange> {
    let mut changes = Vec::new();

    if old.config.max_concurrent_tasks != new.config.max_concurrent_tasks {
        changes.push(ConfigChange {
            field: "max_concurrent_tasks".to_string(),
            old_value: old.config.max_concurrent_tasks.to_string(),
            new_value: new.config.max_concurrent_tasks.to_string(),
        });
    }

    if old.config.fail_fast != new.config.fail_fast {
        changes.push(ConfigChange {
            field: "fail_fast".to_string(),
            old_value: old.config.fail_fast.to_string(),
            new_value: new.config.fail_fast.to_string(),
        });
    }

    if old.config.continue_on_error != new.config.continue_on_error {
        changes.push(ConfigChange {
            field: "continue_on_error".to_string(),
            old_value: old.config.continue_on_error.to_string(),
            new_value: new.config.continue_on_error.to_string(),
        });
    }

    if old.config.global_timeout != new.config.global_timeout {
        changes.push(ConfigChange {
            field: "global_timeout".to_string(),
            old_value: format!("{:?}", old.config.global_timeout),
            new_value: format!("{:?}", new.config.global_timeout),
        });
    }

    changes
}

/// Diff workflow metadata.
fn diff_metadata(old: &Workflow, new: &Workflow) -> Vec<FieldChange> {
    let mut changes = Vec::new();

    let old_keys: HashSet<&String> = old.metadata.keys().collect();
    let new_keys: HashSet<&String> = new.metadata.keys().collect();

    for key in new_keys.difference(&old_keys) {
        changes.push(FieldChange {
            field: (*key).clone(),
            old_value: String::new(),
            new_value: new.metadata.get(*key).cloned().unwrap_or_default(),
        });
    }

    for key in old_keys.difference(&new_keys) {
        changes.push(FieldChange {
            field: (*key).clone(),
            old_value: old.metadata.get(*key).cloned().unwrap_or_default(),
            new_value: String::new(),
        });
    }

    for key in old_keys.intersection(&new_keys) {
        let old_val = old.metadata.get(*key).cloned().unwrap_or_default();
        let new_val = new.metadata.get(*key).cloned().unwrap_or_default();
        if old_val != new_val {
            changes.push(FieldChange {
                field: (*key).clone(),
                old_value: old_val,
                new_value: new_val,
            });
        }
    }

    changes.sort_by(|a, b| a.field.cmp(&b.field));
    changes
}

/// Generate a human-readable text summary of a diff.
#[must_use]
pub fn format_diff(diff: &WorkflowDiff) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "Workflow Diff: '{}' -> '{}'",
        diff.old_name, diff.new_name
    ));
    lines.push(format!(
        "Tasks: +{} -{} ~{} ={}\n",
        diff.summary.tasks_added,
        diff.summary.tasks_removed,
        diff.summary.tasks_modified,
        diff.summary.tasks_unchanged,
    ));

    if diff.name_changed {
        lines.push(format!(
            "  Name: '{}' -> '{}'",
            diff.old_name, diff.new_name
        ));
    }
    if diff.description_changed {
        lines.push(format!(
            "  Description: '{}' -> '{}'",
            diff.old_description, diff.new_description
        ));
    }

    for tc in &diff.task_changes {
        if tc.change_type == ChangeType::Unchanged {
            continue;
        }
        lines.push(format!("  Task '{}': {}", tc.task_name, tc.change_type));
        for fc in &tc.field_changes {
            lines.push(format!(
                "    {}: '{}' -> '{}'",
                fc.field, fc.old_value, fc.new_value
            ));
        }
    }

    for ec in &diff.edge_changes {
        lines.push(format!(
            "  Edge {} -> {}: {}",
            ec.from_task, ec.to_task, ec.change_type
        ));
    }

    for cc in &diff.config_changes {
        lines.push(format!(
            "  Config {}: '{}' -> '{}'",
            cc.field, cc.old_value, cc.new_value
        ));
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::{TaskPriority, TaskType};
    use std::time::Duration;

    fn make_task(name: &str) -> Task {
        Task::new(
            name,
            TaskType::Wait {
                duration: Duration::from_secs(1),
            },
        )
    }

    fn make_simple_workflow() -> Workflow {
        let mut wf = Workflow::new("test-workflow").with_description("A test workflow");
        let t1 = wf.add_task(make_task("ingest"));
        let t2 = wf.add_task(make_task("transcode"));
        let t3 = wf.add_task(make_task("deliver"));
        wf.add_edge(t1, t2).expect("edge");
        wf.add_edge(t2, t3).expect("edge");
        wf
    }

    #[test]
    fn test_identical_workflows_no_changes() {
        let wf1 = make_simple_workflow();
        let wf2 = make_simple_workflow();

        let diff = diff_workflows(&wf1, &wf2);
        assert!(!diff.summary.has_changes);
        assert_eq!(diff.summary.tasks_added, 0);
        assert_eq!(diff.summary.tasks_removed, 0);
        assert_eq!(diff.summary.tasks_modified, 0);
        assert_eq!(diff.summary.tasks_unchanged, 3);
    }

    #[test]
    fn test_added_task() {
        let wf1 = make_simple_workflow();
        let mut wf2 = make_simple_workflow();
        wf2.add_task(make_task("qc_check"));

        let diff = diff_workflows(&wf1, &wf2);
        assert!(diff.summary.has_changes);
        assert_eq!(diff.summary.tasks_added, 1);

        let added = diff
            .task_changes
            .iter()
            .find(|c| c.change_type == ChangeType::Added)
            .expect("find added");
        assert_eq!(added.task_name, "qc_check");
    }

    #[test]
    fn test_removed_task() {
        let mut wf1 = Workflow::new("old");
        wf1.add_task(make_task("ingest"));
        wf1.add_task(make_task("obsolete"));

        let mut wf2 = Workflow::new("old");
        wf2.add_task(make_task("ingest"));

        let diff = diff_workflows(&wf1, &wf2);
        assert_eq!(diff.summary.tasks_removed, 1);

        let removed = diff
            .task_changes
            .iter()
            .find(|c| c.change_type == ChangeType::Removed)
            .expect("find removed");
        assert_eq!(removed.task_name, "obsolete");
    }

    #[test]
    fn test_modified_task_priority() {
        let mut wf1 = Workflow::new("wf");
        wf1.add_task(make_task("ingest"));

        let mut wf2 = Workflow::new("wf");
        wf2.add_task(make_task("ingest").with_priority(TaskPriority::High));

        let diff = diff_workflows(&wf1, &wf2);
        assert_eq!(diff.summary.tasks_modified, 1);

        let modified = diff
            .task_changes
            .iter()
            .find(|c| c.change_type == ChangeType::Modified)
            .expect("find modified");
        assert_eq!(modified.task_name, "ingest");
        assert!(modified.field_changes.iter().any(|f| f.field == "priority"));
    }

    #[test]
    fn test_added_edge() {
        let mut wf1 = Workflow::new("wf");
        let _t1 = wf1.add_task(make_task("a"));
        let _t2 = wf1.add_task(make_task("b"));

        let mut wf2 = Workflow::new("wf");
        let t1b = wf2.add_task(make_task("a"));
        let t2b = wf2.add_task(make_task("b"));
        wf2.add_edge(t1b, t2b).expect("edge");

        let diff = diff_workflows(&wf1, &wf2);
        assert_eq!(diff.summary.edges_added, 1);
    }

    #[test]
    fn test_removed_edge() {
        let mut wf1 = Workflow::new("wf");
        let t1 = wf1.add_task(make_task("a"));
        let t2 = wf1.add_task(make_task("b"));
        wf1.add_edge(t1, t2).expect("edge");

        let mut wf2 = Workflow::new("wf");
        wf2.add_task(make_task("a"));
        wf2.add_task(make_task("b"));

        let diff = diff_workflows(&wf1, &wf2);
        assert_eq!(diff.summary.edges_removed, 1);
    }

    #[test]
    fn test_modified_edge_condition() {
        let mut wf1 = Workflow::new("wf");
        let t1 = wf1.add_task(make_task("a"));
        let t2 = wf1.add_task(make_task("b"));
        wf1.add_edge(t1, t2).expect("edge");

        let mut wf2 = Workflow::new("wf");
        let t1b = wf2.add_task(make_task("a"));
        let t2b = wf2.add_task(make_task("b"));
        wf2.add_conditional_edge(t1b, t2b, "status == ok".to_string())
            .expect("edge");

        let diff = diff_workflows(&wf1, &wf2);
        assert_eq!(diff.summary.edges_modified, 1);
    }

    #[test]
    fn test_config_change() {
        let mut wf1 = Workflow::new("wf");
        wf1.config.max_concurrent_tasks = 4;

        let mut wf2 = Workflow::new("wf");
        wf2.config.max_concurrent_tasks = 8;

        let diff = diff_workflows(&wf1, &wf2);
        assert_eq!(diff.summary.config_changes, 1);
        assert_eq!(diff.config_changes[0].field, "max_concurrent_tasks");
    }

    #[test]
    fn test_name_change() {
        let wf1 = Workflow::new("old-name");
        let wf2 = Workflow::new("new-name");

        let diff = diff_workflows(&wf1, &wf2);
        assert!(diff.name_changed);
        assert!(diff.summary.has_changes);
    }

    #[test]
    fn test_description_change() {
        let wf1 = Workflow::new("wf").with_description("old desc");
        let wf2 = Workflow::new("wf").with_description("new desc");

        let diff = diff_workflows(&wf1, &wf2);
        assert!(diff.description_changed);
    }

    #[test]
    fn test_metadata_change() {
        let wf1 = Workflow::new("wf").with_metadata("env", "dev");
        let wf2 = Workflow::new("wf").with_metadata("env", "prod");

        let diff = diff_workflows(&wf1, &wf2);
        assert!(!diff.metadata_changes.is_empty());
    }

    #[test]
    fn test_metadata_added() {
        let wf1 = Workflow::new("wf");
        let wf2 = Workflow::new("wf").with_metadata("version", "2.0");

        let diff = diff_workflows(&wf1, &wf2);
        assert_eq!(diff.metadata_changes.len(), 1);
        assert_eq!(diff.metadata_changes[0].field, "version");
    }

    #[test]
    fn test_metadata_removed() {
        let wf1 = Workflow::new("wf").with_metadata("version", "1.0");
        let wf2 = Workflow::new("wf");

        let diff = diff_workflows(&wf1, &wf2);
        assert_eq!(diff.metadata_changes.len(), 1);
    }

    #[test]
    fn test_format_diff_output() {
        let wf1 = make_simple_workflow();
        let mut wf2 = make_simple_workflow();
        wf2.add_task(make_task("qc_check"));

        let diff = diff_workflows(&wf1, &wf2);
        let text = format_diff(&diff);

        assert!(text.contains("Workflow Diff"));
        assert!(text.contains("qc_check"));
        assert!(text.contains("added"));
    }

    #[test]
    fn test_empty_workflows_diff() {
        let wf1 = Workflow::new("empty1");
        let wf2 = Workflow::new("empty2");

        let diff = diff_workflows(&wf1, &wf2);
        assert!(diff.name_changed);
        assert_eq!(diff.summary.tasks_added, 0);
        assert_eq!(diff.summary.tasks_removed, 0);
    }

    #[test]
    fn test_complex_diff_multiple_changes() {
        let mut wf1 = Workflow::new("v1");
        let t1 = wf1.add_task(make_task("ingest"));
        let t2 = wf1.add_task(make_task("transcode"));
        let t3 = wf1.add_task(make_task("old_step"));
        wf1.add_edge(t1, t2).expect("edge");
        wf1.add_edge(t2, t3).expect("edge");

        let mut wf2 = Workflow::new("v2");
        let t1b = wf2.add_task(make_task("ingest").with_priority(TaskPriority::High));
        let t2b = wf2.add_task(make_task("transcode"));
        let t4b = wf2.add_task(make_task("new_step"));
        wf2.add_edge(t1b, t2b).expect("edge");
        wf2.add_edge(t2b, t4b).expect("edge");

        let diff = diff_workflows(&wf1, &wf2);
        assert!(diff.summary.has_changes);
        assert_eq!(diff.summary.tasks_added, 1); // new_step
        assert_eq!(diff.summary.tasks_removed, 1); // old_step
        assert_eq!(diff.summary.tasks_modified, 1); // ingest (priority changed)
        assert_eq!(diff.summary.tasks_unchanged, 1); // transcode
    }

    #[test]
    fn test_task_timeout_change() {
        let mut wf1 = Workflow::new("wf");
        wf1.add_task(make_task("t1"));

        let mut wf2 = Workflow::new("wf");
        let mut t = make_task("t1");
        t.timeout = Duration::from_secs(7200);
        wf2.add_task(t);

        let diff = diff_workflows(&wf1, &wf2);
        assert_eq!(diff.summary.tasks_modified, 1);
        let modified = diff
            .task_changes
            .iter()
            .find(|c| c.change_type == ChangeType::Modified)
            .expect("find modified");
        assert!(modified.field_changes.iter().any(|f| f.field == "timeout"));
    }

    #[test]
    fn test_fail_fast_config_change() {
        let mut wf1 = Workflow::new("wf");
        wf1.config.fail_fast = false;

        let mut wf2 = Workflow::new("wf");
        wf2.config.fail_fast = true;

        let diff = diff_workflows(&wf1, &wf2);
        assert!(diff.config_changes.iter().any(|c| c.field == "fail_fast"));
    }
}
