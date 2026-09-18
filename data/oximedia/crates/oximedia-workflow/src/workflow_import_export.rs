//! Workflow import/export as portable YAML/JSON bundles.
//!
//! Provides serialization and deserialization of complete workflow definitions
//! including tasks, edges, configuration, and metadata into self-contained
//! bundles that can be shared, version-controlled, and imported across
//! different OxiMedia installations.

use crate::dag::{DagError, WorkflowDag, WorkflowEdge, WorkflowNode};
use crate::error::{Result, WorkflowError};
use crate::task::{Task, TaskType};
use crate::workflow::{Workflow, WorkflowConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Format for import/export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleFormat {
    /// JSON format.
    Json,
    /// YAML format.
    Yaml,
    /// Pretty-printed JSON.
    JsonPretty,
}

/// Version of the bundle format.
const BUNDLE_FORMAT_VERSION: &str = "1.0";

/// A portable workflow bundle that can be serialized/deserialized.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowBundle {
    /// Bundle format version.
    pub version: String,
    /// When the bundle was created (ISO 8601 string).
    pub created_at: String,
    /// Human-readable description of the bundle.
    #[serde(default)]
    pub description: String,
    /// The workflow definition.
    pub workflow: WorkflowDefinition,
    /// Optional metadata about the bundle.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    /// Tags for categorization.
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Simplified workflow definition for portable bundles.
///
/// Uses string-based task references instead of UUIDs so the bundle
/// is human-readable and can be hand-edited.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowDefinition {
    /// Workflow name.
    pub name: String,
    /// Workflow description.
    #[serde(default)]
    pub description: String,
    /// Task definitions keyed by task name.
    pub tasks: Vec<TaskDefinition>,
    /// Dependencies as `(from_task_name, to_task_name, optional_condition)`.
    #[serde(default)]
    pub edges: Vec<EdgeDefinition>,
    /// Workflow configuration.
    #[serde(default)]
    pub config: WorkflowConfigDef,
    /// Workflow-level metadata.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// A task definition within a bundle (uses names instead of UUIDs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDefinition {
    /// Unique name within the workflow.
    pub name: String,
    /// Task type and configuration.
    pub task_type: TaskType,
    /// Task priority as string.
    #[serde(default = "default_priority")]
    pub priority: String,
    /// Timeout in seconds.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    /// Task conditions.
    #[serde(default)]
    pub conditions: Vec<String>,
    /// Task metadata.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

fn default_priority() -> String {
    "normal".to_string()
}

fn default_timeout_secs() -> u64 {
    3600
}

/// An edge definition within a bundle (uses task names).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeDefinition {
    /// Source task name.
    pub from: String,
    /// Destination task name.
    pub to: String,
    /// Optional condition expression.
    #[serde(default)]
    pub condition: Option<String>,
}

/// Workflow configuration definition for bundles.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorkflowConfigDef {
    /// Maximum concurrent tasks.
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_tasks: usize,
    /// Whether to stop on first error.
    #[serde(default)]
    pub fail_fast: bool,
    /// Whether to continue on task failure.
    #[serde(default)]
    pub continue_on_error: bool,
    /// Variables.
    #[serde(default)]
    pub variables: HashMap<String, serde_json::Value>,
}

fn default_max_concurrent() -> usize {
    4
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

/// Export a `Workflow` to a bundle.
#[must_use]
pub fn export_workflow(workflow: &Workflow) -> WorkflowBundle {
    // Build name -> TaskId mapping for edge references
    let mut task_defs = Vec::new();
    let mut id_to_name: HashMap<crate::task::TaskId, String> = HashMap::new();

    for task in workflow.tasks.values() {
        id_to_name.insert(task.id, task.name.clone());

        task_defs.push(TaskDefinition {
            name: task.name.clone(),
            task_type: task.task_type.clone(),
            priority: format!("{:?}", task.priority).to_lowercase(),
            timeout_secs: task.timeout.as_secs(),
            conditions: task.conditions.clone(),
            metadata: task.metadata.clone(),
        });
    }

    // Sort task definitions by name for deterministic output
    task_defs.sort_by(|a, b| a.name.cmp(&b.name));

    let edges: Vec<EdgeDefinition> = workflow
        .edges
        .iter()
        .filter_map(|edge| {
            let from_name = id_to_name.get(&edge.from)?.clone();
            let to_name = id_to_name.get(&edge.to)?.clone();
            Some(EdgeDefinition {
                from: from_name,
                to: to_name,
                condition: edge.condition.clone(),
            })
        })
        .collect();

    let config_def = WorkflowConfigDef {
        max_concurrent_tasks: workflow.config.max_concurrent_tasks,
        fail_fast: workflow.config.fail_fast,
        continue_on_error: workflow.config.continue_on_error,
        variables: workflow.config.variables.clone(),
    };

    let now = chrono::Utc::now().to_rfc3339();

    WorkflowBundle {
        version: BUNDLE_FORMAT_VERSION.to_string(),
        created_at: now,
        description: workflow.description.clone(),
        workflow: WorkflowDefinition {
            name: workflow.name.clone(),
            description: workflow.description.clone(),
            tasks: task_defs,
            edges,
            config: config_def,
            metadata: workflow.metadata.clone(),
        },
        metadata: HashMap::new(),
        tags: Vec::new(),
    }
}

/// Serialize a bundle to the specified format.
///
/// # Errors
///
/// Returns an error if serialization fails.
pub fn serialize_bundle(bundle: &WorkflowBundle, format: BundleFormat) -> Result<String> {
    match format {
        BundleFormat::Json => serde_json::to_string(bundle).map_err(WorkflowError::Serialization),
        BundleFormat::JsonPretty => {
            serde_json::to_string_pretty(bundle).map_err(WorkflowError::Serialization)
        }
        BundleFormat::Yaml => serde_yaml::to_string(bundle).map_err(WorkflowError::YamlParsing),
    }
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

/// Deserialize a bundle from a string, auto-detecting format.
///
/// # Errors
///
/// Returns an error if the string is not valid JSON or YAML.
pub fn deserialize_bundle(data: &str) -> Result<WorkflowBundle> {
    let trimmed = data.trim();

    // Try JSON first (starts with '{')
    if trimmed.starts_with('{') {
        return serde_json::from_str(trimmed).map_err(WorkflowError::Serialization);
    }

    // Try YAML
    serde_yaml::from_str(trimmed).map_err(WorkflowError::YamlParsing)
}

/// Import a bundle into a `Workflow`.
///
/// Creates fresh `TaskId` and `WorkflowId` values during import.
///
/// # Errors
///
/// Returns an error if the bundle references unknown tasks in edges
/// or has other structural issues.
pub fn import_workflow(bundle: &WorkflowBundle) -> Result<Workflow> {
    let def = &bundle.workflow;

    let mut workflow = Workflow::new(&def.name).with_description(&def.description);

    // Apply config
    workflow.config = WorkflowConfig {
        max_concurrent_tasks: def.config.max_concurrent_tasks,
        fail_fast: def.config.fail_fast,
        continue_on_error: def.config.continue_on_error,
        variables: def.config.variables.clone(),
        global_timeout: None,
    };

    // Apply metadata
    for (k, v) in &def.metadata {
        workflow.metadata.insert(k.clone(), v.clone());
    }

    // Create tasks and build name -> TaskId mapping
    let mut name_to_id: HashMap<String, crate::task::TaskId> = HashMap::new();

    for task_def in &def.tasks {
        let mut task = Task::new(&task_def.name, task_def.task_type.clone());
        task.timeout = std::time::Duration::from_secs(task_def.timeout_secs);
        task.conditions = task_def.conditions.clone();
        task.metadata = task_def.metadata.clone();

        // Parse priority
        task.priority = match task_def.priority.to_lowercase().as_str() {
            "low" => crate::task::TaskPriority::Low,
            "high" => crate::task::TaskPriority::High,
            "critical" => crate::task::TaskPriority::Critical,
            _ => crate::task::TaskPriority::Normal,
        };

        let task_id = task.id;
        name_to_id.insert(task_def.name.clone(), task_id);
        workflow.add_task(task);
    }

    // Create edges
    for edge_def in &def.edges {
        let from_id = name_to_id
            .get(&edge_def.from)
            .ok_or_else(|| WorkflowError::TaskNotFound(edge_def.from.clone()))?;
        let to_id = name_to_id
            .get(&edge_def.to)
            .ok_or_else(|| WorkflowError::TaskNotFound(edge_def.to.clone()))?;

        if let Some(ref condition) = edge_def.condition {
            workflow.add_conditional_edge(*from_id, *to_id, condition.clone())?;
        } else {
            workflow.add_edge(*from_id, *to_id)?;
        }
    }

    Ok(workflow)
}

// ---------------------------------------------------------------------------
// DAG bundle support
// ---------------------------------------------------------------------------

/// A portable bundle for `WorkflowDag` instances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagBundle {
    /// Bundle format version.
    pub version: String,
    /// Description.
    #[serde(default)]
    pub description: String,
    /// Node definitions keyed by a string identifier.
    pub nodes: Vec<DagNodeDef>,
    /// Edge definitions.
    pub edges: Vec<DagEdgeDef>,
    /// Metadata.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Node definition for DAG bundles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagNodeDef {
    /// Unique name within the bundle.
    pub name: String,
    /// Task type.
    pub task_type: String,
    /// Parameters.
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Edge definition for DAG bundles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagEdgeDef {
    /// Source node name.
    pub from: String,
    /// Destination node name.
    pub to: String,
    /// Data type.
    pub data_type: String,
    /// Condition.
    #[serde(default)]
    pub condition: Option<String>,
}

/// Export a `WorkflowDag` to a `DagBundle`.
#[must_use]
pub fn export_dag(dag: &WorkflowDag, description: &str) -> DagBundle {
    use crate::dag::NodeId;

    // Assign names to nodes (use task_type + index if names collide)
    let mut node_names: HashMap<NodeId, String> = HashMap::new();
    let mut used_names: HashMap<String, usize> = HashMap::new();

    for (node_id, node) in &dag.nodes {
        let base_name = node.task_type.clone();
        let count = used_names.entry(base_name.clone()).or_insert(0);
        let name = if *count == 0 {
            base_name.clone()
        } else {
            format!("{base_name}_{count}")
        };
        *count += 1;
        node_names.insert(*node_id, name);
    }

    let nodes: Vec<DagNodeDef> = dag
        .nodes
        .iter()
        .map(|(id, node)| DagNodeDef {
            name: node_names.get(id).cloned().unwrap_or_default(),
            task_type: node.task_type.clone(),
            parameters: node.parameters.clone(),
        })
        .collect();

    let edges: Vec<DagEdgeDef> = dag
        .edges
        .iter()
        .filter_map(|edge| {
            let from = node_names.get(&edge.from_node)?.clone();
            let to = node_names.get(&edge.to_node)?.clone();
            Some(DagEdgeDef {
                from,
                to,
                data_type: edge.data_type.clone(),
                condition: edge.condition.clone(),
            })
        })
        .collect();

    DagBundle {
        version: BUNDLE_FORMAT_VERSION.to_string(),
        description: description.to_string(),
        nodes,
        edges,
        metadata: HashMap::new(),
    }
}

/// Import a `DagBundle` into a `WorkflowDag`.
///
/// # Errors
///
/// Returns `DagError` if nodes are referenced in edges but not defined,
/// or if the resulting DAG would contain a cycle.
pub fn import_dag(bundle: &DagBundle) -> std::result::Result<WorkflowDag, DagError> {
    let mut dag = WorkflowDag::new();
    let mut name_to_id: HashMap<String, crate::dag::NodeId> = HashMap::new();

    for node_def in &bundle.nodes {
        let mut node = WorkflowNode::new(&node_def.task_type);
        node.parameters = node_def.parameters.clone();
        let id = dag.add_node(node)?;
        name_to_id.insert(node_def.name.clone(), id);
    }

    for edge_def in &bundle.edges {
        let from_id = name_to_id
            .get(&edge_def.from)
            .ok_or_else(|| DagError::NodeNotFound(crate::dag::NodeId::new()))?;
        let to_id = name_to_id
            .get(&edge_def.to)
            .ok_or_else(|| DagError::NodeNotFound(crate::dag::NodeId::new()))?;

        let edge = if let Some(ref condition) = edge_def.condition {
            WorkflowEdge::with_condition(*from_id, *to_id, &edge_def.data_type, condition)
        } else {
            WorkflowEdge::new(*from_id, *to_id, &edge_def.data_type)
        };
        dag.add_edge(edge)?;
    }

    Ok(dag)
}

/// Serialize a DAG bundle.
///
/// # Errors
///
/// Returns an error if serialization fails.
pub fn serialize_dag_bundle(bundle: &DagBundle, format: BundleFormat) -> Result<String> {
    match format {
        BundleFormat::Json => serde_json::to_string(bundle).map_err(WorkflowError::Serialization),
        BundleFormat::JsonPretty => {
            serde_json::to_string_pretty(bundle).map_err(WorkflowError::Serialization)
        }
        BundleFormat::Yaml => serde_yaml::to_string(bundle).map_err(WorkflowError::YamlParsing),
    }
}

/// Deserialize a DAG bundle from a string.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn deserialize_dag_bundle(data: &str) -> Result<DagBundle> {
    let trimmed = data.trim();
    if trimmed.starts_with('{') {
        serde_json::from_str(trimmed).map_err(WorkflowError::Serialization)
    } else {
        serde_yaml::from_str(trimmed).map_err(WorkflowError::YamlParsing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::{TaskPriority, TaskType};
    use std::time::Duration;

    fn make_test_workflow() -> Workflow {
        let mut workflow = Workflow::new("test-pipeline")
            .with_description("A test pipeline")
            .with_metadata("env", "test");

        let task1 = Task::new(
            "ingest",
            TaskType::Wait {
                duration: Duration::from_secs(5),
            },
        )
        .with_priority(TaskPriority::High);
        let task2 = Task::new(
            "transcode",
            TaskType::Wait {
                duration: Duration::from_secs(60),
            },
        );
        let task3 = Task::new(
            "deliver",
            TaskType::Wait {
                duration: Duration::from_secs(10),
            },
        )
        .with_priority(TaskPriority::Low);

        let id1 = workflow.add_task(task1);
        let id2 = workflow.add_task(task2);
        let id3 = workflow.add_task(task3);

        workflow.add_edge(id1, id2).expect("add edge");
        workflow.add_edge(id2, id3).expect("add edge");

        workflow
    }

    // --- Export ---

    #[test]
    fn test_export_workflow() {
        let workflow = make_test_workflow();
        let bundle = export_workflow(&workflow);

        assert_eq!(bundle.version, "1.0");
        assert_eq!(bundle.workflow.name, "test-pipeline");
        assert_eq!(bundle.workflow.tasks.len(), 3);
        assert_eq!(bundle.workflow.edges.len(), 2);
    }

    #[test]
    fn test_export_preserves_metadata() {
        let workflow = make_test_workflow();
        let bundle = export_workflow(&workflow);

        assert_eq!(
            bundle.workflow.metadata.get("env"),
            Some(&"test".to_string())
        );
    }

    #[test]
    fn test_export_edges_use_names() {
        let workflow = make_test_workflow();
        let bundle = export_workflow(&workflow);

        // Edges should reference task names, not UUIDs
        let edge_names: Vec<(&str, &str)> = bundle
            .workflow
            .edges
            .iter()
            .map(|e| (e.from.as_str(), e.to.as_str()))
            .collect();
        assert!(edge_names.contains(&("ingest", "transcode")));
        assert!(edge_names.contains(&("transcode", "deliver")));
    }

    // --- Serialize/Deserialize ---

    #[test]
    fn test_serialize_json() {
        let workflow = make_test_workflow();
        let bundle = export_workflow(&workflow);
        let json = serialize_bundle(&bundle, BundleFormat::Json).expect("serialize");

        assert!(json.contains("test-pipeline"));
        assert!(!json.contains('\n')); // compact JSON
    }

    #[test]
    fn test_serialize_json_pretty() {
        let workflow = make_test_workflow();
        let bundle = export_workflow(&workflow);
        let json = serialize_bundle(&bundle, BundleFormat::JsonPretty).expect("serialize");

        assert!(json.contains("test-pipeline"));
        assert!(json.contains('\n')); // pretty-printed
    }

    #[test]
    fn test_serialize_yaml() {
        let workflow = make_test_workflow();
        let bundle = export_workflow(&workflow);
        let yaml = serialize_bundle(&bundle, BundleFormat::Yaml).expect("serialize");

        assert!(yaml.contains("test-pipeline"));
        assert!(yaml.contains("name:"));
    }

    #[test]
    fn test_deserialize_json() {
        let workflow = make_test_workflow();
        let bundle = export_workflow(&workflow);
        let json = serialize_bundle(&bundle, BundleFormat::Json).expect("serialize");

        let restored = deserialize_bundle(&json).expect("deserialize");
        assert_eq!(restored.workflow.name, "test-pipeline");
        assert_eq!(restored.workflow.tasks.len(), 3);
    }

    #[test]
    fn test_deserialize_yaml() {
        let workflow = make_test_workflow();
        let bundle = export_workflow(&workflow);
        let yaml = serialize_bundle(&bundle, BundleFormat::Yaml).expect("serialize");

        let restored = deserialize_bundle(&yaml).expect("deserialize");
        assert_eq!(restored.workflow.name, "test-pipeline");
    }

    // --- Import ---

    #[test]
    fn test_import_workflow() {
        let workflow = make_test_workflow();
        let bundle = export_workflow(&workflow);
        let imported = import_workflow(&bundle).expect("import");

        assert_eq!(imported.name, "test-pipeline");
        assert_eq!(imported.tasks.len(), 3);
        assert_eq!(imported.edges.len(), 2);
    }

    #[test]
    fn test_import_preserves_config() {
        let mut workflow = make_test_workflow();
        workflow.config.fail_fast = true;
        workflow.config.max_concurrent_tasks = 8;

        let bundle = export_workflow(&workflow);
        let imported = import_workflow(&bundle).expect("import");

        assert!(imported.config.fail_fast);
        assert_eq!(imported.config.max_concurrent_tasks, 8);
    }

    #[test]
    fn test_import_preserves_priority() {
        let workflow = make_test_workflow();
        let bundle = export_workflow(&workflow);
        let imported = import_workflow(&bundle).expect("import");

        let ingest_task = imported
            .tasks
            .values()
            .find(|t| t.name == "ingest")
            .expect("find task");
        assert_eq!(ingest_task.priority, TaskPriority::High);
    }

    #[test]
    fn test_roundtrip_json() {
        let workflow = make_test_workflow();
        let bundle = export_workflow(&workflow);
        let json = serialize_bundle(&bundle, BundleFormat::Json).expect("serialize");
        let restored = deserialize_bundle(&json).expect("deserialize");
        let imported = import_workflow(&restored).expect("import");

        assert_eq!(imported.name, workflow.name);
        assert_eq!(imported.tasks.len(), workflow.tasks.len());
        assert_eq!(imported.edges.len(), workflow.edges.len());
    }

    #[test]
    fn test_roundtrip_yaml() {
        let workflow = make_test_workflow();
        let bundle = export_workflow(&workflow);
        let yaml = serialize_bundle(&bundle, BundleFormat::Yaml).expect("serialize");
        let restored = deserialize_bundle(&yaml).expect("deserialize");
        let imported = import_workflow(&restored).expect("import");

        assert_eq!(imported.name, workflow.name);
        assert_eq!(imported.tasks.len(), workflow.tasks.len());
    }

    #[test]
    fn test_import_bad_edge_reference() {
        let mut bundle = export_workflow(&make_test_workflow());
        bundle.workflow.edges.push(EdgeDefinition {
            from: "nonexistent".to_string(),
            to: "ingest".to_string(),
            condition: None,
        });

        let result = import_workflow(&bundle);
        assert!(result.is_err());
    }

    // --- Conditional edges ---

    #[test]
    fn test_conditional_edge_roundtrip() {
        let mut workflow = Workflow::new("conditional-wf");
        let task1 = Task::new(
            "check",
            TaskType::Wait {
                duration: Duration::from_secs(1),
            },
        );
        let task2 = Task::new(
            "branch_a",
            TaskType::Wait {
                duration: Duration::from_secs(1),
            },
        );
        let id1 = workflow.add_task(task1);
        let id2 = workflow.add_task(task2);
        workflow
            .add_conditional_edge(id1, id2, "result == success".to_string())
            .expect("add edge");

        let bundle = export_workflow(&workflow);
        assert_eq!(
            bundle.workflow.edges[0].condition,
            Some("result == success".to_string())
        );

        let imported = import_workflow(&bundle).expect("import");
        assert_eq!(
            imported.edges[0].condition,
            Some("result == success".to_string())
        );
    }

    // --- DAG bundles ---

    #[test]
    fn test_dag_export() {
        let mut dag = WorkflowDag::new();
        let a = dag.add_node(WorkflowNode::new("ingest")).expect("add node");
        let b = dag
            .add_node(WorkflowNode::new("transcode"))
            .expect("add node");
        dag.add_edge(WorkflowEdge::new(a, b, "raw_media"))
            .expect("add edge");

        let bundle = export_dag(&dag, "test dag");
        assert_eq!(bundle.nodes.len(), 2);
        assert_eq!(bundle.edges.len(), 1);
        assert_eq!(bundle.description, "test dag");
    }

    #[test]
    fn test_dag_roundtrip() {
        let mut dag = WorkflowDag::new();
        let a = dag.add_node(WorkflowNode::new("ingest")).expect("add node");
        let b = dag.add_node(WorkflowNode::new("encode")).expect("add node");
        let c = dag
            .add_node(WorkflowNode::new("deliver"))
            .expect("add node");
        dag.add_edge(WorkflowEdge::new(a, b, "raw")).expect("edge");
        dag.add_edge(WorkflowEdge::new(b, c, "encoded"))
            .expect("edge");

        let bundle = export_dag(&dag, "pipeline");
        let json = serialize_dag_bundle(&bundle, BundleFormat::JsonPretty).expect("serialize");
        let restored_bundle = deserialize_dag_bundle(&json).expect("deserialize");
        let restored_dag = import_dag(&restored_bundle).expect("import");

        assert_eq!(restored_dag.nodes.len(), 3);
        assert_eq!(restored_dag.edges.len(), 2);
        assert!(!restored_dag.has_cycle());
    }

    #[test]
    fn test_dag_yaml_roundtrip() {
        let mut dag = WorkflowDag::new();
        let a = dag.add_node(WorkflowNode::new("src")).expect("add node");
        let b = dag.add_node(WorkflowNode::new("dst")).expect("add node");
        dag.add_edge(WorkflowEdge::new(a, b, "data")).expect("edge");

        let bundle = export_dag(&dag, "simple");
        let yaml = serialize_dag_bundle(&bundle, BundleFormat::Yaml).expect("serialize");
        let restored = deserialize_dag_bundle(&yaml).expect("deserialize");
        let imported = import_dag(&restored).expect("import");

        assert_eq!(imported.nodes.len(), 2);
    }

    #[test]
    fn test_dag_with_parameters() {
        let mut dag = WorkflowDag::new();
        let node = WorkflowNode::new("encode")
            .with_parameter("preset", serde_json::json!("slow"))
            .with_parameter("crf", serde_json::json!(23));
        let _ = dag.add_node(node).expect("add node");

        let bundle = export_dag(&dag, "params test");
        let json = serialize_dag_bundle(&bundle, BundleFormat::Json).expect("serialize");
        let restored = deserialize_dag_bundle(&json).expect("deserialize");
        let imported = import_dag(&restored).expect("import");

        let imported_node = imported.nodes.values().next().expect("get node");
        assert_eq!(
            imported_node.parameters["preset"],
            serde_json::json!("slow")
        );
        assert_eq!(imported_node.parameters["crf"], serde_json::json!(23));
    }

    #[test]
    fn test_bundle_format_version() {
        let workflow = make_test_workflow();
        let bundle = export_workflow(&workflow);
        assert_eq!(bundle.version, "1.0");
    }

    #[test]
    fn test_bundle_tags() {
        let workflow = make_test_workflow();
        let mut bundle = export_workflow(&workflow);
        bundle.tags = vec!["media".to_string(), "transcode".to_string()];

        let json = serialize_bundle(&bundle, BundleFormat::Json).expect("serialize");
        let restored = deserialize_bundle(&json).expect("deserialize");
        assert_eq!(restored.tags, vec!["media", "transcode"]);
    }
}
