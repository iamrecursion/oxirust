//! REST API for workflow management.

use crate::error::{Result, WorkflowError};
use crate::executor::{TaskExecutor, WorkflowExecutor};
use crate::monitoring::MonitoringService;
use crate::persistence::PersistenceManager;
use crate::scheduler::{Trigger, WorkflowScheduler};
use crate::task::{Task, TaskId};
use crate::workflow::{Workflow, WorkflowId};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// API server state.
#[derive(Clone)]
pub struct ApiState {
    /// Persistence manager.
    pub persistence: Arc<PersistenceManager>,
    /// Workflow scheduler.
    pub scheduler: Arc<WorkflowScheduler>,
    /// Monitoring service.
    pub monitoring: Arc<MonitoringService>,
    /// Task executor.
    pub executor: Arc<dyn TaskExecutor>,
    /// Active workflow executors.
    pub active_workflows: Arc<RwLock<std::collections::HashMap<WorkflowId, Arc<WorkflowExecutor>>>>,
}

/// Create API router.
pub fn create_router(state: ApiState) -> Router {
    Router::new()
        .route(
            "/api/v1/workflows",
            get(list_workflows).post(create_workflow),
        )
        .route(
            "/api/v1/workflows/:id",
            get(get_workflow)
                .put(update_workflow)
                .delete(delete_workflow),
        )
        .route("/api/v1/workflows/:id/execute", post(execute_workflow))
        .route("/api/v1/workflows/:id/cancel", post(cancel_workflow))
        .route("/api/v1/workflows/:id/tasks", get(list_tasks))
        .route("/api/v1/workflows/:id/tasks/:task_id", get(get_task))
        .route("/api/v1/workflows/:id/status", get(get_workflow_status))
        .route(
            "/api/v1/schedules",
            get(list_schedules).post(create_schedule),
        )
        .route(
            "/api/v1/schedules/:id",
            get(get_schedule).delete(delete_schedule),
        )
        .route("/api/v1/schedules/:id/enable", post(enable_schedule))
        .route("/api/v1/schedules/:id/disable", post(disable_schedule))
        .route("/api/v1/monitoring/workflows", get(get_active_workflows))
        .route("/api/v1/monitoring/statistics", get(get_statistics))
        .route("/api/v1/monitoring/history", get(get_history))
        .route("/api/v1/health", get(health_check))
        .with_state(state)
}

/// List all workflows.
async fn list_workflows(State(state): State<ApiState>) -> Result<Json<Vec<String>>> {
    let workflows = state.persistence.list_workflows()?;
    Ok(Json(
        workflows
            .iter()
            .map(std::string::ToString::to_string)
            .collect(),
    ))
}

/// Create a new workflow.
async fn create_workflow(
    State(state): State<ApiState>,
    Json(workflow): Json<Workflow>,
) -> Result<Json<CreateWorkflowResponse>> {
    state.persistence.save_workflow(&workflow)?;
    info!("Created workflow: {}", workflow.id);

    Ok(Json(CreateWorkflowResponse {
        workflow_id: workflow.id,
        message: "Workflow created successfully".to_string(),
    }))
}

/// Get workflow by ID.
async fn get_workflow(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<Workflow>> {
    let workflow_id = parse_workflow_id(&id)?;
    let workflow = state.persistence.load_workflow(workflow_id)?;
    Ok(Json(workflow))
}

/// Update workflow.
async fn update_workflow(
    State(state): State<ApiState>,
    Path(id): Path<String>,
    Json(workflow): Json<Workflow>,
) -> Result<Json<UpdateWorkflowResponse>> {
    let workflow_id = parse_workflow_id(&id)?;

    // Verify workflow exists
    state.persistence.load_workflow(workflow_id)?;

    state.persistence.save_workflow(&workflow)?;
    info!("Updated workflow: {}", workflow_id);

    Ok(Json(UpdateWorkflowResponse {
        workflow_id,
        message: "Workflow updated successfully".to_string(),
    }))
}

/// Delete workflow.
async fn delete_workflow(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<DeleteWorkflowResponse>> {
    let workflow_id = parse_workflow_id(&id)?;
    state.persistence.delete_workflow(workflow_id)?;
    info!("Deleted workflow: {}", workflow_id);

    Ok(Json(DeleteWorkflowResponse {
        workflow_id,
        message: "Workflow deleted successfully".to_string(),
    }))
}

/// Execute workflow.
async fn execute_workflow(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<ExecuteWorkflowResponse>> {
    let workflow_id = parse_workflow_id(&id)?;
    let mut workflow = state.persistence.load_workflow(workflow_id)?;

    // Create executor
    let executor = Arc::new(WorkflowExecutor::new(state.executor.clone()));

    // Store in active workflows
    state
        .active_workflows
        .write()
        .await
        .insert(workflow_id, executor.clone());

    // Start monitoring
    state
        .monitoring
        .start_workflow(workflow_id, workflow.name.clone(), workflow.tasks.len());

    // Execute in background
    let persistence = state.persistence.clone();
    let monitoring = state.monitoring.clone();
    let active = state.active_workflows.clone();

    tokio::spawn(async move {
        let result = executor.execute(&mut workflow).await;

        match result {
            Ok(exec_result) => {
                monitoring.complete_workflow(
                    workflow_id,
                    exec_result.state == crate::workflow::WorkflowState::Completed,
                );
                let _ = persistence.save_workflow(&workflow);
            }
            Err(e) => {
                monitoring.complete_workflow(workflow_id, false);
                tracing::error!("Workflow execution failed: {}", e);
            }
        }

        active.write().await.remove(&workflow_id);
    });

    Ok(Json(ExecuteWorkflowResponse {
        workflow_id,
        message: "Workflow execution started".to_string(),
    }))
}

/// Cancel workflow execution.
async fn cancel_workflow(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<CancelWorkflowResponse>> {
    let workflow_id = parse_workflow_id(&id)?;

    // Remove from active workflows
    let removed = state
        .active_workflows
        .write()
        .await
        .remove(&workflow_id)
        .is_some();

    if removed {
        state.monitoring.complete_workflow(workflow_id, false);
        Ok(Json(CancelWorkflowResponse {
            workflow_id,
            message: "Workflow cancelled".to_string(),
        }))
    } else {
        Err(WorkflowError::NotRunning(workflow_id.to_string()))
    }
}

/// List tasks in workflow.
async fn list_tasks(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<Task>>> {
    let workflow_id = parse_workflow_id(&id)?;
    let workflow = state.persistence.load_workflow(workflow_id)?;
    let tasks: Vec<_> = workflow.tasks.values().cloned().collect();
    Ok(Json(tasks))
}

/// Get specific task.
async fn get_task(
    State(state): State<ApiState>,
    Path((id, task_id)): Path<(String, String)>,
) -> Result<Json<Task>> {
    let workflow_id = parse_workflow_id(&id)?;
    let task_id = parse_task_id(&task_id)?;

    let workflow = state.persistence.load_workflow(workflow_id)?;
    let task = workflow
        .get_task(&task_id)
        .ok_or_else(|| WorkflowError::TaskNotFound(task_id.to_string()))?;

    Ok(Json(task.clone()))
}

/// Get workflow status and metrics.
async fn get_workflow_status(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<crate::monitoring::WorkflowMetrics>> {
    let workflow_id = parse_workflow_id(&id)?;
    let metrics = state
        .monitoring
        .get_workflow_metrics(&workflow_id)
        .ok_or_else(|| WorkflowError::WorkflowNotFound(workflow_id.to_string()))?;

    Ok(Json(metrics))
}

/// List all schedules.
async fn list_schedules(State(state): State<ApiState>) -> Result<Json<Vec<ScheduleInfo>>> {
    let schedules = state.scheduler.list_schedules().await;
    let schedule_infos: Vec<_> = schedules
        .into_iter()
        .map(|(id, sched)| ScheduleInfo {
            workflow_id: id,
            workflow_name: sched.workflow.name,
            trigger: sched.trigger,
            enabled: sched.enabled,
            next_execution: sched.next_execution,
        })
        .collect();

    Ok(Json(schedule_infos))
}

/// Create a new schedule.
async fn create_schedule(
    State(state): State<ApiState>,
    Json(req): Json<CreateScheduleRequest>,
) -> Result<Json<CreateScheduleResponse>> {
    let workflow = state.persistence.load_workflow(req.workflow_id)?;
    let workflow_id = state.scheduler.add_schedule(workflow, req.trigger).await?;

    Ok(Json(CreateScheduleResponse {
        workflow_id,
        message: "Schedule created successfully".to_string(),
    }))
}

/// Get schedule details.
async fn get_schedule(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<ScheduleInfo>> {
    let workflow_id = parse_workflow_id(&id)?;
    let schedules = state.scheduler.list_schedules().await;

    let schedule = schedules
        .into_iter()
        .find(|(id, _)| *id == workflow_id)
        .map(|(id, sched)| ScheduleInfo {
            workflow_id: id,
            workflow_name: sched.workflow.name,
            trigger: sched.trigger,
            enabled: sched.enabled,
            next_execution: sched.next_execution,
        })
        .ok_or_else(|| WorkflowError::WorkflowNotFound(workflow_id.to_string()))?;

    Ok(Json(schedule))
}

/// Delete a schedule.
async fn delete_schedule(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<DeleteScheduleResponse>> {
    let workflow_id = parse_workflow_id(&id)?;
    state.scheduler.remove_schedule(workflow_id).await?;

    Ok(Json(DeleteScheduleResponse {
        workflow_id,
        message: "Schedule deleted successfully".to_string(),
    }))
}

/// Enable a schedule.
async fn enable_schedule(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<EnableScheduleResponse>> {
    let workflow_id = parse_workflow_id(&id)?;
    state
        .scheduler
        .set_schedule_enabled(workflow_id, true)
        .await?;

    Ok(Json(EnableScheduleResponse {
        workflow_id,
        enabled: true,
    }))
}

/// Disable a schedule.
async fn disable_schedule(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<EnableScheduleResponse>> {
    let workflow_id = parse_workflow_id(&id)?;
    state
        .scheduler
        .set_schedule_enabled(workflow_id, false)
        .await?;

    Ok(Json(EnableScheduleResponse {
        workflow_id,
        enabled: false,
    }))
}

/// Get active workflows.
async fn get_active_workflows(
    State(state): State<ApiState>,
) -> Result<Json<Vec<crate::monitoring::WorkflowMetrics>>> {
    let workflows = state.monitoring.get_active_workflows();
    Ok(Json(workflows))
}

/// Get system statistics.
async fn get_statistics(
    State(state): State<ApiState>,
) -> Result<Json<crate::monitoring::SystemStatistics>> {
    let stats = state.monitoring.get_statistics();
    Ok(Json(stats))
}

/// Get workflow history.
async fn get_history(
    State(state): State<ApiState>,
) -> Result<Json<Vec<crate::monitoring::WorkflowMetrics>>> {
    let history = state.monitoring.get_history(Some(100));
    Ok(Json(history))
}

/// Health check endpoint.
async fn health_check() -> Json<HealthCheckResponse> {
    Json(HealthCheckResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

// Helper functions

fn parse_workflow_id(id_str: &str) -> Result<WorkflowId> {
    let uuid = uuid::Uuid::parse_str(id_str).map_err(|_e| WorkflowError::InvalidParameter {
        param: "workflow_id".to_string(),
        value: id_str.to_string(),
    })?;
    Ok(WorkflowId::from(uuid))
}

fn parse_task_id(id_str: &str) -> Result<TaskId> {
    let uuid = uuid::Uuid::parse_str(id_str).map_err(|_e| WorkflowError::InvalidParameter {
        param: "task_id".to_string(),
        value: id_str.to_string(),
    })?;
    Ok(TaskId::from(uuid))
}

// Request/Response types

/// Response from creating a workflow.
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateWorkflowResponse {
    /// The ID of the created workflow.
    pub workflow_id: WorkflowId,
    /// Message from the operation.
    pub message: String,
}

/// Response from updating a workflow.
#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateWorkflowResponse {
    /// The ID of the updated workflow.
    pub workflow_id: WorkflowId,
    /// Message from the operation.
    pub message: String,
}

/// Response from deleting a workflow.
#[derive(Debug, Serialize, Deserialize)]
pub struct DeleteWorkflowResponse {
    /// The ID of the deleted workflow.
    pub workflow_id: WorkflowId,
    /// Message from the operation.
    pub message: String,
}

/// Response from executing a workflow.
#[derive(Debug, Serialize, Deserialize)]
pub struct ExecuteWorkflowResponse {
    /// The ID of the executed workflow.
    pub workflow_id: WorkflowId,
    /// Message from the operation.
    pub message: String,
}

/// Response from canceling a workflow.
#[derive(Debug, Serialize, Deserialize)]
pub struct CancelWorkflowResponse {
    /// The ID of the canceled workflow.
    pub workflow_id: WorkflowId,
    /// Message from the operation.
    pub message: String,
}

/// Request to create a schedule.
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateScheduleRequest {
    /// The workflow ID to schedule.
    pub workflow_id: WorkflowId,
    /// The trigger for the schedule.
    pub trigger: Trigger,
}

/// Response from creating a schedule.
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateScheduleResponse {
    /// The workflow ID of the schedule.
    pub workflow_id: WorkflowId,
    /// Message from the operation.
    pub message: String,
}

/// Response from deleting a schedule.
#[derive(Debug, Serialize, Deserialize)]
pub struct DeleteScheduleResponse {
    /// The workflow ID of the deleted schedule.
    pub workflow_id: WorkflowId,
    /// Message from the operation.
    pub message: String,
}

/// Response from enabling/disabling a schedule.
#[derive(Debug, Serialize, Deserialize)]
pub struct EnableScheduleResponse {
    /// The workflow ID of the schedule.
    pub workflow_id: WorkflowId,
    /// Whether the schedule is enabled.
    pub enabled: bool,
}

/// Information about a schedule.
#[derive(Debug, Serialize, Deserialize)]
pub struct ScheduleInfo {
    /// The workflow ID.
    pub workflow_id: WorkflowId,
    /// The workflow name.
    pub workflow_name: String,
    /// The trigger for the schedule.
    pub trigger: Trigger,
    /// Whether the schedule is enabled.
    pub enabled: bool,
    /// The next execution time.
    pub next_execution: Option<chrono::DateTime<chrono::Utc>>,
}

/// Health check response.
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthCheckResponse {
    /// The health status.
    pub status: String,
    /// The version information.
    pub version: String,
}

// Error conversion

impl IntoResponse for WorkflowError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            WorkflowError::WorkflowNotFound(_) | WorkflowError::TaskNotFound(_) => {
                (StatusCode::NOT_FOUND, self.to_string())
            }
            WorkflowError::InvalidConfiguration(_)
            | WorkflowError::InvalidParameter { .. }
            | WorkflowError::CycleDetected => (StatusCode::BAD_REQUEST, self.to_string()),
            WorkflowError::AlreadyRunning(_) => (StatusCode::CONFLICT, self.to_string()),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };

        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_workflow_id() {
        let uuid = uuid::Uuid::new_v4();
        let id_str = uuid.to_string();
        let result = parse_workflow_id(&id_str);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_invalid_workflow_id() {
        let result = parse_workflow_id("invalid-id");
        assert!(result.is_err());
    }
}
