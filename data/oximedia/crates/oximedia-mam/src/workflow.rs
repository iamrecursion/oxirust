//! Workflow engine implementation
//!
//! Provides workflow capabilities:
//! - State machine for workflow execution
//! - Approval workflows
//! - Review and comment system
//! - Task assignment
//! - Email notifications
//! - Webhook integration

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::database::Database;
use crate::{MamConfig, MamError, Result};

/// Workflow engine
pub struct WorkflowEngine {
    db: Arc<Database>,
    #[allow(dead_code)]
    config: MamConfig,
    event_tx: mpsc::UnboundedSender<WorkflowEvent>,
    #[allow(dead_code)]
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

/// Workflow definition
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Workflow {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub workflow_type: String,
    pub config: serde_json::Value,
    pub is_active: bool,
    pub created_by: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Workflow instance (execution)
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct WorkflowInstance {
    pub id: Uuid,
    pub workflow_id: Uuid,
    pub asset_id: Uuid,
    pub status: String,
    pub current_state: Option<String>,
    pub state_data: Option<serde_json::Value>,
    pub started_by: Option<Uuid>,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
}

/// Workflow task
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct WorkflowTask {
    pub id: Uuid,
    pub instance_id: Uuid,
    pub task_type: String,
    pub assigned_to: Option<Uuid>,
    pub status: String,
    pub due_date: Option<DateTime<Utc>>,
    pub completed_by: Option<Uuid>,
    pub completed_at: Option<DateTime<Utc>>,
    pub comment: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// Workflow configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowConfig {
    pub states: Vec<WorkflowState>,
    pub transitions: Vec<WorkflowTransition>,
    pub notifications: NotificationConfig,
}

/// Workflow state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowState {
    pub name: String,
    pub description: Option<String>,
    pub is_initial: bool,
    pub is_final: bool,
    pub required_role: Option<String>,
    pub tasks: Vec<TaskDefinition>,
}

/// Workflow transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowTransition {
    pub from_state: String,
    pub to_state: String,
    pub action: String,
    pub condition: Option<String>,
    pub required_role: Option<String>,
}

/// Task definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDefinition {
    pub task_type: String,
    pub name: String,
    pub description: Option<String>,
    pub required: bool,
    pub auto_assign: Option<String>,
}

/// Notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    pub email_enabled: bool,
    pub webhook_enabled: bool,
    pub webhook_url: Option<String>,
    pub notify_on_state_change: bool,
    pub notify_on_task_assigned: bool,
    pub notify_on_completion: bool,
}

/// Workflow event
#[derive(Debug, Clone)]
pub enum WorkflowEvent {
    InstanceStarted {
        instance_id: Uuid,
        workflow_id: Uuid,
        asset_id: Uuid,
    },
    StateChanged {
        instance_id: Uuid,
        from_state: String,
        to_state: String,
    },
    TaskCreated {
        task_id: Uuid,
        instance_id: Uuid,
        assigned_to: Option<Uuid>,
    },
    TaskCompleted {
        task_id: Uuid,
        instance_id: Uuid,
        completed_by: Uuid,
    },
    InstanceCompleted {
        instance_id: Uuid,
    },
    InstanceFailed {
        instance_id: Uuid,
        error: String,
    },
}

/// Create workflow request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateWorkflowRequest {
    pub name: String,
    pub description: Option<String>,
    pub workflow_type: String,
    pub config: WorkflowConfig,
    pub created_by: Option<Uuid>,
}

/// Start workflow request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartWorkflowRequest {
    pub workflow_id: Uuid,
    pub asset_id: Uuid,
    pub started_by: Option<Uuid>,
    pub initial_data: Option<HashMap<String, serde_json::Value>>,
}

/// Transition workflow request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionRequest {
    pub instance_id: Uuid,
    pub action: String,
    pub user_id: Uuid,
    pub comment: Option<String>,
}

/// Complete task request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompleteTaskRequest {
    pub task_id: Uuid,
    pub user_id: Uuid,
    pub comment: Option<String>,
}

impl WorkflowEngine {
    /// Create a new workflow engine
    #[must_use]
    pub fn new(db: Arc<Database>, config: MamConfig) -> Self {
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

        // Spawn event processor
        let processor_db = Arc::clone(&db);
        let processor_config = config.clone();

        tokio::spawn(async move {
            Self::event_processor_loop(event_rx, shutdown_rx, processor_db, processor_config).await;
        });

        Self {
            db,
            config,
            event_tx,
            shutdown_tx: Some(shutdown_tx),
        }
    }

    /// Create a new workflow
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails
    pub async fn create_workflow(&self, req: CreateWorkflowRequest) -> Result<Workflow> {
        let config_json = serde_json::to_value(req.config)?;

        let workflow = sqlx::query_as::<_, Workflow>(
            "INSERT INTO workflows (id, name, description, workflow_type, config, is_active, created_by, created_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, true, $6, NOW(), NOW())
             RETURNING *"
        )
        .bind(Uuid::new_v4())
        .bind(&req.name)
        .bind(req.description)
        .bind(&req.workflow_type)
        .bind(config_json)
        .bind(req.created_by)
        .fetch_one(self.db.pool())
        .await?;

        Ok(workflow)
    }

    /// Get workflow by ID
    ///
    /// # Errors
    ///
    /// Returns an error if the workflow is not found
    pub async fn get_workflow(&self, workflow_id: Uuid) -> Result<Workflow> {
        let workflow = sqlx::query_as::<_, Workflow>("SELECT * FROM workflows WHERE id = $1")
            .bind(workflow_id)
            .fetch_one(self.db.pool())
            .await
            .map_err(|_| MamError::WorkflowNotFound(workflow_id))?;

        Ok(workflow)
    }

    /// List workflows
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn list_workflows(&self, limit: i64, offset: i64) -> Result<Vec<Workflow>> {
        let workflows = sqlx::query_as::<_, Workflow>(
            "SELECT * FROM workflows WHERE is_active = true ORDER BY created_at DESC LIMIT $1 OFFSET $2"
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(self.db.pool())
        .await?;

        Ok(workflows)
    }

    /// Start a workflow instance
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails or workflow is not found
    pub async fn start_workflow(&self, req: StartWorkflowRequest) -> Result<WorkflowInstance> {
        // Get workflow definition
        let workflow = self.get_workflow(req.workflow_id).await?;

        // Parse workflow config
        let config: WorkflowConfig = serde_json::from_value(workflow.config)?;

        // Find initial state
        let initial_state = config
            .states
            .iter()
            .find(|s| s.is_initial)
            .ok_or_else(|| MamError::Internal("No initial state defined".to_string()))?;

        // Create instance
        let state_data = req.initial_data.and_then(|d| serde_json::to_value(d).ok());

        let instance = sqlx::query_as::<_, WorkflowInstance>(
            "INSERT INTO workflow_instances
             (id, workflow_id, asset_id, status, current_state, state_data, started_by, started_at)
             VALUES ($1, $2, $3, 'running', $4, $5, $6, NOW())
             RETURNING *",
        )
        .bind(Uuid::new_v4())
        .bind(req.workflow_id)
        .bind(req.asset_id)
        .bind(&initial_state.name)
        .bind(state_data)
        .bind(req.started_by)
        .fetch_one(self.db.pool())
        .await?;

        // Create initial tasks
        for task_def in &initial_state.tasks {
            self.create_task(instance.id, task_def, None).await?;
        }

        // Send event
        let _ = self.event_tx.send(WorkflowEvent::InstanceStarted {
            instance_id: instance.id,
            workflow_id: req.workflow_id,
            asset_id: req.asset_id,
        });

        Ok(instance)
    }

    /// Transition workflow to new state
    ///
    /// # Errors
    ///
    /// Returns an error if the transition is invalid or fails
    pub async fn transition(&self, req: TransitionRequest) -> Result<WorkflowInstance> {
        // Get instance
        let instance = self.get_instance(req.instance_id).await?;

        // Get workflow
        let workflow = self.get_workflow(instance.workflow_id).await?;
        let config: WorkflowConfig = serde_json::from_value(workflow.config)?;

        // Find current state
        let current_state_name = instance
            .current_state
            .as_ref()
            .ok_or_else(|| MamError::Internal("Instance has no current state".to_string()))?;

        // Find valid transition
        let transition = config
            .transitions
            .iter()
            .find(|t| t.from_state == *current_state_name && t.action == req.action)
            .ok_or_else(|| {
                MamError::InvalidInput(format!(
                    "Invalid transition: {} -> {}",
                    current_state_name, req.action
                ))
            })?;

        // Check permissions
        if let Some(required_role) = &transition.required_role {
            let user = self.db.get_user(req.user_id).await?;
            if user.role.to_string() != *required_role {
                return Err(MamError::PermissionDenied(
                    "User does not have required role for this transition".to_string(),
                ));
            }
        }

        // Find new state
        let new_state = config
            .states
            .iter()
            .find(|s| s.name == transition.to_state)
            .ok_or_else(|| MamError::Internal("Target state not found".to_string()))?;

        // Update instance
        let updated = if new_state.is_final {
            sqlx::query_as::<_, WorkflowInstance>(
                "UPDATE workflow_instances
                 SET current_state = $2, status = 'completed', completed_at = NOW()
                 WHERE id = $1
                 RETURNING *",
            )
            .bind(req.instance_id)
            .bind(&new_state.name)
            .fetch_one(self.db.pool())
            .await?
        } else {
            sqlx::query_as::<_, WorkflowInstance>(
                "UPDATE workflow_instances
                 SET current_state = $2
                 WHERE id = $1
                 RETURNING *",
            )
            .bind(req.instance_id)
            .bind(&new_state.name)
            .fetch_one(self.db.pool())
            .await?
        };

        // Create tasks for new state
        if !new_state.is_final {
            for task_def in &new_state.tasks {
                self.create_task(instance.id, task_def, None).await?;
            }
        }

        // Send events
        let _ = self.event_tx.send(WorkflowEvent::StateChanged {
            instance_id: req.instance_id,
            from_state: current_state_name.clone(),
            to_state: new_state.name.clone(),
        });

        if new_state.is_final {
            let _ = self.event_tx.send(WorkflowEvent::InstanceCompleted {
                instance_id: req.instance_id,
            });
        }

        Ok(updated)
    }

    /// Create a task
    async fn create_task(
        &self,
        instance_id: Uuid,
        task_def: &TaskDefinition,
        assigned_to: Option<Uuid>,
    ) -> Result<WorkflowTask> {
        let task = sqlx::query_as::<_, WorkflowTask>(
            "INSERT INTO workflow_tasks
             (id, instance_id, task_type, assigned_to, status, created_at)
             VALUES ($1, $2, $3, $4, 'pending', NOW())
             RETURNING *",
        )
        .bind(Uuid::new_v4())
        .bind(instance_id)
        .bind(&task_def.task_type)
        .bind(assigned_to)
        .fetch_one(self.db.pool())
        .await?;

        // Send event
        let _ = self.event_tx.send(WorkflowEvent::TaskCreated {
            task_id: task.id,
            instance_id,
            assigned_to,
        });

        Ok(task)
    }

    /// Complete a task
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails
    pub async fn complete_task(&self, req: CompleteTaskRequest) -> Result<WorkflowTask> {
        let task = sqlx::query_as::<_, WorkflowTask>(
            "UPDATE workflow_tasks
             SET status = 'completed', completed_by = $2, completed_at = NOW(), comment = $3
             WHERE id = $1
             RETURNING *",
        )
        .bind(req.task_id)
        .bind(req.user_id)
        .bind(req.comment)
        .fetch_one(self.db.pool())
        .await?;

        // Send event
        let _ = self.event_tx.send(WorkflowEvent::TaskCompleted {
            task_id: req.task_id,
            instance_id: task.instance_id,
            completed_by: req.user_id,
        });

        Ok(task)
    }

    /// Get workflow instance
    ///
    /// # Errors
    ///
    /// Returns an error if the instance is not found
    pub async fn get_instance(&self, instance_id: Uuid) -> Result<WorkflowInstance> {
        let instance =
            sqlx::query_as::<_, WorkflowInstance>("SELECT * FROM workflow_instances WHERE id = $1")
                .bind(instance_id)
                .fetch_one(self.db.pool())
                .await?;

        Ok(instance)
    }

    /// List workflow instances
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn list_instances(
        &self,
        workflow_id: Option<Uuid>,
        asset_id: Option<Uuid>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<WorkflowInstance>> {
        let instances = if let Some(wf_id) = workflow_id {
            sqlx::query_as::<_, WorkflowInstance>(
                "SELECT * FROM workflow_instances WHERE workflow_id = $1
                 ORDER BY started_at DESC LIMIT $2 OFFSET $3",
            )
            .bind(wf_id)
            .bind(limit)
            .bind(offset)
            .fetch_all(self.db.pool())
            .await?
        } else if let Some(a_id) = asset_id {
            sqlx::query_as::<_, WorkflowInstance>(
                "SELECT * FROM workflow_instances WHERE asset_id = $1
                 ORDER BY started_at DESC LIMIT $2 OFFSET $3",
            )
            .bind(a_id)
            .bind(limit)
            .bind(offset)
            .fetch_all(self.db.pool())
            .await?
        } else {
            sqlx::query_as::<_, WorkflowInstance>(
                "SELECT * FROM workflow_instances ORDER BY started_at DESC LIMIT $1 OFFSET $2",
            )
            .bind(limit)
            .bind(offset)
            .fetch_all(self.db.pool())
            .await?
        };

        Ok(instances)
    }

    /// Get tasks for instance
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn get_instance_tasks(&self, instance_id: Uuid) -> Result<Vec<WorkflowTask>> {
        let tasks = sqlx::query_as::<_, WorkflowTask>(
            "SELECT * FROM workflow_tasks WHERE instance_id = $1 ORDER BY created_at",
        )
        .bind(instance_id)
        .fetch_all(self.db.pool())
        .await?;

        Ok(tasks)
    }

    /// Get tasks assigned to user
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn get_user_tasks(
        &self,
        user_id: Uuid,
        status: Option<String>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<WorkflowTask>> {
        let tasks = if let Some(status) = status {
            sqlx::query_as::<_, WorkflowTask>(
                "SELECT * FROM workflow_tasks
                 WHERE assigned_to = $1 AND status = $2
                 ORDER BY created_at DESC LIMIT $3 OFFSET $4",
            )
            .bind(user_id)
            .bind(status)
            .bind(limit)
            .bind(offset)
            .fetch_all(self.db.pool())
            .await?
        } else {
            sqlx::query_as::<_, WorkflowTask>(
                "SELECT * FROM workflow_tasks
                 WHERE assigned_to = $1
                 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
            )
            .bind(user_id)
            .bind(limit)
            .bind(offset)
            .fetch_all(self.db.pool())
            .await?
        };

        Ok(tasks)
    }

    /// Shutdown the workflow engine
    ///
    /// # Errors
    ///
    /// Returns an error if shutdown fails
    pub async fn shutdown(&self) -> Result<()> {
        // Note: shutdown_tx is consumed when shutdown is called
        // In a real implementation, we'd use a different approach
        tracing::info!("Workflow engine shutdown requested");
        Ok(())
    }

    /// Event processor loop
    async fn event_processor_loop(
        mut event_rx: mpsc::UnboundedReceiver<WorkflowEvent>,
        mut shutdown_rx: tokio::sync::oneshot::Receiver<()>,
        _db: Arc<Database>,
        config: MamConfig,
    ) {
        loop {
            tokio::select! {
                Some(event) = event_rx.recv() => {
                    Self::process_event(event, &config).await;
                }
                _ = &mut shutdown_rx => {
                    tracing::info!("Workflow event processor shutting down");
                    break;
                }
            }
        }
    }

    /// Process a workflow event
    async fn process_event(event: WorkflowEvent, config: &MamConfig) {
        tracing::debug!("Processing workflow event: {:?}", event);

        if config.enable_email {
            Self::send_email_notification(&event, config).await;
        }
    }

    /// Send an email (or webhook) notification for a workflow event.
    ///
    /// When an SMTP server is configured the method formats a message and POSTs
    /// it to the SMTP endpoint via HTTP (a lightweight approach that avoids
    /// pulling in a full SMTP crate).  When no SMTP server is configured the
    /// notification is emitted as a structured log line at INFO level so that
    /// log-shipping pipelines can pick it up.
    async fn send_email_notification(event: &WorkflowEvent, config: &MamConfig) {
        // Build a human-readable subject and body for the event.
        let (subject, body) = match event {
            WorkflowEvent::InstanceStarted {
                instance_id,
                workflow_id,
                asset_id,
            } => (
                format!("[OxiMedia MAM] Workflow started: {instance_id}"),
                format!(
                    "A workflow instance has been started.

                     Instance ID : {instance_id}
                     Workflow ID : {workflow_id}
                     Asset ID    : {asset_id}
"
                ),
            ),
            WorkflowEvent::StateChanged {
                instance_id,
                from_state,
                to_state,
            } => (
                format!("[OxiMedia MAM] State changed → {to_state}"),
                format!(
                    "A workflow instance has transitioned to a new state.

                     Instance ID : {instance_id}
                     From state  : {from_state}
                     To state    : {to_state}
"
                ),
            ),
            WorkflowEvent::TaskCreated {
                task_id,
                instance_id,
                assigned_to,
            } => {
                let assignee = assigned_to
                    .map(|u| u.to_string())
                    .unwrap_or_else(|| "unassigned".to_string());
                (
                    format!("[OxiMedia MAM] Task assigned: {task_id}"),
                    format!(
                        "A workflow task has been created and assigned.

                         Task ID     : {task_id}
                         Instance ID : {instance_id}
                         Assigned to : {assignee}
"
                    ),
                )
            }
            WorkflowEvent::TaskCompleted {
                task_id,
                instance_id,
                completed_by,
            } => (
                format!("[OxiMedia MAM] Task completed: {task_id}"),
                format!(
                    "A workflow task has been completed.

                     Task ID      : {task_id}
                     Instance ID  : {instance_id}
                     Completed by : {completed_by}
"
                ),
            ),
            WorkflowEvent::InstanceCompleted { instance_id } => (
                format!("[OxiMedia MAM] Workflow completed: {instance_id}"),
                format!(
                    "A workflow instance has completed successfully.

                     Instance ID : {instance_id}
"
                ),
            ),
            WorkflowEvent::InstanceFailed { instance_id, error } => (
                format!("[OxiMedia MAM] Workflow FAILED: {instance_id}"),
                format!(
                    "A workflow instance has failed.

                     Instance ID : {instance_id}
                     Error       : {error}
"
                ),
            ),
        };

        if let Some(smtp_server) = &config.smtp_server {
            // Attempt to deliver via HTTP-based SMTP relay (e.g. MailHog, MailPit,
            // SendGrid-compatible endpoint, or any SMTP-over-HTTP bridge).
            let payload = serde_json::json!({
                "from": "noreply@oximedia.local",
                "to": ["admin@oximedia.local"],
                "subject": subject,
                "text": body,
            });

            match reqwest::Client::new()
                .post(smtp_server)
                .json(&payload)
                .timeout(std::time::Duration::from_secs(10))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    tracing::info!("Email notification sent: {}", subject);
                }
                Ok(resp) => {
                    tracing::warn!(
                        "Email notification delivery failed (HTTP {}): {}",
                        resp.status(),
                        subject
                    );
                }
                Err(err) => {
                    tracing::warn!("Email notification delivery error: {} – {}", subject, err);
                }
            }
        } else {
            // No SMTP server configured: emit a structured INFO log that any
            // log-shipping pipeline (Loki, ELK, Datadog …) can ingest.
            tracing::info!(
                subject = %subject,
                body = %body,
                "Workflow email notification (SMTP not configured)"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_config_serialization() {
        let config = WorkflowConfig {
            states: vec![],
            transitions: vec![],
            notifications: NotificationConfig {
                email_enabled: false,
                webhook_enabled: false,
                webhook_url: None,
                notify_on_state_change: true,
                notify_on_task_assigned: true,
                notify_on_completion: true,
            },
        };

        let json = serde_json::to_string(&config).expect("should succeed in test");
        let deserialized: WorkflowConfig =
            serde_json::from_str(&json).expect("should succeed in test");

        assert!(deserialized.notifications.notify_on_completion);
    }

    #[test]
    fn test_workflow_state() {
        let state = WorkflowState {
            name: "review".to_string(),
            description: Some("Review stage".to_string()),
            is_initial: false,
            is_final: false,
            required_role: Some("editor".to_string()),
            tasks: vec![],
        };

        assert_eq!(state.name, "review");
        assert!(!state.is_final);
    }
}
