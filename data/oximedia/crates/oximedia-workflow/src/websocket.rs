//! WebSocket support for real-time workflow updates.

use crate::monitoring::{MonitoringService, WorkflowMetrics};
use crate::workflow::WorkflowId;
use axum::{
    extract::{
        ws::{Message, WebSocket},
        State, WebSocketUpgrade,
    },
    response::Response,
};
use futures::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{debug, error, info};

/// WebSocket event types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkflowEvent {
    /// Workflow started.
    WorkflowStarted {
        /// Workflow ID.
        workflow_id: WorkflowId,
        /// Workflow name.
        workflow_name: String,
    },

    /// Workflow completed.
    WorkflowCompleted {
        /// Workflow ID.
        workflow_id: WorkflowId,
        /// Success status.
        success: bool,
    },

    /// Task state changed.
    TaskStateChanged {
        /// Workflow ID.
        workflow_id: WorkflowId,
        /// Task ID.
        task_id: crate::task::TaskId,
        /// Task name.
        task_name: String,
        /// New state.
        state: crate::task::TaskState,
    },

    /// Progress update.
    ProgressUpdate {
        /// Workflow ID.
        workflow_id: WorkflowId,
        /// Progress percentage (0-100).
        progress: f64,
        /// Completed tasks.
        completed_tasks: usize,
        /// Total tasks.
        total_tasks: usize,
    },

    /// Metrics update.
    MetricsUpdate {
        /// Workflow metrics.
        metrics: WorkflowMetrics,
    },

    /// Error occurred.
    Error {
        /// Workflow ID.
        workflow_id: WorkflowId,
        /// Error message.
        message: String,
    },
}

/// WebSocket state.
#[derive(Clone)]
pub struct WebSocketState {
    /// Event broadcaster.
    pub broadcaster: Arc<broadcast::Sender<WorkflowEvent>>,
    /// Monitoring service.
    pub monitoring: Arc<MonitoringService>,
}

impl WebSocketState {
    /// Create a new WebSocket state.
    #[must_use]
    pub fn new(monitoring: Arc<MonitoringService>) -> Self {
        let (tx, _) = broadcast::channel(1000);
        Self {
            broadcaster: Arc::new(tx),
            monitoring,
        }
    }

    /// Broadcast an event to all connected clients.
    pub fn broadcast(&self, event: WorkflowEvent) {
        let _ = self.broadcaster.send(event);
    }

    /// Broadcast workflow started event.
    pub fn broadcast_workflow_started(&self, workflow_id: WorkflowId, workflow_name: String) {
        self.broadcast(WorkflowEvent::WorkflowStarted {
            workflow_id,
            workflow_name,
        });
    }

    /// Broadcast workflow completed event.
    pub fn broadcast_workflow_completed(&self, workflow_id: WorkflowId, success: bool) {
        self.broadcast(WorkflowEvent::WorkflowCompleted {
            workflow_id,
            success,
        });
    }

    /// Broadcast task state changed event.
    pub fn broadcast_task_state_changed(
        &self,
        workflow_id: WorkflowId,
        task_id: crate::task::TaskId,
        task_name: String,
        state: crate::task::TaskState,
    ) {
        self.broadcast(WorkflowEvent::TaskStateChanged {
            workflow_id,
            task_id,
            task_name,
            state,
        });
    }

    /// Broadcast progress update.
    pub fn broadcast_progress_update(
        &self,
        workflow_id: WorkflowId,
        progress: f64,
        completed_tasks: usize,
        total_tasks: usize,
    ) {
        self.broadcast(WorkflowEvent::ProgressUpdate {
            workflow_id,
            progress,
            completed_tasks,
            total_tasks,
        });
    }

    /// Broadcast metrics update.
    pub fn broadcast_metrics_update(&self, metrics: WorkflowMetrics) {
        self.broadcast(WorkflowEvent::MetricsUpdate { metrics });
    }

    /// Broadcast error event.
    pub fn broadcast_error(&self, workflow_id: WorkflowId, message: String) {
        self.broadcast(WorkflowEvent::Error {
            workflow_id,
            message,
        });
    }
}

/// WebSocket handler.
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<WebSocketState>,
) -> Response {
    ws.on_upgrade(|socket| handle_websocket(socket, state))
}

async fn handle_websocket(socket: WebSocket, state: WebSocketState) {
    let (mut sender, mut receiver) = socket.split();

    // Subscribe to events
    let mut event_rx = state.broadcaster.subscribe();

    info!("WebSocket client connected");

    // Send initial state
    let active_workflows = state.monitoring.get_active_workflows();
    for metrics in active_workflows {
        let event = WorkflowEvent::MetricsUpdate { metrics };
        if let Ok(json) = serde_json::to_string(&event) {
            if sender.send(Message::Text(json.into())).await.is_err() {
                error!("Failed to send initial state");
                return;
            }
        }
    }

    // Spawn event broadcaster task
    let mut send_task = tokio::spawn(async move {
        while let Ok(event) = event_rx.recv().await {
            if let Ok(json) = serde_json::to_string(&event) {
                if sender.send(Message::Text(json.into())).await.is_err() {
                    break;
                }
            }
        }
    });

    // Handle incoming messages
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    debug!("Received WebSocket message: {}", text);
                    // Handle client messages (e.g., subscribe to specific workflows)
                }
                Message::Close(_) => {
                    info!("WebSocket client disconnected");
                    break;
                }
                Message::Ping(_data) => {
                    debug!("Received ping");
                    // Pong is sent automatically by axum
                }
                _ => {}
            }
        }
    });

    // Wait for either task to finish
    tokio::select! {
        _ = &mut send_task => {
            recv_task.abort();
        }
        _ = &mut recv_task => {
            send_task.abort();
        }
    }

    info!("WebSocket connection closed");
}

/// WebSocket client message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Subscribe to workflow updates.
    Subscribe {
        /// Workflow ID to subscribe to.
        workflow_id: WorkflowId,
    },

    /// Unsubscribe from workflow updates.
    Unsubscribe {
        /// Workflow ID to unsubscribe from.
        workflow_id: WorkflowId,
    },

    /// Subscribe to all workflows.
    SubscribeAll,

    /// Ping message.
    Ping,
}

/// Enhanced WebSocket handler with subscription management.
pub struct WebSocketManager {
    state: WebSocketState,
}

impl WebSocketManager {
    /// Create a new WebSocket manager.
    #[must_use]
    pub fn new(monitoring: Arc<MonitoringService>) -> Self {
        Self {
            state: WebSocketState::new(monitoring),
        }
    }

    /// Get WebSocket state.
    #[must_use]
    pub fn state(&self) -> WebSocketState {
        self.state.clone()
    }

    /// Broadcast an event.
    pub fn broadcast(&self, event: WorkflowEvent) {
        self.state.broadcast(event);
    }

    /// Get subscriber count.
    #[must_use]
    pub fn subscriber_count(&self) -> usize {
        self.state.broadcaster.receiver_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_websocket_state_creation() {
        let monitoring = Arc::new(MonitoringService::new());
        let state = WebSocketState::new(monitoring);
        assert_eq!(state.broadcaster.receiver_count(), 0);
    }

    #[tokio::test]
    async fn test_broadcast_event() {
        let monitoring = Arc::new(MonitoringService::new());
        let state = WebSocketState::new(monitoring);

        let mut rx = state.broadcaster.subscribe();

        let workflow_id = WorkflowId::new();
        state.broadcast_workflow_started(workflow_id, "test-workflow".to_string());

        let event = rx.recv().await.expect("should succeed in test");
        match event {
            WorkflowEvent::WorkflowStarted {
                workflow_id: id,
                workflow_name,
            } => {
                assert_eq!(id, workflow_id);
                assert_eq!(workflow_name, "test-workflow");
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[test]
    fn test_websocket_manager() {
        let monitoring = Arc::new(MonitoringService::new());
        let manager = WebSocketManager::new(monitoring);

        assert_eq!(manager.subscriber_count(), 0);
    }

    #[test]
    fn test_workflow_event_serialization() {
        let event = WorkflowEvent::WorkflowStarted {
            workflow_id: WorkflowId::new(),
            workflow_name: "test".to_string(),
        };

        let json = serde_json::to_string(&event).expect("should succeed in test");
        assert!(json.contains("workflow_started"));
    }

    #[test]
    fn test_client_message_serialization() {
        let msg = ClientMessage::Subscribe {
            workflow_id: WorkflowId::new(),
        };

        let json = serde_json::to_string(&msg).expect("should succeed in test");
        assert!(json.contains("subscribe"));
    }
}
