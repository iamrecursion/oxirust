// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! REST API for render farm.

use crate::coordinator::{Coordinator, CoordinatorStats};
use crate::error::{Error, Result};
use crate::job::{JobId, JobSubmission};
use crate::worker::{WorkerId, WorkerRegistration};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing::info;

/// API configuration
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// Listen address
    pub listen_addr: String,
    /// Listen port
    pub port: u16,
    /// Enable CORS
    pub enable_cors: bool,
    /// API key for authentication (optional)
    pub api_key: Option<String>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0".to_string(),
            port: 8080,
            enable_cors: true,
            api_key: None,
        }
    }
}

/// API error response
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let status = match self {
            Error::JobNotFound(_) | Error::WorkerNotFound(_) | Error::PoolNotFound(_) => {
                StatusCode::NOT_FOUND
            }
            Error::InvalidStateTransition { .. } | Error::InvalidFrameRange { .. } => {
                StatusCode::BAD_REQUEST
            }
            Error::BudgetExceeded { .. } => StatusCode::PAYMENT_REQUIRED,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };

        let body = Json(ErrorResponse {
            error: self.to_string(),
        });

        (status, body).into_response()
    }
}

/// Job submission request
#[derive(Debug, Deserialize)]
struct SubmitJobRequest {
    submission: JobSubmission,
}

/// Job submission response
#[derive(Debug, Serialize)]
struct SubmitJobResponse {
    job_id: JobId,
}

/// Job info response
#[derive(Debug, Serialize)]
struct JobInfoResponse {
    job_id: JobId,
    state: String,
    progress: f64,
}

/// Worker registration request
#[derive(Debug, Deserialize)]
struct RegisterWorkerRequest {
    registration: WorkerRegistration,
}

/// Worker registration response
#[derive(Debug, Serialize)]
struct RegisterWorkerResponse {
    worker_id: WorkerId,
}

/// Statistics response
#[derive(Debug, Serialize)]
struct StatsResponse {
    stats: CoordinatorStats,
}

/// Render farm API
pub struct RenderFarmApi {
    config: ApiConfig,
    coordinator: Arc<Coordinator>,
}

impl RenderFarmApi {
    /// Create a new API instance
    #[must_use]
    pub fn new(config: ApiConfig, coordinator: Arc<Coordinator>) -> Self {
        Self {
            config,
            coordinator,
        }
    }

    /// Create router
    fn create_router(&self) -> Router {
        let mut router = Router::new()
            .route("/api/v1/jobs", post(submit_job))
            .route("/api/v1/jobs", get(list_jobs))
            .route("/api/v1/jobs/:id", get(get_job))
            .route("/api/v1/jobs/:id", delete(cancel_job))
            .route("/api/v1/workers", post(register_worker))
            .route("/api/v1/workers", get(list_workers))
            .route("/api/v1/workers/:id", get(get_worker))
            .route("/api/v1/stats", get(get_stats))
            .route("/health", get(health_check))
            .with_state(self.coordinator.clone());

        if self.config.enable_cors {
            router = router.layer(CorsLayer::permissive());
        }

        router
    }

    /// Start API server
    pub async fn start(&self) -> Result<()> {
        let addr = format!("{}:{}", self.config.listen_addr, self.config.port);
        info!("Starting API server on {}", addr);

        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .map_err(|e| Error::Network(e.to_string()))?;

        let router = self.create_router();

        axum::serve(listener, router)
            .await
            .map_err(|e| Error::Api(e.to_string()))?;

        Ok(())
    }
}

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({"status": "ok"})))
}

/// Submit job endpoint
async fn submit_job(
    State(coordinator): State<Arc<Coordinator>>,
    Json(request): Json<SubmitJobRequest>,
) -> Result<Json<SubmitJobResponse>> {
    let job_id = coordinator.submit_job(request.submission).await?;
    Ok(Json(SubmitJobResponse { job_id }))
}

/// List jobs endpoint
async fn list_jobs(State(coordinator): State<Arc<Coordinator>>) -> Result<Json<Vec<JobId>>> {
    let jobs = coordinator.list_jobs();
    Ok(Json(jobs))
}

/// Get job endpoint
async fn get_job(
    State(coordinator): State<Arc<Coordinator>>,
    Path(id): Path<String>,
) -> Result<Json<JobInfoResponse>> {
    let job_id: JobId =
        serde_json::from_str(&format!("\"{id}\"")).map_err(|_| Error::JobNotFound(id.clone()))?;

    let job_arc = coordinator.get_job(job_id).ok_or(Error::JobNotFound(id))?;

    let job = job_arc.read();

    Ok(Json(JobInfoResponse {
        job_id: job.id,
        state: job.state.to_string(),
        progress: job.progress,
    }))
}

/// Cancel job endpoint
async fn cancel_job(
    State(coordinator): State<Arc<Coordinator>>,
    Path(id): Path<String>,
) -> Result<StatusCode> {
    let job_id: JobId =
        serde_json::from_str(&format!("\"{id}\"")).map_err(|_| Error::JobNotFound(id.clone()))?;

    coordinator.cancel_job(job_id).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Register worker endpoint
async fn register_worker(
    State(coordinator): State<Arc<Coordinator>>,
    Json(request): Json<RegisterWorkerRequest>,
) -> Result<Json<RegisterWorkerResponse>> {
    let worker_id = coordinator.register_worker(request.registration).await?;
    Ok(Json(RegisterWorkerResponse { worker_id }))
}

/// List workers endpoint
async fn list_workers(State(coordinator): State<Arc<Coordinator>>) -> Result<Json<Vec<WorkerId>>> {
    let workers = coordinator.list_workers();
    Ok(Json(workers))
}

/// Get worker endpoint
async fn get_worker(
    State(coordinator): State<Arc<Coordinator>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>> {
    let worker_id: WorkerId = serde_json::from_str(&format!("\"{id}\""))
        .map_err(|_| Error::WorkerNotFound(id.clone()))?;

    let worker_arc = coordinator
        .get_worker(worker_id)
        .ok_or(Error::WorkerNotFound(id))?;

    let worker = worker_arc.read();

    Ok(Json(serde_json::json!({
        "id": worker.id,
        "hostname": worker.registration.hostname,
        "state": worker.state,
    })))
}

/// Get statistics endpoint
async fn get_stats(State(coordinator): State<Arc<Coordinator>>) -> Result<Json<StatsResponse>> {
    let stats = coordinator.get_stats();
    Ok(Json(StatsResponse { stats }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coordinator::CoordinatorConfig;

    #[tokio::test]
    async fn test_api_creation() -> Result<()> {
        let config = ApiConfig::default();
        let coordinator_config = CoordinatorConfig::default();
        let coordinator = Arc::new(Coordinator::new(coordinator_config).await?);
        let _api = RenderFarmApi::new(config, coordinator);
        Ok(())
    }

    #[test]
    fn test_error_response() {
        let error = Error::JobNotFound("test".to_string());
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_api_config_default() {
        let config = ApiConfig::default();
        assert_eq!(config.port, 8080);
        assert!(config.enable_cors);
    }
}
