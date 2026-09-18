//! Protocol buffer generated code stub
//! This is a minimal stub to allow the crate to compile
//! Full protobuf generation requires tonic build configuration

use async_trait::async_trait;
use std::collections::HashMap;
use tonic::server::NamedService;
use tonic::{Request, Response, Status};

// ============================================================
// Message Types
// ============================================================

// Worker capabilities
#[derive(Clone, PartialEq, Debug, Default)]
pub struct WorkerCapabilities {
    pub cpu_cores: u32,
    pub memory_bytes: u64,
    pub gpu_devices: Vec<String>,
    pub supported_codecs: Vec<String>,
    pub supported_hwaccels: Vec<String>,
    pub relative_speed: f32,
    pub max_concurrent_jobs: u32,
}

// Worker registration
#[derive(Clone, PartialEq, Debug, Default)]
pub struct WorkerRegistration {
    pub worker_id: String,
    pub hostname: String,
    pub ip_address: String,
    pub port: u32,
    pub capabilities: Option<Box<WorkerCapabilities>>,
    pub metadata: HashMap<String, String>,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct WorkerRegistrationResponse {
    pub success: bool,
    pub message: String,
    pub assigned_worker_id: String,
}

// Heartbeat
#[derive(Clone, PartialEq, Debug, Default)]
pub struct WorkerHeartbeat {
    pub worker_id: String,
    pub status: Option<Box<WorkerStatus>>,
    pub active_job_ids: Vec<String>,
    pub metrics: Option<Box<WorkerMetrics>>,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct HeartbeatResponse {
    pub acknowledged: bool,
    pub jobs_to_cancel: Vec<String>,
    pub should_drain: bool,
}

// Worker status
#[derive(Clone, PartialEq, Debug, Default)]
pub struct WorkerStatus {
    pub state: i32,
    pub active_jobs: u32,
    pub queued_jobs: u32,
}

// Worker metrics
#[derive(Clone, PartialEq, Debug, Default)]
pub struct WorkerMetrics {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub gpu_usage: f32,
    pub bytes_processed: u64,
    pub frames_encoded: u32,
}

// Worker unregistration
#[derive(Clone, PartialEq, Debug, Default)]
pub struct WorkerUnregistration {
    pub worker_id: String,
    pub reason: String,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct UnregistrationResponse {
    pub success: bool,
}

// Job request and assignment
#[derive(Clone, PartialEq, Debug, Default)]
pub struct JobRequest {
    pub worker_id: String,
    pub max_jobs: u32,
    pub preferred_codecs: Vec<String>,
}

pub mod job_assignment {
    use super::EncodingTask;

    #[derive(Clone, PartialEq, Debug, Default)]
    pub struct Job {
        pub job_id: String,
        pub task: Option<Box<EncodingTask>>,
        pub priority: u32,
        pub deadline_timestamp: i64,
    }
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct JobAssignment {
    pub jobs: Vec<job_assignment::Job>,
    pub has_more: bool,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct Job {
    pub job_id: String,
    pub task: Option<Box<EncodingTask>>,
    pub priority: u32,
    pub deadline_timestamp: i64,
}

// Encoding task
#[derive(Clone, PartialEq, Debug, Default)]
pub struct EncodingTask {
    pub task_id: String,
    pub source_url: String,
    pub codec: String,
    pub strategy: i32,
    pub params: Option<Box<EncodingParams>>,
    pub output_url: String,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct EncodingParams {
    pub bitrate: u32,
    pub width: u32,
    pub height: u32,
    pub preset: String,
    pub profile: String,
    pub crf: u32,
    pub extra_params: HashMap<String, String>,
}

// Time segment
#[derive(Clone, PartialEq, Debug, Default)]
pub struct TimeSegment {
    pub start_time: f64,
    pub end_time: f64,
    pub overlap: f64,
}

// Tile segment
#[derive(Clone, PartialEq, Debug, Default)]
pub struct TileSegment {
    pub tile_x: u32,
    pub tile_y: u32,
    pub tile_width: u32,
    pub tile_height: u32,
}

// GOP segment
#[derive(Clone, PartialEq, Debug, Default)]
pub struct GopSegment {
    pub start_frame: u64,
    pub end_frame: u64,
    pub keyframe_indices: Vec<u64>,
}

// Progress reporting
#[derive(Clone, PartialEq, Debug, Default)]
pub struct ProgressReport {
    pub job_id: String,
    pub worker_id: String,
    pub progress: f32,
    pub frames_encoded: u64,
    pub bytes_written: u64,
    pub encoding_speed: f32,
    pub estimated_completion_timestamp: i64,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct ProgressAcknowledgment {
    pub acknowledged: bool,
}

// Result submission
#[derive(Clone, PartialEq, Debug, Default)]
pub struct JobResult {
    pub job_id: String,
    pub worker_id: String,
    pub output_url: String,
    pub output_size: u64,
    pub encoding_time: f64,
    pub metadata: Option<Box<ResultMetadata>>,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct ResultMetadata {
    pub frames_encoded: u64,
    pub average_bitrate: f64,
    pub checksum: String,
    pub extra_metadata: HashMap<String, String>,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct ResultAcknowledgment {
    pub acknowledged: bool,
    pub next_job_id: String,
}

// Failure reporting
#[derive(Clone, PartialEq, Debug, Default)]
pub struct JobFailure {
    pub job_id: String,
    pub worker_id: String,
    pub error_message: String,
    pub error_code: String,
    pub is_transient: bool,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct FailureAcknowledgment {
    pub should_retry: bool,
    pub reassigned_job_id: String,
}

// Status queries with nested Query enum
pub mod worker_status_request {
    #[derive(Clone, PartialEq, Debug)]
    pub enum Query {
        WorkerId(String),
        AllWorkers(bool),
    }
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct WorkerStatusRequest {
    pub query: Option<worker_status_request::Query>,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct WorkerStatusResponse {
    pub workers: Vec<WorkerInfo>,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct WorkerInfo {
    pub worker_id: String,
    pub hostname: String,
    pub status: Option<Box<WorkerStatus>>,
    pub metrics: Option<Box<WorkerMetrics>>,
    pub last_heartbeat_timestamp: i64,
}

pub mod job_status_request {
    #[derive(Clone, PartialEq, Debug)]
    pub enum Query {
        JobId(String),
        TaskId(String),
    }
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct JobStatusRequest {
    pub query: Option<job_status_request::Query>,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct JobStatusResponse {
    pub job_id: String,
    pub state: i32,
    pub assigned_worker_id: String,
    pub progress: f32,
    pub started_timestamp: i64,
    pub completed_timestamp: i64,
}

// Job cancellation with nested Target enum
pub mod job_cancellation {
    #[derive(Clone, PartialEq, Debug)]
    pub enum Target {
        JobId(String),
        TaskId(String),
    }
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct JobCancellation {
    pub target: Option<job_cancellation::Target>,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct CancellationResponse {
    pub success: bool,
    pub cancelled_job_ids: Vec<String>,
}

// ============================================================
// Service Definitions
// ============================================================

pub mod coordinator_service_server {
    use super::{
        async_trait, CancellationResponse, FailureAcknowledgment, HeartbeatResponse, JobAssignment,
        JobCancellation, JobFailure, JobRequest, JobResult, JobStatusRequest, JobStatusResponse,
        NamedService, ProgressAcknowledgment, ProgressReport, Request, Response,
        ResultAcknowledgment, Status, UnregistrationResponse, WorkerHeartbeat, WorkerRegistration,
        WorkerRegistrationResponse, WorkerStatusRequest, WorkerStatusResponse,
        WorkerUnregistration,
    };

    #[async_trait]
    pub trait CoordinatorService: Send + Sync + 'static {
        async fn register_worker(
            &self,
            request: Request<WorkerRegistration>,
        ) -> Result<Response<WorkerRegistrationResponse>, Status>;
        async fn heartbeat(
            &self,
            request: Request<WorkerHeartbeat>,
        ) -> Result<Response<HeartbeatResponse>, Status>;
        async fn unregister_worker(
            &self,
            request: Request<WorkerUnregistration>,
        ) -> Result<Response<UnregistrationResponse>, Status>;
        async fn request_job(
            &self,
            request: Request<JobRequest>,
        ) -> Result<Response<JobAssignment>, Status>;
        async fn report_progress(
            &self,
            request: Request<ProgressReport>,
        ) -> Result<Response<ProgressAcknowledgment>, Status>;
        async fn submit_result(
            &self,
            request: Request<JobResult>,
        ) -> Result<Response<ResultAcknowledgment>, Status>;
        async fn report_failure(
            &self,
            request: Request<JobFailure>,
        ) -> Result<Response<FailureAcknowledgment>, Status>;
        async fn get_worker_status(
            &self,
            request: Request<WorkerStatusRequest>,
        ) -> Result<Response<WorkerStatusResponse>, Status>;
        async fn get_job_status(
            &self,
            request: Request<JobStatusRequest>,
        ) -> Result<Response<JobStatusResponse>, Status>;
        async fn cancel_job(
            &self,
            request: Request<JobCancellation>,
        ) -> Result<Response<CancellationResponse>, Status>;
    }

    #[derive(Clone)]
    #[allow(dead_code)]
    pub struct CoordinatorServiceServer<T> {
        inner: std::sync::Arc<T>,
    }

    impl<T: CoordinatorService> CoordinatorServiceServer<T> {
        pub fn new(inner: T) -> Self {
            Self {
                inner: std::sync::Arc::new(inner),
            }
        }

        #[must_use]
        pub fn inner_ref(&self) -> &T {
            &self.inner
        }
    }

    impl<T> NamedService for CoordinatorServiceServer<T> {
        const NAME: &'static str = "coordinator.CoordinatorService";
    }
}

pub mod coordinator_service_client {
    use super::{
        CancellationResponse, FailureAcknowledgment, HeartbeatResponse, JobAssignment,
        JobCancellation, JobFailure, JobRequest, JobResult, JobStatusRequest, JobStatusResponse,
        ProgressAcknowledgment, ProgressReport, Request, Response, ResultAcknowledgment, Status,
        UnregistrationResponse, WorkerHeartbeat, WorkerRegistration, WorkerRegistrationResponse,
        WorkerStatusRequest, WorkerStatusResponse, WorkerUnregistration,
    };
    use tonic::transport::Channel;

    #[derive(Clone)]
    pub struct CoordinatorServiceClient<T> {
        inner: T,
    }

    impl CoordinatorServiceClient<Channel> {
        pub async fn connect<D>(
            dst: D,
        ) -> std::result::Result<Self, Box<dyn std::error::Error + Send + Sync>>
        where
            D: std::convert::TryInto<tonic::transport::Endpoint> + Clone,
            D::Error: Into<Box<dyn std::error::Error + Send + Sync + 'static>>,
        {
            let endpoint: tonic::transport::Endpoint =
                dst.try_into().map_err(std::convert::Into::into)?;
            let channel = endpoint.connect().await?;
            Ok(Self::new(channel))
        }
    }

    impl<T> CoordinatorServiceClient<T> {
        pub fn new(inner: T) -> Self {
            Self { inner }
        }

        pub fn inner_ref(&self) -> &T {
            &self.inner
        }
    }

    impl CoordinatorServiceClient<Channel> {
        pub async fn register_worker(
            &mut self,
            _request: Request<WorkerRegistration>,
        ) -> Result<Response<WorkerRegistrationResponse>, Status> {
            Ok(Response::new(WorkerRegistrationResponse::default()))
        }

        pub async fn heartbeat(
            &mut self,
            _request: Request<WorkerHeartbeat>,
        ) -> Result<Response<HeartbeatResponse>, Status> {
            Ok(Response::new(HeartbeatResponse::default()))
        }

        pub async fn unregister_worker(
            &mut self,
            _request: Request<WorkerUnregistration>,
        ) -> Result<Response<UnregistrationResponse>, Status> {
            Ok(Response::new(UnregistrationResponse::default()))
        }

        pub async fn request_job(
            &mut self,
            _request: Request<JobRequest>,
        ) -> Result<Response<JobAssignment>, Status> {
            Ok(Response::new(JobAssignment::default()))
        }

        pub async fn report_progress(
            &mut self,
            _request: Request<ProgressReport>,
        ) -> Result<Response<ProgressAcknowledgment>, Status> {
            Ok(Response::new(ProgressAcknowledgment::default()))
        }

        pub async fn submit_result(
            &mut self,
            _request: Request<JobResult>,
        ) -> Result<Response<ResultAcknowledgment>, Status> {
            Ok(Response::new(ResultAcknowledgment::default()))
        }

        pub async fn report_failure(
            &mut self,
            _request: Request<JobFailure>,
        ) -> Result<Response<FailureAcknowledgment>, Status> {
            Ok(Response::new(FailureAcknowledgment::default()))
        }

        pub async fn get_worker_status(
            &mut self,
            _request: Request<WorkerStatusRequest>,
        ) -> Result<Response<WorkerStatusResponse>, Status> {
            Ok(Response::new(WorkerStatusResponse::default()))
        }

        pub async fn get_job_status(
            &mut self,
            _request: Request<JobStatusRequest>,
        ) -> Result<Response<JobStatusResponse>, Status> {
            Ok(Response::new(JobStatusResponse::default()))
        }

        pub async fn cancel_job(
            &mut self,
            _request: Request<JobCancellation>,
        ) -> Result<Response<CancellationResponse>, Status> {
            Ok(Response::new(CancellationResponse::default()))
        }
    }
}
