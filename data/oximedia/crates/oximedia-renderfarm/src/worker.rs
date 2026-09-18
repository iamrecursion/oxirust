// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Worker management for render farm.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::IpAddr;
use uuid::Uuid;

/// Unique identifier for a worker
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkerId(Uuid);

impl WorkerId {
    /// Create a new worker ID
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Get the inner UUID
    #[must_use]
    pub const fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for WorkerId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for WorkerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Worker state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerState {
    /// Worker is idle and ready
    Idle,
    /// Worker is busy rendering
    Busy,
    /// Worker is offline
    Offline,
    /// Worker has errors
    Error,
    /// Worker is paused
    Paused,
    /// Worker is starting up
    Starting,
    /// Worker is shutting down
    ShuttingDown,
}

impl std::fmt::Display for WorkerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Busy => write!(f, "Busy"),
            Self::Offline => write!(f, "Offline"),
            Self::Error => write!(f, "Error"),
            Self::Paused => write!(f, "Paused"),
            Self::Starting => write!(f, "Starting"),
            Self::ShuttingDown => write!(f, "ShuttingDown"),
        }
    }
}

/// Worker capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerCapabilities {
    /// CPU cores
    pub cpu_cores: u32,
    /// CPU model
    pub cpu_model: String,
    /// RAM in GB
    pub ram_gb: u32,
    /// GPU available
    pub has_gpu: bool,
    /// GPU model
    pub gpu_model: Option<String>,
    /// GPU memory in GB
    pub gpu_memory_gb: Option<u32>,
    /// Installed software
    pub software: Vec<String>,
    /// Available licenses
    pub licenses: Vec<String>,
    /// Supported job types
    pub supported_job_types: Vec<String>,
    /// Operating system
    pub os: String,
    /// Architecture
    pub arch: String,
}

impl Default for WorkerCapabilities {
    fn default() -> Self {
        Self {
            cpu_cores: num_cpus::get() as u32,
            cpu_model: "Unknown".to_string(),
            ram_gb: 8,
            has_gpu: false,
            gpu_model: None,
            gpu_memory_gb: None,
            software: Vec::new(),
            licenses: Vec::new(),
            supported_job_types: Vec::new(),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
        }
    }
}

/// Worker performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerMetrics {
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: f64,
    /// GPU utilization (0.0 to 1.0)
    pub gpu_utilization: Option<f64>,
    /// Network bandwidth usage (MB/s)
    pub network_bandwidth: f64,
    /// Disk I/O (MB/s)
    pub disk_io: f64,
    /// CPU temperature (Celsius)
    pub cpu_temp: Option<f64>,
    /// GPU temperature (Celsius)
    pub gpu_temp: Option<f64>,
    /// Average frame time (seconds)
    pub avg_frame_time: Option<f64>,
    /// Frames rendered in last hour
    pub frames_per_hour: u32,
    /// Error rate (errors per 100 frames)
    pub error_rate: f64,
    /// Last updated
    pub updated_at: DateTime<Utc>,
}

impl Default for WorkerMetrics {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            gpu_utilization: None,
            network_bandwidth: 0.0,
            disk_io: 0.0,
            cpu_temp: None,
            gpu_temp: None,
            avg_frame_time: None,
            frames_per_hour: 0,
            error_rate: 0.0,
            updated_at: Utc::now(),
        }
    }
}

/// Worker health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHealth {
    /// Is worker healthy
    pub healthy: bool,
    /// Health score (0.0 to 1.0)
    pub score: f64,
    /// Issues detected
    pub issues: Vec<String>,
    /// Last heartbeat
    pub last_heartbeat: DateTime<Utc>,
    /// Consecutive failures
    pub consecutive_failures: u32,
}

impl Default for WorkerHealth {
    fn default() -> Self {
        Self {
            healthy: true,
            score: 1.0,
            issues: Vec::new(),
            last_heartbeat: Utc::now(),
            consecutive_failures: 0,
        }
    }
}

/// Worker registration request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRegistration {
    /// Worker hostname
    pub hostname: String,
    /// Worker IP address
    pub ip_address: IpAddr,
    /// Worker port
    pub port: u16,
    /// Worker capabilities
    pub capabilities: WorkerCapabilities,
    /// Geographic location (optional)
    pub location: Option<String>,
    /// Custom tags
    pub tags: HashMap<String, String>,
}

/// Render worker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Worker {
    /// Worker ID
    pub id: WorkerId,
    /// Registration details
    pub registration: WorkerRegistration,
    /// Current state
    pub state: WorkerState,
    /// Performance metrics
    pub metrics: WorkerMetrics,
    /// Health status
    pub health: WorkerHealth,
    /// Registration time
    pub registered_at: DateTime<Utc>,
    /// Last activity
    pub last_active: DateTime<Utc>,
    /// Current job ID
    pub current_job_id: Option<String>,
    /// Total frames rendered
    pub total_frames_rendered: u64,
    /// Total render time (seconds)
    pub total_render_time: f64,
    /// Pool membership
    pub pool_ids: Vec<String>,
    /// Blacklisted until
    pub blacklisted_until: Option<DateTime<Utc>>,
}

impl Worker {
    /// Create a new worker from registration
    #[must_use]
    pub fn new(registration: WorkerRegistration) -> Self {
        Self {
            id: WorkerId::new(),
            registration,
            state: WorkerState::Starting,
            metrics: WorkerMetrics::default(),
            health: WorkerHealth::default(),
            registered_at: Utc::now(),
            last_active: Utc::now(),
            current_job_id: None,
            total_frames_rendered: 0,
            total_render_time: 0.0,
            pool_ids: Vec::new(),
            blacklisted_until: None,
        }
    }

    /// Check if worker is available for work
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.state == WorkerState::Idle
            && self.health.healthy
            && self
                .blacklisted_until
                .map_or(true, |until| Utc::now() >= until)
    }

    /// Check if worker is online
    #[must_use]
    pub fn is_online(&self) -> bool {
        let timeout = chrono::Duration::seconds(60);
        Utc::now() - self.health.last_heartbeat < timeout
    }

    /// Update worker state
    pub fn update_state(&mut self, new_state: WorkerState) {
        self.state = new_state;
        self.last_active = Utc::now();
    }

    /// Update metrics
    pub fn update_metrics(&mut self, metrics: WorkerMetrics) {
        self.metrics = metrics;
        self.last_active = Utc::now();
    }

    /// Record heartbeat
    pub fn heartbeat(&mut self) {
        self.health.last_heartbeat = Utc::now();
        self.last_active = Utc::now();

        // Check if worker should come back online
        if self.state == WorkerState::Offline && self.is_online() {
            self.state = WorkerState::Idle;
        }
    }

    /// Record frame rendered
    pub fn record_frame(&mut self, render_time: f64) {
        self.total_frames_rendered += 1;
        self.total_render_time += render_time;
        self.last_active = Utc::now();
    }

    /// Record error
    pub fn record_error(&mut self, error: String) {
        self.health.consecutive_failures += 1;
        self.health.issues.push(error);

        // Update health score
        self.update_health_score();

        // Mark as unhealthy if too many failures
        if self.health.consecutive_failures >= 3 {
            self.health.healthy = false;
            self.state = WorkerState::Error;
        }
    }

    /// Clear errors
    pub fn clear_errors(&mut self) {
        self.health.consecutive_failures = 0;
        self.health.issues.clear();
        self.health.healthy = true;
        self.update_health_score();
    }

    /// Update health score based on metrics and issues
    pub fn update_health_score(&mut self) {
        let mut score = 1.0;

        // Penalize for failures
        score -= f64::from(self.health.consecutive_failures) * 0.1;

        // Penalize for high error rate
        score -= self.metrics.error_rate * 0.01;

        // Penalize for high resource utilization
        if self.metrics.cpu_utilization > 0.95 {
            score -= 0.1;
        }
        if self.metrics.memory_utilization > 0.95 {
            score -= 0.1;
        }

        // Penalize for high temperature
        if let Some(temp) = self.metrics.cpu_temp {
            if temp > 80.0 {
                score -= 0.1;
            }
        }

        self.health.score = score.clamp(0.0, 1.0);
        self.health.healthy = self.health.score > 0.5;
    }

    /// Blacklist worker temporarily
    pub fn blacklist(&mut self, duration: chrono::Duration) {
        self.blacklisted_until = Some(Utc::now() + duration);
        self.state = WorkerState::Error;
    }

    /// Calculate performance score
    #[must_use]
    pub fn performance_score(&self) -> f64 {
        let mut score = 0.0;

        // Base score from CPU cores
        score += f64::from(self.registration.capabilities.cpu_cores) * 10.0;

        // Bonus for GPU
        if self.registration.capabilities.has_gpu {
            score += 50.0;
        }

        // Bonus for RAM
        score += f64::from(self.registration.capabilities.ram_gb) * 2.0;

        // Adjust for current utilization
        score *= 1.0 - self.metrics.cpu_utilization * 0.5;
        score *= 1.0 - self.metrics.memory_utilization * 0.3;

        // Adjust for health
        score *= self.health.score;

        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn create_test_registration() -> WorkerRegistration {
        WorkerRegistration {
            hostname: "worker01".to_string(),
            ip_address: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)),
            port: 8080,
            capabilities: WorkerCapabilities::default(),
            location: Some("US-West".to_string()),
            tags: HashMap::new(),
        }
    }

    #[test]
    fn test_worker_id_generation() {
        let id1 = WorkerId::new();
        let id2 = WorkerId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_worker_creation() {
        let registration = create_test_registration();
        let worker = Worker::new(registration);

        assert_eq!(worker.state, WorkerState::Starting);
        assert!(worker.health.healthy);
        assert_eq!(worker.total_frames_rendered, 0);
    }

    #[test]
    fn test_worker_availability() {
        let registration = create_test_registration();
        let mut worker = Worker::new(registration);

        // Starting state - not available
        assert!(!worker.is_available());

        // Idle state - available
        worker.state = WorkerState::Idle;
        assert!(worker.is_available());

        // Blacklisted - not available
        worker.blacklist(chrono::Duration::minutes(5));
        assert!(!worker.is_available());
    }

    #[test]
    fn test_worker_heartbeat() {
        let registration = create_test_registration();
        let mut worker = Worker::new(registration);

        let initial_heartbeat = worker.health.last_heartbeat;
        std::thread::sleep(std::time::Duration::from_millis(10));

        worker.heartbeat();
        assert!(worker.health.last_heartbeat > initial_heartbeat);
    }

    #[test]
    fn test_worker_error_recording() {
        let registration = create_test_registration();
        let mut worker = Worker::new(registration);
        worker.state = WorkerState::Idle;

        // Record errors
        worker.record_error("Error 1".to_string());
        assert_eq!(worker.health.consecutive_failures, 1);
        assert!(worker.health.healthy);

        worker.record_error("Error 2".to_string());
        assert_eq!(worker.health.consecutive_failures, 2);

        worker.record_error("Error 3".to_string());
        assert_eq!(worker.health.consecutive_failures, 3);
        assert!(!worker.health.healthy);
        assert_eq!(worker.state, WorkerState::Error);
    }

    #[test]
    fn test_worker_clear_errors() {
        let registration = create_test_registration();
        let mut worker = Worker::new(registration);

        worker.record_error("Error 1".to_string());
        worker.record_error("Error 2".to_string());
        assert_eq!(worker.health.consecutive_failures, 2);

        worker.clear_errors();
        assert_eq!(worker.health.consecutive_failures, 0);
        assert!(worker.health.issues.is_empty());
        assert!(worker.health.healthy);
    }

    #[test]
    fn test_worker_frame_recording() {
        let registration = create_test_registration();
        let mut worker = Worker::new(registration);

        worker.record_frame(0.5);
        assert_eq!(worker.total_frames_rendered, 1);
        assert_eq!(worker.total_render_time, 0.5);

        worker.record_frame(1.0);
        assert_eq!(worker.total_frames_rendered, 2);
        assert_eq!(worker.total_render_time, 1.5);
    }

    #[test]
    fn test_worker_health_score() {
        let registration = create_test_registration();
        let mut worker = Worker::new(registration);

        // Initial score
        assert_eq!(worker.health.score, 1.0);

        // Record errors
        worker.record_error("Error".to_string());
        assert!(worker.health.score < 1.0);

        // Clear errors
        worker.clear_errors();
        assert_eq!(worker.health.score, 1.0);
    }

    #[test]
    fn test_worker_blacklist() {
        let registration = create_test_registration();
        let mut worker = Worker::new(registration);
        worker.state = WorkerState::Idle;

        worker.blacklist(chrono::Duration::minutes(5));
        assert!(worker.blacklisted_until.is_some());
        assert_eq!(worker.state, WorkerState::Error);
        assert!(!worker.is_available());
    }

    #[test]
    fn test_worker_performance_score() {
        let mut registration = create_test_registration();
        registration.capabilities.cpu_cores = 16;
        registration.capabilities.ram_gb = 64;
        registration.capabilities.has_gpu = true;

        let worker = Worker::new(registration);
        let score = worker.performance_score();
        assert!(score > 0.0);
    }

    #[test]
    fn test_worker_online_check() {
        let registration = create_test_registration();
        let mut worker = Worker::new(registration);

        // Just created - should be online
        assert!(worker.is_online());

        // Simulate old heartbeat
        worker.health.last_heartbeat = Utc::now() - chrono::Duration::minutes(2);
        assert!(!worker.is_online());
    }
}
