// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Real-time telemetry collection, anomaly detection, and Prometheus export.
//!
//! Provides lock-free per-worker metric counters, a rolling-window snapshot
//! store, z-score anomaly detection, and a Prometheus text-format exporter.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// WorkerTelemetry
// ---------------------------------------------------------------------------

/// Lock-free per-worker telemetry counters.
///
/// All fields use atomic integers so that a shared reference can be updated
/// from any thread without acquiring a mutex.
pub struct WorkerTelemetry {
    /// Worker identifier.
    pub worker_id: String,
    /// CPU utilisation in hundredths of a percent (0 = 0.00%, 10000 = 100.00%).
    pub cpu_usage_hundredths: AtomicU64,
    /// Memory usage in megabytes.
    pub memory_usage_mb: AtomicU64,
    /// GPU utilisation in hundredths of a percent (same scale as CPU).
    pub gpu_usage_hundredths: AtomicU64,
    /// Total frames rendered by this worker since boot.
    pub frames_rendered: AtomicU64,
    /// Total errors encountered by this worker since boot.
    pub errors_total: AtomicU64,
    /// Unix-millisecond timestamp of the last heartbeat.
    pub last_heartbeat_ms: AtomicI64,
    /// Total bytes read from storage.
    pub bytes_read: AtomicU64,
    /// Total bytes written to storage.
    pub bytes_written: AtomicU64,
    /// Number of jobs completed successfully.
    pub jobs_completed: AtomicU64,
    /// Number of jobs that failed.
    pub jobs_failed: AtomicU64,
    /// Current queue depth on this worker.
    pub local_queue_depth: AtomicU64,
    /// Average frame render time in milliseconds.
    pub avg_frame_time_ms: AtomicU64,
}

impl WorkerTelemetry {
    /// Create a zeroed telemetry block for the given worker.
    #[must_use]
    pub fn new(worker_id: impl Into<String>) -> Self {
        Self {
            worker_id: worker_id.into(),
            cpu_usage_hundredths: AtomicU64::new(0),
            memory_usage_mb: AtomicU64::new(0),
            gpu_usage_hundredths: AtomicU64::new(0),
            frames_rendered: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
            last_heartbeat_ms: AtomicI64::new(0),
            bytes_read: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            jobs_completed: AtomicU64::new(0),
            jobs_failed: AtomicU64::new(0),
            local_queue_depth: AtomicU64::new(0),
            avg_frame_time_ms: AtomicU64::new(0),
        }
    }

    /// Record a heartbeat at `now_ms`.
    pub fn heartbeat(&self, now_ms: i64) {
        self.last_heartbeat_ms.store(now_ms, Ordering::Relaxed);
    }

    /// Increment frames rendered by one.
    pub fn inc_frames_rendered(&self) {
        self.frames_rendered.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment error counter by one.
    pub fn inc_errors(&self) {
        self.errors_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Update CPU usage (in hundredths of percent).
    pub fn set_cpu_usage(&self, hundredths: u64) {
        self.cpu_usage_hundredths
            .store(hundredths, Ordering::Relaxed);
    }

    /// Update memory usage in MB.
    pub fn set_memory_usage_mb(&self, mb: u64) {
        self.memory_usage_mb.store(mb, Ordering::Relaxed);
    }

    /// Update GPU usage (in hundredths of percent).
    pub fn set_gpu_usage(&self, hundredths: u64) {
        self.gpu_usage_hundredths
            .store(hundredths, Ordering::Relaxed);
    }

    /// Set the average frame render time in milliseconds.
    pub fn set_avg_frame_time_ms(&self, ms: u64) {
        self.avg_frame_time_ms.store(ms, Ordering::Relaxed);
    }

    /// Record a completed job.
    pub fn inc_jobs_completed(&self) {
        self.jobs_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a failed job.
    pub fn inc_jobs_failed(&self) {
        self.jobs_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Add bytes read.
    pub fn add_bytes_read(&self, bytes: u64) {
        self.bytes_read.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Add bytes written.
    pub fn add_bytes_written(&self, bytes: u64) {
        self.bytes_written.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Set the local queue depth.
    pub fn set_local_queue_depth(&self, depth: u64) {
        self.local_queue_depth.store(depth, Ordering::Relaxed);
    }

    /// Take a point-in-time snapshot of all counters.
    #[must_use]
    pub fn snapshot(&self) -> WorkerSnapshot {
        WorkerSnapshot {
            worker_id: self.worker_id.clone(),
            cpu_usage_hundredths: self.cpu_usage_hundredths.load(Ordering::Relaxed),
            memory_usage_mb: self.memory_usage_mb.load(Ordering::Relaxed),
            gpu_usage_hundredths: self.gpu_usage_hundredths.load(Ordering::Relaxed),
            frames_rendered: self.frames_rendered.load(Ordering::Relaxed),
            errors_total: self.errors_total.load(Ordering::Relaxed),
            last_heartbeat_ms: self.last_heartbeat_ms.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            jobs_completed: self.jobs_completed.load(Ordering::Relaxed),
            jobs_failed: self.jobs_failed.load(Ordering::Relaxed),
            local_queue_depth: self.local_queue_depth.load(Ordering::Relaxed),
            avg_frame_time_ms: self.avg_frame_time_ms.load(Ordering::Relaxed),
        }
    }
}

impl std::fmt::Debug for WorkerTelemetry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerTelemetry")
            .field("worker_id", &self.worker_id)
            .field(
                "cpu_hundredths",
                &self.cpu_usage_hundredths.load(Ordering::Relaxed),
            )
            .field("memory_mb", &self.memory_usage_mb.load(Ordering::Relaxed))
            .field("frames", &self.frames_rendered.load(Ordering::Relaxed))
            .field("errors", &self.errors_total.load(Ordering::Relaxed))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// WorkerSnapshot
// ---------------------------------------------------------------------------

/// Serialisable point-in-time snapshot of a single worker's counters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerSnapshot {
    pub worker_id: String,
    pub cpu_usage_hundredths: u64,
    pub memory_usage_mb: u64,
    pub gpu_usage_hundredths: u64,
    pub frames_rendered: u64,
    pub errors_total: u64,
    pub last_heartbeat_ms: i64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub jobs_completed: u64,
    pub jobs_failed: u64,
    pub local_queue_depth: u64,
    pub avg_frame_time_ms: u64,
}

impl WorkerSnapshot {
    /// CPU usage as a floating-point percentage (0.0–100.0).
    #[must_use]
    pub fn cpu_pct(&self) -> f64 {
        self.cpu_usage_hundredths as f64 / 100.0
    }

    /// GPU usage as a floating-point percentage (0.0–100.0).
    #[must_use]
    pub fn gpu_pct(&self) -> f64 {
        self.gpu_usage_hundredths as f64 / 100.0
    }

    /// Error rate = errors / (errors + frames), or 0.0 if both are zero.
    #[must_use]
    pub fn error_rate(&self) -> f64 {
        let total = self.errors_total + self.frames_rendered;
        if total == 0 {
            return 0.0;
        }
        self.errors_total as f64 / total as f64
    }
}

// ---------------------------------------------------------------------------
// FarmSnapshot
// ---------------------------------------------------------------------------

/// Aggregated point-in-time snapshot of the entire farm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FarmSnapshot {
    /// Unix-millisecond timestamp when this snapshot was taken.
    pub timestamp_ms: i64,
    /// Per-worker snapshots.
    pub workers: Vec<WorkerSnapshot>,
    /// Total active workers in the farm.
    pub total_workers: u32,
    /// Global job queue depth.
    pub global_queue_depth: u32,
    /// Total frames rendered across all workers.
    pub total_frames_rendered: u64,
    /// Total errors across all workers.
    pub total_errors: u64,
}

impl FarmSnapshot {
    /// Build a farm snapshot from individual worker snapshots.
    #[must_use]
    pub fn from_workers(
        workers: Vec<WorkerSnapshot>,
        timestamp_ms: i64,
        global_queue_depth: u32,
    ) -> Self {
        let total_workers = workers.len() as u32;
        let total_frames_rendered: u64 = workers.iter().map(|w| w.frames_rendered).sum();
        let total_errors: u64 = workers.iter().map(|w| w.errors_total).sum();
        Self {
            timestamp_ms,
            workers,
            total_workers,
            global_queue_depth,
            total_frames_rendered,
            total_errors,
        }
    }

    /// Average CPU usage across all workers (percentage, 0.0–100.0).
    #[must_use]
    pub fn avg_cpu_pct(&self) -> f64 {
        if self.workers.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.workers.iter().map(|w| w.cpu_pct()).sum();
        sum / self.workers.len() as f64
    }

    /// Average memory usage across all workers (in MB).
    #[must_use]
    pub fn avg_memory_mb(&self) -> f64 {
        if self.workers.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.workers.iter().map(|w| w.memory_usage_mb).sum();
        sum as f64 / self.workers.len() as f64
    }

    /// Average GPU usage across all workers (percentage, 0.0–100.0).
    #[must_use]
    pub fn avg_gpu_pct(&self) -> f64 {
        if self.workers.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.workers.iter().map(|w| w.gpu_pct()).sum();
        sum / self.workers.len() as f64
    }

    /// Farm-wide error rate.
    #[must_use]
    pub fn error_rate(&self) -> f64 {
        let total = self.total_errors + self.total_frames_rendered;
        if total == 0 {
            return 0.0;
        }
        self.total_errors as f64 / total as f64
    }

    /// Average frame render time across workers that have rendered at least one frame.
    #[must_use]
    pub fn avg_frame_time_ms(&self) -> f64 {
        let active: Vec<&WorkerSnapshot> = self
            .workers
            .iter()
            .filter(|w| w.frames_rendered > 0)
            .collect();
        if active.is_empty() {
            return 0.0;
        }
        let sum: u64 = active.iter().map(|w| w.avg_frame_time_ms).sum();
        sum as f64 / active.len() as f64
    }
}

// ---------------------------------------------------------------------------
// TelemetryStore
// ---------------------------------------------------------------------------

/// Rolling-window store that keeps the last `capacity` farm snapshots.
///
/// Default capacity is 1440 (24 hours at 1-minute intervals).
#[derive(Debug)]
pub struct TelemetryStore {
    /// The ring buffer of farm snapshots.
    snapshots: VecDeque<FarmSnapshot>,
    /// Maximum number of snapshots to retain.
    capacity: usize,
}

impl TelemetryStore {
    /// Create a store with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            snapshots: VecDeque::with_capacity(capacity.min(4096)),
            capacity,
        }
    }

    /// Create a store with the default capacity of 1440 snapshots.
    #[must_use]
    pub fn default_capacity() -> Self {
        Self::new(1440)
    }

    /// Push a new snapshot, evicting the oldest if at capacity.
    pub fn push(&mut self, snapshot: FarmSnapshot) {
        if self.snapshots.len() >= self.capacity {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(snapshot);
    }

    /// Number of snapshots currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Whether the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Maximum capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the most recent snapshot, if any.
    #[must_use]
    pub fn latest(&self) -> Option<&FarmSnapshot> {
        self.snapshots.back()
    }

    /// Get the oldest snapshot in the window.
    #[must_use]
    pub fn oldest(&self) -> Option<&FarmSnapshot> {
        self.snapshots.front()
    }

    /// Iterate over all snapshots (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &FarmSnapshot> {
        self.snapshots.iter()
    }

    /// Return the last `n` snapshots (oldest first).
    #[must_use]
    pub fn last_n(&self, n: usize) -> Vec<&FarmSnapshot> {
        let start = self.snapshots.len().saturating_sub(n);
        self.snapshots.iter().skip(start).collect()
    }

    /// Rolling mean of total_errors across stored snapshots.
    #[must_use]
    pub fn rolling_mean_errors(&self) -> f64 {
        if self.snapshots.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.snapshots.iter().map(|s| s.total_errors).sum();
        sum as f64 / self.snapshots.len() as f64
    }

    /// Rolling mean of total_frames_rendered across stored snapshots.
    #[must_use]
    pub fn rolling_mean_frames(&self) -> f64 {
        if self.snapshots.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.snapshots.iter().map(|s| s.total_frames_rendered).sum();
        sum as f64 / self.snapshots.len() as f64
    }

    /// Rolling mean of avg CPU pct across stored snapshots.
    #[must_use]
    pub fn rolling_mean_cpu_pct(&self) -> f64 {
        if self.snapshots.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.snapshots.iter().map(|s| s.avg_cpu_pct()).sum();
        sum / self.snapshots.len() as f64
    }

    /// Standard deviation of total_errors across stored snapshots.
    #[must_use]
    pub fn stddev_errors(&self) -> f64 {
        compute_stddev(self.snapshots.iter().map(|s| s.total_errors as f64))
    }

    /// Standard deviation of total_frames across stored snapshots.
    #[must_use]
    pub fn stddev_frames(&self) -> f64 {
        compute_stddev(
            self.snapshots
                .iter()
                .map(|s| s.total_frames_rendered as f64),
        )
    }

    /// Standard deviation of avg CPU pct across stored snapshots.
    #[must_use]
    pub fn stddev_cpu_pct(&self) -> f64 {
        compute_stddev(self.snapshots.iter().map(|s| s.avg_cpu_pct()))
    }

    /// Detect anomalies in the most recent snapshot using z-scores.
    ///
    /// A metric is flagged when its z-score exceeds `threshold` (typically 3.0).
    /// Returns an empty vector if there are fewer than 2 snapshots.
    #[must_use]
    pub fn detect_anomalies(&self, threshold: f64) -> Vec<AnomalyAlert> {
        if self.snapshots.len() < 2 {
            return Vec::new();
        }

        let latest = match self.snapshots.back() {
            Some(s) => s,
            None => return Vec::new(),
        };

        let mut alerts = Vec::new();

        // Check errors
        let mean_err = self.rolling_mean_errors();
        let std_err = self.stddev_errors();
        if std_err > 0.0 {
            let z = (latest.total_errors as f64 - mean_err) / std_err;
            if z.abs() > threshold {
                alerts.push(AnomalyAlert {
                    metric: "total_errors".to_string(),
                    value: latest.total_errors as f64,
                    mean: mean_err,
                    stddev: std_err,
                    z_score: z,
                    timestamp_ms: latest.timestamp_ms,
                    severity: anomaly_severity(z),
                });
            }
        }

        // Check frames
        let mean_frames = self.rolling_mean_frames();
        let std_frames = self.stddev_frames();
        if std_frames > 0.0 {
            let z = (latest.total_frames_rendered as f64 - mean_frames) / std_frames;
            if z.abs() > threshold {
                alerts.push(AnomalyAlert {
                    metric: "total_frames_rendered".to_string(),
                    value: latest.total_frames_rendered as f64,
                    mean: mean_frames,
                    stddev: std_frames,
                    z_score: z,
                    timestamp_ms: latest.timestamp_ms,
                    severity: anomaly_severity(z),
                });
            }
        }

        // Check CPU
        let mean_cpu = self.rolling_mean_cpu_pct();
        let std_cpu = self.stddev_cpu_pct();
        if std_cpu > 0.0 {
            let z = (latest.avg_cpu_pct() - mean_cpu) / std_cpu;
            if z.abs() > threshold {
                alerts.push(AnomalyAlert {
                    metric: "avg_cpu_pct".to_string(),
                    value: latest.avg_cpu_pct(),
                    mean: mean_cpu,
                    stddev: std_cpu,
                    z_score: z,
                    timestamp_ms: latest.timestamp_ms,
                    severity: anomaly_severity(z),
                });
            }
        }

        // Check queue depth
        let mean_q = self.rolling_mean_queue_depth();
        let std_q = self.stddev_queue_depth();
        if std_q > 0.0 {
            let z = (latest.global_queue_depth as f64 - mean_q) / std_q;
            if z.abs() > threshold {
                alerts.push(AnomalyAlert {
                    metric: "global_queue_depth".to_string(),
                    value: latest.global_queue_depth as f64,
                    mean: mean_q,
                    stddev: std_q,
                    z_score: z,
                    timestamp_ms: latest.timestamp_ms,
                    severity: anomaly_severity(z),
                });
            }
        }

        alerts
    }

    /// Rolling mean of global_queue_depth across stored snapshots.
    #[must_use]
    pub fn rolling_mean_queue_depth(&self) -> f64 {
        if self.snapshots.is_empty() {
            return 0.0;
        }
        let sum: u64 = self
            .snapshots
            .iter()
            .map(|s| u64::from(s.global_queue_depth))
            .sum();
        sum as f64 / self.snapshots.len() as f64
    }

    /// Standard deviation of global_queue_depth across stored snapshots.
    #[must_use]
    pub fn stddev_queue_depth(&self) -> f64 {
        compute_stddev(self.snapshots.iter().map(|s| s.global_queue_depth as f64))
    }
}

impl Default for TelemetryStore {
    fn default() -> Self {
        Self::default_capacity()
    }
}

// ---------------------------------------------------------------------------
// AnomalyAlert
// ---------------------------------------------------------------------------

/// An alert raised when a metric deviates significantly from its rolling mean.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyAlert {
    /// Name of the metric that triggered the alert.
    pub metric: String,
    /// The observed value that was anomalous.
    pub value: f64,
    /// Rolling mean of the metric at the time of detection.
    pub mean: f64,
    /// Standard deviation of the metric at the time of detection.
    pub stddev: f64,
    /// The z-score (positive = above mean, negative = below).
    pub z_score: f64,
    /// Timestamp of the snapshot that triggered the alert.
    pub timestamp_ms: i64,
    /// Severity classification derived from z-score magnitude.
    pub severity: AnomalySeverity,
}

impl AnomalyAlert {
    /// Human-readable description of the anomaly.
    #[must_use]
    pub fn description(&self) -> String {
        format!(
            "{} anomaly: value={:.2}, mean={:.2}, stddev={:.2}, z={:.2} ({})",
            self.metric,
            self.value,
            self.mean,
            self.stddev,
            self.z_score,
            self.severity.label(),
        )
    }
}

/// Severity of an anomaly based on z-score magnitude.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnomalySeverity {
    /// z-score between 3 and 4.
    Warning,
    /// z-score between 4 and 5.
    High,
    /// z-score above 5.
    Critical,
}

impl AnomalySeverity {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Warning => "warning",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }
}

/// Map absolute z-score to severity.
fn anomaly_severity(z: f64) -> AnomalySeverity {
    let abs_z = z.abs();
    if abs_z >= 5.0 {
        AnomalySeverity::Critical
    } else if abs_z >= 4.0 {
        AnomalySeverity::High
    } else {
        AnomalySeverity::Warning
    }
}

// ---------------------------------------------------------------------------
// Prometheus export
// ---------------------------------------------------------------------------

/// Format the latest `FarmSnapshot` as Prometheus text exposition format.
///
/// Returns an empty string if the snapshot is `None`.
#[must_use]
pub fn to_prometheus(snapshot: Option<&FarmSnapshot>) -> String {
    let snapshot = match snapshot {
        Some(s) => s,
        None => return String::new(),
    };

    let mut out = String::with_capacity(4096);

    // Farm-level gauges
    out.push_str("# HELP oximedia_renderfarm_workers_total Total number of active workers.\n");
    out.push_str("# TYPE oximedia_renderfarm_workers_total gauge\n");
    push_metric_line(
        &mut out,
        "oximedia_renderfarm_workers_total",
        &[],
        snapshot.total_workers as f64,
    );

    out.push_str("# HELP oximedia_renderfarm_queue_depth Global job queue depth.\n");
    out.push_str("# TYPE oximedia_renderfarm_queue_depth gauge\n");
    push_metric_line(
        &mut out,
        "oximedia_renderfarm_queue_depth",
        &[],
        snapshot.global_queue_depth as f64,
    );

    out.push_str("# HELP oximedia_renderfarm_frames_total Total frames rendered.\n");
    out.push_str("# TYPE oximedia_renderfarm_frames_total counter\n");
    push_metric_line(
        &mut out,
        "oximedia_renderfarm_frames_total",
        &[],
        snapshot.total_frames_rendered as f64,
    );

    out.push_str("# HELP oximedia_renderfarm_errors_total Total errors.\n");
    out.push_str("# TYPE oximedia_renderfarm_errors_total counter\n");
    push_metric_line(
        &mut out,
        "oximedia_renderfarm_errors_total",
        &[],
        snapshot.total_errors as f64,
    );

    out.push_str("# HELP oximedia_renderfarm_avg_cpu_pct Average CPU usage percentage.\n");
    out.push_str("# TYPE oximedia_renderfarm_avg_cpu_pct gauge\n");
    push_metric_line(
        &mut out,
        "oximedia_renderfarm_avg_cpu_pct",
        &[],
        snapshot.avg_cpu_pct(),
    );

    out.push_str("# HELP oximedia_renderfarm_avg_memory_mb Average memory usage in MB.\n");
    out.push_str("# TYPE oximedia_renderfarm_avg_memory_mb gauge\n");
    push_metric_line(
        &mut out,
        "oximedia_renderfarm_avg_memory_mb",
        &[],
        snapshot.avg_memory_mb(),
    );

    // Per-worker metrics
    out.push_str("# HELP oximedia_renderfarm_worker_cpu_pct Per-worker CPU usage percentage.\n");
    out.push_str("# TYPE oximedia_renderfarm_worker_cpu_pct gauge\n");
    for w in &snapshot.workers {
        push_metric_line(
            &mut out,
            "oximedia_renderfarm_worker_cpu_pct",
            &[("worker", &w.worker_id)],
            w.cpu_pct(),
        );
    }

    out.push_str("# HELP oximedia_renderfarm_worker_memory_mb Per-worker memory usage.\n");
    out.push_str("# TYPE oximedia_renderfarm_worker_memory_mb gauge\n");
    for w in &snapshot.workers {
        push_metric_line(
            &mut out,
            "oximedia_renderfarm_worker_memory_mb",
            &[("worker", &w.worker_id)],
            w.memory_usage_mb as f64,
        );
    }

    out.push_str("# HELP oximedia_renderfarm_worker_gpu_pct Per-worker GPU usage percentage.\n");
    out.push_str("# TYPE oximedia_renderfarm_worker_gpu_pct gauge\n");
    for w in &snapshot.workers {
        push_metric_line(
            &mut out,
            "oximedia_renderfarm_worker_gpu_pct",
            &[("worker", &w.worker_id)],
            w.gpu_pct(),
        );
    }

    out.push_str("# HELP oximedia_renderfarm_worker_frames_total Per-worker frames rendered.\n");
    out.push_str("# TYPE oximedia_renderfarm_worker_frames_total counter\n");
    for w in &snapshot.workers {
        push_metric_line(
            &mut out,
            "oximedia_renderfarm_worker_frames_total",
            &[("worker", &w.worker_id)],
            w.frames_rendered as f64,
        );
    }

    out.push_str("# HELP oximedia_renderfarm_worker_errors_total Per-worker error count.\n");
    out.push_str("# TYPE oximedia_renderfarm_worker_errors_total counter\n");
    for w in &snapshot.workers {
        push_metric_line(
            &mut out,
            "oximedia_renderfarm_worker_errors_total",
            &[("worker", &w.worker_id)],
            w.errors_total as f64,
        );
    }

    out.push_str("# HELP oximedia_renderfarm_worker_queue_depth Per-worker local queue depth.\n");
    out.push_str("# TYPE oximedia_renderfarm_worker_queue_depth gauge\n");
    for w in &snapshot.workers {
        push_metric_line(
            &mut out,
            "oximedia_renderfarm_worker_queue_depth",
            &[("worker", &w.worker_id)],
            w.local_queue_depth as f64,
        );
    }

    out.push_str(
        "# HELP oximedia_renderfarm_worker_avg_frame_time_ms Per-worker avg frame time.\n",
    );
    out.push_str("# TYPE oximedia_renderfarm_worker_avg_frame_time_ms gauge\n");
    for w in &snapshot.workers {
        push_metric_line(
            &mut out,
            "oximedia_renderfarm_worker_avg_frame_time_ms",
            &[("worker", &w.worker_id)],
            w.avg_frame_time_ms as f64,
        );
    }

    out
}

/// Append a single metric line in Prometheus text format.
fn push_metric_line(out: &mut String, name: &str, labels: &[(&str, &str)], value: f64) {
    out.push_str(name);
    if !labels.is_empty() {
        out.push('{');
        for (i, (k, v)) in labels.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(k);
            out.push_str("=\"");
            out.push_str(v);
            out.push('"');
        }
        out.push('}');
    }
    out.push(' ');
    // Use integer formatting when the value has no fractional part.
    if value == value.floor() && value.is_finite() {
        out.push_str(&format!("{}", value as i64));
    } else {
        out.push_str(&format!("{value:.6}"));
    }
    out.push('\n');
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the population standard deviation of an iterator of f64 values.
fn compute_stddev(values: impl Iterator<Item = f64>) -> f64 {
    let data: Vec<f64> = values.collect();
    if data.len() < 2 {
        return 0.0;
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    var.sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_worker_snapshot(
        worker_id: &str,
        cpu_hundredths: u64,
        mem_mb: u64,
        gpu_hundredths: u64,
        frames: u64,
        errors: u64,
    ) -> WorkerSnapshot {
        WorkerSnapshot {
            worker_id: worker_id.to_string(),
            cpu_usage_hundredths: cpu_hundredths,
            memory_usage_mb: mem_mb,
            gpu_usage_hundredths: gpu_hundredths,
            frames_rendered: frames,
            errors_total: errors,
            last_heartbeat_ms: 0,
            bytes_read: 0,
            bytes_written: 0,
            jobs_completed: frames,
            jobs_failed: errors,
            local_queue_depth: 0,
            avg_frame_time_ms: 100,
        }
    }

    fn make_farm_snapshot(
        timestamp: i64,
        workers: Vec<WorkerSnapshot>,
        queue: u32,
    ) -> FarmSnapshot {
        FarmSnapshot::from_workers(workers, timestamp, queue)
    }

    // --- WorkerTelemetry ---

    #[test]
    fn test_worker_telemetry_new_zeroed() {
        let t = WorkerTelemetry::new("w1");
        assert_eq!(t.worker_id, "w1");
        assert_eq!(t.cpu_usage_hundredths.load(Ordering::Relaxed), 0);
        assert_eq!(t.frames_rendered.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_worker_telemetry_inc_frames() {
        let t = WorkerTelemetry::new("w1");
        t.inc_frames_rendered();
        t.inc_frames_rendered();
        assert_eq!(t.frames_rendered.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_worker_telemetry_inc_errors() {
        let t = WorkerTelemetry::new("w1");
        t.inc_errors();
        assert_eq!(t.errors_total.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_worker_telemetry_set_cpu() {
        let t = WorkerTelemetry::new("w1");
        t.set_cpu_usage(5000);
        assert_eq!(t.cpu_usage_hundredths.load(Ordering::Relaxed), 5000);
    }

    #[test]
    fn test_worker_telemetry_heartbeat() {
        let t = WorkerTelemetry::new("w1");
        t.heartbeat(12345);
        assert_eq!(t.last_heartbeat_ms.load(Ordering::Relaxed), 12345);
    }

    #[test]
    fn test_worker_telemetry_snapshot() {
        let t = WorkerTelemetry::new("w1");
        t.set_cpu_usage(7500);
        t.set_memory_usage_mb(2048);
        t.inc_frames_rendered();
        let snap = t.snapshot();
        assert_eq!(snap.worker_id, "w1");
        assert_eq!(snap.cpu_usage_hundredths, 7500);
        assert_eq!(snap.memory_usage_mb, 2048);
        assert_eq!(snap.frames_rendered, 1);
    }

    #[test]
    fn test_worker_telemetry_bytes_tracking() {
        let t = WorkerTelemetry::new("w1");
        t.add_bytes_read(1024);
        t.add_bytes_written(512);
        let snap = t.snapshot();
        assert_eq!(snap.bytes_read, 1024);
        assert_eq!(snap.bytes_written, 512);
    }

    #[test]
    fn test_worker_telemetry_jobs_tracking() {
        let t = WorkerTelemetry::new("w1");
        t.inc_jobs_completed();
        t.inc_jobs_completed();
        t.inc_jobs_failed();
        let snap = t.snapshot();
        assert_eq!(snap.jobs_completed, 2);
        assert_eq!(snap.jobs_failed, 1);
    }

    // --- WorkerSnapshot ---

    #[test]
    fn test_worker_snapshot_cpu_pct() {
        let snap = make_worker_snapshot("w1", 5000, 1024, 0, 10, 0);
        assert!((snap.cpu_pct() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_worker_snapshot_gpu_pct() {
        let snap = make_worker_snapshot("w1", 0, 0, 9900, 0, 0);
        assert!((snap.gpu_pct() - 99.0).abs() < 0.01);
    }

    #[test]
    fn test_worker_snapshot_error_rate_zero() {
        let snap = make_worker_snapshot("w1", 0, 0, 0, 0, 0);
        assert_eq!(snap.error_rate(), 0.0);
    }

    #[test]
    fn test_worker_snapshot_error_rate_nonzero() {
        let snap = make_worker_snapshot("w1", 0, 0, 0, 90, 10);
        assert!((snap.error_rate() - 0.1).abs() < 0.001);
    }

    // --- FarmSnapshot ---

    #[test]
    fn test_farm_snapshot_from_workers() {
        let ws = vec![
            make_worker_snapshot("w1", 5000, 1024, 0, 100, 5),
            make_worker_snapshot("w2", 7500, 2048, 0, 200, 10),
        ];
        let farm = FarmSnapshot::from_workers(ws, 1000, 42);
        assert_eq!(farm.total_workers, 2);
        assert_eq!(farm.total_frames_rendered, 300);
        assert_eq!(farm.total_errors, 15);
        assert_eq!(farm.global_queue_depth, 42);
    }

    #[test]
    fn test_farm_snapshot_avg_cpu() {
        let ws = vec![
            make_worker_snapshot("w1", 4000, 0, 0, 0, 0),
            make_worker_snapshot("w2", 8000, 0, 0, 0, 0),
        ];
        let farm = FarmSnapshot::from_workers(ws, 0, 0);
        // (40.0 + 80.0) / 2 = 60.0
        assert!((farm.avg_cpu_pct() - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_farm_snapshot_avg_memory() {
        let ws = vec![
            make_worker_snapshot("w1", 0, 1000, 0, 0, 0),
            make_worker_snapshot("w2", 0, 3000, 0, 0, 0),
        ];
        let farm = FarmSnapshot::from_workers(ws, 0, 0);
        assert!((farm.avg_memory_mb() - 2000.0).abs() < 0.01);
    }

    #[test]
    fn test_farm_snapshot_empty_workers() {
        let farm = FarmSnapshot::from_workers(vec![], 0, 0);
        assert_eq!(farm.avg_cpu_pct(), 0.0);
        assert_eq!(farm.avg_memory_mb(), 0.0);
        assert_eq!(farm.avg_gpu_pct(), 0.0);
        assert_eq!(farm.error_rate(), 0.0);
    }

    #[test]
    fn test_farm_snapshot_error_rate() {
        let ws = vec![make_worker_snapshot("w1", 0, 0, 0, 80, 20)];
        let farm = FarmSnapshot::from_workers(ws, 0, 0);
        assert!((farm.error_rate() - 0.2).abs() < 0.001);
    }

    // --- TelemetryStore ---

    #[test]
    fn test_store_push_and_len() {
        let mut store = TelemetryStore::new(10);
        assert!(store.is_empty());
        store.push(make_farm_snapshot(1, vec![], 0));
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_store_capacity_eviction() {
        let mut store = TelemetryStore::new(3);
        for i in 0..5 {
            store.push(make_farm_snapshot(i, vec![], 0));
        }
        assert_eq!(store.len(), 3);
        // Oldest should be timestamp 2
        assert_eq!(store.oldest().map(|s| s.timestamp_ms), Some(2));
        assert_eq!(store.latest().map(|s| s.timestamp_ms), Some(4));
    }

    #[test]
    fn test_store_last_n() {
        let mut store = TelemetryStore::new(100);
        for i in 0..10 {
            store.push(make_farm_snapshot(i, vec![], 0));
        }
        let last3 = store.last_n(3);
        assert_eq!(last3.len(), 3);
        assert_eq!(last3[0].timestamp_ms, 7);
        assert_eq!(last3[2].timestamp_ms, 9);
    }

    #[test]
    fn test_store_rolling_mean_errors() {
        let mut store = TelemetryStore::new(100);
        // Push 3 snapshots with 10, 20, 30 errors
        store.push(make_farm_snapshot(
            0,
            vec![make_worker_snapshot("w", 0, 0, 0, 0, 10)],
            0,
        ));
        store.push(make_farm_snapshot(
            1,
            vec![make_worker_snapshot("w", 0, 0, 0, 0, 20)],
            0,
        ));
        store.push(make_farm_snapshot(
            2,
            vec![make_worker_snapshot("w", 0, 0, 0, 0, 30)],
            0,
        ));
        assert!((store.rolling_mean_errors() - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_store_rolling_mean_frames() {
        let mut store = TelemetryStore::new(100);
        store.push(make_farm_snapshot(
            0,
            vec![make_worker_snapshot("w", 0, 0, 0, 100, 0)],
            0,
        ));
        store.push(make_farm_snapshot(
            1,
            vec![make_worker_snapshot("w", 0, 0, 0, 200, 0)],
            0,
        ));
        assert!((store.rolling_mean_frames() - 150.0).abs() < 0.01);
    }

    #[test]
    fn test_store_empty_rolling_means() {
        let store = TelemetryStore::new(10);
        assert_eq!(store.rolling_mean_errors(), 0.0);
        assert_eq!(store.rolling_mean_frames(), 0.0);
        assert_eq!(store.rolling_mean_cpu_pct(), 0.0);
    }

    // --- Anomaly detection ---

    #[test]
    fn test_detect_anomalies_insufficient_data() {
        let mut store = TelemetryStore::new(100);
        store.push(make_farm_snapshot(0, vec![], 0));
        let alerts = store.detect_anomalies(3.0);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_detect_anomalies_no_anomaly() {
        let mut store = TelemetryStore::new(100);
        // All same errors → stddev = 0, no anomaly reported
        for i in 0..10 {
            store.push(make_farm_snapshot(
                i,
                vec![make_worker_snapshot("w", 5000, 1024, 0, 100, 5)],
                10,
            ));
        }
        let alerts = store.detect_anomalies(3.0);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_detect_anomalies_error_spike() {
        let mut store = TelemetryStore::new(100);
        // 9 normal snapshots with 1 error each
        for i in 0..9 {
            store.push(make_farm_snapshot(
                i,
                vec![make_worker_snapshot("w", 5000, 1024, 0, 100, 1)],
                10,
            ));
        }
        // 1 anomalous snapshot with 100 errors
        store.push(make_farm_snapshot(
            9,
            vec![make_worker_snapshot("w", 5000, 1024, 0, 100, 100)],
            10,
        ));
        let alerts = store.detect_anomalies(3.0);
        assert!(!alerts.is_empty());
        let err_alert = alerts.iter().find(|a| a.metric == "total_errors");
        assert!(err_alert.is_some());
        let alert = err_alert.expect("just checked");
        assert!(alert.z_score > 3.0);
    }

    #[test]
    fn test_anomaly_severity_levels() {
        assert_eq!(anomaly_severity(3.5), AnomalySeverity::Warning);
        assert_eq!(anomaly_severity(4.5), AnomalySeverity::High);
        assert_eq!(anomaly_severity(5.5), AnomalySeverity::Critical);
        assert_eq!(anomaly_severity(-6.0), AnomalySeverity::Critical);
    }

    #[test]
    fn test_anomaly_alert_description() {
        let alert = AnomalyAlert {
            metric: "test_metric".to_string(),
            value: 100.0,
            mean: 10.0,
            stddev: 5.0,
            z_score: 18.0,
            timestamp_ms: 12345,
            severity: AnomalySeverity::Critical,
        };
        let desc = alert.description();
        assert!(desc.contains("test_metric"));
        assert!(desc.contains("critical"));
    }

    // --- Prometheus export ---

    #[test]
    fn test_prometheus_none_returns_empty() {
        assert!(to_prometheus(None).is_empty());
    }

    #[test]
    fn test_prometheus_basic_output() {
        let ws = vec![make_worker_snapshot("w1", 5000, 2048, 3000, 42, 3)];
        let farm = FarmSnapshot::from_workers(ws, 1000, 10);
        let output = to_prometheus(Some(&farm));

        assert!(output.contains("oximedia_renderfarm_workers_total 1\n"));
        assert!(output.contains("oximedia_renderfarm_queue_depth 10\n"));
        assert!(output.contains("oximedia_renderfarm_frames_total 42\n"));
        assert!(output.contains("oximedia_renderfarm_errors_total 3\n"));
        assert!(output.contains("oximedia_renderfarm_worker_cpu_pct{worker=\"w1\"}"));
        assert!(output.contains("# HELP"));
        assert!(output.contains("# TYPE"));
    }

    #[test]
    fn test_prometheus_multiple_workers() {
        let ws = vec![
            make_worker_snapshot("w1", 5000, 1024, 0, 10, 1),
            make_worker_snapshot("w2", 8000, 2048, 0, 20, 2),
        ];
        let farm = FarmSnapshot::from_workers(ws, 0, 5);
        let output = to_prometheus(Some(&farm));
        assert!(output.contains("worker=\"w1\""));
        assert!(output.contains("worker=\"w2\""));
    }

    #[test]
    fn test_prometheus_empty_farm() {
        let farm = FarmSnapshot::from_workers(vec![], 0, 0);
        let output = to_prometheus(Some(&farm));
        assert!(output.contains("oximedia_renderfarm_workers_total 0\n"));
    }

    // --- compute_stddev ---

    #[test]
    fn test_stddev_empty() {
        assert_eq!(compute_stddev(std::iter::empty()), 0.0);
    }

    #[test]
    fn test_stddev_single() {
        assert_eq!(compute_stddev(std::iter::once(42.0)), 0.0);
    }

    #[test]
    fn test_stddev_known_values() {
        // [2, 4, 4, 4, 5, 5, 7, 9] => mean=5, var=4, std=2
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = compute_stddev(data.into_iter());
        assert!((std - 2.0).abs() < 0.001);
    }

    // --- TelemetryStore default ---

    #[test]
    fn test_store_default_capacity() {
        let store = TelemetryStore::default();
        assert_eq!(store.capacity(), 1440);
    }

    // --- FarmSnapshot avg_frame_time ---

    #[test]
    fn test_farm_snapshot_avg_frame_time_no_active() {
        let ws = vec![make_worker_snapshot("w1", 0, 0, 0, 0, 0)];
        let farm = FarmSnapshot::from_workers(ws, 0, 0);
        assert_eq!(farm.avg_frame_time_ms(), 0.0);
    }

    #[test]
    fn test_farm_snapshot_avg_frame_time_with_active() {
        let ws = vec![
            make_worker_snapshot("w1", 0, 0, 0, 10, 0),
            make_worker_snapshot("w2", 0, 0, 0, 20, 0),
        ];
        let farm = FarmSnapshot::from_workers(ws, 0, 0);
        // Both have avg_frame_time_ms = 100
        assert!((farm.avg_frame_time_ms() - 100.0).abs() < 0.01);
    }
}
