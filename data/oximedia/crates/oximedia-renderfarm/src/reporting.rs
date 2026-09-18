// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Reporting and analytics.

use crate::job::JobId;
use crate::worker::WorkerId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Report type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportType {
    /// Job performance report
    JobPerformance,
    /// Worker utilization report
    WorkerUtilization,
    /// Cost analysis report
    CostAnalysis,
    /// Resource usage report
    ResourceUsage,
    /// Error analysis report
    ErrorAnalysis,
}

/// Job performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobPerformanceReport {
    /// Report period
    pub period_start: DateTime<Utc>,
    /// Period end
    pub period_end: DateTime<Utc>,
    /// Total jobs
    pub total_jobs: usize,
    /// Completed jobs
    pub completed_jobs: usize,
    /// Failed jobs
    pub failed_jobs: usize,
    /// Average completion time (seconds)
    pub avg_completion_time: f64,
    /// Average frames per hour
    pub avg_throughput: f64,
    /// Jobs by priority
    pub jobs_by_priority: HashMap<String, usize>,
}

/// Worker utilization report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerUtilizationReport {
    /// Report period
    pub period_start: DateTime<Utc>,
    /// Period end
    pub period_end: DateTime<Utc>,
    /// Total workers
    pub total_workers: usize,
    /// Average utilization (0.0 to 1.0)
    pub avg_utilization: f64,
    /// Utilization by worker
    pub utilization_by_worker: HashMap<WorkerId, f64>,
    /// Idle time by worker (seconds)
    pub idle_time_by_worker: HashMap<WorkerId, f64>,
}

/// Cost analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAnalysisReport {
    /// Report period
    pub period_start: DateTime<Utc>,
    /// Period end
    pub period_end: DateTime<Utc>,
    /// Total cost
    pub total_cost: f64,
    /// Cost per job
    pub cost_per_job: f64,
    /// Cost per frame
    pub cost_per_frame: f64,
    /// Cost by project
    pub cost_by_project: HashMap<String, f64>,
    /// Cost trend
    pub cost_trend: Vec<(DateTime<Utc>, f64)>,
}

/// Resource usage report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageReport {
    /// Report period
    pub period_start: DateTime<Utc>,
    /// Period end
    pub period_end: DateTime<Utc>,
    /// Average CPU usage (0.0 to 1.0)
    pub avg_cpu: f64,
    /// Average memory usage (0.0 to 1.0)
    pub avg_memory: f64,
    /// Average GPU usage (0.0 to 1.0)
    pub avg_gpu: Option<f64>,
    /// Peak CPU usage
    pub peak_cpu: f64,
    /// Peak memory usage
    pub peak_memory: f64,
}

/// Error analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAnalysisReport {
    /// Report period
    pub period_start: DateTime<Utc>,
    /// Period end
    pub period_end: DateTime<Utc>,
    /// Total errors
    pub total_errors: usize,
    /// Errors by type
    pub errors_by_type: HashMap<String, usize>,
    /// Errors by worker
    pub errors_by_worker: HashMap<WorkerId, usize>,
    /// Most common errors
    pub common_errors: Vec<(String, usize)>,
}

/// Report generator
pub struct ReportGenerator {
    job_data: Vec<JobDataPoint>,
    worker_data: Vec<WorkerDataPoint>,
    cost_data: Vec<CostDataPoint>,
    error_data: Vec<ErrorDataPoint>,
}

/// Job data point
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct JobDataPoint {
    job_id: JobId,
    completed_at: DateTime<Utc>,
    duration: f64,
    frames: u32,
    priority: String,
    success: bool,
}

/// Worker data point
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct WorkerDataPoint {
    worker_id: WorkerId,
    timestamp: DateTime<Utc>,
    cpu_usage: f64,
    memory_usage: f64,
    gpu_usage: Option<f64>,
}

/// Cost data point
#[derive(Debug, Clone)]
struct CostDataPoint {
    timestamp: DateTime<Utc>,
    project_id: String,
    cost: f64,
}

/// Error data point
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ErrorDataPoint {
    timestamp: DateTime<Utc>,
    worker_id: WorkerId,
    error_type: String,
    message: String,
}

impl ReportGenerator {
    /// Create a new report generator
    #[must_use]
    pub fn new() -> Self {
        Self {
            job_data: Vec::new(),
            worker_data: Vec::new(),
            cost_data: Vec::new(),
            error_data: Vec::new(),
        }
    }

    /// Record job completion
    pub fn record_job(
        &mut self,
        job_id: JobId,
        duration: f64,
        frames: u32,
        priority: String,
        success: bool,
    ) {
        self.job_data.push(JobDataPoint {
            job_id,
            completed_at: Utc::now(),
            duration,
            frames,
            priority,
            success,
        });
    }

    /// Record worker metrics
    pub fn record_worker_metrics(
        &mut self,
        worker_id: WorkerId,
        cpu: f64,
        memory: f64,
        gpu: Option<f64>,
    ) {
        self.worker_data.push(WorkerDataPoint {
            worker_id,
            timestamp: Utc::now(),
            cpu_usage: cpu,
            memory_usage: memory,
            gpu_usage: gpu,
        });
    }

    /// Record cost
    pub fn record_cost(&mut self, project_id: String, cost: f64) {
        self.cost_data.push(CostDataPoint {
            timestamp: Utc::now(),
            project_id,
            cost,
        });
    }

    /// Record error
    pub fn record_error(&mut self, worker_id: WorkerId, error_type: String, message: String) {
        self.error_data.push(ErrorDataPoint {
            timestamp: Utc::now(),
            worker_id,
            error_type,
            message,
        });
    }

    /// Generate job performance report
    #[must_use]
    pub fn generate_job_performance_report(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> JobPerformanceReport {
        let jobs: Vec<_> = self
            .job_data
            .iter()
            .filter(|j| j.completed_at >= start && j.completed_at <= end)
            .collect();

        let total_jobs = jobs.len();
        let completed_jobs = jobs.iter().filter(|j| j.success).count();
        let failed_jobs = total_jobs - completed_jobs;

        let avg_completion_time = if jobs.is_empty() {
            0.0
        } else {
            jobs.iter().map(|j| j.duration).sum::<f64>() / jobs.len() as f64
        };

        let avg_throughput = if jobs.is_empty() {
            0.0
        } else {
            let total_frames: u32 = jobs.iter().map(|j| j.frames).sum();
            let total_hours = jobs.iter().map(|j| j.duration).sum::<f64>() / 3600.0;
            if total_hours > 0.0 {
                f64::from(total_frames) / total_hours
            } else {
                0.0
            }
        };

        let mut jobs_by_priority = HashMap::new();
        for job in &jobs {
            *jobs_by_priority.entry(job.priority.clone()).or_insert(0) += 1;
        }

        JobPerformanceReport {
            period_start: start,
            period_end: end,
            total_jobs,
            completed_jobs,
            failed_jobs,
            avg_completion_time,
            avg_throughput,
            jobs_by_priority,
        }
    }

    /// Generate worker utilization report
    #[must_use]
    pub fn generate_worker_utilization_report(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> WorkerUtilizationReport {
        let metrics: Vec<_> = self
            .worker_data
            .iter()
            .filter(|m| m.timestamp >= start && m.timestamp <= end)
            .collect();

        let total_workers = metrics
            .iter()
            .map(|m| m.worker_id)
            .collect::<std::collections::HashSet<_>>()
            .len();

        let avg_utilization = if metrics.is_empty() {
            0.0
        } else {
            metrics.iter().map(|m| m.cpu_usage).sum::<f64>() / metrics.len() as f64
        };

        let mut utilization_by_worker = HashMap::new();
        let mut idle_time_by_worker = HashMap::new();

        for worker_id in metrics
            .iter()
            .map(|m| m.worker_id)
            .collect::<std::collections::HashSet<_>>()
        {
            let worker_metrics: Vec<_> = metrics
                .iter()
                .filter(|m| m.worker_id == worker_id)
                .collect();
            let avg_util = worker_metrics.iter().map(|m| m.cpu_usage).sum::<f64>()
                / worker_metrics.len() as f64;
            let idle_time =
                worker_metrics.iter().filter(|m| m.cpu_usage < 0.1).count() as f64 * 60.0;

            utilization_by_worker.insert(worker_id, avg_util);
            idle_time_by_worker.insert(worker_id, idle_time);
        }

        WorkerUtilizationReport {
            period_start: start,
            period_end: end,
            total_workers,
            avg_utilization,
            utilization_by_worker,
            idle_time_by_worker,
        }
    }

    /// Generate cost analysis report
    #[must_use]
    pub fn generate_cost_analysis_report(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> CostAnalysisReport {
        let costs: Vec<_> = self
            .cost_data
            .iter()
            .filter(|c| c.timestamp >= start && c.timestamp <= end)
            .collect();

        let total_cost: f64 = costs.iter().map(|c| c.cost).sum();

        let jobs: Vec<_> = self
            .job_data
            .iter()
            .filter(|j| j.completed_at >= start && j.completed_at <= end)
            .collect();

        let cost_per_job = if jobs.is_empty() {
            0.0
        } else {
            total_cost / jobs.len() as f64
        };

        let total_frames: u32 = jobs.iter().map(|j| j.frames).sum();
        let cost_per_frame = if total_frames > 0 {
            total_cost / f64::from(total_frames)
        } else {
            0.0
        };

        let mut cost_by_project = HashMap::new();
        for cost_entry in &costs {
            *cost_by_project
                .entry(cost_entry.project_id.clone())
                .or_insert(0.0) += cost_entry.cost;
        }

        let cost_trend: Vec<_> = costs.iter().map(|c| (c.timestamp, c.cost)).collect();

        CostAnalysisReport {
            period_start: start,
            period_end: end,
            total_cost,
            cost_per_job,
            cost_per_frame,
            cost_by_project,
            cost_trend,
        }
    }
}

impl Default for ReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_generator_creation() {
        let generator = ReportGenerator::new();
        assert_eq!(generator.job_data.len(), 0);
    }

    #[test]
    fn test_record_job() {
        let mut generator = ReportGenerator::new();
        generator.record_job(JobId::new(), 100.0, 10, "Normal".to_string(), true);
        assert_eq!(generator.job_data.len(), 1);
    }

    #[test]
    fn test_job_performance_report() {
        let mut generator = ReportGenerator::new();

        for _ in 0..5 {
            generator.record_job(JobId::new(), 100.0, 10, "Normal".to_string(), true);
        }

        let start = Utc::now() - chrono::Duration::hours(1);
        let end = Utc::now() + chrono::Duration::hours(1);

        let report = generator.generate_job_performance_report(start, end);
        assert_eq!(report.total_jobs, 5);
        assert_eq!(report.completed_jobs, 5);
    }
}
