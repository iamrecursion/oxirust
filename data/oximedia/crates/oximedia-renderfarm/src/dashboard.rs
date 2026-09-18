// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Dashboard and visualization.

use crate::job::{JobId, JobState};
use crate::worker::{WorkerId, WorkerState};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Dashboard data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Job grid
    pub job_grid: JobGrid,
    /// Worker map
    pub worker_map: WorkerMap,
    /// Performance graphs
    pub performance: PerformanceData,
    /// Cost tracking
    pub costs: CostData,
}

/// Job status grid
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobGrid {
    /// Jobs by state
    pub jobs_by_state: HashMap<String, Vec<JobSummary>>,
    /// Total jobs
    pub total: usize,
}

/// Job summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobSummary {
    /// Job ID
    pub id: JobId,
    /// Name
    pub name: String,
    /// State
    pub state: JobState,
    /// Progress
    pub progress: f64,
    /// ETA
    pub eta: Option<DateTime<Utc>>,
}

/// Worker status map
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerMap {
    /// Workers by state
    pub workers_by_state: HashMap<String, Vec<WorkerSummary>>,
    /// Total workers
    pub total: usize,
}

/// Worker summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerSummary {
    /// Worker ID
    pub id: WorkerId,
    /// Hostname
    pub hostname: String,
    /// State
    pub state: WorkerState,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Current job
    pub current_job: Option<JobId>,
}

/// Performance data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceData {
    /// CPU usage over time
    pub cpu_usage: Vec<TimeSeriesPoint>,
    /// Memory usage over time
    pub memory_usage: Vec<TimeSeriesPoint>,
    /// Throughput over time (frames/hour)
    pub throughput: Vec<TimeSeriesPoint>,
    /// Queue depth over time
    pub queue_depth: Vec<TimeSeriesPoint>,
}

/// Time series data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Value
    pub value: f64,
}

/// Cost data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostData {
    /// Total cost
    pub total: f64,
    /// Cost by project
    pub by_project: HashMap<String, f64>,
    /// Cost over time
    pub over_time: Vec<TimeSeriesPoint>,
}

/// Dashboard builder
pub struct Dashboard {
    job_grid: JobGrid,
    worker_map: WorkerMap,
    performance: PerformanceData,
    costs: CostData,
}

impl Dashboard {
    /// Create a new dashboard
    #[must_use]
    pub fn new() -> Self {
        Self {
            job_grid: JobGrid {
                jobs_by_state: HashMap::new(),
                total: 0,
            },
            worker_map: WorkerMap {
                workers_by_state: HashMap::new(),
                total: 0,
            },
            performance: PerformanceData {
                cpu_usage: Vec::new(),
                memory_usage: Vec::new(),
                throughput: Vec::new(),
                queue_depth: Vec::new(),
            },
            costs: CostData {
                total: 0.0,
                by_project: HashMap::new(),
                over_time: Vec::new(),
            },
        }
    }

    /// Update job grid
    pub fn update_jobs(&mut self, jobs: Vec<JobSummary>) {
        self.job_grid.total = jobs.len();
        self.job_grid.jobs_by_state.clear();

        for job in jobs {
            let state_key = job.state.to_string();
            self.job_grid
                .jobs_by_state
                .entry(state_key)
                .or_default()
                .push(job);
        }
    }

    /// Update worker map
    pub fn update_workers(&mut self, workers: Vec<WorkerSummary>) {
        self.worker_map.total = workers.len();
        self.worker_map.workers_by_state.clear();

        for worker in workers {
            let state_key = worker.state.to_string();
            self.worker_map
                .workers_by_state
                .entry(state_key)
                .or_default()
                .push(worker);
        }
    }

    /// Add performance metric
    pub fn add_metric(&mut self, metric_type: &str, value: f64) {
        let point = TimeSeriesPoint {
            timestamp: Utc::now(),
            value,
        };

        match metric_type {
            "cpu" => self.performance.cpu_usage.push(point),
            "memory" => self.performance.memory_usage.push(point),
            "throughput" => self.performance.throughput.push(point),
            "queue" => self.performance.queue_depth.push(point),
            _ => {}
        }
    }

    /// Update costs
    pub fn update_costs(&mut self, total: f64, by_project: HashMap<String, f64>) {
        self.costs.total = total;
        self.costs.by_project = by_project;

        let point = TimeSeriesPoint {
            timestamp: Utc::now(),
            value: total,
        };
        self.costs.over_time.push(point);
    }

    /// Get dashboard data
    #[must_use]
    pub fn get_data(&self) -> DashboardData {
        DashboardData {
            timestamp: Utc::now(),
            job_grid: self.job_grid.clone(),
            worker_map: self.worker_map.clone(),
            performance: self.performance.clone(),
            costs: self.costs.clone(),
        }
    }
}

impl Default for Dashboard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_creation() {
        let dashboard = Dashboard::new();
        let data = dashboard.get_data();
        assert_eq!(data.job_grid.total, 0);
    }

    #[test]
    fn test_update_jobs() {
        let mut dashboard = Dashboard::new();
        let jobs = vec![JobSummary {
            id: JobId::new(),
            name: "Job 1".to_string(),
            state: JobState::Rendering,
            progress: 0.5,
            eta: None,
        }];

        dashboard.update_jobs(jobs);
        let data = dashboard.get_data();
        assert_eq!(data.job_grid.total, 1);
    }

    #[test]
    fn test_add_metric() {
        let mut dashboard = Dashboard::new();
        dashboard.add_metric("cpu", 50.0);
        dashboard.add_metric("memory", 60.0);

        let data = dashboard.get_data();
        assert_eq!(data.performance.cpu_usage.len(), 1);
        assert_eq!(data.performance.memory_usage.len(), 1);
    }
}
