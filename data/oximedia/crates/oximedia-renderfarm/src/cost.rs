// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Cost tracking and management.

use crate::error::{Error, Result};
use crate::job::JobId;
use crate::worker::WorkerId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cost calculation model
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CostModel {
    /// Per-hour pricing
    PerHour,
    /// Per-frame pricing
    PerFrame,
    /// Per-CPU-hour pricing
    PerCpuHour,
    /// Per-GPU-hour pricing
    PerGpuHour,
    /// Custom pricing
    Custom,
}

/// Cost entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEntry {
    /// Job ID
    pub job_id: JobId,
    /// Worker ID
    pub worker_id: WorkerId,
    /// Cost amount
    pub amount: f64,
    /// Cost type
    pub cost_type: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Cost report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostReport {
    /// Total cost
    pub total_cost: f64,
    /// Cost by job
    pub cost_by_job: HashMap<JobId, f64>,
    /// Cost by worker
    pub cost_by_worker: HashMap<WorkerId, f64>,
    /// Cost by type
    pub cost_by_type: HashMap<String, f64>,
    /// Start time
    pub start_time: DateTime<Utc>,
    /// End time
    pub end_time: DateTime<Utc>,
}

/// Cost tracker
pub struct CostTracker {
    entries: Vec<CostEntry>,
    cost_per_hour: f64,
    cost_per_frame: f64,
    cost_per_cpu_hour: f64,
    cost_per_gpu_hour: f64,
}

impl CostTracker {
    /// Create a new cost tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            cost_per_hour: 1.0,
            cost_per_frame: 0.01,
            cost_per_cpu_hour: 0.50,
            cost_per_gpu_hour: 2.00,
        }
    }

    /// Set pricing
    pub fn set_pricing(
        &mut self,
        per_hour: f64,
        per_frame: f64,
        per_cpu_hour: f64,
        per_gpu_hour: f64,
    ) {
        self.cost_per_hour = per_hour;
        self.cost_per_frame = per_frame;
        self.cost_per_cpu_hour = per_cpu_hour;
        self.cost_per_gpu_hour = per_gpu_hour;
    }

    /// Record cost
    pub fn record_cost(
        &mut self,
        job_id: JobId,
        worker_id: WorkerId,
        amount: f64,
        cost_type: String,
    ) {
        let entry = CostEntry {
            job_id,
            worker_id,
            amount,
            cost_type,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };
        self.entries.push(entry);
    }

    /// Calculate render time cost
    #[must_use]
    pub fn calculate_render_time_cost(&self, hours: f64, has_gpu: bool) -> f64 {
        if has_gpu {
            hours * self.cost_per_gpu_hour
        } else {
            hours * self.cost_per_cpu_hour
        }
    }

    /// Calculate frame cost
    #[must_use]
    pub fn calculate_frame_cost(&self, frame_count: u32) -> f64 {
        f64::from(frame_count) * self.cost_per_frame
    }

    /// Generate cost report
    #[must_use]
    pub fn generate_report(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> CostReport {
        let mut total_cost = 0.0;
        let mut cost_by_job: HashMap<JobId, f64> = HashMap::new();
        let mut cost_by_worker: HashMap<WorkerId, f64> = HashMap::new();
        let mut cost_by_type: HashMap<String, f64> = HashMap::new();

        for entry in &self.entries {
            if entry.timestamp >= start && entry.timestamp <= end {
                total_cost += entry.amount;

                *cost_by_job.entry(entry.job_id).or_insert(0.0) += entry.amount;
                *cost_by_worker.entry(entry.worker_id).or_insert(0.0) += entry.amount;
                *cost_by_type.entry(entry.cost_type.clone()).or_insert(0.0) += entry.amount;
            }
        }

        CostReport {
            total_cost,
            cost_by_job,
            cost_by_worker,
            cost_by_type,
            start_time: start,
            end_time: end,
        }
    }

    /// Get job cost
    #[must_use]
    pub fn get_job_cost(&self, job_id: JobId) -> f64 {
        self.entries
            .iter()
            .filter(|e| e.job_id == job_id)
            .map(|e| e.amount)
            .sum()
    }

    /// Get worker cost
    #[must_use]
    pub fn get_worker_cost(&self, worker_id: WorkerId) -> f64 {
        self.entries
            .iter()
            .filter(|e| e.worker_id == worker_id)
            .map(|e| e.amount)
            .sum()
    }
}

impl Default for CostTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Budget manager
pub struct BudgetManager {
    budgets: HashMap<String, Budget>,
}

/// Budget for a project
#[derive(Debug, Clone)]
pub struct Budget {
    /// Project ID
    pub project_id: String,
    /// Total budget
    pub total: f64,
    /// Spent amount
    pub spent: f64,
    /// Warning threshold (0.0 to 1.0)
    pub warning_threshold: f64,
}

impl Budget {
    /// Create a new budget
    #[must_use]
    pub fn new(project_id: String, total: f64) -> Self {
        Self {
            project_id,
            total,
            spent: 0.0,
            warning_threshold: 0.8,
        }
    }

    /// Get remaining budget
    #[must_use]
    pub fn remaining(&self) -> f64 {
        (self.total - self.spent).max(0.0)
    }

    /// Check if budget is exceeded
    #[must_use]
    pub fn is_exceeded(&self) -> bool {
        self.spent >= self.total
    }

    /// Check if warning threshold is reached
    #[must_use]
    pub fn is_warning(&self) -> bool {
        self.spent >= self.total * self.warning_threshold
    }

    /// Spend from budget
    pub fn spend(&mut self, amount: f64) -> Result<()> {
        if self.spent + amount > self.total {
            return Err(Error::BudgetExceeded {
                allocated: self.total,
                spent: self.spent + amount,
            });
        }
        self.spent += amount;
        Ok(())
    }
}

impl BudgetManager {
    /// Create a new budget manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            budgets: HashMap::new(),
        }
    }

    /// Set budget for project
    pub fn set_budget(&mut self, project_id: String, total: f64) {
        let budget = Budget::new(project_id.clone(), total);
        self.budgets.insert(project_id, budget);
    }

    /// Get budget for project
    #[must_use]
    pub fn get_budget(&self, project_id: &str) -> Option<&Budget> {
        self.budgets.get(project_id)
    }

    /// Spend from project budget
    pub fn spend(&mut self, project_id: &str, amount: f64) -> Result<()> {
        let budget = self.budgets.get_mut(project_id).ok_or_else(|| {
            Error::Configuration(format!("Budget not found for project: {project_id}"))
        })?;

        budget.spend(amount)
    }

    /// Check if project can spend amount
    #[must_use]
    pub fn can_spend(&self, project_id: &str, amount: f64) -> bool {
        if let Some(budget) = self.budgets.get(project_id) {
            budget.spent + amount <= budget.total
        } else {
            false
        }
    }
}

impl Default for BudgetManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_tracker_creation() {
        let tracker = CostTracker::new();
        assert_eq!(tracker.entries.len(), 0);
    }

    #[test]
    fn test_record_cost() {
        let mut tracker = CostTracker::new();
        let job_id = JobId::new();
        let worker_id = WorkerId::new();

        tracker.record_cost(job_id, worker_id, 10.0, "render".to_string());
        assert_eq!(tracker.entries.len(), 1);
    }

    #[test]
    fn test_calculate_render_time_cost() {
        let tracker = CostTracker::new();

        // CPU rendering
        let cost = tracker.calculate_render_time_cost(2.0, false);
        assert_eq!(cost, 1.0); // 2 hours * 0.50 per hour

        // GPU rendering
        let cost = tracker.calculate_render_time_cost(2.0, true);
        assert_eq!(cost, 4.0); // 2 hours * 2.00 per hour
    }

    #[test]
    fn test_calculate_frame_cost() {
        let tracker = CostTracker::new();
        let cost = tracker.calculate_frame_cost(100);
        assert_eq!(cost, 1.0); // 100 frames * 0.01 per frame
    }

    #[test]
    fn test_get_job_cost() {
        let mut tracker = CostTracker::new();
        let job_id = JobId::new();
        let worker_id = WorkerId::new();

        tracker.record_cost(job_id, worker_id, 10.0, "render".to_string());
        tracker.record_cost(job_id, worker_id, 5.0, "storage".to_string());

        let cost = tracker.get_job_cost(job_id);
        assert_eq!(cost, 15.0);
    }

    #[test]
    fn test_generate_report() {
        let mut tracker = CostTracker::new();
        let job_id = JobId::new();
        let worker_id = WorkerId::new();

        let start = Utc::now();
        tracker.record_cost(job_id, worker_id, 10.0, "render".to_string());
        let end = Utc::now();

        let report = tracker.generate_report(start, end);
        assert_eq!(report.total_cost, 10.0);
        assert_eq!(report.cost_by_job.len(), 1);
    }

    #[test]
    fn test_budget_creation() {
        let budget = Budget::new("project1".to_string(), 1000.0);
        assert_eq!(budget.total, 1000.0);
        assert_eq!(budget.spent, 0.0);
        assert_eq!(budget.remaining(), 1000.0);
    }

    #[test]
    fn test_budget_spend() -> Result<()> {
        let mut budget = Budget::new("project1".to_string(), 1000.0);

        budget.spend(100.0)?;
        assert_eq!(budget.spent, 100.0);
        assert_eq!(budget.remaining(), 900.0);

        Ok(())
    }

    #[test]
    fn test_budget_exceeded() {
        let mut budget = Budget::new("project1".to_string(), 1000.0);

        let result = budget.spend(1500.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_budget_warning() {
        let mut budget = Budget::new("project1".to_string(), 1000.0);

        assert!(!budget.is_warning());

        budget.spend(850.0).expect("should succeed in test");
        assert!(budget.is_warning());
    }

    #[test]
    fn test_budget_manager() -> Result<()> {
        let mut manager = BudgetManager::new();

        manager.set_budget("project1".to_string(), 1000.0);
        assert!(manager.can_spend("project1", 500.0));

        manager.spend("project1", 500.0)?;
        assert_eq!(
            manager
                .get_budget("project1")
                .expect("should succeed in test")
                .spent,
            500.0
        );

        Ok(())
    }
}
