// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Health monitoring for workers.

use crate::worker::WorkerId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    /// Worker ID
    pub worker_id: WorkerId,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Is healthy
    pub healthy: bool,
    /// Latency (ms)
    pub latency_ms: u64,
    /// Checks passed
    pub checks: HashMap<String, bool>,
}

/// Health monitor
pub struct HealthMonitor {
    results: HashMap<WorkerId, Vec<HealthCheck>>,
    #[allow(dead_code)]
    check_interval: u64,
}

impl HealthMonitor {
    /// Create a new health monitor
    #[must_use]
    pub fn new(check_interval: u64) -> Self {
        Self {
            results: HashMap::new(),
            check_interval,
        }
    }

    /// Record health check
    pub fn record(&mut self, check: HealthCheck) {
        self.results.entry(check.worker_id).or_default().push(check);
    }

    /// Get latest health check
    #[must_use]
    pub fn get_latest(&self, worker_id: WorkerId) -> Option<&HealthCheck> {
        self.results.get(&worker_id)?.last()
    }

    /// Get health history
    #[must_use]
    pub fn get_history(&self, worker_id: WorkerId) -> Vec<&HealthCheck> {
        self.results
            .get(&worker_id)
            .map_or_else(Vec::new, |checks| checks.iter().collect())
    }
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::new(30)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_monitor() {
        let mut monitor = HealthMonitor::new(30);
        let worker_id = WorkerId::new();

        let check = HealthCheck {
            worker_id,
            timestamp: Utc::now(),
            healthy: true,
            latency_ms: 10,
            checks: HashMap::new(),
        };

        monitor.record(check);
        assert!(monitor.get_latest(worker_id).is_some());
    }
}
