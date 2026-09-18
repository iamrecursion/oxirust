// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Task distribution strategies.

use crate::worker::Worker;
use serde::{Deserialize, Serialize};

/// Distribution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributionStrategy {
    /// Even distribution
    Even,
    /// Weighted by performance
    Weighted,
    /// Locality-aware
    LocalityAware,
}

/// Task distributor
pub struct Distributor {
    strategy: DistributionStrategy,
}

impl Distributor {
    /// Create a new distributor
    #[must_use]
    pub fn new(strategy: DistributionStrategy) -> Self {
        Self { strategy }
    }

    /// Distribute tasks to workers
    #[must_use]
    pub fn distribute(&self, task_count: usize, workers: &[Worker]) -> Vec<(usize, Vec<usize>)> {
        match self.strategy {
            DistributionStrategy::Even => self.distribute_even(task_count, workers),
            DistributionStrategy::Weighted => self.distribute_weighted(task_count, workers),
            DistributionStrategy::LocalityAware => {
                self.distribute_locality_aware(task_count, workers)
            }
        }
    }

    fn distribute_even(&self, task_count: usize, workers: &[Worker]) -> Vec<(usize, Vec<usize>)> {
        let mut result = Vec::new();
        let tasks_per_worker = task_count / workers.len();
        let remainder = task_count % workers.len();

        let mut task_id = 0;
        for (i, _worker) in workers.iter().enumerate() {
            let count = tasks_per_worker + usize::from(i < remainder);
            let tasks: Vec<usize> = (task_id..task_id + count).collect();
            result.push((i, tasks));
            task_id += count;
        }

        result
    }

    fn distribute_weighted(
        &self,
        task_count: usize,
        workers: &[Worker],
    ) -> Vec<(usize, Vec<usize>)> {
        let total_weight: f64 = workers
            .iter()
            .map(super::worker::Worker::performance_score)
            .sum();
        let mut result = Vec::new();
        let mut task_id = 0;

        for (i, worker) in workers.iter().enumerate() {
            let weight = worker.performance_score() / total_weight;
            let count = (task_count as f64 * weight).round() as usize;
            let tasks: Vec<usize> = (task_id..task_id + count.min(task_count - task_id)).collect();
            let task_len = tasks.len();
            result.push((i, tasks));
            task_id += task_len;
        }

        result
    }

    fn distribute_locality_aware(
        &self,
        task_count: usize,
        workers: &[Worker],
    ) -> Vec<(usize, Vec<usize>)> {
        // Simplified: same as even distribution
        self.distribute_even(task_count, workers)
    }
}

impl Default for Distributor {
    fn default() -> Self {
        Self::new(DistributionStrategy::Weighted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::WorkerRegistration;
    use std::collections::HashMap;
    use std::net::{IpAddr, Ipv4Addr};

    fn create_test_worker() -> Worker {
        let registration = WorkerRegistration {
            hostname: "worker".to_string(),
            ip_address: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)),
            port: 8080,
            capabilities: Default::default(),
            location: None,
            tags: HashMap::new(),
        };
        Worker::new(registration)
    }

    #[test]
    fn test_distributor_creation() {
        let dist = Distributor::new(DistributionStrategy::Even);
        assert_eq!(dist.strategy, DistributionStrategy::Even);
    }

    #[test]
    fn test_even_distribution() {
        let dist = Distributor::new(DistributionStrategy::Even);
        let workers = vec![create_test_worker(), create_test_worker()];

        let result = dist.distribute(10, &workers);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].1.len(), 5);
        assert_eq!(result[1].1.len(), 5);
    }
}
