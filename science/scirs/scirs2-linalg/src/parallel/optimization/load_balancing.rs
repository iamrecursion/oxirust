//! Dynamic load balancer for irregular workloads
//!
//! This module provides dynamic load balancing for workloads where
//! different items may take varying amounts of time to process.

use super::super::strategies::work_stealing::WorkStealingScheduler;
use super::super::WorkerConfig;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Dynamic load balancer for irregular workloads
///
/// This struct provides dynamic load balancing for workloads where
/// different items may take varying amounts of time to process.
pub struct DynamicLoadBalancer {
    scheduler: WorkStealingScheduler,
    /// Tracks execution time statistics for adaptive scheduling
    timing_stats: Arc<Mutex<TimingStats>>,
}

#[derive(Default)]
struct TimingStats {
    total_items: usize,
    total_time_ms: u128,
    min_time_ms: u128,
    max_time_ms: u128,
}

impl DynamicLoadBalancer {
    /// Create a new dynamic load balancer
    pub fn new(config: &WorkerConfig) -> Self {
        Self {
            scheduler: WorkStealingScheduler::new(config),
            timing_stats: Arc::new(Mutex::new(TimingStats::default())),
        }
    }

    /// Execute work items with dynamic load balancing and timing
    pub fn execute_timed<T, R, F>(&self, items: &[T], f: F) -> Vec<R>
    where
        T: Send + Sync,
        R: Send + Default + Clone,
        F: Fn(&T) -> R + Send + Sync,
    {
        use std::time::Instant;

        let n = items.len();
        if n == 0 {
            return Vec::new();
        }

        let results = Arc::new(Mutex::new(vec![R::default(); n]));
        let work_counter = Arc::new(AtomicUsize::new(0));
        let timing_stats = self.timing_stats.clone();

        std::thread::scope(|s| {
            let handles: Vec<_> = (0..self.scheduler.num_workers)
                .map(|_| {
                    let work_counter = work_counter.clone();
                    let results = results.clone();
                    let timing_stats = timing_stats.clone();
                    let items_ref = items;
                    let f_ref = &f;

                    s.spawn(move || {
                        let mut local_min = u128::MAX;
                        let mut local_max = 0u128;
                        let mut local_total = 0u128;
                        let mut local_count = 0usize;

                        loop {
                            let idx = work_counter.fetch_add(1, Ordering::SeqCst);
                            if idx >= n {
                                break;
                            }

                            // Time the execution
                            let start = Instant::now();
                            let result = f_ref(&items_ref[idx]);
                            let elapsed = start.elapsed().as_millis();

                            // Update local statistics
                            local_min = local_min.min(elapsed);
                            local_max = local_max.max(elapsed);
                            local_total += elapsed;
                            local_count += 1;

                            // Store result
                            let mut results_guard = results.lock().expect("Operation failed");
                            results_guard[idx] = result;
                        }

                        // Update global statistics
                        if local_count > 0 {
                            let mut stats = timing_stats.lock().expect("Operation failed");
                            stats.total_items += local_count;
                            stats.total_time_ms += local_total;
                            stats.min_time_ms = stats.min_time_ms.min(local_min);
                            stats.max_time_ms = stats.max_time_ms.max(local_max);
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().expect("Operation failed");
            }
        });

        Arc::try_unwrap(results)
            .unwrap_or_else(|_| panic!("Failed to extract results"))
            .into_inner()
            .unwrap_or_else(|_| panic!("Failed to extract mutex inner value"))
    }

    /// Get average execution time per item
    pub fn get_average_time_ms(&self) -> f64 {
        let stats = self.timing_stats.lock().expect("Operation failed");
        if stats.total_items > 0 {
            stats.total_time_ms as f64 / stats.total_items as f64
        } else {
            0.0
        }
    }

    /// Get timing variance to detect irregular workloads
    pub fn get_time_variance(&self) -> f64 {
        let stats = self.timing_stats.lock().expect("Operation failed");
        if stats.total_items > 0 && stats.max_time_ms > stats.min_time_ms {
            (stats.max_time_ms - stats.min_time_ms) as f64 / stats.min_time_ms as f64
        } else {
            0.0
        }
    }

    /// Reset timing statistics
    pub fn reset_stats(&self) {
        let mut stats = self.timing_stats.lock().expect("Operation failed");
        *stats = TimingStats::default();
    }

    /// Get detailed timing statistics
    pub fn get_timing_stats(&self) -> LoadBalancingStats {
        let stats = self.timing_stats.lock().expect("Operation failed");
        LoadBalancingStats {
            total_items: stats.total_items,
            total_time_ms: stats.total_time_ms,
            min_time_ms: stats.min_time_ms,
            max_time_ms: stats.max_time_ms,
            average_time_ms: self.get_average_time_ms(),
            time_variance: self.get_time_variance(),
        }
    }
}

/// Statistics for load balancing performance analysis
#[derive(Debug, Clone)]
pub struct LoadBalancingStats {
    pub total_items: usize,
    pub total_time_ms: u128,
    pub min_time_ms: u128,
    pub max_time_ms: u128,
    pub average_time_ms: f64,
    pub time_variance: f64,
}

impl LoadBalancingStats {
    /// Check if the workload shows signs of being irregular
    pub fn is_irregular_workload(&self) -> bool {
        self.time_variance > 0.5 // 50% variance threshold
    }

    /// Get efficiency score (0.0 to 1.0, higher is better)
    pub fn efficiency_score(&self) -> f64 {
        if self.max_time_ms == 0 {
            return 1.0;
        }
        1.0 - (self.time_variance / 2.0).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_load_balancer() {
        let config = WorkerConfig::default();
        let balancer = DynamicLoadBalancer::new(&config);
        
        let items = vec![1, 2, 3, 4, 5];
        let results = balancer.execute_timed(&items, |x| {
            std::thread::sleep(std::time::Duration::from_millis(*x as u64));
            x * 2
        });
        
        assert_eq!(results, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_timing_stats() {
        let config = WorkerConfig::default();
        let balancer = DynamicLoadBalancer::new(&config);
        
        let items = vec![1, 10, 1, 10]; // Irregular workload
        let _results = balancer.execute_timed(&items, |x| {
            std::thread::sleep(std::time::Duration::from_millis(*x as u64));
            x * 2
        });
        
        let stats = balancer.get_timing_stats();
        assert!(stats.time_variance > 0.0);
        assert!(stats.is_irregular_workload());
    }

    #[test]
    fn test_efficiency_score() {
        let stats = LoadBalancingStats {
            total_items: 100,
            total_time_ms: 1000,
            min_time_ms: 5,
            max_time_ms: 15,
            average_time_ms: 10.0,
            time_variance: 0.2,
        };
        
        let score = stats.efficiency_score();
        assert!(score > 0.8); // Should be high efficiency for low variance
    }
}