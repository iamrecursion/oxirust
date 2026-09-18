//! Benchmark runner.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Benchmark result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark name.
    pub name: String,

    /// Number of iterations.
    pub iterations: u64,

    /// Mean execution time.
    pub mean: Duration,

    /// Median execution time.
    pub median: Duration,

    /// Standard deviation.
    pub std_dev: Duration,

    /// Minimum time.
    pub min: Duration,

    /// Maximum time.
    pub max: Duration,

    /// Throughput (iterations/sec).
    pub throughput: f64,
}

/// Benchmark runner.
#[derive(Debug)]
pub struct BenchmarkRunner {
    warmup_iterations: u64,
    measurement_iterations: u64,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner.
    pub fn new(warmup_iterations: u64, measurement_iterations: u64) -> Self {
        Self {
            warmup_iterations,
            measurement_iterations,
        }
    }

    /// Run a benchmark.
    pub fn run<F>(&self, name: String, mut bench_fn: F) -> BenchmarkResult
    where
        F: FnMut(),
    {
        // Warmup
        for _ in 0..self.warmup_iterations {
            bench_fn();
        }

        // Measure
        let mut times = Vec::with_capacity(self.measurement_iterations as usize);

        for _ in 0..self.measurement_iterations {
            let start = Instant::now();
            bench_fn();
            let elapsed = start.elapsed();
            times.push(elapsed);
        }

        self.calculate_result(name, times)
    }

    /// Calculate benchmark result from timings.
    fn calculate_result(&self, name: String, mut times: Vec<Duration>) -> BenchmarkResult {
        times.sort();

        let mean = times.iter().sum::<Duration>() / times.len() as u32;
        let median = times[times.len() / 2];
        let min = times[0];
        let max = times[times.len() - 1];

        let mean_secs = mean.as_secs_f64();
        let variance = times
            .iter()
            .map(|t| {
                let diff = t.as_secs_f64() - mean_secs;
                diff * diff
            })
            .sum::<f64>()
            / times.len() as f64;
        let std_dev = Duration::from_secs_f64(variance.sqrt());

        let throughput = if mean_secs > 0.0 {
            1.0 / mean_secs
        } else {
            0.0
        };

        BenchmarkResult {
            name,
            iterations: self.measurement_iterations,
            mean,
            median,
            std_dev,
            min,
            max,
            throughput,
        }
    }
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new(100, 1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_runner() {
        let runner = BenchmarkRunner::new(10, 100);
        let result = runner.run("test".to_string(), || {
            std::thread::sleep(Duration::from_micros(1));
        });

        assert_eq!(result.name, "test");
        assert_eq!(result.iterations, 100);
        assert!(result.mean > Duration::ZERO);
    }

    #[test]
    fn test_benchmark_statistics() {
        let runner = BenchmarkRunner::new(5, 50);
        let result = runner.run("stats_test".to_string(), || {
            // Empty benchmark
        });

        assert!(result.min <= result.mean);
        assert!(result.mean <= result.max);
        assert!(result.throughput > 0.0);
    }
}
