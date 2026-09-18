//! Benchmark suite management.

use super::runner::{BenchmarkResult, BenchmarkRunner};
use std::collections::HashMap;

/// A single benchmark.
pub struct Benchmark {
    /// Benchmark name.
    pub name: String,

    /// Benchmark function.
    pub func: Box<dyn FnMut()>,
}

/// Benchmark suite.
#[derive(Debug)]
pub struct BenchmarkSuite {
    runner: BenchmarkRunner,
    benchmarks: HashMap<String, Vec<u8>>, // Placeholder since we can't store closures easily
}

impl BenchmarkSuite {
    /// Create a new benchmark suite.
    pub fn new(runner: BenchmarkRunner) -> Self {
        Self {
            runner,
            benchmarks: HashMap::new(),
        }
    }

    /// Register a benchmark name.
    pub fn register(&mut self, name: String) {
        self.benchmarks.insert(name, Vec::new());
    }

    /// Run all benchmarks.
    pub fn run_all<F>(&self, mut bench_fn: F) -> Vec<BenchmarkResult>
    where
        F: FnMut(&str),
    {
        let mut results = Vec::new();

        for name in self.benchmarks.keys() {
            let result = self.runner.run(name.clone(), || {
                bench_fn(name);
            });
            results.push(result);
        }

        results
    }

    /// Get benchmark count.
    pub fn benchmark_count(&self) -> usize {
        self.benchmarks.len()
    }

    /// Get benchmark names.
    pub fn benchmark_names(&self) -> Vec<&String> {
        self.benchmarks.keys().collect()
    }

    /// Generate a summary report.
    pub fn summary(&self, results: &[BenchmarkResult]) -> String {
        let mut report = String::new();

        report.push_str(&format!(
            "Benchmark Suite Results ({} benchmarks)\n\n",
            results.len()
        ));

        for result in results {
            report.push_str(&format!(
                "{}: {:?} (±{:?})\n",
                result.name, result.mean, result.std_dev
            ));
        }

        report
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::new(BenchmarkRunner::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_suite() {
        let mut suite = BenchmarkSuite::default();
        suite.register("bench1".to_string());
        suite.register("bench2".to_string());

        assert_eq!(suite.benchmark_count(), 2);
        assert_eq!(suite.benchmark_names().len(), 2);
    }

    #[test]
    fn test_run_all() {
        let mut suite = BenchmarkSuite::default();
        suite.register("test".to_string());

        let results = suite.run_all(|_name| {
            // Benchmark implementation
        });

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "test");
    }

    #[test]
    fn test_summary() {
        let suite = BenchmarkSuite::default();
        let results = vec![];
        let summary = suite.summary(&results);

        assert!(summary.contains("Benchmark Suite Results"));
    }
}
