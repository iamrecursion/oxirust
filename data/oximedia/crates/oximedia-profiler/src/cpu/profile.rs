//! CPU profiling functionality.

use crate::{ProfilerError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// CPU profiler that tracks function execution time.
#[derive(Debug)]
pub struct CpuProfiler {
    sample_rate: u32,
    running: bool,
    start_time: Option<Instant>,
    samples: Vec<super::sample::Sample>,
    function_times: HashMap<String, Duration>,
    call_counts: HashMap<String, u64>,
}

/// CPU profiling statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuStats {
    /// Total CPU time.
    pub total_time: Duration,

    /// Number of samples collected.
    pub sample_count: usize,

    /// Top functions by CPU time.
    pub top_functions: Vec<FunctionStat>,

    /// Average CPU usage percentage.
    pub avg_cpu_usage: f64,
}

/// Statistics for a single function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionStat {
    /// Function name.
    pub name: String,

    /// Total time spent in function.
    pub total_time: Duration,

    /// Number of calls.
    pub call_count: u64,

    /// Average time per call.
    pub avg_time: Duration,

    /// Percentage of total CPU time.
    pub cpu_percentage: f64,
}

impl CpuProfiler {
    /// Create a new CPU profiler with the given sample rate.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            running: false,
            start_time: None,
            samples: Vec::new(),
            function_times: HashMap::new(),
            call_counts: HashMap::new(),
        }
    }

    /// Start CPU profiling.
    pub fn start(&mut self) -> Result<()> {
        if self.running {
            return Err(ProfilerError::AlreadyRunning);
        }

        self.start_time = Some(Instant::now());
        self.running = true;
        self.samples.clear();
        self.function_times.clear();
        self.call_counts.clear();

        Ok(())
    }

    /// Stop CPU profiling.
    pub fn stop(&mut self) -> Result<()> {
        if !self.running {
            return Err(ProfilerError::NotRunning);
        }

        self.running = false;
        Ok(())
    }

    /// Record a function call.
    pub fn record_call(&mut self, function: String, duration: Duration) {
        *self
            .function_times
            .entry(function.clone())
            .or_insert(Duration::ZERO) += duration;
        *self.call_counts.entry(function).or_insert(0) += 1;
    }

    /// Add a sample to the profiler.
    pub fn add_sample(&mut self, sample: super::sample::Sample) {
        self.samples.push(sample);
    }

    /// Get CPU profiling statistics.
    pub fn stats(&self) -> CpuStats {
        let total_time = self.start_time.map(|s| s.elapsed()).unwrap_or_default();
        let sample_count = self.samples.len();

        let mut top_functions: Vec<FunctionStat> = self
            .function_times
            .iter()
            .map(|(name, &total_time)| {
                let call_count = self.call_counts.get(name).copied().unwrap_or(0);
                let avg_time = if call_count > 0 {
                    total_time / call_count as u32
                } else {
                    Duration::ZERO
                };
                let cpu_percentage = if total_time.as_secs_f64() > 0.0 {
                    (total_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
                } else {
                    0.0
                };

                FunctionStat {
                    name: name.clone(),
                    total_time,
                    call_count,
                    avg_time,
                    cpu_percentage,
                }
            })
            .collect();

        top_functions.sort_by(|a, b| b.total_time.cmp(&a.total_time));
        top_functions.truncate(10);

        let avg_cpu_usage = if sample_count > 0 {
            let system = sysinfo::System::new_all();
            system.global_cpu_usage() as f64
        } else {
            0.0
        };

        CpuStats {
            total_time,
            sample_count,
            top_functions,
            avg_cpu_usage,
        }
    }

    /// Generate a summary report.
    pub fn summary(&self) -> String {
        let stats = self.stats();
        let mut report = String::new();

        report.push_str(&format!("  Total Time: {:?}\n", stats.total_time));
        report.push_str(&format!("  Samples: {}\n", stats.sample_count));
        report.push_str(&format!("  Avg CPU Usage: {:.2}%\n", stats.avg_cpu_usage));

        if !stats.top_functions.is_empty() {
            report.push_str("  Top Functions:\n");
            for func in &stats.top_functions {
                report.push_str(&format!(
                    "    {} - {:?} ({} calls, avg {:?})\n",
                    func.name, func.total_time, func.call_count, func.avg_time
                ));
            }
        }

        report
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Check if profiler is running.
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Get the number of samples collected.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Get samples.
    pub fn samples(&self) -> &[super::sample::Sample] {
        &self.samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_profiler_creation() {
        let profiler = CpuProfiler::new(100);
        assert_eq!(profiler.sample_rate(), 100);
        assert!(!profiler.is_running());
    }

    #[test]
    fn test_cpu_profiler_start_stop() {
        let mut profiler = CpuProfiler::new(100);
        assert!(profiler.start().is_ok());
        assert!(profiler.is_running());
        assert!(profiler.start().is_err());
        assert!(profiler.stop().is_ok());
        assert!(!profiler.is_running());
    }

    #[test]
    fn test_record_call() {
        let mut profiler = CpuProfiler::new(100);
        profiler.start().expect("should succeed in test");
        profiler.record_call("test_function".to_string(), Duration::from_millis(100));
        profiler.record_call("test_function".to_string(), Duration::from_millis(50));
        profiler.stop().expect("should succeed in test");

        let stats = profiler.stats();
        assert_eq!(stats.top_functions.len(), 1);
        assert_eq!(stats.top_functions[0].name, "test_function");
        assert_eq!(stats.top_functions[0].call_count, 2);
    }

    #[test]
    fn test_cpu_stats() {
        let mut profiler = CpuProfiler::new(100);
        profiler.start().expect("should succeed in test");
        profiler.record_call("func1".to_string(), Duration::from_millis(100));
        profiler.record_call("func2".to_string(), Duration::from_millis(50));
        profiler.stop().expect("should succeed in test");

        let stats = profiler.stats();
        assert_eq!(stats.sample_count, 0);
        assert!(stats.top_functions.len() <= 2);
    }

    #[test]
    fn test_summary() {
        let mut profiler = CpuProfiler::new(100);
        profiler.start().expect("should succeed in test");
        profiler.record_call("test".to_string(), Duration::from_millis(100));
        profiler.stop().expect("should succeed in test");

        let summary = profiler.summary();
        assert!(summary.contains("Total Time"));
        assert!(summary.contains("Samples"));
    }
}
