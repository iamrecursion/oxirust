//! FFT performance tracking system

use super::types::*;
use std::collections::{HashMap, VecDeque};

/// FFT performance tracker
#[derive(Debug, Default)]
pub struct FftPerformanceTracker {
    /// Execution time history
    pub execution_times: VecDeque<f64>,
    /// Memory usage history
    pub memory_usage: VecDeque<usize>,
    /// Accuracy measurements
    pub accuracy_measurements: VecDeque<f64>,
    /// Algorithm usage statistics
    pub algorithm_usage: HashMap<FftAlgorithmType, AlgorithmStats>,
    /// Performance trends
    pub performance_trends: PerformanceTrends,
}
