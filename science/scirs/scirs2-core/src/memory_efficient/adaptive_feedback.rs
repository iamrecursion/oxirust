//! Adaptive feedback system for chunk size optimization.
//!
//! This module provides a feedback-based system that learns optimal chunk sizes
//! by tracking performance metrics and adjusting parameters over time.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Default timestamp for deserialization
fn default_instant() -> Instant {
    Instant::now()
}

/// Performance metrics for a chunking operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Chunk size used (in elements)
    pub chunk_size: usize,

    /// Time taken to process the chunk
    pub duration: Duration,

    /// Throughput (elements per second)
    pub throughput: f64,

    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,

    /// Memory usage during processing (bytes)
    pub memory_usage: u64,

    /// Timestamp when metric was recorded
    #[serde(skip, default = "default_instant")]
    pub timestamp: Instant,
}

impl PerformanceMetrics {
    /// Create new performance metrics
    pub fn new(
        chunk_size: usize,
        duration: Duration,
        cache_hit_rate: f64,
        memory_usage: u64,
    ) -> Self {
        let throughput = if duration.as_secs_f64() > 0.0 {
            chunk_size as f64 / duration.as_secs_f64()
        } else {
            0.0
        };

        Self {
            chunk_size,
            duration,
            throughput,
            cache_hit_rate,
            memory_usage,
            timestamp: Instant::now(),
        }
    }

    /// Calculate a performance score (higher is better)
    pub fn score(&self) -> f64 {
        // Weighted combination of throughput and cache hit rate
        // Normalize throughput to prevent dominating the score
        let normalized_throughput = self.throughput.log10().max(0.0);
        let cache_weight = 0.3;
        let throughput_weight = 0.7;

        (cache_weight * self.cache_hit_rate) + (throughput_weight * normalized_throughput)
    }
}

/// Chunk size predictor with historical tracking
#[derive(Debug, Clone)]
pub struct ChunkSizePredictor {
    /// Historical performance metrics (limited to last N entries)
    history: VecDeque<PerformanceMetrics>,

    /// Maximum history size
    max_history: usize,

    /// Current best chunk size
    best_chunk_size: usize,

    /// Current best score
    best_score: f64,

    /// Moving average window size
    window_size: usize,

    /// Minimum chunk size to consider
    min_chunk_size: usize,

    /// Maximum chunk size to consider
    max_chunk_size: usize,

    /// Learning rate for adjustments (0.0 to 1.0)
    learning_rate: f64,
}

impl ChunkSizePredictor {
    /// Create a new chunk size predictor
    pub fn new(initial_chunk_size: usize, min_chunk_size: usize, max_chunk_size: usize) -> Self {
        Self {
            history: VecDeque::new(),
            max_history: 100,
            best_chunk_size: initial_chunk_size,
            best_score: 0.0,
            window_size: 10,
            min_chunk_size,
            max_chunk_size,
            learning_rate: 0.1,
        }
    }

    /// Create a new predictor with custom configuration
    pub fn with_config(
        initial_chunk_size: usize,
        min_chunk_size: usize,
        max_chunk_size: usize,
        max_history: usize,
        window_size: usize,
        learning_rate: f64,
    ) -> Self {
        Self {
            history: VecDeque::new(),
            max_history,
            best_chunk_size: initial_chunk_size,
            best_score: 0.0,
            window_size,
            min_chunk_size,
            max_chunk_size,
            learning_rate,
        }
    }

    /// Record a performance measurement
    pub fn record(&mut self, metrics: PerformanceMetrics) {
        let score = metrics.score();

        // Update best chunk size if this performs better
        if score > self.best_score {
            self.best_score = score;
            self.best_chunk_size = metrics.chunk_size;
        }

        // Add to history
        self.history.push_back(metrics);

        // Limit history size
        while self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Predict optimal chunk size based on historical data
    pub fn predict(&self) -> usize {
        if self.history.is_empty() {
            return self.best_chunk_size;
        }

        // Use moving average of recent best performers
        let recent_metrics: Vec<&PerformanceMetrics> =
            self.history.iter().rev().take(self.window_size).collect();

        if recent_metrics.is_empty() {
            return self.best_chunk_size;
        }

        // Calculate weighted average chunk size
        let total_score: f64 = recent_metrics.iter().map(|m| m.score()).sum();

        if total_score == 0.0 {
            return self.best_chunk_size;
        }

        let weighted_chunk_size: f64 = recent_metrics
            .iter()
            .map(|m| m.chunk_size as f64 * m.score())
            .sum::<f64>()
            / total_score;

        // Apply learning rate to blend with current best
        let predicted_size = (self.best_chunk_size as f64 * (1.0 - self.learning_rate))
            + (weighted_chunk_size * self.learning_rate);

        // Clamp to min/max bounds
        predicted_size
            .round()
            .max(self.min_chunk_size as f64)
            .min(self.max_chunk_size as f64) as usize
    }

    /// Get the current best chunk size
    pub fn best_chunk_size(&self) -> usize {
        self.best_chunk_size
    }

    /// Get the average throughput over recent operations
    pub fn average_throughput(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }

        let recent: Vec<&PerformanceMetrics> =
            self.history.iter().rev().take(self.window_size).collect();

        recent.iter().map(|m| m.throughput).sum::<f64>() / recent.len() as f64
    }

    /// Get the average cache hit rate over recent operations
    pub fn average_cache_hit_rate(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }

        let recent: Vec<&PerformanceMetrics> =
            self.history.iter().rev().take(self.window_size).collect();

        recent.iter().map(|m| m.cache_hit_rate).sum::<f64>() / recent.len() as f64
    }

    /// Clear all historical data
    pub fn reset(&mut self) {
        self.history.clear();
        self.best_score = 0.0;
    }

    /// Get the number of recorded metrics
    pub fn history_size(&self) -> usize {
        self.history.len()
    }
}

/// Thread-safe wrapper for chunk size predictor
pub type SharedPredictor = Arc<Mutex<ChunkSizePredictor>>;

/// Create a new shared predictor
#[allow(dead_code)]
pub fn create_shared_predictor(
    initial_chunk_size: usize,
    min_chunk_size: usize,
    max_chunk_size: usize,
) -> SharedPredictor {
    Arc::new(Mutex::new(ChunkSizePredictor::new(
        initial_chunk_size,
        min_chunk_size,
        max_chunk_size,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics_creation() {
        let metrics = PerformanceMetrics::new(1000, Duration::from_secs(1), 0.8, 1024 * 1024);

        assert_eq!(metrics.chunk_size, 1000);
        assert_eq!(metrics.duration, Duration::from_secs(1));
        assert!((metrics.throughput - 1000.0).abs() < 1e-6);
        assert_eq!(metrics.cache_hit_rate, 0.8);
        assert_eq!(metrics.memory_usage, 1024 * 1024);
    }

    #[test]
    fn test_performance_score() {
        let metrics1 = PerformanceMetrics::new(1000, Duration::from_millis(100), 0.9, 1024 * 1024);

        let metrics2 = PerformanceMetrics::new(1000, Duration::from_millis(100), 0.5, 1024 * 1024);

        // Higher cache hit rate should result in higher score
        assert!(metrics1.score() > metrics2.score());
    }

    #[test]
    fn test_predictor_creation() {
        let predictor = ChunkSizePredictor::new(1000, 100, 10000);

        assert_eq!(predictor.best_chunk_size(), 1000);
        assert_eq!(predictor.history_size(), 0);
    }

    #[test]
    fn test_predictor_record_and_predict() {
        let mut predictor = ChunkSizePredictor::new(1000, 100, 10000);

        // Record some metrics with varying performance
        let metrics1 = PerformanceMetrics::new(1000, Duration::from_millis(100), 0.8, 1024 * 1024);
        predictor.record(metrics1);

        let metrics2 = PerformanceMetrics::new(1500, Duration::from_millis(80), 0.85, 1024 * 1024);
        predictor.record(metrics2);

        let metrics3 = PerformanceMetrics::new(2000, Duration::from_millis(90), 0.9, 1024 * 1024);
        predictor.record(metrics3);

        assert_eq!(predictor.history_size(), 3);

        // Predict should return a chunk size within bounds
        let predicted = predictor.predict();
        assert!(predicted >= 100);
        assert!(predicted <= 10000);
    }

    #[test]
    fn test_predictor_best_tracking() {
        let mut predictor = ChunkSizePredictor::new(1000, 100, 10000);

        // Record a high-performing configuration
        let good_metrics =
            PerformanceMetrics::new(2000, Duration::from_millis(50), 0.95, 1024 * 1024);
        predictor.record(good_metrics);

        // Record a lower-performing configuration
        let bad_metrics =
            PerformanceMetrics::new(1000, Duration::from_millis(200), 0.5, 1024 * 1024);
        predictor.record(bad_metrics);

        // Best should still be the high-performing one
        assert_eq!(predictor.best_chunk_size(), 2000);
    }

    #[test]
    fn test_predictor_history_limit() {
        let mut predictor = ChunkSizePredictor::with_config(
            1000, 100, 10000, 5, // max_history
            3, 0.1,
        );

        // Record more than max_history entries
        for i in 0..10 {
            let metrics = PerformanceMetrics::new(
                1000 + i * 100,
                Duration::from_millis(100),
                0.8,
                1024 * 1024,
            );
            predictor.record(metrics);
        }

        // Should be limited to max_history
        assert_eq!(predictor.history_size(), 5);
    }

    #[test]
    fn test_predictor_averages() {
        let mut predictor = ChunkSizePredictor::new(1000, 100, 10000);

        let metrics1 = PerformanceMetrics::new(1000, Duration::from_millis(100), 0.8, 1024 * 1024);
        predictor.record(metrics1);

        let metrics2 = PerformanceMetrics::new(1000, Duration::from_millis(100), 0.6, 1024 * 1024);
        predictor.record(metrics2);

        let avg_cache_hit = predictor.average_cache_hit_rate();
        assert!((avg_cache_hit - 0.7).abs() < 1e-6);

        assert!(predictor.average_throughput() > 0.0);
    }

    #[test]
    fn test_predictor_reset() {
        let mut predictor = ChunkSizePredictor::new(1000, 100, 10000);

        let metrics = PerformanceMetrics::new(2000, Duration::from_millis(100), 0.8, 1024 * 1024);
        predictor.record(metrics);

        assert_eq!(predictor.history_size(), 1);

        predictor.reset();

        assert_eq!(predictor.history_size(), 0);
    }

    #[test]
    fn test_shared_predictor() {
        let predictor = create_shared_predictor(1000, 100, 10000);

        // Should be able to lock and use it
        {
            let mut p = predictor.lock().expect("Lock failed");
            let metrics =
                PerformanceMetrics::new(1500, Duration::from_millis(100), 0.8, 1024 * 1024);
            p.record(metrics);
        }

        // Verify the record was stored
        {
            let p = predictor.lock().expect("Lock failed");
            assert_eq!(p.history_size(), 1);
        }
    }
}
