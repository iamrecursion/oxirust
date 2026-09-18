//! Performance tracking and auto-tuning for pipeline optimization
//!
//! This module provides machine learning-based performance tracking, historical
//! analysis, and automatic parameter tuning for optimal pipeline performance.

use crate::error::{IoError, Result};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

use super::config::{
    AutoTuningParameters, ExecutionRecord, OptimizedPipelineConfig, PipelinePerformanceMetrics,
    RegressionDetector, SystemMetrics,
};

/// Performance history tracker for machine learning optimization
#[derive(Debug)]
pub struct PerformanceHistory {
    executions: Vec<ExecutionRecord>,
    pipeline_profiles: HashMap<String, PipelineProfile>,
    max_history_size: usize,
}

impl Default for PerformanceHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceHistory {
    pub fn new() -> Self {
        Self {
            executions: Vec::new(),
            pipeline_profiles: HashMap::new(),
            max_history_size: 10000,
        }
    }

    pub fn record_execution(
        &mut self,
        pipeline_id: &str,
        config: &OptimizedPipelineConfig,
        metrics: &PipelinePerformanceMetrics,
    ) -> Result<()> {
        let record = ExecutionRecord {
            timestamp: Utc::now(),
            pipeline_id: pipeline_id.to_string(),
            config: config.clone(),
            metrics: metrics.clone(),
        };

        self.executions.push(record);

        // Maintain history size limit
        if self.executions.len() > self.max_history_size {
            self.executions.remove(0);
        }

        // Update or create pipeline profile
        self.update_pipeline_profile(pipeline_id, config, metrics);

        Ok(())
    }

    pub fn get_similar_configurations(
        &self,
        pipeline_id: &str,
        data_size: usize,
    ) -> Vec<&ExecutionRecord> {
        let size_threshold = 0.2; // 20% size difference tolerance

        self.executions
            .iter()
            .filter(|record| {
                record.pipeline_id == pipeline_id
                    && (record.metrics.data_size as f64 - data_size as f64).abs()
                        / (data_size as f64)
                        < size_threshold
            })
            .collect()
    }

    fn update_pipeline_profile(
        &mut self,
        pipeline_id: &str,
        config: &OptimizedPipelineConfig,
        metrics: &PipelinePerformanceMetrics,
    ) {
        let profile = self
            .pipeline_profiles
            .entry(pipeline_id.to_string())
            .or_insert_with(|| PipelineProfile::new(pipeline_id));

        profile.update(config, metrics);
    }

    pub fn get_pipeline_profile(&self, pipeline_id: &str) -> Option<&PipelineProfile> {
        self.pipeline_profiles.get(pipeline_id)
    }

    pub fn get_best_configurations(
        &self,
        pipeline_id: &str,
        limit: usize,
    ) -> Vec<&ExecutionRecord> {
        let mut records: Vec<&ExecutionRecord> = self
            .executions
            .iter()
            .filter(|record| record.pipeline_id == pipeline_id)
            .collect();

        records.sort_by(|a, b| {
            b.metrics
                .throughput
                .partial_cmp(&a.metrics.throughput)
                .expect("Operation failed")
        });

        records.into_iter().take(limit).collect()
    }
}

/// Pipeline performance profile with statistical analysis
#[derive(Debug)]
pub struct PipelineProfile {
    pub pipeline_id: String,
    pub execution_count: usize,
    pub avg_throughput: f64,
    pub avg_memory_usage: f64,
    pub avg_cpu_utilization: f64,
    pub optimal_configurations: Vec<OptimizedPipelineConfig>,
    pub performance_regression_detector: RegressionDetector,
}

impl PipelineProfile {
    pub fn new(pipeline_id: &str) -> Self {
        Self {
            pipeline_id: pipeline_id.to_string(),
            execution_count: 0,
            avg_throughput: 0.0,
            avg_memory_usage: 0.0,
            avg_cpu_utilization: 0.0,
            optimal_configurations: Vec::new(),
            performance_regression_detector: RegressionDetector::new(),
        }
    }

    pub fn update(
        &mut self,
        config: &OptimizedPipelineConfig,
        metrics: &PipelinePerformanceMetrics,
    ) {
        self.execution_count += 1;

        // Update running averages
        let weight = 1.0 / self.execution_count as f64;
        self.avg_throughput += weight * (metrics.throughput - self.avg_throughput);
        self.avg_memory_usage +=
            weight * (metrics.peak_memory_usage as f64 - self.avg_memory_usage);
        self.avg_cpu_utilization += weight * (metrics.cpu_utilization - self.avg_cpu_utilization);

        // Check for performance regression
        self.performance_regression_detector
            .check_regression(metrics);

        // Update optimal configurations if this is better
        if self.is_better_configuration(config, metrics) {
            self.optimal_configurations.push(config.clone());
            // Keep only top 5 configurations
            if self.optimal_configurations.len() > 5 {
                self.optimal_configurations.remove(0);
            }
        }
    }

    fn is_better_configuration(
        &self,
        _config: &OptimizedPipelineConfig,
        metrics: &PipelinePerformanceMetrics,
    ) -> bool {
        // Score based on throughput, memory efficiency, and CPU utilization
        let score = metrics.throughput * 0.5
            + (1.0 / metrics.peak_memory_usage as f64) * 0.3
            + metrics.cpu_utilization * 0.2;

        // Compare with average performance
        let avg_score = self.avg_throughput * 0.5
            + (1.0 / self.avg_memory_usage) * 0.3
            + self.avg_cpu_utilization * 0.2;

        score > avg_score * 1.1 // 10% improvement threshold
    }

    pub fn get_performance_trend(&self) -> PerformanceTrend {
        // Simplified trend analysis
        PerformanceTrend {
            direction: TrendDirection::Stable,
            magnitude: 0.0,
            confidence: 0.8,
        }
    }
}

#[derive(Debug)]
pub struct PerformanceTrend {
    pub direction: TrendDirection,
    pub magnitude: f64,
    pub confidence: f64,
}

#[derive(Debug)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
}

/// Machine learning-based auto-tuner for optimal parameter selection
#[derive(Debug)]
pub struct AutoTuner {
    /// Feature weights for parameter optimization
    feature_weights: Vec<f64>,
    /// Learning rate for model updates
    learning_rate: f64,
    /// Historical performance data for model training
    training_data: Vec<TrainingExample>,
    /// Maximum number of training examples to keep
    max_training_data: usize,
}

impl Default for AutoTuner {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoTuner {
    pub fn new() -> Self {
        Self {
            feature_weights: vec![0.1; 10], // Initialize with small random weights
            learning_rate: 0.01,
            training_data: Vec::new(),
            max_training_data: 1000,
        }
    }

    pub fn optimize_parameters(
        &mut self,
        system_metrics: &SystemMetrics,
        historical_data: &[&ExecutionRecord],
        estimated_data_size: usize,
    ) -> Result<AutoTuningParameters> {
        // Extract features from system metrics and historical data
        let features = self.extract_features(system_metrics, historical_data, estimated_data_size);

        // Use learned model to predict optimal parameters
        let predicted_params = self.predict_optimal_parameters(&features)?;

        Ok(predicted_params)
    }

    pub fn update_model(
        &mut self,
        config: &OptimizedPipelineConfig,
        metrics: &PipelinePerformanceMetrics,
    ) -> Result<()> {
        // Convert config and metrics to training example
        let training_example = TrainingExample {
            features: self.config_to_features(config),
            performance_score: metrics.throughput,
        };

        self.training_data.push(training_example);

        // Maintain training data size limit
        if self.training_data.len() > self.max_training_data {
            self.training_data.remove(0);
        }

        // Update model weights using gradient descent
        self.update_weights()?;

        Ok(())
    }

    fn extract_features(
        &self,
        system_metrics: &SystemMetrics,
        historical_data: &[&ExecutionRecord],
        estimated_data_size: usize,
    ) -> Vec<f64> {
        let mut features = Vec::new();

        // System metrics features
        features.push(system_metrics.cpu_usage);
        features.push(system_metrics.memory_usage.utilization);
        features.push(system_metrics.io_utilization);
        features.push(system_metrics.cache_performance.l1_hit_rate);

        // Data size feature (normalized)
        features.push((estimated_data_size as f64).ln() / 20.0); // Log-scale normalization

        // Historical performance features
        if !historical_data.is_empty() {
            let avg_throughput: f64 = historical_data
                .iter()
                .map(|record| record.metrics.throughput)
                .sum::<f64>()
                / historical_data.len() as f64;
            features.push(avg_throughput / 1000.0); // Normalize
        } else {
            features.push(0.0);
        }

        // Add more features as needed
        while features.len() < self.feature_weights.len() {
            features.push(0.0);
        }

        features
    }

    fn predict_optimal_parameters(&self, features: &[f64]) -> Result<AutoTuningParameters> {
        // Simple linear model prediction
        let prediction_score: f64 = features
            .iter()
            .zip(self.feature_weights.iter())
            .map(|(f, w)| f * w)
            .sum();

        // Convert prediction to concrete parameters
        let thread_count = ((prediction_score * 8.0).exp() as usize).clamp(1, num_cpus::get());
        let chunk_size = ((prediction_score * 1000.0).abs() as usize).clamp(256, 8192);
        let simd_enabled = prediction_score > 0.0;
        let gpu_enabled = prediction_score > 0.5 && self.is_gpu_beneficial(features);

        Ok(AutoTuningParameters {
            thread_count,
            chunk_size,
            simd_enabled,
            gpu_enabled,
            prefetch_strategy: super::config::PrefetchStrategy::Sequential { distance: 64 },
            compression_level: (prediction_score * 9.0).abs() as u8,
            io_buffer_size: ((prediction_score * 64.0 * 1024.0).abs() as usize)
                .clamp(4096, 1024 * 1024),
            batch_processing: super::config::BatchProcessingMode::Disabled,
        })
    }

    fn config_to_features(&self, config: &OptimizedPipelineConfig) -> Vec<f64> {
        let mut features = Vec::new();
        features.push(config.thread_count as f64 / num_cpus::get() as f64);
        features.push((config.chunk_size as f64).ln() / 10.0);
        features.push(if config.simd_optimization { 1.0 } else { 0.0 });
        features.push(if config.gpu_acceleration { 1.0 } else { 0.0 });
        features.push(config.compression_level as f64 / 9.0);

        // Pad with zeros if needed
        while features.len() < self.feature_weights.len() {
            features.push(0.0);
        }

        features
    }

    fn update_weights(&mut self) -> Result<()> {
        if self.training_data.len() < 10 {
            return Ok(()); // Need minimum data for training
        }

        // Simple gradient descent update
        for example in &self.training_data {
            let predicted = self.predict_score(&example.features);
            let error = example.performance_score - predicted;

            // Update weights
            for (i, &feature) in example.features.iter().enumerate() {
                if i < self.feature_weights.len() {
                    self.feature_weights[i] += self.learning_rate * error * feature;
                }
            }
        }

        Ok(())
    }

    fn predict_score(&self, features: &[f64]) -> f64 {
        features
            .iter()
            .zip(self.feature_weights.iter())
            .map(|(f, w)| f * w)
            .sum()
    }

    fn is_gpu_beneficial(&self, features: &[f64]) -> bool {
        // Simple heuristic: GPU is beneficial for large data sizes and high parallelizability
        if features.len() >= 5 {
            features[4] > 0.5 // Data size feature
        } else {
            false
        }
    }

    pub fn get_model_accuracy(&self) -> f64 {
        if self.training_data.len() < 10 {
            return 0.0;
        }

        let mut total_error = 0.0;
        for example in &self.training_data {
            let predicted = self.predict_score(&example.features);
            let error = (example.performance_score - predicted).abs();
            total_error += error;
        }

        let mean_error = total_error / self.training_data.len() as f64;
        let mean_performance: f64 = self
            .training_data
            .iter()
            .map(|e| e.performance_score)
            .sum::<f64>()
            / self.training_data.len() as f64;

        if mean_performance > 0.0 {
            1.0 - (mean_error / mean_performance)
        } else {
            0.0
        }
    }
}

/// Training example for the auto-tuner machine learning model
#[derive(Debug, Clone)]
struct TrainingExample {
    features: Vec<f64>,
    performance_score: f64,
}

/// Performance predictor for estimating pipeline execution time and resource usage
#[derive(Debug)]
pub struct PerformancePredictor {
    /// Regression model for throughput prediction
    throughput_model: LinearRegressionModel,
    /// Regression model for memory usage prediction
    memory_model: LinearRegressionModel,
    /// Regression model for CPU utilization prediction
    cpu_model: LinearRegressionModel,
}

impl PerformancePredictor {
    pub fn new() -> Self {
        Self {
            throughput_model: LinearRegressionModel::new(8),
            memory_model: LinearRegressionModel::new(8),
            cpu_model: LinearRegressionModel::new(8),
        }
    }

    pub fn predict_performance(
        &self,
        config: &OptimizedPipelineConfig,
        data_size: usize,
    ) -> PipelinePerformanceMetrics {
        let features = self.extract_prediction_features(config, data_size);

        let predicted_throughput = self.throughput_model.predict(&features).max(0.0);
        let predicted_memory = self.memory_model.predict(&features).max(0.0) as usize;
        let predicted_cpu = self.cpu_model.predict(&features).clamp(0.0, 1.0);

        PipelinePerformanceMetrics {
            throughput: predicted_throughput,
            peak_memory_usage: predicted_memory,
            cpu_utilization: predicted_cpu,
            data_size,
            ..Default::default()
        }
    }

    fn extract_prediction_features(
        &self,
        config: &OptimizedPipelineConfig,
        data_size: usize,
    ) -> Vec<f64> {
        vec![
            config.thread_count as f64,
            (config.chunk_size as f64).ln(),
            if config.simd_optimization { 1.0 } else { 0.0 },
            if config.gpu_acceleration { 1.0 } else { 0.0 },
            (data_size as f64).ln(),
            config.compression_level as f64,
            (config.io_buffer_size as f64).ln(),
            match config.batch_processing {
                super::config::BatchProcessingMode::Disabled => 0.0,
                _ => 1.0,
            },
        ]
    }
}

impl Default for PerformancePredictor {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple linear regression model for performance prediction
#[derive(Debug)]
struct LinearRegressionModel {
    weights: Vec<f64>,
    bias: f64,
}

impl LinearRegressionModel {
    fn new(num_features: usize) -> Self {
        Self {
            weights: vec![0.1; num_features],
            bias: 0.0,
        }
    }

    fn predict(&self, features: &[f64]) -> f64 {
        let prediction: f64 = features
            .iter()
            .zip(self.weights.iter())
            .map(|(f, w)| f * w)
            .sum();
        prediction + self.bias
    }
}
