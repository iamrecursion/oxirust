//! Custom partitioning strategies for different data distributions
//!
//! This module provides advanced partitioning strategies that adapt to
//! various data distributions for optimal load balancing in parallel processing.
//! It includes support for uniform, skewed, Gaussian, and custom distributions.

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use crate::parallel_ops::*;
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::time::Duration;

/// Data distribution types that affect partitioning strategy
#[derive(Debug, Clone, PartialEq)]
pub enum DataDistribution {
    /// Uniform distribution - data is evenly distributed
    Uniform,
    /// Skewed distribution - data is concentrated in certain regions
    Skewed {
        /// Skewness factor (0.0 = no skew, positive = right skew, negative = left skew)
        skewness: f64,
    },
    /// Gaussian/Normal distribution
    Gaussian {
        /// Mean of the distribution
        mean: f64,
        /// Standard deviation
        std_dev: f64,
    },
    /// Power law distribution (e.g., Zipf distribution)
    PowerLaw {
        /// Exponent parameter
        alpha: f64,
    },
    /// Bimodal distribution - two peaks
    Bimodal {
        /// First peak mean
        mean1: f64,
        /// Second peak mean
        mean2: f64,
        /// Mixing ratio (0.0 to 1.0)
        mix_ratio: f64,
    },
    /// Custom distribution defined by density function
    Custom {
        /// Name or description of the distribution
        name: String,
    },
}

/// Partitioning strategy for dividing work among threads
#[derive(Debug, Clone)]
pub enum PartitionStrategy {
    /// Equal-sized partitions (traditional approach)
    EqualSize,
    /// Weighted partitions based on data distribution
    Weighted {
        /// Weights for each partition
        weights: Vec<f64>,
    },
    /// Dynamic partitioning that adjusts at runtime
    Dynamic {
        /// Initial partition sizes
        initial_sizes: Vec<usize>,
        /// Whether to allow stealing between partitions
        allow_stealing: bool,
    },
    /// Hierarchical partitioning for nested parallelism
    Hierarchical {
        /// Number of levels in the hierarchy
        levels: usize,
        /// Branching factor at each level
        branching_factor: usize,
    },
    /// Range-based partitioning for sorted data
    RangeBased {
        /// Boundary values for each partition
        boundaries: Vec<f64>,
    },
    /// Hash-based partitioning for key-value data
    HashBased {
        /// Number of hash buckets
        num_buckets: usize,
    },
}

/// Configuration for the partitioner
#[derive(Debug, Clone)]
pub struct PartitionerConfig {
    /// Number of partitions (usually number of threads)
    pub num_partitions: usize,
    /// Minimum partition size
    pub min_partition_size: usize,
    /// Maximum partition size (0 for unlimited)
    pub max_partition_size: usize,
    /// Whether to enable load balancing
    pub enable_load_balancing: bool,
    /// Target load imbalance factor (1.0 = perfect balance)
    pub target_imbalance_factor: f64,
    /// Whether to consider NUMA topology
    pub numa_aware: bool,
    /// Whether to enable work stealing
    pub enable_work_stealing: bool,
}

impl Default for PartitionerConfig {
    fn default() -> Self {
        Self {
            num_partitions: num_threads(),
            min_partition_size: 1000,
            max_partition_size: 0,
            enable_load_balancing: true,
            target_imbalance_factor: 1.1,
            numa_aware: false,
            enable_work_stealing: true,
        }
    }
}

/// Partitioner for dividing data based on distribution characteristics
pub struct DataPartitioner<T> {
    config: PartitionerConfig,
    phantom: PhantomData<T>,
}

impl<T> DataPartitioner<T>
where
    T: Send + Sync + Clone,
{
    /// Create a new partitioner with the given configuration
    pub fn new(config: PartitionerConfig) -> Self {
        Self {
            config,
            phantom: PhantomData,
        }
    }

    /// Create a partitioner with default configuration
    pub fn with_defaultconfig() -> Self {
        Self::new(PartitionerConfig::default())
    }

    /// Analyze data to determine its distribution
    pub fn analyze_distribution(&self, data: &[T]) -> DataDistribution
    where
        T: Into<f64> + Copy,
    {
        if data.is_empty() {
            return DataDistribution::Uniform;
        }

        // Convert data to float values for analysis
        let values: Vec<f64> = data.iter().map(|&x| x.into()).collect();

        // Calculate basic statistics
        let n = values.len() as f64;
        let mean = values.iter().copied().sum::<f64>() / n;

        // Calculate variance and standard deviation
        let variance = values
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / n;
        let std_dev = variance.sqrt();

        // Calculate skewness
        let skewness = if std_dev > 0.0 {
            let sum_cubed = values
                .iter()
                .map(|&x| {
                    let z = (x - mean) / std_dev;
                    z * z * z
                })
                .sum::<f64>();
            sum_cubed / n
        } else {
            0.0
        };

        // Calculate kurtosis to detect bimodality
        let kurtosis = if std_dev > 0.0 {
            let sum_fourth = values
                .iter()
                .map(|&x| {
                    let z = (x - mean) / std_dev;
                    z * z * z * z
                })
                .sum::<f64>();
            sum_fourth / n - 3.0
        } else {
            0.0
        };

        // Determine distribution type based on statistics
        if skewness.abs() < 0.5 && kurtosis > -1.5 && kurtosis < -0.8 {
            // Uniform distribution has kurtosis around -1.2
            DataDistribution::Uniform
        } else if skewness.abs() < 0.5 && kurtosis.abs() < 1.0 {
            // Approximately normal
            DataDistribution::Gaussian { mean, std_dev }
        } else if skewness.abs() > 2.0 {
            // Heavily skewed
            DataDistribution::Skewed { skewness }
        } else if kurtosis < -1.5 {
            // Very negative kurtosis may indicate bimodality
            // Simple bimodal detection - in practice would use more sophisticated methods
            DataDistribution::Bimodal {
                mean1: mean - std_dev,
                mean2: mean + std_dev,
                mix_ratio: 0.5,
            }
        } else {
            // Default to uniform if no clear pattern
            DataDistribution::Uniform
        }
    }

    /// Create a partitioning strategy based on data distribution
    pub fn create_strategy(
        &self,
        distribution: &DataDistribution,
        data_size: usize,
    ) -> CoreResult<PartitionStrategy> {
        let num_partitions = self.config.num_partitions;

        match distribution {
            DataDistribution::Uniform => {
                // Equal-sized partitions work well for uniform data
                Ok(PartitionStrategy::EqualSize)
            }

            DataDistribution::Skewed { skewness } => {
                // Create weighted partitions based on skewness
                let weights = self.calculate_skewed_weights(*skewness, num_partitions)?;
                Ok(PartitionStrategy::Weighted { weights })
            }

            DataDistribution::Gaussian { mean, std_dev } => {
                // Create range-based partitions using quantiles
                let boundaries =
                    self.calculate_gaussian_boundaries(*mean, *std_dev, num_partitions)?;
                Ok(PartitionStrategy::RangeBased { boundaries })
            }

            DataDistribution::PowerLaw { alpha } => {
                // Use logarithmic partitioning for power law
                let weights = self.calculate_power_law_weights(*alpha, num_partitions)?;
                Ok(PartitionStrategy::Weighted { weights })
            }

            DataDistribution::Bimodal {
                mean1,
                mean2,
                mix_ratio,
            } => {
                // Create partitions around the two modes
                let boundaries =
                    self.calculate_bimodal_boundaries(*mean1, *mean2, *mix_ratio, num_partitions)?;
                Ok(PartitionStrategy::RangeBased { boundaries })
            }

            DataDistribution::Custom { .. } => {
                // For custom distributions, use dynamic partitioning
                let initial_sizes = vec![data_size / num_partitions; num_partitions];
                Ok(PartitionStrategy::Dynamic {
                    initial_sizes,
                    allow_stealing: self.config.enable_work_stealing,
                })
            }
        }
    }

    /// Partition data according to the given strategy
    pub fn partition(&self, data: &[T], strategy: &PartitionStrategy) -> CoreResult<Vec<Vec<T>>> {
        let data_size = data.len();
        let num_partitions = self.config.num_partitions;

        if data_size < num_partitions * self.config.min_partition_size {
            // Not enough data to partition effectively
            return Ok(vec![data.to_vec()]);
        }

        match strategy {
            PartitionStrategy::EqualSize => self.partition_equal_size(data),

            PartitionStrategy::Weighted { weights } => self.partition_weighted(data, weights),

            PartitionStrategy::Dynamic { initial_sizes, .. } => {
                self.partition_dynamic(data, initial_sizes)
            }

            PartitionStrategy::Hierarchical {
                levels,
                branching_factor,
            } => self.partition_hierarchical(data, *levels, *branching_factor),

            PartitionStrategy::RangeBased { boundaries: _ } => {
                // Range-based partitioning requires specific trait bounds
                // For now, return an error if T doesn't meet requirements
                Err(CoreError::InvalidArgument(
                    ErrorContext::new(
                        "Range-based partitioning requires PartialOrd + Into<f64> + Copy traits"
                            .to_string(),
                    )
                    .with_location(ErrorLocation::new(file!(), line!())),
                ))
            }

            PartitionStrategy::HashBased { num_buckets: _ } => {
                // Hash-based partitioning requires Hash trait
                // For now, return an error if T doesn't implement Hash
                Err(CoreError::InvalidArgument(
                    ErrorContext::new("Hash-based partitioning requires Hash trait".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                ))
            }
        }
    }

    /// Partition data into equal-sized chunks
    fn partition_equal_size(&self, data: &[T]) -> CoreResult<Vec<Vec<T>>> {
        let chunk_size = data.len().div_ceil(self.config.num_partitions);
        let mut partitions = Vec::with_capacity(self.config.num_partitions);

        for chunk in data.chunks(chunk_size) {
            partitions.push(chunk.to_vec());
        }

        Ok(partitions)
    }

    /// Partition data according to weights
    fn partition_weighted(&self, data: &[T], weights: &[f64]) -> CoreResult<Vec<Vec<T>>> {
        if weights.len() != self.config.num_partitions {
            return Err(CoreError::InvalidArgument(
                ErrorContext::new("Weight count does not match partition count".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        let total_weight: f64 = weights.iter().sum();
        if total_weight <= 0.0 {
            return Err(CoreError::InvalidArgument(
                ErrorContext::new("Total weight must be positive".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        let mut partitions = Vec::with_capacity(self.config.num_partitions);
        let mut start = 0;

        for weight in weights {
            let size = ((weight / total_weight) * data.len() as f64) as usize;
            let end = (start + size).min(data.len());

            if start < data.len() {
                partitions.push(data[start..end].to_vec());
            }

            start = end;
        }

        // Add any remaining elements to the last partition
        if start < data.len() && !partitions.is_empty() {
            if let Some(last) = partitions.last_mut() {
                last.extend_from_slice(&data[start..]);
            }
        }

        Ok(partitions)
    }

    /// Dynamic partitioning with runtime adjustment
    fn partition_dynamic(&self, data: &[T], initialsizes: &[usize]) -> CoreResult<Vec<Vec<T>>> {
        // For now, use initial sizes as-is
        // In a full implementation, this would monitor progress and adjust
        let mut partitions = Vec::with_capacity(initialsizes.len());
        let mut start = 0;

        for &size in initialsizes {
            let end = (start + size).min(data.len());
            if start < data.len() {
                partitions.push(data[start..end].to_vec());
            }
            start = end;
        }

        Ok(partitions)
    }

    /// Hierarchical partitioning for nested parallelism
    fn partition_hierarchical(
        &self,
        data: &[T],
        levels: usize,
        branching_factor: usize,
    ) -> CoreResult<Vec<Vec<T>>> {
        if levels == 0 || branching_factor == 0 {
            return Err(CoreError::InvalidArgument(
                ErrorContext::new("Invalid hierarchical parameters".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Calculate total number of leaf partitions
        let num_leaves = branching_factor.pow(levels as u32);
        let chunk_size = data.len().div_ceil(num_leaves);

        let mut partitions = Vec::with_capacity(num_leaves);
        for chunk in data.chunks(chunk_size) {
            partitions.push(chunk.to_vec());
        }

        Ok(partitions)
    }

    /// Range-based partitioning for sorted data
    #[allow(dead_code)]
    fn partition_rangebased(&self, data: &[T], boundaries: &[f64]) -> CoreResult<Vec<Vec<T>>>
    where
        T: PartialOrd + Into<f64> + Copy,
    {
        let mut partitions = vec![Vec::new(); boundaries.len() + 1];

        for &item in data {
            let value: f64 = item.into();
            let mut partition_idx = boundaries.len();

            for (i, &boundary) in boundaries.iter().enumerate() {
                if value <= boundary {
                    partition_idx = i;
                    break;
                }
            }

            partitions[partition_idx].push(item);
        }

        Ok(partitions)
    }

    /// Hash-based partitioning
    #[allow(dead_code)]
    fn partition_hashbased(&self, data: &[T], numbuckets: usize) -> CoreResult<Vec<Vec<T>>>
    where
        T: std::hash::Hash,
    {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut partitions = vec![Vec::new(); numbuckets];

        for item in data {
            let mut hasher = DefaultHasher::new();
            item.hash(&mut hasher);
            let hash = hasher.finish();
            let bucket = (hash % numbuckets as u64) as usize;
            partitions[bucket].push(item.clone());
        }

        Ok(partitions)
    }

    /// Calculate weights for skewed distribution
    fn calculate_skewed_weights(
        &self,
        skewness: f64,
        num_partitions: usize,
    ) -> CoreResult<Vec<f64>> {
        let mut weights = Vec::with_capacity(num_partitions);

        // Use exponential weights for skewed distributions
        let base = 1.0 + skewness.abs() / 10.0;

        for i in 0..num_partitions {
            let weight = if skewness > 0.0 {
                // Right skew - more weight on early partitions
                base.powf((num_partitions - i.saturating_sub(1)) as f64)
            } else {
                // Left skew - more weight on later partitions
                base.powf(i as f64)
            };
            weights.push(weight);
        }

        Ok(weights)
    }

    /// Calculate boundaries for Gaussian distribution
    fn calculate_gaussian_boundaries(
        &self,
        mean: f64,
        std_dev: f64,
        num_partitions: usize,
    ) -> CoreResult<Vec<f64>> {
        let mut boundaries = Vec::with_capacity(num_partitions - 1);

        // Use simplified quantile approximation for normal distribution
        // For a standard normal distribution, we can use the inverse error function
        for i in 1..num_partitions {
            let quantile = i as f64 / num_partitions as f64;
            // Simple approximation of the inverse normal CDF
            // For more accuracy, would use a proper inverse normal CDF implementation
            let z_score = if quantile == 0.5 {
                0.0
            } else if quantile < 0.5 {
                // Approximation for left tail
                -((1.0 - 2.0 * quantile).ln() * 2.0).sqrt()
            } else {
                // Approximation for right tail
                ((2.0 * quantile - 1.0).ln() * 2.0).sqrt()
            };
            let boundary = mean + z_score * std_dev;
            boundaries.push(boundary);
        }

        Ok(boundaries)
    }

    /// Calculate weights for power law distribution
    fn calculate_power_law_weights(
        &self,
        alpha: f64,
        num_partitions: usize,
    ) -> CoreResult<Vec<f64>> {
        let mut weights = Vec::with_capacity(num_partitions);

        for i in 0..num_partitions {
            // Power law: weight ∝ (i+1)^(-alpha)
            let weight = ((i + 1) as f64).powf(-alpha);
            weights.push(weight);
        }

        Ok(weights)
    }

    /// Calculate boundaries for bimodal distribution
    fn calculate_bimodal_boundaries(
        &self,
        mean1: f64,
        mean2: f64,
        mix_ratio: f64,
        num_partitions: usize,
    ) -> CoreResult<Vec<f64>> {
        let mut boundaries = Vec::with_capacity(num_partitions - 1);

        // Split partitions between the two modes
        let partitions_mode1 = ((num_partitions as f64) * mix_ratio) as usize;
        let partitions_mode2 = num_partitions - partitions_mode1;

        // Create boundaries around first mode
        let range1 = (mean2 - mean1).abs() * 0.5;
        for i in 1..partitions_mode1 {
            let boundary = mean1 - range1 * 0.5 + (range1 / partitions_mode1 as f64) * i as f64;
            boundaries.push(boundary);
        }

        // Boundary between modes
        boundaries.push((mean1 + mean2) / 2.0);

        // Create boundaries around second mode
        for i in 1..partitions_mode2 {
            let boundary = mean2 - range1 * 0.5 + (range1 / partitions_mode2 as f64) * i as f64;
            boundaries.push(boundary);
        }

        boundaries.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        Ok(boundaries)
    }
}

/// Load balancer for runtime adjustment of partitions
pub struct LoadBalancer {
    /// Target imbalance factor
    target_imbalance: f64,
    /// History of partition execution times
    execution_times: Vec<Vec<Duration>>,
    /// Current partition weights
    weights: Vec<f64>,
}

impl LoadBalancer {
    /// Create a new load balancer
    pub fn new(num_partitions: usize, targetimbalance: f64) -> Self {
        Self {
            target_imbalance: targetimbalance,
            execution_times: vec![Vec::new(); num_partitions],
            weights: vec![1.0; num_partitions],
        }
    }

    /// Record execution time for a partition
    pub fn recordexecution_time(&mut self, partitionid: usize, duration: Duration) {
        if partitionid < self.execution_times.len() {
            self.execution_times[partitionid].push(duration);

            // Keep only recent history (last 10 measurements)
            if self.execution_times[partitionid].len() > 10 {
                self.execution_times[partitionid].remove(0);
            }
        }
    }

    /// Calculate new weights based on execution history
    pub fn rebalance(&mut self) -> Vec<f64> {
        let mut avg_times = Vec::with_capacity(self.weights.len());

        // Calculate average execution time for each partition
        for times in &self.execution_times {
            if times.is_empty() {
                avg_times.push(1.0);
            } else {
                let sum: Duration = times.iter().sum();
                let avg = sum.as_secs_f64() / times.len() as f64;
                avg_times.push(avg);
            }
        }

        // Calculate total average time
        let total_avg: f64 = avg_times.iter().sum();
        let mean_time = total_avg / avg_times.len() as f64;

        // Adjust weights inversely proportional to execution time
        for (i, &avg_time) in avg_times.iter().enumerate() {
            if avg_time > mean_time * self.target_imbalance {
                // This partition is too slow, reduce its weight
                self.weights[i] *= 0.9;
            } else if avg_time < mean_time / self.target_imbalance {
                // This partition is too fast, increase its weight
                self.weights[i] *= 1.1;
            }

            // Keep weights within reasonable bounds
            self.weights[i] = self.weights[i].clamp(0.1, 10.0);
        }

        self.weights.clone()
    }

    /// Get current load imbalance factor
    pub fn get_imbalance_factor(&self) -> f64 {
        let mut min_time = f64::MAX;
        let mut max_time = 0.0f64;

        for times in &self.execution_times {
            if !times.is_empty() {
                let avg: f64 =
                    times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / times.len() as f64;
                min_time = min_time.min(avg);
                max_time = max_time.max(avg);
            }
        }

        if min_time > 0.0 {
            max_time / min_time
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_distribution_detection() {
        let partitioner = DataPartitioner::<f64>::with_defaultconfig();
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();

        let distribution = partitioner.analyze_distribution(&data);
        match distribution {
            DataDistribution::Uniform | DataDistribution::Gaussian { .. } => {
                // Uniform sequence might be detected as either
            }
            _ => panic!("Expected uniform or gaussian distribution"),
        }
    }

    #[test]
    fn test_skewed_distribution_detection() {
        let partitioner = DataPartitioner::<f64>::with_defaultconfig();
        // Create heavily skewed data
        let mut data = vec![1.0; 900];
        data.extend(vec![100.0; 100]);

        let distribution = partitioner.analyze_distribution(&data);
        match distribution {
            DataDistribution::Skewed { skewness } => {
                assert!(skewness > 2.0, "Expected high positive skewness");
            }
            _ => panic!("Expected skewed distribution"),
        }
    }

    #[test]
    fn test_equal_size_partitioning() {
        let config = PartitionerConfig {
            num_partitions: 4,
            ..Default::default()
        };
        let partitioner = DataPartitioner::<i32>::new(config);
        let data: Vec<i32> = (0..100).collect();

        let partitions = partitioner
            .partition_equal_size(&data)
            .expect("Operation failed");
        assert_eq!(partitions.len(), 4);
        assert_eq!(partitions[0].len(), 25);
        assert_eq!(partitions[3].len(), 25);
    }

    #[test]
    fn test_weighted_partitioning() {
        let config = PartitionerConfig {
            num_partitions: 3,
            ..Default::default()
        };
        let partitioner = DataPartitioner::<i32>::new(config);
        let data: Vec<i32> = (0..90).collect();
        let weights = vec![1.0, 2.0, 3.0];

        let partitions = partitioner
            .partition_weighted(&data, &weights)
            .expect("Operation failed");
        assert_eq!(partitions.len(), 3);
        assert_eq!(partitions[0].len(), 15); // 1/6 of 90
        assert_eq!(partitions[1].len(), 30); // 2/6 of 90
        assert_eq!(partitions[2].len(), 45); // 3/6 of 90
    }

    #[test]
    fn test_load_balancer() {
        let mut balancer = LoadBalancer::new(3, 1.2);

        // Record some execution times
        use std::time::Duration;
        balancer.recordexecution_time(0, Duration::from_millis(100));
        balancer.recordexecution_time(1, Duration::from_millis(200));
        balancer.recordexecution_time(2, Duration::from_millis(150));

        let new_weights = balancer.rebalance();
        assert_eq!(new_weights.len(), 3);

        // Partition 1 was slowest, should have reduced weight
        assert!(new_weights[1] < new_weights[0]);
    }
}
