//! Memory-efficient partitioned reduction over flat tensors.
//!
//! # Design
//!
//! All reductions are processed in fixed-size chunks of at most
//! `PartitionConfig::chunk_size` elements.  This caps peak working-set memory
//! independently of total tensor size.
//!
//! `reduce_axis` implements an axis-wise reduction by iterating over the
//! *slices* of the flattened N-D tensor that correspond to a given axis.

use super::config::{AccumulationStrategy, PartitionConfig};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by [`PartitionedReducer`].
#[derive(Debug, thiserror::Error)]
pub enum PartitionedError {
    #[error("Empty input for reduction")]
    EmptyInput,

    #[error("Chunk size must be > 0, got {0}")]
    InvalidChunkSize(usize),

    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Numerical issue: {0}")]
    NumericalIssue(String),

    #[error("Axis {axis} out of range for shape {ndim}D tensor")]
    AxisOutOfRange { axis: usize, ndim: usize },
}

// ---------------------------------------------------------------------------
// PartitionedStats
// ---------------------------------------------------------------------------

/// Statistics collected during a partitioned reduction.
#[derive(Debug, Clone, Default)]
pub struct PartitionedStats {
    pub chunks_processed: usize,
    pub total_elements_processed: usize,
    pub peak_chunk_size: usize,
}

// ---------------------------------------------------------------------------
// PartitionedReducer
// ---------------------------------------------------------------------------

/// Performs memory-efficient reductions by splitting input into fixed-size
/// chunks and accumulating partial results.
pub struct PartitionedReducer {
    config: PartitionConfig,
    stats: PartitionedStats,
}

impl PartitionedReducer {
    /// Create a new reducer with the given configuration.
    pub fn new(config: PartitionConfig) -> Self {
        PartitionedReducer {
            config,
            stats: PartitionedStats::default(),
        }
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Reduce all elements of a flat 1-D slice to a single scalar.
    pub fn reduce_all(&mut self, data: &[f64]) -> Result<f64, PartitionedError> {
        if data.is_empty() {
            return Err(PartitionedError::EmptyInput);
        }
        if self.config.chunk_size == 0 {
            return Err(PartitionedError::InvalidChunkSize(0));
        }

        if self.config.accumulation == AccumulationStrategy::LogSumExp {
            return self.log_sum_exp(data);
        }

        let (mut acc, needs_count) = self.initial_accumulator();
        let mut total_count = 0usize;

        for chunk in data.chunks(self.config.chunk_size) {
            let chunk_len = chunk.len();
            let chunk_result = self.reduce_chunk(chunk)?;
            acc = self.combine(acc, chunk_result, &self.config.accumulation)?;
            total_count += chunk_len;
            self.stats.chunks_processed += 1;
            self.stats.total_elements_processed += chunk_len;
            if chunk_len > self.stats.peak_chunk_size {
                self.stats.peak_chunk_size = chunk_len;
            }
        }

        if needs_count {
            // Mean: divide accumulated sum by total element count
            let count = total_count as f64;
            if count == 0.0 {
                return Err(PartitionedError::NumericalIssue(
                    "zero element count for mean".to_string(),
                ));
            }
            acc /= count;
        }

        Ok(acc)
    }

    /// Reduce along a single axis of an N-dimensional tensor.
    ///
    /// `data` is the row-major flat representation of a tensor with the given
    /// `shape`.  The returned value is the flat representation of the reduced
    /// tensor together with its shape (the axis dimension is removed).
    pub fn reduce_axis(
        &mut self,
        data: &[f64],
        shape: &[usize],
        axis: usize,
    ) -> Result<(Vec<f64>, Vec<usize>), PartitionedError> {
        if shape.is_empty() {
            return Err(PartitionedError::AxisOutOfRange { axis, ndim: 0 });
        }
        if axis >= shape.len() {
            return Err(PartitionedError::AxisOutOfRange {
                axis,
                ndim: shape.len(),
            });
        }

        let total_elements: usize = shape.iter().product();
        if data.len() != total_elements {
            return Err(PartitionedError::ShapeMismatch {
                expected: shape.to_vec(),
                got: vec![data.len()],
            });
        }
        if data.is_empty() {
            return Err(PartitionedError::EmptyInput);
        }

        // Compute output shape (remove the axis dimension)
        let out_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &d)| d)
            .collect();
        let out_len: usize = out_shape.iter().product::<usize>().max(1);

        // stride_before: product of dims before the axis
        // axis_len: size of the reduced axis
        // stride_after: product of dims after the axis
        let stride_before: usize = shape[..axis].iter().product::<usize>().max(1);
        let axis_len: usize = shape[axis];
        let stride_after: usize = shape[axis + 1..].iter().product::<usize>().max(1);

        let mut out = vec![self.initial_scalar(); out_len];
        let mut counts = vec![0usize; out_len];

        // For each element in the output we accumulate all axis values in
        // chunks to stay within memory budget.
        for before in 0..stride_before {
            for after in 0..stride_after {
                let out_idx = before * stride_after + after;
                // Collect all values along this axis, then reduce
                let values: Vec<f64> = (0..axis_len)
                    .map(|k| data[before * axis_len * stride_after + k * stride_after + after])
                    .collect();

                // Use reduce_all with a temporary config for the strategy
                let mut tmp = PartitionedReducer::new(self.config.clone());
                let reduced = tmp.reduce_all(&values).map_err(|e| match e {
                    PartitionedError::EmptyInput => PartitionedError::EmptyInput,
                    other => other,
                })?;
                self.stats.chunks_processed += tmp.stats.chunks_processed;
                self.stats.total_elements_processed += tmp.stats.total_elements_processed;
                if tmp.stats.peak_chunk_size > self.stats.peak_chunk_size {
                    self.stats.peak_chunk_size = tmp.stats.peak_chunk_size;
                }

                out[out_idx] = reduced;
                counts[out_idx] += axis_len;
            }
        }

        // For Mean, the reduce_all already divided by count, nothing more needed.
        let _ = counts;

        Ok((out, out_shape))
    }

    /// Numerically stable log-sum-exp: `log(Σ exp(x_i))`.
    ///
    /// Uses the max-subtraction trick to avoid overflow.
    pub fn log_sum_exp(&self, data: &[f64]) -> Result<f64, PartitionedError> {
        if data.is_empty() {
            return Err(PartitionedError::EmptyInput);
        }

        // Pass 1: find max value (chunked to reuse chunk_size discipline)
        let mut global_max = f64::NEG_INFINITY;
        for chunk in data.chunks(self.config.chunk_size.max(1)) {
            for &x in chunk {
                if x > global_max {
                    global_max = x;
                }
            }
        }

        if !global_max.is_finite() {
            return Err(PartitionedError::NumericalIssue(
                "all -inf values in log_sum_exp input".to_string(),
            ));
        }

        // Pass 2: accumulate shifted exponentials
        let mut sum_exp = 0.0_f64;
        for chunk in data.chunks(self.config.chunk_size.max(1)) {
            for &x in chunk {
                sum_exp += (x - global_max).exp();
            }
        }

        if sum_exp <= 0.0 || !sum_exp.is_finite() {
            return Err(PartitionedError::NumericalIssue(format!(
                "sum_exp={sum_exp} after max subtraction"
            )));
        }

        Ok(global_max + sum_exp.ln())
    }

    /// Return the accumulated statistics since the last reset.
    pub fn stats(&self) -> &PartitionedStats {
        &self.stats
    }

    /// Reset accumulated statistics.
    pub fn reset_stats(&mut self) {
        self.stats = PartitionedStats::default();
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Reduce a single chunk according to the configured strategy.
    fn reduce_chunk(&self, chunk: &[f64]) -> Result<f64, PartitionedError> {
        if chunk.is_empty() {
            return Err(PartitionedError::EmptyInput);
        }
        match self.config.accumulation {
            AccumulationStrategy::Sum | AccumulationStrategy::Mean => Ok(chunk.iter().sum::<f64>()),
            AccumulationStrategy::Max => chunk
                .iter()
                .copied()
                .reduce(f64::max)
                .ok_or(PartitionedError::EmptyInput),
            AccumulationStrategy::Min => chunk
                .iter()
                .copied()
                .reduce(f64::min)
                .ok_or(PartitionedError::EmptyInput),
            AccumulationStrategy::Product => Ok(chunk.iter().product::<f64>()),
            AccumulationStrategy::LogSumExp => {
                // Handled separately in reduce_all
                Err(PartitionedError::NumericalIssue(
                    "LogSumExp should be routed through log_sum_exp()".to_string(),
                ))
            }
        }
    }

    /// Combine a running accumulator with a new chunk result.
    fn combine(
        &self,
        acc: f64,
        new_val: f64,
        strategy: &AccumulationStrategy,
    ) -> Result<f64, PartitionedError> {
        match strategy {
            AccumulationStrategy::Sum | AccumulationStrategy::Mean => Ok(acc + new_val),
            AccumulationStrategy::Max => Ok(acc.max(new_val)),
            AccumulationStrategy::Min => Ok(acc.min(new_val)),
            AccumulationStrategy::Product => Ok(acc * new_val),
            AccumulationStrategy::LogSumExp => Err(PartitionedError::NumericalIssue(
                "LogSumExp should be routed through log_sum_exp()".to_string(),
            )),
        }
    }

    /// Initial accumulator value for a given strategy.
    fn initial_accumulator(&self) -> (f64, bool) {
        match self.config.accumulation {
            AccumulationStrategy::Sum => (0.0, false),
            AccumulationStrategy::Mean => (0.0, true), // divide by count at end
            AccumulationStrategy::Max => (f64::NEG_INFINITY, false),
            AccumulationStrategy::Min => (f64::INFINITY, false),
            AccumulationStrategy::Product => (1.0, false),
            AccumulationStrategy::LogSumExp => (0.0, false),
        }
    }

    /// Scalar identity for axis reduction initialisation.
    fn initial_scalar(&self) -> f64 {
        match self.config.accumulation {
            AccumulationStrategy::Sum | AccumulationStrategy::Mean => 0.0,
            AccumulationStrategy::Max => f64::NEG_INFINITY,
            AccumulationStrategy::Min => f64::INFINITY,
            AccumulationStrategy::Product => 1.0,
            AccumulationStrategy::LogSumExp => 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_reducer(strategy: AccumulationStrategy) -> PartitionedReducer {
        let cfg = PartitionConfig::new(4).with_strategy(strategy);
        PartitionedReducer::new(cfg)
    }

    #[test]
    fn test_reduce_all_sum() {
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let mut r = make_reducer(AccumulationStrategy::Sum);
        let result = r.reduce_all(&data).expect("sum ok");
        assert!((result - 55.0).abs() < 1e-12, "sum={result} expected=55");
    }

    #[test]
    fn test_reduce_all_max() {
        let data = vec![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let mut r = make_reducer(AccumulationStrategy::Max);
        let result = r.reduce_all(&data).expect("max ok");
        assert!((result - 9.0).abs() < 1e-12, "max={result} expected=9");
    }

    #[test]
    fn test_reduce_all_min() {
        let data = vec![3.0_f64, 1.0, 4.0, 1.0, 5.0, -2.0, 9.0, 6.0];
        let mut r = make_reducer(AccumulationStrategy::Min);
        let result = r.reduce_all(&data).expect("min ok");
        assert!((result - (-2.0)).abs() < 1e-12, "min={result} expected=-2");
    }

    #[test]
    fn test_reduce_all_mean() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let mut r = make_reducer(AccumulationStrategy::Mean);
        let result = r.reduce_all(&data).expect("mean ok");
        // Mean of 1..5 = 15/5 = 3.0
        assert!((result - 3.0).abs() < 1e-10, "mean={result} expected=3.0");
    }

    #[test]
    fn test_log_sum_exp_numerically_stable() {
        // log(exp(1000) + exp(1001)) = log(exp(1000)(1 + exp(1))) = 1000 + log(1 + e)
        let data = vec![1000.0_f64, 1001.0];
        let cfg = PartitionConfig::new(16).with_strategy(AccumulationStrategy::LogSumExp);
        let r = PartitionedReducer::new(cfg);
        let result = r.log_sum_exp(&data).expect("lse ok");
        let expected = 1000.0_f64 + (1.0_f64 + std::f64::consts::E).ln();
        assert!(
            (result - expected).abs() < 1e-10,
            "lse={result} expected={expected}"
        );
    }

    #[test]
    fn test_empty_input_error() {
        let mut r = make_reducer(AccumulationStrategy::Sum);
        let err = r.reduce_all(&[]);
        assert!(
            matches!(err, Err(PartitionedError::EmptyInput)),
            "expected EmptyInput error"
        );
    }
}
