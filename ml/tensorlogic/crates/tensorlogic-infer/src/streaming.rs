//! Streaming execution support for large graphs and datasets.

use tensorlogic_ir::EinsumGraph;

use crate::batch::BatchResult;

/// Streaming execution mode for handling large datasets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingMode {
    /// Process all at once (no streaming)
    None,
    /// Stream inputs in fixed-size chunks
    FixedChunk(usize),
    /// Stream with dynamic chunk sizing based on memory
    DynamicChunk { target_memory_mb: usize },
    /// Stream with adaptive chunking based on performance
    Adaptive { initial_chunk: usize },
}

/// Configuration for streaming execution
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub mode: StreamingMode,
    pub prefetch_chunks: usize,
    pub overlap_compute_io: bool,
    pub checkpoint_interval: Option<usize>,
}

impl StreamingConfig {
    pub fn new(mode: StreamingMode) -> Self {
        StreamingConfig {
            mode,
            prefetch_chunks: 1,
            overlap_compute_io: true,
            checkpoint_interval: None,
        }
    }

    pub fn with_prefetch(mut self, num_chunks: usize) -> Self {
        self.prefetch_chunks = num_chunks;
        self
    }

    pub fn with_checkpointing(mut self, interval: usize) -> Self {
        self.checkpoint_interval = Some(interval);
        self
    }

    pub fn disable_overlap(mut self) -> Self {
        self.overlap_compute_io = false;
        self
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self::new(StreamingMode::None)
    }
}

/// Stream chunk metadata
#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    pub chunk_id: usize,
    pub start_idx: usize,
    pub end_idx: usize,
    pub size: usize,
    pub is_last: bool,
}

impl ChunkMetadata {
    pub fn new(chunk_id: usize, start_idx: usize, end_idx: usize, total_size: usize) -> Self {
        let size = end_idx - start_idx;
        let is_last = end_idx >= total_size;
        ChunkMetadata {
            chunk_id,
            start_idx,
            end_idx,
            size,
            is_last,
        }
    }
}

/// Streaming execution result with chunk information
#[derive(Debug, Clone)]
pub struct StreamResult<T> {
    pub outputs: Vec<T>,
    pub metadata: ChunkMetadata,
    pub processing_time_ms: f64,
}

impl<T> StreamResult<T> {
    pub fn new(outputs: Vec<T>, metadata: ChunkMetadata, processing_time_ms: f64) -> Self {
        StreamResult {
            outputs,
            metadata,
            processing_time_ms,
        }
    }

    pub fn throughput_items_per_sec(&self) -> f64 {
        if self.processing_time_ms > 0.0 {
            (self.metadata.size as f64) / (self.processing_time_ms / 1000.0)
        } else {
            0.0
        }
    }
}

/// Trait for executors that support streaming execution
pub trait TlStreamingExecutor {
    type Tensor;
    type Error;

    /// Execute graph on a stream of input chunks
    fn execute_stream(
        &mut self,
        graph: &EinsumGraph,
        input_stream: Vec<Vec<Vec<Self::Tensor>>>,
        config: &StreamingConfig,
    ) -> Result<Vec<StreamResult<Self::Tensor>>, Self::Error>;

    /// Execute graph on a single chunk with metadata
    fn execute_chunk(
        &mut self,
        graph: &EinsumGraph,
        chunk_inputs: Vec<Self::Tensor>,
        metadata: &ChunkMetadata,
    ) -> Result<StreamResult<Self::Tensor>, Self::Error>;

    /// Get recommended chunk size based on available memory
    fn recommend_chunk_size(&self, graph: &EinsumGraph, available_memory_mb: usize) -> usize {
        let _ = (graph, available_memory_mb);
        32 // Default recommendation
    }

    /// Estimate memory usage per chunk
    fn estimate_chunk_memory(&self, graph: &EinsumGraph, chunk_size: usize) -> usize {
        let _ = (graph, chunk_size);
        chunk_size * 1024 * 1024 // Default: 1MB per item
    }
}

/// Chunk iterator for breaking large batches into streams
pub struct ChunkIterator {
    total_size: usize,
    chunk_size: usize,
    current_chunk: usize,
}

impl ChunkIterator {
    pub fn new(total_size: usize, chunk_size: usize) -> Self {
        ChunkIterator {
            total_size,
            chunk_size,
            current_chunk: 0,
        }
    }

    pub fn from_config(total_size: usize, config: &StreamingConfig) -> Self {
        let chunk_size = match config.mode {
            StreamingMode::None => total_size,
            StreamingMode::FixedChunk(size) => size,
            StreamingMode::DynamicChunk { target_memory_mb } => {
                // Estimate: ~1MB per item, adjust based on target memory
                (target_memory_mb).max(1)
            }
            StreamingMode::Adaptive { initial_chunk } => initial_chunk,
        };

        ChunkIterator::new(total_size, chunk_size)
    }

    pub fn num_chunks(&self) -> usize {
        self.total_size.div_ceil(self.chunk_size)
    }

    pub fn current_chunk(&self) -> usize {
        self.current_chunk
    }
}

impl Iterator for ChunkIterator {
    type Item = ChunkMetadata;

    fn next(&mut self) -> Option<Self::Item> {
        let start_idx = self.current_chunk * self.chunk_size;
        if start_idx >= self.total_size {
            return None;
        }

        let end_idx = (start_idx + self.chunk_size).min(self.total_size);
        let metadata = ChunkMetadata::new(self.current_chunk, start_idx, end_idx, self.total_size);

        self.current_chunk += 1;
        Some(metadata)
    }
}

/// Stream processor for handling streaming execution
pub struct StreamProcessor {
    config: StreamingConfig,
}

impl StreamProcessor {
    pub fn new(config: StreamingConfig) -> Self {
        StreamProcessor { config }
    }

    /// Split batch result into chunks based on configuration
    pub fn split_batch<T: Clone>(&self, batch: &BatchResult<T>) -> Vec<(ChunkMetadata, Vec<T>)> {
        let total_size = batch.len();
        let iter = ChunkIterator::from_config(total_size, &self.config);

        iter.map(|metadata| {
            let chunk_data: Vec<T> = batch.outputs[metadata.start_idx..metadata.end_idx].to_vec();
            (metadata, chunk_data)
        })
        .collect()
    }

    /// Merge stream results back into a single batch
    pub fn merge_results<T>(results: Vec<StreamResult<T>>) -> BatchResult<T> {
        let total_size: usize = results.iter().map(|r| r.outputs.len()).sum();
        let mut outputs = Vec::with_capacity(total_size);

        for result in results {
            outputs.extend(result.outputs);
        }

        BatchResult::new(outputs)
    }

    /// Calculate adaptive chunk size based on performance metrics
    pub fn adaptive_chunk_size(&self, results: &[StreamResult<impl Clone>]) -> usize {
        if results.is_empty() {
            return 32; // Default
        }

        // Calculate average throughput
        let avg_throughput: f64 = results
            .iter()
            .map(|r| r.throughput_items_per_sec())
            .sum::<f64>()
            / results.len() as f64;

        // Adjust chunk size based on throughput
        // Goal: maintain ~100ms per chunk for good responsiveness
        let target_time_ms = 100.0;
        let items_per_chunk = (avg_throughput * target_time_ms / 1000.0) as usize;

        items_per_chunk.clamp(1, 1000) // Clamp between 1 and 1000
    }

    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }
}

impl Default for StreamProcessor {
    fn default() -> Self {
        Self::new(StreamingConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config() {
        let config = StreamingConfig::new(StreamingMode::FixedChunk(64))
            .with_prefetch(2)
            .with_checkpointing(100);

        assert_eq!(config.mode, StreamingMode::FixedChunk(64));
        assert_eq!(config.prefetch_chunks, 2);
        assert_eq!(config.checkpoint_interval, Some(100));
    }

    #[test]
    fn test_chunk_metadata() {
        let metadata = ChunkMetadata::new(0, 0, 32, 100);
        assert_eq!(metadata.chunk_id, 0);
        assert_eq!(metadata.size, 32);
        assert!(!metadata.is_last);

        let last_metadata = ChunkMetadata::new(3, 96, 100, 100);
        assert!(last_metadata.is_last);
    }

    #[test]
    fn test_stream_result() {
        let metadata = ChunkMetadata::new(0, 0, 32, 100);
        let result: StreamResult<i32> = StreamResult::new(vec![1, 2, 3], metadata, 100.0);

        assert_eq!(result.outputs.len(), 3);
        let throughput = result.throughput_items_per_sec();
        assert!(throughput > 0.0);
    }

    #[test]
    fn test_chunk_iterator() {
        let iter = ChunkIterator::new(100, 32);
        assert_eq!(iter.num_chunks(), 4); // 32, 32, 32, 4

        let chunks: Vec<_> = iter.collect();
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].size, 32);
        assert_eq!(chunks[3].size, 4);
        assert!(chunks[3].is_last);
    }

    #[test]
    fn test_chunk_iterator_from_config() {
        let config = StreamingConfig::new(StreamingMode::FixedChunk(25));
        let iter = ChunkIterator::from_config(100, &config);

        assert_eq!(iter.chunk_size, 25);
        assert_eq!(iter.num_chunks(), 4);
    }

    #[test]
    fn test_stream_processor_split() {
        let batch = BatchResult::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let config = StreamingConfig::new(StreamingMode::FixedChunk(3));
        let processor = StreamProcessor::new(config);

        let chunks = processor.split_batch(&batch);
        assert_eq!(chunks.len(), 4); // 3, 3, 3, 1

        assert_eq!(chunks[0].1, vec![1, 2, 3]);
        assert_eq!(chunks[1].1, vec![4, 5, 6]);
        assert_eq!(chunks[2].1, vec![7, 8, 9]);
        assert_eq!(chunks[3].1, vec![10]);
    }

    #[test]
    fn test_stream_processor_merge() {
        let metadata1 = ChunkMetadata::new(0, 0, 3, 10);
        let metadata2 = ChunkMetadata::new(1, 3, 6, 10);
        let metadata3 = ChunkMetadata::new(2, 6, 10, 10);

        let results = vec![
            StreamResult::new(vec![1, 2, 3], metadata1, 10.0),
            StreamResult::new(vec![4, 5, 6], metadata2, 10.0),
            StreamResult::new(vec![7, 8, 9, 10], metadata3, 10.0),
        ];

        let batch = StreamProcessor::merge_results(results);
        assert_eq!(batch.len(), 10);
        assert_eq!(batch.outputs, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_adaptive_chunk_size() {
        let processor = StreamProcessor::default();

        let metadata = ChunkMetadata::new(0, 0, 100, 1000);
        let results = vec![
            StreamResult::new(vec![(); 100], metadata.clone(), 50.0), // 2000 items/sec
            StreamResult::new(vec![(); 100], metadata.clone(), 100.0), // 1000 items/sec
            StreamResult::new(vec![(); 100], metadata, 75.0),         // 1333 items/sec
        ];

        let chunk_size = processor.adaptive_chunk_size(&results);
        assert!(chunk_size > 0);
        assert!(chunk_size <= 1000); // Within clamp range
    }

    #[test]
    fn test_streaming_modes() {
        assert_eq!(StreamingMode::None, StreamingConfig::default().mode);

        let fixed = StreamingMode::FixedChunk(64);
        assert_eq!(fixed, StreamingMode::FixedChunk(64));

        let dynamic = StreamingMode::DynamicChunk {
            target_memory_mb: 512,
        };
        match dynamic {
            StreamingMode::DynamicChunk { target_memory_mb } => {
                assert_eq!(target_memory_mb, 512);
            }
            _ => panic!("Wrong mode"),
        }
    }
}

// ============================================================
// V2 Extensions: Backpressure, Watermarks, StreamingConfigV2,
// and StreamingStats
// ============================================================

/// Strategy applied when the buffer is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressureStrategy {
    /// Block until space is available.
    Block,
    /// Drop the oldest buffered chunk to make room.
    DropOldest,
    /// Drop the newly arriving chunk.
    DropNewest,
    /// Return an error when the buffer is full.
    ErrorOnFull,
}

/// Backpressure configuration for stream processing.
#[derive(Debug, Clone)]
pub struct BackpressureConfig {
    /// Maximum number of buffered chunks before backpressure activates.
    pub max_buffered_chunks: usize,
    /// High watermark as a fraction of `max_buffered_chunks` (0.0–1.0).
    /// Backpressure activates when the buffer exceeds this fraction.
    pub high_watermark: f64,
    /// Low watermark as a fraction (0.0–1.0).
    /// Normal processing resumes when the buffer drops below this fraction.
    pub low_watermark: f64,
    /// Strategy to apply when the buffer is full.
    pub strategy: BackpressureStrategy,
}

impl BackpressureConfig {
    /// Create a new `BackpressureConfig` with sensible defaults.
    /// Defaults: high_watermark=0.8, low_watermark=0.2, strategy=Block.
    pub fn new(max_buffered: usize) -> Self {
        BackpressureConfig {
            max_buffered_chunks: max_buffered,
            high_watermark: 0.8,
            low_watermark: 0.2,
            strategy: BackpressureStrategy::Block,
        }
    }

    /// Override the high and low watermark fractions.
    pub fn with_watermarks(mut self, high: f64, low: f64) -> Self {
        self.high_watermark = high;
        self.low_watermark = low;
        self
    }

    /// Override the backpressure strategy.
    pub fn with_strategy(mut self, strategy: BackpressureStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Returns `true` when the current buffer level exceeds the high watermark.
    pub fn is_above_high_watermark(&self, current_buffered: usize) -> bool {
        let threshold = (self.max_buffered_chunks as f64 * self.high_watermark) as usize;
        current_buffered > threshold
    }

    /// Returns `true` when the current buffer level is below the low watermark.
    pub fn is_below_low_watermark(&self, current_buffered: usize) -> bool {
        let threshold = (self.max_buffered_chunks as f64 * self.low_watermark) as usize;
        current_buffered < threshold
    }

    /// Returns `true` when backpressure should be applied (buffer is above the high watermark).
    pub fn should_apply_backpressure(&self, current_buffered: usize) -> bool {
        self.is_above_high_watermark(current_buffered)
    }
}

/// Watermark configuration for handling out-of-order events.
#[derive(Debug, Clone)]
pub struct WatermarkConfig {
    /// Maximum allowed out-of-order delay in milliseconds.
    pub max_out_of_order_ms: u64,
    /// Optional idle timeout: emit partial windows after this many ms of silence.
    pub idle_timeout_ms: Option<u64>,
    /// Whether to silently drop events that arrive beyond the watermark.
    pub drop_late_events: bool,
}

impl WatermarkConfig {
    /// Create a new `WatermarkConfig` with the given out-of-order tolerance.
    /// Defaults: idle_timeout_ms=None, drop_late_events=false.
    pub fn new(max_out_of_order_ms: u64) -> Self {
        WatermarkConfig {
            max_out_of_order_ms,
            idle_timeout_ms: None,
            drop_late_events: false,
        }
    }

    /// Set the idle timeout in milliseconds.
    pub fn with_idle_timeout(mut self, timeout_ms: u64) -> Self {
        self.idle_timeout_ms = Some(timeout_ms);
        self
    }

    /// Configure whether late events should be dropped.
    pub fn with_drop_late(mut self, drop: bool) -> Self {
        self.drop_late_events = drop;
        self
    }

    /// Compute the current watermark given the maximum observed event timestamp.
    ///
    /// The watermark is `max_event_time_ms - max_out_of_order_ms`, saturating at zero.
    pub fn current_watermark(&self, max_event_time_ms: u64) -> u64 {
        max_event_time_ms.saturating_sub(self.max_out_of_order_ms)
    }

    /// Returns `true` when `event_time_ms` is behind the current watermark (late event).
    pub fn is_late(&self, event_time_ms: u64, watermark_ms: u64) -> bool {
        event_time_ms < watermark_ms
    }
}

/// Extended streaming configuration combining the base config with v2 features.
#[derive(Debug, Clone)]
pub struct StreamingConfigV2 {
    /// Base streaming configuration.
    pub base: StreamingConfig,
    /// Optional backpressure configuration.
    pub backpressure: Option<BackpressureConfig>,
    /// Optional watermark configuration.
    pub watermark: Option<WatermarkConfig>,
}

impl StreamingConfigV2 {
    /// Create a new `StreamingConfigV2` wrapping the given base config.
    pub fn new(base: StreamingConfig) -> Self {
        StreamingConfigV2 {
            base,
            backpressure: None,
            watermark: None,
        }
    }

    /// Attach a backpressure configuration.
    pub fn with_backpressure(mut self, config: BackpressureConfig) -> Self {
        self.backpressure = Some(config);
        self
    }

    /// Attach a watermark configuration.
    pub fn with_watermark(mut self, config: WatermarkConfig) -> Self {
        self.watermark = Some(config);
        self
    }

    /// Returns `true` when backpressure should be applied for the given buffer level.
    pub fn should_apply_backpressure(&self, current_buffered: usize) -> bool {
        self.backpressure
            .as_ref()
            .is_some_and(|bp| bp.should_apply_backpressure(current_buffered))
    }

    /// Returns `true` when the given event timestamp is late relative to the watermark.
    pub fn is_late_event(&self, event_time_ms: u64, watermark_ms: u64) -> bool {
        self.watermark
            .as_ref()
            .is_some_and(|wm| wm.is_late(event_time_ms, watermark_ms))
    }
}

impl Default for StreamingConfigV2 {
    fn default() -> Self {
        Self::new(StreamingConfig::default())
    }
}

/// Runtime statistics for a stream processing session.
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// Number of chunks that were successfully processed.
    pub chunks_processed: usize,
    /// Number of chunks that were dropped (e.g. due to backpressure or late arrival).
    pub chunks_dropped: usize,
    /// Number of times backpressure was triggered.
    pub backpressure_events: usize,
    /// Number of events dropped because they were late (beyond the watermark).
    pub late_events_dropped: usize,
    /// Total wall-clock processing time in milliseconds.
    pub total_processing_time_ms: u64,
    /// Total number of individual data elements processed across all chunks.
    pub total_elements_processed: usize,
}

impl StreamingStats {
    /// Average latency per processed chunk in milliseconds.
    /// Returns `0.0` when no chunks have been processed.
    pub fn average_latency_ms(&self) -> f64 {
        if self.chunks_processed == 0 {
            return 0.0;
        }
        self.total_processing_time_ms as f64 / self.chunks_processed as f64
    }

    /// Fraction of chunks that were dropped: `dropped / (processed + dropped)`.
    /// Returns `0.0` when no chunks have been seen at all.
    pub fn drop_rate(&self) -> f64 {
        let total = self.chunks_processed + self.chunks_dropped;
        if total == 0 {
            return 0.0;
        }
        self.chunks_dropped as f64 / total as f64
    }

    /// Throughput in chunks per second.
    /// Returns `0.0` when no processing time has been recorded.
    pub fn throughput_chunks_per_sec(&self) -> f64 {
        if self.total_processing_time_ms == 0 {
            return 0.0;
        }
        self.chunks_processed as f64 / (self.total_processing_time_ms as f64 / 1000.0)
    }

    /// Merge another `StreamingStats` into this one by summing all fields.
    pub fn merge(&mut self, other: &StreamingStats) {
        self.chunks_processed += other.chunks_processed;
        self.chunks_dropped += other.chunks_dropped;
        self.backpressure_events += other.backpressure_events;
        self.late_events_dropped += other.late_events_dropped;
        self.total_processing_time_ms += other.total_processing_time_ms;
        self.total_elements_processed += other.total_elements_processed;
    }
}

// ============================================================
// V2 Tests
// ============================================================

#[cfg(test)]
mod v2_tests {
    use super::*;

    // ---- BackpressureConfig tests ----

    #[test]
    fn test_backpressure_config_new() {
        let cfg = BackpressureConfig::new(100);
        assert_eq!(cfg.max_buffered_chunks, 100);
        assert!((cfg.high_watermark - 0.8).abs() < f64::EPSILON);
        assert!((cfg.low_watermark - 0.2).abs() < f64::EPSILON);
        assert_eq!(cfg.strategy, BackpressureStrategy::Block);
    }

    #[test]
    fn test_backpressure_above_high_watermark() {
        let cfg = BackpressureConfig::new(100); // threshold = floor(100 * 0.8) = 80
                                                // 81 > 80 → above
        assert!(cfg.is_above_high_watermark(81));
        // 80 == 80 → NOT above (strictly greater than)
        assert!(!cfg.is_above_high_watermark(80));
        // 0 → not above
        assert!(!cfg.is_above_high_watermark(0));
    }

    #[test]
    fn test_backpressure_below_low_watermark() {
        let cfg = BackpressureConfig::new(100); // threshold = floor(100 * 0.2) = 20
                                                // 19 < 20 → below
        assert!(cfg.is_below_low_watermark(19));
        // 20 == 20 → NOT below (strictly less than)
        assert!(!cfg.is_below_low_watermark(20));
        // 100 → not below
        assert!(!cfg.is_below_low_watermark(100));
    }

    #[test]
    fn test_backpressure_between_watermarks() {
        let cfg = BackpressureConfig::new(100);
        // 50 is between low (20) and high (80) → no backpressure
        assert!(!cfg.should_apply_backpressure(50));
        // 81 > 80 → backpressure active
        assert!(cfg.should_apply_backpressure(81));
    }

    #[test]
    fn test_backpressure_strategy_variants() {
        let block = BackpressureStrategy::Block;
        let drop_oldest = BackpressureStrategy::DropOldest;
        let drop_newest = BackpressureStrategy::DropNewest;
        let error = BackpressureStrategy::ErrorOnFull;

        // All four variants exist and are distinct from each other.
        assert_ne!(drop_oldest, block);
        assert_ne!(drop_newest, block);
        assert_ne!(error, block);
        assert_ne!(drop_oldest, drop_newest);

        let cfg = BackpressureConfig::new(10).with_strategy(BackpressureStrategy::DropOldest);
        assert_eq!(cfg.strategy, drop_oldest);
        let _ = error; // suppress unused warning
    }

    // ---- WatermarkConfig tests ----

    #[test]
    fn test_watermark_config_new() {
        let wm = WatermarkConfig::new(100);
        assert_eq!(wm.max_out_of_order_ms, 100);
        assert_eq!(wm.idle_timeout_ms, None);
        assert!(!wm.drop_late_events);
    }

    #[test]
    fn test_watermark_current_watermark_calculation() {
        let wm = WatermarkConfig::new(100);
        assert_eq!(wm.current_watermark(500), 400);

        // Saturating subtraction: 1000 > 500, so result is 0.
        let wm2 = WatermarkConfig::new(1000);
        assert_eq!(wm2.current_watermark(500), 0);
    }

    #[test]
    fn test_watermark_is_late_event() {
        let wm = WatermarkConfig::new(100);
        // event at 300 vs watermark 400 → late
        assert!(wm.is_late(300, 400));
        // event at 400 vs watermark 400 → NOT late (equal is on-time)
        assert!(!wm.is_late(400, 400));
        // event at 500 vs watermark 400 → on-time
        assert!(!wm.is_late(500, 400));
    }

    #[test]
    fn test_watermark_with_idle_timeout() {
        let wm = WatermarkConfig::new(100).with_idle_timeout(5000);
        assert_eq!(wm.idle_timeout_ms, Some(5000));
        assert_eq!(wm.max_out_of_order_ms, 100);
    }

    // ---- StreamingStats tests ----

    #[test]
    fn test_streaming_stats_default() {
        let stats = StreamingStats::default();
        assert_eq!(stats.chunks_processed, 0);
        assert_eq!(stats.chunks_dropped, 0);
        assert!((stats.average_latency_ms() - 0.0).abs() < f64::EPSILON);
        assert!((stats.drop_rate() - 0.0).abs() < f64::EPSILON);
        assert!((stats.throughput_chunks_per_sec() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_streaming_stats_drop_rate() {
        let stats = StreamingStats {
            chunks_processed: 9,
            chunks_dropped: 1,
            ..Default::default()
        };
        // 1 / (9+1) = 0.1
        assert!((stats.drop_rate() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_streaming_stats_merge() {
        let mut a = StreamingStats {
            chunks_processed: 10,
            chunks_dropped: 2,
            backpressure_events: 1,
            late_events_dropped: 3,
            total_processing_time_ms: 500,
            total_elements_processed: 100,
        };
        let b = StreamingStats {
            chunks_processed: 5,
            chunks_dropped: 1,
            backpressure_events: 2,
            late_events_dropped: 0,
            total_processing_time_ms: 250,
            total_elements_processed: 50,
        };
        a.merge(&b);
        assert_eq!(a.chunks_processed, 15);
        assert_eq!(a.chunks_dropped, 3);
        assert_eq!(a.backpressure_events, 3);
        assert_eq!(a.late_events_dropped, 3);
        assert_eq!(a.total_processing_time_ms, 750);
        assert_eq!(a.total_elements_processed, 150);
    }

    // ---- StreamingConfigV2 tests ----

    #[test]
    fn test_streaming_config_v2_new() {
        let cfg = StreamingConfigV2::new(StreamingConfig::default());
        assert!(cfg.backpressure.is_none());
        assert!(cfg.watermark.is_none());
    }

    #[test]
    fn test_streaming_config_v2_with_backpressure() {
        // Without backpressure: never applies.
        let cfg_none = StreamingConfigV2::new(StreamingConfig::default());
        assert!(!cfg_none.should_apply_backpressure(0));
        assert!(!cfg_none.should_apply_backpressure(usize::MAX));

        // With backpressure configured: threshold at 80.
        let bp = BackpressureConfig::new(100);
        let cfg = StreamingConfigV2::new(StreamingConfig::default()).with_backpressure(bp);
        assert!(!cfg.should_apply_backpressure(50));
        assert!(cfg.should_apply_backpressure(81));
    }

    #[test]
    fn test_streaming_config_v2_combined() {
        let bp = BackpressureConfig::new(50);
        let wm = WatermarkConfig::new(200);
        let cfg = StreamingConfigV2::new(StreamingConfig::default())
            .with_backpressure(bp)
            .with_watermark(wm);
        assert!(cfg.backpressure.is_some());
        assert!(cfg.watermark.is_some());

        // Backpressure: threshold = floor(50 * 0.8) = 40; 41 → active
        assert!(cfg.should_apply_backpressure(41));
        // Watermark: is_late(100, 300) → true (100 < 300)
        assert!(cfg.is_late_event(100, 300));
        // is_late(400, 300) → false
        assert!(!cfg.is_late_event(400, 300));
    }
}
