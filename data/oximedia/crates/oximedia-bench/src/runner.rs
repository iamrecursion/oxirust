//! Benchmark execution engine for running codec benchmarks.
//!
//! This module provides the core execution engine that runs codec benchmarks,
//! manages parallel execution, handles warmup iterations, and collects performance
//! metrics.

use crate::metrics::{MetricsCalculator, QualityMetrics};
use crate::sequences::TestSequence;
use crate::{BenchError, BenchResult, BenchmarkConfig, CodecConfig, SequenceResult};
use oximedia_codec::VideoFrame;
use oximedia_core::types::CodecId;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Benchmark execution result for a single run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Frames processed
    pub frames: usize,
    /// Time taken
    #[serde(with = "crate::duration_serde")]
    pub duration: Duration,
    /// FPS achieved
    pub fps: f64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: Option<u64>,
    /// CPU utilization percentage
    pub cpu_utilization: Option<f64>,
}

impl ExecutionResult {
    /// Create a new execution result.
    #[must_use]
    pub fn new(frames: usize, duration: Duration) -> Self {
        let fps = if duration.as_secs_f64() > 0.0 {
            frames as f64 / duration.as_secs_f64()
        } else {
            0.0
        };
        Self {
            frames,
            duration,
            fps,
            peak_memory_bytes: None,
            cpu_utilization: None,
        }
    }

    /// Create with memory tracking.
    #[must_use]
    pub fn with_memory(mut self, peak_memory_bytes: u64) -> Self {
        self.peak_memory_bytes = Some(peak_memory_bytes);
        self
    }

    /// Create with CPU tracking.
    #[must_use]
    pub fn with_cpu(mut self, cpu_utilization: f64) -> Self {
        self.cpu_utilization = Some(cpu_utilization);
        self
    }
}

/// Cache for benchmark results to enable incremental runs.
#[derive(Debug, Clone, Default)]
pub struct ResultCache {
    cache: Arc<Mutex<HashMap<String, SequenceResult>>>,
    cache_dir: Option<PathBuf>,
}

impl ResultCache {
    /// Create a new result cache.
    #[must_use]
    pub fn new(cache_dir: Option<PathBuf>) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            cache_dir,
        }
    }

    /// Get a cached result.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<SequenceResult> {
        let cache = self.cache.lock().ok()?;
        cache.get(key).cloned()
    }

    /// Store a result in the cache.
    pub fn set(&self, key: String, result: SequenceResult) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(key, result);
        }
    }

    /// Load cache from disk.
    ///
    /// # Errors
    ///
    /// Returns an error if loading fails.
    pub fn load_from_disk(&self) -> BenchResult<()> {
        if let Some(cache_dir) = &self.cache_dir {
            let cache_file = cache_dir.join("bench_cache.json");
            if cache_file.exists() {
                let data = std::fs::read_to_string(&cache_file)?;
                let cached: HashMap<String, SequenceResult> = serde_json::from_str(&data)?;
                if let Ok(mut cache) = self.cache.lock() {
                    *cache = cached;
                }
            }
        }
        Ok(())
    }

    /// Save cache to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if saving fails.
    pub fn save_to_disk(&self) -> BenchResult<()> {
        if let Some(cache_dir) = &self.cache_dir {
            std::fs::create_dir_all(cache_dir)?;
            let cache_file = cache_dir.join("bench_cache.json");
            if let Ok(cache) = self.cache.lock() {
                let data = serde_json::to_string_pretty(&*cache)?;
                std::fs::write(&cache_file, data)?;
            }
        }
        Ok(())
    }

    /// Clear the cache.
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }

    /// Generate cache key for a benchmark run.
    ///
    /// The key incorporates codec parameters and a content-hash of the sequence
    /// file (first 64 KiB via XXH3-64) so that modifying a test sequence
    /// automatically invalidates stale cache entries.  When the file cannot be
    /// read the sequence name itself is hashed as a stable fallback.
    #[must_use]
    pub fn generate_key(codec_config: &CodecConfig, sequence_name: &str) -> String {
        let config_str = format!(
            "{:?}_{}_{}_{}_{}",
            codec_config.codec_id,
            codec_config.preset.as_deref().unwrap_or("default"),
            codec_config.bitrate_kbps.unwrap_or(0),
            codec_config.cq_level.unwrap_or(0),
            sequence_name
        );

        // Try to read first 64 KiB of the sequence file for a content fingerprint.
        // Falls back to hashing the sequence name when the file is unavailable.
        let content_hash = {
            use std::io::Read;
            std::path::Path::new(sequence_name)
                .metadata()
                .ok()
                .and_then(|m| {
                    let cap = (m.len() as usize).min(65536);
                    let mut buf = vec![0u8; cap];
                    let mut f = std::fs::File::open(sequence_name).ok()?;
                    let n = f.read(&mut buf).ok()?;
                    buf.truncate(n);
                    Some(xxhash_rust::xxh3::xxh3_64(&buf))
                })
                .unwrap_or_else(|| xxhash_rust::xxh3::xxh3_64(sequence_name.as_bytes()))
        };

        format!("{}_{:016x}", config_str, content_hash)
    }
}

/// Progress tracking for benchmark execution.
#[derive(Debug, Clone)]
pub struct ProgressTracker {
    total_tasks: usize,
    completed_tasks: Arc<Mutex<usize>>,
    start_time: Instant,
}

impl ProgressTracker {
    /// Create a new progress tracker.
    #[must_use]
    pub fn new(total_tasks: usize) -> Self {
        Self {
            total_tasks,
            completed_tasks: Arc::new(Mutex::new(0)),
            start_time: Instant::now(),
        }
    }

    /// Mark a task as completed.
    pub fn complete_task(&self) {
        if let Ok(mut completed) = self.completed_tasks.lock() {
            *completed += 1;
        }
    }

    /// Get current progress (0.0 to 1.0).
    #[must_use]
    pub fn progress(&self) -> f64 {
        if self.total_tasks == 0 {
            return 1.0;
        }

        let completed = self.completed_tasks.lock().map(|c| *c).unwrap_or(0);

        completed as f64 / self.total_tasks as f64
    }

    /// Get elapsed time.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Estimate remaining time.
    #[must_use]
    pub fn estimate_remaining(&self) -> Option<Duration> {
        let progress = self.progress();
        if progress == 0.0 || progress >= 1.0 {
            return None;
        }

        let elapsed = self.elapsed().as_secs_f64();
        let total_estimate = elapsed / progress;
        let remaining = total_estimate - elapsed;

        Some(Duration::from_secs_f64(remaining))
    }
}

/// The benchmark runner handles execution of codec benchmarks.
#[allow(dead_code)]
pub struct BenchmarkRunner {
    parallel_jobs: usize,
    warmup_iterations: usize,
    measurement_iterations: usize,
    metrics_calculator: MetricsCalculator,
    result_cache: ResultCache,
    max_frames: Option<usize>,
}

#[allow(dead_code)]
impl BenchmarkRunner {
    /// Create a new benchmark runner.
    #[must_use]
    pub fn new(config: &BenchmarkConfig) -> Self {
        let metrics_calculator =
            MetricsCalculator::new(config.enable_psnr, config.enable_ssim, config.enable_vmaf);

        let result_cache = ResultCache::new(config.cache_dir.clone());

        Self {
            parallel_jobs: config.parallel_jobs,
            warmup_iterations: config.warmup_iterations,
            measurement_iterations: config.measurement_iterations,
            metrics_calculator,
            result_cache,
            max_frames: config.max_frames,
        }
    }

    /// Run benchmarks for all sequences with a specific codec.
    ///
    /// # Errors
    ///
    /// Returns an error if benchmark execution fails.
    pub fn run_codec_sequences(
        &self,
        _codec_config: &CodecConfig,
    ) -> BenchResult<Vec<SequenceResult>> {
        // Placeholder implementation
        // In reality, this would iterate over sequences and run benchmarks

        Ok(Vec::new())
    }

    /// Run benchmark for a single sequence.
    fn run_sequence_benchmark(
        &self,
        codec_config: &CodecConfig,
        sequence: &TestSequence,
    ) -> BenchResult<SequenceResult> {
        // Check cache first
        let cache_key = ResultCache::generate_key(codec_config, &sequence.name);
        if let Some(cached_result) = self.result_cache.get(&cache_key) {
            return Ok(cached_result);
        }

        // Run warmup iterations
        for _ in 0..self.warmup_iterations {
            let _ = self.run_iteration(codec_config, sequence)?;
        }

        // Run measurement iterations
        let mut results = Vec::new();
        for _ in 0..self.measurement_iterations {
            results.push(self.run_iteration(codec_config, sequence)?);
        }

        // Take median result
        results.sort_by(|a, b| {
            a.encoding_fps
                .partial_cmp(&b.encoding_fps)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let result = results[results.len() / 2].clone();

        // Cache the result
        self.result_cache.set(cache_key, result.clone());

        Ok(result)
    }

    /// Run a single benchmark iteration.
    fn run_iteration(
        &self,
        _codec_config: &CodecConfig,
        _sequence: &TestSequence,
    ) -> BenchResult<SequenceResult> {
        // Placeholder for running a single benchmark iteration

        Err(BenchError::ExecutionFailed(
            "Benchmark execution not fully implemented".to_string(),
        ))
    }

    /// Encode a sequence with the specified codec.
    fn encode_sequence(
        &self,
        _codec_id: CodecId,
        _frames: &[VideoFrame],
    ) -> BenchResult<(Vec<u8>, ExecutionResult)> {
        // Placeholder for encoding

        Err(BenchError::ExecutionFailed(
            "Encoding not fully implemented".to_string(),
        ))
    }

    /// Decode a sequence.
    fn decode_sequence(
        &self,
        _codec_id: CodecId,
        _data: &[u8],
    ) -> BenchResult<(Vec<VideoFrame>, ExecutionResult)> {
        // Placeholder for decoding

        Err(BenchError::ExecutionFailed(
            "Decoding not fully implemented".to_string(),
        ))
    }

    /// Calculate quality metrics between original and reconstructed frames.
    fn calculate_metrics(
        &self,
        original: &[VideoFrame],
        reconstructed: &[VideoFrame],
    ) -> BenchResult<QualityMetrics> {
        self.metrics_calculator
            .calculate_sequence(original, reconstructed)
    }

    /// Load result cache from disk.
    ///
    /// # Errors
    ///
    /// Returns an error if loading fails.
    pub fn load_cache(&self) -> BenchResult<()> {
        self.result_cache.load_from_disk()
    }

    /// Save result cache to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if saving fails.
    pub fn save_cache(&self) -> BenchResult<()> {
        self.result_cache.save_to_disk()
    }

    /// Clear the result cache.
    pub fn clear_cache(&self) {
        self.result_cache.clear();
    }
}

/// Parallel benchmark executor for running multiple benchmarks concurrently.
pub struct ParallelExecutor {
    max_parallel: usize,
}

impl ParallelExecutor {
    /// Create a new parallel executor.
    #[must_use]
    pub fn new(max_parallel: usize) -> Self {
        Self { max_parallel }
    }

    /// Execute benchmarks in parallel.
    ///
    /// # Errors
    ///
    /// Returns an error if any benchmark fails.
    pub fn execute<F>(&self, tasks: Vec<String>, task_fn: F) -> BenchResult<Vec<SequenceResult>>
    where
        F: Fn(&str) -> BenchResult<SequenceResult> + Send + Sync,
    {
        // Configure rayon thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.max_parallel)
            .build()
            .map_err(|e| BenchError::ExecutionFailed(format!("Thread pool error: {e}")))?;

        // Execute tasks in parallel
        pool.install(|| {
            tasks
                .par_iter()
                .map(|task| task_fn(task))
                .collect::<Result<Vec<_>, _>>()
        })
    }
}

/// Run multiple iterations and return the median result.
pub fn run_multiple_iterations<F>(
    iterations: usize,
    mut benchmark_fn: F,
) -> BenchResult<ExecutionResult>
where
    F: FnMut() -> BenchResult<ExecutionResult>,
{
    if iterations == 0 {
        return Err(BenchError::InvalidConfig(
            "Iterations must be > 0".to_string(),
        ));
    }

    let mut results = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        results.push(benchmark_fn()?);
    }

    // Return median result based on FPS
    results.sort_by(|a, b| {
        a.fps
            .partial_cmp(&b.fps)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(results[results.len() / 2].clone())
}

/// Memory profiler for tracking memory usage during benchmarks.
#[derive(Debug)]
pub struct MemoryProfiler {
    baseline: u64,
    peak: u64,
}

impl MemoryProfiler {
    /// Create a new memory profiler.
    #[must_use]
    pub fn new() -> Self {
        let baseline = Self::current_memory_usage();
        Self {
            baseline,
            peak: baseline,
        }
    }

    /// Update peak memory usage.
    pub fn update(&mut self) {
        let current = Self::current_memory_usage();
        if current > self.peak {
            self.peak = current;
        }
    }

    /// Get peak memory usage above baseline.
    #[must_use]
    pub fn peak_usage(&self) -> u64 {
        self.peak.saturating_sub(self.baseline)
    }

    /// Get current memory usage (placeholder - would use actual system calls).
    fn current_memory_usage() -> u64 {
        // Placeholder - in reality would use platform-specific APIs
        // like getrusage on Unix or GetProcessMemoryInfo on Windows
        0
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// CPU profiler for tracking CPU usage during benchmarks.
#[derive(Debug)]
pub struct CpuProfiler {
    start_time: Instant,
    total_cpu_time: Duration,
}

impl CpuProfiler {
    /// Create a new CPU profiler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            total_cpu_time: Duration::ZERO,
        }
    }

    /// Calculate CPU utilization percentage.
    #[must_use]
    pub fn utilization(&self) -> f64 {
        let wall_time = self.start_time.elapsed();
        if wall_time.as_secs_f64() == 0.0 {
            return 0.0;
        }

        (self.total_cpu_time.as_secs_f64() / wall_time.as_secs_f64()) * 100.0
    }
}

impl Default for CpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Benchmark configuration validator.
pub struct ConfigValidator;

impl ConfigValidator {
    /// Validate codec configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid.
    pub fn validate_codec(config: &CodecConfig) -> BenchResult<()> {
        // Check for conflicting bitrate and CQ settings
        if config.bitrate_kbps.is_some() && config.cq_level.is_some() {
            return Err(BenchError::InvalidConfig(
                "Cannot specify both bitrate and CQ level".to_string(),
            ));
        }

        // Validate bitrate range
        if let Some(bitrate) = config.bitrate_kbps {
            if bitrate == 0 {
                return Err(BenchError::InvalidConfig(
                    "Bitrate must be greater than 0".to_string(),
                ));
            }
            if bitrate > 100_000 {
                return Err(BenchError::InvalidConfig(
                    "Bitrate unreasonably high (> 100 Mbps)".to_string(),
                ));
            }
        }

        // Validate passes
        if config.passes == 0 {
            return Err(BenchError::InvalidConfig(
                "Number of passes must be greater than 0".to_string(),
            ));
        }
        if config.passes > 3 {
            return Err(BenchError::InvalidConfig(
                "Number of passes > 3 not supported".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if sequence is invalid.
    pub fn validate_sequence(sequence: &TestSequence) -> BenchResult<()> {
        sequence.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BenchmarkConfig;

    #[test]
    fn test_execution_result() {
        let result = ExecutionResult::new(100, Duration::from_secs(2));
        assert_eq!(result.frames, 100);
        assert_eq!(result.fps, 50.0);
    }

    #[test]
    fn test_execution_result_zero_duration() {
        let result = ExecutionResult::new(100, Duration::ZERO);
        assert_eq!(result.fps, 0.0);
    }

    #[test]
    fn test_execution_result_with_memory() {
        let result =
            ExecutionResult::new(100, Duration::from_secs(1)).with_memory(1024 * 1024 * 100);
        assert_eq!(result.peak_memory_bytes, Some(1024 * 1024 * 100));
    }

    #[test]
    fn test_execution_result_with_cpu() {
        let result = ExecutionResult::new(100, Duration::from_secs(1)).with_cpu(75.5);
        assert_eq!(result.cpu_utilization, Some(75.5));
    }

    #[test]
    fn test_benchmark_runner_creation() {
        let config = BenchmarkConfig::default();
        let runner = BenchmarkRunner::new(&config);
        assert_eq!(runner.warmup_iterations, 1);
    }

    #[test]
    fn test_run_multiple_iterations() {
        let mut counter = 0;
        let result = run_multiple_iterations(3, || {
            counter += 1;
            Ok(ExecutionResult::new(
                100,
                Duration::from_millis(counter * 10),
            ))
        });

        assert!(result.is_ok());
        let result = result.expect("result should be valid");
        assert_eq!(result.frames, 100);
    }

    #[test]
    fn test_run_multiple_iterations_zero() {
        let result =
            run_multiple_iterations(0, || Ok(ExecutionResult::new(100, Duration::from_secs(1))));

        assert!(result.is_err());
    }

    #[test]
    fn test_result_cache() {
        let cache = ResultCache::new(None);
        assert!(cache.get("test").is_none());

        // Would need a full SequenceResult to test set
    }

    #[test]
    fn test_progress_tracker() {
        let tracker = ProgressTracker::new(10);
        assert_eq!(tracker.progress(), 0.0);

        tracker.complete_task();
        assert_eq!(tracker.progress(), 0.1);

        for _ in 0..9 {
            tracker.complete_task();
        }
        assert_eq!(tracker.progress(), 1.0);
    }

    #[test]
    fn test_progress_tracker_zero_tasks() {
        let tracker = ProgressTracker::new(0);
        assert_eq!(tracker.progress(), 1.0);
    }

    #[test]
    fn test_memory_profiler() {
        let mut profiler = MemoryProfiler::new();
        profiler.update();
        let _ = profiler.peak_usage();
    }

    #[test]
    fn test_cpu_profiler() {
        let profiler = CpuProfiler::new();
        let _ = profiler.utilization();
    }

    #[test]
    fn test_cache_key_generation() {
        let config = CodecConfig::new(CodecId::Av1)
            .with_preset("medium")
            .with_bitrate(2000);
        let key = ResultCache::generate_key(&config, "test_sequence");
        assert!(key.contains("Av1"));
        assert!(key.contains("medium"));
        assert!(key.contains("2000"));
        assert!(key.contains("test_sequence"));
    }

    #[test]
    fn test_config_validator_bitrate_and_cq() {
        let config = CodecConfig::new(CodecId::Av1)
            .with_bitrate(2000)
            .with_cq_level(30);

        assert!(ConfigValidator::validate_codec(&config).is_err());
    }

    #[test]
    fn test_config_validator_zero_bitrate() {
        let config = CodecConfig::new(CodecId::Av1).with_bitrate(0);

        assert!(ConfigValidator::validate_codec(&config).is_err());
    }

    #[test]
    fn test_config_validator_high_bitrate() {
        let config = CodecConfig::new(CodecId::Av1).with_bitrate(150_000);

        assert!(ConfigValidator::validate_codec(&config).is_err());
    }

    #[test]
    fn test_config_validator_zero_passes() {
        let mut config = CodecConfig::new(CodecId::Av1);
        config.passes = 0;

        assert!(ConfigValidator::validate_codec(&config).is_err());
    }

    #[test]
    fn test_config_validator_too_many_passes() {
        let mut config = CodecConfig::new(CodecId::Av1);
        config.passes = 4;

        assert!(ConfigValidator::validate_codec(&config).is_err());
    }

    #[test]
    fn test_config_validator_valid_config() {
        let config = CodecConfig::new(CodecId::Av1)
            .with_bitrate(2000)
            .with_passes(2);

        assert!(ConfigValidator::validate_codec(&config).is_ok());
    }

    #[test]
    fn test_parallel_executor_creation() {
        let executor = ParallelExecutor::new(4);
        assert_eq!(executor.max_parallel, 4);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Reusable luma scratch buffer — capacity stability + correctness
    // ───────────────────────────────────────────────────────────────────────

    /// Build a single-luma-plane `VideoFrame` from raw byte data.
    fn luma_frame(width: u32, height: u32, data: Vec<u8>) -> oximedia_codec::VideoFrame {
        use oximedia_codec::{Plane, VideoFrame};
        use oximedia_core::{PixelFormat, Rational, Timestamp};
        let plane = Plane {
            stride: width as usize,
            width,
            height,
            data,
        };
        VideoFrame {
            format: PixelFormat::Yuv420p,
            width,
            height,
            planes: vec![plane],
            timestamp: Timestamp::new(0, Rational::new(1, 1000)),
            frame_type: oximedia_codec::FrameType::Key,
            color_info: oximedia_codec::ColorInfo::default(),
            corrupt: false,
        }
    }

    /// `extract_luma_into` reuses the same buffer across successive frames
    /// without reallocating: capacity stays constant once primed, and the
    /// extracted values are the byte-widened luma plane.
    #[test]
    fn test_extract_luma_into_capacity_stable() {
        let (w, h) = (8_u32, 4_u32);
        let n = (w * h) as usize;

        let mut buf: Vec<f64> = Vec::with_capacity(n);
        let primed_cap = buf.capacity();
        assert!(primed_cap >= n);

        // Three successive frames with distinct content of identical size.
        for round in 0u8..3 {
            let data: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_add(round * 7)).collect();
            let frame = luma_frame(w, h, data.clone());

            let dims = extract_luma_into(&frame, &mut buf).expect("luma extracted");
            assert_eq!(dims, (w as usize, h as usize));

            // Values match the byte-widened luma plane exactly.
            assert_eq!(buf.len(), n);
            for (got, &raw) in buf.iter().zip(data.iter()) {
                assert!((*got - f64::from(raw)).abs() < f64::EPSILON);
            }

            // Capacity must not have grown — buffer is reused, not realloc'd.
            assert_eq!(
                buf.capacity(),
                primed_cap,
                "capacity changed on round {round}: realloc occurred"
            );
        }
    }

    /// `LumaScratch` ping-pong buffers keep stable capacity across ≥3 frames
    /// and surface the previous frame's dimensions after `advance`.
    #[test]
    fn test_luma_scratch_no_realloc_across_frames() {
        let (w, h) = (16_u32, 16_u32);
        let n = (w * h) as usize;

        let mut scratch = LumaScratch::with_capacity(w as usize, h as usize);
        let curr_cap0 = scratch.curr.capacity();
        let prev_cap0 = scratch.prev.capacity();
        assert!(curr_cap0 >= n && prev_cap0 >= n);

        for round in 0u8..4 {
            let data: Vec<u8> = (0..n).map(|i| (i as u8) ^ (round * 13)).collect();
            let frame = luma_frame(w, h, data);

            let (fw, fh) = scratch.fill_current(&frame).expect("filled");
            assert_eq!((fw, fh), (w as usize, h as usize));
            assert_eq!(scratch.curr.len(), n);

            scratch.advance((fw, fh));
            assert_eq!(scratch.prev_dims, Some((w as usize, h as usize)));
            assert_eq!(scratch.prev.len(), n);

            // Neither buffer reallocated across any frame.
            assert_eq!(
                scratch.curr.capacity().max(scratch.prev.capacity()),
                curr_cap0.max(prev_cap0),
                "scratch reallocated on round {round}"
            );
        }
    }

    /// The in-place extractor and the allocating wrapper produce identical
    /// numerical output, so routing the hot path through the reused buffer
    /// keeps results bit-for-bit unchanged.
    #[test]
    fn test_extract_luma_into_matches_allocating() {
        let (w, h) = (5_u32, 3_u32);
        let n = (w * h) as usize;
        let data: Vec<u8> = (0..n).map(|i| (i * 9 + 1) as u8).collect();
        let frame = luma_frame(w, h, data);

        let (alloc_luma, aw, ah) = extract_luma(&frame).expect("alloc path");
        let mut buf = Vec::new();
        let (iw, ih) = extract_luma_into(&frame, &mut buf).expect("inplace path");

        assert_eq!((aw, ah), (iw, ih));
        assert_eq!(alloc_luma, buf);
    }

    /// `extract_luma_into` rejects frames with no usable luma plane.
    #[test]
    fn test_extract_luma_into_rejects_short_plane() {
        // Declared 8×8 but only 10 bytes of data → too short.
        let frame = luma_frame(8, 8, vec![0u8; 10]);
        let mut buf = Vec::new();
        assert!(extract_luma_into(&frame, &mut buf).is_none());

        // Zero-dimension frame → None.
        let zero = luma_frame(0, 0, Vec::new());
        let mut buf2 = Vec::new();
        assert!(extract_luma_into(&zero, &mut buf2).is_none());
    }
}

// ─── ITU-T P.910 spatial / temporal / motion complexity ──────────────────────

/// Per-sequence spatial, temporal, and motion complexity metrics (ITU-T P.910).
#[derive(Debug, Clone, PartialEq)]
pub struct ComputedComplexity {
    /// Spatial Information (SI) — standard deviation of Sobel magnitude per frame, averaged.
    pub si: f64,
    /// Temporal Information (TI) — standard deviation of inter-frame pixel difference, averaged.
    pub ti: f64,
    /// Mean block-SAD for 16×16 macroblocks compared to the previous frame.
    pub motion_sad_mean: f64,
}

/// Error produced by the complexity analysis functions.
#[derive(Debug, thiserror::Error)]
pub enum ComplexityError {
    /// No frames available to analyse.
    #[error("no frames provided for complexity analysis")]
    NoFrames,
    /// Frame dimensions are too small (need at least 3×3 for Sobel).
    #[error("frame is too small ({width}×{height}): need at least 3×3")]
    FrameTooSmall {
        /// Frame width in pixels.
        width: usize,
        /// Frame height in pixels.
        height: usize,
    },
    /// Frame luma plane length does not match width×height.
    #[error("luma plane length {got} does not match {width}×{height}={expected}")]
    LumaSizeMismatch {
        /// Actual byte count of the luma plane.
        got: usize,
        /// Declared frame width.
        width: usize,
        /// Declared frame height.
        height: usize,
        /// Expected byte count (`width × height`).
        expected: usize,
    },
}

// ── Internal luma extraction ───────────────────────────────────────────────────

/// Extract the luma (Y) plane as `f64` values in [0, 255] from a `VideoFrame`.
///
/// Supports planar YUV formats where the luma plane is always the first plane.
///
/// This allocates a fresh `Vec<f64>` on every call; the production hot path
/// uses [`extract_luma_into`] with a reusable scratch buffer instead, so this
/// allocating wrapper is retained only as a parity reference for tests.
#[cfg(test)]
fn extract_luma(frame: &oximedia_codec::VideoFrame) -> Option<(Vec<f64>, usize, usize)> {
    let mut buf = Vec::new();
    let (w, h) = extract_luma_into(frame, &mut buf)?;
    Some((buf, w, h))
}

/// Extract the luma (Y) plane into a caller-supplied reusable buffer.
///
/// The destination is `clear()`ed and refilled in place, so a `Vec<f64>` whose
/// capacity already covers `width × height` is reused without reallocation
/// across successive frames of the same resolution.  Returns the `(width,
/// height)` of the extracted plane, or `None` when the frame has no usable luma
/// plane (zero dimensions or a too-short first plane).
///
/// The numerical output is identical to [`extract_luma`] — each luma byte is
/// widened to `f64` in the same order.
fn extract_luma_into(
    frame: &oximedia_codec::VideoFrame,
    dst: &mut Vec<f64>,
) -> Option<(usize, usize)> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    if w == 0 || h == 0 {
        return None;
    }

    // Prefer the first plane (Y for planar YUV, or R for packed RGB).
    let plane = frame.planes.first()?;
    let raw = &plane.data;
    let len = w.checked_mul(h)?;
    if raw.len() < len {
        return None;
    }

    dst.clear();
    dst.reserve(len);
    dst.extend(raw[..len].iter().map(|&b| f64::from(b)));
    Some((w, h))
}

/// Double-buffered scratch space for ITU-T P.910 sequence analysis.
///
/// Holds two `Vec<f64>` luma buffers — one for the current frame and one for
/// the previous frame — that are reused across the whole sequence.  After the
/// first two extractions of a fixed-resolution sequence neither buffer
/// reallocates: each frame `clear()`s and refills the buffer that previously
/// held the frame-before-last, then swaps the roles.  This eliminates the
/// per-frame `Vec<f64>` allocation that a naive `collect()`-per-frame loop
/// incurs.
struct LumaScratch {
    /// Buffer receiving the most recently extracted (current) frame.
    curr: Vec<f64>,
    /// Buffer holding the previously extracted frame, or empty before the first.
    prev: Vec<f64>,
    /// Dimensions of the data currently in `prev` (`None` until the first fill).
    prev_dims: Option<(usize, usize)>,
}

impl LumaScratch {
    /// Create a scratch pair pre-sized for `width × height` luma planes.
    ///
    /// Sizing both buffers up front means a sequence of that resolution never
    /// reallocates during extraction.  A zero capacity is tolerated (the
    /// buffers simply grow on first use).
    fn with_capacity(width: usize, height: usize) -> Self {
        let cap = width.saturating_mul(height);
        Self {
            curr: Vec::with_capacity(cap),
            prev: Vec::with_capacity(cap),
            prev_dims: None,
        }
    }

    /// Extract `frame`'s luma into the `curr` buffer in place.
    ///
    /// Returns the `(width, height)` of the extracted plane, or `None` if the
    /// frame has no usable luma plane.
    fn fill_current(&mut self, frame: &oximedia_codec::VideoFrame) -> Option<(usize, usize)> {
        extract_luma_into(frame, &mut self.curr)
    }

    /// Promote the current buffer to `prev` for the next iteration.
    ///
    /// Swaps the `curr`/`prev` buffers (so the old `prev` storage is recycled
    /// as the next frame's `curr` target) and records the dimensions just
    /// extracted.  No allocation occurs.
    fn advance(&mut self, dims: (usize, usize)) {
        std::mem::swap(&mut self.curr, &mut self.prev);
        self.prev_dims = Some(dims);
    }
}

// ── SI: Spatial Information ────────────────────────────────────────────────────

/// Compute the 3×3 Sobel magnitude at every interior pixel and return the
/// standard deviation of those magnitudes.  Border pixels are skipped.
///
/// SI = σ( |∇|Sobel(frame)| )
pub fn compute_si(luma: &[f64], width: usize, height: usize) -> Result<f64, ComplexityError> {
    if width < 3 || height < 3 {
        return Err(ComplexityError::FrameTooSmall { width, height });
    }
    let expected = width * height;
    if luma.len() != expected {
        return Err(ComplexityError::LumaSizeMismatch {
            got: luma.len(),
            width,
            height,
            expected,
        });
    }

    let pixel = |row: usize, col: usize| luma[row * width + col];

    let mut magnitudes: Vec<f64> = Vec::with_capacity((width - 2) * (height - 2));

    for row in 1..(height - 1) {
        for col in 1..(width - 1) {
            // Sobel Gx: [-1 0 +1; -2 0 +2; -1 0 +1]
            let gx = -pixel(row - 1, col - 1)
                + pixel(row - 1, col + 1)
                + -2.0 * pixel(row, col - 1)
                + 2.0 * pixel(row, col + 1)
                + -pixel(row + 1, col - 1)
                + pixel(row + 1, col + 1);

            // Sobel Gy: [-1 -2 -1;  0  0  0; +1 +2 +1]
            let gy = -pixel(row - 1, col - 1) - 2.0 * pixel(row - 1, col) - pixel(row - 1, col + 1)
                + pixel(row + 1, col - 1)
                + 2.0 * pixel(row + 1, col)
                + pixel(row + 1, col + 1);

            magnitudes.push((gx * gx + gy * gy).sqrt());
        }
    }

    Ok(stddev(&magnitudes))
}

// ── TI: Temporal Information ───────────────────────────────────────────────────

/// Compute the standard deviation of the per-pixel absolute difference between
/// two consecutive frames' luma planes.
///
/// `TI = σ( |frame[t] − frame[t-1]| )`
pub fn compute_ti(
    curr: &[f64],
    prev: &[f64],
    width: usize,
    height: usize,
) -> Result<f64, ComplexityError> {
    let expected = width * height;
    if curr.len() != expected {
        return Err(ComplexityError::LumaSizeMismatch {
            got: curr.len(),
            width,
            height,
            expected,
        });
    }
    if prev.len() != expected {
        return Err(ComplexityError::LumaSizeMismatch {
            got: prev.len(),
            width,
            height,
            expected,
        });
    }

    let diff: Vec<f64> = curr
        .iter()
        .zip(prev.iter())
        .map(|(&c, &p)| (c - p).abs())
        .collect();
    Ok(stddev(&diff))
}

// ── Motion: 16×16 block SAD ────────────────────────────────────────────────────

/// Compute the mean Sum-of-Absolute-Differences (SAD) over all aligned 16×16
/// macroblocks between two consecutive luma frames.
///
/// Blocks that extend outside the frame boundary are skipped.
pub fn compute_motion_sad(
    curr: &[f64],
    prev: &[f64],
    width: usize,
    height: usize,
) -> Result<f64, ComplexityError> {
    let expected = width * height;
    if curr.len() != expected {
        return Err(ComplexityError::LumaSizeMismatch {
            got: curr.len(),
            width,
            height,
            expected,
        });
    }
    if prev.len() != expected {
        return Err(ComplexityError::LumaSizeMismatch {
            got: prev.len(),
            width,
            height,
            expected,
        });
    }

    const BLOCK: usize = 16;
    let cols = width / BLOCK;
    let rows = height / BLOCK;

    if cols == 0 || rows == 0 {
        // Frame smaller than one macroblock — return 0.
        return Ok(0.0);
    }

    let mut total_sad = 0.0_f64;
    let mut block_count = 0_usize;

    for br in 0..rows {
        for bc in 0..cols {
            let mut block_sad = 0.0_f64;
            for dy in 0..BLOCK {
                let row = br * BLOCK + dy;
                let offset = row * width + bc * BLOCK;
                for dx in 0..BLOCK {
                    block_sad += (curr[offset + dx] - prev[offset + dx]).abs();
                }
            }
            total_sad += block_sad;
            block_count += 1;
        }
    }

    if block_count == 0 {
        Ok(0.0)
    } else {
        Ok(total_sad / block_count as f64)
    }
}

// ── Sequence analysis entry-point ─────────────────────────────────────────────

/// Analyse a slice of `VideoFrame`s and return aggregate ITU-T P.910 complexity
/// metrics (SI, TI, motion SAD mean) averaged across all frames.
///
/// Frame `t = 0` contributes to SI only (no previous frame for TI/motion).
pub fn analyze_sequence(
    frames: &[oximedia_codec::VideoFrame],
) -> Result<ComputedComplexity, ComplexityError> {
    if frames.is_empty() {
        return Err(ComplexityError::NoFrames);
    }

    let mut si_values: Vec<f64> = Vec::with_capacity(frames.len());
    let mut ti_values: Vec<f64> = Vec::with_capacity(frames.len().saturating_sub(1));
    let mut sad_values: Vec<f64> = Vec::with_capacity(frames.len().saturating_sub(1));

    // Pre-size the reusable luma buffers from the first frame's declared
    // resolution so the per-frame extraction in the loop never reallocates.
    let (cap_w, cap_h) = frames
        .first()
        .map(|f| (f.width as usize, f.height as usize))
        .unwrap_or((0, 0));
    let mut scratch = LumaScratch::with_capacity(cap_w, cap_h);
    let mut have_prev = false;

    for frame in frames {
        // Refill the current buffer in place — no per-frame allocation.
        let (w, h) = match scratch.fill_current(frame) {
            Some(v) => v,
            None => continue,
        };

        // SI for every frame — silently skip frames that are too small for Sobel.
        if let Ok(si) = compute_si(&scratch.curr, w, h) {
            si_values.push(si);
        }

        // TI and motion require a previous frame of matching dimensions.
        if have_prev {
            if let Some((pw, ph)) = scratch.prev_dims {
                if pw == w && ph == h {
                    if let Ok(ti) = compute_ti(&scratch.curr, &scratch.prev, w, h) {
                        ti_values.push(ti);
                    }
                    if let Ok(sad) = compute_motion_sad(&scratch.curr, &scratch.prev, w, h) {
                        sad_values.push(sad);
                    }
                }
            }
        }

        scratch.advance((w, h));
        have_prev = true;
    }

    let si = if si_values.is_empty() {
        0.0
    } else {
        mean(&si_values)
    };
    let ti = if ti_values.is_empty() {
        0.0
    } else {
        mean(&ti_values)
    };
    let motion_sad_mean = if sad_values.is_empty() {
        0.0
    } else {
        mean(&sad_values)
    };

    Ok(ComputedComplexity {
        si,
        ti,
        motion_sad_mean,
    })
}

// ── Statistics helpers (private to this module) ───────────────────────────────

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn stddev(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / n as f64;
    variance.sqrt()
}
