//! Comprehensive codec benchmarking suite for `OxiMedia`.
//!
//! This crate provides a complete benchmarking framework for evaluating codec performance,
//! quality metrics, and efficiency across different encoding parameters and content types.
//!
//! # Features
//!
//! - **Multi-codec support**: Benchmark AV1, VP9, VP8, and Theora
//! - **Quality metrics**: PSNR, SSIM, and optional VMAF
//! - **Performance metrics**: Encoding/decoding speed, memory usage
//! - **Statistical analysis**: Mean, median, percentiles, standard deviation
//! - **Parallel execution**: Multi-threaded benchmark execution
//! - **Report generation**: JSON, CSV, and HTML output formats
//! - **Incremental benchmarking**: Result caching and differential runs
//!
//! # Example
//!
//! ```
//! use oximedia_bench::{BenchmarkConfig, BenchmarkSuite, CodecConfig};
//! use oximedia_core::types::CodecId;
//!
//! # fn example() -> oximedia_bench::BenchResult<()> {
//! // Create a benchmark configuration
//! let config = BenchmarkConfig::builder()
//!     .add_codec(CodecConfig::new(CodecId::Av1))
//!     .add_codec(CodecConfig::new(CodecId::Vp9))
//!     .parallel_jobs(4)
//!     .build()?;
//!
//! // Create and run the benchmark suite
//! let suite = BenchmarkSuite::new(config);
//! let results = suite.run_all()?;
//!
//! // Generate reports
//! results.export_json("results.json")?;
//! results.export_csv("results.csv")?;
//! # Ok(())
//! # }
//! ```
//!
//! # Architecture
//!
//! The benchmarking suite consists of several key components:
//!
//! - **Sequences**: Test video sequences with various characteristics
//! - **Metrics**: Quality and performance measurement tools
//! - **Runner**: Execution engine for running benchmarks
//! - **Comparison**: Tools for comparing codec performance
//! - **Reports**: Export and visualization of results
//! - **Statistics**: Statistical analysis of benchmark data

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

pub mod ab_comparison;
pub mod audio_bench;
pub mod baseline;
pub mod bd_rate;
pub mod bench_suite;
pub mod codec_bench;
pub mod comparison;
pub mod container_bench;
pub mod cpu_profile;
pub mod examples;
pub mod flamegraph_integration;
pub mod gpu_bench;
pub mod hardware_info;
pub mod historical_db;
pub mod io_bench;
pub mod itu_p910;
pub mod latency;
pub mod memory;
pub mod metrics;
pub mod percentile_tracker;
pub mod perf_comparison;
pub mod pipeline_bench;
pub mod rate_distortion;
pub mod regression;
pub mod regression_bench;
pub mod regression_detect;
pub mod report;
pub mod resource_monitor;
pub mod runner;
pub mod scalability_bench;
pub mod sequences;
pub mod statistical;
pub mod stats;
pub mod streaming_export;
pub mod system_fingerprint;
pub mod throughput;
pub mod warmup_strategy;
pub mod y4m_rw;

use oximedia_core::types::CodecId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use thiserror::Error;

pub use comparison::{CodecComparison, ComparisonResult};
pub use metrics::{MetricsCalculator, QualityMetrics};
pub use report::{HtmlReport, ReportExporter};
pub use runner::{BenchmarkRunner, ExecutionResult};
pub use sequences::{ContentType, MotionCharacteristics, TestSequence};
pub use stats::{StatisticalAnalysis, Statistics};

/// Result type for benchmarking operations.
pub type BenchResult<T> = Result<T, BenchError>;

/// Errors that can occur during benchmarking.
#[derive(Debug, Error)]
pub enum BenchError {
    /// I/O error occurred
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// OxiMedia core error
    #[error("OxiMedia error: {0}")]
    Oxi(#[from] oximedia_core::error::OxiError),

    /// Codec error
    #[error("Codec error: {0}")]
    Codec(#[from] oximedia_codec::error::CodecError),

    /// Graph error
    #[error("Graph error: {0}")]
    Graph(#[from] oximedia_graph::error::GraphError),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// CSV error
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Sequence not found
    #[error("Test sequence not found: {0}")]
    SequenceNotFound(String),

    /// Codec not supported
    #[error("Codec not supported: {0:?}")]
    UnsupportedCodec(CodecId),

    /// Benchmark execution failed
    #[error("Benchmark execution failed: {0}")]
    ExecutionFailed(String),

    /// Metric calculation failed
    #[error("Metric calculation failed: {0}")]
    MetricFailed(String),

    /// Invalid benchmark results
    #[error("Invalid benchmark results: {0}")]
    InvalidResults(String),
}

/// Configuration for a codec benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecConfig {
    /// Codec identifier
    pub codec_id: CodecId,

    /// Encoding preset (if applicable)
    pub preset: Option<String>,

    /// Target bitrate in kbps (if applicable)
    pub bitrate_kbps: Option<u32>,

    /// Constant quality mode value (if applicable)
    pub cq_level: Option<u32>,

    /// Number of encoding passes
    pub passes: u32,

    /// Enable rate control
    pub rate_control: bool,

    /// Additional codec-specific parameters
    pub extra_params: HashMap<String, String>,
}

impl CodecConfig {
    /// Create a new codec configuration with default settings.
    #[must_use]
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            preset: None,
            bitrate_kbps: None,
            cq_level: None,
            passes: 1,
            rate_control: false,
            extra_params: HashMap::new(),
        }
    }

    /// Set the encoding preset.
    #[must_use]
    pub fn with_preset(mut self, preset: impl Into<String>) -> Self {
        self.preset = Some(preset.into());
        self
    }

    /// Set the target bitrate.
    #[must_use]
    pub fn with_bitrate(mut self, bitrate_kbps: u32) -> Self {
        self.bitrate_kbps = Some(bitrate_kbps);
        self
    }

    /// Set the constant quality level.
    #[must_use]
    pub fn with_cq_level(mut self, cq_level: u32) -> Self {
        self.cq_level = Some(cq_level);
        self
    }

    /// Set the number of encoding passes.
    #[must_use]
    pub fn with_passes(mut self, passes: u32) -> Self {
        self.passes = passes;
        self
    }

    /// Enable or disable rate control.
    #[must_use]
    pub fn with_rate_control(mut self, enabled: bool) -> Self {
        self.rate_control = enabled;
        self
    }

    /// Add a codec-specific parameter.
    #[must_use]
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_params.insert(key.into(), value.into());
        self
    }
}

/// Configuration for the benchmark suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Codec configurations to benchmark
    pub codecs: Vec<CodecConfig>,

    /// Test sequences to use
    pub sequences: Vec<PathBuf>,

    /// Number of parallel jobs
    pub parallel_jobs: usize,

    /// Enable quality metric calculation
    pub enable_psnr: bool,

    /// Enable SSIM calculation
    pub enable_ssim: bool,

    /// Enable VMAF calculation
    pub enable_vmaf: bool,

    /// Cache directory for intermediate results
    pub cache_dir: Option<PathBuf>,

    /// Output directory for results
    pub output_dir: PathBuf,

    /// Maximum number of frames to process per sequence
    pub max_frames: Option<usize>,

    /// Warmup iterations before measurement
    pub warmup_iterations: usize,

    /// Number of measurement iterations
    pub measurement_iterations: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            codecs: Vec::new(),
            sequences: Vec::new(),
            parallel_jobs: num_cpus(),
            enable_psnr: true,
            enable_ssim: true,
            enable_vmaf: false,
            cache_dir: None,
            output_dir: PathBuf::from("./bench_results"),
            max_frames: None,
            warmup_iterations: 1,
            measurement_iterations: 3,
        }
    }
}

impl BenchmarkConfig {
    /// Create a new builder for benchmark configuration.
    #[must_use]
    pub fn builder() -> BenchmarkConfigBuilder {
        BenchmarkConfigBuilder::default()
    }
}

/// Builder for creating benchmark configurations.
#[derive(Debug, Default)]
pub struct BenchmarkConfigBuilder {
    config: BenchmarkConfig,
}

impl BenchmarkConfigBuilder {
    /// Add a codec configuration.
    #[must_use]
    pub fn add_codec(mut self, codec: CodecConfig) -> Self {
        self.config.codecs.push(codec);
        self
    }

    /// Add a test sequence.
    #[must_use]
    pub fn add_sequence(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.sequences.push(path.into());
        self
    }

    /// Set the number of parallel jobs.
    #[must_use]
    pub fn parallel_jobs(mut self, jobs: usize) -> Self {
        self.config.parallel_jobs = jobs;
        self
    }

    /// Enable PSNR calculation.
    #[must_use]
    pub fn enable_psnr(mut self, enable: bool) -> Self {
        self.config.enable_psnr = enable;
        self
    }

    /// Enable SSIM calculation.
    #[must_use]
    pub fn enable_ssim(mut self, enable: bool) -> Self {
        self.config.enable_ssim = enable;
        self
    }

    /// Enable VMAF calculation.
    #[must_use]
    pub fn enable_vmaf(mut self, enable: bool) -> Self {
        self.config.enable_vmaf = enable;
        self
    }

    /// Set the cache directory.
    #[must_use]
    pub fn cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.cache_dir = Some(dir.into());
        self
    }

    /// Set the output directory.
    #[must_use]
    pub fn output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.output_dir = dir.into();
        self
    }

    /// Set the maximum number of frames to process.
    #[must_use]
    pub fn max_frames(mut self, max: usize) -> Self {
        self.config.max_frames = Some(max);
        self
    }

    /// Set the number of warmup iterations.
    #[must_use]
    pub fn warmup_iterations(mut self, iterations: usize) -> Self {
        self.config.warmup_iterations = iterations;
        self
    }

    /// Set the number of measurement iterations.
    #[must_use]
    pub fn measurement_iterations(mut self, iterations: usize) -> Self {
        self.config.measurement_iterations = iterations;
        self
    }

    /// Build the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn build(self) -> BenchResult<BenchmarkConfig> {
        if self.config.codecs.is_empty() {
            return Err(BenchError::InvalidConfig("No codecs specified".to_string()));
        }

        if self.config.parallel_jobs == 0 {
            return Err(BenchError::InvalidConfig(
                "Parallel jobs must be greater than 0".to_string(),
            ));
        }

        if self.config.measurement_iterations == 0 {
            return Err(BenchError::InvalidConfig(
                "Measurement iterations must be greater than 0".to_string(),
            ));
        }

        Ok(self.config)
    }
}

/// Complete benchmark results for all codecs and sequences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Individual codec results
    pub codec_results: Vec<CodecBenchmarkResult>,

    /// Timestamp when benchmark was run
    pub timestamp: String,

    /// Total execution time
    #[serde(with = "duration_serde")]
    pub total_duration: Duration,

    /// Configuration used
    pub config: BenchmarkConfig,
}

impl BenchmarkResults {
    /// Export results to JSON format.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn export_json(&self, path: impl AsRef<Path>) -> BenchResult<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }

    /// Export results to CSV format.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn export_csv(&self, path: impl AsRef<Path>) -> BenchResult<()> {
        let mut writer = csv::Writer::from_path(path)?;

        // Write header
        writer.write_record([
            "Codec",
            "Sequence",
            "Preset",
            "Bitrate (kbps)",
            "Encoding FPS",
            "Decoding FPS",
            "File Size (bytes)",
            "PSNR (dB)",
            "SSIM",
            "VMAF",
        ])?;

        // Write data rows
        for codec_result in &self.codec_results {
            for seq_result in &codec_result.sequence_results {
                writer.write_record(&[
                    format!("{:?}", codec_result.codec_id),
                    seq_result.sequence_name.clone(),
                    codec_result.preset.clone().unwrap_or_default(),
                    codec_result
                        .bitrate_kbps
                        .map_or(String::new(), |b| b.to_string()),
                    format!("{:.2}", seq_result.encoding_fps),
                    format!("{:.2}", seq_result.decoding_fps),
                    seq_result.file_size_bytes.to_string(),
                    seq_result
                        .metrics
                        .psnr
                        .map_or(String::new(), |p| format!("{p:.2}")),
                    seq_result
                        .metrics
                        .ssim
                        .map_or(String::new(), |s| format!("{s:.4}")),
                    seq_result
                        .metrics
                        .vmaf
                        .map_or(String::new(), |v| format!("{v:.2}")),
                ])?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Stream results as CSV into an arbitrary [`std::io::Write`] sink.
    ///
    /// Unlike [`export_csv`](Self::export_csv) (which writes to a file path via
    /// the `csv` crate), this delegates to
    /// [`StreamingCsvWriter`](streaming_export::StreamingCsvWriter) and never
    /// holds the full row set in memory — rows are flattened lazily from
    /// `self` and written one at a time.  The header is written automatically.
    ///
    /// # Errors
    ///
    /// Returns an error if any write to `writer` fails.
    pub fn export_csv_streaming<W: std::io::Write>(&self, writer: &mut W) -> BenchResult<()> {
        let mut csv = streaming_export::StreamingCsvWriter::new(writer);
        csv.write_header()?;
        csv.write_all(self)?;
        csv.flush()
    }

    /// Stream results as JSON Lines (NDJSON) into an arbitrary
    /// [`std::io::Write`] sink.
    ///
    /// Each `(codec, sequence)` pair is emitted as a self-contained JSON object
    /// on its own line via
    /// [`StreamingJsonWriter`](streaming_export::StreamingJsonWriter), so the
    /// output streams without buffering every row.
    ///
    /// # Errors
    ///
    /// Returns an error if serialisation or any write to `writer` fails.
    pub fn export_json_streaming<W: std::io::Write>(&self, writer: &mut W) -> BenchResult<()> {
        let mut json = streaming_export::StreamingJsonWriter::new(writer);
        json.write_all(self)?;
        json.flush()
    }

    /// Generate an HTML report.
    ///
    /// # Errors
    ///
    /// Returns an error if the report cannot be generated.
    pub fn export_html(&self, path: impl AsRef<Path>) -> BenchResult<()> {
        let report = HtmlReport::new(self);
        report.write_to_file(path)
    }

    /// Get all results for a specific codec.
    #[must_use]
    pub fn get_codec_results(&self, codec_id: CodecId) -> Vec<&CodecBenchmarkResult> {
        self.codec_results
            .iter()
            .filter(|r| r.codec_id == codec_id)
            .collect()
    }

    /// Compare two codecs.
    #[must_use]
    pub fn compare_codecs(&self, codec_a: CodecId, codec_b: CodecId) -> Option<ComparisonResult> {
        let results_a: Vec<_> = self.get_codec_results(codec_a);
        let results_b: Vec<_> = self.get_codec_results(codec_b);

        if results_a.is_empty() || results_b.is_empty() {
            return None;
        }

        Some(CodecComparison::compare(results_a, results_b))
    }
}

/// Benchmark results for a single codec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecBenchmarkResult {
    /// Codec identifier
    pub codec_id: CodecId,

    /// Preset used
    pub preset: Option<String>,

    /// Target bitrate
    pub bitrate_kbps: Option<u32>,

    /// Constant quality level
    pub cq_level: Option<u32>,

    /// Results for each sequence
    pub sequence_results: Vec<SequenceResult>,

    /// Aggregated statistics
    pub statistics: Statistics,
}

/// Results for a single test sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceResult {
    /// Sequence name/identifier
    pub sequence_name: String,

    /// Number of frames processed
    pub frames_processed: usize,

    /// Encoding frames per second
    pub encoding_fps: f64,

    /// Decoding frames per second
    pub decoding_fps: f64,

    /// Encoded file size in bytes
    pub file_size_bytes: u64,

    /// Quality metrics
    pub metrics: QualityMetrics,

    /// Encoding duration
    #[serde(with = "duration_serde")]
    pub encoding_duration: Duration,

    /// Decoding duration
    #[serde(with = "duration_serde")]
    pub decoding_duration: Duration,
}

/// The main benchmark suite.
pub struct BenchmarkSuite {
    config: BenchmarkConfig,
    runner: BenchmarkRunner,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite with the given configuration.
    #[must_use]
    pub fn new(config: BenchmarkConfig) -> Self {
        let runner = BenchmarkRunner::new(&config);
        Self { config, runner }
    }

    /// Run all benchmarks in parallel using rayon.
    ///
    /// Each codec configuration is run on its own rayon worker so that
    /// multi-codec benchmark suites scale with available CPU cores.  The
    /// output order matches the input `config.codecs` order (rayon preserves
    /// `par_iter` order through `collect`).
    ///
    /// # Errors
    ///
    /// Returns an error if any benchmark fails.
    pub fn run_all(&self) -> BenchResult<BenchmarkResults> {
        use rayon::prelude::*;

        let start_time = std::time::Instant::now();

        // Parallel fan-out: each codec_config is independent of the others.
        // BenchmarkRunner is Sync because ResultCache uses Arc<Mutex<…>>.
        let codec_results: Vec<CodecBenchmarkResult> = self
            .config
            .codecs
            .par_iter()
            .map(|codec_config| self.run_codec_benchmark(codec_config))
            .collect::<Result<Vec<_>, _>>()?;

        let total_duration = start_time.elapsed();

        Ok(BenchmarkResults {
            codec_results,
            timestamp: format_timestamp(),
            total_duration,
            config: self.config.clone(),
        })
    }

    /// Run benchmark for a single codec.
    fn run_codec_benchmark(&self, codec_config: &CodecConfig) -> BenchResult<CodecBenchmarkResult> {
        let sequence_results = self.runner.run_codec_sequences(codec_config)?;
        let statistics = stats::compute_statistics(&sequence_results);

        Ok(CodecBenchmarkResult {
            codec_id: codec_config.codec_id,
            preset: codec_config.preset.clone(),
            bitrate_kbps: codec_config.bitrate_kbps,
            cq_level: codec_config.cq_level,
            sequence_results,
            statistics,
        })
    }
}

/// Get the number of CPU cores available.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
}

/// Format a timestamp in RFC3339 format using correct Gregorian calendar math.
fn format_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let secs = duration.as_secs();
    let nanos = duration.subsec_nanos();

    let hours = (secs % 86400) / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;

    // Proper Gregorian calendar calculation from Unix epoch day count.
    // Algorithm: civil_from_days (Howard Hinnant's algorithm, public domain).
    // https://howardhinnant.github.io/date_algorithms.html#civil_from_days
    let z = (secs / 86400) as i64 + 719_468_i64;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // year of era [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year [0, 365]
    let mp = (5 * doy + 2) / 153; // month of year [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // day [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // month [1, 12]
    let yr = if m <= 2 { y + 1 } else { y };

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:09}Z",
        yr, m, d, hours, minutes, seconds, nanos
    )
}

/// Serde serialization/deserialization for Duration.
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs_f64().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(deserializer)?;
        Ok(Duration::from_secs_f64(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_config_builder() {
        let config = CodecConfig::new(CodecId::Av1)
            .with_preset("medium")
            .with_bitrate(2000)
            .with_passes(2);

        assert_eq!(config.codec_id, CodecId::Av1);
        assert_eq!(config.preset, Some("medium".to_string()));
        assert_eq!(config.bitrate_kbps, Some(2000));
        assert_eq!(config.passes, 2);
    }

    #[test]
    fn test_benchmark_config_builder() {
        let config = BenchmarkConfig::builder()
            .add_codec(CodecConfig::new(CodecId::Av1))
            .add_codec(CodecConfig::new(CodecId::Vp9))
            .parallel_jobs(4)
            .enable_psnr(true)
            .enable_ssim(true)
            .build();

        assert!(config.is_ok());
        let config = config.expect("config should be valid");
        assert_eq!(config.codecs.len(), 2);
        assert_eq!(config.parallel_jobs, 4);
        assert!(config.enable_psnr);
        assert!(config.enable_ssim);
    }

    #[test]
    fn test_invalid_config_no_codecs() {
        let config = BenchmarkConfig::builder().build();
        assert!(config.is_err());
        assert!(matches!(config.unwrap_err(), BenchError::InvalidConfig(_)));
    }

    #[test]
    fn test_invalid_config_zero_jobs() {
        let config = BenchmarkConfig::builder()
            .add_codec(CodecConfig::new(CodecId::Av1))
            .parallel_jobs(0)
            .build();

        assert!(config.is_err());
    }

    #[test]
    fn test_codec_config_with_params() {
        let config = CodecConfig::new(CodecId::Vp9)
            .with_param("cpu-used", "4")
            .with_param("threads", "8");

        assert_eq!(config.extra_params.get("cpu-used"), Some(&"4".to_string()));
        assert_eq!(config.extra_params.get("threads"), Some(&"8".to_string()));
    }

    #[test]
    fn test_num_cpus() {
        let cpus = num_cpus();
        assert!(cpus > 0);
    }

    #[test]
    fn test_format_timestamp() {
        let ts = format_timestamp();
        assert!(!ts.is_empty());
        assert!(ts.contains('T'));
        assert!(ts.contains('Z'));
    }

    // -----------------------------------------------------------------------
    // Wave 22 Slice 3 — parallel run_all + content-hash cache key
    // -----------------------------------------------------------------------

    /// `run_all` with 3 codec configs returns results for every codec (parallel
    /// path produces the same codec_id set as the sequential baseline).
    #[test]
    fn test_parallel_run_all_result_matches_sequential() {
        // run_codec_sequences returns Ok(vec![]) for any config (stub impl),
        // so both paths should produce identical CodecBenchmarkResult vectors.
        let config = BenchmarkConfig::builder()
            .add_codec(CodecConfig::new(CodecId::Av1))
            .add_codec(CodecConfig::new(CodecId::Vp9))
            .add_codec(CodecConfig::new(CodecId::Vp8))
            .parallel_jobs(4)
            .build()
            .expect("valid config");

        let suite = BenchmarkSuite::new(config);
        let results = suite.run_all().expect("run_all should succeed");

        assert_eq!(results.codec_results.len(), 3);

        let ids: Vec<CodecId> = results.codec_results.iter().map(|r| r.codec_id).collect();
        assert!(ids.contains(&CodecId::Av1));
        assert!(ids.contains(&CodecId::Vp9));
        assert!(ids.contains(&CodecId::Vp8));
    }

    /// Two different byte sequences must produce different XXH3 cache keys.
    #[test]
    fn test_cache_key_different_content_different_hash() {
        use crate::runner::ResultCache;

        let cfg = CodecConfig::new(CodecId::Av1)
            .with_preset("fast")
            .with_bitrate(2000);

        // Use two distinct but non-existent paths — generate_key falls back to
        // hashing the sequence_name string, so two distinct names → two keys.
        let key_a = ResultCache::generate_key(&cfg, "sequence_alpha.y4m");
        let key_b = ResultCache::generate_key(&cfg, "sequence_beta.y4m");
        assert_ne!(
            key_a, key_b,
            "distinct sequence names must yield distinct keys"
        );
    }

    /// The same inputs always produce the same cache key (determinism).
    #[test]
    fn test_cache_key_same_content_same_hash() {
        use crate::runner::ResultCache;

        let cfg = CodecConfig::new(CodecId::Vp9).with_cq_level(32);

        let key_a = ResultCache::generate_key(&cfg, "test_seq.y4m");
        let key_b = ResultCache::generate_key(&cfg, "test_seq.y4m");
        assert_eq!(
            key_a, key_b,
            "identical inputs must always produce the same key"
        );
    }

    /// A cached `SequenceResult` is returned directly without re-running the
    /// benchmark: inserting a sentinel result and calling `get` with the same
    /// key must return the sentinel unchanged.
    #[test]
    fn test_cache_hit_skips_recompute() {
        use crate::metrics::QualityMetrics;
        use crate::runner::ResultCache;

        let cache = ResultCache::new(None);
        let cfg = CodecConfig::new(CodecId::Av1);
        let key = ResultCache::generate_key(&cfg, "cached_seq.y4m");

        let sentinel = SequenceResult {
            sequence_name: "cached_seq.y4m".to_string(),
            frames_processed: 42,
            encoding_fps: 999.0,
            decoding_fps: 888.0,
            file_size_bytes: 12345,
            metrics: QualityMetrics::default(),
            encoding_duration: Duration::from_millis(10),
            decoding_duration: Duration::from_millis(5),
        };

        cache.set(key.clone(), sentinel.clone());

        let hit = cache
            .get(&key)
            .expect("cache should contain the inserted entry");
        assert_eq!(hit.frames_processed, 42);
        assert!((hit.encoding_fps - 999.0).abs() < f64::EPSILON);
    }

    /// Codec IDs in a `BenchmarkFilter` successfully narrow down results;
    /// a filter for a codec that is absent from the results returns an empty
    /// slice — this validates the incremental "skip cached sequences" path via
    /// the filter API.
    #[test]
    fn test_incremental_benchmark_skips_cached() {
        let config = BenchmarkConfig::builder()
            .add_codec(CodecConfig::new(CodecId::Av1))
            .add_codec(CodecConfig::new(CodecId::Vp9))
            .parallel_jobs(2)
            .build()
            .expect("valid config");

        let suite = BenchmarkSuite::new(config);
        let results = suite.run_all().expect("run_all should succeed");

        // Filter to only AV1 results — should return exactly one entry.
        let av1_only = BenchmarkFilter::new()
            .with_codec_ids(vec![CodecId::Av1])
            .apply(&results);
        assert_eq!(av1_only.len(), 1);
        assert_eq!(av1_only[0].codec_id, CodecId::Av1);

        // Filter for a codec that was not benchmarked — returns empty.
        let absent = BenchmarkFilter::new()
            .with_codec_ids(vec![CodecId::Theora])
            .apply(&results);
        assert!(
            absent.is_empty(),
            "filter for absent codec should return empty"
        );
    }

    // -----------------------------------------------------------------------
    // Streaming CSV / JSON export convenience methods on BenchmarkResults
    // -----------------------------------------------------------------------

    /// Build a small two-sequence single-codec result set for streaming tests.
    fn streaming_sample_results() -> BenchmarkResults {
        use crate::metrics::QualityMetrics;
        let seq_a = SequenceResult {
            sequence_name: "alpha".to_string(),
            frames_processed: 60,
            encoding_fps: 24.0,
            decoding_fps: 120.0,
            file_size_bytes: 250_000,
            metrics: QualityMetrics {
                psnr: Some(34.5),
                ssim: Some(0.93),
                vmaf: None,
                ..QualityMetrics::default()
            },
            encoding_duration: Duration::from_secs(2),
            decoding_duration: Duration::from_secs(1),
        };
        let seq_b = SequenceResult {
            sequence_name: "beta".to_string(),
            frames_processed: 120,
            encoding_fps: 48.0,
            decoding_fps: 240.0,
            file_size_bytes: 600_000,
            metrics: QualityMetrics {
                psnr: Some(41.0),
                ssim: Some(0.985),
                vmaf: Some(95.5),
                ..QualityMetrics::default()
            },
            encoding_duration: Duration::from_secs(3),
            decoding_duration: Duration::from_secs(1),
        };
        let codec = CodecBenchmarkResult {
            codec_id: CodecId::Vp9,
            preset: Some("good".to_string()),
            bitrate_kbps: Some(1500),
            cq_level: None,
            sequence_results: vec![seq_a, seq_b],
            statistics: Statistics::default(),
        };
        BenchmarkResults {
            codec_results: vec![codec],
            timestamp: "2026-06-04".to_string(),
            total_duration: Duration::from_secs(8),
            config: BenchmarkConfig::default(),
        }
    }

    /// Two independent `export_csv_streaming` runs of the same results produce
    /// byte-for-byte identical output (deterministic streaming parity).
    #[test]
    fn test_export_csv_streaming_byte_for_byte() {
        let results = streaming_sample_results();

        let mut sink_a: Vec<u8> = Vec::new();
        let mut sink_b: Vec<u8> = Vec::new();
        results
            .export_csv_streaming(&mut sink_a)
            .expect("csv stream a");
        results
            .export_csv_streaming(&mut sink_b)
            .expect("csv stream b");

        assert_eq!(sink_a, sink_b, "streaming CSV must be deterministic");

        // Sanity: header + two data rows, content present.
        let text = String::from_utf8(sink_a).expect("utf8");
        assert!(text.starts_with("Codec,"));
        assert!(text.contains("alpha"));
        assert!(text.contains("beta"));
        assert_eq!(text.lines().count(), 3);
    }

    /// `export_json_streaming` output parses back (NDJSON) to records whose key
    /// fields equal the source results.
    #[test]
    fn test_export_json_streaming_roundtrip() {
        let results = streaming_sample_results();

        let mut sink: Vec<u8> = Vec::new();
        results
            .export_json_streaming(&mut sink)
            .expect("json stream");
        let text = String::from_utf8(sink).expect("utf8");

        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "one NDJSON line per sequence");

        // Flatten the source the same way the writer does for comparison.
        let source: Vec<(String, String, f64, u64)> = results
            .codec_results
            .iter()
            .flat_map(|c| {
                c.sequence_results.iter().map(move |s| {
                    (
                        format!("{:?}", c.codec_id),
                        s.sequence_name.clone(),
                        s.encoding_fps,
                        s.file_size_bytes,
                    )
                })
            })
            .collect();

        for (line, (codec, name, enc_fps, size)) in lines.iter().zip(source.iter()) {
            let v: serde_json::Value = serde_json::from_str(line).expect("parse ndjson line");
            assert_eq!(v["codec"], serde_json::Value::String(codec.clone()));
            assert_eq!(v["sequence_name"], serde_json::Value::String(name.clone()));
            assert!((v["encoding_fps"].as_f64().expect("enc fps") - enc_fps).abs() < 1e-9);
            assert_eq!(v["file_size_bytes"].as_u64().expect("size"), *size);
        }

        // Spot-check the optional metric carried through on the second row.
        let v1: serde_json::Value = serde_json::from_str(lines[1]).expect("parse second line");
        assert!((v1["vmaf"].as_f64().expect("vmaf") - 95.5).abs() < 1e-9);
    }
}

/// Benchmark preset configurations for common scenarios.
pub struct BenchmarkPresets;

impl BenchmarkPresets {
    /// Create a quick benchmark preset (fast, fewer iterations).
    #[must_use]
    pub fn quick() -> BenchmarkConfig {
        BenchmarkConfig {
            codecs: vec![
                CodecConfig::new(CodecId::Av1).with_preset("fast"),
                CodecConfig::new(CodecId::Vp9).with_preset("fast"),
            ],
            sequences: Vec::new(),
            parallel_jobs: num_cpus(),
            enable_psnr: true,
            enable_ssim: false,
            enable_vmaf: false,
            cache_dir: None,
            output_dir: PathBuf::from("./bench_quick"),
            max_frames: Some(30),
            warmup_iterations: 0,
            measurement_iterations: 1,
        }
    }

    /// Create a standard benchmark preset (balanced settings).
    #[must_use]
    pub fn standard() -> BenchmarkConfig {
        BenchmarkConfig {
            codecs: vec![
                CodecConfig::new(CodecId::Av1),
                CodecConfig::new(CodecId::Vp9),
            ],
            sequences: Vec::new(),
            parallel_jobs: num_cpus(),
            enable_psnr: true,
            enable_ssim: true,
            enable_vmaf: false,
            cache_dir: Some(PathBuf::from("./bench_cache")),
            output_dir: PathBuf::from("./bench_results"),
            max_frames: Some(300),
            warmup_iterations: 1,
            measurement_iterations: 3,
        }
    }

    /// Create a comprehensive benchmark preset (all metrics, high quality).
    #[must_use]
    pub fn comprehensive() -> BenchmarkConfig {
        BenchmarkConfig {
            codecs: vec![
                CodecConfig::new(CodecId::Av1).with_preset("medium"),
                CodecConfig::new(CodecId::Av1).with_preset("slow"),
                CodecConfig::new(CodecId::Vp9).with_preset("good"),
                CodecConfig::new(CodecId::Vp9).with_preset("best"),
            ],
            sequences: Vec::new(),
            parallel_jobs: num_cpus(),
            enable_psnr: true,
            enable_ssim: true,
            enable_vmaf: true,
            cache_dir: Some(PathBuf::from("./bench_cache")),
            output_dir: PathBuf::from("./bench_comprehensive"),
            max_frames: None,
            warmup_iterations: 2,
            measurement_iterations: 5,
        }
    }

    /// Create a quality-focused benchmark preset.
    #[must_use]
    pub fn quality_focus() -> BenchmarkConfig {
        BenchmarkConfig {
            codecs: vec![
                CodecConfig::new(CodecId::Av1).with_cq_level(20),
                CodecConfig::new(CodecId::Av1).with_cq_level(30),
                CodecConfig::new(CodecId::Av1).with_cq_level(40),
            ],
            sequences: Vec::new(),
            parallel_jobs: num_cpus() / 2,
            enable_psnr: true,
            enable_ssim: true,
            enable_vmaf: true,
            cache_dir: Some(PathBuf::from("./bench_cache")),
            output_dir: PathBuf::from("./bench_quality"),
            max_frames: None,
            warmup_iterations: 1,
            measurement_iterations: 3,
        }
    }

    /// Create a speed-focused benchmark preset.
    #[must_use]
    pub fn speed_focus() -> BenchmarkConfig {
        BenchmarkConfig {
            codecs: vec![
                CodecConfig::new(CodecId::Av1).with_preset("ultrafast"),
                CodecConfig::new(CodecId::Av1).with_preset("fast"),
                CodecConfig::new(CodecId::Vp9).with_preset("realtime"),
            ],
            sequences: Vec::new(),
            parallel_jobs: num_cpus(),
            enable_psnr: true,
            enable_ssim: false,
            enable_vmaf: false,
            cache_dir: Some(PathBuf::from("./bench_cache")),
            output_dir: PathBuf::from("./bench_speed"),
            max_frames: Some(100),
            warmup_iterations: 2,
            measurement_iterations: 5,
        }
    }
}

/// Benchmark utilities for common operations.
pub struct BenchmarkUtils;

impl BenchmarkUtils {
    /// Calculate bitrate from file size and duration.
    #[must_use]
    pub fn calculate_bitrate(file_size_bytes: u64, duration_seconds: f64) -> f64 {
        if duration_seconds == 0.0 {
            return 0.0;
        }
        (file_size_bytes as f64 * 8.0) / duration_seconds / 1000.0
    }

    /// Calculate bits per pixel.
    #[must_use]
    pub fn calculate_bpp(
        file_size_bytes: u64,
        width: usize,
        height: usize,
        frame_count: usize,
    ) -> f64 {
        let total_pixels = (width * height * frame_count) as f64;
        if total_pixels == 0.0 {
            return 0.0;
        }
        (file_size_bytes as f64 * 8.0) / total_pixels
    }

    /// Calculate compression ratio.
    #[must_use]
    pub fn calculate_compression_ratio(
        original_size_bytes: u64,
        compressed_size_bytes: u64,
    ) -> f64 {
        if compressed_size_bytes == 0 {
            return 0.0;
        }
        original_size_bytes as f64 / compressed_size_bytes as f64
    }

    /// Format bytes as human-readable string.
    #[must_use]
    pub fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        format!("{:.2} {}", size, UNITS[unit_index])
    }

    /// Format duration as human-readable string.
    #[must_use]
    pub fn format_duration(duration: Duration) -> String {
        let secs = duration.as_secs();
        let hours = secs / 3600;
        let minutes = (secs % 3600) / 60;
        let seconds = secs % 60;

        if hours > 0 {
            format!("{hours}h {minutes}m {seconds}s")
        } else if minutes > 0 {
            format!("{minutes}m {seconds}s")
        } else {
            format!("{seconds}s")
        }
    }

    /// Parse bitrate string (e.g., "2000kbps", "5Mbps").
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    pub fn parse_bitrate(bitrate_str: &str) -> BenchResult<u32> {
        let bitrate_str = bitrate_str.to_lowercase();

        if let Some(stripped) = bitrate_str.strip_suffix("kbps") {
            stripped
                .trim()
                .parse()
                .map_err(|_| BenchError::InvalidConfig(format!("Invalid bitrate: {bitrate_str}")))
        } else if let Some(stripped) = bitrate_str.strip_suffix("mbps") {
            stripped
                .trim()
                .parse::<f64>()
                .map(|v| (v * 1000.0) as u32)
                .map_err(|_| BenchError::InvalidConfig(format!("Invalid bitrate: {bitrate_str}")))
        } else {
            bitrate_str
                .parse()
                .map_err(|_| BenchError::InvalidConfig(format!("Invalid bitrate: {bitrate_str}")))
        }
    }

    /// Generate a benchmark summary.
    #[must_use]
    pub fn generate_summary(results: &BenchmarkResults) -> String {
        let mut summary = String::new();
        summary.push_str("# Benchmark Summary\n\n");

        for codec_result in &results.codec_results {
            summary.push_str(&format!("## {:?}\n", codec_result.codec_id));

            if let Some(preset) = &codec_result.preset {
                summary.push_str(&format!("Preset: {preset}\n"));
            }

            summary.push_str(&format!(
                "Mean Encoding FPS: {:.2}\n",
                codec_result.statistics.mean_encoding_fps
            ));

            summary.push_str(&format!(
                "Mean Decoding FPS: {:.2}\n",
                codec_result.statistics.mean_decoding_fps
            ));

            if let Some(psnr) = codec_result.statistics.mean_psnr {
                summary.push_str(&format!("Mean PSNR: {psnr:.2} dB\n"));
            }

            if let Some(ssim) = codec_result.statistics.mean_ssim {
                summary.push_str(&format!("Mean SSIM: {ssim:.4}\n"));
            }

            summary.push('\n');
        }

        summary
    }
}

/// Benchmark filter for filtering results based on criteria.
///
/// Supports filtering by codec, quality metrics, encoding speed, and
/// optionally by timestamp range for historical comparison.
#[derive(Debug, Clone, Default)]
pub struct BenchmarkFilter {
    min_encoding_fps: Option<f64>,
    max_encoding_fps: Option<f64>,
    min_psnr: Option<f64>,
    max_psnr: Option<f64>,
    min_ssim: Option<f64>,
    max_ssim: Option<f64>,
    codec_ids: Vec<CodecId>,
    /// Inclusive lower bound on the result timestamp (ISO-8601 prefix, e.g. `"2024-01-01"`).
    date_from: Option<String>,
    /// Inclusive upper bound on the result timestamp (ISO-8601 prefix).
    date_to: Option<String>,
    /// Only include results with at least this many sequence results.
    min_sequence_count: Option<usize>,
    /// Only include results with a preset matching one of these strings.
    presets: Vec<String>,
}

impl BenchmarkFilter {
    /// Create a new filter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum encoding FPS.
    #[must_use]
    pub fn with_min_encoding_fps(mut self, fps: f64) -> Self {
        self.min_encoding_fps = Some(fps);
        self
    }

    /// Set maximum encoding FPS.
    #[must_use]
    pub fn with_max_encoding_fps(mut self, fps: f64) -> Self {
        self.max_encoding_fps = Some(fps);
        self
    }

    /// Set minimum PSNR.
    #[must_use]
    pub fn with_min_psnr(mut self, psnr: f64) -> Self {
        self.min_psnr = Some(psnr);
        self
    }

    /// Set maximum PSNR.
    #[must_use]
    pub fn with_max_psnr(mut self, psnr: f64) -> Self {
        self.max_psnr = Some(psnr);
        self
    }

    /// Set minimum SSIM.
    #[must_use]
    pub fn with_min_ssim(mut self, ssim: f64) -> Self {
        self.min_ssim = Some(ssim);
        self
    }

    /// Set maximum SSIM.
    #[must_use]
    pub fn with_max_ssim(mut self, ssim: f64) -> Self {
        self.max_ssim = Some(ssim);
        self
    }

    /// Set codec IDs to include.
    #[must_use]
    pub fn with_codec_ids(mut self, codec_ids: Vec<CodecId>) -> Self {
        self.codec_ids = codec_ids;
        self
    }

    /// Set the inclusive lower bound on the result timestamp for historical filtering.
    ///
    /// The timestamp is matched as a prefix (e.g. `"2024-01"` matches any January 2024 run).
    #[must_use]
    pub fn with_date_from(mut self, date_from: impl Into<String>) -> Self {
        self.date_from = Some(date_from.into());
        self
    }

    /// Set the inclusive upper bound on the result timestamp for historical filtering.
    #[must_use]
    pub fn with_date_to(mut self, date_to: impl Into<String>) -> Self {
        self.date_to = Some(date_to.into());
        self
    }

    /// Only include codec results with at least `count` sequence results.
    #[must_use]
    pub fn with_min_sequence_count(mut self, count: usize) -> Self {
        self.min_sequence_count = Some(count);
        self
    }

    /// Only include codec results whose preset is one of `presets`.
    #[must_use]
    pub fn with_presets(mut self, presets: Vec<String>) -> Self {
        self.presets = presets;
        self
    }

    /// Apply filter to results.
    ///
    /// The `timestamp` on [`BenchmarkResults`] is used for date-range checks.
    #[must_use]
    pub fn apply<'a>(&self, results: &'a BenchmarkResults) -> Vec<&'a CodecBenchmarkResult> {
        // Date-range check applies to the whole result set.
        if let Some(ref from) = self.date_from {
            if results.timestamp.as_str() < from.as_str() {
                return Vec::new();
            }
        }
        if let Some(ref to) = self.date_to {
            // Inclusive: allow timestamps that are <= date_to (prefix compare).
            if results.timestamp.as_str() > to.as_str() {
                return Vec::new();
            }
        }

        results
            .codec_results
            .iter()
            .filter(|r| self.matches_codec(r))
            .collect()
    }

    fn matches_codec(&self, result: &CodecBenchmarkResult) -> bool {
        // Check codec ID
        if !self.codec_ids.is_empty() && !self.codec_ids.contains(&result.codec_id) {
            return false;
        }

        // Check preset filter
        if !self.presets.is_empty() {
            let preset_match = result
                .preset
                .as_deref()
                .map(|p| self.presets.iter().any(|fp| fp == p))
                .unwrap_or(false);
            if !preset_match {
                return false;
            }
        }

        // Check minimum sequence count
        if let Some(min_count) = self.min_sequence_count {
            if result.sequence_results.len() < min_count {
                return false;
            }
        }

        // Check encoding FPS
        if let Some(min) = self.min_encoding_fps {
            if result.statistics.mean_encoding_fps < min {
                return false;
            }
        }

        if let Some(max) = self.max_encoding_fps {
            if result.statistics.mean_encoding_fps > max {
                return false;
            }
        }

        // Check PSNR
        if let Some(min) = self.min_psnr {
            if result.statistics.mean_psnr.map_or(true, |psnr| psnr < min) {
                return false;
            }
        }

        if let Some(max) = self.max_psnr {
            if result.statistics.mean_psnr.map_or(true, |psnr| psnr > max) {
                return false;
            }
        }

        // Check SSIM
        if let Some(min) = self.min_ssim {
            if result.statistics.mean_ssim.map_or(true, |ssim| ssim < min) {
                return false;
            }
        }

        if let Some(max) = self.max_ssim {
            if result.statistics.mean_ssim.map_or(true, |ssim| ssim > max) {
                return false;
            }
        }

        true
    }
}

/// Command-line interface helpers for benchmark tool.
pub struct CliHelpers;

impl CliHelpers {
    /// Parse codec from string.
    ///
    /// # Errors
    ///
    /// Returns an error if codec string is invalid.
    pub fn parse_codec(codec_str: &str) -> BenchResult<CodecId> {
        match codec_str.to_lowercase().as_str() {
            "av1" => Ok(CodecId::Av1),
            "vp9" => Ok(CodecId::Vp9),
            "vp8" => Ok(CodecId::Vp8),
            "theora" => Ok(CodecId::Theora),
            _ => Err(BenchError::InvalidConfig(format!(
                "Unknown codec: {codec_str}"
            ))),
        }
    }

    /// Generate example configuration file.
    #[must_use]
    pub fn generate_example_config() -> String {
        serde_json::to_string_pretty(&BenchmarkConfig::default())
            .unwrap_or_else(|_| String::from("{}"))
    }

    /// Print progress bar.
    pub fn print_progress(current: usize, total: usize, bar_width: usize) {
        let progress = if total > 0 {
            current as f64 / total as f64
        } else {
            0.0
        };

        let filled = (bar_width as f64 * progress) as usize;
        let empty = bar_width - filled;

        print!("\r[");
        for _ in 0..filled {
            print!("=");
        }
        for _ in 0..empty {
            print!(" ");
        }
        print!("] {:.1}% ({}/{})", progress * 100.0, current, total);

        use std::io::Write;
        std::io::stdout().flush().ok();
    }

    /// Clear progress bar.
    pub fn clear_progress() {
        print!("\r");
        for _ in 0..100 {
            print!(" ");
        }
        print!("\r");

        use std::io::Write;
        std::io::stdout().flush().ok();
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;

    #[test]
    fn test_benchmark_presets_quick() {
        let config = BenchmarkPresets::quick();
        assert_eq!(config.codecs.len(), 2);
        assert_eq!(config.max_frames, Some(30));
        assert_eq!(config.warmup_iterations, 0);
    }

    #[test]
    fn test_benchmark_presets_standard() {
        let config = BenchmarkPresets::standard();
        assert_eq!(config.codecs.len(), 2);
        assert_eq!(config.measurement_iterations, 3);
        assert!(config.enable_psnr);
        assert!(config.enable_ssim);
    }

    #[test]
    fn test_benchmark_presets_comprehensive() {
        let config = BenchmarkPresets::comprehensive();
        assert_eq!(config.codecs.len(), 4);
        assert!(config.enable_vmaf);
        assert_eq!(config.measurement_iterations, 5);
    }

    #[test]
    fn test_calculate_bitrate() {
        let bitrate = BenchmarkUtils::calculate_bitrate(1_000_000, 10.0);
        assert_eq!(bitrate, 800.0); // 1MB over 10s = 800 kbps
    }

    #[test]
    fn test_calculate_bpp() {
        let bpp = BenchmarkUtils::calculate_bpp(1_000_000, 1920, 1080, 100);
        assert!(bpp > 0.0);
    }

    #[test]
    fn test_calculate_compression_ratio() {
        let ratio = BenchmarkUtils::calculate_compression_ratio(10_000_000, 1_000_000);
        assert_eq!(ratio, 10.0);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(BenchmarkUtils::format_bytes(1024), "1.00 KB");
        assert_eq!(BenchmarkUtils::format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(BenchmarkUtils::format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(
            BenchmarkUtils::format_duration(Duration::from_secs(30)),
            "30s"
        );
        assert_eq!(
            BenchmarkUtils::format_duration(Duration::from_secs(90)),
            "1m 30s"
        );
        assert_eq!(
            BenchmarkUtils::format_duration(Duration::from_secs(3665)),
            "1h 1m 5s"
        );
    }

    #[test]
    fn test_parse_bitrate() {
        assert_eq!(
            BenchmarkUtils::parse_bitrate("2000kbps").expect("test expectation failed"),
            2000
        );
        assert_eq!(
            BenchmarkUtils::parse_bitrate("5Mbps").expect("test expectation failed"),
            5000
        );
        assert_eq!(
            BenchmarkUtils::parse_bitrate("1500").expect("test expectation failed"),
            1500
        );
    }

    #[test]
    fn test_parse_codec() {
        assert!(matches!(
            CliHelpers::parse_codec("av1").expect("test expectation failed"),
            CodecId::Av1
        ));
        assert!(matches!(
            CliHelpers::parse_codec("vp9").expect("test expectation failed"),
            CodecId::Vp9
        ));
        assert!(matches!(
            CliHelpers::parse_codec("vp8").expect("test expectation failed"),
            CodecId::Vp8
        ));
    }

    #[test]
    fn test_benchmark_filter() {
        let filter = BenchmarkFilter::new()
            .with_min_encoding_fps(30.0)
            .with_min_psnr(35.0);

        assert_eq!(filter.min_encoding_fps, Some(30.0));
        assert_eq!(filter.min_psnr, Some(35.0));
    }

    #[test]
    fn test_generate_summary() {
        let results = BenchmarkResults {
            codec_results: vec![],
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            total_duration: Duration::from_secs(100),
            config: BenchmarkConfig::default(),
        };

        let summary = BenchmarkUtils::generate_summary(&results);
        assert!(summary.contains("# Benchmark Summary"));
    }

    #[test]
    fn test_generate_example_config() {
        let config = CliHelpers::generate_example_config();
        assert!(!config.is_empty());
    }
}
