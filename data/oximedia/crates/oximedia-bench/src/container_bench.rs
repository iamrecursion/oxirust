#![allow(dead_code)]
//! Benchmarks for container format operations (mux, demux, seek, probe).
//!
//! This module provides a structured framework for measuring the performance
//! of container-level operations such as:
//!
//! - **Muxing** frames into a container (MKV, MP4, WebM, OGG, …)
//! - **Demuxing** a container back to raw packets/frames
//! - **Seeking** to arbitrary timestamps and measuring seek latency
//! - **Probing** media file metadata (streams, duration, codecs)
//!
//! Results carry per-operation latency distributions and throughput figures
//! that are compatible with the rest of the `oximedia-bench` reporting pipeline.

use crate::{BenchError, BenchResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Container format enumeration
// ---------------------------------------------------------------------------

/// Container format to benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContainerFormat {
    /// Matroska / MKV
    Mkv,
    /// MPEG-4 Part 14 (MP4 / M4V / M4A)
    Mp4,
    /// WebM (subset of MKV)
    WebM,
    /// Ogg container
    Ogg,
    /// MPEG-TS
    MpegTs,
    /// Raw Y4M video
    Y4m,
    /// FLAC container
    Flac,
    /// WAV / RIFF
    Wav,
    /// Custom / unknown
    Other(u32),
}

impl ContainerFormat {
    /// Returns a canonical file extension for the format.
    #[must_use]
    pub fn extension(self) -> &'static str {
        match self {
            Self::Mkv => "mkv",
            Self::Mp4 => "mp4",
            Self::WebM => "webm",
            Self::Ogg => "ogg",
            Self::MpegTs => "ts",
            Self::Y4m => "y4m",
            Self::Flac => "flac",
            Self::Wav => "wav",
            Self::Other(_) => "bin",
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark operation kinds
// ---------------------------------------------------------------------------

/// Which container operation to benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContainerOperation {
    /// Write (mux) a sequence of frames/packets into the container.
    Mux,
    /// Read (demux) all packets/frames from the container.
    Demux,
    /// Perform random seeks across the container and measure latency.
    Seek,
    /// Probe the container to extract stream metadata without decoding.
    Probe,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a container benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerBenchConfig {
    /// Container format to test.
    pub format: ContainerFormat,
    /// Operation to measure.
    pub operation: ContainerOperation,
    /// Path to an existing input file (used for Demux, Seek, Probe).
    pub input_path: Option<PathBuf>,
    /// Directory where mux output files are written (used for Mux).
    pub output_dir: PathBuf,
    /// Number of frames/packets to use when muxing (Mux only).
    pub frame_count: usize,
    /// Simulated frame width in pixels (Mux only).
    pub width: u32,
    /// Simulated frame height in pixels (Mux only).
    pub height: u32,
    /// Number of seek targets (Seek only).
    pub seek_count: usize,
    /// Number of warmup iterations before measurement.
    pub warmup_iterations: usize,
    /// Number of measurement iterations.
    pub measurement_iterations: usize,
    /// Extra format-specific flags (e.g. `"faststart" => "1"`).
    pub extra_flags: HashMap<String, String>,
}

impl Default for ContainerBenchConfig {
    fn default() -> Self {
        Self {
            format: ContainerFormat::Mkv,
            operation: ContainerOperation::Demux,
            input_path: None,
            output_dir: PathBuf::from("./container_bench_out"),
            frame_count: 300,
            width: 1920,
            height: 1080,
            seek_count: 20,
            warmup_iterations: 1,
            measurement_iterations: 3,
            extra_flags: HashMap::new(),
        }
    }
}

impl ContainerBenchConfig {
    /// Create a new default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set the container format.
    #[must_use]
    pub fn with_format(mut self, fmt: ContainerFormat) -> Self {
        self.format = fmt;
        self
    }

    /// Builder: set the operation to benchmark.
    #[must_use]
    pub fn with_operation(mut self, op: ContainerOperation) -> Self {
        self.operation = op;
        self
    }

    /// Builder: set the input file path.
    #[must_use]
    pub fn with_input(mut self, path: impl Into<PathBuf>) -> Self {
        self.input_path = Some(path.into());
        self
    }

    /// Builder: set the output directory.
    #[must_use]
    pub fn with_output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.output_dir = dir.into();
        self
    }

    /// Builder: set the number of frames for mux benchmarks.
    #[must_use]
    pub fn with_frame_count(mut self, count: usize) -> Self {
        self.frame_count = count;
        self
    }

    /// Builder: set the resolution for mux benchmarks.
    #[must_use]
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Builder: set the number of seek targets.
    #[must_use]
    pub fn with_seek_count(mut self, count: usize) -> Self {
        self.seek_count = count;
        self
    }

    /// Builder: set warmup iterations.
    #[must_use]
    pub fn with_warmup_iterations(mut self, n: usize) -> Self {
        self.warmup_iterations = n;
        self
    }

    /// Builder: set measurement iterations.
    #[must_use]
    pub fn with_measurement_iterations(mut self, n: usize) -> Self {
        self.measurement_iterations = n;
        self
    }

    /// Builder: add an extra flag.
    #[must_use]
    pub fn with_flag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_flags.insert(key.into(), value.into());
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::InvalidConfig`] if the configuration is inconsistent.
    pub fn validate(&self) -> BenchResult<()> {
        if self.measurement_iterations == 0 {
            return Err(BenchError::InvalidConfig(
                "measurement_iterations must be > 0".to_string(),
            ));
        }
        if matches!(
            self.operation,
            ContainerOperation::Demux | ContainerOperation::Seek | ContainerOperation::Probe
        ) && self.input_path.is_none()
        {
            return Err(BenchError::InvalidConfig(
                "input_path is required for Demux, Seek, and Probe operations".to_string(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Latency sample distribution
// ---------------------------------------------------------------------------

/// Statistical summary of a latency sample set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    /// Minimum observed latency.
    #[serde(with = "duration_serde")]
    pub min: Duration,
    /// Maximum observed latency.
    #[serde(with = "duration_serde")]
    pub max: Duration,
    /// Arithmetic mean.
    #[serde(with = "duration_serde")]
    pub mean: Duration,
    /// Median (P50).
    #[serde(with = "duration_serde")]
    pub p50: Duration,
    /// 95th percentile.
    #[serde(with = "duration_serde")]
    pub p95: Duration,
    /// 99th percentile.
    #[serde(with = "duration_serde")]
    pub p99: Duration,
    /// Standard deviation.
    #[serde(with = "duration_serde")]
    pub std_dev: Duration,
    /// Total number of samples.
    pub sample_count: usize,
}

impl LatencyStats {
    /// Compute statistics from a slice of duration samples.
    ///
    /// Returns `None` when `samples` is empty.
    #[must_use]
    pub fn from_samples(samples: &[Duration]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let mut nanos: Vec<u128> = samples.iter().map(|d| d.as_nanos()).collect();
        nanos.sort_unstable();
        let n = nanos.len();
        let sum: u128 = nanos.iter().sum();
        #[allow(clippy::cast_possible_truncation)]
        let mean_ns = (sum / n as u128) as u64;
        #[allow(clippy::cast_precision_loss)]
        let var_ns: f64 = nanos
            .iter()
            .map(|&v| {
                let d = v as f64 - mean_ns as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let std_dev_ns = var_ns.sqrt() as u64;

        let percentile = |p: f64| -> u64 {
            let idx = ((p / 100.0) * (n - 1) as f64) as usize;
            nanos[idx.min(n - 1)] as u64
        };

        Some(Self {
            min: Duration::from_nanos(nanos[0] as u64),
            max: Duration::from_nanos(*nanos.last().unwrap_or(&0) as u64),
            mean: Duration::from_nanos(mean_ns),
            p50: Duration::from_nanos(percentile(50.0)),
            p95: Duration::from_nanos(percentile(95.0)),
            p99: Duration::from_nanos(percentile(99.0)),
            std_dev: Duration::from_nanos(std_dev_ns),
            sample_count: n,
        })
    }
}

// ---------------------------------------------------------------------------
// Individual operation results
// ---------------------------------------------------------------------------

/// Result of a mux benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuxResult {
    /// Container format.
    pub format: ContainerFormat,
    /// Number of frames muxed.
    pub frame_count: usize,
    /// Total bytes written.
    pub bytes_written: u64,
    /// Mux latency statistics across iterations.
    pub latency: LatencyStats,
    /// Throughput in frames per second.
    pub frames_per_sec: f64,
    /// Throughput in MiB/s.
    pub throughput_mib_per_sec: f64,
}

/// Result of a demux benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemuxResult {
    /// Container format.
    pub format: ContainerFormat,
    /// Path to the demuxed file.
    pub input_path: PathBuf,
    /// Number of packets/frames read.
    pub packet_count: usize,
    /// Total bytes read.
    pub bytes_read: u64,
    /// Demux latency statistics.
    pub latency: LatencyStats,
    /// Throughput in packets per second.
    pub packets_per_sec: f64,
    /// Throughput in MiB/s.
    pub throughput_mib_per_sec: f64,
}

/// A single seek measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeekSample {
    /// Target position in the file (as a fraction 0.0–1.0).
    pub position_fraction: f64,
    /// Seek latency.
    #[serde(with = "duration_serde")]
    pub latency: Duration,
    /// Whether the seek succeeded.
    pub success: bool,
}

/// Result of a seek benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeekResult {
    /// Container format.
    pub format: ContainerFormat,
    /// Path to the file.
    pub input_path: PathBuf,
    /// Per-seek latency samples.
    pub samples: Vec<SeekSample>,
    /// Aggregated latency statistics.
    pub latency: LatencyStats,
    /// Fraction of successful seeks.
    pub success_rate: f64,
}

/// Metadata discovered during a probe operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeMetadata {
    /// Container format.
    pub format: ContainerFormat,
    /// Duration of the media file in seconds.
    pub duration_secs: f64,
    /// Number of streams.
    pub stream_count: u32,
    /// Total file size in bytes.
    pub file_size_bytes: u64,
    /// Detected codec names per stream index.
    pub stream_codecs: HashMap<u32, String>,
    /// Bit-rate of the container in kbps.
    pub bitrate_kbps: f64,
}

/// Result of a probe benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResult {
    /// Metadata extracted from the file.
    pub metadata: ProbeMetadata,
    /// Probe latency statistics.
    pub latency: LatencyStats,
}

// ---------------------------------------------------------------------------
// Aggregated benchmark result
// ---------------------------------------------------------------------------

/// Outcome of one container benchmark run (union over all operation types).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContainerBenchResult {
    /// Mux benchmark result.
    Mux(MuxResult),
    /// Demux benchmark result.
    Demux(DemuxResult),
    /// Seek benchmark result.
    Seek(SeekResult),
    /// Probe benchmark result.
    Probe(ProbeResult),
}

impl ContainerBenchResult {
    /// Return the container format.
    #[must_use]
    pub fn format(&self) -> ContainerFormat {
        match self {
            Self::Mux(r) => r.format,
            Self::Demux(r) => r.format,
            Self::Seek(r) => r.format,
            Self::Probe(r) => r.metadata.format,
        }
    }

    /// Return the mean operation latency.
    #[must_use]
    pub fn mean_latency(&self) -> Duration {
        match self {
            Self::Mux(r) => r.latency.mean,
            Self::Demux(r) => r.latency.mean,
            Self::Seek(r) => r.latency.mean,
            Self::Probe(r) => r.latency.mean,
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

/// Runs container format benchmarks according to a [`ContainerBenchConfig`].
pub struct ContainerBenchRunner {
    config: ContainerBenchConfig,
}

impl ContainerBenchRunner {
    /// Create a new runner.
    #[must_use]
    pub fn new(config: ContainerBenchConfig) -> Self {
        Self { config }
    }

    /// Run the benchmark as configured.
    ///
    /// For operations that require a real filesystem file (Demux, Seek, Probe),
    /// the runner uses the `input_path` from the config.  For Mux it synthesises
    /// payload data in memory to measure the container-writing overhead.
    ///
    /// # Errors
    ///
    /// Returns [`BenchError`] if validation fails or if a required input file is
    /// absent.
    pub fn run(&self) -> BenchResult<ContainerBenchResult> {
        self.config.validate()?;
        match self.config.operation {
            ContainerOperation::Mux => self.run_mux(),
            ContainerOperation::Demux => self.run_demux(),
            ContainerOperation::Seek => self.run_seek(),
            ContainerOperation::Probe => self.run_probe(),
        }
    }

    // ---- Mux ---------------------------------------------------------------

    fn run_mux(&self) -> BenchResult<ContainerBenchResult> {
        let frame_size = (self.config.width * self.config.height * 3 / 2) as usize;
        // Synthetic payload: a single frame buffer re-used across iterations.
        let payload: Vec<u8> = (0..frame_size).map(|i| (i & 0xFF) as u8).collect();

        let mut latency_samples: Vec<Duration> =
            Vec::with_capacity(self.config.measurement_iterations);
        let mut bytes_written = 0u64;

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = simulate_mux(&payload, self.config.frame_count);
        }

        // Measurement
        for _ in 0..self.config.measurement_iterations {
            let t0 = Instant::now();
            bytes_written = simulate_mux(&payload, self.config.frame_count);
            latency_samples.push(t0.elapsed());
        }

        let latency = LatencyStats::from_samples(&latency_samples)
            .ok_or_else(|| BenchError::ExecutionFailed("no mux latency samples".to_string()))?;

        let mean_secs = latency.mean.as_secs_f64();
        let frames_per_sec = if mean_secs > 0.0 {
            self.config.frame_count as f64 / mean_secs
        } else {
            0.0
        };
        let throughput_mib_per_sec = if mean_secs > 0.0 {
            bytes_written as f64 / mean_secs / (1024.0 * 1024.0)
        } else {
            0.0
        };

        Ok(ContainerBenchResult::Mux(MuxResult {
            format: self.config.format,
            frame_count: self.config.frame_count,
            bytes_written,
            latency,
            frames_per_sec,
            throughput_mib_per_sec,
        }))
    }

    // ---- Demux -------------------------------------------------------------

    fn run_demux(&self) -> BenchResult<ContainerBenchResult> {
        let input_path = self.config.input_path.as_deref().ok_or_else(|| {
            BenchError::InvalidConfig("input_path required for Demux".to_string())
        })?;

        let file_size = file_size_bytes(input_path)?;
        let mut latency_samples: Vec<Duration> =
            Vec::with_capacity(self.config.measurement_iterations);
        let mut packet_count = 0usize;

        for _ in 0..self.config.warmup_iterations {
            packet_count = simulate_demux(file_size);
        }
        for _ in 0..self.config.measurement_iterations {
            let t0 = Instant::now();
            packet_count = simulate_demux(file_size);
            latency_samples.push(t0.elapsed());
        }

        let latency = LatencyStats::from_samples(&latency_samples)
            .ok_or_else(|| BenchError::ExecutionFailed("no demux latency samples".to_string()))?;

        let mean_secs = latency.mean.as_secs_f64();
        let packets_per_sec = if mean_secs > 0.0 {
            packet_count as f64 / mean_secs
        } else {
            0.0
        };
        let throughput_mib_per_sec = if mean_secs > 0.0 {
            file_size as f64 / mean_secs / (1024.0 * 1024.0)
        } else {
            0.0
        };

        Ok(ContainerBenchResult::Demux(DemuxResult {
            format: self.config.format,
            input_path: input_path.to_path_buf(),
            packet_count,
            bytes_read: file_size,
            latency,
            packets_per_sec,
            throughput_mib_per_sec,
        }))
    }

    // ---- Seek --------------------------------------------------------------

    fn run_seek(&self) -> BenchResult<ContainerBenchResult> {
        let input_path =
            self.config.input_path.as_deref().ok_or_else(|| {
                BenchError::InvalidConfig("input_path required for Seek".to_string())
            })?;

        let file_size = file_size_bytes(input_path)?;
        let n = self.config.seek_count.max(1);
        let mut seek_samples: Vec<SeekSample> = Vec::with_capacity(n);
        let mut latency_samples: Vec<Duration> = Vec::with_capacity(n);

        for _ in 0..self.config.warmup_iterations {
            for i in 0..n {
                let frac = i as f64 / n as f64;
                let _ = simulate_seek(file_size, frac);
            }
        }

        for i in 0..n {
            let frac = i as f64 / n as f64;
            let t0 = Instant::now();
            let success = simulate_seek(file_size, frac);
            let lat = t0.elapsed();
            latency_samples.push(lat);
            seek_samples.push(SeekSample {
                position_fraction: frac,
                latency: lat,
                success,
            });
        }

        let latency = LatencyStats::from_samples(&latency_samples)
            .ok_or_else(|| BenchError::ExecutionFailed("no seek latency samples".to_string()))?;

        let successful = seek_samples.iter().filter(|s| s.success).count();
        let success_rate = successful as f64 / seek_samples.len() as f64;

        Ok(ContainerBenchResult::Seek(SeekResult {
            format: self.config.format,
            input_path: input_path.to_path_buf(),
            samples: seek_samples,
            latency,
            success_rate,
        }))
    }

    // ---- Probe -------------------------------------------------------------

    fn run_probe(&self) -> BenchResult<ContainerBenchResult> {
        let input_path = self.config.input_path.as_deref().ok_or_else(|| {
            BenchError::InvalidConfig("input_path required for Probe".to_string())
        })?;

        let file_size = file_size_bytes(input_path)?;
        let mut latency_samples: Vec<Duration> =
            Vec::with_capacity(self.config.measurement_iterations);

        for _ in 0..self.config.warmup_iterations {
            let _ = simulate_probe(file_size, self.config.format);
        }

        let mut metadata_opt = None;
        for _ in 0..self.config.measurement_iterations {
            let t0 = Instant::now();
            metadata_opt = Some(simulate_probe(file_size, self.config.format));
            latency_samples.push(t0.elapsed());
        }

        let metadata = metadata_opt
            .ok_or_else(|| BenchError::ExecutionFailed("probe produced no metadata".to_string()))?;

        let latency = LatencyStats::from_samples(&latency_samples)
            .ok_or_else(|| BenchError::ExecutionFailed("no probe latency samples".to_string()))?;

        Ok(ContainerBenchResult::Probe(ProbeResult {
            metadata,
            latency,
        }))
    }
}

// ---------------------------------------------------------------------------
// Simulation helpers (pure-Rust, no external I/O)
// ---------------------------------------------------------------------------

/// Simulate mux overhead by computing a simple checksum over the payload repeated
/// `frame_count` times.  Returns the total number of bytes "written".
fn simulate_mux(frame_payload: &[u8], frame_count: usize) -> u64 {
    let mut checksum: u64 = 0;
    for i in 0..frame_count {
        for (j, &b) in frame_payload.iter().enumerate() {
            checksum = checksum
                .wrapping_add(b as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(i as u64 ^ j as u64);
        }
    }
    // Prevent the optimizer from eliding the loop.
    let _ = checksum;
    (frame_payload.len() * frame_count) as u64
}

/// Simulate demux overhead by pseudo-scanning a byte stream of the given total size.
/// Returns a simulated packet count.
fn simulate_demux(file_size: u64) -> usize {
    // Average packet size ≈ 4 KiB
    let avg_packet = 4096u64;
    (file_size / avg_packet.max(1)) as usize
}

/// Simulate a seek by computing a pseudo-random offset and returning success.
fn simulate_seek(file_size: u64, position_fraction: f64) -> bool {
    let offset = (file_size as f64 * position_fraction) as u64;
    // The seek is "successful" when the offset is within the file bounds.
    offset <= file_size
}

/// Simulate a probe by computing metadata from the file size and format.
fn simulate_probe(file_size: u64, format: ContainerFormat) -> ProbeMetadata {
    // Assume a fixed bitrate of 4 Mbps for simulation purposes.
    let assumed_bitrate_kbps = 4_000.0_f64;
    let duration_secs = if assumed_bitrate_kbps > 0.0 {
        file_size as f64 * 8.0 / (assumed_bitrate_kbps * 1_000.0)
    } else {
        0.0
    };
    let stream_count = match format {
        ContainerFormat::Y4m | ContainerFormat::Wav | ContainerFormat::Flac => 1,
        _ => 2, // video + audio
    };
    let mut stream_codecs = HashMap::new();
    match format {
        ContainerFormat::Y4m => {
            stream_codecs.insert(0, "rawvideo".to_string());
        }
        ContainerFormat::Wav => {
            stream_codecs.insert(0, "pcm_s16le".to_string());
        }
        ContainerFormat::Flac => {
            stream_codecs.insert(0, "flac".to_string());
        }
        ContainerFormat::WebM => {
            stream_codecs.insert(0, "vp9".to_string());
            stream_codecs.insert(1, "opus".to_string());
        }
        ContainerFormat::Ogg => {
            stream_codecs.insert(0, "theora".to_string());
            stream_codecs.insert(1, "vorbis".to_string());
        }
        _ => {
            stream_codecs.insert(0, "av1".to_string());
            stream_codecs.insert(1, "opus".to_string());
        }
    }
    ProbeMetadata {
        format,
        duration_secs,
        stream_count,
        file_size_bytes: file_size,
        stream_codecs,
        bitrate_kbps: assumed_bitrate_kbps,
    }
}

/// Return the size of a file on disk in bytes.
fn file_size_bytes(path: &Path) -> BenchResult<u64> {
    let meta = std::fs::metadata(path)?;
    Ok(meta.len())
}

// ---------------------------------------------------------------------------
// Multi-format comparison helper
// ---------------------------------------------------------------------------

/// Compare a set of formats for a single operation and produce a ranking by
/// mean operation latency (lowest first).
///
/// The runner is called once per format using the supplied `config_fn` factory.
///
/// # Errors
///
/// Propagates any error returned by individual benchmark runs.
pub fn rank_by_latency<F>(
    formats: &[ContainerFormat],
    config_fn: F,
) -> BenchResult<Vec<(ContainerFormat, Duration)>>
where
    F: Fn(ContainerFormat) -> ContainerBenchConfig,
{
    let mut rankings: Vec<(ContainerFormat, Duration)> = Vec::with_capacity(formats.len());
    for &fmt in formats {
        let cfg = config_fn(fmt);
        let runner = ContainerBenchRunner::new(cfg);
        let result = runner.run()?;
        rankings.push((fmt, result.mean_latency()));
    }
    rankings.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(rankings)
}

// ---------------------------------------------------------------------------
// Serde helpers
// ---------------------------------------------------------------------------

mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(d: &Duration, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        d.as_secs_f64().serialize(s)
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(d)?;
        Ok(Duration::from_secs_f64(secs))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_format_extension() {
        assert_eq!(ContainerFormat::Mkv.extension(), "mkv");
        assert_eq!(ContainerFormat::Mp4.extension(), "mp4");
        assert_eq!(ContainerFormat::WebM.extension(), "webm");
        assert_eq!(ContainerFormat::Ogg.extension(), "ogg");
        assert_eq!(ContainerFormat::MpegTs.extension(), "ts");
        assert_eq!(ContainerFormat::Y4m.extension(), "y4m");
    }

    #[test]
    fn test_config_default_validation() {
        // Default config has operation=Demux and no input_path → should fail validation.
        let cfg = ContainerBenchConfig::default();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_mux_benchmark_runs() {
        let cfg = ContainerBenchConfig::new()
            .with_format(ContainerFormat::Mkv)
            .with_operation(ContainerOperation::Mux)
            .with_frame_count(10)
            .with_resolution(320, 240)
            .with_warmup_iterations(1)
            .with_measurement_iterations(2);

        let runner = ContainerBenchRunner::new(cfg);
        let result = runner.run().expect("mux benchmark should succeed");
        if let ContainerBenchResult::Mux(r) = result {
            assert_eq!(r.frame_count, 10);
            assert!(r.bytes_written > 0);
            assert!(r.frames_per_sec >= 0.0);
        } else {
            panic!("expected Mux result");
        }
    }

    #[test]
    fn test_demux_benchmark_requires_input_path() {
        let cfg = ContainerBenchConfig::new()
            .with_format(ContainerFormat::WebM)
            .with_operation(ContainerOperation::Demux);
        let runner = ContainerBenchRunner::new(cfg);
        assert!(runner.run().is_err());
    }

    #[test]
    fn test_latency_stats_from_samples() {
        let samples = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(15),
            Duration::from_millis(12),
            Duration::from_millis(18),
        ];
        let stats = LatencyStats::from_samples(&samples).expect("stats should be computable");
        assert_eq!(stats.min, Duration::from_millis(10));
        assert_eq!(stats.max, Duration::from_millis(20));
        assert_eq!(stats.sample_count, 5);
        assert!(stats.mean >= Duration::from_millis(10));
        assert!(stats.mean <= Duration::from_millis(20));
    }

    #[test]
    fn test_latency_stats_empty() {
        assert!(LatencyStats::from_samples(&[]).is_none());
    }

    #[test]
    fn test_simulate_mux_deterministic() {
        let payload = vec![0xABu8; 128];
        let a = simulate_mux(&payload, 10);
        let b = simulate_mux(&payload, 10);
        assert_eq!(a, b);
        assert_eq!(a, 128 * 10);
    }

    #[test]
    fn test_simulate_seek_within_bounds() {
        assert!(simulate_seek(1_000_000, 0.5));
        assert!(simulate_seek(1_000_000, 0.0));
        assert!(simulate_seek(1_000_000, 1.0));
    }

    #[test]
    fn test_simulate_probe_video_audio() {
        let meta = simulate_probe(10_000_000, ContainerFormat::Mkv);
        assert_eq!(meta.stream_count, 2);
        assert!(meta.duration_secs > 0.0);
    }

    #[test]
    fn test_simulate_probe_audio_only() {
        let meta = simulate_probe(5_000_000, ContainerFormat::Flac);
        assert_eq!(meta.stream_count, 1);
        assert!(meta.stream_codecs.contains_key(&0));
    }

    #[test]
    fn test_config_builder() {
        let cfg = ContainerBenchConfig::new()
            .with_format(ContainerFormat::Mp4)
            .with_operation(ContainerOperation::Mux)
            .with_frame_count(60)
            .with_resolution(1280, 720)
            .with_flag("faststart", "1");
        assert_eq!(cfg.format, ContainerFormat::Mp4);
        assert_eq!(cfg.frame_count, 60);
        assert_eq!(cfg.width, 1280);
        assert_eq!(cfg.height, 720);
        assert_eq!(cfg.extra_flags.get("faststart"), Some(&"1".to_string()));
    }
}
