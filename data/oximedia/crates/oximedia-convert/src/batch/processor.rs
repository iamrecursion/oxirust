// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Batch conversion processor for handling multiple file conversions.

use crate::{ConversionError, ConversionOptions, ConversionReport, Converter, Result};
use std::path::{Path, PathBuf};

/// Batch processor for converting multiple files.
#[derive(Debug)]
pub struct BatchProcessor {
    converter: Converter,
    max_parallel: usize,
    resume_support: bool,
}

impl BatchProcessor {
    /// Create a new batch processor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            converter: Converter::new(),
            max_parallel: num_cpus(),
            resume_support: true,
        }
    }

    /// Set the maximum number of parallel conversions.
    #[must_use]
    pub fn with_max_parallel(mut self, max: usize) -> Self {
        self.max_parallel = max.max(1);
        self
    }

    /// Enable or disable resume support.
    #[must_use]
    pub fn with_resume_support(mut self, enabled: bool) -> Self {
        self.resume_support = enabled;
        self
    }

    /// Process a batch of conversions.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn process_batch(&self, jobs: Vec<BatchJob>) -> Result<BatchReport> {
        use std::sync::Arc;
        use tokio::sync::Semaphore;

        let semaphore = Arc::new(Semaphore::new(self.max_parallel));
        let mut handles = Vec::new();
        let total = jobs.len();

        for (index, job) in jobs.into_iter().enumerate() {
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|e| ConversionError::Io(std::io::Error::other(e)))?;
            let converter = self.converter.clone();

            let handle = tokio::spawn(async move {
                let result = converter
                    .convert(&job.input, &job.output, job.options)
                    .await;
                drop(permit);
                (index, job.input, job.output, result)
            });

            handles.push(handle);
        }

        let mut successful = Vec::new();
        let mut failed = Vec::new();

        for handle in handles {
            match handle.await {
                Ok((index, _input, _output, Ok(report))) => {
                    successful.push((index, report));
                }
                Ok((index, input, output, Err(e))) => {
                    failed.push(BatchFailure {
                        index,
                        input,
                        output,
                        error: e.to_string(),
                    });
                }
                Err(e) => {
                    return Err(ConversionError::Io(std::io::Error::other(e)));
                }
            }
        }

        Ok(BatchReport {
            total,
            successful,
            failed,
        })
    }

    /// Process a batch of conversions sequentially on wasm32 targets.
    #[cfg(target_arch = "wasm32")]
    pub async fn process_batch(&self, jobs: Vec<BatchJob>) -> Result<BatchReport> {
        let total = jobs.len();
        let mut successful = Vec::new();
        let mut failed = Vec::new();

        for (index, job) in jobs.into_iter().enumerate() {
            match self
                .converter
                .convert(&job.input, &job.output, job.options)
                .await
            {
                Ok(report) => successful.push((index, report)),
                Err(error) => failed.push(BatchFailure {
                    index,
                    input: job.input,
                    output: job.output,
                    error: error.to_string(),
                }),
            }
        }

        Ok(BatchReport {
            total,
            successful,
            failed,
        })
    }

    /// Process files from a directory with a pattern.
    pub async fn process_directory<P: AsRef<Path>>(
        &self,
        input_dir: P,
        output_dir: P,
        pattern: &str,
        options: ConversionOptions,
    ) -> Result<BatchReport> {
        let input_dir = input_dir.as_ref();
        let output_dir = output_dir.as_ref();

        if !input_dir.is_dir() {
            return Err(ConversionError::InvalidInput(
                "Input must be a directory".to_string(),
            ));
        }

        std::fs::create_dir_all(output_dir).map_err(ConversionError::Io)?;

        let mut jobs = Vec::new();
        let entries = std::fs::read_dir(input_dir).map_err(ConversionError::Io)?;

        for entry in entries {
            let entry = entry.map_err(ConversionError::Io)?;
            let path = entry.path();

            if path.is_file() && matches_pattern(&path, pattern) {
                let file_name = path.file_stem().and_then(|s| s.to_str()).ok_or_else(|| {
                    ConversionError::InvalidInput("Invalid file name".to_string())
                })?;

                let output = output_dir.join(format!(
                    "{}.{}",
                    file_name,
                    get_extension_from_profile(&options.profile)
                ));

                jobs.push(BatchJob {
                    input: path,
                    output,
                    options: options.clone(),
                });
            }
        }

        self.process_batch(jobs).await
    }

    /// Process a batch of [`crate::pipeline::ConversionJob`]s in parallel with
    /// configurable concurrency.
    ///
    /// Uses `std::thread::spawn` with a counting semaphore so no async runtime
    /// is required.  Each job runs a lightweight stub that validates paths;
    /// a full encode pipeline would be wired here in production.
    ///
    /// When `config.fail_fast` is `true` the result list is truncated at the
    /// first failure (remaining threads are joined first to avoid leaks).
    pub fn process_parallel(
        &self,
        jobs: Vec<crate::pipeline::ConversionJob>,
        config: BatchConfig,
    ) -> Vec<BatchResult> {
        use std::sync::{mpsc, Arc, Condvar, Mutex};

        let max_concurrent = config.max_concurrent.max(1);
        let (tx, rx) = mpsc::channel::<BatchResult>();
        let permits = Arc::new((Mutex::new(max_concurrent), Condvar::new()));

        let mut handles: Vec<std::thread::JoinHandle<()>> = Vec::with_capacity(jobs.len());

        for job in jobs {
            // Acquire a permit.
            {
                let (lock, cvar) = &*permits;
                let mut available = lock.lock().unwrap_or_else(|p| p.into_inner());
                while *available == 0 {
                    available = cvar.wait(available).unwrap_or_else(|p| p.into_inner());
                }
                *available -= 1;
            }

            let tx_clone = tx.clone();
            let permits_clone = Arc::clone(&permits);
            let job_id = job.id.clone();
            let input = job.input.clone();
            let output = job.output.clone();

            let handle = std::thread::spawn(move || {
                let result = run_batch_job_sync(&job_id, &input, &output);
                let _ = tx_clone.send(result);
                // Release the permit.
                let (lock, cvar) = &*permits_clone;
                let mut available = lock.lock().unwrap_or_else(|p| p.into_inner());
                *available += 1;
                cvar.notify_one();
            });

            handles.push(handle);
        }

        drop(tx); // close channel when all workers finish

        let mut results: Vec<BatchResult> = rx.iter().collect();
        for h in handles {
            let _ = h.join();
        }

        results.sort_by(|a, b| a.job_id.cmp(&b.job_id));

        if config.fail_fast {
            if let Some(pos) = results.iter().position(|r| r.error.is_some()) {
                results.truncate(pos + 1);
            }
        }

        results
    }
}

/// Execute a single conversion job synchronously (stub).
fn run_batch_job_sync(job_id: &str, input: &Path, _output: &Path) -> BatchResult {
    if input.as_os_str().is_empty() {
        return BatchResult {
            job_id: job_id.to_string(),
            error: Some("empty input path".to_string()),
        };
    }
    BatchResult {
        job_id: job_id.to_string(),
        error: None,
    }
}

/// Configuration for the `process_parallel` batch API.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of jobs running concurrently.
    pub max_concurrent: usize,
    /// Stop on the first failed job.
    pub fail_fast: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_concurrent: num_cpus(),
            fail_fast: false,
        }
    }
}

/// Result of a single job from `process_parallel`.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Job identifier (mirrors [`crate::pipeline::ConversionJob::id`]).
    pub job_id: String,
    /// Error message if the job failed, `None` on success.
    pub error: Option<String>,
}

impl BatchResult {
    /// Returns `true` if the job completed without error.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.error.is_none()
    }
}

impl Default for BatchProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// A single conversion job in a batch.
#[derive(Debug, Clone)]
pub struct BatchJob {
    /// Input file path
    pub input: PathBuf,
    /// Output file path
    pub output: PathBuf,
    /// Conversion options
    pub options: ConversionOptions,
}

/// Report from a batch conversion.
#[derive(Debug)]
pub struct BatchReport {
    /// Total number of jobs
    pub total: usize,
    /// Successful conversions with their reports
    pub successful: Vec<(usize, ConversionReport)>,
    /// Failed conversions
    pub failed: Vec<BatchFailure>,
}

impl BatchReport {
    /// Get the success rate as a percentage.
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.successful.len() as f64 / self.total as f64) * 100.0
    }

    /// Get the total duration of all conversions.
    #[must_use]
    pub fn total_duration(&self) -> std::time::Duration {
        self.successful
            .iter()
            .map(|(_, report)| report.duration)
            .sum()
    }
}

/// Information about a failed conversion.
#[derive(Debug)]
pub struct BatchFailure {
    /// Job index
    pub index: usize,
    /// Input file
    pub input: PathBuf,
    /// Output file
    pub output: PathBuf,
    /// Error message
    pub error: String,
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(4)
}

fn matches_pattern(path: &Path, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            if pattern.starts_with("*.") {
                return ext_str == &pattern[2..];
            }
        }
    }

    false
}

fn get_extension_from_profile(profile: &crate::Profile) -> &'static str {
    match profile {
        crate::Profile::WebOptimized => "mp4",
        crate::Profile::Streaming => "m3u8",
        crate::Profile::Archive => "mkv",
        crate::Profile::Email => "mp4",
        crate::Profile::Mobile => "mp4",
        crate::Profile::YouTube => "mp4",
        crate::Profile::Instagram => "mp4",
        crate::Profile::TikTok => "mp4",
        crate::Profile::Broadcast => "mxf",
        crate::Profile::AudioMp3 => "mp3",
        crate::Profile::AudioFlac => "flac",
        crate::Profile::AudioAac => "m4a",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_processor_creation() {
        let processor = BatchProcessor::new();
        assert!(processor.max_parallel > 0);
        assert!(processor.resume_support);
    }

    #[test]
    fn test_batch_processor_config() {
        let processor = BatchProcessor::new()
            .with_max_parallel(8)
            .with_resume_support(false);

        assert_eq!(processor.max_parallel, 8);
        assert!(!processor.resume_support);
    }

    #[test]
    fn test_matches_pattern() {
        let path = Path::new("test.mp4");
        assert!(matches_pattern(path, "*"));
        assert!(matches_pattern(path, "*.mp4"));
        assert!(!matches_pattern(path, "*.mkv"));
    }

    #[test]
    fn test_batch_report_success_rate() {
        let report = BatchReport {
            total: 10,
            successful: vec![],
            failed: vec![],
        };
        assert_eq!(report.success_rate(), 0.0);
    }

    // ── process_parallel tests ────────────────────────────────────────────────

    fn make_job(input: &str, output: &str) -> crate::pipeline::ConversionJob {
        use crate::formats::ContainerFormat;
        use crate::pipeline::JobPriority;
        use std::collections::HashMap;
        crate::pipeline::ConversionJob::new(
            std::path::PathBuf::from(input),
            std::path::PathBuf::from(output),
            ContainerFormat::Matroska,
            None,
            None,
            None,
            HashMap::new(),
            JobPriority::Normal,
        )
    }

    #[test]
    fn test_process_parallel_four_jobs_max_two_concurrent() {
        let processor = BatchProcessor::new();
        let jobs = vec![
            make_job("/tmp/a.mkv", "/tmp/out_a.mkv"),
            make_job("/tmp/b.mkv", "/tmp/out_b.mkv"),
            make_job("/tmp/c.mkv", "/tmp/out_c.mkv"),
            make_job("/tmp/d.mkv", "/tmp/out_d.mkv"),
        ];
        let config = BatchConfig {
            max_concurrent: 2,
            fail_fast: false,
        };
        let results = processor.process_parallel(jobs, config);
        assert_eq!(results.len(), 4, "all 4 jobs should complete");
        for r in &results {
            assert!(
                r.is_success(),
                "job {} should succeed: {:?}",
                r.job_id,
                r.error
            );
        }
    }

    #[test]
    fn test_process_parallel_empty_jobs() {
        let processor = BatchProcessor::new();
        let config = BatchConfig::default();
        let results = processor.process_parallel(vec![], config);
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert!(config.max_concurrent > 0);
        assert!(!config.fail_fast);
    }

    #[test]
    fn test_batch_result_is_success() {
        let ok = BatchResult {
            job_id: "j1".to_string(),
            error: None,
        };
        assert!(ok.is_success());

        let fail = BatchResult {
            job_id: "j2".to_string(),
            error: Some("err".to_string()),
        };
        assert!(!fail.is_success());
    }
}
