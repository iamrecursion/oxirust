//! Batch processing for multiple conform sessions.

use crate::config::ConformConfig;
use crate::error::ConformResult;
use crate::exporters::report::MatchReport;
use crate::session::ConformSession;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{error, info};

/// Batch job configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchJob {
    /// Job name.
    pub name: String,
    /// Timeline file path.
    pub timeline_path: PathBuf,
    /// Source media paths.
    pub source_paths: Vec<PathBuf>,
    /// Output path.
    pub output_path: PathBuf,
    /// Conform configuration.
    pub config: ConformConfig,
}

/// Batch processing result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    /// Job name.
    pub job_name: String,
    /// Success status.
    pub success: bool,
    /// Match report (if successful).
    pub report: Option<MatchReport>,
    /// Error message (if failed).
    pub error: Option<String>,
    /// Processing time in seconds.
    pub duration_seconds: f64,
}

/// Batch processing statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStatistics {
    /// Total jobs.
    pub total_jobs: usize,
    /// Successful jobs.
    pub successful_jobs: usize,
    /// Failed jobs.
    pub failed_jobs: usize,
    /// Total processing time in seconds.
    pub total_duration_seconds: f64,
    /// Average processing time per job.
    pub avg_duration_seconds: f64,
}

/// Batch processor for running multiple conform sessions.
pub struct BatchProcessor {
    /// Jobs to process.
    jobs: Vec<BatchJob>,
    /// Results.
    results: Arc<RwLock<Vec<BatchResult>>>,
    /// Parallel processing enabled.
    parallel: bool,
    /// Maximum parallel jobs.
    max_parallel: usize,
}

impl BatchProcessor {
    /// Create a new batch processor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            jobs: Vec::new(),
            results: Arc::new(RwLock::new(Vec::new())),
            parallel: true,
            max_parallel: num_cpus::get(),
        }
    }

    /// Add a job to the batch.
    pub fn add_job(&mut self, job: BatchJob) {
        self.jobs.push(job);
    }

    /// Add multiple jobs.
    pub fn add_jobs(&mut self, jobs: Vec<BatchJob>) {
        self.jobs.extend(jobs);
    }

    /// Set parallel processing.
    pub fn set_parallel(&mut self, parallel: bool) {
        self.parallel = parallel;
    }

    /// Set maximum parallel jobs.
    pub fn set_max_parallel(&mut self, max: usize) {
        self.max_parallel = max;
    }

    /// Process all jobs.
    ///
    /// # Errors
    ///
    /// Returns an error if batch processing fails entirely.
    pub async fn process(&mut self) -> ConformResult<BatchStatistics> {
        info!("Starting batch processing of {} jobs", self.jobs.len());
        let start_time = std::time::Instant::now();

        #[cfg(not(target_arch = "wasm32"))]
        if self.parallel {
            self.process_parallel().await?;
        } else {
            self.process_sequential().await?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.process_sequential().await?;
        }

        let duration = start_time.elapsed().as_secs_f64();
        let stats = self.compute_statistics(duration);

        info!(
            "Batch processing complete: {}/{} successful in {:.2}s",
            stats.successful_jobs, stats.total_jobs, duration
        );

        Ok(stats)
    }

    /// Process jobs sequentially.
    async fn process_sequential(&mut self) -> ConformResult<()> {
        for job in &self.jobs {
            let result = Self::process_job(job.clone()).await;
            self.results.write().push(result);
        }
        Ok(())
    }

    /// Process jobs in parallel.
    ///
    /// Prior to the oxisql 0.3.1 sync migration the sqlite layer was async,
    /// requiring a fresh `tokio::runtime::Runtime` per rayon thread so that
    /// `block_on` could drive the per-job future.  `SqliteConnectionBlocking`
    /// is fully synchronous, so we now use `tokio::task::JoinSet` — jobs run
    /// concurrently on the tokio thread pool with no manual runtime
    /// construction and no `block_on`.
    #[cfg(not(target_arch = "wasm32"))]
    async fn process_parallel(&mut self) -> ConformResult<()> {
        let capacity = self.jobs.len();
        let mut join_set = tokio::task::JoinSet::new();
        for job in &self.jobs {
            let job = job.clone();
            join_set.spawn(async move { Self::process_job(job).await });
        }

        let mut results = Vec::with_capacity(capacity);
        while let Some(outcome) = join_set.join_next().await {
            match outcome {
                Ok(result) => results.push(result),
                Err(join_err) => {
                    error!("parallel batch job task panicked: {join_err}");
                    results.push(BatchResult {
                        job_name: String::new(),
                        success: false,
                        report: None,
                        error: Some(format!("task panicked: {join_err}")),
                        duration_seconds: 0.0,
                    });
                }
            }
        }
        *self.results.write() = results;
        Ok(())
    }

    /// Execute a single conform job end-to-end.
    ///
    /// Takes an owned `BatchJob` so it can be moved into a
    /// `tokio::task::JoinSet`-spawned future (`Send + 'static`).
    async fn process_job(job: BatchJob) -> BatchResult {
        info!("Processing job: {}", job.name);
        let start_time = std::time::Instant::now();

        match ConformSession::new(
            job.name.clone(),
            &job.timeline_path,
            job.source_paths.clone(),
            job.output_path.clone(),
            job.config.clone(),
        ) {
            Ok(mut session) => match session.run().await {
                Ok(report) => {
                    info!("Job {} completed successfully", job.name);
                    BatchResult {
                        job_name: job.name,
                        success: true,
                        report: Some(report),
                        error: None,
                        duration_seconds: start_time.elapsed().as_secs_f64(),
                    }
                }
                Err(e) => {
                    error!("Job {} failed: {}", job.name, e);
                    BatchResult {
                        job_name: job.name,
                        success: false,
                        report: None,
                        error: Some(e.to_string()),
                        duration_seconds: start_time.elapsed().as_secs_f64(),
                    }
                }
            },
            Err(e) => {
                error!("Failed to create session for job {}: {}", job.name, e);
                BatchResult {
                    job_name: job.name,
                    success: false,
                    report: None,
                    error: Some(e.to_string()),
                    duration_seconds: start_time.elapsed().as_secs_f64(),
                }
            }
        }
    }

    /// Compute statistics from results.
    fn compute_statistics(&self, total_duration: f64) -> BatchStatistics {
        let results = self.results.read();
        let total_jobs = results.len();
        let successful_jobs = results.iter().filter(|r| r.success).count();
        let failed_jobs = total_jobs - successful_jobs;

        let avg_duration = if total_jobs > 0 {
            total_duration / total_jobs as f64
        } else {
            0.0
        };

        BatchStatistics {
            total_jobs,
            successful_jobs,
            failed_jobs,
            total_duration_seconds: total_duration,
            avg_duration_seconds: avg_duration,
        }
    }

    /// Get all results.
    #[must_use]
    pub fn get_results(&self) -> Vec<BatchResult> {
        self.results.read().clone()
    }

    /// Get successful results.
    #[must_use]
    pub fn get_successful_results(&self) -> Vec<BatchResult> {
        self.results
            .read()
            .iter()
            .filter(|r| r.success)
            .cloned()
            .collect()
    }

    /// Get failed results.
    #[must_use]
    pub fn get_failed_results(&self) -> Vec<BatchResult> {
        self.results
            .read()
            .iter()
            .filter(|r| !r.success)
            .cloned()
            .collect()
    }

    /// Export results to JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn export_results_json(&self) -> ConformResult<String> {
        let results = self.results.read();
        Ok(serde_json::to_string_pretty(&*results)?)
    }

    /// Save results to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save_results<P: AsRef<Path>>(&self, path: P) -> ConformResult<()> {
        let json = self.export_results_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load jobs from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn load_jobs<P: AsRef<Path>>(path: P) -> ConformResult<Vec<BatchJob>> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }

    /// Save jobs to a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save_jobs<P: AsRef<Path>>(jobs: &[BatchJob], path: P) -> ConformResult<()> {
        let json = serde_json::to_string_pretty(jobs)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

impl Default for BatchProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_processor_creation() {
        let processor = BatchProcessor::new();
        assert_eq!(processor.jobs.len(), 0);
        assert!(processor.parallel);
    }

    #[test]
    fn test_add_job() {
        let mut processor = BatchProcessor::new();
        let job = BatchJob {
            name: "Test Job".to_string(),
            timeline_path: PathBuf::from("/test/timeline.edl"),
            source_paths: vec![PathBuf::from("/test/media")],
            output_path: PathBuf::from("/test/output"),
            config: ConformConfig::default(),
        };

        processor.add_job(job);
        assert_eq!(processor.jobs.len(), 1);
    }

    #[test]
    fn test_statistics_computation() {
        let processor = BatchProcessor::new();
        let stats = processor.compute_statistics(100.0);
        assert_eq!(stats.total_jobs, 0);
        assert_eq!(stats.successful_jobs, 0);
    }
}
