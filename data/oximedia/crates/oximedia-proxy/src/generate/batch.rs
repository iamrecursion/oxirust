//! Batch proxy generation for processing multiple files.

use super::{encoder::ProxyEncodeResult, settings::ProxyGenerationSettings, ProxyEncoder};
use crate::Result;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// Batch proxy generator for processing multiple files in parallel.
pub struct BatchProxyGenerator {
    settings: ProxyGenerationSettings,
    max_parallel: usize,
}

impl BatchProxyGenerator {
    /// Create a new batch proxy generator.
    pub fn new(settings: ProxyGenerationSettings) -> Self {
        Self {
            settings,
            max_parallel: num_cpus::get(),
        }
    }

    /// Set the maximum number of parallel encodes.
    #[must_use]
    pub fn with_max_parallel(mut self, max_parallel: usize) -> Self {
        self.max_parallel = max_parallel.max(1);
        self
    }

    /// Generate proxies for multiple input files.
    ///
    /// # Errors
    ///
    /// Returns an error if any encoding operation fails.
    pub async fn generate_batch(&self, inputs: &[(PathBuf, PathBuf)]) -> Result<Vec<BatchResult>> {
        tracing::info!("Starting batch generation for {} files", inputs.len());

        // Process in parallel using rayon
        let settings = self.settings.clone();
        let results: Vec<BatchResult> = inputs
            .par_iter()
            .map(|(input, output)| {
                let encoder = ProxyEncoder::new(settings.clone())?;
                let result =
                    tokio::runtime::Handle::current().block_on(encoder.encode(input, output));

                match result {
                    Ok(encode_result) => Ok(BatchResult::Success {
                        input: input.clone(),
                        output: output.clone(),
                        result: encode_result,
                    }),
                    Err(e) => Ok(BatchResult::Failed {
                        input: input.clone(),
                        output: output.clone(),
                        error: e.to_string(),
                    }),
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let success_count = results
            .iter()
            .filter(|r| matches!(r, BatchResult::Success { .. }))
            .count();
        let failed_count = results.len() - success_count;

        tracing::info!(
            "Batch generation complete: {} succeeded, {} failed",
            success_count,
            failed_count
        );

        Ok(results)
    }

    /// Generate proxies with a callback for progress tracking.
    pub async fn generate_batch_with_progress<F>(
        &self,
        inputs: &[(PathBuf, PathBuf)],
        mut progress_callback: F,
    ) -> Result<Vec<BatchResult>>
    where
        F: FnMut(usize, usize) + Send,
    {
        let total = inputs.len();
        let mut completed = 0;

        let settings = self.settings.clone();
        let mut results = Vec::with_capacity(total);

        for (input, output) in inputs {
            let encoder = ProxyEncoder::new(settings.clone())?;
            let result = encoder.encode(input, output).await;

            let batch_result = match result {
                Ok(encode_result) => BatchResult::Success {
                    input: input.clone(),
                    output: output.clone(),
                    result: encode_result,
                },
                Err(e) => BatchResult::Failed {
                    input: input.clone(),
                    output: output.clone(),
                    error: e.to_string(),
                },
            };

            results.push(batch_result);
            completed += 1;
            progress_callback(completed, total);
        }

        Ok(results)
    }
}

/// Result of a batch proxy generation operation.
#[derive(Debug, Clone)]
pub enum BatchResult {
    /// Successful generation.
    Success {
        /// Input file path.
        input: PathBuf,
        /// Output file path.
        output: PathBuf,
        /// Encoding result.
        result: ProxyEncodeResult,
    },
    /// Failed generation.
    Failed {
        /// Input file path.
        input: PathBuf,
        /// Output file path.
        output: PathBuf,
        /// Error message.
        error: String,
    },
}

impl BatchResult {
    /// Check if this result is successful.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }

    /// Check if this result is a failure.
    #[must_use]
    pub const fn is_failed(&self) -> bool {
        matches!(self, Self::Failed { .. })
    }

    /// Get the input path.
    #[must_use]
    pub fn input(&self) -> &Path {
        match self {
            Self::Success { input, .. } | Self::Failed { input, .. } => input,
        }
    }

    /// Get the output path.
    #[must_use]
    pub fn output(&self) -> &Path {
        match self {
            Self::Success { output, .. } | Self::Failed { output, .. } => output,
        }
    }
}

/// Helper function to get the number of CPUs.
#[allow(dead_code)]
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_generator_creation() {
        let settings = ProxyGenerationSettings::quarter_res_h264();
        let generator = BatchProxyGenerator::new(settings);
        assert!(generator.max_parallel > 0);
    }

    #[test]
    fn test_max_parallel() {
        let settings = ProxyGenerationSettings::quarter_res_h264();
        let generator = BatchProxyGenerator::new(settings).with_max_parallel(4);
        assert_eq!(generator.max_parallel, 4);
    }

    #[test]
    fn test_batch_result() {
        let result = BatchResult::Failed {
            input: PathBuf::from("input.mov"),
            output: PathBuf::from("output.mp4"),
            error: "test error".to_string(),
        };

        assert!(result.is_failed());
        assert!(!result.is_success());
        assert_eq!(result.input(), Path::new("input.mov"));
        assert_eq!(result.output(), Path::new("output.mp4"));
    }
}
