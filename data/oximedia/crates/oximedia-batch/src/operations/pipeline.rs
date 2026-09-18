//! Pipeline executor for multi-step operations

use crate::error::{BatchError, Result};
use crate::job::{BatchJob, PipelineStep};
use crate::operations::{
    file_ops::FileOperationExecutor, media_ops::MediaOperationExecutor, OperationExecutor,
};
use async_trait::async_trait;
use std::path::PathBuf;

/// Pipeline executor
pub struct PipelineExecutor {
    file_executor: FileOperationExecutor,
    media_executor: MediaOperationExecutor,
}

impl PipelineExecutor {
    /// Create a new pipeline executor
    #[must_use]
    pub const fn new() -> Self {
        Self {
            file_executor: FileOperationExecutor::new(),
            media_executor: MediaOperationExecutor::new(),
        }
    }

    async fn execute_step(
        &self,
        step: &PipelineStep,
        job: &BatchJob,
        input_files: &[PathBuf],
    ) -> Result<Vec<PathBuf>> {
        tracing::info!("Executing pipeline step: {}", step.name);

        // Check condition if specified
        if let Some(condition) = &step.condition {
            if !Self::evaluate_condition(condition, input_files) {
                tracing::info!("Skipping step due to condition: {}", condition);
                return Ok(input_files.to_vec());
            }
        }

        // Execute the operation
        let result = match &step.operation {
            crate::job::BatchOperation::FileOp { .. } => {
                self.file_executor.execute(job, input_files).await
            }
            crate::job::BatchOperation::Transcode { .. }
            | crate::job::BatchOperation::QualityCheck { .. }
            | crate::job::BatchOperation::Analyze { .. } => {
                self.media_executor.execute(job, input_files).await
            }
            _ => Err(BatchError::ExecutionError(
                "Unsupported operation in pipeline".to_string(),
            )),
        };

        match result {
            Ok(outputs) => Ok(outputs),
            Err(e) => {
                if step.continue_on_error {
                    tracing::warn!("Step failed but continuing: {}", e);
                    Ok(input_files.to_vec())
                } else {
                    Err(e)
                }
            }
        }
    }

    fn evaluate_condition(condition: &str, input_files: &[PathBuf]) -> bool {
        // Evaluate simple conditions based on input file properties
        let trimmed = condition.trim();

        // "has_files" or "!empty" - check if input files exist
        if trimmed == "has_files" || trimmed == "!empty" {
            return !input_files.is_empty();
        }

        // "no_files" or "empty" - check if no input files
        if trimmed == "no_files" || trimmed == "empty" {
            return input_files.is_empty();
        }

        // "file_count > N" - compare file count
        if let Some(rest) = trimmed.strip_prefix("file_count") {
            let rest = rest.trim();
            if let Some(n_str) = rest.strip_prefix('>') {
                if let Ok(n) = n_str.trim().parse::<usize>() {
                    return input_files.len() > n;
                }
            }
            if let Some(n_str) = rest.strip_prefix(">=") {
                if let Ok(n) = n_str.trim().parse::<usize>() {
                    return input_files.len() >= n;
                }
            }
            if let Some(n_str) = rest.strip_prefix('<') {
                if let Ok(n) = n_str.trim().parse::<usize>() {
                    return input_files.len() < n;
                }
            }
        }

        // Default: condition not recognized, proceed (true)
        tracing::debug!(
            "Unknown pipeline condition '{}', defaulting to true",
            trimmed
        );
        true
    }
}

impl Default for PipelineExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl OperationExecutor for PipelineExecutor {
    async fn execute(&self, job: &BatchJob, input_files: &[PathBuf]) -> Result<Vec<PathBuf>> {
        let start = std::time::Instant::now();

        match &job.operation {
            crate::job::BatchOperation::Pipeline { steps } => {
                let mut current_files = input_files.to_vec();

                for step in steps {
                    current_files = self.execute_step(step, job, &current_files).await?;
                }

                tracing::info!("Pipeline completed in {:?}", start.elapsed());

                Ok(current_files)
            }
            _ => Err(BatchError::ValidationError(
                "Not a pipeline operation".to_string(),
            )),
        }
    }

    fn validate(&self, job: &BatchJob) -> Result<()> {
        match &job.operation {
            crate::job::BatchOperation::Pipeline { steps } => {
                if steps.is_empty() {
                    return Err(BatchError::ValidationError(
                        "Pipeline must have at least one step".to_string(),
                    ));
                }
                Ok(())
            }
            _ => Err(BatchError::ValidationError(
                "Not a pipeline operation".to_string(),
            )),
        }
    }

    fn estimate_duration(&self, job: &BatchJob, input_files: &[PathBuf]) -> Option<u64> {
        match &job.operation {
            crate::job::BatchOperation::Pipeline { steps } => {
                Some(steps.len() as u64 * input_files.len() as u64 * 60)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::FileOperation;

    #[test]
    fn test_pipeline_executor_creation() {
        let executor = PipelineExecutor::new();
        let _ = executor; // executor created successfully
    }

    #[tokio::test]
    async fn test_empty_pipeline_validation() {
        let executor = PipelineExecutor::new();
        let job = BatchJob::new(
            "test".to_string(),
            crate::job::BatchOperation::Pipeline { steps: vec![] },
        );

        let result = executor.validate(&job);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_pipeline_with_steps() {
        let executor = PipelineExecutor::new();

        let step = PipelineStep {
            name: "copy".to_string(),
            operation: crate::job::BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
            continue_on_error: false,
            condition: None,
        };

        let job = BatchJob::new(
            "test".to_string(),
            crate::job::BatchOperation::Pipeline { steps: vec![step] },
        );

        let result = executor.validate(&job);
        assert!(result.is_ok());
    }
}
