//! Batch operation implementations

pub mod file_ops;
pub mod media_ops;
pub mod mmap_reader;
pub mod pipeline;

pub use file_ops::FileOperation;
pub use media_ops::{AnalysisType, MediaOperation};
pub use mmap_reader::{open_smart, MmapReader, MMAP_THRESHOLD};
pub use pipeline::PipelineExecutor;

use crate::error::Result;
use crate::job::BatchJob;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Output format types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    /// MP4 container
    Mp4,
    /// MKV container
    Mkv,
    /// MOV container
    Mov,
    /// MXF container
    Mxf,
    /// JPEG image
    Jpeg,
    /// PNG image
    Png,
    /// JSON data
    Json,
    /// CSV data
    Csv,
    /// XML data
    Xml,
    /// Custom format
    Custom(String),
}

/// Operation executor trait
#[async_trait]
pub trait OperationExecutor: Send + Sync {
    /// Execute the operation
    ///
    /// # Arguments
    ///
    /// * `job` - The batch job to execute
    /// * `input_files` - List of input files
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails
    async fn execute(&self, job: &BatchJob, input_files: &[PathBuf]) -> Result<Vec<PathBuf>>;

    /// Validate the operation
    ///
    /// # Arguments
    ///
    /// * `job` - The batch job to validate
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails
    fn validate(&self, job: &BatchJob) -> Result<()>;

    /// Estimate execution time in seconds
    ///
    /// # Arguments
    ///
    /// * `job` - The batch job
    /// * `input_files` - List of input files
    fn estimate_duration(&self, job: &BatchJob, input_files: &[PathBuf]) -> Option<u64>;
}

/// Operation result
#[derive(Debug, Clone)]
pub struct OperationResult {
    /// Input files processed
    pub input_files: Vec<PathBuf>,
    /// Output files generated
    pub output_files: Vec<PathBuf>,
    /// Success flag
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Duration in seconds
    pub duration_secs: f64,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl OperationResult {
    /// Create a successful result
    #[must_use]
    pub fn success(
        input_files: Vec<PathBuf>,
        output_files: Vec<PathBuf>,
        duration_secs: f64,
    ) -> Self {
        Self {
            input_files,
            output_files,
            success: true,
            error: None,
            duration_secs,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Create a failed result
    #[must_use]
    pub fn failure(input_files: Vec<PathBuf>, error: String, duration_secs: f64) -> Self {
        Self {
            input_files,
            output_files: Vec::new(),
            success: false,
            error: Some(error),
            duration_secs,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add metadata
    #[must_use]
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_result_success() {
        let result = OperationResult::success(
            vec![PathBuf::from("input.mp4")],
            vec![PathBuf::from("output.mp4")],
            10.5,
        );

        assert!(result.success);
        assert!(result.error.is_none());
        assert_eq!(result.input_files.len(), 1);
        assert_eq!(result.output_files.len(), 1);
    }

    #[test]
    fn test_operation_result_failure() {
        let result = OperationResult::failure(
            vec![PathBuf::from("input.mp4")],
            "Processing failed".to_string(),
            5.0,
        );

        assert!(!result.success);
        assert_eq!(result.error, Some("Processing failed".to_string()));
        assert!(result.output_files.is_empty());
    }

    #[test]
    fn test_operation_result_metadata() {
        let result = OperationResult::success(vec![], vec![], 1.0)
            .with_metadata("fps".to_string(), "30".to_string());

        assert_eq!(result.metadata.get("fps"), Some(&"30".to_string()));
    }
}
