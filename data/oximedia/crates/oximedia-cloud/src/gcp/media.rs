//! Google Cloud Media Services integration

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{CloudError, Result};

/// GCP Media Services wrapper
pub struct GcpMediaServices {
    client: Client,
    project_id: String,
    access_token: String,
}

impl GcpMediaServices {
    /// Create new GCP Media Services client
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails
    #[allow(clippy::unused_async)]
    pub async fn new(project_id: String) -> Result<Self> {
        crate::tls_provider::install_default_crypto_provider();
        let client = Client::new();

        // In production, would use Google Auth library to get token
        let access_token = std::env::var("GCP_ACCESS_TOKEN")
            .map_err(|_| CloudError::Authentication("GCP_ACCESS_TOKEN not set".to_string()))?;

        Ok(Self {
            client,
            project_id,
            access_token,
        })
    }

    /// Submit transcoding job using Transcoder API
    ///
    /// # Errors
    ///
    /// Returns an error if job submission fails
    pub async fn submit_transcode_job(&self, config: TranscodeJobConfig) -> Result<String> {
        let url = format!(
            "https://transcoder.googleapis.com/v1/projects/{}/locations/{}/jobs",
            self.project_id, config.location
        );

        let request_body = serde_json::json!({
            "inputUri": config.input_uri,
            "outputUri": config.output_uri,
            "templateId": config.template_id,
            "config": config.config,
        });

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.access_token)
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(CloudError::MediaService(format!(
                "Failed to submit transcode job: {error_text}"
            )));
        }

        let job_response: JobResponse = response.json().await?;
        Ok(job_response.name)
    }

    /// Get transcode job status
    ///
    /// # Errors
    ///
    /// Returns an error if retrieving job status fails
    pub async fn get_job_status(&self, job_name: &str) -> Result<TranscodeJobStatus> {
        let url = format!("https://transcoder.googleapis.com/v1/{job_name}");

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::MediaService(
                "Failed to get job status".to_string(),
            ));
        }

        let job_response: JobResponse = response.json().await?;

        let status = match job_response.state.as_str() {
            "PENDING" => TranscodeJobStatus::Pending,
            "RUNNING" => TranscodeJobStatus::Running,
            "SUCCEEDED" => TranscodeJobStatus::Succeeded,
            "FAILED" => TranscodeJobStatus::Failed,
            _ => TranscodeJobStatus::Unknown,
        };

        Ok(status)
    }

    /// Analyze video using Video Intelligence API
    ///
    /// # Errors
    ///
    /// Returns an error if analysis fails
    pub async fn analyze_video(&self, config: VideoAnalysisConfig) -> Result<String> {
        let url = "https://videointelligence.googleapis.com/v1/videos:annotate".to_string();

        let request_body = serde_json::json!({
            "inputUri": config.input_uri,
            "features": config.features,
            "outputUri": config.output_uri,
        });

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.access_token)
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(CloudError::MediaService(format!(
                "Failed to submit video analysis: {error_text}"
            )));
        }

        let analysis_response: AnalysisResponse = response.json().await?;
        Ok(analysis_response.name)
    }

    /// Get video analysis results
    ///
    /// # Errors
    ///
    /// Returns an error if retrieving results fails
    pub async fn get_analysis_results(&self, operation_name: &str) -> Result<AnalysisResults> {
        let url = operation_name.to_string();

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::MediaService(
                "Failed to get analysis results".to_string(),
            ));
        }

        let results: AnalysisResults = response.json().await?;
        Ok(results)
    }

    /// Create Cloud Storage transfer job
    ///
    /// # Errors
    ///
    /// Returns an error if transfer job creation fails
    pub async fn create_transfer_job(&self, config: TransferJobConfig) -> Result<String> {
        let url = "https://storagetransfer.googleapis.com/v1/transferJobs";

        let request_body = serde_json::json!({
            "description": config.description,
            "projectId": self.project_id,
            "transferSpec": {
                "gcsDataSource": {
                    "bucketName": config.source_bucket,
                },
                "gcsDataSink": {
                    "bucketName": config.dest_bucket,
                },
            },
            "schedule": config.schedule,
        });

        let response = self
            .client
            .post(url)
            .bearer_auth(&self.access_token)
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(CloudError::MediaService(format!(
                "Failed to create transfer job: {error_text}"
            )));
        }

        let transfer_response: TransferJobResponse = response.json().await?;
        Ok(transfer_response.name)
    }

    /// Run transfer job
    ///
    /// # Errors
    ///
    /// Returns an error if running the transfer job fails
    pub async fn run_transfer_job(&self, job_name: &str) -> Result<()> {
        let url = format!("https://storagetransfer.googleapis.com/v1/{job_name}:run");

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CloudError::MediaService(
                "Failed to run transfer job".to_string(),
            ));
        }

        Ok(())
    }
}

/// Transcode job configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscodeJobConfig {
    /// Input GCS URI (<gs://bucket/path>)
    pub input_uri: String,
    /// Output GCS URI
    pub output_uri: String,
    /// Template ID
    pub template_id: Option<String>,
    /// GCP location (e.g., us-central1)
    pub location: String,
    /// Job configuration
    pub config: HashMap<String, serde_json::Value>,
}

/// Transcode job status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TranscodeJobStatus {
    /// Job pending
    Pending,
    /// Job running
    Running,
    /// Job succeeded
    Succeeded,
    /// Job failed
    Failed,
    /// Unknown status
    Unknown,
}

/// Video analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoAnalysisConfig {
    /// Input video URI
    pub input_uri: String,
    /// Output URI for results
    pub output_uri: Option<String>,
    /// Features to analyze
    pub features: Vec<String>,
}

/// Transfer job configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferJobConfig {
    /// Job description
    pub description: String,
    /// Source bucket
    pub source_bucket: String,
    /// Destination bucket
    pub dest_bucket: String,
    /// Schedule configuration
    pub schedule: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct JobResponse {
    name: String,
    state: String,
}

#[derive(Debug, Deserialize)]
struct AnalysisResponse {
    name: String,
}

/// Analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResults {
    /// Done status
    pub done: bool,
    /// Results data
    pub response: Option<HashMap<String, serde_json::Value>>,
    /// Error if failed
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TransferJobResponse {
    name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcode_job_status() {
        assert_eq!(TranscodeJobStatus::Pending, TranscodeJobStatus::Pending);
        assert_ne!(TranscodeJobStatus::Running, TranscodeJobStatus::Succeeded);
    }

    #[test]
    fn test_transcode_job_config() {
        let config = TranscodeJobConfig {
            input_uri: "gs://bucket/input.mp4".to_string(),
            output_uri: "gs://bucket/output/".to_string(),
            template_id: Some("template-1".to_string()),
            location: "us-central1".to_string(),
            config: HashMap::new(),
        };

        assert!(!config.input_uri.is_empty());
        assert_eq!(config.location, "us-central1");
    }

    #[test]
    fn test_video_analysis_config() {
        let config = VideoAnalysisConfig {
            input_uri: "gs://bucket/video.mp4".to_string(),
            output_uri: Some("gs://bucket/results/".to_string()),
            features: vec![
                "LABEL_DETECTION".to_string(),
                "SHOT_CHANGE_DETECTION".to_string(),
            ],
        };

        assert_eq!(config.features.len(), 2);
    }
}
