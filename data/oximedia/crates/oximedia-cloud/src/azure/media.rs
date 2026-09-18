//! Azure Media Services integration

use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{CloudError, Result};

/// Azure Media Services wrapper
pub struct AzureMediaServices {
    client: Client,
    account_name: String,
    resource_group: String,
    subscription_id: String,
    access_token: String,
}

impl AzureMediaServices {
    /// Create new Azure Media Services client
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails
    #[allow(clippy::unused_async)]
    pub async fn new(
        account_name: String,
        resource_group: String,
        subscription_id: String,
    ) -> Result<Self> {
        crate::tls_provider::install_default_crypto_provider();
        let client = Client::new();

        // In production, would use Azure Identity SDK to get token
        let access_token = std::env::var("AZURE_ACCESS_TOKEN")
            .map_err(|_| CloudError::Authentication("AZURE_ACCESS_TOKEN not set".to_string()))?;

        Ok(Self {
            client,
            account_name,
            resource_group,
            subscription_id,
            access_token,
        })
    }

    /// Submit encoding job
    ///
    /// # Errors
    ///
    /// Returns an error if job submission fails
    pub async fn submit_encoding_job(&self, config: EncodingJobConfig) -> Result<String> {
        let url = format!(
            "https://management.azure.com/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Media/mediaServices/{}/transforms/{}/jobs/{}?api-version=2021-11-01",
            self.subscription_id,
            self.resource_group,
            self.account_name,
            config.transform_name,
            config.job_name
        );

        let job_request = JobRequest {
            properties: JobProperties {
                input: config.input_asset,
                outputs: config.outputs,
            },
        };

        let response = self
            .client
            .put(&url)
            .bearer_auth(&self.access_token)
            .json(&job_request)
            .send()
            .await
            .map_err(|e| CloudError::MediaService(format!("Failed to submit encoding job: {e}")))?;

        if !response.status().is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(CloudError::MediaService(format!(
                "Failed to submit job: {error_text}"
            )));
        }

        Ok(config.job_name)
    }

    /// Get encoding job status
    ///
    /// # Errors
    ///
    /// Returns an error if retrieving job status fails
    pub async fn get_job_status(
        &self,
        transform_name: &str,
        job_name: &str,
    ) -> Result<EncodingJobStatus> {
        let url = format!(
            "https://management.azure.com/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Media/mediaServices/{}/transforms/{}/jobs/{}?api-version=2021-11-01",
            self.subscription_id, self.resource_group, self.account_name, transform_name, job_name
        );

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await
            .map_err(|e| CloudError::MediaService(format!("Failed to get job status: {e}")))?;

        if !response.status().is_success() {
            return Err(CloudError::MediaService(
                "Failed to retrieve job status".to_string(),
            ));
        }

        let job_response: JobResponse = response
            .json()
            .await
            .map_err(|e| CloudError::Serialization(format!("Failed to parse job response: {e}")))?;

        let status = match job_response.properties.state.as_str() {
            "Queued" => EncodingJobStatus::Queued,
            "Scheduled" => EncodingJobStatus::Scheduled,
            "Processing" => EncodingJobStatus::Processing,
            "Finished" => EncodingJobStatus::Finished,
            "Error" => EncodingJobStatus::Error,
            "Canceled" => EncodingJobStatus::Canceled,
            _ => EncodingJobStatus::Unknown,
        };

        Ok(status)
    }

    /// Create streaming endpoint
    ///
    /// # Errors
    ///
    /// Returns an error if endpoint creation fails
    pub async fn create_streaming_endpoint(
        &self,
        endpoint_name: &str,
        config: StreamingEndpointConfig,
    ) -> Result<String> {
        let url = format!(
            "https://management.azure.com/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Media/mediaServices/{}/streamingEndpoints/{}?api-version=2021-11-01",
            self.subscription_id, self.resource_group, self.account_name, endpoint_name
        );

        let request_body = serde_json::json!({
            "location": config.location,
            "properties": {
                "scaleUnits": config.scale_units,
                "cdnEnabled": config.cdn_enabled,
            }
        });

        let response = self
            .client
            .put(&url)
            .bearer_auth(&self.access_token)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                CloudError::MediaService(format!("Failed to create streaming endpoint: {e}"))
            })?;

        if !response.status().is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(CloudError::MediaService(format!(
                "Failed to create endpoint: {error_text}"
            )));
        }

        Ok(endpoint_name.to_string())
    }

    /// Start streaming endpoint
    ///
    /// # Errors
    ///
    /// Returns an error if starting the endpoint fails
    pub async fn start_streaming_endpoint(&self, endpoint_name: &str) -> Result<()> {
        let url = format!(
            "https://management.azure.com/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Media/mediaServices/{}/streamingEndpoints/{}/start?api-version=2021-11-01",
            self.subscription_id, self.resource_group, self.account_name, endpoint_name
        );

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await
            .map_err(|e| {
                CloudError::MediaService(format!("Failed to start streaming endpoint: {e}"))
            })?;

        if !response.status().is_success() {
            return Err(CloudError::MediaService(
                "Failed to start streaming endpoint".to_string(),
            ));
        }

        Ok(())
    }

    /// Stop streaming endpoint
    ///
    /// # Errors
    ///
    /// Returns an error if stopping the endpoint fails
    pub async fn stop_streaming_endpoint(&self, endpoint_name: &str) -> Result<()> {
        let url = format!(
            "https://management.azure.com/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Media/mediaServices/{}/streamingEndpoints/{}/stop?api-version=2021-11-01",
            self.subscription_id, self.resource_group, self.account_name, endpoint_name
        );

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await
            .map_err(|e| {
                CloudError::MediaService(format!("Failed to stop streaming endpoint: {e}"))
            })?;

        if !response.status().is_success() {
            return Err(CloudError::MediaService(
                "Failed to stop streaming endpoint".to_string(),
            ));
        }

        Ok(())
    }

    /// Create or update asset
    ///
    /// # Errors
    ///
    /// Returns an error if asset creation fails
    pub async fn create_asset(&self, asset_name: &str) -> Result<String> {
        let url = format!(
            "https://management.azure.com/subscriptions/{}/resourceGroups/{}/providers/Microsoft.Media/mediaServices/{}/assets/{}?api-version=2021-11-01",
            self.subscription_id, self.resource_group, self.account_name, asset_name
        );

        let response = self
            .client
            .put(&url)
            .bearer_auth(&self.access_token)
            .json(&serde_json::json!({
                "properties": {}
            }))
            .send()
            .await
            .map_err(|e| CloudError::MediaService(format!("Failed to create asset: {e}")))?;

        if !response.status().is_success() {
            return Err(CloudError::MediaService(
                "Failed to create asset".to_string(),
            ));
        }

        Ok(asset_name.to_string())
    }
}

/// Encoding job configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingJobConfig {
    /// Transform name
    pub transform_name: String,
    /// Job name
    pub job_name: String,
    /// Input asset
    pub input_asset: String,
    /// Output assets
    pub outputs: Vec<String>,
}

/// Encoding job status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncodingJobStatus {
    /// Job queued
    Queued,
    /// Job scheduled
    Scheduled,
    /// Job processing
    Processing,
    /// Job finished
    Finished,
    /// Job error
    Error,
    /// Job canceled
    Canceled,
    /// Unknown status
    Unknown,
}

/// Streaming endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingEndpointConfig {
    /// Azure location
    pub location: String,
    /// Number of scale units
    pub scale_units: u32,
    /// Whether CDN is enabled
    pub cdn_enabled: bool,
}

#[derive(Debug, Serialize)]
struct JobRequest {
    properties: JobProperties,
}

#[derive(Debug, Serialize)]
struct JobProperties {
    input: String,
    outputs: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct JobResponse {
    properties: JobResponseProperties,
}

#[derive(Debug, Deserialize)]
struct JobResponseProperties {
    state: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_job_status() {
        assert_eq!(EncodingJobStatus::Queued, EncodingJobStatus::Queued);
        assert_ne!(EncodingJobStatus::Processing, EncodingJobStatus::Finished);
    }

    #[test]
    fn test_streaming_endpoint_config() {
        let config = StreamingEndpointConfig {
            location: "eastus".to_string(),
            scale_units: 1,
            cdn_enabled: true,
        };

        assert_eq!(config.scale_units, 1);
        assert!(config.cdn_enabled);
    }
}
