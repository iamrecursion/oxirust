//! Cloud provider selection and storage creation

use std::sync::Arc;
use url::Url;

#[cfg(feature = "aws-sdk")]
use crate::aws::S3Storage;
use crate::azure::AzureBlobStorage;
use crate::error::{CloudError, Result};
use crate::gcp::GcsStorage;
use crate::generic::GenericStorage;
use crate::security::Credentials;
use crate::types::CloudStorage;

/// Cloud provider configuration
#[derive(Debug, Clone)]
pub enum CloudProvider {
    /// Amazon S3
    S3 {
        /// Bucket name
        bucket: String,
        /// AWS region
        region: String,
    },
    /// Azure Blob Storage
    Azure {
        /// Container name
        container: String,
        /// Storage account name
        account: String,
    },
    /// Google Cloud Storage
    GCS {
        /// Bucket name
        bucket: String,
        /// GCP project ID
        project: String,
    },
    /// Generic S3-compatible storage
    Generic {
        /// Endpoint URL
        endpoint: Url,
        /// Credentials
        credentials: Credentials,
        /// Bucket name
        bucket: String,
    },
}

impl CloudProvider {
    /// Get a human-readable name for the provider
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            CloudProvider::S3 { .. } => "AWS S3",
            CloudProvider::Azure { .. } => "Azure Blob Storage",
            CloudProvider::GCS { .. } => "Google Cloud Storage",
            CloudProvider::Generic { .. } => "Generic S3-compatible",
        }
    }

    /// Validate the provider configuration
    pub fn validate(&self) -> Result<()> {
        match self {
            CloudProvider::S3 { bucket, region } => {
                if bucket.is_empty() {
                    return Err(CloudError::InvalidConfig(
                        "Bucket name is empty".to_string(),
                    ));
                }
                if region.is_empty() {
                    return Err(CloudError::InvalidConfig("Region is empty".to_string()));
                }
            }
            CloudProvider::Azure { container, account } => {
                if container.is_empty() {
                    return Err(CloudError::InvalidConfig(
                        "Container name is empty".to_string(),
                    ));
                }
                if account.is_empty() {
                    return Err(CloudError::InvalidConfig(
                        "Account name is empty".to_string(),
                    ));
                }
            }
            CloudProvider::GCS { bucket, project } => {
                if bucket.is_empty() {
                    return Err(CloudError::InvalidConfig(
                        "Bucket name is empty".to_string(),
                    ));
                }
                if project.is_empty() {
                    return Err(CloudError::InvalidConfig("Project ID is empty".to_string()));
                }
            }
            CloudProvider::Generic {
                endpoint,
                credentials,
                bucket,
            } => {
                if endpoint.scheme() != "http" && endpoint.scheme() != "https" {
                    return Err(CloudError::InvalidConfig(
                        "Endpoint must use http or https".to_string(),
                    ));
                }
                if bucket.is_empty() {
                    return Err(CloudError::InvalidConfig(
                        "Bucket name is empty".to_string(),
                    ));
                }
                credentials.validate()?;
            }
        }
        Ok(())
    }
}

/// Create a cloud storage instance from a provider configuration
///
/// # Errors
///
/// Returns an error if:
/// - The provider configuration is invalid
/// - Authentication fails
/// - The storage service is unavailable
pub async fn create_storage(provider: CloudProvider) -> Result<Arc<dyn CloudStorage>> {
    provider.validate()?;

    match provider {
        #[cfg(feature = "aws-sdk")]
        CloudProvider::S3 { bucket, region } => {
            let storage = S3Storage::new(bucket, region).await?;
            Ok(Arc::new(storage))
        }
        #[cfg(not(feature = "aws-sdk"))]
        CloudProvider::S3 { .. } => Err(CloudError::ServiceUnavailable(
            "the AWS S3 backend requires the non-default `aws-sdk` cargo feature \
             (the AWS smithy TLS stack compiles C code and is excluded from the \
             Pure-Rust default build); use CloudProvider::Generic for \
             S3-compatible endpoints instead"
                .to_string(),
        )),
        CloudProvider::Azure { container, account } => {
            let storage = AzureBlobStorage::new(account, container).await?;
            Ok(Arc::new(storage))
        }
        CloudProvider::GCS { bucket, project } => {
            let storage = GcsStorage::new(bucket, project).await?;
            Ok(Arc::new(storage))
        }
        CloudProvider::Generic {
            endpoint,
            credentials,
            bucket,
        } => {
            let storage = GenericStorage::new(endpoint, credentials, bucket)?;
            Ok(Arc::new(storage))
        }
    }
}

/// Multi-region provider with automatic failover
#[derive(Debug, Clone)]
pub struct MultiRegionProvider {
    /// Primary provider
    pub primary: CloudProvider,
    /// Secondary providers for failover
    pub secondaries: Vec<CloudProvider>,
    /// Maximum retry attempts
    pub max_retries: usize,
}

impl MultiRegionProvider {
    /// Create a new multi-region provider
    #[must_use]
    pub fn new(primary: CloudProvider) -> Self {
        Self {
            primary,
            secondaries: Vec::new(),
            max_retries: 3,
        }
    }

    /// Add a secondary region
    pub fn add_secondary(&mut self, provider: CloudProvider) {
        self.secondaries.push(provider);
    }

    /// Create storage with automatic failover
    ///
    /// # Errors
    ///
    /// Returns an error if all providers fail to initialize
    pub async fn create_storage_with_failover(&self) -> Result<Arc<dyn CloudStorage>> {
        // Try primary first
        match create_storage(self.primary.clone()).await {
            Ok(storage) => return Ok(storage),
            Err(e) => {
                tracing::warn!("Primary storage failed: {}", e);
            }
        }

        // Try secondaries
        for (i, secondary) in self.secondaries.iter().enumerate() {
            match create_storage(secondary.clone()).await {
                Ok(storage) => {
                    tracing::info!("Using secondary storage #{}", i + 1);
                    return Ok(storage);
                }
                Err(e) => {
                    tracing::warn!("Secondary storage #{} failed: {}", i + 1, e);
                }
            }
        }

        Err(CloudError::ServiceUnavailable(
            "All storage providers unavailable".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_validation() {
        let valid_s3 = CloudProvider::S3 {
            bucket: "test-bucket".to_string(),
            region: "us-east-1".to_string(),
        };
        assert!(valid_s3.validate().is_ok());

        let invalid_s3 = CloudProvider::S3 {
            bucket: "".to_string(),
            region: "us-east-1".to_string(),
        };
        assert!(invalid_s3.validate().is_err());
    }

    #[test]
    fn test_provider_name() {
        let s3 = CloudProvider::S3 {
            bucket: "test".to_string(),
            region: "us-east-1".to_string(),
        };
        assert_eq!(s3.name(), "AWS S3");

        let azure = CloudProvider::Azure {
            container: "test".to_string(),
            account: "account".to_string(),
        };
        assert_eq!(azure.name(), "Azure Blob Storage");
    }

    #[test]
    fn test_multi_region_provider() {
        let primary = CloudProvider::S3 {
            bucket: "primary".to_string(),
            region: "us-east-1".to_string(),
        };
        let mut multi = MultiRegionProvider::new(primary);

        let secondary = CloudProvider::S3 {
            bucket: "secondary".to_string(),
            region: "us-west-2".to_string(),
        };
        multi.add_secondary(secondary);

        assert_eq!(multi.secondaries.len(), 1);
    }
}
