#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::option_if_let_else,
    clippy::pedantic,
    clippy::unused_self
)]

//! # Oximedia Cloud Platform Integration
//!
//! Comprehensive cloud storage and media services integration for AWS, Azure, and GCP.
//!
//! ## Features
//!
//! - Multi-cloud storage abstraction (S3, Azure Blob, GCS)
//! - Media processing service integration
//! - Transfer management with retry and resume
//! - Cost optimization strategies
//! - Security and encryption
//! - Advanced features like failover, replication, archival
//!
//! ## Example
//!
//! ```no_run
//! use oximedia_cloud::{CloudProvider, CloudStorage, create_storage};
//! use bytes::Bytes;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = CloudProvider::S3 {
//!     bucket: "my-bucket".to_string(),
//!     region: "us-east-1".to_string(),
//! };
//!
//! let storage = create_storage(provider).await?;
//! storage.upload("test.mp4", Bytes::from("data")).await?;
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

pub mod auto_scaling;
/// AWS SDK backend (S3 + Elemental media services).
///
/// Opt-in only: requires the non-default `aws-sdk` cargo feature because the
/// AWS smithy TLS stack has no Pure-Rust crypto provider (ring / aws-lc /
/// s2n are all C). The default build stays 100% Pure Rust; S3-compatible
/// endpoints are served by [`generic::GenericStorage`] instead.
#[cfg(feature = "aws-sdk")]
pub mod aws;
pub mod azure;
/// Backblaze B2 low-cost storage provider.
pub mod b2;
/// Cloud transfer bandwidth throttling and scheduling.
pub mod bandwidth_throttle;
/// Edge cache invalidation patterns (wildcard, tag-based).
pub mod cache_invalidation;
pub mod cdn;
pub mod cdn_config;
pub mod cdn_edge;
pub mod cloud_auth;
/// Cloud backup strategies: incremental, differential, and versioned backups.
pub mod cloud_backup;
pub mod cloud_credentials;
pub mod cloud_extras;
pub mod cloud_job;
/// Object lifecycle management: tier transitions, expiration, and archival rules.
pub mod cloud_lifecycle;
pub mod cloud_monitor;
pub mod cloud_queue;
/// HTTP connection pooling with keep-alive and timeout management.
pub mod connection_pool;
pub mod cost;
pub mod cost_model;
pub mod cost_monitor;
pub mod egress_policy;
pub mod error;
pub mod event_bridge;
pub mod gcp;
pub mod generic;
pub mod multicloud;
pub mod multiregion;
/// Cross-region transfer optimisation and routing.
pub mod multiregion_transfer;
pub mod object_store;
pub mod oci;
/// Pre-signed POST policy generation for browser-based direct uploads.
pub mod presigned_post;
pub mod provider;
pub mod region_selector;
pub mod replication_policy;
pub mod security;
pub mod storage;
pub mod storage_class;
pub mod storage_provider;
pub mod task_queue;
/// Cloud-native video thumbnail generation.
pub mod thumbnail;
/// Process-wide Pure-Rust rustls `CryptoProvider` bootstrap.
pub mod tls_provider;
pub mod transcoding;
pub mod transcoding_pipeline;
pub mod transfer;
pub mod types;
pub mod upload_manager;

pub use error::{CloudError, Result};
pub use provider::{create_storage, CloudProvider};
pub use types::{
    CloudStorage, ObjectInfo, ObjectMetadata, StorageClass, TransferProgress, UploadOptions,
};

// Re-export commonly used types
#[cfg(feature = "aws-sdk")]
pub use aws::{AwsMediaServices, S3Storage};
pub use azure::{AzureBlobStorage, AzureMediaServices};
pub use cost::{CostEstimator, StorageTier};
pub use gcp::{GcpMediaServices, GcsStorage};
pub use generic::GenericStorage;
pub use security::{Credentials, EncryptionConfig, KmsConfig};
pub use tls_provider::install_default_crypto_provider;
pub use transfer::{TransferConfig, TransferManager};
