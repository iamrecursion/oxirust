//! Proxy and offline editing workflow system for OxiMedia.
//!
//! This crate provides comprehensive proxy workflow management for professional video editing,
//! enabling efficient offline-to-online workflows with full conforming support.
//!
//! # Features
//!
//! ## Proxy Generation
//!
//! - **Multiple Resolutions** - Quarter, half, and full resolution proxies
//! - **Codec Selection** - H.264, VP9, and other efficient codecs
//! - **Quality Presets** - Predefined quality levels for different workflows
//! - **Batch Processing** - Generate proxies for multiple files simultaneously
//! - **Automatic Creation** - Auto-generate proxies on media ingest
//!
//! ## Proxy Linking
//!
//! - **Original Association** - Link proxies to high-resolution originals
//! - **Database Management** - Persistent link database with SQLite
//! - **Link Verification** - Validate proxy-original relationships
//! - **Metadata Tracking** - Store timecode, duration, and other metadata
//!
//! ## Conforming
//!
//! - **EDL Support** - Conform from CMX 3600 and other EDL formats
//! - **XML Support** - Final Cut Pro XML and Premiere Pro XML
//! - **Automatic Relink** - Relink edited proxies to original media
//! - **Frame-Accurate** - Preserve exact frame accuracy during conform
//!
//! ## Workflows
//!
//! - **Offline Editing** - Edit with low-res proxies for performance
//! - **Online Finishing** - Conform to high-res for final output
//! - **Round-trip** - Complete offline-to-online-to-delivery pipeline
//! - **Multi-resolution** - Switch between resolutions seamlessly
//!
//! ## Additional Features
//!
//! - **Timecode Preservation** - Maintain accurate timecode across workflow
//! - **Metadata Sync** - Synchronize metadata between proxy and original
//! - **Smart Caching** - Intelligent proxy cache management
//! - **Validation** - Comprehensive validation and reporting
//!
//! # Quick Start
//!
//! ## Generate Proxies
//!
//! ```rust,no_run
//! use oximedia_proxy::{ProxyGenerator, ProxyPreset};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Generate a quarter-resolution H.264 proxy
//! let generator = ProxyGenerator::new();
//! let proxy_path = generator
//!     .generate("original.mov", "proxy.mp4", ProxyPreset::QuarterResH264)
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Link Proxy to Original
//!
//! ```rust,no_run
//! use oximedia_proxy::ProxyLinkManager;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut manager = ProxyLinkManager::new("links.db").await?;
//! manager.link_proxy("proxy.mp4", "original.mov").await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Conform from EDL
//!
//! ```rust,no_run
//! use oximedia_proxy::ConformEngine;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let engine = ConformEngine::new("links.db").await?;
//! let conformed = engine.conform_from_edl("edit.edl", "output.mov").await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Complete Workflow
//!
//! ```rust,no_run
//! use oximedia_proxy::{OfflineWorkflow, ProxyPreset};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut workflow = OfflineWorkflow::new("project.db").await?;
//!
//! // Ingest and create proxies
//! workflow.ingest("camera/clip001.mov", "proxy.mp4", ProxyPreset::QuarterResH264).await?;
//!
//! // After editing, conform to original
//! workflow.conform("edit.edl", "final.mov").await?;
//! # Ok(())
//! # }
//! ```

// `deny` (not `forbid`) so the single audited `memmap2::Mmap::map` call in
// `proxy_fingerprint::FingerprintEngine::hash_via_mmap` can opt in via a
// scoped `#[allow(unsafe_code)]` with a `// SAFETY:` justification. All other
// unsafe usage remains denied crate-wide.
#![deny(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::too_many_arguments)]

pub mod cache;
pub mod cleanup;
pub mod cloud_proxy;
pub mod conform;
pub mod examples;
pub mod format_compat;
pub mod frame_map;
pub mod generate;
pub mod generation;
pub mod link;
pub mod linking;
pub mod media_link;
pub mod metadata;
pub mod nle_format_select;
pub mod offline_edit;
pub mod offline_proxy;
pub mod proxy_aging;
pub mod proxy_api;
pub mod proxy_audit;
pub mod proxy_bandwidth;
pub mod proxy_cache;
pub mod proxy_checksum;
pub mod proxy_compare;
pub mod proxy_fingerprint;
pub mod proxy_format;
pub mod proxy_index;
pub mod proxy_manifest;
pub mod proxy_pipeline;
pub mod proxy_pool;
pub mod proxy_quality;
pub mod proxy_registry_ext;
pub mod proxy_scheduler;
pub mod proxy_status;
pub mod proxy_streaming;
pub mod proxy_sync;
pub mod registry;
pub mod relink_proxy;
pub mod render;
pub mod resolution;
pub mod sidecar;
pub mod smart_proxy;
pub mod spec;
pub mod tiers;
pub mod timecode;
pub mod transcode_proxy;
pub mod transcode_queue;
pub mod utils;
pub mod validation;
pub mod workflow;

use thiserror::Error;

/// Errors that can occur during proxy operations.
#[derive(Debug, Error)]
pub enum ProxyError {
    /// Invalid input file or configuration.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Invalid output configuration.
    #[error("Invalid output: {0}")]
    InvalidOutput(String),

    /// Proxy generation error.
    #[error("Generation error: {0}")]
    GenerationError(String),

    /// Proxy link error.
    #[error("Link error: {0}")]
    LinkError(String),

    /// Conform error.
    #[error("Conform error: {0}")]
    ConformError(String),

    /// Database error.
    #[error("Database error: {0}")]
    DatabaseError(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Timecode error.
    #[error("Timecode error: {0}")]
    TimecodeError(String),

    /// Metadata error.
    #[error("Metadata error: {0}")]
    MetadataError(String),

    /// Cache error.
    #[error("Cache error: {0}")]
    CacheError(String),

    /// Validation error.
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Workflow error.
    #[error("Workflow error: {0}")]
    WorkflowError(String),

    /// Unsupported operation or feature.
    #[error("Unsupported: {0}")]
    Unsupported(String),

    /// File not found.
    #[error("File not found: {0}")]
    FileNotFound(String),

    /// Link not found.
    #[error("Link not found for: {0}")]
    LinkNotFound(String),
}

/// Result type for proxy operations.
pub type Result<T> = std::result::Result<T, ProxyError>;

// Re-export main types
pub use cache::{CacheCleanup, CacheManager, CacheStrategy, CleanupPolicy};
pub use conform::{ConformEngine, ConformResult, EdlConformer, XmlConformer};
pub use generate::{
    BatchProxyGenerator, BatchResult, CompleteOutcome, ConstantBitrateModel, GenerationProgress,
    PresetInfo, ProgressiveProxy, ProxyEncodeResult, ProxyEncoder, ProxyGenerationSettings,
    ProxyGenerator, ProxyOptimizer, ProxyPreset, ProxyPresets, ProxySegment, SegmentPlan,
    SegmentSpec, StreamingProxyConfig, StreamingProxyGenerator,
};
pub use link::{LinkDatabase, LinkStatistics, ProxyLink, ProxyLinkManager, ProxyVerifier};
pub use metadata::{MetadataSync, MetadataTransfer};
pub use registry::{ProxyEntry, ProxyRegistry, RegistryRecord};
pub use render::{RenderConform, RenderReplace};
pub use resolution::{ProxyResolution, ProxyVariant, ResolutionManager, ResolutionSwitcher};
pub use sidecar::{
    Checksum, ChecksumAlgorithm, ProcessingRecord, SideCar, SidecarData, SidecarTimecode,
};
pub use spec::{ProxyCodec, ProxyResolutionMode, ProxySpec};
pub use timecode::{TimecodePreserver, TimecodeVerifier};
pub use validation::{
    DirectoryValidation, EdlValidationResult, PathValidator, ValidationChecker, ValidationReport,
    WorkflowValidator,
};
pub use workflow::{
    MediaInfo, OfflineWorkflow, OfflineWorkflowPlan, OnlineWorkflow, RoundtripWorkflow,
    StorageEstimate, WorkflowPhase, WorkflowPlan, WorkflowPlanner,
};

/// Proxy quality preset for quick setup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    /// Low quality, smallest file size (good for remote editing).
    Low,
    /// Medium quality, balanced size and quality (recommended for most workflows).
    Medium,
    /// High quality, larger files (for critical color work).
    High,
}

impl Quality {
    /// Get the recommended bitrate for this quality level at 1080p.
    #[must_use]
    pub const fn bitrate_1080p(&self) -> u64 {
        match self {
            Self::Low => 2_000_000,    // 2 Mbps
            Self::Medium => 5_000_000, // 5 Mbps
            Self::High => 10_000_000,  // 10 Mbps
        }
    }

    /// Get the recommended bitrate for this quality level at 4K.
    #[must_use]
    pub const fn bitrate_4k(&self) -> u64 {
        match self {
            Self::Low => 8_000_000,     // 8 Mbps
            Self::Medium => 20_000_000, // 20 Mbps
            Self::High => 40_000_000,   // 40 Mbps
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_bitrates() {
        assert_eq!(Quality::Low.bitrate_1080p(), 2_000_000);
        assert_eq!(Quality::Medium.bitrate_1080p(), 5_000_000);
        assert_eq!(Quality::High.bitrate_1080p(), 10_000_000);

        assert_eq!(Quality::Low.bitrate_4k(), 8_000_000);
        assert_eq!(Quality::Medium.bitrate_4k(), 20_000_000);
        assert_eq!(Quality::High.bitrate_4k(), 40_000_000);
    }

    #[test]
    fn test_error_display() {
        let err = ProxyError::InvalidInput("test".to_string());
        assert_eq!(err.to_string(), "Invalid input: test");

        let err = ProxyError::FileNotFound("test.mov".to_string());
        assert_eq!(err.to_string(), "File not found: test.mov");
    }
}
