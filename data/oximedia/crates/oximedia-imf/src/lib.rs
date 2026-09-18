//! IMF (Interoperable Master Format) support for `OxiMedia`
//!
//! This crate provides comprehensive support for the IMF (Interoperable Master Format)
//! standard as defined by SMPTE ST 2067. IMF is a file-based format for the delivery
//! of finished media assets to broadcast and streaming platforms.
//!
//! # Overview
//!
//! IMF packages consist of:
//! - **CPL** (Composition Playlist): Defines the editorial composition - SMPTE ST 2067-3
//! - **PKL** (Packing List): Lists all assets with checksums - SMPTE ST 429-8
//! - **ASSETMAP**: Maps assets to files on disk - SMPTE ST 429-9
//! - **OPL** (Output Profile List): Defines output requirements - SMPTE ST 2067-8
//! - **MXF Essence**: Video/audio/subtitle track files
//!
//! # SMPTE Standards Compliance
//!
//! This implementation conforms to:
//! - SMPTE ST 2067-2: IMF Core Constraints
//! - SMPTE ST 2067-3: Composition Playlist (CPL)
//! - SMPTE ST 2067-5: Essence Component
//! - SMPTE ST 2067-8: Output Profile List (OPL)
//! - SMPTE ST 429-8: Packing List (PKL)
//! - SMPTE ST 429-9: Asset Map
//!
//! # Example: Reading an IMF Package
//!
//! ```ignore
//! use oximedia_imf::{ImfPackage, ImfError};
//!
//! // Load an IMF package from a directory
//! let package = ImfPackage::open("/path/to/imp")?;
//!
//! // Access the Composition Playlist
//! let cpl = package.cpl()?;
//! println!("Title: {}", cpl.content_title());
//! println!("Duration: {} frames", cpl.total_duration());
//! println!("Edit rate: {}", cpl.edit_rate());
//!
//! // List all sequences
//! for sequence in cpl.sequences() {
//!     println!("Sequence: {:?}", sequence.sequence_type());
//!     for resource in sequence.resources() {
//!         println!("  Resource: {}", resource.id());
//!     }
//! }
//!
//! // Validate the package
//! package.validate()?;
//! ```
//!
//! # Example: Creating an IMF Package
//!
//! ```ignore
//! use oximedia_imf::{ImfPackageBuilder, CplBuilder, EditRate};
//! use uuid::Uuid;
//!
//! let builder = ImfPackageBuilder::new("/path/to/output")
//!     .with_title("My IMF Package")
//!     .with_creator("OxiMedia")
//!     .with_edit_rate(EditRate::new(24, 1));
//!
//! // Add video track
//! builder.add_video_track("/path/to/video.mxf")?;
//!
//! // Add audio track
//! builder.add_audio_track("/path/to/audio.mxf")?;
//!
//! // Build the package (creates CPL, PKL, ASSETMAP)
//! let package = builder.build()?;
//! ```
//!
//! # Features
//!
//! - Parse and generate CPL, PKL, ASSETMAP, OPL files
//! - MXF essence file handling and validation
//! - Hash verification (SHA-1, MD5)
//! - Timeline validation
//! - SMPTE conformance checking
//! - Support for:
//!   - Multiple compositions per package
//!   - Supplemental packages
//!   - Versioning and incremental updates
//!   - HDR metadata
//!   - Multiple audio channels
//!   - Subtitles and captions
//!   - Markers and annotations

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod application_profile;
pub mod asset_map;
mod assetmap;
pub mod audio_layout;
pub mod compliance_matrix;
pub mod composition_map;
pub mod composition_sequence;
pub mod content_version;
mod cpl;
pub mod cpl_cache;
pub mod cpl_merge;
pub mod cpl_parser;
pub mod cpl_segment;
pub mod cpl_validator;
pub mod delivery;
mod essence;
pub mod essence_constraints;
pub mod essence_descriptor;
pub mod essence_hash;
pub mod essence_probe;
pub mod imf_archive;
pub mod imf_builder;
pub mod imf_diff;
pub mod imf_inspector;
pub mod imf_report;
pub mod imf_timeline;
pub mod imsc1;
pub mod integrity_check;
pub mod marker_list;
pub mod marker_resource;
pub mod metadata_extractor;
pub mod mxf_descriptor;
mod opl;
pub mod opl_document;
pub mod output_profile_list;
mod package;
pub mod package_validator;
pub mod partial_restore;
mod pkl;
pub mod pkl_document;
pub mod pkl_gen;
pub mod qc_report;
pub mod sidecar;
pub mod subtitle_resource;
pub mod supplemental_package;
pub mod track_file;
mod validator;
pub mod versioning;
pub mod xml_util;

use oximedia_core::error::OxiError;
use thiserror::Error;

// Re-exports
pub use assetmap::{Asset, AssetMap, Chunk, ChunkList};
pub use cpl::{
    CompositionPlaylist, CompositionTimecode, EditRate, MainAudioSequence, MainImageSequence,
    MarkerSequence, Resource, Sequence, SequenceType,
};
pub use essence::{
    AudioChannelConfig, ColorSpace, EssenceDescriptor, EssenceTrack, MxfEssence, TimecodeTrack,
};
pub use opl::{DeviceConstraints, OutputProfile, OutputProfileList};
pub use package::{ImfPackage, ImfPackageBuilder, SupplementalPackage};
pub use pkl::{Asset as PklAsset, HashAlgorithm, PackingList};
pub use validator::{ConformanceLevel, ValidationError, ValidationReport, Validator};

/// IMF-specific errors
#[derive(Error, Debug)]
pub enum ImfError {
    /// XML parsing error
    #[error("XML parsing error: {0}")]
    XmlError(String),

    /// Invalid XML structure
    #[error("Invalid XML structure: {0}")]
    InvalidStructure(String),

    /// Missing required element
    #[error("Missing required element: {0}")]
    MissingElement(String),

    /// Invalid UUID format
    #[error("Invalid UUID: {0}")]
    InvalidUuid(String),

    /// Hash mismatch
    #[error("Hash mismatch for asset {id}: expected {expected}, got {actual}")]
    HashMismatch {
        /// Asset ID
        id: String,
        /// Expected hash
        expected: String,
        /// Actual hash
        actual: String,
    },

    /// Asset not found
    #[error("Asset not found: {0}")]
    AssetNotFound(String),

    /// File not found
    #[error("File not found: {0}")]
    FileNotFound(String),

    /// Invalid edit rate
    #[error("Invalid edit rate: {0}")]
    InvalidEditRate(String),

    /// Timeline validation error
    #[error("Timeline validation error: {0}")]
    TimelineError(String),

    /// MXF parsing error
    #[error("MXF parsing error: {0}")]
    MxfError(String),

    /// SMPTE conformance violation
    #[error("SMPTE conformance violation: {0}")]
    ConformanceViolation(String),

    /// Invalid package structure or directory
    #[error("Invalid package: {0}")]
    InvalidPackage(String),

    /// Unsupported feature
    #[error("Unsupported feature: {0}")]
    Unsupported(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// `OxiMedia` core error
    #[error("OxiMedia error: {0}")]
    Oxi(#[from] OxiError),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

impl From<quick_xml::Error> for ImfError {
    fn from(err: quick_xml::Error) -> Self {
        Self::XmlError(err.to_string())
    }
}

impl From<uuid::Error> for ImfError {
    fn from(err: uuid::Error) -> Self {
        Self::InvalidUuid(err.to_string())
    }
}

impl From<chrono::ParseError> for ImfError {
    fn from(err: chrono::ParseError) -> Self {
        Self::InvalidStructure(format!("Date/time parse error: {err}"))
    }
}

/// Result type for IMF operations
pub type ImfResult<T> = Result<T, ImfError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_types() {
        let err = ImfError::MissingElement("TestElement".to_string());
        assert!(err.to_string().contains("Missing required element"));

        let err = ImfError::HashMismatch {
            id: "test-id".to_string(),
            expected: "abc123".to_string(),
            actual: "def456".to_string(),
        };
        assert!(err.to_string().contains("Hash mismatch"));
    }
}
