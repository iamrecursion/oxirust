//! Professional Archive and Digital Preservation System for `OxiMedia`
//!
//! `oximedia-archive-pro` provides comprehensive tools for long-term digital preservation
//! of media files, including:
//!
//! - **Preservation Packaging**: `BagIt`, OAIS (SIP/AIP/DIP), TAR, and ZIP
//! - **Format Migration**: Planning, execution, and validation of format migrations
//! - **Checksum Management**: Multi-algorithm checksums (MD5, SHA-256, SHA-512, xxHash, BLAKE3)
//! - **Metadata Preservation**: PREMIS, METS, Dublin Core metadata
//! - **Version Control**: Track versions and changes over time
//! - **Fixity Checking**: Periodic integrity verification
//! - **Risk Assessment**: Format obsolescence monitoring
//! - **Emulation Support**: Prepare for future emulation needs
//! - **Documentation**: Auto-generate preservation documentation
//!
//! # Example
//!
//! ```rust,no_run
//! use oximedia_archive_pro::{
//!     package::bagit::BagItBuilder,
//!     checksum::{ChecksumAlgorithm, ChecksumGenerator},
//!     metadata::premis::PremisMetadata,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a BagIt package
//! let bag = BagItBuilder::new(std::path::PathBuf::from("/path/to/bag"))
//!     .with_algorithm(ChecksumAlgorithm::Sha256)
//!     .with_metadata("Contact-Name", "Archivist")
//!     .add_file(std::path::Path::new("/path/to/media.mkv"))?
//!     .build()?;
//!
//! // Generate preservation metadata
//! let premis = PremisMetadata::for_file(std::path::Path::new("/path/to/media.mkv"))?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]

pub mod access_policy;
pub mod archive_extensions;
pub mod archive_report;
pub mod archive_stats;
pub mod audit_trail;
pub mod bit_rot_detection;
pub mod checksum;
pub mod cold_storage;
pub mod collection_manifest;
pub mod cost_estimator;
pub mod datacite;
pub mod deaccession;
pub mod deep_archive;
pub mod disaster_recovery;
pub mod docs;
pub mod emulation;
pub mod emulation_planning;
pub mod fixity;
pub mod format_migration;
pub mod format_registry;
pub mod format_validator;
pub mod iiif_manifest;
pub mod ingest;
pub mod integrity_check;
pub mod integrity_dashboard;
pub mod lockss_package;
pub mod metadata;
pub mod metadata_crosswalk;
pub mod migrate;
pub mod migration_plan;
pub mod oais_model;
pub mod package;
pub mod policy;
pub mod preservation_level;
pub mod provenance_chain;
pub mod replication_verify;
pub mod restore_workflow;
pub mod retention;
pub mod risk;
pub mod secure_delete;
pub mod sip_builder;
pub mod storage_quota;
pub mod version;
pub mod workflow_state;

use thiserror::Error;

/// Result type for archive preservation operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for archive preservation operations
#[derive(Error, Debug)]
pub enum Error {
    /// I/O error occurred
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Checksum verification failed
    #[error("Checksum verification failed: expected {expected}, got {actual}")]
    ChecksumMismatch {
        /// Expected checksum value
        expected: String,
        /// Actual checksum value
        actual: String,
    },

    /// Invalid `BagIt` package
    #[error("Invalid BagIt package: {0}")]
    InvalidBag(String),

    /// Invalid OAIS package
    #[error("Invalid OAIS package: {0}")]
    InvalidOais(String),

    /// Format migration error
    #[error("Format migration error: {0}")]
    Migration(String),

    /// Metadata error
    #[error("Metadata error: {0}")]
    Metadata(String),

    /// XML parsing/serialization error
    #[error("XML error: {0}")]
    Xml(#[from] quick_xml::Error),

    /// Archive creation error
    #[error("Archive error: {0}")]
    Archive(String),

    /// Policy violation
    #[error("Policy violation: {0}")]
    PolicyViolation(String),

    /// Format at risk
    #[error("Format obsolescence risk: {0}")]
    FormatRisk(String),

    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

/// Preservation format recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PreservationFormat {
    /// Video: FFV1 in Matroska
    VideoFfv1Mkv,
    /// Video: UT Video in AVI
    VideoUtVideo,
    /// Audio: FLAC
    AudioFlac,
    /// Audio: WAV PCM
    AudioWav,
    /// Image: TIFF (uncompressed)
    ImageTiff,
    /// Image: PNG
    ImagePng,
    /// Image: JPEG 2000 (lossless)
    ImageJpeg2000,
    /// Document: PDF/A
    DocumentPdfA,
    /// Document: Plain text (UTF-8)
    DocumentText,
}

impl PreservationFormat {
    /// Returns the recommended file extension
    #[must_use]
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::VideoFfv1Mkv => "mkv",
            Self::VideoUtVideo => "avi",
            Self::AudioFlac => "flac",
            Self::AudioWav => "wav",
            Self::ImageTiff => "tiff",
            Self::ImagePng => "png",
            Self::ImageJpeg2000 => "jp2",
            Self::DocumentPdfA => "pdf",
            Self::DocumentText => "txt",
        }
    }

    /// Returns the MIME type
    #[must_use]
    pub const fn mime_type(&self) -> &'static str {
        match self {
            Self::VideoFfv1Mkv => "video/x-matroska",
            Self::VideoUtVideo => "video/x-msvideo",
            Self::AudioFlac => "audio/flac",
            Self::AudioWav => "audio/wav",
            Self::ImageTiff => "image/tiff",
            Self::ImagePng => "image/png",
            Self::ImageJpeg2000 => "image/jp2",
            Self::DocumentPdfA => "application/pdf",
            Self::DocumentText => "text/plain",
        }
    }

    /// Returns a description of the format
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::VideoFfv1Mkv => "FFV1 lossless video in Matroska container",
            Self::VideoUtVideo => "UT Video lossless codec in AVI container",
            Self::AudioFlac => "Free Lossless Audio Codec",
            Self::AudioWav => "WAV PCM uncompressed audio",
            Self::ImageTiff => "TIFF uncompressed image",
            Self::ImagePng => "PNG lossless image",
            Self::ImageJpeg2000 => "JPEG 2000 lossless image",
            Self::DocumentPdfA => "PDF/A archival document format",
            Self::DocumentText => "Plain text UTF-8",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preservation_format_extensions() {
        assert_eq!(PreservationFormat::VideoFfv1Mkv.extension(), "mkv");
        assert_eq!(PreservationFormat::AudioFlac.extension(), "flac");
        assert_eq!(PreservationFormat::ImageTiff.extension(), "tiff");
    }

    #[test]
    fn test_preservation_format_mime_types() {
        assert_eq!(
            PreservationFormat::VideoFfv1Mkv.mime_type(),
            "video/x-matroska"
        );
        assert_eq!(PreservationFormat::AudioFlac.mime_type(), "audio/flac");
    }

    /// End-to-end OAIS lifecycle: SIP → AIP → DIP with full metadata preservation.
    ///
    /// This test exercises:
    /// 1. SIP creation with content + descriptive metadata (Dublin Core + PREMIS)
    /// 2. AIP creation from the same content with preservation metadata
    /// 3. DIP generation from the AIP
    /// 4. Verification that all key metadata fields survive the full lifecycle
    #[test]
    fn test_oais_lifecycle() {
        use crate::metadata::dublin_core::DublinCoreRecord;
        use crate::metadata::premis::{
            EventType, PremisEvent, PremisMetadata, PremisMetadataWithRights, PremisObject,
            PremisRights, RightsBasis,
        };
        use crate::package::oais::{
            DipGenerationConfig, DipGenerator, OaisBuilder, OaisPackage, OaisPackageType,
        };
        use std::io::Write;
        use tempfile::{NamedTempFile, TempDir};

        let temp_dir = TempDir::new().expect("temp dir creation should succeed");

        // ── Step 1: Create a content file ─────────────────────────────────────
        let mut content_file = NamedTempFile::new().expect("temp file creation should succeed");
        content_file
            .write_all(b"OxiMedia test video frame data for OAIS lifecycle test")
            .expect("write should succeed");
        content_file.flush().expect("flush should succeed");

        // ── Step 2: Build descriptive metadata (Dublin Core) ──────────────────
        let dc = DublinCoreRecord::new()
            .with_title("OAIS Lifecycle Test Film")
            .with_creator("COOLJAPAN OU")
            .with_subject("digital preservation")
            .with_subject("archival testing")
            .with_description("Round-trip OAIS lifecycle test for oximedia-archive-pro")
            .with_publisher("OxiMedia Archive")
            .with_date("2025-03-14")
            .with_type("MovingImage")
            .with_format("video/x-matroska")
            .with_identifier("urn:test:oais-lifecycle-001")
            .with_language("en")
            .with_rights("CC0 1.0 Universal");

        // Serialize DC to a temp file for inclusion in packages
        let dc_xml = dc.to_xml().expect("DC XML serialization should succeed");
        let mut dc_file = NamedTempFile::new().expect("DC temp file should be created");
        dc_file
            .write_all(dc_xml.as_bytes())
            .expect("write DC should succeed");
        dc_file.flush().expect("flush DC should succeed");

        // ── Step 3: Build preservation metadata (PREMIS with rights) ──────────
        let premis_object =
            PremisObject::from_file(content_file.path(), "obj-lifecycle-001".to_string())
                .expect("PremisObject::from_file should succeed")
                .with_checksum("SHA-256", "placeholder-sha256");

        let ingest_event = PremisEvent::new("evt-ingest-001".to_string(), EventType::Ingestion)
            .with_detail("Ingested into OxiMedia test archive")
            .with_outcome("success")
            .with_linking_object("obj-lifecycle-001");

        let rights = PremisRights::new("rights-lifecycle-001", RightsBasis::License)
            .with_license_uri("https://creativecommons.org/publicdomain/zero/1.0/")
            .with_license_terms("CC0 1.0 Universal")
            .with_act_granted("disseminate")
            .with_act_granted("reproduce");

        let premis_doc = PremisMetadataWithRights {
            premis: PremisMetadata::new()
                .with_object(premis_object)
                .with_event(ingest_event),
            rights: vec![rights],
        };

        let premis_xml = premis_doc
            .to_xml()
            .expect("PREMIS XML serialization should succeed");
        let mut premis_file = NamedTempFile::new().expect("PREMIS temp file should be created");
        premis_file
            .write_all(premis_xml.as_bytes())
            .expect("write PREMIS should succeed");
        premis_file.flush().expect("flush PREMIS should succeed");

        // ── Step 4: Create SIP ────────────────────────────────────────────────
        let sip_dir = temp_dir.path().join("lifecycle-sip");
        let sip = OaisBuilder::new(
            sip_dir.clone(),
            OaisPackageType::Sip,
            "SIP-LC-001".to_string(),
        )
        .with_metadata("Creator", "COOLJAPAN OU")
        .with_metadata("Title", "OAIS Lifecycle Test Film")
        .with_metadata("Identifier", "urn:test:oais-lifecycle-001")
        .add_content_file(content_file.path(), std::path::Path::new("video.mkv"))
        .expect("add_content_file should succeed")
        .add_metadata_file(dc_file.path(), std::path::Path::new("dublin_core.xml"))
        .expect("add dc metadata should succeed")
        .add_metadata_file(premis_file.path(), std::path::Path::new("premis.xml"))
        .expect("add premis metadata should succeed")
        .build()
        .expect("SIP build should succeed");

        assert_eq!(sip.package_type, OaisPackageType::Sip);
        assert_eq!(sip.id, "SIP-LC-001");
        assert!(sip_dir.join("content/video.mkv").exists());
        assert!(sip_dir.join("metadata/dublin_core.xml").exists());
        assert!(sip_dir.join("metadata/premis.xml").exists());

        // ── Step 5: Create AIP (same content, preservation focus) ────────────
        let aip_dir = temp_dir.path().join("lifecycle-aip");
        let aip = OaisBuilder::new(
            aip_dir.clone(),
            OaisPackageType::Aip,
            "AIP-LC-001".to_string(),
        )
        .with_metadata("Creator", "COOLJAPAN OU")
        .with_metadata("Title", "OAIS Lifecycle Test Film")
        .with_metadata("Identifier", "urn:test:oais-lifecycle-001")
        .with_metadata("Source-SIP-ID", "SIP-LC-001")
        .add_content_file(content_file.path(), std::path::Path::new("video.mkv"))
        .expect("add_content_file should succeed")
        .add_metadata_file(dc_file.path(), std::path::Path::new("dublin_core.xml"))
        .expect("add dc metadata should succeed")
        .add_metadata_file(premis_file.path(), std::path::Path::new("premis.xml"))
        .expect("add premis metadata should succeed")
        .build()
        .expect("AIP build should succeed");

        assert_eq!(aip.package_type, OaisPackageType::Aip);
        assert_eq!(aip.id, "AIP-LC-001");
        assert!(aip_dir.join("content/video.mkv").exists());
        assert!(aip_dir.join("preservation").is_dir());

        // ── Step 6: Generate DIP from AIP ────────────────────────────────────
        let dip_dir = temp_dir.path().join("lifecycle-dip");
        let dip_config = DipGenerationConfig {
            include_preservation_metadata: true,
            include_submission_metadata: false,
            format_filter: None,
            max_files: None,
            extra_metadata: {
                let mut m = std::collections::HashMap::new();
                m.insert("Access-Level".to_string(), "Public".to_string());
                m.insert(
                    "Requested-By".to_string(),
                    "test-user@example.org".to_string(),
                );
                m
            },
        };

        let dip = DipGenerator::generate(&aip, &dip_dir, "DIP-LC-001", &dip_config)
            .expect("DIP generation from AIP should succeed");

        assert_eq!(dip.package_type, OaisPackageType::Dip);
        assert_eq!(dip.id, "DIP-LC-001");
        assert!(dip_dir.join("content/video.mkv").exists());
        assert!(dip_dir.join("metadata/dublin_core.xml").exists());
        assert!(dip_dir.join("metadata/premis.xml").exists());
        assert!(dip_dir.join("CHECKSUMS.txt").exists());

        // ── Step 7: Verify metadata field preservation ────────────────────────
        // DIP should reference its source AIP
        assert_eq!(
            dip.metadata.get("Source-AIP-ID").map(String::as_str),
            Some("AIP-LC-001"),
            "DIP must record its source AIP ID"
        );
        assert_eq!(
            dip.metadata.get("Access-Level").map(String::as_str),
            Some("Public"),
            "Extra metadata should be preserved in DIP"
        );
        assert_eq!(
            dip.metadata.get("Title").map(String::as_str),
            Some("OAIS Lifecycle Test Film"),
            "Title metadata should be preserved from AIP to DIP"
        );
        assert_eq!(
            dip.metadata.get("Identifier").map(String::as_str),
            Some("urn:test:oais-lifecycle-001"),
            "Identifier must be preserved from AIP to DIP"
        );

        // ── Step 8: Load DIP back and verify round-trip integrity ─────────────
        let loaded_dip = OaisPackage::load(&dip_dir).expect("Loading DIP from disk should succeed");
        assert_eq!(loaded_dip.package_type, OaisPackageType::Dip);
        assert_eq!(loaded_dip.id, "DIP-LC-001");
        assert_eq!(
            loaded_dip.metadata.get("Title").map(String::as_str),
            Some("OAIS Lifecycle Test Film"),
            "Title must round-trip through JSON manifest"
        );

        // ── Step 9: Verify Dublin Core XML integrity in the DIP ───────────────
        let dip_dc_xml = std::fs::read_to_string(dip_dir.join("metadata/dublin_core.xml"))
            .expect("Reading DIP DC XML should succeed");
        let recovered_dc =
            DublinCoreRecord::from_xml(&dip_dc_xml).expect("Parsing DIP DC XML should succeed");
        assert_eq!(
            recovered_dc.title.as_deref(),
            Some("OAIS Lifecycle Test Film"),
            "DC title must be preserved in DIP metadata"
        );
        assert_eq!(
            recovered_dc.creator.as_deref(),
            Some("COOLJAPAN OU"),
            "DC creator must be preserved"
        );
        assert_eq!(
            recovered_dc.subject.len(),
            2,
            "Both DC subjects must be preserved"
        );
        assert_eq!(
            recovered_dc.identifier.as_deref(),
            Some("urn:test:oais-lifecycle-001"),
            "DC identifier must be preserved end-to-end"
        );
        assert_eq!(
            recovered_dc.rights.as_deref(),
            Some("CC0 1.0 Universal"),
            "DC rights must survive the SIP → AIP → DIP pipeline"
        );

        // ── Step 10: Verify PREMIS XML integrity in the DIP ──────────────────
        let dip_premis_xml = std::fs::read_to_string(dip_dir.join("metadata/premis.xml"))
            .expect("Reading DIP PREMIS XML should succeed");
        assert!(
            dip_premis_xml.contains("obj-lifecycle-001"),
            "PREMIS object ID must be preserved in DIP"
        );
        assert!(
            dip_premis_xml.contains("<rightsBasis>license</rightsBasis>"),
            "PREMIS rights basis must be preserved"
        );
        assert!(
            dip_premis_xml.contains("CC0 1.0 Universal"),
            "License terms must be preserved end-to-end"
        );
    }
}
