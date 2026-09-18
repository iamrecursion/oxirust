//! Comprehensive metadata standards support for OxiMedia.
//!
//! This crate provides parsing and writing support for all major media metadata formats:
//! - **ID3v2** (v2.3, v2.4) - MP3 metadata
//! - **Vorbis Comments** - Ogg/FLAC/Opus metadata
//! - **APEv2** - APE and other formats
//! - **iTunes metadata** - MP4/M4A metadata atoms
//! - **XMP** - Adobe Extensible Metadata Platform
//! - **EXIF** - Image metadata (JPEG, TIFF)
//! - **IPTC** - Photo metadata (legacy and IIM)
//! - **QuickTime metadata** - MOV user data atoms
//! - **Matroska tags** - MKV/WebM tags
//!
//! # Features
//!
//! - Parse and write all major metadata formats
//! - Unicode support (UTF-8, UTF-16)
//! - Picture/artwork handling (embedded images)
//! - Custom tags/fields
//! - Format conversion with field mapping
//! - Metadata validation
//! - Character encoding detection/conversion
//! - Multiple value support
//! - Hierarchical metadata (XMP)
//!
//! # Example
//!
//! ```
//! use oximedia_metadata::{Metadata, MetadataFormat, MetadataValue};
//!
//! // Create new metadata
//! let mut metadata = Metadata::new(MetadataFormat::Id3v2);
//!
//! // Insert fields
//! metadata.insert("TIT2".to_string(), MetadataValue::Text("My Song".to_string()));
//! metadata.insert("TPE1".to_string(), MetadataValue::Text("Artist Name".to_string()));
//!
//! // Access fields
//! if let Some(MetadataValue::Text(title)) = metadata.get("TIT2") {
//!     println!("Title: {}", title);
//! }
//! ```

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    dead_code,
    clippy::pedantic,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    unused_imports,
    clippy::unnecessary_cast,
    clippy::useless_conversion,
    clippy::string_lit_as_bytes
)]

use std::collections::HashMap;
use std::fmt;

pub mod apev2;
pub mod av1_metadata;
pub mod batch_editor;
pub mod bulk_metadata;
pub mod bulk_update;
pub mod c2pa;
pub mod chapter;
pub mod common;
pub mod converter;
pub mod csv_export;
pub mod embed;
pub mod embedding;
pub mod exif;
pub mod exif_parse;
pub mod field_validator;
pub mod geotag;
pub mod id3v2;
pub mod id3v2_utf8;
pub mod inherit;
pub mod iptc;
pub mod iptc_iim;
pub mod itunes;
pub mod linked_data;
pub mod lyrics;
pub mod matroska;
pub mod media_metadata;
pub mod metadata_diff;
pub mod metadata_export;
pub mod metadata_fingerprint;
pub mod metadata_history;
pub mod metadata_index;
pub mod metadata_json;
pub mod metadata_merge;
pub mod metadata_query;
pub mod metadata_sanitize;
pub mod metadata_stats;
pub mod metadata_streaming;
pub mod metadata_template;
pub mod metadata_validate;
pub mod mime_metadata;
pub mod musicbrainz;
pub mod opengraph;
pub mod opus_tags;
pub mod podcast;
pub mod provenance;
pub mod quicktime;
pub mod replaygain;
pub mod rights_metadata;
pub mod schema;
pub mod schema_org;
pub mod schema_registry;
pub mod search;
pub mod sidecar;
pub mod tag_normalize;
pub mod vorbis;
pub mod xmp;
pub mod xmp_document;

pub use common::CommonFields;
pub use converter::MetadataConverter;
pub use embed::MetadataEmbed;

/// Metadata format types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum MetadataFormat {
    /// ID3v2 tags (MP3)
    Id3v2,
    /// Vorbis Comments (Ogg, FLAC, Opus)
    VorbisComments,
    /// APEv2 tags
    Apev2,
    /// iTunes/MP4 metadata
    iTunes,
    /// XMP (Extensible Metadata Platform)
    Xmp,
    /// EXIF (Exchangeable Image File Format)
    Exif,
    /// IPTC (International Press Telecommunications Council)
    Iptc,
    /// QuickTime user data atoms
    QuickTime,
    /// Matroska tags
    Matroska,
}

impl fmt::Display for MetadataFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Id3v2 => write!(f, "ID3v2"),
            Self::VorbisComments => write!(f, "Vorbis Comments"),
            Self::Apev2 => write!(f, "APEv2"),
            Self::iTunes => write!(f, "iTunes"),
            Self::Xmp => write!(f, "XMP"),
            Self::Exif => write!(f, "EXIF"),
            Self::Iptc => write!(f, "IPTC"),
            Self::QuickTime => write!(f, "QuickTime"),
            Self::Matroska => write!(f, "Matroska"),
        }
    }
}

/// Metadata value types.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    /// Text string (UTF-8)
    Text(String),
    /// Multiple text strings
    TextList(Vec<String>),
    /// Binary data
    Binary(Vec<u8>),
    /// Integer value
    Integer(i64),
    /// Floating point value
    Float(f64),
    /// Picture/artwork
    Picture(Picture),
    /// Multiple pictures
    Pictures(Vec<Picture>),
    /// Boolean value
    Boolean(bool),
    /// Date/time (ISO 8601 format)
    DateTime(String),
}

impl MetadataValue {
    /// Get value as text if possible.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Get value as text list if possible.
    #[must_use]
    pub fn as_text_list(&self) -> Option<&[String]> {
        match self {
            Self::TextList(list) => Some(list),
            _ => None,
        }
    }

    /// Get value as binary if possible.
    #[must_use]
    pub fn as_binary(&self) -> Option<&[u8]> {
        match self {
            Self::Binary(data) => Some(data),
            _ => None,
        }
    }

    /// Get value as integer if possible.
    #[must_use]
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// Get value as float if possible.
    #[must_use]
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Get value as picture if possible.
    #[must_use]
    pub fn as_picture(&self) -> Option<&Picture> {
        match self {
            Self::Picture(pic) => Some(pic),
            _ => None,
        }
    }

    /// Get value as pictures if possible.
    #[must_use]
    pub fn as_pictures(&self) -> Option<&[Picture]> {
        match self {
            Self::Pictures(pics) => Some(pics),
            _ => None,
        }
    }

    /// Get value as boolean if possible.
    #[must_use]
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            Self::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// Get value as datetime if possible.
    #[must_use]
    pub fn as_datetime(&self) -> Option<&str> {
        match self {
            Self::DateTime(dt) => Some(dt),
            _ => None,
        }
    }
}

/// Picture/artwork metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Picture {
    /// MIME type (e.g., "image/jpeg", "image/png")
    pub mime_type: String,
    /// Picture type (e.g., front cover, back cover, etc.)
    pub picture_type: PictureType,
    /// Description
    pub description: String,
    /// Picture data
    pub data: Vec<u8>,
    /// Width in pixels (if known)
    pub width: Option<u32>,
    /// Height in pixels (if known)
    pub height: Option<u32>,
    /// Color depth in bits (if known)
    pub color_depth: Option<u32>,
}

impl Picture {
    /// Create a new picture with minimal information.
    #[must_use]
    pub fn new(mime_type: String, picture_type: PictureType, data: Vec<u8>) -> Self {
        Self {
            mime_type,
            picture_type,
            description: String::new(),
            data,
            width: None,
            height: None,
            color_depth: None,
        }
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    /// Set the dimensions.
    #[must_use]
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Set the color depth.
    #[must_use]
    pub fn with_color_depth(mut self, color_depth: u32) -> Self {
        self.color_depth = Some(color_depth);
        self
    }
}

/// Picture type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PictureType {
    /// Other
    Other,
    /// 32x32 pixels file icon
    FileIcon,
    /// Other file icon
    OtherFileIcon,
    /// Front cover
    FrontCover,
    /// Back cover
    BackCover,
    /// Leaflet page
    LeafletPage,
    /// Media (e.g., label side of CD)
    Media,
    /// Lead artist/lead performer/soloist
    LeadArtist,
    /// Artist/performer
    Artist,
    /// Conductor
    Conductor,
    /// Band/Orchestra
    Band,
    /// Composer
    Composer,
    /// Lyricist/text writer
    Lyricist,
    /// Recording Location
    RecordingLocation,
    /// During recording
    DuringRecording,
    /// During performance
    DuringPerformance,
    /// Movie/video screen capture
    VideoCapture,
    /// A bright colored fish
    BrightFish,
    /// Illustration
    Illustration,
    /// Band/artist logotype
    BandLogo,
    /// Publisher/Studio logotype
    PublisherLogo,
}

impl PictureType {
    /// Get the `ID3v2` picture type code.
    #[must_use]
    pub fn to_id3v2_code(self) -> u8 {
        match self {
            Self::Other => 0,
            Self::FileIcon => 1,
            Self::OtherFileIcon => 2,
            Self::FrontCover => 3,
            Self::BackCover => 4,
            Self::LeafletPage => 5,
            Self::Media => 6,
            Self::LeadArtist => 7,
            Self::Artist => 8,
            Self::Conductor => 9,
            Self::Band => 10,
            Self::Composer => 11,
            Self::Lyricist => 12,
            Self::RecordingLocation => 13,
            Self::DuringRecording => 14,
            Self::DuringPerformance => 15,
            Self::VideoCapture => 16,
            Self::BrightFish => 17,
            Self::Illustration => 18,
            Self::BandLogo => 19,
            Self::PublisherLogo => 20,
        }
    }

    /// Create from `ID3v2` picture type code.
    #[must_use]
    #[allow(clippy::match_same_arms)]
    pub fn from_id3v2_code(code: u8) -> Self {
        match code {
            0 => Self::Other,
            1 => Self::FileIcon,
            2 => Self::OtherFileIcon,
            3 => Self::FrontCover,
            4 => Self::BackCover,
            5 => Self::LeafletPage,
            6 => Self::Media,
            7 => Self::LeadArtist,
            8 => Self::Artist,
            9 => Self::Conductor,
            10 => Self::Band,
            11 => Self::Composer,
            12 => Self::Lyricist,
            13 => Self::RecordingLocation,
            14 => Self::DuringRecording,
            15 => Self::DuringPerformance,
            16 => Self::VideoCapture,
            17 => Self::BrightFish,
            18 => Self::Illustration,
            19 => Self::BandLogo,
            20 => Self::PublisherLogo,
            _ => Self::Other,
        }
    }
}

impl fmt::Display for PictureType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Other => write!(f, "Other"),
            Self::FileIcon => write!(f, "File Icon"),
            Self::OtherFileIcon => write!(f, "Other File Icon"),
            Self::FrontCover => write!(f, "Front Cover"),
            Self::BackCover => write!(f, "Back Cover"),
            Self::LeafletPage => write!(f, "Leaflet Page"),
            Self::Media => write!(f, "Media"),
            Self::LeadArtist => write!(f, "Lead Artist"),
            Self::Artist => write!(f, "Artist"),
            Self::Conductor => write!(f, "Conductor"),
            Self::Band => write!(f, "Band"),
            Self::Composer => write!(f, "Composer"),
            Self::Lyricist => write!(f, "Lyricist"),
            Self::RecordingLocation => write!(f, "Recording Location"),
            Self::DuringRecording => write!(f, "During Recording"),
            Self::DuringPerformance => write!(f, "During Performance"),
            Self::VideoCapture => write!(f, "Video Capture"),
            Self::BrightFish => write!(f, "Bright Fish"),
            Self::Illustration => write!(f, "Illustration"),
            Self::BandLogo => write!(f, "Band Logo"),
            Self::PublisherLogo => write!(f, "Publisher Logo"),
        }
    }
}

/// Generic metadata container.
#[derive(Debug, Clone)]
pub struct Metadata {
    /// Metadata format
    format: MetadataFormat,
    /// Tag fields (key-value pairs)
    fields: HashMap<String, MetadataValue>,
}

impl Metadata {
    /// Create a new metadata container.
    #[must_use]
    pub fn new(format: MetadataFormat) -> Self {
        Self {
            format,
            fields: HashMap::new(),
        }
    }

    /// Get the metadata format.
    #[must_use]
    pub fn format(&self) -> MetadataFormat {
        self.format
    }

    /// Insert a field.
    pub fn insert(&mut self, key: String, value: MetadataValue) {
        self.fields.insert(key, value);
    }

    /// Get a field value.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.fields.get(key)
    }

    /// Remove a field.
    pub fn remove(&mut self, key: &str) -> Option<MetadataValue> {
        self.fields.remove(key)
    }

    /// Check if a field exists.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.fields.contains_key(key)
    }

    /// Get all field keys.
    #[must_use]
    pub fn keys(&self) -> Vec<&String> {
        self.fields.keys().collect()
    }

    /// Get all fields.
    #[must_use]
    pub fn fields(&self) -> &HashMap<String, MetadataValue> {
        &self.fields
    }

    /// Get mutable access to all fields.
    pub fn fields_mut(&mut self) -> &mut HashMap<String, MetadataValue> {
        &mut self.fields
    }

    /// Clear all fields.
    pub fn clear(&mut self) {
        self.fields.clear();
    }

    /// Parse metadata from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    pub fn parse(data: &[u8], format: MetadataFormat) -> Result<Self, Error> {
        match format {
            MetadataFormat::Id3v2 => id3v2::parse(data),
            MetadataFormat::VorbisComments => vorbis::parse(data),
            MetadataFormat::Apev2 => apev2::parse(data),
            MetadataFormat::iTunes => itunes::parse(data),
            MetadataFormat::Xmp => xmp::parse(data),
            MetadataFormat::Exif => exif::parse(data),
            MetadataFormat::Iptc => iptc::parse(data),
            MetadataFormat::QuickTime => quicktime::parse(data),
            MetadataFormat::Matroska => matroska::parse(data),
        }
    }

    /// Write metadata to bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn write(&self) -> Result<Vec<u8>, Error> {
        match self.format {
            MetadataFormat::Id3v2 => id3v2::write(self),
            MetadataFormat::VorbisComments => vorbis::write(self),
            MetadataFormat::Apev2 => apev2::write(self),
            MetadataFormat::iTunes => itunes::write(self),
            MetadataFormat::Xmp => xmp::write(self),
            MetadataFormat::Exif => exif::write(self),
            MetadataFormat::Iptc => iptc::write(self),
            MetadataFormat::QuickTime => quicktime::write(self),
            MetadataFormat::Matroska => matroska::write(self),
        }
    }

    /// Convert metadata to another format.
    ///
    /// # Errors
    ///
    /// Returns an error if conversion fails.
    pub fn convert(&self, target: MetadataFormat) -> Result<Self, Error> {
        converter::convert(self, target)
    }

    /// Get common metadata fields.
    #[must_use]
    pub fn common(&self) -> CommonFields {
        CommonFields::from_metadata(self)
    }

    /// Set common metadata fields.
    pub fn set_common(&mut self, common: &CommonFields) {
        common.apply_to_metadata(self);
    }
}

/// Metadata trait for format-specific implementations.
pub trait MetadataParser {
    /// Parse metadata from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    fn parse(data: &[u8]) -> Result<Metadata, Error>;

    /// Write metadata to bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    fn write(metadata: &Metadata) -> Result<Vec<u8>, Error>;
}

/// Metadata error types.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Invalid format
    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Write error
    #[error("Write error: {0}")]
    WriteError(String),

    /// Unsupported feature
    #[error("Unsupported feature: {0}")]
    Unsupported(String),

    /// Encoding error
    #[error("Encoding error: {0}")]
    EncodingError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// XML error
    #[error("XML error: {0}")]
    Xml(String),

    /// Conversion error
    #[error("Conversion error: {0}")]
    ConversionError(String),
}

impl From<quick_xml::Error> for Error {
    fn from(err: quick_xml::Error) -> Self {
        Self::Xml(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_format_display() {
        assert_eq!(MetadataFormat::Id3v2.to_string(), "ID3v2");
        assert_eq!(
            MetadataFormat::VorbisComments.to_string(),
            "Vorbis Comments"
        );
        assert_eq!(MetadataFormat::Apev2.to_string(), "APEv2");
    }

    #[test]
    fn test_picture_type_id3v2_conversion() {
        assert_eq!(PictureType::FrontCover.to_id3v2_code(), 3);
        assert_eq!(PictureType::from_id3v2_code(3), PictureType::FrontCover);
        assert_eq!(PictureType::BackCover.to_id3v2_code(), 4);
        assert_eq!(PictureType::from_id3v2_code(4), PictureType::BackCover);
    }

    #[test]
    fn test_metadata_value_accessors() {
        let text = MetadataValue::Text("test".to_string());
        assert_eq!(text.as_text(), Some("test"));
        assert_eq!(text.as_integer(), None);

        let int = MetadataValue::Integer(42);
        assert_eq!(int.as_integer(), Some(42));
        assert_eq!(int.as_text(), None);

        let float = MetadataValue::Float(3.14);
        assert_eq!(float.as_float(), Some(3.14));
        assert_eq!(float.as_integer(), None);

        let bool = MetadataValue::Boolean(true);
        assert_eq!(bool.as_boolean(), Some(true));
        assert_eq!(bool.as_text(), None);
    }

    #[test]
    fn test_metadata_new() {
        let metadata = Metadata::new(MetadataFormat::Id3v2);
        assert_eq!(metadata.format(), MetadataFormat::Id3v2);
        assert_eq!(metadata.fields().len(), 0);
    }

    #[test]
    fn test_metadata_insert_get() {
        let mut metadata = Metadata::new(MetadataFormat::Id3v2);
        metadata.insert("title".to_string(), MetadataValue::Text("Test".to_string()));

        assert!(metadata.contains("title"));
        assert_eq!(
            metadata.get("title").and_then(MetadataValue::as_text),
            Some("Test")
        );
        assert!(!metadata.contains("artist"));
    }

    #[test]
    fn test_metadata_remove() {
        let mut metadata = Metadata::new(MetadataFormat::Id3v2);
        metadata.insert("title".to_string(), MetadataValue::Text("Test".to_string()));

        assert!(metadata.contains("title"));
        let removed = metadata.remove("title");
        assert!(removed.is_some());
        assert!(!metadata.contains("title"));
    }

    #[test]
    fn test_metadata_clear() {
        let mut metadata = Metadata::new(MetadataFormat::Id3v2);
        metadata.insert("title".to_string(), MetadataValue::Text("Test".to_string()));
        metadata.insert(
            "artist".to_string(),
            MetadataValue::Text("Artist".to_string()),
        );

        assert_eq!(metadata.fields().len(), 2);
        metadata.clear();
        assert_eq!(metadata.fields().len(), 0);
    }

    #[test]
    fn test_picture_builder() {
        let picture = Picture::new(
            "image/jpeg".to_string(),
            PictureType::FrontCover,
            vec![0xFF, 0xD8, 0xFF, 0xE0],
        )
        .with_description("Album cover".to_string())
        .with_dimensions(1000, 1000)
        .with_color_depth(24);

        assert_eq!(picture.mime_type, "image/jpeg");
        assert_eq!(picture.picture_type, PictureType::FrontCover);
        assert_eq!(picture.description, "Album cover");
        assert_eq!(picture.width, Some(1000));
        assert_eq!(picture.height, Some(1000));
        assert_eq!(picture.color_depth, Some(24));
    }
}
