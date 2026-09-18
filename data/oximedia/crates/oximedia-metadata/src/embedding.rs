//! Metadata embedding specifications for media files.
//!
//! This module provides types for describing how metadata fields
//! should be embedded into various media container formats.

#![allow(dead_code)]

/// The target container or tag structure to embed metadata into.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbedTarget {
    /// XMP packet (sidecar or embedded).
    XmpPacket,
    /// ID3 tag (used in MP3 files).
    Id3Tag,
    /// EXIF header (used in JPEG/TIFF images).
    ExifHeader,
    /// QuickTime atom (used in MOV/MP4 files).
    QuicktimeAtom,
    /// Matroska/WebM tag.
    MkvTag,
}

impl EmbedTarget {
    /// Returns true if this target supports structured/hierarchical data.
    pub fn supports_structured(&self) -> bool {
        matches!(self, Self::XmpPacket | Self::QuicktimeAtom)
    }

    /// Returns the maximum payload size in bytes for this target.
    pub fn max_size_bytes(&self) -> usize {
        match self {
            Self::XmpPacket => 65_535,
            Self::Id3Tag => 16_777_216, // 16 MB
            Self::ExifHeader => 65_535,
            Self::QuicktimeAtom => 4_294_967_295, // 4 GB
            Self::MkvTag => 16_777_216,
        }
    }
}

/// The kind of embed operation to perform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbedOperation {
    /// Write a new field (creates if not present).
    Write,
    /// Update an existing field (no-op if absent).
    Update,
    /// Remove a field.
    Remove,
}

impl EmbedOperation {
    /// Returns true if this operation is potentially destructive (i.e., removes data).
    pub fn is_destructive(&self) -> bool {
        matches!(self, Self::Remove)
    }
}

/// A specification describing a single metadata embed action.
#[derive(Debug, Clone)]
pub struct EmbedSpec {
    /// The target format/container.
    pub target: EmbedTarget,
    /// The operation to perform.
    pub operation: EmbedOperation,
    /// Name of the metadata field.
    pub field_name: String,
    /// Value to write (ignored for Remove operations).
    pub value: String,
}

impl EmbedSpec {
    /// Create a new embed specification.
    pub fn new(
        target: EmbedTarget,
        operation: EmbedOperation,
        field_name: String,
        value: String,
    ) -> Self {
        Self {
            target,
            operation,
            field_name,
            value,
        }
    }

    /// Returns true if the spec is valid.
    ///
    /// A spec is valid if the field name is non-empty, and if the operation
    /// is `Write` or `Update`, the value must also be non-empty.
    pub fn is_valid(&self) -> bool {
        if self.field_name.is_empty() {
            return false;
        }
        match self.operation {
            EmbedOperation::Write | EmbedOperation::Update => !self.value.is_empty(),
            EmbedOperation::Remove => true,
        }
    }
}

/// A collection of embed specifications for a media file.
#[derive(Debug, Clone, Default)]
pub struct MetadataEmbedder {
    /// All embed specifications.
    pub specs: Vec<EmbedSpec>,
}

impl MetadataEmbedder {
    /// Create a new embedder with no specs.
    pub fn new() -> Self {
        Self { specs: Vec::new() }
    }

    /// Add a spec to the embedder.
    pub fn add_spec(&mut self, spec: EmbedSpec) {
        self.specs.push(spec);
    }

    /// Returns all specs targeting a specific embed target.
    pub fn specs_for_target(&self, t: &EmbedTarget) -> Vec<&EmbedSpec> {
        self.specs.iter().filter(|s| &s.target == t).collect()
    }

    /// Returns true if all specs are valid.
    pub fn validate_all(&self) -> bool {
        self.specs.iter().all(EmbedSpec::is_valid)
    }

    /// Returns the number of specs in this embedder.
    pub fn spec_count(&self) -> usize {
        self.specs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spec(target: EmbedTarget, op: EmbedOperation, field: &str, value: &str) -> EmbedSpec {
        EmbedSpec::new(target, op, field.to_string(), value.to_string())
    }

    #[test]
    fn test_embed_target_supports_structured() {
        assert!(EmbedTarget::XmpPacket.supports_structured());
        assert!(EmbedTarget::QuicktimeAtom.supports_structured());
        assert!(!EmbedTarget::Id3Tag.supports_structured());
        assert!(!EmbedTarget::ExifHeader.supports_structured());
        assert!(!EmbedTarget::MkvTag.supports_structured());
    }

    #[test]
    fn test_embed_target_max_size() {
        assert_eq!(EmbedTarget::XmpPacket.max_size_bytes(), 65_535);
        assert_eq!(EmbedTarget::Id3Tag.max_size_bytes(), 16_777_216);
        assert_eq!(EmbedTarget::ExifHeader.max_size_bytes(), 65_535);
        assert_eq!(EmbedTarget::QuicktimeAtom.max_size_bytes(), 4_294_967_295);
        assert_eq!(EmbedTarget::MkvTag.max_size_bytes(), 16_777_216);
    }

    #[test]
    fn test_embed_operation_is_destructive() {
        assert!(!EmbedOperation::Write.is_destructive());
        assert!(!EmbedOperation::Update.is_destructive());
        assert!(EmbedOperation::Remove.is_destructive());
    }

    #[test]
    fn test_embed_spec_valid_write() {
        let spec = make_spec(
            EmbedTarget::Id3Tag,
            EmbedOperation::Write,
            "TIT2",
            "My Song",
        );
        assert!(spec.is_valid());
    }

    #[test]
    fn test_embed_spec_invalid_empty_field_name() {
        let spec = make_spec(EmbedTarget::Id3Tag, EmbedOperation::Write, "", "value");
        assert!(!spec.is_valid());
    }

    #[test]
    fn test_embed_spec_invalid_empty_value_for_write() {
        let spec = make_spec(
            EmbedTarget::XmpPacket,
            EmbedOperation::Write,
            "dc:title",
            "",
        );
        assert!(!spec.is_valid());
    }

    #[test]
    fn test_embed_spec_valid_remove_with_empty_value() {
        let spec = make_spec(EmbedTarget::MkvTag, EmbedOperation::Remove, "TITLE", "");
        assert!(spec.is_valid());
    }

    #[test]
    fn test_embedder_empty() {
        let embedder = MetadataEmbedder::new();
        assert_eq!(embedder.spec_count(), 0);
        assert!(embedder.validate_all());
    }

    #[test]
    fn test_embedder_add_spec() {
        let mut embedder = MetadataEmbedder::new();
        embedder.add_spec(make_spec(
            EmbedTarget::Id3Tag,
            EmbedOperation::Write,
            "TIT2",
            "Song",
        ));
        assert_eq!(embedder.spec_count(), 1);
    }

    #[test]
    fn test_embedder_specs_for_target() {
        let mut embedder = MetadataEmbedder::new();
        embedder.add_spec(make_spec(
            EmbedTarget::Id3Tag,
            EmbedOperation::Write,
            "TIT2",
            "Song",
        ));
        embedder.add_spec(make_spec(
            EmbedTarget::Id3Tag,
            EmbedOperation::Write,
            "TPE1",
            "Artist",
        ));
        embedder.add_spec(make_spec(
            EmbedTarget::XmpPacket,
            EmbedOperation::Write,
            "dc:title",
            "T",
        ));

        let id3_specs = embedder.specs_for_target(&EmbedTarget::Id3Tag);
        assert_eq!(id3_specs.len(), 2);

        let xmp_specs = embedder.specs_for_target(&EmbedTarget::XmpPacket);
        assert_eq!(xmp_specs.len(), 1);

        let mkv_specs = embedder.specs_for_target(&EmbedTarget::MkvTag);
        assert!(mkv_specs.is_empty());
    }

    #[test]
    fn test_embedder_validate_all_passes() {
        let mut embedder = MetadataEmbedder::new();
        embedder.add_spec(make_spec(
            EmbedTarget::Id3Tag,
            EmbedOperation::Write,
            "TIT2",
            "Song",
        ));
        embedder.add_spec(make_spec(
            EmbedTarget::MkvTag,
            EmbedOperation::Remove,
            "TAG",
            "",
        ));
        assert!(embedder.validate_all());
    }

    #[test]
    fn test_embedder_validate_all_fails() {
        let mut embedder = MetadataEmbedder::new();
        embedder.add_spec(make_spec(
            EmbedTarget::Id3Tag,
            EmbedOperation::Write,
            "TIT2",
            "Song",
        ));
        embedder.add_spec(make_spec(
            EmbedTarget::Id3Tag,
            EmbedOperation::Write,
            "",
            "bad",
        ));
        assert!(!embedder.validate_all());
    }

    #[test]
    fn test_embedder_spec_count() {
        let mut embedder = MetadataEmbedder::new();
        for i in 0..5 {
            embedder.add_spec(make_spec(
                EmbedTarget::ExifHeader,
                EmbedOperation::Update,
                &format!("field_{i}"),
                "value",
            ));
        }
        assert_eq!(embedder.spec_count(), 5);
    }
}
