//! Media format family classification and format information registry.
//!
//! Provides structured metadata about media formats including professional
//! classification, edit-readiness, and a queryable format registry.

#![allow(dead_code)]

use std::collections::HashMap;

/// Family/category of a media format.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MediaFormatFamily {
    /// Broadcast/professional linear formats (MXF, GXF, etc.)
    Broadcast,
    /// Streaming/delivery formats (MP4, DASH, HLS segments)
    Streaming,
    /// Editing/intermediate formats (ProRes, DNxHD, etc.)
    Editing,
    /// Archive/lossless formats (FLAC, DPX, OpenEXR)
    Archive,
    /// Consumer formats (AVI, WMV, MOV consumer)
    Consumer,
    /// Raw camera formats (BRAW, R3D, ARRIRAW)
    RawCamera,
}

impl MediaFormatFamily {
    /// Returns `true` if this family is considered professional/broadcast grade.
    pub fn is_professional(&self) -> bool {
        matches!(
            self,
            MediaFormatFamily::Broadcast
                | MediaFormatFamily::Editing
                | MediaFormatFamily::Archive
                | MediaFormatFamily::RawCamera
        )
    }

    /// Human-readable label for the family.
    pub fn label(&self) -> &'static str {
        match self {
            MediaFormatFamily::Broadcast => "Broadcast",
            MediaFormatFamily::Streaming => "Streaming",
            MediaFormatFamily::Editing => "Editing",
            MediaFormatFamily::Archive => "Archive",
            MediaFormatFamily::Consumer => "Consumer",
            MediaFormatFamily::RawCamera => "Raw Camera",
        }
    }
}

/// Detailed information about a specific media format.
#[derive(Debug, Clone)]
pub struct MediaFormatInfo {
    /// Short format identifier, e.g. `"mxf"`.
    pub id: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Format family classification.
    pub family: MediaFormatFamily,
    /// Common file extensions (without leading dot).
    pub extensions: Vec<String>,
    /// Whether the format supports random-access editing without transcoding.
    pub edit_ready: bool,
    /// Maximum supported bit depth (0 = unlimited / format-dependent).
    pub max_bit_depth: u8,
    /// Whether the format natively carries timecode.
    pub supports_timecode: bool,
}

impl MediaFormatInfo {
    /// Create a new `MediaFormatInfo`.
    pub fn new(
        id: impl Into<String>,
        display_name: impl Into<String>,
        family: MediaFormatFamily,
        extensions: Vec<&str>,
        edit_ready: bool,
        max_bit_depth: u8,
        supports_timecode: bool,
    ) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            family,
            extensions: extensions.into_iter().map(|s| s.to_string()).collect(),
            edit_ready,
            max_bit_depth,
            supports_timecode,
        }
    }

    /// Returns `true` if the format can be used for editing without prior transcoding.
    pub fn is_edit_ready(&self) -> bool {
        self.edit_ready
    }

    /// Returns `true` if the format belongs to a professional family.
    pub fn is_professional(&self) -> bool {
        self.family.is_professional()
    }

    /// Returns `true` if `extension` (without dot, case-insensitive) matches this format.
    pub fn matches_extension(&self, extension: &str) -> bool {
        let ext = extension.to_lowercase();
        self.extensions.iter().any(|e| e.to_lowercase() == ext)
    }
}

/// Registry mapping format IDs to their `MediaFormatInfo`.
#[derive(Debug, Default)]
pub struct FormatInfoRegistry {
    formats: HashMap<String, MediaFormatInfo>,
}

impl FormatInfoRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry pre-populated with common broadcast/post formats.
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();

        reg.register(MediaFormatInfo::new(
            "mxf",
            "Material Exchange Format",
            MediaFormatFamily::Broadcast,
            vec!["mxf"],
            true,
            12,
            true,
        ));
        reg.register(MediaFormatInfo::new(
            "prores",
            "Apple ProRes (MOV)",
            MediaFormatFamily::Editing,
            vec!["mov"],
            true,
            12,
            true,
        ));
        reg.register(MediaFormatInfo::new(
            "dnxhd",
            "Avid DNxHD/DNxHR (MXF/MOV)",
            MediaFormatFamily::Editing,
            vec!["mxf", "mov"],
            true,
            12,
            true,
        ));
        reg.register(MediaFormatInfo::new(
            "mp4",
            "MPEG-4 Part 14",
            MediaFormatFamily::Streaming,
            vec!["mp4", "m4v"],
            false,
            10,
            false,
        ));
        reg.register(MediaFormatInfo::new(
            "dpx",
            "Digital Picture Exchange",
            MediaFormatFamily::Archive,
            vec!["dpx"],
            false,
            16,
            false,
        ));
        reg.register(MediaFormatInfo::new(
            "r3d",
            "RED RAW",
            MediaFormatFamily::RawCamera,
            vec!["r3d"],
            false,
            16,
            true,
        ));
        reg.register(MediaFormatInfo::new(
            "avi",
            "Audio Video Interleave",
            MediaFormatFamily::Consumer,
            vec!["avi"],
            false,
            8,
            false,
        ));

        reg
    }

    /// Register a format, replacing any existing entry with the same ID.
    pub fn register(&mut self, info: MediaFormatInfo) {
        self.formats.insert(info.id.clone(), info);
    }

    /// Look up a format by its ID.
    pub fn find(&self, id: &str) -> Option<&MediaFormatInfo> {
        self.formats.get(id)
    }

    /// Look up a format by file extension (case-insensitive).
    pub fn find_by_extension(&self, ext: &str) -> Vec<&MediaFormatInfo> {
        self.formats
            .values()
            .filter(|f| f.matches_extension(ext))
            .collect()
    }

    /// Total number of registered formats.
    pub fn count(&self) -> usize {
        self.formats.len()
    }

    /// Return all formats belonging to the given family.
    pub fn by_family(&self, family: &MediaFormatFamily) -> Vec<&MediaFormatInfo> {
        self.formats
            .values()
            .filter(|f| &f.family == family)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_info(id: &str, family: MediaFormatFamily, edit_ready: bool) -> MediaFormatInfo {
        MediaFormatInfo::new(id, id, family, vec!["ext"], edit_ready, 8, false)
    }

    #[test]
    fn test_format_family_is_professional_broadcast() {
        assert!(MediaFormatFamily::Broadcast.is_professional());
    }

    #[test]
    fn test_format_family_is_professional_editing() {
        assert!(MediaFormatFamily::Editing.is_professional());
    }

    #[test]
    fn test_format_family_is_professional_archive() {
        assert!(MediaFormatFamily::Archive.is_professional());
    }

    #[test]
    fn test_format_family_is_professional_raw() {
        assert!(MediaFormatFamily::RawCamera.is_professional());
    }

    #[test]
    fn test_format_family_not_professional_streaming() {
        assert!(!MediaFormatFamily::Streaming.is_professional());
    }

    #[test]
    fn test_format_family_not_professional_consumer() {
        assert!(!MediaFormatFamily::Consumer.is_professional());
    }

    #[test]
    fn test_format_family_label() {
        assert_eq!(MediaFormatFamily::Broadcast.label(), "Broadcast");
        assert_eq!(MediaFormatFamily::RawCamera.label(), "Raw Camera");
    }

    #[test]
    fn test_media_format_info_is_edit_ready_true() {
        let info = make_info("prores", MediaFormatFamily::Editing, true);
        assert!(info.is_edit_ready());
    }

    #[test]
    fn test_media_format_info_is_edit_ready_false() {
        let info = make_info("mp4", MediaFormatFamily::Streaming, false);
        assert!(!info.is_edit_ready());
    }

    #[test]
    fn test_media_format_info_is_professional() {
        let info = make_info("mxf", MediaFormatFamily::Broadcast, true);
        assert!(info.is_professional());
    }

    #[test]
    fn test_media_format_info_matches_extension_case_insensitive() {
        let info = MediaFormatInfo::new(
            "mxf",
            "MXF",
            MediaFormatFamily::Broadcast,
            vec!["mxf"],
            true,
            12,
            true,
        );
        assert!(info.matches_extension("MXF"));
        assert!(info.matches_extension("mxf"));
        assert!(!info.matches_extension("mov"));
    }

    #[test]
    fn test_registry_register_and_find() {
        let mut reg = FormatInfoRegistry::new();
        reg.register(make_info("test", MediaFormatFamily::Streaming, false));
        assert!(reg.find("test").is_some());
    }

    #[test]
    fn test_registry_find_missing() {
        let reg = FormatInfoRegistry::new();
        assert!(reg.find("nonexistent").is_none());
    }

    #[test]
    fn test_registry_count() {
        let mut reg = FormatInfoRegistry::new();
        reg.register(make_info("a", MediaFormatFamily::Consumer, false));
        reg.register(make_info("b", MediaFormatFamily::Consumer, false));
        assert_eq!(reg.count(), 2);
    }

    #[test]
    fn test_registry_with_defaults_count() {
        let reg = FormatInfoRegistry::with_defaults();
        assert!(reg.count() >= 7);
    }

    #[test]
    fn test_registry_by_family() {
        let reg = FormatInfoRegistry::with_defaults();
        let editing = reg.by_family(&MediaFormatFamily::Editing);
        assert!(!editing.is_empty());
        for f in editing {
            assert!(f.is_professional());
        }
    }

    #[test]
    fn test_registry_find_by_extension() {
        let reg = FormatInfoRegistry::with_defaults();
        let results = reg.find_by_extension("mxf");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_registry_register_overwrites() {
        let mut reg = FormatInfoRegistry::new();
        reg.register(make_info("dup", MediaFormatFamily::Streaming, false));
        reg.register(make_info("dup", MediaFormatFamily::Broadcast, true));
        assert_eq!(reg.count(), 1);
        let found = reg.find("dup").expect("should succeed in test");
        assert!(found.is_edit_ready());
    }

    #[test]
    fn test_defaults_mxf_timecode() {
        let reg = FormatInfoRegistry::with_defaults();
        let mxf = reg.find("mxf").expect("should succeed in test");
        assert!(mxf.supports_timecode);
    }

    #[test]
    fn test_defaults_mp4_not_edit_ready() {
        let reg = FormatInfoRegistry::with_defaults();
        let mp4 = reg.find("mp4").expect("should succeed in test");
        assert!(!mp4.is_edit_ready());
    }
}
