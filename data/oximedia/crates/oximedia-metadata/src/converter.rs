//! Cross-format metadata conversion utilities.
//!
//! This module provides functionality to convert metadata between different formats,
//! mapping fields appropriately and preserving as much information as possible.

use crate::{CommonFields, Error, Metadata, MetadataFormat};

/// Metadata converter for cross-format conversion.
pub struct MetadataConverter;

impl MetadataConverter {
    /// Convert metadata from one format to another.
    ///
    /// # Errors
    ///
    /// Returns an error if conversion fails.
    pub fn convert(source: &Metadata, target_format: MetadataFormat) -> Result<Metadata, Error> {
        convert(source, target_format)
    }
}

/// Convert metadata from one format to another.
///
/// This function uses common fields as an intermediate representation to convert
/// metadata between different formats.
///
/// # Errors
///
/// Returns an error if conversion fails.
pub fn convert(source: &Metadata, target_format: MetadataFormat) -> Result<Metadata, Error> {
    // Extract common fields from source
    let common = CommonFields::from_metadata(source);

    // Create target metadata
    let mut target = Metadata::new(target_format);

    // Apply common fields to target
    common.apply_to_metadata(&mut target);

    // Copy format-specific fields that can be mapped
    for (key, value) in source.fields() {
        if !target.contains(key) {
            // Try to map the field to target format
            if let Some(mapped_key) = map_field_name(key, source.format(), target_format) {
                target.insert(mapped_key, value.clone());
            }
        }
    }

    Ok(target)
}

/// Map a field name from one format to another.
fn map_field_name(
    field: &str,
    source_format: MetadataFormat,
    target_format: MetadataFormat,
) -> Option<String> {
    // Define field mappings between formats
    match (source_format, target_format) {
        // ID3v2 -> Vorbis Comments
        (MetadataFormat::Id3v2, MetadataFormat::VorbisComments) => match field {
            "TIT2" => Some("TITLE".to_string()),
            "TPE1" => Some("ARTIST".to_string()),
            "TALB" => Some("ALBUM".to_string()),
            "TPE2" => Some("ALBUMARTIST".to_string()),
            "TCON" => Some("GENRE".to_string()),
            "TDRC" | "TYER" => Some("DATE".to_string()),
            "COMM" => Some("COMMENT".to_string()),
            "TCOM" => Some("COMPOSER".to_string()),
            "TPE3" => Some("CONDUCTOR".to_string()),
            "TEXT" => Some("LYRICIST".to_string()),
            "TCOP" => Some("COPYRIGHT".to_string()),
            "TPUB" => Some("PUBLISHER".to_string()),
            "TSRC" => Some("ISRC".to_string()),
            "TSSE" => Some("ENCODER".to_string()),
            "TLAN" => Some("LANGUAGE".to_string()),
            _ => None,
        },

        // Vorbis Comments -> ID3v2
        (MetadataFormat::VorbisComments, MetadataFormat::Id3v2) => match field {
            "TITLE" => Some("TIT2".to_string()),
            "ARTIST" => Some("TPE1".to_string()),
            "ALBUM" => Some("TALB".to_string()),
            "ALBUMARTIST" => Some("TPE2".to_string()),
            "GENRE" => Some("TCON".to_string()),
            "DATE" => Some("TDRC".to_string()),
            "COMMENT" => Some("COMM".to_string()),
            "COMPOSER" => Some("TCOM".to_string()),
            "CONDUCTOR" => Some("TPE3".to_string()),
            "LYRICIST" => Some("TEXT".to_string()),
            "COPYRIGHT" => Some("TCOP".to_string()),
            "PUBLISHER" => Some("TPUB".to_string()),
            "ISRC" => Some("TSRC".to_string()),
            "ENCODER" => Some("TSSE".to_string()),
            "LANGUAGE" => Some("TLAN".to_string()),
            _ => None,
        },

        // ID3v2 -> iTunes
        (MetadataFormat::Id3v2, MetadataFormat::iTunes) => match field {
            "TIT2" => Some("©nam".to_string()),
            "TPE1" => Some("©ART".to_string()),
            "TALB" => Some("©alb".to_string()),
            "TPE2" => Some("aART".to_string()),
            "TCON" => Some("©gen".to_string()),
            "TDRC" | "TYER" => Some("©day".to_string()),
            "COMM" => Some("©cmt".to_string()),
            "TCOM" => Some("©wrt".to_string()),
            "TCOP" => Some("cprt".to_string()),
            "TSSE" => Some("©too".to_string()),
            _ => None,
        },

        // iTunes -> ID3v2
        (MetadataFormat::iTunes, MetadataFormat::Id3v2) => match field {
            "©nam" => Some("TIT2".to_string()),
            "©ART" => Some("TPE1".to_string()),
            "©alb" => Some("TALB".to_string()),
            "aART" => Some("TPE2".to_string()),
            "©gen" => Some("TCON".to_string()),
            "©day" => Some("TDRC".to_string()),
            "©cmt" => Some("COMM".to_string()),
            "©wrt" => Some("TCOM".to_string()),
            "cprt" => Some("TCOP".to_string()),
            "©too" => Some("TSSE".to_string()),
            _ => None,
        },

        // Vorbis Comments -> iTunes
        (MetadataFormat::VorbisComments, MetadataFormat::iTunes) => match field {
            "TITLE" => Some("©nam".to_string()),
            "ARTIST" => Some("©ART".to_string()),
            "ALBUM" => Some("©alb".to_string()),
            "ALBUMARTIST" => Some("aART".to_string()),
            "GENRE" => Some("©gen".to_string()),
            "DATE" => Some("©day".to_string()),
            "COMMENT" => Some("©cmt".to_string()),
            "COMPOSER" => Some("©wrt".to_string()),
            "COPYRIGHT" => Some("cprt".to_string()),
            "ENCODER" => Some("©too".to_string()),
            _ => None,
        },

        // iTunes -> Vorbis Comments
        (MetadataFormat::iTunes, MetadataFormat::VorbisComments) => match field {
            "©nam" => Some("TITLE".to_string()),
            "©ART" => Some("ARTIST".to_string()),
            "©alb" => Some("ALBUM".to_string()),
            "aART" => Some("ALBUMARTIST".to_string()),
            "©gen" => Some("GENRE".to_string()),
            "©day" => Some("DATE".to_string()),
            "©cmt" => Some("COMMENT".to_string()),
            "©wrt" => Some("COMPOSER".to_string()),
            "cprt" => Some("COPYRIGHT".to_string()),
            "©too" => Some("ENCODER".to_string()),
            _ => None,
        },

        // Default: no mapping
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetadataValue;

    #[test]
    fn test_convert_id3v2_to_vorbis() {
        let mut source = Metadata::new(MetadataFormat::Id3v2);
        source.insert(
            "TIT2".to_string(),
            MetadataValue::Text("Test Title".to_string()),
        );
        source.insert(
            "TPE1".to_string(),
            MetadataValue::Text("Test Artist".to_string()),
        );
        source.insert(
            "TALB".to_string(),
            MetadataValue::Text("Test Album".to_string()),
        );

        let target = convert(&source, MetadataFormat::VorbisComments).expect("Conversion failed");

        assert_eq!(target.format(), MetadataFormat::VorbisComments);
        assert_eq!(
            target.get("TITLE").and_then(|v| v.as_text()),
            Some("Test Title")
        );
        assert_eq!(
            target.get("ARTIST").and_then(|v| v.as_text()),
            Some("Test Artist")
        );
        assert_eq!(
            target.get("ALBUM").and_then(|v| v.as_text()),
            Some("Test Album")
        );
    }

    #[test]
    fn test_convert_vorbis_to_id3v2() {
        let mut source = Metadata::new(MetadataFormat::VorbisComments);
        source.insert(
            "TITLE".to_string(),
            MetadataValue::Text("Test Title".to_string()),
        );
        source.insert(
            "ARTIST".to_string(),
            MetadataValue::Text("Test Artist".to_string()),
        );
        source.insert(
            "ALBUM".to_string(),
            MetadataValue::Text("Test Album".to_string()),
        );

        let target = convert(&source, MetadataFormat::Id3v2).expect("Conversion failed");

        assert_eq!(target.format(), MetadataFormat::Id3v2);
        assert_eq!(
            target.get("TIT2").and_then(|v| v.as_text()),
            Some("Test Title")
        );
        assert_eq!(
            target.get("TPE1").and_then(|v| v.as_text()),
            Some("Test Artist")
        );
        assert_eq!(
            target.get("TALB").and_then(|v| v.as_text()),
            Some("Test Album")
        );
    }

    #[test]
    fn test_convert_id3v2_to_itunes() {
        let mut source = Metadata::new(MetadataFormat::Id3v2);
        source.insert(
            "TIT2".to_string(),
            MetadataValue::Text("Test Title".to_string()),
        );
        source.insert(
            "TPE1".to_string(),
            MetadataValue::Text("Test Artist".to_string()),
        );

        let target = convert(&source, MetadataFormat::iTunes).expect("Conversion failed");

        assert_eq!(target.format(), MetadataFormat::iTunes);
        assert_eq!(
            target.get("©nam").and_then(|v| v.as_text()),
            Some("Test Title")
        );
        assert_eq!(
            target.get("©ART").and_then(|v| v.as_text()),
            Some("Test Artist")
        );
    }

    #[test]
    fn test_map_field_name() {
        assert_eq!(
            map_field_name(
                "TIT2",
                MetadataFormat::Id3v2,
                MetadataFormat::VorbisComments
            ),
            Some("TITLE".to_string())
        );
        assert_eq!(
            map_field_name(
                "TITLE",
                MetadataFormat::VorbisComments,
                MetadataFormat::Id3v2
            ),
            Some("TIT2".to_string())
        );
        assert_eq!(
            map_field_name(
                "UNKNOWN",
                MetadataFormat::Id3v2,
                MetadataFormat::VorbisComments
            ),
            None
        );
    }
}
