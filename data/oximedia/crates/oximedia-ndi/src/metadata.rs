//! NDI metadata XML handling.
//!
//! This module provides utilities for building, parsing, and converting NDI metadata
//! frames which carry XML payloads including tally state and PTZ data.

#![allow(dead_code)]

use std::collections::HashMap;

/// An NDI metadata frame carrying an XML payload.
#[derive(Debug, Clone)]
pub struct NdiMetadataFrame {
    /// Timestamp in milliseconds since Unix epoch.
    pub timestamp_ms: u64,
    /// XML data payload.
    pub data: String,
}

impl NdiMetadataFrame {
    /// Creates a new `NdiMetadataFrame` with the given timestamp and XML data.
    pub fn new(timestamp_ms: u64, data: String) -> Self {
        Self { timestamp_ms, data }
    }
}

/// Builder for constructing `NdiMetadataFrame` XML payloads.
///
/// # Example
///
/// ```
/// use oximedia_ndi::metadata::NdiMetadataBuilder;
///
/// let frame = NdiMetadataBuilder::new(1000)
///     .add_field("camera", "CAM1")
///     .add_field("scene", "intro")
///     .build();
///
/// assert!(frame.data.contains("camera"));
/// ```
#[derive(Debug)]
pub struct NdiMetadataBuilder {
    timestamp_ms: u64,
    fields: Vec<(String, String)>,
}

impl NdiMetadataBuilder {
    /// Creates a new builder with the given timestamp.
    pub fn new(timestamp_ms: u64) -> Self {
        Self {
            timestamp_ms,
            fields: Vec::new(),
        }
    }

    /// Adds a key-value field to the metadata.
    pub fn add_field(&mut self, key: &str, value: &str) -> &mut Self {
        self.fields.push((key.to_string(), value.to_string()));
        self
    }

    /// Builds the `NdiMetadataFrame` with the accumulated fields.
    ///
    /// The generated XML format is:
    /// `<metadata><field key="key">value</field></metadata>`
    pub fn build(&self) -> NdiMetadataFrame {
        let mut xml = String::from("<metadata>");
        for (key, value) in &self.fields {
            xml.push_str(&format!(
                "<field key=\"{}\">{}</field>",
                escape_xml(key),
                escape_xml(value)
            ));
        }
        xml.push_str("</metadata>");

        NdiMetadataFrame {
            timestamp_ms: self.timestamp_ms,
            data: xml,
        }
    }
}

/// Escapes XML special characters in a string.
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Parser for extracting key-value pairs from `NdiMetadataFrame` XML payloads.
pub struct NdiMetadataParser;

impl NdiMetadataParser {
    /// Parses a `NdiMetadataFrame` and extracts key-value field pairs.
    ///
    /// Handles XML of the form:
    /// `<metadata><field key="k">v</field></metadata>`
    ///
    /// Returns a `HashMap` of field keys to their values.
    pub fn parse(frame: &NdiMetadataFrame) -> HashMap<String, String> {
        let mut result = HashMap::new();
        let xml = &frame.data;

        // Simple extraction: find all <field key="...">...</field> occurrences
        let mut search_from = 0;
        while let Some(start) = xml[search_from..].find("<field key=\"") {
            let abs_start = search_from + start;
            let key_start = abs_start + "<field key=\"".len();

            // Find closing quote of key attribute
            if let Some(key_end_rel) = xml[key_start..].find('"') {
                let key_end = key_start + key_end_rel;
                let key = unescape_xml(&xml[key_start..key_end]);

                // Find the closing '>' of the opening tag
                if let Some(tag_close_rel) = xml[key_end..].find('>') {
                    let value_start = key_end + tag_close_rel + 1;

                    // Find </field>
                    if let Some(value_end_rel) = xml[value_start..].find("</field>") {
                        let value_end = value_start + value_end_rel;
                        let value = unescape_xml(&xml[value_start..value_end]);
                        result.insert(key, value);
                        search_from = value_end + "</field>".len();
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        result
    }
}

/// Unescapes XML special characters.
fn unescape_xml(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

/// Tally metadata for NDI tally light state transmission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TallyMetadata {
    /// Whether this source is on program (red tally).
    pub program: bool,
    /// Whether this source is on preview (green tally).
    pub preview: bool,
}

impl TallyMetadata {
    /// Creates a new `TallyMetadata` with the given program and preview states.
    pub fn new(program: bool, preview: bool) -> Self {
        Self { program, preview }
    }

    /// Serializes this tally state to an XML string.
    ///
    /// Format: `<tally program="true|false" preview="true|false"/>`
    pub fn to_xml(&self) -> String {
        format!("<tally program=\"{}\" preview=\"\"/>", self.program)
            // Include preview correctly
            .replace("preview=\"\"", &format!("preview=\"{}\"", self.preview))
    }

    /// Parses tally state from an XML string.
    ///
    /// Expected format: `<tally program="true|false" preview="true|false"/>`
    pub fn from_xml(xml: &str) -> Self {
        let program = xml.contains("program=\"true\"");
        let preview = xml.contains("preview=\"true\"");
        Self { program, preview }
    }
}

impl Default for TallyMetadata {
    fn default() -> Self {
        Self {
            program: false,
            preview: false,
        }
    }
}

/// PTZ (Pan-Tilt-Zoom) metadata for NDI camera control.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PtzMetadata {
    /// Pan angle in degrees (-180.0 to 180.0).
    pub pan: f32,
    /// Tilt angle in degrees (-90.0 to 90.0).
    pub tilt: f32,
    /// Zoom level (1.0 = no zoom, higher = more zoom).
    pub zoom: f32,
}

impl PtzMetadata {
    /// Creates a new `PtzMetadata` with given pan, tilt, and zoom values.
    pub fn new(pan: f32, tilt: f32, zoom: f32) -> Self {
        Self { pan, tilt, zoom }
    }

    /// Serializes this PTZ state to an XML string.
    ///
    /// Format: `<ptz pan="..." tilt="..." zoom="..."/>`
    pub fn to_xml(&self) -> String {
        format!(
            "<ptz pan=\"{:.4}\" tilt=\"{:.4}\" zoom=\"{:.4}\"/>",
            self.pan, self.tilt, self.zoom
        )
    }
}

impl Default for PtzMetadata {
    fn default() -> Self {
        Self {
            pan: 0.0,
            tilt: 0.0,
            zoom: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_builder_empty() {
        let frame = NdiMetadataBuilder::new(1000).build();
        assert_eq!(frame.timestamp_ms, 1000);
        assert_eq!(frame.data, "<metadata></metadata>");
    }

    #[test]
    fn test_metadata_builder_single_field() {
        let mut builder = NdiMetadataBuilder::new(500);
        builder.add_field("camera", "CAM1");
        let frame = builder.build();
        assert!(frame.data.contains("key=\"camera\""));
        assert!(frame.data.contains(">CAM1<"));
    }

    #[test]
    fn test_metadata_builder_multiple_fields() {
        let mut builder = NdiMetadataBuilder::new(0);
        builder.add_field("scene", "outdoor");
        builder.add_field("take", "3");
        let frame = builder.build();
        assert!(frame.data.contains("scene"));
        assert!(frame.data.contains("take"));
        assert!(frame.data.contains("outdoor"));
        assert!(frame.data.contains("3"));
    }

    #[test]
    fn test_metadata_builder_chaining() {
        let mut builder = NdiMetadataBuilder::new(100);
        let frame = builder.add_field("a", "1").add_field("b", "2").build();
        assert!(frame.data.contains("a"));
        assert!(frame.data.contains("b"));
    }

    #[test]
    fn test_metadata_parser_empty() {
        let frame = NdiMetadataFrame::new(0, "<metadata></metadata>".to_string());
        let parsed = NdiMetadataParser::parse(&frame);
        assert!(parsed.is_empty());
    }

    #[test]
    fn test_metadata_parser_round_trip() {
        let mut builder = NdiMetadataBuilder::new(9999);
        builder.add_field("key1", "value1");
        builder.add_field("key2", "value2");
        let frame = builder.build();

        let parsed = NdiMetadataParser::parse(&frame);
        assert_eq!(parsed.get("key1"), Some(&"value1".to_string()));
        assert_eq!(parsed.get("key2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_tally_metadata_to_xml_program() {
        let tally = TallyMetadata::new(true, false);
        let xml = tally.to_xml();
        assert!(xml.contains("program=\"true\""));
        assert!(xml.contains("preview=\"false\""));
    }

    #[test]
    fn test_tally_metadata_to_xml_preview() {
        let tally = TallyMetadata::new(false, true);
        let xml = tally.to_xml();
        assert!(xml.contains("program=\"false\""));
        assert!(xml.contains("preview=\"true\""));
    }

    #[test]
    fn test_tally_metadata_from_xml() {
        let xml = "<tally program=\"true\" preview=\"false\"/>";
        let tally = TallyMetadata::from_xml(xml);
        assert!(tally.program);
        assert!(!tally.preview);
    }

    #[test]
    fn test_tally_metadata_round_trip() {
        let original = TallyMetadata::new(true, true);
        let xml = original.to_xml();
        let parsed = TallyMetadata::from_xml(&xml);
        assert_eq!(original, parsed);
    }

    #[test]
    fn test_ptz_metadata_to_xml() {
        let ptz = PtzMetadata::new(45.0, -10.0, 2.5);
        let xml = ptz.to_xml();
        assert!(xml.contains("pan="));
        assert!(xml.contains("tilt="));
        assert!(xml.contains("zoom="));
        assert!(xml.contains("45.0000"));
    }

    #[test]
    fn test_ptz_metadata_default() {
        let ptz = PtzMetadata::default();
        assert_eq!(ptz.pan, 0.0);
        assert_eq!(ptz.tilt, 0.0);
        assert_eq!(ptz.zoom, 1.0);
    }

    #[test]
    fn test_xml_escaping() {
        let mut builder = NdiMetadataBuilder::new(0);
        builder.add_field("test", "a & b");
        let frame = builder.build();
        assert!(frame.data.contains("&amp;"));
    }
}
