//! NDI metadata frames: XML metadata embedding, PTZ control frames, and runtime config.
//!
//! NDI carries arbitrary XML metadata alongside audio/video.  This module provides
//! builders and parsers for the standard metadata payloads (PTZ control, tally
//! feedback, connection metadata, and custom key/value runtime configuration).

#![allow(dead_code)]

/// The XML envelope tag used for NDI metadata frames.
pub const NDI_METADATA_TAG: &str = "ndi_metadata";

/// Well-known NDI metadata frame types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataFrameType {
    /// PTZ control command.
    Ptz,
    /// Tally status update.
    Tally,
    /// Source connection information.
    Connection,
    /// Runtime configuration key/value pair.
    Config,
    /// Arbitrary user-defined XML.
    Custom(String),
}

impl MetadataFrameType {
    /// Return the XML element name for this frame type.
    pub fn element_name(&self) -> &str {
        match self {
            Self::Ptz => "ptz",
            Self::Tally => "tally",
            Self::Connection => "connection",
            Self::Config => "config",
            Self::Custom(name) => name.as_str(),
        }
    }
}

/// A serialised NDI metadata frame.
#[derive(Debug, Clone)]
pub struct MetadataFrame {
    /// Frame type.
    pub frame_type: MetadataFrameType,
    /// Raw XML payload (the inner content, without the outer envelope).
    pub xml: String,
    /// Optional timecode in 100 ns units (NDI convention).
    pub timecode: Option<i64>,
}

impl MetadataFrame {
    /// Create a new metadata frame.
    pub fn new(frame_type: MetadataFrameType, xml: String) -> Self {
        Self {
            frame_type,
            xml,
            timecode: None,
        }
    }

    /// Attach a timecode to this frame.
    pub fn with_timecode(mut self, timecode: i64) -> Self {
        self.timecode = Some(timecode);
        self
    }

    /// Serialise the frame to a complete XML string.
    pub fn to_xml(&self) -> String {
        let tag = self.frame_type.element_name();
        format!("<{tag}>{}</{tag}>", self.xml)
    }

    /// Byte length of the serialised XML.
    pub fn byte_len(&self) -> usize {
        self.to_xml().len()
    }
}

/// Builder for PTZ control metadata frames.
#[derive(Debug, Clone, Default)]
pub struct PtzFrameBuilder {
    pan: Option<f32>,
    tilt: Option<f32>,
    zoom: Option<f32>,
    focus: Option<f32>,
    preset_recall: Option<u8>,
    preset_store: Option<u8>,
    auto_focus: bool,
    stop: bool,
}

impl PtzFrameBuilder {
    /// Create a new PTZ frame builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set pan speed in the range -1.0 … +1.0 (negative = left, positive = right).
    pub fn pan(mut self, speed: f32) -> Self {
        self.pan = Some(speed.clamp(-1.0, 1.0));
        self
    }

    /// Set tilt speed in the range -1.0 … +1.0 (negative = down, positive = up).
    pub fn tilt(mut self, speed: f32) -> Self {
        self.tilt = Some(speed.clamp(-1.0, 1.0));
        self
    }

    /// Set zoom speed in the range -1.0 … +1.0 (negative = out, positive = in).
    pub fn zoom(mut self, speed: f32) -> Self {
        self.zoom = Some(speed.clamp(-1.0, 1.0));
        self
    }

    /// Set focus speed in the range -1.0 … +1.0 (negative = near, positive = far).
    pub fn focus(mut self, speed: f32) -> Self {
        self.focus = Some(speed.clamp(-1.0, 1.0));
        self
    }

    /// Recall a camera preset by index (0-99).
    pub fn recall_preset(mut self, index: u8) -> Self {
        self.preset_recall = Some(index);
        self
    }

    /// Store the current camera position as preset `index`.
    pub fn store_preset(mut self, index: u8) -> Self {
        self.preset_store = Some(index);
        self
    }

    /// Enable auto-focus.
    pub fn auto_focus(mut self) -> Self {
        self.auto_focus = true;
        self
    }

    /// Stop all camera motion.
    pub fn stop(mut self) -> Self {
        self.stop = true;
        self
    }

    /// Build the metadata frame XML payload.
    pub fn build(self) -> MetadataFrame {
        let mut attrs = String::new();
        if let Some(v) = self.pan {
            attrs.push_str(&format!(" pan=\"{v:.4}\""));
        }
        if let Some(v) = self.tilt {
            attrs.push_str(&format!(" tilt=\"{v:.4}\""));
        }
        if let Some(v) = self.zoom {
            attrs.push_str(&format!(" zoom=\"{v:.4}\""));
        }
        if let Some(v) = self.focus {
            attrs.push_str(&format!(" focus=\"{v:.4}\""));
        }
        if let Some(p) = self.preset_recall {
            attrs.push_str(&format!(" recall_preset=\"{p}\""));
        }
        if let Some(p) = self.preset_store {
            attrs.push_str(&format!(" store_preset=\"{p}\""));
        }
        if self.auto_focus {
            attrs.push_str(" autofocus=\"true\"");
        }
        if self.stop {
            attrs.push_str(" stop=\"true\"");
        }
        let xml = format!("<ndi_ptz{attrs}/>");
        MetadataFrame::new(MetadataFrameType::Ptz, xml)
    }
}

/// Builder for runtime configuration metadata frames.
#[derive(Debug, Clone)]
pub struct ConfigFrameBuilder {
    entries: Vec<(String, String)>,
}

impl ConfigFrameBuilder {
    /// Create a new config frame builder.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a key/value configuration entry.
    pub fn set(mut self, key: &str, value: &str) -> Self {
        self.entries.push((key.to_string(), value.to_string()));
        self
    }

    /// Build the metadata frame.
    pub fn build(self) -> MetadataFrame {
        let inner: String = self
            .entries
            .iter()
            .map(|(k, v)| format!("<entry key=\"{k}\" value=\"{v}\"/>"))
            .collect::<Vec<_>>()
            .join("");
        MetadataFrame::new(MetadataFrameType::Config, inner)
    }
}

impl Default for ConfigFrameBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a simple key/value config frame from its inner XML string.
///
/// Expects entries in the form `<entry key="K" value="V"/>`.
pub fn parse_config_frame(xml: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    // Minimal hand-rolled parser: find all `key="..." value="..."` occurrences.
    let mut remaining = xml;
    while let Some(key_start) = remaining.find("key=\"") {
        remaining = &remaining[key_start + 5..];
        let key_end = remaining.find('"').unwrap_or(0);
        let key = remaining[..key_end].to_string();
        remaining = &remaining[key_end..];
        if let Some(val_start) = remaining.find("value=\"") {
            remaining = &remaining[val_start + 7..];
            let val_end = remaining.find('"').unwrap_or(0);
            let value = remaining[..val_end].to_string();
            remaining = &remaining[val_end..];
            pairs.push((key, value));
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_frame_type_element_names() {
        assert_eq!(MetadataFrameType::Ptz.element_name(), "ptz");
        assert_eq!(MetadataFrameType::Tally.element_name(), "tally");
        assert_eq!(MetadataFrameType::Connection.element_name(), "connection");
        assert_eq!(MetadataFrameType::Config.element_name(), "config");
        assert_eq!(
            MetadataFrameType::Custom("hello".into()).element_name(),
            "hello"
        );
    }

    #[test]
    fn test_metadata_frame_to_xml() {
        let frame = MetadataFrame::new(MetadataFrameType::Config, "inner".to_string());
        let xml = frame.to_xml();
        assert!(xml.starts_with("<config>"));
        assert!(xml.ends_with("</config>"));
        assert!(xml.contains("inner"));
    }

    #[test]
    fn test_metadata_frame_byte_len() {
        let frame = MetadataFrame::new(MetadataFrameType::Tally, "x".to_string());
        assert_eq!(frame.byte_len(), frame.to_xml().len());
    }

    #[test]
    fn test_metadata_frame_with_timecode() {
        let frame =
            MetadataFrame::new(MetadataFrameType::Ptz, "".to_string()).with_timecode(1_000_000);
        assert_eq!(frame.timecode, Some(1_000_000));
    }

    #[test]
    fn test_ptz_builder_pan_clamp() {
        let frame = PtzFrameBuilder::new().pan(2.0).build();
        assert!(frame.xml.contains("pan=\"1.0000\""));
    }

    #[test]
    fn test_ptz_builder_tilt() {
        let frame = PtzFrameBuilder::new().tilt(-0.5).build();
        assert!(frame.xml.contains("tilt=\"-0.5000\""));
    }

    #[test]
    fn test_ptz_builder_zoom() {
        let frame = PtzFrameBuilder::new().zoom(0.75).build();
        assert!(frame.xml.contains("zoom=\"0.7500\""));
    }

    #[test]
    fn test_ptz_builder_recall_preset() {
        let frame = PtzFrameBuilder::new().recall_preset(3).build();
        assert!(frame.xml.contains("recall_preset=\"3\""));
    }

    #[test]
    fn test_ptz_builder_store_preset() {
        let frame = PtzFrameBuilder::new().store_preset(7).build();
        assert!(frame.xml.contains("store_preset=\"7\""));
    }

    #[test]
    fn test_ptz_builder_auto_focus() {
        let frame = PtzFrameBuilder::new().auto_focus().build();
        assert!(frame.xml.contains("autofocus=\"true\""));
    }

    #[test]
    fn test_ptz_builder_stop() {
        let frame = PtzFrameBuilder::new().stop().build();
        assert!(frame.xml.contains("stop=\"true\""));
    }

    #[test]
    fn test_ptz_frame_type() {
        let frame = PtzFrameBuilder::new().pan(0.0).build();
        assert_eq!(frame.frame_type, MetadataFrameType::Ptz);
    }

    #[test]
    fn test_config_frame_builder() {
        let frame = ConfigFrameBuilder::new()
            .set("bitrate", "10000")
            .set("quality", "high")
            .build();
        assert!(frame.xml.contains("bitrate"));
        assert!(frame.xml.contains("10000"));
        assert!(frame.xml.contains("quality"));
        assert!(frame.xml.contains("high"));
    }

    #[test]
    fn test_parse_config_frame_roundtrip() {
        let frame = ConfigFrameBuilder::new()
            .set("fps", "60")
            .set("codec", "h264")
            .build();
        let pairs = parse_config_frame(&frame.xml);
        assert_eq!(pairs.len(), 2);
        let map: std::collections::HashMap<_, _> = pairs.into_iter().collect();
        assert_eq!(map.get("fps").map(|s| s.as_str()), Some("60"));
        assert_eq!(map.get("codec").map(|s| s.as_str()), Some("h264"));
    }

    #[test]
    fn test_parse_config_frame_empty() {
        let pairs = parse_config_frame("");
        assert!(pairs.is_empty());
    }
}
