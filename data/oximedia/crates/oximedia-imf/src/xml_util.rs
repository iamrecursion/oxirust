//! XML construction utilities for IMF document generation.
//!
//! IMF relies heavily on XML for CPL, PKL, ASSETMAP, and OPL documents.
//! This module provides lightweight builder types for programmatically
//! constructing XML trees and serialising them to strings, without
//! depending on a full XML DOM library.

use std::fmt;

// ---------------------------------------------------------------------------
// XmlAttribute
// ---------------------------------------------------------------------------

/// A single key-value attribute on an XML element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XmlAttribute {
    /// Attribute name.
    pub name: String,
    /// Attribute value (will be XML-escaped on serialisation).
    pub value: String,
}

impl XmlAttribute {
    /// Create a new attribute.
    #[must_use]
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }

    /// Format as `name="escaped_value"`.
    #[must_use]
    pub fn to_xml_string(&self) -> String {
        format!("{}=\"{}\"", self.name, escape_xml_attr(&self.value))
    }
}

impl fmt::Display for XmlAttribute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_xml_string())
    }
}

// ---------------------------------------------------------------------------
// XmlElement
// ---------------------------------------------------------------------------

/// An XML element node with optional attributes, text content, and children.
#[derive(Debug, Clone)]
pub struct XmlElement {
    /// Element tag name (may include namespace prefix, e.g., "cpl:Id").
    pub tag: String,
    /// Attributes on this element.
    pub attributes: Vec<XmlAttribute>,
    /// Direct text content (if any). Mutually exclusive with children in
    /// well-formed documents, but both are allowed for flexibility.
    pub text: Option<String>,
    /// Child elements.
    pub children: Vec<XmlElement>,
}

impl XmlElement {
    /// Create a new element with the given tag name.
    #[must_use]
    pub fn new(tag: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            attributes: Vec::new(),
            text: None,
            children: Vec::new(),
        }
    }

    /// Add an attribute.
    #[must_use]
    pub fn with_attr(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.push(XmlAttribute::new(name, value));
        self
    }

    /// Set the text content.
    #[must_use]
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Add a child element.
    #[must_use]
    pub fn with_child(mut self, child: XmlElement) -> Self {
        self.children.push(child);
        self
    }

    /// Returns `true` if element has no children and no text.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.children.is_empty() && self.text.is_none()
    }

    /// Count of direct children.
    #[must_use]
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Count of attributes.
    #[must_use]
    pub fn attr_count(&self) -> usize {
        self.attributes.len()
    }

    /// Look up an attribute value by name.
    #[must_use]
    pub fn get_attr(&self, name: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.value.as_str())
    }

    /// Find first child with the given tag name.
    #[must_use]
    pub fn find_child(&self, tag: &str) -> Option<&XmlElement> {
        self.children.iter().find(|c| c.tag == tag)
    }

    /// Recursively count all descendant elements (not including self).
    #[must_use]
    pub fn descendant_count(&self) -> usize {
        self.children.iter().map(|c| 1 + c.descendant_count()).sum()
    }

    /// Serialize the element to an XML string (no XML declaration).
    #[must_use]
    pub fn to_xml_string(&self) -> String {
        self.write_xml(0)
    }

    /// Internal recursive writer with indentation level.
    fn write_xml(&self, indent: usize) -> String {
        let prefix = "  ".repeat(indent);
        let mut buf = String::new();

        // Opening tag
        buf.push_str(&prefix);
        buf.push('<');
        buf.push_str(&self.tag);
        for attr in &self.attributes {
            buf.push(' ');
            buf.push_str(&attr.to_xml_string());
        }

        if self.is_empty() {
            buf.push_str("/>\n");
            return buf;
        }

        buf.push('>');

        if let Some(ref text) = self.text {
            if self.children.is_empty() {
                // Inline text: <Tag>text</Tag>
                buf.push_str(&escape_xml_text(text));
                buf.push_str("</");
                buf.push_str(&self.tag);
                buf.push_str(">\n");
                return buf;
            }
        }

        buf.push('\n');

        // Text content on its own line (when children also exist)
        if let Some(ref text) = self.text {
            buf.push_str(&"  ".repeat(indent + 1));
            buf.push_str(&escape_xml_text(text));
            buf.push('\n');
        }

        for child in &self.children {
            buf.push_str(&child.write_xml(indent + 1));
        }

        buf.push_str(&prefix);
        buf.push_str("</");
        buf.push_str(&self.tag);
        buf.push_str(">\n");
        buf
    }
}

impl fmt::Display for XmlElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_xml_string())
    }
}

// ---------------------------------------------------------------------------
// XmlBuilder
// ---------------------------------------------------------------------------

/// Convenience builder for creating IMF XML documents from a root element.
#[derive(Debug, Clone)]
pub struct XmlBuilder {
    /// Optional XML declaration version string (e.g. "1.0").
    pub xml_version: Option<String>,
    /// Optional encoding declaration (e.g. "UTF-8").
    pub encoding: Option<String>,
    /// Root element of the document.
    pub root: XmlElement,
}

impl XmlBuilder {
    /// Create a new XML builder with a root element tag.
    #[must_use]
    pub fn new(root_tag: impl Into<String>) -> Self {
        Self {
            xml_version: Some("1.0".to_string()),
            encoding: Some("UTF-8".to_string()),
            root: XmlElement::new(root_tag),
        }
    }

    /// Set the XML version.
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.xml_version = Some(version.into());
        self
    }

    /// Disable the XML declaration.
    #[must_use]
    pub fn without_declaration(mut self) -> Self {
        self.xml_version = None;
        self.encoding = None;
        self
    }

    /// Add an attribute to the root element.
    #[must_use]
    pub fn with_root_attr(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.root.attributes.push(XmlAttribute::new(name, value));
        self
    }

    /// Add a child element to the root.
    #[must_use]
    pub fn with_child(mut self, child: XmlElement) -> Self {
        self.root.children.push(child);
        self
    }

    /// Serialize the full document to a string (including XML declaration
    /// if configured).
    #[must_use]
    pub fn to_string(&self) -> String {
        let mut buf = String::new();

        if let Some(ref ver) = self.xml_version {
            buf.push_str("<?xml version=\"");
            buf.push_str(ver);
            buf.push('"');
            if let Some(ref enc) = self.encoding {
                buf.push_str(" encoding=\"");
                buf.push_str(enc);
                buf.push('"');
            }
            buf.push_str("?>\n");
        }

        buf.push_str(&self.root.to_xml_string());
        buf
    }
}

impl fmt::Display for XmlBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string())
    }
}

// ---------------------------------------------------------------------------
// XML escaping helpers
// ---------------------------------------------------------------------------

/// Escape special characters for XML attribute values.
fn escape_xml_attr(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Escape special characters for XML text content.
fn escape_xml_text(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribute_new() {
        let a = XmlAttribute::new("id", "abc");
        assert_eq!(a.name, "id");
        assert_eq!(a.value, "abc");
    }

    #[test]
    fn test_attribute_to_xml_string() {
        let a = XmlAttribute::new("name", "hello");
        assert_eq!(a.to_xml_string(), "name=\"hello\"");
    }

    #[test]
    fn test_attribute_escaping() {
        let a = XmlAttribute::new("val", "a<b&c\"d");
        let s = a.to_xml_string();
        assert!(s.contains("&lt;"));
        assert!(s.contains("&amp;"));
        assert!(s.contains("&quot;"));
    }

    #[test]
    fn test_element_empty() {
        let e = XmlElement::new("EmptyTag");
        assert!(e.is_empty());
        let xml = e.to_xml_string();
        assert!(xml.contains("<EmptyTag/>"));
    }

    #[test]
    fn test_element_with_text() {
        let e = XmlElement::new("Title").with_text("My Movie");
        let xml = e.to_xml_string();
        assert!(xml.contains("<Title>My Movie</Title>"));
    }

    #[test]
    fn test_element_text_escaping() {
        let e = XmlElement::new("Data").with_text("a < b & c");
        let xml = e.to_xml_string();
        assert!(xml.contains("a &lt; b &amp; c"));
    }

    #[test]
    fn test_element_with_attr() {
        let e = XmlElement::new("Resource").with_attr("id", "r1");
        let xml = e.to_xml_string();
        assert!(xml.contains("id=\"r1\""));
    }

    #[test]
    fn test_element_with_children() {
        let e = XmlElement::new("Root")
            .with_child(XmlElement::new("A").with_text("1"))
            .with_child(XmlElement::new("B").with_text("2"));
        assert_eq!(e.child_count(), 2);
        let xml = e.to_xml_string();
        assert!(xml.contains("<A>1</A>"));
        assert!(xml.contains("<B>2</B>"));
    }

    #[test]
    fn test_element_get_attr() {
        let e = XmlElement::new("X")
            .with_attr("foo", "bar")
            .with_attr("baz", "qux");
        assert_eq!(e.get_attr("foo"), Some("bar"));
        assert_eq!(e.get_attr("missing"), None);
    }

    #[test]
    fn test_element_find_child() {
        let e = XmlElement::new("Root")
            .with_child(XmlElement::new("Alpha"))
            .with_child(XmlElement::new("Beta"));
        assert!(e.find_child("Alpha").is_some());
        assert!(e.find_child("Gamma").is_none());
    }

    #[test]
    fn test_element_descendant_count() {
        let e = XmlElement::new("Root")
            .with_child(
                XmlElement::new("A")
                    .with_child(XmlElement::new("A1"))
                    .with_child(XmlElement::new("A2")),
            )
            .with_child(XmlElement::new("B"));
        // A + A1 + A2 + B = 4
        assert_eq!(e.descendant_count(), 4);
    }

    #[test]
    fn test_builder_with_declaration() {
        let doc = XmlBuilder::new("Root")
            .with_root_attr("xmlns", "http://example.com")
            .with_child(XmlElement::new("Title").with_text("Test"))
            .to_string();
        assert!(doc.starts_with("<?xml version=\"1.0\""));
        assert!(doc.contains("encoding=\"UTF-8\""));
        assert!(doc.contains("<Root"));
        assert!(doc.contains("<Title>Test</Title>"));
    }

    #[test]
    fn test_builder_without_declaration() {
        let doc = XmlBuilder::new("Root").without_declaration().to_string();
        assert!(!doc.contains("<?xml"));
        assert!(doc.contains("<Root/>"));
    }

    #[test]
    fn test_builder_display() {
        let doc = XmlBuilder::new("Root").with_child(XmlElement::new("A"));
        let s = format!("{doc}");
        assert!(s.contains("<Root>"));
        assert!(s.contains("</Root>"));
    }

    #[test]
    fn test_attribute_display() {
        let a = XmlAttribute::new("x", "y");
        assert_eq!(format!("{a}"), "x=\"y\"");
    }

    #[test]
    fn test_element_display() {
        let e = XmlElement::new("Tag").with_text("val");
        let s = format!("{e}");
        assert!(s.contains("<Tag>val</Tag>"));
    }
}

// ---------------------------------------------------------------------------
// XmlStreamWriter  (streaming / zero-copy incremental XML emission)
// ---------------------------------------------------------------------------

use crate::cpl_parser::CompositionPlaylist;
use crate::ImfError;
use quick_xml::{
    events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event},
    Writer,
};

/// A streaming XML writer that emits elements incrementally to any
/// [`std::io::Write`] target.
///
/// Unlike [`XmlBuilder`], this type does not buffer the full document in
/// memory; each `write_*` call flushes bytes directly to the underlying
/// writer, making it suitable for large CPL / OPL generation.
pub struct XmlStreamWriter<W: std::io::Write> {
    writer: Writer<W>,
}

impl<W: std::io::Write> XmlStreamWriter<W> {
    /// Create a new streaming writer wrapping `inner`.
    pub fn new(inner: W) -> Self {
        Self {
            writer: Writer::new_with_indent(inner, b' ', 2),
        }
    }

    /// Write the XML declaration (`<?xml version="1.0" encoding="UTF-8"?>`).
    pub fn write_declaration(&mut self) -> std::io::Result<()> {
        self.writer
            .write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))
    }

    /// Write `<name>text</name>` as a single inline element.
    pub fn write_element(&mut self, name: &str, text: &str) -> std::io::Result<()> {
        self.writer
            .create_element(name)
            .write_text_content(BytesText::new(text))?;
        Ok(())
    }

    /// Write an opening tag `<name>`.
    pub fn start_element(&mut self, name: &str) -> std::io::Result<()> {
        self.writer
            .write_event(Event::Start(BytesStart::new(name.to_string())))
    }

    /// Write an opening tag with a single attribute.
    pub fn start_element_with_attr(
        &mut self,
        name: &str,
        attr_name: &str,
        attr_value: &str,
    ) -> std::io::Result<()> {
        let mut start = BytesStart::new(name.to_string());
        start.push_attribute((attr_name, attr_value));
        self.writer.write_event(Event::Start(start))
    }

    /// Write a closing tag `</name>`.
    pub fn end_element(&mut self, name: &str) -> std::io::Result<()> {
        self.writer
            .write_event(Event::End(BytesEnd::new(name.to_string())))
    }

    /// Consume this writer and return the inner `W`.
    pub fn into_inner(self) -> W {
        self.writer.into_inner()
    }
}

/// Stream a [`CompositionPlaylist`] to any [`std::io::Write`] target.
///
/// The output conforms to a simplified subset of SMPTE ST 2067-3 (the same
/// schema as [`CompositionPlaylist::to_xml`]) and can be round-tripped back
/// with [`CompositionPlaylist::from_xml`].
pub fn write_cpl_streaming<W: std::io::Write>(
    cpl: &CompositionPlaylist,
    writer: W,
) -> Result<(), ImfError> {
    let mut sw = XmlStreamWriter::new(writer);
    sw.write_declaration()
        .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;

    sw.start_element_with_attr(
        "CompositionPlaylist",
        "xmlns",
        "http://www.smpte-ra.org/schemas/2067-3/2016",
    )
    .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;

    sw.write_element("Id", &format!("urn:uuid:{}", cpl.id))
        .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;
    sw.write_element("ContentTitle", &cpl.content_title)
        .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;

    let (rate_num, rate_den) = cpl.edit_rate;
    sw.write_element("EditRate", &format!("{rate_num} {rate_den}"))
        .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;

    sw.start_element("SegmentList")
        .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;

    for seg in cpl.segments() {
        sw.start_element("Segment")
            .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;

        sw.write_element("Id", &format!("urn:uuid:{}", seg.id))
            .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;

        if let Some(ref ann) = seg.annotation {
            sw.write_element("Annotation", ann)
                .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;
        }

        sw.start_element("SequenceList")
            .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;

        for seq in &seg.sequences {
            sw.start_element("Sequence")
                .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;

            sw.write_element("Id", &format!("urn:uuid:{}", seq.id))
                .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;
            sw.write_element("TrackId", &format!("urn:uuid:{}", seq.track_id))
                .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;

            sw.start_element("ResourceList")
                .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;

            for res in &seq.resources {
                sw.start_element("Resource")
                    .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;

                sw.write_element("TrackFileId", &format!("urn:uuid:{}", res.track_file_id))
                    .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;
                sw.write_element("SourceDuration", &res.source_duration.to_string())
                    .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;
                sw.write_element("EntryPoint", &res.entry_point.to_string())
                    .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;
                sw.write_element("IntrinsicDuration", &res.intrinsic_duration.to_string())
                    .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;
                sw.write_element("RepeatCount", &res.repeat_count.to_string())
                    .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;

                sw.end_element("Resource")
                    .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;
            }

            sw.end_element("ResourceList")
                .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;
            sw.end_element("Sequence")
                .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;
        }

        sw.end_element("SequenceList")
            .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;
        sw.end_element("Segment")
            .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;
    }

    sw.end_element("SegmentList")
        .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;
    sw.end_element("CompositionPlaylist")
        .map_err(|e: std::io::Error| ImfError::XmlError(e.to_string()))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Streaming writer tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use crate::cpl_parser::{CompositionPlaylist, CplResource, CplSegment, CplSequence};

    fn reference_cpl() -> CompositionPlaylist {
        let mut cpl = CompositionPlaylist::new("stream-cpl-001", "Streaming Test Film", (25, 1));

        let mut seg = CplSegment::new("stream-seg-001");
        seg.annotation = Some("Chapter 1".to_string());
        let mut seq = CplSequence::new("stream-seq-001", "stream-track-001");
        seq.add_resource(CplResource::simple("stream-tf-001", 2500));
        seg.add_sequence(seq);
        cpl.add_segment(seg);

        let mut seg2 = CplSegment::new("stream-seg-002");
        let mut seq2 = CplSequence::new("stream-seq-002", "stream-track-001");
        let mut res2 = CplResource::simple("stream-tf-002", 5000);
        res2.entry_point = 25;
        res2.repeat_count = 2;
        seq2.add_resource(res2);
        seg2.add_sequence(seq2);
        cpl.add_segment(seg2);

        cpl
    }

    #[test]
    fn test_streaming_writer_valid_xml() {
        let cpl = reference_cpl();
        let mut buf: Vec<u8> = Vec::new();
        write_cpl_streaming(&cpl, &mut buf).expect("streaming write must succeed");

        let xml = String::from_utf8(buf).expect("output must be valid UTF-8");

        // Validate that quick-xml can parse it back without errors.
        use quick_xml::events::Event;
        use quick_xml::Reader;
        let mut reader = Reader::from_str(&xml);
        let mut event_buf = Vec::new();
        let mut element_count = 0usize;
        loop {
            match reader.read_event_into(&mut event_buf) {
                Ok(Event::Start(_)) => element_count += 1,
                Ok(Event::Eof) => break,
                Err(e) => panic!("streaming output is not valid XML: {e}"),
                _ => {}
            }
            event_buf.clear();
        }
        assert!(
            element_count > 0,
            "output must contain at least one element"
        );
        assert!(
            xml.contains("CompositionPlaylist"),
            "must contain CPL root element"
        );
        assert!(xml.contains("stream-cpl-001"), "must contain CPL id");
    }

    #[test]
    fn test_streaming_matches_eager() {
        let cpl = reference_cpl();

        // Eager string-builder output.
        let eager_xml = cpl.to_xml();

        // Streaming output.
        let mut streaming_buf: Vec<u8> = Vec::new();
        write_cpl_streaming(&cpl, &mut streaming_buf).expect("streaming write must succeed");
        let streaming_xml = String::from_utf8(streaming_buf).expect("valid UTF-8");

        // Both outputs must round-trip to structurally equivalent CPLs.
        let eager_parsed =
            CompositionPlaylist::from_xml(&eager_xml).expect("eager round-trip must succeed");
        let streaming_parsed = CompositionPlaylist::from_xml(&streaming_xml)
            .expect("streaming round-trip must succeed");

        assert_eq!(eager_parsed.id, streaming_parsed.id, "id must match");
        assert_eq!(
            eager_parsed.content_title, streaming_parsed.content_title,
            "content_title must match"
        );
        assert_eq!(
            eager_parsed.edit_rate, streaming_parsed.edit_rate,
            "edit_rate must match"
        );
        assert_eq!(
            eager_parsed.segment_count(),
            streaming_parsed.segment_count(),
            "segment count must match"
        );
        assert_eq!(
            eager_parsed.total_duration(),
            streaming_parsed.total_duration(),
            "total_duration must match"
        );
    }
}
