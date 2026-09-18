//! Dublin Core metadata support
//!
//! Implements the 15-element Dublin Core Metadata Element Set (ISO 15836)
//! with XML serialization and deserialization via quick-xml, and JSON
//! serialization via serde_json.
//!
//! See: <https://dublincore.org/specifications/dublin-core/dces/>

use crate::{Error, Result};
use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
use quick_xml::Writer;
use serde::{Deserialize, Serialize};
use std::io::Cursor;

/// A Dublin Core metadata record following the 15-element DCES (ISO 15836).
///
/// All fields are optional per the specification, and `subject`, `contributor`,
/// `relation`, and `type_` may be repeated, so they are `Vec<String>`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DublinCoreRecord {
    /// `dc:title` — the name given to the resource.
    pub title: Option<String>,
    /// `dc:creator` — entity primarily responsible for the content.
    pub creator: Option<String>,
    /// `dc:subject` — topic of the content (repeatable).
    pub subject: Vec<String>,
    /// `dc:description` — account of the content.
    pub description: Option<String>,
    /// `dc:publisher` — entity responsible for making the resource available.
    pub publisher: Option<String>,
    /// `dc:contributor` — entity making a secondary contribution (repeatable).
    pub contributor: Vec<String>,
    /// `dc:date` — date associated with the resource (`YYYY-MM-DD` or ISO 8601).
    pub date: Option<String>,
    /// `dc:type` — nature or genre of the resource (repeatable).
    pub type_: Vec<String>,
    /// `dc:format` — file format or medium (e.g. `"video/x-matroska"`).
    pub format: Option<String>,
    /// `dc:identifier` — unambiguous reference (URI, DOI, ISRC, …).
    pub identifier: Option<String>,
    /// `dc:source` — resource from which this resource is derived.
    pub source: Option<String>,
    /// `dc:language` — language of the content (RFC 5646, e.g. `"en-US"`).
    pub language: Option<String>,
    /// `dc:relation` — related resource (repeatable).
    pub relation: Vec<String>,
    /// `dc:coverage` — spatial or temporal extent.
    pub coverage: Option<String>,
    /// `dc:rights` — rights information (license, copyright statement).
    pub rights: Option<String>,
}

impl DublinCoreRecord {
    /// Create an empty Dublin Core record.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `dc:title`.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set `dc:creator`.
    #[must_use]
    pub fn with_creator(mut self, creator: impl Into<String>) -> Self {
        self.creator = Some(creator.into());
        self
    }

    /// Add a `dc:subject`.
    #[must_use]
    pub fn with_subject(mut self, subject: impl Into<String>) -> Self {
        self.subject.push(subject.into());
        self
    }

    /// Set `dc:description`.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set `dc:publisher`.
    #[must_use]
    pub fn with_publisher(mut self, publisher: impl Into<String>) -> Self {
        self.publisher = Some(publisher.into());
        self
    }

    /// Add a `dc:contributor`.
    #[must_use]
    pub fn with_contributor(mut self, contributor: impl Into<String>) -> Self {
        self.contributor.push(contributor.into());
        self
    }

    /// Set `dc:date`.
    #[must_use]
    pub fn with_date(mut self, date: impl Into<String>) -> Self {
        self.date = Some(date.into());
        self
    }

    /// Add a `dc:type`.
    #[must_use]
    pub fn with_type(mut self, type_: impl Into<String>) -> Self {
        self.type_.push(type_.into());
        self
    }

    /// Set `dc:format`.
    #[must_use]
    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    /// Set `dc:identifier`.
    #[must_use]
    pub fn with_identifier(mut self, identifier: impl Into<String>) -> Self {
        self.identifier = Some(identifier.into());
        self
    }

    /// Set `dc:source`.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set `dc:language`.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Add a `dc:relation`.
    #[must_use]
    pub fn with_relation(mut self, relation: impl Into<String>) -> Self {
        self.relation.push(relation.into());
        self
    }

    /// Set `dc:coverage`.
    #[must_use]
    pub fn with_coverage(mut self, coverage: impl Into<String>) -> Self {
        self.coverage = Some(coverage.into());
        self
    }

    /// Set `dc:rights`.
    #[must_use]
    pub fn with_rights(mut self, rights: impl Into<String>) -> Self {
        self.rights = Some(rights.into());
        self
    }

    // ── XML serialization ─────────────────────────────────────────────────────

    /// Serialize this record to a standalone XML document using the standard
    /// `http://purl.org/dc/elements/1.1/` namespace.
    ///
    /// The output is a `<metadata>` root element with `dc:*` child elements.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying quick-xml writer fails.
    pub fn to_xml(&self) -> Result<String> {
        let mut writer = Writer::new_with_indent(Cursor::new(Vec::new()), b' ', 2);

        // XML declaration
        writer
            .write_event(Event::Decl(quick_xml::events::BytesDecl::new(
                "1.0",
                Some("UTF-8"),
                None,
            )))
            .map_err(|e| Error::Xml(e.into()))?;

        // Root <metadata> element
        let mut root = BytesStart::new("metadata");
        root.push_attribute(("xmlns:dc", "http://purl.org/dc/elements/1.1/"));
        root.push_attribute(("xmlns:dcterms", "http://purl.org/dc/terms/"));
        writer
            .write_event(Event::Start(root))
            .map_err(|e| Error::Xml(e.into()))?;

        // Helper closure to write a single optional element
        let write_opt = |w: &mut Writer<Cursor<Vec<u8>>>,
                         tag: &str,
                         value: &Option<String>|
         -> std::result::Result<(), quick_xml::Error> {
            if let Some(ref v) = *value {
                w.write_event(Event::Start(BytesStart::new(tag)))?;
                w.write_event(Event::Text(BytesText::new(v)))?;
                w.write_event(Event::End(BytesEnd::new(tag)))?;
            }
            Ok(())
        };

        // Helper closure to write repeated elements
        let write_vec = |w: &mut Writer<Cursor<Vec<u8>>>,
                         tag: &str,
                         values: &[String]|
         -> std::result::Result<(), quick_xml::Error> {
            for v in values {
                w.write_event(Event::Start(BytesStart::new(tag)))?;
                w.write_event(Event::Text(BytesText::new(v)))?;
                w.write_event(Event::End(BytesEnd::new(tag)))?;
            }
            Ok(())
        };

        write_opt(&mut writer, "dc:title", &self.title).map_err(|e| Error::Xml(e.into()))?;
        write_opt(&mut writer, "dc:creator", &self.creator).map_err(|e| Error::Xml(e.into()))?;
        write_vec(&mut writer, "dc:subject", &self.subject).map_err(|e| Error::Xml(e.into()))?;
        write_opt(&mut writer, "dc:description", &self.description)
            .map_err(|e| Error::Xml(e.into()))?;
        write_opt(&mut writer, "dc:publisher", &self.publisher)
            .map_err(|e| Error::Xml(e.into()))?;
        write_vec(&mut writer, "dc:contributor", &self.contributor)
            .map_err(|e| Error::Xml(e.into()))?;
        write_opt(&mut writer, "dc:date", &self.date).map_err(|e| Error::Xml(e.into()))?;
        write_vec(&mut writer, "dc:type", &self.type_).map_err(|e| Error::Xml(e.into()))?;
        write_opt(&mut writer, "dc:format", &self.format).map_err(|e| Error::Xml(e.into()))?;
        write_opt(&mut writer, "dc:identifier", &self.identifier)
            .map_err(|e| Error::Xml(e.into()))?;
        write_opt(&mut writer, "dc:source", &self.source).map_err(|e| Error::Xml(e.into()))?;
        write_opt(&mut writer, "dc:language", &self.language).map_err(|e| Error::Xml(e.into()))?;
        write_vec(&mut writer, "dc:relation", &self.relation).map_err(|e| Error::Xml(e.into()))?;
        write_opt(&mut writer, "dc:coverage", &self.coverage).map_err(|e| Error::Xml(e.into()))?;
        write_opt(&mut writer, "dc:rights", &self.rights).map_err(|e| Error::Xml(e.into()))?;

        writer
            .write_event(Event::End(BytesEnd::new("metadata")))
            .map_err(|e| Error::Xml(e.into()))?;

        let bytes = writer.into_inner().into_inner();
        String::from_utf8(bytes)
            .map_err(|e| Error::Metadata(format!("Dublin Core XML is not valid UTF-8: {e}")))
    }

    // ── XML deserialization ───────────────────────────────────────────────────

    /// Deserialize a `DublinCoreRecord` from an XML string.
    ///
    /// Expects the same format produced by [`to_xml`](Self::to_xml).
    ///
    /// # Errors
    ///
    /// Returns an error if the XML is malformed or contains invalid UTF-8.
    pub fn from_xml(xml: &str) -> Result<Self> {
        use quick_xml::events::Event as XmlEvent;
        use quick_xml::Reader;

        let mut record = Self::new();
        let mut reader = Reader::from_str(xml);
        // Do NOT trim text — we accumulate text manually to avoid dropped fragments.
        reader.config_mut().trim_text(false);

        let mut current_tag: Option<String> = None;
        // Accumulation buffer for text content of the current element.
        let mut text_buf = String::new();

        /// Commit the accumulated text buffer into the record.
        fn commit_text(tag: &str, text: &str, record: &mut DublinCoreRecord) {
            let trimmed = text.trim().to_string();
            if trimmed.is_empty() {
                return;
            }
            match tag {
                "dc:title" => record.title = Some(trimmed),
                "dc:creator" => record.creator = Some(trimmed),
                "dc:subject" => record.subject.push(trimmed),
                "dc:description" => record.description = Some(trimmed),
                "dc:publisher" => record.publisher = Some(trimmed),
                "dc:contributor" => record.contributor.push(trimmed),
                "dc:date" => record.date = Some(trimmed),
                "dc:type" => record.type_.push(trimmed),
                "dc:format" => record.format = Some(trimmed),
                "dc:identifier" => record.identifier = Some(trimmed),
                "dc:source" => record.source = Some(trimmed),
                "dc:language" => record.language = Some(trimmed),
                "dc:relation" => record.relation.push(trimmed),
                "dc:coverage" => record.coverage = Some(trimmed),
                "dc:rights" => record.rights = Some(trimmed),
                _ => {}
            }
        }

        loop {
            match reader.read_event() {
                Ok(XmlEvent::Start(ref e)) => {
                    // Before switching to a new tag, commit any buffered text for the old one.
                    if let Some(ref old_tag) = current_tag {
                        commit_text(old_tag, &text_buf, &mut record);
                        text_buf.clear();
                    }
                    let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    current_tag = Some(tag);
                    text_buf.clear();
                }
                Ok(XmlEvent::Text(ref e)) => {
                    if current_tag.is_some() {
                        // decode() returns the raw (still entity-escaped) bytes as a string.
                        // We accumulate these fragments and unescape at commit time.
                        let raw = e
                            .decode()
                            .map_err(|err| Error::Metadata(format!("XML decode error: {err}")))?;
                        text_buf.push_str(&raw);
                    }
                }
                Ok(XmlEvent::GeneralRef(ref e)) => {
                    // quick-xml 0.39 emits GeneralRef for entity references like &amp; &lt; etc.
                    // We reconstruct the entity reference string and add to buffer so that
                    // unescape() can decode it at commit time.
                    if current_tag.is_some() {
                        let ref_name = e.decode().map_err(|err| {
                            Error::Metadata(format!("XML ref decode error: {err}"))
                        })?;
                        // Reconstruct as `&name;` for later unescaping.
                        text_buf.push('&');
                        text_buf.push_str(&ref_name);
                        text_buf.push(';');
                    }
                }
                Ok(XmlEvent::End(_)) => {
                    if let Some(ref tag) = current_tag {
                        // Unescape accumulated buffer and commit.
                        let unescaped = quick_xml::escape::unescape(&text_buf)
                            .map_err(|err| {
                                Error::Metadata(format!("XML entity unescape error: {err}"))
                            })?
                            .to_string();
                        commit_text(tag, &unescaped, &mut record);
                    }
                    current_tag = None;
                    text_buf.clear();
                }
                Ok(XmlEvent::Eof) => break,
                Err(e) => return Err(Error::Xml(e.into())),
                _ => {}
            }
        }

        Ok(record)
    }

    // ── JSON serialization ────────────────────────────────────────────────────

    /// Serialize to JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if JSON serialization fails.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| Error::Metadata(format!("Dublin Core JSON serialization failed: {e}")))
    }

    /// Deserialize from JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if JSON deserialization fails.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| Error::Metadata(format!("Dublin Core JSON deserialization failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record() -> DublinCoreRecord {
        DublinCoreRecord::new()
            .with_title("Preservation Test Film")
            .with_creator("COOLJAPAN OU")
            .with_subject("digital preservation")
            .with_subject("archival media")
            .with_description("A test film for the OxiMedia archive.")
            .with_publisher("OxiMedia Archive")
            .with_contributor("Test Contributor")
            .with_date("2025-03-14")
            .with_type("MovingImage")
            .with_format("video/x-matroska")
            .with_identifier("urn:uuid:00000000-0000-0000-0000-000000000001")
            .with_language("en-US")
            .with_rights("CC0 1.0 Universal")
    }

    #[test]
    fn test_dublin_core_builder() {
        let record = sample_record();
        assert_eq!(record.title.as_deref(), Some("Preservation Test Film"));
        assert_eq!(record.creator.as_deref(), Some("COOLJAPAN OU"));
        assert_eq!(record.subject.len(), 2);
        assert_eq!(record.contributor.len(), 1);
        assert_eq!(record.type_.len(), 1);
    }

    #[test]
    fn test_dublin_core_xml_serialization() {
        let record = sample_record();
        let xml = record.to_xml().expect("XML serialization should succeed");
        assert!(xml.contains("<?xml version="));
        assert!(xml.contains("xmlns:dc=\"http://purl.org/dc/elements/1.1/\""));
        assert!(xml.contains("<dc:title>Preservation Test Film</dc:title>"));
        assert!(xml.contains("<dc:creator>COOLJAPAN OU</dc:creator>"));
        // Both subjects should appear
        assert!(xml.contains("<dc:subject>digital preservation</dc:subject>"));
        assert!(xml.contains("<dc:subject>archival media</dc:subject>"));
        assert!(xml.contains("<dc:format>video/x-matroska</dc:format>"));
        assert!(xml.contains("<dc:rights>CC0 1.0 Universal</dc:rights>"));
    }

    #[test]
    fn test_dublin_core_xml_roundtrip() {
        let original = sample_record();
        let xml = original.to_xml().expect("serialization should succeed");
        let recovered = DublinCoreRecord::from_xml(&xml).expect("deserialization should succeed");

        assert_eq!(original.title, recovered.title);
        assert_eq!(original.creator, recovered.creator);
        assert_eq!(original.subject, recovered.subject);
        assert_eq!(original.description, recovered.description);
        assert_eq!(original.publisher, recovered.publisher);
        assert_eq!(original.date, recovered.date);
        assert_eq!(original.format, recovered.format);
        assert_eq!(original.identifier, recovered.identifier);
        assert_eq!(original.language, recovered.language);
        assert_eq!(original.rights, recovered.rights);
    }

    #[test]
    fn test_dublin_core_json_roundtrip() {
        let original = sample_record();
        let json = original
            .to_json()
            .expect("JSON serialization should succeed");
        let recovered =
            DublinCoreRecord::from_json(&json).expect("JSON deserialization should succeed");
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_dublin_core_empty_record() {
        let record = DublinCoreRecord::new();
        let xml = record.to_xml().expect("empty record should serialize");
        // Should produce a valid but sparse XML document
        assert!(xml.contains("<metadata"));
        assert!(xml.contains("</metadata>"));
        // No dc:title element should be present
        assert!(!xml.contains("dc:title"));
    }

    #[test]
    fn test_dublin_core_xml_special_chars() {
        // Test that XML special characters are correctly escaped and recovered.
        // quick-xml 0.39 splits entity references into separate events, so our
        // from_xml accumulates Text + GeneralRef events into a single buffer.
        let record = DublinCoreRecord::new()
            .with_title("Film & Video: <A Study>")
            .with_description("Description with \"quotes\" and 'apostrophes'");
        let xml = record.to_xml().expect("serialization should succeed");
        // Verify the XML has properly escaped entities
        assert!(
            xml.contains("&amp;"),
            "XML should contain escaped & as &amp; in title"
        );
        assert!(
            xml.contains("&lt;"),
            "XML should contain escaped < as &lt; in title"
        );
        // Full XML round-trip
        let recovered = DublinCoreRecord::from_xml(&xml).expect("deserialization should succeed");
        assert_eq!(
            recovered.title.as_deref(),
            Some("Film & Video: <A Study>"),
            "Title with & and < must round-trip via XML"
        );
        assert_eq!(
            recovered.description.as_deref(),
            Some("Description with \"quotes\" and 'apostrophes'"),
            "Description with quotes must round-trip via XML"
        );
    }
}
