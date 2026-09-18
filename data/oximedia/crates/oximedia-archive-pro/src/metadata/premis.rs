//! PREMIS (Preservation Metadata Implementation Strategies) support
//!
//! PREMIS is a standard for preservation metadata in digital archives.
//! See: <https://www.loc.gov/standards/premis/>

use crate::Result;
use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
use quick_xml::Writer;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write as IoWrite;
use std::path::Path;

/// PREMIS object types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectType {
    /// File object
    File,
    /// Bitstream
    Bitstream,
    /// Representation
    Representation,
    /// Intellectual entity
    IntellectualEntity,
}

/// PREMIS event types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    /// Capture/creation
    Capture,
    /// Ingestion into repository
    Ingestion,
    /// Format migration
    Migration,
    /// Validation
    Validation,
    /// Fixity check
    FixityCheck,
    /// Replication
    Replication,
    /// Other event
    Other(String),
}

/// PREMIS object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PremisObject {
    /// Object identifier
    pub identifier: String,
    /// Object type
    pub object_type: ObjectType,
    /// Original name
    pub original_name: Option<String>,
    /// File size in bytes
    pub size: Option<u64>,
    /// Format
    pub format: Option<String>,
    /// Creation date
    pub creation_date: chrono::DateTime<chrono::Utc>,
    /// Checksums
    pub checksums: Vec<(String, String)>, // (algorithm, value)
}

impl PremisObject {
    /// Create a PREMIS object from a file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read
    pub fn from_file(path: &Path, identifier: String) -> Result<Self> {
        let metadata = fs::metadata(path)?;
        let size = metadata.len();
        let original_name = path.file_name().and_then(|n| n.to_str()).map(String::from);

        // Detect format from extension
        let format = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| format!("fmt/{}", e.to_uppercase()));

        Ok(Self {
            identifier,
            object_type: ObjectType::File,
            original_name,
            size: Some(size),
            format,
            creation_date: chrono::Utc::now(),
            checksums: Vec::new(),
        })
    }

    /// Add a checksum
    #[must_use]
    pub fn with_checksum(mut self, algorithm: &str, value: &str) -> Self {
        self.checksums
            .push((algorithm.to_string(), value.to_string()));
        self
    }

    /// Convert to XML
    ///
    /// # Errors
    ///
    /// Returns an error if XML serialization fails
    pub fn to_xml(&self) -> Result<String> {
        let mut xml = String::new();
        xml.push_str("  <object>\n");
        xml.push_str(&format!(
            "    <objectIdentifier>{}</objectIdentifier>\n",
            escape_xml(&self.identifier)
        ));
        xml.push_str(&format!(
            "    <objectCategory>{:?}</objectCategory>\n",
            self.object_type
        ));

        if let Some(ref name) = self.original_name {
            xml.push_str(&format!(
                "    <originalName>{}</originalName>\n",
                escape_xml(name)
            ));
        }

        if let Some(size) = self.size {
            xml.push_str(&format!("    <size>{size}</size>\n"));
        }

        if let Some(ref format) = self.format {
            xml.push_str(&format!("    <format>{}</format>\n", escape_xml(format)));
        }

        for (algo, value) in &self.checksums {
            xml.push_str("    <fixity>\n");
            xml.push_str(&format!(
                "      <messageDigestAlgorithm>{}</messageDigestAlgorithm>\n",
                escape_xml(algo)
            ));
            xml.push_str(&format!(
                "      <messageDigest>{}</messageDigest>\n",
                escape_xml(value)
            ));
            xml.push_str("    </fixity>\n");
        }

        xml.push_str("  </object>\n");
        Ok(xml)
    }

    /// Write this object as a `<premis:object>` element using the zero-copy
    /// `quick-xml` streaming API (see the non-streaming counterpart [`Self::to_xml`]).
    ///
    /// # Errors
    ///
    /// Returns an error if writing to the underlying writer fails.
    pub fn write_streaming<W: IoWrite>(&self, writer: &mut Writer<W>) -> Result<()> {
        writer.write_event(Event::Start(BytesStart::new("object")))?;

        writer.write_event(Event::Start(BytesStart::new("objectIdentifier")))?;
        writer.write_event(Event::Text(BytesText::new(&self.identifier)))?;
        writer.write_event(Event::End(BytesEnd::new("objectIdentifier")))?;

        let category = format!("{:?}", self.object_type);
        writer.write_event(Event::Start(BytesStart::new("objectCategory")))?;
        writer.write_event(Event::Text(BytesText::new(&category)))?;
        writer.write_event(Event::End(BytesEnd::new("objectCategory")))?;

        if let Some(ref name) = self.original_name {
            writer.write_event(Event::Start(BytesStart::new("originalName")))?;
            writer.write_event(Event::Text(BytesText::new(name)))?;
            writer.write_event(Event::End(BytesEnd::new("originalName")))?;
        }

        if let Some(size) = self.size {
            let size_str = size.to_string();
            writer.write_event(Event::Start(BytesStart::new("size")))?;
            writer.write_event(Event::Text(BytesText::new(&size_str)))?;
            writer.write_event(Event::End(BytesEnd::new("size")))?;
        }

        if let Some(ref format) = self.format {
            writer.write_event(Event::Start(BytesStart::new("format")))?;
            writer.write_event(Event::Text(BytesText::new(format)))?;
            writer.write_event(Event::End(BytesEnd::new("format")))?;
        }

        for (algo, value) in &self.checksums {
            writer.write_event(Event::Start(BytesStart::new("fixity")))?;

            writer.write_event(Event::Start(BytesStart::new("messageDigestAlgorithm")))?;
            writer.write_event(Event::Text(BytesText::new(algo)))?;
            writer.write_event(Event::End(BytesEnd::new("messageDigestAlgorithm")))?;

            writer.write_event(Event::Start(BytesStart::new("messageDigest")))?;
            writer.write_event(Event::Text(BytesText::new(value)))?;
            writer.write_event(Event::End(BytesEnd::new("messageDigest")))?;

            writer.write_event(Event::End(BytesEnd::new("fixity")))?;
        }

        writer.write_event(Event::End(BytesEnd::new("object")))?;
        Ok(())
    }
}

/// PREMIS event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PremisEvent {
    /// Event identifier
    pub identifier: String,
    /// Event type
    pub event_type: EventType,
    /// Event date/time
    pub date_time: chrono::DateTime<chrono::Utc>,
    /// Event detail
    pub detail: Option<String>,
    /// Outcome
    pub outcome: Option<String>,
    /// Linking objects
    pub linking_objects: Vec<String>,
}

impl PremisEvent {
    /// Create a new PREMIS event
    #[must_use]
    pub fn new(identifier: String, event_type: EventType) -> Self {
        Self {
            identifier,
            event_type,
            date_time: chrono::Utc::now(),
            detail: None,
            outcome: None,
            linking_objects: Vec::new(),
        }
    }

    /// Set event detail
    #[must_use]
    pub fn with_detail(mut self, detail: &str) -> Self {
        self.detail = Some(detail.to_string());
        self
    }

    /// Set outcome
    #[must_use]
    pub fn with_outcome(mut self, outcome: &str) -> Self {
        self.outcome = Some(outcome.to_string());
        self
    }

    /// Add linking object
    #[must_use]
    pub fn with_linking_object(mut self, object_id: &str) -> Self {
        self.linking_objects.push(object_id.to_string());
        self
    }

    /// Convert to XML
    ///
    /// # Errors
    ///
    /// Returns an error if XML serialization fails
    pub fn to_xml(&self) -> Result<String> {
        let mut xml = String::new();
        xml.push_str("  <event>\n");
        xml.push_str(&format!(
            "    <eventIdentifier>{}</eventIdentifier>\n",
            escape_xml(&self.identifier)
        ));
        xml.push_str(&format!(
            "    <eventType>{}</eventType>\n",
            event_type_to_string(&self.event_type)
        ));
        xml.push_str(&format!(
            "    <eventDateTime>{}</eventDateTime>\n",
            self.date_time.to_rfc3339()
        ));

        if let Some(ref detail) = self.detail {
            xml.push_str(&format!(
                "    <eventDetail>{}</eventDetail>\n",
                escape_xml(detail)
            ));
        }

        if let Some(ref outcome) = self.outcome {
            xml.push_str(&"    <eventOutcomeInformation>\n".to_string());
            xml.push_str(&format!(
                "      <eventOutcome>{}</eventOutcome>\n",
                escape_xml(outcome)
            ));
            xml.push_str("    </eventOutcomeInformation>\n");
        }

        for obj_id in &self.linking_objects {
            xml.push_str(&format!(
                "    <linkingObjectIdentifier>{}</linkingObjectIdentifier>\n",
                escape_xml(obj_id)
            ));
        }

        xml.push_str("  </event>\n");
        Ok(xml)
    }

    /// Write this event as a `<premis:event>` element using the zero-copy
    /// `quick-xml` streaming API (see the non-streaming counterpart [`Self::to_xml`]).
    ///
    /// # Errors
    ///
    /// Returns an error if writing to the underlying writer fails.
    pub fn write_streaming<W: IoWrite>(&self, writer: &mut Writer<W>) -> Result<()> {
        writer.write_event(Event::Start(BytesStart::new("event")))?;

        writer.write_event(Event::Start(BytesStart::new("eventIdentifier")))?;
        writer.write_event(Event::Text(BytesText::new(&self.identifier)))?;
        writer.write_event(Event::End(BytesEnd::new("eventIdentifier")))?;

        let event_type_str = event_type_to_string(&self.event_type);
        writer.write_event(Event::Start(BytesStart::new("eventType")))?;
        writer.write_event(Event::Text(BytesText::new(&event_type_str)))?;
        writer.write_event(Event::End(BytesEnd::new("eventType")))?;

        let date_str = self.date_time.to_rfc3339();
        writer.write_event(Event::Start(BytesStart::new("eventDateTime")))?;
        writer.write_event(Event::Text(BytesText::new(&date_str)))?;
        writer.write_event(Event::End(BytesEnd::new("eventDateTime")))?;

        if let Some(ref detail) = self.detail {
            writer.write_event(Event::Start(BytesStart::new("eventDetail")))?;
            writer.write_event(Event::Text(BytesText::new(detail)))?;
            writer.write_event(Event::End(BytesEnd::new("eventDetail")))?;
        }

        if let Some(ref outcome) = self.outcome {
            writer.write_event(Event::Start(BytesStart::new("eventOutcomeInformation")))?;
            writer.write_event(Event::Start(BytesStart::new("eventOutcome")))?;
            writer.write_event(Event::Text(BytesText::new(outcome)))?;
            writer.write_event(Event::End(BytesEnd::new("eventOutcome")))?;
            writer.write_event(Event::End(BytesEnd::new("eventOutcomeInformation")))?;
        }

        for obj_id in &self.linking_objects {
            writer.write_event(Event::Start(BytesStart::new("linkingObjectIdentifier")))?;
            writer.write_event(Event::Text(BytesText::new(obj_id)))?;
            writer.write_event(Event::End(BytesEnd::new("linkingObjectIdentifier")))?;
        }

        writer.write_event(Event::End(BytesEnd::new("event")))?;
        Ok(())
    }
}

fn event_type_to_string(event_type: &EventType) -> String {
    match event_type {
        EventType::Capture => "capture".to_string(),
        EventType::Ingestion => "ingestion".to_string(),
        EventType::Migration => "migration".to_string(),
        EventType::Validation => "validation".to_string(),
        EventType::FixityCheck => "fixity check".to_string(),
        EventType::Replication => "replication".to_string(),
        EventType::Other(s) => s.clone(),
    }
}

/// PREMIS metadata document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PremisMetadata {
    /// Objects
    pub objects: Vec<PremisObject>,
    /// Events
    pub events: Vec<PremisEvent>,
}

impl Default for PremisMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl PremisMetadata {
    /// Create a new PREMIS metadata document
    #[must_use]
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            events: Vec::new(),
        }
    }

    /// Create PREMIS metadata for a file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read
    pub fn for_file(path: &Path) -> Result<Self> {
        let identifier = format!("obj-{}", chrono::Utc::now().timestamp());
        let object = PremisObject::from_file(path, identifier.clone())?;

        let event = PremisEvent::new(
            format!("evt-{}", chrono::Utc::now().timestamp()),
            EventType::Capture,
        )
        .with_detail("File captured for preservation")
        .with_outcome("success")
        .with_linking_object(&identifier);

        Ok(Self {
            objects: vec![object],
            events: vec![event],
        })
    }

    /// Add an object
    #[must_use]
    pub fn with_object(mut self, object: PremisObject) -> Self {
        self.objects.push(object);
        self
    }

    /// Add an event
    #[must_use]
    pub fn with_event(mut self, event: PremisEvent) -> Self {
        self.events.push(event);
        self
    }

    /// Convert to XML
    ///
    /// # Errors
    ///
    /// Returns an error if XML serialization fails
    pub fn to_xml(&self) -> Result<String> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<premis xmlns=\"http://www.loc.gov/premis/v3\" version=\"3.0\">\n");

        for object in &self.objects {
            xml.push_str(&object.to_xml()?);
        }

        for event in &self.events {
            xml.push_str(&event.to_xml()?);
        }

        xml.push_str("</premis>\n");
        Ok(xml)
    }

    /// Save to file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written
    pub fn save(&self, path: &Path) -> Result<()> {
        let xml = self.to_xml()?;
        fs::write(path, xml)?;
        Ok(())
    }
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// ─── PREMIS Rights ────────────────────────────────────────────────────────────

/// The basis for a PREMIS rights statement (§1.2.3 of the PREMIS Data Dictionary).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RightsBasis {
    /// Copyright protection
    Copyright,
    /// License agreement
    License,
    /// Statute or regulation
    Statute,
    /// Other / institutional policy
    Other(String),
}

impl RightsBasis {
    /// Human-readable label used in XML serialization.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Copyright => "copyright",
            Self::License => "license",
            Self::Statute => "statute",
            Self::Other(_) => "other",
        }
    }
}

/// Copyright status values following PREMIS controlled vocabulary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CopyrightStatus {
    /// Under copyright protection
    Copyrighted,
    /// In the public domain
    PublicDomain,
    /// Copyright status unknown
    Unknown,
}

impl CopyrightStatus {
    fn label(&self) -> &str {
        match self {
            Self::Copyrighted => "copyrighted",
            Self::PublicDomain => "publicdomain",
            Self::Unknown => "unknown",
        }
    }
}

/// A PREMIS rights statement describing the access permissions for a digital object.
///
/// Implements the `<premis:rights>` element per PREMIS Data Dictionary v3.0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PremisRights {
    /// Unique identifier for this rights statement
    pub rights_statement_id: String,
    /// The basis for this rights statement
    pub rights_basis: RightsBasis,
    /// Copyright status (populated when `rights_basis == Copyright`)
    pub copyright_status: Option<CopyrightStatus>,
    /// Jurisdiction (e.g. `"US"`, `"EU"`)
    pub copyright_jurisdiction: Option<String>,
    /// Copyright determination note (free text)
    pub copyright_note: Option<String>,
    /// License terms (populated when `rights_basis == License`)
    pub license_terms: Option<String>,
    /// License URI (e.g. `"https://creativecommons.org/licenses/by/4.0/"`)
    pub license_uri: Option<String>,
    /// Start date of the rights grant (`YYYY-MM-DD`)
    pub start_date: Option<String>,
    /// End date of the rights grant (`YYYY-MM-DD` or `"open"` for perpetual)
    pub end_date: Option<String>,
    /// Rights granted (e.g. `["disseminate", "reproduce"]`)
    pub acts_granted: Vec<String>,
    /// Restrictions (free text)
    pub restriction_note: Option<String>,
}

impl PremisRights {
    /// Create a new rights statement with the given identifier and basis.
    #[must_use]
    pub fn new(rights_statement_id: impl Into<String>, rights_basis: RightsBasis) -> Self {
        Self {
            rights_statement_id: rights_statement_id.into(),
            rights_basis,
            copyright_status: None,
            copyright_jurisdiction: None,
            copyright_note: None,
            license_terms: None,
            license_uri: None,
            start_date: None,
            end_date: None,
            acts_granted: Vec::new(),
            restriction_note: None,
        }
    }

    /// Set copyright status.
    #[must_use]
    pub fn with_copyright_status(mut self, status: CopyrightStatus) -> Self {
        self.copyright_status = Some(status);
        self
    }

    /// Set copyright jurisdiction.
    #[must_use]
    pub fn with_copyright_jurisdiction(mut self, jurisdiction: impl Into<String>) -> Self {
        self.copyright_jurisdiction = Some(jurisdiction.into());
        self
    }

    /// Set copyright note.
    #[must_use]
    pub fn with_copyright_note(mut self, note: impl Into<String>) -> Self {
        self.copyright_note = Some(note.into());
        self
    }

    /// Set license terms.
    #[must_use]
    pub fn with_license_terms(mut self, terms: impl Into<String>) -> Self {
        self.license_terms = Some(terms.into());
        self
    }

    /// Set license URI.
    #[must_use]
    pub fn with_license_uri(mut self, uri: impl Into<String>) -> Self {
        self.license_uri = Some(uri.into());
        self
    }

    /// Set rights start date (`YYYY-MM-DD`).
    #[must_use]
    pub fn with_start_date(mut self, date: impl Into<String>) -> Self {
        self.start_date = Some(date.into());
        self
    }

    /// Set rights end date (`YYYY-MM-DD` or `"open"`).
    #[must_use]
    pub fn with_end_date(mut self, date: impl Into<String>) -> Self {
        self.end_date = Some(date.into());
        self
    }

    /// Add an act granted (e.g. `"disseminate"`, `"reproduce"`, `"publish"`).
    #[must_use]
    pub fn with_act_granted(mut self, act: impl Into<String>) -> Self {
        self.acts_granted.push(act.into());
        self
    }

    /// Set a restriction note.
    #[must_use]
    pub fn with_restriction_note(mut self, note: impl Into<String>) -> Self {
        self.restriction_note = Some(note.into());
        self
    }

    /// Serialize this rights statement to a PREMIS v3 `<premis:rights>` XML fragment.
    ///
    /// # Errors
    ///
    /// Returns an error if string formatting fails (infallible in practice).
    pub fn to_xml(&self) -> Result<String> {
        let mut xml = String::new();
        xml.push_str("  <rights>\n");
        xml.push_str("    <rightsStatement>\n");
        xml.push_str(&format!(
            "      <rightsStatementIdentifier>{}</rightsStatementIdentifier>\n",
            escape_xml(&self.rights_statement_id)
        ));
        xml.push_str(&format!(
            "      <rightsBasis>{}</rightsBasis>\n",
            self.rights_basis.label()
        ));

        // Copyright sub-element
        if self.rights_basis == RightsBasis::Copyright {
            xml.push_str("      <copyrightInformation>\n");
            if let Some(ref status) = self.copyright_status {
                xml.push_str(&format!(
                    "        <copyrightStatus>{}</copyrightStatus>\n",
                    status.label()
                ));
            }
            if let Some(ref jur) = self.copyright_jurisdiction {
                xml.push_str(&format!(
                    "        <copyrightJurisdiction>{}</copyrightJurisdiction>\n",
                    escape_xml(jur)
                ));
            }
            if let Some(ref note) = self.copyright_note {
                xml.push_str(&format!(
                    "        <copyrightNote>{}</copyrightNote>\n",
                    escape_xml(note)
                ));
            }
            xml.push_str("      </copyrightInformation>\n");
        }

        // License sub-element
        if self.rights_basis == RightsBasis::License {
            xml.push_str("      <licenseInformation>\n");
            if let Some(ref uri) = self.license_uri {
                xml.push_str(&format!(
                    "        <licenseIdentifier>{}</licenseIdentifier>\n",
                    escape_xml(uri)
                ));
            }
            if let Some(ref terms) = self.license_terms {
                xml.push_str(&format!(
                    "        <licenseTerms>{}</licenseTerms>\n",
                    escape_xml(terms)
                ));
            }
            xml.push_str("      </licenseInformation>\n");
        }

        // Other basis note
        if let RightsBasis::Other(ref other_note) = self.rights_basis {
            xml.push_str("      <otherRightsInformation>\n");
            xml.push_str(&format!(
                "        <otherRightsBasis>{}</otherRightsBasis>\n",
                escape_xml(other_note)
            ));
            xml.push_str("      </otherRightsInformation>\n");
        }

        // Rights granted
        for act in &self.acts_granted {
            xml.push_str("      <rightsGranted>\n");
            xml.push_str(&format!("        <act>{}</act>\n", escape_xml(act)));
            if let Some(ref sd) = self.start_date {
                xml.push_str(&format!(
                    "        <termOfGrant><startDate>{}</startDate>",
                    escape_xml(sd)
                ));
                if let Some(ref ed) = self.end_date {
                    xml.push_str(&format!("<endDate>{}</endDate>", escape_xml(ed)));
                }
                xml.push_str("</termOfGrant>\n");
            }
            if let Some(ref note) = self.restriction_note {
                xml.push_str(&format!(
                    "        <rightsGrantedNote>{}</rightsGrantedNote>\n",
                    escape_xml(note)
                ));
            }
            xml.push_str("      </rightsGranted>\n");
        }

        xml.push_str("    </rightsStatement>\n");
        xml.push_str("  </rights>\n");
        Ok(xml)
    }

    /// Serialize a full PREMIS XML document containing only this rights statement.
    ///
    /// # Errors
    ///
    /// Returns an error if XML serialization fails.
    pub fn to_premis_xml(&self) -> Result<String> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<premis xmlns=\"http://www.loc.gov/premis/v3\" version=\"3.0\">\n");
        xml.push_str(&self.to_xml()?);
        xml.push_str("</premis>\n");
        Ok(xml)
    }
}

/// Extend `PremisMetadata` to carry rights statements.
///
/// This newtype wraps `PremisMetadata` and adds a `rights` field, providing
/// full-document XML serialization via `to_xml_with_rights`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PremisMetadataWithRights {
    /// Core PREMIS metadata (objects + events).
    pub premis: PremisMetadata,
    /// Rights statements.
    pub rights: Vec<PremisRights>,
}

impl PremisMetadataWithRights {
    /// Create a new wrapper.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a rights statement.
    #[must_use]
    pub fn with_rights(mut self, rights: PremisRights) -> Self {
        self.rights.push(rights);
        self
    }

    /// Serialize the full PREMIS document including rights statements.
    ///
    /// # Errors
    ///
    /// Returns an error if XML serialization fails.
    pub fn to_xml(&self) -> Result<String> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<premis xmlns=\"http://www.loc.gov/premis/v3\" version=\"3.0\">\n");

        for object in &self.premis.objects {
            xml.push_str(&object.to_xml()?);
        }
        for event in &self.premis.events {
            xml.push_str(&event.to_xml()?);
        }
        for rights in &self.rights {
            xml.push_str(&rights.to_xml()?);
        }

        xml.push_str("</premis>\n");
        Ok(xml)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_premis_object_creation() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Test content")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let object = PremisObject::from_file(file.path(), "obj-001".to_string())
            .expect("operation should succeed")
            .with_checksum("SHA-256", "abc123");

        assert_eq!(object.identifier, "obj-001");
        assert_eq!(object.object_type, ObjectType::File);
        assert_eq!(object.checksums.len(), 1);
    }

    #[test]
    fn test_premis_event_creation() {
        let event = PremisEvent::new("evt-001".to_string(), EventType::Ingestion)
            .with_detail("Ingested into archive")
            .with_outcome("success")
            .with_linking_object("obj-001");

        assert_eq!(event.identifier, "evt-001");
        assert_eq!(event.event_type, EventType::Ingestion);
        assert_eq!(event.linking_objects.len(), 1);
    }

    #[test]
    fn test_premis_metadata_xml() {
        let metadata = PremisMetadata::new().with_object(PremisObject {
            identifier: "obj-001".to_string(),
            object_type: ObjectType::File,
            original_name: Some("test.mkv".to_string()),
            size: Some(1024),
            format: Some("video/x-matroska".to_string()),
            creation_date: chrono::Utc::now(),
            checksums: vec![("SHA-256".to_string(), "abc123".to_string())],
        });

        let xml = metadata.to_xml().expect("operation should succeed");
        assert!(xml.contains("<premis"));
        assert!(xml.contains("obj-001"));
        assert!(xml.contains("SHA-256"));
    }

    #[test]
    fn test_for_file() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Preservation test")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let metadata = PremisMetadata::for_file(file.path()).expect("operation should succeed");
        assert_eq!(metadata.objects.len(), 1);
        assert_eq!(metadata.events.len(), 1);
    }

    // ── PremisRights tests ────────────────────────────────────────────────────

    #[test]
    fn test_premis_rights_copyright_xml() {
        let rights = PremisRights::new("rights-001", RightsBasis::Copyright)
            .with_copyright_status(CopyrightStatus::Copyrighted)
            .with_copyright_jurisdiction("US")
            .with_copyright_note("Copyright 2025 COOLJAPAN OU")
            .with_act_granted("disseminate")
            .with_start_date("2025-01-01")
            .with_end_date("open");

        let xml = rights.to_xml().expect("XML serialization should succeed");
        assert!(xml.contains("<rightsBasis>copyright</rightsBasis>"));
        assert!(xml.contains("<copyrightStatus>copyrighted</copyrightStatus>"));
        assert!(xml.contains("<copyrightJurisdiction>US</copyrightJurisdiction>"));
        assert!(xml.contains("Copyright 2025 COOLJAPAN OU"));
        assert!(xml.contains("<act>disseminate</act>"));
        assert!(xml.contains("<startDate>2025-01-01</startDate>"));
        assert!(xml.contains("<endDate>open</endDate>"));
    }

    #[test]
    fn test_premis_rights_license_xml() {
        let rights = PremisRights::new("rights-002", RightsBasis::License)
            .with_license_terms("Attribution 4.0 International")
            .with_license_uri("https://creativecommons.org/licenses/by/4.0/")
            .with_act_granted("reproduce")
            .with_act_granted("publish");

        let xml = rights.to_xml().expect("XML serialization should succeed");
        assert!(xml.contains("<rightsBasis>license</rightsBasis>"));
        assert!(xml.contains("creativecommons.org"));
        assert!(xml.contains("Attribution 4.0 International"));
        assert!(xml.contains("<act>reproduce</act>"));
        assert!(xml.contains("<act>publish</act>"));
    }

    #[test]
    fn test_premis_rights_other_xml() {
        let rights = PremisRights::new(
            "rights-003",
            RightsBasis::Other("institutional policy".to_string()),
        )
        .with_restriction_note("Internal use only");

        let xml = rights.to_xml().expect("XML serialization should succeed");
        assert!(xml.contains("<rightsBasis>other</rightsBasis>"));
        assert!(xml.contains("institutional policy"));
    }

    #[test]
    fn test_premis_rights_full_premis_document() {
        let rights = PremisRights::new("rights-004", RightsBasis::License)
            .with_license_uri("https://creativecommons.org/publicdomain/zero/1.0/")
            .with_license_terms("CC0 1.0 Universal");

        let xml = rights
            .to_premis_xml()
            .expect("XML serialization should succeed");
        assert!(xml.contains("<?xml version=\"1.0\""));
        assert!(xml.contains("<premis xmlns=\"http://www.loc.gov/premis/v3\""));
        assert!(xml.contains("</premis>"));
        assert!(xml.contains("rights-004"));
    }

    #[test]
    fn test_premis_metadata_with_rights() {
        let premis_core = PremisMetadata::new().with_object(PremisObject {
            identifier: "obj-100".to_string(),
            object_type: ObjectType::File,
            original_name: Some("archive.mkv".to_string()),
            size: Some(2048),
            format: Some("video/x-matroska".to_string()),
            creation_date: chrono::Utc::now(),
            checksums: Vec::new(),
        });

        let rights = PremisRights::new("rights-100", RightsBasis::Copyright)
            .with_copyright_status(CopyrightStatus::PublicDomain);

        let doc = PremisMetadataWithRights {
            premis: premis_core,
            rights: vec![rights],
        };

        let xml = doc.to_xml().expect("serialization should succeed");
        assert!(xml.contains("obj-100"));
        assert!(xml.contains("<rightsBasis>copyright</rightsBasis>"));
        assert!(xml.contains("<copyrightStatus>publicdomain</copyrightStatus>"));
    }

    #[test]
    fn test_rights_basis_label() {
        assert_eq!(RightsBasis::Copyright.label(), "copyright");
        assert_eq!(RightsBasis::License.label(), "license");
        assert_eq!(RightsBasis::Statute.label(), "statute");
        assert_eq!(RightsBasis::Other("custom".to_string()).label(), "other");
    }
}
