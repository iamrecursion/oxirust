//! Rights data export to industry-standard formats.
//!
//! Exports rights records to three interchange formats:
//!
//! | Format | Authority | Use |
//! |--------|-----------|-----|
//! | **EIDR** | Entertainment Identifier Registry | Film & TV metadata |
//! | **DDEX** | Digital Data Exchange | Music / digital distribution |
//! | **CWR** | Common Works Registration | Musical work registration |
//!
//! This module generates human-readable / machine-parseable text representations
//! of rights data.  For production use the output should be further validated
//! against the relevant official schemas.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

use crate::rights_import::ImportedRight;
use crate::Result;

// ── ExportFormat ──────────────────────────────────────────────────────────────

/// Supported export formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// EIDR simplified XML representation.
    Eidr,
    /// DDEX ERN (Electronic Release Notification) simplified XML.
    Ddex,
    /// CWR (Common Works Registration) fixed-width text.
    Cwr,
}

impl ExportFormat {
    /// File extension typically used for this format.
    #[must_use]
    pub fn file_extension(&self) -> &'static str {
        match self {
            Self::Eidr => "xml",
            Self::Ddex => "xml",
            Self::Cwr => "cwr",
        }
    }

    /// MIME type for this format.
    #[must_use]
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Eidr | Self::Ddex => "application/xml",
            Self::Cwr => "text/plain",
        }
    }
}

// ── ExportRecord ──────────────────────────────────────────────────────────────

/// A rights record prepared for export, with optional extra fields used by
/// specific formats.
#[derive(Debug, Clone)]
pub struct ExportRecord {
    /// Source record.
    pub right: ImportedRight,
    /// EIDR identifier (if known).
    pub eidr_id: Option<String>,
    /// ISRC code (if applicable; for audio works).
    pub isrc: Option<String>,
    /// ISWC code (if applicable; for musical compositions).
    pub iswc: Option<String>,
    /// Territory codes (ISO 3166-1 alpha-2, empty = worldwide).
    pub territories: Vec<String>,
    /// Rights type description (e.g. "PerformingRights", "MechanicalRights").
    pub rights_type: String,
}

impl ExportRecord {
    /// Create an export record from a base import record.
    #[must_use]
    pub fn from_right(right: ImportedRight) -> Self {
        Self {
            right,
            eidr_id: None,
            isrc: None,
            iswc: None,
            territories: Vec::new(),
            rights_type: "GeneralRights".to_string(),
        }
    }

    /// Builder: set EIDR ID.
    #[must_use]
    pub fn with_eidr(mut self, eidr: impl Into<String>) -> Self {
        self.eidr_id = Some(eidr.into());
        self
    }

    /// Builder: set ISRC.
    #[must_use]
    pub fn with_isrc(mut self, isrc: impl Into<String>) -> Self {
        self.isrc = Some(isrc.into());
        self
    }

    /// Builder: set ISWC.
    #[must_use]
    pub fn with_iswc(mut self, iswc: impl Into<String>) -> Self {
        self.iswc = Some(iswc.into());
        self
    }

    /// Builder: add a territory.
    #[must_use]
    pub fn with_territory(mut self, code: impl Into<String>) -> Self {
        self.territories.push(code.into().to_uppercase());
        self
    }

    /// Builder: set rights type.
    #[must_use]
    pub fn with_rights_type(mut self, rtype: impl Into<String>) -> Self {
        self.rights_type = rtype.into();
        self
    }
}

// ── RightsExporter ────────────────────────────────────────────────────────────

/// Converts [`ExportRecord`]s to industry-standard text formats.
#[derive(Debug, Default)]
pub struct RightsExporter;

impl RightsExporter {
    /// Create a new exporter.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Export a slice of records in the specified format.
    ///
    /// # Errors
    /// Returns `RightsError::Serialization` if the format-specific serialisation
    /// encounters an unrecoverable error.
    pub fn export(&self, records: &[ExportRecord], format: ExportFormat) -> Result<String> {
        match format {
            ExportFormat::Eidr => self.export_eidr(records),
            ExportFormat::Ddex => self.export_ddex(records),
            ExportFormat::Cwr => self.export_cwr(records),
        }
    }

    // ── EIDR ──────────────────────────────────────────────────────────────────

    fn export_eidr(&self, records: &[ExportRecord]) -> Result<String> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<EIDRRegistration xmlns=\"urn:eidr:reg\">\n");

        for rec in records {
            xml.push_str("  <ContentObject>\n");
            xml.push_str(&format!(
                "    <RecordID>{}</RecordID>\n",
                escape_xml(&rec.right.record_id)
            ));
            xml.push_str(&format!(
                "    <AssetID>{}</AssetID>\n",
                escape_xml(&rec.right.asset_id)
            ));
            if let Some(eidr) = &rec.eidr_id {
                xml.push_str(&format!("    <EIDR>{}</EIDR>\n", escape_xml(eidr)));
            }
            xml.push_str(&format!(
                "    <RightsHolder>{}</RightsHolder>\n",
                escape_xml(&rec.right.holder)
            ));
            xml.push_str(&format!("    <Active>{}</Active>\n", rec.right.active));
            xml.push_str(&format!(
                "    <GrantedAt>{}</GrantedAt>\n",
                rec.right.granted_at
            ));
            if let Some(exp) = rec.right.expires_at {
                xml.push_str(&format!("    <ExpiresAt>{exp}</ExpiresAt>\n"));
            }
            xml.push_str(&format!(
                "    <RightsType>{}</RightsType>\n",
                escape_xml(&rec.rights_type)
            ));
            if !rec.territories.is_empty() {
                xml.push_str("    <Territories>\n");
                for t in &rec.territories {
                    xml.push_str(&format!("      <Territory>{}</Territory>\n", escape_xml(t)));
                }
                xml.push_str("    </Territories>\n");
            }
            xml.push_str("  </ContentObject>\n");
        }

        xml.push_str("</EIDRRegistration>\n");
        Ok(xml)
    }

    // ── DDEX ──────────────────────────────────────────────────────────────────

    fn export_ddex(&self, records: &[ExportRecord]) -> Result<String> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<NewReleaseMessage xmlns=\"http://ddex.net/xml/ern/43\">\n");
        xml.push_str("  <MessageHeader>\n");
        xml.push_str("    <MessageSchemaVersionId>43</MessageSchemaVersionId>\n");
        xml.push_str("    <MessageSender><PartyId>OxiMedia</PartyId></MessageSender>\n");
        xml.push_str("  </MessageHeader>\n");
        xml.push_str("  <ResourceList>\n");

        for rec in records {
            xml.push_str("    <SoundRecording>\n");
            xml.push_str(&format!(
                "      <SoundRecordingId>\n        <ProprietaryId>{}</ProprietaryId>\n      </SoundRecordingId>\n",
                escape_xml(&rec.right.record_id)
            ));
            if let Some(isrc) = &rec.isrc {
                xml.push_str(&format!("      <ISRC>{}</ISRC>\n", escape_xml(isrc)));
            }
            xml.push_str(&format!(
                "      <ResourceReference>{}</ResourceReference>\n",
                escape_xml(&rec.right.asset_id)
            ));
            xml.push_str(&format!(
                "      <RightsController><PartyName><FullName>{}</FullName></PartyName></RightsController>\n",
                escape_xml(&rec.right.holder)
            ));
            if let Some(exp) = rec.right.expires_at {
                xml.push_str(&format!(
                    "      <ValidityPeriod><EndDate>{exp}</EndDate></ValidityPeriod>\n"
                ));
            }
            if !rec.territories.is_empty() {
                let territory_list = rec.territories.join(" ");
                xml.push_str(&format!(
                    "      <TerritoryCode>{territory_list}</TerritoryCode>\n"
                ));
            }
            xml.push_str("    </SoundRecording>\n");
        }

        xml.push_str("  </ResourceList>\n");
        xml.push_str("</NewReleaseMessage>\n");
        Ok(xml)
    }

    // ── CWR ───────────────────────────────────────────────────────────────────
    //
    // CWR uses 3-character record type codes and fixed-width fields.
    // This implementation produces a simplified valid CWR-2.2 format.

    fn export_cwr(&self, records: &[ExportRecord]) -> Result<String> {
        let mut lines = Vec::new();

        // HDR (transmission header)
        lines.push(format!(
            "HDR{:9}{:45}{:30}{:8}",
            "OXIMEDIA", "OxiMedia Rights Export", "OXIMEDIA", "20240101",
        ));

        for (idx, rec) in records.iter().enumerate() {
            let seq = idx + 1;

            // NWR – New Work Registration
            let iswc = rec.iswc.as_deref().unwrap_or("           "); // 11-char ISWC field
            let title = pad_right(&rec.right.asset_id, 60);
            let language = "EN";
            lines.push(format!("NWR{seq:08}{title}{iswc:11}{language}"));

            // SPU – Publisher controlled by submitter
            let holder = pad_right(&rec.right.holder, 45);
            lines.push(format!("SPU{seq:08}{holder}"));

            // PWR – publisher for writer
            if let Some(isrc) = &rec.isrc {
                lines.push(format!("PWR{seq:08}{}", pad_right(isrc, 12)));
            }

            // TER – territory
            let territory = if rec.territories.is_empty() {
                "WW".to_string()
            } else {
                rec.territories.join(",")
            };
            lines.push(format!("TER{seq:08}{territory}"));
        }

        // TRL (transmission trailer)
        lines.push(format!("TRL{:08}", records.len()));

        Ok(lines.join("\n") + "\n")
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Escape XML special characters.
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Right-pad a string to exactly `width` characters, truncating if necessary.
fn pad_right(s: &str, width: usize) -> String {
    if s.len() >= width {
        s[..width].to_string()
    } else {
        format!("{:width$}", s, width = width)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rights_import::ImportedRight;

    fn sample_record(id: &str, asset: &str, holder: &str) -> ExportRecord {
        ExportRecord::from_right(ImportedRight::new(id, asset, holder, 1_000_000))
            .with_eidr("10.5240/0000-0000-0000-0000-0000-C")
            .with_isrc("US-ABC-23-00001")
            .with_iswc("T-034.524.680-1")
            .with_territory("US")
            .with_territory("GB")
            .with_rights_type("PerformingRights")
    }

    #[test]
    fn test_eidr_export_contains_record_id() {
        let exporter = RightsExporter::new();
        let recs = vec![sample_record("rec-1", "asset-A", "Alice Corp")];
        let xml = exporter.export(&recs, ExportFormat::Eidr).expect("export");
        assert!(xml.contains("rec-1"));
        assert!(xml.contains("Alice Corp"));
        assert!(xml.contains("PerformingRights"));
        assert!(xml.contains("EIDRRegistration"));
    }

    #[test]
    fn test_eidr_export_territories() {
        let exporter = RightsExporter::new();
        let recs = vec![sample_record("r1", "a", "h")];
        let xml = exporter.export(&recs, ExportFormat::Eidr).expect("export");
        assert!(xml.contains("<Territory>US</Territory>"));
        assert!(xml.contains("<Territory>GB</Territory>"));
    }

    #[test]
    fn test_eidr_export_eidr_id() {
        let exporter = RightsExporter::new();
        let recs = vec![sample_record("r1", "a", "h")];
        let xml = exporter.export(&recs, ExportFormat::Eidr).expect("export");
        assert!(xml.contains("10.5240/0000-0000-0000-0000-0000-C"));
    }

    #[test]
    fn test_eidr_export_with_expiry() {
        let exporter = RightsExporter::new();
        let mut rec = sample_record("r1", "a", "h");
        rec.right.expires_at = Some(9_999_999);
        let xml = exporter.export(&[rec], ExportFormat::Eidr).expect("export");
        assert!(xml.contains("<ExpiresAt>9999999</ExpiresAt>"));
    }

    #[test]
    fn test_ddex_export_structure() {
        let exporter = RightsExporter::new();
        let recs = vec![sample_record("rec-D", "asset-D", "DDEXHolder")];
        let xml = exporter.export(&recs, ExportFormat::Ddex).expect("export");
        assert!(xml.contains("NewReleaseMessage"));
        assert!(xml.contains("DDEXHolder"));
        assert!(xml.contains("US-ABC-23-00001"));
    }

    #[test]
    fn test_ddex_export_territory() {
        let exporter = RightsExporter::new();
        let recs = vec![sample_record("r", "a", "h")];
        let xml = exporter.export(&recs, ExportFormat::Ddex).expect("export");
        assert!(xml.contains("TerritoryCode"));
        assert!(xml.contains("US"));
    }

    #[test]
    fn test_cwr_export_has_hdr_and_trl() {
        let exporter = RightsExporter::new();
        let recs = vec![sample_record("r1", "asset-C", "Publisher A")];
        let cwr = exporter.export(&recs, ExportFormat::Cwr).expect("export");
        assert!(cwr.contains("HDR"));
        assert!(cwr.contains("TRL"));
    }

    #[test]
    fn test_cwr_export_has_nwr() {
        let exporter = RightsExporter::new();
        let recs = vec![sample_record("r1", "asset-C", "Publisher A")];
        let cwr = exporter.export(&recs, ExportFormat::Cwr).expect("export");
        assert!(cwr.contains("NWR"));
    }

    #[test]
    fn test_cwr_export_multiple_records() {
        let exporter = RightsExporter::new();
        let recs = vec![
            sample_record("r1", "a1", "h1"),
            sample_record("r2", "a2", "h2"),
        ];
        let cwr = exporter.export(&recs, ExportFormat::Cwr).expect("export");
        // Two NWR lines
        assert_eq!(cwr.matches("NWR").count(), 2);
    }

    #[test]
    fn test_export_format_extension() {
        assert_eq!(ExportFormat::Eidr.file_extension(), "xml");
        assert_eq!(ExportFormat::Ddex.file_extension(), "xml");
        assert_eq!(ExportFormat::Cwr.file_extension(), "cwr");
    }

    #[test]
    fn test_export_format_mime_type() {
        assert_eq!(ExportFormat::Eidr.mime_type(), "application/xml");
        assert_eq!(ExportFormat::Cwr.mime_type(), "text/plain");
    }

    #[test]
    fn test_escape_xml_ampersand() {
        assert_eq!(escape_xml("A&B"), "A&amp;B");
    }

    #[test]
    fn test_escape_xml_angle_brackets() {
        assert_eq!(escape_xml("<tag>"), "&lt;tag&gt;");
    }

    #[test]
    fn test_export_empty_records() {
        let exporter = RightsExporter::new();
        let xml = exporter
            .export(&[], ExportFormat::Eidr)
            .expect("empty export");
        assert!(xml.contains("EIDRRegistration"));
    }

    #[test]
    fn test_export_record_builder() {
        let rec = ExportRecord::from_right(ImportedRight::new("r", "a", "h", 0))
            .with_eidr("10.5240/test")
            .with_isrc("USABC0000001")
            .with_iswc("T-123.456.789-0")
            .with_territory("DE")
            .with_rights_type("MechanicalRights");
        assert_eq!(rec.eidr_id.as_deref(), Some("10.5240/test"));
        assert_eq!(rec.territories, vec!["DE"]);
        assert_eq!(rec.rights_type, "MechanicalRights");
    }
}
