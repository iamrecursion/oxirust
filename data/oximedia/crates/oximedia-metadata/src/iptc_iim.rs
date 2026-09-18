//! IPTC IIM (Information Interchange Model) structured metadata types.
//!
//! Provides typed structs for working with IPTC datasets, records, and builder patterns.
//! Includes:
//! - `IptcPhotoMeta` — high-level photo metadata (headline, caption, credit, source, etc.)
//! - `encode_iptc_iim` / `parse_iptc_iim` — binary IIM encode/decode with roundtrip support
//! - `XmpIptcBridge` — convert IPTC metadata to XMP/RDF with photoshop: and dc: namespaces

use crate::Error;

/// An individual IPTC dataset entry consisting of record number, dataset number, and raw data.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct IptcDataset {
    /// IPTC record number (1 = Envelope, 2 = Application, 3 = NewsPhoto).
    pub record: u8,
    /// IPTC dataset number within the record.
    pub dataset: u8,
    /// Raw dataset data bytes.
    pub data: Vec<u8>,
}

impl IptcDataset {
    /// Create a new IPTC dataset.
    #[allow(dead_code)]
    pub fn new(record: u8, dataset: u8, data: Vec<u8>) -> Self {
        Self {
            record,
            dataset,
            data,
        }
    }

    /// Return the (record, dataset) tag pair.
    #[allow(dead_code)]
    pub fn tag(&self) -> (u8, u8) {
        (self.record, self.dataset)
    }

    /// Attempt to decode the dataset data as a UTF-8 string.
    #[allow(dead_code)]
    pub fn as_string(&self) -> Option<String> {
        String::from_utf8(self.data.clone()).ok()
    }
}

/// IPTC record type identifiers.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IptcRecord {
    /// Record 1: Envelope record (routing/service information).
    EnvelopeRecord,
    /// Record 2: Application record (editorial/content metadata).
    ApplicationRecord,
    /// Record 3: News photo record.
    NewsPhotoRecord,
}

impl IptcRecord {
    /// Return the numeric record number for this record type.
    #[allow(dead_code)]
    pub fn record_number(self) -> u8 {
        match self {
            Self::EnvelopeRecord => 1,
            Self::ApplicationRecord => 2,
            Self::NewsPhotoRecord => 3,
        }
    }
}

/// Structured IPTC Application Record (Record 2) metadata.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct IptcApplicationRecord {
    /// Dataset 2:105 - Headline.
    pub headline: Option<String>,
    /// Dataset 2:120 - Caption/Abstract.
    pub caption: Option<String>,
    /// Dataset 2:25 - Keywords (repeatable).
    pub keywords: Vec<String>,
    /// Dataset 2:80 - By-line (photographer/creator).
    pub byline: Option<String>,
    /// Dataset 2:90 - City.
    pub city: Option<String>,
    /// Dataset 2:101 - Country.
    pub country: Option<String>,
    /// Dataset 2:116 - Copyright notice.
    pub copyright: Option<String>,
    /// Dataset 2:10 - Urgency (0-9, where 1 = most urgent).
    pub urgency: u8,
}

impl IptcApplicationRecord {
    /// Serialize this record into a flat list of `IptcDataset` entries.
    #[allow(dead_code)]
    pub fn to_datasets(&self) -> Vec<IptcDataset> {
        let mut datasets = Vec::new();
        let rec = IptcRecord::ApplicationRecord.record_number();

        if let Some(ref h) = self.headline {
            datasets.push(IptcDataset::new(rec, 105, h.as_bytes().to_vec()));
        }
        if let Some(ref c) = self.caption {
            datasets.push(IptcDataset::new(rec, 120, c.as_bytes().to_vec()));
        }
        for kw in &self.keywords {
            datasets.push(IptcDataset::new(rec, 25, kw.as_bytes().to_vec()));
        }
        if let Some(ref b) = self.byline {
            datasets.push(IptcDataset::new(rec, 80, b.as_bytes().to_vec()));
        }
        if let Some(ref c) = self.city {
            datasets.push(IptcDataset::new(rec, 90, c.as_bytes().to_vec()));
        }
        if let Some(ref c) = self.country {
            datasets.push(IptcDataset::new(rec, 101, c.as_bytes().to_vec()));
        }
        if let Some(ref c) = self.copyright {
            datasets.push(IptcDataset::new(rec, 116, c.as_bytes().to_vec()));
        }
        if self.urgency > 0 {
            datasets.push(IptcDataset::new(rec, 10, vec![b'0' + self.urgency.min(9)]));
        }

        datasets
    }

    /// Count the total number of words across headline, caption, and keywords.
    #[allow(dead_code)]
    pub fn word_count(&self) -> usize {
        let mut count = 0;
        if let Some(ref h) = self.headline {
            count += h.split_whitespace().count();
        }
        if let Some(ref c) = self.caption {
            count += c.split_whitespace().count();
        }
        for kw in &self.keywords {
            count += kw.split_whitespace().count();
        }
        count
    }
}

/// Parser for raw IPTC binary data (IIM format with 0x1C tag marker).
#[allow(dead_code)]
pub struct IptcParser;

impl IptcParser {
    /// Parse a byte slice containing IPTC IIM data into a list of datasets.
    ///
    /// Each dataset is expected to be preceded by the 0x1C marker byte, followed
    /// by record number, dataset number, and a 2-byte big-endian length field.
    #[allow(dead_code)]
    pub fn parse_iptc(data: &[u8]) -> Vec<IptcDataset> {
        const MARKER: u8 = 0x1C;
        let mut datasets = Vec::new();
        let mut i = 0;

        while i + 4 < data.len() {
            if data[i] != MARKER {
                i += 1;
                continue;
            }
            let record = data[i + 1];
            let dataset = data[i + 2];
            let length = u16::from_be_bytes([data[i + 3], data[i + 4]]) as usize;
            i += 5;

            if i + length > data.len() {
                break;
            }
            let payload = data[i..i + length].to_vec();
            datasets.push(IptcDataset::new(record, dataset, payload));
            i += length;
        }

        datasets
    }
}

/// Builder for constructing an `IptcApplicationRecord` using a fluent API.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct IptcBuilder {
    record: IptcApplicationRecord,
}

impl IptcBuilder {
    /// Create a new builder with all fields empty.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the headline (dataset 2:105).
    #[allow(dead_code)]
    pub fn headline(mut self, value: impl Into<String>) -> Self {
        self.record.headline = Some(value.into());
        self
    }

    /// Set the caption (dataset 2:120).
    #[allow(dead_code)]
    pub fn caption(mut self, value: impl Into<String>) -> Self {
        self.record.caption = Some(value.into());
        self
    }

    /// Add a keyword (dataset 2:25, repeatable).
    #[allow(dead_code)]
    pub fn add_keyword(mut self, kw: impl Into<String>) -> Self {
        self.record.keywords.push(kw.into());
        self
    }

    /// Set the by-line/creator (dataset 2:80).
    #[allow(dead_code)]
    pub fn byline(mut self, value: impl Into<String>) -> Self {
        self.record.byline = Some(value.into());
        self
    }

    /// Set the copyright notice (dataset 2:116).
    #[allow(dead_code)]
    pub fn copyright(mut self, value: impl Into<String>) -> Self {
        self.record.copyright = Some(value.into());
        self
    }

    /// Set the urgency (dataset 2:10, clamped 1-9).
    #[allow(dead_code)]
    pub fn urgency(mut self, value: u8) -> Self {
        self.record.urgency = value.min(9);
        self
    }

    /// Consume the builder and return the completed `IptcApplicationRecord`.
    #[allow(dead_code)]
    pub fn build(self) -> IptcApplicationRecord {
        self.record
    }
}

// ────────────────────────────────────────────────────────────────────────────
// IptcPhotoMeta — high-level photo metadata struct
// ────────────────────────────────────────────────────────────────────────────

/// High-level IPTC photo metadata combining the most commonly used fields
/// from the Application Record (Record 2).
///
/// This struct maps cleanly to the fields used by photo agencies, stock photo
/// services, and news organizations.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IptcPhotoMeta {
    /// Headline (2:105) — short publishable title.
    pub headline: Option<String>,
    /// Caption/Abstract (2:120) — longer description of the image.
    pub caption: Option<String>,
    /// Credit (2:110) — credit line / provider of the image.
    pub credit: Option<String>,
    /// Source (2:115) — original owner/creator of the intellectual content.
    pub source: Option<String>,
    /// Copyright Notice (2:116) — copyright statement.
    pub copyright: Option<String>,
    /// Keywords (2:25) — repeatable subject keywords.
    pub keywords: Vec<String>,
    /// City (2:90) — city of content origin.
    pub city: Option<String>,
    /// Country (2:101) — full country name.
    pub country: Option<String>,
    /// Province/State (2:95) — province or state.
    pub province_state: Option<String>,
    /// Urgency (2:10) — editorial urgency, 1 (most urgent) to 8 (least), 0=unset, 9=user-defined.
    pub urgency: u8,
    /// By-line (2:80) — name of the photographer/creator.
    pub byline: Option<String>,
    /// By-line Title (2:85) — title of the photographer/creator.
    pub byline_title: Option<String>,
    /// Special Instructions (2:40) — editorial instructions.
    pub special_instructions: Option<String>,
    /// Date Created (2:55) — CCYYMMDD format.
    pub date_created: Option<String>,
    /// Object Name (2:05) — shorthand reference for the object.
    pub object_name: Option<String>,
    /// Category (2:15) — subject category code.
    pub category: Option<String>,
    /// Supplemental Categories (2:20) — repeatable.
    pub supplemental_categories: Vec<String>,
}

impl IptcPhotoMeta {
    /// Create an empty `IptcPhotoMeta`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert to an `IptcApplicationRecord` (lossy — only shared fields are mapped).
    #[must_use]
    pub fn to_application_record(&self) -> IptcApplicationRecord {
        IptcApplicationRecord {
            headline: self.headline.clone(),
            caption: self.caption.clone(),
            keywords: self.keywords.clone(),
            byline: self.byline.clone(),
            city: self.city.clone(),
            country: self.country.clone(),
            copyright: self.copyright.clone(),
            urgency: self.urgency,
        }
    }

    /// Create from an `IptcApplicationRecord` (lossy — extra PhotoMeta fields will be None).
    #[must_use]
    pub fn from_application_record(rec: &IptcApplicationRecord) -> Self {
        Self {
            headline: rec.headline.clone(),
            caption: rec.caption.clone(),
            keywords: rec.keywords.clone(),
            byline: rec.byline.clone(),
            city: rec.city.clone(),
            country: rec.country.clone(),
            copyright: rec.copyright.clone(),
            urgency: rec.urgency,
            ..Default::default()
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// IIM binary encoding
// ────────────────────────────────────────────────────────────────────────────

/// IPTC IIM marker byte.
const IIM_MARKER: u8 = 0x1C;

/// Encode a single IIM dataset: `[0x1C][record][dataset][len_hi][len_lo][data]`.
fn encode_iim_dataset(record: u8, dataset: u8, data: &[u8]) -> Vec<u8> {
    let len = data.len().min(0xFFFF) as u16;
    let mut buf = Vec::with_capacity(5 + data.len());
    buf.push(IIM_MARKER);
    buf.push(record);
    buf.push(dataset);
    buf.extend_from_slice(&len.to_be_bytes());
    buf.extend_from_slice(&data[..len as usize]);
    buf
}

/// Encode a string field into an IIM dataset if the string is `Some`.
fn encode_string_field(out: &mut Vec<u8>, record: u8, dataset: u8, value: &Option<String>) {
    if let Some(ref s) = value {
        if !s.is_empty() {
            out.extend_from_slice(&encode_iim_dataset(record, dataset, s.as_bytes()));
        }
    }
}

/// Encode an [`IptcPhotoMeta`] into IPTC IIM binary format.
///
/// Produces a byte vector containing concatenated IIM datasets for all non-empty fields.
/// Each dataset follows the standard `[0x1C][record][dataset][len_hi][len_lo][data]` format.
///
/// All fields are encoded in Application Record (Record 2).
#[must_use]
pub fn encode_iptc_iim(meta: &IptcPhotoMeta) -> Vec<u8> {
    let mut out = Vec::new();
    let rec = 2u8;

    // Object Name (2:05)
    encode_string_field(&mut out, rec, 5, &meta.object_name);
    // Urgency (2:10)
    if meta.urgency > 0 && meta.urgency <= 9 {
        out.extend_from_slice(&encode_iim_dataset(rec, 10, &[b'0' + meta.urgency]));
    }
    // Category (2:15)
    encode_string_field(&mut out, rec, 15, &meta.category);
    // Supplemental Categories (2:20)
    for sc in &meta.supplemental_categories {
        if !sc.is_empty() {
            out.extend_from_slice(&encode_iim_dataset(rec, 20, sc.as_bytes()));
        }
    }
    // Keywords (2:25)
    for kw in &meta.keywords {
        if !kw.is_empty() {
            out.extend_from_slice(&encode_iim_dataset(rec, 25, kw.as_bytes()));
        }
    }
    // Special Instructions (2:40)
    encode_string_field(&mut out, rec, 40, &meta.special_instructions);
    // Date Created (2:55)
    encode_string_field(&mut out, rec, 55, &meta.date_created);
    // By-line (2:80)
    encode_string_field(&mut out, rec, 80, &meta.byline);
    // By-line Title (2:85)
    encode_string_field(&mut out, rec, 85, &meta.byline_title);
    // City (2:90)
    encode_string_field(&mut out, rec, 90, &meta.city);
    // Province/State (2:95)
    encode_string_field(&mut out, rec, 95, &meta.province_state);
    // Country (2:101)
    encode_string_field(&mut out, rec, 101, &meta.country);
    // Headline (2:105)
    encode_string_field(&mut out, rec, 105, &meta.headline);
    // Credit (2:110)
    encode_string_field(&mut out, rec, 110, &meta.credit);
    // Source (2:115)
    encode_string_field(&mut out, rec, 115, &meta.source);
    // Copyright (2:116)
    encode_string_field(&mut out, rec, 116, &meta.copyright);
    // Caption (2:120)
    encode_string_field(&mut out, rec, 120, &meta.caption);

    out
}

// ────────────────────────────────────────────────────────────────────────────
// IIM binary parsing into IptcPhotoMeta
// ────────────────────────────────────────────────────────────────────────────

/// Parse raw IPTC IIM binary data into an [`IptcPhotoMeta`].
///
/// Scans for 0x1C markers and extracts Application Record (Record 2) datasets.
/// Non-Record-2 datasets and unrecognized dataset numbers are silently ignored.
///
/// # Errors
///
/// Returns [`Error::ParseError`] if the data is truncated or contains invalid lengths.
pub fn parse_iptc_iim(data: &[u8]) -> Result<IptcPhotoMeta, Error> {
    let mut meta = IptcPhotoMeta::new();
    let mut i = 0;

    while i + 4 < data.len() {
        if data[i] != IIM_MARKER {
            i += 1;
            continue;
        }
        let record = data[i + 1];
        let dataset = data[i + 2];
        let length = u16::from_be_bytes([data[i + 3], data[i + 4]]) as usize;
        i += 5;

        if i + length > data.len() {
            return Err(Error::ParseError(format!(
                "IPTC IIM dataset ({record}:{dataset}) length {length} exceeds data at offset {}",
                i - 5
            )));
        }

        let payload = &data[i..i + length];
        i += length;

        // Only process Application Record (Record 2)
        if record != 2 {
            continue;
        }

        let text = String::from_utf8_lossy(payload).to_string();

        match dataset {
            5 => meta.object_name = Some(text),
            10 => {
                // Urgency is a single digit character
                if let Some(ch) = text.chars().next() {
                    if let Some(d) = ch.to_digit(10) {
                        meta.urgency = d as u8;
                    }
                }
            }
            15 => meta.category = Some(text),
            20 => meta.supplemental_categories.push(text),
            25 => meta.keywords.push(text),
            40 => meta.special_instructions = Some(text),
            55 => meta.date_created = Some(text),
            80 => meta.byline = Some(text),
            85 => meta.byline_title = Some(text),
            90 => meta.city = Some(text),
            95 => meta.province_state = Some(text),
            101 => meta.country = Some(text),
            105 => meta.headline = Some(text),
            110 => meta.credit = Some(text),
            115 => meta.source = Some(text),
            116 => meta.copyright = Some(text),
            120 => meta.caption = Some(text),
            _ => {} // silently ignore unknown datasets
        }
    }

    Ok(meta)
}

// ────────────────────────────────────────────────────────────────────────────
// XmpIptcBridge — convert IPTC to XMP/RDF
// ────────────────────────────────────────────────────────────────────────────

/// Bridge for converting between IPTC IIM metadata and XMP/RDF format.
///
/// Uses the standard XMP namespace mappings:
/// - `photoshop:` — `http://ns.adobe.com/photoshop/1.0/`
/// - `dc:` — `http://purl.org/dc/elements/1.1/`
/// - `Iptc4xmpCore:` — `http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/`
pub struct XmpIptcBridge;

/// Escape text for embedding in XML.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}

impl XmpIptcBridge {
    /// Convert an [`IptcPhotoMeta`] to an XMP/RDF document string.
    ///
    /// Produces a complete `<x:xmpmeta>` document with `rdf:RDF` containing
    /// a single `rdf:Description` using `photoshop:` and `dc:` namespaces.
    #[must_use]
    pub fn to_xmp(meta: &IptcPhotoMeta) -> String {
        let mut out = String::new();
        out.push_str("<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n");
        out.push_str("<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">\n");
        out.push_str("<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n");
        out.push_str("<rdf:Description\n");
        out.push_str("  xmlns:dc=\"http://purl.org/dc/elements/1.1/\"\n");
        out.push_str("  xmlns:photoshop=\"http://ns.adobe.com/photoshop/1.0/\"\n");
        out.push_str("  xmlns:Iptc4xmpCore=\"http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/\">\n");

        // dc:title
        if let Some(ref headline) = meta.headline {
            out.push_str("<dc:title>\n");
            out.push_str("  <rdf:Alt>\n");
            out.push_str("    <rdf:li xml:lang=\"x-default\">");
            out.push_str(&xml_escape(headline));
            out.push_str("</rdf:li>\n");
            out.push_str("  </rdf:Alt>\n");
            out.push_str("</dc:title>\n");
        }

        // dc:description (caption)
        if let Some(ref caption) = meta.caption {
            out.push_str("<dc:description>\n");
            out.push_str("  <rdf:Alt>\n");
            out.push_str("    <rdf:li xml:lang=\"x-default\">");
            out.push_str(&xml_escape(caption));
            out.push_str("</rdf:li>\n");
            out.push_str("  </rdf:Alt>\n");
            out.push_str("</dc:description>\n");
        }

        // dc:creator (byline)
        if let Some(ref byline) = meta.byline {
            out.push_str("<dc:creator>\n");
            out.push_str("  <rdf:Seq>\n");
            out.push_str("    <rdf:li>");
            out.push_str(&xml_escape(byline));
            out.push_str("</rdf:li>\n");
            out.push_str("  </rdf:Seq>\n");
            out.push_str("</dc:creator>\n");
        }

        // dc:rights (copyright)
        if let Some(ref copyright) = meta.copyright {
            out.push_str("<dc:rights>\n");
            out.push_str("  <rdf:Alt>\n");
            out.push_str("    <rdf:li xml:lang=\"x-default\">");
            out.push_str(&xml_escape(copyright));
            out.push_str("</rdf:li>\n");
            out.push_str("  </rdf:Alt>\n");
            out.push_str("</dc:rights>\n");
        }

        // dc:subject (keywords)
        if !meta.keywords.is_empty() {
            out.push_str("<dc:subject>\n");
            out.push_str("  <rdf:Bag>\n");
            for kw in &meta.keywords {
                out.push_str("    <rdf:li>");
                out.push_str(&xml_escape(kw));
                out.push_str("</rdf:li>\n");
            }
            out.push_str("  </rdf:Bag>\n");
            out.push_str("</dc:subject>\n");
        }

        // photoshop:Headline
        if let Some(ref headline) = meta.headline {
            out.push_str("<photoshop:Headline>");
            out.push_str(&xml_escape(headline));
            out.push_str("</photoshop:Headline>\n");
        }

        // photoshop:Credit
        if let Some(ref credit) = meta.credit {
            out.push_str("<photoshop:Credit>");
            out.push_str(&xml_escape(credit));
            out.push_str("</photoshop:Credit>\n");
        }

        // photoshop:Source
        if let Some(ref source) = meta.source {
            out.push_str("<photoshop:Source>");
            out.push_str(&xml_escape(source));
            out.push_str("</photoshop:Source>\n");
        }

        // photoshop:City
        if let Some(ref city) = meta.city {
            out.push_str("<photoshop:City>");
            out.push_str(&xml_escape(city));
            out.push_str("</photoshop:City>\n");
        }

        // photoshop:State
        if let Some(ref state) = meta.province_state {
            out.push_str("<photoshop:State>");
            out.push_str(&xml_escape(state));
            out.push_str("</photoshop:State>\n");
        }

        // photoshop:Country
        if let Some(ref country) = meta.country {
            out.push_str("<photoshop:Country>");
            out.push_str(&xml_escape(country));
            out.push_str("</photoshop:Country>\n");
        }

        // photoshop:DateCreated
        if let Some(ref date) = meta.date_created {
            out.push_str("<photoshop:DateCreated>");
            out.push_str(&xml_escape(date));
            out.push_str("</photoshop:DateCreated>\n");
        }

        // photoshop:Instructions
        if let Some(ref instr) = meta.special_instructions {
            out.push_str("<photoshop:Instructions>");
            out.push_str(&xml_escape(instr));
            out.push_str("</photoshop:Instructions>\n");
        }

        // photoshop:Urgency
        if meta.urgency > 0 && meta.urgency <= 9 {
            out.push_str("<photoshop:Urgency>");
            out.push_str(&meta.urgency.to_string());
            out.push_str("</photoshop:Urgency>\n");
        }

        // Iptc4xmpCore:CreatorContactInfo — byline title
        if let Some(ref title) = meta.byline_title {
            out.push_str("<Iptc4xmpCore:CreatorContactInfo>\n");
            out.push_str("  <rdf:Description>\n");
            out.push_str("    <Iptc4xmpCore:CiAdrPcode>");
            out.push_str(&xml_escape(title));
            out.push_str("</Iptc4xmpCore:CiAdrPcode>\n");
            out.push_str("  </rdf:Description>\n");
            out.push_str("</Iptc4xmpCore:CreatorContactInfo>\n");
        }

        out.push_str("</rdf:Description>\n");
        out.push_str("</rdf:RDF>\n");
        out.push_str("</x:xmpmeta>\n");
        out.push_str("<?xpacket end=\"w\"?>");
        out
    }

    /// Check if XMP output contains the expected namespace declarations.
    #[must_use]
    pub fn validate_xmp_namespaces(xmp: &str) -> bool {
        xmp.contains("xmlns:dc=\"http://purl.org/dc/elements/1.1/\"")
            && xmp.contains("xmlns:photoshop=\"http://ns.adobe.com/photoshop/1.0/\"")
    }

    /// Extract a simple text element value from XMP.
    ///
    /// Looks for `<tag>value</tag>` and returns the value.
    #[must_use]
    pub fn extract_xmp_simple_element(xmp: &str, tag: &str) -> Option<String> {
        let open = format!("<{tag}>");
        let close = format!("</{tag}>");
        let start = xmp.find(&open)? + open.len();
        let end = xmp[start..].find(&close)? + start;
        Some(xmp[start..end].to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iptc_dataset_new_and_tag() {
        let ds = IptcDataset::new(2, 80, b"Photographer".to_vec());
        assert_eq!(ds.tag(), (2, 80));
        assert_eq!(ds.record, 2);
        assert_eq!(ds.dataset, 80);
    }

    #[test]
    fn test_iptc_dataset_as_string_valid_utf8() {
        let ds = IptcDataset::new(2, 120, b"Hello World".to_vec());
        assert_eq!(ds.as_string(), Some("Hello World".to_string()));
    }

    #[test]
    fn test_iptc_dataset_as_string_invalid_utf8() {
        let ds = IptcDataset::new(2, 120, vec![0xFF, 0xFE]);
        assert_eq!(ds.as_string(), None);
    }

    #[test]
    fn test_iptc_record_numbers() {
        assert_eq!(IptcRecord::EnvelopeRecord.record_number(), 1);
        assert_eq!(IptcRecord::ApplicationRecord.record_number(), 2);
        assert_eq!(IptcRecord::NewsPhotoRecord.record_number(), 3);
    }

    #[test]
    fn test_iptc_application_record_to_datasets_headline() {
        let rec = IptcApplicationRecord {
            headline: Some("Breaking News".to_string()),
            ..Default::default()
        };
        let datasets = rec.to_datasets();
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].dataset, 105);
        assert_eq!(
            datasets[0].as_string().expect("should succeed in test"),
            "Breaking News"
        );
    }

    #[test]
    fn test_iptc_application_record_keywords() {
        let rec = IptcApplicationRecord {
            keywords: vec!["rust".to_string(), "media".to_string()],
            ..Default::default()
        };
        let datasets = rec.to_datasets();
        let kw_datasets: Vec<_> = datasets.iter().filter(|d| d.dataset == 25).collect();
        assert_eq!(kw_datasets.len(), 2);
    }

    #[test]
    fn test_iptc_application_record_word_count() {
        let rec = IptcApplicationRecord {
            headline: Some("Two Words".to_string()),
            caption: Some("One two three".to_string()),
            keywords: vec!["single".to_string()],
            ..Default::default()
        };
        assert_eq!(rec.word_count(), 6);
    }

    #[test]
    fn test_iptc_application_record_word_count_empty() {
        let rec = IptcApplicationRecord::default();
        assert_eq!(rec.word_count(), 0);
    }

    #[test]
    fn test_iptc_parser_parse_valid() {
        // Build a minimal IPTC IIM block manually
        let mut data = Vec::new();
        let payload = b"TestHeadline";
        data.push(0x1C);
        data.push(2); // record
        data.push(105); // dataset (headline)
        let len = payload.len() as u16;
        data.extend_from_slice(&len.to_be_bytes());
        data.extend_from_slice(payload);

        let datasets = IptcParser::parse_iptc(&data);
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].tag(), (2, 105));
        assert_eq!(
            datasets[0].as_string().expect("should succeed in test"),
            "TestHeadline"
        );
    }

    #[test]
    fn test_iptc_parser_parse_multiple() {
        let mut data = Vec::new();
        for (ds_num, text) in &[(80u8, b"Alice" as &[u8]), (90, b"Paris")] {
            data.push(0x1C);
            data.push(2);
            data.push(*ds_num);
            let len = text.len() as u16;
            data.extend_from_slice(&len.to_be_bytes());
            data.extend_from_slice(text);
        }
        let datasets = IptcParser::parse_iptc(&data);
        assert_eq!(datasets.len(), 2);
        assert_eq!(datasets[0].dataset, 80);
        assert_eq!(datasets[1].dataset, 90);
    }

    #[test]
    fn test_iptc_parser_skips_non_marker_bytes() {
        let mut data = vec![0x00, 0xFF, 0xAB]; // garbage
        let payload = b"skip test";
        data.push(0x1C);
        data.push(2);
        data.push(101);
        let len = payload.len() as u16;
        data.extend_from_slice(&len.to_be_bytes());
        data.extend_from_slice(payload);

        let datasets = IptcParser::parse_iptc(&data);
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].dataset, 101);
    }

    #[test]
    fn test_iptc_builder_basic() {
        let rec = IptcBuilder::new()
            .headline("Big Story")
            .caption("Details here")
            .add_keyword("news")
            .byline("Jane Doe")
            .copyright("2026 NewsOrg")
            .build();

        assert_eq!(rec.headline.as_deref(), Some("Big Story"));
        assert_eq!(rec.caption.as_deref(), Some("Details here"));
        assert_eq!(rec.keywords, vec!["news"]);
        assert_eq!(rec.byline.as_deref(), Some("Jane Doe"));
        assert_eq!(rec.copyright.as_deref(), Some("2026 NewsOrg"));
    }

    #[test]
    fn test_iptc_builder_multiple_keywords() {
        let rec = IptcBuilder::new()
            .add_keyword("alpha")
            .add_keyword("beta")
            .add_keyword("gamma")
            .build();
        assert_eq!(rec.keywords.len(), 3);
    }

    #[test]
    fn test_iptc_builder_urgency_clamped() {
        let rec = IptcBuilder::new().urgency(15).build();
        assert_eq!(rec.urgency, 9);

        let rec2 = IptcBuilder::new().urgency(5).build();
        assert_eq!(rec2.urgency, 5);
    }

    #[test]
    fn test_iptc_application_record_urgency_dataset() {
        let rec = IptcApplicationRecord {
            urgency: 3,
            ..Default::default()
        };
        let datasets = rec.to_datasets();
        let urg: Vec<_> = datasets.iter().filter(|d| d.dataset == 10).collect();
        assert_eq!(urg.len(), 1);
        assert_eq!(urg[0].data, vec![b'0' + 3]);
    }

    // ── IptcPhotoMeta ───────────────────────────────────────────────────

    #[test]
    fn test_iptc_photo_meta_default() {
        let meta = IptcPhotoMeta::new();
        assert!(meta.headline.is_none());
        assert!(meta.keywords.is_empty());
        assert_eq!(meta.urgency, 0);
    }

    #[test]
    fn test_iptc_photo_meta_to_application_record() {
        let meta = IptcPhotoMeta {
            headline: Some("Test Headline".to_string()),
            caption: Some("Test Caption".to_string()),
            copyright: Some("2026 Test".to_string()),
            keywords: vec!["rust".to_string(), "media".to_string()],
            city: Some("Tokyo".to_string()),
            country: Some("Japan".to_string()),
            byline: Some("Photographer".to_string()),
            urgency: 3,
            ..Default::default()
        };
        let rec = meta.to_application_record();
        assert_eq!(rec.headline.as_deref(), Some("Test Headline"));
        assert_eq!(rec.caption.as_deref(), Some("Test Caption"));
        assert_eq!(rec.keywords.len(), 2);
        assert_eq!(rec.urgency, 3);
    }

    #[test]
    fn test_iptc_photo_meta_from_application_record() {
        let rec = IptcApplicationRecord {
            headline: Some("Headline".to_string()),
            byline: Some("Author".to_string()),
            ..Default::default()
        };
        let meta = IptcPhotoMeta::from_application_record(&rec);
        assert_eq!(meta.headline.as_deref(), Some("Headline"));
        assert_eq!(meta.byline.as_deref(), Some("Author"));
        assert!(meta.credit.is_none()); // not in ApplicationRecord
    }

    // ── encode_iptc_iim ────────────────────────────────────────────────

    #[test]
    fn test_encode_iptc_iim_empty() {
        let meta = IptcPhotoMeta::new();
        let data = encode_iptc_iim(&meta);
        assert!(data.is_empty());
    }

    #[test]
    fn test_encode_iptc_iim_headline_only() {
        let meta = IptcPhotoMeta {
            headline: Some("Breaking News".to_string()),
            ..Default::default()
        };
        let data = encode_iptc_iim(&meta);
        assert!(!data.is_empty());
        // Should start with marker
        assert_eq!(data[0], 0x1C);
        // Record 2, dataset 105 (headline)
        assert_eq!(data[1], 2);
        assert_eq!(data[2], 105);
    }

    #[test]
    fn test_encode_iptc_iim_multiple_keywords() {
        let meta = IptcPhotoMeta {
            keywords: vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()],
            ..Default::default()
        };
        let data = encode_iptc_iim(&meta);
        // Count how many 0x1C markers with dataset=25
        let datasets = IptcParser::parse_iptc(&data);
        let kw_count = datasets.iter().filter(|d| d.dataset == 25).count();
        assert_eq!(kw_count, 3);
    }

    #[test]
    fn test_encode_iptc_iim_urgency() {
        let meta = IptcPhotoMeta {
            urgency: 5,
            ..Default::default()
        };
        let data = encode_iptc_iim(&meta);
        let datasets = IptcParser::parse_iptc(&data);
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].dataset, 10);
        assert_eq!(datasets[0].data, vec![b'5']);
    }

    // ── parse_iptc_iim ─────────────────────────────────────────────────

    #[test]
    fn test_parse_iptc_iim_empty() {
        let meta = parse_iptc_iim(&[]).expect("empty data should be ok");
        assert!(meta.headline.is_none());
        assert!(meta.keywords.is_empty());
    }

    #[test]
    fn test_parse_iptc_iim_single_field() {
        let mut data = Vec::new();
        let payload = b"Test Headline";
        data.push(0x1C);
        data.push(2);
        data.push(105);
        data.extend_from_slice(&(payload.len() as u16).to_be_bytes());
        data.extend_from_slice(payload);

        let meta = parse_iptc_iim(&data).expect("should parse");
        assert_eq!(meta.headline.as_deref(), Some("Test Headline"));
    }

    #[test]
    fn test_parse_iptc_iim_truncated_data() {
        // Claim length of 100 but only provide 5 bytes of payload
        let data = vec![0x1C, 2, 105, 0, 100, b'H', b'e', b'l', b'l', b'o'];
        let result = parse_iptc_iim(&data);
        assert!(result.is_err());
    }

    // ── IIM roundtrip ──────────────────────────────────────────────────

    #[test]
    fn test_iptc_iim_roundtrip_full() {
        let original = IptcPhotoMeta {
            headline: Some("Major Event".to_string()),
            caption: Some("A significant event occurred today.".to_string()),
            credit: Some("AP Photo".to_string()),
            source: Some("Associated Press".to_string()),
            copyright: Some("2026 AP".to_string()),
            keywords: vec!["news".to_string(), "event".to_string(), "world".to_string()],
            city: Some("New York".to_string()),
            country: Some("United States".to_string()),
            province_state: Some("NY".to_string()),
            urgency: 2,
            byline: Some("Jane Reporter".to_string()),
            byline_title: Some("Staff Photographer".to_string()),
            special_instructions: Some("Do not crop".to_string()),
            date_created: Some("20260311".to_string()),
            object_name: Some("EVENT-2026".to_string()),
            category: Some("I".to_string()),
            supplemental_categories: vec!["Politics".to_string(), "Intl".to_string()],
        };

        let encoded = encode_iptc_iim(&original);
        let decoded = parse_iptc_iim(&encoded).expect("roundtrip should succeed");

        assert_eq!(decoded.headline, original.headline);
        assert_eq!(decoded.caption, original.caption);
        assert_eq!(decoded.credit, original.credit);
        assert_eq!(decoded.source, original.source);
        assert_eq!(decoded.copyright, original.copyright);
        assert_eq!(decoded.keywords, original.keywords);
        assert_eq!(decoded.city, original.city);
        assert_eq!(decoded.country, original.country);
        assert_eq!(decoded.province_state, original.province_state);
        assert_eq!(decoded.urgency, original.urgency);
        assert_eq!(decoded.byline, original.byline);
        assert_eq!(decoded.byline_title, original.byline_title);
        assert_eq!(decoded.special_instructions, original.special_instructions);
        assert_eq!(decoded.date_created, original.date_created);
        assert_eq!(decoded.object_name, original.object_name);
        assert_eq!(decoded.category, original.category);
        assert_eq!(
            decoded.supplemental_categories,
            original.supplemental_categories
        );
    }

    #[test]
    fn test_iptc_iim_roundtrip_minimal() {
        let original = IptcPhotoMeta {
            headline: Some("Simple".to_string()),
            ..Default::default()
        };
        let encoded = encode_iptc_iim(&original);
        let decoded = parse_iptc_iim(&encoded).expect("roundtrip should succeed");
        assert_eq!(decoded.headline, original.headline);
    }

    // ── XmpIptcBridge ──────────────────────────────────────────────────

    #[test]
    fn test_xmp_iptc_bridge_basic() {
        let meta = IptcPhotoMeta {
            headline: Some("Test Headline".to_string()),
            caption: Some("Test Caption".to_string()),
            byline: Some("Photographer".to_string()),
            copyright: Some("2026 Owner".to_string()),
            keywords: vec!["test".to_string()],
            ..Default::default()
        };
        let xmp = XmpIptcBridge::to_xmp(&meta);
        assert!(xmp.contains("xmpmeta"));
        assert!(xmp.contains("rdf:RDF"));
        assert!(xmp.contains("photoshop:Headline"));
        assert!(xmp.contains("Test Headline"));
        assert!(xmp.contains("dc:title"));
        assert!(xmp.contains("dc:description"));
        assert!(xmp.contains("dc:creator"));
        assert!(xmp.contains("dc:rights"));
        assert!(xmp.contains("dc:subject"));
    }

    #[test]
    fn test_xmp_iptc_bridge_namespaces() {
        let meta = IptcPhotoMeta {
            headline: Some("H".to_string()),
            ..Default::default()
        };
        let xmp = XmpIptcBridge::to_xmp(&meta);
        assert!(XmpIptcBridge::validate_xmp_namespaces(&xmp));
    }

    #[test]
    fn test_xmp_iptc_bridge_empty_meta() {
        let meta = IptcPhotoMeta::new();
        let xmp = XmpIptcBridge::to_xmp(&meta);
        // Should still produce valid XMP structure
        assert!(xmp.contains("xmpmeta"));
        assert!(xmp.contains("rdf:Description"));
        // Should not contain photoshop tags for empty fields
        assert!(!xmp.contains("photoshop:Headline"));
    }

    #[test]
    fn test_xmp_iptc_bridge_xml_escaping() {
        let meta = IptcPhotoMeta {
            headline: Some("Title & <More>".to_string()),
            ..Default::default()
        };
        let xmp = XmpIptcBridge::to_xmp(&meta);
        assert!(xmp.contains("&amp;"));
        assert!(xmp.contains("&lt;More&gt;"));
    }

    #[test]
    fn test_xmp_iptc_bridge_location_fields() {
        let meta = IptcPhotoMeta {
            city: Some("Paris".to_string()),
            province_state: Some("Ile-de-France".to_string()),
            country: Some("France".to_string()),
            ..Default::default()
        };
        let xmp = XmpIptcBridge::to_xmp(&meta);
        assert!(xmp.contains("photoshop:City>Paris"));
        assert!(xmp.contains("photoshop:State>Ile-de-France"));
        assert!(xmp.contains("photoshop:Country>France"));
    }

    #[test]
    fn test_xmp_extract_simple_element() {
        let xmp = "<photoshop:Credit>Reuters</photoshop:Credit>";
        let val = XmpIptcBridge::extract_xmp_simple_element(xmp, "photoshop:Credit");
        assert_eq!(val.as_deref(), Some("Reuters"));
    }

    #[test]
    fn test_xmp_extract_simple_element_missing() {
        let xmp = "<photoshop:City>Tokyo</photoshop:City>";
        let val = XmpIptcBridge::extract_xmp_simple_element(xmp, "photoshop:Credit");
        assert!(val.is_none());
    }

    #[test]
    fn test_xmp_iptc_bridge_urgency_and_date() {
        let meta = IptcPhotoMeta {
            urgency: 7,
            date_created: Some("20260311".to_string()),
            special_instructions: Some("Embargo until noon".to_string()),
            ..Default::default()
        };
        let xmp = XmpIptcBridge::to_xmp(&meta);
        assert!(xmp.contains("photoshop:Urgency>7"));
        assert!(xmp.contains("photoshop:DateCreated>20260311"));
        assert!(xmp.contains("photoshop:Instructions>Embargo until noon"));
    }
}
