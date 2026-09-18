//! DataCite 4.x DOI metadata schema for archived research media.
//!
//! Implements the DataCite Metadata Schema 4.x (<https://schema.datacite.org/>)
//! for describing archived research media objects with persistent DOI identifiers.
//!
//! Reference: <https://schema.datacite.org/meta/kernel-4/>

use serde::{Deserialize, Serialize};

/// XML namespace for DataCite Metadata Schema kernel-4.
const DATACITE_NAMESPACE: &str = "http://datacite.org/schema/kernel-4";
/// DataCite schema location.
const DATACITE_SCHEMA_LOCATION: &str =
    "http://datacite.org/schema/kernel-4 https://schema.datacite.org/meta/kernel-4/metadata.xsd";

/// A DataCite 4.x resource descriptor.
///
/// Represents the mandatory and recommended fields from the DataCite
/// Metadata Schema 4.x for identifying and describing research media outputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataCiteResource {
    /// DOI identifier, e.g. "10.1234/example".
    /// Maps to `<identifier identifierType="DOI">`.
    pub identifier: String,
    /// Creator names (at least one required by schema).
    pub creators: Vec<DataCiteCreator>,
    /// Resource titles (at least one required).
    pub titles: Vec<String>,
    /// Publisher name (required).
    pub publisher: String,
    /// Publication year (required), e.g. 2026.
    pub publication_year: u16,
    /// Resource type general category (required), e.g. "Audiovisual", "Dataset".
    /// Maps to `resourceTypeGeneral` attribute on `<resourceType>`.
    pub resource_type: String,
    /// Specific resource type description (free text, e.g. "Video").
    /// Maps to text content of `<resourceType>`.
    pub resource_type_description: String,
    /// Optional description.
    pub description: Option<String>,
    /// Optional subject keywords.
    pub subjects: Vec<String>,
    /// Optional language code, e.g. "en".
    pub language: Option<String>,
    /// Optional related identifier list.
    pub related_identifiers: Vec<DataCiteRelatedIdentifier>,
}

/// A creator entry in a DataCite resource.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataCiteCreator {
    /// Creator name (family, given or organization).
    pub name: String,
    /// Name identifier URI (ORCID, ISNI, etc.), e.g. `"https://orcid.org/0000-0001-2345-6789"`.
    pub name_identifier: Option<String>,
    /// Affiliation name.
    pub affiliation: Option<String>,
}

/// A related identifier entry referencing another resource.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataCiteRelatedIdentifier {
    /// The identifier value.
    pub identifier: String,
    /// Relation type, e.g. "IsPartOf", "IsDerivedFrom", "Cites", "IsSupplementTo".
    pub relation_type: String,
    /// Identifier type, e.g. "DOI", "URL", "ARK", "ISBN".
    pub identifier_type: String,
}

impl DataCiteResource {
    /// Creates a new DataCite resource with mandatory fields only.
    #[must_use]
    pub fn new(
        identifier: impl Into<String>,
        creator_name: impl Into<String>,
        title: impl Into<String>,
        publisher: impl Into<String>,
        publication_year: u16,
        resource_type: impl Into<String>,
    ) -> Self {
        Self {
            identifier: identifier.into(),
            creators: vec![DataCiteCreator {
                name: creator_name.into(),
                name_identifier: None,
                affiliation: None,
            }],
            titles: vec![title.into()],
            publisher: publisher.into(),
            publication_year,
            resource_type: resource_type.into(),
            resource_type_description: String::new(),
            description: None,
            subjects: Vec::new(),
            language: None,
            related_identifiers: Vec::new(),
        }
    }

    /// Serializes this resource to a DataCite 4.x XML document string.
    ///
    /// Produces a well-formed XML document with the DataCite kernel-4
    /// namespace and all mandatory fields populated.
    #[must_use]
    pub fn to_xml(&self) -> String {
        let mut out = String::with_capacity(2048);
        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str("<resource xmlns=\"");
        out.push_str(DATACITE_NAMESPACE);
        out.push_str("\"\n");
        out.push_str("  xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n");
        out.push_str("  xsi:schemaLocation=\"");
        out.push_str(DATACITE_SCHEMA_LOCATION);
        out.push_str("\">\n");

        // identifier
        out.push_str("  <identifier identifierType=\"DOI\">");
        out.push_str(&escape_xml(&self.identifier));
        out.push_str("</identifier>\n");

        // creators
        out.push_str("  <creators>\n");
        for creator in &self.creators {
            out.push_str("    <creator>\n");
            out.push_str("      <creatorName>");
            out.push_str(&escape_xml(&creator.name));
            out.push_str("</creatorName>\n");
            if let Some(ref orcid) = creator.name_identifier {
                out.push_str("      <nameIdentifier nameIdentifierScheme=\"ORCID\">");
                out.push_str(&escape_xml(orcid));
                out.push_str("</nameIdentifier>\n");
            }
            if let Some(ref affil) = creator.affiliation {
                out.push_str("      <affiliation>");
                out.push_str(&escape_xml(affil));
                out.push_str("</affiliation>\n");
            }
            out.push_str("    </creator>\n");
        }
        out.push_str("  </creators>\n");

        // titles
        out.push_str("  <titles>\n");
        for title in &self.titles {
            out.push_str("    <title>");
            out.push_str(&escape_xml(title));
            out.push_str("</title>\n");
        }
        out.push_str("  </titles>\n");

        // publisher
        out.push_str("  <publisher>");
        out.push_str(&escape_xml(&self.publisher));
        out.push_str("</publisher>\n");

        // publicationYear
        out.push_str("  <publicationYear>");
        out.push_str(&self.publication_year.to_string());
        out.push_str("</publicationYear>\n");

        // resourceType
        out.push_str("  <resourceType resourceTypeGeneral=\"");
        out.push_str(&escape_xml(&self.resource_type));
        out.push_str("\">");
        out.push_str(&escape_xml(&self.resource_type_description));
        out.push_str("</resourceType>\n");

        // subjects (optional)
        if !self.subjects.is_empty() {
            out.push_str("  <subjects>\n");
            for subject in &self.subjects {
                out.push_str("    <subject>");
                out.push_str(&escape_xml(subject));
                out.push_str("</subject>\n");
            }
            out.push_str("  </subjects>\n");
        }

        // language (optional)
        if let Some(ref lang) = self.language {
            out.push_str("  <language>");
            out.push_str(&escape_xml(lang));
            out.push_str("</language>\n");
        }

        // description (optional)
        if let Some(ref desc) = self.description {
            out.push_str("  <descriptions>\n");
            out.push_str("    <description descriptionType=\"Abstract\">");
            out.push_str(&escape_xml(desc));
            out.push_str("</description>\n");
            out.push_str("  </descriptions>\n");
        }

        // relatedIdentifiers (optional)
        if !self.related_identifiers.is_empty() {
            out.push_str("  <relatedIdentifiers>\n");
            for rel in &self.related_identifiers {
                out.push_str("    <relatedIdentifier relationType=\"");
                out.push_str(&escape_xml(&rel.relation_type));
                out.push_str("\" relatedIdentifierType=\"");
                out.push_str(&escape_xml(&rel.identifier_type));
                out.push_str("\">");
                out.push_str(&escape_xml(&rel.identifier));
                out.push_str("</relatedIdentifier>\n");
            }
            out.push_str("  </relatedIdentifiers>\n");
        }

        out.push_str("</resource>\n");
        out
    }

    /// Serializes this resource to a JSON string.
    ///
    /// Uses the serde_json serialization with the field names defined on the struct.
    ///
    /// # Errors
    ///
    /// Returns an error string if JSON serialization fails (should not happen for
    /// well-formed structs).
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    /// Parses a DataCite 4.x XML document and reconstructs the resource.
    ///
    /// Performs a best-effort extraction of mandatory and optional fields.
    /// Unknown or malformed optional fields are silently skipped.
    ///
    /// # Errors
    ///
    /// Returns an error if mandatory fields (identifier, at least one creator,
    /// at least one title, publisher, publicationYear, resourceType) are absent
    /// or the XML is fundamentally unparseable.
    pub fn parse_xml(xml: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let identifier =
            extract_element_text(xml, "identifier").ok_or("missing <identifier> element")?;
        let publisher =
            extract_element_text(xml, "publisher").ok_or("missing <publisher> element")?;
        let pub_year_str = extract_element_text(xml, "publicationYear")
            .ok_or("missing <publicationYear> element")?;
        let publication_year: u16 = pub_year_str
            .trim()
            .parse()
            .map_err(|e| format!("invalid publicationYear: {e}"))?;

        // resourceType — get resourceTypeGeneral attribute and text content
        let (resource_type, resource_type_description) = parse_resource_type(xml);

        // creators
        let creators = parse_creators(xml);
        if creators.is_empty() {
            return Err("at least one <creator> is required".into());
        }

        // titles
        let titles = parse_all_elements(xml, "title");
        if titles.is_empty() {
            return Err("at least one <title> is required".into());
        }

        // optional fields
        let description = extract_element_text(xml, "description");
        let language = extract_element_text(xml, "language");
        let subjects = parse_all_elements(xml, "subject");
        let related_identifiers = parse_related_identifiers(xml);

        Ok(Self {
            identifier,
            creators,
            titles,
            publisher,
            publication_year,
            resource_type,
            resource_type_description,
            description,
            subjects,
            language,
            related_identifiers,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal XML helpers
// ---------------------------------------------------------------------------

/// Escapes special XML characters in a string.
fn escape_xml(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            other => out.push(other),
        }
    }
    out
}

/// Unescapes XML entities in a string.
fn unescape_xml(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

/// Extracts text content of the first occurrence of a simple element tag.
///
/// Handles elements with and without attributes. Returns `None` if tag not found.
fn extract_element_text(xml: &str, tag: &str) -> Option<String> {
    // Match <tag> or <tag attr="..."> pattern
    let open_exact = format!("<{tag}>");
    let open_attr = format!("<{tag} ");

    let content_start = if let Some(pos) = xml.find(&open_exact) {
        pos + open_exact.len()
    } else if let Some(pos) = xml.find(&open_attr) {
        // find the closing > of the opening tag
        let rest = &xml[pos..];
        let close = rest.find('>')?;
        pos + close + 1
    } else {
        return None;
    };

    let close_tag = format!("</{tag}>");
    let content_end = xml[content_start..].find(&close_tag)?;
    let text = xml[content_start..content_start + content_end].trim();
    Some(unescape_xml(text))
}

/// Extracts all text contents of a repeated element tag.
fn parse_all_elements(xml: &str, tag: &str) -> Vec<String> {
    let mut results = Vec::new();
    let open_exact = format!("<{tag}>");
    let open_attr = format!("<{tag} ");
    let close_tag = format!("</{tag}>");

    let mut search_from = 0usize;
    loop {
        // Find next occurrence of <tag> or <tag ...>
        let start_pos = {
            let pos_exact = xml[search_from..]
                .find(&open_exact)
                .map(|p| (p, open_exact.len()));
            let pos_attr = xml[search_from..].find(&open_attr).map(|p| {
                let rest = &xml[search_from + p..];
                let close = rest.find('>').unwrap_or(0);
                (p, close + 1)
            });
            match (pos_exact, pos_attr) {
                (Some((pe, le)), Some((pa, la))) => {
                    if pe <= pa {
                        Some((search_from + pe + le, search_from + pe))
                    } else {
                        Some((search_from + pa + la, search_from + pa))
                    }
                }
                (Some((pe, le)), None) => Some((search_from + pe + le, search_from + pe)),
                (None, Some((pa, la))) => Some((search_from + pa + la, search_from + pa)),
                (None, None) => None,
            }
        };

        let (content_start, _tag_start) = match start_pos {
            Some(s) => s,
            None => break,
        };

        // Find closing tag
        let Some(rel_end) = xml[content_start..].find(&close_tag) else {
            break;
        };
        let text = xml[content_start..content_start + rel_end].trim();
        results.push(unescape_xml(text));
        search_from = content_start + rel_end + close_tag.len();
    }
    results
}

/// Parses `resourceTypeGeneral` attribute and text content from `<resourceType>`.
fn parse_resource_type(xml: &str) -> (String, String) {
    let tag = "resourceType";
    let open = format!("<{tag}");
    let close = format!("</{tag}>");

    let Some(tag_start) = xml.find(&open) else {
        return (String::new(), String::new());
    };

    let rest = &xml[tag_start..];
    let Some(close_open_tag) = rest.find('>') else {
        return (String::new(), String::new());
    };

    let attrs_str = &rest[tag.len() + 1..close_open_tag];
    let general = extract_attr_value(attrs_str, "resourceTypeGeneral")
        .map(unescape_xml)
        .unwrap_or_default();

    let content_start = tag_start + close_open_tag + 1;
    let description = if let Some(rel_end) = xml[content_start..].find(&close) {
        unescape_xml(xml[content_start..content_start + rel_end].trim())
    } else {
        String::new()
    };

    (general, description)
}

/// Parses all `<creator>` blocks from the XML.
fn parse_creators(xml: &str) -> Vec<DataCiteCreator> {
    let mut creators = Vec::new();
    let open = "<creator>";
    let close = "</creator>";
    let mut pos = 0;

    loop {
        let Some(start) = xml[pos..].find(open) else {
            break;
        };
        let creator_start = pos + start + open.len();
        let Some(end) = xml[creator_start..].find(close) else {
            break;
        };
        let creator_xml = &xml[creator_start..creator_start + end];

        let name = extract_element_text(creator_xml, "creatorName").unwrap_or_default();
        let name_identifier = extract_element_text(creator_xml, "nameIdentifier");
        let affiliation = extract_element_text(creator_xml, "affiliation");

        creators.push(DataCiteCreator {
            name,
            name_identifier,
            affiliation,
        });

        pos = creator_start + end + close.len();
    }
    creators
}

/// Parses all `<relatedIdentifier>` elements.
fn parse_related_identifiers(xml: &str) -> Vec<DataCiteRelatedIdentifier> {
    let mut result = Vec::new();
    let open = "<relatedIdentifier ";
    let close_tag = "</relatedIdentifier>";
    let mut pos = 0;

    loop {
        let Some(start) = xml[pos..].find(open) else {
            break;
        };
        let el_start = pos + start;
        let rest = &xml[el_start..];

        let Some(close_open) = rest.find('>') else {
            break;
        };
        let attrs_str = &rest[open.len()..close_open];
        let content_start = el_start + close_open + 1;

        let Some(rel_end) = xml[content_start..].find(close_tag) else {
            break;
        };
        let identifier = unescape_xml(xml[content_start..content_start + rel_end].trim());

        let relation_type = extract_attr_value(attrs_str, "relationType")
            .map(unescape_xml)
            .unwrap_or_default();
        let identifier_type = extract_attr_value(attrs_str, "relatedIdentifierType")
            .map(unescape_xml)
            .unwrap_or_default();

        result.push(DataCiteRelatedIdentifier {
            identifier,
            relation_type,
            identifier_type,
        });

        pos = content_start + rel_end + close_tag.len();
    }
    result
}

/// Extracts a named attribute value from an attribute string like `key="val" key2="val2"`.
fn extract_attr_value<'a>(attrs: &'a str, key: &str) -> Option<&'a str> {
    let search = format!("{key}=\"");
    let start = attrs.find(&search)? + search.len();
    let end = attrs[start..].find('"')?;
    Some(&attrs[start..start + end])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_resource() -> DataCiteResource {
        DataCiteResource {
            identifier: "10.1234/oximedia.test.001".to_string(),
            creators: vec![DataCiteCreator {
                name: "Kitasan, Team".to_string(),
                name_identifier: Some("https://orcid.org/0000-0001-2345-6789".to_string()),
                affiliation: Some("COOLJAPAN OU".to_string()),
            }],
            titles: vec!["OxiMedia Test Archive Video".to_string()],
            publisher: "COOLJAPAN Research Archive".to_string(),
            publication_year: 2026,
            resource_type: "Audiovisual".to_string(),
            resource_type_description: "Video".to_string(),
            description: Some("A test video for the OxiMedia archive system.".to_string()),
            subjects: vec![
                "digital preservation".to_string(),
                "video codec".to_string(),
            ],
            language: Some("en".to_string()),
            related_identifiers: vec![DataCiteRelatedIdentifier {
                identifier: "10.5678/related.dataset".to_string(),
                relation_type: "IsSupplementTo".to_string(),
                identifier_type: "DOI".to_string(),
            }],
        }
    }

    #[test]
    fn test_datacite_round_trip_xml() {
        let original = sample_resource();
        let xml = original.to_xml();
        let parsed = DataCiteResource::parse_xml(&xml)
            .expect("parse_xml should succeed on well-formed output");

        assert_eq!(parsed.identifier, original.identifier);
        assert_eq!(parsed.publisher, original.publisher);
        assert_eq!(parsed.publication_year, original.publication_year);
        assert_eq!(parsed.resource_type, original.resource_type);
        assert_eq!(parsed.creators.len(), original.creators.len());
        assert_eq!(parsed.creators[0].name, original.creators[0].name);
        assert_eq!(parsed.titles.len(), original.titles.len());
        assert_eq!(parsed.titles[0], original.titles[0]);
        assert_eq!(parsed.language, original.language);
        assert_eq!(parsed.description, original.description);
        assert_eq!(parsed.subjects.len(), original.subjects.len());
        assert_eq!(
            parsed.related_identifiers.len(),
            original.related_identifiers.len()
        );
        assert_eq!(
            parsed.related_identifiers[0].identifier,
            original.related_identifiers[0].identifier
        );
    }

    #[test]
    fn test_datacite_xml_has_required_fields() {
        let resource = sample_resource();
        let xml = resource.to_xml();

        assert!(xml.contains("identifier"), "XML must contain 'identifier'");
        assert!(xml.contains("creator"), "XML must contain 'creator'");
        assert!(xml.contains("title"), "XML must contain 'title'");
        assert!(xml.contains("publisher"), "XML must contain 'publisher'");
        assert!(
            xml.contains("publicationYear"),
            "XML must contain 'publicationYear'"
        );
        assert!(
            xml.contains("resourceType"),
            "XML must contain 'resourceType'"
        );
        assert!(
            xml.contains(DATACITE_NAMESPACE),
            "XML must reference DataCite namespace"
        );
    }

    #[test]
    fn test_datacite_json_keys() {
        let resource = sample_resource();
        let json = resource.to_json().expect("to_json should succeed");

        assert!(
            json.contains("\"identifier\""),
            "JSON must have 'identifier' key"
        );
        assert!(
            json.contains("\"creators\""),
            "JSON must have 'creators' key"
        );
        assert!(json.contains("\"titles\""), "JSON must have 'titles' key");
        assert!(
            json.contains("\"publisher\""),
            "JSON must have 'publisher' key"
        );
        assert!(
            json.contains("\"publication_year\""),
            "JSON must have 'publication_year' key"
        );
        assert!(
            json.contains("\"resource_type\""),
            "JSON must have 'resource_type' key"
        );
    }

    #[test]
    fn test_datacite_xml_escape() {
        let mut resource = sample_resource();
        resource.publisher = "A & B <Archive>".to_string();
        let xml = resource.to_xml();
        assert!(
            xml.contains("A &amp; B &lt;Archive&gt;"),
            "XML entities must be escaped"
        );
    }

    #[test]
    fn test_datacite_minimal_roundtrip() {
        let minimal = DataCiteResource::new(
            "10.9999/minimal",
            "Doe, Jane",
            "Minimal Test Dataset",
            "Test Publisher",
            2025,
            "Dataset",
        );
        let xml = minimal.to_xml();
        let parsed = DataCiteResource::parse_xml(&xml)
            .expect("parse_xml of minimal resource should succeed");
        assert_eq!(parsed.identifier, "10.9999/minimal");
        assert_eq!(parsed.creators[0].name, "Doe, Jane");
        assert_eq!(parsed.titles[0], "Minimal Test Dataset");
        assert_eq!(parsed.publisher, "Test Publisher");
        assert_eq!(parsed.publication_year, 2025);
        assert_eq!(parsed.resource_type, "Dataset");
    }

    #[test]
    fn test_datacite_creator_optional_fields() {
        let resource = sample_resource();
        let xml = resource.to_xml();
        assert!(xml.contains("nameIdentifier"), "ORCID should appear in XML");
        assert!(
            xml.contains("affiliation"),
            "Affiliation should appear in XML"
        );

        let parsed = DataCiteResource::parse_xml(&xml).expect("parse should succeed");
        let creator = &parsed.creators[0];
        assert!(
            creator.name_identifier.is_some(),
            "nameIdentifier must survive round-trip"
        );
        assert!(
            creator.affiliation.is_some(),
            "affiliation must survive round-trip"
        );
    }
}
