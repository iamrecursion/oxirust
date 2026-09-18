//! Structured XMP document model with namespace-aware property access.
//!
//! Provides `XmpDocument`, `XmpProperty`, `XmpNamespace`, `XmpSerializer`, and `XmpBuilder`.

/// A single XMP property identified by namespace and local name.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct XmpProperty {
    /// XML namespace URI or prefix identifier.
    pub namespace: String,
    /// Local property name.
    pub name: String,
    /// String value of the property.
    pub value: String,
}

impl XmpProperty {
    /// Create a new property.
    #[allow(dead_code)]
    pub fn new(
        namespace: impl Into<String>,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            name: name.into(),
            value: value.into(),
        }
    }

    /// Return the qualified name as `"namespace:name"`.
    #[allow(dead_code)]
    pub fn qualified_name(&self) -> String {
        format!("{}:{}", self.namespace, self.name)
    }
}

/// Well-known XMP namespaces.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum XmpNamespace {
    /// Dublin Core (`dc`).
    DublinCore,
    /// EXIF Auxiliary (`aux`).
    ExifAux,
    /// EXIF IFD (`exif`).
    ExifIFD,
    /// Photoshop (`photoshop`).
    Photoshop,
    /// XMP Basic (`xmp`).
    Xmp,
    /// XMP Rights Management (`xmpRights`).
    Rights,
}

impl XmpNamespace {
    /// Return the conventional namespace prefix.
    #[allow(dead_code)]
    pub fn prefix(self) -> &'static str {
        match self {
            Self::DublinCore => "dc",
            Self::ExifAux => "aux",
            Self::ExifIFD => "exif",
            Self::Photoshop => "photoshop",
            Self::Xmp => "xmp",
            Self::Rights => "xmpRights",
        }
    }

    /// Return the namespace URI.
    #[allow(dead_code)]
    pub fn uri(self) -> &'static str {
        match self {
            Self::DublinCore => "http://purl.org/dc/elements/1.1/",
            Self::ExifAux => "http://ns.adobe.com/exif/1.0/aux/",
            Self::ExifIFD => "http://ns.adobe.com/exif/1.0/",
            Self::Photoshop => "http://ns.adobe.com/photoshop/1.0/",
            Self::Xmp => "http://ns.adobe.com/xap/1.0/",
            Self::Rights => "http://ns.adobe.com/xap/1.0/rights/",
        }
    }
}

/// A collection of XMP properties.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct XmpDocument {
    /// All XMP properties stored in insertion order.
    pub properties: Vec<XmpProperty>,
}

impl XmpDocument {
    /// Create a new empty document.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Retrieve the value of the first property matching `(namespace, name)`.
    #[allow(dead_code)]
    pub fn get(&self, namespace: &str, name: &str) -> Option<&str> {
        self.properties
            .iter()
            .find(|p| p.namespace == namespace && p.name == name)
            .map(|p| p.value.as_str())
    }

    /// Insert or update the property identified by `(namespace, name)`.
    #[allow(dead_code)]
    pub fn set(
        &mut self,
        namespace: impl Into<String>,
        name: impl Into<String>,
        value: impl Into<String>,
    ) {
        let ns = namespace.into();
        let nm = name.into();
        let val = value.into();

        if let Some(prop) = self
            .properties
            .iter_mut()
            .find(|p| p.namespace == ns && p.name == nm)
        {
            prop.value = val;
        } else {
            self.properties.push(XmpProperty::new(ns, nm, val));
        }
    }

    /// Remove the first property matching `(namespace, name)`. Returns `true` if found.
    #[allow(dead_code)]
    pub fn remove(&mut self, namespace: &str, name: &str) -> bool {
        let before = self.properties.len();
        self.properties
            .retain(|p| !(p.namespace == namespace && p.name == name));
        self.properties.len() < before
    }

    /// Return all properties whose namespace matches `namespace`.
    #[allow(dead_code)]
    pub fn properties_in(&self, namespace: &str) -> Vec<&XmpProperty> {
        self.properties
            .iter()
            .filter(|p| p.namespace == namespace)
            .collect()
    }
}

/// Serializes an `XmpDocument` to a minimal RDF/XML string.
#[allow(dead_code)]
pub struct XmpSerializer;

impl XmpSerializer {
    /// Produce a minimal but valid RDF/XML representation of the document.
    #[allow(dead_code)]
    pub fn to_xml(doc: &XmpDocument) -> String {
        let mut xml = String::new();

        xml.push_str(r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>"#);
        xml.push('\n');
        xml.push_str(r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">"#);
        xml.push('\n');
        xml.push_str(r#"  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">"#);
        xml.push('\n');
        xml.push_str(r#"    <rdf:Description"#);

        // Collect unique namespaces
        let mut ns_seen: Vec<(&str, &str)> = Vec::new();
        for prop in &doc.properties {
            // Find matching XmpNamespace for the prefix
            for candidate in &[
                XmpNamespace::DublinCore,
                XmpNamespace::ExifAux,
                XmpNamespace::ExifIFD,
                XmpNamespace::Photoshop,
                XmpNamespace::Xmp,
                XmpNamespace::Rights,
            ] {
                if candidate.prefix() == prop.namespace
                    && !ns_seen.iter().any(|(p, _)| *p == candidate.prefix())
                {
                    ns_seen.push((candidate.prefix(), candidate.uri()));
                }
            }
        }

        for (prefix, uri) in &ns_seen {
            xml.push_str(&format!(r#" xmlns:{prefix}="{uri}""#));
        }

        if doc.properties.is_empty() {
            xml.push_str("/>\n");
        } else {
            xml.push_str(">\n");
            for prop in &doc.properties {
                let qname = prop.qualified_name();
                let escaped = prop
                    .value
                    .replace('&', "&amp;")
                    .replace('<', "&lt;")
                    .replace('>', "&gt;")
                    .replace('"', "&quot;");
                xml.push_str(&format!("      <{qname}>{escaped}</{qname}>\n"));
            }
            xml.push_str("    </rdf:Description>\n");
        }

        xml.push_str("  </rdf:RDF>\n");
        xml.push_str("</x:xmpmeta>\n");
        xml.push_str(r#"<?xpacket end="w"?>"#);
        xml.push('\n');

        xml
    }
}

/// Builder for constructing an `XmpDocument` for common photographic/editorial fields.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct XmpBuilder {
    doc: XmpDocument,
}

impl XmpBuilder {
    /// Create a new builder.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the Dublin Core title (`dc:title`).
    #[allow(dead_code)]
    pub fn title(mut self, value: impl Into<String>) -> Self {
        self.doc.set("dc", "title", value);
        self
    }

    /// Set the Dublin Core creator (`dc:creator`).
    #[allow(dead_code)]
    pub fn creator(mut self, value: impl Into<String>) -> Self {
        self.doc.set("dc", "creator", value);
        self
    }

    /// Set the Dublin Core subject/keywords (`dc:subject`).
    #[allow(dead_code)]
    pub fn subject(mut self, value: impl Into<String>) -> Self {
        self.doc.set("dc", "subject", value);
        self
    }

    /// Set the XMP Rights usage terms (`xmpRights:UsageTerms`).
    #[allow(dead_code)]
    pub fn rights(mut self, value: impl Into<String>) -> Self {
        self.doc.set("xmpRights", "UsageTerms", value);
        self
    }

    /// Set the Dublin Core description (`dc:description`).
    #[allow(dead_code)]
    pub fn description(mut self, value: impl Into<String>) -> Self {
        self.doc.set("dc", "description", value);
        self
    }

    /// Consume the builder and return the completed `XmpDocument`.
    #[allow(dead_code)]
    pub fn build(self) -> XmpDocument {
        self.doc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xmp_property_qualified_name() {
        let prop = XmpProperty::new("dc", "title", "My Photo");
        assert_eq!(prop.qualified_name(), "dc:title");
    }

    #[test]
    fn test_xmp_property_fields() {
        let prop = XmpProperty::new("xmp", "CreateDate", "2026-03-01");
        assert_eq!(prop.namespace, "xmp");
        assert_eq!(prop.name, "CreateDate");
        assert_eq!(prop.value, "2026-03-01");
    }

    #[test]
    fn test_xmp_namespace_prefix() {
        assert_eq!(XmpNamespace::DublinCore.prefix(), "dc");
        assert_eq!(XmpNamespace::ExifIFD.prefix(), "exif");
        assert_eq!(XmpNamespace::Rights.prefix(), "xmpRights");
    }

    #[test]
    fn test_xmp_namespace_uri() {
        assert!(XmpNamespace::DublinCore.uri().contains("purl.org"));
        assert!(XmpNamespace::Xmp.uri().contains("adobe.com"));
    }

    #[test]
    fn test_xmp_document_set_and_get() {
        let mut doc = XmpDocument::new();
        doc.set("dc", "title", "Sunrise");
        assert_eq!(doc.get("dc", "title"), Some("Sunrise"));
        assert_eq!(doc.get("dc", "creator"), None);
    }

    #[test]
    fn test_xmp_document_set_updates_existing() {
        let mut doc = XmpDocument::new();
        doc.set("dc", "title", "Old");
        doc.set("dc", "title", "New");
        assert_eq!(doc.get("dc", "title"), Some("New"));
        // Still only one property for "dc:title"
        assert_eq!(doc.properties_in("dc").len(), 1);
    }

    #[test]
    fn test_xmp_document_remove_existing() {
        let mut doc = XmpDocument::new();
        doc.set("dc", "title", "To Remove");
        assert!(doc.remove("dc", "title"));
        assert_eq!(doc.get("dc", "title"), None);
    }

    #[test]
    fn test_xmp_document_remove_missing() {
        let mut doc = XmpDocument::new();
        assert!(!doc.remove("dc", "nonexistent"));
    }

    #[test]
    fn test_xmp_document_properties_in() {
        let mut doc = XmpDocument::new();
        doc.set("dc", "title", "T");
        doc.set("dc", "creator", "C");
        doc.set("xmp", "CreateDate", "D");
        let dc_props = doc.properties_in("dc");
        assert_eq!(dc_props.len(), 2);
    }

    #[test]
    fn test_xmp_serializer_to_xml_empty() {
        let doc = XmpDocument::new();
        let xml = XmpSerializer::to_xml(&doc);
        assert!(xml.contains("xmpmeta"));
        assert!(xml.contains("rdf:RDF"));
        assert!(xml.contains(r#"<?xpacket end="w"?>"#));
    }

    #[test]
    fn test_xmp_serializer_to_xml_with_properties() {
        let mut doc = XmpDocument::new();
        doc.set("dc", "title", "Test Title");
        let xml = XmpSerializer::to_xml(&doc);
        assert!(xml.contains("dc:title"));
        assert!(xml.contains("Test Title"));
    }

    #[test]
    fn test_xmp_serializer_escapes_special_chars() {
        let mut doc = XmpDocument::new();
        doc.set("dc", "description", "A & B < C > D");
        let xml = XmpSerializer::to_xml(&doc);
        assert!(xml.contains("&amp;"));
        assert!(xml.contains("&lt;"));
        assert!(xml.contains("&gt;"));
    }

    #[test]
    fn test_xmp_builder_fields() {
        let doc = XmpBuilder::new()
            .title("Landscape")
            .creator("Alice")
            .subject("nature, outdoor")
            .rights("CC-BY 4.0")
            .description("A beautiful landscape photo")
            .build();

        assert_eq!(doc.get("dc", "title"), Some("Landscape"));
        assert_eq!(doc.get("dc", "creator"), Some("Alice"));
        assert_eq!(doc.get("dc", "subject"), Some("nature, outdoor"));
        assert_eq!(doc.get("xmpRights", "UsageTerms"), Some("CC-BY 4.0"));
        assert_eq!(
            doc.get("dc", "description"),
            Some("A beautiful landscape photo")
        );
    }

    #[test]
    fn test_xmp_builder_empty() {
        let doc = XmpBuilder::new().build();
        assert!(doc.properties.is_empty());
    }
}
