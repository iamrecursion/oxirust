//! The fixed namespace names and element/attribute spellings of XML Schema.

/// The XML Schema namespace, `http://www.w3.org/2001/XMLSchema`.
///
/// This is the namespace of the schema *vocabulary* (`xs:schema`,
/// `xs:element`, …) and, at the same time, of every built-in datatype.
pub const XSD_NS: &str = "http://www.w3.org/2001/XMLSchema";

/// The XML Schema instance namespace,
/// `http://www.w3.org/2001/XMLSchema-instance`.
///
/// The four attributes a *instance* document may carry live here: `xsi:type`,
/// `xsi:nil`, `xsi:schemaLocation` and `xsi:noNamespaceSchemaLocation`.
pub const XSI_NS: &str = "http://www.w3.org/2001/XMLSchema-instance";

/// The `xml` namespace, `http://www.w3.org/XML/1998/namespace`.
pub const XML_NS: &str = "http://www.w3.org/XML/1998/namespace";

/// The XSD 1.1 "version control" namespace used by `vc:minVersion` and
/// `vc:maxVersion`, `http://www.w3.org/2007/XMLSchema-versioning`.
pub const VC_NS: &str = "http://www.w3.org/2007/XMLSchema-versioning";

/// Returns `true` when `namespace` is the XML Schema namespace.
#[must_use]
pub fn is_xsd_namespace(namespace: Option<&str>) -> bool {
    namespace == Some(XSD_NS)
}

/// Which version of XML Schema a [`SchemaSet`](crate::SchemaSet) implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum XsdVersion {
    /// [XML Schema 1.0](https://www.w3.org/TR/xmlschema-1/) (the default).
    #[default]
    V1_0,
    /// [XML Schema 1.1](https://www.w3.org/TR/xmlschema11-1/).
    V1_1,
}

impl XsdVersion {
    /// Returns `true` for [`XsdVersion::V1_1`].
    #[must_use]
    pub const fn is_1_1(self) -> bool {
        matches!(self, Self::V1_1)
    }
}

impl std::fmt::Display for XsdVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::V1_0 => "1.0",
            Self::V1_1 => "1.1",
        })
    }
}
