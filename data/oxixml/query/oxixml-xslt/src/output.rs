//! [`OutputProperties`] — the serialization parameters `xsl:output` sets.
//!
//! The property names, their defaults and the way `method="xml"` and `method="html"` differ are
//! specified by [XSLT 3.0 §26](https://www.w3.org/TR/xslt-30/#serialization) by reference to
//! [XSLT and XQuery Serialization 3.1](https://www.w3.org/TR/xslt-xquery-serialization-31/).
//! Everything here is a plain value: nothing is computed until a tree is written.

use oxixml_infoset::ExpandedName;
use std::collections::BTreeMap;

/// The serialization method — how a result tree becomes text.
///
/// # Examples
///
/// ```
/// use oxixml_xslt::OutputMethod;
///
/// assert_eq!(OutputMethod::parse("html"), Some(OutputMethod::Html));
/// assert_eq!(OutputMethod::parse("json"), Some(OutputMethod::Json));
/// assert_eq!(OutputMethod::parse("nonesuch"), None);
/// assert!(OutputMethod::Xhtml.is_xml_like());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OutputMethod {
    /// Well-formed XML, the default when the result does not look like HTML.
    #[default]
    Xml,
    /// HTML 4/5 conventions: empty tags for the void elements, unescaped `<script>` content,
    /// minimized boolean attributes.
    Html,
    /// XHTML: XML syntax with HTML's empty-element and `<meta>` conventions.
    Xhtml,
    /// Text: only the string values of the text nodes.
    Text,
    /// JSON, as
    /// [Serialization 3.1 §10](https://www.w3.org/TR/xslt-xquery-serialization-31/#json-output)
    /// defines it: the result must be a single map, array, or atomic value.
    Json,
    /// The adaptive method of
    /// [Serialization 3.1 §11](https://www.w3.org/TR/xslt-xquery-serialization-31/#adaptive-output),
    /// which writes every item in the sequence in a form that reflects its own kind.
    Adaptive,
}

impl OutputMethod {
    /// Parses the value of the `method` attribute.
    ///
    /// [XSLT 3.0 §26](https://www.w3.org/TR/xslt-30/#serialization) types the attribute as an
    /// `EQName`, so `Q{}xml` names the same method as `xml`; a name in a namespace would select
    /// an implementation-defined method, of which this processor provides none.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        let local = match value.strip_prefix("Q{") {
            None => value,
            Some(rest) => match rest.split_once('}') {
                Some(("", local)) => local,
                _ => return None,
            },
        };
        match local {
            "xml" => Some(Self::Xml),
            "html" => Some(Self::Html),
            "xhtml" => Some(Self::Xhtml),
            "text" => Some(Self::Text),
            "json" => Some(Self::Json),
            "adaptive" => Some(Self::Adaptive),
            _ => None,
        }
    }

    /// Whether the method writes XML syntax — that is, XML or XHTML.
    #[must_use]
    pub const fn is_xml_like(self) -> bool {
        matches!(self, Self::Xml | Self::Xhtml)
    }

    /// The method's name, as `xsl:output` spells it.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Xml => "xml",
            Self::Html => "html",
            Self::Xhtml => "xhtml",
            Self::Text => "text",
            Self::Json => "json",
            Self::Adaptive => "adaptive",
        }
    }
}

/// How `standalone` was specified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Standalone {
    /// Not specified: the XML declaration omits the pseudo-attribute.
    #[default]
    Omit,
    /// `standalone="yes"`.
    Yes,
    /// `standalone="no"`.
    No,
}

/// The complete set of serialization parameters.
///
/// # Examples
///
/// ```
/// use oxixml_xslt::{OutputMethod, OutputProperties};
///
/// let properties = OutputProperties::default();
/// assert_eq!(properties.method, OutputMethod::Xml);
/// assert_eq!(properties.encoding, "UTF-8");
/// assert!(!properties.indent);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputProperties {
    /// `method`.
    pub method: OutputMethod,
    /// Whether the method was stated explicitly, which decides whether the HTML sniffing rule
    /// of the specification applies.
    pub method_specified: bool,
    /// `version` — the XML or HTML version written into the declaration.
    pub version: Option<String>,
    /// `encoding`.
    pub encoding: String,
    /// `omit-xml-declaration`.
    pub omit_xml_declaration: bool,
    /// `standalone`.
    pub standalone: Standalone,
    /// `doctype-public`.
    pub doctype_public: Option<String>,
    /// `doctype-system`.
    pub doctype_system: Option<String>,
    /// `cdata-section-elements`.
    pub cdata_section_elements: Vec<ExpandedName>,
    /// `indent`.
    pub indent: bool,
    /// `media-type`.
    pub media_type: Option<String>,
    /// `item-separator` (XSLT 3.0).
    pub item_separator: Option<String>,
    /// `normalization-form`.
    pub normalization_form: Option<String>,
    /// `include-content-type` (HTML and XHTML).
    pub include_content_type: bool,
    /// `escape-uri-attributes` (HTML and XHTML).
    pub escape_uri_attributes: bool,
    /// `undeclare-prefixes` (XML 1.1).
    pub undeclare_prefixes: bool,
    /// `use-character-maps`, resolved into a single code point map.
    pub character_map: BTreeMap<char, String>,
    /// `suppress-indentation` (XSLT 3.0).
    pub suppress_indentation: Vec<ExpandedName>,
    /// `byte-order-mark` — whether the serialized string starts with `U+FEFF`.
    pub byte_order_mark: bool,
    /// `html-version` (XSLT 3.0), which decides whether `<!DOCTYPE html>` is written.
    pub html_version: Option<String>,
    /// `json-node-output-method` — the method a node *inside* a JSON value is serialized with
    /// before its lexical form is written as a JSON string
    /// ([Serialization 3.1 §10](https://www.w3.org/TR/xslt-xquery-serialization-31/#json-output)).
    /// It has no effect unless [`method`](Self::method) is `json`.
    pub json_node_output_method: OutputMethod,
    /// `allow-duplicate-names` — whether the `json` method accepts two map entries whose keys
    /// serialize alike rather than raising `SERE0022`.
    pub allow_duplicate_names: bool,
    /// `build-tree` — whether the destination's content is normalized into a result tree before
    /// it is serialized, or handed to the method as the raw sequence it was.
    ///
    /// `None` leaves it to the method, which is what
    /// [XSLT 3.0 §26](https://www.w3.org/TR/xslt-30/#serialization) specifies: `yes` "if the
    /// `method` attribute specifies `xml`, `html`, `xhtml`, or `text`, or if it is omitted", and
    /// `no` for the two methods that write items rather than nodes.
    pub build_tree: Option<bool>,
}

impl Default for OutputProperties {
    fn default() -> Self {
        Self {
            method: OutputMethod::Xml,
            method_specified: false,
            version: None,
            encoding: "UTF-8".to_owned(),
            omit_xml_declaration: false,
            standalone: Standalone::Omit,
            doctype_public: None,
            doctype_system: None,
            cdata_section_elements: Vec::new(),
            indent: false,
            media_type: None,
            item_separator: None,
            normalization_form: None,
            include_content_type: true,
            escape_uri_attributes: true,
            undeclare_prefixes: false,
            character_map: BTreeMap::new(),
            suppress_indentation: Vec::new(),
            byte_order_mark: false,
            html_version: None,
            json_node_output_method: OutputMethod::Xml,
            allow_duplicate_names: false,
            build_tree: None,
        }
    }
}

impl OutputProperties {
    /// The XML version to write, defaulting to `1.0`.
    #[must_use]
    pub fn xml_version(&self) -> &str {
        match &self.version {
            Some(version) if self.method.is_xml_like() => version,
            _ => "1.0",
        }
    }

    /// Whether the element's content must be wrapped in a CDATA section.
    #[must_use]
    pub fn is_cdata_element(&self, name: &ExpandedName) -> bool {
        self.cdata_section_elements.contains(name)
    }

    /// Whether the HTML version asked for is 5 or later, which is what makes the `html` and
    /// `xhtml` methods write a bare `<!DOCTYPE html>` with no external identifier.
    #[must_use]
    pub fn wants_html5_doctype(&self) -> bool {
        let stated = self.html_version.as_deref().or(self.version.as_deref());
        stated.is_some_and(|value| {
            value
                .split('.')
                .next()
                .and_then(|major| major.trim().parse::<u32>().ok())
                .is_some_and(|major| major >= 5)
        })
    }

    /// The `media-type`, defaulting per method as the serialization specification requires.
    #[must_use]
    pub fn effective_media_type(&self) -> &str {
        match &self.media_type {
            Some(value) if !value.is_empty() => value,
            _ => match self.method {
                OutputMethod::Html | OutputMethod::Xhtml => "text/html",
                OutputMethod::Text | OutputMethod::Adaptive => "text/plain",
                OutputMethod::Json => "application/json",
                OutputMethod::Xml => "application/xml",
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn methods_parse_and_classify() {
        assert_eq!(OutputMethod::parse("xml"), Some(OutputMethod::Xml));
        assert_eq!(OutputMethod::parse("text"), Some(OutputMethod::Text));
        assert_eq!(OutputMethod::parse("XML"), None);
        assert!(OutputMethod::Xml.is_xml_like());
        assert!(!OutputMethod::Html.is_xml_like());
        assert_eq!(OutputMethod::Xhtml.as_str(), "xhtml");
    }

    #[test]
    fn the_json_and_adaptive_methods_are_named_and_classified() {
        assert_eq!(OutputMethod::parse("json"), Some(OutputMethod::Json));
        assert_eq!(
            OutputMethod::parse("adaptive"),
            Some(OutputMethod::Adaptive)
        );
        assert_eq!(OutputMethod::parse("Q{}json"), Some(OutputMethod::Json));
        assert_eq!(
            OutputMethod::parse("Q{}adaptive"),
            Some(OutputMethod::Adaptive)
        );
        assert_eq!(OutputMethod::parse("Q{}xml"), Some(OutputMethod::Xml));
        assert_eq!(OutputMethod::parse("Q{urn:example}mine"), None);
        assert_eq!(OutputMethod::parse("Q{unterminated"), None);
        assert!(!OutputMethod::Json.is_xml_like());
        assert!(!OutputMethod::Adaptive.is_xml_like());
        assert_eq!(OutputMethod::Json.as_str(), "json");
        assert_eq!(OutputMethod::Adaptive.as_str(), "adaptive");

        let mut properties = OutputProperties {
            method: OutputMethod::Json,
            ..OutputProperties::default()
        };
        assert_eq!(properties.effective_media_type(), "application/json");
        assert_eq!(properties.xml_version(), "1.0");
        properties.method = OutputMethod::Adaptive;
        assert_eq!(properties.effective_media_type(), "text/plain");
    }

    #[test]
    fn the_defaults_match_the_specification() {
        let properties = OutputProperties::default();
        assert_eq!(properties.encoding, "UTF-8");
        assert_eq!(properties.standalone, Standalone::Omit);
        assert!(!properties.omit_xml_declaration);
        assert_eq!(properties.xml_version(), "1.0");
        assert!(!properties.is_cdata_element(&ExpandedName::new(None::<&str>, "p")));
    }
}
