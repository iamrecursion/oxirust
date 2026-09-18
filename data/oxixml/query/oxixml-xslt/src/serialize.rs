//! Turning a [`ResultTree`] into text, under a set of [`OutputProperties`].
//!
//! This is the serializer of
//! [XSLT and XQuery Serialization 3.1](https://www.w3.org/TR/xslt-xquery-serialization-31/):
//! the `xml`, `xhtml`, `html` and `text` output methods, the XML declaration and document type
//! declaration, CDATA sections, indentation, character maps, and the escaping rules — including
//! the `disable-output-escaping` flag XSLT 1.0 puts on individual text nodes.

use crate::error::{XsltError, XsltResult};
use crate::output::{OutputMethod, OutputProperties, Standalone};
use crate::tree::ResultTree;
use oxixml_infoset::{ExpandedName, NodeKind, Sequence};

/// The HTML elements written without an end tag.
const HTML_VOID_ELEMENTS: &[&str] = &[
    "area", "base", "basefont", "br", "col", "embed", "frame", "hr", "img", "input", "isindex",
    "keygen", "link", "meta", "param", "source", "track", "wbr",
];

/// The HTML elements whose content is CDATA and is therefore never escaped.
const HTML_RAW_TEXT_ELEMENTS: &[&str] = &["script", "style"];

/// The HTML attributes that are written minimized when their value equals their name.
const HTML_BOOLEAN_ATTRIBUTES: &[&str] = &[
    "checked", "compact", "declare", "defer", "disabled", "ismap", "multiple", "nohref",
    "noresize", "noshade", "nowrap", "readonly", "selected",
];

/// The `(element, attribute)` pairs whose values are URIs, for `escape-uri-attributes`.
const HTML_URI_ATTRIBUTES: &[(&str, &str)] = &[
    ("a", "href"),
    ("area", "href"),
    ("base", "href"),
    ("blockquote", "cite"),
    ("del", "cite"),
    ("form", "action"),
    ("frame", "longdesc"),
    ("frame", "src"),
    ("head", "profile"),
    ("iframe", "longdesc"),
    ("iframe", "src"),
    ("img", "longdesc"),
    ("img", "src"),
    ("img", "usemap"),
    ("input", "src"),
    ("input", "usemap"),
    ("ins", "cite"),
    ("link", "href"),
    ("object", "classid"),
    ("object", "codebase"),
    ("object", "data"),
    ("object", "usemap"),
    ("q", "cite"),
    ("script", "src"),
];

/// Serializes `tree` under `properties`.
///
/// # Examples
///
/// ```
/// use oxixml_xslt::{OutputProperties, Stylesheet};
/// use oxixml_infoset::Tree;
///
/// let stylesheet = Tree::parse_str(
///     r#"<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
///          <xsl:template match="/"><out/></xsl:template>
///        </xsl:stylesheet>"#,
/// )?;
/// let compiled = Stylesheet::compile(&stylesheet.root())?;
/// let text = compiled.transform_to_string(&Tree::parse_str("<in/>")?.root())?;
/// assert!(text.ends_with("<out/>"));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[must_use]
pub fn serialize(tree: &ResultTree, properties: &OutputProperties) -> String {
    // A destination whose method builds no tree was serialized where its raw result still
    // existed, by that method's own rules. Where the raw result happened to be node-shaped the
    // tree carries a navigable view of it as well, for a caller that wants to query the result —
    // but the text is the method's and is handed back exactly as it was produced.
    if let Some(raw) = tree.raw_serialization() {
        return raw.to_owned();
    }
    let method = effective_method(tree, properties);
    let mut writer = Writer {
        out: String::new(),
        properties,
        method,
        scope: vec![(
            Some("xml".to_owned()),
            oxixml_qname::XML_NAMESPACE.to_owned(),
        )],
        suppressing: false,
    };
    writer.document(tree);
    writer.out
}

/// Whether a destination under `properties` builds a final result tree at all.
///
/// [XSLT 3.0 §26](https://www.w3.org/TR/xslt-30/#serialization) makes `build-tree` default to
/// `yes` "if the `method` attribute specifies `xml`, `html`, `xhtml`, or `text`, or if it is
/// omitted", and notes that "the value `no` would be appropriate for a JSON serialization
/// method" — which is what the `json` and `adaptive` methods need, since neither writes a tree:
/// a map or an array is not a node and could not be put into one. A `build-tree` written out
/// overrides that default either way.
#[must_use]
pub(crate) fn builds_a_tree(properties: &OutputProperties) -> bool {
    properties.build_tree.unwrap_or(!matches!(
        properties.method,
        OutputMethod::Json | OutputMethod::Adaptive
    ))
}

/// Serializes a *raw result* — the sequence a destination whose method builds no tree produced —
/// under `properties`.
///
/// The `json` and `adaptive` methods of
/// [Serialization 3.1 §10 and §11](https://www.w3.org/TR/xslt-xquery-serialization-31/#json-output)
/// write items rather than nodes, and [`oxixml_fn::xmlio`] already implements both for
/// `fn:serialize`. This translates the `xsl:output` parameter set into the one that function
/// takes and hands the sequence over, so the two spellings of a serialization cannot drift.
///
/// # Errors
///
/// Whatever the method raises over the value — `SERE0021` for an item the `json` method cannot
/// write, `SERE0022` for duplicate keys under `allow-duplicate-names="no"`, `SESU0011` for an
/// unsupported normalization form.
pub(crate) fn serialize_raw(
    sequence: &Sequence,
    properties: &OutputProperties,
) -> XsltResult<String> {
    let parameters = raw_parameters(properties);
    Ok(oxixml_fn::xmlio::serialize(sequence, &parameters)?)
}

/// The [`oxixml_fn::xmlio`] parameter set that `properties` describes.
fn raw_parameters(properties: &OutputProperties) -> oxixml_fn::xmlio::SerializationParameters {
    let method = raw_method(properties.method);
    let mut parameters = oxixml_fn::xmlio::SerializationParameters::serialization_defaults()
        .with_method(method)
        .with_json_node_output_method(raw_method(properties.json_node_output_method))
        .with_allow_duplicate_names(properties.allow_duplicate_names)
        .with_include_content_type(properties.include_content_type)
        .with_indent(properties.indent)
        .with_encoding(properties.encoding.clone())
        .with_normalization_form(properties.normalization_form.clone().unwrap_or_default())
        .with_byte_order_mark(properties.byte_order_mark)
        .with_item_separator(properties.item_separator.clone())
        .with_omit_xml_declaration(properties.omit_xml_declaration)
        .with_standalone(match properties.standalone {
            crate::output::Standalone::Yes => Some(true),
            crate::output::Standalone::No => Some(false),
            crate::output::Standalone::Omit => None,
        })
        .with_doctype_public(properties.doctype_public.clone())
        .with_doctype_system(properties.doctype_system.clone())
        .with_escape_uri_attributes(properties.escape_uri_attributes)
        .with_undeclare_prefixes(properties.undeclare_prefixes)
        .with_cdata_section_elements(properties.cdata_section_elements.clone())
        .with_suppress_indentation(properties.suppress_indentation.clone())
        .with_character_map(
            properties
                .character_map
                .iter()
                .map(|(character, replacement)| (*character, replacement.clone()))
                .collect(),
        );
    if let Some(version) = &properties.version {
        parameters = parameters.with_version(version.clone());
    }
    if let Some(html_version) = &properties.html_version {
        parameters = parameters.with_html_version(html_version.clone());
    }
    parameters
}

/// The [`oxixml_fn::xmlio`] spelling of a serialization method.
const fn raw_method(method: OutputMethod) -> oxixml_fn::xmlio::OutputMethod {
    use oxixml_fn::xmlio::OutputMethod as FnMethod;
    match method {
        OutputMethod::Json => FnMethod::Json,
        OutputMethod::Adaptive => FnMethod::Adaptive,
        OutputMethod::Html => FnMethod::Html,
        OutputMethod::Xhtml => FnMethod::Xhtml,
        OutputMethod::Text => FnMethod::Text,
        OutputMethod::Xml => FnMethod::Xml,
    }
}

/// The serialization errors [Serialization 3.1 §3](https://www.w3.org/TR/xslt-xquery-serialization-31/#serparam)
/// raises before a single character is written.
///
/// Every one of these is a property of the parameter set, or of the shape of the tree, rather
/// than of the text that comes out — so they are decided in one pass here and reported as
/// dynamic errors, leaving [`serialize`] itself total.
///
/// # Errors
///
/// `SEPM0004`, `SEPM0009`, `SEPM0010`, `SESU0007`, `SESU0011`, `SESU0013`, `SERE0014` or
/// `SERE0015`, each under the condition the specification attaches to it.
pub(crate) fn check(tree: &ResultTree, properties: &OutputProperties) -> XsltResult<()> {
    // The raw result has already been through its own serializer, which raised whatever its
    // method raises; none of the parameter checks below belongs to `json` or `adaptive`.
    if tree.raw_serialization().is_some() {
        return Ok(());
    }
    let method = effective_method(tree, properties);
    // "It is an error if an output encoding other than UTF-8 or UTF-16 is requested and the
    // serializer does not support that encoding." A name this processor does not recognize at
    // all is the case the suite states (`output-0184`, `-0185`); a recognized legacy encoding is
    // still served, since the result is text a caller re-encodes.
    if !is_known_encoding(&properties.encoding) {
        return Err(XsltError::dynamic(
            "SESU0007",
            format!(
                "the `{}` output encoding is not supported",
                properties.encoding
            ),
        ));
    }
    // "It is a serialization error if the value of the normalization-form parameter specifies a
    // normalization form that is not supported." `fully-normalized` is a documented gap.
    if let Some(form) = properties.normalization_form.as_deref()
        && !matches!(form.trim(), "none" | "NFC" | "NFD" | "NFKC" | "NFKD")
    {
        return Err(XsltError::dynamic(
            "SESU0011",
            format!("the `{form}` normalization form is not supported"),
        ));
    }
    // "It is a serialization error if the serializer does not support the version of XML or HTML
    // specified by the version parameter."
    if let Some(version) = properties.version.as_deref() {
        let supported = match method {
            OutputMethod::Html => matches!(version.trim(), "4.0" | "4.01" | "5" | "5.0"),
            OutputMethod::Xhtml => matches!(version.trim(), "1.0" | "1.1" | "5" | "5.0"),
            // `version` is not a parameter of the `json` or `adaptive` methods at all
            // (Serialization 3.1 §10, §11), so no value of it can be unsupported.
            OutputMethod::Xml
            | OutputMethod::Text
            | OutputMethod::Json
            | OutputMethod::Adaptive => true,
        };
        if !supported {
            return Err(XsltError::dynamic(
                "SESU0013",
                format!(
                    "version `{version}` of the `{}` output method is not supported",
                    method.as_str()
                ),
            ));
        }
    }
    if method.is_xml_like() {
        // "It is a serialization error to specify the doctype-system parameter, or to specify
        // the standalone parameter with a value other than omit, if the instance of the data
        // model contains text nodes or multiple element nodes as children of the root node."
        let mut elements = 0usize;
        let mut text = false;
        for index in tree.root_children() {
            match tree.raw(index).map(|node| node.kind) {
                Some(NodeKind::Element) => elements += 1,
                Some(NodeKind::Text) => text = true,
                _ => {}
            }
        }
        if (properties.doctype_system.is_some() || properties.standalone != Standalone::Omit)
            && (text || elements > 1)
        {
            return Err(XsltError::dynamic(
                "SEPM0004",
                "`doctype-system` and `standalone` cannot be written for a result whose root has \
                 a text node child or more than one element child",
            ));
        }
        // "It is a serialization error if the omit-xml-declaration parameter has the value yes,
        // and the standalone attribute has a value other than omit; or the version parameter has
        // a value other than 1.0 and the doctype-system parameter is specified."
        if properties.omit_xml_declaration {
            if properties.standalone != Standalone::Omit {
                return Err(XsltError::dynamic(
                    "SEPM0009",
                    "`standalone` was requested for a result whose XML declaration is omitted",
                ));
            }
            if properties.doctype_system.is_some()
                && properties
                    .version
                    .as_deref()
                    .is_some_and(|version| version.trim() != "1.0")
            {
                return Err(XsltError::dynamic(
                    "SEPM0009",
                    "`doctype-system` cannot be written for a result whose XML declaration is \
                     omitted and whose version is not 1.0",
                ));
            }
        }
    }
    // "It is a serialization error if the output method is xml, the value of the
    // undeclare-prefixes parameter is yes, and the value of the version parameter is 1.0."
    if method == OutputMethod::Xml
        && properties.undeclare_prefixes
        && properties.xml_version() == "1.0"
    {
        return Err(XsltError::dynamic(
            "SEPM0010",
            "`undeclare-prefixes` has no meaning under XML 1.0, which cannot undeclare a \
             namespace",
        ));
    }
    if method.is_xml_like() {
        check_xml_characters(tree, properties.xml_version())?;
    }
    // `SERE0014` belongs to HTML 4: HTML 5 gives the `C1` controls a numeric character
    // reference, which is what this serializer writes and what `output-0195a`/`-0195b` assert.
    // Only a version explicitly stated below 5 asks for the error (`output-0195`).
    if method == OutputMethod::Html && states_html_before_5(properties) {
        check_html_content(tree)?;
    } else if method == OutputMethod::Html {
        check_html_processing_instructions(tree)?;
    }
    Ok(())
}

/// Whether `character` may appear in a document written to `version` of XML.
///
/// [XML 1.0 §2.2](https://www.w3.org/TR/xml/#charsets) admits `#x9`, `#xA`, `#xD` and everything
/// from `#x20` up, minus the two non-characters at the end of the basic plane;
/// [XML 1.1 §2.2](https://www.w3.org/TR/xml11/#charsets) admits the C0 controls as well, `#x0`
/// alone excepted — but only as character references, which is [`Serializer::escapes_as_reference`]'s
/// business rather than this one's. Rust's `char` already excludes the surrogates and everything
/// above `#x10FFFF`, so those ranges need no test here.
fn is_xml_character(character: char, version: &str) -> bool {
    if matches!(character as u32, 0xfffe | 0xffff) {
        return false;
    }
    if version == "1.1" {
        return character != '\0';
    }
    matches!(character, '\t' | '\n' | '\r') || (character as u32) >= 0x20
}

/// Rejects a result tree holding a character the target version of XML cannot spell.
///
/// # Errors
///
/// `SERE0006` — "it is a serialization error if the result tree contains a character that is not
/// permitted by the version of XML or HTML being output". `xml-version-029` copies a source
/// document holding `#x7` and serializes it as XML 1.0, where no spelling of that character
/// exists: a character reference to it is as ill-formed as the character itself.
fn check_xml_characters(tree: &ResultTree, version: &str) -> XsltResult<()> {
    for index in 0..tree.len() {
        let Some(node) = tree.raw(index) else {
            continue;
        };
        if !matches!(
            node.kind,
            NodeKind::Text
                | NodeKind::Attribute
                | NodeKind::Comment
                | NodeKind::ProcessingInstruction
        ) {
            continue;
        }
        if let Some(character) = node
            .value
            .chars()
            .find(|character| !is_xml_character(*character, version))
        {
            return Err(XsltError::dynamic(
                "SERE0006",
                format!(
                    "U+{:04X} is not a character XML {version} permits",
                    character as u32
                ),
            ));
        }
    }
    Ok(())
}

/// Whether the parameters state an HTML version earlier than 5.
///
/// The default is HTML 5, which is what makes `SERE0014` conditional: an unstated version is not
/// HTML 4.
fn states_html_before_5(properties: &OutputProperties) -> bool {
    properties
        .html_version
        .as_deref()
        .or(properties.version.as_deref())
        .and_then(|value| value.split('.').next())
        .and_then(|major| major.trim().parse::<u32>().ok())
        .is_some_and(|major| major < 5)
}

/// The two serialization errors the `html` method raises over the *content* of the result.
///
/// # Errors
///
/// `SERE0014` for a `C1` control character, which HTML 4 gives no meaning to, and `SERE0015` for
/// a `>` inside a processing instruction, which ends the instruction in HTML syntax.
fn check_html_content(tree: &ResultTree) -> XsltResult<()> {
    for index in 0..tree.len() {
        let Some(node) = tree.raw(index) else {
            continue;
        };
        match node.kind {
            NodeKind::Text | NodeKind::Attribute | NodeKind::Comment => {
                if let Some(character) = node
                    .value
                    .chars()
                    .find(|character| ('\u{7f}'..='\u{9f}').contains(character))
                {
                    return Err(XsltError::dynamic(
                        "SERE0014",
                        format!(
                            "U+{:04X} is legal in XML but has no meaning in HTML",
                            character as u32
                        ),
                    ));
                }
            }
            NodeKind::ProcessingInstruction => {}
            _ => {}
        }
    }
    check_html_processing_instructions(tree)
}

/// `SERE0015` alone, which every version of the `html` method raises.
///
/// # Errors
///
/// `SERE0015` for a `>` inside a processing instruction, which ends the instruction in HTML
/// syntax whatever the version.
fn check_html_processing_instructions(tree: &ResultTree) -> XsltResult<()> {
    for index in 0..tree.len() {
        let Some(node) = tree.raw(index) else {
            continue;
        };
        if node.kind == NodeKind::ProcessingInstruction && node.value.contains('>') {
            return Err(XsltError::dynamic(
                "SERE0015",
                "a `>` inside a processing instruction ends it in HTML syntax",
            ));
        }
    }
    Ok(())
}

/// Whether `name` is an output encoding this processor recognizes.
///
/// The result of a transformation is a Rust string, so every encoding is served by re-encoding
/// the text a caller receives; what the serializer really has to reject is a name that denotes
/// no encoding at all, which is what `SESU0007` is for.
fn is_known_encoding(name: &str) -> bool {
    let lowered = name.trim().to_ascii_lowercase();
    matches!(
        lowered.as_str(),
        "utf-8"
            | "utf8"
            | "utf-16"
            | "utf16"
            | "utf-16be"
            | "utf-16le"
            | "us-ascii"
            | "ascii"
            | "iso-8859-1"
            | "iso8859-1"
            | "latin1"
            | "iso-8859-2"
            | "iso-8859-5"
            | "iso-8859-15"
            | "windows-1250"
            | "windows-1251"
            | "windows-1252"
            | "shift_jis"
            | "euc-jp"
            | "koi8-r"
    )
}

/// Applies the `normalization-form` serialization parameter to the serialized result.
///
/// [XSLT and XQuery Serialization 3.1 §3](https://www.w3.org/TR/xslt-xquery-serialization-31/#serparam-normalization-form)
/// names the four Unicode normalization forms plus `none`; `fully-normalized` is a documented
/// gap and is treated as `none`.
#[must_use]
fn normalize(text: String, form: Option<&str>) -> String {
    match form.map(str::trim) {
        Some("NFC") => oxixml_unicode::norm::nfc(&text),
        Some("NFD") => oxixml_unicode::norm::nfd(&text),
        Some("NFKC") => oxixml_unicode::norm::nfkc(&text),
        Some("NFKD") => oxixml_unicode::norm::nfkd(&text),
        _ => text,
    }
}

/// Applies the specification's method-sniffing rule.
///
/// [XSLT 3.0 §26.1](https://www.w3.org/TR/xslt-30/#serialization) settles an unspecified
/// `method` on the document element of the result: `html` when it is `html` (in any case) in no
/// namespace, `xhtml` when it is `html` in the XHTML namespace, and `xml` otherwise —
/// `output-0130` writes an XHTML page with no `xsl:output` at all and asserts both the XML
/// declaration the `xhtml` method keeps and the `meta` element it injects.
fn effective_method(tree: &ResultTree, properties: &OutputProperties) -> OutputMethod {
    if properties.method_specified {
        return properties.method;
    }
    for index in tree.root_children() {
        let Some(node) = tree.raw(index) else {
            continue;
        };
        match node.kind {
            NodeKind::Element => {
                let Some(name) = node.name.as_ref() else {
                    return OutputMethod::Xml;
                };
                if !name.local_name().eq_ignore_ascii_case("html") {
                    return OutputMethod::Xml;
                }
                return match name.namespace() {
                    None => OutputMethod::Html,
                    Some(XHTML_NAMESPACE) => OutputMethod::Xhtml,
                    Some(_) => OutputMethod::Xml,
                };
            }
            NodeKind::Text if !node.value.trim().is_empty() => return OutputMethod::Xml,
            _ => {}
        }
    }
    OutputMethod::Xml
}

struct Writer<'a> {
    out: String,
    properties: &'a OutputProperties,
    method: OutputMethod,
    /// The namespace bindings an ancestor already declared, so that only the ones a start tag
    /// actually changes are written out.
    scope: Vec<(Option<String>, String)>,
    /// Whether an ancestor element is named by `suppress-indentation`, which puts the whole
    /// subtree below it out as written.
    suppressing: bool,
}

/// The namespace the `xhtml` output method treats as HTML.
const XHTML_NAMESPACE: &str = "http://www.w3.org/1999/xhtml";

impl Writer<'_> {
    fn document(&mut self, tree: &ResultTree) {
        if self.properties.byte_order_mark {
            self.out.push('\u{feff}');
        }
        if self.method == OutputMethod::Text {
            self.text_only(tree, 0);
            return;
        }
        self.declaration();
        let children = self.visible_children(tree, 0);
        let indent = self.properties.indent && self.indentable(tree, &children);
        // The document type declaration goes immediately before the first element, so any
        // comment or processing instruction the result starts with precedes it.
        let first_element = children.iter().position(|index| {
            tree.raw(*index)
                .is_some_and(|node| node.kind == NodeKind::Element)
        });
        for (position, index) in children.iter().enumerate() {
            if indent && position > 0 {
                self.out.push('\n');
            }
            if first_element == Some(position) {
                self.doctype(tree, *index);
            }
            self.node(tree, *index, 0, indent);
        }
        if self.properties.indent && !self.out.is_empty() && !self.out.ends_with('\n') {
            self.out.push('\n');
        }
    }

    fn declaration(&mut self) {
        if self.method == OutputMethod::Html || self.properties.omit_xml_declaration {
            return;
        }
        self.out.push_str("<?xml version=\"");
        self.out.push_str(self.properties.xml_version());
        self.out.push_str("\" encoding=\"");
        self.out.push_str(&self.properties.encoding);
        self.out.push('"');
        match self.properties.standalone {
            Standalone::Omit => {}
            Standalone::Yes => self.out.push_str(" standalone=\"yes\""),
            Standalone::No => self.out.push_str(" standalone=\"no\""),
        }
        self.out.push_str("?>");
        // Saxon and the suite's expected serializations put the declaration and the document
        // element on one line unless indentation was asked for.
        if self.properties.indent {
            self.out.push('\n');
        }
    }

    /// Writes the document type declaration for the document element at `index`.
    ///
    /// [Serialization 3.1 §5](https://www.w3.org/TR/xslt-xquery-serialization-31/) makes
    /// `doctype-system` the trigger: `doctype-public` on its own is ignored. The `html` and
    /// `xhtml` methods add one more case — an HTML version of 5 or later writes the bare
    /// `<!DOCTYPE html>` that HTML5 defines.
    fn doctype(&mut self, tree: &ResultTree, index: usize) {
        // XSLT 2.0 erratum E31: a zero-length `doctype-system`/`doctype-public` *overrides* one
        // an imported module set rather than requesting an empty identifier, so it behaves as
        // if the parameter had been left unset — `output-0312` imports a module that asks for
        // both and expects no document type declaration at all.
        let public = self
            .properties
            .doctype_public
            .as_deref()
            .filter(|value| !value.is_empty());
        let system = self
            .properties
            .doctype_system
            .as_deref()
            .filter(|value| !value.is_empty());
        let Some(node) = tree.raw(index) else { return };
        let name = qualified_name(node.prefix.as_deref(), node.name.as_ref());
        match system {
            Some(system) => {
                self.out.push_str("<!DOCTYPE ");
                self.out.push_str(&name);
                match public {
                    Some(public) => {
                        self.out
                            .push_str(&format!(" PUBLIC {} {}", quoted(public), quoted(system)))
                    }
                    None => self.out.push_str(&format!(" SYSTEM {}", quoted(system))),
                }
                self.out.push_str(">\n");
            }
            None => {
                if !matches!(self.method, OutputMethod::Html | OutputMethod::Xhtml)
                    || !self.properties.wants_html5_doctype()
                {
                    return;
                }
                // Serialization 3.1 §7.1: the HTML5 document type declaration goes in front of
                // an `html` document element only. `output-0215`, whose document element is
                // `html` in an alien namespace, asserts that no `DOCTYPE` appears at all.
                let qualifies = node.name.as_ref().is_some_and(|name| {
                    name.local_name().eq_ignore_ascii_case("html")
                        && match self.method {
                            OutputMethod::Xhtml => name.namespace() == Some(XHTML_NAMESPACE),
                            _ => matches!(name.namespace(), None | Some(XHTML_NAMESPACE)),
                        }
                });
                if !qualifies {
                    return;
                }
                // The HTML5 document type declaration is the fixed string `<!DOCTYPE html>`;
                // a prefix the result tree carried on its `html` element plays no part in it.
                let name = match self.method {
                    OutputMethod::Xhtml => node
                        .name
                        .as_ref()
                        .map(ExpandedName::local_name)
                        .unwrap_or(&name)
                        .to_owned(),
                    _ => name,
                };
                self.out.push_str(&format!("<!DOCTYPE {name}>\n"));
            }
        }
    }

    fn text_only(&mut self, tree: &ResultTree, index: usize) {
        let Some(node) = tree.raw(index) else { return };
        match node.kind {
            NodeKind::Text => {
                let text = self.prepare_raw(&node.value);
                self.out.push_str(&text);
            }
            NodeKind::Document | NodeKind::Element => {
                for child in &node.children {
                    self.text_only(tree, *child);
                }
            }
            _ => {}
        }
    }

    /// The children that survive indentation, with whitespace-only text dropped when the
    /// element is being indented.
    fn visible_children(&self, tree: &ResultTree, index: usize) -> Vec<usize> {
        let Some(node) = tree.raw(index) else {
            return Vec::new();
        };
        if !self.properties.indent {
            return node.children.clone();
        }
        let has_significant_text = node.children.iter().any(|child| {
            tree.raw(*child)
                .is_some_and(|data| data.kind == NodeKind::Text && !data.value.trim().is_empty())
        });
        if has_significant_text {
            return node.children.clone();
        }
        node.children
            .iter()
            .copied()
            .filter(|child| {
                tree.raw(*child)
                    .is_some_and(|data| data.kind != NodeKind::Text)
            })
            .collect()
    }

    /// Whether an element's children may be laid out one per line.
    fn indentable(&self, tree: &ResultTree, children: &[usize]) -> bool {
        !children.is_empty()
            && children.iter().all(|child| {
                tree.raw(*child)
                    .is_some_and(|data| data.kind != NodeKind::Text)
            })
    }

    fn node(&mut self, tree: &ResultTree, index: usize, depth: usize, indented: bool) {
        let Some(node) = tree.raw(index) else { return };
        if indented {
            self.push_indent(depth);
        }
        match node.kind {
            NodeKind::Element => self.element(tree, index, depth),
            NodeKind::Text => {
                if node.unescaped {
                    self.out.push_str(&self.prepare_raw(&node.value));
                } else {
                    self.out.push_str(&self.escape_text(&node.value));
                }
            }
            NodeKind::Comment => {
                self.out.push_str("<!--");
                self.out.push_str(&node.value);
                self.out.push_str("-->");
            }
            NodeKind::ProcessingInstruction => {
                let target = node
                    .name
                    .as_ref()
                    .map(ExpandedName::local_name)
                    .unwrap_or_default();
                self.out.push_str("<?");
                self.out.push_str(target);
                if !node.value.is_empty() {
                    self.out.push(' ');
                    self.out.push_str(&node.value);
                }
                self.out.push_str(if self.method == OutputMethod::Html {
                    ">"
                } else {
                    "?>"
                });
            }
            NodeKind::Document => {
                for child in node.children.clone() {
                    self.node(tree, child, depth, false);
                }
            }
            NodeKind::Attribute | NodeKind::Namespace => {}
        }
    }

    /// Whether the *prefix normalization* of
    /// [Serialization 3.1 §8](https://www.w3.org/TR/xslt-xquery-serialization-31/#xhtml-output)
    /// applies.
    ///
    /// §8 states it for the `xhtml` method, and §7 states a narrower rule for `html` whose two
    /// clauses contradict each other — which `output-0602a` records in its own description,
    /// settling the question by requiring the `html` method to normalize prefixes exactly as
    /// `xhtml` does. Both are HTML 5 rules: before HTML 5 there was no way to spell a foreign
    /// vocabulary in an HTML document at all, so a serialization that is not HTML 5 keeps the
    /// prefixes the tree holds.
    fn normalizes_prefixes(&self) -> bool {
        matches!(self.method, OutputMethod::Html | OutputMethod::Xhtml)
            && self.properties.wants_html5_doctype()
    }

    /// Whether a namespace binding an element holds is left out of its start tag.
    ///
    /// Two rules drop one. An element written unprefixed is written for a reader that has no
    /// notion of prefixes, so a prefix that none of its own attributes carries is not declared
    /// on it. And prefix normalization removes the namespace *nodes* of the three vocabularies
    /// HTML 5 spells without a prefix — "any namespace node for any of those three namespaces
    /// that was previously present on any element node … is also removed" — wherever they
    /// occur, not only on the elements that belong to them: `output-0602a` requires that no
    /// `xmlns:svg` survives anywhere, though the tree declares one on `body`. An attribute name
    /// has no default namespace to fall back on, so a prefix one of this element's attributes
    /// carries is kept whichever rule would otherwise drop it — which is what `output-0603a`
    /// asks for when it wants `xmlns:svg` back on the `mi` element carrying `svg:att`.
    fn drops_binding(
        &self,
        tree: &ResultTree,
        node: &crate::tree::ResultNode,
        prefix: Option<&str>,
        namespace: &str,
        unprefixed: bool,
    ) -> bool {
        let unused_prefix = unprefixed && prefix.is_some();
        let normalized = self.normalizes_prefixes() && is_html5_namespace(namespace);
        if !(unused_prefix || normalized) {
            return false;
        }
        !node.attributes.iter().any(|attribute| {
            tree.raw(*attribute)
                .is_some_and(|data| data.prefix.as_deref() == prefix)
        })
    }

    /// Writes `xmlns=""` when the element at `index` has no default namespace in scope but an
    /// ancestor start tag already put one there.
    ///
    /// An element's namespace nodes are its *whole* in-scope set, so a default binding that an
    /// ancestor has and this element has not is one the element removed. XML Namespaces §6.2
    /// spells that removal `xmlns=""`, and without it the start tag would read back as being in
    /// the ancestor's namespace — which is what `bug-1801`, `bug-1802` and `bug-5101` assert.
    /// `scope[..outer]` is the ancestors' half of the stack, before this element's own
    /// declarations were pushed onto it.
    fn undeclare_default_namespace(
        &mut self,
        tree: &ResultTree,
        index: usize,
        outer: usize,
        unprefixed: bool,
    ) {
        // The `html` method writes no namespace declarations at all, and the HTML5 spelling of
        // the `xhtml` method has just supplied its own default declaration.
        if unprefixed || self.method == OutputMethod::Html {
            return;
        }
        let inherited = self
            .scope
            .get(..outer)
            .unwrap_or_default()
            .iter()
            .rev()
            .find(|(prefix, _)| prefix.is_none())
            .is_some_and(|(_, uri)| !uri.is_empty());
        if !inherited {
            return;
        }
        let keeps_default = tree.raw(index).is_some_and(|node| {
            node.namespaces.iter().any(|namespace| {
                tree.raw(*namespace)
                    .is_some_and(|binding| binding.name.is_none())
            })
        });
        if keeps_default {
            return;
        }
        self.scope.push((None, String::new()));
        self.out.push_str(" xmlns=\"\"");
    }

    fn element(&mut self, tree: &ResultTree, index: usize, depth: usize) {
        let Some(node) = tree.raw(index) else { return };
        // Serialization 3.1 §7 (as amended by the resolution of W3C bug 20261): an HTML5
        // serialization is markup an HTML5 parser can read, and an HTML5 parser has no notion of
        // namespace prefixes. Every element in one of the three vocabularies HTML5 embeds is
        // therefore serialized unprefixed, with its namespace supplied as the default
        // declaration — `<h:html xmlns:h="…xhtml">` comes out as `<html xmlns="…xhtml">`.
        // `output-0215` keeps `<z:html xmlns:z="http://www.example.com/not-xhtml">` prefixed
        // precisely because an alien namespace has no unprefixed HTML5 spelling.
        let unprefixed = self.normalizes_prefixes()
            && node
                .name
                .as_ref()
                .and_then(ExpandedName::namespace)
                .is_some_and(is_html5_namespace);
        let name = if unprefixed {
            node.name
                .as_ref()
                .map(ExpandedName::local_name)
                .unwrap_or_default()
                .to_owned()
        } else {
            qualified_name(node.prefix.as_deref(), node.name.as_ref())
        };
        let local = node
            .name
            .as_ref()
            .map(ExpandedName::local_name)
            .unwrap_or_default()
            .to_owned();
        // The `html` method treats elements in no namespace as HTML; the `xhtml` method treats
        // elements in the XHTML namespace as HTML. Accepting either under both methods is what
        // a result tree built by a stylesheet that declares `xmlns="…xhtml"` needs.
        let html_element = node.name.as_ref().is_some_and(|name| {
            matches!(name.namespace(), None | Some(XHTML_NAMESPACE))
                && matches!(self.method, OutputMethod::Html | OutputMethod::Xhtml)
        });
        self.out.push('<');
        self.out.push_str(&name);

        let outer = self.scope.len();
        if unprefixed && let Some(uri) = node.name.as_ref().and_then(ExpandedName::namespace) {
            let already = self
                .scope
                .iter()
                .any(|(bound, bound_uri)| bound.is_none() && bound_uri == uri);
            if !already {
                self.scope.push((None, uri.to_owned()));
                self.out.push_str(" xmlns=\"");
                self.out.push_str(&self.escape_markup(uri));
                self.out.push('"');
            }
        }
        for namespace in node.namespaces.clone() {
            let Some(binding) = tree.raw(namespace) else {
                continue;
            };
            let prefix = binding.name.as_ref().map(ExpandedName::local_name);
            if prefix == Some("xml") {
                continue;
            }
            if self.drops_binding(tree, node, prefix, &binding.value, unprefixed) {
                continue;
            }
            let owned = prefix.map(str::to_owned);
            let already = self
                .scope
                .iter()
                .any(|(bound, uri)| bound.as_deref() == prefix && *uri == binding.value);
            self.scope.push((owned.clone(), binding.value.clone()));
            if already || (self.method == OutputMethod::Html && !self.normalizes_prefixes()) {
                continue;
            }
            match prefix {
                Some(prefix) => {
                    self.out.push_str(&format!(" xmlns:{prefix}=\""));
                }
                None => self.out.push_str(" xmlns=\""),
            }
            self.out.push_str(&self.escape_markup(&binding.value));
            self.out.push('"');
        }
        self.undeclare_default_namespace(tree, index, outer, unprefixed);

        for attribute in node.attributes.clone() {
            let Some(data) = tree.raw(attribute) else {
                continue;
            };
            let attribute_name = qualified_name(data.prefix.as_deref(), data.name.as_ref());
            if self.method == OutputMethod::Html
                && HTML_BOOLEAN_ATTRIBUTES.contains(&attribute_name.to_ascii_lowercase().as_str())
                && data.value.eq_ignore_ascii_case(&attribute_name)
            {
                self.out.push(' ');
                self.out.push_str(&attribute_name);
                continue;
            }
            // Serialization 3.1 §5.1.5 and the resolution of CR bug 2459: URI escaping
            // happens on the attribute's own value, and the character map is applied
            // afterwards, as part of writing the escaped value out.
            let mut value = data.value.clone();
            let uri_escaped = html_element
                && self.properties.escape_uri_attributes
                && is_uri_attribute(&local, &attribute_name);
            if uri_escaped {
                value = escape_uri(&value);
            }
            self.out.push(' ');
            self.out.push_str(&attribute_name);
            self.out.push_str("=\"");
            if uri_escaped {
                // Serialization 3.1 §5.1.6: a character map is not applied where URI escaping
                // has been. `character-map-009` maps `z` and expects `href="z-linkage.html"`
                // to come through untouched.
                self.out.push_str(&self.escape_attribute_run(&value));
            } else if data.unescaped {
                self.out.push_str(&self.prepare_raw(&value));
            } else {
                self.out.push_str(&self.escape_attribute(&value));
            }
            self.out.push('"');
        }

        let html_void =
            html_element && HTML_VOID_ELEMENTS.contains(&local.to_ascii_lowercase().as_str());
        let mut children = self.visible_children(tree, index);
        // `include-content-type` replaces any `<meta http-equiv="Content-Type">` the stylesheet
        // wrote with one naming the encoding actually used.
        let inject_content_type = html_element
            && self.properties.include_content_type
            && local.eq_ignore_ascii_case("head");
        if inject_content_type {
            children.retain(|child| !self.is_content_type_meta(tree, *child));
        }

        if children.is_empty() && !inject_content_type {
            match self.method {
                OutputMethod::Html => {
                    self.out.push('>');
                    if !html_void {
                        self.out.push_str(&format!("</{name}>"));
                    }
                }
                OutputMethod::Xhtml if html_void => self.out.push_str(" />"),
                OutputMethod::Xhtml => self.out.push_str(&format!("></{name}>")),
                _ => self.out.push_str("/>"),
            }
            self.scope.truncate(outer);
            return;
        }
        self.out.push('>');

        if inject_content_type {
            self.content_type_meta();
        }
        // Serialization 3.1 §7.3 gives the `html` method raw `script` and `style` content —
        // HTML parses those elements as CDATA — but §7.2 keeps the `xhtml` method's escaping
        // exactly as XML's, because an XML parser does not. `output-0107` writes an escaped
        // `document.write("&lt;p&gt;…")` and expects it back escaped.
        let raw_text = html_element
            && self.method == OutputMethod::Html
            && HTML_RAW_TEXT_ELEMENTS.contains(&local.to_ascii_lowercase().as_str());
        let cdata = node
            .name
            .as_ref()
            .is_some_and(|name| self.properties.is_cdata_element(name));
        // `suppress-indentation` names an element whose *content* is laid out as written, and
        // that applies to everything below it as well: `output-0232` suppresses `p` and expects
        // the `wizard`/`cap`/`small` elements nested inside one `p` to stay on its line.
        let enclosing_suppression = self.suppressing;
        self.suppressing = enclosing_suppression || self.suppressed(node.name.as_ref());
        let indent =
            self.properties.indent && self.indentable(tree, &children) && !self.suppressing;

        for child in &children {
            if indent {
                self.out.push('\n');
            }
            if raw_text
                && let Some(data) = tree.raw(*child)
                && data.kind == NodeKind::Text
            {
                self.out.push_str(&self.prepare_raw(&data.value));
                continue;
            }
            if cdata
                && let Some(data) = tree.raw(*child)
                && data.kind == NodeKind::Text
            {
                // Serialization 3.1 §5.1.2: text written with `disable-output-escaping="yes"`
                // is already-serialized markup, and a CDATA section escapes nothing — it only
                // spells the escaping differently. Wrapping such text would turn the markup it
                // carries back into character data, which is the opposite of what was asked
                // for, so `cdata-section-elements` passes over it (`bug-3801`).
                if data.unescaped {
                    self.out.push_str(&self.prepare_raw(&data.value));
                    continue;
                }
                // Serialization 3.1 §5.1.6 applies a character map to a text node only where
                // the text is escaped; the content of a CDATA section is written verbatim, so
                // `output-0115c` asserts that the map leaves it alone.
                self.write_cdata(&self.normalized(&data.value));
                continue;
            }
            self.node(tree, *child, depth + 1, indent);
        }
        if indent {
            self.out.push('\n');
            self.push_indent(depth);
        }
        self.suppressing = enclosing_suppression;
        self.out.push_str(&format!("</{name}>"));
        self.scope.truncate(outer);
    }

    /// Whether the node is a `meta` element that states the content type, which
    /// `include-content-type` replaces with one naming the encoding actually used.
    fn is_content_type_meta(&self, tree: &ResultTree, index: usize) -> bool {
        let Some(node) = tree.raw(index) else {
            return false;
        };
        if node.kind != NodeKind::Element {
            return false;
        }
        let is_meta = node.name.as_ref().is_some_and(|name| {
            matches!(name.namespace(), None | Some(XHTML_NAMESPACE))
                && name.local_name().eq_ignore_ascii_case("meta")
        });
        if !is_meta {
            return false;
        }
        node.attributes.iter().any(|attribute| {
            tree.raw(*attribute).is_some_and(|data| {
                data.name
                    .as_ref()
                    .is_some_and(|name| name.local_name().eq_ignore_ascii_case("http-equiv"))
                    && data.value.trim().eq_ignore_ascii_case("content-type")
            })
        })
    }

    /// The `media-type` in force.
    ///
    /// [`OutputProperties::effective_media_type`] defaults from the *declared* method, which is
    /// still `xml` while the method is being sniffed off the result tree; `output-0130` writes
    /// an XHTML page with no `xsl:output` at all and expects the injected `meta` to name
    /// `text/html`, the default Serialization 3.1 §5.1.10 gives the `xhtml` method.
    fn media_type(&self) -> &str {
        match self.properties.media_type.as_deref() {
            Some(value) if !value.is_empty() => value,
            _ => match self.method {
                OutputMethod::Html | OutputMethod::Xhtml => "text/html",
                OutputMethod::Text | OutputMethod::Adaptive => "text/plain",
                OutputMethod::Json => "application/json",
                OutputMethod::Xml => "application/xml",
            },
        }
    }

    /// Writes the `meta` element `include-content-type` asks for.
    fn content_type_meta(&mut self) {
        let media = self.media_type().to_owned();
        let encoding = self.properties.encoding.clone();
        // `character-map-017` notes that the map does reach the generated `meta` element:
        // "Not very desirable but that's what the spec says."
        let content = self.prepare_raw(&format!("{media}; charset={encoding}"));
        self.out.push_str(&format!(
            "<meta http-equiv=\"Content-Type\" content=\"{content}\""
        ));
        self.out.push_str(if self.method == OutputMethod::Xhtml {
            " />"
        } else {
            ">"
        });
    }

    fn suppressed(&self, name: Option<&ExpandedName>) -> bool {
        name.is_some_and(|name| self.properties.suppress_indentation.contains(name))
    }

    fn push_indent(&mut self, depth: usize) {
        for _ in 0..depth {
            self.out.push_str("   ");
        }
    }

    /// Writes a CDATA section, splitting it around any `]]>` the text contains and around any
    /// character the output encoding cannot represent.
    fn write_cdata(&mut self, text: &str) {
        let escaped = text
            .split("]]>")
            .collect::<Vec<_>>()
            .join("]]]]><![CDATA[>");
        let mut open = false;
        for character in escaped.chars() {
            if self.representable(character) {
                if !open {
                    self.out.push_str("<![CDATA[");
                    open = true;
                }
                self.out.push(character);
                continue;
            }
            if open {
                self.out.push_str("]]>");
                open = false;
            }
            self.out.push_str(&format!("&#{};", character as u32));
        }
        if open {
            self.out.push_str("]]>");
        }
    }

    /// Whether the output encoding can represent the character; anything else is written as a
    /// numeric character reference.
    fn representable(&self, character: char) -> bool {
        match self.properties.encoding.to_ascii_uppercase().as_str() {
            "US-ASCII" | "ASCII" | "ANSI_X3.4-1968" => character.is_ascii(),
            "ISO-8859-1" | "LATIN1" | "ISO_8859-1" => (character as u32) < 0x100,
            _ => true,
        }
    }

    /// Applies the `normalization-form` serialization parameter to a run of content.
    ///
    /// Normalization has to happen *before* escaping, so that a character the encoding cannot
    /// represent is written as a reference to the normalized code point rather than to the one
    /// the tree happened to hold — see `output-0115d`, which normalizes to NFD and expects the
    /// decomposed `c` to stay inside the CDATA section with only the combining cedilla
    /// escaped.
    fn normalized(&self, text: &str) -> String {
        match self.properties.normalization_form.as_deref() {
            None => text.to_owned(),
            Some(form) => normalize(text.to_owned(), Some(form)),
        }
    }

    /// Splits `text` into the runs a character map leaves alone and the replacements it
    /// substitutes.
    ///
    /// [Serialization 3.1 §5.1.6](https://www.w3.org/TR/xslt-xquery-serialization-31/#character-maps)
    /// makes a replacement immune to everything that follows: it is neither escaped nor
    /// Unicode-normalized. `character-map-028` pins the consequence down — with
    /// `normalization-form="NFD"` and a map from `c` to `#ç#`, the `ç` the *map* produced stays
    /// composed while the `ç` the *source* held is decomposed.
    fn runs(&self, text: &str) -> Vec<(bool, String)> {
        if self.properties.character_map.is_empty() {
            return vec![(false, text.to_owned())];
        }
        let mut out: Vec<(bool, String)> = Vec::new();
        for character in text.chars() {
            let (replacement, piece) = match self.properties.character_map.get(&character) {
                Some(replacement) => (true, replacement.clone()),
                None => (false, character.to_string()),
            };
            match out.last_mut() {
                Some((kind, run)) if *kind == replacement => run.push_str(&piece),
                _ => out.push((replacement, piece)),
            }
        }
        out
    }

    /// Writes content out the way a text node or an attribute value needs it: character-map
    /// replacements verbatim, everything else normalized and then escaped.
    fn prepare(&self, text: &str, escape: fn(&Self, &str) -> String) -> String {
        let mut out = String::with_capacity(text.len());
        for (replacement, run) in self.runs(text) {
            if replacement {
                out.push_str(&run);
                continue;
            }
            out.push_str(&escape(self, &self.normalized(&run)));
        }
        out
    }

    /// The same, for text that is written without escaping — `disable-output-escaping`, and
    /// the content of an HTML `script` or `style` element.
    fn prepare_raw(&self, text: &str) -> String {
        self.prepare(text, |_, run| run.to_owned())
    }

    /// Escapes a text node.
    fn escape_text(&self, text: &str) -> String {
        self.prepare(text, Self::escape_text_run)
    }

    fn escape_text_run(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        for character in text.chars() {
            match character {
                '&' => out.push_str("&amp;"),
                '<' => out.push_str("&lt;"),
                '>' => out.push_str("&gt;"),
                '\r' => out.push_str("&#xD;"),
                other if !self.representable(other) || self.escapes_as_reference(other) => {
                    out.push_str(&format!("&#{};", other as u32));
                }
                other => out.push(other),
            }
        }
        out
    }

    /// Whether the output method insists on writing `character` as a character reference even
    /// when the encoding could carry it.
    ///
    /// [Serialization 3.1 §7.3](https://www.w3.org/TR/xslt-xquery-serialization-31/#html-output)
    /// gives the C1 control range `#x7F`–`#x9F` no literal spelling under the HTML output
    /// method — HTML would read those bytes back as the Windows-1252 characters they are not —
    /// so an HTML5 serializer writes them numerically instead. `output-0195a` puts `#x9F` in a
    /// result tree serialized as HTML and asks for `&#159;`; `output-0102e` and `output-0103e`
    /// put `#x96` in an *attribute* value and ask for `&#150;`, so the rule is the escaper's
    /// rather than the text node's.
    ///
    /// [Serialization 3.1 §7.2](https://www.w3.org/TR/xslt-xquery-serialization-31/#xml-output)
    /// asks the same of the `xml` method under `version="1.1"`, for a wider range and a different
    /// reason: XML 1.1 admits the C0 and C1 controls *only* written as character references, so a
    /// serializer that writes one literally produces a document no parser will read back.
    /// `xml-version-007` puts `#x8` and `#x1F` in an attribute and asks for both back numerically.
    fn escapes_as_reference(&self, character: char) -> bool {
        match self.method {
            OutputMethod::Html | OutputMethod::Xhtml => matches!(character as u32, 0x7f..=0x9f),
            // Tab, newline and carriage return are outside the rule: they are ordinary
            // whitespace in every version of XML, and a text node full of `&#10;` would make an
            // indented result unreadable. The attribute escaper writes all three numerically for
            // its own reason — a literal one there is normalized away on the way back in.
            OutputMethod::Xml => {
                self.properties.xml_version() == "1.1"
                    && !matches!(character, '\t' | '\n' | '\r')
                    && matches!(character as u32, 0x1..=0x1f | 0x7f..=0x9f)
            }
            OutputMethod::Text | OutputMethod::Json | OutputMethod::Adaptive => false,
        }
    }

    /// Escapes an attribute value.
    fn escape_attribute(&self, text: &str) -> String {
        self.prepare(text, Self::escape_attribute_run)
    }

    /// Escapes a value a character map must not reach: a namespace URI, which is markup rather
    /// than content, so `character-map-017` expects `http://www.w3.org/1999/xhtml` to survive a
    /// map that rewrites `x`.
    fn escape_markup(&self, text: &str) -> String {
        self.escape_attribute_run(text)
    }

    fn escape_attribute_run(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        for character in text.chars() {
            match character {
                '&' => out.push_str("&amp;"),
                '<' => out.push_str("&lt;"),
                '>' if self.method.is_xml_like() => out.push_str("&gt;"),
                // The quotation mark closes the value, so it always becomes a reference. The
                // numeric spelling is written rather than `&quot;` because it is the one every
                // output method here reads back the same way, including `html`, whose entity
                // set is the document type's rather than XML's.
                '"' => out.push_str("&#34;"),
                '\n' => out.push_str("&#xA;"),
                '\r' => out.push_str("&#xD;"),
                '\t' => out.push_str("&#x9;"),
                other if !self.representable(other) || self.escapes_as_reference(other) => {
                    out.push_str(&format!("&#{};", other as u32));
                }
                other => out.push(other),
            }
        }
        out
    }
}

/// The namespaces the HTML5 syntax spells without a prefix.
fn is_html5_namespace(uri: &str) -> bool {
    matches!(
        uri,
        XHTML_NAMESPACE | "http://www.w3.org/2000/svg" | "http://www.w3.org/1998/Math/MathML"
    )
}

fn is_uri_attribute(element: &str, attribute: &str) -> bool {
    HTML_URI_ATTRIBUTES.iter().any(|(owner, name)| {
        owner.eq_ignore_ascii_case(element) && name.eq_ignore_ascii_case(attribute)
    })
}

/// Percent-encodes the characters `escape-uri-attributes` requires, leaving an existing escape
/// sequence alone.
///
/// [Serialization 3.1 §7.4](https://www.w3.org/TR/xslt-xquery-serialization-31/#HTML_URI-ESCAPING)
/// defers to Appendix B.2.1 of HTML 4.0, which escapes exactly the characters a URI reference
/// may not spell literally: everything outside the printable ASCII range `#x20`–`#x7E`. A space,
/// a quotation mark and the delimiters `<`, `>`, `{`, `}`, `|`, `\`, `^` and `` ` `` are all
/// inside it and stay as they were written — `output-0102c` writes
/// `href="% %C2%96 &#x96; a " …"` and asks for the space, the percent sign and the quotation
/// mark back unescaped, with only the non-ASCII `#x96` becoming `%C2%96`. The attribute escaper
/// then turns what is left into `&#34;`, `&lt;` and `&amp;` as it would for any other value.
fn escape_uri(value: &str) -> String {
    // The escaped octets are those of the URI in Unicode Normalization Form C, so that a
    // decomposed `å` and a precomposed one escape alike.
    let value = &oxixml_unicode::norm::nfc(value);
    let mut out = String::with_capacity(value.len());
    for character in value.chars() {
        if matches!(character as u32, 0x20..=0x7e) {
            out.push(character);
            continue;
        }
        let mut buffer = [0_u8; 4];
        for byte in character.encode_utf8(&mut buffer).as_bytes() {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

/// Quotes an external identifier, switching to apostrophes when it contains a quotation mark.
fn quoted(value: &str) -> String {
    if value.contains('"') {
        return format!("'{value}'");
    }
    format!("\"{value}\"")
}

/// Reassembles `prefix:local` for output.
fn qualified_name(prefix: Option<&str>, name: Option<&ExpandedName>) -> String {
    let local = name.map(ExpandedName::local_name).unwrap_or_default();
    match prefix.filter(|prefix| !prefix.is_empty()) {
        Some(prefix) => format!("{prefix}:{local}"),
        None => local.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uri_escaping_leaves_printable_ascii_alone() {
        assert_eq!(escape_uri("a/b?c=1"), "a/b?c=1");
        assert_eq!(escape_uri("a b \"c\""), "a b \"c\"");
        assert_eq!(escape_uri("\u{e9}"), "%C3%A9");
        assert_eq!(escape_uri("\u{96}"), "%C2%96");
        assert_eq!(escape_uri("a\tb"), "a%09b");
    }

    #[test]
    fn qualified_names_reassemble() {
        let name = ExpandedName::new(Some("urn:x"), "e");
        assert_eq!(qualified_name(Some("p"), Some(&name)), "p:e");
        assert_eq!(qualified_name(None, Some(&name)), "e");
        assert_eq!(qualified_name(Some(""), Some(&name)), "e");
    }

    #[test]
    fn uri_attributes_are_recognized_case_insensitively() {
        assert!(is_uri_attribute("A", "HREF"));
        assert!(!is_uri_attribute("a", "title"));
    }
}
