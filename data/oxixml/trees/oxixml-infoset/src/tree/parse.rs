//! Building a [`Tree`] from XML text, via the [`oxixml_xml`] pull parser.
//!
//! # Entities
//!
//! Character references and the five predefined entities always stand for the characters they
//! denote. A `&name;` that names neither is looked up among the general entities the document's
//! own `<!DOCTYPE>` declares, and replaced by what that declaration stands for, following the
//! two-phase rules of [XML 1.0 §4.4](https://www.w3.org/TR/xml/#entproc): a character reference
//! inside an `EntityValue` is resolved where the entity is *declared*, a general entity
//! reference where it is *used*.
//!
//! An internal entity usually stands for character data, and then the reference simply becomes
//! that text. It may equally stand for markup — [XML 1.0
//! §4.3.2](https://www.w3.org/TR/xml/#wf-entities) asks only that the replacement text match the
//! `content` production, so `<!ENTITY e "<b>x</b>">` is a legal declaration — and then the
//! replacement text is *parsed*, in the namespace scope of the reference, and the nodes it
//! yields are spliced in place of the reference. Replacement text that is neither, such as the
//! unbalanced `<b>` or a bare `a < b`, is rejected.
//!
//! An external parsed entity — `<!ENTITY e SYSTEM "e.xml">` — has no replacement text until
//! something reads it, which a non-validating processor is allowed not to do
//! ([XML 1.0 §5.1](https://www.w3.org/TR/xml/#proc-types)) and this parser does not do on its
//! own. [`Tree::parse_str_with_resolver`] takes the
//! [`ExternalEntityResolver`](crate::ExternalEntityResolver) that opts in; without one, such an
//! entity — like one that is not declared at all, or one that is recursive — leaves the document
//! rejected with [`ErrorCode::Fodc0006`](crate::ErrorCode::Fodc0006) rather than silently losing
//! content. The external *subset* of a document type declaration is never retrieved at all, so
//! an entity declared only there is undeclared as far as this parser is concerned.
//!
//! Which characters a reference may denote follows the version the XML declaration gives, in the
//! replacement text of an entity as well as in content: [XML 1.1](https://www.w3.org/TR/xml11/)
//! widens `Char` to every C0 control except `#x0`, and folds `#x85` and `#x2028` into `#xA`
//! alongside `#xD` when line endings are normalized.
//!
//! # The rest of the document type declaration
//!
//! The unparsed (`NDATA`) entities and the notations the internal subset declares become
//! properties of the document node, reachable through
//! [`Tree::unparsed_entity`](crate::Tree::unparsed_entity),
//! [`Tree::notation`](crate::Tree::notation) and their
//! [`XdmNode`](crate::XdmNode) counterparts. An `<!ATTLIST …>` reaches the data model twice
//! over: its defaults are supplied to every element that leaves the attribute out
//! ([XML 1.0 §3.3.2](https://www.w3.org/TR/xml/#sec-attr-defaults)), and its declared types
//! decide both how a value is normalized ([§3.3.3](https://www.w3.org/TR/xml/#AVNormalize)) and
//! which attributes carry the `is-id` and `is-idref` properties `fn:id` and `fn:idref` read.
//! Nothing else in the declaration — element declarations, parameter entities — reaches the data
//! model, which is what a non-validating processor is required to keep.
//!
//! Because this parser always runs the full well-formedness checker of [`oxixml_xml`], the
//! ceiling that checker places on entity expansion applies here too, and it is the lower of the
//! two: it counts *every* reference in the document rather than only the nested expansions that
//! actually multiply it. A document that repeats a reference more times than that is therefore
//! rejected here while `oxixml-dom` and `oxixml-sax`, which do not run the checker by default,
//! still accept it. The character data the three produce is identical wherever all three accept
//! the document, and where an entity stands for markup they necessarily differ: a DOM has a node
//! kind for an unexpanded entity reference and keeps one, while the data model built here has
//! none, so the only thing this parser can do is parse the replacement text.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use super::builder::{Annotation, TreeBuilder, resolve_system_id};
use super::dtd::{
    AttributeDeclaration, ContentModel, DeclaredType, ExternalEntityDeclaration, InternalSubset,
    external_subset_id, is_space, scan_declarations, scan_internal_subset,
};
use super::entity::{ExpandError, Expander, MAX_ENTITY_DEPTH, is_char};
use super::external::{EXTERNAL_SUBSET_NAME, ExternalEntity, ExternalEntityResolver};
use super::position::{PositionCursor, SourcePosition};
use super::tree_impl::Tree;
use crate::error::{ErrorCode, XdmError};
use oxixml_qname::{ExpandedName, NameContext, NamespaceStack, NamespacesVersion, XML_NAMESPACE};
use oxixml_xml::XmlVersion;
use oxixml_xml::escape::resolve_xml_entity;
use oxixml_xml::events::attributes::Attribute;
use oxixml_xml::events::{BytesRef, BytesStart, Event};
use oxixml_xml::name::QName;
use oxixml_xml::reader::Reader;

/// The name of the element the replacement text of an entity is wrapped in before it is parsed.
///
/// The wrapper is never added to the tree; it exists so that a reader which only accepts a whole
/// `document` accepts a `content` fragment — several top-level elements, character data, or
/// nothing at all — and so that markup which closes more than it opened is caught as the
/// well-formedness violation it is rather than escaping into the surrounding document.
const ENTITY_WRAPPER: &str = "entity";

/// How [`Tree::parse_str_with`] should turn XML text into a tree.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{ParseOptions, Tree, XdmNode};
///
/// // Boundary whitespace is significant by default…
/// let kept = Tree::parse_str("<a>\n  <b/>\n</a>")?;
/// assert_eq!(kept.document_element().expect("root").children().len(), 3);
///
/// // …unless it is stripped.
/// let stripped = Tree::parse_str_with(
///     "<a>\n  <b/>\n</a>",
///     ParseOptions::default().strip_boundary_whitespace(true),
/// )?;
/// assert_eq!(stripped.document_element().expect("root").children().len(), 1);
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct ParseOptions {
    document_uri: Option<String>,
    strip_boundary_whitespace: bool,
    namespaces_version: Option<NamespacesVersion>,
    record_source_positions: bool,
    read_external_subset: bool,
    preserve_element_content_whitespace: bool,
}

impl ParseOptions {
    /// Sets the document URI, which becomes the tree's `document-uri` property and seeds the
    /// base URI of every node.
    #[must_use]
    pub fn document_uri(mut self, uri: impl Into<String>) -> Self {
        self.document_uri = Some(uri.into());
        self
    }

    /// Drops text nodes that consist entirely of white space and have no non-whitespace sibling
    /// text — the "strip boundary whitespace" behaviour of XSLT's `xsl:strip-space` and of
    /// XQuery's default boundary-space policy.
    #[must_use]
    pub const fn strip_boundary_whitespace(mut self, strip: bool) -> Self {
        self.strip_boundary_whitespace = strip;
        self
    }

    /// Selects the version of Namespaces in XML whose rules apply. Defaults to
    /// [`NamespacesVersion::default`].
    #[must_use]
    pub const fn namespaces_version(mut self, version: NamespacesVersion) -> Self {
        self.namespaces_version = Some(version);
        self
    }

    /// Keeps the `(line, column)` each node began at, which
    /// [`XdmNode::source_position`](crate::XdmNode::source_position) then reports.
    ///
    /// Off by default: the data model has no such property, and a tree that is only queried has
    /// no use for one. Turn it on for a *source module* — a stylesheet or a query — whose
    /// diagnostics must name a place in the file the user wrote.
    ///
    /// Positions are recorded for element, text, comment and processing-instruction nodes.
    /// Attribute, namespace and document nodes have none: an attribute is written inside its
    /// element's start tag, which is where a diagnostic points anyway, and the other two have no
    /// markup of their own.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::{ParseOptions, SourcePosition, Tree, XdmNode};
    ///
    /// let tree = Tree::parse_str_with(
    ///     "<a>\n  <b/>\n</a>",
    ///     ParseOptions::default().record_source_positions(true),
    /// )?;
    /// let a = tree.document_element().expect("a document element");
    /// assert_eq!(a.source_position(), Some(SourcePosition { line: 1, column: 1 }));
    ///
    /// // Nothing is recorded unless it was asked for.
    /// let plain = Tree::parse_str("<a>\n  <b/>\n</a>")?;
    /// assert_eq!(plain.document_element().expect("root").source_position(), None);
    /// # Ok::<(), oxixml_infoset::XdmError>(())
    /// ```
    #[must_use]
    pub const fn record_source_positions(mut self, record: bool) -> Self {
        self.record_source_positions = record;
        self
    }

    /// Retrieves and scans the *external subset* of the document type declaration, when the
    /// document has one and parsing was given a resolver to retrieve it with.
    ///
    /// Off by default, and it has no effect at all unless the tree is built through
    /// [`Tree::parse_str_with_resolver`]: reading the external subset means reading whatever the
    /// `<!DOCTYPE … SYSTEM "…">` points at, and [XML 1.0
    /// §5.1](https://www.w3.org/TR/xml/#proc-types) explicitly allows a non-validating processor
    /// not to. Turning it on makes the entity, notation and attribute-default declarations written
    /// there count exactly as if they had been written in the internal subset — except that the
    /// internal subset is read first, so where both declare the same name the internal one binds
    /// ([XML 1.0 §2.8](https://www.w3.org/TR/xml/#sec-prolog-dtd)).
    ///
    /// The subset is requested from the resolver under the name
    /// [`EXTERNAL_SUBSET_NAME`](crate::EXTERNAL_SUBSET_NAME). A resolver that declines it —
    /// `Ok(None)` — leaves the document parsed as though the subset had not been read, which is
    /// the behaviour the specification permits; a resolver that *fails* rejects the document,
    /// because the caller asked for the subset and cannot be told it was skipped otherwise.
    ///
    /// Parameter entities are still not expanded, so a declaration assembled out of `%name;`
    /// references is not one this parser can see.
    #[must_use]
    pub const fn read_external_subset(mut self, read: bool) -> Self {
        self.read_external_subset = read;
        self
    }

    /// Keeps the white space that separates the children of an element the document type
    /// declaration gives an *element content* model.
    ///
    /// Off by default, because a [`Tree`] is an XDM instance rather than a raw infoset.
    /// [XML 1.0 §3.2.1](https://www.w3.org/TR/xml/#sec-element-content) gives an element type
    /// declared as `<!ELEMENT chapter (section*)>` element content — child elements "optionally
    /// separated by white space" — and [XML 1.0
    /// §2.10](https://www.w3.org/TR/xml/#sec-white-space) calls that separating white space
    /// *element content white space*. Constructing an XDM tree from an infoset discards it, which
    /// is why [XSLT 3.0 §4.4](https://www.w3.org/TR/xslt-30/#stylesheet-stripping) records that
    /// "elements that are defined in a DTD or a Schema to contain element-only content will have
    /// whitespace text nodes stripped, regardless of the `xsl:strip-space` and
    /// `xsl:preserve-space` declarations in the stylesheet".
    ///
    /// Turning this on keeps those text nodes, which is what a consumer that wants the infoset
    /// the document literally wrote — a formatter, a round-tripping editor — needs. Nothing else
    /// changes: an element declared `EMPTY`, `ANY` or with a mixed model has no element content
    /// white space to begin with, and a document without a document type declaration, or one
    /// whose content models this parser cannot see, is unaffected either way.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::{ParseOptions, Tree, XdmNode};
    ///
    /// let xml = "<!DOCTYPE a [<!ELEMENT a (b*)><!ELEMENT b EMPTY>]>\n<a>\n  <b/>\n</a>";
    ///
    /// // The layout white space between the children of `a` is element content white space.
    /// let tree = Tree::parse_str(xml)?;
    /// assert_eq!(tree.document_element().expect("root").children().len(), 1);
    ///
    /// // Unless the infoset is wanted verbatim.
    /// let verbatim = Tree::parse_str_with(
    ///     xml,
    ///     ParseOptions::default().preserve_element_content_whitespace(true),
    /// )?;
    /// assert_eq!(verbatim.document_element().expect("root").children().len(), 3);
    /// # Ok::<(), oxixml_infoset::XdmError>(())
    /// ```
    #[must_use]
    pub const fn preserve_element_content_whitespace(mut self, preserve: bool) -> Self {
        self.preserve_element_content_whitespace = preserve;
        self
    }
}

impl Tree {
    /// Parses `xml` into a tree, with well-formedness and namespace constraints enforced.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Fodc0006`] if `xml` is not a well-formed, namespace-well-formed XML
    /// document.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::{Tree, XdmNode};
    ///
    /// let tree = Tree::parse_str(r#"<p xmlns="urn:x">hello <b>world</b></p>"#)?;
    /// let p = tree.document_element().expect("a document element");
    /// assert_eq!(
    ///     p.node_name().map(|name| name.to_string()).as_deref(),
    ///     Some("{urn:x}p")
    /// );
    /// assert_eq!(p.string_value(), "hello world");
    ///
    /// assert!(Tree::parse_str("<a><b></a>").is_err());
    /// assert!(Tree::parse_str("<a:b/>").is_err(), "the prefix is not bound");
    /// # Ok::<(), oxixml_infoset::XdmError>(())
    /// ```
    pub fn parse_str(xml: &str) -> Result<Self, XdmError> {
        Self::parse_str_with(xml, ParseOptions::default())
    }

    /// Parses `xml` into a tree under `options`.
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Fodc0006`] if `xml` is not a well-formed, namespace-well-formed XML
    /// document.
    pub fn parse_str_with(xml: &str, options: ParseOptions) -> Result<Self, XdmError> {
        Parser::new(xml, options, None).run()
    }

    /// Parses `xml` under `options`, retrieving the external parsed general entities it
    /// references through `resolver`.
    ///
    /// Without a resolver — which is what every other entry point uses — an `&name;` whose
    /// declaration is `<!ENTITY name SYSTEM "…">` has no replacement text and the document is
    /// rejected. With one, the text the resolver hands back is parsed as `content`
    /// ([XML 1.0 §4.3.2](https://www.w3.org/TR/xml/#wf-entities)) in the namespace scope of the
    /// reference and spliced into the tree in its place; a resolver that returns `Ok(None)`
    /// declines, which leaves the document rejected exactly as if it had not been asked.
    ///
    /// Nothing else changes: internal entities, character references and every constraint on the
    /// document are treated as they are by [`Tree::parse_str_with`].
    ///
    /// # Errors
    ///
    /// [`ErrorCode::Fodc0006`] if `xml` is not a well-formed, namespace-well-formed XML document,
    /// if an entity the document references cannot be retrieved, or if what the resolver hands
    /// back is not the `content` production.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_infoset::{
    ///     ExternalEntity, ExternalEntityError, ExternalEntityResolver, ParseOptions, Tree,
    ///     XdmNode,
    /// };
    ///
    /// struct Table;
    /// impl ExternalEntityResolver for Table {
    ///     fn resolve(
    ///         &mut self,
    ///         entity: &ExternalEntity<'_>,
    ///     ) -> Result<Option<String>, ExternalEntityError> {
    ///         Ok((entity.name == "chapter").then(|| "<p>text</p>".to_owned()))
    ///     }
    /// }
    ///
    /// let xml = r#"<!DOCTYPE d [<!ENTITY chapter SYSTEM "ch1.xml">]><d>&chapter;</d>"#;
    /// let tree = Tree::parse_str_with_resolver(xml, ParseOptions::default(), &mut Table)?;
    /// assert_eq!(tree.document_element().expect("root").string_value(), "text");
    /// # Ok::<(), oxixml_infoset::XdmError>(())
    /// ```
    pub fn parse_str_with_resolver(
        xml: &str,
        options: ParseOptions,
        resolver: &mut dyn ExternalEntityResolver,
    ) -> Result<Self, XdmError> {
        Parser::new(xml, options, Some(resolver)).run()
    }
}

fn malformed(message: impl Into<String>) -> XdmError {
    XdmError::new(ErrorCode::Fodc0006, message)
}

/// Normalizes line endings per [XML 1.0 §2.11](https://www.w3.org/TR/xml/#sec-line-ends):
/// `\r\n` and a lone `\r` both become `\n`. The operation is idempotent.
///
/// [XML 1.1 §2.11](https://www.w3.org/TR/xml11/#sec-line-ends) extends the set: `#xD #x85`, a
/// lone `#x85` (NEL) and `#x2028` (LINE SEPARATOR) become `#xA` as well. Neither version touches
/// a character *reference* to one of those code points — normalization happens to the entity's
/// characters before parsing, and a reference is not yet a character at that point.
fn normalize_line_endings(text: &str, version: XmlVersion) -> String {
    let xml11 = matches!(version, XmlVersion::Explicit1_1);
    let breaks_a_line = |character: char| {
        character == '\r' || (xml11 && matches!(character, '\u{85}' | '\u{2028}'))
    };
    if !text.contains(breaks_a_line) {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(character) = chars.next() {
        if character == '\r' {
            // `#xD #xA` and, in XML 1.1, `#xD #x85` are single line ends.
            if chars.peek() == Some(&'\n') || (xml11 && chars.peek() == Some(&'\u{85}')) {
                chars.next();
            }
            out.push('\n');
        } else if breaks_a_line(character) {
            out.push('\n');
        } else {
            out.push(character);
        }
    }
    out
}

/// Removes the `TextDecl` an external parsed entity may begin with
/// ([XML 1.0 §4.3.1](https://www.w3.org/TR/xml/#sec-TextDecl)), which declares the entity's own
/// version and encoding and is not part of what it stands for.
fn strip_text_declaration(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("<?xml") else {
        return text;
    };
    // `<?xmlfoo?>` is a processing instruction, not a text declaration.
    if !rest.starts_with([' ', '\t', '\r', '\n']) {
        return text;
    }
    match rest.find("?>") {
        Some(end) => &rest[end + "?>".len()..],
        None => text,
    }
}

/// Rejects a character the document's XML version does not admit.
///
/// Text a resolver hands back was never part of the document, so it has not been through the
/// well-formedness checker that vets everything else; the `Char` production is applied to it
/// here instead, before it reaches a reader that is not running that check. `what` names the
/// retrieved thing in the resulting message.
fn check_characters(text: &str, version: XmlVersion, what: &str) -> Result<(), XdmError> {
    match text.chars().find(|&character| !is_char(character, version)) {
        Some(character) => Err(malformed(format!(
            "{what} contains U+{:04X}, which the `Char` production does not admit",
            u32::from(character)
        ))),
        None => Ok(()),
    }
}

/// The prefix an attribute name declares a namespace for, if it declares one.
///
/// `Some(None)` is the default-namespace declaration `xmlns`, `Some(Some(prefix))` is
/// `xmlns:prefix`, and `None` is an ordinary attribute name. Namespaces in XML §3 gives the two
/// `xmlns` forms this meaning wherever the attribute comes from, so a name supplied by attribute
/// defaulting is read here exactly as one written in the start tag is.
fn declared_prefix(name: &[u8]) -> Option<Option<&[u8]>> {
    if name == b"xmlns" {
        return Some(None);
    }
    name.strip_prefix(b"xmlns:".as_slice()).map(Some)
}

/// The attributes the declarations of an element supply for a start tag that left them out:
/// the ones with a default value whose name the tag did not write.
fn defaulted_attributes(
    declared: &[AttributeDeclaration],
    start: &BytesStart<'_>,
) -> Result<Vec<AttributeDeclaration>, XdmError> {
    if declared.is_empty() {
        return Ok(Vec::new());
    }
    let mut written: Vec<&[u8]> = Vec::new();
    for attribute in start.attributes() {
        let attribute = attribute.map_err(|error| malformed(error.to_string()))?;
        written.push(attribute.key.into_inner());
    }
    Ok(declared
        .iter()
        .filter(|declaration| {
            declaration.default.is_some() && !written.contains(&declaration.name.as_bytes())
        })
        .cloned()
        .collect())
}

/// The type `<!ATTLIST …>` declared for the attribute written as `key`, or `CDATA` when nothing
/// declared it.
///
/// The match is lexical, as [XML 1.0 §3.3](https://www.w3.org/TR/xml/#attdecls) writes it: a
/// declaration of `id` says nothing about an attribute written `p:id`, even where the prefix is
/// bound to no namespace.
fn declared_type_of(declared: &[AttributeDeclaration], key: &[u8]) -> DeclaredType {
    declared
        .iter()
        .find(|declaration| declaration.name.as_bytes() == key)
        .map_or(DeclaredType::String, |declaration| {
            declaration.declared_type
        })
}

/// The type an attribute is treated as having: the one its declaration gave it, except that
/// `xml:id` is an ID however — and whether or not — it was declared.
///
/// [xml:id 1.0 §4](https://www.w3.org/TR/xml-id/#processing) makes an `xml:id` attribute an ID
/// and has its value normalized "as if the attribute were declared to be of type `ID`", which is
/// what makes `id(' x')` find `xml:id="x "`.
fn effective_type(name: &ExpandedName, declared: DeclaredType) -> DeclaredType {
    match name.namespace() == Some(XML_NAMESPACE) && name.local_name() == "id" {
        true => DeclaredType::Id,
        false => declared,
    }
}

/// Applies the second half of attribute-value normalization
/// ([XML 1.0 §3.3.3](https://www.w3.org/TR/xml/#AVNormalize)): for a type other than `CDATA` the
/// value is a sequence of tokens, so leading and trailing spaces go and a run of them counts as
/// one.
///
/// Only the space character itself, so a line feed that arrived as `&#10;` is left where it is.
fn tokenize(value: String, declared: DeclaredType) -> String {
    if !declared.is_tokenized() {
        return value;
    }
    value
        .split(' ')
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// The data-model properties a declared type gives an attribute node.
fn annotation_of(declared: DeclaredType) -> Annotation {
    Annotation {
        is_id: declared.is_id(),
        is_idref: declared.is_idref(),
        ..Annotation::default()
    }
}

struct Parser<'a> {
    xml: &'a str,
    options: ParseOptions,
    builder: TreeBuilder,
    namespaces: NamespaceStack,
    depth: usize,
    /// Text accumulated since the last non-text event, together with whether any of it was
    /// non-whitespace or came from a CDATA section (both of which make it unstrippable).
    pending_text: String,
    pending_is_significant: bool,
    /// The version declared by the XML declaration, which decides how attribute values are
    /// normalized (XML 1.1 also folds `&#xD;` and friends).
    xml_version: XmlVersion,
    /// The internal general entities declared by the document's `<!DOCTYPE>`, keyed on their
    /// name, each holding the replacement text of
    /// [XML 1.0 §4.5](https://www.w3.org/TR/xml/#intern-replacement).
    entities: HashMap<String, String>,
    /// The external *parsed* general entities the same declaration names, which only a resolver
    /// can turn into replacement text.
    external_entities: BTreeMap<String, ExternalEntityDeclaration>,
    /// The attribute declarations the same document type declaration makes, keyed on the element
    /// name as written. Empty for the overwhelming majority of documents, which is the case the
    /// element path is written to keep free.
    ///
    /// The lists are shared rather than owned outright so that every start tag of a declared
    /// element can take one out of the map — which is borrowed for as long as the tree is being
    /// built — without copying the names and default values it holds.
    attribute_declarations: BTreeMap<String, Rc<[AttributeDeclaration]>>,
    /// The content model each `<!ELEMENT …>` declaration of the same document type declaration
    /// gives an element type, keyed on the element name as written — the same lexical match the
    /// attribute-list declarations use. Empty when the document declares no element type, which
    /// is the case the element path is written to keep free.
    content_models: BTreeMap<String, ContentModel>,
    /// Whether each element currently open was declared with an *element content* model, one
    /// entry per open element and the innermost last. Empty — and never grown — for a document
    /// that declares no element content model at all.
    element_content: Vec<bool>,
    /// The caller's retrieval hook, when parsing was asked to retrieve external entities.
    resolver: Option<&'a mut dyn ExternalEntityResolver>,
    /// The entities whose replacement text is currently being parsed, innermost last. An entity
    /// that reaches itself through this stack is recursive, which XML 1.0 forbids.
    open_entities: Vec<String>,
    /// The expansion budget for this document's entity references.
    expander: Expander,
    /// Turns the reader's byte offsets into line and column numbers. `None` unless the caller
    /// asked for source positions, so a parse that does not want them counts no line ends.
    cursor: Option<PositionCursor>,
    /// Where the event now being handled began, when positions are being recorded.
    event_position: Option<SourcePosition>,
    /// Where the text accumulated in `pending_text` began — the position of its *first*
    /// character, which is the one the resulting text node gets.
    pending_position: Option<SourcePosition>,
}

impl<'a> Parser<'a> {
    fn new(
        xml: &'a str,
        options: ParseOptions,
        resolver: Option<&'a mut dyn ExternalEntityResolver>,
    ) -> Self {
        let builder = match &options.document_uri {
            Some(uri) => TreeBuilder::with_document_uri(uri.clone()),
            None => TreeBuilder::new(),
        };
        let namespaces = NamespaceStack::new(options.namespaces_version.unwrap_or_default());
        let cursor = options
            .record_source_positions
            .then(PositionCursor::default);
        Self {
            xml,
            options,
            builder,
            namespaces,
            depth: 0,
            pending_text: String::new(),
            pending_is_significant: false,
            xml_version: XmlVersion::default(),
            entities: HashMap::new(),
            external_entities: BTreeMap::new(),
            attribute_declarations: BTreeMap::new(),
            content_models: BTreeMap::new(),
            element_content: Vec::new(),
            resolver,
            open_entities: Vec::new(),
            expander: Expander::default(),
            cursor,
            event_position: None,
            pending_position: None,
        }
    }

    fn run(mut self) -> Result<Tree, XdmError> {
        let mut reader = Reader::from_str(self.xml);
        reader.config_mut().check_well_formedness = true;
        reader.config_mut().expand_empty_elements = false;
        loop {
            // The reader's events cover the document without gaps, so where the previous one
            // ended is where this one begins.
            let start = reader.buffer_position();
            let event = reader
                .read_event()
                .map_err(|error| malformed(error.to_string()))?;
            self.event_position = self.cursor.as_mut().map(|cursor| {
                cursor.position_at(self.xml, usize::try_from(start).unwrap_or(usize::MAX))
            });
            if !self.handle_event(event)? {
                break;
            }
        }
        if self.depth != 0 {
            return Err(malformed("the document ends with unclosed elements"));
        }
        self.builder.build()
    }

    /// Adds one event to the tree being built. Returns whether any event follows.
    fn handle_event(&mut self, event: Event<'_>) -> Result<bool, XdmError> {
        match event {
            Event::Start(start) => {
                self.flush_text()?;
                self.open_element(&start)?;
            }
            Event::Empty(start) => {
                self.flush_text()?;
                self.open_element(&start)?;
                self.close_element()?;
            }
            Event::End(_) => {
                self.flush_text()?;
                self.close_element()?;
            }
            Event::Text(text) => {
                let decoded = text
                    .decode()
                    .map_err(|error| malformed(error.to_string()))?;
                self.push_text(&normalize_line_endings(&decoded, self.xml_version), false);
            }
            Event::CData(cdata) => {
                let decoded = cdata
                    .decode()
                    .map_err(|error| malformed(error.to_string()))?;
                self.push_text(&normalize_line_endings(&decoded, self.xml_version), true);
            }
            Event::GeneralRef(reference) => self.reference(&reference)?,
            Event::Comment(comment) => {
                self.flush_text()?;
                let decoded = comment
                    .decode()
                    .map_err(|error| malformed(error.to_string()))?;
                self.builder.set_source_position(self.current_position());
                self.builder
                    .comment(normalize_line_endings(&decoded, self.xml_version))?;
            }
            Event::PI(instruction) => {
                self.flush_text()?;
                let target = std::str::from_utf8(instruction.target())
                    .map_err(|error| malformed(error.to_string()))?
                    .to_owned();
                let content = std::str::from_utf8(instruction.content())
                    .map_err(|error| malformed(error.to_string()))?;
                self.builder.set_source_position(self.current_position());
                self.builder.processing_instruction(
                    target,
                    normalize_line_endings(content.trim_start_matches(is_space), self.xml_version),
                )?;
            }
            Event::Decl(declaration) => {
                self.flush_text()?;
                self.xml_version = declaration
                    .xml_version()
                    .map_err(|error| malformed(error.to_string()))?;
                // The declaration precedes every character reference in the document,
                // including the ones inside an entity's replacement text, so the expander
                // never has to reinterpret what it already resolved.
                self.expander.set_version(self.xml_version);
            }
            Event::DocType(doctype) => {
                self.flush_text()?;
                // The declaration itself is not part of the data model, but five things
                // its subsets declare are: the general entities, without which a `&name;` in
                // content has nothing to stand for, the external parsed ones a resolver may
                // retrieve, the unparsed entities and notations, which become properties of the
                // document node, the attribute declarations, whose defaults put attributes —
                // namespace declarations among them — on elements that did not write them, and
                // whose types decide normalization and ID-ness, and the element content models,
                // which decide which white space is element content white space.
                let decoded = doctype
                    .decode()
                    .map_err(|error| malformed(error.to_string()))?;
                let mut subset = scan_internal_subset(&decoded, self.xml_version);
                self.read_external_subset(&decoded, &mut subset)?;
                self.entities = subset.entities;
                self.external_entities = subset.external_entities;
                self.attribute_declarations = subset
                    .attribute_declarations
                    .into_iter()
                    .map(|(element, declared)| (element, Rc::from(declared)))
                    .collect();
                self.content_models = subset.content_models;
                for (name, declaration) in subset.unparsed_entities {
                    self.builder.declare_unparsed_entity(
                        name,
                        declaration.public_id.as_deref(),
                        declaration.system_id,
                        declaration.notation,
                    )?;
                }
                for (name, declaration) in subset.notations {
                    self.builder.declare_notation(
                        name,
                        declaration.public_id.as_deref(),
                        declaration.system_id.as_deref(),
                    )?;
                }
            }
            Event::Eof => {
                self.flush_text()?;
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Where the event now being handled began, or `None` when the position is not known.
    ///
    /// It is not known while an entity's replacement text is being parsed: that text is read by a
    /// second reader, over a synthesized source, so its offsets index the replacement text rather
    /// than the document. Reporting them would put a plausible but wrong line number on every node
    /// an entity contributes.
    fn current_position(&self) -> Option<SourcePosition> {
        match self.open_entities.is_empty() {
            true => self.event_position,
            false => None,
        }
    }

    fn push_text(&mut self, text: &str, significant: bool) {
        if text.is_empty() {
            return;
        }
        if self.pending_text.is_empty() {
            self.pending_position = self.current_position();
        }
        self.pending_text.push_str(text);
        // Text that reached the document through an entity reference is never boundary white
        // space: the document asked for it by name, and stripping it would change what the
        // reference stands for.
        self.pending_is_significant |= significant
            || !self.open_entities.is_empty()
            || text.chars().any(|c| !matches!(c, ' ' | '\t' | '\r' | '\n'));
    }

    fn flush_text(&mut self) -> Result<(), XdmError> {
        let text = std::mem::take(&mut self.pending_text);
        let significant = std::mem::take(&mut self.pending_is_significant);
        let position = self.pending_position.take();
        if text.is_empty() || self.depth == 0 {
            // Whitespace outside the document element is not part of the data model, and the
            // reader has already rejected anything else there.
            return Ok(());
        }
        if self.options.strip_boundary_whitespace && !significant {
            return Ok(());
        }
        if self.in_element_content_whitespace(&text) {
            return Ok(());
        }
        self.builder.set_source_position(position);
        self.builder.text(text)?;
        Ok(())
    }

    /// Whether `text`, about to become a text node inside the innermost open element, is
    /// *element content white space* the XDM construction discards.
    ///
    /// [XML 1.0 §3.2.1](https://www.w3.org/TR/xml/#sec-element-content) gives an element type
    /// declared with a `children` content model element content: child elements "optionally
    /// separated by white space". That separating white space is not character data —
    /// [XML 1.0 §2.10](https://www.w3.org/TR/xml/#sec-white-space) names it, and
    /// [XSLT 3.0 §4.4](https://www.w3.org/TR/xslt-30/#stylesheet-stripping) states the
    /// consequence for the tree: "elements that are defined in a DTD or a Schema to contain
    /// element-only content will have whitespace text nodes stripped, regardless of the
    /// `xsl:strip-space` and `xsl:preserve-space` declarations in the stylesheet".
    ///
    /// The test is on the *whole* accumulated run rather than on each event: a run that holds one
    /// non-whitespace character is character data, and character data written where the model
    /// forbids it is a validity error this non-validating parser reports by keeping the text.
    fn in_element_content_whitespace(&self, text: &str) -> bool {
        if self.options.preserve_element_content_whitespace {
            return false;
        }
        self.element_content.last().copied().unwrap_or(false) && text.chars().all(is_space)
    }

    fn open_element(&mut self, start: &BytesStart<'_>) -> Result<(), XdmError> {
        self.namespaces.push_scope();
        let position = self.current_position();
        let declared = self.declarations_for(start)?;
        let defaults = defaulted_attributes(&declared, start)?;

        // Namespace declarations must be processed before any name is resolved, because they
        // are in scope on the element that carries them. A declaration the start tag wrote comes
        // first, so it is the one that binds when the DTD declares the same prefix.
        let mut declarations: Vec<(Option<String>, String)> = Vec::new();
        for attribute in start.attributes() {
            let attribute = attribute.map_err(|error| malformed(error.to_string()))?;
            let key = attribute.key.as_ref();
            let Some(prefix) = declared_prefix(key) else {
                continue;
            };
            let prefix = prefix
                .map(|prefix| {
                    std::str::from_utf8(prefix)
                        .map(str::to_owned)
                        .map_err(|error| malformed(error.to_string()))
                })
                .transpose()?;
            let value = self.normalized_value(&attribute)?.into_owned();
            self.namespaces
                .declare(prefix.as_deref(), &value)
                .map_err(|error| malformed(error.to_string()))?;
            declarations.push((prefix, value));
        }
        for default in &defaults {
            let Some(prefix) = declared_prefix(default.name.as_bytes()) else {
                continue;
            };
            let prefix = prefix
                .map(|prefix| {
                    std::str::from_utf8(prefix)
                        .map(str::to_owned)
                        .map_err(|error| malformed(error.to_string()))
                })
                .transpose()?;
            let value = self.default_value(default, default.declared_type)?;
            self.namespaces
                .declare(prefix.as_deref(), &value)
                .map_err(|error| malformed(error.to_string()))?;
            declarations.push((prefix, value));
        }

        let name = self.resolve(start.name().as_ref(), NameContext::Element)?;
        self.builder.set_source_position(position);
        self.builder.start_element(name)?;
        for (prefix, value) in declarations {
            self.builder.declare_namespace(prefix.as_deref(), &value)?;
        }

        for attribute in start.attributes() {
            let attribute = attribute.map_err(|error| malformed(error.to_string()))?;
            let key = attribute.key.as_ref();
            if declared_prefix(key).is_some() {
                continue;
            }
            let name = self.resolve(key, NameContext::Attribute)?;
            let declared_type = effective_type(&name, declared_type_of(&declared, key));
            let value = self.normalized_value(&attribute)?.into_owned();
            self.builder.annotated_attribute(
                name,
                tokenize(value, declared_type),
                annotation_of(declared_type),
            )?;
        }
        for default in &defaults {
            if declared_prefix(default.name.as_bytes()).is_some() {
                continue;
            }
            let name = self.resolve(default.name.as_bytes(), NameContext::Attribute)?;
            let declared_type = effective_type(&name, default.declared_type);
            let value = self.default_value(default, declared_type)?;
            self.builder
                .annotated_attribute(name, value, annotation_of(declared_type))?;
        }

        self.depth += 1;
        self.push_content_model(start)?;
        Ok(())
    }

    /// Records whether the element just opened was declared with an element content model, so
    /// that the white space between its children can be recognized when it is flushed.
    ///
    /// The declaration is matched to the element by the qualified name as written, exactly as
    /// [`declarations_for`](Self::declarations_for) matches an attribute-list declaration:
    /// element declarations are lexical, so two prefixes bound to the same namespace name two
    /// different element types here.
    fn push_content_model(&mut self, start: &BytesStart<'_>) -> Result<(), XdmError> {
        if self.content_models.is_empty() {
            return Ok(());
        }
        let name = start.name();
        let element =
            std::str::from_utf8(name.as_ref()).map_err(|error| malformed(error.to_string()))?;
        let element_only = self
            .content_models
            .get(element)
            .is_some_and(|model| model.is_element_only());
        self.element_content.push(element_only);
        Ok(())
    }

    /// The attribute-list declarations that apply to this start tag.
    ///
    /// [XML 1.0 §3.3.2](https://www.w3.org/TR/xml/#sec-attr-defaults) matches the declaration to
    /// the element by name, and the name in question is the qualified name as written — Namespaces
    /// in XML leaves attribute-list declarations lexical, so two prefixes bound to the same
    /// namespace name two different elements here.
    fn declarations_for(
        &self,
        start: &BytesStart<'_>,
    ) -> Result<Rc<[AttributeDeclaration]>, XdmError> {
        if self.attribute_declarations.is_empty() {
            return Ok(Rc::from([]));
        }
        let name = start.name();
        let element =
            std::str::from_utf8(name.as_ref()).map_err(|error| malformed(error.to_string()))?;
        Ok(match self.attribute_declarations.get(element) {
            Some(declared) => Rc::clone(declared),
            None => Rc::from([]),
        })
    }

    /// Normalizes the value an attribute-list declaration supplies, exactly as the same value
    /// written in the start tag would have been normalized
    /// ([XML 1.0 §3.3.3](https://www.w3.org/TR/xml/#AVNormalize)).
    ///
    /// The references inside it are resolved here rather than where the declaration was scanned,
    /// because an entity may be declared after the attribute list that uses it.
    fn default_value(
        &mut self,
        declaration: &AttributeDeclaration,
        declared_type: DeclaredType,
    ) -> Result<String, XdmError> {
        let literal = declaration.default.as_deref().unwrap_or_default();
        let attribute = Attribute {
            key: QName(declaration.name.as_bytes()),
            value: Cow::Borrowed(literal.as_bytes()),
        };
        let value = self.normalized_value(&attribute)?.into_owned();
        Ok(tokenize(value, declared_type))
    }

    /// Retrieves and scans the external subset the document type declaration names, when the
    /// caller asked for it and supplied a resolver to retrieve it with.
    fn read_external_subset(
        &mut self,
        doctype: &str,
        subset: &mut InternalSubset,
    ) -> Result<(), XdmError> {
        if !self.options.read_external_subset {
            return Ok(());
        }
        let (Some(declaration), Some(resolver)) =
            (external_subset_id(doctype), self.resolver.as_mut())
        else {
            return Ok(());
        };
        let request = ExternalEntity {
            name: EXTERNAL_SUBSET_NAME,
            public_id: declaration.public_id.as_deref(),
            system_id: &declaration.system_id,
        };
        let retrieved = resolver.resolve(&request).map_err(|error| {
            malformed(format!(
                "the external subset declared with the system identifier {:?} could not be \
                 retrieved: {error}",
                declaration.system_id
            ))
        })?;
        // Declining is not a failure: a non-validating processor is allowed not to read the
        // external subset at all, so a resolver that will not produce it leaves the document
        // exactly as it would have been without the request.
        let Some(text) = retrieved else {
            return Ok(());
        };
        check_characters(
            &text,
            self.xml_version,
            "the external subset of the document type declaration",
        )?;
        // A byte-order mark is an encoding signature and the `TextDecl` declares the subset's own
        // version and encoding; neither is a markup declaration.
        let content = strip_text_declaration(text.strip_prefix('\u{FEFF}').unwrap_or(&text));
        let content = normalize_line_endings(content, self.xml_version);
        scan_declarations(subset, &content, self.xml_version);
        Ok(())
    }

    fn close_element(&mut self) -> Result<(), XdmError> {
        self.builder.end_element()?;
        self.namespaces.pop_scope();
        self.depth = self.depth.saturating_sub(1);
        self.element_content.pop();
        Ok(())
    }

    /// Resolves a `&…;` reference found in content, appending what it stands for.
    ///
    /// Character references and the five predefined general entities stand for themselves; any
    /// other name must have been declared in the document's own internal subset. An internal
    /// entity's replacement text is expanded in turn (XML 1.0 §4.4) when it is character data
    /// and parsed as `content` when it is markup; an external one is retrieved through the
    /// caller's resolver, if there is one. An entity that is undeclared, recursive, or that
    /// nothing can retrieve has nothing this parser can put in the tree, so it is reported as a
    /// malformed document rather than silently dropped.
    fn reference(&mut self, reference: &BytesRef<'_>) -> Result<(), XdmError> {
        if let Some(character) = reference
            .resolve_char_ref()
            .map_err(|error| malformed(error.to_string()))?
        {
            self.push_text(character.encode_utf8(&mut [0_u8; 4]), true);
            return Ok(());
        }
        let name = reference
            .decode()
            .map_err(|error| malformed(error.to_string()))?;
        if let Some(predefined) = resolve_xml_entity(&name) {
            self.push_text(predefined, true);
            return Ok(());
        }
        if self.entities.contains_key(name.as_ref()) {
            return self.internal_entity(&name);
        }
        if self.external_entities.contains_key(name.as_ref()) {
            return self.external_entity(&name);
        }
        Err(malformed(format!(
            "the general entity `{name}` is not declared; this crate expands character \
             references, the five predefined entities, and the entities declared in the \
             document's own internal subset"
        )))
    }

    /// Includes the internal general entity `name`, which the document declared.
    fn internal_entity(&mut self, name: &str) -> Result<(), XdmError> {
        match self.expander.expand(name, &self.entities) {
            Ok(text) => {
                self.push_text(&text, true);
                Ok(())
            }
            // The replacement text holds a `<`, so it is markup rather than character data.
            // XML 1.0 §4.3.2 allows exactly that as long as it matches the `content`
            // production, so it is parsed instead of being included as text. What the
            // abandoned attempt already charged against the expansion budget is not refunded:
            // the two routes bound the same document, and a reference that costs a little more
            // than it had to can only make the parser stricter.
            Err(ExpandError::Markup { .. }) => {
                let Some(text) = self.entities.get(name).cloned() else {
                    // `expand` reports `Markup` only for an entity it found in the map.
                    return Err(malformed(
                        ExpandError::Undeclared {
                            name: name.to_owned(),
                        }
                        .to_string(),
                    ));
                };
                // An internal entity's replacement text is part of the entity the reference
                // stands in, so it does not move the base URI (XML Base §4.2).
                self.splice(name, &text, None)
            }
            Err(error) => Err(malformed(error.to_string())),
        }
    }

    /// Retrieves the external parsed general entity `name` through the caller's resolver and
    /// splices its content into the tree in place of the reference.
    fn external_entity(&mut self, name: &str) -> Result<(), XdmError> {
        let Some(declaration) = self.external_entities.get(name) else {
            return Err(malformed(
                ExpandError::Undeclared {
                    name: name.to_owned(),
                }
                .to_string(),
            ));
        };
        // The request outlives the borrow of the declaration map, which the splice below needs
        // back.
        let public_id = declaration.public_id.clone();
        let system_id = declaration.system_id.clone();
        let request = ExternalEntity {
            name,
            public_id: public_id.as_deref(),
            system_id: &system_id,
        };
        let retrieved = match self.resolver.as_mut() {
            Some(resolver) => resolver.resolve(&request).map_err(|error| {
                malformed(format!(
                    "the external general entity `&{name};` could not be retrieved: {error}"
                ))
            })?,
            None => None,
        };
        match retrieved {
            Some(text) => {
                check_characters(
                    &text,
                    self.xml_version,
                    "the replacement text of an external general entity",
                )?;
                let entity_base = self.entity_base_uri(&system_id);
                self.splice(name, &text, entity_base)
            }
            None => Err(malformed(format!(
                "the external general entity `&{name};`, declared with the system identifier \
                 {system_id:?}, has no replacement text: an external entity is retrieved only \
                 through the resolver `Tree::parse_str_with_resolver` takes"
            ))),
        }
    }

    /// The absolute URI an external parsed general entity was retrieved from, which becomes the
    /// base URI of the content it contributes.
    ///
    /// [XML 1.0 §4.2.2](https://www.w3.org/TR/xml/#sec-external-ent) resolves the system
    /// identifier against the base URI of the entity the *declaration* appears in, which for a
    /// declaration in the internal subset is the document itself. `None` when the identifier is
    /// relative and the document has no URI to resolve it against: there is then no absolute URI
    /// to give the content, and it keeps the base URI of the reference point, as it did before
    /// any of this was recorded.
    fn entity_base_uri(&self, system_id: &str) -> Option<String> {
        resolve_system_id(self.options.document_uri.as_deref(), system_id)
    }

    /// Parses `text` — the replacement text of the general entity `name` — as `content` and
    /// appends the nodes it yields at the current position.
    ///
    /// `entity_base` is the absolute URI the text was retrieved from, for an external entity
    /// whose URI is known; `None` leaves the base URI in force unchanged, which is what an
    /// internal entity — replacement text that is part of the surrounding entity — needs.
    ///
    /// Element and attribute names inside it are resolved in the namespace scope of the
    /// *reference*, as [Namespaces in XML](https://www.w3.org/TR/xml-names/) requires, which
    /// costs nothing here: the scope stack the surrounding document is being read with is the
    /// one the sub-parse resolves against.
    ///
    /// The version in force is the *document's* throughout. An external entity may carry a
    /// `TextDecl` of its own naming a different one, but the entity's characters become
    /// characters of the document that referenced it, and a tree cannot be half 1.0 and half
    /// 1.1 — so the declaration is removed with the rest of the `TextDecl` and the document's
    /// version governs which characters are admitted and how line endings are folded.
    fn splice(
        &mut self,
        name: &str,
        text: &str,
        entity_base: Option<String>,
    ) -> Result<(), XdmError> {
        if self.open_entities.iter().any(|open| open == name) {
            return Err(malformed(format!(
                "the general entity `&{name};` refers to itself, which XML 1.0 forbids"
            )));
        }
        if self.open_entities.len() >= MAX_ENTITY_DEPTH {
            return Err(malformed(format!(
                "parsing the replacement text of the general entity `&{name};` would exceed the \
                 maximum entity nesting depth of {MAX_ENTITY_DEPTH}"
            )));
        }
        self.expander
            .charge_spliced_entity(name, text.len())
            .map_err(|error| malformed(error.to_string()))?;

        // A byte-order mark is an encoding signature rather than content
        // ([XML 1.0 §4.3.3](https://www.w3.org/TR/xml/#charencoding)), and the `TextDecl` an
        // external entity may begin with declares that entity's own version and encoding rather
        // than standing for anything in the document.
        let content = strip_text_declaration(text.strip_prefix('\u{FEFF}').unwrap_or(text));
        let source = format!("<{ENTITY_WRAPPER}>{content}</{ENTITY_WRAPPER}>");
        self.open_entities.push(name.to_owned());
        // The entity that referenced this one is what the base URI goes back to afterwards,
        // which nesting makes observable: an entity referenced from inside another based the
        // content it contributes on itself, not on the one it interrupted.
        let outer_base = self.builder.entity_base_uri().map(str::to_owned);
        if entity_base.is_some() {
            self.builder.set_entity_base_uri(entity_base);
        }
        let result = self.splice_content(&source);
        self.builder.set_entity_base_uri(outer_base);
        self.open_entities.pop();
        result
    }

    /// Parses `source` — an entity's content inside the wrapper [`splice`](Self::splice) puts
    /// around it — and appends everything inside the wrapper to the tree.
    ///
    /// The reader runs without the full well-formedness checker, which a fragment cannot pass:
    /// the checker applies WFC: Entity Declared to every reference it sees, and the references
    /// an entity's replacement text holds were declared by the *document*, whose type
    /// declaration the fragment does not repeat. Everything that constraint aside is still a
    /// constraint on a fragment — balanced markup, a matching end tag for every start tag, no
    /// end tag for anything the fragment did not open, a comment without `--` in it — the reader
    /// enforces regardless, and the wrapper is what makes "balanced" mean the `content`
    /// production rather than "one element". Character legality was checked where the text came
    /// from: an internal entity's value is part of the document the outer reader checked, and a
    /// retrieved one is checked below.
    fn splice_content(&mut self, source: &str) -> Result<(), XdmError> {
        /// Reported for entity text that closes the wrapper put around it, which no `content`
        /// production can do.
        const UNBALANCED: &str = "the replacement text of a general entity is not balanced content: it closes more \
             elements than it opens";

        let mut reader = Reader::from_str(source);
        reader.config_mut().expand_empty_elements = false;
        reader.config_mut().check_comments = true;
        // `0` while the wrapper has not been opened (or after it has closed), `1` inside it,
        // deeper for each element of the entity's own content.
        let mut depth = 0_usize;
        let mut closed = false;
        loop {
            let event = reader
                .read_event()
                .map_err(|error| malformed(error.to_string()))?;
            if closed && !matches!(event, Event::Eof) {
                return Err(malformed(UNBALANCED));
            }
            match event {
                Event::Eof => break,
                // The wrapper itself is not content: only what it contains is.
                Event::Start(_) if depth == 0 => depth = 1,
                Event::End(_) if depth == 1 => {
                    depth = 0;
                    closed = true;
                }
                Event::Start(_) => {
                    depth += 1;
                    self.handle_event(event)?;
                }
                Event::End(_) => {
                    depth = depth.checked_sub(1).ok_or_else(|| malformed(UNBALANCED))?;
                    self.handle_event(event)?;
                }
                // Neither an XML declaration nor a document type declaration is `content`. The
                // one that may legitimately open an external entity — its `TextDecl` — was
                // removed before parsing, so anything reaching here is misplaced, and letting it
                // through would let an entity redeclare the document's own version or entities.
                Event::Decl(_) | Event::DocType(_) => {
                    return Err(malformed(
                        "the replacement text of a general entity holds a declaration, which is \
                         not part of the `content` production",
                    ));
                }
                other => {
                    self.handle_event(other)?;
                }
            }
        }
        if !closed {
            return Err(malformed(
                "the replacement text of a general entity ends while an element is still open",
            ));
        }
        Ok(())
    }

    /// Normalizes an attribute value, resolving the general entities declared by the document's
    /// internal subset as well as the predefined ones.
    ///
    /// XML 1.0 §4.4 has an internal general entity *included in literal* inside an attribute
    /// value, exactly as it is included in content, so an attribute that uses one must not be
    /// rejected as an unknown escape.
    fn normalized_value<'v>(
        &mut self,
        attribute: &Attribute<'v>,
    ) -> Result<Cow<'v, str>, XdmError> {
        let entities = &self.entities;
        let expander = &mut self.expander;
        // The budget is charged from inside the resolver because
        // `normalized_value_with` walks the nested references itself: it is the only point at
        // which each occurrence — including the ones it reaches recursively — is visible.
        // Refusing to resolve is how a spent budget is signalled.
        let mut over_budget: Option<String> = None;
        let value = attribute.normalized_value_with(self.xml_version, MAX_ENTITY_DEPTH, |name| {
            if let Some(text) = resolve_xml_entity(name) {
                return Some(text);
            }
            let text = entities.get(name)?;
            if over_budget.is_some() || !expander.charge_attribute_expansion(name, text.len()) {
                over_budget.get_or_insert_with(|| name.to_owned());
                return None;
            }
            Some(text.as_str())
        });
        match (value, over_budget) {
            (Ok(value), _) => Ok(value),
            (Err(_), Some(entity)) => Err(malformed(format!(
                "expanding the general entity `{entity}` would exceed the limits on entity \
                 expansion"
            ))),
            (Err(error), None) => Err(malformed(error.to_string())),
        }
    }

    /// Expands a lexical name, *retaining* the prefix it was written with.
    ///
    /// XDM 3.1 §5.4 keeps the prefix as a third component of a node's name so that
    /// `fn:node-name` and `fn:prefix-from-QName` can report the spelling the document used. The
    /// prefix takes no part in equality, so name tests and comparisons are unaffected.
    fn resolve(&self, name: &[u8], context: NameContext) -> Result<ExpandedName, XdmError> {
        let name = std::str::from_utf8(name).map_err(|error| malformed(error.to_string()))?;
        let expanded = self
            .namespaces
            .resolve_qname(name, context)
            .map_err(|error| malformed(error.to_string()))?;
        Ok(match name.split_once(':') {
            Some((prefix, _)) => expanded.with_prefix(prefix),
            None => expanded,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{NodeKind, XdmNode};

    #[test]
    fn namespaces_are_resolved_for_elements_and_attributes() -> Result<(), XdmError> {
        let tree = Tree::parse_str(
            r#"<root xmlns="urn:default" xmlns:p="urn:p" p:a="1" b="2"><p:child/></root>"#,
        )?;
        let root = tree.document_element().expect("a document element");
        assert_eq!(
            root.node_name().map(|name| name.to_string()).as_deref(),
            Some("{urn:default}root")
        );

        let attribute_names: Vec<_> = root
            .attributes()
            .iter()
            .filter_map(XdmNode::node_name)
            .map(|name| name.to_string())
            .collect();
        assert_eq!(attribute_names, ["{urn:p}a", "b"]);

        let child = root.children().first().cloned().expect("one child");
        assert_eq!(
            child.node_name().map(|name| name.to_string()).as_deref(),
            Some("{urn:p}child")
        );
        Ok(())
    }

    #[test]
    fn namespace_nodes_are_inherited() -> Result<(), XdmError> {
        let tree = Tree::parse_str(r#"<a xmlns:p="urn:p"><b xmlns:q="urn:q"><c/></b></a>"#)?;
        let a = tree.document_element().expect("a document element");
        let b = a.children().first().cloned().expect("one child");
        let c = b.children().first().cloned().expect("one grandchild");

        let prefixes = |node: &crate::node::NodeRef| -> Vec<String> {
            node.namespaces()
                .iter()
                .filter_map(XdmNode::node_name)
                .map(|name| name.local_name().to_owned())
                .collect()
        };
        assert_eq!(prefixes(&a), ["p", "xml"]);
        assert_eq!(prefixes(&b), ["p", "q", "xml"]);
        assert_eq!(prefixes(&c), ["p", "q", "xml"]);
        Ok(())
    }

    #[test]
    fn entities_and_cdata_merge_into_one_text_node() -> Result<(), XdmError> {
        let tree = Tree::parse_str("<a>x &amp; y<![CDATA[ & z]]>&#65;</a>")?;
        let a = tree.document_element().expect("a document element");
        assert_eq!(a.children().len(), 1, "no adjacent text nodes");
        assert_eq!(a.string_value(), "x & y & zA");
        assert_eq!(a.children()[0].node_kind(), NodeKind::Text);
        Ok(())
    }

    #[test]
    fn undeclared_entities_and_prefixes_are_errors() {
        assert!(Tree::parse_str("<a>&nope;</a>").is_err());
        assert!(Tree::parse_str("<p:a/>").is_err());
        assert!(Tree::parse_str("<a></b>").is_err());
        assert!(Tree::parse_str("<a/><b/>").is_err());
    }

    /// Documents whose internal subset declares an entity whose value itself
    /// holds references: `(name, internal subset, the reference as written, the
    /// character data it stands for)`.
    ///
    /// XML 1.0 §4.4 gives this a two-phase answer: a character reference inside
    /// an `EntityValue` is resolved where the entity is *declared*, a general
    /// entity reference where it is *used*, and neither pass rescans what it
    /// produced. The same answer is required in content and in an attribute
    /// value, and `oxixml-dom` and `oxixml-sax` are held to this very table, so
    /// a document means the same thing wherever the reference appears and
    /// whichever of the three crates reads it.
    const ENTITY_EXPANSIONS: &[(&str, &str, &str, &str)] = &[
        (
            "predefined-inside-the-value",
            r#"<!ENTITY e "bar &amp; baz">"#,
            "&e;",
            "bar & baz",
        ),
        (
            "character-reference-inside-the-value",
            r#"<!ENTITY copy "&#169;">"#,
            "&copy;",
            "\u{a9}",
        ),
        (
            "character-reference-in-running-text",
            r#"<!ENTITY nbsp "&#160;">"#,
            "x&nbsp;y",
            "x\u{a0}y",
        ),
        (
            "hexadecimal-character-reference",
            r#"<!ENTITY e "&#xe9;t&#xe9;">"#,
            "&e;",
            "\u{e9}t\u{e9}",
        ),
        (
            "nested-general-entity",
            r#"<!ENTITY inner "in"><!ENTITY outer "[&inner;]">"#,
            "&outer;",
            "[in]",
        ),
        (
            "escaped-markup-stays-text",
            r#"<!ENTITY tag "&lt;foo&gt;">"#,
            "&tag;",
            "<foo>",
        ),
        (
            // A `>` needs no escape anywhere, so the scanner that finds the end of the
            // declaration has to look past the one inside the quotes.
            "unescaped-greater-than-in-the-value",
            r#"<!ENTITY e "a > b">"#,
            "&e;",
            "a > b",
        ),
        (
            // A literal delimited by apostrophes may hold quotation marks, and a quotation mark
            // that reaches an attribute value through an entity is data rather than the end of
            // the literal (XML 1.0 §4.4, "Included in Literal").
            "quotation-marks-in-an-apostrophe-quoted-value",
            r#"<!ENTITY e 'say "hi"'>"#,
            "&e;",
            "say \"hi\"",
        ),
        (
            "doubly-escaped-less-than",
            r#"<!ENTITY e "&#38;#60;">"#,
            "&e;",
            "<",
        ),
        (
            "doubly-escaped-ampersand",
            r#"<!ENTITY e "&#38;#38;">"#,
            "&e;",
            "&",
        ),
        (
            "triply-escaped-ampersand",
            r#"<!ENTITY e "&#38;#38;#38;">"#,
            "&e;",
            "&#38;",
        ),
        (
            "the-specifications-own-example",
            r#"<!ENTITY rights "All rights reserved"><!ENTITY book "La Peste: &#38;#38; 1947, &#38;rights;">"#,
            "&book;",
            "La Peste: & 1947, All rights reserved",
        ),
    ];

    /// The document that puts `reference` in content.
    fn in_content(subset: &str, reference: &str) -> String {
        format!("<!DOCTYPE r [{subset}]><r>{reference}</r>")
    }

    /// The document that puts `reference` in an attribute value.
    fn in_attribute(subset: &str, reference: &str) -> String {
        format!("<!DOCTYPE r [{subset}]><r a=\"{reference}\"/>")
    }

    #[test]
    fn declared_entities_expand_to_the_agreed_character_data() -> Result<(), XdmError> {
        for &(name, subset, reference, expected) in ENTITY_EXPANSIONS {
            let tree = Tree::parse_str(&in_content(subset, reference))?;
            let root = tree.document_element().expect("a document element");
            assert_eq!(root.string_value(), expected, "{name}");
        }
        Ok(())
    }

    #[test]
    fn an_attribute_value_expands_to_the_same_character_data_as_content() -> Result<(), XdmError> {
        for &(name, subset, reference, expected) in ENTITY_EXPANSIONS {
            let tree = Tree::parse_str(&in_attribute(subset, reference))?;
            let root = tree.document_element().expect("a document element");
            let attribute = root.attributes().first().cloned().expect("one attribute");
            assert_eq!(attribute.string_value(), expected, "{name}");
        }
        Ok(())
    }

    #[test]
    fn entities_without_usable_replacement_text_are_rejected() {
        for xml in [
            // External: nothing retrieves it unless the caller opts in.
            r#"<!DOCTYPE r [<!ENTITY e SYSTEM "e.xml">]><r>&e;</r>"#,
            r#"<!DOCTYPE r [<!ENTITY e PUBLIC "-//x//TEXT//EN" "e.xml">]><r>&e;</r>"#,
            // Declared in an external subset this crate never reads.
            r#"<!DOCTYPE r SYSTEM "r.dtd"><r>&e;</r>"#,
        ] {
            assert!(Tree::parse_str(xml).is_err(), "{xml}");
        }
    }

    /// Internal general entities whose replacement text is markup: `(name, internal subset, the
    /// reference in context, the string value of the document element, the kinds of its
    /// children)`.
    ///
    /// [XML 1.0 §4.3.2](https://www.w3.org/TR/xml/#wf-entities) makes an internal general entity
    /// well-formed when its replacement text matches the `content` production, which is what
    /// every row here does. The reference is then replaced by the *nodes* the replacement text
    /// parses to, in the namespace scope of the reference.
    const MARKUP_ENTITIES: &[(&str, &str, &str, &str, &[NodeKind])] = &[
        (
            "one-element",
            r#"<!ENTITY e "<b>x</b>">"#,
            "&e;",
            "x",
            &[NodeKind::Element],
        ),
        (
            // A `<` that reaches the replacement text through a *single* escape is markup,
            // which is exactly why XML 1.0 §4.6 requires `lt` to be declared doubly escaped.
            "singly-escaped-less-than",
            r#"<!ENTITY e "&#60;b/>">"#,
            "&e;",
            "",
            &[NodeKind::Element],
        ),
        (
            "several-top-level-nodes",
            r#"<!ENTITY e "<b/>text<c/>">"#,
            "&e;",
            "text",
            &[NodeKind::Element, NodeKind::Text, NodeKind::Element],
        ),
        (
            "text-on-both-sides-of-the-reference",
            r#"<!ENTITY e "in<b/>out">"#,
            "before&e;after",
            "beforeinoutafter",
            &[NodeKind::Text, NodeKind::Element, NodeKind::Text],
        ),
        (
            // The reference is expanded where it is *used*, so the nested one is too.
            "markup-nested-in-markup",
            r#"<!ENTITY inner "<i/>"><!ENTITY outer "<b>&inner;</b>">"#,
            "&outer;",
            "",
            &[NodeKind::Element],
        ),
        (
            // The outer entity's own replacement text is character data; the markup arrives
            // through the entity it names, and the whole thing is still `content`.
            "markup-reached-through-a-text-entity",
            r#"<!ENTITY inner "<i>x</i>"><!ENTITY outer "[&inner;]">"#,
            "&outer;",
            "[x]",
            &[NodeKind::Text, NodeKind::Element, NodeKind::Text],
        ),
        (
            "comments-and-processing-instructions-are-content",
            r#"<!ENTITY e "<!--c--><?t d?>">"#,
            "&e;",
            "",
            &[NodeKind::Comment, NodeKind::ProcessingInstruction],
        ),
        (
            "a-cdata-section-is-content",
            r#"<!ENTITY e "<![CDATA[<not markup>]]>">"#,
            "&e;",
            "<not markup>",
            &[NodeKind::Text],
        ),
    ];

    #[test]
    fn an_internal_entity_may_stand_for_markup() -> Result<(), XdmError> {
        for &(name, subset, reference, expected, kinds) in MARKUP_ENTITIES {
            let tree = Tree::parse_str(&in_content(subset, reference))?;
            let root = tree.document_element().expect("a document element");
            assert_eq!(root.string_value(), expected, "{name}");
            let actual: Vec<NodeKind> = root.children().iter().map(XdmNode::node_kind).collect();
            assert_eq!(actual, kinds, "{name}");
        }
        Ok(())
    }

    #[test]
    fn a_spliced_element_keeps_the_namespaces_in_force_at_the_reference() -> Result<(), XdmError> {
        let tree = Tree::parse_str(
            r#"<!DOCTYPE r [<!ENTITY e "<p:b><c/></p:b>">]>
               <r xmlns:p="urn:p" xmlns="urn:d">&e;</r>"#,
        )?;
        let root = tree.document_element().expect("a document element");
        let b = root
            .children()
            .into_iter()
            .find(|child| child.node_kind() == NodeKind::Element)
            .expect("the spliced element");
        assert_eq!(
            b.node_name().map(|name| name.to_string()).as_deref(),
            Some("{urn:p}b")
        );
        let c = b.children().first().cloned().expect("its child");
        assert_eq!(
            c.node_name().map(|name| name.to_string()).as_deref(),
            Some("{urn:d}c")
        );
        // A prefix the entity uses but nothing binds is as unbound there as anywhere else.
        assert!(
            Tree::parse_str(r#"<!DOCTYPE r [<!ENTITY e "<q:b/>">]><r>&e;</r>"#).is_err(),
            "an unbound prefix inside an entity is still unbound"
        );
        Ok(())
    }

    #[test]
    fn a_spliced_entity_may_declare_its_own_namespaces() -> Result<(), XdmError> {
        let tree = Tree::parse_str(
            r#"<!DOCTYPE r [<!ENTITY e "<b xmlns='urn:e'><c/></b>">]><r><a/>&e;</r>"#,
        )?;
        let root = tree.document_element().expect("a document element");
        let b = root
            .children()
            .get(1)
            .cloned()
            .expect("the spliced element");
        assert_eq!(
            b.node_name().map(|name| name.to_string()).as_deref(),
            Some("{urn:e}b")
        );
        // The declaration goes out of scope with the element that carries it.
        let a = root.children().first().cloned().expect("the first child");
        assert!(a.namespaces().iter().all(|node| {
            node.node_name()
                .is_none_or(|name| name.local_name() == "xml")
        }));
        Ok(())
    }

    #[test]
    fn replacement_text_that_is_not_content_is_rejected() {
        // For an *internal* entity the well-formedness checker of `oxixml_xml` gets there first:
        // the declaration is part of the document it is checking, so it validates the
        // replacement text against `content` before a reference to it is ever expanded. The
        // same six shapes reaching the tree through a resolver instead are caught by the
        // sub-parse (`tests/external_entities.rs`), which is the only line of defence there.
        for (name, subset) in [
            ("an unclosed element", r#"<!ENTITY e "<b>">"#),
            ("a stray end tag", r#"<!ENTITY e "</b>">"#),
            ("a mismatched end tag", r#"<!ENTITY e "<b></c>">"#),
            ("more closed than opened", r#"<!ENTITY e "<b/></r>">"#),
            ("a bare less-than sign", r#"<!ENTITY e "a &#60; b">"#),
            (
                "a comment holding a double hyphen",
                r#"<!ENTITY e "<!--a--b-->">"#,
            ),
        ] {
            let xml = in_content(subset, "&e;");
            assert!(Tree::parse_str(&xml).is_err(), "{name}: {xml}");
        }
    }

    #[test]
    fn a_recursive_markup_entity_is_bounded_rather_than_followed() {
        for (name, subset) in [
            ("directly recursive", r#"<!ENTITY e "<b>&e;</b>">"#),
            (
                "mutually recursive",
                r#"<!ENTITY a "<x>&b;</x>"><!ENTITY b "<y>&a;</y>">"#,
            ),
            (
                // The cycle runs through an entity whose own replacement text is character data.
                "recursive through a text entity",
                r#"<!ENTITY a "<x>&b;</x>"><!ENTITY b "&a;">"#,
            ),
        ] {
            let reference = if subset.contains("<!ENTITY a ") {
                "&a;"
            } else {
                "&e;"
            };
            let xml = in_content(subset, reference);
            let error = Tree::parse_str(&xml).expect_err(name);
            assert_eq!(error.code(), ErrorCode::Fodc0006, "{name}");
        }
    }

    #[test]
    fn nesting_spliced_entities_is_bounded() {
        // Each level wraps the next in an element, so the recursion is entirely on the splice
        // path and only its own depth limit can stop it.
        let mut subset = String::new();
        for level in 0..MAX_ENTITY_DEPTH + 4 {
            subset.push_str(&format!(r#"<!ENTITY e{level} "<w>&e{};</w>">"#, level + 1));
        }
        subset.push_str(&format!(r#"<!ENTITY e{} "leaf">"#, MAX_ENTITY_DEPTH + 4));
        let error = Tree::parse_str(&in_content(&subset, "&e0;"))
            .expect_err("a chain deeper than the limit must be refused");
        assert_eq!(error.code(), ErrorCode::Fodc0006);
        assert!(
            error.to_string().contains("nesting depth"),
            "the message must name the limit: {error}"
        );

        // One level short of the limit still parses, so the limit is where it says it is.
        let mut shallow = String::new();
        for level in 0..MAX_ENTITY_DEPTH - 1 {
            shallow.push_str(&format!(r#"<!ENTITY e{level} "<w>&e{};</w>">"#, level + 1));
        }
        shallow.push_str(&format!(r#"<!ENTITY e{} "leaf">"#, MAX_ENTITY_DEPTH - 1));
        assert!(Tree::parse_str(&in_content(&shallow, "&e0;")).is_ok());
    }

    #[test]
    fn a_multiplying_markup_entity_is_refused() {
        // "Billion laughs" written with elements rather than characters: the budget has to bound
        // the splice path exactly as it bounds the text one.
        let mut subset = String::from(r#"<!ENTITY lol0 "<w/>">"#);
        for level in 1..10 {
            subset.push_str(&format!(
                r#"<!ENTITY lol{level} "{}">"#,
                format!("&lol{};", level - 1).repeat(10)
            ));
        }
        assert!(Tree::parse_str(&in_content(&subset, "&lol9;")).is_err());
    }

    #[test]
    fn an_entity_declares_nothing_for_the_document_around_it() {
        // A document type declaration is not `content`, so an entity cannot smuggle one in…
        assert!(
            Tree::parse_str(&in_content(
                r#"<!ENTITY e "<b/><!DOCTYPE q [<!ENTITY x 'no'>]>">"#,
                "&e;"
            ))
            .is_err()
        );
        // …and an entity the *document* never declared is undeclared wherever it is referenced.
        assert!(Tree::parse_str(&in_content(r#"<!ENTITY e "<b/>">"#, "&e;&x;")).is_err());
    }

    #[test]
    fn a_recursive_entity_is_rejected_instead_of_hanging() {
        for xml in [
            r#"<!DOCTYPE r [<!ENTITY e "&e;">]><r>&e;</r>"#,
            r#"<!DOCTYPE r [<!ENTITY a "&b;"><!ENTITY b "&a;">]><r>&a;</r>"#,
        ] {
            assert!(Tree::parse_str(xml).is_err(), "{xml}");
        }
    }

    #[test]
    fn a_reference_may_be_repeated_as_often_as_a_document_likes() -> Result<(), XdmError> {
        // Repeating a reference is not "amplification": what the entity stands for is written
        // out once per occurrence, so an ordinary page full of `&nbsp;` must parse like any
        // other. The ceiling here is the one the well-formedness checker of `oxixml-xml`
        // applies to any document it checks, which this parser always turns on.
        const OCCURRENCES: usize = 9_000;
        let mut xml = String::from(r#"<!DOCTYPE r [<!ENTITY nbsp "&#160;">]><r>"#);
        for _ in 0..OCCURRENCES {
            xml.push_str("&nbsp;");
        }
        xml.push_str("</r>");

        let tree = Tree::parse_str(&xml)?;
        let text = tree.document_element().expect("root").string_value();
        assert_eq!(text.chars().count(), OCCURRENCES);
        assert!(text.chars().all(|character| character == '\u{a0}'));
        Ok(())
    }

    #[test]
    fn a_billion_laughs_declaration_is_refused() {
        let mut xml = String::from(r#"<!DOCTYPE r [<!ENTITY lol0 "lololololololololol">"#);
        for level in 1..10 {
            xml.push_str(&format!(
                r#"<!ENTITY lol{level} "{}">"#,
                format!("&lol{};", level - 1).repeat(10)
            ));
        }
        xml.push_str("]><r>&lol9;</r>");
        assert!(Tree::parse_str(&xml).is_err());
    }

    #[test]
    fn a_billion_laughs_attribute_value_is_refused_like_one_in_content() {
        // The attribute path expands through the underlying reader rather than through this
        // crate's own expander, so it needs its own proof that the budget still bites: a *wide*
        // declaration multiplies enormously without ever nesting deeply enough to trip the depth
        // limit alone.
        let mut subset = String::from(r#"<!ENTITY lol0 "lololololololololol">"#);
        for level in 1..10 {
            subset.push_str(&format!(
                r#"<!ENTITY lol{level} "{}">"#,
                format!("&lol{};", level - 1).repeat(10)
            ));
        }
        for xml in [
            format!("<!DOCTYPE r [{subset}]><r>&lol9;</r>"),
            format!("<!DOCTYPE r [{subset}]><r a=\"&lol9;\"/>"),
        ] {
            assert!(Tree::parse_str(&xml).is_err(), "{xml}");
        }
    }

    #[test]
    fn the_first_declaration_of_an_entity_binds() -> Result<(), XdmError> {
        let tree =
            Tree::parse_str(r#"<!DOCTYPE r [<!ENTITY e "first"><!ENTITY e "second">]><r>&e;</r>"#)?;
        assert_eq!(
            tree.document_element().expect("root").string_value(),
            "first"
        );
        Ok(())
    }

    #[test]
    fn comments_and_processing_instructions_survive() -> Result<(), XdmError> {
        let tree = Tree::parse_str("<?xml version='1.0'?><!--top--><a><?target data?></a>")?;
        let root_children = tree.root().children();
        assert_eq!(root_children.len(), 2);
        assert_eq!(root_children[0].node_kind(), NodeKind::Comment);
        assert_eq!(root_children[0].string_value(), "top");

        let a = tree.document_element().expect("a document element");
        let instruction = a.children().first().cloned().expect("one child");
        assert_eq!(instruction.node_kind(), NodeKind::ProcessingInstruction);
        assert_eq!(
            instruction
                .node_name()
                .map(|name| name.local_name().to_owned())
                .as_deref(),
            Some("target")
        );
        assert_eq!(instruction.string_value(), "data");
        Ok(())
    }

    #[test]
    fn line_endings_are_normalized() -> Result<(), XdmError> {
        let tree = Tree::parse_str("<a>one\r\ntwo\rthree</a>")?;
        assert_eq!(
            tree.document_element().expect("root").string_value(),
            "one\ntwo\nthree"
        );
        assert_eq!(
            normalize_line_endings("no carriage returns", XmlVersion::Implicit1_0),
            "no carriage returns"
        );
        Ok(())
    }

    #[test]
    fn boundary_whitespace_stripping_is_opt_in() -> Result<(), XdmError> {
        let xml = "<a>\n  <b>text</b>\n  <c> </c>\n</a>";
        let kept = Tree::parse_str(xml)?;
        assert_eq!(kept.document_element().expect("root").children().len(), 5);

        let stripped =
            Tree::parse_str_with(xml, ParseOptions::default().strip_boundary_whitespace(true))?;
        let a = stripped.document_element().expect("root");
        assert_eq!(a.children().len(), 2);
        assert_eq!(a.string_value(), "text");
        Ok(())
    }

    /// `NameStartChar` grew in XML 1.0 *fifth edition* — `[#x37F-#x1FFF]` replaced the fourth
    /// edition's `Letter` production — so `Ϳ` (U+037F) and `ſ` (U+017F) begin a name here and
    /// did not before. The QT3 and XSTS drivers grade `xml-version` dependencies on that being
    /// true of this parser, so it is asserted rather than assumed.
    #[test]
    fn fifth_edition_name_start_characters_are_accepted() -> Result<(), XdmError> {
        for (xml, expected) in [
            ("<\u{37F}node/>", "\u{37F}node"),
            ("<egg\u{17F}/>", "egg\u{17F}"),
        ] {
            let tree = Tree::parse_str(xml)?;
            let root = tree.document_element().expect("a document element");
            assert_eq!(
                root.node_name().map(|name| name.local_name().to_owned()),
                Some(expected.to_owned())
            );
        }
        // A digit still begins no name, in either edition.
        assert!(Tree::parse_str("<0bad/>").is_err());
        Ok(())
    }

    #[test]
    fn document_uri_seeds_the_base_uri() -> Result<(), XdmError> {
        let tree = Tree::parse_str_with(
            "<a/>",
            ParseOptions::default().document_uri("http://example.com/d.xml"),
        )?;
        assert_eq!(tree.document_uri(), Some("http://example.com/d.xml"));
        assert_eq!(
            tree.document_element().expect("root").base_uri().as_deref(),
            Some("http://example.com/d.xml")
        );
        assert_eq!(
            tree.root().document_uri().as_deref(),
            Some("http://example.com/d.xml")
        );
        Ok(())
    }
}
