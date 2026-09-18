//! [`fn:parse-xml`](https://www.w3.org/TR/xpath-functions-31/#func-parse-xml) and
//! [`fn:parse-xml-fragment`](https://www.w3.org/TR/xpath-functions-31/#func-parse-xml-fragment).
//!
//! `fn:parse-xml` is [`oxixml_infoset::Tree::parse_str_with`] behind a small [`XdmStore`],
//! `ParsedDocument`, that hides the tree's *document* URI: the result carries the static base URI
//! of the call but was never retrieved from a URI of its own. The fragment form needs more: an
//! *external general parsed entity* may contain several top-level elements, text, comments and
//! processing instructions, which is a shape the in-memory `Tree` builder rejects because it
//! enforces the one-document-element rule. The fragment is therefore parsed inside a synthetic
//! wrapper element and then *re-rooted* onto a document node by `FragmentStore`, a store of its
//! own.
//!
//! # External entities
//!
//! [F&O 3.1 §14.10.1](https://www.w3.org/TR/xpath-functions-31/#func-parse-xml) leaves it
//! implementation-defined whether `fn:parse-xml` retrieves the external entities the text it is
//! given declares. It does, but only where retrieval cannot reach anything the *caller* could not
//! already read: the entity is looked for below the directory of the **static base URI**, and
//! only when that base URI is a `file:` URI. A call with no static base URI, or with an
//! `http:` one, retrieves nothing at all — an external entity reference then stays the
//! `FODC0006` it has always been. Within that directory the retrieval is
//! [`oxixml_infoset::FileEntityResolver`]'s, which refuses an absolute path, a `..` that climbs
//! out, a host-bearing `file://` authority, every other scheme, and anything past
//! [`oxixml_infoset::DEFAULT_MAX_ENTITY_BYTES`].

use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_infoset::{
    AtomicValue, DocumentOrderKey, ExpandedName, FileEntityResolver, NodeId, NodeKind, NodeRef,
    ParseOptions, Sequence, Tree, XdmError, XdmNode, XdmStore, next_tree_id,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// [`fn:parse-xml`](https://www.w3.org/TR/xpath-functions-31/#func-parse-xml): parses a string
/// as a well-formed XML document and returns its document node.
///
/// `base_uri` is the static base URI of the calling expression, which
/// [F&O 3.1 §14.7.1](https://www.w3.org/TR/xpath-functions-31/#func-parse-xml) makes the *base*
/// URI of the result — so that a relative URI written inside the parsed text resolves the way one
/// written in the calling expression would. The result's `document-uri` property stays **absent**:
/// the document was constructed from a string rather than retrieved from a URI.
///
/// An external entity the text declares is retrieved from the directory of the static base URI,
/// and only from there — see the [module documentation](self#external-entities).
///
/// # Errors
///
/// [`FnErrorCode::Fodc0006`] when the string is not a well-formed document, an external entity
/// it declares cannot be retrieved included, and [`FnErrorCode::Fodc0007`] when `base_uri` is not
/// an absolute URI.
///
/// # Examples
///
/// ```
/// use oxixml_fn::xmlio::parse_xml;
/// use oxixml_infoset::{NodeKind, XdmNode};
///
/// let document = parse_xml("<a><b>text</b></a>", None)?;
/// assert_eq!(document.node_kind(), NodeKind::Document);
/// assert_eq!(document.string_value(), "text");
/// assert!(parse_xml("<a>", None).is_err());
///
/// // The base URI is the caller's; the document URI is absent.
/// let document = parse_xml("<a/>", Some("http://example.com/query.xq"))?;
/// assert_eq!(document.base_uri().as_deref(), Some("http://example.com/query.xq"));
/// assert_eq!(document.document_uri(), None);
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn parse_xml(xml: &str, base_uri: Option<&str>) -> FnResult<NodeRef> {
    let mut options = ParseOptions::default();
    let mut entity_directory = None;
    if let Some(base_uri) = base_uri {
        if !crate::strings::uri::is_absolute_uri(base_uri) {
            return Err(FnError::new(
                FnErrorCode::Fodc0007,
                format!("`{base_uri}` is not a valid absolute URI"),
            ));
        }
        // The tree builder seeds the base URI of every node from the document URI, which is the
        // only way to give the whole tree a base URI to resolve `xml:base` against; the store
        // built around it then hides the document URI again.
        options = options.document_uri(base_uri);
        entity_directory = entity_directory_of(base_uri);
    }
    let parsed = match entity_directory {
        Some(directory) => {
            let mut resolver = FileEntityResolver::new(directory);
            Tree::parse_str_with_resolver(xml, options, &mut resolver)
        }
        None => Tree::parse_str_with(xml, options),
    };
    let tree = parsed.map_err(|error| {
        FnError::new(
            FnErrorCode::Fodc0006,
            format!("fn:parse-xml: {}", error.message()),
        )
    })?;
    Ok(ParsedDocument::wrap(&tree))
}

/// The directory that external entities declared inside the parsed text are looked for in: the
/// one holding the document the static base URI names.
///
/// `None` — meaning nothing is retrieved — for every base URI that does not name a file on this
/// machine: another scheme, a `file://host/…` authority naming another machine, or a URI whose
/// path has no parent directory at all.
fn entity_directory_of(base_uri: &str) -> Option<PathBuf> {
    let (scheme, rest) = base_uri.split_once(':')?;
    if !scheme.eq_ignore_ascii_case("file") {
        return None;
    }
    // A query or fragment is no part of the path.
    let path = &rest[..rest.find(['?', '#']).unwrap_or(rest.len())];
    let path = match path.strip_prefix("//") {
        // `file:///path` has an empty authority and means "this machine"; `file://host/path`
        // names another one and is not read.
        Some(authority_and_path) => {
            if !authority_and_path.starts_with('/') {
                return None;
            }
            authority_and_path
        }
        None => path,
    };
    let decoded = percent_decode(path);
    // `file:///C:/dir/doc.xml` — the drive letter is preceded by the URI path's own separator,
    // which is not part of the file-system path it denotes.
    let decoded = match decoded.strip_prefix('/') {
        Some(rest) if has_drive_prefix(rest) => rest.to_owned(),
        _ => decoded,
    };
    Path::new(&decoded).parent().map(Path::to_path_buf)
}

/// Whether `path` opens with a `C:` drive specification.
fn has_drive_prefix(path: &str) -> bool {
    let mut characters = path.chars();
    characters.next().is_some_and(|first| first.is_alphabetic())
        && characters.next() == Some(':')
        && matches!(characters.next(), Some('/') | Some('\\') | None)
}

/// Decodes the `%XX` escapes of a URI reference. A malformed escape is left alone rather than
/// being an error: it is then just a character of the path.
fn percent_decode(text: &str) -> String {
    if !text.contains('%') {
        return text.to_owned();
    }
    let bytes = text.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let high = char::from(bytes[index + 1]).to_digit(16);
            let low = char::from(bytes[index + 2]).to_digit(16);
            if let (Some(high), Some(low)) = (high, low) {
                if let Ok(byte) = u8::try_from(high * 16 + low) {
                    decoded.push(byte);
                    index += 3;
                    continue;
                }
            }
        }
        decoded.push(bytes[index]);
        index += 1;
    }
    String::from_utf8(decoded).unwrap_or_else(|_| text.to_owned())
}

/// [`fn:parse-xml-fragment`](https://www.w3.org/TR/xpath-functions-31/#func-parse-xml-fragment):
/// parses a string as an external general parsed entity and returns a document node whose
/// children are the fragment's top-level nodes.
///
/// # Errors
///
/// [`FnErrorCode::Fodc0006`] when the fragment is not well-balanced, and
/// [`FnErrorCode::Fodc0007`] when `base_uri` is not an absolute URI.
///
/// # Examples
///
/// ```
/// use oxixml_fn::xmlio::parse_xml_fragment;
/// use oxixml_infoset::{NodeKind, XdmNode};
///
/// // Several top-level elements, which a document may not have but a fragment may.
/// let document = parse_xml_fragment("<a/>text<b/>", None)?;
/// assert_eq!(document.node_kind(), NodeKind::Document);
/// assert_eq!(document.children().len(), 3);
/// assert_eq!(document.string_value(), "text");
///
/// // A fragment may also be empty.
/// assert!(parse_xml_fragment("", None)?.children().is_empty());
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn parse_xml_fragment(fragment: &str, base_uri: Option<&str>) -> FnResult<NodeRef> {
    if let Some(base_uri) = base_uri {
        if !crate::strings::uri::is_absolute_uri(base_uri) {
            return Err(FnError::new(
                FnErrorCode::Fodc0007,
                format!("`{base_uri}` is not a valid absolute URI"),
            ));
        }
    }
    let wrapped = format!(
        "<{WRAPPER}>{}</{WRAPPER}>",
        without_text_declaration(fragment)?
    );
    let mut options = ParseOptions::default();
    if let Some(base_uri) = base_uri {
        options = options.document_uri(base_uri);
    }
    let tree = Tree::parse_str_with(&wrapped, options).map_err(|error| {
        FnError::new(
            FnErrorCode::Fodc0006,
            format!("fn:parse-xml-fragment: {}", error.message()),
        )
    })?;
    let wrapper = tree.document_element().ok_or_else(|| {
        FnError::new(
            FnErrorCode::Fodc0006,
            "fn:parse-xml-fragment: the fragment did not parse",
        )
    })?;
    Ok(FragmentStore::reroot(&wrapper, base_uri))
}

/// Strips a leading text declaration from a fragment, rejecting a malformed one.
///
/// F&O 3.1 §14.7.2 parses the argument as an *external general parsed entity*, whose production
/// is `extParsedEnt ::= TextDecl? content` — so a leading `<?xml … ?>` is allowed, but only in
/// the shape [XML §4.3.1](https://www.w3.org/TR/xml/#sec-TextDecl) gives it:
///
/// ```text
/// TextDecl ::= '<?xml' VersionInfo? EncodingDecl S? '?>'
/// ```
///
/// The encoding is *required* and `standalone` is not permitted, which is why
/// `<?xml version='1.0'?><a/>` and `<?xml version='1.0' encoding='utf-8' standalone='yes'?><a/>`
/// are both `FODC0006` (QT3 `parse-xml-fragment-016`, `-017`) while
/// `<?xml version='1.0' encoding='utf-8'?>abc` is the single text node `abc` (`-014`, `-015`).
///
/// Removing the declaration is necessary because the fragment is parsed inside a wrapper
/// element, where XML would no longer accept it at all.
fn without_text_declaration(fragment: &str) -> FnResult<&str> {
    let Some(rest) = fragment.strip_prefix("<?xml") else {
        return Ok(fragment);
    };
    // `<?xmlfoo?>` is an ordinary processing instruction: a name character may not follow.
    if !rest.starts_with([' ', '\t', '\r', '\n']) {
        return Ok(fragment);
    }
    let malformed = || {
        FnError::new(
            FnErrorCode::Fodc0006,
            "fn:parse-xml-fragment: the text declaration must be \
             `<?xml version=\"…\"? encoding=\"…\" ?>`",
        )
    };
    let end = rest.find("?>").ok_or_else(malformed)?;
    // The name of each pseudo-attribute is the last word before its `=`, which tolerates the
    // whitespace XML allows on either side of the equals sign — the four characters of the `S`
    // production and no others, so a name run together with the one before it by any other
    // character is not the name the declaration needs.
    let body = &rest[..end];
    let segments: Vec<&str> = body.split('=').collect();
    let mut pseudo_attributes =
        segments[..segments.len().saturating_sub(1)]
            .iter()
            .map(|segment| {
                segment
                    .split([' ', '\t', '\r', '\n'])
                    .rfind(|word| !word.is_empty())
                    .unwrap_or_default()
            });
    let mut next = pseudo_attributes.next();
    if next == Some("version") {
        next = pseudo_attributes.next();
    }
    if next != Some("encoding") || pseudo_attributes.next().is_some() {
        return Err(malformed());
    }
    Ok(&rest[end + "?>".len()..])
}

/// The element name the fragment is wrapped in while it is parsed. It never appears in the
/// result, because the wrapper's children are re-rooted onto a fresh document node.
const WRAPPER: &str = "oxixml-parse-xml-fragment";

/// A parsed document seen without its document URI.
///
/// [XDM 3.1 §6.1.2](https://www.w3.org/TR/xpath-datamodel-31/#DocumentNode) gives a document node
/// two independent URI properties: `base-uri`, which relative URIs inside the document resolve
/// against, and `document-uri`, which is the URI the document was *retrieved* from and which
/// `fn:doc($uri)` must round-trip. `fn:parse-xml` sets the first and leaves the second absent
/// (QT3 `parse-xml-007` and `parse-xml-011` pin the pair), but the tree builder has a single
/// document URI that seeds both. This store is that tree with one accessor overridden, so no node
/// is copied and every other property — including document order, on which node identity and the
/// `is` operator rest — is the parsed tree's own.
#[derive(Debug)]
struct ParsedDocument {
    tree: Arc<dyn XdmStore>,
}

impl ParsedDocument {
    /// The root of `tree`, seen through a store that reports no document URI.
    fn wrap(tree: &Tree) -> NodeRef {
        let root = tree.root();
        let store = Self {
            tree: Arc::clone(root.store()),
        };
        NodeRef::new(Arc::new(store), root.id())
    }
}

impl XdmStore for ParsedDocument {
    fn node_kind(&self, node: NodeId) -> NodeKind {
        self.tree.node_kind(node)
    }

    fn node_name(&self, node: NodeId) -> Option<ExpandedName> {
        self.tree.node_name(node)
    }

    fn string_value(&self, node: NodeId) -> String {
        self.tree.string_value(node)
    }

    fn typed_value(&self, node: NodeId) -> Result<Sequence, XdmError> {
        self.tree.typed_value(node)
    }

    fn type_name(&self, node: NodeId) -> Option<ExpandedName> {
        self.tree.type_name(node)
    }

    fn nilled(&self, node: NodeId) -> Option<bool> {
        self.tree.nilled(node)
    }

    fn parent(&self, node: NodeId) -> Option<NodeId> {
        self.tree.parent(node)
    }

    fn children(&self, node: NodeId) -> Vec<NodeId> {
        self.tree.children(node)
    }

    fn attributes(&self, node: NodeId) -> Vec<NodeId> {
        self.tree.attributes(node)
    }

    fn namespaces(&self, node: NodeId) -> Vec<NodeId> {
        self.tree.namespaces(node)
    }

    fn base_uri(&self, node: NodeId) -> Option<String> {
        self.tree.base_uri(node)
    }

    fn document_uri(&self, _node: NodeId) -> Option<String> {
        // The whole point of the wrapper: fn:parse-xml builds a document, it does not retrieve
        // one, so the document-uri property is absent however the base URI was seeded.
        None
    }

    fn document_order_key(&self, node: NodeId) -> DocumentOrderKey {
        self.tree.document_order_key(node)
    }

    fn is_id(&self, node: NodeId) -> bool {
        self.tree.is_id(node)
    }

    fn is_idref(&self, node: NodeId) -> bool {
        self.tree.is_idref(node)
    }

    fn root(&self, node: NodeId) -> NodeId {
        self.tree.root(node)
    }
}

/// One node of a re-rooted fragment. Every property is computed once, at construction time, so
/// that the store's accessors are plain lookups.
#[derive(Debug, Clone)]
struct FragmentNode {
    kind: NodeKind,
    name: Option<ExpandedName>,
    string_value: String,
    typed_value: Option<Sequence>,
    type_name: Option<ExpandedName>,
    nilled: Option<bool>,
    parent: Option<usize>,
    children: Vec<usize>,
    attributes: Vec<usize>,
    namespaces: Vec<usize>,
    base_uri: Option<String>,
    is_id: bool,
    is_idref: bool,
}

/// An [`XdmStore`] holding a document node whose children are the top-level nodes of a parsed
/// external entity.
#[derive(Debug)]
struct FragmentStore {
    tree_id: u64,
    nodes: Vec<FragmentNode>,
}

impl FragmentStore {
    /// Copies the children of `wrapper` under a fresh document node and returns that node.
    fn reroot(wrapper: &NodeRef, base_uri: Option<&str>) -> NodeRef {
        let mut store = Self {
            tree_id: next_tree_id(),
            nodes: Vec::new(),
        };
        store.nodes.push(FragmentNode {
            kind: NodeKind::Document,
            name: None,
            string_value: String::new(),
            typed_value: None,
            type_name: None,
            nilled: None,
            parent: None,
            children: Vec::new(),
            attributes: Vec::new(),
            namespaces: Vec::new(),
            base_uri: base_uri.map(str::to_owned),
            is_id: false,
            is_idref: false,
        });
        let mut children = Vec::new();
        for child in wrapper.children() {
            children.push(store.copy(&child, 0));
        }
        let mut text = String::new();
        for child in &children {
            if let Some(node) = store.nodes.get(*child) {
                text.push_str(&node.string_value);
            }
        }
        if let Some(document) = store.nodes.first_mut() {
            document.children = children;
            document.string_value.clone_from(&text);
            document.typed_value = Some(Sequence::from(AtomicValue::untyped_atomic(text)));
        }
        NodeRef::new(Arc::new(store), NodeId(0))
    }

    /// Copies `source` and its subtree into the arena, in document order, and returns its index.
    fn copy(&mut self, source: &NodeRef, parent: usize) -> usize {
        let index = self.nodes.len();
        self.nodes.push(FragmentNode {
            kind: source.node_kind(),
            name: source.node_name(),
            string_value: source.string_value(),
            typed_value: source.typed_value().ok(),
            type_name: source.type_name(),
            nilled: source.nilled(),
            parent: Some(parent),
            children: Vec::new(),
            attributes: Vec::new(),
            namespaces: Vec::new(),
            base_uri: source.base_uri(),
            is_id: source.is_id(),
            is_idref: source.is_idref(),
        });

        let namespaces: Vec<usize> = source
            .namespaces()
            .iter()
            .map(|node| self.copy_leaf(node, index))
            .collect();
        let attributes: Vec<usize> = source
            .attributes()
            .iter()
            .map(|node| self.copy_leaf(node, index))
            .collect();
        let mut children = Vec::new();
        for child in source.children() {
            children.push(self.copy(&child, index));
        }

        if let Some(node) = self.nodes.get_mut(index) {
            node.namespaces = namespaces;
            node.attributes = attributes;
            node.children = children;
        }
        index
    }

    fn copy_leaf(&mut self, source: &NodeRef, parent: usize) -> usize {
        let index = self.nodes.len();
        self.nodes.push(FragmentNode {
            kind: source.node_kind(),
            name: source.node_name(),
            string_value: source.string_value(),
            typed_value: source.typed_value().ok(),
            type_name: source.type_name(),
            nilled: source.nilled(),
            parent: Some(parent),
            children: Vec::new(),
            attributes: Vec::new(),
            namespaces: Vec::new(),
            base_uri: source.base_uri(),
            is_id: source.is_id(),
            is_idref: source.is_idref(),
        });
        index
    }

    fn node(&self, node: NodeId) -> Option<&FragmentNode> {
        usize::try_from(node.0)
            .ok()
            .and_then(|index| self.nodes.get(index))
    }

    fn ids(indices: &[usize]) -> Vec<NodeId> {
        indices
            .iter()
            .map(|index| NodeId(u64::try_from(*index).unwrap_or_default()))
            .collect()
    }
}

impl XdmStore for FragmentStore {
    fn node_kind(&self, node: NodeId) -> NodeKind {
        self.node(node).map_or(NodeKind::Document, |data| data.kind)
    }

    fn node_name(&self, node: NodeId) -> Option<ExpandedName> {
        self.node(node).and_then(|data| data.name.clone())
    }

    fn string_value(&self, node: NodeId) -> String {
        self.node(node)
            .map(|data| data.string_value.clone())
            .unwrap_or_default()
    }

    fn typed_value(&self, node: NodeId) -> Result<Sequence, XdmError> {
        match self.node(node).and_then(|data| data.typed_value.clone()) {
            Some(value) => Ok(value),
            None => Err(XdmError::new(
                oxixml_infoset::ErrorCode::Foty0012,
                "the node has no typed value",
            )),
        }
    }

    fn type_name(&self, node: NodeId) -> Option<ExpandedName> {
        self.node(node).and_then(|data| data.type_name.clone())
    }

    fn nilled(&self, node: NodeId) -> Option<bool> {
        self.node(node).and_then(|data| data.nilled)
    }

    fn parent(&self, node: NodeId) -> Option<NodeId> {
        self.node(node)
            .and_then(|data| data.parent)
            .map(|index| NodeId(u64::try_from(index).unwrap_or_default()))
    }

    fn children(&self, node: NodeId) -> Vec<NodeId> {
        self.node(node)
            .map(|data| Self::ids(&data.children))
            .unwrap_or_default()
    }

    fn attributes(&self, node: NodeId) -> Vec<NodeId> {
        self.node(node)
            .map(|data| Self::ids(&data.attributes))
            .unwrap_or_default()
    }

    fn namespaces(&self, node: NodeId) -> Vec<NodeId> {
        self.node(node)
            .map(|data| Self::ids(&data.namespaces))
            .unwrap_or_default()
    }

    fn base_uri(&self, node: NodeId) -> Option<String> {
        self.node(node).and_then(|data| data.base_uri.clone())
    }

    fn document_uri(&self, _node: NodeId) -> Option<String> {
        // fn:parse-xml-fragment leaves the document-uri property absent.
        None
    }

    fn document_order_key(&self, node: NodeId) -> DocumentOrderKey {
        DocumentOrderKey {
            tree: self.tree_id,
            position: node.0,
        }
    }

    fn is_id(&self, node: NodeId) -> bool {
        self.node(node).is_some_and(|data| data.is_id)
    }

    fn is_idref(&self, node: NodeId) -> bool {
        self.node(node).is_some_and(|data| data.is_idref)
    }

    fn root(&self, _node: NodeId) -> NodeId {
        NodeId(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::{Axis, axis_nodes};

    /// XML §4.3.1: `TextDecl ::= '<?xml' VersionInfo? EncodingDecl S? '?>'` — the encoding is
    /// required, and `standalone` is not part of it.
    #[test]
    fn a_fragment_may_open_with_a_text_declaration() -> FnResult<()> {
        assert_eq!(
            parse_xml_fragment("<?xml version='1.0' encoding='utf-8'?>abc", None)?.string_value(),
            "abc"
        );
        assert_eq!(
            parse_xml_fragment("<?xml encoding = \"UTF-8\" ?><a/>", None)?
                .children()
                .len(),
            1,
            "whitespace around the equals sign is allowed"
        );
        assert!(
            parse_xml_fragment("<?xml version='1.0'?><a/>", None).is_err(),
            "a text declaration must state the encoding"
        );
        assert!(
            parse_xml_fragment(
                "<?xml version='1.0' encoding='utf-8' standalone='yes'?>",
                None
            )
            .is_err(),
            "`standalone` belongs to an XML declaration, not a text declaration"
        );
        // An ordinary processing instruction whose target merely starts with `xml` is untouched.
        assert_eq!(
            parse_xml_fragment("<?xmlfoo bar?>abc", None)?
                .children()
                .len(),
            2
        );
        // XML separates the parts of a declaration with the `S` production and nothing else, so a
        // no-break space between them does not make two pseudo-attributes out of one.
        assert!(
            parse_xml_fragment("<?xml version='1.0'\u{a0}encoding='utf-8'?><a/>", None).is_err(),
            "only XML white space separates the parts of a text declaration"
        );
        Ok(())
    }

    #[test]
    fn parse_xml_returns_a_document_node() -> FnResult<()> {
        let document = parse_xml("<a><b>text</b></a>", None)?;
        assert_eq!(document.node_kind(), NodeKind::Document);
        assert_eq!(document.string_value(), "text");
        assert_eq!(document.parent(), None);
        let children = document.children();
        let [element] = children.as_slice() else {
            return Err(FnError::new(
                FnErrorCode::Fodc0006,
                "a document has one document element",
            ));
        };
        assert_eq!(element.node_kind(), NodeKind::Element);
        Ok(())
    }

    #[test]
    fn parse_xml_reports_ill_formed_input() {
        let error = parse_xml("<a>", None).expect_err("not well formed");
        assert!(error.is(FnErrorCode::Fodc0006));
        assert!(parse_xml("<a/><b/>", None).is_err());
    }

    #[test]
    fn parse_xml_validates_the_base_uri() -> FnResult<()> {
        assert!(
            parse_xml("<a/>", Some("relative/path"))
                .expect_err("not absolute")
                .is(FnErrorCode::Fodc0007)
        );
        let document = parse_xml("<a/>", Some("http://example.com/doc.xml"))?;
        assert_eq!(
            document.base_uri().as_deref(),
            Some("http://example.com/doc.xml")
        );
        Ok(())
    }

    /// F&O 3.1 §14.7.1 gives the result the static base URI of the call and leaves its
    /// `document-uri` absent — the two properties the QT3 pair `parse-xml-007` (`base-uri(…) eq
    /// static-base-uri()`) and `parse-xml-011` (`document-uri(…)` is empty) pin against each
    /// other, so that neither can be satisfied by dropping the other.
    #[test]
    fn parse_xml_has_a_base_uri_but_no_document_uri() -> FnResult<()> {
        let document = parse_xml("<a>foo</a>", Some("http://example.com/query.xq"))?;
        assert_eq!(
            document.base_uri().as_deref(),
            Some("http://example.com/query.xq"),
            "the base URI is the static base URI of the call"
        );
        assert_eq!(
            document.document_uri(),
            None,
            "the document was built from a string, not retrieved from a URI"
        );
        // Every node of the result agrees, and navigation stays inside the wrapper.
        let children = document.children();
        let [element] = children.as_slice() else {
            return Err(FnError::new(
                FnErrorCode::Fodc0006,
                "a document has one document element",
            ));
        };
        assert_eq!(
            element.base_uri().as_deref(),
            Some("http://example.com/query.xq")
        );
        assert_eq!(element.document_uri(), None);
        assert_eq!(element.root(), document);
        assert_eq!(element.parent().as_ref(), Some(&document));

        // The document-uri stays absent when there is no base URI to seed it with either.
        assert_eq!(parse_xml("<a/>", None)?.document_uri(), None);
        Ok(())
    }

    /// An `xml:base` inside the parsed text resolves against the static base URI, which is what
    /// the base URI has to be seeded through the tree for.
    #[test]
    fn parse_xml_resolves_xml_base_against_the_static_base_uri() -> FnResult<()> {
        let document = parse_xml(
            r#"<a xml:base="sub/"><b/></a>"#,
            Some("http://example.com/here/query.xq"),
        )?;
        let children = document.children();
        let [element] = children.as_slice() else {
            return Err(FnError::new(
                FnErrorCode::Fodc0006,
                "a document has one document element",
            ));
        };
        assert_eq!(
            element.base_uri().as_deref(),
            Some("http://example.com/here/sub/")
        );
        Ok(())
    }

    #[test]
    fn a_fragment_may_have_several_top_level_nodes() -> FnResult<()> {
        let document = parse_xml_fragment("<a/>text<b/>", None)?;
        assert_eq!(document.node_kind(), NodeKind::Document);
        assert_eq!(document.children().len(), 3);
        assert_eq!(document.string_value(), "text");
        assert_eq!(document.parent(), None);
        Ok(())
    }

    #[test]
    fn a_fragment_preserves_structure_and_navigation() -> FnResult<()> {
        let document = parse_xml_fragment("<a x='1'><c/></a><b/>", None)?;
        let children = document.children();
        assert_eq!(children.len(), 2);

        let a = children[0].clone();
        assert_eq!(a.node_kind(), NodeKind::Element);
        assert_eq!(
            a.node_name().map(|name| name.local_name().to_owned()),
            Some("a".to_owned())
        );
        assert_eq!(a.attributes().len(), 1);
        assert_eq!(a.attributes()[0].string_value(), "1");
        assert_eq!(a.parent(), Some(document.clone()));
        assert_eq!(a.root(), document);

        // Document order is preserved through the copy.
        let descendants = axis_nodes(&document, Axis::Descendant);
        assert_eq!(descendants.len(), 3);
        assert!(descendants[0].document_order_key() < descendants[1].document_order_key());
        Ok(())
    }

    #[test]
    fn an_empty_fragment_has_no_children() -> FnResult<()> {
        let document = parse_xml_fragment("", None)?;
        assert!(document.children().is_empty());
        assert_eq!(document.string_value(), "");
        Ok(())
    }

    #[test]
    fn a_fragment_atomizes_as_untyped() -> FnResult<()> {
        let document = parse_xml_fragment("<a>x</a>", None)?;
        let atomized = document.typed_value()?;
        assert_eq!(atomized.len(), 1);
        assert_eq!(
            atomized.first().map(ToString::to_string),
            Some("x".to_owned())
        );
        Ok(())
    }

    #[test]
    fn an_unbalanced_fragment_is_an_error() {
        assert!(parse_xml_fragment("<a>", None).is_err());
        assert!(parse_xml_fragment("</a>", None).is_err());
    }

    #[test]
    fn namespaces_survive_the_reroot() -> FnResult<()> {
        let document = parse_xml_fragment("<p:a xmlns:p='urn:p'/>", None)?;
        let a = document
            .children()
            .first()
            .cloned()
            .ok_or_else(|| FnError::new(FnErrorCode::Fodc0006, "expected one child"))?;
        assert_eq!(
            a.node_name()
                .and_then(|name| name.namespace().map(str::to_owned)),
            Some("urn:p".to_owned())
        );
        assert!(
            a.namespaces()
                .iter()
                .any(|node| node.string_value() == "urn:p")
        );
        Ok(())
    }

    /// F&O 3.1 §14.10.1 leaves external-entity retrieval implementation-defined, and this
    /// implementation retrieves — from the directory of the static base URI, and only when that
    /// base URI names a file (QT3 `fn/parse-xml/parse-xml-010`, whose document declares
    /// `<!ENTITY foo SYSTEM 'parse-xml/foo.entity'>` and expects `<a><bar>baz</bar></a>`).
    ///
    /// Both directions are pinned here: with a `file:` base URI the entity contributes its
    /// nodes, and without one the same document is `FODC0006`, so no caller acquires the ability
    /// to read a file by accident.
    #[test]
    fn an_external_entity_is_retrieved_beside_the_static_base_uri()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-fn-parse-xml-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let entities = directory.join("parse-xml");
        std::fs::create_dir_all(&entities)?;
        std::fs::write(entities.join("foo.entity"), "<bar>baz</bar>")?;

        // The case's document, verbatim.
        let document = concat!(
            "<!DOCTYPE a [<!ELEMENT a (#PCDATA)>",
            "<!ENTITY foo SYSTEM 'parse-xml/foo.entity'>]><a>&foo;</a>"
        );
        let base_uri = format!("file://{}/parse-xml.xml", directory.display());
        let parsed = parse_xml(document, Some(&base_uri));
        let outcome = parsed.map(|node| {
            crate::xmlio::serialize(
                &Sequence::from(node),
                &crate::xmlio::SerializationParameters::default(),
            )
        });
        std::fs::remove_dir_all(&directory).ok();
        assert_eq!(outcome??, "<a><bar>baz</bar></a>");

        // No static base URI, and a base URI in another scheme, retrieve nothing at all.
        for base in [None, Some("http://example.com/query.xq")] {
            let error = parse_xml(document, base).expect_err("no entity may be retrieved");
            assert!(error.is(FnErrorCode::Fodc0006), "{base:?} gave {error}");
        }
        Ok(())
    }

    /// The entity directory is the *parent* of the file the static base URI names, and only a
    /// `file:` URI naming this machine has one.
    #[test]
    fn only_a_local_file_base_uri_names_an_entity_directory() {
        assert_eq!(
            entity_directory_of("file:///suite/fn/parse-xml.xml"),
            Some(PathBuf::from("/suite/fn"))
        );
        // A percent escape in the path is decoded, and `file:/…` is the same as `file:///…`.
        assert_eq!(
            entity_directory_of("file:///suite/my%20tests/parse-xml.xml"),
            Some(PathBuf::from("/suite/my tests"))
        );
        assert_eq!(
            entity_directory_of("file:/suite/fn/parse-xml.xml"),
            Some(PathBuf::from("/suite/fn"))
        );
        // A query or fragment is not part of the path.
        assert_eq!(
            entity_directory_of("file:///suite/fn/parse-xml.xml?v=1#top"),
            Some(PathBuf::from("/suite/fn"))
        );
        for elsewhere in [
            "http://example.com/query.xq",
            "https://example.com/query.xq",
            "urn:example:query",
            "file://elsewhere/suite/fn/parse-xml.xml",
        ] {
            assert_eq!(entity_directory_of(elsewhere), None, "{elsewhere}");
        }
    }
}
