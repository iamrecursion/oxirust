//! The functions on nodes of
//! [F&O 3.1 §14](https://www.w3.org/TR/xpath-functions-31/#nodes): naming (`fn:name`,
//! `fn:local-name`, `fn:namespace-uri`), navigation (`fn:root`, `fn:has-children`), language
//! matching (`fn:lang`), the ID functions (`fn:id`, `fn:idref`, `fn:element-with-id`) and
//! `fn:generate-id`.

use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_infoset::{
    Annotation, Axis, ExpandedName, NodeKind, NodeRef, TreeBuilder, XdmError, XdmNode, axis_nodes,
    descendant_or_self, is_ancestor_of, sort_document_order,
};
use oxixml_qname::XML_NAMESPACE;

/// [`fn:name`](https://www.w3.org/TR/xpath-functions-31/#func-name): the node's name in its
/// lexical `QName` form, or the zero-length string for a node with no name.
///
/// F&O 3.1 §14.1.2 defines this as `string(node-name($arg))`, and XDM 3.1 §5.4 makes a node
/// name a *triple* whose third component is the prefix the name was written with. So the
/// node's own prefix is what this reports whenever the name carries one — a computed attribute
/// named `QName('urn:example', 'p:foo')` is `p:foo` even with no parent to look anything up in,
/// and even where an ancestor happens to bind `p` to a different namespace. Only a name that
/// retained no prefix falls back to the in-scope bindings of the element the name belongs to.
///
/// # Examples
///
/// ```
/// use oxixml_fn::nodes::name;
/// use oxixml_infoset::{Tree, XdmNode};
///
/// let tree = Tree::parse_str("<p:a xmlns:p='urn:p'><b/></p:a>")?;
/// let root = tree.document_element().expect("a document element");
/// assert_eq!(name(&root), "p:a");
/// assert_eq!(name(&root.children()[0]), "b");
/// assert_eq!(name(&tree.root()), "");
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[must_use]
pub fn name(node: &NodeRef) -> String {
    let Some(expanded) = node.node_name() else {
        return String::new();
    };
    let Some(namespace) = expanded.namespace() else {
        return expanded.local_name().to_owned();
    };
    if let Some(prefix) = expanded.prefix() {
        return format!("{prefix}:{}", expanded.local_name());
    }
    if namespace == XML_NAMESPACE {
        return format!("xml:{}", expanded.local_name());
    }
    let element = if node.node_kind() == NodeKind::Element {
        Some(node.clone())
    } else {
        node.parent()
    };
    let prefix = element.and_then(|element| {
        crate::qnames::in_scope_namespaces(&element)
            .into_iter()
            .find(|(prefix, bound)| bound == namespace && prefix.is_some())
            .and_then(|(prefix, _)| prefix)
    });
    match prefix {
        Some(prefix) => format!("{prefix}:{}", expanded.local_name()),
        None => expanded.local_name().to_owned(),
    }
}

/// [`fn:local-name`](https://www.w3.org/TR/xpath-functions-31/#func-local-name).
#[must_use]
pub fn local_name(node: &NodeRef) -> String {
    node.node_name()
        .map(|name| name.local_name().to_owned())
        .unwrap_or_default()
}

/// [`fn:namespace-uri`](https://www.w3.org/TR/xpath-functions-31/#func-namespace-uri), which is
/// the zero-length `xs:anyURI` for a node with no name or a name in no namespace.
#[must_use]
pub fn namespace_uri(node: &NodeRef) -> String {
    node.node_name()
        .and_then(|name| name.namespace().map(str::to_owned))
        .unwrap_or_default()
}

/// [`fn:root`](https://www.w3.org/TR/xpath-functions-31/#func-root).
#[must_use]
pub fn root(node: &NodeRef) -> NodeRef {
    node.root()
}

/// [`fn:has-children`](https://www.w3.org/TR/xpath-functions-31/#func-has-children).
#[must_use]
pub fn has_children(node: &NodeRef) -> bool {
    !node.children().is_empty()
}

/// [`fn:lang`](https://www.w3.org/TR/xpath-functions-31/#func-lang): whether the node's
/// `xml:lang` — its own or the nearest ancestor's — is `test_language` or a sublanguage of it.
///
/// The comparison is case-insensitive, and a sublanguage match requires the extra text to begin
/// with a hyphen, so `en` matches `en-GB` but not `english`.
///
/// # Examples
///
/// ```
/// use oxixml_fn::nodes::lang;
/// use oxixml_infoset::{Tree, XdmNode};
///
/// let tree = Tree::parse_str("<a xml:lang='en'><b/><c xml:lang='en-GB'/><d xml:lang='fr'/></a>")?;
/// let a = tree.document_element().expect("a document element");
/// let children = a.children();
///
/// assert!(lang("en", &a));
/// assert!(lang("EN", &a));
/// // Inherited from the ancestor.
/// assert!(lang("en", &children[0]));
/// // `en` matches the sublanguage `en-GB`.
/// assert!(lang("en", &children[1]));
/// assert!(!lang("en-US", &children[1]));
/// assert!(!lang("en", &children[2]));
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[must_use]
pub fn lang(test_language: &str, node: &NodeRef) -> bool {
    let Some(declared) = xml_lang(node) else {
        return false;
    };
    let declared = declared.to_lowercase();
    let wanted = test_language.to_lowercase();
    declared == wanted
        || (declared.len() > wanted.len()
            && declared.starts_with(&wanted)
            && declared.as_bytes().get(wanted.len()) == Some(&b'-'))
}

/// The `xml:lang` in force on a node: its own, or the nearest ancestor's.
#[must_use]
pub fn xml_lang(node: &NodeRef) -> Option<String> {
    let mut current = Some(node.clone());
    while let Some(candidate) = current {
        if candidate.node_kind() == NodeKind::Element {
            for attribute in candidate.attributes() {
                let is_xml_lang = attribute.node_name().is_some_and(|name| {
                    name.namespace() == Some(XML_NAMESPACE) && name.local_name() == "lang"
                });
                if is_xml_lang {
                    return Some(attribute.string_value());
                }
            }
        }
        current = candidate.parent();
    }
    None
}

/// [`fn:generate-id`](https://www.w3.org/TR/xpath-functions-31/#func-generate-id): a string that
/// uniquely identifies a node and is a valid `NCName`.
///
/// The identifier is derived from the node's document-order key, so it is stable for as long as
/// the tree is alive and distinct for every node of every tree in the process.
///
/// # Examples
///
/// ```
/// use oxixml_fn::nodes::generate_id;
/// use oxixml_infoset::{Tree, XdmNode};
///
/// let tree = Tree::parse_str("<a><b/><c/></a>")?;
/// let root = tree.document_element().expect("a document element");
/// let children = root.children();
///
/// assert_eq!(generate_id(&root), generate_id(&root));
/// assert_ne!(generate_id(&children[0]), generate_id(&children[1]));
/// assert!(generate_id(&root).starts_with('d'));
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[must_use]
pub fn generate_id(node: &NodeRef) -> String {
    let key = node.document_order_key();
    format!("d{}n{}", key.tree, key.position)
}

/// [`fn:path`](https://www.w3.org/TR/xpath-functions-31/#func-path): a path expression that
/// selects `node` and nothing else, written with the `Q{uri}local` notation so that it needs no
/// namespace bindings.
///
/// A node whose root is not a document node is rooted at `Q{...}root()`, as the specification
/// requires.
///
/// # Examples
///
/// ```
/// use oxixml_fn::nodes::path;
/// use oxixml_infoset::{Tree, XdmNode};
///
/// let tree = Tree::parse_str("<a><b/><b x='1'/></a>")?;
/// assert_eq!(path(&tree.root()), "/");
///
/// let a = tree.document_element().expect("a document element");
/// assert_eq!(path(&a), "/Q{}a[1]");
///
/// let second = a.children()[1].clone();
/// assert_eq!(path(&second), "/Q{}a[1]/Q{}b[2]");
/// assert_eq!(path(&second.attributes()[0]), "/Q{}a[1]/Q{}b[2]/@x");
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[must_use]
pub fn path(node: &NodeRef) -> String {
    // The root of the tree is what `Q{…}root()` denotes, so it contributes no step of its own —
    // whichever node it is. In a tree with no document node the root is an element (or a
    // parentless attribute or text node), and `fn:path(<a b="c"/>/@b)` is therefore
    // `Q{…}root()/@b`, not `Q{…}root()/Q{}a[1]/@b` (F&O 3.1 §14.7.1; `path016`–`path019`).
    let root = node.root();
    let mut steps = Vec::new();
    let mut current = Some(node.clone());
    while let Some(candidate) = current {
        if candidate.node_kind() == NodeKind::Document || candidate == root {
            break;
        }
        steps.push(path_step(&candidate));
        current = candidate.parent();
    }
    steps.reverse();
    let rooted_at_document = root.node_kind() == NodeKind::Document;
    if steps.is_empty() {
        return if rooted_at_document {
            "/".to_owned()
        } else {
            format!("Q{{{}}}root()", oxixml_fn_namespace())
        };
    }
    if rooted_at_document {
        format!("/{}", steps.join("/"))
    } else {
        format!("Q{{{}}}root()/{}", oxixml_fn_namespace(), steps.join("/"))
    }
}

fn oxixml_fn_namespace() -> &'static str {
    "http://www.w3.org/2005/xpath-functions"
}

fn path_step(node: &NodeRef) -> String {
    let name = node.node_name();
    let namespace = name
        .as_ref()
        .and_then(|name| name.namespace())
        .unwrap_or_default();
    let local = name
        .as_ref()
        .map(|name| name.local_name().to_owned())
        .unwrap_or_default();
    match node.node_kind() {
        NodeKind::Element => {
            format!("Q{{{namespace}}}{local}[{}]", sibling_position(node))
        }
        NodeKind::Attribute => {
            if namespace.is_empty() {
                format!("@{local}")
            } else {
                format!("@Q{{{namespace}}}{local}")
            }
        }
        // F&O 3.1 §14.7.1: a namespace node's step is `namespace::<prefix>`, but the node that
        // binds the *default* namespace has no name to spell, and the specification gives it a
        // predicate instead of a name test (QT3 `path013`).
        NodeKind::Namespace => {
            if local.is_empty() {
                format!(
                    "namespace::*[Q{{{}}}local-name()=\"\"]",
                    oxixml_fn_namespace()
                )
            } else {
                format!("namespace::{local}")
            }
        }
        NodeKind::Text => format!("text()[{}]", sibling_position(node)),
        NodeKind::Comment => format!("comment()[{}]", sibling_position(node)),
        NodeKind::ProcessingInstruction => {
            format!(
                "processing-instruction({local})[{}]",
                sibling_position(node)
            )
        }
        NodeKind::Document => String::new(),
    }
}

/// The one-based position of `node` among the siblings that a `fn:path` step would select.
fn sibling_position(node: &NodeRef) -> usize {
    let Some(parent) = node.parent() else {
        return 1;
    };
    let kind = node.node_kind();
    let name = node.node_name();
    parent
        .children()
        .into_iter()
        .filter(|sibling| sibling.node_kind() == kind && sibling.node_name() == name)
        .position(|sibling| sibling == *node)
        .map_or(1, |index| index + 1)
}

/// [`fn:innermost`](https://www.w3.org/TR/xpath-functions-31/#func-innermost): the nodes of
/// `sequence` that are not ancestors of any other node in it, in document order.
#[must_use]
pub fn innermost(nodes: &[NodeRef]) -> Vec<NodeRef> {
    let mut kept: Vec<NodeRef> = nodes
        .iter()
        .filter(|candidate| {
            !nodes
                .iter()
                .any(|other| other != *candidate && is_ancestor_of(candidate, other))
        })
        .cloned()
        .collect();
    sort_document_order(&mut kept);
    kept.dedup();
    kept
}

/// [`fn:outermost`](https://www.w3.org/TR/xpath-functions-31/#func-outermost): the nodes of
/// `sequence` that have no ancestor in it, in document order.
#[must_use]
pub fn outermost(nodes: &[NodeRef]) -> Vec<NodeRef> {
    let mut kept: Vec<NodeRef> = nodes
        .iter()
        .filter(|candidate| {
            !nodes
                .iter()
                .any(|other| other != *candidate && is_ancestor_of(other, candidate))
        })
        .cloned()
        .collect();
    sort_document_order(&mut kept);
    kept.dedup();
    kept
}

/// Splits an `fn:id` argument into whitespace-separated `xs:ID` tokens.
///
/// [F&O 3.1 §14.5.1](https://www.w3.org/TR/xpath-functions-31/#func-id) reads each string in the
/// argument "as a whitespace-separated sequence of tokens", which is the `xs:IDREFS` list
/// separator: XML's four characters — space, tab, carriage return and line feed — and no others.
/// Every other character Unicode calls white space, the no-break space among them, stays inside
/// the token it appears in, where it makes that token no `xs:ID` and so a candidate that matches
/// nothing.
#[must_use]
pub fn id_tokens(values: &[String]) -> Vec<String> {
    values
        .iter()
        .flat_map(|value| value.split(oxixml_text::ws::is_xml_ws))
        .filter(|token| !token.is_empty())
        .map(str::to_owned)
        .collect()
}

/// [`fn:id`](https://www.w3.org/TR/xpath-functions-31/#func-id): the element nodes of `node`'s
/// tree whose `xs:ID` matches one of the supplied tokens, in document order.
///
/// # Errors
///
/// [`FnErrorCode::Fodc0001`] when `node` is not in a tree whose root is a document node.
///
/// # Examples
///
/// ```
/// use oxixml_fn::nodes::id;
/// use oxixml_infoset::{Tree, XdmNode};
///
/// let tree = Tree::parse_str("<a><b xml:id='x'/><c xml:id='y'/></a>")?;
/// let root = tree.document_element().expect("a document element");
///
/// let found = id(&["x y".to_owned()], &root)?;
/// assert_eq!(found.len(), 2);
/// assert!(id(&["z".to_owned()], &root)?.is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn id(values: &[String], node: &NodeRef) -> FnResult<Vec<NodeRef>> {
    let tokens = id_tokens(values);
    let root = require_document_root(node, "fn:id")?;
    let mut found: Vec<NodeRef> = descendant_or_self(&root)
        .into_iter()
        .filter(|candidate| candidate.node_kind() == NodeKind::Element)
        .filter(|element| element_has_id(element, &tokens))
        .collect();
    sort_document_order(&mut found);
    Ok(found)
}

/// [`fn:element-with-id`](https://www.w3.org/TR/xpath-functions-31/#func-element-with-id), which
/// for a tree without a DTD or schema behaves exactly like [`id`].
///
/// # Errors
///
/// See [`id`].
pub fn element_with_id(values: &[String], node: &NodeRef) -> FnResult<Vec<NodeRef>> {
    id(values, node)
}

/// [`fn:idref`](https://www.w3.org/TR/xpath-functions-31/#func-idref): the nodes of `node`'s tree
/// whose `xs:IDREF` or `xs:IDREFS` value contains one of the supplied candidate values.
///
/// The argument is *not* tokenized. [F&O 3.1
/// §14.5.3](https://www.w3.org/TR/xpath-functions-31/#func-idref) reserves the whitespace
/// tokenization of [`id`] for `fn:id` alone: here each string in `values` is one candidate IDREF
/// value in its own right. A candidate that could not be an IDREF at all — one holding
/// whitespace, or the empty string — therefore matches nothing, because the values it is compared
/// against are the individual whitespace-separated tokens of a node's `xs:IDREFS` value and every
/// one of those is a non-empty name.
///
/// # Errors
///
/// [`FnErrorCode::Fodc0001`] when `node` is not in a tree whose root is a document node.
///
/// # Examples
///
/// ```
/// use oxixml_fn::nodes::idref;
/// use oxixml_infoset::{Tree, XdmNode};
///
/// let tree = Tree::parse_str(
///     "<!DOCTYPE r [<!ELEMENT r (a*)><!ELEMENT a EMPTY>\
///      <!ATTLIST a id ID #REQUIRED><!ATTLIST a ref IDREFS #IMPLIED>]>\
///      <r><a id='x'/><a id='y' ref='x z'/></r>",
/// )?;
/// let root = tree.document_element().expect("a document element");
///
/// // `ref='x z'` holds the token `x`, so its attribute node is the answer …
/// assert_eq!(idref(&["x".to_owned()], &root)?.len(), 1);
/// // … but `"x z"` as a whole is not an IDREF value, and matches nothing.
/// assert!(idref(&["x z".to_owned()], &root)?.is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn idref(values: &[String], node: &NodeRef) -> FnResult<Vec<NodeRef>> {
    let root = require_document_root(node, "fn:idref")?;
    let mut found = Vec::new();
    for element in descendant_or_self(&root) {
        if element.node_kind() != NodeKind::Element {
            continue;
        }
        if element.is_idref() && references(&element.string_value(), values) {
            found.push(element.clone());
        }
        for attribute in axis_nodes(&element, Axis::Attribute) {
            if attribute.is_idref() && references(&attribute.string_value(), values) {
                found.push(attribute);
            }
        }
    }
    sort_document_order(&mut found);
    Ok(found)
}

fn element_has_id(element: &NodeRef, tokens: &[String]) -> bool {
    if element.is_id() && tokens.iter().any(|token| *token == element.string_value()) {
        return true;
    }
    element.attributes().iter().any(|attribute| {
        attribute.is_id()
            && tokens
                .iter()
                .any(|token| *token == attribute.string_value())
    })
}

/// Whether the `xs:IDREF`/`xs:IDREFS` string `value` holds a reference equal to one of the
/// `candidates`.
///
/// An `xs:IDREFS` value is a list, and a list is separated by XML's four whitespace characters
/// and no others — the same rule [`id_tokens`] states, and for the same reason: splitting on any
/// other character Unicode calls white space would break a single reference in two and answer
/// for a document that does not say what the split one says.
fn references(value: &str, candidates: &[String]) -> bool {
    value
        .split(oxixml_text::ws::is_xml_ws)
        .filter(|reference| !reference.is_empty())
        .any(|reference| candidates.iter().any(|candidate| candidate == reference))
}

fn require_document_root(node: &NodeRef, what: &str) -> FnResult<NodeRef> {
    let root = node.root();
    if root.node_kind() == NodeKind::Document {
        Ok(root)
    } else {
        Err(FnError::new(
            FnErrorCode::Fodc0001,
            format!("{what} needs a node whose root is a document node"),
        ))
    }
}

/// [`fn:copy-of`](https://www.w3.org/TR/xslt-30/#func-copy-of) applied to a single node: a deep
/// copy, sharing no node with `node`'s own tree.
///
/// Every descendant is copied too, but `node`'s ancestors are not: a copied element becomes the
/// root of a brand new tree, exactly as evaluating `xsl:copy-of` with `copy-namespaces="yes"`
/// does. Compare [`snapshot`], whose copy keeps the path back to the root.
///
/// The caller maps a non-node item straight through unchanged rather than calling this — a
/// `copy-of`/`snapshot` argument is `item()*`, and only the node items need a fresh identity.
///
/// # Errors
///
/// [`FnErrorCode::Fodc0006`] when `node` is an [`NodeKind::Attribute`], [`NodeKind::Namespace`]
/// or [`NodeKind::Text`] node — a node the copy of which would itself have to be parentless,
/// since `copy_of` carries no parent over. [`TreeBuilder`] — the general-purpose tree constructor
/// every copy in this crate goes through — always roots what it builds at a document node,
/// which XDM does not allow a text node to be a direct child of, and which an attribute or a
/// namespace node never has anyway; every other kind, and every node *nested inside* a larger
/// copy regardless of its own kind, copies without error. [`snapshot`] does carry the parent
/// over, and so copies all three whenever they have one.
///
/// # Examples
///
/// ```
/// use oxixml_fn::nodes::copy_of;
/// use oxixml_infoset::{NodeKind, Tree, XdmNode};
///
/// let tree = Tree::parse_str("<r><a x='1'><b/></a></r>")?;
/// let root = tree.document_element().expect("a document element");
/// let a = root.children()[0].clone();
/// let copy = copy_of(&a)?;
///
/// assert_ne!(copy, a, "a fresh identity");
/// // `a`'s own new document, not `r` — the ancestor is not carried over.
/// let parent = copy.parent().expect("the document node that roots the copy");
/// assert_eq!(parent.node_kind(), NodeKind::Document);
/// assert_ne!(parent, tree.root());
/// assert_eq!(copy.attributes()[0].string_value(), "1");
/// assert_eq!(copy.children().len(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn copy_of(node: &NodeRef) -> FnResult<NodeRef> {
    if let Some(error) = standalone_copy_unsupported(node, false) {
        return Err(error);
    }
    let mut builder = TreeBuilder::new();
    if node.node_kind() == NodeKind::Document {
        // The copy *is* a document node standing for the same document, so it carries the
        // document's own properties — see `carry_declarations`.
        builder.carry_declarations(node)?;
    }
    copy_subtree(&mut builder, node)?;
    Ok(extract_copy(builder, node.node_kind())?)
}

/// [`fn:snapshot`](https://www.w3.org/TR/xslt-30/#func-snapshot) applied to a single node: like
/// [`copy_of`], but the copy also carries `node`'s ancestors up to and including the root — with
/// each ancestor's own attributes and in-scope namespaces, but *not* any of the ancestor's other
/// descendants — so that `root()` and the parent axis still work on the result the way they did
/// on `node` itself.
///
/// Because the ancestors come too, an attribute, namespace or text node that *has* a parent
/// snapshots successfully here even though [`copy_of`] cannot copy one: the copy is attached to
/// the copy of that parent, and it is the attached node — not the root of what was built — that
/// is returned.
///
/// # Errors
///
/// [`FnErrorCode::Fodc0006`] for an attribute, namespace or text node with no parent, whose copy
/// would itself have to be parentless. See [`copy_of`].
///
/// # Examples
///
/// ```
/// use oxixml_fn::nodes::{local_name, snapshot};
/// use oxixml_infoset::{NodeKind, Tree, XdmNode};
///
/// let tree = Tree::parse_str("<a x='1'><b/><c><d/></c></a>")?;
/// let a = tree.document_element().expect("a document element");
/// let c = a.children()[1].clone();
/// let copy = snapshot(&c)?;
///
/// // The ancestor chain survives, carrying the ancestor's own attributes …
/// let copied_parent = copy.parent().expect("the ancestor path was kept");
/// assert_eq!(local_name(&copied_parent), "a");
/// assert_eq!(copied_parent.attributes()[0].string_value(), "1");
/// assert_eq!(copy.root().node_kind(), NodeKind::Document);
/// // … but not the ancestor's other children (`b` is not among them).
/// assert_eq!(copied_parent.children().len(), 1);
/// // `c`'s own subtree, by contrast, is copied in full.
/// assert_eq!(copy.children().len(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn snapshot(node: &NodeRef) -> FnResult<NodeRef> {
    if let Some(error) = standalone_copy_unsupported(node, true) {
        return Err(error);
    }
    let mut chain = vec![node.clone()];
    let mut current = node.parent();
    while let Some(ancestor) = current {
        current = ancestor.parent();
        chain.push(ancestor);
    }
    chain.reverse(); // root-first, `node` last.
    // Every ancestor written by `copy_ancestor_chain` — every `Element` in `chain` before
    // `node`, a leading `Document` contributes no explicit level of its own — becomes one level
    // of nesting under `TreeBuilder`'s own implicit document; `node` itself is one level deeper
    // still, unless it *is* the document (which never has ancestors, and is `TreeBuilder`'s
    // implicit root already) or an attribute or namespace node, which is not a child of the
    // copied parent element but a property of it and so adds no level at all.
    let element_ancestors = chain[..chain.len() - 1]
        .iter()
        .filter(|ancestor| ancestor.node_kind() == NodeKind::Element)
        .count();
    let attached = matches!(node.node_kind(), NodeKind::Attribute | NodeKind::Namespace);
    let descents =
        element_ancestors + usize::from(!attached && node.node_kind() != NodeKind::Document);
    let mut builder = TreeBuilder::new();
    // A snapshot keeps the path back to the root, so its root stands for the root of the tree
    // `node` was taken from — and carries that document node's own properties with it.
    builder.carry_declarations(node)?;
    copy_ancestor_chain(&mut builder, &chain)?;
    let tree = builder.build()?;
    let mut cursor = tree.root();
    for _ in 0..descents {
        cursor = cursor.children().into_iter().next().ok_or_else(|| {
            XdmError::new(
                oxixml_infoset::ErrorCode::Fodc0006,
                "the snapshot copy came out shallower than the original ancestor path",
            )
        })?;
    }
    if attached {
        return attached_copy(&cursor, node);
    }
    Ok(cursor)
}

/// Picks the copy of attribute or namespace node `original` out of `element`, the copy of the
/// element that owns it.
///
/// `open_element_shell` writes every attribute and every namespace binding of an ancestor, so the
/// copy is already there and only has to be found again — by name. That is enough to identify it:
/// an element's attribute names are distinct, and the namespaces it reports are the ones *in
/// scope* on it, where a prefix stands for exactly one URI however many ancestors also bind it
/// (the default binding being the one whose name is [`None`]).
fn attached_copy(element: &NodeRef, original: &NodeRef) -> FnResult<NodeRef> {
    let wanted = original.node_name();
    let candidates = match original.node_kind() {
        NodeKind::Attribute => element.attributes(),
        _ => element.namespaces(),
    };
    candidates
        .into_iter()
        .find(|candidate| candidate.node_name() == wanted)
        .ok_or_else(|| {
            FnError::new(
                FnErrorCode::Fodc0006,
                format!(
                    "the snapshot of a {} node lost it from the copy of its parent element",
                    original.node_kind()
                ),
            )
        })
}

/// Whether `copy_of`/`snapshot` can build a copy of `node` at all, given that
/// [`TreeBuilder`] always roots what it builds at a document node.
///
/// Three kinds need an element to belong to. An [`NodeKind::Attribute`] and a
/// [`NodeKind::Namespace`] node are properties of an element rather than children of anything,
/// and a [`NodeKind::Text`] node needs *some* element to be its parent because XDM does not allow
/// a text node to be a direct child of a document node. `snapshot` (`carries_ancestor = true`)
/// supplies that element whenever `node` itself has a parent to carry over; `copy_of` never
/// carries the original parent over, so for it all three are unsupported regardless. What is left
/// in either case — a node of one of those kinds with no parent at all — would have to be copied
/// as a node with no parent whatsoever, which `TreeBuilder` has no way to build.
fn standalone_copy_unsupported(node: &NodeRef, carries_ancestor: bool) -> Option<FnError> {
    let blocked = match node.node_kind() {
        NodeKind::Attribute | NodeKind::Namespace | NodeKind::Text => {
            !(carries_ancestor && node.parent().is_some())
        }
        _ => false,
    };
    blocked.then(|| {
        FnError::new(
            FnErrorCode::Fodc0006,
            format!(
                "this processor cannot build a parentless copy of a {} node: fn:copy-of and \
                 fn:snapshot need a node whose copy can be given a document or element parent",
                node.node_kind()
            ),
        )
    })
}

/// Writes a deep copy of `node` — attributes, in-scope namespaces (`xml` aside, which every
/// tree has implicitly) and children, recursively — into whatever `builder` currently has open.
fn copy_subtree(builder: &mut TreeBuilder, node: &NodeRef) -> Result<(), XdmError> {
    match node.node_kind() {
        NodeKind::Document => {
            for child in node.children() {
                copy_subtree(builder, &child)?;
            }
        }
        NodeKind::Element => {
            open_element_shell(builder, node)?;
            for child in node.children() {
                copy_subtree(builder, &child)?;
            }
            builder.end_element()?;
        }
        NodeKind::Text => {
            builder.text(node.string_value())?;
        }
        NodeKind::Comment => {
            builder.comment(node.string_value())?;
        }
        NodeKind::ProcessingInstruction => {
            builder.processing_instruction(pi_target(node), node.string_value())?;
        }
        // Neither kind can be a document's or an element's child, and both are written by
        // `open_element_shell` as part of the element that owns them. The only way either can
        // reach here is as the copied node itself: `copy_of` rejects that outright and
        // `snapshot` picks the copy back out of the parent element it just wrote.
        NodeKind::Attribute | NodeKind::Namespace => {}
    }
    Ok(())
}

/// Writes the root-to-`node` path `chain` describes (outermost first, `node` itself last) into
/// `builder`: every ancestor element is opened with its own attributes and namespaces but only
/// the next step of the path as a child, and the last element of `chain` gets its full subtree.
fn copy_ancestor_chain(builder: &mut TreeBuilder, chain: &[NodeRef]) -> Result<(), XdmError> {
    let [here, rest @ ..] = chain else {
        return Ok(());
    };
    match here.node_kind() {
        // A document node has no parent, so it can only ever be `chain`'s first entry — either
        // with more of the path still ahead (the ordinary case) or, when `node` itself is the
        // document being snapshotted, as the sole entry: `rest` is empty and there is no
        // "shell" to open, only the document's own children to copy in full.
        NodeKind::Document if rest.is_empty() => {
            for child in here.children() {
                copy_subtree(builder, &child)?;
            }
        }
        NodeKind::Document => copy_ancestor_chain(builder, rest)?,
        NodeKind::Element => {
            open_element_shell(builder, here)?;
            if rest.is_empty() {
                for child in here.children() {
                    copy_subtree(builder, &child)?;
                }
            } else {
                copy_ancestor_chain(builder, rest)?;
            }
            builder.end_element()?;
        }
        // Only a document node or an element node can have children, so every other kind can
        // only be the last (and first) element of the chain — `node` itself, with no ancestors.
        _ => copy_subtree(builder, here)?,
    }
    Ok(())
}

/// Opens a copy of element `node` — name, in-scope namespaces and attributes — leaving it open
/// for the caller to add children and then close it.
fn open_element_shell(builder: &mut TreeBuilder, node: &NodeRef) -> Result<(), XdmError> {
    let name = node
        .node_name()
        .unwrap_or_else(|| ExpandedName::new(None::<&str>, "element"));
    builder.start_element(name)?;
    for namespace in node.namespaces() {
        let prefix = namespace
            .node_name()
            .map(|name| name.local_name().to_owned());
        if prefix.as_deref() == Some("xml") {
            // Every tree `TreeBuilder` produces already binds `xml` implicitly.
            continue;
        }
        builder.declare_namespace(prefix.as_deref(), &namespace.string_value())?;
    }
    for attribute in node.attributes() {
        let name = attribute
            .node_name()
            .unwrap_or_else(|| ExpandedName::new(None::<&str>, "attribute"));
        // A copy is the same attribute in a new tree, so the properties that make it an ID or a
        // reference to one travel with it: XSLT 3.0 §19.5.1 keeps `[is-id]` and `[is-idref]` on a
        // copied node, and without them `fn:id` over a copied subtree would answer differently
        // from `fn:id` over the original.
        builder.annotated_attribute(
            name,
            attribute.string_value(),
            Annotation {
                is_id: attribute.is_id(),
                is_idref: attribute.is_idref(),
                ..Annotation::default()
            },
        )?;
    }
    Ok(())
}

fn pi_target(node: &NodeRef) -> String {
    node.node_name()
        .map(|name| name.local_name().to_owned())
        .unwrap_or_default()
}

/// Materializes `builder` and picks out the node that corresponds to what was copied: the
/// document node itself for a document, the document element for an element, or the sole
/// top-level child for anything else (text, comment or processing instruction).
fn extract_copy(builder: TreeBuilder, kind: NodeKind) -> Result<NodeRef, XdmError> {
    let tree = builder.build()?;
    match kind {
        NodeKind::Document => Ok(tree.root()),
        NodeKind::Element => tree.document_element().ok_or_else(|| {
            XdmError::new(
                oxixml_infoset::ErrorCode::Fodc0006,
                "the copy has no document element",
            )
        }),
        // Only `copy_of` materializes through here, and it rejects an attribute, a namespace and
        // a text node before ever building — so in practice only `Comment` and
        // `ProcessingInstruction` reach this arm; it is written to degrade to the same "no node
        // was written" error rather than panic if that ever stops being true.
        NodeKind::Attribute
        | NodeKind::Namespace
        | NodeKind::Text
        | NodeKind::Comment
        | NodeKind::ProcessingInstruction => {
            tree.root().children().into_iter().next().ok_or_else(|| {
                XdmError::new(
                    oxixml_infoset::ErrorCode::Fodc0006,
                    "the copy produced no node",
                )
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::Tree;

    #[test]
    fn naming_functions() -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<p:a xmlns:p='urn:p'><b c='1'/></p:a>")?;
        let root = tree.document_element().ok_or("no root")?;
        assert_eq!(name(&root), "p:a");
        assert_eq!(local_name(&root), "a");
        assert_eq!(namespace_uri(&root), "urn:p");

        let b = root.children().first().cloned().ok_or("no child")?;
        assert_eq!(name(&b), "b");
        assert_eq!(namespace_uri(&b), "");

        assert_eq!(name(&tree.root()), "");
        assert_eq!(local_name(&tree.root()), "");
        assert_eq!(namespace_uri(&tree.root()), "");
        Ok(())
    }

    /// `fn:name` is `string(node-name($arg))`, so the prefix the *name itself* carries wins over
    /// anything in scope: an attribute named `QName('http://www.example.com/', 'sgml:foo')` is
    /// `sgml:foo` even inside an element that binds `sgml` to some other namespace, and even
    /// with no element to look anything up in (QT3 `cbcl-constr-compattr-007`).
    #[test]
    fn name_prefers_the_names_own_prefix() -> Result<(), Box<dyn std::error::Error>> {
        let mut builder = TreeBuilder::new();
        builder.start_element(ExpandedName::new(None::<&str>, "element"))?;
        builder.declare_namespace(Some("sgml"), "http://www.example.com/other")?;
        builder.attribute(
            ExpandedName::new(Some("http://www.example.com/"), "foo").with_prefix("sgml"),
            "",
        )?;
        builder.end_element()?;
        let tree = builder.build()?;
        let element = tree.document_element().ok_or("no document element")?;
        let attribute = element
            .attributes()
            .first()
            .cloned()
            .ok_or("no attribute")?;
        assert_eq!(name(&attribute), "sgml:foo");
        assert_eq!(namespace_uri(&attribute), "http://www.example.com/");

        // A name that retained no prefix still falls back to the element's in-scope bindings.
        let mut builder = TreeBuilder::new();
        builder.start_element(ExpandedName::new(None::<&str>, "element"))?;
        builder.declare_namespace(Some("sgml"), "http://www.example.com/other")?;
        builder.attribute(
            ExpandedName::new(Some("http://www.example.com/other"), "bar"),
            "",
        )?;
        builder.end_element()?;
        let tree = builder.build()?;
        let element = tree.document_element().ok_or("no document element")?;
        let attribute = element
            .attributes()
            .first()
            .cloned()
            .ok_or("no attribute")?;
        assert_eq!(name(&attribute), "sgml:bar");
        Ok(())
    }

    /// F&O 3.1 §14.7.1 has no name to put in a `namespace::` step for the node that binds the
    /// default namespace, and gives it a predicate instead (QT3 `path013`).
    #[test]
    fn path_of_an_unnamed_namespace_node_uses_the_predicate_form()
    -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<test-set xmlns='urn:example' xmlns:p='urn:p'/>")?;
        let element = tree.document_element().ok_or("no document element")?;
        let namespaces = element.namespaces();
        let default = namespaces
            .iter()
            .find(|node| node.node_name().is_none())
            .cloned()
            .ok_or("no default-namespace node")?;
        assert_eq!(
            path(&default),
            concat!(
                "/Q{urn:example}test-set[1]/namespace::*",
                "[Q{http://www.w3.org/2005/xpath-functions}local-name()=\"\"]"
            )
        );

        // A named binding still gets the plain name test.
        let prefixed = namespaces
            .iter()
            .find(|node| {
                node.node_name()
                    .is_some_and(|name| name.local_name() == "p")
            })
            .cloned()
            .ok_or("no `p` namespace node")?;
        assert_eq!(path(&prefixed), "/Q{urn:example}test-set[1]/namespace::p");
        Ok(())
    }

    #[test]
    fn root_and_has_children() -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<a><b/></a>")?;
        let a = tree.document_element().ok_or("no root")?;
        let b = a.children().first().cloned().ok_or("no child")?;
        assert_eq!(root(&b), tree.root());
        assert!(has_children(&a));
        assert!(!has_children(&b));
        Ok(())
    }

    #[test]
    fn lang_examples_from_the_specification() -> Result<(), Box<dyn std::error::Error>> {
        let tree =
            Tree::parse_str("<a xml:lang='en'><b/><c xml:lang='en-GB'/><d xml:lang='fr'/></a>")?;
        let a = tree.document_element().ok_or("no root")?;
        let children = a.children();
        assert!(lang("en", &a));
        assert!(lang("EN", &a));
        assert!(lang("en", &children[0]));
        assert!(lang("en", &children[1]));
        assert!(!lang("en-US", &children[1]));
        assert!(!lang("en", &children[2]));

        let plain = Tree::parse_str("<a/>")?;
        assert!(!lang("en", &plain.document_element().ok_or("no root")?));
        Ok(())
    }

    #[test]
    fn generate_id_is_stable_and_unique() -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<a><b/><c/></a>")?;
        let a = tree.document_element().ok_or("no root")?;
        let children = a.children();
        assert_eq!(generate_id(&a), generate_id(&a));
        assert_ne!(generate_id(&children[0]), generate_id(&children[1]));

        let other = Tree::parse_str("<a><b/><c/></a>")?;
        assert_ne!(
            generate_id(&a),
            generate_id(&other.document_element().ok_or("no root")?)
        );
        Ok(())
    }

    #[test]
    fn id_finds_elements_by_xml_id() -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<a><b xml:id='x'/><c xml:id='y'/></a>")?;
        let root = tree.document_element().ok_or("no root")?;
        let found = id(&["x y".to_owned()], &root)?;
        assert_eq!(found.len(), 2);
        assert_eq!(local_name(&found[0]), "b");
        assert_eq!(local_name(&found[1]), "c");
        assert!(id(&["z".to_owned()], &root)?.is_empty());
        assert_eq!(element_with_id(&["x".to_owned()], &root)?.len(), 1);
        Ok(())
    }

    #[test]
    fn id_tokens_split_on_whitespace() {
        assert_eq!(
            id_tokens(&["a  b".to_owned(), "\tc\n".to_owned()]),
            vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]
        );
        // A separator at either end contributes no token, and an argument that is nothing but
        // separators contributes none at all.
        assert_eq!(id_tokens(&[" x ".to_owned()]), vec!["x".to_owned()]);
        assert!(id_tokens(&[" \r\n\t".to_owned(), String::new()]).is_empty());
        // Only XML's four separators split: every other white-space character stays in its token.
        assert_eq!(
            id_tokens(&["a\u{a0}b".to_owned()]),
            vec!["a\u{a0}b".to_owned()]
        );
    }

    /// A leading space in the argument and a trailing one in the `xml:id` are both separators,
    /// so neither can defeat the lookup: F&O 3.1 §14.5.1 tokenizes the argument, and
    /// [xml:id 1.0 §4](https://www.w3.org/TR/xml-id/#processing) normalizes the attribute value
    /// as if it had been declared `ID`.
    #[test]
    fn id_ignores_space_around_the_token_and_around_the_identifier()
    -> Result<(), Box<dyn std::error::Error>> {
        let tree =
            Tree::parse_str("<doc><div xml:id='id3 '><title>Expressions</title></div></doc>")?;
        let root = tree.document_element().ok_or("no root")?;
        let found = id(&[" id3".to_owned()], &root)?;
        assert_eq!(found.len(), 1);
        assert_eq!(local_name(&found[0]), "div");
        Ok(())
    }

    /// An attribute an `<!ATTLIST …>` declares `ID` is one, which is what makes `fn:id` answer
    /// anything at all for a document that does not use `xml:id`.
    #[test]
    fn id_finds_elements_by_a_dtd_declared_id() -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str(
            "<!DOCTYPE doc [<!ATTLIST b code ID #IMPLIED>]>\
             <doc><b code='one'/><b code='two'/><c code='one'/></doc>",
        )?;
        let root = tree.document_element().ok_or("no root")?;
        let found = id(&["one".to_owned()], &root)?;
        assert_eq!(found.len(), 1);
        assert_eq!(local_name(&found[0]), "b");
        // `<c code='one'>` is not selected: the declaration named `b`, so `c`'s attribute is
        // ordinary `CDATA`.
        assert!(id(&["two".to_owned()], &root)?.len() == 1);
        Ok(())
    }

    /// The mirror image: an `IDREF`/`IDREFS` attribute is what `fn:idref` searches.
    #[test]
    fn idref_finds_nodes_by_a_dtd_declared_reference() -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str(
            "<!DOCTYPE doc [<!ATTLIST b to IDREFS #IMPLIED><!ATTLIST c to CDATA #IMPLIED>]>\
             <doc><b to='one two'/><c to='one'/></doc>",
        )?;
        let root = tree.document_element().ok_or("no root")?;
        let found = idref(&["two".to_owned()], &root)?;
        assert_eq!(found.len(), 1);
        assert_eq!(
            found[0]
                .node_name()
                .map(|name| name.local_name().to_owned()),
            Some("to".to_owned())
        );
        assert!(idref(&["three".to_owned()], &root)?.is_empty());
        Ok(())
    }

    /// The document the XSLT 3.0 suite's `id-04x` cases are written against: four `a` elements
    /// cross-referring through one `IDREFS` attribute.
    fn cross_referring_document() -> Result<oxixml_infoset::Tree, XdmError> {
        Tree::parse_str(
            "<!DOCTYPE t04 [<!ELEMENT t04 (a*)><!ELEMENT a EMPTY>\
             <!ATTLIST a id ID #REQUIRED><!ATTLIST a ref IDREFS #IMPLIED>]>\
             <t04><a id='a'/><a id='b' ref='c d'/><a id='c' ref='d a'/><a id='d' ref='a b'/></t04>",
        )
    }

    /// The `id` of every element holding a reference the call matched, which is the shape the
    /// suite's cases report (`for $n in idref(…) return $n/../@id`).
    fn referring_ids(found: &[NodeRef]) -> Vec<String> {
        found
            .iter()
            .filter_map(|node| node.parent())
            .flat_map(|element| element.attributes())
            .filter(|attribute| attribute.is_id())
            .map(|attribute| attribute.string_value())
            .collect()
    }

    /// `fn:idref` does not tokenize its argument the way `fn:id` does: each supplied string is
    /// one candidate IDREF value, and one holding whitespace is not a legal IDREF at all, so it
    /// can never match (`id-041`, whose own description states the rule).
    #[test]
    fn idref_does_not_tokenize_its_argument() -> Result<(), Box<dyn std::error::Error>> {
        let tree = cross_referring_document()?;
        let root = tree.document_element().ok_or("no root")?;
        for id in ["a", "b", "c", "d"] {
            let candidate = format!("a {id}");
            assert!(
                idref(std::slice::from_ref(&candidate), &root)?.is_empty(),
                "`{candidate}` is one candidate value, not two tokens"
            );
        }
        // `fn:id`, by contrast, keeps tokenizing: the same string finds both elements.
        assert_eq!(id(&["a b".to_owned()], &root)?.len(), 2);
        Ok(())
    }

    /// Each string of the argument sequence is its own candidate, and all of them are searched
    /// for (`id-042`: `idref(('a', @id))` finds every reference to either).
    #[test]
    fn idref_searches_for_every_supplied_candidate() -> Result<(), Box<dyn std::error::Error>> {
        let tree = cross_referring_document()?;
        let root = tree.document_element().ok_or("no root")?;
        assert_eq!(
            referring_ids(&idref(&["a".to_owned(), "a".to_owned()], &root)?),
            vec!["c".to_owned(), "d".to_owned()]
        );
        assert_eq!(
            referring_ids(&idref(&["a".to_owned(), "c".to_owned()], &root)?),
            vec!["b".to_owned(), "c".to_owned(), "d".to_owned()]
        );
        Ok(())
    }

    /// An `IDREFS` value is a list separated by XML's four whitespace characters and no others,
    /// so a reference holding a no-break space — a `NameChar` — stays one reference. The mirror
    /// of the rule `id_tokens` states for the argument side.
    #[test]
    fn an_idrefs_value_is_split_on_xml_whitespace_only() -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str(
            "<!DOCTYPE doc [<!ATTLIST b to IDREFS #IMPLIED>]>\
             <doc><b to='one\u{a0}two'/></doc>",
        )?;
        let root = tree.document_element().ok_or("no root")?;
        assert_eq!(idref(&["one\u{a0}two".to_owned()], &root)?.len(), 1);
        assert!(idref(&["one".to_owned()], &root)?.is_empty());
        assert!(idref(&["two".to_owned()], &root)?.is_empty());
        Ok(())
    }

    /// The second argument names the document to search, which need not be the one the caller is
    /// standing in (`id-043`: `idref(('a', .), doc('id302.xml'))`).
    #[test]
    fn idref_searches_the_tree_of_the_node_it_is_given() -> Result<(), Box<dyn std::error::Error>> {
        let elsewhere = cross_referring_document()?;
        let here = Tree::parse_str("<root/>")?;
        let root = here.document_element().ok_or("no root")?;
        assert!(idref(&["a".to_owned()], &root)?.is_empty());
        assert_eq!(
            referring_ids(&idref(&["a".to_owned()], &elsewhere.root())?),
            vec!["c".to_owned(), "d".to_owned()]
        );
        Ok(())
    }

    #[test]
    fn path_names_a_node_uniquely() -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<a xmlns:p='urn:p'><b/><b x='1'/><!-- c -->text</a>")?;
        assert_eq!(path(&tree.root()), "/");

        let a = tree.document_element().ok_or("no root")?;
        assert_eq!(path(&a), "/Q{}a[1]");

        let children = a.children();
        assert_eq!(path(&children[0]), "/Q{}a[1]/Q{}b[1]");
        assert_eq!(path(&children[1]), "/Q{}a[1]/Q{}b[2]");
        assert_eq!(path(&children[2]), "/Q{}a[1]/comment()[1]");
        assert_eq!(path(&children[3]), "/Q{}a[1]/text()[1]");
        assert_eq!(path(&children[1].attributes()[0]), "/Q{}a[1]/Q{}b[2]/@x");
        Ok(())
    }

    #[test]
    fn innermost_and_outermost() -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<a><b><c/></b></a>")?;
        let a = tree.document_element().ok_or("no root")?;
        let b = a.children().first().cloned().ok_or("no child")?;
        let c = b.children().first().cloned().ok_or("no grandchild")?;
        let all = [a.clone(), b.clone(), c.clone()];

        assert_eq!(innermost(&all), vec![c.clone()]);
        assert_eq!(outermost(&all), vec![a.clone()]);
        assert_eq!(innermost(&[b.clone(), c.clone()]), vec![c]);
        assert_eq!(outermost(&[a.clone(), b]), vec![a]);
        assert!(innermost(&[]).is_empty());
        Ok(())
    }

    #[test]
    fn idref_needs_annotated_nodes() -> Result<(), Box<dyn std::error::Error>> {
        // Without a DTD or schema no attribute is an IDREF, so the result is empty rather than
        // an error.
        let tree = Tree::parse_str("<a><b ref='x'/></a>")?;
        let root = tree.document_element().ok_or("no root")?;
        assert!(idref(&["x".to_owned()], &root)?.is_empty());
        Ok(())
    }

    #[test]
    fn copy_of_a_document_copies_the_whole_tree() -> Result<(), Box<dyn std::error::Error>> {
        // unparsed-entity-uri-07/08 (shape): `copy-of(.)` at the top level of a `match="/"`
        // template, where `.` is the document node.
        let tree = Tree::parse_str("<!--top--><a x='1'><b/>text</a>")?;
        let copy = copy_of(&tree.root())?;
        assert_eq!(copy.node_kind(), NodeKind::Document);
        assert_ne!(copy, tree.root());
        let copied_a = copy
            .children()
            .into_iter()
            .find(|child| child.node_kind() == NodeKind::Element)
            .ok_or("copy kept the document element")?;
        assert_eq!(local_name(&copied_a), "a");
        assert_eq!(copied_a.attributes()[0].string_value(), "1");
        assert_eq!(copied_a.children().len(), 2, "the element and the text");
        assert_eq!(copy.children().len(), 2, "the comment and the element");
        Ok(())
    }

    #[test]
    fn copy_of_comments_and_processing_instructions() -> Result<(), Box<dyn std::error::Error>> {
        // Unlike a text node (below), a comment or a processing instruction is a legal direct
        // child of a document node, so `copy_of` can place its copy there with no wrapping
        // element — no ancestor is needed either way.
        let tree = Tree::parse_str("<a><!--c--><?pi data?></a>")?;
        let a = tree.document_element().ok_or("no root")?;
        let children = a.children();

        let comment = copy_of(&children[0])?;
        assert_eq!(comment.node_kind(), NodeKind::Comment);
        assert_eq!(comment.string_value(), "c");

        let pi = copy_of(&children[1])?;
        assert_eq!(pi.node_kind(), NodeKind::ProcessingInstruction);
        assert_eq!(local_name(&pi), "pi");
        assert_eq!(pi.string_value(), "data");
        Ok(())
    }

    #[test]
    fn copy_of_a_text_node_is_unsupported_even_though_it_has_a_parent()
    -> Result<(), Box<dyn std::error::Error>> {
        // `copy_of` never carries the original parent over, and XDM does not allow a text node
        // to be a direct child of a document node — so even a text node that has a parent in
        // the source tree cannot be placed anywhere in `copy_of`'s own (parent-less) output.
        let tree = Tree::parse_str("<a>text</a>")?;
        let a = tree.document_element().ok_or("no root")?;
        let text = a.children()[0].clone();
        assert!(text.parent().is_some());
        let error = copy_of(&text).expect_err("a bare copy of a text node is unsupported");
        assert!(error.is(FnErrorCode::Fodc0006));
        Ok(())
    }

    #[test]
    fn snapshot_of_a_text_node_keeps_it_under_its_copied_parent()
    -> Result<(), Box<dyn std::error::Error>> {
        // Unlike `copy_of`, `snapshot` carries the ancestor over, so a text node *with* a
        // parent copies successfully — nested under the copy of that parent.
        let tree = Tree::parse_str("<a x='1'>text</a>")?;
        let a = tree.document_element().ok_or("no root")?;
        let text = a.children()[0].clone();

        let copy = snapshot(&text)?;
        assert_eq!(copy.node_kind(), NodeKind::Text);
        assert_eq!(copy.string_value(), "text");
        let parent = copy.parent().ok_or("the text's ancestor was kept")?;
        assert_eq!(local_name(&parent), "a");
        assert_eq!(parent.attributes()[0].string_value(), "1");
        Ok(())
    }

    #[test]
    fn copy_of_carries_inherited_namespaces_but_not_the_ancestor()
    -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<r xmlns:p='urn:p'><a><p:b/></a></r>")?;
        let root = tree.document_element().ok_or("no root")?;
        let a = root.children()[0].clone();
        let copy = copy_of(&a)?;
        assert_eq!(copy.children().len(), 1);
        // `p` was declared on `r`, an ancestor `copy_of` does not carry over — but the binding
        // must still resolve on the copy, since a namespace node reports the full in-scope set.
        assert_eq!(name(&copy.children()[0]), "p:b");
        assert_eq!(namespace_uri(&copy.children()[0]), "urn:p");
        Ok(())
    }

    #[test]
    fn copy_of_rejects_a_standalone_attribute_or_namespace()
    -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<a x='1'/>")?;
        let root = tree.document_element().ok_or("no root")?;
        let attribute = root.attributes()[0].clone();
        let error = copy_of(&attribute).expect_err("a standalone attribute is not copyable");
        assert!(error.is(FnErrorCode::Fodc0006));
        Ok(())
    }

    #[test]
    fn snapshot_of_the_document_node_matches_copy_of() -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<a><b/></a>")?;
        let copy = snapshot(&tree.root())?;
        assert_eq!(copy.node_kind(), NodeKind::Document);
        let document_element = copy
            .children()
            .into_iter()
            .find(|child| child.node_kind() == NodeKind::Element)
            .ok_or("no root")?;
        assert_eq!(local_name(&document_element), "a");
        Ok(())
    }

    #[test]
    fn snapshot_keeps_the_ancestor_path_and_its_attributes_but_not_its_siblings()
    -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<a x='1'><b/><c><d/><e/></c></a>")?;
        let root = tree.document_element().ok_or("no root")?;
        let c = root.children()[1].clone();
        let d = c.children()[0].clone();

        let copy = snapshot(&d)?;
        assert_eq!(copy.node_kind(), NodeKind::Element);
        assert_eq!(local_name(&copy), "d");
        assert_ne!(copy, d, "a fresh identity, same as copy_of");

        let copied_c = copy.parent().ok_or("d's ancestor c was kept")?;
        assert_eq!(local_name(&copied_c), "c");
        assert_eq!(
            copied_c.children().len(),
            1,
            "e, c's other child, is excluded"
        );

        let copied_a = copied_c.parent().ok_or("c's ancestor a was kept")?;
        assert_eq!(local_name(&copied_a), "a");
        assert_eq!(
            copied_a.attributes()[0].string_value(),
            "1",
            "a's attributes are kept"
        );
        assert_eq!(
            copied_a.children().len(),
            1,
            "b, a's other child, is excluded"
        );

        assert_eq!(
            copy.root(),
            copied_a.root(),
            "root() reaches the same copied document"
        );
        assert_eq!(copy.root().node_kind(), NodeKind::Document);
        Ok(())
    }

    #[test]
    fn snapshot_of_a_root_element_needs_no_ancestor_wrapping()
    -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<a x='1'><b/></a>")?;
        let a = tree.document_element().ok_or("no root")?;
        let copy = snapshot(&a)?;
        assert_eq!(local_name(&copy), "a");
        assert_eq!(copy.attributes()[0].string_value(), "1");
        assert_eq!(copy.children().len(), 1);
        let wrapping_document = copy.parent().ok_or("the implicit document root")?;
        assert_eq!(wrapping_document.node_kind(), NodeKind::Document);
        Ok(())
    }

    /// An attribute node has ancestors, so `fn:snapshot` of one succeeds and answers the copy of
    /// the attribute itself — attached to the copy of the element that carries it, which keeps
    /// its own other attributes and its in-scope namespaces (xslt30 `snapshot-0101e`, which reads
    /// `snapshot(//@srsName)/parent::Envelope` back out of the result).
    #[test]
    fn snapshot_of_an_attribute_keeps_it_on_its_copied_parent()
    -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<r xmlns:p='urn:p'><e x='1' y='2'><f/></e></r>")?;
        let root = tree.document_element().ok_or("no root")?;
        let e = root.children()[0].clone();
        let x = e.attributes()[0].clone();

        let copy = snapshot(&x)?;
        assert_eq!(copy.node_kind(), NodeKind::Attribute);
        assert_eq!(local_name(&copy), "x");
        assert_eq!(copy.string_value(), "1");
        assert_ne!(copy, x, "a fresh identity");

        let parent = copy.parent().ok_or("the attribute's element was kept")?;
        assert_eq!(local_name(&parent), "e");
        assert_eq!(
            parent.attributes().len(),
            2,
            "an ancestor keeps all of its attributes"
        );
        assert!(
            parent.children().is_empty(),
            "but none of its other descendants"
        );
        let grandparent = parent.parent().ok_or("the whole ancestor path was kept")?;
        assert_eq!(local_name(&grandparent), "r");
        assert_eq!(copy.root().node_kind(), NodeKind::Document);
        Ok(())
    }

    /// The same of a namespace node, whose parent is the element that binds the prefix (xslt30
    /// `snapshot-0101f`). The `xml` binding every tree has implicitly is included, so the lookup
    /// has to find a binding no ancestor declared explicitly.
    #[test]
    fn snapshot_of_a_namespace_node_keeps_it_on_its_copied_parent()
    -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<r xmlns:p='urn:p' xmlns='urn:d'><e/></r>")?;
        let root = tree.document_element().ok_or("no root")?;
        for (prefix, uri) in [
            (Some("p"), "urn:p"),
            (None, "urn:d"),
            (Some("xml"), "http://www.w3.org/XML/1998/namespace"),
        ] {
            let binding = root
                .namespaces()
                .into_iter()
                .find(|binding| {
                    binding.node_name().map(|name| name.local_name().to_owned())
                        == prefix.map(str::to_owned)
                })
                .ok_or("the binding is in scope on the original")?;
            let copy = snapshot(&binding)?;
            assert_eq!(copy.node_kind(), NodeKind::Namespace);
            assert_eq!(copy.string_value(), uri);
            let parent = copy.parent().ok_or("the binding's element was kept")?;
            assert_eq!(local_name(&parent), "r");
            assert!(parent.children().is_empty(), "and none of its descendants");
        }
        Ok(())
    }

    /// A namespace node reports the bindings in scope on its element, not the ones that element
    /// declared, so the snapshot of one has to come back with the binding that is in scope *there*
    /// — the inner one where a prefix is rebound, and the inherited one where it is not.
    #[test]
    fn the_snapshot_of_a_rebound_prefix_keeps_the_inner_binding()
    -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<r xmlns:p='urn:a' xmlns:q='urn:q'><e xmlns:p='urn:b'/></r>")?;
        let root = tree.document_element().ok_or("no root")?;
        let e = root.children()[0].clone();
        for (prefix, uri) in [("p", "urn:b"), ("q", "urn:q")] {
            let binding = e
                .namespaces()
                .into_iter()
                .find(|binding| {
                    binding.node_name().map(|name| name.local_name().to_owned())
                        == Some(prefix.to_owned())
                })
                .ok_or("the binding is in scope on the original")?;
            assert_eq!(binding.string_value(), uri, "on the original");

            let copy = snapshot(&binding)?;
            assert_eq!(copy.node_kind(), NodeKind::Namespace);
            assert_eq!(copy.string_value(), uri, "on the snapshot");
            let parent = copy.parent().ok_or("the binding's element was kept")?;
            assert_eq!(local_name(&parent), "e");
            // The outer declaration is still there, shadowed, on the copied ancestor.
            let outer = parent.parent().ok_or("e's ancestor r was kept")?;
            assert_eq!(local_name(&outer), "r");
            assert!(
                outer.namespaces().into_iter().any(|binding| {
                    binding.node_name().map(|name| name.local_name().to_owned())
                        == Some("p".to_owned())
                        && binding.string_value() == "urn:a"
                }),
                "r's own binding of `p` is unchanged by the copy"
            );
        }
        Ok(())
    }

    /// `fn:copy-of` is unchanged by any of that: it never carries the parent over, so a copy of
    /// an attribute or a namespace node would have nowhere to live whether or not the original
    /// has one.
    #[test]
    fn copy_of_still_rejects_an_attribute_that_has_a_parent()
    -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<r xmlns:p='urn:p'><e x='1'/></r>")?;
        let root = tree.document_element().ok_or("no root")?;
        let e = root.children()[0].clone();
        for node in [e.attributes()[0].clone(), e.namespaces()[0].clone()] {
            assert!(node.parent().is_some());
            let error = copy_of(&node).expect_err("copy_of has no parent to attach the copy to");
            assert!(error.is(FnErrorCode::Fodc0006));
        }
        Ok(())
    }

    /// The XML the unparsed-entity cases are written against: two unparsed entities, one with a
    /// public identifier and one without, each naming a notation.
    const WITH_UNPARSED_ENTITIES: &str = r#"<!DOCTYPE r [
        <!NOTATION gif SYSTEM "image/gif">
        <!ENTITY hatch-pic SYSTEM "grafix/OpenHatch.gif" NDATA gif>
        <!ENTITY watch-pic PUBLIC "-//Textuality//TEXT boilerplate//EN"
                                  "grafix/OpenHatch.gif" NDATA gif>
      ]><r><a/></r>"#;

    fn with_unparsed_entities() -> Result<Tree, Box<dyn std::error::Error>> {
        Ok(Tree::parse_str_with(
            WITH_UNPARSED_ENTITIES,
            oxixml_infoset::ParseOptions::default().document_uri("http://example.com/fn/doc.xml"),
        )?)
    }

    /// Both properties, as `fn:unparsed-entity-uri` and `fn:unparsed-entity-public-id` read
    /// them, plus the notations — asserted the same way for the original and for a copy.
    fn declarations(node: &NodeRef) -> (Option<String>, Option<String>, Vec<String>) {
        let hatch = node
            .unparsed_entity("hatch-pic")
            .and_then(|entity| entity.uri);
        let watch = node
            .unparsed_entity("watch-pic")
            .and_then(|entity| entity.public_id);
        (hatch, watch, node.notation_names())
    }

    /// xslt30 `unparsed-entity-07`: `fn:copy-of` of a document node keeps `[unparsed entities]`
    /// and `[notations]`, with the system identifiers still resolved — the copy has no document
    /// URI of its own to resolve them against a second time.
    #[test]
    fn copy_of_a_document_carries_its_unparsed_entities_and_notations()
    -> Result<(), Box<dyn std::error::Error>> {
        let tree = with_unparsed_entities()?;
        let expected = (
            Some("http://example.com/fn/grafix/OpenHatch.gif".to_owned()),
            Some("-//Textuality//TEXT boilerplate//EN".to_owned()),
            vec!["gif".to_owned()],
        );
        assert_eq!(declarations(&tree.root()), expected, "the original");

        let copy = copy_of(&tree.root())?;
        assert_eq!(declarations(&copy), expected, "the copy");
        Ok(())
    }

    /// xslt30 `unparsed-entity-08`: the same of `fn:snapshot`, whose copy keeps the path back to
    /// the root — from the document node itself and from a node deep inside it.
    #[test]
    fn snapshot_carries_the_unparsed_entities_of_the_root_it_keeps()
    -> Result<(), Box<dyn std::error::Error>> {
        let tree = with_unparsed_entities()?;
        let expected = declarations(&tree.root());

        assert_eq!(declarations(&snapshot(&tree.root())?), expected);

        let a = tree
            .document_element()
            .ok_or("no document element")?
            .children()
            .first()
            .cloned()
            .ok_or("no child")?;
        assert_eq!(declarations(&snapshot(&a)?.root()), expected);
        Ok(())
    }

    /// A copy of an *element* is not a copy of the document, so it declares nothing: XDM makes
    /// both properties properties of the document node.
    #[test]
    fn copy_of_an_element_carries_no_declarations() -> Result<(), Box<dyn std::error::Error>> {
        let tree = with_unparsed_entities()?;
        let copy = copy_of(&tree.document_element().ok_or("no document element")?)?;
        assert_eq!(
            declarations(&copy.root()),
            (None, None, Vec::new()),
            "the wrapping document node stands for no document"
        );
        Ok(())
    }

    /// XSLT 3.0 §19.5.1 keeps `[is-id]` and `[is-idref]` on a copied node, so `fn:id` answers the
    /// same over a copy as over the original — including for the DTD-declared attributes that
    /// only the document type declaration of the *source* made IDs.
    #[test]
    fn a_copy_keeps_the_id_properties_of_its_attributes() -> Result<(), Box<dyn std::error::Error>>
    {
        let tree = Tree::parse_str(
            "<!DOCTYPE doc [<!ATTLIST b code ID #IMPLIED to IDREF #IMPLIED>]>\
             <doc><b code='one' to='two' plain='three'/></doc>",
        )?;
        let root = tree.document_element().ok_or("no root")?;
        for copy in [copy_of(&root)?, snapshot(&root)?] {
            let child = copy.children().first().cloned().ok_or("no child")?;
            let properties: Vec<(String, bool, bool)> = child
                .attributes()
                .iter()
                .map(|attribute| {
                    (
                        attribute.string_value(),
                        attribute.is_id(),
                        attribute.is_idref(),
                    )
                })
                .collect();
            assert_eq!(
                properties,
                [
                    ("one".to_owned(), true, false),
                    ("two".to_owned(), false, true),
                    ("three".to_owned(), false, false),
                ]
            );
            assert_eq!(id(&["one".to_owned()], &child)?.len(), 1);
        }
        Ok(())
    }

    /// `Q{…}root()` denotes the root of the tree, so the root contributes no step of its own.
    #[test]
    fn a_path_is_relative_to_the_root_node() -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<a b='c'><d/><d/></a>")?;
        // Rooted at a document node, the document element is still a step.
        let root = tree.document_element().ok_or("no document element")?;
        assert_eq!(path(&root), "/Q{}a[1]");
        let second = root.children().get(1).cloned().ok_or("no second child")?;
        assert_eq!(path(&second), "/Q{}a[1]/Q{}d[2]");
        assert_eq!(path(&tree.root()), "/");
        Ok(())
    }
}
