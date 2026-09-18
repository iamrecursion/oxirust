//! Constructed nodes: a pluggable [`XdmStore`] for the trees a node constructor builds.
//!
//! A constructed node is *not* a document. `<a/>` is a parentless element, `attribute a {1}` is a
//! parentless attribute, and `text {"x"}` is a parentless text node — none of which
//! [`oxixml_infoset::TreeBuilder`] can produce, because it builds documents. This module
//! therefore has its own arena, laid out in document order (an element, then its namespace nodes,
//! then its attribute nodes, then its children, recursively), and implements the store trait
//! directly.
//!
//! It also implements the *copy* half of construction: XQuery
//! [§3.9.1.3](https://www.w3.org/TR/xquery-31/#id-content) says every node placed in a
//! constructor's content is copied, with a new identity, and that the copy-namespaces mode
//! decides which namespace bindings the copy keeps and which it gains.

use crate::context::{ConstructionMode, CopyNamespaces};
use oxixml_infoset::{
    AtomicValue, DocumentOrderKey, ExpandedName, NodeId, NodeKind, NodeRef, Sequence, XdmNode,
    XdmStore, next_tree_id,
};
use oxixml_qname::XML_NAMESPACE;
use std::sync::Arc;

/// One node of a constructed tree, before it is laid out.
#[derive(Debug, Clone)]
pub(crate) struct Draft {
    pub(crate) kind: NodeKind,
    pub(crate) name: Option<ExpandedName>,
    pub(crate) value: String,
    /// `(prefix, namespace)`; `None` is the default namespace.
    pub(crate) namespaces: Vec<(Option<String>, String)>,
    pub(crate) attributes: Vec<Draft>,
    pub(crate) children: Vec<Draft>,
    pub(crate) base_uri: Option<String>,
    pub(crate) type_name: Option<ExpandedName>,
    pub(crate) typed_value: Option<Sequence>,
    pub(crate) is_id: bool,
    pub(crate) is_idref: bool,
}

impl Draft {
    pub(crate) fn new(kind: NodeKind) -> Self {
        Self {
            kind,
            name: None,
            value: String::new(),
            namespaces: Vec::new(),
            attributes: Vec::new(),
            children: Vec::new(),
            base_uri: None,
            type_name: None,
            typed_value: None,
            is_id: false,
            is_idref: false,
        }
    }

    pub(crate) fn element(name: ExpandedName) -> Self {
        let mut draft = Self::new(NodeKind::Element);
        draft.name = Some(name);
        draft
    }

    pub(crate) fn attribute(name: ExpandedName, value: impl Into<String>) -> Self {
        let mut draft = Self::new(NodeKind::Attribute);
        let value = value.into();
        draft.is_id = name.namespace() == Some(XML_NAMESPACE) && name.local_name() == "id";
        draft.name = Some(name);
        draft.value = value;
        draft
    }

    pub(crate) fn text(value: impl Into<String>) -> Self {
        let mut draft = Self::new(NodeKind::Text);
        draft.value = value.into();
        draft
    }

    pub(crate) fn comment(value: impl Into<String>) -> Self {
        let mut draft = Self::new(NodeKind::Comment);
        draft.value = value.into();
        draft
    }

    pub(crate) fn processing_instruction(target: &str, value: impl Into<String>) -> Self {
        let mut draft = Self::new(NodeKind::ProcessingInstruction);
        draft.name = Some(ExpandedName::new(None::<&str>, target));
        draft.value = value.into();
        draft
    }

    pub(crate) fn namespace(prefix: Option<&str>, uri: impl Into<String>) -> Self {
        let mut draft = Self::new(NodeKind::Namespace);
        draft.name = prefix.map(|prefix| ExpandedName::new(None::<&str>, prefix));
        draft.value = uri.into();
        draft
    }

    pub(crate) fn document() -> Self {
        Self::new(NodeKind::Document)
    }

    /// Materializes this draft as a standalone tree and returns a handle on its root.
    pub(crate) fn build(self) -> NodeRef {
        let mut nodes = Vec::new();
        flatten(self, None, &mut nodes);
        let store = ConstructedStore {
            tree_id: next_tree_id(),
            nodes,
        };
        NodeRef::new(Arc::new(store), NodeId(0))
    }
}

fn flatten(draft: Draft, parent: Option<usize>, out: &mut Vec<RawNode>) -> usize {
    let index = out.len();
    out.push(RawNode {
        kind: draft.kind,
        name: draft.name,
        value: draft.value,
        parent,
        children: Vec::new(),
        attributes: Vec::new(),
        namespaces: Vec::new(),
        base_uri: draft.base_uri,
        type_name: draft.type_name,
        typed_value: draft.typed_value,
        is_id: draft.is_id,
        is_idref: draft.is_idref,
    });
    let mut namespaces = Vec::with_capacity(draft.namespaces.len());
    for (prefix, uri) in draft.namespaces {
        namespaces.push(flatten(
            Draft::namespace(prefix.as_deref(), uri),
            Some(index),
            out,
        ));
    }
    let mut attributes = Vec::with_capacity(draft.attributes.len());
    for attribute in draft.attributes {
        attributes.push(flatten(attribute, Some(index), out));
    }
    let mut children = Vec::with_capacity(draft.children.len());
    for child in draft.children {
        children.push(flatten(child, Some(index), out));
    }
    if let Some(node) = out.get_mut(index) {
        node.namespaces = namespaces;
        node.attributes = attributes;
        node.children = children;
    }
    index
}

/// One laid-out node. Its index in the arena *is* its document-order position.
#[derive(Debug)]
struct RawNode {
    kind: NodeKind,
    name: Option<ExpandedName>,
    value: String,
    parent: Option<usize>,
    children: Vec<usize>,
    attributes: Vec<usize>,
    namespaces: Vec<usize>,
    base_uri: Option<String>,
    type_name: Option<ExpandedName>,
    typed_value: Option<Sequence>,
    is_id: bool,
    is_idref: bool,
}

/// The arena a constructed node lives in.
#[derive(Debug)]
struct ConstructedStore {
    tree_id: u64,
    nodes: Vec<RawNode>,
}

impl ConstructedStore {
    fn node(&self, id: NodeId) -> Option<&RawNode> {
        usize::try_from(id.0).ok().and_then(|id| self.nodes.get(id))
    }

    fn text_content(&self, index: usize, out: &mut String) {
        let Some(node) = self.nodes.get(index) else {
            return;
        };
        for &child in &node.children {
            match self.nodes.get(child).map(|data| data.kind) {
                Some(NodeKind::Text) => {
                    if let Some(data) = self.nodes.get(child) {
                        out.push_str(&data.value);
                    }
                }
                Some(NodeKind::Element) => self.text_content(child, out),
                _ => {}
            }
        }
    }

    fn ids(indices: &[usize]) -> Vec<NodeId> {
        indices.iter().map(|index| NodeId(*index as u64)).collect()
    }
}

impl XdmStore for ConstructedStore {
    fn node_kind(&self, node: NodeId) -> NodeKind {
        self.node(node).map_or(NodeKind::Element, |data| data.kind)
    }

    fn node_name(&self, node: NodeId) -> Option<ExpandedName> {
        self.node(node).and_then(|data| data.name.clone())
    }

    fn string_value(&self, node: NodeId) -> String {
        let Some(data) = self.node(node) else {
            return String::new();
        };
        match data.kind {
            NodeKind::Document | NodeKind::Element => {
                let mut out = String::new();
                if let Ok(index) = usize::try_from(node.0) {
                    self.text_content(index, &mut out);
                }
                out
            }
            _ => data.value.clone(),
        }
    }

    fn typed_value(&self, node: NodeId) -> Result<Sequence, oxixml_infoset::XdmError> {
        let Some(data) = self.node(node) else {
            return Ok(Sequence::empty());
        };
        if let Some(typed) = &data.typed_value {
            return Ok(typed.clone());
        }
        let string_value = self.string_value(node);
        Ok(match data.kind {
            NodeKind::Document | NodeKind::Element | NodeKind::Attribute | NodeKind::Text => {
                Sequence::from(AtomicValue::untyped_atomic(string_value))
            }
            NodeKind::Comment | NodeKind::ProcessingInstruction | NodeKind::Namespace => {
                Sequence::from(AtomicValue::string(string_value))
            }
        })
    }

    fn type_name(&self, node: NodeId) -> Option<ExpandedName> {
        self.node(node).and_then(|data| data.type_name.clone())
    }

    fn nilled(&self, node: NodeId) -> Option<bool> {
        let data = self.node(node)?;
        (data.kind == NodeKind::Element).then_some(false)
    }

    fn parent(&self, node: NodeId) -> Option<NodeId> {
        self.node(node)
            .and_then(|data| data.parent)
            .map(|index| NodeId(index as u64))
    }

    fn children(&self, node: NodeId) -> Vec<NodeId> {
        self.node(node)
            .map_or_else(Vec::new, |data| Self::ids(&data.children))
    }

    fn attributes(&self, node: NodeId) -> Vec<NodeId> {
        self.node(node)
            .map_or_else(Vec::new, |data| Self::ids(&data.attributes))
    }

    fn namespaces(&self, node: NodeId) -> Vec<NodeId> {
        self.node(node)
            .map_or_else(Vec::new, |data| Self::ids(&data.namespaces))
    }

    fn base_uri(&self, node: NodeId) -> Option<String> {
        let mut current = self.node(node);
        while let Some(data) = current {
            if let Some(base) = &data.base_uri {
                return Some(base.clone());
            }
            current = data.parent.and_then(|index| self.nodes.get(index));
        }
        None
    }

    fn document_uri(&self, _node: NodeId) -> Option<String> {
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
}

/// How a node placed in a constructor's content is copied.
#[derive(Debug, Clone)]
pub(crate) struct CopyOptions {
    pub(crate) copy_namespaces: CopyNamespaces,
    pub(crate) construction: ConstructionMode,
    /// The namespace bindings in scope at the constructor, added to a copied element when the
    /// copy-namespaces mode says `inherit`.
    pub(crate) inherited: Vec<(Option<String>, String)>,
    /// The static base URI, which an `xml:base` attribute on a copied element is resolved
    /// against.
    pub(crate) base_uri: Option<String>,
}

/// Copies `node` for placement in a constructor's content.
pub(crate) fn copy_node(node: &NodeRef, options: &CopyOptions) -> Draft {
    copy_inner(node, options)
}

fn copy_inner(node: &NodeRef, options: &CopyOptions) -> Draft {
    match node.node_kind() {
        NodeKind::Element => {
            let mut draft = Draft::element(node.node_name().unwrap_or_else(empty_name));
            if options.construction == ConstructionMode::Preserve {
                draft.type_name = node.type_name();
            }
            // §3.9.1.3: a copied element's base URI is that of its *new* parent, unless it
            // carries an `xml:base` attribute of its own. Leaving it unset is how it inherits.
            draft.base_uri = node
                .attributes()
                .into_iter()
                .find(|attribute| {
                    attribute.node_name().is_some_and(|name| {
                        name.namespace() == Some(XML_NAMESPACE) && name.local_name() == "base"
                    })
                })
                .map(|attribute| {
                    crate::eval::construct::resolve_uri(
                        options.base_uri.as_deref(),
                        &attribute.string_value(),
                    )
                });
            let mut bindings: Vec<(Option<String>, String)> = Vec::new();
            if options.copy_namespaces.preserve {
                for namespace in node.namespaces() {
                    let prefix = namespace
                        .node_name()
                        .map(|name| name.local_name().to_owned());
                    push_binding(&mut bindings, prefix, namespace.string_value());
                }
            } else {
                // `no-preserve` keeps only the bindings the copied names actually need, under
                // the prefixes the original used for them.
                let attributes = node.attributes();
                let names = std::iter::once(node.node_name())
                    .chain(attributes.iter().map(XdmNode::node_name))
                    .flatten();
                for name in names {
                    keep_binding_for(&mut bindings, node, &name);
                }
            }
            // `inherit` gives the copied node the in-scope namespaces of the constructed node.
            // Its descendants are copied nodes too, and their in-scope namespaces are recorded
            // per element in the data model, so the inherited bindings have to reach all the way
            // down rather than stopping at the topmost copy.
            if options.copy_namespaces.inherit {
                // An element whose own name is in no namespace cannot have a default namespace
                // binding in scope: XDM requires an element's in-scope namespaces to be able to
                // spell its own name, and `xmlns="urn:x"` would make the unprefixed name mean
                // `{urn:x}name`. Inheritance therefore skips the default binding for such an
                // element, and only for it (`K2-InScopePrefixesFunc-12`).
                let unqualified = node
                    .node_name()
                    .is_none_or(|name| name.namespace().is_none());
                for (prefix, uri) in &options.inherited {
                    if unqualified && prefix.is_none() && !uri.is_empty() {
                        continue;
                    }
                    push_binding(&mut bindings, prefix.clone(), uri.clone());
                }
            }
            draft.namespaces = bindings;
            for attribute in node.attributes() {
                draft.attributes.push(copy_inner(&attribute, options));
            }
            // §3.9.1.3 resolves a copied element's `xml:base` "against the base-uri property of
            // its new parent node", and the new parent of a copied *descendant* is the copy just
            // made — so the chain has to be rebased as it is walked, not resolved against the
            // outermost constructor throughout. `<e xml:base="…/ABC/"><a xml:base="../">
            // <b xml:base="DEF/f"/></a></e>` gives `b` the base `…/DEF/f`, not `…/ABC/DEF/f`
            // (`K2-BaseURIFunc-33`).
            let rebased;
            let child_options = match &draft.base_uri {
                Some(base) => {
                    rebased = CopyOptions {
                        base_uri: Some(base.clone()),
                        ..options.clone()
                    };
                    &rebased
                }
                None => options,
            };
            for child in node.children() {
                draft.children.push(copy_inner(&child, child_options));
            }
            draft
        }
        NodeKind::Attribute => {
            let mut draft = Draft::attribute(
                node.node_name().unwrap_or_else(empty_name),
                node.string_value(),
            );
            draft.is_id = node.is_id();
            draft.is_idref = node.is_idref();
            if options.construction == ConstructionMode::Preserve {
                draft.type_name = node.type_name();
                draft.typed_value = node.typed_value().ok();
            }
            draft
        }
        NodeKind::Text => Draft::text(node.string_value()),
        NodeKind::Comment => Draft::comment(node.string_value()),
        NodeKind::ProcessingInstruction => Draft::processing_instruction(
            node.node_name()
                .map(|name| name.local_name().to_owned())
                .unwrap_or_default()
                .as_str(),
            node.string_value(),
        ),
        NodeKind::Namespace => Draft::namespace(
            node.node_name()
                .map(|name| name.local_name().to_owned())
                .as_deref(),
            node.string_value(),
        ),
        NodeKind::Document => {
            let mut draft = Draft::document();
            draft.base_uri = node.base_uri();
            for child in node.children() {
                draft.children.push(copy_inner(&child, options));
            }
            draft
        }
    }
}

fn empty_name() -> ExpandedName {
    ExpandedName::new(None::<&str>, "")
}

/// Adds a binding unless the prefix is already bound, which keeps the innermost declaration.
pub(crate) fn push_binding(
    bindings: &mut Vec<(Option<String>, String)>,
    prefix: Option<String>,
    uri: String,
) {
    if bindings.iter().any(|(bound, _)| bound == &prefix) {
        return;
    }
    if uri.is_empty() && prefix.is_some() {
        return;
    }
    bindings.push((prefix, uri));
}

/// Keeps the binding `original` uses for `name`'s namespace, so that a `no-preserve` copy still
/// has a prefix for every name it carries.
fn keep_binding_for(
    bindings: &mut Vec<(Option<String>, String)>,
    original: &NodeRef,
    name: &ExpandedName,
) {
    let Some(namespace) = name.namespace() else {
        return;
    };
    if namespace == XML_NAMESPACE || bindings.iter().any(|(_, uri)| uri == namespace) {
        return;
    }
    let prefix = original
        .namespaces()
        .into_iter()
        .find(|binding| binding.string_value() == namespace)
        .and_then(|binding| binding.node_name().map(|name| name.local_name().to_owned()));
    match prefix {
        Some(prefix) => bindings.push((Some(prefix), namespace.to_owned())),
        None => bindings.push((
            Some(format!("ns{}", bindings.len() + 1)),
            namespace.to_owned(),
        )),
    }
}

/// Namespace fixup, per
/// [XQuery 3.1 §3.9.4](https://www.w3.org/TR/xquery-31/#id-in-scope-namespaces): a binding must
/// exist for the prefix each of a constructed element's own names uses, and where that would
/// need two different bindings of one prefix, the *name's* prefix is changed instead.
///
/// The prefix as written matters. §3.9.4 closes that rule with "if there is an in-scope default
/// namespace, then a binding is created between the empty prefix and that URI", so an element
/// name written without a prefix binds the *empty* prefix — even when some other prefix already
/// denotes the same namespace. `declare default element namespace "urn:x"; <e xmlns:p="urn:x"/>`
/// therefore has three in-scope namespaces (`""`, `p`, `xml`), not two.
///
/// An *attribute* has no such rule: an unprefixed attribute name is in no namespace, so a
/// namespaced attribute name that carries no prefix must be given an invented `ns1`, `ns2`, …
/// rather than the empty prefix.
///
/// The prefix the name ends up with is written back onto `name`, because §3.9.4 says the prefix
/// "used in the node name is changed" when it conflicts — and `fn:name`, `fn:node-name` and
/// `fn:prefix-from-QName` all report it.
pub(crate) fn fixup(
    bindings: &mut Vec<(Option<String>, String)>,
    name: &mut crate::eval::construct::ConstructedName,
) {
    fixup_name(bindings, name, false);
}

/// [`fixup`] for an attribute name, which may never bind the empty prefix.
pub(crate) fn fixup_attribute(
    bindings: &mut Vec<(Option<String>, String)>,
    name: &mut crate::eval::construct::ConstructedName,
) {
    fixup_name(bindings, name, true);
}

fn fixup_name(
    bindings: &mut Vec<(Option<String>, String)>,
    name: &mut crate::eval::construct::ConstructedName,
    attribute: bool,
) {
    let Some(namespace) = name.name.namespace().map(str::to_owned) else {
        return;
    };
    if namespace == XML_NAMESPACE {
        // The `xml` binding always exists, so a name in that namespace is always spelt with it.
        name.prefix = Some("xml".to_owned());
        return;
    }
    let wanted = name.prefix.clone();
    let existing = bindings
        .iter()
        .find(|(prefix, _)| prefix == &wanted)
        .map(|(_, uri)| uri.clone());
    // A prefix that already denotes this namespace can spell the name just as well as an
    // invented one, and keeps the query's own spelling where there is a choice.
    let bound_elsewhere = bindings
        .iter()
        .find(|(_, uri)| uri == &namespace)
        .and_then(|(prefix, _)| prefix.clone());
    // An attribute in a namespace cannot use the default binding, so a name that carries no
    // prefix needs one invented — unless some prefix already denotes its namespace.
    if attribute && wanted.is_none() {
        if let Some(prefix) = bound_elsewhere {
            name.prefix = Some(prefix);
            return;
        }
    } else {
        match existing {
            // Already bound the way the name needs it.
            Some(uri) if uri == namespace => return,
            // The prefix is taken by another namespace, so an invented one is needed — unless
            // some other prefix already denotes this namespace, which serializes just as well.
            Some(_) => {
                if let Some(prefix) = bound_elsewhere {
                    name.prefix = Some(prefix);
                    return;
                }
            }
            None => {
                bindings.push((wanted, namespace));
                return;
            }
        }
    }
    let mut index = bindings.len() + 1;
    let mut prefix = format!("ns{index}");
    while bindings
        .iter()
        .any(|(bound, _)| bound.as_deref() == Some(prefix.as_str()))
    {
        index += 1;
        prefix = format!("ns{index}");
    }
    bindings.push((Some(prefix.clone()), namespace));
    name.prefix = Some(prefix);
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::{Axis, Tree, axis_nodes};

    #[test]
    fn a_constructed_element_has_no_parent() {
        let node = Draft::element(ExpandedName::new(None::<&str>, "a")).build();
        assert_eq!(node.node_kind(), NodeKind::Element);
        assert!(node.parent().is_none());
        assert_eq!(node.string_value(), "");
    }

    #[test]
    fn a_constructed_text_node_stands_alone() {
        let node = Draft::text("hello").build();
        assert_eq!(node.node_kind(), NodeKind::Text);
        assert_eq!(node.string_value(), "hello");
        assert!(node.parent().is_none());
    }

    #[test]
    fn document_order_puts_namespaces_then_attributes_then_children() {
        let mut element = Draft::element(ExpandedName::new(None::<&str>, "a"));
        element
            .namespaces
            .push((Some("p".to_owned()), "urn:p".to_owned()));
        element
            .attributes
            .push(Draft::attribute(ExpandedName::new(None::<&str>, "x"), "1"));
        element.children.push(Draft::text("body"));
        let node = element.build();

        assert_eq!(node.namespaces().len(), 1);
        assert_eq!(node.attributes().len(), 1);
        assert_eq!(node.children().len(), 1);
        assert_eq!(node.string_value(), "body");

        let namespace = node.namespaces().first().cloned().expect("one namespace");
        let attribute = node.attributes().first().cloned().expect("one attribute");
        let child = node.children().first().cloned().expect("one child");
        assert!(node.document_order_key() < namespace.document_order_key());
        assert!(namespace.document_order_key() < attribute.document_order_key());
        assert!(attribute.document_order_key() < child.document_order_key());
        assert_eq!(axis_nodes(&node, Axis::Descendant).len(), 1);
        assert_eq!(child.parent(), Some(node));
    }

    #[test]
    fn copying_preserves_namespaces_by_default() -> Result<(), oxixml_infoset::XdmError> {
        let tree = Tree::parse_str(r#"<r xmlns:p="urn:p"><p:c a="1">text</p:c></r>"#)?;
        let source = tree
            .document_element()
            .and_then(|root| root.children().first().cloned())
            .expect("the child element");
        let options = CopyOptions {
            copy_namespaces: CopyNamespaces::default(),
            construction: ConstructionMode::Preserve,
            inherited: vec![(Some("q".to_owned()), "urn:q".to_owned())],
            base_uri: None,
        };
        let copy = copy_node(&source, &options).build();
        assert_eq!(copy.string_value(), "text");
        assert_eq!(copy.attributes().len(), 1);
        let prefixes: Vec<_> = copy
            .namespaces()
            .iter()
            .filter_map(|node| node.node_name().map(|name| name.local_name().to_owned()))
            .collect();
        assert!(prefixes.contains(&"p".to_owned()));
        assert!(prefixes.contains(&"q".to_owned()), "inherit adds `q`");
        assert!(copy.parent().is_none(), "a copy is a new root");
        Ok(())
    }

    #[test]
    fn no_preserve_keeps_only_what_the_names_need() -> Result<(), oxixml_infoset::XdmError> {
        let tree = Tree::parse_str(r#"<r xmlns:p="urn:p" xmlns:unused="urn:u"><p:c/></r>"#)?;
        let source = tree
            .document_element()
            .and_then(|root| root.children().first().cloned())
            .expect("the child element");
        let options = CopyOptions {
            copy_namespaces: CopyNamespaces {
                preserve: false,
                inherit: false,
            },
            construction: ConstructionMode::Preserve,
            inherited: Vec::new(),
            base_uri: None,
        };
        let copy = copy_node(&source, &options).build();
        assert_eq!(copy.namespaces().len(), 1);
        Ok(())
    }
}
