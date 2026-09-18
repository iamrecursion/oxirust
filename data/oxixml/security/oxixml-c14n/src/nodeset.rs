//! The document subset — Canonical XML's input model.
//!
//! [Canonical XML §2.1](https://www.w3.org/TR/xml-c14n#DataModel) takes an
//! *XPath node-set*: a plain set of nodes, not a set of subtrees. A node that is
//! not in the set produces no output of its own, yet its descendants that *are*
//! in the set still do — and the omitted node keeps affecting them, because the
//! namespace context and the `xml:*` attributes it contributes are still in
//! scope. [`NodeSet`] is that set.
//!
//! # Namespace nodes
//!
//! XPath models an element's in-scope namespace bindings as *namespace nodes*
//! hanging off the element, and each of them can be in or out of the node-set
//! independently. Real selections practically never split them (the canonical
//! `(//. | //@* | //namespace::*)[…]` idiom keeps an element's namespace nodes
//! exactly when it keeps the element), so [`NodeSet`] follows that rule by
//! default and only materialises a per-element prefix set once
//! [`NodeSet::set_namespace_node`] is called.
//!
//! # Example
//!
//! ```
//! use oxixml_c14n::NodeSet;
//! use oxixml_dom::build;
//!
//! let doc = build::parse_str("<a><b/><c/></a>")?;
//! let a = doc.document_element().expect("a document element");
//! let b = doc.children(a).next().expect("a first child");
//!
//! // Everything …
//! let all = NodeSet::document(&doc);
//! assert!(all.contains(doc.root()) && all.contains(b));
//!
//! // … or one subtree.
//! let subtree = NodeSet::subtree(&doc, b);
//! assert!(subtree.contains(b));
//! assert!(!subtree.contains(a));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::{BTreeMap, BTreeSet};

use oxixml_dom::{Document, NodeId, NodeKind};

/// A set of document nodes: the *document subset* that gets canonicalized.
///
/// A node that is not in the set produces no output of its own, yet its
/// descendants that *are* in the set still do, and the omitted node keeps
/// shaping them through the namespace context and the `xml:*` attributes it
/// contributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeSet {
    included: Vec<bool>,
    /// `None` keeps namespace nodes tied to their element; `Some` is an
    /// explicit per-element set of included prefixes (`""` is the default
    /// namespace node).
    namespaces: Option<BTreeMap<NodeId, BTreeSet<String>>>,
}

impl NodeSet {
    /// Creates an empty subset sized for `document`.
    #[must_use]
    pub fn empty(document: &Document) -> Self {
        Self {
            included: vec![false; document.len()],
            namespaces: None,
        }
    }

    /// Creates the subset that the default XPath expression
    /// `(//. | //@* | //namespace::*)` selects: every node of the document,
    /// comments included.
    #[must_use]
    pub fn document(document: &Document) -> Self {
        let mut set = Self::empty(document);
        set.insert_subtree(document, document.root());
        set
    }

    /// Creates the subset that
    /// `(//. | //@* | //namespace::*)[not(self::comment())]` selects: every node
    /// of the document except the comments.
    #[must_use]
    pub fn document_without_comments(document: &Document) -> Self {
        let mut set = Self::document(document);
        set.remove_comments(document);
        set
    }

    /// Creates the subset holding `root`, its attributes, and every descendant
    /// with their attributes — the "canonicalize this element" case.
    ///
    /// The document node is *not* in the result, so `root` is an
    /// [orphan](https://www.w3.org/TR/xml-exc-c14n/#dfn-orphan-node) and the
    /// `xml:*` inheritance rules of a document subset apply to it.
    #[must_use]
    pub fn subtree(document: &Document, root: NodeId) -> Self {
        let mut set = Self::empty(document);
        set.insert_subtree(document, root);
        set
    }

    /// Adds a single node.
    pub fn insert(&mut self, node: NodeId) -> &mut Self {
        if let Some(slot) = self.included.get_mut(node.index()) {
            *slot = true;
        }
        self
    }

    /// Removes a single node.
    pub fn remove(&mut self, node: NodeId) -> &mut Self {
        if let Some(slot) = self.included.get_mut(node.index()) {
            *slot = false;
        }
        self
    }

    /// Adds `node`, its attribute nodes, and all of its descendants together
    /// with theirs.
    pub fn insert_subtree(&mut self, document: &Document, node: NodeId) -> &mut Self {
        self.set_subtree(document, node, true)
    }

    /// Removes `node`, its attribute nodes, and all of its descendants together
    /// with theirs.
    pub fn remove_subtree(&mut self, document: &Document, node: NodeId) -> &mut Self {
        self.set_subtree(document, node, false)
    }

    /// Removes every comment node from the subset, which is what the `false`
    /// value of the *comment* parameter of Canonical XML means.
    pub fn remove_comments(&mut self, document: &Document) -> &mut Self {
        for index in 0..self.included.len() {
            let node = NodeId::from_index(index as u32);
            if document.kind(node) == Some(NodeKind::Comment) {
                self.included[index] = false;
            }
        }
        self
    }

    fn set_subtree(&mut self, document: &Document, node: NodeId, value: bool) -> &mut Self {
        let mut stack = vec![node];
        while let Some(current) = stack.pop() {
            if let Some(slot) = self.included.get_mut(current.index()) {
                *slot = value;
            }
            for attribute in document.attributes(current) {
                if let Some(slot) = self.included.get_mut(attribute.index()) {
                    *slot = value;
                }
            }
            stack.extend(document.children(current));
        }
        self
    }

    /// Returns `true` when `node` is a member of the subset.
    #[must_use]
    pub fn contains(&self, node: NodeId) -> bool {
        self.included.get(node.index()).copied().unwrap_or(false)
    }

    /// Returns the number of nodes in the subset, not counting namespace nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.included.iter().filter(|included| **included).count()
    }

    /// Returns `true` when no node at all is in the subset.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        !self.included.iter().any(|included| *included)
    }

    /// Includes or excludes one namespace node.
    ///
    /// `prefix` is the declared prefix, or `""` for the default namespace node.
    /// The first call switches the subset from the implicit rule ("an element's
    /// namespace nodes are in the subset exactly when the element is") to an
    /// explicit per-element list seeded from that rule.
    pub fn set_namespace_node(
        &mut self,
        document: &Document,
        element: NodeId,
        prefix: &str,
        included: bool,
    ) -> &mut Self {
        let namespaces = self
            .namespaces
            .get_or_insert_with(|| seed_namespace_nodes(document, &self.included));
        let entry = namespaces.entry(element).or_default();
        if included {
            entry.insert(prefix.to_owned());
        } else {
            entry.remove(prefix);
        }
        self
    }

    /// Returns `true` when the namespace node for `prefix` on `element` is a
    /// member of the subset.
    ///
    /// This only answers the *membership* question; whether such a namespace
    /// node exists at all is decided by the in-scope bindings of `element`.
    #[must_use]
    pub fn contains_namespace_node(&self, element: NodeId, prefix: &str) -> bool {
        match &self.namespaces {
            None => self.contains(element),
            Some(explicit) => explicit
                .get(&element)
                .is_some_and(|prefixes| prefixes.contains(prefix)),
        }
    }
}

/// Builds the explicit namespace-node map that matches the implicit rule, so
/// that switching to explicit bookkeeping does not change the current subset.
fn seed_namespace_nodes(
    document: &Document,
    included: &[bool],
) -> BTreeMap<NodeId, BTreeSet<String>> {
    let mut map = BTreeMap::new();
    for (index, present) in included.iter().enumerate() {
        if !present {
            continue;
        }
        let element = NodeId::from_index(index as u32);
        if document.kind(element) != Some(NodeKind::Element) {
            continue;
        }
        let prefixes = crate::namespaces::in_scope_prefixes(document, element);
        map.insert(element, prefixes);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_dom::build;

    #[test]
    fn subtree_excludes_the_document_node() -> Result<(), Box<dyn std::error::Error>> {
        let doc = build::parse_str("<a x='1'><b/></a>")?;
        let a = doc.document_element().ok_or("no document element")?;
        let set = NodeSet::subtree(&doc, a);
        assert!(!set.contains(doc.root()));
        assert!(set.contains(a));
        assert_eq!(set.len(), 3); // a, @x, b
        Ok(())
    }

    #[test]
    fn comments_can_be_dropped() -> Result<(), Box<dyn std::error::Error>> {
        let doc = build::parse_str("<a><!-- c --><b/></a>")?;
        let with = NodeSet::document(&doc);
        let without = NodeSet::document_without_comments(&doc);
        assert_eq!(with.len(), without.len() + 1);
        Ok(())
    }

    #[test]
    fn explicit_namespace_nodes_start_from_the_implicit_rule()
    -> Result<(), Box<dyn std::error::Error>> {
        let doc = build::parse_str("<a xmlns:p='urn:p'><b/></a>")?;
        let a = doc.document_element().ok_or("no document element")?;
        let mut set = NodeSet::document(&doc);
        assert!(set.contains_namespace_node(a, "p"));
        set.set_namespace_node(&doc, a, "p", false);
        assert!(!set.contains_namespace_node(a, "p"));
        assert!(set.contains_namespace_node(a, "xml"));
        Ok(())
    }
}
