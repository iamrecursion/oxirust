//! Rendering the namespace axis of an element.
//!
//! This is the one place where the three classic algorithms genuinely differ.
//!
//! # Inclusive (Canonical XML 1.0 and 1.1)
//!
//! [Canonical XML §2.3](https://www.w3.org/TR/xml-c14n#ProcessingModel) walks a
//! list `L` of the namespace nodes that are both on the axis and in the
//! node-set, in ascending prefix order:
//!
//! * if `L` does not start with the default namespace node, ` xmlns=""` is
//!   emitted *iff* the element is in the node-set **and** its nearest ancestor
//!   element in the node-set has a default namespace node in the node-set — the
//!   rule that keeps `xmlns=""` from appearing where it would be redundant;
//! * the `xml` prefix bound to the XML namespace is never rendered;
//! * a namespace node is *superfluous*, and therefore skipped, when the nearest
//!   ancestor element in the node-set already carries a namespace node with the
//!   same prefix and the same value.
//!
//! # Exclusive
//!
//! [Exclusive C14N §3](https://www.w3.org/TR/xml-exc-c14n/#sec-Specification)
//! renders a declaration only where it is *visibly utilized* — that is, where
//! the element's own qualified name or one of its attribute names actually uses
//! the prefix — or where the prefix is named in the `InclusiveNamespaces`
//! `PrefixList`. The inherited context is therefore not reproduced at the apex;
//! instead declarations are "pushed down" to the elements that use them, which
//! is what makes an extracted subdocument canonicalize identically no matter how
//! it was enveloped.

use std::collections::{BTreeMap, BTreeSet};

use oxixml_dom::{NodeId, XML_NAMESPACE};

use crate::error::C14nResult;
use crate::escape::write_attribute;
use crate::namespaces::{
    Bindings, DEFAULT_PREFIX, XML_PREFIX, check_absolute, is_namespace_declaration,
    namespace_node_name,
};

use super::engine::{Engine, Frame};
use super::{InclusiveNamespaces, Variant};

impl Engine<'_> {
    /// Emits the namespace nodes of `element` that belong in the canonical form.
    pub(crate) fn render_namespace_axis(
        &mut self,
        element: NodeId,
        bindings: &Bindings,
        frame: &Frame,
        rendered: &mut BTreeMap<String, String>,
    ) -> C14nResult<()> {
        for (prefix, namespace) in bindings.iter() {
            check_absolute(prefix, namespace)?;
        }
        match self.variant {
            Variant::Inclusive10 | Variant::Inclusive11 => {
                self.render_inclusive(element, bindings, frame)
            }
            Variant::Exclusive(list) => {
                // Cloning keeps the borrow checker happy without touching the
                // engine's own borrow of `self.variant`.
                let list = list.clone();
                self.render_exclusive(element, bindings, &list, rendered)
            }
        }
    }

    fn render_inclusive(
        &mut self,
        element: NodeId,
        bindings: &Bindings,
        frame: &Frame,
    ) -> C14nResult<()> {
        let in_subset = self.subset.contains(element);

        // L: the namespace nodes on the axis that are also in the node-set,
        // already in ascending prefix order because `Bindings` is a `BTreeMap`
        // and the default namespace node keys on the empty string.
        let axis: Vec<(&str, &str)> = bindings
            .iter()
            .filter(|(prefix, _)| self.subset.contains_namespace_node(element, prefix))
            .collect();

        let starts_with_default = axis
            .first()
            .is_some_and(|(prefix, _)| *prefix == DEFAULT_PREFIX);
        if !starts_with_default && in_subset && self.ancestor_has_default_namespace_node(frame) {
            write_attribute(&mut self.out, "xmlns", "");
        }

        for (prefix, namespace) in axis {
            if prefix == XML_PREFIX && namespace == XML_NAMESPACE {
                continue;
            }
            if self.ancestor_renders_same_namespace_node(frame, prefix, namespace) {
                continue;
            }
            write_attribute(&mut self.out, &namespace_node_name(prefix), namespace);
        }
        Ok(())
    }

    /// "The nearest ancestor element of `E` in the node-set has a default
    /// namespace node in the node-set."
    fn ancestor_has_default_namespace_node(&self, frame: &Frame) -> bool {
        match &frame.visible {
            None => false,
            Some((ancestor, ancestor_bindings)) => {
                !ancestor_bindings.default_namespace().is_empty()
                    && self
                        .subset
                        .contains_namespace_node(*ancestor, DEFAULT_PREFIX)
            }
        }
    }

    /// "A namespace node `N` is ignored if the nearest ancestor element of the
    /// node's parent element that is in the node-set has a namespace node in the
    /// node-set with the same local name and value as `N`."
    fn ancestor_renders_same_namespace_node(
        &self,
        frame: &Frame,
        prefix: &str,
        namespace: &str,
    ) -> bool {
        match &frame.visible {
            None => false,
            Some((ancestor, ancestor_bindings)) => {
                ancestor_bindings.get(prefix) == Some(namespace)
                    && self.subset.contains_namespace_node(*ancestor, prefix)
            }
        }
    }

    fn render_exclusive(
        &mut self,
        element: NodeId,
        bindings: &Bindings,
        inclusive: &InclusiveNamespaces,
        rendered: &mut BTreeMap<String, String>,
    ) -> C14nResult<()> {
        // "Its parent element is in the node-set" — an omitted element renders
        // no namespace node at all.
        if !self.subset.contains(element) {
            return Ok(());
        }

        let utilized = self.visibly_utilized_prefixes(element);
        let mut candidates: BTreeSet<&str> = utilized.iter().map(String::as_str).collect();
        for prefix in inclusive.iter() {
            if prefix == InclusiveNamespaces::DEFAULT_TOKEN {
                candidates.insert(DEFAULT_PREFIX);
            } else {
                candidates.insert(prefix);
            }
        }

        for prefix in candidates {
            if !self.subset.contains_namespace_node(element, prefix) {
                continue;
            }
            let namespace = bindings.get(prefix).unwrap_or("");
            if prefix == XML_PREFIX && namespace == XML_NAMESPACE {
                continue;
            }
            if prefix == DEFAULT_PREFIX && namespace.is_empty() {
                // `xmlns=""` is the *absence* of a namespace node, so it is only
                // written to cancel a default namespace an output ancestor set.
                let cancels = rendered
                    .get(DEFAULT_PREFIX)
                    .is_some_and(|value| !value.is_empty());
                if cancels {
                    write_attribute(&mut self.out, "xmlns", "");
                    rendered.insert(DEFAULT_PREFIX.to_owned(), String::new());
                }
                continue;
            }
            if namespace.is_empty() {
                // A prefix named only by the `PrefixList` that is not bound
                // here has no namespace node to render.
                continue;
            }
            if rendered.get(prefix).map(String::as_str) == Some(namespace) {
                continue;
            }
            write_attribute(&mut self.out, &namespace_node_name(prefix), namespace);
            rendered.insert(prefix.to_owned(), namespace.to_owned());
        }
        Ok(())
    }

    /// The prefixes an element "visibly utilizes": the one in its own qualified
    /// name (or the default namespace when it has none) and the ones in the
    /// qualified names of its attributes that are in the node-set.
    fn visibly_utilized_prefixes(&self, element: NodeId) -> BTreeSet<String> {
        let mut utilized = BTreeSet::new();
        if let Some(data) = self.document.element(element) {
            utilized.insert(data.prefix().unwrap_or(DEFAULT_PREFIX).to_owned());
        }
        for attribute in self.document.attributes(element) {
            if !self.subset.contains(attribute)
                || is_namespace_declaration(self.document, attribute)
            {
                continue;
            }
            if let Some(data) = self.document.attr(attribute) {
                // An unprefixed attribute is locally scoped: it does *not*
                // visibly utilize the default namespace.
                if let Some(prefix) = data.prefix() {
                    utilized.insert(prefix.to_owned());
                }
            }
        }
        utilized
    }
}
