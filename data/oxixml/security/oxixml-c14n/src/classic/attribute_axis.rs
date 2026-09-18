//! Rendering the attribute axis of an element.
//!
//! Attributes are sorted "with namespace URI as the primary key and local name
//! as the secondary key (an empty namespace URI is lexicographically least)"
//! ([Canonical XML §2.2](https://www.w3.org/TR/xml-c14n#DocumentOrder)), and
//! namespace declarations are not attributes at all — they are the namespace
//! axis and were already written.
//!
//! # Document subsets
//!
//! When an element's parent is *omitted* from the node-set, the element becomes
//! an orphan and the context its ancestors used to supply has to be recreated,
//! or the canonical form would silently change meaning. The three algorithms
//! disagree on how much of it to recreate:
//!
//! * **Canonical XML 1.0** ([§2.4](https://www.w3.org/TR/xml-c14n#DocSubsets))
//!   copies the nearest occurrence of *every* attribute in the `xml` namespace
//!   found along the ancestor axis, minus the ones the element already has.
//! * **Canonical XML 1.1** ([§2.4](https://www.w3.org/TR/xml-c14n11/#DocSubsets))
//!   narrows that to the *simple inheritable* attributes `xml:lang` and
//!   `xml:space`; `xml:id` is deliberately not inherited (it would duplicate an
//!   identifier), and `xml:base` gets a dedicated fixup that folds the base URIs
//!   of the dropped ancestors into a single relative reference.
//! * **Exclusive C14N** ([§3](https://www.w3.org/TR/xml-exc-c14n/#sec-Specification))
//!   omits the search entirely — importing context is exactly what it is
//!   designed not to do.

use std::collections::BTreeMap;

use oxixml_dom::{NodeId, XML_NAMESPACE};

use crate::error::C14nResult;
use crate::escape::write_attribute;
use crate::namespaces::is_namespace_declaration;
use crate::uri::join_uri_references;

use super::Variant;
use super::engine::Engine;

/// `xml:base`, the attribute Canonical XML 1.1 fixes up.
const XML_BASE: &str = "base";
/// The local names of the *simple inheritable* attributes of Canonical XML 1.1.
const SIMPLE_INHERITABLE: [&str; 2] = ["lang", "space"];

/// One attribute ready to be written, with the keys it sorts on.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Rendered {
    namespace: String,
    local: String,
    qualified_name: String,
    value: String,
}

impl Engine<'_> {
    /// Emits the attribute nodes of `element` that are in the node-set, plus the
    /// `xml:*` attributes a document subset has to re-import.
    pub(crate) fn render_attribute_axis(&mut self, element: NodeId) -> C14nResult<()> {
        let mut attributes = self.own_attributes(element);

        let in_subset = self.subset.contains(element);
        let parent_omitted = match self.document.parent(element) {
            None => true,
            Some(parent) => !self.subset.contains(parent),
        };
        if in_subset && parent_omitted {
            match self.variant {
                Variant::Inclusive10 => self.import_xml_attributes(element, &mut attributes, None),
                Variant::Inclusive11 => {
                    self.import_xml_attributes(element, &mut attributes, Some(&SIMPLE_INHERITABLE));
                    self.fix_up_xml_base(element, &mut attributes);
                }
                Variant::Exclusive(_) => {}
            }
        }

        attributes.sort();
        for attribute in &attributes {
            write_attribute(&mut self.out, &attribute.qualified_name, &attribute.value);
        }
        Ok(())
    }

    /// The element's own attribute nodes that are in the node-set, with the
    /// namespace declarations filtered out.
    fn own_attributes(&self, element: NodeId) -> Vec<Rendered> {
        let mut out = Vec::new();
        for attribute in self.document.attributes(element) {
            if !self.subset.contains(attribute)
                || is_namespace_declaration(self.document, attribute)
            {
                continue;
            }
            if let Some(data) = self.document.attr(attribute) {
                out.push(Rendered {
                    namespace: data.namespace_uri().unwrap_or_default().to_owned(),
                    local: data.local_name().to_owned(),
                    qualified_name: data.qualified_name(),
                    value: data.value().to_owned(),
                });
            }
        }
        out
    }

    /// Copies the nearest occurrence of each inheritable `xml:*` attribute from
    /// the ancestor axis, skipping the ones the element declares itself.
    ///
    /// `only` restricts the search to a set of local names; `None` means "every
    /// attribute in the XML namespace", which is the Canonical XML 1.0 rule.
    fn import_xml_attributes(
        &self,
        element: NodeId,
        attributes: &mut Vec<Rendered>,
        only: Option<&[&str]>,
    ) {
        let mut inherited: BTreeMap<String, Rendered> = BTreeMap::new();
        for ancestor in self.document.ancestors(element) {
            for attribute in self.document.attributes(ancestor) {
                let Some(data) = self.document.attr(attribute) else {
                    continue;
                };
                if data.namespace_uri() != Some(XML_NAMESPACE) {
                    continue;
                }
                let local = data.local_name();
                if let Some(allowed) = only {
                    if !allowed.contains(&local) {
                        continue;
                    }
                }
                inherited
                    .entry(local.to_owned())
                    .or_insert_with(|| Rendered {
                        namespace: XML_NAMESPACE.to_owned(),
                        local: local.to_owned(),
                        qualified_name: data.qualified_name(),
                        value: data.value().to_owned(),
                    });
            }
        }

        // "From this list of attributes, remove any that are in E's attribute
        // axis (whether or not they are in the node-set)."
        for attribute in self.document.attributes(element) {
            if let Some(data) = self.document.attr(attribute) {
                if data.namespace_uri() == Some(XML_NAMESPACE) {
                    inherited.remove(data.local_name());
                }
            }
        }

        attributes.extend(inherited.into_values());
    }

    /// Canonical XML 1.1 `xml:base` fixup.
    ///
    /// The base URIs contributed by the ancestors that the node-set drops are
    /// folded, innermost first, into the element's own `xml:base` with
    /// [`join_uri_references`]. An empty result is not rendered at all, as the
    /// Recommendation requires.
    fn fix_up_xml_base(&self, element: NodeId, attributes: &mut Vec<Rendered>) {
        // "Let E be an element in the node set whose ancestor axis contains
        // *successive* elements En … E1 (in reverse document order) that are
        // omitted and E=En+1 is included" — the omitted ancestors have to be
        // contiguous. `ancestors` yields them nearest first, so the run ends at
        // the first ancestor that is itself in the node-set: that ancestor was
        // rendered with its own folded `xml:base`, and folding it in again would
        // apply it twice.
        let mut ancestors: Vec<NodeId> = self
            .document
            .ancestors(element)
            .take_while(|node| !self.subset.contains(*node))
            .collect();
        ancestors.reverse();

        let mut values: Vec<String> = ancestors
            .into_iter()
            .filter_map(|ancestor| self.xml_base_value(ancestor))
            .collect();
        if values.is_empty() {
            return;
        }
        if let Some(own) = self.xml_base_value(element) {
            values.push(own);
        }

        let mut folded = String::new();
        for (index, value) in values.iter().enumerate().rev() {
            folded = if index + 1 == values.len() {
                value.clone()
            } else {
                join_uri_references(value, &folded)
            };
        }

        attributes.retain(|attribute| {
            !(attribute.namespace == XML_NAMESPACE && attribute.local == XML_BASE)
        });
        if !folded.is_empty() {
            attributes.push(Rendered {
                namespace: XML_NAMESPACE.to_owned(),
                local: XML_BASE.to_owned(),
                qualified_name: "xml:base".to_owned(),
                value: folded,
            });
        }
    }

    /// Returns the `xml:base` attribute value written on `node`, if any.
    fn xml_base_value(&self, node: NodeId) -> Option<String> {
        self.document.attributes(node).find_map(|attribute| {
            let data = self.document.attr(attribute)?;
            (data.namespace_uri() == Some(XML_NAMESPACE) && data.local_name() == XML_BASE)
                .then(|| data.value().to_owned())
        })
    }
}
