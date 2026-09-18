//! Canonical XML 2.0 — a staged, best-effort implementation.
//!
//! [Canonical XML 2.0](https://www.w3.org/TR/xml-c14n2/) never left Candidate
//! Recommendation: the XML Security Working Group closed before it advanced, so
//! the specification is stable but not a Recommendation, and no widely deployed
//! signature profile requires it. This crate therefore implements it as a
//! **staged feature**: the algorithm, the parameter model and the
//! `<ds:CanonicalizationMethod>` binding are all here and are checked against
//! the W3C [test cases][tc], but the two areas the Note itself calls out as
//! under-specified are handled as described below rather than guessed at.
//!
//! # What is implemented
//!
//! * The restricted input model of [§2.1][data]: an inclusion list of element
//!   nodes (or the document node) minus an exclusion list of element and
//!   attribute nodes — see [`C14n2Subset`].
//! * All four parameters of [§2.2][params] — `IgnoreComments`,
//!   `TrimTextNodes`, `PrefixRewrite` and `QNameAware` — including reading them
//!   out of a `<ds:CanonicalizationMethod>` element.
//! * The full namespace processing algorithm of [§2.5.3][ns]: visible
//!   utilization (names, `QName`-valued content and XPath expressions),
//!   sequential prefix rewriting with a document-wide URI → prefix map, the
//!   "already output by an ancestor" filter, and prefix-order sorting.
//! * Attribute processing of [§2.6][attrs], and the node rendering of
//!   [§2.3][model].
//!
//! # Known deviations
//!
//! * **Parameter defaults.** This crate follows the published defaults
//!   (`IgnoreComments` and `TrimTextNodes` both default to `true`). The W3C test
//!   vectors were produced against an earlier draft in which `TrimTextNodes`
//!   defaulted to `false`, so reproducing them requires setting the parameters
//!   explicitly; the conformance test does exactly that.
//! * **`PrefixRewrite="sequential"` and the empty namespace.** Rewriting maps
//!   *every* visibly utilized namespace name to a prefix, including the empty
//!   one, which yields `xmlns:n0=""` — a binding that Namespaces in XML forbids.
//!   The W3C test vectors require it, so it is what this implementation
//!   produces.
//!
//! [tc]: https://www.w3.org/TR/xml-c14n2-testcases/
//! [data]: https://www.w3.org/TR/xml-c14n2/#sec-DataModel
//! [params]: https://www.w3.org/TR/xml-c14n2/#sec-Parameters
//! [model]: https://www.w3.org/TR/xml-c14n2/#sec-Processing-Model
//! [ns]: https://www.w3.org/TR/xml-c14n2/#sec-Namespace-Processing-Algorithm
//! [attrs]: https://www.w3.org/TR/xml-c14n2/#sec-Attribute-Processing

mod engine;
mod params;
mod qname;

use std::collections::BTreeSet;

use oxixml_dom::{Document, NodeId};

use crate::error::C14nResult;

pub use params::{C14N2_NAMESPACE, C14n2Parameters, PrefixRewrite, QNameAware};

/// The document subset Canonical XML 2.0 operates on.
///
/// [§2.1](https://www.w3.org/TR/xml-c14n2/#sec-DataModel) deliberately drops the
/// full XPath node-set of the earlier versions in favour of "one intersect
/// followed by one optional subtract": an *inclusion list* of element nodes (or
/// the document node), minus an *exclusion list* of whole element subtrees and
/// of individual attribute nodes. Namespace declarations and attributes in the
/// `xml` namespace cannot be excluded.
///
/// # Example
///
/// ```
/// use oxixml_c14n::{C14n2Parameters, C14n2Subset, c14n_2_0};
/// use oxixml_dom::build;
///
/// let doc = build::parse_str("<r><keep>a</keep><drop>b</drop></r>")?;
/// let root = doc.document_element().expect("a document element");
/// let drop = doc.children(root).nth(1).expect("a second child");
///
/// let mut subset = C14n2Subset::document();
/// subset.exclude_element(drop);
///
/// let parameters = C14n2Parameters::new().with_trim_text_nodes(false);
/// assert_eq!(
///     c14n_2_0(&doc, &subset, &parameters)?,
///     b"<r><keep>a</keep></r>",
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct C14n2Subset {
    inclusion: Vec<NodeId>,
    excluded_elements: BTreeSet<NodeId>,
    excluded_attributes: BTreeSet<NodeId>,
}

impl C14n2Subset {
    /// Selects the whole document, i.e. an inclusion list holding only the
    /// document node.
    #[must_use]
    pub fn document() -> Self {
        Self::default()
    }

    /// Selects one or more element subtrees.
    ///
    /// Elements that are descendants of another listed element are ignored, as
    /// [§2.3](https://www.w3.org/TR/xml-c14n2/#sec-Processing-Model) requires.
    pub fn elements<I: IntoIterator<Item = NodeId>>(elements: I) -> Self {
        Self {
            inclusion: elements.into_iter().collect(),
            ..Self::default()
        }
    }

    /// Adds an element subtree to the inclusion list.
    pub fn include_element(&mut self, element: NodeId) -> &mut Self {
        self.inclusion.push(element);
        self
    }

    /// Prunes an element and everything below it.
    pub fn exclude_element(&mut self, element: NodeId) -> &mut Self {
        self.excluded_elements.insert(element);
        self
    }

    /// Drops a single attribute node.
    pub fn exclude_attribute(&mut self, attribute: NodeId) -> &mut Self {
        self.excluded_attributes.insert(attribute);
        self
    }

    /// Returns `true` when the inclusion list is the document node.
    #[must_use]
    pub fn is_whole_document(&self) -> bool {
        self.inclusion.is_empty()
    }

    /// Returns `true` when `element` — or any of its ancestors — was excluded.
    #[must_use]
    pub fn is_excluded_element(&self, element: NodeId) -> bool {
        self.excluded_elements.contains(&element)
    }

    /// Returns `true` when `attribute` was excluded.
    #[must_use]
    pub fn is_excluded_attribute(&self, attribute: NodeId) -> bool {
        self.excluded_attributes.contains(&attribute)
    }

    /// Approximates an XPath [`NodeSet`](crate::NodeSet) in the restricted input
    /// model of Canonical XML 2.0.
    ///
    /// The mapping is the obvious one, and is lossy in exactly the ways
    /// [§2.1](https://www.w3.org/TR/xml-c14n2/#sec-DataModel) says the model is
    /// narrower than a node-set:
    ///
    /// * if the document node is in the node-set the whole document is included,
    ///   otherwise the *apex* elements — those with no element ancestor in the
    ///   node-set — become the inclusion list;
    /// * every element below an apex that is *not* in the node-set is excluded,
    ///   together with its subtree, and so is every attribute that is not in the
    ///   node-set;
    /// * text, comment and processing-instruction nodes cannot be excluded
    ///   individually, so dropping one of those from the node-set has no effect.
    #[must_use]
    pub fn from_node_set(document: &Document, subset: &crate::NodeSet) -> Self {
        let mut result = if subset.contains(document.root()) {
            Self::document()
        } else {
            let apexes: Vec<NodeId> = document
                .nodes_in_document_order()
                .into_iter()
                .filter(|node| {
                    document.kind(*node) == Some(oxixml_dom::NodeKind::Element)
                        && subset.contains(*node)
                        && !document
                            .ancestors(*node)
                            .any(|ancestor| subset.contains(ancestor))
                })
                .collect();
            Self::elements(apexes)
        };

        let roots = if result.is_whole_document() {
            vec![document.root()]
        } else {
            result.apex_nodes(document)
        };
        let mut stack = roots;
        while let Some(current) = stack.pop() {
            for attribute in document.attributes(current) {
                if !subset.contains(attribute) {
                    result.exclude_attribute(attribute);
                }
            }
            for child in document.children(current) {
                if document.kind(child) == Some(oxixml_dom::NodeKind::Element)
                    && !subset.contains(child)
                {
                    result.exclude_element(child);
                    continue;
                }
                stack.push(child);
            }
        }
        result
    }

    /// Returns the apex nodes: the inclusion list with descendants removed and
    /// the rest sorted into document order.
    #[must_use]
    pub fn apex_nodes(&self, document: &Document) -> Vec<NodeId> {
        let mut apexes: Vec<NodeId> = self
            .inclusion
            .iter()
            .copied()
            .filter(|candidate| {
                !self.inclusion.iter().any(|other| {
                    *other != *candidate && document.is_ancestor_of(*other, *candidate)
                })
            })
            .collect();
        apexes.sort_by(|a, b| {
            document
                .compare_document_order(*a, *b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        apexes.dedup();
        apexes
    }
}

/// Runs Canonical XML 2.0 over a document subset.
pub(crate) fn canonicalize(
    document: &Document,
    subset: &C14n2Subset,
    parameters: &C14n2Parameters,
) -> C14nResult<Vec<u8>> {
    engine::Engine::new(document, subset, parameters).run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_dom::build;

    #[test]
    fn descendants_drop_out_of_the_inclusion_list() -> Result<(), Box<dyn std::error::Error>> {
        let doc = build::parse_str("<r><a><b/></a><c/></r>")?;
        let root = doc.document_element().ok_or("no document element")?;
        let a = doc.children(root).next().ok_or("no first child")?;
        let b = doc.children(a).next().ok_or("no grandchild")?;
        let c = doc.children(root).nth(1).ok_or("no second child")?;

        let subset = C14n2Subset::elements([c, b, a]);
        assert_eq!(subset.apex_nodes(&doc), vec![a, c]);
        Ok(())
    }
}
