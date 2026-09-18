//! Resolving a shortname XPointer — the `foo` in `URI="#foo"` — to an element.
//!
//! # Why this is not simply `getElementById`
//!
//! [XML Signature §4.4.3.3](https://www.w3.org/TR/xmldsig-core1/#sec-ReferenceProcessingModel)
//! says a bare fragment is a shortname XPointer, which
//! [XPointer Framework](https://www.w3.org/TR/xptr-framework/) resolves against
//! nodes of type `ID`. A validating parser learns which attributes are of type
//! `ID` from the DTD, and `xml:id` is `ID` by definition — but virtually every
//! real XML signature is produced and consumed without a DTD, and points at an
//! attribute spelled `Id`. Both `<ds:Signature Id="…">` and
//! `<ds:Object Id="…">` are declared that way in the XML Signature schema
//! itself, and every interoperating implementation (Apache Santuario's
//! `registerIdAttribute`, .NET's `GetIdElement`, …) hard-codes a list of
//! conventional names to make it work.
//!
//! So does this one, through [`IdIndex`]: the DTD- and `xml:id`-declared
//! identifiers first, then a configurable list of conventional attribute names
//! that defaults to `Id`, `ID` and `id`.
//!
//! # Duplicate identifiers
//!
//! An identifier that names two elements is the raw material of an
//! [XML Signature Wrapping](https://www.ws-attacks.org/XML_Signature_Wrapping)
//! attack: the attacker adds a second element with the same `Id`, arranges for
//! the verifier to digest one and the application to read the other, and the
//! signature still checks out. The defence is to refuse to resolve an ambiguous
//! identifier at all, which is what [`IdIndex`] does by default —
//! see [`Verifier::allow_duplicate_ids`](crate::Verifier::allow_duplicate_ids)
//! for the escape hatch.

use std::collections::BTreeMap;

use oxixml_dom::{Document, NodeId, NodeKind};

use crate::xpath::IdResolver;

/// The attribute names treated as identifiers when nothing declares them.
///
/// `Id` is what the XML Signature schema uses, `ID` and `id` are what other
/// vocabularies (SAML, WS-Security, XAdES) use.
pub const DEFAULT_ID_ATTRIBUTES: [&str; 3] = ["Id", "ID", "id"];

/// An identifier-to-element index over one document.
#[derive(Debug, Clone)]
pub(crate) struct IdIndex {
    unique: BTreeMap<String, NodeId>,
    ambiguous: BTreeMap<String, usize>,
    allow_duplicates: bool,
}

impl IdIndex {
    /// Builds the index.
    ///
    /// `attribute_names` are the conventional names to honour on top of what
    /// the document itself declares.
    pub(crate) fn build(
        document: &Document,
        attribute_names: &[String],
        allow_duplicates: bool,
    ) -> Self {
        let mut counts: BTreeMap<String, Vec<NodeId>> = BTreeMap::new();
        for node in document.nodes_in_document_order() {
            if document.kind(node) != Some(NodeKind::Element) {
                continue;
            }
            for attribute in document.attributes(node) {
                let Some(data) = document.attr(attribute) else {
                    continue;
                };
                let is_identifier = document.is_id_attribute(node, attribute)
                    || (data.prefix().is_none()
                        && attribute_names.iter().any(|name| name == data.local_name()));
                if !is_identifier {
                    continue;
                }
                let entry = counts.entry(data.value().to_owned()).or_default();
                if !entry.contains(&node) {
                    entry.push(node);
                }
            }
        }

        let mut unique = BTreeMap::new();
        let mut ambiguous = BTreeMap::new();
        for (id, nodes) in counts {
            match nodes.len() {
                0 => {}
                1 => {
                    if let Some(node) = nodes.first() {
                        unique.insert(id, *node);
                    }
                }
                count => {
                    if allow_duplicates {
                        if let Some(node) = nodes.first() {
                            unique.insert(id.clone(), *node);
                        }
                    }
                    ambiguous.insert(id, count);
                }
            }
        }

        Self {
            unique,
            ambiguous,
            allow_duplicates,
        }
    }

    /// The element carrying `id`, or [`None`] when there is none — or when
    /// there is more than one and duplicates are not allowed.
    pub(crate) fn get(&self, id: &str) -> Option<NodeId> {
        self.unique.get(id).copied()
    }

    /// How many elements carry `id`, when that is more than one.
    pub(crate) fn duplicate_count(&self, id: &str) -> Option<usize> {
        self.ambiguous.get(id).copied()
    }

    /// Whether duplicates were configured to be tolerated.
    pub(crate) const fn allows_duplicates(&self) -> bool {
        self.allow_duplicates
    }
}

impl IdResolver for IdIndex {
    fn resolve(&self, id: &str) -> Option<NodeId> {
        self.get(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_dom::build;

    fn names() -> Vec<String> {
        DEFAULT_ID_ATTRIBUTES
            .iter()
            .map(|name| (*name).to_owned())
            .collect()
    }

    #[test]
    fn conventional_attribute_names_are_honoured() {
        let document = build::parse_str(
            r##"<r><a Id="one"/><b ID="two"/><c id="three"/><d xml:id="four"/></r>"##,
        )
        .expect("well-formed test input");
        let index = IdIndex::build(&document, &names(), false);
        for id in ["one", "two", "three", "four"] {
            assert!(index.get(id).is_some(), "{id}");
        }
        assert_eq!(index.get("missing"), None);
    }

    #[test]
    fn a_duplicate_identifier_resolves_to_nothing() {
        let document =
            build::parse_str(r##"<r><a Id="x"/><b Id="x"/></r>"##).expect("well-formed test input");
        let strict = IdIndex::build(&document, &names(), false);
        assert_eq!(strict.get("x"), None);
        assert_eq!(strict.duplicate_count("x"), Some(2));

        let lenient = IdIndex::build(&document, &names(), true);
        assert!(lenient.get("x").is_some());
        assert_eq!(lenient.duplicate_count("x"), Some(2));
    }

    #[test]
    fn a_prefixed_attribute_is_not_a_conventional_identifier() {
        let document = build::parse_str(r##"<r xmlns:q="urn:q"><a q:Id="x"/></r>"##)
            .expect("well-formed test input");
        let index = IdIndex::build(&document, &names(), false);
        assert_eq!(index.get("x"), None);
    }

    #[test]
    fn a_declared_id_attribute_is_honoured_without_a_conventional_name() {
        let mut document =
            build::parse_str(r##"<r><a ref="x"/></r>"##).expect("well-formed test input");
        document.declare_id_attribute(Some("a"), "ref");
        let index = IdIndex::build(&document, &[], false);
        assert!(index.get("x").is_some());
    }
}
