//! Which namespaces one schema document is allowed to name.
//!
//! A schema is assembled from several documents, and every global component
//! any of them declares lands in the same arenas. That is *not* a licence for
//! any of them to refer to any of the others: the schema representation
//! constraint *QName resolution (Schema Document)* — [XSD 1.0 §3.15.3][r10],
//! [XSD 1.1 §3.17.6.2][r11] — makes the resolution of a `QName` a question
//! about **the document the QName is written in**, and only that document:
//!
//! > 4 The appropriate case among the following is true:
//! >
//! > 4.1 If the namespace name of the QName is *absent*, then one of the
//! > following must be true:
//! >   4.1.1 the `<schema>` of the schema document containing the QName has no
//! >   `targetNamespace`;
//! >   4.1.2 that `<schema>` contains an `<import>` with no `namespace`.
//! >
//! > 4.2 otherwise the namespace name of the QName is the same as one of the
//! > following:
//! >   4.2.1 the `targetNamespace` of that `<schema>`;
//! >   4.2.2 the `namespace` of some `<import>` contained in that `<schema>`.
//!
//! So the licence is per **document**, never per schema, and it does not
//! travel: `msData/element/elemZ007` puts the `<xs:import>` in an *included*
//! document and the reference in the *including* one and is expected invalid,
//! as is `msData/schema/schZ005`, which is the same shape the other way round.
//! W3C bug 4057 settled that reading. `msData/identityConstraint/idC019` shows
//! the set has to be over an *optional* namespace rather than a string: the
//! reference it makes is to the **absent** namespace, from a document whose
//! `targetNamespace` is `diffNS`, and clause 4.1 refuses it.
//!
//! Three namespaces are licensed everywhere and are not written down in the
//! document at all: the XSD namespace, whose built-in definitions [XSD 1.1
//! §3.17.6.2][r11] calls "always available"; the XML namespace, whose
//! attributes and attribute group this crate supplies out of the box so that a
//! schema naming `xml:lang` need import nothing
//! (`msData/additional/addB187`, `addB188`); and the XSI namespace, whose four
//! attributes §2.6 makes present in every instance.
//!
//! [r10]: https://www.w3.org/TR/xmlschema-1/#src-resolve
//! [r11]: https://www.w3.org/TR/xmlschema11-1/#src-resolve

use std::collections::HashSet;

use oxixml_dom::{Document, NodeId};

use crate::names::{XML_NS, XSD_NS, XSI_NS};
use crate::parse::{attr, child_elements, xsd_name};
use crate::value::collapse;

/// The namespaces one schema document may name in a `QName` reference.
///
/// Held behind an [`Rc`](std::rc::Rc) in
/// [`DocInfo`](crate::compile::DocInfo), which is cloned once per component
/// built.
#[derive(Debug, Clone)]
pub(crate) struct Licence {
    /// The namespace names, clause 4.2.
    named: HashSet<String>,
    /// Whether the *absent* namespace is licensed, clause 4.1.
    absent: bool,
}

impl Licence {
    /// The licence of the document rooted at `root`, whose *effective* target
    /// namespace — the one a chameleon include has already absorbed it into —
    /// is `target_namespace`.
    pub(crate) fn of(doc: &Document, root: NodeId, target_namespace: Option<&str>) -> Self {
        let mut named = HashSet::from([XSD_NS.to_owned(), XML_NS.to_owned(), XSI_NS.to_owned()]);
        // Clause 4.2.1 / 4.1.1: the document's own target namespace, which for
        // a chameleon include is the namespace it was absorbed into — exactly
        // the namespace its unqualified references name.
        let mut absent = target_namespace.is_none();
        if let Some(target) = target_namespace {
            named.insert(target.to_owned());
        }
        // Clauses 4.2.2 and 4.1.2: the document's *own* `xs:import` children.
        // Whether the import found a document is beside the point — an import
        // with no `schemaLocation` still licenses the namespace it names, and
        // that is how a schema says "components of this namespace reach me by
        // some means I am not writing down".
        for child in child_elements(doc, root) {
            if xsd_name(doc, child) != Some("import") {
                continue;
            }
            match attr(doc, child, "namespace") {
                Some(text) => {
                    named.insert(collapse(text));
                }
                None => absent = true,
            }
        }
        Self { named, absent }
    }

    /// Whether a reference to `namespace` is licensed.
    pub(crate) fn allows(&self, namespace: Option<&str>) -> bool {
        match namespace {
            None => self.absent,
            Some(namespace) => self.named.contains(namespace),
        }
    }
}

/// The message the compiler reports when a document names a namespace it was
/// never licensed to name.
pub(crate) fn message(namespace: Option<&str>, uri: &str) -> String {
    match namespace {
        None => format!(
            "`{uri}` has a target namespace and does not import the absent namespace, so it may \
             not refer to an unqualified name"
        ),
        Some(namespace) => format!(
            "`{uri}` neither declares nor imports the namespace `{namespace}`, so it may not \
             refer to a name in it"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn licence(text: &str) -> Licence {
        let doc = oxixml_dom::build::parse_str(text).expect("the fixture parses");
        let root = doc.document_element().expect("the fixture has a root");
        let target = attr(&doc, root, "targetNamespace").map(str::to_owned);
        Licence::of(&doc, root, target.as_deref())
    }

    #[test]
    fn a_document_without_a_target_namespace_licenses_the_absent_one() {
        let licence = licence(r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"/>"#);
        assert!(licence.allows(None));
        assert!(licence.allows(Some(XSD_NS)));
        assert!(!licence.allows(Some("urn:other")));
    }

    #[test]
    fn a_target_namespace_does_not_license_the_absent_one() {
        let licence = licence(
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="urn:own"/>"#,
        );
        assert!(licence.allows(Some("urn:own")));
        assert!(!licence.allows(None));
    }

    #[test]
    fn an_import_licenses_the_namespace_it_names() {
        let licence = licence(
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="urn:own">
                 <xs:import namespace="urn:other"/>
               </xs:schema>"#,
        );
        assert!(licence.allows(Some("urn:other")));
        assert!(!licence.allows(None));
        assert!(!licence.allows(Some("urn:third")));
    }

    #[test]
    fn an_import_without_a_namespace_licenses_the_absent_one() {
        let licence = licence(
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="urn:own">
                 <xs:import/>
               </xs:schema>"#,
        );
        assert!(licence.allows(None));
    }

    #[test]
    fn the_xml_and_xsi_namespaces_need_no_import() {
        let licence = licence(
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="urn:own"/>"#,
        );
        assert!(licence.allows(Some(XML_NS)));
        assert!(licence.allows(Some(XSI_NS)));
    }

    #[test]
    fn a_chameleon_include_is_licensed_for_the_namespace_it_was_absorbed_into() {
        let doc = oxixml_dom::build::parse_str(
            r#"<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"/>"#,
        )
        .expect("the fixture parses");
        let root = doc.document_element().expect("the fixture has a root");
        let licence = Licence::of(&doc, root, Some("urn:includer"));
        assert!(licence.allows(Some("urn:includer")));
        assert!(!licence.allows(None));
    }
}
