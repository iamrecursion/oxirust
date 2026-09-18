//! Element content white space, and the `<!ELEMENT …>` declarations that identify it.
//!
//! [XML 1.0 §3.2.1](https://www.w3.org/TR/xml/#sec-element-content) gives an element type
//! declared with a `children` content model *element content*: child elements "optionally
//! separated by white space". [XML 1.0 §2.10](https://www.w3.org/TR/xml/#sec-white-space) names
//! that separating white space, and constructing an XDM tree from the infoset discards it —
//! [XSLT 3.0 §4.4](https://www.w3.org/TR/xslt-30/#stylesheet-stripping) states the consequence
//! outright: "elements that are defined in a DTD or a Schema to contain element-only content will
//! have whitespace text nodes stripped, regardless of the `xsl:strip-space` and
//! `xsl:preserve-space` declarations in the stylesheet".
//!
//! Every other content specification leaves the white space where the document wrote it: `EMPTY`
//! and `ANY` are not element content, a mixed model allows character data, and a model this
//! parser cannot classify — one assembled out of parameter entities — is not classified at all.

use oxixml_infoset::{
    ExternalEntity, ExternalEntityError, ExternalEntityResolver, NodeKind, ParseOptions, Tree,
    XdmError, XdmNode,
};

/// The children of `<a>` as node kinds, so that a dropped text node is visible as a shorter list.
fn root_children(xml: &str) -> Result<Vec<NodeKind>, XdmError> {
    kinds_of(&Tree::parse_str(xml)?)
}

fn kinds_of(tree: &Tree) -> Result<Vec<NodeKind>, XdmError> {
    Ok(tree
        .document_element()
        .expect("a document element")
        .children()
        .iter()
        .map(XdmNode::node_kind)
        .collect())
}

/// A resolver that answers every request with one fixed external subset.
struct Subset(&'static str);

impl ExternalEntityResolver for Subset {
    fn resolve(
        &mut self,
        _entity: &ExternalEntity<'_>,
    ) -> Result<Option<String>, ExternalEntityError> {
        Ok(Some(self.0.to_owned()))
    }
}

/// Parses `xml` with `dtd` supplied as the external subset of its document type declaration.
fn with_external_subset(xml: &str, dtd: &'static str) -> Result<Tree, XdmError> {
    Tree::parse_str_with_resolver(
        xml,
        ParseOptions::default().read_external_subset(true),
        &mut Subset(dtd),
    )
}

#[test]
fn element_content_whitespace_does_not_reach_the_tree() -> Result<(), XdmError> {
    let kinds = root_children(
        "<!DOCTYPE a [<!ELEMENT a (b*)><!ELEMENT b EMPTY>]>\n<a>\n  <b/>\n  <b/>\n</a>",
    )?;
    assert_eq!(kinds, [NodeKind::Element, NodeKind::Element]);
    Ok(())
}

#[test]
fn the_same_document_without_a_content_model_keeps_its_whitespace() -> Result<(), XdmError> {
    let kinds = root_children("<a>\n  <b/>\n  <b/>\n</a>")?;
    assert_eq!(
        kinds,
        [
            NodeKind::Text,
            NodeKind::Element,
            NodeKind::Text,
            NodeKind::Element,
            NodeKind::Text,
        ]
    );
    Ok(())
}

#[test]
fn a_mixed_model_is_not_element_content() -> Result<(), XdmError> {
    let kinds = root_children(
        "<!DOCTYPE a [<!ELEMENT a (#PCDATA|b)*><!ELEMENT b EMPTY>]>\n<a>\n  <b/>\n</a>",
    )?;
    assert_eq!(kinds, [NodeKind::Text, NodeKind::Element, NodeKind::Text]);
    Ok(())
}

#[test]
fn any_and_empty_models_are_not_element_content() -> Result<(), XdmError> {
    let any =
        root_children("<!DOCTYPE a [<!ELEMENT a ANY><!ELEMENT b EMPTY>]>\n<a>\n  <b/>\n</a>")?;
    assert_eq!(any, [NodeKind::Text, NodeKind::Element, NodeKind::Text]);

    // `<!ELEMENT a EMPTY>` with content at all is a validity error; a non-validating parser
    // reports it by keeping what the document wrote.
    let empty = root_children("<!DOCTYPE a [<!ELEMENT a EMPTY>]>\n<a>\n</a>")?;
    assert_eq!(empty, [NodeKind::Text]);
    Ok(())
}

#[test]
fn a_declaration_in_the_external_subset_counts() -> Result<(), XdmError> {
    // The shape a document type declaration usually has: the content model lives in a DTD file
    // beside the document, so it is only visible once the external subset has been read.
    let xml = "<!DOCTYPE a SYSTEM 'a.dtd'>\n<a>\n  <b/>\n  <b/>\n</a>";
    let tree = with_external_subset(xml, "<!ELEMENT a (b*)>\n<!ELEMENT b EMPTY>")?;
    assert_eq!(kinds_of(&tree)?, [NodeKind::Element, NodeKind::Element]);

    // The same document parsed without reading the subset declares no content model at all.
    assert_eq!(
        root_children(xml)?,
        [
            NodeKind::Text,
            NodeKind::Element,
            NodeKind::Text,
            NodeKind::Element,
            NodeKind::Text,
        ]
    );
    Ok(())
}

#[test]
fn a_content_model_written_with_a_parameter_entity_is_not_classified() -> Result<(), XdmError> {
    // Nothing here expands `%inner;`, so `(%inner;)` could equally be `(b*)` or `(#PCDATA|b)*`.
    // Guessing would strip white space that belongs to the document.
    let tree = with_external_subset(
        "<!DOCTYPE a SYSTEM 'a.dtd'>\n<a>\n  <b/>\n</a>",
        "<!ENTITY % inner \"b*\">\n<!ELEMENT a (%inner;)>\n<!ELEMENT b EMPTY>",
    )?;
    assert_eq!(
        kinds_of(&tree)?,
        [NodeKind::Text, NodeKind::Element, NodeKind::Text]
    );
    Ok(())
}

#[test]
fn character_data_in_element_content_survives() -> Result<(), XdmError> {
    // The run holds a non-whitespace character, so it is character data — a validity error the
    // model forbids, not element content white space, and a non-validating parser keeps it.
    let tree =
        Tree::parse_str("<!DOCTYPE a [<!ELEMENT a (b*)><!ELEMENT b EMPTY>]>\n<a>\n x <b/>\n</a>")?;
    let root = tree.document_element().expect("a document element");
    let children = root.children();
    assert_eq!(children.len(), 2);
    assert_eq!(children[0].node_kind(), NodeKind::Text);
    assert_eq!(children[0].string_value(), "\n x ");
    Ok(())
}

#[test]
fn only_the_declared_element_loses_its_whitespace() -> Result<(), XdmError> {
    // `a` has element content; `b` has a mixed model, so the white space inside `b` stays.
    let tree = Tree::parse_str(
        "<!DOCTYPE a [<!ELEMENT a (b*)><!ELEMENT b (#PCDATA)>]>\n<a>\n  <b>\n</b>\n</a>",
    )?;
    let root = tree.document_element().expect("a document element");
    let children = root.children();
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].node_kind(), NodeKind::Element);
    assert_eq!(children[0].string_value(), "\n");
    Ok(())
}

#[test]
fn a_declaration_matches_the_qualified_name_as_written() -> Result<(), XdmError> {
    // Element declarations are lexical: `<!ELEMENT p:a (p:b*)>` declares the element written
    // `p:a`, and an element written `a` in the same namespace is a different element type here.
    let tree = Tree::parse_str(
        "<!DOCTYPE p:a [<!ELEMENT p:a (p:b*)><!ELEMENT p:b EMPTY>]>\n\
         <p:a xmlns:p='http://example.com/'>\n  <p:b/>\n</p:a>",
    )?;
    let root = tree.document_element().expect("a document element");
    assert_eq!(root.children().len(), 1);

    let undeclared = Tree::parse_str(
        "<!DOCTYPE q:a [<!ELEMENT q:a (q:b*)><!ELEMENT q:b EMPTY>]>\n\
         <p:a xmlns:p='http://example.com/'>\n  <p:b/>\n</p:a>",
    )?;
    assert_eq!(
        undeclared
            .document_element()
            .expect("a document element")
            .children()
            .len(),
        3
    );
    Ok(())
}

#[test]
fn preserving_element_content_whitespace_is_opt_in() -> Result<(), XdmError> {
    let xml = "<!DOCTYPE a [<!ELEMENT a (b*)><!ELEMENT b EMPTY>]>\n<a>\n  <b/>\n</a>";
    let verbatim = Tree::parse_str_with(
        xml,
        ParseOptions::default().preserve_element_content_whitespace(true),
    )?;
    assert_eq!(
        verbatim
            .document_element()
            .expect("a document element")
            .children()
            .len(),
        3
    );
    Ok(())
}

#[test]
fn nested_element_content_is_tracked_per_element() -> Result<(), XdmError> {
    // `a` and `c` have element content, `b` a mixed model: only `b`'s own white space stays,
    // which needs the model of the *innermost* open element rather than the document's.
    let tree = Tree::parse_str(
        "<!DOCTYPE a [<!ELEMENT a (b*)><!ELEMENT b (#PCDATA|c)*><!ELEMENT c (d*)>\
         <!ELEMENT d EMPTY>]>\n<a>\n  <b>\n    <c>\n      <d/>\n    </c>\n  </b>\n</a>",
    )?;
    let root = tree.document_element().expect("a document element");
    assert_eq!(root.children().len(), 1);

    let b = root.children()[0].clone();
    let inside_b: Vec<NodeKind> = b.children().iter().map(XdmNode::node_kind).collect();
    assert_eq!(
        inside_b,
        [NodeKind::Text, NodeKind::Element, NodeKind::Text]
    );

    let c = b.children()[1].clone();
    let inside_c: Vec<NodeKind> = c.children().iter().map(XdmNode::node_kind).collect();
    assert_eq!(inside_c, [NodeKind::Element]);
    Ok(())
}
