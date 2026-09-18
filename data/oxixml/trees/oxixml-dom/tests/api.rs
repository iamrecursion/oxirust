//! The smaller pieces of the public surface: node metadata, the error
//! vocabulary, the writer entry points and the arena's own invariants.

use std::fmt::Write as _;

use oxixml_dom::{Document, DomError, NodeId, NodeKind, SerializeError, SerializeOptions, build};

#[test]
fn node_kinds_carry_the_dom_node_type_constants() {
    for (kind, node_type, name) in [
        (NodeKind::Element, 1, "element"),
        (NodeKind::Attr, 2, "attribute"),
        (NodeKind::Text, 3, "text"),
        (NodeKind::CdataSection, 4, "CDATA section"),
        (NodeKind::EntityReference, 5, "entity reference"),
        (NodeKind::ProcessingInstruction, 7, "processing instruction"),
        (NodeKind::Comment, 8, "comment"),
        (NodeKind::Document, 9, "document"),
        (NodeKind::DocumentType, 10, "document type"),
        (NodeKind::DocumentFragment, 11, "document fragment"),
    ] {
        assert_eq!(kind.node_type(), node_type, "{kind}");
        assert_eq!(kind.name(), name);
        assert_eq!(kind.to_string(), name);
    }

    assert!(NodeKind::Text.is_character_data());
    assert!(NodeKind::CdataSection.is_character_data());
    assert!(NodeKind::Comment.is_character_data());
    assert!(!NodeKind::Element.is_character_data());
    assert!(!NodeKind::Document.can_be_child());
    assert!(!NodeKind::Attr.can_be_child());
    assert!(NodeKind::Element.can_be_child());
}

#[test]
fn the_content_model_matches_dom_core() {
    use NodeKind::{
        Attr, CdataSection, Comment, Document as Doc, DocumentFragment, DocumentType, Element,
        EntityReference, ProcessingInstruction, Text,
    };

    assert!(Document::can_contain(Doc, Element));
    assert!(Document::can_contain(Doc, Comment));
    assert!(Document::can_contain(Doc, ProcessingInstruction));
    assert!(Document::can_contain(Doc, DocumentType));
    assert!(!Document::can_contain(Doc, Text));
    assert!(!Document::can_contain(Doc, CdataSection));

    for child in [
        Element,
        Text,
        CdataSection,
        Comment,
        ProcessingInstruction,
        EntityReference,
    ] {
        assert!(Document::can_contain(Element, child), "{child}");
        assert!(Document::can_contain(DocumentFragment, child), "{child}");
        assert!(Document::can_contain(EntityReference, child), "{child}");
    }
    assert!(!Document::can_contain(Element, DocumentType));
    assert!(!Document::can_contain(Element, Attr));

    for parent in [
        Attr,
        DocumentType,
        Text,
        CdataSection,
        Comment,
        ProcessingInstruction,
    ] {
        assert!(!Document::can_contain(parent, Element), "{parent}");
        assert!(!Document::can_contain(parent, Text), "{parent}");
    }
}

#[test]
fn node_identifiers_are_transparent() {
    assert_eq!(NodeId::DOCUMENT.index(), 0);
    assert_eq!(NodeId::DOCUMENT.get(), 0);
    assert_eq!(NodeId::from_index(7).index(), 7);
    assert_eq!(NodeId::from_index(7).to_string(), "7");
    assert!(NodeId::from_index(1) < NodeId::from_index(2));
}

#[test]
fn a_fresh_document_holds_only_its_document_node() {
    let doc = Document::new();
    assert_eq!(doc.len(), 1);
    assert!(!doc.is_empty());
    assert!(doc.contains(doc.root()));
    assert!(!doc.contains(NodeId::from_index(1)));
    assert_eq!(doc.kind(doc.root()), Some(NodeKind::Document));
    assert_eq!(doc.document_element(), None);
    assert_eq!(doc.doctype(), None);
    assert_eq!(doc.to_xml(), "");
    assert_eq!(Document::default().len(), 1);
}

#[test]
fn the_arena_never_reuses_a_slot() -> Result<(), DomError> {
    let mut doc = Document::new();
    let root = doc.create_element("r")?;
    let document = doc.root();
    doc.append_child(document, root)?;
    let before = doc.len();

    let child = doc.create_element("c")?;
    doc.append_child(root, child)?;
    doc.remove_child(root, child)?;
    assert_eq!(doc.len(), before + 1, "the removed node keeps its slot");
    assert!(doc.contains(child));
    assert_eq!(doc.kind(child), Some(NodeKind::Element));
    Ok(())
}

#[test]
fn error_codes_match_the_dom_exception_table() {
    let cases: &[(DomError, u16, &str)] = &[
        (
            DomError::IndexSize {
                index: 1,
                length: 0,
            },
            1,
            "INDEX_SIZE_ERR",
        ),
        (
            DomError::WrongDocument { detail: "x" },
            4,
            "WRONG_DOCUMENT_ERR",
        ),
        (
            DomError::InvalidCharacter {
                production: "Name",
                value: "1".to_owned(),
            },
            5,
            "INVALID_CHARACTER_ERR",
        ),
        (
            DomError::NotSupported { detail: "x" },
            9,
            "NOT_SUPPORTED_ERR",
        ),
        (
            DomError::InvalidState { detail: "x" },
            11,
            "INVALID_STATE_ERR",
        ),
        (
            DomError::InvalidModification { detail: "x" },
            13,
            "INVALID_MODIFICATION_ERR",
        ),
    ];
    for (error, code, name) in cases {
        assert_eq!(error.code(), *code, "{name}");
        assert_eq!(error.code_name(), *name);
        assert!(error.to_string().starts_with(name), "{error}");
    }
}

#[test]
fn documents_display_as_xml() -> Result<(), Box<dyn std::error::Error>> {
    let doc = build::parse_str("<a><b>x</b></a>")?;
    let mut out = String::new();
    write!(out, "{doc}")?;
    assert_eq!(out, "<a><b>x</b></a>");
    assert_eq!(doc.to_string(), doc.to_xml());
    Ok(())
}

#[test]
fn serializing_an_unknown_node_is_an_error() -> Result<(), Box<dyn std::error::Error>> {
    let doc = build::parse_str("<a/>")?;
    let mut out = String::new();
    let error = doc
        .write_node(
            NodeId::from_index(9_999),
            &mut out,
            &SerializeOptions::new(),
        )
        .expect_err("an unknown node must be rejected");
    assert!(matches!(error, SerializeError::UnknownNode(_)), "{error}");
    assert!(error.to_string().contains("9999"));

    let mut bytes: Vec<u8> = Vec::new();
    assert!(
        doc.write_node_io(
            NodeId::from_index(9_999),
            &mut bytes,
            &SerializeOptions::new()
        )
        .is_err()
    );
    assert!(bytes.is_empty());
    Ok(())
}

#[test]
fn io_errors_are_propagated() -> Result<(), Box<dyn std::error::Error>> {
    /// A sink that refuses to accept anything.
    struct Broken;
    impl std::io::Write for Broken {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("nope"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let doc = build::parse_str("<a/>")?;
    let error = doc
        .write_xml_io(Broken, &SerializeOptions::new())
        .expect_err("the broken sink must surface");
    assert!(matches!(error, SerializeError::Io(_)), "{error}");
    assert!(std::error::Error::source(&error).is_some());
    Ok(())
}

#[test]
fn node_values_can_be_replaced() -> Result<(), Box<dyn std::error::Error>> {
    let mut doc = build::parse_str("<a>text<!--c--><?p d?><![CDATA[raw]]></a>")?;
    let root = doc.document_element().expect("a document element");
    let children: Vec<_> = doc.children(root).collect();

    doc.set_node_value(children[0], "changed")?;
    doc.set_node_value(children[1], "comment")?;
    doc.set_node_value(children[2], "data")?;
    doc.set_node_value(children[3], "cdata")?;
    assert_eq!(
        doc.to_xml(),
        "<a>changed<!--comment--><?p data?><![CDATA[cdata]]></a>"
    );

    assert!(matches!(
        doc.set_node_value(children[1], "bad -- comment"),
        Err(DomError::InvalidCharacter { .. })
    ));
    assert!(matches!(
        doc.set_node_value(root, "x"),
        Err(DomError::NotSupported { .. })
    ));
    Ok(())
}

#[test]
fn internal_subsets_are_preserved_verbatim() -> Result<(), Box<dyn std::error::Error>> {
    let xml = "<!DOCTYPE a PUBLIC \"-//X//DTD a//EN\" \"a.dtd\" \
               [<!ELEMENT a EMPTY><!ENTITY e \"v\">]><a/>";
    let doc = build::parse_str(xml)?;
    let doctype = doc.doctype().expect("a document type");
    let data = doc.doctype_data(doctype).expect("the document type data");
    assert_eq!(data.name(), "a");
    assert_eq!(data.public_id(), Some("-//X//DTD a//EN"));
    assert_eq!(data.system_id(), Some("a.dtd"));
    assert_eq!(
        data.internal_subset(),
        Some("<!ELEMENT a EMPTY><!ENTITY e \"v\">")
    );
    assert_eq!(doc.to_xml(), xml);
    Ok(())
}

#[test]
fn the_builder_rejects_unbalanced_streams() {
    let mut builder = oxixml_dom::DocumentBuilder::new();
    builder
        .push_element_start(None, "a")
        .expect("the document element");
    assert!(builder.finish().is_err(), "an open element must be caught");

    let mut builder = oxixml_dom::DocumentBuilder::new();
    assert!(
        builder.push_element_end().is_err(),
        "a stray end tag must be caught"
    );

    let mut builder = oxixml_dom::DocumentBuilder::new();
    assert!(
        builder.push_attribute(None, "a", "1").is_err(),
        "an attribute needs an open element"
    );
}

#[test]
fn the_builder_exposes_the_document_under_construction() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = oxixml_dom::DocumentBuilder::new();
    builder.set_declaration(Some("1.0"), Some("UTF-8"), Some(true));
    let root = builder.push_element_start(None, "a")?;
    assert_eq!(builder.current_element(), Some(root));
    assert_eq!(builder.document().kind(root), Some(NodeKind::Element));
    builder.push_comment("c")?;
    builder.push_processing_instruction("pi", "d")?;
    builder.push_cdata("raw")?;
    builder.push_entity_reference("e")?;
    builder.push_text("t");
    builder.push_element_end()?;
    assert_eq!(builder.current_element(), None);

    let doc = builder.finish()?;
    assert_eq!(doc.xml_version(), Some("1.0"));
    assert_eq!(doc.encoding(), Some("UTF-8"));
    assert_eq!(doc.standalone(), Some(true));
    assert_eq!(doc.to_xml(), "<a><!--c--><?pi d?><![CDATA[raw]]>&e;t</a>");
    Ok(())
}
