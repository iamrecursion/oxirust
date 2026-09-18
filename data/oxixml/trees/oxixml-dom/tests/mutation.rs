//! The DOM Core mutation operations and the exceptions they raise.

use oxixml_dom::{
    Document, DomError, HierarchyRequest, NamespaceViolation, NodeKind, NotFound, XML_NAMESPACE,
    XMLNS_NAMESPACE, build,
};

fn document_with_root() -> Result<(Document, oxixml_dom::NodeId), DomError> {
    let mut doc = Document::new();
    let root = doc.create_element("root")?;
    let document = doc.root();
    doc.append_child(document, root)?;
    Ok((doc, root))
}

#[test]
fn append_child_links_siblings_both_ways() -> Result<(), DomError> {
    let (mut doc, root) = document_with_root()?;
    let a = doc.create_element("a")?;
    let b = doc.create_element("b")?;
    let c = doc.create_element("c")?;
    doc.append_child(root, a)?;
    doc.append_child(root, b)?;
    doc.append_child(root, c)?;

    assert_eq!(doc.children(root).collect::<Vec<_>>(), [a, b, c]);
    assert_eq!(
        doc.children(root).rev().collect::<Vec<_>>(),
        [c, b, a],
        "the child list must be doubly linked"
    );
    assert_eq!(doc.first_child(root), Some(a));
    assert_eq!(doc.last_child(root), Some(c));
    assert_eq!(doc.next_sibling(a), Some(b));
    assert_eq!(doc.previous_sibling(c), Some(b));
    assert_eq!(doc.previous_sibling(a), None);
    assert_eq!(doc.next_sibling(c), None);
    assert_eq!(doc.parent(b), Some(root));
    assert_eq!(doc.child_index(b), Some(1));
    Ok(())
}

#[test]
fn insert_before_places_the_node_at_the_right_position() -> Result<(), DomError> {
    let (mut doc, root) = document_with_root()?;
    let a = doc.create_element("a")?;
    let c = doc.create_element("c")?;
    doc.append_child(root, a)?;
    doc.append_child(root, c)?;

    let b = doc.create_element("b")?;
    doc.insert_before(root, b, Some(c))?;
    assert_eq!(doc.children(root).collect::<Vec<_>>(), [a, b, c]);

    let first = doc.create_element("first")?;
    doc.insert_before(root, first, Some(a))?;
    assert_eq!(doc.children(root).collect::<Vec<_>>(), [first, a, b, c]);
    assert_eq!(doc.first_child(root), Some(first));

    let last = doc.create_element("last")?;
    doc.insert_before(root, last, None)?;
    assert_eq!(doc.last_child(root), Some(last));
    Ok(())
}

#[test]
fn inserting_a_node_that_already_has_a_parent_moves_it() -> Result<(), DomError> {
    let (mut doc, root) = document_with_root()?;
    let a = doc.create_element("a")?;
    let b = doc.create_element("b")?;
    doc.append_child(root, a)?;
    doc.append_child(root, b)?;

    doc.append_child(b, a)?;
    assert_eq!(doc.children(root).collect::<Vec<_>>(), [b]);
    assert_eq!(doc.children(b).collect::<Vec<_>>(), [a]);
    assert_eq!(doc.parent(a), Some(b));
    Ok(())
}

#[test]
fn remove_child_detaches_but_keeps_the_subtree() -> Result<(), DomError> {
    let (mut doc, root) = document_with_root()?;
    let a = doc.create_element("a")?;
    let inner = doc.create_element("inner")?;
    doc.append_child(root, a)?;
    doc.append_child(a, inner)?;

    let removed = doc.remove_child(root, a)?;
    assert_eq!(removed, a);
    assert_eq!(doc.parent(a), None);
    assert!(!doc.has_child_nodes(root));
    assert_eq!(
        doc.children(a).collect::<Vec<_>>(),
        [inner],
        "a detached node keeps its own children"
    );

    // The identifier is still valid and can be re-inserted.
    doc.append_child(root, a)?;
    assert_eq!(doc.children(root).collect::<Vec<_>>(), [a]);
    Ok(())
}

#[test]
fn replace_child_swaps_in_place() -> Result<(), DomError> {
    let (mut doc, root) = document_with_root()?;
    let a = doc.create_element("a")?;
    let b = doc.create_element("b")?;
    let c = doc.create_element("c")?;
    doc.append_child(root, a)?;
    doc.append_child(root, b)?;
    doc.append_child(root, c)?;

    let replacement = doc.create_element("x")?;
    let old = doc.replace_child(root, replacement, b)?;
    assert_eq!(old, b);
    assert_eq!(doc.parent(b), None);
    assert_eq!(doc.children(root).collect::<Vec<_>>(), [a, replacement, c]);
    Ok(())
}

#[test]
fn document_fragments_move_their_children() -> Result<(), DomError> {
    let (mut doc, root) = document_with_root()?;
    let fragment = doc.create_document_fragment();
    let a = doc.create_element("a")?;
    let b = doc.create_element("b")?;
    doc.append_child(fragment, a)?;
    doc.append_child(fragment, b)?;

    doc.append_child(root, fragment)?;
    assert_eq!(doc.children(root).collect::<Vec<_>>(), [a, b]);
    assert!(
        !doc.has_child_nodes(fragment),
        "the fragment must be left empty"
    );
    assert_eq!(doc.parent(a), Some(root));
    Ok(())
}

#[test]
fn a_node_cannot_become_its_own_ancestor() -> Result<(), DomError> {
    let (mut doc, root) = document_with_root()?;
    let child = doc.create_element("child")?;
    doc.append_child(root, child)?;

    let error = doc
        .append_child(child, root)
        .expect_err("a cycle must be rejected");
    assert!(
        matches!(
            error,
            DomError::HierarchyRequest(HierarchyRequest::WouldCreateCycle { .. })
        ),
        "{error}"
    );
    assert_eq!(error.code_name(), "HIERARCHY_REQUEST_ERR");
    assert_eq!(error.code(), 3);

    let error = doc
        .append_child(child, child)
        .expect_err("self-insertion must be rejected");
    assert!(
        matches!(
            error,
            DomError::HierarchyRequest(HierarchyRequest::WouldCreateCycle { .. })
        ),
        "{error}"
    );
    Ok(())
}

#[test]
fn the_document_accepts_only_one_document_element() -> Result<(), DomError> {
    let (mut doc, _root) = document_with_root()?;
    let second = doc.create_element("second")?;
    let document = doc.root();
    let error = doc
        .append_child(document, second)
        .expect_err("a second document element must be rejected");
    assert!(
        matches!(
            error,
            DomError::HierarchyRequest(HierarchyRequest::MultipleDocumentElements)
        ),
        "{error}"
    );
    Ok(())
}

#[test]
fn the_document_rejects_text_children() -> Result<(), DomError> {
    let mut doc = Document::new();
    let text = doc.create_text_node("stray")?;
    let document = doc.root();
    let error = doc
        .append_child(document, text)
        .expect_err("text must not be a child of the document");
    assert!(
        matches!(
            error,
            DomError::HierarchyRequest(HierarchyRequest::KindNotAllowed {
                parent: NodeKind::Document,
                child: NodeKind::Text
            })
        ),
        "{error}"
    );
    Ok(())
}

#[test]
fn the_document_type_must_precede_the_document_element() -> Result<(), DomError> {
    let (mut doc, root) = document_with_root()?;
    let doctype = doc.create_document_type("root", None, Some("root.dtd"))?;
    let document = doc.root();

    let error = doc
        .append_child(document, doctype)
        .expect_err("a document type after the document element must be rejected");
    assert!(
        matches!(
            error,
            DomError::HierarchyRequest(HierarchyRequest::DoctypeAfterDocumentElement)
        ),
        "{error}"
    );

    doc.insert_before(document, doctype, Some(root))?;
    assert_eq!(doc.doctype(), Some(doctype));
    assert_eq!(doc.document_element(), Some(root));

    let second = doc.create_document_type("root", None, None)?;
    let error = doc
        .insert_before(document, second, Some(root))
        .expect_err("a second document type must be rejected");
    assert!(
        matches!(
            error,
            DomError::HierarchyRequest(HierarchyRequest::MultipleDoctypes)
        ),
        "{error}"
    );
    Ok(())
}

#[test]
fn attributes_are_not_children() -> Result<(), DomError> {
    let (mut doc, root) = document_with_root()?;
    let attribute = doc.create_attribute("a")?;
    let error = doc
        .append_child(root, attribute)
        .expect_err("an attribute must not be a child");
    assert!(
        matches!(
            error,
            DomError::HierarchyRequest(HierarchyRequest::AttributeIsNotAChild)
        ),
        "{error}"
    );
    Ok(())
}

#[test]
fn unknown_and_foreign_nodes_are_reported() -> Result<(), DomError> {
    let (mut doc, root) = document_with_root()?;
    let missing = oxixml_dom::NodeId::from_index(9_999);
    assert!(matches!(
        doc.append_child(root, missing),
        Err(DomError::NotFound(NotFound::Node(_)))
    ));

    let orphan = doc.create_element("orphan")?;
    assert!(matches!(
        doc.remove_child(root, orphan),
        Err(DomError::NotFound(NotFound::NotAChild { .. }))
    ));

    let other = doc.create_element("other")?;
    let child = doc.create_element("child")?;
    doc.append_child(root, child)?;
    assert!(
        matches!(
            doc.insert_before(root, other, Some(orphan)),
            Err(DomError::NotFound(NotFound::NotAChild { .. }))
        ),
        "the reference child must belong to the parent"
    );
    Ok(())
}

#[test]
fn import_node_rejects_unknown_and_document_nodes() -> Result<(), DomError> {
    let (mut doc, root) = document_with_root()?;
    let source = doc.clone();

    let imported = doc.import_node(&source, root, true)?;
    assert_eq!(doc.kind(imported), Some(NodeKind::Element));

    assert!(matches!(
        doc.import_node(&source, oxixml_dom::NodeId::from_index(9_999), false),
        Err(DomError::NotFound(NotFound::Node(_)))
    ));
    let source_document = source.root();
    assert!(matches!(
        doc.import_node(&source, source_document, false),
        Err(DomError::NotSupported { .. })
    ));
    Ok(())
}

#[test]
fn invalid_names_and_characters_are_rejected() {
    let mut doc = Document::new();
    assert!(matches!(
        doc.create_element("1bad"),
        Err(DomError::InvalidCharacter { .. })
    ));
    assert!(matches!(
        doc.create_element(""),
        Err(DomError::InvalidCharacter { .. })
    ));
    assert!(matches!(
        doc.create_processing_instruction("xml", "x"),
        Err(DomError::InvalidCharacter { .. })
    ));
    assert!(matches!(
        doc.create_processing_instruction("pi", "a ?> b"),
        Err(DomError::InvalidCharacter { .. })
    ));
    assert!(matches!(
        doc.create_comment("a -- b"),
        Err(DomError::InvalidCharacter { .. })
    ));
    assert!(matches!(
        doc.create_cdata_section("a ]]> b"),
        Err(DomError::InvalidCharacter { .. })
    ));
    assert!(matches!(
        doc.create_text_node("bell\u{7}"),
        Err(DomError::InvalidCharacter { .. })
    ));
}

#[test]
fn namespace_constraints_are_enforced() {
    let mut doc = Document::new();
    assert!(matches!(
        doc.create_element_ns(None, "p:a"),
        Err(DomError::Namespace(
            NamespaceViolation::PrefixWithoutNamespace { .. }
        ))
    ));
    assert!(matches!(
        doc.create_element_ns(Some("urn:x"), "xml:a"),
        Err(DomError::Namespace(
            NamespaceViolation::ReservedPrefixXml { .. }
        ))
    ));
    assert!(matches!(
        doc.create_element_ns(Some(XMLNS_NAMESPACE), "xmlns:a"),
        Err(DomError::Namespace(
            NamespaceViolation::XmlnsPrefixOnElement
        ))
    ));
    assert!(matches!(
        doc.create_element_ns(Some(XMLNS_NAMESPACE), "a"),
        Err(DomError::Namespace(
            NamespaceViolation::ReservedNamespaceXmlns { .. }
        ))
    ));
    assert!(matches!(
        doc.create_attribute_ns(Some("urn:x"), "xmlns:a"),
        Err(DomError::Namespace(
            NamespaceViolation::ReservedPrefixXmlns { .. }
        ))
    ));
    assert!(matches!(
        doc.create_element_ns(Some("urn:x"), "a:b:c"),
        Err(DomError::Namespace(
            NamespaceViolation::MalformedQualifiedName { .. }
        ))
    ));
    // The reserved combinations that *are* legal.
    assert!(doc.create_element_ns(Some(XML_NAMESPACE), "xml:a").is_ok());
    assert!(
        doc.create_attribute_ns(Some(XMLNS_NAMESPACE), "xmlns:a")
            .is_ok()
    );
    assert!(
        doc.create_attribute_ns(Some(XMLNS_NAMESPACE), "xmlns")
            .is_ok()
    );
}

#[test]
fn attributes_can_be_set_read_and_removed() -> Result<(), DomError> {
    let (mut doc, root) = document_with_root()?;
    let a = doc.set_attribute(root, "one", "1")?;
    doc.set_attribute(root, "two", "2")?;
    assert_eq!(doc.get_attribute(root, "one"), Some("1"));
    assert_eq!(doc.attributes(root).count(), 2);
    assert!(doc.has_attribute(root, "two"));
    assert!(!doc.has_attribute(root, "three"));

    // Setting an existing attribute replaces the value and does not add a node.
    let again = doc.set_attribute(root, "one", "1'")?;
    assert_eq!(again, a);
    assert_eq!(doc.get_attribute(root, "one"), Some("1'"));
    assert_eq!(doc.attributes(root).count(), 2);

    let removed = doc.remove_attribute(root, "one")?;
    assert_eq!(removed, Some(a));
    assert_eq!(doc.attributes(root).count(), 1);
    assert_eq!(doc.owner_element(a), None);
    assert_eq!(doc.remove_attribute(root, "missing")?, None);
    Ok(())
}

#[test]
fn namespaced_attributes_are_keyed_on_the_expanded_name() -> Result<(), DomError> {
    let (mut doc, root) = document_with_root()?;
    doc.declare_namespace(root, Some("p"), "urn:p")?;
    doc.set_attribute_ns(root, Some("urn:p"), "p:x", "1")?;
    doc.set_attribute_ns(root, None, "x", "2")?;

    assert_eq!(doc.get_attribute_ns(root, Some("urn:p"), "x"), Some("1"));
    assert_eq!(doc.get_attribute_ns(root, None, "x"), Some("2"));
    assert_eq!(doc.get_attribute(root, "p:x"), Some("1"));
    assert_eq!(doc.get_attribute(root, "x"), Some("2"));

    // Re-setting with a different prefix updates the prefix, not the identity.
    doc.declare_namespace(root, Some("q"), "urn:p")?;
    let node = doc.set_attribute_ns(root, Some("urn:p"), "q:x", "3")?;
    assert_eq!(doc.get_attribute_ns(root, Some("urn:p"), "x"), Some("3"));
    assert_eq!(
        doc.attr(node).map(|data| data.qualified_name()),
        Some("q:x".to_owned())
    );

    doc.remove_attribute_ns(root, Some("urn:p"), "x")?;
    assert_eq!(doc.get_attribute_ns(root, Some("urn:p"), "x"), None);
    assert_eq!(doc.get_attribute_ns(root, None, "x"), Some("2"));
    Ok(())
}

#[test]
fn an_attribute_node_cannot_be_shared_between_elements() -> Result<(), DomError> {
    let (mut doc, root) = document_with_root()?;
    let other = doc.create_element("other")?;
    doc.append_child(root, other)?;

    let attribute = doc.create_attribute("a")?;
    doc.set_attribute_node(root, attribute)?;
    let error = doc
        .set_attribute_node(other, attribute)
        .expect_err("an attribute in use must be rejected");
    assert!(matches!(error, DomError::InUseAttribute { .. }), "{error}");
    assert_eq!(error.code_name(), "INUSE_ATTRIBUTE_ERR");
    Ok(())
}

#[test]
fn normalize_merges_adjacent_text_and_drops_empty_nodes() -> Result<(), DomError> {
    let (mut doc, root) = document_with_root()?;
    for part in ["a", "", "b", "c"] {
        let text = doc.create_text_node(part)?;
        doc.append_child(root, text)?;
    }
    let element = doc.create_element("e")?;
    doc.append_child(root, element)?;
    for part in ["", "d", ""] {
        let text = doc.create_text_node(part)?;
        doc.append_child(root, text)?;
    }
    let inner_a = doc.create_text_node("x")?;
    let inner_b = doc.create_text_node("y")?;
    doc.append_child(element, inner_a)?;
    doc.append_child(element, inner_b)?;

    assert_eq!(doc.children(root).count(), 8);
    doc.normalize(doc.root())?;

    let children: Vec<_> = doc.children(root).collect();
    assert_eq!(children.len(), 3);
    assert_eq!(doc.character_data(children[0]), Some("abc"));
    assert_eq!(doc.kind(children[1]), Some(NodeKind::Element));
    assert_eq!(doc.character_data(children[2]), Some("d"));
    assert_eq!(doc.children(element).count(), 1);
    assert_eq!(doc.text_content(element).as_deref(), Some("xy"));
    Ok(())
}

#[test]
fn normalize_leaves_cdata_sections_alone() -> Result<(), Box<dyn std::error::Error>> {
    let mut doc = build::parse_str("<a>x<![CDATA[y]]>z</a>")?;
    doc.normalize(doc.root())?;
    let root = doc.document_element().expect("a document element");
    let kinds: Vec<_> = doc
        .children(root)
        .filter_map(|node| doc.kind(node))
        .collect();
    assert_eq!(
        kinds,
        [NodeKind::Text, NodeKind::CdataSection, NodeKind::Text]
    );
    Ok(())
}

#[test]
fn clone_node_is_shallow_or_deep() -> Result<(), Box<dyn std::error::Error>> {
    let doc_source = build::parse_str(r#"<a p="1" xmlns:n="urn:n" n:q="2"><b>text</b><c/></a>"#)?;
    let mut doc = doc_source;
    let root = doc.document_element().expect("a document element");

    let shallow = doc.clone_node(root, false)?;
    assert_eq!(doc.parent(shallow), None);
    assert!(
        !doc.has_child_nodes(shallow),
        "a shallow clone has no children"
    );
    assert_eq!(
        doc.attributes(shallow).count(),
        3,
        "a shallow clone still copies every attribute"
    );
    assert_eq!(doc.get_attribute(shallow, "p"), Some("1"));
    assert_eq!(doc.get_attribute_ns(shallow, Some("urn:n"), "q"), Some("2"));

    let deep = doc.clone_node(root, true)?;
    assert!(doc.is_equal_node(root, &doc, deep));
    assert_ne!(root, deep, "the clone is a distinct node");
    assert_eq!(doc.text_content(deep).as_deref(), Some("text"));

    // Mutating the clone must not affect the original.
    let cloned_b = doc.first_child(deep).expect("the cloned b element");
    doc.set_text_content(cloned_b, "changed")?;
    assert_eq!(doc.text_content(root).as_deref(), Some("text"));
    assert_eq!(doc.text_content(deep).as_deref(), Some("changed"));
    Ok(())
}

#[test]
fn import_node_copies_between_documents() -> Result<(), Box<dyn std::error::Error>> {
    let source = build::parse_str(r#"<a xmlns:n="urn:n"><n:b n:x="1"><c/></n:b></a>"#)?;
    let source_root = source.document_element().expect("a document element");
    let b = source.first_child(source_root).expect("the n:b element");

    let mut target = Document::new();
    let imported = target.import_node(&source, b, true)?;
    let target_root = target.root();
    target.append_child(target_root, imported)?;

    assert!(source.is_equal_node(b, &target, imported));
    assert_eq!(
        target.get_attribute_ns(imported, Some("urn:n"), "x"),
        Some("1")
    );
    assert_eq!(
        target.to_xml(),
        "<n:b xmlns:n=\"urn:n\" n:x=\"1\"><c/></n:b>"
    );

    Ok(())
}

#[test]
fn set_text_content_replaces_the_children() -> Result<(), Box<dyn std::error::Error>> {
    let mut doc = build::parse_str("<a><b/><c>x</c></a>")?;
    let root = doc.document_element().expect("a document element");
    doc.set_text_content(root, "replaced")?;
    assert_eq!(doc.children(root).count(), 1);
    assert_eq!(doc.to_xml(), "<a>replaced</a>");

    doc.set_text_content(root, "")?;
    assert_eq!(doc.children(root).count(), 0);
    assert_eq!(doc.to_xml(), "<a/>");
    Ok(())
}

#[test]
fn text_content_follows_the_dom_table() -> Result<(), Box<dyn std::error::Error>> {
    let doc = build::parse_str("<a>x<!--c--><?p d?><b>y</b><![CDATA[z]]></a>")?;
    let root = doc.document_element().expect("a document element");
    assert_eq!(doc.text_content(root).as_deref(), Some("xyz"));
    assert_eq!(
        doc.text_content(doc.root()),
        None,
        "the document node has no text content"
    );

    let comment = doc
        .children(root)
        .find(|&node| doc.kind(node) == Some(NodeKind::Comment))
        .expect("a comment");
    assert_eq!(doc.text_content(comment).as_deref(), Some("c"));

    let pi = doc
        .children(root)
        .find(|&node| doc.kind(node) == Some(NodeKind::ProcessingInstruction))
        .expect("a processing instruction");
    assert_eq!(doc.text_content(pi).as_deref(), Some("d"));
    Ok(())
}

#[test]
fn node_names_and_values_follow_the_dom_table() -> Result<(), Box<dyn std::error::Error>> {
    let doc = build::parse_str(
        "<!DOCTYPE root><root p:a=\"v\" xmlns:p=\"urn:p\">t<![CDATA[c]]><!--m--><?pi d?></root>",
    )?;
    let root = doc.document_element().expect("a document element");
    assert_eq!(doc.node_name(doc.root()).as_deref(), Some("#document"));
    assert_eq!(doc.node_name(root).as_deref(), Some("root"));
    assert_eq!(
        doc.node_name(doc.doctype().expect("a doctype")).as_deref(),
        Some("root")
    );

    let attribute = doc
        .get_attribute_node(root, "p:a")
        .expect("the p:a attribute");
    assert_eq!(doc.node_name(attribute).as_deref(), Some("p:a"));
    assert_eq!(doc.node_value(attribute), Some("v"));

    let children: Vec<_> = doc.children(root).collect();
    assert_eq!(doc.node_name(children[0]).as_deref(), Some("#text"));
    assert_eq!(
        doc.node_name(children[1]).as_deref(),
        Some("#cdata-section")
    );
    assert_eq!(doc.node_name(children[2]).as_deref(), Some("#comment"));
    assert_eq!(doc.node_name(children[3]).as_deref(), Some("pi"));
    assert_eq!(doc.node_value(children[3]), Some("d"));
    Ok(())
}

#[test]
fn detach_removes_attributes_from_their_element() -> Result<(), DomError> {
    let (mut doc, root) = document_with_root()?;
    let attribute = doc.set_attribute(root, "a", "1")?;
    doc.detach(attribute)?;
    assert_eq!(doc.attributes(root).count(), 0);
    assert_eq!(doc.owner_element(attribute), None);

    let error = doc
        .detach(doc.root())
        .expect_err("the document node cannot be detached");
    assert!(matches!(error, DomError::HierarchyRequest(_)), "{error}");
    Ok(())
}

#[test]
fn get_element_by_id_honours_every_source_of_id_ness() -> Result<(), Box<dyn std::error::Error>> {
    // 1. `xml:id` is always an ID.
    let doc = build::parse_str(r#"<a><b xml:id="one"/></a>"#)?;
    let b = doc
        .get_elements_by_tag_name(doc.root(), "b")
        .next()
        .expect("the b element");
    assert_eq!(doc.get_element_by_id("one"), Some(b));

    // 2. An `<!ATTLIST … ID …>` declaration in the internal subset.
    let doc = build::parse_str(
        "<!DOCTYPE a [<!ATTLIST b key ID #IMPLIED>]><a><b key=\"two\"/><c key=\"two\"/></a>",
    )?;
    let b = doc
        .get_elements_by_tag_name(doc.root(), "b")
        .next()
        .expect("the b element");
    assert_eq!(
        doc.get_element_by_id("two"),
        Some(b),
        "only the declared element/attribute pair counts"
    );

    // 3. An explicit API call.
    let mut doc = build::parse_str(r#"<a><b key="three"/></a>"#)?;
    assert_eq!(doc.get_element_by_id("three"), None);
    let b = doc
        .get_elements_by_tag_name(doc.root(), "b")
        .next()
        .expect("the b element");
    let attribute = doc.get_attribute_node(b, "key").expect("the key attribute");
    doc.set_id_attribute_node(attribute, true)?;
    assert_eq!(doc.get_element_by_id("three"), Some(b));
    Ok(())
}

#[test]
fn get_elements_by_tag_name_supports_wildcards() -> Result<(), Box<dyn std::error::Error>> {
    let doc = build::parse_str(
        r#"<r xmlns:p="urn:p" xmlns:q="urn:q"><p:x/><q:x/><y/><p:y><p:x/></p:y></r>"#,
    )?;
    let root = doc.root();

    assert_eq!(doc.get_elements_by_tag_name(root, "p:x").count(), 2);
    assert_eq!(doc.get_elements_by_tag_name(root, "*").count(), 6);
    assert_eq!(
        doc.get_elements_by_tag_name_ns(root, Some("urn:p"), "x")
            .count(),
        2
    );
    assert_eq!(
        doc.get_elements_by_tag_name_ns(root, Some("*"), "x")
            .count(),
        3
    );
    assert_eq!(
        doc.get_elements_by_tag_name_ns(root, Some("urn:p"), "*")
            .count(),
        3
    );
    assert_eq!(
        doc.get_elements_by_tag_name_ns(root, None, "y").count(),
        1,
        "an element in no namespace"
    );

    // Searching from an element excludes the element itself.
    let p_y = doc
        .get_elements_by_tag_name(root, "p:y")
        .next()
        .expect("p:y");
    assert_eq!(doc.get_elements_by_tag_name(p_y, "p:y").count(), 0);
    assert_eq!(doc.get_elements_by_tag_name(p_y, "p:x").count(), 1);
    Ok(())
}

#[test]
fn structural_equality_reports_where_trees_differ() -> Result<(), Box<dyn std::error::Error>> {
    let a = build::parse_str("<r><x/><y>t</y></r>")?;
    let b = build::parse_str("<r><x/><y>u</y></r>")?;
    assert!(!a.structural_eq(&b));
    let diff = a.structural_diff(&b).expect("a difference");
    assert!(diff.contains("character data differs"), "{diff}");

    let c = build::parse_str("<r><x/><y>t</y></r>")?;
    assert!(a.structural_eq(&c));
    assert_eq!(a.structural_diff(&c), None);

    let d = build::parse_str(r#"<r a="1" b="2"/>"#)?;
    let e = build::parse_str(r#"<r b="2" a="1"/>"#)?;
    assert!(
        d.structural_eq(&e),
        "attribute order is not part of the tree"
    );
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Regression: a node must never be linked before itself (WHATWG DOM pre-insert)
////////////////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn insert_before_with_the_node_itself_as_reference_is_a_no_op()
-> Result<(), Box<dyn std::error::Error>> {
    // WHATWG DOM "pre-insert" step 3: if referenceChild is node, use node's next
    // sibling instead. Linking `b` before `b` used to produce a self-referential
    // child list whose traversal (and therefore `to_xml`) never terminated.
    let mut doc = build::parse_str("<r><a/><b/></r>")?;
    let root = doc.document_element().expect("a document element");
    let children: Vec<_> = doc.children(root).collect();
    let b = children[1];

    doc.insert_before(root, b, Some(b))?;

    assert_eq!(doc.children(root).take(10).collect::<Vec<_>>(), children);
    assert_eq!(doc.next_sibling(b), None);
    assert_eq!(doc.previous_sibling(b), Some(children[0]));
    assert_eq!(doc.first_child(root), Some(children[0]));
    assert_eq!(doc.last_child(root), Some(b));
    assert_eq!(doc.to_xml(), "<r><a/><b/></r>");
    Ok(())
}

#[test]
fn replace_child_with_the_next_sibling_keeps_the_list_intact()
-> Result<(), Box<dyn std::error::Error>> {
    // `replace_child(root, new = b, old = a)` where `b` is `a`'s next sibling:
    // the reference computed from `a` *is* `b`, so the WHATWG "replace" step 6
    // adjustment has to kick in or `b` is linked before itself.
    let mut doc = build::parse_str("<r><a/><b/></r>")?;
    let root = doc.document_element().expect("a document element");
    let children: Vec<_> = doc.children(root).collect();
    let (a, b) = (children[0], children[1]);

    let removed = doc.replace_child(root, b, a)?;

    assert_eq!(removed, a);
    assert_eq!(doc.parent(a), None);
    assert_eq!(doc.children(root).take(10).collect::<Vec<_>>(), vec![b]);
    assert_eq!(doc.first_child(root), Some(b));
    assert_eq!(doc.last_child(root), Some(b));
    assert_eq!(doc.next_sibling(b), None);
    assert_eq!(doc.previous_sibling(b), None);
    assert_eq!(doc.to_xml(), "<r><b/></r>");
    Ok(())
}

#[test]
fn repeated_set_attribute_ns_does_not_grow_the_arena() -> Result<(), Box<dyn std::error::Error>> {
    let mut doc = Document::new();
    let element = doc.create_element_ns(Some("urn:n"), "p:e")?;
    let document = doc.root();
    doc.append_child(document, element)?;
    let before = doc.len();

    for index in 0..1000 {
        doc.set_attribute_ns(element, Some("urn:n"), "p:x", index.to_string())?;
    }

    assert_eq!(doc.attributes(element).count(), 1);
    assert_eq!(
        doc.len(),
        before + 1,
        "updating one attribute must allocate exactly one node"
    );
    Ok(())
}

#[test]
fn a_fragment_may_not_carry_a_second_document_element() -> Result<(), DomError> {
    let (mut doc, _root) = document_with_root()?;
    let document = doc.root();
    let fragment = doc.create_document_fragment();
    let a = doc.create_element("a")?;
    doc.append_child(fragment, a)?;

    // One element still fits only because the document already has one.
    let error = doc
        .append_child(document, fragment)
        .expect_err("a second document element must be rejected");
    assert!(
        matches!(
            error,
            DomError::HierarchyRequest(HierarchyRequest::MultipleDocumentElements)
        ),
        "{error}"
    );
    Ok(())
}

#[test]
fn a_fragment_holding_two_elements_is_rejected_at_the_document_node() -> Result<(), DomError> {
    let mut doc = Document::new();
    let document = doc.root();
    let fragment = doc.create_document_fragment();
    let a = doc.create_element("a")?;
    let b = doc.create_element("b")?;
    doc.append_child(fragment, a)?;
    doc.append_child(fragment, b)?;

    let error = doc
        .append_child(document, fragment)
        .expect_err("a fragment with two document elements must be rejected");
    assert!(
        matches!(
            error,
            DomError::HierarchyRequest(HierarchyRequest::MultipleDocumentElements)
        ),
        "{error}"
    );
    assert_eq!(
        doc.children(document).count(),
        0,
        "a rejected insertion must leave the tree untouched"
    );

    // The same fragment is fine anywhere an element may repeat.
    let root = doc.create_element("root")?;
    doc.append_child(document, root)?;
    doc.append_child(root, fragment)?;
    assert_eq!(doc.to_xml(), "<root><a/><b/></root>");
    Ok(())
}

#[test]
fn replace_child_rejects_a_multi_element_fragment_too() -> Result<(), DomError> {
    let (mut doc, root) = document_with_root()?;
    let document = doc.root();
    let fragment = doc.create_document_fragment();
    let a = doc.create_element("a")?;
    let b = doc.create_element("b")?;
    doc.append_child(fragment, a)?;
    doc.append_child(fragment, b)?;

    let error = doc
        .replace_child(document, fragment, root)
        .expect_err("replacing the document element with two elements must be rejected");
    assert!(
        matches!(
            error,
            DomError::HierarchyRequest(HierarchyRequest::MultipleDocumentElements)
        ),
        "{error}"
    );
    assert_eq!(doc.children(document).collect::<Vec<_>>(), [root]);
    Ok(())
}

#[test]
fn a_fragment_cannot_hold_a_document_type_declaration() -> Result<(), DomError> {
    // DOM Level 3 Core's content model does not list DocumentType among a
    // fragment's children, so the prolog-ordering constraints of the document
    // node can never be reached through one.
    let mut doc = Document::new();
    let fragment = doc.create_document_fragment();
    let doctype = doc.create_document_type("html", None, None)?;
    let error = doc
        .append_child(fragment, doctype)
        .expect_err("a fragment must not accept a document type declaration");
    assert!(
        matches!(
            error,
            DomError::HierarchyRequest(HierarchyRequest::KindNotAllowed {
                parent: NodeKind::DocumentFragment,
                child: NodeKind::DocumentType,
            })
        ),
        "{error}"
    );
    Ok(())
}

#[test]
fn a_fragment_in_prolog_order_is_accepted_at_the_document_node() -> Result<(), DomError> {
    let mut doc = Document::new();
    let document = doc.root();
    let doctype = doc.create_document_type("html", None, None)?;
    doc.append_child(document, doctype)?;

    let fragment = doc.create_document_fragment();
    let comment = doc.create_comment("prolog")?;
    let element = doc.create_element("html")?;
    doc.append_child(fragment, comment)?;
    doc.append_child(fragment, element)?;

    doc.append_child(document, fragment)?;
    assert_eq!(
        doc.children(document)
            .filter_map(|node| doc.kind(node))
            .collect::<Vec<_>>(),
        [NodeKind::DocumentType, NodeKind::Comment, NodeKind::Element]
    );
    Ok(())
}
