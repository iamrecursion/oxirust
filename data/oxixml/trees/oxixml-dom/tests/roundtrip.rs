//! Build → serialize → re-parse → structural equality over a hand-written
//! corpus, plus the exact serialized form for the cases where the output shape
//! itself matters (escaping, attribute quoting, empty-element folding, namespace
//! declarations).

use oxixml_dom::{
    BuildError, Document, EntityPolicy, NodeKind, SerializeOptions, XMLNS_NAMESPACE,
    XmlDeclaration, build,
};

/// The corpus every round-trip property is checked against.
const CORPUS: &[(&str, &str)] = &[
    ("minimal", "<a/>"),
    ("empty-with-attributes", r#"<a x="1" y="2"/>"#),
    ("text", "<a>hello</a>"),
    ("nested", "<a><b><c/></b><b/></a>"),
    (
        "mixed-content",
        "<p>before <b>bold</b> after <i>italic</i> end</p>",
    ),
    ("comment", "<a><!-- a comment --><b/><!--another--></a>"),
    (
        "top-level-comments-and-pis",
        "<?first?><!--lead--><a/><!--trail--><?last data?>",
    ),
    (
        "processing-instruction",
        r#"<a><?xml-stylesheet href="s.css" type="text/css"?><b/></a>"#,
    ),
    ("cdata", "<a><![CDATA[<not> &markup;]]></a>"),
    ("cdata-and-text", "<a>plain<![CDATA[raw ]]>more</a>"),
    (
        "escapes-in-text",
        "<a>&lt;&amp;&gt;&quot;&apos; and &#65;&#x42;</a>",
    ),
    (
        "escapes-in-attributes",
        r#"<a v="&lt;&amp;&gt;&quot;&apos;" w="&#10;tab&#9;"/>"#,
    ),
    (
        "default-namespace",
        r#"<a xmlns="urn:example:a"><b/><c xmlns="urn:example:c"><d/></c></a>"#,
    ),
    (
        "prefixed-namespace",
        r#"<p:a xmlns:p="urn:example:p" p:x="1" y="2"><p:b/></p:a>"#,
    ),
    (
        "namespace-undeclaration",
        r#"<a xmlns="urn:example:a"><b xmlns=""><c/></b></a>"#,
    ),
    (
        "many-namespaces",
        r#"<a xmlns:p="urn:p" xmlns:q="urn:q" xmlns="urn:d"><p:b q:x="1"><q:c p:y="2"/></p:b></a>"#,
    ),
    (
        "xml-prefix",
        r#"<a xml:lang="ja" xml:space="preserve" xml:id="top"><b xml:lang="en"/></a>"#,
    ),
    ("doctype-external", "<!DOCTYPE a SYSTEM \"a.dtd\">\n<a/>"),
    (
        "doctype-public",
        "<!DOCTYPE a PUBLIC \"-//EXAMPLE//DTD a//EN\" \"a.dtd\">\n<a/>",
    ),
    (
        "doctype-internal-subset",
        "<!DOCTYPE a [<!ELEMENT a (#PCDATA)><!ENTITY greeting \"hi\">]>\n<a>&greeting;</a>",
    ),
    (
        "unicode",
        "<日本語 属性=\"値\">こんにちは、世界 — ﬀ 𝄞</日本語>",
    ),
    ("whitespace-heavy", "<a>\n  <b>  </b>\n  <c>\t</c>\n</a>"),
    (
        "declaration",
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><a/>",
    ),
];

fn parse(xml: &str) -> Result<Document, BuildError> {
    build::parse_str(xml)
}

#[test]
fn corpus_round_trips_through_the_serializer() -> Result<(), BuildError> {
    for (name, xml) in CORPUS {
        let original = parse(xml)?;
        let serialized = original.to_xml();
        let reparsed = build::parse_str(&serialized)
            .unwrap_or_else(|error| panic!("{name}: re-parsing {serialized:?} failed: {error}"));
        assert!(
            original.structural_eq(&reparsed),
            "{name}: {:?}\noriginal:   {xml}\nserialized: {serialized}",
            original.structural_diff(&reparsed)
        );
    }
    Ok(())
}

#[test]
fn corpus_round_trips_twice() -> Result<(), BuildError> {
    // Serializing a re-parsed document must be a fixed point: the second and
    // third generations are byte-identical.
    for (name, xml) in CORPUS {
        let first = parse(xml)?.to_xml();
        let second = build::parse_str(&first)?.to_xml();
        assert_eq!(first, second, "{name} is not a serialization fixed point");
    }
    Ok(())
}

/// Removes the white-space-only text nodes that the pretty printer is allowed to
/// introduce, i.e. those in element-only content.
///
/// The printer indents a node's children only when none of them is a text, CDATA
/// or entity-reference node, so this is the exact inverse of what it adds and
/// nothing else can be removed by it.
fn strip_indentation_whitespace(doc: &mut Document) {
    let candidates: Vec<_> = doc
        .traverse(doc.root())
        .filter(|&node| matches!(doc.kind(node), Some(NodeKind::Document | NodeKind::Element)))
        .collect();
    for parent in candidates {
        let children: Vec<_> = doc.children(parent).collect();
        let element_only = children.iter().all(|&child| {
            !matches!(
                doc.kind(child),
                Some(NodeKind::CdataSection | NodeKind::EntityReference)
            )
        }) && children.iter().all(|&child| {
            doc.kind(child) != Some(NodeKind::Text)
                || doc
                    .character_data(child)
                    .is_some_and(|text| text.trim().is_empty())
        });
        if !element_only {
            continue;
        }
        for child in children {
            if doc.kind(child) == Some(NodeKind::Text) {
                doc.remove_child(parent, child).expect("a child of parent");
            }
        }
    }
}

#[test]
fn corpus_round_trips_through_pretty_printing() -> Result<(), BuildError> {
    let options = SerializeOptions::pretty();
    for (name, xml) in CORPUS {
        let mut original = parse(xml)?;
        let serialized = original.to_xml_with(&options);
        let mut reparsed = build::parse_str(&serialized)
            .unwrap_or_else(|error| panic!("{name}: re-parsing {serialized:?} failed: {error}"));
        strip_indentation_whitespace(&mut original);
        strip_indentation_whitespace(&mut reparsed);
        assert!(
            original.structural_eq(&reparsed),
            "{name}: {:?}\npretty: {serialized}",
            original.structural_diff(&reparsed)
        );
    }
    Ok(())
}

#[test]
fn corpus_round_trips_through_an_io_writer() -> Result<(), Box<dyn std::error::Error>> {
    for (name, xml) in CORPUS {
        let original = parse(xml)?;
        let mut bytes: Vec<u8> = Vec::new();
        original.write_xml_io(&mut bytes, &SerializeOptions::default())?;
        let text = String::from_utf8(bytes)?;
        assert_eq!(text, original.to_xml(), "{name}");
        let reparsed = build::parse_str(&text)?;
        assert!(original.structural_eq(&reparsed), "{name}");
    }
    Ok(())
}

#[test]
fn deeply_nested_documents_round_trip() -> Result<(), BuildError> {
    const DEPTH: usize = 2_000;
    let mut xml = String::new();
    for level in 0..DEPTH {
        xml.push_str(&format!("<n{}>", level % 10));
    }
    xml.push_str("leaf");
    for level in (0..DEPTH).rev() {
        xml.push_str(&format!("</n{}>", level % 10));
    }

    // The reader caps namespace-scope nesting at `DEFAULT_MAX_ELEMENT_DEPTH`;
    // this test is about the tree, so raise the cap past the depth it builds.
    let options = build::BuilderOptions::new().with_max_element_depth(DEPTH + 1);
    let original = build::parse_str_with(&xml, &options)?;
    let serialized = original.to_xml();
    assert_eq!(serialized, xml);
    let reparsed = build::parse_str_with(&serialized, &options)?;
    assert!(original.structural_eq(&reparsed));

    // The tree must also be walkable and droppable without recursion blowing up.
    let leaf = original
        .descendants(original.root())
        .find(|&node| original.kind(node) == Some(NodeKind::Text))
        .expect("a text node");
    assert_eq!(original.ancestors(leaf).count(), DEPTH + 1);
    Ok(())
}

#[test]
fn wide_documents_round_trip() -> Result<(), BuildError> {
    const WIDTH: usize = 5_000;
    let mut xml = String::from("<root>");
    for index in 0..WIDTH {
        xml.push_str(&format!("<item n=\"{index}\"/>"));
    }
    xml.push_str("</root>");

    let original = build::parse_str(&xml)?;
    let root = original.document_element().expect("a document element");
    assert_eq!(original.children(root).count(), WIDTH);
    assert_eq!(original.to_xml(), xml);
    Ok(())
}

#[test]
fn text_is_coalesced_across_entity_references() -> Result<(), BuildError> {
    let doc = build::parse_str("<a>x&amp;y&#32;z</a>")?;
    let root = doc.document_element().expect("a document element");
    let children: Vec<_> = doc.children(root).collect();
    assert_eq!(children.len(), 1, "the text must be a single node");
    assert_eq!(doc.character_data(children[0]), Some("x&y z"));
    assert_eq!(doc.to_xml(), "<a>x&amp;y z</a>");
    Ok(())
}

#[test]
fn undeclared_entities_survive_as_entity_reference_nodes() -> Result<(), BuildError> {
    let doc = build::parse_str("<a>before &undeclared; after</a>")?;
    let root = doc.document_element().expect("a document element");
    let kinds: Vec<_> = doc
        .children(root)
        .filter_map(|node| doc.kind(node))
        .collect();
    assert_eq!(
        kinds,
        [NodeKind::Text, NodeKind::EntityReference, NodeKind::Text]
    );
    assert_eq!(doc.to_xml(), "<a>before &undeclared; after</a>");

    let reparsed = build::parse_str(&doc.to_xml())?;
    assert!(doc.structural_eq(&reparsed));
    Ok(())
}

#[test]
fn declared_entities_are_expanded_or_kept_according_to_the_policy() -> Result<(), BuildError> {
    let xml = "<!DOCTYPE a [<!ENTITY e \"expanded\">]><a>&e;</a>";

    let resolved = build::parse_str(xml)?;
    let root = resolved.document_element().expect("a document element");
    assert_eq!(resolved.text_content(root).as_deref(), Some("expanded"));

    let kept = build::parse_str_with(
        xml,
        &oxixml_dom::BuilderOptions::new().with_entity_policy(EntityPolicy::Keep),
    )?;
    let root = kept.document_element().expect("a document element");
    let child = kept.first_child(root).expect("a child");
    assert_eq!(kept.kind(child), Some(NodeKind::EntityReference));
    assert_eq!(kept.entity_reference(child), Some("e"));
    Ok(())
}

#[test]
fn strict_entity_policy_rejects_unknown_entities() {
    let error = build::parse_str_with(
        "<a>&nope;</a>",
        &oxixml_dom::BuilderOptions::new().with_entity_policy(EntityPolicy::Strict),
    )
    .expect_err("an unknown entity must be rejected");
    assert!(
        matches!(error, BuildError::UnknownEntity { ref name } if name == "nope"),
        "{error}"
    );
}

#[test]
fn entity_references_are_expanded_inside_attribute_values() -> Result<(), BuildError> {
    let doc = build::parse_str("<!DOCTYPE a [<!ENTITY e \"E\">]><a v=\"x&e;y&#33;\"/>")?;
    let root = doc.document_element().expect("a document element");
    assert_eq!(doc.get_attribute(root, "v"), Some("xEy!"));
    Ok(())
}

#[test]
fn attribute_values_are_normalized() -> Result<(), BuildError> {
    // XML 1.0 §3.3.3: literal tabs and line breaks in an attribute value become
    // spaces, while the equivalent character references do not.
    let doc = build::parse_str("<a v=\"one\ttwo\nthree\" w=\"one&#9;two&#10;three\"/>")?;
    let root = doc.document_element().expect("a document element");
    assert_eq!(doc.get_attribute(root, "v"), Some("one two three"));
    assert_eq!(doc.get_attribute(root, "w"), Some("one\ttwo\nthree"));
    assert_eq!(
        doc.to_xml(),
        "<a v=\"one two three\" w=\"one&#9;two&#10;three\"/>"
    );
    Ok(())
}

#[test]
fn namespace_declarations_are_attributes() -> Result<(), BuildError> {
    let doc = build::parse_str(r#"<p:a xmlns:p="urn:p" xmlns="urn:d" p:x="1"/>"#)?;
    let root = doc.document_element().expect("a document element");

    let declarations = doc.namespace_declarations(root);
    assert_eq!(
        declarations,
        [(Some("p"), "urn:p"), (None, "urn:d")],
        "declarations are kept in document order"
    );

    let xmlns_p = doc
        .get_attribute_node_ns(root, Some(XMLNS_NAMESPACE), "p")
        .expect("the xmlns:p attribute node");
    let data = doc.attr(xmlns_p).expect("an attribute");
    assert_eq!(data.prefix(), Some("xmlns"));
    assert_eq!(data.local_name(), "p");
    assert_eq!(data.qualified_name(), "xmlns:p");
    assert_eq!(data.value(), "urn:p");

    assert_eq!(doc.lookup_namespace_uri(root, Some("p")), Some("urn:p"));
    assert_eq!(doc.lookup_namespace_uri(root, None), Some("urn:d"));
    assert_eq!(doc.lookup_prefix(root, "urn:p"), Some("p"));
    assert!(doc.is_default_namespace(root, "urn:d"));
    Ok(())
}

#[test]
fn namespace_fixup_synthesizes_missing_declarations() -> Result<(), oxixml_dom::DomError> {
    let mut doc = Document::new();
    let root = doc.create_element_ns(Some("urn:example:r"), "r:root")?;
    doc.append_child(doc.root(), root)?;
    // A namespaced attribute with no prefix of its own must get a generated one.
    doc.set_attribute_ns(root, Some("urn:example:a"), "flag", "on")?;
    let child = doc.create_element_ns(Some("urn:example:c"), "child")?;
    doc.append_child(root, child)?;

    let xml = doc.to_xml();
    assert_eq!(
        xml,
        "<r:root xmlns:r=\"urn:example:r\" xmlns:ns1=\"urn:example:a\" ns1:flag=\"on\">\
         <child xmlns=\"urn:example:c\"/></r:root>"
    );

    let reparsed = build::parse_str(&xml).expect("the fixed-up output must parse");
    let reparsed_root = reparsed.document_element().expect("a document element");
    assert_eq!(
        reparsed.get_attribute_ns(reparsed_root, Some("urn:example:a"), "flag"),
        Some("on")
    );
    Ok(())
}

#[test]
fn namespace_fixup_can_be_turned_off() -> Result<(), oxixml_dom::DomError> {
    let mut doc = Document::new();
    let root = doc.create_element_ns(Some("urn:example:r"), "r:root")?;
    doc.append_child(doc.root(), root)?;
    let options = SerializeOptions::new().with_namespace_fixup(false);
    assert_eq!(doc.to_xml_with(&options), "<r:root/>");
    Ok(())
}

#[test]
fn empty_element_folding_is_configurable() -> Result<(), BuildError> {
    let doc = build::parse_str("<a><b></b><c/></a>")?;
    assert_eq!(doc.to_xml(), "<a><b/><c/></a>");
    let options = SerializeOptions::new().with_fold_empty_elements(false);
    assert_eq!(doc.to_xml_with(&options), "<a><b></b><c></c></a>");
    Ok(())
}

#[test]
fn the_xml_declaration_is_reproduced_on_request() -> Result<(), BuildError> {
    let doc = build::parse_str("<?xml version=\"1.1\" encoding=\"utf-8\" standalone=\"no\"?><a/>")?;
    assert_eq!(doc.xml_version(), Some("1.1"));
    assert_eq!(doc.encoding(), Some("utf-8"));
    assert_eq!(doc.standalone(), Some(false));

    let options = SerializeOptions::new().with_xml_declaration(XmlDeclaration::Include);
    assert_eq!(
        doc.to_xml_with(&options),
        "<?xml version=\"1.1\" encoding=\"utf-8\" standalone=\"no\"?>\n<a/>"
    );
    Ok(())
}

#[test]
fn a_subtree_can_be_serialized_on_its_own() -> Result<(), BuildError> {
    let doc = build::parse_str(r#"<a xmlns:p="urn:p"><p:b><c/></p:b></a>"#)?;
    let root = doc.document_element().expect("a document element");
    let b = doc.first_child(root).expect("the p:b element");
    // The declaration is in scope at the subtree root, so it is not repeated…
    assert_eq!(
        doc.node_to_xml(b, &SerializeOptions::default()),
        "<p:b><c/></p:b>"
    );
    // …but a document built around the fragment alone has to declare it.
    let mut extracted = Document::new();
    let imported = extracted.import_node(&doc, b, true).expect("import");
    let extracted_root = extracted.root();
    extracted
        .append_child(extracted_root, imported)
        .expect("append");
    assert_eq!(extracted.to_xml(), "<p:b xmlns:p=\"urn:p\"><c/></p:b>");
    Ok(())
}

#[test]
fn cdata_can_be_merged_into_text() -> Result<(), BuildError> {
    let options = oxixml_dom::BuilderOptions::new().with_cdata_as_text(true);
    let doc = build::parse_str_with("<a>x<![CDATA[<y>]]>z</a>", &options)?;
    let root = doc.document_element().expect("a document element");
    let children: Vec<_> = doc.children(root).collect();
    assert_eq!(children.len(), 1);
    assert_eq!(doc.character_data(children[0]), Some("x<y>z"));
    assert_eq!(doc.to_xml(), "<a>x&lt;y&gt;z</a>");
    Ok(())
}

#[test]
fn comments_and_processing_instructions_can_be_dropped() -> Result<(), BuildError> {
    let options = oxixml_dom::BuilderOptions::new()
        .with_comments(false)
        .with_processing_instructions(false);
    let doc = build::parse_str_with("<a><!--c--><?pi d?><b/></a>", &options)?;
    assert_eq!(doc.to_xml(), "<a><b/></a>");
    Ok(())
}

#[test]
fn parsing_from_a_reader_matches_parsing_from_a_string() -> Result<(), BuildError> {
    for (name, xml) in CORPUS {
        let from_str = build::parse_str(xml)?;
        let from_reader = build::parse_reader(std::io::Cursor::new(xml.as_bytes()))?;
        assert!(
            from_str.structural_eq(&from_reader),
            "{name}: {:?}",
            from_str.structural_diff(&from_reader)
        );
    }
    Ok(())
}

#[test]
fn malformed_documents_are_rejected() {
    for xml in [
        "<a>",
        "<a></b>",
        "<a><b></a></b>",
        "not markup at all",
        "<a attr=unquoted/>",
    ] {
        assert!(
            build::parse_str(xml).is_err(),
            "{xml:?} should not have parsed"
        );
    }
}

#[test]
fn unbound_prefixes_are_reported() {
    let error = build::parse_str("<p:a/>").expect_err("an unbound prefix must be rejected");
    let message = error.to_string();
    assert!(message.contains('p'), "{message}");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Regression: XML 1.0 §2.11 end-of-line normalization, namespace fixup, WF checks
////////////////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn line_endings_are_normalized_in_character_data() -> Result<(), Box<dyn std::error::Error>> {
    // XML 1.0 §2.11: `#xD #xA` and a lone `#xD` must both become a single `#xA`
    // on input. Leaving them alone made the DOM disagree with the Infoset tree
    // and turned the stray CR into a permanent `&#13;` on the way out.
    for (xml, expected) in [
        ("<a>x\r\ny</a>", "x\ny"),
        ("<a>x\ry</a>", "x\ny"),
        ("<a><![CDATA[x\r\ny]]></a>", "x\ny"),
        ("<a><![CDATA[x\ry]]></a>", "x\ny"),
    ] {
        let doc = build::parse_str(xml)?;
        let root = doc.document_element().expect("a document element");
        assert_eq!(
            doc.text_content(root).as_deref(),
            Some(expected),
            "input {xml:?}"
        );
    }

    let doc = build::parse_str("<a>x\r\ny</a>")?;
    assert_eq!(doc.to_xml(), "<a>x\ny</a>");

    let doc = build::parse_str("<a><!--x\r\ny--><?pi d\r\ne?></a>")?;
    assert_eq!(doc.to_xml(), "<a><!--x\ny--><?pi d\ne?></a>");
    Ok(())
}

#[test]
fn duplicate_expanded_attribute_names_are_rejected() {
    // Namespaces in XML 1.0 §6.3: no element may have two attributes with the
    // same expanded name. `p` and `q` are bound to the same URI here, so `p:x`
    // and `q:x` collide and the document must be rejected, not silently
    // half-dropped.
    let error = build::parse_str(r#"<a xmlns:p="u" xmlns:q="u" p:x="1" q:x="2"/>"#)
        .expect_err("a namespace-level duplicate attribute must be rejected");
    assert!(
        matches!(error, BuildError::DuplicateAttribute { ref qualified_name } if qualified_name == "q:x"),
        "{error:?}"
    );

    // Different local names in the same namespace are of course still fine.
    assert!(build::parse_str(r#"<a xmlns:p="u" xmlns:q="u" p:x="1" q:y="2"/>"#).is_ok());
}

#[test]
fn character_data_outside_the_document_element_is_rejected() {
    let error = build::parse_str("<a>t</a> trailing")
        .expect_err("character data after the root is not well-formed");
    assert!(
        matches!(error, BuildError::ContentOutsideDocumentElement { ref text } if text.contains("trailing")),
        "{error:?}"
    );

    // `Misc` white space around the document element stays legal.
    assert!(build::parse_str("\n <a>t</a> \n").is_ok());
}

#[test]
fn a_conflicting_attribute_prefix_gets_a_generated_prefix() -> Result<(), Box<dyn std::error::Error>>
{
    // DOM L3 Core Appendix B.1: when the attribute's declared prefix is already
    // bound to a different namespace, the fixup must *generate* a new prefix.
    // Re-binding `p` produced a duplicate `xmlns:p` (non-well-formed output) and
    // silently changed the element's own namespace.
    let mut doc = Document::new();
    let element = doc.create_element_ns(Some("urn:A"), "p:e")?;
    let document = doc.root();
    doc.append_child(document, element)?;
    doc.set_attribute_ns(element, Some("urn:B"), "p:a", "v")?;

    let xml = doc.to_xml();
    assert_eq!(xml.matches("xmlns:p=").count(), 1, "{xml}");
    let reparsed = build::parse_str(&xml)?;
    let root = reparsed.document_element().expect("a document element");
    assert_eq!(
        reparsed.element(root).map(|d| d.namespace_uri()),
        Some(Some("urn:A"))
    );

    // Nested case: the child element must keep `urn:A` after a round trip.
    let mut doc = Document::new();
    let root = doc.create_element_ns(Some("urn:A"), "p:r")?;
    let document = doc.root();
    doc.append_child(document, root)?;
    doc.set_attribute_ns(root, Some(XMLNS_NAMESPACE), "xmlns:p", "urn:A")?;
    let child = doc.create_element_ns(Some("urn:A"), "p:c")?;
    doc.append_child(root, child)?;
    doc.set_attribute_ns(child, Some("urn:B"), "p:a", "v")?;

    let xml = doc.to_xml();
    let reparsed = build::parse_str(&xml)?;
    let reparsed_root = reparsed.document_element().expect("a document element");
    let reparsed_child = reparsed
        .children(reparsed_root)
        .next()
        .expect("the child element");
    assert_eq!(
        reparsed.element(reparsed_child).map(|d| d.namespace_uri()),
        Some(Some("urn:A")),
        "the child's own namespace must survive the round trip: {xml}"
    );
    assert_eq!(
        reparsed
            .get_attribute_node_ns(reparsed_child, Some("urn:B"), "a")
            .and_then(|id| reparsed.attr(id))
            .map(|data| data.value().to_owned()),
        Some("v".to_owned()),
        "{xml}"
    );
    Ok(())
}

#[test]
fn a_level_one_xmlns_attribute_is_not_declared_twice() -> Result<(), Box<dyn std::error::Error>> {
    // `set_attribute` cannot assign a namespace, so the resulting `xmlns:p`
    // attribute has no `namespaceURI`; the fixup must still recognise it as a
    // declaration instead of emitting a second one.
    let mut doc = Document::new();
    let element = doc.create_element_ns(Some("urn:A"), "p:e")?;
    let document = doc.root();
    doc.append_child(document, element)?;
    doc.set_attribute(element, "xmlns:p", "urn:A")?;

    let xml = doc.to_xml();
    assert_eq!(xml.matches("xmlns:p=").count(), 1, "{xml}");
    assert!(build::parse_str(&xml).is_ok(), "{xml}");
    Ok(())
}

/// Documents whose internal subset declares an entity whose value itself holds
/// references: `(name, internal subset, the reference as written, the character
/// data it stands for)`.
///
/// XML 1.0 §4.4 gives this a two-phase answer: a character reference inside an
/// `EntityValue` is resolved where the entity is *declared*, a general entity
/// reference where it is *used*, and neither pass rescans what it produced. The
/// same answer is required in content and in an attribute value, and
/// `oxixml-sax` and `oxixml-infoset` are held to this very table, so a document
/// means the same thing wherever the reference appears and whichever of the
/// three crates reads it.
const ENTITY_EXPANSIONS: &[(&str, &str, &str, &str)] = &[
    (
        "predefined-inside-the-value",
        r#"<!ENTITY e "bar &amp; baz">"#,
        "&e;",
        "bar & baz",
    ),
    (
        "character-reference-inside-the-value",
        r#"<!ENTITY copy "&#169;">"#,
        "&copy;",
        "\u{a9}",
    ),
    (
        "character-reference-in-running-text",
        r#"<!ENTITY nbsp "&#160;">"#,
        "x&nbsp;y",
        "x\u{a0}y",
    ),
    (
        "hexadecimal-character-reference",
        r#"<!ENTITY e "&#xe9;t&#xe9;">"#,
        "&e;",
        "\u{e9}t\u{e9}",
    ),
    (
        "nested-general-entity",
        r#"<!ENTITY inner "in"><!ENTITY outer "[&inner;]">"#,
        "&outer;",
        "[in]",
    ),
    (
        "escaped-markup-stays-text",
        r#"<!ENTITY tag "&lt;foo&gt;">"#,
        "&tag;",
        "<foo>",
    ),
    (
        // A `>` needs no escape anywhere, so the scanner that finds the end of
        // the declaration has to look past the one inside the quotes.
        "unescaped-greater-than-in-the-value",
        r#"<!ENTITY e "a > b">"#,
        "&e;",
        "a > b",
    ),
    (
        // A literal delimited by apostrophes may hold quotation marks, and a
        // quotation mark that reaches an attribute value through an entity is
        // data rather than the end of the literal (XML 1.0 §4.4, "Included in
        // Literal").
        "quotation-marks-in-an-apostrophe-quoted-value",
        r#"<!ENTITY e 'say "hi"'>"#,
        "&e;",
        "say \"hi\"",
    ),
    (
        "doubly-escaped-less-than",
        r#"<!ENTITY e "&#38;#60;">"#,
        "&e;",
        "<",
    ),
    (
        "doubly-escaped-ampersand",
        r#"<!ENTITY e "&#38;#38;">"#,
        "&e;",
        "&",
    ),
    (
        "triply-escaped-ampersand",
        r#"<!ENTITY e "&#38;#38;#38;">"#,
        "&e;",
        "&#38;",
    ),
    (
        "the-specifications-own-example",
        r#"<!ENTITY rights "All rights reserved"><!ENTITY book "La Peste: &#38;#38; 1947, &#38;rights;">"#,
        "&book;",
        "La Peste: & 1947, All rights reserved",
    ),
];

/// The document that puts `reference` in content.
fn in_content(subset: &str, reference: &str) -> String {
    format!("<!DOCTYPE r [{subset}]><r>{reference}</r>")
}

/// The document that puts `reference` in an attribute value.
fn in_attribute(subset: &str, reference: &str) -> String {
    format!("<!DOCTYPE r [{subset}]><r a=\"{reference}\"/>")
}

#[test]
fn declared_entities_expand_to_the_agreed_character_data() -> Result<(), BuildError> {
    for &(name, subset, reference, expected) in ENTITY_EXPANSIONS {
        let xml = in_content(subset, reference);
        let doc = build::parse_str(&xml)?;
        let root = doc.document_element().expect("a document element");
        assert_eq!(doc.text_content(root).as_deref(), Some(expected), "{name}");

        // The expansion is character data, so serializing escapes it exactly
        // once — the reference must not come back doubly escaped.
        let reparsed = build::parse_str(&doc.to_xml())?;
        let root = reparsed.document_element().expect("a document element");
        assert_eq!(
            reparsed.text_content(root).as_deref(),
            Some(expected),
            "{name} does not survive a round-trip"
        );
    }
    Ok(())
}

#[test]
fn an_attribute_value_expands_to_the_same_character_data_as_content() -> Result<(), BuildError> {
    for &(name, subset, reference, expected) in ENTITY_EXPANSIONS {
        let xml = in_attribute(subset, reference);
        let doc = build::parse_str(&xml)?;
        let root = doc.document_element().expect("a document element");
        assert_eq!(doc.get_attribute(root, "a"), Some(expected), "{name}");

        let reparsed = build::parse_str(&doc.to_xml())?;
        let root = reparsed.document_element().expect("a document element");
        assert_eq!(
            reparsed.get_attribute(root, "a"),
            Some(expected),
            "{name} does not survive a round-trip"
        );
    }
    Ok(())
}

#[test]
fn an_expanded_entity_is_escaped_exactly_once() -> Result<(), BuildError> {
    let doc = build::parse_str(r#"<!DOCTYPE r [<!ENTITY e "bar &amp; baz">]><r>&e;</r>"#)?;
    assert_eq!(
        doc.to_xml(),
        "<!DOCTYPE r [<!ENTITY e \"bar &amp; baz\">]><r>bar &amp; baz</r>"
    );
    Ok(())
}

#[test]
fn an_entity_whose_replacement_text_is_markup_keeps_its_reference() -> Result<(), BuildError> {
    let doc = build::parse_str(r#"<!DOCTYPE r [<!ENTITY e "<b>x</b>">]><r>&e;</r>"#)?;
    let root = doc.document_element().expect("a document element");
    let child = doc.first_child(root).expect("a child");
    assert_eq!(doc.kind(child), Some(NodeKind::EntityReference));
    assert_eq!(doc.entity_reference(child), Some("e"));
    Ok(())
}

#[test]
fn a_recursive_entity_is_rejected_instead_of_hanging() {
    for xml in [
        r#"<!DOCTYPE r [<!ENTITY e "&e;">]><r>&e;</r>"#,
        r#"<!DOCTYPE r [<!ENTITY a "&b;"><!ENTITY b "&a;">]><r>&a;</r>"#,
    ] {
        let error = build::parse_str(xml).expect_err("a recursive entity must be rejected");
        assert!(
            matches!(error, BuildError::EntityExpansion { .. }),
            "{error}"
        );
    }
}

#[test]
fn a_billion_laughs_declaration_is_refused() {
    let mut xml = String::from(r#"<!DOCTYPE r [<!ENTITY lol0 "lololololololololol">"#);
    for level in 1..10 {
        xml.push_str(&format!(
            r#"<!ENTITY lol{level} "{}">"#,
            format!("&lol{};", level - 1).repeat(10)
        ));
    }
    xml.push_str("]><r>&lol9;</r>");

    let error = build::parse_str(&xml).expect_err("an exponential expansion must be refused");
    assert!(
        matches!(error, BuildError::EntityExpansion { .. }),
        "{error}"
    );
}

#[test]
fn the_first_declaration_of_an_entity_binds() -> Result<(), BuildError> {
    let doc =
        build::parse_str(r#"<!DOCTYPE r [<!ENTITY e "first"><!ENTITY e "second">]><r>&e;</r>"#)?;
    let root = doc.document_element().expect("a document element");
    assert_eq!(doc.text_content(root).as_deref(), Some("first"));
    Ok(())
}

#[test]
fn a_reference_may_be_repeated_as_often_as_a_document_likes() -> Result<(), BuildError> {
    // Repeating a reference is not "amplification": what the entity stands for
    // is written out once per occurrence, so an ordinary page with tens of
    // thousands of `&nbsp;` in it must parse like any other.
    const OCCURRENCES: usize = 50_000;
    let mut xml = String::from(r#"<!DOCTYPE r [<!ENTITY nbsp "&#160;">]><r>"#);
    for _ in 0..OCCURRENCES {
        xml.push_str("&nbsp;");
    }
    xml.push_str("</r>");

    let doc = build::parse_str(&xml)?;
    let root = doc.document_element().expect("a document element");
    let text = doc.text_content(root).expect("character data");
    assert_eq!(text.chars().count(), OCCURRENCES);
    assert!(text.chars().all(|character| character == '\u{a0}'));
    Ok(())
}

/// The internal subset of a "billion laughs" declaration: ten levels, each
/// naming the previous one ten times, so the expansion of `&lol9;` is 10^9
/// copies of the innermost text while the document itself stays a few hundred
/// bytes long.
fn billion_laughs_subset() -> String {
    let mut subset = String::from(r#"<!ENTITY lol0 "lololololololololol">"#);
    for level in 1..10 {
        subset.push_str(&format!(
            r#"<!ENTITY lol{level} "{}">"#,
            format!("&lol{};", level - 1).repeat(10)
        ));
    }
    subset
}

#[test]
fn a_billion_laughs_attribute_value_is_refused_like_one_in_content() {
    // The attribute path expands through the underlying reader rather than
    // through this crate's own expander, so it needs its own proof that the
    // budget still bites: a *wide* declaration multiplies enormously without
    // ever nesting deeply enough to trip the depth limit alone.
    let subset = billion_laughs_subset();
    for xml in [
        format!("<!DOCTYPE r [{subset}]><r>&lol9;</r>"),
        format!("<!DOCTYPE r [{subset}]><r a=\"&lol9;\"/>"),
    ] {
        let error = build::parse_str(&xml).expect_err("an exponential expansion must be refused");
        assert!(
            matches!(error, BuildError::EntityExpansion { .. }),
            "{error}"
        );
    }
}
