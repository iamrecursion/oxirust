//! End-to-end tests driving [`process_document`] over in-memory fixtures.

use oxixml_dom::{Document, NodeKind, XML_NAMESPACE, build};
use oxixml_xinclude::{
    LoadError, MapLoader, NullLoader, ResourceLoader, XIncludeError, XIncludeOptions,
    process_document, process_document_with,
};

const XI: &str = r#"xmlns:xi="http://www.w3.org/2001/XInclude""#;

fn parse(xml: &str) -> Document {
    build::parse_str(xml).expect("fixture must be well-formed XML")
}

fn first_element_named(doc: &Document, name: &str) -> oxixml_dom::NodeId {
    let root = doc.document_element().expect("document element");
    doc.get_elements_by_tag_name(root, name)
        .next()
        .unwrap_or_else(|| panic!("no <{name}> element found in the processed document"))
}

// ---------------------------------------------------------------------- xml inclusion + fixup

#[test]
fn xml_inclusion_grafts_the_referenced_document_element() {
    let mut doc = parse(&format!(r#"<doc {XI}><xi:include href="part.xml"/></doc>"#));
    let loader = MapLoader::new().with_text("https://example.com/part.xml", "<p>included</p>");

    process_document(&mut doc, "https://example.com/index.xml", &loader).expect("processes");

    let p = first_element_named(&doc, "p");
    assert_eq!(doc.text_content(p).as_deref(), Some("included"));
}

#[test]
fn xml_inclusion_adds_base_and_lang_fixup_when_they_differ() {
    let mut doc = parse(&format!(
        r#"<doc {XI} xml:lang="en"><xi:include href="sub/part.xml"/></doc>"#
    ));
    let loader = MapLoader::new().with_text(
        "https://example.com/sub/part.xml",
        r#"<p xml:lang="fr">bonjour</p>"#,
    );

    process_document(&mut doc, "https://example.com/index.xml", &loader).expect("processes");

    let p = first_element_named(&doc, "p");
    assert_eq!(
        doc.get_attribute_ns(p, Some(XML_NAMESPACE), "base"),
        Some("https://example.com/sub/part.xml")
    );
    assert_eq!(
        doc.get_attribute_ns(p, Some(XML_NAMESPACE), "lang"),
        Some("fr")
    );
}

#[test]
fn xml_inclusion_clears_language_explicitly_when_the_acquired_item_has_none() {
    let mut doc = parse(&format!(
        r#"<doc {XI} xml:lang="en"><xi:include href="part.xml"/></doc>"#
    ));
    let loader = MapLoader::new().with_text("https://example.com/part.xml", "<p>hi</p>");

    process_document(&mut doc, "https://example.com/index.xml", &loader).expect("processes");

    let p = first_element_named(&doc, "p");
    // The acquired item has no language of its own, but the include parent does: the fixup
    // algorithm makes that explicit with `xml:lang=""` rather than leaving `<p>` to silently
    // inherit "en" from its new surroundings.
    assert_eq!(
        doc.get_attribute_ns(p, Some(XML_NAMESPACE), "lang"),
        Some("")
    );
}

#[test]
fn self_reference_adds_no_base_fixup_since_the_target_shares_the_document_base() {
    // The acquired item's base URI (the including document's own base URI, since this is a
    // self-reference) is, by construction, identical to the include parent's base URI: no
    // `xml:base` fixup should be added.
    let mut doc = parse(&format!(
        r#"<doc {XI} xml:id="root" xml:lang="en">
             <a xml:id="target" xml:lang="en"><v>42</v></a>
             <xi:include xpointer="target"/>
           </doc>"#
    ));

    process_document(&mut doc, "https://example.com/index.xml", &NullLoader).expect("processes");

    let root = doc.document_element().expect("document element");
    let grafted = doc
        .get_elements_by_tag_name(root, "a")
        .nth(1)
        .expect("the self-included copy of <a>");
    assert_eq!(
        doc.get_attribute_ns(grafted, Some(XML_NAMESPACE), "base"),
        None
    );
    // The `xml:lang="en"` here is the one already present verbatim in the source (also matching
    // the include parent's), so the fixup algorithm did not need to touch it either.
    assert_eq!(
        doc.get_attribute_ns(grafted, Some(XML_NAMESPACE), "lang"),
        Some("en")
    );
}

// -------------------------------------------------------------------------------- nested includes

#[test]
fn nested_includes_are_resolved_relative_to_their_own_resource() {
    let mut doc = parse(&format!(r#"<doc {XI}><xi:include href="a.xml"/></doc>"#));
    let loader = MapLoader::new()
        .with_text(
            "https://example.com/a.xml",
            format!(r#"<a {XI}><xi:include href="sub/b.xml"/></a>"#),
        )
        .with_text("https://example.com/sub/b.xml", "<b>leaf</b>");

    process_document(&mut doc, "https://example.com/index.xml", &loader).expect("processes");

    let b = first_element_named(&doc, "b");
    assert_eq!(doc.text_content(b).as_deref(), Some("leaf"));
    let a = first_element_named(&doc, "a");
    assert_eq!(doc.parent(b), Some(a));
}

// --------------------------------------------------------------------------------- text inclusion

#[test]
fn text_inclusion_splices_a_single_text_node() {
    let mut doc = parse(&format!(
        r#"<doc {XI}><p><xi:include href="snippet.txt" parse="text"/></p></doc>"#
    ));
    let loader = MapLoader::new().with_text("https://example.com/snippet.txt", "plain & <raw>");

    process_document(&mut doc, "https://example.com/index.xml", &loader).expect("processes");

    let p = first_element_named(&doc, "p");
    assert_eq!(doc.text_content(p).as_deref(), Some("plain & <raw>"));
    let child = doc.first_child(p).expect("spliced text node");
    assert_eq!(doc.kind(child), Some(NodeKind::Text));
}

#[test]
fn text_inclusion_honors_the_encoding_attribute() {
    let mut doc = parse(&format!(
        r#"<doc {XI}><p><xi:include href="latin1.txt" parse="text" encoding="ISO-8859-1"/></p></doc>"#
    ));
    // 0xE9 in Latin-1 is U+00E9 (é).
    let loader = MapLoader::new().with("https://example.com/latin1.txt", vec![b'c', 0xE9]);

    process_document(&mut doc, "https://example.com/index.xml", &loader).expect("processes");

    let p = first_element_named(&doc, "p");
    assert_eq!(doc.text_content(p).as_deref(), Some("c\u{e9}"));
}

#[test]
fn text_inclusion_with_missing_encoding_falls_back_via_xi_fallback() {
    let mut doc = parse(&format!(
        r#"<doc {XI}><p><xi:include href="x.txt" parse="text" encoding="bogus-8"><xi:fallback>oops</xi:fallback></xi:include></p></doc>"#
    ));
    let loader = MapLoader::new().with_text("https://example.com/x.txt", "hello");

    process_document(&mut doc, "https://example.com/index.xml", &loader).expect("processes");

    let p = first_element_named(&doc, "p");
    assert_eq!(doc.text_content(p).as_deref(), Some("oops"));
}

// ------------------------------------------------------------------------------------- xpointer

#[test]
fn xpointer_shorthand_selects_by_id() {
    let mut doc = parse(&format!(
        r#"<doc {XI}><xi:include href="a.xml" xpointer="chapter2"/></doc>"#
    ));
    let loader = MapLoader::new().with_text(
        "https://example.com/a.xml",
        r#"<book><chapter xml:id="chapter1">one</chapter><chapter xml:id="chapter2">two</chapter></book>"#,
    );

    process_document(&mut doc, "https://example.com/index.xml", &loader).expect("processes");

    let chapter = first_element_named(&doc, "chapter");
    assert_eq!(doc.text_content(chapter).as_deref(), Some("two"));
}

#[test]
fn xpointer_element_scheme_child_sequence_selects_by_position() {
    let mut doc = parse(&format!(
        r#"<doc {XI}><xi:include href="a.xml" xpointer="element(/1/2)"/></doc>"#
    ));
    let loader = MapLoader::new().with_text(
        "https://example.com/a.xml",
        "<book><chapter>one</chapter><chapter>two</chapter></book>",
    );

    process_document(&mut doc, "https://example.com/index.xml", &loader).expect("processes");

    let chapter = first_element_named(&doc, "chapter");
    assert_eq!(doc.text_content(chapter).as_deref(), Some("two"));
}

#[test]
fn xpointer_that_matches_nothing_triggers_fallback() {
    let mut doc = parse(&format!(
        r#"<doc {XI}><xi:include href="a.xml" xpointer="missing"><xi:fallback><na/></xi:fallback></xi:include></doc>"#
    ));
    let loader = MapLoader::new().with_text("https://example.com/a.xml", "<book/>");

    process_document(&mut doc, "https://example.com/index.xml", &loader).expect("processes");

    first_element_named(&doc, "na");
}

// -------------------------------------------------------------------------------------- fallback

#[test]
fn fallback_is_used_when_the_resource_is_unavailable() {
    let mut doc = parse(&format!(
        r#"<doc {XI}><xi:include href="missing.xml"><xi:fallback><na>n/a</na></xi:fallback></xi:include></doc>"#
    ));

    process_document(&mut doc, "https://example.com/index.xml", &NullLoader).expect("processes");

    let na = first_element_named(&doc, "na");
    assert_eq!(doc.text_content(na).as_deref(), Some("n/a"));
}

#[test]
fn fallback_content_is_itself_xinclude_processed() {
    let mut doc = parse(&format!(
        r#"<doc {XI}><xi:include href="missing.xml"><xi:fallback><xi:include href="rescue.xml"/></xi:fallback></xi:include></doc>"#
    ));
    let loader = MapLoader::new().with_text("https://example.com/rescue.xml", "<rescued/>");

    process_document(&mut doc, "https://example.com/index.xml", &loader).expect("processes");

    first_element_named(&doc, "rescued");
}

#[test]
fn unused_fallback_content_is_never_validated() {
    // The include resolves successfully, so the (structurally invalid, doubly-nested)
    // `xi:fallback` content must never be visited or reported.
    let mut doc = parse(&format!(
        r#"<doc {XI}><xi:include href="a.xml"><xi:fallback><xi:fallback/></xi:fallback></xi:include></doc>"#
    ));
    let loader = MapLoader::new().with_text("https://example.com/a.xml", "<ok/>");

    process_document(&mut doc, "https://example.com/index.xml", &loader).expect("processes");

    first_element_named(&doc, "ok");
}

#[test]
fn wrong_number_of_fallback_children_is_a_fatal_error_when_fallback_is_needed() {
    let mut doc = parse(&format!(
        r#"<doc {XI}><xi:include href="missing.xml"><xi:fallback>a</xi:fallback><xi:fallback>b</xi:fallback></xi:include></doc>"#
    ));

    let err = process_document(&mut doc, "https://example.com/index.xml", &NullLoader)
        .expect_err("two fallbacks is an error");
    assert!(matches!(err, XIncludeError::FallbackCount { found: 2, .. }));
}

#[test]
fn missing_resource_without_fallback_is_a_fatal_error() {
    let mut doc = parse(&format!(
        r#"<doc {XI}><xi:include href="missing.xml"/></doc>"#
    ));

    let err = process_document(&mut doc, "https://example.com/index.xml", &NullLoader)
        .expect_err("no fallback to recover with");
    assert!(matches!(err, XIncludeError::FallbackCount { found: 0, .. }));
}

// --------------------------------------------------------------------------------- self-reference

#[test]
fn empty_href_refers_to_the_document_itself() {
    let mut doc = parse(&format!(
        r#"<doc {XI} xml:id="root"><a xml:id="target"><v>42</v></a><xi:include xpointer="target"/></doc>"#
    ));

    process_document(&mut doc, "https://example.com/index.xml", &NullLoader).expect("processes");

    let root = doc.document_element().expect("document element");
    let values: Vec<Option<String>> = doc
        .get_elements_by_tag_name(root, "v")
        .map(|v| doc.text_content(v))
        .collect();
    // The original <a>/<v> plus the self-included copy.
    assert_eq!(values, vec![Some("42".to_owned()), Some("42".to_owned())]);
}

// ------------------------------------------------------------------------------------ loop guards

#[test]
fn direct_self_loop_is_a_fatal_error() {
    let mut doc = parse(&format!(r#"<doc {XI}><xi:include href="a.xml"/></doc>"#));
    let loader = MapLoader::new().with_text(
        "https://example.com/a.xml",
        format!(r#"<a {XI}><xi:include href="a.xml"/></a>"#),
    );

    let err = process_document(&mut doc, "https://example.com/index.xml", &loader)
        .expect_err("a.xml including itself is a loop");
    assert!(matches!(err, XIncludeError::LoopDetected { .. }));
}

#[test]
fn indirect_loop_between_two_resources_is_a_fatal_error() {
    let mut doc = parse(&format!(r#"<doc {XI}><xi:include href="a.xml"/></doc>"#));
    let loader = MapLoader::new()
        .with_text(
            "https://example.com/a.xml",
            format!(r#"<a {XI}><xi:include href="b.xml"/></a>"#),
        )
        .with_text(
            "https://example.com/b.xml",
            format!(r#"<b {XI}><xi:include href="a.xml"/></b>"#),
        );

    let err = process_document(&mut doc, "https://example.com/index.xml", &loader)
        .expect_err("a -> b -> a is a loop");
    assert!(matches!(err, XIncludeError::LoopDetected { .. }));
}

#[test]
fn same_href_with_different_xpointers_is_not_a_loop() {
    let mut doc = parse(&format!(
        r#"<doc {XI}>
             <xi:include href="a.xml" xpointer="one"/>
             <xi:include href="a.xml" xpointer="two"/>
           </doc>"#
    ));
    let loader = MapLoader::new().with_text(
        "https://example.com/a.xml",
        r#"<r><x xml:id="one">1</x><x xml:id="two">2</x></r>"#,
    );

    process_document(&mut doc, "https://example.com/index.xml", &loader).expect("not a loop");

    let root = doc.document_element().expect("document element");
    let texts: Vec<Option<String>> = doc
        .get_elements_by_tag_name(root, "x")
        .map(|x| doc.text_content(x))
        .collect();
    assert_eq!(texts, vec![Some("1".to_owned()), Some("2".to_owned())]);
}

#[test]
fn repeating_the_same_href_with_parse_text_is_legal() {
    let mut doc = parse(&format!(
        r#"<doc {XI}>
             <p><xi:include href="a.txt" parse="text"/></p>
             <q><xi:include href="a.txt" parse="text"/></q>
           </doc>"#
    ));
    let loader = MapLoader::new().with_text("https://example.com/a.txt", "shared");

    process_document(&mut doc, "https://example.com/index.xml", &loader).expect("legal repeat");

    let root = doc.document_element().expect("document element");
    let p = doc.get_elements_by_tag_name(root, "p").next().expect("p");
    let q = doc.get_elements_by_tag_name(root, "q").next().expect("q");
    assert_eq!(doc.text_content(p).as_deref(), Some("shared"));
    assert_eq!(doc.text_content(q).as_deref(), Some("shared"));
}

// ------------------------------------------------------------------------------------ max depth

struct ChainLoader;

impl ResourceLoader for ChainLoader {
    fn load(
        &self,
        iri: &str,
        _accept: Option<&str>,
        _accept_language: Option<&str>,
    ) -> Result<Vec<u8>, LoadError> {
        let n: u32 = iri
            .strip_prefix("https://example.com/chain/")
            .and_then(|rest| rest.strip_suffix(".xml"))
            .and_then(|digits| digits.parse().ok())
            .ok_or(LoadError::NotFound)?;
        Ok(format!(r#"<n {XI}>{n}<xi:include href="{}.xml"/></n>"#, n + 1).into_bytes())
    }
}

#[test]
fn max_depth_guards_a_long_non_repeating_chain() {
    let mut doc = parse(&format!(
        r#"<doc {XI}><xi:include href="chain/0.xml"/></doc>"#
    ));
    let options = XIncludeOptions::new().with_max_depth(5);

    let err = process_document_with(
        &mut doc,
        "https://example.com/index.xml",
        &ChainLoader,
        &options,
    )
    .expect_err("the chain never repeats, so only the depth guard can stop it");
    assert!(matches!(err, XIncludeError::MaxDepthExceeded));
}

/// Builds a hand-assembled [`Document`] — through its own public tree-mutation API, not through
/// [`build::parse_str`] — nesting `depth` plain `<e>` elements around an `<xi:include>`.
fn hand_built_deep_document(depth: usize) -> Document {
    let mut doc = Document::new();
    let root = doc.create_element("doc").expect("create the root element");
    let doc_node = doc.root();
    doc.append_child(doc_node, root)
        .expect("append the root element to the document");
    doc.set_attribute_ns(
        root,
        Some("http://www.w3.org/2000/xmlns/"),
        "xmlns:xi",
        "http://www.w3.org/2001/XInclude",
    )
    .expect("declare the xi prefix");

    let mut cursor = root;
    for _ in 0..depth {
        let e = doc.create_element("e").expect("create a nested element");
        doc.append_child(cursor, e)
            .expect("append the nested element");
        cursor = e;
    }
    let include = doc
        .create_element_ns(Some("http://www.w3.org/2001/XInclude"), "xi:include")
        .expect("create xi:include");
    doc.append_child(cursor, include)
        .expect("append xi:include at the bottom of the chain");
    doc.set_attribute(include, "href", "part.xml")
        .expect("set href");
    doc
}

#[test]
fn a_hand_built_document_nested_too_deeply_is_a_clean_error_not_a_stack_overflow() {
    // `build::parse_str` bounds element nesting on its own, so this shape is unreachable by
    // parsing hostile document *text* — but `Document::create_element`/`append_child` is a fully
    // public API with no depth check of its own, so a caller assembling a `Document` directly
    // (rather than parsing one) could otherwise still exhaust the native call stack here.
    let mut doc = hand_built_deep_document(1000);
    let loader = MapLoader::new().with_text("https://example.com/part.xml", "<p>ok</p>");
    let err = process_document(&mut doc, "https://example.com/index.xml", &loader)
        .expect_err("nesting this deep must be rejected");
    assert!(
        matches!(err, XIncludeError::ElementNestingTooDeep),
        "{err:?}"
    );
}

#[test]
fn a_hand_built_document_at_the_element_depth_cap_still_processes() {
    // The guard must not reject a tree that lands exactly on the cap — only one deeper — and the
    // resource must still be genuinely included, not merely fail to error. One shallower than
    // the crash-triggering shape above: the root `<doc>` element itself is one level of nesting
    // too, so 255 nested `<e>`s plus that root lands the `xi:include` exactly at the cap.
    let mut doc = hand_built_deep_document(255);
    let loader = MapLoader::new().with_text("https://example.com/part.xml", "<p>ok</p>");
    process_document(&mut doc, "https://example.com/index.xml", &loader)
        .expect("nesting exactly at the cap must still process");
    let p = first_element_named(&doc, "p");
    assert_eq!(doc.text_content(p).as_deref(), Some("ok"));
}

#[test]
fn the_element_depth_cap_still_applies_inside_fallback_content_reached_from_deep_nesting() {
    // `xi:fallback` content sits exactly as deep in the tree as the `xi:include` it replaces —
    // the depth counter must carry through `use_fallback` rather than resetting to 0, or a
    // failing resource plus deeply nested fallback content would silently bypass the guard.
    // Nested close to the cap and then padded further *inside* the fallback: if the counter
    // were wrongly reset, this would process without error; since it is not, it must still be
    // rejected.
    let mut doc = Document::new();
    let root = doc.create_element("doc").expect("create the root element");
    let doc_node = doc.root();
    doc.append_child(doc_node, root)
        .expect("append the root element to the document");
    doc.set_attribute_ns(
        root,
        Some("http://www.w3.org/2000/xmlns/"),
        "xmlns:xi",
        "http://www.w3.org/2001/XInclude",
    )
    .expect("declare the xi prefix");

    let mut cursor = root;
    for _ in 0..250 {
        let e = doc.create_element("e").expect("create a nested element");
        doc.append_child(cursor, e)
            .expect("append the nested element");
        cursor = e;
    }
    let include = doc
        .create_element_ns(Some("http://www.w3.org/2001/XInclude"), "xi:include")
        .expect("create xi:include");
    doc.append_child(cursor, include)
        .expect("append xi:include near the bottom of the chain");
    doc.set_attribute(include, "href", "missing.xml")
        .expect("set href to an unreachable resource, forcing xi:fallback");
    let fallback = doc
        .create_element_ns(Some("http://www.w3.org/2001/XInclude"), "xi:fallback")
        .expect("create xi:fallback");
    doc.append_child(include, fallback)
        .expect("append xi:fallback inside xi:include");

    let mut fallback_cursor = fallback;
    for _ in 0..50 {
        let e = doc
            .create_element("e")
            .expect("create a nested fallback element");
        doc.append_child(fallback_cursor, e)
            .expect("append the nested fallback element");
        fallback_cursor = e;
    }

    let err = process_document(&mut doc, "https://example.com/index.xml", &NullLoader)
        .expect_err("the combined depth, reached only through xi:fallback, must be rejected");
    assert!(
        matches!(err, XIncludeError::ElementNestingTooDeep),
        "{err:?}"
    );
}

// ---------------------------------------------------------------------------------- fatal errors

#[test]
fn unsupported_parse_type_is_a_fatal_error() {
    let mut doc = parse(&format!(
        r#"<doc {XI}><xi:include href="a.xml" parse="bogus"/></doc>"#
    ));
    let err = process_document(&mut doc, "https://example.com/index.xml", &NullLoader)
        .expect_err("parse=bogus is invalid");
    assert!(matches!(err, XIncludeError::UnsupportedParseType { .. }));
}

#[test]
fn xpointer_with_parse_text_is_a_fatal_error() {
    let mut doc = parse(&format!(
        r#"<doc {XI}><xi:include href="a.txt" parse="text" xpointer="x"/></doc>"#
    ));
    let err = process_document(&mut doc, "https://example.com/index.xml", &NullLoader)
        .expect_err("xpointer is forbidden with parse=text");
    assert!(matches!(
        err,
        XIncludeError::XpointerNotAllowedForText { .. }
    ));
}

#[test]
fn missing_href_and_xpointer_together_is_a_fatal_error() {
    let mut doc = parse(&format!(r#"<doc {XI}><xi:include/></doc>"#));
    let err = process_document(&mut doc, "https://example.com/index.xml", &NullLoader)
        .expect_err("no href, no xpointer: nothing to include");
    assert!(matches!(err, XIncludeError::MissingHref { .. }));
}

#[test]
fn fragment_identifier_in_href_is_a_fatal_error() {
    let mut doc = parse(&format!(
        r#"<doc {XI}><xi:include href="a.xml#frag"/></doc>"#
    ));
    let err = process_document(&mut doc, "https://example.com/index.xml", &NullLoader)
        .expect_err("fragments are forbidden in href");
    assert!(matches!(err, XIncludeError::FragmentInHref { .. }));
}

#[test]
fn invalid_accept_header_is_a_fatal_error() {
    let mut doc = parse(&format!(
        r#"<doc {XI}><xi:include href="a.xml" accept="text/x-japanese-日本"/></doc>"#
    ));
    let err = process_document(&mut doc, "https://example.com/index.xml", &NullLoader)
        .expect_err("accept must stay within #x20-#x7E");
    assert!(matches!(err, XIncludeError::InvalidHeaderValue { .. }));
}

#[test]
fn malformed_included_xml_is_a_fatal_error() {
    let mut doc = parse(&format!(r#"<doc {XI}><xi:include href="a.xml"/></doc>"#));
    let loader = MapLoader::new().with_text("https://example.com/a.xml", "<not-closed>");

    let err = process_document(&mut doc, "https://example.com/index.xml", &loader)
        .expect_err("non-well-formed XML is fatal, not a resource error");
    assert!(matches!(err, XIncludeError::MalformedXml { .. }));
}

#[test]
fn stray_fallback_outside_include_is_a_fatal_error() {
    let mut doc = parse(&format!(r#"<doc {XI}><xi:fallback/></doc>"#));
    let err = process_document(&mut doc, "https://example.com/index.xml", &NullLoader)
        .expect_err("xi:fallback must be a child of xi:include");
    assert!(matches!(err, XIncludeError::FallbackOutsideInclude { .. }));
}

#[test]
fn unsupported_xinclude_namespace_element_is_a_fatal_error() {
    let mut doc = parse(&format!(r#"<doc {XI}><xi:whatever/></doc>"#));
    let err = process_document(&mut doc, "https://example.com/index.xml", &NullLoader)
        .expect_err("only xi:include and xi:fallback are defined");
    assert!(matches!(
        err,
        XIncludeError::UnsupportedXIncludeElement { .. }
    ));
}

#[test]
fn invalid_base_iri_is_a_fatal_error() {
    let mut doc = parse("<doc/>");
    let err = process_document(&mut doc, "not an iri", &NullLoader)
        .expect_err("the base IRI itself must be valid and absolute");
    assert!(matches!(err, XIncludeError::InvalidBaseIri { .. }));
}

// ------------------------------------------------------------------ §4.1.1 href escaping

/// A `ResourceLoader` that records every URI it is asked for, and answers each one with the
/// same document, so that a test can assert on the *exact* URI the processor dereferenced.
#[derive(Debug, Default)]
struct RecordingLoader {
    asked: std::cell::RefCell<Vec<String>>,
}

impl ResourceLoader for RecordingLoader {
    fn load(
        &self,
        location: &str,
        _accept: Option<&str>,
        _accept_language: Option<&str>,
    ) -> Result<Vec<u8>, LoadError> {
        self.asked.borrow_mut().push(location.to_owned());
        Ok(b"<p>loaded</p>".to_vec())
    }
}

#[test]
fn href_with_a_space_is_escaped_rather_than_rejected() {
    // XInclude 1.0 §4.1.1: a space is disallowed in a URI, so the processor must escape it
    // before resolving — it is emphatically *not* a fatal `InvalidHref`.
    let mut doc = parse(&format!(
        r#"<doc {XI}><xi:include href="my file.xml"/></doc>"#
    ));
    let loader = RecordingLoader::default();

    process_document(&mut doc, "https://example.com/index.xml", &loader)
        .expect("a space in href is escaped, not fatal");

    assert_eq!(
        loader.asked.borrow().as_slice(),
        ["https://example.com/my%20file.xml".to_owned()]
    );
    let p = first_element_named(&doc, "p");
    assert_eq!(doc.text_content(p).as_deref(), Some("loaded"));
}

#[test]
fn href_with_non_ascii_characters_is_escaped_as_utf8() {
    // §4.1.1: "each character outside US-ASCII must be escaped" as the %HH encoding of its
    // UTF-8 bytes — `é` is U+00E9, i.e. C3 A9.
    let accented = "caf\u{e9}.xml";
    let mut doc = parse(&format!(
        r#"<doc {XI}><xi:include href="{accented}"/></doc>"#
    ));
    let loader = RecordingLoader::default();

    process_document(&mut doc, "https://example.com/index.xml", &loader).expect("processes");

    assert_eq!(
        loader.asked.borrow().as_slice(),
        ["https://example.com/caf%C3%A9.xml".to_owned()]
    );
}

#[test]
fn href_that_is_already_escaped_is_not_escaped_twice() {
    let mut doc = parse(&format!(
        r#"<doc {XI}><xi:include href="my%20file.xml"/></doc>"#
    ));
    let loader = RecordingLoader::default();

    process_document(&mut doc, "https://example.com/index.xml", &loader).expect("processes");

    assert_eq!(
        loader.asked.borrow().as_slice(),
        ["https://example.com/my%20file.xml".to_owned()]
    );
}

// ----------------------------------------------------- §3.1 xi:include content model

#[test]
fn two_fallback_children_are_fatal_even_when_the_resource_loads() {
    // §3.1 states the rule unconditionally: the content model of `xi:include` is wrong whether
    // or not the resource turns out to be reachable.
    let mut doc = parse(&format!(
        r#"<doc {XI}><xi:include href="part.xml"><xi:fallback/><xi:fallback/></xi:include></doc>"#
    ));
    let loader = MapLoader::new().with_text("https://example.com/part.xml", "<p>ok</p>");

    let err = process_document(&mut doc, "https://example.com/index.xml", &loader)
        .expect_err("more than one xi:fallback is a fatal error");
    assert!(matches!(err, XIncludeError::FallbackCount { found: 2, .. }));
}

#[test]
fn a_nested_xinclude_namespace_child_is_fatal_even_when_the_resource_loads() {
    let mut doc = parse(&format!(
        r#"<doc {XI}><xi:include href="part.xml"><xi:include href="other.xml"/></xi:include></doc>"#
    ));
    let loader = MapLoader::new()
        .with_text("https://example.com/part.xml", "<p>ok</p>")
        .with_text("https://example.com/other.xml", "<q>other</q>");

    let err = process_document(&mut doc, "https://example.com/index.xml", &loader)
        .expect_err("only xi:fallback may appear inside xi:include");
    assert!(matches!(
        err,
        XIncludeError::UnsupportedXIncludeElement { .. }
    ));
}

#[test]
fn one_fallback_child_remains_legal_when_the_resource_loads() {
    let mut doc = parse(&format!(
        r#"<doc {XI}><xi:include href="part.xml"><xi:fallback>oops</xi:fallback></xi:include></doc>"#
    ));
    let loader = MapLoader::new().with_text("https://example.com/part.xml", "<p>ok</p>");

    process_document(&mut doc, "https://example.com/index.xml", &loader).expect("processes");

    let p = first_element_named(&doc, "p");
    assert_eq!(doc.text_content(p).as_deref(), Some("ok"));
}
