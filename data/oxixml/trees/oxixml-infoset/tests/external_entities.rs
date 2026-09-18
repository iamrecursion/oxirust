//! Retrieval of external parsed general entities, and the markup a general entity may stand for.
//!
//! [XML 1.0 §4.3.2](https://www.w3.org/TR/xml/#wf-entities) makes a general parsed entity
//! well-formed when its replacement text matches the `content` production, so an entity may
//! equally stand for character data, for elements, or for a mixture of the two. An *external*
//! one has no replacement text at all until something reads the system identifier, which a
//! non-validating processor need not do — so it is opt-in, through
//! [`Tree::parse_str_with_resolver`] and a caller-supplied resolver.

use std::collections::HashMap;

use oxixml_infoset::{
    ExternalEntity, ExternalEntityError, ExternalEntityResolver, FileEntityResolver, NodeKind,
    NodeRef, ParseOptions, Tree, XdmError, XdmNode,
};

/// A resolver backed by an in-memory table, which is all these tests need to hand a document
/// some replacement text.
#[derive(Debug, Default)]
struct Table {
    entries: HashMap<String, String>,
    /// System identifiers that must be reported as a retrieval failure rather than resolved.
    failing: Vec<String>,
    /// How many times the resolver was consulted.
    calls: usize,
}

impl Table {
    fn with(entries: &[(&str, &str)]) -> Self {
        Self {
            entries: entries
                .iter()
                .map(|&(system_id, text)| (system_id.to_owned(), text.to_owned()))
                .collect(),
            failing: Vec::new(),
            calls: 0,
        }
    }

    fn failing_on(mut self, system_id: &str) -> Self {
        self.failing.push(system_id.to_owned());
        self
    }
}

impl ExternalEntityResolver for Table {
    fn resolve(
        &mut self,
        entity: &ExternalEntity<'_>,
    ) -> Result<Option<String>, ExternalEntityError> {
        self.calls += 1;
        if self.failing.iter().any(|id| id == entity.system_id) {
            return Err(ExternalEntityError::Refused {
                system_id: entity.system_id.to_owned(),
                detail: "this resolver was told to refuse it",
            });
        }
        Ok(self.entries.get(entity.system_id).cloned())
    }
}

/// A document declaring `e` as an external entity with the system identifier `e.ent`.
fn document(reference: &str) -> String {
    format!(r#"<!DOCTYPE r [<!ENTITY e SYSTEM "e.ent">]><r>{reference}</r>"#)
}

fn parse_with(xml: &str, resolver: &mut dyn ExternalEntityResolver) -> Result<Tree, XdmError> {
    Tree::parse_str_with_resolver(xml, ParseOptions::default(), resolver)
}

/// The kinds of the document element's children, which is what says whether the reference became
/// nodes or text.
fn child_kinds(tree: &Tree) -> Vec<NodeKind> {
    tree.document_element()
        .expect("a document element")
        .children()
        .iter()
        .map(XdmNode::node_kind)
        .collect()
}

#[test]
fn an_entity_matrix_covers_every_route_a_reference_can_take() -> Result<(), XdmError> {
    // `(name, the retrieved text, the string value of the root, the kinds of its children)`.
    let rows: &[(&str, &str, &str, &[NodeKind])] = &[
        (
            "character data",
            "just text",
            "just text",
            &[NodeKind::Text],
        ),
        ("one element", "<b>x</b>", "x", &[NodeKind::Element]),
        (
            "several top-level nodes",
            "<b/>mid<c/>",
            "mid",
            &[NodeKind::Element, NodeKind::Text, NodeKind::Element],
        ),
        ("nothing at all", "", "", &[]),
        (
            "a comment and a processing instruction",
            "<!--c--><?t d?>",
            "",
            &[NodeKind::Comment, NodeKind::ProcessingInstruction],
        ),
        (
            "a byte-order mark and a text declaration",
            "\u{FEFF}<?xml version='1.0' encoding='UTF-8'?><b>x</b>",
            "x",
            &[NodeKind::Element],
        ),
    ];
    for &(name, text, expected, kinds) in rows {
        let mut resolver = Table::with(&[("e.ent", text)]);
        let tree = parse_with(&document("&e;"), &mut resolver)?;
        assert_eq!(
            tree.document_element().expect("root").string_value(),
            expected,
            "{name}"
        );
        assert_eq!(child_kinds(&tree), kinds, "{name}");
        assert_eq!(resolver.calls, 1, "{name}");
    }
    Ok(())
}

#[test]
fn without_a_resolver_an_external_entity_leaves_the_document_rejected() {
    let error = Tree::parse_str(&document("&e;")).expect_err("nothing can retrieve it");
    assert!(
        error.to_string().contains("e.ent"),
        "the message must name the system identifier: {error}"
    );
    // A document that never *references* the entity is unaffected by it.
    assert!(Tree::parse_str(&document("plain")).is_ok());
}

#[test]
fn a_resolver_that_declines_leaves_the_document_rejected() {
    let mut resolver = Table::default();
    let error =
        parse_with(&document("&e;"), &mut resolver).expect_err("the resolver knows no such entity");
    assert!(error.to_string().contains("e.ent"), "{error}");
    assert_eq!(resolver.calls, 1);
}

#[test]
fn a_resolver_that_fails_reports_why() {
    let mut resolver = Table::with(&[("e.ent", "<b/>")]).failing_on("e.ent");
    let error = parse_with(&document("&e;"), &mut resolver).expect_err("retrieval failed");
    assert!(
        error.to_string().contains("told to refuse it"),
        "the resolver's own explanation must survive: {error}"
    );
}

#[test]
fn retrieved_text_that_is_not_content_is_rejected() {
    for (name, text) in [
        ("an unclosed element", "<b>"),
        ("a stray end tag", "</b>"),
        ("a mismatched end tag", "<b></c>"),
        ("more closed than opened", "<b/></r>"),
        ("a bare less-than sign", "a < b"),
        ("an unterminated comment", "<!--c"),
        // A `TextDecl` belongs at the very start of an external entity, where it is stripped
        // before parsing; anywhere else it is a declaration in the middle of content.
        ("a misplaced text declaration", "<b/><?xml version='1.0'?>"),
        (
            "a document type declaration",
            "<!DOCTYPE q [<!ENTITY x 'no'>]><b/>",
        ),
    ] {
        let mut resolver = Table::with(&[("e.ent", text)]);
        assert!(
            parse_with(&document("&e;"), &mut resolver).is_err(),
            "{name}: {text:?}"
        );
    }
}

#[test]
fn a_character_the_version_forbids_is_rejected_before_it_reaches_the_tree() {
    // The text never went through the well-formedness check the document itself passed, so the
    // `Char` production has to be applied to it on retrieval.
    let mut resolver = Table::with(&[("e.ent", "a\u{0}b")]);
    let error = parse_with(&document("&e;"), &mut resolver).expect_err("U+0000 is not a `Char`");
    assert!(error.to_string().contains("U+0000"), "{error}");

    // XML 1.1 admits the C0 controls that 1.0 does not.
    let xml11 = r#"<?xml version="1.1"?><!DOCTYPE r [<!ENTITY e SYSTEM "e.ent">]><r>&e;</r>"#;
    let mut bell = Table::with(&[("e.ent", "a\u{7}b")]);
    assert!(
        parse_with(xml11, &mut bell).is_ok(),
        "U+0007 is a 1.1 `Char`"
    );
    let xml10 = document("&e;");
    let mut bell = Table::with(&[("e.ent", "a\u{7}b")]);
    assert!(
        parse_with(&xml10, &mut bell).is_err(),
        "U+0007 is not a 1.0 `Char`"
    );
}

#[test]
fn a_recursive_external_entity_is_bounded_rather_than_followed() {
    for (name, entries) in [
        ("directly recursive", &[("e.ent", "<b>&e;</b>")][..]),
        (
            "mutually recursive",
            &[("a.ent", "<x>&b;</x>"), ("b.ent", "<y>&a;</y>")][..],
        ),
    ] {
        let subset = entries
            .iter()
            .map(|(system_id, _)| {
                let entity = system_id.trim_end_matches(".ent");
                format!(r#"<!ENTITY {entity} SYSTEM "{system_id}">"#)
            })
            .collect::<String>();
        let first = entries[0].0.trim_end_matches(".ent");
        let xml = format!("<!DOCTYPE r [{subset}]><r>&{first};</r>");
        let mut resolver = Table::with(entries);
        assert!(parse_with(&xml, &mut resolver).is_err(), "{name}");
    }
}

#[test]
fn a_spliced_external_entity_resolves_prefixes_where_the_reference_is() -> Result<(), XdmError> {
    let xml = r#"<!DOCTYPE r [<!ENTITY e SYSTEM "e.ent">]>
                 <r xmlns:p="urn:p"><host xmlns="urn:d">&e;</host></r>"#;
    let mut resolver = Table::with(&[("e.ent", "<p:b><c/></p:b>")]);
    let tree = parse_with(xml, &mut resolver)?;
    let host = tree
        .document_element()
        .expect("root")
        .children()
        .into_iter()
        .find(|child| child.node_kind() == NodeKind::Element)
        .expect("the host element");
    let b = host
        .children()
        .first()
        .cloned()
        .expect("the spliced element");
    assert_eq!(
        b.node_name().map(|name| name.to_string()).as_deref(),
        Some("{urn:p}b")
    );
    let c = b.children().first().cloned().expect("its child");
    assert_eq!(
        c.node_name().map(|name| name.to_string()).as_deref(),
        Some("{urn:d}c")
    );

    // A prefix nothing binds at the reference is unbound inside the entity too.
    let mut unbound = Table::with(&[("e.ent", "<q:b/>")]);
    assert!(parse_with(xml, &mut unbound).is_err());
    Ok(())
}

/// [XML Base §4.2](https://www.w3.org/TR/xmlbase/#granularity) bases an element on the parent
/// element *within the same entity*, and on the entity itself when there is none — so a node
/// spliced in from an external entity is based on that entity rather than on the document that
/// referenced it.
#[test]
fn spliced_nodes_take_the_base_uri_of_the_entity_they_came_from() -> Result<(), XdmError> {
    let mut resolver = Table::with(&[("e.ent", "<b/>")]);
    let tree = Tree::parse_str_with_resolver(
        &document("&e;"),
        ParseOptions::default().document_uri("http://example.com/a/b.xml"),
        &mut resolver,
    )?;
    let root = tree.document_element().expect("root");
    assert_eq!(
        root.base_uri().as_deref(),
        Some("http://example.com/a/b.xml"),
        "the document's own content is unaffected"
    );
    let b = root
        .children()
        .first()
        .cloned()
        .expect("the spliced element");
    assert_eq!(b.base_uri().as_deref(), Some("http://example.com/a/e.ent"));
    Ok(())
}

/// Parses `xml` with `entities` retrievable and, when given, a document URI to resolve their
/// system identifiers against.
fn with_entities(
    xml: &str,
    document_uri: Option<&str>,
    entities: &[(&str, &str)],
) -> Result<Tree, XdmError> {
    let options = match document_uri {
        Some(uri) => ParseOptions::default().document_uri(uri),
        None => ParseOptions::default(),
    };
    Tree::parse_str_with_resolver(xml, options, &mut Table::with(entities))
}

/// The kind and base URI of each child of `root`, in document order.
fn child_bases(root: &NodeRef) -> Vec<(NodeKind, Option<String>)> {
    root.children()
        .iter()
        .map(|child| (child.node_kind(), child.base_uri()))
        .collect()
}

/// xslt30 `resolve-uri-021`: an element and a processing instruction spliced in from an
/// external entity are based on *that entity*, not on the document that referenced it.
///
/// Run over both an `http:` and a `file:` document URI: a document read from disk is the shape
/// the case itself has, and a base URI that is not a parseable IRI resolves nothing at all.
#[test]
fn spliced_content_carries_the_entitys_own_base_uri() -> Result<(), XdmError> {
    for base in [
        "http://example.com/fn/reluri18.xml",
        "file:///t/fn/reluri18.xml",
    ] {
        let directory = base
            .strip_suffix("reluri18.xml")
            .expect("the base names the document");
        let tree = with_entities(
            r#"<!DOCTYPE test [
                 <!ENTITY extElement SYSTEM "level1/element.xml">
                 <!ENTITY extPI SYSTEM "level1/pi.xml">
               ]><test>&extElement;&extPI;</test>"#,
            Some(base),
            &[
                (
                    "level1/element.xml",
                    r#"<extEnt attr="x"><sub2>z</sub2></extEnt>"#,
                ),
                ("level1/pi.xml", "<?pi data?>"),
            ],
        )?;
        let test = tree.document_element().expect("a document element");
        assert_eq!(
            test.base_uri().as_deref(),
            Some(base),
            "the referencing document is unaffected"
        );
        assert_eq!(
            child_bases(&test),
            vec![
                (
                    NodeKind::Element,
                    Some(format!("{directory}level1/element.xml"))
                ),
                (
                    NodeKind::ProcessingInstruction,
                    Some(format!("{directory}level1/pi.xml"))
                ),
            ],
            "over {base}"
        );
        // Everything below the spliced element inherits the entity's URI without being
        // recorded separately, and an attribute reads its element's.
        let element = test
            .children()
            .first()
            .cloned()
            .expect("the spliced element");
        assert_eq!(
            element.children().first().and_then(XdmNode::base_uri),
            element.base_uri()
        );
        assert_eq!(
            element.attributes().first().and_then(XdmNode::base_uri),
            element.base_uri()
        );
    }
    Ok(())
}

/// An entity referenced from inside another bases its content on itself, and the outer
/// entity's base URI comes back afterwards. A relative system identifier is resolved
/// against the document, as XML 1.0 §4.2.2 does for a declaration in the internal subset.
#[test]
fn nested_entities_each_base_their_own_content() -> Result<(), XdmError> {
    let tree = with_entities(
        r#"<!DOCTYPE r [
             <!ENTITY outer SYSTEM "a/outer.xml">
             <!ENTITY inner SYSTEM "b/inner.xml">
           ]><r>&outer;<after/></r>"#,
        Some("http://example.com/d/r.xml"),
        &[
            ("a/outer.xml", "<one>&inner;</one><two/>"),
            ("b/inner.xml", "<deep/>"),
        ],
    )?;
    let root = tree.document_element().expect("a document element");
    let bases: Vec<Option<String>> = root.children().iter().map(XdmNode::base_uri).collect();
    assert_eq!(
        bases,
        vec![
            Some("http://example.com/d/a/outer.xml".to_owned()),
            Some("http://example.com/d/a/outer.xml".to_owned()),
            Some("http://example.com/d/r.xml".to_owned()),
        ],
        "both of the outer entity's elements, then the document's own"
    );
    let one = root.children().first().cloned().expect("the first element");
    assert_eq!(
        one.children()
            .first()
            .and_then(XdmNode::base_uri)
            .as_deref(),
        Some("http://example.com/d/b/inner.xml"),
        "the inner entity bases its own content"
    );
    Ok(())
}

/// An `xml:base` written inside an entity is resolved against the entity's URI, and its
/// descendants inherit the result rather than the entity URI a second time.
#[test]
fn xml_base_inside_an_entity_resolves_against_the_entity() -> Result<(), XdmError> {
    let tree = with_entities(
        r#"<!DOCTYPE r [<!ENTITY e SYSTEM "a/e.xml">]><r>&e;</r>"#,
        Some("http://example.com/d/r.xml"),
        &[("a/e.xml", r#"<b xml:base="sub/"><c/></b>"#)],
    )?;
    let b = tree
        .document_element()
        .expect("a document element")
        .children()
        .first()
        .cloned()
        .expect("the spliced element");
    assert_eq!(b.base_uri().as_deref(), Some("http://example.com/d/a/sub/"));
    assert_eq!(
        b.children().first().and_then(XdmNode::base_uri),
        b.base_uri()
    );
    Ok(())
}

/// A comment and a text node have no `[base URI]` property of their own, and an *internal*
/// entity does not move the base URI at all: its replacement text is part of the entity that
/// referenced it.
#[test]
fn only_the_kinds_with_a_base_uri_property_take_the_entitys() -> Result<(), XdmError> {
    let tree = with_entities(
        r#"<!DOCTYPE r [
             <!ENTITY e SYSTEM "a/e.xml">
             <!ENTITY i "<in/>">
           ]><r>&e;&i;</r>"#,
        Some("http://example.com/d/r.xml"),
        &[("a/e.xml", "<!--c-->text<el/>")],
    )?;
    let root = tree.document_element().expect("a document element");
    assert_eq!(
        child_bases(&root),
        vec![
            (
                NodeKind::Comment,
                Some("http://example.com/d/r.xml".to_owned())
            ),
            (
                NodeKind::Text,
                Some("http://example.com/d/r.xml".to_owned())
            ),
            (
                NodeKind::Element,
                Some("http://example.com/d/a/e.xml".to_owned())
            ),
            (
                NodeKind::Element,
                Some("http://example.com/d/r.xml".to_owned())
            ),
        ]
    );
    Ok(())
}

/// With no document URI there is nothing to resolve a relative system identifier against,
/// so the spliced content keeps the base URI of the reference point — which is `None` for a
/// document that has none at all.
#[test]
fn without_a_document_uri_splicing_changes_no_base_uri() -> Result<(), XdmError> {
    let xml = r#"<!DOCTYPE r [<!ENTITY e SYSTEM "a/e.xml">]><r>&e;</r>"#;
    let entities: &[(&str, &str)] = &[("a/e.xml", "<b/>")];

    let tree = with_entities(xml, None, entities)?;
    let b = tree
        .document_element()
        .expect("a document element")
        .children()
        .first()
        .cloned()
        .expect("the spliced element");
    assert_eq!(b.base_uri(), None);

    // An *absolute* system identifier needs nothing to resolve against and is still used.
    let absolute: &[(&str, &str)] = &[("http://elsewhere.example/e.xml", "<b/>")];
    let tree = with_entities(
        r#"<!DOCTYPE r [<!ENTITY e SYSTEM "http://elsewhere.example/e.xml">]><r>&e;</r>"#,
        None,
        absolute,
    )?;
    let b = tree
        .document_element()
        .expect("a document element")
        .children()
        .first()
        .cloned()
        .expect("the spliced element");
    assert_eq!(
        b.base_uri().as_deref(),
        Some("http://elsewhere.example/e.xml")
    );
    Ok(())
}

#[test]
fn an_internal_declaration_of_the_same_name_binds_first() -> Result<(), XdmError> {
    // XML 1.0 §4.2: the first declaration of an entity binds, whichever kind it is.
    let mut resolver = Table::with(&[("e.ent", "external")]);
    let internal_first =
        r#"<!DOCTYPE r [<!ENTITY e "internal"><!ENTITY e SYSTEM "e.ent">]><r>&e;</r>"#;
    let tree = parse_with(internal_first, &mut resolver)?;
    assert_eq!(
        tree.document_element().expect("root").string_value(),
        "internal"
    );
    assert_eq!(resolver.calls, 0, "the external declaration never binds");

    let external_first =
        r#"<!DOCTYPE r [<!ENTITY e SYSTEM "e.ent"><!ENTITY e "internal">]><r>&e;</r>"#;
    let tree = parse_with(external_first, &mut resolver)?;
    assert_eq!(
        tree.document_element().expect("root").string_value(),
        "external"
    );
    assert_eq!(resolver.calls, 1);
    Ok(())
}

#[test]
fn an_unparsed_entity_is_never_retrieved() -> Result<(), XdmError> {
    // An `NDATA` entity's content is not XML, so it is a property of the document node rather
    // than something a reference may include — and referencing one is forbidden outright.
    let xml = r#"<!DOCTYPE r [
        <!NOTATION png SYSTEM "image/png">
        <!ENTITY logo SYSTEM "logo.png" NDATA png>
    ]><r/>"#;
    let mut resolver = Table::with(&[("logo.png", "<b/>")]);
    let tree = parse_with(xml, &mut resolver)?;
    assert_eq!(tree.unparsed_entity_names(), ["logo"]);
    assert_eq!(resolver.calls, 0);
    Ok(())
}

#[test]
fn the_file_resolver_reads_below_its_base_and_refuses_everything_else() -> Result<(), XdmError> {
    let dir = std::env::temp_dir().join("oxixml-infoset-entity-corpus");
    let nested = dir.join("sub");
    std::fs::create_dir_all(&nested).expect("a writable temporary directory");
    std::fs::write(nested.join("chapter.ent"), "<p>chapter one</p>")
        .expect("a writable temporary file");

    let xml = r#"<!DOCTYPE d [<!ENTITY chapter SYSTEM "sub/chapter.ent">]><d>&chapter;</d>"#;
    let mut resolver = FileEntityResolver::new(&dir);
    let tree = parse_with(xml, &mut resolver)?;
    assert_eq!(
        tree.document_element().expect("root").string_value(),
        "chapter one"
    );

    // A system identifier that climbs out of the base directory is refused without any I/O…
    let escaping = r#"<!DOCTYPE d [<!ENTITY e SYSTEM "../../etc/passwd">]><d>&e;</d>"#;
    let error =
        parse_with(escaping, &mut FileEntityResolver::new(&dir)).expect_err("outside the base");
    assert!(
        error.to_string().contains("climbs out of the base"),
        "{error}"
    );

    // …as is every scheme but `file:`, so a document cannot open a network connection.
    let remote = r#"<!DOCTYPE d [<!ENTITY e SYSTEM "http://example.invalid/e.ent">]><d>&e;</d>"#;
    let error =
        parse_with(remote, &mut FileEntityResolver::new(&dir)).expect_err("a remote identifier");
    assert!(
        error.to_string().contains("only relative and `file:`"),
        "{error}"
    );

    // The size cap is enforced before anything is parsed.
    let error = parse_with(xml, &mut FileEntityResolver::new(&dir).with_max_bytes(4))
        .expect_err("over the ceiling");
    assert!(error.to_string().contains("4-byte limit"), "{error}");

    let _ = std::fs::remove_file(nested.join("chapter.ent"));
    Ok(())
}

#[test]
fn an_element_declaration_beside_the_entity_does_not_get_in_the_way() -> Result<(), XdmError> {
    // The shape a document takes when it declares both the content model of its root and the
    // external entity that fills it: everything but the entity declarations is skipped, and the
    // entity's markup still lands where the reference was.
    let dir = std::env::temp_dir().join("oxixml-infoset-entity-corpus-declared");
    std::fs::create_dir_all(&dir).expect("a writable temporary directory");
    std::fs::write(dir.join("foo.entity"), "<bar>baz</bar>").expect("a writable temporary file");

    let xml = "<!DOCTYPE a [<!ELEMENT a (#PCDATA)>\
               <!ENTITY foo SYSTEM 'foo.entity'>]><a>&foo;</a>";
    let tree = parse_with(xml, &mut FileEntityResolver::new(&dir))?;
    let root = tree.document_element().expect("a document element");
    assert_eq!(child_kinds(&tree), [NodeKind::Element]);
    let bar = root
        .children()
        .first()
        .cloned()
        .expect("the spliced element");
    assert_eq!(
        bar.node_name().map(|name| name.to_string()).as_deref(),
        Some("bar")
    );
    assert_eq!(bar.string_value(), "baz");

    let _ = std::fs::remove_file(dir.join("foo.entity"));
    Ok(())
}
