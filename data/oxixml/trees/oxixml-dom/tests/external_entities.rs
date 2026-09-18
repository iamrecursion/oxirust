//! Retrieval of external parsed general entities, which is opt-in.
//!
//! The default — every entry point that does not take a resolver — must behave
//! exactly as it did before the feature existed: an external entity is never
//! retrieved, and its reference stays in the tree as a node of its own.

use std::collections::HashMap;

use oxixml_dom::build::{self, BuilderOptions};
use oxixml_dom::external::{
    ExternalEntity, ExternalEntityError, ExternalEntityResolver, FileEntityResolver,
};
use oxixml_dom::{BuildError, NodeKind};

/// A resolver backed by a table, so that the splicing can be tested without
/// touching the file system.
#[derive(Default)]
struct Table {
    entries: HashMap<String, String>,
    calls: usize,
}

impl Table {
    fn with(mut self, system_id: &str, text: &str) -> Self {
        self.entries.insert(system_id.to_owned(), text.to_owned());
        self
    }
}

impl ExternalEntityResolver for Table {
    fn resolve(
        &mut self,
        entity: &ExternalEntity<'_>,
    ) -> Result<Option<String>, ExternalEntityError> {
        self.calls += 1;
        Ok(self.entries.get(entity.system_id).cloned())
    }
}

/// A resolver that always fails, to prove the failure is reported and not
/// silently swallowed.
struct Broken;

impl ExternalEntityResolver for Broken {
    fn resolve(
        &mut self,
        entity: &ExternalEntity<'_>,
    ) -> Result<Option<String>, ExternalEntityError> {
        Err(ExternalEntityError::Refused {
            system_id: entity.system_id.to_owned(),
            detail: "this resolver retrieves nothing",
        })
    }
}

const DOC: &str =
    r#"<!DOCTYPE a [<!ELEMENT a (#PCDATA)><!ENTITY foo SYSTEM "foo.entity">]><a>&foo;</a>"#;

#[test]
fn without_a_resolver_an_external_entity_is_not_retrieved() {
    let doc = build::parse_str(DOC).expect("well-formed");
    let root = doc.document_element().expect("a document element");
    let child = doc.children(root).next().expect("one child");
    assert_eq!(doc.kind(child), Some(NodeKind::EntityReference));
    assert!(doc.to_xml().ends_with("<a>&foo;</a>"), "{}", doc.to_xml());
}

#[test]
fn a_resolver_that_declines_leaves_the_reference_alone() {
    let mut resolver = Table::default();
    let doc = build::parse_str_with_resolver(DOC, &BuilderOptions::new(), &mut resolver)
        .expect("well-formed");
    assert_eq!(resolver.calls, 1);
    let root = doc.document_element().expect("a document element");
    let child = doc.children(root).next().expect("one child");
    assert_eq!(doc.kind(child), Some(NodeKind::EntityReference));
}

#[test]
fn a_resolved_entity_is_spliced_in_as_markup() {
    let mut resolver = Table::default().with("foo.entity", "<bar>baz</bar>");
    let doc = build::parse_str_with_resolver(DOC, &BuilderOptions::new(), &mut resolver)
        .expect("well-formed");
    assert!(
        doc.to_xml().ends_with("<a><bar>baz</bar></a>"),
        "{}",
        doc.to_xml()
    );
}

#[test]
fn a_resolved_entity_may_hold_text_several_elements_or_nothing() {
    for (text, expected) in [
        ("plain text", "<a>plain text</a>"),
        ("<b/><c/>", "<a><b/><c/></a>"),
        ("", "<a/>"),
        ("<b>x</b> and <c>y</c>", "<a><b>x</b> and <c>y</c></a>"),
        ("<!-- note -->", "<a><!-- note --></a>"),
        ("<![CDATA[<raw>]]>", "<a><![CDATA[<raw>]]></a>"),
    ] {
        let mut resolver = Table::default().with("foo.entity", text);
        let doc = build::parse_str_with_resolver(DOC, &BuilderOptions::new(), &mut resolver)
            .unwrap_or_else(|error| panic!("{text:?}: {error}"));
        let xml = doc.to_xml();
        assert!(xml.ends_with(expected), "{text:?} gave {xml}");
    }
}

#[test]
fn a_text_declaration_is_not_part_of_the_content() {
    let mut resolver =
        Table::default().with("foo.entity", "<?xml version='1.0' encoding='UTF-8'?><b/>");
    let doc = build::parse_str_with_resolver(DOC, &BuilderOptions::new(), &mut resolver)
        .expect("well-formed");
    assert!(doc.to_xml().ends_with("<a><b/></a>"), "{}", doc.to_xml());
}

#[test]
fn the_namespace_scope_of_the_reference_applies_inside_the_entity() {
    let source = r#"<!DOCTYPE a [<!ENTITY foo SYSTEM "foo.entity">]>
        <a xmlns="urn:example:default" xmlns:p="urn:example:p"><p:wrap>&foo;</p:wrap></a>"#;
    let mut resolver = Table::default().with("foo.entity", "<plain/><p:qualified/>");
    let doc = build::parse_str_with_resolver(source, &BuilderOptions::new(), &mut resolver)
        .expect("well-formed");
    let namespaces: Vec<Option<String>> = doc
        .descendants(doc.root())
        .filter(|node| doc.kind(*node) == Some(NodeKind::Element))
        .filter_map(|node| doc.element(node))
        .map(|element| element.namespace_uri().map(str::to_owned))
        .collect();
    assert_eq!(
        namespaces,
        vec![
            Some("urn:example:default".to_owned()),
            Some("urn:example:p".to_owned()),
            Some("urn:example:default".to_owned()),
            Some("urn:example:p".to_owned()),
        ]
    );
}

#[test]
fn a_nested_external_entity_is_retrieved_too() {
    let source = r#"<!DOCTYPE a [<!ENTITY outer SYSTEM "outer.entity"><!ENTITY inner SYSTEM "inner.entity">]><a>&outer;</a>"#;
    let mut resolver = Table::default()
        .with("outer.entity", "<b>&inner;</b>")
        .with("inner.entity", "<c/>");
    let doc = build::parse_str_with_resolver(source, &BuilderOptions::new(), &mut resolver)
        .expect("well-formed");
    assert!(
        doc.to_xml().ends_with("<a><b><c/></b></a>"),
        "{}",
        doc.to_xml()
    );
}

#[test]
fn a_self_referential_external_entity_is_refused() {
    let source = r#"<!DOCTYPE a [<!ENTITY loop SYSTEM "loop.entity">]><a>&loop;</a>"#;
    let mut resolver = Table::default().with("loop.entity", "<b>&loop;</b>");
    let error = build::parse_str_with_resolver(source, &BuilderOptions::new(), &mut resolver)
        .expect_err("a cycle must not be followed");
    assert!(
        matches!(error, BuildError::EntityExpansion { ref name, .. } if name == "loop"),
        "{error}"
    );
}

#[test]
fn a_failing_resolver_is_reported() {
    let error = build::parse_str_with_resolver(DOC, &BuilderOptions::new(), &mut Broken)
        .expect_err("the resolver failed");
    assert!(
        matches!(error, BuildError::EntityExpansion { ref name, .. } if name == "foo"),
        "{error}"
    );
    assert!(error.to_string().contains("retrieves nothing"), "{error}");
}

#[test]
fn an_internal_declaration_still_wins_over_an_external_one() {
    // XML 1.0 §4.2: the first declaration of an entity binds, so an internal
    // one seen first is what a reference means, resolver or no resolver.
    let source =
        r#"<!DOCTYPE a [<!ENTITY foo "internal"><!ENTITY foo SYSTEM "foo.entity">]><a>&foo;</a>"#;
    let mut resolver = Table::default().with("foo.entity", "<bar/>");
    let doc = build::parse_str_with_resolver(source, &BuilderOptions::new(), &mut resolver)
        .expect("well-formed");
    assert!(
        doc.to_xml().ends_with("<a>internal</a>"),
        "{}",
        doc.to_xml()
    );
    assert_eq!(resolver.calls, 0);
}

#[test]
fn an_unparsed_entity_is_never_retrieved() {
    // `NDATA` makes it unparsed: it can only be an `ENTITY` attribute value, so
    // a reference to it stays unresolved.
    let source = r#"<!DOCTYPE a [<!NOTATION g SYSTEM "g"><!ENTITY pic SYSTEM "p.png" NDATA g>]><a>&pic;</a>"#;
    let mut resolver = Table::default().with("p.png", "<b/>");
    let doc = build::parse_str_with_resolver(source, &BuilderOptions::new(), &mut resolver)
        .expect("well-formed");
    let root = doc.document_element().expect("a document element");
    let child = doc.children(root).next().expect("one child");
    assert_eq!(doc.kind(child), Some(NodeKind::EntityReference));
    assert_eq!(resolver.calls, 0);
}

#[test]
fn the_file_resolver_reads_the_entity_from_disk() {
    let dir = std::env::temp_dir().join("oxixml-dom-external-entities-test");
    std::fs::create_dir_all(&dir).expect("a writable temporary directory");
    let path = dir.join("foo.entity");
    std::fs::write(&path, "<bar>baz</bar>").expect("a writable temporary file");

    let mut resolver = FileEntityResolver::new(&dir);
    let doc = build::parse_str_with_resolver(DOC, &BuilderOptions::new(), &mut resolver)
        .expect("well-formed");
    assert!(
        doc.to_xml().ends_with("<a><bar>baz</bar></a>"),
        "{}",
        doc.to_xml()
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn the_file_resolver_refuses_to_leave_its_base_directory() {
    let source = r#"<!DOCTYPE a [<!ENTITY foo SYSTEM "../../../../etc/passwd">]><a>&foo;</a>"#;
    let mut resolver = FileEntityResolver::new(std::env::temp_dir());
    let error = build::parse_str_with_resolver(source, &BuilderOptions::new(), &mut resolver)
        .expect_err("the identifier climbs out of the base directory");
    assert!(error.to_string().contains("refused to retrieve"), "{error}");
}

#[test]
fn the_file_resolver_never_opens_a_network_connection() {
    for system_id in [
        "http://example.invalid/foo.entity",
        "https://example.invalid/foo.entity",
        "ftp://example.invalid/foo.entity",
    ] {
        let source = format!(r#"<!DOCTYPE a [<!ENTITY foo SYSTEM "{system_id}">]><a>&foo;</a>"#);
        let mut resolver = FileEntityResolver::new(std::env::temp_dir());
        let error = build::parse_str_with_resolver(&source, &BuilderOptions::new(), &mut resolver)
            .expect_err("a remote identifier must be refused");
        assert!(
            error.to_string().contains("only relative and `file:`"),
            "{system_id}: {error}"
        );
    }
}

#[test]
fn entity_text_that_is_not_balanced_content_is_rejected() {
    // The `content` production requires every element the entity opens to be
    // closed in it, and forbids it from closing anything it did not open. Text
    // that tries either — including text aimed at the wrapper the splicing puts
    // around it — is an error, never a silently truncated tree and never a
    // panic.
    for text in [
        "</x>",
        "</e>",
        "</e><e>",
        "<a>",
        "<a></b>",
        "&amp;</e>",
        "<![CDATA[x",
    ] {
        let mut resolver = Table::default().with("foo.entity", text);
        assert!(
            build::parse_str_with_resolver(DOC, &BuilderOptions::new(), &mut resolver).is_err(),
            "{text:?} must be rejected"
        );
    }
    // A stray `]]>` in content is data, not markup, and is escaped on output.
    let mut resolver = Table::default().with("foo.entity", "]]>");
    let doc = build::parse_str_with_resolver(DOC, &BuilderOptions::new(), &mut resolver)
        .expect("`]]>` is character data");
    assert!(doc.to_xml().ends_with("<a>]]&gt;</a>"), "{}", doc.to_xml());
}
