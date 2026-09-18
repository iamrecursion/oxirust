//! Reading the *external* subset of a document type declaration.
//!
//! [XML 1.0 §5.1](https://www.w3.org/TR/xml/#proc-types) lets a non-validating processor leave the
//! external subset unread, and by default this one does. A caller that wants the entities,
//! notations and attribute defaults declared there asks for them with
//! [`ParseOptions::read_external_subset`] *and* supplies the resolver that retrieves the subset —
//! reading it means reading whatever the system identifier points at, which only the application
//! can authorize.
//!
//! Where both subsets declare the same thing the internal one binds, because
//! [XML 1.0 §2.8](https://www.w3.org/TR/xml/#sec-prolog-dtd) has it read first.

use oxixml_infoset::{
    EXTERNAL_SUBSET_NAME, ErrorCode, ExternalEntity, ExternalEntityError, ExternalEntityResolver,
    ParseOptions, Tree, XdmError, XdmNode,
};

/// A resolver backed by one fixed answer, which also remembers what it was asked for.
struct Subset {
    text: Option<String>,
    requests: Vec<(String, Option<String>, String)>,
}

impl Subset {
    fn new(text: &str) -> Self {
        Self {
            text: Some(text.to_owned()),
            requests: Vec::new(),
        }
    }

    /// A resolver that declines everything.
    fn declining() -> Self {
        Self {
            text: None,
            requests: Vec::new(),
        }
    }
}

impl ExternalEntityResolver for Subset {
    fn resolve(
        &mut self,
        entity: &ExternalEntity<'_>,
    ) -> Result<Option<String>, ExternalEntityError> {
        self.requests.push((
            entity.name.to_owned(),
            entity.public_id.map(str::to_owned),
            entity.system_id.to_owned(),
        ));
        Ok(self.text.clone())
    }
}

/// A resolver whose retrieval fails.
struct Failing;

impl ExternalEntityResolver for Failing {
    fn resolve(
        &mut self,
        entity: &ExternalEntity<'_>,
    ) -> Result<Option<String>, ExternalEntityError> {
        Err(ExternalEntityError::Refused {
            system_id: entity.system_id.to_owned(),
            detail: "the test refuses everything",
        })
    }
}

/// Parses `xml` with the external subset read from `dtd`.
fn with_subset(xml: &str, dtd: &str) -> Result<Tree, XdmError> {
    let mut resolver = Subset::new(dtd);
    Tree::parse_str_with_resolver(
        xml,
        ParseOptions::default()
            .read_external_subset(true)
            .document_uri("http://example.com/a/doc.xml"),
        &mut resolver,
    )
}

const DOCUMENT: &str = r#"<!DOCTYPE Section SYSTEM "section.dtd" [<!ENTITY inside "internal">]>
<Section><Media src="picture"/></Section>"#;

const SUBSET: &str = r#"<!NOTATION JPG PUBLIC "JPEG Image">
  <!ENTITY picture SYSTEM "images/first.jpg" NDATA JPG>
  <!ENTITY picture SYSTEM "images/second.jpg" NDATA JPG>
  <!ENTITY inside "external">
  <!ENTITY outside "from the external subset">
  <!ELEMENT Section (Media)>
  <!ATTLIST Media src ENTITY #REQUIRED kind CDATA "photograph">"#;

#[test]
fn the_external_subset_is_not_read_by_default() -> Result<(), XdmError> {
    let mut resolver = Subset::new(SUBSET);
    let tree = Tree::parse_str_with_resolver(DOCUMENT, ParseOptions::default(), &mut resolver)?;
    assert!(resolver.requests.is_empty(), "{:?}", resolver.requests);
    assert!(tree.unparsed_entity_names().is_empty());
    assert!(tree.notation_names().is_empty());
    let media = tree
        .document_element()
        .and_then(|section| section.children().first().cloned())
        .expect("the child element");
    assert_eq!(media.attributes().len(), 1, "only the written attribute");
    Ok(())
}

#[test]
fn the_flag_alone_does_nothing_without_a_resolver() -> Result<(), XdmError> {
    let tree = Tree::parse_str_with(DOCUMENT, ParseOptions::default().read_external_subset(true))?;
    assert!(tree.unparsed_entity_names().is_empty());
    Ok(())
}

#[test]
fn declarations_in_the_external_subset_reach_the_document() -> Result<(), XdmError> {
    let tree = with_subset(DOCUMENT, SUBSET)?;
    let picture = tree.unparsed_entity("picture").expect("`picture`");
    // The first of two declarations of the same name binds, wherever they are written.
    assert_eq!(picture.system_id, "images/first.jpg");
    assert_eq!(picture.notation, "JPG");
    assert_eq!(
        picture.uri.as_deref(),
        Some("http://example.com/a/images/first.jpg")
    );

    let jpg = tree.notation("JPG").expect("`JPG`");
    assert_eq!(jpg.public_id.as_deref(), Some("JPEG Image"));
    assert_eq!(jpg.system_id, None);
    Ok(())
}

#[test]
fn an_attribute_default_in_the_external_subset_applies() -> Result<(), XdmError> {
    let tree = with_subset(DOCUMENT, SUBSET)?;
    let media = tree
        .document_element()
        .and_then(|section| section.children().first().cloned())
        .expect("the child element");
    let attributes: Vec<(String, String)> = media
        .attributes()
        .iter()
        .map(|attribute| {
            (
                attribute
                    .node_name()
                    .map_or_else(String::new, |name| name.to_string()),
                attribute.string_value(),
            )
        })
        .collect();
    assert_eq!(
        attributes,
        [
            ("src".to_owned(), "picture".to_owned()),
            ("kind".to_owned(), "photograph".to_owned())
        ]
    );
    Ok(())
}

#[test]
fn a_namespace_declaration_may_be_defaulted_from_the_external_subset() -> Result<(), XdmError> {
    let tree = with_subset(
        r#"<!DOCTYPE svg SYSTEM "svg.dtd"><svg xmlns="http://www.w3.org/2000/svg"/>"#,
        r#"<!ATTLIST svg xmlns CDATA #IMPLIED
                        xmlns:xlink CDATA #FIXED "http://www.w3.org/1999/xlink">"#,
    )?;
    let mut prefixes: Vec<String> = tree
        .document_element()
        .expect("a document element")
        .namespaces()
        .iter()
        .map(|namespace| {
            namespace
                .node_name()
                .map_or_else(String::new, |name| name.local_name().to_owned())
        })
        .collect();
    prefixes.sort();
    assert_eq!(prefixes, ["", "xlink", "xml"]);
    Ok(())
}

#[test]
fn the_internal_subset_wins_over_the_external_one() -> Result<(), XdmError> {
    let tree = with_subset(
        r#"<!DOCTYPE r SYSTEM "r.dtd" [
             <!ENTITY e "internal">
             <!ATTLIST r a CDATA "internal">
             <!NOTATION n SYSTEM "internal.exe">
           ]><r>&e;</r>"#,
        r#"<!ENTITY e "external">
           <!ATTLIST r a CDATA "external" b CDATA "only external">
           <!NOTATION n SYSTEM "external.exe">"#,
    )?;
    let root = tree.document_element().expect("a document element");
    assert_eq!(root.string_value(), "internal");
    let values: Vec<String> = root
        .attributes()
        .iter()
        .map(XdmNode::string_value)
        .collect();
    // The external subset still contributes what the internal one left undeclared.
    assert_eq!(values, ["internal", "only external"]);
    assert_eq!(
        tree.notation("n").and_then(|n| n.system_id.clone()),
        Some("internal.exe".to_owned())
    );
    Ok(())
}

#[test]
fn a_general_entity_declared_only_externally_becomes_usable() -> Result<(), XdmError> {
    let tree = with_subset(
        r#"<!DOCTYPE r SYSTEM "r.dtd"><r a="[&outside;]">&outside;</r>"#,
        SUBSET,
    )?;
    let root = tree.document_element().expect("a document element");
    assert_eq!(root.string_value(), "from the external subset");
    assert_eq!(
        root.attributes()
            .first()
            .map(XdmNode::string_value)
            .as_deref(),
        Some("[from the external subset]")
    );
    Ok(())
}

#[test]
fn the_subset_is_requested_under_its_own_name() -> Result<(), XdmError> {
    let mut resolver = Subset::new("");
    let _ = Tree::parse_str_with_resolver(
        r#"<!DOCTYPE r PUBLIC "-//x//DTD R//EN" "r.dtd"><r/>"#,
        ParseOptions::default().read_external_subset(true),
        &mut resolver,
    )?;
    assert_eq!(
        resolver.requests,
        [(
            EXTERNAL_SUBSET_NAME.to_owned(),
            Some("-//x//DTD R//EN".to_owned()),
            "r.dtd".to_owned()
        )]
    );
    Ok(())
}

#[test]
fn a_declining_resolver_leaves_the_subset_unread() -> Result<(), XdmError> {
    let mut resolver = Subset::declining();
    let tree = Tree::parse_str_with_resolver(
        r#"<!DOCTYPE r SYSTEM "r.dtd"><r/>"#,
        ParseOptions::default().read_external_subset(true),
        &mut resolver,
    )?;
    assert_eq!(resolver.requests.len(), 1, "it was asked");
    assert!(tree.unparsed_entity_names().is_empty());
    Ok(())
}

#[test]
fn a_failing_resolver_rejects_the_document() {
    let error = Tree::parse_str_with_resolver(
        r#"<!DOCTYPE r SYSTEM "r.dtd"><r/>"#,
        ParseOptions::default().read_external_subset(true),
        &mut Failing,
    )
    .expect_err("retrieval failed and the caller asked for the subset");
    assert_eq!(error.code(), ErrorCode::Fodc0006);
    assert!(
        error.to_string().contains("external subset"),
        "the message must name what could not be retrieved: {error}"
    );
}

#[test]
fn a_declaration_the_document_never_names_is_not_retrieved() -> Result<(), XdmError> {
    // No external identifier means no external subset, whatever the options say.
    let mut resolver = Subset::new(SUBSET);
    let tree = Tree::parse_str_with_resolver(
        r#"<!DOCTYPE r [<!ENTITY e "x">]><r>&e;</r>"#,
        ParseOptions::default().read_external_subset(true),
        &mut resolver,
    )?;
    assert!(resolver.requests.is_empty(), "{:?}", resolver.requests);
    assert_eq!(tree.document_element().expect("root").string_value(), "x");
    Ok(())
}

#[test]
fn a_text_declaration_at_the_start_of_the_subset_is_not_a_declaration() -> Result<(), XdmError> {
    let tree = with_subset(
        r#"<!DOCTYPE r SYSTEM "r.dtd"><r/>"#,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!ATTLIST r a CDATA \"v\">",
    )?;
    let root = tree.document_element().expect("a document element");
    assert_eq!(
        root.attributes()
            .first()
            .map(XdmNode::string_value)
            .as_deref(),
        Some("v")
    );
    Ok(())
}

#[test]
fn conditional_sections_include_and_ignore_what_they_say() -> Result<(), XdmError> {
    let tree = with_subset(
        r#"<!DOCTYPE r SYSTEM "r.dtd"><r/>"#,
        r#"<![INCLUDE[ <!ATTLIST r kept CDATA "yes"> ]]>
           <![IGNORE[ <!ATTLIST r dropped CDATA "no">
             <![INCLUDE[ <!ATTLIST r nested CDATA "no"> ]]>
           ]]>
           <!ATTLIST r after CDATA "yes">"#,
    )?;
    let names: Vec<String> = tree
        .document_element()
        .expect("a document element")
        .attributes()
        .iter()
        .filter_map(XdmNode::node_name)
        .map(|name| name.local_name().to_owned())
        .collect();
    assert_eq!(names, ["kept", "after"]);
    Ok(())
}

#[test]
fn a_document_of_the_transformation_test_suite_reads_its_own_subset() -> Result<(), XdmError> {
    // The suite's `unparsed-entity-uri-050` declares its entities in an external subset and
    // expects the *first* of two declarations of the same name to bind. The file is found through
    // `OXIXML_XML_SUITES_DIR`, and without it there is nothing to check.
    let Some(root) = std::env::var_os("OXIXML_XML_SUITES_DIR") else {
        return Ok(());
    };
    let directory = std::path::Path::new(&root)
        .join("xslt30-test")
        .join("tests")
        .join("fn")
        .join("unparsed-entity-uri");
    let Ok(text) = std::fs::read_to_string(directory.join("unparsed-entity-uri-050.xml")) else {
        return Ok(());
    };
    let mut resolver = oxixml_infoset::FileEntityResolver::new(&directory);
    let tree = Tree::parse_str_with_resolver(
        &text,
        ParseOptions::default().read_external_subset(true),
        &mut resolver,
    )?;
    let entity = tree
        .unparsed_entity("dgr-08sdexample06")
        .expect("the entity the external subset declares");
    assert_eq!(
        entity.system_id,
        r"images\repository\dgr-en-08sdexample06-a.jpg"
    );
    assert_eq!(entity.notation, "JPG");
    assert_eq!(
        tree.notation("JPG").and_then(|jpg| jpg.public_id.clone()),
        Some("JPEG Image".to_owned())
    );
    Ok(())
}

#[test]
fn a_subset_holding_a_forbidden_character_is_rejected() {
    // The retrieved text never went through the well-formedness checker, so the `Char`
    // production is applied to it here.
    let mut resolver = Subset::new("<!ENTITY e \"a\u{1}b\">");
    let error = Tree::parse_str_with_resolver(
        r#"<!DOCTYPE r SYSTEM "r.dtd"><r/>"#,
        ParseOptions::default().read_external_subset(true),
        &mut resolver,
    )
    .expect_err("U+0001 is not a `Char` in XML 1.0");
    assert_eq!(error.code(), ErrorCode::Fodc0006);
}
