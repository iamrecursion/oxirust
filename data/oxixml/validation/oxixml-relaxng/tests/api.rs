//! Integration tests for the public API: the entry points, the resolver, the
//! error surface, and the datatype libraries.

use std::io::Cursor;

use oxixml_relaxng::{
    RelaxNg, RelaxNgBuilder, ResolveError, SchemaError, SchemaErrorKind, SchemaResolver,
    ValidationErrorKind,
};

const RNG: &str = r#"xmlns="http://relaxng.org/ns/structure/1.0""#;

#[test]
fn every_entry_point_accepts_the_same_schema() -> Result<(), Box<dyn std::error::Error>> {
    let xml = format!(r#"<element {RNG} name="a"><text/></element>"#);
    let compact = "element a { text }";

    let from_xml = RelaxNg::from_xml(&xml)?;
    let from_compact = RelaxNg::from_compact(compact)?;

    for schema in [&from_xml, &from_compact] {
        assert!(schema.validate("<a>x</a>").is_ok());
        assert!(schema.validate_reader(Cursor::new(b"<a>x</a>")).is_ok());
        let document = oxixml_dom::build::parse_str("<a>x</a>")?;
        assert!(schema.validate_document(&document).is_ok());
        assert!(schema.validate("<b/>").is_err());
    }
    Ok(())
}

#[test]
fn schemas_load_from_files_with_includes() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::temp_dir().join(format!(
        "oxixml-relaxng-api-{}-{}",
        std::process::id(),
        line!()
    ));
    std::fs::create_dir_all(&directory)?;

    std::fs::write(
        directory.join("common.rng"),
        format!(
            r#"<grammar {RNG}>
                 <start><notAllowed/></start>
                 <define name="title"><element name="title"><text/></element></define>
               </grammar>"#
        ),
    )?;
    std::fs::write(
        directory.join("book.rng"),
        format!(
            r#"<grammar {RNG}>
                 <include href="common.rng">
                   <start><ref name="book"/></start>
                 </include>
                 <define name="book">
                   <element name="book"><ref name="title"/></element>
                 </define>
               </grammar>"#
        ),
    )?;
    std::fs::write(
        directory.join("book.rnc"),
        "include \"common.rng\" { start = book }\nbook = element book { title }\n",
    )?;

    for name in ["book.rng", "book.rnc"] {
        let schema = RelaxNg::from_file(directory.join(name))?;
        assert!(
            schema
                .validate("<book><title>Kokoro</title></book>")
                .is_ok(),
            "{name}"
        );
        assert!(schema.validate("<book/>").is_err(), "{name}");
    }

    std::fs::remove_dir_all(&directory)?;
    Ok(())
}

struct InMemory;

impl SchemaResolver for InMemory {
    fn resolve(&self, uri: &str) -> Result<String, ResolveError> {
        match uri {
            "urn:example:card" => Ok(format!(r#"<element {RNG} name="card"><text/></element>"#)),
            other => Err(ResolveError::new(format!("no such resource: {other}"))),
        }
    }
}

#[test]
fn a_custom_resolver_serves_external_references() -> Result<(), SchemaError> {
    let schema = RelaxNgBuilder::new()
        .resolver(Box::new(InMemory))
        .compile_xml(&format!(
            r#"<element {RNG} name="deck">
                 <zeroOrMore><externalRef href="urn:example:card"/></zeroOrMore>
               </element>"#
        ))?;
    assert!(schema.validate("<deck><card>ace</card></deck>").is_ok());
    assert!(schema.validate("<deck><joker/></deck>").is_err());

    let missing = RelaxNgBuilder::new()
        .resolver(Box::new(InMemory))
        .compile_xml(&format!(
            r#"<element {RNG} name="deck"><externalRef href="urn:example:nope"/></element>"#
        ));
    assert_eq!(
        missing.err().map(|error| error.kind()),
        Some(SchemaErrorKind::Resource)
    );
    Ok(())
}

#[test]
fn schema_errors_carry_their_specification_section() {
    let cases: [(&str, SchemaErrorKind, &str); 4] = [
        ("<thisIsJunk/>", SchemaErrorKind::Grammar, "3"),
        (
            r#"<element name="a"><data type="nope"/></element>"#,
            SchemaErrorKind::Datatype,
            "4.16",
        ),
        (
            r#"<element name="a"><list><list><text/></list></list></element>"#,
            SchemaErrorKind::Restriction,
            "7.1.3",
        ),
        (
            r#"<element name="a"><group><data type="token"/><text/></group></element>"#,
            SchemaErrorKind::Restriction,
            "7.2",
        ),
    ];
    for (body, kind, section) in cases {
        let schema = body.replace("<element", &format!("<element {RNG}"));
        let schema = schema.replace("<thisIsJunk/>", &format!("<thisIsJunk {RNG}/>"));
        let error = RelaxNg::from_xml(&schema).expect_err(&format!("{schema} must be rejected"));
        assert_eq!(error.kind(), kind, "{schema}");
        assert_eq!(error.section(), section, "{schema}");
    }
}

#[test]
fn validation_errors_are_classified() -> Result<(), SchemaError> {
    let schema = RelaxNg::from_compact(
        r#"element r { attribute a { xsd:int }, element b { empty }, element c { empty } }"#,
    )?;

    let errors = schema
        .validate("<r><x/></r>")
        .expect_err("nothing matches here");
    let kinds: Vec<ValidationErrorKind> = errors
        .iter()
        .map(oxixml_relaxng::RelaxNgError::kind)
        .collect();
    assert!(
        kinds.contains(&ValidationErrorKind::UnexpectedElement),
        "{errors:?}"
    );
    assert!(
        kinds.contains(&ValidationErrorKind::UnexpectedAttribute),
        "{errors:?}"
    );

    let errors = schema
        .validate(r#"<r a="x"><b/><c/></r>"#)
        .expect_err("`x` is not an int");
    assert_eq!(errors[0].kind(), ValidationErrorKind::UnexpectedAttribute);

    let errors = schema
        .validate(r#"<r a="1"><b/>text<c/></r>"#)
        .expect_err("text is not allowed");
    assert_eq!(errors[0].kind(), ValidationErrorKind::UnexpectedText);

    let errors = schema
        .validate("<r>")
        .expect_err("the instance is not well-formed");
    assert_eq!(errors[0].kind(), ValidationErrorKind::NotWellFormed);
    Ok(())
}

#[test]
fn the_xsd_datatype_library_is_available_with_facets() -> Result<(), SchemaError> {
    let schema = RelaxNg::from_compact(
        r#"element r {
             element n { xsd:decimal { minInclusive = "0" fractionDigits = "2" } },
             element s { xsd:token { pattern = "[a-z]{3}" } },
             element d { xsd:dateTime },
             element l { list { xsd:NCName+ } }
           }"#,
    )?;
    assert!(
        schema
            .validate("<r><n>1.25</n><s>abc</s><d>2000-01-01T00:00:00Z</d><l>a b c</l></r>")
            .is_ok()
    );
    assert!(
        schema
            .validate("<r><n>-1</n><s>abc</s><d>2000-01-01T00:00:00Z</d><l>a</l></r>")
            .is_err()
    );
    assert!(
        schema
            .validate("<r><n>1.255</n><s>abc</s><d>2000-01-01T00:00:00Z</d><l>a</l></r>")
            .is_err()
    );
    assert!(
        schema
            .validate("<r><n>1</n><s>ABC</s><d>2000-01-01T00:00:00Z</d><l>a</l></r>")
            .is_err()
    );
    assert!(
        schema
            .validate("<r><n>1</n><s>abc</s><d>not-a-date</d><l>a</l></r>")
            .is_err()
    );
    assert!(
        schema
            .validate("<r><n>1</n><s>abc</s><d>2000-01-01T00:00:00Z</d><l>1a</l></r>")
            .is_err()
    );
    Ok(())
}

#[test]
fn values_compare_in_the_value_space() -> Result<(), SchemaError> {
    let schema = RelaxNg::from_compact(r#"element r { xsd:decimal "1.0" }"#)?;
    assert!(schema.validate("<r>1.00</r>").is_ok());
    assert!(schema.validate("<r> 1 </r>").is_ok());
    assert!(schema.validate("<r>1.01</r>").is_err());

    // The built-in library: `token` compares after whitespace normalization,
    // `string` does not.
    let token = RelaxNg::from_compact(r#"element r { "a b" }"#)?;
    assert!(token.validate("<r>  a   b  </r>").is_ok());
    let string = RelaxNg::from_xml(&format!(
        r#"<element {RNG} name="r"><value type="string">a b</value></element>"#
    ))?;
    assert!(string.validate("<r>a b</r>").is_ok());
    assert!(string.validate("<r>  a   b  </r>").is_err());
    Ok(())
}

#[test]
fn deeply_nested_parentheses_in_the_compact_syntax_are_a_clean_error_not_a_stack_overflow() {
    // The compact-syntax parser's pattern/primary productions are mutually recursive with no
    // depth limit of their own; parenthesised nesting tens of thousands deep used to exhaust
    // the native call stack (a genuine, uncatchable process abort) instead of failing to parse.
    let depth = 5_000;
    let schema = format!(
        "element root {{ {}text{} }}",
        "(".repeat(depth),
        ")".repeat(depth)
    );
    let error =
        RelaxNg::from_compact(&schema).expect_err("pathological pattern nesting must be rejected");
    assert_eq!(error.kind(), SchemaErrorKind::Syntax);
}

#[test]
fn deeply_nested_div_blocks_are_a_clean_error_not_a_stack_overflow() {
    // `div { div { … } }` recurses through `component`/`grammar_content` without ever calling
    // `pattern` or `primary` — a separate mutually-recursive family from ordinary pattern
    // nesting, and one a guard placed only on `primary` would miss entirely.
    let depth = 5_000;
    let schema = format!("{}{}", "div { ".repeat(depth), "}".repeat(depth));
    let error =
        RelaxNg::from_compact(&schema).expect_err("pathological div nesting must be rejected");
    assert_eq!(error.kind(), SchemaErrorKind::Syntax);
}

#[test]
fn a_flat_grammar_with_many_non_nested_defines_still_compiles()
-> Result<(), Box<dyn std::error::Error>> {
    // The nesting-depth guard must track how deeply constructs are nested inside one another,
    // not how many of them a document contains in total — a wide, flat grammar is completely
    // ordinary and must not be mistaken for pathological nesting.
    let count = 1_000;
    let mut schema = String::from("start = a0\n");
    for i in 0..count {
        schema.push_str(&format!("a{i} = element a{i} {{ text }}\n"));
    }
    RelaxNg::from_compact(&schema)?;
    Ok(())
}

/// `depth` nested `<element name="a">` elements around a `<text/>` leaf, in the XML syntax —
/// and the document of `depth` nested `<a>` elements around a text node that matches it.
fn nested_element_xml(depth: usize) -> (String, String) {
    let mut schema = format!(r#"<element {RNG} name="a">"#);
    for _ in 1..depth {
        schema.push_str(r#"<element name="a">"#);
    }
    schema.push_str("<text/>");
    for _ in 1..depth {
        schema.push_str("</element>");
    }
    schema.push_str("</element>");

    let mut document = "<a>".repeat(depth);
    document.push('x');
    document.push_str(&"</a>".repeat(depth));
    (schema, document)
}

#[test]
fn deeply_nested_elements_in_the_xml_syntax_are_a_clean_error_not_a_stack_overflow() {
    // Unlike the compact-syntax parser (guarded above), the XML syntax's own reader already
    // bounds raw element nesting — but `xml_syntax::convert`, which turns that already-bounded
    // DOM tree into this crate's own data model, recurses once per level with no depth guard of
    // its own, and pays enough stack per level (a cloned namespace map, an attribute list) that
    // a schema comfortably within the reader's own limit could still exhaust the native stack (a
    // genuine, uncatchable process abort) before this was fixed.
    let (schema, _) = nested_element_xml(1_000);
    let error =
        RelaxNg::from_xml(&schema).expect_err("pathological element nesting must be rejected");
    assert_eq!(error.kind(), SchemaErrorKind::Syntax);
}

#[test]
fn nested_elements_in_the_xml_syntax_at_the_depth_cap_still_compile()
-> Result<(), Box<dyn std::error::Error>> {
    // The guard must not reject a schema that lands exactly on the cap — only one deeper — and
    // the resulting schema must still work, not merely have compiled.
    let (schema, document) = nested_element_xml(64);
    let schema = RelaxNg::from_xml(&schema)?;
    assert!(schema.validate(&document).is_ok());

    let (one_deeper, _) = nested_element_xml(65);
    let error =
        RelaxNg::from_xml(&one_deeper).expect_err("one level past the cap must still be rejected");
    assert_eq!(error.kind(), SchemaErrorKind::Syntax);
    Ok(())
}

#[test]
fn qnames_resolve_against_the_instance_context() -> Result<(), SchemaError> {
    let schema = RelaxNg::from_compact(
        r#"namespace ex = "urn:example"
           element r { xsd:QName "ex:target" }"#,
    )?;
    assert!(
        schema
            .validate(r#"<r xmlns:p="urn:example">p:target</r>"#)
            .is_ok()
    );
    assert!(
        schema
            .validate(r#"<r xmlns:p="urn:other">p:target</r>"#)
            .is_err()
    );
    Ok(())
}

#[test]
fn validating_a_deeply_recursive_instance_against_a_self_referential_schema_does_not_overflow_the_stack()
-> Result<(), SchemaError> {
    // `walk` (validate.rs) recurses once per level of the *instance* document being checked —
    // independent of how deep the compiled pattern itself is, and unguarded by any depth cap of
    // this crate's own. A schema that legitimately describes a recursive tree structure
    // (`tree = element node { tree* }`, an entirely ordinary RELAX NG idiom for validating
    // things like nested folders or outlines) can be checked against an instance nested
    // arbitrarily deep, so this confirms that walking stays safe up to the depth the shared XML
    // reader itself allows, rather than exhausting the native call stack first.
    let schema = RelaxNg::from_compact("start = tree\ntree = element node { tree* }")?;
    let depth = 1_000;
    let mut document = "<node>".repeat(depth);
    document.push_str(&"</node>".repeat(depth));
    assert!(schema.validate(&document).is_ok());
    Ok(())
}
