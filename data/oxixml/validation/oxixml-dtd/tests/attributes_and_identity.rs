//! Attribute defaulting and typing, and the `ID`/`IDREF` validity constraints
//! of XML 1.0 §3.3.

use oxixml_dtd::{
    AttributeType, DefaultDecl, DtdParser, ValidationOptions, Validator, ValidityError,
    ValidityErrorKind, validate_str,
};

/// The kinds of every validity error a document produces.
fn error_kinds(xml: &str) -> Vec<ValidityErrorKind> {
    validate_str(xml)
        .expect("well formed")
        .errors()
        .iter()
        .map(|error| error.kind().clone())
        .collect()
}

#[test]
fn every_attribute_type_parses() {
    let dtd = DtdParser::new()
        .parse_internal_subset(
            r#"<!NOTATION gif SYSTEM "image/gif">
               <!NOTATION png SYSTEM "image/png">
               <!ELEMENT a ANY>
               <!ATTLIST a
                   c CDATA     #IMPLIED
                   i ID        #IMPLIED
                   r IDREF     #IMPLIED
                   rs IDREFS   #IMPLIED
                   e ENTITY    #IMPLIED
                   es ENTITIES #IMPLIED
                   n NMTOKEN   #IMPLIED
                   ns NMTOKENS #IMPLIED
                   t NOTATION (gif | png) #IMPLIED
                   u (one | 2 | -three) "one">"#,
        )
        .expect("valid DTD");
    let list = dtd.attributes("a").expect("declared");
    assert_eq!(list.len(), 10);
    assert_eq!(
        list.get("c").map(|d| d.att_type().clone()),
        Some(AttributeType::Cdata)
    );
    assert_eq!(
        list.get("rs").map(|d| d.att_type().clone()),
        Some(AttributeType::IdRefs)
    );
    assert_eq!(
        list.get("es").map(|d| d.att_type().clone()),
        Some(AttributeType::Entities)
    );
    assert_eq!(
        list.get("ns").map(|d| d.att_type().clone()),
        Some(AttributeType::NmTokens)
    );
    assert_eq!(
        list.get("t").map(|d| d.att_type().clone()),
        Some(AttributeType::Notation(vec![
            "gif".to_owned(),
            "png".to_owned()
        ]))
    );
    assert_eq!(
        list.get("u").map(|d| d.att_type().clone()),
        Some(AttributeType::Enumeration(vec![
            "one".to_owned(),
            "2".to_owned(),
            "-three".to_owned()
        ]))
    );
    assert_eq!(
        list.get("u").map(|d| d.default().clone()),
        Some(DefaultDecl::Default("one".to_owned()))
    );
}

#[test]
fn every_default_declaration_parses() {
    let dtd = DtdParser::new()
        .parse_internal_subset(
            r#"<!ATTLIST a
                   q CDATA #REQUIRED
                   w CDATA #IMPLIED
                   e CDATA #FIXED "k"
                   r CDATA "plain">"#,
        )
        .expect("valid DTD");
    let list = dtd.attributes("a").expect("declared");
    assert_eq!(
        list.get("q").map(|d| d.default().clone()),
        Some(DefaultDecl::Required)
    );
    assert_eq!(
        list.get("w").map(|d| d.default().clone()),
        Some(DefaultDecl::Implied)
    );
    assert_eq!(
        list.get("e").map(|d| d.default().clone()),
        Some(DefaultDecl::Fixed("k".to_owned()))
    );
    assert_eq!(
        list.get("r").map(|d| d.default().clone()),
        Some(DefaultDecl::Default("plain".to_owned()))
    );
}

#[test]
fn defaults_are_supplied_for_absent_attributes() {
    let dtd = DtdParser::new()
        .parse_internal_subset(r#"<!ELEMENT a EMPTY><!ATTLIST a x CDATA "d" y CDATA #IMPLIED>"#)
        .expect("valid DTD");
    let mut validator = Validator::new(&dtd, ValidationOptions::default());
    let added = validator.start_element("a", &[("y", "given")]);
    validator.end_element("a");
    assert_eq!(added, vec![("x".to_owned(), "d".to_owned())]);
    assert!(validator.finish().is_empty());
}

#[test]
fn tokenized_values_are_collapsed_but_cdata_values_are_not() {
    let dtd = DtdParser::new()
        .parse_internal_subset("<!ELEMENT a EMPTY><!ATTLIST a t NMTOKEN #IMPLIED c CDATA #IMPLIED>")
        .expect("valid DTD");
    let mut validator = Validator::new(&dtd, ValidationOptions::default());
    // `  x  ` collapses to `x`, which is a legal Nmtoken; the same value as
    // CDATA keeps its spaces, and CDATA accepts anything anyway.
    validator.start_element("a", &[("t", "  x  "), ("c", "  x  ")]);
    validator.end_element("a");
    assert!(validator.finish().is_empty());

    let mut validator = Validator::new(&dtd, ValidationOptions::default());
    validator.start_element("a", &[("t", "x y")]);
    validator.end_element("a");
    assert!(matches!(
        validator.finish().first().map(ValidityError::kind),
        Some(ValidityErrorKind::AttributeValueInvalid { .. })
    ));
}

#[test]
fn attribute_values_are_normalized_before_being_typed() {
    // A literal newline in an attribute value becomes a space (§3.3.3), so the
    // NMTOKENS value below is two tokens and both are legal.
    let report = validate_str(
        "<!DOCTYPE a [<!ELEMENT a EMPTY><!ATTLIST a t NMTOKENS #IMPLIED>]><a t=\"x\ny\"/>",
    )
    .expect("well formed");
    assert!(report.is_valid(), "{:?}", report.errors());
}

#[test]
fn required_attributes_must_be_present() {
    let kinds =
        error_kinds(r#"<!DOCTYPE a [<!ELEMENT a EMPTY><!ATTLIST a x CDATA #REQUIRED>]><a/>"#);
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::RequiredAttributeMissing { .. }))
    );
}

#[test]
fn fixed_attributes_must_match() {
    assert!(
        validate_str(
            r#"<!DOCTYPE a [<!ELEMENT a EMPTY><!ATTLIST a x CDATA #FIXED "k">]><a x="k"/>"#
        )
        .expect("well formed")
        .is_valid()
    );
    let kinds = error_kinds(
        r#"<!DOCTYPE a [<!ELEMENT a EMPTY><!ATTLIST a x CDATA #FIXED "k">]><a x="other"/>"#,
    );
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::FixedAttributeMismatch { .. }))
    );
}

#[test]
fn undeclared_attributes_are_reported() {
    let kinds = error_kinds(r#"<!DOCTYPE a [<!ELEMENT a EMPTY>]><a x="1"/>"#);
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::AttributeNotDeclared { .. }))
    );
}

#[test]
fn enumeration_values_must_be_listed() {
    let dtd = r#"<!DOCTYPE a [<!ELEMENT a EMPTY><!ATTLIST a x (one|two) #IMPLIED>]>"#;
    assert!(
        validate_str(&format!("{dtd}<a x=\"two\"/>"))
            .expect("wf")
            .is_valid()
    );
    let kinds = error_kinds(&format!("{dtd}<a x=\"three\"/>"));
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::AttributeValueInvalid { .. }))
    );
}

#[test]
fn notation_values_must_name_declared_notations() {
    let prolog = r#"<!DOCTYPE a [
        <!NOTATION gif SYSTEM "image/gif">
        <!ELEMENT a ANY>
        <!ATTLIST a x NOTATION (gif) #IMPLIED>]>"#;
    assert!(
        validate_str(&format!("{prolog}<a x=\"gif\"/>"))
            .expect("wf")
            .is_valid()
    );
    let kinds = error_kinds(&format!("{prolog}<a x=\"png\"/>"));
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::AttributeValueInvalid { .. }))
    );
}

#[test]
fn entity_attributes_must_name_unparsed_entities() {
    let prolog = r#"<!DOCTYPE a [
        <!NOTATION gif SYSTEM "image/gif">
        <!ENTITY logo SYSTEM "logo.gif" NDATA gif>
        <!ENTITY text "just text">
        <!ELEMENT a ANY>
        <!ATTLIST a x ENTITY #IMPLIED xs ENTITIES #IMPLIED>]>"#;
    assert!(
        validate_str(&format!("{prolog}<a x=\"logo\" xs=\"logo logo\"/>"))
            .expect("wf")
            .is_valid()
    );

    let kinds = error_kinds(&format!("{prolog}<a x=\"text\"/>"));
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::AttributeValueInvalid { .. }))
    );
    let kinds = error_kinds(&format!("{prolog}<a x=\"nothing\"/>"));
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::AttributeValueInvalid { .. }))
    );
}

#[test]
fn ids_are_unique_across_the_whole_document() {
    let prolog = r#"<!DOCTYPE a [
        <!ELEMENT a (b*)>
        <!ELEMENT b EMPTY>
        <!ATTLIST b id ID #REQUIRED>]>"#;
    assert!(
        validate_str(&format!("{prolog}<a><b id=\"x\"/><b id=\"y\"/></a>"))
            .expect("wf")
            .is_valid()
    );
    let kinds = error_kinds(&format!("{prolog}<a><b id=\"x\"/><b id=\"x\"/></a>"));
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::DuplicateId { .. }))
    );
}

#[test]
fn id_values_must_be_names() {
    let kinds = error_kinds(
        r#"<!DOCTYPE a [<!ELEMENT a EMPTY><!ATTLIST a id ID #REQUIRED>]><a id="1st"/>"#,
    );
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::AttributeValueInvalid { .. }))
    );
}

#[test]
fn idrefs_must_resolve_even_when_the_target_comes_later() {
    let prolog = r#"<!DOCTYPE a [
        <!ELEMENT a (b*)>
        <!ELEMENT b EMPTY>
        <!ATTLIST b id ID #IMPLIED ref IDREF #IMPLIED refs IDREFS #IMPLIED>]>"#;
    // Forward reference: `ref` points at an `id` declared afterwards.
    assert!(
        validate_str(&format!(
            "{prolog}<a><b ref=\"later\"/><b id=\"later\"/></a>"
        ))
        .expect("wf")
        .is_valid()
    );
    assert!(
        validate_str(&format!(
            "{prolog}<a><b id=\"one\"/><b id=\"two\"/><b refs=\"one two\"/></a>"
        ))
        .expect("wf")
        .is_valid()
    );

    let kinds = error_kinds(&format!("{prolog}<a><b ref=\"nowhere\"/></a>"));
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::UnresolvedIdref { .. }))
    );

    let kinds = error_kinds(&format!(
        "{prolog}<a><b id=\"one\"/><b refs=\"one two\"/></a>"
    ));
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::UnresolvedIdref { .. }))
    );
}

#[test]
fn defaulted_id_values_take_part_in_the_id_bookkeeping() {
    // The default is an IDREF, so it must still resolve.
    let kinds = error_kinds(
        r#"<!DOCTYPE a [
             <!ELEMENT a (b)>
             <!ELEMENT b EMPTY>
             <!ATTLIST b ref IDREF "ghost">]>
           <a><b/></a>"#,
    );
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::UnresolvedIdref { .. }))
    );
}

#[test]
fn one_id_attribute_per_element_type() {
    let kinds = error_kinds(
        r#"<!DOCTYPE a [<!ELEMENT a EMPTY><!ATTLIST a p ID #IMPLIED q ID #IMPLIED>]><a/>"#,
    );
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::MultipleIdAttributes { .. }))
    );
}

#[test]
fn id_attributes_may_not_have_defaults() {
    let kinds = error_kinds(r#"<!DOCTYPE a [<!ELEMENT a EMPTY><!ATTLIST a p ID "x">]><a/>"#);
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::IdAttributeHasDefault { .. }))
    );
}

#[test]
fn the_first_declaration_of_an_attribute_wins() {
    let dtd = DtdParser::new()
        .parse_internal_subset(
            r#"<!ATTLIST a x CDATA "first"><!ATTLIST a x CDATA "second" y CDATA "new">"#,
        )
        .expect("valid DTD");
    let list = dtd.attributes("a").expect("declared");
    assert_eq!(
        list.get("x").and_then(|d| d.default().value()),
        Some("first"),
        "a later declaration of the same attribute is discarded"
    );
    assert_eq!(list.get("y").and_then(|d| d.default().value()), Some("new"));
}

#[test]
fn entities_in_attribute_defaults_are_expanded() {
    let dtd = DtdParser::new()
        .parse_internal_subset(r#"<!ENTITY who "world"><!ATTLIST a greeting CDATA "hello &who;">"#)
        .expect("valid DTD");
    assert_eq!(
        dtd.attributes("a")
            .and_then(|list| list.get("greeting"))
            .and_then(|d| d.default().value()),
        Some("hello world")
    );
}

#[test]
fn built_in_entities_need_no_declaration_in_a_default() {
    let dtd = DtdParser::new()
        .parse_internal_subset(r#"<!ATTLIST a x CDATA "&quot;&amp;&lt;&gt;&apos;">"#)
        .expect("valid DTD");
    assert_eq!(
        dtd.attributes("a")
            .and_then(|list| list.get("x"))
            .and_then(|d| d.default().value()),
        Some("\"&<>'")
    );
}
