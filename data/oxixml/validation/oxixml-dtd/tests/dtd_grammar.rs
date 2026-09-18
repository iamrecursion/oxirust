//! The DTD grammar itself, and the validity constraints that are properties of
//! the DTD rather than of any one document.

use oxixml_dtd::{
    ContentSpec, DtdError, DtdErrorKind, DtdParser, EntityDecl, EntityDefinition, EntityKind,
    MemoryResolver, ValidityErrorKind,
};

/// The declaration errors a DTD text produces.
fn declaration_errors(external_subset: &str) -> Vec<ValidityErrorKind> {
    DtdParser::new()
        .parse_external_subset(external_subset)
        .expect("the DTD parses")
        .declaration_errors()
        .iter()
        .map(|error| error.kind().clone())
        .collect()
}

#[test]
fn a_document_type_declaration_yields_name_and_identifiers() -> Result<(), DtdError> {
    let resolver = MemoryResolver::new().with("x.dtd", "<!ELEMENT root EMPTY>");
    let dtd = DtdParser::new()
        .with_resolver(Box::new(resolver))
        .parse_doctype(r#"root PUBLIC "-//X//DTD Root//EN" "x.dtd" [ <!ENTITY e "v"> ]"#)?;
    assert_eq!(dtd.name(), Some("root"));
    assert_eq!(
        dtd.external_id()
            .and_then(oxixml_dtd::ExternalId::public_id),
        Some("-//X//DTD Root//EN")
    );
    assert_eq!(
        dtd.external_id()
            .and_then(oxixml_dtd::ExternalId::system_id),
        Some("x.dtd")
    );
    assert_eq!(
        dtd.general_entity("e").and_then(EntityDecl::internal_text),
        Some("v")
    );
    assert!(dtd.element("root").is_some());
    Ok(())
}

#[test]
fn public_identifiers_have_their_white_space_collapsed() -> Result<(), DtdError> {
    let dtd = DtdParser::new()
        .parse_internal_subset("<!NOTATION n PUBLIC \"-//X//NOTATION\n   Y//EN\">")?;
    assert_eq!(
        dtd.notation("n")
            .and_then(oxixml_dtd::NotationDecl::public_id),
        Some("-//X//NOTATION Y//EN")
    );
    Ok(())
}

#[test]
fn a_public_identifier_may_not_contain_arbitrary_characters() {
    let error = DtdParser::new()
        .parse_internal_subset("<!NOTATION n PUBLIC \"a<b\">")
        .expect_err("`<` is not a PubidChar");
    assert!(
        matches!(error.kind(), DtdErrorKind::InvalidPublicId(_)),
        "{error}"
    );
}

#[test]
fn every_entity_shape_parses() -> Result<(), DtdError> {
    let dtd = DtdParser::new().parse_internal_subset(
        r#"<!NOTATION gif SYSTEM "image/gif">
           <!ENTITY internal "text">
           <!ENTITY external SYSTEM "e.ent">
           <!ENTITY public PUBLIC "-//X//TEXT//EN" "p.ent">
           <!ENTITY unparsed SYSTEM "logo.gif" NDATA gif>
           <!ENTITY % pe-internal "<!ELEMENT z EMPTY>">
           <!ENTITY % pe-external SYSTEM "pe.ent">"#,
    )?;
    assert_eq!(
        dtd.general_entity("internal")
            .and_then(EntityDecl::internal_text),
        Some("text")
    );
    assert!(
        dtd.general_entity("external")
            .is_some_and(|e| e.definition().is_external())
    );
    assert_eq!(
        dtd.general_entity("public")
            .map(|e| e.definition().is_external()),
        Some(true)
    );
    assert_eq!(
        dtd.general_entity("unparsed")
            .and_then(EntityDecl::notation),
        Some("gif")
    );
    assert!(
        dtd.general_entity("unparsed")
            .is_some_and(EntityDecl::is_unparsed)
    );
    assert_eq!(dtd.parameter_entities().count(), 2);
    assert_eq!(dtd.general_entities().count(), 4);
    assert_eq!(
        dtd.parameter_entity("pe-internal").map(|e| e.kind()),
        Some(EntityKind::Parameter)
    );
    Ok(())
}

#[test]
fn character_references_in_entity_values_are_resolved_at_declaration_time() -> Result<(), DtdError>
{
    let dtd = DtdParser::new().parse_internal_subset(r#"<!ENTITY e "&#65;&#x42;&#x1F6A2;">"#)?;
    assert_eq!(
        dtd.general_entity("e").and_then(EntityDecl::internal_text),
        Some("AB\u{1F6A2}")
    );
    Ok(())
}

#[test]
fn a_bad_character_reference_in_an_entity_value_is_rejected() {
    for body in ["&#0;", "&#xD800;", "&#xZZ;", "&#;"] {
        let error = DtdParser::new()
            .parse_internal_subset(&format!(r#"<!ENTITY e "{body}">"#))
            .expect_err("bad character reference");
        assert!(
            matches!(
                error.kind(),
                DtdErrorKind::InvalidCharRef(_) | DtdErrorKind::Unexpected { .. }
            ),
            "{body}: {error}"
        );
    }
}

#[test]
fn the_first_declaration_of_an_entity_binds() -> Result<(), DtdError> {
    let dtd =
        DtdParser::new().parse_internal_subset(r#"<!ENTITY e "first"><!ENTITY e "second">"#)?;
    assert_eq!(
        dtd.general_entity("e").and_then(EntityDecl::internal_text),
        Some("first")
    );
    assert!(
        dtd.declaration_errors().is_empty(),
        "a repeated declaration is not an error"
    );
    Ok(())
}

#[test]
fn general_and_parameter_entities_live_in_separate_namespaces() -> Result<(), DtdError> {
    let dtd = DtdParser::new()
        .parse_internal_subset(r#"<!ENTITY e "general"><!ENTITY % e "parameter">"#)?;
    assert_eq!(
        dtd.general_entity("e").and_then(EntityDecl::internal_text),
        Some("general")
    );
    assert_eq!(
        dtd.parameter_entity("e")
            .and_then(EntityDecl::internal_text),
        Some("parameter")
    );
    Ok(())
}

#[test]
fn declaring_the_same_element_type_twice_is_a_validity_error() {
    let kinds = declaration_errors("<!ELEMENT a EMPTY><!ELEMENT a ANY>");
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::DuplicateElementDeclaration { .. }))
    );
}

#[test]
fn a_mixed_content_model_may_not_repeat_an_element_type() {
    let kinds = declaration_errors("<!ELEMENT a (#PCDATA|b|c|b)*>");
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::DuplicateTypeInMixedContent { .. }))
    );
}

#[test]
fn an_enumeration_may_not_repeat_a_token() {
    let kinds = declaration_errors("<!ELEMENT a EMPTY><!ATTLIST a x (p|q|p) #IMPLIED>");
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::DuplicateEnumerationToken { .. }))
    );
}

#[test]
fn an_empty_element_may_not_have_a_notation_attribute() {
    let kinds = declaration_errors(
        r#"<!NOTATION n SYSTEM "n"><!ELEMENT a EMPTY><!ATTLIST a x NOTATION (n) #IMPLIED>"#,
    );
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::NotationOnEmptyElement { .. }))
    );
}

#[test]
fn only_one_notation_attribute_per_element_type() {
    let kinds = declaration_errors(
        r#"<!NOTATION n SYSTEM "n">
           <!ELEMENT a ANY>
           <!ATTLIST a x NOTATION (n) #IMPLIED y NOTATION (n) #IMPLIED>"#,
    );
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::MultipleNotationAttributes { .. }))
    );
}

#[test]
fn notation_attribute_tokens_must_be_declared_notations() {
    let kinds = declaration_errors("<!ELEMENT a ANY><!ATTLIST a x NOTATION (ghost) #IMPLIED>");
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::NotationNotDeclared { .. }))
    );
}

#[test]
fn a_declaration_must_start_and_finish_in_the_same_entity() {
    // VC: Proper Declaration/PE Nesting (§2.8) — a validity error, not a fatal
    // one, so the declaration is still recorded.
    let kinds = declaration_errors(r#"<!ENTITY % tail "EMPTY>"><!ELEMENT a %tail;"#);
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::ImproperDeclarationNesting { .. })),
        "{kinds:?}"
    );
}

#[test]
fn a_content_model_group_must_start_and_finish_in_the_same_entity() {
    // VC: Proper Group/PE Nesting (§3.2.1).
    let kinds = declaration_errors(r#"<!ENTITY % open "(#PCDATA"><!ELEMENT a %open;)>"#);
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::ImproperGroupNesting { .. })),
        "{kinds:?}"
    );

    // …and a model whose parentheses are wholly inside one entity is fine.
    let kinds = declaration_errors(r#"<!ENTITY % whole "(#PCDATA)"><!ELEMENT a %whole;>"#);
    assert!(
        !kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::ImproperGroupNesting { .. })),
        "{kinds:?}"
    );
}

#[test]
fn a_conditional_section_must_be_delimited_within_one_entity() {
    // VC: Proper Conditional Section/PE Nesting (§3.4).
    let kinds = declaration_errors(
        r#"<!ENTITY % k "INCLUDE["><!ELEMENT a EMPTY><![ %k; <!ATTLIST a v CDATA "x"> ]]>"#,
    );
    assert!(
        kinds
            .iter()
            .any(|kind| matches!(kind, ValidityErrorKind::ImproperConditionalSectionNesting)),
        "{kinds:?}"
    );
}

#[test]
fn parameter_entities_may_supply_whole_declarations() -> Result<(), DtdError> {
    let dtd = DtdParser::new().parse_internal_subset(
        r#"<!ENTITY % decls "<!ELEMENT a EMPTY><!ELEMENT b EMPTY>">
           %decls;"#,
    )?;
    assert_eq!(dtd.elements().count(), 2);
    Ok(())
}

#[test]
fn parameter_entities_compose_inside_declarations_in_the_external_subset() -> Result<(), DtdError> {
    let dtd = DtdParser::new().parse_external_subset(
        r#"<!ENTITY % common "id ID #IMPLIED">
           <!ENTITY % extra "class NMTOKEN #IMPLIED">
           <!ELEMENT a EMPTY>
           <!ATTLIST a %common; %extra;>"#,
    )?;
    let list = dtd.attributes("a").expect("declared");
    assert_eq!(list.len(), 2);
    assert!(list.get("id").is_some());
    assert!(list.get("class").is_some());
    Ok(())
}

#[test]
fn parameter_entities_nest() -> Result<(), DtdError> {
    let dtd = DtdParser::new().parse_external_subset(
        r#"<!ENTITY % inner "EMPTY">
           <!ENTITY % outer "<!ELEMENT a %inner;>">
           %outer;"#,
    )?;
    assert_eq!(
        dtd.element("a").map(|d| d.content().clone()),
        Some(ContentSpec::Empty)
    );
    Ok(())
}

#[test]
fn an_undeclared_parameter_entity_is_fatal() {
    let error = DtdParser::new()
        .parse_external_subset("%missing;")
        .expect_err("undeclared parameter entity");
    assert!(
        matches!(error.kind(), DtdErrorKind::UndeclaredParameterEntity(_)),
        "{error}"
    );
}

#[test]
fn errors_name_the_entity_they_occurred_in() {
    let error = DtdParser::new()
        .parse_external_subset(r#"<!ENTITY % broken "<!ELEMENT (bad)>">%broken;"#)
        .expect_err("a broken declaration inside an entity");
    assert_eq!(error.entity_stack(), ["broken"]);
}

#[test]
fn comments_and_processing_instructions_are_ignored_everywhere() -> Result<(), DtdError> {
    let dtd = DtdParser::new().parse_external_subset(
        r#"<?target instruction?>
           <!-- a comment -->
           <!ELEMENT a EMPTY>
           <!-- another -->
           <?another?>"#,
    )?;
    assert_eq!(dtd.elements().count(), 1);
    Ok(())
}

#[test]
fn an_unterminated_construct_reports_end_of_input() {
    for text in [
        "<!-- unterminated",
        "<?pi unterminated",
        "<!ELEMENT a EMPTY",
        "<!ENTITY e \"x",
    ] {
        let error = DtdParser::new()
            .parse_external_subset(text)
            .expect_err("unterminated construct");
        assert!(
            matches!(
                error.kind(),
                DtdErrorKind::UnexpectedEof { .. } | DtdErrorKind::Unexpected { .. }
            ),
            "{text}: {error}"
        );
    }
}

#[test]
fn a_dtd_can_be_built_by_hand_and_used_for_validation() {
    use oxixml_dtd::{Dtd, ElementDecl, ValidationOptions, Validator};

    let mut dtd = Dtd::new();
    dtd.set_name("a");
    dtd.add_element(ElementDecl::new("a", ContentSpec::Any, false));
    dtd.add_entity(EntityDecl::new(
        "e",
        EntityKind::General,
        EntityDefinition::Internal("v".to_owned()),
        false,
    ));

    let mut validator = Validator::new(&dtd, ValidationOptions::default());
    validator.start_element("a", &[]);
    validator.characters("anything");
    validator.end_element("a");
    assert!(validator.finish().is_empty());
}
