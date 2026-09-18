//! Entity expansion, and the guards that keep it from becoming a denial of
//! service.
//!
//! The two attacks these tests pin down are the ones every XML processor has to
//! survive: exponential ("billion laughs") expansion, and quadratic expansion
//! of one large entity referenced many times. Both must produce
//! [`DtdErrorKind::EntityExpansionLimit`] promptly — never a hang, never an
//! out-of-memory abort.

use std::time::{Duration, Instant};

use oxixml_dtd::{
    DocumentValidator, DtdError, DtdErrorKind, DtdParser, Error, ExpansionLimit, ExpansionLimits,
    MemoryResolver,
};

/// The declarations of the classic "billion laughs" bomb, `depth` levels deep.
///
/// Each level references the one below it ten times, so `lol{depth}` expands to
/// `10^depth` copies of `"lol"` — from a few hundred bytes of declarations.
fn billion_laughs(depth: usize) -> String {
    let mut text = String::from(r#"<!ENTITY lol "lol">"#);
    for level in 1..=depth {
        let inner = if level == 1 {
            "lol".to_owned()
        } else {
            format!("lol{}", level - 1)
        };
        let references = format!("&{inner};").repeat(10);
        text.push_str(&format!(r#"<!ENTITY lol{level} "{references}">"#));
    }
    text
}

/// Runs `body`, asserting that it finishes well inside `limit`.
fn within<T>(limit: Duration, body: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let value = body();
    let elapsed = started.elapsed();
    assert!(
        elapsed < limit,
        "took {elapsed:?}, which is longer than {limit:?}"
    );
    value
}

#[test]
fn declaring_a_billion_laughs_bomb_is_cheap_but_expanding_it_is_refused() {
    let dtd = DtdParser::new()
        .parse_internal_subset(&billion_laughs(9))
        .expect("declaring the entities costs nothing");
    assert!(dtd.general_entity("lol9").is_some());

    let mut budget = ExpansionLimits::default().budget();
    let error = within(Duration::from_secs(5), || {
        dtd.expand_general_entity("lol9", &mut budget)
            .expect_err("the bomb must be refused")
    });
    assert!(error.is_expansion_limit(), "{error}");
    assert!(matches!(
        error.kind(),
        DtdErrorKind::EntityExpansionLimit(ExpansionLimit::Count(_) | ExpansionLimit::TotalSize(_))
    ));
}

#[test]
fn a_billion_laughs_bomb_in_a_document_is_refused_too() {
    let xml = format!(
        "<!DOCTYPE a [<!ELEMENT a (#PCDATA)>{}]><a>&lol9;</a>",
        billion_laughs(9)
    );
    let error = within(Duration::from_secs(5), || {
        DocumentValidator::new()
            .validate_str(&xml)
            .expect_err("the bomb must be refused")
    });
    let Error::Dtd(error) = error else {
        panic!("expected a DTD error, got {error:?}");
    };
    assert!(error.is_expansion_limit(), "{error}");
}

#[test]
fn a_billion_laughs_bomb_in_an_attribute_value_is_refused() {
    let xml = format!(
        r#"<!DOCTYPE a [<!ELEMENT a EMPTY><!ATTLIST a x CDATA #IMPLIED>{}]><a x="&lol9;"/>"#,
        billion_laughs(9)
    );
    let error = within(Duration::from_secs(5), || {
        DocumentValidator::new()
            .validate_str(&xml)
            .expect_err("the bomb must be refused")
    });
    let Error::Dtd(error) = error else {
        panic!("expected a DTD error, got {error:?}");
    };
    assert!(error.is_expansion_limit(), "{error}");
}

#[test]
fn a_bomb_hidden_in_an_attribute_default_is_refused_while_reading_the_dtd() {
    let subset = format!(r#"{}<!ATTLIST a x CDATA "&lol9;">"#, billion_laughs(9));
    let error = within(Duration::from_secs(5), || {
        DtdParser::new()
            .parse_internal_subset(&subset)
            .expect_err("the bomb must be refused")
    });
    assert!(error.is_expansion_limit(), "{error}");
}

#[test]
fn quadratic_blow_up_is_refused_by_the_total_size_budget() {
    // One 100 KiB entity referenced 500 times is 50 MiB, which is well past the
    // 10 MiB default budget but nowhere near the expansion *count* limit.
    let payload = "x".repeat(100 * 1024);
    let subset = format!(
        r#"<!ENTITY big "{payload}"><!ENTITY blow "{}">"#,
        "&big;".repeat(500)
    );
    let dtd = DtdParser::new()
        .parse_internal_subset(&subset)
        .expect("valid DTD");

    let mut budget = ExpansionLimits::default().budget();
    let error = within(Duration::from_secs(5), || {
        dtd.expand_general_entity("blow", &mut budget)
            .expect_err("over the size budget")
    });
    assert!(matches!(
        error.kind(),
        DtdErrorKind::EntityExpansionLimit(ExpansionLimit::TotalSize(_))
    ));
}

#[test]
fn a_parameter_entity_bomb_is_refused_while_reading_the_dtd() {
    // The same trick with parameter entities detonates during parsing rather
    // than during document processing.
    let mut subset = String::from(r#"<!ENTITY % lol "lol">"#);
    for level in 1..=9 {
        let inner = if level == 1 {
            "lol".to_owned()
        } else {
            format!("lol{}", level - 1)
        };
        let references = format!("%{inner};").repeat(10);
        subset.push_str(&format!(r#"<!ENTITY % lol{level} "{references}">"#));
    }
    subset.push_str("%lol9;");

    let error = within(Duration::from_secs(5), || {
        DtdParser::new()
            .parse_external_subset(&subset)
            .expect_err("the bomb must be refused")
    });
    assert!(error.is_expansion_limit(), "{error}");
}

#[test]
fn nesting_depth_is_capped_independently_of_size() {
    let mut subset = String::from(r#"<!ENTITY e0 "leaf">"#);
    for level in 1..=100 {
        subset.push_str(&format!(r#"<!ENTITY e{level} "&e{};">"#, level - 1));
    }
    let dtd = DtdParser::new()
        .parse_internal_subset(&subset)
        .expect("valid DTD");

    let mut budget = ExpansionLimits {
        max_depth: 8,
        max_expansions: 10_000,
        max_expanded_size: 1 << 20,
    }
    .budget();
    let error = dtd
        .expand_general_entity("e100", &mut budget)
        .expect_err("too deep");
    assert!(matches!(
        error.kind(),
        DtdErrorKind::EntityExpansionLimit(ExpansionLimit::Depth(8))
    ));

    // With room to breathe, the very same chain expands fine.
    let mut budget = ExpansionLimits::RELAXED.budget();
    assert_eq!(
        dtd.expand_general_entity("e100", &mut budget).as_deref(),
        Ok("leaf")
    );
}

#[test]
fn self_referential_entities_are_reported_as_recursion_not_as_a_stack_overflow() {
    let dtd = DtdParser::new()
        .parse_internal_subset(r#"<!ENTITY a "&b;"><!ENTITY b "&a;">"#)
        .expect("valid DTD");
    let mut budget = ExpansionLimits::default().budget();
    let error = dtd
        .expand_general_entity("a", &mut budget)
        .expect_err("mutual recursion");
    assert!(
        matches!(error.kind(), DtdErrorKind::RecursiveEntity { .. }),
        "{error}"
    );
}

#[test]
fn a_parameter_entity_cannot_reference_itself_in_its_own_literal() {
    // XML 1.0 §4.4.5 includes a parameter entity *in literal*, so `%p;` inside
    // `p`'s own value is expanded while `p` is still being declared — at which
    // point `p` does not exist yet.
    let error = DtdParser::new()
        .parse_external_subset(r#"<!ENTITY % p "%p;">%p;"#)
        .expect_err("`p` is not declared yet inside its own literal");
    assert!(
        matches!(error.kind(), DtdErrorKind::UndeclaredParameterEntity(_)),
        "{error}"
    );
}

#[test]
fn a_recursive_external_parameter_entity_is_reported() {
    // An *external* parameter entity is fetched at reference time, so its text
    // can genuinely refer back to itself.
    let resolver = MemoryResolver::new().with("loop.ent", "%r;");
    let error = DtdParser::new()
        .with_resolver(Box::new(resolver))
        .parse_external_subset(r#"<!ENTITY % r SYSTEM "loop.ent">%r;"#)
        .expect_err("recursion");
    assert!(
        matches!(
            error.kind(),
            DtdErrorKind::RecursiveEntity { .. } | DtdErrorKind::EntityExpansionLimit(_)
        ),
        "{error}"
    );
}

#[test]
fn a_recursive_entity_reference_in_content_is_reported() {
    let xml = r#"<!DOCTYPE a [<!ELEMENT a (#PCDATA)><!ENTITY loop "&loop;">]><a>&loop;</a>"#;
    let error = DocumentValidator::new()
        .validate_str(xml)
        .expect_err("recursion");
    let Error::Dtd(error) = error else {
        panic!("expected a DTD error, got {error:?}");
    };
    assert!(
        matches!(
            error.kind(),
            DtdErrorKind::RecursiveEntity { .. } | DtdErrorKind::EntityExpansionLimit(_)
        ),
        "{error}"
    );
}

#[test]
fn strict_limits_stop_expansions_that_default_limits_allow() {
    let dtd = DtdParser::new()
        .parse_internal_subset(&billion_laughs(4))
        .expect("valid DTD");

    let mut relaxed = ExpansionLimits::RELAXED.budget();
    assert!(dtd.expand_general_entity("lol4", &mut relaxed).is_ok());

    let mut strict = ExpansionLimits::STRICT.budget();
    let error = dtd
        .expand_general_entity("lol4", &mut strict)
        .expect_err("over the strict budget");
    assert!(error.is_expansion_limit(), "{error}");
}

#[test]
fn the_budget_accumulates_across_a_whole_document() {
    // Neither reference alone exceeds the budget; together they do.
    let payload = "y".repeat(4096);
    let xml = format!(
        r#"<!DOCTYPE a [<!ELEMENT a (#PCDATA)><!ENTITY big "{payload}">]><a>{}</a>"#,
        "&big;".repeat(20)
    );
    let limits = ExpansionLimits {
        max_depth: 8,
        max_expansions: 64,
        max_expanded_size: 40_000,
    };
    let error = DocumentValidator::new()
        .with_limits(limits)
        .validate_str(&xml)
        .expect_err("the accumulated size is over budget");
    let Error::Dtd(error) = error else {
        panic!("expected a DTD error, got {error:?}");
    };
    assert!(matches!(
        error.kind(),
        DtdErrorKind::EntityExpansionLimit(ExpansionLimit::TotalSize(40_000))
    ));
}

#[test]
fn expansion_in_an_attribute_default_and_expansion_in_content_share_one_budget() {
    // The DTD-parse phase (normalizing the `<!ATTLIST>` default) and the document-content
    // phase (expanding the reference in `<a>`'s content) each charge less than the budget on
    // their own; only their *sum* is over it. Catching that requires the two phases to draw
    // from one genuinely shared running total instead of each silently getting its own full
    // allowance — which is exactly the shape of the classic split-the-attack-across-two-places
    // dodge `Budget` exists to close off (see the crate-level module docs).
    let default_payload = "x".repeat(30_000);
    let content_payload = "y".repeat(30_000);
    let xml = format!(
        r#"<!DOCTYPE a [
             <!ELEMENT a (#PCDATA)>
             <!ENTITY default_bomb "{default_payload}">
             <!ATTLIST a attr CDATA "&default_bomb;">
             <!ENTITY content_bomb "{content_payload}">
           ]><a>&content_bomb;</a>"#
    );
    let limits = ExpansionLimits {
        max_depth: 8,
        max_expansions: 64,
        max_expanded_size: 40_000,
    };
    let error = DocumentValidator::new()
        .with_limits(limits)
        .validate_str(&xml)
        .expect_err("the combined DTD-default and document-content expansion is over budget");
    let Error::Dtd(error) = error else {
        panic!("expected a DTD error, got {error:?}");
    };
    assert!(
        matches!(
            error.kind(),
            DtdErrorKind::EntityExpansionLimit(ExpansionLimit::TotalSize(40_000))
        ),
        "{error}"
    );
}

#[test]
fn honest_documents_are_unaffected_by_the_guards() {
    let xml = r#"<!DOCTYPE a [
        <!ELEMENT a (#PCDATA)>
        <!ENTITY name "OxiXML">
        <!ENTITY greeting "Hello, &name;!">
    ]><a>&greeting;</a>"#;
    let report = DocumentValidator::new()
        .validate_str(xml)
        .expect("well formed");
    assert!(report.is_valid(), "{:?}", report.errors());
}

#[test]
fn entity_replacement_text_keeps_general_references_unexpanded() -> Result<(), DtdError> {
    // XML 1.0 §4.4.7: a general-entity reference inside an EntityValue is
    // *bypassed*, so `outer` stores `&inner;`, not `deep`. That is what makes a
    // bomb's declarations cheap — and why expansion must be metered.
    let dtd = DtdParser::new()
        .parse_internal_subset(r#"<!ENTITY inner "deep"><!ENTITY outer "&inner;&inner;">"#)?;
    assert_eq!(
        dtd.general_entity("outer").and_then(|e| e.internal_text()),
        Some("&inner;&inner;")
    );

    let mut budget = ExpansionLimits::default().budget();
    assert_eq!(
        dtd.expand_general_entity("outer", &mut budget).as_deref(),
        Ok("deepdeep")
    );
    assert_eq!(
        budget.expansions(),
        3,
        "one for `outer` and one for each `inner`"
    );
    Ok(())
}

// ------------------------------------ WFC: No `<` in Attribute Values (XML 1.0 §3.1)

#[test]
fn a_less_than_from_entity_replacement_text_is_rejected_in_an_attribute_value() {
    // "The replacement text of any entity referred to directly or indirectly in an attribute
    // value must not contain a `<`." Here it arrives through a character reference, which the
    // replacement text of an internal entity has already resolved (§4.5).
    let dtd = DtdParser::new()
        .parse_internal_subset(r#"<!ENTITY bad "&#60;tag>">"#)
        .expect("the declaration itself is fine");
    let mut budget = ExpansionLimits::default().budget();
    let error = dtd
        .normalize_attribute_value("&bad;", false, &mut budget)
        .expect_err("a `<` may not reach an attribute value through an entity");
    assert!(
        matches!(error.kind(), DtdErrorKind::LessThanInAttributeValue { chain } if chain == &["bad".to_owned()]),
        "unexpected error kind: {:?}",
        error.kind()
    );
}

#[test]
fn a_less_than_reached_indirectly_through_a_nested_entity_is_rejected() {
    let dtd = DtdParser::new()
        .parse_internal_subset(r#"<!ENTITY lt2 "&#60;"><!ENTITY outer "x&lt2;y">"#)
        .expect("the declarations parse");
    let mut budget = ExpansionLimits::default().budget();
    let error = dtd
        .normalize_attribute_value("&outer;", false, &mut budget)
        .expect_err("indirect references are covered too");
    assert!(matches!(
        error.kind(),
        DtdErrorKind::LessThanInAttributeValue { .. }
    ));
}

#[test]
fn a_literal_less_than_in_entity_replacement_text_is_rejected_too() {
    let dtd = DtdParser::new()
        .parse_internal_subset(r#"<!ENTITY bad "<">"#)
        .expect("`<` is legal inside an EntityValue");
    let mut budget = ExpansionLimits::default().budget();
    assert!(matches!(
        dtd.normalize_attribute_value("&bad;", false, &mut budget)
            .expect_err("but not once expanded into an attribute value")
            .kind(),
        DtdErrorKind::LessThanInAttributeValue { .. }
    ));
}

#[test]
fn a_less_than_written_directly_in_the_attribute_value_is_still_allowed_as_a_reference() {
    // `&#60;` and `&lt;` in the attribute value *itself* are not markup and stay legal: the
    // constraint is about entity *replacement text*, not about the value's own references.
    let dtd = DtdParser::new()
        .parse_internal_subset(r#"<!ENTITY outer "x&lt;y">"#)
        .expect("the declaration parses");
    let mut budget = ExpansionLimits::default().budget();
    assert_eq!(
        dtd.normalize_attribute_value("&#60;a", false, &mut budget)
            .as_deref(),
        Ok("<a")
    );
    // The replacement text of `outer` holds the *reference* `&lt;`, not a `<`, so expanding it
    // is legal — this is the standard way to get a `<` into an attribute value.
    assert_eq!(
        dtd.normalize_attribute_value("&outer;", false, &mut budget)
            .as_deref(),
        Ok("x<y")
    );
}

#[test]
fn the_document_validator_rejects_a_less_than_arriving_through_an_entity() {
    // The same WFC, reached through the crate's headline entry point.
    let document = concat!(
        "<!DOCTYPE doc [<!ELEMENT doc EMPTY>",
        r#"<!ATTLIST doc a CDATA #IMPLIED><!ENTITY bad "&#60;tag>">]>"#,
        r#"<doc a="&bad;"/>"#
    );
    let error = oxixml_dtd::validate_str(document)
        .expect_err("the document is not well-formed under WFC: No < in Attribute Values");
    assert!(
        matches!(
            error,
            Error::Dtd(ref boxed)
                if matches!(boxed.kind(), DtdErrorKind::LessThanInAttributeValue { .. })
        ),
        "unexpected error: {error:?}"
    );
}
