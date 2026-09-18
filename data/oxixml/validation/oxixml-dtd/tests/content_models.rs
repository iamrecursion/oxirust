//! Acceptance and rejection of child sequences by the compiled content-model
//! automata (XML 1.0 §3.2, VC: Element Valid).

use oxixml_dtd::{
    CompiledContent, DtdParser, ValidationOptions, Validator, ValidityError, ValidityErrorKind,
    validate_str,
};

/// Compiles `model` as the content model of element `a` and walks `children`
/// through it, reporting whether the sequence is accepted in full.
fn accepts(model: &str, children: &[&str]) -> bool {
    let dtd = DtdParser::new()
        .parse_internal_subset(&format!("<!ELEMENT a {model}>"))
        .expect("the content model parses");
    let compiled = dtd.compiled_content("a").expect("`a` is declared");
    let mut matcher = compiled.matcher();
    for child in children {
        if !matcher.accept(child) {
            return false;
        }
    }
    matcher.is_final()
}

#[test]
fn sequences_require_every_member_in_order() {
    assert!(accepts("(b, c, d)", &["b", "c", "d"]));
    assert!(!accepts("(b, c, d)", &["b", "d", "c"]));
    assert!(!accepts("(b, c, d)", &["b", "c"]));
    assert!(!accepts("(b, c, d)", &["b", "c", "d", "d"]));
}

#[test]
fn choices_accept_exactly_one_alternative() {
    assert!(accepts("(b | c)", &["b"]));
    assert!(accepts("(b | c)", &["c"]));
    assert!(!accepts("(b | c)", &[]));
    assert!(!accepts("(b | c)", &["b", "c"]));
    assert!(!accepts("(b | c)", &["d"]));
}

#[test]
fn occurrence_operators_behave_as_the_grammar_says() {
    assert!(accepts("(b?)", &[]));
    assert!(accepts("(b?)", &["b"]));
    assert!(!accepts("(b?)", &["b", "b"]));

    assert!(accepts("(b*)", &[]));
    assert!(accepts("(b*)", &["b", "b", "b"]));

    assert!(!accepts("(b+)", &[]));
    assert!(accepts("(b+)", &["b", "b"]));
}

#[test]
fn nested_groups_compose() {
    let model = "(head, (p | list)*, tail?)";
    assert!(accepts(model, &["head"]));
    assert!(accepts(model, &["head", "tail"]));
    assert!(accepts(model, &["head", "p", "list", "p", "tail"]));
    assert!(!accepts(model, &["p", "head"]));
    assert!(!accepts(model, &["head", "tail", "p"]));
    assert!(!accepts(model, &["head", "tail", "tail"]));
}

#[test]
fn repeated_groups_loop_back_to_their_own_start() {
    assert!(accepts("((b, c)+)", &["b", "c", "b", "c"]));
    assert!(!accepts("((b, c)+)", &["b", "c", "b"]));
    assert!(accepts("((b | c)*, d)", &["d"]));
    assert!(accepts("((b | c)*, d)", &["c", "b", "b", "d"]));
}

#[test]
fn empty_and_any_are_the_two_extremes() {
    assert!(accepts("EMPTY", &[]));
    assert!(!accepts("EMPTY", &["b"]));
    assert!(accepts("ANY", &[]));
    assert!(accepts("ANY", &["b", "c", "b"]));
}

#[test]
fn mixed_content_is_an_unordered_set() {
    assert!(accepts("(#PCDATA)", &[]));
    assert!(!accepts("(#PCDATA)", &["b"]));
    assert!(accepts("(#PCDATA | b | c)*", &["c", "b", "c"]));
    assert!(!accepts("(#PCDATA | b | c)*", &["d"]));
}

#[test]
fn a_deeply_nested_model_compiles_and_matches() {
    let model = "((a1 | a2), (b1, (c1 | c2)+)*, d1?, (e1, e2)?)";
    assert!(accepts(model, &["a2"]));
    assert!(accepts(model, &["a1", "b1", "c1", "c2", "d1", "e1", "e2"]));
    assert!(!accepts(model, &["a1", "b1", "d1", "e1"]));
    assert!(!accepts(model, &["a1", "e1"]));
}

/// The names a matcher would accept next, after walking `children`.
fn expected_after(model: &str, children: &[&str]) -> Vec<String> {
    let dtd = DtdParser::new()
        .parse_internal_subset(&format!("<!ELEMENT a {model}>"))
        .expect("the content model parses");
    let compiled = dtd.compiled_content("a").expect("`a` is declared");
    let mut matcher = compiled.matcher();
    for child in children {
        assert!(matcher.accept(child), "`{child}` should be accepted");
    }
    matcher.expected()
}

#[test]
fn the_matcher_reports_what_would_come_next() {
    assert_eq!(expected_after("(b, (c | d))", &[]), ["b"]);
    assert_eq!(expected_after("(b, (c | d))", &["b"]), ["c", "d"]);
    assert_eq!(
        expected_after("(b, (c | d))", &["b", "c"]),
        Vec::<String>::new()
    );
}

#[test]
fn element_content_forbids_character_data_but_not_white_space() {
    let report =
        validate_str("<!DOCTYPE a [<!ELEMENT a (b)><!ELEMENT b EMPTY>]>\n<a>\n  <b/>\n</a>")
            .expect("well formed");
    assert!(report.is_valid(), "{:?}", report.errors());

    let report = validate_str("<!DOCTYPE a [<!ELEMENT a (b)><!ELEMENT b EMPTY>]><a>x<b/></a>")
        .expect("well formed");
    assert!(report.errors().iter().any(|error| matches!(
        error.kind(),
        ValidityErrorKind::CharacterDataNotAllowed { .. }
    )));
}

#[test]
fn a_bad_child_is_reported_once_and_does_not_cascade() {
    let dtd = DtdParser::new()
        .parse_internal_subset("<!ELEMENT a (b, c)><!ELEMENT b EMPTY><!ELEMENT c EMPTY>")
        .expect("valid DTD");
    let mut validator = Validator::new(&dtd, ValidationOptions::default());
    validator.start_element("a", &[]);
    for child in ["c", "c", "c"] {
        validator.start_element(child, &[]);
        validator.end_element(child);
    }
    validator.end_element("a");
    let errors: Vec<&ValidityErrorKind> =
        validator.errors().iter().map(ValidityError::kind).collect();
    assert_eq!(errors.len(), 1, "{errors:?}");
}

#[test]
fn ambiguous_models_still_validate_correctly() {
    // XML 1.0 Appendix E asks for deterministic models; this one is not, and is
    // matched through the subset construction rather than rejected.
    let dtd = DtdParser::new()
        .parse_internal_subset("<!ELEMENT a ((b, c) | (b, d))><!ELEMENT b EMPTY>")
        .expect("valid DTD");
    let compiled = dtd.compiled_content("a").expect("declared");
    assert!(!compiled.is_deterministic());
    assert!(matches!(compiled, CompiledContent::Children(_)));

    let mut matcher = compiled.matcher();
    assert!(matcher.accept("b"));
    assert!(matcher.accept("d"));
    assert!(matcher.is_final());

    // …and the ambiguity is reported when asked for.
    let mut validator = Validator::new(&dtd, ValidationOptions::pedantic());
    validator.start_element("a", &[]);
    validator.end_element("a");
    assert!(validator.errors().iter().any(|error| matches!(
        error.kind(),
        ValidityErrorKind::NonDeterministicContentModel { .. }
    )));
}

#[test]
fn a_wide_model_compiles_without_blowing_up() {
    // Wide but legitimate: well under `MAX_CONTENT_MODEL_POSITIONS`, comfortably past what any
    // real DTD names in one group, yet still nowhere near the cap this crate now enforces.
    let count = oxixml_dtd::MAX_CONTENT_MODEL_POSITIONS / 2;
    let alternatives: Vec<String> = (0..count).map(|index| format!("e{index}")).collect();
    let model = format!("({})*", alternatives.join(" | "));
    let dtd = DtdParser::new()
        .parse_internal_subset(&format!("<!ELEMENT a {model}>"))
        .expect("the content model parses");
    let compiled = dtd.compiled_content("a").expect("declared");
    let mut matcher = compiled.matcher();
    for name in &alternatives {
        assert!(matcher.accept(name));
    }
    assert!(matcher.is_final());
    assert!(compiled.is_deterministic());
}

// ------------------------------------------------- hostile grammar depth (§3.2.1 [47]-[50])

#[test]
fn deeply_nested_content_model_groups_are_a_clean_error_not_a_stack_overflow() {
    // `children`/`cp`/`choice`/`seq` are mutually recursive productions, so a declaration that
    // nests parentheses tens of thousands deep would exhaust the stack of a naive parser. The
    // depth cap turns it into an ordinary error, the way the entity-expansion budget does.
    let depth = oxixml_dtd::MAX_CONTENT_MODEL_DEPTH * 4;
    let text = format!(
        "<!ELEMENT root {}a{}>",
        "(".repeat(depth),
        ")".repeat(depth)
    );
    let error = oxixml_dtd::DtdParser::new()
        .parse_internal_subset(&text)
        .expect_err("a content model nested that deep must be rejected");
    assert!(
        matches!(
            error.kind(),
            oxixml_dtd::DtdErrorKind::ContentModelDepth(limit)
                if *limit == oxixml_dtd::MAX_CONTENT_MODEL_DEPTH
        ),
        "unexpected error kind: {:?}",
        error.kind()
    );
}

#[test]
fn honestly_nested_content_model_groups_still_parse() {
    // The cap must be far above anything a real content model reaches.
    let depth = 32;
    let text = format!(
        "<!ELEMENT root {}a{}>",
        "(".repeat(depth),
        ")".repeat(depth)
    );
    let dtd = DtdParser::new()
        .parse_internal_subset(&text)
        .expect("a modestly nested content model is fine");
    assert!(dtd.compiled_content("root").is_some());
}

// ------------------------------------------------- hostile grammar position count (§3.2.1 [47])

#[test]
fn a_flat_group_naming_too_many_children_is_a_clean_error_not_a_superlinear_stall() {
    // A single, flat, repeatable group of many distinct child names — `(e0|e1|...|eN)+` — sits
    // at parenthesis-nesting depth 1, entirely unbounded by `MAX_CONTENT_MODEL_DEPTH`, yet
    // compiling it into the position automaton is superlinear in the number of leaves. The
    // position-count cap must reject it before the expensive automaton construction runs.
    let count = oxixml_dtd::MAX_CONTENT_MODEL_POSITIONS + 1;
    let names: Vec<String> = (0..count).map(|i| format!("e{i}")).collect();
    let text = format!("<!ELEMENT root ({})+>", names.join("|"));
    let error = DtdParser::new()
        .parse_internal_subset(&text)
        .expect_err("a content model naming this many positions must be rejected");
    assert!(
        matches!(
            error.kind(),
            oxixml_dtd::DtdErrorKind::ContentModelPositions(limit)
                if *limit == oxixml_dtd::MAX_CONTENT_MODEL_POSITIONS
        ),
        "unexpected error kind: {:?}",
        error.kind()
    );
}

#[test]
fn a_flat_group_at_the_position_cap_still_compiles() {
    // The cap must be far above anything a real content model reaches, and must not reject a
    // model that lands exactly on the limit.
    let count = oxixml_dtd::MAX_CONTENT_MODEL_POSITIONS;
    let names: Vec<String> = (0..count).map(|i| format!("e{i}")).collect();
    let text = format!("<!ELEMENT root ({})+>", names.join("|"));
    let dtd = DtdParser::new()
        .parse_internal_subset(&text)
        .expect("a content model naming exactly the cap's worth of positions is fine");
    assert!(dtd.compiled_content("root").is_some());
}

#[test]
fn positions_spread_across_several_nested_groups_still_count_toward_the_cap() {
    // The cap counts leaves anywhere in the particle tree, not just within one flat group, so
    // splitting a hostile declaration across nested groups does not evade it.
    let count = oxixml_dtd::MAX_CONTENT_MODEL_POSITIONS + 1;
    let names: Vec<String> = (0..count).map(|i| format!("e{i}")).collect();
    let groups: Vec<String> = names
        .chunks(4)
        .map(|chunk| format!("({})", chunk.join("|")))
        .collect();
    let text = format!("<!ELEMENT root ({})>", groups.join(","));
    let error = DtdParser::new()
        .parse_internal_subset(&text)
        .expect_err("splitting the same leaf count across nested groups is still rejected");
    assert!(
        matches!(
            error.kind(),
            oxixml_dtd::DtdErrorKind::ContentModelPositions(limit)
                if *limit == oxixml_dtd::MAX_CONTENT_MODEL_POSITIONS
        ),
        "unexpected error kind: {:?}",
        error.kind()
    );
}
