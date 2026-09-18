//! End-to-end tests driving [`Linter::lint`] over small, hand-written SVG
//! documents — one fixture per rule, covering both a positive case (the
//! code fires) and a negative case (it does not), per this crate's mission
//! brief. Narrower, single-function unit tests live alongside the code
//! they test (`content.rs`, and inline in `attrs.rs`/`refs.rs`).
//!
//! Every fixture's root element carries `xmlns="http://www.w3.org/2000/svg"`
//! — without it, the document element is not in [`crate::ns::SVG_NAMESPACE`]
//! and `lint` short-circuits to a lone [`codes::NOT_SVG`], which would make
//! every *negative* assertion in this file pass for the wrong reason. The
//! shared [`lint_codes`] helper guards against exactly that by asserting
//! [`codes::NOT_SVG`] is absent before returning anything to the caller.

use oxixml_dom::build::parse_str;

use crate::lint::diagnostic::{LintOptions, Severity, codes};

use super::Linter;

/// Lints `svg` with every check enabled and returns the codes found, in
/// the order [`Linter::lint`] returned them. Panics if `svg` does not
/// parse as XML, or if the document element is not in the SVG namespace
/// (see the [module documentation](self)).
fn lint_codes(svg: &str) -> Vec<&'static str> {
    lint_codes_with(svg, &Linter::new())
}

/// As [`lint_codes`], but with a caller-supplied [`Linter`] (for the
/// [`LintOptions`] toggle tests).
fn lint_codes_with(svg: &str, linter: &Linter) -> Vec<&'static str> {
    let doc =
        parse_str(svg).unwrap_or_else(|e| panic!("fixture {svg:?} is not well-formed XML: {e}"));
    let diagnostics = linter.lint(&doc);
    assert!(
        diagnostics.iter().all(|d| d.code() != codes::NOT_SVG),
        "fixture {svg:?} must have an `svg` root in the SVG namespace, got {diagnostics:?}"
    );
    diagnostics.iter().map(|d| d.code()).collect()
}

fn count(codes: &[&'static str], code: &'static str) -> usize {
    codes.iter().filter(|&&c| c == code).count()
}

/// As [`lint_codes`], but pairs each code with the [`Severity`] it actually
/// fired at — for tests pinning a specific rule's severity (not just
/// whether the rule fires at all), so a refactor that quietly changes a
/// rule's severity (e.g. the code/severity coupling in
/// `attrs::report_value_check`) cannot pass silently.
fn lint_code_severities(svg: &str) -> Vec<(&'static str, Severity)> {
    let doc =
        parse_str(svg).unwrap_or_else(|e| panic!("fixture {svg:?} is not well-formed XML: {e}"));
    let diagnostics = Linter::new().lint(&doc);
    assert!(
        diagnostics.iter().all(|d| d.code() != codes::NOT_SVG),
        "fixture {svg:?} must have an `svg` root in the SVG namespace, got {diagnostics:?}"
    );
    diagnostics
        .iter()
        .map(|d| (d.code(), d.severity()))
        .collect()
}

/// Looks up the [`Severity`] the first diagnostic with `code` fired at,
/// among `pairs` (as returned by [`lint_code_severities`]). Panics if
/// `code` did not fire at all, so a rule that silently stops firing fails
/// this helper loudly rather than leaving its caller's assertion vacuously
/// unreachable.
fn severity_of(pairs: &[(&'static str, Severity)], code: &'static str) -> Severity {
    pairs
        .iter()
        .find(|pair| pair.0 == code)
        .unwrap_or_else(|| panic!("{code} did not fire in {pairs:?}"))
        .1
}

// ---- NOT_SVG ----

#[test]
fn not_svg_fires_when_the_document_element_is_not_svg() {
    let doc = parse_str(r##"<foo xmlns="urn:example"/>"##).expect("well-formed XML");
    let diagnostics = Linter::new().lint(&doc);
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].code(), codes::NOT_SVG);
}

#[test]
fn not_svg_fires_when_the_document_has_no_element_at_all() {
    // `parse_str` itself rejects a document with no element (it is not
    // well-formed XML), so this branch of `NOT_SVG` is exercised directly
    // against a freshly-constructed, still-empty `Document`.
    let empty = oxixml_dom::Document::new();
    let diagnostics = Linter::new().lint(&empty);
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].code(), codes::NOT_SVG);
    assert_eq!(diagnostics[0].node(), oxixml_dom::NodeId::DOCUMENT);
}

#[test]
fn not_svg_does_not_fire_for_a_real_svg_root() {
    let codes = lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"/>"##);
    assert!(!codes.contains(&codes::NOT_SVG));
}

// ---- UNKNOWN_ELEMENT / UNKNOWN_ATTRIBUTE ----

#[test]
fn unknown_element_fires_for_an_unrecognized_svg_namespace_name() {
    let codes = lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><frobnicate/></svg>"##);
    assert!(codes.contains(&codes::UNKNOWN_ELEMENT));
}

#[test]
fn unknown_element_does_not_fire_for_a_recognized_name() {
    let codes = lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>"##);
    assert!(!codes.contains(&codes::UNKNOWN_ELEMENT));
}

#[test]
fn unknown_attribute_fires_but_data_and_aria_attributes_are_silent() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect frobnicate="1" data-x="1" aria-hidden="true"/></svg>"##,
    );
    assert_eq!(
        count(&codes, codes::UNKNOWN_ATTRIBUTE),
        1,
        "only `frobnicate` should be unknown: {codes:?}"
    );
}

#[test]
fn unknown_attribute_does_not_fire_for_a_recognized_name() {
    let codes = lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><rect fill="red"/></svg>"##);
    assert!(!codes.contains(&codes::UNKNOWN_ATTRIBUTE));
}

#[test]
fn xmlns_declarations_are_never_reported_as_unknown() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg" xmlns:x="urn:example"><rect/></svg>"##,
    );
    assert!(!codes.contains(&codes::UNKNOWN_ATTRIBUTE));
}

// ---- ATTRIBUTE_NOT_APPLICABLE ----

#[test]
fn attribute_not_applicable_fires_for_cx_on_a_rect() {
    let codes = lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><rect cx="1"/></svg>"##);
    assert!(codes.contains(&codes::ATTRIBUTE_NOT_APPLICABLE));
}

#[test]
fn attribute_not_applicable_does_not_fire_for_cx_on_a_circle() {
    let codes = lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><circle cx="1"/></svg>"##);
    assert!(!codes.contains(&codes::ATTRIBUTE_NOT_APPLICABLE));
}

// ---- INVALID_ATTRIBUTE_VALUE ----

#[test]
fn invalid_attribute_value_fires_for_a_malformed_length() {
    let codes =
        lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><rect width="abc"/></svg>"##);
    assert!(codes.contains(&codes::INVALID_ATTRIBUTE_VALUE));
}

#[test]
fn invalid_attribute_value_does_not_fire_for_a_well_formed_length() {
    let codes = lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><rect width="10"/></svg>"##);
    assert!(!codes.contains(&codes::INVALID_ATTRIBUTE_VALUE));
}

#[test]
fn css_wide_keyword_inherit_is_accepted_on_a_presentation_attribute() {
    let codes =
        lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><rect fill="inherit"/></svg>"##);
    assert!(!codes.contains(&codes::INVALID_ATTRIBUTE_VALUE));
}

#[test]
fn css_wide_keyword_is_rejected_on_a_non_presentation_attribute() {
    // `viewBox` is not a presentation property (it is not in
    // `AttrId::is_presentation`'s set), so `"inherit"` is not
    // short-circuited and is instead run through `ViewBox::parse`, which
    // rejects it as a malformed number list.
    let codes =
        lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="inherit"><rect/></svg>"##);
    assert!(codes.contains(&codes::INVALID_ATTRIBUTE_VALUE));
    assert!(!codes.contains(&codes::INVALID_VIEW_BOX));
}

#[test]
fn fill_on_animate_accepts_freeze_and_remove_but_rejects_other_values() {
    let ok = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect><animate attributeName="x" fill="freeze"/></rect></svg>"##,
    );
    assert!(!ok.contains(&codes::INVALID_ATTRIBUTE_VALUE), "{ok:?}");

    let ok_remove = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect><animate attributeName="x" fill="remove"/></rect></svg>"##,
    );
    assert!(
        !ok_remove.contains(&codes::INVALID_ATTRIBUTE_VALUE),
        "{ok_remove:?}"
    );

    let bad = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect><animate attributeName="x" fill="red"/></rect></svg>"##,
    );
    assert!(bad.contains(&codes::INVALID_ATTRIBUTE_VALUE), "{bad:?}");
}

#[test]
fn fill_on_a_shape_is_paint_not_freeze_or_remove() {
    let ok = lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><rect fill="red"/></svg>"##);
    assert!(!ok.contains(&codes::INVALID_ATTRIBUTE_VALUE), "{ok:?}");

    let bad =
        lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><rect fill="freeze"/></svg>"##);
    assert!(
        bad.contains(&codes::INVALID_ATTRIBUTE_VALUE),
        "`freeze` is not a valid <paint> keyword: {bad:?}"
    );
}

#[test]
fn stroke_miterlimit_below_one_is_invalid_but_one_and_above_are_fine() {
    let bad = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect stroke-miterlimit="0.5"/></svg>"##,
    );
    assert!(bad.contains(&codes::INVALID_ATTRIBUTE_VALUE), "{bad:?}");

    let ok = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect stroke-miterlimit="1"/></svg>"##,
    );
    assert!(!ok.contains(&codes::INVALID_ATTRIBUTE_VALUE), "{ok:?}");

    let ok_four = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect stroke-miterlimit="4"/></svg>"##,
    );
    assert!(
        !ok_four.contains(&codes::INVALID_ATTRIBUTE_VALUE),
        "{ok_four:?}"
    );
}

// ---- INVALID_STYLE_ATTRIBUTE ----

#[test]
fn invalid_style_attribute_fires_for_a_declaration_with_no_colon() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect style="not-a-declaration"/></svg>"##,
    );
    assert!(codes.contains(&codes::INVALID_STYLE_ATTRIBUTE));
}

#[test]
fn invalid_style_attribute_does_not_fire_for_a_well_formed_declaration_list() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect style="fill: red; stroke: blue"/></svg>"##,
    );
    assert!(!codes.contains(&codes::INVALID_STYLE_ATTRIBUTE));
}

#[test]
fn a_malformed_value_inside_style_is_reported_through_the_shared_dispatcher() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect style="stroke-miterlimit: 0.5"/></svg>"##,
    );
    assert!(codes.contains(&codes::INVALID_ATTRIBUTE_VALUE));
}

// ---- DUPLICATE_ID ----

#[test]
fn duplicate_id_fires_for_a_repeated_id() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect id="a"/><circle id="a"/></svg>"##,
    );
    assert_eq!(count(&codes, codes::DUPLICATE_ID), 1);
}

#[test]
fn duplicate_id_does_not_fire_for_distinct_ids() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect id="a"/><circle id="b"/></svg>"##,
    );
    assert!(!codes.contains(&codes::DUPLICATE_ID));
}

// ---- DANGLING_REFERENCE ----

#[test]
fn dangling_reference_fires_for_url_in_a_fill_attribute() {
    let codes =
        lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><rect fill="url(#nope)"/></svg>"##);
    assert!(codes.contains(&codes::DANGLING_REFERENCE));
}

#[test]
fn dangling_reference_fires_for_url_inside_a_style_declaration() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect style="fill:url(#nope)"/></svg>"##,
    );
    assert!(codes.contains(&codes::DANGLING_REFERENCE));
}

#[test]
fn dangling_reference_does_not_fire_when_the_target_exists() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect fill="url(#a)"/><linearGradient id="a"/></svg>"##,
    );
    assert!(
        !codes.contains(&codes::DANGLING_REFERENCE),
        "a forward reference to an id defined later in the document must resolve: {codes:?}"
    );
}

// ---- CIRCULAR_REFERENCE ----

#[test]
fn circular_reference_fires_for_a_use_referencing_itself() {
    let codes =
        lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><use id="u1" href="#u1"/></svg>"##);
    assert_eq!(count(&codes, codes::CIRCULAR_REFERENCE), 1);
}

#[test]
fn circular_reference_fires_for_a_mutual_two_use_cycle() {
    // A ↔ B: the `<use>` inside `A` targets `B`, whose own nested `<use>`
    // targets `A` back. Both participating `<use>` elements independently
    // discover, from their own vantage point, that expansion revisits an
    // ancestor of the walk they started — so both are reported, per the
    // algorithm's own "flag *that* `use`" framing (see `refs.rs`).
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg">
             <g id="A"><use href="#B"/></g>
             <g id="B"><use href="#A"/></g>
           </svg>"##,
    );
    assert_eq!(count(&codes, codes::CIRCULAR_REFERENCE), 2, "{codes:?}");
}

#[test]
fn circular_reference_does_not_fire_for_two_uses_sharing_one_legitimate_target() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg">
             <symbol id="s"><rect/></symbol>
             <use href="#s"/>
             <use href="#s"/>
           </svg>"##,
    );
    assert!(
        !codes.contains(&codes::CIRCULAR_REFERENCE),
        "two independent uses of the same symbol is a DAG, not a cycle: {codes:?}"
    );
}

#[test]
fn circular_reference_fires_for_a_gradient_href_chain_cycle() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg">
             <linearGradient id="a" href="#b"/>
             <linearGradient id="b" href="#a"/>
           </svg>"##,
    );
    assert_eq!(count(&codes, codes::CIRCULAR_REFERENCE), 2, "{codes:?}");
}

#[test]
fn circular_reference_does_not_fire_for_a_plain_gradient_template_chain() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg">
             <linearGradient id="base"><stop offset="0" stop-color="red"/></linearGradient>
             <linearGradient id="derived" href="#base"/>
           </svg>"##,
    );
    assert!(!codes.contains(&codes::CIRCULAR_REFERENCE));
}

// ---- INVALID_VIEW_BOX ----

#[test]
fn invalid_view_box_fires_for_a_negative_width() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 -100 100"><rect/></svg>"##,
    );
    assert!(codes.contains(&codes::INVALID_VIEW_BOX));
    assert!(
        !codes.contains(&codes::INVALID_ATTRIBUTE_VALUE),
        "a negative viewBox dimension is the more specific code, not the generic one: {codes:?}"
    );
}

#[test]
fn invalid_view_box_does_not_fire_for_a_well_formed_view_box() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect/></svg>"##,
    );
    assert!(!codes.contains(&codes::INVALID_VIEW_BOX));
}

#[test]
fn zero_width_view_box_is_grammar_valid_not_invalid() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 0 100"><rect/></svg>"##,
    );
    assert!(!codes.contains(&codes::INVALID_VIEW_BOX));
    assert!(!codes.contains(&codes::INVALID_ATTRIBUTE_VALUE));
}

// ---- NEGATIVE_DIMENSION ----

#[test]
fn negative_dimension_fires_for_a_negative_rect_width() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect width="-1" height="10"/></svg>"##,
    );
    assert!(codes.contains(&codes::NEGATIVE_DIMENSION));
    assert!(
        !codes.contains(&codes::INVALID_ATTRIBUTE_VALUE),
        "a negative dimension is the more specific code: {codes:?}"
    );
}

#[test]
fn negative_dimension_does_not_fire_for_a_non_negative_width() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect width="10" height="10"/></svg>"##,
    );
    assert!(!codes.contains(&codes::NEGATIVE_DIMENSION));
}

#[test]
fn negative_dimension_does_not_apply_to_signed_coordinates() {
    // `x`/`y` may legitimately be negative; only the five attributes in
    // `attrs::NON_NEGATIVE_LENGTH_ATTRS` are checked for sign.
    let codes =
        lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><rect x="-5" y="-5"/></svg>"##);
    assert!(!codes.contains(&codes::NEGATIVE_DIMENSION));
}

#[test]
fn auto_keyword_is_accepted_for_width_height_rx_ry() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect width="auto" height="auto" rx="auto" ry="auto"/></svg>"##,
    );
    assert!(!codes.contains(&codes::INVALID_ATTRIBUTE_VALUE));
    assert!(!codes.contains(&codes::NEGATIVE_DIMENSION));
}

// ---- DEPRECATED_ELEMENT ----

#[test]
fn deprecated_element_fires_for_tref() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><text><tref xlink:href="#x"/></text></svg>"##,
    );
    assert!(codes.contains(&codes::DEPRECATED_ELEMENT));
}

#[test]
fn deprecated_element_does_not_fire_for_a_stable_element() {
    let codes = lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>"##);
    assert!(!codes.contains(&codes::DEPRECATED_ELEMENT));
}

// ---- DEPRECATED_ATTRIBUTE ----

#[test]
fn deprecated_attribute_fires_for_xml_space() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect xml:space="preserve"/></svg>"##,
    );
    assert!(codes.contains(&codes::DEPRECATED_ATTRIBUTE));
}

#[test]
fn deprecated_attribute_does_not_fire_for_the_ubiquitous_xlink_href() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><a xlink:href="#x"/></svg>"##,
    );
    assert!(
        !codes.contains(&codes::DEPRECATED_ATTRIBUTE),
        "xlink:href remains valid indefinitely: {codes:?}"
    );
}

// ---- CONTENT_MODEL_VIOLATION ----

#[test]
fn content_model_violation_fires_for_a_rect_inside_a_rect() {
    let codes =
        lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><rect><rect/></rect></svg>"##);
    assert!(codes.contains(&codes::CONTENT_MODEL_VIOLATION));
}

#[test]
fn content_model_violation_does_not_fire_for_a_rect_inside_a_g() {
    let codes = lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><g><rect/></g></svg>"##);
    assert!(!codes.contains(&codes::CONTENT_MODEL_VIOLATION));
}

// ---- CHARACTER_DATA_NOT_ALLOWED ----

#[test]
fn character_data_not_allowed_fires_for_text_inside_a_group() {
    let codes = lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><g>hello</g></svg>"##);
    assert!(codes.contains(&codes::CHARACTER_DATA_NOT_ALLOWED));
}

#[test]
fn character_data_not_allowed_does_not_fire_for_whitespace_only_text() {
    let codes = lint_codes("<svg xmlns=\"http://www.w3.org/2000/svg\"><g>\n  </g></svg>");
    assert!(!codes.contains(&codes::CHARACTER_DATA_NOT_ALLOWED));
}

#[test]
fn character_data_not_allowed_does_not_fire_inside_a_text_element() {
    let codes = lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><text>hello</text></svg>"##);
    assert!(!codes.contains(&codes::CHARACTER_DATA_NOT_ALLOWED));
}

// ---- HREF_DUPLICATED ----

#[test]
fn href_duplicated_fires_when_both_forms_are_present() {
    let codes = lint_codes(
        r##"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><use href="#a" xlink:href="#a"/></svg>"##,
    );
    assert!(codes.contains(&codes::HREF_DUPLICATED));
}

#[test]
fn href_duplicated_does_not_fire_for_a_single_form() {
    let codes = lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><use href="#a"/></svg>"##);
    assert!(!codes.contains(&codes::HREF_DUPLICATED));
}

// ---- EMPTY_USE_REFERENCE ----

#[test]
fn empty_use_reference_fires_when_there_is_no_href_at_all() {
    let codes = lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><use/></svg>"##);
    assert!(codes.contains(&codes::EMPTY_USE_REFERENCE));
}

#[test]
fn empty_use_reference_fires_for_an_empty_href_value() {
    let codes = lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><use href=""/></svg>"##);
    assert!(codes.contains(&codes::EMPTY_USE_REFERENCE));
}

#[test]
fn empty_use_reference_does_not_fire_for_a_non_empty_href() {
    // Even a dangling reference is not "empty" — that is a distinct,
    // separately-reported condition (`DANGLING_REFERENCE`).
    let codes =
        lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><use href="#nope"/></svg>"##);
    assert!(!codes.contains(&codes::EMPTY_USE_REFERENCE));
    assert!(codes.contains(&codes::DANGLING_REFERENCE));
}

// ---- TRUNCATED ----

#[test]
fn truncated_fires_when_the_diagnostic_count_exceeds_the_limit() {
    let linter = Linter::with_options(LintOptions::new().max_diagnostics(1));
    let codes = lint_codes_with(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect foo="1"/><rect bar="1"/></svg>"##,
        &linter,
    );
    assert_eq!(codes.last(), Some(&codes::TRUNCATED));
    // Exactly `limit` real diagnostics, plus the truncation notice itself.
    assert_eq!(codes.len(), 2);
}

#[test]
fn truncated_does_not_fire_under_the_default_limit() {
    let codes = lint_codes(r##"<svg xmlns="http://www.w3.org/2000/svg"><rect foo="1"/></svg>"##);
    assert!(!codes.contains(&codes::TRUNCATED));
}

#[test]
fn truncated_with_a_zero_limit_reports_only_the_notice() {
    let linter = Linter::with_options(LintOptions::new().max_diagnostics(0));
    let codes = lint_codes_with(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect foo="1"/></svg>"##,
        &linter,
    );
    assert_eq!(codes, vec![codes::TRUNCATED]);
}

// ---- Severity: the mission pins specific rules to specific severities,
// not just whether they fire ----

#[test]
fn each_pinned_rule_fires_at_its_documented_severity() {
    let pairs = lint_code_severities(
        r##"<svg xmlns="http://www.w3.org/2000/svg">
             <rect id="a" width="-1"/>
             <circle id="a"/>
             <use id="u1" href="#u1"/>
             <rect><rect/></rect>
             <g xml:space="preserve"/>
           </svg>"##,
    );
    assert_eq!(
        severity_of(&pairs, codes::NEGATIVE_DIMENSION),
        Severity::Error,
        "a negative width/height/r/rx/ry is a hard conformance failure"
    );
    assert_eq!(
        severity_of(&pairs, codes::DUPLICATE_ID),
        Severity::Error,
        "a duplicate id breaks reference resolution"
    );
    assert_eq!(
        severity_of(&pairs, codes::CIRCULAR_REFERENCE),
        Severity::Error,
        "a reference cycle is a hard conformance failure"
    );
    assert_eq!(
        severity_of(&pairs, codes::CONTENT_MODEL_VIOLATION),
        Severity::Warning,
        "a content-model violation is non-conformant SVG but still well-formed XML"
    );
    assert_eq!(
        severity_of(&pairs, codes::DEPRECATED_ATTRIBUTE),
        Severity::Info,
        "deliberately quieter than DEPRECATED_ELEMENT — xlink:href is ubiquitous \
         and this severity applies to every member of that code, not just xlink:href"
    );
}

// ---- LintOptions toggles ----

#[test]
fn check_values_false_suppresses_value_grammar_codes() {
    let linter = Linter::with_options(LintOptions::new().check_values(false));
    let codes = lint_codes_with(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect width="abc"/></svg>"##,
        &linter,
    );
    assert!(!codes.contains(&codes::INVALID_ATTRIBUTE_VALUE));
}

#[test]
fn check_content_model_false_suppresses_content_model_violation() {
    let linter = Linter::with_options(LintOptions::new().check_content_model(false));
    let codes = lint_codes_with(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect><rect/></rect></svg>"##,
        &linter,
    );
    assert!(!codes.contains(&codes::CONTENT_MODEL_VIOLATION));
}

#[test]
fn check_references_false_suppresses_reference_codes() {
    let linter = Linter::with_options(LintOptions::new().check_references(false));
    let codes = lint_codes_with(
        r##"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
             <rect fill="url(#nope)"/><use/><use href="#a" xlink:href="#a"/>
           </svg>"##,
        &linter,
    );
    assert!(!codes.contains(&codes::DANGLING_REFERENCE));
    assert!(!codes.contains(&codes::EMPTY_USE_REFERENCE));
    // `HREF_DUPLICATED` is a pure name-collision finding, not a reference
    // *resolution* — but `LintOptions::check_references`'s own
    // documentation groups it under this toggle, and `Walker::walk` gates
    // it on `checks_references()` accordingly. Pin that coupling here so a
    // refactor cannot silently move it to another toggle unnoticed.
    assert!(!codes.contains(&codes::HREF_DUPLICATED));
}

#[test]
fn check_deprecated_false_suppresses_deprecation_codes() {
    let linter = Linter::with_options(LintOptions::new().check_deprecated(false));
    let codes = lint_codes_with(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect xml:space="preserve"/></svg>"##,
        &linter,
    );
    assert!(!codes.contains(&codes::DEPRECATED_ATTRIBUTE));
}

#[test]
fn check_unknown_false_suppresses_unknown_and_applicability_codes() {
    let linter = Linter::with_options(LintOptions::new().check_unknown(false));
    let codes = lint_codes_with(
        r##"<svg xmlns="http://www.w3.org/2000/svg"><rect frobnicate="1" cx="1"/></svg>"##,
        &linter,
    );
    assert!(!codes.contains(&codes::UNKNOWN_ATTRIBUTE));
    assert!(!codes.contains(&codes::ATTRIBUTE_NOT_APPLICABLE));
}
