//! Regression tests for the audit findings against `oxixml-fn`.
//!
//! Each case is the hostile input that used to give a wrong answer:
//!
//! * an `xs:integer` argument above 2^53 was routed through `f64` and clamped;
//! * `fn:deep-equal` under any collation beyond the two built-ins either raised `FOCH0002` on
//!   two identical nodes or silently answered `false`;
//! * `fn:serialize`'s option-map form reported an invalid `method` as the generic `FORG0006`.

use oxixml_fn::{Collations, FnContext, FnErrorCode, deep_equal_with};
use oxixml_infoset::{AtomicValue, Item, Sequence, Tree, XdmMap};
use oxixml_xsd::TimezoneOffset;

const UCA_PRIMARY: &str = "http://www.w3.org/2013/collation/UCA?lang=en;strength=primary";

fn uca() -> std::sync::Arc<dyn oxixml_fn::Collation> {
    Collations::new()
        .resolve(UCA_PRIMARY)
        .expect("the UCA collation resolves")
}

fn codepoint() -> std::sync::Arc<dyn oxixml_fn::Collation> {
    Collations::new().codepoint()
}

fn element(xml: &str) -> Sequence {
    let tree = Tree::parse_str(xml).expect("the document parses");
    Sequence::from(tree.document_element().expect("a document element"))
}

// --- fn:deep-equal under a resolved collation --------------------------------------------------

#[test]
fn deep_equal_does_not_raise_on_two_identical_elements_under_a_uca_collation() {
    let left = element("<a>x</a>");
    let right = element("<a>x</a>");
    assert_eq!(
        deep_equal_with(&left, &right, &uca(), TimezoneOffset::UTC),
        Ok(true),
        "two identical elements are deep-equal under every collation"
    );
}

#[test]
fn deep_equal_compares_attribute_values_under_the_collation() {
    let left = element(r#"<a b="x"/>"#);
    let right = element(r#"<a b="x"/>"#);
    assert_eq!(
        deep_equal_with(&left, &right, &uca(), TimezoneOffset::UTC),
        Ok(true)
    );

    // At primary strength the two attribute values differ only in case, so they are equal.
    let upper = element(r#"<a b="X"/>"#);
    assert_eq!(
        deep_equal_with(&left, &upper, &uca(), TimezoneOffset::UTC),
        Ok(true)
    );
    assert_eq!(
        deep_equal_with(&left, &upper, &codepoint(), TimezoneOffset::UTC),
        Ok(false)
    );
}

#[test]
fn deep_equal_compares_atomic_strings_under_the_collation() {
    let left = Sequence::from(AtomicValue::string("a"));
    let right = Sequence::from(AtomicValue::string("A"));
    assert_eq!(
        deep_equal_with(&left, &right, &uca(), TimezoneOffset::UTC),
        Ok(true)
    );
    assert_eq!(
        deep_equal_with(&left, &right, &codepoint(), TimezoneOffset::UTC),
        Ok(false)
    );
}

#[test]
fn deep_equal_compares_text_children_under_the_collation() {
    let left = element("<a>x</a>");
    let right = element("<a>X</a>");
    assert_eq!(
        deep_equal_with(&left, &right, &uca(), TimezoneOffset::UTC),
        Ok(true)
    );
    assert_eq!(
        deep_equal_with(&left, &right, &codepoint(), TimezoneOffset::UTC),
        Ok(false)
    );
}

#[test]
fn deep_equal_keeps_comparing_numbers_by_value() {
    let left = Sequence::from(AtomicValue::integer(1_i64));
    let right = Sequence::from(AtomicValue::double(1.0));
    assert_eq!(
        deep_equal_with(&left, &right, &uca(), TimezoneOffset::UTC),
        Ok(true)
    );

    let unrelated = Sequence::from(AtomicValue::string("1"));
    assert_eq!(
        deep_equal_with(&left, &unrelated, &uca(), TimezoneOffset::UTC),
        Ok(false)
    );
}

#[test]
fn deep_equal_map_keys_stay_collation_independent() {
    // F&O 3.1 compares map *keys* with `op:same-key`, which no collation influences.
    let mut lower = XdmMap::new();
    lower.insert(
        AtomicValue::string("a"),
        Sequence::from(AtomicValue::integer(1_i64)),
    );
    let mut upper = XdmMap::new();
    upper.insert(
        AtomicValue::string("A"),
        Sequence::from(AtomicValue::integer(1_i64)),
    );
    let left = Sequence::singleton(Item::Map(lower));
    let right = Sequence::singleton(Item::Map(upper));
    assert_eq!(
        deep_equal_with(&left, &right, &uca(), TimezoneOffset::UTC),
        Ok(false)
    );
}

#[test]
fn deep_equal_through_the_function_context_resolves_the_collation() {
    let context = FnContext::new();
    let left = Sequence::from(AtomicValue::string("a"));
    let right = Sequence::from(AtomicValue::string("A"));
    assert_eq!(
        oxixml_fn::sequences::deep_equal(&context, &left, &right, Some(UCA_PRIMARY)),
        Ok(true)
    );
    let error = oxixml_fn::sequences::deep_equal(&context, &left, &right, Some("urn:nope"))
        .expect_err("an unknown collation is still FOCH0002");
    assert_eq!(error.code(), Some(FnErrorCode::Foch0002));
}

#[test]
fn deep_equal_reads_a_floating_date_time_in_the_context_implicit_timezone() {
    // F&O 3.1 §15.3.1: two atomic values are deep-equal exactly when they are `eq`, and `eq`
    // reads a date/time operand with no timezone in the dynamic context's implicit timezone
    // (§10.4.4) — so this must vary with the context exactly as `atomic_equal_in_context` does.
    let floating = Sequence::from(date_time("2000-01-01T00:00:00"));
    let zoned = Sequence::from(date_time("2000-01-01T05:00:00Z"));

    assert_eq!(
        oxixml_fn::sequences::deep_equal(&FnContext::new(), &floating, &zoned, None),
        Ok(false),
        "under the default (UTC) implicit timezone the two denote different instants"
    );
    assert_eq!(
        oxixml_fn::sequences::deep_equal(&minus_five(), &floating, &zoned, None),
        Ok(true),
        "fn:deep-equal must use the context's implicit timezone, not hardcode UTC"
    );
}

// --- exact xs:integer arguments ----------------------------------------------------------------

use oxixml_fn::registry;
use oxixml_infoset::ExpandedName;

fn call(local: &str, arguments: &[Sequence]) -> Result<Sequence, oxixml_fn::FnError> {
    let catalogue = registry::builtins();
    let name = ExpandedName::new(Some(registry::FN_NAMESPACE), local);
    let function = catalogue
        .lookup(&name, arguments.len())
        .unwrap_or_else(|| panic!("fn:{local}#{} exists", arguments.len()));
    function.call(&FnContext::new(), arguments)
}

fn rendered(sequence: &Sequence) -> String {
    sequence
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

#[test]
fn format_integer_keeps_integers_above_two_to_the_fifty_third_exact() {
    for value in [
        9_007_199_254_740_993_i64,
        1_234_567_890_123_456_789_i64,
        i64::MAX,
        i64::MIN + 1,
    ] {
        let formatted = call(
            "format-integer",
            &[
                Sequence::from(AtomicValue::integer(value)),
                Sequence::from(AtomicValue::string("1")),
            ],
        )
        .expect("format-integer accepts an xs:integer");
        assert_eq!(rendered(&formatted), value.to_string());
    }
}

#[test]
fn remove_and_insert_before_address_positions_exactly() {
    let items = Sequence::from_iter(["a", "b", "c"].map(AtomicValue::string));
    let removed = call(
        "remove",
        &[items.clone(), Sequence::from(AtomicValue::integer(2_i64))],
    )
    .expect("fn:remove");
    assert_eq!(rendered(&removed), "a,c");

    // A position far beyond 2^53 used to be clamped to 9.0e18 and could alias a real position.
    let untouched = call(
        "remove",
        &[
            items.clone(),
            Sequence::from(AtomicValue::integer(i64::MAX)),
        ],
    )
    .expect("fn:remove");
    assert_eq!(rendered(&untouched), "a,b,c");
}

#[test]
fn an_out_of_range_integer_argument_is_a_range_error_not_a_silent_clamp() {
    let error = call(
        "format-integer",
        &[
            Sequence::from(AtomicValue::double(1.0e30)),
            Sequence::from(AtomicValue::string("1")),
        ],
    )
    .expect_err("1.0e30 is outside the xs:integer range this processor represents");
    // The point is that it is refused rather than silently clamped to 9.0e18: the signature
    // layer rejects it as a type error, and the argument reader as a range error.
    assert!(
        matches!(
            error.code(),
            Some(FnErrorCode::Foca0003 | FnErrorCode::Xpty0004)
        ),
        "unexpected code {:?}",
        error.code()
    );
}

// --- fn:serialize option map -------------------------------------------------------------------

#[test]
fn an_invalid_serialization_method_is_sepm0016() {
    let mut options = XdmMap::new();
    options.insert(
        AtomicValue::string("method"),
        Sequence::from(AtomicValue::string("bogus")),
    );
    let error = call(
        "serialize",
        &[
            Sequence::from(AtomicValue::integer(1_i64)),
            Sequence::singleton(Item::Map(options)),
        ],
    )
    .expect_err("`bogus` is not a serialization method");
    assert_eq!(error.code(), Some(FnErrorCode::Sepm0016));
}

#[test]
fn an_invalid_json_node_output_method_is_sepm0016() {
    let mut options = XdmMap::new();
    options.insert(
        AtomicValue::string("json-node-output-method"),
        Sequence::from(AtomicValue::string("bogus")),
    );
    let error = call(
        "serialize",
        &[
            Sequence::from(AtomicValue::integer(1_i64)),
            Sequence::singleton(Item::Map(options)),
        ],
    )
    .expect_err("`bogus` is not a serialization method");
    assert_eq!(error.code(), Some(FnErrorCode::Sepm0016));
}

// --- implicit timezone in fn:min, fn:max, fn:distinct-values and fn:index-of --------------------
//
// F&O 3.1 §10.4.4 reads a date/time value with no timezone in the *dynamic context's* implicit
// timezone before any comparison, including the ones `fn:min`, `fn:max`, `fn:distinct-values` and
// `fn:index-of` make internally. Each case below pins a non-UTC implicit timezone, `-05:00`, and
// checks the answer changes exactly as F&O predicts — proving the context is threaded through
// rather than a hardcoded UTC default silently standing in for it.

fn date_time(lexical: &str) -> AtomicValue {
    AtomicValue::parse(oxixml_infoset::AtomicType::DateTime, lexical)
        .expect("a valid xs:dateTime lexical form")
}

fn minus_five() -> FnContext {
    FnContext::new().with_implicit_timezone(TimezoneOffset::new(-300).expect("-05:00 is valid"))
}

#[test]
fn index_of_reads_a_floating_date_time_in_the_context_implicit_timezone() {
    let items = Sequence::from_iter([
        date_time("2000-01-01T00:00:00"),
        date_time("2000-01-01T10:00:00Z"),
    ]);
    let search = date_time("2000-01-01T05:00:00Z");

    // Under the default (UTC) implicit timezone the floating item reads as 00:00:00Z, which does
    // not match.
    assert_eq!(
        oxixml_fn::sequences::index_of(&FnContext::new(), &items, &search, None),
        Ok(Vec::new())
    );

    // Under -05:00 the floating item reads as 05:00:00Z, the same instant as `search`.
    assert_eq!(
        oxixml_fn::sequences::index_of(&minus_five(), &items, &search, None),
        Ok(vec![1]),
        "fn:index-of must use the context's implicit timezone, not hardcode UTC"
    );
}

#[test]
fn distinct_values_reads_a_floating_date_time_in_the_context_implicit_timezone() {
    let items = Sequence::from_iter([
        date_time("2000-01-01T00:00:00"),
        date_time("2000-01-01T05:00:00Z"),
    ]);

    // Under the default (UTC) implicit timezone the two items denote different instants.
    let default_distinct = oxixml_fn::sequences::distinct_values(&FnContext::new(), &items, None)
        .expect("fn:distinct-values succeeds");
    assert_eq!(default_distinct.len(), 2);

    // Under -05:00 they denote the same instant, so only one survives.
    let shifted_distinct = oxixml_fn::sequences::distinct_values(&minus_five(), &items, None)
        .expect("fn:distinct-values succeeds");
    assert_eq!(
        shifted_distinct.len(),
        1,
        "fn:distinct-values must use the context's implicit timezone, not hardcode UTC"
    );
}

#[test]
fn max_reads_a_floating_date_time_in_the_context_implicit_timezone() {
    let items = Sequence::from_iter([
        date_time("2000-01-01T00:00:00"),
        date_time("2000-01-01T01:00:00Z"),
    ]);

    // Under the default (UTC) implicit timezone the floating item (00:00:00Z) is earlier.
    let default_max = oxixml_fn::sequences::max(&FnContext::new(), &items, None)
        .expect("fn:max succeeds")
        .expect("a non-empty sequence has a maximum");
    assert_eq!(default_max.string_value(), "2000-01-01T01:00:00Z");

    // Under -05:00 the floating item reads as 05:00:00Z, five hours *after* 01:00:00Z, so it
    // wins — the value itself is returned exactly as its own lexical form, with no timezone.
    let shifted_max = oxixml_fn::sequences::max(&minus_five(), &items, None)
        .expect("fn:max succeeds")
        .expect("a non-empty sequence has a maximum");
    assert_eq!(
        shifted_max.string_value(),
        "2000-01-01T00:00:00",
        "fn:max must use the context's implicit timezone, not hardcode UTC"
    );
}

#[test]
fn min_reads_a_floating_date_time_in_the_context_implicit_timezone() {
    let items = Sequence::from_iter([
        date_time("2000-01-01T00:00:00"),
        date_time("2000-01-01T01:00:00Z"),
    ]);

    // Under the default (UTC) implicit timezone the floating item (00:00:00Z) is the minimum.
    let default_min = oxixml_fn::sequences::min(&FnContext::new(), &items, None)
        .expect("fn:min succeeds")
        .expect("a non-empty sequence has a minimum");
    assert_eq!(default_min.string_value(), "2000-01-01T00:00:00");

    // Under -05:00 the floating item reads as 05:00:00Z, so the zoned item is now the minimum.
    let shifted_min = oxixml_fn::sequences::min(&minus_five(), &items, None)
        .expect("fn:min succeeds")
        .expect("a non-empty sequence has a minimum");
    assert_eq!(
        shifted_min.string_value(),
        "2000-01-01T01:00:00Z",
        "fn:min must use the context's implicit timezone, not hardcode UTC"
    );
}

#[test]
fn sort_reads_a_floating_date_time_key_in_the_context_implicit_timezone() {
    let items = Sequence::from_iter([
        date_time("2000-01-01T00:00:00"),
        date_time("2000-01-01T01:00:00Z"),
    ]);

    // Under the default (UTC) implicit timezone the floating item (00:00:00Z) sorts first.
    let default_sorted = oxixml_fn::higher_order::sort(&FnContext::new(), &items, None, None)
        .expect("fn:sort succeeds");
    assert_eq!(
        default_sorted
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        vec!["2000-01-01T00:00:00", "2000-01-01T01:00:00Z"]
    );

    // Under -05:00 the floating item reads as 05:00:00Z, so it now sorts last.
    let shifted_sorted =
        oxixml_fn::higher_order::sort(&minus_five(), &items, None, None).expect("fn:sort succeeds");
    assert_eq!(
        shifted_sorted
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        vec!["2000-01-01T01:00:00Z", "2000-01-01T00:00:00"],
        "fn:sort must use the context's implicit timezone, not hardcode UTC"
    );
}
